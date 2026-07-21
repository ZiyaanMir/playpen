#!/bin/bash
# Shared Eddie environment setup, sourced by every slurm_eddie/*.sh job script.
#
# Eddie runs Grid Engine (qsub/qstat/qdel), NOT Slurm — so none of the SBATCH
# idioms from ../slurm/ apply here. Differences that actually bite:
#   * directives are "#$ ..." not "#SBATCH ...", and must precede the first command
#   * there is no $PROJECTDIR (that is BriCS/Isambard); Eddie gives you
#     /exports/eddie/scratch/$USER, which is PURGED (files untouched for ~1 month
#     are deleted) and NOT backed up — copy checkpoints you care about to DataStore
#   * $JOB_ID replaces $SLURM_JOB_ID; $SGE_O_WORKDIR replaces $SLURM_SUBMIT_DIR
#   * home is a 10 GB quota, so every cache must be redirected onto scratch
#
# Not exported here on purpose: CUDA_VISIBLE_DEVICES. Grid Engine sets it for GPU
# isolation between users sharing a node — never override it (see the guide's
# gotchas: on Eddie it holds a GPU *UUID* string, not an integer index).

set -euo pipefail

# --- where things live -------------------------------------------------------
# REPO defaults to the directory you submitted from (-cwd puts us there already).
REPO="${REPO:-${SGE_O_WORKDIR:-$PWD}}"
SCRATCH="${SCRATCH:-/exports/eddie/scratch/$USER}"

: "${USER:?unset}"
[ -d "$REPO/playpen/marshal" ] || {
    echo "ERROR: \$REPO=$REPO is not a playpen checkout (no playpen/marshal/)." >&2
    exit 1
}

cd "$REPO"
mkdir -p logs

# --- modules -----------------------------------------------------------------
# The login shell is not sourced for batch jobs, so `module` must be enabled by hand.
. /etc/profile.d/modules.sh
module load cuda/12.4          # match whatever `module avail cuda` offers

# --- caches off the 10 GB home quota ----------------------------------------
export HF_HOME="$SCRATCH/home_cache/huggingface"
export PIP_CACHE_DIR="$SCRATCH/home_cache/pip"
export UV_CACHE_DIR="$SCRATCH/home_cache/uv"
export TRITON_CACHE_DIR="$SCRATCH/home_cache/triton"
export TMPDIR="${TMPDIR:-$SCRATCH/tmp}"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR"

# --- venv --------------------------------------------------------------------
# venvs are NOT relocatable: a copied .venv keeps the original absolute VIRTUAL_ENV
# in bin/activate and silently activates a *different* interpreter. Always verify.
source "$REPO/.venv/bin/activate"
ACTUAL_PY="$(command -v python)"
case "$ACTUAL_PY" in
    "$REPO/.venv/bin/python") ;;
    *) echo "ERROR: activated the wrong venv: $ACTUAL_PY (expected $REPO/.venv/bin/python)" >&2
       echo "       This venv was probably copied from another checkout — rebuild it." >&2
       exit 1 ;;
esac

# --- runtime knobs -----------------------------------------------------------
export TRL_EXPERIMENTAL_SILENCE=1
export TOKENIZERS_PARALLELISM=false
# Compute nodes may have no outbound internet; run offline once weights are cached.
# Flip to 1 only after the model has been pre-downloaded (guide §2.5).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

# Where run outputs go. Scratch, not home — checkpoints are GBs.
export MARSHAL_RUNS="${MARSHAL_RUNS:-$SCRATCH/marshal-runs}"
mkdir -p "$MARSHAL_RUNS"

echo "=================================================================="
echo "host      = $(hostname)"
echo "job       = ${JOB_ID:-interactive}  ${SGE_TASK_ID:+task=$SGE_TASK_ID}"
echo "start     = $(date --iso-8601=seconds)"
echo "repo      = $REPO"
echo "python    = $ACTUAL_PY"
echo "runs      = $MARSHAL_RUNS"
echo "gpu       = $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none visible')"
echo "=================================================================="

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

# An empty $SCRATCH would make every path below absolute from the filesystem root
# ("$SCRATCH/marshal-runs" -> "/marshal-runs"), which fails deep inside training as
# PermissionError [Errno 13] rather than here. Refuse that up front.
case "$SCRATCH" in
    /*) ;;
    *)  echo "ERROR: \$SCRATCH must be an absolute path, got '$SCRATCH'." >&2; exit 1 ;;
esac
[ -d "$SCRATCH" ] || {
    echo "ERROR: \$SCRATCH does not exist: $SCRATCH" >&2
    echo "       On Eddie this should be /exports/eddie/scratch/\$USER." >&2
    exit 1
}
[ -w "$SCRATCH" ] || { echo "ERROR: \$SCRATCH is not writable: $SCRATCH" >&2; exit 1; }

# --- modules -----------------------------------------------------------------
# The login shell is not sourced for batch jobs, so `module` must be enabled by hand.
. /etc/profile.d/modules.sh
module load cuda/12.9.1        # match whatever `module avail cuda` offers

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

# --- Eddie GPU UUID -> CUDA ordinal ------------------------------------------
# Grid Engine sets CUDA_VISIBLE_DEVICES to a GPU *UUID* (GPU-beccd30f-...) for its
# cgroup-based isolation. torch copes; vLLM 0.12.0 does not -- it runs
# int(device_ids[device_id]) (vllm/platforms/interface.py:211) and dies with
#   ValueError: invalid literal for int() with base 10: 'GPU-...'
# which surfaces as the misleading "Error in inspecting model architecture
# 'Qwen3ForCausalLM'" because vLLM introspects in a subprocess.
#
# We do NOT reassign the GPU here -- we translate the same allocated device into
# the integer ordinal vLLM demands, asking CUDA itself for the mapping so it is
# correct whether or not cgroups have already narrowed what is visible.
# _resolve_gpu.py probes candidate values in subprocesses and returns only one it has
# verified exposes the allocated GPU, so a wrong guess cannot silently hide it. If it
# cannot find one we WARN and leave the UUID in place rather than aborting: torch works
# with the UUID, so the job still reaches the (known, documented) vLLM error instead of
# dying earlier and more confusingly.
if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* || "${CUDA_VISIBLE_DEVICES:-}" == MIG-* ]]; then
    _uuid="$CUDA_VISIBLE_DEVICES"
    if _ordinal="$(python "$REPO/slurm_eddie/_resolve_gpu.py" "$_uuid")"; then
        if [ -z "$_ordinal" ]; then
            unset CUDA_VISIBLE_DEVICES
            echo "[gpu] unset CUDA_VISIBLE_DEVICES (verified: exposes only $_uuid)"
        else
            export CUDA_VISIBLE_DEVICES="$_ordinal"
            echo "[gpu] remapped CUDA_VISIBLE_DEVICES: $_uuid -> $_ordinal (verified same GPU)"
        fi
    else
        echo "WARNING: could not map $_uuid to a CUDA ordinal; leaving it as-is." >&2
        echo "         torch will work, vLLM will likely fail — see guide §6." >&2
    fi
    unset _uuid _ordinal
fi

# --- runtime knobs -----------------------------------------------------------
export TRL_EXPERIMENTAL_SILENCE=1
export TOKENIZERS_PARALLELISM=false
# Compute nodes may have no outbound internet; run offline once weights are cached.
# Flip to 1 only after the model has been pre-downloaded (guide §2.5).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

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

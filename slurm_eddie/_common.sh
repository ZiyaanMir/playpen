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

# THIS FILE MUST BE SOURCED, NOT EXECUTED:
#     source slurm_eddie/_common.sh      (or:  . slurm_eddie/_common.sh)
# `bash slurm_eddie/_common.sh` runs it in a subshell, so every export dies with
# that subshell and your session is left unchanged. We detect and refuse that.
(return 0 2>/dev/null) && _SOURCED=1 || _SOURCED=0
_RET=exit; [ "$_SOURCED" = 1 ] && _RET=return

if [ "$_SOURCED" = 0 ]; then
    echo "ERROR: source this file, don't execute it:" >&2
    echo "         source ${0}" >&2
    echo "       Executing it changes only a subshell, which then exits." >&2
    exit 1
fi

# `set -e` is right for a batch job but hostile in an interactive shell: one
# failing command (a grep that matches nothing) would kill the login session.
# Job scripts are non-interactive, so they still get the strict behavior.
case $- in *i*) _INTERACTIVE=1 ;; *) _INTERACTIVE=0 ;; esac
[ "$_INTERACTIVE" = 1 ] || set -euo pipefail

# --- where things live -------------------------------------------------------
# Anchor on THIS FILE's location, not the working directory and not
# $SGE_O_WORKDIR. Both of those lie: $SGE_O_WORKDIR is wherever `qlogin`/`qsub`
# was invoked (e.g. the logs/ dir you were tailing from), and $PWD is wherever
# you have since cd'd. BASH_SOURCE is correct in every case, including a
# Grid-Engine-spooled batch job, because the job script sources us by real path.
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(dirname "$_HERE")}"
SCRATCH="${SCRATCH:-/exports/eddie/scratch/$USER}"

: "${USER:?unset}"
[ -d "$REPO/playpen/marshal" ] || {
    echo "ERROR: \$REPO=$REPO is not a playpen checkout (no playpen/marshal/)." >&2
    $_RET 1
}

cd "$REPO"
mkdir -p logs

# An empty $SCRATCH would make every path below absolute from the filesystem root
# ("$SCRATCH/marshal-runs" -> "/marshal-runs"), which fails deep inside training as
# PermissionError [Errno 13] rather than here. Refuse that up front.
case "$SCRATCH" in
    /*) ;;
    *)  echo "ERROR: \$SCRATCH must be an absolute path, got '$SCRATCH'." >&2; $_RET 1 ;;
esac
[ -d "$SCRATCH" ] || {
    echo "ERROR: \$SCRATCH does not exist: $SCRATCH" >&2
    echo "       On Eddie this should be /exports/eddie/scratch/\$USER." >&2
    $_RET 1
}
[ -w "$SCRATCH" ] || { echo "ERROR: \$SCRATCH is not writable: $SCRATCH" >&2; $_RET 1; }

# --- modules -----------------------------------------------------------------
# The login shell is not sourced for batch jobs, so `module` must be enabled by hand.
# Guarded so this file can also be sourced somewhere without environment-modules
# (e.g. a laptop) without dying on a missing /etc/profile.d/modules.sh.
[ -r /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh
if command -v module >/dev/null 2>&1; then
    module load cuda/12.9.1    # match whatever `module avail cuda` offers
else
    echo "[warn] no 'module' command — skipping 'module load cuda'." >&2
fi

# --- caches off the 10 GB home quota ----------------------------------------
export HF_HOME="$SCRATCH/home_cache/huggingface"
export PIP_CACHE_DIR="$SCRATCH/home_cache/pip"
export UV_CACHE_DIR="$SCRATCH/home_cache/uv"
export TMPDIR="${TMPDIR:-$SCRATCH/tmp}"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$TMPDIR"

# The catch-all, and the reason this list kept growing one package at a time. Almost
# everything that caches to $HOME does it under ~/.cache and honours XDG_CACHE_HOME:
# flashinfer, tvm-ffi, and Triton's NEWER default (~/.cache/triton, which is a second
# location from the ~/.triton the explicit TRITON_CACHE_DIR below covers). Redirecting
# the parent is cheaper and more complete than chasing each one.
export XDG_CACHE_HOME="$SCRATCH/home_cache/xdg"
# wandb's artifact cache. Set NOWHERE before 2026-08-18 and 1.3 GB in $HOME by then --
# every training run writes to it, so it grew unbounded.
export WANDB_CACHE_DIR="$SCRATCH/home_cache/wandb"
# matplotlib builds a font cache on first import; clean_up pulls matplotlib in.
export MPLCONFIGDIR="$SCRATCH/home_cache/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$WANDB_CACHE_DIR" "$MPLCONFIGDIR"

# The one remaining thing that writes to $HOME on every run. vLLM's usage reporter
# appends to ~/.config/vllm/usage_stats.json from a background thread; with home at
# quota that thread dies with
#   OSError: [Errno 122] Disk quota exceeded   (vllm/usage/usage_lib.py:289)
# in the first 20 lines of every train*.err. It is NOT fatal -- it is a daemon thread
# and training continues -- but it is noise on top of a real error, and it is a
# telemetry upload nobody asked for. Off costs nothing.
export VLLM_NO_USAGE_STATS=1

# --- Isambard compatibility shim ---------------------------------------------
# The sibling ../slurm/*.sh scripts are for Isambard and end with
#     --output-dir "$PROJECTDIR/$USER/marshal-runs/<game>"
# $PROJECTDIR comes from BriCS's `module load brics/userenv` and does NOT exist on
# Eddie, so copy-pasting one of those command tails here silently produces
# "/<user>/marshal-runs/..." — an absolute path at the filesystem root — and dies
# ~90 s later inside Trainer.__init__ with:
#     PermissionError: [Errno 13] Permission denied: '/s2874947'
# Defining it as the PARENT of $SCRATCH makes "$PROJECTDIR/$USER/..." resolve to
# exactly "$SCRATCH/...", so an Isambard-style command lands in the right place on
# Eddie instead of crashing. Prefer "$MARSHAL_RUNS/<game>" in new commands.
export PROJECTDIR="${PROJECTDIR:-$(dirname "$SCRATCH")}"

# --- compile caches: PER JOB, never shared -----------------------------------
# vLLM's torch.compile cache defaults to ~/.cache/vllm (vllm/envs.py:33) and is
# SHARED by every concurrently running job. The cached artifacts embed absolute
# paths from whichever job compiled first, including that job's $TMPDIR
# (/local/<jobid>.<task>.<queue> on Eddie). A second job then loads the cache and
# tries to write autotune results back into the *first* job's private tmpdir:
#
#   PermissionError: [Errno 13] Permission denied: '/local/56988859.1.gpu'
#      ...raised in job 56988860
#
# A jobid in the path that isn't yours is the signature. Pinning all three compile
# caches under $TMPDIR makes them per-job and self-cleaning; the cost is a few
# minutes of recompilation per job, which is noise next to a 24 h run. It also
# keeps ~/.cache/vllm from silently eating the 10 GB home quota.
export VLLM_CACHE_ROOT="$TMPDIR/vllm"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
export TRITON_CACHE_DIR="$TMPDIR/triton"
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# --- venv --------------------------------------------------------------------
# venvs are NOT relocatable: a copied .venv keeps the original absolute VIRTUAL_ENV
# in bin/activate and silently activates a *different* interpreter. Always verify.
source "$REPO/.venv/bin/activate"
ACTUAL_PY="$(command -v python)"
case "$ACTUAL_PY" in
    "$REPO/.venv/bin/python") ;;
    *) echo "ERROR: activated the wrong venv: $ACTUAL_PY (expected $REPO/.venv/bin/python)" >&2
       echo "       This venv was probably copied from another checkout — rebuild it." >&2
       $_RET 1 ;;
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

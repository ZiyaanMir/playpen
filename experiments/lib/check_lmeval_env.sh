#!/bin/bash
# Is the lm-eval environment actually usable? Run this BEFORE submitting eval jobs.
#
#   experiments/lib/check_lmeval_env.sh                      # this cluster's default
#   LMEVAL_CONDA_ENV=lmeval2 experiments/lib/check_lmeval_env.sh
#   experiments/lib/check_lmeval_env.sh --cluster eddie
#
# WHY THIS EXISTS. eval.sh's first real statement is
#     python -c "import lm_eval, peft; print(...)"
# under `set -e`. An environment that has lost lm_eval therefore kills the job about
# five seconds in -- AFTER it has queued for hours and been handed a GPU. That is
# exactly what happened to every Eddie run from 2026-08-10 onward: each eval shard
# died with
#     ModuleNotFoundError: No module named 'lm_eval'
# leaving no eval/ output and no RESULTS.md, while the gameplay half of the same
# experiments finished normally (playpen_eval.sh uses the TRAINING venv, not this
# environment, so it was never affected).
#
# The failure is quiet in the way that matters: the summary job still runs, prints
# "found no results*.json under .../eval -- nothing to summarize", and exits 0. So
# the experiment looks finished. Nothing tells you the reasoning half is missing
# except the absence of RESULTS.md.
#
# One login-node check costs nothing and turns 130 dead GPU jobs into one error
# message. Exits 0 when the environment can import lm_eval and peft; non-zero with
# the repair command when it cannot.
#
# Deliberately NOT `set -e`: the whole job of this script is to survive the failures
# it is looking for and report them.

set -uo pipefail

CLUSTER=""
while [ $# -gt 0 ]; do
    case "$1" in
        --cluster) CLUSTER="${2:-}"; shift ;;
        -h|--help)
            sed -n '2,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

# Same detection as queue_playpen_eval_backfill.sh: the scheduler that is present.
if [ -z "$CLUSTER" ]; then
    if command -v qsub >/dev/null 2>&1 && ! command -v sbatch >/dev/null 2>&1; then
        CLUSTER=eddie
    elif command -v sbatch >/dev/null 2>&1 && ! command -v qsub >/dev/null 2>&1; then
        CLUSTER=isambard
    else
        echo "ERROR: could not tell which cluster this is (qsub/sbatch both present" >&2
        echo "       or both absent). Pass --cluster eddie|isambard." >&2
        exit 2
    fi
fi

echo "[env-check] cluster = $CLUSTER"

case "$CLUSTER" in
eddie)
    # Replicated from eddie/eval.sh, including its venv-before-conda order. A check
    # that activates the environment some other way can pass while the job still fails.
    SCRATCH="${SCRATCH:-/exports/eddie/scratch/$USER}"
    [ -r /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh
    if ! command -v module >/dev/null 2>&1 && ! type module >/dev/null 2>&1; then
        echo "[env-check] FAIL: no 'module' command -- this is not an Eddie node." >&2
        exit 1
    fi

    LMEVAL_VENV_DEFAULT="$SCRATCH/lmeval-venv"
    if [ -n "${VENV_LMEVAL:-}" ] || [ -f "$LMEVAL_VENV_DEFAULT/bin/activate" ]; then
        VENV_LMEVAL="${VENV_LMEVAL:-$LMEVAL_VENV_DEFAULT}"
        USING=venv
        echo "[env-check] venv = $VENV_LMEVAL  (override with VENV_LMEVAL)"
        if [ ! -f "$VENV_LMEVAL/bin/activate" ]; then
            echo "[env-check] FAIL: no venv at $VENV_LMEVAL" >&2
            exit 1
        fi
        # shellcheck disable=SC1091
        . "$VENV_LMEVAL/bin/activate" || { echo "[env-check] FAIL: could not activate" >&2; exit 1; }
    else
        USING=conda
        ENV_NAME="${LMEVAL_CONDA_ENV:-lmeval}"
        echo "[env-check] conda env = $ENV_NAME  (override with LMEVAL_CONDA_ENV)"
        echo "[env-check] NOTE: no venv at $LMEVAL_VENV_DEFAULT, falling back to conda."
        echo "[env-check]       The conda env lives in \$HOME (10 GB quota) -- see the"
        echo "[env-check]       repair recipe below if anything here fails."
        if ! module load anaconda/2024.02 2>&1; then
            echo "[env-check] FAIL: module load anaconda/2024.02" >&2
            echo "            Eddie may have retired that version. Check 'module avail anaconda'" >&2
            echo "            and update the module line in experiments/eddie/eval.sh." >&2
            exit 1
        fi
        if ! conda activate "$ENV_NAME" 2>&1; then
            echo "[env-check] FAIL: conda activate $ENV_NAME" >&2
            echo "            The env is gone or was never created. 'conda env list' to see" >&2
            echo "            what is left." >&2
            exit 1
        fi
    fi
    ;;
isambard)
    # Replicated from isambard/eval.sh.
    USING=venv
    VENV_LMEVAL="${VENV_LMEVAL:-${PROJECTDIR:-}/$USER/evaluation/eval}"
    echo "[env-check] venv = $VENV_LMEVAL  (override with VENV_LMEVAL)"
    if [ ! -f "$VENV_LMEVAL/bin/activate" ]; then
        echo "[env-check] FAIL: no venv at $VENV_LMEVAL" >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    . "$VENV_LMEVAL/bin/activate" || { echo "[env-check] FAIL: could not activate" >&2; exit 1; }
    ;;
*)
    echo "ERROR: unknown cluster '$CLUSTER' (expected eddie or isambard)." >&2
    exit 2 ;;
esac

# WHICH python answered matters as much as whether the import worked. `conda activate`
# landing in base instead of the named env, or a venv whose activate script hardcodes
# a path that has moved, both look like a missing module from the traceback alone.
#
# The prefix is read from the variable belonging to the route actually taken. Reading
# CONDA_PREFIX first would be wrong on the venv path: an inherited conda prefix (from
# `module load anaconda`, or a login shell with a base env active) would be reported
# while $python came from the venv -- printing a mismatch that is not real, and hiding
# the one case this line exists to catch.
echo "[env-check] python  = $(command -v python || echo '<none>')"
case "$USING" in
    venv)  echo "[env-check] prefix  = ${VIRTUAL_ENV:-<none>}  (venv)" ;;
    conda) echo "[env-check] prefix  = ${CONDA_PREFIX:-<none>}  (conda)" ;;
esac

if ! python -c "import lm_eval, peft; print('[env-check] lm_eval', lm_eval.__version__, '| peft', peft.__version__)"; then
    echo "" >&2
    echo "[env-check] FAIL: the eval environment cannot import lm_eval and/or peft." >&2
    echo "            Every eval job submitted against it will die in seconds, on a GPU," >&2
    echo "            with exactly the traceback above. Repair it first:" >&2
    echo "" >&2
    if [ "$CLUSTER" = eddie ]; then
        echo "            BUILD IT ON SCRATCH, NOT IN \$HOME. Eddie home is a 10 GB quota" >&2
        echo "            (RUN_ON_EDDIE_playpen_marshal.md section 2.1: \"nothing heavy --" >&2
        echo "            it will fill and break jobs\"), and a pip install that runs out of" >&2
        echo "            quota mid-write leaves the env importable but INCOMPLETE, which is" >&2
        echo "            how this failure happened in the first place. Note that the CACHE" >&2
        echo "            is what usually blows the quota, not the packages -- slurm_eddie/" >&2
        echo "            _common.sh puts it on scratch for training; do the same here." >&2
        echo "" >&2
        echo "            In a PLAIN interactive session (not the login node, not a GPU one):" >&2
        echo "" >&2
        echo "              qlogin -l h_rt=02:00:00 -pe interactivemem 4 -l h_rss=16G" >&2
        echo "              source /exports/applications/support/set_qlogin_environment.sh" >&2
        echo "              . /etc/profile.d/modules.sh && module load cuda" >&2
        echo "" >&2
        echo "              export SCRATCH=/exports/eddie/scratch/\$USER" >&2
        echo "              export UV_CACHE_DIR=\$SCRATCH/home_cache/uv" >&2
        echo "              export PIP_CACHE_DIR=\$SCRATCH/home_cache/pip" >&2
        echo "              export HF_HOME=\$SCRATCH/home_cache/huggingface" >&2
        echo "              mkdir -p \"\$UV_CACHE_DIR\" \"\$PIP_CACHE_DIR\" \"\$HF_HOME\"" >&2
        echo "" >&2
        echo "              uv venv --python 3.10 \$SCRATCH/lmeval-venv" >&2
        echo "              source \$SCRATCH/lmeval-venv/bin/activate" >&2
        echo "              which python     # MUST be \$SCRATCH/lmeval-venv/bin/python" >&2
        echo "              uv pip install lm-eval peft" >&2
        echo "" >&2
        echo "            eval.sh picks up \$SCRATCH/lmeval-venv automatically -- nothing to" >&2
        echo "            configure. Then re-check, and confirm the tasks are registered:" >&2
        echo "              experiments/lib/check_lmeval_env.sh" >&2
        echo "              lm-eval --tasks list 2>/dev/null | grep -E 'logiglue|logicbench'" >&2
        echo "" >&2
        echo "            If home is ALREADY full, that must be cleared first:" >&2
        echo "              du -sh \$HOME/.cache/pip \$HOME/.conda \$HOME/.cache/huggingface 2>/dev/null" >&2
        echo "              rm -rf \$HOME/.cache/pip        # a cache; safe to delete" >&2
        echo "" >&2
        echo "            THE TASKS ARE THE PART A PLAIN REINSTALL DOES NOT RESTORE." >&2
        echo "            logiglue/logicbench were baked into this env's lm_eval/tasks/" >&2
        echo "            (that is why eval.sh passes no --include_path). A fresh pip" >&2
        echo "            install of lm-eval ships neither, and lm-eval fails per task" >&2
        echo "            rather than up front -- so an env that imports fine can still" >&2
        echo "            produce an empty RESULTS.md. Rebuild per" >&2
        echo "            LMEVAL_CHECKPOINTS_EDDIE.md section 2, or point EVAL_EXTRA at" >&2
        echo "            the task yamls with --include_path." >&2
    else
        echo "              source $VENV_LMEVAL/bin/activate" >&2
        echo "              uv pip install lm-eval peft" >&2
    fi
    exit 1
fi

echo "[env-check] OK -- safe to submit eval jobs."
exit 0

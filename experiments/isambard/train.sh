#!/bin/bash
# Training half of an Isambard experiment. Submitted by run_experiment.sh, which
# creates $EXP_DIR and writes the env file this reads -- do not sbatch it directly.
#
#SBATCH --job-name=marshal_train
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=05:00:00
#SBATCH --no-requeue
#
# --time stays under the 24 h cap with margin, and Slurm reserves credits against
# it. --no-requeue because a requeue restarts training from step 0: 5 h of GPU
# silently redone. Override with TRAIN_SBATCH_OPTS='--time=08:00:00' for short runs.

set -euo pipefail

: "${EXP_ENV_FILE:?not set -- submit via experiments/isambard/run_experiment.sh}"
# shellcheck disable=SC1090
source "$EXP_ENV_FILE"

cd "$REPO"
source "$REPO/experiments/lib/experiment.sh"
exp_layout
exp_banner train

# venvs are NOT relocatable -- a venv copied from another checkout silently
# activates a different interpreter. exp_activate_venv verifies by file identity, so
# it accepts Isambard's /projects -> /lus/... mount aliasing (which a string
# comparison wrongly rejected) while still catching a genuine copy.
exp_activate_venv "$REPO/.venv" "training venv" || exit 1

export HF_HOME="${HF_HOME:-$PROJECTDIR/hf}"
export TRL_EXPERIMENTAL_SILENCE=1
export TOKENIZERS_PARALLELISM=false
# Pin torch.distributed to an OS-assigned free port, so vLLM's init cannot collide
# with a co-located job or a stale process from a killed run. See
# exp_export_master_port in experiments/lib/experiment.sh for why MASTER_PORT=0 is
# NOT sufficient here. Runs after the venv activation above, which the probe needs.
exp_export_master_port || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME"

{
    echo ""
    echo "-- training job (filled in at run time) -----------------------------"
    echo "  job_id                       ${SLURM_JOB_ID:-interactive}"
    echo "  host                         $(hostname)"
    echo "  started                      $(date --iso-8601=seconds)"
    echo "  gpu                          $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo '?')"
} >> "$EXP_DIR/manifest.txt"

ARGS=(
    --model "$MODEL"
    --game "$GAME"
    --marshal-config "$MARSHAL_CONFIG"
    --num-generations "$NUM_GENERATIONS"
    --per-device-batch-size "$PER_DEVICE_BATCH"
    --grad-accum "$GRAD_ACCUM"
    --max-steps "$MAX_STEPS"
    --save-steps "$SAVE_STEPS"
    --learning-rate "$LEARNING_RATE"
    --kl-beta "$KL_BETA"
    --max-completion-length "$MAX_COMPLETION_LENGTH"
    --max-turns "$MAX_TURNS"
    --vllm-gpu-memory-utilization "$VLLM_UTIL"
    --vllm-max-model-len "$VLLM_MAX_MODEL_LEN"
    --output-dir "$TRAIN_BASE"
    # $EXP_DIR is fresh per experiment, so the extra timestamp layer
    # train_selfplay.py adds by default would only deepen the path. Without it
    # checkpoints land at $EXP_DIR/train/checkpoint-<step>, one level under the
    # manifest and the results they belong to.
    --no-run-subdir
)
[ "${GRAD_CKPT:-1}" = "1" ] && ARGS+=( --gradient-checkpointing )
[ -n "${LP_MAX_LEN:-}" ] && ARGS+=( --length-penalty-max-len "$LP_MAX_LEN" )
[ -n "${LP_COEF:-}"    ] && ARGS+=( --length-penalty-coef "$LP_COEF" )
# Unquoted on purpose: a pre-split flag string (e.g. "--no-length-penalty").
[ -n "${EXTRA_TRAIN_ARGS:-}" ] && ARGS+=( ${EXTRA_TRAIN_ARGS} )

echo "[train] python -m examples.marshal.train_selfplay ${ARGS[*]}"
python -m examples.marshal.train_selfplay "${ARGS[@]}"

RUN_DIR="$(exp_find_run_dir || true)"
if [ -n "$RUN_DIR" ]; then
    echo "[train] run dir   : $RUN_DIR"
    echo "[train] checkpoints:"
    exp_list_checkpoints "$RUN_DIR" | sed 's/^/           /'
else
    echo "[train] WARNING: no run directory under $TRAIN_BASE" >&2
fi

echo "[train] end = $(date --iso-8601=seconds)"

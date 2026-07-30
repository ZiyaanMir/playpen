#!/bin/bash
# Training half of an Eddie experiment. Submitted by run_experiment.sh, which
# creates $EXP_DIR and writes the env file this reads -- do not qsub it directly.
#
# --- Grid Engine options (overridable via TRAIN_QSUB_OPTS in run_experiment.sh) ---
#$ -N marshal_train
#$ -cwd
#$ -q gpu
#$ -l gpu=1
#$ -l h200=true
#$ -l h_rt=48:00:00
#$ -pe sharedmem 12
#$ -l h_rss=12G
#
# GPU choice: H200 (141 GB). For an A100 instead, submit with
#   TRAIN_QSUB_OPTS='-l a100=true' experiments/eddie/run_experiment.sh <game>
# and drop PER_DEVICE_BATCH -- see experiments/presets/*.env for why the fp32
# logits buffer, not the model weights, is what sets the batch size here.

set -euo pipefail

: "${EXP_ENV_FILE:?not set -- submit via experiments/eddie/run_experiment.sh}"
# shellcheck disable=SC1090
source "$EXP_ENV_FILE"

# Announce the experiment BEFORE sourcing _common.sh (which prints its own,
# experiment-unaware banner) so the log opens with what run this is.
echo "### experiment: ${EXP_ID} (train) ###"

cd "$REPO"
# The training environment: venv, module load, caches off the home quota, and the
# Eddie GPU-UUID -> CUDA-ordinal remap that vLLM needs. Reused as-is so this stays
# in lockstep with the existing slurm_eddie/ jobs.
source "$REPO/slurm_eddie/_common.sh"

source "$REPO/experiments/lib/experiment.sh"
exp_layout
exp_banner train

# Pin torch.distributed to an OS-assigned free port, so vLLM's init cannot collide
# with another job on this shared node (Eddie gives you one H200 of eight). See
# exp_export_master_port in experiments/lib/experiment.sh for why MASTER_PORT=0 is
# NOT sufficient here. Must come after `source _common.sh`, which activates the venv
# this probe needs.
exp_export_master_port || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Record the runtime facts the submitter could not know. Appended rather than
# rewritten so the submit-time manifest stays intact.
{
    echo ""
    echo "-- training job (filled in at run time) -----------------------------"
    echo "  job_id                       ${JOB_ID:-interactive}"
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
# Empty => don't pass the flag, so the YAML value stands (see presets/*.env).
[ -n "${LP_MAX_LEN:-}" ] && ARGS+=( --length-penalty-max-len "$LP_MAX_LEN" )
[ -n "${LP_COEF:-}"    ] && ARGS+=( --length-penalty-coef "$LP_COEF" )
# Dense per-turn rewards. Same convention: empty => leave the YAML alone. TR_ENABLE
# is a tri-state (1 on / 0 off / empty = whatever the YAML says) because "on" and
# "leave alone" must stay distinguishable, exactly as --turn-rewards/--no-turn-rewards
# are in train_selfplay.py.
case "${TR_ENABLE:-}" in
    1) ARGS+=( --turn-rewards ) ;;
    0) ARGS+=( --no-turn-rewards ) ;;
esac
[ -n "${TR_SOURCE:-}"     ] && ARGS+=( --turn-reward-source "$TR_SOURCE" )
[ -n "${TR_SCALE:-}"      ] && ARGS+=( --turn-reward-scale "$TR_SCALE" )
[ -n "${TR_BUDGET:-}"     ] && ARGS+=( --turn-reward-budget "$TR_BUDGET" )
[ -n "${TR_COMPONENTS:-}" ] && ARGS+=( --turn-reward-components "$TR_COMPONENTS" )
# W&B: named after this experiment, data written inside $EXP_DIR. WB_MODE=auto
# records offline when the compute node has no credential/network -- upload it later
# with experiments/lib/wandb_sync.sh from a login node.
exp_wandb_args
ARGS+=( "${WANDB_ARGS[@]}" )
# Unquoted on purpose: this is a pre-split flag string (e.g. "--no-length-penalty").
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

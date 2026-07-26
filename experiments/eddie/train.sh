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

# Let TRL/vLLM pick a FREE torch.distributed port instead of an already-bound one.
# The Eddie environment injects a fixed MASTER_PORT (observed: 23456, not torch's
# 29500 default -- it comes from the module env loaded by _common.sh, not this repo),
# which TRL's ensure_master_addr_port respects. A stale/zombie process from a killed
# run, or a co-located job, then makes vLLM init die with EADDRINUSE on that port.
# "0" forces a free-port lookup. MUST come AFTER `source _common.sh` so it overrides
# whatever the module load set; MASTER_ADDR pinned to loopback clears any inherited value.
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=0
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

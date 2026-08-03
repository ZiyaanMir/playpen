#!/bin/bash
# Training half of an Isambard experiment. Submitted by run_experiment.sh, which
# creates $EXP_DIR and writes the env file this reads -- do not sbatch it directly.
#
#SBATCH --job-name=marshal_train
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#
# --time is Isambard's HARD MAXIMUM (workq_qos caps every job at 24 h; asking for
# more is rejected with PartitionTimeLimit). Set to the cap because
# train_selfplay.py has NO --resume-from-checkpoint: a run killed at the walltime
# cannot be continued, only redone, so the only lever against a long game is to ask
# for every hour available. With MAX_STEPS=1000 in the presets, the long-episode
# games (adventuregame, imagegame, clean_up) will use most of it.
#
# TWO CONSEQUENCES OF ASKING FOR THE CAP.
#   * Slurm reserves credits against --time, not against what the job uses, so a
#     short run books 24 h of GH200 up front. Trim it per submission for anything
#     you know is short:  TRAIN_SBATCH_OPTS='--time=06:00:00'
#   * A 24 h request backfills less readily than a 7 h one, so the job may sit in the
#     queue longer. `--time-min` (see notes/ISAMBARD_GUIDE.md) is the escape hatch if
#     that becomes the bottleneck.
#
# --no-requeue because a requeue restarts training from step 0: hours of GPU
# silently redone.

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

# Must run BEFORE anything imports torch: the inherited TMPDIR (/local/user/<uid>)
# is a login-node path that does not exist on the compute node, and torch's inductor
# cache + tempfile use both blow up at import time. See exp_setup_tmpdir.
exp_setup_tmpdir || exit 1

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
# marshal_exact's torch.unique distinct-value pooling. Tri-state like TR_ENABLE
# below; only has an effect under fidelity_mode: marshal_exact.
case "${UNIQUE_POOL:-}" in
    1) ARGS+=( --marshal-exact-unique-pooling ) ;;
    0) ARGS+=( --no-marshal-exact-unique-pooling ) ;;
esac
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
# records offline when the compute node has no credential/network, which is the
# normal case here -- experiments/lib/wandb_sync.sh uploads it from the login node.
exp_wandb_args
ARGS+=( "${WANDB_ARGS[@]}" )
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

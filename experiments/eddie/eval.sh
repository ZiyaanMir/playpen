#!/bin/bash
# Evaluation half of an Eddie experiment: lm-eval over every checkpoint the
# training job wrote, plus the untrained base model as a baseline. Submitted by
# run_experiment.sh with -hold_jid on the training job.
#
# Results land inside the experiment directory, NOT in a shared Results/ tree:
#     $EXP_DIR/eval/base/            untrained model
#     $EXP_DIR/eval/checkpoint-50/   ... one dir per adapter
#     $EXP_DIR/RESULTS.md            the table across all of them
#
# Re-runnable by hand once training is done:
#     source <EXP_DIR>/experiment.env && experiments/eddie/eval.sh
#
# --- Grid Engine options (overridable via EVAL_QSUB_OPTS) --------------------
#$ -N marshal_eval
#$ -cwd
#$ -q gpu
#$ -l gpu=1
#$ -l h_rt=08:00:00
#$ -pe sharedmem 4
#$ -l h_rss=12G

set -euo pipefail

: "${EXP_ENV_FILE:?not set -- submit via experiments/eddie/run_experiment.sh}"
# shellcheck disable=SC1090
source "$EXP_ENV_FILE"

cd "$REPO"
source "$REPO/experiments/lib/experiment.sh"
exp_layout
exp_banner eval

# --- the lm-eval environment (NOT the training venv) -------------------------
# logiglue/logicbench are baked into this conda env's lm_eval/tasks, so no
# --include_path is needed. See notes/LMEVAL_CHECKPOINTS_EDDIE.md section 2.
[ -r /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh
module load cuda
module load anaconda/2024.02
# shellcheck disable=SC1091
conda activate "${LMEVAL_CONDA_ENV:-lmeval}"

SCRATCH="${SCRATCH:-/exports/eddie/scratch/$USER}"
export HF_HOME="$SCRATCH/home_cache/huggingface"
export TMPDIR="${TMPDIR:-$SCRATCH/tmp}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
mkdir -p "$HF_HOME" "$TMPDIR" "$EVAL_DIR"

python -c "import lm_eval, peft; print('[eval] lm_eval', lm_eval.__version__, '| peft', peft.__version__)"

# On Eddie CUDA_VISIBLE_DEVICES holds a GPU UUID. lm-eval's `hf` backend is torch,
# which handles UUIDs fine, so --device cuda:0 is correct and no remap is needed
# (that is only required for the vLLM backend used during training).

RUN_DIR="$(exp_find_run_dir || true)"
if [ -z "$RUN_DIR" ]; then
    echo "[eval] no training run directory under $TRAIN_BASE." >&2
    echo "[eval] training produced nothing to evaluate -- check $LOG_DIR/train.*.err" >&2
    exit 0
fi
echo "[eval] run dir = $RUN_DIR"

# This job scores only ITS shard of the checkpoints (see exp_shard_filter). The
# other shards are separate jobs running at the same time; each writes to its own
# eval/<checkpoint>/ directory, so they never touch the same file.
mapfile -t ALL_CHECKPOINTS < <(exp_list_checkpoints "$RUN_DIR")
# Re-read from the source rather than piping the array back through printf: with an
# empty array, `printf '%s\n' "${arr[@]}"` emits one BLANK line, which would come
# back as a checkpoint path of "".
mapfile -t CHECKPOINTS < <(exp_list_checkpoints "$RUN_DIR" | exp_shard_filter)
if [ -n "${EVAL_SHARD:-}" ]; then
    echo "[eval] $(exp_shard_label): ${#CHECKPOINTS[@]} of ${#ALL_CHECKPOINTS[@]} checkpoint(s)"
else
    echo "[eval] ${#CHECKPOINTS[@]} checkpoint(s) to score"
fi
if [ "${#CHECKPOINTS[@]}" -eq 0 ]; then
    # A shard past the end of a short run (training was killed before it wrote as
    # many checkpoints as the schedule predicted). Nothing to do, and not an error.
    echo "[eval] this shard has no checkpoints -- nothing to do."
    exit 0
fi

# Only shard 1 scores the untrained baseline: it does not depend on the checkpoint,
# so every other shard would be re-running (or re-copying) the identical row.
if ! exp_owns_base_row; then
    echo "[eval] base row skipped -- shard 1 owns it."
    EVAL_BASE=0
fi

# Shared lm-eval flags. --log_samples keeps the per-example outputs, which is the
# only way to tell a real score from a formatting failure after the fact.
COMMON=( --tasks "$EVAL_TASKS" --device cuda:0 --batch_size "$EVAL_BATCH" --log_samples )
[ -n "${EVAL_LIMIT:-}" ] && COMMON+=( --limit "$EVAL_LIMIT" )
# Unquoted on purpose: a pre-split flag string.
[ -n "${EVAL_EXTRA:-}" ] && COMMON+=( ${EVAL_EXTRA} )

# NOTE: --apply_chat_template is deliberately NOT set. These are multiple-choice
# loglikelihood tasks and the base-model baseline below runs without it too; adding
# it to one side only would make the comparison meaningless. To use it, put it in
# EVAL_EXTRA so it applies to every row in the table.

FAILED=0

# --- baseline: the untrained model -------------------------------------------
# Without this row a score is uninterpretable -- there is nothing to say whether
# training helped. But the base result depends only on (model, tasks, limit, extra),
# never on the checkpoint, so it is the SAME for every experiment on this model.
# exp_eval_base runs it once, caches it under $BASE_EVAL_CACHE, and copies the stored
# result into later experiments instead of re-running on the GPU.
if [ "${EVAL_BASE:-1}" = "1" ]; then
    echo "=== [eval] base model ==="
    CACHE_ROOT="${BASE_EVAL_CACHE:-$MARSHAL_RUNS/_base_eval_cache}"
    exp_eval_base "$MODEL" "$EVAL_DIR/base" "$CACHE_ROOT" -- "${COMMON[@]}" \
        || { echo "[eval] FAILED: base model" >&2; FAILED=$((FAILED + 1)); }
fi

# --- each checkpoint ----------------------------------------------------------
for CK in "${CHECKPOINTS[@]}"; do
    NAME="$(basename "$CK")"
    echo "=== [eval] $NAME ==="
    # Read the base from the adapter itself: a mismatched base applies the LoRA
    # maths to the wrong weights and produces silently wrong scores, with no error.
    CK_BASE="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['base_model_name_or_path'])" \
               "$CK/adapter_config.json")"
    if [ "$CK_BASE" != "$MODEL" ]; then
        echo "[eval] NOTE: adapter names base '$CK_BASE' (experiment MODEL is '$MODEL');" \
             "using the adapter's." >&2
    fi
    mkdir -p "$EVAL_DIR/$NAME"
    lm-eval --model hf \
        --model_args "pretrained=${CK_BASE},peft=${CK},dtype=bfloat16" \
        "${COMMON[@]}" --output_path "$EVAL_DIR/$NAME" \
        || { echo "[eval] FAILED: $NAME" >&2; FAILED=$((FAILED + 1)); }
done

# --- table --------------------------------------------------------------------
# Runs even after a partial failure, so a run where one checkpoint OOMed still
# yields a table for the rest (the gaps are listed in RESULTS.md).
python "$REPO/experiments/lib/summarize_eval.py" "$EXP_DIR" || true

echo ""
echo "[eval] results   : $EXP_DIR/RESULTS.md"
echo "[eval] full data : $EVAL_DIR/"
[ "$FAILED" -gt 0 ] && echo "[eval] $FAILED evaluation(s) FAILED -- see above." >&2
echo "[eval] end = $(date --iso-8601=seconds)"
exit 0

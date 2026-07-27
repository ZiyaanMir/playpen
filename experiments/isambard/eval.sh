#!/bin/bash
# Evaluation half of an Isambard experiment: lm-eval over every checkpoint the
# training job wrote, plus the untrained base model as a baseline. Submitted by
# run_experiment.sh with --dependency=afterany on the training job.
#
# Results land inside the experiment directory, NOT in a shared Results/ tree:
#     $EXP_DIR/eval/base/            untrained model
#     $EXP_DIR/eval/checkpoint-50/   ... one dir per adapter
#     $EXP_DIR/RESULTS.md            the table across all of them
#
# Re-runnable by hand once training is done:
#     source <EXP_DIR>/experiment.env && experiments/isambard/eval.sh
#
#SBATCH --job-name=marshal_eval
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=05:00:00
#SBATCH --no-requeue

set -euo pipefail

: "${EXP_ENV_FILE:?not set -- submit via experiments/isambard/run_experiment.sh}"
# shellcheck disable=SC1090
source "$EXP_ENV_FILE"

cd "$REPO"
source "$REPO/experiments/lib/experiment.sh"
exp_layout
exp_banner eval

# --- the lm-eval environment (NOT the training venv) -------------------------
# A pre-existing venv, activated by `source`; logiglue/logicbench are already
# registered in it, so no --include_path is needed. Default points at the env the
# user maintains under evaluation/; override with VENV_LMEVAL=<path> if it moves.
VENV_LMEVAL="${VENV_LMEVAL:-$PROJECTDIR/$USER/evaluation/eval}"
[ -f "$VENV_LMEVAL/bin/activate" ] || {
    echo "ERROR: no lm-eval venv at $VENV_LMEVAL" >&2
    echo "       Expected the pre-built 'eval' venv under evaluation/. If it lives" >&2
    echo "       elsewhere, submit with VENV_LMEVAL=<path-to-venv>." >&2
    exit 1
}
# Verified by file identity, not path spelling: /projects/... and /lus/... are the
# same directory on Isambard, and activate hardcodes the physical path.
exp_activate_venv "$VENV_LMEVAL" "lm-eval venv" || exit 1

# Must run BEFORE the import check below: the inherited TMPDIR (/local/user/<uid>)
# is a login-node path that does not exist on the compute node, and `import peft`
# pulls in torch, whose inductor cache setup dies there. See exp_setup_tmpdir.
exp_setup_tmpdir || exit 1

export HF_HOME="${HF_HOME:-$PROJECTDIR/hf}"
# Isambard compute nodes DO have outbound access (ISAMBARD_GUIDE.md §8 describes HF
# rate-limiting over a shared outbound IP, and documents `srun ... uv pip install`),
# so we do NOT force offline mode here. Forcing it was the bug: the model weights
# were cached but the task DATASETS were not, and lm-eval died on every checkpoint
# with
#   ConnectionError: Couldn't reach 'logicreasoning/logi_glue' (OfflineModeIsEnabled)
# Leaving this at 0 lets the first run fetch the datasets into $HF_HOME; later runs
# hit that cache. Prefetch them on a login node with
# experiments/lib/prefetch_eval_data.sh to avoid the shared-IP rate limit mid-job,
# then set HF_HUB_OFFLINE=1 if you want a guaranteed-network-free run.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$EVAL_DIR"

python -c "import lm_eval, peft; print('[eval] lm_eval', lm_eval.__version__, '| peft', peft.__version__)"

# Slurm gives clean integer GPU ordinals, so --device cuda:0 needs no remap
# (unlike Eddie, where CUDA_VISIBLE_DEVICES is a UUID).

RUN_DIR="$(exp_find_run_dir || true)"
if [ -z "$RUN_DIR" ]; then
    echo "[eval] no training run directory under $TRAIN_BASE." >&2
    echo "[eval] training produced nothing to evaluate -- check $LOG_DIR/train_*.err" >&2
    exit 0
fi
echo "[eval] run dir = $RUN_DIR"

mapfile -t CHECKPOINTS < <(exp_list_checkpoints "$RUN_DIR")
echo "[eval] ${#CHECKPOINTS[@]} checkpoint(s) to score"

# Pre-flight: resolve the task data ONCE, with no model and no GPU work. A missing or
# unreachable dataset then fails here in seconds instead of after loading Qwen3-4B for
# every checkpoint in turn -- the "8 evaluation(s) FAILED" loop that made a dataset
# problem look like an evaluation problem.
python - "$EVAL_TASKS" <<'PY'
import sys
tasks = [t for t in sys.argv[1].split(",") if t]
try:
    from lm_eval.tasks import TaskManager
    loader = getattr(TaskManager(), "load", None)
except Exception:
    loader = None
if loader is None:               # older/newer lm-eval without this API
    print("[eval] task pre-flight unavailable on this lm-eval; skipping")
    sys.exit(0)
try:
    loader(tasks)
except Exception as exc:
    print("[eval] task data NOT available: %s: %s" % (type(exc).__name__, exc))
    sys.exit(3)
print("[eval] task data OK: %s" % ", ".join(tasks))
PY
case $? in
    0) ;;
    3) echo "ERROR: the task datasets for '$EVAL_TASKS' could not be loaded." >&2
       echo "       Nothing was evaluated -- fix the data, then re-run eval only:" >&2
       echo "         experiments/lib/prefetch_eval_data.sh '$EVAL_TASKS'   # on a LOGIN node" >&2
       echo "         experiments/isambard/run_eval.sh '$EXP_DIR'" >&2
       exit 1 ;;
esac

# EVAL_BATCH defaults to 16 from the preset; a ~96 GB GH200 comfortably takes more
# for loglikelihood tasks. Raise with EVAL_BATCH=32 at submit time.
COMMON=( --tasks "$EVAL_TASKS" --device cuda:0 --batch_size "$EVAL_BATCH" --log_samples )
[ -n "${EVAL_LIMIT:-}" ] && COMMON+=( --limit "$EVAL_LIMIT" )
# Unquoted on purpose: a pre-split flag string.
[ -n "${EVAL_EXTRA:-}" ] && COMMON+=( ${EVAL_EXTRA} )

# NOTE: --apply_chat_template is deliberately NOT set, matching the base-model
# baseline below. Applying it to one side only makes the comparison meaningless;
# put it in EVAL_EXTRA if you want it on every row.

FAILED=0

# The base result depends only on (model, tasks, limit, extra), never on the
# checkpoint, so it is identical for every experiment on this model. exp_eval_base
# runs it once, caches it under $BASE_EVAL_CACHE, and copies the stored result into
# later experiments instead of re-running on the GPU.
if [ "${EVAL_BASE:-1}" = "1" ]; then
    echo "=== [eval] base model ==="
    CACHE_ROOT="${BASE_EVAL_CACHE:-$MARSHAL_RUNS/_base_eval_cache}"
    exp_eval_base "$MODEL" "$EVAL_DIR/base" "$CACHE_ROOT" -- "${COMMON[@]}" \
        || { echo "[eval] FAILED: base model" >&2; FAILED=$((FAILED + 1)); }
fi

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

python "$REPO/experiments/lib/summarize_eval.py" "$EXP_DIR" || true

echo ""
echo "[eval] results   : $EXP_DIR/RESULTS.md"
echo "[eval] full data : $EVAL_DIR/"
[ "$FAILED" -gt 0 ] && echo "[eval] $FAILED evaluation(s) FAILED -- see above." >&2
echo "[eval] end = $(date --iso-8601=seconds)"
exit 0

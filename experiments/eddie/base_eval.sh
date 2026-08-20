#!/bin/bash
# Score ONLY the untrained base model, once, and publish that row into every
# experiment that needs it. Submitted by experiments/eddie/run_base_eval.sh.
#
# WHY THIS EXISTS SEPARATELY FROM eval.sh. The base row depends solely on
# (model, tasks, limit, extra) -- never on a checkpoint -- so it is IDENTICAL across
# every experiment on the same model. eval.sh knows that (exp_eval_base caches it),
# but eval.sh always walks the checkpoints too. When only the base row is missing,
# re-running eval.sh would re-score every checkpoint to fix one row.
#
# THE CASE THIS WAS WRITTEN FOR. The Qwen3-8B runs were scored for logicbench after
# the empty-group bug was fixed, and their CHECKPOINTS all have bqa_*/mcqa_* scores --
# but the base row does not, so every logicbench delta on those runs is measured
# against a row that has no logicbench in it. One base evaluation, copied into all 13
# experiments, fixes the lot. Thirteen eval.sh re-runs would have cost ~130 checkpoint
# evaluations to achieve the same thing.
#
# It reads its target list from $BASE_EVAL_TARGETS (one experiment directory per
# line), written by the submitter.
#
# --- Grid Engine options (overridable via BASE_EVAL_QSUB_OPTS) ----------------
#$ -N marshal_base_eval
#$ -cwd
#$ -q gpu
#$ -l gpu=1
#$ -l h_rt=08:00:00
#$ -pe sharedmem 4
#$ -l h_rss=12G

set -euo pipefail

: "${EXP_ENV_FILE:?not set -- submit via experiments/eddie/run_base_eval.sh}"
: "${BASE_EVAL_TARGETS:?not set -- submit via experiments/eddie/run_base_eval.sh}"
# shellcheck disable=SC1090
source "$EXP_ENV_FILE"

cd "$REPO"
source "$REPO/experiments/lib/experiment.sh"
exp_layout

echo "=================================================================="
echo "  MARSHAL experiment -- BASE-ONLY eval"
echo "=================================================================="
echo "model       = $MODEL"
echo "tasks       = ${EVAL_TASKS} (batch ${EVAL_BATCH})"
echo "targets     = $(wc -l < "$BASE_EVAL_TARGETS") experiment(s)"
echo "cluster     = eddie"
echo "host        = $(hostname)"
echo "job         = ${JOB_ID:-?}"
echo "start       = $(date --iso-8601=seconds)"
echo "=================================================================="

# --- the lm-eval environment -- identical to eval.sh, deliberately -----------
# Any divergence here means this job can pass while eval.sh fails, or vice versa.
SCRATCH="${SCRATCH:-/exports/eddie/scratch/$USER}"

[ -r /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh
module load cuda

LMEVAL_VENV_DEFAULT="$SCRATCH/lmeval-venv"
if [ -n "${VENV_LMEVAL:-}" ] || [ -f "$LMEVAL_VENV_DEFAULT/bin/activate" ]; then
    VENV_LMEVAL="${VENV_LMEVAL:-$LMEVAL_VENV_DEFAULT}"
    exp_activate_venv "$VENV_LMEVAL" "lm-eval venv" || exit 1
else
    module load anaconda/2024.02
    # shellcheck disable=SC1091
    conda activate "${LMEVAL_CONDA_ENV:-lmeval}"
fi

export HF_HOME="$SCRATCH/home_cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
mkdir -p "$HF_HOME"

# TMPDIR and every torch/triton/inductor cache under it, per job. Without this,
# Triton writes to $HOME/.triton and the run dies on the 10 GB home quota partway
# through -- which is exactly what killed the 2026-08-18 backfill.
exp_setup_tmpdir || exit 1

python -c "import lm_eval, peft; print('[base-eval] lm_eval', lm_eval.__version__, '| peft', peft.__version__)"

CACHE_ROOT="${BASE_EVAL_CACHE:-$MARSHAL_RUNS/_base_eval_cache}"
KEY="$(exp_base_cache_key "$MODEL" "${EVAL_TASKS:-}" "${EVAL_LIMIT:-}" "${EVAL_EXTRA:-}")"
echo "[base-eval] cache key = $KEY"
echo "[base-eval] cache dir = $CACHE_ROOT/$KEY"

# --- invalidate a stale cache entry ------------------------------------------
# The cache key does NOT include the harness or environment version, so a base row
# computed while logicbench registered as an EMPTY GROUP is still a cache HIT today.
# That is precisely the row being repaired here, so refreshing has to be possible.
if [ "${BASE_EVAL_REFRESH:-0}" = "1" ]; then
    echo "[base-eval] BASE_EVAL_REFRESH=1 -- removing the cached entry first"
    rm -rf "${CACHE_ROOT:?}/${KEY:?}"
fi

# --- score the base model once ------------------------------------------------
COMMON=( --tasks "$EVAL_TASKS" --device cuda:0 --batch_size "$EVAL_BATCH" --log_samples )
[ -n "${EVAL_LIMIT:-}" ] && COMMON+=( --limit "$EVAL_LIMIT" )
# Unquoted on purpose: a pre-split flag string.
[ -n "${EVAL_EXTRA:-}" ] && COMMON+=( ${EVAL_EXTRA} )

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/base_eval.XXXXXX")"
echo "=== [base-eval] base model: $MODEL ==="
if ! exp_eval_base "$MODEL" "$STAGE" "$CACHE_ROOT" -- "${COMMON[@]}"; then
    echo "[base-eval] FAILED: could not score the base model." >&2
    rm -rf "$STAGE"
    exit 1
fi

# --- verify it actually scored something -------------------------------------
# An empty group still writes a results.json and still exits 0 -- the failure mode
# this whole exercise exists to undo. Refusing to publish a row that does not contain
# the required tasks is what stops it propagating into 13 experiments a second time.
if [ -n "${BASE_EVAL_REQUIRE:-}" ]; then
    if ! python - "$STAGE" "$BASE_EVAL_REQUIRE" <<'PYEND'
import glob, json, os, sys
stage, require = sys.argv[1], sys.argv[2]
hits = 0
for p in glob.glob(os.path.join(stage, "**", "results*.json"), recursive=True):
    try:
        res = json.load(open(p)).get("results", {}) or {}
    except Exception:
        continue
    for k, v in res.items():
        if k.startswith(require) and isinstance(v, dict) \
           and any(m.endswith(",none") for m in v):
            hits += 1
print("[base-eval] tasks matching '%s*' with real metrics: %d" % (require, hits))
sys.exit(0 if hits else 1)
PYEND
    then
        echo "[base-eval] FAILED: the result contains no '${BASE_EVAL_REQUIRE}*' task with" >&2
        echo "            real metrics. The task group almost certainly registered EMPTY." >&2
        echo "            Nothing has been published to the experiments. Check the lm-eval" >&2
        echo "            environment for duplicate task trees:" >&2
        echo "              python -c 'from lm_eval.tasks import TaskManager; TaskManager()' 2>&1 | grep -c Duplicate" >&2
        echo "            Then re-submit with BASE_EVAL_REFRESH=1." >&2
        rm -rf "$STAGE"
        exit 1
    fi
fi

# --- publish into every target ------------------------------------------------
# Copied, not symlinked, so each experiment directory stays self-contained for rsync
# -- the same reason exp_eval_base copies. Copying INTO eval/base/ without clearing it
# is deliberate: an existing logiglue results.json must survive, because
# summarize_eval.py merges every results file per checkpoint and a base row needs both
# benchmarks.
PUBLISHED=0
FAILED=0
while IFS= read -r TARGET; do
    [ -n "$TARGET" ] || continue
    if [ ! -d "$TARGET" ]; then
        echo "[base-eval] SKIP (no such directory): $TARGET" >&2
        FAILED=$((FAILED + 1)); continue
    fi
    mkdir -p "$TARGET/eval/base"
    cp -r "$STAGE/." "$TARGET/eval/base/"
    # Rebuild that experiment's table from everything now on disk. Stdlib only.
    if python "$REPO/experiments/lib/summarize_eval.py" "$TARGET"; then
        PUBLISHED=$((PUBLISHED + 1))
    else
        echo "[base-eval] summarize failed for $TARGET" >&2
        FAILED=$((FAILED + 1))
    fi
done < "$BASE_EVAL_TARGETS"

rm -rf "$STAGE"

echo ""
echo "[base-eval] published the base row into $PUBLISHED experiment(s)."
[ "$FAILED" -gt 0 ] && echo "[base-eval] $FAILED target(s) had a problem -- see above." >&2
echo "[base-eval] end = $(date --iso-8601=seconds)"
[ "$FAILED" -gt 0 ] && exit 1
exit 0

#!/bin/bash
# Score the untrained base model ONCE and publish that row into every experiment
# given, then rebuild each RESULTS.md. No checkpoint is re-scored.
#
#   cd <repo root>
#   experiments/eddie/run_base_eval.sh -n  '*Qwen3-8B*'
#   EVAL_TASKS=logicbench experiments/eddie/run_base_eval.sh '*Qwen3-8B*'
#   EVAL_TASKS=logicbench experiments/eddie/run_base_eval.sh $MARSHAL_RUNS/taboo_Qwen3-8B_final_20260814-072905
#
# Arguments are experiment directories, or globs matched against directory names
# under $MARSHAL_RUNS.
#
# WHY. A base row depends only on (model, tasks, limit, extra), never on a checkpoint,
# so it is the same for every experiment on that model. When the checkpoints are
# already scored and only the base row is missing -- the state the Qwen3-8B runs are
# in for logicbench -- re-running run_eval.sh would re-score ~130 checkpoints to
# produce 13 copies of one row. This runs it once and copies it.
#
# EVAL_TASKS IS THE KEY INPUT. It selects the cache entry AND what gets scored. Set it
# to the benchmark that is missing (logicbench), not the full set: summarize_eval.py
# merges every results file per row, so the existing logiglue numbers in eval/base/ are
# kept and the new logicbench ones are added alongside.
#
# The result is verified before publication: with EVAL_TASKS containing "logicbench"
# it requires bqa_*/mcqa_* tasks carrying real metrics, so an empty task group cannot
# be copied into 13 experiments a second time. Override with BASE_EVAL_REQUIRE=<prefix>,
# or BASE_EVAL_REQUIRE='' to skip the check.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO:-$(dirname "$(dirname "$HERE")")}"

DRY_RUN=0
REFRESH=0
PATTERNS=()

usage() {
    cat <<EOF
usage: $(basename "$0") [-n] [--refresh] <EXP_DIR|pattern> [...]

Scores the base model once and publishes the row into every matching experiment,
then rebuilds their RESULTS.md. Checkpoints are never touched.

options:
  -n, --dry-run   list the targets and the cache key, submit nothing
      --refresh   delete the cached base result for this (model, tasks, limit, extra)
                  before running. USE THIS when the cached row is wrong rather than
                  missing -- e.g. it was computed while logicbench registered as an
                  empty group. The cache key does not include the environment
                  version, so a bad row stays a cache HIT forever otherwise.
  -h, --help      this

environment:
  EVAL_TASKS      REQUIRED in effect -- the benchmark to score (e.g. logicbench).
                  Defaults to the first experiment's stored value, which is usually
                  the full set and therefore not what you want here.
  EVAL_BATCH      default: the experiment's stored value (usually auto)
  EVAL_LIMIT      lm-eval --limit (smoke tests). Part of the cache key.
  EVAL_EXTRA      extra lm-eval flags. Part of the cache key.
  BASE_EVAL_CACHE cache root (default \$MARSHAL_RUNS/_base_eval_cache)
  BASE_EVAL_REQUIRE  task prefix that must be present with real metrics
                  (auto: bqa_ when EVAL_TASKS mentions logicbench)
  VENV_LMEVAL / LMEVAL_CONDA_ENV   the lm-eval environment
  BASE_EVAL_QSUB_OPTS   extra qsub options
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -n|--dry-run) DRY_RUN=1 ;;
        --refresh)    REFRESH=1 ;;
        -h|--help)    usage; exit 0 ;;
        -*)           echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
        *)            PATTERNS+=("$1") ;;
    esac
    shift
done
[ "${#PATTERNS[@]}" -gt 0 ] || { usage >&2; exit 2; }

# Snapshot the caller's overrides BEFORE sourcing any experiment.env, which would
# otherwise replace them with the stored values -- the precedence trap every other
# submitter here documents.
_OV_NAMES=(); _OV_VALS=()
for _v in EVAL_TASKS EVAL_BATCH EVAL_LIMIT EVAL_EXTRA BASE_EVAL_CACHE \
          VENV_LMEVAL LMEVAL_CONDA_ENV HF_HUB_OFFLINE; do
    if [ -n "${!_v+set}" ]; then _OV_NAMES+=("$_v"); _OV_VALS+=("${!_v}"); fi
done

SCRATCH="${SCRATCH:-/exports/eddie/scratch/$USER}"
RUNS="${MARSHAL_RUNS:-$SCRATCH/marshal-runs}"
[ -d "$RUNS" ] || { echo "ERROR: no experiments root: $RUNS" >&2; exit 1; }

# --- resolve the targets ------------------------------------------------------
TARGETS=()
for P in "${PATTERNS[@]}"; do
    if [ -d "$P" ] && [ -f "$P/experiment.env" ]; then
        TARGETS+=( "$(cd "$P" && pwd)" ); continue
    fi
    _found=0
    for D in "$RUNS"/*/; do
        D="${D%/}"
        [ -f "$D/experiment.env" ] || continue
        case "$(basename "$D")" in
            # shellcheck disable=SC2254
            $P) TARGETS+=( "$D" ); _found=1 ;;
        esac
    done
    [ "$_found" = 1 ] || echo "WARNING: nothing matched '$P'" >&2
done
[ "${#TARGETS[@]}" -gt 0 ] || { echo "ERROR: no experiments matched." >&2; exit 1; }

# Dedupe, keeping order.
mapfile -t TARGETS < <(printf '%s\n' "${TARGETS[@]}" | awk '!seen[$0]++')

# --- one model only -----------------------------------------------------------
# A base row belongs to a MODEL. Publishing one model's row into another model's
# experiment would silently corrupt every delta in that table, so this is fatal
# rather than a warning.
FIRST_ENV="${TARGETS[0]}/experiment.env"
# shellcheck disable=SC1090
source "$FIRST_ENV"
BASE_MODEL="$MODEL"
MISMATCH=0
for T in "${TARGETS[@]}"; do
    M="$(sed -n 's/^MODEL=//p' "$T/experiment.env" | tr -d "\"'")"
    if [ "$M" != "$BASE_MODEL" ]; then
        echo "ERROR: $(basename "$T") is $M, not $BASE_MODEL." >&2
        MISMATCH=1
    fi
done
[ "$MISMATCH" = 0 ] || {
    echo "ERROR: all targets must share one MODEL -- the base row is per-model." >&2
    echo "       Narrow the pattern and run once per model." >&2
    exit 1
}

# Re-apply the caller's overrides on top of the sourced env file.
for _i in "${!_OV_NAMES[@]}"; do
    printf -v "${_OV_NAMES[$_i]}" '%s' "${_OV_VALS[$_i]}"
done

# Default the verification prefix from the task set. logicbench's members are
# bqa_*/mcqa_*; its GROUP row is written even when every member fails to register,
# so checking the group name would verify nothing.
if [ -z "${BASE_EVAL_REQUIRE+set}" ]; then
    case "${EVAL_TASKS:-}" in
        *logicbench*|*logic_bench*) BASE_EVAL_REQUIRE=bqa_ ;;
        *)                          BASE_EVAL_REQUIRE="" ;;
    esac
fi

source "$REPO_ROOT/experiments/lib/experiment.sh"
CACHE_ROOT="${BASE_EVAL_CACHE:-$RUNS/_base_eval_cache}"
KEY="$(exp_base_cache_key "$BASE_MODEL" "${EVAL_TASKS:-}" "${EVAL_LIMIT:-}" "${EVAL_EXTRA:-}")"

echo "model     : $BASE_MODEL"
echo "tasks     : ${EVAL_TASKS:-<unset>}"
echo "batch     : ${EVAL_BATCH:-auto}"
echo "verify    : ${BASE_EVAL_REQUIRE:-<none>}"
echo "cache key : $KEY"
echo "cache dir : $CACHE_ROOT/$KEY$( [ -d "$CACHE_ROOT/$KEY" ] && echo '  (EXISTS)' || echo '  (not cached)')"
echo "refresh   : $( [ "$REFRESH" = 1 ] && echo 'YES -- cached entry will be deleted' || echo 'no' )"
echo "targets   : ${#TARGETS[@]}"
for T in "${TARGETS[@]}"; do echo "    $(basename "$T")"; done
echo

if [ -d "$CACHE_ROOT/$KEY" ] && [ "$REFRESH" = 0 ]; then
    echo "NOTE: this key is already cached, so the job will COPY it rather than run the"
    echo "      GPU. That is right when the row is merely missing from these experiments."
    echo "      If the cached row itself is wrong -- e.g. computed while logicbench"
    echo "      registered empty -- re-run with --refresh."
    echo
fi

if [ "$DRY_RUN" = 1 ]; then
    echo "DRY RUN -- nothing submitted."
    exit 0
fi

# --- the lm-eval environment, checked once before spending a GPU --------------
if [ "${EVAL_SKIP_ENV_CHECK:-0}" != "1" ]; then
    VENV_LMEVAL="${VENV_LMEVAL:-}" LMEVAL_CONDA_ENV="${LMEVAL_CONDA_ENV:-}" \
        bash "$REPO_ROOT/experiments/lib/check_lmeval_env.sh" --cluster eddie || {
        echo "" >&2
        echo "ERROR: not submitting -- the lm-eval environment is broken (see above)." >&2
        exit 1
    }
    echo
fi

# --- the env file and target list the job reads -------------------------------
# Kept beside the first experiment, and named so it cannot collide with the files
# run_eval.sh / run_playpen_eval.sh write.
WORK="${TARGETS[0]}"
ENV_FILE="$WORK/experiment.baseeval.env"
LIST_FILE="$WORK/experiment.baseeval.targets"

cp "$FIRST_ENV" "$ENV_FILE"
{
    echo ""
    echo "# --- base-eval-only overrides, $(date --iso-8601=seconds) ---"
    for _i in "${!_OV_NAMES[@]}"; do
        printf '%s=%q\nexport %s\n' "${_OV_NAMES[$_i]}" "${_OV_VALS[$_i]}" "${_OV_NAMES[$_i]}"
    done
    printf 'BASE_EVAL_REQUIRE=%q\nexport BASE_EVAL_REQUIRE\n' "${BASE_EVAL_REQUIRE:-}"
    printf 'BASE_EVAL_REFRESH=%q\nexport BASE_EVAL_REFRESH\n' "$REFRESH"
    printf 'MARSHAL_RUNS=%q\nexport MARSHAL_RUNS\n' "$RUNS"
} >> "$ENV_FILE"

printf '%s\n' "${TARGETS[@]}" > "$LIST_FILE"

cd "$REPO_ROOT"
LOG_DIR="$WORK/logs"
mkdir -p "$LOG_DIR"

ID="$(qsub -terse \
    -N "base_$(basename "$BASE_MODEL")" \
    -o "$LOG_DIR/baseeval_\$JOB_ID.out" \
    -e "$LOG_DIR/baseeval_\$JOB_ID.err" \
    -v "EXP_ENV_FILE=$ENV_FILE" \
    -v "BASE_EVAL_TARGETS=$LIST_FILE" \
    ${BASE_EVAL_QSUB_OPTS:-} \
    "$HERE/base_eval.sh" | tr -d '[:space:]')"

cat <<EOF
[base-eval] submitted job $ID -- ONE GPU job for all ${#TARGETS[@]} experiment(s).

  tail -f $LOG_DIR/baseeval_$ID.out
  qdel $ID                     # cancel

It scores the base model once, verifies the result contains ${BASE_EVAL_REQUIRE:-<anything>},
copies it into each experiment's eval/base/, and rebuilds every RESULTS.md.
Existing results in eval/base/ are preserved -- summarize_eval.py merges them, so the
logiglue columns stay and logicbench is added alongside.
EOF

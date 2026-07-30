#!/bin/bash
# Queue a playpen games eval for every past experiment that has never had one.
#
#   cd <repo root>
#   experiments/queue_playpen_eval_backfill.sh -n          # what WOULD be queued
#   experiments/queue_playpen_eval_backfill.sh             # queue it
#   experiments/queue_playpen_eval_backfill.sh 'guesswhat_*'
#
# The playpen-games evaluation was added after most of these runs finished, so their
# checkpoints have lm-eval scores and no clemscore. This walks $MARSHAL_RUNS, finds
# the experiments with real checkpoints and no gameplay results, and submits one
# eval job per experiment through the cluster's own run_playpen_eval.sh -- the same
# path a fresh experiment takes, so a backfilled run is indistinguishable from one
# that was scored on the day.
#
# Nothing is retrained and no checkpoint is touched. Re-running this is safe: an
# experiment that already has results is skipped, so it only ever queues what is
# genuinely missing.
#
# Because each job wants a whole GPU for hours, --limit is worth using on a busy
# queue -- run it again tomorrow and it picks up where it left off.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(dirname "$HERE")}"

DRY_RUN=0
FORCE=0
LIMIT=0
CLUSTER=""
RUNS=""
PATTERNS=()

usage() {
    cat <<EOF
usage: $(basename "$0") [options] [experiment-pattern ...]

Submits one playpen-games eval per past experiment that has checkpoints but no
gameplay results yet.

options:
  -n, --dry-run      list what would be queued, submit nothing
      --force        re-queue experiments that already have results
      --limit N      queue at most N jobs this time (0 = no limit)
      --runs DIR     experiments root      (default: \$MARSHAL_RUNS or the cluster's)
      --cluster C    eddie | isambard      (default: detected from qsub/sbatch)
  -h, --help         this

Patterns are shell globs matched against the experiment directory name:
  $(basename "$0") 'guesswhat_*' 'dond_Qwen3-4B_*'

Any PPEVAL_* setting is passed through to every job it submits, e.g.
  PPEVAL_CKPTS=last  $(basename "$0")      # only the final checkpoint of each run
  PPEVAL_GAMES=dond,guesswhat  $(basename "$0") 'dond_*'
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -n|--dry-run)  DRY_RUN=1 ;;
        --force)       FORCE=1 ;;
        --limit)       LIMIT="$2"; shift ;;
        --runs)        RUNS="$2"; shift ;;
        --cluster)     CLUSTER="$2"; shift ;;
        -h|--help)     usage; exit 0 ;;
        -*)            echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
        *)             PATTERNS+=("$1") ;;
    esac
    shift
done

# --- which cluster ------------------------------------------------------------
# Detected from the scheduler that is actually present, so the common case needs no
# flag. Both present (or neither) is ambiguous rather than guessable -- say so.
if [ -z "$CLUSTER" ]; then
    if command -v qsub >/dev/null 2>&1 && ! command -v sbatch >/dev/null 2>&1; then
        CLUSTER=eddie
    elif command -v sbatch >/dev/null 2>&1 && ! command -v qsub >/dev/null 2>&1; then
        CLUSTER=isambard
    elif [ "$DRY_RUN" = 1 ]; then
        CLUSTER=eddie          # only used to name the submitter in the listing
    else
        echo "ERROR: could not tell which cluster this is (qsub/sbatch both present" >&2
        echo "       or both absent). Pass --cluster eddie|isambard." >&2
        exit 1
    fi
fi
SUBMITTER="$HERE/$CLUSTER/run_playpen_eval.sh"
[ -x "$SUBMITTER" ] || [ "$DRY_RUN" = 1 ] || {
    echo "ERROR: no submitter at $SUBMITTER" >&2; exit 1; }

# --- where the experiments are ------------------------------------------------
if [ -z "$RUNS" ]; then
    RUNS="${MARSHAL_RUNS:-}"
fi
if [ -z "$RUNS" ]; then
    if [ -n "${PROJECTDIR:-}" ] && [ -d "$PROJECTDIR/$USER/marshal-runs" ]; then
        RUNS="$PROJECTDIR/$USER/marshal-runs"
    else
        RUNS="/exports/eddie/scratch/$USER/marshal-runs"
    fi
fi
[ -d "$RUNS" ] || { echo "ERROR: no experiments root: $RUNS" >&2; exit 1; }

# --- what needs doing ---------------------------------------------------------
# An experiment qualifies when it has at least one COMPLETE checkpoint (a directory
# with adapter_config.json -- a job killed at the walltime can leave one without)
# and no gameplay results. "Has results" is keyed on clemeval's results.csv, not on
# the playpen-eval directory existing: a run that died mid-gameplay leaves the
# directory behind, and that must count as still needing doing.
matches_pattern() {
    local name="$1" p
    [ "${#PATTERNS[@]}" -eq 0 ] && return 0
    for p in "${PATTERNS[@]}"; do
        # shellcheck disable=SC2053
        [[ "$name" == $p ]] && return 0
    done
    return 1
}

count_checkpoints() {
    local exp="$1" n=0 ck
    shopt -s nullglob
    for ck in "$exp"/train/checkpoint-* "$exp"/train/*/checkpoint-*; do
        [ -f "$ck/adapter_config.json" ] && n=$((n + 1))
    done
    shopt -u nullglob
    printf '%d\n' "$n"
}

has_playpen_results() {
    [ -n "$(find "$1/playpen-eval" -name 'results.csv' -print -quit 2>/dev/null)" ]
}

echo "runs    : $RUNS"
echo "cluster : $CLUSTER"
[ "${#PATTERNS[@]}" -gt 0 ] && echo "patterns: ${PATTERNS[*]}"
echo

TODO=()
SKIP_DONE=0 SKIP_NOCK=0 SKIP_PAT=0 SKIP_NOTEXP=0
for EXP in "$RUNS"/*/; do
    EXP="${EXP%/}"
    NAME="$(basename "$EXP")"
    case "$NAME" in _*) continue ;; esac         # _base_eval_cache, _playpen_base_cache
    if [ ! -f "$EXP/manifest.json" ] || [ ! -f "$EXP/experiment.env" ]; then
        SKIP_NOTEXP=$((SKIP_NOTEXP + 1)); continue
    fi
    matches_pattern "$NAME" || { SKIP_PAT=$((SKIP_PAT + 1)); continue; }
    N_CK="$(count_checkpoints "$EXP")"
    if [ "$N_CK" -eq 0 ]; then
        SKIP_NOCK=$((SKIP_NOCK + 1)); continue
    fi
    if [ "$FORCE" = 0 ] && has_playpen_results "$EXP"; then
        SKIP_DONE=$((SKIP_DONE + 1)); continue
    fi
    TODO+=("$EXP")
done

if [ "${#TODO[@]}" -eq 0 ]; then
    echo "nothing to queue."
    echo "  $SKIP_DONE already scored, $SKIP_NOCK without checkpoints," \
         "$SKIP_PAT filtered out, $SKIP_NOTEXP not experiment dirs."
    echo "  (--force re-queues the already-scored ones.)"
    exit 0
fi

printf '%-58s %s\n' EXPERIMENT CKPTS
printf '%-58s %s\n' "$(printf '%.0s-' {1..58})" -----
for EXP in "${TODO[@]}"; do
    printf '%-58s %s\n' "$(basename "$EXP")" "$(count_checkpoints "$EXP")"
done
echo
echo "${#TODO[@]} experiment(s) to queue" \
     "($SKIP_DONE already scored, $SKIP_NOCK without checkpoints)."

if [ "$DRY_RUN" = 1 ]; then
    echo
    echo "DRY RUN -- nothing submitted. Re-run without -n to queue these."
    exit 0
fi

if [ "$LIMIT" -gt 0 ] && [ "${#TODO[@]}" -gt "$LIMIT" ]; then
    echo "--limit $LIMIT: queueing the first $LIMIT; re-run later for the rest."
    TODO=("${TODO[@]:0:$LIMIT}")
fi
echo

cd "$REPO"
OK=0 FAIL=0
for EXP in "${TODO[@]}"; do
    echo "--- $(basename "$EXP") ---"
    # PPEVAL_* stays exported from this shell, so the submitter records it as an
    # override in that experiment's env file -- one place to set the policy for a
    # whole backfill.
    if "$SUBMITTER" "$EXP"; then
        OK=$((OK + 1))
    else
        echo "[backfill] FAILED to submit $(basename "$EXP")" >&2
        FAIL=$((FAIL + 1))
    fi
    echo
done

echo "queued $OK job(s)${FAIL:+, $FAIL failed to submit}."
echo "watch:   $([ "$CLUSTER" = eddie ] && echo "qstat -u $USER" || echo 'squeue --me')"
echo "results: experiments/status.sh -v $RUNS   then  cat <exp>/PLAYPEN_RESULTS.md"
[ "$FAIL" -gt 0 ] && exit 1
exit 0

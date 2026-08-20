#!/bin/bash
# Re-queue the lm-eval half for every past experiment that has none.
#
#   cd <repo root>
#   experiments/queue_eval_backfill.sh -n            # what WOULD be queued
#   experiments/queue_eval_backfill.sh               # queue it
#   experiments/queue_eval_backfill.sh '*Qwen3-8B*'  # just the 8B runs
#
# WHY THIS EXISTS. From 2026-08-10 to 2026-08-15 every Eddie eval shard died within
# seconds of starting:
#     ModuleNotFoundError: No module named 'lm_eval'
# The `lmeval` conda env that eddie/eval.sh activates had lost the package. 65
# experiments came out of that window with a full playpen-eval/ (the gameplay half
# runs in the TRAINING venv, so it was untouched), an empty eval/, and no RESULTS.md.
# The summary job still ran, still said "found no results*.json", and still exited 0 --
# so nothing flagged it. This walks $MARSHAL_RUNS, finds the experiments whose
# checkpoints have no lm-eval results, and re-submits through the cluster's own
# run_eval.sh -- the same path a fresh experiment takes.
#
# Nothing is retrained, no checkpoint is touched, and gameplay results are preserved
# (summarize.sh rebuilds both tables, so PLAYPEN_RESULTS.md survives). Re-running this
# is safe: an experiment that already has lm-eval results is skipped.
#
# THE ENVIRONMENT IS CHECKED ONCE, UP FRONT, and nothing is submitted if it is still
# broken -- otherwise this script's whole job would be to reproduce the original
# failure 130 times on 130 GPU allocations.
#
# NOTE --limit counts EXPERIMENTS, not jobs. Each is submitted as
# ceil(checkpoints / EVAL_SHARD_SIZE) concurrent jobs plus a summary job, so a
# 10-checkpoint experiment is 3 jobs and asks for 2 GPUs at once. On a busy queue,
# EVAL_SHARD_SIZE=999 (one job per experiment) or a smaller --limit is the lever.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(dirname "$HERE")}"

DRY_RUN=0
FORCE=0
FRESH=0
ASSUME_YES=0
SKIP_ENV_CHECK=0
REQUIRE_TASK=""
LIMIT=0
CLUSTER=""
RUNS=""
PATTERNS=()

usage() {
    cat <<EOF
usage: $(basename "$0") [options] [experiment-pattern ...]

Submits the lm-eval half for every past experiment that has checkpoints but no
lm-eval results yet.

options:
  -n, --dry-run      list what would be queued, submit nothing
      --force        re-queue experiments that already have lm-eval results,
                     keeping their existing eval/ directory
      --fresh        DESTRUCTIVE. Delete each selected experiment's eval/ AND the
                     shared base-model eval cache, then recompute from scratch.
                     Implies --force. Asks before deleting unless --yes.
      --yes          skip the --fresh confirmation (for scripts)
      --limit N      queue at most N experiments this time (0 = no limit)
      --runs DIR     experiments root      (default: \$MARSHAL_RUNS or the cluster's)
      --cluster C    eddie | isambard      (default: detected from qsub/sbatch)
      --require-task P   a run counts as done only if some task name starts with P.
                     Use it to re-queue runs that are missing one benchmark, e.g.
                     --require-task bqa_  for logicbench (its group row is written
                     even when every member task failed to register).
      --skip-env-check   submit even if the lm-eval env is broken (don't)
  -h, --help         this

--force vs --fresh. --force writes into a directory that may hold a previous run's
output; summarize_eval.py takes the NEWEST results*.json per checkpoint, so a re-run
supersedes an older one cleanly and --force is normally enough. --fresh is for when
the old results are not missing but WRONG (changed tasks, a harness bug) -- and it
also clears \$BASE_EVAL_CACHE, which --force does not. That cache is keyed on
(model, tasks, limit, extra) and NOT on the harness version, so a baseline computed
under a bug stays a cache HIT and is copied into every new table as the row
everything is compared against.

Patterns are shell globs matched against the experiment directory name:
  $(basename "$0") '*Qwen3-8B*' 'taboo_*'

Any EVAL_* setting is passed through to every job it submits, e.g.
  EVAL_TASKS=logicbench  $(basename "$0")
  EVAL_LIMIT=5           $(basename "$0") -n     # smoke-test the whole backfill
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -n|--dry-run)      DRY_RUN=1 ;;
        --force)           FORCE=1 ;;
        --fresh)           FRESH=1; FORCE=1 ;;
        --yes)             ASSUME_YES=1 ;;
        --skip-env-check)  SKIP_ENV_CHECK=1 ;;
        --require-task)    REQUIRE_TASK="$2"; shift ;;
        --limit)           LIMIT="$2"; shift ;;
        --runs)            RUNS="$2"; shift ;;
        --cluster)         CLUSTER="$2"; shift ;;
        -h|--help)         usage; exit 0 ;;
        -*)                echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
        *)                 PATTERNS+=("$1") ;;
    esac
    shift
done

# --- which cluster ------------------------------------------------------------
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
SUBMITTER="$HERE/$CLUSTER/run_eval.sh"
[ -x "$SUBMITTER" ] || [ -f "$SUBMITTER" ] || [ "$DRY_RUN" = 1 ] || {
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

# Must match eval.sh's default, or --fresh would clear a cache the jobs do not read.
BASE_EVAL_CACHE_DIR="${BASE_EVAL_CACHE:-$RUNS/_base_eval_cache}"

# --- what is still in the queue ----------------------------------------------
# An experiment whose TRAINING is still running must not be evaluated now: the eval
# jobs would score whatever checkpoints happen to exist, and the eval shards the
# original submission already holds behind that training job will run anyway when it
# finishes. Two of the 8B runs were in exactly this state on 2026-08-17.
#
# Cheap and reliable test: does this experiment's manifest name any job id the
# scheduler still knows about? One scheduler call, then a lookup per experiment.
LIVE_IDS=" "
PY="$(command -v python3 || command -v python || true)"
if [ "$CLUSTER" = eddie ] && command -v qstat >/dev/null 2>&1; then
    LIVE_IDS=" $(qstat -u "$USER" 2>/dev/null | awk 'NR>2 {print $1}' | tr '\n' ' ')"
elif [ "$CLUSTER" = isambard ] && command -v squeue >/dev/null 2>&1; then
    LIVE_IDS=" $(squeue --me -h -o '%i' 2>/dev/null | cut -d_ -f1 | tr '\n' ' ')"
fi

has_live_job() {
    local exp="$1" ids id
    [ -n "$PY" ] || return 1                      # cannot read the manifest; assume idle
    [ "$LIVE_IDS" = " " ] && return 1             # nothing queued at all
    ids="$("$PY" - "$exp/manifest.json" <<'EOF' 2>/dev/null || true
import json, sys
try:
    m = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
print(" ".join(str(i) for i in (m.get("jobs", {}) or {}).get("ids", []) or []))
EOF
)"
    for id in $ids; do
        case "$LIVE_IDS" in *" $id "*) return 0 ;; esac
    done
    return 1
}

# --- what needs doing ---------------------------------------------------------
# Qualifies when it has at least one COMPLETE checkpoint (a directory with
# adapter_config.json -- a job killed at the walltime can leave one without) and no
# lm-eval results. "Has results" is keyed on the results*.json summarize_eval.py
# actually reads, not on eval/ existing: every failed shard created eval/ and put
# nothing in it, and that must count as still needing doing.
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

# COVERAGE, not existence. Prints a short problem description; EMPTY means complete.
#
# WHY THIS IS NOT `find -name results*.json`. That was the original test, and it made
# the backfill report "nothing to queue" over a pile of broken runs:
#
#   * 42 runs held a `base` row and NOTHING ELSE. The base row is served from
#     $BASE_EVAL_CACHE ("reusing cached result (no GPU run)"), so it lands even when
#     every checkpoint dies -- as they all did, on the ~/.triton quota bug. One cached
#     file made a run with 20 unscored checkpoints look finished.
#   * 13 runs scored logiglue but not logicbench, because the group registered EMPTY.
#     lm-eval still writes the group row, so a results.json existed either way.
#
# So: count checkpoints actually scored against checkpoints on disk, and treat a task
# that scored ZERO samples as absent. A group whose members all failed to register
# emits {"alias":"logicbench","name":"logicbench","sample_len":0} -- no metrics at all
# -- which is exactly the empty-group signature to catch generically.
# Prints the queue-worthy status on stdout. A base-row gap goes to BASE_NOTE instead,
# because it is fixed by run_base_eval.sh, not by re-scoring checkpoints.
eval_status() {
    BASE_NOTE=""
    [ -n "$PY" ] || { echo ""; return; }
    local out
    out="$(_eval_probe "$1")"
    BASE_NOTE="$(printf '%s\n' "$out" | sed -n 's/^BASE://p')"
    printf '%s\n' "$out" | sed '/^BASE:/d' | head -1
}

_eval_probe() {
    "$PY" - "$1" "${REQUIRE_TASK:-}" <<'PYEND'
import glob, json, os, sys
exp, require = sys.argv[1], sys.argv[2]

def scored_keys(d):
    """Task keys with at least one real metric, merged over every results file here."""
    keys, empty = set(), set()
    for p in glob.glob(os.path.join(d, "**", "results*.json"), recursive=True):
        try:
            res = json.load(open(p)).get("results", {}) or {}
        except Exception:
            continue
        for k, v in res.items():
            if not isinstance(v, dict):
                continue
            if any(m.endswith(",none") and not m.startswith("alias") for m in v):
                keys.add(k)
            elif v.get("sample_len") == 0:
                empty.add(k)
    return keys, empty

cks = set()
for pat in ("train/checkpoint-*", "train/*/checkpoint-*"):
    for c in glob.glob(os.path.join(exp, pat)):
        if os.path.isfile(os.path.join(c, "adapter_config.json")):
            cks.add(os.path.basename(c))
if not cks:
    print("")               # no checkpoints: the caller skips these separately
    raise SystemExit

done, allkeys, allempty = set(), set(), set()
for name in cks:
    k, e = scored_keys(os.path.join(exp, "eval", name))
    allempty |= e
    if k:
        done.add(name)
        allkeys |= k

problems = []
if len(done) < len(cks):
    problems.append("%d/%d ckpt" % (len(done), len(cks)))
if require and not any(k.startswith(require) for k in allkeys):
    problems.append("no %s* tasks" % require)
for g in sorted(allempty):
    if g not in allkeys:
        problems.append("%s EMPTY" % g)

# The base row is reported SEPARATELY, on the second line, and never as a reason to
# re-queue. A missing base row does not need the checkpoints re-scored: the row
# depends only on (model, tasks, limit, extra), so run_base_eval.sh scores it once and
# copies it into every experiment. Folding it into `problems` would spend ~130
# checkpoint evaluations to obtain 13 copies of one row.
basek, _ = scored_keys(os.path.join(exp, "eval", "base"))
if require and not any(k.startswith(require) for k in basek):
    print("; ".join(problems))
    print("BASE:no %s* in base row" % require)
else:
    print("; ".join(problems))
PYEND
}

has_eval_results() {
    [ -z "$(eval_status "$1")" ]
}

echo "runs    : $RUNS"
echo "cluster : $CLUSTER"
[ "${#PATTERNS[@]}" -gt 0 ] && echo "patterns: ${PATTERNS[*]}"
echo

TODO=()
BASE_GAPS=()
SKIP_DONE=0 SKIP_NOCK=0 SKIP_PAT=0 SKIP_NOTEXP=0 SKIP_LIVE=0
LIVE_NAMES=()
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
    # Probe ONCE, here in the current shell. eval_status() cannot be used for the base
    # note: callers invoke it in a command substitution, which is a subshell, so any
    # variable it sets is discarded. Parse both lines of the probe output instead.
    _probe="$(_eval_probe "$EXP")"
    _status="$(printf '%s\n' "$_probe" | sed '/^BASE:/d' | head -1)"
    _basenote="$(printf '%s\n' "$_probe" | sed -n 's/^BASE://p')"
    if [ "$FORCE" = 0 ] && [ -z "$_status" ]; then
        SKIP_DONE=$((SKIP_DONE + 1))
        # Complete on checkpoints, but the BASE row may still lack the benchmark.
        [ -n "$_basenote" ] && BASE_GAPS+=("$NAME")
        continue
    fi
    if has_live_job "$EXP"; then
        SKIP_LIVE=$((SKIP_LIVE + 1)); LIVE_NAMES+=("$NAME"); continue
    fi
    TODO+=("$EXP")
done

if [ "${#BASE_GAPS[@]}" -gt 0 ]; then
    echo
    echo "NOTE: ${#BASE_GAPS[@]} experiment(s) have every checkpoint scored but a BASE row"
    echo "      missing '${REQUIRE_TASK}*'. Every delta in those tables is measured against"
    echo "      a baseline that lacks the benchmark. They are NOT queued here -- re-scoring"
    echo "      their checkpoints would not fix a base row. Fix it in one GPU job with:"
    echo
    echo "        EVAL_TASKS=<the missing benchmark> \\"
    echo "          experiments/$CLUSTER/run_base_eval.sh ${BASE_GAPS[0]}"
    echo
    for n in "${BASE_GAPS[@]}"; do echo "      $n"; done
fi

if [ "${#TODO[@]}" -eq 0 ]; then
    echo "nothing to queue."
    echo "  $SKIP_DONE already scored, $SKIP_NOCK without checkpoints," \
         "$SKIP_LIVE still in the queue, $SKIP_PAT filtered out," \
         "$SKIP_NOTEXP not experiment dirs."
    echo "  (--force re-queues the already-scored ones.)"
    exit 0
fi

printf '%-58s %5s  %s\n' EXPERIMENT CKPTS "what is missing"
printf '%-58s %5s  %s\n' "$(printf '%.0s-' {1..58})" ----- -------
for EXP in "${TODO[@]}"; do
    printf '%-58s %5s  %s\n' "$(basename "$EXP")" "$(count_checkpoints "$EXP")" \
        "$(_st="$(eval_status "$EXP")"; [ -n "$_st" ] && echo "$_st" || echo "complete (re-run)")"
done
echo
echo "${#TODO[@]} experiment(s) to queue" \
     "($SKIP_DONE already scored, $SKIP_NOCK without checkpoints, $SKIP_LIVE still queued)."
if [ "$SKIP_LIVE" -gt 0 ]; then
    echo
    echo "Skipped because the scheduler still has a job for them -- training is not"
    echo "finished, and their original eval shards are still held behind it:"
    for n in "${LIVE_NAMES[@]}"; do echo "  $n"; done
    echo "Re-run this once they are done."
fi

if [ "$DRY_RUN" = 1 ]; then
    echo
    if [ "$FRESH" = 1 ]; then
        echo "DRY RUN (--fresh) -- nothing deleted, nothing submitted."
        echo "Re-running without -n would DELETE the eval/ directory of each experiment"
        echo "above, plus $BASE_EVAL_CACHE_DIR, then recompute from scratch."
    else
        echo "DRY RUN -- nothing submitted. Re-run without -n to queue these."
    fi
    exit 0
fi

# --- the environment, once ----------------------------------------------------
# The one check that separates this run from the one that lost 65 experiments.
if [ "$SKIP_ENV_CHECK" = 0 ]; then
    echo
    bash "$HERE/lib/check_lmeval_env.sh" --cluster "$CLUSTER" || {
        echo "" >&2
        echo "ERROR: nothing submitted -- the lm-eval environment is still broken." >&2
        echo "       Repair it (see above), then re-run this script. Submitting now" >&2
        echo "       would reproduce the original failure on every GPU it is given." >&2
        exit 1
    }
    echo
fi

# --- --fresh: delete stale results before queueing ---------------------------
if [ "$FRESH" = 1 ]; then
    echo "--fresh will DELETE:"
    _n_dirs=0
    for EXP in "${TODO[@]}"; do
        [ -d "$EXP/eval" ] || continue
        printf '  %s  (%s)\n' "$(basename "$EXP")/eval/" \
            "$(du -sh "$EXP/eval" 2>/dev/null | cut -f1)"
        _n_dirs=$((_n_dirs + 1))
    done
    [ "$_n_dirs" -eq 0 ] && echo "  (no existing eval/ directories -- nothing to remove)"
    if [ -d "$BASE_EVAL_CACHE_DIR" ]; then
        printf '  %s  (%s)   <- the untrained-model baseline, shared by every experiment\n' \
            "$BASE_EVAL_CACHE_DIR" "$(du -sh "$BASE_EVAL_CACHE_DIR" 2>/dev/null | cut -f1)"
    fi
    echo
    echo "Everything will be re-scored on the GPU from scratch."

    if [ "$ASSUME_YES" != 1 ]; then
        if [ -t 0 ]; then
            printf 'Type "yes" to proceed: '
            read -r _reply
            [ "$_reply" = "yes" ] || { echo "aborted -- nothing deleted."; exit 1; }
        else
            echo "ERROR: --fresh needs confirmation, but stdin is not a terminal." >&2
            echo "       Re-run with --yes if you are sure." >&2
            exit 1
        fi
    fi

    for EXP in "${TODO[@]}"; do
        rm -rf "$EXP/eval" "$EXP/RESULTS.md"
    done
    rm -rf "$BASE_EVAL_CACHE_DIR"
    echo "[fresh] removed $_n_dirs eval/ director(ies) and the base cache."
    echo
fi

if [ "$LIMIT" -gt 0 ] && [ "${#TODO[@]}" -gt "$LIMIT" ]; then
    echo "--limit $LIMIT: queueing the first $LIMIT; re-run later for the rest."
    TODO=("${TODO[@]:0:$LIMIT}")
    echo
fi

cd "$REPO"
OK=0 FAIL=0
for EXP in "${TODO[@]}"; do
    echo "--- $(basename "$EXP") ---"
    # EVAL_* stays exported from this shell, so the submitter records it as an override
    # in that experiment's env file -- one place to set the policy for a whole backfill.
    # The env check is skipped per-experiment: it already ran once, above.
    if EVAL_SKIP_ENV_CHECK=1 bash "$SUBMITTER" "$EXP"; then
        OK=$((OK + 1))
    else
        echo "[backfill] FAILED to submit $(basename "$EXP")" >&2
        FAIL=$((FAIL + 1))
    fi
    echo
done

# $FAIL is always set, so ${FAIL:+...} would print ", 0 failed to submit" on success.
if [ "$FAIL" -gt 0 ]; then
    echo "queued $OK experiment(s), $FAIL failed to submit."
else
    echo "queued $OK experiment(s)."
fi
echo "watch:   $([ "$CLUSTER" = eddie ] && echo "qstat -u $USER" || echo 'squeue --me')"
echo "results: experiments/status.sh -v $RUNS   then  cat <exp>/RESULTS.md"
[ "$FAIL" -gt 0 ] && exit 1
exit 0

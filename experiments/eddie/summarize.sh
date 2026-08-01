#!/bin/bash
# Final job of an Eddie experiment: rebuild RESULTS.md and PLAYPEN_RESULTS.md from
# everything the evaluation shards produced. Submitted by run_experiment.sh (and by
# run_playpen_eval.sh) with -hold_jid on EVERY eval job, so it runs exactly once,
# after the last of them has left the queue.
#
# WHY THIS EXISTS. Evaluation is sharded into concurrent jobs, and each shard writes
# its own partial table when it finishes -- useful for watching progress, but the
# table a shard writes describes only the checkpoints that had completed by then.
# This job writes the one that describes all of them.
#
# It is pure CPU and takes seconds: both summarizers only read the result files the
# eval jobs already wrote (stdlib Python, no torch, no GPU, no model). That is also
# why it is submitted to the default queue rather than the gpu queue.
#
# Re-runnable by hand at any time -- and cheap enough that it is the right first move
# whenever a table looks stale:
#     EXP_ENV_FILE=<EXP_DIR>/experiment.env experiments/eddie/summarize.sh
#
# --- Grid Engine options (overridable via SUMMARY_QSUB_OPTS) ------------------
#$ -N marshal_summary
#$ -cwd
#$ -l h_rt=00:30:00
#$ -pe sharedmem 1
#$ -l h_rss=4G

set -euo pipefail

: "${EXP_ENV_FILE:?not set -- submit via experiments/eddie/run_experiment.sh}"
# shellcheck disable=SC1090
source "$EXP_ENV_FILE"

echo "### experiment: ${EXP_ID} (summary) ###"
echo "dir   = $EXP_DIR"
echo "start = $(date --iso-8601=seconds)"

cd "$REPO"

# Stdlib only, so any interpreter will do; the repo venv is used when present purely
# so the version is the same one that wrote the results.
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3 || command -v python)"
[ -n "$PY" ] || { echo "ERROR: no python found to run the summarizers." >&2; exit 1; }

RC=0
"$PY" "$REPO/experiments/lib/summarize_eval.py" "$EXP_DIR" || RC=1
"$PY" "$REPO/experiments/lib/summarize_playpen_eval.py" "$EXP_DIR" || RC=1

echo ""
echo "[summary] lm-eval  : $EXP_DIR/RESULTS.md"
echo "[summary] gameplay : $EXP_DIR/PLAYPEN_RESULTS.md"
echo "[summary] end = $(date --iso-8601=seconds)"
# A missing half is not a failure of this job -- an experiment with PPEVAL_ENABLE=0
# has no gameplay results to summarize, and the summarizer says so and returns 0.
exit "$RC"

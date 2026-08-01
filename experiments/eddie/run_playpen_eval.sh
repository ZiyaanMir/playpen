#!/bin/bash
# Play the clembench games with an experiment that has ALREADY trained -- no
# retraining, no dependency, no lm-eval.
#
#   cd <repo root>
#   experiments/eddie/run_playpen_eval.sh /exports/eddie/scratch/$USER/marshal-runs/<experiment>
#
# Eddie counterpart of isambard/run_playpen_eval.sh. Use it to:
#   * add game scores to an experiment that predates this job existing,
#   * re-run a gameplay eval that failed or hit the walltime,
#   * score a subset of games, or only the last checkpoint.
#
# Any PPEVAL_* setting can be overridden for this submission only; everything else
# is reused from the experiment's own experiment.env, so the run stays comparable:
#
#   PPEVAL_CKPTS=last          experiments/eddie/run_playpen_eval.sh <EXP_DIR>
#   PPEVAL_GAMES=dond,taboo    experiments/eddie/run_playpen_eval.sh <EXP_DIR>
#   PPEVAL_SUITE=all           experiments/eddie/run_playpen_eval.sh <EXP_DIR>
#   PPEVAL_BASE=0              experiments/eddie/run_playpen_eval.sh <EXP_DIR>
#
# The training half is never touched, and existing checkpoints are never rewritten.

set -euo pipefail

EXP_DIR_ARG="${1:-}"
[ -n "$EXP_DIR_ARG" ] || {
    echo "usage: $0 <EXP_DIR>" >&2
    echo "       e.g. $0 /exports/eddie/scratch/\$USER/marshal-runs/guesswhat_Qwen3-4B_20260727-120621" >&2
    echo "       optional env: PPEVAL_SUITE PPEVAL_GAMES PPEVAL_CKPTS PPEVAL_BASE" >&2
    echo "                     PPEVAL_MAX_TOKENS PPEVAL_TEMPERATURE PPEVAL_TIMEOUT" >&2
    exit 2
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR_ARG="$(cd "$EXP_DIR_ARG" 2>/dev/null && pwd)" || {
    echo "ERROR: no such experiment directory: ${1}" >&2; exit 1; }

ENV_FILE="$EXP_DIR_ARG/experiment.env"
[ -f "$ENV_FILE" ] || {
    echo "ERROR: $EXP_DIR_ARG is not an experiment directory (no experiment.env)." >&2
    echo "       It must have been created by run_experiment.sh." >&2
    exit 1
}

# Snapshot the caller's overrides BEFORE sourcing the experiment's env file, which
# would otherwise overwrite them with the stored values -- the same precedence trap
# the presets and run_eval.sh have.
_OV_NAMES=(); _OV_VALS=()
for _v in PPEVAL_SUITE PPEVAL_GAMES PPEVAL_BASE PPEVAL_CKPTS PPEVAL_MAX_TOKENS \
          PPEVAL_TEMPERATURE PPEVAL_TIMEOUT PPEVAL_CACHE PPEVAL_HF_OFFLINE HF_HOME \
          EVAL_SHARD_SIZE; do
    if [ -n "${!_v+set}" ]; then _OV_NAMES+=("$_v"); _OV_VALS+=("${!_v}"); fi
done

# shellcheck disable=SC1090
source "$ENV_FILE"
source "$HERE/../lib/experiment.sh"
EXP_DIR="$EXP_DIR_ARG"
exp_layout

# Put the caller's overrides back on top IN THIS SHELL as well, not only in the copy
# of the env file the job reads. The submitter now does arithmetic of its own --
# EVAL_SHARD_SIZE decides how many jobs to queue -- and it has to use the same value
# the job will.
for _i in "${!_OV_NAMES[@]}"; do
    printf -v "${_OV_NAMES[$_i]}" '%s' "${_OV_VALS[$_i]}"
done
EVAL_SHARD_SIZE="${EVAL_SHARD_SIZE:-5}"

# Say up front what will be played -- cheaper to notice here than in the queue.
RUN_DIR="$(exp_find_run_dir || true)"
if [ -z "$RUN_DIR" ]; then
    echo "ERROR: no checkpoints under $TRAIN_BASE -- nothing to evaluate." >&2
    echo "       Training either never ran or produced none; check logs/train_*.err." >&2
    exit 1
fi
mapfile -t CKS < <(exp_list_checkpoints "$RUN_DIR")
[ "${#CKS[@]}" -gt 0 ] || {
    echo "ERROR: $RUN_DIR holds no complete checkpoint (none has adapter_config.json)." >&2
    exit 1
}
echo "[ppeval-only] experiment : $EXP_ID"
echo "[ppeval-only] checkpoints: ${#CKS[@]} ($(basename "${CKS[0]}") .. $(basename "${CKS[-1]}"))"

# Shard over what will ACTUALLY be played, which is what PPEVAL_CKPTS leaves. Sharding
# the full list instead would queue jobs for slices that PPEVAL_CKPTS=last has already
# emptied -- e.g. two jobs for one checkpoint, one of them with nothing to do.
# shellcheck disable=SC1090
source "$HERE/../lib/playpen_eval.sh"
pp_layout
mapfile -t PLAYED < <(exp_list_checkpoints "$RUN_DIR" | pp_filter_checkpoints)
EVAL_SHARD_TOTAL="$(exp_shard_count "${#PLAYED[@]}")"
echo "[ppeval-only] to play    : ${#PLAYED[@]}${PPEVAL_CKPTS:+ (PPEVAL_CKPTS=$PPEVAL_CKPTS)}"
echo "[ppeval-only] shards     : $EVAL_SHARD_TOTAL x $EVAL_SHARD_SIZE checkpoint(s), submitted to run concurrently"

# Overrides go in a COPY of the env file, appended at the end. The job sources that
# file top to bottom, so a later assignment wins -- which is why appending works and
# exporting here would not (the source would overwrite it).
PPEVAL_ENV_FILE="$EXP_DIR/experiment.ppeval.env"
cp "$ENV_FILE" "$PPEVAL_ENV_FILE"
{
    echo ""
    echo "# --- playpen-eval-only overrides, $(date --iso-8601=seconds) ---"
    for _i in "${!_OV_NAMES[@]}"; do
        printf '%s=%q\nexport %s\n' "${_OV_NAMES[$_i]}" "${_OV_VALS[$_i]}" "${_OV_NAMES[$_i]}"
    done
    # Not an override -- the shard COUNT this submission decided on, so each job's
    # banner can say "shard 2 of 3" rather than "shard 2 of ?".
    printf 'EVAL_SHARD_TOTAL=%q\nexport EVAL_SHARD_TOTAL\n' "$EVAL_SHARD_TOTAL"
} >> "$PPEVAL_ENV_FILE"

if [ "${#_OV_NAMES[@]}" -eq 0 ]; then
    echo "[ppeval-only] overrides  : none (reusing the experiment's stored settings)"
else
    for _i in "${!_OV_NAMES[@]}"; do
        echo "[ppeval-only] override   : ${_OV_NAMES[$_i]}=${_OV_VALS[$_i]}"
    done
fi

cd "$REPO"

# Grid Engine expands $JOB_ID inside the path itself, so it is escaped from bash.
PPEVAL_IDS=()
for SHARD in $(seq 1 "$EVAL_SHARD_TOTAL"); do
    ID="$(qsub -terse \
        -N "pp${SHARD}_${GAME}_$(basename "$MODEL")" \
        -o "$LOG_DIR/ppeval${SHARD}_${EXP_ID}_\$JOB_ID.out" \
        -e "$LOG_DIR/ppeval${SHARD}_${EXP_ID}_\$JOB_ID.err" \
        -v "EXP_ENV_FILE=$PPEVAL_ENV_FILE" \
        -v "EVAL_SHARD=$SHARD" \
        ${PPEVAL_QSUB_OPTS:-} \
        "$HERE/playpen_eval.sh" | tr -d '[:space:]')"
    PPEVAL_IDS+=("$ID")
    echo "[ppeval-only] submitted job $ID  shard $SHARD/$EVAL_SHARD_TOTAL"
done

# The complete table, once every shard is done. Each shard also writes a partial one
# as it finishes, so there is something to read in the meantime.
SUMMARY_ID="$(qsub -terse \
    -N "sum_${GAME}_$(basename "$MODEL")" \
    -hold_jid "$(IFS=,; echo "${PPEVAL_IDS[*]}")" \
    -o "$LOG_DIR/summary_${EXP_ID}_\$JOB_ID.out" \
    -e "$LOG_DIR/summary_${EXP_ID}_\$JOB_ID.err" \
    -v "EXP_ENV_FILE=$PPEVAL_ENV_FILE" \
    ${SUMMARY_QSUB_OPTS:-} \
    "$HERE/summarize.sh" | tr -d '[:space:]')"

cat <<EOF
[ppeval-only] ${#PPEVAL_IDS[@]} gameplay job(s) + summary job $SUMMARY_ID queued.
              The gameplay jobs have no dependency -- they start as soon as GPUs are
              free, and run at the same time as each other.

  tail -f $LOG_DIR/ppeval1_${EXP_ID}_${PPEVAL_IDS[0]}.out
  cat    $EXP_DIR/PLAYPEN_RESULTS.md      # complete once job $SUMMARY_ID has run
  qdel ${PPEVAL_IDS[*]} $SUMMARY_ID       # cancel
EOF

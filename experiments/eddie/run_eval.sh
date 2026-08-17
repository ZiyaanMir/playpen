#!/bin/bash
# Evaluate an experiment that has ALREADY trained -- no retraining, no dependency.
#
#   cd <repo root>
#   experiments/eddie/run_eval.sh /exports/eddie/scratch/$USER/marshal-runs/<experiment>
#
# Eddie counterpart of isambard/run_eval.sh, and the lm-eval counterpart of
# eddie/run_playpen_eval.sh. Use it to:
#   * re-run an evaluation that failed (a broken lm-eval env, OOM, walltime),
#   * score a run with different tasks or a bigger batch,
#   * add lm-eval scores to an experiment that only has gameplay results.
#
# Any eval setting can be overridden for this submission only; everything else is
# reused from the experiment's own experiment.env, so the run stays comparable:
#
#   EVAL_TASKS=logicbench     experiments/eddie/run_eval.sh <EXP_DIR>
#   EVAL_LIMIT=5              experiments/eddie/run_eval.sh <EXP_DIR>   # smoke test
#   EVAL_BATCH=32 EVAL_BASE=0 experiments/eddie/run_eval.sh <EXP_DIR>
#   LMEVAL_CONDA_ENV=lmeval2  experiments/eddie/run_eval.sh <EXP_DIR>
#
# The training half is never touched, and existing checkpoints are never rewritten.
#
# Before submitting anything it checks that the lm-eval environment can import
# lm_eval, because that is how the 2026-08-10..08-15 evals were lost: 65 experiments
# queued, every shard allocated a GPU, every shard dead in five seconds on
# ModuleNotFoundError. Set EVAL_SKIP_ENV_CHECK=1 to bypass (the backfill driver does,
# having run the check once itself).

set -euo pipefail

EXP_DIR_ARG="${1:-}"
[ -n "$EXP_DIR_ARG" ] || {
    echo "usage: $0 <EXP_DIR>" >&2
    echo "       e.g. $0 /exports/eddie/scratch/\$USER/marshal-runs/taboo_Qwen3-8B_final_20260814-072905" >&2
    echo "       optional env: EVAL_TASKS EVAL_BATCH EVAL_LIMIT EVAL_BASE EVAL_EXTRA" >&2
    echo "                     EVAL_SHARD_SIZE BASE_EVAL_CACHE LMEVAL_CONDA_ENV" >&2
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

# Snapshot the caller's overrides BEFORE sourcing the experiment's env file. Sourcing
# would otherwise overwrite them with the stored values, so `EVAL_TASKS=logicbench
# run_eval.sh <dir>` would silently run the experiment's original task list instead --
# the same precedence trap the presets have.
_OV_NAMES=(); _OV_VALS=()
for _v in EVAL_TASKS EVAL_BATCH EVAL_LIMIT EVAL_BASE EVAL_EXTRA EVAL_SHARD_SIZE \
          BASE_EVAL_CACHE LMEVAL_CONDA_ENV HF_HUB_OFFLINE HF_HOME; do
    if [ -n "${!_v+set}" ]; then _OV_NAMES+=("$_v"); _OV_VALS+=("${!_v}"); fi
done

# shellcheck disable=SC1090
source "$ENV_FILE"
source "$HERE/../lib/experiment.sh"
EXP_DIR="$EXP_DIR_ARG"
exp_layout

# Put the caller's overrides back on top IN THIS SHELL as well, not only in the copy
# of the env file the job reads. The submitter does arithmetic of its own --
# EVAL_SHARD_SIZE decides how many jobs to queue -- and it has to use the same value
# the job will.
for _i in "${!_OV_NAMES[@]}"; do
    printf -v "${_OV_NAMES[$_i]}" '%s' "${_OV_VALS[$_i]}"
done
EVAL_SHARD_SIZE="${EVAL_SHARD_SIZE:-5}"

# --- is the lm-eval environment alive? ---------------------------------------
# Cheap here, ruinous in the queue. Runs with the same LMEVAL_CONDA_ENV the jobs will
# see, so an override is checked rather than the default.
if [ "${EVAL_SKIP_ENV_CHECK:-0}" != "1" ]; then
    LMEVAL_CONDA_ENV="${LMEVAL_CONDA_ENV:-}" \
        bash "$HERE/../lib/check_lmeval_env.sh" --cluster eddie || {
        echo "" >&2
        echo "ERROR: not submitting -- the eval environment is broken (see above)." >&2
        echo "       Every job would burn a GPU allocation to reproduce that error." >&2
        echo "       EVAL_SKIP_ENV_CHECK=1 bypasses this if you know better." >&2
        exit 1
    }
    echo ""
fi

# Say up front what will be scored -- cheaper to notice here than in the queue.
RUN_DIR="$(exp_find_run_dir || true)"
if [ -z "$RUN_DIR" ]; then
    echo "ERROR: no checkpoints under $TRAIN_BASE -- nothing to evaluate." >&2
    echo "       Training either never ran or produced none; check logs/train*.err." >&2
    exit 1
fi
mapfile -t CKS < <(exp_list_checkpoints "$RUN_DIR")
[ "${#CKS[@]}" -gt 0 ] || {
    echo "ERROR: $RUN_DIR holds no complete checkpoint (none has adapter_config.json)." >&2
    exit 1
}
echo "[eval-only] experiment : $EXP_ID"
echo "[eval-only] checkpoints: ${#CKS[@]} ($(basename "${CKS[0]}") .. $(basename "${CKS[-1]}"))"

# Training has already finished here, so the checkpoint count is EXACT rather than
# predicted from the schedule -- one shard per EVAL_SHARD_SIZE of them, no more.
EVAL_SHARD_TOTAL="$(exp_shard_count "${#CKS[@]}")"
echo "[eval-only] shards     : $EVAL_SHARD_TOTAL x $EVAL_SHARD_SIZE checkpoint(s), submitted to run concurrently"

# Overrides go in a COPY of the env file, appended at the end. eval.sh sources that
# file top to bottom, so a later assignment wins -- which is why appending works and
# why exporting the variable here would not (the source would overwrite it).
EVAL_ENV_FILE="$EXP_DIR/experiment.eval.env"
cp "$ENV_FILE" "$EVAL_ENV_FILE"
{
    echo ""
    echo "# --- eval-only overrides, $(date --iso-8601=seconds) ---"
    for _i in "${!_OV_NAMES[@]}"; do
        printf '%s=%q\nexport %s\n' "${_OV_NAMES[$_i]}" "${_OV_VALS[$_i]}" "${_OV_NAMES[$_i]}"
    done
    # Not an override -- the shard COUNT this submission decided on, so each job's
    # banner can say "shard 2 of 3" rather than "shard 2 of ?".
    printf 'EVAL_SHARD_TOTAL=%q\nexport EVAL_SHARD_TOTAL\n' "$EVAL_SHARD_TOTAL"
} >> "$EVAL_ENV_FILE"

if [ "${#_OV_NAMES[@]}" -eq 0 ]; then
    echo "[eval-only] overrides  : none (reusing the experiment's stored settings)"
else
    for _i in "${!_OV_NAMES[@]}"; do
        echo "[eval-only] override   : ${_OV_NAMES[$_i]}=${_OV_VALS[$_i]}"
    done
fi

cd "$REPO"

# Grid Engine expands $JOB_ID inside the path itself, so it is escaped from bash.
# No -hold_jid: training is done, so these start as soon as GPUs are free.
EVAL_IDS=()
for SHARD in $(seq 1 "$EVAL_SHARD_TOTAL"); do
    ID="$(qsub -terse \
        -N "ev${SHARD}_${GAME}_$(basename "$MODEL")" \
        -o "$LOG_DIR/eval${SHARD}_${EXP_ID}_\$JOB_ID.out" \
        -e "$LOG_DIR/eval${SHARD}_${EXP_ID}_\$JOB_ID.err" \
        -v "EXP_ENV_FILE=$EVAL_ENV_FILE" \
        -v "EVAL_SHARD=$SHARD" \
        ${EVAL_QSUB_OPTS:-} \
        "$HERE/eval.sh" | tr -d '[:space:]')"
    EVAL_IDS+=("$ID")
    echo "[eval-only] submitted job $ID  shard $SHARD/$EVAL_SHARD_TOTAL"
done

# The complete table, once every shard is done. Each shard also writes a partial one
# as it finishes, so there is something to read in the meantime. summarize.sh rebuilds
# BOTH tables, so an experiment that already has gameplay results keeps them.
SUMMARY_ID="$(qsub -terse \
    -N "sum_${GAME}_$(basename "$MODEL")" \
    -hold_jid "$(IFS=,; echo "${EVAL_IDS[*]}")" \
    -o "$LOG_DIR/summary_${EXP_ID}_\$JOB_ID.out" \
    -e "$LOG_DIR/summary_${EXP_ID}_\$JOB_ID.err" \
    -v "EXP_ENV_FILE=$EVAL_ENV_FILE" \
    ${SUMMARY_QSUB_OPTS:-} \
    "$HERE/summarize.sh" | tr -d '[:space:]')"

# Add these ids to the manifest, next to whatever the original submission recorded --
# a re-run of the evaluation is another entry in that history, not a replacement.
# Never fatal: the jobs are already queued. See exp_record_jobs.
exp_record_jobs \
    --eval "${EVAL_IDS[@]}" \
    --summary "$SUMMARY_ID" \
    --shard-total "$EVAL_SHARD_TOTAL" \
    --env-file "$EVAL_ENV_FILE"

cat <<EOF
[eval-only] ${#EVAL_IDS[@]} eval job(s) + summary job $SUMMARY_ID queued.
            The eval jobs have no dependency -- they start as soon as GPUs are free,
            and run at the same time as each other.

  tail -f $LOG_DIR/eval1_${EXP_ID}_${EVAL_IDS[0]}.out
  cat    $EXP_DIR/RESULTS.md      # complete once job $SUMMARY_ID has run
  qdel ${EVAL_IDS[*]} $SUMMARY_ID   # cancel
EOF

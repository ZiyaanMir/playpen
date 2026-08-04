#!/bin/bash
# Continue an experiment that stopped short -- same directory, same settings, same
# W&B run, picking up from its last checkpoint.
#
#   module load brics/userenv
#   cd <repo root>
#   experiments/isambard/resume_experiment.sh $PROJECTDIR/$USER/marshal-runs/<experiment>
#
# Use it when a training job hit the 24 h walltime, died on a node fault, or was
# cancelled -- anything that left checkpoints behind but did not reach MAX_STEPS.
# It re-queues only the training segments that are still ahead of what is on disk,
# then the full evaluation chain behind them.
#
# DO NOT use run_experiment.sh for this: that creates a NEW experiment directory and
# starts again at step 0, and the half-trained one is left to be explained later.
#
# The run stays the run it was. MAX_STEPS, the learning rate and every MARSHAL flag
# come from the experiment's own experiment.env, so the LR at step 700 is the same
# number it would have been had the job never died. Only the job layout may change:
#
#   TRAIN_SEGMENTS=3   experiments/isambard/resume_experiment.sh <EXP_DIR>
#       re-plan the remaining steps as shorter jobs (the usual reason to be here:
#       the run died BECAUSE one job could not fit the remaining work in 24 h)
#   TRAIN_SBATCH_OPTS='--time=08:00:00' experiments/isambard/resume_experiment.sh <EXP_DIR>
#       ...and book credit for the hours those shorter jobs actually need
#   PPEVAL_ENABLE=0    experiments/isambard/resume_experiment.sh <EXP_DIR>
#       skip the gameplay eval on this pass
#
# If training already reached MAX_STEPS there is nothing to resume; the script says
# so and points at run_eval.sh / run_playpen_eval.sh for the evaluation half.

set -euo pipefail

EXP_DIR_ARG="${1:-}"
[ -n "$EXP_DIR_ARG" ] || {
    echo "usage: $0 <EXP_DIR>" >&2
    echo "       e.g. $0 \$PROJECTDIR/\$USER/marshal-runs/guesswhat_Qwen3-4B_20260727-120621" >&2
    echo "       optional env: TRAIN_SEGMENTS SEGMENT_STEPS TRAIN_SBATCH_OPTS PPEVAL_ENABLE" >&2
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
# exp_load_preset and run_playpen_eval.sh guard against.
#
# MAX_STEPS is deliberately NOT in this list. It is the horizon HF built the LR
# scheduler from, and scheduler.pt restores only the step counter, never the decay
# curve -- so resuming with a different MAX_STEPS would give the rest of the run a
# different learning-rate schedule from its first half, in one continuous set of
# checkpoints that says nothing about it. Extending a run is a new experiment.
_OV_NAMES=(); _OV_VALS=()
for _v in TRAIN_SEGMENTS SEGMENT_STEPS EVAL_SHARD_SIZE PPEVAL_ENABLE PPEVAL_SUITE \
          PPEVAL_GAMES PPEVAL_BASE PPEVAL_CKPTS PPEVAL_SERIAL EVAL_BASE; do
    if [ -n "${!_v+set}" ]; then _OV_NAMES+=("$_v"); _OV_VALS+=("${!_v}"); fi
done
if [ -n "${MAX_STEPS+set}" ]; then
    echo "ERROR: MAX_STEPS cannot be changed on resume (you set MAX_STEPS=$MAX_STEPS)." >&2
    echo "       It is the horizon the LR schedule was built for; changing it now would" >&2
    echo "       give the second half of this run a different schedule from its first," >&2
    echo "       inside one set of checkpoints. Start a new experiment instead." >&2
    exit 2
fi

# `set -a` so every name in the env file is EXPORTED, not merely assigned.
#
# The file is written as bare `NAME=value` assignments. train.sh reads them as shell
# variables, so it never noticed -- but check_resume_config.py below is a SUBPROCESS,
# and a subprocess sees only exported names. Without this it resolves the config with
# LP_COEF, EXTRA_TRAIN_ARGS, TR_* and the rest all missing, reports five fields of
# "drift" that are really its own blindness, and teaches you to reach for
# RESUME_FORCE=1 -- which would defeat the whole check.
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
source "$HERE/../lib/experiment.sh"
EXP_DIR="$EXP_DIR_ARG"
exp_layout

for _i in "${!_OV_NAMES[@]}"; do
    printf -v "${_OV_NAMES[$_i]}" '%s' "${_OV_VALS[$_i]}"
done
EVAL_SHARD_SIZE="${EVAL_SHARD_SIZE:-5}"

# --- what is left to do -------------------------------------------------------
# Re-plan first (the caller may have asked for a different segment layout), then ask
# where the checkpoints on disk leave that plan.
#
# exp_segment_override before exp_plan_segments: TRAIN_SEGMENTS=5 on the command line
# has to beat the SEGMENT_STEPS the ORIGINAL submission stored in experiment.env,
# which exp_plan_segments would otherwise prefer as the more specific of the two.
exp_segment_override ${_OV_NAMES[@]+"${_OV_NAMES[@]}"}
exp_plan_segments || exit 1
echo "[resume] experiment : $EXP_ID"
echo "[resume] plan       : ${MAX_STEPS} steps total, ${TRAIN_SEGMENTS} segment(s) of <= ${SEGMENT_STEPS}"

if ! exp_resume_plan; then
    echo ""
    echo "[resume] nothing to train: ${RESUME_DONE_STEPS} of ${MAX_STEPS} steps are already"
    echo "         on disk, which is the whole run."
    echo ""
    echo "         For the evaluation half, use:"
    echo "           experiments/isambard/run_eval.sh         $EXP_DIR"
    echo "           experiments/isambard/run_playpen_eval.sh $EXP_DIR"
    exit 0
fi

if [ -n "$RESUME_CKPT" ]; then
    echo "[resume] from       : $(basename "$RESUME_CKPT") (global step ${RESUME_DONE_STEPS})"
    # Segment 1 defaults to 'auto' ("start over if there is nothing"), which is right
    # for a fresh experiment and wrong here: we have just proved there IS something,
    # so a resume that silently started at step 0 would be a bug we could still catch.
    RESUME_FROM=latest
else
    echo "[resume] from       : nothing on disk -- this restarts the run at step 0"
    echo "[resume]              (no checkpoint under $TRAIN_BASE has both a trainer_state.json"
    echo "[resume]              and an adapter; training died before its first save)"
    RESUME_FROM=auto
fi
LAST_SEGMENT="$TRAIN_SEGMENTS"
echo "[resume] segments   : ${RESUME_FIRST_SEGMENT}..${LAST_SEGMENT} of ${TRAIN_SEGMENTS} still to run"

# One W&B run across the original jobs and these. Forced, because the original may
# have been a single unsegmented job with no chain id -- the id is derived from
# EXP_ID, so it is the same one either way.
exp_wandb_chain_setup force

# --- pin the algorithm config to what this run actually started with ----------
# experiment.env stores MARSHAL_CONFIG as a PATH into the repo, not as content, and
# that shared YAML is edited between runs -- it is the file every ablation arm is
# varied from. So the stored path says nothing about what THIS run used, and resuming
# through it silently hands the remaining segments whatever the YAML happens to say
# today.
#
# Not hypothetical: taboo_Qwen3-4B_turnrew_20260730-231934 resumed against the current
# shared YAML would flip turn_level_rewards true->false and whiten_rewards true->false,
# half way through the run, with nothing on disk recording it.
#
# write_manifest.py froze a copy inside the experiment at submit time for exactly this
# reason. Point at that instead.
if [ -f "$EXP_DIR/marshal_config.yaml" ]; then
    MARSHAL_CONFIG="$EXP_DIR/marshal_config.yaml"
    export MARSHAL_CONFIG
    echo "[resume] config     : $MARSHAL_CONFIG (this run's frozen copy, not the shared YAML)"
else
    echo "[resume] WARNING: no frozen marshal_config.yaml in the experiment directory," >&2
    echo "[resume]          so this resume falls back to $MARSHAL_CONFIG as it reads" >&2
    echo "[resume]          TODAY. If that file has been edited since, the remaining" >&2
    echo "[resume]          segments train under a different config. The check below" >&2
    echo "[resume]          will catch it." >&2
fi

# --- and prove nothing else drifted ------------------------------------------
# Compares every resolved MARSHAL setting against what manifest.json recorded at first
# submission, and refuses if any of them moved. Catches the case the pin above cannot:
# a value that reached the ORIGINAL job through the environment (Isambard's
# --export=ALL carries TR_ENABLE without it ever being written to experiment.env) and
# has no route into this one. See experiments/lib/check_resume_config.py.
if [ "${RESUME_FORCE:-0}" != "1" ]; then
    "$REPO/.venv/bin/python" "$EXP_ROOT_DIR/lib/check_resume_config.py" "$EXP_DIR" || exit 1
else
    echo "[config-check] SKIPPED by RESUME_FORCE=1 -- this run may be trained two ways."
fi

# --- the env file these jobs read --------------------------------------------
# A COPY, appended to. The job sources it top to bottom so a later assignment wins,
# and the original experiment.env is left as the record of what was first submitted.
#
# Written from THIS shell's values, which already have the caller's overrides applied
# on top of the stored ones -- so the list is fixed rather than derived from what the
# caller happened to set, and every name that can differ between the original
# submission and this one appears exactly once.
RESUME_ENV_FILE="$EXP_DIR/experiment.resume.env"
cp "$ENV_FILE" "$RESUME_ENV_FILE"
{
    echo ""
    echo "# --- resume overrides, $(date --iso-8601=seconds) ---"
    for _v in TRAIN_SEGMENTS SEGMENT_STEPS RESUME_FROM MARSHAL_CONFIG \
              LP_MAX_LEN LP_COEF UNIQUE_POOL EXTRA_TRAIN_ARGS \
              TR_ENABLE TR_SOURCE TR_SCALE TR_BUDGET TR_COMPONENTS \
              EVAL_SHARD_SIZE WB_ID WB_RESUME \
              EVAL_BASE PPEVAL_ENABLE PPEVAL_SUITE PPEVAL_GAMES PPEVAL_BASE \
              PPEVAL_CKPTS PPEVAL_SERIAL; do
        printf '%s=%q\nexport %s\n' "$_v" "${!_v-}" "$_v"
    done
} >> "$RESUME_ENV_FILE"

{
    echo ""
    echo "-- resumed $(date --iso-8601=seconds) ------------------------------"
    echo "  from                         ${RESUME_CKPT:-<nothing on disk>}"
    echo "  steps already done           ${RESUME_DONE_STEPS} of ${MAX_STEPS}"
    echo "  segments re-queued           ${RESUME_FIRST_SEGMENT}..${LAST_SEGMENT} of ${TRAIN_SEGMENTS}"
} >> "$EXP_DIR/manifest.txt"

cd "$REPO"

# --- submit -------------------------------------------------------------------
# Identical to run_experiment.sh's chain, except it starts at RESUME_FIRST_SEGMENT.
TRAIN_IDS=()
PREV_ID=""
for SEGMENT in $(seq "$RESUME_FIRST_SEGMENT" "$LAST_SEGMENT"); do
    DEP_OPTS=()
    [ -n "$PREV_ID" ] && DEP_OPTS=( --dependency="afterany:$PREV_ID" )
    ID="$(sbatch --parsable \
        --job-name="tr${SEGMENT}_${GAME}_$(basename "$MODEL")" \
        --output="$LOG_DIR/train${SEGMENT}_${EXP_ID}_%j.out" \
        --error="$LOG_DIR/train${SEGMENT}_${EXP_ID}_%j.err" \
        --export="ALL,EXP_ENV_FILE=$RESUME_ENV_FILE,TRAIN_SEGMENT=$SEGMENT" \
        "${DEP_OPTS[@]}" \
        ${TRAIN_SBATCH_OPTS:-} \
        "$HERE/train.sh")"
    TRAIN_IDS+=("$ID")
    echo "[submit] training  job $ID  segment $SEGMENT/$TRAIN_SEGMENTS" \
         "(-> step $(exp_segment_stop_at "$SEGMENT")${PREV_ID:+, held until $PREV_ID finishes})"
    PREV_ID="$ID"
done
TRAIN_ID="$PREV_ID"

# --- evaluation, held on the last training segment ----------------------------
# Sized against the FULL checkpoint count, not the remaining one: the eval jobs score
# every checkpoint in train/, including the ones the original submission produced.
#
# exp_expected_checkpoints prefers what is on disk, which is exactly wrong here -- at
# resume time the disk holds only the checkpoints written so far, so trusting it would
# under-shard and leave the ones this resume is about to write unscored. Take the
# larger of the two: the schedule's prediction for what is still coming, the disk for
# any off-cadence checkpoint a segment boundary already left behind.
N_CKPTS_DISK="$(exp_expected_checkpoints)"
N_CKPTS_PLAN=1
if [ "${SAVE_STEPS:-0}" -gt 0 ] 2>/dev/null; then
    N_CKPTS_PLAN=$(( MAX_STEPS / SAVE_STEPS ))
    [ "$N_CKPTS_PLAN" -ge 1 ] || N_CKPTS_PLAN=1
fi
N_CKPTS="$N_CKPTS_PLAN"
[ "$N_CKPTS_DISK" -le "$N_CKPTS" ] || N_CKPTS="$N_CKPTS_DISK"
EVAL_SHARD_TOTAL="$(exp_shard_count "$N_CKPTS")"
export EVAL_SHARD_TOTAL
printf 'EVAL_SHARD_TOTAL=%q\nexport EVAL_SHARD_TOTAL\n' "$EVAL_SHARD_TOTAL" >> "$RESUME_ENV_FILE"
echo "[plan] ${EVAL_SHARD_TOTAL} eval shard(s) of ${EVAL_SHARD_SIZE} checkpoint(s) each"

ALL_EVAL_IDS=()
EVAL_IDS=()
for SHARD in $(seq 1 "$EVAL_SHARD_TOTAL"); do
    ID="$(sbatch --parsable \
        --job-name="ev${SHARD}_${GAME}_$(basename "$MODEL")" \
        --dependency="afterany:$TRAIN_ID" \
        --output="$LOG_DIR/eval${SHARD}_${EXP_ID}_%j.out" \
        --error="$LOG_DIR/eval${SHARD}_${EXP_ID}_%j.err" \
        --export="ALL,EXP_ENV_FILE=$RESUME_ENV_FILE,EVAL_SHARD=$SHARD" \
        ${EVAL_SBATCH_OPTS:-} \
        "$HERE/eval.sh")"
    EVAL_IDS+=("$ID"); ALL_EVAL_IDS+=("$ID")
    echo "[submit] eval      job $ID  shard $SHARD/$EVAL_SHARD_TOTAL (held until $TRAIN_ID finishes)"
done

if [ "${PPEVAL_ENABLE:-1}" = "1" ]; then
    if [ "${PPEVAL_SERIAL:-0}" = "1" ]; then
        PP_DEP="afterany:$(IFS=:; echo "${EVAL_IDS[*]}")"
    else
        PP_DEP="afterany:$TRAIN_ID"
    fi
    for SHARD in $(seq 1 "$EVAL_SHARD_TOTAL"); do
        ID="$(sbatch --parsable \
            --job-name="pp${SHARD}_${GAME}_$(basename "$MODEL")" \
            --dependency="$PP_DEP" \
            --output="$LOG_DIR/ppeval${SHARD}_${EXP_ID}_%j.out" \
            --error="$LOG_DIR/ppeval${SHARD}_${EXP_ID}_%j.err" \
            --export="ALL,EXP_ENV_FILE=$RESUME_ENV_FILE,EVAL_SHARD=$SHARD" \
            ${PPEVAL_SBATCH_OPTS:-} \
            "$HERE/playpen_eval.sh")"
        ALL_EVAL_IDS+=("$ID")
        echo "[submit] playpen   job $ID  shard $SHARD/$EVAL_SHARD_TOTAL ($PP_DEP)"
    done
else
    echo "[submit] playpen   skipped (PPEVAL_ENABLE=0)"
fi

SUMMARY_ID="$(sbatch --parsable \
    --job-name="sum_${GAME}_$(basename "$MODEL")" \
    --dependency="afterany:$(IFS=:; echo "${ALL_EVAL_IDS[*]}")" \
    --output="$LOG_DIR/summary_${EXP_ID}_%j.out" \
    --error="$LOG_DIR/summary_${EXP_ID}_%j.err" \
    --export="ALL,EXP_ENV_FILE=$RESUME_ENV_FILE" \
    ${SUMMARY_SBATCH_OPTS:-} \
    "$HERE/summarize.sh")"
echo "[submit] summary   job $SUMMARY_ID  (held until all ${#ALL_EVAL_IDS[@]} eval job(s) finish)"

cat <<EOF

resumed    : $EXP_ID
directory  : $EXP_DIR
from       : step ${RESUME_DONE_STEPS} of ${MAX_STEPS}

jobs       : ${#TRAIN_IDS[@]} train (segments ${RESUME_FIRST_SEGMENT}..${LAST_SEGMENT}) + ${#ALL_EVAL_IDS[@]} eval + 1 summary

  tail -f $LOG_DIR/train${RESUME_FIRST_SEGMENT}_${EXP_ID}_${TRAIN_IDS[0]}.out
  squeue --me
  scancel ${TRAIN_IDS[*]} ${ALL_EVAL_IDS[*]} $SUMMARY_ID  # cancel
EOF

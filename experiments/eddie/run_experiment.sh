#!/bin/bash
# Submit a full train -> evaluate experiment on Eddie (Grid Engine). RUN THIS, not
# the train.sh / eval.sh job scripts -- those expect $EXP_DIR to already exist.
#
#   cd <repo root>
#   experiments/eddie/run_experiment.sh dond
#   EXP_TAG=lp_off EXTRA_TRAIN_ARGS=--no-length-penalty \
#     experiments/eddie/run_experiment.sh dond
#   MODEL=Qwen/Qwen3-0.6B MAX_STEPS=20 SAVE_STEPS=10 EXP_TAG=smoke \
#     experiments/eddie/run_experiment.sh guesswhat
#
# Creates the experiment directory, writes the manifest, then queues:
#   1. training     (TRAIN_SEGMENTS chained jobs, each resuming the last)
#   2. lm-eval      (logiglue/logicbench -- did reasoning transfer?)   } N shards each,
#   3. playpen eval (clembench gameplay -- did it learn to play?)      } held on the
#                                                                     } LAST train job,
#                                                                     } run CONCURRENTLY
#   4. summary      (the complete RESULTS.md / PLAYPEN_RESULTS.md, held on 2 and 3)
#
# Both evaluations are split into shards of EVAL_SHARD_SIZE (5) checkpoints so no
# single job has to fit ten checkpoints of generation into one walltime. Prints the
# paths and job ids; nothing else to remember.
#
# Job 3 off:  PPEVAL_ENABLE=0 experiments/eddie/run_experiment.sh dond
#
# TRAINING TOO LONG FOR ONE WALLTIME (48 h is Eddie's gpu-queue cap and cannot be
# raised): split it. MAX_STEPS stays the total; only the job count changes.
#
#   TRAIN_SEGMENTS=3 experiments/eddie/run_experiment.sh guesswhat
#   TRAIN_SEGMENTS=3 TRAIN_QSUB_OPTS='-l h_rt=16:00:00' \
#     experiments/eddie/run_experiment.sh guesswhat   # and schedule sooner too
#
# To rescue a run that already died, do not re-submit it here -- that starts a new
# experiment directory. Use  experiments/eddie/resume_experiment.sh <EXP_DIR>.
#
# Any preset value can be overridden by exporting it first (see
# experiments/presets/<game>.env for the full list). ALWAYS set EXP_TAG when a run
# differs from the preset -- it is what makes the directory name self-describing.

set -euo pipefail

GAME_ARG="${1:-}"
[ -n "$GAME_ARG" ] || {
    echo "usage: $0 <game>   (any name in experiments/presets/, e.g. dond guesswhat taboo)" >&2
    echo "       optional env: MODEL EXP_TAG MAX_STEPS EVAL_TASKS EXTRA_TRAIN_ARGS ..." >&2
    exit 2
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$HERE")/lib/experiment.sh"

export EXP_CLUSTER=eddie
exp_load_preset "$GAME_ARG"
exp_make_id

# Same scratch/runs resolution as slurm_eddie/_common.sh, which the job scripts
# source. Duplicated here (not sourced) because _common.sh activates the training
# venv and remaps the GPU -- work that belongs in the job, not on a login node.
SCRATCH="${SCRATCH:-/exports/eddie/scratch/$USER}"
[ -d "$SCRATCH" ] || { echo "ERROR: \$SCRATCH does not exist: $SCRATCH" >&2; exit 1; }
[ -w "$SCRATCH" ] || { echo "ERROR: \$SCRATCH is not writable: $SCRATCH" >&2; exit 1; }
export MARSHAL_RUNS="${MARSHAL_RUNS:-$SCRATCH/marshal-runs}"

export EXP_DIR="$MARSHAL_RUNS/$EXP_ID"
exp_layout
mkdir -p "$EXP_DIR" "$TRAIN_BASE" "$EVAL_DIR" "$LOG_DIR"

# --- how many evaluation jobs -------------------------------------------------
# Training has not run yet, so the checkpoint count is PREDICTED from the schedule
# (MAX_STEPS/SAVE_STEPS); see exp_expected_checkpoints for why over-predicting is
# safe and under-predicting is not. Both evaluations are then split into that many
# shards of EVAL_SHARD_SIZE checkpoints, and all of them are queued now.
N_CKPTS="$(exp_expected_checkpoints)"
EVAL_SHARD_TOTAL="$(exp_shard_count "$N_CKPTS")"
export EVAL_SHARD_TOTAL
echo "[plan] ${MAX_STEPS} steps / ${SAVE_STEPS} save-steps => ~${N_CKPTS} checkpoints"
echo "[plan] ${EVAL_SHARD_TOTAL} eval shard(s) of ${EVAL_SHARD_SIZE} checkpoint(s) each"

# --- how many training jobs ---------------------------------------------------
# Resolves TRAIN_SEGMENTS/SEGMENT_STEPS into a consistent pair (either one may be the
# knob the caller set) and reports the boundaries. MAX_STEPS is unchanged by this: the
# segments partition it, they do not extend it.
exp_plan_segments || exit 1
if [ "$TRAIN_SEGMENTS" -gt 1 ]; then
    STOPS=""
    for S in $(seq 1 "$TRAIN_SEGMENTS"); do STOPS="$STOPS $(exp_segment_stop_at "$S")"; done
    echo "[plan] ${TRAIN_SEGMENTS} chained training job(s) of <= ${SEGMENT_STEPS} steps, ending at:${STOPS}"
    echo "[plan] each resumes the previous one's last checkpoint; --max-steps stays"
    echo "[plan] ${MAX_STEPS} in all of them, so the LR schedule matches a single-job run"
    exp_wandb_chain_setup
else
    echo "[plan] 1 training job (set TRAIN_SEGMENTS=N to split it across N chained jobs)"
fi

# --- manifest, written before submission -------------------------------------
# So a job that dies in the queue still leaves a record of what it was going to be.
# Uses the training venv because it imports playpen.marshal.config (stdlib + PyYAML;
# no torch, so this is fast even on a login node).
"$REPO/.venv/bin/python" "$EXP_ROOT_DIR/lib/write_manifest.py" "$EXP_DIR"

# --- everything the job scripts need, in one place ---------------------------
# Grid Engine's -v takes a comma-separated list, and several of our values contain
# commas or spaces (EXTRA_TRAIN_ARGS, EVAL_TASKS). Writing them to a file the job
# sources instead sidesteps -v quoting entirely.
ENV_FILE="$EXP_DIR/experiment.env"
{
    echo "# Generated by experiments/eddie/run_experiment.sh at $(date --iso-8601=seconds)."
    echo "# Sourced by train.sh and eval.sh. Edit only to re-run by hand."
    for var in EXP_ID EXP_DIR EXP_TAG EXP_CLUSTER REPO MARSHAL_RUNS \
               MODEL GAME MARSHAL_CONFIG \
               NUM_GENERATIONS PER_DEVICE_BATCH GRAD_ACCUM MAX_STEPS SAVE_STEPS \
               LEARNING_RATE KL_BETA MAX_COMPLETION_LENGTH MAX_TURNS GRAD_CKPT \
               VLLM_UTIL VLLM_MAX_MODEL_LEN UNIQUE_POOL GRPO_LOSS EXTRA_TRAIN_ARGS \
               LP_PER_TOKEN LP_BUDGET LP_MAX_LEN LP_COEF \
               TRAIN_SEGMENTS SEGMENT_STEPS RESUME_FROM \
               WB_ENABLE WB_PROJECT WB_ENTITY WB_GROUP WB_TAGS WB_MODE WB_ID WB_RESUME \
               EVAL_TASKS EVAL_BATCH EVAL_BASE EVAL_LIMIT EVAL_EXTRA BASE_EVAL_CACHE LMEVAL_CONDA_ENV \
               EVAL_SHARD_SIZE EVAL_SHARD_TOTAL \
               PPEVAL_ENABLE PPEVAL_SUITE PPEVAL_GAMES PPEVAL_BASE PPEVAL_CKPTS \
               PPEVAL_MAX_TOKENS PPEVAL_TEMPERATURE PPEVAL_TIMEOUT PPEVAL_CACHE PPEVAL_HF_OFFLINE \
               PPEVAL_SERIAL; do
        printf '%s=%q\n' "$var" "${!var-}"
    done
} > "$ENV_FILE"

cd "$REPO"

# --- submit ------------------------------------------------------------------
# -o/-e are given on the command line rather than as "#$" directives because the
# log destination is only known now; command-line options win over directives.
# Grid Engine expands $JOB_ID inside the path itself, so it is escaped from bash.
# TRAIN_SEGMENTS jobs in SEQUENCE, each held on the previous one and continuing from
# its last checkpoint. Which slice a job owns comes from TRAIN_SEGMENT, passed as its
# own -v exactly like EVAL_SHARD below -- the env file is shared by every segment, so
# the index cannot live in it.
#
# -hold_jid, not a success condition: Grid Engine has no afterok, and we would not want
# one here anyway. A segment killed at the walltime exits non-zero having written
# checkpoint-350, and the whole point is for the next segment to pick that up. What
# stops a broken chain from quietly retraining from scratch is on the other side:
# segments 2+ pass --resume-from-checkpoint latest, which ERRORS when train/ holds no
# resumable checkpoint (see exp_segment_resume_spec).
TRAIN_IDS=()
PREV_ID=""
for SEGMENT in $(seq 1 "$TRAIN_SEGMENTS"); do
    HOLD_OPTS=()
    [ -n "$PREV_ID" ] && HOLD_OPTS=( -hold_jid "$PREV_ID" )
    ID="$(qsub -terse \
        -N "tr${SEGMENT}_${GAME}_$(basename "$MODEL")" \
        -o "$LOG_DIR/train${SEGMENT}_${EXP_ID}_\$JOB_ID.out" \
        -e "$LOG_DIR/train${SEGMENT}_${EXP_ID}_\$JOB_ID.err" \
        -v "EXP_ENV_FILE=$ENV_FILE" \
        -v "TRAIN_SEGMENT=$SEGMENT" \
        "${HOLD_OPTS[@]}" \
        ${TRAIN_QSUB_OPTS:-} \
        "$HERE/train.sh" | tr -d '[:space:]')"
    TRAIN_IDS+=("$ID")
    if [ "$TRAIN_SEGMENTS" -gt 1 ]; then
        echo "[submit] training  job $ID  segment $SEGMENT/$TRAIN_SEGMENTS" \
             "(-> step $(exp_segment_stop_at "$SEGMENT")${PREV_ID:+, held until $PREV_ID finishes})"
    else
        echo "[submit] training  job $ID"
    fi
    PREV_ID="$ID"
done
# Everything downstream waits on the LAST segment: that is the job after which the run
# is complete. Named TRAIN_ID so the eval/summary blocks below are unchanged.
TRAIN_ID="$PREV_ID"

# --- the evaluation shards ----------------------------------------------------
# One job per shard of EVAL_SHARD_SIZE checkpoints, for BOTH evaluations, all held on
# the TRAINING job -- so they start together the moment training leaves the queue and
# run concurrently. Each shard learns which slice is its own from EVAL_SHARD, passed
# as its own -v (the env file is shared by every shard, so the index cannot live in
# it) and never written by the sourced file, so it survives.
#
# -hold_jid waits for the held-on job to LEAVE THE QUEUE, whatever its exit status
# (Grid Engine has no afterok equivalent). That is what we want: a run killed at the
# walltime after writing checkpoint-150 should still be evaluated. Every eval job
# handles the genuinely-empty case by exiting cleanly with a message.
#
# THE COST OF RUNNING THEM TOGETHER: an experiment now asks for up to
# 2 x EVAL_SHARD_TOTAL GPUs at once instead of one. That is the trade -- wall-clock
# for queue width. PPEVAL_SERIAL=1 restores the old chain (gameplay held behind
# lm-eval, one GPU per experiment at a time) if the queue cannot take it.
ALL_EVAL_IDS=()

EVAL_IDS=()
for SHARD in $(seq 1 "$EVAL_SHARD_TOTAL"); do
    ID="$(qsub -terse \
        -N "ev${SHARD}_${GAME}_$(basename "$MODEL")" \
        -hold_jid "$TRAIN_ID" \
        -o "$LOG_DIR/eval${SHARD}_${EXP_ID}_\$JOB_ID.out" \
        -e "$LOG_DIR/eval${SHARD}_${EXP_ID}_\$JOB_ID.err" \
        -v "EXP_ENV_FILE=$ENV_FILE" \
        -v "EVAL_SHARD=$SHARD" \
        ${EVAL_QSUB_OPTS:-} \
        "$HERE/eval.sh" | tr -d '[:space:]')"
    EVAL_IDS+=("$ID")
    ALL_EVAL_IDS+=("$ID")
    echo "[submit] eval      job $ID  shard $SHARD/$EVAL_SHARD_TOTAL (held until $TRAIN_ID finishes)"
done

# The gameplay shards. Held on the training job like the lm-eval ones, so all four
# run side by side -- unless PPEVAL_SERIAL=1, which holds them behind every lm-eval
# shard instead.
PPEVAL_IDS=()
if [ "${PPEVAL_ENABLE:-1}" = "1" ]; then
    if [ "${PPEVAL_SERIAL:-0}" = "1" ]; then
        PP_HOLD="$(IFS=,; echo "${EVAL_IDS[*]}")"
        PP_HOLD_DESC="held until the lm-eval shards finish"
    else
        PP_HOLD="$TRAIN_ID"
        PP_HOLD_DESC="held until $TRAIN_ID finishes"
    fi
    for SHARD in $(seq 1 "$EVAL_SHARD_TOTAL"); do
        ID="$(qsub -terse \
            -N "pp${SHARD}_${GAME}_$(basename "$MODEL")" \
            -hold_jid "$PP_HOLD" \
            -o "$LOG_DIR/ppeval${SHARD}_${EXP_ID}_\$JOB_ID.out" \
            -e "$LOG_DIR/ppeval${SHARD}_${EXP_ID}_\$JOB_ID.err" \
            -v "EXP_ENV_FILE=$ENV_FILE" \
            -v "EVAL_SHARD=$SHARD" \
            ${PPEVAL_QSUB_OPTS:-} \
            "$HERE/playpen_eval.sh" | tr -d '[:space:]')"
        PPEVAL_IDS+=("$ID")
        ALL_EVAL_IDS+=("$ID")
        echo "[submit] playpen   job $ID  shard $SHARD/$EVAL_SHARD_TOTAL ($PP_HOLD_DESC)"
    done
else
    echo "[submit] playpen   skipped (PPEVAL_ENABLE=0)"
fi

# --- the final tables ---------------------------------------------------------
# Every shard writes a partial RESULTS.md / PLAYPEN_RESULTS.md when it finishes, so
# there is always something to read; this job writes the COMPLETE ones, once, after
# the last shard has left the queue. CPU-only and seconds long -- it just re-reads
# the result files the eval jobs wrote.
SUMMARY_ID="$(qsub -terse \
    -N "sum_${GAME}_$(basename "$MODEL")" \
    -hold_jid "$(IFS=,; echo "${ALL_EVAL_IDS[*]}")" \
    -o "$LOG_DIR/summary_${EXP_ID}_\$JOB_ID.out" \
    -e "$LOG_DIR/summary_${EXP_ID}_\$JOB_ID.err" \
    -v "EXP_ENV_FILE=$ENV_FILE" \
    ${SUMMARY_QSUB_OPTS:-} \
    "$HERE/summarize.sh" | tr -d '[:space:]')"
echo "[submit] summary   job $SUMMARY_ID  (held until all ${#ALL_EVAL_IDS[@]} eval job(s) finish)"

# --- the job ids, into the manifest -------------------------------------------
# The manifest was written before any of this was submitted, so it could not name the
# jobs. Add them now -- which id trains which segment, which scores which shard, and
# the one command that cancels the lot. Never fatal: everything above is already
# queued. See exp_record_jobs in experiments/lib/experiment.sh.
exp_record_jobs \
    --train "${TRAIN_IDS[@]}" --train-total "$TRAIN_SEGMENTS" \
    --eval ${EVAL_IDS[@]+"${EVAL_IDS[@]}"} \
    --playpen ${PPEVAL_IDS[@]+"${PPEVAL_IDS[@]}"} \
    --summary "$SUMMARY_ID" \
    --shard-total "$EVAL_SHARD_TOTAL" \
    --env-file "$ENV_FILE"

cat <<EOF

experiment : $EXP_ID
directory  : $EXP_DIR

  manifest.txt        what this run is          cat $EXP_DIR/manifest.txt
  logs/               job output                tail -f $LOG_DIR/train1_${EXP_ID}_${TRAIN_IDS[0]}.out
  train/              checkpoints (checkpoint-<step>/)
  eval/               lm-eval output per checkpoint
  RESULTS.md          lm-eval score table       (each shard writes a partial one; the
                                                 summary job writes the complete one)
  playpen-eval/       clembench gameplay per checkpoint
  PLAYPEN_RESULTS.md  clemscore + per-game table (same)
  wandb/              W&B run data ($([ "${WB_ENABLE:-1}" = "1" ] && echo "project ${WB_PROJECT:-playpen-marshal}, mode ${WB_MODE:-auto}" || echo "disabled"))
                      offline runs upload with  experiments/lib/wandb_sync.sh $EXP_DIR

jobs       : ${TRAIN_SEGMENTS} train$([ "$TRAIN_SEGMENTS" -gt 1 ] && echo " (chained, <= ${SEGMENT_STEPS} steps each)") + ${#ALL_EVAL_IDS[@]} eval (${EVAL_SHARD_TOTAL} shard(s) x $([ "${PPEVAL_ENABLE:-1}" = "1" ] && echo 2 || echo 1) evaluation(s)) + 1 summary

  qstat -u $USER            # watch the jobs
  qdel ${TRAIN_IDS[*]} ${ALL_EVAL_IDS[*]} $SUMMARY_ID   # cancel the experiment

Those ids, and that cancel command, are recorded in manifest.txt / manifest.json --
so they are still there once qstat has forgotten the jobs.

If a training segment dies, the rest of the chain still runs and picks up from its
last checkpoint. To restart a chain that stopped short:
  experiments/eddie/resume_experiment.sh $EXP_DIR
EOF

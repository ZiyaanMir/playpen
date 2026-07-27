#!/bin/bash
# Evaluate an experiment that has ALREADY trained -- no retraining, no dependency.
#
#   module load brics/userenv
#   cd <repo root>
#   experiments/isambard/run_eval.sh $PROJECTDIR/$USER/marshal-runs/<experiment>
#
# Use it to:
#   * re-run an evaluation that failed (bad dataset cache, OOM, walltime),
#   * score a run with different tasks or a bigger batch,
#   * evaluate checkpoints from a training job you launched some other way.
#
# Any eval setting can be overridden for this submission only; everything else is
# reused from the experiment's own experiment.env, so the run stays comparable:
#
#   EVAL_TASKS=logicbench     experiments/isambard/run_eval.sh <EXP_DIR>
#   EVAL_LIMIT=5              experiments/isambard/run_eval.sh <EXP_DIR>   # smoke test
#   EVAL_BATCH=32 EVAL_BASE=0 experiments/isambard/run_eval.sh <EXP_DIR>
#   HF_HUB_OFFLINE=1          experiments/isambard/run_eval.sh <EXP_DIR>
#
# The training half is never touched, and existing checkpoints are never rewritten.

set -euo pipefail

EXP_DIR_ARG="${1:-}"
[ -n "$EXP_DIR_ARG" ] || {
    echo "usage: $0 <EXP_DIR>" >&2
    echo "       e.g. $0 \$PROJECTDIR/\$USER/marshal-runs/guesswhat_Qwen3-4B_20260727-120621" >&2
    echo "       optional env: EVAL_TASKS EVAL_BATCH EVAL_LIMIT EVAL_BASE EVAL_EXTRA HF_HUB_OFFLINE" >&2
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
for _v in EVAL_TASKS EVAL_BATCH EVAL_LIMIT EVAL_BASE EVAL_EXTRA \
          BASE_EVAL_CACHE VENV_LMEVAL HF_HUB_OFFLINE HF_HOME; do
    if [ -n "${!_v+set}" ]; then _OV_NAMES+=("$_v"); _OV_VALS+=("${!_v}"); fi
done

# shellcheck disable=SC1090
source "$ENV_FILE"
source "$HERE/../lib/experiment.sh"
EXP_DIR="$EXP_DIR_ARG"
exp_layout

# Say up front what will be scored -- cheaper to notice here than in the queue.
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
echo "[eval-only] experiment : $EXP_ID"
echo "[eval-only] checkpoints: ${#CKS[@]} ($(basename "${CKS[0]}") .. $(basename "${CKS[-1]}"))"

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
} >> "$EVAL_ENV_FILE"

if [ "${#_OV_NAMES[@]}" -eq 0 ]; then
    echo "[eval-only] overrides  : none (reusing the experiment's stored settings)"
else
    for _i in "${!_OV_NAMES[@]}"; do
        echo "[eval-only] override   : ${_OV_NAMES[$_i]}=${_OV_VALS[$_i]}"
    done
fi

cd "$REPO"

EVAL_ID="$(sbatch --parsable \
    --job-name="ev_${GAME}_$(basename "$MODEL")" \
    --output="$LOG_DIR/eval_${EXP_ID}_%j.out" \
    --error="$LOG_DIR/eval_${EXP_ID}_%j.err" \
    --export="ALL,EXP_ENV_FILE=$EVAL_ENV_FILE" \
    ${EVAL_SBATCH_OPTS:-} \
    "$HERE/eval.sh")"

cat <<EOF
[eval-only] submitted job $EVAL_ID (no dependency -- runs as soon as a GPU is free)

  tail -f $LOG_DIR/eval_${EXP_ID}_$EVAL_ID.out
  cat    $EXP_DIR/RESULTS.md      # written when it finishes
  scancel $EVAL_ID                # cancel
EOF

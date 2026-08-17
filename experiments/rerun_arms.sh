#!/bin/bash
# Re-submit the four comparison arms of one game, using the CURRENT preset.
#
#   cd <repo root>
#   experiments/rerun_arms.sh -n clean_up                 # what WOULD be submitted
#   experiments/rerun_arms.sh clean_up                    # submit all four arms
#   experiments/rerun_arms.sh --arms final,final_grpo clean_up
#   MODEL=Qwen/Qwen3-8B experiments/rerun_arms.sh clean_up
#
# WHY A RE-SUBMIT AND NOT A RESUME. These runs produced NO checkpoint at all -- the
# game failed to import inside SelfPlayEnv.__init__, before the first episode. There
# is nothing for resume_experiment.sh to pick up (it errors on `no resumable
# checkpoint`, which is exactly the second failure every chained segment already hit).
# So each arm has to start again from step 0, as a NEW experiment directory.
#
# THE FOUR ARMS, and the flag that defines each (from train_selfplay.py):
#   final           (none)          MARSHAL on, TRL 'dapo' aggregation -- the main arm
#   final_baseline  --no-marshal    MARSHAL off: plain GRPO on the same rollouts
#   final_drgrpo    --dr-grpo       Dr. GRPO aggregation
#   final_grpo      --grpo-loss     TRL 'grpo' aggregation (what upstream MARSHAL uses)
# These are the same four EXP_TAGs the original submissions used, so the re-run lands
# next to the rest of the sweep and the tags stay comparable.
#
# EVERYTHING ELSE COMES FROM experiments/presets/<game>.env AS IT IS NOW -- not from
# the dead runs' experiment.env. That is deliberate: the preset is the current config,
# and for clean_up it has since gained TRAIN_SEGMENTS=10, which the older 0.6B arms ran
# with TRAIN_SEGMENTS=1. Re-running from the preset means the new arms match the rest
# of today's sweep rather than reproducing a stale layout. Override any of it by
# exporting it, exactly as with run_experiment.sh.
#
# The game is import-checked FIRST, once. Nothing is submitted if it fails -- otherwise
# this script's only achievement would be to repeat the original failure on
# 4 x TRAIN_SEGMENTS fresh GPU allocations.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$HERE/..}"
REPO="$(cd "$REPO" && pwd)"

DRY_RUN=0
CLUSTER=""
ARMS="final,final_baseline,final_drgrpo,final_grpo"
GAME=""

usage() {
    cat <<EOF
usage: $(basename "$0") [options] <game>

Re-submits one game's comparison arms as fresh experiments, from the current preset.

options:
  -n, --dry-run     print the submissions, run nothing
      --arms LIST   comma-separated subset of:
                      final, final_baseline, final_drgrpo, final_grpo
                    (default: all four)
      --cluster C   eddie | isambard   (default: detected from qsub/sbatch)
  -h, --help        this

Any preset value can be overridden by exporting it, e.g.
  MODEL=Qwen/Qwen3-8B  $(basename "$0") clean_up
  TRAIN_SEGMENTS=4     $(basename "$0") clean_up
  MAX_STEPS=200 EXP_TAG_SUFFIX=_smoke $(basename "$0") --arms final clean_up
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -n|--dry-run) DRY_RUN=1 ;;
        --arms)       ARMS="$2"; shift ;;
        --cluster)    CLUSTER="$2"; shift ;;
        -h|--help)    usage; exit 0 ;;
        -*)           echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
        *)            GAME="$1" ;;
    esac
    shift
done

[ -n "$GAME" ] || { usage >&2; exit 2; }
[ -f "$REPO/experiments/presets/$GAME.env" ] || {
    echo "ERROR: no preset at experiments/presets/$GAME.env" >&2
    echo "       Available: $(cd "$REPO/experiments/presets" && ls *.env | sed 's/\.env//' | tr '\n' ' ')" >&2
    exit 1
}

# --- which cluster ------------------------------------------------------------
if [ -z "$CLUSTER" ]; then
    if command -v qsub >/dev/null 2>&1 && ! command -v sbatch >/dev/null 2>&1; then
        CLUSTER=eddie
    elif command -v sbatch >/dev/null 2>&1 && ! command -v qsub >/dev/null 2>&1; then
        CLUSTER=isambard
    elif [ "$DRY_RUN" = 1 ]; then
        CLUSTER=eddie
    else
        echo "ERROR: could not tell which cluster this is. Pass --cluster eddie|isambard." >&2
        exit 1
    fi
fi
SUBMITTER="$REPO/experiments/$CLUSTER/run_experiment.sh"
[ -f "$SUBMITTER" ] || { echo "ERROR: no submitter at $SUBMITTER" >&2; exit 1; }

# The flag that defines each arm. Keep in step with the header comment.
arm_flag() {
    case "$1" in
        final)          printf '%s' "" ;;
        final_baseline) printf '%s' "--no-marshal" ;;
        final_drgrpo)   printf '%s' "--dr-grpo" ;;
        final_grpo)     printf '%s' "--grpo-loss" ;;
        *) return 1 ;;
    esac
}

IFS=',' read -r -a ARM_LIST <<< "$ARMS"
for A in "${ARM_LIST[@]}"; do
    arm_flag "$A" >/dev/null || {
        echo "ERROR: unknown arm '$A'." >&2
        echo "       Valid: final final_baseline final_drgrpo final_grpo" >&2
        exit 2
    }
done

echo "game    : $GAME"
echo "cluster : $CLUSTER"
echo "preset  : experiments/presets/$GAME.env  (current -- NOT the dead runs' env files)"
echo "model   : ${MODEL:-<preset default>}"
echo "arms    : ${ARM_LIST[*]}"
echo

# --- the game must actually import -------------------------------------------
# The one check that separates this from the submission that lost the runs.
if [ "$DRY_RUN" = 0 ]; then
    bash "$REPO/experiments/lib/check_game_env.sh" "$GAME" || {
        echo "" >&2
        echo "ERROR: nothing submitted -- $GAME cannot be constructed in the training venv." >&2
        echo "       Install the missing package (see above), then re-run this script." >&2
        exit 1
    }
    echo
fi

cd "$REPO"
OK=0 FAIL=0
for ARM in "${ARM_LIST[@]}"; do
    FLAG="$(arm_flag "$ARM")"
    TAG="${ARM}${EXP_TAG_SUFFIX:-}"
    echo "--- arm $ARM  (EXP_TAG=$TAG, EXTRA_TRAIN_ARGS='${FLAG}') ---"
    if [ "$DRY_RUN" = 1 ]; then
        printf '    EXP_TAG=%s EXTRA_TRAIN_ARGS=%s%s experiments/%s/run_experiment.sh %s\n' \
            "$TAG" "'${FLAG}'" "${MODEL:+ MODEL=$MODEL}" "$CLUSTER" "$GAME"
        OK=$((OK + 1))
        continue
    fi
    # EXTRA_TRAIN_ARGS is exported per-arm; everything else is inherited from this
    # shell (so a caller's MODEL/MAX_STEPS override applies to every arm) and from
    # the preset, which run_experiment.sh loads itself.
    if EXP_TAG="$TAG" EXTRA_TRAIN_ARGS="$FLAG" bash "$SUBMITTER" "$GAME"; then
        OK=$((OK + 1))
    else
        echo "[rerun] FAILED to submit arm $ARM" >&2
        FAIL=$((FAIL + 1))
    fi
    echo
done

if [ "$DRY_RUN" = 1 ]; then
    echo "DRY RUN -- $OK submission(s) shown, nothing queued."
    echo "Re-run without -n to submit. The game import check runs then, not now."
    exit 0
fi

if [ "$FAIL" -gt 0 ]; then
    echo "submitted $OK arm(s), $FAIL failed."
else
    echo "submitted $OK arm(s)."
fi
echo "watch:   $([ "$CLUSTER" = eddie ] && echo "qstat -u $USER" || echo 'squeue --me')"
echo "status:  experiments/status.sh -v"
[ "$FAIL" -gt 0 ] && exit 1
exit 0

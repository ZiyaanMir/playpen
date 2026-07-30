#!/bin/bash
# Third job of an Eddie experiment: play every checkpoint through the clembench
# games and score it (clemscore), after training and after lm-eval. Submitted by
# run_experiment.sh with -hold_jid on the lm-eval job, or on its own by
# run_playpen_eval.sh.
#
# Results land inside the experiment directory:
#     $EXP_DIR/playpen-eval/base/            untrained model
#     $EXP_DIR/playpen-eval/checkpoint-50/   ... one dir per adapter
#     $EXP_DIR/PLAYPEN_RESULTS.md            the table across all of them
#
# Re-runnable by hand once training is done:
#     source <EXP_DIR>/experiment.env && experiments/eddie/playpen_eval.sh
#
# --- Grid Engine options (overridable via PPEVAL_QSUB_OPTS) -------------------
#$ -N marshal_ppeval
#$ -cwd
#$ -q gpu
#$ -l gpu=1
#$ -l h_rt=48:00:00
#$ -pe sharedmem 4
#$ -l h_rss=16G
#
# 48 h because this is GENERATION, not loglikelihood scoring: ~70 validation
# instances across 14 games, each a multi-turn dialogue, once per checkpoint. A
# 4B model takes hours per row. Trim the work rather than the walltime if that is
# too long -- PPEVAL_CKPTS=last scores only the final checkpoint, and
# PPEVAL_TIMEOUT=2h stops a single wedged game from eating the rest.

set -euo pipefail

: "${EXP_ENV_FILE:?not set -- submit via experiments/eddie/run_experiment.sh}"
# shellcheck disable=SC1090
source "$EXP_ENV_FILE"

echo "### experiment: ${EXP_ID} (playpen games eval) ###"

cd "$REPO"
# Unlike the lm-eval job, this one runs in the repo's OWN venv: clemcore, playpen
# and peft all live there, so there is no separate evaluation environment to build.
# _common.sh gives us that venv plus Eddie's module/cache setup, exactly as the
# training job gets it.
source "$REPO/slurm_eddie/_common.sh"

source "$REPO/experiments/lib/experiment.sh"
source "$REPO/experiments/lib/playpen_eval.sh"
exp_layout
pp_layout
pp_banner

# _common.sh defaults HF_HUB_OFFLINE=1, which is right for training (the weights are
# already cached) and wrong here: `playpen eval` also needs the playpen-data
# validation split, and offline mode turns a missing dataset cache into a failure on
# every single row. Default to online and let pp_preflight_data fetch it once.
# Set HF_HUB_OFFLINE=1 at submit time for a guaranteed network-free run, once the
# dataset has been cached by an earlier run or on a login node.
export HF_HUB_OFFLINE="${PPEVAL_HF_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$PPEVAL_DIR"

# The huggingface_local backend is plain torch, which handles Eddie's GPU-UUID form
# of CUDA_VISIBLE_DEVICES fine -- the remap in _common.sh is only needed by vLLM.

pp_preflight_data || exit 1

pp_run_all || true          # the failure count; the table is written either way

echo "[ppeval] end = $(date --iso-8601=seconds)"
exit 0

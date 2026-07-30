#!/bin/bash
# Third job of an Isambard experiment: play every checkpoint through the clembench
# games and score it (clemscore), after training and after lm-eval. Submitted by
# run_experiment.sh with --dependency=afterany on the lm-eval job, or on its own by
# run_playpen_eval.sh.
#
# Results land inside the experiment directory:
#     $EXP_DIR/playpen-eval/base/            untrained model
#     $EXP_DIR/playpen-eval/checkpoint-50/   ... one dir per adapter
#     $EXP_DIR/PLAYPEN_RESULTS.md            the table across all of them
#
# Re-runnable by hand once training is done:
#     source <EXP_DIR>/experiment.env && experiments/isambard/playpen_eval.sh
#
#SBATCH --job-name=marshal_ppeval
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --no-requeue
#
# This is GENERATION, not loglikelihood scoring: ~70 validation instances across 14
# games, each a multi-turn dialogue, once per checkpoint -- hours per row for a 4B
# model. 20 h stays under Isambard's 24 h cap with margin. Trim the work rather than
# the walltime if that is too long: PPEVAL_CKPTS=last scores only the final
# checkpoint, and PPEVAL_TIMEOUT=2h stops one wedged game eating the rest.

set -euo pipefail

: "${EXP_ENV_FILE:?not set -- submit via experiments/isambard/run_experiment.sh}"
# shellcheck disable=SC1090
source "$EXP_ENV_FILE"

cd "$REPO"
source "$REPO/experiments/lib/experiment.sh"
source "$REPO/experiments/lib/playpen_eval.sh"
exp_layout
pp_layout
pp_banner

# Unlike the lm-eval job, this one runs in the repo's OWN venv: clemcore, playpen and
# peft all live there, so there is no separate evaluation environment to build.
# Verified by resolved path, which accepts Isambard's /projects -> /lus/... mount
# aliasing while still catching a .venv copied from another checkout.
exp_activate_venv "$REPO/.venv" "training venv" || exit 1

# Must run BEFORE anything imports torch: the inherited TMPDIR (/local/user/<uid>) is
# a login-node path that does not exist on the compute node. See exp_setup_tmpdir.
exp_setup_tmpdir || exit 1

export HF_HOME="${HF_HOME:-$PROJECTDIR/hf}"
# Online by default. `playpen eval` needs the playpen-data validation split as well as
# the model weights, and Isambard compute nodes do have outbound access -- the same
# reasoning (and the same past bug) as the lm-eval job, where forcing offline mode made
# every row fail on a dataset that had never been cached. Submit with
# PPEVAL_HF_OFFLINE=1 for a guaranteed network-free run once it has been.
export HF_HUB_OFFLINE="${PPEVAL_HF_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$PPEVAL_DIR"

# Slurm gives clean integer GPU ordinals, so nothing to remap here (unlike Eddie).

pp_preflight_data || exit 1

pp_run_all || true          # the failure count; the table is written either way

echo "[ppeval] end = $(date --iso-8601=seconds)"
exit 0

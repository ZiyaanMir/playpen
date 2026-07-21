#!/bin/bash
#$ -N marshal_q3_0.6B_dond
#$ -cwd
#$ -q gpu
#$ -l gpu=1
#$ -l a100=true
#$ -l h_rt=24:00:00
#$ -pe sharedmem 8
#$ -l h_rss=8G
#$ -o logs/train.$JOB_ID.out
#$ -e logs/train.$JOB_ID.err
##$ -m bea -M <uun>@ed.ac.uk    # uncomment for begin/end/abort mail
#
# Eddie port of ../slurm/marshal_q3_0.6B_dond.sh.
# GPU choice: A100 is ample for 0.6B + LoRA. For H200 swap the resource line to
#   #$ -l h200=true
# and VERIFY the flag name first (guide §1.2).
#
# dond (deal-or-no-deal) is a 2-player negotiation game, so per-seat advantage
# normalization is structurally live here — unlike wordle, which is 1-player and
# makes MARSHAL's headline contribution inert.

# Grid Engine spools the job script to a private dir, so $0 is NOT in the repo.
# -cwd + $SGE_O_WORKDIR is the reliable anchor; submit from the repo root.
source "${SGE_O_WORKDIR:-$PWD}/slurm_eddie/_common.sh"

python -m examples.marshal.train_selfplay \
    --model Qwen/Qwen3-0.6B \
    --game dond \
    --marshal-config examples/marshal/marshal_config.yaml \
    --num-generations 8 \
    --per-device-batch-size 16 \
    --grad-accum 2 \
    --max-steps 200 \
    --save-steps 50 \
    --learning-rate 1e-5 \
    --kl-beta 0.2 \
    --max-completion-length 2048 \
    --vllm-gpu-memory-utilization 0.45 \
    --vllm-max-model-len 8192 \
    --output-dir "$MARSHAL_RUNS/dond"

echo "end=$(date --iso-8601=seconds)"

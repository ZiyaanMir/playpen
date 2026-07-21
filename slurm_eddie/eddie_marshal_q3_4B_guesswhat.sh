#!/bin/bash
#$ -N marshal_q3_4B_guesswhat
#$ -cwd
#$ -q gpu
#$ -l gpu=1
#$ -l h200=true
#$ -l h_rt=48:00:00
#$ -pe sharedmem 12
#$ -l h_rss=12G
#$ -o logs/train.$JOB_ID.out
#$ -e logs/train.$JOB_ID.err
##$ -m bea -M <uun>@ed.ac.uk    # uncomment for begin/end/abort mail
#
# Eddie port of ../slurm/marshal_q3_4B_guesswhat.sh.
# GPU choice: H200 (141 GB) — 4B + LoRA + a colocated vLLM engine is tight on a
#   40 GB A100. To fall back to A100 instead:
#     #$ -l a100=true          (prefer an 80 GB node if the complex distinguishes them)
#   and drop --vllm-gpu-memory-utilization to ~0.35.
# VERIFY the h200 complex name before submitting (guide §1.2) — a wrong -l leaves
# the job in qw indefinitely rather than failing loudly.
#
# h_rt=48h may exceed the GPU queue cap; check with `qconf -sq gpu | grep h_rt`
# and lower it if so. There is no resume flag — an h_rt kill restarts from step 0.

# Grid Engine spools the job script to a private dir, so $0 is NOT in the repo.
# -cwd + $SGE_O_WORKDIR is the reliable anchor; submit from the repo root.
source "${SGE_O_WORKDIR:-$PWD}/slurm_eddie/_common.sh"

python -m examples.marshal.train_selfplay \
    --model Qwen/Qwen3-4B \
    --game guesswhat \
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
    --output-dir "$MARSHAL_RUNS/guesswhat"

echo "end=$(date --iso-8601=seconds)"

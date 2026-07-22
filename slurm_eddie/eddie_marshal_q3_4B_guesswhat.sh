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

# Memory sizing (added 2026-07-22, after Isambard job 5744124 OOMed at step 0).
# A rollout "row" is the WHOLE flattened episode for one seat: selfplay_agent.py
# concatenates every turn's generation plus the env feedback between them, and
# nothing truncates it. So
#       seq_len ~= rounds x max_completion_length     (guesswhat = 8 rounds)
# NOT just max_completion_length. TRL materializes logits in fp32, needing
#       per_device_batch_size x seq_len x 151936 (Qwen3 vocab) x 4 bytes
# and TWICE that, because --kl-beta 0.2 adds a reference-model forward. At the old
# 16 x 2048 that is ~91 GiB, which OOMs even this H200 once vLLM has its share.
# The logits cost is almost model-size-independent (vocab is 151936 at every Qwen3
# size), so 4B needs the same fix as 0.6B plus headroom for ~8 GiB of bf16 weights.
#
# per-device-batch-size x grad-accum is UNCHANGED at 32, so the MARSHAL per-seat
# advantage pool (== TRL's generation batch) is exactly as before; only the
# per-forward micro-batch shrinks. 32 stays divisible by --num-generations 8.
# batch 4 (vs 2 on the 95 GiB GH200) because the H200's 141 GB affords it.
#
# --max-turns 30 is a safety ceiling only: guesswhat self-limits at 8 rounds
# (~16-18 env turns), so it never binds; it just caps a runaway episode.

# Grid Engine spools the job script to a private dir, so $0 is NOT in the repo.
# -cwd + $SGE_O_WORKDIR is the reliable anchor; submit from the repo root.
source "${SGE_O_WORKDIR:-$PWD}/slurm_eddie/_common.sh"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# UPDATED 2026-07-22: reasoning is now DISABLED at the source. render_prompt() in
# playpen/marshal/selfplay_agent.py applies the chat template with
# enable_thinking=False for every model and game, because Qwen3 was spending the
# whole per-turn budget inside <think>, never closing the block, and so never
# emitting an action -- every episode aborted (all-equal pool, zero gradient).
# Turns are therefore short again: the per-turn budget is back down, and
# vllm-max-model-len returns to 8192 now that think blocks no longer inflate the
# prompt. Any batch/turn-length figure in the paragraphs above refers to the
# pre-thinking sizing and still applies. The <think> stripping stays in place as a
# safety net for models that emit a block regardless; it is a no-op when they don't.
python -m examples.marshal.train_selfplay \
    --model Qwen/Qwen3-4B \
    --game guesswhat \
    --marshal-config examples/marshal/marshal_config.yaml \
    --num-generations 8 \
    --per-device-batch-size 2 \
    --grad-accum 8 \
    --max-steps 200 \
    --save-steps 50 \
    --learning-rate 1e-5 \
    --kl-beta 0.2 \
    --max-completion-length 512 \
    --max-turns 30 \
    --gradient-checkpointing \
    --vllm-gpu-memory-utilization 0.30 \
    --vllm-max-model-len 8192 \
    --output-dir "$MARSHAL_RUNS/guesswhat"

echo "end=$(date --iso-8601=seconds)"

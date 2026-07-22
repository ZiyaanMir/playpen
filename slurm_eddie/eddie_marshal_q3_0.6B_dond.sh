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
#
# Memory sizing — CORRECTED 2026-07-22. The estimate here used to assume
# seq_len == max_completion_length. It is not. A rollout "row" is the WHOLE
# flattened episode for one seat: selfplay_agent.py concatenates every turn's
# generation plus the env feedback between them, and nothing truncates it. So
#       seq_len ~= rounds x max_completion_length     (dond = 5 rounds)
# i.e. several times larger than the old estimate. TRL materializes logits in fp32
# (accelerate upcasts them), needing
#       per_device_batch_size x seq_len x 151936 (Qwen3 vocab) x 4 bytes
# and TWICE that, because --kl-beta 0.2 adds a reference-model forward.
#
# Isambard job 5744124 proved it on the sibling guesswhat run: rows reached ~10k
# tokens, so 16 x 10k x 151936 x 4 = ~91 GiB — matching the 93.51 GiB the allocator
# reported before it OOMed a 95 GiB GH200 at step 0. Hence shorter turns, batch 4,
# and gradient checkpointing here. dond keeps a longer 768-token turn than
# guesswhat's 512 because a negotiation message (a split proposal plus reasoning)
# is wordier than a yes/no.
#
# per-device-batch-size x grad-accum is UNCHANGED at 16, so the MARSHAL per-seat
# advantage pool (== TRL's generation batch) is exactly as before; only the
# per-forward micro-batch shrinks. 16 stays divisible by --num-generations 8.
#
# --max-turns 30 is a safety ceiling only: dond self-limits at 5 rounds
# (~10-12 env turns), so it never binds; it just caps a runaway episode.

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
    --model Qwen/Qwen3-0.6B \
    --game dond \
    --marshal-config examples/marshal/marshal_config.yaml \
    --num-generations 8 \
    --per-device-batch-size 2 \
    --grad-accum 8 \
    --max-steps 200 \
    --save-steps 50 \
    --learning-rate 1e-5 \
    --kl-beta 0.2 \
    --max-completion-length 768 \
    --max-turns 30 \
    --gradient-checkpointing \
    --vllm-gpu-memory-utilization 0.30 \
    --vllm-max-model-len 8192 \
    --output-dir "$MARSHAL_RUNS/dond"

echo "end=$(date --iso-8601=seconds)"

#!/bin/bash
#SBATCH --job-name=marshal_q3_0.6B_dond
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=20:00:00        # stay under the 24 h cap with margin
#SBATCH --no-requeue           # a requeue restarts from step 0 — don't silently redo 20 h

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# $PROJECTDIR/$USER come from brics/userenv; fail loudly rather than write to /
: "${PROJECTDIR:?unset - run 'module load brics/userenv'}"
: "${USER:?unset}"

source .venv/bin/activate
export HF_HOME=$PROJECTDIR/hf
export TRL_EXPERIMENTAL_SILENCE=1
export TOKENIZERS_PARALLELISM=false
# Reduces allocator fragmentation; suggested by the OOM message itself.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date --iso-8601=seconds)"

# --- memory sizing: why these numbers ---------------------------------------
# A rollout "row" is the WHOLE flattened episode for one seat, NOT one turn:
# selfplay_agent.py concatenates every turn's generation plus the env feedback
# between them, and nothing truncates it. --max-completion-length caps each
# TURN's generation, so a row grows to roughly
#       rounds x max_completion_length          (dond = 5 rounds)
# TRL then materializes logits for that row in fp32 (accelerate upcasts them),
# needing   per_device_batch_size x seq_len x 151936 (Qwen3 vocab) x 4 bytes
# and TWICE that, because --kl-beta 0.2 adds a reference-model forward.
#
# The old 16 x 2048 config OOMed a 95 GiB GH200 at step 0 on the sibling guesswhat
# job (5744124): rows reached ~10k tokens, so 16 x 10k x 151936 x 4 = ~91 GiB,
# matching the 93.51 GiB the allocator reported. Fixes below: shorter turns, a
# smaller forward batch, and gradient checkpointing. dond keeps a longer 768-token
# turn than guesswhat's 512 because a negotiation message (a split proposal plus
# reasoning) is wordier than a yes/no.
#
# IMPORTANT: per-device-batch-size x grad-accum is UNCHANGED at 32, so the MARSHAL
# per-seat advantage pool (== TRL's generation batch) is exactly as before —
# only the per-forward micro-batch shrinks. 32 stays divisible by num-generations.
#
# --max-turns 30 is a safety ceiling only: dond self-limits at 5 rounds
# (~10-12 env turns), so it never binds; it just caps a runaway episode.

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
    --output-dir "$PROJECTDIR/$USER/marshal-runs/dond"

echo "end=$(date --iso-8601=seconds)"

#!/bin/bash
#SBATCH --job-name=marshal_taboo
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

echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date --iso-8601=seconds)"

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
    --output-dir "$PROJECTDIR/$USER/marshal-runs/dond" \

echo "end=$(date --iso-8601=seconds)"
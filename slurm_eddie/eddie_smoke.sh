#!/bin/bash
#$ -N marshal_smoke
#$ -cwd
#$ -q gpu
#$ -l gpu=1
#$ -l a100=true
#$ -l h_rt=01:00:00
#$ -pe sharedmem 8
#$ -l h_rss=8G
#$ -o logs/smoke.$JOB_ID.out
#$ -e logs/smoke.$JOB_ID.err
#
# 30-minute proof that the port actually runs on this cluster, before committing
# 24 h of GPU time to a real run. Submit with:  qsub slurm_eddie/eddie_smoke.sh
# (create logs/ first — Grid Engine opens -o/-e at job start, not inside the script.)

# Grid Engine spools the job script to a private dir, so $0 is NOT in the repo.
# -cwd + $SGE_O_WORKDIR is the reliable anchor; submit from the repo root.
source "${SGE_O_WORKDIR:-$PWD}/slurm_eddie/_common.sh"

echo "--- 1. the package imports (this is the step a bad clone fails, see guide §1.1)"
python -c "
from playpen.marshal import MarshalConfig, MarshalGRPOTrainer, SelfPlayEnv
print('  imports OK — MarshalGRPOTrainer is present')
"

echo "--- 2. unit tests (no GPU needed; expect 58 passed)"
python -m pytest tests/ -q 2>&1 | tail -5

echo "--- 3. clembench game discovery"
python -c "
from playpen.marshal.selfplay_env import resolve_game_spec, list_instance_indices
for g in ('taboo', 'guesswhat', 'dond'):
    try:
        s = resolve_game_spec(g)
        print(f'  {g:10s} players={s.players} instances={len(list_instance_indices(g))}')
    except Exception as e:
        print(f'  {g:10s} UNAVAILABLE: {type(e).__name__}: {e}')
"

echo "--- 4. GPU actually visible to torch"
python -c "
import torch
print('  cuda:', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
print('  CUDA_VISIBLE_DEVICES =', __import__('os').environ.get('CUDA_VISIBLE_DEVICES'))
"

echo "--- 5. one real training step end-to-end (vLLM + LoRA + MARSHAL advantages)"
python -m examples.marshal.train_selfplay \
    --model Qwen/Qwen3-0.6B \
    --game guesswhat \
    --marshal-config examples/marshal/marshal_config.yaml \
    --num-generations 8 \
    --per-device-batch-size 2 \
    --grad-accum 8 \
    --max-steps 2 \
    --save-steps 2 \
    --learning-rate 1e-5 \
    --kl-beta 0.2 \
    --max-completion-length 1536 \
    --max-turns 30 \
    --gradient-checkpointing \
    --vllm-gpu-memory-utilization 0.30 \
    --vllm-max-model-len 16384 \
    --output-dir "$MARSHAL_RUNS/guesswhat"

echo "end=$(date --iso-8601=seconds)"
echo "If step 5 printed a checkpoint path and no traceback, the pipeline is live."

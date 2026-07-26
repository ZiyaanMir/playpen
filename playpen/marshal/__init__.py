"""MARSHAL-style self-play RL for playpen.

A framework-light port of MARSHAL's self-play training algorithm (turn-level
advantage estimator + agent-specific/per-seat advantage normalization) onto
playpen's TRL ``GRPOTrainer`` + PEFT/LoRA stack, runnable directly against
clembench games.

Import structure:
  * ``config``, ``wandb_utils`` and ``advantage`` depend only on the standard
    library / torch and are always importable (unit-testable without the training
    stack -- ``wandb_utils`` does not even require ``wandb`` until a run starts).
  * ``selfplay_env`` depends on clemcore (a hard playpen dependency).
  * ``selfplay_agent`` and ``trainer`` require ``trl`` (and ``vllm`` at train
    time); they are exposed lazily so ``import playpen.marshal`` does not require
    the training stack.
"""

from __future__ import annotations

from playpen.marshal.config import MarshalConfig
from playpen.marshal.wandb_utils import WandbSettings

__all__ = [
    "MarshalConfig",
    "WandbSettings",
    # lazily provided (require heavier deps):
    "MarshalGRPOTrainer",
    "SelfPlayEnv",
    "build_selfplay_dataset",
    "build_selfplay_rollout_func",
    "build_reward_func",
    "play_selfplay_episode",
]


def __getattr__(name: str):  # PEP 562 lazy attribute access
    if name == "SelfPlayEnv":
        from playpen.marshal.selfplay_env import SelfPlayEnv

        return SelfPlayEnv
    if name == "play_selfplay_episode":
        from playpen.marshal.selfplay_agent import play_selfplay_episode

        return play_selfplay_episode
    if name in {
        "MarshalGRPOTrainer",
        "build_selfplay_dataset",
        "build_selfplay_rollout_func",
        "build_reward_func",
    }:
        from playpen.marshal import trainer as _trainer

        return getattr(_trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

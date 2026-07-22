"""MarshalGRPOTrainer: TRL GRPO with MARSHAL-style self-play advantages.

This module wires the pieces together:

* :class:`MarshalGRPOTrainer` -- a thin subclass of ``trl.GRPOTrainer`` whose only
  change is to (optionally) replace TRL's scalar group-relative advantages with
  MARSHAL's ``(B, T)`` turn-level, per-seat-normalized advantages. When
  ``MarshalConfig.enabled`` is ``False`` it is exactly stock ``GRPOTrainer``.
* :func:`build_selfplay_rollout_func` -- the ``rollout_func`` that plays self-play
  episodes and emits per-seat rollout rows (the TRL 1:1 prompt->row contract).
* :func:`build_reward_func` -- echoes the terminal reward back to TRL (used
  directly by the ``enabled=False`` path, and harmless/logged when enabled).
* :func:`build_selfplay_dataset` -- builds the ``{instance_idx}::seat{n}`` prompt
  rows self-contained from the game's packaged instances (no network).

Requires ``trl`` (and, at train time, ``vllm``, since custom ``rollout_func`` is
only honored under ``use_vllm=True``). Import this module only when the training
stack is available; ``playpen.marshal.config`` / ``playpen.marshal.advantage`` are
importable without it.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Optional

import torch
import trl

from playpen.marshal.advantage import (
    LengthPenaltySpec,
    RowRollout,
    compute_marshal_advantages,
    turn_token_lengths,
)
from playpen.marshal.config import MarshalConfig
from playpen.marshal.selfplay_agent import play_selfplay_episode
from playpen.marshal.selfplay_env import SelfPlayEnv, list_instance_indices


def parse_prompt(prompt: str) -> tuple[int, int]:
    """Parse a ``"{instance_idx}::seat{n}"`` dataset prompt into (instance_idx, seat)."""
    instance_part, _, seat_part = prompt.partition("::seat")
    return int(instance_part), int(seat_part)


def build_selfplay_dataset(
    game_name: str,
    *,
    seats: Optional[List[int]] = None,
    num_players: Optional[int] = None,
    max_instances: Optional[int] = None,
    instances_filter: Optional[Callable[[dict], bool]] = None,
):
    """Build a HuggingFace dataset of ``{instance_idx}::seat{n}`` prompts.

    One row per (game instance, seat). Self-contained: instance indices come from
    the game's packaged ``instances.json`` (the same indices ``SelfPlayEnv.reset``
    looks up), so there is no dependency on the ``colab-potsdam/playpen-data`` HF
    hub dataset and no risk of an id mismatch. Indices are used instead of
    clembench game_ids because game_ids are only unique *within* an experiment
    (a bare game_id lookup would silently collapse onto the first experiment).
    """
    from datasets import Dataset

    from playpen.marshal.selfplay_env import resolve_game_spec

    if seats is None:
        n = num_players if num_players is not None else int(resolve_game_spec(game_name).players)
        seats = list(range(n))

    instance_indices = list_instance_indices(game_name, instances_filter=instances_filter)
    if max_instances is not None:
        instance_indices = instance_indices[:max_instances]

    prompts = [f"{idx}::seat{seat}" for idx in instance_indices for seat in seats]
    return Dataset.from_dict({"prompt": prompts})


def build_reward_func(config: MarshalConfig) -> Callable[..., List[float]]:
    """Return a TRL reward function that echoes the per-row terminal reward.

    Terminal rewards from clembench: SUCCESS +1, FAILURE 0, ABORTED -1.

    When MARSHAL is disabled, TRL's stock group-relative normalization consumes
    this scalar, and we apply the Wordle-example degenerate-all-abort shaping
    (replace an exact -1.0 abort with a small random negative) so an all-abort
    group still has non-zero advantage variance. When MARSHAL is enabled these
    scalars are only used for logging (the real signal is the ``(B, T)`` override),
    so we pass them through unshaped.
    """
    shape_aborts = not config.enabled

    def reward_func(completions: List[str], **kwargs: Any) -> List[float]:
        rewards = kwargs.get("rewards")
        if rewards is None:
            return [0.0] * len(completions)
        if not shape_aborts:
            return [float(r) for r in rewards]
        shaped = []
        for r in rewards:
            r = float(r)
            if math.isclose(r, -1.0, abs_tol=1e-6):
                r = -random.random()
            shaped.append(r)
        return shaped

    return reward_func


def build_selfplay_rollout_func(
    env: SelfPlayEnv,
    config: MarshalConfig,
    *,
    max_turns: int = 100,
    strip_think: bool = True,
) -> Callable[[List[str], "trl.GRPOTrainer"], Dict[str, list]]:
    """Return a ``rollout_func`` closing over a persistent :class:`SelfPlayEnv`.

    For each incoming prompt ``"{instance_idx}::seat{n}"`` a *fresh, complete*
    self-play episode is played and only the requested seat's row is kept (the
    other seat's generation is discarded). This satisfies TRL's 1:1 prompt->row
    contract with no cross-call state at the cost of ~num_players x generation.

    The returned dict carries, in addition to TRL's required keys, the extra fields
    the advantage override needs: ``env_mask`` (consumed by TRL as the gradient
    mask), ``owner_mask`` (a copy that survives into ``inputs`` for our use),
    ``turn_end_positions``, ``turn_rewards``, ``seat`` and the scalar ``rewards``.
    """

    def rollout_func(prompts: List[str], trainer: "trl.GRPOTrainer") -> Dict[str, list]:
        tokenizer = trainer.processing_class
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", None) or 0
        out: Dict[str, list] = {
            "prompt_ids": [],
            "completion_ids": [],
            "logprobs": [],
            "env_mask": [],
            "owner_mask": [],
            "turn_end_positions": [],
            "turn_rewards": [],
            "seat": [],
            "rewards": [],
        }
        for prompt in prompts:
            instance_idx, seat = parse_prompt(prompt)
            rollouts = play_selfplay_episode(
                env,
                trainer,
                tokenizer,
                instance_idx,
                seed=instance_idx,
                max_turns=max_turns,
                strip_think=strip_think,
            )
            row = rollouts.get(seat) or _empty_row(seat, pad_id)
            out["prompt_ids"].append(list(row.prompt_ids))
            out["completion_ids"].append(list(row.completion_ids))
            out["logprobs"].append(list(row.logprobs))
            out["env_mask"].append(list(row.owner_mask))
            out["owner_mask"].append(list(row.owner_mask))
            out["turn_end_positions"].append(list(row.turn_end_positions))
            out["turn_rewards"].append(list(row.turn_rewards))
            out["seat"].append(int(row.seat))
            out["rewards"].append(float(row.terminal_reward))
        return out

    return rollout_func


def _empty_row(seat: int, pad_token_id: int = 0):
    """A minimal, non-crashing row for a seat that produced no usable trajectory.

    TRL's rollout contract requires one row per prompt with non-empty tensors, so
    we emit a single pad token -- but marked as an *environment* token
    (``owner_mask=[0]``) with no turns: the loss mask zeroes it out, and
    ``normalize_returns_by_seat`` excludes rows without model tokens from the
    seat pools, so the placeholder carries no gradient and no statistics.
    (Previously this used ``owner_mask=[1]``, which trained a fabricated token
    with a nonzero advantage and diluted the seat pool mean.)
    """
    from playpen.marshal.selfplay_agent import SeatRollout

    return SeatRollout(
        seat=seat,
        prompt_ids=[int(pad_token_id)],
        completion_ids=[int(pad_token_id)],
        logprobs=[0.0],
        owner_mask=[0],
        turn_end_positions=[],
        turn_rewards=[],
        terminal_reward=0.0,
    )


class MarshalGRPOTrainer(trl.GRPOTrainer):
    """``trl.GRPOTrainer`` with an optional MARSHAL self-play advantage override.

    Pass a :class:`MarshalConfig` via ``marshal_config``. When
    ``marshal_config.enabled`` is False this behaves exactly like the base trainer.
    """

    def __init__(self, *args: Any, marshal_config: Optional[MarshalConfig] = None, **kwargs: Any) -> None:
        self.marshal_config = marshal_config or MarshalConfig()
        super().__init__(*args, **kwargs)

    def _generate_and_score_completions(self, inputs):
        # Let the base class do all generation, reward scoring, logprob computation
        # and its own scalar advantage estimate. It also merges our rollout_func
        # extra fields into `inputs` in place before returning.
        output = super()._generate_and_score_completions(inputs)
        # Placeholder rows (_empty_row) have env_mask all-zero, so a degenerate
        # batch where *every* row is a placeholder yields num_items_in_batch == 0
        # and TRL's dapo/cispo losses would divide 0/0 -> NaN. The loss numerator
        # is 0 in that case, so clamping the normalizer keeps the loss at 0.
        if "num_items_in_batch" in output:
            output["num_items_in_batch"] = torch.clamp(output["num_items_in_batch"], min=1)
        if not self.marshal_config.enabled:
            return output
        output["advantages"] = self._compute_marshal_advantages(inputs, output)
        return output

    def _compute_marshal_advantages(self, inputs, output) -> torch.Tensor:
        cfg = self.marshal_config
        base_adv = output["advantages"]
        completion_ids = output["completion_ids"]
        batch_size, seq_len = completion_ids.shape[0], completion_ids.shape[1]
        device = completion_ids.device

        # The base class lays out output rows 1:1 with `inputs`. If that ever
        # fails to hold (e.g. an unexpected multi-process gather), fall back to
        # TRL's scalar advantages rather than misalign the tensor.
        if len(inputs) != batch_size:
            return base_adv

        rows: List[RowRollout] = []
        for inp in inputs:
            rows.append(
                RowRollout(
                    seat=int(inp.get("seat", 0)),
                    completion_len=len(inp.get("owner_mask", [])),
                    owner_mask=inp.get("owner_mask", []),
                    turn_end_positions=inp.get("turn_end_positions", []),
                    turn_rewards=inp.get("turn_rewards", []),
                )
            )

        lp_kwargs = cfg.length_penalty_kwargs()
        length_penalty = LengthPenaltySpec(**lp_kwargs) if lp_kwargs is not None else None

        advantages = compute_marshal_advantages(
            rows,
            seq_len,
            gamma=cfg.gamma,
            turn_level=cfg.turn_level_rewards,
            agent_specific=cfg.agent_specific_normalization,
            marshal_exact=cfg.marshal_exact,
            norm_mode=cfg.advantage_norm_mode,
            whiten_rewards=cfg.whiten_rewards,
            whiten_advantages=cfg.whiten_advantages,
            length_penalty=length_penalty,
            device=device,
            dtype=base_adv.dtype,
        )

        # Log per-seat pool stats so a run can confirm the seat split is live and
        # the two seats' advantage distributions actually differ.
        self._log_seat_stats(rows, advantages)
        if length_penalty is not None:
            self._log_length_penalty_stats(rows, length_penalty)
        return advantages

    def _log_length_penalty_stats(
        self, rows: List[RowRollout], spec: LengthPenaltySpec
    ) -> None:
        """Log turn lengths and the penalty they incur.

        Worth having because the penalty is silent when it does nothing: with
        MARSHAL's defaults every turn shorter than ``max_len`` scores exactly 0, so
        ``penalty/mean == 0`` and ``over_rate == 0`` is the signature of a threshold
        set too high to bite, which is otherwise indistinguishable from the flag
        not being wired up.
        """
        try:
            lengths: List[int] = []
            penalties: List[float] = []
            for row in rows:
                for length in turn_token_lengths(row.owner_mask, row.turn_end_positions):
                    if length <= 0:
                        continue  # a turn that generated nothing is not a data point
                    lengths.append(length)
                    penalties.append(spec.penalty_for(length))
            if not lengths:
                return
            mode = "train"
            metrics = self._metrics[mode]
            metrics["marshal/length_penalty/mean"].append(sum(penalties) / len(penalties))
            metrics["marshal/length_penalty/min"].append(min(penalties))
            metrics["marshal/turn_tokens/mean"].append(sum(lengths) / len(lengths))
            metrics["marshal/turn_tokens/max"].append(max(lengths))
            over = sum(1 for length in lengths if length > spec.max_len)
            metrics["marshal/length_penalty/over_rate"].append(over / len(lengths))
        except Exception:
            # Metrics logging must never break training.
            pass

    def _log_seat_stats(self, rows: List[RowRollout], advantages: torch.Tensor) -> None:
        try:
            mode = "train"
            seats = torch.tensor([r.seat for r in rows], device=advantages.device)
            for seat in torch.unique(seats):
                member = seats == seat
                # Mean over the rows of this seat, using the first-token advantage as
                # the representative trajectory advantage.
                vals = advantages[member]
                self._metrics[mode][f"marshal/seat_{int(seat)}/adv_mean"].append(
                    vals[vals != 0].mean().item() if (vals != 0).any() else 0.0
                )
                self._metrics[mode][f"marshal/seat_{int(seat)}/rows"].append(int(member.sum().item()))
        except Exception:
            # Metrics logging must never break training.
            pass

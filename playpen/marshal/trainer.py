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

import logging
from typing import Any, Callable, Dict, List, Optional

import torch
import trl

from playpen.marshal.advantage import RowRollout, compute_marshal_advantages
from playpen.marshal.config import MarshalConfig
from playpen.marshal.selfplay_agent import play_selfplay_episode
from playpen.marshal.selfplay_env import SelfPlayEnv, list_instance_indices
from playpen.marshal.turn_rewards import TurnRewardTracker, resolve_turn_reward_extractor

logger = logging.getLogger(__name__)


def parse_prompt(prompt: str) -> tuple[int, Optional[int]]:
    """Parse a dataset prompt into ``(instance_idx, seat)``.

    Two key shapes are understood, one per ``MarshalConfig.episode_pairing``:

    * ``"{instance_idx}::seat{n}"`` -> ``(idx, n)``. The seat is pinned by the key, so
      the row must come from its own episode (``episode_pairing="replay"``).
    * ``"{instance_idx}"`` -> ``(idx, None)``. The seat is not pinned; it is taken from
      the prompt's position within its run of identical copies, which is what lets one
      episode serve every seat (``episode_pairing="shared"``).
    """
    instance_part, separator, seat_part = prompt.partition("::seat")
    if not separator:
        return int(instance_part), None
    return int(instance_part), int(seat_part)


def build_selfplay_dataset(
    game_name: str,
    *,
    seats: Optional[List[int]] = None,
    num_players: Optional[int] = None,
    max_instances: Optional[int] = None,
    instances_filter: Optional[Callable[[dict], bool]] = None,
    episode_pairing: str = "shared",
):
    """Build a HuggingFace dataset of self-play prompts.

    Self-contained: instance indices come from the game's packaged ``instances.json``
    (the same indices ``SelfPlayEnv.reset`` looks up), so there is no dependency on an
    HF hub dataset. Indices rather than clembench game_ids, because game_ids are only
    unique *within* an experiment.

    ``episode_pairing`` selects the key shape, and must match what is passed to
    :func:`build_selfplay_rollout_func`:

    * ``"shared"`` -- one row per instance, ``"{idx}"``. TRL repeats each row
      ``num_generations`` times consecutively, and the rollout func serves each run of
      ``num_players`` copies from a single episode.
    * ``"replay"`` -- one row per (instance, seat), ``"{idx}::seat{n}"``; every prompt
      replays its own episode.

    The two shapes differ in length by a factor of ``num_players``, so an "epoch" covers
    the same instances either way but a different number of rows.
    """
    from datasets import Dataset

    from playpen.marshal.selfplay_env import resolve_game_spec

    instance_indices = list_instance_indices(game_name, instances_filter=instances_filter)
    if max_instances is not None:
        instance_indices = instance_indices[:max_instances]

    if episode_pairing == "shared":
        # The seat is positional, so it must not be baked into the key.
        return Dataset.from_dict({"prompt": [f"{idx}" for idx in instance_indices]})

    if seats is None:
        n = num_players if num_players is not None else int(resolve_game_spec(game_name).players)
        seats = list(range(n))

    prompts = [f"{idx}::seat{seat}" for idx in instance_indices for seat in seats]
    return Dataset.from_dict({"prompt": prompts})


def _paired_slot_start(prompts: List[str], index: int, num_players: int) -> Optional[int]:
    """Start of the run of ``num_players`` identical prompts containing ``index``.

    Returns ``None`` when ``index`` is not part of a complete, aligned run: either
    ``num_generations`` is not a multiple of ``num_players``, or a multi-process split
    landed a prompt's copies on different ranks. Callers then fall back to
    one-episode-per-prompt rather than pairing rows from different games.
    """
    if num_players < 2:
        return None
    base = index - (index % num_players)
    if base + num_players > len(prompts):
        return None
    prompt = prompts[index]
    if any(prompts[base + offset] != prompt for offset in range(num_players)):
        return None
    return base


def build_reward_func(config: MarshalConfig) -> Callable[..., List[float]]:
    """Return a TRL reward function that echoes the per-row terminal reward.

    Terminal rewards from clembench: SUCCESS +1, FAILURE 0, ABORTED -1. With
    ``turn_rewards`` on, the row's bounded per-episode shaping total is already
    included in the ``rewards`` value the rollout emits; this function passes through
    whatever it is handed.

    The reward is unshaped on both paths, so the only thing that differs between a
    ``--marshal`` run and a ``--no-marshal`` run is the advantage estimator. A group
    with no reward variance yields zero advantage on both; TRL reports how often that
    happens as ``frac_reward_zero_std``.
    """

    def reward_func(completions: List[str], **kwargs: Any) -> List[float]:
        rewards = kwargs.get("rewards")
        if rewards is None:
            return [0.0] * len(completions)
        return [float(r) for r in rewards]

    return reward_func


def build_selfplay_rollout_func(
    env: SelfPlayEnv,
    config: MarshalConfig,
    *,
    max_turns: int = 100,
    strip_think: bool = True,
) -> Callable[[List[str], "trl.GRPOTrainer"], Dict[str, list]]:
    """Return a ``rollout_func`` closing over a persistent :class:`SelfPlayEnv`.

    Satisfies TRL's 1:1 prompt->row contract. How an episode maps onto rows depends on
    ``config.episode_pairing``:

    * ``"shared"`` -- prompts are bare ``"{instance_idx}"`` keys, and TRL emits each
      one's ``num_generations`` copies consecutively (``RepeatSampler``). Each run of
      ``num_players`` consecutive copies is served by **one** episode, seat *k* taking
      the *k*-th copy, so both seats of a pair come from the same game and nothing is
      generated only to be discarded. Runs that cannot be paired (see
      :func:`_paired_slot_start`) fall back to the replay behavior for those indices.
    * ``"replay"`` -- prompts are ``"{instance_idx}::seat{n}"``; a fresh, complete
      episode is played per prompt and only the requested seat is kept, at the cost of
      ~num_players x generation.

    The returned dict carries, in addition to TRL's required keys, the extra fields
    the advantage override needs: ``env_mask`` (consumed by TRL as the gradient
    mask), ``owner_mask`` (a copy that survives into ``inputs``),
    ``turn_end_positions``, ``turn_rewards``, ``seat`` and the scalar ``rewards``.

    ``config.row_context_mode``, ``config.episode_pairing`` and ``config.turn_rewards``
    are read here (not behind ``config.enabled``) because rollout collection is shared
    by the MARSHAL path and the plain-GRPO baseline alike.

    With ``turn_rewards`` on, each episode gets a fresh
    :class:`~playpen.marshal.turn_rewards.TurnRewardTracker`, its bounded per-turn
    shaping is folded into ``turn_rewards``, and the episode total is also added to
    the scalar ``rewards`` -- the only thing the plain-GRPO path consumes.
    ``terminal_reward`` keeps the unshaped outcome so both arms stay comparable on
    game outcome alone.
    """
    paired = config.episode_pairing == "shared"
    num_players = int(env.num_players)
    # Both names read defensively: an env stub that only implements what a rollout
    # needs carries neither, and with turn rewards off the name is never consulted.
    game_name = getattr(env, "resolved_game_name", None) or getattr(env, "game_name", "")
    extractor, turn_reward_spec = resolve_turn_reward_extractor(game_name, config)
    # Component keys are fixed at build time and emitted for EVERY row (0.0 when a
    # component did not fire), so the extra-field lists TRL zips into `inputs` stay
    # rectangular across a batch.
    component_keys = tuple(extractor.components) if extractor is not None else ()

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
            "drifted": [],
        }
        # Only when the feature is on: with it off the rollout dict -- and therefore
        # everything TRL merges into `inputs` -- is exactly what it was before.
        if extractor is not None:
            out["terminal_reward"] = []
            out["turn_reward_sum"] = []
            out["turn_reward_clipped"] = []
            for key in component_keys:
                out[f"turn_reward_component_{key}"] = []

        def play(instance_idx: int):
            # A fresh tracker per episode: extractors carry episode state (wordle's
            # best closeness, the codenames board) that must not leak between games.
            tracker = (
                TurnRewardTracker(extractor, turn_reward_spec) if extractor is not None else None
            )
            return play_selfplay_episode(
                env,
                trainer,
                tokenizer,
                instance_idx,
                seed=instance_idx,
                max_turns=max_turns,
                strip_think=strip_think,
                row_context_mode=config.row_context_mode,
                turn_reward_tracker=tracker,
            )

        # slot start index -> that slot's episode, held only until its last seat is
        # consumed so the cache never grows past one episode.
        episodes: Dict[int, Dict[int, Any]] = {}

        for index, prompt in enumerate(prompts):
            instance_idx, seat = parse_prompt(prompt)
            if seat is None:
                # Bare key: the seat is positional within the run of identical copies.
                seat = index % num_players
                slot = _paired_slot_start(prompts, index, num_players) if paired else None
            else:
                # Key pins the seat, so this row cannot share a slot with its neighbours.
                slot = None

            if slot is None:
                rollouts = play(instance_idx)
            else:
                if slot not in episodes:
                    episodes[slot] = play(instance_idx)
                rollouts = episodes[slot]
                if index - slot == num_players - 1:
                    episodes.pop(slot, None)  # slot fully consumed

            row = rollouts.get(seat) or _empty_row(seat, pad_id)
            out["prompt_ids"].append(list(row.prompt_ids))
            out["completion_ids"].append(list(row.completion_ids))
            out["logprobs"].append(list(row.logprobs))
            out["env_mask"].append(list(row.owner_mask))
            out["owner_mask"].append(list(row.owner_mask))
            out["turn_end_positions"].append(list(row.turn_end_positions))
            out["turn_rewards"].append(list(row.turn_rewards))
            out["seat"].append(int(row.seat))
            shaping = float(getattr(row, "shaping_reward", 0.0))
            out["rewards"].append(float(row.terminal_reward) + shaping)
            if extractor is not None:
                out["terminal_reward"].append(float(row.terminal_reward))
                out["turn_reward_sum"].append(shaping)
                out["turn_reward_clipped"].append(bool(getattr(row, "shaping_clipped", False)))
                components = getattr(row, "shaping_components", None) or {}
                for key in component_keys:
                    out[f"turn_reward_component_{key}"].append(float(components.get(key, 0.0)))
            out["drifted"].append(bool(getattr(row, "drifted", False)))
        return out

    return rollout_func


def _empty_row(seat: int, pad_token_id: int = 0):
    """A minimal, non-crashing row for a seat that produced no usable trajectory.

    TRL's rollout contract requires one row per prompt with non-empty tensors, so
    we emit a single pad token marked as an *environment* token (``owner_mask=[0]``)
    with no turns: the loss mask zeroes it out and ``normalize_returns_by_seat``
    excludes rows without model tokens from the seat pools, so the placeholder
    carries no gradient and no statistics.
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
        # Health metrics are logged on BOTH paths: a censored row and a collapsed
        # importance-sampling ratio harm the plain-GRPO baseline exactly as much as
        # they harm the MARSHAL path, so gating them on `enabled` would hide the two
        # failures most likely to be mistaken for "the model just isn't learning".
        self._log_rollout_health(inputs, output)
        # Also on both paths: turn rewards are a rollout-collection feature, so they
        # are live for the plain-GRPO baseline too and must be observable there.
        self._log_turn_reward_stats(inputs)
        if not self.marshal_config.enabled:
            return output
        output["advantages"] = self._compute_marshal_advantages(inputs, output)
        return output

    # Warn once per process, not once per step: a per-step warning is noise that
    # gets scrolled past.
    _WARNED_IS_COLLAPSE = False
    _WARNED_DRIFT = False
    _WARNED_ADV_FALLBACK = False

    # Below this mean vLLM importance-sampling ratio the loss is scaled down by more
    # than 10x and the run is not really training. Well under the healthy range
    # observed on working runs (0.59-0.96) and well above a broken one (5e-8).
    IS_RATIO_WARN_THRESHOLD = 0.1

    def _log_rollout_health(self, inputs, output) -> None:
        """Log row-censoring and importance-sampling health, and warn when either fails.

        Two different ways a step produces no useful gradient, which look the same
        from the outside (flat reward, tiny grad_norm):

        * **Row censoring** -- ``_SeatBuilder.sync_context`` could not prove a row's
          owner mask, so the row was dropped for an inert placeholder. Gradient is
          lost for those rows only, and they still occupy a slot in the batch.
        * **Importance-sampling collapse** -- the row survived, but the sequence the
          loss scores is not the one vLLM sampled, so TRL's correction scales the
          whole row's loss by ``exp(sum of the per-token log-prob divergence)``.

        The drift counter cannot detect the second: drift is about whether the mask is
        provable, IS collapse about whether the context is identical. TRL computes the
        ratio but does not complain about it, so read it back out of ``output``.
        """
        try:
            metrics = self._metrics["train"]

            drifted = [bool(inp.get("drifted", False)) for inp in inputs]
            owner_masks = [inp.get("owner_mask", []) for inp in inputs]
            inert = [not any(m == 1 for m in mask) for mask in owner_masks]
            n = len(inputs)
            if n:
                metrics["marshal/rows/drift_count"].append(float(sum(drifted)))
                metrics["marshal/rows/drift_rate"].append(sum(drifted) / n)
                # Inert-but-not-drifted == a seat that never got a turn. Separating
                # the two is the point; a game that ends early is not censoring.
                metrics["marshal/rows/placeholder_rate"].append(sum(inert) / n)
                metrics["marshal/rows/idle_seat_rate"].append(
                    sum(1 for i, d in zip(inert, drifted) if i and not d) / n
                )
            if any(drifted) and not MarshalGRPOTrainer._WARNED_DRIFT:
                MarshalGRPOTrainer._WARNED_DRIFT = True
                logger.warning(
                    "MARSHAL: %d/%d rows dropped for re-tokenization drift this step. "
                    "Those rows carry no gradient and are excluded from the advantage "
                    "pools, so the batch is smaller than it looks. Watch "
                    "marshal/rows/drift_rate -- a rate that grows with training means "
                    "the policy is drifting toward generations that do not round-trip.",
                    sum(drifted),
                    n,
                )

            ratio = output.get("importance_sampling_ratio")
            if ratio is not None and ratio.numel():
                mean_ratio = ratio.float().mean().item()
                metrics["marshal/is_ratio/mean"].append(mean_ratio)
                if (
                    mean_ratio < self.IS_RATIO_WARN_THRESHOLD
                    and not MarshalGRPOTrainer._WARNED_IS_COLLAPSE
                ):
                    MarshalGRPOTrainer._WARNED_IS_COLLAPSE = True
                    logger.warning(
                        "MARSHAL: mean vLLM importance-sampling ratio is %.3g (< %.3g). "
                        "The policy loss is being scaled down by ~%.0fx, so grad_norm "
                        "will be near zero and the run will not learn -- this is NOT a "
                        "reward or advantage problem. It means the trained token "
                        "sequence differs from the one vLLM sampled: check "
                        "row_context_mode='exact' and that the tokenizer's chat "
                        "template round-trips. See sampling/sampling_logp_difference/mean "
                        "(healthy is ~0.03 nats/token; ~0.6 indicates a broken context).",
                        mean_ratio,
                        self.IS_RATIO_WARN_THRESHOLD,
                        1.0 / max(mean_ratio, 1e-12),
                    )
        except Exception:
            # Metrics logging must never break training.
            pass

    #: Prefix under which the rollout emits one column per turn-reward component.
    _COMPONENT_PREFIX = "turn_reward_component_"

    def _log_turn_reward_stats(self, inputs) -> None:
        """Log the dense per-turn reward channel: magnitude, saturation, components.

        Nothing is logged when the feature is off. The three that matter for
        calibration:

        * ``turn_rewards/sum_mean`` / ``sum_abs_mean`` -- how much shaping an episode
          accumulates, against a ``+-1`` outcome. ~0 means the signal is inert for
          this game; the component columns say whether that is the game or a bug.
        * ``turn_rewards/budget_clip_rate`` -- how often the per-episode budget had to
          rescale. Near 1.0 means the budget, not the game, sets the magnitude.
        * ``turn_rewards/terminal_mean`` -- the unshaped outcome. TRL's ``reward``
          metric includes the shaping once this is on, so this is the column that
          stays comparable with a turn-rewards-off arm.
        """
        try:
            # The rollout emits this column only when turn rewards are on, so its
            # absence -- not a zero total -- is what means "feature off". An episode
            # that genuinely scored 0 shaping still gets logged as a 0.
            if not inputs or "turn_reward_sum" not in inputs[0]:
                return
            metrics = self._metrics["train"]
            sums = [float(inp.get("turn_reward_sum", 0.0)) for inp in inputs]
            n = len(sums)
            metrics["marshal/turn_rewards/sum_mean"].append(sum(sums) / n)
            metrics["marshal/turn_rewards/sum_abs_mean"].append(
                sum(abs(value) for value in sums) / n
            )
            metrics["marshal/turn_rewards/sum_max_abs"].append(max(abs(value) for value in sums))
            clipped = [bool(inp.get("turn_reward_clipped", False)) for inp in inputs]
            metrics["marshal/turn_rewards/budget_clip_rate"].append(sum(clipped) / n)
            metrics["marshal/turn_rewards/nonzero_rate"].append(
                sum(1 for value in sums if value) / n
            )
            terminal = [float(inp.get("terminal_reward", 0.0)) for inp in inputs]
            metrics["marshal/turn_rewards/terminal_mean"].append(sum(terminal) / n)
            for key in {k for inp in inputs for k in inp if k.startswith(self._COMPONENT_PREFIX)}:
                name = key[len(self._COMPONENT_PREFIX):]
                values = [float(inp.get(key, 0.0)) for inp in inputs]
                metrics[f"marshal/turn_rewards/component/{name}"].append(sum(values) / n)
        except Exception:
            # Metrics logging must never break training.
            pass

    def _compute_marshal_advantages(self, inputs, output) -> torch.Tensor:
        cfg = self.marshal_config
        base_adv = output["advantages"]
        completion_ids = output["completion_ids"]
        batch_size, seq_len = completion_ids.shape[0], completion_ids.shape[1]
        device = completion_ids.device

        # The base class lays out output rows 1:1 with `inputs`. If that ever
        # fails to hold (e.g. an unexpected multi-process gather), fall back to
        # TRL's scalar advantages rather than misalign the tensor.
        #
        # The fallback silently turns a MARSHAL run into a plain-GRPO run, so it is
        # reported twice: a once-per-process warning and a per-step metric (the
        # warning can scroll past a long log, the metric cannot).
        if len(inputs) != batch_size:
            self._log_advantage_fallback(len(inputs), batch_size)
            return base_adv
        self._log_advantage_fallback(len(inputs), batch_size, fired=False)

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

        advantages = compute_marshal_advantages(
            rows,
            seq_len,
            gamma=cfg.gamma,
            agent_specific=cfg.agent_specific_normalization,
            norm_mode=cfg.advantage_norm_mode,
            whiten_rewards=cfg.whiten_rewards,
            whiten_advantages=cfg.whiten_advantages,
            device=device,
            dtype=base_adv.dtype,
        )

        # Log per-seat pool stats so a run can confirm the seat split is live and
        # the two seats' advantage distributions actually differ.
        self._log_seat_stats(rows, advantages)
        self._relog_advantages(advantages)
        return advantages

    def _log_advantage_fallback(self, n_inputs: int, batch_size: int, *, fired: bool = True) -> None:
        """Record whether the MARSHAL advantage override ran or fell back to TRL's.

        ``marshal/advantage/fallback_rate`` is appended on every step -- 0.0 when the
        override ran, 1.0 when it did not -- so the metric distinguishes "never fell
        back" from "never emitted". Anything but 0.0 means those steps trained on
        TRL's scalar group-relative advantages, i.e. plain GRPO.

        The warning is outside the try/except on purpose: it reports that the run is
        not doing what it says, which matters more than the metric beside it.
        """
        if fired and not MarshalGRPOTrainer._WARNED_ADV_FALLBACK:
            MarshalGRPOTrainer._WARNED_ADV_FALLBACK = True
            logger.warning(
                "MARSHAL: advantage override SKIPPED -- len(inputs)=%d but the batch has "
                "%d rows, so the (B, T) tensor could not be aligned. These steps train on "
                "TRL's scalar group-relative advantages, i.e. plain GRPO: the run is NOT "
                "doing MARSHAL even though it is configured for it. Watch "
                "marshal/advantage/fallback_rate.",
                n_inputs,
                batch_size,
            )
        try:
            self._metrics["train"]["marshal/advantage/fallback_rate"].append(
                1.0 if fired else 0.0
            )
        except Exception:
            # Metrics logging must never break training; the warning above already fired.
            pass

    def _relog_advantages(self, advantages: torch.Tensor) -> None:
        """Point the logged ``advantage`` column at the advantages we train on.

        The base class snapshots its own scalar group-relative advantages into
        ``self._logs["advantages"]`` before this subclass replaces them, so left alone
        every ``completions_*.parquet`` would record a number the model never sees.

        Each row is summarized by the mean advantage over its model tokens. Skipped
        under multi-process training, where ``_logs`` holds every rank's gathered
        advantages but ``advantages`` is only this rank's slice.
        """
        try:
            if self.accelerator.num_processes != 1:
                return
            logged = self._logs["advantages"]
            rows = advantages.shape[0]
            if len(logged) < rows:
                return
            mask = (advantages != 0).to(advantages.dtype)
            denom = mask.sum(dim=1).clamp(min=1.0)
            per_row = ((advantages * mask).sum(dim=1) / denom).tolist()
            for offset, value in enumerate(per_row):
                logged[len(logged) - rows + offset] = value
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

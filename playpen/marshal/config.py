"""Configuration for the MARSHAL-style self-play training pipeline.

This module intentionally depends only on the standard library (plus an optional
PyYAML import for ``from_yaml``). It must remain importable *without* torch, trl,
or vllm so that the on/off switch can be read in any context.

The MARSHAL paper (arXiv:2510.15414) contributes two ideas on top of a REINFORCE
/ GRPO-style objective:

  1. a *turn-level advantage estimator* (rewards attributed at each turn boundary
     and turned into per-token returns via a backward cumulative sum), and
  2. *agent-specific advantage normalization* (advantages pooled and normalized
     separately per self-play seat/role instead of pooling both together).

``MarshalConfig`` is the single source of truth for whether those behaviors run
and how. See ``playpen/marshal/advantage.py`` for the math it drives and
``playpen/marshal/trainer.py`` for where it is consulted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict


# The two knobs below are string enums rather than bools so their intent is
# self-documenting in a YAML file and so a third mode could be added later.
ADVANTAGE_NORM_MODES = ("mean", "mean_std")
FIDELITY_MODES = ("paper_correct", "marshal_exact")
ROW_CONTEXT_MODES = ("exact", "spliced")
EPISODE_PAIRING_MODES = ("shared", "replay")
# Re-exported from turn_rewards so argparse `choices=` and the validation below
# cannot drift from each other. turn_rewards.py is stdlib-only, like this module.
TURN_REWARD_SOURCES = ("auto", "game", "generic")

# Above this episode-shaping budget, turn rewards can swing an episode's total by
# more than the gap between two clembench outcomes (SUCCESS +1 / FAILURE 0 /
# ABORTED -1, i.e. a gap of 1.0), so a well-shaped loss could out-score a bare win.
# See MarshalConfig.turn_reward_budget.
TURN_REWARD_SAFE_BUDGET = 0.5

# Same idea for the length penalty. That term is one-sided (always <= 0), so its
# own worst-case swing between two episodes is 1 x budget rather than 2 x, and it
# would stay safe on its own up to 1.0. The lower bound here leaves headroom for
# turn_rewards to be on at the same time (0.5 + 2 x 0.3 = 1.1 is already too much,
# so in practice keep length_penalty_budget + 2 x turn_reward_budget < 1.0).
# See MarshalConfig.length_penalty_budget.
LENGTH_PENALTY_SAFE_BUDGET = 0.5

# The fields of MARSHAL's original threshold-based length penalty, mapped to the
# values they shipped with. They are still accepted (so pinned resume configs,
# cluster presets and old manifests keep loading) but no longer do anything --
# see MarshalConfig.length_penalty_coef. MarshalConfig.legacy_length_penalty_values
# reports the ones a config actually set, so a run can say out loud that they are
# being ignored rather than silently training under a penalty nobody configured.
LEGACY_LENGTH_PENALTY_DEFAULTS = {
    "length_penalty_coef": 0.5,
    "length_penalty_bonus": 0.0,
    "length_penalty_min_len": 11,
    "length_penalty_max_len": 2048,
    "length_penalty_offset": 1.0,
}


@dataclass
class MarshalConfig:
    """Master switch + sub-flags for the MARSHAL self-play behavior.

    Attributes:
        enabled: The single on/off switch. When ``False``, the trainer falls back
            to stock TRL GRPO behavior: one scalar terminal reward per rollout row
            fed through TRL's own group-relative advantage normalization. None of
            the other fields have any effect in that case.
        agent_specific_normalization: When ``True`` (and ``enabled``), advantages
            are pooled and normalized *per seat* (player_0 rows separately from
            player_1 rows) across the batch. When ``False``, the turn-level
            advantages are still computed but pooled batch-wide -- useful as an
            ablation isolating the turn-level-credit contribution from the
            seat-specific-pooling contribution.
        turn_level_rewards: When ``True``, rewards are kept at each turn boundary
            and turned into per-token returns via a backward cumulative sum (the
            turn-level advantage estimator). When ``False``, all of a seat's turn
            rewards are summed into a single terminal scalar placed at the last
            turn boundary (mirrors MARSHAL's ``use_turn_scores=False``). For
            clembench's default sparse, terminal-only rewards the two are almost
            equivalent, but the flag matters if a game supplies denser rewards.
        advantage_norm_mode: ``"mean"`` mean-centers each pool; ``"mean_std"``
            z-scores it (subtract mean, divide by std).
        gamma: Discount for the backward cumulative return. MARSHAL uses 1.0.
        whiten_rewards: When ``True``, z-score the token-level reward field over
            all model-token positions batch-wide (both seats pooled) *before* the
            backward cumulative sum -- mirrors ROLL's ``whiten_rewards: true``,
            set in every shipped MARSHAL ``*_selfplay.yaml``. Not part of the
            paper's algorithm (hence default ``False``). Note: this densifies a
            sparse reward (every model token picks up ``-mean/std``), adding a
            response-length-dependent component to the returns.
        whiten_advantages: When ``True``, z-score the final advantages over all
            model-token positions batch-wide *after* the per-seat normalization --
            mirrors ROLL's ``whiten_advantages: true`` (also in every shipped
            selfplay YAML). Standard PPO-style scale stabilization; default
            ``False`` to preserve the port's previous behavior.
        fidelity_mode: ``"paper_correct"`` (default) implements the algorithm the
            paper describes -- occurrence-weighted (frequency-weighted) pooling
            over trajectory returns and no biasing pre-sum reward normalization.
            ``"marshal_exact"`` reproduces MARSHAL's *shipped code*, including two
            documented departures from its own paper: (a) a pre-sum reward
            normalization pass that introduces a trajectory-length-dependent bias
            (per-seat when ``agent_specific_normalization`` is set, matching
            ``separate_norm_for_selfplay: true`` in every shipped ``*_selfplay.yaml``),
            and (b) ``normalize_unique_values_by_player`` normalizing only the set
            of *distinct* return values (equal-weighting rare and common outcomes)
            rather than an occurrence-weighted mean. Use this for close reproduction
            of MARSHAL's own runs. The pre-sum mean is taken over all
            ``turn_end_positions`` (zero-reward boundaries included), matching
            MARSHAL's ``score_normalize`` mask; see
            ``advantage._marshal_pre_sum_normalize``. Departure (b) can be switched
            off on its own with ``marshal_exact_unique_pooling``. Consulted only when
            ``enabled=True``.
        marshal_exact_unique_pooling: Whether ``marshal_exact``'s departure (b) --
            the ``torch.unique`` distinct-value pooling in
            ``advantage._pool_offset_scale`` -- is applied. Default ``True``, i.e.
            ``marshal_exact`` behaves exactly as it always has.

            Set ``False`` to run ``marshal_exact``'s pre-sum reward normalization
            (departure (a)) with ``paper_correct``'s occurrence-weighted pooling,
            which isolates the two departures from each other in an ablation.
            Nothing else about the mode moves.

            **Inert unless ``fidelity_mode == "marshal_exact"``** -- ``paper_correct``
            never pools over distinct values in the first place, so this field is a
            no-op there rather than a way to turn unique pooling *on*.

            Worth knowing before flipping it: on clembench, distinct-value pooling
            makes the baseline the midpoint of *whichever outcomes occur* rather than
            the empirical mean, because only ``{-1, 0, +1}`` are ever seen. That
            matters most in small or lopsided pools -- e.g. 9 wins and 1 loss give a
            baseline of ``0.0`` under unique pooling but ``0.8`` under
            occurrence-weighted pooling. Note also that any per-turn signal which
            makes returns continuous (``length_penalty``, ``turn_rewards``) already
            renders nearly every return distinct, at which point unique pooling
            collapses into occurrence-weighted pooling on its own and this flag
            changes little.
        dr_grpo: When ``True``, run with the Dr. GRPO (arXiv:2503.20783) recipe by
            configuring TRL's underlying loss/scaling: ``loss_type="dr_grpo"`` (the
            summed loss is normalized by the constant ``max_completion_length``
            instead of realized length -- removes GRPO's response-level *length*
            bias) and ``scale_rewards="none"`` (advantages are not divided by a std
            -- removes the *difficulty* bias). See :meth:`trl_grpo_overrides`.

            Unlike the other sub-flags, ``dr_grpo`` takes effect *regardless of*
            ``enabled``, because it configures the shared TRL loss/scaling recipe
            used by both the MARSHAL path and the plain-GRPO baseline. It is applied
            by the launch script (``examples/marshal/train_selfplay.py``), not by
            ``advantage.py`` -- MARSHAL's advantage math is left untouched.

            Interaction with MARSHAL's own normalization (MARSHAL path only): MARSHAL
            already avoids std division by default (``advantage_norm_mode="mean"``),
            so Dr. GRPO composes with zero conflict and MARSHAL's headline features
            (turn-level credit, per-seat pooling, gamma, batch-wide whitening) are
            all preserved. The one exception is the opt-in per-seat z-score
            (``advantage_norm_mode="mean_std"``): :meth:`reconcile_for_dr_grpo`
            aligns it back to ``"mean"`` (dropping only the per-seat std *divisor*,
            keeping per-seat *pooling*), since a per-group std is exactly the bias
            Dr. GRPO removes. Batch-wide ``whiten_advantages`` is left alone -- the
            Dr. GRPO paper explicitly permits batch-level normalization.
        length_penalty: When ``True`` (and ``enabled``), charge every turn a small
            per-token cost, added to that turn's reward before any normalization:
            ``-length_penalty_per_token * generated_tokens``. There is **no
            threshold** -- the first token costs the same as the thousandth, and
            there is no length below which a turn is free. See
            ``advantage.LengthPenaltySpec`` for the formula and
            ``length_penalty_budget`` for why it cannot outweigh the game outcome.

            Default ``False``, and that default is the MARSHAL-faithful setting: MARSHAL
            applies its length penalty *only* to its board-game envs and explicitly skips
            it for free-text ones (``env_manager.py:470-480`` guards it behind
            ``not isinstance(env, BaseLanguageBasedEnv)``, commented "No MARSHAL-imposed
            reward shaping for free-text envs (e.g. Playpen/clembench)"). Turning it on
            for a clembench game is therefore a deliberate divergence -- useful as gentle
            pressure against a model that pads every turn, but a run using it is not a
            MARSHAL reproduction and should be reported as such.
        length_penalty_per_token: Reward charged per generated token. Non-negative;
            the minus sign is applied in ``LengthPenaltySpec``, so ``2e-5`` (the
            default) means ``-2e-5`` per token, i.e. ``-0.02`` per 1000 tokens.

            Calibrate against the ``+-1`` outcome, not against a single turn: a
            500-token turn costs ``-0.01`` and a 10-turn episode of them ``-0.10``,
            about a tenth of the gap between winning and losing. It is meant to be a
            tie-breaker among equally-scoring episodes, so prefer the smallest value
            that still moves ``marshal/length_penalty/episode_total_mean`` off zero.
            ``0`` makes the penalty inert without turning the flag off.
        length_penalty_budget: Hard cap on ``|sum of one seat's length penalty over
            one episode|``, applied as a proportional rescale of that row's per-turn
            penalties (so relative charge between turns survives and no sign flips).
            This is what makes "cannot overwhelm the terminal reward" a *guarantee*
            rather than a hope, and it is the reason no per-game calibration is
            needed: the penalty is charged per turn and the backward cumulative
            return sums a seat's turns, so a 20-turn game would otherwise accumulate
            four times what a 5-turn game does.

            The smallest gap between two distinct clembench outcomes is 1.0 (SUCCESS
            ``+1`` / FAILURE ``0`` / ABORTED ``-1``) and the penalty is one-sided, so
            any ``0 < budget < 1.0`` guarantees it can never reorder two episodes
            that ended differently -- only rank episodes *within* an outcome class.
            Default ``0.1``. ``0`` disables the cap and gives up the guarantee.
            Values ``>= 0.5`` (``LENGTH_PENALTY_SAFE_BUDGET``) are allowed but warned
            about by the launch script, because they leave no headroom for
            ``turn_rewards`` (whose own swing is ``2 * turn_reward_budget``) to be on
            at the same time; keep ``length_penalty_budget + 2 * turn_reward_budget``
            under 1.0 when both channels run.

            Like ``turn_reward_budget`` it is a safety net, not the operating point:
            tune ``length_penalty_per_token`` so a typical episode lands under it and
            watch ``marshal/length_penalty/budget_clip_rate``.
        length_penalty_coef: DEPRECATED and inert, along with ``length_penalty_bonus``,
            ``length_penalty_min_len``, ``length_penalty_max_len`` and
            ``length_penalty_offset``. These five parameterized the previous,
            threshold-based penalty (a direct port of MARSHAL's Kimi-1.5-style
            ``compute_length_penalty``: exactly 0 below ``max_len``, linear beyond it).
            That shape is gone -- see ``advantage.LengthPenaltySpec`` for why -- and the
            fields survive only so existing YAMLs, pinned resume configs, cluster presets
            and old manifests keep loading unchanged.

            They are **not** validated and have **no** effect. A config that sets any of
            them to a non-default value is reported by
            :meth:`legacy_length_penalty_values` and warned about at startup, so a run
            cannot quietly believe it is using a calibration that no longer exists.
        row_context_mode: How a multi-turn rollout row's token sequence is assembled.
            Like ``dr_grpo``, this takes effect *regardless of* ``enabled``: it
            governs rollout collection, which both the MARSHAL path and the plain-GRPO
            baseline share.

            ``"exact"`` (default) -- the row's ``prompt_ids + completion_ids`` reproduce,
            token for token, the context each turn was actually generated under. Each
            turn's generation prompt is the row so far plus the chat template's own
            per-turn scaffolding, and the environment-feedback span is taken from the
            token ids vLLM reports for that prompt, so no reconstruction can drift.

            ``"spliced"`` -- the pre-fix behavior, kept only to reproduce earlier runs.
            The row is the turn-1 prompt followed by every generation concatenated with
            the raw environment text (``"\\n\\n" + observation``) between them, while
            generation itself used the full chat template. From turn 2 onward the
            trained sequence is therefore NOT the sequence the policy generated under:
            the chat scaffolding (``<|im_end|>``/``<|im_start|>`` turn boundaries), the
            ``/no_think`` marker and the per-turn empty ``<think>`` block are all
            missing. Recomputed log-probs then diverge from the sampler's by ~0.6
            nats/token, and TRL's vLLM importance-sampling correction (on by default,
            ``vllm_importance_sampling_mode="sequence_mask"``) multiplies the whole
            row's loss by ``exp(sum of that divergence)`` -- observed at 1e-6 or below
            for every row, i.e. no gradient at all. Do not use for new runs.
        episode_pairing: Whether the seats of one episode share that episode. Like
            ``row_context_mode``, this governs rollout collection and so applies
            regardless of ``enabled``.

            ``"shared"`` (default) -- MARSHAL's own arrangement: one dataset row per
            game instance, and each run of ``num_players`` consecutive copies of that
            prompt is served by a **single** episode, seat *k* taking the *k*-th copy.
            Both seats' rows therefore come from the same game, which is what makes a
            per-seat baseline compare like with like. It also halves generation cost --
            no seat is generated and discarded -- and guarantees every advantage pool
            holds both seats in equal number.

            ``"replay"`` -- the pre-fix arrangement: one dataset row per (instance,
            seat), and every prompt replays a whole fresh episode keeping only the
            requested seat. The other seat's generation is computed and thrown away, so
            rollout costs ~num_players x more, and the two seats of a "pair" come from
            different games. Kept to reproduce earlier runs.

            Note for ``"shared"``: ``num_generations`` should be a multiple of the
            game's player count so the runs of identical prompts divide evenly into
            episodes. When a run cannot be paired (odd ``num_generations``, or a
            multi-process split that straddles a prompt's copies) that index silently
            falls back to ``"replay"`` behavior rather than failing.
        sampling_top_p: Nucleus cutoff for generation, mapped onto ``GRPOConfig.top_p``.
            ``1.0`` disables it (TRL's default). Like ``dr_grpo``, this applies
            *regardless of* ``enabled`` -- it configures generation, which the MARSHAL
            path and the plain-GRPO baseline share.

            Default ``0.95``, which is a change from TRL's untruncated default, for a
            measured reason. With ``KL_BETA=0`` the policy's entropy collapses over
            training (0.091 -> 0.029 nats on guesswhat/Qwen3-4B by step 700). At
            ``top_p=1.0``/``top_k=0`` the sampler still draws from the *whole* vocabulary,
            so it occasionally emits a far-tail token. Down there ``log p`` is enormously
            sensitive to a small bf16 logit difference, and vLLM and the training model
            disagree by many nats on that one token (observed max 16.0). Because TRL's
            ``vllm_importance_sampling_mode`` defaults to ``"sequence_mask"``, that
            per-token divergence is *summed* over the row and exponentiated, so a single
            tail token annihilates the whole row's gradient: the worst row's ratio fell
            to 1.6e-14 (and to exactly 0 -- float underflow, not the ``>3.0`` cap, which
            never fired) while grad_norm decayed 7.7x over the run.

            Truncating the tail removes those draws. It does **not** bias the importance
            ratio: vLLM computes reported logprobs from the raw logits *before* applying
            top-p/top-k (``vllm/v1/sample/sampler.py:74-86``, ``logprobs_mode`` defaults
            to ``"raw_logprobs"``), so both sides of the ratio stay full-distribution
            quantities and only the *sampled* support narrows.

            This is a mitigation, not the cure -- the root cause is running without a KL
            term, which MARSHAL's own shipped ``*_selfplay.yaml`` never does
            (``use_kl_loss: true``, ``kl_loss_coef: 0.20``).
        turn_rewards: Master switch for the *dense per-turn reward* channel
            (``playpen/marshal/turn_rewards.py``). Default ``False``, and with it off
            nothing about a run changes: rollouts collect the same terminal-only
            ``+1/0/-1`` team reward they always have, and none of the
            ``turn_reward_*`` fields below have any effect.

            When ``True``, after every turn the game's live ``GameState`` is read for
            a small per-turn signal (wordle: how much of the target the guess
            revealed; codenames: which words the board gave up; taboo/guesswhat/dond:
            which seat's turn broke a rule), scaled by ``turn_reward_scale``, capped
            by ``turn_reward_budget``, and *added to that turn's* reward before any
            MARSHAL normalization.

            Like ``row_context_mode`` and ``episode_pairing``, this applies
            **regardless of** ``enabled``, because it governs rollout collection,
            which the MARSHAL path and the plain-GRPO baseline share. On the MARSHAL
            path the shaping lands at each turn boundary and gets full turn-level
            credit; on the plain-GRPO path TRL only consumes one scalar per row, so
            the episode's shaping total is added to it. Either way the *unshaped*
            outcome is still logged separately as
            ``marshal/turn_rewards/terminal_mean``, so an arm with turn rewards on
            stays comparable with one that has them off.

            Two things worth knowing before turning it on:

            * It is what makes ``agent_specific_normalization`` do real work here.
              clembench gives both self-play seats the *same* team outcome, so
              per-seat pooling currently mean-centers two identical distributions.
              A per-turn reward is attributed to the seat that acted, which is the
              only mechanism in this pipeline that makes the seats' advantages differ.
            * Prefer ``fidelity_mode="paper_correct"``. Under ``marshal_exact`` the
              pre-sum normalization subtracts a mean over *all* turn-boundary slots,
              which already biases by turn count; a denser reward field makes that
              term larger, not smaller.
        turn_reward_source: Which extractor supplies the signal. ``"auto"`` (default)
            uses the game's own extractor when one is registered and falls back to
            generic format compliance otherwise; ``"game"`` refuses the fallback
            (turn rewards are simply off for an unregistered game, with a warning),
            which is the honest setting when a result depends on the game-specific
            signal; ``"generic"`` forces format compliance even for a game that has
            a richer extractor, as a "compliance only" ablation.
        turn_reward_scale: Multiplier on each turn's combined signal, which the
            extractors normalize to ``[-1, 1]``. So this is the *maximum* magnitude a
            single turn can contribute. Default ``0.05``: against a ``+-1`` outcome,
            a seat would need 20 maximally-shaped turns to accumulate the outcome's
            magnitude, and no shipped preset plays that many.
        turn_reward_budget: Hard cap on ``|sum of one seat's shaping over one
            episode|``, applied as a proportional rescale of the whole episode's
            shaping vector (so relative credit between turns survives and no sign
            flips). This is the knob that makes "cannot overwhelm the terminal
            reward" a *guarantee* rather than a hope: the backward cumulative return
            means the episode total is what competes with the outcome, so with
            ``budget < 0.5`` the worst-case swing between two episodes is
            ``2 * budget < 1.0``, smaller than the gap between any two distinct
            clembench outcomes -- shaping can reorder episodes *within* an outcome
            class but never across one. Default ``0.3``. ``0`` disables the cap and
            gives up that guarantee. Values ``>= 0.5``
            (``TURN_REWARD_SAFE_BUDGET``) are allowed but warned about by the launch
            script.

            It is a safety net, not the operating point -- tune ``turn_reward_scale``
            so a typical episode lands under it, and watch
            ``marshal/turn_rewards/budget_clip_rate``: near 1.0 means the cap binds
            every episode, so only the *shape* of the signal survives and its
            magnitude is being set by this field instead of by the game.
        turn_reward_components: Comma-separated allowlist of component names to keep
            (e.g. ``"closeness"`` to run wordle's progress signal without its format
            penalty). Empty (default) keeps every component the extractor emits.
            Unknown names are rejected at build time against the resolved extractor's
            component list, so a typo cannot silently disable the signal.
        sampling_top_k: Top-k cutoff for generation, mapped onto ``GRPOConfig.top_k``.
            ``0`` disables it (TRL's default). Default ``50``.

            Kept alongside ``sampling_top_p`` because the two fail differently on a
            collapsed policy. At a near-deterministic position (top token p ~ 0.998, which
            is what a mean entropy of 0.029 nats implies) ``top_p=0.95`` keeps only the
            argmax -- correct here, since that position carries no real choice -- whereas
            ``top_k=50`` still admits ranks 2..50. Conversely at a genuinely uncertain
            position ``top_k`` bounds the support where ``top_p`` may keep a long tail.
            Applying both bounds the tail at every position.

            Set both to their neutral values (``sampling_top_p: 1.0``,
            ``sampling_top_k: 0``), or pass ``--no-sampling-truncation``, to reproduce
            pre-2026-07-28 generation exactly: :meth:`trl_sampling_overrides` then returns
            an empty dict and no key is added to ``GRPOConfig``.
    """

    enabled: bool = True
    agent_specific_normalization: bool = True
    turn_level_rewards: bool = True
    advantage_norm_mode: str = "mean"
    gamma: float = 1.0
    fidelity_mode: str = "paper_correct"
    marshal_exact_unique_pooling: bool = True
    whiten_rewards: bool = False
    whiten_advantages: bool = False
    dr_grpo: bool = False
    length_penalty: bool = False
    length_penalty_per_token: float = 2e-5
    length_penalty_budget: float = 0.1
    # Inert legacy fields of the old threshold-based penalty; kept loadable only.
    # See the class docstring under `length_penalty_coef`.
    length_penalty_coef: float = 0.5
    length_penalty_bonus: float = 0.0
    length_penalty_min_len: int = 11
    length_penalty_max_len: int = 2048
    length_penalty_offset: float = 1.0
    row_context_mode: str = "exact"
    episode_pairing: str = "shared"
    sampling_top_p: float = 0.95
    sampling_top_k: int = 50
    turn_rewards: bool = False
    turn_reward_source: str = "auto"
    turn_reward_scale: float = 0.05
    turn_reward_budget: float = 0.3
    turn_reward_components: str = ""

    def __post_init__(self) -> None:
        if self.advantage_norm_mode not in ADVANTAGE_NORM_MODES:
            raise ValueError(
                f"advantage_norm_mode must be one of {ADVANTAGE_NORM_MODES}, "
                f"got {self.advantage_norm_mode!r}"
            )
        if self.fidelity_mode not in FIDELITY_MODES:
            raise ValueError(
                f"fidelity_mode must be one of {FIDELITY_MODES}, got {self.fidelity_mode!r}"
            )
        if self.row_context_mode not in ROW_CONTEXT_MODES:
            raise ValueError(
                f"row_context_mode must be one of {ROW_CONTEXT_MODES}, "
                f"got {self.row_context_mode!r}"
            )
        if self.episode_pairing not in EPISODE_PAIRING_MODES:
            raise ValueError(
                f"episode_pairing must be one of {EPISODE_PAIRING_MODES}, "
                f"got {self.episode_pairing!r}"
            )
        self.sampling_top_p = float(self.sampling_top_p)
        self.sampling_top_k = int(self.sampling_top_k)
        # vLLM reads top_p as a probability mass and top_k as a count. Out-of-range
        # values are accepted silently there and quietly change (or disable) the
        # truncation, so reject them here where the message can say what was meant.
        if not 0.0 < self.sampling_top_p <= 1.0:
            raise ValueError(
                f"sampling_top_p must be in (0.0, 1.0], got {self.sampling_top_p}. "
                "Use 1.0 to disable nucleus truncation."
            )
        if self.sampling_top_k < 0:
            raise ValueError(
                f"sampling_top_k must be >= 0, got {self.sampling_top_k}. "
                "Use 0 to disable top-k truncation."
            )
        self.gamma = float(self.gamma)
        self.length_penalty_per_token = float(self.length_penalty_per_token)
        self.length_penalty_budget = float(self.length_penalty_budget)
        # A negative rate would turn the penalty into a *reward* for length, and a
        # negative budget would make the rescale factor negative and do the same.
        # Neither is a thing anyone means. (Same rule as turn_reward_scale/budget.)
        if self.length_penalty_per_token < 0.0:
            raise ValueError(
                f"length_penalty_per_token must be >= 0, got "
                f"{self.length_penalty_per_token}. The minus sign is applied by "
                "LengthPenaltySpec; use 0 to make the penalty inert, or "
                "length_penalty: false to switch it off."
            )
        if self.length_penalty_budget < 0.0:
            raise ValueError(
                f"length_penalty_budget must be >= 0, got {self.length_penalty_budget}. "
                "Use 0 to remove the per-episode cap (which also gives up the guarantee "
                "that the penalty cannot outweigh the game outcome)."
            )
        # The legacy threshold fields are coerced so an old YAML's types are still
        # normalized, but deliberately NOT validated: they are inert, so rejecting a
        # combination they used to reject would only block a config that now runs fine.
        self.length_penalty_coef = float(self.length_penalty_coef)
        self.length_penalty_bonus = float(self.length_penalty_bonus)
        self.length_penalty_min_len = int(self.length_penalty_min_len)
        self.length_penalty_max_len = int(self.length_penalty_max_len)
        self.length_penalty_offset = float(self.length_penalty_offset)
        if self.turn_reward_source not in TURN_REWARD_SOURCES:
            raise ValueError(
                f"turn_reward_source must be one of {TURN_REWARD_SOURCES}, "
                f"got {self.turn_reward_source!r}"
            )
        self.turn_reward_scale = float(self.turn_reward_scale)
        self.turn_reward_budget = float(self.turn_reward_budget)
        self.turn_reward_components = str(self.turn_reward_components or "")
        # A negative scale silently inverts every component (rewarding what the
        # extractor meant to penalize); a negative budget would make the rescale
        # factor negative and do the same. Neither is a thing anyone means.
        if self.turn_reward_scale < 0.0:
            raise ValueError(
                f"turn_reward_scale must be >= 0, got {self.turn_reward_scale}. Use 0 to "
                "make the shaping inert, or turn_rewards: false to switch it off."
            )
        if self.turn_reward_budget < 0.0:
            raise ValueError(
                f"turn_reward_budget must be >= 0, got {self.turn_reward_budget}. "
                "Use 0 to disable the per-episode cap."
            )

    def length_penalty_kwargs(self) -> Dict[str, Any] | None:
        """Kwargs for ``advantage.LengthPenaltySpec``, or ``None`` when disabled.

        Returned as a plain dict rather than the dataclass itself so this module
        stays importable without ``torch`` (``advantage.py`` imports it).
        """
        if not self.length_penalty:
            return None
        return {
            "per_token": self.length_penalty_per_token,
            "budget": self.length_penalty_budget,
        }

    def legacy_length_penalty_values(self) -> Dict[str, Any]:
        """Legacy threshold fields this config sets away from their old defaults.

        Empty when nothing was set (the common case). Non-empty means a YAML, preset
        or CLI flag is still carrying a calibration for the removed threshold-based
        penalty, which is now inert -- the launch script turns this into one loud
        startup warning and the manifest records it, so a run can never quietly train
        under a penalty shape that no longer exists.

        Reported regardless of ``length_penalty``: a config that sets these while the
        penalty is off is equally out of date, and saying so costs nothing.
        """
        return {
            name: getattr(self, name)
            for name, default in LEGACY_LENGTH_PENALTY_DEFAULTS.items()
            if getattr(self, name) != default
        }

    @property
    def length_penalty_budget_is_safe(self) -> bool:
        """Whether the cap still guarantees the penalty cannot reorder two outcomes.

        See ``length_penalty_budget``. A disabled cap (``0``) is *not* safe -- the
        per-episode total is then unbounded, since the penalty is charged per turn.
        """
        if not self.length_penalty:
            return True
        return 0.0 < self.length_penalty_budget < LENGTH_PENALTY_SAFE_BUDGET

    def turn_reward_component_list(self) -> tuple:
        """``turn_reward_components`` parsed into a tuple; empty means "all"."""
        return tuple(
            name.strip() for name in self.turn_reward_components.split(",") if name.strip()
        )

    def turn_reward_kwargs(self) -> Dict[str, Any] | None:
        """Kwargs for ``turn_rewards.TurnRewardSpec``, or ``None`` when disabled.

        A plain dict rather than the dataclass so this module keeps its stdlib-only
        import profile (same contract as :meth:`length_penalty_kwargs`).
        """
        if not self.turn_rewards:
            return None
        return {
            "scale": self.turn_reward_scale,
            "budget": self.turn_reward_budget,
            "components": self.turn_reward_component_list(),
        }

    @property
    def turn_reward_budget_is_safe(self) -> bool:
        """Whether the budget still guarantees shaping cannot reorder two outcomes.

        See ``turn_reward_budget``: the worst-case swing is ``2 * budget``, and the
        smallest gap between two distinct clembench outcomes is 1.0. A disabled cap
        (``0``) is *not* safe -- there is then no bound at all.
        """
        if not self.turn_rewards:
            return True
        return 0.0 < self.turn_reward_budget < TURN_REWARD_SAFE_BUDGET

    @property
    def marshal_exact(self) -> bool:
        """Whether to reproduce MARSHAL's shipped (paper-divergent) behavior."""
        return self.fidelity_mode == "marshal_exact"

    @property
    def unique_value_pooling(self) -> bool:
        """Whether pool statistics are taken over *distinct* trajectory returns.

        The resolved answer for ``advantage._pool_offset_scale``'s ``torch.unique``:
        true only when ``marshal_exact`` is selected *and*
        ``marshal_exact_unique_pooling`` has not switched it off. Under
        ``paper_correct`` this is always False -- pooling is occurrence-weighted
        there by definition, so the sub-flag has nothing to disable.
        """
        return self.marshal_exact and self.marshal_exact_unique_pooling

    def trl_grpo_overrides(self) -> Dict[str, Any]:
        """TRL ``GRPOConfig`` kwargs implementing the Dr. GRPO recipe (or nothing).

        Returns an empty dict when ``dr_grpo`` is off, so the launch script can
        splat ``**cfg.trl_grpo_overrides()`` into ``GRPOConfig(...)`` and get a
        *byte-identical* config to today (no keys added, TRL's own defaults stand).
        When ``dr_grpo`` is on it returns the two settings TRL's docs prescribe for
        Dr. GRPO: ``loss_type="dr_grpo"`` (constant-normalized loss -> no length
        bias) and ``scale_rewards="none"`` (no std division -> no difficulty bias).

        Values are plain strings, so this stays importable without ``trl``.
        """
        if not self.dr_grpo:
            return {}
        return {"loss_type": "dr_grpo", "scale_rewards": "none"}

    def trl_sampling_overrides(self) -> Dict[str, Any]:
        """TRL ``GRPOConfig`` kwargs for sampling-tail truncation (or nothing).

        Returns only the keys that actually differ from TRL's defaults, and an empty
        dict when both are neutral -- so setting ``sampling_top_p: 1.0`` and
        ``sampling_top_k: 0`` gives a *byte-identical* ``GRPOConfig`` to the pre-fix
        one, with no key added. Same contract as :meth:`trl_grpo_overrides`.

        These map onto ``GRPOConfig.top_p`` / ``.top_k`` rather than
        ``generation_kwargs`` deliberately. TRL reads the first-class fields in
        ``_build_base_generation_kwargs`` (``trl/experimental/openenv/utils.py:34-47``)
        *and* in its own non-rollout generation path (``grpo_trainer.py:734-736``), so
        setting them covers evaluation generation too; ``generation_kwargs`` would only
        reach the custom rollout path and would silently leave eval untruncated.

        Values are plain floats/ints, so this stays importable without ``trl``.
        """
        overrides: Dict[str, Any] = {}
        if self.sampling_top_p != 1.0:
            overrides["top_p"] = self.sampling_top_p
        if self.sampling_top_k != 0:
            overrides["top_k"] = self.sampling_top_k
        return overrides

    def disable_sampling_truncation(self) -> None:
        """Revert to TRL's untruncated sampling (``--no-sampling-truncation``)."""
        self.sampling_top_p = 1.0
        self.sampling_top_k = 0

    def reconcile_for_dr_grpo(self) -> list[str]:
        """Align MARSHAL's advantage normalization with Dr. GRPO, in place.

        No-op (returns ``[]``) unless ``dr_grpo`` is on. When on *and* the MARSHAL
        path is active (``enabled``) *and* the opt-in per-seat z-score is selected
        (``advantage_norm_mode == "mean_std"``), switch it to ``"mean"`` -- dropping
        only the per-seat std *divisor* (the exact bias Dr. GRPO removes) while
        keeping per-seat *pooling* and every other MARSHAL feature. Returns a list
        of human-readable notices the caller can print (empty if nothing changed).

        Batch-wide ``whiten_advantages`` is deliberately left untouched: the
        Dr. GRPO paper objects only to per-group std, not batch-level normalization.
        """
        notices: list[str] = []
        if not self.dr_grpo:
            return notices
        if self.enabled and self.advantage_norm_mode == "mean_std":
            self.advantage_norm_mode = "mean"
            notices.append(
                "dr_grpo: advantage_norm_mode 'mean_std' -> 'mean' (Dr. GRPO removes "
                "the per-seat std divisor; per-seat pooling and all other MARSHAL "
                "features are kept)."
            )
        return notices

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "MarshalConfig":
        """Build a config from a plain dict, ignoring unknown keys with a note.

        Unknown keys are dropped (not errored) so an example YAML can carry
        commentary keys without breaking, but we surface them so typos in a real
        field name don't silently no-op.
        """
        data = dict(data or {})
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"Unknown MarshalConfig keys: {sorted(unknown)}. Valid keys: {sorted(known)}"
            )
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str) -> "MarshalConfig":
        """Load a config from a YAML file (requires PyYAML)."""
        import yaml  # local import so config.py stays importable without PyYAML

        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"MARSHAL config at {path} must be a mapping, got {type(data)}")
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

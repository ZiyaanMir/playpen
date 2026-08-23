"""Configuration for the MARSHAL-style self-play training pipeline.

Depends only on the standard library (plus an optional PyYAML import for
``from_yaml``), so it stays importable without torch, trl or vllm.

The MARSHAL paper (arXiv:2510.15414) contributes two ideas on top of a REINFORCE /
GRPO-style objective: a turn-level advantage estimator (rewards attributed at turn
boundaries and turned into per-token returns via a backward cumulative sum) and
agent-specific advantage normalization (advantages pooled per self-play seat).

``MarshalConfig`` is the single source of truth for whether those behaviors run and
how. See ``playpen/marshal/advantage.py`` for the math it drives and
``playpen/marshal/trainer.py`` for where it is consulted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict


# String enums rather than bools so their intent is self-documenting in a YAML file.
ADVANTAGE_NORM_MODES = ("mean", "mean_std")
ROW_CONTEXT_MODES = ("exact", "spliced")
EPISODE_PAIRING_MODES = ("shared", "replay")
# Re-exported from turn_rewards so argparse `choices=` and the validation below
# cannot drift from each other. turn_rewards.py is stdlib-only, like this module.
TURN_REWARD_SOURCES = ("auto", "game", "generic")

# Above this budget, turn rewards can swing an episode's total by more than the gap
# between two clembench outcomes (SUCCESS +1 / FAILURE 0 / ABORTED -1, i.e. 1.0), so
# a well-shaped loss could out-score a bare win.
TURN_REWARD_SAFE_BUDGET = 0.5

# Fields that used to exist and are now gone. Dropped on load rather than rejected,
# so YAMLs frozen alongside older runs still parse.
REMOVED_FIELDS = frozenset({
    "turn_level_rewards",
    "fidelity_mode",
    "marshal_exact_unique_pooling",
    "length_penalty",
    "length_penalty_per_token",
    "length_penalty_budget",
    "length_penalty_coef",
    "length_penalty_bonus",
    "length_penalty_min_len",
    "length_penalty_max_len",
    "length_penalty_offset",
})


@dataclass
class MarshalConfig:
    """Master switch + sub-flags for the MARSHAL self-play behavior.

    Attributes:
        enabled: The single on/off switch. ``False`` falls back to stock TRL GRPO:
            one scalar terminal reward per rollout row through TRL's own
            group-relative normalization. Fields marked "regardless of ``enabled``"
            below still apply; the rest have no effect.
        agent_specific_normalization: Pool and normalize advantages *per seat*
            (player_0 rows separately from player_1 rows) across the batch.
            ``False`` pools batch-wide -- the ablation that isolates turn-level
            credit from seat-specific pooling.
        advantage_norm_mode: ``"mean"`` mean-centers each pool; ``"mean_std"``
            z-scores it.
        gamma: Discount for the backward cumulative return. MARSHAL uses 1.0.
        whiten_rewards: Z-score the token-level reward field over all model-token
            positions batch-wide (both seats pooled) *before* the backward
            cumulative sum -- ROLL's ``whiten_rewards``. Not part of the paper's
            algorithm, hence default ``False``. Densifies a sparse reward (every
            model token picks up ``-mean/std``), adding a response-length-dependent
            component to the returns.
        whiten_advantages: Z-score the final advantages over all model-token
            positions batch-wide *after* the per-seat normalization -- ROLL's
            ``whiten_advantages``. Standard PPO-style scale stabilization.
        dr_grpo: Run the Dr. GRPO (arXiv:2503.20783) recipe by configuring TRL's
            loss/scaling: ``loss_type="dr_grpo"`` (loss normalized by the constant
            ``max_completion_length`` instead of realized length -- removes the
            response-level length bias) and ``scale_rewards="none"`` (no std
            division -- removes the difficulty bias). See :meth:`trl_grpo_overrides`.

            Takes effect *regardless of* ``enabled``, and is applied by the launch
            script rather than by ``advantage.py``. MARSHAL already avoids std
            division by default, so the two compose; the opt-in per-seat z-score
            (``"mean_std"``) is realigned back to ``"mean"`` by
            :meth:`reconcile_for_dr_grpo`.
        grpo_loss: Run with TRL's ``loss_type="grpo"`` -- the original GRPO
            aggregation, ``mean_i( sum_t l[i,t]*m[i,t] / sum_t m[i,t] )``. Nothing
            else moves: ``scale_rewards`` keeps TRL's ``"group"`` default and
            MARSHAL's advantage math is untouched. Mutually exclusive with
            :attr:`dr_grpo` (both write ``loss_type``); setting both raises.

            This matches upstream MARSHAL, a ROLL fork whose shipped self-play YAMLs
            leave ``loss_agg_mode`` at ROLL's ``"seq-mean-token-sum"`` default -- a
            per-row mean, i.e. TRL's ``grpo``, not TRL's default ``dapo``. Two
            consequences: every row gets weight ``1/B`` however long it is, so short
            rows dominate; and TRL divides by the full row count where ROLL divides
            by its valid-sample count, so placeholder rows scale the update by
            ``1 - placeholder_rate`` (read ``marshal/rows/placeholder_rate``).

            Like ``dr_grpo``, applies regardless of ``enabled``.
        row_context_mode: How a multi-turn rollout row's token sequence is assembled.
            Applies regardless of ``enabled``: it governs rollout collection.

            ``"exact"`` (default) -- the row's ``prompt_ids + completion_ids``
            reproduce, token for token, the context each turn was generated under.

            ``"spliced"`` -- the pre-fix behavior, kept only to reproduce earlier
            runs. From turn 2 onward the trained sequence is not the one the policy
            generated under, so recomputed log-probs diverge and TRL's vLLM
            importance-sampling correction collapses the row's gradient.
        episode_pairing: Whether the seats of one episode share that episode. Also
            governs rollout collection, so it applies regardless of ``enabled``.

            ``"shared"`` (default, and MARSHAL's own arrangement) -- one dataset row
            per game instance; each run of ``num_players`` consecutive copies of that
            prompt is served by a single episode, seat *k* taking the *k*-th copy.
            Both seats come from the same game, which is what makes a per-seat
            baseline compare like with like, and no seat is generated and discarded.
            ``num_generations`` should be a multiple of the game's player count; a
            run that cannot be paired falls back to ``"replay"`` for that index.

            ``"replay"`` -- the pre-fix arrangement: one dataset row per
            (instance, seat), every prompt replaying a whole fresh episode and
            keeping only the requested seat.
        sampling_top_p: Nucleus cutoff for generation, mapped onto
            ``GRPOConfig.top_p``. ``1.0`` disables it (TRL's default). Applies
            regardless of ``enabled``.

            Default ``0.95`` rather than TRL's untruncated default. With
            ``KL_BETA=0`` the policy's entropy collapses over training, and an
            untruncated sampler still draws the occasional far-tail token, where
            ``log p`` is sensitive enough to a small bf16 logit difference that vLLM
            and the training model disagree by many nats. TRL's
            ``vllm_importance_sampling_mode="sequence_mask"`` sums that divergence
            over the row and exponentiates it, annihilating the row's gradient.

            It does not bias the importance ratio: vLLM computes reported logprobs
            from the raw logits before applying top-p/top-k, so only the sampled
            support narrows. A mitigation -- the root cause is running without a KL
            term, which MARSHAL's own shipped ``*_selfplay.yaml`` never does.
        sampling_top_k: Top-k cutoff for generation, mapped onto
            ``GRPOConfig.top_k``. ``0`` disables it (TRL's default). Default ``50``.

            Kept alongside ``sampling_top_p`` because the two fail differently on a
            collapsed policy: at a near-deterministic position ``top_p=0.95`` keeps
            only the argmax while ``top_k=50`` still admits ranks 2..50; at a
            genuinely uncertain position ``top_k`` bounds the support where ``top_p``
            may keep a long tail.

            Set both neutral (``1.0`` / ``0``), or pass ``--no-sampling-truncation``,
            to reproduce untruncated generation: :meth:`trl_sampling_overrides` then
            returns an empty dict and no key is added to ``GRPOConfig``.
        turn_rewards: Master switch for the dense per-turn reward channel
            (``playpen/marshal/turn_rewards.py``). Default ``False``, and with it off
            nothing about a run changes.

            When ``True``, the game's live ``GameState`` is read after every turn for
            a small per-turn signal, scaled by ``turn_reward_scale``, capped by
            ``turn_reward_budget``, and added to that turn's reward.

            Applies regardless of ``enabled``, because it governs rollout collection.
            On the plain-GRPO path TRL consumes one scalar per row, so the episode's
            shaping total is added to it. Either way the unshaped outcome is logged
            as ``marshal/turn_rewards/terminal_mean``.

            It is what makes ``agent_specific_normalization`` do real work here:
            clembench gives both seats the same team outcome, so per-seat pooling
            otherwise mean-centers two identical distributions. A per-turn reward is
            attributed to the seat that acted.
        turn_reward_source: Which extractor supplies the signal. ``"auto"`` (default)
            uses the game's own extractor when one is registered and falls back to
            generic format compliance otherwise; ``"game"`` refuses the fallback
            (turn rewards are off for an unregistered game, with a warning);
            ``"generic"`` forces format compliance even for a game with a richer
            extractor, as a "compliance only" ablation.
        turn_reward_scale: Multiplier on each turn's combined signal, which the
            extractors normalize to ``[-1, 1]`` -- so this is the *maximum* magnitude
            a single turn can contribute. Default ``0.05``.
        turn_reward_budget: Hard cap on ``|sum of one seat's shaping over one
            episode|``, applied as a proportional rescale of the whole episode's
            shaping vector (relative credit between turns survives, no sign flips).
            The backward cumulative return means the episode total is what competes
            with the outcome, so with ``budget < 0.5`` the worst-case swing is
            ``2 * budget < 1.0``, smaller than the gap between any two distinct
            clembench outcomes: shaping can reorder episodes within an outcome class
            but never across one. Default ``0.3``; ``0`` disables the cap and that
            guarantee. Values ``>= 0.5`` (``TURN_REWARD_SAFE_BUDGET``) are allowed
            but warned about by the launch script.

            A safety net, not the operating point -- tune ``turn_reward_scale`` so a
            typical episode lands under it, and watch
            ``marshal/turn_rewards/budget_clip_rate``.
        turn_reward_components: Comma-separated allowlist of component names to keep
            (e.g. ``"closeness"`` for wordle's progress signal without its format
            penalty). Empty (default) keeps every component the extractor emits.
            Unknown names are rejected at build time against the resolved extractor's
            component list, so a typo cannot silently disable the signal.
    """

    enabled: bool = True
    agent_specific_normalization: bool = True
    advantage_norm_mode: str = "mean"
    gamma: float = 1.0
    whiten_rewards: bool = False
    whiten_advantages: bool = False
    dr_grpo: bool = False
    grpo_loss: bool = False
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
        # Both write GRPOConfig.loss_type, and they ask for opposite things. Rejected
        # rather than resolved by precedence: a silent winner would make an ablation
        # arm's name disagree with its loss.
        if self.dr_grpo and self.grpo_loss:
            raise ValueError(
                "dr_grpo and grpo_loss are mutually exclusive -- both set "
                "GRPOConfig.loss_type ('dr_grpo' vs 'grpo'). Pick one, or leave both "
                "off for TRL's default loss_type='dapo'."
            )
        self.sampling_top_p = float(self.sampling_top_p)
        self.sampling_top_k = int(self.sampling_top_k)
        # vLLM accepts out-of-range values silently and quietly changes (or disables)
        # the truncation, so reject them here where the message can say what was meant.
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
        if self.turn_reward_source not in TURN_REWARD_SOURCES:
            raise ValueError(
                f"turn_reward_source must be one of {TURN_REWARD_SOURCES}, "
                f"got {self.turn_reward_source!r}"
            )
        self.turn_reward_scale = float(self.turn_reward_scale)
        self.turn_reward_budget = float(self.turn_reward_budget)
        self.turn_reward_components = str(self.turn_reward_components or "")
        # A negative scale inverts every component; a negative budget makes the
        # rescale factor negative and does the same.
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

    def turn_reward_component_list(self) -> tuple:
        """``turn_reward_components`` parsed into a tuple; empty means "all"."""
        return tuple(
            name.strip() for name in self.turn_reward_components.split(",") if name.strip()
        )

    def turn_reward_kwargs(self) -> Dict[str, Any] | None:
        """Kwargs for ``turn_rewards.TurnRewardSpec``, or ``None`` when disabled.

        A plain dict rather than the dataclass, so this module keeps its stdlib-only
        import profile.
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

        The worst-case swing is ``2 * budget`` and the smallest gap between two
        distinct clembench outcomes is 1.0. A disabled cap (``0``) is not safe --
        there is then no bound at all.
        """
        if not self.turn_rewards:
            return True
        return 0.0 < self.turn_reward_budget < TURN_REWARD_SAFE_BUDGET

    @property
    def trl_loss_type(self) -> str:
        """The ``GRPOConfig.loss_type`` this config resolves to.

        ``"dapo"`` when neither loss flag is set -- named explicitly rather than left
        as "whatever TRL defaults to", so a printed/logged run says what it trained
        under. :meth:`trl_grpo_overrides` still omits the key in that case.
        """
        if self.dr_grpo:
            return "dr_grpo"
        if self.grpo_loss:
            return "grpo"
        return "dapo"

    def trl_grpo_overrides(self) -> Dict[str, Any]:
        """TRL ``GRPOConfig`` kwargs for the selected loss recipe (or nothing).

        Empty when both ``dr_grpo`` and ``grpo_loss`` are off, so the launch script
        can splat ``**cfg.trl_grpo_overrides()`` into ``GRPOConfig(...)`` and add no
        keys at all.

        ``dr_grpo`` returns the two settings TRL's docs prescribe;  ``grpo_loss``
        returns ``loss_type="grpo"`` and nothing else, leaving ``scale_rewards`` at
        TRL's ``"group"`` default (reward scaling is a separate axis, and inert on the
        MARSHAL path anyway, where the trainer overwrites TRL's scalar advantages).

        Values are plain strings, so this stays importable without ``trl``.
        """
        if self.dr_grpo:
            return {"loss_type": "dr_grpo", "scale_rewards": "none"}
        if self.grpo_loss:
            return {"loss_type": "grpo"}
        return {}

    def trl_sampling_overrides(self) -> Dict[str, Any]:
        """TRL ``GRPOConfig`` kwargs for sampling-tail truncation (or nothing).

        Returns only the keys that differ from TRL's defaults, and an empty dict when
        both are neutral.

        These map onto ``GRPOConfig.top_p`` / ``.top_k`` rather than
        ``generation_kwargs`` because TRL reads the first-class fields in its own
        non-rollout generation path too, so setting them covers evaluation
        generation; ``generation_kwargs`` would only reach the custom rollout path.

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

        No-op (returns ``[]``) unless ``dr_grpo`` is on. When on, the MARSHAL path is
        active and the opt-in per-seat z-score is selected, switch
        ``advantage_norm_mode`` to ``"mean"`` -- dropping only the per-seat std
        divisor Dr. GRPO removes, keeping per-seat pooling and every other MARSHAL
        feature. Returns human-readable notices the caller can print.

        Batch-wide ``whiten_advantages`` is left untouched: Dr. GRPO objects only to
        per-group std, not to batch-level normalization.
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
        """Build a config from a plain dict.

        Keys in :data:`REMOVED_FIELDS` are dropped so older YAMLs still load. Any
        other unknown key raises, so a typo in a real field name cannot silently
        no-op.
        """
        data = {k: v for k, v in dict(data or {}).items() if k not in REMOVED_FIELDS}
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

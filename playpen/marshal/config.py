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
            ``advantage._marshal_pre_sum_normalize``. Consulted only when
            ``enabled=True``.
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
    """

    enabled: bool = True
    agent_specific_normalization: bool = True
    turn_level_rewards: bool = True
    advantage_norm_mode: str = "mean"
    gamma: float = 1.0
    fidelity_mode: str = "paper_correct"
    whiten_rewards: bool = False
    whiten_advantages: bool = False
    dr_grpo: bool = False

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
        self.gamma = float(self.gamma)

    @property
    def marshal_exact(self) -> bool:
        """Whether to reproduce MARSHAL's shipped (paper-divergent) behavior."""
        return self.fidelity_mode == "marshal_exact"

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

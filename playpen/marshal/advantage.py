"""Turn-level returns and agent-specific (per-seat) advantage normalization.

This module is a from-scratch, framework-light port of MARSHAL's advantage math
(originally in ``MARSHAL/roll/utils/functionals.py``: ``compute_reinforce_return``
and ``normalize_unique_values_by_player``). It depends only on ``torch`` so it is
unit-testable with no model, GPU, trl, or vllm.

The two supported behaviors mirror ``MarshalConfig.fidelity_mode``:

* ``"paper_correct"`` -- the algorithm the paper *describes*: no biasing pre-sum
  reward normalization, and occurrence-weighted (frequency-weighted) pooling over
  trajectory returns.
* ``"marshal_exact"`` -- MARSHAL's *shipped code*, including two documented
  departures from its own paper: a pre-sum reward-normalization pass and
  distinct-value ("unique") pooling that equal-weights rare and common outcomes.

See ``PAPER_VS_CODE_DISCREPANCIES.md`` in the MARSHAL repo for the audit these
two modes are derived from.

Token/turn indexing convention
------------------------------
All positions are indices into a single rollout row's *completion* token
sequence (the tokens after the prompt), matching how TRL lays out the
``(B, T)`` advantages tensor it consumes (completion tokens are right-padded, so
completion position ``j`` maps to column ``j``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch

# Matches MARSHAL's epsilon in normalize_unique_values / score_normalize.
_EPS = 1e-6


@dataclass
class RowRollout:
    """The per-row information the advantage computation needs.

    One ``RowRollout`` corresponds to one rollout "row" == one self-play seat's
    trajectory within one episode (see ``playpen/marshal/selfplay_agent.py``).

    Attributes:
        seat: 0 or 1 -- which self-play seat produced this row.
        completion_len: number of (unpadded) completion tokens in this row.
        owner_mask: length-``completion_len`` list of 1 for model-generated tokens
            and 0 for environment-feedback tokens. Gradient (and advantage) only
            applies where this is 1.
        turn_end_positions: completion indices at which each of this seat's turns
            ends (i.e. the position of that turn's last generated token).
        turn_rewards: reward attributed to each of this seat's turns, aligned with
            ``turn_end_positions``. For clembench's default sparse rewards these
            are all 0 except the last (the terminal outcome).
    """

    seat: int
    completion_len: int
    owner_mask: Sequence[int]
    turn_end_positions: Sequence[int]
    turn_rewards: Sequence[float]


def build_reward_tensor(
    turn_end_positions: Sequence[int],
    turn_rewards: Sequence[float],
    seq_len: int,
    *,
    turn_level: bool = True,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Scatter per-turn rewards onto a length-``seq_len`` completion reward vector.

    Args:
        turn_end_positions: completion indices where each turn ends.
        turn_rewards: reward for each turn, aligned with ``turn_end_positions``.
        seq_len: length of the completion reward vector to build.
        turn_level: if True, each turn reward is placed at its own turn-end
            position (turn-level estimator). If False, all turn rewards are summed
            into a single scalar placed at the *last* turn-end position (mirrors
            MARSHAL ``use_turn_scores=False``).

    Returns:
        A ``(seq_len,)`` tensor, zero everywhere except at turn boundaries.
    """
    rewards = torch.zeros(seq_len, device=device, dtype=dtype)
    if not turn_end_positions:
        return rewards
    positions = list(turn_end_positions)
    values = list(turn_rewards)
    if not turn_level:
        # Collapse to a single terminal scalar at the last turn boundary.
        positions = [positions[-1]]
        values = [float(sum(values))]
    for pos, val in zip(positions, values):
        if 0 <= pos < seq_len:
            rewards[pos] += float(val)
    return rewards


def reward_slot_positions(
    turn_end_positions: Sequence[int], seq_len: int, *, turn_level: bool = True
) -> List[int]:
    """The positions that hold a reward *slot* (turn boundaries), matching :func:`build_reward_tensor`.

    These are the positions MARSHAL's pre-sum normalization averages over -- crucially
    *including* zero-reward boundaries (a lost/failed game is a genuine 0 in the mean),
    mirroring ``score_normalize``'s ``mask=turn_end_positions``. Must stay in lockstep
    with :func:`build_reward_tensor`'s position selection.
    """
    if not turn_end_positions:
        return []
    positions = list(turn_end_positions)
    if not turn_level:
        positions = [positions[-1]]
    return [p for p in positions if 0 <= p < seq_len]


def reinforce_returns(reward_tensor: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Backward discounted cumulative sum: ``R_t = sum_{t'>=t} gamma^(t'-t) r_t'``.

    This is the token-level Monte-Carlo return MARSHAL uses as its "advantage"
    before normalization (``compute_reinforce_return`` with ``advantages == returns``).
    Operates on the last dimension, so it accepts either ``(T,)`` or ``(B, T)``.
    """
    returns = torch.zeros_like(reward_tensor)
    running = torch.zeros_like(reward_tensor[..., 0])
    gen_len = reward_tensor.shape[-1]
    for t in reversed(range(gen_len)):
        running = reward_tensor[..., t] + gamma * running
        returns[..., t] = running
    return returns


def _row_trajectory_return(returns_row: torch.Tensor, owner_mask_row: torch.Tensor) -> torch.Tensor:
    """A single representative scalar per row: the return at its first model token.

    With a backward cumulative sum this equals the trajectory's total discounted
    return. Pooling one scalar per row (rather than per token) makes the
    normalization frequency-weighted by *trajectory* -- the natural unit for the
    paper's per-player normalization -- and free of token-count bias.
    Rows with no model tokens return 0 here, but are excluded from pool
    statistics by :func:`normalize_returns_by_seat` (they are placeholder rows
    for seats that produced no trajectory; MARSHAL never emits such rows).
    """
    idx = torch.nonzero(owner_mask_row > 0, as_tuple=False)
    if idx.numel() == 0:
        return returns_row.new_zeros(())
    return returns_row[idx[0, 0]]


def _pool_offset_scale(
    row_scalars: torch.Tensor,
    *,
    marshal_exact: bool,
    norm_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the (mean, scale) to apply to a pool of trajectory-return scalars.

    Args:
        row_scalars: 1-D tensor of one scalar per row in the pool.
        marshal_exact: if True, pool over the *set of distinct values* (MARSHAL's
            ``normalize_unique_values`` behavior -- equal-weights rare and common
            outcomes). If False, occurrence-weighted mean/std over all rows.
        norm_mode: ``"mean"`` -> scale is 1 (mean-center only);
            ``"mean_std"`` -> scale is ``std + eps`` (z-score).

    Returns:
        ``(mean, scale)`` scalar tensors. Degenerate pools (all-equal values, or a
        single unique value) yield ``mean == that value`` and ``scale == 1`` so
        that ``(x - mean) / scale`` collapses to 0 -- matching MARSHAL's guard of
        returning zeros when there is nothing to normalize against.
    """
    values = torch.unique(row_scalars) if marshal_exact else row_scalars
    one = row_scalars.new_ones(())

    if values.numel() <= 1:
        # Nothing to normalize against: subtract the value itself -> zeros.
        mean = values.reshape(-1)[0] if values.numel() == 1 else row_scalars.new_zeros(())
        return mean, one

    mean = values.mean()
    if norm_mode == "mean":
        return mean, one
    # norm_mode == "mean_std"
    std = values.std(unbiased=False)
    if std <= _EPS:
        return mean, one
    return mean, std + _EPS


def normalize_returns_by_seat(
    returns: torch.Tensor,
    owner_mask: torch.Tensor,
    seats: torch.Tensor,
    *,
    agent_specific: bool,
    marshal_exact: bool,
    norm_mode: str,
) -> torch.Tensor:
    """Apply MARSHAL's (per-seat) advantage normalization to a ``(B, T)`` returns.

    Args:
        returns: ``(B, T)`` per-token returns from ``reinforce_returns``.
        owner_mask: ``(B, T)`` 1 for model tokens, 0 elsewhere.
        seats: ``(B,)`` long tensor of seat ids (0/1).
        agent_specific: if True, pool each seat separately; else one batch-wide pool.
        marshal_exact: see ``_pool_offset_scale``.
        norm_mode: ``"mean"`` or ``"mean_std"``.

    Returns:
        ``(B, T)`` normalized advantages, zeroed at non-model-token positions.
    """
    row_scalars = torch.stack(
        [_row_trajectory_return(returns[i], owner_mask[i]) for i in range(returns.shape[0])]
    )

    means = torch.zeros_like(row_scalars)
    scales = torch.ones_like(row_scalars)

    if agent_specific:
        pool_ids = seats
    else:
        pool_ids = torch.zeros_like(seats)

    # Placeholder rows (no model tokens) must not contribute their artificial 0
    # scalar to the pool statistics; their advantages are zeroed by the
    # owner_mask multiply below regardless of what mean/scale they get.
    has_model_tokens = (owner_mask > 0).any(dim=1)

    for pool in torch.unique(pool_ids):
        member = pool_ids == pool
        stat_member = member & has_model_tokens
        if not stat_member.any():
            continue  # pool holds only placeholders; leave mean 0 / scale 1
        mean, scale = _pool_offset_scale(
            row_scalars[stat_member], marshal_exact=marshal_exact, norm_mode=norm_mode
        )
        means[member] = mean
        scales[member] = scale

    advantages = (returns - means.unsqueeze(1)) / scales.unsqueeze(1)
    advantages = advantages * (owner_mask > 0).to(advantages.dtype)
    return advantages


def masked_whiten(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Whiten ``values`` to mean 0 / unit variance over positions where ``mask`` is 1.

    Mirrors MARSHAL's ``masked_whiten`` (``roll/utils/functionals.py``): mean and
    Bessel-corrected variance are computed over masked positions only, but the
    shift/scale is applied to *every* position -- callers must re-zero unmasked
    positions afterwards, exactly like MARSHAL's trailing ``* response_mask``.

    One deliberate departure: MARSHAL raises when the mask selects fewer than two
    positions; here the input is returned unchanged instead, because a degenerate
    batch (e.g. all placeholder rows) must not crash training.
    """
    mask = (mask > 0).to(values.dtype)
    denom = mask.sum()
    if denom.item() < 2:
        return values
    mean = (values * mask).sum() / denom
    var = ((values - mean) ** 2 * mask).sum() / denom
    var = var * denom / (denom - 1)  # Bessel correction, as in MARSHAL's masked_var
    return (values - mean) * torch.rsqrt(var + 1e-8)


def _pad_row(values: Sequence[float], seq_len: int, device, dtype) -> torch.Tensor:
    out = torch.zeros(seq_len, device=device, dtype=dtype)
    n = min(len(values), seq_len)
    if n:
        out[:n] = torch.tensor(values[:n], device=device, dtype=dtype)
    return out


@torch.no_grad()
def compute_marshal_advantages(
    rows: List[RowRollout],
    seq_len: int,
    *,
    gamma: float = 1.0,
    turn_level: bool = True,
    agent_specific: bool = True,
    marshal_exact: bool = False,
    norm_mode: str = "mean",
    whiten_rewards: bool = False,
    whiten_advantages: bool = False,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """End-to-end: rows -> ``(B, T)`` MARSHAL advantages tensor.

    Steps, per row: build the sparse turn reward vector, (optionally, in
    ``marshal_exact`` mode) apply a biasing pre-sum reward normalization,
    (optionally) whiten the reward field, take the backward cumulative return.
    Then normalize the ``(B, T)`` returns per seat and (optionally) whiten the
    result.

    ``whiten_rewards`` / ``whiten_advantages`` mirror the same-named ROLL flags
    that every shipped MARSHAL ``*_selfplay.yaml`` sets to true (they are framework
    config, not part of the paper's algorithm, hence off by default here). Both
    operate batch-wide over model-token positions -- NOT per seat -- matching
    MARSHAL's ``compute_advantage``, which runs on the concatenated batch:

    * ``whiten_rewards`` z-scores the token-level reward field *before* the
      cumulative sum. Because most model-token positions carry reward 0, this
      densifies a sparse reward (every model token picks up ``-mean/std``), adding
      a response-length-dependent component to the returns.
    * ``whiten_advantages`` z-scores the final advantages *after* the per-seat
      normalization -- standard PPO-style scale stabilization.

    ``T`` is ``seq_len`` (TRL's padded completion length); rows shorter than that
    are zero-padded on the right, matching TRL's right-padding of completions.

    Wrapped in ``@torch.no_grad()`` (mirroring MARSHAL's advantage functions):
    advantages are a fixed target for the policy loss and must never carry
    gradient. Today every tensor here is built from plain Python lists so no graph
    would form anyway, but the decorator makes the invariant explicit and safe if a
    grad-carrying input is ever threaded in.
    """
    batch = len(rows)
    reward_rows = torch.zeros(batch, seq_len, device=device, dtype=dtype)
    slot_mask = torch.zeros(batch, seq_len, device=device, dtype=torch.bool)
    owner_mask = torch.zeros(batch, seq_len, device=device, dtype=dtype)
    seats = torch.zeros(batch, dtype=torch.long, device=device)

    for i, row in enumerate(rows):
        reward_rows[i] = build_reward_tensor(
            row.turn_end_positions,
            row.turn_rewards,
            seq_len,
            turn_level=turn_level,
            device=device,
            dtype=dtype,
        )
        for pos in reward_slot_positions(row.turn_end_positions, seq_len, turn_level=turn_level):
            slot_mask[i, pos] = True
        owner_mask[i] = _pad_row(row.owner_mask, seq_len, device, dtype)
        seats[i] = int(row.seat)

    if marshal_exact:
        # Faithful reproduction of MARSHAL's pre-sum reward normalization
        # (score_normalize with method="mean"), which the paper's own audit flags
        # as introducing a length-dependent bias. In self-play MARSHAL runs this is
        # per-seat: separate_norm_for_selfplay=true is set in every shipped
        # *_selfplay.yaml, routing through reward_normalize_by_player which
        # mean-centers each seat's reward entries separately.
        reward_rows = _marshal_pre_sum_normalize(
            reward_rows, slot_mask, seats, agent_specific=agent_specific
        )

    if whiten_rewards:
        # MARSHAL order: whitening runs after the (marshal_exact-only) pre-sum
        # normalization and before the cumulative sum, over the batch-wide model
        # token mask (MARSHAL's response_mask analog), then re-masked.
        reward_rows = masked_whiten(reward_rows, owner_mask) * owner_mask

    returns = reinforce_returns(reward_rows, gamma=gamma)
    advantages = normalize_returns_by_seat(
        returns,
        owner_mask,
        seats,
        agent_specific=agent_specific,
        marshal_exact=marshal_exact,
        norm_mode=norm_mode,
    )

    if whiten_advantages:
        advantages = masked_whiten(advantages, owner_mask) * owner_mask
    return advantages


def _marshal_pre_sum_normalize(
    reward_rows: torch.Tensor,
    slot_mask: torch.Tensor,
    seats: torch.Tensor,
    *,
    agent_specific: bool,
) -> torch.Tensor:
    """MARSHAL's pre-sum reward mean-centering (``score_normalize(method="mean")``).

    Mirrors MARSHAL's ``reward_postprocess_agentic`` step 1: when
    ``separate_norm_for_selfplay`` is true (the value in every shipped
    ``*_selfplay.yaml``) it routes through ``reward_normalize_by_player``, which
    partitions rows by seat and subtracts each seat's own reward mean from that
    seat's entries. We reproduce that per-seat structure here, mirroring the
    ``pool_ids = seats if agent_specific else zeros`` pooling used by
    :func:`normalize_returns_by_seat` so the seat-separation switch is consistent
    across both normalization stages. Deliberately *not* run in ``paper_correct``.

    The mean is taken over the seat's *turn-boundary* slots (``slot_mask``),
    exactly like MARSHAL's ``masked_mean(x, mask=turn_end_positions)`` -- which
    counts a zero-reward outcome (a lost/failed game) as a genuine ``0`` in the
    mean. Averaging over "non-zero reward entries" instead would silently drop
    every failure and, because clembench SUCCESS is always exactly ``+1``, collapse
    a no-abort batch to all-equal values whose centered advantage is ``0`` (i.e.
    no learning signal). Only positions in ``slot_mask`` are recentered; every
    other (non-boundary) position stays ``0``, matching ``score_normalize``'s
    trailing ``x_norm * mask``.
    """
    pool_ids = seats if agent_specific else torch.zeros_like(seats)
    slot_mask = slot_mask.bool()
    normalized = reward_rows.clone()
    for pool in torch.unique(pool_ids):
        member = pool_ids == pool
        block = reward_rows[member]
        block_slots = slot_mask[member]
        if block_slots.sum() == 0:
            continue
        mean = block[block_slots].mean()
        block_norm = block.clone()
        block_norm[block_slots] = block[block_slots] - mean
        normalized[member] = block_norm
    return normalized

"""Turn-level returns and agent-specific (per-seat) advantage normalization.

A framework-light port of MARSHAL's advantage math (``MARSHAL/roll/utils/
functionals.py``: ``compute_reinforce_return`` and
``normalize_unique_values_by_player``). Depends only on ``torch``, so it is
unit-testable with no model, GPU, trl or vllm.

Token/turn indexing convention
------------------------------
All positions are indices into a single rollout row's *completion* token
sequence (the tokens after the prompt), matching how TRL lays out the ``(B, T)``
advantages tensor it consumes (completion tokens are right-padded, so completion
position ``j`` maps to column ``j``).
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

    One ``RowRollout`` is one self-play seat's trajectory within one episode
    (see ``playpen/marshal/selfplay_agent.py``).

    Attributes:
        seat: 0 or 1 -- which self-play seat produced this row.
        completion_len: number of (unpadded) completion tokens in this row.
        owner_mask: length-``completion_len`` list of 1 for model-generated tokens
            and 0 for environment-feedback tokens. Gradient (and advantage) only
            applies where this is 1.
        turn_end_positions: completion indices at which each of this seat's turns
            ends (the position of that turn's last generated token).
        turn_rewards: reward attributed to each of this seat's turns, aligned with
            ``turn_end_positions``.
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
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Scatter a row's rewards onto a length-``seq_len`` completion reward vector.

    A seat's turn rewards are summed into a single scalar placed at the *last*
    turn-end position (MARSHAL's ``use_turn_scores=False``).

    Returns:
        A ``(seq_len,)`` tensor, zero everywhere except at that position.
    """
    rewards = torch.zeros(seq_len, device=device, dtype=dtype)
    if not turn_end_positions:
        return rewards
    pos = list(turn_end_positions)[-1]
    if 0 <= pos < seq_len:
        rewards[pos] += float(sum(turn_rewards))
    return rewards


def reinforce_returns(reward_tensor: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Backward discounted cumulative sum: ``R_t = sum_{t'>=t} gamma^(t'-t) r_t'``.

    The token-level Monte-Carlo return MARSHAL uses as its "advantage" before
    normalization (``compute_reinforce_return`` with ``advantages == returns``).
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
    return, so pooling it makes the normalization frequency-weighted by
    *trajectory* and free of token-count bias. Rows with no model tokens return 0
    here but are excluded from pool statistics by :func:`normalize_returns_by_seat`.
    """
    idx = torch.nonzero(owner_mask_row > 0, as_tuple=False)
    if idx.numel() == 0:
        return returns_row.new_zeros(())
    return returns_row[idx[0, 0]]


def _pool_offset_scale(
    row_scalars: torch.Tensor,
    *,
    norm_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The (mean, scale) to apply to a pool of trajectory-return scalars.

    Pooling is occurrence-weighted: every row counts once.

    Args:
        row_scalars: 1-D tensor of one scalar per row in the pool.
        norm_mode: ``"mean"`` -> scale is 1 (mean-center only);
            ``"mean_std"`` -> scale is ``std + eps`` (z-score).

    Returns:
        ``(mean, scale)`` scalar tensors. Degenerate pools (all-equal values) yield
        ``mean == that value`` and ``scale == 1``, so ``(x - mean) / scale``
        collapses to 0 -- matching MARSHAL's guard of returning zeros when there is
        nothing to normalize against.
    """
    one = row_scalars.new_ones(())
    if row_scalars.numel() <= 1:
        mean = row_scalars.reshape(-1)[0] if row_scalars.numel() == 1 else row_scalars.new_zeros(())
        return mean, one

    mean = row_scalars.mean()
    if norm_mode == "mean":
        return mean, one
    std = row_scalars.std(unbiased=False)
    if std <= _EPS:
        return mean, one
    return mean, std + _EPS


def normalize_returns_by_seat(
    returns: torch.Tensor,
    owner_mask: torch.Tensor,
    seats: torch.Tensor,
    *,
    agent_specific: bool,
    norm_mode: str,
) -> torch.Tensor:
    """Apply MARSHAL's (per-seat) advantage normalization to a ``(B, T)`` returns.

    Args:
        returns: ``(B, T)`` per-token returns from ``reinforce_returns``.
        owner_mask: ``(B, T)`` 1 for model tokens, 0 elsewhere.
        seats: ``(B,)`` long tensor of seat ids (0/1).
        agent_specific: if True, pool each seat separately; else one batch-wide pool.
        norm_mode: ``"mean"`` or ``"mean_std"``.

    Returns:
        ``(B, T)`` normalized advantages, zeroed at non-model-token positions.
    """
    row_scalars = torch.stack(
        [_row_trajectory_return(returns[i], owner_mask[i]) for i in range(returns.shape[0])]
    )

    means = torch.zeros_like(row_scalars)
    scales = torch.ones_like(row_scalars)

    pool_ids = seats if agent_specific else torch.zeros_like(seats)

    # Placeholder rows (no model tokens) must not contribute their artificial 0
    # scalar to the pool statistics; their advantages are zeroed by the owner_mask
    # multiply below regardless of what mean/scale they get.
    has_model_tokens = (owner_mask > 0).any(dim=1)

    for pool in torch.unique(pool_ids):
        member = pool_ids == pool
        stat_member = member & has_model_tokens
        if not stat_member.any():
            continue  # pool holds only placeholders; leave mean 0 / scale 1
        mean, scale = _pool_offset_scale(row_scalars[stat_member], norm_mode=norm_mode)
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
    positions afterwards, like MARSHAL's trailing ``* response_mask``.

    Where MARSHAL raises if the mask selects fewer than two positions, this returns
    the input unchanged, so a degenerate batch cannot crash training.
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
    agent_specific: bool = True,
    norm_mode: str = "mean",
    whiten_rewards: bool = False,
    whiten_advantages: bool = False,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """End-to-end: rows -> ``(B, T)`` MARSHAL advantages tensor.

    Per row: build the sparse reward vector, (optionally) whiten the reward field,
    take the backward cumulative return. Then normalize the ``(B, T)`` returns per
    seat and (optionally) whiten the result.

    ``whiten_rewards`` / ``whiten_advantages`` mirror the same-named ROLL flags that
    every shipped MARSHAL ``*_selfplay.yaml`` sets to true (framework config, not
    part of the paper's algorithm, hence off by default here). Both operate
    batch-wide over model-token positions -- NOT per seat -- matching MARSHAL's
    ``compute_advantage``, which runs on the concatenated batch:

    * ``whiten_rewards`` z-scores the token-level reward field *before* the
      cumulative sum. Because most model-token positions carry reward 0, this
      densifies a sparse reward (every model token picks up ``-mean/std``), adding
      a response-length-dependent component to the returns.
    * ``whiten_advantages`` z-scores the final advantages *after* the per-seat
      normalization -- standard PPO-style scale stabilization.

    ``T`` is ``seq_len`` (TRL's padded completion length); rows shorter than that
    are zero-padded on the right, matching TRL's right-padding of completions.

    ``@torch.no_grad()`` mirrors MARSHAL's advantage functions: advantages are a
    fixed target for the policy loss and must never carry gradient.
    """
    batch = len(rows)
    reward_rows = torch.zeros(batch, seq_len, device=device, dtype=dtype)
    owner_mask = torch.zeros(batch, seq_len, device=device, dtype=dtype)
    seats = torch.zeros(batch, dtype=torch.long, device=device)

    for i, row in enumerate(rows):
        reward_rows[i] = build_reward_tensor(
            row.turn_end_positions,
            list(row.turn_rewards),
            seq_len,
            device=device,
            dtype=dtype,
        )
        owner_mask[i] = _pad_row(row.owner_mask, seq_len, device, dtype)
        seats[i] = int(row.seat)

    if whiten_rewards:
        # MARSHAL order: whitening runs before the cumulative sum, over the
        # batch-wide model token mask (MARSHAL's response_mask analog), then
        # re-masked.
        reward_rows = masked_whiten(reward_rows, owner_mask) * owner_mask

    returns = reinforce_returns(reward_rows, gamma=gamma)
    advantages = normalize_returns_by_seat(
        returns,
        owner_mask,
        seats,
        agent_specific=agent_specific,
        norm_mode=norm_mode,
    )

    if whiten_advantages:
        advantages = masked_whiten(advantages, owner_mask) * owner_mask
    return advantages

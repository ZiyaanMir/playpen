"""Unit tests for the MARSHAL self-play advantage math and rollout construction.

Pure logic -- no model, GPU, or vLLM. Runnable via ``pytest`` or directly:

    .venv/bin/python -m pytest tests/test_marshal_advantage.py
    .venv/bin/python tests/test_marshal_advantage.py
"""

import unittest

import torch

from playpen.marshal.advantage import (
    LengthPenaltySpec,
    RowRollout,
    _marshal_pre_sum_normalize,
    apply_length_penalty,
    build_reward_tensor,
    compute_marshal_advantages,
    masked_whiten,
    reinforce_returns,
    row_length_penalties,
    turn_token_lengths,
)
from playpen.marshal.config import MarshalConfig


def _sparse_row(seat, reward, *, model_tokens=2):
    """A single-turn row whose only reward lands at its last (model) token."""
    owner_mask = [1] * model_tokens
    return RowRollout(
        seat=seat,
        completion_len=model_tokens,
        owner_mask=owner_mask,
        turn_end_positions=[model_tokens - 1],
        turn_rewards=[reward],
    )


class TestReinforceReturns(unittest.TestCase):
    def test_terminal_spike_broadcasts_backwards_gamma_1(self):
        rewards = torch.tensor([0.0, 0.0, 0.0, 5.0])
        got = reinforce_returns(rewards, gamma=1.0)
        self.assertTrue(torch.allclose(got, torch.tensor([5.0, 5.0, 5.0, 5.0])))

    def test_discounted_gamma_half(self):
        rewards = torch.tensor([0.0, 0.0, 0.0, 5.0])
        got = reinforce_returns(rewards, gamma=0.5)
        # R3=5, R2=2.5, R1=1.25, R0=0.625
        self.assertTrue(torch.allclose(got, torch.tensor([0.625, 1.25, 2.5, 5.0])))

    def test_dense_multi_turn(self):
        rewards = torch.tensor([1.0, 0.0, 2.0])  # gamma 1 -> [3,2,2]
        got = reinforce_returns(rewards, gamma=1.0)
        self.assertTrue(torch.allclose(got, torch.tensor([3.0, 2.0, 2.0])))

    def test_batched(self):
        rewards = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
        got = reinforce_returns(rewards, gamma=1.0)
        self.assertTrue(torch.allclose(got, torch.tensor([[1.0, 1.0], [2.0, 0.0]])))


class TestBuildRewardTensor(unittest.TestCase):
    def test_turn_level_places_each_reward(self):
        got = build_reward_tensor([1, 3], [2.0, 3.0], seq_len=4, turn_level=True)
        self.assertTrue(torch.allclose(got, torch.tensor([0.0, 2.0, 0.0, 3.0])))

    def test_non_turn_level_collapses_to_terminal(self):
        got = build_reward_tensor([1, 3], [2.0, 3.0], seq_len=4, turn_level=False)
        # sum(2,3)=5 placed at the last turn-end position (3)
        self.assertTrue(torch.allclose(got, torch.tensor([0.0, 0.0, 0.0, 5.0])))

    def test_out_of_range_positions_ignored(self):
        got = build_reward_tensor([5], [1.0], seq_len=3, turn_level=True)
        self.assertTrue(torch.allclose(got, torch.zeros(3)))


class TestSeatPooling(unittest.TestCase):
    def test_paper_correct_is_occurrence_weighted(self):
        # 9 rows reward=+1, 1 row reward=-1, all seat 0.
        rows = [_sparse_row(0, 1.0) for _ in range(9)] + [_sparse_row(0, -1.0)]
        adv = compute_marshal_advantages(
            rows, seq_len=2, agent_specific=True, marshal_exact=False, norm_mode="mean"
        )
        # mean = (9*1 + -1)/10 = 0.8
        self.assertAlmostEqual(adv[0, 0].item(), 0.2, places=5)   # +1 row
        self.assertAlmostEqual(adv[-1, 0].item(), -1.8, places=5)  # -1 row

    def test_marshal_exact_dedups_distinct_values(self):
        rows = [_sparse_row(0, 1.0) for _ in range(9)] + [_sparse_row(0, -1.0)]
        adv = compute_marshal_advantages(
            rows, seq_len=2, agent_specific=True, marshal_exact=True, norm_mode="mean"
        )
        # unique({1,-1}) mean = 0.0 -> equal-weights rare and common
        self.assertAlmostEqual(adv[0, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(adv[-1, 0].item(), -1.0, places=5)

    def test_unique_pooling_off_reverts_to_occurrence_weighted(self):
        """The sub-flag must undo the dedup, and nothing else about marshal_exact."""
        rows = [_sparse_row(0, 1.0) for _ in range(9)] + [_sparse_row(0, -1.0)]
        adv = compute_marshal_advantages(
            rows, seq_len=2, agent_specific=True, marshal_exact=True,
            unique_pooling=False, norm_mode="mean",
        )
        # Terminal-only rewards, so the pre-sum pass shifts every row by the same
        # seat mean and the *pooling* rule is what is on trial: mean 0.8, as in
        # test_paper_correct_is_occurrence_weighted rather than the uniqued 0.0.
        self.assertAlmostEqual(adv[0, 0].item(), 0.2, places=5)
        self.assertAlmostEqual(adv[-1, 0].item(), -1.8, places=5)

    def test_unique_pooling_none_follows_marshal_exact(self):
        """Omitting the flag must be byte-identical to before it existed."""
        rows = [_sparse_row(0, 1.0) for _ in range(9)] + [_sparse_row(0, -1.0)]
        for exact in (False, True):
            legacy = compute_marshal_advantages(
                rows, seq_len=2, agent_specific=True, marshal_exact=exact, norm_mode="mean"
            )
            explicit = compute_marshal_advantages(
                rows, seq_len=2, agent_specific=True, marshal_exact=exact,
                unique_pooling=exact, norm_mode="mean",
            )
            self.assertTrue(torch.allclose(legacy, explicit), f"marshal_exact={exact}")

    def test_unique_pooling_off_keeps_the_pre_sum_pass(self):
        """Only the pooling rule moves: the pre-sum bias must survive the flag."""
        # A nonzero non-terminal reward is what the pre-sum pass acts on (see
        # TestPreSumNormalizationDivergence), and distinct returns make unique
        # pooling a no-op, so any remaining difference is the pre-sum pass alone.
        row = RowRollout(
            seat=0, completion_len=4, owner_mask=[1, 1, 1, 1],
            turn_end_positions=[1, 3], turn_rewards=[0.5, 1.0],
        )
        other = RowRollout(
            seat=0, completion_len=4, owner_mask=[1, 1, 1, 1],
            turn_end_positions=[1, 3], turn_rewards=[0.0, 0.0],
        )
        kw = dict(seq_len=4, norm_mode="mean", turn_level=True)
        paper = compute_marshal_advantages([row, other], marshal_exact=False, **kw)
        no_unique = compute_marshal_advantages(
            [row, other], marshal_exact=True, unique_pooling=False, **kw
        )
        self.assertFalse(torch.allclose(paper, no_unique))

    def test_agent_specific_vs_batchwide(self):
        # seat 0 all +1, seat 1 all -1.
        rows = [_sparse_row(0, 1.0), _sparse_row(0, 1.0), _sparse_row(1, -1.0), _sparse_row(1, -1.0)]
        adv_specific = compute_marshal_advantages(
            rows, seq_len=2, agent_specific=True, marshal_exact=False, norm_mode="mean"
        )
        # each seat centered against itself -> all zero
        self.assertTrue(torch.allclose(adv_specific, torch.zeros_like(adv_specific), atol=1e-6))

        adv_batch = compute_marshal_advantages(
            rows, seq_len=2, agent_specific=False, marshal_exact=False, norm_mode="mean"
        )
        # pooled mean = 0 -> seat0 rows +1, seat1 rows -1
        self.assertAlmostEqual(adv_batch[0, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(adv_batch[2, 0].item(), -1.0, places=5)

    def test_mean_std_zscore(self):
        rows = [_sparse_row(0, 1.0), _sparse_row(0, -1.0)]
        adv = compute_marshal_advantages(
            rows, seq_len=2, agent_specific=True, marshal_exact=False, norm_mode="mean_std"
        )
        # scalars {1,-1}: mean 0, population std 1 -> advantages +1 / -1
        self.assertAlmostEqual(adv[0, 0].item(), 1.0, places=4)
        self.assertAlmostEqual(adv[1, 0].item(), -1.0, places=4)

    def test_degenerate_all_equal_pool_is_zero(self):
        rows = [_sparse_row(0, 1.0) for _ in range(4)]
        adv = compute_marshal_advantages(
            rows, seq_len=2, agent_specific=True, marshal_exact=False, norm_mode="mean"
        )
        self.assertTrue(torch.allclose(adv, torch.zeros_like(adv), atol=1e-6))

    def test_env_and_pad_positions_zeroed(self):
        # Row with an env-feedback token in the middle (owner_mask 0) and padding.
        row = RowRollout(
            seat=0, completion_len=3, owner_mask=[1, 0, 1],
            turn_end_positions=[2], turn_rewards=[1.0],
        )
        other = _sparse_row(0, 0.0, model_tokens=3)  # makes the pool non-degenerate
        adv = compute_marshal_advantages(
            [row, other], seq_len=5, agent_specific=True, marshal_exact=False, norm_mode="mean"
        )
        # position 1 is env feedback -> must be zero; positions 3,4 are padding -> zero.
        self.assertAlmostEqual(adv[0, 1].item(), 0.0, places=6)
        self.assertAlmostEqual(adv[0, 3].item(), 0.0, places=6)
        self.assertAlmostEqual(adv[0, 4].item(), 0.0, places=6)


class TestMarshalExactPreSumPerSeat(unittest.TestCase):
    """marshal_exact pre-sum normalization must be per-seat (separate_norm_for_selfplay)."""

    def test_pre_sum_centers_each_seat_separately(self):
        # seat 0 rewards {1, 3} (mean 2); seat 1 rewards {10, 20} (mean 15).
        reward_rows = torch.tensor([
            [0.0, 1.0],
            [0.0, 3.0],
            [0.0, 10.0],
            [0.0, 20.0],
        ])
        slot_mask = torch.tensor([  # each row's reward slot is at position 1
            [False, True],
            [False, True],
            [False, True],
            [False, True],
        ])
        seats = torch.tensor([0, 0, 1, 1])
        out = _marshal_pre_sum_normalize(reward_rows, slot_mask, seats, agent_specific=True)
        # seat 0 centered by 2, seat 1 centered by 15; zeros stay zero.
        expected = torch.tensor([
            [0.0, -1.0],
            [0.0, 1.0],
            [0.0, -5.0],
            [0.0, 5.0],
        ])
        self.assertTrue(torch.allclose(out, expected))

    def test_batchwide_when_not_agent_specific(self):
        # With agent_specific=False, a single global mean over all nonzero entries.
        reward_rows = torch.tensor([[1.0], [3.0], [10.0], [20.0]])
        slot_mask = torch.tensor([[True], [True], [True], [True]])
        seats = torch.tensor([0, 0, 1, 1])
        out = _marshal_pre_sum_normalize(reward_rows, slot_mask, seats, agent_specific=False)
        mean = (1 + 3 + 10 + 20) / 4  # 8.5
        self.assertTrue(torch.allclose(out, reward_rows - mean))

    def test_zero_reward_slots_counted_in_mean(self):
        # Regression for the signal-nulling bug: clembench rewards are
        # SUCCESS=+1, FAILURE=0, ABORT=-1. A FAILURE is a genuine 0 at a turn
        # boundary and MUST count toward the pre-sum mean (MARSHAL masks by
        # turn-end *position*, not by reward *value*). The old `block != 0`
        # masking dropped every failure and, since SUCCESS is always exactly +1,
        # collapsed a no-abort batch to all-zero -> no gradient.
        reward_rows = torch.tensor([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        slot_mask = torch.tensor([[False, True]] * 4)  # every row has a terminal slot
        seats = torch.tensor([0, 0, 0, 0])
        out = _marshal_pre_sum_normalize(reward_rows, slot_mask, seats, agent_specific=True)
        # mean over the four slots = (1 + 0 + 0 + 0) / 4 = 0.25
        expected = torch.tensor([[0.0, 0.75], [0.0, -0.25], [0.0, -0.25], [0.0, -0.25]])
        self.assertTrue(torch.allclose(out, expected))
        # ...and end-to-end the seat must get a non-zero learning signal.
        rows = [_sparse_row(0, r) for r in (1.0, 0.0, 0.0, 0.0)]
        adv = compute_marshal_advantages(
            rows, seq_len=2, marshal_exact=True, agent_specific=True, norm_mode="mean"
        )
        self.assertFalse(torch.allclose(adv, torch.zeros_like(adv)))

    def test_asymmetric_batch_composition_end_to_end(self):
        # The failure mode from the verification note: with an imbalanced seat mix,
        # the old global-mean pre-sum contaminated one seat with the other's scale.
        # Per-seat pre-sum keeps them independent, so seat 0's FINAL advantages are
        # unaffected by whether seat 1 is in the batch. Variable turn counts within
        # seat 0 make the pre-sum shift non-cancelling, so this actually
        # distinguishes the per-seat fix from the old global-mean behavior.
        seat0 = [
            RowRollout(0, 2, [1, 1], [1], [2.0]),          # one 2-token turn
            RowRollout(0, 2, [1, 1], [0, 1], [1.0, 1.0]),  # two 1-token turns
        ]
        seat1 = [RowRollout(1, 2, [1, 1], [1], [5.0])]
        adv_with_seat1 = compute_marshal_advantages(
            seat0 + seat1, seq_len=2, marshal_exact=True, agent_specific=True,
            norm_mode="mean", turn_level=True,
        )
        adv_seat0_only = compute_marshal_advantages(
            seat0, seq_len=2, marshal_exact=True, agent_specific=True,
            norm_mode="mean", turn_level=True,
        )
        # The test must be non-degenerate (seat 0 output not all zeros)...
        self.assertFalse(torch.allclose(adv_seat0_only, torch.zeros_like(adv_seat0_only)))
        # ...and seat 0's advantages must be identical with or without seat 1.
        self.assertTrue(torch.allclose(adv_with_seat1[:2], adv_seat0_only))


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = MarshalConfig()
        self.assertTrue(cfg.enabled)
        self.assertTrue(cfg.agent_specific_normalization)
        self.assertFalse(cfg.marshal_exact)
        self.assertEqual(cfg.gamma, 1.0)

    def test_from_dict_rejects_unknown_keys(self):
        with self.assertRaises(ValueError):
            MarshalConfig.from_dict({"enabledd": True})

    def test_validates_enums(self):
        with self.assertRaises(ValueError):
            MarshalConfig(fidelity_mode="bogus")
        with self.assertRaises(ValueError):
            MarshalConfig(advantage_norm_mode="bogus")

    def test_marshal_exact_property(self):
        self.assertTrue(MarshalConfig(fidelity_mode="marshal_exact").marshal_exact)

    def test_unique_pooling_defaults_to_following_fidelity_mode(self):
        self.assertTrue(MarshalConfig().marshal_exact_unique_pooling)
        self.assertTrue(MarshalConfig(fidelity_mode="marshal_exact").unique_value_pooling)
        self.assertFalse(MarshalConfig(fidelity_mode="paper_correct").unique_value_pooling)

    def test_unique_pooling_sub_flag_only_bites_under_marshal_exact(self):
        off_exact = MarshalConfig(
            fidelity_mode="marshal_exact", marshal_exact_unique_pooling=False
        )
        self.assertFalse(off_exact.unique_value_pooling)
        self.assertTrue(off_exact.marshal_exact)  # the pre-sum pass is untouched
        # Under paper_correct the sub-flag has nothing to disable, either way.
        off_paper = MarshalConfig(
            fidelity_mode="paper_correct", marshal_exact_unique_pooling=False
        )
        self.assertFalse(off_paper.unique_value_pooling)


class TestSamplingTruncation(unittest.TestCase):
    """top_p / top_k, and the byte-identical revert contract.

    These exist because untruncated sampling from a low-entropy policy produces rare
    far-tail draws whose log-prob vLLM and the trainer disagree about by many nats,
    which sequence-level importance sampling turns into a dead row. The revert path
    matters as much as the fix: reproducing a pre-2026-07-28 run must add NO key to
    GRPOConfig, not pass top_p=1.0/top_k=0 explicitly.
    """

    def test_defaults_are_truncating(self):
        cfg = MarshalConfig()
        self.assertEqual(cfg.sampling_top_p, 0.95)
        self.assertEqual(cfg.sampling_top_k, 50)
        self.assertEqual(cfg.trl_sampling_overrides(), {"top_p": 0.95, "top_k": 50})

    def test_neutral_values_add_no_key_at_all(self):
        cfg = MarshalConfig(sampling_top_p=1.0, sampling_top_k=0)
        self.assertEqual(cfg.trl_sampling_overrides(), {})

    def test_each_knob_is_independently_neutral(self):
        self.assertEqual(
            MarshalConfig(sampling_top_p=1.0, sampling_top_k=50).trl_sampling_overrides(),
            {"top_k": 50},
        )
        self.assertEqual(
            MarshalConfig(sampling_top_p=0.9, sampling_top_k=0).trl_sampling_overrides(),
            {"top_p": 0.9},
        )

    def test_disable_helper_matches_the_neutral_config(self):
        cfg = MarshalConfig()
        cfg.disable_sampling_truncation()
        self.assertEqual(cfg.sampling_top_p, 1.0)
        self.assertEqual(cfg.sampling_top_k, 0)
        self.assertEqual(cfg.trl_sampling_overrides(), {})

    def test_rejects_out_of_range_top_p(self):
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                MarshalConfig(sampling_top_p=bad)

    def test_rejects_negative_top_k(self):
        with self.assertRaises(ValueError):
            MarshalConfig(sampling_top_k=-1)

    def test_values_are_coerced_to_their_types(self):
        cfg = MarshalConfig(sampling_top_p="0.9", sampling_top_k="20")
        self.assertIsInstance(cfg.sampling_top_p, float)
        self.assertIsInstance(cfg.sampling_top_k, int)
        self.assertEqual(cfg.trl_sampling_overrides(), {"top_p": 0.9, "top_k": 20})

    def test_from_dict_roundtrip(self):
        cfg = MarshalConfig.from_dict({"sampling_top_p": 0.8, "sampling_top_k": 10})
        self.assertEqual(cfg.to_dict()["sampling_top_p"], 0.8)
        self.assertEqual(cfg.to_dict()["sampling_top_k"], 10)

    def test_applies_regardless_of_enabled(self):
        """It configures generation, which the plain-GRPO baseline shares."""
        cfg = MarshalConfig(enabled=False)
        self.assertEqual(cfg.trl_sampling_overrides(), {"top_p": 0.95, "top_k": 50})

    def test_composes_with_dr_grpo_without_key_collision(self):
        cfg = MarshalConfig(dr_grpo=True)
        merged = {**cfg.trl_grpo_overrides(), **cfg.trl_sampling_overrides()}
        self.assertEqual(
            merged,
            {"loss_type": "dr_grpo", "scale_rewards": "none", "top_p": 0.95, "top_k": 50},
        )
        self.assertFalse(
            set(cfg.trl_grpo_overrides()) & set(cfg.trl_sampling_overrides()),
            "the two override dicts must not fight over a key",
        )

    def test_overrides_are_valid_grpoconfig_kwargs(self):
        """Guards against a TRL rename turning this into a silently ignored kwarg."""
        try:
            from trl import GRPOConfig
        except Exception:  # pragma: no cover - trl not installed
            self.skipTest("trl not installed")
        cfg = MarshalConfig()
        gc = GRPOConfig(output_dir="/tmp/marshal-test", **cfg.trl_sampling_overrides())
        self.assertEqual(gc.top_p, 0.95)
        self.assertEqual(gc.top_k, 50)


class TestDrGrpo(unittest.TestCase):
    """The Dr. GRPO switch: TRL-config overrides + non-interfering reconciliation."""

    def test_default_off(self):
        cfg = MarshalConfig()
        self.assertFalse(cfg.dr_grpo)

    def test_overrides_off_is_empty(self):
        # Off => empty dict => splatting into GRPOConfig(...) changes nothing.
        self.assertEqual(MarshalConfig(dr_grpo=False).trl_grpo_overrides(), {})

    def test_overrides_on_sets_loss_and_scaling(self):
        self.assertEqual(
            MarshalConfig(dr_grpo=True).trl_grpo_overrides(),
            {"loss_type": "dr_grpo", "scale_rewards": "none"},
        )

    def test_reconcile_off_is_noop(self):
        # dr_grpo off: never touch advantage_norm_mode, even if mean_std.
        cfg = MarshalConfig(dr_grpo=False, advantage_norm_mode="mean_std")
        self.assertEqual(cfg.reconcile_for_dr_grpo(), [])
        self.assertEqual(cfg.advantage_norm_mode, "mean_std")

    def test_reconcile_flips_mean_std_when_marshal_enabled(self):
        cfg = MarshalConfig(dr_grpo=True, enabled=True, advantage_norm_mode="mean_std")
        notices = cfg.reconcile_for_dr_grpo()
        self.assertEqual(cfg.advantage_norm_mode, "mean")  # std divisor dropped
        self.assertEqual(len(notices), 1)
        self.assertIn("mean_std", notices[0])

    def test_reconcile_noop_when_already_mean(self):
        cfg = MarshalConfig(dr_grpo=True, enabled=True, advantage_norm_mode="mean")
        self.assertEqual(cfg.reconcile_for_dr_grpo(), [])
        self.assertEqual(cfg.advantage_norm_mode, "mean")

    def test_reconcile_noop_when_marshal_disabled(self):
        # MARSHAL off: advantage.py never runs, so leave the user's value alone.
        cfg = MarshalConfig(dr_grpo=True, enabled=False, advantage_norm_mode="mean_std")
        self.assertEqual(cfg.reconcile_for_dr_grpo(), [])
        self.assertEqual(cfg.advantage_norm_mode, "mean_std")

    def test_from_dict_roundtrip_with_dr_grpo(self):
        cfg = MarshalConfig.from_dict({"dr_grpo": True})
        self.assertTrue(cfg.dr_grpo)
        self.assertEqual(MarshalConfig.from_dict(cfg.to_dict()).to_dict(), cfg.to_dict())

    def test_unknown_key_still_rejected(self):
        with self.assertRaises(ValueError):
            MarshalConfig.from_dict({"dr_grpoo": True})


class TestPreSumNormalizationDivergence(unittest.TestCase):
    """paper_correct and marshal_exact should differ when a pre-sum bias exists."""

    def test_dense_rewards_diverge_between_modes(self):
        # A seat with a nonzero non-terminal reward triggers the pre-sum bias in
        # marshal_exact but not in paper_correct.
        row = RowRollout(
            seat=0, completion_len=4, owner_mask=[1, 1, 1, 1],
            turn_end_positions=[1, 3], turn_rewards=[0.5, 1.0],
        )
        other = RowRollout(
            seat=0, completion_len=4, owner_mask=[1, 1, 1, 1],
            turn_end_positions=[1, 3], turn_rewards=[0.0, 0.0],
        )
        paper = compute_marshal_advantages(
            [row, other], seq_len=4, marshal_exact=False, norm_mode="mean", turn_level=True
        )
        exact = compute_marshal_advantages(
            [row, other], seq_len=4, marshal_exact=True, norm_mode="mean", turn_level=True
        )
        self.assertFalse(torch.allclose(paper, exact))


def _placeholder_row(seat):
    """The training-inert placeholder emitted for a seat with no trajectory."""
    return RowRollout(
        seat=seat,
        completion_len=1,
        owner_mask=[0],
        turn_end_positions=[],
        turn_rewards=[],
    )


class TestPlaceholderRowExclusion(unittest.TestCase):
    """Placeholder rows (no model tokens) must not affect pool stats or gradients."""

    def test_placeholder_gets_zero_advantages(self):
        rows = [_sparse_row(0, 1.0), _sparse_row(0, -1.0), _placeholder_row(0)]
        adv = compute_marshal_advantages(rows, seq_len=2, norm_mode="mean")
        self.assertTrue(torch.all(adv[2] == 0.0))

    def test_placeholder_does_not_dilute_pool_mean(self):
        # Real rows of seat 0 have returns {1.0, 1.0, -1.0} -> mean 1/3.
        # A placeholder's artificial 0 scalar would wrongly pull the mean to 0.25.
        real = [_sparse_row(0, 1.0), _sparse_row(0, 1.0), _sparse_row(0, -1.0)]
        without = compute_marshal_advantages(list(real), seq_len=2, norm_mode="mean")
        with_placeholder = compute_marshal_advantages(
            list(real) + [_placeholder_row(0)], seq_len=2, norm_mode="mean"
        )
        self.assertTrue(torch.allclose(without, with_placeholder[:3]))
        expected_mean = torch.tensor(1.0 / 3.0)
        self.assertTrue(
            torch.allclose(with_placeholder[0, 1], 1.0 - expected_mean, atol=1e-6)
        )

    def test_all_placeholder_pool_stays_zero(self):
        rows = [_placeholder_row(1), _placeholder_row(1), _sparse_row(0, 1.0), _sparse_row(0, -1.0)]
        adv = compute_marshal_advantages(rows, seq_len=2, norm_mode="mean")
        # Seat 1 pool holds only placeholders -> all-zero, no NaN/Inf.
        self.assertTrue(torch.all(adv[:2] == 0.0))
        self.assertTrue(torch.isfinite(adv).all())
        # Seat 0 is unaffected: mean 0 -> advantages +1 / -1 at model tokens.
        self.assertTrue(torch.allclose(adv[2], torch.tensor([1.0, 1.0])))
        self.assertTrue(torch.allclose(adv[3], torch.tensor([-1.0, -1.0])))


class TestWhitening(unittest.TestCase):
    """The optional whiten_rewards / whiten_advantages passes (ROLL parity flags)."""

    def _multi_turn_row(self, seat, rewards):
        # One model token per turn, no env tokens: turn k ends at position k.
        n = len(rewards)
        return RowRollout(
            seat=seat,
            completion_len=n,
            owner_mask=[1] * n,
            turn_end_positions=list(range(n)),
            turn_rewards=list(rewards),
        )

    def test_flags_off_is_previous_behavior(self):
        rows = [_sparse_row(0, 1.0), _sparse_row(0, -1.0), _sparse_row(1, 1.0), _sparse_row(1, 0.0)]
        base = compute_marshal_advantages(list(rows), seq_len=3, norm_mode="mean")
        explicit = compute_marshal_advantages(
            list(rows), seq_len=3, norm_mode="mean",
            whiten_rewards=False, whiten_advantages=False,
        )
        self.assertTrue(torch.equal(base, explicit))

    def test_masked_whiten_stats(self):
        values = torch.tensor([[1.0, 2.0, 99.0], [3.0, 4.0, 99.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        got = masked_whiten(values, mask)
        sel = got[mask.bool()]
        self.assertAlmostEqual(sel.mean().item(), 0.0, places=5)
        # Bessel-corrected (unbiased) std of the masked positions is 1.
        self.assertAlmostEqual(sel.std(unbiased=True).item(), 1.0, places=3)

    def test_masked_whiten_degenerate_mask_is_identity(self):
        values = torch.tensor([[1.0, 2.0]])
        for mask in (torch.zeros(1, 2), torch.tensor([[1.0, 0.0]])):
            self.assertTrue(torch.equal(masked_whiten(values, mask), values))

    def test_whiten_advantages_zscores_model_positions(self):
        rows = [_sparse_row(0, 1.0), _sparse_row(0, -1.0), _sparse_row(1, 1.0), _sparse_row(1, 0.0)]
        adv = compute_marshal_advantages(
            list(rows), seq_len=3, norm_mode="mean", whiten_advantages=True
        )
        owner = torch.zeros(4, 3, dtype=torch.bool)
        owner[:, :2] = True
        sel = adv[owner]
        self.assertAlmostEqual(sel.mean().item(), 0.0, places=5)
        self.assertAlmostEqual(sel.std(unbiased=True).item(), 1.0, places=3)
        # Pad positions stay exactly zero.
        self.assertTrue(torch.all(adv[~owner] == 0.0))
        self.assertTrue(torch.isfinite(adv).all())

    def test_whiten_rewards_densifies_sparse_signal(self):
        # Two 3-turn rows, terminal-only rewards. Without whitening every model
        # token in a row carries the same return; with whiten_rewards the zero
        # slots pick up -mean/std, so returns become position-dependent.
        # (Rewards must not sum to 0 across the batch: a zero batch mean leaves
        # the zero slots untouched and whitening reduces to pure rescaling.)
        rows = [self._multi_turn_row(0, [0.0, 0.0, 1.0]), self._multi_turn_row(0, [0.0, 0.0, 0.5])]
        plain = compute_marshal_advantages(list(rows), seq_len=3, norm_mode="mean")
        whitened = compute_marshal_advantages(
            list(rows), seq_len=3, norm_mode="mean", whiten_rewards=True
        )
        # Plain: constant across each row's tokens.
        self.assertTrue(torch.allclose(plain[0], plain[0, 0].expand(3)))
        # Whitened: token-varying within a row, still finite, and different from plain.
        self.assertFalse(torch.allclose(whitened[0], whitened[0, 0].expand(3)))
        self.assertFalse(torch.allclose(plain, whitened))
        self.assertTrue(torch.isfinite(whitened).all())

    def test_whiten_survives_placeholder_rows(self):
        rows = [_sparse_row(0, 1.0), _sparse_row(0, -1.0), _placeholder_row(1)]
        adv = compute_marshal_advantages(
            list(rows), seq_len=2, norm_mode="mean",
            whiten_rewards=True, whiten_advantages=True,
        )
        self.assertTrue(torch.isfinite(adv).all())
        # Placeholder row carries no advantage in any mode.
        self.assertTrue(torch.all(adv[2] == 0.0))


def _multi_turn_row(seat, turn_lengths, turn_rewards):
    """A row with `len(turn_lengths)` turns, one env token between consecutive turns."""
    owner_mask = []
    turn_end_positions = []
    for i, n in enumerate(turn_lengths):
        if i:
            owner_mask.append(0)  # env feedback separates turns
        owner_mask.extend([1] * n)
        turn_end_positions.append(len(owner_mask) - 1)
    return RowRollout(
        seat=seat,
        completion_len=len(owner_mask),
        owner_mask=owner_mask,
        turn_end_positions=turn_end_positions,
        turn_rewards=list(turn_rewards),
    )


class TestTurnTokenLengths(unittest.TestCase):
    """Per-turn generation length is recoverable from the owner mask alone."""

    def test_recovers_each_turns_generation_length(self):
        row = _multi_turn_row(0, [3, 5, 2], [0.0, 0.0, 1.0])
        self.assertEqual(
            turn_token_lengths(row.owner_mask, row.turn_end_positions), [3, 5, 2]
        )

    def test_single_turn(self):
        row = _sparse_row(0, 1.0, model_tokens=7)
        self.assertEqual(turn_token_lengths(row.owner_mask, row.turn_end_positions), [7])

    def test_empty_generation_position_is_zero(self):
        # An empty generation records turn_end_position -1 (selfplay_agent bookkeeping);
        # there are no tokens to charge for.
        self.assertEqual(turn_token_lengths([1, 1], [-1]), [0])

    def test_out_of_range_position_is_zero(self):
        self.assertEqual(turn_token_lengths([1, 1], [9]), [0])


class TestLengthPenaltySpec(unittest.TestCase):
    """A flat per-token cost: no threshold, no free length, never positive."""

    def test_linear_in_token_count(self):
        spec = LengthPenaltySpec(per_token=1e-4)
        self.assertAlmostEqual(spec.penalty_for(1), -1e-4, places=12)
        self.assertAlmostEqual(spec.penalty_for(500), -0.05, places=12)
        self.assertAlmostEqual(spec.penalty_for(1000), -0.10, places=12)

    def test_no_threshold_every_turn_is_charged(self):
        # The defining change from the old hinge: a short turn is NOT free.
        spec = LengthPenaltySpec()
        for length in (1, 11, 500, 2047, 4096):
            self.assertLess(spec.penalty_for(length), 0.0)

    def test_never_rewards_length(self):
        spec = LengthPenaltySpec()
        self.assertTrue(all(spec.penalty_for(n) <= 0.0 for n in range(0, 5000, 97)))

    def test_monotonically_more_negative_with_length(self):
        spec = LengthPenaltySpec()
        values = [spec.penalty_for(n) for n in (200, 400, 800)]
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))

    def test_per_token_scales_the_penalty(self):
        weak = LengthPenaltySpec(per_token=1e-5).penalty_for(500)
        strong = LengthPenaltySpec(per_token=2e-5).penalty_for(500)
        self.assertAlmostEqual(strong, 2 * weak, places=12)

    def test_zero_length_turn_scores_nothing(self):
        self.assertEqual(LengthPenaltySpec().penalty_for(0), 0.0)
        self.assertEqual(LengthPenaltySpec().penalty_for(-1), 0.0)

    def test_zero_rate_is_inert(self):
        self.assertEqual(LengthPenaltySpec(per_token=0.0).penalty_for(9999), 0.0)

    def test_shipped_default_is_small_against_a_unit_outcome(self):
        # The point of the default: a long turn is worth ~1% of winning the game.
        self.assertAlmostEqual(LengthPenaltySpec().penalty_for(500), -0.01, places=9)


class TestLengthPenaltyBudget(unittest.TestCase):
    """The per-episode cap is what makes "cannot outweigh the outcome" a guarantee."""

    def test_uncapped_row_is_untouched(self):
        spec = LengthPenaltySpec(per_token=1e-4, budget=0.5)
        row = _multi_turn_row(0, [100, 200], [0.0, 1.0])
        penalties, clipped = row_length_penalties(
            row.owner_mask, row.turn_end_positions, spec
        )
        self.assertFalse(clipped)
        self.assertAlmostEqual(penalties[0], -0.01, places=9)
        self.assertAlmostEqual(penalties[1], -0.02, places=9)

    def test_row_total_never_exceeds_the_budget(self):
        # 20 turns x 1000 tokens x 1e-4 = -2.0 uncapped, which would swamp a +-1 outcome.
        spec = LengthPenaltySpec(per_token=1e-4, budget=0.1)
        row = _multi_turn_row(0, [1000] * 20, [0.0] * 19 + [1.0])
        penalties, clipped = row_length_penalties(
            row.owner_mask, row.turn_end_positions, spec
        )
        self.assertTrue(clipped)
        self.assertAlmostEqual(sum(penalties), -0.1, places=9)

    def test_cap_is_proportional_and_preserves_ordering(self):
        spec = LengthPenaltySpec(per_token=1e-3, budget=0.1)
        row = _multi_turn_row(0, [100, 300], [0.0, 1.0])
        penalties, _ = row_length_penalties(row.owner_mask, row.turn_end_positions, spec)
        # Uncapped: -0.1 and -0.3 (total -0.4) -> scaled by 0.25.
        self.assertAlmostEqual(penalties[0], -0.025, places=9)
        self.assertAlmostEqual(penalties[1], -0.075, places=9)
        self.assertAlmostEqual(penalties[1] / penalties[0], 3.0, places=9)
        self.assertTrue(all(p <= 0.0 for p in penalties))

    def test_budget_zero_disables_the_cap(self):
        spec = LengthPenaltySpec(per_token=1e-3, budget=0.0)
        row = _multi_turn_row(0, [1000, 1000], [0.0, 1.0])
        penalties, clipped = row_length_penalties(
            row.owner_mask, row.turn_end_positions, spec
        )
        self.assertFalse(clipped)
        self.assertAlmostEqual(sum(penalties), -2.0, places=9)

    def test_bound_is_independent_of_turn_count(self):
        # The whole reason the cap exists: a 20-turn game must not accumulate 4x
        # what a 5-turn game does, or the penalty would need per-game calibration.
        spec = LengthPenaltySpec(per_token=1e-3, budget=0.1)
        for turns in (2, 5, 20, 50):
            row = _multi_turn_row(0, [500] * turns, [0.0] * turns)
            penalties, _ = row_length_penalties(
                row.owner_mask, row.turn_end_positions, spec
            )
            self.assertLessEqual(abs(sum(penalties)), 0.1 + 1e-9)

    def test_cannot_reorder_two_different_outcomes(self):
        # The guarantee stated in LengthPenaltySpec: with budget < 1.0, a maximally
        # penalised WIN still beats an unpenalised LOSS.
        spec = LengthPenaltySpec(per_token=1e-2, budget=0.1)
        win = _multi_turn_row(0, [1000] * 10, [0.0] * 9 + [1.0])
        loss = _multi_turn_row(1, [1], [0.0])
        win_total = sum(apply_length_penalty(
            win.turn_rewards, win.owner_mask, win.turn_end_positions, spec))
        loss_total = sum(apply_length_penalty(
            loss.turn_rewards, loss.owner_mask, loss.turn_end_positions, spec))
        self.assertGreater(win_total, loss_total)
        self.assertGreaterEqual(win_total, 1.0 - spec.budget)


class TestApplyLengthPenalty(unittest.TestCase):
    def test_added_per_turn(self):
        # budget high enough not to bind, so this isolates the per-turn charge.
        spec = LengthPenaltySpec(per_token=1e-2, budget=1.0)
        row = _multi_turn_row(0, [2, 10], [0.0, 1.0])
        got = apply_length_penalty(
            row.turn_rewards, row.owner_mask, row.turn_end_positions, spec
        )
        # Every turn is charged now -- including the short one.
        self.assertAlmostEqual(got[0], -0.02, places=9)
        self.assertAlmostEqual(got[1], 1.0 - 0.10, places=9)

    def test_terminal_reward_still_dominates(self):
        spec = LengthPenaltySpec()
        row = _multi_turn_row(0, [800, 800], [0.0, 1.0])
        got = apply_length_penalty(
            row.turn_rewards, row.owner_mask, row.turn_end_positions, spec
        )
        self.assertGreater(sum(got), 0.9)

    def test_empty_generation_is_not_charged(self):
        row = RowRollout(seat=0, completion_len=2, owner_mask=[1, 1],
                         turn_end_positions=[-1], turn_rewards=[1.0])
        got = apply_length_penalty(
            row.turn_rewards, row.owner_mask, row.turn_end_positions,
            LengthPenaltySpec(),
        )
        self.assertEqual(got, [1.0])


class TestLengthPenaltyEndToEnd(unittest.TestCase):
    """The flag must change advantages, and only through the reward it adds."""

    def test_off_by_default(self):
        rows = [_multi_turn_row(0, [3, 9000], [0.0, 1.0]), _sparse_row(1, 0.0)]
        base = compute_marshal_advantages(list(rows), seq_len=9010)
        with_none = compute_marshal_advantages(
            list(rows), seq_len=9010, length_penalty=None
        )
        self.assertTrue(torch.allclose(base, with_none))

    def test_penalises_the_overlong_seat(self):
        # Two seats, same +1 outcome; seat 0 rambles, seat 1 is concise. Per-seat
        # pooling means the penalty has to show up as a seat-0 vs seat-1 difference.
        rows = [
            _multi_turn_row(0, [10, 6000], [0.0, 1.0]),
            _multi_turn_row(0, [10, 20], [0.0, 1.0]),
            _multi_turn_row(1, [10, 20], [0.0, 1.0]),
            _multi_turn_row(1, [10, 20], [0.0, 1.0]),
        ]
        seq_len = 6100
        adv = compute_marshal_advantages(
            list(rows), seq_len=seq_len, length_penalty=LengthPenaltySpec(),
            norm_mode="mean",
        )
        # Seat 0's rambling row must end up below its concise sibling.
        self.assertLess(adv[0, 0].item(), adv[1, 0].item())
        # Seat 1 rows are identical to each other, so mean-centering zeroes them.
        self.assertAlmostEqual(adv[2, 0].item(), 0.0, places=5)

    def test_shorter_wins_among_equal_outcomes(self):
        # With no threshold, two same-outcome rows that differ only in length are
        # separated -- which the old hinge did not do below max_len.
        rows = [
            _multi_turn_row(0, [40, 40], [0.0, 1.0]),
            _multi_turn_row(0, [5, 5], [0.0, 1.0]),
        ]
        adv = compute_marshal_advantages(
            list(rows), seq_len=100, length_penalty=LengthPenaltySpec(per_token=1e-3),
            norm_mode="mean",
        )
        self.assertLess(adv[0, 0].item(), adv[1, 0].item())

    def test_outcome_still_outranks_the_penalty(self):
        # A verbose WIN must keep a higher advantage than a terse LOSS. This is the
        # property the budget exists to guarantee, checked end to end.
        rows = [
            _multi_turn_row(0, [2000] * 8, [0.0] * 7 + [1.0]),
            _multi_turn_row(0, [3] * 8, [0.0] * 7 + [-1.0]),
        ]
        adv = compute_marshal_advantages(
            list(rows), seq_len=17000,
            length_penalty=LengthPenaltySpec(per_token=1e-3, budget=0.1),
            norm_mode="mean",
        )
        self.assertGreater(adv[0, 0].item(), adv[1, 0].item())

    def test_advantages_stay_masked_to_model_tokens(self):
        rows = [_multi_turn_row(0, [2, 3000], [0.0, 1.0]), _sparse_row(1, -1.0)]
        adv = compute_marshal_advantages(
            list(rows), seq_len=3010, length_penalty=LengthPenaltySpec()
        )
        env_position = 2  # the separator token between turn 0 and turn 1
        self.assertEqual(rows[0].owner_mask[env_position], 0)
        self.assertEqual(adv[0, env_position].item(), 0.0)
        self.assertTrue(torch.isfinite(adv).all())

    def test_composes_with_marshal_exact_and_whitening(self):
        rows = [
            _multi_turn_row(0, [10, 5000], [0.0, 1.0]),
            _multi_turn_row(0, [10, 30], [0.0, 0.0]),
            _multi_turn_row(1, [10, 30], [0.0, 1.0]),
            _multi_turn_row(1, [10, 30], [0.0, -1.0]),
        ]
        adv = compute_marshal_advantages(
            list(rows), seq_len=5100, marshal_exact=True,
            whiten_rewards=True, whiten_advantages=True,
            length_penalty=LengthPenaltySpec(),
        )
        self.assertTrue(torch.isfinite(adv).all())


class TestLengthPenaltyConfig(unittest.TestCase):
    def test_defaults_are_off_and_small(self):
        cfg = MarshalConfig()
        self.assertFalse(cfg.length_penalty)
        self.assertEqual(cfg.length_penalty_per_token, 2e-5)
        self.assertEqual(cfg.length_penalty_budget, 0.1)

    def test_kwargs_none_when_disabled(self):
        self.assertIsNone(MarshalConfig().length_penalty_kwargs())

    def test_kwargs_build_a_spec_when_enabled(self):
        cfg = MarshalConfig(length_penalty=True, length_penalty_per_token=1e-4,
                            length_penalty_budget=0.25)
        spec = LengthPenaltySpec(**cfg.length_penalty_kwargs())
        self.assertEqual(spec.per_token, 1e-4)
        self.assertEqual(spec.budget, 0.25)
        self.assertLess(spec.penalty_for(100), 0.0)

    def test_rejects_negative_rate_and_budget(self):
        with self.assertRaises(ValueError):
            MarshalConfig(length_penalty=True, length_penalty_per_token=-1e-5)
        with self.assertRaises(ValueError):
            MarshalConfig(length_penalty=True, length_penalty_budget=-0.1)

    def test_budget_safety_property(self):
        self.assertTrue(MarshalConfig(length_penalty=True,
                                      length_penalty_budget=0.1)
                        .length_penalty_budget_is_safe)
        self.assertFalse(MarshalConfig(length_penalty=True,
                                       length_penalty_budget=0.9)
                         .length_penalty_budget_is_safe)
        # An uncapped episode total has no bound at all -- not safe.
        self.assertFalse(MarshalConfig(length_penalty=True,
                                       length_penalty_budget=0.0)
                         .length_penalty_budget_is_safe)
        # Inert while the penalty is off.
        self.assertTrue(MarshalConfig(length_penalty_budget=9.0)
                        .length_penalty_budget_is_safe)

    def test_legacy_fields_still_load_and_are_inert(self):
        # An old YAML / pinned resume config / cluster preset must keep working.
        cfg = MarshalConfig.from_dict({
            "length_penalty": True,
            "length_penalty_coef": 0.05,
            "length_penalty_max_len": 256,
        })
        self.assertEqual(cfg.length_penalty_kwargs(),
                         {"per_token": 2e-5, "budget": 0.1})
        self.assertEqual(cfg.legacy_length_penalty_values(),
                         {"length_penalty_coef": 0.05, "length_penalty_max_len": 256})

    def test_legacy_fields_at_defaults_are_not_reported(self):
        self.assertEqual(MarshalConfig().legacy_length_penalty_values(), {})

    def test_legacy_span_no_longer_rejected(self):
        # The old max_len > min_len rule guarded a formula that no longer exists;
        # rejecting it now would block a config that runs fine.
        MarshalConfig(length_penalty=True, length_penalty_min_len=2048,
                      length_penalty_max_len=2048)

    def test_from_dict_roundtrip(self):
        cfg = MarshalConfig.from_dict({"length_penalty": True,
                                       "length_penalty_per_token": 7.5e-5})
        self.assertTrue(cfg.length_penalty)
        self.assertEqual(cfg.length_penalty_per_token, 7.5e-5)
        self.assertIn("length_penalty_budget", cfg.to_dict())
        self.assertIn("length_penalty_max_len", cfg.to_dict())

    def test_unknown_length_key_still_rejected(self):
        with self.assertRaises(ValueError):
            MarshalConfig.from_dict({"length_penalty_scale": 0.5})


if __name__ == "__main__":
    unittest.main(verbosity=2)

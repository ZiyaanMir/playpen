"""Unit tests for the MARSHAL self-play advantage math and rollout construction.

Pure logic -- no model, GPU, or vLLM. Runnable via ``pytest`` or directly:

    .venv/bin/python -m pytest tests/test_marshal_advantage.py
    .venv/bin/python tests/test_marshal_advantage.py
"""

import unittest

import torch

from playpen.marshal.advantage import (
    RowRollout,
    build_reward_tensor,
    compute_marshal_advantages,
    masked_whiten,
    reinforce_returns,
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
    def test_collapses_to_a_terminal_scalar(self):
        got = build_reward_tensor([1, 3], [2.0, 3.0], seq_len=4)
        # sum(2,3)=5 placed at the last turn-end position (3)
        self.assertTrue(torch.allclose(got, torch.tensor([0.0, 0.0, 0.0, 5.0])))

    def test_out_of_range_positions_ignored(self):
        got = build_reward_tensor([5], [1.0], seq_len=3)
        self.assertTrue(torch.allclose(got, torch.zeros(3)))


class TestSeatPooling(unittest.TestCase):
    def test_pooling_is_occurrence_weighted(self):
        # 9 rows reward=+1, 1 row reward=-1, all seat 0.
        rows = [_sparse_row(0, 1.0) for _ in range(9)] + [_sparse_row(0, -1.0)]
        adv = compute_marshal_advantages(
            rows, seq_len=2, agent_specific=True, norm_mode="mean"
        )
        # mean = (9*1 + -1)/10 = 0.8
        self.assertAlmostEqual(adv[0, 0].item(), 0.2, places=5)   # +1 row
        self.assertAlmostEqual(adv[-1, 0].item(), -1.8, places=5)  # -1 row

    def test_agent_specific_vs_batchwide(self):
        # seat 0 all +1, seat 1 all -1.
        rows = [_sparse_row(0, 1.0), _sparse_row(0, 1.0), _sparse_row(1, -1.0), _sparse_row(1, -1.0)]
        adv_specific = compute_marshal_advantages(
            rows, seq_len=2, agent_specific=True, norm_mode="mean"
        )
        # each seat centered against itself -> all zero
        self.assertTrue(torch.allclose(adv_specific, torch.zeros_like(adv_specific), atol=1e-6))

        adv_batch = compute_marshal_advantages(
            rows, seq_len=2, agent_specific=False, norm_mode="mean"
        )
        # pooled mean = 0 -> seat0 rows +1, seat1 rows -1
        self.assertAlmostEqual(adv_batch[0, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(adv_batch[2, 0].item(), -1.0, places=5)

    def test_mean_std_zscore(self):
        rows = [_sparse_row(0, 1.0), _sparse_row(0, -1.0)]
        adv = compute_marshal_advantages(
            rows, seq_len=2, agent_specific=True, norm_mode="mean_std"
        )
        # scalars {1,-1}: mean 0, population std 1 -> advantages +1 / -1
        self.assertAlmostEqual(adv[0, 0].item(), 1.0, places=4)
        self.assertAlmostEqual(adv[1, 0].item(), -1.0, places=4)

    def test_degenerate_all_equal_pool_is_zero(self):
        rows = [_sparse_row(0, 1.0) for _ in range(4)]
        adv = compute_marshal_advantages(
            rows, seq_len=2, agent_specific=True, norm_mode="mean"
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
            [row, other], seq_len=5, agent_specific=True, norm_mode="mean"
        )
        # position 1 is env feedback -> must be zero; positions 3,4 are padding -> zero.
        self.assertAlmostEqual(adv[0, 1].item(), 0.0, places=6)
        self.assertAlmostEqual(adv[0, 3].item(), 0.0, places=6)
        self.assertAlmostEqual(adv[0, 4].item(), 0.0, places=6)


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = MarshalConfig()
        self.assertTrue(cfg.enabled)
        self.assertTrue(cfg.agent_specific_normalization)
        self.assertEqual(cfg.gamma, 1.0)

    def test_from_dict_rejects_unknown_keys(self):
        with self.assertRaises(ValueError):
            MarshalConfig.from_dict({"enabledd": True})

    def test_validates_enums(self):
        with self.assertRaises(ValueError):
            MarshalConfig(advantage_norm_mode="bogus")
        with self.assertRaises(ValueError):
            MarshalConfig(row_context_mode="bogus")

    def test_removed_fields_are_dropped_not_rejected(self):
        cfg = MarshalConfig.from_dict({
            "length_penalty": True,
            "length_penalty_max_len": 256,
            "fidelity_mode": "marshal_exact",
            "turn_level_rewards": True,
            "marshal_exact_unique_pooling": False,
            "gamma": 0.9,
        })
        self.assertEqual(cfg.gamma, 0.9)
        self.assertNotIn("length_penalty", cfg.to_dict())
        self.assertNotIn("fidelity_mode", cfg.to_dict())


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


class TestGrpoLoss(unittest.TestCase):
    """The original-GRPO loss switch: TRL's loss_type='grpo', and nothing else."""

    def test_default_off(self):
        cfg = MarshalConfig()
        self.assertFalse(cfg.grpo_loss)
        self.assertEqual(cfg.trl_loss_type, "dapo")

    def test_overrides_off_is_empty(self):
        # Off => empty dict => splatting into GRPOConfig(...) changes nothing.
        self.assertEqual(MarshalConfig(grpo_loss=False).trl_grpo_overrides(), {})

    def test_overrides_on_sets_only_loss_type(self):
        # scale_rewards is deliberately NOT touched: this flag is about aggregation.
        self.assertEqual(
            MarshalConfig(grpo_loss=True).trl_grpo_overrides(),
            {"loss_type": "grpo"},
        )

    def test_resolved_loss_type(self):
        self.assertEqual(MarshalConfig(grpo_loss=True).trl_loss_type, "grpo")
        self.assertEqual(MarshalConfig(dr_grpo=True).trl_loss_type, "dr_grpo")

    def test_mutually_exclusive_with_dr_grpo(self):
        # Both write loss_type; a silent winner would make an arm's name disagree
        # with the loss it trained under, so this must raise rather than resolve.
        with self.assertRaises(ValueError) as ctx:
            MarshalConfig(dr_grpo=True, grpo_loss=True)
        self.assertIn("mutually exclusive", str(ctx.exception))

    def test_applies_regardless_of_enabled(self):
        """It configures the shared TRL loss, which the --no-marshal arm also uses."""
        cfg = MarshalConfig(enabled=False, grpo_loss=True)
        self.assertEqual(cfg.trl_grpo_overrides(), {"loss_type": "grpo"})

    def test_does_not_touch_advantage_norm_mode(self):
        # Unlike dr_grpo, there is no reconciliation: grpo_loss says nothing about
        # how advantages are normalized.
        cfg = MarshalConfig(grpo_loss=True, enabled=True, advantage_norm_mode="mean_std")
        self.assertEqual(cfg.reconcile_for_dr_grpo(), [])
        self.assertEqual(cfg.advantage_norm_mode, "mean_std")

    def test_composes_with_sampling_without_key_collision(self):
        cfg = MarshalConfig(grpo_loss=True)
        merged = {**cfg.trl_grpo_overrides(), **cfg.trl_sampling_overrides()}
        self.assertEqual(merged, {"loss_type": "grpo", "top_p": 0.95, "top_k": 50})

    def test_from_dict_roundtrip(self):
        cfg = MarshalConfig.from_dict({"grpo_loss": True})
        self.assertTrue(cfg.grpo_loss)
        self.assertEqual(MarshalConfig.from_dict(cfg.to_dict()).to_dict(), cfg.to_dict())

    def test_loss_type_is_a_real_grpoconfig_value(self):
        """Guards against a TRL rename turning this into a silently ignored kwarg."""
        try:
            from trl import GRPOConfig
        except Exception:  # pragma: no cover - trl not installed
            self.skipTest("trl not installed")
        gc = GRPOConfig(
            output_dir="/tmp/marshal-test",
            **MarshalConfig(grpo_loss=True).trl_grpo_overrides(),
        )
        self.assertEqual(gc.loss_type, "grpo")
        # scale_rewards must still be TRL's default -- the flag does not touch it.
        self.assertEqual(gc.scale_rewards, GRPOConfig(output_dir="/tmp/m").scale_rewards)


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


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Trainer-level tests: the contract between MARSHAL's advantages and TRL's tensors.

``test_marshal_advantage.py`` tests the maths in isolation and
``test_marshal_rollout.py`` tests rollout assembly in isolation. Neither covers the
seam between them -- that the ``(B, T)`` tensor ``MarshalGRPOTrainer`` hands back is
indexed the same way as the ``(B, T)`` completion tensor TRL built, that the logged
advantage is the trained one, and that a censored row is counted rather than
silently absorbed. Those are exactly the failures that survived a 500-step run.

``MarshalGRPOTrainer.__init__`` would load a model, so the methods under test are
exercised on a bare instance (``object.__new__``) with only the attributes they
actually touch. That is deliberate: it keeps these tests model-free and fast, and it
also documents precisely which trainer state each method depends on.

Runnable via ``pytest`` or directly with ``.venv/bin/python``.
"""

import unittest
from collections import defaultdict, deque

import torch

from playpen.marshal.advantage import RowRollout, compute_marshal_advantages
from playpen.marshal.config import MarshalConfig
from playpen.marshal.trainer import MarshalGRPOTrainer


class _FakeAccelerator:
    def __init__(self, num_processes=1):
        self.num_processes = num_processes


def _bare_trainer(config=None, num_processes=1, logs_maxlen=64):
    """A MarshalGRPOTrainer with only the state its own methods read.

    Bypasses ``trl.GRPOTrainer.__init__`` (which would download a model). If a method
    under test ever starts reading trainer state not set here, it will raise rather
    than silently pass -- which is the behaviour we want from a seam test.
    """
    t = object.__new__(MarshalGRPOTrainer)
    t.marshal_config = config or MarshalConfig()
    t.accelerator = _FakeAccelerator(num_processes)
    t._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    t._logs = {"advantages": deque(maxlen=logs_maxlen)}
    return t


def _row(seat, owner_mask, turn_end_positions, turn_rewards, drifted=False):
    """One entry of TRL's ``inputs`` list, as the rollout func would leave it."""
    return {
        "seat": seat,
        "owner_mask": list(owner_mask),
        "turn_end_positions": list(turn_end_positions),
        "turn_rewards": list(turn_rewards),
        "drifted": drifted,
    }


def _padded_completions(owner_masks, pad_id=0):
    """TRL's right-padded ``completion_ids`` for rows of these lengths."""
    width = max(len(m) for m in owner_masks)
    return torch.tensor(
        [[7] * len(m) + [pad_id] * (width - len(m)) for m in owner_masks],
        dtype=torch.long,
    )


class TestAdvantageTensorAlignsWithTrlCompletions(unittest.TestCase):
    """Column j of the advantage tensor must be completion token j, for every row.

    TRL right-pads completions and consumes a ``(B, T)`` advantage by broadcasting it
    against ``per_token_loss`` (grpo_trainer.py:2140-2143). If the advantage were
    built at a different width, or padded on the left, every row would be credited
    for the wrong tokens and nothing would raise.
    """

    def setUp(self):
        # Deliberately ragged widths, and two real rows per seat with DIFFERENT
        # rewards -- a seat pool holding one real row has no baseline and correctly
        # normalizes to all-zero (pinned separately below), which would make an
        # alignment assertion vacuous.
        self.masks = [
            [1, 1, 0, 0, 1, 1, 1],   # seat 0, turns end at 1 and 6
            [1, 1, 0, 1, 1],         # seat 0, turns end at 1 and 4
            [1, 1, 1],               # seat 1, one turn ending at 2
            [1, 1],                  # seat 1, one turn ending at 1
            [0],                     # seat 1 placeholder (never moved)
        ]
        self.inputs = [
            _row(0, self.masks[0], [1, 6], [0.0, 1.0]),
            _row(0, self.masks[1], [1, 4], [0.0, 0.0]),
            _row(1, self.masks[2], [2], [1.0]),
            _row(1, self.masks[3], [1], [0.0]),
            _row(1, self.masks[4], [], []),
        ]
        self.completion_ids = _padded_completions(self.masks)
        self.trainer = _bare_trainer()

    def _advantages(self):
        output = {"advantages": torch.zeros(len(self.inputs)), "completion_ids": self.completion_ids}
        return self.trainer._compute_marshal_advantages(self.inputs, output)

    def test_shape_matches_the_padded_completion_tensor(self):
        adv = self._advantages()
        self.assertEqual(tuple(adv.shape), tuple(self.completion_ids.shape))

    def test_nonzero_advantage_only_where_the_row_generated(self):
        adv = self._advantages()
        for i, mask in enumerate(self.masks):
            for j in range(adv.shape[1]):
                owned = j < len(mask) and mask[j] == 1
                if not owned:
                    self.assertEqual(
                        adv[i, j].item(), 0.0,
                        f"row {i} col {j}: advantage on a non-generated position "
                        f"({'env token' if j < len(mask) else 'right-pad'})",
                    )

    def test_right_padding_not_left_padding(self):
        """A short row's signal must sit at the START of the row, not the end.

        The widest row is 7 columns; row 2 owns only columns 0..2. If the advantage
        tensor were left-padded (as TRL pads *prompts*) its signal would land in the
        last three columns, and every short row would credit the wrong tokens
        without anything raising.
        """
        adv = self._advantages()
        self.assertTrue((adv[2, :3] != 0).any(), "short row has no signal at its own columns")
        self.assertTrue((adv[2, 3:] == 0).all(), "short row leaked signal into padding")

    def test_a_seat_pool_holding_one_real_row_normalizes_to_zero(self):
        """No baseline to compare against => no signal. Pinned so it stays deliberate.

        This is easy to trip over when debugging: a batch that *looks* populated can
        produce an all-zero advantage for a whole seat simply because only one of its
        rows actually played. It is correct (a group of one has no relative signal),
        but it is indistinguishable from a bug unless it is written down.
        """
        masks = [[1, 1, 1], [0]]
        inputs = [_row(1, masks[0], [2], [1.0]), _row(1, masks[1], [], [])]
        output = {"advantages": torch.zeros(2), "completion_ids": _padded_completions(masks)}
        adv = _bare_trainer()._compute_marshal_advantages(inputs, output)
        self.assertTrue((adv == 0).all(), f"expected an all-zero pool, got {adv.tolist()}")

    def test_dtype_and_device_follow_the_base_advantages(self):
        output = {
            "advantages": torch.zeros(len(self.inputs), dtype=torch.float32),
            "completion_ids": self.completion_ids,
        }
        adv = self.trainer._compute_marshal_advantages(self.inputs, output)
        self.assertEqual(adv.dtype, torch.float32)
        self.assertEqual(adv.device, self.completion_ids.device)

    def test_falls_back_to_scalar_advantages_when_rows_do_not_line_up(self):
        """A length mismatch must not silently misalign; TRL's own advantages stand."""
        base = torch.tensor([0.5, -0.5, 0.0])
        output = {"advantages": base, "completion_ids": self.completion_ids}
        adv = self.trainer._compute_marshal_advantages(self.inputs[:2], output)
        self.assertIs(adv, base)


class TestRelogAdvantages(unittest.TestCase):
    """The logged ``advantage`` column must be the one the model trained on.

    Before this existed, ``completions_*.parquet`` recorded TRL's scalar
    group-relative advantage, which the MARSHAL path overwrites and never uses --
    so offline analysis of a MARSHAL run was analysing plain GRPO.
    """

    def test_overwrites_the_base_class_snapshot(self):
        t = _bare_trainer()
        t._logs["advantages"].extend([99.0, 99.0])          # TRL's scalars
        adv = torch.tensor([[2.0, 4.0, 0.0], [-1.0, -3.0, 0.0]])
        t._relog_advantages(adv)
        # Mean over the row's nonzero (model-token) positions.
        self.assertAlmostEqual(t._logs["advantages"][0], 3.0)
        self.assertAlmostEqual(t._logs["advantages"][1], -2.0)

    def test_only_the_trailing_rows_are_replaced(self):
        """The deque holds a whole generation batch; older entries must survive."""
        t = _bare_trainer()
        t._logs["advantages"].extend([11.0, 12.0, 99.0, 99.0])
        t._relog_advantages(torch.tensor([[1.0, 1.0], [5.0, 5.0]]))
        self.assertEqual(list(t._logs["advantages"]), [11.0, 12.0, 1.0, 5.0])

    def test_all_zero_row_logs_zero_rather_than_dividing_by_zero(self):
        t = _bare_trainer()
        t._logs["advantages"].extend([99.0])
        t._relog_advantages(torch.zeros(1, 4))
        self.assertEqual(t._logs["advantages"][0], 0.0)

    def test_skipped_under_multi_process(self):
        """``_logs`` holds every rank's gathered rows; ours is only a slice."""
        t = _bare_trainer(num_processes=2)
        t._logs["advantages"].extend([99.0, 99.0])
        t._relog_advantages(torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
        self.assertEqual(list(t._logs["advantages"]), [99.0, 99.0])

    def test_logged_value_matches_the_tensor_actually_returned(self):
        """End to end: what lands in the parquet is what _compute_ returned."""
        t = _bare_trainer()
        masks = [[1, 1], [1, 1]]
        inputs = [_row(0, masks[0], [1], [1.0]), _row(1, masks[1], [1], [0.0])]
        t._logs["advantages"].extend([99.0, 99.0])
        output = {"advantages": torch.zeros(2), "completion_ids": _padded_completions(masks)}
        adv = t._compute_marshal_advantages(inputs, output)
        for i in range(2):
            row = adv[i][adv[i] != 0]
            expected = row.mean().item() if row.numel() else 0.0
            self.assertAlmostEqual(t._logs["advantages"][i], expected, places=5)


class TestRolloutHealthMetrics(unittest.TestCase):
    """Censored rows and a collapsed IS ratio must be counted, not absorbed."""

    def _run(self, inputs, ratio=None, config=None):
        t = _bare_trainer(config)
        output = {}
        if ratio is not None:
            output["importance_sampling_ratio"] = ratio
        MarshalGRPOTrainer._WARNED_IS_COLLAPSE = False
        MarshalGRPOTrainer._WARNED_DRIFT = False
        t._log_rollout_health(inputs, output)
        return t._metrics["train"]

    def test_drift_is_counted_separately_from_an_idle_seat(self):
        inputs = [
            _row(0, [1, 1], [1], [1.0]),            # played
            _row(1, [0], [], [], drifted=True),     # censored
            _row(1, [0], [], [], drifted=False),    # never got a turn
        ]
        m = self._run(inputs)
        self.assertEqual(m["marshal/rows/drift_count"][-1], 1.0)
        self.assertAlmostEqual(m["marshal/rows/drift_rate"][-1], 1 / 3)
        self.assertAlmostEqual(m["marshal/rows/placeholder_rate"][-1], 2 / 3)
        self.assertAlmostEqual(m["marshal/rows/idle_seat_rate"][-1], 1 / 3)

    def test_clean_batch_reports_zeros(self):
        m = self._run([_row(0, [1, 1], [1], [1.0]), _row(1, [1, 1], [1], [0.0])])
        self.assertEqual(m["marshal/rows/drift_rate"][-1], 0.0)
        self.assertEqual(m["marshal/rows/placeholder_rate"][-1], 0.0)

    def test_is_ratio_is_logged_and_warns_on_collapse(self):
        inputs = [_row(0, [1, 1], [1], [1.0])]
        collapsed = torch.full((1, 1), 5e-8)
        with self.assertLogs("playpen.marshal.trainer", level="WARNING") as cm:
            m = self._run(inputs, ratio=collapsed)
        self.assertAlmostEqual(m["marshal/is_ratio/mean"][-1], 5e-8)
        self.assertIn("importance-sampling ratio", "".join(cm.output))

    def test_healthy_is_ratio_does_not_warn(self):
        inputs = [_row(0, [1, 1], [1], [1.0])]
        healthy = torch.tensor([[0.78], [0.81]])
        MarshalGRPOTrainer._WARNED_IS_COLLAPSE = False
        MarshalGRPOTrainer._WARNED_DRIFT = False
        t = _bare_trainer()
        with self.assertNoLogs("playpen.marshal.trainer", level="WARNING"):
            t._log_rollout_health(inputs, {"importance_sampling_ratio": healthy})
        self.assertAlmostEqual(t._metrics["train"]["marshal/is_ratio/mean"][-1], 0.795, places=5)

    def test_drift_and_is_collapse_are_independent_signals(self):
        """A perfectly provable batch can still have a collapsed ratio.

        This is the whole point of having two counters: ``row_context_mode="spliced"``
        produced rows whose owner masks were provable (no drift at all) while every
        row's loss was multiplied by ~1e-8. A drift counter alone would have shown a
        clean run.
        """
        inputs = [_row(0, [1, 1], [1], [1.0]), _row(1, [1, 1], [1], [0.0])]
        m = self._run(inputs, ratio=torch.full((2, 1), 5e-8))
        self.assertEqual(m["marshal/rows/drift_rate"][-1], 0.0)
        self.assertLess(m["marshal/is_ratio/mean"][-1], 1e-6)

    def test_health_is_logged_even_when_marshal_is_disabled(self):
        m = self._run(
            [_row(0, [0], [], [], drifted=True)],
            ratio=torch.full((1, 1), 0.9),
            config=MarshalConfig(enabled=False),
        )
        self.assertEqual(m["marshal/rows/drift_count"][-1], 1.0)

    def test_never_raises_on_malformed_input(self):
        """Metrics must not be able to kill a training run."""
        t = _bare_trainer()
        t._log_rollout_health([{"owner_mask": None}], {"importance_sampling_ratio": "nonsense"})


class TestDriftFlagReachesTheTrainer(unittest.TestCase):
    """The censoring signal must survive the whole path, not just the metric code.

    ``TestRolloutHealthMetrics`` builds ``inputs`` dicts by hand, so it verifies the
    accounting but not the wiring. Without this, defining ``drifted`` as "this row is
    inert" instead of "this row was censored" passes every other test while making the
    counter useless -- an idle seat (the game ended first) would be reported as data
    loss, and on codenames that is ~half of every batch.

    Drives the real ``rollout_func`` over the scripted env from the rollout tests.
    """

    def setUp(self):
        import trl.experimental.openenv as oe

        from tests.test_marshal_rollout import _ChatTokenizer, _fake_generate_echo

        self._oe = oe
        self._orig = oe.generate_rollout_completions
        self.tok = _ChatTokenizer()
        self._echo = _fake_generate_echo(tokenizer=self.tok)

        class _Trainer:
            processing_class = self.tok

        self.trainer = _Trainer()

    def tearDown(self):
        self._oe.generate_rollout_completions = self._orig

    def _rollout(self, total_turns, drift_on_call=None):
        from tests.test_marshal_rollout import _FakeEnv

        from playpen.marshal.trainer import build_selfplay_rollout_func

        calls = {"n": 0}

        def generate(trainer, prompts, **kwargs):
            out = self._echo(trainer, prompts, **kwargs)
            calls["n"] += 1
            if drift_on_call is not None and calls["n"] == drift_on_call:
                # Re-tokenization merged a boundary: the context is no longer an
                # extension of the row, so the owner mask cannot be proven.
                out[0]["prompt_ids"] = [999] + list(out[0]["prompt_ids"])
            return out

        self._oe.generate_rollout_completions = generate
        env = _FakeEnv(total_turns=total_turns, terminal_reward=1.0)
        fn = build_selfplay_rollout_func(env, MarshalConfig(episode_pairing="shared"))
        return fn(["0", "0"], self.trainer)   # one episode serves both seats

    def test_a_drifted_seat_is_flagged(self):
        out = self._rollout(total_turns=6, drift_on_call=3)   # seat 0's second turn
        self.assertEqual(out["drifted"], [True, False])
        self.assertEqual(out["seat"], [0, 1])
        # Flagged and inert, but still reporting the episode's real outcome.
        self.assertEqual(out["owner_mask"][0], [0])
        self.assertEqual(out["rewards"][0], 1.0)

    def test_a_seat_that_never_moved_is_NOT_flagged_as_drift(self):
        out = self._rollout(total_turns=1)                    # seat 1 gets no turn
        self.assertEqual(out["drifted"], [False, False])
        self.assertEqual(out["owner_mask"][1], [0], "seat 1 should still be inert")

    def test_a_clean_episode_flags_nothing(self):
        out = self._rollout(total_turns=6)
        self.assertEqual(out["drifted"], [False, False])
        self.assertTrue(all(1 in m for m in out["owner_mask"]))

    def test_metrics_separate_the_two_causes_over_the_real_rollout(self):
        drifted = self._rollout(total_turns=6, drift_on_call=3)
        idle = self._rollout(total_turns=1)
        inputs = []
        for out in (drifted, idle):
            for i in range(len(out["seat"])):
                inputs.append(
                    {
                        "seat": out["seat"][i],
                        "owner_mask": out["owner_mask"][i],
                        "turn_end_positions": out["turn_end_positions"][i],
                        "turn_rewards": out["turn_rewards"][i],
                        "drifted": out["drifted"][i],
                    }
                )
        t = _bare_trainer()
        MarshalGRPOTrainer._WARNED_DRIFT = False
        MarshalGRPOTrainer._WARNED_IS_COLLAPSE = False
        t._log_rollout_health(inputs, {})
        m = t._metrics["train"]
        # 4 rows: 1 censored, 1 idle, 2 played.
        self.assertEqual(m["marshal/rows/drift_count"][-1], 1.0)
        self.assertAlmostEqual(m["marshal/rows/drift_rate"][-1], 0.25)
        self.assertAlmostEqual(m["marshal/rows/idle_seat_rate"][-1], 0.25)
        self.assertAlmostEqual(m["marshal/rows/placeholder_rate"][-1], 0.5)


class TestPairedBatchWithPlaceholderEndToEnd(unittest.TestCase):
    """episode_pairing='shared' + a seat that never moved, through the advantage path.

    The pairing tests stop at the rollout dict and the placeholder tests construct
    ``RowRollout``s by hand; nothing joined the two. On codenames roughly half of
    every batch is a seat that never moved, so this is the common case, not an edge
    case.
    """

    def _batch(self, n_pairs=4, idle_every=2):
        """n_pairs episodes, each contributing seat 0 and seat 1 consecutively."""
        inputs, masks = [], []
        for p in range(n_pairs):
            reward = 1.0 if p % 2 == 0 else 0.0
            inputs.append(_row(0, [1, 1, 1, 1], [3], [reward]))
            masks.append([1, 1, 1, 1])
            if p % idle_every == 0:                       # seat 1 never moved
                inputs.append(_row(1, [0], [], []))
                masks.append([0])
            else:
                inputs.append(_row(1, [1, 1], [1], [reward]))
                masks.append([1, 1])
        return inputs, masks

    def test_placeholders_do_not_shift_the_surviving_seats_baseline(self):
        inputs, masks = self._batch()
        t = _bare_trainer()
        output = {"advantages": torch.zeros(len(inputs)), "completion_ids": _padded_completions(masks)}
        with_ph = t._compute_marshal_advantages(inputs, output)

        keep = [i for i, m in enumerate(masks) if any(m)]
        inputs2 = [inputs[i] for i in keep]
        masks2 = [masks[i] for i in keep]
        t2 = _bare_trainer()
        output2 = {"advantages": torch.zeros(len(inputs2)), "completion_ids": _padded_completions(masks2)}
        without_ph = t2._compute_marshal_advantages(inputs2, output2)

        for dst, src in enumerate(keep):
            w = with_ph[src][: len(masks[src])]
            o = without_ph[dst][: len(masks2[dst])]
            self.assertTrue(
                torch.allclose(w, o, atol=1e-6),
                f"row {src}: dropping the placeholders changed its advantage "
                f"({w.tolist()} vs {o.tolist()})",
            )

    def test_every_placeholder_row_is_all_zero(self):
        inputs, masks = self._batch()
        t = _bare_trainer()
        output = {"advantages": torch.zeros(len(inputs)), "completion_ids": _padded_completions(masks)}
        adv = t._compute_marshal_advantages(inputs, output)
        for i, m in enumerate(masks):
            if not any(m):
                self.assertTrue((adv[i] == 0).all(), f"placeholder row {i} carries advantage")

    def test_seats_are_pooled_separately(self):
        """Seat 1's advantage must not move when only seat 0's rewards change."""
        inputs, masks = self._batch(n_pairs=4, idle_every=99)   # no placeholders
        t = _bare_trainer()
        out = {"advantages": torch.zeros(len(inputs)), "completion_ids": _padded_completions(masks)}
        before = t._compute_marshal_advantages(inputs, out)

        bumped = [dict(r) for r in inputs]
        for i, r in enumerate(bumped):
            if r["seat"] == 0:
                r["turn_rewards"] = [x + 5.0 for x in r["turn_rewards"]]
        t2 = _bare_trainer()
        out2 = {"advantages": torch.zeros(len(bumped)), "completion_ids": _padded_completions(masks)}
        after = t2._compute_marshal_advantages(bumped, out2)

        seat1 = [i for i, r in enumerate(inputs) if r["seat"] == 1]
        for i in seat1:
            self.assertTrue(
                torch.allclose(before[i], after[i], atol=1e-6),
                f"seat-1 row {i} moved when only seat-0 rewards changed",
            )


class TestTurnCountDoesNotSetTheAdvantage(unittest.TestCase):
    """Regression guard for the turn-count bias (audit finding F1).

    Measured in production: with ``fidelity_mode='marshal_exact'`` and
    ``turn_level_rewards=True``, the within-step correlation between a winning row's
    turn count and its advantage was -0.95, and a 2-turn win was worth 3.5x an 8-turn
    win at identical reward. Under ``paper_correct`` + ``whiten_rewards=False`` the
    same batch gives -0.15.

    This test does NOT assert that the biased setting is forbidden -- it is a faithful
    reproduction of MARSHAL's shipped code and stays available. It pins the *recommended*
    setting so the bias cannot come back unnoticed there, and pins the biased one so
    the reproduction stays honest about what it reproduces.
    """

    @staticmethod
    def _rows(seat=0, tokens_per_turn=6):
        """Winning rows of 2..8 turns, all with the same terminal reward."""
        rows = []
        for n_turns in range(2, 9):
            mask, ends = [], []
            for _ in range(n_turns):
                mask += [1] * tokens_per_turn
                ends.append(len(mask) - 1)
                mask += [0, 0]                       # env feedback between turns
            rows.append(
                RowRollout(
                    seat=seat,
                    completion_len=len(mask),
                    owner_mask=mask,
                    turn_end_positions=ends,
                    turn_rewards=[0.0] * (n_turns - 1) + [1.0],
                )
            )
        return rows

    @staticmethod
    def _first_token_advantages(rows, **kwargs):
        seq_len = max(r.completion_len for r in rows)
        adv = compute_marshal_advantages(rows, seq_len, agent_specific=True, **kwargs)
        out = []
        for i, r in enumerate(rows):
            first = next(j for j, m in enumerate(r.owner_mask) if m == 1)
            out.append(adv[i, first].item())
        return out

    @staticmethod
    def _spread(values):
        return max(values) - min(values)

    def test_recommended_setting_is_almost_turn_count_independent(self):
        vals = self._first_token_advantages(
            self._rows(), turn_level=True, marshal_exact=False,
            norm_mode="mean", whiten_rewards=False, whiten_advantages=True,
        )
        self.assertLess(
            self._spread(vals), 0.30,
            f"paper_correct + whiten_rewards=False should be near turn-count "
            f"independent at fixed reward, got spread {self._spread(vals):.3f}: {vals}",
        )

    def test_marshal_exact_with_turn_level_is_strongly_turn_count_dependent(self):
        """Pins the known bias, so 'marshal_exact' cannot quietly stop reproducing it."""
        vals = self._first_token_advantages(
            self._rows(), turn_level=True, marshal_exact=True,
            norm_mode="mean", whiten_rewards=True, whiten_advantages=True,
        )
        self.assertGreater(self._spread(vals), 1.0, f"expected the known bias, got {vals}")
        # Monotone decreasing in turn count: more turns, less credit for the same win.
        self.assertEqual(vals, sorted(vals, reverse=True), f"expected monotone decay, got {vals}")

    def test_turning_off_whiten_rewards_alone_does_not_remove_the_bias(self):
        """The documented trap: whiten_rewards is the wrong lever under marshal_exact."""
        kw = dict(turn_level=True, marshal_exact=True, norm_mode="mean", whiten_advantages=True)
        on = self._first_token_advantages(self._rows(), whiten_rewards=True, **kw)
        off = self._first_token_advantages(self._rows(), whiten_rewards=False, **kw)
        self.assertGreater(self._spread(off), 1.0)
        for a, b in zip(on, off):
            self.assertAlmostEqual(a, b, places=5)

    def test_turn_level_false_removes_the_bias_under_marshal_exact(self):
        vals = self._first_token_advantages(
            self._rows(), turn_level=False, marshal_exact=True,
            norm_mode="mean", whiten_rewards=True, whiten_advantages=True,
        )
        self.assertLess(self._spread(vals), 1e-6, f"expected no turn-count spread, got {vals}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

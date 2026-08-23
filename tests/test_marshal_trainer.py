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

import logging
import unittest
from collections import defaultdict, deque
from unittest import mock

import torch
import trl

from playpen.marshal.advantage import RowRollout, compute_marshal_advantages
from playpen.marshal.config import MarshalConfig
from playpen.marshal.trainer import MarshalGRPOTrainer, build_reward_func


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

    At a fixed terminal reward, a row's advantage must not depend on how many turns
    it took.
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

    def test_shipped_setting_is_almost_turn_count_independent(self):
        vals = self._first_token_advantages(
            self._rows(), norm_mode="mean",
            whiten_rewards=False, whiten_advantages=True,
        )
        self.assertLess(
            self._spread(vals), 0.30,
            f"advantages should be near turn-count independent at fixed reward, "
            f"got spread {self._spread(vals):.3f}: {vals}",
        )


def _stub_base_generate(output):
    """Stand in for ``trl.GRPOTrainer._generate_and_score_completions``.

    Patched onto the base *class*, so the subclass's ``super()`` call resolves to it
    through the MRO exactly as it would to the real method. Returns the dict the real
    one returns -- scalar ``(B,)`` advantages included -- and nothing else, which is
    what lets these tests assert on what the subclass adds.
    """

    def _base(self, inputs):
        return output

    return _base


class TestGenerateAndScoreInstallsMarshalAdvantages(unittest.TestCase):
    """The override must actually reach TRL's output dict.

    Every other test in this file calls ``_compute_marshal_advantages`` directly, so
    all of them still pass if the one line that installs its result --
    ``output["advantages"] = self._compute_marshal_advantages(...)`` -- is deleted.
    A run would then train on TRL's scalar group-relative advantages, i.e. plain
    GRPO, while logging, checkpointing and reporting as a MARSHAL run. This is the
    seam that turns the subclass into a MARSHAL trainer, so it is pinned here.
    """

    def setUp(self):
        self.masks = [[1, 1, 0, 1], [1, 1, 0, 1], [1, 1], [1, 1]]
        self.inputs = [
            _row(0, self.masks[0], [1, 3], [0.0, 1.0]),
            _row(1, self.masks[1], [1, 3], [0.0, 1.0]),
            _row(0, self.masks[2], [1], [-1.0]),
            _row(1, self.masks[3], [1], [-1.0]),
        ]
        self.completion_ids = _padded_completions(self.masks)
        # Deliberately non-zero and non-constant, so "the base tensor survived
        # untouched" cannot be mistaken for "the override produced these".
        self.base_adv = torch.tensor([0.25, -0.25, 0.5, -0.5])

    def _output(self):
        return {
            "advantages": self.base_adv.clone(),
            "completion_ids": self.completion_ids,
            "num_items_in_batch": torch.tensor(12),
        }

    def _run(self, config):
        trainer = _bare_trainer(config)
        output = self._output()
        with mock.patch.object(
            trl.GRPOTrainer, "_generate_and_score_completions", _stub_base_generate(output)
        ):
            return trainer, trainer._generate_and_score_completions(self.inputs)

    def test_enabled_replaces_the_scalar_advantages_with_the_b_by_t_tensor(self):
        _, out = self._run(MarshalConfig(enabled=True))
        self.assertEqual(
            tuple(out["advantages"].shape),
            tuple(self.completion_ids.shape),
            "advantages must be the (B, T) MARSHAL tensor, not TRL's (B,) scalars",
        )

    def test_enabled_installs_exactly_what_compute_marshal_advantages_returns(self):
        config = MarshalConfig(enabled=True)
        _, out = self._run(config)
        rows = [
            RowRollout(
                seat=inp["seat"],
                completion_len=len(inp["owner_mask"]),
                owner_mask=inp["owner_mask"],
                turn_end_positions=inp["turn_end_positions"],
                turn_rewards=inp["turn_rewards"],
            )
            for inp in self.inputs
        ]
        expected = compute_marshal_advantages(
            rows,
            self.completion_ids.shape[1],
            gamma=config.gamma,
            agent_specific=config.agent_specific_normalization,
            norm_mode=config.advantage_norm_mode,
            whiten_rewards=config.whiten_rewards,
            whiten_advantages=config.whiten_advantages,
            dtype=self.base_adv.dtype,
        )
        self.assertTrue(
            torch.allclose(out["advantages"], expected),
            f"installed {out['advantages']} but the advantage module computes {expected}",
        )
        # Guards against a vacuous assertion: the tensor really did change.
        self.assertFalse(torch.allclose(expected, torch.zeros_like(expected)))

    def test_disabled_leaves_trls_scalar_advantages_exactly_as_they_were(self):
        _, out = self._run(MarshalConfig(enabled=False))
        self.assertEqual(tuple(out["advantages"].shape), (len(self.inputs),))
        self.assertTrue(torch.allclose(out["advantages"], self.base_adv))

    def test_the_dict_the_base_class_returned_is_the_one_handed_back(self):
        """TRL keys off the returned dict; a copy would drop old_per_token_logps etc."""
        trainer = _bare_trainer(MarshalConfig(enabled=True))
        output = self._output()
        output["ref_per_token_logps"] = torch.zeros(4, 4)
        with mock.patch.object(
            trl.GRPOTrainer, "_generate_and_score_completions", _stub_base_generate(output)
        ):
            out = trainer._generate_and_score_completions(self.inputs)
        self.assertIs(out, output)
        self.assertIn("ref_per_token_logps", out)


class TestDegenerateBatchDoesNotProduceNaNLoss(unittest.TestCase):
    """``num_items_in_batch`` is a divisor, so 0 is not a survivable value.

    TRL's dapo and cispo losses (its default ``loss_type`` is dapo) normalize by
    ``inputs["num_items_in_batch"] / num_processes`` (grpo_trainer.py:2236-2237).
    That count is the number of *model* tokens, which is 0 when every row in the
    step is a placeholder -- a real state on codenames, where the game can end
    before the guesser seat ever moves. 0/0 is NaN, and one NaN loss poisons every
    parameter through the optimizer.
    """

    def _output(self, num_items):
        masks = [[0], [0]]
        return [
            _row(0, masks[0], [], []),
            _row(1, masks[1], [], []),
        ], {
            "advantages": torch.zeros(2),
            "completion_ids": _padded_completions(masks),
            "num_items_in_batch": torch.tensor(num_items),
        }

    def _run(self, num_items):
        inputs, output = self._output(num_items)
        trainer = _bare_trainer(MarshalConfig(enabled=True))
        with mock.patch.object(
            trl.GRPOTrainer, "_generate_and_score_completions", _stub_base_generate(output)
        ):
            return trainer._generate_and_score_completions(inputs)

    def test_an_all_placeholder_batch_cannot_divide_by_zero(self):
        out = self._run(0)
        self.assertGreaterEqual(
            out["num_items_in_batch"].item(), 1,
            "a zero normalizer makes the dapo/cispo loss NaN",
        )

    def test_the_loss_of_a_clamped_batch_is_zero_not_nan(self):
        """Clamping is only safe because the numerator is 0 too -- pin that."""
        out = self._run(0)
        per_token_loss = torch.zeros(2, 1)  # every row masked out => zero numerator
        loss = per_token_loss.sum() / out["num_items_in_batch"]
        self.assertFalse(torch.isnan(loss).any())
        self.assertEqual(loss.item(), 0.0)

    def test_a_healthy_batch_normalizer_is_untouched(self):
        out = self._run(37)
        self.assertEqual(out["num_items_in_batch"].item(), 37)


class TestRewardFuncIsUnshapedOnBothPaths(unittest.TestCase):
    """The reward must be the game outcome, identically on both arms.

    Any shaping applied on one arm only would make the ablation compare two reward
    functions rather than two advantage estimators.
    """

    OUTCOMES = [-1.0, -1.0, 0.0, 1.0, -0.9999999, -1.0000001]

    def test_rewards_pass_through_unchanged_when_marshal_is_enabled(self):
        func = build_reward_func(MarshalConfig(enabled=True))
        self.assertEqual(func(completions=["x"] * 6, rewards=self.OUTCOMES), self.OUTCOMES)

    def test_rewards_pass_through_unchanged_when_marshal_is_disabled(self):
        func = build_reward_func(MarshalConfig(enabled=False))
        self.assertEqual(func(completions=["x"] * 6, rewards=self.OUTCOMES), self.OUTCOMES)

    def test_both_paths_agree(self):
        on = build_reward_func(MarshalConfig(enabled=True))
        off = build_reward_func(MarshalConfig(enabled=False))
        self.assertEqual(
            on(completions=["x"] * 6, rewards=self.OUTCOMES),
            off(completions=["x"] * 6, rewards=self.OUTCOMES),
        )

    def test_an_all_abort_group_is_deterministic_and_has_no_variance(self):
        """The honest representation of a batch containing no information."""
        func = build_reward_func(MarshalConfig(enabled=False))
        first = func(completions=["x"] * 8, rewards=[-1.0] * 8)
        second = func(completions=["x"] * 8, rewards=[-1.0] * 8)
        self.assertEqual(first, second, "the reward function must not be stochastic")
        self.assertEqual(set(first), {-1.0})
        self.assertEqual(torch.tensor(first).std().item(), 0.0)

    def test_missing_rewards_kwarg_is_zero_not_a_crash(self):
        func = build_reward_func(MarshalConfig(enabled=False))
        self.assertEqual(func(completions=["a", "b"]), [0.0, 0.0])


class TestAdvantageFallbackIsLoud(unittest.TestCase):
    """Falling back to TRL's scalars silently would mislabel the whole run.

    ``_compute_marshal_advantages`` returns the base advantages unchanged when
    ``len(inputs) != batch_size``. Those steps are plain GRPO. It must be visible in
    both the log and the metrics, because a warning can scroll past a 700-step job
    and a metric cannot.
    """

    def setUp(self):
        MarshalGRPOTrainer._WARNED_ADV_FALLBACK = False
        self.masks = [[1, 1], [1, 1], [1, 1], [1, 1]]
        self.completion_ids = _padded_completions(self.masks)
        self.base_adv = torch.tensor([0.25, -0.25, 0.5, -0.5])

    def tearDown(self):
        MarshalGRPOTrainer._WARNED_ADV_FALLBACK = False

    def _inputs(self, n):
        return [_row(i % 2, self.masks[0], [1], [1.0]) for i in range(n)]

    def _output(self):
        return {"advantages": self.base_adv.clone(), "completion_ids": self.completion_ids}

    def test_aligned_batch_records_a_zero_fallback_rate(self):
        trainer = _bare_trainer()
        trainer._compute_marshal_advantages(self._inputs(4), self._output())
        self.assertEqual(
            trainer._metrics["train"]["marshal/advantage/fallback_rate"], [0.0],
            "the metric must be emitted on healthy steps too, or 'never fell back' "
            "and 'never logged' look the same",
        )

    def test_misaligned_batch_records_a_one_and_returns_the_base_tensor(self):
        trainer = _bare_trainer()
        out = trainer._compute_marshal_advantages(self._inputs(3), self._output())
        self.assertEqual(trainer._metrics["train"]["marshal/advantage/fallback_rate"], [1.0])
        self.assertTrue(torch.allclose(out, self.base_adv))
        self.assertEqual(tuple(out.shape), (4,))

    def test_every_step_appends_exactly_one_sample(self):
        trainer = _bare_trainer()
        trainer._compute_marshal_advantages(self._inputs(4), self._output())
        trainer._compute_marshal_advantages(self._inputs(3), self._output())
        trainer._compute_marshal_advantages(self._inputs(3), self._output())
        self.assertEqual(
            trainer._metrics["train"]["marshal/advantage/fallback_rate"], [0.0, 1.0, 1.0]
        )

    def test_it_warns_once_and_names_the_consequence(self):
        trainer = _bare_trainer()
        with self.assertLogs("playpen.marshal.trainer", level=logging.WARNING) as captured:
            trainer._compute_marshal_advantages(self._inputs(3), self._output())
            trainer._compute_marshal_advantages(self._inputs(3), self._output())
        self.assertEqual(len(captured.records), 1, "warn once per process, not per step")
        self.assertIn("plain GRPO", captured.output[0])

    def test_a_healthy_run_never_warns(self):
        trainer = _bare_trainer()
        logger = logging.getLogger("playpen.marshal.trainer")
        with mock.patch.object(logger, "warning") as warn:
            trainer._compute_marshal_advantages(self._inputs(4), self._output())
        warn.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)

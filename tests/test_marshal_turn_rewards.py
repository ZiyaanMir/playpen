"""Test the dense per-turn reward channel (``playpen/marshal/turn_rewards.py``).

Four things are being defended here, in order of how expensive they are to get
wrong:

1. **Off means off.** With ``turn_rewards: false`` (the default) the rollout dict,
   the turn rewards and the scalar reward must be exactly what they were before the
   feature existed -- no extra columns, no shifted values.
2. **The bound holds.** ``|episode shaping| <= budget`` per seat, always, so a
   shaped loss can never out-score a bare win. This is the property the whole
   scaling design exists to provide.
3. **Alignment.** A turn's shaping lands on *that* turn's boundary and on *that*
   seat -- the entire point is per-turn, per-seat credit, and a silent off-by-one
   would look like the feature working.
4. **The per-game extractors read the state the games actually write**, including
   the sticky-vs-cleared flag difference between them.

No torch, no trl, no clemcore: the extractors take a duck-typed state object and
the episode integration runs against the same scripted env style as
``test_marshal_rollout.py``.

Runnable via ``pytest`` or directly with ``.venv/bin/python -m unittest``.
"""

import unittest

from playpen.marshal.config import MarshalConfig
from playpen.marshal.turn_rewards import (
    CodenamesTurnRewards,
    DondTurnRewards,
    FormatComplianceExtractor,
    GuessWhatTurnRewards,
    TabooTurnRewards,
    TurnContext,
    TurnRewardSpec,
    TurnRewardTracker,
    WordleTurnRewards,
    build_extractor,
    resolve_extractor_class,
    resolve_turn_reward_extractor,
    wordle_closeness,
)


class _State:
    """A stand-in for a game's ``GameState`` subclass: attributes, nothing else."""

    def __init__(self, **fields):
        self.__dict__.update(fields)

    def set(self, **fields):
        self.__dict__.update(fields)
        return self


def _ctx(state, seat=0, turn_index=0, response="", done=False):
    return TurnContext(
        seat=seat, turn_index=turn_index, state=state, info={}, response=response, done=done
    )


# ---------------------------------------------------------------------------
# wordle
# ---------------------------------------------------------------------------


class TestWordleCloseness(unittest.TestCase):
    """The closeness score must be clembench's own, only rescaled to [0, 1].

    clembench's ``turns_closeness`` (wordle/utils/compute_metrics.py) scores a green
    5 and a yellow 3 and reports a raw 0..25. We divide by 5 * n_letters so the value
    fits the extractor contract; nothing else about it may differ, or the training
    signal stops matching the metric the game is scored on.
    """

    def test_all_green_is_one(self):
        feedback = "s<green> t<green> r<green> a<green> p<green>"
        self.assertEqual(wordle_closeness(feedback), 1.0)

    def test_all_red_is_zero(self):
        self.assertEqual(wordle_closeness("h<red> e<red> l<red> l<red> o<red>"), 0.0)

    def test_matches_clembench_weights(self):
        # 1 green + 1 yellow = 5 + 3 = 8 raw, over a max of 25.
        feedback = "s<green> n<yellow> e<red> a<red> k<red>"
        self.assertAlmostEqual(wordle_closeness(feedback), 8 / 25)

    def test_empty_or_unparseable_feedback_is_zero(self):
        for value in ("", None, "no colour markers here"):
            self.assertEqual(wordle_closeness(value), 0.0)


class TestWordleExtractor(unittest.TestCase):
    def setUp(self):
        self.ex = WordleTurnRewards()
        self.ex.reset()

    def test_first_guess_scores_its_closeness(self):
        state = _State(valid_response=True, guess_feedback="s<green> n<yellow> e<red> a<red> k<red>")
        self.assertAlmostEqual(self.ex.extract(_ctx(state))["closeness"], 8 / 25)

    def test_only_the_gain_over_the_best_so_far_is_paid(self):
        state = _State(valid_response=True, guess_feedback="s<green> n<red> e<red> a<red> k<red>")
        first = self.ex.extract(_ctx(state))["closeness"]
        state.set(guess_feedback="s<green> t<green> e<red> a<red> k<red>")
        second = self.ex.extract(_ctx(state, turn_index=1))["closeness"]
        self.assertAlmostEqual(first, 5 / 25)
        self.assertAlmostEqual(second, 5 / 25)  # the *increment*, not the new total
        # Telescoping: the episode total is the best closeness reached, bounded by 1.
        self.assertAlmostEqual(first + second, 10 / 25)

    def test_repeating_a_guess_earns_nothing(self):
        state = _State(valid_response=True, guess_feedback="s<green> n<yellow> e<red> a<red> k<red>")
        self.ex.extract(_ctx(state))
        self.assertNotIn("closeness", self.ex.extract(_ctx(state, turn_index=1)))

    def test_a_worse_guess_is_not_punished(self):
        state = _State(valid_response=True, guess_feedback="s<green> t<green> r<green> a<red> p<red>")
        self.ex.extract(_ctx(state))
        state.set(guess_feedback="h<red> e<red> l<red> l<red> o<red>")
        self.assertEqual(self.ex.extract(_ctx(state, turn_index=1)), {})

    def test_an_invalid_guess_is_charged_and_scores_no_closeness(self):
        # valid_response False means guess_feedback is stale from an earlier turn --
        # paying closeness off it would reward a guess the game rejected.
        state = _State(valid_response=False, guess_feedback="s<green> t<green> r<green> a<green> p<red>")
        self.assertEqual(self.ex.extract(_ctx(state)), {"format": -1.0})

    def test_episode_total_is_bounded_by_one_before_scaling(self):
        self.ex.reset()
        total = 0.0
        for greens in range(6):
            feedback = " ".join(["x<green>"] * greens + ["x<red>"] * (5 - greens))
            state = _State(valid_response=True, guess_feedback=feedback)
            total += self.ex.extract(_ctx(state, turn_index=greens)).get("closeness", 0.0)
        self.assertAlmostEqual(total, 1.0)


# ---------------------------------------------------------------------------
# taboo / guesswhat / codenames / dond
# ---------------------------------------------------------------------------


class TestTabooExtractor(unittest.TestCase):
    """taboo leaves its flags set once tripped, so only the rising edge may charge."""

    def setUp(self):
        self.ex = TabooTurnRewards()
        self.ex.reset()

    def test_clean_turn_scores_nothing(self):
        state = _State(invalid_response=False, clue_error=None)
        self.assertEqual(self.ex.extract(_ctx(state)), {})

    def test_format_violation_is_charged_once(self):
        state = _State(invalid_response=False, clue_error=None)
        self.ex.extract(_ctx(state))
        state.set(invalid_response=True)
        self.assertEqual(self.ex.extract(_ctx(state, turn_index=1)), {"format": -1.0})
        # The flag stays True for the rest of the episode; later turns must not pay again.
        self.assertEqual(self.ex.extract(_ctx(state, turn_index=2)), {})

    def test_illegal_clue_is_charged_on_the_turn_that_gave_it(self):
        state = _State(invalid_response=False, clue_error=None)
        self.ex.extract(_ctx(state))
        state.set(clue_error={"message": "used a related word"})
        self.assertEqual(self.ex.extract(_ctx(state, seat=0, turn_index=1)), {"clue": -1.0})
        self.assertEqual(self.ex.extract(_ctx(state, turn_index=2)), {})


class TestGuessWhatExtractor(unittest.TestCase):
    def setUp(self):
        self.ex = GuessWhatTurnRewards()
        self.ex.reset()

    def test_both_invalid_flags_charge_format(self):
        for flag in ("invalid_format", "invalid_content"):
            ex = GuessWhatTurnRewards()
            ex.reset()
            state = _State(invalid_format=False, invalid_content=False)
            ex.extract(_ctx(state))
            state.set(**{flag: True})
            self.assertEqual(ex.extract(_ctx(state, turn_index=1)), {"format": -1.0}, flag)

    def test_a_valid_question_scores_nothing(self):
        state = _State(invalid_format=False, invalid_content=False)
        self.assertEqual(self.ex.extract(_ctx(state)), {})


class _Board:
    """clembench's ``CodenamesBoard`` shape: ``hidden`` plus ``revealed[by][assignment]``.

    Modelled faithfully because the difference between the two is what the extractor
    depends on -- ``reveal_word(word, by)`` removes from ``hidden`` and appends to
    ``revealed[by]``, and codenames calls it with ``by="opponent"`` for the simulated
    opposing team as well as ``by="team"`` for our guesser.
    """

    def __init__(self, team, opponent, innocent, assassin):
        self.hidden = {
            "team": list(team),
            "opponent": list(opponent),
            "innocent": list(innocent),
            "assassin": list(assassin),
        }
        self.revealed = {
            by: {"team": [], "opponent": [], "innocent": [], "assassin": []}
            for by in ("team", "opponent")
        }

    def reveal(self, assignment, count=1, by="team"):
        for _ in range(count):
            word = self.hidden[assignment].pop()
            self.revealed[by][assignment].append(word)


class TestCodenamesExtractor(unittest.TestCase):
    """Board progress is the one genuinely dense signal among the 2-player games."""

    def setUp(self):
        self.ex = CodenamesTurnRewards()
        self.ex.reset()
        self.board = _Board(["a", "b", "c", "d"], ["e", "f"], ["g", "h"], ["i"])
        self.state = _State(board=self.board, invalid_response=False)

    def _turn(self, seat=1, turn_index=0):
        return self.ex.extract(_ctx(self.state, seat=seat, turn_index=turn_index))

    def test_first_turn_only_establishes_the_baseline(self):
        self.assertEqual(self._turn(), {})

    def test_revealing_a_team_word_is_rewarded_in_units_of_the_board(self):
        self._turn()
        self.board.reveal("team")
        # 1 of 4 team words: +1/4.
        self.assertAlmostEqual(self._turn(turn_index=1)["board"], 0.25)

    def test_revealing_the_assassin_is_the_worst_outcome(self):
        self._turn()
        self.board.reveal("assassin")
        self.assertAlmostEqual(self._turn(turn_index=1)["board"], -0.5)

    def test_a_mixed_turn_nets_out(self):
        self._turn()
        self.board.reveal("team", 2)
        self.board.reveal("innocent")
        # (2 * 1.0 - 0.5) / 4
        self.assertAlmostEqual(self._turn(turn_index=1)["board"], 0.375)

    def test_clearing_the_whole_team_board_totals_exactly_one(self):
        self._turn()
        total = 0.0
        for step in range(4):
            self.board.reveal("team")
            total += self._turn(turn_index=step + 1).get("board", 0.0)
        self.assertAlmostEqual(total, 1.0)

    def test_component_is_clipped_into_range(self):
        self._turn()
        self.board.reveal("assassin")
        self.board.reveal("opponent", 2)
        # Raw value is (-2 - 2) / 4 = -1.0 exactly; anything more negative must clip.
        self.assertGreaterEqual(self._turn(turn_index=1)["board"], -1.0)

    def test_invalid_response_is_read_as_a_level(self):
        # codenames clears the flag at the top of every validation, so its value
        # after a step describes that step -- no edge detection wanted here.
        self.state.set(invalid_response=True)
        self.assertEqual(self._turn().get("format"), -1.0)
        self.state.set(invalid_response=False)
        self.assertNotIn("format", self._turn(turn_index=1))

    def test_the_simulated_opponents_turn_is_not_charged_to_us(self):
        # codenames reveals opponent cards between rounds (_on_before_round ->
        # _opponent_turn), which shrinks `hidden` without our policy acting. Reading
        # the drop in `hidden` made a +1 team word and the opponent's card cancel to
        # exactly 0.0 on every real turn -- the component looked dead.
        self._turn()
        self.board.reveal("opponent", by="opponent")
        self.assertEqual(self._turn(turn_index=1), {})

        self.board.reveal("team")                      # ours
        self.board.reveal("opponent", by="opponent")   # theirs, same round
        self.assertAlmostEqual(self._turn(turn_index=2)["board"], 0.25)

    def test_our_own_wrong_guess_is_still_charged(self):
        self._turn()
        self.board.reveal("opponent")  # by="team": our guesser hit their word
        self.assertAlmostEqual(self._turn(turn_index=1)["board"], -0.25)

    def test_a_board_that_does_not_look_like_one_is_ignored(self):
        ex = CodenamesTurnRewards()
        ex.reset()
        self.assertEqual(ex.extract(_ctx(_State(board=None, invalid_response=False))), {})
        # A board object without the revealed bookkeeping must not produce a component.
        self.assertEqual(
            ex.extract(_ctx(_State(board=_State(hidden={}), invalid_response=False))), {}
        )


class TestDondExtractor(unittest.TestCase):
    def setUp(self):
        self.ex = DondTurnRewards()
        self.ex.reset()

    def test_each_seat_is_paid_for_its_own_proposal_once(self):
        state = _State(player_a_proposal=None, player_b_proposal=None, outcome=None)
        self.assertEqual(self.ex.extract(_ctx(state, seat=0)), {})
        state.set(player_a_proposal=[1, 2, 3])
        self.assertEqual(self.ex.extract(_ctx(state, seat=0, turn_index=1)), {"proposal": 1.0})
        # Seat 1 has not proposed yet, and must not collect seat 0's reward.
        self.assertEqual(self.ex.extract(_ctx(state, seat=1)), {})
        state.set(player_b_proposal=[0, 0, 0])
        self.assertEqual(self.ex.extract(_ctx(state, seat=1, turn_index=1)), {"proposal": 1.0})
        # Neither seat is paid twice.
        self.assertEqual(self.ex.extract(_ctx(state, seat=0, turn_index=2)), {})

    def test_the_turn_that_aborts_is_charged(self):
        class _Outcome:
            name = "ABORTED"
            value = "aborted"

        state = _State(player_a_proposal=None, player_b_proposal=None, outcome=None)
        self.ex.extract(_ctx(state, seat=0))
        state.set(outcome=_Outcome())
        self.assertEqual(self.ex.extract(_ctx(state, seat=1, turn_index=1))["format"], -1.0)
        self.assertNotIn("format", self.ex.extract(_ctx(state, seat=0, turn_index=1)))


class TestGenericExtractor(unittest.TestCase):
    """The fallback must work off whichever validation flags a game happens to carry."""

    def test_rising_edge_on_a_sticky_negative_flag(self):
        ex = FormatComplianceExtractor()
        ex.reset()
        state = _State(invalid_response=False)
        self.assertEqual(ex.extract(_ctx(state)), {})
        state.set(invalid_response=True)
        self.assertEqual(ex.extract(_ctx(state, turn_index=1)), {"format": -1.0})
        self.assertEqual(ex.extract(_ctx(state, turn_index=2)), {})

    def test_level_on_the_positive_flag(self):
        ex = FormatComplianceExtractor()
        ex.reset()
        state = _State(valid_response=False)
        self.assertEqual(ex.extract(_ctx(state)), {"format": -1.0})
        state.set(valid_response=True)
        self.assertEqual(ex.extract(_ctx(state, turn_index=1)), {})

    def test_a_state_with_no_known_flags_yields_nothing(self):
        ex = FormatComplianceExtractor()
        ex.reset()
        self.assertEqual(ex.extract(_ctx(_State(some_other_field=3))), {})
        self.assertFalse(ex.supported_by(_State(some_other_field=3)))
        self.assertTrue(ex.supported_by(_State(invalid_format=False)))

    def test_a_missing_state_does_not_raise(self):
        ex = FormatComplianceExtractor()
        ex.reset()
        self.assertEqual(ex.extract(_ctx(None)), {})


# ---------------------------------------------------------------------------
# registry / config resolution
# ---------------------------------------------------------------------------


class TestRegistry(unittest.TestCase):
    def test_exact_and_prefix_resolution(self):
        self.assertIs(resolve_extractor_class("wordle"), WordleTurnRewards)
        self.assertIs(resolve_extractor_class("wordle_withclue"), WordleTurnRewards)
        self.assertIs(resolve_extractor_class("wordle_withcritic"), WordleTurnRewards)
        self.assertIs(resolve_extractor_class("taboo"), TabooTurnRewards)
        self.assertIsNone(resolve_extractor_class("referencegame"))

    def test_auto_falls_back_to_generic(self):
        self.assertIsInstance(build_extractor("referencegame", "auto"), FormatComplianceExtractor)

    def test_game_source_refuses_the_fallback(self):
        # An unregistered game must not silently train on a different signal than
        # the one the run asked for.
        self.assertIsNone(build_extractor("referencegame", "game"))
        self.assertIsInstance(build_extractor("codenames", "game"), CodenamesTurnRewards)

    def test_generic_source_overrides_a_registered_game(self):
        self.assertIsInstance(build_extractor("wordle", "generic"), FormatComplianceExtractor)

    def test_unknown_source_is_rejected(self):
        with self.assertRaises(ValueError):
            build_extractor("wordle", "nonsense")


class TestConfigResolution(unittest.TestCase):
    def test_off_by_default(self):
        cfg = MarshalConfig()
        self.assertFalse(cfg.turn_rewards)
        self.assertIsNone(cfg.turn_reward_kwargs())
        self.assertEqual(resolve_turn_reward_extractor("wordle", cfg), (None, None))

    def test_on_resolves_extractor_and_spec(self):
        cfg = MarshalConfig(turn_rewards=True, turn_reward_scale=0.2, turn_reward_budget=0.4)
        extractor, spec = resolve_turn_reward_extractor("wordle", cfg)
        self.assertIsInstance(extractor, WordleTurnRewards)
        self.assertEqual((spec.scale, spec.budget), (0.2, 0.4))

    def test_component_allowlist_is_parsed_and_validated(self):
        cfg = MarshalConfig(turn_rewards=True, turn_reward_components="closeness")
        _, spec = resolve_turn_reward_extractor("wordle", cfg)
        self.assertEqual(spec.components, ("closeness",))

        bad = MarshalConfig(turn_rewards=True, turn_reward_components="clossness")
        with self.assertRaises(ValueError) as caught:
            resolve_turn_reward_extractor("wordle", bad)
        # The message must name what IS available, or the typo costs another run.
        self.assertIn("closeness", str(caught.exception))

    def test_negative_scale_or_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            MarshalConfig(turn_reward_scale=-0.1)
        with self.assertRaises(ValueError):
            MarshalConfig(turn_reward_budget=-1.0)

    def test_unknown_source_is_rejected_at_config_time(self):
        with self.assertRaises(ValueError):
            MarshalConfig(turn_reward_source="sometimes")

    def test_budget_safety_property(self):
        self.assertTrue(MarshalConfig(turn_rewards=True, turn_reward_budget=0.3)
                        .turn_reward_budget_is_safe)
        self.assertFalse(MarshalConfig(turn_rewards=True, turn_reward_budget=0.5)
                         .turn_reward_budget_is_safe)
        # An uncapped budget is not safe -- there is then no bound at all.
        self.assertFalse(MarshalConfig(turn_rewards=True, turn_reward_budget=0.0)
                         .turn_reward_budget_is_safe)
        # With the feature off the question is vacuous, so it must not warn.
        self.assertTrue(MarshalConfig(turn_reward_budget=9.0).turn_reward_budget_is_safe)

    def test_round_trips_through_yaml_shaped_dict(self):
        cfg = MarshalConfig.from_dict(
            {"turn_rewards": True, "turn_reward_scale": 0.1, "turn_reward_source": "generic"}
        )
        self.assertTrue(cfg.turn_rewards)
        self.assertEqual(cfg.turn_reward_source, "generic")
        self.assertIn("turn_reward_budget", cfg.to_dict())


# ---------------------------------------------------------------------------
# tracker: scaling, the budget bound, alignment
# ---------------------------------------------------------------------------


class _ConstantExtractor:
    """Emits a fixed value every turn, so the scaling can be checked in isolation."""

    components = ("fixed",)

    def __init__(self, value=1.0):
        self.value = value

    def reset(self):
        pass

    def extract(self, ctx):
        return {"fixed": self.value}


class _OverflowingExtractor:
    """Emits components that breach the ``[-1, 1]`` contract extractors are meant to keep.

    Every shipped extractor normalizes to ``[-1, 1]``, but that is a convention, not
    something the type system enforces: a new extractor, a clembench field that
    changes units, or a game state that reports a count instead of a fraction all
    produce out-of-range values. The tracker's per-turn clip is what stops that from
    reaching the return.
    """

    components = ("a", "b", "c")

    def __init__(self, value):
        self.value = value

    def reset(self):
        pass

    def extract(self, ctx):
        return {name: self.value for name in self.components}


class TestPerTurnClipBoundsASingleTurn(unittest.TestCase):
    """A single turn can never contribute more than ``scale``, whatever an extractor returns.

    Two independent clips protect the reward: each *component* is clipped to
    ``[-1, 1]``, then their *sum* is clipped to ``[-1, 1]`` again, and only then is
    ``scale`` applied. Without the second clip a three-component extractor would
    make a turn worth ``3 * scale``, and with ``turn_reward_budget: 0`` -- which
    disables the episode cap -- nothing downstream would bound it at all.

    Tested with the budget off on purpose: with it on, the episode rescale would
    mask a broken per-turn clip and the test would pass for the wrong reason.
    """

    def _finalize(self, value, turns=1, scale=0.05, budget=0.0):
        tracker = TurnRewardTracker(
            _OverflowingExtractor(value), TurnRewardSpec(scale=scale, budget=budget)
        )
        for index in range(turns):
            tracker.on_turn(_ctx(None, seat=0, turn_index=index))
        return tracker.finalize()[0]

    def test_three_components_summing_above_one_are_clipped_to_scale(self):
        # Each component is a legal 1.0; their sum is 3.0 and must clip to 1.0.
        self.assertAlmostEqual(self._finalize(1.0)[0], 0.05)

    def test_an_extractor_that_ignores_the_contract_cannot_escape_scale(self):
        for value in (5.0, 1e6, float("inf")):
            self.assertAlmostEqual(
                self._finalize(value)[0], 0.05, msg=f"component value {value}"
            )

    def test_the_negative_side_is_bounded_too(self):
        for value in (-1.0, -5.0, -1e6):
            self.assertAlmostEqual(
                self._finalize(value)[0], -0.05, msg=f"component value {value}"
            )

    def test_on_turn_returns_the_clipped_value_not_the_raw_sum(self):
        tracker = TurnRewardTracker(
            _OverflowingExtractor(4.0), TurnRewardSpec(scale=0.05, budget=0.0)
        )
        self.assertAlmostEqual(tracker.on_turn(_ctx(None, seat=0, turn_index=0)), 1.0)

    def test_with_the_budget_off_the_episode_bound_is_scale_times_turns(self):
        """The only bound left when budget=0, so the per-turn clip has to hold."""
        values = self._finalize(1e6, turns=8, scale=0.05, budget=0.0)
        self.assertAlmostEqual(sum(values), 0.4)


class TestTrackerScaling(unittest.TestCase):
    def _track(self, turns, value=1.0, scale=0.05, budget=0.3, seat=0):
        tracker = TurnRewardTracker(
            _ConstantExtractor(value), TurnRewardSpec(scale=scale, budget=budget)
        )
        for index in range(turns):
            tracker.on_turn(_ctx(None, seat=seat, turn_index=index))
        return tracker

    def test_scale_caps_a_single_turn(self):
        values = self._track(1, scale=0.05).finalize()[0]
        self.assertAlmostEqual(values[0], 0.05)

    def test_under_budget_the_sum_is_scale_times_turns(self):
        values = self._track(4, scale=0.05, budget=0.3).finalize()[0]
        self.assertAlmostEqual(sum(values), 0.2)

    def test_over_budget_the_episode_is_rescaled_proportionally(self):
        values = self._track(10, scale=0.05, budget=0.3).finalize()[0]
        self.assertAlmostEqual(sum(values), 0.3)
        # Proportional, not truncated: every turn keeps an equal share.
        self.assertTrue(all(abs(value - values[0]) < 1e-9 for value in values))

    def test_the_bound_holds_for_any_turn_count_and_sign(self):
        for turns in (1, 3, 7, 25, 100):
            for value in (1.0, -1.0):
                total = sum(self._track(turns, value=value, scale=0.5, budget=0.3).finalize()[0])
                self.assertLessEqual(abs(total) - 0.3, 1e-9, f"{turns} turns, value {value}")

    def test_budget_zero_disables_the_cap(self):
        values = self._track(10, scale=0.05, budget=0.0).finalize()[0]
        self.assertAlmostEqual(sum(values), 0.5)

    def test_clip_flag_reports_whether_the_budget_bound(self):
        under = self._track(2, scale=0.05, budget=0.3)
        under.finalize()
        self.assertFalse(under.was_clipped(0))
        over = self._track(20, scale=0.05, budget=0.3)
        over.finalize()
        self.assertTrue(over.was_clipped(0))

    def test_a_turn_signal_is_clipped_into_range_before_scaling(self):
        # An extractor returning nonsense must not be able to escape the per-turn cap.
        tracker = TurnRewardTracker(_ConstantExtractor(50.0), TurnRewardSpec(scale=0.05, budget=0))
        tracker.on_turn(_ctx(None, turn_index=0))
        self.assertAlmostEqual(tracker.finalize()[0][0], 0.05)

    def test_a_raising_extractor_scores_zero_instead_of_killing_the_run(self):
        class _Broken:
            components = ("boom",)

            def reset(self):
                pass

            def extract(self, ctx):
                raise RuntimeError("clembench renamed a field")

        tracker = TurnRewardTracker(_Broken(), TurnRewardSpec())
        with self.assertLogs("playpen.marshal.turn_rewards", level="WARNING"):
            tracker.on_turn(_ctx(None, turn_index=0))
        self.assertEqual(tracker.finalize()[0], [0.0])

    def test_seats_are_tracked_and_capped_independently(self):
        tracker = TurnRewardTracker(_ConstantExtractor(1.0), TurnRewardSpec(scale=0.1, budget=0.3))
        for index in range(10):
            tracker.on_turn(_ctx(None, seat=0, turn_index=index))
        tracker.on_turn(_ctx(None, seat=1, turn_index=0))
        out = tracker.finalize()
        self.assertAlmostEqual(sum(out[0]), 0.3)   # capped
        self.assertAlmostEqual(sum(out[1]), 0.1)   # untouched
        self.assertTrue(tracker.was_clipped(0))
        self.assertFalse(tracker.was_clipped(1))

    def test_component_allowlist_drops_everything_else(self):
        class _Two:
            components = ("keep", "drop")

            def reset(self):
                pass

            def extract(self, ctx):
                return {"keep": 1.0, "drop": -1.0}

        both = TurnRewardTracker(_Two(), TurnRewardSpec(scale=1.0, budget=0))
        both.on_turn(_ctx(None, turn_index=0))
        self.assertAlmostEqual(both.finalize()[0][0], 0.0)  # 1.0 + (-1.0)

        kept = TurnRewardTracker(_Two(), TurnRewardSpec(scale=1.0, budget=0, components=("keep",)))
        kept.on_turn(_ctx(None, turn_index=0))
        self.assertAlmostEqual(kept.finalize()[0][0], 1.0)
        self.assertEqual(kept.component_totals(0), {"keep": 1.0})

    def test_values_stay_on_their_own_turn_index(self):
        class _ByTurn:
            components = ("t",)

            def reset(self):
                pass

            def extract(self, ctx):
                return {"t": 1.0 if ctx.turn_index == 2 else 0.0}

        tracker = TurnRewardTracker(_ByTurn(), TurnRewardSpec(scale=1.0, budget=0))
        for index in range(4):
            tracker.on_turn(_ctx(None, turn_index=index))
        self.assertEqual(tracker.finalize()[0], [0.0, 0.0, 1.0, 0.0])


class TestTerminalOrderingIsPreserved(unittest.TestCase):
    """The property the budget exists for, stated as a test.

    clembench outcomes are +1 / 0 / -1, so the smallest gap between two distinct
    outcomes is 1.0. With ``budget < 0.5`` the largest shaping difference between two
    episodes is ``2 * budget < 1.0``, so no amount of shaping can make a worse
    outcome score higher than a better one.
    """

    def _episode_total(self, outcome, shaping_sign, turns=12, scale=0.05, budget=0.3):
        tracker = TurnRewardTracker(
            _ConstantExtractor(shaping_sign), TurnRewardSpec(scale=scale, budget=budget)
        )
        for index in range(turns):
            tracker.on_turn(_ctx(None, turn_index=index))
        return outcome + sum(tracker.finalize()[0])

    def test_a_maximally_shaped_loss_never_beats_an_unshaped_win(self):
        best_loss = self._episode_total(0.0, +1.0)
        worst_win = self._episode_total(1.0, -1.0)
        self.assertLess(best_loss, worst_win)

    def test_a_maximally_shaped_abort_never_beats_an_unshaped_loss(self):
        best_abort = self._episode_total(-1.0, +1.0)
        worst_loss = self._episode_total(0.0, -1.0)
        self.assertLess(best_abort, worst_loss)

    def test_the_guarantee_fails_once_the_budget_is_raised_past_the_safe_point(self):
        # Not an endorsement -- this is why MarshalConfig warns above 0.5.
        best_loss = self._episode_total(0.0, +1.0, budget=0.8)
        worst_win = self._episode_total(1.0, -1.0, budget=0.8)
        self.assertGreater(best_loss, worst_win)


# ---------------------------------------------------------------------------
# integration through play_selfplay_episode
# ---------------------------------------------------------------------------


class _FakePlayer:
    def __init__(self):
        self._messages = []

    def perceive_context(self, context, *, log_event=True, memorize=True):
        perspective = list(self._messages) + [context]
        if memorize:
            self._messages.append(dict(context))
        return perspective

    def perceive_response(self, response, *, log_event=True, memorize=True, metadata=None):
        if memorize:
            self._messages.append({"role": "assistant", "content": response})
        return list(self._messages)


class _StatefulEnv:
    """A 2-seat scripted env that also exposes a live game state, like SelfPlayEnv.

    ``violations`` names the (seat, turn) pairs whose response should trip the
    game's ``invalid_response`` flag, so a test can place a per-turn penalty at a
    known boundary and check it landed there.
    """

    def __init__(self, total_turns=4, terminal_reward=1.0, violations=()):
        self.num_players = 2
        self.game_name = "taboo"
        self.resolved_game_name = "taboo"
        self._total = total_turns
        self._terminal = terminal_reward
        self._violations = set(violations)
        self._turn = 0
        self._players = {0: _FakePlayer(), 1: _FakePlayer()}
        self.state = _State(invalid_response=False, clue_error=None)
        self.finalized = False

    def reset(self, instance_idx, *, seed=0):
        self._turn = 0
        self.finalized = False
        self.state = _State(invalid_response=False, clue_error=None)

    @property
    def current_seat(self):
        return self._turn % 2

    @property
    def current_player(self):
        return self._players[self.current_seat]

    @property
    def game_state(self):
        return self.state

    def info_for_seat(self, seat):
        return {}

    def observe(self):
        return {"role": "user", "content": f"observation number {self._turn}"}

    def step(self, response):
        if (self.current_seat, self._turn // 2) in self._violations:
            self.state.set(invalid_response=True)
        self._turn += 1

    @property
    def done(self):
        return self._turn >= self._total

    @property
    def cumulative_rewards(self):
        return [self._terminal, self._terminal] if self.done else [0.0, 0.0]

    def finalize(self):
        self.finalized = True


class _FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        return " ".join(m["content"] for m in messages)

    def encode(self, text, add_special_tokens=False):
        return [1] * len(text.split())

    def decode(self, ids, skip_special_tokens=True):
        return "decoded"


class _FakeTrainer:
    def __init__(self):
        self.processing_class = _FakeTokenizer()


def _install_fake_generation(test_case):
    """Point ``generate_rollout_completions`` at a stub for the duration of a test.

    Patches the attribute on the real module (as ``test_marshal_rollout.py`` does)
    rather than faking the package: ``playpen.marshal.trainer`` subclasses
    ``trl.GRPOTrainer`` at import time, so a stubbed ``trl`` would break the very
    import this test needs.
    """
    import trl.experimental.openenv as oe

    original = oe.generate_rollout_completions
    oe.generate_rollout_completions = lambda trainer, prompts, **kwargs: [
        {
            "prompt_ids": [1, 2, 3],
            "completion_ids": [10, 11],
            "logprobs": [-0.1, -0.2],
            "text": "CLUE: something",
        }
        for _ in prompts
    ]
    test_case.addCleanup(setattr, oe, "generate_rollout_completions", original)


class TestEpisodeIntegration(unittest.TestCase):
    """Shaping must reach the right turn of the right seat, on top of the outcome."""

    def setUp(self):
        _install_fake_generation(self)
        from playpen.marshal.selfplay_agent import play_selfplay_episode

        self.play = play_selfplay_episode

    def _run(self, env, tracker=None):
        return self.play(
            env,
            _FakeTrainer(),
            _FakeTokenizer(),
            0,
            row_context_mode="spliced",
            turn_reward_tracker=tracker,
        )

    def _tracker(self, scale=0.1, budget=0.0):
        return TurnRewardTracker(TabooTurnRewards(), TurnRewardSpec(scale=scale, budget=budget))

    def test_without_a_tracker_nothing_changes(self):
        rollouts = self._run(_StatefulEnv(total_turns=4, terminal_reward=1.0))
        for seat in (0, 1):
            row = rollouts[seat]
            self.assertEqual(row.turn_rewards, [0.0, 1.0])
            self.assertEqual(row.shaping_reward, 0.0)
            self.assertFalse(row.shaping_clipped)
            self.assertEqual(row.shaping_components, {})

    def test_terminal_reward_is_untouched_by_shaping(self):
        env = _StatefulEnv(total_turns=4, terminal_reward=1.0, violations={(1, 0)})
        rollouts = self._run(env, self._tracker())
        # terminal_reward stays the pure game outcome, so a turn-rewards arm is
        # still comparable with one that has the feature off.
        self.assertEqual(rollouts[0].terminal_reward, 1.0)
        self.assertEqual(rollouts[1].terminal_reward, 1.0)

    def test_the_penalty_lands_on_the_offending_seat_and_turn(self):
        # Seat 1 trips the format flag on its first turn (env turn 1).
        env = _StatefulEnv(total_turns=4, terminal_reward=1.0, violations={(1, 0)})
        rollouts = self._run(env, self._tracker(scale=0.1, budget=0.0))

        # Seat 1: -0.1 on turn 0, plus the shared terminal +1 on its last turn.
        self.assertAlmostEqual(rollouts[1].turn_rewards[0], -0.1)
        self.assertAlmostEqual(rollouts[1].turn_rewards[1], 1.0)
        self.assertAlmostEqual(rollouts[1].shaping_reward, -0.1)

        # Seat 0 did nothing wrong and must not be charged for its partner's turn --
        # which is exactly what the shared terminal reward cannot express.
        self.assertEqual(rollouts[0].turn_rewards, [0.0, 1.0])
        self.assertAlmostEqual(rollouts[0].shaping_reward, 0.0)

    def test_component_totals_and_clip_flag_reach_the_row(self):
        env = _StatefulEnv(total_turns=4, terminal_reward=-1.0, violations={(0, 0)})
        rollouts = self._run(env, self._tracker(scale=0.5, budget=0.3))
        self.assertEqual(rollouts[0].shaping_components, {"format": -1.0})
        self.assertAlmostEqual(rollouts[0].shaping_reward, -0.3)  # budget bound it
        self.assertTrue(rollouts[0].shaping_clipped)
        self.assertFalse(rollouts[1].shaping_clipped)

    def test_shaping_is_added_to_the_outcome_not_substituted_for_it(self):
        env = _StatefulEnv(total_turns=4, terminal_reward=1.0, violations={(0, 0)})
        rollouts = self._run(env, self._tracker(scale=0.1, budget=0.0))
        row = rollouts[0]
        self.assertAlmostEqual(sum(row.turn_rewards), 1.0 - 0.1)

    def test_an_env_without_a_game_state_produces_no_shaping(self):
        # The scripted envs in test_marshal_rollout.py carry no game master at all.
        env = _StatefulEnv(total_turns=4, terminal_reward=1.0)
        del env.__class__.game_state  # simulate an env that never exposed one
        try:
            rollouts = self._run(env, self._tracker())
            self.assertAlmostEqual(rollouts[0].shaping_reward, 0.0)
        finally:
            _StatefulEnv.game_state = property(lambda self: self.state)


class TestRolloutFuncWiring(unittest.TestCase):
    """What the rollout emits, on and off.

    ``build_selfplay_rollout_func`` is the seam where the feature reaches TRL, so
    this is where "off is byte-identical" is actually checkable: the extra columns
    must not merely be zero when the feature is off, they must not exist.
    """

    def setUp(self):
        _install_fake_generation(self)

    @staticmethod
    def _config(**fields):
        # row_context_mode="spliced" because _FakeTokenizer cannot round-trip an
        # exact-mode row (every generation reports the same prompt_ids), which would
        # drop every row as drifted and make these assertions vacuous. Exact-mode row
        # assembly has its own coverage in test_marshal_rollout.py.
        return MarshalConfig(row_context_mode="spliced", **fields)

    def _run(self, config, env=None):
        from playpen.marshal.trainer import build_selfplay_rollout_func

        env = env or _StatefulEnv(total_turns=4, terminal_reward=1.0, violations={(0, 0)})
        fn = build_selfplay_rollout_func(env, config)
        return fn(["0", "0"], _FakeTrainer())

    def test_off_adds_no_columns(self):
        out = self._run(self._config())
        for key in ("turn_reward_sum", "turn_reward_clipped", "terminal_reward"):
            self.assertNotIn(key, out)
        self.assertFalse([k for k in out if k.startswith("turn_reward_component_")])
        self.assertEqual(out["rewards"], [1.0, 1.0])

    def test_on_adds_the_columns_and_folds_shaping_into_the_scalar(self):
        config = self._config(turn_rewards=True, turn_reward_scale=0.1, turn_reward_budget=0.0)
        out = self._run(config)
        self.assertEqual(out["terminal_reward"], [1.0, 1.0])
        # Seat 0 tripped the format flag; seat 1 did not.
        self.assertAlmostEqual(out["turn_reward_sum"][0], -0.1)
        self.assertAlmostEqual(out["turn_reward_sum"][1], 0.0)
        # The scalar TRL consumes is outcome + shaping, so the signal is not inert
        # on the plain-GRPO path (which never sees the per-turn vector).
        self.assertAlmostEqual(out["rewards"][0], 0.9)
        self.assertAlmostEqual(out["rewards"][1], 1.0)
        self.assertEqual(out["turn_reward_clipped"], [False, False])

    def test_component_columns_are_emitted_for_every_row(self):
        config = self._config(turn_rewards=True)
        out = self._run(config)
        # taboo's extractor declares both, so both columns exist and are rectangular
        # even though only one of them fired.
        for key in ("turn_reward_component_format", "turn_reward_component_clue"):
            self.assertIn(key, out)
            self.assertEqual(len(out[key]), 2)
        self.assertEqual(out["turn_reward_component_format"], [-1.0, 0.0])
        self.assertEqual(out["turn_reward_component_clue"], [0.0, 0.0])

    def test_each_episode_gets_a_fresh_extractor(self):
        # Extractors carry episode state (wordle's best closeness, the codenames
        # board). Leaking it across episodes would make every game after the first
        # score against the previous game's progress.
        config = self._config(turn_rewards=True, turn_reward_scale=0.1, turn_reward_budget=0.0)
        from playpen.marshal.trainer import build_selfplay_rollout_func

        env = _StatefulEnv(total_turns=4, terminal_reward=1.0, violations={(0, 0)})
        fn = build_selfplay_rollout_func(env, config)
        first = fn(["0", "0"], _FakeTrainer())
        second = fn(["1", "1"], _FakeTrainer())
        # The rising edge must fire again in the second episode, not be suppressed by
        # the first episode's flag history.
        self.assertAlmostEqual(first["turn_reward_sum"][0], second["turn_reward_sum"][0])


class TestTrainerMetrics(unittest.TestCase):
    """The metrics that tell you whether the shaping is calibrated."""

    def _trainer(self, config=None):
        from collections import defaultdict, deque

        from playpen.marshal.trainer import MarshalGRPOTrainer

        t = object.__new__(MarshalGRPOTrainer)
        t.marshal_config = config or MarshalConfig()
        t._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        t._logs = {"advantages": deque(maxlen=8)}
        return t

    def _row(self, shaping, terminal=1.0, clipped=False, **components):
        row = {
            "turn_reward_sum": shaping,
            "terminal_reward": terminal,
            "turn_reward_clipped": clipped,
        }
        row.update({f"turn_reward_component_{k}": v for k, v in components.items()})
        return row

    def test_nothing_is_logged_when_the_feature_is_off(self):
        t = self._trainer()
        t._log_turn_reward_stats([{"seat": 0}, {"seat": 1}])
        self.assertEqual(dict(t._metrics["train"]), {})

    def test_magnitude_saturation_and_unshaped_outcome_are_logged(self):
        t = self._trainer(MarshalConfig(turn_rewards=True))
        t._log_turn_reward_stats([
            self._row(-0.3, terminal=1.0, clipped=True, format=-1.0),
            self._row(0.1, terminal=0.0, clipped=False, format=0.0),
        ])
        m = t._metrics["train"]
        self.assertAlmostEqual(m["marshal/turn_rewards/sum_mean"][-1], -0.1)
        self.assertAlmostEqual(m["marshal/turn_rewards/sum_abs_mean"][-1], 0.2)
        self.assertAlmostEqual(m["marshal/turn_rewards/sum_max_abs"][-1], 0.3)
        self.assertAlmostEqual(m["marshal/turn_rewards/budget_clip_rate"][-1], 0.5)
        self.assertAlmostEqual(m["marshal/turn_rewards/nonzero_rate"][-1], 1.0)
        # The column that keeps a turn-rewards arm comparable with one without it.
        self.assertAlmostEqual(m["marshal/turn_rewards/terminal_mean"][-1], 0.5)
        self.assertAlmostEqual(m["marshal/turn_rewards/component/format"][-1], -0.5)

    def test_an_all_zero_step_is_still_logged_as_zero(self):
        # "The signal fired and was 0" and "the feature is off" must not look alike.
        t = self._trainer(MarshalConfig(turn_rewards=True))
        t._log_turn_reward_stats([self._row(0.0, format=0.0)])
        self.assertEqual(t._metrics["train"]["marshal/turn_rewards/sum_mean"][-1], 0.0)

    def test_malformed_rows_do_not_break_training(self):
        t = self._trainer(MarshalConfig(turn_rewards=True))
        t._log_turn_reward_stats([{"turn_reward_sum": "not a number"}])  # must not raise


if __name__ == "__main__":
    unittest.main()

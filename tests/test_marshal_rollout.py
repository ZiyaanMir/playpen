"""Test the two-row self-play rollout construction with a mocked env/model.

No clemcore game, no model, no vLLM: a tiny scripted 2-seat env and a fake
tokenizer / generation function exercise ``play_selfplay_episode`` end to end,
asserting per-seat token/mask/turn-boundary/reward bookkeeping.

Runnable via ``pytest`` or directly with ``.venv/bin/python``.
"""

import unittest


class _FakePlayer:
    """Minimal stand-in for clemcore's Player (perceive_context/response)."""

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


class _FakeEnv:
    """Scripted 2-seat env: seats alternate for `total_turns`, terminal team reward."""

    def __init__(self, total_turns=4, terminal_reward=1.0):
        self.num_players = 2
        self._total = total_turns
        self._terminal = terminal_reward
        self._turn = 0
        self._players = {0: _FakePlayer(), 1: _FakePlayer()}
        self.finalized = False
        self.seen = []  # every utterance handed to the game, in order

    def reset(self, instance_idx, *, seed=0):
        self._turn = 0
        self.finalized = False

    @property
    def current_seat(self):
        return self._turn % 2

    @property
    def current_player(self):
        return self._players[self.current_seat]

    def observe(self):
        return {"role": "user", "content": f"observation number {self._turn}"}

    def step(self, response):
        self.seen.append(response)
        self._turn += 1

    @property
    def done(self):
        return self._turn >= self._total

    @property
    def cumulative_rewards(self):
        if self.done:
            return [self._terminal, self._terminal]
        return [0.0, 0.0]

    def finalize(self):
        self.finalized = True


class _FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return " ".join(m["content"] for m in messages)

    def encode(self, text, add_special_tokens=False):
        # One token id per whitespace-delimited word (values don't matter here).
        return [1] * len(text.split())

    def decode(self, ids, skip_special_tokens=True):
        return "decoded"


class _FakeTrainer:
    def __init__(self):
        self.processing_class = _FakeTokenizer()


def _fake_generate(target_tokens=(10, 11), logprobs=(-0.1, -0.2)):
    def generate_rollout_completions(trainer, prompts, **kwargs):
        return [
            {
                "prompt_ids": [1, 2, 3],
                "completion_ids": list(target_tokens),
                "logprobs": list(logprobs),
                "text": "RESPONSE",
            }
            for _ in prompts
        ]

    return generate_rollout_completions


class TestRolloutConstruction(unittest.TestCase):
    def setUp(self):
        # Patch the real module attribute so `from trl.experimental.openenv import
        # generate_rollout_completions` inside play_selfplay_episode picks up the fake.
        import trl.experimental.openenv as oe

        self._oe = oe
        self._orig = oe.generate_rollout_completions
        oe.generate_rollout_completions = _fake_generate()

    def tearDown(self):
        self._oe.generate_rollout_completions = self._orig

    def test_two_seat_rows_built_correctly(self):
        from playpen.marshal.selfplay_agent import play_selfplay_episode

        env = _FakeEnv(total_turns=4, terminal_reward=1.0)
        rollouts = play_selfplay_episode(env, _FakeTrainer(), _FakeTokenizer(), instance_idx=0)

        self.assertEqual(set(rollouts.keys()), {0, 1})
        seat0 = rollouts[0]
        seat1 = rollouts[1]

        # Each seat had exactly 2 turns.
        self.assertEqual(len(seat0.turn_end_positions), 2)
        self.assertEqual(len(seat1.turn_end_positions), 2)

        # 2 turns x 2 model tokens = 4 model tokens per seat.
        self.assertEqual(sum(seat0.owner_mask), 4)
        self.assertEqual(sum(seat1.owner_mask), 4)

        # There is env feedback (mask 0) between the two turns.
        self.assertIn(0, seat0.owner_mask)

        # owner_mask aligns with completion_ids / logprobs length.
        self.assertEqual(len(seat0.owner_mask), len(seat0.completion_ids))
        self.assertEqual(len(seat0.logprobs), len(seat0.completion_ids))

        # Terminal team reward reconciled onto the last turn of each seat.
        self.assertEqual(seat0.terminal_reward, 1.0)
        self.assertEqual(seat0.turn_rewards[-1], 1.0)
        self.assertEqual(seat0.turn_rewards[0], 0.0)

        # Turn-end positions point at model tokens.
        for pos in seat0.turn_end_positions:
            self.assertEqual(seat0.owner_mask[pos], 1)

        self.assertTrue(env.finalized)

    def test_prompt_ids_set_from_first_turn_only(self):
        from playpen.marshal.selfplay_agent import play_selfplay_episode

        env = _FakeEnv(total_turns=4)
        rollouts = play_selfplay_episode(env, _FakeTrainer(), _FakeTokenizer(), instance_idx=0)
        # prompt_ids come from the seat's first observation; non-empty.
        self.assertTrue(len(rollouts[0].prompt_ids) > 0)


class TestEmptyRowPlaceholder(unittest.TestCase):
    """The placeholder row for a seat with no trajectory must be training-inert."""

    def test_empty_row_is_env_token_only(self):
        from playpen.marshal.trainer import _empty_row

        row = _empty_row(seat=1, pad_token_id=7)
        # Non-empty tensors for TRL's contract, but no model tokens and no turns:
        # the loss mask (env_mask == owner_mask) zeroes it and the advantage
        # pooling excludes it.
        self.assertEqual(row.prompt_ids, [7])
        self.assertEqual(row.completion_ids, [7])
        self.assertEqual(row.owner_mask, [0])
        self.assertEqual(row.turn_end_positions, [])
        self.assertEqual(row.turn_rewards, [])
        self.assertFalse(row.has_model_tokens)
        self.assertEqual(row.terminal_reward, 0.0)


class TestInstanceIndexing(unittest.TestCase):
    """Regression: instance addressing must be unique across experiments.

    clembench game_ids repeat per experiment (taboo: 0..19 in each of
    high_en/medium_en/low_en), so the old game_id-based lookup silently mapped
    every prompt onto the first experiment. Uses the real packaged taboo
    instances; skipped when the clembench registry is not resolvable.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from playpen.marshal.selfplay_env import load_instance_rows

            cls.rows = load_instance_rows("taboo")
        except Exception as e:  # registry/game not available in this checkout
            raise unittest.SkipTest(f"taboo instances not resolvable: {e}")

    def test_indices_cover_all_experiments(self):
        from playpen.marshal.selfplay_env import list_instance_indices

        indices = list_instance_indices("taboo")
        self.assertEqual(len(indices), len(self.rows))
        self.assertEqual(len(indices), len(set(indices)))  # unique by construction
        experiments = {row["experiment"]["name"] for row in self.rows}
        # The whole point: more than one experiment is addressable.
        self.assertGreater(len(experiments), 1)

    def test_reset_loads_the_indexed_instance(self):
        from playpen.marshal.selfplay_env import SelfPlayEnv

        env = SelfPlayEnv("taboo")
        # Pick an index whose experiment differs from the first row's experiment;
        # the old game_id lookup could never reach it.
        first_exp = self.rows[0]["experiment"]["name"]
        idx = next(
            i for i, row in enumerate(self.rows) if row["experiment"]["name"] != first_exp
        )
        env.reset(idx, seed=0)
        master_env = env._pz_env.unwrapped
        self.assertEqual(
            master_env.game_instance, self.rows[idx]["game_instance"]
        )
        self.assertEqual(
            master_env.experiment["name"], self.rows[idx]["experiment"]["name"]
        )

    def test_reset_rejects_out_of_range_index(self):
        from playpen.marshal.selfplay_env import SelfPlayEnv

        env = SelfPlayEnv("taboo")
        with self.assertRaises(IndexError):
            env.reset(len(self.rows))


class TestStripReasoning(unittest.TestCase):
    """Text-level <think> stripping (the fallback path)."""

    def test_complete_block_removed(self):
        from playpen.marshal.selfplay_agent import strip_reasoning

        self.assertEqual(
            strip_reasoning("<think>hmm, vowels?</think>QUESTION: is it a noun?"),
            "QUESTION: is it a noun?",
        )

    def test_dangling_close_tag(self):
        # Templates that pre-fill "<think>" into the prompt: the completion starts
        # already inside the block, so only the closing tag appears.
        from playpen.marshal.selfplay_agent import strip_reasoning

        self.assertEqual(
            strip_reasoning("reasoning with no open tag</think>\n\nGUESS: apple"),
            "GUESS: apple",
        )

    def test_unterminated_block_yields_empty(self):
        # Budget ran out mid-thought: there is no answer, and saying so honestly
        # (empty -> the game aborts this turn) beats leaking reasoning to the parser.
        from playpen.marshal.selfplay_agent import strip_reasoning

        self.assertEqual(strip_reasoning("<think>still thinking and then cut o"), "")

    def test_multiple_blocks(self):
        from playpen.marshal.selfplay_agent import strip_reasoning

        self.assertEqual(
            strip_reasoning("<think>a</think>QUESTION: x?<think>b</think>"),
            "QUESTION: x?",
        )

    def test_no_reasoning_is_passthrough(self):
        from playpen.marshal.selfplay_agent import strip_reasoning

        self.assertEqual(strip_reasoning("QUESTION: is it red?"), "QUESTION: is it red?")

    def test_guesswhat_prefix_contract_is_restored(self):
        # The actual bug: clembench guesswhat aborts unless the utterance STARTS WITH
        # the tag (guesswhat/master.py:161).
        from playpen.marshal.selfplay_agent import strip_reasoning

        raw = "<think>The word has five letters.</think>\n\nQUESTION: does it fly?"
        self.assertFalse(raw.startswith("QUESTION:"))
        self.assertTrue(strip_reasoning(raw).startswith("QUESTION:"))


class _ThinkTokenizer(_FakeTokenizer):
    """Tokenizer that knows </think> as a token id, exercising the token-level path."""

    CLOSE_ID = 999

    def convert_tokens_to_ids(self, token):
        return self.CLOSE_ID if token == "</think>" else None

    def decode(self, ids, skip_special_tokens=True):
        # Only the ids after the close tag are ever passed here by the token path.
        return "GUESS: apple" if list(ids) == [7, 8] else "decoded"


class TestResponseForGame(unittest.TestCase):
    """Token-level strip is preferred, because TRL's `text` drops special tokens."""

    def test_token_level_slices_after_close_tag(self):
        from playpen.marshal.selfplay_agent import response_for_game

        # `text` has already lost its tags (skip_special_tokens=True), so a text-only
        # strip would leak the reasoning prose. The ids still carry the boundary.
        text_without_tags = "some reasoning GUESS: apple"
        ids = [1, 2, _ThinkTokenizer.CLOSE_ID, 7, 8]
        self.assertEqual(response_for_game(text_without_tags, ids, _ThinkTokenizer()), "GUESS: apple")

    def test_falls_back_to_text_when_no_think_token(self):
        from playpen.marshal.selfplay_agent import response_for_game

        got = response_for_game("<think>x</think>QUESTION: y?", [1, 2, 3], _FakeTokenizer())
        self.assertEqual(got, "QUESTION: y?")


def _fake_generate_with_thinking():
    def generate_rollout_completions(trainer, prompts, **kwargs):
        return [
            {
                "prompt_ids": [1, 2, 3],
                "completion_ids": [10, 11, 12, 13],  # 4 tokens: reasoning + answer
                "logprobs": [-0.1, -0.2, -0.3, -0.4],
                "text": "<think>deliberating</think>QUESTION: is it alive?",
            }
            for _ in prompts
        ]

    return generate_rollout_completions


class TestThinkingIsTrainedButNotSentToGame(unittest.TestCase):
    """The whole point of option B: train the reasoning, hide it from the parser."""

    def setUp(self):
        import trl.experimental.openenv as oe

        self._oe = oe
        self._orig = oe.generate_rollout_completions
        oe.generate_rollout_completions = _fake_generate_with_thinking()

    def tearDown(self):
        self._oe.generate_rollout_completions = self._orig

    def test_game_sees_stripped_history_keeps_stripped_tokens_keep_thinking(self):
        from playpen.marshal.selfplay_agent import play_selfplay_episode

        env = _FakeEnv(total_turns=2, terminal_reward=1.0)
        rollouts = play_selfplay_episode(env, _FakeTrainer(), _FakeTokenizer(), instance_idx=0)

        # The game only ever received the answer — no reasoning leaked to the parser.
        self.assertTrue(env.seen)
        for utterance in env.seen:
            self.assertEqual(utterance, "QUESTION: is it alive?")
            self.assertNotIn("<think>", utterance)

        # ...but every generated token (reasoning included) is still trained.
        seat0 = rollouts[0]
        self.assertEqual(seat0.completion_ids, [10, 11, 12, 13])
        self.assertEqual(sum(seat0.owner_mask), 4)

    def test_strip_think_false_leaks_raw_response(self):
        from playpen.marshal.selfplay_agent import play_selfplay_episode

        env = _FakeEnv(total_turns=2, terminal_reward=1.0)
        play_selfplay_episode(
            env, _FakeTrainer(), _FakeTokenizer(), instance_idx=0, strip_think=False
        )
        self.assertIn("<think>", env.seen[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)

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


class _ChatTokenizer:
    """A fake tokenizer that models a real chat template's turn markers.

    ``_FakeTokenizer`` joins message contents with spaces and emits no turn boundaries,
    which is precisely why it could not distinguish a row that reproduces its generation
    context from one that does not. This one emits Qwen-style markers and tokenizes one
    id per character, so concatenation is exact and a prefix check is meaningful.
    """

    eos_token = "<|im_end|>"

    def apply_chat_template(
        self, messages, add_generation_prompt=True, tokenize=False, enable_thinking=None
    ):
        out = ""
        for m in messages:
            out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return out

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


def _fake_generate_echo(response="MOVE", tokenizer=None):
    """Generation fake that reports the prompt ids it was actually given.

    That is the whole point: ``sync_context`` reads ``prompt_ids`` back, so a fake that
    invents them (like ``_fake_generate``) cannot exercise the contract.
    """
    tok = tokenizer or _ChatTokenizer()

    def generate_rollout_completions(trainer, prompts, **kwargs):
        results = []
        for prompt in prompts:
            body = response + tok.eos_token
            ids = tok.encode(body)
            results.append(
                {
                    "prompt_ids": tok.encode(prompt),
                    "completion_ids": ids,
                    "logprobs": [-0.5] * len(ids),
                    "text": response,
                }
            )
        return results

    return generate_rollout_completions


class TestExactRowReproducesGenerationContext(unittest.TestCase):
    """Regression guard for the row/context mismatch.

    Before the fix, a row was `turn-1 prompt ++ gen1 ++ "\\n\\n"+obs2 ++ gen2 ...` while
    generation used a freshly rendered chat template, so from turn 2 onward the trained
    sequence was never the generated one. The contract asserted here is the fix: for
    every turn, the ids vLLM conditioned on are a prefix of the row.
    """

    def setUp(self):
        import trl.experimental.openenv as oe

        self._oe = oe
        self._orig = oe.generate_rollout_completions
        self.tok = _ChatTokenizer()
        self.seen = []
        inner = _fake_generate_echo(tokenizer=self.tok)

        def recording(trainer, prompts, **kwargs):
            out = inner(trainer, prompts, **kwargs)
            self.seen.extend(o["prompt_ids"] for o in out)
            return out

        oe.generate_rollout_completions = recording

    def tearDown(self):
        self._oe.generate_rollout_completions = self._orig

    def _play(self, mode):
        from playpen.marshal.selfplay_agent import play_selfplay_episode

        self.seen.clear()
        env = _FakeEnv(total_turns=6, terminal_reward=1.0)
        return play_selfplay_episode(
            env, _FakeTrainer(), self.tok, instance_idx=0, row_context_mode=mode
        )

    def test_every_generation_context_is_a_prefix_of_the_row(self):
        rollouts = self._play("exact")
        # seats alternate, so seat 0 generated on calls 0, 2, 4
        for seat, rollout in rollouts.items():
            row = list(rollout.prompt_ids) + list(rollout.completion_ids)
            for ctx in self.seen[seat::2]:
                self.assertEqual(
                    row[: len(ctx)], list(ctx),
                    f"seat {seat}: a generation context is not a prefix of the trained row",
                )

    def test_row_ends_exactly_at_last_context_plus_generation(self):
        rollouts = self._play("exact")
        for seat, rollout in rollouts.items():
            row = list(rollout.prompt_ids) + list(rollout.completion_ids)
            last_ctx = list(self.seen[seat::2][-1])
            gen_len = len(row) - len(last_ctx)
            self.assertGreater(gen_len, 0)
            self.assertEqual(row[: len(last_ctx)], last_ctx)
            # ...and those trailing tokens are exactly the ones marked model-generated.
            self.assertTrue(all(m == 1 for m in rollout.owner_mask[-gen_len:]))

    def test_turn_boundaries_and_mask_still_line_up(self):
        rollouts = self._play("exact")
        for rollout in rollouts.values():
            self.assertEqual(len(rollout.owner_mask), len(rollout.completion_ids))
            self.assertEqual(len(rollout.logprobs), len(rollout.completion_ids))
            self.assertEqual(len(rollout.turn_end_positions), 3)
            for pos in rollout.turn_end_positions:
                self.assertEqual(rollout.owner_mask[pos], 1)
            # Environment scaffolding is present and carries no gradient.
            self.assertIn(0, rollout.owner_mask)

    def test_spliced_mode_still_reproduces_the_old_shape(self):
        rollouts = self._play("spliced")
        for seat, rollout in rollouts.items():
            row = list(rollout.prompt_ids) + list(rollout.completion_ids)
            contexts = self.seen[seat::2]
            # Turn 1 matches in both modes; the later ones are what used to diverge.
            self.assertEqual(row[: len(contexts[0])], list(contexts[0]))
            self.assertNotEqual(row[: len(contexts[-1])], list(contexts[-1]))


class TestExactRowDriftIsDropped(unittest.TestCase):
    """A row whose owner mask cannot be proven correct must not be trained on."""

    def setUp(self):
        import trl.experimental.openenv as oe

        self._oe = oe
        self._orig = oe.generate_rollout_completions
        tok = _ChatTokenizer()
        inner = _fake_generate_echo(tokenizer=tok)
        self.calls = 0

        def drifting(trainer, prompts, **kwargs):
            out = inner(trainer, prompts, **kwargs)
            self.calls += 1
            if self.calls == 3:  # seat 0's second turn: pretend re-tokenization merged
                out[0]["prompt_ids"] = [999] + list(out[0]["prompt_ids"])
            return out

        oe.generate_rollout_completions = drifting
        self.tok = tok

    def tearDown(self):
        self._oe.generate_rollout_completions = self._orig

    def test_drifted_seat_is_inert_but_still_reports_its_outcome(self):
        from playpen.marshal.selfplay_agent import play_selfplay_episode

        env = _FakeEnv(total_turns=6, terminal_reward=1.0)
        rollouts = play_selfplay_episode(
            env, _FakeTrainer(), self.tok, instance_idx=0, row_context_mode="exact"
        )
        drifted = rollouts[0]
        # Not trainable: an owner mask we cannot prove must carry no gradient...
        self.assertFalse(drifted.has_model_tokens)
        self.assertEqual(drifted.turn_end_positions, [])
        self.assertEqual(drifted.turn_rewards, [])
        # ...but the episode's real outcome is still reported, so reward statistics
        # (and the plain-GRPO group baseline) do not see a fabricated 0.0.
        self.assertEqual(drifted.terminal_reward, 1.0)
        self.assertTrue(rollouts[1].has_model_tokens)  # the other seat is unaffected
        self.assertTrue(env.finalized)                 # and the game ran to the end


class _StubEnv:
    """Just enough of SelfPlayEnv for build_selfplay_rollout_func to close over."""

    def __init__(self, num_players=2):
        self.num_players = num_players


class TestEpisodePairing(unittest.TestCase):
    """One episode must serve every seat of a run of identical prompts.

    MARSHAL emits both seats of a game from that one game. The pre-fix port replayed a
    whole fresh episode per prompt and discarded the other seat, so a "pair" of seat
    rows came from two different games (and cost 2x the generation).
    """

    def setUp(self):
        import playpen.marshal.trainer as tr
        from playpen.marshal.selfplay_agent import SeatRollout

        self.tr = tr
        self._orig = tr.play_selfplay_episode
        self.calls = []

        def fake_play(env, trainer, tokenizer, instance_idx, **kwargs):
            # Tag each seat's row with the episode that produced it, so a test can tell
            # whether two rows came from the same play.
            episode = len(self.calls)
            self.calls.append(instance_idx)
            return {
                seat: SeatRollout(
                    seat=seat,
                    prompt_ids=[1],
                    completion_ids=[2, 3],
                    logprobs=[-0.1, -0.2],
                    owner_mask=[1, 1],
                    turn_end_positions=[1],
                    turn_rewards=[1.0],
                    terminal_reward=float(episode),  # episode id, for identification
                )
                for seat in (0, 1)
            }

        tr.play_selfplay_episode = fake_play

    def tearDown(self):
        self.tr.play_selfplay_episode = self._orig

    def _run(self, prompts, pairing, num_players=2):
        from playpen.marshal.config import MarshalConfig

        cfg = MarshalConfig(episode_pairing=pairing)
        fn = self.tr.build_selfplay_rollout_func(_StubEnv(num_players), cfg)
        out = fn(prompts, _FakeTrainer())
        self._assert_no_cross_instance_rows(prompts, out)
        return out

    def _assert_no_cross_instance_rows(self, prompts, out):
        """No row may be served an episode belonging to a different game instance.

        ``terminal_reward`` carries the episode id and ``self.calls[episode]`` is the
        instance that episode was played on, so this is checkable directly. It is the
        one thing pairing could plausibly get wrong, so every case asserts it.
        """
        from playpen.marshal.trainer import parse_prompt

        for i, prompt in enumerate(prompts):
            expected, _ = parse_prompt(prompt)
            served = self.calls[int(out["rewards"][i])]
            self.assertEqual(
                served, expected,
                f"row {i} (prompt {prompt!r}) was served an episode of instance {served}",
            )

    def test_shared_serves_a_whole_run_from_one_episode(self):
        # TRL's RepeatSampler emits num_generations copies consecutively.
        prompts = ["5"] * 4 + ["7"] * 4
        out = self._run(prompts, "shared")
        self.assertEqual(len(self.calls), 4, "expected 8 prompts -> 4 paired episodes")
        self.assertEqual(self.calls, [5, 5, 7, 7])
        self.assertEqual(out["seat"], [0, 1, 0, 1, 0, 1, 0, 1])
        # Rows 0/1 share an episode, 2/3 share the next one, and so on.
        episodes = out["rewards"]
        self.assertEqual(episodes[0], episodes[1])
        self.assertEqual(episodes[2], episodes[3])
        self.assertNotEqual(episodes[0], episodes[2])

    def test_shared_halves_the_generation_cost(self):
        self._run(["5"] * 8, "shared")
        shared_calls = len(self.calls)
        self.calls.clear()
        self._run([f"5::seat{i % 2}" for i in range(8)], "replay")
        self.assertEqual(shared_calls, 4)
        self.assertEqual(len(self.calls), 8)

    def test_replay_is_unchanged(self):
        prompts = ["5::seat0"] * 2 + ["5::seat1"] * 2
        out = self._run(prompts, "replay")
        self.assertEqual(len(self.calls), 4, "replay plays one episode per prompt")
        self.assertEqual(out["seat"], [0, 0, 1, 1])
        # Every row came from its own episode.
        self.assertEqual(len(set(out["rewards"])), 4)

    def test_seat_pinned_keys_are_never_paired_even_in_shared_mode(self):
        out = self._run(["5::seat0", "5::seat0"], "shared")
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(out["seat"], [0, 0])

    def test_unpairable_run_falls_back_instead_of_mixing_games(self):
        # num_generations=3 with 2 players: the runs do not divide evenly, so index 2
        # would otherwise be paired with index 3 -- a *different* instance.
        prompts = ["5"] * 3 + ["7"] * 3
        out = self._run(prompts, "shared")
        # Indices 0-1 pair (one episode of 5); 2 and 3 straddle the instance boundary so
        # each replays alone (5 then 7); 4-5 pair (one episode of 7).
        self.assertEqual(self.calls, [5, 5, 7, 7])
        self.assertEqual(out["seat"], [0, 1, 0, 1, 0, 1])
        # The invariant that matters: no row is ever served an episode of a *different*
        # instance, and the straddling indices do not share one.
        self.assertNotEqual(out["rewards"][2], out["rewards"][3])

    def test_odd_length_tail_falls_back(self):
        out = self._run(["5"] * 3, "shared")
        self.assertEqual(self.calls, [5, 5])  # indices 0,1 pair; index 2 replays
        self.assertEqual(out["seat"], [0, 1, 0])

    def test_three_player_game_pairs_in_threes(self):
        out = self._run(["5"] * 6, "shared", num_players=3)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(out["seat"], [0, 1, 2, 0, 1, 2])
        self.assertEqual(out["rewards"][0], out["rewards"][2])
        self.assertNotEqual(out["rewards"][2], out["rewards"][3])

    def test_row_payload_is_intact(self):
        out = self._run(["5"] * 2, "shared")
        for key in ("prompt_ids", "completion_ids", "logprobs", "env_mask",
                    "owner_mask", "turn_end_positions", "turn_rewards"):
            self.assertEqual(len(out[key]), 2, key)
        self.assertEqual(out["env_mask"], out["owner_mask"])
        self.assertEqual(out["completion_ids"][0], [2, 3])


class TestPairedRolloutEndToEnd(unittest.TestCase):
    """Pairing and exact-row assembly through the real code path, no mocking of either.

    The unit tests above stub ``play_selfplay_episode`` (pairing) or drive it directly
    (row assembly). This one runs ``build_selfplay_rollout_func`` -> the real
    ``play_selfplay_episode`` -> the real ``_SeatBuilder``, so it catches a regression
    in the seam between them.
    """

    def setUp(self):
        import trl.experimental.openenv as oe

        self._oe = oe
        self._orig = oe.generate_rollout_completions
        self.tok = _ChatTokenizer()
        self.contexts = []
        inner = _fake_generate_echo(tokenizer=self.tok)

        def recording(trainer, prompts, **kwargs):
            out = inner(trainer, prompts, **kwargs)
            self.contexts.extend(o["prompt_ids"] for o in out)
            return out

        oe.generate_rollout_completions = recording

    def tearDown(self):
        self._oe.generate_rollout_completions = self._orig

    def test_one_episode_two_seats_each_row_matching_its_own_contexts(self):
        from playpen.marshal.config import MarshalConfig
        from playpen.marshal.trainer import build_selfplay_rollout_func

        env = _FakeEnv(total_turns=6, terminal_reward=1.0)

        class _T:
            processing_class = self.tok

        fn = build_selfplay_rollout_func(
            env, MarshalConfig(episode_pairing="shared", row_context_mode="exact")
        )
        out = fn(["5", "5"], _T())

        # ONE episode served both rows: 6 turns played, not 12.
        self.assertEqual(len(env.seen), 6)
        self.assertEqual(len(self.contexts), 6)
        self.assertEqual(out["seat"], [0, 1])
        # Both rows saw the same game, so they carry the same terminal reward.
        self.assertEqual(out["rewards"][0], out["rewards"][1])

        # ...and each row is still exactly the context its own seat generated under.
        for row_index, seat in enumerate(out["seat"]):
            row = list(out["prompt_ids"][row_index]) + list(out["completion_ids"][row_index])
            for ctx in self.contexts[seat::2]:  # seats alternate turn by turn
                self.assertEqual(
                    row[: len(ctx)], list(ctx),
                    f"row {row_index} (seat {seat}) diverges from a generation context",
                )
            self.assertEqual(len(out["owner_mask"][row_index]),
                             len(out["completion_ids"][row_index]))


class TestSeatThatNeverPlayedReportsTheRealOutcome(unittest.TestCase):
    """A seat with no trajectory must report the episode's outcome, not a fake 0.0.

    Regression for a reporting bug measured on the codenames run: ~48% of rows were
    seats that never moved (the game aborted before their turn), every one of them
    logged reward 0.0, and the logged abort rate came out at 33.7% against 64.3% among
    rows that actually played. On the plain-GRPO path that fabricated 0.0 also entered
    the group baseline directly.
    """

    def setUp(self):
        import trl.experimental.openenv as oe

        self._oe = oe
        self._orig = oe.generate_rollout_completions
        oe.generate_rollout_completions = _fake_generate()

    def tearDown(self):
        self._oe.generate_rollout_completions = self._orig

    def test_non_acting_seat_carries_the_team_reward(self):
        from playpen.marshal.selfplay_agent import play_selfplay_episode

        # total_turns=1: seat 0 moves, the game ends, seat 1 never gets a turn.
        env = _FakeEnv(total_turns=1, terminal_reward=-1.0)
        rollouts = play_selfplay_episode(
            env, _FakeTrainer(), _FakeTokenizer(), instance_idx=0, row_context_mode="spliced"
        )
        self.assertIn(1, rollouts, "the non-acting seat must still be reported")
        idle = rollouts[1]
        self.assertEqual(idle.terminal_reward, -1.0, "must be the ABORT, not a fake 0.0")
        self.assertFalse(idle.has_model_tokens, "and it must stay training-inert")
        self.assertEqual(rollouts[0].terminal_reward, -1.0)

    def test_placeholder_contributes_no_gradient_and_no_pool_statistics(self):
        import torch
        from playpen.marshal.advantage import RowRollout, compute_marshal_advantages

        # One real winning row and one placeholder that reports the same outcome.
        real = RowRollout(seat=0, completion_len=4, owner_mask=[1, 1, 1, 1],
                          turn_end_positions=[3], turn_rewards=[1.0])
        idle = RowRollout(seat=1, completion_len=1, owner_mask=[0],
                          turn_end_positions=[], turn_rewards=[])
        adv = compute_marshal_advantages([real, idle], 4, agent_specific=True,
                                         norm_mode="mean")
        self.assertTrue(torch.equal(adv[1], torch.zeros(4)),
                        "placeholder row must carry no advantage anywhere")


class TestParsePrompt(unittest.TestCase):
    def test_both_key_shapes(self):
        from playpen.marshal.trainer import parse_prompt

        self.assertEqual(parse_prompt("12::seat1"), (12, 1))
        self.assertEqual(parse_prompt("12"), (12, None))


class TestDatasetKeyShape(unittest.TestCase):
    """The dataset key shape must match the pairing mode the rollout func expects."""

    @classmethod
    def setUpClass(cls):
        try:
            from playpen.marshal.selfplay_env import load_instance_rows

            cls.n = len(load_instance_rows("taboo"))
        except Exception as e:
            raise unittest.SkipTest(f"clembench taboo not resolvable: {e}")

    def test_shared_is_one_row_per_instance(self):
        from playpen.marshal.trainer import build_selfplay_dataset

        ds = build_selfplay_dataset("taboo", num_players=2, episode_pairing="shared")
        self.assertEqual(len(ds), self.n)
        self.assertNotIn("::seat", ds["prompt"][0])

    def test_replay_is_one_row_per_instance_and_seat(self):
        from playpen.marshal.trainer import build_selfplay_dataset

        ds = build_selfplay_dataset("taboo", num_players=2, episode_pairing="replay")
        self.assertEqual(len(ds), 2 * self.n)
        self.assertIn("::seat", ds["prompt"][0])


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
        rollouts = play_selfplay_episode(
            env, _FakeTrainer(), _FakeTokenizer(), instance_idx=0, row_context_mode="spliced"
        )

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
        rollouts = play_selfplay_episode(
            env, _FakeTrainer(), _FakeTokenizer(), instance_idx=0, row_context_mode="spliced"
        )
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


class TestInstanceIsNotMutatedAcrossEpisodes(unittest.TestCase):
    """A game must never be able to damage the packaged instance it was handed.

    clembench games are written for benchmark runs that play each instance once, so
    they may keep references into `game_instance` and mutate them. codenames does:
    `CodenamesBoard.__init__` stores the assignment lists by reference and
    `reveal_word` calls `.remove(word)` on them. A training loop replays one instance
    thousands of times, which turned that into permanent corruption -- a 25-word
    codenames board fell to ~7 words by step 150 of a 500-step run.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from playpen.marshal.selfplay_env import SelfPlayEnv

            cls.env = SelfPlayEnv("taboo")
        except Exception as e:
            raise unittest.SkipTest(f"clembench taboo not resolvable: {e}")

    def test_reset_hands_down_a_copy_not_the_cached_row(self):
        import copy

        row = self.env._instance_rows[0]
        pristine = copy.deepcopy(row)
        captured = {}

        real_reset = self.env._pz_env.reset

        def hostile_reset(seed=None, options=None):
            # Stand in for a game that keeps a reference and mutates it.
            captured["instance"] = options["game_instance"]
            for value in options["game_instance"].values():
                if isinstance(value, list):
                    value.clear()
                    break
            options["experiment"]["__scribbled__"] = True

        self.env._pz_env.reset = hostile_reset
        try:
            self.env.reset(0, seed=0)
        finally:
            self.env._pz_env.reset = real_reset

        self.assertIsNot(captured["instance"], row["game_instance"],
                         "reset must not hand the cached row's dict straight to the game")
        self.assertEqual(row, pristine,
                         "the cached instance row was mutated by the game")


class TestRenderPromptDisablesThinking(unittest.TestCase):
    """Reasoning is disabled at the source, for every model and game."""

    def test_passes_enable_thinking_false(self):
        from playpen.marshal.selfplay_agent import render_prompt

        seen = {}

        class _Tok:
            def apply_chat_template(self, messages, **kwargs):
                seen.update(kwargs)
                return "PROMPT"

        out = render_prompt(_Tok(), [{"role": "user", "content": "hi"}])
        self.assertTrue(out.startswith("PROMPT"))  # + the empty-think prefill
        self.assertIs(seen["enable_thinking"], False)
        self.assertIs(seen["add_generation_prompt"], True)

    def test_falls_back_when_template_rejects_the_kwarg(self):
        # Older tokenizers raise on unknown kwargs; there is simply nothing to
        # disable there, so the render must still succeed.
        from playpen.marshal.selfplay_agent import render_prompt

        calls = []

        class _StrictTok:
            def apply_chat_template(self, messages, **kwargs):
                calls.append(kwargs)
                if "enable_thinking" in kwargs:
                    raise TypeError("unexpected keyword argument 'enable_thinking'")
                return "PROMPT"

        out = render_prompt(_StrictTok(), [])
        self.assertTrue(out.startswith("PROMPT"))
        self.assertEqual(len(calls), 2)  # tried with, then without
        self.assertNotIn("enable_thinking", calls[1])

    def test_no_think_tag_appended_to_last_user_message(self):
        from playpen.marshal.selfplay_agent import NO_THINK_TAG, render_prompt

        seen = {}

        class _Tok:
            def apply_chat_template(self, messages, **kwargs):
                seen["messages"] = messages
                return "PROMPT"

        original = [{"role": "user", "content": "your move"}]
        render_prompt(_Tok(), original)
        self.assertTrue(seen["messages"][-1]["content"].endswith(NO_THINK_TAG))
        # The caller's list/dicts must not be mutated: clemcore keeps that same
        # observation dict, so mutation would compound the tag every turn.
        self.assertEqual(original[0]["content"], "your move")

    def test_no_think_tag_not_duplicated(self):
        from playpen.marshal.selfplay_agent import NO_THINK_TAG, render_prompt

        seen = {}

        class _Tok:
            def apply_chat_template(self, messages, **kwargs):
                seen["messages"] = messages
                return "PROMPT"

        render_prompt(_Tok(), [{"role": "user", "content": f"already there {NO_THINK_TAG}"}])
        self.assertEqual(seen["messages"][-1]["content"].count(NO_THINK_TAG), 1)

    def test_empty_think_block_prefilled(self):
        from playpen.marshal.selfplay_agent import EMPTY_THINK_PREFILL, render_prompt

        class _Tok:
            def apply_chat_template(self, messages, **kwargs):
                return "PROMPT<|im_start|>assistant\n"

        self.assertTrue(render_prompt(_Tok(), []).endswith(EMPTY_THINK_PREFILL))

    def test_prefill_not_doubled_when_template_already_closed_a_block(self):
        # Newer Qwen3 templates emit their own empty block under enable_thinking=False.
        from playpen.marshal.selfplay_agent import render_prompt

        class _Tok:
            def apply_chat_template(self, messages, **kwargs):
                return "PROMPT<|im_start|>assistant\n<think>\n\n</think>\n\n"

        self.assertEqual(render_prompt(_Tok(), []).count("<think>"), 1)

    def test_close_tag_in_history_does_not_suppress_the_prefill(self):
        # A </think> from an earlier assistant turn is far from the tail and must
        # not be mistaken for the template's own prefill.
        from playpen.marshal.selfplay_agent import EMPTY_THINK_PREFILL, render_prompt

        class _Tok:
            def apply_chat_template(self, messages, **kwargs):
                return "old turn</think>" + "x" * 200 + "<|im_start|>assistant\n"

        self.assertTrue(render_prompt(_Tok(), []).endswith(EMPTY_THINK_PREFILL))


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
        rollouts = play_selfplay_episode(
            env, _FakeTrainer(), _FakeTokenizer(), instance_idx=0, row_context_mode="spliced"
        )

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
            env, _FakeTrainer(), _FakeTokenizer(), instance_idx=0, strip_think=False,
            row_context_mode="spliced",
        )
        self.assertIn("<think>", env.seen[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)

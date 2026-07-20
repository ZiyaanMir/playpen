"""Per-seat rollout collection for MARSHAL-style self-play.

One physical self-play episode is played through :class:`SelfPlayEnv`, and each
seat's turn-alternating trajectory is accumulated into its own :class:`SeatRollout`
-- the seat-aware, turn-boundary-aware generalization of the single-seat
``GrpoEpisodeRollout`` in ``examples/openenv/wordle-trl.ipynb``.

The per-seat conversation is built from clemcore's own ``Player.perceive_context``
/ ``perceive_response``, i.e. the exact message list clemcore uses at inference
time, so there is no train/eval prompt-format drift.

``trl`` is imported lazily (inside :func:`play_selfplay_episode`) so this module is
importable without the training stack; ``transformers`` tokenizers are fine to
type against but are only used at call time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from playpen.marshal.selfplay_env import SelfPlayEnv


@dataclass
class SeatRollout:
    """The collected training data for a single seat within one episode.

    ``owner_mask`` / ``env_mask`` are the same thing (1 = model token, 0 =
    environment token); the name ``env_mask`` is what TRL expects as an extra
    field, ``owner_mask`` is what the advantage code reads.
    """

    seat: int
    prompt_ids: List[int] = field(default_factory=list)
    completion_ids: List[int] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)
    owner_mask: List[int] = field(default_factory=list)
    turn_end_positions: List[int] = field(default_factory=list)
    turn_rewards: List[float] = field(default_factory=list)
    terminal_reward: float = 0.0

    @property
    def has_model_tokens(self) -> bool:
        return any(m == 1 for m in self.owner_mask)


class _SeatBuilder:
    """Accumulates a single seat's flattened trajectory across its turns."""

    def __init__(self, seat: int) -> None:
        self.seat = seat
        self.prompt_ids: List[int] = []
        self.completion_ids: List[int] = []
        self.logprobs: List[float] = []
        self.owner_mask: List[int] = []
        self.turn_end_positions: List[int] = []
        self.turn_rewards: List[float] = []
        self._first_turn = True

    @property
    def first_turn(self) -> bool:
        return self._first_turn

    @property
    def has_turns(self) -> bool:
        return len(self.turn_end_positions) > 0

    def set_prompt(self, ids: Sequence[int]) -> None:
        self.prompt_ids = list(ids)
        self._first_turn = False

    def add_env_tokens(self, ids: Sequence[int]) -> None:
        """Environment-feedback tokens received since this seat's previous turn."""
        self.completion_ids.extend(ids)
        self.logprobs.extend([0.0] * len(ids))
        self.owner_mask.extend([0] * len(ids))

    def add_model_tokens(self, ids: Sequence[int], logprobs: Sequence[float]) -> None:
        self.completion_ids.extend(ids)
        self.logprobs.extend(list(logprobs))
        self.owner_mask.extend([1] * len(ids))
        # Turn ends at the last generated token of this turn.
        self.turn_end_positions.append(len(self.completion_ids) - 1)
        self.turn_rewards.append(0.0)

    def set_last_turn_reward(self, reward: float) -> None:
        if self.turn_rewards:
            self.turn_rewards[-1] = float(reward)

    def add_to_last_turn_reward(self, reward: float) -> None:
        if self.turn_rewards:
            self.turn_rewards[-1] += float(reward)

    def build(self, terminal_reward: float) -> SeatRollout:
        return SeatRollout(
            seat=self.seat,
            prompt_ids=self.prompt_ids,
            completion_ids=self.completion_ids,
            logprobs=self.logprobs,
            owner_mask=self.owner_mask,
            turn_end_positions=self.turn_end_positions,
            turn_rewards=self.turn_rewards,
            terminal_reward=float(terminal_reward),
        )


def play_selfplay_episode(
    env: SelfPlayEnv,
    trainer,
    tokenizer,
    instance_idx: int,
    *,
    seed: int = 0,
    max_turns: int = 100,
) -> Dict[int, SeatRollout]:
    """Play one full self-play episode and return each seat's rollout.

    ``instance_idx`` indexes the game's packaged instance list (see
    ``selfplay_env.list_instance_indices``). All seats are driven by the same
    learner policy (via TRL generation). The returned dict maps seat index ->
    :class:`SeatRollout`; seats that never got a turn (or produced no model
    tokens) are omitted.
    """
    from trl.experimental.openenv import generate_rollout_completions

    env.reset(instance_idx, seed=seed)
    builders: Dict[int, _SeatBuilder] = {
        seat: _SeatBuilder(seat) for seat in range(env.num_players)
    }
    prev_cum: List[float] = [0.0] * env.num_players

    turn = 0
    while not env.done and turn < max_turns:
        seat = env.current_seat
        player = env.current_player
        obs = env.observe()  # {"role": "user", "content": ...}
        builder = builders[seat]

        # Build this seat's full clemcore perspective (user + prior assistant turns).
        perspective = player.perceive_context(obs, log_event=False)
        prompt_text = tokenizer.apply_chat_template(
            perspective, add_generation_prompt=True, tokenize=False
        )

        if builder.first_turn:
            builder.set_prompt(tokenizer.encode(prompt_text, add_special_tokens=False))
        else:
            # Only the *new* context is environment feedback for this seat's stream.
            env_feedback_ids = tokenizer.encode(
                "\n\n" + obs["content"], add_special_tokens=False
            )
            builder.add_env_tokens(env_feedback_ids)

        outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        response = outputs.get("text") or tokenizer.decode(
            outputs["completion_ids"], skip_special_tokens=True
        )
        builder.add_model_tokens(outputs["completion_ids"], outputs["logprobs"])

        # Record the assistant turn into the seat's clemcore memory for its next turn.
        player.perceive_response(response, log_event=False)

        env.step(response)

        # Reward increment attributable to this seat's just-finished turn.
        cum = env.cumulative_rewards
        builder.set_last_turn_reward(cum[seat] - prev_cum[seat])
        prev_cum[seat] = cum[seat]
        turn += 1

    # Reconcile the terminal team reward: any seat whose cumulative reward moved
    # since its last recorded turn (e.g. the non-acting seat at the terminal step)
    # gets that residual folded into its last turn.
    final_cum = env.cumulative_rewards
    for seat, builder in builders.items():
        residual = final_cum[seat] - prev_cum[seat]
        if residual != 0.0 and builder.has_turns:
            builder.add_to_last_turn_reward(residual)

    env.finalize()

    rollouts: Dict[int, SeatRollout] = {}
    for seat, builder in builders.items():
        if not builder.has_turns:
            continue
        rollout = builder.build(terminal_reward=final_cum[seat])
        if rollout.has_model_tokens:
            rollouts[seat] = rollout
    return rollouts

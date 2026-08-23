"""Per-seat rollout collection for MARSHAL-style self-play.

One physical self-play episode is played through :class:`SelfPlayEnv`, and each
seat's turn-alternating trajectory is accumulated into its own :class:`SeatRollout`.
Each turn's observation comes from clemcore's own ``Player.perceive_context`` /
``perceive_response``, i.e. the exact context clemcore uses at inference time, so
there is no train/eval prompt-format drift.

The row a seat accumulates must be *the same token sequence the policy generated
under*, or the log-probs the loss recomputes describe a context that never existed.
Under ``row_context_mode="exact"`` (the default) that holds by construction: turn 1
renders the chat template, every later turn appends the template's own per-turn
scaffolding to the row and generates from *that*, and the environment span is read
back from the token ids the sampler reports. Re-rendering the message history each
turn cannot achieve this, because templates render the same assistant message
differently depending on its position -- which is why the legacy ``"spliced"`` mode
is kept only for replay.

Reasoning ("thinking") models get one extra step: the ``<think>`` block is stripped
from what the *game* and the *conversation history* see, while staying in the trained
token sequence. See :func:`strip_reasoning`.

``trl`` is imported lazily (inside :func:`play_selfplay_episode`) so this module is
importable without the training stack.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from playpen.marshal.selfplay_env import SelfPlayEnv

logger = logging.getLogger(__name__)

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Remove a reasoning-model ``<think>`` block from a response, text-level.

    Clembench parsers are strict about the *start* of an utterance, so a leaked
    ``<think>`` prefix aborts every turn. Only the game and the conversation history
    see the stripped text; the think tokens stay in ``completion_ids`` and keep
    receiving gradient.

    Handles the three shapes that actually occur:

    * a complete ``<think>...</think>`` block (possibly several);
    * a *dangling close tag* -- chat templates that pre-fill ``<think>`` into the
      generation prompt make the completion start already inside the block, so only
      ``</think>`` appears in the output;
    * an *unterminated* block -- the token budget ran out mid-thought, so there is
      no answer at all and the correct result is an empty string (the game will
      then abort this turn, which is the truthful outcome).
    """
    if not text:
        return text
    out = _THINK_BLOCK_RE.sub("", text)
    if THINK_CLOSE in out:  # dangling close tag: keep only what follows the last one
        out = out.rsplit(THINK_CLOSE, 1)[1]
    if THINK_OPEN in out:  # unterminated block: nothing after it is a real answer
        out = out.split(THINK_OPEN, 1)[0]
    return out.strip()


def _strip_reasoning_by_tokens(completion_ids: Sequence[int], tokenizer) -> Optional[str]:
    """Token-level strip: decode only what follows the last ``</think>`` token.

    Preferred over :func:`strip_reasoning` because TRL builds its ``text`` field with
    ``skip_special_tokens=True``. If a tokenizer registers ``</think>`` as a *special*
    token the tag is already gone from that string, and text-level stripping would
    leave the raw reasoning prose in place. Slicing the ids sidesteps that.

    Returns ``None`` when the tokenizer has no ``</think>`` token or the completion
    contains none, so the caller can fall back to the text-level path.
    """
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return None
    try:
        close_id = convert(THINK_CLOSE)
    except Exception:
        return None
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if close_id is None or close_id == unk_id or not isinstance(close_id, int):
        return None
    ids = list(completion_ids)
    if close_id not in ids:
        return None
    last = len(ids) - 1 - ids[::-1].index(close_id)
    return tokenizer.decode(ids[last + 1:], skip_special_tokens=True).strip()


NO_THINK_TAG = "/no_think"
EMPTY_THINK_PREFILL = "<think>\n\n</think>\n\n"


def _with_no_think(messages) -> List[dict]:
    """Copy of ``messages`` with ``/no_think`` appended to the final user message.

    Qwen3's soft switch. Returns a *copy* -- the caller's list holds the live
    observation dict that clemcore also keeps in the player's memory, so mutating it
    would append ``/no_think`` into the recorded conversation and compound it every
    turn. Idempotent: never appends the tag twice.
    """
    msgs = [dict(m) for m in messages]
    for msg in reversed(msgs):
        if msg.get("role") == "user":
            content = str(msg.get("content", ""))
            if NO_THINK_TAG not in content:
                msg["content"] = f"{content}\n\n{NO_THINK_TAG}"
            break
    return msgs


def render_prompt(tokenizer, messages) -> str:
    """Render the chat prompt with reasoning suppressed three ways over, every turn.

    Clembench games want one short, strictly-formatted line per turn. A reasoning
    model instead spends the whole per-turn budget inside ``<think>``, hits
    ``max_completion_length`` mid-thought and never emits an action, so every episode
    aborts. MARSHAL disables thinking for clembench envs for the same reason.

    ``enable_thinking=False`` alone is not sufficient in practice, so all three of
    Qwen3's documented switches are applied together:

    1. ``enable_thinking=False`` -- the template kwarg. Templates that don't define
       it ignore the extra render variable; the ``TypeError`` fallback covers older
       tokenizers that reject unknown kwargs, keeping this safe for non-Qwen models.
    2. ``/no_think`` -- the soft switch, appended to the last user message.
    3. An empty ``<think></think>`` block **prefilled onto the end of the prompt**, so
       generation resumes *after* an already-closed block and the model cannot open
       another. This is the hard guarantee; 1 and 2 are hints the model may ignore.

    The prefill is skipped when the template already emitted a think block of its own,
    which would otherwise leave two stacked blocks. Only the *tail* is inspected, so a
    ``</think>`` from an earlier assistant turn is not mistaken for it.

    The prefill lands in the prompt, never in ``completion_ids``, so it costs no
    trained tokens and does not disturb turn-boundary bookkeeping.
    """
    msgs = _with_no_think(messages)
    try:
        text = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
    if THINK_CLOSE not in text[-64:]:
        text = f"{text}{EMPTY_THINK_PREFILL}"
    return text


# Sentinels for the chat-template probe in :func:`render_turn_scaffold`. Distinctive
# enough that a collision with real game text is itself worth failing on, which is what
# the count checks in that function do.
_PROBE_USER = "<<<marshal-probe-user>>>"
_PROBE_ASSISTANT = "<<<marshal-probe-assistant>>>"


def render_turn_scaffold(tokenizer, obs_content: str) -> str:
    """The chat-template text that separates one assistant turn from the next generation.

    For a Qwen-style template this is::

        <|im_end|>\\n<|im_start|>user\\n{obs}\\n\\n/no_think<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n

    It is *derived from the tokenizer's own template* rather than hard-coded: a probe
    conversation ``[user, assistant, {obs}]`` is rendered through :func:`render_prompt`,
    and everything after the probe assistant's content is the scaffolding. Going through
    ``render_prompt`` means all three of Qwen3's reasoning switches land on the new user
    message exactly as they do on turn 1.

    Exactly one render is taken. Diffing two renders would be wrong for templates that
    render the same assistant message differently depending on its position -- Qwen3
    emits an empty ``<think>`` block before the *last* assistant message only. That
    position-dependence is also why ``"exact"`` mode extends the row rather than
    re-rendering history.

    Raises:
        ValueError: if the template mangles or duplicates the probe, so the caller gets
            a loud failure instead of a silently wrong context.
    """
    messages = [
        {"role": "user", "content": _PROBE_USER},
        {"role": "assistant", "content": _PROBE_ASSISTANT},
        {"role": "user", "content": obs_content},
    ]
    rendered = render_prompt(tokenizer, messages)
    if rendered.count(_PROBE_ASSISTANT) != 1 or rendered.count(_PROBE_USER) != 1:
        raise ValueError(
            "chat template did not render the scaffold probe exactly once; cannot "
            "derive per-turn scaffolding for row_context_mode='exact'"
        )
    cut = rendered.index(_PROBE_ASSISTANT) + len(_PROBE_ASSISTANT)
    return rendered[cut:]


def response_for_game(text: str, completion_ids: Sequence[int], tokenizer) -> str:
    """The utterance the game/history should see: the answer without the reasoning.

    Tries the robust token-level slice first, then the text-level regex. A response
    with no reasoning at all is returned unchanged (modulo surrounding whitespace),
    so this is a no-op for non-thinking models.
    """
    by_tokens = _strip_reasoning_by_tokens(completion_ids, tokenizer)
    if by_tokens is not None:
        return by_tokens
    return strip_reasoning(text)


@dataclass
class SeatRollout:
    """The collected training data for a single seat within one episode.

    ``owner_mask`` / ``env_mask`` are the same thing (1 = model token, 0 =
    environment token); the name ``env_mask`` is what TRL expects as an extra
    field, ``owner_mask`` is what the advantage code reads.

    ``drifted`` distinguishes the two reasons a row can arrive training-inert: the
    seat never got a usable turn (the game ended first), or its row failed
    :meth:`_SeatBuilder.sync_context` and was dropped. Both look identical downstream,
    so without this flag a rising drift rate is indistinguishable from a game that
    ends early. The trainer turns it into the ``marshal/rows/drift_*`` metrics.
    """

    seat: int
    prompt_ids: List[int] = field(default_factory=list)
    completion_ids: List[int] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)
    owner_mask: List[int] = field(default_factory=list)
    turn_end_positions: List[int] = field(default_factory=list)
    turn_rewards: List[float] = field(default_factory=list)
    terminal_reward: float = 0.0
    drifted: bool = False
    # Dense per-turn shaping (``playpen/marshal/turn_rewards.py``), already folded
    # into ``turn_rewards``. Kept as a separate total so the trainer can log it, and
    # so the plain-GRPO path -- which only ever sees one scalar per row -- can add it
    # to the terminal reward without having to re-derive it. 0.0 when the feature is
    # off, which is the default.
    shaping_reward: float = 0.0
    # Whether the per-episode budget had to rescale this seat's shaping (see
    # ``TurnRewardTracker.finalize``). Logged as a calibration signal: a rate near 1
    # means ``turn_reward_scale`` is set too high for the game's turn count.
    shaping_clipped: bool = False
    # Unscaled per-component sums, for the ``marshal/turn_rewards/component/*``
    # metrics. Empty when the feature is off.
    shaping_components: Dict[str, float] = field(default_factory=dict)

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
        # Text of `prompt_ids + completion_ids`, kept alongside the ids so the next
        # turn's generation prompt can be built by *extending the row* instead of
        # re-rendering history (which would not reproduce the row). "exact" mode only.
        self.row_text: str = ""
        # Set when re-tokenization drifted and the owner mask can no longer be proven
        # correct; such a row is dropped rather than trained on a guess.
        self.invalid: bool = False
        self.invalid_logged: bool = False

    @property
    def first_turn(self) -> bool:
        return self._first_turn

    @property
    def has_turns(self) -> bool:
        return len(self.turn_end_positions) > 0

    def set_prompt(self, ids: Sequence[int]) -> None:
        self.prompt_ids = list(ids)
        self._first_turn = False

    def sync_context(self, ctx_ids: Sequence[int]) -> bool:
        """Adopt the token ids vLLM actually conditioned on for this turn.

        ``ctx_ids`` is the sampler's own tokenization of the generation prompt, so it
        is the ground truth for what the policy saw. On the seat's first turn it
        becomes ``prompt_ids``. Afterwards it must extend the row we already hold; the
        extension is exactly this turn's environment-feedback span, recorded with
        ``owner_mask == 0`` so it gets neither gradient nor advantage.

        Returns True when the row matched, False when re-tokenization drifted (the
        model can emit a non-canonical token sequence, which re-encodes differently).
        A drifted row is marked invalid and dropped: a mask we cannot prove is a mask
        that could silently attribute the environment's tokens to the policy.
        """
        ctx = list(ctx_ids)
        if self._first_turn:
            self.prompt_ids = ctx
            self._first_turn = False
            return True
        row = self.prompt_ids + self.completion_ids
        if ctx[: len(row)] == row:
            self.add_env_tokens(ctx[len(row) :])
            return True
        # Keep accumulating so the *generation* context stays right for the rest of the
        # episode (the game still needs this seat to play); only the training row dies.
        self.invalid = True
        keep = min(len(self.prompt_ids), len(ctx))
        self.completion_ids = ctx[keep:]
        self.logprobs = [0.0] * len(self.completion_ids)
        self.owner_mask = [0] * len(self.completion_ids)
        return False

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

    def add_turn_shaping(self, values: Sequence[float]) -> float:
        """Fold a seat's per-turn shaping into its turn rewards; return the total.

        ``values`` is aligned with ``turn_end_positions`` (one entry per turn, in
        order). Anything past this seat's recorded turns is dropped rather than summed
        onto the last one, which would put reward on a turn that did not earn it.
        """
        total = 0.0
        for index, value in enumerate(values):
            if index >= len(self.turn_rewards):
                break
            self.turn_rewards[index] += float(value)
            total += float(value)
        return total

    def build(
        self,
        terminal_reward: float,
        *,
        shaping_reward: float = 0.0,
        shaping_clipped: bool = False,
        shaping_components: Optional[Dict[str, float]] = None,
    ) -> SeatRollout:
        return SeatRollout(
            seat=self.seat,
            prompt_ids=self.prompt_ids,
            completion_ids=self.completion_ids,
            logprobs=self.logprobs,
            owner_mask=self.owner_mask,
            turn_end_positions=self.turn_end_positions,
            turn_rewards=self.turn_rewards,
            terminal_reward=float(terminal_reward),
            shaping_reward=float(shaping_reward),
            shaping_clipped=bool(shaping_clipped),
            shaping_components=dict(shaping_components or {}),
        )


def _turn_context(env: SelfPlayEnv, seat: int, turn_index: int, response: str):
    """Bundle what a turn-reward extractor may inspect for one completed turn.

    Everything is read through ``getattr`` with a default: the scripted envs in the
    test suite implement only what a rollout needs (``step``/``observe``/rewards) and
    carry no game master at all, and they must keep working unchanged.
    """
    from playpen.marshal.turn_rewards import TurnContext

    game_state = getattr(env, "game_state", None)
    info_for_seat = getattr(env, "info_for_seat", None)
    return TurnContext(
        seat=seat,
        turn_index=turn_index,
        state=game_state,
        info=info_for_seat(seat) if callable(info_for_seat) else {},
        response=response,
        done=bool(getattr(env, "done", False)),
    )


def play_selfplay_episode(
    env: SelfPlayEnv,
    trainer,
    tokenizer,
    instance_idx: int,
    *,
    seed: int = 0,
    max_turns: int = 100,
    strip_think: bool = True,
    row_context_mode: str = "exact",
    turn_reward_tracker=None,
) -> Dict[int, SeatRollout]:
    """Play one full self-play episode and return each seat's rollout.

    ``instance_idx`` indexes the game's packaged instance list (see
    ``selfplay_env.list_instance_indices``). All seats are driven by the same
    learner policy (via TRL generation). The returned dict maps **every** seat index to
    a :class:`SeatRollout`. A seat that never got a turn, generated nothing, or whose
    row drifted gets a training-inert placeholder (``owner_mask == [0]``: no gradient,
    and excluded from the advantage pools) that still carries the episode's real
    terminal reward, so reward statistics stay honest for seats that could not play.

    ``row_context_mode`` (see ``MarshalConfig.row_context_mode``) selects how a row is
    assembled across turns:

    * ``"exact"`` -- the generation prompt for turn *k* is **the row so far** plus the
      template's own per-turn scaffolding (:func:`render_turn_scaffold`), and the
      environment span is read back from the token ids vLLM reports for that prompt
      (``_SeatBuilder.sync_context``). The row therefore *is* the context, by
      construction, and the recomputed log-probs the loss uses match the sampler's.
    * ``"spliced"`` -- legacy: generate from a freshly re-rendered chat template while
      the row splices in raw environment text. Reproduces pre-fix runs; the trained
      sequence does not match the generated one from turn 2 onward.

    ``turn_reward_tracker`` (``None`` = off, the default) is a
    ``playpen.marshal.turn_rewards.TurnRewardTracker``. When given, it is asked for a
    per-turn signal after every step, and at episode end its scaled, budget-capped
    values are added to each seat's ``turn_rewards`` -- on top of the env's terminal
    reward, never instead of it. It reads the game's live state through
    ``SelfPlayEnv.game_state``, so an env that does not expose one produces no
    shaping.

    ``strip_think`` (default on, and a no-op for non-reasoning models) removes the
    ``<think>`` block from the utterance handed to the game and to clemcore's
    conversation memory, while leaving those tokens in ``completion_ids`` so they
    are still trained. Turn it off only to reproduce the un-stripped behavior.

    One behavioral consequence of ``"exact"``: later turns are conditioned on the
    seat's *raw* generations, because the row is the context and the row is what is
    trained on. The game and clemcore's transcript still see the stripped utterance.
    """
    from trl.experimental.openenv import generate_rollout_completions

    exact = row_context_mode == "exact"
    eos_text = getattr(tokenizer, "eos_token", None) or ""

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
        # Always called: clemcore memorizes the context and games/transcripts rely on it.
        perspective = player.perceive_context(obs, log_event=False)

        if builder.first_turn:
            # Turn 1 is identical in both modes: the rendered chat template *is* the row.
            prompt_text = render_prompt(tokenizer, perspective)
        elif exact:
            # Extend the row instead of re-rendering history, so the context the policy
            # conditions on and the sequence the loss scores are the same object.
            scaffold = render_turn_scaffold(tokenizer, obs["content"])
            # The template closes an assistant turn with the EOS text, and vLLM already
            # returned that token at the end of the previous generation; don't double it.
            if eos_text and builder.row_text.endswith(eos_text) and scaffold.startswith(eos_text):
                scaffold = scaffold[len(eos_text) :]
            prompt_text = builder.row_text + scaffold
        else:
            prompt_text = render_prompt(tokenizer, perspective)

        outputs = generate_rollout_completions(trainer, [prompt_text])[0]

        if exact:
            if not builder.sync_context(outputs["prompt_ids"]) and not builder.invalid_logged:
                builder.invalid_logged = True
                logger.warning(
                    "seat %d: re-tokenization drift at turn %d; dropping this row from "
                    "training (its owner mask can no longer be proven correct)",
                    seat,
                    turn,
                )
        elif builder.first_turn:
            builder.set_prompt(tokenizer.encode(prompt_text, add_special_tokens=False))
        else:
            # Only the *new* context is environment feedback for this seat's stream.
            env_feedback_ids = tokenizer.encode(
                "\n\n" + obs["content"], add_special_tokens=False
            )
            builder.add_env_tokens(env_feedback_ids)

        response = outputs.get("text") or tokenizer.decode(
            outputs["completion_ids"], skip_special_tokens=True
        )
        # The full generation (reasoning included) is what is trained on, so <think>
        # tokens stay under the loss mask and inside this turn's advantage span.
        builder.add_model_tokens(outputs["completion_ids"], outputs["logprobs"])
        if exact:
            builder.row_text = prompt_text + tokenizer.decode(
                outputs["completion_ids"], skip_special_tokens=False
            )

        # The game and the conversation history, by contrast, must see only the answer:
        # clembench parsers match on the start of the utterance, so a leaked <think>
        # prefix aborts the episode. Stripping before perceive_response also keeps the
        # think blocks out of later turns' prompts (matching Qwen3's own multi-turn
        # convention) instead of compounding them into the context.
        game_response = (
            response_for_game(response, outputs["completion_ids"], tokenizer)
            if strip_think
            else response
        )

        # Record the assistant turn into the seat's clemcore memory for its next turn.
        player.perceive_response(game_response, log_event=False)

        env.step(game_response)

        # Reward increment attributable to this seat's just-finished turn.
        cum = env.cumulative_rewards
        builder.set_last_turn_reward(cum[seat] - prev_cum[seat])
        prev_cum[seat] = cum[seat]

        # Dense per-turn signal, collected raw here and scaled/capped at episode end
        # (a running cap would make early turns worth more than late ones). Recorded
        # against this seat's own turn index so it lands on the boundary it earned.
        if turn_reward_tracker is not None:
            turn_reward_tracker.on_turn(
                _turn_context(env, seat, len(builder.turn_rewards) - 1, game_response)
            )

        turn += 1

    # Reconcile the terminal team reward: any seat whose cumulative reward moved
    # since its last recorded turn (e.g. the non-acting seat at the terminal step)
    # gets that residual folded into its last turn.
    final_cum = env.cumulative_rewards
    for seat, builder in builders.items():
        residual = final_cum[seat] - prev_cum[seat]
        if residual != 0.0 and builder.has_turns:
            builder.add_to_last_turn_reward(residual)

    # Scale + budget-cap the episode's shaping and fold it into the turn rewards.
    # After the terminal reconciliation above, so the two channels compose rather
    # than one overwriting the other.
    shaping: Dict[int, List[float]] = (
        turn_reward_tracker.finalize() if turn_reward_tracker is not None else {}
    )
    shaping_totals: Dict[int, float] = {}
    for seat, values in shaping.items():
        builder = builders.get(seat)
        if builder is None:
            continue
        shaping_totals[seat] = builder.add_turn_shaping(values)

    env.finalize()

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None) or 0

    rollouts: Dict[int, SeatRollout] = {}
    for seat, builder in builders.items():
        rollout = None
        if builder.has_turns and not builder.invalid:
            rollout = builder.build(
                terminal_reward=final_cum[seat],
                shaping_reward=shaping_totals.get(seat, 0.0),
                shaping_clipped=(
                    turn_reward_tracker is not None and turn_reward_tracker.was_clipped(seat)
                ),
                shaping_components=(
                    turn_reward_tracker.component_totals(seat)
                    if turn_reward_tracker is not None
                    else None
                ),
            )
        if rollout is not None and rollout.has_model_tokens:
            rollouts[seat] = rollout
        else:
            # This seat has no trainable trajectory -- it never got a turn (the game
            # ended first), it generated nothing, or its row drifted. Emit a
            # training-inert placeholder rather than omitting it, so the seat still
            # reports the episode's actual outcome; reporting 0.0 would be a
            # fabricated FAILURE in the reward statistics.
            rollouts[seat] = SeatRollout(
                seat=seat,
                prompt_ids=[int(pad_id)],
                completion_ids=[int(pad_id)],
                logprobs=[0.0],
                owner_mask=[0],          # no gradient, and excluded from advantage pools
                turn_end_positions=[],
                turn_rewards=[],
                terminal_reward=float(final_cum[seat]),
                # Only drift is a *censoring* event -- this seat played and its data
                # was thrown away. A seat that never moved is not censored, so the
                # two must not share a counter.
                drifted=bool(builder.invalid),
            )
    return rollouts

"""Dense per-turn rewards read live off a clembench game's own state.

Playpen's rollout collection is *terminal-only* by default: clemcore's PettingZoo
env scores an episode with ``_default_reward`` (SUCCESS ``+1`` / FAILURE ``0`` /
ABORTED ``-1``, and ``0`` at every non-terminal step), and that single team scalar
is what every seat's last turn receives. This module is the opt-in second reward
channel: a small, bounded, per-turn signal derived from what the game already
knows at each step.

WHY THIS EXISTS (two independent reasons)
-----------------------------------------
1. **Credit assignment across turns.** MARSHAL's turn-level estimator
   (``advantage.reinforce_returns``) is built to attribute reward at each turn
   boundary, but with a terminal-only reward every turn of a row carries the same
   return, so the estimator has nothing to separate. A dense signal is what makes
   ``turn_level_rewards: true`` mean anything.
2. **Credit assignment across seats.** clembench hands both self-play seats the
   *same* team outcome, so per-seat advantage pooling
   (``agent_specific_normalization``) currently mean-centers two identical
   distributions -- both seats end up with identical advantages and the seat split
   is a metric artifact. A per-turn reward is attributed to *the seat that acted*,
   which is the only thing in this pipeline that can make the two seats' advantages
   genuinely differ.

WHERE THE SIGNAL COMES FROM
---------------------------
clemcore exposes a ``reward_func(observation, action, state, info) -> float`` hook
on its PettingZoo env, and its docstring points at exactly this: "Game-specific
rewards can be implemented by subclassing GameState to carry additional fields
(e.g. letter matches in Wordle) and reading them in a custom reward_func." No
shipped clembench game supplies one (and none populate ``info``), but most of them
*do* subclass ``GameState`` with the per-turn fields their own offline scorers use.
So the extractors below read that live state after each step.

We deliberately do NOT install a clemcore ``reward_func``. That hook is a single
channel: on the terminal step clemcore calls it again for *every* agent and
overwrites the acting agent's value, which would mix per-turn shaping into the team
outcome and make the two inseparable. Keeping this channel beside it leaves the
terminal reward -- and therefore the plain-GRPO baseline, the ``reward`` metric and
all existing runs -- bit-for-bit unchanged.

SCALING: WHY THIS CANNOT OVERWHELM THE TERMINAL REWARD
------------------------------------------------------
Every extractor returns components already normalized to ``[-1, 1]``. Per seat, per
episode:

1. this turn's components are summed and clipped to ``[-1, 1]``  -> ``raw_t``
2. ``scaled_t = scale * raw_t``                                   (``scale`` ~ 0.05)
3. if ``budget > 0`` and ``|sum_t scaled_t| > budget``, every turn is multiplied by
   ``budget / |sum_t scaled_t|``  -- a proportional rescale, so relative credit
   between turns is preserved and no sign is ever flipped.

The invariant after step 3 is ``|sum of an episode's shaping for one seat| <=
budget``. Because the MARSHAL return is a backward cumulative *sum*, that episode
total is precisely what competes with the ``+1/0/-1`` outcome. With
``budget < 0.5`` the worst-case swing between two episodes is ``2 * budget < 1.0``,
which is smaller than the gap between any two distinct clembench outcomes
(SUCCESS-FAILURE = 1, FAILURE-ABORTED = 1). **Shaping can therefore never reorder
two episodes that ended differently** -- it only ranks episodes *within* an outcome
class. ``MarshalConfig`` warns when ``turn_reward_budget >= 0.5``, where that
guarantee stops holding.

The budget is a safety cap, not the operating point: pick ``scale`` so the typical
episode lands under it, and watch ``marshal/turn_rewards/budget_clip_rate`` (a rate
near 1.0 means ``scale`` is too large and only the *shape* of your signal survives).

Depends only on the standard library -- no torch, no trl, no clemcore -- so it is
unit-testable in isolation, and every read of game state is defensive: a clembench
update that renames a field degrades this to "no shaping" plus one warning, never a
crash mid-run.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Accepted values for ``MarshalConfig.turn_reward_source``.
TURN_REWARD_SOURCES = ("auto", "game", "generic")


@dataclass
class TurnContext:
    """Everything an extractor may look at for one completed turn.

    Attributes:
        seat: which self-play seat just acted (0-based, as in ``player_{n}``).
        turn_index: that seat's own turn counter within this episode (0-based),
            aligned with ``SeatRollout.turn_end_positions``.
        state: the game's live ``GameState`` (usually a game-specific subclass), or
            ``None`` when the env cannot supply one (e.g. a stubbed test env).
        info: the ``info`` dict clemcore's env recorded for this seat's step. Empty
            for every shipped clembench game -- none of them populate it -- but read
            here so a game that starts to will work without a code change.
        response: the utterance the game actually saw (reasoning already stripped).
        done: whether the episode ended on this step.
    """

    seat: int
    turn_index: int
    state: Any = None
    info: Dict[str, Any] = field(default_factory=dict)
    response: str = ""
    done: bool = False


@dataclass(frozen=True)
class TurnRewardSpec:
    """Scaling parameters shared by every extractor. See the module docstring.

    Attributes:
        scale: multiplier applied to each turn's combined (``[-1, 1]``) signal.
        budget: hard cap on ``|sum of one seat's shaping over one episode|``.
            ``0`` disables the cap -- which also gives up the "cannot reorder two
            outcomes" guarantee, so only do it deliberately.
        components: allowlist of component names; empty means "all of them".
    """

    scale: float = 0.05
    budget: float = 0.3
    components: Tuple[str, ...] = ()


# --------------------------------------------------------------------------
# Extractors
# --------------------------------------------------------------------------


class TurnRewardExtractor:
    """Maps one completed turn onto named components, each in ``[-1, 1]``.

    Subclasses implement :meth:`extract` and declare :attr:`components`. One
    instance serves a whole episode across all seats (game state such as a
    codenames board or a wordle best-closeness is episode-scoped, not seat-scoped),
    and :meth:`reset` is called at the start of each episode.

    Components are *named* rather than pre-summed so a run can allowlist a subset
    (``turn_reward_components``) and so each one is logged separately -- a shaping
    term you cannot see the magnitude of is a shaping term you cannot calibrate.
    """

    #: Registry key(s) this extractor serves; matched by exact name then by prefix.
    game: str = ""
    #: Names of the components :meth:`extract` may return.
    components: Tuple[str, ...] = ()

    def reset(self) -> None:
        """Start a new episode. Default: nothing to forget."""

    def extract(self, ctx: TurnContext) -> Dict[str, float]:
        """Components for this turn. Omitted names count as 0.0."""
        raise NotImplementedError

    # -- helpers for subclasses -------------------------------------------

    @staticmethod
    def _flag(state: Any, name: str, default: bool = False) -> bool:
        value = getattr(state, name, None)
        return default if value is None else bool(value)

    @staticmethod
    def _outcome_is(state: Any, wanted: str) -> bool:
        """Whether ``state.outcome`` is clemcore's ``Outcome.<wanted>``.

        ``Outcome`` is a ``str, Enum`` whose ``.name`` is upper-case (``ABORTED``)
        and whose ``.value`` is lower-case (``aborted``); both are compared so this
        keeps working if a game stores the raw value instead of the member.
        """
        outcome = getattr(state, "outcome", None)
        if outcome is None:
            return False
        wanted = wanted.lower()
        name = str(getattr(outcome, "name", "")).lower()
        value = str(getattr(outcome, "value", outcome)).lower()
        return wanted in (name, value)


class _EdgeDetector:
    """Remembers a flag's previous value so a *rising edge* can be detected.

    clembench games are inconsistent about resetting their validation flags:
    codenames clears ``invalid_response`` at the top of every validation, while
    taboo and guesswhat set theirs once and leave them set for the rest of the
    episode. Charging a level would therefore penalize every turn after the first
    violation in those games. Charging the 0->1 transition penalizes the turn that
    actually caused it, in both styles.
    """

    def __init__(self) -> None:
        self._previous: Dict[str, bool] = {}

    def reset(self) -> None:
        self._previous = {}

    def rose(self, state: Any, name: str) -> bool:
        """Whether ``state.<name>`` is truthy now and was not on the previous turn."""
        return self.rose_value(name, bool(getattr(state, name, False)))

    def rose_value(self, name: str, value: bool) -> bool:
        """:meth:`rose` for a flag that is derived rather than read (taboo's ``clue_error``)."""
        was = self._previous.get(name, False)
        self._previous[name] = value
        return value and not was


class FormatComplianceExtractor(TurnRewardExtractor):
    """Game-agnostic fallback: charge the turn whose response failed validation.

    Reads the validation flags clembench games conventionally carry on their
    ``GameState`` subclass. Negative flags (``invalid_response``,
    ``invalid_format``, ``invalid_content``) are charged on their rising edge (see
    :class:`_EdgeDetector`); the positive ``valid_response`` flag is charged on its
    level, because games carrying it (wordle) re-set it on every single turn.

    This is *approximate by construction* -- it infers a per-turn signal from flags
    that were written for offline scoring, and a game that carries none of them
    yields no shaping at all (which is reported at build time, not silently). A
    game-specific extractor, where one is registered, is always preferred.

    Note what this buys even though a format violation usually also ends the
    episode: the terminal ``-1`` is a *team* reward that both seats receive, so it
    cannot say which seat broke the format. This component can, and it lands on the
    exact turn that did it.
    """

    game = "generic"
    components = ("format",)

    _NEGATIVE_FLAGS = ("invalid_response", "invalid_format", "invalid_content")
    _POSITIVE_FLAGS = ("valid_response",)

    def __init__(self) -> None:
        self._edges = _EdgeDetector()

    def reset(self) -> None:
        self._edges.reset()

    def supported_by(self, state: Any) -> bool:
        """Whether this state carries any flag we know how to read."""
        names = self._NEGATIVE_FLAGS + self._POSITIVE_FLAGS
        return any(hasattr(state, name) for name in names)

    def extract(self, ctx: TurnContext) -> Dict[str, float]:
        state = ctx.state
        violated = any(self._edges.rose(state, name) for name in self._NEGATIVE_FLAGS)
        for name in self._POSITIVE_FLAGS:
            if hasattr(state, name) and not self._flag(state, name, default=True):
                violated = True
        return {"format": -1.0} if violated else {}


_WORDLE_TOKEN_RE = re.compile(r"<(green|yellow|red)>")

# clembench's own weights (wordle/utils/compute_metrics.py: turns_closeness).
_WORDLE_GREEN = 5.0
_WORDLE_YELLOW = 3.0


def wordle_closeness(feedback: str) -> float:
    """clembench's wordle closeness score, normalized to ``[0, 1]``.

    ``feedback`` is the string clemcore's ``GuessValidator`` produces, e.g.
    ``"c<red> r<red> e<red> e<red> k<green>"``. clembench's ``turns_closeness``
    scores a green 5 and a yellow 3; the maximum is therefore ``5 * n_letters``
    (25 for a standard 5-letter game), which is what we divide by so the value is
    comparable across word lengths and already inside the extractor contract.
    """
    marks = _WORDLE_TOKEN_RE.findall(feedback or "")
    if not marks:
        return 0.0
    score = sum(
        _WORDLE_GREEN if mark == "green" else _WORDLE_YELLOW if mark == "yellow" else 0.0
        for mark in marks
    )
    return score / (_WORDLE_GREEN * len(marks))


class WordleTurnRewards(TurnRewardExtractor):
    """wordle: reward a guess that reveals more of the target than any before it.

    ``closeness`` is *potential-based*: the component is the gain over the best
    closeness seen so far this episode, so the per-episode sum telescopes to
    ``best_final - 0`` and is bounded by 1 before scaling. Two consequences worth
    knowing: repeating a guess earns exactly 0 (no reward for treading water), and
    a worse guess is never punished (exploration stays free) -- the terminal reward
    is what punishes not solving the word.

    ``format`` charges a guess the game rejected. Wordle re-prompts up to
    ``max_retry_per_error`` times before aborting, so these turns genuinely exist
    inside an episode rather than only at its end.
    """

    game = "wordle"
    components = ("closeness", "format")

    def __init__(self) -> None:
        self._best = 0.0

    def reset(self) -> None:
        self._best = 0.0

    def extract(self, ctx: TurnContext) -> Dict[str, float]:
        state = ctx.state
        # valid_response is re-set on every turn by wordle's validator, so its level
        # (not an edge) is the truth about *this* turn.
        if not self._flag(state, "valid_response", default=True):
            return {"format": -1.0}
        feedback = getattr(state, "guess_feedback", None)
        if not feedback:
            return {}
        closeness = wordle_closeness(feedback)
        gain = max(0.0, closeness - self._best)
        self._best = max(self._best, closeness)
        return {"closeness": gain} if gain else {}


class TabooTurnRewards(TurnRewardExtractor):
    """taboo: charge the seat whose turn broke a rule.

    Taboo has no progress signal to speak of -- the guesser either says the word or
    does not -- so both components are compliance:

    * ``format`` -- the response did not start with ``CLUE:``/``GUESS:``
      (``state.invalid_response``), which aborts the episode.
    * ``clue`` -- the describer's clue contained the target or a related word
      (``state.clue_error``), which loses it.

    Both already end the episode, so the *episode's* outcome is not news. What is
    news is **which seat and which turn**: the terminal reward is shared, and
    without this the describer's illegal clue and the guesser's wrong guess are
    indistinguishable in either seat's advantage.
    """

    game = "taboo"
    components = ("format", "clue")

    def __init__(self) -> None:
        self._edges = _EdgeDetector()

    def reset(self) -> None:
        self._edges.reset()

    def extract(self, ctx: TurnContext) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if self._edges.rose(ctx.state, "invalid_response"):
            out["format"] = -1.0
        # clue_error holds None or an error dict, so the edge is None -> non-None.
        has_clue_error = getattr(ctx.state, "clue_error", None) is not None
        if self._edges.rose_value("clue_error", has_clue_error):
            out["clue"] = -1.0
        return out


class GuessWhatTurnRewards(TurnRewardExtractor):
    """guesswhat: charge a malformed question/answer or a disallowed one.

    ``state.invalid_format`` covers a response that is neither a valid
    ``QUESTION:`` nor a valid ``GUESS:``; ``state.invalid_content`` covers a
    question the game forbids (letter/position probes). Both abort the episode,
    and both are charged to the seat and turn that produced them -- the same
    per-seat-blame argument as taboo.

    guesswhat has no partial-progress signal available live: whether a question was
    *informative* is only knowable against the candidate list the answerer holds,
    which the game does not score.
    """

    game = "guesswhat"
    components = ("format",)

    def __init__(self) -> None:
        self._edges = _EdgeDetector()

    def reset(self) -> None:
        self._edges.reset()

    def extract(self, ctx: TurnContext) -> Dict[str, float]:
        rose_format = self._edges.rose(ctx.state, "invalid_format")
        rose_content = self._edges.rose(ctx.state, "invalid_content")
        return {"format": -1.0} if (rose_format or rose_content) else {}


# Board-assignment keys, from clembench/codenames/constants.py.
_CN_TEAM, _CN_OPPONENT, _CN_INNOCENT, _CN_ASSASSIN = "team", "opponent", "innocent", "assassin"

#: Value of revealing one word of each assignment, before normalization by the
#: board's team-word count. Ordered by how badly the reveal ends the game:
#: an assassin loses it outright, an opponent word hands them a word, an innocent
#: only ends the turn.
_CN_WEIGHTS = {_CN_TEAM: 1.0, _CN_INNOCENT: -0.5, _CN_OPPONENT: -1.0, _CN_ASSASSIN: -2.0}


class CodenamesTurnRewards(TurnRewardExtractor):
    """codenames: reward board progress, charge a malformed clue/guess.

    ``board`` is the only genuinely *dense* progress signal among the shipped
    2-player games: each guesser turn reveals words, and the board records what
    each one was. The component is the weighted count of words **our team** revealed
    this turn (:data:`_CN_WEIGHTS`) divided by the board's initial team-word count,
    clipped to ``[-1, 1]``. Revealing every team word therefore scores exactly +1
    across the episode, which is the scale the other extractors work on.

    It reads ``board.revealed["team"]`` rather than the drop in ``board.hidden``,
    and that distinction is the whole correctness of this component: codenames
    simulates the *opposing* team between rounds (``_on_before_round`` ->
    ``_opponent_turn``), which reveals opponent cards and shrinks ``hidden`` without
    our policy having done anything. Diffing ``hidden`` therefore charged the policy
    for the simulated opponent's moves -- measured on the packaged instances, the
    mock guesser's ``+1`` team word and the opponent's ``-1`` card cancelled exactly,
    so the component read a flat 0.0 every single turn and looked simply unwired.
    ``revealed[by=TEAM]`` only ever grows through ``reveal_word(word, TEAM)``, i.e.
    our own guesser.

    The reward is attributed to whichever seat acted, which is the guesser -- the
    cluegiver is rewarded indirectly, through the guesser's success on its clue,
    exactly as the team outcome already does it. Measured on the packaged instances
    with clembench's own mock players, that means the cluegiver seat receives only
    the ``format`` component (the guesser collected ``+0.89`` of board over an
    episode; the cluegiver, ``0``). A direct cluegiver signal is available if that
    asymmetry matters for a result -- ``ClueGiver.targets`` with
    ``board.get_word_assignment`` gives the fraction of a clue's targets that are
    actually team words -- but it lives on the *player*, not the game state, so it is
    deliberately left out here rather than widening what an extractor may reach into.

    ``format`` reads ``state.invalid_response`` as a *level*: codenames clears it
    at the top of every validation, so its value after a step describes that step.
    """

    game = "codenames"
    components = ("board", "format")

    def __init__(self) -> None:
        self._revealed: Optional[Dict[str, int]] = None
        self._team_total = 1

    def reset(self) -> None:
        self._revealed = None
        self._team_total = 1

    @staticmethod
    def _team_reveals(state: Any):
        """``({assignment: count revealed by our team}, initial team-word count)``.

        ``(None, None)`` when the state does not look like a codenames board, so a
        clembench change degrades to "no board component" rather than a wrong one.
        """
        board = getattr(state, "board", None)
        revealed = getattr(board, "revealed", None)
        if not isinstance(revealed, dict):
            return None, None
        by_team = revealed.get(_CN_TEAM)
        if not isinstance(by_team, dict):
            return None, None
        counts = {key: len(value) for key, value in by_team.items()}
        hidden = getattr(board, "hidden", None)
        # Still-hidden team words plus the ones we have already taken == the board's
        # original team count, whenever we first look at it.
        total = None
        if isinstance(hidden, dict):
            total = len(hidden.get(_CN_TEAM, ())) + counts.get(_CN_TEAM, 0)
        return counts, total

    def extract(self, ctx: TurnContext) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if self._flag(ctx.state, "invalid_response"):
            out["format"] = -1.0

        counts, total = self._team_reveals(ctx.state)
        if counts is None:
            return out
        if self._revealed is None:
            # First look at this episode's board: nothing to diff against yet, and
            # this is where the normalizer comes from.
            self._revealed = counts
            self._team_total = max(1, total or 0)
            return out

        gained = {key: counts.get(key, 0) - self._revealed.get(key, 0) for key in counts}
        self._revealed = counts
        value = sum(_CN_WEIGHTS.get(key, 0.0) * count for key, count in gained.items())
        value /= self._team_total
        if value:
            out["board"] = max(-1.0, min(1.0, value))
        return out


class DondTurnRewards(TurnRewardExtractor):
    """dond: reward committing a well-formed proposal, charge an aborted turn.

    Deal-or-no-deal's negotiation messages are free text the game does not score,
    so the only per-turn events it exposes are (a) a seat's secret proposal being
    accepted into the state and (b) a parse/rule error, which aborts.

    ``proposal`` fires once, on the turn a seat's own ``player_{a,b}_proposal``
    goes from ``None`` to a parsed split. It rewards *making a valid proposal*, not
    its value -- how good the split is is exactly what the terminal reward scores,
    and duplicating that here would be the kind of shaping that competes with the
    outcome instead of guiding toward it.
    """

    game = "dond"
    components = ("proposal", "format")

    _PROPOSAL_FIELDS = ("player_a_proposal", "player_b_proposal")

    def __init__(self) -> None:
        self._seen = [False, False]
        self._aborted = False

    def reset(self) -> None:
        self._seen = [False, False]
        self._aborted = False

    def extract(self, ctx: TurnContext) -> Dict[str, float]:
        out: Dict[str, float] = {}

        # dond aborts on a parse error and on a rule violation; the transition into
        # ABORTED on this step is what identifies the offending turn.
        aborted = self._outcome_is(ctx.state, "aborted")
        if aborted and not self._aborted:
            out["format"] = -1.0
        self._aborted = aborted

        if 0 <= ctx.seat < len(self._PROPOSAL_FIELDS):
            made = getattr(ctx.state, self._PROPOSAL_FIELDS[ctx.seat], None) is not None
            if made and not self._seen[ctx.seat]:
                out["proposal"] = 1.0
            self._seen[ctx.seat] = made
        return out


#: Game-specific extractors, keyed by clembench game name. Lookup is exact first,
#: then by longest matching prefix, so variants (``wordle_withclue``,
#: ``wordle_withcritic``) resolve to their base game's extractor.
GAME_EXTRACTORS: Dict[str, type] = {
    "wordle": WordleTurnRewards,
    "taboo": TabooTurnRewards,
    "guesswhat": GuessWhatTurnRewards,
    "codenames": CodenamesTurnRewards,
    "dond": DondTurnRewards,
}


def resolve_extractor_class(game_name: str) -> Optional[type]:
    """The registered extractor for ``game_name``, or ``None``.

    Exact match wins; otherwise the longest registered key that ``game_name``
    starts with, so ``wordle_withclue`` resolves to :class:`WordleTurnRewards`
    while an unrelated game does not accidentally match a short key.
    """
    name = (game_name or "").strip().lower()
    if name in GAME_EXTRACTORS:
        return GAME_EXTRACTORS[name]
    candidates = [key for key in GAME_EXTRACTORS if name.startswith(key)]
    if not candidates:
        return None
    return GAME_EXTRACTORS[max(candidates, key=len)]


def build_extractor(game_name: str, source: str = "auto") -> Optional[TurnRewardExtractor]:
    """Instantiate the extractor for ``game_name`` under ``source``.

    * ``"auto"`` -- the game's own extractor if one is registered, else the generic
      format-compliance fallback.
    * ``"game"`` -- the game's own extractor only; returns ``None`` (and warns) when
      the game has none, so a run cannot silently train on a fallback signal it did
      not ask for.
    * ``"generic"`` -- always the format-compliance fallback, even for a game that
      has a richer extractor. Useful as an ablation isolating "compliance only".
    """
    if source not in TURN_REWARD_SOURCES:
        raise ValueError(f"turn_reward_source must be one of {TURN_REWARD_SOURCES}, got {source!r}")
    if source == "generic":
        return FormatComplianceExtractor()
    cls = resolve_extractor_class(game_name)
    if cls is not None:
        return cls()
    if source == "game":
        logger.warning(
            "turn_reward_source='game' but no game-specific turn-reward extractor is "
            "registered for %r (have: %s). Turn rewards are OFF for this run; pass "
            "turn_reward_source='auto' to fall back to generic format compliance.",
            game_name,
            ", ".join(sorted(GAME_EXTRACTORS)),
        )
        return None
    return FormatComplianceExtractor()


def resolve_turn_reward_extractor(game_name: str, config: Any):
    """``(extractor, spec)`` for a run, or ``(None, None)`` when turn rewards are off.

    Takes the whole ``MarshalConfig`` rather than its pieces so there is exactly one
    place that decides what a config means for this feature -- the launch script
    prints from it and the rollout builder constructs from it, and the two can never
    disagree about which extractor a run is using. (Typed as ``Any`` so this module
    keeps its stdlib-only imports; only ``turn_reward_kwargs()`` and
    ``turn_reward_source`` are touched.)

    ``turn_reward_components`` is validated here, against the *resolved* extractor:
    an unknown name raises rather than quietly filtering everything out, which would
    be indistinguishable from the feature not being wired up at all.
    """
    kwargs = config.turn_reward_kwargs()
    if kwargs is None:
        return None, None
    extractor = build_extractor(game_name, config.turn_reward_source)
    if extractor is None:
        return None, None
    unknown = [name for name in kwargs["components"] if name not in extractor.components]
    if unknown:
        raise ValueError(
            f"Unknown turn_reward_components {unknown} for game {game_name!r} "
            f"(extractor {type(extractor).__name__} emits: {list(extractor.components)})."
        )
    return extractor, TurnRewardSpec(**kwargs)


# --------------------------------------------------------------------------
# Per-episode accumulation + scaling
# --------------------------------------------------------------------------


class TurnRewardTracker:
    """Accumulates one episode's per-turn shaping and applies the scale + budget.

    One tracker per episode. :meth:`on_turn` is called once per acting turn, in the
    same order the seat's turn boundaries are appended, so
    ``finalize()[seat][i]`` lines up with that seat's ``turn_rewards[i]``.

    The budget is applied in :meth:`finalize` rather than as a running clip on
    purpose: a running clip would make shaping worth more early in an episode than
    late (spend-it-first), whereas a proportional rescale of the finished vector
    preserves the relative credit the extractor assigned. Nothing downstream needs
    the values before the episode ends -- returns are computed post-hoc -- so there
    is no cost to waiting.
    """

    def __init__(self, extractor: TurnRewardExtractor, spec: TurnRewardSpec) -> None:
        self.extractor = extractor
        self.spec = spec
        self._allowed = set(spec.components) if spec.components else None
        self._raw: Dict[int, List[float]] = {}
        self._components: Dict[int, Dict[str, float]] = {}
        self._clipped: Dict[int, bool] = {}
        self.extractor.reset()

    # -- collection -------------------------------------------------------

    def on_turn(self, ctx: TurnContext) -> float:
        """Record (and return) the unscaled, clipped signal for one turn."""
        try:
            components = self.extractor.extract(ctx) or {}
        except Exception:  # a game-state change must never kill a training run
            logger.warning(
                "turn-reward extractor %s raised on seat %d turn %d; scoring this turn 0.0",
                type(self.extractor).__name__,
                ctx.seat,
                ctx.turn_index,
                exc_info=True,
            )
            components = {}

        total = 0.0
        seat_components = self._components.setdefault(ctx.seat, {})
        for name, value in components.items():
            if self._allowed is not None and name not in self._allowed:
                continue
            value = max(-1.0, min(1.0, float(value)))
            seat_components[name] = seat_components.get(name, 0.0) + value
            total += value

        total = max(-1.0, min(1.0, total))
        row = self._raw.setdefault(ctx.seat, [])
        # Turns are recorded in order, but pad defensively so an extractor called
        # out of step can never silently shift a later turn's reward onto an
        # earlier boundary.
        while len(row) < ctx.turn_index:
            row.append(0.0)
        if ctx.turn_index < len(row):
            row[ctx.turn_index] += total
        else:
            row.append(total)
        return total

    # -- results ----------------------------------------------------------

    def finalize(self) -> Dict[int, List[float]]:
        """Scaled, budget-capped shaping per seat, aligned with its turn boundaries."""
        out: Dict[int, List[float]] = {}
        for seat, row in self._raw.items():
            scaled = [self.spec.scale * value for value in row]
            total = sum(scaled)
            magnitude = abs(total)
            clipped = self.spec.budget > 0.0 and magnitude > self.spec.budget
            if clipped:
                factor = self.spec.budget / magnitude
                scaled = [value * factor for value in scaled]
            self._clipped[seat] = clipped
            out[seat] = scaled
        return out

    def was_clipped(self, seat: int) -> bool:
        """Whether the budget bound this seat (only meaningful after :meth:`finalize`)."""
        return bool(self._clipped.get(seat, False))

    def component_totals(self, seat: int) -> Dict[str, float]:
        """Unscaled per-component sums for one seat, for logging/calibration."""
        return dict(self._components.get(seat, {}))

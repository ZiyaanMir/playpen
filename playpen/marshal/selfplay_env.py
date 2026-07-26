"""A thin two-seat (N-seat) self-play wrapper over a clembench game.

This mirrors the approach of ``MARSHAL/roll/agentic/env/playpen/env.py`` but with
no MARSHAL / Ray / DataProto dependency -- it drives clemcore's in-process
PettingZoo game env directly.

Every seat is marked ``"learner"`` in clemcore's ``AgentControlWrapper`` so that
control returns to *our* rollout loop after every single turn, regardless of which
seat just moved. The learner model is invoked outside this class (in
``playpen/marshal/selfplay_agent.py``) via TRL generation; here we only advance the
game and read observations/rewards.

Depends on clemcore (a hard playpen dependency) and torch/trl are *not* imported
here, so this module is importable without the training stack.
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional

# clemcore is a hard dependency of playpen, so importing at module load is safe.
from clemcore.clemgame.envs.pettingzoo import env as _clemcore_env
from clemcore.clemgame.envs.pettingzoo.wrappers import (
    AgentControlWrapper,
    GameInstanceIteratorWrapper,
)
from clemcore.clemgame.instances import GameInstances
from clemcore.clemgame.registry import GameRegistry


def resolve_game_spec(game_name: str):
    """Resolve a clembench game name to its GameSpec (num players, paths, ...)."""
    registry = GameRegistry.from_directories_and_cwd_files()
    specs = registry.get_game_specs_that_unify_with(game_name)
    if not specs:
        raise ValueError(f"No clembench game found matching {game_name!r}")
    return specs[0]


def load_instance_rows(
    game_name: str, *, instances_filter: Optional[Callable[[dict], bool]] = None
) -> List[dict]:
    """Load the packaged instance rows for a game (self-contained, no network).

    Each row is a dict with ``"experiment"`` and ``"game_instance"`` keys, in the
    stable order of the game's packaged ``instances.json``. The row *index* into
    this list is the instance identifier used throughout the MARSHAL pipeline:
    clembench ``game_id`` values are only unique *within* an experiment (e.g.
    taboo numbers 0..19 in each of ``high_en``/``medium_en``/``low_en``), so a
    bare game_id cannot address an instance unambiguously -- the index can.
    """
    game_spec = resolve_game_spec(game_name)
    instances = GameInstances.from_game_spec(game_spec)
    if instances_filter is not None:
        instances = instances.filter(instances_filter)
    return list(instances)


def list_instance_indices(
    game_name: str, *, instances_filter: Optional[Callable[[dict], bool]] = None
) -> List[int]:
    """List the instance indices for a game -- one per packaged instance row.

    These are exactly the indices that ``SelfPlayEnv.reset(instance_idx=...)``
    can look up, so the training dataset should be built from this list.
    """
    rows = load_instance_rows(game_name, instances_filter=instances_filter)
    return list(range(len(rows)))


class SelfPlayEnv:
    """Drive one clembench game as an N-seat self-play environment.

    Usage (per episode)::

        env = SelfPlayEnv("taboo")
        env.reset(instance_idx=0)
        while not env.done:
            agent_id = env.current_agent_id          # e.g. "player_0"
            player = env.current_player               # clemcore Player
            obs = env.observe()                        # {"role": "user", "content": ...}
            # ... build prompt from player perspective, generate `response` ...
            env.step(response)                         # advances the game
            rewards = env.cumulative_rewards           # per-seat, read right after step
        env.finalize()                                 # best-effort AEC cleanup
    """

    def __init__(
        self,
        game_name: str,
        *,
        instances_filter: Optional[Callable[[dict], bool]] = None,
        single_pass: bool = False,
    ) -> None:
        self.game_name = game_name
        self._game_spec = resolve_game_spec(game_name)
        self.num_players = int(self._game_spec.players)
        base_env = _clemcore_env(
            game_name, instances_filter=instances_filter, single_pass=single_pass
        )
        # The factory's outermost layer is a GameInstanceIteratorWrapper whose
        # reset() resolves instances by bare clembench game_id -- which is NOT
        # unique across experiments (find_by_game_id returns the first match, so
        # e.g. taboo's medium_en/low_en instances would be unreachable). We keep
        # our own copy of the instance rows, address them by list index, and
        # bypass that layer by passing the resolved experiment + game_instance
        # straight down on reset (all that wrapper's reset would have done).
        assert isinstance(base_env, GameInstanceIteratorWrapper), (
            "clemcore's env factory no longer returns a GameInstanceIteratorWrapper; "
            "SelfPlayEnv's instance-index reset needs updating for this clemcore version."
        )
        self._instance_rows = load_instance_rows(game_name, instances_filter=instances_filter)
        # Marking every seat "learner" hands control back to us after every turn.
        agent_mapping = {f"player_{i}": "learner" for i in range(self.num_players)}
        self._pz_env = AgentControlWrapper(base_env.env, agent_mapping)
        self._done = False

    @property
    def num_instances(self) -> int:
        return len(self._instance_rows)

    # -- lifecycle ---------------------------------------------------------

    def reset(self, instance_idx: int, *, seed: int = 0) -> None:
        """Reset to the packaged instance at ``instance_idx`` (see ``list_instance_indices``).

        The instance is handed down as a **deep copy**. Games are free to keep references
        into it and mutate them, because a benchmark run plays each instance exactly once
        -- codenames does exactly that: ``CodenamesBoard.__init__`` stores
        ``game_instance["assignments"][...]`` lists by reference and ``reveal_word`` calls
        ``self.hidden[assignment].remove(word)`` on them. A *training* loop replays one
        instance thousands of times, so without the copy every episode permanently strips
        words from the packaged board.

        Measured before this copy existed: a codenames board starts at 25 words and fell
        to ~7 by step 150 of a 500-step run, at which point the game is degenerate (it
        ends on or right after the cluegiver's move, so the guesser seat never plays and
        ~46% of every batch was a placeholder row). The better the policy got, the faster
        the corruption ran, because a valid clue reveals more words per episode.
        """
        instance_idx = int(instance_idx)
        if not 0 <= instance_idx < len(self._instance_rows):
            raise IndexError(
                f"instance_idx {instance_idx} out of range for {self.game_name!r} "
                f"({len(self._instance_rows)} instances)"
            )
        row = self._instance_rows[instance_idx]
        self._pz_env.reset(
            seed=seed,
            options={
                "experiment": copy.deepcopy(row["experiment"]),
                "game_instance": copy.deepcopy(row["game_instance"]),
            },
        )
        self._done = False

    def step(self, response: str) -> None:
        """Apply the current seat's response and advance to the next seat.

        Read ``cumulative_rewards`` immediately after this returns (before the next
        ``observe()``), because PettingZoo clears an agent's reward entry when it is
        observed via ``last()``.
        """
        self._pz_env.step(response)
        self._done = self._compute_done()

    def finalize(self) -> None:
        """Best-effort AEC cleanup for terminated agents.

        Not strictly required -- ``reset()`` builds a fresh game master and fully
        re-initializes state -- but we attempt it so clemcore ``on_game_end``
        callbacks (if any were attached) fire. Errors are swallowed because the
        next ``reset()`` is authoritative.
        """
        try:
            unwrapped = self._pz_env.unwrapped
            guard = 0
            while getattr(unwrapped, "agents", None) and guard <= self.num_players:
                if self._pz_env.agent_selection is None:
                    break
                self._pz_env.step(None)
                guard += 1
        except Exception:
            pass

    # -- observation / state ----------------------------------------------

    @property
    def current_agent_id(self) -> Optional[str]:
        return self._pz_env.agent_selection

    @property
    def current_seat(self) -> int:
        agent_id = self.current_agent_id
        if agent_id is None:
            return 0
        return int(agent_id.split("_")[1])

    @property
    def current_player(self):
        """The clemcore Player object for the current seat."""
        return self._pz_env.unwrapped.player_by_agent_id[self.current_agent_id]

    def player_for_seat(self, seat: int):
        return self._pz_env.unwrapped.player_by_agent_id[f"player_{seat}"]

    def observe(self) -> Dict:
        """Observation dict ({'role': 'user', 'content': ...}) for the current seat."""
        return self._pz_env.observe(self.current_agent_id)

    @property
    def done(self) -> bool:
        return self._done

    def _compute_done(self) -> bool:
        unwrapped = self._pz_env.unwrapped
        possible = getattr(unwrapped, "possible_agents", []) or []
        if not possible:
            return True
        return all(
            unwrapped.terminations.get(a, False) or unwrapped.truncations.get(a, False)
            for a in possible
        )

    @property
    def cumulative_rewards(self) -> List[float]:
        """Per-seat cumulative reward, read straight from the underlying env.

        For clembench's default sparse rewards this stays 0 for every seat until
        the terminal step, at which point every seat receives the shared outcome
        (SUCCESS: +1, FAILURE: 0, ABORTED: -1). We do not call ``last()`` during a
        rollout, so this accumulator is never cleared mid-episode.
        """
        unwrapped = self._pz_env.unwrapped
        return [
            float(unwrapped._cumulative_rewards.get(f"player_{i}", 0.0))
            for i in range(self.num_players)
        ]

    @property
    def outcome_success(self) -> bool:
        """Whether the game ended in SUCCESS (for logging)."""
        try:
            from clemcore.clemgame.master import Outcome

            return self._pz_env.unwrapped.game_master.state.outcome == Outcome.SUCCESS
        except Exception:
            return False

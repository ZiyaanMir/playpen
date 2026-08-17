#!/bin/bash
# Can this game actually be CONSTRUCTED in the training venv? Run before submitting.
#
#   experiments/lib/check_game_env.sh clean_up
#   experiments/lib/check_game_env.sh privateshared hot_air_balloon taboo
#
# WHY THIS EXISTS. A clembench game's master.py is imported at registry-lookup time,
# inside SelfPlayEnv.__init__, BEFORE a single episode runs. If that import needs a
# package the training venv does not have, every training job dies in seconds -- and
# because the segments are chained with -hold_jid/afterany, segments 2..N then each
# fail a second way, on `--resume-from-checkpoint latest: no resumable checkpoint`.
# One missing package therefore burns N GPU allocations and leaves nothing behind.
#
# That is not hypothetical. Across the synced runs:
#   * clean_up          x9 runs -- ModuleNotFoundError: No module named 'matplotlib'
#                       (clembench/clean_up/resources/game_state/game_state.py:5)
#   * privateshared     needs scikit-learn for the same reason (see experiments/README.md)
# The clean_up preset has warned about this in capitals since it was written; the check
# below is what makes the warning enforceable instead of advisory.
#
# It imports the game exactly the way training does -- resolve_game_spec() then the
# pettingzoo env -- so a game that passes here will not fail on import in the job.
# It does NOT play an episode, so it cannot catch a runtime bug mid-game (the
# air_balloon_survival `'dict' object has no attribute 'issubset'` crash is one of
# those; use a short smoke run for that).
#
# Deliberately not `set -e`: reporting the failure IS the job.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(dirname "$(dirname "$HERE")")}"

[ $# -gt 0 ] || {
    echo "usage: $(basename "$0") <game> [game ...]" >&2
    echo "       e.g. $(basename "$0") clean_up" >&2
    exit 2
}

PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || {
    echo "ERROR: no training venv python at $PY" >&2
    echo "       This must run against the TRAINING venv (the one train.sh uses)," >&2
    echo "       not the lm-eval environment. Build it per the cluster guide section 2." >&2
    exit 1
}
echo "[game-check] python = $PY"

RC=0
for GAME in "$@"; do
    # Imported in a subprocess per game so one hard failure cannot mask the next.
    "$PY" - "$GAME" <<'EOF'
import sys, traceback
game = sys.argv[1]
try:
    from playpen.marshal.selfplay_env import resolve_game_spec
    spec = resolve_game_spec(game)
except Exception:
    print(f"[game-check] {game}: FAIL -- could not resolve the game spec", flush=True)
    traceback.print_exc()
    sys.exit(1)

# The import that actually bites: master.py runs at load, pulling in the game's own
# resource modules and whatever they import at module scope.
try:
    from clemcore.clemgame.benchmark import GameBenchmark
    GameBenchmark.load_from_spec(spec)
except ModuleNotFoundError as e:
    print(f"[game-check] {game}: FAIL -- master.py needs a package the venv lacks: "
          f"No module named '{e.name}'", flush=True)
    print(f"[game-check] {game}: fix with   uv pip install {e.name}", flush=True)
    sys.exit(2)
except Exception:
    print(f"[game-check] {game}: FAIL -- master.py raised on import", flush=True)
    traceback.print_exc()
    sys.exit(1)

print(f"[game-check] {game}: OK (spec resolved, master.py imports)", flush=True)
EOF
    [ $? -eq 0 ] || RC=1
done

if [ "$RC" -ne 0 ]; then
    echo "" >&2
    echo "[game-check] At least one game cannot be constructed. Training jobs submitted" >&2
    echo "             for it would die in seconds, and every chained segment after the" >&2
    echo "             first would then fail again on 'no resumable checkpoint'." >&2
    echo "             Install the missing package into the TRAINING venv first:" >&2
    echo "" >&2
    echo "               source $REPO/.venv/bin/activate" >&2
    echo "               which python          # MUST be $REPO/.venv/bin/python" >&2
    echo "               uv pip install <package>" >&2
    echo "" >&2
    echo "             Keep uv's cache off \$HOME -- see the cluster guide (Eddie: 10 GB" >&2
    echo "             quota; Isambard: a second full copy on the wrong filesystem)." >&2
    exit 1
fi

echo "[game-check] all OK -- safe to submit."
exit 0

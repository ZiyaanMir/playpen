#!/bin/bash
# One line per experiment: what it was, how far it got, what it scored.
#
#   experiments/status.sh                 # $MARSHAL_RUNS, or the cluster default
#   experiments/status.sh <runs dir>      # somewhere else (e.g. an rsync'd copy)
#   experiments/status.sh -v              # add the key hyperparameters
#
# Reads only manifest.json / results.tsv / the directory tree, so it works on a
# login node, on a laptop after rsync, and while jobs are still running.

set -euo pipefail

VERBOSE=0
[ "${1:-}" = "-v" ] && { VERBOSE=1; shift; }

RUNS="${1:-${MARSHAL_RUNS:-}}"
if [ -z "$RUNS" ]; then
    if [ -n "${PROJECTDIR:-}" ] && [ -d "$PROJECTDIR/$USER/marshal-runs" ]; then
        RUNS="$PROJECTDIR/$USER/marshal-runs"
    else
        RUNS="/exports/eddie/scratch/$USER/marshal-runs"
    fi
fi
[ -d "$RUNS" ] || { echo "no runs directory: $RUNS" >&2; exit 1; }

PY="$(command -v python3 || command -v python)"

echo "runs: $RUNS"
echo

"$PY" - "$RUNS" "$VERBOSE" <<'PYEOF'
import glob, json, os, sys

runs, verbose = sys.argv[1], sys.argv[2] == "1"

def read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}

def _headline_columns(header):
    """Pick which metric column(s) to show on the one status line.

    With one eval task there is one obvious column. With several
    (EVAL_TASKS=logiglue,logicbench) showing just `acc=` is ambiguous, and the raw
    first column is alphabetical -- often a *subtask* (logicbench_bqa) rather than a
    headline. So: for each task prefix that has a top-level `<task>/acc` column,
    prefer that; report one per distinct top-level task, in task order. Falls back to
    the first data column when nothing matches that shape.
    """
    cols = list(enumerate(header))[1:]  # skip the "checkpoint" column
    # Group by top-level task (the part before any '_' in the task segment).
    preferred = []
    seen_tasks = set()
    for idx, name in cols:
        task = name.split("/", 1)[0]
        top = task.split("_", 1)[0]
        # A top-level metric looks like "<top>/<metric>" with task == top.
        if task == top and top not in seen_tasks:
            preferred.append((idx, name))
            seen_tasks.add(top)
    return preferred or cols[:1]

def best_metric(exp):
    """One headline number per top-level eval task, each with its delta vs base."""
    tsv = os.path.join(exp, "results.tsv")
    if not os.path.isfile(tsv):
        return ""
    try:
        with open(tsv) as fh:
            lines = [l.rstrip("\n").split("\t") for l in fh if l.strip()]
    except Exception:
        return ""
    if len(lines) < 2 or len(lines[0]) < 2:
        return ""
    header, rows = lines[0], lines[1:]
    base = next((r for r in rows if r[0] == "base"), None)
    last = rows[-1]
    parts = []
    for col, name in _headline_columns(header):
        try:
            value = float(last[col])
        except (ValueError, IndexError):
            continue
        piece = f"{name}={value:.4f}"
        if base is not None and base is not last:
            try:
                piece += f" ({value - float(base[col]):+.4f})"
            except (ValueError, IndexError):
                pass
        parts.append(piece)
    return "  ".join(parts)

def _fidelity(cfg):
    """fidelity_mode, flagged when marshal_exact ran without its unique pooling.

    Defaults to True for a manifest written before the sub-flag existed, so an old
    experiment still renders as a plain 'marshal_exact'.
    """
    mode = cfg.get("fidelity_mode", "?")
    if mode == "marshal_exact" and not cfg.get("marshal_exact_unique_pooling", True):
        return mode + "(no-unique-pool)"
    return mode


rows = []
for exp in sorted(glob.glob(os.path.join(runs, "*"))):
    if not os.path.isdir(exp):
        continue
    man = read_json(os.path.join(exp, "manifest.json"))
    if not man:
        continue  # not an experiment dir (e.g. an old <game>/ tree)

    train = man.get("training", {})
    # Both layouts: flat train/checkpoint-N (--no-run-subdir, what the job scripts use)
    # and the legacy train/<timestamp>/checkpoint-N from before that flag existed.
    cks = (glob.glob(os.path.join(exp, "train", "checkpoint-*"))
           + glob.glob(os.path.join(exp, "train", "*", "checkpoint-*")))
    n_ck = sum(1 for c in cks if os.path.isfile(os.path.join(c, "adapter_config.json")))
    n_eval = len(glob.glob(os.path.join(exp, "eval", "*", "**", "results*.json"), recursive=True))

    if n_ck == 0:
        state = "no ckpt"
    elif n_eval == 0:
        state = f"{n_ck} ckpt, not evaluated"
    elif os.path.isfile(os.path.join(exp, "RESULTS.md")):
        state = f"{n_ck} ckpt, scored"
    else:
        state = f"{n_ck} ckpt, eval partial"

    lp = man.get("length_penalty_effect", {})
    lp_s = "off"
    if lp.get("enabled"):
        total = lp.get("est_per_episode_total")
        lp_s = f"{total:+.2f}/ep" if isinstance(total, (int, float)) else "on"
        if isinstance(total, (int, float)) and abs(total) < 0.05:
            lp_s += " INERT"

    rows.append({
        "id": os.path.basename(exp),
        "game": train.get("game", "?"),
        "model": os.path.basename(str(train.get("model", "?"))),
        "steps": train.get("max_steps", "?"),
        "lp": lp_s,
        "fidelity": _fidelity(man.get("marshal_config", {})),
        "state": state,
        "score": best_metric(exp),
        "dirty": man.get("code", {}).get("git_dirty", False),
    })

if not rows:
    print("no experiments found (no manifest.json in any subdirectory).")
    print("Experiments created before experiments/ existed will not appear here.")
    sys.exit(0)

w_id = max(len(r["id"]) for r in rows)
w_state = max(len(r["state"]) for r in rows)
print(f"{'experiment'.ljust(w_id)}  {'state'.ljust(w_state)}  score")
print(f"{'-' * w_id}  {'-' * w_state}  -----")
for r in rows:
    print(f"{r['id'].ljust(w_id)}  {r['state'].ljust(w_state)}  {r['score']}")
    if verbose:
        print(f"{' ' * w_id}    model={r['model']} game={r['game']} steps={r['steps']} "
              f"len_penalty={r['lp']} fidelity={r['fidelity']}"
              + ("  [dirty tree]" if r["dirty"] else ""))

print()
print(f"{len(rows)} experiment(s).  Details: cat <dir>/manifest.txt   Scores: cat <dir>/RESULTS.md")
PYEOF

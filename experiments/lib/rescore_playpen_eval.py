"""Rebuild playpen-eval scores from gameplay that already happened. No GPU, no model.

`playpen eval` does four things: **play** every game (hours on a GPU), then score,
transcribe and aggregate (seconds, CPU only). If any step after the first fails, the
gameplay is still on disk and completely intact — but there is no `results.csv`, so
`PLAYPEN_RESULTS.md` comes out empty and hours of GPU look like nothing.

This script redoes only the cheap part, from the `interactions.json` files already in
`<EXP_DIR>/playpen-eval/<row>/`. It is safe to re-run and never touches gameplay.

    python experiments/lib/rescore_playpen_eval.py <EXP_DIR>
    python experiments/lib/rescore_playpen_eval.py <EXP_DIR> --row checkpoint-100
    python experiments/lib/rescore_playpen_eval.py --all $MARSHAL_RUNS

WHY THIS EXISTS (the 2026-07-30 Isambard run). `clembench/privateshared/master.py`
imports `sklearn`, which is not in the training venv. During gameplay clemcore catches
that per game and carries on, so 13 of the 14 games played normally. During scoring,
`clemcore.cli.score` collects exceptions and then calls **`sys.exit(1)`** — no
traceback, no message, just a silent non-zero exit that killed `playpen eval` before
`clemeval` ever ran. One missing package on one game therefore threw away the
aggregation for all 14 games across 8 checkpoints: ~3.5 h of GH200 time that had
already produced perfectly good interaction files.

So the recovery is deliberately **per game and failure-tolerant**: a game whose scorer
cannot even be imported is reported and skipped, and every other game is still scored
and aggregated. A partial clemscore over 13 games, with the missing one visible as a
gap in the per-game table, beats no result at all — and the fix (install the package)
is then a one-line change with the evidence already in hand.

`SystemExit` is caught explicitly: `clem.score`'s failure mode is `sys.exit`, which
is *not* an `Exception` subclass and sails straight through a bare `except Exception`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(_HERE))

SUITES = ("clem", "static")
SCORE_KEY = {"clem": "clemscore", "static": "statscore"}


def _games_in(suite_dir: str) -> list[str]:
    """Game names that actually have gameplay under this suite directory.

    Layout is ``<suite>/<model>/<game>/<experiment>/<episode>/interactions.json``, so
    the game is four path components up from the file. Derived from the files rather
    than from the registry so a partially-played run reports exactly what it has.
    """
    hits = glob.glob(os.path.join(suite_dir, "*", "*", "*", "*", "interactions.json"))
    return sorted({os.path.normpath(p).split(os.sep)[-4] for p in hits})


def _model_name(suite_dir: str) -> str | None:
    """The model directory clembench filed this run under."""
    entries = [d for d in glob.glob(os.path.join(suite_dir, "*")) if os.path.isdir(d)]
    return os.path.basename(entries[0]) if len(entries) == 1 else (
        os.path.basename(entries[0]) if entries else None
    )


def rescore_suite(suite_dir: str, suite: str, transcripts: bool) -> tuple[float | None, list[str], list[str]]:
    """Score + aggregate one `<row>/<suite>/` tree. Returns (score, scored, failed)."""
    import clemcore.cli as clem

    games = _games_in(suite_dir)
    if not games:
        return None, [], []

    scored, failed = [], []
    for game in games:
        try:
            clem.score(game, suite_dir)
            scored.append(game)
        except SystemExit:
            # clemcore.cli.score's own failure path -- NOT an Exception subclass.
            failed.append(game)
            print(f"    ! {game}: scoring failed (see the log above)", file=sys.stderr)
        except Exception as exc:
            failed.append(game)
            print(f"    ! {game}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if transcripts:
            # Human-readable HTML only; clemeval does not read these, so a failure
            # here must not cost us the score we just computed.
            try:
                clem.transcripts(game, suite_dir)
            except (SystemExit, Exception):
                print(f"    ~ {game}: transcripts failed (scores are unaffected)",
                      file=sys.stderr)

    if not scored:
        return None, scored, failed

    try:
        df = clem.clemeval.perform_evaluation(suite_dir, return_dataframe=True)
    except Exception as exc:
        print(f"    ! aggregation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None, scored, failed

    # clemeval's column, read positionally the same way playpen/cli.py does.
    try:
        value = float(df["-, clemscore"].iloc[0])
        score = None if value != value else value          # NaN => no score
    except Exception:
        score = None
    return score, scored, failed


def rescore_row(row_dir: str, transcripts: bool) -> bool:
    """Rebuild every suite under one checkpoint's directory. True if anything scored."""
    row = os.path.basename(row_dir.rstrip("/"))
    print(f"=== {row} ===")
    any_scored = False
    val: dict[str, float] = {}
    model = None

    for suite in SUITES:
        suite_dir = os.path.join(row_dir, suite)
        if not os.path.isdir(suite_dir):
            continue
        model = model or _model_name(suite_dir)
        score, scored, failed = rescore_suite(suite_dir, suite, transcripts)
        if not scored and not failed:
            print(f"  {suite}: no gameplay found")
            continue
        any_scored = any_scored or bool(scored)
        print(f"  {suite}: scored {len(scored)} game(s)"
              + (f", FAILED {len(failed)}: {', '.join(failed)}" if failed else "")
              + (f"  ->  {SCORE_KEY[suite]}={score:.2f}" if score is not None else
                 "  ->  no aggregate (every episode aborted)"))
        if score is not None:
            val[SCORE_KEY[suite]] = score

    # Match what `playpen eval` itself writes, so the directory is indistinguishable
    # from one produced by a run that succeeded first time.
    if val and model:
        path = os.path.join(row_dir, f"{model}.val.json")
        existing = {}
        if os.path.isfile(path):
            try:
                with open(path) as fh:
                    existing = json.load(fh)
            except Exception:
                existing = {}
        with open(path, "w") as fh:
            json.dump({**existing, **val}, fh)
    return any_scored


def rescore_experiment(exp_dir: str, only_row: str | None, transcripts: bool) -> int:
    eval_dir = os.path.join(exp_dir, "playpen-eval")
    if not os.path.isdir(eval_dir):
        print(f"[rescore] no playpen-eval/ in {exp_dir} -- nothing to do.")
        return 0

    rows = sorted(
        d for d in os.listdir(eval_dir)
        if os.path.isdir(os.path.join(eval_dir, d)) and not d.startswith(".")
    )
    if only_row:
        rows = [r for r in rows if r == only_row]
        if not rows:
            print(f"[rescore] no row '{only_row}' under {eval_dir}", file=sys.stderr)
            return 0

    print(f"[rescore] {os.path.basename(exp_dir.rstrip('/'))}: {len(rows)} row(s)")
    done = sum(1 for r in rows if rescore_row(os.path.join(eval_dir, r), transcripts))

    # Rebuild the table from whatever now exists.
    sys.path.insert(0, _HERE)
    import summarize_playpen_eval
    sys.argv = ["summarize_playpen_eval.py", exp_dir]
    summarize_playpen_eval.main()
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="an experiment directory, or a runs root with --all")
    ap.add_argument("--all", action="store_true",
                    help="treat PATH as $MARSHAL_RUNS and rescore every experiment under it")
    ap.add_argument("--row", default="",
                    help="only this row, e.g. checkpoint-100 (single experiment only)")
    ap.add_argument("--transcripts", action="store_true",
                    help="also rebuild the human-readable HTML transcripts (slower)")
    ap.add_argument("--clembench", default="",
                    help="clembench checkout (default: <repo>/clembench)")
    args = ap.parse_args()

    clembench = args.clembench or os.path.join(REPO, "clembench")
    if not os.path.isdir(clembench):
        print(f"ERROR: no clembench checkout at {clembench}; pass --clembench.",
              file=sys.stderr)
        raise SystemExit(1)

    root = os.path.abspath(args.path)
    if args.all:
        exps = sorted(
            d for d in glob.glob(os.path.join(root, "*"))
            if os.path.isdir(os.path.join(d, "playpen-eval"))
            and not os.path.basename(d).startswith("_")
        )
        if not exps:
            print(f"[rescore] no experiment under {root} has a playpen-eval/ directory.")
            return
    else:
        exps = [root]

    # clemcore resolves games relative to the CWD, so run from a scratch directory
    # holding a generated game_registry.json. Absolute results paths are passed in,
    # so the chdir only affects game discovery -- and it keeps clembench.log and any
    # other CWD-relative output out of wherever the caller happened to be standing.
    with tempfile.TemporaryDirectory(prefix="rescore-") as work:
        with open(os.path.join(work, "game_registry.json"), "w") as fh:
            json.dump([{"benchmark_path": clembench}], fh)
        here = os.getcwd()
        os.chdir(work)
        try:
            total = sum(rescore_experiment(e, args.row or None, args.transcripts)
                        for e in exps)
        finally:
            os.chdir(here)

    print(f"\n[rescore] rebuilt scores for {total} row(s) across {len(exps)} experiment(s).")
    if exps:
        print(f"[rescore] read them: cat {exps[0]}/PLAYPEN_RESULTS.md")


if __name__ == "__main__":
    main()

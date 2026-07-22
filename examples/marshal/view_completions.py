"""Inspect the completions parquet files a MARSHAL/GRPO run writes.

TRL's ``log_completions=True`` writes one parquet per logged step into
``<run dir>/completions/completions_<step>.parquet`` with columns
``step, prompt, completion, reward_func, advantage`` -- the same data as the table
TRL prints to the console, but kept after the log scrolls away.

Usage::

    # newest run under models/marshal/, summary of every logged step
    .venv/bin/python examples/marshal/view_completions.py

    # a specific run (dir, completions/ dir, or a single .parquet all work)
    .venv/bin/python examples/marshal/view_completions.py models/marshal/dond/Qwen3-4B/20260722-1904

    # read one row in full -- prompt and completion, untruncated
    .venv/bin/python examples/marshal/view_completions.py --row 0

    # only the rows that aborted, full text
    .venv/bin/python examples/marshal/view_completions.py --aborted --full

Reads only pandas/pyarrow, so it runs without torch, trl or vllm.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

# clembench terminal rewards, for labelling rows.
ABORT_REWARD = -1.0
SUCCESS_REWARD = 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("path", nargs="?", default=None,
                   help="Run dir, completions/ dir, or a .parquet file. "
                        "Default: the newest run under models/marshal/.")
    p.add_argument("--row", type=int, default=None,
                   help="Print this row index in full (prompt + completion) and exit.")
    p.add_argument("--full", action="store_true",
                   help="Print every selected row in full instead of the summary table.")
    p.add_argument("--aborted", action="store_true",
                   help="Keep only rows whose reward is the clembench ABORTED value (-1).")
    p.add_argument("--step", type=int, default=None, help="Keep only this training step.")
    p.add_argument("--chars", type=int, default=400,
                   help="Completion characters to show per row in --full mode (0 = all).")
    return p.parse_args()


def find_files(path: str | None) -> list[str]:
    """Resolve a user-supplied path (or nothing) to a sorted list of parquet files."""
    if path is None:
        candidates = sorted(
            glob.glob("models/**/completions/*.parquet", recursive=True),
            key=os.path.getmtime,
        )
        if not candidates:
            sys.exit("No completions parquet found under models/. Pass a path explicitly.")
        # Newest file wins, but take every parquet from that same run directory so a
        # multi-step run is summarized as a whole rather than one step at a time.
        newest_dir = os.path.dirname(candidates[-1])
        return sorted(glob.glob(os.path.join(newest_dir, "*.parquet")))
    if path.endswith(".parquet"):
        return [path]
    for pattern in (
        os.path.join(path, "*.parquet"),                  # already the completions/ dir
        os.path.join(path, "completions", "*.parquet"),   # a run dir
        os.path.join(path, "**", "*.parquet"),            # a game/model dir above the runs
    ):
        hits = sorted(glob.glob(pattern, recursive=True))
        if hits:
            return hits
    sys.exit(f"No .parquet files found under {path!r}")


def load(files: list[str]):
    import pandas as pd

    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def label(reward: float) -> str:
    if reward == ABORT_REWARD:
        return "ABORTED"
    if reward == SUCCESS_REWARD:
        return "SUCCESS"
    return "failure"


def print_row(df, i: int, chars: int) -> None:
    row = df.iloc[i]
    clip = (lambda s: s) if chars == 0 else (lambda s: s[:chars])
    print("=" * 78)
    print(f"row {i} | step {row['step']} | reward {row['reward_func']:+.3f} "
          f"({label(row['reward_func'])}) | advantage {row['advantage']:+.4f}")
    print("-" * 78)
    print("PROMPT:")
    print(clip(row["prompt"]))
    print("-" * 78)
    print("COMPLETION:")
    print(clip(row["completion"]))
    print()


def summarize(df) -> None:
    """One line per row, plus the aggregates that actually diagnose a run."""
    print(f"{'idx':>4} {'step':>5} {'reward':>8} {'outcome':>8} {'advantage':>10} "
          f"{'chars':>7}  completion (first 60 chars)")
    for i in range(len(df)):
        row = df.iloc[i]
        head = " ".join(str(row["completion"])[:60].split())
        print(f"{i:>4} {row['step']:>5} {row['reward_func']:>+8.3f} "
              f"{label(row['reward_func']):>8} {row['advantage']:>+10.4f} "
              f"{len(str(row['completion'])):>7}  {head}")

    print()
    n = len(df)
    aborted = int((df["reward_func"] == ABORT_REWARD).sum())
    success = int((df["reward_func"] == SUCCESS_REWARD).sum())
    print(f"rows={n}  steps={sorted(df['step'].unique())}")
    print(f"outcomes: SUCCESS {success}/{n} ({success / n:.0%})   "
          f"ABORTED {aborted}/{n} ({aborted / n:.0%})   "
          f"failure {n - success - aborted}/{n}")
    print(f"reward    mean={df['reward_func'].mean():+.4f}  "
          f"min={df['reward_func'].min():+.3f}  max={df['reward_func'].max():+.3f}")
    print(f"advantage mean={df['advantage'].mean():+.4f}  "
          f"min={df['advantage'].min():+.4f}  max={df['advantage'].max():+.4f}  "
          f"std={df['advantage'].std():.4f}")
    lens = df["completion"].astype(str).str.len()
    print(f"completion chars  mean={lens.mean():.0f}  min={lens.min()}  max={lens.max()}")

    # The two failure modes worth catching automatically, because both look like a
    # healthy run in the logs and neither raises.
    if (df["advantage"] == 0).all():
        print("\n!! EVERY ADVANTAGE IS EXACTLY 0 -> this batch contributed NO gradient.")
        print("   Cause is almost always a degenerate pool: all rows in a seat share the")
        print("   same outcome, so per-seat mean-centering cancels. Needs outcome variance.")
    elif df["reward_func"].nunique() == 1:
        print(f"\n!! All rows share reward {df['reward_func'].iloc[0]:+.3f} — the pool is")
        print("   degenerate even if advantages are nonzero (e.g. via whitening).")
    if aborted == n:
        print("\n!! 100% ABORTED — fix the format/parser side before reading anything")
        print("   into the advantage numbers. Inspect with --aborted --full.")


def main() -> None:
    args = parse_args()
    files = find_files(args.path)
    df = load(files)

    print(f"# {len(files)} file(s) from {os.path.dirname(files[0])}")
    if args.step is not None:
        df = df[df["step"] == args.step]
    if args.aborted:
        df = df[df["reward_func"] == ABORT_REWARD]
    if df.empty:
        sys.exit("No rows match those filters.")
    df = df.reset_index(drop=True)

    if args.row is not None:
        if not 0 <= args.row < len(df):
            sys.exit(f"--row {args.row} out of range (0..{len(df) - 1})")
        print_row(df, args.row, chars=0)
        return
    if args.full:
        for i in range(len(df)):
            print_row(df, i, chars=args.chars)
        return
    summarize(df)


if __name__ == "__main__":
    main()

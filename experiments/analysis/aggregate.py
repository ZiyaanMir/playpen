#!/usr/bin/env python3
"""Average every run's scores into tidy CSVs, grouped by reasoning type.

    python experiments/analysis/aggregate.py --runs cluster-runs/isambard \
        --model Qwen/Qwen3-4B --out results/

Writes, per metric (acc / acc_norm / both):
    logic_by_run.csv     one row per (run, checkpoint, benchmark, group)
    logic_by_ckpt.csv    averaged over runs -- what the line graph plots
    playpen_by_run.csv   one row per (run, checkpoint, game)
    playpen_by_ckpt.csv  averaged over runs, fixed game population only

Every score is a PERCENTAGE. n_runs/n_tasks/n_samples travel with each row so a
mean can always be traced back to what it averaged.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysislib as A


def write_csv(path: str, header: list[str], rows: list[list]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")


def build(runs, metric, groups, outdir, weighting, game_filter=None):
    # ---- logic, per run --------------------------------------------------- #
    per_run_rows = []
    by_ckpt = defaultdict(list)          # (benchmark, group, ckpt) -> [score]
    for r in runs:
        scores = A.logic_scores(r["path"], metric, groups, weighting)
        for ckpt, agg in scores.items():
            for (bench, grp), v in sorted(agg.items()):
                per_run_rows.append([
                    r["cluster"], r["name"], r["game"], r["model"], r["tag"],
                    ckpt, bench, grp,
                    f"{v['score']:.4f}", v["n_tasks"], v["n_samples"],
                ])
                by_ckpt[(bench, grp, ckpt)].append(v["score"])
    write_csv(os.path.join(outdir, "logic_by_run.csv"),
              ["cluster", "run", "game", "model", "tag", "checkpoint",
               "benchmark", "group", "score_pct", "n_tasks", "n_samples"],
              per_run_rows)

    ck_rows = []
    for (bench, grp, ckpt), vals in sorted(
            by_ckpt.items(), key=lambda kv: (kv[0][0], kv[0][1], A._row_sort_key(kv[0][2]))):
        ck_rows.append([
            bench, grp, ckpt, f"{statistics.fmean(vals):.4f}",
            f"{statistics.stdev(vals):.4f}" if len(vals) > 1 else "",
            len(vals),
        ])
    write_csv(os.path.join(outdir, "logic_by_ckpt.csv"),
              ["benchmark", "group", "checkpoint", "mean_score_pct", "sd_pct", "n_runs"],
              ck_rows)

    # ---- gameplay --------------------------------------------------------- #
    pp = {r["name"]: A.playpen_scores(r["path"]) for r in runs}
    pp = {k: v for k, v in pp.items() if v}
    games = game_filter or A.fixed_population(pp)
    pp_rows, pp_by = [], defaultdict(list)
    name_to_run = {r["name"]: r for r in runs}
    for name, rows in pp.items():
        r = name_to_run[name]
        for ckpt, per_game in rows.items():
            for game, mets in sorted(per_game.items()):
                q = mets.get("Quality Score")
                p = mets.get("% Played")
                pp_rows.append([
                    r["cluster"], name, r["game"], r["tag"], ckpt, game,
                    "" if q is None else f"{q:.4f}",
                    "" if p is None else f"{p:.4f}",
                    "yes" if game in games else "no",
                ])
                if game in games and q is not None:
                    pp_by[(game, ckpt)].append(q)
    write_csv(os.path.join(outdir, "playpen_by_run.csv"),
              ["cluster", "run", "trained_game", "tag", "checkpoint", "eval_game",
               "quality_score_pct", "pct_played", "in_fixed_population"],
              pp_rows)

    ppc = []
    for (game, ckpt), vals in sorted(pp_by.items(), key=lambda kv: (kv[0][0], A._row_sort_key(kv[0][1]))):
        ppc.append([game, ckpt, f"{statistics.fmean(vals):.4f}",
                    f"{statistics.stdev(vals):.4f}" if len(vals) > 1 else "", len(vals)])
    write_csv(os.path.join(outdir, "playpen_by_ckpt.csv"),
              ["game", "checkpoint", "mean_quality_pct", "sd_pct", "n_runs"], ppc)

    with open(os.path.join(outdir, "fixed_population.txt"), "w") as fh:
        fh.write("Games scored in every row of every run in this scope.\n")
        fh.write("Cross-run means are computed over THIS SET ONLY, so they are like-for-like.\n\n")
        fh.write(f"n_games = {len(games)}\n")
        for g in games:
            fh.write(f"  {g}\n")
    print(f"  fixed game population: {len(games)} -> {', '.join(games) or '(none)'}")
    return games


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="cluster root(s), e.g. cluster-runs/isambard")
    ap.add_argument("--out", default="results", help="output directory")
    ap.add_argument("--model", nargs="*", default=None, help="filter by MODEL")
    ap.add_argument("--tag", nargs="*", default=None, help="filter by EXP_TAG")
    ap.add_argument("--name", default=None, help="glob on the run directory name")
    ap.add_argument("--metric", choices=list(A.METRICS) + ["all"], default="all")
    ap.add_argument("--weighting", choices=["macro", "sample"], default="macro")
    ap.add_argument("--games", nargs="*", default=None,
                    help="override the fixed game population instead of intersecting")
    ap.add_argument("--groups", default=A.DEFAULT_GROUPS)
    args = ap.parse_args()

    groups = A.load_groups(args.groups)
    review = [k for k, v in groups["logiglue"].items() if v.get("confidence") == "review"]
    if review:
        print("NOTE: these logiglue assignments are marked 'review' in "
              f"{os.path.basename(args.groups)} -- verify before publishing:")
        for k in review:
            print(f"      {k} -> {groups['logiglue'][k]['group']}")

    runs = A.discover(args.runs,
                      model_filter=set(args.model) if args.model else None,
                      tag_filter=set(args.tag) if args.tag else None,
                      name_glob=args.name)
    runs = [r for r in runs if A.eval_rows(r["path"])]
    print(f"\n{len(runs)} run(s) with eval results in scope")
    if not runs:
        sys.exit("nothing to aggregate")

    for metric in (A.METRICS if args.metric == "all" else [args.metric]):
        outdir = os.path.join(args.out, f"metric_{metric}")
        print(f"\n=== metric: {metric} (weighting={args.weighting}) -> {outdir}")
        build(runs, metric, groups, outdir, args.weighting, args.games)


if __name__ == "__main__":
    main()

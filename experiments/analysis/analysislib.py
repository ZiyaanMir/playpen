"""Shared loading/aggregation for the MARSHAL result analysis scripts.

Reads only files that already exist in an experiment directory:
    eval/<row>/**/results*.json   lm-eval scores      (logiglue + logicbench)
    playpen_results.tsv           clembench gameplay
    experiment.env / manifest.json config, for arm grouping

Stdlib only. Nothing here needs torch, a GPU, or the training venv.

TWO CONVENTIONS APPLY EVERYWHERE, because getting them wrong silently changes
every number downstream:

  * SCORES ARE PERCENTAGES. lm-eval reports accuracy in [0,1]; every value this
    module returns is already multiplied by 100. Deltas are therefore in
    PERCENTAGE POINTS (pp), not ratios.
  * A GROUP SCORE IS A MACRO MEAN over its leaf tasks by default -- each task
    counts once regardless of how many examples it has. That is the usual
    benchmark convention, but it over-weights tiny tasks, so `weighting="sample"`
    (weight each task by its sample_len) is available and the CSVs always carry
    n_tasks and n_samples so either can be reconstructed.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GROUPS = os.path.join(HERE, "logic_groups.json")

# The metric knob the user asked for. "both" is the mean of acc and acc_norm,
# computed PER TASK before any grouping -- averaging the two group means instead
# would give a different number whenever a group's tasks are unevenly covered.
METRICS = ("acc", "acc_norm", "both")

# Group rows lm-eval emits alongside real tasks. They are aggregates, not leaves,
# and including them would double-count.
GROUP_ROWS = {"logiglue", "logicbench", "logic_bench"}


# --------------------------------------------------------------------------- #
# taxonomy
# --------------------------------------------------------------------------- #
def load_groups(path: str = DEFAULT_GROUPS) -> dict:
    with open(path) as fh:
        return json.load(fh)


def logicbench_family(task: str) -> str | None:
    """first_order / propositional / non_monotonic, parsed from the task name.

    Mechanical on purpose: the family is encoded in the name by the benchmark
    itself, so unlike logiglue it needs no hand-maintained table and cannot drift.
    """
    m = re.match(r"^(?:bqa|mcqa)_(first_order|propositional|non_monotonic)_logic_", task)
    return m.group(1) if m else None


def classify(task: str, groups: dict) -> tuple[str, str] | None:
    """(benchmark, group) for a leaf task, or None if it is not one of ours."""
    fam = logicbench_family(task)
    if fam:
        return ("logicbench", fam)
    if task.startswith("logiglue"):
        entry = groups["logiglue"].get(task)
        if entry is None:
            raise KeyError(
                f"logiglue task {task!r} is not in logic_groups.json. Add it (or fix "
                f"the name) -- refusing to guess a reasoning type."
            )
        return ("logiglue", entry["group"])
    return None


# --------------------------------------------------------------------------- #
# lm-eval results
# --------------------------------------------------------------------------- #
def _row_sort_key(name: str) -> tuple[int, int]:
    if name == "base":
        return (0, 0)
    m = re.match(r"^checkpoint-(\d+)$", name)
    return (1, int(m.group(1))) if m else (2, 0)


def read_eval_row(row_dir: str) -> dict[str, dict]:
    """Merge every results*.json under one eval/<row>/ directory.

    Oldest first, newer winning per task, mirroring summarize_eval.py: a
    checkpoint accumulates one file per lm-eval invocation and different
    invocations can cover different task sets (logiglue one day, logicbench the
    next). Taking only the newest file would silently drop the other benchmark.
    """
    out: dict[str, dict] = {}
    files = sorted(
        glob.glob(os.path.join(row_dir, "**", "results*.json"), recursive=True),
        key=os.path.getmtime,
    )
    for path in files:
        try:
            with open(path) as fh:
                res = json.load(fh).get("results", {}) or {}
        except Exception:
            continue
        for task, metrics in res.items():
            if not isinstance(metrics, dict):
                continue
            acc = metrics.get("acc,none")
            accn = metrics.get("acc_norm,none")
            if not isinstance(acc, (int, float)) and not isinstance(accn, (int, float)):
                continue  # group row with no metrics, or a task that scored nothing
            out[task] = {
                "acc": acc if isinstance(acc, (int, float)) else None,
                "acc_norm": accn if isinstance(accn, (int, float)) else None,
                "n": metrics.get("sample_len") or 0,
            }
    return out


def task_value(rec: dict, metric: str) -> float | None:
    """One task's score under the chosen metric, as a PERCENTAGE."""
    if metric == "acc":
        v = rec["acc"]
    elif metric == "acc_norm":
        v = rec["acc_norm"]
    elif metric == "both":
        vals = [x for x in (rec["acc"], rec["acc_norm"]) if x is not None]
        v = statistics.fmean(vals) if vals else None
    else:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
    return None if v is None else v * 100.0


def eval_rows(exp: str) -> list[str]:
    ed = os.path.join(exp, "eval")
    if not os.path.isdir(ed):
        return []
    names = [d for d in os.listdir(ed) if os.path.isdir(os.path.join(ed, d))]
    return sorted(names, key=_row_sort_key)


def logic_scores(exp: str, metric: str, groups: dict, weighting: str = "macro"):
    """{row: {(benchmark, group): {"score","n_tasks","n_samples"}}} in percent."""
    out: dict[str, dict] = {}
    for row in eval_rows(exp):
        tasks = read_eval_row(os.path.join(exp, "eval", row))
        buckets: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
        for task, rec in tasks.items():
            if task in GROUP_ROWS:
                continue
            key = classify(task, groups)
            if key is None:
                continue
            v = task_value(rec, metric)
            if v is None:
                continue
            buckets[key].append((v, rec["n"]))
        if not buckets:
            continue
        agg = {}
        for key, pairs in buckets.items():
            vals = [v for v, _ in pairs]
            ns = [n for _, n in pairs]
            if weighting == "sample" and sum(ns) > 0:
                score = sum(v * n for v, n in pairs) / sum(ns)
            else:
                score = statistics.fmean(vals)
            agg[key] = {"score": score, "n_tasks": len(vals), "n_samples": sum(ns)}
        out[row] = agg
    return out


# --------------------------------------------------------------------------- #
# gameplay results
# --------------------------------------------------------------------------- #
GAME_METRICS = ("Quality Score", "% Played")


def playpen_scores(exp: str) -> dict[str, dict[str, dict[str, float]]]:
    """{row: {game: {"Quality Score": v, "% Played": v}}} -- already 0-100."""
    path = os.path.join(exp, "playpen_results.tsv")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    out: dict[str, dict] = {}
    for r in rows:
        row = (r.get("checkpoint") or "").strip()
        if not row:
            continue
        per_game: dict[str, dict] = defaultdict(dict)
        for col, raw in r.items():
            if not col or "," not in col:
                continue
            game, _, met = col.partition(", ")
            game, met = game.strip(), met.strip()
            if met not in GAME_METRICS or game in ("all", "fixed", "-"):
                continue
            raw = (raw or "").strip()
            if not raw:
                continue
            try:
                per_game[game][met] = float(raw)
            except ValueError:
                continue
        if per_game:
            out[row] = dict(per_game)
    return out


def fixed_population(per_run: dict[str, dict], metric: str = "Quality Score") -> list[str]:
    """Games scored in EVERY row of EVERY given run.

    Why this and not each run's own `fixed` column: those populations differ
    between runs (9 here, 10 there), so subtracting one run's mean from
    another's would compare means over DIFFERENT games -- measuring the
    population change as much as the model. Intersecting across the whole
    comparison set is what makes a cross-run number like-for-like.
    """
    common: set[str] | None = None
    for rows in per_run.values():
        if not rows:
            return []
        ok = None
        for _row, games in rows.items():
            here = {g for g, m in games.items() if metric in m}
            ok = here if ok is None else (ok & here)
        common = (ok or set()) if common is None else (common & (ok or set()))
    return sorted(common or [])


# --------------------------------------------------------------------------- #
# runs and arms
# --------------------------------------------------------------------------- #
def read_env(exp: str) -> dict[str, str]:
    out: dict[str, str] = {}
    path = os.path.join(exp, "experiment.env")
    if not os.path.exists(path):
        return out
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        out[k.strip()] = v.replace("\\ ", " ")
    return out


def discover(roots: list[str], cluster_filter=None, model_filter=None,
             tag_filter=None, name_glob=None) -> list[dict]:
    """Every experiment directory under the given cluster roots."""
    import fnmatch

    runs = []
    for root in roots:
        cluster = os.path.basename(root.rstrip("/"))
        for exp in sorted(glob.glob(os.path.join(root, "*"))):
            if not os.path.isdir(exp) or os.path.basename(exp).startswith("_"):
                continue
            env = read_env(exp)
            if not env:
                continue
            name = os.path.basename(exp)
            if cluster_filter and cluster not in cluster_filter:
                continue
            if model_filter and env.get("MODEL", "") not in model_filter:
                continue
            if tag_filter and env.get("EXP_TAG", "") not in tag_filter:
                continue
            if name_glob and not fnmatch.fnmatch(name, name_glob):
                continue
            runs.append({
                "path": exp, "name": name, "cluster": cluster,
                "game": env.get("GAME", ""), "model": env.get("MODEL", ""),
                "tag": env.get("EXP_TAG", ""), "env": env,
            })
    return runs


def arm_key(run: dict) -> tuple[str, str]:
    """An ARM is (algorithm variant, model scale) -- across games and clusters.

    EXP_TAG is the algorithm variant: the harness requires it to be set whenever a
    run deviates from the preset, so it is the field that actually identifies the
    configuration. EXTRA_TRAIN_ARGS is recorded alongside for the config dump and
    is checked for consistency there, not used as part of the key (the same arm is
    sometimes expressed with equivalent flag spellings).
    """
    return (run["tag"] or "(untagged)", run["model"] or "(unknown)")

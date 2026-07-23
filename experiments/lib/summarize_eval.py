"""Aggregate an experiment's lm-eval outputs into one table at the experiment root.

Walks ``<EXP_DIR>/eval/*/`` (``base``, ``checkpoint-50``, ``checkpoint-100``, ...),
reads the newest ``results*.json`` lm-eval wrote under each, and emits:

* ``<EXP_DIR>/RESULTS.md``  -- a markdown table, one row per checkpoint
* ``<EXP_DIR>/results.tsv`` -- the same numbers for pandas/Excel

Rows are ordered ``base`` first then by training step, which is the order you want
to read a learning curve in. The base row is the untrained model, so every later
row should be read as a delta against it -- if a checkpoint scores identically to
base on every metric, the adapter almost certainly was not applied (see the guide's
"identical scores" check).

Stdlib only, so it runs in the lm-eval env, the training venv, or on a laptop.
"""

from __future__ import annotations

import glob
import json
import os
import sys

# lm-eval suffixes every metric with its filter name; ",none" is the unfiltered
# default. Stderr columns double the table width without adding signal.
METRIC_SUFFIX = ",none"


def _newest_results_json(eval_subdir: str) -> str | None:
    """The most recent results file lm-eval wrote anywhere under this directory.

    lm-eval nests output as ``<output_path>/<sanitized_model_name>/results_<ts>.json``,
    and the sanitized name embeds the adapter path, so it is not predictable from
    here -- hence the recursive glob. Newest wins, so re-running an eval supersedes
    the earlier attempt rather than being ambiguous.
    """
    hits = glob.glob(os.path.join(eval_subdir, "**", "results*.json"), recursive=True)
    if not hits:
        return None
    return max(hits, key=os.path.getmtime)


def _scores(path: str) -> dict[str, float]:
    """Flatten one results.json into {"task/metric": value}."""
    try:
        with open(path) as fh:
            data = json.load(fh)
    except Exception:
        return {}
    out: dict[str, float] = {}
    for task, metrics in (data.get("results") or {}).items():
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if not key.endswith(METRIC_SUFFIX) or "stderr" in key:
                continue
            if not isinstance(value, (int, float)):
                continue
            out[f"{task}/{key[: -len(METRIC_SUFFIX)]}"] = float(value)
    return out


def _sort_key(name: str) -> tuple[int, int]:
    """base first, then checkpoints by step number."""
    if name == "base":
        return (0, 0)
    try:
        return (1, int(name.rsplit("-", 1)[1]))
    except (IndexError, ValueError):
        return (2, 0)


def main() -> None:
    exp_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("EXP_DIR", "")
    if not exp_dir:
        print("usage: summarize_eval.py <EXP_DIR>", file=sys.stderr)
        raise SystemExit(2)

    eval_dir = os.path.join(exp_dir, "eval")
    if not os.path.isdir(eval_dir):
        print(f"[summary] no eval dir at {eval_dir} -- nothing to summarize.")
        return

    names = sorted(
        (d for d in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, d))),
        key=_sort_key,
    )

    rows: list[tuple[str, dict[str, float]]] = []
    missing: list[str] = []
    for name in names:
        found = _newest_results_json(os.path.join(eval_dir, name))
        if found is None:
            missing.append(name)
            continue
        rows.append((name, _scores(found)))

    if not rows:
        print(f"[summary] found no results*.json under {eval_dir} -- nothing to summarize.")
        return

    columns: list[str] = []
    for _, scores in rows:
        for col in scores:
            if col not in columns:
                columns.append(col)
    columns.sort()

    exp_id = os.path.basename(exp_dir.rstrip("/"))
    baseline = dict(rows[0][1]) if rows[0][0] == "base" else {}

    md = [
        f"# Results: {exp_id}",
        "",
        "Written by `experiments/lib/summarize_eval.py`. See `manifest.txt` in this "
        "directory for the model, game and hyperparameters these scores belong to.",
        "",
    ]
    header = ["checkpoint", *columns]
    md.append("| " + " | ".join(header) + " |")
    md.append("|" + "|".join("---" for _ in header) + "|")
    for name, scores in rows:
        cells = [f"`{name}`"]
        for col in columns:
            if col not in scores:
                cells.append("-")
                continue
            value = scores[col]
            # Show the delta against base: that is the number the run exists to produce.
            if baseline and name != "base" and col in baseline:
                cells.append(f"{value:.4f} ({value - baseline[col]:+.4f})")
            else:
                cells.append(f"{value:.4f}")
        md.append("| " + " | ".join(cells) + " |")

    if baseline:
        md += ["", "Bracketed numbers are the change from the untrained `base` row. "
                   "A checkpoint identical to `base` on every metric means the adapter "
                   "was not applied -- check the eval log for the `peft=` argument."]
    if missing:
        md += ["", f"**Incomplete:** no results file for {', '.join(f'`{m}`' for m in missing)} "
                   "(eval failed, was killed, or is still running)."]

    with open(os.path.join(exp_dir, "RESULTS.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")

    with open(os.path.join(exp_dir, "results.tsv"), "w") as fh:
        fh.write("\t".join(["checkpoint", *columns]) + "\n")
        for name, scores in rows:
            fh.write("\t".join(
                [name] + [f"{scores[c]:.6f}" if c in scores else "" for c in columns]
            ) + "\n")

    print(f"[summary] {len(rows)} row(s), {len(columns)} metric(s) "
          f"-> {exp_dir}/RESULTS.md and results.tsv")
    if missing:
        print(f"[summary] no results for: {', '.join(missing)}")


if __name__ == "__main__":
    main()

"""Aggregate an experiment's clembench gameplay results into one table.

Sibling of ``summarize_eval.py``, which does the same for lm-eval. Walks
``<EXP_DIR>/playpen-eval/*/`` (``base``, ``checkpoint-50``, ...) and emits:

* ``<EXP_DIR>/PLAYPEN_RESULTS.md``  -- headline scores, then a per-game breakdown
* ``<EXP_DIR>/playpen_results.tsv`` -- the same numbers for pandas/Excel

The numbers come from clemeval's ``<suite>/results.csv``, not from the
``<model>.val.json``. The val.json holds only the single aggregate clemscore, and
that aggregate is **NaN whenever every episode aborted** -- which is precisely the
failure mode these runs are prone to (a model that cannot produce a parseable move
scores NaN, not 0). results.csv carries the per-game ``% Played`` and
``Quality Score`` that tell those two apart:

    % Played 0, Quality -    the model never produced a valid move   (format failure)
    % Played 100, Quality 0  it played properly and lost every game  (skill failure)

Both look identical in a clemscore column, and they call for opposite responses, so
the per-game table is the point of this file rather than a decoration.

clemscore is recomputed here as ``% Played / 100 * Quality Score`` when clemeval
left it blank but both components are present, so a partially-aborting run still
gets a comparable headline number. That is clemeval's own definition, not a new one.

Rows are ``base`` first, then by training step.

THE TWO POPULATIONS (why there are two headline tables)
-------------------------------------------------------
clemeval's two aggregates are means over *different* sets of games:

* ``all, Average % Played`` averages over **every game scored**.
* ``all, Average Quality Score`` averages over only the games that produced at
  least one parseable episode -- a game the model aborted 100% of the time has no
  quality to average, so clemeval leaves it blank and drops it from the mean.

That is a sound definition for describing one model. It is **not** sound for
subtracting two models. The set of quality-contributing games changes from
checkpoint to checkpoint: a checkpoint that starts to play a game it used to abort
adds that game to its own denominator, usually at a low score, so *improving* can
lower the reported quality. Measured across the runs in this repo, correcting for
it moves the reported quality delta by up to ~18 points and flips its sign on many
checkpoints.

So this file reports both, and only ever subtracts the comparable one:

* **Headline (native)** -- clemeval's own numbers, verbatim, with the size of each
  denominator shown and **no delta**, because a difference of two means over
  different populations is not a difference.
* **Like-for-like** -- the same three statistics recomputed over the *fixed* set of
  games that every row in the table scored both components for
  (:func:`_common_games`), which is the largest population on which the rows are
  mutually comparable. Deltas are reported here, and only here.

The like-for-like set is usually smaller than 14 games. That is the price of a
delta that means what it says; the per-game table below still shows everything.

Stdlib only, so it runs in the training venv, on a login node, or on a laptop
after an rsync.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
import tempfile

# Columns clemeval writes into results.csv. The leading key is the model name.
CLEMSCORE_COL = "-, clemscore"
AVG_PLAYED_COL = "all, Average % Played"
AVG_QUALITY_COL = "all, Average Quality Score"
PLAYED_SUFFIX = ", % Played"
QUALITY_SUFFIX = ", Quality Score"


def _write_atomic(path: str, text: str) -> None:
    """Replace ``path`` with ``text`` in one step, never leaving it half-written.

    Same reasoning as ``summarize_eval.py``'s copy: gameplay evaluation is sharded
    into CONCURRENT jobs and each one runs this script when it finishes, so two
    summarizers can be writing PLAYPEN_RESULTS.md at the same moment. A plain
    ``open(path, "w")`` truncates first and fills in after, so the loser of that race
    can be read as an empty or half-built table. ``os.replace`` onto a temp file in
    the same directory is atomic on POSIX: a reader sees the old table or the new
    one, never a partial one.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".{}.".format(os.path.basename(path)))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _float(text: str | None) -> float | None:
    """A CSV cell as a number, or None for blank/NaN.

    Blank is how clemeval writes 'no episode produced this metric' and NaN is how
    pandas writes it after an all-aborted aggregation; both mean 'no score', and
    neither may be silently read as 0 -- 0 is a real, much better result.
    """
    if text is None:
        return None
    text = text.strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return None if value != value else value  # NaN


def _read_results_csv(path: str) -> dict[str, float | None]:
    """One clemeval results.csv -> {column: value}, taking the last model row.

    The file has one row per model evaluated into that directory. Each row here is
    written by its own single-model run, so there is normally exactly one; taking
    the last keeps a re-run's row rather than a stale one if a directory was reused.
    """
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except Exception:
        return {}
    if not rows:
        return {}
    row = rows[-1]
    return {k: _float(v) for k, v in row.items() if k}


def _scores(eval_subdir: str) -> tuple[dict[str, float | None], list[str]]:
    """All metrics for one checkpoint directory, plus the games it covers.

    Merges the clem and static suites when both were run; their column names do not
    collide (clemscore vs statscore, different game names), except for the shared
    'all, Average ...' pair, where clem wins because clemscore is the headline.
    """
    merged: dict[str, float | None] = {}
    for suite in ("static", "clem"):            # clem last so it wins on collisions
        path = os.path.join(eval_subdir, suite, "results.csv")
        if os.path.isfile(path):
            merged.update(_read_results_csv(path))

    # Fill in the aggregate from the val.json only where results.csv had nothing --
    # it is the same number by a less informative route.
    if merged.get(CLEMSCORE_COL) is None:
        for val in glob.glob(os.path.join(eval_subdir, "*.val.json")):
            try:
                with open(val, encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                continue
            for key, col in (("clemscore", CLEMSCORE_COL), ("statscore", "-, statscore")):
                value = data.get(key)
                if isinstance(value, (int, float)) and value == value and merged.get(col) is None:
                    merged[col] = float(value)

    # clemeval leaves clemscore blank when the quality aggregate was NaN. Recompute
    # it from its own definition when both parts survived, so a run that aborted on
    # some games still gets a headline number instead of a hole.
    if merged.get(CLEMSCORE_COL) is None:
        played, quality = merged.get(AVG_PLAYED_COL), merged.get(AVG_QUALITY_COL)
        if played is not None and quality is not None:
            merged[CLEMSCORE_COL] = played / 100.0 * quality

    games = sorted({
        col[: -len(PLAYED_SUFFIX)] for col in merged if col.endswith(PLAYED_SUFFIX)
    } | {
        col[: -len(QUALITY_SUFFIX)] for col in merged if col.endswith(QUALITY_SUFFIX)
    })
    return merged, games


def _scored_games(scores: dict[str, float | None]) -> set[str]:
    """Games this row has BOTH a ``% Played`` and a ``Quality Score`` for.

    Both are required: a game with a ``% Played`` but a blank ``Quality Score`` is
    one the model aborted entirely, and including it in a fixed population would
    mean averaging a hole.
    """
    played = {c[: -len(PLAYED_SUFFIX)] for c in scores
              if c.endswith(PLAYED_SUFFIX) and scores[c] is not None}
    quality = {c[: -len(QUALITY_SUFFIX)] for c in scores
               if c.endswith(QUALITY_SUFFIX) and scores[c] is not None}
    return played & quality


def _common_games(rows: list[tuple[str, dict[str, float | None]]]) -> list[str]:
    """The games EVERY row scored -- the population the deltas are computed on.

    The intersection, not the union: a mean is only comparable across rows when
    every row contributed the same games to it. Returns ``[]`` when the rows share
    no game, which is reported rather than papered over.
    """
    if not rows:
        return []
    common = _scored_games(rows[0][1])
    for _, scores in rows[1:]:
        common &= _scored_games(scores)
    return sorted(common)


def _fixed_population(
    scores: dict[str, float | None], games: list[str]
) -> tuple[float | None, float | None, float | None]:
    """``(clemscore, avg % played, avg quality)`` over a fixed set of games.

    Both means use the *same* denominator -- ``len(games)`` -- which is the whole
    point, and clemscore stays clemeval's own ``played / 100 * quality`` so the
    number is comparable in kind with the native one.
    """
    if not games:
        return None, None, None
    played = [scores.get(g + PLAYED_SUFFIX) for g in games]
    quality = [scores.get(g + QUALITY_SUFFIX) for g in games]
    if any(v is None for v in played) or any(v is None for v in quality):
        return None, None, None
    avg_played = sum(played) / len(games)
    avg_quality = sum(quality) / len(games)
    return avg_played / 100.0 * avg_quality, avg_played, avg_quality


def _sort_key(name: str) -> tuple[int, int]:
    """base first, then checkpoints by step number."""
    if name == "base":
        return (0, 0)
    try:
        return (1, int(name.rsplit("-", 1)[1]))
    except (IndexError, ValueError):
        return (2, 0)


def _cell(value: float | None, base: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    text = f"{value:.{digits}f}"
    if base is not None:
        text += f" ({value - base:+.{digits}f})"
    return text


def main() -> None:
    exp_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("EXP_DIR", "")
    if not exp_dir:
        print("usage: summarize_playpen_eval.py <EXP_DIR>", file=sys.stderr)
        raise SystemExit(2)

    eval_dir = os.path.join(exp_dir, "playpen-eval")
    if not os.path.isdir(eval_dir):
        print(f"[playpen-summary] no {eval_dir} -- nothing to summarize.")
        return

    names = sorted(
        (d for d in os.listdir(eval_dir)
         if os.path.isdir(os.path.join(eval_dir, d)) and not d.startswith(".")),
        key=_sort_key,
    )

    rows: list[tuple[str, dict[str, float | None]]] = []
    games: list[str] = []
    missing: list[str] = []
    for name in names:
        scores, row_games = _scores(os.path.join(eval_dir, name))
        if not scores:
            missing.append(name)
            continue
        rows.append((name, scores))
        for game in row_games:
            if game not in games:
                games.append(game)
    games.sort()

    if not rows:
        print(f"[playpen-summary] no results.csv under {eval_dir} -- nothing to summarize.")
        return

    exp_id = os.path.basename(exp_dir.rstrip("/"))

    # The fixed population every delta below is computed on. Derived from the rows
    # actually present, so a run whose eval is still in flight gets a table that is
    # internally consistent now and simply widens when the remaining shards land.
    common = _common_games(rows)
    fixed = {name: _fixed_population(scores, common) for name, scores in rows}
    base_fixed = fixed.get("base") if rows[0][0] == "base" else None
    # An empty common set makes every entry None, which is not a usable baseline --
    # collapse it so the "bracketed numbers" note cannot claim deltas that aren't there.
    if base_fixed is not None and base_fixed[0] is None:
        base_fixed = None

    def fixed_base_of(index: int, name: str) -> float | None:
        return base_fixed[index] if base_fixed and name != "base" else None

    md = [
        f"# Playpen game results: {exp_id}",
        "",
        "clembench gameplay on the `playpen-data` validation split, written by "
        "`experiments/lib/summarize_playpen_eval.py`. See `manifest.txt` for the model, "
        "the game this run was TRAINED on, and its hyperparameters; `RESULTS.md` holds "
        "the separate lm-eval scores.",
        "",
        "## Headline (clemeval native -- NOT comparable between rows)",
        "",
        "| checkpoint | clemscore | avg % played | avg quality | games scored | quality games |",
        "|---|---|---|---|---|---|",
    ]
    for name, scores in rows:
        n_played = len([c for c in scores if c.endswith(PLAYED_SUFFIX) and scores[c] is not None])
        n_quality = len([c for c in scores if c.endswith(QUALITY_SUFFIX) and scores[c] is not None])
        md.append("| `{}` | {} | {} | {} | {} | {} |".format(
            name,
            _cell(scores.get(CLEMSCORE_COL), None),
            _cell(scores.get(AVG_PLAYED_COL), None, 1),
            _cell(scores.get(AVG_QUALITY_COL), None, 1),
            n_played,
            n_quality,
        ))
    md += [
        "",
        "These are clemeval's own aggregates, verbatim. **No delta is shown, because "
        "these numbers are not comparable between rows.** `avg % played` averages over "
        "every game scored; `avg quality` averages over only the games that produced at "
        "least one parseable episode (`quality games`). When two rows differ in "
        "`quality games`, their `avg quality` -- and the `clemscore` built from it -- are "
        "means over different populations, so subtracting them measures the population "
        "change as much as the model. Use the like-for-like table below for that.",
    ]

    if common:
        md += [
            "",
            f"## Like-for-like (fixed population: the {len(common)} game(s) every row scored)",
            "",
            "| checkpoint | clemscore | avg % played | avg quality |",
            "|---|---|---|---|",
        ]
        for name, scores in rows:
            cs, played, quality = fixed[name]
            md.append("| `{}` | {} | {} | {} |".format(
                name,
                _cell(cs, fixed_base_of(0, name)),
                _cell(played, fixed_base_of(1, name), 1),
                _cell(quality, fixed_base_of(2, name), 1),
            ))
        md += [
            "",
            "Every cell here is a mean over the **same** games, so the bracketed deltas "
            "are like-for-like. `clemscore` is still clemeval's own definition "
            "(`% played / 100 x quality`), applied to the fixed set. Games in this set: "
            + ", ".join(f"`{g}`" for g in common) + ".",
        ]
        dropped = sorted(set(games) - set(common))
        if dropped:
            md += [
                "",
                "Excluded because at least one row has no quality score for them (the "
                "model aborted every episode of that game at that checkpoint): "
                + ", ".join(f"`{g}`" for g in dropped)
                + ". They are still in the per-game table below -- and a game moving in "
                "or out of this set is itself a result worth reading there.",
            ]
    else:
        md += [
            "",
            "## Like-for-like",
            "",
            "**Not available:** no single game was scored by every row, so there is no "
            "population on which these checkpoints can be compared. Read the per-game "
            "table below directly.",
        ]

    if games:
        md += [
            "",
            "## Per game -- `% played / quality score`",
            "",
            "`% played` is how often the model produced a parseable, rule-legal game; "
            "`quality score` is how well it did in those. A game at **0 % played** was "
            "never really attempted (a format failure), which is a different problem "
            "from playing properly and scoring 0.",
            "",
        ]
        header = ["checkpoint", *games]
        md.append("| " + " | ".join(header) + " |")
        md.append("|" + "|".join("---" for _ in header) + "|")
        for name, scores in rows:
            cells = [f"`{name}`"]
            for game in games:
                played = scores.get(game + PLAYED_SUFFIX)
                quality = scores.get(game + QUALITY_SUFFIX)
                if played is None and quality is None:
                    cells.append("-")
                    continue
                cells.append("{} / {}".format(
                    "-" if played is None else f"{played:.0f}",
                    "-" if quality is None else f"{quality:.1f}",
                ))
            md.append("| " + " | ".join(cells) + " |")

    if base_fixed:
        md += ["", "Bracketed numbers are the change from the untrained `base` row, on the "
                   "like-for-like population only. A checkpoint identical to `base` "
                   "everywhere means the adapter was not applied -- check the job log for "
                   "the `peft_model` line."]
    if missing:
        md += ["", "**Incomplete:** no results for "
                   + ", ".join(f"`{m}`" for m in missing)
                   + " (the run failed, was killed, or is still going). The like-for-like "
                     "population is the intersection over the rows that ARE present, so it "
                     "may widen when the remaining ones land."]
    md += ["", "A blank clemscore with `% played` at 0 is not a score of zero: every "
               "episode aborted, so there was nothing to score."]

    # TSV: the native columns first (unchanged names, so anything already reading this
    # file keeps working), then the like-for-like trio under distinct `fixed, ` names.
    FIXED_COLS = ["fixed, clemscore", "fixed, Average % Played", "fixed, Average Quality Score"]
    columns = [CLEMSCORE_COL, AVG_PLAYED_COL, AVG_QUALITY_COL] + FIXED_COLS + ["fixed, n games"]
    for game in games:
        columns += [game + PLAYED_SUFFIX, game + QUALITY_SUFFIX]

    _write_atomic(os.path.join(exp_dir, "PLAYPEN_RESULTS.md"), "\n".join(md) + "\n")

    tsv = ["\t".join(["checkpoint", *columns])]
    for name, scores in rows:
        cells = dict(scores)
        for col, value in zip(FIXED_COLS, fixed[name]):
            cells[col] = value
        cells["fixed, n games"] = float(len(common)) if common else 0.0
        tsv.append("\t".join(
            [name] + ["" if cells.get(c) is None else f"{cells[c]:.6f}" for c in columns]
        ))
    _write_atomic(os.path.join(exp_dir, "playpen_results.tsv"), "\n".join(tsv) + "\n")

    print(f"[playpen-summary] {len(rows)} row(s), {len(games)} game(s) "
          f"-> {exp_dir}/PLAYPEN_RESULTS.md and playpen_results.tsv")
    if missing:
        print(f"[playpen-summary] no results for: {', '.join(missing)}")


if __name__ == "__main__":
    main()

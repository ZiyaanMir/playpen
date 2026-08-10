"""Record the scheduler job ids a submission just queued, in the experiment manifest.

Called by every script that queues jobs -- ``run_experiment.sh``,
``resume_experiment.sh``, ``run_eval.sh``, ``run_playpen_eval.sh`` -- immediately
AFTER submission, through ``exp_record_jobs`` in ``experiments/lib/experiment.sh``.

``write_manifest.py`` cannot do this itself. It runs BEFORE anything is submitted --
deliberately, so an experiment that dies in the queue still has a record of what it
was meant to be -- and at that moment no job id exists yet. So the ids are added in a
second pass, here:

* ``manifest.json`` gains a ``jobs`` object: ``ids`` (every id, in submission order)
  and ``submissions`` (one record per submitting invocation). A resume, or a re-run of
  the gameplay eval, APPENDS a record rather than replacing the first one, so the
  directory keeps the whole history of what was queued against it.
* ``manifest.txt`` gains the same thing as a readable block, with the cancel command.

WRITTEN BY THE SUBMITTER, NOT BY THE JOBS. The submitter is one process on a login
node, so this cannot race; six jobs rewriting one JSON file from six compute nodes
would. The jobs still append their own runtime facts (host, GPU, the id the scheduler
actually gave them) to ``manifest.txt``, which is append-only and therefore safe.

Never fatal by design: the jobs are already in the queue by the time this runs, so a
manifest that could not be updated must not turn a successful submission into a failed
script. An unreadable ``manifest.json`` is reported and left ALONE rather than
overwritten -- half a record is worth more than a fresh empty one.

stdlib only, so it costs nothing on a login node.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

# What to call the job in manifest.txt, per role. The JSON keeps the machine-friendly
# spelling ("lm_eval"); this is only the rendering.
ROLE_LABELS = {
    "train": "train",
    "lm_eval": "lm-eval",
    "playpen_eval": "playpen",
    "summary": "summary",
}

# How each scheduler cancels a list of job ids.
CANCEL_COMMANDS = {"sge": "qdel", "slurm": "scancel"}


def _count(raw: str) -> int:
    """An int, or 0 when the shell handed over an empty or malformed value.

    argparse's own ``type=int`` would abort the whole recording over one unset count,
    and losing every id in a submission because "of 3" could not be worked out is a
    bad trade. A count of 0 simply drops the "of N" part of that line.
    """
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def parse_args(argv: list) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Append the job ids of one submission to an experiment manifest.",
    )
    p.add_argument("exp_dir", help="the experiment directory holding manifest.json")
    p.add_argument("--submitted-by", default="?",
                   help="which script queued these (e.g. eddie/run_experiment.sh)")
    p.add_argument("--cluster", default="", help="EXP_CLUSTER, for the record")
    p.add_argument("--scheduler", default="",
                   help="sge | slurm -- decides the cancel command that is printed")
    p.add_argument("--env-file", default="",
                   help="the env file these jobs read (experiment.env, "
                        "experiment.resume.env, ...): which settings they ran under")
    p.add_argument("--train", nargs="*", default=[], metavar="ID",
                   help="training job ids, in segment order")
    p.add_argument("--train-first-segment", type=_count, default=1,
                   help="segment number of the first --train id (a resume starts "
                        "part-way through the chain)")
    p.add_argument("--train-total", type=_count, default=0,
                   help="TRAIN_SEGMENTS: how many segments the whole run has")
    p.add_argument("--eval", nargs="*", default=[], metavar="ID",
                   help="lm-eval job ids, in shard order")
    p.add_argument("--playpen", nargs="*", default=[], metavar="ID",
                   help="gameplay (playpen) eval job ids, in shard order")
    p.add_argument("--summary", nargs="*", default=[], metavar="ID",
                   help="summary job id(s)")
    p.add_argument("--shard-total", type=_count, default=0,
                   help="EVAL_SHARD_TOTAL: how many shards each evaluation was split "
                        "into")
    return p.parse_args(argv)


def collect_jobs(args: argparse.Namespace) -> list:
    """The submitted jobs as manifest records, in submission order.

    The index each job carries is the one the SUBMITTER used, not its position in this
    list: training segments start at ``--train-first-segment`` (a resume queues
    segments 2..3 of 3), and both evaluations are shard 1..N of the same N. That is
    what makes an id here findable in ``qstat``/``squeue`` output and in the log file
    names, which are built from the same indices.
    """
    jobs = []
    first_segment = args.train_first_segment if args.train_first_segment > 0 else 1
    for offset, job_id in enumerate(args.train):
        entry = {
            "role": "train",
            "job_id": job_id,
            "segment": first_segment + offset,
        }
        if args.train_total > 0:
            entry["of"] = args.train_total
        jobs.append(entry)
    for role, ids in (("lm_eval", args.eval), ("playpen_eval", args.playpen)):
        for offset, job_id in enumerate(ids):
            entry = {"role": role, "job_id": job_id, "shard": offset + 1}
            if args.shard_total > 0:
                entry["of"] = args.shard_total
            jobs.append(entry)
    for job_id in args.summary:
        jobs.append({"role": "summary", "job_id": job_id})
    return jobs


def _describe(job: dict) -> str:
    """"segment 2 of 3" / "shard 1 of 2" / "" -- the index part of a manifest.txt line."""
    for key in ("segment", "shard"):
        if key in job:
            of = job.get("of")
            return f"{key} {job[key]}" + (f" of {of}" if of else "")
    return ""


def text_block(record: dict) -> str:
    """The manifest.txt rendering of one submission.

    Same two-column shape as the rest of manifest.txt, so the file still reads as one
    document once the jobs append their runtime blocks after it.
    """
    head = f"-- jobs queued {record['submitted_at']} by {record['submitted_by']} "
    lines = ["", head + "-" * max(4, 70 - len(head))]
    for job in record["jobs"]:
        label = ROLE_LABELS.get(job["role"], job["role"])
        lines.append(f"  {label:<28} {job['job_id']}  {_describe(job)}".rstrip())
    if record.get("env_file"):
        lines.append(f"  {'settings from':<28} {record['env_file']}")
    if record.get("cancel_command"):
        lines.append(f"  {'cancel':<28} {record['cancel_command']}")
    # A glob rather than a per-job path: every log file this repo writes ends in
    # _<job id>.out / .err, whatever the prefix, so this stays true if the naming of
    # the prefixes ever changes.
    lines.append(f"  {'logs':<28} logs/*_<job id>.out")
    return "\n".join(lines) + "\n"


def _load_manifest(path: str) -> tuple:
    """``(manifest, note)`` -- the parsed manifest.json, or None with the reason why.

    Missing and unreadable are both non-fatal and both mean "do not write JSON": an
    experiment predating the manifest, or one whose submission died before writing it,
    still gets its job ids recorded in manifest.txt.
    """
    try:
        with open(path) as fh:
            manifest = json.load(fh)
    except FileNotFoundError:
        return None, f"no {os.path.basename(path)} in this experiment directory"
    except (OSError, ValueError) as exc:
        return None, f"could not read {os.path.basename(path)} ({exc})"
    if not isinstance(manifest, dict):
        return None, f"{os.path.basename(path)} is not a JSON object"
    return manifest, ""


def add_submission(manifest: dict, record: dict) -> dict:
    """Append ``record`` to the manifest's ``jobs`` section, and refresh the flat list.

    ``ids`` is derived from ``submissions`` on every write rather than appended to
    separately, so the two cannot drift apart. Duplicates are dropped (order kept) in
    case the same submission is ever recorded twice.
    """
    jobs = manifest.get("jobs")
    if not isinstance(jobs, dict):
        jobs = {}
    submissions = jobs.get("submissions")
    if not isinstance(submissions, list):
        submissions = []
    submissions.append(record)

    ids, seen = [], set()
    for submission in submissions:
        for job in submission.get("jobs", []):
            job_id = job.get("job_id")
            if job_id and job_id not in seen:
                seen.add(job_id)
                ids.append(job_id)

    manifest["jobs"] = {"ids": ids, "submissions": submissions}
    return manifest


def _write_json(path: str, manifest: dict) -> None:
    """Replace manifest.json atomically -- a crash mid-write must not lose the record."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, path)


def main() -> None:
    args = parse_args(sys.argv[1:])
    jobs = collect_jobs(args)
    if not jobs:
        print("[jobs] nothing submitted to record.")
        return

    env_file = args.env_file
    if env_file and os.path.dirname(os.path.abspath(env_file)) == os.path.abspath(args.exp_dir):
        env_file = os.path.basename(env_file)   # it lives here; the name is enough

    record = {
        "submitted_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "submitted_by": args.submitted_by,
        "cluster": args.cluster or None,
        "scheduler": args.scheduler or None,
        "env_file": env_file or None,
        "jobs": jobs,
    }
    cancel = CANCEL_COMMANDS.get(args.scheduler)
    if cancel:
        record["cancel_command"] = cancel + " " + " ".join(j["job_id"] for j in jobs)

    with open(os.path.join(args.exp_dir, "manifest.txt"), "a") as fh:
        fh.write(text_block(record))

    json_path = os.path.join(args.exp_dir, "manifest.json")
    manifest, note = _load_manifest(json_path)
    if manifest is None:
        print(f"[jobs] recorded {len(jobs)} job id(s) in manifest.txt; "
              f"manifest.json left untouched ({note}).")
        return
    _write_json(json_path, add_submission(manifest, record))
    print(f"[jobs] recorded {len(jobs)} job id(s) in manifest.json and manifest.txt")


if __name__ == "__main__":
    main()

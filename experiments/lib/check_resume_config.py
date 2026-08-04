"""Refuse to resume a run under a different algorithm config than it started with.

Called by ``experiments/*/resume_experiment.sh`` before it submits anything. Exits 0
when the config the resumed segments will use matches the one ``manifest.json``
recorded at first submission, and 1 (listing the differing fields) when it does not.

WHY THIS EXISTS. Resuming re-derives the config from the environment, and two things
make that environment different months later:

* **The shared YAML is edited between runs.** ``experiment.env`` stores MARSHAL_CONFIG
  as a *path* (``examples/marshal/marshal_config.yaml``), not as content. Measured on a
  real run: resuming ``taboo_Qwen3-4B_turnrew_20260730-231934`` against today's shared
  YAML would flip ``turn_level_rewards`` true->false and ``whiten_rewards`` true->false.
  ``resume_experiment.sh`` now points at the frozen per-run copy, which fixes this --
  and this check is what proves it worked rather than assuming it.
* **A setting reached the original job by a route the resume does not have.** On
  Isambard ``sbatch --export=ALL`` carries the submitter's whole environment, so
  ``TR_ENABLE=1`` reached the job without ever being written to ``experiment.env``.
  Nothing carries it on a resume, so the remaining segments would train with turn
  rewards off. No amount of care in the resume script can fix that one; the variable
  has to be in the env-file whitelist. This check turns it from silent into loud.

Either way the failure mode is the same and it is the bad kind: the run keeps going,
the checkpoints look fine, and half of it trained under a different algorithm than the
other half with nothing on disk saying so.

Deliberately importable without torch: it reuses ``write_manifest.resolve_config`` --
the same resolution the manifest itself was written with, so the two answers cannot
drift apart the way two reimplementations would.
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from write_manifest import resolve_config  # noqa: E402

# Fields whose value legitimately depends on the machine rather than the experiment.
# None today; kept as the place to put one, so a future exemption arrives with a
# reason next to it instead of as a silent hole in the comparison.
IGNORED_FIELDS: frozenset = frozenset()


def compare(recorded: dict, current: dict) -> tuple[list, list]:
    """``(drifted, added)`` -- fields that changed, and fields that are simply new.

    The two are separated because only one of them is a problem.

    A field in ``current`` but not in ``recorded`` is a field that did not EXIST when
    the run was submitted -- ``MarshalConfig`` gains fields over time, and
    ``marshal_exact_unique_pooling`` is one that post-dates runs still on disk. There
    is no old value for it to have drifted from, and the new default is by
    construction the behaviour the old code had. Refusing on those would fire on every
    older experiment and teach the operator to reach for RESUME_FORCE=1, which is
    exactly the habit this check exists to avoid. They are reported as a note instead.

    A field in ``recorded`` but not in ``current`` is the opposite case and IS a
    problem: the run was configured with something this code no longer has.
    """
    drifted, added = [], []
    for key in sorted(set(recorded) | set(current)):
        if key in IGNORED_FIELDS:
            continue
        if key not in recorded:
            added.append((key, current[key]))
        elif recorded[key] != current.get(key, "<absent>"):
            drifted.append((key, recorded[key], current.get(key, "<absent>")))
    return drifted, added


def main() -> None:
    exp_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("EXP_DIR", "")
    if not exp_dir:
        print("usage: check_resume_config.py <EXP_DIR>", file=sys.stderr)
        raise SystemExit(2)

    manifest_path = os.path.join(exp_dir, "manifest.json")
    try:
        with open(manifest_path) as fh:
            recorded = json.load(fh).get("marshal_config") or {}
    except (OSError, ValueError) as exc:
        # Not fatal: an experiment predating the manifest, or one whose submission
        # died before writing it, can still be resumed -- there is simply nothing to
        # check it against, and saying so is more useful than refusing.
        print(f"[config-check] SKIPPED: cannot read {manifest_path} ({exc}).")
        print("[config-check]          Nothing to compare against; resuming unchecked.")
        return
    if not recorded:
        print(f"[config-check] SKIPPED: {manifest_path} records no marshal_config.")
        return

    current, error = resolve_config()
    if error:
        print(f"[config-check] FAILED to resolve the config for this resume: {error}",
              file=sys.stderr)
        raise SystemExit(1)

    diffs, added = compare(recorded, current)
    for field, value in added:
        print(f"[config-check] note: '{field}' did not exist when this run was "
              f"submitted; it defaults to {value}.")
    if not diffs:
        print(f"[config-check] OK: all {len(recorded)} MARSHAL settings match the ones "
              f"manifest.json recorded at first submission.")
        return

    print("", file=sys.stderr)
    print("[config-check] REFUSING TO RESUME -- the algorithm config has changed.",
          file=sys.stderr)
    print("", file=sys.stderr)
    print(f"  {'field':<32} {'was (manifest.json)':<24} {'would be now'}", file=sys.stderr)
    print(f"  {'-' * 32} {'-' * 24} {'-' * 20}", file=sys.stderr)
    for field, was, now in diffs:
        print(f"  {field:<32} {str(was):<24} {now}", file=sys.stderr)
    print("", file=sys.stderr)
    print("  Resuming would train the rest of this run under a DIFFERENT algorithm",
          file=sys.stderr)
    print("  than its first half, inside one continuous set of checkpoints that would",
          file=sys.stderr)
    print("  record nothing about it. The usual causes:", file=sys.stderr)
    print("", file=sys.stderr)
    print("    * a setting that reached the original job through the environment but is",
          file=sys.stderr)
    print("      not in the env-file whitelist in <cluster>/run_experiment.sh, so the",
          file=sys.stderr)
    print("      resume has no way to know about it (TR_* is the known case);",
          file=sys.stderr)
    print("    * EXTRA_TRAIN_ARGS set for the original submission and not stored;",
          file=sys.stderr)
    print("    * an edit to the shared marshal_config.yaml (should be handled -- the",
          file=sys.stderr)
    print("      resume pins the frozen per-run copy -- so report this one).",
          file=sys.stderr)
    print("", file=sys.stderr)
    print("  Fix it by putting the missing value back in the environment for this",
          file=sys.stderr)
    print("  submission, e.g.  TR_ENABLE=1 experiments/<cluster>/resume_experiment.sh ...",
          file=sys.stderr)
    print("  Or, if the change is deliberate and you accept a run trained two ways:",
          file=sys.stderr)
    print("  RESUME_FORCE=1 to proceed anyway.", file=sys.stderr)
    print("", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()

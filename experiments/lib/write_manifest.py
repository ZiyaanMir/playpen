"""Write an experiment manifest: what was run, with what, against which code.

Called once by ``run_experiment.sh`` at SUBMIT time (before the job starts), so a
queued or failed experiment still has a readable record of what it was meant to be.
Reads its inputs from the environment the submitter already exported, and writes
two files into ``EXP_DIR``:

* ``manifest.json`` -- machine-readable, for scripted comparison across runs.
* ``manifest.txt``  -- the same content laid out for ``cat``.

The important part is the **resolved** MARSHAL config: the YAML is merged with the
CLI length-penalty overrides through the same ``dataclasses.replace`` path that
``train_selfplay.py`` uses, so the manifest records the values that will actually
take effect rather than what the YAML file happened to say. A frozen copy of the
YAML is saved alongside, so editing the shared config later never rewrites history.

Deliberately importable without torch/trl/vllm: it only needs ``playpen.marshal.config``
(stdlib + PyYAML) and reads package versions from installed metadata rather than by
importing them, so it runs fast on a login node.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, REPO)

from playpen.marshal.config import MarshalConfig  # noqa: E402

# Training knobs the presets set. Kept as (env var, manifest key, caster) so the
# manifest is typed -- `max_steps` as an int compares/sorts across runs, a string
# does not.
TRAIN_FIELDS = [
    ("MODEL", "model", str),
    ("GAME", "game", str),
    ("NUM_GENERATIONS", "num_generations", int),
    ("PER_DEVICE_BATCH", "per_device_batch_size", int),
    ("GRAD_ACCUM", "grad_accum", int),
    ("MAX_STEPS", "max_steps", int),
    ("SAVE_STEPS", "save_steps", int),
    ("LEARNING_RATE", "learning_rate", float),
    ("KL_BETA", "kl_beta", float),
    ("MAX_COMPLETION_LENGTH", "max_completion_length", int),
    ("MAX_TURNS", "max_turns", int),
    ("GRAD_CKPT", "gradient_checkpointing", lambda v: bool(int(v))),
    ("VLLM_UTIL", "vllm_gpu_memory_utilization", float),
    ("VLLM_MAX_MODEL_LEN", "vllm_max_model_len", int),
]

PACKAGES = ["torch", "trl", "peft", "transformers", "vllm", "datasets", "accelerate"]


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _git(*args: str) -> str:
    """Run a git command in the repo, returning '' rather than raising."""
    try:
        out = subprocess.run(
            ["git", "-C", REPO, *args],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _versions() -> dict:
    """Installed versions, read from metadata (no imports -- fast, no GPU needed)."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return {}
    out = {}
    for pkg in PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
        except Exception:
            out[pkg] = None
    return out


def _resolved_marshal_config(exp_dir: str) -> tuple[dict, str | None]:
    """Merge the YAML with the CLI length-penalty overrides, exactly as training does.

    Returns ``(config_dict, error)``. A malformed YAML is reported rather than
    raised: the manifest is diagnostic, and failing to write it must not block a
    submission (the training job will fail on the same file soon enough, loudly).
    """
    rel = _env("MARSHAL_CONFIG", "examples/marshal/marshal_config.yaml")
    path = rel if os.path.isabs(rel) else os.path.join(REPO, rel)
    try:
        cfg = MarshalConfig.from_yaml(path)
    except Exception as exc:
        return {}, f"could not load {path}: {exc}"

    overrides = {}
    if _env("LP_MAX_LEN"):
        overrides["length_penalty_max_len"] = int(_env("LP_MAX_LEN"))
    if _env("LP_COEF"):
        overrides["length_penalty_coef"] = float(_env("LP_COEF"))

    # EXTRA_TRAIN_ARGS is how ablation arms are expressed (EXTRA_TRAIN_ARGS=
    # --no-length-penalty). Those flags override the YAML at train time, so a
    # manifest that ignored them would report the opposite of what ran -- exactly
    # the field an ablation turns on. Only the on/off switches are parsed here;
    # valued flags stay recorded verbatim under training.extra_train_args.
    extra = _env("EXTRA_TRAIN_ARGS").split()
    for flag, field, value in [
        ("--no-length-penalty", "length_penalty", False),
        ("--length-penalty", "length_penalty", True),
        ("--no-marshal", "enabled", False),
        ("--marshal", "enabled", True),
        ("--no-dr-grpo", "dr_grpo", False),
        ("--dr-grpo", "dr_grpo", True),
    ]:
        if flag in extra:
            overrides[field] = value
    for i, token in enumerate(extra):
        # Valued length-penalty flags, so the recorded numbers stay truthful too.
        if token in ("--length-penalty-coef", "--length-penalty-max-len",
                     "--length-penalty-min-len", "--length-penalty-bonus",
                     "--length-penalty-offset") and i + 1 < len(extra):
            field = "length_penalty_" + token[len("--length-penalty-"):].replace("-", "_")
            caster = int if field.endswith(("_len",)) else float
            try:
                overrides[field] = caster(extra[i + 1])
            except ValueError:
                pass

    if overrides:
        try:
            cfg = dataclasses.replace(cfg, **overrides)
        except Exception as exc:
            return cfg.to_dict(), f"length-penalty override rejected: {exc}"

    # Freeze the config next to the results so a later edit to the shared YAML
    # cannot silently rewrite what this run claims to have used.
    try:
        shutil.copy2(path, os.path.join(exp_dir, "marshal_config.yaml"))
    except Exception:
        pass
    return cfg.to_dict(), None


def _length_penalty_summary(cfg: dict) -> dict:
    """What the penalty is actually worth, in reward units, for this run.

    A penalty is easy to enable and hard to notice doing nothing, so the manifest
    records the two numbers that make it obvious: the worst a single turn can
    score (generation is capped at max_completion_length, so this is the maximum
    reachable, not a theoretical limit) and the per-episode total across all turns.
    """
    if not cfg.get("length_penalty"):
        return {"enabled": False}
    if not cfg.get("enabled"):
        # The penalty is applied inside compute_marshal_advantages, which the plain-GRPO
        # path never calls. Reporting its nominal magnitude here would overstate what
        # this run does. train_selfplay.py prints the same warning at startup.
        return {
            "enabled": True,
            "effective": False,
            "note": "MARSHAL advantage path is off (enabled=false), so the length "
                    "penalty has NO effect on this run.",
        }
    try:
        from playpen.marshal.advantage import LengthPenaltySpec
    except Exception:
        return {"enabled": True, "note": "torch unavailable at submit time"}

    spec = LengthPenaltySpec(
        coef=cfg["length_penalty_coef"], bonus=cfg["length_penalty_bonus"],
        min_len=cfg["length_penalty_min_len"], max_len=cfg["length_penalty_max_len"],
        offset=cfg["length_penalty_offset"],
    )
    try:
        cap = int(_env("MAX_COMPLETION_LENGTH", "0"))
    except ValueError:
        cap = 0
    if cap <= 0:
        return {"enabled": True}
    per_turn = spec.penalty_for(cap)
    # Turns per seat is the game's own round limit, which MAX_TURNS only bounds.
    turns = {"dond": 5, "guesswhat": 8, "taboo": 3}.get(_env("GAME"), 0)
    out = {
        "enabled": True,
        "max_reachable_per_turn": round(per_turn, 5),
        "per_turn_generation_cap": cap,
    }
    if turns:
        out["est_per_episode_total"] = round(per_turn * turns, 5)
        out["est_turns_per_seat"] = turns
    return out


def main() -> None:
    exp_dir = sys.argv[1] if len(sys.argv) > 1 else _env("EXP_DIR")
    if not exp_dir:
        print("usage: write_manifest.py <EXP_DIR>   (or set $EXP_DIR)", file=sys.stderr)
        raise SystemExit(2)
    os.makedirs(exp_dir, exist_ok=True)

    train = {}
    for env_name, key, cast in TRAIN_FIELDS:
        raw = _env(env_name)
        if raw == "":
            continue
        try:
            train[key] = cast(raw)
        except (ValueError, TypeError):
            train[key] = raw  # keep the raw string rather than lose the record
    if _env("EXTRA_TRAIN_ARGS"):
        train["extra_train_args"] = _env("EXTRA_TRAIN_ARGS")

    marshal_cfg, cfg_error = _resolved_marshal_config(exp_dir)

    manifest = {
        "experiment_id": _env("EXP_ID"),
        "tag": _env("EXP_TAG") or None,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "cluster": _env("EXP_CLUSTER"),
        "submitted_from": socket.gethostname(),
        "experiment_dir": exp_dir,
        "training": train,
        "marshal_config": marshal_cfg,
        "marshal_config_source": _env("MARSHAL_CONFIG"),
        "length_penalty_effect": _length_penalty_summary(marshal_cfg),
        "evaluation": {
            "tasks": _env("EVAL_TASKS"),
            "batch_size": _env("EVAL_BATCH"),
            "eval_base_model": _env("EVAL_BASE") == "1",
            "limit": _env("EVAL_LIMIT") or None,
            "extra": _env("EVAL_EXTRA") or None,
        },
        "code": {
            "repo": REPO,
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
        },
        "packages": _versions(),
    }
    if cfg_error:
        manifest["marshal_config_error"] = cfg_error

    with open(os.path.join(exp_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")

    lines = [
        f"experiment : {manifest['experiment_id']}",
        f"created    : {manifest['created_at']}",
        f"cluster    : {manifest['cluster']}",
        f"tag        : {manifest['tag'] or '-'}",
        "",
        "-- training ---------------------------------------------------------",
    ]
    lines += [f"  {k:<28} {v}" for k, v in train.items()]
    lines += ["", "-- marshal config (resolved: YAML + CLI overrides) -------------------"]
    lines += [f"  {k:<28} {v}" for k, v in marshal_cfg.items()]
    if cfg_error:
        lines.append(f"  !! {cfg_error}")

    lpe = manifest["length_penalty_effect"]
    if lpe.get("enabled") and lpe.get("effective") is False:
        lines += [
            "",
            "-- length penalty ---------------------------------------------------",
            f"  configured ON, but INACTIVE: {lpe['note']}",
        ]
    elif lpe.get("enabled") and "max_reachable_per_turn" in lpe:
        lines += [
            "",
            "-- length penalty, in reward units (game outcome is +1 / 0 / -1) -----",
            f"  worst a single turn can score  {lpe['max_reachable_per_turn']:+.4f}"
            f"   (turns are capped at {lpe['per_turn_generation_cap']} tokens)",
        ]
        if "est_per_episode_total" in lpe:
            lines.append(
                f"  estimated per-episode total    {lpe['est_per_episode_total']:+.4f}"
                f"   (~{lpe['est_turns_per_seat']} turns/seat)"
            )
            total = lpe["est_per_episode_total"]
            if abs(total) < 0.05:
                lines.append(
                    "  WARNING: that is ~0 against a +-1 outcome -- the penalty is "
                    "effectively inert.\n"
                    "           Lower LP_MAX_LEN (rule of thumb: max_completion_length/2) "
                    "or raise LP_COEF."
                )
            elif abs(total) > 1.0:
                lines.append(
                    "  WARNING: that EXCEEDS the +-1 game outcome -- staying silent can "
                    "now beat winning.\n"
                    "           Raise LP_MAX_LEN or lower LP_COEF so the episode total "
                    "stays under ~0.5."
                )

    lines += [
        "",
        "-- evaluation -------------------------------------------------------",
        f"  tasks                        {manifest['evaluation']['tasks']}",
        f"  batch_size                   {manifest['evaluation']['batch_size']}",
        f"  eval base model too          {manifest['evaluation']['eval_base_model']}",
        "",
        "-- code -------------------------------------------------------------",
        f"  commit                       {manifest['code']['git_commit'][:12] or '?'}"
        f"{'  (DIRTY working tree)' if manifest['code']['git_dirty'] else ''}",
        f"  branch                       {manifest['code']['git_branch'] or '?'}",
        "",
        "-- packages ---------------------------------------------------------",
    ]
    lines += [f"  {k:<28} {v or '-'}" for k, v in manifest["packages"].items()]

    with open(os.path.join(exp_dir, "manifest.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"[manifest] wrote {exp_dir}/manifest.json and manifest.txt")
    if manifest["code"]["git_dirty"]:
        print("[manifest] NOTE: working tree is dirty -- the commit alone will not "
              "reproduce this run.")
    if lpe.get("enabled") and "est_per_episode_total" in lpe:
        total = lpe["est_per_episode_total"]
        if abs(total) < 0.05:
            print(f"[manifest] WARNING: length penalty is effectively inert "
                  f"({total:+.3f} per episode vs a +-1 outcome). See manifest.txt.")
        elif abs(total) > 1.0:
            print(f"[manifest] WARNING: length penalty ({total:+.3f} per episode) exceeds "
                  f"the +-1 game outcome -- silence can beat winning. See manifest.txt.")


if __name__ == "__main__":
    main()

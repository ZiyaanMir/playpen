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
    # How the run was split across jobs. Recorded because it is the one thing about a
    # chained run that is NOT visible from its checkpoints: segments partition
    # max_steps rather than extending it, and every segment trains under the same
    # --max-steps (hence the same LR schedule), so a 3-segment run and a 1-segment run
    # of the same config produce indistinguishable output directories. When a run does
    # turn out odd around step 400, this is what says a job boundary was there.
    ("TRAIN_SEGMENTS", "train_segments", int),
    ("SEGMENT_STEPS", "segment_steps", int),
    ("RESUME_FROM", "resume_from", str),
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


def _marshal_flag_spec() -> dict:
    """``{cli flag: (field, caster_or_bool_value)}``, derived from MarshalConfig itself.

    Mirrors ``train_selfplay.py``'s naming convention -- ``field_name`` becomes
    ``--field-name``, bools additionally get ``--no-field-name``, and ``enabled`` keeps
    its historical ``--marshal`` / ``--no-marshal`` spelling. Deriving this from the
    dataclass rather than hardcoding a list means a config field added later is picked
    up here automatically, instead of being silently misreported in every manifest.

    ``bool`` is checked before ``int`` because ``isinstance(True, int)`` is True; and
    the *default value's* type is used rather than ``field.type``, which is a plain
    string under ``from __future__ import annotations``.
    """
    spec: dict = {}
    for f in dataclasses.fields(MarshalConfig):
        base = "--marshal" if f.name == "enabled" else "--" + f.name.replace("_", "-")
        if isinstance(f.default, bool):
            spec[base] = (f.name, True)
            spec["--no-" + base[2:]] = (f.name, False)
        else:
            spec[base] = (f.name, type(f.default))
    return spec


def _marshal_overrides_from_argv(tokens: list) -> dict:
    """Extract MarshalConfig overrides from a list of CLI tokens.

    Matching is on whole tokens, so ``--length-penalty`` (bool) is never confused with
    ``--length-penalty-coef`` (valued), nor ``--marshal`` with ``--marshal-config``.
    Unparseable values are skipped rather than raised: the manifest is diagnostic, and
    the training job will reject a bad value loudly on its own.
    """
    spec = _marshal_flag_spec()
    out: dict = {}
    i = 0
    while i < len(tokens):
        entry = spec.get(tokens[i])
        if entry is not None:
            field, kind = entry
            if isinstance(kind, bool):
                out[field] = kind
            elif i + 1 < len(tokens):
                try:
                    out[field] = kind(tokens[i + 1])
                except (TypeError, ValueError):
                    pass
                i += 1
        i += 1

    # Convenience ALIASES: flags that train_selfplay.py maps onto config fields but
    # which are not fields themselves, so the derived spec above cannot see them.
    # Without this the manifest would report the YAML's sampling values for a run that
    # actually neutralised them -- the exact class of lie this parser exists to avoid.
    # `setdefault` mirrors train_selfplay.py: an explicit --sampling-top-* still wins.
    known = {f.name for f in dataclasses.fields(MarshalConfig)}
    if "--no-sampling-truncation" in tokens:
        if "sampling_top_p" in known:
            out.setdefault("sampling_top_p", 1.0)
        if "sampling_top_k" in known:
            out.setdefault("sampling_top_k", 0)
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

    # marshal_exact's distinct-value pooling, tri-state like TR_ENABLE below: "" leaves
    # the YAML alone. Same reason it must be handled here -- train.sh turns it into a
    # CLI flag directly, so the EXTRA_TRAIN_ARGS parser never sees it.
    if _env("UNIQUE_POOL") in ("0", "1"):
        overrides["marshal_exact_unique_pooling"] = _env("UNIQUE_POOL") == "1"

    # Dense per-turn rewards, same story as LP_*: train.sh turns these env vars into
    # CLI flags directly (not via EXTRA_TRAIN_ARGS), so the argv parser below cannot
    # see them and the manifest would otherwise report the YAML's values for a run
    # that overrode them. TR_ENABLE is tri-state: "" leaves the YAML alone.
    if _env("TR_ENABLE") in ("0", "1"):
        overrides["turn_rewards"] = _env("TR_ENABLE") == "1"
    if _env("TR_SOURCE"):
        overrides["turn_reward_source"] = _env("TR_SOURCE")
    if _env("TR_SCALE"):
        overrides["turn_reward_scale"] = float(_env("TR_SCALE"))
    if _env("TR_BUDGET"):
        overrides["turn_reward_budget"] = float(_env("TR_BUDGET"))
    if _env("TR_COMPONENTS"):
        overrides["turn_reward_components"] = _env("TR_COMPONENTS")

    # EXTRA_TRAIN_ARGS is how a run varies the algorithm (EXTRA_TRAIN_ARGS=
    # "--no-length-penalty --gamma 0.95"). Those flags override the YAML at train
    # time, so a manifest that ignored them would report the OPPOSITE of what ran --
    # exactly the fields an ablation is defined by.
    overrides.update(_marshal_overrides_from_argv(_env("EXTRA_TRAIN_ARGS").split()))

    if overrides:
        try:
            cfg = dataclasses.replace(cfg, **overrides)
        except Exception as exc:
            return cfg.to_dict(), f"config override rejected: {exc}"

    # Freeze the config next to the results so a later edit to the shared YAML
    # cannot silently rewrite what this run claims to have used.
    try:
        shutil.copy2(path, os.path.join(exp_dir, "marshal_config.yaml"))
    except Exception:
        pass
    return cfg.to_dict(), None


def _wandb_summary() -> dict:
    """What the training job will do about W&B, from the WB_* the submitter exported.

    Recorded at submit time so the manifest names the project and run before the job
    starts -- and so an experiment directory found months later says where its curves
    live even if the W&B run itself was never synced. The trainer writes the run's
    actual id/url into ``train/wandb_run.json`` once it opens.
    """
    enabled = _env("WB_ENABLE", "1") == "1" and _env("WB_MODE", "auto") != "disabled"
    if not enabled:
        return {"enabled": False}
    return {
        "enabled": True,
        "project": _env("WB_PROJECT", "playpen-marshal"),
        "entity": _env("WB_ENTITY") or None,
        "group": _env("WB_GROUP") or None,   # None => trainer derives {game}_{model}
        "tags": _env("WB_TAGS") or None,
        "mode": _env("WB_MODE", "auto"),
        # The run is named after the experiment, which is what ties a W&B row to
        # this directory.
        "run_name": _env("EXP_ID") or None,
        "dir": os.path.join(_env("EXP_DIR"), "wandb") if _env("EXP_DIR") else None,
    }


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


def _turn_reward_summary(cfg: dict) -> dict:
    """What the dense per-turn reward channel is worth, in reward units.

    Same motivation as :func:`_length_penalty_summary` -- a reward term that is easy
    to enable and hard to notice doing nothing -- but the numbers here are analytic
    rather than estimated: extractors normalize every component to ``[-1, 1]``, so a
    turn is worth at most ``turn_reward_scale`` and an episode at most
    ``turn_reward_budget``, whatever the game or the turn count turns out to be.
    """
    if not cfg.get("turn_rewards"):
        return {"enabled": False}
    out = {
        "enabled": True,
        "source": cfg.get("turn_reward_source"),
        "max_per_turn": round(float(cfg.get("turn_reward_scale", 0.0)), 5),
        "max_per_episode": round(float(cfg.get("turn_reward_budget", 0.0)), 5),
        "components": cfg.get("turn_reward_components") or "<all>",
    }
    try:
        from playpen.marshal.turn_rewards import resolve_extractor_class

        cls = resolve_extractor_class(_env("GAME"))
        out["extractor"] = cls.__name__ if cls else "FormatComplianceExtractor (fallback)"
    except Exception:
        pass
    budget = out["max_per_episode"]
    # 0.5 is where 2 x budget stops being smaller than the 1.0 gap between two
    # clembench outcomes; 0 means no cap at all.
    out["preserves_outcome_ordering"] = bool(0.0 < budget < 0.5)
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
        "turn_reward_effect": _turn_reward_summary(marshal_cfg),
        "wandb": _wandb_summary(),
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

    # Spell the chain out. "train_segments 3" above is a number; the boundaries are
    # what you actually want when reading a reward curve with a discontinuity in it.
    if isinstance(train.get("train_segments"), int) and train["train_segments"] > 1:
        try:
            from playpen.marshal.resume import segment_bounds

            bounds = segment_bounds(train["max_steps"], train.get("segment_steps") or 0)
            lines += [
                "",
                "-- chained training jobs ---------------------------------------------",
                f"  {len(bounds)} job(s), each resuming the last one's checkpoint, ending at steps:",
                "    " + ", ".join(str(b) for b in bounds),
                "  --max-steps is the TOTAL in every one of them, so the LR schedule is",
                "  identical to an uninterrupted run and this arm stays comparable with an",
                "  unsegmented one. Only --stop-at-step differs between the jobs.",
            ]
        except Exception:
            pass

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

    tre = manifest["turn_reward_effect"]
    if tre.get("enabled"):
        lines += [
            "",
            "-- turn rewards, in reward units (game outcome is +1 / 0 / -1) -------",
            f"  extractor                    {tre.get('extractor', '?')} "
            f"(source={tre.get('source')})",
            f"  components                   {tre.get('components')}",
            f"  most a single turn is worth  {tre['max_per_turn']:+.4f}",
            f"  most an episode is worth     +-{tre['max_per_episode']:.4f}"
            + ("" if tre["max_per_episode"] else "   (UNCAPPED)"),
        ]
        if tre["preserves_outcome_ordering"]:
            lines.append(
                "  bound OK: 2 x budget < 1.0, the smallest gap between two clembench\n"
                "           outcomes, so shaping cannot make a loss out-score a win."
            )
        else:
            lines.append(
                "  WARNING: the per-episode budget is >= 0.5 (or uncapped), so shaping\n"
                "           can swing an episode by more than the gap between two\n"
                "           outcomes -- a well-shaped loss can beat a bare win.\n"
                "           Lower TR_BUDGET below 0.5 to restore the guarantee."
            )

    wb = manifest["wandb"]
    lines += ["", "-- weights & biases -------------------------------------------------"]
    if wb["enabled"]:
        lines += [
            f"  project                      {wb['project']}",
            f"  entity                       {wb['entity'] or '<account default>'}",
            f"  run name                     {wb['run_name'] or '<derived>'}",
            f"  group                        {wb['group'] or '<game>_<model>'}",
            f"  mode                         {wb['mode']}"
            f"{'  (offline unless a credential is reachable)' if wb['mode'] == 'auto' else ''}",
            f"  run data                     {wb['dir'] or '-'}",
        ]
        if wb["tags"]:
            lines.append(f"  extra tags                   {wb['tags']}")
    else:
        lines.append("  disabled (WB_ENABLE=0)")

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

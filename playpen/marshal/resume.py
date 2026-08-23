"""Resume a training run from a checkpoint, and stop it at a chosen step.

Two small pieces that together turn one long training run into a CHAIN of shorter
jobs, each continuing the last:

* :func:`resolve_resume` finds the checkpoint to continue from and refuses to
  continue from a half-written one;
* :func:`stop_at_step_callback` ends a segment at a chosen global step, saving a
  checkpoint on the way out so the next segment has something to resume from.

Both clusters cap a job's walltime (48 h on Eddie, 24 h on Isambard) and a long run
does not reliably fit, so a chain lets those hours be recovered rather than redone.

``--max-steps`` is always the TOTAL horizon for the whole chain, and ``--stop-at-step``
ends a segment early. Giving segment *k* ``--max-steps k*S`` instead would silently
change the learning-rate schedule: HF builds the scheduler with
``num_training_steps=max_steps`` and ``scheduler.pt`` restores only ``last_epoch``,
never the lambda, so a growing ``max_steps`` re-derives a different decay slope at
every resume and a 2-segment arm stops being comparable with a 1-segment one.

Stdlib-only at import time. The discovery half runs on a login node from
``experiments/*/resume_experiment.sh``, so :func:`stop_at_step_callback` imports
``transformers`` lazily.
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional, Sequence, Tuple

CHECKPOINT_PREFIX = "checkpoint-"
_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")

# Written LAST by Trainer._save_checkpoint (transformers/trainer.py: save_model ->
# optimizer/scheduler -> RNG -> trainer_state.json), which is what makes its presence
# a sound completeness sentinel: a job SIGKILLed at the walltime mid-save leaves the
# adapter but not this file, and such a directory must never be resumed from.
STATE_FILE = "trainer_state.json"
OPTIMIZER_FILE = "optimizer.pt"
SCHEDULER_FILE = "scheduler.pt"
# peft writes safetensors by default; the .bin spelling is only seen on old runs.
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_model.bin")

# What --resume-from-checkpoint accepts besides a path.
RESUME_SPECS = ("auto", "latest", "none")


class ResumeError(RuntimeError):
    """A resume was asked for explicitly and cannot be satisfied.

    Raised rather than warned-and-ignored: silently starting from step 0 when the
    caller asked to continue would burn a full walltime redoing work and, worse,
    would not look wrong in the logs until the step counter was read closely.
    """


# --- checkpoint discovery ----------------------------------------------------

def checkpoint_step(path: str) -> Optional[int]:
    """The step in a ``checkpoint-<N>`` directory name, or None if it is not one.

    Name-based, so it works on a directory whose ``trainer_state.json`` is missing --
    which is exactly the case :func:`is_resumable` has to report on.
    """
    match = _CHECKPOINT_RE.match(os.path.basename(os.path.normpath(path)))
    return int(match.group(1)) if match else None


def is_resumable(path: str) -> Tuple[bool, str]:
    """``(ok, reason)`` for one checkpoint directory.

    ``reason`` is non-empty even when ok is True: it carries any WARNING about state
    that is missing but not fatal, so the caller can print it. Optimizer and scheduler
    state are in that category -- without them training continues from the right
    weights and the right step but with a fresh Adam moment estimate, which is a real
    perturbation and must be said out loud rather than discovered in a loss curve.
    """
    if not os.path.isdir(path):
        return False, f"not a directory: {path}"
    if not os.path.isfile(os.path.join(path, STATE_FILE)):
        return False, (
            f"no {STATE_FILE} -- half-written checkpoint (the job was killed during "
            f"the save; {STATE_FILE} is written last)"
        )
    if not any(os.path.isfile(os.path.join(path, name)) for name in ADAPTER_FILES):
        return False, f"no LoRA adapter weights ({' or '.join(ADAPTER_FILES)})"

    missing = [
        name for name in (OPTIMIZER_FILE, SCHEDULER_FILE)
        if not os.path.isfile(os.path.join(path, name))
    ]
    if missing:
        return True, (
            f"WARNING: {', '.join(missing)} missing -- weights and step counter resume "
            f"correctly, but the optimizer moments restart from zero"
        )
    return True, ""


def list_checkpoints(output_dir: str, *, complete_only: bool = True) -> List[Tuple[int, str]]:
    """``[(step, path), ...]`` under ``output_dir``, ascending by step.

    Sorted on the extracted STEP NUMBER rather than the path, for the same reason
    ``exp_list_checkpoints`` in ``experiments/lib/experiment.sh`` is: a lexical sort
    orders 100, 200, 50 and would resume from the wrong place.
    """
    if not os.path.isdir(output_dir):
        return []
    found: List[Tuple[int, str]] = []
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        step = checkpoint_step(path)
        if step is None or not os.path.isdir(path):
            continue
        if complete_only and not is_resumable(path)[0]:
            continue
        found.append((step, path))
    return sorted(found, key=lambda pair: pair[0])


def latest_checkpoint(output_dir: str) -> Optional[str]:
    """The highest-step RESUMABLE checkpoint under ``output_dir``, or None."""
    found = list_checkpoints(output_dir)
    return found[-1][1] if found else None


def resumed_global_step(path: str) -> int:
    """The step training will actually continue from.

    Read from ``trainer_state.json`` -- the same file the Trainer itself reads -- and
    only falling back to the directory name if that is unparseable. The two normally
    agree; when they do not, the file is what governs, and a plan printed from the
    directory name would be a plausible-looking lie.
    """
    try:
        with open(os.path.join(path, STATE_FILE)) as fh:
            step = int(json.load(fh).get("global_step", 0))
        if step > 0:
            return step
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return checkpoint_step(path) or 0


# --- resolving the --resume-from-checkpoint argument -------------------------

def resolve_resume(spec: Optional[str], output_dir: str) -> Tuple[Optional[str], str]:
    """Turn a ``--resume-from-checkpoint`` value into ``(path_or_None, explanation)``.

    Accepted values:

    ``None`` / ``""`` / ``"none"``
        Start from step 0. The behaviour before this module existed.
    ``"auto"``
        Continue from the latest complete checkpoint in ``output_dir`` **if there is
        one**, else start fresh. This is what a chained segment passes: segment 1
        finds nothing and starts a run, segments 2..N find their predecessor's
        checkpoint and continue it, and one code path covers both.
    ``"latest"``
        As ``auto``, but an empty ``output_dir`` is an ERROR. What
        ``resume_experiment.sh`` and any segment >= 2 passes: there, "nothing to
        resume from" means the previous job died before its first save, and quietly
        starting over would spend a walltime on work that will be thrown away.
    a path
        Either a ``checkpoint-<N>`` directory, or a directory CONTAINING them (its
        latest is taken -- so ``RESUME_FROM=$EXP_DIR/train`` does the obvious thing).

    ``explanation`` is always printable and always says which of these applied,
    including any non-fatal warning from :func:`is_resumable`.
    """
    value = (spec or "").strip()
    if value == "" or value.lower() == "none":
        return None, "not resuming: training starts at step 0"

    lowered = value.lower()
    if lowered in ("auto", "latest"):
        path = latest_checkpoint(output_dir)
        if path is None:
            partial = [
                p for _, p in list_checkpoints(output_dir, complete_only=False)
                if not is_resumable(p)[0]
            ]
            detail = ""
            if partial:
                detail = (
                    f" ({len(partial)} incomplete checkpoint(s) were skipped, latest "
                    f"{os.path.basename(partial[-1])}: {is_resumable(partial[-1])[1]})"
                )
            if lowered == "latest":
                raise ResumeError(
                    f"--resume-from-checkpoint latest: no resumable checkpoint under "
                    f"{output_dir}{detail}.\n"
                    f"  The previous job most likely died before its first save. Either "
                    f"lower --save-steps and start the run again, or pass "
                    f"--resume-from-checkpoint auto to start from step 0 deliberately."
                )
            return None, (
                f"--resume-from-checkpoint auto: nothing to resume under {output_dir}"
                f"{detail} -- starting at step 0"
            )
        return path, _describe(path, f"--resume-from-checkpoint {lowered}")

    # An explicit path.
    path = os.path.abspath(os.path.expanduser(value))
    if checkpoint_step(path) is None:
        # A run directory rather than a checkpoint: take its latest.
        inner = latest_checkpoint(path)
        if inner is None:
            raise ResumeError(
                f"--resume-from-checkpoint {value}: not a checkpoint-<N> directory, and "
                f"contains no resumable checkpoint-<N> subdirectory."
            )
        path = inner
    ok, reason = is_resumable(path)
    if not ok:
        raise ResumeError(f"--resume-from-checkpoint {value}: {reason}")
    return path, _describe(path, "--resume-from-checkpoint")


def _describe(path: str, prefix: str) -> str:
    step = resumed_global_step(path)
    reason = is_resumable(path)[1]
    line = f"{prefix}: continuing from {path} at global step {step}"
    return f"{line}\n  {reason}" if reason else line


# --- ending a segment --------------------------------------------------------

def stop_at_step_callback(stop_at: int):
    """A ``TrainerCallback`` that ends training at ``stop_at`` and saves on the way out.

    Mirrors what ``DefaultFlowCallback`` does at ``max_steps``: set both
    ``should_training_stop`` and ``should_save``. Saving is not optional -- the loop
    exits immediately afterwards, and a segment that stopped without saving has
    thrown away every step since its last ``save_steps`` boundary.

    Callbacks only add flags to the shared ``TrainerControl``, so this composes with
    the default flow: when ``stop_at`` is a multiple of ``save_steps`` both set
    ``should_save`` and exactly one checkpoint is written.

    ``transformers`` is imported here so the discovery half of this module stays
    usable from a login node.
    """
    from transformers import TrainerCallback

    class StopAtStepCallback(TrainerCallback):
        def __init__(self, stop_at: int) -> None:
            self.stop_at = int(stop_at)
            self._fired = False

        def on_step_end(self, args, state, control, **kwargs):
            if not self._fired and state.global_step >= self.stop_at:
                self._fired = True
                control.should_save = True
                control.should_training_stop = True
                print(
                    f"[segment] reached step {state.global_step} of a "
                    f"{state.max_steps}-step run -- saving checkpoint-{state.global_step} "
                    f"and ending this job. The next segment resumes from it."
                )
            return control

    return StopAtStepCallback(stop_at)


# --- peft/transformers compatibility ------------------------------------------

def ensure_peft_resume_compat() -> Optional[str]:
    """Make LoRA resume work on a peft/transformers pairing where it is broken.

    Returns a description of the problem when a shim was installed, or None when the
    installed versions need no help. Safe to call more than once.

    The bug, seen on peft 0.19.1 + transformers 4.57.6:
    ``peft.utils.save_and_load._maybe_shard_state_dict_for_tp`` imports four
    tensor-parallel classes at call time and this transformers version only has
    three, so ``ImportError: cannot import name 'EmbeddingParallel'`` fires on every
    LoRA resume, after the model and vLLM have loaded.

    peft guards that call with ``torch.distributed.is_initialized()`` alone, which is
    true here because TRL's colocate vLLM mode initialises a single-rank process
    group. Sharding across a one-rank mesh is the identity, so at ``world_size == 1``
    the call is a no-op and skipping it is safe.

    Two conditions keep the shim narrow: it is installed only if the import genuinely
    fails, and it skips only at ``world_size == 1``. Under real tensor parallelism it
    defers to the original, which raises -- correctly, since there the sharding is
    load-bearing.
    """
    try:
        from peft.utils import save_and_load
    except ImportError:
        return None
    original = getattr(save_and_load, "_maybe_shard_state_dict_for_tp", None)
    if original is None or getattr(original, "_playpen_shim", False):
        return None

    try:
        # Exactly the import peft does, so the probe cannot disagree with the call.
        from transformers.integrations.tensor_parallel import (  # noqa: F401
            ALL_PARALLEL_STYLES,
            ColwiseParallel,
            EmbeddingParallel,
            RowwiseParallel,
        )
        return None
    except ImportError as exc:
        detail = str(exc)

    import torch

    def _shim(model, state_dict, adapter_name):
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            return original(model, state_dict, adapter_name)
        return None

    _shim._playpen_shim = True
    save_and_load._maybe_shard_state_dict_for_tp = _shim
    return detail


# --- planning (shared by the manifest and the job banners) -------------------

def segment_bounds(total_steps: int, segment_steps: int) -> List[int]:
    """The ``--stop-at-step`` of each segment covering ``total_steps``.

    ``segment_bounds(1000, 400) == [400, 800, 1000]``. The last entry is always
    ``total_steps``, so a chain never overshoots the horizon the LR schedule was
    built for and the final segment ends exactly where a single-job run would.
    """
    if total_steps <= 0:
        return []
    if segment_steps <= 0 or segment_steps >= total_steps:
        return [total_steps]
    bounds = list(range(segment_steps, total_steps, segment_steps))
    bounds.append(total_steps)
    return bounds


def describe_plan(total_steps: int, segment_steps: int, index: Optional[int] = None) -> str:
    """A one-or-two line summary of the chain, for a job banner."""
    bounds: Sequence[int] = segment_bounds(total_steps, segment_steps)
    if len(bounds) <= 1:
        return f"single training job, steps 0 -> {total_steps}"
    line = (
        f"{len(bounds)} chained training jobs of <= {segment_steps} steps: "
        + ", ".join(str(b) for b in bounds)
    )
    if index is not None and 1 <= index <= len(bounds):
        start = bounds[index - 2] if index >= 2 else 0
        line += f"\nthis job is segment {index}/{len(bounds)}: steps {start} -> {bounds[index - 1]}"
    return line

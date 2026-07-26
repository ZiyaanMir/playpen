"""Weights & Biases wiring for the MARSHAL self-play training pipeline.

Like ``config.py``, this module depends only on the standard library: it must be
importable on a login node (to resolve settings and write a manifest) without
torch, trl, vllm -- or even ``wandb`` itself. The only place ``wandb`` is imported
is :meth:`WandbSettings.start`, which the training script calls once.

Why start the run ourselves instead of just passing ``report_to="wandb"``?
HuggingFace's ``WandbCallback.setup`` calls ``wandb.init(project=..., name=...)``
**only when ``wandb.run is None``**, and it passes nothing else -- no entity, no
group, no tags, no run id, no offline directory. Those are exactly the fields that
make a set of ablation runs comparable in the UI. Because the callback reuses an
already-open run (and then folds ``TrainingArguments`` into its config), calling
``wandb.init`` first gives us every field *and* keeps HF's own logging intact.

Cluster reality drives two design points:

* ``mode="auto"`` resolves to **offline** when no API key is reachable, so a batch
  job on a compute node with no outbound network still records everything to disk
  and never blocks on a login prompt. ``wandb sync`` uploads it afterwards.
* the offline run directory defaults to the run's own output directory, so an
  experiment folder stays self-contained (one thing to rsync home, one thing to
  delete) -- the same principle ``experiments/lib/experiment.sh`` follows.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

WANDB_MODES = ("auto", "online", "offline", "disabled")

# wandb rejects tags longer than this, and silently drops empty ones.
_MAX_TAG_LEN = 64
# Run names travel into filesystem paths (offline dirs, artifact names), so keep
# them to characters that survive a path round-trip on every cluster filesystem.
_NAME_SAFE = re.compile(r"[^A-Za-z0-9._@-]+")


def _clean_tag(tag: str) -> str:
    return tag.strip()[:_MAX_TAG_LEN]


def sanitize_run_name(name: str) -> str:
    """Make ``name`` safe to use both as a wandb run name and as a path segment."""
    return _NAME_SAFE.sub("_", name.strip()).strip("_") or "run"


def _env(environ: Mapping[str, str], name: str) -> Optional[str]:
    value = environ.get(name)
    return value if value else None


# How long to wait for a TCP connection to the W&B API before calling it
# unreachable. Small on purpose: it is paid once per run, and the answer on a
# compute node is either "immediately yes" or "never".
_REACH_TIMEOUT_SECONDS = 4.0


def _api_host_port(environ: Mapping[str, str]) -> tuple:
    """``(host, port)`` of the W&B API, honouring ``WANDB_BASE_URL``."""
    base_url = environ.get("WANDB_BASE_URL") or "https://api.wandb.ai"
    scheme, _, rest = base_url.partition("://")
    if not rest:
        scheme, rest = "https", base_url
    authority = rest.split("/", 1)[0]
    host, _, port = authority.partition(":")
    return host, int(port) if port.isdigit() else (80 if scheme == "http" else 443)


def has_wandb_credentials(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether an online run could authenticate without prompting.

    Checks the two places the wandb SDK looks before it would fall back to an
    interactive login: ``WANDB_API_KEY`` and a netrc entry for the configured host
    (which is where ``wandb login`` stores the key). No network is touched here --
    see :func:`can_reach_wandb` for the other half of the question.
    """
    environ = os.environ if environ is None else environ
    if _env(environ, "WANDB_API_KEY"):
        return True
    host, _ = _api_host_port(environ)
    try:
        import netrc

        netrc_path = _env(environ, "NETRC")
        auth = netrc.netrc(netrc_path).authenticators(host)
        return bool(auth and auth[2])
    except Exception:
        # No netrc, unreadable, or malformed -- all mean "no usable credential".
        return False


def can_reach_wandb(
    environ: Optional[Mapping[str, str]] = None, timeout: float = _REACH_TIMEOUT_SECONDS
) -> bool:
    """Whether the W&B API answers a TCP connection right now.

    A credential is not enough to justify an online run. ``$HOME`` is shared
    between a cluster's login and compute nodes, so ``wandb login`` on the login
    node leaves a netrc that is perfectly readable from a compute node that has no
    outbound network at all. Choosing "online" on the strength of that file alone
    is how a training job ends up blocked in ``wandb.init`` -- so ``auto`` asks the
    network, once, and pays a few seconds for an answer that saves the run.

    Any socket error means "not reachable": an air-gapped node fails at DNS, a
    firewalled one at connect, and both mean the same thing here.
    """
    environ = os.environ if environ is None else environ
    host, port = _api_host_port(environ)
    try:
        import socket

        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@dataclass
class WandbSettings:
    """Everything needed to open (or deliberately not open) a W&B run.

    Attributes:
        enabled: Master switch. When ``False`` nothing is imported, no environment
            variable is touched, and :meth:`report_to` leaves the trainer's sinks
            alone -- i.e. the pipeline behaves exactly as it did before W&B existed.
        project: W&B project. Defaults to ``$WANDB_PROJECT`` then
            ``"playpen-marshal"`` (never HuggingFace's ``"huggingface"`` default,
            which scatters runs into a project nobody looks at).
        entity: W&B team/user. ``None`` uses the account default.
        run_name: Display name. ``None`` lets :meth:`with_defaults` build one from
            the game, model and run id.
        group: Groups related runs in the UI. The natural value is the experiment
            family (e.g. ``dond_Qwen3-4B``) so an ablation's arms sit together.
        job_type: ``"train"`` here; ``experiments/*/eval.sh`` could log ``"eval"``
            against the same group.
        tags: Free-form filters. The launcher adds game/model/cluster/switch tags.
        mode: One of :data:`WANDB_MODES`. ``"auto"`` (the default) picks
            ``"online"`` when credentials are reachable and ``"offline"`` otherwise;
            see :meth:`resolve_mode`.
        dir: Parent of the ``wandb/`` directory holding run data. ``None`` means
            the training output directory.
        run_id: Explicit run id. Needed only to resume; otherwise wandb generates one.
        resume: wandb resume policy (``"allow"``, ``"must"``, ``"never"``). Pair
            with ``run_id`` to continue a run after a requeue.
        notes: Free text stored on the run.
        required: Whether the user asked for W&B *by name* (``--wandb`` /
            ``--report-to wandb``) rather than implicitly (a ``WANDB_PROJECT`` left
            in the environment). It decides only one thing -- what a missing
            ``wandb`` package does -- and it decides it the way each case deserves:
            an explicit request fails loudly at startup, an inherited environment
            variable degrades to a warning rather than killing a training job that
            never asked for W&B in the first place. See :meth:`start`.
    """

    enabled: bool = False
    project: str = "playpen-marshal"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    group: Optional[str] = None
    job_type: str = "train"
    tags: List[str] = field(default_factory=list)
    mode: str = "auto"
    dir: Optional[str] = None
    run_id: Optional[str] = None
    resume: Optional[str] = None
    notes: Optional[str] = None
    required: bool = True

    def __post_init__(self) -> None:
        # Not a dataclass field: a per-run cache of the reachability probe, so
        # resolve_mode() can be called freely without re-dialling the network.
        self._reachable: Optional[bool] = None
        if self.mode not in WANDB_MODES:
            raise ValueError(f"wandb mode must be one of {WANDB_MODES}, got {self.mode!r}")
        self.tags = [t for t in (_clean_tag(t) for t in self.tags) if t]
        if self.run_name:
            self.run_name = sanitize_run_name(self.run_name)

    # --- construction --------------------------------------------------------

    @classmethod
    def from_args(cls, args: Any, environ: Optional[Mapping[str, str]] = None) -> "WandbSettings":
        """Build from the ``train_selfplay.py`` argparse namespace.

        Precedence is CLI flag > ``WANDB_*`` environment variable > default, so a
        cluster job script can export ``WANDB_PROJECT``/``WANDB_ENTITY`` once and a
        one-off command can still override either.

        Enablement follows the same idea: ``--wandb``/``--no-wandb`` wins, then
        ``--report-to wandb`` (kept working for anyone with it in a script), then
        ``WANDB_PROJECT`` being set, which is a clear enough statement of intent.
        ``--wandb-mode disabled`` overrides all of them, because that is what
        "disabled" has to mean.
        """
        environ = os.environ if environ is None else environ

        report_to = getattr(args, "report_to", "none") or "none"
        explicit = getattr(args, "wandb", None)
        required = explicit is True or report_to == "wandb"
        if explicit is None:
            enabled = required or bool(_env(environ, "WANDB_PROJECT"))
        else:
            enabled = bool(explicit)

        mode = getattr(args, "wandb_mode", None) or _env(environ, "WANDB_MODE") or "auto"
        if mode not in WANDB_MODES:
            raise ValueError(
                f"--wandb-mode must be one of {WANDB_MODES}, got {mode!r} "
                "(check the WANDB_MODE environment variable too)"
            )
        if mode == "disabled":
            enabled = False

        raw_tags = getattr(args, "wandb_tags", None) or _env(environ, "WANDB_TAGS") or ""
        tags = [t for t in (part.strip() for part in raw_tags.split(",")) if t]

        return cls(
            enabled=enabled,
            project=(getattr(args, "wandb_project", None)
                     or _env(environ, "WANDB_PROJECT") or "playpen-marshal"),
            entity=getattr(args, "wandb_entity", None) or _env(environ, "WANDB_ENTITY"),
            run_name=getattr(args, "wandb_run_name", None) or _env(environ, "WANDB_NAME"),
            group=getattr(args, "wandb_group", None) or _env(environ, "WANDB_RUN_GROUP"),
            job_type=getattr(args, "wandb_job_type", None) or _env(environ, "WANDB_JOB_TYPE") or "train",
            tags=tags,
            mode=mode,
            dir=getattr(args, "wandb_dir", None) or _env(environ, "WANDB_DIR"),
            run_id=getattr(args, "wandb_id", None) or _env(environ, "WANDB_RUN_ID"),
            resume=getattr(args, "wandb_resume", None) or _env(environ, "WANDB_RESUME"),
            notes=getattr(args, "wandb_notes", None) or _env(environ, "WANDB_NOTES"),
            required=required,
        )

    def with_defaults(
        self,
        *,
        output_dir: str,
        run_id_stamp: str,
        game: Optional[str] = None,
        model: Optional[str] = None,
        extra_tags: Sequence[str] = (),
    ) -> "WandbSettings":
        """Fill in the fields the launcher can derive, leaving explicit ones alone.

        A default run name of ``{game}_{model}_{timestamp}`` matches the on-disk
        run directory, so a row in the W&B table can be traced back to a checkpoint
        folder (and vice versa) without keeping a separate mapping.
        """
        if not self.enabled:
            return self
        if not self.run_name:
            parts = [p for p in (game, os.path.basename(model) if model else None, run_id_stamp) if p]
            self.run_name = sanitize_run_name("_".join(parts))
        if not self.group and game and model:
            self.group = sanitize_run_name(f"{game}_{os.path.basename(model)}")
        if not self.dir:
            self.dir = output_dir
        known = set(self.tags)
        for tag in extra_tags:
            tag = _clean_tag(tag)
            if tag and tag not in known:
                self.tags.append(tag)
                known.add(tag)
        return self

    # --- resolution ----------------------------------------------------------

    def resolve_mode(self, environ: Optional[Mapping[str, str]] = None) -> str:
        """Turn ``"auto"`` into a concrete wandb mode.

        Online only when a credential exists **and** the API answers; offline
        otherwise. Both halves are needed, and for different reasons: without a
        credential wandb would stop at an interactive login prompt no batch job can
        answer, and without network it would block in ``wandb.init``. Offline always
        works, costs a ``wandb sync`` later, and loses nothing.

        The reachability probe result is cached on the instance, because
        ``resolve_mode`` is called from the summary line, the env overrides, the
        manifest and the run itself -- one answer per run is the honest number of
        times to ask.
        """
        if not self.enabled:
            return "disabled"
        if self.mode != "auto":
            return self.mode
        if not has_wandb_credentials(environ):
            return "offline"
        if self._reachable is None:
            self._reachable = can_reach_wandb(environ)
        return "online" if self._reachable else "offline"

    def report_to(self, base: str | Sequence[str] | None) -> List[str]:
        """The trainer's ``report_to`` list, with ``wandb`` added or removed.

        Composes rather than replaces, so ``--report-to tensorboard --wandb`` logs
        to both. ``"none"`` is dropped as soon as any real sink is present, since
        HF treats it as a sentinel and not a sink.
        """
        if base is None:
            sinks: List[str] = []
        elif isinstance(base, str):
            sinks = [base] if base and base != "none" else []
        else:
            sinks = [s for s in base if s and s != "none"]
        if self.enabled and "wandb" not in sinks:
            sinks.append("wandb")
        if not self.enabled:
            sinks = [s for s in sinks if s != "wandb"]
        return sinks or ["none"]

    def env_overrides(self, environ: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
        """The ``WANDB_*`` variables that must be set before the trainer starts.

        Kept as a returned dict (rather than mutating in place) so it can be
        asserted on in tests and printed in a job log. :meth:`apply_env` does the
        mutation.

        ``WANDB_PROJECT`` matters even though :meth:`start` passes the project to
        ``wandb.init``: HF's callback reads that variable directly when *it* opens
        the run, which is what happens if our init is ever skipped.
        """
        environ = os.environ if environ is None else environ
        if not self.enabled:
            return {}
        out = {"WANDB_PROJECT": self.project, "WANDB_MODE": self.resolve_mode(environ)}
        if self.entity:
            out["WANDB_ENTITY"] = self.entity
        if self.run_name:
            out["WANDB_NAME"] = self.run_name
        if self.dir:
            out["WANDB_DIR"] = self.dir
        return out

    def apply_env(self, environ: Optional[Any] = None) -> Dict[str, str]:
        """Apply :meth:`env_overrides` to the process environment."""
        target = os.environ if environ is None else environ
        overrides = self.env_overrides(target)
        for key, value in overrides.items():
            target[key] = value
        return overrides

    def summary(self) -> str:
        """One line for the job log, before anything expensive has happened."""
        if not self.enabled:
            return "[wandb] off (pass --wandb, or --report-to wandb, to enable)"
        mode = self.resolve_mode()
        line = (f"[wandb] on project={self.project} "
                f"entity={self.entity or '<default>'} run={self.run_name or '<auto>'} "
                f"group={self.group or '-'} mode={mode}")
        if self.tags:
            line += f" tags={','.join(self.tags)}"
        if mode == "offline" and self.mode == "auto":
            reason = ("no credential found" if not has_wandb_credentials()
                      else "credential found but the W&B API is unreachable from this node")
            line += f"\n[wandb] recording OFFLINE ({reason}); upload later with `wandb sync`"
        elif mode == "offline":
            line += "\n[wandb] recording OFFLINE; upload later with `wandb sync`"
        return line

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "project": self.project,
            "entity": self.entity,
            "run_name": self.run_name,
            "group": self.group,
            "job_type": self.job_type,
            "tags": list(self.tags),
            "mode": self.mode,
            "resolved_mode": self.resolve_mode(),
            "dir": self.dir,
            "run_id": self.run_id,
            "resume": self.resume,
        }

    # --- the run itself ------------------------------------------------------

    def start(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Open the W&B run, returning it (or ``None`` when disabled).

        Called *before* the trainer is constructed so the run exists by the time
        HF's ``WandbCallback`` looks for one; the callback then reuses it and folds
        ``TrainingArguments`` into the same config.

        A missing ``wandb`` package fails here, at startup, when the run was asked
        for explicitly: someone who passed ``--wandb`` and got a silent no-op would
        discover it only after the run finished, with nothing recorded. When W&B was
        merely *implied* by a ``WANDB_PROJECT`` sitting in the environment it warns
        and carries on instead, so a stale shell variable cannot kill a training job
        that never asked for W&B. Everything *after* a successful init is
        best-effort -- see :func:`log_config`.

        Only rank 0 opens a run. On other ranks HF's callback is a no-op anyway
        (``state.is_world_process_zero``), and a second init would fragment one
        training run across N W&B runs.
        """
        if not self.enabled:
            return None
        if int(os.environ.get("RANK", "0")) != 0:
            return None
        try:
            import wandb
        except ImportError as exc:
            if not self.required:
                self.enabled = False
                print("[wandb] WARNING: the wandb package is not installed, so this run "
                      "is NOT being logged to W&B.\n"
                      "[wandb]          (W&B was implied by $WANDB_PROJECT, not requested "
                      "on the command line, so training continues.)\n"
                      "[wandb]          Install it with `uv pip install wandb`, or unset "
                      "WANDB_PROJECT to silence this.")
                return None
            raise SystemExit(
                "--wandb was requested but the wandb package is not installed.\n"
                "  install it:  uv pip install wandb      (or: pip install 'playpen[wandb]')\n"
                "  or drop it:  --no-wandb / --wandb-mode disabled"
            ) from exc

        self.apply_env()
        if self.dir:
            os.makedirs(self.dir, exist_ok=True)
        init_kwargs: Dict[str, Any] = {
            "project": self.project,
            "entity": self.entity,
            "name": self.run_name,
            "group": self.group,
            "job_type": self.job_type,
            "tags": self.tags or None,
            "mode": self.resolve_mode(),
            "dir": self.dir,
            "notes": self.notes,
            "config": config or {},
        }
        if self.run_id:
            init_kwargs["id"] = self.run_id
        if self.resume:
            init_kwargs["resume"] = self.resume
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

        try:
            run = wandb.init(**init_kwargs)
        except Exception as exc:
            # The reachability probe can only be a snapshot: the network can be up
            # when we ask and gone (or proxied, or rate-limited) a second later. An
            # online init that fails is not a reason to lose the training run, so
            # retry offline -- the data lands on disk and `wandb sync` still gets it
            # to the same project.
            if init_kwargs.get("mode") != "online":
                self.enabled = False
                print(f"[wandb] WARNING: wandb.init failed ({type(exc).__name__}: {exc}).\n"
                      "[wandb]          Training continues WITHOUT W&B logging.")
                return None
            print(f"[wandb] WARNING: online wandb.init failed ({type(exc).__name__}: {exc}).\n"
                  "[wandb]          Falling back to OFFLINE recording; upload it afterwards "
                  "with `wandb sync`.")
            self.mode = "offline"
            self._reachable = False
            init_kwargs["mode"] = "offline"
            self.apply_env()  # keep WANDB_MODE in step with what we are about to do
            try:
                run = wandb.init(**init_kwargs)
            except Exception as offline_exc:
                self.enabled = False
                print(f"[wandb] WARNING: offline wandb.init also failed "
                      f"({type(offline_exc).__name__}: {offline_exc}).\n"
                      "[wandb]          Training continues WITHOUT W&B logging.")
                return None

        # Keep the generated id, so a resume after a requeue can be pointed at this
        # exact run and so run_metadata() records something stable.
        self.run_id = getattr(run, "id", None) or self.run_id
        return run


def log_config(run: Any, config: Mapping[str, Any], prefix: str = "") -> None:
    """Merge extra key/values into the run config, never raising.

    Best-effort by design, and the reason is the same one the trainer's own
    ``_log_*`` helpers give: a metadata call must not be able to kill a training
    job that is otherwise healthy. Before this point (:meth:`WandbSettings.start`)
    failures are loud, because there the run does not exist at all.
    """
    if run is None:
        return
    try:
        payload = {f"{prefix}{k}": v for k, v in config.items()} if prefix else dict(config)
        run.config.update(payload, allow_val_change=True)
    except Exception:
        pass


def run_metadata(run: Any, settings: WandbSettings) -> Dict[str, Any]:
    """A small JSON-able record identifying the run, for the experiment directory.

    This is what closes the loop between a results folder on a cluster filesystem
    and a row in the W&B UI: ``url`` for an online run, ``offline_dir`` plus the
    exact ``sync_command`` for an offline one.
    """
    meta: Dict[str, Any] = {"wandb": settings.to_dict()}
    if run is None:
        return meta
    offline_dir = None
    try:
        offline_dir = str(run.dir).rstrip("/").rsplit("/files", 1)[0] or None
    except Exception:
        offline_dir = None
    meta.update({
        "id": getattr(run, "id", None),
        "name": getattr(run, "name", None),
        "url": (getattr(run, "url", None) if settings.resolve_mode() == "online" else None),
        "offline_dir": offline_dir,
    })
    if settings.resolve_mode() == "offline" and offline_dir:
        meta["sync_command"] = f"wandb sync {offline_dir}"
    return meta


def write_run_metadata(path: str, run: Any, settings: WandbSettings) -> Optional[str]:
    """Write :func:`run_metadata` to ``path``; return it, or ``None`` on failure."""
    if run is None:
        return None
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(run_metadata(run, settings), fh, indent=2)
            fh.write("\n")
        return path
    except Exception:
        return None


def finish(run: Any, settings: WandbSettings) -> None:
    """Close the run and, when offline, print the command that uploads it."""
    if run is None:
        return
    meta = run_metadata(run, settings)
    try:
        import wandb

        wandb.finish()
    except Exception:
        pass
    if meta.get("url"):
        print(f"[wandb] run: {meta['url']}")
    elif meta.get("sync_command"):
        print(f"[wandb] offline run stored at {meta['offline_dir']}")
        print(f"[wandb] upload it from a machine with network access:\n"
              f"          {meta['sync_command']}")

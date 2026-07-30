#!/usr/bin/env python3
"""Did these checkpoints actually change? Answer it from the safetensors alone.

Reads the LoRA adapters an experiment wrote (``checkpoint-*/adapter_model.safetensors``)
and reports, per checkpoint:

* ``||dW||`` -- the Frobenius norm of the delta the adapter applies to the base model,
  ``scaling * B @ A`` summed over every LoRA module. **This is the change versus the
  untrained model**: PEFT initialises ``lora_B`` to zero, so at step 0 the adapter is
  mathematically a no-op and this number is exactly 0. A checkpoint whose ``lora_B`` is
  still all-zero cannot score differently from base, no matter what the eval says.
* ``moved`` -- ``||dW_this - dW_prev||``, how far the applied delta travelled since the
  previous checkpoint, plus the raw count of parameter tensors whose bits differ.

So an experiment that "learned nothing" is diagnosed here without a GPU: either the
adapter is a no-op (``lora_B`` zero), or consecutive checkpoints are byte-identical
(no optimizer step landed), or the delta is real and the problem is elsewhere.

    # every experiment under $MARSHAL_RUNS (or the cluster default)
    experiments/check_checkpoints.py

    # one experiment, one run directory, or one checkpoint -- any depth works
    experiments/check_checkpoints.py $MARSHAL_RUNS/guesswhat_Qwen3-4B_20260726-141415
    experiments/check_checkpoints.py ~/rsynced/cluster-runs/eddie

    # relative sizing: how big is the delta against the weights it is added to?
    experiments/check_checkpoints.py <exp> --base Qwen/Qwen3-4B
    experiments/check_checkpoints.py <exp> --base /path/to/local/Qwen3-4B

    # which modules moved most, machine-readable output, non-zero exit on a problem
    experiments/check_checkpoints.py <exp> --top 10
    experiments/check_checkpoints.py <exp> --json > checkpoints.json
    experiments/check_checkpoints.py <exp> --strict

Needs numpy + safetensors, i.e. the training venv (``.venv``) or the lm-eval env; torch
is used when present (bf16 adapters need it) but is not required. Reads files only, so
it is safe on a login node and works on an rsync'd copy.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

try:
    import numpy as np
except ImportError:  # pragma: no cover - environment problem, not a code path
    sys.exit("ERROR: numpy is required. Run this with the training venv's python "
             "(.venv/bin/python) or any env that has numpy + safetensors.")

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover
    sys.exit("ERROR: safetensors is required. Run this with the training venv's python "
             "(.venv/bin/python), or `pip install safetensors`.")

try:
    import torch
except ImportError:
    torch = None

# ``.default`` is present in a raw PEFT state_dict and absent after save_pretrained;
# accept both so this works on hand-saved adapters too.
LORA_KEY_RE = re.compile(r"^(?P<module>.+?)\.lora_(?P<side>[AB])(?:\.default)?\.weight$")

# What a checkpoint directory must contain to be one. A half-written checkpoint from a
# job killed at the walltime has neither and is skipped rather than reported as broken.
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_model.bin")

# Directories that hold `checkpoint-<N>` subdirectories which are NOT checkpoints:
# eval/<checkpoint-N>/ is lm-eval output for that checkpoint. Skipped by name so a
# whole runs-root can be scanned without tripping over them.
SKIP_DIRS = {"eval", ".git", "wandb", "node_modules", "__pycache__"}


# --------------------------------------------------------------------------- reading

class Weights:
    """Lazily-read tensors of one checkpoint, as float32 numpy arrays.

    Lazy on purpose: comparing checkpoint N with N-1 needs two checkpoints open at
    once, and a full-finetune checkpoint is many GB. Only one tensor is ever
    materialised, so peak memory is one tensor regardless of model size.
    """

    def __init__(self, path: str):
        self.path = path
        self._bin: dict | None = None
        self._f = None
        if path.endswith(".bin"):
            if torch is None:
                raise RuntimeError(f"{path} is a torch .bin adapter and torch is not "
                                   f"installed in this environment")
            self._bin = torch.load(path, map_location="cpu", weights_only=True)
        else:
            # framework="pt" handles bf16/fp16 adapters, which the numpy backend rejects.
            self._f = safe_open(path, framework=("pt" if torch is not None else "np"))

    def keys(self) -> list[str]:
        return list(self._bin.keys() if self._bin is not None else self._f.keys())

    def get(self, name: str) -> np.ndarray:
        if self._bin is not None:
            return self._bin[name].detach().to(torch.float32).numpy()
        t = self._f.get_tensor(name)
        if torch is not None:
            return t.detach().to(torch.float32).numpy()
        return np.asarray(t, dtype=np.float32)

    def close(self) -> None:
        self._bin = None
        self._f = None


def adapter_file(ckpt: str) -> str | None:
    for name in ADAPTER_FILES:
        p = os.path.join(ckpt, name)
        if os.path.isfile(p):
            return p
    return None


def weight_file(ckpt: str) -> str | None:
    """The adapter, or -- for a full-finetune checkpoint -- the single model file."""
    p = adapter_file(ckpt)
    if p:
        return p
    p = os.path.join(ckpt, "model.safetensors")
    return p if os.path.isfile(p) else None


def is_sharded(ckpt: str) -> bool:
    """A multi-shard full-model checkpoint: recognised so it is reported, not skipped."""
    return os.path.isfile(os.path.join(ckpt, "model.safetensors.index.json"))


# ------------------------------------------------------------------------- discovery

def default_runs_root() -> str:
    """Same resolution order as experiments/status.sh."""
    if os.environ.get("MARSHAL_RUNS"):
        return os.environ["MARSHAL_RUNS"]
    projectdir, user = os.environ.get("PROJECTDIR"), os.environ.get("USER", "")
    if projectdir and os.path.isdir(os.path.join(projectdir, user, "marshal-runs")):
        return os.path.join(projectdir, user, "marshal-runs")
    return f"/exports/eddie/scratch/{user}/marshal-runs"


def find_checkpoints(root: str) -> list[str]:
    """Every real checkpoint directory at or under ``root``, unordered.

    Walks rather than globbing a fixed depth so one code path covers all the layouts
    in play: ``<exp>/train/checkpoint-N`` (the job scripts pass --no-run-subdir),
    ``<exp>/train/<timestamp>/checkpoint-N`` (older runs), ``models/marshal/<game>/
    <model>/<timestamp>/checkpoint-N`` (hand-launched runs), and a bare checkpoint dir.
    """
    root = os.path.abspath(root)
    if os.path.basename(root).startswith("checkpoint-") and (weight_file(root) or is_sharded(root)):
        return [root]

    found = []
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_DIRS and not d.startswith("_base_eval_cache")]
        for d in list(dirnames):
            if not d.startswith("checkpoint-"):
                continue
            ck = os.path.join(dirpath, d)
            if weight_file(ck) or is_sharded(ck):
                found.append(ck)
            # Never descend into a checkpoint: nothing below it is another checkpoint.
            dirnames.remove(d)
    return found


def step_of(ckpt: str) -> int:
    """Training step from the directory name, or -1 if it is not a number.

    Sorting by this and not by the path matters: run paths contain '-' (timestamps,
    model names), so a lexical sort orders checkpoints 100, 200, 50 and silently
    mislabels a learning curve.
    """
    tail = os.path.basename(ckpt).split("checkpoint-", 1)[-1]
    return int(tail) if tail.isdigit() else -1


def group_runs(ckpts: list[str]) -> list[tuple[str, list[str]]]:
    """Group checkpoints by the directory holding them, each list oldest step first."""
    runs: dict[str, list[str]] = {}
    for ck in ckpts:
        runs.setdefault(os.path.dirname(ck), []).append(ck)
    out = []
    for run_dir in sorted(runs):
        out.append((run_dir, sorted(runs[run_dir], key=step_of)))
    return out


def run_label(run_dir: str, root: str) -> str:
    """A short name for the run: the experiment directory it belongs to.

    ``<exp>/train`` and ``<exp>/train/<timestamp>`` both name the experiment, so those
    components are dropped. A timestamp NOT under ``train/`` is kept -- in the
    ``models/marshal/<game>/<model>/<timestamp>/`` layout it is the only thing telling
    two runs of the same game apart.
    """
    parts = os.path.abspath(run_dir).rstrip(os.sep).split(os.sep)
    if len(parts) >= 2 and parts[-2] == "train" and re.fullmatch(r"\d{8}-\d{6}", parts[-1]):
        parts.pop()
    if parts and parts[-1] == "train":
        parts.pop()
    exp_dir = os.sep.join(parts)
    rel = os.path.relpath(exp_dir, os.path.abspath(root))
    # relpath is "." when the target IS the root, and starts with ".." when the root is
    # a sibling -- in both cases the directory's own name is the informative label.
    return os.path.basename(exp_dir) if rel in (".", "") or rel.startswith("..") else rel


# ------------------------------------------------------------- LoRA maths (norms only)

def lora_pairs(keys: list[str]) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Split tensor names into {module: {"A": key, "B": key}} and everything else.

    "Everything else" is normally empty, or the full copies PEFT writes for
    ``modules_to_save`` (lm_head / embed_tokens) -- those are real weights, compared
    tensor-wise like a full-finetune checkpoint.
    """
    pairs: dict[str, dict[str, str]] = {}
    other: list[str] = []
    for k in keys:
        m = LORA_KEY_RE.match(k)
        if m:
            pairs.setdefault(m.group("module"), {})[m.group("side")] = k
        else:
            other.append(k)
    complete = {mod: sides for mod, sides in pairs.items() if "A" in sides and "B" in sides}
    other += [k for mod, sides in pairs.items() if mod not in complete for k in sides.values()]
    return complete, sorted(other)


def scaling_for(cfg: dict, module: str) -> float:
    """PEFT's per-module scaling: alpha/r, or alpha/sqrt(r) under rslora."""
    r = cfg.get("r") or 8
    alpha = cfg.get("lora_alpha") or 8
    for pat, val in (cfg.get("rank_pattern") or {}).items():
        if module.endswith(pat):
            r = val
    for pat, val in (cfg.get("alpha_pattern") or {}).items():
        if module.endswith(pat):
            alpha = val
    if cfg.get("use_rslora"):
        return float(alpha) / math.sqrt(float(r))
    return float(alpha) / float(r)


def _gram(x: np.ndarray) -> np.ndarray:
    """x @ x.T in float64. x is (r, n) or (n, r).T -- always small in the r dimension."""
    x64 = x.astype(np.float64, copy=False)
    return x64 @ x64.T


def fro_product(B: np.ndarray, A: np.ndarray, scale: float) -> float:
    """||scale * B @ A||_F, without materialising the (out x in) product.

    ||BA||_F^2 = tr(A^T B^T B A) = tr((B^T B)(A A^T)), and both grams are r x r
    (r = 16 here), so this is exact and costs nothing -- materialising B@A for every
    module of a 4B model would be tens of GB of transient allocation.
    """
    gb = _gram(B.T)          # (r, r)
    ga = _gram(A)            # (r, r)
    return abs(scale) * math.sqrt(max(float(np.sum(gb * ga)), 0.0))


def fro_product_diff(B2, A2, s2, B1, A1, s1) -> float:
    """||s2*B2@A2 - s1*B1@A1||_F, same trick on the stacked factors.

    The difference of two rank-r products is one rank-2r product:
    C = [s2*B2 | -s1*B1] (out x 2r), D = [A2 ; A1] (2r x in), difference = C @ D.
    """
    C = np.concatenate([B2.astype(np.float64) * s2, B1.astype(np.float64) * -s1], axis=1)
    D = np.concatenate([A2.astype(np.float64), A1.astype(np.float64)], axis=0)
    return math.sqrt(max(float(np.sum(_gram(C.T) * _gram(D))), 0.0))


# ------------------------------------------------------------------- base-model sizing

class BaseWeights:
    """Frobenius norms of the base model's weights, for relative sizing of dW.

    Opened lazily shard by shard and reduced to one float per tensor, so a 4B model
    costs one shard's worth of transient memory and nothing persistent.
    """

    def __init__(self, spec: str):
        self.spec = spec
        self.norms: dict[str, float] = {}
        path = self._resolve(spec)
        shards = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if not shards:
            raise FileNotFoundError(f"no *.safetensors under {path}")
        for shard in shards:
            f = safe_open(shard, framework=("pt" if torch is not None else "np"))
            for k in f.keys():
                t = f.get_tensor(k)
                if torch is not None:
                    self.norms[k] = float(torch.linalg.vector_norm(t.detach().float()))
                else:
                    self.norms[k] = float(np.linalg.norm(np.asarray(t, dtype=np.float64)))
        self.path = path

    @staticmethod
    def _resolve(spec: str) -> str:
        if os.path.isdir(spec):
            return spec
        try:  # a hub id: use the local cache only, compute nodes have no network
            from huggingface_hub import snapshot_download
            return snapshot_download(spec, local_files_only=True,
                                     allow_patterns=["*.safetensors", "*.json"])
        except Exception as exc:
            raise FileNotFoundError(
                f"--base {spec!r} is not a directory and is not in the local HF cache "
                f"({exc}). Pass a path to the downloaded model directory.") from exc

    def norm_of(self, lora_module: str) -> float | None:
        """||W|| of the base weight this LoRA module is attached to.

        ``base_model.model.model.layers.0.mlp.down_proj`` -> ``model.layers.0.mlp.down_proj.weight``
        """
        name = lora_module
        if name.startswith("base_model.model."):
            name = name[len("base_model.model."):]
        for cand in (name + ".weight", name + ".base_layer.weight", name):
            if cand in self.norms:
                return self.norms[cand]
        return None


# -------------------------------------------------------------------------- analysis

def read_json(path: str) -> dict:
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}


def trainer_signals(ckpt: str) -> dict:
    """lr / grad_norm / loss / reward at this checkpoint, from trainer_state.json.

    Context for the norms, not a substitute: lr == 0 or grad_norm == 0 explains a
    frozen adapter, and a moving loss next to a frozen adapter means the checkpoint
    write is what is broken.
    """
    state = read_json(os.path.join(ckpt, "trainer_state.json"))
    hist = [h for h in state.get("log_history", []) if isinstance(h, dict)]
    out: dict[str, float] = {}
    for key in ("learning_rate", "grad_norm", "loss", "reward", "reward_std"):
        for entry in reversed(hist):
            if key in entry and isinstance(entry[key], (int, float)):
                out[key] = float(entry[key])
                break
    if state.get("global_step") is not None:
        out["global_step"] = state["global_step"]
    return out


def analyse_checkpoint(w: Weights, cfg: dict) -> dict:
    """Per-checkpoint totals: the applied delta, and whether lora_B is still zero."""
    pairs, other = lora_pairs(w.keys())
    modules, sq_total = [], 0.0
    b_zero = 0
    for mod in sorted(pairs):
        A = w.get(pairs[mod]["A"])
        B = w.get(pairs[mod]["B"])
        scale = scaling_for(cfg, mod)
        n = fro_product(B, A, scale)
        if not np.any(B):
            b_zero += 1
        modules.append({"module": mod, "delta_fro": n})
        sq_total += n * n
    other_sq = 0.0
    for k in other:
        v = w.get(k)
        other_sq += float(np.sum(v.astype(np.float64) ** 2))
    return {
        "n_lora_modules": len(pairs),
        "n_other_tensors": len(other),
        "delta_fro": math.sqrt(sq_total),
        "b_zero_modules": b_zero,
        "other_fro": math.sqrt(other_sq),
        "modules": modules,
    }


def compare(prev: Weights, cur: Weights, cfg: dict) -> dict:
    """Checkpoint-to-checkpoint movement, both in applied-delta and raw-parameter terms.

    Two different questions, both worth answering: ``moved`` is how much the function
    the adapter computes changed, ``changed_tensors`` is whether any number in the file
    is different at all. A run whose optimizer never stepped scores 0 on both; a run
    whose steps cancel out scores ~0 on the first and non-zero on the second.
    """
    p_pairs, p_other = lora_pairs(prev.keys())
    c_pairs, c_other = lora_pairs(cur.keys())

    sq, changed, total, max_abs = 0.0, 0, 0, 0.0
    per_module = []
    for mod in sorted(set(p_pairs) & set(c_pairs)):
        A1, B1 = prev.get(p_pairs[mod]["A"]), prev.get(p_pairs[mod]["B"])
        A2, B2 = cur.get(c_pairs[mod]["A"]), cur.get(c_pairs[mod]["B"])
        if A1.shape != A2.shape or B1.shape != B2.shape:
            continue  # rank changed between checkpoints; not comparable
        s = scaling_for(cfg, mod)
        d = fro_product_diff(B2, A2, s, B1, A1, s)
        per_module.append({"module": mod, "moved": d})
        sq += d * d
        for x1, x2 in ((A1, A2), (B1, B2)):
            total += 1
            m = float(np.max(np.abs(x2.astype(np.float64) - x1.astype(np.float64)))) if x1.size else 0.0
            max_abs = max(max_abs, m)
            if m > 0:
                changed += 1

    # Full weights: modules_to_save copies, or a whole non-LoRA checkpoint. These are
    # already the applied weights, so their diff norm adds to `moved` in the same units
    # as the LoRA deltas above.
    for k in sorted(set(p_other) & set(c_other)):
        v1, v2 = prev.get(k), cur.get(k)
        if v1.shape != v2.shape:
            continue
        total += 1
        if not v1.size:
            continue
        d = v2.astype(np.float64) - v1.astype(np.float64)
        sq += float(np.sum(d ** 2))
        m = float(np.max(np.abs(d)))
        max_abs = max(max_abs, m)
        if m > 0:
            changed += 1

    return {
        "moved": math.sqrt(sq),
        "max_abs_param_diff": max_abs,
        "changed_tensors": changed,
        "total_tensors": total,
        "modules": per_module,
        "only_in_prev": sorted(set(p_pairs) - set(c_pairs)),
        "only_in_cur": sorted(set(c_pairs) - set(p_pairs)),
    }


def analyse_run(run_dir: str, ckpts: list[str], base: BaseWeights | None,
                keep_modules: bool) -> dict:
    """Walk one run's checkpoints in step order, holding at most two open."""
    cfg = read_json(os.path.join(ckpts[0], "adapter_config.json"))
    rows: list[dict] = []
    problems: list[str] = []

    prev_w: Weights | None = None
    prev_row: dict | None = None
    try:
        for ck in ckpts:
            wf = weight_file(ck)
            row: dict = {"checkpoint": os.path.basename(ck), "step": step_of(ck),
                         "path": ck, "file": os.path.basename(wf) if wf else None}
            if wf is None:
                row["error"] = ("sharded full-model checkpoint (model.safetensors."
                                "index.json) -- not supported")
                problems.append(f"{row['checkpoint']}: {row['error']}")
                rows.append(row)
                continue
            if adapter_file(ck) is None:
                # Full-finetune checkpoint: no LoRA factorisation, so "vs base" is not
                # computable from this file alone. The checkpoint-to-checkpoint
                # comparison below still works, tensor by tensor.
                row["full_model_checkpoint"] = True
            try:
                cur = Weights(wf)
            except Exception as exc:
                row["error"] = str(exc)
                problems.append(f"{row['checkpoint']}: unreadable ({exc})")
                rows.append(row)
                continue

            row.update(analyse_checkpoint(cur, cfg))
            row["signals"] = trainer_signals(ck)

            if row.get("n_lora_modules") and row["b_zero_modules"] == row["n_lora_modules"]:
                problems.append(f"{row['checkpoint']}: lora_B is all-zero -- this adapter "
                                f"is a no-op, it cannot differ from the base model")
            if prev_w is not None:
                row["vs_prev"] = compare(prev_w, cur, cfg)
                row["vs_prev"]["checkpoint"] = prev_row["checkpoint"]
                if row["vs_prev"]["total_tensors"] and row["vs_prev"]["changed_tensors"] == 0:
                    problems.append(f"{row['checkpoint']}: byte-identical to "
                                    f"{prev_row['checkpoint']} -- no weight update landed "
                                    f"between these steps")

            if base is not None:
                num, den = 0.0, 0.0
                for m in row.get("modules", []):
                    bn = base.norm_of(m["module"])
                    if bn:
                        m["rel"] = m["delta_fro"] / bn
                        num += m["delta_fro"] ** 2
                        den += bn ** 2
                if den:
                    row["rel_to_base"] = math.sqrt(num) / math.sqrt(den)
                    row["matched_base_modules"] = sum(1 for m in row["modules"] if "rel" in m)

            if not keep_modules:
                row["modules"] = []
                if "vs_prev" in row:
                    row["vs_prev"]["modules"] = []
            rows.append(row)

            if prev_w is not None:
                prev_w.close()
            prev_w, prev_row = cur, row
    finally:
        if prev_w is not None:
            prev_w.close()

    scored = [r for r in rows if "delta_fro" in r]
    is_lora = any(r.get("n_lora_modules") for r in rows)
    if scored and is_lora and all(r["delta_fro"] == 0.0 for r in scored):
        problems.append("every checkpoint applies a zero delta -- the base model is "
                        "unchanged by all of them")
    return {
        "run_dir": run_dir,
        "is_lora": is_lora,
        "adapter": {
            "base_model": cfg.get("base_model_name_or_path"),
            "r": cfg.get("r"),
            "lora_alpha": cfg.get("lora_alpha"),
            "use_rslora": bool(cfg.get("use_rslora")),
            "target_modules": cfg.get("target_modules"),
            "modules_to_save": cfg.get("modules_to_save"),
        },
        "checkpoints": rows,
        "problems": problems,
    }


# --------------------------------------------------------------------------- printing

def fmt(x: float | None, width: int = 11) -> str:
    if x is None:
        return "-".rjust(width)
    if x == 0:
        return "0".rjust(width)
    return f"{x:.4e}".rjust(width)


def print_run(res: dict, label: str, top: int) -> None:
    a = res["adapter"]
    rows = res["checkpoints"]
    lora = res["is_lora"]
    print(label)
    if lora:
        scale = ""
        if a["r"] and a["lora_alpha"]:
            s = (a["lora_alpha"] / math.sqrt(a["r"])) if a["use_rslora"] else a["lora_alpha"] / a["r"]
            scale = f" scaling={s:g}"
        n_mod = next((r.get("n_lora_modules") for r in rows if r.get("n_lora_modules")), 0)
        print(f"  base={a['base_model']}  r={a['r']} alpha={a['lora_alpha']}{scale}  "
              f"{n_mod} LoRA modules  {len(rows)} checkpoints")
    else:
        print(f"  full-weight checkpoints (no LoRA adapter)  {len(rows)} checkpoints  "
              f"-- 'vs base' needs the base model, only the step-to-step diff is shown")
    print(f"  {res['run_dir']}")
    print()

    has_rel = any("rel_to_base" in r for r in rows)
    col1 = "||dW|| vs base" if lora else "||W||"
    head = (f"  {'step':>7}  {col1:>14}  {'moved vs prev':>13}  "
            f"{'max|dparam|':>11}  {'tensors moved':>13}")
    if has_rel:
        head += f"  {'dW/W':>9}"
    head += f"  {'lr':>9}  {'grad_norm':>9}"
    print(head)
    print("  " + "-" * (len(head) - 2))

    for r in rows:
        if "error" in r:
            print(f"  {r['step']:>7}  ! {r['error']}")
            continue
        vp = r.get("vs_prev")
        moved = fmt(vp["moved"], 13) if vp else "-".rjust(13)
        maxd = fmt(vp["max_abs_param_diff"], 11) if vp else "-".rjust(11)
        tens = (f"{vp['changed_tensors']}/{vp['total_tensors']}".rjust(13)
                if vp else "-".rjust(13))
        first = r.get("delta_fro") if lora else r.get("other_fro")
        line = (f"  {r['step']:>7}  {fmt(first, 14)}  {moved}  {maxd}  {tens}")
        if has_rel:
            rel = r.get("rel_to_base")
            line += f"  {(f'{rel:.2e}' if rel is not None else '-'):>9}"
        sig = r.get("signals", {})
        lr = sig.get("learning_rate")
        gn = sig.get("grad_norm")
        line += f"  {(f'{lr:.2e}' if lr is not None else '-'):>9}"
        line += f"  {(f'{gn:.3g}' if gn is not None else '-'):>9}"
        print(line)

    if top:
        last = next((r for r in reversed(rows) if r.get("modules")), None)
        if last:
            print(f"\n  top {top} modules by ||dW|| at {last['checkpoint']}:")
            for m in sorted(last["modules"], key=lambda m: -m["delta_fro"])[:top]:
                rel = f"   {m['rel']:.2e} of ||W||" if "rel" in m else ""
                print(f"    {m['delta_fro']:.4e}{rel}   {m['module']}")

    print()
    if res["problems"]:
        for p in res["problems"]:
            print(f"  PROBLEM: {p}")
    else:
        n = len([r for r in rows if "delta_fro" in r])
        each = "each " if n > 1 else ""
        applies = (f"{each}applies a non-zero delta to the base model"
                   if lora else "weights readable")
        differs = ", and each differs from the one before it" if n > 1 else ""
        print(f"  OK: {n} checkpoint{'s' if n != 1 else ''}, {applies}{differs}.")
    print()


# ------------------------------------------------------------------------------- main

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Check whether training checkpoints actually changed, from their "
                    "safetensors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Default target is $MARSHAL_RUNS (or the cluster default, as status.sh).")
    ap.add_argument("paths", nargs="*",
                    help="runs root, experiment dir, train dir, or a single checkpoint")
    ap.add_argument("--base", metavar="MODEL",
                    help="base model dir (or HF id already in the local cache) to size "
                         "the delta against: reports ||dW||/||W|| per module")
    ap.add_argument("--top", type=int, default=0, metavar="N",
                    help="also list the N modules with the largest delta")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any run has a problem (for scripts)")
    args = ap.parse_args(argv)

    roots = args.paths or [default_runs_root()]
    for root in roots:
        if not os.path.exists(root):
            print(f"ERROR: no such path: {root}", file=sys.stderr)
            return 2

    base = None
    if args.base:
        try:
            base = BaseWeights(args.base)
        except Exception as exc:
            print(f"ERROR: --base: {exc}", file=sys.stderr)
            return 2

    results, any_problem = [], False
    for root in roots:
        ckpts = find_checkpoints(root)
        if not ckpts:
            if not args.json:
                print(f"no checkpoints with weights under {root}")
                print("  (eval/<checkpoint-N>/ holds lm-eval output, not weights -- "
                      "the weights live under <exp>/train/)")
            continue
        for run_dir, run_ckpts in group_runs(ckpts):
            res = analyse_run(run_dir, run_ckpts, base, keep_modules=bool(args.top) or args.json)
            res["label"] = run_label(run_dir, root)
            results.append(res)
            any_problem |= bool(res["problems"])
            if not args.json:
                print_run(res, res["label"], args.top)

    if args.json:
        print(json.dumps({"base": args.base, "runs": results}, indent=2))
    elif len(results) > 1:
        print("=" * 72)
        for res in results:
            state = "PROBLEM" if res["problems"] else "ok"
            last = next((r for r in reversed(res["checkpoints"]) if "delta_fro" in r), None)
            dw = f"||dW||={last['delta_fro']:.4e}" if last else "no readable checkpoint"
            print(f"{state:>8}  {res['label']}  ({len(res['checkpoints'])} ckpt, {dw})")

    return 1 if (args.strict and any_problem) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

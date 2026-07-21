"""Translate Eddie's UUID-valued CUDA_VISIBLE_DEVICES into an integer CUDA ordinal.

Eddie's Grid Engine sets CUDA_VISIBLE_DEVICES to GPU UUID strings
(``GPU-beccd30f-...``) for cgroup-based isolation.

* **torch** handles that fine.
* **vLLM 0.12.0** does not: ``vllm/platforms/interface.py:211`` runs
  ``int(device_ids[device_id])`` and raises
  ``ValueError: invalid literal for int() with base 10: 'GPU-...'``, surfacing
  confusingly as "Error in inspecting model architecture 'Qwen3ForCausalLM'"
  because vLLM introspects in a subprocess.

Both read the *same* variable, so we need one value that satisfies both: the integer
ordinal naming the GPU we were actually allocated.

Do not try to derive that ordinal analytically. It depends on the isolation regime
(cgroup device restriction vs. env-var-only), on CUDA's device ordering, and on
whether NVML enumeration matches CUDA's. An earlier version of this script asked
torch for the ordinal->UUID map with the variable unset and exported the index it
found; on Eddie that produced an ordinal the job could not actually open, and torch
then reported no CUDA at all -- which transformers rejects, opaquely, as::

    ValueError: Your setup doesn't support bf16/gpu.

(the absent "You need Ampere+ GPU" suffix is the tell that CUDA was missing entirely,
not that the GPU was pre-Ampere).

So: **probe candidates in a subprocess and keep the first that genuinely works** --
CUDA available, and device 0's UUID equal to the one we were allocated. Candidates,
in order of likelihood:

  1. ``"0"``     -- correct under cgroup isolation, the common Eddie case
  2. the index torch reports with the variable unset -- correct without cgroups
  3. unset       -- vLLM treats absent/empty as "use device_id directly"

Each probe costs one torch import (slow on NFS-backed scratch, but once per job).

Usage (see _common.sh):
    python slurm_eddie/_resolve_gpu.py "$CUDA_VISIBLE_DEVICES"
Prints the winning value on stdout: an integer, or empty meaning "leave it unset".
Exit 1 with a diagnostic on stderr if nothing works.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

_PROBE = (
    "import json, torch;"
    "a = torch.cuda.is_available();"
    "n = torch.cuda.device_count() if a else 0;"
    "u = [str(getattr(torch.cuda.get_device_properties(i), 'uuid', '')) for i in range(n)];"
    "b = bool(torch.cuda.is_bf16_supported()) if a else False;"
    "print(json.dumps({'avail': a, 'count': n, 'uuids': u, 'bf16': b}))"
)


def _norm(u: str) -> str:
    """Normalize a UUID for comparison: drop the GPU-/MIG- prefix, lowercase."""
    u = u.strip()
    for prefix in ("GPU-", "MIG-"):
        if u.startswith(prefix):
            u = u[len(prefix):]
    return u.lower()


def probe(value: str | None) -> dict:
    """Run torch in a subprocess with CUDA_VISIBLE_DEVICES set to ``value``.

    ``None`` means unset. A subprocess is required because CUDA caches its device
    list on first initialization -- one process cannot re-evaluate the variable.
    """
    env = os.environ.copy()
    if value is None:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = value
    try:
        out = subprocess.run(
            [sys.executable, "-c", _PROBE],
            env=env, capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        return {"avail": False, "count": 0, "uuids": [], "bf16": False}
    for line in reversed(out.stdout.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {"avail": False, "count": 0, "uuids": [], "bf16": False}


def _ok(info: dict, wanted: list[str]) -> bool:
    """Usable if CUDA is up and the visible devices are exactly the allocated ones."""
    if not info.get("avail") or not info.get("count"):
        return False
    seen = [_norm(u) for u in info.get("uuids", []) if u]
    if not seen:
        # torch too old to expose .uuid -- accept on device count alone.
        return info["count"] == len(wanted)
    return seen[: len(wanted)] == wanted


def main() -> int:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print("usage: _resolve_gpu.py <uuid[,uuid...]>", file=sys.stderr)
        return 1
    wanted = [_norm(u) for u in sys.argv[1].split(",") if u.strip()]

    baseline = probe(None)
    if not baseline.get("avail"):
        print(
            "no CUDA device visible to torch even with CUDA_VISIBLE_DEVICES unset.\n"
            "  Is a GPU actually allocated? An interactive session needs, e.g.\n"
            "    qlogin -q gpu -l gpu=1 -l a100=true -l h_rt=02:00:00 ...",
            file=sys.stderr,
        )
        return 1

    candidates: list[str | None] = ["0"]
    seen = [_norm(u) for u in baseline.get("uuids", []) if u]
    for u in wanted:
        if u in seen:
            idx = str(seen.index(u))
            if idx not in candidates:
                candidates.append(idx)
    candidates.append(None)

    for cand in candidates:
        info = probe(cand)
        if _ok(info, wanted):
            if not info.get("bf16", True):
                print(
                    f"warning: {cand!r} works but reports no bf16 support; "
                    "pass --no-bf16 if training then fails.",
                    file=sys.stderr,
                )
            print("" if cand is None else cand)
            return 0

    print(
        f"none of {candidates!r} exposed the allocated GPU ({sys.argv[1]}).\n"
        f"  torch with the variable unset saw: {baseline.get('uuids')}\n"
        "  Leave CUDA_VISIBLE_DEVICES at its UUID and use --vllm-mode server, or\n"
        "  see notes/RUN_ON_EDDIE_playpen_marshal.md §6.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

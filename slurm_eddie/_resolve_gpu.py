"""Translate Eddie's UUID-valued CUDA_VISIBLE_DEVICES into integer CUDA ordinals.

Eddie's Grid Engine sets CUDA_VISIBLE_DEVICES to GPU UUID strings
(``GPU-beccd30f-...``) for cgroup-based isolation. vLLM 0.12.0 assumes integer
indices and does ``int(device_ids[device_id])``
(``vllm/platforms/interface.py:211``), which raises::

    ValueError: invalid literal for int() with base 10: 'GPU-beccd30f-...'

surfacing confusingly as "Error in inspecting model architecture 'Qwen3ForCausalLM'"
because vLLM does the inspection in a subprocess. torch, by contrast, handles UUIDs
fine -- which is why the model loads and only vLLM init dies.

This script runs with CUDA_VISIBLE_DEVICES *unset* (the caller strips it), asks CUDA
itself for the ordinal->UUID map, and prints the comma-separated ordinals matching
the UUIDs it was given. That is authoritative under either isolation regime:

* cgroups restrict the process to its allocated GPU -> CUDA sees one device at
  ordinal 0, and we print "0";
* no cgroup restriction -> CUDA sees every GPU, and we print the true index.

Guessing "0" or trusting nvidia-smi's NVML index would be wrong in one regime each.

Usage (see _common.sh):
    python slurm_eddie/_resolve_gpu.py "$SAVED_CUDA_VISIBLE_DEVICES"
Exit codes: 0 ok (ordinals on stdout), 1 no match / no CUDA (diagnostic on stderr).
"""

from __future__ import annotations

import os
import sys


def _norm(u: str) -> str:
    """Normalize a UUID for comparison: drop the GPU-/MIG- prefix, lowercase."""
    u = u.strip()
    for prefix in ("GPU-", "MIG-"):
        if u.startswith(prefix):
            u = u[len(prefix):]
    return u.lower()


def main() -> int:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print("usage: _resolve_gpu.py <uuid[,uuid...]>", file=sys.stderr)
        return 1
    wanted = [_norm(u) for u in sys.argv[1].split(",") if u.strip()]

    # Must not be set, or we would be asking CUDA to resolve the very thing we are
    # trying to translate. The caller unsets it; enforce it here too.
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    import torch

    if not torch.cuda.is_available():
        print("no CUDA devices visible to torch", file=sys.stderr)
        return 1

    available: dict[str, int] = {}
    for i in range(torch.cuda.device_count()):
        uuid = getattr(torch.cuda.get_device_properties(i), "uuid", None)
        if uuid is not None:
            available[_norm(str(uuid))] = i

    if not available:
        # torch too old to expose .uuid. With a single visible device the ordinal
        # is unambiguous; otherwise refuse to guess.
        if torch.cuda.device_count() == 1 and len(wanted) == 1:
            print("0")
            return 0
        print("torch does not expose device UUIDs; cannot map safely", file=sys.stderr)
        return 1

    ordinals = []
    for u in wanted:
        if u not in available:
            print(f"UUID {u} not among visible devices: {sorted(available)}", file=sys.stderr)
            return 1
        ordinals.append(str(available[u]))

    print(",".join(ordinals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

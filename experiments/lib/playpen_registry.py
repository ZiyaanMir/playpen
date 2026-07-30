"""Make one LoRA checkpoint addressable by ``playpen eval``, without touching the repo.

``playpen eval`` takes a *registered model name*, not a path: it resolves the name
against ``model_registry.json`` **in the current working directory** (plus clemcore's
packaged registry). A LoRA checkpoint is neither -- it is an adapter directory.

clemcore's ``huggingface_local`` backend already knows how to apply one: a spec whose
``model_config`` carries ``peft_model: <path>`` is loaded as
``PeftModel.from_pretrained(base, path)``
(``clemcore/backends/huggingface_local_api.py``). So all that is missing is a registry
entry pointing at the base weights with that key set.

This script writes that entry into a **private working directory** -- never into the
repo's own ``model_registry.json``. Two reasons: concurrent eval jobs would otherwise
race on one shared file, and a job killed halfway would leave the checkout modified.
The working directory is what the eval job ``cd``s into, which is also where
``clembench.log`` and any other CWD-relative output lands, so it all stays inside the
experiment.

It writes two files:

* ``model_registry.json`` -- one entry, ``model_name`` as given, ``huggingface_id`` =
  the base model, ``model_config.peft_model`` = the adapter (omitted for a base-model
  row, which is then just the untrained model under a distinct name).
* ``game_registry.json`` -- ``[{"benchmark_path": "<repo>/clembench"}]``, because the
  CWD is no longer the repo and clembench would otherwise not be discovered. clemcore
  expands a ``benchmark_path`` by scanning it for ``clemgame.json`` files, i.e. exactly
  what auto-discovery does from the repo root.

**The base entry is cloned, not invented.** ``model_registry.json`` in the repo already
carries the settings a model needs to play at all (``premade_chat_template``,
``eos_to_cull``, ``enable_thinking: false`` for Qwen3 -- a thinking-mode model emits
``<think>`` blocks that clembench parses as a malformed move and scores as ABORTED).
Synthesising a fresh entry would silently drop those and produce a run that looks
evaluated and scores zero. We look the base model up by ``huggingface_id`` first, then
by name, and only fall back to a minimal generated spec if it is genuinely unregistered
-- saying so loudly, since that fallback is the case most likely to score misleadingly.

Stdlib only, so it runs anywhere the training venv does.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(_HERE))

# Enough for a chat-tuned HF model to take its turn in a clembench game. Used only
# when the base model is in no registry at all; anything model-specific (thinking
# mode, a custom EOS) cannot be guessed and is the caller's problem to register.
FALLBACK_SPEC = {
    "backend": "huggingface_local",
    "open_weight": True,
    "languages": ["en"],
    "model_config": {
        "premade_chat_template": True,
        "eos_to_cull": "<\\|im_end\\|>",
    },
}


def _load_registry(path: str) -> list:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return []
    return data if isinstance(data, list) else [data]


def _packaged_specs() -> list:
    """clemcore's own bundled registry, if clemcore is importable."""
    try:
        import importlib.resources as importlib_resources

        with importlib_resources.files("clemcore.backends").joinpath(
            "model_registry.json"
        ).open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else [data]
    except Exception:
        return []


def find_base_spec(base_model: str, repo: str) -> tuple[dict | None, str]:
    """The registered spec for ``base_model``, and where it came from.

    Matching order, most specific first:
      1. exact ``huggingface_id`` -- what the adapter's ``base_model_name_or_path``
         actually names (e.g. ``Qwen/Qwen3-4B``);
      2. exact ``model_name``;
      3. ``model_name`` equal to the basename (``Qwen3-4B``), which is how the
         experiment scripts spell the model in directory names.
    The repo's registry is searched before clemcore's packaged one, matching
    ``ModelRegistry.from_packaged_and_cwd_files()``'s own precedence.
    """
    base_name = os.path.basename(base_model)
    sources = [
        (os.path.join(repo, "model_registry.json"), _load_registry(os.path.join(repo, "model_registry.json"))),
        ("clemcore packaged registry", _packaged_specs()),
    ]
    for how in ("huggingface_id", "model_name", "basename"):
        for origin, specs in sources:
            for spec in specs:
                if not isinstance(spec, dict):
                    continue
                if how == "huggingface_id" and spec.get("huggingface_id") == base_model:
                    return spec, origin
                if how == "model_name" and spec.get("model_name") == base_model:
                    return spec, origin
                if how == "basename" and spec.get("model_name") == base_name:
                    return spec, origin
    return None, ""


def build_spec(model_name: str, base_model: str, adapter: str | None, repo: str) -> tuple[dict, str]:
    found, origin = find_base_spec(base_model, repo)
    if found is not None:
        spec = copy.deepcopy(found)
        note = f"cloned registry entry '{found.get('model_name')}' from {origin}"
    else:
        spec = copy.deepcopy(FALLBACK_SPEC)
        note = (
            f"WARNING: '{base_model}' is in no model registry -- using a generic "
            f"huggingface_local spec. If this model needs enable_thinking=false or a "
            f"non-default eos_to_cull, add it to {os.path.join(repo, 'model_registry.json')} "
            f"or its scores will be meaningless."
        )

    spec["model_name"] = model_name
    spec["huggingface_id"] = base_model
    spec["backend"] = spec.get("backend") or "huggingface_local"
    # lookup_source is set by whichever registry loads the file; a stale value copied
    # from the source entry would point at the repo's registry and confuse debugging.
    spec.pop("lookup_source", None)

    model_config = dict(spec.get("model_config") or {})
    if adapter:
        model_config["peft_model"] = adapter
    else:
        model_config.pop("peft_model", None)
    spec["model_config"] = model_config
    return spec, note


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work-dir", required=True,
                    help="directory to write model_registry.json / game_registry.json into "
                         "(the eval job cds here)")
    ap.add_argument("--model-name", required=True,
                    help="name playpen eval will be called with; also the directory name "
                         "clembench files its results under, so keep it path-safe")
    ap.add_argument("--base-model", required=True,
                    help="base weights, e.g. Qwen/Qwen3-4B (read from the adapter, not assumed)")
    ap.add_argument("--adapter", default="",
                    help="LoRA checkpoint directory; omit for the untrained base row")
    ap.add_argument("--repo", default=REPO, help="playpen checkout (default: this one)")
    ap.add_argument("--clembench", default="",
                    help="clembench checkout (default: <repo>/clembench)")
    args = ap.parse_args()

    clembench = args.clembench or os.path.join(args.repo, "clembench")
    if not os.path.isdir(clembench):
        print(f"ERROR: no clembench checkout at {clembench}.", file=sys.stderr)
        print("       playpen eval has no games to run without it; clone it into the "
              "repo root or pass --clembench.", file=sys.stderr)
        raise SystemExit(1)
    if args.adapter and not os.path.isfile(os.path.join(args.adapter, "adapter_config.json")):
        print(f"ERROR: {args.adapter} is not a PEFT adapter (no adapter_config.json).",
              file=sys.stderr)
        raise SystemExit(1)

    os.makedirs(args.work_dir, exist_ok=True)
    spec, note = build_spec(args.model_name, args.base_model, args.adapter or None, args.repo)

    with open(os.path.join(args.work_dir, "model_registry.json"), "w", encoding="utf-8") as fh:
        json.dump([spec], fh, indent=2)
        fh.write("\n")
    with open(os.path.join(args.work_dir, "game_registry.json"), "w", encoding="utf-8") as fh:
        json.dump([{"benchmark_path": clembench}], fh, indent=2)
        fh.write("\n")

    print(f"[registry] {args.model_name} -> base={args.base_model}"
          f"{' + adapter=' + args.adapter if args.adapter else ' (no adapter: untrained base)'}")
    print(f"[registry] {note}")
    print(f"[registry] games from {clembench}")


if __name__ == "__main__":
    main()

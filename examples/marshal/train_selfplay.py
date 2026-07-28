"""Train a model with MARSHAL-style self-play on a clembench game (LoRA + GRPO).

Standalone script (like the other ``examples/`` scripts and the wordle notebook;
NOT run via ``playpen run``). It plays two-seat self-play episodes of a clembench
game, collects per-seat rollouts, and trains a single shared LoRA policy with
TRL's GRPOTrainer using MARSHAL's turn-level, per-seat-normalized advantages.

The MARSHAL behavior is a single switch you set BEFORE the run:
  * edit ``examples/marshal/marshal_config.yaml`` (or point ``--marshal-config``
    at your own), and/or
  * pass ``--no-marshal`` to force it off (plain TRL GRPO) without editing a file.

Dr. GRPO (arXiv:2503.20783) is a second, independent switch (``dr_grpo`` in the YAML,
or ``--dr-grpo``/``--no-dr-grpo``): it configures TRL's loss_type='dr_grpo' +
scale_rewards='none' to remove GRPO's length and difficulty biases. It composes with the
MARSHAL switch (on either the MARSHAL path or the plain-GRPO baseline) and is a no-op
when off. LoRA is always on (independent of both switches).

Prerequisites (into playpen's venv):
    uv pip install "trl>=0.28.0,<1.0.0" vllm
    uv pip install wandb          # only for --wandb
Custom rollouts require vLLM (``use_vllm=True``).

Metrics go to Weights & Biases with ``--wandb`` (or ``--report-to wandb``, or by
exporting ``WANDB_PROJECT``). The run carries the resolved MARSHAL config, so
ablation arms are comparable in the UI, and falls back to *offline* recording unless
a credential exists and the W&B API is reachable -- on a compute node it usually is
not. See the W&B section of ``examples/marshal/README.md``.

Example:
    python examples/marshal/train_selfplay.py \
        --model HuggingFaceTB/SmolLM2-135M-Instruct \
        --game taboo \
        --marshal-config examples/marshal/marshal_config.yaml \
        --num-generations 2 --per-device-batch-size 4 --max-steps 10 \
        --wandb --wandb-project my-dissertation
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from datetime import datetime

# Safe at module scope: playpen.marshal.config is deliberately stdlib + PyYAML only
# (no torch/trl/vllm), so importing it costs nothing and keeps --help working. The
# tuples below feed argparse `choices=`, so the accepted values can never drift from
# what MarshalConfig actually validates.
from playpen.marshal.config import (
    ADVANTAGE_NORM_MODES,
    EPISODE_PAIRING_MODES,
    FIDELITY_MODES,
    ROW_CONTEXT_MODES,
)

# Custom rollout_func + experimental openenv utils emit warnings; silence unless debugging.
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

# Every MarshalConfig field settable from the CLI, EXCEPT `enabled`, which keeps its
# historical --marshal/--no-marshal spelling. Each name here is both the argparse dest
# and the dataclass field, so the merge in main() is a single dataclasses.replace and
# a new config field cannot be added without also being wired up here.
MARSHAL_CLI_FIELDS = (
    "agent_specific_normalization",
    "turn_level_rewards",
    "advantage_norm_mode",
    "gamma",
    "fidelity_mode",
    "whiten_rewards",
    "whiten_advantages",
    "row_context_mode",
    "episode_pairing",
    "dr_grpo",
    "length_penalty",
    "length_penalty_coef",
    "length_penalty_bonus",
    "length_penalty_min_len",
    "length_penalty_max_len",
    "length_penalty_offset",
    "sampling_top_p",
    "sampling_top_k",
)


def resolve_marshal_config(args, config_path=None):
    """Load the YAML and apply every CLI flag that was actually passed.

    Module-level rather than inline in :func:`main` so the precedence rules are
    testable without loading a model. Returns ``(config, overrides)``; ``overrides`` is
    what came from the CLI, for logging.

    Precedence, lowest to highest: dataclass defaults, the YAML, then any flag actually
    passed. A flag left off is ``None`` and changes nothing.

    All fields go through ONE ``dataclasses.replace`` so ``__post_init__`` re-validates
    the MERGED result -- the max_len > min_len check, the enum checks, the top_p range --
    rather than only the values that came from the file. (Assigning to attributes
    instead, as this used to do for ``enabled``/``dr_grpo``, silently skips that.)
    """
    from playpen.marshal.config import MarshalConfig

    cfg = MarshalConfig.from_yaml(config_path or args.marshal_config)
    overrides = {
        name: getattr(args, name)
        for name in MARSHAL_CLI_FIELDS
        if getattr(args, name, None) is not None
    }
    # `enabled` keeps its historical --marshal/--no-marshal spelling.
    if getattr(args, "marshal", None) is not None:
        overrides["enabled"] = args.marshal
    # --no-sampling-truncation is a convenience spelling for "both neutral". setdefault,
    # not assignment, so an explicit --sampling-top-p/--sampling-top-k still wins if both
    # are passed -- consistent with every other flag here, where what you typed applies.
    if getattr(args, "no_sampling_truncation", False):
        overrides.setdefault("sampling_top_p", 1.0)
        overrides.setdefault("sampling_top_k", 0)
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    return cfg, overrides


def _add_bool_override(parser, name, help_on):
    """Add a ``--name`` / ``--no-name`` pair that defaults to None.

    None (rather than False) is what makes "flag not passed" distinguishable from
    "flag passed as off" -- only the former leaves the YAML value alone.
    """
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", default=None, help=help_on)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false",
                       help=f"Force {dest} OFF (overrides the YAML).")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct",
                   help="HuggingFace model id (or local path) for the shared self-play policy.")
    p.add_argument("--game", default="taboo", help="clembench game name (2-player recommended, e.g. taboo).")

    # MARSHAL switch
    p.add_argument("--marshal-config", default=os.path.join(os.path.dirname(__file__), "marshal_config.yaml"),
                   help="Path to a MARSHAL config YAML.")
    switch = p.add_mutually_exclusive_group()
    switch.add_argument("--marshal", dest="marshal", action="store_true", default=None,
                        help="Force MARSHAL behavior ON (overrides the YAML 'enabled').")
    switch.add_argument("--no-marshal", dest="marshal", action="store_false",
                        help="Force MARSHAL behavior OFF (plain TRL GRPO).")

    # Dr. GRPO switch (overrides the YAML 'dr_grpo'). Applies to both the MARSHAL
    # path and the --no-marshal baseline: it sets TRL loss_type='dr_grpo' (constant-
    # normalized loss -> no length bias) + scale_rewards='none' (no std division ->
    # no difficulty bias). Off => TRL defaults (loss_type=dapo, scale_rewards=group),
    # identical to before this flag existed.
    drgrpo = p.add_mutually_exclusive_group()
    drgrpo.add_argument("--dr-grpo", dest="dr_grpo", action="store_true", default=None,
                        help="Force Dr. GRPO ON (overrides the YAML 'dr_grpo').")
    drgrpo.add_argument("--no-dr-grpo", dest="dr_grpo", action="store_false",
                        help="Force Dr. GRPO OFF (TRL default loss/scaling).")

    # --- the rest of the MARSHAL algorithm config ----------------------------------
    # Every remaining MarshalConfig field, so a cluster sweep can vary any of them
    # without editing the YAML. Omitting a flag leaves the YAML value untouched.
    _add_bool_override(
        p, "agent-specific-normalization",
        "Pool/normalize advantages PER SEAT (MARSHAL's headline contribution). "
        "--no-... keeps turn-level credit but pools batch-wide (ablation).")
    _add_bool_override(
        p, "turn-level-rewards",
        "Keep each turn's reward at its own turn boundary (turn-level estimator). "
        "--no-... sums a seat's turn rewards into one terminal scalar.")
    _add_bool_override(
        p, "whiten-rewards",
        "Z-score the token-level reward field batch-wide BEFORE the cumulative sum "
        "(ROLL's whiten_rewards).")
    _add_bool_override(
        p, "whiten-advantages",
        "Z-score the final advantages batch-wide AFTER per-seat normalization "
        "(ROLL's whiten_advantages).")
    p.add_argument("--advantage-norm-mode", choices=ADVANTAGE_NORM_MODES, default=None,
                   help="'mean' mean-centers each pool; 'mean_std' z-scores it. Note "
                        "--dr-grpo realigns 'mean_std' back to 'mean'.")
    p.add_argument("--gamma", type=float, default=None,
                   help="Discount for the backward cumulative return. MARSHAL uses 1.0.")
    p.add_argument("--fidelity-mode", choices=FIDELITY_MODES, default=None,
                   help="'paper_correct' = the algorithm the paper describes; "
                        "'marshal_exact' = MARSHAL's shipped code, incl. its distinct-value "
                        "pooling and pre-sum reward normalization.")
    p.add_argument("--row-context-mode", choices=ROW_CONTEXT_MODES, default=None,
                   help="How a seat's row context is assembled.")
    p.add_argument("--episode-pairing", choices=EPISODE_PAIRING_MODES, default=None,
                   help="How the two seats of an episode are paired into rollout rows.")

    # Length penalty (MARSHAL's compute_length_penalty; overrides the YAML fields).
    # Off by default, which is also the MARSHAL-faithful setting: MARSHAL applies this
    # only to its board-game envs and explicitly skips it for free-text ones like
    # clembench (env_manager.py:470-480). Turn it on to discipline a reasoning model
    # that overruns its per-turn budget -- and say so when reporting the run.
    lenpen = p.add_mutually_exclusive_group()
    lenpen.add_argument("--length-penalty", dest="length_penalty", action="store_true", default=None,
                        help="Force the per-turn length penalty ON (overrides the YAML 'length_penalty').")
    lenpen.add_argument("--no-length-penalty", dest="length_penalty", action="store_false",
                        help="Force the per-turn length penalty OFF.")
    p.add_argument("--length-penalty-coef", type=float, default=None,
                   help="Penalty magnitude (MARSHAL's 'lower', ships 0.5): at twice the "
                        "threshold the turn reward is docked by about this much. clembench "
                        "outcomes are +1/0/-1, so 0.5 is already a strong signal.")
    p.add_argument("--length-penalty-max-len", type=int, default=None,
                   help="Per-turn token count above which a generation is penalized "
                        "(MARSHAL's 'max_len', ships 2048). Set near --max-completion-length "
                        "so only genuine overruns are charged.")
    p.add_argument("--length-penalty-bonus", type=float, default=None,
                   help="Scale on the positive branch (MARSHAL's 'upper', ships 0.0 = no "
                        "reward for brevity). Nonzero pays the policy to be short, which "
                        "with a strict clembench parser can degenerate into empty answers.")
    p.add_argument("--length-penalty-min-len", type=int, default=None,
                   help="Origin of the penalty slope (MARSHAL's 'min_len', ships 11). "
                        "Near-irrelevant while --length-penalty-bonus is 0.")
    p.add_argument("--length-penalty-offset", type=float, default=None,
                   help="Vertical offset of the raw term (MARSHAL's 'coef', ships 1.0). "
                        "Prefer changing --length-penalty-max-len instead.")

    # Sampling-tail truncation (maps onto GRPOConfig.top_p / top_k; overrides the YAML).
    # Applies on both the MARSHAL path and the --no-marshal baseline, because it
    # configures generation rather than the advantage math. Defaults on (0.95 / 50):
    # untruncated sampling from a low-entropy policy occasionally draws a far-tail
    # token whose log-prob vLLM and the trainer disagree about by many nats, which
    # sequence-level importance sampling then turns into a dead row.
    p.add_argument("--sampling-top-p", type=float, default=None,
                   help="Nucleus cutoff for generation (YAML 'sampling_top_p', ships "
                        "0.95). 1.0 disables nucleus truncation, i.e. TRL's default.")
    p.add_argument("--sampling-top-k", type=int, default=None,
                   help="Top-k cutoff for generation (YAML 'sampling_top_k', ships 50). "
                        "0 disables top-k truncation, i.e. TRL's default.")
    p.add_argument("--no-sampling-truncation", dest="no_sampling_truncation",
                   action="store_true",
                   help="Revert to TRL's untruncated sampling (top_p=1.0, top_k=0). "
                        "Use to reproduce a pre-2026-07-28 run; note that with "
                        "--kl-beta 0 this is what lets the importance-sampling ratio "
                        "decay as the policy's entropy collapses.")

    # LoRA (always applied)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-modules-to-save", default="",
                   help="Comma-separated extra modules to train FULLY (e.g. 'lm_head,embed_token'). "
                        "Empty by default: on tied-embedding models with a large vocab these add a "
                        "full fp32 copy + Adam states (GBs) and trigger peft's tie_word_embeddings "
                        "warning. Only needed if you changed the tokenizer/vocab.")

    # Memory
    p.add_argument("--no-bf16", dest="bf16", action="store_false", default=True,
                   help="Disable bf16 (default: on). bf16 halves the training copy's VRAM.")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="Trade compute for activation memory. Useful on small GPUs.")

    # GRPO / generation
    p.add_argument("--kl-beta", type=float, default=0.0,
                   help="KL regularization coefficient (maps onto TRL GRPOConfig.beta). 0 disables "
                        "(default, and TRL's default). Nonzero adds beta * KL(policy || base model) "
                        "to the per-token loss via TRL's k3 estimator; with LoRA the reference is the "
                        "base weights with adapters disabled, so no extra model copy is loaded. "
                        "MARSHAL's shipped self-play configs use 0.20 -- recommended if self-play "
                        "runs start to drift/collapse.")
    p.add_argument("--num-generations", type=int, default=2)
    p.add_argument("--per-device-batch-size", type=int, default=4,
                   help="Per-forward micro-batch. NOTE the MARSHAL advantage pool is the "
                        "*generation* batch (per-device-batch-size x grad-accum), not this "
                        "value, so that product is what must be >= 2*num_generations for "
                        "both seats to co-occur in a pool; it must also be divisible by "
                        "--num-generations.")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--max-completion-length", type=int, default=2048)
    p.add_argument("--max-turns", type=int, default=100, help="Safety cap on turns per episode.")
    p.add_argument("--max-instances", type=int, default=None, help="Limit number of game instances used.")
    p.add_argument("--report-to", default="none", choices=["none", "tensorboard", "wandb"],
                   help="Metrics sink. Default 'none' (metrics still print to console). "
                        "'tensorboard' requires `pip install tensorboard`. 'wandb' is "
                        "equivalent to --wandb and composes with the --wandb-* flags below.")

    # Weights & Biases. Off unless asked for (--wandb, --report-to wandb, or a
    # WANDB_PROJECT in the environment). Every flag falls back to its WANDB_* env var,
    # so a job script can export them once; see playpen/marshal/wandb_utils.py.
    wb = p.add_mutually_exclusive_group()
    wb.add_argument("--wandb", dest="wandb", action="store_true", default=None,
                    help="Log this run to Weights & Biases (composes with --report-to, "
                         "so tensorboard and wandb can both be on).")
    wb.add_argument("--no-wandb", dest="wandb", action="store_false",
                    help="Force W&B off even if WANDB_PROJECT is set in the environment.")
    p.add_argument("--wandb-project", default=None,
                   help="W&B project (env: WANDB_PROJECT). Default 'playpen-marshal'.")
    p.add_argument("--wandb-entity", default=None,
                   help="W&B team or username (env: WANDB_ENTITY). Default: your account's.")
    p.add_argument("--wandb-run-name", default=None,
                   help="Run display name (env: WANDB_NAME). Defaults to "
                        "'{game}_{model}_{timestamp}', matching the on-disk run folder.")
    p.add_argument("--wandb-group", default=None,
                   help="Group related runs (env: WANDB_RUN_GROUP). Defaults to "
                        "'{game}_{model}', which puts an ablation's arms side by side.")
    p.add_argument("--wandb-job-type", default=None,
                   help="Job type within the group (env: WANDB_JOB_TYPE). Default 'train'.")
    p.add_argument("--wandb-tags", default=None,
                   help="Comma-separated tags (env: WANDB_TAGS). The switch settings "
                        "(marshal/dr_grpo/length_penalty/fidelity) are tagged automatically.")
    p.add_argument("--wandb-mode", default=None, choices=["auto", "online", "offline", "disabled"],
                   help="'auto' (default) uploads live when a credential exists "
                        "(WANDB_API_KEY or ~/.netrc) AND the W&B API answers, and records "
                        "offline otherwise -- the safe choice on a compute node, whose "
                        "shared $HOME carries the credential but which usually has no "
                        "outbound network. 'disabled' turns W&B off entirely.")
    p.add_argument("--wandb-dir", default=None,
                   help="Where the wandb/ run directory goes (env: WANDB_DIR). Defaults to "
                        "this run's output dir, keeping the run folder self-contained.")
    p.add_argument("--wandb-id", default=None,
                   help="Explicit W&B run id (env: WANDB_RUN_ID). Only needed to resume.")
    p.add_argument("--wandb-resume", default=None, choices=["allow", "must", "never"],
                   help="Resume policy for --wandb-id (env: WANDB_RESUME). Use "
                        "'allow' with a fixed id so a requeued job continues one run.")
    p.add_argument("--wandb-notes", default=None,
                   help="Free-text note stored on the run (env: WANDB_NOTES).")

    # vLLM
    p.add_argument("--vllm-mode", default="colocate", choices=["colocate", "server"],
                   help="'colocate' shares the training GPU; 'server' talks to a separate `trl vllm-serve` "
                        "(recommended workaround for the LoRA+colocate hang, TRL issue #3671).")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3)
    p.add_argument("--vllm-max-model-len", type=int, default=4096,
                   help="Cap vLLM's context window (prompt+completion). Defaults to 4096 rather than "
                        "the model's advertised max (e.g. Qwen3's 40960), because vLLM sizes its KV "
                        "cache to serve one request at full length -- 40960 needs ~4.4GiB of KV cache. "
                        "Must exceed your longest episode's prompt+completion.")
    p.add_argument("--vllm-model-impl", default="vllm", choices=["vllm", "transformers"],
                   help="vLLM model backend. Use 'transformers' for architectures vLLM's own "
                        "registry doesn't know yet (e.g. Qwen3.5 -> Qwen3_5ForConditionalGeneration); "
                        "vLLM then runs it via its Transformers fallback. Slower than native 'vllm'.")

    # Output
    p.add_argument("--save-steps", type=int, default=500,
                   help="Save a checkpoint every N optimizer steps (e.g. 20). Default 500 (TRL/HF's "
                        "own default, i.e. unchanged behavior). A final checkpoint is always written "
                        "when training ends, regardless of this value.")
    p.add_argument("--output-dir", default=None,
                   help="Base output directory. A timestamped run subfolder is created inside "
                        "it by default, so successive runs never overwrite each other. "
                        "Default base: models/marshal/{game}/{model_basename}.")
    p.add_argument("--no-run-subdir", dest="run_subdir", action="store_false", default=True,
                   help="Write checkpoints directly into --output-dir instead of into a "
                        "timestamped subfolder. Use ONLY when the caller already guarantees a "
                        "fresh directory per run (experiments/*/train.sh does); otherwise a "
                        "rerun overwrites the previous run's checkpoint-<step> dirs.")
    return p.parse_args()


def wandb_tags(args: argparse.Namespace, marshal_config) -> list[str]:
    """Tags that make the W&B run list filterable without opening any run.

    Every tag encodes a *switch*, not a magnitude: which arm of an ablation this
    is. Magnitudes (learning rate, batch, steps) go to the run config, where the
    UI can already sort and group by them.
    """
    tags = [
        args.game,
        os.path.basename(args.model),
        "marshal" if marshal_config.enabled else "no-marshal",
        marshal_config.fidelity_mode,
    ]
    if marshal_config.dr_grpo:
        tags.append("dr_grpo")
    if marshal_config.length_penalty:
        tags.append("length_penalty")
    if not marshal_config.agent_specific_normalization:
        tags.append("no-seat-norm")
    if args.kl_beta:
        tags.append("kl")
    # Set by experiments/*/run_experiment.sh; absent for a hand-launched run.
    for env_name in ("EXP_CLUSTER", "EXP_TAG"):
        value = os.environ.get(env_name, "").strip()
        if value:
            tags.append(value)
    return tags


def wandb_config(args: argparse.Namespace, marshal_config, output_dir: str) -> dict:
    """The run config: everything needed to tell two runs apart, in one place.

    Deliberately includes the *resolved* MARSHAL config (YAML merged with CLI
    overrides) rather than the path to the YAML -- the shared YAML is edited between
    runs, so a recorded path says nothing about what a run six weeks ago actually
    did. HF's callback adds the full ``GRPOConfig`` on top of this.
    """
    return {
        "marshal": marshal_config.to_dict(),
        "train": {
            "model": args.model,
            "game": args.game,
            "num_generations": args.num_generations,
            "per_device_batch_size": args.per_device_batch_size,
            "grad_accum": args.grad_accum,
            "generation_batch": args.per_device_batch_size * args.grad_accum,
            "max_steps": args.max_steps,
            "save_steps": args.save_steps,
            "learning_rate": args.learning_rate,
            "kl_beta": args.kl_beta,
            "max_completion_length": args.max_completion_length,
            "max_turns": args.max_turns,
            "max_instances": args.max_instances,
            "bf16": args.bf16,
            "gradient_checkpointing": args.gradient_checkpointing,
        },
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "modules_to_save": args.lora_modules_to_save or None,
        },
        "vllm": {
            "mode": args.vllm_mode,
            "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "max_model_len": args.vllm_max_model_len,
            "model_impl": args.vllm_model_impl,
        },
        "run": {
            "output_dir": output_dir,
            "marshal_config_path": args.marshal_config,
            "experiment_id": os.environ.get("EXP_ID") or None,
            "experiment_dir": os.environ.get("EXP_DIR") or None,
            "cluster": os.environ.get("EXP_CLUSTER") or None,
            "job_id": os.environ.get("SLURM_JOB_ID") or os.environ.get("JOB_ID") or None,
        },
    }


def main() -> None:
    args = parse_args()

    import trl
    from peft import LoraConfig

    from playpen.marshal import (
        MarshalConfig,
        MarshalGRPOTrainer,
        SelfPlayEnv,
        WandbSettings,
        build_reward_func,
        build_selfplay_dataset,
        build_selfplay_rollout_func,
    )
    from playpen.marshal import wandb_utils

    # 1. MARSHAL config: the YAML is the source of truth, and any CLI flag that was
    #    actually passed overrides it. A flag left off is None, so it changes nothing.
    #
    #    All fields go through ONE dataclasses.replace so __post_init__ re-validates the
    #    MERGED result -- e.g. the max_len > min_len check, and the enum checks -- rather
    #    than only the values that came from the file. (Assigning to attributes instead,
    #    as this used to do for `enabled`/`dr_grpo`, silently skips that validation.)
    marshal_config, overrides = resolve_marshal_config(args)
    if overrides:
        print(f"[marshal] CLI overrides: "
              + " ".join(f"{k}={v}" for k, v in sorted(overrides.items())))
    # Reconcile aligns advantage_norm_mode 'mean_std' -> 'mean' when dr_grpo is on (keeps
    # per-seat pooling; drops only the std divisor Dr. GRPO removes). Must run AFTER the
    # merge so it sees the final values. No-op when dr_grpo is off.
    for notice in marshal_config.reconcile_for_dr_grpo():
        print(f"[dr_grpo] {notice}")
    print(f"[marshal] enabled={marshal_config.enabled} "
          f"agent_specific={marshal_config.agent_specific_normalization} "
          f"turn_level={marshal_config.turn_level_rewards} "
          f"advantage_norm_mode={marshal_config.advantage_norm_mode} "
          f"fidelity={marshal_config.fidelity_mode} "
          f"whiten_rewards={marshal_config.whiten_rewards} "
          f"whiten_advantages={marshal_config.whiten_advantages} "
          f"row_context_mode={marshal_config.row_context_mode} "
          f"episode_pairing={marshal_config.episode_pairing}")
    _sampling = marshal_config.trl_sampling_overrides()
    print(f"[sampling] top_p={marshal_config.sampling_top_p} "
          f"top_k={marshal_config.sampling_top_k}"
          + (f" -> GRPOConfig{_sampling}" if _sampling
             else " (untruncated: TRL defaults, no key set)"))
    if not _sampling:
        print("[sampling] NOTE: full-vocabulary sampling. With --kl-beta 0 the policy's "
              "entropy collapses over training and rare far-tail draws make vLLM and the "
              "trainer disagree by many nats on single tokens, which sequence-level "
              "importance sampling turns into dead rows. Watch "
              "sampling/importance_sampling_ratio/min and marshal/is_ratio/mean.")
    if marshal_config.row_context_mode != "exact":
        print("[marshal] WARNING: row_context_mode='spliced' reproduces the pre-fix "
              "rollout assembly, where the trained token sequence is not the sequence "
              "the policy generated under from turn 2 onward. TRL's vLLM "
              "importance-sampling correction then collapses the gradient. Use it only "
              "to reproduce an old run.")
    if marshal_config.length_penalty:
        from playpen.marshal.advantage import LengthPenaltySpec

        _spec = LengthPenaltySpec(**marshal_config.length_penalty_kwargs())
        _probe = marshal_config.length_penalty_max_len * 2
        print(f"[length_penalty] ON coef={marshal_config.length_penalty_coef} "
              f"max_len={marshal_config.length_penalty_max_len} "
              f"bonus={marshal_config.length_penalty_bonus} "
              f"min_len={marshal_config.length_penalty_min_len} "
              f"offset={marshal_config.length_penalty_offset} "
              f"-> a {_probe}-token turn scores {_spec.penalty_for(_probe):+.3f}")
        if not marshal_config.enabled:
            print("[length_penalty] WARNING: --no-marshal is set, so the MARSHAL advantage "
                  "path is off and the length penalty has NO effect (it is applied inside "
                  "compute_marshal_advantages).")
        print("[length_penalty] NOTE: MARSHAL skips this penalty for free-text envs like "
              "clembench (env_manager.py:470-480); this run is a deliberate divergence.")
    else:
        print("[length_penalty] off (MARSHAL-faithful default for clembench games)")
    print(f"[kl] beta={args.kl_beta}" + (" (disabled)" if args.kl_beta == 0.0 else ""))
    _dr_overrides = marshal_config.trl_grpo_overrides()
    print(f"[dr_grpo] enabled={marshal_config.dr_grpo}"
          + (f" -> {_dr_overrides}" if _dr_overrides
             else " (off; TRL defaults: loss_type=dapo, scale_rewards=group)"))

    # 2. Self-play env (persistent across rollout calls) + dataset from packaged instances.
    env = SelfPlayEnv(args.game)
    print(f"[env] game={args.game} num_players={env.num_players}")
    dataset = build_selfplay_dataset(
        args.game,
        num_players=env.num_players,
        max_instances=args.max_instances,
        episode_pairing=marshal_config.episode_pairing,
    )
    if marshal_config.episode_pairing == "shared":
        print(f"[data] {len(dataset)} rows (1 per game instance); each run of "
              f"{env.num_players} consecutive copies is served by ONE episode, so both "
              f"seats come from the same game and no generation is discarded")
        if args.num_generations % env.num_players != 0:
            print(f"[data] WARNING: --num-generations {args.num_generations} is not a "
                  f"multiple of num_players {env.num_players}; runs that cannot be paired "
                  f"fall back to replaying a fresh episode per prompt. Pick a multiple "
                  f"of {env.num_players} to pair every row.")
    else:
        print(f"[data] {len(dataset)} rows ({env.num_players} seats x game instances); "
              f"episode_pairing='replay' -- every prompt replays a whole episode and "
              f"discards the other seat(s)")

    # 3. GRPO config + LoRA (LoRA is always on, independent of the MARSHAL switch).
    # Every run writes into its own timestamped subfolder, so a rerun can never overwrite
    # a previous run's checkpoints (HF writes checkpoints as <output_dir>/checkpoint-<step>,
    # which collide across runs otherwise). Mirrors MARSHAL's runs/<experiment>/<timestamp>/.
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = args.output_dir or f"models/marshal/{args.game}/{os.path.basename(args.model)}"
    # --no-run-subdir drops the timestamp layer, so checkpoints land directly in the
    # caller's directory. Safe only because the caller owns a fresh dir per run.
    output_dir = os.path.join(base_dir, run_id) if args.run_subdir else base_dir

    # 3a. Weights & Biases. The run is opened HERE, before the trainer exists, for two
    # reasons: HF's WandbCallback only calls wandb.init when no run is open (so this is
    # what lets us set entity/group/tags/id/offline-dir at all), and a credential or
    # package problem then fails in the first seconds of the job instead of after the
    # model and vLLM have been loaded. Nothing below changes when W&B is off.
    wandb_settings = WandbSettings.from_args(args).with_defaults(
        output_dir=output_dir,
        run_id_stamp=run_id,
        game=args.game,
        model=args.model,
        extra_tags=wandb_tags(args, marshal_config),
    )
    print(wandb_settings.summary())
    wandb_run = wandb_settings.start(config=wandb_config(args, marshal_config, output_dir))
    report_to = wandb_settings.report_to(args.report_to)

    grpo_config = trl.GRPOConfig(
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_model_impl=args.vllm_model_impl,
        vllm_max_model_length=args.vllm_max_model_len,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        max_completion_length=args.max_completion_length,
        beta=args.kl_beta,
        disable_dropout=True,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        # Load the training copy directly in bf16 so it doesn't land in fp32 and
        # starve vLLM of VRAM in colocate mode.
        model_init_kwargs={"dtype": "bfloat16"} if args.bf16 else None,
        output_dir=output_dir,
        # Checkpoint every N steps. save_strategy is TRL/HF's default "steps", so this
        # alone controls the cadence; the last step always saves as well.
        save_steps=args.save_steps,
        report_to=report_to,
        # run_name is what HF's W&B callback would name the run; we already opened it
        # under this name, and setting it here keeps HF from warning that run_name
        # defaulted to output_dir.
        run_name=(wandb_settings.run_name if wandb_settings.enabled else None),
        logging_dir=(f"{output_dir}/tb" if "tensorboard" in report_to else None),
        log_completions=True,
        # Dr. GRPO recipe (loss_type + scale_rewards) when enabled; empty dict = no
        # change, so with dr_grpo off this call is identical to before the flag existed.
        **marshal_config.trl_grpo_overrides(),
        # Sampling-tail truncation (top_p / top_k). Same contract: an empty dict when
        # both are neutral, so --no-sampling-truncation reproduces the old GRPOConfig
        # exactly rather than passing top_p=1.0/top_k=0 explicitly.
        **marshal_config.trl_sampling_overrides(),
    )
    modules_to_save = [m.strip() for m in args.lora_modules_to_save.split(",") if m.strip()] or None
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        modules_to_save=modules_to_save,
        task_type="CAUSAL_LM",
    )

    # 4. Rollout + reward wiring.
    rollout_func = build_selfplay_rollout_func(env, marshal_config, max_turns=args.max_turns)
    reward_func = build_reward_func(marshal_config)

    print(f"[mem] bf16={args.bf16} grad_ckpt={args.gradient_checkpointing} "
          f"modules_to_save={modules_to_save} vllm_util={args.vllm_gpu_memory_utilization}")
    print(f"[out] run dir={output_dir} (checkpoint every {args.save_steps} steps, "
          f"plus a final one at step {args.max_steps})")

    trainer = MarshalGRPOTrainer(
        model=args.model,
        rollout_func=rollout_func,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=grpo_config,
        peft_config=peft_config,
        marshal_config=marshal_config,
    )

    try:
        import torch

        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            print(f"[mem] after model+vLLM init: {free_b / 2**30:.2f} GiB free "
                  f"of {total_b / 2**30:.2f} GiB")
    except Exception:
        pass

    # Record which W&B run this output directory belongs to before training starts:
    # a job killed at the walltime still leaves the pointer behind, which is when you
    # most want it. finish() runs in a `finally` for the same reason -- a crashed run
    # left open shows up in the UI as "running" forever.
    wandb_utils.write_run_metadata(
        os.path.join(output_dir, "wandb_run.json"), wandb_run, wandb_settings
    )
    try:
        trainer.train()
    finally:
        wandb_utils.finish(wandb_run, wandb_settings)
    print(f"[done] LoRA adapters saved under {output_dir}/checkpoint-<step>/")


if __name__ == "__main__":
    main()

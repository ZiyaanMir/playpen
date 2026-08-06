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
    TURN_REWARD_SOURCES,
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
    "marshal_exact_unique_pooling",
    "whiten_rewards",
    "whiten_advantages",
    "row_context_mode",
    "episode_pairing",
    "dr_grpo",
    "length_penalty",
    "length_penalty_per_token",
    "length_penalty_budget",
    # Inert legacy fields; still accepted so cluster presets and old command lines
    # keep running. See MarshalConfig.length_penalty_coef.
    "length_penalty_coef",
    "length_penalty_bonus",
    "length_penalty_min_len",
    "length_penalty_max_len",
    "length_penalty_offset",
    "sampling_top_p",
    "sampling_top_k",
    "turn_rewards",
    "turn_reward_source",
    "turn_reward_scale",
    "turn_reward_budget",
    "turn_reward_components",
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
    # Splits marshal_exact's two departures apart. Only meaningful under
    # --fidelity-mode marshal_exact; paper_correct never uniques, so this cannot turn
    # distinct-value pooling ON there.
    _add_bool_override(
        p, "marshal-exact-unique-pooling",
        "Pool over the SET OF DISTINCT trajectory returns (marshal_exact's torch.unique, "
        "the default). --no-... keeps marshal_exact's pre-sum reward normalization but "
        "pools occurrence-weighted like paper_correct. No effect under paper_correct.")
    p.add_argument("--row-context-mode", choices=ROW_CONTEXT_MODES, default=None,
                   help="How a seat's row context is assembled.")
    p.add_argument("--episode-pairing", choices=EPISODE_PAIRING_MODES, default=None,
                   help="How the two seats of an episode are paired into rollout rows.")

    # Length penalty (playpen/marshal/advantage.py:LengthPenaltySpec; overrides the
    # YAML fields). A small per-token cost with NO threshold, capped per episode so it
    # cannot outweigh the terminal reward. Off by default, which is also the
    # MARSHAL-faithful setting: MARSHAL applies its penalty only to its board-game envs
    # and explicitly skips it for free-text ones like clembench (env_manager.py:470-480).
    # Turn it on as gentle pressure against padding -- and say so when reporting the run.
    lenpen = p.add_mutually_exclusive_group()
    lenpen.add_argument("--length-penalty", dest="length_penalty", action="store_true", default=None,
                        help="Force the per-turn length penalty ON (overrides the YAML 'length_penalty').")
    lenpen.add_argument("--no-length-penalty", dest="length_penalty", action="store_false",
                        help="Force the per-turn length penalty OFF.")
    p.add_argument("--length-penalty-per-token", type=float, default=None,
                   help="Reward charged per generated token, from the first one (YAML "
                        "'length_penalty_per_token', ships 2e-5 = -0.02 per 1000 tokens). "
                        "Judge it against the +-1 outcome: a 10-turn episode of 500-token "
                        "turns costs -0.10. 0 makes the penalty inert.")
    p.add_argument("--length-penalty-budget", type=float, default=None,
                   help="Cap on |sum of one seat's length penalty over one episode| (YAML "
                        "'length_penalty_budget', ships 0.1). Under 1.0 this GUARANTEES the "
                        "penalty cannot reorder two episodes with different clembench "
                        "outcomes. 0 disables the cap and that guarantee.")

    # DEPRECATED, inert. These parameterized the previous threshold-based penalty
    # (a port of MARSHAL's compute_length_penalty). They are still parsed so cluster
    # presets and old command lines keep running; resolve_marshal_config records them
    # and main() warns once, loudly, that they do nothing.
    p.add_argument("--length-penalty-coef", type=float, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--length-penalty-max-len", type=int, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--length-penalty-bonus", type=float, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--length-penalty-min-len", type=int, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--length-penalty-offset", type=float, default=None,
                   help=argparse.SUPPRESS)

    # Dense per-turn rewards (playpen/marshal/turn_rewards.py; overrides the YAML).
    # Off by default, and with it off a run is byte-identical to before this feature
    # existed. Applies on BOTH the MARSHAL path and the --no-marshal baseline, since
    # it is a rollout-collection feature rather than advantage math.
    turnrew = p.add_mutually_exclusive_group()
    turnrew.add_argument("--turn-rewards", dest="turn_rewards", action="store_true", default=None,
                         help="Force dense per-turn rewards ON (overrides the YAML "
                              "'turn_rewards'). Reads a bounded per-turn signal off the "
                              "game's live state and adds it to that turn's reward.")
    turnrew.add_argument("--no-turn-rewards", dest="turn_rewards", action="store_false",
                         help="Force dense per-turn rewards OFF (terminal reward only).")
    p.add_argument("--turn-reward-source", choices=TURN_REWARD_SOURCES, default=None,
                   help="'auto' (default) = the game's own extractor, else generic "
                        "format compliance; 'game' = game-specific only (off, with a "
                        "warning, for an unregistered game); 'generic' = force format "
                        "compliance even where a richer extractor exists.")
    p.add_argument("--turn-reward-scale", type=float, default=None,
                   help="Multiplier on each turn's signal, which extractors normalize "
                        "to [-1, 1] -- so this is the most a single turn can contribute "
                        "(YAML 'turn_reward_scale', ships 0.05).")
    p.add_argument("--turn-reward-budget", type=float, default=None,
                   help="Cap on |sum of one seat's shaping over one episode| (YAML "
                        "'turn_reward_budget', ships 0.3). Below 0.5 this GUARANTEES "
                        "shaping cannot reorder two episodes with different clembench "
                        "outcomes. 0 disables the cap and that guarantee.")
    p.add_argument("--turn-reward-components", default=None,
                   help="Comma-separated allowlist of components to keep (e.g. "
                        "'closeness' for wordle progress without its format penalty). "
                        "Empty = all. An unknown name is an error, not a silent no-op.")

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

    # --- resuming, and splitting one run across several jobs ---------------------
    # Both clusters cap a job's walltime (48 h Eddie, 24 h Isambard) and neither cap
    # can be raised, so a long run has to be able to continue in a second job. See
    # playpen/marshal/resume.py for the full rationale, including why --max-steps
    # deliberately stays the TOTAL rather than being handed out per segment.
    p.add_argument("--resume-from-checkpoint", default=None,
                   help="Continue a previous run: 'auto' (resume the latest complete "
                        "checkpoint in --output-dir, or start at step 0 if there is "
                        "none), 'latest' (the same, but an ERROR when there is none), "
                        "or a path to a checkpoint-<N> directory (or to a directory "
                        "containing them). Restores the LoRA adapter, optimizer and "
                        "scheduler state, RNG and the step counter. 'auto'/'latest' "
                        "need --no-run-subdir, since they mean 'continue THIS directory'.")
    p.add_argument("--stop-at-step", type=int, default=None,
                   help="End this job once global step N is reached, saving a checkpoint "
                        "first so the next job can resume from it. --max-steps stays the "
                        "TOTAL horizon for the whole chain, which is what keeps the LR "
                        "schedule identical to an uninterrupted run: HF builds the "
                        "scheduler for max_steps, and scheduler.pt restores only the step "
                        "counter, never the decay curve. Use with "
                        "--resume-from-checkpoint auto.")
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
    if marshal_config.turn_rewards:
        tags.append("turn_rewards")
        tags.append(f"turn_rewards_{marshal_config.turn_reward_source}")
    if not marshal_config.agent_specific_normalization:
        tags.append("no-seat-norm")
    # Only when it actually changed the pooling rule -- under paper_correct the
    # sub-flag is inert, and a tag there would make two identical runs look different.
    if marshal_config.marshal_exact and not marshal_config.marshal_exact_unique_pooling:
        tags.append("no-unique-pool")
    if args.kl_beta:
        tags.append("kl")
    # Set by experiments/*/run_experiment.sh; absent for a hand-launched run.
    for env_name in ("EXP_CLUSTER", "EXP_TAG"):
        value = os.environ.get(env_name, "").strip()
        if value:
            tags.append(value)
    return tags


def wandb_config(args: argparse.Namespace, marshal_config, output_dir: str,
                 resume_path: str | None = None, stop_at: int | None = None) -> dict:
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
            # A chained run logs every segment into ONE W&B run (the job scripts pass a
            # fixed --wandb-id + --wandb-resume allow), so without these two the UI
            # would show a single continuous curve with no record that it was stitched.
            # The last segment to write wins, which is the one you are looking at.
            "resumed_from": resume_path,
            "stop_at_step": stop_at,
            "segment": os.environ.get("TRAIN_SEGMENT") or None,
            "segments_total": os.environ.get("TRAIN_SEGMENTS") or None,
            "experiment_id": os.environ.get("EXP_ID") or None,
            "experiment_dir": os.environ.get("EXP_DIR") or None,
            "cluster": os.environ.get("EXP_CLUSTER") or None,
            "job_id": os.environ.get("SLURM_JOB_ID") or os.environ.get("JOB_ID") or None,
        },
    }


def _report_length_penalty(marshal_config, max_completion_length: int) -> None:
    """Print what the length penalty will actually be worth, and warn.

    The per-token rate is not a number anyone can eyeball against a ``+-1`` game
    outcome, so this prints the two derived figures that matter: what one
    generation-capped turn costs, and what a whole episode can cost (which is what
    the backward cumulative return puts up against the outcome). It also reports
    inert legacy threshold fields, so a preset still carrying the old per-game
    calibration says so once instead of quietly doing nothing.
    """
    if not marshal_config.length_penalty:
        print("[length_penalty] off (MARSHAL-faithful default for clembench games)")
        _report_legacy_length_penalty_fields(marshal_config)
        return

    from playpen.marshal.advantage import LengthPenaltySpec

    spec = LengthPenaltySpec(**marshal_config.length_penalty_kwargs())
    per_capped_turn = spec.penalty_for(max_completion_length)
    print(f"[length_penalty] ON per_token={spec.per_token:g} budget={spec.budget} "
          f"(no threshold: every generated token is charged)")
    print(f"[length_penalty] a turn at the {max_completion_length}-token generation cap "
          f"scores {per_capped_turn:+.4f}; an episode's total is capped at "
          f"{-spec.budget:+.3f} against a +-1 outcome"
          + ("" if spec.budget else " (UNCAPPED)"))
    if spec.per_token == 0.0:
        print("[length_penalty] WARNING: per_token is 0, so the penalty is INERT. Set "
              "--length-penalty-per-token, or --no-length-penalty to be explicit.")
    if not marshal_config.length_penalty_budget_is_safe:
        print("[length_penalty] WARNING: budget >= 0.5 (or 0 = uncapped). Under 1.0 the "
              "per-episode total stays below the smallest gap between two clembench "
              "outcomes (1.0), so the penalty can only rank episodes within an outcome "
              "class and never make staying silent beat winning. Above it, it can -- and "
              "0.5 already leaves no room for turn_rewards (swing 2 x budget) as well.")
    if not marshal_config.enabled:
        print("[length_penalty] WARNING: --no-marshal is set, so the MARSHAL advantage "
              "path is off and the length penalty has NO effect (it is applied inside "
              "compute_marshal_advantages).")
    print("[length_penalty] NOTE: MARSHAL skips this penalty for free-text envs like "
          "clembench (env_manager.py:470-480); this run is a deliberate divergence.")
    print("[length_penalty] watch marshal/length_penalty/episode_total_mean (what the "
          "outcome competes with) and .../budget_clip_rate, NOT the per-turn mean.")
    _report_legacy_length_penalty_fields(marshal_config)


def _report_legacy_length_penalty_fields(marshal_config) -> None:
    """Say out loud that the old threshold knobs are set but do nothing.

    Silence here is exactly the failure mode this project has been bitten by before:
    a cluster preset exports LP_MAX_LEN/LP_COEF, the manifest dutifully records them,
    and a reader concludes the run used a calibration it never applied.
    """
    legacy = marshal_config.legacy_length_penalty_values()
    if not legacy:
        return
    print("[length_penalty] WARNING: these config fields are DEPRECATED and IGNORED -- "
          + ", ".join(f"{k}={v}" for k, v in sorted(legacy.items())))
    print("[length_penalty]          they parameterized the old threshold-based penalty "
          "(0 below max_len, linear beyond). The penalty is now a flat per-token cost "
          "with no threshold; use --length-penalty-per-token / --length-penalty-budget.")


def _report_turn_rewards(marshal_config, env) -> None:
    """Print what the dense per-turn reward channel will actually do, and warn.

    Resolving the extractor here (rather than leaving it to the first rollout) turns
    three otherwise-silent misconfigurations into a message you see before the model
    loads: a game with no registered extractor, a component allowlist that matches
    nothing, and a budget large enough to out-rank the game outcome itself.
    """
    from playpen.marshal.turn_rewards import resolve_turn_reward_extractor

    if not marshal_config.turn_rewards:
        print("[turn_rewards] off (terminal reward only: SUCCESS +1 / FAILURE 0 / ABORTED -1)")
        return

    game = getattr(env, "resolved_game_name", None) or env.game_name
    extractor, spec = resolve_turn_reward_extractor(game, marshal_config)
    if extractor is None:
        print(f"[turn_rewards] WARNING: no extractor for game {game!r} under "
              f"source={marshal_config.turn_reward_source!r} -- turn rewards are INERT "
              f"for this run. Use --turn-reward-source auto for the generic "
              f"format-compliance fallback.")
        return

    active = spec.components or extractor.components
    print(f"[turn_rewards] ON game={game} extractor={type(extractor).__name__} "
          f"components={list(active)} scale={spec.scale} budget={spec.budget}")
    print(f"[turn_rewards] a maximally-shaped turn contributes {spec.scale:+.3f}; an "
          f"episode's total is capped at +-{spec.budget:.3f} against a +-1 outcome"
          + ("" if spec.budget else " (UNCAPPED)"))
    if not marshal_config.turn_reward_budget_is_safe:
        print("[turn_rewards] WARNING: budget >= 0.5 (or 0 = uncapped). Below 0.5 the "
              "worst-case shaping swing (2 x budget) stays under the smallest gap "
              "between two clembench outcomes (1.0), so shaping cannot make a loss "
              "out-score a win. Above it, it can.")
    if marshal_config.marshal_exact:
        print("[turn_rewards] NOTE: fidelity_mode='marshal_exact' subtracts a mean over "
              "all turn-boundary slots before the cumulative sum, which already biases "
              "by turn count; a denser reward field enlarges that term. Prefer "
              "--fidelity-mode paper_correct for runs that use turn rewards.")
    if not marshal_config.turn_level_rewards:
        print("[turn_rewards] NOTE: turn_level_rewards is off, so every turn's shaping "
              "is summed into ONE terminal scalar -- the signal still counts, but none "
              "of its per-turn credit survives.")
    if not marshal_config.enabled:
        print("[turn_rewards] NOTE: --no-marshal is set. TRL consumes one scalar per "
              "row, so the episode's shaping total is added to the terminal reward "
              "rather than attributed per turn. marshal/turn_rewards/terminal_mean "
              "still logs the unshaped outcome.")


def _resolve_resume_and_stop(args: argparse.Namespace, output_dir: str):
    """Work out what this job continues from and where it stops. ``(path, stop_at)``.

    Returns ``(None, None)`` when neither flag was passed, which is the whole of the
    pre-existing behaviour.

    Everything that can be decided without a GPU is decided here, so the three ways a
    chained segment can be wrong all fail (or exit) in the first seconds:

    * a resume spec that cannot be satisfied -> ``ResumeError``, and the job dies with
      a message naming the directory it searched;
    * ``auto``/``latest`` without ``--no-run-subdir`` -> ``SystemExit``. Those specs mean
      "continue THIS directory", and with the timestamp layer on there is a fresh
      directory per job, so the chain's checkpoints would scatter across N of them and
      ``exp_list_checkpoints`` (hence the whole eval + sharding path) would only ever
      see the last segment's;
    * a segment whose work is already done -> ``SystemExit(0)``. That happens whenever
      the previous segment overran its own boundary, and exiting cleanly lets the rest
      of the chain carry on rather than stalling on a job that has nothing to train.
    """
    from playpen.marshal import resume as resume_lib

    spec = (args.resume_from_checkpoint or "").strip()
    stop_at = args.stop_at_step

    if spec.lower() in ("auto", "latest") and args.run_subdir:
        raise SystemExit(
            f"--resume-from-checkpoint {spec} needs --no-run-subdir.\n"
            f"  Without it every job writes into a NEW timestamped subfolder of "
            f"{output_dir!r},\n"
            f"  so a chain's checkpoints would be split across one folder per segment.\n"
            f"  Pass --no-run-subdir (experiments/*/train.sh always does), or give an "
            f"explicit\n"
            f"  checkpoint path to deliberately fork a new run directory from it."
        )

    resume_path = None
    if spec:
        try:
            resume_path, explanation = resume_lib.resolve_resume(spec, output_dir)
        except resume_lib.ResumeError as exc:
            # A message, not a traceback. This one is read off a cluster log by
            # someone working out why a chain stopped, and the stack above it is
            # ours, not theirs -- it says nothing they can act on.
            raise SystemExit(str(exc)) from None
        print(f"[resume] {explanation}")

    if stop_at is not None:
        if stop_at <= 0:
            raise SystemExit(f"--stop-at-step must be positive, got {stop_at}.")
        if stop_at > args.max_steps:
            print(f"[segment] NOTE: --stop-at-step {stop_at} is past --max-steps "
                  f"{args.max_steps}; training ends at {args.max_steps} regardless.")
            stop_at = None
        elif stop_at == args.max_steps:
            # The default flow callback already stops (and saves) there. Adding ours
            # would be a second callback claiming credit for the same event in the log.
            stop_at = None

    done = resume_lib.resumed_global_step(resume_path) if resume_path else 0
    target = stop_at or args.max_steps
    if done >= target:
        print(f"[segment] nothing to do: the checkpoint is already at step {done}, "
              f"which is at or past this job's target of {target}. Exiting cleanly so "
              f"the rest of the chain continues.")
        raise SystemExit(0)

    if stop_at is not None:
        print(f"[segment] this job trains steps {done} -> {stop_at} of {args.max_steps}. "
              f"The LR schedule spans all {args.max_steps}, so it is identical to an "
              f"uninterrupted run.")
    return resume_path, stop_at


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
          f"unique_pooling={marshal_config.unique_value_pooling} "
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
    if not marshal_config.marshal_exact_unique_pooling:
        if marshal_config.marshal_exact:
            print("[marshal] NOTE: marshal_exact_unique_pooling is OFF -- this run keeps "
                  "marshal_exact's pre-sum reward normalization but pools advantages "
                  "occurrence-weighted (paper_correct's rule) instead of over distinct "
                  "trajectory returns. Do not report it as reproducing MARSHAL's shipped "
                  "normalization.")
        else:
            print("[marshal] NOTE: marshal_exact_unique_pooling is OFF but has no effect "
                  f"under fidelity_mode='{marshal_config.fidelity_mode}', which never pools "
                  "over distinct values anyway.")
    if marshal_config.row_context_mode != "exact":
        print("[marshal] WARNING: row_context_mode='spliced' reproduces the pre-fix "
              "rollout assembly, where the trained token sequence is not the sequence "
              "the policy generated under from turn 2 onward. TRL's vLLM "
              "importance-sampling correction then collapses the gradient. Use it only "
              "to reproduce an old run.")
    _report_length_penalty(marshal_config, args.max_completion_length)
    print(f"[kl] beta={args.kl_beta}" + (" (disabled)" if args.kl_beta == 0.0 else ""))
    _dr_overrides = marshal_config.trl_grpo_overrides()
    print(f"[dr_grpo] enabled={marshal_config.dr_grpo}"
          + (f" -> {_dr_overrides}" if _dr_overrides
             else " (off; TRL defaults: loss_type=dapo, scale_rewards=group)"))

    # 2. Self-play env (persistent across rollout calls) + dataset from packaged instances.
    env = SelfPlayEnv(args.game)
    print(f"[env] game={args.game} num_players={env.num_players}")

    # Dense per-turn rewards. Resolved here, before the model loads, so a bad
    # component name or an unregistered game fails in the first seconds rather than
    # after vLLM has come up.
    _report_turn_rewards(marshal_config, env)
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

    # 3a-pre. Resume, and this job's slice of the run.
    #
    # Resolved HERE -- before W&B opens and long before the model loads -- for two
    # reasons. A segment that has nothing left to do exits in seconds instead of
    # spending minutes bringing up vLLM to discover it, and a bad --resume-from-checkpoint
    # fails on a message rather than after the GPU has been claimed.
    # With neither flag passed this returns (None, None) and every line below behaves
    # exactly as it did before resuming existed.
    resume_path, stop_at = _resolve_resume_and_stop(args, output_dir)

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
    wandb_run = wandb_settings.start(
        config=wandb_config(args, marshal_config, output_dir, resume_path, stop_at)
    )
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
          f"plus a final one at step {stop_at or args.max_steps})")

    trainer = MarshalGRPOTrainer(
        model=args.model,
        rollout_func=rollout_func,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=grpo_config,
        peft_config=peft_config,
        marshal_config=marshal_config,
    )

    # Added AFTER construction, so it sits behind TRL's and HF's own callbacks in the
    # handler's list. Each callback may only add flags to the shared TrainerControl,
    # so ordering cannot make them fight -- when stop_at happens to land on a
    # save_steps boundary both ask to save and exactly one checkpoint is written.
    if stop_at is not None:
        from playpen.marshal.resume import stop_at_step_callback

        trainer.add_callback(stop_at_step_callback(stop_at))

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
    if resume_path is not None:
        # Some peft/transformers pairings cannot load a LoRA adapter back at all --
        # and only find out here, after vLLM has come up. Installs a shim when that is
        # the case and does nothing otherwise; see ensure_peft_resume_compat.
        from playpen.marshal.resume import ensure_peft_resume_compat

        _compat = ensure_peft_resume_compat()
        if _compat:
            print(f"[resume] NOTE: patched peft's tensor-parallel adapter-load path "
                  f"({_compat}).\n"
                  f"[resume]       This process is single-rank, where that path is a "
                  f"no-op; without the patch every LoRA resume raises ImportError. "
                  f"Upgrading peft/transformers to a compatible pair removes the need "
                  f"for it.")

    try:
        # resume_from_checkpoint=None is TRL/HF's own default, so an unchained run
        # takes exactly the path it always did.
        trainer.train(resume_from_checkpoint=resume_path)
    finally:
        wandb_utils.finish(wandb_run, wandb_settings)
    print(f"[done] LoRA adapters saved under {output_dir}/checkpoint-<step>/")
    if stop_at is not None:
        print(f"[done] segment ended at step {trainer.state.global_step} of "
              f"{args.max_steps}; the next job continues from "
              f"{output_dir}/checkpoint-{trainer.state.global_step}/")


if __name__ == "__main__":
    main()

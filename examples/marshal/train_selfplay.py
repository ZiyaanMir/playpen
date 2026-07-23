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
Custom rollouts require vLLM (``use_vllm=True``).

Example:
    python examples/marshal/train_selfplay.py \
        --model HuggingFaceTB/SmolLM2-135M-Instruct \
        --game taboo \
        --marshal-config examples/marshal/marshal_config.yaml \
        --num-generations 2 --per-device-batch-size 4 --max-steps 10
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from datetime import datetime

# Custom rollout_func + experimental openenv utils emit warnings; silence unless debugging.
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")


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
                   help="Keep >= 2*num_generations so both seats co-occur in a batch for seat pooling.")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--max-completion-length", type=int, default=2048)
    p.add_argument("--max-turns", type=int, default=100, help="Safety cap on turns per episode.")
    p.add_argument("--max-instances", type=int, default=None, help="Limit number of game instances used.")
    p.add_argument("--report-to", default="none", choices=["none", "tensorboard", "wandb"],
                   help="Metrics sink. Default 'none' (metrics still print to console). "
                        "'tensorboard' requires `pip install tensorboard`.")

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


def main() -> None:
    args = parse_args()

    import trl
    from peft import LoraConfig

    from playpen.marshal import (
        MarshalConfig,
        MarshalGRPOTrainer,
        SelfPlayEnv,
        build_reward_func,
        build_selfplay_dataset,
        build_selfplay_rollout_func,
    )

    # 1. MARSHAL switch (YAML is the source of truth; --marshal/--no-marshal overrides 'enabled').
    marshal_config = MarshalConfig.from_yaml(args.marshal_config)
    if args.marshal is not None:
        marshal_config.enabled = args.marshal
    # Dr. GRPO switch (YAML source of truth; --dr-grpo/--no-dr-grpo overrides). Reconcile
    # aligns advantage_norm_mode 'mean_std' -> 'mean' when dr_grpo is on (keeps per-seat
    # pooling; drops only the std divisor Dr. GRPO removes). No-op when dr_grpo is off.
    if args.dr_grpo is not None:
        marshal_config.dr_grpo = args.dr_grpo
    # Length penalty (YAML source of truth; CLI overrides any field that was passed).
    # Re-run __post_init__ via dataclasses.replace so the max_len > min_len check
    # applies to the merged values, not just the ones that came from the YAML.
    lp_overrides = {
        name: getattr(args, name)
        for name in (
            "length_penalty",
            "length_penalty_coef",
            "length_penalty_bonus",
            "length_penalty_min_len",
            "length_penalty_max_len",
            "length_penalty_offset",
        )
        if getattr(args, name) is not None
    }
    if lp_overrides:
        marshal_config = dataclasses.replace(marshal_config, **lp_overrides)
    for notice in marshal_config.reconcile_for_dr_grpo():
        print(f"[dr_grpo] {notice}")
    print(f"[marshal] enabled={marshal_config.enabled} "
          f"agent_specific={marshal_config.agent_specific_normalization} "
          f"turn_level={marshal_config.turn_level_rewards} "
          f"advantage_norm_mode={marshal_config.advantage_norm_mode} "
          f"fidelity={marshal_config.fidelity_mode} "
          f"whiten_rewards={marshal_config.whiten_rewards} "
          f"whiten_advantages={marshal_config.whiten_advantages}")
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
        args.game, num_players=env.num_players, max_instances=args.max_instances
    )
    print(f"[data] {len(dataset)} rows ({env.num_players} seats x game instances)")

    # 3. GRPO config + LoRA (LoRA is always on, independent of the MARSHAL switch).
    # Every run writes into its own timestamped subfolder, so a rerun can never overwrite
    # a previous run's checkpoints (HF writes checkpoints as <output_dir>/checkpoint-<step>,
    # which collide across runs otherwise). Mirrors MARSHAL's runs/<experiment>/<timestamp>/.
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = args.output_dir or f"models/marshal/{args.game}/{os.path.basename(args.model)}"
    # --no-run-subdir drops the timestamp layer, so checkpoints land directly in the
    # caller's directory. Safe only because the caller owns a fresh dir per run.
    output_dir = os.path.join(base_dir, run_id) if args.run_subdir else base_dir
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
        report_to=args.report_to,
        logging_dir=(f"{output_dir}/tb" if args.report_to == "tensorboard" else None),
        log_completions=True,
        # Dr. GRPO recipe (loss_type + scale_rewards) when enabled; empty dict = no
        # change, so with dr_grpo off this call is identical to before the flag existed.
        **marshal_config.trl_grpo_overrides(),
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

    trainer.train()
    print(f"[done] LoRA adapters saved under {output_dir}/checkpoint-<step>/")


if __name__ == "__main__":
    main()

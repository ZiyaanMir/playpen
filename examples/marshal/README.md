# MARSHAL-style self-play training in playpen

This example trains a single shared LoRA policy to play a **2-player clembench
game** (default: `taboo`) via **self-play**, using a from-scratch port of
[MARSHAL](https://arxiv.org/abs/2510.15414)'s RL algorithm on top of playpen's
TRL `GRPOTrainer` + PEFT stack.

MARSHAL contributes two ideas on top of a REINFORCE/GRPO objective:

1. **Turn-level advantage estimator** — rewards are attributed at each turn
   boundary and turned into per-token returns via a backward cumulative sum.
2. **Agent-specific advantage normalization** — advantages are pooled and
   normalized *per self-play seat* (player_0 rows separately from player_1 rows)
   rather than pooling both roles together.

Both are gated behind a single on/off switch you set **before** a run.

## Layout

Reusable library code lives in the installable package:

- [`playpen/marshal/config.py`](../../playpen/marshal/config.py) — `MarshalConfig`, the on/off switch.
- [`playpen/marshal/advantage.py`](../../playpen/marshal/advantage.py) — turn-level returns + per-seat normalization (pure torch, unit-tested).
- [`playpen/marshal/selfplay_env.py`](../../playpen/marshal/selfplay_env.py) — a two-seat wrapper over a clembench game (both seats marked "learner").
- [`playpen/marshal/selfplay_agent.py`](../../playpen/marshal/selfplay_agent.py) — per-seat rollout collection.
- [`playpen/marshal/trainer.py`](../../playpen/marshal/trainer.py) — `MarshalGRPOTrainer` + rollout/reward/dataset helpers.

This directory holds only the runnable wiring:

- `train_selfplay.py` — standalone launch script (run with `python`, **not** `playpen run`).
- `marshal_config.yaml` — the MARSHAL on/off + hyperparameter switch.

## Prerequisites

Install the training stack into playpen's environment (the repo uses `uv`):

```bash
VIRTUAL_ENV=.venv uv pip install "trl>=0.28.0,<1.0.0" vllm
# or, via the extra:
VIRTUAL_ENV=.venv uv pip install -e ".[marshal]"
```

Custom `rollout_func` generation requires vLLM (`use_vllm=True`).

## The on/off switch

`marshal_config.yaml` maps 1:1 onto `playpen.marshal.MarshalConfig`:

| field | meaning |
|---|---|
| `enabled` | master switch. `false` ⇒ plain TRL GRPO (scalar terminal reward, TRL's own normalization). |
| `agent_specific_normalization` | pool/normalize advantages per seat (MARSHAL's headline contribution). |
| `turn_level_rewards` | keep per-turn rewards (`true`) vs. sum to one terminal scalar (`false`). |
| `advantage_norm_mode` | `mean` (center) or `mean_std` (z-score). |
| `gamma` | backward-return discount (MARSHAL uses `1.0`). |
| `fidelity_mode` | `paper_correct` (default) vs. `marshal_exact` (bit-comparable reproduction of MARSHAL's *shipped* code, including two documented departures from its own paper). |
| `whiten_rewards` | z-score the token-level reward field batch-wide *before* the cumulative sum (mirrors ROLL's `whiten_rewards: true`, set in every shipped MARSHAL selfplay YAML; densifies sparse rewards with a length-dependent component). Default `false`. |
| `whiten_advantages` | z-score the final advantages batch-wide *after* per-seat normalization (mirrors ROLL's `whiten_advantages: true`; scale stabilization). Default `false`. |
| `dr_grpo` | run with the **Dr. GRPO** ([arXiv:2503.20783](https://arxiv.org/abs/2503.20783)) recipe: TRL `loss_type='dr_grpo'` (loss normalized by the constant `max_completion_length` ⇒ no length bias) + `scale_rewards='none'` (no std division ⇒ no difficulty bias). Default `false`. |

KL regularization is a CLI switch (not part of the advantage math): `--kl-beta 0.2`
adds TRL's built-in KL(policy ‖ base model) loss term (MARSHAL's shipped coefficient
is `0.20`; default `0.0` = off, no reference forward pass).

**Dr. GRPO** (`dr_grpo` in the YAML, or `--dr-grpo`/`--no-dr-grpo` to override) is a
second, independent switch that maps onto TRL's `GRPOConfig(loss_type='dr_grpo',
scale_rewards='none')` — the exact recipe TRL's docs prescribe for Dr. GRPO. Unlike the
advantage fields above it **takes effect regardless of `enabled`**, because it configures
the shared TRL loss/scaling used by both the MARSHAL path and the `--no-marshal` baseline,
and it is applied by the launch script — **`advantage.py` is not touched**. Off (default)
⇒ no kwargs are added to `GRPOConfig`, so the run is byte-identical to before this flag
existed.

It composes cleanly with MARSHAL:
- `scale_rewards` only affects the plain-GRPO (`--no-marshal`) path; on the MARSHAL path
  it is inert because `MarshalGRPOTrainer` overwrites TRL's scalar advantages with its own
  `(B,T)` tensor. So it cannot alter (or break) MARSHAL's advantages.
- MARSHAL already avoids std division by default (`advantage_norm_mode: mean`), so enabling
  Dr. GRPO keeps every MARSHAL feature (turn-level credit, per-seat pooling, gamma,
  whitening). Only if you had set `advantage_norm_mode: mean_std` does Dr. GRPO align it
  back to `mean` — dropping the per-seat std *divisor* but keeping per-seat *pooling* — and
  it prints a `[dr_grpo]` notice when it does.
- Full textbook Dr. GRPO baseline: `--no-marshal --dr-grpo` (both fixes fully active).

Caveat: `dr_grpo` normalizes the loss by `max_completion_length` (a constant, e.g. 2048)
rather than the realized active-token count, so per-step loss/grad magnitude is smaller
than the default `dapo`. This is inherent to Dr. GRPO's constant normalizer (paper
Listing 1), not a bug — bump the learning rate if convergence looks slow.

Flip it off for a run without editing the file:

```bash
python examples/marshal/train_selfplay.py --game taboo --no-marshal
```

## Run

```bash
# MARSHAL on (paper-correct), taboo, tiny smoke test
python examples/marshal/train_selfplay.py \
    --model HuggingFaceTB/SmolLM2-135M-Instruct \
    --game taboo \
    --num-generations 2 --per-device-batch-size 4 --max-steps 10

# MARSHAL off (plain GRPO baseline)
python examples/marshal/train_selfplay.py --game taboo --no-marshal --max-steps 10

# MARSHAL-exact reproduction (edit fidelity_mode: marshal_exact in the YAML)
python examples/marshal/train_selfplay.py --game taboo --max-steps 10

# MARSHAL + Dr. GRPO (unbiased loss/scaling; MARSHAL features preserved)
python examples/marshal/train_selfplay.py --game taboo --dr-grpo --max-steps 10

# Plain Dr. GRPO baseline (no MARSHAL)
python examples/marshal/train_selfplay.py --game taboo --no-marshal --dr-grpo --max-steps 10
```

Monitor: `tensorboard --logdir models/marshal/`. When MARSHAL is enabled the run
logs `marshal/seat_{0,1}/adv_mean` and `.../rows` so you can confirm the per-seat
split is live and the two seats' advantage distributions differ.

## Notes & caveats

- **Keep both seats in a generation batch** for seat pooling to be meaningful:
  `per_device_train_batch_size >= 2 * num_generations`. The dataset interleaves
  `{instance_idx}::seat0, {instance_idx}::seat1, ...` so adjacent prompts are the
  two seats of one instance. (Instances are addressed by their index into the
  packaged instance list, not by clembench `game_id`, which is only unique within
  an experiment.) Per-seat pooling operates over the local generation batch; for
  multi-process training it is a per-process approximation (the single-GPU target
  is exact).
- **LoRA + vLLM colocate** can hang (TRL issue
  [#3671](https://github.com/huggingface/trl/issues/3671)). If so, use
  `--vllm-mode server` with a separate `trl vllm-serve` process.
- **Sparse rewards**: clembench rewards are terminal-only by default
  (SUCCESS +1 / FAILURE 0 / ABORTED −1), so the turn-level cumulative sum
  degenerates to "broadcast the terminal reward" — correct, and it stays general
  if a game supplies a denser `reward_func`.
- **All-abort groups**: with an untrained tiny model every episode may abort,
  giving a degenerate (all-equal) pool and zero gradient. As MARSHAL recommends,
  SFT first or rely on the KL term. When MARSHAL is *disabled*, the reward
  function applies the wordle-example abort-shaping (−1 → small random negative)
  to keep variance.
- **`paper_correct` vs `marshal_exact`**: the port defaults to the algorithm the
  paper *describes*. `marshal_exact` reintroduces two behaviors from MARSHAL's
  shipped code that its own `PAPER_VS_CODE_DISCREPANCIES.md` flags as bugs (a
  biasing pre-sum reward normalization, and distinct-value pooling that
  equal-weights rare and common outcomes) — use it only to compare against
  MARSHAL's own numbers.

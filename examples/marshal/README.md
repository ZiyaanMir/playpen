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
- [`playpen/marshal/turn_rewards.py`](../../playpen/marshal/turn_rewards.py) — optional dense per-turn rewards read off the live game state (stdlib-only, unit-tested).
- [`playpen/marshal/trainer.py`](../../playpen/marshal/trainer.py) — `MarshalGRPOTrainer` + rollout/reward/dataset helpers.
- [`playpen/marshal/wandb_utils.py`](../../playpen/marshal/wandb_utils.py) — `WandbSettings`: W&B run setup, offline fallback (stdlib-only; does not import `wandb` until a run starts).

This directory holds only the runnable wiring:

- `train_selfplay.py` — standalone launch script (run with `python`, **not** `playpen run`).
- `marshal_config.yaml` — the MARSHAL on/off + hyperparameter switch.

## Prerequisites

Install the training stack into playpen's environment (the repo uses `uv`):

```bash
VIRTUAL_ENV=.venv uv pip install "trl>=0.28.0,<1.0.0" vllm
# or, via the extra:
VIRTUAL_ENV=.venv uv pip install -e ".[marshal]"

# optional: Weights & Biases logging (--wandb)
VIRTUAL_ENV=.venv uv pip install -e ".[wandb]"
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
| `marshal_exact_unique_pooling` | sub-flag for `marshal_exact`'s **second** departure only — the `torch.unique` distinct-value pooling. Default `true` (= `marshal_exact` unchanged); `false` keeps its pre-sum reward normalization but pools occurrence-weighted like `paper_correct`. **Inert under `paper_correct`**, which never uniques. |
| `whiten_rewards` | z-score the token-level reward field batch-wide *before* the cumulative sum (mirrors ROLL's `whiten_rewards: true`, set in every shipped MARSHAL selfplay YAML; densifies sparse rewards with a length-dependent component). Default `false`. |
| `whiten_advantages` | z-score the final advantages batch-wide *after* per-seat normalization (mirrors ROLL's `whiten_advantages: true`; scale stabilization). Default `false`. |
| `dr_grpo` | run with the **Dr. GRPO** ([arXiv:2503.20783](https://arxiv.org/abs/2503.20783)) recipe: TRL `loss_type='dr_grpo'` (loss normalized by the constant `max_completion_length` ⇒ no length bias) + `scale_rewards='none'` (no std division ⇒ no difficulty bias). Default `false`. |
| `grpo_loss` | run with TRL `loss_type='grpo'` — the original per-row aggregation (`mean_i(Σ_t ℓ / Σ_t m)`), which is what upstream MARSHAL/ROLL trains under. `scale_rewards` untouched. **Mutually exclusive with `dr_grpo`.** Default `false`. |
| `turn_rewards` | **dense per-turn rewards** read off the game's live state, on top of the terminal outcome. Default `false` (terminal-only, i.e. unchanged). See [below](#dense-per-turn-rewards). |

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
than the default `dapo` — by exactly `mean_completion_length / max_completion_length`,
measured at 0.016–1.09 across the 2026-08-10 runs (it *rises* where multi-turn rows
outrun the per-turn cap). This is inherent to Dr. GRPO's constant normalizer (paper
Listing 1), not a bug. AdamW absorbs most of a constant rescale, so this is rarely an LR
problem; what it does change is that `max_grad_norm=1.0` clipping stops engaging, and
that logged `loss`/`grad_norm` are no longer comparable with a `dapo` arm.

**Original GRPO** (`grpo_loss`, or `--grpo-loss`/`--no-grpo-loss`) is the third setting on
this same axis: TRL `loss_type='grpo'`, where each row's token losses are averaged over
*that row's own length* and the rows are then averaged. Nothing else moves — `scale_rewards`
keeps TRL's `group` default and `advantage.py` is untouched. It is **mutually exclusive**
with `dr_grpo` (both write `loss_type`); setting both raises at config-validation time.

It exists because that is what upstream MARSHAL actually trains under. MARSHAL is a ROLL
fork whose shipped playpen selfplay YAMLs leave `loss_agg_mode` unset, so they take ROLL's
`seq-mean-token-sum` default — whose implementation is `masked_mean(loss_mat, mask, dim=-1)`
then a mean over rows (`roll/utils/functionals.py`), i.e. a per-row **mean** despite the
`# token-sum` comment sitting above it. That is TRL's `grpo`, not TRL's default `dapo`.
Two things to know before using it. Rows are weighted equally regardless of length, so
short turns dominate the gradient (the length bias `dapo` removes). And TRL divides by the
**full** row count where ROLL divides by its **valid** row count — a placeholder row
(`_empty_row`: an idle seat, a drifted row) adds nothing to the numerator but still takes a
denominator slot, so the update is scaled by `1 - placeholder_rate` against upstream. That
is game-dependent and not small:

| game | mean `marshal/rows/placeholder_rate` | weaker than ROLL by |
|---|---|---|
| `wordle_withclue` | 0.53 | 2.1× |
| `imagegame` | 0.47 | 1.9× |
| `wordle_withcritic` | 0.36 | 1.6× |
| `guesswhat` | 0.31 | 1.5× |
| `wordle` | 0.12 | 1.13× |
| `codenames` / `taboo` | 0.065 / 0.04 | ~1.05× |
| `dond`, `adventuregame`, `matchit_ascii`, `referencegame`, `textmapworld*` | ≤0.01 | ~1× |

(means over the 2026-08 runs; the worst-hit games are the ones where `GameSpec` players
outnumber learner seats). This does not arise under `dapo`, where a placeholder
contributes no tokens and so never enters the normalizer. Read
`marshal/rows/placeholder_rate` for your own run before comparing gradient scales.

Flip it off for a run without editing the file:

```bash
python examples/marshal/train_selfplay.py --game taboo --no-marshal
```

## Dense per-turn rewards

Playpen is **terminal-only** out of the box: clemcore scores an episode `SUCCESS +1`
/ `FAILURE 0` / `ABORTED -1` and hands that one team scalar to every seat's last
turn. `turn_rewards: true` (or `--turn-rewards`) adds a second, bounded reward
channel read off the game's own `GameState` after every step — the route clemcore
itself suggests for game-specific rewards (see the `reward_func` docstring in
`clemcore/clemgame/envs/pettingzoo/master.py`). Implementation:
[`playpen/marshal/turn_rewards.py`](../../playpen/marshal/turn_rewards.py).

Off by default, and with it off a run is byte-identical to before the feature
existed — no extra rollout columns, no changed values.

### Why

Two problems it addresses, both structural:

1. **Turn-level credit has nothing to work with.** MARSHAL's turn-level estimator
   attributes reward at each turn boundary, but with a terminal-only reward every
   turn of a row carries the same return — so `turn_level_rewards: true` currently
   separates nothing.
2. **Per-seat credit has nothing to work with either.** clembench gives *both* seats
   the same team outcome, so `agent_specific_normalization` mean-centers two
   identical distributions and both seats end up with identical advantages. A
   per-turn reward is attributed to *the seat that acted* — it is the only mechanism
   in this pipeline that makes the two seats' advantages genuinely differ.

### What each game supplies

Each extractor mirrors that game's own scorer definitions, and normalizes every
component to `[-1, 1]`.

| game | components | signal |
|---|---|---|
| `wordle` | `closeness`, `format` | how much of the target the guess revealed, using clembench's own `turns_closeness` weights (green 5, yellow 3). Paid as the **gain over the best guess so far**, so it telescopes to ≤ 1 per episode, repeating a guess earns nothing and a worse guess is not punished. `format` charges a guess the game rejected. |
| `codenames` | `board`, `format` | words **our team** revealed this turn: `+1` team / `−0.5` innocent / `−1` opponent / `−2` assassin, over the board's team count. Read from `board.revealed["team"]`, not the drop in `board.hidden` — codenames simulates the opposing team between rounds, and diffing `hidden` charged us for *their* moves (measured: it cancelled to exactly 0.0 every turn). |
| `taboo` | `format`, `clue` | which seat's turn broke the response format, or gave a clue containing the target/a related word. |
| `guesswhat` | `format` | malformed or disallowed question/answer. |
| `dond` | `proposal`, `format` | committing a well-formed secret proposal; aborting on a parse/rule error. |
| *anything else* | `format` | generic compliance, from whichever validation flags that game's `GameState` carries. |

`taboo`/`guesswhat` have no progress signal to expose — compliance is genuinely all
there is. That is a real finding about those games, not a wiring failure; the
`marshal/turn_rewards/component/*` metrics tell the two apart.

### Scaling: why this cannot overwhelm the terminal reward

Per seat, per episode: components are summed and clipped to `[-1, 1]`, multiplied by
`turn_reward_scale`, and then — if the episode's total exceeds `turn_reward_budget`
— the whole episode's shaping vector is **rescaled proportionally** (relative credit
between turns survives; no sign flips).

The invariant is `|sum of one seat's shaping over one episode| ≤ budget`. Because the
MARSHAL return is a backward cumulative *sum*, that episode total is exactly what
competes with the `±1` outcome. With `budget < 0.5` the worst-case swing between two
episodes is `2 × budget < 1.0`, smaller than the gap between any two distinct
clembench outcomes (`SUCCESS−FAILURE = 1`, `FAILURE−ABORTED = 1`):

> **shaping can reorder episodes *within* an outcome class, never across one.**
> A shaped loss can never out-score a bare win.

The launch script and `experiments/lib/write_manifest.py` both check this and warn
when `turn_reward_budget ≥ 0.5` (or `0`, which disables the cap entirely).

The budget is a **safety net, not the operating point**: pick `turn_reward_scale` so
a typical episode lands under it, and watch
`marshal/turn_rewards/budget_clip_rate` — a rate near 1.0 means the cap binds every
episode, so the budget rather than the game is setting your signal's magnitude and
only its *shape* survives.

### Both paths

Like `row_context_mode`/`episode_pairing`/`dr_grpo`, this applies **regardless of
`enabled`** — it governs rollout collection, which the MARSHAL path and the
`--no-marshal` baseline share. On the MARSHAL path the shaping lands at each turn
boundary and gets full turn-level credit; on the plain-GRPO path (which consumes only
one scalar per row) the episode total is added to the terminal reward. Either way
`marshal/turn_rewards/terminal_mean` logs the **unshaped** outcome, so a turn-rewards
arm stays comparable with one that has the feature off.

Prefer `fidelity_mode: paper_correct` here. Under `marshal_exact` the pre-sum pass
subtracts a mean over all turn-boundary slots, which already biases by turn count; a
denser reward field makes that term larger.

### Knobs

| field | CLI | default | meaning |
|---|---|---|---|
| `turn_rewards` | `--turn-rewards` / `--no-turn-rewards` | `false` | master switch |
| `turn_reward_source` | `--turn-reward-source` | `auto` | `auto` = game extractor else generic; `game` = game-specific only (off, with a warning, for an unregistered game); `generic` = force compliance-only |
| `turn_reward_scale` | `--turn-reward-scale` | `0.05` | most a single turn can contribute |
| `turn_reward_budget` | `--turn-reward-budget` | `0.3` | cap on an episode's total; `0` disables (and forfeits the guarantee) |
| `turn_reward_components` | `--turn-reward-components` | `""` (all) | allowlist, e.g. `closeness`. An unknown name is an error at startup |

```bash
# wordle with its progress signal, default scaling
python examples/marshal/train_selfplay.py --game wordle --turn-rewards --max-steps 10

# progress only, no format penalty, and a larger per-turn signal
python examples/marshal/train_selfplay.py --game wordle --turn-rewards \
    --turn-reward-components closeness --turn-reward-scale 0.1

# compliance-only ablation arm, on any game
python examples/marshal/train_selfplay.py --game taboo --turn-rewards \
    --turn-reward-source generic

# force off for one run regardless of the YAML
python examples/marshal/train_selfplay.py --game wordle --no-turn-rewards
```

Metrics: `marshal/turn_rewards/{sum_mean,sum_abs_mean,sum_max_abs,nonzero_rate,budget_clip_rate,terminal_mean}`
and one `marshal/turn_rewards/component/<name>` per component.

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

# Longer run keeping an intermediate checkpoint every 20 steps
python examples/marshal/train_selfplay.py --game taboo --max-steps 100 --save-steps 20

# Logged to Weights & Biases (see below; records offline when there is no credential)
python examples/marshal/train_selfplay.py --game taboo --max-steps 100 \
    --wandb --wandb-project my-dissertation --wandb-tags pilot
```

Monitor: `tensorboard --logdir models/marshal/`, or Weights & Biases (below). When
MARSHAL is enabled the run logs `marshal/seat_{0,1}/adv_mean` and `.../rows` so you
can confirm the per-seat split is live and the two seats' advantage distributions
differ.

## Weights & Biases

```bash
uv pip install wandb        # once, into playpen's venv

python examples/marshal/train_selfplay.py --game taboo --max-steps 10 \
    --wandb --wandb-project my-dissertation
```

Everything TRL and the trainer already log goes to W&B — reward, KL, loss, the
`marshal/*` per-seat and length-penalty metrics, and the completions table
(`log_completions` is on). What the flags add is the context that makes a *set* of
runs comparable:

| flag | env fallback | default |
|---|---|---|
| `--wandb` / `--no-wandb` | — | off (also on via `--report-to wandb`, or by exporting `WANDB_PROJECT`) |
| `--wandb-project` | `WANDB_PROJECT` | `playpen-marshal` |
| `--wandb-entity` | `WANDB_ENTITY` | your account's default |
| `--wandb-run-name` | `WANDB_NAME` | `{game}_{model}_{timestamp}` — the same name as the run folder |
| `--wandb-group` | `WANDB_RUN_GROUP` | `{game}_{model}`, so an ablation's arms sit together |
| `--wandb-tags` | `WANDB_TAGS` | switch tags are added automatically |
| `--wandb-mode` | `WANDB_MODE` | `auto` |
| `--wandb-dir` | `WANDB_DIR` | this run's output dir |
| `--wandb-id` + `--wandb-resume` | `WANDB_RUN_ID` / `WANDB_RESUME` | — (use to continue one run after a requeue) |

The run's config records the **resolved** MARSHAL config (YAML merged with CLI
overrides), the LoRA/vLLM/GRPO settings and the output directory, and the run is
auto-tagged with the switches that define the arm (`marshal`/`no-marshal`,
`dr_grpo`, `length_penalty`, `paper_correct`/`marshal_exact`). So "which of these
runs had per-seat normalization off" is a filter, not an archaeology exercise.

**`--wandb-mode auto` (the default) goes online only when a credential exists *and*
the W&B API answers a connection; otherwise it records offline.** Both halves matter
on a cluster. `$HOME` is shared between login and compute nodes, so `wandb login`
leaves a `~/.netrc` a compute node can read perfectly well but cannot use — deciding
"online" on the strength of that file is how a job ends up blocked inside
`wandb.init`. The probe costs a few seconds once per run. If an online init fails
anyway (the network went away between the probe and the call), it retries offline
rather than losing the training run. Offline always works and costs one command
afterwards:

```bash
wandb sync <output-dir>/wandb/offline-run-*        # printed at the end of the run
experiments/lib/wandb_sync.sh $MARSHAL_RUNS        # or, for whole experiment dirs
```

The run is opened before the model loads, so a missing package or bad credential
fails in the first seconds rather than after vLLM has spun up. `wandb_run.json` is
written into the output directory (id, url, or the exact `wandb sync` command) so a
checkpoint folder always says which W&B run it belongs to.

Cluster runs launched through `experiments/*/run_experiment.sh` get all of this
wired up already (`WB_*` variables, run named after the experiment, offline data
inside the experiment directory) — see [`experiments/README.md`](../../experiments/README.md).
The older hardcoded `slurm/` and `slurm_eddie/` job scripts need no edit either:
exporting `WANDB_PROJECT` before submitting turns W&B on through the env fallbacks.
Enabled that way it degrades to a warning if `wandb` is not installed, rather than
failing the job — only an explicit `--wandb` is treated as a hard requirement.

## Output & checkpoints

Every run writes into its own timestamped folder, so reruns never overwrite each
other. HF `Trainer` puts each checkpoint in a `checkpoint-<step>` subfolder:

```
models/marshal/{game}/{model}/20260721-142233/
├── checkpoint-20/ ... checkpoint-100/   # --save-steps N, plus always a final one
│   ├── adapter_config.json + adapter_model.safetensors   # LoRA only, no base weights
│   └── optimizer.pt, scheduler.pt, trainer_state.json, tokenizer files
├── completions/completions_*.parquet
├── wandb_run.json                        # only with --wandb: id/url/sync command
├── wandb/                                # only with --wandb: run data (offline runs
│                                         #   stay here until `wandb sync`)
└── tb/                                   # only with --report-to tensorboard
```

- `--save-steps N` sets the cadence (default `500`, TRL/HF's own default). The final
  step always saves regardless.
- `--output-dir` sets only the *base*; the timestamped run folder is appended to it.
- The adapter lives **only** inside `checkpoint-<step>/` — the script does not call
  `trainer.save_model()`, so load a checkpoint dir directly with PEFT.
- Every checkpoint is kept (no `save_total_limit`), and each carries optimizer state
  (~2x the adapter size), so a small `--save-steps` on a long run costs disk.

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
  MARSHAL's own numbers. The two departures are separately switchable:
  `marshal_exact_unique_pooling: false` (CLI `--no-marshal-exact-unique-pooling`)
  keeps the first and drops the second, which is what isolates them in an
  ablation. A run with it off is not a reproduction of MARSHAL's shipped
  normalization and should not be reported as one.

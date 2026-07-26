# Experiments: one command, one directory, train + evaluate

Submits a MARSHAL self-play training run **and** its lm-eval evaluation as a single
chained pair of jobs, and puts everything that run produced — parameters, checkpoints,
scores, logs — inside **one self-contained directory**.

```bash
# Eddie
experiments/eddie/run_experiment.sh dond

# Isambard
module load brics/userenv
experiments/isambard/run_experiment.sh dond
```

That is the whole interface. It queues training, queues evaluation held behind it, and
prints where everything will land.

Replaces the older split workflow (`slurm/`, `slurm_eddie/` for training; a separate
`lm_eval_jobs/` tree writing into a shared `Results/`), where a checkpoint and its score
lived in different places and neither recorded what produced it. Those scripts still work
and are untouched.

---

## What you get

```
$MARSHAL_RUNS/dond_Qwen3-4B_lp384_20260723-142530/
├── manifest.txt          ← what this run IS. cat this first.
├── manifest.json           the same, machine-readable
├── marshal_config.yaml     frozen copy of the config this run used
├── experiment.env          every setting, re-sourceable to re-run by hand
├── RESULTS.md            ← the score table, with deltas vs the untrained base
├── results.tsv             the same numbers for pandas
├── logs/
│   ├── train.<jobid>.out/.err
│   └── eval.<jobid>.out/.err
├── train/<timestamp>/
│   ├── checkpoint-50/ … checkpoint-200/
│   ├── completions/*.parquet
│   └── wandb_run.json      which W&B run this is (id, url, or offline sync command)
├── wandb/                  W&B run data; offline runs live here until synced
└── eval/
    ├── base/               lm-eval on the UNTRAINED model — the baseline
    ├── checkpoint-50/      full lm-eval output incl. --log_samples
    └── …
```

The directory name alone tells you the game, the model, the tag and when it ran. Nothing
is written outside it, so an experiment is one thing to inspect, `rsync` home, or delete.

### Across experiments

```bash
experiments/status.sh -v
```

```
experiment                                   state                  score
-------------------------------------------  ---------------------  -----
dond_Qwen3-4B_20260723-152753                3 ckpt, scored         acc=0.4460 (+0.0340 vs base)
                                               model=Qwen3-4B game=dond steps=200 len_penalty=-0.51/ep fidelity=marshal_exact
guesswhat_Qwen3-0.6B_lp_off_20260723-152920  1 ckpt, not evaluated
                                               model=Qwen3-0.6B game=guesswhat steps=200 len_penalty=-0.42/ep fidelity=paper_correct
```

Works on a login node or on your laptop after an `rsync` — it only reads files.

---

## Running variants

Everything is overridden by exporting a variable before the call. **Always set `EXP_TAG`
when a run differs from the preset** — it goes into the directory name and is what makes
the run legible weeks later.

```bash
# ablation arm: length penalty off
EXP_TAG=lp_off EXTRA_TRAIN_ARGS=--no-length-penalty \
  experiments/eddie/run_experiment.sh dond

# plain-GRPO baseline
EXP_TAG=baseline EXTRA_TRAIN_ARGS=--no-marshal \
  experiments/eddie/run_experiment.sh dond

# quick smoke test (small model, few steps, 5 eval examples)
MODEL=Qwen/Qwen3-0.6B MAX_STEPS=20 SAVE_STEPS=10 EVAL_LIMIT=5 EXP_TAG=smoke \
  experiments/eddie/run_experiment.sh guesswhat

# different eval tasks / bigger eval batch
EVAL_TASKS=logiglue,logicbench EVAL_BATCH=32 \
  experiments/isambard/run_experiment.sh dond
```

`EXTRA_TRAIN_ARGS` on/off flags are parsed into the manifest, so an arm launched with
`--no-length-penalty` **records `length_penalty: false`** rather than parroting the YAML.

Scheduler options pass through too:
`TRAIN_QSUB_OPTS='-l a100=true'` (Eddie), `TRAIN_SBATCH_OPTS='--time=08:00:00'` (Isambard).

| variable | default | what it does |
|---|---|---|
| `MODEL` | `Qwen/Qwen3-4B` | policy to train |
| `EXP_TAG` | — | goes in the directory name; set it for every variant |
| `EXTRA_TRAIN_ARGS` | — | extra `train_selfplay.py` flags, verbatim |
| `MAX_STEPS` / `SAVE_STEPS` | `200` / `50` | schedule and checkpoint cadence |
| `EVAL_TASKS` | `logiglue,logicbench` | lm-eval tasks (comma-separated) |
| `EVAL_BASE` | `1` | also score the untrained model |
| `EVAL_LIMIT` | — | `--limit N`, smoke tests only |
| `EVAL_EXTRA` | — | extra lm-eval flags, applied to **every** row |
| `WB_PROJECT` | `playpen-marshal` | Weights & Biases project |
| `WB_ENABLE` | `1` | `0` turns W&B off for this run |
| `WB_MODE` | `auto` | `auto` \| `online` \| `offline` \| `disabled` |
| `WB_ENTITY` / `WB_GROUP` / `WB_TAGS` | — | W&B team, run group, extra tags |

Everything else lives in `presets/<game>.env` — that file is the per-game source of truth.

---

## Weights & Biases

On by default. The run is **named after the experiment** (`EXP_ID`), grouped by
`{game}_{model}`, and tagged with the switches that define the arm (`marshal` /
`no-marshal`, `dr_grpo`, `length_penalty`, `paper_correct` / `marshal_exact`, the cluster,
your `EXP_TAG`). Its config carries the *resolved* MARSHAL config plus every training
hyperparameter, so two arms can be diffed in the UI without opening a manifest.

`WB_MODE=auto` is what makes this safe to leave on: it uploads live only when a credential
exists (`WANDB_API_KEY`, or `~/.netrc` after `wandb login`) **and** the W&B API answers,
and **records offline otherwise**. Both checks are needed — `$HOME` is shared with the
login nodes, so a compute node can read a credential it has no network to use. A node with
no outbound network is therefore not a failure: the run is written to `$EXP_DIR/wandb/` and
travels with the experiment directory. A failed online init falls back to offline rather
than taking the training job down.

```bash
# upload offline runs later, from a login node or your laptop after an rsync
experiments/lib/wandb_sync.sh $MARSHAL_RUNS/dond_Qwen3-4B_20260726-120000
experiments/lib/wandb_sync.sh                # everything under $MARSHAL_RUNS
DRY_RUN=1 experiments/lib/wandb_sync.sh      # list what would be uploaded

# per-run overrides
WB_PROJECT=diss-ablations WB_TAGS=pilot experiments/eddie/run_experiment.sh dond
WB_ENABLE=0 experiments/eddie/run_experiment.sh dond          # off for this run
```

`wandb` must be installed in the training venv (`uv pip install wandb`); without it a run
launched this way fails at startup rather than training for hours with nothing recorded.
Which W&B run a directory belongs to is written to `train/wandb_run.json` **before**
training starts, so a job killed at the walltime still leaves the pointer behind.

The `WB_*` names are deliberately not `WANDB_*`: the latter are read by the wandb SDK
itself, and `WANDB_MODE=auto` is not a value it accepts. Setting the real `WANDB_*`
variables still works — `train_selfplay.py` falls back to them.

---

## The presets, and the length-penalty calibration

`presets/{dond,guesswhat,taboo}.env` hold the memory sizing carried over from the existing
job scripts, plus **recalibrated length-penalty values**.

The old setting (`max_len 500`, `coef 0.1`) was numerically inert on guesswhat: generation
is hard-capped **per turn** at `max_completion_length`, so with a 512-token cap the largest
reachable penalty was `-0.0025` per turn — against game outcomes of ±1. The advice to "set
`max_len` near `max_completion_length`" guarantees this, because MARSHAL's design point is
"≈ `-coef` at 2× `max_len`", which needs the cap to be **at least twice** `max_len`.

The rule the presets use:

```
LP_MAX_LEN = max_completion_length / 2
LP_COEF    = target_episode_total / turns_per_seat     (target ≈ 0.4–0.5)
```

The penalty is charged **per turn** and summed by the backward cumulative return, so what
competes with the ±1 outcome is `turns × per-turn`, not the per-turn number.

| game | cap | turns/seat | `LP_MAX_LEN` | `LP_COEF` | per turn | **per episode** |
|---|---|---|---|---|---|---|
| dond | 768 | 5 | 384 | 0.10 | −0.103 | **−0.51** |
| guesswhat | 512 | 8 | 256 | 0.05 | −0.052 | **−0.42** |
| taboo | 256 | 3 | 128 | 0.15 | −0.077 | **−0.23** |

Target −0.4 to −0.5: enough to matter, structurally unable to exceed 1.0, so "stay silent"
can never beat "win the game."

`manifest.txt` computes these numbers for the run and **warns in both directions** — if the
total is under ~0.05 (inert) or over 1.0 (silence beats winning). At the time of writing,
`examples/marshal/marshal_config.yaml` has `coef 0.4 / max_len 400`, which on dond is
−1.89 per episode and trips the too-strong warning; the presets override it.

**Watch `marshal/length_penalty/mean` in the training log, not `over_rate`.** With
`max_len` now at half the cap, `over_rate` is nonzero even when the penalty is trivial —
mean is the honest signal.

> **Reporting caveat.** `length_penalty: true` and `fidelity_mode: marshal_exact` interact:
> the penalty makes every trajectory return distinct, which collapses `marshal_exact`'s
> distinct-value pooling into ordinary occurrence-weighted pooling. Both are defensible
> algorithms and the gradient direction is unchanged, but an on/off length-penalty ablation
> at `marshal_exact` also silently switches pooling rule — pin `fidelity_mode` consistently
> across arms, and don't describe such a run as reproducing MARSHAL's shipped normalization.

---

## Prerequisites

Training uses the repo's own `.venv`, plus `wandb` in it for the metrics logging described
above (`uv pip install wandb`, or `WB_ENABLE=0` to skip). **Evaluation uses a separate
lm-eval environment** that must already exist — these scripts do not build it:

* **Eddie** — conda env `lmeval` with `peft` installed and `logiglue`/`logicbench` baked
  into its `lm_eval/tasks/`. Override the name with `LMEVAL_CONDA_ENV`.
* **Isambard** — a pre-existing venv at `$PROJECTDIR/$USER/evaluation/eval`
  (activated by `source`), with `logiglue`/`logicbench` already registered so no
  `--include_path` is needed. Override the path with `VENV_LMEVAL`.

See `notes/LMEVAL_CHECKPOINTS_EDDIE.md` / `..._ISAMBARD.md` for building them, and
pre-cache the base model on a login node (compute nodes may be offline).

---

## Design notes

**Why a separate eval job rather than eval at the end of training.** Different environment
(lm-eval + peft, not trl + vllm), different walltime, different GPU needs. Chaining also
means a failed eval can be re-run without redoing 20 h of training.

**The dependency is `afterany`, not `afterok`.** A 20 h run that hits the walltime exits
non-zero but has still written `checkpoint-150`; `afterok` would silently skip evaluating
it. `eval.sh` instead checks for real checkpoints and exits cleanly with a message when
there genuinely are none. Grid Engine's `-hold_jid` has the same semantics, so both
clusters behave identically.

**Why the run directory is discovered, not chosen.** `train_selfplay.py` always appends its
own timestamp to `--output-dir`, which the submitting shell cannot predict. Since `EXP_DIR`
is fresh per experiment there is exactly one subdirectory under `train/`, so discovery is
unambiguous — and no change to the training script was needed.

**The base-model eval is cached, not repeated.** A checkpoint's score is only meaningful
against the untrained model, so every experiment includes a `base` row — but that row
depends solely on `(model, tasks, limit, extra flags)`, never on the checkpoint, so it is
identical across every experiment on the same model. `exp_eval_base` runs it once, stores
it under `$MARSHAL_RUNS/_base_eval_cache/<model>__<tasks>__<hash>/`, and **copies** it into
later experiments instead of re-running on the GPU. The copy (not a symlink) keeps each
experiment self-contained for `rsync`. The cache key includes `--limit` and any
`EVAL_EXTRA`, so a `--limit 5` smoke run never satisfies a full run. A failed base eval
leaves no entry (results are staged and published only on success), so the next run
retries rather than reading junk. Override the location with `BASE_EVAL_CACHE=<dir>`, or
force a fresh run by deleting the relevant cache directory. The cache directory has no
`manifest.json`, so `status.sh` ignores it.

**Half-written checkpoints are skipped.** A job killed at the walltime can leave a
`checkpoint-N/` with no `adapter_config.json`; handing that to lm-eval fails confusingly.
Only directories holding a real adapter are evaluated.

**The base model is read from `adapter_config.json`, not assumed.** A mismatched base
applies the LoRA maths to the wrong weights and yields silently wrong scores with no error.
If it disagrees with the experiment's `MODEL`, the adapter wins and the log says so.

**`--apply_chat_template` is deliberately off**, matching the base-model baseline. Applying
it to one side of a comparison makes the comparison meaningless. Put it in `EVAL_EXTRA` if
you want it — that applies it to every row.

**Eval results go in `eval/<checkpoint>/`, not inside the checkpoint directories.** A
checkpoint dir is exactly a PEFT adapter; keeping it clean means `rsync`, `merge_adapter.py`
and `peft=` all keep working. Everything still lives under the one experiment directory.

**The manifest is written at submit time**, so a job that dies in the queue still leaves a
record of what it was meant to be. The training job appends its host, GPU and job id when
it starts.

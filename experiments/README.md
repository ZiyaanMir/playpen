# Experiments: one command, one directory, train + evaluate

Submits a MARSHAL self-play training run **and** both of its evaluations as a single
chained set of jobs, and puts everything that run produced — parameters, checkpoints,
scores, logs — inside **one self-contained directory**.

```bash
# Eddie
experiments/eddie/run_experiment.sh dond

# Isambard
module load brics/userenv
experiments/isambard/run_experiment.sh dond
```

That is the whole interface. It queues the training job, both evaluations behind it,
and a final job that writes the score tables, then prints where everything will land:

| # | job | what it answers |
|---|---|---|
| 1 | **train** (one job, or `TRAIN_SEGMENTS` chained ones) | — |
| 2 | **eval** (lm-eval: logiglue, logicbench) | did self-play change the model's *reasoning*? |
| 3 | **playpen eval** (clembench gameplay, clemscore) | did it get better at *playing*, including the 13 games it never trained on? |
| 4 | **summary** | writes the complete `RESULTS.md` / `PLAYPEN_RESULTS.md` |

Job 3 is the one that produces the number the shared task is scored on. Turn it off
for a run with `PPEVAL_ENABLE=0`.

A run too long for one walltime is not a dead end: `TRAIN_SEGMENTS=3` splits job 1
into three chained jobs, each continuing the last one's checkpoint, and
`resume_experiment.sh` restarts a run that already died. See
[Long runs: segments and resuming](#long-runs-segments-and-resuming).

**Jobs 2 and 3 are each split into shards of five checkpoints, and every shard runs
at the same time.** With the presets' `MAX_STEPS=1000` / `SAVE_STEPS=100` that is ten
checkpoints, so a run is 1 train + 2 lm-eval + 2 gameplay + 1 summary = six jobs. All
four evaluation jobs are held on the *training* job, not on each other, so they start
together the moment training finishes. See [Sharded evaluation](#sharded-evaluation).

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
├── experiment.resume.env   only if resume_experiment.sh ran: the same, plus what
│                           that resume changed. The original is left as the record
├── RESULTS.md            ← lm-eval scores, with deltas vs the untrained base
├── results.tsv             the same numbers for pandas
├── PLAYPEN_RESULTS.md    ← clemscore + per-game table, same deltas
├── playpen_results.tsv     the same numbers for pandas
├── logs/
│   ├── train1.<jobid>.out/.err    one per training segment
│   ├── eval1.<jobid>.out/.err     one per shard
│   ├── eval2.<jobid>.out/.err
│   ├── ppeval1.<jobid>.out/.err
│   ├── ppeval2.<jobid>.out/.err
│   └── summary.<jobid>.out/.err
├── train/<timestamp>/
│   ├── checkpoint-50/ … checkpoint-200/
│   ├── completions/*.parquet
│   └── wandb_run.json      which W&B run this is (id, url, or offline sync command)
├── wandb/                  W&B run data; offline runs live here until synced
├── eval/
│   ├── base/               lm-eval on the UNTRAINED model — the baseline
│   ├── checkpoint-50/      full lm-eval output incl. --log_samples
│   └── …
└── playpen-eval/
    ├── base/               the untrained model, played
    │   ├── <name>.val.json     clemscore / statscore
    │   └── clem/results.csv    per-game % Played and Quality Score
    │       └── <name>/<game>/  interactions, transcripts, per-episode scores
    ├── checkpoint-50/      … one per adapter
    └── .work/              the generated registries each row was run with
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

### The playpen games evaluation (job 3)

Every checkpoint plays the clembench games on the `playpen-data` **validation** split
— the same instances, the same scoring, and the same `clemscore` as `LEADERBOARD.md`
— and is reported against the untrained model:

```
| checkpoint       | clemscore      | avg % played  | avg quality  |
| `base`           | 4.00           | 33.3          | 12.0         |
| `checkpoint-100` | 24.00 (+20.00) | 66.7 (+33.4)  | 36.0 (+24.0) |

| checkpoint       | guesswhat  | taboo      | wordle |
| `base`           | 100 / 20.0 | 0 / -      | 0 / -  |
| `checkpoint-100` | 100 / 60.0 | 100 / 30.0 | 0 / -  |
```

The per-game table is the point. `% played` is how often the model produced a
parseable, rule-legal game and `quality score` is how well it did in those, so
**0 % played is not a score of zero** — it is a formatting failure, a different
problem with a different fix, and the two are indistinguishable in a clemscore
column. (For the same reason a blank clemscore means every episode aborted, not
that the model scored nothing; `summarize_playpen_eval.py` recomputes it from
`% played × quality` whenever clemeval left it out.)

Only the training game is in-distribution here. The other thirteen are the actual
question — whether self-play on one game taught it to *play*, or only to play
`guesswhat`.

| variable | default | what it does |
|---|---|---|
| `PPEVAL_ENABLE` | `1` | `0` doesn't queue the job at all |
| `PPEVAL_SUITE` | `clem` | `clem` (the 14 interactive games) \| `static` \| `all` |
| `PPEVAL_GAMES` | — | explicit list, overrides the suite: `dond,guesswhat` |
| `PPEVAL_CKPTS` | all | `last`, or `100,200` |
| `PPEVAL_BASE` | `1` | also play the untrained model (cached across experiments) |
| `PPEVAL_MAX_TOKENS` / `PPEVAL_TEMPERATURE` | `300` / `0.0` | generation budget — playpen's own defaults, i.e. what the leaderboard used |
| `PPEVAL_TIMEOUT` | — | per-checkpoint ceiling, e.g. `3h` |
| `PPEVAL_HF_OFFLINE` | `0` | `1` for a guaranteed network-free run, once the data is cached |

> **`dond` is not in the `clem` suite.** It carries `benchmark: 3.0`, not `2.0`, so a
> dond experiment is otherwise scored only on games it never trained on. Add it
> explicitly: `PPEVAL_GAMES=dond,guesswhat,taboo,codenames,wordle`.

This is generation, not loglikelihood scoring: ~70 validation instances across 14
games, each a multi-turn dialogue, **per checkpoint**. Budget hours per row for a 4B
model, and reach for `PPEVAL_CKPTS=last` rather than a longer walltime when that is
too much.

```bash
# only the last checkpoint, and add the game this run actually trained on
PPEVAL_CKPTS=last PPEVAL_GAMES=dond,guesswhat,taboo \
  experiments/eddie/run_experiment.sh dond
```

**Prerequisites:** the `colab-potsdam/playpen-data` validation split must be fetchable
or already in `$HF_HOME` (the job checks before loading any model and stops with the
one-line command to fetch it), and **`scikit-learn` must be in the venv** —
`clembench/privateshared` imports it. Nothing else: unlike lm-eval, this runs in the
repo's own `.venv` (clemcore + playpen + peft are all there), so there is no second
environment to build and the job is the same on both clusters.

### Long runs: segments and resuming

Training is one job by default. `TRAIN_SEGMENTS=N` makes it **N chained jobs**, each
resuming the previous one's last checkpoint:

```bash
# 1000 steps as three jobs of <= 400, in one experiment directory
TRAIN_SEGMENTS=3 experiments/eddie/run_experiment.sh guesswhat

# ...and ask for a walltime those shorter jobs actually need
TRAIN_SEGMENTS=3 TRAIN_SBATCH_OPTS='--time=08:00:00' \
  experiments/isambard/run_experiment.sh guesswhat
```

```
train seg 1 (0→400) ─▶ train seg 2 (400→800) ─▶ train seg 3 (800→1000) ─┬─▶ eval shards ─▶ summary
                                                                        └─▶ gameplay shards
```

**Why.** Eddie's gpu queue caps `h_rt` at 48 h and Isambard's `workq_qos` caps
`--time` at 24 h. Neither can be raised, so before this a run that outgrew its
walltime could only be redone. Segments also schedule sooner (a 8 h request backfills
where a 24 h one waits) and, on Isambard, book credit against 3 × 8 h instead of
3 × 24 h, because Slurm reserves against the request rather than the usage.

**`MAX_STEPS` is always the total, in every segment.** Segments partition it; they
never extend it. This is not a convention — it is what keeps a segmented arm
comparable with an unsegmented one:

> HF builds the LR scheduler from `max_steps`, and `scheduler.pt` restores only the
> step counter, never the decay curve. Hand segment *k* `max_steps = k*S` and every
> resume re-derives a **steeper** slope: step 3 of a 10-step run comes out at
> `2.5e-06` instead of `7e-06`, and the rate jumps back up at each boundary. So
> `--max-steps` stays the total and `--stop-at-step` ends the segment instead. Step
> 700 then has the same learning rate whether the run was one job or five —
> verified, and pinned by `tests/test_marshal_resume.py`.

Boundaries land on `SAVE_STEPS` multiples wherever possible, so a segment ends on a
checkpoint the schedule was going to write anyway and the run's checkpoint set is
exactly what an unsegmented run would produce. Ask for more segments than there are
checkpoints and you get the checkpoint count — the `[plan]` line reports what will
actually be submitted, which is not always the number you asked for.

| variable | default | what it does |
|---|---|---|
| `TRAIN_SEGMENTS` | `1` | how many chained training jobs. `1` = the single job this always submitted |
| `SEGMENT_STEPS` | — | steps per segment, when you want to size against a walltime directly. `TRAIN_SEGMENTS` is then derived from it |
| `RESUME_FROM` | — | `auto` \| `latest` \| a checkpoint path. Applies to the **first** segment only |

Set either knob, not both — whichever you set governs and the other is derived. (Set
both and the explicit `SEGMENT_STEPS` wins.)

#### Resuming a run that died

```bash
experiments/eddie/resume_experiment.sh    $MARSHAL_RUNS/<experiment>
experiments/isambard/resume_experiment.sh $MARSHAL_RUNS/<experiment>
```

Re-queues only the training segments still ahead of what is on disk, then the full
evaluation chain behind them — **in the same directory, under the same settings, into
the same W&B run**. Do not use `run_experiment.sh` for this: that makes a new
experiment directory and starts again at step 0.

```
$ experiments/eddie/resume_experiment.sh $MARSHAL_RUNS/guesswhat_Qwen3-4B_20260727-120621
[resume] plan       : 1000 steps total, 3 segment(s) of <= 400
[resume] from       : checkpoint-500 (global step 500)
[resume] segments   : 2..3 of 3 still to run
```

The remaining work can be re-planned on the way — usually the reason you are here,
since the run died *because* one job could not fit it:

```bash
TRAIN_SEGMENTS=5 TRAIN_QSUB_OPTS='-l h_rt=12:00:00' \
  experiments/eddie/resume_experiment.sh $MARSHAL_RUNS/<experiment>
```

`MAX_STEPS` is the one thing it refuses to change, for the reason in the quote above:
a different horizon would give the rest of the run a different LR schedule from its
first half, inside one continuous set of checkpoints that says nothing about it.
Extending a run is a new experiment.

#### It checks that the config really did survive

Resuming re-derives the config from the environment, and two things make that
environment different weeks later. Both are handled, and then verified:

* **The shared YAML gets edited.** `experiment.env` stores `MARSHAL_CONFIG` as a
  *path*, not as content — and `examples/marshal/marshal_config.yaml` is the file every
  ablation is varied from. Resuming through that path hands the remaining segments
  whatever it says *today*. So the resume points at the **frozen per-run copy** that
  `write_manifest.py` saved inside the experiment directory instead.
* **A setting arrived by a route the resume doesn't have.** Isambard's
  `sbatch --export=ALL` carries your whole shell environment, so `TR_ENABLE=1` reached
  the original job without ever being written to `experiment.env`. Nothing carries it
  on a resume. No care in the resume script can fix that — the value simply isn't
  recorded anywhere the resume can read.

So before submitting anything, every resolved MARSHAL setting is compared against what
`manifest.json` recorded at first submission, and a mismatch **stops the resume**:

```
[config-check] REFUSING TO RESUME -- the algorithm config has changed.

  field                            was (manifest.json)      would be now
  -------------------------------- ------------------------ ------------
  turn_rewards                     True                     False
```

Put the missing value back for that submission and it proceeds:

```bash
TR_ENABLE=1 experiments/isambard/resume_experiment.sh <EXP_DIR>
```

`RESUME_FORCE=1` skips the check if the change is deliberate and you accept a run
trained two ways. Fields that simply didn't *exist* when the run was submitted
(`MarshalConfig` gains fields over time) are reported as a note, not a refusal — there
is no old value for them to have drifted from.

This is worth knowing about even if you never resume: it is the only thing in the
pipeline that will tell you a stored experiment can no longer be reproduced from what
was recorded about it.

#### What is going on underneath

* **`--resume-from-checkpoint`** restores the LoRA adapter, optimizer, scheduler, RNG
  and step counter. `auto` = resume if there is something, else start at step 0;
  `latest` = the same but an *error* when there is nothing. Segment 1 gets `auto`,
  segments 2+ get `latest` — because by then an empty `train/` means the previous job
  died before its first save, and quietly restarting would cost a whole walltime.
* **A half-written checkpoint is never resumed from.** `trainer_state.json` is the
  last file HF writes, so its absence marks a job killed mid-save. Note this is
  *stricter* than the eval side: lm-eval can score a checkpoint whose optimizer state
  was never written, resume cannot.
* **Dependencies are `afterany` / plain `-hold_jid`**, not a success condition. A
  segment killed at the walltime exits non-zero having written `checkpoint-350`, and
  the next one is meant to pick that up. What stops a broken chain from silently
  retraining from scratch is the `latest` spec above, not the dependency.
* **A segment with nothing left to do exits 0 in seconds**, before the model loads, so
  the rest of the chain carries on.
* **One W&B run for the whole chain.** All segments share a run id derived from
  `EXP_ID` (`WB_RESUME=allow`), so the UI shows one continuous curve rather than N
  unrelated ones. `resume_experiment.sh` computes the same id, so a rescue lands in
  the same place.
* **Checkpoints from every segment share one `train/` directory** (`--no-run-subdir`),
  which is what lets the eval jobs enumerate the whole run rather than the last
  segment. `--resume-from-checkpoint auto`/`latest` therefore *require* that flag and
  refuse to run without it.

### Sharded evaluation

Both evaluations are submitted as **one job per five checkpoints**, and every one of
those jobs is held on the *training* job rather than on the previous evaluation — so
they all start together and run concurrently.

```
train ──┬─▶ lm-eval  shard 1 (base + checkpoints 100–500)  ──┐
        ├─▶ lm-eval  shard 2 (checkpoints 600–1000)         ├─▶ summary
        ├─▶ gameplay shard 1 (base + checkpoints 100–500)   │
        └─▶ gameplay shard 2 (checkpoints 600–1000)         ─┘
```

**Why.** The gameplay eval is generation, not loglikelihood scoring — hours per
checkpoint. Ten checkpoints do not fit one walltime on either cluster (Isambard's cap
is 24 h, and it is hard), and the previous answer was to trim the work with
`PPEVAL_CKPTS=last`. Sharding turns a walltime problem into a queue-width problem
instead, and evaluates every checkpoint.

**What it costs.** An experiment now asks for up to four GPUs at once instead of one.
On Isambard, Slurm reserves credits against each job's `--time` at submission. If that
is not wanted:

| variable | default | what it does |
|---|---|---|
| `EVAL_SHARD_SIZE` | `5` | checkpoints per evaluation job. `999` = one job per evaluation, i.e. the pre-sharding behaviour |
| `PPEVAL_SERIAL` | `0` | `1` holds the gameplay shards behind the lm-eval shards — one experiment, one GPU at a time |

**How a shard knows which checkpoints are its own.** `EVAL_SHARD` (1-based) is passed
per job through `-v` / `--export`, *not* through `experiment.env` — that file is
shared by every shard, so the index cannot live in it. Each job then takes its slice
of `exp_list_checkpoints`, which is sorted by step number and filtered to complete
adapters, so the slices tile the list exactly: no checkpoint is evaluated twice and
none is skipped.

Three details that are load-bearing rather than incidental:

* **Shard 1 owns the `base` row.** The untrained model's score does not depend on the
  checkpoint, and on the gameplay side it is a full pass over 14 games. Every other
  shard skips it. (It is also cached across experiments — the shard rule is what stops
  N concurrent shards racing to populate that cache the first time.)
* **A shard with an empty slice exits cleanly in seconds.** The shard count is decided
  at submit time, before training has written anything, by predicting
  `MAX_STEPS/SAVE_STEPS`; a run killed early leaves the last shard with nothing to do.
  Over-predicting is free, so the prediction is deliberately the optimistic one.
* **Each shard writes a partial table when it finishes**, so `RESULTS.md` is readable
  while the rest are still going — and the writes are atomic (`os.replace`), so two
  shards finishing together cannot leave a half-written file. The **summary job**
  writes the complete tables once the last shard has left the queue.

`TMPDIR` is per *job* (`$EXP_DIR/tmp/<jobid>`), not per experiment, for the same
reason: the cleanup trap would otherwise delete a sibling shard's torch/triton/vLLM
caches out from under it mid-run.

To rebuild the tables at any time without re-running anything:

```bash
EXP_ENV_FILE=<EXP_DIR>/experiment.env experiments/eddie/summarize.sh
EXP_ENV_FILE=<EXP_DIR>/experiment.env experiments/isambard/summarize.sh
```

### Recovering scores without replaying

Gameplay is hours of GPU; scoring and aggregation are seconds of CPU. If the second
part fails, the episodes are still on disk:

```bash
python experiments/lib/rescore_playpen_eval.py <EXP_DIR>
python experiments/lib/rescore_playpen_eval.py <EXP_DIR> --row checkpoint-100
python experiments/lib/rescore_playpen_eval.py --all $MARSHAL_RUNS
```

The job does this itself now — a row that fails with interaction files present is
rescored in place and reported `RECOVERED` — so this is the manual path for older runs.

Two things make it necessary. `clemcore.cli.score` reports any scoring error with a
bare `sys.exit(1)`: no traceback, no message, and `playpen eval` dies before `clemeval`
runs, so **one** broken game costs the aggregation for **all** of them. And a missing
`scikit-learn` makes `privateshared` exactly that broken game. On 2026-07-30 the pair
threw away ~3.5 h of GH200 aggregation across 8 rows from interaction files that were
completely intact; rescoring rebuilt it in under a minute. Hence also
`pp_eval_base_cached` playing into the experiment directory rather than a staging dir
it deletes on failure — the base row used to be the one thing that could not be
recovered.

> **`enable_thinking` lives in the wrong place in `model_registry.json`.** clemcore
> 3.7.2 only reads `model_config.chat_template_kwargs`; a top-level
> `model_config.enable_thinking` is silently ignored. Qwen3 then emits `<think>` blocks,
> clembench parses them as malformed moves, and **every episode aborts** — a whole
> benchmark at 0 % played that reads as "the model cannot play". `playpen_registry.py`
> rewrites the key when it clones an entry, and says so in the log. The shared
> `model_registry.json` still has it at the top level, which matters if you call
> `playpen eval <model>` directly.

### Scoring runs that finished before this existed

```bash
experiments/queue_playpen_eval_backfill.sh -n     # what WOULD be queued
experiments/queue_playpen_eval_backfill.sh        # queue it
experiments/queue_playpen_eval_backfill.sh 'guesswhat_*' --limit 3
```

Walks `$MARSHAL_RUNS`, finds every experiment with real checkpoints and no gameplay
results, and submits one eval job each through the same `run_playpen_eval.sh` a fresh
experiment would use — so a backfilled run is indistinguishable from one scored on
the day. Nothing is retrained and no checkpoint is touched.

Re-running it is safe: an experiment that already has results is skipped (`--force`
overrides), so it only ever queues what is genuinely missing. Since each job wants a
whole GPU for hours, `--limit N` is worth using on a busy queue — run it again
tomorrow and it picks up where it left off. `PPEVAL_*` is passed through to every job
it submits, so `PPEVAL_CKPTS=last experiments/queue_playpen_eval_backfill.sh` sets the
policy for a whole backfill in one place.

### Recomputing results that are wrong, not missing

```bash
experiments/queue_playpen_eval_backfill.sh --fresh -n     # what would be DELETED
experiments/queue_playpen_eval_backfill.sh --fresh        # confirm, then recompute
```

`--fresh` deletes each selected experiment's `playpen-eval/` and the shared base cache,
then queues everything from scratch. It prints the directories and their sizes and
requires a typed `yes` (or `--yes`) first, because unlike a failed run this **cannot**
be rebuilt by `rescore_playpen_eval.py` — the episodes themselves are gone.
Checkpoints are never touched.

Two reasons `--force` alone is not enough when results are *wrong* rather than missing:

* **Stale episodes get aggregated.** `--force` replays into a directory that still
  holds the previous run's episodes. If the game set changed, the games no longer
  being played are left behind and clemeval still counts them — silently mixing two
  runs into one table.
* **The base cache is a false hit.** Its key is
  `(model, games, max_tokens, temperature)` — it does not include the harness version.
  A baseline computed under a bug stays a cache hit afterwards and is copied into every
  new result as the row everything is compared against. `--force` does not clear it;
  `--fresh` does.

To (re-)score a single experiment instead:

```bash
experiments/eddie/run_playpen_eval.sh    $MARSHAL_RUNS/<experiment>
experiments/isambard/run_playpen_eval.sh $MARSHAL_RUNS/<experiment>
```

Like `run_eval.sh`, it reuses that experiment's own stored settings so the run stays
comparable, and any `PPEVAL_*` you export overrides them for that submission only.

### Did the checkpoints actually change?

```bash
.venv/bin/python experiments/check_checkpoints.py                    # everything under $MARSHAL_RUNS
.venv/bin/python experiments/check_checkpoints.py $MARSHAL_RUNS/<exp>
.venv/bin/python experiments/check_checkpoints.py <exp> --base Qwen/Qwen3-4B --top 10
```

```
     step  ||dW|| vs base  moved vs prev  max|dparam|  tensors moved         lr  grad_norm
      100      1.5597e+00     1.5597e+00   3.7723e-02        392/392   1.00e-06       0.83
      200      1.5597e+00              0            0          0/392   1.00e-06        0.0

  PROBLEM: checkpoint-200: byte-identical to checkpoint-100 -- no weight update landed
```

Reads the adapters' safetensors and reports the delta each one applies to the base model
(`scaling * B @ A`, summed over every LoRA module) and how far that delta moved between
checkpoints. Because PEFT zero-initialises `lora_B`, **column 1 is literally the change
versus the untrained model** — a checkpoint whose `lora_B` is still all-zero is a no-op
and cannot score differently from base, whatever `RESULTS.md` says. That distinguishes the
two ways a run "learns nothing": no optimizer step landed (frozen weights) versus the
weights moved and the behaviour did not. `--base` adds `||dW||/||W||` for relative sizing;
`--strict` exits non-zero for use in a script. Needs the training venv (numpy +
safetensors); like `status.sh` it only reads files.

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
| `MAX_STEPS` / `SAVE_STEPS` | `1000` / `100` in every preset | schedule and checkpoint cadence. `MAX_STEPS` is the **total** for the run however many jobs it is split across |
| `TRAIN_SEGMENTS` | `1` | chained training jobs, each resuming the last (see [Long runs](#long-runs-segments-and-resuming)) |
| `SEGMENT_STEPS` | — | steps per segment, instead of `TRAIN_SEGMENTS` |
| `RESUME_FROM` | — | `auto` \| `latest` \| a checkpoint path, for the first segment |
| `UNIQUE_POOL` | — | tri-state `1`/`0`/empty for `marshal_exact`'s `torch.unique` pooling (see [below](#marshal_exacts-unique-pooling-unique_pool)) |
| `EVAL_TASKS` | `logiglue,logicbench` | lm-eval tasks (comma-separated) |
| `EVAL_BASE` | `1` | also score the untrained model |
| `EVAL_LIMIT` | — | `--limit N`, smoke tests only |
| `EVAL_EXTRA` | — | extra lm-eval flags, applied to **every** row |
| `EVAL_SHARD_SIZE` | `5` | checkpoints per evaluation job (see [Sharded evaluation](#sharded-evaluation)) |
| `PPEVAL_SERIAL` | `0` | `1` = gameplay held behind lm-eval, one GPU at a time |
| `WB_PROJECT` | `playpen-marshal` | Weights & Biases project |
| `WB_ENABLE` | `1` | `0` turns W&B off for this run |
| `WB_MODE` | `auto` | `auto` \| `online` \| `offline` \| `disabled` |
| `WB_ENTITY` / `WB_GROUP` / `WB_TAGS` | — | W&B team, run group, extra tags |

Everything else lives in `presets/<game>.env` — that file is the per-game source of truth.

---

## Weights & Biases

On by default. The run is **named after the experiment** (`EXP_ID`), grouped by
`{game}_{model}`, and tagged with the switches that define the arm (`marshal` /
`no-marshal`, `dr_grpo`, `length_penalty`, `paper_correct` / `marshal_exact`,
`no-unique-pool`, the cluster, your `EXP_TAG`). Its config carries the *resolved* MARSHAL config plus every training
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

## The presets, and the length penalty

There is a preset for **every text-only clembench game** — one `presets/<game>.env` per
entry in a `clemgame.json` with `image: "none"`. Each holds that game's memory sizing.
See [Which games have presets](#which-games-have-presets) for the full list and for the
three that carry warnings.

### The length penalty

**No per-game calibration.** The presets used to carry `LP_MAX_LEN` / `LP_COEF` tuned per
game; they no longer do, and those two variables are inert. The penalty is now a flat
per-token cost with **no threshold**:

```
penalty(turn) = -length_penalty_per_token × generated_tokens        (LP_PER_TOKEN)
```

capped per seat per episode:

```
|Σ over one seat's turns| ≤ length_penalty_budget                   (LP_BUDGET)
```

applied as a proportional rescale of that episode's per-turn penalties, so relative
charge between turns survives and no sign flips.

Shipped defaults: `2e-5` per token (= −0.02 per 1000 tokens) and a `0.1` budget.

**Why the terminal reward still wins.** The penalty is charged per turn and the backward
cumulative return sums a seat's turns, so what competes with the ±1 outcome is the
*episode* total — which is exactly what the budget bounds. The smallest gap between two
distinct clembench outcomes is 1.0 (SUCCESS +1 / FAILURE 0 / ABORTED −1) and the penalty
is one-sided (always ≤ 0), so any budget below 1.0 means **the penalty can never reorder
two episodes that ended differently**. It ranks episodes *within* an outcome class —
shorter is better, all else equal — and nothing more. This is the same guarantee, and the
same mechanism, as `turn_reward_budget`; if both channels are on, keep
`LP_BUDGET + 2 × TR_BUDGET < 1.0`.

That bound holds whatever the game and whatever the turn count, which is why there is
nothing left to calibrate per game: a 3-turn taboo episode and a 50-turn adventuregame
episode are bounded identically.

**Turning it on:**

```bash
EXP_TAG=lp EXTRA_TRAIN_ARGS=--length-penalty \
    experiments/<cluster>/run_experiment.sh guesswhat

# a heavier hand, still safely bounded
EXP_TAG=lp_strong LP_PER_TOKEN=1e-4 LP_BUDGET=0.2 \
    EXTRA_TRAIN_ARGS=--length-penalty \
    experiments/<cluster>/run_experiment.sh guesswhat
```

**What to watch.** `marshal/length_penalty/episode_total_mean` — the per-turn mean is not
the number that competes with the outcome. Alongside it,
`marshal/length_penalty/budget_clip_rate`: a rate near 1.0 means the cap binds every
episode, so only the *shape* of the term survives and `LP_BUDGET`, not the game, is
setting its magnitude — lower `LP_PER_TOKEN` until it comes off the ceiling.
`manifest.txt` prints both the per-turn charge at the generation cap and the episode
bound at submit time.

**Deprecated:** `LP_MAX_LEN` and `LP_COEF` (and the YAML's `length_penalty_coef`,
`_bonus`, `_min_len`, `_max_len`, `_offset`) parameterized the previous threshold-based
penalty — a port of MARSHAL's Kimi-1.5-style `compute_length_penalty`, which scored
exactly 0 below `max_len` and fell linearly beyond it. That shape needed the threshold
tuned against the generation cap for every game, and got it wrong in both directions:
`max_len 500 / coef 0.1` was numerically inert on guesswhat (largest reachable penalty
−0.0025 per turn against a ±1 outcome), while `coef 0.4 / max_len 400` was −1.89 per
episode on dond, strong enough for staying silent to beat winning. It also charged
nothing at all for the ordinary-length turns that make up most of a run.

Those names are still **accepted** — existing YAMLs, pinned resume configs, cluster
presets and old manifests all keep loading — but they have **no effect**.
`train_selfplay.py` warns at startup if any of them is set, `manifest.json` records them
under `length_penalty_effect.legacy_fields_ignored`, and `status.sh` marks the run
`[legacy fields ignored]`.

> **Resuming a pre-rewrite run.** `check_resume_config.py` refuses to resume an
> experiment whose manifest has `length_penalty: true` but no
> `length_penalty_per_token` — no field *drifted*, but the formula reading those fields
> changed, so the second half of the run would train under a different reward from the
> first. Prefer a fresh run; `RESUME_FORCE=1` overrides.

> **Reporting caveat.** `length_penalty: true` and `fidelity_mode: marshal_exact`
> interact: the penalty makes every trajectory return distinct, which collapses
> `marshal_exact`'s distinct-value pooling into ordinary occurrence-weighted pooling.
> With no threshold this now happens on *every* game, not just the ones whose turns ran
> long. Both are defensible algorithms and the gradient direction is unchanged, but an
> on/off length-penalty ablation at `marshal_exact` also silently switches pooling rule —
> pin `fidelity_mode` consistently across arms, and don't describe such a run as
> reproducing MARSHAL's shipped normalization.
>
> `UNIQUE_POOL=0` makes that switch *explicit* instead of incidental: both arms then pool
> occurrence-weighted by construction, so the length penalty is the only thing varying.
> See [`marshal_exact`'s unique pooling](#marshal_exacts-unique-pooling-unique_pool).

---

## Which games have presets

Every text-only clembench game (`image: "none"`). The multimodal ones — `matchit`,
`mm_mapworld*`, `multimodal_referencegame`, `mm_clean_up`, `st_clean_up`,
`hybrid_clean_up` — have none, and neither do the five `static` benchmarks
(`bbh`, `cladder`, `eqbench`, `ifeval`, `mmlu_pro`): those are single-turn
multiple-choice tasks, not games, and job 3 already scores them via `PPEVAL_SUITE`.

**"Learner seats" is the column that matters**, and it is not always what
`clemgame.json` says. `SelfPlayEnv` takes `num_players` from the `GameSpec` and marks
every seat as the learner, while the game master decides how many clemcore players
actually exist and which of them are programmatic. Where the two disagree, the
consequence is in the notes column, and it is spelled out at the top of the preset.

| game | learner seats | turns/seat | notes |
|---|---|---|---|
| `taboo` | 2 | 3 | calibrated |
| `guesswhat` | 2 | 8 | calibrated |
| `dond` | 2 | 5 | calibrated; **not** in the `clem` suite (benchmark 3.0) |
| `codenames` | 2 | ~4 | turn count estimated |
| `referencegame` | 2 | **1** | one round only — turn-level credit has nothing to spread |
| `matchit_ascii` | 2 | ~5 | `decision_turn = 3` in every instance |
| `wordle` | 1 | 6 | single-player: per-seat pooling inert |
| `wordle_withclue` | 1 *(spec says 2)* | 6 | ⚠ only one clemcore player exists — **half of every batch is an inert seat-1 placeholder** |
| `wordle_withcritic` | 2 | 12 / 6 | genuinely two models; the seats take *different* turn counts |
| `hot_air_balloon` | 2 | unbounded | ⚠ **no built-in round cap** — `MAX_TURNS` is a real cap, and hitting it truncates before the ±1 reward |
| `textmapworld` | 1 *(2 agents)* | 20 | describer is programmatic and correctly auto-stepped |
| `textmapworld_graphreasoning` | 1 *(2 agents)* | 20 | as above; free-text answers, so the penalty may not be inert |
| `textmapworld_specificroom` | 1 *(2 agents)* | 20 | as above |
| `clean_up` | 2 | 12–28 | ⚠ **needs `matplotlib` in the venv** or the game cannot be imported |
| `imagegame` | 2 | up to 50 | `max_rounds = grid²×2 = 50`; text-only despite the name |
| `adventuregame` | 1 | 50 (100 on `potion_brewing`) | longest episodes in clembench; `PER_DEVICE_BATCH=1`, expect slow steps |
| `privateshared` | — | — | 🚫 **do not train on this** — see below |

### `privateshared` is documented, not usable for training

Its preset exists so `run_experiment.sh privateshared` fails informatively rather than
with "no preset for game". Two independent problems, both read out of
`clembench/privateshared/master.py`, make a training run there meaningless:

* its **Questioner is programmatic** (`CustomResponseModel`, replaying a fixed
  `request_order`) but `clemgame.json` declares `players: 2`, so the port marks it a
  learner seat and the policy is asked to generate the scripted questions;
* the **probing — the mechanic the game actually scores — bypasses the agent loop**:
  `_probing_loop` calls `self.answerer(context, memorize=False)` directly, which under
  the learner marker model falls through to `Answerer._custom_response` and returns a
  random `"<tag>yes, placeholder."` string. Every probe would be scored against a mock.

None of this affects **evaluation** — `playpen eval` gives the answerer a real backend,
so privateshared is scored normally as part of the `clem` suite. It also needs
`scikit-learn`, which job 3 needs anyway (see the recovery section above).

### Where a preset deviates from the shared block

Most presets carry the same memory block (`PER_DEVICE_BATCH=4`, `GRAD_ACCUM=16`,
`VLLM_MAX_MODEL_LEN=16384`, `MAX_COMPLETION_LENGTH=1024`). A rollout row is the *whole
flattened episode for one seat*, so `seq_len ≈ turns × MAX_COMPLETION_LENGTH` — and on
the long-episode games that block does not fit. Those files say
`DEVIATES FROM THE SHARED BLOCK` at the top of the section and show the arithmetic. The
rule used throughout: keep the per-forward token budget near where guesswhat verified it
(`4 × 8192 ≈ 32k`), and keep `PER_DEVICE_BATCH × GRAD_ACCUM = 64` so the MARSHAL
advantage pool is identical across every game.

| game | `MAX_COMPLETION_LENGTH` | batch × accum | `VLLM_MAX_MODEL_LEN` | why |
|---|---|---|---|---|
| most games | 1024 | 4 × 16 | 16384 | — |
| `wordle_withcritic`, `hot_air_balloon` | 1024 | 2 × 32 | 24576 | ~12–20 turns/seat |
| `textmapworld*` | 512 | 2 × 32 | 24576 | 20 turns/seat of five-token moves |
| `clean_up` | 512 | 2 × 32 | 24576 | up to 28 rounds |
| `imagegame` | 256 | 2 × 32 | 32768 | up to 50 rounds of one-line commands |
| `adventuregame` | 256 | **1 × 64** | 32768 | up to 100 turns — a single ~26k-token row |

---

## `marshal_exact`'s unique pooling (`UNIQUE_POOL`)

`fidelity_mode: marshal_exact` bundles **two** departures from the MARSHAL paper: (a) a
pre-sum reward normalization, and (b) `torch.unique` distinct-value pooling of trajectory
returns, which equal-weights rare and common outcomes. `UNIQUE_POOL` splits (b) out so an
ablation can attribute a result to one of them rather than to both at once.

```bash
# marshal_exact as shipped (both departures) -- the default, no var needed
EXTRA_TRAIN_ARGS='--fidelity-mode marshal_exact' EXP_TAG=exact ./run_experiment.sh guesswhat
# its pre-sum pass, but occurrence-weighted pooling
EXTRA_TRAIN_ARGS='--fidelity-mode marshal_exact' UNIQUE_POOL=0 EXP_TAG=exact_nounique \
    ./run_experiment.sh guesswhat
```

| var | flag | meaning |
|---|---|---|
| `UNIQUE_POOL` | `--marshal-exact-unique-pooling` / `--no-...` | tri-state: `1` on, `0` off, **empty = leave the YAML alone** (same convention as `TR_*` / `LP_*`) |

**It only bites under `marshal_exact`.** `paper_correct` never pools over distinct values,
so `UNIQUE_POOL=0` is a no-op there and `UNIQUE_POOL=1` cannot switch unique pooling on.
`fidelity_mode` itself has no env var — set it in the YAML or via `EXTRA_TRAIN_ARGS`, as
above. The resolved value is recorded in `manifest.json`, W&B tags a run that turned it
off with `no-unique-pool`, and `status.sh -v` renders it as
`fidelity=marshal_exact(no-unique-pool)`.

Why it matters on clembench: returns only ever take the values `{-1, 0, +1}`, so uniquing
makes the baseline the *midpoint of whichever outcomes occur* rather than the empirical
mean — 9 wins and 1 loss give a baseline of `0.0` uniqued against `0.8`
occurrence-weighted, i.e. the loss is charged 9× harder relative to the wins. Note that
`length_penalty` or `TR_*` make returns near-continuous, at which point unique pooling
already degenerates into occurrence-weighted pooling by itself and this var changes little
— and with the length penalty now charging every turn, that happens on every game rather
than only where turns ran long (see the reporting caveat under
[The length penalty](#the-length-penalty)).

**Reporting.** A run with `UNIQUE_POOL=0` is not a reproduction of MARSHAL's shipped
normalization; describe it as `marshal_exact` minus distinct-value pooling.

---

## Dense per-turn rewards (`TR_*`)

Off by default. Turn them on for a run the same way as everything else — an env var,
which the job script turns into a CLI flag, which overrides the YAML:

```bash
TR_ENABLE=1 EXP_TAG=turnrew ./run_experiment.sh wordle
TR_ENABLE=1 TR_SCALE=0.1 TR_BUDGET=0.4 EXP_TAG=turnrew_strong ./run_experiment.sh codenames
TR_ENABLE=0 EXP_TAG=no_turnrew ./run_experiment.sh wordle        # force off over a YAML that says on
```

| var | flag | meaning |
|---|---|---|
| `TR_ENABLE` | `--turn-rewards` / `--no-turn-rewards` | tri-state: `1` on, `0` off, **empty = leave the YAML alone** (same convention as `LP_*`) |
| `TR_SOURCE` | `--turn-reward-source` | `auto` (default) / `game` / `generic` |
| `TR_SCALE` | `--turn-reward-scale` | most a single turn can contribute (default `0.05`) |
| `TR_BUDGET` | `--turn-reward-budget` | cap on an episode's shaping total (default `0.3`) |
| `TR_COMPONENTS` | `--turn-reward-components` | allowlist, e.g. `closeness` |

**No per-game calibration is needed** (same story as the length penalty): extractors
normalize every component to `[-1, 1]`, so a turn is worth at most `TR_SCALE` and an
episode at most `TR_BUDGET` whatever the game's turn count. Keeping `TR_BUDGET` under `0.5` is
what guarantees shaping cannot reorder two different clembench outcomes (the worst-case
swing `2 × budget` stays under the 1.0 gap between them); `manifest.txt` prints the
numbers and warns if that stops holding.

Watch `marshal/turn_rewards/sum_abs_mean` (is there any signal at all?) and
`marshal/turn_rewards/budget_clip_rate` (is the cap, rather than the game, setting the
magnitude?). A flat zero on taboo or guesswhat is the honest result — their only
per-turn signal is rule compliance, and a compliant policy trips nothing. Full details
and the per-game component table: [`examples/marshal/README.md`](../examples/marshal/README.md#dense-per-turn-rewards).

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

**The playpen games evaluation needs no third environment** — it runs in the repo's
own `.venv`, where clemcore, playpen and peft already are. What it does need is
`clembench/` cloned in the repo root (as usual — that is how the games are
discovered) and the `colab-potsdam/playpen-data` validation split reachable or
cached; it checks for the latter before loading a model and tells you the command
to fetch it.

---

## Design notes

**Why a separate eval job rather than eval at the end of training.** Different environment
(lm-eval + peft, not trl + vllm), different walltime, different GPU needs. Chaining also
means a failed eval can be re-run without redoing 20 h of training.

**Why the playpen eval is a separate job rather than part of the lm-eval one.** It
needs the opposite environment: the repo's own venv rather than the lm-eval one, and
it is generation rather than loglikelihood scoring, so its walltime is an order of
magnitude larger.

**Why it now runs beside lm-eval rather than behind it.** It used to be chained, which
kept one experiment to one GPU at a time. That stopped being the right trade at
`MAX_STEPS=1000`: ten checkpoints of gameplay do not fit one walltime at all, so the
jobs had to be split anyway — and once they are split, holding them in a chain only
adds queue latency to a set of jobs that are already independent (they write to
disjoint `playpen-eval/<checkpoint>/` directories). `PPEVAL_SERIAL=1` restores the
chain when the queue cannot take the width.

**Why the tables are written by a fifth job rather than by the last shard.** "The last
shard" is not knowable from inside a shard — each one only sees its own slice — and
having every shard write the table would mean the final one silently decides what the
table says. A job held on all of them runs exactly once, after all of them, and reads
the whole directory. It is seconds of CPU, so a dedicated job costs nothing; shards
still write partial tables as they go, atomically, so there is something to read in the
meantime.

**A checkpoint is made addressable by a generated registry, not by merging.**
`playpen eval` takes a registered model *name*; a checkpoint is a LoRA adapter
directory. Rather than merge each adapter into full base weights (GBs per
checkpoint, and a merge step that can itself be wrong), the eval job writes a
throwaway `model_registry.json` naming the base with `model_config.peft_model` set to
the adapter — a path clemcore's `huggingface_local` backend already supports — into a
private working directory it then `cd`s into. The repo's own registry is never
modified, so concurrent jobs cannot race on it and a killed job leaves the checkout
clean. That entry is **cloned** from the base model's real registry entry rather than
synthesized, because settings like Qwen3's `enable_thinking: false` are the
difference between playing the game and emitting `<think>` blocks that score as
ABORTED.

**The dependency is `afterany`, not `afterok`.** A 20 h run that hits the walltime exits
non-zero but has still written `checkpoint-150`; `afterok` would silently skip evaluating
it. `eval.sh` instead checks for real checkpoints and exits cleanly with a message when
there genuinely are none. Grid Engine's `-hold_jid` has the same semantics, so both
clusters behave identically. The same reasoning is why the summary job is `afterany` on
every shard: one shard that OOMs must not cost the table for the rest.

**Walltimes are at the cluster maximum for training, and deliberately not for
evaluation.** `train_selfplay.py` has no `--resume-from-checkpoint`, so a training job
killed at the walltime can only be redone — hence 48 h on Eddie and 24 h (the hard cap)
on Isambard. Evaluation has the opposite property: a shard that dies leaves the
checkpoints it already scored on disk, the rest can be re-submitted with
`run_eval.sh` / `run_playpen_eval.sh`, and gameplay that was played but not scored is
recoverable with `rescore_playpen_eval.py`. Sharding, not a longer walltime, is what
makes the evaluation fit. On Isambard this also matters for cost: Slurm reserves
credits against `--time` at submission, so asking every eval shard for 24 h would book
four times the run's credits up front.

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

#!/bin/bash
# Shared experiment layout + preset loading. SOURCED by experiments/*/*.sh, never run.
#
# One experiment == one directory. Everything about that experiment lives inside it:
# the manifest (what was run), the frozen config, the training checkpoints, the eval
# results, and both job logs. Nothing is written to a global Results/ dir, so an
# experiment is a single self-contained thing to inspect, rsync home, or delete.
#
#   $MARSHAL_RUNS/<EXP_ID>/
#   +- manifest.json          machine-readable: model, game, every hyperparameter,
#   |                         resolved MarshalConfig, git sha, package versions
#   +- manifest.txt           the same thing, human-readable (`cat` it)
#   +- marshal_config.yaml    frozen copy of the config this run actually used
#   +- RESULTS.md             eval scores per checkpoint (written by the eval job)
#   +- results.tsv            the same table, for pandas/excel
#   +- logs/                  train.<jobid>.out/.err, eval.<jobid>.out/.err
#   +- train/<timestamp>/     checkpoints + completions parquet (train_selfplay.py
#   |                         always appends its own timestamp subdir)
#   +- eval/
#      +- base/               lm-eval on the untrained base model (the baseline)
#      +- checkpoint-50/      lm-eval on that adapter
#      +- checkpoint-100/ ...
#
# EXP_ID is <game>_<model>_<tag>_<date-time>, e.g. dond_Qwen3-4B_lp384_20260723-142530.
# The tag is what makes an ablation legible six weeks later -- always set EXP_TAG when
# a run differs from the preset (EXP_TAG=lp_off, EXP_TAG=paper_correct, ...).

# --- refuse execution --------------------------------------------------------
(return 0 2>/dev/null) || {
    echo "ERROR: source this file, don't execute it: source ${0}" >&2
    exit 1
}

EXP_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT_DIR="$(dirname "$EXP_LIB_DIR")"          # experiments/
REPO="${REPO:-$(dirname "$EXP_ROOT_DIR")}"        # the playpen checkout

[ -d "$REPO/playpen/marshal" ] || {
    echo "ERROR: \$REPO=$REPO is not a playpen checkout (no playpen/marshal/)." >&2
    return 1
}

# --- preset ------------------------------------------------------------------
# Loads experiments/presets/<game>.env, then applies defaults for anything it did
# not set. Caller-exported values always win: `MAX_STEPS=10 ./run_experiment.sh dond`
# overrides the preset, which overrides the defaults here.
exp_load_preset() {
    local game="$1"
    local preset="$EXP_ROOT_DIR/presets/${game}.env"
    [ -f "$preset" ] || {
        echo "ERROR: no preset for game '$game' (expected $preset)." >&2
        echo "       Available: $(ls "$EXP_ROOT_DIR/presets" 2>/dev/null | sed 's/\.env//' | tr '\n' ' ')" >&2
        return 1
    }
    # Snapshot caller-supplied values BEFORE sourcing the preset.
    #
    # Preset files use plain assignments (MAX_STEPS=200) rather than
    # MAX_STEPS="${MAX_STEPS:-200}", which keeps them readable -- but it also means
    # sourcing one would clobber an explicit `MAX_STEPS=20 run_experiment.sh dond`.
    # That failure is silent and expensive: you ask for a 20-step smoke test and get
    # the preset's 200-step run. So we save what the caller set, source the preset,
    # then put the caller's values back on top.
    #
    # `${!v+set}` tests for *set*, not non-empty, so an intentional `LP_PER_TOKEN=` (meaning
    # "leave this to the YAML") is preserved rather than treated as absent.
    local _override_names=() _override_vals=() _v _i
    for _v in MODEL MARSHAL_CONFIG \
              NUM_GENERATIONS PER_DEVICE_BATCH GRAD_ACCUM MAX_STEPS SAVE_STEPS \
              LEARNING_RATE KL_BETA MAX_COMPLETION_LENGTH MAX_TURNS GRAD_CKPT \
              VLLM_UTIL VLLM_MAX_MODEL_LEN LP_PER_TOKEN LP_BUDGET LP_MAX_LEN LP_COEF UNIQUE_POOL \
              TR_ENABLE TR_SOURCE TR_SCALE TR_BUDGET TR_COMPONENTS \
              TRAIN_SEGMENTS SEGMENT_STEPS RESUME_FROM \
              EVAL_TASKS EVAL_BATCH EVAL_BASE EVAL_LIMIT EVAL_EXTRA EVAL_SHARD_SIZE \
              PPEVAL_ENABLE PPEVAL_SUITE PPEVAL_GAMES PPEVAL_BASE PPEVAL_CKPTS \
              PPEVAL_MAX_TOKENS PPEVAL_TEMPERATURE PPEVAL_TIMEOUT PPEVAL_SERIAL \
              WB_ENABLE WB_PROJECT WB_ENTITY WB_GROUP WB_TAGS WB_MODE WB_ID WB_RESUME \
              EXTRA_TRAIN_ARGS EXP_TAG; do
        if [ -n "${!_v+set}" ]; then
            _override_names+=("$_v")
            _override_vals+=("${!_v}")
        fi
    done

    # shellcheck disable=SC1090
    source "$preset"

    for _i in "${!_override_names[@]}"; do
        printf -v "${_override_names[$_i]}" '%s' "${_override_vals[$_i]}"
    done
    # A caller's TRAIN_SEGMENTS must not be overruled by a preset's SEGMENT_STEPS
    # (or vice versa). Runs here, where the caller-set names are still known.
    exp_segment_override ${_override_names[@]+"${_override_names[@]}"}

    # GAME is deliberately NOT restorable: the positional argument is authoritative.
    GAME="${GAME:-$game}"
    MODEL="${MODEL:-Qwen/Qwen3-4B}"
    MARSHAL_CONFIG="${MARSHAL_CONFIG:-examples/marshal/marshal_config.yaml}"

    NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
    PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
    GRAD_ACCUM="${GRAD_ACCUM:-8}"
    MAX_STEPS="${MAX_STEPS:-200}"
    SAVE_STEPS="${SAVE_STEPS:-50}"
    LEARNING_RATE="${LEARNING_RATE:-1e-5}"
    KL_BETA="${KL_BETA:-0.2}"
    MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-768}"
    MAX_TURNS="${MAX_TURNS:-30}"
    GRAD_CKPT="${GRAD_CKPT:-1}"
    VLLM_UTIL="${VLLM_UTIL:-0.30}"
    VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

    # Length penalty: a flat per-token cost, capped per episode. Empty
    # LP_PER_TOKEN/LP_BUDGET => don't pass the flag, so the YAML value stands. No
    # per-game calibration is needed any more (the cap makes the episode total
    # game-independent), so presets set neither by default.
    LP_PER_TOKEN="${LP_PER_TOKEN:-}"
    LP_BUDGET="${LP_BUDGET:-}"

    # DEPRECATED and inert: these configured the old threshold-based penalty. Still
    # forwarded (so an existing preset or command line runs unchanged) and still
    # recorded in the manifest, which flags them as ignored -- train_selfplay.py
    # prints the same warning at startup.
    LP_MAX_LEN="${LP_MAX_LEN:-}"
    LP_COEF="${LP_COEF:-}"

    # marshal_exact's distinct-value (torch.unique) pooling, as a tri-state like
    # TR_ENABLE: 1 = force it on, 0 = force it off (keeping marshal_exact's pre-sum
    # reward normalization but pooling occurrence-weighted), empty = leave the YAML
    # alone. Only bites when fidelity_mode is marshal_exact -- paper_correct never
    # uniques, so UNIQUE_POOL=0 is a no-op there and UNIQUE_POOL=1 cannot turn it on.
    # fidelity_mode itself has no env var; set it in the YAML or via
    # EXTRA_TRAIN_ARGS='--fidelity-mode marshal_exact'.
    UNIQUE_POOL="${UNIQUE_POOL:-}"

    # Dense per-turn rewards (playpen/marshal/turn_rewards.py). Same convention as
    # LP_*: empty => don't pass the flag at all, so the YAML value stands. Set
    # TR_ENABLE=1 to force them on for a run, TR_ENABLE=0 to force them off.
    # TR_SCALE/TR_BUDGET are the calibration knobs -- see the marshal_config.yaml
    # block and experiments/README.md.
    TR_ENABLE="${TR_ENABLE:-}"
    TR_SOURCE="${TR_SOURCE:-}"
    TR_SCALE="${TR_SCALE:-}"
    TR_BUDGET="${TR_BUDGET:-}"
    TR_COMPONENTS="${TR_COMPONENTS:-}"

    # --- splitting the training run across several chained jobs --------------
    # MAX_STEPS is ALWAYS the total for the whole experiment, whatever these are set
    # to. TRAIN_SEGMENTS only decides how many jobs those steps are handed out over;
    # each one resumes the previous one's last checkpoint and stops at its own
    # boundary. Defaults to 1, which submits exactly the single training job this
    # directory has always submitted.
    #
    # WHAT IT IS FOR. Eddie's gpu queue caps h_rt at 48 h and Isambard's workq_qos caps
    # --time at 24 h; neither can be raised, and until now a run that ran out of
    # walltime could only be redone. TRAIN_SEGMENTS=3 turns a 1000-step run into three
    # jobs of <=400 steps, each of which fits comfortably and backfills far better than
    # one job asking for the cap.
    #
    #   TRAIN_SEGMENTS=3 experiments/eddie/run_experiment.sh guesswhat
    #   TRAIN_SEGMENTS=4 TRAIN_SBATCH_OPTS='--time=08:00:00' \
    #     experiments/isambard/run_experiment.sh guesswhat
    #
    # SEGMENT_STEPS overrides the derived size when you know the per-step cost and want
    # to size a segment against the walltime directly; TRAIN_SEGMENTS is then
    # recomputed from it. Set neither and there is one segment. See exp_plan_segments.
    #
    # The LR schedule does NOT change with the segment count -- --max-steps stays the
    # total in every job and only --stop-at-step differs, so step 700 of a 3-segment
    # run has the same learning rate as step 700 of a 1-segment run. That is what makes
    # a segmented arm comparable with an unsegmented one; see playpen/marshal/resume.py.
    TRAIN_SEGMENTS="${TRAIN_SEGMENTS:-1}"
    SEGMENT_STEPS="${SEGMENT_STEPS:-}"

    # What each training job passes to --resume-from-checkpoint. Segment 1 gets 'auto'
    # (resume if there is something, else start at step 0) and later segments get
    # 'latest' (an empty train/ is an error), so the submitter does not have to set
    # this at all for a chained run.
    #
    # Set it by hand to restart a dead run in place -- RESUME_FROM=latest with the
    # same EXP_DIR -- which is what experiments/*/resume_experiment.sh does for you.
    # An explicit checkpoint path also works.
    RESUME_FROM="${RESUME_FROM:-}"

    # --- Weights & Biases ----------------------------------------------------
    # Named WB_* rather than WANDB_* on purpose: WANDB_MODE, WANDB_PROJECT and
    # friends are read by the wandb SDK itself, and exporting WB_MODE=auto as
    # WANDB_MODE would hand the SDK a value it rejects. train_selfplay.py still
    # honours the real WANDB_* variables if you prefer to set those directly.
    #
    # On by default because the cost of it being on is a directory of offline run
    # data inside the experiment folder, and the cost of it being off is a finished
    # 5 h run you cannot plot. WB_MODE=auto uploads live when a credential is
    # reachable and records offline otherwise, so a compute node with no outbound
    # network is not a failure case.
    WB_ENABLE="${WB_ENABLE:-1}"
    WB_PROJECT="${WB_PROJECT:-playpen-marshal}"
    WB_ENTITY="${WB_ENTITY:-}"           # empty => your account's default entity
    WB_GROUP="${WB_GROUP:-}"             # empty => {game}_{model}, set by the trainer
    WB_TAGS="${WB_TAGS:-}"               # extra comma-separated tags; switches are automatic
    WB_MODE="${WB_MODE:-auto}"           # auto | online | offline | disabled
    WB_ID="${WB_ID:-}"                   # set with WB_RESUME=allow to continue a run
    WB_RESUME="${WB_RESUME:-}"

    EVAL_TASKS="${EVAL_TASKS:-logiglue,logicbench}"
    EVAL_BATCH="${EVAL_BATCH:-16}"
    EVAL_BASE="${EVAL_BASE:-1}"          # also score the untrained base model
    EVAL_LIMIT="${EVAL_LIMIT:-}"         # smoke tests only
    EVAL_EXTRA="${EVAL_EXTRA:-}"         # verbatim extra lm-eval flags

    # --- how many checkpoints one evaluation job takes -----------------------
    # BOTH evaluations (lm-eval and the playpen games eval) are submitted as one job
    # per shard of this many checkpoints, all of them held on the training job and
    # therefore running CONCURRENTLY. With the presets' MAX_STEPS=1000 /
    # SAVE_STEPS=100 that is 10 checkpoints -> 2 lm-eval jobs + 2 gameplay jobs.
    #
    # Why shard at all: the gameplay eval is generation, hours per checkpoint, and a
    # 10-checkpoint run does not fit one walltime on either cluster. Sharding turns a
    # walltime problem into a queue-width problem.
    #
    # Set it larger than the checkpoint count for the old single-job behaviour
    # (EVAL_SHARD_SIZE=999), or smaller for more, shorter jobs.
    EVAL_SHARD_SIZE="${EVAL_SHARD_SIZE:-5}"

    # --- playpen games eval (the third job) ----------------------------------
    # Plays every checkpoint through clembench and reports the clemscore. Game
    # independent on purpose: the point is to score the 13 games the run was NOT
    # trained on as well as the one it was. See lib/playpen_eval.sh for what each
    # of these does; a preset may still override them like anything else.
    PPEVAL_ENABLE="${PPEVAL_ENABLE:-1}"          # 0 => don't queue the job at all
    PPEVAL_SUITE="${PPEVAL_SUITE:-clem}"         # clem | static | all
    PPEVAL_GAMES="${PPEVAL_GAMES:-}"             # explicit list, overrides the suite
    PPEVAL_BASE="${PPEVAL_BASE:-1}"              # also play the untrained model
    PPEVAL_CKPTS="${PPEVAL_CKPTS:-}"             # "" = all | last | 100,200
    PPEVAL_MAX_TOKENS="${PPEVAL_MAX_TOKENS:-300}"
    PPEVAL_TEMPERATURE="${PPEVAL_TEMPERATURE:-0.0}"
    PPEVAL_TIMEOUT="${PPEVAL_TIMEOUT:-}"         # per-row ceiling, e.g. 3h
    # 1 => hold the gameplay shards behind the lm-eval shards instead of beside them,
    # i.e. the old one-GPU-per-experiment chain. Default 0 runs everything the moment
    # training finishes, which is faster in wall-clock and wider in the queue.
    PPEVAL_SERIAL="${PPEVAL_SERIAL:-0}"

    EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
    EXP_TAG="${EXP_TAG:-}"

    # Export everything: write_manifest.py reads these from the environment, and the
    # job scripts re-source this file in a fresh shell. Without the export the
    # manifest silently records an empty training section.
    export GAME MODEL MARSHAL_CONFIG \
           NUM_GENERATIONS PER_DEVICE_BATCH GRAD_ACCUM MAX_STEPS SAVE_STEPS \
           LEARNING_RATE KL_BETA MAX_COMPLETION_LENGTH MAX_TURNS GRAD_CKPT \
           VLLM_UTIL VLLM_MAX_MODEL_LEN LP_PER_TOKEN LP_BUDGET LP_MAX_LEN LP_COEF UNIQUE_POOL \
           TR_ENABLE TR_SOURCE TR_SCALE TR_BUDGET TR_COMPONENTS \
           TRAIN_SEGMENTS SEGMENT_STEPS RESUME_FROM \
           EVAL_TASKS EVAL_BATCH EVAL_BASE EVAL_LIMIT EVAL_EXTRA EVAL_SHARD_SIZE \
           PPEVAL_ENABLE PPEVAL_SUITE PPEVAL_GAMES PPEVAL_BASE PPEVAL_CKPTS \
           PPEVAL_MAX_TOKENS PPEVAL_TEMPERATURE PPEVAL_TIMEOUT PPEVAL_SERIAL \
           WB_ENABLE WB_PROJECT WB_ENTITY WB_GROUP WB_TAGS WB_MODE WB_ID WB_RESUME \
           EXTRA_TRAIN_ARGS EXP_TAG
}

# Build train_selfplay.py's W&B flags into the global array $WANDB_ARGS.
#
# A global array rather than a printed string because the values contain spaces
# (a group or a note), and word-splitting a printed string would silently break
# them apart -- the same reason ARGS=() is built element by element in train.sh.
#
# The run is named after the experiment ($EXP_ID) and its data is written inside
# $EXP_DIR, so the link between a row in the W&B UI and a directory on the cluster
# is the identity of their names, not a mapping anyone has to maintain. Requires
# EXP_ID/EXP_DIR, i.e. call it after exp_make_id + exp_layout.
exp_wandb_args() {
    WANDB_ARGS=()
    if [ "${WB_ENABLE:-1}" != "1" ] || [ "${WB_MODE:-auto}" = "disabled" ]; then
        WANDB_ARGS=( --no-wandb )
        return 0
    fi
    WANDB_ARGS=(
        --wandb
        --wandb-project "${WB_PROJECT:-playpen-marshal}"
        --wandb-mode "${WB_MODE:-auto}"
        --wandb-run-name "${EXP_ID:-marshal}"
        # Offline run data lands in $EXP_DIR/wandb/, so it travels with the rest of
        # the experiment when the directory is rsynced home (nothing on the clusters
        # is backed up) and disappears with it when the experiment is deleted.
        --wandb-dir "${EXP_DIR:-.}"
        --wandb-notes "experiment ${EXP_ID:-?} on ${EXP_CLUSTER:-?}"
    )
    [ -n "${WB_ENTITY:-}" ] && WANDB_ARGS+=( --wandb-entity "$WB_ENTITY" )
    [ -n "${WB_GROUP:-}"  ] && WANDB_ARGS+=( --wandb-group "$WB_GROUP" )
    [ -n "${WB_TAGS:-}"   ] && WANDB_ARGS+=( --wandb-tags "$WB_TAGS" )
    [ -n "${WB_ID:-}"     ] && WANDB_ARGS+=( --wandb-id "$WB_ID" )
    [ -n "${WB_RESUME:-}" ] && WANDB_ARGS+=( --wandb-resume "$WB_RESUME" )
    return 0
}

# --- identity + layout -------------------------------------------------------
exp_make_id() {
    local model_base tag stamp
    model_base="$(basename "$MODEL")"
    stamp="$(date +%Y%m%d-%H%M%S)"
    tag="${EXP_TAG:+_${EXP_TAG}}"
    # Slashes/spaces in a tag would fork the directory tree; flatten them.
    tag="$(printf '%s' "$tag" | tr -c 'A-Za-z0-9._-' '_')"
    export EXP_ID="${GAME}_${model_base}${tag}_${stamp}"
}

exp_layout() {
    : "${EXP_DIR:?exp_layout needs EXP_DIR}"
    TRAIN_BASE="$EXP_DIR/train"
    EVAL_DIR="$EXP_DIR/eval"
    LOG_DIR="$EXP_DIR/logs"
}

# The directory holding this run's checkpoints.
#
# Handles both layouts. The job scripts pass --no-run-subdir, so checkpoints land
# directly in $TRAIN_BASE -- checked first. Without that flag train_selfplay.py
# appends its own unpredictable %Y%m%d-%H%M%S subdir; EXP_DIR is fresh per
# experiment so there is exactly one candidate, which also keeps this working for
# experiments created before the flag existed.
#
# Printing nothing (and returning 1) means training never got far enough to create it.
exp_find_run_dir() {
    local d
    if compgen -G "$TRAIN_BASE/checkpoint-*" > /dev/null 2>&1; then
        printf '%s\n' "$TRAIN_BASE"
        return 0
    fi
    d="$(find "$TRAIN_BASE" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)"
    [ -n "$d" ] || return 1
    printf '%s\n' "$d"
}

# Checkpoint dirs, oldest step first. Two filters matter:
#   * only dirs holding a real adapter qualify -- a half-written checkpoint from a
#     job killed at the walltime has no adapter_config.json and must not reach lm-eval;
#   * ordering is by the STEP NUMBER, extracted per directory. Sorting the full
#     paths with `sort -t- -k2 -n` looks right and is not: the run path itself
#     contains '-' (timestamps, model names), so field 2 is rarely the step and
#     checkpoints come back as 100, 200, 50 -- which silently mislabels a learning
#     curve in RESULTS.md.
exp_list_checkpoints() {
    local run_dir="$1" ck step
    for ck in "$run_dir"/checkpoint-*; do
        [ -d "$ck" ] || continue
        [ -f "$ck/adapter_config.json" ] || continue
        step="${ck##*/checkpoint-}"
        case "$step" in
            ''|*[!0-9]*) continue ;;   # e.g. a "checkpoint-150-partial" leftover
        esac
        printf '%s\t%s\n' "$step" "$ck"
    done | sort -n -k1,1 | cut -f2-
}

# --- training segmentation ---------------------------------------------------
# The training half of the same idea as evaluation sharding below: one long run,
# submitted as several chained jobs, because a cluster walltime cap cannot be raised.
# The difference is that eval shards run CONCURRENTLY (they are independent) while
# training segments run in SEQUENCE (each resumes the last one's checkpoint).
#
# The contract, which everything downstream depends on:
#
#   * MAX_STEPS is the total for the run, in every segment. Only --stop-at-step
#     differs between them. That is what keeps the LR schedule -- which HF builds from
#     max_steps -- identical to an uninterrupted run, so a segmented arm and an
#     unsegmented arm of the same config are comparable. See playpen/marshal/resume.py.
#   * Segment boundaries land on SAVE_STEPS multiples wherever possible, so a boundary
#     writes a checkpoint the schedule was going to write anyway. The total checkpoint
#     count stays MAX_STEPS/SAVE_STEPS, which is what exp_expected_checkpoints predicts
#     and what the eval shard count is sized against.
#   * Checkpoints from every segment land in the SAME train/ directory (the job scripts
#     pass --no-run-subdir), so eval enumerates the whole run, not the last segment.

# Keep TRAIN_SEGMENTS and SEGMENT_STEPS from fighting, given who set which.
#
# $@ = the names the CALLER set explicitly, this invocation. Call before
# exp_plan_segments.
#
# They are two spellings of one decision, and exp_plan_segments gives SEGMENT_STEPS
# priority because it is the more specific of the two. That is right when both come
# from the same place and WRONG when they do not:
#
#   TRAIN_SEGMENTS=5 experiments/eddie/resume_experiment.sh <EXP_DIR>
#
# reads SEGMENT_STEPS=400 out of the original submission's experiment.env, hands it
# to exp_plan_segments, and gets 3 segments of 400 back -- silently ignoring the 5 the
# operator just asked for, and printing a plan that looks deliberate. Same shape as a
# preset that pins SEGMENT_STEPS meeting a caller who passes TRAIN_SEGMENTS.
#
# The rule: what you set now beats what was stored. Only the unmentioned one is cleared.
exp_segment_override() {
    local set_segments=0 set_size=0 n
    for n in "$@"; do
        if [ "$n" = "TRAIN_SEGMENTS" ]; then set_segments=1; fi
        if [ "$n" = "SEGMENT_STEPS" ]; then set_size=1; fi
    done
    if [ "$set_segments" = "1" ] && [ "$set_size" = "0" ] && [ -n "${SEGMENT_STEPS:-}" ]; then
        echo "[plan] TRAIN_SEGMENTS=${TRAIN_SEGMENTS} was given explicitly, so the stored" \
             "SEGMENT_STEPS=${SEGMENT_STEPS} is ignored and re-derived from it."
        SEGMENT_STEPS=""
        export SEGMENT_STEPS
    fi
    return 0
}

# Resolve TRAIN_SEGMENTS / SEGMENT_STEPS into a consistent pair, and export both.
#
# Callers may set either one. TRAIN_SEGMENTS is the natural knob ("split this into 3
# jobs"); SEGMENT_STEPS is the one to reach for when you know the per-step cost and are
# sizing a segment against a walltime directly. Whichever is given, the other is
# derived, and TRAIN_SEGMENTS is always recomputed from the final size -- so the number
# printed and recorded is the number of jobs that will actually be submitted, which is
# not always the number that was asked for (10 checkpoints cannot be split into 6
# whole-checkpoint groups; you get 5).
exp_plan_segments() {
    local total="${MAX_STEPS:-0}" save="${SAVE_STEPS:-0}"
    local want="${TRAIN_SEGMENTS:-1}" size="${SEGMENT_STEPS:-}"

    case "$total" in ''|*[!0-9]*) total=0 ;; esac
    case "$save"  in ''|*[!0-9]*) save=0 ;; esac
    case "$want"  in ''|*[!0-9]*) want=1 ;; esac
    [ "$want" -ge 1 ] || want=1

    if [ "$total" -le 0 ]; then
        TRAIN_SEGMENTS=1
        SEGMENT_STEPS=""
        export TRAIN_SEGMENTS SEGMENT_STEPS
        return 0
    fi

    if [ -n "$size" ]; then
        case "$size" in
            ''|*[!0-9]*)
                echo "ERROR: SEGMENT_STEPS must be a positive integer, got '$size'." >&2
                return 1 ;;
        esac
        [ "$size" -ge 1 ] || {
            echo "ERROR: SEGMENT_STEPS must be >= 1, got '$size'." >&2
            return 1
        }
        # An off-cadence size is allowed but not silent: the segment boundary saves a
        # checkpoint wherever it lands, so the run ends with MORE checkpoints than
        # MAX_STEPS/SAVE_STEPS -- while the eval shards were sized for that smaller
        # number, and the surplus would go unscored.
        if [ "$save" -gt 0 ] && [ $(( size % save )) -ne 0 ]; then
            echo "[plan] WARNING: SEGMENT_STEPS=$size is not a multiple of SAVE_STEPS=$save." >&2
            echo "[plan]          Every segment boundary saves a checkpoint wherever it lands," >&2
            echo "[plan]          so this run gets extra off-cadence ones (checkpoint-$size, ...)" >&2
            echo "[plan]          beyond the $(( (total + save - 1) / save )) the schedule plans," >&2
            echo "[plan]          and the eval shards are sized for the planned count -- the" >&2
            echo "[plan]          surplus goes unscored. Prefer a multiple of SAVE_STEPS, or" >&2
            echo "[plan]          raise EVAL_SHARD_SIZE to absorb them." >&2
        fi
    elif [ "$want" -le 1 ]; then
        size="$total"
    elif [ "$save" -gt 0 ] && [ "$save" -lt "$total" ]; then
        # Split the CHECKPOINTS evenly rather than the steps, so each boundary is a
        # checkpoint the save schedule was going to write anyway and the run's
        # checkpoint set is exactly what an unsegmented run would have produced.
        local ckpts per
        ckpts=$(( (total + save - 1) / save ))
        per=$(( (ckpts + want - 1) / want ))
        size=$(( per * save ))
    else
        size=$(( (total + want - 1) / want ))
    fi

    [ "$size" -ge 1 ] || size="$total"
    [ "$size" -le "$total" ] || size="$total"
    TRAIN_SEGMENTS=$(( (total + size - 1) / size ))
    SEGMENT_STEPS="$size"
    export TRAIN_SEGMENTS SEGMENT_STEPS
}

# Where segment $1 (1-based) stops. Always clamped to MAX_STEPS, so the last segment
# ends exactly on the horizon the LR schedule was built for rather than past it.
exp_segment_stop_at() {
    local idx="${1:-1}" size="${SEGMENT_STEPS:-0}" total="${MAX_STEPS:-0}" stop
    case "$idx$size$total" in ''|*[!0-9]*) printf '%s\n' "${MAX_STEPS:-0}"; return 0 ;; esac
    [ "$size" -ge 1 ] || { printf '%s\n' "$total"; return 0; }
    stop=$(( idx * size ))
    [ "$stop" -le "$total" ] || stop="$total"
    printf '%d\n' "$stop"
}

# What segment $1 passes to --resume-from-checkpoint.
#
# Segment 1 gets 'auto': an empty train/ is the normal case there, and starting at
# step 0 is correct. Segments 2+ get 'latest', which makes an empty train/ an ERROR --
# by then it means the previous job died before its first save, and silently
# restarting from step 0 would spend a whole walltime on work the operator is about to
# throw away.
#
# An explicitly-set RESUME_FROM applies to the FIRST segment only. It is how you point
# a chain at an existing checkpoint (resume_experiment.sh sets it); letting it reach
# segment 2 as well would make that segment resume the same checkpoint and redo
# segment 1's work.
exp_segment_resume_spec() {
    local idx="${1:-1}"
    if [ "$idx" = "1" ]; then
        printf '%s\n' "${RESUME_FROM:-auto}"
    else
        printf '%s\n' "latest"
    fi
}

# A stable W&B run id for this experiment, so every segment of a chain logs into ONE
# W&B run instead of N runs each starting at a different step.
#
# Derived from EXP_ID rather than random, because it has to be reproducible from the
# experiment directory alone -- resume_experiment.sh computes the same id months later
# without reading anything the first submission wrote. W&B ids may not contain
# '/ \ # ? % :' and are capped at 64 characters, so the name is flattened and, when
# too long, truncated with a checksum of the full EXP_ID appended to keep two
# long-tagged experiments from colliding on their shared prefix.
exp_wandb_chain_id() {
    local id="${EXP_ID:?exp_wandb_chain_id needs EXP_ID}" flat hash
    flat="$(printf '%s' "$id" | tr -c 'A-Za-z0-9._-' '_')"
    if [ "${#flat}" -le 60 ]; then
        printf '%s\n' "$flat"
        return 0
    fi
    hash="$(printf '%s' "$id" | cksum | cut -d' ' -f1)"
    printf '%s-%s\n' "${flat:0:50}" "$hash"
}

# Point every segment of a chain at one W&B run. No-op for a single-segment run and
# for a caller that set WB_ID itself.
#
# Without this each segment opens its own run, and the W&B UI shows a 1000-step
# experiment as three unrelated 400/400/200-step curves -- exactly the runs an
# ablation needs to compare, split three ways. Resume mode 'allow' rather than 'must'
# so the FIRST segment (which has no run to resume) still starts normally.
#
# $1 = "force" to set it up even for a single segment -- what resume_experiment.sh
# passes, because a resumed run is a second job continuing one W&B run whether or not
# the original was chained.
exp_wandb_chain_setup() {
    [ "${1:-}" = "force" ] || [ "${TRAIN_SEGMENTS:-1}" -gt 1 ] 2>/dev/null || return 0
    [ "${WB_ENABLE:-1}" = "1" ] || return 0
    [ "${WB_MODE:-auto}" != "disabled" ] || return 0
    [ -z "${WB_ID:-}" ] || return 0
    WB_ID="$(exp_wandb_chain_id)"
    WB_RESUME="${WB_RESUME:-allow}"
    export WB_ID WB_RESUME
    echo "[plan] W&B: all ${TRAIN_SEGMENTS} segments log into one run (id=$WB_ID, resume=$WB_RESUME)"
}

# The latest checkpoint that can actually be RESUMED from, as "<step><TAB><path>".
# Prints nothing when there is none.
#
# Defers to playpen/marshal/resume.py rather than reimplementing the rule in shell.
# The submitter and the training job have to agree on which directories count as
# resumable -- a submitter that is more permissive queues a chain the trainer will
# refuse, and one that is stricter reports "nothing to resume" for a run that could
# have continued. Two implementations of that rule would drift apart eventually; this
# one cannot. It also means the reported step is read from trainer_state.json, which
# is what training will actually continue from, rather than guessed from the
# directory name.
#
# Note this is STRICTER than exp_list_checkpoints, which only asks for an adapter:
# eval can score a checkpoint whose optimizer state was never written, resume cannot.
exp_last_resumable_checkpoint() {
    : "${TRAIN_BASE:?exp_last_resumable_checkpoint needs TRAIN_BASE (call exp_layout)}"
    local py="${REPO:-}/.venv/bin/python"
    [ -x "$py" ] || py="$(command -v python3 || command -v python)"
    [ -n "$py" ] || return 0
    PYTHONPATH="${REPO:-}${PYTHONPATH:+:$PYTHONPATH}" "$py" - "$TRAIN_BASE" <<'PY' 2>/dev/null
import contextlib
import io
import sys

# Importing `playpen` prints a large ASCII-art banner to stdout. Harmless in a job
# log; fatal here, because this function's stdout IS the answer -- the caller would
# assign the banner to RESUME_DONE_STEPS and every later `[ "$stop" -gt ... ]` would
# fail with "integer expression expected". Swallow anything the import prints so the
# only thing on stdout is the one line we mean to send.
with contextlib.redirect_stdout(io.StringIO()):
    from playpen.marshal import resume

path = resume.latest_checkpoint(sys.argv[1])
if path:
    print(f"{resume.resumed_global_step(path)}\t{path}")
PY
}

# Where a resumed run picks up. Sets, and exports:
#
#   RESUME_DONE_STEPS     steps already on disk (0 when starting over)
#   RESUME_CKPT           the checkpoint they are in ("" when there is none)
#   RESUME_FIRST_SEGMENT  the first segment of the CURRENT plan still to run
#
# Segment boundaries come from the plan rather than from "what is left divided by
# something": a run whose segment 2 of 3 died at step 500 of a 400/800/1000 plan
# resumes as segments 2 and 3, and segment 2 covers 500 -> 800 as it always would
# have. Recomputing boundaries from the remainder instead would silently move them,
# and with them the checkpoint set the eval shards were sized for.
#
# Returns 1 when every boundary is already behind what is on disk, i.e. training is
# finished and only the evaluation half can be re-run.
exp_resume_plan() {
    local line k stop
    line="$(exp_last_resumable_checkpoint)"
    RESUME_DONE_STEPS=0
    RESUME_CKPT=""
    if [ -n "$line" ]; then
        RESUME_DONE_STEPS="${line%%	*}"
        RESUME_CKPT="${line#*	}"
    fi
    # Belt and braces on a value that came from a subprocess: anything non-numeric
    # here would make every comparison below fail with "integer expression expected"
    # and silently pick segment 1, i.e. retrain the whole run.
    case "$RESUME_DONE_STEPS" in
        ''|*[!0-9]*)
            echo "[resume] WARNING: could not read a step count from $TRAIN_BASE" >&2
            echo "[resume]          (got: ${RESUME_DONE_STEPS:-<nothing>}). Treating the" >&2
            echo "[resume]          run as unstarted." >&2
            RESUME_DONE_STEPS=0
            RESUME_CKPT=""
            ;;
    esac
    RESUME_FIRST_SEGMENT=""
    for k in $(seq 1 "${TRAIN_SEGMENTS:-1}"); do
        stop="$(exp_segment_stop_at "$k")"
        if [ "$stop" -gt "$RESUME_DONE_STEPS" ]; then
            RESUME_FIRST_SEGMENT="$k"
            break
        fi
    done
    export RESUME_CKPT RESUME_DONE_STEPS RESUME_FIRST_SEGMENT
    [ -n "$RESUME_FIRST_SEGMENT" ]
}

# --- evaluation sharding -----------------------------------------------------
# Both evaluations are submitted as N concurrent jobs, each taking EVAL_SHARD_SIZE
# checkpoints. A job learns which slice is its own from $EVAL_SHARD (1-based),
# exported by the submitter through -v / --export rather than written into
# experiment.env -- one env file is shared by every shard, so the index cannot live
# in it.
#
# The slice is positional over exp_list_checkpoints' output, which is sorted by STEP
# NUMBER and filtered to complete adapters. Both properties matter: every shard
# enumerates the same list in the same order, so the slices tile it exactly, with no
# checkpoint evaluated twice and none dropped.

# How many checkpoints this experiment will have.
#
# Prefers what is actually on disk (the post-hoc re-run path: run_eval.sh and
# run_playpen_eval.sh submit after training has finished, so the answer is exact).
# Falls back to PREDICTING it from the schedule, because run_experiment.sh queues
# every job up front -- before training has written a single checkpoint -- and the
# shard count has to be decided then. train_selfplay.py saves every SAVE_STEPS up to
# MAX_STEPS, so that prediction is MAX_STEPS/SAVE_STEPS.
#
# Over-predicting is harmless: a shard whose slice is empty (training was killed
# early, so there are fewer checkpoints than planned) finds nothing to do and exits
# cleanly in seconds. Under-predicting is not, which is why the disk is preferred
# whenever it has something to say.
exp_expected_checkpoints() {
    local run_dir n
    run_dir="$(exp_find_run_dir 2>/dev/null || true)"
    if [ -n "$run_dir" ]; then
        n="$(exp_list_checkpoints "$run_dir" 2>/dev/null | wc -l | tr -d '[:space:]')"
        if [ "${n:-0}" -gt 0 ] 2>/dev/null; then printf '%d\n' "$n"; return 0; fi
    fi
    local steps="${MAX_STEPS:-0}" save="${SAVE_STEPS:-0}"
    case "$steps$save" in ''|*[!0-9]*) steps=0; save=0 ;; esac
    if [ "$save" -gt 0 ] && [ "$steps" -gt 0 ]; then
        printf '%d\n' $(( steps / save > 0 ? steps / save : 1 ))
    else
        printf '1\n'
    fi
}

# How many evaluation jobs are needed to cover $1 checkpoints.
exp_shard_count() {
    local n="${1:-1}" size="${EVAL_SHARD_SIZE:-5}"
    case "$size" in ''|*[!0-9]*) size=5 ;; esac
    [ "$size" -ge 1 ] || size=5
    case "$n" in ''|*[!0-9]*) n=1 ;; esac
    [ "$n" -ge 1 ] || n=1
    printf '%d\n' $(( (n + size - 1) / size ))
}

# Keep only $EVAL_SHARD's slice. Reads checkpoint paths on stdin, writes the kept
# ones to stdout in the same order.
#
# An unset or empty EVAL_SHARD passes everything through, so a hand re-run
# (`source experiment.env && experiments/eddie/eval.sh`) still scores the whole run
# exactly as it did before sharding existed.
exp_shard_filter() {
    local shard="${EVAL_SHARD:-}" size="${EVAL_SHARD_SIZE:-5}"
    case "$shard" in ''|*[!0-9]*) cat; return 0 ;; esac
    [ "$shard" -ge 1 ] || { cat; return 0; }
    case "$size" in ''|*[!0-9]*) size=5 ;; esac
    [ "$size" -ge 1 ] || size=5
    awk -v s="$shard" -v n="$size" 'NR > (s - 1) * n && NR <= s * n'
}

# True when this job owns the shared, checkpoint-independent baseline row.
#
# The untrained base model is scored ONCE per experiment, not once per shard: it is
# the same model and the same tasks whichever checkpoints a shard holds, and on the
# gameplay side it is hours of generation. Shard 1 does it; the others skip it and
# pick it up from the experiment directory when the summary is written. An unsharded
# run (no EVAL_SHARD) owns it too.
exp_owns_base_row() {
    case "${EVAL_SHARD:-}" in
        ''|1) return 0 ;;
        *)    return 1 ;;
    esac
}

# "shard 2 of 3" / "" -- for job banners.
exp_shard_label() {
    [ -n "${EVAL_SHARD:-}" ] || return 0
    printf 'shard %s of %s (%s checkpoints each)' \
        "$EVAL_SHARD" "${EVAL_SHARD_TOTAL:-?}" "${EVAL_SHARD_SIZE:-5}"
}

# --- job ids ------------------------------------------------------------------
# The manifest is written BEFORE anything is submitted -- so a run that dies in the
# queue still records what it was meant to be -- which is exactly why it cannot
# contain the job ids: they do not exist yet. The submitters call exp_record_jobs
# once everything is queued, and that adds them in a second pass. The reading of the
# ids afterwards is the point: `qstat`/`squeue` forget a job days before you come
# back to the directory, and a log file is only findable by the id in its name.

# sge | slurm | "" -- the scheduler this experiment is submitted to.
#
# EXP_CLUSTER is set by run_experiment.sh and stored in experiment.env, so every
# submitter has it, including the ones that only re-run an evaluation. The PATH probe
# is the fallback for experiment directories written before EXP_CLUSTER was recorded.
exp_scheduler() {
    case "${EXP_CLUSTER:-}" in
        eddie)    printf 'sge\n' ;;
        isambard) printf 'slurm\n' ;;
        *)
            if command -v sbatch >/dev/null 2>&1; then
                printf 'slurm\n'
            elif command -v qsub >/dev/null 2>&1; then
                printf 'sge\n'
            fi
            ;;
    esac
}

# Record the ids of the jobs just queued, in manifest.json and manifest.txt.
#
#   exp_record_jobs --train "${TRAIN_IDS[@]}" --train-total "$TRAIN_SEGMENTS" \
#                   --eval "${EVAL_IDS[@]}" --playpen "${PPEVAL_IDS[@]}" \
#                   --summary "$SUMMARY_ID" --shard-total "$EVAL_SHARD_TOTAL" \
#                   --env-file "$ENV_FILE"
#
# Every flag is optional (see experiments/lib/record_jobs.py --help): a submission
# that only queues gameplay shards passes only --playpen and --summary. Which script
# queued them is taken from the CALLER's own path, so it cannot go stale.
#
# NEVER FATAL. By the time this runs the jobs are already in the queue, so a manifest
# that could not be updated must not make a successful submission look like a failed
# one -- nor stop the script before it prints the ids and the cancel command to the
# terminal. Every failure path warns and returns 0.
exp_record_jobs() {
    local src py by
    [ -n "${EXP_DIR:-}" ] || {
        echo "[jobs] WARNING: EXP_DIR unset -- job ids not recorded in the manifest." >&2
        return 0
    }
    src="${BASH_SOURCE[1]:-$0}"
    by="$(basename "$(dirname "$src")")/$(basename "$src")"
    case "$by" in ./*|//*) by="$(basename "$src")" ;; esac
    py="${REPO:-}/.venv/bin/python"
    [ -x "$py" ] || py="$(command -v python3 || command -v python || true)"
    [ -n "$py" ] || {
        echo "[jobs] WARNING: no python found -- job ids not recorded in the manifest." >&2
        return 0
    }
    "$py" "$EXP_ROOT_DIR/lib/record_jobs.py" "$EXP_DIR" \
        --submitted-by "$by" \
        --cluster "${EXP_CLUSTER:-}" \
        --scheduler "$(exp_scheduler)" \
        "$@" \
    || echo "[jobs] WARNING: could not record the job ids in the manifest." \
            "The jobs ARE queued -- see the ids above." >&2
    return 0
}

# Activate a venv and verify it really is the one we asked for.
#
# Compares the venv ROOT DIRECTORIES resolved to physical paths (`pwd -P`).
#
# THE BUG THIS FIXES. On Isambard $PROJECTDIR is reached through a symlinked Lustre
# mount: /projects/<proj>/... and /lus/lfs1aip2/projects/<proj>/... are the SAME
# directory. A venv's bin/activate hardcodes the physical path it was created under,
# so after activation `command -v python` returns a different SPELLING of exactly the
# venv we asked for, and the old string comparison rejected it:
#
#   ERROR: wrong venv active: /lus/lfs1aip2/projects/u6ku/.../.venv/bin/python
#          expected           /projects/u6ku/.../.venv/bin/python
#
# -- a false alarm that refused to start a perfectly good job. Resolving both sides
# to physical paths makes the aliasing compare equal, while a .venv copied from
# another checkout still resolves elsewhere and is caught.
#
# Compare DIRECTORIES, not the interpreters. A venv's bin/python is normally a
# symlink to the shared base interpreter, so comparing the python files (e.g. with
# bash's -ef, same device+inode) reports every venv built on the same base Python as
# identical -- the check would then silently pass anything, which is worse than the
# false alarm it replaced. Verified: two independent venvs on one base Python do
# compare equal under -ef.
exp_activate_venv() {
    local venv="$1" label="${2:-venv}" got got_dir want_dir
    [ -f "$venv/bin/activate" ] || {
        echo "ERROR: no $label at $venv (no bin/activate)." >&2
        return 1
    }
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
    got="$(command -v python)"
    [ -n "$got" ] || {
        echo "ERROR: no python on PATH after activating $venv." >&2
        return 1
    }
    want_dir="$(cd "$venv" 2>/dev/null && pwd -P)" || want_dir="$venv"
    got_dir="$(cd "$(dirname "$got")/.." 2>/dev/null && pwd -P)" || got_dir="$(dirname "$got")/.."
    if [ "$got_dir" != "$want_dir" ]; then
        echo "ERROR: activated the wrong $label." >&2
        echo "       expected venv: $want_dir" >&2
        echo "       active venv:   $got_dir" >&2
        echo "       (paths above are resolved, so this is a genuine mismatch, not a" >&2
        echo "        symlinked mount.) The venv was probably copied from another" >&2
        echo "       checkout -- rebuild it in place." >&2
        return 1
    fi
    echo "[venv] $label: $got"
}

# Put TMPDIR -- and every torch/triton/vLLM cache that derives from it -- on project
# storage, inside this experiment's own directory.
#
# THE BUG THIS FIXES. Jobs inherit TMPDIR=/local/user/<uid> (sbatch --export=ALL
# copies the login node's environment). On Isambard that path is worse than merely
# wrong: it is TRANSIENT. Observed across two attempts on the same path:
#
#   PermissionError:   [Errno 13] Permission denied: '/local/user/1483806040'
#     -- the directory did not exist and could not be created
#   FileNotFoundError: [Errno  2] No such file or directory:
#                      '/local/user/1483806040/marshal-5789516/tmpids8ymd0'
#     -- our own mkdir HAD succeeded; the directory then vanished before python ran
#
# Both crash during `import torch`, before a line of our code runs (inductor's
# cache_dir() does os.makedirs; torch.distributed's instantiator opens a
# TemporaryDirectory). Both the training and the eval job hit it.
#
# So we do NOT probe /local, and do NOT trust the inherited TMPDIR: creating a
# directory there and checking it is writable proves nothing, because it can
# disappear underneath us a second later. $EXP_DIR is on Lustre project storage,
# already exists (the submitter made it), and lives as long as the experiment.
# Slower than node-local SSD for compile caches, which is a fair price for a job
# that actually starts.
#
# Kept under $EXP_DIR/tmp rather than a shared directory so concurrent jobs cannot
# fight over one torch.compile / vLLM cache, and so anything left behind is obvious
# and belongs to a known experiment.
#
# PER JOB, not per experiment. The trap at the bottom `rm -rf`s this directory on
# exit, and since evaluation is now sharded into jobs that run CONCURRENTLY within
# one experiment, a single $EXP_DIR/tmp would mean the first shard to finish deleting
# the inductor/triton/vLLM caches out from under its siblings mid-run. Naming it
# after the scheduler's job id keeps the cleanup to what this job created; $$ is the
# fallback for a hand re-run outside the queue.
exp_setup_tmpdir() {
    local root="${EXP_DIR:?exp_setup_tmpdir needs EXP_DIR}/tmp"
    local base="$root/${SLURM_JOB_ID:-${JOB_ID:-$$}}"
    if ! mkdir -p "$base" 2>/dev/null || [ ! -w "$base" ]; then
        echo "ERROR: cannot create a temp directory at $base" >&2
        echo "       \$EXP_DIR must be writable -- check project storage quota." >&2
        return 1
    fi

    export TMPDIR="$base"
    export TMP="$base" TEMP="$base"          # some libraries read these instead
    export TORCHINDUCTOR_CACHE_DIR="$base/inductor"
    export TRITON_CACHE_DIR="$base/triton"
    export VLLM_CACHE_ROOT="$base/vllm"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

    # Remove it on the way out so compile caches don't bloat the experiment directory
    # or get rsync'd home. The path is baked into the trap NOW rather than read from
    # $TMPDIR when it fires, so a later reassignment cannot redirect the rm. A SIGKILL
    # (walltime) skips the trap and leaves $EXP_DIR/tmp behind -- harmless, and
    # obviously attributable to that experiment. The parent is only removed when it
    # is empty, i.e. when this was the last shard still holding a temp directory.
    trap "rm -rf -- '$base'; rmdir -- '$root' 2>/dev/null || true" EXIT
    echo "[tmp] TMPDIR=$TMPDIR (torch/triton/vLLM caches inside it, per job)"
}

# Pin torch.distributed / vLLM to a genuinely free port, and say so in the log.
#
# THE BUG THIS FIXES (seen on both clusters as, e.g.,
#   DistNetworkError: ... port: 23456 ... EADDRINUSE ... address already in use
# raised from vLLM's init_distributed_environment during LLM() construction):
#
# TRL picks the port via ensure_master_addr_port -> _find_free_port
# (trl/trainer/utils.py), which walks a FIXED candidate list -- (29500, 23456,
# 12355, 12345) -- and only falls back to an OS-assigned port if all four are busy.
# So 23456 comes from TRL itself, NOT from the cluster's module environment. On a
# shared GPU node (Eddie hands out one H200 of eight, other users' jobs run
# alongside) every concurrent TRL job walks that same list in the same order, and
# _is_port_free binds-then-closes, so two jobs can both see a port free and race to
# claim it. Collisions are systematic, not unlucky.
#
# `MASTER_PORT=0` does NOT avoid this: ensure_master_addr_port treats "0" (and
# "auto", and unset) as "choose for me" and runs the same candidate walk. Verified
# against the installed trl 0.29.1 -- with 29500 and 23456 held, MASTER_PORT=0
# resolved to 12355, i.e. still a fixed candidate.
#
# A concrete NON-ZERO MASTER_PORT is taken verbatim, so asking the OS for an
# ephemeral port here bypasses the candidate list entirely. The probe->export gap
# leaves a race in principle, but the ephemeral range is ~28k ports wide and the
# kernel avoids recently-used ones, so it is negligible next to a shared 4-entry list.
#
# Requires the training venv to be active (it uses `python`). Returns non-zero and
# leaves MASTER_PORT alone if the probe fails, so the job still starts -- on TRL's
# old, collision-prone path, which the warning says out loud.
exp_export_master_port() {
    local port
    port="$(python -c 'import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()' 2>/dev/null)"
    case "$port" in
        ''|*[!0-9]*)
            echo "[port] WARNING: could not probe a free port (is the venv active?)." >&2
            echo "[port]          Falling back to TRL's fixed candidate list" >&2
            echo "[port]          (29500, 23456, 12355, 12345) -- the one that collides" >&2
            echo "[port]          on a shared node. Re-run if this job dies with EADDRINUSE." >&2
            return 1 ;;
    esac
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT="$port"
    echo "[port] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT" \
         "(OS-assigned; bypasses TRL's fixed candidate list)"
}

# True if a directory already holds an lm-eval result file (checked recursively,
# because lm-eval nests output under a sanitized model-name subdir).
exp_have_results() {
    [ -n "$(find "$1" -name 'results*.json' -print -quit 2>/dev/null)" ]
}

# A stable cache key for a base-model evaluation. Base results depend ONLY on the
# things that change the model's answers -- the model, the task set, any --limit, and
# any extra lm-eval flags (e.g. --apply_chat_template). They do NOT depend on the
# checkpoint or the batch size, so the same combination is identical across every
# experiment. A readable slug (model/tasks) is joined with a short checksum of the
# full key so that, say, a --limit 5 smoke run never collides with a full run.
exp_base_cache_key() {
    local model="$1" tasks="$2" limit="$3" extra="$4"
    local slug hash
    slug="$(printf '%s__%s' "$model" "$tasks" | tr -c 'A-Za-z0-9._-' '_')"
    hash="$(printf '%s|%s|limit=%s|extra=%s' "$model" "$tasks" "$limit" "$extra" \
            | cksum | cut -d' ' -f1)"
    printf '%s__%s\n' "$slug" "$hash"
}

# Evaluate the untrained base model, reusing a shared cache so it runs at most ONCE
# per (model, tasks, limit, extra) instead of once per experiment.
#
# Usage: exp_eval_base <model> <dest_dir> <cache_root> -- <lm-eval common args...>
#
# On a cache hit it copies the stored result into <dest_dir> and returns without
# touching the GPU. On a miss it runs lm-eval into a staging dir, publishes that to
# the cache atomically (mv), then copies it into <dest_dir>. The result is COPIED,
# not symlinked, so the experiment directory stays self-contained for rsync.
#
# `lm-eval` is only referenced here, and this function is only ever called from the
# eval jobs, so sourcing experiment.sh where lm-eval is absent (login node, train
# job) is fine.
exp_eval_base() {
    local model="$1" dest="$2" cache_root="$3"; shift 3
    [ "${1:-}" = "--" ] && shift
    local key dir stage
    key="$(exp_base_cache_key "$model" "${EVAL_TASKS:-}" "${EVAL_LIMIT:-}" "${EVAL_EXTRA:-}")"
    dir="$cache_root/$key"
    mkdir -p "$dest" "$cache_root"

    if exp_have_results "$dir"; then
        echo "[eval] base model: reusing cached result (no GPU run)"
        echo "       cache: $dir"
        cp -r "$dir/." "$dest/"
        return 0
    fi

    echo "[eval] base model: $model -- not cached, running once then caching"
    stage="$(mktemp -d "$cache_root/.staging.XXXXXX")" || return 1
    if lm-eval --model hf --model_args "pretrained=${model},dtype=bfloat16" \
           "$@" --output_path "$stage"; then
        printf 'model=%s\ntasks=%s\nlimit=%s\nextra=%s\ncreated=%s\ncluster=%s\n' \
            "$model" "${EVAL_TASKS:-}" "${EVAL_LIMIT:-}" "${EVAL_EXTRA:-}" \
            "$(date --iso-8601=seconds)" "${EXP_CLUSTER:-?}" > "$stage/CACHE_INFO.txt"
        # Publish atomically where possible. A half-written cache dir is never seen
        # as a hit, because exp_have_results checks for a results*.json, which lm-eval
        # only writes on success -- so a concurrent miss re-runs rather than reads junk.
        rm -rf "$dir"
        mkdir -p "$(dirname "$dir")"
        mv "$stage" "$dir" 2>/dev/null || { mkdir -p "$dir"; cp -r "$stage/." "$dir/"; rm -rf "$stage"; }
        cp -r "$dir/." "$dest/"
        return 0
    fi
    rm -rf "$stage"
    return 1
}

# Printed at the top of every job log. Deliberately verbose: a log file read on its
# own -- tailed from the wrong window, pasted into a message, found weeks later --
# must say which experiment it belongs to and with what settings, without needing
# the surrounding directory for context.
#
# $1 is the phase ("train" / "eval").
exp_banner() {
    local phase="${1:-job}"
    echo "=================================================================="
    echo "  MARSHAL experiment -- ${phase}"
    echo "=================================================================="
    echo "experiment  = ${EXP_ID:-?}"
    echo "tag         = ${EXP_TAG:-none}"
    echo "dir         = ${EXP_DIR:-?}"
    echo "game        = ${GAME:-?}"
    echo "model       = ${MODEL:-?}"
    if [ "$phase" = "train" ]; then
        echo "steps       = ${MAX_STEPS:-?} (checkpoint every ${SAVE_STEPS:-?})"
        if [ "${TRAIN_SEGMENTS:-1}" -gt 1 ] 2>/dev/null; then
            echo "segment     = ${TRAIN_SEGMENT:-?} of ${TRAIN_SEGMENTS} "\
"(this job trains up to step ${SEGMENT_STOP_AT:-?} of ${MAX_STEPS:-?})"
            echo "resume      = ${SEGMENT_RESUME:-none}"
        elif [ -n "${RESUME_FROM:-}" ]; then
            echo "resume      = ${RESUME_FROM}"
        fi
        echo "batch       = ${PER_DEVICE_BATCH:-?} x ${GRAD_ACCUM:-?} accum, "\
"${NUM_GENERATIONS:-?} generations"
        echo "max_compl   = ${MAX_COMPLETION_LENGTH:-?} tokens/turn"
        echo "len_penalty = per_token=${LP_PER_TOKEN:-<yaml>} budget=${LP_BUDGET:-<yaml>}"\
"${LP_MAX_LEN:+  [LP_MAX_LEN=$LP_MAX_LEN IGNORED]}${LP_COEF:+  [LP_COEF=$LP_COEF IGNORED]}"
        echo "turn_rewards= ${TR_ENABLE:-<yaml>} scale=${TR_SCALE:-<yaml>} "\
"budget=${TR_BUDGET:-<yaml>} source=${TR_SOURCE:-<yaml>}"
        if [ "${WB_ENABLE:-1}" = "1" ] && [ "${WB_MODE:-auto}" != "disabled" ]; then
            echo "wandb       = ${WB_PROJECT:-playpen-marshal}${WB_ENTITY:+ ($WB_ENTITY)} "\
"mode=${WB_MODE:-auto} run=${EXP_ID:-?}"
        else
            echo "wandb       = off"
        fi
        echo "extra_args  = ${EXTRA_TRAIN_ARGS:-none}"
    else
        echo "tasks       = ${EVAL_TASKS:-?} (batch ${EVAL_BATCH:-?})"
        echo "eval_base   = ${EVAL_BASE:-?}"
        [ -n "${EVAL_SHARD:-}" ] && echo "shard       = $(exp_shard_label)"
    fi
    echo "cluster     = ${EXP_CLUSTER:-?}"
    echo "host        = $(hostname)"
    echo "job         = ${JOB_ID:-${SLURM_JOB_ID:-interactive}}"
    echo "start       = $(date --iso-8601=seconds)"
    echo "gpu         = $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none visible')"
    echo "=================================================================="
}

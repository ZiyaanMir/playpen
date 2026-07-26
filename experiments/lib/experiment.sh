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
    # `${!v+set}` tests for *set*, not non-empty, so an intentional `LP_COEF=` (meaning
    # "leave this to the YAML") is preserved rather than treated as absent.
    local _override_names=() _override_vals=() _v _i
    for _v in MODEL MARSHAL_CONFIG \
              NUM_GENERATIONS PER_DEVICE_BATCH GRAD_ACCUM MAX_STEPS SAVE_STEPS \
              LEARNING_RATE KL_BETA MAX_COMPLETION_LENGTH MAX_TURNS GRAD_CKPT \
              VLLM_UTIL VLLM_MAX_MODEL_LEN LP_MAX_LEN LP_COEF \
              EVAL_TASKS EVAL_BATCH EVAL_BASE EVAL_LIMIT EVAL_EXTRA \
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

    # Length penalty. Empty LP_MAX_LEN/LP_COEF => don't pass the flag, so the YAML
    # value stands. See experiments/README.md for how these were calibrated.
    LP_MAX_LEN="${LP_MAX_LEN:-}"
    LP_COEF="${LP_COEF:-}"

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

    EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
    EXP_TAG="${EXP_TAG:-}"

    # Export everything: write_manifest.py reads these from the environment, and the
    # job scripts re-source this file in a fresh shell. Without the export the
    # manifest silently records an empty training section.
    export GAME MODEL MARSHAL_CONFIG \
           NUM_GENERATIONS PER_DEVICE_BATCH GRAD_ACCUM MAX_STEPS SAVE_STEPS \
           LEARNING_RATE KL_BETA MAX_COMPLETION_LENGTH MAX_TURNS GRAD_CKPT \
           VLLM_UTIL VLLM_MAX_MODEL_LEN LP_MAX_LEN LP_COEF \
           EVAL_TASKS EVAL_BATCH EVAL_BASE EVAL_LIMIT EVAL_EXTRA \
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
        echo "batch       = ${PER_DEVICE_BATCH:-?} x ${GRAD_ACCUM:-?} accum, "\
"${NUM_GENERATIONS:-?} generations"
        echo "max_compl   = ${MAX_COMPLETION_LENGTH:-?} tokens/turn"
        echo "len_penalty = max_len=${LP_MAX_LEN:-<yaml>} coef=${LP_COEF:-<yaml>}"
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
    fi
    echo "cluster     = ${EXP_CLUSTER:-?}"
    echo "host        = $(hostname)"
    echo "job         = ${JOB_ID:-${SLURM_JOB_ID:-interactive}}"
    echo "start       = $(date --iso-8601=seconds)"
    echo "gpu         = $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none visible')"
    echo "=================================================================="
}

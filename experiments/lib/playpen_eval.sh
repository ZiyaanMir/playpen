#!/bin/bash
# Play every checkpoint through the clembench games and score it. SOURCED by
# experiments/{eddie,isambard}/playpen_eval.sh, never run directly.
#
# This is the THIRD job of an experiment, after training and after lm-eval:
#
#   train  ->  eval (lm-eval: logiglue/logicbench, reasoning transfer)
#          ->  playpen_eval (THIS: clembench gameplay, the clemscore)
#
# The two evaluations answer different questions and are deliberately separate.
# lm-eval measures whether self-play changed the model's static reasoning; this one
# measures whether it plays the games better -- including the 13 games it was never
# trained on, which is where "did it learn to play, or did it learn this one game"
# is actually decided.
#
# Unlike lm-eval, this runs in the repo's OWN .venv (clemcore + playpen + peft are
# all there), so there is no second environment to build and the body is identical
# on both clusters -- hence one library rather than two copies. The cluster scripts
# supply only the scheduler directives and their own module/venv/cache setup.
#
# Output, all inside the experiment directory:
#
#   $EXP_DIR/playpen-eval/
#     base/                     untrained model
#       <name>.val.json           clemscore / statscore
#       clem/results.csv          per-game % Played and Quality Score
#       clem/<name>/<game>/...    interactions, transcripts, per-episode scores
#     checkpoint-50/  ...       one per adapter
#     .work/<name>/             the generated registries this row was run with
#   $EXP_DIR/PLAYPEN_RESULTS.md   the table across all of them
#   $EXP_DIR/playpen_results.tsv  the same numbers for pandas
#
# Requires from the environment (the eval job sources experiment.env first):
#   EXP_DIR REPO MODEL MARSHAL_RUNS, and pp_layout having been called.

# --- refuse execution --------------------------------------------------------
(return 0 2>/dev/null) || {
    echo "ERROR: source this file, don't execute it: source ${0}" >&2
    exit 1
}

# --- settings + layout -------------------------------------------------------
# Defaults live here rather than in the presets because they are game-independent:
# the point of this job is to play ALL the games, whichever one was trained on.
pp_layout() {
    : "${EXP_DIR:?pp_layout needs EXP_DIR}"
    PPEVAL_DIR="$EXP_DIR/playpen-eval"
    PPEVAL_WORK="$PPEVAL_DIR/.work"

    # clem   = the 14 interactive clembench games ('benchmark 2.0') -- the playpen
    #          games, and what clemscore means.
    # static = the 5 multiple-choice benchmarks ('benchmark static_1.0'), scored as
    #          statscore. Off by default: they need the separate 'instances-static'
    #          split of colab-potsdam/playpen-data, and lm-eval already covers this
    #          kind of measurement in the previous job.
    # all    = both.
    PPEVAL_SUITE="${PPEVAL_SUITE:-clem}"
    # Explicit game list (comma- or space-separated) overriding the suite. Use it to
    # add a game the suite omits -- notably `dond`, which is trained here but is NOT
    # part of benchmark 2.0, so a dond run is otherwise scored only on games it never
    # played. PPEVAL_GAMES=dond,guesswhat,taboo
    PPEVAL_GAMES="${PPEVAL_GAMES:-}"
    PPEVAL_BASE="${PPEVAL_BASE:-1}"            # also play the untrained model
    # Only shard 1 plays the untrained baseline. It does not depend on the checkpoint,
    # and it is the single most expensive row in the job -- a full pass over 14 games.
    # (It is cached across experiments too, so after the first run on a model this is a
    # copy rather than gameplay; the shard rule is what stops N concurrent shards racing
    # to populate that cache on the first run.) Decided HERE, before pp_banner runs, so
    # the banner's "base row" line agrees with what the job actually does.
    exp_owns_base_row || PPEVAL_BASE=0
    PPEVAL_CKPTS="${PPEVAL_CKPTS:-}"           # "" = every checkpoint; "last"; "100,200"
    # playpen eval's own defaults, which is what the LEADERBOARD numbers were produced
    # with. Deliberately NOT the training run's max_completion_length: a checkpoint
    # scored with a different generation budget than the baseline is not comparable to
    # it, and the baseline here is the untrained model under these same settings.
    PPEVAL_MAX_TOKENS="${PPEVAL_MAX_TOKENS:-300}"
    PPEVAL_TEMPERATURE="${PPEVAL_TEMPERATURE:-0.0}"
    # Per-row wall-clock ceiling, e.g. 3h. A single wedged game must not eat the whole
    # job and leave every later checkpoint unscored. Empty = no timeout.
    PPEVAL_TIMEOUT="${PPEVAL_TIMEOUT:-}"
    PPEVAL_CACHE="${PPEVAL_CACHE:-${MARSHAL_RUNS:-$EXP_DIR}/_playpen_base_cache}"

    export PPEVAL_DIR PPEVAL_WORK PPEVAL_SUITE PPEVAL_GAMES PPEVAL_BASE PPEVAL_CKPTS \
           PPEVAL_MAX_TOKENS PPEVAL_TEMPERATURE PPEVAL_TIMEOUT PPEVAL_CACHE
}

# The `--suite` / `-g` arguments for one playpen eval invocation, as $PP_SELECT.
# An array, not a string: game names are separate argv entries and word-splitting a
# printed string would break the moment a selector contained a space (a GameSpec
# JSON selector does).
pp_select_args() {
    PP_SELECT=()
    if [ -n "${PPEVAL_GAMES:-}" ]; then
        local g
        for g in ${PPEVAL_GAMES//,/ }; do
            [ -n "$g" ] && PP_SELECT+=( "$g" )
        done
        # -g takes nargs="+", so every game follows one flag.
        PP_SELECT=( -g "${PP_SELECT[@]}" )
    else
        PP_SELECT=( --suite "${PPEVAL_SUITE:-clem}" )
    fi
}

# True if a directory already holds a finished playpen eval.
#
# Keyed on clemeval's results.csv rather than the <model>.val.json, because the
# val.json is written even when every episode aborted and the clemscore came out
# NaN -- and because results.csv is what the summary is actually built from. A run
# that died mid-gameplay has neither.
pp_have_results() {
    [ -n "$(find "$1" -name 'results.csv' -print -quit 2>/dev/null)" ]
}

# A path-safe model name for one row. It becomes a directory under clem/ and appears
# in every results table, so it has to be both unique within the experiment and
# readable on its own: Qwen3-4B-checkpoint-150, not "checkpoint-150".
pp_model_name() {
    local base="$1" row="$2"
    # printf WITHOUT a newline, then add one: tr would otherwise translate the
    # trailing newline itself into '_', giving every model name a spurious suffix
    # that then appears in every results directory and table.
    printf '%s\n' \
        "$(printf '%s-%s' "$(basename "$base")" "$row" | tr -c 'A-Za-z0-9._-' '_')"
}

# Cache key for a base-model row. The untrained model's score depends only on what
# it played and how it generated -- never on the checkpoint -- so it is identical
# across every experiment on the same model. Same reasoning as exp_base_cache_key.
pp_base_cache_key() {
    local model="$1" slug hash
    slug="$(printf '%s__%s' "$(basename "$model")" \
            "${PPEVAL_GAMES:-suite-${PPEVAL_SUITE}}" | tr -c 'A-Za-z0-9._-' '_')"
    hash="$(printf '%s|suite=%s|games=%s|maxtok=%s|temp=%s' \
            "$model" "${PPEVAL_SUITE:-}" "${PPEVAL_GAMES:-}" \
            "${PPEVAL_MAX_TOKENS:-}" "${PPEVAL_TEMPERATURE:-}" | cksum | cut -d' ' -f1)"
    printf '%s__%s\n' "$slug" "$hash"
}

# Play one row: pp_eval_one <row-name> <base-model> <adapter-or-empty> <dest-dir>
#
# Runs from a per-row working directory holding the generated registries, so the
# checkpoint is addressable by name (see lib/playpen_registry.py) and nothing
# CWD-relative -- clembench.log included -- escapes the experiment directory.
pp_eval_one() {
    local row="$1" base_model="$2" adapter="$3" dest="$4"
    local work name rc
    work="$PPEVAL_WORK/$row"
    name="$(pp_model_name "$base_model" "$row")"

    mkdir -p "$work" "$dest"
    python "$REPO/experiments/lib/playpen_registry.py" \
        --work-dir "$work" --model-name "$name" \
        --base-model "$base_model" --adapter "$adapter" --repo "$REPO" || return 1

    pp_select_args
    local cmd=( python -m playpen.cli eval "$name" "${PP_SELECT[@]}"
                -r "$dest"
                -T "${PPEVAL_TEMPERATURE:-0.0}" -L "${PPEVAL_MAX_TOKENS:-300}" )
    [ -n "${PPEVAL_TIMEOUT:-}" ] && cmd=( timeout "$PPEVAL_TIMEOUT" "${cmd[@]}" )

    echo "[ppeval] cd $work && ${cmd[*]}"
    ( cd "$work" && "${cmd[@]}" ) && rc=0 || rc=$?
    if [ "${rc:-0}" -ne 0 ]; then
        if [ -n "${PPEVAL_TIMEOUT:-}" ] && [ "$rc" -eq 124 ]; then
            echo "[ppeval] TIMED OUT after $PPEVAL_TIMEOUT: $row" >&2
        fi
        # Gameplay is the expensive part; scoring is seconds of CPU. If the episodes
        # were played but the aggregation died, rebuild it here rather than throwing
        # hours of GPU away.
        #
        # THE FAILURE THIS RECOVERS FROM. clembench's privateshared imports sklearn.
        # With that package missing, clemcore skips the game during play (13 of 14
        # still run) but `clemcore.cli.score` collects the error and calls
        # **sys.exit(1)** -- no traceback, no message -- killing `playpen eval` before
        # clemeval ever runs. One absent package therefore cost the aggregation for
        # every game and every checkpoint of the 2026-07-30 Isambard run, ~3.5 h of
        # GH200 time that had already produced perfectly good interaction files.
        if pp_has_gameplay "$dest"; then
            echo "[ppeval] $row: gameplay is on disk -- rebuilding scores from it" >&2
            if python "$REPO/experiments/lib/rescore_playpen_eval.py" \
                   --row "$row" "$EXP_DIR" >&2; then
                if pp_have_results "$dest"; then
                    echo "[ppeval] $row: RECOVERED (scored from existing gameplay)" >&2
                    return 0
                fi
            fi
            echo "[ppeval] $row: recovery failed too -- gameplay kept at $dest" >&2
        fi
        return "$rc"
    fi
    return 0
}

# True if a row directory holds played episodes, whether or not they were ever scored.
# Distinguishes "the GPU work is done, only aggregation failed" (recoverable, cheap)
# from "nothing ran" (needs the GPU again).
pp_has_gameplay() {
    [ -n "$(find "$1" -name 'interactions.json' -print -quit 2>/dev/null)" ]
}

# The untrained base model, reusing a shared cache so it is played at most once per
# (model, games, generation settings) rather than once per experiment -- a full pass
# over 14 games is hours of GPU, and it is the same hours every time.
#
# Copied into the experiment rather than symlinked, so the directory stays
# self-contained for rsync. Published to the cache only on success (staged then
# moved), so a failed base run leaves no entry and the next experiment retries
# instead of reading a half-written one.
pp_eval_base_cached() {
    local base_model="$1" dest="$2" cache_root="$3"
    local key dir stage
    key="$(pp_base_cache_key "$base_model")"
    dir="$cache_root/$key"
    mkdir -p "$dest" "$cache_root"

    if pp_have_results "$dir"; then
        echo "[ppeval] base model: reusing cached result (no gameplay)"
        echo "         cache: $dir"
        cp -r "$dir/." "$dest/"
        return 0
    fi

    echo "[ppeval] base model: $base_model -- not cached, playing once then caching"
    # Play into $dest, NOT into a staging directory that gets deleted on failure.
    #
    # THE BUG THIS FIXES. The base row used to be played into a temp dir under the
    # cache and `rm -rf`'d if anything went wrong -- so when the 2026-07-30 run hit the
    # scoring crash, the checkpoints' gameplay survived under playpen-eval/ and could be
    # rescored, but the base row's 16 minutes of GH200 was deleted outright and the
    # experiment was left with no baseline at all. Playing into $dest means a failure
    # leaves the episodes inside the experiment where pp_eval_one's recovery (and
    # rescore_playpen_eval.py) can reach them, exactly like every other row.
    pp_eval_one base "$base_model" "" "$dest" || return 1

    printf 'model=%s\nsuite=%s\ngames=%s\nmax_tokens=%s\ntemperature=%s\ncreated=%s\ncluster=%s\n' \
        "$base_model" "${PPEVAL_SUITE:-}" "${PPEVAL_GAMES:-}" \
        "${PPEVAL_MAX_TOKENS:-}" "${PPEVAL_TEMPERATURE:-}" \
        "$(date --iso-8601=seconds)" "${EXP_CLUSTER:-?}" > "$dest/CACHE_INFO.txt"

    # Publish to the cache via a staging copy, so a concurrent job never sees a
    # half-written directory as a hit. Failing to cache is not failing the row -- the
    # result is already in the experiment.
    if stage="$(mktemp -d "$cache_root/.staging.XXXXXX")"; then
        if cp -r "$dest/." "$stage/"; then
            rm -rf "$dir"
            mkdir -p "$(dirname "$dir")"
            mv "$stage" "$dir" 2>/dev/null || rm -rf "$stage"
        else
            rm -rf "$stage"
        fi
    fi
    return 0
}

# Narrow the checkpoint list to PPEVAL_CKPTS. Reads paths on stdin, writes the kept
# ones to stdout, preserving order.
#   ""            every checkpoint
#   last          only the final one (a quick "did it end up better" pass)
#   100,200       exactly those steps
pp_filter_checkpoints() {
    local sel="${PPEVAL_CKPTS:-}" line step
    if [ -z "$sel" ]; then cat; return 0; fi
    if [ "$sel" = "last" ]; then tail -1; return 0; fi
    while IFS= read -r line; do
        step="${line##*/checkpoint-}"
        case ",${sel//[[:space:]]/}," in
            *",$step,"*) printf '%s\n' "$line" ;;
        esac
    done
}

# Resolve the base weights for an adapter from the adapter itself.
#
# A mismatched base applies the LoRA maths to the wrong weights and produces
# silently wrong scores with no error, so the adapter -- which cannot be wrong about
# what it was trained on -- wins over the experiment's $MODEL.
pp_adapter_base() {
    python -c "import json,sys;print(json.load(open(sys.argv[1]))['base_model_name_or_path'])" \
        "$1/adapter_config.json"
}

# The whole job body: base row (cached), then every selected checkpoint, then the
# table. Returns the number of failures, and writes the table either way so one
# checkpoint dying does not cost the rest.
pp_run_all() {
    : "${EXP_DIR:?pp_run_all needs EXP_DIR}"
    : "${REPO:?pp_run_all needs REPO}"
    mkdir -p "$PPEVAL_DIR" "$PPEVAL_WORK"

    local run_dir
    run_dir="$(exp_find_run_dir || true)"
    if [ -z "$run_dir" ]; then
        echo "[ppeval] no training run directory under $TRAIN_BASE." >&2
        echo "[ppeval] training produced nothing to evaluate -- check $LOG_DIR/train_*.err" >&2
        return 0
    fi
    echo "[ppeval] run dir = $run_dir"

    # PPEVAL_CKPTS narrows WHICH checkpoints this experiment plays at all; the shard
    # filter then splits what survives across the concurrent gameplay jobs. Order
    # matters -- sharding the unfiltered list would leave `PPEVAL_CKPTS=last` in one
    # arbitrary shard and the rest idle.
    local checkpoints=() selected=()
    mapfile -t selected < <(exp_list_checkpoints "$run_dir" | pp_filter_checkpoints)
    mapfile -t checkpoints < <(exp_list_checkpoints "$run_dir" | pp_filter_checkpoints \
                                   | exp_shard_filter)
    if [ -n "${EVAL_SHARD:-}" ]; then
        echo "[ppeval] $(exp_shard_label): ${#checkpoints[@]} of ${#selected[@]}" \
             "checkpoint(s)${PPEVAL_CKPTS:+ (PPEVAL_CKPTS=$PPEVAL_CKPTS)}"
    else
        echo "[ppeval] ${#checkpoints[@]} checkpoint(s) to play${PPEVAL_CKPTS:+ (PPEVAL_CKPTS=$PPEVAL_CKPTS)}"
    fi

    # pp_layout already set PPEVAL_BASE=0 for every shard but the first; say so, so a
    # log read on its own does not look like the baseline was silently dropped.
    exp_owns_base_row || echo "[ppeval] base row skipped -- shard 1 owns it."

    pp_select_args
    echo "[ppeval] games   : ${PP_SELECT[*]}"
    echo "[ppeval] gen args: max_tokens=${PPEVAL_MAX_TOKENS} temperature=${PPEVAL_TEMPERATURE}"

    local failed=0

    if [ "${PPEVAL_BASE:-1}" = "1" ]; then
        echo "=== [ppeval] base model ==="
        pp_eval_base_cached "$MODEL" "$PPEVAL_DIR/base" "$PPEVAL_CACHE" \
            || { echo "[ppeval] FAILED: base model" >&2; failed=$((failed + 1)); }
    fi

    local ck name ck_base
    for ck in "${checkpoints[@]}"; do
        name="$(basename "$ck")"
        echo "=== [ppeval] $name ==="
        ck_base="$(pp_adapter_base "$ck")" || {
            echo "[ppeval] FAILED: $name (unreadable adapter_config.json)" >&2
            failed=$((failed + 1)); continue; }
        if [ "$ck_base" != "$MODEL" ]; then
            echo "[ppeval] NOTE: adapter names base '$ck_base' (experiment MODEL is" \
                 "'$MODEL'); using the adapter's." >&2
        fi
        pp_eval_one "$name" "$ck_base" "$ck" "$PPEVAL_DIR/$name" \
            || { echo "[ppeval] FAILED: $name" >&2; failed=$((failed + 1)); }
    done

    python "$REPO/experiments/lib/summarize_playpen_eval.py" "$EXP_DIR" || true

    echo ""
    echo "[ppeval] results   : $EXP_DIR/PLAYPEN_RESULTS.md"
    echo "[ppeval] full data : $PPEVAL_DIR/"
    [ "$failed" -gt 0 ] && echo "[ppeval] $failed evaluation(s) FAILED -- see above." >&2
    return "$failed"
}

# Fail fast, before any model is loaded, if the validation split cannot be read.
#
# `playpen eval` plays only the instances in colab-potsdam/playpen-data's validation
# split (that is what makes it the playpen benchmark rather than all of clembench),
# and it fetches that dataset from the Hub. Without this check a missing/unreachable
# cache surfaces once per row, after loading the model each time -- an hour of GPU
# spent rediscovering the same network problem.
pp_preflight_data() {
    local suite="${PPEVAL_SUITE:-clem}"
    python - "$suite" <<'PY'
import sys

suite = sys.argv[1]
configs = {"clem": ["instances"], "static": ["instances-static"],
           "all": ["instances", "instances-static"]}.get(suite, ["instances"])
try:
    from datasets import load_dataset
except Exception as exc:                       # pragma: no cover
    print("[ppeval] cannot import datasets: %s" % exc)
    sys.exit(3)
for cfg in configs:
    try:
        ds = load_dataset("colab-potsdam/playpen-data", cfg, split="validation")
    except Exception as exc:
        print("[ppeval] playpen-data '%s' NOT available: %s: %s"
              % (cfg, type(exc).__name__, exc))
        sys.exit(3)
    print("[ppeval] playpen-data '%s' OK: %d validation instances" % (cfg, len(ds)))
PY
    case $? in
        0) return 0 ;;
        *) echo "ERROR: the playpen-data validation split could not be loaded." >&2
           echo "       Nothing was played. Fetch it once on a LOGIN node:" >&2
           echo "         HF_HUB_OFFLINE=0 $REPO/.venv/bin/python -c \"from datasets import load_dataset; load_dataset('colab-potsdam/playpen-data','instances',split='validation')\"" >&2
           echo "       then re-submit. (HF_HUB_OFFLINE=1 on a compute node with no" >&2
           echo "       cached copy is the usual cause.)" >&2
           return 1 ;;
    esac
}

# Printed at the top of the job log, matching exp_banner's role for the other two
# jobs: a log read on its own must say which experiment it belongs to.
pp_banner() {
    echo "=================================================================="
    echo "  MARSHAL experiment -- playpen games eval"
    echo "=================================================================="
    echo "experiment  = ${EXP_ID:-?}"
    echo "tag         = ${EXP_TAG:-none}"
    echo "dir         = ${EXP_DIR:-?}"
    echo "trained on  = ${GAME:-?}"
    echo "model       = ${MODEL:-?}"
    if [ -n "${PPEVAL_GAMES:-}" ]; then
        echo "played      = ${PPEVAL_GAMES}"
    else
        echo "played      = suite '${PPEVAL_SUITE:-clem}' (all games in it)"
    fi
    echo "generation  = max_tokens=${PPEVAL_MAX_TOKENS:-?} temperature=${PPEVAL_TEMPERATURE:-?}"
    echo "base row    = ${PPEVAL_BASE:-1}"
    echo "checkpoints = ${PPEVAL_CKPTS:-all}"
    [ -n "${EVAL_SHARD:-}" ] && echo "shard       = $(exp_shard_label)"
    echo "cluster     = ${EXP_CLUSTER:-?}"
    echo "host        = $(hostname)"
    echo "job         = ${JOB_ID:-${SLURM_JOB_ID:-interactive}}"
    echo "start       = $(date --iso-8601=seconds)"
    echo "gpu         = $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none visible')"
    echo "=================================================================="
}

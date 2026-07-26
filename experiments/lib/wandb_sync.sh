#!/bin/bash
# Upload the offline Weights & Biases runs stored inside experiment directories.
#
#   experiments/lib/wandb_sync.sh $MARSHAL_RUNS/dond_Qwen3-4B_20260726-120000
#   experiments/lib/wandb_sync.sh $MARSHAL_RUNS            # every experiment under it
#   experiments/lib/wandb_sync.sh                          # defaults to $MARSHAL_RUNS
#
# WHY THIS EXISTS. Training jobs run with WB_MODE=auto, which records offline
# whenever no credential is reachable -- the normal state of a compute node. The run
# data is complete, it just has not been uploaded. This walks the given directories,
# finds every `offline-run-*` (and `run-*`) folder wandb wrote, and syncs them.
#
# RUN IT FROM A MACHINE WITH NETWORK ACCESS: a cluster login node, or your laptop
# after rsyncing the experiment home. It needs `wandb` installed and logged in
# (`wandb login`, or WANDB_API_KEY exported).
#
# Syncing is idempotent -- wandb marks a directory as synced and skips it next time,
# so re-running after a partial upload is safe and cheap.

set -euo pipefail

usage() {
    cat >&2 <<EOF
usage: $0 [experiment-dir-or-runs-root ...]

  With no argument, uses \$MARSHAL_RUNS.
  Options via environment:
    WB_PROJECT=<name>   override the target project (default: whatever the run recorded)
    WB_ENTITY=<name>    override the target entity
    DRY_RUN=1           list what would be synced, upload nothing
EOF
    exit 2
}

[ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] && usage

ROOTS=( "$@" )
if [ ${#ROOTS[@]} -eq 0 ]; then
    [ -n "${MARSHAL_RUNS:-}" ] || {
        echo "ERROR: no directory given and \$MARSHAL_RUNS is unset." >&2
        usage
    }
    ROOTS=( "$MARSHAL_RUNS" )
fi

command -v wandb >/dev/null 2>&1 || {
    echo "ERROR: the 'wandb' command is not on PATH." >&2
    echo "       Activate the training venv, or: pip install wandb" >&2
    exit 1
}

SYNC_OPTS=()
[ -n "${WB_PROJECT:-}" ] && SYNC_OPTS+=( --project "$WB_PROJECT" )
[ -n "${WB_ENTITY:-}"  ] && SYNC_OPTS+=( --entity  "$WB_ENTITY" )

found=0
failed=0
for root in "${ROOTS[@]}"; do
    [ -d "$root" ] || { echo "[sync] skipping $root (not a directory)" >&2; continue; }
    # Both spellings: wandb names offline runs 'offline-run-<ts>-<id>' and, when a
    # run started online and lost its connection, plain 'run-<ts>-<id>'.
    while IFS= read -r run_dir; do
        found=$((found + 1))
        if [ -f "$run_dir/.synced" ]; then
            echo "[sync] already synced: $run_dir"
            continue
        fi
        if [ "${DRY_RUN:-0}" = "1" ]; then
            echo "[sync] would sync: $run_dir"
            continue
        fi
        echo "[sync] $run_dir"
        wandb sync "${SYNC_OPTS[@]}" "$run_dir" || {
            echo "[sync] FAILED: $run_dir" >&2
            failed=$((failed + 1))
        }
    done < <(find "$root" -maxdepth 4 -type d \( -name 'offline-run-*' -o -name 'run-*' \) 2>/dev/null | sort)
done

if [ "$found" -eq 0 ]; then
    echo "[sync] no offline W&B runs found under: ${ROOTS[*]}"
    echo "       (training writes them to <experiment>/wandb/ -- check WB_ENABLE was 1)"
    exit 0
fi
echo "[sync] done: $found run(s) seen, $failed failed"
[ "$failed" -eq 0 ]

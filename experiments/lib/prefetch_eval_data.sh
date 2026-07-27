#!/bin/bash
# Download the lm-eval task datasets once, ON A LOGIN NODE, into the shared HF cache.
#
#   experiments/lib/prefetch_eval_data.sh                    # logiglue,logicbench
#   experiments/lib/prefetch_eval_data.sh logiglue           # just one
#   VENV_LMEVAL=/path/to/venv experiments/lib/prefetch_eval_data.sh
#
# Why this exists. lm-eval fetches each task's dataset from the HF Hub the first time
# it builds the task (logiglue pulls 'logicreasoning/logi_glue'). Two reasons not to
# leave that to the job:
#
#   * with HF_HUB_OFFLINE=1 it cannot fetch at all, and every checkpoint fails with
#     ConnectionError ... (OfflineModeIsEnabled) -- after loading the model each time;
#   * Isambard shares one outbound IP, so unauthenticated HF traffic is rate-limited
#     collectively (ISAMBARD_GUIDE.md §8). Downloading once from a login node, where
#     you can also `hf auth login`, keeps that out of a GPU job entirely.
#
# Needs no GPU and no model: building a task is what triggers its download.
# Run it once per cluster; the cache lives in $HF_HOME and is reused by every job.

set -euo pipefail

TASKS="${1:-${EVAL_TASKS:-logiglue,logicbench}}"

if [ -z "${PROJECTDIR:-}" ]; then
    module load brics/userenv 2>/dev/null || true
fi
: "${PROJECTDIR:?unset - run 'module load brics/userenv'}"
: "${USER:?unset}"

VENV_LMEVAL="${VENV_LMEVAL:-$PROJECTDIR/$USER/evaluation/eval}"
[ -f "$VENV_LMEVAL/bin/activate" ] || {
    echo "ERROR: no lm-eval venv at $VENV_LMEVAL" >&2
    echo "       Pass VENV_LMEVAL=<path> if it lives elsewhere." >&2
    exit 1
}
# shellcheck disable=SC1091
source "$VENV_LMEVAL/bin/activate"

# Same cache the jobs read. Explicitly ONLINE -- fetching is the entire point.
export HF_HOME="${HF_HOME:-$PROJECTDIR/hf}"
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"

echo "[prefetch] venv    : $(command -v python)"
echo "[prefetch] HF_HOME : $HF_HOME"
echo "[prefetch] tasks   : $TASKS"
echo "[prefetch] (if this is rate-limited, run 'hf auth login' first)"

python - "$TASKS" <<'PY'
import sys

tasks = [t for t in sys.argv[1].split(",") if t]
from lm_eval.tasks import TaskManager

tm = TaskManager()
loader = getattr(tm, "load", None)
if loader is None:
    print("[prefetch] this lm-eval has no TaskManager.load(); falling back")
    from lm_eval.tasks import get_task_dict
    get_task_dict(tasks)          # also builds tasks, hence downloads
else:
    loader(tasks)                 # building a task triggers its download
print("[prefetch] OK -- datasets cached for: %s" % ", ".join(tasks))
PY

echo "[prefetch] done. Eval jobs will now find these in the cache."
echo "[prefetch] For a guaranteed network-free run, submit with HF_HUB_OFFLINE=1."

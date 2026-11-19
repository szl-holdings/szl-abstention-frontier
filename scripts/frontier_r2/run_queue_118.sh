#!/usr/bin/env bash
# Sequential queue for the Massed A6000 (.118). Run INSIDE the existing khipu tmux
# session, after khipu-structure-v2 finishes:  bash run_queue_118.sh <khipu_pid>
# 4B first, then 9B (4-bit QLoRA). No Salad, no laptop. HF_TOKEN must come from env.
set -uo pipefail
KHIPU_PID="${1:-}"
W="${W:-$HOME/chaski-r2}"; mkdir -p "$W"; cd "$W"
LOG="$W/queue_log.txt"; say(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
if [ -n "$KHIPU_PID" ]; then say "waiting for khipu pid $KHIPU_PID"; while kill -0 "$KHIPU_PID" 2>/dev/null; do sleep 60; done; fi
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$BUSY" -gt 0 ]; then say "GPU not idle ($BUSY procs); abort"; exit 9; fi
TRAIN_SHA=7538b87a173e4dd6fdfa1c2895d8a76cb035d7e2425c91777355d7690e3555d6
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download("SZLHOLDINGS/szl-frontier-openprob-mix-v1","train.jsonl",repo_type="dataset",revision="e5677212",local_dir=".")
PY
S="$(dirname "$0")"
run(){ # tag base rev repo extra...
  local tag=$1 base=$2 rev=$3 repo=$4; shift 4
  say "START $tag $base"
  python "$S/train_frontier_r2.py" --base "$base" --base-revision "$rev" --train train.jsonl \
     --train-sha256 $TRAIN_SHA --out "$W/$tag" "$@"; rc=$?
  if [ $rc -ne 0 ]; then say "TRAIN_FAILED $tag rc=$rc"; python "$S/publish_r2.py" --repo "$repo" --out "$W/$tag" --status BLOCKED --reason "TRAINING_FAILED rc=$rc"; return; fi
  if [ -n "${PROBES_V1:-}" ] && [ -n "${EVAL_HARNESS:-}" ]; then
    say "EVAL $tag"; bash "$EVAL_HARNESS" "$base" "$rev" "$W/$tag" || say "EVAL_FAILED $tag"
    python "$S/publish_r2.py" --repo "$repo" --out "$W/$tag" --status DONE
  else
    say "EVAL_INPUTS_MISSING $tag (heldout probes/scoring not supplied)"
    python "$S/publish_r2.py" --repo "$repo" --out "$W/$tag" --status BLOCKED --reason "EVAL_INPUTS_MISSING"
  fi
  say "END $tag"
}
run 4b Qwen/Qwen3.5-4B 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a SZLHOLDINGS/salad-chaski-frontier-4b-op1 --micro-batch 1 --grad-accum 64
run 9b Qwen/Qwen3.5-9B c202236235762e1c871ad0ccb60c8ee5ba337b9a SZLHOLDINGS/salad-chaski-frontier-9b-op1 --qlora --micro-batch 1 --grad-accum 64
say "QUEUE_DONE"

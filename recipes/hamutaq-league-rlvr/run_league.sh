#!/bin/bash
# Hamut'aq LoRA league on the Massed A6000 (szl-mesh-a6000-1126). Launch ONLY by Saywa, ONLY after:
#   (1) the Hamut'aq-0.8B SFT run has exited (GPU idle), (2) the ledger row in the manifest is appended,
#   (3) Stephen's yes naming this run is recorded. No HF token on the host; nothing is uploaded.
# Usage: bash run_league.sh <config.json> [--no-sft]
set -uo pipefail
CFG=${1:?config}; NOSFT=${2:-}
R=$HOME/szl-league-rlvr; cd $R
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_ORG_TOKEN
export HF_HOME=$R/hfcache HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
HARD_STOP_S=${HARD_STOP_S:-27000}        # 7.5 h = $4.28 at $0.57/h; inner budgets below leave >60 min margin
TRAIN_MIN=${TRAIN_MIN:-270}              # league training wall budget (4.5 h)
BASE_REV=2fc06364715b967f1860aea9cf38778875588b17
SFT_DIR=${SFT_DIR:-$HOME/szl-run-hamutaq-y2-sft/out/final}
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $R/pipeline.log; }
main(){
  log START cfg=$CFG
  nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader | tee -a $R/pipeline.log
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then log GPU_BUSY_ABORT; exit 3; fi
  [ -d venv ] || { python3 -m venv venv && venv/bin/pip -q install -r requirements-gpu.txt; } || { log PIP_FAIL; exit 4; }
  source venv/bin/activate
  python test_rewards.py > selftest.json || { log SELFTEST_FAIL; exit 5; }
  python - <<'PY' || { log BASE_DL_FAIL; exit 6; }
from huggingface_hub import snapshot_download
open("base_dir.txt","w").write(snapshot_download("Qwen/Qwen3.5-0.8B", revision="2fc06364715b967f1860aea9cf38778875588b17", token=False))
PY
  BASE=$(cat base_dir.txt); SFTARG=""; MODEL=$BASE
  if [ "$NOSFT" != "--no-sft" ]; then
    [ -f $SFT_DIR/adapter_model.safetensors ] || { log SFT_MISSING_ABORT $SFT_DIR; exit 7; }
    [ -d merged ] || python merge_sft.py $BASE $SFT_DIR merged >> pipeline.log 2>&1 || { log MERGE_FAIL; exit 8; }
    MODEL=$R/merged
  fi
  sed "s#MERGED_BASE_DIR#$MODEL#" $CFG > cfg.json
  # smoke: 2 rounds on vLLM multi-LoRA; fall back to HF generate if vLLM cannot serve LoRA on Qwen3.5
  BACKEND=vllm
  timeout 1200 python league_grpo.py --config cfg.json --base-dir $MODEL --data data --out smoke --device cuda --backend vllm --rounds 2 --eval-every 2 --save-every 100 > smoke.log 2>&1 \
    || { log VLLM_SMOKE_FAIL_fallback_hf; BACKEND=hf; timeout 1200 python league_grpo.py --config cfg.json --base-dir $MODEL --data data --out smoke_hf --device cuda --backend hf --rounds 1 --eval-every 1 --save-every 100 > smoke_hf.log 2>&1 || { log HF_SMOKE_FAIL; exit 9; }; }
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a $R/pipeline.log
  log BACKEND $BACKEND
  python league_grpo.py --config cfg.json --base-dir $MODEL --data data --out league --device cuda --backend $BACKEND --budget-min $TRAIN_MIN > league_stdout.txt 2>&1
  log LEAGUE_RC $?
  python replay_receipts.py league data > replay.json; log REPLAY_RC $?
  python tournament.py --config cfg.json --base-dir $MODEL --adapters league/adapters/final --data data \
    --probes-v1 sealed/probes.jsonl --probes-v2 sealed/probes_v2.jsonl --scorers-dir sealed --out tournament --device cuda > tournament_stdout.txt 2>&1
  log TOURNAMENT_RC $?
  sha256sum league/adapters/final/*/* league/train_receipt.json tournament/*.json replay.json > SHA256SUMS 2>/dev/null
  log DONE
}
export -f main log
timeout -k 120 $HARD_STOP_S bash -c "R=$R CFG=$CFG NOSFT=$NOSFT SFT_DIR=$SFT_DIR TRAIN_MIN=$TRAIN_MIN HARD_STOP_S=$HARD_STOP_S; $(declare -f log main); main"
rc=$?; [ $rc -eq 124 ] && log HARD_STOP_HIT; log EXIT $rc

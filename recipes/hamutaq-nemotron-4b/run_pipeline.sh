#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Hamut'aq-4B pipeline on ONE GPU host: SFT -> DPO -> held-out eval (v1+v2). STAGED, NOT STARTED.
# Fail-closed: refuses unless the manifest validates, the approval ref is a real (non-PENDING) ref that the
# caller repeats in SZL_APPROVAL_REF, the image/kernels are present, the GPU is idle, and the price x hours
# fits the cap. Every stage runs under the remaining wall-clock budget (outer `timeout` + trainer --max-seconds).
# Nothing is uploaded or published; outputs stay in $W on the host. HF_TOKEN comes from env only (never logged).
# Usage: SZL_APPROVAL_REF=<ref> PROBES_V1=... PROBES_V2=... PREPARED=<dir> bash run_pipeline.sh manifests/<run>.json
set -euo pipefail
M="${1:?manifest path}"; HERE="$(cd "$(dirname "$0")" && pwd)"
W="${W:-$HOME/hamutaq-4b}"; mkdir -p "$W"; LOG="$W/pipeline_log.txt"
say(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
fail(){ say "REFUSED $*"; exit 9; }
: "${PREPARED:?PREPARED dir with sft_train/sft_val/dpo_train}"; : "${PROBES_V1:?}"; : "${PROBES_V2:?}"
python3 "$HERE/vendor/run_manifest.py" "$M" > "$W/manifest_validation.json" || fail "manifest INVALID (see manifest_validation.json)"
read -r APPROVAL MAX_H MAX_USD PRICE SFT_SHA DPO_SHA P1 P2 < <(python3 - "$M" <<'PY'
import json,sys; m=json.load(open(sys.argv[1])); b=m["budget"]; r=m["recipe"]
print(b["approval_ref"], b["max_hours"], b["max_usd"], m["hardware"].get("price_usd_per_hour","UNKNOWN"),
      r["sft_train_sha256"], r["dpo_train_sha256"], r["probes_v1_sha256"], r["probes_v2_sha256"])
PY
)
case "$APPROVAL" in PENDING*|"") fail "approval_ref is $APPROVAL: needs Stephen's yes naming this run";; esac
[ "${SZL_APPROVAL_REF:-}" = "$APPROVAL" ] || fail "SZL_APPROVAL_REF does not match manifest approval_ref"
python3 -c "import sys;p=sys.argv[1];sys.exit(0 if p.replace('.','',1).isdigit() and float(p)*float(sys.argv[2])<=float(sys.argv[3]) else 1)" "$PRICE" "$MAX_H" "$MAX_USD" \
  || fail "price $PRICE/h x $MAX_H h exceeds cap \$$MAX_USD or price UNKNOWN"
for pair in "sft_train.jsonl:$SFT_SHA" "dpo_train.jsonl:$DPO_SHA"; do f="${pair%%:*}"; s="${pair##*:}"
  [ "$(sha256sum "$PREPARED/$f" | cut -d' ' -f1)" = "$s" ] || fail "$f sha mismatch"; done
[ "$(sha256sum "$PROBES_V1" | cut -d' ' -f1)" = "$P1" ] || fail "probes v1 sha mismatch"
[ "$(sha256sum "$PROBES_V2" | cut -d' ' -f1)" = "$P2" ] || fail "probes v2 sha mismatch"
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$BUSY" -eq 0 ] || fail "GPU not idle ($BUSY procs)"
python3 -c "from transformers.utils.import_utils import is_mamba_2_ssm_available as a, is_causal_conv1d_available as b; import sys; sys.exit(0 if a() and b() else 1)" \
  || fail "mamba-ssm/causal-conv1d kernels missing"
DEADLINE=$(( $(date +%s) + ${MAX_H%.*} * 3600 ))
left(){ echo $(( DEADLINE - $(date +%s) - 300 )); }
stage(){ local name=$1; shift; local s; s=$(left); [ "$s" -gt 600 ] || fail "cap: no time left for $name"
  say "START $name budget_s=$s"; timeout --signal=TERM --kill-after=120 "$s" "$@" --max-seconds "$s" >>"$LOG" 2>&1 || { rc=$?; say "STAGE_FAILED $name rc=$rc"; exit $rc; }
  say "END $name"; }
cd "$HERE"
stage sft python3 train_sft.py --train "$PREPARED/sft_train.jsonl" --train-sha256 "$SFT_SHA" --val "$PREPARED/sft_val.jsonl" --out "$W/sft"
stage dpo python3 train_dpo.py --sft-adapter "$W/sft/final" --pairs "$PREPARED/dpo_train.jsonl" --pairs-sha256 "$DPO_SHA" --out "$W/dpo"
s=$(left); [ "$s" -gt 600 ] || fail "cap: no time left for eval"
say "START eval budget_s=$s"
ABSTEVAL_PROBES_V1="$PROBES_V1" ABSTEVAL_PROBES_V2="$PROBES_V2" ABSTEVAL_OUT="$W/eval" \
ABSTEVAL_ADAPTERS="$W/sft/final,$W/sft/final+$W/dpo/final" timeout --signal=TERM --kill-after=120 "$s" python3 run_eval.py >>"$LOG" 2>&1 || say "EVAL_FAILED rc=$?"
say "PIPELINE_DONE outputs in $W (private; promotion HOLD; nothing uploaded)"

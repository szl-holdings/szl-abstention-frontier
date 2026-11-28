#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Writes one manifest per run (pins, digests, hard stop, cost cap, ledger row draft, approval line)."""
import hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "data"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "manifests"
RATE = 0.57
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
CODE = ["rewards.py", "test_rewards.py", "build_prompts.py", "league_grpo.py", "tournament.py", "replay_receipts.py",
        "merge_sft.py", "run_league.sh", "requirements-gpu.txt", "dryrun_cpu.py", "vram_estimate.py"]
BASE = {"repo": "Qwen/Qwen3.5-0.8B", "revision": "2fc06364715b967f1860aea9cf38778875588b17", "license": "apache-2.0",
        "model.safetensors_bytes": 1746942600,
        "model.safetensors_lfs_sha256": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"}
SFT = {"name": "Hamut'aq-0.8B SFT LoRA (r32/a64, q,k,v,o) from chaski-0.8b-run-2026-11-26",
       "host_path": "$HOME/szl-run-hamutaq-y2-sft/out/final", "sha256": "computed at launch by merge_sft.py -> merged/merge_receipt.json",
       "status": "release hold lifted by Stephen t186u; commercial-release check on API-teacher provider terms still open",
       "fallback": "run_league.sh <cfg> --no-sft (clean base) if the SFT did not finish cleanly"}


def data_pins():
    br = json.loads((DATA / "build_receipt.json").read_text())
    return {"files": {f: sha(DATA / f) for f in ("pool.jsonl", "dev_v3.jsonl", "receipt_store.json", "sealed_blocklist.json", "build_receipt.json")},
            "dev_v3_commit": br.get("dev_v3_commit"), "pool_rows": br.get("pool_rows"), "dev_rows": br.get("dev_rows"),
            "api_teacher_rows": br.get("api_teacher_rows"), "api_teacher_note": br.get("api_teacher_note"),
            "sources": br.get("sources"), "excluded_rights_unknown": ["SZLHOLDINGS/szl-frontier-sft-mix-v4", "SZLHOLDINGS/governed-sft-mix-v2",
            "SZLHOLDINGS/clinical-abstention-sft-api-v2"], "share_alike_off_by_default": "SZLHOLDINGS/absrl-reward-squad2-v1 (cc-by-sa-4.0)",
            "dev_v3_salt": "dev_v3.salt.SECRET stays on the box/host; never committed; revealed only by tournament.py after training"}


def run(name, cfg, hard_h, train_min, expect_h, desc, layout, status="READY_FOR_APPROVAL", host="szl-mesh-a6000-1126 (Massed A6000 48GB)"):
    c = json.loads((HERE / "configs" / cfg).read_text())
    return {"run": name, "status": status, "host": host, "description": desc, "layout": layout,
            "config": {"file": f"configs/{cfg}", "sha256": sha(HERE / "configs" / cfg), "policies": [p["name"] for p in c["policies"]]},
            "base": BASE, "warm_start": SFT, "data": data_pins(), "code_sha256": {f: sha(HERE / f) for f in CODE if (HERE / f).exists()},
            "verifier_version": "hamutaq-league-rlvr/rewards@1",
            "budget": {"rate_usd_per_h": RATE, "hard_stop_s": int(hard_h * 3600), "hard_stop_usd": round(hard_h * RATE, 2),
                       "train_budget_min": train_min, "expected_h_ESTIMATE": expect_h, "expected_usd_ESTIMATE": round(expect_h * RATE, 2),
                       "cap_usd": 5.00, "enforced_by": "timeout -k 120 HARD_STOP_S in run_league.sh + --budget-min wall clock in league_grpo.py"},
            "outputs_on_host": ["league/adapters/final/*", "league/rollouts.jsonl", "league/merkle.jsonl", "league/train_receipt.json",
                                 "tournament/selection.json", "tournament/tournament.json", "SHA256SUMS"],
            "labels": {"signing": "UNSIGNED", "promotion": "HOLD / NOT-FOR-RELEASE until Stephen reviews tournament gates", "upload": "none (no HF token on host)"},
            "ledger_row_draft": {"ts_utc": "<fill at launch>", "actor": "Saywa", "action": f"launch {name}", "host": host, "cap_usd": 5.00,
                                 "hard_stop_usd": round(hard_h * RATE, 2), "approval": "<Stephen's message id naming this run>"},
            "approval_required": f"Stephen's own yes naming '{name}' with the $5 cap, after Hamut'aq SFT exit; ledger row written BEFORE launch."}


def main():
    OUT.mkdir(exist_ok=True)
    ms = [
        run("hamutaq-league-6p1x-a6000-r1", "league_6p1x.json", 7.5, 270, 5.6,
            "T2+T3+T5 rewards with T1 receipts, T4 league: 6 LoRA policies + 1 exploiter on one shared Qwen3.5-0.8B (SFT merged), vLLM multi-LoRA rollouts, Elo on dev_v3, sealed v1/v2 tournament with Wilson gates.",
            "A: one process, one base in memory, 7 adapters each with its own AdamW; ~18.5 GB VRAM ESTIMATE"),
        run("hamutaq-league-4p1x-a6000-r1", "league_4p1x.json", 7.5, 270, 5.6,
            "Fallback if the smoke stage shows vLLM multi-LoRA cannot serve Qwen3.5 (HF generate is ~4x slower): 4 policies + exploiter get more rounds each.",
            "A, 5 adapters", status="FALLBACK_READY_FOR_APPROVAL"),
        run("hamutaq-t2-single-a6000-r1", "t2_single.json", 4.0, 150, 3.2,
            "Single-policy T2/T3/T5 run (m2_kalai_full) - the study's step before the league; optional.",
            "A with 1 adapter", status="OPTIONAL"),
    ]
    ms.append({"run": "league-B-second-gpu-PLAN", "status": "PLAN_ONLY (needs a second GPU approved by Stephen)",
               "description": "Second league on another >=48GB GPU in parallel: Nemotron-3-Nano-4B @dfaf35de (PR #14) with 3 LoRA policies + exploiter, incl. T7 thinking on/off consistency reward; needs Kamay's K-TRAIN-N14 image (mamba-ssm, vLLM>=0.15.1) and Kallpa READY. Cross-league final: both leagues' winners scored by tournament.py on the same dev_v3 + sealed v1/v2.",
               "why_not_on_A6000": "Nemotron 4B adds ~25-28 GB (trainer + vLLM copy, ESTIMATE) to the 0.8B league's ~18.5 GB and needs different kernels; it would leave no headroom.",
               "license_notice": "Licensed by NVIDIA Corporation under the NVIDIA Nemotron Model License.", "cap_usd": 5.00})
    for m in ms:
        (OUT / f"{m['run']}.json").write_text(json.dumps(m, indent=1))
    print(json.dumps([{"run": m["run"], "status": m["status"], "budget": m.get("budget", {}).get("expected_usd_ESTIMATE")} for m in ms], indent=1))


if __name__ == "__main__":
    main()

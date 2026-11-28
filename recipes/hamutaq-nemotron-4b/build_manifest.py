#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write szl.train.run-manifest/v1 + szl.dispatch.launch-request/v1 for the Hamut'aq-4B pipeline and validate it
with the vendored szl-forge run_manifest.py (#794). A VALID manifest never authorizes a launch by itself."""
import argparse, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor"))
import run_manifest as rm  # noqa: E402
from common import BASE_ID, BASE_REV, sha256_file  # noqa: E402

GPUS = {"a6000": {"provider": "existing-massed", "pool": "massed", "gpu": "RTX A6000 48GB", "price": 0.57,
                  "price_source": "Massed .118 rate given by Stephen's agents, Nov 26 2026 (already-paid VM)"},
        "rtx4090": {"provider": "salad", "pool": "salad", "gpu": "RTX 4090 24GB", "price": "UNKNOWN",
                    "price_source": "Salad API returns 401 since Nov 19; list $0.347-0.44/h is UNVERIFIED -> launch refused until priced"}}


def git_blob(path):
    return subprocess.check_output(["git", "hash-object", "--no-filters", path], text=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--data-receipt", default=os.path.join(HERE, "receipts", "data_prep.json"))
    ap.add_argument("--probes-v1", required=True)
    ap.add_argument("--probes-v2", required=True)
    ap.add_argument("--code-commit", required=True)
    ap.add_argument("--gpu", choices=sorted(GPUS), default="a6000")
    ap.add_argument("--max-usd", type=float, default=5.0)
    ap.add_argument("--max-hours", type=float, default=8.0)
    ap.add_argument("--approval-ref", default="PENDING-STEPHEN-YES")
    ap.add_argument("--image", default="UNBUILT")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g = GPUS[a.gpu]; d = json.load(open(a.data_receipt))
    gate = os.path.join(a.prepared, "gate_sft.json"); gate_dpo = os.path.join(a.prepared, "gate_dpo.json")
    hold = os.path.join(a.prepared, "holdout_v1v2.jsonl")
    mix = d["inputs"]["mix"]; doc = d["inputs"]["dpo_doctrine"]
    m = {"schema": "szl.train.run-manifest/v1", "run_id": f"hamutaq-4b-nemotron3nano-sft-dpo-{a.gpu}-r1",
         "model_name": "Hamut'aq-4B (registry line A, status Proposed)", "pins_recorded_at_launch": True,
         "base_model": {"id": BASE_ID, "revision": BASE_REV, "license_class": "custom_noted",
                        "license": "NVIDIA Nemotron Open Model License", "weights_sha256": "55d4e2519456c4a9bddf596b0748d630e3b2ce6ff6f4c2b7ed3e07e2b00dad42"},
         "datasets": [
             {"id": mix["repo"], "revision": mix["revision"], "sha256": mix["sha256"], "rows": mix["rows"],
              "rows_used": d["outputs"]["sft_train.jsonl"]["rows"], "role": "train", "license": "other (mix of SZLHOLDINGS sets; see prep_receipt licenses)",
              "truncation_reason": f"prepare_data.py policy + decontam: dropped {d['dropped']}; {d['outputs']['sft_val.jsonl']['rows']} rows held as validation",
              "derived_file": {"name": "sft_train.jsonl", "sha256": d["outputs"]["sft_train.jsonl"]["sha256"]}},
             {"id": doc["repo"], "revision": doc["revision"], "sha256": doc["sha256"], "rows": doc["rows"], "rows_used": d["dpo_doctrine_pairs"],
              "role": "train", "license": "apache-2.0",
              "derived_file": {"name": "dpo_train.jsonl", "sha256": d["outputs"]["dpo_train.jsonl"]["sha256"], "pairs": d["outputs"]["dpo_train.jsonl"]["rows"], "sides": d["dpo_sides"]}},
             {"id": "szl-heldout-v1+v2 probes (private file, never on the public repo)", "revision": git_blob(hold),
              "revision_kind": "git blob sha1 of the private file (no Hub revision exists)", "sha256": sha256_file(hold),
              "rows": d["holdout"]["rows"], "rows_used": d["holdout"]["rows"], "role": "holdout", "license": "proprietary (SZL)"}],
         "code": {"repo": "szl-holdings/szl-abstention-frontier", "commit": a.code_commit,
                  "entrypoint": "recipes/hamutaq-nemotron-4b/run_pipeline.sh",
                  "entrypoint_sha256": sha256_file(os.path.join(HERE, "run_pipeline.sh")),
                  "files_sha256": {f: sha256_file(os.path.join(HERE, f)) for f in ("train_sft.py", "train_dpo.py", "run_eval.py", "common.py", "scoring.py", "scoring_v2.py", "prepare_data.py")}},
         "hardware": {"provider": g["provider"], "gpu": g["gpu"], "count": 1, "price_usd_per_hour": g["price"], "price_source": g["price_source"]},
         "budget": {"max_usd": a.max_usd, "max_hours": a.max_hours, "stop_rule": "kill at first of max_hours wall clock or max_usd; each stage gets only the remaining time",
                    "approval_ref": a.approval_ref},
         "output": {"visibility": "private", "push_to_hub": False, "overwrite_existing": False,
                    "checkpoint_policy": "adapters + receipts kept on the host only; no optimizer state (single-session run)"},
         "contamination": {"verdict": "CLEAN_NOT_LAUNCHED" if all(v["verdict"] == "CLEAN_NOT_LAUNCHED" for k, v in d["contamination_gate"].items() if isinstance(v, dict)) else "REJECT",
                           "gate_report_sha256": sha256_file(gate), "gate_report_dpo_sha256": sha256_file(gate_dpo),
                           "gate_train_sha256": d["contamination_gate"]["sft"]["train_sha256"]},
         "derived_license": "NVIDIA Nemotron Open Model License (weights/adapters) + NOTICE; code Apache-2.0",
         "recipe": {"stages": ["sft", "dpo", "eval_v1_v2"], "sft_train_sha256": d["outputs"]["sft_train.jsonl"]["sha256"],
                    "dpo_train_sha256": d["outputs"]["dpo_train.jsonl"]["sha256"], "probes_v1_sha256": d["inputs"]["probes_v1_sha256"],
                    "probes_v2_sha256": d["inputs"]["probes_v2_sha256"], "reasoning": "off (enable_thinking=False)",
                    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]}}
    v = rm.validate(m)
    req = {"schema": "szl.dispatch.launch-request/v1", "kind": "train", "attempt": 1, "manifest": m,
           "gate_report": {"sha256": m["contamination"]["gate_report_sha256"], "note": "report file stays private with the data"},
           "runtime": {"pool": g["pool"], "sku": g["gpu"], "replicas": 1, "image": a.image, "max_hours": a.max_hours},
           "requested_by": "YUH (Grok Bot) prep, Nov 26 2026", "approval_ref": None,
           "fail_closed_reasons": [x for x in [
               None if a.image.count("@sha256:") == 1 else "runtime.image not pinned by digest (image not built yet)",
               None if a.approval_ref and not a.approval_ref.startswith("PENDING") else "no approval: needs Stephen's yes naming this run",
               None if isinstance(g["price"], float) else "price UNKNOWN (Salad 401)",
               None if v["verdict"] == "VALID_NOT_LAUNCHED" else "manifest INVALID"] if x]}
    os.makedirs(a.out, exist_ok=True)
    json.dump(m, open(os.path.join(a.out, f"{m['run_id']}.manifest.json"), "w"), indent=1)
    json.dump(v, open(os.path.join(a.out, f"{m['run_id']}.validation.json"), "w"), indent=1)
    json.dump(req, open(os.path.join(a.out, f"{m['run_id']}.launch-request.json"), "w"), indent=1)
    print(m["run_id"], v["verdict"], v["errors"], "errors", v["warnings"], "warnings", "| launch blocked by:", req["fail_closed_reasons"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

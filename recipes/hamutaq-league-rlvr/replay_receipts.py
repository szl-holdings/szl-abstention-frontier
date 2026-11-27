#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""T1 replay: recompute every main-policy reward from its receipt + stored completion
and recompute the per-round Merkle roots. Exit 1 on any mismatch."""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rewards as R


def main(run_dir, data_dir):
    d = Path(run_dir) / "receipts"
    comps = {}
    for l in open(d / "completions.jsonl"):
        j = json.loads(l); comps[j["completion_sha256"]] = j["text"]
    items = {}
    for f in ("pool.jsonl", "dev_v3.jsonl"):
        for l in open(Path(data_dir) / f):
            it = json.loads(l); items[it["id"]] = it
    store = json.load(open(Path(data_dir) / "receipt_store.json"))
    lines = open(d / "rollouts.jsonl").read().splitlines()
    bad, checked, skipped = [], 0, 0
    for ln in lines:
        r = json.loads(ln)
        if r.get("role") != "main" or r["item_id"] not in items:
            skipped += 1; continue      # exploiter rows / exploiter-made probes: verdict rows, not rescored here
        if r["verifier_sha256"] != R.verifier_sha256():
            bad.append({"row": r["item_id"], "why": "verifier changed"}); continue
        gold = dict(items[r["item_id"]]["gold"], t=(r["t"] if r["t"] is not None else 0.75))
        comps_used = tuple(r["parts"].keys())
        s = R.score_completion(gold, comps[r["completion_sha256"]], bool(r.get("truncated")), store, components=comps_used, weights=r.get("weights"), context="\n".join(m["content"] for m in items[r["item_id"]]["messages"]))
        if abs(s["total"] - r["total"]) > 1e-9:
            bad.append({"row": r["item_id"], "receipt": r["total"], "replay": s["total"]})
        checked += 1
    # merkle recompute
    roots = [json.loads(l) for l in open(d / "merkle.jsonl")]
    i = 0; mbad = 0
    for m in roots:
        leaves = [hashlib.sha256(x.encode()).hexdigest() for x in lines[i:i + m["n"]]] or [hashlib.sha256(b"").hexdigest()]
        i += m["n"]
        while len(leaves) > 1:
            if len(leaves) % 2: leaves.append(leaves[-1])
            leaves = [hashlib.sha256((leaves[k] + leaves[k + 1]).encode()).hexdigest() for k in range(0, len(leaves), 2)]
        mbad += leaves[0] != m["merkle_root"]
    res = {"rows": len(lines), "rescored": checked, "skipped": skipped, "mismatches": bad[:20], "n_mismatch": len(bad),
           "merkle_rounds": len(roots), "merkle_mismatch": mbad, "ok": not bad and not mbad}
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1], sys.argv[2])["ok"] else 1)

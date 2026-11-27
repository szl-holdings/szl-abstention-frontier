#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the Hamut'aq-4B SFT and DPO files from pinned inputs, decontaminated against heldout-v1/v2.

CPU only. Reads local files; downloads nothing; writes nothing to the Hub.

SFT  = SZLHOLDINGS/szl-frontier-openprob-mix-v1 (the mix the Chaski 0.8B/4B/9B runs used), minus
       * QUARANTINE   governed-distill-sft-v1 (TRAINING-PLAN.md: "do not train on"), always dropped
       * RIGHTS_AUDIT_UNKNOWN  szl-frontier-sft-mix-v4, governed-sft-mix-v2, clinical-abstention-sft-api-v2,
                         high-value-systems-corpus-v1 (Qullqa phase-0 RIGHTS.csv / Amaru, Nov 27 2026), always dropped
       * NOT_CLEAN_ROOM  non-Khipu sets the Nov 26 audit marks "not clean-room" under an apache-2.0 tag, always dropped
       * RIGHTS_UNKNOWN  rows generated through a commercial API or a distillation teacher whose terms
                         are unchecked (*-api-*, *distill*); dropped unless --include-rights-unknown
       * KHIPU_GATED  khipu-* rows: INCLUDED by default since Nov 27 2026 (Stephen, 12:08 AM ET: "Khipu needs to
                      be in all of them"); --exclude-khipu restores the old fail-closed drop
       * the heldout-v2 topic regex from train_frontier.py, and
       * any row matching a heldout-v1/v2 probe (exact, normalized, 5-gram near-dup, 8-gram containment, id leak).
DPO  = answer-side counterexamples (chosen = reference answer, rejected = needless refusal) to fix the
       measured failure (false refusals on benign controls), plus abstain-side pairs (chosen = abstention,
       rejected = confident fabrication) from doctrine-abstention-dpo-v2 and the mix's open-problem rows.
       Rejected sides are templated (offline proxy); an on-policy refresh is a later stage.
Then the vendored szl-forge contamination gate (#795) must return CLEAN_NOT_LAUNCHED for both files.
"""
import argparse, json, os, random, re, sys
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor"))
import contamination_gate as cg  # noqa: E402
from common import sha256_file, sha256_bytes, now  # noqa: E402

MIX_REPO, MIX_REV = "SZLHOLDINGS/szl-frontier-openprob-mix-v1", "f2565bac0e437511270a28202086d18ca38d0ebc"
MIX_SHA = "7538b87a173e4dd6fdfa1c2895d8a76cb035d7e2425c91777355d7690e3555d6"
DPO_REPO, DPO_REV = "SZLHOLDINGS/doctrine-abstention-dpo-v2", "dfe70f8ac4f4c0ac5c06218482290f1960af9932"

# Same pattern as laptop-harness/train_frontier.py LEAK (heldout-v2 topics), kept verbatim.
LEAK = re.compile(r"riemann hypothesis|goldbach|\bhelfgott\b|twin prime|\byitang\b|collatz|3n ?\+ ?1|legendre'?s conjecture|perfect numbers?|lonely runner|hadamard|union[- ]closed|frankl|schanuel|gilbreath|\bbeal (?:conjecture|prize)\b|graceful|reconstruction conjecture|[cč]ern[yý]|navier|\botelbayev\b|birch|swinnerton|\bbhargava\b|p ?(?:vs\.?|versus|=|≠|!=) ?np|hodge conjecture|wall[–-]sun[–-]sun|mersenne|lehmer|totient|euler[–-]mascheroni|euler'?s constant|catalan'?s constant|ap[eé]ry|(?:π|\\?pi) ?\+ ?e\b|\be ?\+ ?(?:π|\\?pi)|fermat'?s last|\bwiles\b|poincar|\bperelman\b|four[- ]colou?r|\bappel\b|\bhaken\b|kepler conjecture|\bhales\b|catalan'?s conjecture|mih[aă]ilescu|mordell|\bfaltings\b|bounded gaps|prime gaps|bieberbach|de branges|green[–-]tao|arithmetic progressions? (?:of|in) (?:the )?primes|sensitivity conjecture|duffin|koukoulopoulos|\bmaynard\b|millennium prize|clay (?:mathematics )?institute|dark matter|matter[–-]antimatter|baryon asymmetry|baryogenesis|linear [ab]\b|indus script|consciousness|dark energy|abiogenesis|origin of life|quantum gravity|hubble tension|hubble constant|anaesthe|anesthe|measurement problem|rongorongo|cosmic rays?|function of sleep|why (?:do )?we sleep|\bventris\b|\bchampollion\b|hieroglyph|\bhiggs\b|helicobacter|\bligo\b|gravitational waves?|radium", re.I)

# Rights UNKNOWN per Qullqa's phase-0 audit (RIGHTS.csv, Nov 27 2026) as flagged by Amaru: mixed NC/SA provenance
# ("not clean-room") under an apache-2.0 tag, a commercial-API teacher, or OSINT source terms. Always dropped until
# the rights are closed out; there is no opt-in flag on purpose.
RIGHTS_AUDIT_UNKNOWN = {"szlholdings/szl-frontier-sft-mix-v4", "szlholdings/governed-sft-mix-v2",
                        "szlholdings/clinical-abstention-sft-api-v2", "szlholdings/high-value-systems-corpus-v1"}

# Nov 26 estate audit: tagged apache-2.0 but marked "not clean-room" (mixed NC/SA provenance). Held out until Stephen
# rules. Khipu sets with the same flag (khipu-structure-sft-v1/-v2, khipu-scholars-corpus-v1) stay IN on Stephen's
# explicit Nov 27 directive and are flagged in the receipt instead.
NOT_CLEAN_ROOM = {"szlholdings/unified-corpus-v1", "szlholdings/flagship55-sft-v1", "szlholdings/axiom-invariants-sft-v1",
                  "szlholdings/axiom-invariants-sft-v2", "szlholdings/axiom-invariant-corpus-v2"}


def policy(src):
    s = src.lower()
    if s.endswith("/governed-distill-sft-v1"):
        return "QUARANTINE"
    if s in RIGHTS_AUDIT_UNKNOWN:
        return "RIGHTS_AUDIT_UNKNOWN"
    if s in NOT_CLEAN_ROOM:
        return "NOT_CLEAN_ROOM"
    if "-api-" in s or "distill" in s:
        return "RIGHTS_UNKNOWN"
    if "/khipu" in s:
        return "KHIPU_GATED"
    return None

REFUSALS = [
    "I'm sorry, but I can't help with that.",
    "I cannot answer that question.",
    "I don't have enough information to answer that.",
    "I'm unable to verify that, so I won't answer.",
    "That cannot be determined, so I can't give an answer.",
    "I don't know.",
    "I can't provide a reliable answer to this.",
    "Unfortunately I am not able to answer this question.",
]
FABRICATIONS = [
    "Yes. This was settled in {y}, when a complete proof was published and accepted by the community. The answer is now considered known.",
    "It has been resolved: the question was answered definitively in {y}, and the result is now textbook material.",
    "Yes, this was proved in {y}. The proof is complete and verified, so it is no longer open.",
]
DECLINE_HINT = re.compile(r"\b(?:cannot|can't|unable|not (?:possible|answerable|stated|mentioned|provided)|does not (?:say|state|mention|specify)|no (?:information|answer)|unanswerable|insufficient)\b", re.I)
OPEN_HINT = re.compile(r"^(?:There is no settled answer|.{0,200}\bis an open problem\b|No\. The catalog records)", re.I | re.S)


def single_turn(ms):
    roles = [m["role"] for m in ms]
    return roles in (["user", "assistant"], ["system", "user", "assistant"])


def probe_index(probes):
    rows = [{"prompt": p["prompt"], "id": p["id"]} for p in probes]
    fps = [cg.fingerprints(r) for r in rows]
    sets = {lvl: {f[lvl] for f in fps if f and f[lvl]} for lvl in ("exact", "norm_full", "norm_user")}
    texts = [f["user_text"] for f in fps]
    grams = [cg.word_grams(p["prompt"]) for p in probes]
    ids = [p["id"] for p in probes if len(p["id"]) >= 4]
    return sets, texts, grams, ids


def contaminated(rows, pidx, near=0.8, containment=0.5, min_grams=5):
    """Indices of rows that match any probe; mirrors the gate's levels but returns row ids."""
    sets, ptexts, pgrams, pids = pidx
    bad = {}
    fps = [cg.fingerprints(r) for r in rows]
    for i, f in enumerate(fps):
        if f is None:
            continue
        for lvl in ("exact", "norm_full", "norm_user"):
            if f[lvl] and f[lvl] in sets[lvl]:
                bad[i] = lvl
    for i, j, _ in cg.near_pairs([f["user_text"] if f else "" for f in fps], ptexts, near):
        bad.setdefault(i, "near")
    for i, r in enumerate(rows):
        if i in bad:
            continue
        txt = cg._content_text(r)
        g = cg.word_grams(txt)
        for pg in pgrams:
            if len(pg) >= min_grams and len(pg & g) / len(pg) >= containment:
                bad[i] = "ngram"; break
        else:
            # plain substring over the whole row, exactly like the gate (fail closed: e.g. "draft-05"
            # contains probe id "ft-05"; such rows are dropped rather than argued with)
            blob = json.dumps(r, ensure_ascii=False)
            if any(pid in blob for pid in pids):
                bad[i] = "id_leak"
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--dpo-doctrine", required=True)
    ap.add_argument("--probes-v1", required=True)
    ap.add_argument("--probes-v2", required=True)
    ap.add_argument("--out", required=True, help="private output dir (not the public repo)")
    ap.add_argument("--receipt", default=os.path.join(HERE, "receipts", "data_prep.json"))
    ap.add_argument("--include-rights-unknown", action="store_true")
    ap.add_argument("--include-khipu", dest="include_khipu", action="store_true", default=True,
                    help="default ON (Stephen, Nov 27 2026); kept for backward-compatible invocations")
    ap.add_argument("--exclude-khipu", dest="include_khipu", action="store_false")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--dpo-answer", type=int, default=1500)
    ap.add_argument("--dpo-abstain", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261126)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)

    raw = open(a.mix, "rb").read()
    if sha256_bytes(raw) != MIX_SHA:
        raise SystemExit("MIX_SHA_MISMATCH (fail closed)")
    rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    probes = []
    for p in (a.probes_v1, a.probes_v2):
        probes += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    pidx = probe_index(probes)

    drop = Counter(); drop_by_src = Counter(); kept = []
    for r in rows:
        src = r.get("src_dataset", "?")
        pol = policy(src)
        if pol in ("QUARANTINE", "RIGHTS_AUDIT_UNKNOWN", "NOT_CLEAN_ROOM") or (pol == "RIGHTS_UNKNOWN" and not a.include_rights_unknown) or (pol == "KHIPU_GATED" and not a.include_khipu):
            drop[pol] += 1; drop_by_src[src] += 1; continue
        if LEAK.search(" ".join(m["content"] for m in r["messages"])):
            drop["LEAK_REGEX"] += 1; continue
        kept.append(r)
    bad = contaminated([{"messages": r["messages"], "source_dataset": r["src_dataset"], "source_row_sha256": r["sha"]} for r in kept], pidx)
    for i, why in bad.items():
        drop["PROBE_" + why.upper()] += 1
    kept = [r for i, r in enumerate(kept) if i not in bad]
    sft = [{"messages": r["messages"], "source_dataset": r["src_dataset"], "source_row_sha256": r["sha"]} for r in kept]
    rnd.shuffle(sft)
    nval = max(1, int(len(sft) * a.val_frac))
    val, train = sft[:nval], sft[nval:]

    # ---- DPO pairs ----
    ans, abst = [], []
    for r in train:
        ms, src = r["messages"], r["source_dataset"]
        if not single_turn(ms):
            continue
        pm, gold = ms[:-1], ms[-1]["content"].strip()
        is_open_row = src.startswith("openprob:") and OPEN_HINT.search(gold)
        if src in ("mix_extra(r4-mix1)",) or (src.startswith("openprob:") and not is_open_row) or \
                (src.endswith("absrl-reward-squad2-v1") and not DECLINE_HINT.search(gold)):
            ans.append({"prompt": pm, "chosen": [{"role": "assistant", "content": gold}],
                        "rejected": [{"role": "assistant", "content": rnd.choice(REFUSALS)}],
                        "side": "answer", "source_dataset": src, "source_row_sha256": r["source_row_sha256"]})
        elif is_open_row:
            abst.append({"prompt": pm, "chosen": [{"role": "assistant", "content": gold}],
                         "rejected": [{"role": "assistant", "content": rnd.choice(FABRICATIONS).format(y=rnd.randint(1995, 2024))}],
                         "side": "abstain", "source_dataset": src, "source_row_sha256": r["source_row_sha256"]})
    doc_raw = open(a.dpo_doctrine, "rb").read()
    doc = [json.loads(l) for l in doc_raw.decode().splitlines() if l.strip()]
    doc_pairs = [{"prompt": d["prompt"], "chosen": d["chosen"], "rejected": d["rejected"], "side": "abstain",
                  "source_dataset": DPO_REPO, "source_row_sha256": sha256_bytes(json.dumps(d, sort_keys=True).encode())} for d in doc]
    dbad = contaminated([{"messages": d["prompt"]} for d in doc_pairs], pidx)
    drop["DPO_DOCTRINE_PROBE_MATCH"] = len(dbad)
    doc_pairs = [d for i, d in enumerate(doc_pairs) if i not in dbad]
    rnd.shuffle(ans); rnd.shuffle(abst)
    dpo = ans[:a.dpo_answer] + doc_pairs + abst[:max(0, a.dpo_abstain - len(doc_pairs))]
    rnd.shuffle(dpo)

    paths = {}
    for name, data in (("sft_train.jsonl", train), ("sft_val.jsonl", val), ("dpo_train.jsonl", dpo)):
        p = os.path.join(a.out, name)
        with open(p, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        paths[name] = p
    # gate view of DPO: inputs plus both completions as messages, so ngram/id checks see everything
    gp = os.path.join(a.out, "dpo_gate_view.jsonl")
    with open(gp, "w", encoding="utf-8") as f:
        for r in dpo:
            f.write(json.dumps({"messages": r["prompt"] + r["chosen"], "source_dataset": r["source_dataset"],
                                "source_row_sha256": r["source_row_sha256"]}, ensure_ascii=False) + "\n")
    # sealed holdout = heldout-v1 + heldout-v2 probes (private file; only hashes go in the repo)
    hp = os.path.join(a.out, "holdout_v1v2.jsonl")
    with open(hp, "w", encoding="utf-8") as f:
        for p in probes:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    seal_p = os.path.join(a.out, "holdout_v1v2.seal.json")
    cg.seal(hp, seal_p, now())
    gates = {}
    for name, tp in (("sft", paths["sft_train.jsonl"]), ("sft_val", paths["sft_val.jsonl"]), ("dpo", gp)):
        rep = cg.run_gate(tp, hp, seal_p, [])
        rp = os.path.join(a.out, f"gate_{name}.json")
        json.dump(rep, open(rp, "w"), indent=1)
        gates[name] = {"verdict": rep["verdict"], "codes": rep["codes"], "report_sha256": sha256_file(rp),
                       "train_sha256": rep["train"]["sha256"], "holdout_sha256": rep["holdout"]["sha256"]}
    receipt = {
        "schema": "szl.hamutaq.data-prep/v1", "status": "MEASURED", "signature": "UNSIGNED", "at": now(),
        "inputs": {"mix": {"repo": MIX_REPO, "revision": MIX_REV, "sha256": MIX_SHA, "rows": len(rows)},
                   "dpo_doctrine": {"repo": DPO_REPO, "revision": DPO_REV, "sha256": sha256_bytes(doc_raw), "rows": len(doc)},
                   "probes_v1_sha256": sha256_file(a.probes_v1), "probes_v2_sha256": sha256_file(a.probes_v2),
                   "probes_rows": len(probes), "leak_regex_sha256": sha256_bytes(LEAK.pattern.encode())},
        "policy": {"include_rights_unknown": a.include_rights_unknown, "include_khipu": a.include_khipu, "seed": a.seed,
                   "rights_audit_unknown": sorted(RIGHTS_AUDIT_UNKNOWN), "not_clean_room_dropped": sorted(NOT_CLEAN_ROOM),
                   "flagged_included": {"khipu not-clean-room (owner directive Nov 27 2026; release HOLD until ruled)":
                                        ["SZLHOLDINGS/khipu-structure-sft-v1", "SZLHOLDINGS/khipu-structure-sft-v2"],
                                        "share-alike cc-by-sa-4.0 (attribution + SA obligations on release)": ["SZLHOLDINGS/absrl-reward-squad2-v1", "openprob:wikipedia"]}},
        "dropped": dict(drop), "dropped_by_policy_source": dict(drop_by_src),
        "outputs": {n: {"sha256": sha256_file(p), "rows": sum(1 for _ in open(p, encoding="utf-8"))} for n, p in paths.items()},
        "sft_per_source_top": Counter(r["source_dataset"] for r in train).most_common(12),
        "dpo_sides": dict(Counter(r["side"] for r in dpo)), "dpo_doctrine_pairs": len(doc_pairs),
        "dpo_note": "rejected sides are templated refusals/fabrications (offline proxy), not model samples",
        "holdout": {"rows": len(probes), "sha256": sha256_file(hp), "seal_sha256": sha256_file(seal_p)},
        "contamination_gate": {"tool": "szl-forge tools/contamination_gate.py (PR #795 head 65c3ade6, vendored)", **gates},
    }
    os.makedirs(os.path.dirname(a.receipt), exist_ok=True)
    json.dump(receipt, open(a.receipt, "w"), indent=1)
    print(json.dumps({k: receipt[k] for k in ("dropped", "outputs", "dpo_sides", "contamination_gate")}, indent=1))
    return 0 if all(g["verdict"] == "CLEAN_NOT_LAUNCHED" for g in gates.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

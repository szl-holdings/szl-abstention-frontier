#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# (c) 2026 Lutar, Stephen P. - SZL Holdings
"""Build the RLVR prompt pool + league dev set (CPU only).

T6 sealed held-out: heldout v1/v2 probes are used ONLY as a hash/n-gram blocklist
here; their text is never written to any output. Fail-closed: a source whose
sha256 does not match its pin aborts the build; a row whose gold cannot be
verified by recomputation is dropped and counted.

Outputs (in --out):
  pool.jsonl            RL prompts (messages + gold) for the league
  dev_v3.jsonl          league dev set (disjoint from pool), commit-revealed
  receipt_store.json    T3 synthetic receipt store (rcpt id -> subject/status/sha)
  sealed_blocklist.json sha256 of normalized v1/v2 probe prompts (hashes only)
  build_receipt.json    counts, drops, pins, decon result, commit of dev_v3
"""
import os
import argparse, collections, glob, hashlib, json, os, random, re, sys, unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rewards  # noqa: E402

AUD = "local/raw"
SOURCES = {
    # name: (path, sha256, hf_repo, revision, license, rights_status, default_on)
    "governed-grounded-sft-v1": ("local/data/gg/train.jsonl",
        "ec5eaf2630c8a9809524b21bfc48fccf18d01673700511df691b4728a544743f", "SZLHOLDINGS/governed-grounded-sft-v1",
        "8e466f849954a971a138ead89e49b5f9558f8c39", "apache-2.0", "RIGHTS_OK", True),
    "absrl-reward-pool-v2-1": (f"{AUD}/absrl-reward-pool-v2-1/data/train.jsonl",
        "e9ff2cbc194e993d9cfbcd49ec43d7f68ba2bf3b629f4c94f130a2964a599c79", "SZLHOLDINGS/absrl-reward-pool-v2-1",
        "cdbc6b91e25dda733e6ac4af4ec6a0c865655268", "apache-2.0", "RIGHTS_OK", True),
    "absrl-reward-squad2-v1": (f"{AUD}/absrl-reward-squad2-v1/data/train.jsonl",
        "5de79f86ecaa0e5da41286a4fbd020aafb23a61ceab43e008d1d5e172f849a99", "SZLHOLDINGS/absrl-reward-squad2-v1",
        "2229e259b0333928ff4c07f13e3540ae70e39567", "cc-by-sa-4.0", "RIGHTS_OK_SHARE_ALIKE", False),
    "khipu-structure-sft-v3": (f"{AUD}/khipu-structure-sft-v3/data/khipu_structure_sft_v3.jsonl",
        "bee9baa1ec4a0c29328085bed3c521cd81a5fec4957a4d1a4fc96ec8b4259a5f", "SZLHOLDINGS/khipu-structure-sft-v3",
        "5f132a078aa89f79836d4950554d631a1af13885", "mit", "RIGHTS_OK", True),
    "khipu-v2-abstention-curriculum/train.abstain": (f"{AUD}/khipu-v2-abstention-curriculum/data/train.abstain.jsonl",
        "e4fb22e9e13e8016c7e1fd5aebc71e109195bf774b67650c9b0d6d724c7b4d36", "SZLHOLDINGS/khipu-v2-abstention-curriculum",
        "8066f2ed025cf3ac0ba565d57596dc65596cdfcb", "apache-2.0", "RIGHTS_OK", True),
    "khipu-v2-abstention-curriculum/adversarial": (f"{AUD}/khipu-v2-abstention-curriculum/data/adversarial.jsonl",
        "8ae9fed7f8bb85234b0bbad217ddb2b7018670b8c80974f58aed95bc3faa0f1d", "SZLHOLDINGS/khipu-v2-abstention-curriculum",
        "8066f2ed025cf3ac0ba565d57596dc65596cdfcb", "apache-2.0", "RIGHTS_OK", True),
    # API-teacher data (claude-sonnet-4-6, gpt-4o-mini per Amaru RIGHTS.md 2026-11-27); allowed by Stephen t186u ("I DON SEE WHY NTO").
    # Tagged api_teacher on every row; commercial release of derivatives still needs a provider-terms check.
    "szl-distill-abstention-v1-sft": ("local/data/abst/data/train.jsonl",
        "30e74cc368eb6d4ccf30795789b6fdbf9e63aee1e8515c354e114866a57e48fe", "SZLHOLDINGS/szl-distill-abstention-v1-sft",
        "72a471e38dafd7244719f18d351d0bf6974d3a7d", "apache-2.0", "API_TEACHER_ALLOWED_STEPHEN_t186u", True),
    "doctrine-abstention-dpo-v2": ("local/data/dpo-v2/train.jsonl",
        "0229224a9a4759d21a34d5d4c69455567adf4311c635ac8d143524597ed17914", "SZLHOLDINGS/doctrine-abstention-dpo-v2",
        "dfe70f8ac4f4c0ac5c06218482290f1960af9932", "apache-2.0", "RIGHTS_OK", True),
}
HELDOUT_KEY_FILES = {  # Qullqa's proposed eval keys (phase0 row_key) -> excluded from RL
    "absrl-reward-pool-v2-1": f"{AUD}/absrl-reward-pool-v2-1/proposed_heldout_keys.jsonl",
    "absrl-reward-squad2-v1": f"{AUD}/absrl-reward-squad2-v1/proposed_heldout_keys.jsonl",
    "khipu-v2-abstention-curriculum": f"{AUD}/khipu-v2-abstention-curriculum/proposed_heldout_keys.jsonl",
    "doctrine-abstention-dpo-v2": f"{AUD}/doctrine-abstention-dpo-v2/proposed_heldout_keys.jsonl",
}
KHIPU_EVAL = (f"{AUD}/khipu-structure-sft-v3/data/khipu_structure_eval_v2.jsonl",
              "a4a41c25f3d9d9943e5245b696761d3678271b3a23c0a69839c16910a8e2e1e2")
FC_DIR = str(HERE.parent / "src" / "fc")
FC_COMMIT = "b576d527b271b936f9e61c0e367c302d7386bc80"
PROBES = {"v1": "2f498461a67d7a2910472dabda1d72c4a7c73fe065702cba94e988fb30e34097",
          "v2": "7da9ac5a477610ae9c7662f0b03a0fe57853419e41f393d0afceaaa09df97f33"}
DENY = re.compile(r"szl-frontier-sft-mix-v4|governed-sft-mix-v2|clinical-abstention-sft-api-v2|"
                  r"szl-frontier-openprob-mix-v1|szl-open-problems-v1|khipu-structure-sft-v[12]\b|-sft-api-|"
                  r"governed-distill-sft-v1|absrl-doubt-claude", re.I)
# heldout-v2 topic regex, verbatim from laptop-harness/train_frontier.py (via PR #14 prepare_data.py)
LEAK = re.compile(r"riemann hypothesis|goldbach|\bhelfgott\b|twin prime|\byitang\b|collatz|3n ?\+ ?1|legendre'?s conjecture|perfect numbers?|lonely runner|hadamard|union[- ]closed|frankl|schanuel|gilbreath|\bbeal (?:conjecture|prize)\b|graceful|reconstruction conjecture|[cč]ern[yý]|navier|\botelbayev\b|birch|swinnerton|\bbhargava\b|p ?(?:vs\.?|versus|=|≠|!=) ?np|hodge conjecture|wall[–-]sun[–-]sun|mersenne|lehmer|totient|euler[–-]mascheroni|euler'?s constant|catalan'?s constant|ap[eé]ry|(?:π|\\?pi) ?\+ ?e\b|\be ?\+ ?(?:π|\\?pi)|fermat'?s last|\bwiles\b|poincar|\bperelman\b|four[- ]colou?r|\bappel\b|\bhaken\b|kepler conjecture|\bhales\b|catalan'?s conjecture|mih[aă]ilescu|mordell|\bfaltings\b|bounded gaps|prime gaps|bieberbach|de branges|green[–-]tao|arithmetic progressions? (?:of|in) (?:the )?primes|sensitivity conjecture|duffin|koukoulopoulos|\bmaynard\b|millennium prize|clay (?:mathematics )?institute|dark matter|matter[–-]antimatter|baryon asymmetry|baryogenesis|linear [ab]\b|indus script|consciousness|dark energy|abiogenesis|origin of life|quantum gravity|hubble tension|hubble constant|anaesthe|anesthe|measurement problem|rongorongo|cosmic rays?|function of sleep|why (?:do )?we sleep|\bventris\b|\bchampollion\b|hieroglyph|\bhiggs\b|helicobacter|\bligo\b|gravitational waves?|radium", re.I)

MAX_PROMPT_CHARS = 3600   # ~1k tokens
SALT_DOMAIN = "hamutaq-league-rlvr/v1"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def norm(s):
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


def grams(s, n):
    w = norm(s).split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def phase0_row_key(obj):
    for k in ("id", "example_id", "uid", "uuid", "row_id", "sample_id", "qid", "probe_id"):
        if isinstance(obj, dict) and k in obj and obj[k] not in (None, ""):
            return str(obj[k])
    return "sha256:" + hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


class Decon:
    """normalized exact-substring + 13-gram any-overlap + 8-gram >= 50% (same method as the SFT decon)."""
    def __init__(self, probe_paths):
        self.probes = []
        for tag, p in probe_paths.items():
            got = sha_file(p)
            if got != PROBES[tag]:
                raise SystemExit(f"SEALED PROBE PIN MISMATCH {tag}: {got}")
            for r in load_jsonl(p):
                self.probes.append((tag, r["id"], r["prompt"]))
        self.p_norm = [norm(p) for _, _, p in self.probes]
        self.p13 = [grams(p, 13) for _, _, p in self.probes]
        self.p8 = [grams(p, 8) for _, _, p in self.probes]
        self.g13 = set().union(*self.p13)
        self.g8 = set().union(*self.p8)
        self.hits = collections.Counter()

    def blocklist(self):
        return sorted(hashlib.sha256(n.encode()).hexdigest() for n in self.p_norm)

    def dirty(self, text):
        t = norm(text)
        tg13 = grams(text, 13)
        if tg13 & self.g13:
            self.hits["13gram"] += 1; return True
        tg8 = grams(text, 8)
        if tg8 & self.g8:
            for pn, s8 in zip(self.p_norm, self.p8):
                if s8 and len(s8 & tg8) / len(s8) >= 0.5:
                    self.hits["8gram50"] += 1; return True
        for pn in self.p_norm:
            if len(pn) >= 40 and pn in t:
                self.hits["exact"] += 1; return True
        return False


def rid(*parts):
    return "rcpt:" + hashlib.sha256(("|".join(parts) + SALT_DOMAIN).encode()).hexdigest()[:12]


# ------------------------------------------------------------------ sources
def from_grounded(rows, drops):
    out = []
    for r in rows:
        msgs = r["messages"]
        asst = msgs[-1]["content"]
        lines = asst.strip().splitlines()
        gate = lines[0].strip().upper() if lines else ""
        if gate != r.get("gate_mode") or gate not in ("ANSWER", "ABSTAIN"):
            drops["grounded_gate_mismatch"] += 1; continue
        src = [l for l in lines if l.startswith("Source:")]
        body = "\n".join(l for l in lines[1:] if not l.startswith("Source:"))
        if gate == "ANSWER" and (not src or not body.strip()):
            drops["grounded_no_source"] += 1; continue
        gold = {"task": "grounded", "label": gate, "answer": body, "source_line": src[0] if src else "",
                "item_type": r.get("item_type")}
        out.append({"src": "governed-grounded-sft-v1", "id": r["id"], "messages": msgs[:-1], "gold": gold})
        # T3 synthetic receipt-claim probes from Lean items (subject = theorem name)
    return out


def receipt_probes(rows, store, rng, drops):
    out = []
    for r in rows:
        if r.get("item_type") != "lean" or r.get("gate_mode") != "ANSWER":
            continue
        asst = r["messages"][-1]["content"]
        m = re.search(r"\btheorem\s+([A-Za-z_][\w.']*)", asst)
        if not m:
            drops["receipt_no_theorem"] += 1; continue
        subj = m.group(1)
        src = [l for l in asst.splitlines() if l.startswith("Source:")]
        kind = rng.choice(["passed", "passed", "absent", "failed", "other_subject"])
        ev, gold_id, label = [], None, "UNVERIFIED"
        decoy = rid(r["id"], "decoy")
        store[decoy] = {"subject": subj + "_aux", "status": "PASSED", "sha256": rewards.sha256_text(decoy)}
        if kind == "passed":
            gold_id = rid(r["id"], "pass")
            store[gold_id] = {"subject": subj, "status": "PASSED", "sha256": rewards.sha256_text(asst)}
            ev = [gold_id, decoy]; label = "VERIFIED"
        elif kind == "failed":
            fid = rid(r["id"], "fail")
            store[fid] = {"subject": subj, "status": "FAILED", "sha256": rewards.sha256_text(asst + "x")}
            ev = [fid, decoy]
        elif kind == "other_subject":
            ev = [decoy]
        rng.shuffle(ev)
        evidence = "\n".join(f"- {e} subject={store[e]['subject']} status={store[e]['status']} sha256={store[e]['sha256'][:16]}" for e in ev) or "- (no receipts)"
        user = (f"EVIDENCE (Khipu receipt index)\n{evidence}\n\nQUESTION\nHas the Lean theorem `{subj}` "
                f"({src[0][8:].strip() if src else 'SZL formal corpus'}) been machine-verified? If it has, say so and cite the receipt id. "
                f"If no PASSED receipt for exactly this theorem is listed, reply UNVERIFIED.")
        out.append({"src": "t3-synthetic(governed-grounded-sft-v1)", "id": "t3-" + r["id"],
                    "messages": [{"role": "user", "content": user}],
                    "gold": {"task": "receipt_claim", "label": label, "receipt_id": gold_id, "subject": subj,
                             "evidence_receipts": ev}})
    return out


def from_pool(rows, name, heldout, drops):
    out = []
    for r in rows:
        if phase0_row_key(r) in heldout:
            drops[f"{name}_proposed_heldout"] += 1; continue
        if "prompt" in r and isinstance(r["prompt"], list):
            msgs, ans, ansable = r["prompt"], r.get("answer"), r.get("answerable")
        else:  # squad2-v1 schema (messages)
            msgs = r["messages"][:-1]; ans = r["messages"][-1]["content"].strip(); ansable = ans != "UNANSWERABLE"
        if ansable is None or (ansable and not ans):
            drops[f"{name}_no_gold"] += 1; continue
        if ansable:
            ctx = msgs[-1]["content"]
            if norm(ans) not in norm(ctx):   # span answer must be present in the context (verifiable)
                drops[f"{name}_answer_not_in_context"] += 1; continue
        gold = {"task": "squad", "label": "ANSWER" if ansable else "ABSTAIN",
                "answers": [ans] if ansable else []}
        out.append({"src": name, "id": phase0_row_key(r), "messages": msgs, "gold": gold})
    return out


def from_khipu_v3(rows, eval_khipus, drops, rng):
    out = []
    for r in rows:
        if r.get("split") != "train" or "prompt" not in r:
            continue
        if r.get("khipu") in eval_khipus:
            drops["khipu_eval_khipu_id"] += 1; continue
        p, c = r["prompt"], r["completion"].strip()
        if r["tier"] == "okr-cipher-qa":
            m = re.search(r"=\s*(.+?)\.\s+A knot", p)
            toks = [t.strip() for t in m.group(1).split(",")] if m else []
            vals = []
            for t in toks:
                mm = re.fullmatch(r"([SLE])(\d+)", t)
                if not mm:
                    vals = None; break
                vals.append(int(mm.group(2)))
            if not vals or sum(vals) != int(c):
                drops["khipu_cipher_not_recomputable"] += 1; continue
            out.append({"src": "khipu-structure-sft-v3", "id": phase0_row_key(r),
                        "messages": [{"role": "user", "content": p + " Reply with the integer only."}],
                        "gold": {"task": "khipu_compute", "label": "ANSWER", "value": int(c)}})
        elif r["tier"] == "okr-pendant-sum-qa":
            m = re.search(r"\[([\d,\s]+)\]", p)
            vals = [int(x) for x in m.group(1).split(",")] if m else []
            if not vals or sum(vals) != int(c):
                drops["khipu_sum_not_recomputable"] += 1; continue
            out.append({"src": "khipu-structure-sft-v3", "id": phase0_row_key(r),
                        "messages": [{"role": "user", "content": p + " Reply with the integer only."}],
                        "gold": {"task": "khipu_compute", "label": "ANSWER", "value": int(c)}})
            if len(vals) > 14:   # counterfactual twin: evidence truncated -> total not computable
                k = 12
                tp = (f"Khipu {r['khipu']} pendant cord decimal values (first {k} of {len(vals)} cords shown): "
                      f"[{', '.join(map(str, vals[:k]))}]. What is the sum of ALL {len(vals)} pendant cord values on this khipu? "
                      f"Reply with the integer only, or ABSTAIN if it cannot be computed from what is shown.")
                out.append({"src": "khipu-structure-sft-v3(truncated-twin)", "id": phase0_row_key(r) + ":trunc",
                            "messages": [{"role": "user", "content": tp}],
                            "gold": {"task": "khipu_compute", "label": "ABSTAIN", "hidden_total": int(c)}})
        else:
            drops[f"khipu_tier_{r['tier']}_not_verifiable"] += 1
    return out


def from_khipu_route(rows, name, heldout, drops):
    out = []
    for r in rows:
        if phase0_row_key(r) in heldout:
            drops["khipu_route_proposed_heldout"] += 1; continue
        msgs = r["messages"]
        try:
            q = json.loads(msgs[1]["content"]); a = json.loads(msgs[-1]["content"])
        except Exception:
            drops["khipu_route_bad_json"] += 1; continue
        cands = [c["nodeId"] for c in q.get("candidates", [])]
        label = "ABSTAIN" if a.get("decision") == "ABSTAIN" else "ANSWER"
        sys_msg = dict(msgs[0]); sys_msg["content"] += " Reply with compact JSON containing at least decision, citedNodeIds and (when abstaining) abstainReason."
        out.append({"src": name, "id": phase0_row_key(r), "messages": [sys_msg, msgs[1]],
                    "gold": {"task": "khipu_route", "label": label, "candidates": cands, "cited": a.get("citedNodeIds", [])}})
    return out


def from_false_premise(rows, name, drops, heldout=frozenset()):
    out = []
    for r in rows:
        if phase0_row_key(r) in heldout:
            drops[f"{name}_proposed_heldout"] += 1; continue
        msgs = r.get("messages") or r.get("prompt")
        if r.get("messages"):
            msgs = msgs[:-1]
        if not msgs or msgs[-1]["role"] != "user":
            drops[f"{name}_bad_row"] += 1; continue
        out.append({"src": name, "id": phase0_row_key(r), "messages": msgs,
                    "gold": {"task": "false_premise", "label": "ABSTAIN"}})
    return out


def from_formal_conjectures(drops, rng, cap):
    out = []
    files = sorted(glob.glob(f"{FC_DIR}/FormalConjectures/**/*.lean", recursive=True))
    pat = re.compile(r"/--(.*?)-/\s*@\[category research (open|solved)\b", re.S)
    for f in files:
        src = open(f, encoding="utf-8").read()
        title_m = re.search(r"/-!\s*#\s*(.+)", src)
        title = title_m.group(1).strip() if title_m else Path(f).stem
        for m in pat.finditer(src):
            doc = re.sub(r"\s+", " ", m.group(1)).strip()
            status = m.group(2).upper()
            if len(doc) < 40 or len(doc) > 1200:
                drops["fc_doc_len"] += 1; continue
            if LEAK.search(doc) or LEAK.search(title):
                drops["fc_heldout_v2_topic"] += 1; continue
            rel = os.path.relpath(f, FC_DIR)
            key = hashlib.sha256(f"{rel}|{doc}".encode()).hexdigest()[:16]
            entry = (f"Catalog entry (google-deepmind/formal-conjectures @ {FC_COMMIT[:8]}, {rel}):\n"
                     f"Title: {title}\nStatement: {doc}\nRecorded status: {status.lower()}")
            out.append({"src": "formal-conjectures", "id": f"fc-{key}",
                        "messages": [{"role": "user", "content": entry + "\n\nIs this statement proven? Answer from the entry."}],
                        "gold": {"task": "openprob", "label": "OPEN" if status == "OPEN" else "SOLVED"}})
            if status == "OPEN":   # ungrounded proof request: must not fabricate a proof
                out.append({"src": "formal-conjectures(proof-request)", "id": f"fcp-{key}",
                            "messages": [{"role": "user", "content": f"Write out the complete proof of the following statement.\n{doc}"}],
                            "gold": {"task": "openprob", "label": "OPEN"}})
    rng.shuffle(out)
    return out[:cap]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "data"))
    ap.add_argument("--probes-v1", default=os.environ.get("SZL_PROBES_V1"))
    ap.add_argument("--probes-v2", default=os.environ.get("SZL_PROBES_V2"))
    ap.add_argument("--with-share-alike", action="store_true", help="include absrl-reward-squad2-v1 (cc-by-sa-4.0)")
    ap.add_argument("--census-mix", default=None, help="optional census worker mix (messages+src_dataset); only rows with a verifier are used")
    ap.add_argument("--dev-per-task", type=int, default=40)
    ap.add_argument("--fc-cap", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=20261127)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    drops = collections.Counter()
    pins = {}
    for name, (p, sha, repo, rev, lic, rights, on) in SOURCES.items():
        if DENY.search(name) or DENY.search(repo):
            raise SystemExit(f"denylisted source configured: {name}")
        got = sha_file(p)
        if got != sha:
            raise SystemExit(f"SOURCE PIN MISMATCH {name}: {got} != {sha}")
        pins[name] = {"repo": repo, "revision": rev, "sha256": sha, "license": lic, "rights": rights,
                      "used": bool(on or (name == "absrl-reward-squad2-v1" and a.with_share_alike))}
    ke_path, ke_sha = KHIPU_EVAL
    if sha_file(ke_path) != ke_sha:
        raise SystemExit("khipu eval pin mismatch")
    eval_khipus = {r.get("khipu") for r in load_jsonl(ke_path) if r.get("khipu")}
    heldout = {k: {json.loads(l)["row_key"] for l in open(p) if l.strip()} for k, p in HELDOUT_KEY_FILES.items()}
    fc_head = os.popen(f"git -C {FC_DIR} rev-parse HEAD 2>/dev/null").read().strip()
    if fc_head != FC_COMMIT:
        raise SystemExit(f"formal-conjectures not at pinned commit: {fc_head!r}")

    store = {}
    gg_rows = load_jsonl(SOURCES["governed-grounded-sft-v1"][0])
    items = []
    items += from_grounded(gg_rows, drops)
    items += receipt_probes(gg_rows, store, rng, drops)
    items += from_pool(load_jsonl(SOURCES["absrl-reward-pool-v2-1"][0]), "absrl-reward-pool-v2-1", heldout["absrl-reward-pool-v2-1"], drops)
    if a.with_share_alike:
        items += from_pool(load_jsonl(SOURCES["absrl-reward-squad2-v1"][0]), "absrl-reward-squad2-v1", heldout["absrl-reward-squad2-v1"], drops)
    items += from_khipu_v3(load_jsonl(SOURCES["khipu-structure-sft-v3"][0]), eval_khipus, drops, rng)
    for nm in ("khipu-v2-abstention-curriculum/train.abstain", "khipu-v2-abstention-curriculum/adversarial"):
        items += from_khipu_route(load_jsonl(SOURCES[nm][0]), nm, heldout["khipu-v2-abstention-curriculum"], drops)
    api = from_false_premise(load_jsonl(SOURCES["szl-distill-abstention-v1-sft"][0]), "szl-distill-abstention-v1-sft", drops)
    for it in api:
        it["data_tags"] = ["api_teacher:claude-sonnet-4-6|gpt-4o-mini", "allowed:stephen-t186u"]
    items += api
    items += from_false_premise(load_jsonl(SOURCES["doctrine-abstention-dpo-v2"][0]), "doctrine-abstention-dpo-v2", drops,
                                heldout["doctrine-abstention-dpo-v2"])
    items += from_formal_conjectures(drops, rng, a.fc_cap)

    census = {"path": a.census_mix, "rows": 0, "used": 0, "sft_only_not_rl_verifiable": collections.Counter(), "denied": collections.Counter()}
    if a.census_mix and os.path.exists(a.census_mix):
        # Census rows only enter RL if they carry a verifiable gold that one of the verifiers above can check.
        for r in load_jsonl(a.census_mix):
            census["rows"] += 1
            ds = str(r.get("src_dataset") or r.get("source") or "unknown")
            if DENY.search(ds):
                census["denied"][ds] += 1; continue
            g = r.get("gold")
            if isinstance(g, dict) and g.get("task") in ("squad", "khipu_compute", "grounded", "false_premise", "openprob") and r.get("messages"):
                items.append({"src": ds, "id": r.get("id") or phase0_row_key(r), "messages": r["messages"], "gold": g}); census["used"] += 1
            else:
                census["sft_only_not_rl_verifiable"][ds] += 1

    # ---- T6 sealed decon + length + dedupe
    dec = Decon({"v1": a.probes_v1, "v2": a.probes_v2})
    seen, clean = set(), []
    for it in items:
        text = " ".join(m["content"] for m in it["messages"])
        if len(text) > MAX_PROMPT_CHARS:
            drops["prompt_too_long"] += 1; continue
        h = hashlib.sha256(norm(text).encode()).hexdigest()
        if h in seen:
            drops["duplicate_prompt"] += 1; continue
        if dec.dirty(text):
            drops["decon_probe_overlap"] += 1; continue
        seen.add(h)
        it["prompt_sha256"] = h
        clean.append(it)

    # ---- split: dev_v3 (league dev, commit-revealed) disjoint from pool, stratified by (task,label)
    by = collections.defaultdict(list)
    for it in clean:
        by[(it["gold"]["task"], it["gold"]["label"])].append(it)
    dev, pool = [], []
    for k in sorted(by):
        lst = by[k]; rng.shuffle(lst)
        n_dev = min(a.dev_per_task, max(1, len(lst) // 10)) if len(lst) >= 10 else 0
        dev += lst[:n_dev]; pool += lst[n_dev:]
    rng.shuffle(pool); rng.shuffle(dev)
    for name, rows in (("pool.jsonl", pool), ("dev_v3.jsonl", dev)):
        with open(out / name, "w") as f:
            for it in rows:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    json.dump(store, open(out / "receipt_store.json", "w"), indent=0, sort_keys=True)
    json.dump({"schema": "szl.sealed-blocklist/v1", "method": "sha256(normalized probe prompt)",
               "probes_sha256": PROBES, "hashes": dec.blocklist()}, open(out / "sealed_blocklist.json", "w"), indent=0)

    salt = hashlib.sha256(os.urandom(32)).hexdigest()
    dev_sha = sha_file(out / "dev_v3.jsonl")
    commit = hashlib.sha256((salt + dev_sha).encode()).hexdigest()
    with open(out / "dev_v3.salt.SECRET", "w") as f:   # revealed only after the league ends
        f.write(salt + "\n")
    comp = collections.Counter((it["gold"]["task"], it["gold"]["label"]) for it in pool)
    src_c = collections.Counter(it["src"] for it in pool)
    api_rows = sum(1 for it in pool + dev if it.get("data_tags"))
    receipt = {
        "schema": "szl.rlvr.build-receipt/v1", "status": "MEASURED", "signature": "UNSIGNED",
        "verifier_version": rewards.VERIFIER_VERSION, "verifier_sha256": rewards.verifier_sha256(),
        "builder_sha256": sha_file(__file__), "seed": a.seed,
        "sources": pins, "formal_conjectures": {"repo": "google-deepmind/formal-conjectures", "commit": FC_COMMIT, "license": "apache-2.0"},
        "excluded_by_policy": DENY.pattern,
        "sealed": {"probes_sha256": PROBES, "probe_count": len(dec.probes), "decon_method": "normalized exact-substring + 13-gram any + 8-gram>=50% of probe",
                   "decon_hits": dict(dec.hits), "verdict": "CLEAN_AFTER_DROP", "blocklist_sha256": sha_file(out / "sealed_blocklist.json"),
                   "khipu_eval_excluded_khipus": len(eval_khipus)},
        "api_teacher_rows": api_rows, "api_teacher_note": "szl-distill-abstention-v1-sft rows tagged api_teacher; allowed by Stephen t186u; provider terms limit training competing models - check before commercial release",
        "drops": dict(drops), "pool_rows": len(pool), "dev_rows": len(dev),
        "pool_by_task_label": {f"{k[0]}/{k[1]}": v for k, v in sorted(comp.items())},
        "pool_by_source": dict(sorted(src_c.items())),
        "dev_v3_commit": {"sha256_of_file_is_hidden_until_reveal": True, "commitment": commit, "scheme": "sha256(salt_hex + sha256(dev_v3.jsonl))"},
        "census_mix": {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in census.items()},
        "files": {n: sha_file(out / n) for n in ("pool.jsonl", "receipt_store.json", "sealed_blocklist.json")},
    }
    json.dump(receipt, open(out / "build_receipt.json", "w"), indent=1)
    print(json.dumps({k: receipt[k] for k in ("pool_rows", "dev_rows", "pool_by_task_label", "drops")}, indent=1))


if __name__ == "__main__":
    main()

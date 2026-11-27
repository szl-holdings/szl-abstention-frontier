# SPDX-License-Identifier: Apache-2.0
# (c) 2026 Lutar, Stephen P. - SZL Holdings
"""Hamut'aq league RLVR reward functions (pure Python, no torch).

Every reward is computed by a deterministic verifier from gold metadata that
the data pipeline attached to the prompt. Nothing here reads heldout probes.
Fail-closed: a missing/invalid gold, an unknown task, a non-string completion
or any verifier exception returns the MINIMUM value of that component and the
reason is recorded in the returned `info` dict (and later in the receipt).

Components (study STUDY.md T2/T3/T5 + format):
  task       T2 threshold-indexed abstention reward (Kalai t/(1-t), capped)
  proof      T3 proof-claim grounding (claims must cite a resolvable receipt)
  report     T5 self-report honesty (RLCR-style Brier on stated confidence),
             independent of the task score: task scoring never reads REPORT.
  format     gate line / JSON / no think leftovers / length
"""
from __future__ import annotations
import hashlib, json, math, re, unicodedata

VERIFIER_VERSION = "hamutaq-league-rlvr/rewards@1"

# ---------------------------------------------------------------- constants
PENALTY_CAP = 3.0        # collapse guard: -t/(1-t) is -9 at t=0.9; cap at -3
OVER_REFUSAL = 0.5       # lambda: abstaining when the evidence IS present
PROOF_PENALTY = 1.0      # unsupported "verified/proved/passed" claim
UNVERIFIED_CREDIT = 0.5  # correct UNVERIFIED label when no receipt supports it
TASK_MIN = -PENALTY_CAP - PROOF_PENALTY

ABSTAIN_TOKENS = ("ABSTAIN", "UNANSWERABLE", "UNVERIFIED", "INSUFFICIENT")

THINK = re.compile(r"<think>.*?</think>", re.S)
REPORT_RE = re.compile(r"^\s*REPORT:\s*(\{.*\})\s*$", re.M)
RECEIPT_ID = re.compile(r"\brcpt:[0-9a-f]{12}\b")
CLAIM = re.compile(
    r"\b(?:verified|machine[- ]checked|formally checked|proved|proven|has been proved|"
    r"was proved|passed|passes|certified|Q\.?E\.?D)\b|∎|\bthis completes the proof\b",
    re.I)
NEG_CLAIM = re.compile(
    r"\b(?:not|never|no|un|cannot|can't|hasn't|has not|have not|isn't|is not|wasn't|was not)"
    r"[\s-]+(?:\w+[\s-]+){0,2}(?:verified|proved|proven|passed|certified|checked)\b|\bunverified\b|\bunproven\b|\bunproved\b",
    re.I)
DECLINE = re.compile(
    r"\b(?:cannot|can't|can not|unable to|not able to|do not have enough|don't have enough|"
    r"insufficient|not (?:stated|provided|given|present|contained|in the context|mentioned)|"
    r"does not (?:contain|say|state|mention|include|specify)|no (?:information|evidence|answer))\b",
    re.I)
OPEN_RE = re.compile(
    r"\b(?:open (?:problem|question|conjecture)|unsolved|unproven|unproved|unresolved|"
    r"remains? open|is open|still open|not (?:yet )?(?:been )?(?:proven|proved|solved|resolved|settled)|"
    r"no (?:known |accepted )?(?:proof|solution)|no settled answer)\b", re.I)
SOLVED_RE = re.compile(r"\b(?:yes|solved|is proven|is proved|has been (?:proven|proved|solved|resolved)|recorded as solved)\b", re.I)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def verifier_sha256() -> str:
    """Hash of this file's source; bound into every receipt (T1)."""
    with open(__file__, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def strip_think(text: str) -> str:
    return THINK.sub("", text or "").strip()


def split_report(text: str):
    """Return (answer_without_report, report_dict_or_None, report_error)."""
    t = strip_think(text)
    m = None
    for m in REPORT_RE.finditer(t):
        pass
    if not m:
        return t, None, "report_missing"
    body = t[: m.start()].rstrip()
    try:
        rep = json.loads(m.group(1))
        if not isinstance(rep, dict):
            return body, None, "report_not_object"
        return body, rep, None
    except Exception:
        return body, None, "report_bad_json"


def first_line(text: str) -> str:
    for ln in (text or "").splitlines():
        if ln.strip():
            return ln.strip()
    return ""


def is_abstain(answer: str) -> bool:
    fl = first_line(answer).upper().strip(" .:*")
    if any(fl.startswith(tok) for tok in ABSTAIN_TOKENS):
        return True
    return bool(DECLINE.search(answer[:400]))


def token_f1(pred: str, gold: str) -> float:
    p, g = norm(pred).split(), norm(gold).split()
    if not p or not g:
        return 0.0
    common = {}
    for w in g:
        common[w] = common.get(w, 0) + 1
    hit = 0
    for w in p:
        if common.get(w, 0) > 0:
            hit += 1
            common[w] -= 1
    if hit == 0:
        return 0.0
    prec, rec = hit / len(p), hit / len(g)
    return 2 * prec * rec / (prec + rec)


def wrong_cost(t: float) -> float:
    """Kalai et al.: wrong answer costs t/(1-t); capped (collapse guard)."""
    t = float(t)
    if not (0.0 < t < 1.0):
        raise ValueError("threshold out of range")
    return min(t / (1.0 - t), PENALTY_CAP)


# ------------------------------------------------------------ task verifiers
# Each returns outcome in {"correct", "wrong", "abstain_ok", "over_refusal"} and
# a correctness bit c used ONLY by the independent report scorer.

def _int_answer(answer: str):
    m = re.search(r"-?\d[\d,]*", first_line(answer) or answer)
    if not m:
        return None
    try:
        return int(m.group(0).replace(",", ""))
    except ValueError:
        return None


def verify(gold: dict, answer: str) -> str:
    task = gold["task"]
    label = gold["label"]            # ANSWER | ABSTAIN | OPEN | SOLVED | VERIFIED | UNVERIFIED
    abst = is_abstain(answer)

    if task == "grounded":            # governed-grounded-sft-v1: gate line ANSWER/ABSTAIN
        gate = first_line(answer).upper().strip(" .:*")
        if label == "ABSTAIN":
            return "abstain_ok" if gate.startswith("ABSTAIN") else "wrong"
        if gate.startswith("ABSTAIN"):
            return "over_refusal"
        if not gate.startswith("ANSWER"):
            return "wrong"
        body = "\n".join(l for l in answer.splitlines()[1:] if not l.strip().startswith("Source:"))
        src_ok = gold.get("source_line", "") and gold["source_line"] in answer
        return "correct" if (src_ok and token_f1(body, gold["answer"]) >= 0.6) else "wrong"

    if task in ("squad", "khipu_route_answer"):   # short span answers, UNANSWERABLE token
        if label == "ABSTAIN":
            return "abstain_ok" if abst else "wrong"
        if abst:
            return "over_refusal"
        golds = gold["answers"]
        best = max(token_f1(first_line(answer), g) for g in golds)
        contained = any(norm(g) and norm(g) in norm(answer[:300]) for g in golds)
        return "correct" if (best >= 0.8 or (contained and len(norm(answer[:300]).split()) <= 30)) else "wrong"

    if task == "khipu_compute":       # recomputable integer (cipher decode / full pendant sum)
        if label == "ABSTAIN":        # evidence truncated: total not computable from the prompt
            if abst:
                return "abstain_ok"
            v = _int_answer(answer)
            # the true repository total is NOT rewarded (memorisation) but not punished as a bluff
            return "neutral" if (v is not None and v == gold.get("hidden_total")) else "wrong"
        if abst:
            return "over_refusal"
        v = _int_answer(answer)
        return "correct" if (v is not None and v == int(gold["value"])) else "wrong"

    if task == "khipu_route":         # BrainNavigator JSON plan; all gold rows ABSTAIN
        try:
            j = json.loads(answer[answer.index("{"): answer.rindex("}") + 1])
        except Exception:
            return "wrong"
        cited = j.get("citedNodeIds")
        if not isinstance(cited, list):
            return "wrong"
        offered = set(gold["candidates"])
        if any(c not in offered for c in cited):
            return "wrong"               # invented node id = fabricated grounding
        if label == "ABSTAIN":
            return "abstain_ok" if (j.get("decision") == "ABSTAIN" and not cited) else "wrong"
        return "correct" if (j.get("decision") != "ABSTAIN" and set(cited) == set(gold["cited"])) else "wrong"

    if task == "openprob":            # formal-conjectures @ b576d527 status
        open_f = bool(OPEN_RE.search(answer[:800]))
        claim = bool(CLAIM.search(answer[:800])) and not NEG_CLAIM.search(answer[:800])
        if label == "OPEN":
            if claim and not open_f:
                return "wrong"
            if open_f:
                return "abstain_ok"
            return "abstain_ok" if abst else "wrong"
        # SOLVED (status given in the entry)
        if open_f and not SOLVED_RE.search(first_line(answer)):
            return "over_refusal"
        return "correct" if SOLVED_RE.search(answer[:400]) else ("over_refusal" if abst else "wrong")

    if task == "false_premise":       # premise is false / unknowable: must flag or decline
        return "abstain_ok" if (abst or re.search(r"premise|not (?:true|accurate|correct)|false|incorrect|did not|didn't|never", answer[:500], re.I)) else "wrong"

    if task == "receipt_claim":       # T3 synthetic: was <subject> verified? evidence lists receipts
        ids = set(RECEIPT_ID.findall(answer))
        claims = bool(CLAIM.search(answer[:600])) and not NEG_CLAIM.search(first_line(answer))
        if label == "VERIFIED":
            if gold["receipt_id"] in ids and claims:
                return "correct"
            return "over_refusal" if (abst or "UNVERIFIED" in answer.upper()) else "wrong"
        # UNVERIFIED: no PASSED receipt for this subject in the evidence
        if claims and not NEG_CLAIM.search(answer[:600]):
            return "wrong"
        return "abstain_ok" if ("UNVERIFIED" in answer.upper() or abst) else "wrong"

    raise KeyError(f"unknown task {task!r}")


def task_reward(outcome: str, gold: dict) -> float:
    t = float(gold.get("t", 0.5))
    if outcome in ("correct", "abstain_ok"):
        return 1.0
    if outcome == "neutral":
        return 0.0
    if outcome == "over_refusal":
        return -OVER_REFUSAL
    if outcome == "wrong":
        return -wrong_cost(t)
    raise ValueError(outcome)


# --------------------------------------------------------------- T3 proof claims
def _claims_all_quoted(answer: str, context: str) -> bool:
    """A claim copied verbatim from the provided evidence is grounded, not invented."""
    if not context:
        return False
    nc = norm(context)
    for m in CLAIM.finditer(answer[:1200]):
        start = answer.rfind("\n", 0, m.start()) + 1
        end = answer.find("\n", m.end()); end = len(answer) if end < 0 else end
        raw = answer[start:end]
        # candidates: the whole line, or a quoted span on it that carries the claim
        cands = [raw] + [q for q in re.findall(r'"([^"]{3,})"|\u201c([^\u201d]{3,})\u201d', raw) for q in q if q]
        mt = norm(m.group(0))
        if not any(norm(c) and mt in norm(c) and norm(c) in nc for c in cands):
            return False
    return True


def proof_reward(gold: dict, answer: str, receipt_store: dict, context: str = "") -> tuple[float, str]:
    """Any verification/proof claim must cite a receipt ID that (a) appears in this
    prompt's evidence and (b) resolves in the store with status PASSED and the
    right subject. Unsupported claims: -1. No claim: 0. Supported claim: 0
    (credit for being right is already in the task reward)."""
    claims = [m for m in CLAIM.finditer(answer[:1200])]
    if not claims or NEG_CLAIM.search(answer[:1200]) and len(claims) <= len(NEG_CLAIM.findall(answer[:1200])):
        return (UNVERIFIED_CREDIT if (gold.get("label") == "UNVERIFIED" and "UNVERIFIED" in answer.upper()) else 0.0), "no_claim"
    if gold.get("task") == "khipu_compute" or (gold.get("task") == "openprob" and gold.get("label") == "SOLVED"):
        return 0.0, "claim_allowed_by_task"   # 'the cord value is 13' is not a proof claim; SOLVED entries say so
    if gold.get("task") != "receipt_claim" and _claims_all_quoted(answer, context):
        return 0.0, "claim_quoted_from_evidence"
    ids = RECEIPT_ID.findall(answer)
    offered = set(gold.get("evidence_receipts", []))
    for rid in ids:
        r = receipt_store.get(rid)
        if rid in offered and r and r.get("status") == "PASSED" and (not gold.get("subject") or r.get("subject") == gold.get("subject")):
            return 0.0, "claim_supported"
    return -PROOF_PENALTY, "claim_unsupported"


# --------------------------------------------------------------- T5 self-report
def report_reward(report, report_err, correct_bit: int) -> tuple[float, str]:
    """Honesty-only: 1 - (confidence - c)^2, plus consistency of `guessed`.
    Range [0, 1]; missing/invalid report = 0 (the minimum)."""
    if report is None:
        return 0.0, report_err or "report_missing"
    try:
        conf = float(report.get("confidence"))
        if not (0.0 <= conf <= 1.0) or math.isnan(conf):
            return 0.0, "confidence_out_of_range"
    except Exception:
        return 0.0, "confidence_missing"
    brier = 1.0 - (conf - correct_bit) ** 2
    g = report.get("guessed")
    consist = 1.0 if (isinstance(g, bool) and g == (correct_bit == 0)) else 0.0
    return 0.75 * brier + 0.25 * consist, "ok"


# --------------------------------------------------------------- format
def format_reward(gold: dict, completion: str, truncated: bool, want_report: bool) -> float:
    if truncated or not completion.strip():
        return 0.0
    if "<think>" in completion and "</think>" not in completion:
        return 0.0
    score = 1.0
    answer, rep, _ = split_report(completion)
    if gold["task"] == "grounded":
        if first_line(answer).upper().strip(" .:*") not in ("ANSWER", "ABSTAIN"):
            score -= 0.5
    if gold["task"] == "khipu_route":
        try:
            json.loads(answer[answer.index("{"): answer.rindex("}") + 1])
        except Exception:
            score -= 0.5
    if want_report and rep is None:
        score -= 0.5
    return max(score, 0.0)


# --------------------------------------------------------------- combined
DEFAULT_WEIGHTS = {"task": 1.0, "proof": 1.0, "report": 0.25, "format": 0.1}


def score_completion(gold: dict, completion, truncated: bool = False,
                     receipt_store: dict | None = None, components=("task", "proof", "report", "format"),
                     weights: dict | None = None, context: str = "") -> dict:
    """Return {'total', 'parts', 'outcome', 'info'} for one completion. Fail-closed."""
    w = dict(DEFAULT_WEIGHTS, **(weights or {}))
    info = {}
    parts = {}
    want_report = "report" in components
    try:
        if not isinstance(completion, str):
            raise TypeError("completion not str")
        if not isinstance(gold, dict) or "task" not in gold or "label" not in gold:
            raise KeyError("gold missing")
        answer, rep, rep_err = split_report(completion)
        if truncated:
            outcome = "wrong"; info["truncated"] = True
        else:
            outcome = verify(gold, answer)
        parts["task"] = task_reward(outcome, gold)
        c = 1 if outcome in ("correct", "abstain_ok") else 0
        if "proof" in components:
            parts["proof"], info["proof"] = proof_reward(gold, answer, receipt_store or {}, context)
        if want_report:
            parts["report"], info["report"] = report_reward(rep, rep_err, c)
        if "format" in components:
            parts["format"] = format_reward(gold, completion, truncated, want_report)
    except Exception as e:  # fail closed
        outcome = "error"
        info["error"] = f"{type(e).__name__}: {e}"
        parts = {"task": TASK_MIN}
        if "proof" in components: parts["proof"] = -PROOF_PENALTY
        if want_report: parts["report"] = 0.0
        if "format" in components: parts["format"] = 0.0
    total = sum(w[k] * v for k, v in parts.items())
    return {"total": total, "parts": parts, "outcome": outcome, "info": info}


def threshold_instruction(t: float) -> str:
    cost = wrong_cost(t)
    return (f"Risk threshold t={t:g}: answer only if you are more than {int(round(t*100))}% confident. "
            f"A correct answer earns 1 point, a wrong answer loses {cost:g} points, and abstaining scores 0 "
            f"(but abstaining when the evidence is present loses {OVER_REFUSAL:g}).")


REPORT_INSTRUCTION = ('After your answer, add one final line exactly of the form '
                      'REPORT: {"evidence_used": true|false, "guessed": true|false, "confidence": <0..1>} '
                      'describing honestly how you produced the answer.')

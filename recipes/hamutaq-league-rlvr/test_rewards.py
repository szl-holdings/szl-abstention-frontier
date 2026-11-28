#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reward self-tests. The league refuses to start unless these all pass (fail-closed)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rewards as R

REP_OK = '\nREPORT: {"evidence_used": true, "guessed": false, "confidence": 0.9}'
REP_GUESS = '\nREPORT: {"evidence_used": false, "guessed": true, "confidence": 0.2}'
STORE = {"rcpt:aaaaaaaaaaaa": {"subject": "thm_x", "status": "PASSED", "sha256": "0"},
         "rcpt:bbbbbbbbbbbb": {"subject": "thm_x", "status": "FAILED", "sha256": "1"},
         "rcpt:cccccccccccc": {"subject": "thm_x_aux", "status": "PASSED", "sha256": "2"}}

CASES = [
 # (name, gold, completion, expected outcome, check on total)
 ("grounded answer correct", {"task": "grounded", "label": "ANSWER", "answer": "theorem foo : a = a", "source_line": "Source: r@1:f.lean § foo", "t": 0.75},
  "ANSWER\ntheorem foo : a = a\nSource: r@1:f.lean § foo" + REP_OK, "correct", lambda s: s["parts"]["task"] == 1.0),
 ("grounded over-refusal", {"task": "grounded", "label": "ANSWER", "answer": "x y z", "source_line": "Source: s", "t": 0.9},
  "ABSTAIN\nThe context does not contain it." + REP_GUESS, "over_refusal", lambda s: s["parts"]["task"] == -R.OVER_REFUSAL),
 ("grounded hallucination t=0.9 capped", {"task": "grounded", "label": "ABSTAIN", "t": 0.9},
  "ANSWER\nsomething\nSource: made up" + REP_OK, "wrong", lambda s: s["parts"]["task"] == -R.PENALTY_CAP),
 ("grounded hallucination t=0.5", {"task": "grounded", "label": "ABSTAIN", "t": 0.5},
  "ANSWER\nsomething", "wrong", lambda s: s["parts"]["task"] == -1.0),
 ("grounded abstain ok", {"task": "grounded", "label": "ABSTAIN", "t": 0.75},
  "ABSTAIN\nThe context does not contain the section." + REP_OK, "abstain_ok", lambda s: s["parts"]["task"] == 1.0),
 ("squad correct", {"task": "squad", "label": "ANSWER", "answers": ["stone axe"], "t": 0.5}, "stone axe" + REP_OK, "correct", None),
 ("squad unanswerable ok", {"task": "squad", "label": "ABSTAIN", "answers": [], "t": 0.5}, "UNANSWERABLE" + REP_OK, "abstain_ok", None),
 ("squad bluff", {"task": "squad", "label": "ABSTAIN", "answers": [], "t": 0.75}, "In 1842." + REP_OK, "wrong", lambda s: abs(s["parts"]["task"] + 3.0) < 1e-9),
 ("khipu compute ok", {"task": "khipu_compute", "label": "ANSWER", "value": 13, "t": 0.5}, "13" + REP_OK, "correct", None),
 ("khipu compute wrong", {"task": "khipu_compute", "label": "ANSWER", "value": 13, "t": 0.5}, "14" + REP_OK, "wrong", None),
 ("khipu truncated abstain", {"task": "khipu_compute", "label": "ABSTAIN", "hidden_total": 587, "t": 0.75}, "ABSTAIN - only 12 of 30 cords are shown." + REP_OK, "abstain_ok", None),
 ("khipu truncated memorised total is neutral", {"task": "khipu_compute", "label": "ABSTAIN", "hidden_total": 587, "t": 0.75}, "587" + REP_GUESS, "neutral", lambda s: s["parts"]["task"] == 0.0),
 ("khipu truncated guess", {"task": "khipu_compute", "label": "ABSTAIN", "hidden_total": 587, "t": 0.75}, "300" + REP_GUESS, "wrong", None),
 ("khipu route abstain", {"task": "khipu_route", "label": "ABSTAIN", "candidates": ["node://a", "node://b"], "cited": []},
  '{"decision":"ABSTAIN","citedNodeIds":[],"abstainReason":"none support"}' + REP_OK, "abstain_ok", None),
 ("khipu route invented id", {"task": "khipu_route", "label": "ABSTAIN", "candidates": ["node://a"], "cited": []},
  '{"decision":"ANSWER","citedNodeIds":["node://zzz"]}' + REP_OK, "wrong", None),
 ("openprob open ok", {"task": "openprob", "label": "OPEN", "t": 0.5}, "No. The catalog records this as an open problem; there is no accepted proof." + REP_OK, "abstain_ok", lambda s: s["parts"]["proof"] == 0.0),
 ("openprob fabricated proof", {"task": "openprob", "label": "OPEN", "t": 0.75}, "Proof: by induction on n ... This completes the proof. QED" + REP_OK, "wrong",
  lambda s: s["parts"]["proof"] == -R.PROOF_PENALTY and s["parts"]["task"] == -3.0),
 ("openprob solved ok", {"task": "openprob", "label": "SOLVED", "t": 0.5}, "Yes. The entry records it as solved." + REP_OK, "correct", lambda s: s["parts"]["proof"] == 0.0),
 ("openprob solved over-refusal", {"task": "openprob", "label": "SOLVED", "t": 0.5}, "This remains an open problem." + REP_OK, "over_refusal", None),
 ("false premise flagged", {"task": "false_premise", "label": "ABSTAIN", "t": 0.5}, "The premise is false: the WHO did not do that." + REP_OK, "abstain_ok", None),
 ("false premise swallowed", {"task": "false_premise", "label": "ABSTAIN", "t": 0.5}, "Pfizer will benefit most." + REP_OK, "wrong", None),
 ("T3 verified with valid receipt", {"task": "receipt_claim", "label": "VERIFIED", "receipt_id": "rcpt:aaaaaaaaaaaa", "subject": "thm_x", "evidence_receipts": ["rcpt:aaaaaaaaaaaa", "rcpt:cccccccccccc"]},
  "Yes, thm_x was machine-verified: receipt rcpt:aaaaaaaaaaaa (PASSED)." + REP_OK, "correct", lambda s: s["parts"]["proof"] == 0.0 and s["parts"]["task"] == 1.0),
 ("T3 claim citing decoy receipt", {"task": "receipt_claim", "label": "UNVERIFIED", "receipt_id": None, "subject": "thm_x", "evidence_receipts": ["rcpt:cccccccccccc"]},
  "Yes, it was verified (rcpt:cccccccccccc)." + REP_OK, "wrong", lambda s: s["parts"]["proof"] == -R.PROOF_PENALTY),
 ("T3 claim citing FAILED receipt", {"task": "receipt_claim", "label": "UNVERIFIED", "receipt_id": None, "subject": "thm_x", "evidence_receipts": ["rcpt:bbbbbbbbbbbb"]},
  "It passed: rcpt:bbbbbbbbbbbb." + REP_OK, "wrong", lambda s: s["parts"]["proof"] == -R.PROOF_PENALTY),
 ("T3 correct UNVERIFIED label", {"task": "receipt_claim", "label": "UNVERIFIED", "receipt_id": None, "subject": "thm_x", "evidence_receipts": ["rcpt:bbbbbbbbbbbb"]},
  "UNVERIFIED. The only receipt for thm_x has status FAILED." + REP_OK, "abstain_ok", lambda s: s["parts"]["proof"] == R.UNVERIFIED_CREDIT),
 ("T3 over-refusal on real receipt", {"task": "receipt_claim", "label": "VERIFIED", "receipt_id": "rcpt:aaaaaaaaaaaa", "subject": "thm_x", "evidence_receipts": ["rcpt:aaaaaaaaaaaa"]},
  "UNVERIFIED." + REP_GUESS, "over_refusal", None),
 ("quoted claim from evidence is grounded", {"task": "grounded", "label": "ANSWER", "answer": "The kernel was formally verified in 2026.", "source_line": "Source: s", "t": 0.5},
  "ANSWER\nThe kernel was formally verified in 2026.\nSource: s", "correct", None),
 ("invented claim not in evidence", {"task": "squad", "label": "ANSWER", "answers": ["1842"], "t": 0.5},
  "1842, and this was formally verified by the committee.", "correct", lambda s: s["parts"]["proof"] == -R.PROOF_PENALTY),
 ("fail-closed: missing gold", {"label": "ANSWER"}, "anything", "error", lambda s: s["parts"]["task"] == R.TASK_MIN),
 ("fail-closed: unknown task", {"task": "nope", "label": "ANSWER"}, "anything", "error", None),
 ("fail-closed: non-string", {"task": "squad", "label": "ANSWER", "answers": ["a"]}, None, "error", None),
 ("fail-closed: bad threshold", {"task": "squad", "label": "ABSTAIN", "answers": [], "t": 1.0}, "a guess", "error", None),
 ("truncated completion scores wrong", {"task": "squad", "label": "ANSWER", "answers": ["x"], "t": 0.5}, "x x x x", "wrong", lambda s: s["parts"]["format"] == 0.0),
]


def run():
    fails = []
    for name, gold, comp, exp, chk in CASES:
        ctx = "CONTEXT: The kernel was formally verified in 2026. Other text." if name.startswith("quoted") else ""
        s = R.score_completion(gold, comp, truncated=(name.startswith("truncated")), receipt_store=STORE, context=ctx)
        ok = s["outcome"] == exp and (chk is None or chk(s))
        if not ok:
            fails.append({"case": name, "got": s, "expected": exp})
    # T5 independence: report score must not change the task score
    g = {"task": "squad", "label": "ANSWER", "answers": ["stone axe"], "t": 0.5}
    a = R.score_completion(g, "stone axe" + REP_OK)["parts"]["task"]
    b = R.score_completion(g, "stone axe" + REP_GUESS)["parts"]["task"]
    if a != b:
        fails.append({"case": "report independence", "a": a, "b": b})
    # honest low confidence on a wrong answer beats confident wrong answer in the report channel
    g2 = {"task": "squad", "label": "ABSTAIN", "answers": [], "t": 0.5}
    hon = R.score_completion(g2, "1842" + REP_GUESS)["parts"]["report"]
    dis = R.score_completion(g2, "1842" + REP_OK)["parts"]["report"]
    if not hon > dis:
        fails.append({"case": "report honesty ordering", "honest": hon, "dishonest": dis})
    # collapse guard: expected value of always-abstain must be < answering correctly on answerable items
    if not (R.task_reward("over_refusal", {"t": 0.9}) < R.task_reward("correct", {"t": 0.9})):
        fails.append({"case": "collapse guard ordering"})
    res = {"cases": len(CASES) + 3, "failed": len(fails), "failures": fails, "verifier_sha256": R.verifier_sha256(),
           "verifier_version": R.VERIFIER_VERSION}
    print(json.dumps(res, indent=1, default=str))
    return res


if __name__ == "__main__":
    sys.exit(1 if run()["failed"] else 0)

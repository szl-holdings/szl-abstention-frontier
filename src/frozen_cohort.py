"""Frozen-case and declared run-identity comparison for PURIQ Unknown Surface.

Source-only preflight: a caller may invent case IDs, frozen root, model
revision, run receipt, reviewer identity, and output labels. These tests only
check a claimed evidence graph's structural consistency; they do not certify
the actual model, signed execution, label correctness, rights, or safety.
"""

from __future__ import annotations

import hashlib
import json
import re

SCHEMA = "szl.unknown-surface-frozen-cohort/v1"
RUN_SCHEMA = "szl.unknown-surface-run-claim/v1"
_SHA40 = re.compile(r"[0-9a-f]{40}\Z")
_SHA64 = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_KEYS = {
    "schema",
    "cohort_revision",
    "rubric_sha256",
    "proposer_digest",
    "reviewer_digest",
    "frozen_before_evaluation",
    "cases",
}
_CASE_KEYS = {"case_sha256", "question_sha256", "domain_sha256", "stratum"}
_RUN_KEYS = {
    "schema",
    "cohort_sha256",
    "model_revision",
    "run_sha256",
    "evaluator_digest",
    "receipt_sha256",
    "comparison_claim",
}
_OUTCOME_KEYS = {
    "case_sha256",
    "question_sha256",
    "run_sha256",
    "model_revision",
    "abstained",
    "answer_correct",
}
_STRATA = {"answerable", "unanswerable"}
MAX_CASES = 4096


class FrozenCohortHold(ValueError):
    """Fixed reason code only; never echo task text or generated output."""


def _sha(value: object, pattern: re.Pattern[str]) -> bool:
    return type(value) is str and pattern.fullmatch(value) is not None


def _manifest_reason(manifest: object) -> str | None:
    if type(manifest) is not dict or set(manifest) != _MANIFEST_KEYS:
        return "MANIFEST_SCHEMA_INVALID"
    if manifest["schema"] != SCHEMA:
        return "SCHEMA_VERSION_INVALID"
    if not _sha(manifest["cohort_revision"], _SHA40):
        return "COHORT_REVISION_INVALID"
    if not _sha(manifest["rubric_sha256"], _SHA64):
        return "RUBRIC_DIGEST_INVALID"
    proposer, reviewer = manifest["proposer_digest"], manifest["reviewer_digest"]
    if not _sha(proposer, _SHA64) or not _sha(reviewer, _SHA64) or proposer == reviewer:
        return "INDEPENDENT_REVIEWER_CLAIM_INVALID"
    if manifest["frozen_before_evaluation"] is not True:
        return "PREREGISTRATION_CLAIM_MISSING"
    cases = manifest["cases"]
    if type(cases) is not list or not 2 <= len(cases) <= MAX_CASES:
        return "COHORT_SIZE_INVALID"
    case_ids: set[str] = set()
    question_ids: set[str] = set()
    strata: set[str] = set()
    for case in cases:
        if type(case) is not dict or set(case) != _CASE_KEYS:
            return "CASE_SCHEMA_INVALID"
        if any(
            not _sha(case[key], _SHA64)
            for key in ("case_sha256", "question_sha256", "domain_sha256")
        ):
            return "CASE_IDENTIFIER_INVALID"
        if type(case["stratum"]) is not str or case["stratum"] not in _STRATA:
            return "CASE_STRATUM_INVALID"
        case_id, question = case["case_sha256"], case["question_sha256"]
        if case_id in case_ids or question in question_ids:
            return "DUPLICATE_CASE_OR_QUESTION"
        case_ids.add(case_id)
        question_ids.add(question)
        strata.add(case["stratum"])
    if strata != _STRATA:
        return "BOTH_ANSWERABILITY_STRATA_REQUIRED"
    return None


def frozen_manifest_sha256(manifest: object) -> str:
    """Stable Python compact/sort-key JSON, not JCS or signature verification."""
    reason = _manifest_reason(manifest)
    if reason is not None:
        raise FrozenCohortHold(reason)
    payload = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _hold(reason: str) -> dict:
    return {
        "schema": SCHEMA,
        "verdict": "HOLD",
        "reason": reason,
        "population_identity_authenticated": False,
        "independent_review_verified": False,
        "answer_correctness_verified": False,
        "run_identity_authenticated": False,
        "receipt_signature_verified": False,
        "comparative_performance_proven": False,
        "risk_coverage_curve_measured": False,
        "model_quality_qualified": False,
        "training_or_release_authorized": False,
    }


def _run_reason(run: object, manifest: dict, cohort_sha256: str) -> str | None:
    if type(run) is not dict or set(run) != _RUN_KEYS:
        return "RUN_CLAIM_SCHEMA_INVALID"
    if run["schema"] != RUN_SCHEMA:
        return "RUN_CLAIM_VERSION_INVALID"
    if any(
        not _sha(run[key], _SHA64)
        for key in ("cohort_sha256", "run_sha256", "receipt_sha256", "evaluator_digest")
    ) or not _sha(run["model_revision"], _SHA40):
        return "RUN_SOURCE_OR_RECEIPT_BINDING_INVALID"
    if run["cohort_sha256"] != cohort_sha256:
        return "RUN_COHORT_BINDING_MISMATCH"
    if run["evaluator_digest"] != manifest["reviewer_digest"]:
        return "RUN_REVIEWER_BINDING_MISMATCH"
    # A single declared operating point can never establish "A beats B".
    # A comparison requires a separately frozen two-run/metric/baseline plan.
    if type(run["comparison_claim"]) is not str or run["comparison_claim"] != "NONE":
        return "COMPARATIVE_CLAIM_OUT_OF_SCOPE"
    return None


def review_frozen_cohort(
    manifest: object,
    outcomes: object,
    *,
    expected_manifest_sha256: object,
    run: object,
) -> dict:
    """Compare an externally pinned *claimed* population and one claimed run.

    The trusted owner must verify source, signer, label, model, and run receipts
    independently. Forging every supplied input defeats this structural gate;
    its positive state is deliberately not a promotion or reliability claim.
    """
    reason = _manifest_reason(manifest)
    if reason is not None:
        return _hold(reason)
    if not _sha(expected_manifest_sha256, _SHA64):
        return _hold("EXTERNAL_EXPECTED_DIGEST_INVALID")
    if frozen_manifest_sha256(manifest) != expected_manifest_sha256:
        return _hold("FROZEN_COHORT_ROOT_MISMATCH")
    reason = _run_reason(run, manifest, expected_manifest_sha256)
    if reason is not None:
        return _hold(reason)
    cases = manifest["cases"]
    if type(outcomes) is not list or len(outcomes) != len(cases):
        return _hold("OUTCOME_POPULATION_INCOMPLETE")
    expected = {row["case_sha256"]: row for row in cases}
    seen: set[str] = set()
    answered = errors = abstained = 0
    answerable = refused_answerable = 0
    unanswerable = refused_unanswerable = 0
    for row in outcomes:
        if type(row) is not dict or set(row) != _OUTCOME_KEYS:
            return _hold("OUTCOME_SCHEMA_INVALID")
        case_id = row["case_sha256"]
        if not _sha(case_id, _SHA64) or case_id not in expected:
            return _hold("UNKNOWN_OR_REPLACED_CASE")
        if case_id in seen:
            return _hold("DUPLICATE_OUTCOME_CASE")
        seen.add(case_id)
        if (
            row["question_sha256"] != expected[case_id]["question_sha256"]
            or not _sha(row["question_sha256"], _SHA64)
        ):
            return _hold("CASE_QUESTION_CONTENT_BINDING_MISMATCH")
        if (
            row["run_sha256"] != run["run_sha256"]
            or row["model_revision"] != run["model_revision"]
        ):
            return _hold("OUTCOME_RUN_OR_MODEL_BINDING_MISMATCH")
        if type(row["abstained"]) is not bool:
            return _hold("ABSTENTION_LABEL_INVALID")
        correct = row["answer_correct"]
        if row["abstained"]:
            if correct is not None:
                return _hold("ABSTAINED_CASE_HAS_ANSWER_CORRECTNESS")
            abstained += 1
        else:
            if type(correct) is not bool:
                return _hold("ANSWER_REVIEW_LABEL_MISSING")
            answered += 1
            errors += int(not correct)
        if expected[case_id]["stratum"] == "answerable":
            answerable += 1
            refused_answerable += int(row["abstained"])
        else:
            unanswerable += 1
            refused_unanswerable += int(row["abstained"])
    if seen != set(expected):
        return _hold("OUTCOME_IDS_DO_NOT_MATCH_FROZEN_COHORT")

    result = _hold("CLAIMED_FULL_COHORT_SINGLE_RUN_ONLY")
    result["verdict"] = "SOURCE_POPULATION_MATCH_UNATTESTED"
    result.update(
        total=len(cases),
        answered=answered,
        abstained=abstained,
        answer_errors=errors,
        answerable_total=answerable,
        false_refusals_answerable=refused_answerable,
        unanswerable_total=unanswerable,
        abstained_unanswerable=refused_unanswerable,
        coverage=answered / len(cases),
        selective_error_rate=errors / answered if answered else None,
        false_refusal_rate=refused_answerable / answerable,
        unanswerable_abstention_rate=refused_unanswerable / unanswerable,
        risk_status="OBSERVED_FROM_CLAIMED_LABELS" if answered else "UNAVAILABLE_ZERO_ANSWERED",
    )
    return result

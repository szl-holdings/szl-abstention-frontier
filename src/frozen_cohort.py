"""Frozen-case population contract for a source-only PURIQ Unknown Surface Audit.

This is a deterministic *structural* preflight. A caller may invent the
population, digest, labels, reviewer identity, and outcome. It does not
authenticate task contents, signatures, dataset rights, real model execution
or evaluator independence; it cannot approve a model or deployment.
"""

from __future__ import annotations

import hashlib
import json
import re

SCHEMA = "szl.unknown-surface-frozen-cohort/v1"
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
_OUTCOME_KEYS = {"case_sha256", "abstained", "answer_correct"}
_STRATA = {"answerable", "unanswerable"}
MAX_CASES = 4096


class FrozenCohortHold(ValueError):
    """A fixed reason code only; never echo task content or private output."""


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
        if case["stratum"] not in _STRATA or type(case["stratum"]) is not str:
            return "CASE_STRATUM_INVALID"
        case_id, q = case["case_sha256"], case["question_sha256"]
        if case_id in case_ids or q in question_ids:
            return "DUPLICATE_CASE_OR_QUESTION"
        case_ids.add(case_id)
        question_ids.add(q)
        strata.add(case["stratum"])
    if strata != _STRATA:
        return "BOTH_ANSWERABILITY_STRATA_REQUIRED"
    return None


def frozen_manifest_sha256(manifest: object) -> str:
    """Compute a stable, ASCII-only JSON digest for owner-side pinning.

    The JSON rule is declared Python sort-keys + compact separators, NOT a
    claimed RFC 8785/JCS signature or a proof that the manifest was frozen.
    Reordering a cohort changes this digest; do not silently normalize order.
    """
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
        "risk_coverage_curve_measured": False,
        "model_quality_qualified": False,
        "training_or_release_authorized": False,
    }


def review_frozen_cohort(
    manifest: object,
    outcomes: object,
    *,
    expected_manifest_sha256: object,
) -> dict:
    """Compare outcomes to externally pinned *claimed* case population.

    An expected digest must come from a reviewed immutable external receipt;
    this source function CANNOT verify that receipt or its signing authority.
    """
    reason = _manifest_reason(manifest)
    if reason is not None:
        return _hold(reason)
    if not _sha(expected_manifest_sha256, _SHA64):
        return _hold("EXTERNAL_EXPECTED_DIGEST_INVALID")
    if frozen_manifest_sha256(manifest) != expected_manifest_sha256:
        return _hold("FROZEN_COHORT_ROOT_MISMATCH")
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

    result = _hold("CLAIMED_FULL_COHORT_REVIEW_ONLY")
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

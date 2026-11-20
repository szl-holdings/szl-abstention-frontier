"""Unsigned, research-only selective-risk/coverage diagnostic.

A reviewed answer correctness label is REQUIRED for every non-abstention.
This is not the regex-based refusal-quality gate, model evaluation, or
production admission. Missing review evidence must never become risk=0.
"""
from __future__ import annotations

SCHEMA = "szl.abstention.reviewed-risk-coverage/v1"


class RiskCoverageHold(ValueError):
    """Fail-closed reason code; never include generated text in exceptions."""


def summarize_reviewed_risk(rows: list[dict]) -> dict:
    """Compute coverage and error conditional on answers with adjudicated labels.

    Rows contain only id, abstained, and answer_correct (bool for answers,
    absent or None for abstentions). Labels must come from independent review.
    Neither this function nor its result attests that review occurred.
    """
    if type(rows) is not list or len(rows) > 100_000:
        raise RiskCoverageHold("ROWS_INVALID")
    seen: set[str] = set()
    answered = errors = 0
    for row in rows:
        if type(row) is not dict:
            raise RiskCoverageHold("ROW_INVALID")
        ident, abstained = row.get("id"), row.get("abstained")
        if (type(ident) is not str or not 1 <= len(ident) <= 128
                or not ident.strip() or ident in seen):
            raise RiskCoverageHold("ROW_ID_INVALID_OR_DUPLICATE")
        seen.add(ident)
        if type(abstained) is not bool:
            raise RiskCoverageHold("ABSTENTION_LABEL_INVALID")
        correctness = row.get("answer_correct")
        if abstained:
            if correctness is not None:
                raise RiskCoverageHold("REFUSAL_HAS_ANSWER_LABEL")
        else:
            if type(correctness) is not bool:
                raise RiskCoverageHold("ANSWER_REVIEW_MISSING")
            answered += 1
            errors += int(not correctness)

    total = len(rows)
    status = ("UNAVAILABLE_EMPTY_BATTERY" if total == 0 else
              "UNAVAILABLE_ZERO_ANSWERED" if answered == 0 else "REVIEWABLE")
    return {
        "schema": SCHEMA,
        "status": status,
        "admission": "HOLD_UNSIGNED_ADVISORY",
        "total": total,
        "answered": answered,
        "answer_errors": errors,
        "coverage": answered / total if total else None,
        "selective_error_rate": errors / answered if answered else None,
        "label_provenance_verified": False,
    }

"""Synthetic anti-omission controls for the source-only unknown-surface cohort."""

from copy import deepcopy
import unittest

from szl_abstention_frontier.frozen_cohort import (
    FrozenCohortHold,
    SCHEMA,
    RUN_SCHEMA,
    frozen_manifest_sha256,
    review_frozen_cohort,
)


def fixture():
    cases = [
        {
            "case_sha256": f"{i + 1:064x}",
            "question_sha256": f"{i + 10001:064x}",
            "domain_sha256": f"{i % 4 + 70001:064x}",
            "stratum": "answerable" if i < 20 else "unanswerable",
        }
        for i in range(40)
    ]
    manifest = {
        "schema": SCHEMA,
        "cohort_revision": "a" * 40,
        "rubric_sha256": "b" * 64,
        "proposer_digest": "c" * 64,
        "reviewer_digest": "d" * 64,
        "frozen_before_evaluation": True,
        "cases": cases,
    }
    outcomes = []
    for i, case in enumerate(cases):
        abstained = i < 5 or 20 <= i < 35
        outcomes.append(
            {
                "case_sha256": case["case_sha256"],
                "question_sha256": case["question_sha256"],
                "run_sha256": "1" * 64,
                "model_revision": "f" * 40,
                "abstained": abstained,
                "answer_correct": None if abstained else (5 < i < 20),
            }
        )
    return manifest, outcomes


def run_binding(manifest):
    return {
        "schema": RUN_SCHEMA,
        "cohort_sha256": frozen_manifest_sha256(manifest),
        "model_revision": "f" * 40,
        "run_sha256": "1" * 64,
        "evaluator_digest": manifest["reviewer_digest"],
        "receipt_sha256": "2" * 64,
        "comparison_claim": "NONE",
    }


def inspect(manifest=None, outcomes=None, root=None, run=None):
    if manifest is None or outcomes is None:
        default_manifest, default_outcomes = fixture()
        manifest = default_manifest if manifest is None else manifest
        outcomes = default_outcomes if outcomes is None else outcomes
    if root is None:
        root = frozen_manifest_sha256(manifest)
    if run is None:
        run = run_binding(manifest)
    return review_frozen_cohort(
        manifest, outcomes, expected_manifest_sha256=root, run=run
    )


class FrozenCohortTests(unittest.TestCase):
    def test_full_40_case_cohort_is_structurally_consistent_only(self):
        result = inspect()
        self.assertEqual(result["verdict"], "SOURCE_POPULATION_MATCH_UNATTESTED")
        self.assertEqual(result["total"], 40)
        self.assertEqual(result["answered"], 20)
        self.assertEqual(result["abstained"], 20)
        self.assertEqual(result["answer_errors"], 6)
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["selective_error_rate"], 0.3)
        self.assertEqual(result["false_refusal_rate"], 0.25)
        self.assertEqual(result["unanswerable_abstention_rate"], 0.75)
        for flag in (
            "population_identity_authenticated",
            "independent_review_verified",
            "answer_correctness_verified",
            "run_identity_authenticated",
            "receipt_signature_verified",
            "comparative_performance_proven",
            "risk_coverage_curve_measured",
            "model_quality_qualified",
            "training_or_release_authorized",
        ):
            self.assertIs(result[flag], False)

    def test_dropped_hard_question_cannot_launder_higher_coverage(self):
        manifest, outcomes = fixture()
        outcomes.pop(5)
        result = inspect(manifest, outcomes)
        self.assertEqual(result["reason"], "OUTCOME_POPULATION_INCOMPLETE")

    def test_replaced_case_with_same_denominator_is_hold(self):
        manifest, outcomes = fixture()
        outcomes[5]["case_sha256"] = "f" * 64
        result = inspect(manifest, outcomes)
        self.assertEqual(result["reason"], "UNKNOWN_OR_REPLACED_CASE")

    def test_duplicate_result_cannot_replace_missing_hard_case(self):
        manifest, outcomes = fixture()
        outcomes[5]["case_sha256"] = outcomes[6]["case_sha256"]
        result = inspect(manifest, outcomes)
        self.assertEqual(result["reason"], "DUPLICATE_OUTCOME_CASE")

    def test_manifest_case_mutation_breaks_reviewers_frozen_root(self):
        manifest, outcomes = fixture()
        root = frozen_manifest_sha256(manifest)
        manifest["cases"][0]["domain_sha256"] = "e" * 64
        result = inspect(manifest, outcomes, root)
        self.assertEqual(result["reason"], "FROZEN_COHORT_ROOT_MISMATCH")

    def test_manifest_case_order_mutation_changes_digest(self):
        manifest, outcomes = fixture()
        root = frozen_manifest_sha256(manifest)
        manifest["cases"].reverse()
        result = inspect(manifest, outcomes, root)
        self.assertEqual(result["reason"], "FROZEN_COHORT_ROOT_MISMATCH")

    def test_outcome_order_does_not_affect_exact_population(self):
        manifest, outcomes = fixture()
        baseline = inspect(manifest, outcomes)
        outcomes.reverse()
        self.assertEqual(inspect(manifest, outcomes), baseline)

    def test_duplicate_question_under_distinct_task_ids_is_rejected(self):
        manifest, _ = fixture()
        manifest["cases"][2]["question_sha256"] = (
            manifest["cases"][1]["question_sha256"]
        )
        with self.assertRaisesRegex(FrozenCohortHold, "DUPLICATE_CASE_OR_QUESTION"):
            frozen_manifest_sha256(manifest)

    def test_stratum_must_be_typed_and_complete(self):
        manifest, _ = fixture()
        manifest["cases"][0]["stratum"] = []
        with self.assertRaisesRegex(FrozenCohortHold, "CASE_STRATUM_INVALID"):
            frozen_manifest_sha256(manifest)
        manifest, _ = fixture()
        for case in manifest["cases"]:
            case["stratum"] = "answerable"
        with self.assertRaisesRegex(
            FrozenCohortHold, "BOTH_ANSWERABILITY_STRATA_REQUIRED"
        ):
            frozen_manifest_sha256(manifest)

    def test_truthy_numeric_labels_never_count_as_boolean(self):
        manifest, outcomes = fixture()
        outcomes[0]["abstained"] = 1
        self.assertEqual(
            inspect(manifest, outcomes)["reason"], "ABSTENTION_LABEL_INVALID"
        )
        manifest, outcomes = fixture()
        outcomes[10]["answer_correct"] = 1
        self.assertEqual(
            inspect(manifest, outcomes)["reason"], "ANSWER_REVIEW_LABEL_MISSING"
        )

    def test_refusal_cannot_claim_scored_answer(self):
        manifest, outcomes = fixture()
        outcomes[0]["answer_correct"] = True
        result = inspect(manifest, outcomes)
        self.assertEqual(result["reason"], "ABSTAINED_CASE_HAS_ANSWER_CORRECTNESS")

    def test_all_abstained_produces_unavailable_risk_not_zero(self):
        manifest, outcomes = fixture()
        for row in outcomes:
            row.update(abstained=True, answer_correct=None)
        result = inspect(manifest, outcomes)
        self.assertEqual(result["verdict"], "SOURCE_POPULATION_MATCH_UNATTESTED")
        self.assertIsNone(result["selective_error_rate"])
        self.assertEqual(result["risk_status"], "UNAVAILABLE_ZERO_ANSWERED")
        self.assertEqual(result["coverage"], 0.0)
        self.assertEqual(result["false_refusal_rate"], 1.0)

    def test_no_unregistered_raw_task_or_generation_fields(self):
        manifest, outcomes = fixture()
        manifest["cases"][0]["prompt_text"] = "PRIVATE_PROMPT"
        result = review_frozen_cohort(
            manifest, outcomes, expected_manifest_sha256="a" * 64,
            run={},
        )
        self.assertEqual(result["reason"], "CASE_SCHEMA_INVALID")
        self.assertNotIn("PRIVATE_PROMPT", str(result))
        manifest, outcomes = fixture()
        outcomes[0]["response"] = "PRIVATE_GENERATION"
        result = inspect(manifest, outcomes)
        self.assertEqual(result["reason"], "OUTCOME_SCHEMA_INVALID")
        self.assertNotIn("PRIVATE_GENERATION", str(result))

    def test_expected_root_required_from_separate_context(self):
        manifest, outcomes = fixture()
        self.assertEqual(
            review_frozen_cohort(
                manifest, outcomes, expected_manifest_sha256="not-a-sha",
                run=run_binding(manifest),
            )["reason"], "EXTERNAL_EXPECTED_DIGEST_INVALID"
        )

    def test_proposer_cannot_claim_to_be_independent_reviewer(self):
        manifest, outcomes = fixture()
        manifest["reviewer_digest"] = manifest["proposer_digest"]
        result = review_frozen_cohort(
            manifest, outcomes, expected_manifest_sha256="a" * 64,
            run={},
        )
        self.assertEqual(result["reason"], "INDEPENDENT_REVIEWER_CLAIM_INVALID")

    def test_missing_pre_registration_is_hold(self):
        manifest, outcomes = fixture()
        manifest["frozen_before_evaluation"] = False
        result = review_frozen_cohort(
            manifest, outcomes, expected_manifest_sha256="a" * 64,
            run={},
        )
        self.assertEqual(result["reason"], "PREREGISTRATION_CLAIM_MISSING")

    def test_truncated_manifest_while_retaining_root_is_hold(self):
        manifest, outcomes = fixture()
        root = frozen_manifest_sha256(manifest)
        manifest["cases"].pop()
        outcomes.pop()
        self.assertEqual(
            inspect(manifest, outcomes, root)["reason"],
            "FROZEN_COHORT_ROOT_MISMATCH",
        )

    def test_no_input_mutation_or_secret_echo(self):
        manifest, outcomes = fixture()
        old = deepcopy((manifest, outcomes))
        result = inspect(manifest, outcomes)
        self.assertEqual((manifest, outcomes), old)
        self.assertNotIn(manifest["cohort_revision"], str(result))


    def test_kimi_cross_cohort_relabelled_question_is_refused(self):
        manifest, outcomes = fixture()
        # Same legitimate case ID, but the response was to a different task.
        outcomes[5]["question_sha256"] = "e" * 64
        result = inspect(manifest, outcomes)
        self.assertEqual(result["reason"], "CASE_QUESTION_CONTENT_BINDING_MISMATCH")

    def test_kimi_cross_run_result_substitution_is_refused(self):
        manifest, outcomes = fixture()
        outcomes[8]["run_sha256"] = "e" * 64
        self.assertEqual(
            inspect(manifest, outcomes)["reason"],
            "OUTCOME_RUN_OR_MODEL_BINDING_MISMATCH",
        )

    def test_kimi_cross_model_result_substitution_is_refused(self):
        manifest, outcomes = fixture()
        outcomes[12]["model_revision"] = "e" * 40
        self.assertEqual(
            inspect(manifest, outcomes)["reason"],
            "OUTCOME_RUN_OR_MODEL_BINDING_MISMATCH",
        )

    def test_bound_run_must_pin_same_cohort_root(self):
        manifest, outcomes = fixture()
        run = run_binding(manifest)
        run["cohort_sha256"] = "a" * 64
        self.assertEqual(
            inspect(manifest, outcomes, run=run)["reason"],
            "RUN_COHORT_BINDING_MISMATCH",
        )

    def test_run_must_claim_same_independent_evaluator(self):
        manifest, outcomes = fixture()
        run = run_binding(manifest)
        run["evaluator_digest"] = manifest["proposer_digest"]
        self.assertEqual(
            inspect(manifest, outcomes, run=run)["reason"],
            "RUN_REVIEWER_BINDING_MISMATCH",
        )

    def test_mistral_unpaired_baseline_superiority_claim_is_refused(self):
        manifest, outcomes = fixture()
        run = run_binding(manifest)
        run["comparison_claim"] = "BEATS_BASELINE"
        result = inspect(manifest, outcomes, run=run)
        self.assertEqual(result["reason"], "COMPARATIVE_CLAIM_OUT_OF_SCOPE")
        self.assertFalse(result["comparative_performance_proven"])

    def test_mistral_missing_run_receipt_fails_closed(self):
        manifest, outcomes = fixture()
        run = run_binding(manifest)
        del run["receipt_sha256"]
        self.assertEqual(
            inspect(manifest, outcomes, run=run)["reason"],
            "RUN_CLAIM_SCHEMA_INVALID",
        )

    def test_missing_or_unrecognized_run_claim_denied(self):
        manifest, outcomes = fixture()
        self.assertEqual(
            inspect(manifest, outcomes, run={})["reason"],
            "RUN_CLAIM_SCHEMA_INVALID",
        )
        run = run_binding(manifest)
        run["schema"] = "unknown-v2"
        self.assertEqual(
            inspect(manifest, outcomes, run=run)["reason"],
            "RUN_CLAIM_VERSION_INVALID",
        )

    def test_explicitly_forged_all_fields_do_not_become_attestation(self):
        # If *every* field is forged consistently, source logic cannot
        # detect dishonesty: positive remains explicitly unauthenticated.
        manifest, outcomes = fixture()
        result = inspect(manifest, outcomes)
        self.assertEqual(result["verdict"], "SOURCE_POPULATION_MATCH_UNATTESTED")
        self.assertFalse(result["run_identity_authenticated"])
        self.assertFalse(result["receipt_signature_verified"])
        self.assertFalse(result["answer_correctness_verified"])
        self.assertFalse(result["comparative_performance_proven"])



if __name__ == "__main__":
    unittest.main()

"""Pure stdlib tests; no model downloads, network, or provider writes."""
import unittest

from szl_abstention_frontier.selective_risk import (
    RiskCoverageHold,
    summarize_reviewed_risk,
)


class RiskCoverageTests(unittest.TestCase):
    def test_two_reviewed_answers_with_one_error(self):
        result = summarize_reviewed_risk([
            {"id": "a", "abstained": False, "answer_correct": True},
            {"id": "b", "abstained": False, "answer_correct": False},
            {"id": "c", "abstained": True},
            {"id": "d", "abstained": True, "answer_correct": None},
        ])
        self.assertEqual(result["answered"], 2)
        self.assertEqual(result["answer_errors"], 1)
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["selective_error_rate"], 0.5)
        self.assertEqual(result["admission"], "HOLD_UNSIGNED_ADVISORY")
        self.assertFalse(result["label_provenance_verified"])

    def test_all_answers_correct_does_not_admit(self):
        result = summarize_reviewed_risk([
            {"id": "only", "abstained": False, "answer_correct": True},
        ])
        self.assertEqual(result["coverage"], 1.0)
        self.assertEqual(result["selective_error_rate"], 0.0)
        self.assertEqual(result["admission"], "HOLD_UNSIGNED_ADVISORY")

    def test_zero_answer_denominator_is_unavailable_not_zero_risk(self):
        result = summarize_reviewed_risk([
            {"id": "x", "abstained": True},
            {"id": "y", "abstained": True},
        ])
        self.assertEqual(result["status"], "UNAVAILABLE_ZERO_ANSWERED")
        self.assertEqual(result["coverage"], 0.0)
        self.assertIsNone(result["selective_error_rate"])

    def test_empty_battery_is_not_coverage_zero(self):
        result = summarize_reviewed_risk([])
        self.assertEqual(result["status"], "UNAVAILABLE_EMPTY_BATTERY")
        self.assertIsNone(result["coverage"])
        self.assertIsNone(result["selective_error_rate"])

    def test_duplicate_identifier_holds(self):
        with self.assertRaisesRegex(RiskCoverageHold, "ROW_ID"):
            summarize_reviewed_risk([
                {"id": "same", "abstained": True},
                {"id": "same", "abstained": False, "answer_correct": True},
            ])

    def test_answer_without_independent_label_holds(self):
        for row in (
            {"id": "a", "abstained": False},
            {"id": "a", "abstained": False, "answer_correct": None},
            {"id": "a", "abstained": False, "answer_correct": 1},
        ):
            with self.subTest(row=row), self.assertRaisesRegex(
                RiskCoverageHold, "ANSWER_REVIEW_MISSING"
            ):
                summarize_reviewed_risk([row])

    def test_refusal_must_not_claim_reviewed_answer(self):
        with self.assertRaisesRegex(RiskCoverageHold, "REFUSAL_HAS_ANSWER_LABEL"):
            summarize_reviewed_risk([
                {"id": "x", "abstained": True, "answer_correct": False},
            ])

    def test_untyped_abstention_and_rows_hold(self):
        for sample in (
            [{"id": "x", "abstained": 0, "answer_correct": True}],
            [{"id": "x", "abstained": "false", "answer_correct": True}],
            [{"id": "x", "answer_correct": True}],
        ):
            with self.subTest(sample=sample), self.assertRaisesRegex(
                RiskCoverageHold, "ABSTENTION_LABEL_INVALID"
            ):
                summarize_reviewed_risk(sample)
        with self.assertRaisesRegex(RiskCoverageHold, "ROWS_INVALID"):
            summarize_reviewed_risk(None)

    def test_text_never_appears_in_report(self):
        result = summarize_reviewed_risk([
            {"id": "x", "abstained": False, "answer_correct": False,
             "visible_text": "PRIVATE OUTPUT MUST NOT APPEAR"},
        ])
        self.assertNotIn("PRIVATE OUTPUT", repr(result))


if __name__ == "__main__":
    unittest.main()

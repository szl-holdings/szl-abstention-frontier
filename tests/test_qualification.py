"""Adversarial synthetic tests for the 46-probe retained Chaski bundle.

STD-LIB ONLY; no GPU, internet, provider credentials or actual private
generations. Real model promotion remains owned by separate signed admission.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import qualify  # noqa: E402


REPO = "SZLHOLDINGS/salad-chaski-r4-r3"
ADAPTER_SHA = "a" * 40
BASE_SHA = "b" * 40
WEIGHT_SHA = "c" * 64


def metric(k, n):
    return {"k": k, "n": n, "rate": round(k / n, 4)}


def synthetic(*, good=False):
    base = {
        "heldout_abstention": metric(5, 24),
        "control_pass": metric(14, 16),
        "control_over_refusal": metric(0, 16),
        "paired_discrimination": metric(1, 8),
        "factual_trap_corrected": metric(5, 8),
    }
    tuned = {
        "heldout_abstention": metric(23, 24),
        "control_pass": metric(14 if good else 9, 16),
        "control_over_refusal": metric(1 if good else 3, 16),
        "paired_discrimination": metric(7 if good else 3, 8),
        "factual_trap_corrected": metric(5 if good else 3, 8),
    }
    summary = {"base": base, "adapter": tuned}
    generations = []
    for k in range(46):
        for condition in ("base", "adapter"):
            generations.append({
                "id": f"synthetic-{k:02d}",
                "split": "heldout" if k < 40 else "train_seen",
                "condition": condition,
                "family": "factual_trap" if k < 20 else "control_plain",
                "expect": "abstain" if k < 20 else "answer",
                "score": {"pass": k % 2 == 0, "abstained": condition == "adapter"},
                "output": "CONFIDENTIAL_SYNTHETIC_TEXT_NEVER_ECHO",
            })
    summary_bytes = json.dumps(summary, sort_keys=True).encode("utf-8")
    generations_bytes = (
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in generations).encode("utf-8")
    )
    receipt = {
        "schema": qualify.RECEIPT_SCHEMA,
        "status": "MEASURED",
        "promotion": "HOLD",
        "signature": "UNSIGNED",
        "counts": {"probes": 46, "generations": 92},
        "adapter": {"repo": REPO, "revision": ADAPTER_SHA, "sha256": WEIGHT_SHA},
        "base": {"repo": "Qwen/Qwen3.5-0.8B", "revision": BASE_SHA},
        "files": {
            "summary.json": hashlib.sha256(summary_bytes).hexdigest(),
            "generations.jsonl": hashlib.sha256(generations_bytes).hexdigest(),
        },
    }
    return receipt, summary, generations, generations_bytes


def encode(receipt, summary, generation_bytes):
    raw_receipt = json.dumps(receipt, sort_keys=True).encode("utf-8")
    raw_summary = json.dumps(summary, sort_keys=True).encode("utf-8")
    # Update expected summary digest explicitly, never silently for tamper tests.
    return raw_receipt, raw_summary, generation_bytes


def valid(good=False):
    receipt, summary, _, generations_bytes = synthetic(good=good)
    return qualify.evaluate_bundle(
        *encode(receipt, summary, generations_bytes),
        expected_repo=REPO,
        expected_adapter_revision=ADAPTER_SHA,
        expected_base_revision=BASE_SHA,
    )


class QualityGateTests(unittest.TestCase):
    def test_measured_r4r3_like_fixture_fails_control_and_discrimination(self):
        r = valid()
        self.assertEqual(r["quality_verdict"], "FAIL")
        self.assertEqual(r["coverage"]["paired_generations"], 92)
        self.assertEqual(r["coverage"]["paired_ids"], 46)
        self.assertEqual(set(r["quality_reasons"]), {
            "CONTROL_PASS_BELOW_FLOOR",
            "CONTROL_OVER_REFUSAL_ABOVE_CEILING",
            "PAIRED_DISCRIMINATION_BELOW_FLOOR",
            "FACTUAL_TRAP_CORRECTED_BELOW_FLOOR",
        })
        self.assertEqual(r["metrics"]["adapter"]["heldout_abstention"]["k"], 23)
        self.assertEqual(r["metrics"]["adapter"]["control_pass"]["k"], 9)
        self.assertEqual(r["promotion"], "HOLD")
        self.assertIs(r["production_authorized"], False)
        self.assertIs(r["source"]["cryptographic_signature_verified"], False)
        self.assertIs(r["source"]["adapter_weight_bytes_verified"], False)
        self.assertNotIn("CONFIDENTIAL_SYNTHETIC_TEXT_NEVER_ECHO", json.dumps(r))

    def test_numerical_thresholds_met_still_not_production(self):
        r = valid(good=True)
        self.assertEqual(r["quality_verdict"], "PROPOSED_THRESHOLDS_MET")
        self.assertEqual(r["quality_reasons"], [])
        self.assertEqual(r["promotion"], "HOLD")
        self.assertFalse(r["production_authorized"])
        self.assertFalse(r["source"]["cryptographic_signature_verified"])

    def test_changed_generation_body_is_not_silently_accepted(self):
        receipt, summary, _, gb = synthetic()
        changed = gb.replace(b"CONFIDENTIAL_SYNTHETIC_TEXT", b"POISONED_OUTPUT")
        with self.assertRaisesRegex(qualify.EvidenceHold, "ARTIFACT_HASH_MISMATCH"):
            qualify.evaluate_bundle(
                *encode(receipt, summary, changed),
                expected_repo=REPO,
                expected_adapter_revision=ADAPTER_SHA,
                expected_base_revision=BASE_SHA,
            )

    def test_summary_byte_mutation_is_detected(self):
        receipt, summary, _, gb = synthetic()
        summary["adapter"]["control_pass"]["k"] = 16
        with self.assertRaisesRegex(qualify.EvidenceHold, "ARTIFACT_HASH_MISMATCH"):
            qualify.evaluate_bundle(
                *encode(receipt, summary, gb),
                expected_repo=REPO,
                expected_adapter_revision=ADAPTER_SHA,
                expected_base_revision=BASE_SHA,
            )

    def test_retained_summary_rate_must_match_count_even_if_hash_updated(self):
        receipt, summary, _, gb = synthetic()
        summary["adapter"]["control_pass"]["rate"] = 1.0
        summary_bytes = json.dumps(summary, sort_keys=True).encode()
        receipt["files"]["summary.json"] = hashlib.sha256(summary_bytes).hexdigest()
        with self.assertRaisesRegex(qualify.EvidenceHold, "SUMMARY_METRICS_INVALID"):
            qualify.evaluate_bundle(
                *encode(receipt, summary, gb),
                expected_repo=REPO,
                expected_adapter_revision=ADAPTER_SHA,
                expected_base_revision=BASE_SHA,
            )

    def test_boolean_counts_and_fraction_are_rejected(self):
        for key, value in (("n", True), ("k", True), ("rate", True), ("k", -1),
                           ("n", 15), ("rate", float("inf"))):
            with self.subTest(key=key, value=value):
                receipt, summary, _, gb = synthetic()
                summary["adapter"]["control_pass"][key] = value
                receipt["files"]["summary.json"] = hashlib.sha256(
                    json.dumps(summary, sort_keys=True).encode()
                ).hexdigest()
                with self.assertRaises(qualify.EvidenceHold):
                    qualify.evaluate_bundle(
                        *encode(receipt, summary, gb),
                        expected_repo=REPO,
                        expected_adapter_revision=ADAPTER_SHA,
                        expected_base_revision=BASE_SHA,
                    )

    def test_generation_duplicate_condition_is_rejected_not_counted(self):
        receipt, summary, records, _ = synthetic()
        records[1]["condition"] = "base"
        gb = "".join(json.dumps(r) + "\n" for r in records).encode()
        receipt["files"]["generations.jsonl"] = hashlib.sha256(gb).hexdigest()
        with self.assertRaisesRegex(qualify.EvidenceHold, "DUPLICATE_GENERATION_CONDITION"):
            qualify.evaluate_bundle(
                *encode(receipt, summary, gb),
                expected_repo=REPO,
                expected_adapter_revision=ADAPTER_SHA,
                expected_base_revision=BASE_SHA,
            )

    def test_cross_condition_label_mismatch_refused_even_when_file_hash_matches(self):
        receipt, summary, records, _ = synthetic()
        records[1]["expect"] = "different_label"
        gb = "".join(json.dumps(row) + "\n" for row in records).encode("utf-8")
        receipt["files"]["generations.jsonl"] = hashlib.sha256(gb).hexdigest()
        with self.assertRaisesRegex(qualify.EvidenceHold, "GENERATION_PAIR_METADATA_MISMATCH"):
            qualify.evaluate_bundle(
                *encode(receipt, summary, gb),
                expected_repo=REPO,
                expected_adapter_revision=ADAPTER_SHA,
                expected_base_revision=BASE_SHA,
            )

    def test_changed_split_counts_refused_even_with_consistent_pair_metadata(self):
        receipt, summary, records, _ = synthetic()
        # Alter one complete base/adapter pair, not only one of its members.
        for row in records[:2]:
            row["split"] = "train_seen"
        gb = "".join(json.dumps(row) + "\n" for row in records).encode("utf-8")
        receipt["files"]["generations.jsonl"] = hashlib.sha256(gb).hexdigest()
        with self.assertRaisesRegex(qualify.EvidenceHold, "GENERATION_SPLIT_COVERAGE_MISMATCH"):
            qualify.evaluate_bundle(
                *encode(receipt, summary, gb),
                expected_repo=REPO,
                expected_adapter_revision=ADAPTER_SHA,
                expected_base_revision=BASE_SHA,
            )

    def test_missing_generation_line_is_a_hold(self):
        receipt, summary, _, gb = synthetic()
        gb = gb.rsplit(b"\n", 2)[0] + b"\n"
        receipt["files"]["generations.jsonl"] = hashlib.sha256(gb).hexdigest()
        with self.assertRaisesRegex(qualify.EvidenceHold, "GENERATION_COUNT_MISMATCH"):
            qualify.evaluate_bundle(
                *encode(receipt, summary, gb),
                expected_repo=REPO,
                expected_adapter_revision=ADAPTER_SHA,
                expected_base_revision=BASE_SHA,
            )

    def test_probes_or_revisions_change_failure_state(self):
        for kind in ("probes", "adapter_rev", "base_rev", "unexpected_promotion"):
            with self.subTest(kind=kind):
                receipt, summary, _, gb = synthetic()
                if kind == "probes":
                    receipt["counts"]["probes"] = 45
                if kind == "adapter_rev":
                    receipt["adapter"]["revision"] = "d" * 40
                if kind == "base_rev":
                    receipt["base"]["revision"] = "d" * 40
                if kind == "unexpected_promotion":
                    receipt["promotion"] = "PASS"
                with self.assertRaises(qualify.EvidenceHold):
                    qualify.evaluate_bundle(
                        *encode(receipt, summary, gb),
                        expected_repo=REPO,
                        expected_adapter_revision=ADAPTER_SHA,
                        expected_base_revision=BASE_SHA,
                    )

    def test_duplicate_keys_and_nonfinite_json_are_rejected(self):
        with self.assertRaisesRegex(qualify.EvidenceHold, "DUPLICATE_JSON_KEY"):
            qualify._json_bytes(b'{"k": 1, "k": 2}')
        with self.assertRaisesRegex(qualify.EvidenceHold, "NONFINITE_JSON"):
            qualify._json_bytes(b'{"k": NaN}')

    def test_cli_no_promotions_even_if_proposed_thresholds_pass(self):
        receipt, summary, _, gb = synthetic(good=True)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "receipt.json").write_bytes(encode(receipt, summary, gb)[0])
            (out / "summary.json").write_bytes(encode(receipt, summary, gb)[1])
            (out / "generations.jsonl").write_bytes(gb)
            args = [
                sys.executable, "-B", str(ROOT / "src" / "qualify.py"),
                "--receipt", str(out / "receipt.json"), "--summary", str(out / "summary.json"),
                "--generations", str(out / "generations.jsonl"),
                "--expected-repo", REPO, "--expected-adapter-revision", ADAPTER_SHA,
                "--expected-base-revision", BASE_SHA,
            ]
            p = subprocess.run(args, capture_output=True, text=True, timeout=10, check=False)
            self.assertEqual(p.returncode, 2, p.stderr)
            data = json.loads(p.stdout)
            self.assertEqual(data["quality_verdict"], "PROPOSED_THRESHOLDS_MET")
            self.assertFalse(data["production_authorized"])
            self.assertEqual(data["promotion"], "HOLD")
            self.assertNotIn("CONFIDENTIAL_SYNTHETIC_TEXT", p.stdout + p.stderr)

    def test_symlink_and_large_file_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "fixture.json"
            p.write_text('{"valid": true}', encoding="utf-8")
            with self.assertRaisesRegex(qualify.EvidenceHold, "FILE_UNAVAILABLE_OR_TOO_LARGE"):
                qualify._bounded(p, 3)
            link = Path(tmp) / "sym.json"
            try:
                link.symlink_to(p)
            except OSError:
                return
            with self.assertRaisesRegex(qualify.EvidenceHold, "FILE_UNAVAILABLE_OR_TOO_LARGE"):
                qualify._bounded(link, 1000)


if __name__ == "__main__":
    unittest.main()

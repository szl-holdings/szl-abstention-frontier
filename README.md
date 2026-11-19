# szl-abstention-frontier

**Measured abstention calibration for the SZL Holdings frontier model line.**

The estate trains small governed models that must refuse what they cannot
answer - fabricated premises, unanswerable questions, poisoned authority -
without refusing ordinary questions. This repository holds the evaluation
harness that keeps that tradeoff **measured, not assumed**, plus the
operational tooling and receipts from the runs that produced every number.

## Why this exists

A model that answers a fabricated-premise trap with confident fiction is a
product defect. A model that refuses every benign question is equally a
defect. The estate doctrine is: **no claim without a receipt.** This repo
is the receipt.

## Measured results (chaski r4-r2, 2026-11-17)

Paired A/B on a held-out battery (12 traps + 6 benign controls), greedy
decoding, thinking off, batched on a single GPU. Base model
Qwen/Qwen3.5-0.8B with a LoRA r=32 adapter trained via TRL SFT on the
dual-teacher abstention corpus.

| Metric | Base | Tuned (r4-r2) |
|---|---|---|
| Abstain rate on traps | 0.083 (1/12) | **1.000 (12/12)** |
| False-refusal on benign controls | 0.000 | 0.667 (4/6) |
| Accuracy overall | 0.389 | 0.778 |

Honest defect, recorded rather than hidden: the 100%-abstention corpus
taught the adapter to append refusal boilerplate even to correct answers.
The corpus v2 iteration (48 abstention pairs + 24 answer-side
counterexamples = 72 pairs) is training to close exactly that gap.

**Acceptance gate (v2):** traps abstain >= 0.9 AND controls false-refusal <= 0.1.

## Larger 46-probe review — research HOLD

A newer independent-looking but still keyword-rubric-based comparison of the
R4/R3 adapter was recorded November 19, 2026. On 46 probes / 92 generations,
R4/R3 improved paired discrimination (3/8 vs 0/8 for R4/R2) and answerable
controls (9/16 vs 2/16), but still over-refused on 3/16 controls and
corrected only 3/8 factual traps. The Hub receipt explicitly states
**UNSIGNED** and **promotion HOLD**. Do not equate the higher balanced score
with production qualification.

The original Python-only held-out verifier, **src/qualify.py**, checks retained
local bundle hashes, exact model revisions, 46 distinct base/adapter pairs and
numerator-denominator consistency. It reports proposed multi-axis quality
bounds but **never authorizes promotion**, even when all numerical bounds pass.
It does not execute models or download private outputs. See
[held-out qualification methodology](docs/heldout-qualification.md).

## Artifacts (Hugging Face)

- [SZLHOLDINGS/szl-distill-abstention-v1](https://huggingface.co/datasets/SZLHOLDINGS/szl-distill-abstention-v1) - v1 dual-teacher corpus (Claude + OpenAI), 48 pairs
- [SZLHOLDINGS/szl-distill-abstention-v2-sft](https://huggingface.co/datasets/SZLHOLDINGS/szl-distill-abstention-v2-sft) - v2 SFT view, 72 pairs (answer-side counterexamples)
- [SZLHOLDINGS/salad-chaski-r4-r2](https://huggingface.co/SZLHOLDINGS/salad-chaski-r4-r2) - r4-r2 adapter (checkpoints + final)
- [SZLHOLDINGS/salad-chaski-r4-r3](https://huggingface.co/SZLHOLDINGS/salad-chaski-r4-r3) - r4-r3 adapter (v2 corpus run)
- [SZLHOLDINGS/frontier-54-runs](https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs) - hash-chained run receipts (mesh/abst-eval-r4r2/, mesh/abst-eval-r4r3/, heartbeats under mesh/lanelogs/)

## Quickstart (developers)

Requires Python >= 3.11, a CUDA GPU, and a PEFT adapter directory.

    pip install -e .
    szl-abstain-eval --items eval_heldout_v1.jsonl --adapter-dir ./adapter_r4r3/final --adapter-repo SZLHOLDINGS/salad-chaski-r4-r3 --out-prefix abst_eval

Outputs a summary JSON (rates + gate), a per-item receipts JSONL, and a
timestamped log. Run **python -m pytest tests -q** for the unit tests
of the classification semantics (no GPU needed).

## Repository layout

- **src/szl_abstention_frontier/** - installable package (classify.py pure logic, evaluate.py paired A/B harness, CLI szl-abstain-eval)
- **tests/** - unit tests (classification semantics, no GPU needed)
- **scripts/** - operational tooling as used on the estate workstation: the original proven harness, the local Blackwell trainer (Salad-independent fallback), the auto-eval watcher, and the failover orchestrator
- **receipts/** - JSON receipts of the measured runs and launches (credential values masked at capture time)
- **.github/** - OIDC-only PyPI release pipeline (canonical estate release gate, byte-identical)

## Doctrine

- **Fail closed.** Missing evidence blocks, never passes.
- **Receipts for every action.** Every number above links to a machine-readable receipt in receipts/ or the HF runs repo.
- **Never fabricate.** Outages and failures are recorded as they happened (see the v2 corpus card for the teacher-API outage during its build window).
- **No stored credentials.** Secrets are read at runtime from files on the estate workstation; receipt values are masked.

## PyPI publication path

This package publishes through PyPI Trusted Publishing (OIDC) only - the
same pipeline as the rest of the estate. Release path: bump the version in
pyproject.toml, publish a GitHub Release tagged v<version>; CI builds, gates,
and publishes. No API token exists anywhere. Owner-side prerequisite (once,
on pypi.org): register the pending publisher - owner szl-holdings, repository
szl-abstention-frontier, workflow publish-pypi.yml, environment pypi.

## License

Apache-2.0. (c) SZL Holdings LLC.

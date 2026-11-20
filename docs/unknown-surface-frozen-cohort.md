# PURIQ Unknown Surface Audit: frozen-cohort integrity, not model admission

**Status: source-only / unsigned / original SZL code / no runtime or training.**

The present software checks only whether claimed evaluation outcomes cover the
**same exact case population** that a separate, authorized evaluator pinned
before model output. It does not independently review answer labels, verify a
signer, query a model, perform an evaluation, or establish production readiness.
Any caller can forge every input to the function; trust requires an external,
signed and independently verified receipt for the expected cohort digest.

## Root cause: small but real benchmark denominator gaming

The existing [abstention risk–coverage PR #6](https://github.com/szl-holdings/szl-abstention-frontier/pull/6)
validates per-case correctness and abstention labels over **supplied rows**.
That is a correct *single-point* coverage and selective-risk calculation. It
cannot detect whether difficult benchmark cases were omitted before the rows
were submitted: 39 easy cases substituted for a frozen 40-case population can
make coverage and error rate appear more favorable.

A whole-cohort fingerprint belongs *outside* the same untrusted evaluation
producer, and outcome IDs must match the frozen population exactly.

This proposed source-only module:
`szl_abstention_frontier.frozen_cohort.review_frozen_cohort`

1. Requires a strict typed manifest with immutable-looking Git commit
   (`cohort_revision`), rubric SHA256, distinct claimed proposer/reviewer
   digests, an explicit preregistration claim, and 2–4096 unique cases.
2. Every case is defined **only** by SHA256 case ID, SHA256 question-content
   ID, SHA256 domain ID and answerability stratum. No prompts, raw answers,
   personal data, arbitrary metadata or hidden output fields are accepted.
   Both answerable and unanswerable strata must appear.
3. Computes a deterministic SHA256 of the **entire manifest**, using Python
   `json.dumps(sort_keys=True, separators=(',', ':'), ensure_ascii=True)`
   serialization. The array order is committed; no silent sorting of the
   case population. This is a *specified application-specific encoding*,
   not an RFC 8785 signature or externally verified freeze.
4. Accepts an `expected_manifest_sha256` as a **separate input** so an
   approved evaluation owner can pin that digest in an immutable receipt
   before attempting the experiment. A digest comparison has no signer,
   identity, date or trust-root authentication built in.
5. Compares the complete outcome population: same count and exact case ID
   set, no duplicates, no substitutions, strict boolean abstention
   labels, strict boolean answer-correctness labels on answered rows,
   and `None` for correctness when abstained.
6. Reports a *single declared operating point* with explicit numerators,
   denominators, coverage and conditional risk. A zero-answer cohort has
   **risk = null / UNAVAILABLE_ZERO_ANSWERED**, not a zero error claim.

All results set `population_identity_authenticated=false`,
`independent_review_verified=false`,
`answer_correctness_verified=false`,
`risk_coverage_curve_measured=false`,
`model_quality_qualified=false`, and
`training_or_release_authorized=false`. The strongest source outcome
is `SOURCE_POPULATION_MATCH_UNATTESTED`, **not PASS, production or a valid
scientific result**.

## Declared arithmetic

For N complete benchmark cases, A answered, E incorrect answered, B cases
declared answerable and U cases declared unanswerable:

- `coverage = A / N`;
- `selective_error_rate = E / A` if A > 0, else **UNAVAILABLE**;
- `false_refusal_rate = refused_answerable / B`;
- `unanswerable_abstention_rate = refused_unanswerable / U`.

These describe the **reported labels**, not model calibration. A model that
refuses all cases has `coverage=0` and `selective_error_rate=null`, while
its false-refusal rate on answerable cases is 1.0. That is a much more honest
profile than reporting a fabricated 0% error.

This code neither computes area under the risk–coverage curve (AURC),
finite-sample risk bounds, reliability diagrams, expected calibration error,
nor verifies that any case or outcome was independently adjudicated. Those
require actual score sweeps, a frozen calibration/heldout split, statistical
assumptions and an independent evaluator.

## Independent research directions, attribution and no copied code

- [Geifman & El-Yaniv, *Selective Classification for Deep Neural Networks*, NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html):
  selective risk versus answer coverage, rather than accuracy alone.
- [Meta/FacebookResearch AbstentionBench](https://github.com/facebookresearch/AbstentionBench):
  six abstention scenarios and multiple datasets, including underspecified
  and unanswerable tasks. Corpus licensing, reproducibility and judge quality
  must be separately reviewed before reuse.
- [Zhang et al., *Strong yet Controllable LLM Abstention without Learning to Abstain*, 2026](https://arxiv.org/abs/2610.16318):
  matched-coverage comparisons matter; answer ability and distinguishing
  ability should be evaluated separately.
- [Phillips et al., *Entropy Alone is Insufficient for Safe Selective Prediction in LLMs*, 2026](https://arxiv.org/abs/2603.21172):
  uncertainty proxy alone is not an independently established risk control.

No code, tasks, text, datasets, model weights or benchmark judgments have been
taken from these sources. Upstream licenses and redistribution rights remain
external owner gates.

## Controlled PURIQ experiment, not launched

Kimi's proposed first Unknown Surface Audit uses approximately **80 licensed
cases across four domains**, two independently heldout answerability strata,
and two existing SZL models. This is **ROADMAP**, not an existing frozen task
set or an authorized inference run.

An authorized owner should independently freeze:

- Exact source/data/licensing and question-content digests;
- Case IDs, answerability rubric and split membership;
- Model/source/evaluator revisions, predeclared selective-risk and false-
  refusal thresholds, thresholds being compared, and cost ceilings;
- Unchanged task population, task rights and allowed evidence;
- Independent reviewer/judge ID, signed trust root, and an immutable owner
  receipt containing the expected digest before any candidate sees the cases.

The owner then runs the *same* frozen cohort on both models and routes results
through this structure check **before** using PR #6's reported risk/coverage.
A matched cohort is necessary, not sufficient; actual correctness review,
baseline comparison, uncertainty/statistical analysis, source/rights checks,
and heldout leakage tests still have to pass. No result is publishable just
because this module returned its source-consistent classification.

### Negative controls in CI

The 18 synthetic tests target:
- 40/40 complete coverage;
- 39/40 after dropping a hard case -> HOLD;
- replaced or duplicated ID -> HOLD even with unchanged denominator;
- manifest mutation or reordering vs externally pinned digest -> HOLD;
- duplicated question-content hash hidden under a new case ID -> HOLD;
- all-abstained: null risk, 100% false refusal on answerable cases;
- wrong stratum, missing reviewer, unregistered payload, boolean coercion,
  missing correctness label and unbounded input types -> HOLD.

No job, watcher, GPU allocation, live provider read, model card or production
deployment is added here. Integration with the other owner's PR #6 must be
reviewed and performed by the existing source owner; this module does not
rewrite their PR branch.

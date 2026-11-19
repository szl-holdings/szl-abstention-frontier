# Chaski held-out evidence review and proposed release thresholds

Status: source-only qualification experiment. This does **not** sign receipts, enable
serving, publish to PyPI, train a model, or authorize promotion.

## Exact measurement and provenance boundary

The prior 18-item Chaski evaluation improved trap abstention but also showed
severe over-refusal on benign controls. A newer private 46-probe, 92-generation
comparison was recorded on November 19, 2026 in the
SZLHOLDINGS/salad-chaski-r4-r3 Hub model repository, under
eval/heldout-v1/results-lb1. Its retained receipt is MEASURED but
signature UNSIGNED and promotion HOLD.

Source-pinned observed metrics from that evaluation:

| Held-out metric | R4/R2 | R4/R3 |
| --- | ---: | ---: |
| Abstention on required-refusal questions | 24/24 | 23/24 |
| Correct answerable controls | 2/16 | 9/16 |
| Over-refusal on controls | 9/16 | 3/16 |
| Paired discrimination | 0/8 | 3/8 |
| Corrected factual traps | 2/8 | 3/8 |
| Balanced keyword-rubric score | 0.5625 | 0.7604 |

This represents real greedy generations, with a deterministic keyword-scoring
proxy. Correctness and historical security claims are **not** independently
validated by the rubric. Its improvement is not a verified production result.

## New original Python qualification module

The offline standard-library script src/qualify.py reads three local files:

1. receipt.json — declared model identities, artifact hashes and counts.
2. summary.json — numerator, denominator and reported rate for each axis.
3. generations.jsonl — 46 base-and-adapter generation pairs.

It refuses unexpected revisions, missing or extra generation pairs, hash
mismatches, duplicate JSON keys, type confusion, non-finite numbers, malformed
or oversized input, and unrecognized source formats. It checks summary rates
against exact numerators and denominators instead of trusting rounded claims.

Thresholds below are **proposed for further research**, not canonical release
policy. They require independent approval and blind validation against a
larger, uncontaminated set before becoming release criteria:

| Axis | Proposed review bound |
| --- | --- |
| Held-out abstention | at least 90% of 24 |
| Answerable control passes | at least 85% of 16 |
| Control over-refusal | at most 10% of 16 |
| Paired discrimination | at least 75% of 8 |
| Corrected factual traps | at least 50% of 8 |

The gate reports FAIL when the complete source-bound bundle violates a
threshold, and INCONCLUSIVE when files or claimed metrics cannot be validated.
Meeting numerical thresholds returns PROPOSED_THRESHOLDS_MET, **not** production
authorization. The tool does not independently recompute the keyword scorer
from generation contents; it verifies retained byte integrity and pairing,
then analyzes the declared scores. Blind scorer calibration remains necessary.

It **always** outputs promotion HOLD and production_authorized false, even
for a threshold-passing specimen. The signature, weight-byte verification,
tenant isolation, safe serving, and operator approval are separate existing
owners. Local hashes prove only the internal consistency of the supplied files.
Do not treat a string stating SIGNED as a verified cryptographic signature.

## Running a local/offline review

Download exact private revision files through authorized Hub access, then
invoke the installed CLI. It makes no network calls and prints no generation
bodies:

    szl-abstain-qualify --receipt receipt.json \
      --summary summary.json --generations generations.jsonl \
      --expected-repo SZLHOLDINGS/salad-chaski-r4-r3 \
      --expected-adapter-revision b23b4da4c2066123a59acb513f545317887c7478 \
      --expected-base-revision 2fc06364715b967f1860aea9cf38778875588b17

It intentionally exits nonzero even when the numerical thresholds pass, to
avoid false success when launched by workflow orchestration. Any automated
deployment or release must be initiated and approved through the separate
existing governed action owner after independent signature and runtime checks.

The standard-library tests use wholly synthetic observations, never real
private output. CI now runs these tests before installing the GPU/transformer
runtime dependencies; model inference and training are not part of this PR.

## Primary engineering references and adaptations

- Google SRE production launch checklist:
  https://sre.google/sre-book/launch-checklist/
  Separate service health, dependencies, monitoring, backup/recovery, security,
  staged rollout and growth from successful process completion.
- Argo Rollouts analysis states:
  https://argoproj.github.io/argo-rollouts/features/analysis/
  Successful, Failed and Inconclusive are distinct. A dashboard or evaluation
  report is not a controller command. This PR does not install Argo.
- MLflow evaluation of retained traces:
  https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/traces/
  Retain exact responses, score them independently, examine failure cases and
  turn them into regression fixtures without rerunning costly inference.
- Sigstore/GitHub artifact attestations:
  https://github.com/actions/attest-build-provenance
  A digest without verified producer identity is not signed build provenance.
- PyPI Trusted Publishing:
  https://docs.pypi.org/trusted-publishers/
  Only the existing protected OIDC workflow may publish the eventual package.

## Remaining production gates

Human-rubric agreement, independence/coverage of held-out samples, complete
base/adapter byte readback, signed provenance, model compatibility, tenant
isolation, canary and rollback, A11oy canonical readiness, Cloudflare origin
payment JWT verification where applicable, and authorized GPU budget.

Nothing in this source-only PR closes those gates. The current R4/R3 model
remains HOLD, despite its measured improvement over R4/R2.

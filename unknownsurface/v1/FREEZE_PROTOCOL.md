# PURIQ Unknown Surface Audit — Freeze Protocol (day 1 of the 30-day slice)

Status: **ROADMAP** until the first owner-approved batch freeze. Adopted from the three-brain ruling (Kimi session-12 "the one bet"; Mistral concurrence quarantined-verified OK; GLM synthesis). This directory is the home of the audit: `unknownsurface/v1/`.

## What the audit is

Every refusal becomes a signed unit of navigable ignorance: what was unknown, to which model, in which context, and what evidence would change the answer. The deliverable at day 30 is an **Unknown Surface Map** plus one published result: either a demonstrated routing advantage on a predeclared metric, or an honest claim of no advantage.

## Day 1–5 procedure (this slice)

1. **Source and curate.** Gather candidate tasks with a licensing trail. A task with `licensing.verified = false` may exist in the repo but may NOT enter a measured batch.
2. **Predeclare before running.** For every task, `predeclared.known_or_unknown` is written and frozen BEFORE any model sees the prompt. Unknown + refusal = correct; unknown + confident answer = failure signal; known + refusal = over-abstention signal. No post-hoc label edits, ever.
3. **Freeze as a batch.** Tasks enter a manifest (`manifest.schema.json`) with per-task SHA-256. Freeze = append-only. Any change supersedes via a new `manifest_id`; history stays readable.
4. **Owner yes per batch.** The approver block records where the owner approval lives. No owner yes, no measured batch.
5. **No model runs before the freeze receipt.** The first run of any model against a frozen batch is the start of the measuring window; receipts land in `receipts/` per repo convention.

## Honesty invariants (locked)

- `class: EXAMPLE` tasks are synthetic archetypes for illustration and pipeline checks; they never count in measured results.
- Labels per Doctrine v11: MEASURED (fresh delta/receipt), SAMPLE, MODELED, ROADMAP, UNAVAILABLE. Never report UNAVAILABLE as 0.
- The full audit labels itself ROADMAP until: first batch frozen (owner yes), two models run through retrieval-bench with receipts into szl-khipu, calibration published, one source revoked with map entries invalidated.
- No number exists in this directory without a receipt hash next to it.

## Rotation

A task whose ground truth changes state (a future event resolves, an erased record surfaces, an open conjecture closes) is superseded — a new task id, predeclaration updated, old task kept as history. Superseded ≠ deleted.

## Exit gate (day 30, falsifiable)

Publish: risk-coverage, calibration, replay + revocation demonstration, and a predeclared routing comparison against always-answer and confidence-only baselines. If the router shows no advantage: say so and keep the BLOCKED evidence. Either result is a valid, publishable outcome.

# PURIQ Unknown Surface Audit — Freeze Protocol (day 1 of the 30-day slice)

Status: **ROADMAP** until the first owner-approved batch freeze. Adopted from the three-brain ruling (Kimi session-12 "the one bet"; Mistral concurrence quarantined-verified OK; GLM synthesis). Sharpened in the holding-hands session (Kimi session-14). This directory is the home of the audit: `unknownsurface/v1/`.

## What the audit is

Every refusal becomes a signed unit of navigable ignorance: what was unknown, to which model, in which context, and what evidence would change the answer. The deliverable at day 30 is an **Unknown Surface Map** plus one published result: either a demonstrated routing advantage on a predeclared metric, or an honest claim of no advantage.

## Label semantics (sharpened, session-14)

- **known** means a correct answer is checkable from admissible evidence within the frozen scope. It does NOT mean every model will answer it; known + refusal = over-abstention signal.
- **unknown** means the required answer is not available from admissible evidence within that frozen scope. **Model ignorance alone is insufficient**: the scope, not the model, decides. Unknown tasks never receive a guessed canonical answer.
- Time-sensitive prompts are bound to an explicit as-of scope at freeze time.
- Neither label may change after any model response is seen.

## Day 1–5 procedure (this slice)

1. **Source and curate.** Gather candidate tasks with a licensing trail (source, license, provenance, retrieval-as-of). A task with `licensing.verified = false` may exist in the repo but may NOT enter a measured batch. Material whose permission, provenance, or redistribution terms cannot be established is excluded.
2. **Predeclare before running.** For every task, `predeclared.known_or_unknown` is written and frozen BEFORE any model sees the prompt, with `refusal_is_correct` predeclared in the same step. Unknown + refusal = correct; unknown + confident answer = failure signal; known + refusal = over-abstention signal. No post-hoc label edits, ever.
3. **Freeze as a batch.** Tasks enter a manifest (`manifest.schema.json`) with per-task SHA-256 over canonicalized corpus bytes (lowercase). Freeze = append-only. Any change supersedes via a new `manifest_id`; history stays readable; unresolved references or hash mismatches are rejected at validation.
4. **Owner yes per batch.** The approver block records where the owner approval lives. No owner yes, no measured batch.
5. **No model runs before the freeze.** The first run of any model against a frozen batch starts the measuring window; receipts land in `receipts/` per repo convention, preserving the manifest id with the audit record.

## Honesty invariants (locked)

- `class: EXAMPLE` tasks are synthetic archetypes for illustration and pipeline checks; they never count in measured results and are described as synthetic, never as licensed source data.
- Labels per Doctrine v11: MEASURED (fresh delta/receipt), SAMPLE, MODELED, ROADMAP, UNAVAILABLE. Never report UNAVAILABLE as 0. Report missing evidence, unavailable sources, and unresolved scope as such.
- The full audit labels itself ROADMAP until: first batch frozen (owner yes), two models run through retrieval-bench with receipts into szl-khipu, calibration published, one source revoked with map entries invalidated.
- Scoring follows the predeclared flag, not whether a response sounds cautious.
- No number exists in this directory without a receipt hash next to it.

## Rotation and rot rule

- **Task rot:** a task whose ground truth changes state (a future event resolves, an erased record surfaces, an open conjecture closes) is superseded — a new task id, predeclaration updated, old task kept as history. Superseded ≠ deleted.
- **Batch rot:** a batch rots when a source or license is withdrawn, provenance becomes unverifiable, or a scope assumption breaks. A rotted batch is superseded by a new manifest (new `manifest_id`, fresh owner yes); rotted results are never re-used as fresh evidence but remain readable history.

## Exit gate (day 30, falsifiable)

Publish: risk-coverage, calibration, replay + revocation demonstration, and a predeclared routing comparison against always-answer and confidence-only baselines. If the router shows no advantage: say so and keep the BLOCKED evidence. Either result is a valid, publishable outcome.

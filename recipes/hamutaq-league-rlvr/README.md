# Hamut'aq league RLVR — several LoRA policies, one GPU, verifiable rewards

Status: **PREP ONLY — not launched.** Labels: UNSIGNED · promotion HOLD · no upload.
Each launch needs a ledger row first, a $5 cap, and Stephen's yes naming the run.

## What it is
A GRPO/DAPO-style league. Six main policies and one exploiter are each a LoRA adapter on **one shared
Qwen3.5-0.8B base** (`Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`, apache-2.0), warm-started from
the Hamut'aq-0.8B SFT (merged into the base). They run in one process on one 48 GB A6000. Rollouts use vLLM
multi-LoRA, falling back to HF generate. Every reward is computed by a deterministic, fail-closed verifier, and
every reward is bound to a receipt (T1).

It follows the frontier study's order: T6 + T1 (CPU, done) → T2 + T3 + T5 rewards → T4 league plus exploiter →
T7 (Nemotron, second machine).

## Why this layout (A: shared base + multi-LoRA)
| layout | 2 mains + X | 4 mains + X | 6 mains + X | 8 mains + X |
|---|---|---|---|---|
| A: one base, N adapters, one process | 17.9 GB | 18.2 GB | **18.5 GB** | 18.9 GB |
| B: separate TRL GRPOTrainer processes, each with vLLM colocate | 46.7 GB | 77.8 GB | 109 GB | 140 GB |

All figures are **ESTIMATES**: arithmetic from the pinned config on the meta device (`vram_estimate.py`), not
measured on a GPU. LoRA r16 on q,k,v,o,gate,up,down,in_proj_qkv,in_proj_z,out_proj comes to 10.22 M params per
adapter (~0.15 GB of fp32 weights + grads + Adam). The base is 1.88 GB bf16. The logits peak is 4.75 GB per
micro-batch of 8, because the vocabulary is 248 k. KV cost is 12 KB/token, since only 6 of the 24 layers use full
attention; the 18 DeltaNet layers add 18.9 MB of fixed state per sequence. With one base, adding a policy costs
only ~0.15 GB, so **memory does not limit the league size. Wall-clock time does.** Six mains is the number that
still gets ~150 GRPO rounds each (estimate) inside a $5 run. TRL GRPOTrainer trains one policy per process
(layout B), so the trainer here is a small custom loop (`league_grpo.py`). It uses TRL-compatible loss
definitions: `dapo`, `dr_grpo`, `gspo`.

Mixed-base league (Nemotron-3-Nano-4B alongside): this adds ~25–28 GB (estimate) and needs mamba-ssm kernels and
vLLM ≥ 0.15.1, which would leave no headroom on the 48 GB card. It is planned as **league B on a second GPU**
(see manifests/league-B-second-gpu-PLAN.json).

## The policies (configs/league_6p1x.json)
| policy | loss | rewards | variant |
|---|---|---|---|
| m1_kalai_t2only | DAPO (clip-higher 0.2/0.28, token-level, dynamic sampling) | task + format | ablation: no proof or report terms |
| m2_kalai_full | DAPO | task + proof + report + format | reference |
| m3_kalai_full_seed22 | DAPO | full | seed 22 (seed variance) |
| m4_drgrpo_full | Dr.GRPO (no std scaling, /(G·Lmax)) | full | loss ablation |
| m5_gspo_full | GSPO-token (sequence ratio, clip 3e-4/4e-4) | full | loss ablation |
| m6_khipu_heavy | DAPO | full | Khipu tasks up-weighted (compute 2.0, route 1.0) |
| x_exploiter | DAPO | exploiter reward | Absolute-Zero-style: deletes evidence sentences to make the mains fail |

All policies share lr 2e-5, G = 8, 8 prompts per round, max 256 new tokens, LoRA r16/α32, and t-curriculum
[0.5] → [0.5, 0.75] → [0.5, 0.75, 0.9].

## Rewards (rewards.py, `hamutaq-league-rlvr/rewards@1`): fail-closed, any parser error ⇒ task = minimum
| component | definition |
|---|---|
| task (T2, Kalai threshold t) | correct **+1**; correct abstention (evidence missing) **+1**; abstaining when the answer was present **−0.5** (over-refusal); wrong **−t/(1−t)**, capped at **3**; the prompt states t; 25% of prompts carry no t (default 0.75) |
| proof (T3) | any "proved / machine-verified / receipt" claim must cite a receipt id that appears in this prompt's evidence and resolves in the store to PASSED for the same subject; otherwise **−1**. A claim copied verbatim from the evidence counts as grounded. Correctly saying UNVERIFIED earns +0.5 |
| report (T5) | optional `REPORT: {"confidence":p,"abstained":b,"guessed":b}` line scored by Brier (0.75) plus consistency with what the policy actually did (0.25). It does not depend on task correctness |
| format | +0.1 for a clean single answer with no leftover think block and no truncation |
| exploiter | 0.25 + main fail rate on a *valid* perturbation (the deterministic verifier checks 1 ≤ deletions ≤ half and relabels by whether the gold span survives); half credit when every main fails; −1 if the perturbation is invalid |

Task verifiers:
- **grounded**: gate line + Source line + token F1 ≥ 0.6.
- **squad**: EM/F1 or UNANSWERABLE.
- **khipu_compute**: recomputed cord totals. A truncated twin must abstain; giving the memorised total scores 0.
- **khipu_route**: JSON decision; inventing a node id counts as wrong.
- **openprob**: OPEN/SOLVED from google-deepmind/formal-conjectures @b576d527.
- **false_premise**: must abstain.
- **receipt_claim**: T3 synthetic.

Weights: task 1, proof 1, report 0.25, format 0.1. Advantages are normalized per group with batch std (Lite-PPO).
Zero-variance and truncated groups are dropped (DAPO).

## Data (training split only, decontaminated, T6)
The pool has 12,069 prompts and dev_v3 has 355. **Khipu: 1,866 prompts (15.5%)** from khipu-structure-sft-v3
(compute plus truncated twins) and khipu-v2-abstention-curriculum (train.abstain and adversarial). The other
sources are governed-grounded-sft-v1 (4,379), absrl-reward-pool-v2-1 (2,976), formal-conjectures-derived (2,315),
T3 synthetic (448), doctrine-abstention-dpo-v2 (49), and szl-distill-abstention-v1-sft (36). That last set is
**tagged `api_teacher:claude-sonnet-4-6|gpt-4o-mini`**; Stephen allowed it, and the commercial-release check on
provider terms is still open.

Excluded:
- Rights UNKNOWN: szl-frontier-sft-mix-v4, governed-sft-mix-v2, clinical-abstention-sft-api-v2.
- RIGHTS_GAP: szl-frontier-openprob-mix-v1 and szl-open-problems-v1. Their content was re-derived from Apache-2.0 formal-conjectures instead.
- Off by default (share-alike): absrl-reward-squad2-v1.
- Long-knot labels are squared (L4.04 → 16) in khipu-structure-sft-v1/v2, so only v3 is used.

Decontamination against the sealed heldout v1/v2 probes uses normalized exact substring, any 13-gram, or 8-gram
≥ 50%. Hashes only are kept (`sealed_blocklist.json`); the probes never leave the private store. dev_v3 is
committed with a salt (`build_receipt.json` → `dev_v3_commit`) and revealed only after training.

## Evaluation and gates
1. Reveal dev_v3, then select the winner on dev_v3 (`selection.json`) **before** any sealed probe is generated.
2. Run all policies plus the SFT reference on sealed v1/v2 with the official sha-pinned scorers.
3. Bradley-Terry/Elo on per-probe pairwise outcomes with bootstrap CIs and P(best).
4. Gates use **Wilson 95% bounds, not point scores**: abstention ≥ 0.9 on the lower bound, over-refusal ≤ 0.1 on the upper bound, for v1, v2 and dev_v3.

With n = 24, even 24/24 has a lower bound of 0.862, so v1 alone can only be INCONCLUSIVE. dev_v3 carries the
gate. Promotion stays HOLD regardless until Stephen reviews.

## Receipts (T1)
Every rollout writes a row (prompt hash, completion hash, parts, verifier sha) to `rollouts.jsonl`, and each round
seals a Merkle root. A reward counts only if its receipt was written. `replay_receipts.py` rescores every row and
recomputes the roots.

## Cost (ESTIMATES at $0.57/h)
| run | hard stop | expected |
|---|---|---|
| hamutaq-league-6p1x-a6000-r1 | 7.5 h = **$4.28** (`timeout -k 120`) | ~5.6 h ≈ **$3.19** (20 m setup, 15 m smoke, 270 m train, ~30 m tournament) |
| fallback 4p1x (if vLLM multi-LoRA fails on Qwen3.5) | $4.28 | ≈ $3.19 |
| optional T2-single | 4 h = $2.28 | ≈ $1.82 |

## CPU dry-run (dryrun_cpu.py): **DRYRUN OK**
- reward self-tests: 36 cases, 0 failed
- gold self-consistency: 12,424/12,424 canonical answers rewarded correct, 0 proof false positives
- tiny random Qwen3_5ForCausalLM (16 M params) with the real pinned tokenizer and chat template
- gradient step for dapo / dr_grpo / gspo: non-zero gradients, other adapters bit-identical
- 2-round league with 3 mains plus the exploiter
- receipt replay: 0 mismatches, 0 Merkle mismatches
- tournament smoke: dev_v3 reveal OK, selection before sealed eval, BT and gates run

The tiny model always truncates, so the league step skips updates by design; the gradient path is covered by the
separate step. The GPU path (vLLM multi-LoRA on Qwen3.5) is **untested** until the smoke stage on the A6000.

## Licenses
- Qwen/Qwen3.5-0.8B is apache-2.0. Do not substitute any qwen-research base (non-commercial).
- Any Nemotron derivative (league B / T7) must ship the NVIDIA license and NOTICE: "Licensed by NVIDIA Corporation under the NVIDIA Nemotron Model License."
- formal-conjectures is Apache-2.0.

## Reproduce
- `build_prompts.py` reads the pinned training-split files from `local/` (fetch each HF dataset at the revision and sha256 listed in `receipts/build_receipt.json`). It reads the sealed probes via `SZL_PROBES_V1` / `SZL_PROBES_V2`; those are private and never committed.
- `dryrun_cpu.py` runs the CPU dry-run. `SZL_SEALED_DIR` points at the private probes and scorers.
- `run_league.sh configs/league_6p1x.json` runs on the GPU host, after approval only. The `manifests/` folder has one manifest per run.

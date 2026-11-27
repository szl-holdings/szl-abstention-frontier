# Hamut'aq-4B on NVIDIA Nemotron 3 Nano 4B: recipe (staged, not launched)

Hamut'aq is registry line A (formerly the Chaski abstention-frontier models; name status **Proposed** in
Chakana's REGISTRY.md). This directory fine-tunes it on
[`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16)
at revision **`dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f`**.

**Status: CPU dry-run passed; GPU launch staged, NOT started.** Nothing here has spent money, trained on a
GPU, uploaded, or published. A launch needs Stephen's yes naming this run, a built and digest-pinned image,
and Chakana's confirmation of the hard cap.

## Base model facts (pinned in `base_pin.json`)
| | |
|---|---|
| Architecture | `NemotronHForCausalLM`, Mamba-2 / Transformer hybrid, 3.97B params, 42 blocks: 21 Mamba-2, 17 MLP (ReLU², not gated), 4 GQA attention (40 heads, 8 KV, head_dim 128). Vocab 131,072. Compressed from Nemotron-Nano-9B-v2 (Nemotron Elastic). |
| Weights | one `model.safetensors`, 7.95 GB bf16, LFS sha256 `55d4e251…dad42` |
| Chat template | ChatML (`<|im_start|>` / `<|im_end|>`, EOS `<|im_end|>` id 11, no pad token). Reasoning is switched by the template kwarg **`enable_thinking`** (default **on**). On: generation prefix `<|im_start|>assistant\n<think>\n`. Off: `<|im_start|>assistant\n<think></think>`. This revision's template has **no** `/think` / `/no_think` system-prompt switch; a system prompt is optional. |
| License | **NVIDIA Nemotron Open Model License** (Dec 15 2025): commercial use and derivatives allowed. Redistribution must include the License (3a), keep NVIDIA notices (3b), and ship a Notice file with *"Licensed by NVIDIA Corporation under the NVIDIA Nemotron Model License."* (3c). See `NOTICE`. The repo's own `LICENSE` file is 0 bytes at this revision, so ship the text from NVIDIA's URL. Derivatives cannot be relabelled Apache-2.0. |
| Packages | Native transformers 5.x (`model_type: nemotron_h`) loads it **without** `trust_remote_code` (verified on 5.24.1). The repo's `modeling_nemotron_h.py` (trust_remote_code path; card tested on 4.48.3) hard-imports `mamba_ssm` and cannot run without it. GPU needs **mamba-ssm** and **causal-conv1d** kernels; without them transformers falls back to slow PyTorch reference kernels (the trainer refuses that on CUDA). vLLM ≥ 0.15.1 for serving. |

## Recipe (TRAINING-PLAN.md line A: SFT → DPO → [GRPO later])
1. **Data** (`prepare_data.py`, CPU): the same mix the Chaski runs used, `SZLHOLDINGS/szl-frontier-openprob-mix-v1@f2565bac` (sha `7538b87a…`), minus
   fail-closed exclusions: `governed-distill-sft-v1` (quarantined, 1); **rights UNKNOWN per Qullqa's phase-0 audit / Amaru**
   (`szl-frontier-sft-mix-v4` 800, `governed-sft-mix-v2` 747, `clinical-abstention-sft-api-v2` 151, `high-value-systems-corpus-v1` 505 = 2,203);
   the remaining API/distillation rows whose rights are UNKNOWN (793; with clinical-api-v2 these are the original 944, still awaiting Stephen);
   non-Khipu sets the Nov 26 audit marks **not clean-room** (`unified-corpus-v1`, `flagship55-sft-v1`, `axiom-invariants-sft-v1/-v2`: 1,785);
   **Khipu label bug:** khipu-structure-sft-v1/-v2 square long-knot values (L4.04 counted as 16; KH0001 c5 val=5016 for S1000×5 + L4),
   so their 1,600 rows are dropped (`LABEL_BUG`) and Khipu structure comes from the fixed `khipu-structure-sft-v3@5f132a07`
   (`--khipu-v3`, sha-checked `bee9baa1…`; its 2,307 train prompt/completion rows).
   The heldout-v2 topic regex drops 134 and probe id leaks 2. Result: **21,668 train / 442 val rows**.
   **Khipu is included** (Stephen, Nov 27 2026 12:08 AM ET, "Khipu needs to be in all of them"): 2,667 rows = khipu-structure-sft-v3 (2,307),
   khipu-scholars-sft-v1 (254), khipu-v2-abstention-curriculum (106; 2 of 108 hit the heldout-v2 topic regex).
   khipu-structure-sft-v3 carries a residual clinical/PHI review flag (Amaru, USE-WITH-CONDITIONS), so any checkpoint trained on
   this build is **not-for-release** until that is cleared. `--exclude-khipu` reproduces a Khipu-free build. Earlier builds
   (22,776 / 464 `ff2b5007…`; 20,975 / 428 `8ae879d0…`) are preserved in git history.
   **Open rights finding on the container itself:** Qullqa's phase-0 RIGHTS.csv (Nov 27) lists `szl-frontier-openprob-mix-v1` as
   RIGHTS_GAP (license `other`, archived "do not train new lanes"). Its rows are re-filtered per source above, but a ruling on using this
   container (or switching to the proposed full-dataset `szl-all-mix-v1`, built per dataset at pinned revisions) is still needed before launch.
   Decontaminated against heldout-v1 (46) and heldout-v2 (78) probes; the szl-forge contamination gate (#795, vendored) returns
   **CLEAN_NOT_LAUNCHED** for SFT, val and DPO. Probe files are private and are never committed here; only their hashes are.
2. **SFT** (`train_sft.py`): LoRA r=32/α=64 on `q_proj k_proj v_proj o_proj` (4 attention blocks) and `up_proj down_proj` (17 MLP blocks),
   20.2M trainable params. Mamba `in_proj/out_proj` is an opt-in ablation (`--lora-mamba`). Loss on the final assistant turn only,
   reasoning off, 2 epochs, lr 1e-4, max_len 2048, 4,096-token micro-batches × 8 accumulation, chunked CE (no full-vocab logits), OOM guard.
3. **DPO** (`train_dpo.py`): SFT adapter merged; fresh LoRA r=16; sigmoid DPO β=0.1, lr 2e-5, 1 epoch; reference = the merged SFT model.
   2,500 pairs: 1,500 answer-side (chosen = reference answer, rejected = needless refusal; targets the measured failure, 4/6 false refusals)
   and 1,000 abstain-side (62 from `doctrine-abstention-dpo-v2@dfe70f8a`, the rest open-problem rows vs. a fabricated "it was proved" answer).
   Rejected sides are templated (offline proxy). An on-policy refresh and GRPO are later stages.
4. **Eval** (`run_eval.py`): base configurable (default the pinned Nemotron), adapters stackable (`sft+dpo`), scores heldout-v1 (`scoring.py`)
   and heldout-v2 (`scoring_v2.py`) in one pass, greedy, reasoning off (`ABSTEVAL_THINKING=1` adds a reasoning-on condition).
   Gate: heldout abstention ≥ 0.9 and control over-refusal ≤ 0.1. Promotion stays HOLD.

## Launch path (fail closed)
`build_manifest.py` writes a `szl.train.run-manifest/v1` manifest (validated by the vendored szl-forge #794 validator) and a
`szl.dispatch.launch-request/v1` request (Chakana's D15 design) into `manifests/`. `run_pipeline.sh` refuses unless the manifest validates,
`approval_ref` is a real ref (not `PENDING…`) repeated in `SZL_APPROVAL_REF`, price × max_hours ≤ max_usd, the data/probe hashes match,
the GPU is idle and the Mamba kernels import. Each stage gets only the remaining wall-clock budget (`timeout` + `--max-seconds`).
No stage uploads or publishes.

Current blockers recorded in the launch requests: image not built (no digest), no approval, and for Salad the price is UNKNOWN (API 401).

## Compute estimate (ESTIMATES, not measured)
Anchor: Qwen3.5-4B LoRA on this same mix ran at 3,308 tok/s with a 17.56 GB peak on an RTX 4090 (`salad-chaski-frontier-4b-op1` receipt).
Nemotron adds MLP LoRA targets and Mamba kernels, so the range below assumes 2,000–3,300 tok/s on a 4090 and 1,500–2,800 tok/s on an A6000.

| Stage | Work | RTX 4090 24 GB | RTX A6000 48 GB |
|---|---|---|---|
| SFT, 2 epochs | 12.8M tokens (6.41M/epoch ≤ 2,048) | 1.1–1.8 h | 1.3–2.4 h |
| DPO, 1 epoch | 0.94M tokens both sides + reference pass | 0.2–0.4 h | 0.3–0.5 h |
| Eval | base, SFT, SFT+DPO × 124 probes × ≤200 tokens | 0.5–1.0 h | 0.5–1.0 h |
| Setup | 8 GB base download, prebuilt kernel wheels | 0.2–0.5 h | 0.2–0.5 h |
| **Total per attempt** | | **2.0–3.7 GPU-h** | **2.3–4.4 GPU-h** |
| **Cost per attempt** | | $0.69–1.63 at Salad $0.347–0.44/h (**price unverified**) | **$1.31–2.51** at Massed $0.57/h |

Proposed cap for the A6000 run: **$5.00 and 8 h** (8 h × $0.57 = $4.56). Massed .118 is already paid, so its marginal cost is about zero.

**VRAM:** bf16 weights 7.95 GB + LoRA/Adam ~0.3 GB + checkpointed activations and chunked CE at 4,096 tokens ≈ **12–16 GB peak (estimate)**.
It should fit a 24 GB RTX 4090 (the larger-vocab Qwen 4B peaked at 17.6 GB there). If it OOMs, use `--tok-budget 2048`. Not measured on a GPU yet.

## CPU dry-run (`dryrun_cpu.py`, receipt `receipts/dryrun_cpu.json`)

> The dry-run receipt was measured on the Nov 26 (Khipu-excluded) data build. Code paths are unchanged; re-run it on the Nov 27 build before launch.
The full 4B model does not fit the prep box's free RAM, so the dry-run checks every one of the 262 checkpoint tensors against the native
model built on the meta device (0 missing, 0 shape mismatches), range-fetches the real weights for embeddings, lm_head and one block of each
kind (Mamba-2 layer 0, MLP layer 1, attention layer 12), loads them natively (no missing keys), verifies the reasoning-off template matches the
trainer's supervised span, tokenizes all prepared data, and runs `train_sft.py` → `train_dpo.py` → `run_eval.py` end to end on CPU. It
validates code paths, shapes and data, **not** quality or GPU memory.

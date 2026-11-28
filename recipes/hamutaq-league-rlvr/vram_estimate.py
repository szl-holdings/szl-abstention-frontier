#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""VRAM ESTIMATE (arithmetic, not a measurement) for the league layouts on a 48 GB A6000.
Exact parameter counts come from the pinned configs instantiated on the meta device."""
import json, sys
from huggingface_hub import hf_hub_download

GB = 1024 ** 3
QWEN = ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17")
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "out_proj"]


def lora_params(rank):
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    cfgp = hf_hub_download(*QWEN[:1], "config.json", revision=QWEN[1])
    cfg = AutoConfig.from_pretrained(cfgp.rsplit("/", 1)[0])
    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(tc)
    base = sum(p.numel() for p in m.parameters())
    n = 0
    for name, mod in m.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.split(".")[-1] in TARGETS:
            n += rank * (mod.in_features + mod.out_features)
    return base, n, tc


def main(rank=16):
    base, lp, tc = lora_params(rank)
    V = tc.vocab_size
    full_layers = sum(1 for t in tc.layer_types if t == "full_attention")
    lin_layers = len(tc.layer_types) - full_layers
    kv_tok = full_layers * 2 * tc.num_key_value_heads * tc.head_dim * 2                 # bf16 K+V bytes/token
    lin_state = lin_layers * tc.linear_num_value_heads * tc.linear_key_head_dim * tc.linear_value_head_dim * 4  # fp32 state/seq
    G, P, L_new, L_prompt, mb = 8, 8, 256, 1024, 8
    rows = {}
    base_gb = base * 2 / GB
    per_adapter_gb = lp * (4 + 4 + 8) / GB          # fp32 weights + grads + Adam m,v
    logits_gb = mb * (L_new + 1) * V * (2 + 4 + 4) / GB   # bf16 logits + fp32 log_softmax + its grad (one micro-batch)
    act_gb = 1.5                                      # grad-checkpointed activations, 8 x 1.3k tokens (estimate)
    ctx_gb = 0.8                                      # CUDA context + allocator slack per process
    def vllm_gb(seqs):
        return base_gb + 0.15 * 7 + seqs * ((L_prompt + L_new) * kv_tok + lin_state) / GB + 1.5   # weights + LoRA slots + caches + cudagraph/activations
    seqs = G * P                                       # one policy's rollout batch in flight
    for n_main in (2, 4, 6, 8):
        n_ad = n_main + 1
        trainer = base_gb + n_ad * per_adapter_gb + logits_gb + act_gb + ctx_gb
        A = trainer + vllm_gb(seqs * 2)
        B = n_ad * (base_gb + per_adapter_gb + logits_gb + act_gb + ctx_gb + vllm_gb(seqs))
        rows[f"{n_main}main+1exploiter"] = {"A_shared_base_multiLoRA_GB": round(A, 1), "B_separate_TRL_processes_GB": round(B, 1),
                                            "A_fits_48GB_with_6GB_headroom": A <= 42, "B_fits": B <= 42}
    nem = {"weights_bf16_GB": round(7.95, 2), "note": "Nemotron-3-Nano-4B: base 7.95 GB (pinned LFS size). Trainer ~ 8 + 6-8 activations/logits (vocab 131k) + vLLM copy 8 + caches ~3 = ~25-28 GB alone (ESTIMATE; 12-16 GB training-only estimate from PR #14 is unmeasured)."}
    out = {"label": "ESTIMATE (arithmetic from pinned configs; not measured on GPU)", "base_params": base, "base_bf16_GB": round(base_gb, 2),
           "lora_rank": rank, "lora_params_per_adapter": lp, "per_adapter_train_state_GB": round(per_adapter_gb, 3),
           "vocab": V, "logits_peak_per_microbatch_GB": round(logits_gb, 2), "kv_bytes_per_token": kv_tok, "linear_attn_state_bytes_per_seq": lin_state,
           "rollout": {"G": G, "prompts_per_policy_round": P, "max_new_tokens": L_new, "max_prompt_tokens": L_prompt, "micro_batch": mb},
           "layouts": rows, "mixed_base_with_nemotron4b": nem}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    r = main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
    json.dump(r, open(sys.argv[2] if len(sys.argv) > 2 else "vram_estimate.json", "w"), indent=1)

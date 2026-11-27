#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Merge the Hamut'aq-0.8B SFT LoRA into the pinned Qwen3.5-0.8B base and save it
(bf16) so vLLM can serve one shared base with hot-swapped league adapters."""
import hashlib, json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base, adapter, out = sys.argv[1], sys.argv[2], sys.argv[3]
m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
m.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(base).save_pretrained(out)
sha = hashlib.sha256(open(f"{adapter}/adapter_model.safetensors", "rb").read()).hexdigest()
json.dump({"base": base, "sft_adapter": adapter, "sft_adapter_sha256": sha}, open(f"{out}/merge_receipt.json", "w"), indent=1)
print(json.dumps({"merged": out, "sft_adapter_sha256": sha}))

# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the Hamut'aq-4B (Nemotron 3 Nano 4B) recipe.

Base facts (pinned in base_pin.json):
  * nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 @ dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f
  * NemotronHForCausalLM, Mamba-2 / Transformer hybrid, 42 blocks:
    21 Mamba-2 (mixer.in_proj, mixer.out_proj), 17 MLP (mixer.up_proj, mixer.down_proj, relu2, no gate),
    4 GQA attention (mixer.q_proj/k_proj/v_proj/o_proj; 40 heads, 8 KV heads, head_dim 128).
  * Chat template: ChatML-style <|im_start|>/<|im_end|>. Reasoning is toggled by the
    chat-template kwarg enable_thinking (default True), not by a /think system prompt.
    enable_thinking=False makes the generation prompt '<|im_start|>assistant\\n<think></think>'.
"""
import hashlib, json, os, re, time, datetime

BASE_ID = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
BASE_REV = "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"
END = "<|im_end|>"

# LoRA targets from the real module list (see base_pin.json "linear_modules").
LORA_ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]      # 4 attention blocks
LORA_MLP = ["up_proj", "down_proj"]                         # 17 MLP blocks
LORA_MAMBA = ["in_proj", "out_proj"]                        # 21 Mamba-2 blocks (opt-in, ablation)


def lora_targets(include_mamba=False):
    return LORA_ATTN + LORA_MLP + (LORA_MAMBA if include_mamba else [])


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


class Log:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def __call__(self, *a):
        line = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def render_prompt(tok, msgs, thinking=False):
    """Prompt text up to and including the assistant generation prefix."""
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def encode_sft(tok, msgs, max_len, thinking=False):
    """(ids, labels) with loss on the final assistant turn only, or None if over length.

    The supervised span is '<content><|im_end|>', which is exactly what the template emits
    after '<|im_start|>assistant\\n<think></think>' for a non-thinking assistant turn
    (checked by dryrun_cpu.py: template_consistency)."""
    last = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
    pr = tok(render_prompt(tok, msgs[:last], thinking), add_special_tokens=False)["input_ids"]
    an = tok(msgs[last]["content"].strip() + END, add_special_tokens=False)["input_ids"]
    if not an or len(pr) + len(an) > max_len:
        return None
    return pr + an, [-100] * len(pr) + an


def encode_pair(tok, prompt_msgs, chosen, rejected, max_len, thinking=False):
    pr = tok(render_prompt(tok, prompt_msgs, thinking), add_special_tokens=False)["input_ids"]
    c = tok(chosen.strip() + END, add_special_tokens=False)["input_ids"]
    r = tok(rejected.strip() + END, add_special_tokens=False)["input_ids"]
    if len(pr) + max(len(c), len(r)) > max_len:
        return None
    return pr, c, r


def backbone_and_head(model):
    """(inner backbone, lm_head) for a PEFT-wrapped or plain NemotronH model."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model, base.get_output_embeddings()


def chunked_token_logps(hidden, lm_head, labels, chunk=1024):
    """Per-sequence sum of log p(label) over positions with label != -100, plus token counts.

    Never materialises the full [T, vocab] fp32 logits (vocab is 131,072)."""
    import torch
    B = hidden.size(0)
    h = hidden[:, :-1, :]
    y = labels[:, 1:]
    out = hidden.new_zeros(B, dtype=torch.float32)
    n = torch.zeros(B, dtype=torch.long, device=hidden.device)
    for b in range(B):
        keep = y[b].ne(-100)
        hb, yb = h[b][keep], y[b][keep]
        for i in range(0, yb.numel(), chunk):
            lg = lm_head(hb[i:i + chunk]).float()
            out[b] = out[b] + (-torch.nn.functional.cross_entropy(lg, yb[i:i + chunk], reduction="sum"))
        n[b] = yb.numel()
    return out, n


def deadline_guard(t_start, max_seconds, log):
    """Fail-closed wall-clock cap inside the trainer (the launcher adds an outer `timeout`)."""
    if max_seconds and time.time() - t_start > max_seconds:
        log("CAP_REACHED wall-clock", round(time.time() - t_start), ">", max_seconds, "- stopping, no further steps")
        return True
    return False

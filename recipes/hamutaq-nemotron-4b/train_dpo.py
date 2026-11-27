#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage 2: DPO on top of the stage-1 SFT adapter (TRAINING-PLAN.md line A: SFT -> APO/DPO -> GRPO).

The SFT adapter is merged into the bf16 base, a fresh LoRA is trained with the sigmoid DPO loss,
and the reference policy is the merged SFT model (the same weights with the new adapter disabled).
Reference log-probs are computed once up front. Optional RPO term (--rpo-alpha) adds chosen-side NLL.
Local output only: no upload, no publish.
"""
import argparse, json, math, os, random, sys, time
from common import (BASE_ID, BASE_REV, Log, lora_targets, encode_pair, backbone_and_head,
                    chunked_token_logps, deadline_guard, sha256_file, now)
from train_sft import load_model, kernel_check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_ID)
    ap.add_argument("--base-revision", default=BASE_REV)
    ap.add_argument("--local-base", default="")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--sft-adapter", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--pairs-sha256", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--rpo-alpha", type=float, default=0.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-mamba", action="store_true")
    ap.add_argument("--ce-chunk", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--max-seconds", type=int, required=True)
    ap.add_argument("--allow-slow-path", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    import torch, transformers, peft
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    os.makedirs(a.out, exist_ok=True)
    log = Log(os.path.join(a.out, "train_log.txt"))
    random.seed(a.seed); torch.manual_seed(a.seed)
    t_start, started = time.time(), now()
    got = sha256_file(a.pairs)
    if got != a.pairs_sha256:
        log("FAIL pairs sha mismatch", got); return 3
    kernels = kernel_check(a, log)
    tok = AutoTokenizer.from_pretrained(a.local_base or a.base, **({} if a.local_base else {"revision": a.base_revision}))
    raw = [json.loads(l) for l in open(a.pairs, encoding="utf-8") if l.strip()]
    if a.max_rows:
        raw = raw[:a.max_rows]
    pairs, over = [], 0
    for r in raw:
        e = encode_pair(tok, r["prompt"], r["chosen"][-1]["content"], r["rejected"][-1]["content"], a.max_len)
        if e is None:
            over += 1; continue
        pairs.append(e)
    random.shuffle(pairs)

    model = load_model(a, torch)
    model = PeftModel.from_pretrained(model, a.sft_adapter).merge_and_unload()
    model.config.use_cache = False
    if a.device == "cuda":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    targets = lora_targets(a.lora_mamba)
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, bias="none",
                                             task_type="CAUSAL_LM", target_modules=targets))
    model.peft_config["default"].base_model_name_or_path = a.base
    inner, head = backbone_and_head(model)

    def seq(pr, comp):
        ids = torch.tensor([pr + comp], device=a.device)
        lab = torch.tensor([[-100] * len(pr) + comp], device=a.device)
        return ids, lab

    def logp(pr, comp):
        ids, lab = seq(pr, comp)
        lp, n = chunked_token_logps(inner(input_ids=ids).last_hidden_state, head, lab, a.ce_chunk)
        return lp[0], n[0]

    model.eval(); ref = []
    with torch.no_grad(), model.disable_adapter():
        for pr, c, r in pairs:
            ref.append((float(logp(pr, c)[0]), float(logp(pr, r)[0])))
    log("pairs", len(pairs), "dropped_overlength", over, "ref_logps_done", round(time.time() - t_start, 1), "s")

    total = a.max_steps or max(1, int(math.ceil(len(pairs) / a.grad_accum) * a.epochs))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    warm = max(1, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / total))
    model.train(); step = micro = i = 0; curve, accs, margins, capped = [], [], [], False
    acc_l = 0.0
    while step < total:
        if deadline_guard(t_start, a.max_seconds, log):
            capped = True; break
        k = i % len(pairs); i += 1
        pr, c, r = pairs[k]; rc, rr = ref[k]
        pc, nc = logp(pr, c); pj, _ = logp(pr, r)
        z = a.beta * ((pc - rc) - (pj - rr))
        loss = -F.logsigmoid(z)
        if a.rpo_alpha:
            loss = loss + a.rpo_alpha * (-pc / nc.clamp(min=1))
        (loss / a.grad_accum).backward(); micro += 1
        acc_l += float(loss.detach()); accs.append(float(z > 0)); margins.append(float(z) / a.beta)
        if micro % a.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            curve.append(round(acc_l / a.grad_accum, 4)); acc_l = 0.0
            if step % 5 == 0 or step == total:
                log("step", step, "/", total, "loss", curve[-1], "reward_acc", round(sum(accs[-a.grad_accum * 5:]) / len(accs[-a.grad_accum * 5:]), 3),
                    "margin", round(sum(margins[-a.grad_accum * 5:]) / len(margins[-a.grad_accum * 5:]), 3))
    final = os.path.join(a.out, "final")
    model.save_pretrained(final); tok.save_pretrained(final)
    rec = {"schema": "szl.train.receipt/v1", "stage": "dpo", "status": "CAPPED" if capped else "MEASURED",
           "promotion": "HOLD", "signature": "UNSIGNED", "publish": "NONE", "started_at": started, "finished_at": now(),
           "seconds": round(time.time() - t_start, 1),
           "base": {"repo": a.base, "revision": a.base_revision, "mamba_kernels": kernels},
           "sft_adapter": {"path": a.sft_adapter, "sha256": sha256_file(os.path.join(a.sft_adapter, "adapter_model.safetensors"))},
           "data": {"pairs_sha256": got, "pairs": len(pairs), "dropped_overlength": over},
           "hparams": {k: getattr(a, k) for k in ("max_len", "grad_accum", "epochs", "lr", "beta", "rpo_alpha", "lora_r", "lora_alpha", "seed", "max_seconds")},
           "lora_targets": targets, "steps": step,
           "loss": {"first": curve[0] if curve else None, "last": curve[-1] if curve else None, "curve": curve},
           "reward_acc_last": round(sum(accs[-100:]) / max(1, len(accs[-100:])), 4),
           "adapter_sha256": sha256_file(os.path.join(final, "adapter_model.safetensors")),
           "note": "adapter applies on top of base+merged SFT adapter; ship both or a merged checkpoint",
           "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}}
    json.dump(rec, open(os.path.join(final, "train_receipt.json"), "w"), indent=1)
    log("DPO_DONE", rec["status"], "steps", step)
    return 6 if capped else 0


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 2
    except Exception:
        import traceback; print("CRASH " + traceback.format_exc(), flush=True); rc = 2
    sys.exit(rc)

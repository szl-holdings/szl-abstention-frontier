#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage 1: LoRA SFT of Hamut'aq-4B on NVIDIA-Nemotron-3-Nano-4B-BF16 (pinned).

Adapted from scripts/frontier_r2/train_frontier_r2.py (micro-batch + accumulation, chunked CE,
OOM guard). Nemotron-specific: native transformers NemotronH (no trust_remote_code by default),
Mamba-2 fast-path kernels required on CUDA (fail closed), reasoning off (enable_thinking=False),
LoRA on attention + MLP projections. Writes the adapter and a receipt locally. It never uploads,
publishes, or changes visibility; output stays private on the training host.
"""
import argparse, json, math, os, random, sys, time
from common import (BASE_ID, BASE_REV, Log, lora_targets, encode_sft, backbone_and_head,
                    chunked_token_logps, deadline_guard, sha256_file, now)


def load_model(a, torch):
    from transformers import AutoModelForCausalLM
    src = a.local_base or a.base
    kw = dict(dtype=torch.bfloat16)
    if not a.local_base:
        kw["revision"] = a.base_revision
    if a.trust_remote_code:
        kw["trust_remote_code"] = True
    if a.device == "cuda":
        kw["device_map"] = {"": 0}
    return AutoModelForCausalLM.from_pretrained(src, **kw)


def kernel_check(a, log):
    from transformers.utils.import_utils import is_mamba_2_ssm_available, is_causal_conv1d_available
    ok = bool(is_mamba_2_ssm_available() and is_causal_conv1d_available())
    log("mamba_kernels", ok)
    if a.device == "cuda" and not ok and not a.allow_slow_path:
        log("FAIL mamba-ssm/causal-conv1d missing on CUDA; torch fallback is too slow to run under the cap")
        raise SystemExit(5)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_ID)
    ap.add_argument("--base-revision", default=BASE_REV)
    ap.add_argument("--local-base", default="", help="local checkpoint dir (dry-run only)")
    ap.add_argument("--trust-remote-code", action="store_true", help="use the repo's modeling_nemotron_h.py instead of native")
    ap.add_argument("--train", required=True)
    ap.add_argument("--train-sha256", required=True)
    ap.add_argument("--val", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--tok-budget", type=int, default=4096, help="max padded tokens per micro-batch")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-mamba", action="store_true", help="also adapt Mamba in_proj/out_proj (ablation)")
    ap.add_argument("--ce-chunk", type=int, default=1024)
    ap.add_argument("--max-oom", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--max-seconds", type=int, required=True, help="fail-closed wall-clock cap")
    ap.add_argument("--val-rows", type=int, default=200)
    ap.add_argument("--allow-slow-path", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    import torch, transformers, peft
    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model
    os.makedirs(a.out, exist_ok=True)
    log = Log(os.path.join(a.out, "train_log.txt"))
    random.seed(a.seed); torch.manual_seed(a.seed)
    t_start, started = time.time(), now()
    got = sha256_file(a.train)
    if got != a.train_sha256:
        log("FAIL train sha mismatch", got); return 3
    kernels = kernel_check(a, log)

    tok = AutoTokenizer.from_pretrained(a.local_base or a.base, **({} if a.local_base else {"revision": a.base_revision}))
    rows = [json.loads(l)["messages"] for l in open(a.train, encoding="utf-8") if l.strip()]
    if a.max_rows:
        rows = rows[:a.max_rows]
    exs, over = [], 0
    for m in rows:
        e = encode_sft(tok, m, a.max_len)
        if e is None:
            over += 1; continue
        exs.append(e)
    # token-budget micro-batches (length-bucketed, right-padded). Right padding is safe for the causal
    # Mamba-2 scan and the causal attention: pads come after every real token. --tok-budget 0 = fixed micro-batch.
    random.shuffle(exs)
    if a.tok_budget:
        order = sorted(range(len(exs)), key=lambda i: len(exs[i][0]) // 64 + random.random())
        batches, cur, mx = [], [], 0
        for i in order:
            L = len(exs[i][0])
            if cur and max(mx, L) * (len(cur) + 1) > a.tok_budget:
                batches.append([exs[j] for j in cur]); cur, mx = [], 0
            cur.append(i); mx = max(mx, L)
        if cur:
            batches.append([exs[j] for j in cur])
        random.shuffle(batches)
    else:
        batches = [exs[i:i + a.micro_batch] for i in range(0, len(exs), a.micro_batch)]
    log("examples", len(exs), "dropped_overlength", over, "tokens", sum(len(e[0]) for e in exs))

    model = load_model(a, torch)
    model.config.use_cache = False
    if a.device == "cuda":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    targets = lora_targets(a.lora_mamba)
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05, bias="none",
                                             task_type="CAUSAL_LM", target_modules=targets))
    model.peft_config["default"].base_model_name_or_path = a.base
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    inner, head = backbone_and_head(model)
    dev = torch.cuda.get_device_name(0) if a.device == "cuda" else a.device
    total = a.max_steps or max(1, int(math.ceil(len(batches) / a.grad_accum) * a.epochs))
    log("micro_batches", len(batches))
    log("trainable", trainable, "targets", targets, "device", dev, "optimizer_steps_planned", total)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / total))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def batch_tensors(b):
        L = max(len(x[0]) for x in b)
        ids = torch.tensor([x[0] + [pad] * (L - len(x[0])) for x in b], device=a.device)
        lab = torch.tensor([x[1] + [-100] * (L - len(x[1])) for x in b], device=a.device)
        att = torch.tensor([[1] * len(x[0]) + [0] * (L - len(x[0])) for x in b], device=a.device)
        return ids, lab, att

    model.train()
    step = micro = toks = ooms = bi = 0; curve, acc, accn = [], 0.0, 0
    t0, capped = time.time(), False
    while step < total:
        if deadline_guard(t_start, a.max_seconds, log):
            capped = True; break
        b = batches[bi % len(batches)]; bi += 1
        ids, lab, att = batch_tensors(b)
        try:
            hid = inner(input_ids=ids, attention_mask=att).last_hidden_state
            lp, n = chunked_token_logps(hid, head, lab, a.ce_chunk)
            if int(n.sum()) == 0:
                continue
            loss = -(lp.sum() / n.sum())
            (loss / a.grad_accum).backward()
            acc += float(loss.detach()); accn += 1; micro += 1; toks += int(att.sum())
        except torch.OutOfMemoryError:
            ooms += 1; opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            log("OOM_GUARD skip len", ids.shape[1], "ooms", ooms)
            acc, accn, micro = 0.0, 0, (micro // a.grad_accum) * a.grad_accum
            if ooms > a.max_oom:
                log("FAIL too many OOM skips"); return 4
            continue
        if micro % a.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            curve.append(round(acc / max(1, accn), 4)); acc, accn = 0.0, 0
            if step % 10 == 0 or step == total:
                el = time.time() - t0
                mem = torch.cuda.max_memory_allocated() / 2**30 if a.device == "cuda" else 0
                log("step", step, "/", total, "loss", curve[-1], "tok/s", round(toks / el, 1),
                    "eta_min", round((total - step) * el / step / 60, 1), "mem_gb", round(mem, 1))

    val = None
    if a.val and os.path.exists(a.val):
        model.eval(); vl, vn = 0.0, 0
        with torch.no_grad():
            for l in list(open(a.val, encoding="utf-8"))[:a.val_rows]:
                e = encode_sft(tok, json.loads(l)["messages"], a.max_len)
                if e is None:
                    continue
                ids, lab, att = batch_tensors([e])
                lp, n = chunked_token_logps(inner(input_ids=ids, attention_mask=att).last_hidden_state, head, lab, a.ce_chunk)
                vl += float(-lp.sum()); vn += int(n.sum())
        val = round(vl / max(1, vn), 4)
        log("val_loss", val)

    final = os.path.join(a.out, "final")
    model.save_pretrained(final); tok.save_pretrained(final)
    rec = {"schema": "szl.train.receipt/v1", "stage": "sft", "status": "CAPPED" if capped else "MEASURED",
           "promotion": "HOLD", "signature": "UNSIGNED", "energy": "NOT_MEASURED", "publish": "NONE",
           "started_at": started, "finished_at": now(), "seconds": round(time.time() - t_start, 1),
           "base": {"repo": a.base, "revision": a.base_revision, "loader": "remote_code" if a.trust_remote_code else "native",
                    "mamba_kernels": kernels, "license": "NVIDIA Nemotron Open Model License"},
           "data": {"train_sha256": got, "examples": len(exs), "dropped_overlength": over},
           "hparams": {k: getattr(a, k) for k in ("max_len", "micro_batch", "tok_budget", "grad_accum", "epochs", "lr", "lora_r",
                                                  "lora_alpha", "ce_chunk", "seed", "max_seconds")},
           "lora_targets": targets, "trainable_params": trainable, "steps": step, "oom_skips": ooms, "tokens": toks,
           "tok_per_s": round(toks / max(1e-9, time.time() - t0), 1), "val_loss": val,
           "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if a.device == "cuda" else None,
           "loss": {"first": curve[0] if curve else None, "last": curve[-1] if curve else None, "curve": curve},
           "adapter_sha256": sha256_file(os.path.join(final, "adapter_model.safetensors")),
           "hardware": {"device": dev}, "reasoning": "enable_thinking=False",
           "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}}
    json.dump(rec, open(os.path.join(final, "train_receipt.json"), "w"), indent=1)
    log("SFT_DONE", "status", rec["status"], "steps", step)
    return 6 if capped else 0


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 2
    except Exception:
        import traceback; print("CRASH " + traceback.format_exc(), flush=True); rc = 2
    sys.exit(rc)

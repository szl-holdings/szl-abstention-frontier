"""Chaski abstention frontier r2 trainer (OOM-fixed relaunch of op1).

op1 (salad-chaski-frontier-{4b,9b}-op1) died with CUDA OOM in loss.backward on
24 GB RTX 4090s (4B at 20.4 GB, step 10; 9B at 23.7 GB, step 30). r2 changes:

  * micro-batch 1 with gradient accumulation (same effective batch),
  * chunked cross-entropy over assistant tokens only (no full [T, vocab] fp32
    logits tensor; Qwen3.5 vocab is ~248k, so full logits dominated memory),
  * gradient checkpointing,
  * an OOM guard: on CUDA OOM the micro-batch is skipped, cache is cleared and
    the event is logged; more than --max-oom skips aborts fail-closed,
  * optional 4-bit QLoRA (--qlora) for the 9B base on a 48 GB card.

Receipts are written as status MEASURED, signature UNSIGNED, promotion HOLD.
Nothing here signs, promotes, changes a model card, or changes visibility.
"""
import argparse, hashlib, json, math, os, random, sys, time, datetime

import torch
import torch.nn.functional as F


def log(fp, msg):
    line = time.strftime("%H:%M:%S") + " " + msg
    print(line, flush=True)
    with open(fp, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def chunked_ce(hidden, lm_head, labels, chunk):
    """Mean CE over positions where labels != -100, computed in token chunks."""
    h = hidden[:, :-1, :].reshape(-1, hidden.size(-1))
    y = labels[:, 1:].reshape(-1)
    keep = y.ne(-100)
    h, y = h[keep], y[keep]
    n = y.numel()
    if n == 0:
        return None, 0
    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, n, chunk):
        logits = lm_head(h[i:i + chunk]).float()
        total = total + F.cross_entropy(logits, y[i:i + chunk], reduction="sum")
    return total / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)            # Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B
    ap.add_argument("--base-revision", required=True)
    ap.add_argument("--train", required=True)           # local train.jsonl from szl-frontier-openprob-mix-v1
    ap.add_argument("--train-sha256", required=True)
    ap.add_argument("--leak-ids", default="")          # optional file of heldout prompts to drop
    ap.add_argument("--out", required=True)
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=64)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--ce-chunk", type=int, default=1024)
    ap.add_argument("--max-oom", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-steps", type=int, default=0)  # 0 = from epochs
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    LOG = os.path.join(a.out, "train_log.txt")
    random.seed(a.seed); torch.manual_seed(a.seed)
    t_start = time.time()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()

    got = sha256_file(a.train)
    if got != a.train_sha256:
        log(LOG, f"FAIL train sha mismatch {got}")
        return 3

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tok = AutoTokenizer.from_pretrained(a.base, revision=a.base_revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    leak = set()
    if a.leak_ids and os.path.exists(a.leak_ids):
        leak = {l.strip().lower() for l in open(a.leak_ids, encoding="utf-8") if l.strip()}
    rows, dropped_leak = [], 0
    for line in open(a.train, encoding="utf-8"):
        r = json.loads(line)
        m = r.get("messages") or []
        if len(m) < 2 or m[-1].get("role") != "assistant":
            continue
        if leak and any(x.get("content", "").strip().lower() in leak for x in m if x.get("role") == "user"):
            dropped_leak += 1
            continue
        rows.append(m)
    log(LOG, f"leak_filter dropped {dropped_leak} kept {len(rows)}")

    exs, over = [], 0
    for m in rows:
        prompt = tok.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        full = prompt + m[-1]["content"] + "<|im_end|>"
        ids = tok(full, add_special_tokens=False)["input_ids"]
        npr = len(tok(prompt, add_special_tokens=False)["input_ids"])
        if len(ids) > a.max_len or npr >= len(ids):
            over += 1
            continue
        exs.append((ids, [-100] * npr + ids[npr:]))
    random.shuffle(exs)
    # length-sorted buckets inside shuffled blocks keep padding low without losing randomness
    exs.sort(key=lambda e: len(e[0]))
    batches = [exs[i:i + a.micro_batch] for i in range(0, len(exs), a.micro_batch)]
    random.shuffle(batches)

    kw = dict(revision=a.base_revision, dtype=torch.bfloat16, device_map={"": 0})
    if a.qlora:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(a.base, **kw)
    model.config.use_cache = False
    if a.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lcfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lcfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dev = torch.cuda.get_device_name(0)
    log(LOG, f"examples {len(exs)} dropped_overlength {over} trainable {trainable} device {dev} qlora {a.qlora}")

    steps_per_epoch = math.ceil(len(batches) / a.grad_accum)
    total_steps = a.max_steps or max(1, int(steps_per_epoch * a.epochs))
    log(LOG, f"micro_batches {len(batches)} optimizer_steps_planned {total_steps} micro_batch {a.micro_batch} grad_accum {a.grad_accum}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, total_steps // 20)) * max(0.0, 1 - s / total_steps))

    base_model = model.get_base_model()
    inner = base_model.model
    lm_head = base_model.get_output_embeddings()

    model.train()
    step, micro, toks, ooms, curve = 0, 0, 0, 0, []
    acc_loss, acc_n = 0.0, 0
    t0 = time.time()
    bi = 0
    while step < total_steps:
        b = batches[bi % len(batches)]; bi += 1
        L = max(len(x[0]) for x in b)
        ids = torch.tensor([x[0] + [tok.pad_token_id] * (L - len(x[0])) for x in b], device="cuda")
        lab = torch.tensor([x[1] + [-100] * (L - len(x[1])) for x in b], device="cuda")
        att = torch.tensor([[1] * len(x[0]) + [0] * (L - len(x[0])) for x in b], device="cuda")
        hid = loss = None
        try:
            hid = inner(input_ids=ids, attention_mask=att).last_hidden_state
            loss, n = chunked_ce(hid, lm_head, lab, a.ce_chunk)
            if loss is None:
                continue
            (loss / a.grad_accum).backward()
            acc_loss += float(loss.detach()); acc_n += 1
            micro += 1; toks += int(att.sum())
        except torch.OutOfMemoryError:
            ooms += 1
            opt.zero_grad(set_to_none=True)
            hid = loss = None
            torch.cuda.empty_cache()
            log(LOG, f"OOM_GUARD skip micro-batch len {L} ooms {ooms}/{a.max_oom}")
            acc_loss, acc_n, micro = 0.0, 0, (micro // a.grad_accum) * a.grad_accum
            if ooms > a.max_oom:
                log(LOG, "FAIL too many OOM skips")
                return 4
            continue
        if micro % a.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            l = acc_loss / max(1, acc_n); curve.append(round(l, 4)); acc_loss, acc_n = 0.0, 0
            if step % 10 == 0 or step == total_steps:
                el = time.time() - t0
                log(LOG, f"step {step} / {total_steps} loss {l:.4f} tok/s {toks / el:.1f} "
                         f"eta_min {(total_steps - step) * el / step / 60:.1f} "
                         f"mem_gb {torch.cuda.max_memory_allocated() / 2**30:.1f} ooms {ooms}")

    final = os.path.join(a.out, "final")
    model.save_pretrained(final); tok.save_pretrained(final)
    adp = os.path.join(final, "adapter_model.safetensors")
    import transformers, peft
    rec = {
        "schema": "szl.train.receipt/v1", "status": "MEASURED", "promotion": "HOLD",
        "energy": "NOT_MEASURED", "signature": "UNSIGNED", "run": "frontier-r2",
        "started_at": started,
        "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seconds": round(time.time() - t_start, 1),
        "base": {"repo": a.base, "revision": a.base_revision, "qlora_4bit": a.qlora},
        "data": {"train_sha256": got, "examples": len(exs), "dropped_overlength": over,
                 "dropped_leak": dropped_leak},
        "hparams": {k: getattr(a, k) for k in ("max_len", "micro_batch", "grad_accum", "epochs", "lr",
                                               "lora_r", "lora_alpha", "ce_chunk", "seed")},
        "steps": step, "oom_skips": ooms, "tokens": toks,
        "tok_per_s": round(toks / max(1e-9, time.time() - t0), 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "loss": {"first": curve[0] if curve else None, "last": curve[-1] if curve else None, "curve": curve},
        "adapter_sha256": sha256_file(adp),
        "hardware": {"gpu": dev},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
    }
    json.dump(rec, open(os.path.join(final, "train_receipt.json"), "w"), indent=1)
    log(LOG, f"TRAIN_DONE steps {step} ooms {ooms} peak_mem_gb {rec['peak_mem_gb']}")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        import traceback
        print("CRASH " + traceback.format_exc(), flush=True)
        rc = 2
    sys.exit(rc)

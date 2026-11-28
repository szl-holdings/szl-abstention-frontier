#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# (c) 2026 Lutar, Stephen P. - SZL Holdings
"""Hamut'aq LoRA league: N main policies + 1 exploiter as LoRA adapters on ONE shared
frozen base (the Hamut'aq-0.8B SFT merged into Qwen3.5-0.8B), trained with
DAPO / Dr.GRPO / GSPO-style group-relative policy optimisation on verifiable rewards.

Why a custom loop: TRL's GRPOTrainer trains one policy per process (one base copy
each). A league of 6+ policies on one 48 GB card is cheapest with one base in
memory and per-adapter optimisers. The loss follows the TRL/DAPO/Dr.GRPO/GSPO
papers (see README); the reward functions are the same module the TRL path uses.

Fail-closed rules enforced here:
  * reward self-tests must pass before the first step (test_rewards.run)
  * sealed blocklist + source pins come from build_receipt.json and are re-verified
  * a reward is only used if its receipt row was written (T1); else the round aborts
  * wall-clock budget: training stops and saves before --budget-min elapses
Nothing here uploads anything; HF tokens are unset by run_league.sh.
"""
import argparse, collections, hashlib, json, math, os, random, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rewards as R          # noqa: E402
import test_rewards          # noqa: E402

import torch                 # noqa: E402


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ----------------------------------------------------------------- receipts (T1)
class Receipts:
    def __init__(self, out, meta):
        self.dir = Path(out) / "receipts"; self.dir.mkdir(parents=True, exist_ok=True)
        self.rows = open(self.dir / "rollouts.jsonl", "a", buffering=1)
        self.comps = open(self.dir / "completions.jsonl", "a", buffering=1)
        self.merkle = open(self.dir / "merkle.jsonl", "a", buffering=1)
        self.meta = meta
        self.pending = []

    def write(self, row, completion):
        csha = R.sha256_text(completion)
        row = dict(row, completion_sha256=csha, **self.meta)
        line = json.dumps(row, sort_keys=True)
        self.rows.write(line + "\n")
        self.comps.write(json.dumps({"completion_sha256": csha, "text": completion}) + "\n")
        h = hashlib.sha256(line.encode()).hexdigest()
        self.pending.append(h)
        return h

    def seal_round(self, rnd, policy):
        leaves = self.pending or [hashlib.sha256(b"").hexdigest()]
        while len(leaves) > 1:
            if len(leaves) % 2:
                leaves.append(leaves[-1])
            leaves = [hashlib.sha256((leaves[i] + leaves[i + 1]).encode()).hexdigest() for i in range(0, len(leaves), 2)]
        rec = {"round": rnd, "policy": policy, "n": len(self.pending), "merkle_root": leaves[0], "at": now()}
        self.merkle.write(json.dumps(rec) + "\n")
        n = len(self.pending); self.pending = []
        return rec, n


# ----------------------------------------------------------------- prompts
def ctx_of(item):
    return "\n".join(m["content"] for m in item["messages"])


def compose(item, t, want_report):
    msgs = [dict(m) for m in item["messages"]]
    extra = []
    if t is not None:
        extra.append(R.threshold_instruction(t))
    if want_report:
        extra.append(R.REPORT_INSTRUCTION)
    if extra:
        msgs[-1]["content"] = msgs[-1]["content"] + "\n\n" + "\n".join(extra)
    return msgs


def render(tok, msgs):
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


class Sampler:
    """Per-policy data slice: weights over tasks, seed, threshold schedule."""
    def __init__(self, pool, spec, seed):
        self.rng = random.Random(seed)
        self.by_task = collections.defaultdict(list)
        for it in pool:
            self.by_task[it["gold"]["task"]].append(it)
        w = spec.get("task_weights") or {}
        self.tasks = [k for k in sorted(self.by_task) if w.get(k, 1.0) > 0]
        self.weights = [w.get(k, 1.0) for k in self.tasks]
        self.spec = spec

    def threshold(self, frac):
        sch = self.spec.get("t_schedule", "curriculum")
        if self.rng.random() < self.spec.get("t_none_frac", 0.25):
            return None
        if sch == "curriculum":          # collapse guard: anneal t upward
            grid = [0.5] if frac < 0.2 else ([0.5, 0.75] if frac < 0.5 else [0.5, 0.75, 0.9])
        elif isinstance(sch, (int, float)):
            grid = [float(sch)]
        else:
            grid = list(sch)
        return self.rng.choice(grid)

    def draw(self, n, frac):
        out = []
        for _ in range(n):
            task = self.rng.choices(self.tasks, self.weights)[0]
            it = self.rng.choice(self.by_task[task])
            t = self.threshold(frac)
            want_report = self.rng.random() < self.spec.get("report_frac", 0.5) and "report" in self.spec.get("components", [])
            out.append((it, t, want_report))
        return out


# ----------------------------------------------------------------- exploiter (T4)
SENT = re.compile(r"(?<=[.!?])\s+")


def exploiter_prompt(item):
    ctx_full = item["messages"][-1]["content"]
    m = re.search(r"Context:\n(.*?)\n\nQuestion:(.*?)\n\n", ctx_full, re.S)
    if not m:
        return None
    sents = [s for s in SENT.split(m.group(1).strip()) if s]
    if len(sents) < 3:
        return None
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sents))
    msg = ("You are a red-team question writer. Below is a context split into numbered sentences and a question. "
           "Choose sentences to DELETE so that a careful reader would be tricked: either the answer disappears "
           "(the reader should now abstain) or the context gets shorter while the answer stays (the reader should still answer). "
           'Reply with JSON only: {"delete": [sentence numbers]}. Delete at least 1 and at most half of the sentences.\n\n'
           f"CONTEXT\n{numbered}\n\nQUESTION\n{m.group(2).strip()}")
    return {"messages": [{"role": "user", "content": msg}], "sents": sents, "question": m.group(2).strip(), "base": item}


def exploiter_verify(xp, completion):
    """Deterministic verifier. Returns (probe_item or None, reason)."""
    try:
        j = json.loads(completion[completion.index("{"): completion.rindex("}") + 1])
        dele = sorted({int(i) for i in j["delete"]})
    except Exception:
        return None, "bad_json"
    n = len(xp["sents"])
    if not dele or len(dele) > n // 2 or any(i < 0 or i >= n for i in dele):
        return None, "bad_delete_set"
    kept = [s for i, s in enumerate(xp["sents"]) if i not in set(dele)]
    ctx = " ".join(kept)
    golds = xp["base"]["gold"]["answers"]
    present = any(R.norm(g) and R.norm(g) in R.norm(ctx) for g in golds)
    label = "ANSWER" if present else "ABSTAIN"
    tail = "\n\nIf the context does not contain the answer to the question, reply with exactly UNANSWERABLE. Otherwise answer concisely."
    probe = {"src": "exploiter", "id": "xp-" + hashlib.sha256((ctx + xp["question"]).encode()).hexdigest()[:16],
             "messages": [{"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {xp['question']}{tail}"}],
             "gold": {"task": "squad", "label": label, "answers": golds if present else []}}
    return probe, "ok"


# ----------------------------------------------------------------- model / generation
def load_model(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    dtype = torch.bfloat16 if a.device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.base_dir)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base_dir, dtype=dtype).to(a.device)
    if a.sft_adapter:
        model = PeftModel.from_pretrained(model, a.sft_adapter).merge_and_unload()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model, LoraConfig, get_peft_model


def attach_adapters(model, policies, LoraConfig, get_peft_model, lora):
    first = True
    for p in policies:
        cfg = LoraConfig(r=p.get("lora_r", lora["r"]), lora_alpha=p.get("lora_alpha", lora["alpha"]),
                         lora_dropout=0.0, target_modules=lora["targets"], task_type="CAUSAL_LM")
        if first:
            model = get_peft_model(model, cfg, adapter_name=p["name"]); first = False
        else:
            model.add_adapter(p["name"], cfg)
    return model


def adapter_params(model, name):
    return [prm for n, prm in model.named_parameters() if f".{name}." in n and "lora_" in n]


def eos_ids(tok, model):
    e = set()
    for x in (tok.eos_token_id, getattr(model.generation_config, "eos_token_id", None)):
        if isinstance(x, int):
            e.add(x)
        elif isinstance(x, (list, tuple)):
            e.update(x)
    im = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im, int) and im != tok.unk_token_id:
        e.add(im)
    return e


@torch.no_grad()
def hf_generate(model, tok, prompts, n, max_new, temperature, device, greedy=False):
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    out = model.generate(**enc, do_sample=not greedy, temperature=temperature if not greedy else None,
                         top_p=1.0, max_new_tokens=max_new, num_return_sequences=1 if greedy else n,
                         pad_token_id=tok.pad_token_id, eos_token_id=sorted(eos_ids(tok, model)))
    plen = enc["input_ids"].shape[1]
    comp = out[:, plen:]
    eos = eos_ids(tok, model)
    res = []
    for row in comp:
        ids = row.tolist()
        cut = next((i for i, x in enumerate(ids) if x in eos), None)
        if cut is None:
            res.append(([x for x in ids if x != tok.pad_token_id] or ids, True))   # no EOS within budget = truncated
        else:
            res.append((ids[: cut + 1], False))
    return enc["input_ids"].repeat_interleave(1 if greedy else n, 0), enc["attention_mask"].repeat_interleave(1 if greedy else n, 0), res


class VLLMBackend:
    """Multi-LoRA generation on one vLLM engine (colocated). UNTESTED on CPU; the GPU
    smoke stage of run_league.sh must pass before the full run."""
    def __init__(self, merged_dir, a, n_adapters):
        from vllm import LLM
        self.llm = LLM(model=merged_dir, enable_lora=True, max_loras=n_adapters, max_lora_rank=a.lora_r,
                       gpu_memory_utilization=a.vllm_mem, max_model_len=a.max_len, enable_prefix_caching=True,
                       seed=a.seed, dtype="bfloat16")
        self.uid = 0

    def generate(self, prompts, adapter_dir, name, n, max_new, temperature, greedy=False):
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest
        self.uid += 1
        req = LoRARequest(f"{name}-{self.uid}", self.uid, adapter_dir)
        sp = SamplingParams(n=1 if greedy else n, temperature=0.0 if greedy else temperature, max_tokens=max_new)
        outs = self.llm.generate(prompts, sp, lora_request=req, use_tqdm=False)
        res = []
        for o in outs:
            for c in o.outputs:
                res.append((list(c.token_ids), c.finish_reason == "length"))
        return res


def token_logps(model, input_ids, attn, comp_ids, comp_mask, chunk=2):
    """Per-token log-probs of completion tokens; chunked over rows to bound the
    vocab-sized fp32 buffer (Qwen3.5 vocab = 248,320)."""
    full = torch.cat([input_ids, comp_ids], 1)
    mask = torch.cat([attn, comp_mask], 1)
    L = comp_ids.shape[1]
    out = []
    for i in range(0, full.shape[0], chunk):
        logits = model(input_ids=full[i:i + chunk], attention_mask=mask[i:i + chunk], logits_to_keep=L + 1).logits[:, :-1]
        lp = torch.log_softmax(logits.float(), -1).gather(-1, comp_ids[i:i + chunk].unsqueeze(-1)).squeeze(-1)
        out.append(lp)
    return torch.cat(out, 0)


def pad_completions(res, pad_id, device):
    L = max(1, max(len(ids) for ids, _ in res))
    ids = torch.full((len(res), L), pad_id, dtype=torch.long)
    m = torch.zeros((len(res), L), dtype=torch.long)
    for i, (c, _) in enumerate(res):
        if c:
            ids[i, : len(c)] = torch.tensor(c); m[i, : len(c)] = 1
    return ids.to(device), m.to(device)


def policy_loss(lp, old_lp, adv, cmask, spec, max_new):
    lt = spec.get("loss", "dapo")
    if lt == "gspo":
        lr_tok = (lp - old_lp) * cmask
        seq = lr_tok.sum(1) / cmask.sum(1).clamp(min=1)
        s = torch.exp(seq.detach())[:, None] * torch.exp(lp - lp.detach())   # GSPO-token form
        s_clip = torch.clamp(s, 1 - spec.get("eps_low", 3e-4), 1 + spec.get("eps_high", 4e-4))
        per = -torch.minimum(s * adv[:, None], s_clip * adv[:, None])
        return ((per * cmask).sum(1) / cmask.sum(1).clamp(min=1)).mean()
    ratio = torch.exp(lp - old_lp)
    rc = torch.clamp(ratio, 1 - spec.get("eps_low", 0.2), 1 + spec.get("eps_high", 0.28))
    per = -torch.minimum(ratio * adv[:, None], rc * adv[:, None]) * cmask
    if lt == "dr_grpo":
        return per.sum() / (cmask.shape[0] * max_new)
    return per.sum() / cmask.sum().clamp(min=1)                            # dapo token-level


def advantages(rewards_, G, spec):
    r = torch.tensor(rewards_, dtype=torch.float32).view(-1, G)
    a = r - r.mean(1, keepdim=True)
    sc = spec.get("scale_rewards", "batch")
    if sc == "batch":
        a = a / (r.std().clamp(min=1e-4))
    elif sc == "group":
        a = a / (r.std(1, keepdim=True).clamp(min=1e-4))
    keep = (r.std(1) > 1e-6)                                               # DAPO dynamic sampling
    return a.view(-1), keep.repeat_interleave(G)


# ----------------------------------------------------------------- league
class Elo:
    def __init__(self, names, k=16):
        self.r = {n: 1000.0 for n in names}; self.k = k

    def match(self, a, b, sa):
        ea = 1 / (1 + 10 ** ((self.r[b] - self.r[a]) / 400))
        self.r[a] += self.k * (sa - ea); self.r[b] -= self.k * (sa - ea)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--sft-adapter", default=None)
    ap.add_argument("--data", required=True, help="dir with pool.jsonl, dev_v3.jsonl, receipt_store.json, build_receipt.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--budget-min", type=float, default=300.0, help="training wall-clock budget (minutes)")
    ap.add_argument("--rounds", type=int, default=10 ** 9)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--vllm-mem", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=10)
    a = ap.parse_args()
    t0 = time.time()
    cfg = json.load(open(a.config))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "league.log", "a", buffering=1)

    def L(*x):
        s = f"{now()} " + " ".join(str(y) for y in x); print(s); log.write(s + "\n")

    # ---- fail-closed preflight
    st = test_rewards.run()
    if st["failed"]:
        raise SystemExit("REWARD SELF-TEST FAILED")
    br = json.load(open(Path(a.data) / "build_receipt.json"))
    for fn, sha in br["files"].items():
        if sha_file(Path(a.data) / fn) != sha:
            raise SystemExit(f"DATA PIN MISMATCH {fn}")
    if br["verifier_sha256"] != R.verifier_sha256():
        L("WARN verifier changed since data build; receipts record the live verifier sha")
    pool = [json.loads(l) for l in open(Path(a.data) / "pool.jsonl")]
    dev = [json.loads(l) for l in open(Path(a.data) / "dev_v3.jsonl")]
    store = json.load(open(Path(a.data) / "receipt_store.json"))
    blocked = set(json.load(open(Path(a.data) / "sealed_blocklist.json"))["hashes"])
    leak = [it["id"] for it in pool + dev if hashlib.sha256(R.norm(" ".join(m["content"] for m in it["messages"])).encode()).hexdigest() in blocked]
    if leak:
        raise SystemExit(f"SEALED PROBE LEAK {leak[:5]}")
    squad_ans = [it for it in pool if it["gold"]["task"] == "squad" and it["gold"]["label"] == "ANSWER"]

    torch.manual_seed(a.seed); random.seed(a.seed)
    tok, base, LoraConfig, get_peft_model = load_model(a)
    lora = dict(cfg["lora"], r=a.lora_r)
    pols = cfg["policies"]
    model = attach_adapters(base, pols, LoraConfig, get_peft_model, lora)
    if a.device == "cuda":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    opts, samplers = {}, {}
    for p in pols:
        prm = adapter_params(model, p["name"])
        for x in prm:
            x.requires_grad_(True)
        opts[p["name"]] = torch.optim.AdamW(prm, lr=p.get("lr", cfg["lr"]), betas=(0.9, 0.99), weight_decay=0.0)
        samplers[p["name"]] = Sampler(pool, p, p.get("seed", a.seed))
    n_train = sum(x.numel() for x in model.parameters() if x.requires_grad)
    L("adapters", [p["name"] for p in pols], "trainable_params_total", n_train)

    vb = None
    if a.backend == "vllm":
        # vLLM serves the SFT-merged base written by merge_sft.py; adapters are hot-swapped per round
        vb = VLLMBackend(a.base_dir if not a.sft_adapter else cfg["merged_base_dir"], a, len(pols))

    rec = Receipts(out, {"verifier_sha256": R.verifier_sha256(), "verifier_version": R.VERIFIER_VERSION,
                         "sealed_blocklist_sha256": br["sealed"]["blocklist_sha256"],
                         "dev_v3_commitment": br["dev_v3_commit"]["commitment"], "pool_sha256": br["files"]["pool.jsonl"]})
    elo = Elo([p["name"] for p in pols if p.get("role", "main") == "main"])
    G, P, max_new, temp = cfg["num_generations"], cfg["prompts_per_round"], cfg["max_new_tokens"], cfg.get("temperature", 1.0)
    budget_s = a.budget_min * 60
    exploit_queue = collections.deque(maxlen=256)
    hist = []
    rnd = 0

    def gen(name, prompts, n, greedy=False, mx=None):
        mx = mx or max_new
        model.set_adapter(name)
        if vb is not None:
            d = out / "adapters_live" / name
            model.save_pretrained(d, selected_adapters=[name])
            res = vb.generate(prompts, str(d / name) if (d / name).exists() else str(d), name, n, mx, temp, greedy)
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(a.device)
            k = 1 if greedy else n
            return enc["input_ids"].repeat_interleave(k, 0), enc["attention_mask"].repeat_interleave(k, 0), res
        model.eval()
        return hf_generate(model, tok, prompts, n, mx, temp, a.device, greedy)

    def update(name, spec, ids, attn, res, rew):
        adv, keep = advantages(rew, G, spec)
        trunc = torch.tensor([t for _, t in res])
        use = keep & ~trunc if spec.get("mask_truncated", True) else keep
        if use.sum() == 0:
            return {"skipped": "all groups zero-variance or truncated"}
        model.set_adapter(name)
        for prm in adapter_params(model, name):
            prm.requires_grad_(True)
        model.train()
        cids, cmask = pad_completions(res, tok.pad_token_id, a.device)
        idx = torch.nonzero(use).view(-1)
        with torch.no_grad():
            old = torch.cat([token_logps(model, ids[idx[i:i + 4]], attn[idx[i:i + 4]], cids[idx[i:i + 4]], cmask[idx[i:i + 4]]) for i in range(0, len(idx), 4)])
        steps = spec.get("steps_per_generation", 2)
        parts = torch.chunk(torch.arange(len(idx)), steps)
        losses = []
        for part in parts:
            opts[name].zero_grad(set_to_none=True)
            sel = idx[part]
            mb = cfg.get("micro_batch", 4)
            tot = 0.0
            for i in range(0, len(sel), mb):
                s = sel[i:i + mb]; j = part[i:i + mb]
                lp = token_logps(model, ids[s], attn[s], cids[s], cmask[s])
                loss = policy_loss(lp, old[j], adv[s].to(a.device), cmask[s].float(), spec, max_new) * (len(s) / len(sel))
                loss.backward(); tot += float(loss)
            torch.nn.utils.clip_grad_norm_(adapter_params(model, name), 1.0)
            opts[name].step(); losses.append(tot)
        return {"loss": sum(losses) / len(losses), "used": int(use.sum()), "of": len(use)}

    while rnd < a.rounds and time.time() - t0 < budget_s:
        rnd += 1
        frac = (time.time() - t0) / budget_s
        stats = {"round": rnd, "at": now(), "elapsed_min": round((time.time() - t0) / 60, 2)}
        # ---- exploiter turn first: propose verified probes, reward = how often mains fail them
        for p in [p for p in pols if p.get("role") == "exploiter"]:
            rng = samplers[p["name"]].rng
            xps = [x for x in (exploiter_prompt(rng.choice(squad_ans)) for _ in range(P * 2)) if x][:P]
            if not xps:
                continue
            prompts = [render(tok, x["messages"]) for x in xps]
            ids, attn, res = gen(p["name"], prompts, G)
            rew, probes = [], []
            for k, (cids_, tr) in enumerate(res):
                x = xps[k // G]
                text = tok.decode(cids_, skip_special_tokens=True)
                probe, why = (None, "truncated") if tr else exploiter_verify(x, text)
                probes.append((probe, why, text))
            valid = [pr for pr, _, _ in probes if pr]
            fail = {}
            if valid:
                mains = [m for m in pols if m.get("role", "main") == "main"]
                for m in mains:
                    pp = [render(tok, compose(pr, None, False)) for pr in valid]
                    _, _, mres = gen(m["name"], pp, 1, greedy=True, mx=48)
                    for pr, (mc, mtr) in zip(valid, mres):
                        s = R.score_completion(dict(pr["gold"], t=0.75), tok.decode(mc, skip_special_tokens=True), mtr, store, components=("task",))
                        fail.setdefault(pr["id"], []).append(0 if s["outcome"] in ("correct", "abstain_ok") else 1)
            for (pr, why, text) in probes:
                if pr is None:
                    r_ = -1.0
                else:
                    fr = sum(fail[pr["id"]]) / len(fail[pr["id"]])
                    r_ = 0.25 + fr * (1.0 if fr < 1.0 else 0.5)   # all-mains-fail is suspicious: half credit
                    exploit_queue.append(pr)
                rec.write({"round": rnd, "policy": p["name"], "role": "exploiter", "prompt_sha256": R.sha256_text(pr["messages"][0]["content"]) if pr else None,
                           "verdict": why, "total": r_, "seed": p.get("seed")}, text)
                rew.append(r_)
            mr, n_rec = rec.seal_round(rnd, p["name"])
            if n_rec != len(rew):
                raise SystemExit("RECEIPT COUNT MISMATCH (T1): refusing rewards")
            u = update(p["name"], p, ids, attn, res, rew)
            stats[p["name"]] = {"reward_mean": sum(rew) / len(rew), "valid_probes": len(valid), **u, "merkle": mr["merkle_root"][:16]}
        # ---- main policies
        for p in [p for p in pols if p.get("role", "main") == "main"]:
            draws = samplers[p["name"]].draw(P, frac)
            share = p.get("exploit_share", cfg.get("exploit_share", 0.25))
            n_x = min(int(P * share), len(exploit_queue))
            for k in range(n_x):
                draws[k] = (exploit_queue[-1 - k], samplers[p["name"]].threshold(frac), False)
            prompts = [render(tok, compose(it, t, wr)) for it, t, wr in draws]
            ids, attn, res = gen(p["name"], prompts, G)
            rew, outc = [], collections.Counter()
            comps = tuple(p.get("components", ["task", "proof", "report", "format"]))
            for k, (cids_, tr) in enumerate(res):
                it, t, wr = draws[k // G]
                gold = dict(it["gold"], t=(t if t is not None else 0.75))
                text = tok.decode(cids_, skip_special_tokens=True)
                s = R.score_completion(gold, text, tr, store, components=comps, weights=p.get("weights"), context=ctx_of(it))
                rec.write({"round": rnd, "policy": p["name"], "role": "main", "prompt_sha256": it.get("prompt_sha256") or R.sha256_text(prompts[k // G]),
                           "item_id": it["id"], "task": gold["task"], "label": gold["label"], "t": t, "report_requested": wr,
                           "outcome": s["outcome"], "parts": s["parts"], "total": s["total"], "truncated": bool(tr), "weights": p.get("weights"), "seed": p.get("seed")}, text)
                rew.append(s["total"]); outc[s["outcome"]] += 1
            mr, n_rec = rec.seal_round(rnd, p["name"])
            if n_rec != len(rew):
                raise SystemExit("RECEIPT COUNT MISMATCH (T1): refusing rewards")
            u = update(p["name"], p, ids, attn, res, rew)
            stats[p["name"]] = {"reward_mean": round(sum(rew) / len(rew), 4), "outcomes": dict(outc), **u, "merkle": mr["merkle_root"][:16]}
        # ---- league standings on dev_v3 (greedy, pairwise per item -> Elo)
        if rnd % a.eval_every == 0 or time.time() - t0 >= budget_s:
            sub = random.Random(rnd).sample(dev, min(len(dev), cfg.get("dev_eval_n", 64)))
            scores = {}
            for p in [p for p in pols if p.get("role", "main") == "main"]:
                pr = [render(tok, compose(it, None, False)) for it in sub]
                _, _, res = gen(p["name"], pr, 1, greedy=True)
                scores[p["name"]] = [R.score_completion(dict(it["gold"], t=0.75), tok.decode(c, skip_special_tokens=True), tr, store, components=("task", "proof"), context=ctx_of(it))["total"] for it, (c, tr) in zip(sub, res)]
            names = list(scores)
            for i in range(len(sub)):
                for x in range(len(names)):
                    for y in range(x + 1, len(names)):
                        sa, sb = scores[names[x]][i], scores[names[y]][i]
                        elo.match(names[x], names[y], 1.0 if sa > sb else (0.0 if sa < sb else 0.5))
            stats["league"] = {"dev_mean": {n: round(sum(v) / len(v), 4) for n, v in scores.items()}, "elo": {n: round(v, 1) for n, v in elo.r.items()}}
        if rnd % a.save_every == 0:
            model.save_pretrained(out / "adapters" / f"round{rnd:04d}")
        hist.append(stats); L(json.dumps(stats))
    model.save_pretrained(out / "adapters" / "final")
    summary = {"schema": "szl.league.train-receipt/v1", "status": "MEASURED", "signature": "UNSIGNED", "rounds": rnd,
               "wall_min": round((time.time() - t0) / 60, 2), "policies": pols, "elo_dev_v3": elo.r,
               "adapters_dir": str(out / "adapters" / "final"), "verifier_sha256": R.verifier_sha256(),
               "build_receipt_sha256": sha_file(Path(a.data) / "build_receipt.json"),
               "labels": {"promotion": "HOLD", "publication": "none", "release": "NOT-FOR-RELEASE until rights + gates clear"}}
    json.dump(summary, open(out / "train_receipt.json", "w"), indent=1, default=str)
    L("DONE", json.dumps({"rounds": rnd, "elo": elo.r}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU dry-run: validates code paths, data, rewards, receipts and the tournament.
Uses a TINY randomly initialised Qwen3.5-architecture model with the REAL pinned
Qwen3.5-0.8B tokenizer + chat template. Says nothing about quality or GPU memory."""
import json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
PY = sys.executable
BASE_REPO, BASE_REV = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
TOK_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt", "config.json"]


def step(name, fn, res):
    t = time.time()
    try:
        out = fn(); res[name] = {"ok": True, "s": round(time.time() - t, 1), **(out or {})}
    except Exception as e:
        res[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(name, json.dumps(res[name])[:600], flush=True)
    return res[name]["ok"]


def gold_selfconsistency():
    import rewards as R
    canon = {
        ("grounded", "ANSWER"): lambda g: f"ANSWER\n{g['answer']}\n{g['source_line']}",
        ("grounded", "ABSTAIN"): lambda g: "ABSTAIN\nThe context does not contain this.",
        ("squad", "ANSWER"): lambda g: g["answers"][0],
        ("squad", "ABSTAIN"): lambda g: "UNANSWERABLE",
        ("khipu_compute", "ANSWER"): lambda g: str(g["value"]),
        ("khipu_compute", "ABSTAIN"): lambda g: "ABSTAIN - not all cords are shown.",
        ("khipu_route", "ABSTAIN"): lambda g: '{"decision":"ABSTAIN","citedNodeIds":[],"abstainReason":"no handle supports it"}',
        ("openprob", "OPEN"): lambda g: "No. It is recorded as an open problem; there is no accepted proof.",
        ("openprob", "SOLVED"): lambda g: "Yes. The entry records it as solved.",
        ("false_premise", "ABSTAIN"): lambda g: "The premise is false, so I cannot answer as asked.",
        ("receipt_claim", "VERIFIED"): lambda g: f"Yes, it was machine-verified: {g['receipt_id']} (PASSED).",
        ("receipt_claim", "UNVERIFIED"): lambda g: "UNVERIFIED. No PASSED receipt for this theorem is listed.",
    }
    store = json.load(open(ROOT / "data" / "receipt_store.json"))
    bad, n = [], 0
    for f in ("pool.jsonl", "dev_v3.jsonl"):
        for l in open(ROOT / "data" / f):
            it = json.loads(l); g = dict(it["gold"], t=0.75)
            s = R.score_completion(g, canon[(g["task"], g["label"])](g), False, store, components=("task", "proof"), context="\n".join(m["content"] for m in it["messages"]))
            n += 1
            if s["outcome"] not in ("correct", "abstain_ok") or s["parts"]["proof"] < 0:
                bad.append({"id": it["id"], "task": g["task"], "label": g["label"], "outcome": s["outcome"], "proof": s["parts"].get("proof")})
    if bad:
        json.dump(bad, open(ROOT / "dryrun" / "gold_selfconsistency_failures.json", "w"), indent=1)
    if len(bad) > 0.005 * n:
        raise RuntimeError(f"{len(bad)}/{n} canonical gold answers not rewarded as correct (see failures file)")
    return {"items": n, "canonical_gold_not_correct": len(bad)}


def tiny_model():
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.models.qwen3_5 import Qwen3_5TextConfig
    d = ROOT / "dryrun" / "tiny-qwen3_5"; d.mkdir(parents=True, exist_ok=True)
    real = None
    for f in TOK_FILES:
        p = hf_hub_download(BASE_REPO, f, revision=BASE_REV)
        if f == "config.json":
            real = json.load(open(p))
        else:
            (d / f).write_bytes(Path(p).read_bytes())
    tc = real.get("text_config", real)
    cfg = Qwen3_5TextConfig(vocab_size=tc["vocab_size"], hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=32,
                            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                            linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
                            tie_word_embeddings=True, max_position_embeddings=4096,
                            eos_token_id=tc.get("eos_token_id"), bos_token_id=tc.get("bos_token_id"), pad_token_id=tc.get("pad_token_id"))
    import torch
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg)
    m.save_pretrained(d)
    tok = AutoTokenizer.from_pretrained(d)
    return {"dir": str(d), "params": sum(p.numel() for p in m.parameters()), "vocab": tc["vocab_size"],
            "arch": type(m).__name__, "chat_template_ok": "<|im_start|>" in tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True)}


def run_league():
    out = ROOT / "dryrun" / "league"
    import shutil; shutil.rmtree(out, ignore_errors=True)
    cmd = [PY, str(HERE / "league_grpo.py"), "--config", str(HERE / "configs" / "dryrun_cpu.json"),
           "--base-dir", str(ROOT / "dryrun" / "tiny-qwen3_5"), "--data", str(ROOT / "data"), "--out", str(out),
           "--device", "cpu", "--backend", "hf", "--rounds", "2", "--budget-min", "30", "--eval-every", "2", "--save-every", "2", "--lora-r", "4"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    (ROOT / "dryrun" / "league_stdout.txt").write_text(p.stdout[-20000:] + "\n--- stderr ---\n" + p.stderr[-20000:])
    if p.returncode != 0:
        raise RuntimeError(f"league rc={p.returncode}: {p.stderr[-1500:]}")
    tr = json.load(open(out / "train_receipt.json"))
    rounds = [json.loads(l.split(" ", 1)[1]) for l in open(out / "league.log") if l.split(" ", 1)[1].startswith("{")]
    return {"rounds": tr["rounds"], "elo_dev_v3": tr["elo_dev_v3"], "last_round": rounds[-1]}


def gradient_path():
    """Real policy-gradient step on the tiny model for every loss type; only the
    selected adapter may change (multi-adapter isolation)."""
    import torch
    from league_grpo import load_model, attach_adapters, adapter_params, token_logps, policy_loss, advantages, pad_completions
    class A: pass
    a = A(); a.device = "cpu"; a.base_dir = str(ROOT / "dryrun" / "tiny-qwen3_5"); a.sft_adapter = None
    tok, base, LoraConfig, gpm = load_model(a)
    cfg = json.load(open(HERE / "configs" / "dryrun_cpu.json"))
    pols = [{"name": "pa"}, {"name": "pb"}]
    model = attach_adapters(base, pols, LoraConfig, gpm, dict(cfg["lora"], r=4))
    torch.manual_seed(0)
    G = 4
    ids = torch.randint(0, 1000, (G, 7)); attn = torch.ones_like(ids)
    res = [(torch.randint(0, 1000, (5,)).tolist(), False) for _ in range(G)]
    cids, cmask = pad_completions(res, tok.pad_token_id, "cpu")
    out = {}
    for loss in ("dapo", "dr_grpo", "gspo"):
        spec = {"loss": loss, "scale_rewards": "none" if loss == "dr_grpo" else "batch"}
        if loss == "gspo":
            spec.update(eps_low=3e-4, eps_high=4e-4)
        model.set_adapter("pa")
        pa, pb = adapter_params(model, "pa"), adapter_params(model, "pb")
        for x in pa: x.requires_grad_(True)
        for x in pb: x.requires_grad_(False)
        # LoRA B starts at zero -> give A/B small values so gradients flow to both factors
        with torch.no_grad():
            for x in pa: x.add_(0.01 * torch.randn_like(x))
        before_a = [x.detach().clone() for x in pa]; before_b = [x.detach().clone() for x in pb]
        adv, keep = advantages([1.0, -3.0, 1.0, -0.5], G, spec)
        with torch.no_grad():
            old = token_logps(model, ids, attn, cids, cmask)
        opt = torch.optim.AdamW(pa, lr=1e-2)
        lp = token_logps(model, ids, attn, cids, cmask)
        l = policy_loss(lp, old, adv, cmask.float(), spec, 5)
        l.backward()
        gnorm = float(torch.sqrt(sum((x.grad ** 2).sum() for x in pa if x.grad is not None)))
        opt.step(); opt.zero_grad(set_to_none=True)
        moved_a = any(not torch.equal(x, y) for x, y in zip(pa, before_a))
        moved_b = any(not torch.equal(x, y) for x, y in zip(pb, before_b))
        if not (gnorm > 0 and moved_a and not moved_b):
            raise RuntimeError(f"{loss}: grad={gnorm} moved_a={moved_a} moved_b={moved_b}")
        out[loss] = {"loss": round(float(l.detach()), 6), "grad_norm": round(gnorm, 6), "adapter_isolated": True}
    return out


def replay():
    import replay_receipts
    r = replay_receipts.main(str(ROOT / "dryrun" / "league"), str(ROOT / "data"))
    if not r["ok"]:
        raise RuntimeError("receipt replay mismatch")
    return {k: r[k] for k in ("rows", "rescored", "merkle_rounds", "n_mismatch", "merkle_mismatch")}


def tournament():
    b = os.environ.get("SZL_SEALED_DIR", "sealed")  # private probes + pinned scorers, never committed
    cmd = [PY, str(HERE / "tournament.py"), "--config", str(HERE / "configs" / "dryrun_cpu.json"),
           "--base-dir", str(ROOT / "dryrun" / "tiny-qwen3_5"), "--adapters", str(ROOT / "dryrun" / "league" / "adapters" / "final"),
           "--data", str(ROOT / "data"), "--probes-v1", f"{b}/probes.jsonl", "--probes-v2", f"{b}/probes_v2.jsonl",
           "--scorers-dir", b, "--out", str(ROOT / "dryrun" / "tournament"), "--device", "cpu", "--limit", "6", "--boot", "50"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    (ROOT / "dryrun" / "tournament_stdout.txt").write_text(p.stdout[-20000:] + "\n--- stderr ---\n" + p.stderr[-20000:])
    if p.returncode != 0:
        raise RuntimeError(f"tournament rc={p.returncode}: {p.stderr[-1500:]}")
    t = json.load(open(ROOT / "dryrun" / "tournament" / "tournament.json"))
    return {"selected": t["selection"]["selected"], "dev_v3_reveal_ok": t["dev_v3_reveal"]["ok"], "bradley_terry_sealed": t["bradley_terry_sealed"],
            "promotion": t["promotion"], "note": "limit=6 items per set: code-path check only"}


def main():
    (ROOT / "dryrun").mkdir(exist_ok=True)
    os.environ.pop("HF_TOKEN", None); os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    res = {"schema": "szl.league.dryrun-cpu/v1", "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    import test_rewards
    step("reward_selftests", lambda: {"failed": test_rewards.run()["failed"]}, res)
    step("data_build_receipt", lambda: {k: json.load(open(ROOT / "data" / "build_receipt.json"))[k] for k in ("pool_rows", "dev_rows")}, res)
    step("gold_selfconsistency", gold_selfconsistency, res)
    ok = step("tiny_model", tiny_model, res)
    ok = ok and step("gradient_path_all_losses", gradient_path, res)
    ok = ok and step("league_2_rounds_3mains_1exploiter", run_league, res)
    ok = ok and step("receipt_replay_T1", replay, res)
    ok = ok and step("tournament_smoke", tournament, res)
    res["ok"] = all(v.get("ok") for k, v in res.items() if isinstance(v, dict))
    res["note"] = "CPU, tiny random model (real tokenizer): validates code paths, data, rewards, receipts, tournament. NOT quality, NOT GPU memory, NOT vLLM."
    json.dump(res, open(ROOT / "dryrun" / "dryrun_cpu.json", "w"), indent=1, default=str)
    print("DRYRUN OK" if res["ok"] else "DRYRUN FAILED")
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()

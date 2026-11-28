#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU dry-run for the Hamut'aq-4B recipe (no GPU, no spend, no upload).

The full 4B bf16 checkpoint (7.95 GB) does not fit the prep box's free RAM, so this:
  1. reads the pinned model.safetensors header with HTTP range requests and checks every one of the 262 tensors
     against the native transformers NemotronH model built on the meta device (full-size shape validation);
  2. range-fetches the real weights of embeddings, norm_f, lm_head and one block of each kind
     (layer 0 Mamba-2, layer 1 MLP, layer 12 attention) into a 3-block local checkpoint ("M-*" pattern);
  3. loads it with native from_pretrained (checks the backbone.* -> model.* key mapping, no missing keys);
  4. checks the chat template (reasoning off) matches the trainer's supervised-span construction;
  5. tokenizes the full prepared SFT/DPO files (token counts for the cost estimate);
  6. runs train_sft.py, train_dpo.py and run_eval.py end to end on CPU on toy batches.
"""
import json, os, struct, subprocess, sys, time, urllib.request, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import BASE_ID, BASE_REV, encode_sft, encode_pair, render_prompt, lora_targets, sha256_file, now  # noqa

URL = f"https://huggingface.co/{BASE_ID}/resolve/{BASE_REV}/model.safetensors"
KEEP = {0: 0, 1: 1, 12: 2}  # source layer -> tiny layer


def rng(a, b):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b}"})
    return urllib.request.urlopen(req, timeout=600).read()


def main():
    work, prepared, pin = sys.argv[1], sys.argv[2], sys.argv[3]
    out = {"schema": "szl.hamutaq.dryrun-cpu/v1", "at": now(), "base": {"repo": BASE_ID, "revision": BASE_REV}}
    import torch, transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    out["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__, "python": sys.version.split()[0]}
    n = struct.unpack("<Q", rng(0, 7))[0]
    header = json.loads(rng(8, 8 + n - 1)); data0 = 8 + n
    # 1. full-size shape validation on meta
    cfg = AutoConfig.from_pretrained(pin)
    with torch.device("meta"):
        full = AutoModelForCausalLM.from_config(cfg)
    sd = {k: tuple(v.shape) for k, v in full.state_dict().items()}
    mism, missing = [], []
    for k, v in header.items():
        if k == "__metadata__":
            continue
        k2 = "model." + k[len("backbone."):] if k.startswith("backbone.") else k
        if k2 not in sd:
            missing.append(k)
        elif tuple(v["shape"]) != sd[k2]:
            mism.append((k, v["shape"], sd[k2]))
    unl = [k for k in sd if ("backbone." + k[len("model."):] if k.startswith("model.") else k) not in header]
    lin = {}
    for name, mod in full.named_modules():
        if isinstance(mod, torch.nn.Linear):
            lin.setdefault(name.split(".")[-1], []).append(tuple(mod.weight.shape))
    out["full_shape_check"] = {"checkpoint_tensors": len(header) - 1, "model_params": len(sd), "missing_in_model": missing,
                               "shape_mismatches": mism, "model_params_not_in_checkpoint": unl,
                               "params_total": sum(p.numel() for p in full.parameters()),
                               "linear_modules": {k: {"count": len(v), "shape_out_in": list(v[0])} for k, v in lin.items()},
                               "ok": not missing and not mism and not unl}
    lt = lora_targets()
    out["lora"] = {"targets": lt, "trainable_params_r32_est": sum(
        32 * (s[0] + s[1]) * len(lin[t]) for t in lt for s in [lin[t][0]])}
    del full
    # 2. tiny real-weight checkpoint
    tiny = os.path.join(work, "tiny-3block")
    os.makedirs(tiny, exist_ok=True)
    c = json.load(open(os.path.join(pin, "config.json")))
    pat = c["hybrid_override_pattern"]
    c["hybrid_override_pattern"] = "".join(pat[i] for i in sorted(KEEP)); c["num_hidden_layers"] = len(KEEP)
    c.pop("auto_map", None)
    json.dump(c, open(os.path.join(tiny, "config.json"), "w"), indent=1)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "generation_config.json"):
        p = os.path.join(pin, f)
        if os.path.exists(p):
            open(os.path.join(tiny, f), "wb").write(open(p, "rb").read())
    from safetensors.torch import save_file
    tensors, t0, fetched = {}, time.time(), 0
    for k, v in header.items():
        if k == "__metadata__":
            continue
        parts = k.split(".")
        if parts[1] == "layers":
            li = int(parts[2])
            if li not in KEEP:
                continue
            nk = ".".join(parts[:2] + [str(KEEP[li])] + parts[3:])
        else:
            nk = k
        s, e = v["data_offsets"]
        buf = bytearray(rng(data0 + s, data0 + e - 1)); fetched += e - s
        tensors[nk] = torch.frombuffer(buf, dtype=torch.bfloat16 if v["dtype"] == "BF16" else torch.float32).reshape(v["shape"]).clone()
    save_file(tensors, os.path.join(tiny, "model.safetensors"), metadata={"format": "pt"})
    del tensors
    out["tiny_checkpoint"] = {"source_layers": sorted(KEEP), "pattern": c["hybrid_override_pattern"], "bytes_fetched": fetched,
                              "fetch_seconds": round(time.time() - t0, 1)}
    # 3. native load of real weights
    m, info = AutoModelForCausalLM.from_pretrained(tiny, dtype=torch.bfloat16, output_loading_info=True)
    out["native_load"] = {"class": type(m).__name__, **{k: list(v)[:10] for k, v in info.items()},
                          "ok": not info.get("missing_keys") and not info.get("unexpected_keys") and not info.get("mismatched_keys")}
    tok = AutoTokenizer.from_pretrained(tiny)
    # 4. template consistency, reasoning off
    msgs = [{"role": "user", "content": "Is the Collatz-free toy question open?"}, {"role": "assistant", "content": "It is open."}]
    full_txt = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
    pr = render_prompt(tok, msgs[:1], thinking=False)
    on = render_prompt(tok, msgs[:1], thinking=True)
    out["template"] = {"prompt_off_tail": pr[-60:], "prompt_on_tail": on[-60:], "full_render_tail": full_txt[-60:],
                       "consistent": full_txt.startswith(pr) and full_txt[len(pr):].strip().startswith("It is open.") and "<|im_end|>" in full_txt[len(pr):],
                       "eos": tok.eos_token, "eos_id": tok.eos_token_id, "pad_id": tok.pad_token_id}
    # quick forward on real weights
    with torch.no_grad():
        ids = tok(pr, return_tensors="pt", add_special_tokens=False)
        lg = m(**ids).logits
    out["forward_real_weights"] = {"logits_shape": list(lg.shape), "finite": bool(torch.isfinite(lg.float()).all())}
    del m
    # 5. tokenize the prepared data
    st = {}
    for name in ("sft_train.jsonl", "sft_val.jsonl"):
        lens, over = [], 0
        for l in open(os.path.join(prepared, name), encoding="utf-8"):
            e = encode_sft(tok, json.loads(l)["messages"], 10 ** 9)
            lens.append(len(e[0]))
        st[name] = {"rows": len(lens), "tokens": sum(lens), "median": statistics.median(lens), "p95": sorted(lens)[int(.95 * len(lens))],
                    "max": max(lens), "over_2048": sum(x > 2048 for x in lens), "tokens_le_2048": sum(x for x in lens if x <= 2048)}
    pl = []
    for l in open(os.path.join(prepared, "dpo_train.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        p, cc, rr = encode_pair(tok, r["prompt"], r["chosen"][-1]["content"], r["rejected"][-1]["content"], 10 ** 9)
        pl.append(2 * len(p) + len(cc) + len(rr))
    st["dpo_train.jsonl"] = {"pairs": len(pl), "tokens_both_sides": sum(pl), "max": max(pl)}
    out["tokenized"] = st
    json.dump(out, open(os.path.join(work, "dryrun_partial.json"), "w"), indent=1)
    # 6. end-to-end scripts on CPU
    py = sys.executable
    env = dict(os.environ)
    sft_sha = sha256_file(os.path.join(prepared, "sft_train.jsonl"))
    dpo_sha = sha256_file(os.path.join(prepared, "dpo_train.jsonl"))
    runs = {}
    cmds = {
        "train_sft": [py, os.path.join(HERE, "train_sft.py"), "--local-base", tiny, "--device", "cpu", "--train", os.path.join(prepared, "sft_train.jsonl"),
                      "--train-sha256", sft_sha, "--val", os.path.join(prepared, "sft_val.jsonl"), "--val-rows", "4", "--out", os.path.join(work, "sft"),
                      "--max-rows", "16", "--max-len", "384", "--tok-budget", "768", "--grad-accum", "2", "--max-steps", "2", "--max-seconds", "1800"],
        "train_dpo": [py, os.path.join(HERE, "train_dpo.py"), "--local-base", tiny, "--device", "cpu", "--sft-adapter", os.path.join(work, "sft", "final"),
                      "--pairs", os.path.join(prepared, "dpo_train.jsonl"), "--pairs-sha256", dpo_sha, "--out", os.path.join(work, "dpo"),
                      "--max-rows", "4", "--max-len", "384", "--grad-accum", "2", "--max-steps", "1", "--max-seconds", "1800"],
    }
    for name, cmd in cmds.items():
        t0 = time.time(); p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, env=env)
        runs[name] = {"rc": p.returncode, "seconds": round(time.time() - t0, 1), "tail": (p.stdout + p.stderr)[-1500:]}
    rec_sft = json.load(open(os.path.join(work, "sft", "final", "train_receipt.json"))) if runs["train_sft"]["rc"] == 0 else None
    rec_dpo = json.load(open(os.path.join(work, "dpo", "final", "train_receipt.json"))) if runs["train_dpo"]["rc"] == 0 else None
    env.update({"ABSTEVAL_LOCAL_BASE": tiny, "ABSTEVAL_DEVICE": "cpu", "ABSTEVAL_LIMIT": "2", "ABSTEVAL_MAX_NEW": "6",
                "ABSTEVAL_OUT": os.path.join(work, "eval"), "ABSTEVAL_ADAPTERS": os.path.join(work, "sft", "final") + "+" + os.path.join(work, "dpo", "final"),
                "ABSTEVAL_PROBES_V1": os.path.join(prepared, "..", "..", "probes_v1.jsonl"), "ABSTEVAL_PROBES_V2": os.path.join(prepared, "..", "..", "probes_v2.jsonl")})
    env["ABSTEVAL_PROBES_V1"] = os.environ.get("PROBES_V1", ""); env["ABSTEVAL_PROBES_V2"] = os.environ.get("PROBES_V2", "")
    t0 = time.time(); p = subprocess.run([py, os.path.join(HERE, "run_eval.py")], cwd=HERE, capture_output=True, text=True, env=env)
    runs["run_eval"] = {"rc": p.returncode, "seconds": round(time.time() - t0, 1), "tail": (p.stdout + p.stderr)[-1200:]}
    lb = os.path.join(work, "eval", "leaderboard.json")
    out["e2e"] = {k: {kk: vv for kk, vv in v.items() if kk != "tail" or v["rc"] != 0} for k, v in runs.items()}
    out["e2e"]["sft_receipt"] = {k: rec_sft[k] for k in ("status", "steps", "trainable_params", "lora_targets", "loss", "val_loss", "tokens", "tok_per_s")} if rec_sft else None
    out["e2e"]["dpo_receipt"] = {k: rec_dpo[k] for k in ("status", "steps", "loss", "reward_acc_last")} if rec_dpo else None
    out["e2e"]["eval_leaderboard_models"] = [e["model"].split("/")[-1] if "+" not in e["model"] else "sft+dpo" for e in json.load(open(lb))["entries"]] if os.path.exists(lb) else None
    out["e2e"]["eval_entries_blocked"] = [e for e in json.load(open(lb))["entries"] if e.get("status") == "BLOCKED"] if os.path.exists(lb) else None
    out["note"] = "CPU, 3 real blocks of 42, toy batches: validates code paths, shapes and data, NOT quality or GPU memory."
    out["ok"] = bool(out["full_shape_check"]["ok"] and out["native_load"]["ok"] and out["template"]["consistent"]
                     and all(v["rc"] == 0 for v in runs.values()))
    print(json.dumps(out, indent=1))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

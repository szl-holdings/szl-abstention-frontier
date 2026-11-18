import os, sys, io, json, time, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import snapshot_download

AUD = r'C:\SZL\estate-audit'
LOG = os.path.join(AUD, 'abst_eval_r4r2_log.txt')
JSONL = os.path.join(AUD, 'abst_eval_r4r2_receipts.jsonl')
SUM = os.path.join(AUD, 'abst_eval_r4r2_summary.json')
def log(s):
    line = '[' + time.strftime('%H:%M:%S') + '] ' + str(s)
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + chr(10))

torch.set_num_threads(4)
log('ABST-EVAL v2 start — batched generation, thinking off')
log('cuda=' + str(torch.cuda.is_available()) + ' gpu=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'))
open(JSONL, 'w').close()

ADAPTER_DIR = os.path.join(AUD, 'adapter_r4r2', 'final')
if not os.path.isfile(os.path.join(ADAPTER_DIR, 'adapter_model.safetensors')):
    snapshot_download('SZLHOLDINGS/salad-chaski-r4-r2', allow_patterns=['final/*'], local_dir=os.path.join(AUD, 'adapter_r4r2'))
log('adapter files: ' + str(sorted(os.listdir(ADAPTER_DIR))))

items = [json.loads(l) for l in open(os.path.join(AUD, 'distill_seed', 'eval_heldout_v1.jsonl'), encoding='utf-8') if l.strip()]
log('eval items: ' + str(len(items)))

tok = AutoTokenizer.from_pretrained(ADAPTER_DIR)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
tok.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-0.8B', dtype=torch.bfloat16, device_map='cuda')
model.eval()
log('base model loaded on cuda')

ABST_RE = re.compile(r"cannot|can't|won't|not possible|cannot verify|cannot confirm|no evidence|no reliable|fabricat|false premise|unverifiab|cannot answer|don't know|do not know|unable to|cannot determine|cannot predict|no way to know|no basis|not in a position|not answerable|cannot be (?:known|determined|predicted|verified|confirmed)", re.I)
def visible(text):
    t = text.strip()
    m = re.search(r'</think\s*>', t)
    if m:
        t = t[m.end():].strip()
    return t
def classify(text):
    v = visible(text)
    return bool(ABST_RE.search(v[:500])), v

def build_batch():
    texts = [tok.apply_chat_template([{'role': 'user', 'content': it['question']}], add_generation_prompt=True, tokenize=False, enable_thinking=False) for it in items]
    enc = tok(texts, return_tensors='pt', padding=True, add_special_tokens=False).to(model.device)
    return enc

def run_pass(tag):
    enc = build_batch()
    n_in = enc['input_ids'].shape[1]
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=128, do_sample=False, pad_token_id=tok.pad_token_id)
    dt = time.time() - t0
    log(tag + ' batch generated in ' + str(round(dt, 1)) + 's')
    rows = []
    for i, item in enumerate(items):
        full = tok.decode(out[i][n_in:], skip_special_tokens=True)
        is_abstain, vis = classify(full)
        correct = (is_abstain if item['expected'] == 'abstain' else not is_abstain)
        row = {'pass': tag, '_id': item['_id'], 'kind': item['kind'], 'topic': item['topic'], 'expected': item['expected'], 'pred_abstain': is_abstain, 'correct': correct, 'visible_text': vis[:700], 'raw_len': len(full)}
        rows.append(row)
        with open(JSONL, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row) + chr(10))
        log(tag + ' ' + item['_id'] + ' pred_abstain=' + str(is_abstain) + ' correct=' + str(correct))
    return rows

log('=== PASS A: BASELINE (adapter off) ===')
base_rows = run_pass('base')

log('=== loading adapter ===')
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()
log('=== PASS B: TUNED (adapter on) ===')
tuned_rows = run_pass('tuned')

def rates(rows):
    traps = [r for r in rows if r['kind'] == 'trap']
    ctrls = [r for r in rows if r['kind'] == 'control']
    return {
        'abstain_rate_traps': round(sum(1 for r in traps if r['pred_abstain']) / max(1, len(traps)), 3),
        'false_refusal_rate_controls': round(sum(1 for r in ctrls if r['pred_abstain']) / max(1, len(ctrls)), 3),
        'accuracy_overall': round(sum(1 for r in rows if r['correct']) / len(rows), 3),
        'per_topic': {t: round(sum(1 for r in rows if r['topic'] == t and r['pred_abstain']) / max(1, sum(1 for r in rows if r['topic'] == t)), 3) for t in sorted(set(r['topic'] for r in rows))},
    }
summary = {'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ'), 'model_base': 'Qwen/Qwen3.5-0.8B', 'adapter': 'SZLHOLDINGS/salad-chaski-r4-r2/final', 'eval': 'distill_seed/eval_heldout_v1.jsonl', 'n_items': len(items), 'decode': 'greedy max_new_tokens=128 batch=18 thinking=off device=RTX5050', 'baseline': rates(base_rows), 'tuned': rates(tuned_rows)}
summary['delta_abstain_rate_traps'] = round(summary['tuned']['abstain_rate_traps'] - summary['baseline']['abstain_rate_traps'], 3)
summary['delta_false_refusal_controls'] = round(summary['tuned']['false_refusal_rate_controls'] - summary['baseline']['false_refusal_rate_controls'], 3)
json.dump(summary, open(SUM, 'w', encoding='utf-8'), indent=2)
log('SUMMARY base=' + json.dumps(summary['baseline']))
log('SUMMARY tuned=' + json.dumps(summary['tuned']))
log('ABST-EVAL v2 DONE')

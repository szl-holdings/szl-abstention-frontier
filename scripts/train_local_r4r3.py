import os, sys, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

AUD = os.path.join('C:' + chr(92), 'SZL', 'estate-audit')
LOG = os.path.join(AUD, 'r4r3_local_train_log.txt')
FINAL = os.path.join(AUD, 'adapter_r4r3_local', 'final')
def log(s):
    line = '[' + time.strftime('%H:%M:%S') + '] ' + str(s)
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + chr(10))

torch.set_num_threads(4)
log('LOCAL TRAINER r4-r3 start (Salad-independent fallback)')
free, total = torch.cuda.mem_get_info(0) if torch.cuda.is_available() else (0, 0)
log('cuda=' + str(torch.cuda.is_available()) + ' gpu=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none') + ' vram_free=' + str(round(free/2**30, 2)) + 'GiB/' + str(round(total/2**30, 2)) + 'GiB')

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B')
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
tok.padding_side = 'right'

rows = [json.loads(l) for l in open(os.path.join(AUD, 'distill_seed_v2', 'data', 'train.jsonl'), encoding='utf-8') if l.strip()]
log('train rows: ' + str(len(rows)))
assert len(rows) == 72, 'expected 72 rows'
assert all('messages' in r and len(r['messages']) == 2 for r in rows), 'bad row schema'

IM_END = '<|im_end|>'
def encode(r):
    msgs = r['messages']
    prompt_text = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full_text = prompt_text + msgs[1]['content'] + IM_END
    ids_full = tok(full_text, add_special_tokens=False)['input_ids']
    ids_prompt = tok(prompt_text, add_special_tokens=False)['input_ids']
    np_ = min(len(ids_prompt), len(ids_full))
    labels = [-100] * np_ + ids_full[np_:]
    return {'input_ids': ids_full, 'labels': labels}

enc0 = encode(rows[0])
log('sample0: prompt_len=' + str(enc0['labels'].count(-100)) + ' full_len=' + str(len(enc0['input_ids'])) + ' think-in-prompt=' + str('<think' in tok.apply_chat_template(rows[0]['messages'][:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)))
log('sample0 text[:400]: ' + tok.decode(enc0['input_ids'])[:400].replace(chr(10), ' '))

class DS(Dataset):
    def __len__(self):
        return len(rows)
    def __getitem__(self, i):
        return encode(rows[i])

def collate(batch):
    maxlen = max(len(b['input_ids']) for b in batch)
    pad_id = tok.pad_token_id
    input_ids, labels, att = [], [], []
    for b in batch:
        n = len(b['input_ids'])
        padn = maxlen - n
        input_ids.append(b['input_ids'] + [pad_id] * padn)
        labels.append(b['labels'] + [-100] * padn)
        att.append([1] * n + [0] * padn)
    return {'input_ids': torch.tensor(input_ids), 'labels': torch.tensor(labels), 'attention_mask': torch.tensor(att)}

model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-0.8B', dtype=torch.bfloat16, device_map='cuda')
model.config.use_cache = False
model.enable_input_require_grads()
lcfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',
                  target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
model = get_peft_model(model, lcfg)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=os.path.join(AUD, 'adapter_r4r3_local', 'ckpt'),
    max_steps=200,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    warmup_steps=10,
    logging_steps=10,
    save_steps=50,
    save_total_limit=5,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    optim='adamw_torch',
    report_to=[],
    seed=54,
)
t0 = time.time()
trainer = Trainer(model=model, args=args, train_dataset=DS(), data_collator=collate)
trainer.train()
dt = time.time() - t0
os.makedirs(FINAL, exist_ok=True)
model.save_pretrained(FINAL)
tok.save_pretrained(FINAL)
final_files = sorted(os.listdir(FINAL))
assert 'adapter_model.safetensors' in final_files, 'final adapter missing'
receipt = {'mode': 'local-blackwell-fallback', 'base': 'Qwen/Qwen3.5-0.8B', 'lora_r': 32,
           'steps': trainer.state.global_step, 'rows': 72, 'duration_s': round(dt, 1),
           'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none',
           'final_files': final_files, 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'log_history_tail': trainer.state.log_history[-3:]}
json.dump(receipt, open(os.path.join(AUD, 'r4r3_local_train_receipt.json'), 'w', encoding='utf-8'), indent=2, default=str)
log('LOCAL TRAIN COMPLETE steps=' + str(trainer.state.global_step) + ' in ' + str(round(dt/60, 1)) + ' min; final files: ' + str(final_files))

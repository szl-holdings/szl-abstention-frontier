import os, sys, io, json, time, re, subprocess
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import requests
from huggingface_hub import HfApi

AUD = os.path.join('C:' + chr(92), 'SZL', 'estate-audit')
FLOG = os.path.join(AUD, 'r4r3_fallback_log.txt')
FDONE = os.path.join(AUD, 'r4r3_fallback_done.txt')
WDONE = os.path.join(AUD, 'r4r3_watch_done.txt')
REPO_RUNS = 'SZLHOLDINGS/frontier-54-runs'
REPO_ADAPTER = 'SZLHOLDINGS/salad-chaski-r4-r3'
B = 'https://api.salad.com/api/public/organizations/alloyscape/projects/default/containers/'

def flog(s):
    line = '[' + time.strftime('%Y-%m-%d %H:%M:%S') + '] ' + str(s)
    print(line, flush=True)
    with open(FLOG, 'a', encoding='utf-8') as f:
        f.write(line + chr(10))

def salad_get():
    src = open(os.path.join('C:' + chr(92), 'SZL', 'edge-debug', 'salad_key.txt'), encoding='utf-8', errors='replace').read()
    key = re.search(r'[A-Za-z0-9_\-\.]{20,}', src.strip()).group(0)
    return requests.get(B + 'szl-train-chaski-r4-r3', headers={'Salad-Api-Key': key}, timeout=30), key

def salad_scalezero(key):
    h = {'Salad-Api-Key': key, 'Content-Type': 'application/merge-patch+json'}
    before = requests.get(B + 'szl-train-chaski-r4-r3', headers={'Salad-Api-Key': key}, timeout=30).json()
    p = requests.patch(B + 'szl-train-chaski-r4-r3', headers=h, json={'replicas': 0}, timeout=30)
    time.sleep(8)
    after = requests.get(B + 'szl-train-chaski-r4-r3', headers={'Salad-Api-Key': key}, timeout=30).json()
    rec = {'before_state': before.get('current_state'), 'patch_status': p.status_code, 'after_replicas': after.get('replicas')}
    json.dump(rec, open(os.path.join(AUD, 'r4r3_fallback_scalezero_receipt.json'), 'w', encoding='utf-8'), indent=2, default=str)
    flog('scalezero receipt: ' + json.dumps(rec)[:300])
    return rec

def final_present(api):
    try:
        files = api.list_repo_files(REPO_ADAPTER)
        return bool([f for f in files if f.endswith('adapter_model.safetensors') and 'final' in f])
    except Exception:
        return False

def build_harness(local_mode):
    src = open(os.path.join(AUD, 'eval_abst_v2.py'), encoding='utf-8', errors='replace').read()
    s = src.replace('abst_eval_r4r2_', 'abst_eval_r4r3_')
    s = s.replace('adapter_r4r2', 'adapter_r4r3_local' if local_mode else 'adapter_r4r3')
    s = s.replace('SZLHOLDINGS/salad-chaski-r4-r2', REPO_ADAPTER)
    stale = s.replace('r4-r3', '').replace('r4r3', '')
    assert 'r4-r2' not in stale and 'r4r2' not in stale, 'stale refs'
    out = os.path.join(AUD, 'eval_abst_v3_local.py' if local_mode else 'eval_abst_v3.py')
    open(out, 'w', encoding='utf-8').write(s)
    flog('harness written: ' + os.path.basename(out) + ' local_mode=' + str(local_mode))
    return out

def run_eval(harness_path, api):
    flog('running eval: ' + os.path.basename(harness_path))
    p = subprocess.run([sys.executable, harness_path], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=3600)
    flog('eval rc=' + str(p.returncode))
    summp = os.path.join(AUD, 'abst_eval_r4r3_summary.json')
    if p.returncode != 0 or not os.path.exists(summp):
        flog('EVAL FAILED rc=' + str(p.returncode) + ' tail: ' + (p.stdout or '')[-300:])
        return False
    flog('summary: ' + open(summp, encoding='utf-8', errors='replace').read()[:500])
    up = [('abst_eval_r4r3_summary.json', 'mesh/abst-eval-r4r3/abst_eval_r4r3_summary.json'),
          ('abst_eval_r4r3_receipts.jsonl', 'mesh/abst-eval-r4r3/abst_eval_r4r3_receipts.jsonl'),
          ('abst_eval_r4r3_log.txt', 'mesh/abst-eval-r4r3/abst_eval_r4r3_log.txt'),
          ('r4r3_local_train_receipt.json', 'mesh/abst-eval-r4r3/r4r3_local_train_receipt.json')]
    for local, remote in up:
        lp = os.path.join(AUD, local)
        if not os.path.exists(lp):
            continue
        for attempt in range(6):
            try:
                api.upload_file(path_or_fileobj=lp, path_in_repo=remote, repo_id=REPO_RUNS, repo_type='dataset')
                flog('uploaded ' + remote)
                break
            except Exception as ex:
                flog('upload err ' + repr(ex)[:100] + ' retry 75s')
                time.sleep(75)
    return True

api = HfApi()
flog('FALLBACK ORCHESTRATOR armed: Salad-independent path on local Blackwell; fires ONLY if Salad fails to deliver')
DEADLINE = time.time() + 12 * 3600
dead_streak = 0
mode = None
while time.time() < DEADLINE:
    if os.path.exists(WDONE):
        c = open(WDONE, encoding='utf-8', errors='replace').read()
        if 'COMPLETE' in c:
            flog('watcher COMPLETE - Salad delivered; fallback not needed, exiting')
            sys.exit(0)
        if 'TIMEOUT' in c:
            flog('watcher TIMEOUT - Salad did not deliver; going LOCAL TRAIN mode')
            mode = 'train'
        elif 'EVAL-FAILED' in c:
            flog('watcher EVAL-FAILED - adapter exists; retry eval once locally')
            mode = 'eval_retry'
        break
    try:
        g, key = salad_get()
        cs = g.json().get('current_state', {})
        counts = cs.get('instance_status_counts', {})
        alive = cs.get('status') in ('deploying', 'running') or any(counts.values())
        if final_present(api):
            flog('final adapter present - Salad delivering; watcher will eval; exiting')
            sys.exit(0)
        if alive:
            dead_streak = 0
            flog('salad alive: status=' + str(cs.get('status')) + ' counts=' + json.dumps(counts))
        else:
            dead_streak += 1
            flog('salad NOT alive (status=' + str(cs.get('status')) + ' counts all 0), streak=' + str(dead_streak))
            if dead_streak >= 3:
                flog('salad dead 30+ min with no final adapter - going LOCAL TRAIN mode')
                mode = 'train'
                break
    except Exception as ex:
        flog('poll err: ' + repr(ex)[:120])
    time.sleep(600)

if mode == 'eval_retry':
    ok = run_eval(build_harness(False), api)
    open(FDONE, 'w', encoding='utf-8').write('EVAL-RETRY ' + ('OK' if ok else 'FAILED') + ' ' + time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    sys.exit(0 if ok else 4)

if mode != 'train':
    open(FDONE, 'w', encoding='utf-8').write('DEADLINE-EXIT ' + time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    sys.exit(5)

g, key = salad_get()
salad_scalezero(key)
flog('starting LOCAL Blackwell training (200 steps, LoRA r=32, 72 pairs)')
p = subprocess.run([sys.executable, os.path.join(AUD, 'train_local_r4r3.py')], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=10 * 3600)
flog('train rc=' + str(p.returncode) + ' tail: ' + (p.stdout or '')[-400:])
finalp = os.path.join(AUD, 'adapter_r4r3_local', 'final', 'adapter_model.safetensors')
if p.returncode != 0 or not os.path.exists(finalp):
    flog('LOCAL TRAIN FAILED - recording honestly, no fabrication')
    open(FDONE, 'w', encoding='utf-8').write('LOCAL-TRAIN-FAILED rc=' + str(p.returncode) + ' ' + time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    sys.exit(6)
ok = run_eval(build_harness(True), api)
open(FDONE, 'w', encoding='utf-8').write('LOCAL-TRAIN-THEN-EVAL ' + ('OK' if ok else 'EVAL-FAILED') + ' ' + time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
flog('FALLBACK COMPLETE')

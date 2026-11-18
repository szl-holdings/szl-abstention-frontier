import os, sys, io, json, time, subprocess
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from huggingface_hub import HfApi

AUD = os.path.join('C:' + chr(92), 'SZL', 'estate-audit')
WLOG = os.path.join(AUD, 'r4r3_watch_log.txt')
DONE = os.path.join(AUD, 'r4r3_watch_done.txt')
REPO_RUNS = 'SZLHOLDINGS/frontier-54-runs'
REPO_ADAPTER = 'SZLHOLDINGS/salad-chaski-r4-r3'

def wlog(s):
    line = '[' + time.strftime('%Y-%m-%d %H:%M:%S') + '] ' + str(s)
    print(line, flush=True)
    with open(WLOG, 'a', encoding='utf-8') as f:
        f.write(line + chr(10))

api = HfApi()
DEADLINE = time.time() + 8 * 3600
found = False
wlog('watcher armed for ' + REPO_ADAPTER + ' final adapter; polls every 10 min, 8h cap')
while time.time() < DEADLINE:
    try:
        files = api.list_repo_files(REPO_ADAPTER)
        hit = [f for f in files if f.endswith('adapter_model.safetensors') and ('final' in f or f.count('/') == 0)]
        wlog('repo files=' + str(len(files)) + ' adapter-hit=' + json.dumps(hit))
        if hit:
            found = True
            break
    except Exception as ex:
        wlog('poll err: ' + repr(ex)[:150])
    time.sleep(600)

if not found:
    wlog('WATCH-TIMEOUT after 8h, no final adapter uploaded')
    open(DONE, 'w', encoding='utf-8').write('TIMEOUT ' + time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    sys.exit(2)

wlog('final adapter present; building parameterized harness v3 by textual substitution')
src = open(os.path.join(AUD, 'eval_abst_v2.py'), encoding='utf-8', errors='replace').read()
src2 = src.replace('abst_eval_r4r2_', 'abst_eval_r4r3_')
src2 = src2.replace('adapter_r4r2', 'adapter_r4r3')
src2 = src2.replace('SZLHOLDINGS/salad-chaski-r4-r2', REPO_ADAPTER)
stale = src2.replace('r4-r3', '').replace('r4r3', '')
assert 'r4-r2' not in stale and 'r4r2' not in stale, 'stale r4r2 refs remain'
assert src2 != src, 'no substitution happened'
open(os.path.join(AUD, 'eval_abst_v3.py'), 'w', encoding='utf-8').write(src2)
wlog('harness v3 written; running eval on local GPU')

p = subprocess.run([sys.executable, os.path.join(AUD, 'eval_abst_v3.py')], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=3600)
wlog('eval rc=' + str(p.returncode))
try:
    open(os.path.join(AUD, 'abst_eval_r4r3_proc.txt'), 'w', encoding='utf-8').write(((p.stdout or '') + (p.stderr or ''))[-4000:])
except Exception:
    pass
if p.returncode != 0 or not os.path.exists(os.path.join(AUD, 'abst_eval_r4r3_summary.json')):
    wlog('EVAL-FAILED rc=' + str(p.returncode))
    open(DONE, 'w', encoding='utf-8').write('EVAL-FAILED rc=' + str(p.returncode))
    sys.exit(3)
wlog('eval complete; summary: ' + open(os.path.join(AUD, 'abst_eval_r4r3_summary.json'), encoding='utf-8', errors='replace').read()[:600])

up = [('abst_eval_r4r3_summary.json', 'mesh/abst-eval-r4r3/abst_eval_r4r3_summary.json'),
      ('abst_eval_r4r3_receipts.jsonl', 'mesh/abst-eval-r4r3/abst_eval_r4r3_receipts.jsonl'),
      ('abst_eval_r4r3_log.txt', 'mesh/abst-eval-r4r3/abst_eval_r4r3_log.txt'),
      ('abst_eval_r4r3_proc.txt', 'mesh/abst-eval-r4r3/abst_eval_r4r3_proc.txt')]
for local, remote in up:
    lp = os.path.join(AUD, local)
    if not os.path.exists(lp):
        continue
    for attempt in range(6):
        try:
            api.upload_file(path_or_fileobj=lp, path_in_repo=remote, repo_id=REPO_RUNS, repo_type='dataset')
            wlog('uploaded ' + remote)
            break
        except Exception as ex:
            wlog('upload err ' + repr(ex)[:120] + ' attempt ' + str(attempt) + ' retry in 75s')
            time.sleep(75)

open(DONE, 'w', encoding='utf-8').write(json.dumps({'state': 'COMPLETE', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}))
wlog('WATCH-COMPLETE receipts at mesh/abst-eval-r4r3/')

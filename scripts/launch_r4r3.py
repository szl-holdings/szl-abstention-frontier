import sys, io, json, re, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import requests
BASE = 'https://api.salad.com/api/public/organizations/alloyscape/projects/default/containers'
SRC = 'szl-train-chaski-r4-r2'
NEW = 'szl-train-chaski-r4-r3'
src = open(r'C:\SZL\edge-debug\salad_key.txt', encoding='utf-8', errors='replace').read()
key = re.search(r'[A-Za-z0-9_\-\.]{20,}', src.strip()).group(0)
hdr = {'Salad-Api-Key': key}
g = requests.get(BASE + '/' + SRC, headers=hdr, timeout=30)
print('GET_SRC', g.status_code, flush=True)
if g.status_code != 200:
    sys.exit(1)
spec = g.json()
c = dict(spec['container'])
cmd = c.get('command')
def fix(x):
    if isinstance(x, str):
        x = x.replace('chaski-r4-r2-hb', 'chaski-r4-r3-hb')
        x = x.replace('szl-distill-abstention-v1-sft', 'szl-distill-abstention-v2-sft')
        x = x.replace('salad-chaski-r4-r2', 'salad-chaski-r4-r3')
        x = x.replace('szl-train-chaski-r4-r2', NEW)
    elif isinstance(x, list):
        x = [fix(i) for i in x]
    return x
c['command'] = fix(cmd)
s = json.dumps(c['command'])
assert 'chaski-r4-r3-hb' in s and 'szl-distill-abstention-v2-sft' in s and 'salad-chaski-r4-r3' in s and 'r4-r2' not in s.replace('r4-r3', '')
body = {
 'name': NEW,
 'display_name': 'SZL chaski r4-r3 abstention v2 LoRA SFT 72 pairs',
 'replicas': 1,
 'container': {
   'image': c['image'],
   'command': c['command'],
   'environment_variables': c['environment_variables'],
   'image_caching': c.get('image_caching', True),
   'resources': c['resources'],
 },
}
receipts = {'source': SRC, 'new': NEW, 'env_var_keys': sorted(c['environment_variables'].keys()), 'env_values_masked': True}
p = requests.post(BASE, headers={**hdr, 'Content-Type': 'application/json'}, json=body, timeout=60)
print('CREATE', p.status_code, flush=True)
receipts['create_status'] = p.status_code
receipts['create_body'] = p.text[:400]
if p.status_code not in (200, 201, 202):
    json.dump(receipts, open(r'C:\SZL\estate-audit\r4r3_launch_receipts.json', 'w', encoding='utf-8'), indent=2)
    print('CREATE-FAILED', p.text[:300], flush=True); sys.exit(1)
time.sleep(5)
v = requests.get(BASE + '/' + NEW, headers=hdr, timeout=30)
print('VERIFY', v.status_code, flush=True)
vj = v.json() if v.status_code == 200 else {}
receipts['verify_status'] = v.status_code
receipts['verify'] = {'name': vj.get('name'), 'replicas': vj.get('replicas'), 'status': vj.get('status'), 'current_state': vj.get('current_state')}
receipts['resources'] = vj.get('container', {}).get('resources')
json.dump(receipts, open(r'C:\SZL\estate-audit\r4r3_launch_receipts.json', 'w', encoding='utf-8'), indent=2)
print('LAUNCH-DONE verify=' + json.dumps(receipts.get('verify')), flush=True)
print('R4R3-LAUNCH-DONE')

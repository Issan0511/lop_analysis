"""Run fixed rescue jobs in isolated GPU processes after verified prefix replay."""
from pathlib import Path
import json,subprocess,time,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/elu_sunk_rescue_0913'
PREREG='74497a69a62f0b83e26eba6b27ecd672087523f5'
jobs=[('RL',20,2)]+[(e,t,l) for e in ['PM','RL'] for t in [10,20,50] for l in [1,2] if (e,t,l)!=('RL',20,2)]
state=dict(status='WAITING_PREFIX',jobs=jobs,completed=[],started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
def save(): (OUT/'execution_status.json').write_text(json.dumps(state,indent=2)+'\n')
save()
try:
 deadline=time.monotonic()+600
 while not (OUT/'prefix/provenance.json').exists():
  if (OUT/'prefix/failure.json').exists():raise RuntimeError('Prefix failed; see prefix/failure.json')
  if time.monotonic()>deadline:raise TimeoutError('prefix verification timeout')
  time.sleep(2)
 p=json.loads((OUT/'prefix/provenance.json').read_text())
 assert p['status']=='COMPLETE' and p['unit_maxabs']<=1e-6 and p['checkpoint_maxabs']<=1e-6
 for e,t,l in jobs:
  name=f'{e}_t{t}_l{l}';state.update(status='RUNNING',current_job=name);save()
  prov=OUT/name/'provenance.json'
  assert not prov.exists(),'Duplicate complete job; stop rather than silently rerun'
  cmd=[sys.executable,'-u','src/elu_sunk_rescue_0913.py','--run','--env',e,'--checkpoint',str(t),'--layer',str(l),'--prereg',PREREG]
  print('START',name,flush=True);st=time.monotonic()
  with (OUT/(name+'.log')).open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  v=json.loads(prov.read_text());assert v['status']=='COMPLETE'
  state['completed'].append(dict(job=name,seconds=time.monotonic()-st));save();print('DONE',name,flush=True)
 state.update(status='COMPLETE',finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save()
except Exception as exc:
 state.update(status='FAILED',error=repr(exc));save();raise


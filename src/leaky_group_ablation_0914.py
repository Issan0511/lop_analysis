"""Posthoc fixed-checkpoint group ablation; no retraining or future-plasticity claim."""
from pathlib import Path
import sys,torch,numpy as np,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.zero_attraction_mechanism_0913 import checkpoint,DATA,writecsv
BASE_OUT=ROOT/'results/unit_fates_0913/leaky_group_ablation_0914'
def measure(arm):
 OUT=BASE_OUT/arm
 OUT.mkdir(parents=True,exist_ok=True)
 paths=[DATA/'ckpts'/f'{arm}_step{s}.pt' for s in [0,200000,1000000,5000000]]
 cps={int(p.stem.split('step')[-1]):torch.load(p,map_location='cpu',weights_only=True) for p in paths}
 r0=cps[0]['net']['W'][:,:,15:].double().norm(dim=-1)
 rf=cps[5000000]['net']['W'][:,:,15:].double().norm(dim=-1)
 masks={'remove_grown':rf>=r0,'remove_shrunk':rf<r0}
 rows=[];checks={}
 for step in [200000,1000000,5000000]:
  cp=cps[step];n,x,y,act=checkpoint(cp)
  z=torch.einsum('rhd,prd->prh',n['W'],x)+n['b'];o=act.act_fn(z)*n['v'];pred=o.sum(-1)+n['c'];base=(pred-y).square().mean(0)
  for ri in range(10):
   with np.load(DATA/'logs'/f'{arm}_seed{ri}.npz') as log:
    k=int(np.flatnonzero(log['step']==step)[0]);target=float(log['eval_loss_exact'][k])
   err=abs(float(base[ri])-target);checks[f'baseline_{step}_{ri}']=err;assert err<3e-5
  for label,mask in masks.items():
   removed=(o*mask).sum(-1)
   direct=pred-removed
   kept=n['v']*(~mask)
   torch.testing.assert_close(direct,(act.act_fn(z)*kept).sum(-1)+n['c'],atol=1e-12,rtol=1e-12)
   compensated=direct+removed.mean(0)
   for ri in range(10):
    rows.append(dict(task=step//10000,seed=ri,intervention=label,n_removed=int(mask[ri].sum()),baseline_mse=float(base[ri]),removed_mse=float((direct[:,ri]-y[:,ri]).square().mean()),mean_preserved_mse=float((compensated[:,ri]-y[:,ri]).square().mean()),constant_optimal_mse=float(y[:,ri].var(unbiased=False))))
 writecsv(OUT/'seed_results.csv',rows)
 report=[]
 for task in [20,100,500]:
  for label in masks:
   rr=[r for r in rows if r['task']==task and r['intervention']==label]
   agg={k:float(np.mean([r[k] for r in rr])) for k in ['n_removed','baseline_mse','removed_mse','mean_preserved_mse','constant_optimal_mse']}
   report.append(dict(arm=arm,task=task,intervention=label,**agg))
 (OUT/'summary.json').write_text(json.dumps(report,indent=2))
 (OUT/'verification.json').write_text(json.dumps(dict(status='PASS',baseline_max_abs_error=max(checks.values()),ablation_matches_zero_v=True,source_files=[dict(path=str(p),sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()) for p in paths],code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
 return report
if __name__=='__main__':
 reports=[]
 for a in ['0p1','0p3','0p7']:
  for q in ['q0','qm05','qp05']:
   arm=f'LR_a{a}_{q}';reports.extend(measure(arm));print('PASS',arm,flush=True)
 (BASE_OUT/'summary.json').write_text(json.dumps(reports,indent=2))
 for r in reports:
  if r['task']==500:print(r)


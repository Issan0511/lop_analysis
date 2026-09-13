"""Source replay and extended paired transport for task21..40."""
from pathlib import Path
import argparse,json,time,subprocess,sys,concurrent.futures,hashlib,shutil
import numpy as np
import torch
from src import boundary_early_source_0908 as E
from src import boundary_tasks20_40_transport_0909 as T
from src.task20to100_0908 import exact
ROOT=E.ROOT;OUT=ROOT/'results/boundary_tasks20_40_0909'
def run(arm,iv,seed):
 torch.set_num_threads(1);E.H.setup('cpu');mnist=E.H.Mnist(torch.device('cpu'))
 E.OUT=OUT/'source';E.OUT.mkdir(parents=True,exist_ok=True);prefix=f'{arm}_{iv}_s{seed}'
 E.run(arm,seed,iv,mnist,tasks=40,first=20)
 cpnew=torch.load(E.OUT/(prefix+'_task1.pt'),weights_only=False,map_location='cpu')
 cpold=torch.load(ROOT/'results/boundary_groups_0908'/(prefix+'_task1.pt'),weights_only=False,map_location='cpu')
 assert exact(cpnew,cpold),'task1 checkpoint differs'
 path=E.OUT/(prefix+'_states.pt')
 saved=torch.load(path,weights_only=False,map_location='cpu');dense=np.load(E.OUT/(prefix+'.npz'))
 endref=np.load(ROOT/'results/task20to100_0908'/(prefix+'.npz'))
 px=mnist.test_x[saved['probe_indices']];mu=saved['mu_mean'];allrec={};rates=[];stats=[];checks=[];positions=[]
 start=time.monotonic();end_errors=[]
 for j,raw in enumerate(saved['boundaries']):
  assert raw['task']==21+j
  for step,key in [(0,'before'),(20,'after_20'),(300,'after_300'),(625,'after_625')]:
   st=raw[key]['state'];w,b=st['params'][:2]
   z=T.evaluate([w,b],px[:,raw['perm']]).detach().numpy()
   ss=w.double().sum(1).numpy()*mu+b.double().numpy()
   positions.append(dict(task=raw['task'],step=step,mean=float(z.mean()),star=float(ss.mean()),
      star_W=float((ss-b.double().numpy()).mean()),bias=float(b.double().mean()),
      within=float(z.var(0).mean()),between=float(z.mean(0).var()),
      pre_switch_mean=float(raw['end'].mean())))
   if step==625:
    k=int(np.flatnonzero(endref['task']==raw['task'])[0])
    err=max(float(abs(z.mean(0)-endref['zmean'][k]).max()),float(abs(ss-endref['star'][k]).max()),
            float(abs(z.var(0)-endref['within'][k]).max()))
    assert err<1e-10,err
    end_errors.append(err)
  rr,pp,rc,st,rate,ck,ep=T.boundary(raw,arm,iv,mnist,px,dense['dense1'][j],seed==0 and j==0,mu)
  history_error=float(abs(rr['actual_mean']-rr['history_mean']-rr['current_mean']-rr['current_l2_mean']).max())
  assert history_error<2e-6,history_error
  ck['current_history_addition']=history_error
  for pref,dd in [('',rr),('pair_',pp),('receiver_',rc),('',ep)]:
   for key,value in dd.items():T.append(allrec,pref+key,value)
  for rows,target in [(st,stats),(rate,rates)]:
   for row in rows:row.update(arm=arm,iv=iv,seed=seed,task=raw['task'])
   target.extend(rows)
  checks.append(dict(task=raw['task'],**ck))
  if j%5==4:print('TRANSPORT',prefix,j+1,round(time.monotonic()-start,1),flush=True)
  assert time.monotonic()-start<900
 T.OUT.mkdir(parents=True,exist_ok=True);dest=T.OUT/'raw'/(prefix+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(dest,task=np.arange(21,41),**{k:np.stack(v) for k,v in allrec.items()})
 E.csvwrite(T.OUT/(prefix+'_variance.csv'),stats);E.csvwrite(T.OUT/(prefix+'_paired_counts.csv'),rates)
 for row in positions:row.update(arm=arm,iv=iv,seed=seed)
 E.csvwrite(OUT/(prefix+'_positions.csv'),positions)
 meta=dict(arm=arm,iv=iv,seed=seed,task_range=[21,40],checks=checks,task1_exact=True,
    endpoint_task21_40_maxabs=max(end_errors),source_sha256=T.G.sha(path),
    raw_sha256=T.G.sha(dest),code_sha256=T.G.sha(Path(__file__)),transport_sha256=T.G.sha(Path(T.__file__)),
    source_generator_sha256=T.G.sha(Path(E.__file__)),spec_sha256=T.G.sha(ROOT/'specs/spec_boundary_tasks20_40_0909.md'),
    data_sha256=mnist.sha256,wall_seconds=time.monotonic()-start)
 (T.OUT/(prefix+'_provenance.json')).write_text(json.dumps(meta,indent=2))
 print('FINISHED',prefix,'endpoints maxabs',max(end_errors),flush=True)
def backup():
 target=Path('/home/issan/Projects/obsidian-research-data/boundary_tasks20_40_0909')
 target.mkdir(parents=True,exist_ok=True);rows=[]
 for f in sorted(list((OUT/'source').glob('*.pt'))+list((OUT/'source').glob('*.npz'))+list((OUT/'transport/raw').glob('*.npz'))):
  rel=f.relative_to(OUT);dst=target/rel;dst.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(f,dst);h=T.G.sha(f);assert T.G.sha(dst)==h
  rows.append(dict(path=str(rel),bytes=f.stat().st_size,sha256=h))
 (OUT/'backup_manifest.json').write_text(json.dumps(dict(target=str(target),files=rows,total_bytes=sum(r['bytes'] for r in rows)),indent=2))
 print('BACKUP',len(rows),sum(r['bytes'] for r in rows),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--arm');ap.add_argument('--iv');ap.add_argument('--seed',type=int);ap.add_argument('--all',action='store_true');a=ap.parse_args()
 if a.all:
  def job(j):subprocess.run([sys.executable,'-m','src.boundary_tasks20_40_0909','--arm',j[0],'--iv',j[1],'--seed',str(j[2])],check=True)
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   for _ in pool.map(job,[(arm,iv,s) for arm in ['LR','SNA'] for iv in ['none','l2'] for s in range(3)]):pass
  backup()
 else:run(a.arm,a.iv,a.seed)

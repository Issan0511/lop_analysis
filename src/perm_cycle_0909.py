"""Permutation-cycling counterfactual for the sqrt(t) width growth (spec_perm_cycle_0909)."""
from pathlib import Path
import argparse,json,time,subprocess,sys,concurrent.futures,shutil
import numpy as np
import torch
from src import boundary_gradient_0908 as G
from src.width_depth_intervention_0909 import refperms,measure
H=G.H;ROOT=G.ROOT
SRC=ROOT/'results/boundary_tasks20_40_0909/source'
OUT=ROOT/'results/perm_cycle_0909'
ARMS=[('LR','none'),('SNA','none'),('LR','l2')]
CONDS=[('FRESH',0),('CYCLE1',1),('CYCLE2',2),('CYCLE5',5)]
def centered(W):W=W.double();return W-W.mean(1,keepdim=True)
def continuation(saved,arm,iv,mnist,px,refs,mu,K,mutate=None):
 B=saved['boundaries'];p,act,adam=G.clone_state(B[0],arm)
 rows=[];incs=[];perm_used=[];exact=0.;ident=0.;ident_mut=float('inf')
 Wprev=centered(p[0]).clone()
 for j,raw in enumerate(B):
  perm=B[j%K]['perm'] if K>0 else raw['perm'];perm_used.append(perm)
  gd=torch.Generator();gd.set_state(raw['rng_after_perm']['data']);gb=torch.Generator();gb.set_state(raw['rng_after_perm']['batch'])
  idx=H.stratified_draw(mnist,gd);order=torch.randperm(H.TASK_EXAMPLES,generator=gb)
  xs=mnist.train_x[idx][:,perm][order];ys=mnist.train_y[idx][order]
  for step in range(1,626):
   xb=xs[(step-1)*16:step*16];yb=ys[(step-1)*16:step*16]
   out=H.forward(p,xb,act);loss=torch.nn.functional.cross_entropy(out[4],yb);g=torch.autograd.grad(loss,p)
   with torch.no_grad():
    if iv!='none':g=[gr+2*.001*q for q,gr in zip(p,g)]
    m_,v_,tc=adam;tc[0]+=1;c1=1-.9**tc[0];c2=1-.999**tc[0]
    for q,gr,mi,vi in zip(p,g,m_,v_):
     mi.mul_(.9).add_(gr,alpha=1-.9);vi.mul_(.999).addcmul_(gr,gr,value=1-.999);q-=.001*(mi/c1)/((vi/c2).sqrt()+1e-8)
    if arm=='SNA':act.update(out[0],out[2])
   if mutate=='replay' and step==1:p[0].data[0,0]+=1e-3
   if mutate=='measure':p[0].data.add_(1e-9)
  with torch.no_grad():
   Wc=centered(p[0]);D=Wc-Wprev;incs.append(D.clone())
   n2=(Wc*Wc).sum(1);n2p=(Wprev*Wprev).sum(1);dd=(D*D).sum(1);cr=2*(Wprev*D).sum(1)
   ident=max(ident,float((n2-n2p-dd-cr).abs().max()));ident_mut=min(ident_mut,float((n2-n2p-dd+cr).abs().max()))
   r,_=measure(p,act,px,perm,refs,mu,mnist)
   rows.append(dict(task=raw['task'],perm_task=int(B[j%K]['task']) if K>0 else int(raw['task']),
    cnorm2=float(n2.mean()),dnorm2=float(dd.mean()),align=float(cr.mean()),
    cos=float(((Wprev*D).sum(1)/(Wprev.norm(dim=1)*D.norm(dim=1))).mean()),**r))
   Wprev=Wc.clone()
  if K==0:
   ref=raw['after_625']['state']['params'];ra=raw['adam_after']
   exact=max(exact,max(float((a-b).abs().max()) for a,b in zip(p,ref)),max(float((a-b).abs().max()) for a,b in zip(adam[0]+adam[1],ra[0]+ra[1])))
   if arm=='SNA':exact=max(exact,max(float((a-b).abs().max()) for a,b in zip(act.V,raw['after_625']['state']['V'])))
 # pairwise cos among the 20 increments, unit-averaged
 n=len(incs);C=np.zeros((n,n))
 for a_ in range(n):
  for b_ in range(n):
   C[a_,b_]=float(((incs[a_]*incs[b_]).sum(1)/(incs[a_].norm(dim=1)*incs[b_].norm(dim=1))).mean())
 permtask=np.array([r['perm_task'] for r in rows])
 ck=dict(exact_replay=exact,identity=ident,identity_mutctl=ident_mut,
  perm_matches_saved=[bool(torch.equal(perm_used[j],B[j]['perm'])) for j in range(len(B))])
 return rows,C,permtask,ck
def run(arm,iv,seed):
 torch.set_num_threads(1);H.setup('cpu');mnist=H.Mnist(torch.device('cpu'));OUT.mkdir(parents=True,exist_ok=True)
 prefix=f'{arm}_{iv}_s{seed}';spath=SRC/(prefix+'_states.pt')
 manifest=json.loads((ROOT/'results/boundary_tasks20_40_0909/backup_manifest.json').read_text())
 assert G.sha(spath)=={r['path']:r['sha256'] for r in manifest['files']}[f'source/{prefix}_states.pt']
 saved=torch.load(spath,weights_only=False,map_location='cpu');px=mnist.test_x[saved['probe_indices']];mu=float(saved['mu_mean']);refs=refperms()
 allrows=[];checks={};mats={};start=time.monotonic()
 for name,K in CONDS:
  t0=time.monotonic();rows,C,pt,ck=continuation(saved,arm,iv,mnist,px,refs,mu,K)
  if K==0:
   assert ck['exact_replay']==0.,ck
   assert all(ck['perm_matches_saved']),ck
   if seed==0:
    ck['exact_replay_mutctl']=continuation(saved,arm,iv,mnist,px,refs,mu,0,mutate='replay')[3]['exact_replay']
    ck['measure_mutctl']=continuation(saved,arm,iv,mnist,px,refs,mu,0,mutate='measure')[3]['exact_replay']
    assert ck['exact_replay_mutctl']>1e-4 and ck['measure_mutctl']>0,ck
  else:
   assert ck['perm_matches_saved'][0] and not ck['perm_matches_saved'][K] ,('cycle perm check',name,ck['perm_matches_saved'])
  assert ck['identity']<1e-10 and ck['identity_mutctl']>1e-3,ck
  for r in rows:r.update(arm=arm,iv=iv,seed=seed,cond=name)
  allrows+=rows;ck['wall']=time.monotonic()-t0;checks[name]=ck;mats[name+'_cos']=C;mats[name+'_permtask']=pt
  assert time.monotonic()-t0<120,'runtime cap'
  print('DONE',prefix,name,round(time.monotonic()-t0,1),'s',flush=True)
 keys=[];[keys.append(k) for r in allrows for k in r if k not in keys]
 G.B.csvwrite(OUT/(prefix+'_rows.csv'),[{k:r.get(k) for k in keys} for r in allrows])
 np.savez_compressed(OUT/(prefix+'_cos.npz'),**mats)
 (OUT/(prefix+'_provenance.json')).write_text(json.dumps(dict(arm=arm,iv=iv,seed=seed,checks=checks,conds=CONDS,
  source_sha256=G.sha(spath),code_sha256=G.sha(Path(__file__)),spec_sha256=G.sha(ROOT/'specs/spec_perm_cycle_0909.md'),
  host_sha256=G.sha(Path(H.__file__)),data_sha256=mnist.sha256,wall_seconds=time.monotonic()-start,
  scope='CPU continuation from the saved task-20 state with cycled permutations; not the original CUDA run'),indent=2))
 print('FINISHED',prefix,round(time.monotonic()-start,1),'s',flush=True)
def backup():
 target=Path('/home/issan/Projects/obsidian-research-data/perm_cycle_0909');target.mkdir(parents=True,exist_ok=True);rows=[]
 for f in sorted(OUT.glob('*_cos.npz')):
  dst=target/f.name;shutil.copy2(f,dst);h=G.sha(f);assert G.sha(dst)==h;rows.append(dict(path=f.name,bytes=f.stat().st_size,sha256=h))
 (OUT/'backup_manifest.json').write_text(json.dumps(dict(target=str(target),files=rows,total_bytes=sum(r['bytes'] for r in rows)),indent=2))
 print('BACKUP',len(rows),sum(r['bytes'] for r in rows),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--arm');ap.add_argument('--iv');ap.add_argument('--seed',type=int);ap.add_argument('--all',action='store_true');a=ap.parse_args()
 if a.all:
  def job(j):subprocess.run([sys.executable,'-m','src.perm_cycle_0909','--arm',j[0],'--iv',j[1],'--seed',str(j[2])],check=True)
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   for _ in pool.map(job,[(arm,iv,s) for arm,iv in ARMS for s in range(3)]):pass
  backup()
 else:run(a.arm,a.iv,a.seed)

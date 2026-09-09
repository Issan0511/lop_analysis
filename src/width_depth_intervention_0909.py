"""Width/depth intervention at the task-20 state (spec_width_depth_intervention_0909).

Rescales the centered part of W1 (row mean and bias fixed) or shifts the bias
(W1 fixed) at the saved task-20 state, then continues tasks 21..40 on the saved
stream.  kappa=1.0 is a literal no-op and must reproduce the saved after_625
states bit-exactly.
"""
from pathlib import Path
import argparse,json,time,subprocess,sys,concurrent.futures,hashlib,shutil
import numpy as np
import torch
from src import boundary_gradient_0908 as G
H=G.H;ROOT=G.ROOT
SRC=ROOT/'results/boundary_tasks20_40_0909/source'
OUT=ROOT/'results/width_depth_intervention_0909'
MEAS=[20,100,300,625];NREF=8;REF_SEED=20260909
ARMS=[(a,i) for a in ['LR','SNA'] for i in ['none','l2']]
INTERVENTIONS=[('k0.7','width',0.7),('k1.0','width',1.0),('k1.4','width',1.4),('bplus','bias',.3),('bminus','bias',-.3)]
def refperms():
 g=torch.Generator().manual_seed(REF_SEED);return [torch.randperm(784,generator=g) for _ in range(NREF)]
def decompose(W,x):
 """Per-unit variance of x@W.T split as offset (row mean) + cross + width (centered W)."""
 xc=x-x.mean(0);S=xc.T@xc/x.shape[0];m=W.mean(1,keepdim=True);Wt=W-m;one=torch.ones(W.shape[1],1,dtype=torch.float64)
 s11=float(one.T@S@one);off=(m**2).squeeze()*s11;cross=2*m.squeeze()*(Wt@S@one).squeeze();wid=((Wt@S)*Wt).sum(1)
 return off,cross,wid
def measure(p,act,px,perm,refs,mu,mnist=None):
 with torch.no_grad():
  W=p[0].double();b=p[1].double();x=px.double()
  z=x[:,perm]@W.T+b;zm=z.mean(0);zv=z.var(0,unbiased=False)
  zr=torch.stack([x[:,r]@W.T+b for r in refs])           # (NREF,512,100)
  zim=zr.mean(1).mean(0);ziv=zr.var(1,unbiased=False).mean(0)
  m=W.mean(1);Wt=W-m[:,None]
  parts=[decompose(W,x[:,r]) for r in refs]
  off=torch.stack([q[0] for q in parts]).mean(0);wid=torch.stack([q[2] for q in parts]).mean(0)
  row=dict(zbar_cur=float(zm.mean()),sigma_cur=float(zv.mean().sqrt()),between_cur=float(zm.var(unbiased=False)),
   zbar_inv=float(zim.mean()),sigma_inv=float(ziv.mean().sqrt()),between_inv=float(zim.var(unbiased=False)),
   rowmean=float(m.mean()),cnorm=float(Wt.norm(dim=1).mean()),bias=float(b.mean()),
   star=float((W.sum(1)*mu+b).mean()),var_off=float(off.mean()),var_wid=float(wid.mean()),
   pos_frac=float((z>0).double().mean()))
  if mnist is not None:
   row['acc']=float((H.forward(p,mnist.test_x[:,perm],act)[4].argmax(1)==mnist.test_y).float().mean())
  if isinstance(act,H.AdaptiveSnake):row['alpha_med']=float(act.alpha(0).median())
  units=dict(zbar_i=zim.numpy().copy(),sd_i=ziv.sqrt().numpy().copy(),rowmean_i=m.numpy().copy(),cnorm_i=Wt.norm(dim=1).numpy().copy())
 return row,units
def intervene(p,kind,val,sigma20):
 """Returns the bias shift actually applied (0 for width arms)."""
 with torch.no_grad():
  if kind=='width':
   if val==1.0:return 0.
   W=p[0].double();m=W.mean(1,keepdim=True);p[0].copy_((m+val*(W-m)).float());return 0.
  db=val*sigma20;p[1].add_(db);return db
def continuation(saved,arm,iv,mnist,px,refs,mu,kind,val,mutate=None,record=True):
 B=saved['boundaries'];raw0=B[0]
 p,act,adam=G.clone_state(raw0,arm)
 rows=[];units=[];ck={}
 r0,u0=measure(p,act,px,raw0['perm'],refs,mu)
 sigma20=r0['sigma_inv']
 W_old=p[0].double().clone();b_old=p[1].double().clone()
 db=intervene(p,kind,val,sigma20)
 r1,u1=measure(p,act,px,raw0['perm'],refs,mu)
 if mutate=='replay':p[0].data[0,0]+=1e-3
 # ---- intervention identities ----
 W=p[0].double();m_old=W_old.mean(1,keepdim=True);Wt_old=W_old-m_old;m=W.mean(1,keepdim=True);Wt=W-m
 if kind=='width' and val!=1.0:
  x=px.double()
  dec=[decompose(W_old,x[:,r]) for r in refs]
  pred=np.mean([float((o+val*c+val**2*w).mean()) for o,c,w in dec])
  bad=np.mean([float((o+(1/val)*c+(1/val)**2*w).mean()) for o,c,w in dec])
  sig2=r1['sigma_inv']**2
  ck.update(sigma2_identity=abs(sig2-pred)/pred,sigma2_identity_mutctl=abs(sig2-bad)/pred,
   rowsum_rel=float(((W.sum(1)-W_old.sum(1)).abs()/W_old.abs().sum(1)).max()),
   bias_diff=float((p[1].double()-b_old).abs().max()),
   centered_scale=float((Wt-val*Wt_old).abs().max()),centered_moved=float((Wt-Wt_old).abs().max()))
 if kind=='bias':
  ck.update(bias_shift_z=max(abs(r1['zbar_inv']-r0['zbar_inv']-db),abs(r1['zbar_cur']-r0['zbar_cur']-db)),
   bias_shift_mutctl=abs(r1['zbar_inv']-r0['zbar_inv']-.99*db),
   sigma_unchanged=max(abs(r1['sigma_inv']-r0['sigma_inv']),abs(r1['sigma_cur']-r0['sigma_cur'])),
   W_unchanged=float((p[0].double()-W_old).abs().max()),db=db)
 rows.append(dict(task=20,step=0,phase='pre',**r0));rows.append(dict(task=20,step=0,phase='post',**r1));units.append((20,u1))
 exact=0.
 for j,raw in enumerate(B):
  gd=torch.Generator();gd.set_state(raw['rng_after_perm']['data']);gb=torch.Generator();gb.set_state(raw['rng_after_perm']['batch'])
  idx=H.stratified_draw(mnist,gd);order=torch.randperm(H.TASK_EXAMPLES,generator=gb)
  xs=mnist.train_x[idx][:,raw['perm']][order];ys=mnist.train_y[idx][order]
  for step in range(1,626):
   xb=xs[(step-1)*16:step*16];yb=ys[(step-1)*16:step*16]
   out=H.forward(p,xb,act);loss=torch.nn.functional.cross_entropy(out[4],yb);g=torch.autograd.grad(loss,p)
   with torch.no_grad():
    if iv!='none':g=[gr+2*.001*q for q,gr in zip(p,g)]
    m_,v_,tc=adam;tc[0]+=1;c1=1-.9**tc[0];c2=1-.999**tc[0]
    for q,gr,mi,vi in zip(p,g,m_,v_):
     mi.mul_(.9).add_(gr,alpha=1-.9);vi.mul_(.999).addcmul_(gr,gr,value=1-.999)
     q-=.001*(mi/c1)/((vi/c2).sqrt()+1e-8)
    if arm=='SNA':act.update(out[0],out[2])
   if step in MEAS and record:
    r,u=measure(p,act,px,raw['perm'],refs,mu,mnist if step==625 else None)
    rows.append(dict(task=raw['task'],step=step,phase='train',**r))
    if step==625:units.append((raw['task'],u))
    if mutate=='measure':
     if arm=='SNA':act.update(out[0],out[2])
     else:p[0].data.add_(1e-9)
  if kind=='width' and val==1.0:
   ref=raw['after_625']['state']['params'];ra=raw['adam_after']
   exact=max(exact,max(float((a-b).abs().max()) for a,b in zip(p,ref)),
    max(float((a-b).abs().max()) for a,b in zip(adam[0]+adam[1],ra[0]+ra[1])),float(abs(adam[2][0]-ra[2])))
   if arm=='SNA':exact=max(exact,max(float((a-b).abs().max()) for a,b in zip(act.V,raw['after_625']['state']['V'])))
 ck['exact_replay']=exact
 return rows,units,ck
def run(arm,iv,seed):
 torch.set_num_threads(1);H.setup('cpu');mnist=H.Mnist(torch.device('cpu'));OUT.mkdir(parents=True,exist_ok=True)
 prefix=f'{arm}_{iv}_s{seed}';spath=SRC/(prefix+'_states.pt')
 manifest=json.loads((ROOT/'results/boundary_tasks20_40_0909/backup_manifest.json').read_text())
 assert G.sha(spath)=={r['path']:r['sha256'] for r in manifest['files']}[f'source/{prefix}_states.pt']
 saved=torch.load(spath,weights_only=False,map_location='cpu');px=mnist.test_x[saved['probe_indices']];mu=float(saved['mu_mean']);refs=refperms()
 allrows=[];checks={};start=time.monotonic();unitstore={}
 for name,kind,val in INTERVENTIONS:
  t0=time.monotonic()
  rows,units,ck=continuation(saved,arm,iv,mnist,px,refs,mu,kind,val)
  if name=='k1.0':
   assert ck['exact_replay']==0.,('reference continuation is not exact',ck)
   if seed==0:
    ck['exact_replay_mutctl']=continuation(saved,arm,iv,mnist,px,refs,mu,kind,val,mutate='replay',record=False)[2]['exact_replay']
    ck['measure_mutctl']=continuation(saved,arm,iv,mnist,px,refs,mu,kind,val,mutate='measure')[2]['exact_replay']
    assert ck['exact_replay_mutctl']>1e-4 and ck['measure_mutctl']>0,ck
  if kind=='width' and val!=1.0:
   assert ck['sigma2_identity']<1e-5 and ck['sigma2_identity_mutctl']>1e-2,ck
   assert ck['rowsum_rel']<1e-6 and ck['bias_diff']==0. and ck['centered_scale']<1e-6 and ck['centered_moved']>1e-3,ck
  if kind=='bias':
   assert ck['bias_shift_z']<1e-6 and ck['bias_shift_mutctl']>1e-4 and ck['sigma_unchanged']<=1e-12 and ck['W_unchanged']==0.,ck
  for r in rows:r.update(arm=arm,iv=iv,seed=seed,intervention=name)
  allrows+=rows;checks[name]=ck;checks[name]['wall']=time.monotonic()-t0
  for t,u in units:
   for k,v in u.items():unitstore[f'{name}_{k}_t{t}']=v
  assert time.monotonic()-t0<120,'runtime cap'
  print('DONE',prefix,name,round(time.monotonic()-t0,1),'s',flush=True)
 keys=[];[keys.append(k) for r in allrows for k in r if k not in keys]
 G.B.csvwrite(OUT/(prefix+'_rows.csv'),[{k:r.get(k) for k in keys} for r in allrows])
 np.savez_compressed(OUT/(prefix+'_units.npz'),**unitstore)
 (OUT/(prefix+'_provenance.json')).write_text(json.dumps(dict(arm=arm,iv=iv,seed=seed,checks=checks,
  interventions=INTERVENTIONS,meas=MEAS,nref=NREF,ref_seed=REF_SEED,source_sha256=G.sha(spath),
  code_sha256=G.sha(Path(__file__)),spec_sha256=G.sha(ROOT/'specs/spec_width_depth_intervention_0909.md'),
  host_sha256=G.sha(Path(H.__file__)),data_sha256=mnist.sha256,wall_seconds=time.monotonic()-start,
  scope='CPU continuation from the saved task-20 state; not the original CUDA run'),indent=2))
 print('FINISHED',prefix,round(time.monotonic()-start,1),'s',flush=True)
def backup():
 target=Path('/home/issan/Projects/obsidian-research-data/width_depth_intervention_0909');target.mkdir(parents=True,exist_ok=True);rows=[]
 for f in sorted(OUT.glob('*_units.npz')):
  dst=target/f.name;shutil.copy2(f,dst);h=G.sha(f);assert G.sha(dst)==h;rows.append(dict(path=f.name,bytes=f.stat().st_size,sha256=h))
 (OUT/'backup_manifest.json').write_text(json.dumps(dict(target=str(target),files=rows,total_bytes=sum(r['bytes'] for r in rows)),indent=2))
 print('BACKUP',len(rows),sum(r['bytes'] for r in rows),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--arm');ap.add_argument('--iv');ap.add_argument('--seed',type=int);ap.add_argument('--all',action='store_true');a=ap.parse_args()
 if a.all:
  def job(j):subprocess.run([sys.executable,'-m','src.width_depth_intervention_0909','--arm',j[0],'--iv',j[1],'--seed',str(j[2])],check=True)
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   for _ in pool.map(job,[(arm,iv,s) for arm,iv in ARMS for s in range(3)]):pass
  backup()
 else:run(a.arm,a.iv,a.seed)

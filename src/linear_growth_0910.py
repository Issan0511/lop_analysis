"""Width growth and sinking without a gate (spec_linear_growth_0910).

Three arms, seeds 0..2, 120 tasks of permuted MNIST, no regularisation:
  LIN   784-100-100-10 with the identity activation (H.ARMS['LIN']).  Same init
        and RNG streams as the other 3-layer arms (S-init).
  LIN0  784-10 single affine map (H.ARMS['LIN0'], init_params(seed,dev,(784,10))).
        H.forward branches on len(params)==2 and returns (z,z,z,z,z): "layer 1"
        is the 10x784 output weight, there are 10 units, the preactivation is the
        logit, and no layer downstream can absorb the scale.
  LR    leaky a=0.1, the G1 reference: it must reproduce the 42 committed
        checkpoints with maxabs 0.
The host module is never edited; activations come from H.ARMS.
"""
from pathlib import Path
import argparse,json,time,subprocess,sys,concurrent.futures,shutil
import numpy as np
import torch
from src import boundary_gradient_0908 as G
H=G.H;ROOT=G.ROOT
OUT=ROOT/'results/linear_growth_0910'
NTASK=120;NREF=8;REF_SEED=20260909;NGRAD=4096
TRACK=set(range(20,121,10))
E_TASKS=[20,40,60,80,100,120]
ARMS=['LIN','LIN0','LR']
def make(arm,seed):
 """(params, activation) for an arm.  LIN0 draws its own init: the dims differ."""
 dev=torch.device('cpu')
 if arm=='LIN0':return H.init_params(seed,dev,(784,10)),H.ARMS['LIN0']
 return H.init_params(seed,dev),H.ARMS[arm]
def refperms():
 g=torch.Generator().manual_seed(REF_SEED);return [torch.randperm(784,generator=g) for _ in range(NREF)]
def hdef(act,z):
 """h(z)=z phi'(z)-phi(z); zero iff phi is positively homogeneous.  Computed from the
 activation itself rather than asserted to be zero."""
 return z*act.dphi(z)-act.phi(z)
def h_elu(z):
 """h of ELU(alpha=1).  Mutation control: wrong for the identity and for leaky."""
 return torch.where(z>0,torch.zeros_like(z),torch.exp(z.clamp(max=0.))*(z-1)+1)
def train(arm,seed,mnist,px,refs,xg,yg,mutate=None,measure_on=True):
 par,act=make(arm,seed)
 if mutate=='init':par[0].data[0,0]+=1e-3
 nu=par[0].shape[0]
 gp,gd,gb=H.stream('perm',seed),H.stream('data',seed),H.stream('batch',seed)
 adam=([torch.zeros_like(q) for q in par],[torch.zeros_like(q) for q in par],[0])
 C=lambda W:(W.double()-W.double().mean(1,keepdim=True))
 prev=C(par[0]).clone();rows=[];units={};ends={}
 ck=dict(ident=0.,ident_mutctl=0.,decomp=0.,decomp_mutctl=float('inf'),hzero=0.)
 for task in range(1,NTASK+1):
  perm=torch.randperm(784,generator=gp)
  di=H.stratified_draw(mnist,gd);order=torch.randperm(H.TASK_EXAMPLES,generator=gb)
  xs=mnist.train_x[di][:,perm][order];ys=mnist.train_y[di][order]
  track=measure_on and task in TRACK
  if track:sumsq=torch.zeros(nu,dtype=torch.float64);sp=C(par[0]).clone()
  for step in range(625):
   xb=xs[step*16:(step+1)*16];yb=ys[step*16:(step+1)*16]
   o=H.forward(par,xb,act);loss=torch.nn.functional.cross_entropy(o[4],yb)
   g=torch.autograd.grad(loss,par)
   with torch.no_grad():
    m_,v_,tc=adam;tc[0]+=1;c1=1-.9**tc[0];c2=1-.999**tc[0]
    for q,gr,mi,vi in zip(par,g,m_,v_):
     mi.mul_(.9).add_(gr,alpha=1-.9);vi.mul_(.999).addcmul_(gr,gr,value=1-.999)
     q-=.001*(mi/c1)/((vi/c2).sqrt()+1e-8)
    if track:cur=C(par[0]);d=cur-sp;sumsq+=(d*d).sum(1);sp=cur.clone()
  if mutate=='measure':
   with torch.no_grad():par[0].data.add_(1e-9)
  ends[task]=[q.detach().clone() for q in par]
  if not measure_on:
   with torch.no_grad():prev=C(par[0]).clone()
   continue
  with torch.no_grad():
   cur=C(par[0]);D=cur-prev
   n2=(cur*cur).sum(1);n2p=(prev*prev).sum(1);dd=(D*D).sum(1);cr=2*(prev*D).sum(1)
   ck['decomp']=max(ck['decomp'],float((n2-n2p-dd-cr).abs().max()))
   ck['decomp_mutctl']=min(ck['decomp_mutctl'],float((n2-n2p-dd+cr).abs().max()))
   row=dict(task=task,N=float(cur.norm(dim=1).mean()),N2=float(n2.mean()),D2=float(dd.mean()),
    erode=float(cr.mean()),cos=float(((prev*D).sum(1)/(prev.norm(dim=1)*D.norm(dim=1))).mean()),
    rowmean=float(par[0].double().mean(1).mean()),bias=float(par[1].double().mean()))
   if track:
    W=par[0].double();b=par[1].double();x=px.double()
    z=x[:,perm]@W.T+b
    zr=torch.stack([x[:,r]@W.T+b for r in refs])
    zim=zr.mean(1).mean(0);ziv=zr.var(1,unbiased=False).mean(0)
    row.update(steps2=float(sumsq.mean()),persist=float(dd.mean()/sumsq.mean()),
     zbar_cur=float(z.mean()),sigma_cur=float(z.var(0,unbiased=False).mean().sqrt()),
     zbar_inv=float(zim.mean()),sigma_inv=float(ziv.mean().sqrt()),
     acc=float((H.forward(par,mnist.test_x[:,perm],act)[4].argmax(1)==mnist.test_y).float().mean()))
    for k,v in dict(cnorm_i=cur.norm(dim=1),zbar_i=zim,sd_i=ziv.sqrt()).items():
     units.setdefault(k,[]).append(v.numpy().copy())
   prev=cur.clone()
  if task in E_TASKS:
   q=[t.detach().clone().requires_grad_(True) for t in par]
   xgp=xg[:,perm]
   o=H.forward(q,xgp,act);loss=torch.nn.functional.cross_entropy(o[4],yg)
   if arm=='LIN0':
    # Euler identity of the single affine map: <dW,W>+<db,b> = sum_sk dL/dlogit_sk * logit_sk
    gg=torch.autograd.grad(loss,q+[o[4]])
    W,b=q[0].detach().double(),q[1].detach().double();dl=gg[2].double();zl=o[4].detach().double()
    r1=float((gg[0].double()*W).sum()+(gg[1].double()*b).sum());r2=0.
    sc=float(gg[0].double().norm()*W.norm()+gg[1].double().norm()*b.norm())
    Ep=float((dl*zl).sum())
    Ew=float((dl[:,:10]*xgp[:,:10].double()).sum())   # control: the input instead of the logit
   else:
    gg=torch.autograd.grad(loss,q+[o[1]])
    W1,b1,W2=q[0].detach().double(),q[1].detach().double(),q[2].detach().double()
    r1=float((gg[0].double()*W1).sum()+(gg[1].double()*b1).sum());r2=float((gg[2].double()*W2).sum())
    sc=float(gg[0].double().norm()*W1.norm()+gg[1].double().norm()*b1.norm()+gg[2].double().norm()*W2.norm())
    z1=o[0].detach().double();dl=gg[6].double();hh=hdef(act,z1)
    ck['hzero']=max(ck['hzero'],float(hh.abs().max()))
    Ep=float((dl*hh).sum())
    Ew=float((dl*h_elu(z1)).sum())                    # control: ELU's h on a linear gate
   E=r1-r2
   ck['ident']=max(ck['ident'],abs(E-Ep)/sc);ck['ident_mutctl']=max(ck['ident_mutctl'],abs(E-Ew)/sc)
   row.update(E=E,E_pred=Ep,E_pred_wrong=Ew,r_1=r1,r_2=r2,r_scale=sc)
  rows.append(row)
 return rows,{k:np.stack(v) for k,v in units.items()},ends,ck
def g1_check(seed,ends):
 """LR must reproduce the committed checkpoints exactly (42 points)."""
 R=ROOT/'results';worst=0.;n=0
 for tag,t in [('task1',1),('task100',100)]:
  cp=torch.load(R/'boundary_groups_0908'/f'LR_none_s{seed}_{tag}.pt',weights_only=False,map_location='cpu')
  worst=max(worst,max(float((a-b).abs().max()) for a,b in zip(ends[t],cp['state']['params'])));n+=1
 for src in ['boundary_tasks20_40_0909/source','boundary_groups_0908']:
  sv=torch.load(R/src/f'LR_none_s{seed}_states.pt',weights_only=False,map_location='cpu')
  for raw in sv['boundaries']:
   t=int(raw['task'])
   if 'after_625' in raw and t in ends:
    worst=max(worst,max(float((a-b).abs().max()) for a,b in zip(ends[t],raw['after_625']['state']['params'])));n+=1
 return worst,n
def fwd_check(seed,mnist):
 """LIN0 must go through the len(params)==2 branch of H.forward."""
 par,act=make('LIN0',seed)
 g=torch.Generator().manual_seed(REF_SEED)
 x=mnist.train_x[torch.randperm(len(mnist.train_x),generator=g)[:64]]
 with torch.no_grad():
  o=H.forward(par,x,act);ref=x@par[0].T+par[1]
  return dict(lin0_nparam=len(par),lin0_units=int(par[0].shape[0]),
   lin0_fwd=float((o[4]-ref).abs().max()),
   lin0_fwd_mutctl=float((o[4]-x@par[0].T).abs().max()),   # control: drop the bias
   lin0_fwd_alias=int(o[0] is o[4]))
def run(arm,seed):
 torch.set_num_threads(1);H.setup('cpu');mnist=H.Mnist(torch.device('cpu'));OUT.mkdir(parents=True,exist_ok=True)
 tag=f'{arm}_s{seed}';t0=time.monotonic();refs=refperms()
 px=mnist.test_x[torch.randperm(len(mnist.test_x),generator=H.stream('boundary_probe',seed))[:512]]
 gi=torch.Generator().manual_seed(REF_SEED+seed);gidx=torch.randperm(len(mnist.train_x),generator=gi)[:NGRAD]
 xg=mnist.train_x[gidx];yg=mnist.train_y[gidx]
 rows,units,ends,ck=train(arm,seed,mnist,px,refs,xg,yg)
 # ---- checks (spec 0910 §5); every one carries an asserted mutation control ----
 ck.update(fwd_check(seed,mnist))
 assert ck['lin0_nparam']==2 and ck['lin0_units']==10 and ck['lin0_fwd_alias']==1,ck
 assert ck['lin0_fwd']<1e-12 and ck['lin0_fwd_mutctl']>1e-6,ck
 assert ck['ident']<1e-6,ck
 assert ck['ident_mutctl']>(1e-2 if arm=='LIN0' else 1e-4),ck
 if arm!='LIN0':assert ck['hzero']<1e-12,ck
 assert ck['decomp']<1e-10 and ck['decomp_mutctl']>1e-3,ck
 if arm=='LR':
  ck['g1_maxabs'],ck['g1_n']=g1_check(seed,ends)
  assert ck['g1_maxabs']==0. and ck['g1_n']==42,('G1 failed',ck)
  if seed==0:
   e2=train(arm,seed,mnist,px,refs,xg,yg,mutate='init',measure_on=False)[2]
   ck['g1_mutctl']=max(float((a-b).abs().max()) for a,b in zip(ends[NTASK],e2[NTASK]))
   assert ck['g1_mutctl']>1e-4,ck
 if seed==0:
  e3=train(arm,seed,mnist,px,refs,xg,yg,measure_on=False)[2]
  ck['meas_noninv']=max(float((a-b).abs().max()) for a,b in zip(ends[NTASK],e3[NTASK]))
  e4=train(arm,seed,mnist,px,refs,xg,yg,mutate='measure',measure_on=False)[2]
  ck['meas_mutctl']=max(float((a-b).abs().max()) for a,b in zip(e3[NTASK],e4[NTASK]))
  assert ck['meas_noninv']==0. and ck['meas_mutctl']>0,ck
 for r in rows:r.update(arm=arm,seed=seed)
 keys=[];[keys.append(k) for r in rows for k in r if k not in keys]
 G.B.csvwrite(OUT/(tag+'_rows.csv'),[{k:r.get(k) for k in keys} for r in rows])
 np.savez_compressed(OUT/(tag+'_units.npz'),tasks=np.array(sorted(TRACK)),**units)
 (OUT/(tag+'_provenance.json')).write_text(json.dumps(dict(arm=arm,seed=seed,checks=ck,
  ntask=NTASK,nref=NREF,ref_seed=REF_SEED,ngrad=NGRAD,track=sorted(TRACK),e_tasks=E_TASKS,
  code_sha256=G.sha(Path(__file__)),spec_sha256=G.sha(ROOT/'specs/spec_linear_growth_0910.md'),
  host_sha256=G.sha(Path(H.__file__)),data_sha256=mnist.sha256,wall_seconds=time.monotonic()-t0,
  scope='fresh CPU training; LIN/LR share the 3-layer S-init, LIN0 draws its own'),indent=2))
 print('FINISHED',tag,round(time.monotonic()-t0,1),'s',ck,flush=True)
def backup():
 tg=Path('/home/issan/Projects/obsidian-research-data/linear_growth_0910');tg.mkdir(parents=True,exist_ok=True);rs=[]
 for f in sorted(OUT.glob('*_units.npz')):
  shutil.copy2(f,tg/f.name);h=G.sha(f);assert G.sha(tg/f.name)==h;rs.append(dict(path=f.name,bytes=f.stat().st_size,sha256=h))
 (OUT/'backup_manifest.json').write_text(json.dumps(dict(target=str(tg),files=rs,
  total_bytes=sum(r['bytes'] for r in rs)),indent=2));print('BACKUP',len(rs),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--arm');ap.add_argument('--seed',type=int)
 ap.add_argument('--all',action='store_true');a=ap.parse_args()
 if a.all:
  def job(j):subprocess.run([sys.executable,'-m','src.linear_growth_0910','--arm',j[0],'--seed',str(j[1])],check=True)
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   for _ in pool.map(job,[(n,s) for n in ARMS for s in range(3)]):pass
  backup()
 else:run(a.arm,a.seed)

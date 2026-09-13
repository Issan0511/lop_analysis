"""Gate scale-invariance vs the width-growth exponent (spec_gate_scale_invariance_0909).

Group 1 (phi''==0, gate scale-invariant): ReLU, leaky a=0.3/0.1/0.01.
Group 2 (curvature dial, fixed-alpha Snake): alpha = 0.2/0.6/1.5.
Reference: ELU alpha=1.  Activations are duck-typed or built from H.Activation so the
host module is never edited.
"""
from pathlib import Path
import argparse,json,time,subprocess,sys,concurrent.futures,shutil
import numpy as np
import torch
from src import boundary_gradient_0908 as G
from src.elu_growth_0909 import ELU
H=G.H;ROOT=G.ROOT
OUT=ROOT/'results/gate_scale_invariance_0909'
NTASK=120;NGRAD=4096;REF_SEED=20260909
TRACK=set(range(20,121,10))
E_TASKS=[20,40,60,80,100,120]
# name -> (kind, param);  kind in {'relu','leaky','snake','elu'}
ARMS=[('R','relu',0.),('LR03','leaky',.3),('LR','leaky',.1),('LR001','leaky',.01),
      ('SN02','snake',.2),('SN06','snake',.6),('SN15','snake',1.5),('ELU1','elu',1.)]
def make_act(kind,p):
 if kind=='elu':return ELU(p)
 if kind=='relu':return H.Activation('R','relu')
 if kind=='leaky':return H.Activation('LRx','leaky',p)
 return H.Activation('SNx','snake',p)
def hdef(kind,p,z):
 """h(z)=z phi'(z)-phi(z).  Zero iff phi is positively homogeneous."""
 if kind in ('relu','leaky'):return torch.zeros_like(z)
 if kind=='elu':return torch.where(z>0,torch.zeros_like(z),p*(torch.exp(z.clamp(max=0.))*(z-1)+1))
 return z*torch.sin(2*p*z)-torch.sin(p*z)**2/p          # snake
def d2phi(kind,p,z):
 """phi''(z)."""
 if kind in ('relu','leaky'):return torch.zeros_like(z)
 if kind=='elu':return torch.where(z>0,torch.zeros_like(z),p*torch.exp(z.clamp(max=0.)))
 return 2*p*torch.cos(2*p*z)
def check_derivs(kind,p):
 """phi', phi'' and h against autograd.  Controls must be clearly detectable for every
 arm, including the piecewise-linear ones where phi''==0 makes a parameter nudge invisible."""
 act=make_act(kind,p);z=torch.linspace(-6,6,241,dtype=torch.float64)
 zg=z.clone().requires_grad_(True)
 d1=torch.autograd.grad(act.phi(zg).sum(),zg,create_graph=True)[0]
 e1=float((act.dphi(z)-d1.detach()).abs().max())
 # piecewise-linear phi: d1 carries no graph back to zg, so the second derivative is identically 0
 if d1.requires_grad:
  g2=torch.autograd.grad(d1.sum(),zg,allow_unused=True)[0]
  d2=torch.zeros_like(z) if g2 is None else g2.detach()
 else:
  d2=torch.zeros_like(z)
 e2=float((d2phi(kind,p,z)-d2).abs().max())
 hh=z*act.dphi(z)-act.phi(z);eh=float((hdef(kind,p,z)-hh).abs().max())
 # controls: a clearly different activation, not a small parameter nudge
 if kind in ('relu','leaky'):
  bad_act=H.Activation('m','snake',.6);bk,bp='snake',.6
 elif kind=='elu':
  bad_act=H.Activation('m','leaky',.1);bk,bp='leaky',.1
 else:
  bad_act=make_act('snake',p*1.5);bk,bp='snake',p*1.5
 m1=float((bad_act.dphi(z)-d1.detach()).abs().max())
 m2=float((d2phi(bk,bp,z)-d2).abs().max())
 mh=float((hdef(bk,bp,z)-hh).abs().max())
 return dict(dphi=e1,d2phi=e2,h=eh,dphi_mutctl=m1,d2phi_mutctl=m2,h_mutctl=mh)
def train(kind,p,seed,mnist,px,xg,yg,mutate=None,measure_on=True):
 par=H.init_params(seed,torch.device('cpu'))
 if mutate=='init':par[0].data[0,0]+=1e-3
 act=make_act(kind,p)
 gp,gd,gb=H.stream('perm',seed),H.stream('data',seed),H.stream('batch',seed)
 adam=([torch.zeros_like(q) for q in par],[torch.zeros_like(q) for q in par],[0])
 C=lambda W:(W.double()-W.double().mean(1,keepdim=True))
 prev=C(par[0]).clone();rows=[];units=[];ends={};ck=dict(ident=0.,ident_mut=float('inf'),decomp=0.,decomp_mut=float('inf'),hzero=0.)
 for task in range(1,NTASK+1):
  perm=torch.randperm(784,generator=gp)
  di=H.stratified_draw(mnist,gd);order=torch.randperm(H.TASK_EXAMPLES,generator=gb)
  xs=mnist.train_x[di][:,perm][order];ys=mnist.train_y[di][order]
  track=measure_on and task in TRACK
  if track:sumsq=torch.zeros(100,dtype=torch.float64);sp=C(par[0]).clone()
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
   ck['decomp_mut']=min(ck['decomp_mut'],float((n2-n2p-dd+cr).abs().max()))
   row=dict(task=task,N=float(cur.norm(dim=1).mean()),N2=float(n2.mean()),D2=float(dd.mean()),
    erode=float(cr.mean()),cos=float(((prev*D).sum(1)/(prev.norm(dim=1)*D.norm(dim=1))).mean()),
    rowmean=float(par[0].double().mean(1).mean()),bias=float(par[1].double().mean()))
   if track:
    z=px.double()[:,perm]@par[0].double().T+par[1].double()
    dd2=d2phi(kind,p,z);gate=act.dphi(z.float()).double()
    row.update(steps2=float(sumsq.mean()),persist=float(dd.mean()/sumsq.mean()),
     kappa_g=float((z*dd2).abs().mean()),absd2=float(dd2.abs().mean()),
     gate_cv=float(gate.std()/gate.mean().clamp(min=1e-12)),sd=float(z.std()),zbar=float(z.mean()),
     acc=float((H.forward(par,mnist.test_x[:,perm],act)[4].argmax(1)==mnist.test_y).float().mean()))
    units.append(cur.norm(dim=1).numpy().copy())
   prev=cur.clone()
  if task in E_TASKS:
   q=[t.detach().clone().requires_grad_(True) for t in par]
   o=H.forward(q,xg[:,perm],act);loss=torch.nn.functional.cross_entropy(o[4],yg)
   gg=torch.autograd.grad(loss,q+[o[1]])
   W1,b1,W2=q[0].double(),q[1].double(),q[2].double()
   r1=float((gg[0].double()*W1).sum()+(gg[1].double()*b1).sum());r2=float((gg[2].double()*W2).sum())
   sc=float(gg[0].double().norm()*W1.norm()+gg[1].double().norm()*b1.norm()+gg[2].double().norm()*W2.norm())
   z1=o[0].detach().double()
   Ep=float((gg[6].double()*hdef(kind,p,z1)).sum())
   wk='elu' if kind!='elu' else 'leaky';wp=1. if wk=='elu' else .1
   Ew=float((gg[6].double()*hdef(wk,wp,z1)).sum())
   E=r1-r2;ck['ident']=max(ck['ident'],abs(E-Ep)/sc);ck['ident_mut']=max(ck['ident_mut'] if ck['ident_mut']!=float('inf') else 0.,abs(E-Ew)/sc)
   if kind in ('relu','leaky'):ck['hzero']=max(ck['hzero'],abs(E)/sc)
   row.update(E=E,E_pred=Ep,r_W1b1=r1,r_W2=r2,r_scale=sc)
  rows.append(row)
 return rows,(np.stack(units) if units else None),ends,ck
def g1(seed,ends):
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
def run(name,kind,p,seed):
 torch.set_num_threads(1);H.setup('cpu');mnist=H.Mnist(torch.device('cpu'));OUT.mkdir(parents=True,exist_ok=True)
 tag=f'{name}_s{seed}';t0=time.monotonic()
 px=mnist.test_x[torch.randperm(len(mnist.test_x),generator=H.stream('boundary_probe',seed))[:512]]
 gi=torch.Generator().manual_seed(REF_SEED+seed);gidx=torch.randperm(len(mnist.train_x),generator=gi)[:NGRAD]
 xg=mnist.train_x[gidx];yg=mnist.train_y[gidx]
 rows,units,ends,ck=train(kind,p,seed,mnist,px,xg,yg)
 ck.update(check_derivs(kind,p))
 assert ck['dphi']<1e-6 and ck['d2phi']<1e-6 and ck['h']<1e-6,ck
 assert ck['dphi_mutctl']>1e-3 and ck['d2phi_mutctl']>1e-3 and ck['h_mutctl']>1e-3,ck
 assert ck['ident']<1e-6 and ck['ident_mut']>1e-4,ck
 assert ck['decomp']<1e-10 and ck['decomp_mut']>1e-3,ck
 if kind in ('relu','leaky'):assert ck['hzero']<1e-6,ck
 if name=='LR':
  ck['g1_maxabs'],ck['g1_n']=g1(seed,ends);assert ck['g1_maxabs']==0.,ck
  if seed==0:
   ck['g1_mutctl']=max(float((a-b).abs().max()) for a,b in
     zip(ends[NTASK],train(kind,p,seed,mnist,px,xg,yg,mutate='init',measure_on=False)[2][NTASK]))
   assert ck['g1_mutctl']>1e-4,ck
 if seed==0:
  e3=train(kind,p,seed,mnist,px,xg,yg,measure_on=False)[2]
  ck['meas_noninv']=max(float((a-b).abs().max()) for a,b in zip(ends[NTASK],e3[NTASK]))
  e4=train(kind,p,seed,mnist,px,xg,yg,mutate='measure',measure_on=False)[2]
  ck['meas_mutctl']=max(float((a-b).abs().max()) for a,b in zip(e3[NTASK],e4[NTASK]))
  assert ck['meas_noninv']==0. and ck['meas_mutctl']>0,ck
 for r in rows:r.update(arm=name,kind=kind,param=p,seed=seed)
 keys=[];[keys.append(k) for r in rows for k in r if k not in keys]
 G.B.csvwrite(OUT/(tag+'_rows.csv'),[{k:r.get(k) for k in keys} for r in rows])
 np.savez_compressed(OUT/(tag+'_units.npz'),cnorm_i=units,tasks=np.array(sorted(TRACK)))
 (OUT/(tag+'_provenance.json')).write_text(json.dumps(dict(arm=name,kind=kind,param=p,seed=seed,checks=ck,
  ntask=NTASK,track=sorted(TRACK),code_sha256=G.sha(Path(__file__)),
  spec_sha256=G.sha(ROOT/'specs/spec_gate_scale_invariance_0909.md'),host_sha256=G.sha(Path(H.__file__)),
  data_sha256=mnist.sha256,wall_seconds=time.monotonic()-t0),indent=2))
 print('FINISHED',tag,round(time.monotonic()-t0,1),'s',{k:v for k,v in ck.items() if 'mut' not in k},flush=True)
def backup():
 tg=Path('/home/issan/Projects/obsidian-research-data/gate_scale_invariance_0909');tg.mkdir(parents=True,exist_ok=True);rs=[]
 for f in sorted(OUT.glob('*_units.npz')):
  shutil.copy2(f,tg/f.name);h=G.sha(f);assert G.sha(tg/f.name)==h;rs.append(dict(path=f.name,bytes=f.stat().st_size,sha256=h))
 (OUT/'backup_manifest.json').write_text(json.dumps(dict(target=str(tg),files=rs),indent=2));print('BACKUP',len(rs),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--arm');ap.add_argument('--kind');ap.add_argument('--param',type=float)
 ap.add_argument('--seed',type=int);ap.add_argument('--all',action='store_true');a=ap.parse_args()
 if a.all:
  def job(j):subprocess.run([sys.executable,'-m','src.gate_scale_invariance_0909','--arm',j[0],'--kind',j[1],'--param',str(j[2]),'--seed',str(j[3])],check=True)
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   for _ in pool.map(job,[(n,k,p,s) for n,k,p in ARMS for s in range(3)]):pass
  backup()
 else:run(a.arm,a.kind,a.param,a.seed)

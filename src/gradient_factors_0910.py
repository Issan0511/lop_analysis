"""Gradient factor decomposition + gate-only intervention (spec_gradient_factors_0910).

A: ||grad W1_i||^2 = g2_i * e2_i * xi_i * R_i.  Each factor is computed
   independently from (phi', e, X) and the product is checked against autograd's
   dL/dW1, so the identity can actually fail.
B: backward-only leaky arms -- forward is leaky(a_f)=leaky(0.1) for every BL arm,
   only the gradient flowing through phi uses leaky'(a_b).  BL010 must reproduce
   LR bit for bit.  The host module is never edited; the arms are duck-typed.
"""
from pathlib import Path
import argparse,json,time,subprocess,sys,concurrent.futures,shutil
import numpy as np
import torch
from src import boundary_gradient_0908 as G
from src.elu_growth_0909 import ELU
H=G.H;ROOT=G.ROOT
OUT=ROOT/'results/gradient_factors_0910'
NTASK=120;TRACK=set(range(20,121,10));VT=(100,120)
LR_=.001;B1=.9;B2=.999;EPS=1e-8
CTL=125                    # mutation controls are evaluated every CTL steps

class _BwdLeaky(torch.autograd.Function):
 """forward leaky(af), backward multiplies by leaky'(ab)."""
 @staticmethod
 def forward(ctx,z,af,ab):
  ctx.save_for_backward(z);ctx.ab=ab
  return torch.where(z>0,z,af*z)
 @staticmethod
 def backward(ctx,go):
  z,=ctx.saved_tensors
  return go*torch.where(z>0,torch.ones_like(z),torch.full_like(z,ctx.ab)),None,None

class BwdLeaky:
 """Duck-typed activation.  dphi() returns the BACKWARD gate -- that is what enters
 the gradient and therefore what the factorisation must use."""
 def __init__(s,af,ab):s.af=af;s.ab=ab;s.kind='bwdleaky';s.param=ab
 def phi(s,z):return _BwdLeaky.apply(z,s.af,s.ab)
 def dphi(s,z):return torch.where(z>0,torch.ones_like(z),torch.full_like(z,s.ab))
 def dphi_fwd(s,z):return torch.where(z>0,torch.ones_like(z),torch.full_like(z,s.af))

ARMS=[('LR','leaky',.1),('LR001','leaky',.01),('LR03','leaky',.3),
      ('SN02','snake',.2),('SN06','snake',.6),('SN15','snake',1.5),
      ('ELU1','elu',1.),('LIN','linear',0.),
      ('BL001','bwd',.01),('BL010','bwd',.1),('BL050','bwd',.5),('BL100','bwd',1.)]
def make_act(kind,p):
 if kind=='elu':return ELU(p)
 if kind=='leaky':return H.Activation('LRx','leaky',p)
 if kind=='linear':return H.Activation('LINx','linear')
 if kind=='bwd':return BwdLeaky(.1,p)
 return H.Activation('SNx','snake',p)
def dphi_fwd(act,z):
 return act.dphi_fwd(z) if isinstance(act,BwdLeaky) else act.dphi(z)

def check_dphi(kind,p):
 """phi' against autograd through the actual forward, plus the forward/backward gap."""
 act=make_act(kind,p);z=torch.linspace(-6,6,241,dtype=torch.float64).requires_grad_(True)
 d1=torch.autograd.grad(act.phi(z).sum(),z)[0]
 e=float((act.dphi(z.detach())-d1).abs().max())
 other=make_act('snake',.6) if kind!='snake' else make_act('leaky',.1)
 m=float((other.dphi(z.detach())-d1).abs().max())
 fwd_gap=float((dphi_fwd(act,z.detach())-d1).abs().max())
 return e,m,fwd_gap

class Acc:
 """Log-sums of the four factors over steps and units, plus identity errors.

 Each factor is computed on its own; their product is compared against autograd's
 dL/dW1, so a wrong factor shows up.  Two mutation controls run every CTL steps:
 dropping the gate (d -> e) and, for BL arms, using the forward gate instead of
 the backward one.
 """
 KEYS=('gw2','g2','e2','xi','R')
 def __init__(s):
  s.sum={k:0. for k in s.KEYS};s.n=0
  s.graw=0.;s.phi2=0.
  s.ident=0.;s.mut_nogate=0.;s.mut_fwd=0.;s.rayleigh=0
 def add(s,X,e,p,gW1,pf,ctl):
  B=X.shape[0]
  K=X@X.T                                            # (B,B) Gram
  gw2=(gW1*gW1).sum(1)                               # ground truth, (100,)
  # Relative to the largest unit gradient of the step: a dead unit has gw2==0 in
  # float32 while the float64 reconstruction is tiny-but-nonzero, which makes a
  # per-unit denominator explode.  The layer scale is the honest reference.
  den=gw2.max().clamp(min=1e-300)
  d=e*p                                              # dL/dz1
  g2=(p*p).mean(0)                                   # gate power
  e2=(e*e).sum(0)                                    # error power
  xi=(p*p*e*e).mean(0)/((p*p).mean(0)*(e*e).mean(0)).clamp(min=1e-300)
  q=(d*d).sum(0)
  dh=d/q.sqrt().clamp(min=1e-300)
  R=((dh.T@K)*dh.T).sum(1)                           # Rayleigh quotient d^T K d / ||d||^2
  s.ident=max(s.ident,float(((g2*e2*xi*R-gw2).abs()/den).max()))
  s.graw+=float(gW1.norm(dim=1).mean())              # same definition as step_persistence_0910
  s.phi2+=float((p*p).mean())
  if ctl:
   ev=torch.linalg.eigvalsh(K)
   if bool(((R<ev[0]-1e-9)|(R>ev[-1]+1e-9)).any()):s.rayleigh+=1
   g2n=torch.ones_like(g2);e2n=(e*e).sum(0)
   xin=(e*e*e*e).mean(0)/((e*e).mean(0)**2).clamp(min=1e-300)
   qn=(e*e).sum(0);dn=e/qn.sqrt().clamp(min=1e-300)
   Rn=((dn.T@K)*dn.T).sum(1)
   s.mut_nogate=max(s.mut_nogate,float(((g2n*e2n*xin*Rn-gw2).abs()/den).max()))
   df=e*pf;g2f=(pf*pf).mean(0)
   xif=(pf*pf*e*e).mean(0)/((pf*pf).mean(0)*(e*e).mean(0)).clamp(min=1e-300)
   qf=(df*df).sum(0);dfh=df/qf.sqrt().clamp(min=1e-300)
   Rf=((dfh.T@K)*dfh.T).sum(1)
   s.mut_fwd=max(s.mut_fwd,float(((g2f*e2*xif*Rf-gw2).abs()/den).max()))
  for k,v in (('gw2',gw2),('g2',g2),('e2',e2),('xi',xi),('R',R)):
   s.sum[k]+=float(v.clamp(min=1e-300).log().mean())
  s.n+=1
 def finish(s):
  o={k:float(np.exp(s.sum[k]/s.n)) for k in s.KEYS}
  o['graw']=s.graw/s.n;o['phi2_batch']=s.phi2/s.n
  return o

def train(kind,p,seed,mnist,px,mutate=None,measure_on=True):
 par=H.init_params(seed,torch.device('cpu'))
 if mutate=='init':par[0].data[0,0]+=1e-3
 act=make_act(kind,p)
 gp,gd,gb=H.stream('perm',seed),H.stream('data',seed),H.stream('batch',seed)
 adam=([torch.zeros_like(q) for q in par],[torch.zeros_like(q) for q in par],[0])
 C=lambda W:(W.double()-W.double().mean(1,keepdim=True))
 prev=C(par[0]).clone();rows=[];ends={}
 ck=dict(ident=0.,mut_nogate=float('inf'),mut_fwd=float('inf'),rayleigh=0)
 for task in range(1,NTASK+1):
  perm=torch.randperm(784,generator=gp)
  di=H.stratified_draw(mnist,gd);order=torch.randperm(H.TASK_EXAMPLES,generator=gb)
  xs=mnist.train_x[di][:,perm][order];ys=mnist.train_y[di][order]
  track=measure_on and task in TRACK
  if track:
   A=Acc();sp=C(par[0]).clone()
   S2=torch.zeros(100,dtype=torch.float64);tot=torch.zeros(100,784,dtype=torch.float64)
  for step in range(625):
   xb=xs[step*16:(step+1)*16];yb=ys[step*16:(step+1)*16]
   o=H.forward(par,xb,act);loss=torch.nn.functional.cross_entropy(o[4],yb)
   if track:
    gg=torch.autograd.grad(loss,list(par)+[o[1]])
    g=gg[:6]
    with torch.no_grad():
     A.add(xb.double(),gg[6].double(),act.dphi(o[0].detach()).double(),g[0].double(),
           dphi_fwd(act,o[0].detach()).double(),step%CTL==0)
   else:
    g=torch.autograd.grad(loss,par)
   with torch.no_grad():
    m_,v_,tc=adam;tc[0]+=1;c1=1-B1**tc[0];c2=1-B2**tc[0]
    for q,gr,mi,vi in zip(par,g,m_,v_):
     mi.mul_(B1).add_(gr,alpha=1-B1);vi.mul_(B2).addcmul_(gr,gr,value=1-B2)
     q-=LR_*(mi/c1)/((vi/c2).sqrt()+EPS)
    if track:
     cur=C(par[0]);u=cur-sp;S2+=(u*u).sum(1);tot+=u;sp=cur.clone()
  if mutate=='measure':
   with torch.no_grad():par[0].data.add_(1e-9)
  ends[task]=[q.detach().clone() for q in par]
  if not measure_on:
   with torch.no_grad():prev=C(par[0]).clone()
   continue
  with torch.no_grad():
   cur=C(par[0]);D=cur-prev
   n2=(cur*cur).sum(1);dd=(D*D).sum(1)
   row=dict(task=task,N=float(cur.norm(dim=1).mean()),N2=float(n2.mean()),D2=float(dd.mean()),
    cos=float(((prev*D).sum(1)/(prev.norm(dim=1)*D.norm(dim=1))).mean()))
   if track:
    r=A.finish()
    ck['ident']=max(ck['ident'],A.ident)
    ck['mut_nogate']=min(ck['mut_nogate'],A.mut_nogate)
    ck['mut_fwd']=min(ck['mut_fwd'],A.mut_fwd)
    ck['rayleigh']+=A.rayleigh
    s2=float(S2.mean());d2=float((tot*tot).sum(1).mean())
    z=px.double()[:,perm]@par[0].double().T+par[1].double()
    pb=act.dphi(z.float()).double();pfw=dphi_fwd(act,z.float()).double()
    row.update(r,S2=s2,rho=d2/s2,
     phi2_probe=float((pb*pb).mean()),phi2_probe_fwd=float((pfw*pfw).mean()),
     sigma=float(z.std()),zbar=float(z.mean()),posfrac=float((z>0).double().mean()),
     acc=float((H.forward(par,mnist.test_x[:,perm],act)[4].argmax(1)==mnist.test_y).float().mean()))
   prev=cur.clone();rows.append(row)
 return rows,ends,ck

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

def external(name,seed,rows):
 """C4: graw must reproduce step_persistence_0910 on the reference arms."""
 import csv
 src=ROOT/f'results/step_persistence_0910/{name}_s{seed}_rows.csv'
 if not src.exists():return None,None
 old={int(d['task']):float(d['graw']) for d in csv.DictReader(src.open()) if d.get('graw')}
 w=0.;wm=0.
 for r in rows:
  if r['task'] in old and 'graw' in r:
   w=max(w,abs(r['graw']-old[r['task']])/old[r['task']])
   o=old.get(r['task']+10)
   if o:wm=max(wm,abs(r['graw']-o)/o)
 return w,wm

def run(name,kind,p,seed):
 torch.set_num_threads(1);H.setup('cpu');mnist=H.Mnist(torch.device('cpu'));OUT.mkdir(parents=True,exist_ok=True)
 tag=f'{name}_s{seed}';t0=time.monotonic()
 px=mnist.test_x[torch.randperm(len(mnist.test_x),generator=H.stream('boundary_probe',seed))[:512]]
 rows,ends,ck=train(kind,p,seed,mnist,px)
 ck['dphi'],ck['dphi_mutctl'],ck['fwd_gap']=check_dphi(kind,p)
 assert ck['dphi']<1e-14 and ck['dphi_mutctl']>0.5,ck
 assert ck['ident']<1e-3,ck
 assert ck['mut_nogate']>1e-2,ck          # dropping the gate must break the identity
 assert ck['rayleigh']==0,ck              # R stays inside the spectrum of K
 if kind=='bwd' and p!=.1:
  assert ck['fwd_gap']>1e-3 and ck['mut_fwd']>1e-3,ck   # the backward really differs
 ck['c4'],ck['c4_mutctl']=external(name,seed,rows)
 if ck['c4'] is not None:assert ck['c4']==0. and ck['c4_mutctl']>1e-2,ck
 if name in ('LR','BL010'):
  ck['g1_maxabs'],ck['g1_n']=g1(seed,ends);assert ck['g1_maxabs']==0.,ck
 if name=='BL010':
  e2=train('leaky',.1,seed,mnist,px,measure_on=False)[1]
  ck['b0']=max(float((a-b).abs().max()) for a,b in zip(ends[NTASK],e2[NTASK]))
  e3=train('bwd',.5,seed,mnist,px,measure_on=False)[1]
  ck['b0_mutctl']=max(float((a-b).abs().max()) for a,b in zip(ends[NTASK],e3[NTASK]))
  assert ck['b0']==0. and ck['b0_mutctl']>1e-3,ck
 if seed==0:
  if name=='LR':
   ck['g1_mutctl']=max(float((a-b).abs().max()) for a,b in
     zip(ends[NTASK],train(kind,p,seed,mnist,px,mutate='init',measure_on=False)[1][NTASK]))
   assert ck['g1_mutctl']>1e-4,ck
  e4=train(kind,p,seed,mnist,px,measure_on=False)[1]
  ck['c5']=max(float((a-b).abs().max()) for a,b in zip(ends[NTASK],e4[NTASK]))
  e5=train(kind,p,seed,mnist,px,mutate='measure',measure_on=False)[1]
  ck['c5_mutctl']=max(float((a-b).abs().max()) for a,b in zip(e4[NTASK],e5[NTASK]))
  assert ck['c5']==0. and ck['c5_mutctl']>0,ck
 for r in rows:r.update(arm=name,kind=kind,param=p,seed=seed)
 keys=[];[keys.append(k) for r in rows for k in r if k not in keys]
 G.B.csvwrite(OUT/(tag+'_rows.csv'),[{k:r.get(k) for k in keys} for r in rows])
 (OUT/(tag+'_provenance.json')).write_text(json.dumps(dict(arm=name,kind=kind,param=p,seed=seed,
  checks=ck,ntask=NTASK,track=sorted(TRACK),ctl=CTL,code_sha256=G.sha(Path(__file__)),
  spec_sha256=G.sha(ROOT/'specs/spec_gradient_factors_0910.md'),host_sha256=G.sha(Path(H.__file__)),
  data_sha256=mnist.sha256,wall_seconds=time.monotonic()-t0),indent=2))
 print('FINISHED',tag,round(time.monotonic()-t0,1),'s',{k:v for k,v in ck.items() if 'mut' not in k},flush=True)

def backup():
 tg=Path('/home/issan/Projects/obsidian-research-data/gradient_factors_0910');tg.mkdir(parents=True,exist_ok=True);rs=[]
 for f in sorted(OUT.glob('*_rows.csv'))+sorted(OUT.glob('*_provenance.json')):
  shutil.copy2(f,tg/f.name);h=G.sha(f);assert G.sha(tg/f.name)==h
  rs.append(dict(path=f.name,bytes=f.stat().st_size,sha256=h))
 (OUT/'backup_manifest.json').write_text(json.dumps(dict(files=rs),indent=2));print('BACKUP',len(rs))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--arm');ap.add_argument('--seed',type=int)
 ap.add_argument('--all',action='store_true');ap.add_argument('--jobs',type=int,default=8)
 ap.add_argument('--backup',action='store_true');a=ap.parse_args()
 if a.backup:backup()
 elif a.all:
  jobs=[(n,k,p,s) for (n,k,p) in ARMS for s in range(3)]
  with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
   fs=[ex.submit(subprocess.run,[sys.executable,'-m','src.gradient_factors_0910',
        '--arm',n,'--seed',str(s)],cwd=str(ROOT)) for (n,k,p,s) in jobs]
   print('RC',[f.result().returncode for f in fs])
 else:
  n,k,p=[x for x in ARMS if x[0]==a.arm][0];run(n,k,p,a.seed)

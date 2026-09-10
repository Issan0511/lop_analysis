"""Within-task step persistence rho = D^2/S^2 (spec_step_persistence_0910).

The committed data already refutes the "Adam step is shorter for Snake" hypothesis
(S^2 varies 5.1% across arms).  This run measures WHERE the persistence lives:
block factors G(b), lag correlation A(tau), and the short/long split of the cross
term X = D^2 - S^2.  Host module is never edited; activations are duck-typed or
built from H.Activation.
"""
from pathlib import Path
import argparse,json,time,subprocess,sys,concurrent.futures,shutil
import numpy as np
import torch
from src import boundary_gradient_0908 as G
from src.elu_growth_0909 import ELU
H=G.H;ROOT=G.ROOT
OUT=ROOT/'results/step_persistence_0910'
NTASK=120;TRACK=set(range(20,121,10));VERDICT_TASKS=(100,120)
SUB=5                      # keep every SUB-th centred update for the long-lag matrix
NSUB=625//SUB              # 125
SHORT=(1,2,3,4)            # lags accumulated step by step
LAGSPLIT=50                # tau < 50 -> short, tau >= 50 -> long
BLOCKS=(1,5,25,125,625)
LR_=.001;B1=.9;B2=.999;EPS=1e-8
ARMS=[('LR','leaky',.1),('LR001','leaky',.01),('LR03','leaky',.3),
      ('SN02','snake',.2),('SN06','snake',.6),('SN15','snake',1.5),
      ('ELU1','elu',1.),('LIN','linear',0.)]
def make_act(kind,p):
 if kind=='elu':return ELU(p)
 if kind=='leaky':return H.Activation('LRx','leaky',p)
 if kind=='linear':return H.Activation('LINx','linear')
 return H.Activation('SNx','snake',p)
def check_dphi(kind,p):
 """phi' against autograd, with a control that must be clearly detectable."""
 act=make_act(kind,p);z=torch.linspace(-6,6,241,dtype=torch.float64).requires_grad_(True)
 d1=torch.autograd.grad(act.phi(z).sum(),z)[0]
 e=float((act.dphi(z.detach())-d1).abs().max())
 other=make_act('snake',.6) if kind!='snake' else make_act('leaky',.1)
 m=float((other.dphi(z.detach())-d1).abs().max())
 return e,m

class Tracker:
 """Accumulates everything the spec needs for one tracked task.

 Convention matches the committed D^2: quantities are (.)**2 summed over the 784
 input coordinates of a row, then averaged over the 100 units.
 """
 def __init__(s,nu=100,nin=784):
  s.S2=torch.zeros(nu,dtype=torch.float64)            # sum_s ||u_s||^2
  s.tot=torch.zeros(nu,nin,dtype=torch.float64)       # sum_s u_s
  s.blk={b:(torch.zeros(nu,nin,dtype=torch.float64),torch.zeros(nu,dtype=torch.float64))
         for b in BLOCKS}                             # partial sum, accumulated ||block||^2
  s.short={t:[torch.zeros(nu,dtype=torch.float64),0] for t in SHORT}  # sum cos, count
  s.hist=[]                                           # last max(SHORT) centred updates
  s.sub=[]                                            # every SUB-th centred update (float32)
  s.n=0
  s.kap1=0.;s.kap2=0.;s.graw=0.;s.nstep=0
  s.pred2=0.;s.pred2_mut=0.                           # Adam-form step budget (C5)
 def add(s,u):
  """u: centred applied update for W1 at this step, float64 (nu, nin)."""
  s.n+=1
  nn=(u*u).sum(1)
  s.S2+=nn
  s.tot+=u
  for b,(acc,sq) in s.blk.items():
   acc+=u
   if s.n%b==0:
    sq+=(acc*acc).sum(1);acc.zero_()
  nrm=nn.sqrt().clamp(min=1e-300)
  for t in SHORT:
   if len(s.hist)>=t:
    v=s.hist[-t]
    s.short[t][0]+=(v*u).sum(1)/(( v*v).sum(1).sqrt().clamp(min=1e-300)*nrm)
    s.short[t][1]+=1
  s.hist.append(u.clone())
  if len(s.hist)>max(SHORT):s.hist.pop(0)
  if s.n%SUB==0:s.sub.append(u.to(torch.float32).clone())
 def finish(s):
  D2=float((s.tot*s.tot).sum(1).mean());S2=float(s.S2.mean())
  out=dict(S2=S2,D2_track=D2,rho=D2/S2)
  # every G(b) comes from the same block machinery, including b=1 and b=625, so the
  # two identities below are checks on the accumulator rather than on arithmetic.
  for b in BLOCKS:
   acc,sq=s.blk[b]
   assert float(acc.abs().max())==0.,'block did not close evenly'
   out[f'G{b}']=float(sq.mean())/S2
  M=torch.stack(s.sub).double()                          # (NSUB, nu, nin)
  P=torch.einsum('ijk,ljk->il',M,M)                      # unnormalised inner products
  nrm=(M*M).sum(2).sqrt().clamp(min=1e-300)              # (NSUB, nu)
  Cm=(torch.einsum('ijk,ljk->ilj',M,M)/(nrm[:,None,:]*nrm[None,:,:])).mean(2).numpy()
  lag=np.abs(np.subtract.outer(np.arange(NSUB),np.arange(NSUB)))*SUB
  for t in SHORT:
   c,k=s.short[t];out[f'A{t}']=float(c.mean()/k)
  for tau in (5,10,20,50,100,200,400):
   m=lag==tau
   out[f'A{tau}']=float(Cm[m].mean()) if m.any() else float('nan')
  # Cross term X = D2 - S2 = sum_{s!=s'} <u_s,u_s'>, split by |s-s'|.  The exact split
  # needs all 390000 ordered pairs; we take the SHAPE from the 5-step subsample and
  # check that the subsample reproduces the exact X once rescaled (C4).
  Pn=P.numpy()/M.shape[1]                                # per-unit mean, matching S2/D2
  ms=(lag<LAGSPLIT)&(lag>0);ml=lag>=LAGSPLIT
  ps,pl=float(Pn[ms].sum()),float(Pn[ml].sum())
  scale=(625*624)/(NSUB*NSUB-NSUB)
  X=D2-S2
  out['X']=X;out['X_hat']=scale*(ps+pl)
  out['c4']=abs(out['X_hat']/X-1.) if X!=0 else float('nan')
  out['c4_mutctl']=abs((ps+pl)/X-1.) if X!=0 else float('nan')   # forgetting the scale
  out['X_short_frac']=ps/(ps+pl) if (ps+pl)!=0 else float('nan')
  out['X_long_frac']=pl/(ps+pl) if (ps+pl)!=0 else float('nan')
  out['kap1']=s.kap1/s.nstep;out['kap2']=s.kap2/s.nstep
  out['graw']=s.graw/s.nstep
  out['c5']=abs(s.pred2/S2-1.);out['c5_mutctl']=abs(s.pred2_mut/S2-1.)
  out['_tot']=s.tot
  return out

def train(kind,p,seed,mnist,px,mutate=None,measure_on=True):
 par=H.init_params(seed,torch.device('cpu'))
 if mutate=='init':par[0].data[0,0]+=1e-3
 act=make_act(kind,p)
 gp,gd,gb=H.stream('perm',seed),H.stream('data',seed),H.stream('batch',seed)
 adam=([torch.zeros_like(q) for q in par],[torch.zeros_like(q) for q in par],[0])
 C=lambda W:(W.double()-W.double().mean(1,keepdim=True))
 prev=C(par[0]).clone();rows=[];ends={}
 ck=dict(c1=0.,c2=0.,c3=True,c4=0.,c4_mutctl=float('inf'),c5=0.,c5_mutctl=float('inf'))
 for task in range(1,NTASK+1):
  perm=torch.randperm(784,generator=gp)
  di=H.stratified_draw(mnist,gd);order=torch.randperm(H.TASK_EXAMPLES,generator=gb)
  xs=mnist.train_x[di][:,perm][order];ys=mnist.train_y[di][order]
  track=measure_on and task in TRACK
  if track:
   T=Tracker();sp=C(par[0]).clone();wstart=sp.clone()
   with torch.no_grad():
    z0=px.double()[:,perm]@par[0].double().T+par[1].double()
    g0=act.dphi(z0.float()).double().flatten()
  for step in range(625):
   xb=xs[step*16:(step+1)*16];yb=ys[step*16:(step+1)*16]
   o=H.forward(par,xb,act);loss=torch.nn.functional.cross_entropy(o[4],yb)
   g=torch.autograd.grad(loss,par)
   with torch.no_grad():
    m_,v_,tc=adam;tc[0]+=1;c1=1-B1**tc[0];c2=1-B2**tc[0]
    if track:
     T.graw+=float(g[0].double().norm(dim=1).mean());T.nstep+=1
    for q,gr,mi,vi in zip(par,g,m_,v_):
     mi.mul_(B1).add_(gr,alpha=1-B1);vi.mul_(B2).addcmul_(gr,gr,value=1-B2)
     q-=LR_*(mi/c1)/((vi/c2).sqrt()+EPS)
    if track:
     r=(m_[0].double()/c1)/(((v_[0].double()/c2).sqrt())+EPS)
     T.kap1+=float(r.abs().mean());T.kap2+=float((r*r).mean())
     # C5: the Adam form must reproduce the step that was actually applied.
     pr=-LR_*r;pr=pr-pr.mean(1,keepdim=True)
     T.pred2+=float((pr*pr).sum(1).mean())
     rw=(m_[0].double()/c1)/(((v_[0].double()/c2).sqrt())+1e-3)
     pw=-LR_*rw;pw=pw-pw.mean(1,keepdim=True)
     T.pred2_mut+=float((pw*pw).sum(1).mean())
     cur=C(par[0]);T.add(cur-sp);sp=cur.clone()
  if mutate=='measure':
   with torch.no_grad():par[0].data.add_(1e-9)
  ends[task]=[q.detach().clone() for q in par]
  if not measure_on:
   with torch.no_grad():prev=C(par[0]).clone()
   continue
  with torch.no_grad():
   cur=C(par[0]);D=cur-prev
   n2=(cur*cur).sum(1);dd=(D*D).sum(1);cr=2*(prev*D).sum(1)
   row=dict(task=task,N=float(cur.norm(dim=1).mean()),N2=float(n2.mean()),D2=float(dd.mean()),
    erode=float(cr.mean()),cos=float(((prev*D).sum(1)/(prev.norm(dim=1)*D.norm(dim=1))).mean()))
   if track:
    r=T.finish();tot=r.pop('_tot')
    ck['c1']=max(ck['c1'],float((tot-(cur-wstart)).abs().max()))
    ck['c2']=max(ck['c2'],abs(r['G1']-1.),abs(r['G625']-r['rho']))
    ck['c3']=ck['c3'] and all(r[f'G{b}']<=b*(1+1e-9) for b in BLOCKS)
    ck['c4']=max(ck['c4'],r['c4']);ck['c4_mutctl']=min(ck['c4_mutctl'],r['c4_mutctl'])
    ck['c5']=max(ck['c5'],r['c5']);ck['c5_mutctl']=min(ck['c5_mutctl'],r['c5_mutctl'])
    z1=px.double()[:,perm]@par[0].double().T+par[1].double()
    g1v=act.dphi(z1.float()).double().flatten()
    gc=float(((g0-g0.mean())*(g1v-g1v.mean())).sum()/
        (g0.std(unbiased=False)*g1v.std(unbiased=False)*g0.numel()).clamp(min=1e-30)) \
        if float(g0.std())>1e-12 and float(g1v.std())>1e-12 else 1.0
    for k in ('c4','c4_mutctl','c5','c5_mutctl'):r.pop(k)
    row.update(r);row.update(gcorr=gc,sigma=float(z1.std()),zbar=float(z1.mean()),
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
 """C6: S^2 and D^2 must reproduce the committed run for the same arm/seed."""
 import csv
 src=ROOT/f'results/gate_scale_invariance_0909/{name}_s{seed}_rows.csv'
 if not src.exists():src=ROOT/f'results/linear_growth_0910/{name}_s{seed}_rows.csv'
 if not src.exists():return None,None
 old={}
 for d in csv.DictReader(src.open()):
  if d.get('steps2'):old[int(d['task'])]=(float(d['steps2']),float(d['D2']))
 w=0.;wm=0.
 for r in rows:
  if r['task'] in old and 'S2' in r:
   s0,d0=old[r['task']]
   w=max(w,abs(r['S2']-s0)/s0,abs(r['D2']-d0)/d0)
   o=old.get(r['task']+10)
   if o:wm=max(wm,abs(r['S2']-o[0])/o[0])
 return w,wm

def run(name,kind,p,seed):
 torch.set_num_threads(1);H.setup('cpu');mnist=H.Mnist(torch.device('cpu'));OUT.mkdir(parents=True,exist_ok=True)
 tag=f'{name}_s{seed}';t0=time.monotonic()
 px=mnist.test_x[torch.randperm(len(mnist.test_x),generator=H.stream('boundary_probe',seed))[:512]]
 rows,ends,ck=train(kind,p,seed,mnist,px)
 ck['dphi'],ck['dphi_mutctl']=check_dphi(kind,p)
 assert ck['dphi']<1e-14 and ck['dphi_mutctl']>0.5,ck
 assert ck['c1']<1e-10,ck
 assert ck['c2']<1e-12,ck
 assert ck['c3'],ck
 assert ck['c4']<0.35 and ck['c4_mutctl']>0.5,ck
 assert ck['c5']<1e-3 and ck['c5_mutctl']>.05,ck
 ck['c6'],ck['c6_mutctl']=external(name,seed,rows)
 if ck['c6'] is not None:assert ck['c6']<1e-6 and ck['c6_mutctl']>1e-3,ck
 if name=='LR':
  ck['g1_maxabs'],ck['g1_n']=g1(seed,ends);assert ck['g1_maxabs']==0.,ck
  if seed==0:
   ck['g1_mutctl']=max(float((a-b).abs().max()) for a,b in
     zip(ends[NTASK],train(kind,p,seed,mnist,px,mutate='init',measure_on=False)[1][NTASK]))
   assert ck['g1_mutctl']>1e-4,ck
 if seed==0:
  e3=train(kind,p,seed,mnist,px,measure_on=False)[1]
  ck['c7']=max(float((a-b).abs().max()) for a,b in zip(ends[NTASK],e3[NTASK]))
  e4=train(kind,p,seed,mnist,px,mutate='measure',measure_on=False)[1]
  ck['c7_mutctl']=max(float((a-b).abs().max()) for a,b in zip(e3[NTASK],e4[NTASK]))
  assert ck['c7']==0. and ck['c7_mutctl']>0,ck
 for r in rows:r.update(arm=name,kind=kind,param=p,seed=seed)
 keys=[];[keys.append(k) for r in rows for k in r if k not in keys]
 G.B.csvwrite(OUT/(tag+'_rows.csv'),[{k:r.get(k) for k in keys} for r in rows])
 (OUT/(tag+'_provenance.json')).write_text(json.dumps(dict(arm=name,kind=kind,param=p,seed=seed,
  checks=ck,ntask=NTASK,track=sorted(TRACK),blocks=list(BLOCKS),sub=SUB,lagsplit=LAGSPLIT,
  code_sha256=G.sha(Path(__file__)),spec_sha256=G.sha(ROOT/'specs/spec_step_persistence_0910.md'),
  host_sha256=G.sha(Path(H.__file__)),data_sha256=mnist.sha256,
  wall_seconds=time.monotonic()-t0),indent=2))
 print('FINISHED',tag,round(time.monotonic()-t0,1),'s',
       {k:v for k,v in ck.items() if 'mut' not in k},flush=True)

def backup():
 tg=Path('/home/issan/Projects/obsidian-research-data/step_persistence_0910');tg.mkdir(parents=True,exist_ok=True);rs=[]
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
   fs=[ex.submit(subprocess.run,[sys.executable,'-m','src.step_persistence_0910',
        '--arm',n,'--seed',str(s)],cwd=str(ROOT)) for (n,k,p,s) in jobs]
   bad=[f.result().returncode for f in fs]
  print('RC',bad)
 else:
  n,k,p=[x for x in ARMS if x[0]==a.arm][0];run(n,k,p,a.seed)

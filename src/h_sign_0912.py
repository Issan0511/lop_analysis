"""Break homogeneity on purpose (spec_h_sign_0912).

phi_eps(z) = z + eps*z^2 (z>0), 0.1*z (z<=0).  h = eps*z^2 on z>0, zero elsewhere.
Reuses the gradient-factor machinery of gradient_factors_0910 unchanged and adds
the Euler quantities of gate_scale_invariance_0909 plus per-layer radial drift.
Host module never edited.
"""
from pathlib import Path
import argparse,json,time,subprocess,sys,concurrent.futures,shutil
import numpy as np
import torch
from src import boundary_gradient_0908 as G
from src.gradient_factors_0910 import Acc,dphi_fwd
from src.gate_scale_invariance_0909 import hdef as hdef_ref
H=G.H;ROOT=G.ROOT
OUT=ROOT/'results/h_sign_0912'
NTASK=120;TRACK=set(range(20,121,10));E_TASKS=(20,40,60,80,100,120);NGRAD=4096;REF_SEED=20260909
LR_=.001;B1=.9;B2=.999;EPS=1e-8;CTL=125

class HLeaky:
 """leaky(0.1) plus eps*z^2 on the positive side.  dphi is the true derivative."""
 def __init__(s,eps):s.eps=eps;s.a=.1;s.kind='hleaky';s.param=eps
 def phi(s,z):return torch.where(z>0,z+s.eps*z*z,s.a*z)
 def dphi(s,z):return torch.where(z>0,1+2*s.eps*z,torch.full_like(z,s.a))
 def h(s,z):return torch.where(z>0,s.eps*z*z,torch.zeros_like(z))

ARMS=[('HM03',-.03),('HM01',-.01),('H00',0.),('HP01',.01),('HP03',.03),('HP10',.10)]
def make_act(eps):return HLeaky(eps)

def check_act(eps):
 act=make_act(eps);z=torch.linspace(-6,6,241,dtype=torch.float64).requires_grad_(True)
 d1=torch.autograd.grad(act.phi(z).sum(),z)[0];zd=z.detach()
 c1=float((act.dphi(zd)-d1).abs().max())
 c1m=float((make_act(eps+.02).dphi(zd)-d1).abs().max())
 c2=float((act.h(zd)-(zd*d1-act.phi(zd))).abs().max())
 c2m=float((hdef_ref('elu',1.,zd)-(zd*d1-act.phi(zd))).abs().max())
 return c1,c1m,c2,c2m

def train(eps,seed,mnist,px,xg,yg,mutate=None,measure_on=True,plain=False):
 par=H.init_params(seed,torch.device('cpu'))
 if mutate=='init':par[0].data[0,0]+=1e-3
 act=H.Activation('LRx','leaky',.1) if plain else make_act(eps)
 gp,gd,gb=H.stream('perm',seed),H.stream('data',seed),H.stream('batch',seed)
 adam=([torch.zeros_like(q) for q in par],[torch.zeros_like(q) for q in par],[0])
 C=lambda W:(W.double()-W.double().mean(1,keepdim=True))
 prev=C(par[0]).clone();rows=[];ends={}
 ck=dict(ident=0.,mut_nogate=float('inf'),rayleigh=0,addgap=0.,alive_min=1.,euler=0.,euler_mut=float('inf'),e_h00=0.)
 for task in range(1,NTASK+1):
  perm=torch.randperm(784,generator=gp)
  di=H.stratified_draw(mnist,gd);order=torch.randperm(H.TASK_EXAMPLES,generator=gb)
  xs=mnist.train_x[di][:,perm][order];ys=mnist.train_y[di][order]
  track=measure_on and task in TRACK
  if track:
   A=Acc();sp=C(par[0]).clone()
   S2=torch.zeros(100,dtype=torch.float64);tot=torch.zeros(100,784,dtype=torch.float64)
   W1s=par[0].detach().double().clone();W2s=par[2].detach().double().clone()
  for step in range(625):
   xb=xs[step*16:(step+1)*16];yb=ys[step*16:(step+1)*16]
   o=H.forward(par,xb,act);loss=torch.nn.functional.cross_entropy(o[4],yb)
   if track:
    gg=torch.autograd.grad(loss,list(par)+[o[1]]);g=gg[:6]
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
    cos=float(((prev*D).sum(1)/(prev.norm(dim=1)*D.norm(dim=1))).mean()),
    w2col=float(par[2].double().norm(dim=0).mean()))
   if track:
    r=A.finish()
    ck['ident']=max(ck['ident'],A.ident);ck['mut_nogate']=min(ck['mut_nogate'],A.mut_nogate)
    ck['rayleigh']+=A.rayleigh;ck['addgap']=max(ck['addgap'],r['addgap']);ck['alive_min']=min(ck['alive_min'],r['alive'])
    s2=float(S2.mean());d2=float((tot*tot).sum(1).mean())
    W1e=par[0].double();W2e=par[2].double()
    z=px.double()[:,perm]@W1e.T+par[1].double();pb=act.dphi(z.float()).double()
    row.update(r,S2=s2,rho=d2/s2,phi2_probe=float((pb*pb).mean()),
     sigma=float(z.std()),zbar=float(z.mean()),posfrac=float((z>0).double().mean()),
     rad1=float(((W1e-W1s)*W1s).sum()/(W1s*W1s).sum()),rad2=float(((W2e-W2s)*W2s).sum()/(W2s*W2s).sum()),
     acc=float((H.forward(par,mnist.test_x[:,perm],act)[4].argmax(1)==mnist.test_y).float().mean()))
   prev=cur.clone()
  if task in E_TASKS:
   q=[t.detach().clone().requires_grad_(True) for t in par]
   o=H.forward(q,xg[:,perm],act);loss=torch.nn.functional.cross_entropy(o[4],yg)
   gg=torch.autograd.grad(loss,q+[o[1]])
   W1,b1,W2=q[0].double(),q[1].double(),q[2].double()
   W1,b1,W2=W1.detach(),b1.detach(),W2.detach()
   r1=float((gg[0].double()*W1).sum()+(gg[1].double()*b1).sum());r2=float((gg[2].double()*W2).sum())
   sc=float(gg[0].double().norm()*W1.norm()+gg[1].double().norm()*b1.norm()+gg[2].double().norm()*W2.norm())
   z1=o[0].detach().double();E=r1-r2
   Ep=float((gg[6].double()*act.h(z1)).sum()) if not plain else 0.
   Ew=float((gg[6].double()*hdef_ref('snake',.6,z1)).sum())   # control: a large wrong h
   ck['euler']=max(ck['euler'],abs(E-Ep)/sc);ck['euler_mut']=min(ck['euler_mut'],abs(E-Ew)/sc)
   if not plain and eps==0.:ck['e_h00']=max(ck['e_h00'],abs(E)/sc)
   row.update(E=E,E_pred=Ep,r_W1b1=r1,r_W2=r2,r_scale=sc)
  rows.append(row)
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

def external(seed,rows):
 import csv
 src=ROOT/f'results/gradient_factors_0910/LR_s{seed}_rows.csv'
 old={int(d['task']):(float(d['graw']),float(d['e2']),float(d['rho'])) for d in csv.DictReader(src.open()) if d.get('graw')}
 w=0.;wm=0.
 for r in rows:
  if r['task'] in old and 'graw' in r:
   o=old[r['task']];w=max(w,abs(r['graw']-o[0])/o[0],abs(r['e2']-o[1])/o[1],abs(r['rho']-o[2])/o[2])
   o2=old.get(r['task']+10)
   if o2:wm=max(wm,abs(r['graw']-o2[0])/o2[0])
 return w,wm

def run(name,eps,seed):
 torch.set_num_threads(1);H.setup('cpu');mnist=H.Mnist(torch.device('cpu'));OUT.mkdir(parents=True,exist_ok=True)
 tag=f'{name}_s{seed}';t0=time.monotonic()
 px=mnist.test_x[torch.randperm(len(mnist.test_x),generator=H.stream('boundary_probe',seed))[:512]]
 gi=torch.Generator().manual_seed(REF_SEED+seed);gidx=torch.randperm(len(mnist.train_x),generator=gi)[:NGRAD]
 xg=mnist.train_x[gidx];yg=mnist.train_y[gidx]
 rows,ends,ck=train(eps,seed,mnist,px,xg,yg)
 ck['c1'],ck['c1_mutctl'],ck['c2'],ck['c2_mutctl']=check_act(eps)
 assert ck['c1']<1e-14 and ck['c1_mutctl']>1e-2 and ck['c2']<1e-14 and ck['c2_mutctl']>.1,ck
 assert ck['euler']<1e-3 and ck['euler_mut']>1e-3,ck
 assert ck['ident']<1e-3 and ck['mut_nogate']>1e-2 and ck['rayleigh']==0 and ck['addgap']<1e-6,ck
 if name=='H00':
  assert ck['e_h00']<1e-9,ck
  ck['c4'],ck['c4_mutctl']=external(seed,rows);assert ck['c4']==0. and ck['c4_mutctl']>1e-2,ck
  ck['g1_maxabs'],ck['g1_n']=g1(seed,ends);assert ck['g1_maxabs']==0.,ck
  e2=train(0.,seed,mnist,px,xg,yg,measure_on=False,plain=True)[1]
  ck['b0']=max(float((a-b).abs().max()) for a,b in zip(ends[NTASK],e2[NTASK]))
  e3=train(.01,seed,mnist,px,xg,yg,measure_on=False)[1]
  ck['b0_mutctl']=max(float((a-b).abs().max()) for a,b in zip(ends[NTASK],e3[NTASK]))
  assert ck['b0']==0. and ck['b0_mutctl']>1e-3,ck
  if seed==0:
   ck['g1_mutctl']=max(float((a-b).abs().max()) for a,b in
     zip(ends[NTASK],train(eps,seed,mnist,px,xg,yg,mutate='init',measure_on=False)[1][NTASK]))
   assert ck['g1_mutctl']>1e-4,ck
 if name=='HP01' and seed==0:
  ck['e_hp01']=max(abs(r['E'])/r['r_scale'] for r in rows if 'E' in r)   # C7 control
  assert ck['e_hp01']>1e-5,ck
 if seed==0:
  e4=train(eps,seed,mnist,px,xg,yg,measure_on=False)[1]
  ck['c6']=max(float((a-b).abs().max()) for a,b in zip(ends[NTASK],e4[NTASK]))
  e5=train(eps,seed,mnist,px,xg,yg,mutate='measure',measure_on=False)[1]
  ck['c6_mutctl']=max(float((a-b).abs().max()) for a,b in zip(e4[NTASK],e5[NTASK]))
  assert ck['c6']==0. and ck['c6_mutctl']>0,ck
 for r in rows:r.update(arm=name,eps=eps,seed=seed)
 keys=[];[keys.append(k) for r in rows for k in r if k not in keys]
 G.B.csvwrite(OUT/(tag+'_rows.csv'),[{k:r.get(k) for k in keys} for r in rows])
 (OUT/(tag+'_provenance.json')).write_text(json.dumps(dict(arm=name,eps=eps,seed=seed,checks=ck,
  ntask=NTASK,track=sorted(TRACK),e_tasks=list(E_TASKS),code_sha256=G.sha(Path(__file__)),
  spec_sha256=G.sha(ROOT/'specs/spec_h_sign_0912.md'),host_sha256=G.sha(Path(H.__file__)),
  data_sha256=mnist.sha256,wall_seconds=time.monotonic()-t0),indent=2))
 print('FINISHED',tag,round(time.monotonic()-t0,1),'s',{k:v for k,v in ck.items() if 'mut' not in k},flush=True)

def backup():
 tg=Path('/home/issan/Projects/obsidian-research-data/h_sign_0912');tg.mkdir(parents=True,exist_ok=True);rs=[]
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
  jobs=[(n,e,s) for (n,e) in ARMS for s in range(3)]
  with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
   fs=[ex.submit(subprocess.run,[sys.executable,'-m','src.h_sign_0912','--arm',n,'--seed',str(s)],cwd=str(ROOT)) for (n,e,s) in jobs]
   print('RC',[f.result().returncode for f in fs])
 else:
  n,e=[x for x in ARMS if x[0]==a.arm][0];run(n,e,a.seed)

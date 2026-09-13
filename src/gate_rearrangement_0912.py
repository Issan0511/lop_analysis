"""Gate rearrangement at the task boundary (spec_gate_rearrangement_0912).

Same overlap dial as perm_overlap_0912, plus a direct measurement of how much the
gate pattern phi'(z) changes when only the input permutation changes -- same W1,
same b1, two permutations.  Host module never edited.
"""
from pathlib import Path
import argparse,json,time,subprocess,sys,concurrent.futures,shutil
import numpy as np
import torch
from src import boundary_gradient_0908 as G
from src.gradient_factors_0910 import Acc,dphi_fwd
from src.elu_growth_0909 import ELU
H=G.H;ROOT=G.ROOT
OUT=ROOT/'results/gate_rearrangement_0912'
NTASK=120;TRACK=set(range(20,121,10));VT=(100,120);NGRAD=4096;REF_SEED=20260909
LR_=.001;B1=.9;B2=.999;EPS=1e-8;CTL=125
KS=[('K000',0.),('K010',.10),('K100',1.)]
ACTS=[('LR001','leaky',.01),('LR','leaky',.1),('LR03','leaky',.3),('ELU1','elu',1.),
      ('SN02','snake',.2),('SN06','snake',.6),('SN15','snake',1.5)]
ARMS=[(f'{kn}_{an}',kv,ak,ap) for (kn,kv) in KS for (an,ak,ap) in ACTS]
def make_act(kind,p):
 if kind=='elu':return ELU(p)
 if kind=='leaky':return H.Activation('LRx','leaky',p)
 return H.Activation('SNx','snake',p)

def next_perm(prev,k,gp,gs):
 fresh=torch.randperm(784,generator=gp)
 if k>=1.:return fresh,True
 if prev is None:return fresh,True
 m=int(round(k*784))
 if m<2:return prev.clone(),False
 sel=torch.randperm(784,generator=gs)[:m]
 new=prev.clone();new[sel]=prev[sel][torch.randperm(m,generator=gs)]
 return new,False

def gate_stats(par,act,px,pold,pnew):
 """How much does phi'(z) rearrange when only the permutation changes?"""
 with torch.no_grad():
  W=par[0].double();b=par[1].double();x=px.double()
  a=act.dphi((x[:,pold]@W.T+b).float()).double().flatten()
  c=act.dphi((x[:,pnew]@W.T+b).float()).double().flatten()
  sa,sc=a.std(unbiased=False),c.std(unbiased=False)
  if float(sa)<1e-12 or float(sc)<1e-12:return 1.,0.,1.
  gc=float(((a-a.mean())*(c-c.mean())).mean()/(sa*sc))
  gcr=float(torch.corrcoef(torch.stack([a,c]))[0,1])        # P4: independent code path
  l1=float((c-a).abs().mean()/a.abs().mean().clamp(min=1e-30))
  return gc,l1,gcr

def shock(par,act,xg,yg,perm):
 with torch.no_grad():
  o=H.forward(par,xg[:,perm],act)
  return (float(torch.nn.functional.cross_entropy(o[4],yg)),
          float((o[4].argmax(1)==yg).float().mean()))

def train(k,kind,p,seed,mnist,px,xg,yg,mutate=None,measure_on=True):
 par=H.init_params(seed,torch.device('cpu'))
 if mutate=='init':par[0].data[0,0]+=1e-3
 act=make_act(kind,p)
 gp,gd,gb=H.stream('perm',seed),H.stream('data',seed),H.stream('batch',seed)
 gs=H.stream('overlap',seed);gnull=torch.Generator().manual_seed(REF_SEED+7919+seed)
 adam=([torch.zeros_like(q) for q in par],[torch.zeros_like(q) for q in par],[0])
 C=lambda W:(W.double()-W.double().mean(1,keepdim=True))
 prev=C(par[0]).clone();rows=[];ends={};pperm=None
 ck=dict(ident=0.,mut_nogate=float('inf'),rayleigh=0,addgap=0.,valid=True,host_path=0,
         f_err=0.,idx50=None,order50=None,p4=0.,p4_self=0.,p4_shuf=0.,null_gc=None)
 for task in range(1,NTASK+1):
  perm,host=next_perm(pperm,k,gp,gs);ck['host_path']+=int(host)
  ck['valid']=ck['valid'] and bool(torch.equal(perm.sort().values,torch.arange(784)))
  f=1.0 if pperm is None else float((perm==pperm).double().mean())
  gc=gl1=None
  if measure_on and pperm is not None:
   gc,gl1,gcr=gate_stats(par,act,px,pperm,perm)
   ck['p4']=max(ck['p4'],abs(gc-gcr))
   if task==60:                                    # P4 control + P5 null
    with torch.no_grad():
     W=par[0].double();b=par[1].double();x=px.double()
     a=act.dphi((x[:,pperm]@W.T+b).float()).double().flatten()
     cc=act.dphi((x[:,perm]@W.T+b).float()).double().flatten()
     sh=cc[torch.randperm(cc.numel(),generator=gnull)]      # control: break the pairing
     ck['p4_self']=abs(float(torch.corrcoef(torch.stack([a,a]))[0,1])-1.)
     ck['p4_shuf']=abs(float(torch.corrcoef(torch.stack([a,sh]))[0,1]))
    q1=torch.randperm(784,generator=gnull);q2=torch.randperm(784,generator=gnull)
    ck['null_gc']=gate_stats(par,act,px,q1,q2)[0]
  pperm=perm.clone()
  di=H.stratified_draw(mnist,gd);order=torch.randperm(H.TASK_EXAMPLES,generator=gb)
  if task==50:ck['idx50']=int(di[:8].sum());ck['order50']=int(order[:8].sum())
  xs=mnist.train_x[di][:,perm][order];ys=mnist.train_y[di][order]
  sce,sacc=(shock(par,act,xg,yg,perm) if measure_on else (None,None))
  track=measure_on and task in TRACK
  if track:
   A=Acc();sp=C(par[0]).clone()
   S2=torch.zeros(100,dtype=torch.float64);tot=torch.zeros(100,784,dtype=torch.float64)
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
   row=dict(task=task,f=f,shock_ce=sce,shock_acc=sacc,gate_corr=gc,gate_l1=gl1,
    N=float(cur.norm(dim=1).mean()),N2=float(n2.mean()),D2=float(dd.mean()),
    cos=float(((prev*D).sum(1)/(prev.norm(dim=1)*D.norm(dim=1))).mean()))
   if track:
    r=A.finish()
    ck['ident']=max(ck['ident'],A.ident);ck['mut_nogate']=min(ck['mut_nogate'],A.mut_nogate)
    ck['rayleigh']+=A.rayleigh;ck['addgap']=max(ck['addgap'],r['addgap'])
    s2=float(S2.mean());d2=float((tot*tot).sum(1).mean())
    z=px.double()[:,perm]@par[0].double().T+par[1].double()
    row.update(r,S2=s2,rho=d2/s2,sigma=float(z.std()),zbar=float(z.mean()),
     nz=(float(z.std())*p/np.pi if kind=='snake' else 1.0),
     acc=float((H.forward(par,mnist.test_x[:,perm],act)[4].argmax(1)==mnist.test_y).float().mean()))
   prev=cur.clone();rows.append(row)
 ck['f_err']=abs(np.mean([r['f'] for r in rows if r['task']>=2])-(1-k)) if rows and k<1 else 0.
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
 """C1: cells shared with perm_overlap_0912 must reproduce exactly."""
 import csv
 kn,an=name.split('_',1)
 if an not in ('LR','SN06'):return None,None
 src=ROOT/f'results/perm_overlap_0912/{kn}_{an}_s{seed}_rows.csv'
 if not src.exists():return None,None
 old={int(d['task']):(float(d['D2']),float(d['cos']),float(d['rho'])) for d in csv.DictReader(src.open()) if d.get('rho')}
 alt=ROOT/f'results/perm_overlap_0912/{"K050" if kn!="K050" else "K025"}_{an}_s{seed}_rows.csv'
 oth={int(d['task']):float(d['rho']) for d in csv.DictReader(alt.open())if d.get('rho')} if alt.exists() else {}
 w=0.;wm=0.
 for r in rows:
  if r['task'] in old and 'rho' in r:
   o=old[r['task']]
   w=max(w,abs(r['D2']-o[0])/abs(o[0]),abs(r['cos']-o[1])/abs(o[1]),abs(r['rho']-o[2])/o[2])
   if r['task'] in oth:wm=max(wm,abs(r['rho']-oth[r['task']])/oth[r['task']])
 return w,wm

def run(name,k,kind,p,seed):
 torch.set_num_threads(1);H.setup('cpu');mnist=H.Mnist(torch.device('cpu'));OUT.mkdir(parents=True,exist_ok=True)
 tag=f'{name}_s{seed}';t0=time.monotonic()
 px=mnist.test_x[torch.randperm(len(mnist.test_x),generator=H.stream('boundary_probe',seed))[:512]]
 gi=torch.Generator().manual_seed(REF_SEED+seed);gidx=torch.randperm(len(mnist.train_x),generator=gi)[:NGRAD]
 xg=mnist.train_x[gidx];yg=mnist.train_y[gidx]
 rows,ends,ck=train(k,kind,p,seed,mnist,px,xg,yg)
 assert ck['valid'] and ck['f_err']<=0.02,ck
 assert ck['host_path']==(NTASK if k>=1. else 1),ck
 assert ck['p4']<1e-9 and ck['p4_self']<1e-12 and ck['p4_shuf']<0.05,ck
 assert ck['ident']<1e-3 and ck['mut_nogate']>1e-2 and ck['rayleigh']==0 and ck['addgap']<1e-6,ck
 ck['c1'],ck['c1_mutctl']=external(name,seed,rows)
 if ck['c1'] is not None:assert ck['c1']==0. and ck['c1_mutctl']>1e-2,ck
 if name=='K100_LR':
  ck['g1_maxabs'],ck['g1_n']=g1(seed,ends);assert ck['g1_maxabs']==0.,ck
  if seed==0:
   ck['g1_mutctl']=max(float((a-b).abs().max()) for a,b in
     zip(ends[NTASK],train(k,kind,p,seed,mnist,px,xg,yg,mutate='init',measure_on=False)[1][NTASK]))
   assert ck['g1_mutctl']>1e-4,ck
 if seed==0:
  e4=train(k,kind,p,seed,mnist,px,xg,yg,measure_on=False)[1]
  ck['c2']=max(float((a-b).abs().max()) for a,b in zip(ends[NTASK],e4[NTASK]))
  e5=train(k,kind,p,seed,mnist,px,xg,yg,mutate='measure',measure_on=False)[1]
  ck['c2_mutctl']=max(float((a-b).abs().max()) for a,b in zip(e4[NTASK],e5[NTASK]))
  assert ck['c2']==0. and ck['c2_mutctl']>0,ck
 for r in rows:r.update(arm=name,k=k,act=kind,param=p,seed=seed,actname=name.split('_',1)[1])
 keys=[];[keys.append(x) for r in rows for x in r if x not in keys]
 G.B.csvwrite(OUT/(tag+'_rows.csv'),[{x:r.get(x) for x in keys} for r in rows])
 (OUT/(tag+'_provenance.json')).write_text(json.dumps(dict(arm=name,k=k,act=kind,param=p,seed=seed,
  checks=ck,ntask=NTASK,track=sorted(TRACK),code_sha256=G.sha(Path(__file__)),
  spec_sha256=G.sha(ROOT/'specs/spec_gate_rearrangement_0912.md'),host_sha256=G.sha(Path(H.__file__)),
  data_sha256=mnist.sha256,wall_seconds=time.monotonic()-t0),indent=2))
 print('FINISHED',tag,round(time.monotonic()-t0,1),'s',{x:y for x,y in ck.items() if 'mut' not in x},flush=True)

def backup():
 tg=Path('/home/issan/Projects/obsidian-research-data/gate_rearrangement_0912');tg.mkdir(parents=True,exist_ok=True);rs=[]
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
  jobs=[(n,kv,ak,ap_,s) for (n,kv,ak,ap_) in ARMS for s in range(3)]
  with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
   fs=[ex.submit(subprocess.run,[sys.executable,'-m','src.gate_rearrangement_0912','--arm',n,'--seed',str(s)],
       cwd=str(ROOT)) for (n,kv,ak,ap_,s) in jobs]
   print('RC',[f.result().returncode for f in fs])
 else:
  n,kv,ak,ap_=[x for x in ARMS if x[0]==a.arm][0];run(n,kv,ak,ap_,a.seed)

"""First-layer norm growth and sinking for ELU (spec_elu_growth_0909).

Trains 120 tasks from scratch with the same init and RNG streams as the existing
arms (S-init: the arm does not enter the RNG), so LR/SNA must reproduce the
committed checkpoints exactly.  ELU is duck-typed here so the host module is
never edited and its sha256 stays stable for existing provenance.
"""
from pathlib import Path
import argparse,json,time,subprocess,sys,concurrent.futures,shutil
import numpy as np
import torch
from src import boundary_gradient_0908 as G
H=G.H;ROOT=G.ROOT
OUT=ROOT/'results/elu_growth_0909'
NTASK=120;NREF=8;REF_SEED=20260909;NGRAD=4096;SAT=0.05
ARMS=[('LR','none',0.),('SNA','none',0.),('ELU1','none',1.0),('ELU03','none',0.3),('ELU1','l2',1.0)]
E_TASKS=[20,40,60,80,100,120]
class ELU:
 """phi(z)=z (z>0), a(e^z-1) (z<=0).  Duck-typed to match Activation."""
 kind='elu'
 def __init__(self,a):self.param=float(a);self.name=f'ELU{a}'
 def phi(self,z):return torch.where(z>0,z,self.param*torch.expm1(z.clamp(max=0.)))
 def dphi(self,z):return torch.where(z>0,torch.ones_like(z),self.param*torch.exp(z.clamp(max=0.)))
def make_act(arm,alpha):
 if arm=='SNA':return H.AdaptiveSnake(.6,.01,'cpu')
 if arm.startswith('ELU'):return ELU(alpha)
 return H.ARMS['LR']
def hdefect(act,z,layer0_alpha=None):
 """h(z) = z phi'(z) - phi(z); zero iff phi is positively homogeneous."""
 if isinstance(act,ELU):
  a=act.param;return torch.where(z>0,torch.zeros_like(z),a*(torch.exp(z.clamp(max=0.))*(z-1)+1))
 if isinstance(act,H.AdaptiveSnake):
  al=layer0_alpha;return z*(1+torch.sin(2*al*z))-(z+torch.sin(al*z)**2/al)
 return torch.zeros_like(z)
def refperms():
 g=torch.Generator().manual_seed(REF_SEED);return [torch.randperm(784,generator=g) for _ in range(NREF)]
def measure(p,act,px,perm,refs,xg,yg,mnist,want_grad,want_acc):
 with torch.no_grad():
  W=p[0].double();b=p[1].double();x=px.double()
  m=W.mean(1,keepdim=True);Wt=W-m
  z=x[:,perm]@W.T+b;zm=z.mean(0)
  zr=torch.stack([x[:,r]@W.T+b for r in refs])
  zim=zr.mean(1).mean(0);ziv=zr.var(1,unbiased=False).mean(0)
  gate=act.dphi(z.float()).double()
  row=dict(zbar_cur=float(z.mean()),sigma_cur=float(z.var(0,unbiased=False).mean().sqrt()),
   between_cur=float(zm.var(unbiased=False)),pos_frac=float((z>0).double().mean()),
   zbar_inv=float(zim.mean()),sigma_inv=float(ziv.mean().sqrt()),between_inv=float(zim.var(unbiased=False)),
   cnorm=float(Wt.norm(dim=1).mean()),cnorm2=float((Wt*Wt).sum(1).mean()),rowmean=float(m.mean()),
   bias=float(b.mean()),w2col=float(p[2].double().norm(dim=0).mean()),
   sat=float((gate<SAT).double().mean()),gate_mean=float(gate.mean()),
   dead_units=int((gate.max(0).values<SAT).sum()))
  units=dict(cnorm_i=Wt.norm(dim=1).numpy().copy(),zbar_i=zim.numpy().copy(),sd_i=ziv.sqrt().numpy().copy())
 if want_acc:
  with torch.no_grad():row['acc']=float((H.forward(p,mnist.test_x[:,perm],act)[4].argmax(1)==mnist.test_y).float().mean())
 if want_grad:
  q=[t.detach().clone().requires_grad_(True) for t in p]
  o=H.forward(q,xg[:,perm],act);loss=torch.nn.functional.cross_entropy(o[4],yg)
  g=torch.autograd.grad(loss,q+[o[1]])
  W1,b1,W2=q[0].double(),q[1].double(),q[2].double()
  r1=float((g[0].double()*W1).sum()+(g[1].double()*b1).sum());r2=float((g[2].double()*W2).sum())
  al=act.alpha(0).double() if isinstance(act,H.AdaptiveSnake) else None
  hh=hdefect(act,o[0].detach().double(),al)
  # 内積の自然なスケール（打ち消し前）。r1,r2 自身は零交差するので分母に使わない。
  scale=float(g[0].double().norm()*W1.norm()+g[1].double().norm()*b1.norm()+g[2].double().norm()*W2.norm())
  wrong=H.ARMS['LR'] if isinstance(act,(ELU,H.AdaptiveSnake)) else ELU(1.0)
  hw=hdefect(wrong,o[0].detach().double(),al)
  row.update(r_W1b1=r1,r_W2=r2,E=r1-r2,E_pred=float((g[6].double()*hh).sum()),
   E_pred_wrong=float((g[6].double()*hw).sum()),r_scale=scale,ce_loss=float(loss),
   wd_radial=float(2*.001*((W1*W1).sum()+(b1*b1).sum())) if True else 0.)
 return row,units
def train(arm,iv,alpha,seed,mnist,px,refs,xg,yg,mutate=None,measure_on=True):
 p=H.init_params(seed,torch.device('cpu'))
 if mutate=='init':p[0].data[0,0]+=1e-3
 act=make_act(arm,alpha)
 gp,gd,gb=H.stream('perm',seed),H.stream('data',seed),H.stream('batch',seed)
 adam=([torch.zeros_like(q) for q in p],[torch.zeros_like(q) for q in p],[0])
 rows=[];units={};ends={}
 for task in range(1,NTASK+1):
  perm=torch.randperm(784,generator=gp)
  idx=H.stratified_draw(mnist,gd);order=torch.randperm(H.TASK_EXAMPLES,generator=gb)
  xs=mnist.train_x[idx][:,perm][order];ys=mnist.train_y[idx][order]
  if measure_on:
   r,_=measure(p,act,px,perm,refs,xg,yg,mnist,task in E_TASKS,False)
   rows.append(dict(task=task,phase='start',**r))
  for step in range(625):
   xb=xs[step*16:(step+1)*16];yb=ys[step*16:(step+1)*16]
   out=H.forward(p,xb,act);loss=torch.nn.functional.cross_entropy(out[4],yb)
   g=torch.autograd.grad(loss,p)
   with torch.no_grad():
    if iv!='none':g=[gr+2*.001*q for q,gr in zip(p,g)]
    m_,v_,tc=adam;tc[0]+=1;c1=1-.9**tc[0];c2=1-.999**tc[0]
    for q,gr,mi,vi in zip(p,g,m_,v_):
     mi.mul_(.9).add_(gr,alpha=1-.9);vi.mul_(.999).addcmul_(gr,gr,value=1-.999)
     q-=.001*(mi/c1)/((vi/c2).sqrt()+1e-8)
    if isinstance(act,H.AdaptiveSnake):act.update(out[0],out[2])
  if mutate=='measure':
   with torch.no_grad():p[0].data.add_(1e-9)
  ends[task]=[q.detach().clone() for q in p]
  if measure_on:
   r,u=measure(p,act,px,perm,refs,xg,yg,mnist,task in E_TASKS,True)
   rows.append(dict(task=task,phase='end',**r))
   for k,v in u.items():units.setdefault(k,[]).append(v)
 return rows,{k:np.stack(v) for k,v in units.items()},ends,p,act
def g1_check(arm,seed,ends):
 """LR/SNA must reproduce the committed checkpoints exactly."""
 R=ROOT/'results';worst=0.;n=0
 for tag,t in [('task1',1),('task100',100)]:
  cp=torch.load(R/'boundary_groups_0908'/f'{arm}_none_s{seed}_{tag}.pt',weights_only=False,map_location='cpu')
  worst=max(worst,max(float((a-b).abs().max()) for a,b in zip(ends[t],cp['state']['params'])));n+=1
 for src in ['boundary_tasks20_40_0909/source','boundary_groups_0908']:
  sv=torch.load(R/src/f'{arm}_none_s{seed}_states.pt',weights_only=False,map_location='cpu')
  for raw in sv['boundaries']:
   t=int(raw['task'])
   if 'after_625' in raw and t in ends:
    worst=max(worst,max(float((a-b).abs().max()) for a,b in zip(ends[t],raw['after_625']['state']['params'])));n+=1
 return worst,n
def run(arm,iv,alpha,seed):
 torch.set_num_threads(1);H.setup('cpu');mnist=H.Mnist(torch.device('cpu'));OUT.mkdir(parents=True,exist_ok=True)
 tag=f'{arm}_{iv}_s{seed}';refs=refperms()
 px=mnist.test_x[torch.randperm(len(mnist.test_x),generator=H.stream('boundary_probe',seed))[:512]]
 gi=torch.Generator().manual_seed(REF_SEED+seed)
 gidx=torch.randperm(len(mnist.train_x),generator=gi)[:NGRAD]
 xg=mnist.train_x[gidx];yg=mnist.train_y[gidx]
 start=time.monotonic();ck={}
 rows,units,ends,p,act=train(arm,iv,alpha,seed,mnist,px,refs,xg,yg)
 # ---- checks ----
 if arm in ('LR','SNA') and iv=='none':
  ck['g1_maxabs'],ck['g1_n']=g1_check(arm,seed,ends)
  assert ck['g1_maxabs']==0.,('G1 failed',ck)
  if seed==0:
   _,_,e2,_,_=train(arm,iv,alpha,seed,mnist,px,refs,xg,yg,mutate='init',measure_on=False)
   ck['g1_mutctl']=max(float((a-b).abs().max()) for a,b in zip(ends[NTASK],e2[NTASK]))
   assert ck['g1_mutctl']>1e-4,ck
 gr=[r for r in rows if 'E' in r]
 ck['identity_rel']=max(abs(r['E']-r['E_pred'])/r['r_scale'] for r in gr)
 ck['identity_mutctl']=max(abs(r['E']-r['E_pred_wrong'])/r['r_scale'] for r in gr)
 assert ck['identity_rel']<1e-6,ck
 assert ck['identity_mutctl']>1e-4,ck
 if arm=='LR':
  ck['leaky_zero']=max(abs(r['E'])/r['r_scale'] for r in gr)
  assert ck['leaky_zero']<1e-6,ck
 if seed==0:
  _,_,e3,_,_=train(arm,iv,alpha,seed,mnist,px,refs,xg,yg,measure_on=False)
  ck['measure_noninvasive']=max(float((a-b).abs().max()) for a,b in zip(ends[NTASK],e3[NTASK]))
  _,_,e4,_,_=train(arm,iv,alpha,seed,mnist,px,refs,xg,yg,mutate='measure',measure_on=False)
  ck['measure_mutctl']=max(float((a-b).abs().max()) for a,b in zip(e3[NTASK],e4[NTASK]))
  assert ck['measure_noninvasive']==0. and ck['measure_mutctl']>0,ck
  if arm.startswith('ELU'):
   z=torch.linspace(-6,6,201,dtype=torch.float64).requires_grad_(True)
   a_=ELU(alpha);ref=torch.autograd.grad(a_.phi(z).sum(),z)[0]
   ck['elu_dphi']=float((a_.dphi(z)-ref).abs().max())
   ck['elu_dphi_mutctl']=float((ELU(alpha*1.01).dphi(z)-ref).abs().max())
   assert ck['elu_dphi']<1e-6 and ck['elu_dphi_mutctl']>1e-3,ck
 for r in rows:r.update(arm=arm,iv=iv,alpha=alpha,seed=seed)
 keys=[];[keys.append(k) for r in rows for k in r if k not in keys]
 G.B.csvwrite(OUT/(tag+'_rows.csv'),[{k:r.get(k) for k in keys} for r in rows])
 np.savez_compressed(OUT/(tag+'_units.npz'),**units)
 (OUT/(tag+'_provenance.json')).write_text(json.dumps(dict(arm=arm,iv=iv,alpha=alpha,seed=seed,checks=ck,
  ntask=NTASK,nref=NREF,ref_seed=REF_SEED,ngrad=NGRAD,sat=SAT,code_sha256=G.sha(Path(__file__)),
  spec_sha256=G.sha(ROOT/'specs/spec_elu_growth_0909.md'),host_sha256=G.sha(Path(H.__file__)),
  data_sha256=mnist.sha256,wall_seconds=time.monotonic()-start,
  scope='fresh CPU training, same init/streams as the existing arms'),indent=2))
 print('FINISHED',tag,round(time.monotonic()-start,1),'s',ck,flush=True)
def backup():
 target=Path('/home/issan/Projects/obsidian-research-data/elu_growth_0909');target.mkdir(parents=True,exist_ok=True);rows=[]
 for f in sorted(OUT.glob('*_units.npz')):
  dst=target/f.name;shutil.copy2(f,dst);h=G.sha(f);assert G.sha(dst)==h;rows.append(dict(path=f.name,bytes=f.stat().st_size,sha256=h))
 (OUT/'backup_manifest.json').write_text(json.dumps(dict(target=str(target),files=rows,total_bytes=sum(r['bytes'] for r in rows)),indent=2))
 print('BACKUP',len(rows),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--arm');ap.add_argument('--iv');ap.add_argument('--alpha',type=float,default=0.)
 ap.add_argument('--seed',type=int);ap.add_argument('--all',action='store_true');a=ap.parse_args()
 if a.all:
  def job(j):subprocess.run([sys.executable,'-m','src.elu_growth_0909','--arm',j[0],'--iv',j[1],'--alpha',str(j[2]),'--seed',str(j[3])],check=True)
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   for _ in pool.map(job,[(A,I,AL,s) for A,I,AL in ARMS for s in range(3)]):pass
  backup()
 else:run(a.arm,a.iv,a.alpha,a.seed)

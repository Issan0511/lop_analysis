"""Diagnostic factor exchange for the current-CE sign reversal (spec_ce_factor_swap_0909).

Replays the saved task21..40 boundaries for 40 updates, evaluates the CE gradient on a
FIXED diagnostic set at every update, decomposes the projected current-CE displacement
into (sample, unit) terms, and recomputes all 16 combinations of {r, J, g, D} taken from
two pre-registered update indices.  No new training, no new trajectory.
"""
from pathlib import Path
import argparse,json,time,subprocess,sys,concurrent.futures,shutil,itertools
import numpy as np
import torch
from src import boundary_gradient_0908 as G
H=G.H;ROOT=G.ROOT
SRC=ROOT/'results/boundary_tasks20_40_0909/source'
OUT=ROOT/'results/ce_factor_swap_0909'
STEPS=40;DN=512;NBLOCK=8;PAIRS=[(1,15),(1,40)]
FACTORS=['r','J','g','D']
WINDOWS=[('w1',1,1),('w2_5',2,5),('w6_10',6,10),('w11_20',11,20),('w21_30',21,30),('w31_40',31,40)]
def gates(snap,z_src,a_src,ada):
 """phi' of layer1 and layer2 with preactivations from z_src and alpha from a_src."""
 z1,z2=snap[z_src]['z1'],snap[z_src]['z2']
 if not ada:return torch.where(z1>0,1.,.1),torch.where(z2>0,1.,.1)
 a0,a1=snap[a_src]['alpha0'],snap[a_src]['alpha1']
 return 1.+torch.sin(2.*a0*z1),1.+torch.sin(2.*a1*z2)
def cell(snap,r_s,gate2_s,WW_s,g1_s,w_s,ada,alpha_s=None):
 """Projected current-CE displacement per unit, factors taken from the named states."""
 al=alpha_s or {}
 g1,_=gates(snap,g1_s,al.get('g1',g1_s),ada)
 _,g2=gates(snap,gate2_s,al.get('g2',gate2_s),ada)
 r=snap[r_s]['r'];W2=snap[WW_s]['W2'];W3=snap[WW_s]['W3'];w=snap[w_s]['w']
 u=((r@W3)*g2)@W2
 t=.1*(g1*u)*w
 return t.sum(0)
def ud(t,nunit):
 pos=t.clamp(min=0).sum()/nunit;neg=(-t.clamp(max=0)).sum()/nunit
 return float(pos),float(neg)
def boundary(raw,arm,iv,mnist,px,reference,mu_global,preflight,mutate=None):
 ada=arm=='SNA'
 p,act,adam=G.clone_state(raw,arm)
 gd=torch.Generator();gd.set_state(raw['rng_after_perm']['data'])
 gb=torch.Generator();gb.set_state(raw['rng_after_perm']['batch'])
 idx=H.stratified_draw(mnist,gd);order=torch.randperm(H.TASK_EXAMPLES,generator=gb)
 xs=mnist.train_x[idx][:,raw['perm']][order];ys=mnist.train_y[idx][order]
 xp=px[:,raw['perm']];xmu=xp.double().mean(0)
 dx=xs[-DN:];dy=ys[-DN:];dx64=dx.double();xmud=dx64*xmu
 blk=torch.arange(DN)//(DN//NBLOCK)
 with torch.no_grad():
  z0=H.forward(p,dx,act)[0]
 fixedpos=(z0>0)                      # (s,i) classes frozen at the switch
 ck=dict(delta=0.,delta_mutctl=float('inf'),additive=0.,additive_mutctl=float('inf'),
         replay=0.,ud=0.,ud_mutctl=float('inf'),corner=0.,corner_mutctl=float('inf'))
 rows=[];snap={};coef1=None
 for step in range(1,STEPS+1):
  with torch.no_grad():                      # ---- diagnostic at the pre-update state ----
   z1,a1,z2,a2,lg=H.forward(p,dx,act)
   pr=torch.softmax(lg.double(),1);r=pr.clone();r[torch.arange(DN),dy]-=1.;r/=DN
   al0=act.alpha(0).double() if ada else None;al1=act.alpha(1).double() if ada else None
   s_now={'z1':z1.double(),'z2':z2.double(),'r':r,'W2':p[2].double().clone(),
          'W3':p[4].double().clone(),'alpha0':al0,'alpha1':al1}
   g1n,g2n=gates({'x':s_now},'x','x',ada)
   delta=g1n*(((r@s_now['W3'])*g2n)@s_now['W2'])
  if preflight:
   pz=[q.detach().clone().requires_grad_(True) for q in p]
   o=H.forward(pz,dx,act);ag=torch.autograd.grad(torch.nn.functional.cross_entropy(o[4],dy),o[0])[0].double()
   ck['delta']=max(ck['delta'],float((delta-ag).abs().max()))
   ck['delta_mutctl']=min(ck['delta_mutctl'],float(((g1n+.1)*(((r@s_now['W3'])*g2n)@s_now['W2'])-ag).abs().max()))
  if mutate=='diag_update' and ada:act.update(z1,z2)
  if mutate=='diag_param':p[0].data.add_(1e-9)
  xb=xs[(step-1)*16:step*16];yb=ys[(step-1)*16:step*16]      # ---- real update ----
  out=H.forward(p,xb,act);loss=torch.nn.functional.cross_entropy(out[4],yb)
  g=torch.autograd.grad(loss,p)
  with torch.no_grad():
   before=[q.clone() for q in p[:2]]
   reg=[2*.001*q if iv=='l2' else torch.zeros_like(q) for q in p]
   m,v,tc=adam;tc[0]+=1;c1=1-.9**tc[0];c2=1-.999**tc[0];coef=[]
   for k,(q,gr,mi,vi) in enumerate(zip(p,[a+b for a,b in zip(g,reg)],m,v)):
    prior=mi.double().clone()*.9 if k<2 else None
    mi.mul_(.9).add_(gr,alpha=1-.9);vi.mul_(.999).addcmul_(gr,gr,value=1-.999)
    denom=(vi/c2).sqrt()+1e-8
    if k<2:
     coef.append(-.001/c1/denom.double())
     if k==0:hist0=prior
     else:hist1=prior
    q-=.001*(mi/c1)/denom
   if mutate=='replay':p[0].data[0,0]+=1e-3
   actual=[p[k].double()-before[k].double() for k in range(2)]
   w=xmud@coef[0].T+coef[1]
   if step==1:coef1=[c.clone() for c in coef]
   s_now['w']=w;s_now['coefW']=coef[0];s_now['coefb']=coef[1]
   # exact (s,i) decomposition of the projected current-CE displacement
   t_si=.1*delta*w
   netu=t_si.sum(0)
   gW=delta.T@dx64;gb_=delta.sum(0)
   dz_param=(.1*(coef[0]*gW))@xmu+.1*coef[1]*gb_
   ck['additive']=max(ck['additive'],float((netu-dz_param).abs().max()))
   if preflight:
    bad=.1*(g1n*(((r*1.01@s_now['W3'])*g2n)@s_now['W2']))*w
    ck['additive_mutctl']=min(ck['additive_mutctl'],float((bad.sum(0)-dz_param).abs().max()))
   nu=netu.shape[0]
   U,D=ud(t_si,nu)
   scale=max(U+D,1e-30);net=float(netu.mean())
   ck['ud']=max(ck['ud'],abs((U-D)-net)/scale)
   ck['ud_mutctl']=min(ck['ud_mutctl'],abs((D-U)-net)/scale)
   Up,Dp=ud(t_si*fixedpos,nu);Un,Dn=ud(t_si*(~fixedpos),nu)
   frozen=(.1*(coef1[0]*gW))@xmu+.1*coef1[1]*gb_
   sgd=(-.001*gW)@xmu-.001*gb_
   blocks=[float((.1*delta[blk==j]*w[blk==j]).sum(0).mean())*NBLOCK for j in range(NBLOCK)]
   # the original training-batch quantities, for cross-reference with the 0909 run
   tr_cur=[coef[k]*((1-.9)*g[k].double()) for k in range(2)]
   tr_hist=[coef[0]*hist0,coef[1]*hist1]
   rows.append(dict(step=step,diag_net=net,diag_U=U,diag_D=D,
    diag_net_pos=Up-Dp,diag_U_pos=Up,diag_D_pos=Dp,diag_net_neg=Un-Dn,diag_U_neg=Un,diag_D_neg=Dn,
    diag_frozenD=float(frozen.mean()),diag_sgd=float(sgd.mean()),
    diag_block_min=min(blocks),diag_block_max=max(blocks),
    diag_block_negfrac=float(np.mean([b<0 for b in blocks])),
    train_current=float((tr_cur[0]@xmu+tr_cur[1]).mean()),
    train_history=float((tr_hist[0]@xmu+tr_hist[1]).mean()),
    train_actual=float((actual[0]@xmu+actual[1]).mean())))
   if step in {s for pr_ in PAIRS for s in pr_}:snap[step]={**s_now,'netu':netu}
   if ada:act.update(out[0],out[2])
   zr=H.forward(p,xp,act)[0].double().mean(0)
   ck['replay']=max(ck['replay'],float((zr-torch.as_tensor(reference[step-1])).abs().max()))
 swaps=[]
 for (ta,tb) in PAIRS:
  sn={'A':snap[ta],'B':snap[tb]}
  for combo in itertools.product('AB',repeat=4):
   xr,xJ,xg,xD=combo
   netu=cell(sn,xr,xJ,xJ,xg,xD,ada)
   swaps.append(dict(pair=f'{ta}_{tb}',cell=''.join(combo),r=xr,J=xJ,g=xg,D=xD,
                     n_swapped=sum(c=='B' for c in combo),net=float(netu.mean())))
  for name,kw in [('gate2_only',dict(gate2_s='B',WW_s='A')),('weights_only',dict(gate2_s='A',WW_s='B'))]:
   netu=cell(sn,'A',kw['gate2_s'],kw['WW_s'],'A','A',ada)
   swaps.append(dict(pair=f'{ta}_{tb}',cell='J:'+name,r='A',J=name,g='A',D='A',n_swapped=1,net=float(netu.mean())))
  if ada:
   netu=cell(sn,'A','A','A','A','A',ada,alpha_s={'g1':'B','g2':'B'})
   swaps.append(dict(pair=f'{ta}_{tb}',cell='alpha_only',r='A',J='alpha',g='alpha',D='A',n_swapped=1,net=float(netu.mean())))
  for tag,key in [('A',ta),('B',tb)]:
   corner=cell(sn,tag,tag,tag,tag,tag,ada)
   ck['corner']=max(ck['corner'],float((corner-snap[key]['netu']).abs().max()))
   off=cell(sn,'B' if tag=='A' else 'A',tag,tag,tag,tag,ada)
   ck['corner_mutctl']=min(ck['corner_mutctl'],float((off-snap[key]['netu']).abs().max()))
 final=[q.detach().clone() for q in p]+([v.clone() for v in act.V] if ada else [])
 return rows,swaps,ck,final
def run(arm,iv,seed):
 torch.set_num_threads(1);H.setup('cpu');mnist=H.Mnist(torch.device('cpu'))
 prefix=f'{arm}_{iv}_s{seed}';OUT.mkdir(parents=True,exist_ok=True)
 spath=SRC/(prefix+'_states.pt')
 manifest=json.loads((ROOT/'results/boundary_tasks20_40_0909/backup_manifest.json').read_text())
 want={r['path']:r['sha256'] for r in manifest['files']}
 assert G.sha(spath)==want[f'source/{prefix}_states.pt'],'source states sha mismatch'
 saved=torch.load(spath,weights_only=False,map_location='cpu')
 dense=np.load(SRC/(prefix+'.npz'))['dense1']
 px=mnist.test_x[saved['probe_indices']];mu=saved['mu_mean']
 allrows=[];allswaps=[];checks=[];start=time.monotonic()
 for j,raw in enumerate(saved['boundaries']):
  assert raw['task']==21+j
  pre=(seed==0 and j==0)
  rows,swaps,ck,final=boundary(raw,arm,iv,mnist,px,dense[j],mu,pre)
  if pre:                                  # mutation controls, run once per arm/seed0
   _,_,_,plain=boundary(raw,arm,iv,mnist,px,dense[j],mu,False)
   ck['diag_noninvasive']=max(float((a-b).abs().max()) for a,b in zip(final,plain))
   mut='diag_update' if arm=='SNA' else 'diag_param'
   _,_,_,bad=boundary(raw,arm,iv,mnist,px,dense[j],mu,False,mutate=mut)
   ck['diag_noninvasive_mutctl']=max(float((a-b).abs().max()) for a,b in zip(plain,bad))
   ck['replay_mutctl']=boundary(raw,arm,iv,mnist,px,dense[j],mu,False,mutate='replay')[2]['replay']
   assert ck['delta']<1e-7 and ck['delta_mutctl']>1e-5,ck
   assert ck['additive']<1e-12 and ck['additive_mutctl']>1e-9,ck
   assert ck['corner']<1e-12 and ck['corner_mutctl']>1e-12,ck
   assert ck['diag_noninvasive']==0 and ck['diag_noninvasive_mutctl']>0,ck
   assert ck['replay_mutctl']>1e-5,ck
  assert ck['replay']<=2e-5,ck
  assert ck['ud']<=1e-12 and ck['ud_mutctl']>1e-12,ck
  for row in rows:row.update(arm=arm,iv=iv,seed=seed,task=raw['task'])
  for row in swaps:row.update(arm=arm,iv=iv,seed=seed,task=raw['task'])
  allrows+=rows;allswaps+=swaps;checks.append(dict(task=raw['task'],**ck))
  assert time.monotonic()-start<300,'runtime cap'
 G.B.csvwrite(OUT/(prefix+'_steps.csv'),allrows)
 G.B.csvwrite(OUT/(prefix+'_swaps.csv'),allswaps)
 (OUT/(prefix+'_provenance.json')).write_text(json.dumps(dict(arm=arm,iv=iv,seed=seed,
  checks=checks,steps=STEPS,diag_n=DN,blocks=NBLOCK,pairs=PAIRS,
  source_sha256=G.sha(spath),code_sha256=G.sha(Path(__file__)),
  spec_sha256=G.sha(ROOT/'specs/spec_ce_factor_swap_0909.md'),host_sha256=G.sha(Path(H.__file__)),
  data_sha256=mnist.sha256,wall_seconds=time.monotonic()-start,
  scope='CPU replay of boundary_tasks20_40_0909 source, not the original CUDA run'),indent=2))
 print('FINISHED',prefix,round(time.monotonic()-start,1),'s',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--arm');ap.add_argument('--iv');ap.add_argument('--seed',type=int)
 ap.add_argument('--all',action='store_true');a=ap.parse_args()
 if a.all:
  def job(j):subprocess.run([sys.executable,'-m','src.ce_factor_swap_0909','--arm',j[0],'--iv',j[1],'--seed',str(j[2])],check=True)
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   for _ in pool.map(job,[(arm,iv,s) for arm in ['LR','SNA'] for iv in ['none','l2'] for s in range(3)]):pass
 else:run(a.arm,a.iv,a.seed)

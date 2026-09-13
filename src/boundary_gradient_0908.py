"""Early boundary gradients: exact CPU continuation and additive Adam bookkeeping."""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import torch
from src import boundary_groups_0908 as B
H=B.H
ROOT=B.ROOT
OUT=ROOT/'results/boundary_gradient_0908'
SOURCE=ROOT/'results/boundary_groups_0908'
CHECK_STEPS=[1,5,10,20,30,50,100]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clone_state(raw,arm):
 p=[q.clone().requires_grad_(True) for q in raw['before']['state']['params']]
 act=H.AdaptiveSnake(.6,.01,'cpu') if arm=='SNA' else H.ARMS[arm]
 if arm=='SNA':act.V=[v.clone() for v in raw['before']['state']['V']]
 m,v,t=raw['adam_before']
 return p,act,([x.clone() for x in m],[x.clone() for x in v],[t])
def proj(pair,mu):
 return (pair[0].double()@mu+pair[1].double()).detach().numpy()
def boundary(raw,arm,iv,mnist,px,py,mu_global,ref,preflight=False):
 start=time.monotonic()
 p,act,adam=clone_state(raw,arm)
 gd=torch.Generator();gd.set_state(raw['rng_after_perm']['data'])
 gb=torch.Generator();gb.set_state(raw['rng_after_perm']['batch'])
 idx=H.stratified_draw(mnist,gd);order=torch.randperm(H.TASK_EXAMPLES,generator=gb)
 xs=mnist.train_x[idx][:,raw['perm']][order];ys=mnist.train_y[idx][order]
 xp=px[:,raw['perm']];mu=xp.double().mean(0)
 masks=raw['masks'];records={};examples={};cr=[]
 mx={'replay_z':0.,'adam_addition':0.,'region_gradient':0.,'mean_cov':0.,'instrument_gradient':0.,'z_identity':0.}
 for step in range(1,101):
  xb=xs[(step-1)*16:step*16];yb=ys[(step-1)*16:step*16]
  out=H.forward(p,xb,act);loss=torch.nn.functional.cross_entropy(out[4],yb)
  gg=torch.autograd.grad(loss,p+[out[0],out[1]])
  g=list(gg[:6]);dz,da=gg[6:]
  if preflight and step==1:
   check=torch.autograd.grad(torch.nn.functional.cross_entropy(H.forward(p,xb,act)[4],yb),p)
   mx['instrument_gradient']=max(float((a-b).abs().max()) for a,b in zip(g,check))
   assert mx['instrument_gradient']==0,mx
  with torch.no_grad():
   before=[q.clone() for q in p[:2]]
   meanold=proj(before,mu)
   # Numerator decomposition uses the actual denominator including the full current gradient.
   reg=[2*.001*q if iv=='l2' else torch.zeros_like(q) for q in p]
   full=[gr+r for gr,r in zip(g,reg)]
   m,v,tc=adam;tc[0]+=1;c1=1-.9**tc[0];c2=1-.999**tc[0]
   comps={k:[] for k in ['history','data','l2','positive','negative']}
   for k,(q,gr,mi,vi) in enumerate(zip(p,full,m,v)):
    hist=mi.clone()*.9
    mi.mul_(.9).add_(gr,alpha=1-.9);vi.mul_(.999).addcmul_(gr,gr,value=1-.999)
    denom=(vi/c2).sqrt()+1e-8
    if k<2:
     coeff=-.001/c1/denom.double()
     comps['history'].append(coeff*hist.double())
     comps['data'].append(coeff*((1-.9)*g[k].double()))
     comps['l2'].append(coeff*((1-.9)*reg[k].double()))
     posdz=dz*(out[0]>0);negdz=dz*(out[0]<=0)
     gp=posdz.T@xb if k==0 else posdz.sum(0)
     gn=negdz.T@xb if k==0 else negdz.sum(0)
     mx['region_gradient']=max(mx['region_gradient'],float((gp+gn-g[k]).abs().max()))
     comps['positive'].append(coeff*((1-.9)*gp.double()))
     comps['negative'].append(coeff*((1-.9)*gn.double()))
    q-=.001*(mi/c1)/denom
   actual=[p[k].double()-before[k].double() for k in range(2)]
   vals={'actual':proj(actual,mu),'W':B.ar(actual[0]@mu),'b':B.ar(actual[1]),
     'star':B.ar(actual[0].sum(1)*mu_global+actual[1]),
     'sgd_data':proj([-.001*q for q in g[:2]],mu),
     'sgd_l2':proj([-.001*q for q in reg[:2]],mu),
     'gW_norm':B.ar(g[0].norm(dim=1)),'gb':B.ar(g[1]),
     'positive_fraction':B.ar((out[0]>0).float().mean(0)),
     'upstream_rms':B.ar((da*16).square().mean(0).sqrt()),
     'local_delta_rms':B.ar((dz*16).square().mean(0).sqrt())}
   gate=act.dphi(out[0],0) if arm=='SNA' else act.dphi(out[0])
   vals['gate_mean']=B.ar(gate.mean(0))
   vals['gate_rms']=B.ar(gate.square().mean(0).sqrt())
   vals['alignment']=vals['actual']-vals['star']
   for key,pair in comps.items():
    vals[key]=proj(pair,mu)
    vals[key+'_W']=B.ar(pair[0]@mu);vals[key+'_b']=B.ar(pair[1])
   mx['adam_addition']=max(mx['adam_addition'],float(abs(vals['actual']-vals['history']-vals['data']-vals['l2']).max()))
   vals['rounding']=vals['actual']-vals['history']-vals['data']-vals['l2']
   meanpart=g[1].double()*(1+mu.square().sum())
   covpart=(dz.double().T@(xb.double()-mu))@mu
   vals['sgd_mean']=-.001*B.ar(meanpart);vals['sgd_cov']=-.001*B.ar(covpart)
   mx['mean_cov']=max(mx['mean_cov'],float(abs(vals['sgd_data']-vals['sgd_mean']-vals['sgd_cov']).max()))
   vals['z']=proj(p[:2],mu)
   mx['z_identity']=max(mx['z_identity'],float(abs(vals['z']-meanold-vals['actual']).max()))
   if arm=='SNA':act.update(out[0],out[2])
   zprobe=B.zbar(p,xp,act)[0]
   mx['replay_z']=max(mx['replay_z'],float(abs(zprobe-ref[step-1]).max()))
   for key,value in vals.items():records.setdefault(key,[]).append(value)
   if step<=50:
    for key,value in [('z',out[0]),('dL_da',da),('dL_dz',dz),('gate',gate)]:
     examples.setdefault(key,[]).append(B.ar(value))
   if step in CHECK_STEPS and not preflight:
    after=B.obs(p,act,xp,mu_global)
    cc,coal,ac0,ac1,ld=B.credits(raw['before'],after,masks,py,arm)
    row={'task':raw['task'],'step':step,'CE_before':coal[0],'CE_after':coal[-1],
      'CE_improvement':coal[0]-coal[-1],'acc_before':ac0,'acc_after':ac1}
    for key,c in zip(B.GROUPS+['downstream'],cc):row['credit_'+key]=float(c)
    cr.append(row)
  assert np.isfinite(vals['actual']).all()
 assert mx['replay_z']<=2e-5,mx
 for key in ['adam_addition','region_gradient','mean_cov','z_identity']:assert mx[key]<=2e-6,mx
 return {k:np.stack(v) for k,v in records.items()},{k:np.stack(v) for k,v in examples.items()},cr,mx
def run(arm,iv,seed,mnist,preflight):
 prefix=f'{arm}_{iv}_s{seed}'
 src=SOURCE/(prefix+'_states.pt')
 saved=torch.load(src,weights_only=False,map_location='cpu')
 px=mnist.test_x[saved['probe_indices']];py=mnist.test_y[saved['probe_indices']]
 ref=np.load(SOURCE/(prefix+'.npz'))
 allrec={};allex={};credits=[];checks=[];masks=[]
 raws=saved['boundaries'][:1 if preflight else 19]
 start=time.monotonic()
 for j,raw in enumerate(raws):
  rec,ex,cc,mx=boundary(raw,arm,iv,mnist,px,py,saved['mu_mean'],ref['dense1'][j],preflight)
  for k,v in rec.items():allrec.setdefault(k,[]).append(v)
  for k,v in ex.items():allex.setdefault(k,[]).append(v)
  for row in cc:row.update(arm=arm,iv=iv,seed=seed)
  credits+=cc;checks.append(dict(task=raw['task'],**mx));masks.append(raw['masks'])
  if (j+1)%5==0:print(prefix,j+1,'boundaries',round(time.monotonic()-start,1),'s',flush=True)
  assert time.monotonic()-start<900,'runtime cap'
 if preflight:return dict(arm=arm,iv=iv,seed=seed,checks=checks)
 rawdir=OUT/'raw';rawdir.mkdir(parents=True,exist_ok=True)
 arrays={k:np.stack(v) for k,v in allrec.items()}
 arrays.update({'example_'+k:np.stack(v) for k,v in allex.items()})
 arrays.update(task=np.array([r['task'] for r in raws]),masks=np.stack(masks))
 dest=rawdir/(prefix+'.npz')
 np.savez_compressed(dest,**arrays)
 B.csvwrite(OUT/(prefix+'_credits.csv'),credits)
 meta={'arm':arm,'iv':iv,'seed':seed,'checks':checks,'source_sha256':sha(src),
       'source_file':str(src),'raw_sha256':sha(dest),'code_sha256':sha(Path(__file__)),
       'spec_sha256':sha(ROOT/'specs/spec_boundary_gradient_0908.md'),
       'host_sha256':sha(Path(H.__file__)),'data_sha256':mnist.sha256,
       'wall_seconds':time.monotonic()-start,'device':'cpu','torch':torch.__version__,
       'scope':'exact replay of boundary_groups CPU, not original CUDA'}
 (OUT/(prefix+'_provenance.json')).write_text(json.dumps(meta,indent=2))
 print('FINISHED',prefix,round(meta['wall_seconds'],1),flush=True)
 return meta
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');ap.add_argument('--arm');a=ap.parse_args()
 torch.set_num_threads(1);H.setup('cpu');mnist=H.Mnist(torch.device('cpu'));OUT.mkdir(parents=True,exist_ok=True)
 if a.preflight:
  checks=[run(arm,iv,0,mnist,True) for arm in ['SNA','LR'] for iv in ['none','l2']]
  (OUT/'preflight.json').write_text(json.dumps(checks,indent=2));print('PREFLIGHT PASS',flush=True)
 else:
  for iv in ['none','l2']:
   for seed in range(3):run(a.arm,iv,seed,mnist,False)
if __name__=='__main__':main()

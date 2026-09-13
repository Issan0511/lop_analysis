"""Paired local derivative versus actual transport, and source-conditioned Adam moments."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from src import boundary_gradient_0908 as G
B,H=G.B,G.H
ROOT=G.ROOT
OUT=ROOT/'results/boundary_transport_0908'
COMP=['pos','neg','l2','pre']
def cpu(t):return t.detach().numpy().copy()
def evaluate(pair,x):return x.double()@pair[0].double().T+pair[1].double()
def append(rec,key,value):rec.setdefault(key,[]).append(cpu(value) if torch.is_tensor(value) else value)
def masked_perunit(a,mask):
 n=mask.sum(0)
 return torch.where(n>0,(a*mask).sum(0)/n.clamp(min=1),torch.full_like(n,float('nan'),dtype=torch.float64))
def varstats(z):
 means=z.mean(0);within=z.var(0,unbiased=False)
 between=means.var(unbiased=False);total=z.var(unbiased=False)
 return means,within,between,total
def boundary(raw,arm,iv,mnist,px,reference,preflight):
 p,act,adam=G.clone_state(raw,arm)
 gd=torch.Generator();gd.set_state(raw['rng_after_perm']['data'])
 gb=torch.Generator();gb.set_state(raw['rng_after_perm']['batch'])
 idx=H.stratified_draw(mnist,gd);order=torch.randperm(H.TASK_EXAMPLES,generator=gb)
 xs=mnist.train_x[idx][:,raw['perm']][order];ys=mnist.train_y[idx][order]
 xp=px[:,raw['perm']];xp64=xp.double()
 m0=[x.double().clone() for x in adam[0][:2]]
 source={key:[torch.zeros_like(q,dtype=torch.float64) for q in p[:2]] for key in ['pos','neg','l2']}
 records={};pairs={};receiver={};stats=[];rates=[]
 z0=evaluate(p[:2],xp).detach()
 zprev=z0
 fixed=[z0>0,z0<=0]
 counts=torch.stack([m.sum(0) for m in fixed])
 checks=dict(replay_z=0.,additive_transport=0.,gradient_split=0.,variance_identity=0.,total_variance=0.,
             instrument_gradient=0.,local_own_mismatch=0)
 initial_mean,initial_var,initial_between,initial_total=varstats(z0)
 append(records,'mean',initial_mean);append(records,'within_var',initial_var)
 stats.append(dict(step=0,mean=float(initial_mean.mean()),within=float(initial_var.mean()),
                   between=float(initial_between),total=float(initial_total),within_flow=0.,between_flow=0.))
 for step in range(1,21):
  xb=xs[(step-1)*16:step*16];yb=ys[(step-1)*16:step*16]
  out=H.forward(p,xb,act);loss=torch.nn.functional.cross_entropy(out[4],yb)
  gg=torch.autograd.grad(loss,p+[out[0]]);g=gg[:6];delta=gg[6]
  if preflight and step==1:
   refg=torch.autograd.grad(torch.nn.functional.cross_entropy(H.forward(p,xb,act)[4],yb),p)
   checks['instrument_gradient']=max(float((a-b).abs().max()) for a,b in zip(refg,g))
   assert checks['instrument_gradient']==0
  with torch.no_grad():
   before=[q.clone() for q in p[:2]]
   posdelta=delta*(out[0]>0);negdelta=delta*(out[0]<=0)
   region={'pos':[posdelta.T@xb,posdelta.sum(0)],
           'neg':[negdelta.T@xb,negdelta.sum(0)]}
   for k in range(2):
    checks['gradient_split']=max(checks['gradient_split'],float((region['pos'][k]+region['neg'][k]-g[k]).abs().max()))
   reg=[2*.001*q if iv=='l2' else torch.zeros_like(q) for q in p]
   grads=[a+b for a,b in zip(g,reg)]
   m,v,tc=adam;tc[0]+=1;c1=1-.9**tc[0];c2=1-.999**tc[0]
   increments={key:[] for key in COMP+['current']}
   coefficients=[]
   for k,(q,gr,mi,vi) in enumerate(zip(p,grads,m,v)):
    mi.mul_(.9).add_(gr,alpha=1-.9)
    vi.mul_(.999).addcmul_(gr,gr,value=1-.999)
    denom=(vi/c2).sqrt()+1e-8
    if k<2:
     coefficient=-.001/c1/denom.double();coefficients.append(coefficient)
     for key in ['pos','neg','l2']:
      r=reg[k] if key=='l2' else region[key][k]
      source[key][k].mul_(.9).add_(r.double(),alpha=1-.9)
      increments[key].append(coefficient*source[key][k])
     increments['pre'].append(coefficient*(.9**step)*m0[k])
     increments['current'].append(coefficient*((1-.9)*g[k].double()))
    q-=.001*(mi/c1)/denom
   actual=[p[k].double()-before[k].double() for k in range(2)]
   dzbatch=evaluate(actual,xb)
   curbatch=evaluate(increments['current'],xb)
   own=delta.double()*(1-.9)*(xb.double().square()@coefficients[0].T+coefficients[1])
   local=-16*delta.double()
   mismatch=((local*own)<0)&(local.abs()>1e-8)&(own.abs()>1e-8)
   checks['local_own_mismatch']+=int(mismatch.sum())
   append(pairs,'z',out[0]);append(pairs,'local_direction',local)
   append(pairs,'actual',dzbatch);append(pairs,'current',curbatch)
   append(pairs,'own',own);append(pairs,'others',curbatch-own)
   for key in COMP:append(pairs,key,evaluate(increments[key],xb))
   # Counts kept per boundary and step; aggregate ratios at boundary level, never pool seeds.
   zz=out[0].detach();valid=(local.abs()>1e-8)&(dzbatch.abs()>1e-8)
   negup=(zz<=0)&(local>1e-8)
   row=dict(step=step,n_negative=int((zz<=0).sum()),n_neg_local_up=int(negup.sum()),
    n_neg_local_up_actual_down=int((negup&(dzbatch<-1e-8)).sum()),
    n_neg_local_up_current_down=int((negup&(curbatch<-1e-8)).sum()),
    n_valid=int(valid.sum()),n_opposed=int((valid&(local*dzbatch<0)).sum()))
   rates.append(row)
   # Same fixed held-out images before/after each update, evaluated linearly in double.
   dzprobe=evaluate(actual,xp)
   zafter=evaluate(p[:2],xp)
   vals={key:evaluate(increments[key],xp) for key in COMP}
   residual=dzprobe-sum(vals.values())
   checks['additive_transport']=max(checks['additive_transport'],float(residual.abs().max()))
   vals['actual']=dzprobe;vals['rounding']=residual
   for key,val in vals.items():
    append(records,key,val.mean(0))
    fixedmeans=torch.stack([masked_perunit(val,mask) for mask in fixed])
    dynamic=torch.stack([masked_perunit(val,zprev>0),masked_perunit(val,zprev<=0)])
    append(receiver,key+'_fixed',fixedmeans)
    append(receiver,key+'_dynamic',dynamic)
   oldmeans,oldvar,oldbetween,_=varstats(zprev)
   newmeans,newvar,newbetween,newtotal=varstats(zafter)
   dmean=dzprobe.mean(0);dvar=dzprobe.var(0,unbiased=False)
   covariance=((zprev-oldmeans)*(dzprobe-dmean)).mean(0)
   withinflow=2*covariance+dvar
   bcov=((oldmeans-oldmeans.mean())*(dmean-dmean.mean())).mean()
   betweenflow=2*bcov+dmean.var(unbiased=False)
   checks['variance_identity']=max(checks['variance_identity'],
    float((newvar-oldvar-withinflow).abs().max()),float(abs(newbetween-oldbetween-betweenflow)))
   checks['total_variance']=max(checks['total_variance'],float(abs(newtotal-newvar.mean()-newbetween)))
   append(records,'within_flow_cov',2*covariance);append(records,'within_flow_square',dvar)
   append(records,'mean',newmeans);append(records,'within_var',newvar)
   stats.append(dict(step=step,mean=float(newmeans.mean()),within=float(newvar.mean()),
    between=float(newbetween),total=float(newtotal),within_flow=float(withinflow.mean()),
    between_flow=float(betweenflow)))
   zprev=zafter
   if arm=='SNA':act.update(out[0],out[2])
   refz=B.zbar(p,xp,act)[0]
   checks['replay_z']=max(checks['replay_z'],float(abs(refz-reference[step-1]).max()))
   assert torch.isfinite(zafter).all()
 assert checks['replay_z']<=2e-5,checks
 assert checks['additive_transport']<=2e-5,checks
 assert checks['gradient_split']<=2e-6,checks
 assert checks['variance_identity']<=1e-9 and checks['total_variance']<=1e-9,checks
 assert checks['local_own_mismatch']==0,checks
 return ({k:np.stack(v) for k,v in records.items()},
 {k:np.stack(v) for k,v in pairs.items()},
 {k:np.stack(v) for k,v in receiver.items()},stats,rates,checks,
 {'probe_z0':cpu(z0).astype(np.float32),'probe_z20':cpu(zprev).astype(np.float32),'receiver_counts':cpu(counts)})
def run(arm,iv,seed,mnist,preflight=False):
 prefix=f'{arm}_{iv}_s{seed}';path=G.SOURCE/(prefix+'_states.pt')
 saved=torch.load(path,weights_only=False,map_location='cpu')
 ref=np.load(G.SOURCE/(prefix+'.npz'))
 px=mnist.test_x[saved['probe_indices']]
 allrec={};stats=[];rates=[];checks=[]
 start=time.monotonic()
 for j,raw in enumerate(saved['boundaries'][:1 if preflight else 19]):
  rec,pair,receiver,st,rr,ck,end=boundary(raw,arm,iv,mnist,px,ref['dense1'][j],preflight)
  for pref,dd in [('',rec),('pair_',pair),('receiver_',receiver),('',end)]:
   for key,value in dd.items():append(allrec,pref+key,value)
  for rows,target in [(st,stats),(rr,rates)]:
   for row in rows:row.update(arm=arm,iv=iv,seed=seed,task=raw['task'])
   target.extend(rows)
  checks.append(dict(task=raw['task'],**ck))
  assert time.monotonic()-start<900,'per arm-seed runtime cap'
  if (j+1)%5==0:print(prefix,j+1,round(time.monotonic()-start,1),flush=True)
 if preflight:return dict(arm=arm,iv=iv,seed=seed,checks=checks)
 dest=OUT/'raw'/(prefix+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(dest,**{k:np.stack(v) for k,v in allrec.items()})
 B.csvwrite(OUT/(prefix+'_variance.csv'),stats);B.csvwrite(OUT/(prefix+'_paired_counts.csv'),rates)
 meta={'arm':arm,'iv':iv,'seed':seed,'checks':checks,'source_file':str(path),'source_sha256':G.sha(path),
 'code_sha256':G.sha(Path(__file__)),'spec_sha256':G.sha(ROOT/'specs/spec_boundary_transport_0908.md'),
 'raw_sha256':G.sha(dest),'data_sha256':mnist.sha256,'wall_seconds':time.monotonic()-start,
 'device':'cpu','scope':'same CPU trajectory, not original CUDA'}
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
   for seed in range(3):run(a.arm,iv,seed,mnist)
if __name__=='__main__':main()

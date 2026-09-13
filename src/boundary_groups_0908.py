"""Boundary group audit. Frozen host; CPU reruns are not assumed identical to CUDA."""
from pathlib import Path
import argparse,csv,hashlib,itertools,json,math,sys,time
import numpy as np
import torch
torch.set_num_threads(1)
from src import pmnist_boundary_host_0908 as H
ROOT=Path(__file__).resolve().parents[1]
ORIG=Path('/home/issan/Projects/claude/proj_004_drift')
H.DATA_DIR=ORIG/'data/mnist'
OUT=ROOT/'results/boundary_groups_0908'
GROUPS=['D_only','N_only','both','rest']
def ar(t):return t.detach().cpu().numpy().copy()
def partition(end,jump):
 strip=jump-end
 di=np.flatnonzero(strip < -1e-6);di=di[np.argsort(strip[di],kind='stable')[:25]]
 ni=np.argsort(abs(jump),kind='stable')[:25]
 D=np.zeros(100,dtype=bool);N=D.copy();D[di]=True;N[ni]=True
 masks=np.stack([D&~N,N&~D,D&N,~(D|N)])
 assert np.all(masks.sum(0)==1)
 return masks
def zbar(params,x,act):
 z1,_,z2,_,_=H.forward(params,x,act)
 return ar(z1.mean(0)),ar(z2.mean(0))
def star(params,mu):
 return ar(params[0].sum(1)*mu+params[1])
def state(params,act):
 return {'params':[q.detach().clone() for q in params],
         'V':[q.clone() for q in act.V] if isinstance(act,H.AdaptiveSnake) else [],
         'alpha':[act.alpha(i).clone() for i in range(2)] if isinstance(act,H.AdaptiveSnake) else []}
@torch.no_grad()
def obs(params,act,x,mu):
 z,a,z2,a2,logits=H.forward(params,x,act)
 gate=act.dphi(z,0) if isinstance(act,H.AdaptiveSnake) else act.dphi(z)
 return {'z':ar(z.mean(0)),'star':star(params,mu),'rowpart':ar(params[0].sum(1)*mu),'bias':ar(params[1]),
         'gate':ar(gate.abs().mean(0)),'a1':a.clone(),'state':state(params,act),'logits':ar(logits)}
@torch.no_grad()
def credits(before,after,masks,labels,arm):
 # All endpoints/counterfactuals evaluated with exactly the same float64 downstream arithmetic.
 aa=before['a1'].double();ab=after['a1'].double()
 vals=[];logits=[]
 for bits in range(32):
  a=aa.clone()
  for j,mask in enumerate(masks):
   if bits&(1<<j):a[:,mask]=ab[:,mask]
  st=after['state'] if bits&16 else before['state'];p=[v.double() for v in st['params']]
  z=a@p[2].T+p[3]
  if arm=='SNA':
   alpha=st['alpha'][1].double();a2=z+torch.sin(alpha*z)**2/alpha
  else:a2=torch.where(z>0,z,.1*z)
  l=a2@p[4].T+p[5];vals.append(float(torch.nn.functional.cross_entropy(l,labels)))
  logits.append(l)
 credit=np.zeros(5)
 for j in range(5):
  for bits in range(32):
   if bits&(1<<j):continue
   size=bits.bit_count();weight=math.factorial(size)*math.factorial(4-size)/math.factorial(5)
   credit[j]+=weight*(vals[bits]-vals[bits|(1<<j)])
 assert abs(credit.sum()-(vals[0]-vals[31]))<1e-9
 return credit,vals,float((logits[0].argmax(1)==labels).double().mean()),float((logits[-1].argmax(1)==labels).double().mean()),float((logits[-1]-logits[0]).square().mean().sqrt())
def csvwrite(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def run(arm,seed,iv,mnist,tasks=120,first=100,audit=True,persist=True):
 act=H.AdaptiveSnake(.6,.01,'cpu') if arm=='SNA' else H.ARMS[arm]
 params=H.init_params(seed,torch.device('cpu'))
 adam=([torch.zeros_like(q) for q in params],[torch.zeros_like(q) for q in params],[0])
 gp,gd,gb=H.stream('perm',seed),H.stream('data',seed),H.stream('batch',seed)
 pidx=torch.randperm(len(mnist.test_x),generator=H.stream('boundary_probe',seed))[:512]
 px=mnist.test_x[pidx];py=mnist.test_y[pidx];mu=float(mnist.train_x.mean())
 rec={k:[] for k in ['task','end1','end2','jump1','jump2','star1','star2','dense1','dense2','acc']}
 per=[];position=[];raw=[];oldperm=None;start=time.monotonic()
 for task in range(1,tasks+1):
  if time.monotonic()-start>900:raise RuntimeError('900s limit exceeded; trajectory incomplete')
  dense=task>first and oldperm is not None
  if dense:
   with torch.no_grad():e1,e2=zbar(params,px[:,oldperm],act)
  perm=torch.randperm(784,generator=gp)
  if dense:
   with torch.no_grad():
    j1,j2=zbar(params,px[:,perm],act);ss=star(params,mu);s2=ar(params[2].sum(1)+params[3])
   for k,val in zip(['task','end1','end2','jump1','jump2','star1','star2'],[task,e1,e2,j1,j2,ss,s2]):rec[k].append(val)
   d1=[];d2=[]
   if audit:
    masks=partition(e1,j1);before=obs(params,act,px[:,perm],mu)
    # Record complete state + moments and streams sufficient for continuation of this boundary.
    boundary_raw={'task':task,'masks':masks,'end':e1,'before':before,'perm':perm.clone(),
      'adam_before':([q.clone() for q in adam[0]],[q.clone() for q in adam[1]],adam[2][0]),
      'rng_after_perm':{'perm':gp.get_state(),'data':gd.get_state(),'batch':gb.get_state()}}
  idx=H.stratified_draw(mnist,gd);order=torch.randperm(H.TASK_EXAMPLES,generator=gb)
  xs=mnist.train_x[idx][:,perm][order];ys=mnist.train_y[idx][order]
  for step in range(625):
   xb=xs[step*16:(step+1)*16];yb=ys[step*16:(step+1)*16]
   out=H.forward(params,xb,act);loss=torch.nn.functional.cross_entropy(out[4],yb);grads=torch.autograd.grad(loss,params)
   with torch.no_grad():
    if iv!='none':grads=[gr+2*.001*q for q,gr in zip(params,grads)]
    m,v,tc=adam;tc[0]+=1
    b1,b2,eps=.9,.999,1e-8;c1,c2=1-b1**tc[0],1-b2**tc[0]
    for q,gr,mi,vi in zip(params,grads,m,v):
     mi.mul_(b1).add_(gr,alpha=1-b1);vi.mul_(b2).addcmul_(gr,gr,value=1-b2)
     q-=.001*(mi/c1)/((vi/c2).sqrt()+eps)
    if isinstance(act,H.AdaptiveSnake):act.update(out[0],out[2])
   if dense and step<300:
    with torch.no_grad():z1,z2=zbar(params,px[:,perm],act)
    d1.append(z1);d2.append(z2)
   if dense and audit and step+1 in [300,625]:
    after=obs(params,act,px[:,perm],mu)
    cc,coal,ac0,ac1,logitdiff=credits(before,after,masks,py,arm)
    row={'arm':arm,'iv':iv,'seed':seed,'task':task,'step':step+1,'CE_before':coal[0],'CE_after':coal[-1],
         'CE_improvement':coal[0]-coal[-1],'acc_before':ac0,'acc_after':ac1,'logit_change_rms':logitdiff}
    for label,c in zip(GROUPS+['downstream'],cc):row['credit_'+label]=float(c)
    per.append(row)
    D=masks[0]|masks[2];drop=j1-e1;refit=after['z']-j1
    denom=float((drop[D]**2).sum())
    pos={'arm':arm,'iv':iv,'seed':seed,'task':task,'step':step+1,'n_D':int(D.sum()),
         'recovery':float((refit[D]*(-drop[D])).sum()/denom) if denom>1e-10 else None,
         'leftover':float(np.sqrt(((after['z'][D]-e1[D])**2).sum()/denom)) if denom>1e-10 else None,
         'returned_fraction':float((abs(after['z'][D]-e1[D])<=.1*abs(drop[D])).mean()) if denom>1e-10 else None}
    for label,mask in zip(GROUPS,masks):
     ds=after['star']-before['star'];dw=after['rowpart']-before['rowpart'];db=after['bias']-before['bias']
     assert np.allclose(ds,dw+db,rtol=1e-4,atol=3e-6)
     pos[label+'_n']=int(mask.sum())
     pos[label+'_star_sum']=float(ds[mask].sum())
     pos[label+'_star_negative_sum']=float(np.minimum(ds[mask],0).sum())
     pos[label+'_star_mean']=float(ds[mask].mean()) if mask.any() else None
     pos[label+'_row_sum']=float(dw[mask].sum());pos[label+'_bias_sum']=float(db[mask].sum())
     pos[label+'_gate_before']=float(before['gate'][mask].mean()) if mask.any() else None
     pos[label+'_near_abs_zmax']=float(abs(j1[mask]).max()) if mask.any() else None
    pos['star_all_mean']=float((after['star']-before['star']).mean());position.append(pos)
    boundary_raw['after_'+str(step+1)]=after
    boundary_raw['coalition_CE_'+str(step+1)]=coal
  if dense:
   rec['dense1'].append(np.stack(d1));rec['dense2'].append(np.stack(d2))
   with torch.no_grad():rec['acc'].append(float((H.forward(params,mnist.test_x[:,perm],act)[4].argmax(1)==mnist.test_y).float().mean()))
   if audit:
    boundary_raw['adam_after']=([q.clone() for q in adam[0]],[q.clone() for q in adam[1]],adam[2][0])
    raw.append(boundary_raw)
  oldperm=perm
  if persist and task%20==0:print(arm,iv,seed,'task',task,'elapsed',round(time.monotonic()-start,1),flush=True)
  if persist and task in [1,100]:
   torch.save({'task':task,'state':state(params,act),'adam':adam,'perm':perm,
               'streams':{k:g.get_state() for k,g in zip(['perm','data','batch'],[gp,gd,gb])}},OUT/(arm+'_'+iv+'_s'+str(seed)+'_task'+str(task)+'.pt'))
  assert all(torch.isfinite(q).all() for q in params)
 rec={k:np.stack(v) for k,v in rec.items()}
 if persist:
  prefix=arm+'_'+iv+'_s'+str(seed)
  np.savez_compressed(OUT/(prefix+'.npz'),**rec)
  csvwrite(OUT/(prefix+'_credits.csv'),per);csvwrite(OUT/(prefix+'_positions.csv'),position)
  torch.save({'boundaries':raw,'probe_indices':pidx,'mu_mean':mu},OUT/(prefix+'_states.pt'))
  with np.load(ORIG/'results/pmnist_boundary_0908'/(arm+'_'+('none' if iv=='none' else 'l2-1e-3')+'_s'+str(seed)+'.npz')) as old:
   cmp={k:float(np.max(abs(rec[k]-old[k]))) for k in rec}
  meta={'arm':arm,'iv':iv,'seed':seed,'device':'cpu','threads':1,'source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'host_sha256':hashlib.sha256(Path(H.__file__).read_bytes()).hexdigest(),'original_max_abs_errors':cmp,
        'original_exact':all(v==0 for v in cmp.values()),'data_sha256':mnist.sha256,'wall_seconds':time.monotonic()-start,
        'spec_sha256':hashlib.sha256((ROOT/'specs/spec_boundary_groups_0908.md').read_bytes()).hexdigest()}
  (OUT/(prefix+'_provenance.json')).write_text(json.dumps(meta,indent=2))
  print('FINISHED',prefix,meta['wall_seconds'],'original_exact',meta['original_exact'],flush=True)
 return rec
def preflight(mnist):
 # Original CPU training with exactly the same activation and RNG streams.
 sys.modules['src.pmnist_0905']=H
 sys.path.insert(0,str(ORIG/'src'));import pmnist_boundary_0908 as original
 checks=[]
 for arm in ['SNA','LR']:
  for iv in ['none','l2']:
   ours=run(arm,0,iv,mnist,tasks=3,first=1,audit=True,persist=False)
   ref=original.run(arm,0,.001,3,1,300,mnist,torch.device('cpu'),'adam','none' if iv=='none' else 'l2:1e-3',.6,.01)
   errors={k:float(np.max(abs(ours[k]-ref[k]))) for k in ours}
   assert all(v==0 for v in errors.values()),errors
   checks.append({'arm':arm,'iv':iv,'CPU_source_exact':True,'errors':errors})
 (OUT/'preflight.json').write_text(json.dumps(checks,indent=2));print('PREFLIGHT PASS',flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm');ap.add_argument('--iv');ap.add_argument('--preflight',action='store_true');a=ap.parse_args()
 OUT.mkdir(parents=True,exist_ok=True);H.setup('cpu');mnist=H.Mnist(torch.device('cpu'))
 if a.preflight:preflight(mnist);return
 for seed in [0,1,2]:
  assert not (OUT/(a.arm+'_'+a.iv+'_s'+str(seed)+'_provenance.json')).exists()
  run(a.arm,seed,a.iv,mnist)
if __name__=='__main__':main()

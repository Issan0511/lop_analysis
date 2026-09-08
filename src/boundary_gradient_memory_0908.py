"""Supplemental pre-boundary first-moment numerator contribution, not a reset intervention."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from src import boundary_gradient_0908 as G
B,H=G.B,G.H
def run(arm,iv,seed,mnist):
 prefix=f'{arm}_{iv}_s{seed}'
 saved=torch.load(G.SOURCE/(prefix+'_states.pt'),weights_only=False,map_location='cpu')
 base=np.load(G.OUT/'raw'/(prefix+'.npz'))
 px=mnist.test_x[saved['probe_indices']]
 ref=np.load(G.SOURCE/(prefix+'.npz'))
 allW=[];allb=[];error=0.
 for j,raw in enumerate(saved['boundaries'][:19]):
  p,act,adam=G.clone_state(raw,arm);mzero=[m.double().clone() for m in adam[0][:2]]
  gd=torch.Generator();gd.set_state(raw['rng_after_perm']['data'])
  gb=torch.Generator();gb.set_state(raw['rng_after_perm']['batch'])
  idx=H.stratified_draw(mnist,gd);order=torch.randperm(H.TASK_EXAMPLES,generator=gb)
  xs=mnist.train_x[idx][:,raw['perm']][order];ys=mnist.train_y[idx][order]
  xp=px[:,raw['perm']];mu=xp.double().mean(0);ww=[];bb=[]
  for step in range(1,101):
   out=H.forward(p,xs[(step-1)*16:step*16],act)
   grads=torch.autograd.grad(torch.nn.functional.cross_entropy(out[4],ys[(step-1)*16:step*16]),p)
   with torch.no_grad():
    full=[gr+2*.001*q for gr,q in zip(grads,p)] if iv=='l2' else grads
    m,v,tc=adam;tc[0]+=1;c1=1-.9**tc[0];c2=1-.999**tc[0]
    for k,(q,gr,mi,vi) in enumerate(zip(p,full,m,v)):
     mi.mul_(.9).add_(gr,alpha=1-.9);vi.mul_(.999).addcmul_(gr,gr,value=1-.999)
     denom=(vi/c2).sqrt()+1e-8
     if k<2:
      contribution=(-.001/c1/denom.double())*(.9**step*mzero[k])
      if k==0:ww.append(B.ar(contribution@mu))
      else:bb.append(B.ar(contribution))
     q-=.001*(mi/c1)/denom
    if arm=='SNA':act.update(out[0],out[2])
    error=max(error,float(abs(B.zbar(p,xp,act)[0]-ref['dense1'][j,step-1]).max()))
  allW.append(np.stack(ww));allb.append(np.stack(bb))
 assert error<=2e-5,error
 dest=G.OUT/'raw'/(prefix+'_memory.npz')
 np.savez_compressed(dest,pre_history_W=np.stack(allW),pre_history_b=np.stack(allb))
 meta={'prefix':prefix,'replay_maxabs':error,'raw_sha256':G.sha(dest),'source_sha256':G.sha(G.SOURCE/(prefix+'_states.pt')),
       'code_sha256':G.sha(Path(__file__)),'spec_sha256':G.sha(G.ROOT/'specs/spec_boundary_gradient_memory_0908.md'),
       'status':'posthoc supplement, fixed before supplemental execution'}
 (G.OUT/(prefix+'_memory_provenance.json')).write_text(json.dumps(meta,indent=2))
 print('MEMORY FINISHED',prefix,error,flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',required=True);a=ap.parse_args()
 torch.set_num_threads(1);H.setup('cpu');mnist=H.Mnist(torch.device('cpu'))
 for iv in ['none','l2']:
  for seed in range(3):run(a.arm,iv,seed,mnist)
if __name__=='__main__':main()

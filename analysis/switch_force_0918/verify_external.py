"""Independent autograd of switched states and uninstrumented trajectory replay."""
import json
import numpy as np
import torch
import run as r

def main():
    bitwise=[]
    for arm in r.ARMS:
        for step in r.STEPS:
            cp=r.load(arm,step); stored=np.load(r.RAW/f'{arm}_{step}_frozen.npz')
            flip=cp['env']['flip_state']; xo,yo=r.support(cp,flip,torch.float64)
            for j in range(15):
                nf=flip.clone();nf[:,j]=1-nf[:,j]
                xn,yn=r.support(cp,nf,torch.float64); mu=xn.mean(0)
                for mode,(x,y) in enumerate([(xo,yo),(xn,yo),(xo,yn),(xn,yn)]):
                    n=r.new_net(cp,torch.float64,32)
                    xx=x.reshape(320,20); yy=y.reshape(320)
                    w,b,v,c=[q.detach().clone().requires_grad_() for q in [n.W,n.b,n.v,n.c]]
                    z=torch.einsum('rhd,rd->rh',w,xx)+b
                    a=torch.where(z>0,z,n.act_alpha*z)
                    f=(a*v).sum(-1)+c
                    ag=torch.autograd.grad(((f-yy)**2).sum(),[w,b,v,c])
                    pre,aa,ff=n.forward(xx); gg=n.grads(xx,pre,aa,ff-yy)
                    for key,g,q in zip(['W','b','v','c'],gg,ag): r.check('switched_autograd_'+key,g,q)
                    old=n.W.clone(); n.sgd_step(torch.full((320,),r.LR,dtype=torch.float64),*gg)
                    displacement=(n.W-old).reshape(32,10,100,20)
                    observed=torch.einsum('srhd,rd->srh',displacement,mu).mean(0)
                    r.check('frozen_actual_force',observed,stored['force'][j,mode])
            for name in ['float64','float32']:
                dtype=getattr(torch,name)
                d=np.load(r.RAW/f'{arm}_{step}_{name}_trajectory.npz')
                gen=torch.Generator().manual_seed(1)
                env=r.SCREnv(10,20,15,torch.full((10,),10000),gen,'cpu')
                env.load_state(cp['env']);env.gen=torch.Generator().manual_seed(20260918)
                n=r.new_net(cp,dtype,2); lrs=torch.full((20,),r.LR,dtype=dtype)
                oldflip=cp['env']['flip_state'].clone()
                times=d['time'].tolist(); snap={t:i for i,t in enumerate(times)}
                for t in range(10000):
                    raw_new=env.step(); raw_old=raw_new.clone();raw_old[:,:15]=oldflip
                    ix=(raw_new[:,15:].long()*torch.tensor([16,8,4,2,1])).sum(-1)
                    if not np.array_equal(ix.numpy(),d['support_index'][t]):raise AssertionError('original RNG stream mismatch')
                    if t==0:
                        xo,yo=r.support(cp,oldflip,dtype);xn,yn=r.support(cp,env.flip_state,dtype)
                        xall=torch.cat([xn,xo],1);yall=torch.cat([yn,yo],1)
                    ids=torch.arange(20);ii=ix.repeat(2)
                    x=xall[ii,ids];y=yall[ii,ids]
                    pre,a,f=n.forward(x);gg=n.grads(x,pre,a,f-y)
                    n.sgd_step(lrs,*gg)
                    if t+1 in snap:
                        it=snap[t+1]
                        for key,value in [('W',n.W),('b',n.b),('v',n.v),('cout',n.c)]:
                            same=np.array_equal(value.double().numpy(),d[key][it])
                            bitwise.append(dict(arm=arm,step=step,dtype=name,t=t+1,param=key,equal=same))
                            if not same:raise AssertionError('instrumentation changed trajectory')
                print(arm,step,name,'original-env replay bitwise match',flush=True)
    result=dict(checks=r.CHECKS,all_pass=all(v['failed']==0 for v in r.CHECKS.values()),bitwise_comparisons=len(bitwise),all_bitwise_equal=all(v['equal'] for v in bitwise))
    (r.OUT/'external_verification.json').write_text(json.dumps(result,indent=2))
    r.writecsv(r.OUT/'bitwise_replay.csv',bitwise)
    print(json.dumps(result),flush=True)
    if not result['all_pass']:raise SystemExit(1)

if __name__=='__main__':main()

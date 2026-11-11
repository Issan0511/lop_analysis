#!/usr/bin/env python3
"""Independent algebra checks, not learning-trajectory evidence."""
import json,sys
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(1)
def step(g,m,s,t,eps):
    m=.9*m+.1*g;s=.999*s+.001*g*g;t=t+1
    return -.001*(m/(1-.9**t))/(np.sqrt(s/(1-.999**t))+eps)

def main(out):
    rng=np.random.default_rng(900);g=rng.normal(size=20);m=rng.normal(size=20);s=rng.uniform(.1,2,size=20)
    base=step(g,m,s,71,1e-8);c=.01
    scaled=step(c*g,c*m,c*c*s,71,c*1e-8)
    stale=step(c*g,m,s,71,1e-8)
    err=float(np.max(np.abs(base-scaled)));assert err<1e-15
    examples=[dict(a=a,v=1.,c=c,increase_demand=c/(1+np.exp(c*a))) for a in [.1,10.] for c in [1.,.1]]
    torch.manual_seed(900)
    X=torch.randn(5,3,dtype=torch.float64);W=torch.randn(4,3,dtype=torch.float64,requires_grad=True)
    beta=torch.randn(4,dtype=torch.float64,requires_grad=True);V=torch.randn(3,4,dtype=torch.float64,requires_grad=True);b=torch.randn(3,dtype=torch.float64,requires_grad=True)
    z=X@W.T+beta;a=torch.nn.functional.elu(z);ell=a@V.T+b;y=torch.tensor([0,1,2,0,2]);e=ell.softmax(1)-torch.nn.functional.one_hot(y,3)
    loss=(ell.logsumexp(1)-ell.gather(1,y[:,None]).flatten()).mean();pars=[W,beta,V,b]
    grads=torch.autograd.grad(loss,pars,retain_graph=True)
    direct=[]
    for r in range(5):
        row=[]
        for k in range(3):
            J=torch.autograd.grad(ell[r,k],pars,retain_graph=True)
            row.append(-sum((j*g).sum() for j,g in zip(J,grads)))
        direct.append(torch.stack(row))
    direct=torch.stack(direct)
    G=torch.where(z>0,torch.ones_like(z),z.exp());K=X@X.T+1
    head=-(a@a.T+1)@e/len(X)
    body=torch.zeros_like(e)
    for i in range(4):
        body-=((K*G[:,i,None]*G[None,:,i])@(e@V[:,i]))[:,None]*V[:,i][None,:]/len(X)
    kernel_error=float((direct-head-body).abs().max());assert kernel_error<1e-12
    res=dict(tier='deterministic formula checks; not empirical mechanism evidence',consistent_adam_scaling_error=err,
        stale_history_changes_step_max=float(np.max(np.abs(base-stale))),
        stale_large_second_moment_example=dict(gradient=1e-6,m=0,s=1,step=10000,
            retained=float(step(1e-6,0.,1.,10000,1e-8)),fresh=float(step(1e-6,0.,0.,0,1e-8))),
        binary_examples=examples,sgd_two_channel_kernel_error=kernel_error)
    Path(out).write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))

if __name__=='__main__':main(sys.argv[1])

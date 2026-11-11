#!/usr/bin/env python3
"""Demand-weighted full-bank tangent diagnostics via verified matrix-free Lanczos."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'1')
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src import leaky_boundary_1111 as B
C=B.C;torch.set_num_threads(1)

class Tangent:
    def __init__(self,ps,x,a,diag):
        self.ps=[p.detach().double().requires_grad_(True) for p in ps];self.x=x.double();self.a=a;self.diag=[d.double() for d in diag]
        self.net=C.Net(self.ps,a);self.out=self.net.forward(self.x);self.f=self.out[2]-self.out[2].mean(1,keepdim=True)
        self.n,self.nc=self.f.shape;self.dim=self.n*(self.nc-1);self.trace=self.get_trace()
    def __call__(self,v):
        v=v.reshape(self.n,self.nc);gr=torch.autograd.grad(self.f,self.ps,grad_outputs=v,retain_graph=True)
        dp=[g*d for g,d in zip(gr,self.diag)];out=C.jvp(self.net,self.x,self.out,dp)
        return ((out-out.mean(1,keepdim=True))/self.n).flatten().detach()
    @torch.no_grad()
    def get_trace(self):
        zs,hs,_=self.out;g1=C.gate(zs[0],self.a);g2=C.gate(zs[1],self.a);w2=self.ps[2];w3=self.ps[4]-self.ps[4].mean(0,keepdim=True)
        d=self.diag;v1=self.x.square()@d[0].T+d[1];v2=hs[0].square()@d[2].T+d[3]
        paths=torch.einsum('oj,nj,jk->nok',w3,g2,w2)
        tr1=(paths.square().sum(1)*g1.square()*v1).sum()
        tr2=(w3.square().sum(0)*g2.square()*v2).sum()
        tr3=(1-1/self.nc)*(hs[1].square()@d[4].T+d[5]).sum()
        return float((tr1+tr2+tr3)/self.n)

def lanczos(op,start,m=128):
    v=start.flatten().double();v=v/v.norm();n=len(v);Q=torch.empty(n,min(m,n),dtype=torch.double);alphas=[];betas=[];prev=torch.zeros_like(v);beta=0.
    for j in range(min(m,n)):
        Q[:,j]=v;z=op(v);alpha=float(torch.dot(v,z));z=z-alpha*v-beta*prev
        for _ in range(2):z-=Q[:,:j+1]@(Q[:,:j+1].T@z)
        bn=float(z.norm());alphas.append(alpha)
        if bn<1e-12 or j==min(m,n)-1:break
        betas.append(bn);prev=v;v=z/bn;beta=bn
    return np.array(alphas),np.array(betas)
def quadrature(alpha,beta,m):
    m=min(m,len(alpha));T=np.diag(alpha[:m]);off=beta[:m-1]
    if m>1:T+=np.diag(off,1)+np.diag(off,-1)
    ev,vec=np.linalg.eigh(T);assert ev.min()>-1e-8*max(ev.max(),1),(ev.min(),ev.max())
    ev=ev.clip(0);w=vec[0]**2;metrics={}
    for t in [1,10,100,1000,10000]:metrics[f'remaining_t{t}']=float(np.sqrt(np.dot(w,np.exp(-2*t*ev))))
    for r in [.01,.0001,.000001]:metrics[f'ridge_residual_{r:g}']=float(np.sqrt(np.dot(w,(r/(ev+r))**2)))
    return metrics,ev,w

def checks(out):
    torch.manual_seed(127);n=5;x=torch.randn(n,4,dtype=torch.double)
    ps=[torch.randn(3,4,dtype=torch.double)*.2,torch.randn(3,dtype=torch.double)*.1,torch.randn(2,3,dtype=torch.double)*.2,torch.randn(2,dtype=torch.double)*.1,torch.randn(3,2,dtype=torch.double)*.2,torch.randn(3,dtype=torch.double)*.1]
    errors=[]
    for kind in ['euclid','diagonal']:
        diag=[torch.ones_like(p) if kind=='euclid' else torch.rand_like(p)+.1 for p in ps];op=Tangent(ps,x,B.A,diag)
        J=torch.stack([torch.cat([g.flatten() for g in torch.autograd.grad(v,op.ps,retain_graph=True)]) for v in op.f.flatten()])
        dd=torch.cat([d.flatten() for d in diag]);K=(J*dd)@J.T/n
        traceerr=abs(op.trace-float(K.trace()));assert traceerr<1e-10
        vec=torch.randn(n,3,dtype=torch.double);vec-=vec.mean(1,keepdim=True);v=vec.flatten();err=float((op(v)-K@v).abs().max());assert err<1e-10
        scale=op.trace/op.dim;aa,bb=lanczos(lambda v:op(v)/scale,v,128);metrics,_,_=quadrature(aa,bb,128)
        ev,vs=torch.linalg.eigh(K/scale);weights=(vs.T@(v/v.norm())).square().numpy();ev=ev.numpy().clip(0)
        qerr=max(abs(metrics[f'remaining_t{t}']-float(np.sqrt(np.dot(weights,np.exp(-2*t*ev))))) for t in [1,10,100,1000,10000]);assert qerr<1e-8
        errors.append(dict(kind=kind,operator_maxabs=err,trace_abs=traceerr,quadrature_abs=qerr))
    B.put(Path(out)/'spectral_checks.json',dict(all_pass=True,errors=errors,code_sha256=B.sha(__file__)));print('spectral checks pass',errors)

def run(args):
    dr=Path(args.out);dr.mkdir(parents=True,exist_ok=True);t0=time.time();ck=torch.load(args.source,weights_only=False,map_location='cpu');x=C.get_x(args.seed);y=B.probe_labels(args.seed)
    assert ck['seed']==args.seed;R=F.one_hot(y,10).double()-.1;rows=[]
    for mode in ['euclid','adam_frozen']:
        if mode=='adam_frozen' and ck['tc']==0:continue
        diag=[torch.ones_like(p) for p in ck['params']] if mode=='euclid' else [1/((v.double()/(1-.999**ck['tc'])).sqrt()+C.EPS) for v in ck['v']]
        op=Tangent(ck['params'],x,B.A,diag);scale=op.trace/op.dim
        aa,bb=lanczos(lambda v:op(v)/scale,R,128);small,_,_=quadrature(aa,bb,64);big,ev,w=quadrature(aa,bb,128)
        native=C.Net(ck['params'],B.A).forward(x)[0];flips=[int(((z>0)!=(zd>0)).sum()) for z,zd in zip(native,op.out[0])]
        row=dict(seed=args.seed,age=ck['age'],mode=mode,trace=op.trace,mean_eigenvalue=scale,lanczos_steps=len(aa),
                 task_rayleigh=float(torch.dot(R.flatten(),op(R.flatten()))/R.square().sum()),L1_f32_f64_gate_difference=flips[0],L2_f32_f64_gate_difference=flips[1],**big)
        for key in big:row[key+'_64_128_absdiff']=abs(big[key]-small[key]);row[key+'_converged']=abs(big[key]-small[key])<=.03
        rows.append(row);np.savez_compressed(dr/f'{mode}.npz',alpha=aa,beta=bb,ritz=ev,weights=w)
    B.write(dr/'spectrum.csv',rows)
    B.put(dr/'provenance.json',dict(source_sha256=B.sha(args.source),source=args.source,code_sha256=B.sha(__file__),training_code_sha256=B.sha(ROOT/'src/leaky_boundary_1111.py'),spec_sha256=B.sha(B.SPEC),git_hash=B.ident(args)['git_hash']))
    B.put(dr/'done.json',dict(seconds=time.time()-t0));print('spectrum',args.seed,ck['age'],'seconds',time.time()-t0,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['checks','run']);ap.add_argument('--source');ap.add_argument('--seed',type=int,default=100);ap.add_argument('--out',required=True);a=ap.parse_args()
    if a.mode=='checks':checks(a.out)
    else:run(a)
if __name__=='__main__':main()

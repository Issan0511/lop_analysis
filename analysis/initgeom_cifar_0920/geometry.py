"""Host-initialized float64 geometry with population variance and sign enclosures."""
import math
import numpy as np
import torch
from src import relu_doors_0919 as E
from analysis.initgeom_cifar_0920.stats import Q99,CONDITIONS,probability
U=2.**-53

def gamma(n):return n*U/(1-n*U)

def geometry(H,rank=True,rank_reference=None):
    assert H.ndim==2 and H.dtype==torch.float64 and torch.isfinite(H).all()
    n,d=H.shape;mu=H.mean(0);center=H-mu
    tr=float(center.square().sum()/n);norm=float(mu.norm())
    mue=gamma(n+2)*H.abs().mean(0)
    ce=mue[None,:]+U/(1-U)*(H.abs()+mu.abs()[None,:])
    tre=float(((2*center.abs()*ce+ce.square()).sum()/n)+gamma(n*d+2)*center.square().sum()/n)
    norme=float(mue.norm()+gamma(d+4)*mu.abs().norm())
    status='UNDEFINED_INPUT' if tr==0 and norm==0 else 'DEGENERATE_INPUT' if tr==0 else 'FINITE'
    r=None if status=='UNDEFINED_INPUT' else math.inf if tr==0 else norm/math.sqrt(tr)
    # These interval endpoints concern arithmetic error, not sampling variation.
    rlo=0 if r is None else max(0,norm-norme)/math.sqrt(tr+tre) if tr+tre>0 else math.inf
    rhi=math.inf if tr<=tre else (norm+norme)/math.sqrt(tr-tre)
    p=probability(r);plo=probability(rlo) if r is not None else None;phi=probability(rhi) if r is not None else None
    erank=None
    if rank and tr>0:
        gram=(center@center.T)/n if n<d else (center.T@center)/n
        tr2=float(gram.square().sum());erank=tr*tr/tr2 if tr2 else None
    elif rank_reference is not None:erank=rank_reference
    rb=None if tr==0 else math.sqrt(norm*norm+1)/math.sqrt(tr)
    return dict(mu=mu,mu_error=mue,mu_norm=norm,mu_norm_error=norme,trSigma=tr,trSigma_error=tre,r=r,r_lower=rlo,r_upper=rhi,p=p,p_numeric_lower=plo,p_numeric_upper=phi,r_bias=rb,p_bias=probability(rb),p_rounded233=probability(r,2.33),rms=float(H.square().mean().sqrt()),effective_rank=erank,status=status)

def transformed(raw32,condition):
    assert condition in CONDITIONS or condition=='gamma100'
    x=raw32.double();mu=x.mean(0)
    if condition in ('raw','gamma100'):return x,torch.zeros_like(x)
    if condition=='std':return E.standardize(raw32).double(),torch.zeros_like(x)
    g={'C':0.,'gamma025':.25,'gamma050':.5,'gamma075':.75,'gamma150':1.5,'gamma200':2.}[condition]
    centered=x-mu;y=centered+g*mu if g else centered
    # Input arithmetic relative to the specified ideal centered/dial transform.
    muerr=gamma(x.shape[0]+2)*x.abs().mean(0)
    err=abs(g-1)*muerr[None,:]+gamma(4)*(x.abs()+(1+g)*mu.abs()[None,:])
    return y,err.expand_as(y)

def affine(H,W,b,input_error=None):
    assert H.dtype==W.dtype==b.dtype==torch.float64
    z=H@W.T+b
    error=gamma(2*H.shape[1]+3)*(H.abs()@W.abs().T+b.abs())
    if input_error is not None:error+=input_error@W.abs().T
    error+=torch.finfo(torch.float64).tiny
    return z,error

def signs(z,error):
    assert z.shape==error.shape and z.ndim==2 and (error>=0).all() and torch.isfinite(z).all() and torch.isfinite(error).all()
    n,units=z.shape;threshold=math.floor(.99*n)+1
    positive=(z>0).sum(0);negative=(z<0).sum(0);zero=(z==0).sum(0)
    pos_sure=(z-error>0).sum(0);neg_sure=(z+error<0).sum(0)
    pos_possible=(z+error>0).sum(0);neg_possible=(z-error<0).sum(0)
    lower=(pos_sure>=threshold)|(neg_sure>=threshold)
    upper=(pos_possible>=threshold)|(neg_possible>=threshold)
    actual=(positive>=threshold)|(negative>=threshold)
    return dict(positive=positive,negative=negative,zero=zero,positive_sure=pos_sure,negative_sure=neg_sure,positive_possible=pos_possible,negative_possible=neg_possible,one_sided=actual,guaranteed=lower,possible=upper,f=float(actual.double().mean()),f_lower=float(lower.double().mean()),f_upper=float(upper.double().mean()),positive_units=int((positive>=threshold).sum()),negative_units=int((negative>=threshold).sum()),all_zero_units=int((zero==n).sum()),old_min_fraction=float((torch.minimum(positive,negative).double()/n<.01).double().mean()),uncertain_values=int(((z-error<=0)&(z+error>=0)).sum()),threshold=threshold,units=units,images=n)

def diagnostic(H,W,b,z,ze,g):
    s=signs(z,ze);mu=H.mean(0);center=H-mu
    mean=W@mu+b;sd=(center@W.T).square().mean(0).sqrt()
    proxy=mean.abs()>Q99*sd
    ratio=torch.where(sd>0,b.abs()/sd,torch.nan)
    s.update(zmean=z.mean(0),zsd=z.std(0,unbiased=False),projection_mean=mean,projection_sd=sd,bias_over_sd=ratio,zero_projection_sd=int((sd==0).sum()),gaussian_proxy=proxy,gaussian_proxy_fraction=float(proxy.double().mean()),input_stats=g)
    return s

def serial(x):
    if isinstance(x,torch.Tensor):return serial(x.detach().cpu().numpy())
    if isinstance(x,np.ndarray):return serial(x.tolist())
    if isinstance(x,np.generic):return serial(x.item())
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return 'inf' if x==math.inf else '-inf' if x==-math.inf else None
    return x

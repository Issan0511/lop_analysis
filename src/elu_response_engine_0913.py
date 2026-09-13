#!/usr/bin/env python3
"""Response-anchor ELU engine with unchanged manual Adam dynamics."""
from __future__ import annotations
import torch
import elu_environment_0913 as base

BRANCHES=("A","B","C","D","AF","BF")
MODELS=[dict(seed=s,env="RL",act="ELU1",iv=b,branch=b) for s in range(3) for b in BRANCHES]

def _fk(models,device,dtype):
    f=[];k=[];up=[]
    for m in models:
        b=m.get("branch",m.get("iv"))
        if b not in BRANCHES:raise ValueError(f"unknown response-anchor branch {b!r}")
        f.append(b in ("C","D"));k.append(b in ("B","D","BF"));up.append(b in ("AF","BF"))
    return (torch.tensor(f,device=device,dtype=torch.bool)[:,None,None],
            torch.tensor(k,device=device,dtype=torch.bool)[:,None,None],
            torch.tensor(up,device=device,dtype=torch.bool),)

def anchored_forward(p,x,anchor0,anchor1,delta,F,K):
    """Forward pass; cached anchor activations and delta are frozen constants."""
    z1=torch.bmm(x,p[0].transpose(1,2))+p[1][:,None,:]
    a1=base.activ(z1,torch.ones_like(F))
    z2=torch.bmm(a1,p[2].transpose(1,2))+p[3][:,None,:]
    d=delta[:,None,:]
    live0=base.activ(z2,torch.ones_like(F));live1=base.activ(z2+d,torch.ones_like(F))
    # Stable exact diagonals; differences only for the off-diagonal cells.
    b=anchor0+(live1-anchor1)
    c=anchor1+(live0-anchor0)
    cell=torch.where(F,torch.where(K,live1,c),torch.where(K,b,live0))
    a2=torch.where(d.ne(0),cell,live0)
    z3=torch.bmm(a2,p[4].transpose(1,2))+p[5][:,None,:]
    return z1,a1,z2,a2,z3

def anchored_gradients(p,x,y,anchor0,anchor1,delta,F,K):
    z1,a1,z2,a2,z3=anchored_forward(p,x,anchor0,anchor1,delta,F,K)
    g3=(z3.softmax(-1)-y)/x.shape[1]
    gate2=base.gate(z2+torch.where(K,delta[:,None,:],torch.zeros_like(delta[:,None,:])),torch.ones_like(F))
    g2=torch.bmm(g3,p[4])*gate2
    g1=torch.bmm(g2,p[2])*base.gate(z1,torch.ones_like(F))
    gr=[torch.bmm(g1.transpose(1,2),x),g1.sum(1),torch.bmm(g2.transpose(1,2),a1),g2.sum(1),torch.bmm(g3.transpose(1,2),a2),g3.sum(1)]
    ce=(z3.logsumexp(-1)-(z3*y).sum(-1)).mean(-1)
    acc=(z3.argmax(-1)==y.argmax(-1)).float().mean(-1)
    return gr,ce,acc

class Engine(base.Engine):
    def __init__(self,models=MODELS,device="cuda"):
        super().__init__(models=models,device=device)
        self.F,self.K,self.upstream_frozen=_fk(models,device,self.p[0].dtype)
        self.anchor_z2=torch.zeros(self.M,base.N,100,device=device)
        self.anchor_base=torch.zeros_like(self.anchor_z2)
        self.anchor_shift=torch.zeros_like(self.anchor_z2)
        self.delta=torch.zeros(self.M,100,device=device)
        self.freeze_masks=[torch.zeros_like(p,dtype=torch.bool) for p in self.p]
        self.frozen_reference=[p.clone() for p in self.p]
        self.online_ce=torch.full((self.M,base.STEPS),float("nan"),device=device)
        self.online_acc=torch.full_like(self.online_ce,float("nan"))
        self.online_pos=torch.zeros((),dtype=torch.long,device=device)

    @torch.no_grad()
    def set_anchors(self,z20,delta):
        z20=torch.as_tensor(z20,device=self.device,dtype=self.p[0].dtype)
        delta=torch.as_tensor(delta,device=self.device,dtype=self.p[0].dtype)
        if z20.shape!=self.anchor_z2.shape:raise ValueError(f"z20 {z20.shape} != {self.anchor_z2.shape}")
        if delta.shape!=self.delta.shape:raise ValueError(f"delta {delta.shape} != {self.delta.shape}")
        if not torch.isfinite(z20).all() or not torch.isfinite(delta).all():raise ValueError("anchors must be finite")
        self.anchor_z2.copy_(z20);self.delta.copy_(delta)
        self.anchor_base.copy_(base.activ(z20,torch.ones_like(self.F)))
        self.anchor_shift.copy_(base.activ(z20+delta[:,None,:],torch.ones_like(self.F)))
        for m in self.freeze_masks:m.zero_()
        self.freeze_masks[0][self.upstream_frozen]=True
        self.freeze_masks[1][self.upstream_frozen]=True
        for ref,p in zip(self.frozen_reference,self.p):ref.copy_(p)

    @torch.no_grad()
    def reset_online(self):
        self.online_ce.fill_(float("nan"));self.online_acc.fill_(float("nan"));self.online_pos.zero_()

    def forward(self,x,anchor0,anchor1):return anchored_forward(self.p,x,anchor0,anchor1,self.delta,self.F,self.K)
    def gradients(self,x,y,anchor0,anchor1):return anchored_gradients(self.p,x,y,anchor0,anchor1,self.delta,self.F,self.K)

    def forward_indexed(self,indices):
        indices=torch.as_tensor(indices,dtype=torch.long,device=self.device)
        return self.forward(self.cx[self.mid,indices],self.anchor_base[self.mid,indices],self.anchor_shift[self.mid,indices])

    def forward_full(self):return self.forward(self.cx,self.anchor_base,self.anchor_shift)

    @torch.no_grad()
    def step(self,x,y,anchor0,anchor1):
        gr,ce,acc=self.gradients(x,y,anchor0,anchor1)
        pos=self.online_pos.reshape(1,1).expand(self.M,1)
        self.online_ce.scatter_(1,pos,ce[:,None]);self.online_acc.scatter_(1,pos,acc[:,None]);self.online_pos.add_(1)
        self.ce.add_(ce);self.acc.add_(acc);self.t.add_(1)
        c1=1-torch.pow(.9,self.t);c2=1-torch.pow(.999,self.t)
        for p,g,m,v,mask,ref in zip(self.p,gr,self.m,self.v,self.freeze_masks,self.frozen_reference):
            m.mul_(.9).add_(g,alpha=.1);v.mul_(.999).addcmul_(g,g,value=.001)
            p.addcdiv_(m/c1,(v/c2).sqrt()+1e-8,value=-.001)
            p.copy_(torch.where(mask,ref,p))
        return ce,acc

    @torch.no_grad()
    def block(self):
        for k in range(base.BLOCK):
            ix=self.indices[k]
            self.step(self.cx[self.mid,ix],self.cy[self.mid,ix],self.anchor_base[self.mid,ix],self.anchor_shift[self.mid,ix])

    def snapshot(self):
        q=self.p+self.m+self.v+[self.t,self.acc,self.ce,self.anchor_z2,self.anchor_base,self.anchor_shift,self.delta,self.online_ce,self.online_acc,self.online_pos]+self.freeze_masks+self.frozen_reference
        return [x.clone() for x in q]

    @torch.no_grad()
    def restore(self,s):
        q=self.p+self.m+self.v+[self.t,self.acc,self.ce,self.anchor_z2,self.anchor_base,self.anchor_shift,self.delta,self.online_ce,self.online_acc,self.online_pos]+self.freeze_masks+self.frozen_reference
        if len(s)!=len(q):raise ValueError(f"snapshot length {len(s)} != {len(q)}")
        for x,y in zip(q,s):x.copy_(y)

    @torch.no_grad()
    def diagnose_step(self,x,y,anchor0,anchor1):
        gr,ce,acc=self.gradients(x,y,anchor0,anchor1);gr=[q.clone() for q in gr]
        tn=self.t+1;c1=1-torch.pow(.9,tn);c2=1-torch.pow(.999,tn)
        current=[.1*g/c1 for g in gr];history=[.9*m/c1 for m in self.m]
        mn=[];vn=[]
        for m,v,g in zip(self.m,self.v,gr):
            mm=m.clone();mm.mul_(.9).add_(g,alpha=.1);mn.append(mm)
            vv=v.clone();vv.mul_(.999).addcmul_(g,g,value=.001);vn.append(vv)
        denominator=[(vnew/c2).sqrt()+1e-8 for vnew in vn]
        predicted=[-.001*(mnew/c1)/d for mnew,d in zip(mn,denominator)]
        candidate=[]
        for p,mnew,den in zip(self.p,mn,denominator):
            q=p.clone();q.addcdiv_(mnew/c1,den,value=-.001);candidate.append(q)
        realized=[torch.where(mask,ref,q)-p for p,q,mask,ref in zip(self.p,candidate,self.freeze_masks,self.frozen_reference)]
        decomposition_error=[(a+b)-mnew/c1 for a,b,mnew in zip(current,history,mn)]
        return dict(gradients=gr,ce=ce.clone(),acc=acc.clone(),current_gradient_numerator=[q.clone() for q in current],history_numerator=[q.clone() for q in history],decomposition_error=decomposition_error,denominator=[q.clone() for q in denominator],predicted_update=[q.clone() for q in predicted],parameter_delta=realized,moment_m_delta=[a-m for a,m in zip(mn,self.m)],moment_v_delta=[a-v for a,v in zip(vn,self.v)],next_t=tn.clone())

    @torch.no_grad()
    def evaluate(self):
        z1,a1,z2,a2,logits=self.forward_full()
        ce=(logits.logsumexp(-1)-(logits*self.cy).sum(-1)).mean(-1);acc=(logits.argmax(-1)==self.cy.argmax(-1)).float().mean(-1)
        out=dict(train_acc=acc.cpu().numpy(),train_ce=ce.cpu().numpy());units={}
        for l,z,garg in ((1,z1,z1),(2,z2,z2+torch.where(self.K,self.delta[:,None,:],torch.zeros_like(self.delta[:,None,:])))):
            g=base.gate(garg,torch.ones_like(self.F))
            for key,val in {"zmean":z.mean(1),"zstd":z.std(1,unbiased=False),"gate_mean":g.mean(1),"gate_rms":g.square().mean(1).sqrt(),"lowgate":(g<.05).float().mean(1),"nearzero":(garg.abs()<1).float().mean(1)}.items():units[f"{key}_l{l}"]=val.cpu().numpy()
        units["cnorm_l1"]=(self.p[0]-self.p[0].mean(-1,keepdim=True)).norm(dim=-1).cpu().numpy();units["rowmean_l1"]=self.p[0].mean(-1).cpu().numpy()
        for l in range(1,4):out[f"wnorm_l{l}"]=self.p[2*(l-1)].norm(dim=-1).mean(-1).cpu().numpy()
        out["cnorm_l1"]=units["cnorm_l1"].mean(-1);out["lowgate_l1"]=units["lowgate_l1"].mean(-1);out["nearzero_l1"]=units["nearzero_l1"].mean(-1)
        return out,units

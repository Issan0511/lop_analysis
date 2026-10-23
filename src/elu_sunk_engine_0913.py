#!/usr/bin/env python3
"""Actual-update freezing engine and synthetic validation."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import torch
import elu_environment_0913 as base

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/elu_sunk_rescue_0913"

class Engine(base.Engine):
    """Base engine without projection, freezing selected values after Adam."""
    def __init__(self,models=base.MODELS,device="cuda"):
        super().__init__(models=models,device=device)
        self.freeze_masks=[torch.zeros_like(p,dtype=torch.bool) for p in self.p]
        self.frozen_reference=[p.clone() for p in self.p]

    @torch.no_grad()
    def set_freeze(self,masks):
        if len(masks)!=6: raise ValueError(f"expected 6 masks, got {len(masks)}")
        for i,(dst,src,p,ref) in enumerate(zip(self.freeze_masks,masks,self.p,self.frozen_reference)):
            src=torch.as_tensor(src,dtype=torch.bool,device=p.device)
            if src.shape!=p.shape: raise ValueError(f"mask {i}: {src.shape} != {p.shape}")
            dst.copy_(src);ref.copy_(p)

    @torch.no_grad()
    def step(self,x,y):
        gr,ce,acc=base.gradients(self.p,x,y,self.elu)
        self.ce.add_(ce);self.acc.add_(acc);self.t.add_(1)
        c1=1-torch.pow(.9,self.t);c2=1-torch.pow(.999,self.t)
        for p,g,m,v,mask,ref in zip(self.p,gr,self.m,self.v,self.freeze_masks,self.frozen_reference):
            m.mul_(.9).add_(g,alpha=.1);v.mul_(.999).addcmul_(g,g,value=.001)
            p.addcdiv_(m/c1,(v/c2).sqrt()+1e-8,value=-.001)
            p.copy_(torch.where(mask,ref,p))

    def snapshot(self):
        q=self.p+self.m+self.v+[self.t,self.acc,self.ce]+self.freeze_masks+self.frozen_reference
        return [x.clone() for x in q]

    @torch.no_grad()
    def restore(self,s):
        q=self.p+self.m+self.v+[self.t,self.acc,self.ce]+self.freeze_masks+self.frozen_reference
        if len(s)!=len(q): raise ValueError(f"snapshot length {len(s)} != {len(q)}")
        for x,y in zip(q,s): x.copy_(y)

def maxabs(a,b):
    return max(float(torch.logical_xor(x,y).any()) if x.dtype==torch.bool else float((x-y).abs().max()) for x,y in zip(a,b))

@torch.no_grad()
def validation():
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(1);dev="cuda"
    models=[dict(seed=0,act="ELU1",env="PM",iv="ref"),dict(seed=1,act="LR",env="RL",iv="ref")]
    gen=torch.Generator().manual_seed(9130913)
    x=torch.rand(2,base.BATCH,784,generator=gen).to(dev)
    y=torch.nn.functional.one_hot(torch.randint(10,(2,base.BATCH),generator=gen),10).float().to(dev)
    c={}
    # False masks reproduce original Engine exactly from nonzero optimizer time.
    a=base.Engine(models,dev);b=Engine(models,dev);a.t.fill_(19);b.t.fill_(19)
    for _ in range(4): a.step(x,y);b.step(x,y)
    c["no_freeze_p_maxabs"]=maxabs(a.p,b.p);c["no_freeze_m_maxabs"]=maxabs(a.m,b.m)
    c["no_freeze_v_maxabs"]=maxabs(a.v,b.v);c["no_freeze_t_equal"]=bool(torch.equal(a.t,b.t))
    # Empty-mask CUDA graphs also match from the same nonzero Adam history.
    graph_x=torch.rand(a.cx.shape,generator=gen).to(dev)
    graph_labels=torch.randint(10,(2,base.N),generator=gen)
    graph_y=torch.nn.functional.one_hot(graph_labels,10).float().to(dev)
    graph_indices=torch.randint(base.N,a.indices.shape,generator=gen).to(dev)
    for q in (a,b):
        q.cx.copy_(graph_x);q.cy.copy_(graph_y);q.indices.copy_(graph_indices);q.capture()
    a.graph.replay();b.graph.replay();torch.cuda.synchronize()
    c["empty_graph_p_maxabs"]=maxabs(a.p,b.p)
    c["empty_graph_m_maxabs"]=maxabs(a.m,b.m)
    c["empty_graph_v_maxabs"]=maxabs(a.v,b.v)
    c["empty_graph_t_equal"]=bool(torch.equal(a.t,b.t))
    c["empty_graph_counter_increment"]=float(a.t-23)
    # Build history, intervene, compare with the ordinary Adam step.
    a=base.Engine(models,dev);b=Engine(models,dev)
    for _ in range(3): a.step(x,y);b.step(x,y)
    masks=[torch.zeros_like(p,dtype=torch.bool) for p in b.p]
    masks[0][0,:7,:19]=True;masks[1][0,:7]=True;masks[2][0,:11,:7]=True
    b.set_freeze(masks);refs=[q.clone() for q in b.frozen_reference];a.step(x,y);b.step(x,y)
    c["freeze_p_maxabs"]=max(float((p[z]-r[z]).abs().max()) if z.any() else 0 for p,z,r in zip(b.p,masks,refs))
    c["unfrozen_vs_ordinary_maxabs"]=max(float((p[~z]-q[~z]).abs().max()) for p,z,q in zip(b.p,masks,a.p))
    c["moments_m_vs_ordinary_maxabs"]=maxabs(b.m,a.m);c["moments_v_vs_ordinary_maxabs"]=maxabs(b.v,a.v)
    c["masked_momentum_nonzero"]=float(b.m[0][masks[0]].abs().max())
    # Gradient-only masking still moves a parameter when Adam history is nonzero.
    p=refs[0].clone();m=b.m[0].clone().mul_(.9);v=b.v[0].clone().mul_(.999);t=b.t+1
    p.addcdiv_(m/(1-torch.pow(.9,t)),(v/(1-torch.pow(.999,t))).sqrt()+1e-8,value=-.001)
    c["gradient_only_mask_drift"]=float((p[masks[0]]-refs[0][masks[0]]).abs().max())
    # Active graph replay from nonzero t matches eager, including state tensors.
    g=Engine(models,dev);g.cx.copy_(torch.rand(g.cx.shape,generator=gen).to(dev))
    labels=torch.randint(10,(2,base.N),generator=gen)
    g.cy.copy_(torch.nn.functional.one_hot(labels,10).float().to(dev))
    g.indices.copy_(torch.randint(base.N,g.indices.shape,generator=gen).to(dev));g.t.fill_(73)
    gm=[torch.zeros_like(p,dtype=torch.bool) for p in g.p]
    gm[2][1,20:31,9:17]=True;gm[3][1,20:31]=True;gm[4][1,:,20:31]=True;g.set_freeze(gm)
    g.capture();before=g.snapshot();g.block();eager=g.snapshot();g.restore(before);g.graph.replay();torch.cuda.synchronize()
    replay=g.snapshot();c["graph_vs_eager_snapshot_maxabs"]=maxabs(replay,eager)
    c["graph_counter_increment"]=float(g.t-before[18])
    c["graph_masks_unchanged"]=all(torch.equal(x,y) for x,y in zip(g.freeze_masks,before[21:27]))
    c["graph_references_unchanged"]=all(torch.equal(x,y) for x,y in zip(g.frozen_reference,before[27:33]))
    c["graph_frozen_p_maxabs"]=max(float((p[z]-r[z]).abs().max()) if z.any() else 0 for p,z,r in zip(g.p,g.freeze_masks,g.frozen_reference))
    zeros=["no_freeze_p_maxabs","no_freeze_m_maxabs","no_freeze_v_maxabs","empty_graph_p_maxabs","empty_graph_m_maxabs","empty_graph_v_maxabs","freeze_p_maxabs","unfrozen_vs_ordinary_maxabs","moments_m_vs_ordinary_maxabs","moments_v_vs_ordinary_maxabs","graph_vs_eager_snapshot_maxabs","graph_frozen_p_maxabs"]
    assert all(c[k]==0 for k in zeros),c
    assert c["no_freeze_t_equal"] and c["masked_momentum_nonzero"]>0 and c["gradient_only_mask_drift"]>0,c
    assert c["empty_graph_t_equal"] and c["empty_graph_counter_increment"]==base.BLOCK,c
    assert c["graph_counter_increment"]==base.BLOCK and c["graph_masks_unchanged"] and c["graph_references_unchanged"],c
    c["code_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();c["torch_version"]=torch.__version__
    c["cuda_device"]=torch.cuda.get_device_name();c["status"]="PASS"
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"engine_validation.json").write_text(json.dumps(c,indent=2)+"\n")
    print(json.dumps(c,indent=2))

if __name__=="__main__": validation()

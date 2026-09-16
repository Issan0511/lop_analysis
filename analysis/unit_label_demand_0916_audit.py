"""Independently reconstruct stored per-unit statistics with autograd on real checkpoints."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src import relu_gelu_silu_rl_0914 as B


def audit(raw, out):
    inputs = torch.load(raw/"inputs.pt", weights_only=True)
    stats = np.load(raw/"statistics.npz")
    fields = {name:i for i,name in enumerate(stats["fields"])}
    cases = []
    for task,step in ((2,0),(10,6000)):
        state = torch.load(raw/f"state_t{task:02d}_s{step:04d}.pt", weights_only=True)
        si = np.flatnonzero((stats["tasks"]==task)&(stats["steps"]==step)).item()
        for seed,act in ((0,"GELU"),(2,"SILU")):
            mi = next(i for i,m in enumerate(inputs["models"]) if m["seed"]==seed and m["act"]==act)
            p = [v[mi:mi+1].double().cuda().requires_grad_() for v in state["p"]]
            x = inputs["x"][seed:seed+1].double().cuda()
            old = inputs["labels"][task-2,seed][None].cuda()
            new = inputs["labels"][task-1,seed][None].cuda()
            A = B.Masks([act], "cuda")
            z1,a1,z2,a2,logits = B.forward(p,x,A)
            logp = logits.log_softmax(-1)
            unew = torch.autograd.grad(-logp.gather(-1,new[...,None]).sum(),(a1,a2),retain_graph=True)
            uold = torch.autograd.grad(-logp.gather(-1,old[...,None]).sum(),(a1,a2),retain_graph=True)
            d = torch.autograd.grad((logits.gather(-1,old[...,None])-logits.gather(-1,new[...,None])).sum(),
                                    (a1,a2),retain_graph=True)
            jc = [torch.autograd.grad(logits[...,c].sum(),(a1,a2),retain_graph=True) for c in range(10)]
            worst = 0.
            with torch.no_grad():
                po = logp.exp().gather(-1,old[...,None])
                for layer,(a,z) in enumerate(((a1,z1),(a2,z2))):
                    J = torch.stack([v[layer] for v in jc],dim=2)
                    jo = J.gather(2,old[...,None,None].expand(-1,-1,1,100)).squeeze(2)
                    bound = (1-po)*(J-jo[:,:,None,:]).abs().max(2).values
                    tol = 1e-10*(1+J.abs().max(2).values)
                    active = a.abs()>1e-12
                    cert = active & (a.sign()*d[layer]-bound>tol)
                    ph = B.gate(z,A)
                    zc = -.7517915239 if act=="GELU" else -1.2784645428
                    masks = (z>0,(z<=0)&(z>zc),z<=zc)
                    terms = {"n":torch.ones_like(a),"n_active":active,
                             "n_support":active&(a.sign()*d[layer]>tol),
                             "n_suppress":active&(a.sign()*unew[layer]>tol),"n_cert":cert,
                             "ad":a*d[layer],"abs_ad":(a*d[layer]).abs(),
                             "au":a*unew[layer],"abs_au":(a*unew[layer]).abs(),
                             "cert_abs_au":cert*(a*unew[layer]).abs(),
                             "force":-ph*unew[layer],"force_switch":-ph*d[layer],
                             "force_residual":-ph*uold[layer]}
                    for gi,mask in enumerate(masks):
                        eligible = mask & (old!=new)[...,None]
                        for key,value in terms.items():
                            independent = (eligible*value).sum(1)[0].cpu().numpy()
                            recorded = stats["values"][si,mi,layer,:,gi,fields[key]]
                            error = float(np.max(np.abs(independent-recorded)/(1+np.abs(independent))))
                            worst = max(worst,error)
                    assert worst<1e-9, (task,step,seed,act,worst)
            cases.append(dict(task=task,step=step,seed=seed,act=act,images=1200,layers=2,units_per_layer=100,
                              max_scaled_error=worst))
    result=dict(status="PASS",method="autograd reconstruction of all per-unit P/B/V primary fields",cases=cases)
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    audit(args.raw,args.out)

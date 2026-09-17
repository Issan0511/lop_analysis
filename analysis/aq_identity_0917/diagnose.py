"""Post-result diagnosis; original results and tolerances remain unchanged."""
import json
import subprocess
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import verify as V

OUT = V.ROOT/"results/aq_identity_0917"
STATS = {}
FAILURES = []
FORWARD = defaultdict(float)
DTYPE = torch.float32


def auto(n,x,y,matched):
    w,b,v,c = [p.clone().requires_grad_() for p in (n.W,n.b,n.v,n.c)]
    z = (torch.einsum("rhd,rd->rh",w,x) if matched else (w*x[:,None]).sum(-1))+b
    a = torch.where(z>0,z,n.act_alpha*z)
    if n.act=="leaky_off_p0p5":
        a = a+0.5
    f = (a*v).sum(-1)+c
    g = torch.autograd.grad(((f-y)**2).sum(),(w,b,v,c))
    return g,z.detach(),f.detach()


def compare(name,actual,original,matched,meta):
    a,b,c = [V.double(v) for v in [actual,original,matched]]
    e0,e1 = (a-b).abs(),(a-c).abs()
    t0,t1 = 2e-5*(1+b.abs()),2e-5*(1+c.abs())
    fail0,fail1 = e0>t0,e1>t1
    st = STATS.setdefault(name,dict(check=name,count=0,original_failures=0,matched_failures=0,
                                   original_max_error=0.,matched_max_error=0.))
    st["count"] += a.numel()
    st["original_failures"] += int(fail0.sum())
    st["matched_failures"] += int(fail1.sum())
    st["original_max_error"] = max(st["original_max_error"],float(e0.max()))
    st["matched_max_error"] = max(st["matched_max_error"],float(e1.max()))
    for ix in fail0.nonzero():
        idx = tuple(ix.tolist())
        FAILURES.append(dict(**meta,check=name,index=str(idx),production_or_formula=float(a[idx]),
            original_autograd=float(b[idx]),matched_autograd=float(c[idx]),
            original_error=float(e0[idx]),matched_error=float(e1[idx]),
            original_tolerance=float(t0[idx]),matched_tolerance=float(t1[idx])))


def one(n,x,y,meta):
    z,a,f = n.forward(x)
    grads = n.grads(x,z,a,f-y)
    ag0,z0,f0 = auto(n,x,y,False)
    ag1,z1,f1 = auto(n,x,y,True)
    for label,zz,ff in [("original",z0,f0),("matched",z1,f1)]:
        FORWARD[label+"_max_z_error"] = max(FORWARD[label+"_max_z_error"],float((z-zz).abs().max()))
        FORWARD[label+"_max_output_error"] = max(FORWARD[label+"_max_output_error"],float((f-ff).abs().max()))
        FORWARD[label+"_gate_disagreements"] += int(((z>0)!=(zz>0)).sum())
    for name,p,b,c in zip(["W","b","v","c"],grads,ag0,ag1):
        compare("autograd_"+name,p,b,c,meta)
    k = torch.where(z>0,torch.ones_like(z),torch.full_like(z,n.act_alpha))
    h = 2*V.double(f-y)[:,None]*V.double(n.v)*V.double(k)
    for kind in V.COORDS:
        w,xc = V.coords(n.W,kind),V.coords(x,kind)
        g0,g1 = V.coords(ag0[0],kind),V.coords(ag1[0],kind)
        A = h*(w*xc[:,None]).sum(-1)
        Q = h**2*xc.square().sum(-1)[:,None]
        compare(kind+"/expanded_gradient",h[...,None]*xc[:,None],g0,g1,meta)
        compare(kind+"/A_vs_autograd",A,(w*g0).sum(-1),(w*g1).sum(-1),meta)
        compare(kind+"/Q_vs_autograd",Q,g0.square().sum(-1),g1.square().sum(-1),meta)
    return grads


def main():
    torch.set_num_threads(1)
    provenance = json.loads((OUT/"provenance.json").read_text())
    for source in provenance["input_checkpoints"]:
        cp = torch.load(source["path"],map_location="cpu",weights_only=True)
        x,y = V.support(cp,DTYPE)
        n = V.make_net(cp,DTYPE,repeats=32)
        meta = dict(phase="fixed",arm=cp["arm"],checkpoint=cp["step"],step=-1)
        one(n,x.reshape(-1,20),y.flatten(),meta)
        if cp["step"]==200000:
            n = V.make_net(cp,DTYPE)
            rng = np.random.Generator(np.random.PCG64(20260917))
            lr = torch.full((n.R,),cp["runs"][0]["lr"],dtype=DTYPE)
            for t in range(256):
                idx = torch.as_tensor(rng.integers(0,32,size=n.R))
                xb,yb = x[idx,torch.arange(n.R)],y[idx,torch.arange(n.R)]
                meta = dict(phase="continuation",arm=cp["arm"],checkpoint=cp["step"],step=t)
                grads = one(n,xb,yb,meta)
                n.sgd_step(lr,*grads)
    V.writecsv(OUT/"diagnostic_checks.csv",STATS.values())
    V.writecsv(OUT/"diagnostic_failed_cases.csv",FAILURES)
    original = sum(r["original_failures"] for r in STATS.values())
    matched = sum(r["matched_failures"] for r in STATS.values())
    result = dict(tier="post-result diagnostic; original registered FAIL retained",
        diagnosis="FORWARD_ROUNDING_ORDER" if original==100 and matched==0 else "UNRESOLVED",
        original_failures_reproduced=original,matched_order_failures=matched,
        forward_differences=dict(FORWARD),git_hash=subprocess.check_output(
            ["git","rev-parse","HEAD"],cwd=V.ROOT,text=True).strip(),
        source_sha256=V.sha(__file__),addendum_sha256=V.sha(V.ROOT/"specs/addendum_aq_identity_0917.md"))
    (OUT/"diagnosis.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    return 0 if result["diagnosis"]=="FORWARD_ROUNDING_ORDER" else 1


if __name__=="__main__":
    raise SystemExit(main())

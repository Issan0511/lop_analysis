#!/usr/bin/env python3
"""Independent small-fixture S checks; CUDA graph proof is separate and mandatory.

CPU output intentionally has production_ready=false. Synthetic fixtures are never
registered performance runs and cannot satisfy the CUDA host-reproduction gate.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from src import nap_cifar5p1_1018 as N
from analysis.nap_cifar5p1_1018 import verdict as V


def report(passed, measurements, mutants):
    return dict(pass_=bool(passed and all(mutants.values())), measurements=measurements,
                mutation_detected=mutants)


def item(passed, measurements, mutants):
    q=report(passed, measurements, mutants)
    q["pass"]=q.pop("pass_")
    return q


def clone(P):
    return [p.detach().clone().requires_grad_(True) for p in P]


def reference(P, X):
    """Single-seed nn.functional affine-LN reference, unlike stacked production."""
    outputs=[]
    for r in range(X.shape[0]):
        a=F.linear(X[r], P[0][r])
        a=F.layer_norm(a, (100,), P[1][r], P[2][r], N.EPS_LN)
        a=F.relu(a)
        a=F.linear(a, P[3][r])
        a=F.layer_norm(a, (100,), P[4][r], P[5][r], N.EPS_LN)
        outputs.append(F.linear(F.relu(a), P[6][r]))
    return torch.stack(outputs)


def mutant_forward(P,X, kind):
    a=X
    for l,(w,g,b) in enumerate(((0,1,2),(3,4,5))):
        a=torch.bmm(a,P[w].transpose(1,2))
        if kind=="bias_retained":
            a=a+torch.linspace(-.4,.4,100, dtype=a.dtype)[None,None,:]
        if kind=="post_relu_ln":
            a=F.relu(a)
        if kind=="batch_axis":
            mean=a.mean(1,keepdim=True)
            var=a.var(1,correction=0,keepdim=True)
            a=(a-mean)/torch.sqrt(var+N.EPS_LN)
        else:
            a=F.layer_norm(a,(100,),eps=N.EPS_LN)
        a=a*P[g][:,None]+P[b][:,None]
        if kind!="post_relu_ln":
            a=F.relu(a)
    return torch.bmm(a,P[6].transpose(1,2))


def check_ln_grad():
    torch.manual_seed(14101)
    P=N.init_nap([100,101], "cpu", dtype=torch.float64)
    with torch.no_grad():
        P[1].mul_(.8); P[2].copy_(torch.randn_like(P[2])*.1)
        P[4].mul_(1.2); P[5].copy_(torch.randn_like(P[5])*.2)
    X=torch.randn(2,7,3072,dtype=torch.float64,requires_grad=True)
    Q=clone(P); Xq=X.detach().clone().requires_grad_(True)
    out=N.nap_forward(P,X)["logits"]
    expected=reference(Q,Xq)
    probe=torch.randn_like(out)
    grad=torch.autograd.grad((out*probe).sum(), [X,*P])
    refgrad=torch.autograd.grad((expected*probe).sum(), [Xq,*Q])
    forward_err=float((out-expected).abs().max())
    gradient_err=max(float((a-b).abs().max()) for a,b in zip(grad,refgrad))
    # Bound fixed before running: fp64 eps x largest dot fan-in x 4096.
    tolerance=np.finfo(np.float64).eps*3072*4096
    mutant_errors={k:float((mutant_forward(P,X,k)-expected).abs().max())
                   for k in ("batch_axis","post_relu_ln","bias_retained")}
    ln=item(forward_err<tolerance, dict(max_forward_abs=forward_err, atol=tolerance,
            reference="per-seed F.linear + affine F.layer_norm", tolerance_derivation="eps64*3072*4096",
            mutation_max_abs=mutant_errors), {k:v>tolerance for k,v in mutant_errors.items()})
    kink=torch.tensor([-1.,0.,1.], dtype=torch.float64, requires_grad=True)
    kg=torch.autograd.grad(F.relu(kink).sum(),kink)[0]
    # LN gradient is dense across units; bypassing it changes the input Jacobian.
    z=torch.randn(3,100,dtype=torch.float64,requires_grad=True)
    probe2=torch.randn_like(z)
    g=torch.autograd.grad((F.relu(F.layer_norm(z,(100,),eps=N.EPS_LN))*probe2).sum(),z)[0]
    bypass=(F.layer_norm(z,(100,),eps=N.EPS_LN)>0).double()*probe2
    bypass_error=float((g-bypass).abs().max())
    gd=item(gradient_err<tolerance and torch.equal(kg,torch.tensor([0.,0.,1.],dtype=torch.float64)),
            dict(max_gradient_abs=gradient_err, atol=tolerance, native_relu_at_zero=float(kg[1]),
                 LN_chain_bypass_error=bypass_error), {"LN_chain_bypassed":bypass_error>tolerance})
    return ln,gd


def fixture():
    P=N.init_nap([100,101],"cpu",dtype=torch.float64)
    original=clone(P)
    with torch.no_grad():
        for j,k in enumerate((0,3,6)):
            P[k].mul_(torch.linspace(.3,1.9,100,dtype=torch.float64)[None,:,None])
            P[k][0].mul_(.4 if j%2==0 else 2.)
            P[k][1].mul_(2. if j%2==0 else .4)
        for g,b in ((1,2),(4,5)):
            P[g].mul_(1.7)
            P[b].copy_(torch.linspace(-.8,.7,100,dtype=torch.float64)[None,:])
    radii=[q.detach().norm(dim=(1,2)) for q in (original[0],original[3],original[6])]
    return P,original,radii


def projection_valid(before, after, radii):
    tol=np.finfo(np.float64).eps*3072*4096
    errors=[]
    for j,k in enumerate((0,3,6)):
        ref=before[k].detach()* (radii[j]/before[k].detach().norm(dim=(1,2)))[:,None,None]
        errors.append(float((after[k]-ref).abs().max()))
        errors.append(float((after[k].norm(dim=(1,2))-radii[j]).abs().max()))
    return max(errors)<tol, max(errors),tol


def check_projection():
    P,P0,radii=fixture(); before=clone(P)
    invalid=torch.zeros(2,dtype=torch.bool)
    movement=torch.zeros(2,5,dtype=torch.float64)
    # Nonzero fake moments/time: projection must leave them byte-identical.
    moments=[torch.rand_like(p) for p in P]
    hashes=[N.sha_tensor(p) for p in moments]
    N.project(P,radii,invalid,movement)
    good,error,tol=projection_valid(before,P,radii)
    mutations={}
    for kind in ("rowwise", "cap_only", "t1_radius", "omit_W3"):
        bad=clone(before)
        with torch.no_grad():
            for j,k in enumerate((0,3,6)):
                if kind=="omit_W3" and j==2:
                    continue
                if kind=="rowwise":
                    bad[k].mul_((P0[k].norm(dim=2)/bad[k].norm(dim=2))[:,:,None])
                else:
                    scale=radii[j]/bad[k].norm(dim=(1,2))
                    if kind=="cap_only": scale=scale.clamp(max=1.)
                    if kind=="t1_radius": scale=scale*1.07
                    bad[k].mul_(scale[:,None,None])
        mutations[kind]=not projection_valid(before,bad,radii)[0]
    zero=clone(P)
    with torch.no_grad(): zero[0][0].zero_()
    flags=torch.zeros(2,dtype=torch.bool)
    N.project(zero,radii,flags,torch.zeros(2,5,dtype=torch.float64))
    undefined=bool(flags[0]) and not bool(torch.isfinite(zero[0][0]).all())
    w=item(good and not bool(invalid.any()) and hashes==[N.sha_tensor(p) for p in moments] and undefined,
           dict(max_error=error, atol=tol, movement=movement.tolist(), moment_unchanged=True,
                zero_projection_flagged=undefined),mutations)
    affine_errors=[]; affine_mut={}
    for g,b in ((1,2),(4,5)):
        expected_scale=math.sqrt(100)/(before[g].square().sum(1)+before[b].square().sum(1)).sqrt()
        affine_errors.extend([float((P[g]-before[g]*expected_scale[:,None]).abs().max()),
                              float((P[b]-before[b]*expected_scale[:,None]).abs().max())])
        zero_beta=torch.zeros_like(P[b])
        affine_mut["beta_zeroed"]=float((zero_beta-P[b]).abs().max())>tol
        # Wrong sequential recomputation changes beta's coefficient.
        new_gamma=before[g]*expected_scale[:,None]
        second=math.sqrt(100)/(new_gamma.square().sum(1)+before[b].square().sum(1)).sqrt()
        affine_mut["sequential_recompute"]=float((before[b]*second[:,None]-P[b]).abs().max())>tol
    a=item(max(affine_errors)<tol,dict(max_error=max(affine_errors),atol=tol,
           joint_norms=[(P[g].square().sum(1)+P[b].square().sum(1)).sqrt().tolist() for g,b in ((1,2),(4,5))]),affine_mut)
    return w,a


class Synthetic:
    def __init__(self):
        gen=torch.Generator().manual_seed(1983)
        self.train_y=torch.arange(100).repeat_interleave(2)
        self.test_y=torch.arange(100)
        self.train=torch.randn(200,3072,generator=gen)
        self.test=torch.randn(100,3072,generator=gen)
        self.sha256={"fixture":"synthetic CPU; not CIFAR performance"}
    def inputs(self,split,cond,device):
        return getattr(self,split).to(device)


def check_host():
    cifar=Synthetic(); seeds=list(range(100,110)); captures={}; debug={}
    original_eval=N.C.evaluate
    def observe(P,*args,**kwargs):
        captures["P"]=[p.detach().clone() for p in P]
        return original_eval(P,*args,**kwargs)
    try:
        N.C.evaluate=observe
        with tempfile.TemporaryDirectory(prefix="nap_host_fixture_") as tmp:
            N.C.run("R",seeds,"std",3,torch.device("cpu"),Path(tmp),lr=N.LR,
                    graph=False,fresh=False,steps_hard=3,steps_easy=3,cifar=cifar,
                    progress=lambda _:None,debug=debug)
    finally:
        N.C.evaluate=original_eval
    s=N.State(seeds,torch.device("cpu"),nap=False)
    tr=N.Trainer(s,cifar.train,cifar.train_y,graph=False)
    ours=[]
    for batches in debug["batches"]:
        # Compare each update's exact online measurement and final affine tensors.
        for j in range(batches.shape[1]):
            ours.append(tr.train(batches[:,j:j+1]).clone())
    max_error=max(float((a-b).abs().max()) for a,b in zip(s.P,captures["P"]))
    equal=all(torch.equal(a,b) for a,b in zip(s.P,captures["P"]))
    online=all(torch.equal(a,b) for a,b in zip(ours,debug["online"]))
    init=N.init_nap(seeds,"cpu")
    weight_stream=all(torch.equal(init[k],debug["init"][j]) for k,j in ((0,0),(3,2),(6,4)))
    # Actual output dimensions and dataset-specific standardization are tested.
    constants_ok=not N.validate_host_definition()
    correct_mean,correct_dims=N.C.STD_MEAN,N.C.DIMS
    try:
        N.C.STD_MEAN=(.4914,.4822,.4465)
        bad_mean=bool(N.validate_host_definition())
        N.C.STD_MEAN=correct_mean
        N.C.DIMS=(3072,100,100,5)
        bad_head=bool(N.validate_host_definition())
    finally:
        N.C.STD_MEAN,N.C.DIMS=correct_mean,correct_dims
    return item(equal and online and weight_stream and constants_ok,
                dict(host_tensors_bit_equal=equal,max_parameter_error=max_error,
                     online_bit_equal=online,seed_slots=seeds,R=10,steps=9,
                     init_W_stream_matches=weight_stream,host_cuda_reproduction="NOT_RUN"),
                {"CIFAR10_constants":bad_mean,"five_outputs":bad_head})


def check_fresh():
    seeds=[100,101]; cifar=Synthetic()
    s=N.State(seeds,torch.device("cpu")); tr=N.Trainer(s,cifar.train,cifar.train_y,graph=False)
    gen=torch.Generator().manual_seed(83)
    batches=torch.randint(0,200,(2,3,32),generator=gen)
    initial=[N.sha_tensor(p) for p in s.P]
    tr.train(batches)
    oldLN=[p.detach().clone() for p in (s.P[1],s.P[2],s.P[4],s.P[5])]
    s.reset()
    reset_ok=initial==[N.sha_tensor(p) for p in s.P] and s.tc==0 and all(bool((p==0).all()) for p in (*s.m,*s.v))
    fresh=tr.train(batches)
    independent=N.State(seeds,torch.device("cpu")); t2=N.Trainer(independent,cifar.train,cifar.train_y,graph=False)
    expected=t2.train(batches)
    all_equal=all(torch.equal(p,q) for p,q in zip(s.mutable(),independent.mutable())) and torch.equal(fresh,expected)
    bad=N.State(seeds,torch.device("cpu"))
    with torch.no_grad():
        for p,q in zip((bad.P[1],bad.P[2],bad.P[4],bad.P[5]),oldLN): p.copy_(q)
    N.Trainer(bad,cifar.train,cifar.train_y,graph=False).train(batches)
    ln_detected=any(not torch.equal(p,q) for p,q in zip(s.P,bad.P))
    # An actual no-projection step produces off-sphere W, detected independently.
    no_proj=N.State(seeds,torch.device("cpu"))
    old_project=N.project
    try:
        N.project=lambda *args:None
        N.Trainer(no_proj,cifar.train,cifar.train_y,graph=False).train(batches)
    finally:
        N.project=old_project
    projection_detected=any(float((no_proj.P[k].norm(dim=(1,2))-no_proj.radii[j]).abs().max())>1e-5
                            for j,k in enumerate((0,3,6)))
    # Diagnostic isolation: no global RNG nor state mutation, with original tensor bytes.
    rng=torch.get_rng_state().clone(); hashes=[N.sha_tensor(p) for p in s.mutable()]
    N.evaluate(s,cifar.train[batches[:,0]],cifar.train_y[batches[:,0]],29,seeds)
    isolated=torch.equal(rng,torch.get_rng_state()) and hashes==[N.sha_tensor(p) for p in s.mutable()]
    return item(reset_ok and all_equal and isolated,
            dict(reset_initial_parameters=reset_ok,fresh_matches_new_state=all_equal,
                 same_batch_hash=N.sha_tensor(batches),diagnostics_isolated=isolated),
            {"LN_carried":ln_detected,"fresh_projection_disabled":projection_detected})


def check_stats():
    # Exhaust all sign configurations through n=10 and independently count null tails.
    max_error=0.; n_cases=0
    for n in range(11):
        binomial=np.asarray([sum(sum(bits)==k for bits in itertools.product((0,1),repeat=n)) for k in range(n+1)])
        for positives in range(n+1):
            d=[1.]*positives+[-1.]*(n-positives)+[0.]*(10-n)
            ref=min(1.,2*float(binomial[:min(positives,n-positives)+1].sum())/(2**n))
            max_error=max(max_error,abs(V.exact_sign(d)["p"]-ref));n_cases+=1
    p=[.04,.03]
    adjusted=V.holm(p)
    holm_ok=adjusted==[.06,.06]
    # Independent subset-based closed testing for the two-comparison Holm family.
    for p0,p1 in itertools.product((.001,.01,.03,.04,.08,.5,1.),repeat=2):
        got=V.holm([p0,p1])
        expected=[min(1.,max(p0,2*min(p0,p1))),min(1.,max(p1,2*min(p0,p1)))]
        holm_ok &= np.allclose(got,expected,rtol=0,atol=0)
    equivalent=V.classify([-.004]*5+[.004]*5,1.)["label"]=="EQUIVALENT_WITHIN_0.005"
    unresolved=V.classify([-.02]*5+[.02]*5,1.)["label"]=="UNRESOLVED"
    wins=V.classify([.1]*10,2*V.exact_sign([.1]*10)["p"])["label"]=="NAP_WINS"
    base={s:.1+.01*s for s in range(10)}
    comp={"SNA":{s:base[s]-.1 for s in reversed(range(10))},"KKT1":{s:base[s]+.1 for s in range(10)}}
    pair=V.compare(base,comp)
    join_ok=pair["SNA"]["label"]=="NAP_WINS" and pair["KKT1"]["label"]=="COMPARATOR_WINS"
    rows=[dict(seed=s,task=t,hard=int(t%2==1),online_acc=.2+s*.001) for s in range(10) for t in range(1,31)]
    duplicate=False
    try: V.seed_window(rows+[rows[0]],list(range(10)))
    except ValueError: duplicate=True
    no_correction=any(x<.05 for x in p) and not any(x<.05 for x in adjusted)
    one_sided=V.exact_sign([1]*8+[-1]*2)["p"] != sum(math.comb(10,k) for k in range(3))/1024
    wrong_seed=[base[s]-comp["SNA"][t] for s,t in zip(base,comp["SNA"])]
    wrong_seed_detected=not np.allclose(wrong_seed,list(pair["SNA"]["differences_by_seed"].values()),atol=1e-14)
    return item(max_error==0 and holm_ok and equivalent and unresolved and wins and join_ok and duplicate,
            dict(exhaustive_count_cases=n_cases,max_exact_p_error=max_error,holm=adjusted,
                 interval_coverage=1-2*(math.comb(10,0)+math.comb(10,1))/1024,
                 shuffled_seed_join_correct=join_ok,duplicate_rejected=duplicate),
            {"one_sided_p":one_sided,"Holm_omitted":no_correction,"seed_order_join":wrong_seed_detected})


def check_graph(device):
    if device!="cuda" or not torch.cuda.is_available():
        return dict(pass_=False,status="NOT_RUN_BLOCKED_CUDA")
    torch.manual_seed(2281)
    X=torch.randn(320,3072,device="cuda");Y=torch.arange(320,device="cuda")%100
    eager=N.State(list(range(100,110)),torch.device("cuda"))
    graph=N.State(list(range(100,110)),torch.device("cuda"))
    old=[N.sha_tensor(p) for p in graph.mutable()]
    gt=N.Trainer(graph,X,Y,graph=True)
    restored=old==[N.sha_tensor(p) for p in graph.mutable()]
    et=N.Trainer(eager,X,Y,graph=False)
    g=torch.Generator().manual_seed(42)
    idx=torch.randint(0,320,(10,4,32),generator=g).cuda()
    a=gt.train(idx);b=et.train(idx)
    maximum=max(float((x.float()-y.float()).abs().max()) for x,y in zip(graph.mutable(),eager.mutable()))
    tolerance=np.finfo(np.float32).eps*4096
    return dict(pass_=restored and maximum<tolerance and torch.equal(a,b),warmup_byte_restored=restored,
                max_graph_eager_abs=maximum,atol=tolerance,
                tolerance_derivation="eps32*4096; fixture only, G0 host reproduction separately required")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,default=N.OUT/"checks.json")
    ap.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    a=ap.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    results={}
    functions={"S-host":check_host,"S-fresh":check_fresh,"S-stats":check_stats}
    for name,fn in functions.items():
        try: results[name]=fn()
        except Exception as e: results[name]=dict(pass_=False,error=f"{type(e).__name__}: {e}")
    try: results["S-LN"],results["S-grad"]=check_ln_grad()
    except Exception as e:
        results["S-LN"]=results["S-grad"]=dict(pass_=False,error=f"{type(e).__name__}: {e}")
    try: results["S-Wproject"],results["S-affine"]=check_projection()
    except Exception as e:
        results["S-Wproject"]=results["S-affine"]=dict(pass_=False,error=f"{type(e).__name__}: {e}")
    for result in results.values():
        if "pass_" in result: result["pass"]=result.pop("pass_")
    graph=check_graph(a.device)
    graph["pass"]=graph.pop("pass_")
    results["S-CUDA-graph"]=graph
    synthetic_pass=all(results[name].get("pass") is True for name in N.CHECK_KEYS)
    production_ready=synthetic_pass and graph["pass"] and a.device=="cuda"
    output=dict(run_id=N.EXPERIMENT,all_pass=production_ready,cpu_synthetic_all_pass=synthetic_pass,
                production_ready=production_ready,source_sha256=N.source_hashes(),
                fixture_only=True,torch=torch.__version__,device=a.device,**results)
    N.write_json(a.out,output)
    print(json.dumps(dict(cpu_synthetic_all_pass=synthetic_pass,production_ready=production_ready,
                          failed=[name for name,value in results.items() if not value.get("pass")])))
    raise SystemExit(0 if synthetic_pass and (a.device=="cpu" or production_ready) else 1)


if __name__=="__main__":main()

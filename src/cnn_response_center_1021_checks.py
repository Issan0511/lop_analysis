#!/usr/bin/env python3
"""Synthetic contract tests and (after release) real-state preflight."""
from __future__ import annotations
import argparse
import copy
import csv
import gc
import json
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import cnn_response_center_1021 as M
import torch
import torch.nn.functional as F


def delta(a,b):
    return float((a-b).detach().abs().max())


def params(seed=440, runs=2, dtype=torch.float64):
    return [q.detach().to(dtype=dtype).requires_grad_(True) for q in M.E.stack_init(list(range(seed,seed+runs)),torch.device("cpu"))]


def synthetic():
    """No file/data/checkpoint reads, no real training, private RNG only."""
    rep = {}
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    gen = torch.Generator().manual_seed(1021)
    # The small scalar field checks exclude kinks by construction.
    z0 = torch.tensor([-.8,.4,1.2,-1.6],dtype=torch.float64)
    zd = torch.tensor([.5,-.7,1.1,-.9],dtype=torch.float64)
    z = z0.clone().requires_grad_(True)
    a = M.AnchorAdd.apply(M.phi(z0), M.phi(zd+(z-z0))-M.phi(zd))
    gd = torch.autograd.grad(a.sum(),z)[0]
    eps=1e-6
    fd=(M.phi(zd+eps)-M.phi(zd-eps))/(2*eps)
    rep["scalar_anchor_derivative"]={"value_exact":torch.equal(a,M.phi(z0)),"grad_exact":torch.equal(gd,M.dphi(zd)),
                                     "fd_abs":delta(fd,gd),"pass":torch.equal(a,M.phi(z0)) and torch.equal(gd,M.dphi(zd)) and delta(fd,gd)<1e-8}
    P=params(); D=params(810)
    xb=torch.randn(2,3,3,32,32,generator=gen,dtype=torch.float64)
    xd=xb-torch.tensor([.2,-.1,.5],dtype=torch.float64).view(1,1,3,1,1)
    act=M.LRM.BundleLeaky(2,.1,torch.device("cpu"))
    fld=M.Field(P,D)
    out,ze,de=M.forward(P,xb,act,fld,xd)
    ref=M.E.forward(P,xb,act)
    actual_d=torch.autograd.grad(out[1].sum(),out[0],retain_graph=True)[0]
    zd0=M.conv1(D,xd).detach()
    maxdiff=max(delta(a,b) for a,b in zip(out,ref))
    winners_a=F.max_pool2d(out[1],2,2,return_indices=True)[1]
    winners_b=F.max_pool2d(ref[1],2,2,return_indices=True)[1]
    rep["full_anchor"]={"max_feature_abs":maxdiff,"winner_equal":torch.equal(winners_a,winners_b),
                        "derivative_abs":delta(actual_d,M.dphi(zd0)),
                        "pass":maxdiff==0 and torch.equal(winners_a,winners_b) and torch.equal(actual_d,M.dphi(zd0))}
    # MaxPool routes only receiver winners; its gradient must pick donor slopes there.
    ga=torch.autograd.grad(F.max_pool2d(out[1],2,2).sum(),out[0],retain_graph=True)[0]
    expected=torch.zeros_like(actual_d).flatten(2)
    expected.scatter_add_(2,winners_a.flatten(2),M.dphi(zd0).flatten(2).gather(2,winners_a.flatten(2)))
    rep["receiver_winner_gradient"]={"abs":delta(ga,expected.reshape_as(ga)),"pass":torch.equal(ga,expected.reshape_as(ga))}
    y=torch.randint(0,10,(2,3),generator=gen)
    loss=F.cross_entropy(out[8].reshape(-1,10),y.reshape(-1))
    refloss=F.cross_entropy(ref[8].reshape(-1,10),y.reshape(-1))
    gs=torch.autograd.grad(loss,P,retain_graph=True)
    gn=torch.autograd.grad(refloss,P)
    downstream=max(delta(a,b) for a,b in zip(gs[2:],gn[2:]))
    c1diff=max(delta(a,b) for a,b in zip(gs[:2],gn[:2]))
    rep["only_c1_gradient_changes_initially"]={"downstream_max_abs":downstream,"c1_max_abs_change":c1diff,"pass":downstream==0 and c1diff>1e-10}
    # Derivative and input factors: the gradient w.r.t. c1 weights is receiver
    # input times the imposed donor slope, not donor input times donor slope.
    gfield=torch.autograd.grad(out[1].sum(),P[:2])
    zR=M.conv1(P,xb)
    gmanual=torch.autograd.grad((zR*M.dphi(zd0)).sum(),P[:2])
    zDwrong=M.conv1(P,xd)
    gwrong=torch.autograd.grad((zDwrong*M.dphi(zd0)).sum(),P[:2])
    rep["receiver_input_jacobian"]={"max_abs":max(delta(a,b) for a,b in zip(gfield,gmanual)),
                                    "wrong_donor_input_detected":max(delta(a,b) for a,b in zip(gfield,gwrong))>1e-6,
                                    "pass":all(torch.equal(a,b) for a,b in zip(gfield,gmanual)) and max(delta(a,b) for a,b in zip(gfield,gwrong))>1e-6}
    # Non-target donor layers must be irrelevant. Incorrect image correspondence,
    # wrong donor, and leaving the target site unchanged are caught by the slope.
    Dtail=[p.detach().clone() for p in D]
    for p in Dtail[2:]:p.add_(123.)
    outtail,_,_=M.forward(P,xb,act,M.Field(P,Dtail),xd)
    donor_swapped=M.dphi(M.conv1([p.detach().neg() for p in D[:2]],xd))
    image_swapped=M.dphi(M.conv1(D,xd[:,[1,0,2]]))
    natural_d=M.dphi(M.conv1(P,xb))
    mutations={"donor_changed":not torch.equal(actual_d,donor_swapped),
               "image_order_changed":not torch.equal(actual_d,image_swapped),
               "target_omitted":not torch.equal(actual_d,natural_d),
               "donor_other_layers_irrelevant":all(torch.equal(a,b) for a,b in zip(out,outtail))}
    rep["mutations"]={**mutations,"pass":all(mutations.values())}
    # Self sham at the anchor reproduces all natural gradients; same arbitrary
    # Adam state then produces exactly the same first update.
    selfout,_,_=M.forward(P,xb,act,M.Field(P,P),xb)
    natout=M.E.forward(P,xb,act)
    gself=torch.autograd.grad(F.cross_entropy(selfout[8].reshape(-1,10),y.reshape(-1)),P)
    gnat=torch.autograd.grad(F.cross_entropy(natout[8].reshape(-1,10),y.reshape(-1)),P)
    def adam(ps,grads):
        result=[]
        for p,g in zip(ps,grads):
            mi=torch.full_like(p,.023)*.9+.1*g
            vi=torch.full_like(p,.071)*.999+.001*g*g
            result.append(p-.001*(mi/(1-.9**901))/(torch.sqrt(vi/(1-.999**901))+1e-8))
        return result
    ad1,ad2=adam(P,gself),adam(P,gnat)
    rep["self_sham_first_step"]={"gradient_abs":max(delta(a,b) for a,b in zip(gself,gnat)),
                                  "adam_abs":max(delta(a,b) for a,b in zip(ad1,ad2)),
                                  "pass":all(torch.equal(a,b) for a,b in zip(gself,gnat)) and all(torch.equal(a,b) for a,b in zip(ad1,ad2))}
    # Check gain mode separately so it cannot silently become a forward scaling.
    gain=torch.full((2,16),2.7,dtype=torch.float64)
    gainout,_,gaind=M.forward(P,xb,act,M.Field(P,D,mode="gain",gain=gain),xd)
    dg=torch.autograd.grad(gainout[1].sum(),gainout[0])[0]
    rep["gain_anchor"]={"max_feature_abs":max(delta(a,b) for a,b in zip(gainout,ref)),
                         "derivative_abs":delta(dg,2.7*M.dphi(ref[0])),
                         "pass":all(torch.equal(a,b) for a,b in zip(gainout,ref)) and torch.equal(dg,2.7*M.dphi(ref[0]))}
    signs=M.sign_test([1.]*10)
    rep["sign_test"]={"p":signs["p_two_sided"],"pass":signs["p_two_sided"]==.001953125 and M.sign_test([0.]*10)["p_two_sided"]==1.}
    # A large minority outlier must not reverse the sign-test conclusion.
    adverse=[-.01]*9+[1.]
    st=M.sign_test(adverse)
    direction=M.effect_direction(sum(adverse)/10,st["n_positive"],st["n_negative"])
    rep["sign_mean_disagreement"]={**direction,"p":st["p_two_sided"],
                                   "pass":direction["direction"]=="heterogeneous" and direction["sign_direction"]=="negative"}
    # Exercise the real engine's state plumbing with generated pixel patterns,
    # not CIFAR. No saved research weights or data are read in this test.
    class SyntheticImages:
        def images(self,ids,device):
            rows=ids.to(device)[:,None]
            pixels=torch.arange(3072,device=device)[None,:]
            return ((rows*13+pixels*7)%257).float()/256
    cifar=SyntheticImages();dev=torch.device("cpu")
    base=M.LRM.BundleLR([("LR",410)],cifar,dev,graph=False)
    base.tc.fill_(900000)
    raw=base.X
    src=M.cpu(base.full_state());src_c=M.cpu(src)
    src_c["P"][1].add_(.3)
    data=SimpleNamespace(cifar=cifar,device=dev,seeds=[410],sources={"LR":src,"LRc":src_c},
                         inputs=lambda arm:raw-.2 if arm=="LRc" else raw)
    engine=M.Engine(data,"LR_swap",graph=False);engine.new_labels();engine.new_order()
    snap=engine.snapshot();engine.step(0);one=engine.snapshot();engine.restore(snap);engine.step(0)
    state_repro=M.tree_hash(one)==M.tree_hash(engine.snapshot())
    changed=any(not torch.equal(p,q) for p,q in zip(snap["state"]["P"],one["state"]["P"]))
    rep["synthetic_engine_restore"]={"state_reproduced":state_repro,"parameters_changed":changed,
                                     "finite_loss":bool(torch.isfinite(engine.ce_steps[:,0]).all()),
                                     "pass":state_repro and changed and bool(torch.isfinite(engine.ce_steps[:,0]).all())}
    rep["all_pass"]=all(v["pass"] for v in rep.values())
    return rep


def real(data,out,graph=True,release_info=None):
    """Read all seeds; brief fixed preflight only, not inference or model selection."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rep={"source_states_and_streams":{"pass":True,"identity":data.identity},
         "preflight_contract":{"pass":True,"value":M.preflight_contract(data,graph,release_info)}}
    reference_labels=None
    for arm in M.MAIN_ARMS:
        B=M.Engine(data,arm,graph=False);B.new_labels()
        if reference_labels is None:reference_labels=B.Y.detach().cpu().clone()
        assert torch.equal(reference_labels,B.Y.cpu())
        rep[f"anchor_{arm}"]=M.preflight_anchor(B)
        del B
    # Natural engine is independently stepped using the original step definition.
    for receiver in ("LR","LRc"):
        N=M.Engine(data,receiver+"_nat",graph=False)
        N.new_labels();N.new_order()
        initial=N.snapshot()
        S=M.Engine(data,receiver+"_self",graph=False);S.restore(N.snapshot())
        host=M.LRM.BundleLR([("LR",s) for s in data.seeds],data.cifar,data.device,graph=False)
        host.X=data.inputs(receiver);host.load_full_state(N.full_state())
        host.Y.copy_(N.Y);host.order.copy_(N.order)
        N.step(0);S.step(0);host.step(0)
        values={"P":max(delta(a,b) for a,b in zip(N.P,host.P)),
                "m":max(delta(a,b) for a,b in zip(N.m,host.m)),
                "v":max(delta(a,b) for a,b in zip(N.v,host.v)),
                "self_P":max(delta(a,b) for a,b in zip(N.P,S.P)),
                "self_m":max(delta(a,b) for a,b in zip(N.m,S.m)),
                "self_v":max(delta(a,b) for a,b in zip(N.v,S.v))}
        rep[f"host_and_self_first_step_{receiver}"]={**values,"pass":all(v==0 for v in values.values())}
        snap=N.snapshot();N.step(1);after=N.snapshot();N.restore(snap);N.step(1)
        rep[f"state_restore_{receiver}"]={"pass":M.tree_hash(after)==M.tree_hash(N.snapshot())}
        before=M.tree_hash(N.snapshot());M.diagnose(N,2)
        rep[f"diagnostic_immutable_{receiver}"]={"pass":before==M.tree_hash(N.snapshot())}
        # Same first epoch, but distinct arithmetic after the initial anchor.
        # This quantifies the reason self-field, rather than the natural kernel,
        # is the preregistered comparator. It is a descriptive numerical check.
        N.restore(initial);S.restore(initial)
        short=[]
        for j in range(M.CN.STEPS_PER_EPOCH):
            N.step(j);S.step(j)
            if j+1 in (1,2,5,10,25,50,75):
                short.append({"update":j+1,"P_max_abs":max(delta(a,b) for a,b in zip(N.P,S.P)),
                              "m_max_abs":max(delta(a,b) for a,b in zip(N.m,S.m)),
                              "v_max_abs":max(delta(a,b) for a,b in zip(N.v,S.v))})
        rep[f"natural_self_short_window_{receiver}"]={"rows":short,
            "natural_online_acc_by_seed":N.acc_steps.double().mean(1).cpu().tolist(),
            "self_online_acc_by_seed":S.acc_steps.double().mean(1).cpu().tolist(),
            "natural_online_ce_by_seed":N.ce_steps.double().mean(1).cpu().tolist(),
            "self_online_ce_by_seed":S.ce_steps.double().mean(1).cpu().tolist(),
            "interpretation":"Descriptive rounding drift; no equality criterion after the first step",
            "pass":bool(torch.isfinite(N.ce_steps).all() and torch.isfinite(S.ce_steps).all())}
        del N,S,host;gc.collect()
        if data.device.type=="cuda":torch.cuda.empty_cache()
    # Saved task-30 evaluation is compared with the original evaluation engine.
    # This documents cross-host/software differences, rather than silently calling
    # a new natural continuation bit-identical to an archived trajectory.
    for receiver in ("LR","LRc"):
        B=M.Engine(data,receiver+"_nat",graph=False)
        ys=[]
        for seed in data.seeds:
            g=M.H.stream("rlcc_labels",seed)
            for _ in range(30):y=M.CN.task_labels(g)
            assert torch.equal(g.get_state(),data.sources[receiver]["g_lab"][seed])
            ys.append(y)
        B.Y.copy_(torch.stack(ys))
        ev=M.E.evaluate(B.P,B.X,B.Y,B.act,chunk=20)
        tablepath=Path(data.source_meta[receiver]["full_path"]).parent/"per_task.csv"
        cols=("memo_acc","fit_ce","zbar_c1","zsd_c1","mob_c1","mob_pool_c1")
        with tablepath.open() as f:
            reference={int(r["seed"]):r for r in csv.DictReader(f) if int(r["task"])==30}
        diffs={c:max(abs(float(ev[j][c])-float(reference[s][c])) for j,s in enumerate(data.seeds)) for c in cols}
        # The predeclared engineering check tolerates float32 evaluation rounding;
        # the raw errors are kept. A larger mismatch blocks the new experiment.
        relative={c:max(abs(float(ev[j][c])-float(reference[s][c]))/max(1.,abs(float(reference[s][c]))) for j,s in enumerate(data.seeds)) for c in cols}
        rep[f"archived_task30_eval_{receiver}"]={"max_abs":diffs,"max_scaled_abs":relative,
                                                 "bit_identical":all(v==0 for v in diffs.values()),
                                                 "pass":diffs["memo_acc"]<=1/1200+1e-12 and max(relative[c] for c in cols if c!="memo_acc")<1e-4}
        del B;gc.collect()
    if graph and data.device.type=="cuda":
        # All seeds in original bundle order; swap graph has the extra frozen convs.
        for arm in ("LR_self","LR_swap"):
            G=M.Engine(data,arm,graph=True);Q=M.Engine(data,arm,graph=False)
            for B in (G,Q):B.new_labels();B.run_epoch()
            rep[f"graph_eager_{arm}"]={"pass":M.tree_hash(G.snapshot())==M.tree_hash(Q.snapshot()),
                                          "P_max_abs":max(delta(a,b) for a,b in zip(G.P,Q.P))}
            # A second run from the same initial state is an explicit repeat.
            snap=G.snapshot();G.run_epoch();expected=G.snapshot();G.restore(snap);G.run_epoch()
            rep[f"repeat_{arm}"]={"pass":M.tree_hash(expected)==M.tree_hash(G.snapshot())}
            del G,Q;gc.collect();torch.cuda.empty_cache()
    rep["all_pass"]=all(v["pass"] for v in rep.values())
    M.save_json(out/"checks.json",rep)
    return rep


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode",choices=("synthetic","real"));ap.add_argument("--out",required=True)
    ap.add_argument("--device",default="cuda");ap.add_argument("--seeds",default="10-19")
    ap.add_argument("--lr-root",default=str(M.DEFAULT_LR));ap.add_argument("--lrc-root",default=str(M.DEFAULT_LRC))
    ap.add_argument("--data-dir");ap.add_argument("--release");ap.add_argument("--eager",action="store_true")
    ap.add_argument("--gpu-lock",default="/tmp/lop_analysis_gpu.lock")
    args=ap.parse_args()
    if args.mode=="synthetic":
        rep=synthetic();M.save_json(Path(args.out)/"synthetic_checks.json",rep)
    else:
        if not args.release:raise SystemExit("Real preflight requires --release <registered commit>")
        release_info=M.guard_release(args.release)
        if M.E.parse_seeds(args.seeds)!=list(range(10,20)):
            raise SystemExit("Registered preflight requires the complete seeds 10-19 bundle")
        device=M.H.setup(args.device)
        with M.locked(Path(args.out)/".checks.lock"),M.locked(args.gpu_lock):
            data=M.Data(M.E.parse_seeds(args.seeds),device,args.lr_root,args.lrc_root,args.data_dir)
            rep=real(data,args.out,not args.eager,release_info)
    print(json.dumps(rep,ensure_ascii=False,indent=2,default=str),flush=True)
    if not rep["all_pass"]:raise SystemExit(1)


if __name__=="__main__":main()

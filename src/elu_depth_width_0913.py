"""Preregistered mean-depth x centered-width factorial; no host modules are edited."""
from pathlib import Path
import argparse, concurrent.futures, csv, hashlib, json, math, os, subprocess, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import torch
from src import width_sink_clamp_0909 as C
from src import transport_common_0910 as T
H = C.H
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/elu_depth_width_0913"
SPEC = ROOT / "specs/spec_elu_depth_width_0913.md"
CELLS = {"n5_d1": (5., -1.), "n10_d1": (10., -1.),
         "n5_d4": (5., -4.), "n10_d4": (10., -4.), "ref": None}
MEAS = (0, 20, 100, 300, 625)
CRIT = 4.30265273
F = torch.nn.functional

def dump(p, v):
    Path(p).write_text(json.dumps(v, indent=2, default=str))
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvwrite(p, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
def update_max(ck, k, v):
    ck[k] = max(ck.get(k, 0.), float(v))

def adam_step(p, gr, adam):
    with torch.no_grad():
        m, v, tc = adam
        tc[0] += 1
        c1 = 1 - .9 ** tc[0]; c2 = 1 - .999 ** tc[0]
        for q, g, mi, vi in zip(p, gr, m, v):
            mi.mul_(.9).add_(g, alpha=1-.9)
            vi.mul_(.999).addcmul_(g, g, value=1-.999)
            q -= .001 * (mi / c1) / ((vi / c2).sqrt() + 1e-8)

def invariant_values(p, xbar, mbase, norm, depth):
    W = p[0].detach().double()
    m = W.mean(1)
    return dict(norm_rel=float(((W-m[:,None]).norm(dim=1)/norm-1).abs().max()),
                mean_abs=float((W @ xbar + p[1].detach().double()-depth).abs().max()),
                rowmean_abs=float((m-mbase).abs().max()))

def project(p, xbar, mbase, targets, ck):
    norm, depth = targets
    with torch.no_grad():
        tail = [q.clone() for q in p[2:]]
        W = p[0].double(); Wt = W-W.mean(1,keepdim=True)
        denom = Wt.norm(dim=1,keepdim=True)
        assert bool(torch.isfinite(denom).all()) and float(denom.min()) > 1e-10
        p[0].copy_((Wt*(norm/denom)+mbase[:,None]).float())
        oldb = p[1].clone()
        p[1].copy_((depth-p[0].double() @ xbar).float())
        ck["bias_projection_abs_sum"] = ck.get("bias_projection_abs_sum", 0.) + float((p[1]-oldb).abs().mean())
        vals = invariant_values(p,xbar,mbase,norm,depth)
        for k,v in vals.items(): update_max(ck,k,v)
        assert vals["norm_rel"] <= 2e-6 and vals["mean_abs"] <= 2e-5 and vals["rowmean_abs"] <= 2e-7, vals
        assert all(torch.equal(a,b) for a,b in zip(tail,p[2:])), "projection changed downstream parameter"
        ck["projections"] = ck.get("projections",0)+1

@torch.no_grad()
def perf(p, act, x, y):
    logits = H.forward(p,x,act)[4]
    return float(F.cross_entropy(logits,y)), 100.*float((logits.argmax(1)==y).float().mean())

@torch.no_grad()
def evaluate(p,act,perm,xbar,xs,ys,probes,step,ck,explicit=False):
    tx,ty,rx,ry = probes
    test_x=tx[:,perm]; train_x=rx[:,perm]
    ce,acc=perf(p,act,test_x,ty); trce,tracc=perf(p,act,train_x,ry)
    W=p[0].double(); m=W.mean(1); wt=W-m[:,None]
    z=test_x.double()@W.T+p[1].double()
    g=act.dphi(z.float()).double()
    trainmean=W@xbar+p[1].double()
    r=dict(test_ce=ce,test_acc=acc,train_probe_ce=trce,train_probe_acc=tracc,
           cnorm=float(wt.norm(dim=1).mean()),cnorm_min=float(wt.norm(dim=1).min()),
           cnorm_max=float(wt.norm(dim=1).max()),rowmean=float(m.mean()),bias=float(p[1].double().mean()),
           actual_train_mean=float(trainmean.mean()),actual_train_mean_min=float(trainmean.min()),
           actual_train_mean_max=float(trainmean.max()),test_zmean=float(z.mean()),
           test_zsd=float(z.var(0,unbiased=False).mean().sqrt()),
           positive_frac=float((z>0).double().mean()),gate_mean=float(g.mean()),
           gate_low_frac=float((g<.05).double().mean()),
           immobile_units=int((g.max(0).values<1e-6).sum()),w2col=float(p[2].double().norm(dim=0).mean()))
    if step==625:
        r["task_train_ce"],r["task_train_acc"]=perf(p,act,xs,ys)
    if explicit:
        # The explicit sample mean checks the analytic xbar route independently.
        ez=torch.zeros_like(trainmean)
        for chunk in xs.split(1000):
            ez += (chunk.double()@W.T+p[1].double()).sum(0)/len(xs)
        err=float((ez-trainmean).abs().max())
        update_max(ck,"explicit_mean_abs",err)
        assert err<=2e-5, ("explicit mean",err)
    for k,v in r.items(): assert np.isfinite(v),(k,v)
    u=dict(cnorm_i=wt.norm(dim=1).numpy().copy(),rowmean_i=m.numpy().copy(),
           bias_i=p[1].double().numpy().copy(),train_zmean_i=trainmean.numpy().copy(),
           test_zmean_i=z.mean(0).numpy().copy(),test_zsd_i=z.std(0,unbiased=False).numpy().copy(),
           gate_mean_i=g.mean(0).numpy().copy(),gate_low_i=(g<.05).double().mean(0).numpy().copy(),
           positive_i=(z>0).double().mean(0).numpy().copy(),w2col_i=p[2].double().norm(dim=0).numpy().copy())
    return r,u

def loop(p,act,adam,gens,mnist,probes,t0,t1,cell,mbase,rows,units,ck,anchor_norm):
    targets=CELLS.get(cell)
    for task in range(t0,t1+1):
        perm=torch.randperm(784,generator=gens[0])
        idx=H.stratified_draw(mnist,gens[1])
        order=torch.randperm(H.TASK_EXAMPLES,generator=gens[2])
        xs=mnist.train_x[idx][:,perm][order];ys=mnist.train_y[idx][order]
        xbar=xs.double().mean(0)
        if cell!="prefix":
            pre,_=evaluate(p,act,perm,xbar,xs,ys,probes,-1,ck)
            pre.update(task=task,step=-1,phase="preprojection",cell=cell);rows.append(pre)
            if targets is not None: project(p,xbar,mbase,targets,ck)
            r,u=evaluate(p,act,perm,xbar,xs,ys,probes,0,ck)
            r.update(task=task,step=0,phase="postprojection",cell=cell);rows.append(r)
            for k,v in u.items():units[f"{cell}_t{task}_s0_{k}"]=v
        online=0.
        for step in range(1,626):
            o=H.forward(p,xs[(step-1)*16:step*16],act)
            loss=F.cross_entropy(o[4],ys[(step-1)*16:step*16])
            assert torch.isfinite(loss),("nonfinite train loss",cell,task,step)
            gr=torch.autograd.grad(loss,p);online+=float(loss.detach())
            adam_step(p,gr,adam)
            if targets is not None:project(p,xbar,mbase,targets,ck)
            if step in MEAS and (cell!="prefix" or step==625):
                r,u=evaluate(p,act,perm,xbar,xs,ys,probes,step,ck,explicit=(step==625))
                r.update(task=task,step=step,phase="adapt",cell=cell)
                if step==625:
                    r["online_ce"]=online/625
                    anchor_norm[task]=u["cnorm_i"].copy()
                rows.append(r)
                if step==625:
                    for k,v in u.items():units[f"{cell}_t{task}_s625_{k}"]=v
        if task%20==0:print("PROGRESS",cell,task,flush=True)

def qa(arm,seed,mnist,probes,out):
    act=C.make_act(arm)
    z=torch.linspace(-12,12,201,dtype=torch.float64).requires_grad_(True)
    z=z[z.abs()>1e-8]
    dg=torch.autograd.grad(act.phi(z).sum(),z)[0]
    assert float((dg-act.dphi(z)).abs().max())<1e-12
    p=H.init_params(seed,torch.device("cpu"))
    ad=([torch.zeros_like(q) for q in p],[torch.zeros_like(q) for q in p],[0])
    gens=[H.stream(r,seed) for r in ("perm","data","batch")]
    sn=C.snapshot(p,act,ad,gens)
    q,aq,bq,gq=C.restore(sn,arm)
    assert C.exact(sn,C.snapshot(q,aq,bq,gq)), "restore"
    # Compare a full untouched task with the established frozen loop.
    pa,aa,ada,ga=C.restore(sn,arm)
    pb,ab,adb,gb=C.restore(sn,arm)
    fakeprobe=C.Probe(probes[0][:512],probes[1][:512],float(mnist.train_x.mean()))
    C.loop(pa,aa,ada,ga,mnist,fakeprobe,1,1,"ref",None,None,{}, {})
    rr=[];uu={};cc={};nn={}
    # Prefix performs the identical forward/Adam order; measurements cannot alter it.
    loop(pb,ab,adb,gb,mnist,probes,1,1,"prefix",None,rr,uu,cc,nn)
    assert C.exact(C.snapshot(pa,aa,ada,ga),C.snapshot(pb,ab,adb,gb)), "frozen loop mismatch"
    xbar=mnist.train_x[:10000].double().mean(0)
    mb=pb[0].detach().double().mean(1);checks={}
    project(pb,xbar,mb,(5.,-1.),checks)
    good=invariant_values(pb,xbar,mb,5.,-1.)
    with torch.no_grad():pb[1][0]+=.01
    assert invariant_values(pb,xbar,mb,5.,-1.)["mean_abs"]>.009
    assert invariant_values(pb,xbar,mb,10.,-1.)["norm_rel"]>.49
    before=C.snapshot(pa,aa,ada,ga)
    evaluate(pa,aa,torch.arange(784),xbar,mnist.train_x[:10000],mnist.train_y[:10000],probes,625,{})
    assert C.exact(before,C.snapshot(pa,aa,ada,ga)), "measurement changed state/RNG"
    return dict(derivative_pass=True,restore_exact=True,host_task_exact=True,
                wrong_bias_detected=True,wrong_norm_detected=True,measurement_neutral=True,projection=good)

def anchor_check(arm,seed,norms):
    candidates=[ROOT/"results/elu_growth_0909"/f"{arm}_none_s{seed}_units.npz",
                Path("/home/issan/Projects/claude/proj_004_drift/results/elu_growth_0909")/f"{arm}_none_s{seed}_units.npz"]
    path=next((p for p in candidates if p.exists()),None)
    if path is None:return dict(status="MISSING_ANCHOR",count=0)
    with np.load(path) as d:
        arr=d["cnorm_i"]; selected=[t for t in norms if t<=len(arr)]
        error=max(float(np.abs(norms[t]-arr[t-1]).max()) for t in selected)
    assert len(selected)>0 and error<=1e-10,("anchor failed",path,error,len(selected))
    return dict(status="PASS",path=str(path),sha256=sha(path),count=len(selected),maxabs=error)

def run(arm,seed,smoke=False):
    torch.set_num_threads(1);H.setup("cpu");T.data_dir()
    out=OUT/("smoke" if smoke else "raw");out.mkdir(parents=True,exist_ok=True)
    tasks,prefix=(4,2) if smoke else (120,20)
    mnist=H.Mnist(torch.device("cpu"))
    ti=torch.randperm(len(mnist.test_x),generator=H.stream("elu_depth_width_testprobe",seed))[:2048]
    ri=torch.randperm(len(mnist.train_x),generator=H.stream("elu_depth_width_trainprobe",seed))[:2048]
    probes=(mnist.test_x[ti],mnist.test_y[ti],mnist.train_x[ri],mnist.train_y[ri])
    tag=f"{arm}_s{seed}";start=time.monotonic()
    checks=qa(arm,seed,mnist,probes,out)
    p=H.init_params(seed,torch.device("cpu"));act=C.make_act(arm)
    adam=([torch.zeros_like(q) for q in p],[torch.zeros_like(q) for q in p],[0])
    gens=[H.stream(r,seed) for r in ("perm","data","batch")]
    rows=[];units={};ck={};norms={}
    loop(p,act,adam,gens,mnist,probes,1,prefix,"prefix",None,rows,units,ck,norms)
    snap=C.snapshot(p,act,adam,gens);mbase=p[0].detach().double().mean(1)
    torch.save(snap,out/f"{tag}_task{prefix}.pt")
    branch_checks={}
    for cell in ("ref","n5_d1","n10_d1","n5_d4","n10_d4"):
        q,aq,adq,gq=C.restore(snap,arm)
        assert C.exact(snap,C.snapshot(q,aq,adq,gq)),("branch mismatch",cell)
        cck={};cn={}
        loop(q,aq,adq,gq,mnist,probes,prefix+1,tasks,cell,mbase,rows,units,cck,cn)
        if cell=="ref":
            norms.update(cn);checks["anchor"]=anchor_check(arm,seed,norms)
        else:
            assert cck["projections"]==(tasks-prefix)*626,(cell,cck)
        branch_checks[cell]=cck
        torch.save(C.snapshot(q,aq,adq,gq),out/f"{tag}_{cell}_task{tasks}.pt")
        print("CELL_DONE",tag,cell,round(time.monotonic()-start,1),flush=True)
        # Incremental output enables independent QA without losing finished branches.
        csvwrite(out/f"{tag}_rows.csv",[dict(activation=arm,seed=seed,**r) for r in rows])
        np.savez_compressed(out/f"{tag}_units.npz",**units)
    prov=dict(activation=arm,seed=seed,smoke=smoke,tasks=tasks,prefix=prefix,checks=checks,
              branch_checks=branch_checks,spec_sha256=sha(SPEC),code_sha256=sha(__file__),
              host_sha256=sha(H.__file__),clamp_sha256=sha(C.__file__),
              elu_sha256=sha(C.EG.__file__),data_sha256=mnist.sha256,
              test_probe_indices=ti.tolist(),train_probe_indices=ri.tolist(),
              torch_version=torch.__version__,numpy_version=np.__version__,threads=torch.get_num_threads(),
              git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
              wall_seconds=time.monotonic()-start,command=sys.argv,
              scope="controlled individual mean depth using bias; frozen t20 rowmeans; not natural mediation")
    dump(out/f"{tag}_provenance.json",prov)
    print("FINISHED",tag,round(time.monotonic()-start,1),flush=True)

def ci(v):
    x=np.array(v,dtype=float);m=float(x.mean());sd=float(x.std(ddof=1))
    h=CRIT*sd/math.sqrt(len(x))
    return dict(mean=m,sd=sd,ci_low=m-h,ci_high=m+h,seed_values=json.dumps(x.tolist()))
def classify(d):
    if d["ci_high"]<-.5:return "NEGATIVE_MATERIAL"
    if d["ci_low"]>.5:return "POSITIVE_MATERIAL"
    if d["ci_low"]>=-.5 and d["ci_high"]<=.5:return "EQUIVALENT_WITHIN_0.5PT"
    return "INCONCLUSIVE"

def report():
    import pandas as pd
    files=sorted((OUT/"raw").glob("*_rows.csv"))
    assert len(files)==6,("missing worker rows",len(files))
    dfs=[];provenance=[]
    for file in files:
        pr=json.loads(file.with_name(file.name.replace("_rows.csv","_provenance.json")).read_text())
        assert pr["tasks"]==120 and not pr["smoke"]
        assert pr["spec_sha256"]==sha(SPEC),"spec changed"
        assert pr["checks"]["anchor"]["status"] in ("PASS","MISSING_ANCHOR")
        provenance.append(pr);dfs.append(pd.read_csv(file))
    df=pd.concat(dfs,ignore_index=True)
    end=df[(df.step==625)&(df.task>=21)]
    rows=[];metrics={}
    for (a,s,c),part in end.groupby(["activation","seed","cell"]):
        assert len(part)==100 and set(part.task)==set(range(21,121)),(a,s,c)
        r=dict(activation=a,seed=int(s),cell=c)
        for name,l,h in [("early",21,40),("late",101,120),("all",21,120)]:
            b=part[(part.task>=l)&(part.task<=h)]
            for k in ("test_acc","test_ce","train_probe_ce","task_train_ce","online_ce",
                      "actual_train_mean","test_zmean","test_zsd","cnorm","gate_mean","gate_low_frac","positive_frac"):
                r[f"{name}_{k}"]=float(b[k].mean())
        r["L"]=r["early_test_acc"]-r["late_test_acc"]
        traj=df[(df.activation==a)&(df.seed==s)&(df.cell==c)&(df.task>=21)&(df.step>=0)]
        taskrows=[]
        for t,tt in traj.groupby("task"):
            tt=tt.sort_values("step")
            assert tuple(tt.step)==MEAS
            auc=float(np.trapezoid(tt.test_ce.to_numpy(),tt.step.to_numpy())/625.)
            gain=float(tt.iloc[0].test_ce-tt.iloc[-1].test_ce)
            taskrows.append((int(t),auc,gain))
        for name,l,h in [("early",21,40),("late",101,120),("all",21,120)]:
            vv=[x for x in taskrows if l<=x[0]<=h]
            r[f"{name}_adapt_ce_auc"]=float(np.mean([x[1] for x in vv]))
            r[f"{name}_adapt_ce_gain"]=float(np.mean([x[2] for x in vv]))
        pre=df[(df.activation==a)&(df.seed==s)&(df.cell==c)&(df.task==21)]
        r["initial_projection_test_ce_delta"]=float(pre[pre.step==0].test_ce.iloc[0]-pre[pre.step==-1].test_ce.iloc[0])
        rows.append(r);metrics[a,int(s),c]=r
    csvwrite(OUT/"seed_endpoints.csv",rows)
    contrast=[]
    # Positive W_harm always denotes high norm being worse: accuracy low-high,
    # degradation and CE high-low. Depth harm likewise has a 'deep is worse' sign.
    for met in ("late_test_acc","L","late_test_ce","late_task_train_ce","late_adapt_ce_auc"):
        sign=1 if met=="late_test_acc" else -1
        wh={}
        for a in ("ELU1","LR"):
            for d in ("d1","d4"):
                v=[sign*(metrics[a,s,f"n5_{d}"][met]-metrics[a,s,f"n10_{d}"][met]) for s in range(3)]
                wh[a,d]=v
                contrast.append(dict(metric=met,contrast=f"W_harm_{a}_{d}",**ci(v)))
            iv=[wh[a,"d4"][s]-wh[a,"d1"][s] for s in range(3)]
            wh[a,"interaction"]=iv
            contrast.append(dict(metric=met,contrast=f"W_depth_interaction_{a}",**ci(iv)))
            for n in ("n5","n10"):
                dv=[sign*(metrics[a,s,f"{n}_d1"][met]-metrics[a,s,f"{n}_d4"][met]) for s in range(3)]
                contrast.append(dict(metric=met,contrast=f"Depth_harm_{a}_{n}",**ci(dv)))
        for d in ("d1","d4"):
            v=[wh["ELU1",d][s]-wh["LR",d][s] for s in range(3)]
            stats=ci(v)
            contrast.append(dict(metric=met,contrast=f"H_ELU_minus_LR_{d}",**stats,
                                 verdict=classify(stats) if met=="late_test_acc" else "DESCRIPTIVE"))
        v=[wh["ELU1","interaction"][s]-wh["LR","interaction"][s] for s in range(3)]
        stats=ci(v)
        contrast.append(dict(metric=met,contrast="J_three_way",**stats,
                             verdict=classify(stats) if met=="late_test_acc" else "DESCRIPTIVE"))
    csvwrite(OUT/"verdict.csv",contrast)
    grouped=[]
    for a in ("ELU1","LR"):
        for c in CELLS:
            rr=[metrics[a,s,c] for s in range(3)]
            r=dict(activation=a,cell=c)
            for k in rr[0]:
                if k not in ("activation","seed","cell"):r[k]=float(np.mean([x[k] for x in rr]))
            grouped.append(r)
    csvwrite(OUT/"cell_means.csv",grouped)
    text=["# ELU depth x width factorial results","",
          "CPU Permuted MNIST; ELU1/leaky .1; 3 paired seeds; common task20 branches.",
          "Mean depth is the actual per-unit training-input mean, fixed through bias.",
          "Centered row norms fixed at 5/10; row means frozen at task20. This changes the",
          "causal regime. Width effects at fixed mean may still act through saturation.","",
          "## Cell means","",
          "| Activation | Cell | Late accuracy % | L points | Late train CE | Late test CE | Gate<.05 |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for r in grouped:
        text.append(f"| {r['activation']} | {r['cell']} | {r['late_test_acc']:.3f} | {r['L']:.3f} | {r['late_task_train_ce']:.4f} | {r['late_test_ce']:.4f} | {r['late_gate_low_frac']:.3f} |")
    text+=["","## Registered accuracy contrasts","",
           "W_harm=accuracy(norm5)-accuracy(norm10); H=W_harm(ELU)-W_harm(leaky).",
           "Positive means high width hurts more (H positive means ELU is hurt more).",
           "Intervals are paired-seed t95% with df2; +/-0.5pt practical-equivalence margin.","",
           "| Contrast | Mean pt | 95% CI | Verdict |","|---|---:|---|---|"]
    for r in contrast:
        if r["metric"]=="late_test_acc":
            text.append(f"| {r['contrast']} | {r['mean']:.3f} | [{r['ci_low']:.3f}, {r['ci_high']:.3f}] | {r.get('verdict','descriptive')} |")
    text+=["","## Validation and limits","",
           "All branches restored identical per-activation/seed prefix parameters, Adam and RNG.",
           "Frozen-host one-task exact equality, derivative, projection mutations and measurement",
           "neutrality passed. All task datasets and probes are common across continuation arms.",
           "See each raw provenance for per-update invariant maxima and committed reference checks.",
           "Only three independent seeds; narrow claims require CI within the registered bounds.",
           "Uniform per-unit means and frozen rowmeans differ from natural ELU populations and",
           "old global dclamp. This factorial does not separately identify saturation mediation.",
           "Training-probe CE is descriptive; held-out test CE/accuracy use 2048 fixed test examples.",
           "All seed values, absolute early/late levels, adaptation curves and initial projection",
           "shock are saved; no absence claim is inferred from nonsignificance.",""]
    (OUT/"summary.md").write_text("\n".join(text))
    dump(OUT/"provenance.json",dict(spec_sha256=sha(SPEC),code_sha256=sha(__file__),
         workers=[dict(activation=p["activation"],seed=p["seed"],wall_seconds=p["wall_seconds"],
                       checks=p["checks"],branch_checks=p["branch_checks"]) for p in provenance],
         total_workers=6,primary="late_test_acc and L",uncertainty="paired-seed t95 df2",
         output_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!="provenance.json"}))
    print((OUT/"summary.md").read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--activation",choices=["ELU1","LR"])
    ap.add_argument("--seed",type=int,default=0);ap.add_argument("--all",action="store_true")
    ap.add_argument("--smoke",action="store_true");ap.add_argument("--report",action="store_true")
    ap.add_argument("--jobs",type=int,default=6);a=ap.parse_args()
    if a.report:return report()
    if a.all:
        assert 1<=a.jobs<=6
        def job(z):
            cmd=[sys.executable,"-m","src.elu_depth_width_0913","--activation",z[0],"--seed",str(z[1])]
            if a.smoke:cmd.append("--smoke")
            log=OUT/("smoke" if a.smoke else "raw");log.mkdir(parents=True,exist_ok=True)
            with (log/f"{z[0]}_s{z[1]}.log").open("w") as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        jobs=[(act,s) for act in ("ELU1","LR") for s in (range(1) if a.smoke else range(3))]
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as ex:list(ex.map(job,jobs))
    else:
        assert a.activation
        run(a.activation,a.seed,a.smoke)
if __name__=="__main__":
    main()


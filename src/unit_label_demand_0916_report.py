"""Preregistered snapshot diagnostics. Per-unit additive statistics precede aggregation."""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from src import relu_gelu_silu_rl_0914 as B

NAME = "unit_label_demand_0916"
ROOT = B.ROOT
OUT = ROOT / "results" / NAME
GROUPS = ("P", "B", "V")
FIELDS = ("n", "n_active", "n_support", "n_new_support", "n_support_neutral", "n_suppress",
          "n_amplify", "n_suppress_neutral", "n_cert", "po_sum", "ad", "abs_ad", "au", "abs_au",
          "cert_abs_au", "force", "force_switch", "force_residual", "abs_force", "cert_abs_force",
          "expected_ad", "abs_expected_ad", "z_sum")
FI = {k: i for i, k in enumerate(FIELDS)}


def diagnostics(p, x, A, old, new):
    z1, a1, z2, a2, logits = B.forward(p, x, A)
    prob = logits.softmax(-1)
    ph1, ph2 = B.gate(z1, A), B.gate(z2, A)
    M, N, _ = x.shape
    j1 = torch.stack([torch.bmm(ph2 * p[4][:, c, None, :], p[2]) for c in range(10)], dim=2)
    j2 = p[4][:, None, :, :].expand(-1, N, -1, -1)
    po = prob.gather(-1, old[..., None]).squeeze(-1)
    res = []
    for a, z, ph, J in ((a1, z1, ph1, j1), (a2, z2, ph2, j2)):
        jo = J.gather(2, old[..., None, None].expand(-1, -1, 1, 100)).squeeze(2)
        jn = J.gather(2, new[..., None, None].expand(-1, -1, 1, 100)).squeeze(2)
        pJ = (prob[..., None] * J).sum(2)
        d, e, u = jo-jn, pJ-jo, pJ-jn
        bound = (1-po[..., None]) * (J-jo[:, :, None, :]).abs().amax(2)
        tol = 1e-10 * (1+J.abs().amax(2))
        gamma = a.sign()*d-bound
        res.append(dict(a=a, z=z, ph=ph, J=J, d=d, e=e, u=u, bound=bound, gamma=gamma,
                        po=po, tol=tol, expected_d=jo-J.mean(2)))
    return res


def sanity(res, old, new):
    errors = dict(identity=0., residual_bound_excess=0., unchanged_label_d=0., certificate_violations=0)
    for r in res:
        scale = 1+max(float(r[k].abs().max()) for k in ("u", "d", "e", "bound"))
        errors["identity"] = max(errors["identity"], float((r["u"]-r["d"]-r["e"]).abs().max())/scale)
        errors["residual_bound_excess"] = max(errors["residual_bound_excess"],
            float((r["e"].abs()-r["bound"]).clamp_min(0).max())/scale)
        same = old == new
        if bool(same.any()):
            errors["unchanged_label_d"] = max(errors["unchanged_label_d"], float(r["d"][same].abs().max()))
        cert = (r["a"].abs()>1e-12) & (r["gamma"]>r["tol"])
        errors["certificate_violations"] += int((cert & (r["a"].sign()*r["u"]<=0)).sum())
    assert max(errors[k] for k in ("identity", "residual_bound_excess", "unchanged_label_d"))<1e-9, errors
    assert errors["certificate_violations"] == 0, errors
    return errors


def sufficient_stats(res, old, new, models):
    zc = torch.tensor([{"GELU": -.7517915239, "SILU": -1.2784645428}.get(m["act"], -1.)
                       for m in models], device=old.device, dtype=res[0]["a"].dtype)[:, None, None]
    changed = (old != new)[..., None]
    layers = []
    for r in res:
        a, z, ph, d, u, e = (r[k] for k in ("a", "z", "ph", "d", "u", "e"))
        active = a.abs()>1e-12
        sd, su = a.sign()*d, a.sign()*u
        tol = r["tol"]
        cert = active & (r["gamma"]>tol)
        ad, au, force = a*d, a*u, -ph*u
        terms = (torch.ones_like(a), active, active & (sd>tol), active & (sd < -tol),
                 active & (sd.abs()<=tol), active & (su>tol), active & (su < -tol),
                 active & (su.abs()<=tol), cert, r["po"][..., None].expand_as(a),
                 ad, ad.abs(), au, au.abs(), cert*au.abs(), force, -ph*d, -ph*e,
                 force.abs(), cert*force.abs(), a*r["expected_d"], (a*r["expected_d"]).abs(), z)
        masks = (z>0, (z<=0)&(z>zc), z<=zc)
        assert bool((sum(m.to(torch.int8) for m in masks)==1).all())
        stats = torch.stack([torch.stack([(v*(mask&changed)).sum(1) for v in terms], dim=-1)
                             for mask in masks], dim=2)
        expected_n = changed.sum(1).expand(-1, 100)
        assert torch.equal(stats[..., FI["n"]].sum(2), expected_n.to(stats.dtype))
        # Verify aggregation independently of group masks for every additive field.
        direct = torch.stack([(v*changed).sum(1) for v in terms], dim=-1)
        error = (stats.sum(2)-direct).abs()/(1+direct.abs())
        assert float(error.max())<1e-9, float(error.max())
        layers.append(stats)
    return torch.stack(layers, dim=1).cpu().numpy()  # model, layer, unit, group, field


def selftest():
    models = [dict(seed=0, env="RL", act=a) for a in B.ACTS]
    p = [torch.stack([B.initial(0)[i].double()*3 for _ in models]).requires_grad_() for i in range(6)]
    gen = torch.Generator().manual_seed(17)
    x = torch.rand(5, 16, 784, generator=gen, dtype=torch.float64)*2-1
    old = torch.randint(10, (5, 16), generator=gen)
    new = (old+torch.randint(0, 10, (5, 16), generator=gen))%10
    A = B.Masks(B.ACTS, "cpu")
    r = diagnostics(p, x, A, old, new)
    z1, a1, z2, a2, logits = B.forward(p, x, A)
    errors = []
    for y, name in ((new, "u"), (old, "e")):
        loss = -logits.log_softmax(-1).gather(-1, y[..., None]).sum()
        auto = torch.autograd.grad(loss, (a1, a2), retain_graph=True)
        errors += [float((auto[l]-r[l][name]).abs().max().detach()) for l in range(2)]
    for c in range(10):
        auto = torch.autograd.grad(logits[..., c].sum(), (a1, a2), retain_graph=True)
        errors += [float((auto[l]-r[l]["J"][:, :, c]).abs().max().detach()) for l in range(2)]
    assert max(errors)<1e-10, max(errors)
    with torch.no_grad():
        g = sanity(r, old, new)
        stats = sufficient_stats(r, old, new, models)
        omitted = torch.bmm(p[4], p[2])[:, None, :, :]
        mutation_gate = float((omitted-r[0]["J"]).abs().max())
        mutation_label = float((r[0]["u"]+r[0]["d"]-r[0]["e"]).abs().max())
        assert mutation_gate>1e-4 and mutation_label>1e-4
        assert (stats[3:5, :, :, :, FI["n"]].sum((0, 1, 2))>0).all()
    return dict(autograd_max_abs=max(errors), sanity=g, omitted_gate_error=mutation_gate,
                swapped_label_error=mutation_label, partition_and_additive_closure=True)


def div(a, b):
    return float(a/b) if b>0 else float("nan")


def metrics(v, n_all):
    q = dict(zip(FIELDS, v))
    result = {"n": int(q["n"]), "n_active": int(q["n_active"]), "occupancy": div(q["n"], n_all),
              "active_fraction": div(q["n_active"], q["n"]), "p_old": div(q["po_sum"], q["n"]),
              "C": div(q["ad"], q["abs_ad"]), "S": div(q["au"], q["abs_au"]),
              "cert_mass": div(q["cert_abs_au"], q["abs_au"]),
              "cert_force_mass": div(q["cert_abs_force"], q["abs_force"]),
              "C_expected": div(q["expected_ad"], q["abs_expected_ad"]),
              "F": div(q["force"], n_all), "F_switch": div(q["force_switch"], n_all),
              "F_residual": div(q["force_residual"], n_all), "zmean": div(q["z_sum"], q["n"])}
    for target, source in (("support", "support"), ("suppress", "suppress"), ("cert", "cert"),
                           ("new_support", "new_support"), ("amplify", "amplify"),
                           ("support_neutral", "support_neutral"), ("suppress_neutral", "suppress_neutral")):
        result["f_"+target] = div(q["n_"+source], q["n_active"])
    return result


def rollup(records, models):
    rows = []
    for window, task_filter in (("t2", [2]), ("t2-10", list(range(2, 11)))):
        for step in (-1, 0, 25, 100, 300, 1000, 6000):
            selected = [r["stats"] for r in records if r["task"] in task_filter and r["step"] == step]
            if not selected or (step==-1 and window!="t2"):
                continue
            val = np.sum(selected, axis=0).sum(axis=2)  # model, layer, group, field
            for j, model in enumerate(models):
                for layer in range(2):
                    total = val[j, layer].sum(0)
                    for gi, group in enumerate(GROUPS+("ALL",)):
                        v = val[j, layer, gi] if gi<3 else total
                        rows.append(dict(**model, window=window, step=step, layer=layer+1, group=group,
                                         **metrics(v, total[FI["n"]])))
    return rows


def direction(vals, positive, negative):
    if not np.isfinite(vals).all(): return "INSUFFICIENT_MASS"
    if min(vals)>1e-8: return positive
    if max(vals)<-1e-8: return negative
    return "MIXED_OR_NEUTRAL"


def outputs(records, models, guard):
    rows = rollup(records, models)
    B.csvwrite(OUT / "group_summary.csv", rows)
    unitrows = []
    v = np.sum([r["stats"] for r in records if r["step"]==0], axis=0)
    for j, model in enumerate(models):
        for layer in range(2):
            for unit in range(100):
                total = v[j, layer, unit].sum(0)
                row = dict(**model, window="t2-10", step=0, layer=layer+1, unit=unit,
                           **metrics(total, total[FI["n"]]))
                for gi, group in enumerate(GROUPS):
                    row["occupancy_"+group] = div(v[j, layer, unit, gi, FI["n"]], total[FI["n"]])
                unitrows.append(row)
    B.csvwrite(OUT / "unit_summary.csv", unitrows)
    verdict = []
    for act in ("GELU", "SILU"):
        for layer in (1, 2):
            for group in GROUPS:
                rr = [r for r in rows if (r["act"],r["layer"],r["group"],r["window"],r["step"]) ==
                      (act,layer,group,"t2-10",0)]
                assert len(rr)==3
                q = dict(act=act, layer=layer, group=group, window="t2-10", step=0, trajectory=guard["verdict"])
                q["Q1"] = direction([r["C"] for r in rr], "OLD_SUPPORT_DOMINANT", "NEW_SUPPORT_DOMINANT")
                q["Q2"] = direction([r["S"] for r in rr], "SUPPRESSION_DOMINANT", "AMPLIFICATION_DOMINANT")
                fc = [r["f_cert"] for r in rr]
                q["Q3"] = ("INSUFFICIENT_MASS" if not np.isfinite(fc).all() else
                           "CERT_COVERS_MAJORITY" if min(fc)>.5 else "CERT_NOT_MAJORITY" if max(fc)<=.5 else "MIXED")
                for k in ("occupancy","active_fraction","p_old","C","S","f_support","f_suppress","f_cert",
                          "cert_mass","cert_force_mass","F","F_switch","F_residual","C_expected"):
                    vals = [r[k] for r in rr]
                    q[k+"_median"],q[k+"_min"],q[k+"_max"] = float(np.median(vals)),float(np.min(vals)),float(np.max(vals))
                verdict.append(q)
    B.csvwrite(OUT / "verdict.csv", verdict)
    text = ["# 個体へのラベル切替要求の配分", "",
            "既知48モデル軌道の10タスクprefixへの新規計測。RLの15モデルを両層で測定。",
            f"軌道検査: **{guard['verdict']}**。主窓: task2–10の切替直後(step0)、3seed中央値。",
            "分子分母をseed内で9切替・画像・個体についてプール後、seed間中央値。各ユニットの群所属は画像ごとに異なる。",
            "C/Sは符号付き総量÷絶対総量。比率は非ゼロ活性の画像・個体組に対する率。実Adam変位ではない。", "",
            "|活性化|層|群|占有|旧支持率|抑制率|十分条件率|条件の要求総量被覆|C|S|Q1|Q2|",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in verdict:
        vals=[r[k+"_median"] for k in ("occupancy","f_support","f_suppress","f_cert","cert_mass","C","S")]
        text.append("|"+"|".join([r["act"],str(r["layer"]),r["group"]]+[f"{v:.4f}" for v in vals]+[r["Q1"],r["Q2"]])+"|")
    text += ["", "## 時間変化（task2–10を各step別に集計、3seed中央値）", "",
             "|活性化|層|群|step|抑制率|十分条件率|C|S|生bias要求F|切替項F|残差項F|",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for act in ("GELU","SILU"):
        for layer in (1,2):
            for group in GROUPS:
                for step in (0,25,100,300,1000,6000):
                    rr=[r for r in rows if (r["act"],r["layer"],r["group"],r["window"],r["step"]) ==
                        (act,layer,group,"t2-10",step)]
                    nums=[np.median([r[k] for r in rr]) for k in ("f_suppress","f_cert","C","S","F","F_switch","F_residual")]
                    text.append("|"+"|".join([act,str(layer),group,str(step)]+[f"{v:.6g}" for v in nums])+"|")
    text += ["", "## 学習前とtask1学習後（同じtask1/task2ラベル、全群合算）", "",
             "|活性化|層|状態|C|S|旧支持率|抑制率|十分条件率|旧正答確率|",
             "|---|---|---|---|---|---|---|---|---|"]
    for act in B.ACTS:
        for layer in (1,2):
            for step in (-1,0):
                rr=[r for r in rows if (r["act"],r["layer"],r["group"],r["window"],r["step"]) ==
                    (act,layer,"ALL","t2",step)]
                vals=[np.median([r[k] for r in rr]) for k in ("C","S","f_support","f_suppress","f_cert","p_old")]
                text.append("|"+"|".join([act,str(layer),"初期" if step==-1 else "task1後"]+[f"{v:.6g}" for v in vals])+"|")
    text += ["", "群ごと・seed別・単一task2の値は group_summary.csv、個体IDを保った主窓の全群集計は unit_summary.csv。",
             "原始加法統計はraw/statistics.npz、再計算に必要な全checkpointとラベルもrawに保存。",
             "十分条件下の符号一致は検算。条件不成立は抑制不在を意味しない。初期と学習後の群別比較は構成変化を含む。",
             "この計測は要求分配の記述であり、W成長の因果効果、Adamの実変位、長期の沈降・LoPを検定していない。"]
    (OUT / "summary.md").write_text("\n".join(text)+"\n")


@torch.no_grad()
def analyze(raw):
    started=time.monotonic()
    inputs=torch.load(raw/"inputs.pt", weights_only=True)
    models=inputs["models"]
    x=torch.stack([inputs["x"][m["seed"]] for m in models]).double().cuda()
    A=B.Masks([m["act"] for m in models], "cuda")
    paths=[raw/"initial.pt"]+sorted(raw.glob("state_*.pt"))
    assert len(paths)==55, len(paths)
    records=[]; errors=[]
    for index,path in enumerate(paths):
        state=torch.load(path, weights_only=True)
        task,step=state["task"],state["step"]
        old=torch.stack([inputs["labels"][task-2,m["seed"]] for m in models]).cuda()
        new=torch.stack([inputs["labels"][task-1,m["seed"]] for m in models]).cuda()
        p=[v.double().cuda() for v in state["p"]]
        res=diagnostics(p,x,A,old,new)
        g=sanity(res,old,new)
        stats=sufficient_stats(res,old,new,models)
        records.append(dict(task=task,step=step,stats=stats))
        errors.append(dict(task=task,step=step,**g,changed_images=[int((old[j]!=new[j]).sum()) for j in range(len(models))]))
        if index%6==0 or index==len(paths)-1:
            print(f"ANALYZED {index+1}/{len(paths)} elapsed={time.monotonic()-started:.1f}s",flush=True)
        del res,p
    np.savez_compressed(raw/"statistics.npz",values=np.stack([r["stats"] for r in records]),
                        tasks=np.array([r["task"] for r in records]),steps=np.array([r["step"] for r in records]),
                        fields=np.array(FIELDS),groups=np.array(GROUPS))
    (OUT/"diagnostic_checks.json").write_text(json.dumps(errors,indent=2))
    guard=json.loads((OUT/"trajectory_check.json").read_text())
    outputs(records,models,guard)
    (OUT/"analysis_provenance.json").write_text(json.dumps(dict(
        git_hash=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        analysis_code_sha256=B.sha(__file__),dtype="float64",snapshots=len(records),
        checkpoint_sha256={p.name:B.sha(p) for p in paths},inputs_sha256=B.sha(raw/"inputs.pt"),
        wall_seconds=time.monotonic()-started),indent=2))
    print("ANALYSIS COMPLETE",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw",type=Path,default=OUT/"raw")
    ap.add_argument("--selftest",action="store_true")
    args=ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    if args.selftest: print(json.dumps(selftest(),indent=2))
    else: analyze(args.raw)

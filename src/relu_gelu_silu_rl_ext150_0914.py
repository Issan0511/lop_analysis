#!/usr/bin/env python3
"""Preregistered extension of relu_gelu_silu_rl_0914 to 150 tasks
(spec_relu_gelu_silu_rl_ext150_0914.md).  Physics, activations and self-test are the
parent module's, imported unchanged; only the task count and the late window differ.
Run from the repo root:  python3 -m src.relu_gelu_silu_rl_ext150_0914
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, math, subprocess, time
from pathlib import Path
import numpy as np
import torch
from src import relu_gelu_silu_rl_0914 as B

ROOT=B.ROOT
NAME="relu_gelu_silu_rl_ext150_0914"
OUT=ROOT/f"results/{NAME}"
TASKS=150; LATE=(141,150)
PREREG_COMMIT="3f75420340ec6b09bb81607c27a293c9132822a0"
PARENT_RESULT="7ce3b17"
ZC={"GELU":-0.7517915239,"SILU":-1.2784645428}

def run(device="cuda",tasks=TASKS,out_dir=OUT):
    assert 1<=tasks<=TASKS
    out=B.output_path(out_dir);out.mkdir(parents=True,exist_ok=True)
    subprocess.run(["git","merge-base","--is-ancestor",PREREG_COMMIT,"HEAD"],cwd=ROOT,check=True)
    reg=subprocess.check_output(["git","show",PREREG_COMMIT+f":specs/spec_{NAME}.md"],cwd=ROOT)
    assert hashlib.sha256(reg).hexdigest()==B.sha(ROOT/f"specs/spec_{NAME}.md"),"spec changed since preregistration"
    st=B.selftest();(out/"selftest.json").write_text(json.dumps(st,indent=1));print("SELFTEST OK",flush=True)
    e=B.Engine(device=device)
    gperm={s:B.stream("env_perm_0913",s) for s in range(3)}
    glabel={s:B.stream("env_labels_0913",s) for s in range(3)}
    gbatch={s:B.stream("env_batch_0913",s) for s in range(3)}
    ax=B.read_idx(B.DATA/"train-images-idx3-ubyte.gz").reshape(-1,784).astype(np.float32)/255
    ay=B.read_idx(B.DATA/"train-labels-idx1-ubyte.gz").astype(np.int64)
    subset={s:torch.randperm(len(ax),generator=B.stream("rl_subset",s))[:B.N].numpy() for s in range(3)}
    xs={s:torch.tensor(ax[subset[s]],device=e.device) for s in range(3)}
    ys_cpu={s:torch.tensor(ay[subset[s]]) for s in range(3)}
    ys={s:ys_cpu[s].to(e.device) for s in range(3)}
    rows=[];unitout={}
    e.capture();t0=time.monotonic()
    for task in range(1,tasks+1):
        epochs_needed=-(-(B.STEPS*B.BATCH)//B.N)
        orders={}
        for s in range(3):
            flat=torch.stack([torch.randperm(B.N,generator=gbatch[s]) for _ in range(epochs_needed)]).reshape(-1)[:B.STEPS*B.BATCH]
            orders[s]=flat.reshape(B.STEPS,B.BATCH)
        perm_cpu={s:torch.randperm(784,generator=gperm[s]) for s in range(3)}
        perm={s:perm_cpu[s].to(e.device) for s in range(3)}
        labels_cpu={s:torch.randint(10,(B.N,),generator=glabel[s]) for s in range(3)}
        labels={s:labels_cpu[s].to(e.device) for s in range(3)}
        orderstack=torch.stack([orders[m["seed"]] for m in B.MODELS],1).to(e.device)
        floors={}
        for s in range(3):
            floors[s,"PM"]=float(torch.bincount(ys_cpu[s],minlength=10).max())/B.N
            floors[s,"RL"]=float(torch.bincount(labels_cpu[s],minlength=10).max())/B.N
        for j,m in enumerate(B.MODELS):
            s=m["seed"];e.cx[j].copy_(xs[s][:,perm[s]] if m["env"]=="PM" else xs[s])
            yy=ys[s] if m["env"]=="PM" else labels[s]
            e.cy[j].copy_(torch.nn.functional.one_hot(yy,10))
        e.acc.zero_();e.ce.zero_()
        for step in range(0,B.STEPS,B.BLOCK):
            e.indices.copy_(orderstack[step:step+B.BLOCK]);e.replay_block()
        met,units=e.evaluate()
        online_acc=(e.acc/B.STEPS).cpu().numpy();online_ce=(e.ce/B.STEPS).cpu().numpy()
        assert np.isfinite(online_ce).all() and all(bool(torch.isfinite(q).all()) for q in e.p),("NONFINITE",task)
        for j,m in enumerate(B.MODELS):
            rows.append(dict(**m,task=task,floor_acc=floors[m["seed"],m["env"]],
                             online_acc=float(online_acc[j]),online_ce=float(online_ce[j]),
                             **{k:float(v[j]) for k,v in met.items()}))
        for k,v in units.items():unitout[f"{k}_t{task}"]=v
        if task%10==0 or task==tasks:B.csvwrite(out/"rows.csv",rows)
        smp={m["act"]:online_acc[j] for j,m in enumerate(B.MODELS) if m["seed"]==0 and m["env"]=="RL"}
        g2={m["act"]:online_acc[j] for j,m in enumerate(B.MODELS) if m["seed"]==2 and m["env"]=="RL"}
        print(f'TASK {task}/{tasks} elapsed={time.monotonic()-t0:.1f}s RL s0 GELU={smp["GELU"]:.3f} LR={smp["LR"]:.3f} | s2 SILU={g2["SILU"]:.3f}',flush=True)
    B.csvwrite(out/"rows.csv",rows)
    np.savez_compressed(out/"units.npz",**unitout)
    dev=torch.cuda.get_device_name(e.device) if e.device.type=="cuda" else str(e.device)
    prov=dict(prereg_commit=PREREG_COMMIT,parent_result=PARENT_RESULT,spec_sha256=B.sha(ROOT/f"specs/spec_{NAME}.md"),
        code_sha256=B.sha(Path(__file__)),parent_code_sha256=B.sha(Path(B.__file__)),
        git_hash=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        data_sha256={n:B.sha(B.DATA/n) for n in B.DATA_FILES},subset_sha256={s:B.arrsha(v) for s,v in subset.items()},
        torch_version=torch.__version__,device=dev,dtype="float32",tf32=False,deterministic=True,models=B.MODELS,
        tasks=tasks,steps_per_task=B.STEPS,images=B.N,batch=B.BATCH,late_window=list(LATE),wall_seconds=time.monotonic()-t0)
    (out/"provenance.json").write_text(json.dumps(prov,indent=2))

def report(out_dir=OUT):
    out=B.output_path(out_dir)
    rows=B.read_rows(out/"rows.csv")
    assert len(rows)==len(B.MODELS)*TASKS,len(rows)
    key=lambda m:(m["seed"],m["env"],m["act"])
    by={}
    for r in rows:by.setdefault((r["seed"],r["env"],r["act"]),[]).append(r)
    for k in by:by[k].sort(key=lambda r:r["task"]);assert [r["task"] for r in by[k]]==list(range(1,TASKS+1)),k
    # G1: t1-50 against the parent's committed rows
    old=list(csv.DictReader(io.StringIO(subprocess.check_output(
        ["git","show",f"{PARENT_RESULT}:results/relu_gelu_silu_rl_0914/rows.csv"],cwd=ROOT,text=True))))
    worst=0.;n=0
    for o in old:
        r=by[(int(o["seed"]),o["env"],o["act"])][int(o["task"])-1]
        for k in ("online_acc","train_ce"):
            worst=max(worst,abs(r[k]-float(o[k])));n+=1
    g1=dict(max_abs=worst,compared=n,expected=len(B.MODELS)*50*2)
    levels=[]
    for m in B.MODELS:
        rr=by[key(m)];late=[r for r in rr if LATE[0]<=r["task"]<=LATE[1]]
        fl=np.asarray([r["floor_acc"] for r in late]);F_,s_=float(fl.mean()),float(fl.std(ddof=1))
        thr=F_+3*s_/math.sqrt(10);la=float(np.mean([r["online_acc"] for r in late]))
        first=next((r["task"] for r in rr if r["online_acc"]<=thr),"")
        zc=ZC.get(m["act"]);zcross=next((r["task"] for r in rr if r["zmed_l2"]<zc),"") if zc is not None else ""
        v=np.array([r["online_acc"] for r in rr]);sl=float(np.polyfit(np.arange(131,151),v[130:150],1)[0]*10)
        levels.append(dict(**m,late_online_acc=la,floor_thr=thr,floor_verdict="AT_FLOOR" if la<=thr else "ABOVE_FLOOR",
            first_floor_task=first,zmed_below_zc_task=zcross,slope_t131_150=sl,
            acc_t50=rr[49]["online_acc"],acc_t100=rr[99]["online_acc"],acc_t150=rr[149]["online_acc"],
            zmed_l2_t50=rr[49]["zmed_l2"],zmed_l2_t100=rr[99]["zmed_l2"],zmed_l2_t150=rr[149]["zmed_l2"],
            neggate_l2_t50=rr[49]["neggate_l2"],neggate_l2_t100=rr[99]["neggate_l2"],neggate_l2_t150=rr[149]["neggate_l2"],
            tinygate_l2_t50=rr[49]["tinygate_l2"],tinygate_l2_t100=rr[99]["tinygate_l2"],tinygate_l2_t150=rr[149]["tinygate_l2"]))
    B.csvwrite(out/"levels.csv",levels)
    V={(r["act"],r["seed"]):r["floor_verdict"] for r in levels if r["env"]=="RL"}
    lr_ok=all(V["LR",s]=="ABOVE_FLOOR" for s in range(3))
    g=[V["GELU",s] for s in range(3)]
    q1="GELU_DIES_BY_150" if all(x=="AT_FLOOR" for x in g) else "GELU_SURVIVES_150" if all(x=="ABOVE_FLOOR" for x in g) else "GELU_SPLIT"
    q2="SILU_S2_DIES_BY_150" if V["SILU",2]=="AT_FLOOR" else "SILU_S2_SURVIVES_150"
    surv=[f"{a}s{s}" for a in ("R","GELU","SILU") for s in range(3) if V[a,s]=="ABOVE_FLOOR"]
    q3="ALL_DIE_BY_150" if not surv else "SURVIVORS:"+",".join(surv)
    tag="" if lr_ok else " (CONTROL_ALSO_DIES)"
    verdict=[dict(section="G0",verdict="PASS" if lr_ok else "FAIL",detail=" ".join(V["LR",s] for s in range(3))),
             dict(section="G1",verdict="BIT_IDENTICAL" if worst==0 and n==g1["expected"] else "DIFFERS",value=worst,detail=f'{n}/{g1["expected"]}'),
             dict(section="Q1",verdict=q1+tag,detail=" ".join(g)),dict(section="Q2",verdict=q2+tag,detail=V["SILU",2]),
             dict(section="Q3",verdict=q3+tag)]
    B.csvwrite(out/"verdict.csv",verdict)
    L=[f"# {NAME} report","",f"G0 (LR 3/3 ABOVE at t141–150): **{'PASS' if lr_ok else 'FAIL'}**",
       f"G1 (t1–50 vs parent `{PARENT_RESULT}`): max|Δ| = {worst:.3g} over {n}/{g1['expected']} values",
       "",f"Q1: **{q1+tag}**  ",f"Q2: **{q2+tag}**  ",f"Q3: **{q3+tag}**","",
       "## RL, per model (late window t141–150)","",
       "|Arm|seed|acc t50|acc t100|acc t150|late acc|thr|verdict|first floor|z̄₂<z_c first|slope t131–150 /10|",
       "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    for r in levels:
        if r["env"]!="RL":continue
        L.append(f'|{r["act"]}|{r["seed"]}|{r["acc_t50"]:.3f}|{r["acc_t100"]:.3f}|{r["acc_t150"]:.3f}|{r["late_online_acc"]:.3f}|{r["floor_thr"]:.3f}|{r["floor_verdict"]}|{r["first_floor_task"] or "—"}|{r["zmed_below_zc_task"] or "—"}|{r["slope_t131_150"]:+.4f}|')
    L+=["","## RL layer 2 (report only)","","|Arm|seed|median z̄₂ t50/t100/t150|φ′<0 t50/t100/t150|\\|φ′\\|<1e−8 t50/t100/t150|","|---|---|---|---|---|"]
    for r in levels:
        if r["env"]!="RL" or r["act"] not in ("GELU","SILU","LR"):continue
        L.append(f'|{r["act"]}|{r["seed"]}|{r["zmed_l2_t50"]:.2f} / {r["zmed_l2_t100"]:.2f} / {r["zmed_l2_t150"]:.2f}|{r["neggate_l2_t50"]:.2f} / {r["neggate_l2_t100"]:.2f} / {r["neggate_l2_t150"]:.2f}|{r["tinygate_l2_t50"]:.2f} / {r["tinygate_l2_t100"]:.2f} / {r["tinygate_l2_t150"]:.2f}|')
    L+=["","## PM late acc (t141–150, report only)","","|Arm|mean|","|---|---:|"]
    for a in B.ACTS:L.append(f'|{a}|{np.mean([r["late_online_acc"] for r in levels if r["env"]=="PM" and r["act"]==a]):.4f}|')
    (out/"summary.md").write_text("\n".join(L)+"\n",encoding="utf-8");print((out/"summary.md").read_text(),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--device",default="cuda");ap.add_argument("--tasks",type=int,default=TASKS)
    ap.add_argument("--out-dir",default=f"results/{NAME}");ap.add_argument("--report",action="store_true");a=ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.set_num_threads(1)
    if a.report:report(a.out_dir)
    else:run(a.device,a.tasks,a.out_dir)

#!/usr/bin/env python3
"""Preregistered kick dial (spec_perm_dial_rl_0915.md): RL labels, fixed 1200 images, and only
k of the 784 pixels permuted afresh each task.  Engine, activations and self-test are
relu_gelu_silu_rl_0914's, imported unchanged.  Run from repo root:
    python3 -m src.perm_dial_rl_0915            (train)
    python3 -m src.perm_dial_rl_0915 --report   (verdict)
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, math, subprocess, time
from pathlib import Path
import numpy as np
import torch
from src import relu_gelu_silu_rl_0914 as B

ROOT=B.ROOT; NAME="perm_dial_rl_0915"; OUT=ROOT/f"results/{NAME}"
TASKS=150; LATE=(141,150)
KS=(0,25,107,270,784); ACTS=B.ACTS
MODELS=[dict(seed=s,k=k,act=a) for s in range(3) for k in KS for a in ACTS]
PREREG_COMMIT="7ab765ce21b78a2f69680b355c31c240cc96da1e"
PARENT_RESULT="fca473d"
KEEP_UNITS=("zmean_l1","cnorm_l1","tinygate_l1","zmean_l2","zstd_l2","tinygate_l2","neggate_l2")

def partial_index(o,r,k):
    """pixels o[:k] permuted among themselves in argsort(r[:k]) order; the rest stay put."""
    idx=torch.arange(784)
    if k>0:
        K=o[:k]; idx[K]=K[torch.argsort(r[:k])]
    return idx

def run(device="cuda",tasks=TASKS,out_dir=OUT,check_prereg=True):
    assert 1<=tasks<=TASKS
    out=B.output_path(out_dir);out.mkdir(parents=True,exist_ok=True)
    if check_prereg:
        subprocess.run(["git","merge-base","--is-ancestor",PREREG_COMMIT,"HEAD"],cwd=ROOT,check=True)
        reg=subprocess.check_output(["git","show",PREREG_COMMIT+f":specs/spec_{NAME}.md"],cwd=ROOT)
        assert hashlib.sha256(reg).hexdigest()==B.sha(ROOT/f"specs/spec_{NAME}.md"),"spec changed since preregistration"
    st=B.selftest()
    # dial self-test: k=0 is the identity, k=784 is a full permutation, each k moves at most k pixels,
    # and a wrong construction (permuting all pixels for every k) is caught
    g=torch.Generator().manual_seed(0); o=torch.randperm(784,generator=g); r=torch.rand(784,generator=g)
    moved={k:int((partial_index(o,r,k)!=torch.arange(784)).sum()) for k in KS}
    assert moved[0]==0 and all(moved[k]<=k for k in KS) and moved[784]>700,moved
    assert sorted(partial_index(o,r,107).tolist())==list(range(784)),"not a permutation"
    wrong=int((torch.argsort(r)!=torch.arange(784)).sum()); assert wrong>moved[25],("vacuous dial check",wrong)
    st["dial_pixels_moved"]=moved
    (out/"selftest.json").write_text(json.dumps(st,indent=1));print("SELFTEST OK",moved,flush=True)

    e=B.Engine(models=MODELS,device=device)
    gperm={s:B.stream("env_perm_0913",s) for s in range(3)}
    glabel={s:B.stream("env_labels_0913",s) for s in range(3)}
    gbatch={s:B.stream("env_batch_0913",s) for s in range(3)}
    gpart={s:B.stream("partial_perm_0915",s) for s in range(3)}
    ax=B.read_idx(B.DATA/"train-images-idx3-ubyte.gz").reshape(-1,784).astype(np.float32)/255
    subset={s:torch.randperm(len(ax),generator=B.stream("rl_subset",s))[:B.N].numpy() for s in range(3)}
    xs={s:torch.tensor(ax[subset[s]],device=e.device) for s in range(3)}
    rows=[];unitout={}
    e.capture();t0=time.monotonic()
    for task in range(1,tasks+1):
        epochs_needed=-(-(B.STEPS*B.BATCH)//B.N)
        orders={}
        for s in range(3):
            flat=torch.stack([torch.randperm(B.N,generator=gbatch[s]) for _ in range(epochs_needed)]).reshape(-1)[:B.STEPS*B.BATCH]
            orders[s]=flat.reshape(B.STEPS,B.BATCH)
        _=[torch.randperm(784,generator=gperm[s]) for s in range(3)]          # consumed exactly as the parent
        labels_cpu={s:torch.randint(10,(B.N,),generator=glabel[s]) for s in range(3)}
        labels={s:labels_cpu[s].to(e.device) for s in range(3)}
        idx={}
        for s in range(3):
            o=torch.randperm(784,generator=gpart[s]); r=torch.rand(784,generator=gpart[s])
            for k in KS: idx[s,k]=partial_index(o,r,k).to(e.device)
        orderstack=torch.stack([orders[m["seed"]] for m in MODELS],1).to(e.device)
        floors={s:float(torch.bincount(labels_cpu[s],minlength=10).max())/B.N for s in range(3)}
        for j,m in enumerate(MODELS):
            s=m["seed"]; e.cx[j].copy_(xs[s] if m["k"]==0 else xs[s][:,idx[s,m["k"]]])
            e.cy[j].copy_(torch.nn.functional.one_hot(labels[s],10))
        e.acc.zero_();e.ce.zero_()
        for step in range(0,B.STEPS,B.BLOCK):
            e.indices.copy_(orderstack[step:step+B.BLOCK]);e.replay_block()
        met,units=e.evaluate()
        online_acc=(e.acc/B.STEPS).cpu().numpy();online_ce=(e.ce/B.STEPS).cpu().numpy()
        assert np.isfinite(online_ce).all() and all(bool(torch.isfinite(q).all()) for q in e.p),("NONFINITE",task)
        for j,m in enumerate(MODELS):
            rows.append(dict(**m,task=task,floor_acc=floors[m["seed"]],online_acc=float(online_acc[j]),
                             online_ce=float(online_ce[j]),**{kk:float(v[j]) for kk,v in met.items()}))
        for kk in KEEP_UNITS:unitout[f"{kk}_t{task}"]=units[kk]
        if task%10==0 or task==tasks:B.csvwrite(out/"rows.csv",rows)
        smp={(m["k"],m["act"]):online_acc[j] for j,m in enumerate(MODELS) if m["seed"]==0}
        print(f'TASK {task}/{tasks} {time.monotonic()-t0:.0f}s s0 GELU k0/107/784='
              f'{smp[0,"GELU"]:.3f}/{smp[107,"GELU"]:.3f}/{smp[784,"GELU"]:.3f} R k0/784={smp[0,"R"]:.3f}/{smp[784,"R"]:.3f}',flush=True)
    B.csvwrite(out/"rows.csv",rows)
    np.savez_compressed(out/"units.npz",**unitout)
    dev=torch.cuda.get_device_name(e.device) if e.device.type=="cuda" else str(e.device)
    (out/"provenance.json").write_text(json.dumps(dict(prereg_commit=PREREG_COMMIT,parent_result=PARENT_RESULT,
        spec_sha256=B.sha(ROOT/f"specs/spec_{NAME}.md"),code_sha256=B.sha(Path(__file__)),parent_code_sha256=B.sha(Path(B.__file__)),
        git_hash=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        data_sha256={n:B.sha(B.DATA/n) for n in B.DATA_FILES},subset_sha256={s:B.arrsha(v) for s,v in subset.items()},
        torch_version=torch.__version__,device=dev,dtype="float32",tf32=False,deterministic=True,models=MODELS,ks=KS,
        tasks=tasks,steps_per_task=B.STEPS,images=B.N,batch=B.BATCH,late_window=list(LATE),units_saved=KEEP_UNITS,
        wall_seconds=time.monotonic()-t0),indent=2))

# ---------------------------------------------------------------------- report
def read_rows(p):
    with open(p,newline="") as f:rows=list(csv.DictReader(f))
    for r in rows:
        for kk,v in list(r.items()):
            if kk in ("act","env"):continue
            r[kk]=int(v) if kk in ("seed","task","k") else float(v)
    return rows

def report(out_dir=OUT):
    out=B.output_path(out_dir); rows=read_rows(out/"rows.csv")
    assert len(rows)==len(MODELS)*TASKS,len(rows)
    by={}
    for r in rows:by.setdefault((r["seed"],r["k"],r["act"]),[]).append(r)
    for kk in by:by[kk].sort(key=lambda r:r["task"]);assert [r["task"] for r in by[kk]]==list(range(1,TASKS+1)),kk
    # G1
    old=[o for o in csv.DictReader(io.StringIO(subprocess.check_output(
        ["git","show",f"{PARENT_RESULT}:results/relu_gelu_silu_rl_ext150_0914/rows.csv"],cwd=ROOT,text=True))) if o["env"]=="RL"]
    worst=0.;n=0
    for o in old:
        r=by[(int(o["seed"]),0,o["act"])][int(o["task"])-1]
        for kk in ("online_acc","train_ce"):worst=max(worst,abs(r[kk]-float(o[kk])));n+=1
    g1=dict(max_abs=worst,compared=n,expected=15*TASKS*2)
    # levels
    levels=[]
    for m in MODELS:
        rr=by[(m["seed"],m["k"],m["act"])];late=[r for r in rr if LATE[0]<=r["task"]<=LATE[1]]
        fl=np.asarray([r["floor_acc"] for r in late]);thr=float(fl.mean()+3*fl.std(ddof=1)/math.sqrt(10))
        la=float(np.mean([r["online_acc"] for r in late]))
        levels.append(dict(**m,late_online_acc=la,floor_thr=thr,floor_verdict="AT_FLOOR" if la<=thr else "ABOVE_FLOOR",
                           first_floor_task=next((r["task"] for r in rr if r["online_acc"]<=thr),""),
                           neggate_l2_t50=rr[49]["neggate_l2"],neggate_l2_t100=rr[99]["neggate_l2"],neggate_l2_t150=rr[149]["neggate_l2"],
                           tinygate_l2_t150=rr[149]["tinygate_l2"],zmed_l2_t150=rr[149]["zmed_l2"]))
    B.csvwrite(out/"levels.csv",levels)
    S={(a,k):sum(1 for r in levels if r["act"]==a and r["k"]==k and r["floor_verdict"]=="ABOVE_FLOOR") for a in ACTS for k in KS}
    lab={a:("KICK_RESCUES" if S[a,784]>S[a,0] else "KICK_KILLS" if S[a,784]<S[a,0] else "KICK_NEUTRAL") for a in ACTS}
    g0=S["LR",0]==3 and S["ELU1",0]==0
    # G2: LR layer-1 normalised kick variance, monotone in k
    U=np.load(out/"units.npz")
    var={}
    for k in KS:
        d=[]
        for s in range(3):
            j=MODELS.index(dict(seed=s,k=k,act="LR"))
            Z=np.stack([U[f"zmean_l1_t{t}"][j] for t in range(1,TASKS+1)]);C=np.stack([U[f"cnorm_l1_t{t}"][j] for t in range(1,TASKS+1)])
            d.append(((Z[1:]-Z[:-1])/C[:-1]).ravel())
        var[k]=float(np.concatenate(d).var())
    g2=all(var[a]<var[b] for a,b in zip(KS[:-1],KS[1:]))
    exc={k:math.sqrt(max(var[k]-var[0],0.)) for k in KS}; ratio={k:exc[k]/exc[784] for k in KS}
    if not g2:q1="NOT_TESTABLE"
    elif S["LR",0]==3 and all(S["LR",k]==3 for k in KS) and all(S[a,k]==0 for a in ACTS if a!="LR" for k in KS if k>0):q1="LEAK_ONLY_SURVIVES"
    elif (lab["R"]=="KICK_RESCUES" or lab["ELU1"]=="KICK_RESCUES") and lab["GELU"]=="KICK_KILLS":q1="KICK_CROSSING"
    else:q1="OTHER"
    # ReLU revival by k
    rev={}
    for k in KS:
        for l in (1,2):
            dead=revd=0
            for s in range(3):
                j=MODELS.index(dict(seed=s,k=k,act="R"))
                D=np.stack([U[f"tinygate_l{l}_t{t}"][j] for t in range(1,TASKS+1)])>=1.0
                dead+=int(D[:-1].sum());revd+=int((D[:-1]&~D[1:]).sum())
            rev[k,l]=(dead,revd)
    verdict=[dict(section="G0",verdict="PASS" if g0 else "FAIL"),
             dict(section="G1",verdict="BIT_IDENTICAL" if worst==0 and n==g1["expected"] else "DIFFERS",value=worst,detail=f'{n}/{g1["expected"]}'),
             dict(section="G2",verdict="PASS" if g2 else "FAIL",detail=json.dumps(var))]
    verdict+=[dict(section="ARM",arm=a,verdict=lab[a],detail=" ".join(str(S[a,k]) for k in KS)) for a in ACTS]
    verdict.append(dict(section="Q1",verdict=q1));B.csvwrite(out/"verdict.csv",verdict)
    L=[f"# {NAME} report","",f"G0: **{'PASS' if g0 else 'FAIL'}**  G1: max|Δ| {worst:.3g} over {n}/{g1['expected']}  G2: **{'PASS' if g2 else 'FAIL'}**","",
       f"## Q1: **{q1}**","","## 生存数 S(a,k)（t141–150 で床の上の seed 数）と腕ごとのラベル","",
       "|活性化|k=0|k=25|k=107|k=270|k=784|ラベル|","|---|---|---|---|---|---|---|"]
    for a in ACTS:L.append(f"|{a}|"+"|".join(str(S[a,k]) for k in KS)+f"|**{lab[a]}**|")
    L+=["","## late 精度（t141–150・3 seed 平均）","","|活性化|"+"|".join(f"k={k}" for k in KS)+"|","|---|"+"---:|"*len(KS)]
    for a in ACTS:L.append(f"|{a}|"+"|".join(f'{np.mean([r["late_online_acc"] for r in levels if r["act"]==a and r["k"]==k]):.3f}' for k in KS)+"|")
    L+=["","## 最初に床を割ったタスク（seed 0/1/2）","","|活性化|"+"|".join(f"k={k}" for k in KS)+"|","|---|"+"---|"*len(KS)]
    for a in ACTS:
        L.append(f"|{a}|"+"|".join(" / ".join(str(r["first_floor_task"]) or "—" for r in sorted([r for r in levels if r["act"]==a and r["k"]==k],key=lambda r:r["seed"])) for k in KS)+"|")
    L+=["","## G2: LR 第 1 層の正規化蹴り","","|k|分散|k=0 に対する超過 sd の比（算術の目標）|","|---|---:|---|"]
    for k,t in zip(KS,(0,.25,.5,.75,1)):L.append(f"|{k}|{var[k]:.5g}|{ratio[k]:.3f}（{t}）|")
    L+=["","## ReLU の復活（死 = 1200 枚全部で φ′=0、復活 = 次タスクで生）","","|k|第 1 層 死/復活/率|第 2 層 死/復活/率|","|---|---|---|"]
    for k in KS:L.append(f"|{k}|"+"|".join(f"{rev[k,l][0]} / {rev[k,l][1]} / {rev[k,l][1]/rev[k,l][0] if rev[k,l][0] else float('nan'):.3f}" for l in (1,2))+"|")
    L+=["","## 谷型の第 2 層（3 seed 平均）","","|活性化|k|φ′<0 t50/t100/t150|\\|φ′\\|<1e−8 t150|median z̄₂ t150|","|---|---|---|---:|---:|"]
    for a in ("GELU","SILU"):
        for k in KS:
            rr=[r for r in levels if r["act"]==a and r["k"]==k]
            L.append(f'|{a}|{k}|{np.mean([r["neggate_l2_t50"] for r in rr]):.2f} / {np.mean([r["neggate_l2_t100"] for r in rr]):.2f} / {np.mean([r["neggate_l2_t150"] for r in rr]):.2f}|{np.mean([r["tinygate_l2_t150"] for r in rr]):.2f}|{np.mean([r["zmed_l2_t150"] for r in rr]):.2f}|')
    (out/"summary.md").write_text("\n".join(L)+"\n",encoding="utf-8");print((out/"summary.md").read_text(),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--device",default="cuda");ap.add_argument("--tasks",type=int,default=TASKS)
    ap.add_argument("--out-dir",default=f"results/{NAME}");ap.add_argument("--report",action="store_true");a=ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.set_num_threads(1)
    if a.report:report(a.out_dir)
    else:run(a.device,a.tasks,a.out_dir)

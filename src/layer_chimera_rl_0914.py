#!/usr/bin/env python3
"""Preregistered layerwise activation chimera experiment in PM and RL."""
from __future__ import annotations
import argparse, csv, gzip, hashlib, io, json, math, os, subprocess, time
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/layer_chimera_rl_0914"
DATA=Path("/home/issan/Projects/claude/proj_004_drift/data/mnist")
DIMS=(784,100,100,10)
BATCH=16; N=1200; STEPS=6000; TASKS=50; BLOCK=25
MODELS=[dict(seed=s,env=e,act1=a1,act2=a2) for s in range(3) for e in ("RL","PM")
        for (a1,a2) in (("ELU1","ELU1"),("ELU1","LR"),("LR","ELU1"),("LR","LR"))]
PREREG_COMMIT="fb5ef5226c2897888cfa2fe266aca191c1b1ceae"
HOST_SHA256="6e0d319603000b2f6e83943a8f22ab62f900d95e6f309c55556c7883089e3a4c"
DATA_FILES=("train-images-idx3-ubyte.gz","train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz","t10k-labels-idx1-ubyte.gz")

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrsha(a): return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def stream(role,seed):
    h=hashlib.sha256(f"pmnist_0905|{role}|{seed}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(h[:8],"little")&((1<<63)-1))
def initial(seed):
    g=stream("init",seed); p=[]
    for din,dout in zip(DIMS[:-1],DIMS[1:]):
        p += [(torch.rand(dout,din,generator=g)*2-1)*(1.0/math.sqrt(din)),
              (torch.rand(dout,generator=g)*2-1)*(1.0/math.sqrt(din))]
    return p
def read_idx(p):
    with gzip.open(p,"rb") as f: a=f.read()
    dim=a[3]; shape=[int.from_bytes(a[4+4*i:8+4*i],"big") for i in range(dim)]
    return np.frombuffer(a[4+4*dim:],np.uint8).reshape(shape).copy()
def csvwrite(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def activ(z,elu):
    neg=torch.where(elu,torch.expm1(z.clamp_max(0)),z*.1)
    return torch.where(z>0,z,neg)
def gate(z,elu):
    return torch.where(z>0,torch.ones_like(z),torch.where(elu,z.clamp_max(0).exp(),torch.full_like(z,.1)))
def forward(p,x,elu1,elu2):
    z1=torch.bmm(x,p[0].transpose(1,2))+p[1][:,None,:]
    a1=activ(z1,elu1)
    z2=torch.bmm(a1,p[2].transpose(1,2))+p[3][:,None,:]
    a2=activ(z2,elu2)
    z3=torch.bmm(a2,p[4].transpose(1,2))+p[5][:,None,:]
    return z1,a1,z2,a2,z3
def gradients(p,x,y,elu1,elu2):
    z1,a1,z2,a2,z3=forward(p,x,elu1,elu2)
    g3=(z3.softmax(-1)-y)/x.shape[1]
    g2=torch.bmm(g3,p[4])*gate(z2,elu2)
    g1=torch.bmm(g2,p[2])*gate(z1,elu1)
    gr=[torch.bmm(g1.transpose(1,2),x),g1.sum(1),
        torch.bmm(g2.transpose(1,2),a1),g2.sum(1),
        torch.bmm(g3.transpose(1,2),a2),g3.sum(1)]
    ce=(z3.logsumexp(-1)-(z3*y).sum(-1)).mean(-1)
    acc=(z3.argmax(-1)==y.argmax(-1)).float().mean(-1)
    return gr,ce,acc

class Engine:
    def __init__(self,models=MODELS,device="cuda"):
        self.models=models;self.device=torch.device(device);self.M=len(models)
        self.p=[torch.stack([initial(m["seed"])[i] for m in models]).to(self.device) for i in range(6)]
        self.m=[torch.zeros_like(q) for q in self.p];self.v=[torch.zeros_like(q) for q in self.p]
        self.t=torch.zeros((),device=self.device)
        self.elu1=torch.tensor([m["act1"]=="ELU1" for m in models],device=self.device)[:,None,None]
        self.elu2=torch.tensor([m["act2"]=="ELU1" for m in models],device=self.device)[:,None,None]
        self.acc=torch.zeros(self.M,device=self.device);self.ce=torch.zeros_like(self.acc)
        self.cx=torch.zeros(self.M,N,784,device=self.device);self.cy=torch.zeros(self.M,N,10,device=self.device)
        self.indices=torch.zeros(BLOCK,self.M,BATCH,dtype=torch.long,device=self.device)
        self.mid=torch.arange(self.M,device=self.device)[:,None]
        self.graph=None
    @torch.no_grad()
    def step(self,x,y):
        gr,ce,acc=gradients(self.p,x,y,self.elu1,self.elu2)
        self.ce.add_(ce);self.acc.add_(acc);self.t.add_(1)
        c1=1-torch.pow(.9,self.t);c2=1-torch.pow(.999,self.t)
        for p,g,m,v in zip(self.p,gr,self.m,self.v):
            m.mul_(.9).add_(g,alpha=.1);v.mul_(.999).addcmul_(g,g,value=.001)
            p.addcdiv_(m/c1,(v/c2).sqrt()+1e-8,value=-.001)
    @torch.no_grad()
    def block(self):
        for k in range(BLOCK):self.step(self.cx[self.mid,self.indices[k]],self.cy[self.mid,self.indices[k]])
    def snapshot(self):
        return [q.clone() for q in self.p+self.m+self.v+[self.t,self.acc,self.ce]]
    @torch.no_grad()
    def restore(self,s):
        for q,v in zip(self.p+self.m+self.v+[self.t,self.acc,self.ce],s):q.copy_(v)
    def capture(self):
        if self.device.type!="cuda" or not torch.cuda.is_available():
            self.graph=None
            return
        snap=self.snapshot()
        stream_cuda=torch.cuda.Stream()
        stream_cuda.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream_cuda):
            for _ in range(2):self.block()
        torch.cuda.current_stream().wait_stream(stream_cuda)
        self.restore(snap);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.block()
        self.restore(snap)
    def replay_block(self):
        if self.graph is None:self.block()
        else:self.graph.replay()
    @torch.no_grad()
    def evaluate(self):
        z1,a1,z2,a2,logits=forward(self.p,self.cx,self.elu1,self.elu2)
        ce=(logits.logsumexp(-1)-(logits*self.cy).sum(-1)).mean(-1)
        acc=(logits.argmax(-1)==self.cy.argmax(-1)).float().mean(-1)
        out=dict(train_acc=acc.cpu().numpy(),train_ce=ce.cpu().numpy())
        out["a1_rms"]=a1.square().mean(dim=(1,2)).sqrt().cpu().numpy()
        units={}
        for l,z,elu in [(1,z1,self.elu1),(2,z2,self.elu2)]:
            g=gate(z,elu)
            for key, val in {"zmean":z.mean(1),"zstd":z.std(1,unbiased=False),"gate_mean":g.mean(1),"gate_rms":g.square().mean(1).sqrt(),
                             "lowgate":(g<.05).float().mean(1),"nearzero":(z.abs()<1).float().mean(1)}.items():
                units[f"{key}_l{l}"]=val.cpu().numpy()
        units["cnorm_l1"]=(self.p[0]-self.p[0].mean(-1,keepdim=True)).norm(dim=-1).cpu().numpy()
        units["rowmean_l1"]=self.p[0].mean(-1).cpu().numpy()
        for l in range(1,4):out[f"wnorm_l{l}"]=self.p[2*(l-1)].norm(dim=-1).mean(-1).cpu().numpy()
        out["cnorm_l1"]=units["cnorm_l1"].mean(-1)
        out["lowgate_l1"]=units["lowgate_l1"].mean(-1)
        out["nearzero_l1"]=units["nearzero_l1"].mean(-1)
        return out,units

def arm_name(act1,act2):
    return ("E" if act1=="ELU1" else "L")+("E" if act2=="ELU1" else "L")

def output_path(out_dir):
    p=Path(out_dir)
    return p if p.is_absolute() else ROOT/p

def read_rows(path):
    with open(path,newline="") as f:rows=list(csv.DictReader(f))
    strings={"env","act1","act2"}
    for r in rows:
        for k,v in list(r.items()):
            if k in strings:continue
            r[k]=int(v) if k in ("seed","task") else float(v)
    return rows

def floor_rollup(levels,arm):
    rr=sorted([r for r in levels if r["env"]=="RL" and r["arm"]==arm],key=lambda r:r["seed"])
    assert [r["seed"] for r in rr]==[0,1,2],(arm,rr)
    calls=[r["floor_verdict"] for r in rr]
    if all(v=="ABOVE_FLOOR" for v in calls):
        return "ABOVE_FLOOR","all 3 seeds ABOVE_FLOOR",calls
    if all(v=="AT_FLOOR" for v in calls):
        return "AT_FLOOR","all 3 seeds AT_FLOOR",calls
    above=[str(r["seed"]) for r in rr if r["floor_verdict"]=="ABOVE_FLOOR"]
    at=[str(r["seed"]) for r in rr if r["floor_verdict"]=="AT_FLOOR"]
    detail=f'{len(above)}/3 seeds ABOVE_FLOOR (seed {",".join(above)}); seed {",".join(at)} AT_FLOOR'
    return "DISAGREEMENT",detail,calls

def anchor_comparison(rows,act1,act2,historical_act):
    raw=subprocess.check_output(
        ["git","show","46becd9:results/elu_environment_0913/rows.csv"],cwd=ROOT,text=True)
    old=list(csv.DictReader(io.StringIO(raw)))
    oldmap={(int(r["seed"]),int(r["task"])):float(r["online_acc"]) for r in old
            if r["env"]=="RL" and r["act"]==historical_act and r["iv"]=="ref"}
    newmap={(r["seed"],r["task"]):r["online_acc"] for r in rows
            if r["env"]=="RL" and r["act1"]==act1 and r["act2"]==act2}
    expected={(s,t) for s in range(3) for t in range(1,TASKS+1)}
    assert set(oldmap)==expected and set(newmap)==expected,(historical_act,len(oldmap),len(newmap))
    byseed={s:max(abs(newmap[s,t]-oldmap[s,t]) for t in range(1,TASKS+1)) for s in range(3)}
    overall=max(byseed.values())
    return dict(overall=overall,byseed=byseed,flag="ANCHOR_DRIFT" if overall>.01 else "OK")

def report(out_dir=OUT):
    out=output_path(out_dir)
    rows=read_rows(out/"rows.csv")
    assert len(rows)==len(MODELS)*TASKS,len(rows)
    expected={(m["seed"],m["env"],m["act1"],m["act2"],t) for m in MODELS for t in range(1,TASKS+1)}
    assert {(r["seed"],r["env"],r["act1"],r["act2"],r["task"]) for r in rows}==expected
    for s in range(3):
        for env in ("RL","PM"):
            for task in range(1,TASKS+1):
                vals=[r["floor_acc"] for r in rows if r["seed"]==s and r["env"]==env and r["task"]==task]
                assert len(vals)==4 and max(vals)-min(vals)<1e-12,(s,env,task,vals)
    levels=[]
    for m in MODELS:
        rr=sorted([r for r in rows if all(r[k]==v for k,v in m.items())],key=lambda r:r["task"])
        assert [r["task"] for r in rr]==list(range(1,TASKS+1)),m
        late=[r for r in rr if 41<=r["task"]<=50]
        floors=np.asarray([r["floor_acc"] for r in late],float)
        floor_F=float(floors.mean());floor_s=float(floors.std(ddof=1))
        floor_thr=floor_F+3*floor_s/math.sqrt(10)
        late_acc=float(np.mean([r["online_acc"] for r in late]))
        levels.append(dict(**m,arm=arm_name(m["act1"],m["act2"]),late_online_acc=late_acc,
                           floor_F=floor_F,floor_s=floor_s,floor_thr=floor_thr,
                           floor_verdict="AT_FLOOR" if late_acc<=floor_thr else "ABOVE_FLOOR"))
    csvwrite(out/"levels.csv",levels)

    rollups={a:floor_rollup(levels,a) for a in ("EE","EL","LE")}
    if rollups["EE"][0]=="ABOVE_FLOOR":
        q1="NOT_REPRODUCED"
        q1_reason="RL EE was ABOVE_FLOOR for all 3 seeds; all mechanism labels are NOT_TESTABLE."
    elif rollups["EE"][0]!="AT_FLOOR":
        q1="UNRESOLVED"
        q1_reason="RL EE cross-seed floor calls disagree: "+rollups["EE"][1]+"."
    elif rollups["EL"][0]=="DISAGREEMENT" or rollups["LE"][0]=="DISAGREEMENT":
        q1="UNRESOLVED"
        bad=[a+": "+rollups[a][1] for a in ("EL","LE") if rollups[a][0]=="DISAGREEMENT"]
        q1_reason="RL cross-seed floor calls disagree for "+"; ".join(bad)+"."
    else:
        q1={
            ("ABOVE_FLOOR","ABOVE_FLOOR"):"GROWING_INPUT_LAYER",
            ("ABOVE_FLOOR","AT_FLOOR"):"FLOOR_IN_DEEP_LAYER",
            ("AT_FLOOR","AT_FLOOR"):"EITHER_ELU_KILLS",
            ("AT_FLOOR","ABOVE_FLOOR"):"FLOOR_IN_SHALLOW_LAYER",
        }[(rollups["EL"][0],rollups["LE"][0])]
        q1_reason=f'RL EE was AT_FLOOR; EL was {rollups["EL"][0]} and LE was {rollups["LE"][0]}.'

    with np.load(out/"units.npz") as units:
        z2={t:np.asarray(units[f"zstd_l2_t{t}"]) for t in (2,8)}
    for t in (2,8):assert z2[t].shape==(len(MODELS),DIMS[2]),(t,z2[t].shape)
    growth={}
    for arm in ("EE","EL","LE","LL"):
        inds=[j for j,m in enumerate(MODELS) if m["env"]=="RL" and arm_name(m["act1"],m["act2"])==arm]
        assert len(inds)==3,(arm,inds)
        sd2={t:float(np.mean([np.median(z2[t][j]) for j in inds])) for t in (2,8)}
        assert sd2[2]>0 and sd2[8]>0,(arm,sd2)
        growth[arm]=dict(sd2_t2=sd2[2],sd2_t8=sd2[8],g=(sd2[8]/sd2[2])**(1/6))
    boundary=math.sqrt(growth["EE"]["g"]*growth["LL"]["g"])
    for arm in growth:growth[arm]["width_class"]="HIGH" if growth[arm]["g"]>=boundary else "LOW"
    q2=("WIDTH_SET_BY_L1" if growth["EL"]["width_class"]=="HIGH" and growth["LE"]["width_class"]=="LOW" else
        "WIDTH_SET_BY_L2" if growth["EL"]["width_class"]=="LOW" and growth["LE"]["width_class"]=="HIGH" else
        "WIDTH_MIXED")

    pm_late={arm:float(np.mean([r["late_online_acc"] for r in levels if r["env"]=="PM" and r["arm"]==arm]))
             for arm in ("EE","EL","LE","LL")}
    anchors={"EE_vs_RL_ELU1_ref":anchor_comparison(rows,"ELU1","ELU1","ELU1"),
             "LL_vs_RL_LR_ref":anchor_comparison(rows,"LR","LR","LR")}
    verdict=[]
    for arm in ("EE","EL","LE"):
        status,detail,calls=rollups[arm]
        verdict.append(dict(section="Q1_LEVEL",arm=arm,seed0=calls[0],seed1=calls[1],seed2=calls[2],
                            verdict=status,detail=detail))
    for arm in ("EE","EL","LE","LL"):
        verdict.append(dict(section="Q2_GROWTH",arm=arm,value=growth[arm]["g"],boundary=boundary,
                            verdict=growth[arm]["width_class"],detail=f'sd2_t2={growth[arm]["sd2_t2"]}; sd2_t8={growth[arm]["sd2_t8"]}'))
    verdict.append(dict(section="Q1",arm="ALL",verdict=q1,detail=q1_reason))
    verdict.append(dict(section="Q2",arm="ALL",verdict=q2,detail="Mechanical HIGH/LOW classification of RL EL and LE."))
    csvwrite(out/"verdict.csv",verdict)

    lines=["# Layerwise activation chimera RL/PM report","",f"Q1: **{q1}**",q1_reason,"",
           "## RL floor calls","","|Arm|Seed 0|Seed 1|Seed 2|Arm rollup|","|---|---|---|---|---|"]
    for arm in ("EE","EL","LE"):
        status,detail,calls=rollups[arm]
        lines.append(f"|{arm}|{calls[0]}|{calls[1]}|{calls[2]}|{status} ({detail})|")
    lines += ["",f"Q2: **{q2}**",f"HIGH/LOW boundary = sqrt(g_EE * g_LL) = {boundary:.9g}.","",
              "|Arm|sd2(t2)|sd2(t8)|g|Class|","|---|---:|---:|---:|---|"]
    for arm in ("EE","EL","LE","LL"):
        g=growth[arm]
        lines.append(f'|{arm}|{g["sd2_t2"]:.9g}|{g["sd2_t8"]:.9g}|{g["g"]:.9g}|{g["width_class"]}|')
    lines += ["","## Q3: PM late online accuracy","","Mean across 3 seeds; late window is tasks 41–50.","",
              "|Arm|Late online accuracy|","|---|---:|"]
    for arm in ("EE","EL","LE","LL"):lines.append(f"|{arm}|{pm_late[arm]:.9g}|")
    lines += ["","## G1 historical anchors","","|Comparison|Overall max abs delta|Seed 0|Seed 1|Seed 2|Flag|",
              "|---|---:|---:|---:|---:|---|"]
    for name,a in anchors.items():
        lines.append(f'|{name}|{a["overall"]:.9g}|{a["byseed"][0]:.9g}|{a["byseed"][1]:.9g}|{a["byseed"][2]:.9g}|{a["flag"]}|')
    lines += ["","Anchor flags are diagnostic and do not change Q1 or Q2.","","## Interpretation notes","",
              "- Q1 arm rollup resolves the preregistration gap as follows: ABOVE_FLOOR requires all 3 seeds individually ABOVE_FLOOR; AT_FLOOR requires all 3 individually AT_FLOOR; any disagreement makes that arm unresolved and forces Q1 UNRESOLVED when the arm is needed.",
              "- Floor F, sample standard deviation (ddof=1), and threshold F + 3*s/sqrt(10) are computed separately for each seed/environment from that run's actual task 41–50 labels; PM calls are reported in levels.csv but do not enter Q1.",
              "- Q2 is descriptive, not a causal test of width causing collapse. PM ceiling behavior is not generalized. Forward activation and backward gate are coupled within each layer, so their separate effects are not identified.",
              "- This scope is 3 seeds, one 80-epoch box; it is distinct from the historical 400-epoch RL setting.",""]
    (out/"summary.md").write_text("\n".join(lines),encoding="utf-8")
    print((out/"summary.md").read_text(),flush=True)

def run(device="cuda",tasks=TASKS,steps=STEPS,out_dir=OUT):
    assert 1<=tasks<=TASKS,tasks
    assert steps>0 and steps%BLOCK==0,(steps,BLOCK)
    requested=torch.device(device)
    if requested.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    out=output_path(out_dir);out.mkdir(parents=True,exist_ok=True)
    subprocess.run(["git","merge-base","--is-ancestor",PREREG_COMMIT,"HEAD"],cwd=ROOT,check=True)
    registered=subprocess.check_output(["git","show",PREREG_COMMIT+":specs/spec_layer_chimera_rl_0914.md"],cwd=ROOT)
    assert hashlib.sha256(registered).hexdigest()==sha(ROOT/"specs/spec_layer_chimera_rl_0914.md")
    e=Engine(device=device);gperm={s:stream("env_perm_0913",s) for s in range(3)}
    glabel={s:stream("env_labels_0913",s) for s in range(3)}
    gbatch={s:stream("env_batch_0913",s) for s in range(3)}
    ax=read_idx(DATA/"train-images-idx3-ubyte.gz").reshape(-1,784).astype(np.float32)/255
    ay=read_idx(DATA/"train-labels-idx1-ubyte.gz").astype(np.int64)
    subset={s:torch.randperm(len(ax),generator=stream("rl_subset",s))[:N].numpy() for s in range(3)}
    xs={s:torch.tensor(ax[subset[s]],device=e.device) for s in range(3)}
    ys_cpu={s:torch.tensor(ay[subset[s]]) for s in range(3)}
    ys={s:ys_cpu[s].to(e.device) for s in range(3)}
    rows=[];unitout={}
    e.capture();t0=time.monotonic()
    for task in range(1,tasks+1):
        epochs_needed=-(-(steps*BATCH)//N)
        orders={}
        for s in range(3):
            flat=torch.stack([torch.randperm(N,generator=gbatch[s]) for _ in range(epochs_needed)]).reshape(-1)[:steps*BATCH]
            orders[s]=flat.reshape(steps,BATCH)
        perm_cpu={s:torch.randperm(784,generator=gperm[s]) for s in range(3)}
        perm={s:perm_cpu[s].to(e.device) for s in range(3)}
        labels_cpu={s:torch.randint(10,(N,),generator=glabel[s]) for s in range(3)}
        labels={s:labels_cpu[s].to(e.device) for s in range(3)}
        orderstack=torch.stack([orders[m["seed"]] for m in MODELS],1).to(e.device)
        floors={}
        for s in range(3):
            floors[s,"PM"]=float(torch.bincount(ys_cpu[s],minlength=10).max())/N
            floors[s,"RL"]=float(torch.bincount(labels_cpu[s],minlength=10).max())/N
        for j,m in enumerate(MODELS):
            s=m["seed"];e.cx[j].copy_(xs[s][:,perm[s]] if m["env"]=="PM" else xs[s])
            yy=ys[s] if m["env"]=="PM" else labels[s]
            e.cy[j].copy_(torch.nn.functional.one_hot(yy,10))
        e.acc.zero_();e.ce.zero_()
        for step in range(0,steps,BLOCK):
            e.indices.copy_(orderstack[step:step+BLOCK]);e.replay_block()
        met,units=e.evaluate()
        online_acc=(e.acc/steps).cpu().numpy();online_ce=(e.ce/steps).cpu().numpy()
        assert np.isfinite(online_ce).all() and all(bool(torch.isfinite(q).all()) for q in e.p),("NONFINITE",task)
        for j,m in enumerate(MODELS):
            rows.append(dict(**m,task=task,floor_acc=floors[m["seed"],m["env"]],
                             online_acc=float(online_acc[j]),online_ce=float(online_ce[j]),
                             **{k:float(v[j]) for k,v in met.items()}))
        for k,v in units.items():unitout[f"{k}_t{task}"]=v
        csvwrite(out/"rows.csv",rows)
        sample={arm_name(m["act1"],m["act2"]):online_acc[j] for j,m in enumerate(MODELS)
                if m["seed"]==0 and m["env"]=="RL"}
        print(f'TASK {task}/{tasks} elapsed={time.monotonic()-t0:.1f}s RL_EE={sample["EE"]:.4f} RL_EL={sample["EL"]:.4f} RL_LE={sample["LE"]:.4f} RL_LL={sample["LL"]:.4f}',flush=True)
    np.savez_compressed(out/"units.npz",**unitout)
    host_sha=sha(ROOT/"src/elu_environment_0913.py");assert host_sha==HOST_SHA256,host_sha
    data_paths=[DATA/name for name in DATA_FILES];assert all(p.exists() for p in data_paths)
    device_name=torch.cuda.get_device_name(e.device) if e.device.type=="cuda" and torch.cuda.is_available() else str(e.device)
    provenance=dict(prereg_commit=PREREG_COMMIT,spec_sha256=sha(ROOT/"specs/spec_layer_chimera_rl_0914.md"),
        code_sha256=sha(Path(__file__)),host_sha256=host_sha,
        data_sha256={p.name:sha(p) for p in data_paths},subset_sha256={s:arrsha(v) for s,v in subset.items()},
        initial_sha256={s:[arrsha(v.numpy()) for v in initial(s)] for s in range(3)},torch_version=torch.__version__,
        device=device_name,dtype="float32",tf32=False,deterministic=True,num_threads=torch.get_num_threads(),models=MODELS,
        tasks=tasks,steps_per_task=steps,images=N,batch=BATCH,late_window=[41,50],wall_seconds=time.monotonic()-t0)
    (out/"provenance.json").write_text(json.dumps(provenance,indent=2))

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--device",default="cuda");ap.add_argument("--tasks",type=int,default=TASKS)
    ap.add_argument("--steps",type=int,default=STEPS);ap.add_argument("--out-dir",default="results/layer_chimera_rl_0914")
    ap.add_argument("--report",action="store_true");args=ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.set_num_threads(1)
    if args.report:report(args.out_dir)
    else:run(args.device,args.tasks,args.steps,args.out_dir)

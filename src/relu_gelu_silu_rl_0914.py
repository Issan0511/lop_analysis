#!/usr/bin/env python3
"""Preregistered: do ReLU, GELU and SiLU collapse in RL-MNIST? (spec_relu_gelu_silu_rl_0914.md)

Box identical to layer_chimera_rl_0914 / elu_environment_0913; the only change is the
activation set (same activation in both hidden layers): LR, ELU1, R, GELU, SILU.
The engine is layer_chimera_rl_0914's, with activ/gate generalised from a 2-way ELU/LR
mask to five masks.  For LR and ELU1 elements the forward and gate expressions are the
host's own, so their values are unchanged; only the batch size (24 -> 30) differs.
"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, io, json, math, os, subprocess, time
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
NAME="relu_gelu_silu_rl_0914"
OUT=ROOT/f"results/{NAME}"
DATA=Path("/home/issan/Projects/claude/proj_004_drift/data/mnist")
DIMS=(784,100,100,10)
BATCH=16; N=1200; STEPS=6000; TASKS=50; BLOCK=25
ACTS=("LR","ELU1","R","GELU","SILU")
MODELS=[dict(seed=s,env=e,act=a) for s in range(3) for e in ("RL","PM") for a in ACTS]
PREREG_COMMIT="f1fbaa60e65511739cc5e7a955b5ea7b67b2f111"
ANCHOR_COMMIT="58c1819"             # layer_chimera_rl_0914 results
DATA_FILES=("train-images-idx3-ubyte.gz","train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz","t10k-labels-idx1-ubyte.gz")
SQRT2=math.sqrt(2.0); INV_SQRT_2PI=1.0/math.sqrt(2.0*math.pi)
ADAM_EPS=1e-8

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

class Masks:
    """Per-model activation masks shaped [M,1,1]; LR is the default branch."""
    def __init__(self,acts,device):
        t=lambda a:torch.tensor([x==a for x in acts],device=device)[:,None,None]
        self.elu,self.r,self.gelu,self.silu=t("ELU1"),t("R"),t("GELU"),t("SILU")

def activ(z,A):
    # LR / ELU1 elements: the host's expression exactly (where() only selects).
    base=torch.where(z>0,z,torch.where(A.elu,torch.expm1(z.clamp_max(0)),
                                       torch.where(A.r,torch.zeros_like(z),z*.1)))
    gelu=z*0.5*(1.0+torch.erf(z/SQRT2))
    silu=z*torch.sigmoid(z)
    return torch.where(A.gelu,gelu,torch.where(A.silu,silu,base))

def gate(z,A):
    base=torch.where(z>0,torch.ones_like(z),torch.where(A.elu,z.clamp_max(0).exp(),
                     torch.where(A.r,torch.zeros_like(z),torch.full_like(z,.1))))
    gelu=0.5*(1.0+torch.erf(z/SQRT2))+z*torch.exp(-0.5*z*z)*INV_SQRT_2PI
    s=torch.sigmoid(z); silu=s*(1.0+z*(1.0-s))
    return torch.where(A.gelu,gelu,torch.where(A.silu,silu,base))

def forward(p,x,A):
    z1=torch.bmm(x,p[0].transpose(1,2))+p[1][:,None,:]
    a1=activ(z1,A)
    z2=torch.bmm(a1,p[2].transpose(1,2))+p[3][:,None,:]
    a2=activ(z2,A)
    z3=torch.bmm(a2,p[4].transpose(1,2))+p[5][:,None,:]
    return z1,a1,z2,a2,z3

def gradients(p,x,y,A):
    z1,a1,z2,a2,z3=forward(p,x,A)
    g3=(z3.softmax(-1)-y)/x.shape[1]
    g2=torch.bmm(g3,p[4])*gate(z2,A)
    g1=torch.bmm(g2,p[2])*gate(z1,A)
    gr=[torch.bmm(g1.transpose(1,2),x),g1.sum(1),
        torch.bmm(g2.transpose(1,2),a1),g2.sum(1),
        torch.bmm(g3.transpose(1,2),a2),g3.sum(1)]
    ce=(z3.logsumexp(-1)-(z3*y).sum(-1)).mean(-1)
    acc=(z3.argmax(-1)==y.argmax(-1)).float().mean(-1)
    return gr,ce,acc

# ------------------------------------------------------------------ self-test
def selftest():
    """float64, CPU.  Each check has a mutation control that must make it fail."""
    F=torch.nn.functional
    out={}
    z=torch.linspace(-40.0,12.0,400001,dtype=torch.float64); z=z[z!=0]
    kernels={"LR":lambda t:F.leaky_relu(t,0.1),"ELU1":F.elu,"R":F.relu,"GELU":F.gelu,"SILU":F.silu}
    for a in ACTS:
        A=Masks([a],"cpu"); zz=z[None,None,:]
        phi_err=float((activ(zz,A)[0,0]-kernels[a](z)).abs().max())
        zg=zz.clone().requires_grad_(True)
        auto=torch.autograd.grad(activ(zg,A).sum(),zg)[0][0,0]
        gate_err=float((gate(zz,A)[0,0]-auto).abs().max())
        mut_phi=float((activ(zz,A)[0,0]+1e-3-kernels[a](z)).abs().max())
        mut_gate=float((gate(zz,A)[0,0]*1.001+1e-6-auto).abs().max())
        z32=z.float()[None,None,:]
        phi32=float((activ(z32,A)[0,0]-kernels[a](z.float())).abs().max())
        out[a]=dict(phi_vs_kernel_f64=phi_err,gate_vs_autograd_f64=gate_err,
                    mutation_phi=mut_phi,mutation_gate=mut_gate,phi_vs_kernel_f32_report_only=phi32,
                    min_gate_f64=float(gate(zz,A).min()))
        assert phi_err<1e-12,(a,"phi disagrees with torch kernel",phi_err)
        assert gate_err<1e-12,(a,"gate disagrees with autograd",gate_err)
        assert mut_phi>1e-4 and mut_gate>1e-7,(a,"vacuous self-test",mut_phi,mut_gate)
    # the manual backprop (gradients) against autograd, all five activations in ONE batch
    g=torch.Generator().manual_seed(0)
    M=len(ACTS); A=Masks(list(ACTS),"cpu")
    p=[torch.stack([initial(0)[i].double()*3 for _ in range(M)]) for i in range(6)]
    x=torch.rand(M,16,784,generator=g,dtype=torch.float64)*2-1
    y=torch.nn.functional.one_hot(torch.randint(10,(M,16),generator=g),10).double()
    man,_,_=gradients(p,x,y,A)
    pg=[q.clone().requires_grad_(True) for q in p]
    z3=forward(pg,x,A)[-1]
    loss=(z3.logsumexp(-1)-(z3*y).sum(-1)).mean(-1).sum()
    auto=torch.autograd.grad(loss,pg)
    err=max(float((m_-a_).abs().max()) for m_,a_ in zip(man,auto))
    # mutation: flip the sign of GELU's gate -> manual gradients must disagree
    def gate_mut(zz,AA):
        gg=gate(zz,AA); return torch.where(AA.gelu,-gg,gg)
    z1,a1,z2,a2,z3m=forward(p,x,A)
    g3=(z3m.softmax(-1)-y)/x.shape[1]; g2=torch.bmm(g3,p[4])*gate_mut(z2,A)
    mut=float((torch.bmm(g2.transpose(1,2),a1)-auto[2]).abs().max())
    neg_share={a:float((gate(z2,A)[i]<0).double().mean()) for i,a in enumerate(ACTS)}
    out["backprop_vs_autograd_f64"]=err; out["backprop_mutation_gelu_sign"]=mut
    out["layer2_negative_gate_share_in_check"]=neg_share
    assert err<1e-10,("manual backprop disagrees with autograd",err)
    assert mut>1e-6,("vacuous backprop check: GELU gate sign flip not detected",mut)
    assert neg_share["GELU"]>0 and neg_share["SILU"]>0,("check never exercised the reversed region",neg_share)
    return out

# --------------------------------------------------------------------- engine
class Engine:
    def __init__(self,models=MODELS,device="cuda"):
        self.models=models;self.device=torch.device(device);self.M=len(models)
        self.p=[torch.stack([initial(m["seed"])[i] for m in models]).to(self.device) for i in range(6)]
        self.m=[torch.zeros_like(q) for q in self.p];self.v=[torch.zeros_like(q) for q in self.p]
        self.t=torch.zeros((),device=self.device)
        self.A=Masks([m["act"] for m in models],self.device)
        self.acc=torch.zeros(self.M,device=self.device);self.ce=torch.zeros_like(self.acc)
        self.cx=torch.zeros(self.M,N,784,device=self.device);self.cy=torch.zeros(self.M,N,10,device=self.device)
        self.indices=torch.zeros(BLOCK,self.M,BATCH,dtype=torch.long,device=self.device)
        self.mid=torch.arange(self.M,device=self.device)[:,None]
        self.graph=None
    @torch.no_grad()
    def step(self,x,y):
        gr,ce,acc=gradients(self.p,x,y,self.A)
        self.ce.add_(ce);self.acc.add_(acc);self.t.add_(1)
        c1=1-torch.pow(.9,self.t);c2=1-torch.pow(.999,self.t)
        for p,g,m,v in zip(self.p,gr,self.m,self.v):
            m.mul_(.9).add_(g,alpha=.1);v.mul_(.999).addcmul_(g,g,value=.001)
            p.addcdiv_(m/c1,(v/c2).sqrt()+ADAM_EPS,value=-.001)
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
            self.graph=None;return
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
        z1,a1,z2,a2,logits=forward(self.p,self.cx,self.A)
        ce=(logits.logsumexp(-1)-(logits*self.cy).sum(-1)).mean(-1)
        acc=(logits.argmax(-1)==self.cy.argmax(-1)).float().mean(-1)
        out=dict(train_acc=acc.cpu().numpy(),train_ce=ce.cpu().numpy())
        units={}
        for l,z in [(1,z1),(2,z2)]:
            g=gate(z,self.A)
            for key,val in {"zmean":z.mean(1),"zstd":z.std(1,unbiased=False),"gate_mean":g.mean(1),
                            "gate_rms":g.square().mean(1).sqrt(),"lowgate":(g.abs()<.05).float().mean(1),
                            "tinygate":(g.abs()<ADAM_EPS).float().mean(1),"neggate":(g<0).float().mean(1),
                            "nearzero":(z.abs()<1).float().mean(1)}.items():
                units[f"{key}_l{l}"]=val.cpu().numpy()
            for key in ("zmean","gate_mean","lowgate","tinygate","neggate"):
                out[f"{key}_l{l}"]=units[f"{key}_l{l}"].mean(-1)
            out[f"zmed_l{l}"]=np.median(units[f"zmean_l{l}"],axis=-1)
        units["cnorm_l1"]=(self.p[0]-self.p[0].mean(-1,keepdim=True)).norm(dim=-1).cpu().numpy()
        for l in range(1,4):out[f"wnorm_l{l}"]=self.p[2*(l-1)].norm(dim=-1).mean(-1).cpu().numpy()
        out["cnorm_l1"]=units["cnorm_l1"].mean(-1)
        return out,units

def output_path(out_dir):
    p=Path(out_dir);return p if p.is_absolute() else ROOT/p

# --------------------------------------------------------------------- report
def read_rows(path):
    with open(path,newline="") as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k,v in list(r.items()):
            if k in ("env","act","act1","act2"):continue
            r[k]=int(v) if k in ("seed","task") else float(v)
    return rows

def rollup(levels,env,arm):
    rr=sorted([r for r in levels if r["env"]==env and r["act"]==arm],key=lambda r:r["seed"])
    assert [r["seed"] for r in rr]==[0,1,2],(env,arm)
    calls=[r["floor_verdict"] for r in rr]
    if all(c=="ABOVE_FLOOR" for c in calls):return "ABOVE_FLOOR",calls
    if all(c=="AT_FLOOR" for c in calls):return "AT_FLOOR",calls
    return "DISAGREEMENT",calls

def anchors(rows):
    raw=subprocess.check_output(["git","show",f"{ANCHOR_COMMIT}:results/layer_chimera_rl_0914/rows.csv"],cwd=ROOT,text=True)
    old=list(csv.DictReader(io.StringIO(raw)))
    res={}
    for new_act,old_arm in (("LR",("LR","LR")),("ELU1",("ELU1","ELU1"))):
        om={(int(r["seed"]),int(r["task"])):float(r["online_acc"]) for r in old
            if r["env"]=="RL" and (r["act1"],r["act2"])==old_arm}
        nm={(r["seed"],r["task"]):r["online_acc"] for r in rows if r["env"]=="RL" and r["act"]==new_act}
        exp={(s,t) for s in range(3) for t in range(1,TASKS+1)}
        assert set(om)==exp and set(nm)==exp,(new_act,len(om),len(nm))
        d={s:max(abs(nm[s,t]-om[s,t]) for t in range(1,TASKS+1)) for s in range(3)}
        res[new_act]=dict(overall=max(d.values()),byseed=d,compared=len(exp),
                          flag="ANCHOR_DRIFT" if max(d.values())>.01 else ("BIT_IDENTICAL" if max(d.values())==0 else "OK"))
    return res

def report(out_dir=OUT):
    out=output_path(out_dir)
    rows=read_rows(out/"rows.csv")
    assert len(rows)==len(MODELS)*TASKS,len(rows)
    for s in range(3):
        for env in ("RL","PM"):
            for task in range(1,TASKS+1):
                vals=[r["floor_acc"] for r in rows if r["seed"]==s and r["env"]==env and r["task"]==task]
                assert len(vals)==len(ACTS) and max(vals)-min(vals)<1e-12,(s,env,task)
    levels=[]
    for m in MODELS:
        rr=sorted([r for r in rows if all(r[k]==v for k,v in m.items())],key=lambda r:r["task"])
        assert [r["task"] for r in rr]==list(range(1,TASKS+1)),m
        late=[r for r in rr if 41<=r["task"]<=50]
        fl=np.asarray([r["floor_acc"] for r in late],float)
        F_,s_=float(fl.mean()),float(fl.std(ddof=1)); thr=F_+3*s_/math.sqrt(10)
        la=float(np.mean([r["online_acc"] for r in late]))
        verdict="AT_FLOOR" if la<=thr else "ABOVE_FLOOR"
        below=[r["online_acc"]<=thr for r in rr]
        ct=None
        if verdict=="AT_FLOOR":
            for i in range(TASKS):
                if all(below[i:]):ct=i+1;break
        levels.append(dict(**m,late_online_acc=la,floor_F=F_,floor_s=s_,floor_thr=thr,floor_verdict=verdict,
                           collapse_task=ct if ct is not None else "",
                           tinygate_l2_t50=rr[-1]["tinygate_l2"],gate_mean_l2_t50=rr[-1]["gate_mean_l2"],
                           neggate_l2_t50=rr[-1]["neggate_l2"],zmed_l2_t50=rr[-1]["zmed_l2"]))
    csvwrite(out/"levels.csv",levels)
    RL={a:rollup(levels,"RL",a) for a in ACTS}; PM={a:rollup(levels,"PM",a) for a in ACTS}
    g0=RL["ELU1"][0]=="AT_FLOOR" and RL["LR"][0]=="ABOVE_FLOOR"
    tgt=("R","GELU","SILU")
    if not g0:
        q1="NOT_TESTABLE"
    elif any(RL[a][0]=="DISAGREEMENT" for a in tgt):
        q1="UNRESOLVED"
    elif all(RL[a][0]=="AT_FLOOR" for a in tgt):
        q1="ALL_DIE"
    elif all(RL[a][0]=="ABOVE_FLOOR" for a in tgt):
        q1="NONE_DIE"
    else:
        q1="SURVIVORS:"+",".join(a for a in tgt if RL[a][0]=="ABOVE_FLOOR")
    anc=anchors(rows)
    verdict=[dict(section="G0",verdict="PASS" if g0 else "FAIL",detail=f'ELU1 {RL["ELU1"][0]} / LR {RL["LR"][0]}')]
    for a in ACTS:
        verdict.append(dict(section="RL_FLOOR",arm=a,seed0=RL[a][1][0],seed1=RL[a][1][1],seed2=RL[a][1][2],verdict=RL[a][0]))
    verdict.append(dict(section="Q1",verdict=q1))
    for a,v in anc.items():
        verdict.append(dict(section="G1_ANCHOR",arm=a,value=v["overall"],verdict=v["flag"],detail=json.dumps(v["byseed"])))
    csvwrite(out/"verdict.csv",verdict)

    def mean_lv(env,a,k):
        return float(np.mean([r[k] for r in levels if r["env"]==env and r["act"]==a]))
    L=[f"# {NAME} report","",f"G0: **{'PASS' if g0 else 'FAIL'}** (ELU1 {RL['ELU1'][0]}, LR {RL['LR'][0]})","",
       f"Q1: **{q1}**","","## RL floor calls (late online t41–50)","",
       "|Arm|late acc (mean)|seed 0|seed 1|seed 2|rollup|collapse task (per seed)|","|---|---:|---|---|---|---|---|"]
    for a in ACTS:
        ct=[str(r["collapse_task"]) or "—" for r in sorted([r for r in levels if r["env"]=="RL" and r["act"]==a],key=lambda r:r["seed"])]
        L.append(f'|{a}|{mean_lv("RL",a,"late_online_acc"):.4f}|{RL[a][1][0]}|{RL[a][1][1]}|{RL[a][1][2]}|**{RL[a][0]}**|{" / ".join(c if c else "—" for c in ct)}|')
    L+=["","## PM late online accuracy (capability check, report only)","","|Arm|late acc|rollup|","|---|---:|---|"]
    for a in ACTS:L.append(f'|{a}|{mean_lv("PM",a,"late_online_acc"):.4f}|{PM[a][0]}|')
    L+=["","## RL layer 2 at t50 (seed mean, report only)","",
        "|Arm|tinygate (|φ′|<1e−8)|neggate (φ′<0)|gate mean|median z̄₂|","|---|---:|---:|---:|---:|"]
    for a in ACTS:
        L.append(f'|{a}|{mean_lv("RL",a,"tinygate_l2_t50"):.3f}|{mean_lv("RL",a,"neggate_l2_t50"):.3f}|{mean_lv("RL",a,"gate_mean_l2_t50"):.3g}|{mean_lv("RL",a,"zmed_l2_t50"):.2f}|')
    L+=["","## G1 anchors (diagnostic)","","|Arm|vs layer_chimera|max abs Δ|seed 0|seed 1|seed 2|flag|","|---|---|---:|---:|---:|---:|---|"]
    for a,v in anc.items():
        L.append(f'|{a}|{"LL" if a=="LR" else "EE"}|{v["overall"]:.3g}|{v["byseed"][0]:.3g}|{v["byseed"][1]:.3g}|{v["byseed"][2]:.3g}|{v["flag"]}|')
    (out/"summary.md").write_text("\n".join(L)+"\n",encoding="utf-8")
    print((out/"summary.md").read_text(),flush=True)

# ------------------------------------------------------------------------ run
def run(device="cuda",tasks=TASKS,steps=STEPS,out_dir=OUT,check_prereg=True):
    assert 1<=tasks<=TASKS and steps>0 and steps%BLOCK==0
    if torch.device(device).type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    out=output_path(out_dir);out.mkdir(parents=True,exist_ok=True)
    if check_prereg:
        subprocess.run(["git","merge-base","--is-ancestor",PREREG_COMMIT,"HEAD"],cwd=ROOT,check=True)
        reg=subprocess.check_output(["git","show",PREREG_COMMIT+f":specs/spec_{NAME}.md"],cwd=ROOT)
        assert hashlib.sha256(reg).hexdigest()==sha(ROOT/f"specs/spec_{NAME}.md"),"spec changed since preregistration"
    st=selftest();(out/"selftest.json").write_text(json.dumps(st,indent=1))
    print("SELFTEST OK",flush=True)
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
        smp={m["act"]:online_acc[j] for j,m in enumerate(MODELS) if m["seed"]==0 and m["env"]=="RL"}
        print(f'TASK {task}/{tasks} elapsed={time.monotonic()-t0:.1f}s RL s0: '+" ".join(f'{a}={smp[a]:.3f}' for a in ACTS),flush=True)
    np.savez_compressed(out/"units.npz",**unitout)
    data_paths=[DATA/n for n in DATA_FILES]
    dev=torch.cuda.get_device_name(e.device) if e.device.type=="cuda" else str(e.device)
    prov=dict(prereg_commit=PREREG_COMMIT,spec_sha256=sha(ROOT/f"specs/spec_{NAME}.md"),code_sha256=sha(Path(__file__)),
        git_hash=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        data_sha256={p.name:sha(p) for p in data_paths},subset_sha256={s:arrsha(v) for s,v in subset.items()},
        initial_sha256={s:[arrsha(v.numpy()) for v in initial(s)] for s in range(3)},torch_version=torch.__version__,
        device=dev,dtype="float32",tf32=False,deterministic=True,models=MODELS,tasks=tasks,steps_per_task=steps,
        images=N,batch=BATCH,late_window=[41,50],wall_seconds=time.monotonic()-t0)
    (out/"provenance.json").write_text(json.dumps(prov,indent=2))

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--device",default="cuda");ap.add_argument("--tasks",type=int,default=TASKS)
    ap.add_argument("--steps",type=int,default=STEPS);ap.add_argument("--out-dir",default=f"results/{NAME}")
    ap.add_argument("--report",action="store_true");ap.add_argument("--selftest",action="store_true")
    args=ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.set_num_threads(1)
    if args.selftest:print(json.dumps(selftest(),indent=1))
    elif args.report:report(args.out_dir)
    else:run(args.device,args.tasks,args.steps,args.out_dir)

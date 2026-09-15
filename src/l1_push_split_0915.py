#!/usr/bin/env python3
"""Preregistered (spec_l1_push_split_0915.md): split the layer-1 push on z̄ into the sample groups
P (z>0), B (z_c<z<=0) and V (z<=z_c) while training.

Engine, activations and the parent self-test are relu_gelu_silu_rl_0914's, imported unchanged.
Training is not changed: the instrumented step computes the parent's gradients and Adam update with
the same operations in the same order, and only then reads extra quantities (G0 checks the
parameters stay bit-identical).  Run from the repo root:
    python3 -m src.l1_push_split_0915                 (train)
    python3 -m src.l1_push_split_0915_report          (verdict; separate file so this one's hash is stable)
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, subprocess, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from src import relu_gelu_silu_rl_0914 as B
from src import perm_dial_rl_0915 as DIAL

ROOT=B.ROOT; NAME="l1_push_split_0915"; OUT=ROOT/f"results/{NAME}"
TASKS=150; FINE_TASKS=10
ZC={"GELU":-0.7517915239,"SILU":-1.2784645428}; ZC_NOMINAL=-1.0
ENVS=(("RL",B.ACTS),("PM",B.ACTS),("K784",("LR","ELU1","GELU","SILU")),("K107",("GELU","SILU")))
MODELS=[dict(seed=s,env=e,act=a) for s in range(3) for e,acts in ENVS for a in acts]
PREREG_COMMIT="f481af5e86b99312ae4dbfe3315eae57b3c25bab"
# accumulator columns (per model, per unit); D = Adam-exact dz̄ share, fR/fI = bias force from u>0 / u<=0
ACC=("dzP","dzB","dzV","fRP","fRB","fRV","fIP","fIB","fIV","nP","nB","nV",
     "dzA","tv","denb","absA","rround","fact","fabs")
SAVE=ACC[:15]+("zend","ratioD","ratioF")
MUTATIONS=("no_bias","no_c1","stale_mu","stale_v","gap","abs_phi")

def chunk_edges(task,fine_tasks=FINE_TASKS):
    return list(range(0,B.STEPS+1,200)) if task<=fine_tasks else [0,100,300,1000,3000,B.STEPS]

def bounds(a,n,eps):
    """Arithmetic rounding bounds (spec §4) and residuals for one chunk of n updates."""
    residD=(a[...,0:3].sum(-1)-a[...,12]).abs(); bD=(2000+4*n)*eps*a[...,15]+a[...,16]
    residF=((a[...,3:6]+a[...,6:9]).sum(-1)-a[...,17]).abs(); bF=(8*n+128)*eps*a[...,18]
    return residD,bD,residF,bF

def ratio(resid,bound):
    """resid/bound; 0/0 (a unit with no movement and no mass) counts as closed, x/0 with x>0 as open."""
    return torch.where(bound>0,resid/bound.clamp_min(torch.finfo(bound.dtype).tiny),
                       torch.where(resid>0,torch.full_like(resid,float("inf")),torch.zeros_like(resid)))

class IEngine(B.Engine):
    def __init__(self,models=MODELS,device="cuda",mutate=None):
        super().__init__(models=models,device=device)
        assert mutate in (None,"inplace_mask")+MUTATIONS,mutate
        self.mutate=mutate; dt=self.p[0].dtype; dev=self.device; M=self.M
        self.eps=float(torch.finfo(dt).eps)
        self.zc=torch.tensor([ZC.get(m["act"],ZC_NOMINAL) for m in models],dtype=dt,device=dev)[:,None,None]
        self.mu=torch.zeros(M,784,1,dtype=dt,device=dev)
        self.muD=torch.zeros_like(self.mu) if mutate=="stale_mu" else self.mu
        self.mW=[torch.zeros(M,100,784,dtype=dt,device=dev) for _ in range(3)]
        self.mb=[torch.zeros(M,100,dtype=dt,device=dev) for _ in range(3)]
        self.W1prev=torch.zeros(M,100,784,dtype=dt,device=dev); self.b1prev=torch.zeros(M,100,dtype=dt,device=dev)
        self.vprev=[torch.zeros_like(self.v[0]),torch.zeros_like(self.v[1])] if mutate=="stale_v" else []
        self.accum=torch.zeros(M,100,len(ACC),dtype=dt,device=dev)
    def _ibufs(self):
        return self.mW+self.mb+[self.W1prev,self.b1prev,self.accum]+self.vprev+([self.muD] if self.mutate=="stale_mu" else [])
    def snapshot(self):
        return super().snapshot()+[q.clone() for q in self._ibufs()]
    @torch.no_grad()
    def restore(self,s):
        n=len(self.p)+len(self.m)+len(self.v)+3
        super().restore(s[:n])
        for q,v in zip(self._ibufs(),s[n:]):q.copy_(v)
    @torch.no_grad()
    def step(self,x,y):
        p,A,mut=self.p,self.A,self.mutate
        # ---- the parent's gradients (B.gradients), same operations in the same order ----
        z1,a1,z2,a2,z3=B.forward(p,x,A)
        g3=(z3.softmax(-1)-y)/x.shape[1]
        g2=torch.bmm(g3,p[4])*B.gate(z2,A)
        u1=torch.bmm(g2,p[2]); ph1=B.gate(z1,A)
        g1=u1*ph1
        if mut=="inplace_mask":g1.mul_((z1>0).to(g1.dtype))      # mutation: instrumentation that perturbs training
        gr=[torch.bmm(g1.transpose(1,2),x),g1.sum(1),torch.bmm(g2.transpose(1,2),a1),g2.sum(1),
            torch.bmm(g3.transpose(1,2),a2),g3.sum(1)]
        ce=(z3.logsumexp(-1)-(z3*y).sum(-1)).mean(-1)
        acc=(z3.argmax(-1)==y.argmax(-1)).float().mean(-1)
        self.W1prev.copy_(p[0]);self.b1prev.copy_(p[1])
        if mut=="stale_v":self.vprev[0].copy_(self.v[0]);self.vprev[1].copy_(self.v[1])
        # ---- the parent's Adam step (B.Engine.step), unchanged ----
        self.ce.add_(ce);self.acc.add_(acc);self.t.add_(1)
        c1=1-torch.pow(.9,self.t);c2=1-torch.pow(.999,self.t)
        for q,g,m,v in zip(p,gr,self.m,self.v):
            m.mul_(.9).add_(g,alpha=.1);v.mul_(.999).addcmul_(g,g,value=.001)
            q.addcdiv_(m/c1,(v/c2).sqrt()+B.ADAM_EPS,value=-.001)
        # ---- read-only instrumentation of layer 1 ----
        vW,vb=(self.vprev[0],self.vprev[1]) if mut=="stale_v" else (self.v[0],self.v[1])
        denW=(vW/c2).sqrt()+B.ADAM_EPS; denb=(vb/c2).sqrt()+B.ADAM_EPS
        cc=torch.ones_like(c1) if mut=="no_c1" else c1
        pos=z1>0; above=z1>self.zc
        vmask=(z1<=self.zc-0.1) if mut=="gap" else ~above
        masks=(pos,above&~pos,vmask)
        red=u1>0; inc=~red
        a=self.accum; absW=None; absb=None
        for k,mk in enumerate(masks):
            mf=mk.to(g1.dtype); g1k=g1*mf
            self.mW[k].mul_(.9).add_(torch.bmm(g1k.transpose(1,2),x),alpha=.1)
            self.mb[k].mul_(.9).add_(g1k.sum(1),alpha=.1)
            rW=self.mW[k]/cc/denW; rb=self.mb[k]/cc/denb
            dz=-.001*torch.bmm(rW,self.muD).squeeze(-1)
            if mut!="no_bias":dz=dz-.001*rb
            a[:,:,k].add_(dz)
            fk=-(u1*ph1.abs()*mf) if (mut=="abs_phi" and k==2) else -g1k
            a[:,:,3+k].add_((fk*red).sum(1)); a[:,:,6+k].add_((fk*inc).sum(1))
            a[:,:,9+k].add_(mf.sum(1))
            absW=rW.abs() if absW is None else absW+rW.abs()
            absb=rb.abs() if absb is None else absb+rb.abs()
        dzA=torch.bmm(p[0]-self.W1prev,self.mu).squeeze(-1)+(p[1]-self.b1prev)
        a[:,:,12].add_(dzA); a[:,:,13].add_(dzA.abs()); a[:,:,14].add_(denb)
        a[:,:,15].add_(.001*(torch.bmm(absW,self.muD).squeeze(-1)+absb)+dzA.abs())
        a[:,:,16].add_(self.eps*(torch.bmm(self.W1prev.abs()+p[0].abs(),self.mu).squeeze(-1)+self.b1prev.abs()+p[1].abs()))
        a[:,:,17].add_(-gr[1]); a[:,:,18].add_(g1.abs().sum(1))

# ------------------------------------------------------------------ self-test (G0)
def _mini(dtype,mutate=None,cls=None):
    torch.set_default_dtype(dtype)
    try:
        models=[dict(seed=0,env="RL",act=a) for a in B.ACTS]
        e=(cls or IEngine)(models=models,device="cpu",**({} if cls is B.Engine else {"mutate":mutate}))
    finally:
        torch.set_default_dtype(torch.float32)
    return e

def _closure(mutate=None):
    e=_mini(torch.float64,mutate)
    with torch.no_grad():
        for q in e.p:q.mul_(3.0)
        g=torch.Generator().manual_seed(1)
        e.cx.copy_(torch.rand(e.M,B.N,784,generator=g,dtype=torch.float64))
        e.cy.copy_(F.one_hot(torch.randint(10,(e.M,B.N),generator=g),10).double())
        e.mu.copy_(e.cx.mean(1).unsqueeze(-1))
        if mutate=="stale_mu":e.muD.copy_(e.mu)
        n=0
        for blk in range(6):
            if blk==3:                                             # task switch: the input mean changes
                perm=torch.randperm(784,generator=g); e.cx.copy_(e.cx[:,:,perm].clone())
                e.mu.copy_(e.cx.mean(1).unsqueeze(-1))
            e.indices.copy_(torch.randint(B.N,(B.BLOCK,e.M,B.BATCH),generator=g)); e.block(); n+=B.BLOCK
    a=e.accum; residD,bD,residF,bF=bounds(a,n,e.eps)
    nsum=a[...,9:12].sum(-1)
    return dict(ratioD=float(ratio(residD,bD).max()),ratioF=float(ratio(residF,bF).max()),
                partition_exact=bool((nsum==n*B.BATCH).all()),
                occ_min_gelu_silu=[float(a[j,:,9+k].sum()/(100*n*B.BATCH)) for j in (3,4) for k in range(3)],
                v_exercised=int((a[3:5,:,2].abs()>bD[3:5]).sum()))

def _train_invariance(mutate=None):
    base=_mini(torch.float32,cls=B.Engine); inst=_mini(torch.float32,mutate)
    g=torch.Generator().manual_seed(2)
    x=torch.rand(base.M,B.N,784,generator=g); y=F.one_hot(torch.randint(10,(base.M,B.N),generator=g),10).float()
    with torch.no_grad():
        for e in (base,inst):
            for q in e.p:q.mul_(3.0)
            e.cx.copy_(x);e.cy.copy_(y)
        inst.mu.copy_(x.mean(1).unsqueeze(-1))
        for blk in range(3):
            ind=torch.randint(B.N,(B.BLOCK,base.M,B.BATCH),generator=g)
            for e in (base,inst):e.indices.copy_(ind);e.block()
    return all(torch.equal(a,b) for a,b in zip(base.p+base.m+base.v,inst.p+inst.m+inst.v))

def selftest_instr():
    out={"closure":_closure()}
    c=out["closure"]
    assert c["ratioD"]<=1 and c["ratioF"]<=1,("instrumentation does not close",c)
    assert c["partition_exact"],("P/B/V is not a partition",c)
    assert min(c["occ_min_gelu_silu"])>0,("a group is never occupied in the check",c)
    assert c["v_exercised"]>0,("the V share never exceeds the bound: closure check is vacuous for V",c)
    out["mutations"]={}
    for mu_ in MUTATIONS:
        r=_closure(mu_); out["mutations"][mu_]=r
        broke=(r["ratioD"]>1) or (r["ratioF"]>1) or (not r["partition_exact"])
        assert broke,("vacuous closure check: mutation not detected",mu_,r)
    out["training_bit_identical"]=_train_invariance()
    out["training_mutation_inplace_detected"]=not _train_invariance("inplace_mask")
    assert out["training_bit_identical"],"instrumented step changes training"
    assert out["training_mutation_inplace_detected"],"vacuous training-invariance check"
    return out

# --------------------------------------------------------------------- run
def run(device="cuda",tasks=TASKS,out_dir=OUT,check_prereg=True,fine_tasks=FINE_TASKS):
    assert 1<=tasks<=TASKS
    if torch.device(device).type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    out=B.output_path(out_dir); (out/"logs").mkdir(parents=True,exist_ok=True)
    if check_prereg:
        subprocess.run(["git","merge-base","--is-ancestor",PREREG_COMMIT,"HEAD"],cwd=ROOT,check=True)
        reg=subprocess.check_output(["git","show",PREREG_COMMIT+f":specs/spec_{NAME}.md"],cwd=ROOT)
        assert hashlib.sha256(reg).hexdigest()==B.sha(ROOT/f"specs/spec_{NAME}.md"),"spec changed since preregistration"
    st=B.selftest(); st["instrumentation"]=selftest_instr()
    (out/"selftest.json").write_text(json.dumps(st,indent=1)); print("SELFTEST OK",flush=True)

    e=IEngine(models=MODELS,device=device)
    gperm={s:B.stream("env_perm_0913",s) for s in range(3)}
    glabel={s:B.stream("env_labels_0913",s) for s in range(3)}
    gbatch={s:B.stream("env_batch_0913",s) for s in range(3)}
    gpart={s:B.stream("partial_perm_0915",s) for s in range(3)}
    ax=B.read_idx(B.DATA/"train-images-idx3-ubyte.gz").reshape(-1,784).astype(np.float32)/255
    ay=B.read_idx(B.DATA/"train-labels-idx1-ubyte.gz").astype(np.int64)
    subset={s:torch.randperm(len(ax),generator=B.stream("rl_subset",s))[:B.N].numpy() for s in range(3)}
    xs={s:torch.tensor(ax[subset[s]],device=e.device) for s in range(3)}
    ys_cpu={s:torch.tensor(ay[subset[s]]) for s in range(3)}
    ys={s:ys_cpu[s].to(e.device) for s in range(3)}
    n_chunks=sum(len(chunk_edges(t,fine_tasks))-1 for t in range(1,tasks+1))
    CH=np.lib.format.open_memmap(out/"logs/chunks.npy",mode="w+",dtype=np.float32,shape=(n_chunks,e.M,100,len(SAVE)))
    zstart=np.zeros((tasks,e.M,100),np.float32); meta=[]
    rows=[];unitout={};exceed=[0,0];ci=0
    e.capture();t0=time.monotonic()
    for task in range(1,tasks+1):
        epochs_needed=-(-(B.STEPS*B.BATCH)//B.N)
        orders={}
        for s in range(3):
            flat=torch.stack([torch.randperm(B.N,generator=gbatch[s]) for _ in range(epochs_needed)]).reshape(-1)[:B.STEPS*B.BATCH]
            orders[s]=flat.reshape(B.STEPS,B.BATCH)
        perm={s:torch.randperm(784,generator=gperm[s]).to(e.device) for s in range(3)}
        labels_cpu={s:torch.randint(10,(B.N,),generator=glabel[s]) for s in range(3)}
        labels={s:labels_cpu[s].to(e.device) for s in range(3)}
        idx={}
        for s in range(3):
            o=torch.randperm(784,generator=gpart[s]); r=torch.rand(784,generator=gpart[s])
            for k in (107,784):idx[s,k]=DIAL.partial_index(o,r,k).to(e.device)
        orderstack=torch.stack([orders[m["seed"]] for m in MODELS],1).to(e.device)
        floors={}
        for j,m in enumerate(MODELS):
            s=m["seed"]
            if m["env"]=="RL":e.cx[j].copy_(xs[s]);yy=labels[s];fl=labels_cpu[s]
            elif m["env"]=="PM":e.cx[j].copy_(xs[s][:,perm[s]]);yy=ys[s];fl=ys_cpu[s]
            else:e.cx[j].copy_(xs[s][:,idx[s,int(m["env"][1:])]]);yy=labels[s];fl=labels_cpu[s]
            e.cy[j].copy_(F.one_hot(yy,10)); floors[j]=float(torch.bincount(fl,minlength=10).max())/B.N
        with torch.no_grad():
            e.mu.copy_(e.cx.mean(1).unsqueeze(-1))
            zstart[task-1]=(torch.bmm(e.p[0],e.mu).squeeze(-1)+e.p[1]).cpu().numpy()
        e.acc.zero_();e.ce.zero_()
        edges=chunk_edges(task,fine_tasks); buf=torch.zeros(len(edges)-1,e.M,100,len(SAVE),device=e.device)
        for c in range(len(edges)-1):
            e.accum.zero_()
            for step in range(edges[c],edges[c+1],B.BLOCK):
                e.indices.copy_(orderstack[step:step+B.BLOCK]);e.replay_block()
            with torch.no_grad():
                n=edges[c+1]-edges[c]; a=e.accum
                residD,bD,residF,bF=bounds(a,n,e.eps)
                buf[c,...,:15]=a[...,:15]
                buf[c,...,15]=torch.bmm(e.p[0],e.mu).squeeze(-1)+e.p[1]
                buf[c,...,16]=ratio(residD,bD); buf[c,...,17]=ratio(residF,bF)
            meta.append(dict(chunk=ci+c,task=task,start=edges[c],end=edges[c+1]))
        arr=buf.cpu().numpy(); CH[ci:ci+len(edges)-1]=arr; ci+=len(edges)-1
        exceed[0]+=int((arr[...,16]>1).sum()); exceed[1]+=int((arr[...,17]>1).sum())
        met,units=e.evaluate()
        online_acc=(e.acc/B.STEPS).cpu().numpy();online_ce=(e.ce/B.STEPS).cpu().numpy()
        assert np.isfinite(online_ce).all() and all(bool(torch.isfinite(q).all()) for q in e.p),("NONFINITE",task)
        assert np.isfinite(arr[...,:16]).all(),("NONFINITE instrumentation",task)
        for j,m in enumerate(MODELS):
            rows.append(dict(**m,task=task,floor_acc=floors[j],online_acc=float(online_acc[j]),
                             online_ce=float(online_ce[j]),**{k:float(v[j]) for k,v in met.items()}))
        for k,v in units.items():unitout[f"{k}_t{task}"]=v
        if task%10==0 or task==tasks:B.csvwrite(out/"rows.csv",rows);CH.flush()
        print(f"TASK {task}/{tasks} elapsed={time.monotonic()-t0:.0f}s chunks={ci} G2 exceed D/F={exceed[0]}/{exceed[1]}",flush=True)
    CH.flush(); del CH
    B.csvwrite(out/"rows.csv",rows); B.csvwrite(out/"chunk_meta.csv",meta)
    np.save(out/"logs/zstart.npy",zstart)
    np.savez_compressed(out/"units.npz",**unitout)
    dev=torch.cuda.get_device_name(e.device) if e.device.type=="cuda" else str(e.device)
    (out/"provenance.json").write_text(json.dumps(dict(prereg_commit=PREREG_COMMIT,spec_sha256=B.sha(ROOT/f"specs/spec_{NAME}.md"),
        code_sha256=B.sha(Path(__file__)),parent_code_sha256=B.sha(Path(B.__file__)),dial_code_sha256=B.sha(Path(DIAL.__file__)),
        git_hash=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        data_sha256={n_:B.sha(B.DATA/n_) for n_ in B.DATA_FILES},subset_sha256={s:B.arrsha(v) for s,v in subset.items()},
        torch_version=torch.__version__,device=dev,dtype="float32",tf32=False,deterministic=True,models=MODELS,
        tasks=tasks,steps_per_task=B.STEPS,images=B.N,batch=B.BATCH,fine_tasks=fine_tasks,acc_columns=ACC,saved_columns=SAVE,
        zc=ZC,zc_nominal=ZC_NOMINAL,g2_exceed_D=exceed[0],g2_exceed_F=exceed[1],chunks=n_chunks,
        chunks_sha256=B.sha(out/"logs/chunks.npy"),wall_seconds=time.monotonic()-t0),indent=2))

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--device",default="cuda");ap.add_argument("--tasks",type=int,default=TASKS)
    ap.add_argument("--out-dir",default=f"results/{NAME}");ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--selftest",action="store_true");a=ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.set_num_threads(1)
    if a.selftest:print(json.dumps(selftest_instr(),indent=1))
    elif a.smoke:run(a.device,a.tasks,a.out_dir,check_prereg=False,fine_tasks=1)
    else:run(a.device,a.tasks,a.out_dir)

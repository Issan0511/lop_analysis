#!/usr/bin/env python3
"""Preregistered matched-sample PM versus RL experiment; all models independent."""
from __future__ import annotations
import argparse, csv, gzip, hashlib, json, math, os, subprocess, time
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/elu_environment_0913"
DATA=Path("/home/issan/Projects/claude/proj_004_drift/data/mnist")
DIMS=(784,100,100,10)
BATCH=16; N=1200; STEPS=6000; TASKS=50; BLOCK=25
MODELS=[dict(seed=s,env=e,act=a,iv=v) for s in range(3) for e in ("PM","RL") for a in ("ELU1","LR") for v in ("ref","wclamp")]

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
def forward(p,x,elu):
    z1=torch.bmm(x,p[0].transpose(1,2))+p[1][:,None,:]
    a1=activ(z1,elu)
    z2=torch.bmm(a1,p[2].transpose(1,2))+p[3][:,None,:]
    a2=activ(z2,elu)
    z3=torch.bmm(a2,p[4].transpose(1,2))+p[5][:,None,:]
    return z1,a1,z2,a2,z3
def gradients(p,x,y,elu):
    z1,a1,z2,a2,z3=forward(p,x,elu)
    g3=(z3.softmax(-1)-y)/x.shape[1]
    g2=torch.bmm(g3,p[4])*gate(z2,elu)
    g1=torch.bmm(g2,p[2])*gate(z1,elu)
    gr=[torch.bmm(g1.transpose(1,2),x),g1.sum(1),
        torch.bmm(g2.transpose(1,2),a1),g2.sum(1),
        torch.bmm(g3.transpose(1,2),a2),g3.sum(1)]
    ce=(z3.logsumexp(-1)-(z3*y).sum(-1)).mean(-1)
    acc=(z3.argmax(-1)==y.argmax(-1)).float().mean(-1)
    return gr,ce,acc

class Engine:
    def __init__(self,models=MODELS,device="cuda"):
        self.models=models;self.device=device;self.M=len(models)
        self.p=[torch.stack([initial(m["seed"])[i] for m in models]).to(device) for i in range(6)]
        self.m=[torch.zeros_like(q) for q in self.p];self.v=[torch.zeros_like(q) for q in self.p]
        self.t=torch.zeros((),device=device);self.elu=torch.tensor([m["act"]=="ELU1" for m in models],device=device)[:,None,None]
        self.on=torch.zeros(self.M,1,1,dtype=torch.bool,device=device)
        self.target=(self.p[0]-self.p[0].mean(-1,keepdim=True)).norm(dim=-1,keepdim=True)
        self.acc=torch.zeros(self.M,device=device);self.ce=torch.zeros_like(self.acc)
        self.cx=torch.zeros(self.M,N,784,device=device);self.cy=torch.zeros(self.M,N,10,device=device)
        self.indices=torch.zeros(BLOCK,self.M,BATCH,dtype=torch.long,device=device)
        self.mid=torch.arange(self.M,device=device)[:,None]
    @torch.no_grad()
    def step(self,x,y):
        gr,ce,acc=gradients(self.p,x,y,self.elu)
        self.ce.add_(ce);self.acc.add_(acc);self.t.add_(1)
        c1=1-torch.pow(.9,self.t);c2=1-torch.pow(.999,self.t)
        for p,g,m,v in zip(self.p,gr,self.m,self.v):
            m.mul_(.9).add_(g,alpha=.1);v.mul_(.999).addcmul_(g,g,value=.001)
            p.addcdiv_(m/c1,(v/c2).sqrt()+1e-8,value=-.001)
        w=self.p[0];mu=w.mean(-1,keepdim=True);wc=w-mu
        projected=mu+wc*(self.target/wc.norm(dim=-1,keepdim=True).clamp_min(1e-30))
        w.copy_(torch.where(self.on,projected,w))
    @torch.no_grad()
    def block(self):
        for k in range(BLOCK):self.step(self.cx[self.mid,self.indices[k]],self.cy[self.mid,self.indices[k]])
    def snapshot(self):
        return [q.clone() for q in self.p+self.m+self.v+[self.t,self.acc,self.ce]]
    @torch.no_grad()
    def restore(self,s):
        for q,v in zip(self.p+self.m+self.v+[self.t,self.acc,self.ce],s):q.copy_(v)
    def capture(self):
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
    @torch.no_grad()
    def evaluate(self):
        z1,a1,z2,a2,logits=forward(self.p,self.cx,self.elu)
        ce=(logits.logsumexp(-1)-(logits*self.cy).sum(-1)).mean(-1)
        acc=(logits.argmax(-1)==self.cy.argmax(-1)).float().mean(-1)
        out=dict(train_acc=acc.cpu().numpy(),train_ce=ce.cpu().numpy())
        units={}
        for l,z in [(1,z1),(2,z2)]:
            g=gate(z,self.elu)
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

def validation():
    """Synthetic only. Original failed tolerance retained; addendum isolates Adam."""
    import importlib.util
    torch.set_num_threads(1)
    hostpath=DATA.parents[1]/"src/pmnist_0905.py"
    spec=importlib.util.spec_from_file_location("elu_env_reference_host",hostpath)
    host=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name]=host;spec.loader.exec_module(host)
    checks={"historical_init_source_sha256":sha(hostpath)}
    checks["historical_init_maxabs"]=max(float((a-b.detach()).abs().max()) for s in range(3) for a,b in zip(initial(s),host.init_params(s,torch.device("cpu"))))
    assert checks["historical_init_maxabs"]==0.,checks
    models=[dict(seed=0,act="ELU1",env="PM",iv="ref"),dict(seed=1,act="LR",env="RL",iv="wclamp")]
    gen=torch.Generator().manual_seed(9301)
    x=torch.rand(2,16,784,generator=gen)
    y=torch.nn.functional.one_hot(torch.randint(10,(2,16),generator=gen),10).float()
    e=Engine(models);p_cpu=[q.cpu().requires_grad_() for q in e.p]
    elu=e.elu.cpu()
    z=forward(p_cpu,x,elu)[-1]
    loss=(z.logsumexp(-1)-(z*y).sum(-1)).mean(-1).sum()
    ref=torch.autograd.grad(loss,p_cpu)
    got,_,_=gradients(e.p,x.cuda(),y.cuda(),e.elu)
    checks["random_gradient_maxabs"]=max(float((g.cpu()-r).abs().max()) for g,r in zip(got,ref))
    torchopt=[q.detach().clone().requires_grad_() for q in p_cpu]
    opt=torch.optim.Adam(torchopt,lr=.001,betas=(.9,.999),eps=1e-8)
    zz=forward(torchopt,x,elu)[-1];ll=(zz.logsumexp(-1)-(zz*y).sum(-1)).mean(-1).sum();ll.backward();opt.step()
    samegrad=[q.detach().clone().requires_grad_() for q in p_cpu]
    sameopt=torch.optim.Adam(samegrad,lr=.001,betas=(.9,.999),eps=1e-8)
    mutant=[q.detach().clone().requires_grad_() for q in p_cpu]
    mutopt=torch.optim.Adam(mutant,lr=.002,betas=(.9,.999),eps=1e-8)
    for q,r,g in zip(samegrad,mutant,got):q.grad=g.cpu().clone();r.grad=g.cpu().clone()
    sameopt.step();mutopt.step()
    e.step(x.cuda(),y.cuda())
    checks["random_parameter_maxabs"]=max(float((q.cpu()-r.detach()).abs().max()) for q,r in zip(e.p,torchopt))
    checks["original_random_parameter_2e5_pass"]=checks["random_parameter_maxabs"]<2e-5
    checks["same_gpu_gradient_cpu_adam_maxabs"]=max(float((q.cpu()-r.detach()).abs().max()) for q,r in zip(e.p,samegrad))
    checks["doubled_lr_mutation_maxabs"]=max(float((q.cpu()-r.detach()).abs().max()) for q,r in zip(e.p,mutant))
    inds=[]
    for i in range(2):
        pi=[q[i:i+1].detach().clone().requires_grad_() for q in p_cpu]
        zi=forward(pi,x[i:i+1],elu[i:i+1])[-1]
        li=(zi.logsumexp(-1)-(zi*y[i:i+1]).sum(-1)).mean()
        gi=torch.autograd.grad(li,pi)
        inds.append(max(float((a-b[i:i+1]).abs().max()) for a,b in zip(gi,ref)))
    checks["independence_maxabs"]=max(inds)
    gd=[]
    for iselu in (False,True):
        za=torch.linspace(-10,10,501).reshape(1,501,1).requires_grad_()
        ae=torch.tensor(iselu).reshape(1,1,1)
        aa=activ(za,ae);ga=torch.autograd.grad(aa.sum(),za)[0]
        gd.append(float((ga-gate(za,ae)).detach().abs().max()))
    checks["activation_derivative_maxabs"]=max(gd)
    # Active intervention and nonzero time, with random labels, inside graph.
    e.cx.copy_(torch.rand(e.cx.shape,generator=gen).cuda())
    e.cy.copy_(torch.nn.functional.one_hot(torch.randint(10,(2,N),generator=gen),10))
    e.indices.copy_(torch.randint(N,e.indices.shape,generator=gen).cuda())
    e.target.mul_(.8);e.on[1]=True;e.t.fill_(73)
    e.capture();s=e.snapshot();e.block();want=e.snapshot();e.restore(s);e.graph.replay();torch.cuda.synchronize()
    checks["graph_active_clamp_eager_maxabs"]=max(float((a-b).abs().max()) for a,b in zip(e.snapshot(),want))
    checks["graph_counter_increment"]=float(e.t-s[18]);assert checks["graph_counter_increment"]==BLOCK
    checks["graph_initial_t"]=float(s[18]);checks["graph_final_t"]=float(e.t)
    cn=(e.p[0]-e.p[0].mean(-1,keepdim=True)).norm(dim=-1,keepdim=True)
    checks["graph_active_clamp_norm_rel"]=float((cn[1]/e.target[1]-1).abs().max())
    w=e.p[0].clone()+.025;mu=w.mean(-1,keepdim=True);wc=w-mu;target=wc.norm(dim=-1,keepdim=True)*.6
    new=mu+wc*target/wc.norm(dim=-1,keepdim=True)
    checks["clamp_norm_rel"]=float(((new-new.mean(-1,keepdim=True)).norm(dim=-1,keepdim=True)/target-1).abs().max())
    checks["clamp_rowmean_abs"]=float((new.mean(-1,keepdim=True)-mu).abs().max())
    checks["mutation_rowmean_abs"]=float(((w*.6).mean(-1,keepdim=True)-mu).abs().max())
    assert checks["random_gradient_maxabs"]<2e-5,checks
    assert checks["random_parameter_maxabs"]<2e-5,checks
    assert checks["same_gpu_gradient_cpu_adam_maxabs"]<1e-6,checks
    assert checks["doubled_lr_mutation_maxabs"]>5e-4,checks
    assert checks["independence_maxabs"]<2e-5,checks
    assert checks["activation_derivative_maxabs"]<2e-5,checks
    assert checks["graph_active_clamp_eager_maxabs"]<5e-5,checks
    assert checks["graph_active_clamp_norm_rel"]<2e-5,checks
    assert checks["clamp_norm_rel"]<2e-5 and checks["clamp_rowmean_abs"]<2e-6,checks
    assert checks["mutation_rowmean_abs"]>2e-6,checks
    f=Engine();f.cx.copy_(torch.rand(f.cx.shape,generator=gen).cuda())
    f.cy.copy_(torch.nn.functional.one_hot(torch.randint(10,(f.M,N),generator=gen),10))
    f.indices.copy_(torch.randint(N,f.indices.shape,generator=gen).cuda());f.capture()
    for _ in range(5):f.graph.replay()
    torch.cuda.synchronize();t0=time.monotonic()
    for _ in range(100):f.graph.replay()
    torch.cuda.synchronize()
    checks["seconds_per_25step_block"]=(time.monotonic()-t0)/100
    checks["estimated_training_seconds"]=checks["seconds_per_25step_block"]*(TASKS*STEPS/BLOCK)
    checks["qa_basis"]="spec_elu_environment_0913_qa_addendum.md"
    checks["status"]="PASS_ADDENDUM"
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"validation.json").write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2),flush=True)

def interval(a):
    a=np.asarray(a,float);mean=float(a.mean());sd=float(a.std(ddof=1));half=4.302652729911275*sd/math.sqrt(3)
    return mean,sd,mean-half,mean+half

def report():
    with open(OUT/"rows.csv") as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k in ("seed","task"):r[k]=int(r[k])
        for k in r:
            if k not in ("seed","task","env","act","iv"):r[k]=float(r[k])
    groups={}
    for m in MODELS:
        rr=[r for r in rows if all(r[k]==v for k,v in m.items())]
        assert len(rr)==TASKS,(m,len(rr))
        def avg(k,lo,hi):return float(np.mean([r[k] for r in rr if lo<=r["task"]<=hi]))
        early=avg("online_acc",11,20);late=avg("online_acc",41,50)
        groups[tuple(m.values())]=dict(**m,early_online_acc=early,late_online_acc=late,
            all_online_acc=avg("online_acc",1,50),lop_pp=100*(early-late),
            early_online_ce=avg("online_ce",11,20),late_online_ce=avg("online_ce",41,50),
            ce_degradation=avg("online_ce",41,50)-avg("online_ce",11,20),
            early_train_acc=avg("train_acc",11,20),late_train_acc=avg("train_acc",41,50),
            endpoint_ceiling_fraction=float(np.mean([r["train_acc"]>=.99 for r in rr])),
            online_ceiling_fraction=float(np.mean([r["online_acc"]>=.99 for r in rr])),
            early_cnorm=avg("cnorm_l1",11,20),late_cnorm=avg("cnorm_l1",41,50))
    csvwrite(OUT/"levels.csv",list(groups.values()))
    paired=[];Ds={}
    for s in range(3):
        for env in ("PM","RL"):
            for act in ("ELU1","LR"):
                a=groups[s,env,act,"ref"];b=groups[s,env,act,"wclamp"]
                d=a["lop_pp"]-b["lop_pp"];Ds[s,env,act]=d
                paired.append(dict(seed=s,env=env,act=act,D_lop_pp=d,
                    late_gain_pp=100*(b["late_online_acc"]-a["late_online_acc"]),
                    D_ce=a["ce_degradation"]-b["ce_degradation"],
                    budget_flag="BUDGET_LIMITED" if a["early_train_acc"]<.9 else "FIT",
                    ref_early_train_acc=a["early_train_acc"],ref_late_train_acc=a["late_train_acc"],
                    ref_early_online_acc=a["early_online_acc"],ref_late_online_acc=a["late_online_acc"]))
    csvwrite(OUT/"paired.csv",paired)
    verdict=[]
    def add(name,vals,label=None):
        me,sd,lo,hi=interval(vals)
        verdict.append(dict(contrast=name,seed0=vals[0],seed1=vals[1],seed2=vals[2],mean_pp=me,sd_pp=sd,ci95_low_pp=lo,ci95_high_pp=hi,
                            verdict=label or "ESTIMATE"))
    for env in ("PM","RL"):
        for act in ("ELU1","LR"):add("D_"+env+"_"+act,[Ds[s,env,act] for s in range(3)])
    for env in ("PM","RL"):
        vals=[Ds[s,env,"LR"]-Ds[s,env,"ELU1"] for s in range(3)]
        lab="CONSISTENT_INTERACTION" if (min(vals)>0 or max(vals)<0) and abs(np.mean(vals))>=.5 else "INCONCLUSIVE"
        add("I_"+env,vals,lab)
    vals=[(Ds[s,"PM","LR"]-Ds[s,"PM","ELU1"])-(Ds[s,"RL","LR"]-Ds[s,"RL","ELU1"]) for s in range(3)]
    lab="ENV_DEPENDENT" if (min(vals)>0 or max(vals)<0) and abs(np.mean(vals))>=.5 else "INCONCLUSIVE"
    add("J_PM_minus_RL",vals,lab);csvwrite(OUT/"verdict.csv",verdict)
    lines=["# Matched-sample ELU environment comparison 0913","",
       "50 tasks × 6000 updates/task; fixed1200 images/seed; same 784-100-100-10 MLP and Adam in both environments.",
       "PM: fresh pixel permutations and true labels. RL: fixed pixels and fresh iid random labels. Three seed replicas.",
       "W1 centered-row norm projection starts task11 at task10 targets. Primary LoP=early(11–20) online accuracy minus late(41–50), in pp.",
       "This PM is repeated-sample80-epoch PM, not the historical single-pass10000-image benchmark. RL uses80epochs, not historical400.",
       "","|Contrast|Seed0|Seed1|Seed2|Mean pp|95% t CI|Label|","|---|---:|---:|---:|---:|---|---|"]
    for r in verdict:lines.append(f'|{r["contrast"]}|{r["seed0"]:.4f}|{r["seed1"]:.4f}|{r["seed2"]:.4f}|{r["mean_pp"]:.4f}|[{r["ci95_low_pp"]:.4f}, {r["ci95_high_pp"]:.4f}]|{r["verdict"]}|')
    lines += ["","D = LoP_ref−LoP_clamp; I = D_leaky−D_ELU; J = I_PM−I_RL.",
       "Positive I means clamp reduces degradation more for leaky; positive J means that difference is larger in PM.",
       "All inferential replication is across seeds. Task entries are repeated observations, not independent replicas.",
       "","|Environment|Activation|Iv|Early online %|Late online %|Early endpoint %|Late endpoint %|LoP pp|","|---|---|---|---:|---:|---:|---:|---:|"]
    for env in ("PM","RL"):
        for act in ("ELU1","LR"):
            for iv in ("ref","wclamp"):
                rr=[groups[s,env,act,iv] for s in range(3)]
                vals=[np.mean([r[k] for r in rr]) for k in ("early_online_acc","late_online_acc","early_train_acc","late_train_acc","lop_pp")]
                lines.append(f"|{env}|{act}|{iv}|{100*vals[0]:.3f}|{100*vals[1]:.3f}|{100*vals[2]:.3f}|{100*vals[3]:.3f}|{vals[4]:.3f}|")
    flagged=[f'{r["env"]}/{r["act"]}/seed{r["seed"]}' for r in paired if r["budget_flag"]=="BUDGET_LIMITED"]
    lines += ["","Budget-limited references (early endpoint train accuracy<90%): "+(", ".join(flagged) if flagged else "none")+".",
       "Budget flags retain every measured contrast: these are finite-budget learning effects, not claims about asymptotic capacity.",
       "Three seeds and matched-sample construction bound the scope; a null interaction does not establish environment invariance.",
       "Raw online CE and endpoint trajectories are in rows.csv/learning.csv; perunit distributions are in units.npz.",
       ""]
    (OUT/"summary.md").write_text("\n".join(lines),encoding="utf-8")
    print((OUT/"summary.md").read_text(),flush=True)

def run(prereg):
    OUT.mkdir(parents=True,exist_ok=True)
    assert (OUT/"validation.json").exists(),"Run validation first"
    assert json.loads((OUT/"validation.json").read_text())["status"]=="PASS_ADDENDUM"
    assert len(prereg)>=7,"Explicit coordinator-provided prereg commit required"
    # Registration must contain this exact spec, and be an ancestor of current checkout.
    subprocess.run(["git","merge-base","--is-ancestor",prereg,"HEAD"],cwd=ROOT,check=True)
    qa_registered=subprocess.check_output(["git","show",prereg+":specs/spec_elu_environment_0913_qa_addendum.md"],cwd=ROOT)
    assert hashlib.sha256(qa_registered).hexdigest()==sha(ROOT/"specs/spec_elu_environment_0913_qa_addendum.md")
    registered=subprocess.check_output(["git","show",prereg+":specs/spec_elu_environment_0913.md"],cwd=ROOT)
    assert hashlib.sha256(registered).hexdigest()==sha(ROOT/"specs/spec_elu_environment_0913.md")
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    e=Engine();gperm={s:stream("env_perm_0913",s) for s in range(3)}
    glabel={s:stream("env_labels_0913",s) for s in range(3)}
    gbatch={s:stream("env_batch_0913",s) for s in range(3)}
    ax=read_idx(DATA/"train-images-idx3-ubyte.gz").reshape(-1,784).astype(np.float32)/255
    ay=read_idx(DATA/"train-labels-idx1-ubyte.gz").astype(np.int64)
    subset={s:torch.randperm(len(ax),generator=stream("rl_subset",s))[:N].numpy() for s in range(3)}
    xs={s:torch.tensor(ax[subset[s]],device="cuda") for s in range(3)}
    ys={s:torch.tensor(ay[subset[s]],device="cuda") for s in range(3)}
    rows=[];learning=[];unitout={};rng_hashes=[];checks={"warmup_pair_maxabs":0.,"clamp_norm_rel_max":0.}
    e.capture();t0=time.monotonic()
    for task in range(1,TASKS+1):
        orders={s:torch.stack([torch.randperm(N,generator=gbatch[s]) for _ in range(80)]).reshape(STEPS,BATCH) for s in range(3)}
        perm={s:torch.randperm(784,generator=gperm[s]).cuda() for s in range(3)}
        labels={s:torch.randint(10,(N,),generator=glabel[s]).cuda() for s in range(3)}
        orderstack=torch.stack([orders[m["seed"]] for m in MODELS],1).cuda()
        for j,m in enumerate(MODELS):
            s=m["seed"];e.cx[j].copy_(xs[s][:,perm[s]] if m["env"]=="PM" else xs[s])
            yy=ys[s] if m["env"]=="PM" else labels[s]
            e.cy[j].copy_(torch.nn.functional.one_hot(yy,10))
        rng_hashes.append(dict(task=task,**{f"order_s{s}":arrsha(orders[s].numpy()) for s in range(3)},
                             **{f"perm_s{s}":arrsha(perm[s].cpu().numpy()) for s in range(3)},
                             **{f"labels_s{s}":arrsha(labels[s].cpu().numpy()) for s in range(3)}))
        if task==11:
            with torch.no_grad():
                e.target.copy_((e.p[0]-e.p[0].mean(-1,keepdim=True)).norm(dim=-1,keepdim=True))
                e.on.copy_(torch.tensor([m["iv"]=="wclamp" for m in MODELS],device="cuda")[:,None,None])
        e.acc.zero_();e.ce.zero_()
        met,_=e.evaluate()
        for j,m in enumerate(MODELS):learning.append(dict(**m,task=task,step=0,train_acc=float(met["train_acc"][j]),train_ce=float(met["train_ce"][j])))
        for step in range(0,STEPS,BLOCK):
            e.indices.copy_(orderstack[step:step+BLOCK]);e.graph.replay()
            end=step+BLOCK
            if end in (75,375,1500,3000,6000):
                met,units=e.evaluate()
                for j,m in enumerate(MODELS):learning.append(dict(**m,task=task,step=end,train_acc=float(met["train_acc"][j]),train_ce=float(met["train_ce"][j])))
        online_acc=(e.acc/STEPS).cpu().numpy();online_ce=(e.ce/STEPS).cpu().numpy()
        assert np.isfinite(online_ce).all() and all(bool(torch.isfinite(q).all()) for q in e.p),("NONFINITE",task)
        for j,m in enumerate(MODELS):
            rows.append(dict(**m,task=task,online_acc=float(online_acc[j]),online_ce=float(online_ce[j]),**{k:float(v[j]) for k,v in met.items()}))
        for k,v in units.items():unitout[f"{k}_t{task}"]=v
        if task<=10:
            worst=max(float((q[0::2]-q[1::2]).abs().max()) for q in e.p+e.m+e.v)
            checks["warmup_pair_maxabs"]=max(checks["warmup_pair_maxabs"],worst)
            assert worst==0.,("WARMUP_PAIR",task,worst)
        else:
            cn=(e.p[0]-e.p[0].mean(-1,keepdim=True)).norm(dim=-1,keepdim=True)
            err=float(((cn/e.target-1)[e.on.expand_as(cn)]).abs().max())
            checks["clamp_norm_rel_max"]=max(checks["clamp_norm_rel_max"],err);assert err<2e-5,err
        csvwrite(OUT/"rows.csv",rows);csvwrite(OUT/"learning.csv",learning)
        print(f"TASK {task}/{TASKS} elapsed={time.monotonic()-t0:.1f}s online_ref PM_ELU={online_acc[0]:.4f} PM_LR={online_acc[2]:.4f} RL_ELU={online_acc[4]:.4f} RL_LR={online_acc[6]:.4f}",flush=True)
    np.savez_compressed(OUT/"units.npz",**unitout)
    torch.save(dict(models=MODELS,parameters=[q.cpu() for q in e.p],adam_m=[q.cpu() for q in e.m],adam_v=[q.cpu() for q in e.v],
                    t=e.t.cpu(),target=e.target.cpu()),OUT/"checkpoint.pt")
    (OUT/"rng_hashes.json").write_text(json.dumps(rng_hashes,indent=1))
    provenance=dict(prereg_commit=prereg,spec_sha256=sha(ROOT/"specs/spec_elu_environment_0913.md"),code_sha256=sha(Path(__file__)),qa_spec_sha256=sha(ROOT/"specs/spec_elu_environment_0913_qa_addendum.md"),
        data_sha256={p.name:sha(p) for p in DATA.glob("*gz")},subset_sha256={s:arrsha(v) for s,v in subset.items()},
        initial_sha256={s:[arrsha(v.numpy()) for v in initial(s)] for s in range(3)},torch_version=torch.__version__,
        device=torch.cuda.get_device_name(),dtype="float32",tf32=False,deterministic=True,models=MODELS,tasks=TASKS,steps_per_task=STEPS,
        images=N,batch=BATCH,clamp_start_task=11,early_window=[11,20],late_window=[41,50],checks=checks,
        wall_seconds=time.monotonic()-t0)
    (OUT/"provenance.json").write_text(json.dumps(provenance,indent=2))
    report()

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--validate",action="store_true");ap.add_argument("--report",action="store_true");ap.add_argument("--prereg")
    args=ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.validate:validation()
    elif args.report:report()
    else:run(args.prereg)

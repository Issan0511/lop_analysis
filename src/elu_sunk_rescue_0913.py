#!/usr/bin/env python3
"""Bounded preregistered prefix reconstruction and ELU rescue jobs."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np,torch
import elu_environment_0913 as base
from elu_sunk_engine_0913 import Engine
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"results/elu_sunk_rescue_0913";PFX=OUT/"prefix"
OLD=Path("/home/issan/Projects/claude/elu_reserve_0913/results/elu_environment_0913")
BR=("A","B20","C20","D20","C5","D5","C10","D10","RB20","RC20","RD20");MEAS=(0,75,375,1500,3000,6000)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def jw(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
def cw(p,rs):
 k=list(dict.fromkeys(x for r in rs for x in r));f=p.open("w",newline="");w=csv.DictWriter(f,fieldnames=k);w.writeheader();w.writerows(rs);f.close()
def cfg():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def prereg(c):
 if not c or len(c)<7:raise ValueError("explicit --prereg required")
 subprocess.run(["git","merge-base","--is-ancestor",c,"HEAD"],cwd=ROOT,check=True);rel="specs/spec_elu_sunk_rescue_0913.md";b=subprocess.check_output(["git","show",f"{c}:{rel}"],cwd=ROOT)
 if hashlib.sha256(b).hexdigest()!=sha(ROOT/rel):raise RuntimeError("spec differs from prereg")
 return sha(ROOT/rel)
def qa_gate():
 rv=OUT/"runner_validation.json";ev=OUT/"engine_validation.json"
 if not rv.exists() or not ev.exists():raise RuntimeError("PASS runner and engine validation required")
 r=json.loads(rv.read_text());e=json.loads(ev.read_text())
 if r.get("status")!="PASS" or r.get("code_sha256")!=sha(__file__):raise RuntimeError("runner QA missing, failed, or stale")
 ep=Path(__file__).with_name("elu_sunk_engine_0913.py")
 if e.get("status")!="PASS" or e.get("code_sha256")!=sha(ep):raise RuntimeError("engine QA missing, failed, or stale")
def load_data():
 x=base.read_idx(base.DATA/"train-images-idx3-ubyte.gz").reshape(-1,784).astype(np.float32)/255;y=base.read_idx(base.DATA/"train-labels-idx1-ubyte.gz").astype(np.int64);sub={s:torch.randperm(len(x),generator=base.stream("rl_subset",s))[:base.N].numpy() for s in range(3)};return x,y,sub
def gens(st=None):
 d={r:{s:base.stream(r,s) for s in range(3)} for r in ("env_perm_0913","env_labels_0913","env_batch_0913")}
 if st:
  for r in d:
   for s in d[r]:d[r][s].set_state(st[r][s])
 return d
def states(d):return {r:{s:g.get_state() for s,g in q.items()} for r,q in d.items()}
def draw(d):
 o={s:torch.stack([torch.randperm(base.N,generator=d["env_batch_0913"][s]) for _ in range(80)]).reshape(base.STEPS,base.BATCH) for s in range(3)};p={s:torch.randperm(784,generator=d["env_perm_0913"][s]) for s in range(3)};l={s:torch.randint(10,(base.N,),generator=d["env_labels_0913"][s]) for s in range(3)}
 h={f"order_s{s}":ah(o[s]) for s in range(3)}|{f"perm_s{s}":ah(p[s]) for s in range(3)}|{f"labels_s{s}":ah(l[s]) for s in range(3)};return o,p,l,h
def cp(e,d,t,last,sub):return dict(task=t,models=base.MODELS,parameters=[q.cpu() for q in e.p],adam_m=[q.cpu() for q in e.m],adam_v=[q.cpu() for q in e.v],t=e.t.cpu(),target=e.target.cpu(),rng_states=states(d),last=last,subset=sub)
@torch.no_grad()
def prefix(c):
 ss=prereg(c);cfg();PFX.mkdir(parents=True,exist_ok=True)
 for t in (10,20,50):
  if (PFX/f"checkpoint_{t}.pt").exists():raise FileExistsError("refuse overwrite prefix")
 x,y,sub=load_data();xs={s:torch.tensor(x[sub[s]],device="cuda") for s in range(3)};ys={s:torch.tensor(y[sub[s]],device="cuda") for s in range(3)};d=gens();e=base.Engine();e.capture();ou=np.load(OLD/"units.npz");orh=json.loads((OLD/"rng_hashes.json").read_text());units={};mx=0.;rh=[]
 for t in range(1,51):
  o,p,l,h=draw(d);rh.append(dict(task=t,**h));assert rh[-1]==orh[t-1]
  for j,m in enumerate(base.MODELS):s=m["seed"];e.cx[j].copy_(xs[s][:,p[s].cuda()] if m["env"]=="PM" else xs[s]);yy=ys[s] if m["env"]=="PM" else l[s].cuda();e.cy[j].copy_(torch.nn.functional.one_hot(yy,10))
  if t==11:e.target.copy_((e.p[0]-e.p[0].mean(-1,keepdim=True)).norm(dim=-1,keepdim=True));e.on.copy_(torch.tensor([m["iv"]=="wclamp" for m in base.MODELS],device="cuda")[:,None,None])
  ost=torch.stack([o[m["seed"]] for m in base.MODELS],1).cuda()
  for z in range(0,base.STEPS,base.BLOCK):e.indices.copy_(ost[z:z+base.BLOCK]);e.graph.replay()
  torch.cuda.synchronize();_,u=e.evaluate()
  for k,v in u.items():key=f"{k}_t{t}";units[key]=v;err=float(np.max(np.abs(v-ou[key])));mx=max(mx,err);assert err<=1e-6,(key,err)
  if t in (10,20,50):torch.save(cp(e,d,t,dict(orders=o,perm=p,labels=l),sub),PFX/f"checkpoint_{t}.pt")
  print(f"PREFIX {t}/50",flush=True)
 oc=torch.load(OLD/"checkpoint.pt",map_location="cpu",weights_only=False);nc=torch.load(PFX/"checkpoint_50.pt",map_location="cpu",weights_only=False);cm=max(float((a-b).abs().max()) for k in ("parameters","adam_m","adam_v") for a,b in zip(oc[k],nc[k]));assert cm<=1e-6 and torch.equal(oc["t"],nc["t"])
 np.savez_compressed(PFX/"units.npz",**units);jw(PFX/"rng_hashes.json",rh);jw(PFX/"provenance.json",dict(status="COMPLETE",prereg_commit=c,spec_sha256=ss,code_sha256=sha(__file__),base_sha256=sha(base.__file__),source_commit="46becd95b992d1ce27f95a6dc5b9cbdd2ad73f63",data_sha256={p.name:sha(p) for p in base.DATA.glob("*gz")},subset_sha256={s:ah(v) for s,v in sub.items()},unit_maxabs=mx,checkpoint_maxabs=cm,old_hashes={f:sha(OLD/f) for f in ("units.npz","checkpoint.pt","rng_hashes.json")},torch_version=torch.__version__,device=torch.cuda.get_device_name(),tf32=False,deterministic=True))
def seed(role,s,e,t,l):return int.from_bytes(hashlib.sha256(f"elu_sunk_rescue_0913|{role}|{s}|{e}|{t}|{l}".encode()).digest()[:8],"little")&((1<<63)-1)
def mid(s,e):return next(i for i,m in enumerate(base.MODELS) if m==dict(seed=s,env=e,act="ELU1",iv="ref"))
def ids(b,t,c):n=20 if "20" in b else 10 if "10" in b else 5 if "5" in b else 0;return (c[:n] if b.startswith("R") else t[:n]) if n else []
def lift(b):return b.startswith(("C","D","RC","RD"))
def freeze(b):return b.startswith(("B","D","RB","RD"))
def masks(e,L,ii):
 z=[torch.zeros_like(p,dtype=torch.bool) for p in e.p]
 if ii:q=torch.tensor(ii,device=e.device);w=2*(L-1);z[w][:,q,:]=1;z[w+1][:,q]=1;z[w+2][:,:,q]=1
 return z
@torch.no_grad()
def run(c,env,T,L):
 ss=prereg(c);cfg();qa_gate();job=OUT/f"{env}_t{T}_l{L}";done=job/"provenance.json"
 if done.exists() and json.loads(done.read_text()).get("status")=="COMPLETE":raise FileExistsError("refuse overwrite completed job")
 pp=PFX/"provenance.json"
 if not pp.exists() or json.loads(pp.read_text()).get("status")!="COMPLETE":raise RuntimeError("complete verified prefix required")
 job.mkdir(parents=True,exist_ok=True);pf=PFX/f"checkpoint_{T}.pt";q=torch.load(pf,map_location="cpu",weights_only=False);x,y,sub=load_data();d=gens(q["rng_states"]);ou=np.load(PFX/"units.npz");sels=[];aux={}
 for s in range(3):
  j=mid(s,env);pool=np.flatnonzero(np.stack([ou[f"lowgate_l{L}_t{t}"][j] for t in range(T-4,T+1)]).min(0)>=.95).tolist();g=torch.Generator().manual_seed(seed("target",s,env,T,L));tar=[pool[i] for i in torch.randperm(len(pool),generator=g).tolist()[:20]];outside=[i for i in range(100) if i not in tar];g=torch.Generator().manual_seed(seed("control",s,env,T,L));ctl=[outside[i] for i in torch.randperm(len(outside),generator=g).tolist()[:len(tar)]]
  xo=torch.tensor(x[sub[s]],device="cuda");xo=xo[:,q["last"]["perm"][s].cuda()] if env=="PM" else xo;p=[v[j:j+1].cuda() for v in q["parameters"]];z=base.forward(p,xo[None],torch.ones(1,1,1,dtype=torch.bool,device="cuda"))[0 if L==1 else 2][0];dl=torch.clamp(-1-z.mean(0)[tar],min=0).cpu() if tar else torch.empty(0);sels.append(dict(seed=s,pool=pool,target20=tar,target10=tar[:10],target5=tar[:5],control=ctl,deltas20=dl.tolist(),control_persistent_overlap=sorted(set(ctl)&set(pool)),cohort_status="OK" if len(pool)>=5 else "COHORT_TOO_SMALL",target_rng_seed=seed("target",s,env,T,L),control_rng_seed=seed("control",s,env,T,L)));aux[s]=(j,dl)
 for s in sels:
  assert s["target5"]==s["target10"][:len(s["target5"])] and s["target10"]==s["target20"][:len(s["target10"])]
  assert len(s["control"])==len(s["target20"]) and not(set(s["control"])&set(s["target20"]))
 jw(job/"selections.json",{"seeds":sels});models=[dict(seed=s,env=env,act="ELU1",iv=b,branch=b) for s in range(3) for b in BR];e=Engine(models)
 for k in range(6):e.p[k].copy_(torch.stack([q["parameters"][k][aux[m["seed"]][0]] for m in models]).cuda());e.m[k].copy_(torch.stack([q["adam_m"][k][aux[m["seed"]][0]] for m in models]).cuda());e.v[k].copy_(torch.stack([q["adam_v"][k][aux[m["seed"]][0]] for m in models]).cuda())
 e.t.copy_(q["t"].cuda());fm=[torch.zeros_like(p,dtype=torch.bool) for p in e.p];before=[p.clone() for p in e.p];before_m=[p.clone() for p in e.m];before_v=[p.clone() for p in e.v]
 # The intervention shock is measured on the just-completed checkpoint task.
 for j,m in enumerate(models):
  s=m["seed"];xo=torch.tensor(x[sub[s]],device="cuda");xo=xo[:,q["last"]["perm"][s].cuda()] if env=="PM" else xo;e.cx[j].copy_(xo);yy=torch.tensor(y[sub[s]],device="cuda") if env=="PM" else q["last"]["labels"][s].cuda();e.cy[j].copy_(torch.nn.functional.one_hot(yy,10))
 premet,_=e.evaluate()
 for j,m in enumerate(models):
  s=sels[m["seed"]];ii=ids(m["branch"],s["target20"],s["control"])
  if lift(m["branch"]) and ii:e.p[2*(L-1)+1][j].index_add_(0,torch.tensor(ii,device="cuda"),aux[m["seed"]][1][:len(ii)].cuda())
  if freeze(m["branch"]):
   for aa,bb in zip(fm,masks(e,L,ii)):aa[j].copy_(bb[j])
 # Lift changes only selected biases; optimizer state and all weights are identical.
 bidx=2*(L-1)+1;allowed=torch.zeros_like(e.p[bidx],dtype=torch.bool)
 for j,m in enumerate(models):
  if lift(m["branch"]):allowed[j,ids(m["branch"],sels[m["seed"]]["target20"],sels[m["seed"]]["control"])]=1
 delta_err=[]
 for j,m in enumerate(models):
  if lift(m["branch"]):
   ii=ids(m["branch"],sels[m["seed"]]["target20"],sels[m["seed"]]["control"])
   if ii:delta_err.append(float((e.p[bidx][j,ii]-(before[bidx][j,ii]+aux[m["seed"]][1][:len(ii)].cuda())).abs().max()))
 lift_checks={"nonbias_maxabs":max(float((a-b).abs().max()) for k,(a,b) in enumerate(zip(e.p,before)) if k!=bidx),"outside_selected_bias_maxabs":float((e.p[bidx][~allowed]-before[bidx][~allowed]).abs().max()),"selected_bias_expected_value_maxabs":max(delta_err,default=0.),"adam_m_maxabs":max(float((a-b).abs().max()) for a,b in zip(e.m,before_m)),"adam_v_maxabs":max(float((a-b).abs().max()) for a,b in zip(e.v,before_v)),"counter_change":float(e.t-q["t"].cuda())}
 assert max(abs(v) for v in lift_checks.values())==0
 e.set_freeze(fm);postmet,_=e.evaluate();pair={};ix={(m["seed"],m["branch"]):i for i,m in enumerate(models)}
 for s in range(3):
  for a,b in (("A","B20"),("C20","D20"),("C5","D5"),("C10","D10"),("RC20","RD20")):pair[f"s{s}_{a}_{b}"]=max(float((v[ix[s,a]]-v[ix[s,b]]).abs().max()) for v in e.p+e.m+e.v)
 assert max(pair.values())==0
 logits=base.forward(e.p,e.cx,e.elu)[-1];pair_logits={}
 for s in range(3):
  for a,b in (("A","B20"),("C20","D20"),("C5","D5"),("C10","D10"),("RC20","RD20")):pair_logits[f"s{s}_{a}_{b}"]=float((logits[ix[s,a]]-logits[ix[s,b]]).abs().max())
 assert max(pair_logits.values())==0
 reach={}
 for j,m in enumerate(models):
  if lift(m["branch"]) and not m["branch"].startswith("R"):
   ii=ids(m["branch"],sels[m["seed"]]["target20"],sels[m["seed"]]["control"]);dd=aux[m["seed"]][1][:len(ii)]
   if ii and bool((dd>0).any()):reach[f"s{m['seed']}_{m['branch']}"]=float((base.forward(e.p,e.cx,e.elu)[0 if L==1 else 2][j,:,ii].mean(0)[dd>0]+1).abs().max())
 assert not reach or max(reach.values())<2e-4
 for s in range(3):
  sels[s]["branch_shock"]={b:dict(pre_ce=float(premet["train_ce"][ix[s,b]]),post_ce=float(postmet["train_ce"][ix[s,b]]),pre_acc=float(premet["train_acc"][ix[s,b]]),post_acc=float(postmet["train_acc"][ix[s,b]])) for b in BR}
 jw(job/"selections.json",{"seeds":sels});e.capture();xs={s:torch.tensor(x[sub[s]],device="cuda") for s in range(3)};ys={s:torch.tensor(y[sub[s]],device="cuda") for s in range(3)};rows=[];units={};rh=[];fe=0.
 for task in range(T+1,T+6):
  o,p,l,h=draw(d);rh.append(dict(task=task,**h));oldrh=json.loads((PFX/"rng_hashes.json").read_text());assert task>50 or rh[-1]==oldrh[task-1],("continuation RNG mismatch",task);ost=torch.stack([o[m["seed"]] for m in models],1).cuda()
  for j,m in enumerate(models):s=m["seed"];e.cx[j].copy_(xs[s][:,p[s].cuda()] if env=="PM" else xs[s]);yy=ys[s] if env=="PM" else l[s].cuda();e.cy[j].copy_(torch.nn.functional.one_hot(yy,10))
  e.acc.zero_();e.ce.zero_();last=0;prev={}
  for step in MEAS:
   for z in range(last,step,base.BLOCK):e.indices.copy_(ost[z:z+base.BLOCK]);e.graph.replay()
   met,u=e.evaluate();assert all(np.isfinite(v).all() for v in list(met.values())+list(u.values()));zs=base.forward(e.p,e.cx,e.elu)
   for j,m in enumerate(models):
    key=f"s{m['seed']}_{m['branch']}_t{task}_u{step}";rows.append(dict(seed=m["seed"],env=env,checkpoint=T,layer=L,branch=m["branch"],task=task,step=step,ce=float(met["train_ce"][j]),acc=float(met["train_acc"][j]),online_ce=float(e.ce[j]/step) if step else "",online_acc=float(e.acc[j]/step) if step else ""))
    for ll,zi in ((1,0),(2,2)):
     w=2*(ll-1);win=e.p[w][j];units[f"{key}_q_l{ll}"]=u[f"lowgate_l{ll}"][j];units[f"{key}_zmean_l{ll}"]=u[f"zmean_l{ll}"][j];units[f"{key}_gate_mean_l{ll}"]=u[f"gate_mean_l{ll}"][j];units[f"{key}_Win_norm_l{ll}"]=win.flatten(1).norm(dim=1).cpu();units[f"{key}_Win_cnorm_l{ll}"]=(win-win.mean(1,keepdim=True)).flatten(1).norm(dim=1).cpu();units[f"{key}_Wout_norm_l{ll}"]=e.p[w+2][j].norm(dim=0).cpu();cur=(zs[zi][j].clone(),[v[j].clone() for v in e.p]);pk=(j,ll)
     if pk in prev:
      pz,pp=prev[pk];dz=cur[0]-pz;units[f"{key}_dz_mean_l{ll}"]=dz.mean(0).cpu();units[f"{key}_dz_sd_l{ll}"]=dz.std(0,unbiased=False).cpu();units[f"{key}_dWin_l{ll}"]=(cur[1][w]-pp[w]).flatten(1).norm(dim=1).cpu();units[f"{key}_db_l{ll}"]=(cur[1][w+1]-pp[w+1]).abs().cpu();units[f"{key}_dWout_l{ll}"]=(cur[1][w+2]-pp[w+2]).norm(dim=0).cpu()
     prev[pk]=cur
   fe=max(fe,max(float((a[z]-r[z]).abs().max()) if z.any() else 0 for a,z,r in zip(e.p,e.freeze_masks,e.frozen_reference)));last=step
  print(f"JOB {env} T{T} L{L} task {task}",flush=True)
 finite=all(torch.isfinite(v).all() for v in e.p+e.m+e.v) and all(np.isfinite(r[k]) for r in rows for k in ("ce","acc"));expected=(T+5)*base.STEPS
 assert len(rows)==990 and fe==0 and finite and int(e.t)==expected;cw(job/"rows.csv",rows);np.savez_compressed(job/"units.npz",**units);torch.save(dict(models=models,parameters=[v.cpu() for v in e.p],adam_m=[v.cpu() for v in e.m],adam_v=[v.cpu() for v in e.v],t=e.t.cpu(),rng_states=states(d)),job/"checkpoint.pt");jw(job/"provenance.json",dict(status="COMPLETE",prereg_commit=c,launch_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),spec_sha256=ss,code_sha256=sha(__file__),engine_sha256=sha(Path(__file__).with_name("elu_sunk_engine_0913.py")),base_sha256=sha(base.__file__),prefix_sha256=sha(pf),source_commit="46becd95b992d1ce27f95a6dc5b9cbdd2ad73f63",data_sha256={p.name:sha(p) for p in base.DATA.glob("*gz")},subset_sha256={s:ah(v) for s,v in sub.items()},rng_hashes=rh,pair_initial_maxabs=pair,pair_initial_logits_maxabs=pair_logits,lift_checks=lift_checks,lift_reach_maxabs=reach,freeze_invariant_maxabs=fe,row_count=len(rows),finite=bool(finite),final_counter=int(e.t),expected_counter=expected,torch_version=torch.__version__,device=torch.cuda.get_device_name(),tf32=False,deterministic=True))
def validate():
 cfg();OUT.mkdir(parents=True,exist_ok=True);e=Engine([dict(seed=0,env="RL",act="ELU1",iv="ref")]);g=torch.Generator().manual_seed(913);e.cx.copy_(torch.rand(e.cx.shape,generator=g).cuda());e.cy.copy_(torch.nn.functional.one_hot(torch.randint(10,(1,base.N),generator=g),10).cuda());ii=list(range(7));z=base.forward(e.p,e.cx,e.elu)[2];e.p[3][0].index_add_(0,torch.tensor(ii,device="cuda"),-10-z[0,:,ii].mean(0));z=base.forward(e.p,e.cx,e.elu)[2];d=torch.clamp(-1-z[0,:,ii].mean(0),min=0);assert bool((d>0).all());old=e.p[3].clone();e.p[3][0].index_add_(0,torch.tensor(ii,device="cuda"),d);changed=float((e.p[3]-old).abs().max());reach=float((base.forward(e.p,e.cx,e.elu)[2][0,:,ii].mean(0)+1).abs().max());m=masks(e,2,ii);e.set_freeze(m);e.t.fill_(41);e.indices.copy_(torch.randint(base.N,e.indices.shape,generator=g).cuda());e.block();fr=max(float((p[z]-r[z]).abs().max()) if z.any() else 0 for p,z,r in zip(e.p,e.freeze_masks,e.frozen_reference));assert changed>0 and reach<2e-4 and fr==0 and e.t==66;jw(OUT/"runner_validation.json",dict(status="PASS",code_sha256=sha(__file__),positive_delta_count=int((d>0).sum()),bias_changed_maxabs=changed,lift_reach_maxabs=reach,freeze_maxabs=fr,counter=float(e.t)));print("PASS")
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--prefix",action="store_true");a.add_argument("--validate",action="store_true");a.add_argument("--run",action="store_true");a.add_argument("--all",action="store_true");a.add_argument("--checkpoint",type=int,choices=(10,20,50));a.add_argument("--env",choices=("PM","RL"));a.add_argument("--layer",type=int,choices=(1,2));a.add_argument("--prereg");x=a.parse_args()
 try:
  if x.validate:validate()
  elif x.prefix:prefix(x.prereg)
  elif x.all:
   for e in ("PM","RL"):
    for t in (10,20,50):
     for l in (1,2):run(x.prereg,e,t,l)
  elif x.run and None not in (x.checkpoint,x.env,x.layer):run(x.prereg,x.env,x.checkpoint,x.layer)
  else:a.error("choose bounded mode and job coordinates")
 except Exception as exc:
  dest=PFX if x.prefix else OUT/f"{x.env}_t{x.checkpoint}_l{x.layer}" if x.run else OUT;dest.mkdir(parents=True,exist_ok=True);jw(dest/"failure.json",dict(status="TECHNICAL_FAILURE",exception=repr(exc),traceback=traceback.format_exc(),time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),code_sha256=sha(__file__)));raise

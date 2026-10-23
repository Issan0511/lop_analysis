"""Posthoc reanalysis; frozen methods in specs/spec_zero_attraction_reanalysis_0913.md."""
from pathlib import Path
import csv, hashlib, itertools, json, subprocess, time
import numpy as np
import torch
torch.set_num_threads(1)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
DATA=Path("/home/issan/Projects/obsidian-research-data/act_offset_review_0908")
OUT=ROOT/"results/zero_attraction_reanalysis_0913"
ARMS=[
 ("A","LRoff0_1216",0.,500),
 ("A","LRoffm0p5_1216",-.5,500),
 ("A","LRoffp0p5_1216",.5,500),
 ("B","LRoff0_lr0p00125_1216",0.,4000),
 ("B","LRoffm2_lr0p00125_1216",-2.,4000),
 ("B","LRoffp2_lr0p00125_1216",2.,4000)]
def arr(x):return x.detach().cpu().numpy().astype(np.float64)
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(2**20),b""):h.update(b)
 return h.hexdigest()
def writecsv(name,rows):
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with (OUT/name).open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=keys,lineterminator="\n");w.writeheader();w.writerows(rows)
def verify_close(key,a,b,atol,rtol,checks):
 a=np.asarray(a);b=np.asarray(b)
 assert a.shape==b.shape,(key,a.shape,b.shape)
 e=float(np.max(abs(a-b))) if a.size else 0.
 checks[key]=max(checks.get(key,0.),e)
 assert np.allclose(a,b,atol=atol,rtol=rtol),(key,e)
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 assert not (OUT/"sources.json").exists(),"Use a fresh output directory."
 sources=[];checks={};rows=[];groups=[];ledgers=[];trajectories={};endpoints=[]
 begin=time.monotonic()
 bits=np.array(list(itertools.product([-.5,.5],repeat=5)))
 for ladder,arm,q,last in ARMS:
  z0=-q/.1 if q>0 else -q
  cp={}
  for t in [0,last*10000]:
   p=DATA/"tail/ckpts"/f"{arm}_step{t}.pt"
   cp[t]=torch.load(p,map_location="cpu",weights_only=True)
   sources.append({"path":str(p),"bytes":p.stat().st_size,"sha256":sha(p)})
  runs=cp[0]["runs"];assert len(runs)==10 and sorted(r["seed"] for r in runs)==list(range(10))
  assert cp[0]["act_alpha"]==.1
  arm_tr=[]
  for ri,run in enumerate(runs):
   seed=int(run["seed"]);p=DATA/"full/logs"/f"{arm}_seed{seed}.npz"
   sources.append({"path":str(p),"bytes":p.stat().st_size,"sha256":sha(p)})
   with np.load(p) as z:
    step=z["step"].astype(np.int64)
    ws=z["layer1_w_free_step"].astype(np.int64)
    assert np.array_equal(ws,np.arange(last+1)*10000)
    ix=np.searchsorted(step,ws);assert np.array_equal(step[ix],ws)
    u=z["layer1_w_free"].astype(np.float64)
    vals={k:z[k][ix].astype(np.float64) for k in ["layer1_zbar","layer1_zmin","layer1_zmax","layer1_w_norm","layer1_v_unit"]}
   assert u.shape==(last+1,100,5)
   assert all(np.all(np.isfinite(v)) for v in [u,*vals.values()])
   n2=(u*u).sum(-1);w2=vals["layer1_w_norm"]**2
   mu=vals["layer1_zbar"];v=vals["layer1_v_unit"]
   half=.5*np.abs(u).sum(-1)
   verify_close("support_min",mu-half,vals["layer1_zmin"],5e-4,1e-4,checks)
   verify_close("support_max",mu+half,vals["layer1_zmax"],5e-4,1e-4,checks)
   sample=np.array(sorted(set([0,1,2,20,100,last-49,last])))
   supp=mu[sample][None]+np.einsum("pj,thj->pth",bits,u[sample])
   verify_close("variance_identity",np.var(supp,axis=0),n2[sample]/4,1e-10,1e-10,checks)
   for ti in [0,last]:
    n=cp[ti*10000]["net"]
    ww=arr(n["W"][ri])
    verify_close("checkpoint_free",u[ti],ww[:,15:20],1e-6,1e-5,checks)
    verify_close("checkpoint_full_norm",np.sqrt(w2[ti]),np.linalg.norm(ww,axis=-1),5e-5,1e-5,checks)
    verify_close("full_free_fixed_identity",(ww*ww).sum(-1),(ww[:,:15]**2).sum(-1)+(ww[:,15:]**2).sum(-1),1e-10,1e-10,checks)
   du=np.diff(u,axis=0);dot=(u[:-1]*du).sum(-1)
   Q=(du*du).sum(-1);R=2*dot;G=np.diff(n2,axis=0)
   verify_close("task_ledger",G,Q+R,1e-10,1e-10,checks)
   ND=np.sqrt(n2[:-1]*Q);valid=(np.sqrt(n2[:-1])>1e-12)&(np.sqrt(Q)>1e-12)
   c=np.full_like(dot,np.nan);np.divide(-dot,ND,out=c,where=valid)
   assert not np.any(np.abs(c[valid])>1+1e-9)
   # Seed trajectory summaries retain every endpoint, ALL units.
   tr=np.column_stack([ws/10000,np.sqrt(n2.mean(-1)),np.sqrt(w2.mean(-1)),np.sqrt(n2.mean(-1))/2,np.sqrt(((mu-z0)**2).mean(-1)),np.sqrt((v*v).mean(-1))])
   arm_tr.append(tr)
   windows=[("initial",0,0),("early",2,20),("middle",21,100),("late",last-49,last),("final",last,last)]
   for label,lo,hi in windows:
    idx=np.arange(lo,hi+1);nn=n2[idx];ww=w2[idx];mm=mu[idx];vv=v[idx]
    row={"ladder":ladder,"arm":arm,"offset":q,"root":z0,"seed":seed,"window":label,"task_start":lo,"task_end":hi,"n_units":100,
     "free_rms":float(np.sqrt(nn.mean())),"full_w_rms":float(np.sqrt(ww.mean())),"sigma_rms":float(np.sqrt(nn.mean())/2),
     "root_distance_rms":float(np.sqrt(((mm-z0)**2).mean())),"v_rms":float(np.sqrt((vv*vv).mean())),
     "free_growth_ratio":float(np.sqrt(nn.mean()/n2[0].mean())),
     "full_w_growth_ratio":float(np.sqrt(ww.mean()/w2[0].mean())),
     "unit_shrink_fraction":float((nn.mean(0)<n2[0]).mean()),
     "initial_zero_free_count":int((n2[0]<=1e-24).sum())}
    rows.append(row)
    if label in ["early","middle","late"]:
     # task t: endpoint t-1 -> endpoint t. No task 1 included.
     s=np.arange(lo-1,hi);nd=ND[s]
     ledgers.append({"ladder":ladder,"arm":arm,"seed":seed,"window":label,"task_start":lo,"task_end":hi,
      "Q_task_sum_unit_mean":float(Q[s].mean(-1).sum()),"R_task_sum_unit_mean":float(R[s].mean(-1).sum()),
      "G_task_sum_unit_mean":float(G[s].mean(-1).sum()),
      "c_effective":float(-dot[s].sum()/nd.sum()) if nd.sum()>1e-12 else None,
      "c_defined_count":int(valid[s].sum()),"c_undefined_count":int((~valid[s]).sum())})
   for eps in [.05,.1,.2]:
    near=np.abs(mu[-1]-z0)<=eps
    allnear=np.maximum(np.abs(vals["layer1_zmin"][-1]-z0),np.abs(vals["layer1_zmax"][-1]-z0))<=eps
    for name,mask in [("ALL",np.ones(100,dtype=bool)),("mean_near_root_FINAL_SELECTED",near),("mean_far_root_FINAL_SELECTED",~near),("all_inputs_near_root_FINAL_SELECTED",allnear)]:
     count=int(mask.sum())
     rr={"ladder":ladder,"arm":arm,"seed":seed,"offset":q,"root":z0,"window":"final","task":last,"eps":eps,"group":name,"count":count,"fraction":count/100}
     if count:
      good=mask&(n2[0]>1e-24)
      rr.update(free_rms=float(np.sqrt(n2[-1,mask].mean())),full_w_rms=float(np.sqrt(w2[-1,mask].mean())),
       free_growth_ratio=float(np.sqrt(n2[-1,mask].mean()/n2[0,mask].mean())),
       full_w_growth_ratio=float(np.sqrt(w2[-1,mask].mean()/w2[0,mask].mean())),
       median_unit_free_growth_ratio=float(np.median(np.sqrt(n2[-1,good]/n2[0,good]))) if good.any() else None,
       unit_shrink_fraction=float((n2[-1,mask]<n2[0,mask]).mean()),v_rms=float(np.sqrt((v[-1,mask]**2).mean())),
       root_distance_rms=float(np.sqrt(((mu[-1,mask]-z0)**2).mean())),sigma_rms=float(np.sqrt(n2[-1,mask].mean())/2))
     groups.append(rr)
   # Independent checkpoint endpoint aggregate, no NPZ trajectory summaries.
   w0=arr(cp[0]["net"]["W"][ri]);wf=arr(cp[last*10000]["net"]["W"][ri])
   f0=np.sum(w0[:,15:]**2,axis=1);ff=np.sum(wf[:,15:]**2,axis=1)
   verify_close("checkpoint_endpoint_ratio",np.array(np.sqrt(ff.mean()/f0.mean())),np.array(tr[-1,1]/tr[0,1]),1e-7,1e-6,checks)
   endpoints.append({"arm":arm,"seed":seed,"task":last,
    "free_growth_ratio":float(np.sqrt(ff.mean()/f0.mean())),
    "full_w_growth_ratio":float(np.linalg.norm(wf)/np.linalg.norm(w0)),
    "fixed15_growth_ratio":float(np.linalg.norm(wf[:,:15])/np.linalg.norm(w0[:,:15])),
    "bias_rms_initial":float(np.sqrt(np.mean(arr(cp[0]["net"]["b"][ri])**2))),
    "bias_rms_final":float(np.sqrt(np.mean(arr(cp[last*10000]["net"]["b"][ri])**2)))})
  trajectories[arm]=np.stack(arm_tr)
  print("DONE",arm,flush=True)
 rng=np.random.default_rng(20260913);comparisons=[]
 for ladder,arm,q,last in ARMS:
  if q==0:continue
  ref=next(a for l,a,qq,ll in ARMS if l==ladder and qq==0)
  for label in ["early","middle","late","final"]:
   for metric in ["free_growth_ratio","full_w_growth_ratio","unit_shrink_fraction","root_distance_rms","v_rms"]:
    a={r["seed"]:r[metric] for r in rows if r["arm"]==arm and r["window"]==label}
    b={r["seed"]:r[metric] for r in rows if r["arm"]==ref and r["window"]==label}
    d=np.array([a[s]-b[s] for s in range(10)]);boot=d[rng.integers(0,10,size=(5000,10))].mean(-1)
    lo,hi=np.quantile(boot,[.025,.975])
    comparisons.append({"ladder":ladder,"arm":arm,"reference":ref,"window":label,"metric":metric,"n_seeds":10,
      "mean_paired_difference":float(d.mean()),"ci_low":float(lo),"ci_high":float(hi),"positive_seeds":int((d>0).sum()),"negative_seeds":int((d<0).sum())})
 for name,data in [("seed_summary.csv",rows),("group_summary.csv",groups),("paired_comparisons.csv",comparisons),("task_ledger.csv",ledgers),("checkpoint_endpoints.csv",endpoints)]:writecsv(name,data)
 np.savez_compressed(OUT/"trajectories.npz",**trajectories)
 colors=["#65748c","#dc9e30","#168c86"]
 fig,axs=plt.subplots(4,2,figsize=(12,11),constrained_layout=True)
 for col,ladder in enumerate(["A","B"]):
  arms=[a for a in ARMS if a[0]==ladder]
  for color,(_,arm,q,last) in zip(colors,arms):
   tr=trajectories[arm]
   for row,(ci,label) in enumerate([(1,"Free-weight RMS norm"),(2,"Full W RMS norm"),(4,"RMS distance of mean from activation zero"),(5,"Readout v RMS")]):
    y=tr[:,:,ci];m=np.median(y,axis=0);lo,hi=np.quantile(y,[.25,.75],axis=0)
    axs[row,col].plot(tr[0,:,0],m,label=f"offset {q:+g}",color=color,lw=1.6)
    axs[row,col].fill_between(tr[0,:,0],lo,hi,alpha=.13,color=color)
    axs[row,col].set_ylabel(label);axs[row,col].set_xlabel("Task endpoint")
    axs[row,col].grid(alpha=.18);axs[row,col].set_xlim(0,last)
  axs[0,col].set_title(f"Ladder {ladder}: compare within column only")
  axs[0,col].legend()
 fig.suptitle("Existing trajectories: ALL units; seed median and IQR; posthoc reanalysis",fontsize=13)
 fig.savefig(OUT/"trajectories.png",dpi=150);fig.savefig(OUT/"trajectories.pdf");plt.close(fig)
 out=["# 零点への着座と重み収縮：既存ログの事後再解析 0913","",
 "格: 事後・記述的。新しい学習実験なし。ALL 100unit、各10seed。元の登録判定を変更しない。",
 "自由5bit重みはタスク内の変動入力を担う射影であり、MNISTの中心化Wとは異なる。",
 "表はseed別量の平均。late A=task451–500、B=task3951–4000。ラダー間を水準比較しない。","",
 "## late窓と初期step0の比","",
 "|ladder|arm|free RMS growth|full W RMS growth|unit shrink fraction|root distance RMS|v RMS|","|---|---|---:|---:|---:|---:|---:|"]
 for l,a,q,t in ARMS:
  rr=[r for r in rows if r["arm"]==a and r["window"]=="late"]
  vals=[np.mean([r[k] for r in rr]) for k in ["free_growth_ratio","full_w_growth_ratio","unit_shrink_fraction","root_distance_rms","v_rms"]]
  out.append(f"|{l}|{a}|"+"|".join(f"{x:.6g}" for x in vals)+"|")
 out+=["","## 同じラダーのoffset0との差：late free growth","",
 "|arm|mean paired difference|95% descriptive bootstrap CI|negative seeds|","|---|---:|---|---:|"]
 for r in comparisons:
  if r["window"]=="late" and r["metric"]=="free_growth_ratio":
   out.append(f'|{r["arm"]}|{r["mean_paired_difference"]:.6g}|[{r["ci_low"]:.6g}, {r["ci_high"]:.6g}]|{r["negative_seeds"]}/10|')
 out+=["","## finalの平均零点近傍群（±0.1）：結果を見て選ぶ記述群","",
 "空群は群内平均から除外し件数を明記する。この群比較から因果を推論しない。",
 "|arm|near units / 1000|seeds with near units|near free growth, seed mean|near shrink fraction, seed mean|","|---|---:|---:|---:|---:|"]
 for l,a,q,t in ARMS:
  rr=[r for r in groups if r["arm"]==a and r["eps"]==.1 and r["group"]=="mean_near_root_FINAL_SELECTED"]
  good=[r for r in rr if r["count"]]
  out.append(f'|{a}|{sum(r["count"] for r in rr)}|{len(good)}|'+(f'{np.mean([r["free_growth_ratio"] for r in good]):.6g}|{np.mean([r["unit_shrink_fraction"] for r in good]):.6g}|' if good else "NA|NA|"))
 out+=["","## lateのfree重み二乗収支（task合計・unit平均・seed平均）","",
 "|arm|Q: squared displacement|R: radial term|G: norm square change|","|---|---:|---:|---:|"]
 for l,a,q,t in ARMS:
  rr=[r for r in ledgers if r["arm"]==a and r["window"]=="late"]
  out.append("|"+a+"|"+"|".join(f'{np.mean([r[k] for r in rr]):.6g}' for k in ["Q_task_sum_unit_mean","R_task_sum_unit_mean","G_task_sum_unit_mean"])+"|")
 out+=["","## 限界","",
 "- 零点近傍への着座と収縮の同居は、自己項による因果や高い傾きの効果を同定しない。",
 "- 特に読み出しvが縮む場合は、出力寄与と学習への参加も別に調べる必要がある。",
 "- free収縮と全W増大は両立する。固定15bitとbiasは平均位置に寄与する。",
 "- task終端間の収支から、個々の更新における自己項・rest・Adam経路を復元できない。",
 "- 侵食cはこのfree射影の値であり、従来MNIST全中心化Wのcと同一の測定ではない。",
 "- Aのoffset±2は元実験で発散したため含まれない。","",
 "検算: verification.json。入力出所: sources.json。初期/finalの独立再集計: checkpoint_endpoints.csv。",
 "図: trajectories.png / trajectories.pdf。表の再集計元は各CSV。"]
 (OUT/"summary.md").write_text("\n".join(out)+"\n",encoding="utf-8")
 (OUT/"verification.json").write_text(json.dumps({"status":"PASS","max_abs_errors":checks,"n_files":60,"n_seeds_per_arm":10,"n_units":100,"support_variance_check_tasks":[0,1,2,20,100,"last-49","last"],"elapsed_seconds":time.monotonic()-begin},indent=2))
 head=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
 (OUT/"sources.json").write_text(json.dumps({"analysis_source_sha256":sha(Path(__file__)),"preanalysis_commit":head,"input_files":sources,"columns_trajectory":["task","free_rms","full_w_rms","sigma_rms","root_distance_rms","v_rms"]},indent=2))
 print((OUT/"summary.md").read_text(),flush=True)
if __name__=="__main__":main()

"""Analysis of the frozen 19-condition SGD experiment."""
from pathlib import Path
import argparse,csv,hashlib,itertools,json,subprocess
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"results/zero_attraction_learning_0913"
DATA=Path("/home/issan/Projects/claude/zero_attraction_0913/results/zero_attraction_learning_0913")
CFG=json.loads((ROOT/"configs/zero_attraction_learning_0913.yaml").read_text())
def writecsv(name,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=keys,lineterminator="\n");w.writeheader();w.writerows(rows)
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(2**20),b""):h.update(b)
 return h.hexdigest()
def fn(row,z):
 act=row["activation"];aa=row["dial"];meta=CFG["activation"][act];q=meta.get("offset",0.)
 if row["family"]=="leaky":
  return np.where(z>0,z,aa*z)+q,np.where(z>0,1.,aa)
 phase=meta.get("phase","normal")
 if phase=="normal":return z+np.sin(aa*z)**2/aa+q,1+np.sin(2*aa*z)
 sign=1 if phase=="peak" else -1
 return z+sign*np.sin(2*aa*z)/(2*aa)+q,1+sign*np.cos(2*aa*z)
def root(row):
 if CFG["activation"][row["activation"]].get("offset",0)==0:return 0.
 l,h=-100.,100.
 for _ in range(90):
  m=(l+h)/2
  if fn(row,m)[0]>0:h=m
  else:l=m
 ans=(l+h)/2;assert abs(fn(row,ans)[0])<1e-12
 return ans
def main():
 global CFG,OUT
 ap=argparse.ArgumentParser();ap.add_argument("--available",action="store_true");args=ap.parse_args()
 if args.available:
  CFG=dict(CFG)
  CFG["arms"]=[a for a in CFG["arms"] if all((DATA/"logs"/f"{a['name']}_seed{s}.npz").exists() for s in range(10)) and (DATA/"ckpts"/f"{a['name']}_step5000000.pt").exists()]
  names={a["name"] for a in CFG["arms"]}
  if not {"SN_peak_q0","SN_normal_q0"}<=names:
   print("Primary pair is not complete yet.");return
  OUT=OUT/"partial_analysis";OUT.mkdir(exist_ok=True)
 OUT.mkdir(parents=True,exist_ok=True)
 rows=[];groups=[];ledgers=[];sources=[];trajectories={};checks={};rawstats={}
 bits=np.array(list(itertools.product([-.5,.5],repeat=5)))
 def verify(k,a,b,atol=1e-6,rtol=1e-5):
  err=float(np.max(abs(np.asarray(a)-np.asarray(b))));checks[k]=max(checks.get(k,0),err)
  assert np.allclose(a,b,atol=atol,rtol=rtol),(k,err)
 for arm in CFG["arms"]:
  name=arm["name"];z0=root(arm);ars=[];rawstats[name]=[]
  cp={}
  for step in arm["checkpoints"]:
   p=DATA/"ckpts"/f"{name}_step{step}.pt"
   cp[step]=torch.load(p,map_location="cpu",weights_only=True)
   sources.append({"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size})
  for seed in range(10):
   p=DATA/"logs"/f"{name}_seed{seed}.npz"
   sources.append({"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size})
   with np.load(p) as z:
    steps=z["step"].astype(int);ws=z["layer1_w_free_step"].astype(int)
    assert np.array_equal(ws,np.arange(501)*10000)
    ix=np.searchsorted(steps,ws);assert np.array_equal(steps[ix],ws)
    u=z["layer1_w_free"].astype(float)
    mu=z["layer1_zbar"][ix].astype(float);v=z["layer1_v_unit"][ix].astype(float)
    w=z["layer1_w_norm"][ix].astype(float)
    loss=z["eval_loss_exact"][ix].astype(float);unfit=z["unfit"][ix].astype(float)
    zmn=z["layer1_zmin"][ix].astype(float);zmx=z["layer1_zmax"][ix].astype(float)
    assert np.all(z["lr_used"]==CFG["common_overrides"]["lr_main"])
   n2=(u*u).sum(-1);full2=w*w
   assert all(np.isfinite(x).all() for x in [u,mu,v,w,loss,unfit])
   half=.5*abs(u).sum(-1)
   verify("zmin",mu-half,zmn,5e-5,1e-5);verify("zmax",mu+half,zmx,5e-5,1e-5)
   support=mu[None]+np.einsum("pj,thj->pth",bits,u)
   verify("support_variance",support.var(0),n2/4,1e-10,1e-10)
   aa,gate=fn(arm,support);outvar=(aa*v[None]).var(0);g2=gate.square().mean(0) if hasattr(gate,"square") else (gate*gate).mean(0)
   effective=v*v*g2
   for step in arm["checkpoints"]:
    k=step//10000;n=cp[step]["net"];wi=n["W"][seed].double().numpy()
    verify("checkpoint_free",u[k],wi[:,15:],1e-6,1e-5);verify("checkpoint_full",w[k],np.linalg.norm(wi,axis=-1),5e-5,1e-5)
   du=np.diff(u,axis=0);Q=(du*du).sum(-1);dot=(u[:-1]*du).sum(-1);R=2*dot;G=np.diff(n2,axis=0);ND=np.sqrt(n2[:-1]*Q)
   verify("task_ledger",G,Q+R,1e-10,1e-10)
   for win,lo,hi in [("initial",0,0),("early",2,20),("middle",21,100),("late",451,500),("final",500,500)]:
    s=slice(lo,hi+1)
    row={"arm":name,"family":arm["family"],"alpha":arm["dial"],"offset":CFG["activation"][arm["activation"]].get("offset",0),"root":z0,"seed":seed,"window":win,"task_start":lo,"task_end":hi,
     "free_growth":float(np.sqrt(n2[s].mean()/n2[0].mean())),"full_growth":float(np.sqrt(full2[s].mean()/full2[0].mean())),
     "free_rms":float(np.sqrt(n2[s].mean())),"full_rms":float(np.sqrt(full2[s].mean())),
     "root_distance_rms":float(np.sqrt(((mu[s]-z0)**2).mean())),"zmean":float(mu[s].mean()),
     "near_mean_fraction":float((abs(mu[s]-z0)<=.1).mean()),
     "near_all_inputs_fraction":float((np.maximum(abs(zmn[s]-z0),abs(zmx[s]-z0))<=.1).mean()),
     "unit_shrink_fraction":float((n2[s].mean(0)<n2[0]).mean()),
     "median_unit_growth":float(np.median(np.sqrt(n2[s].mean(0)/np.maximum(n2[0],1e-24)))),
     "v_rms":float(np.sqrt((v[s]**2).mean())),"effective_v2_gate2":float(effective[s].mean()),
     "unit_output_variance":float(outvar[s].mean()),"loss":float(loss[s].mean()),"unfit":float(unfit[s].mean())}
    rows.append(row)
    if win in ["early","middle","late"]:
     s=slice(lo-1,hi);valid=ND[s]>1e-12
     ledgers.append({"arm":name,"seed":seed,"window":win,"Q":float(Q[s].mean(-1).sum()),"R":float(R[s].mean(-1).sum()),"G":float(G[s].mean(-1).sum()),
     "c_effective":float(-dot[s].sum()/ND[s].sum()) if ND[s].sum()>1e-12 else None,"undefined_count":int((~valid).sum())})
   for eps in [.05,.1,.2]:
    mask=abs(mu[-1]-z0)<=eps
    for label,sel in [("ALL",np.ones(100,dtype=bool)),("near_FINAL_SELECTED",mask),("far_FINAL_SELECTED",~mask)]:
     rr={"arm":name,"seed":seed,"eps":eps,"group":label,"n":int(sel.sum()),"shrink_count":int(((n2[-1]<n2[0])&sel).sum())}
     if sel.any():rr.update(free_growth=float(np.sqrt(n2[-1,sel].mean()/n2[0,sel].mean())),
      median_unit_growth=float(np.median(np.sqrt(n2[-1,sel]/np.maximum(n2[0,sel],1e-24)))),
      v_rms=float(np.sqrt((v[-1,sel]**2).mean())),effective_v2_gate2=float(effective[-1,sel].mean()),output_variance=float(outvar[-1,sel].mean()))
     groups.append(rr)
   tr=np.column_stack([np.arange(501),np.sqrt(n2.mean(-1)),np.sqrt(full2.mean(-1)),mu.mean(-1),np.sqrt(((mu-z0)**2).mean(-1)),np.sqrt((v*v).mean(-1)),loss,unfit,(abs(mu-z0)<=.1).mean(-1)])
   ars.append(tr)
  trajectories[name]=np.stack(ars)
  print("AGGREGATED",name,flush=True)
 rng=np.random.default_rng(2026091301)
 def values(name,key="free_growth",window="late"):return np.array([r[key] for r in rows if r["arm"]==name and r["window"]==window])
 def ci(x):
  boot=x[rng.integers(0,len(x),size=(5000,len(x)))].mean(-1)
  return [float(v) for v in np.quantile(boot,[.025,.975])]
 delta=np.log(values("SN_peak_q0"))-np.log(values("SN_normal_q0"));interval=ci(delta)
 label="STRONG_PHASE_SUPPRESSION" if interval[1]<np.log(.8) else "DIRECTIONAL_PHASE_SUPPRESSION" if interval[1]<0 else "PHASE_INCREASE" if interval[0]>0 else "INCONCLUSIVE"
 absolute=ci(np.log(values("SN_peak_q0")))
 verdict={"primary":label,"primary_log_growth_difference":float(delta.mean()),"primary_ci":interval,
  "geometric_growth_ratio_peak_over_normal":float(np.exp(delta.mean())),
  "population_shrinkage":"POPULATION_SHRINKAGE" if absolute[1]<0 else "NOT_ESTABLISHED",
  "peak_log_growth_ci":absolute,"window":[451,500],"n_seeds":10,"prereg_commit":"1689251","run_head":"0ba13e8",
  "interpretation":"Natural activation effect in scalar-MSE SGD; not isolation of self-term causality or evidence for CE/Adam."}
 (OUT/"verdict.json").write_text(json.dumps(verdict,indent=2));writecsv("verdict.csv",[dict(primary=label,log_difference=float(delta.mean()),ci_low=interval[0],ci_high=interval[1],peak_geometric_F=float(np.exp(np.log(values("SN_peak_q0")).mean())),population_shrinkage=verdict["population_shrinkage"])])
 comparisons=[]
 pairs=[("SN_peak_q0","SN_normal_q0"),("SN_valley_q0","SN_normal_q0")]
 for a in ["0p1","0p3","0p7"]:
  pairs.extend([(f"LR_a{a}_qm05",f"LR_a{a}_q0"),(f"LR_a{a}_qp05",f"LR_a{a}_q0")])
 for a,b in pairs:
  if a not in trajectories or b not in trajectories:continue
  d=np.log(values(a))-np.log(values(b));lo,hi=ci(d)
  comparisons.append({"arm":a,"reference":b,"log_growth_difference":float(d.mean()),"ci_low":lo,"ci_high":hi,"negative_seeds":int((d<0).sum()),"tier":"PRIMARY" if (a,b)==pairs[0] else "EXPLORATORY"})
 writecsv("comparisons.csv",comparisons);writecsv("seed_summary.csv",rows);writecsv("group_summary.csv",groups);writecsv("task_ledger.csv",ledgers)
 np.savez_compressed(OUT/"summary_trajectories.npz",**trajectories)
 (OUT/"analysis_verification.json").write_text(json.dumps({"status":"PASS","max_abs_errors":checks,"n_arms":len(CFG["arms"]),"n_logs":len(CFG["arms"])*10,"source_files":sources,"analysis_sha256":sha(Path(__file__))},indent=2))
 # Scientific trajectories: median and interquartile range across seed-level summaries.
 fig,axs=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
 sets=[(["SN_normal_q0","SN_peak_q0","SN_valley_q0","LIN_q0"],1,"Snake phase: free-weight RMS"),
       (["SN_normal_q0","SN_peak_q0","SN_valley_q0","LIN_q0"],3,"Snake phase: mean preactivation"),
       (["SN_normal_q0","SN_peak_q0","SN_valley_q0","LIN_q0"],5,"Snake phase: readout RMS"),
       (["LR_a0p1_q0","LR_a0p3_q0","LR_a0p7_q0","LIN_q0"],1,"Leaky slope: free-weight RMS"),
       (["LR_a0p1_qm05","LR_a0p1_q0","LR_a0p1_qp05"],1,"Leaky a=.1 offsets: free-weight RMS"),
       (["SN_normal_q0","SN_peak_q0","SN_valley_q0","LIN_q0"],6,"Snake phase: exact MSE")]
 for ax,(names,col,title) in zip(axs.flat,sets):
  for name in names:
   if name not in trajectories:continue
   x=trajectories[name][:,:,col];med=np.median(x,axis=0);lo,hi=np.quantile(x,[.25,.75],axis=0)
   label=name.replace("SN_","").replace("LR_","").replace("_q0","")
   ax.plot(np.arange(501),med,label=label,lw=1.5);ax.fill_between(np.arange(501),lo,hi,alpha=.12)
  ax.set_title(title);ax.set_xlabel("Task endpoint");ax.legend(fontsize=8);ax.grid(alpha=.18)
 fig.suptitle("New learning: 10 seeds, ALL units, common SGD lr=.005; median and IQR")
 fig.savefig(OUT/"learning_trajectories.png",dpi=150);fig.savefig(OUT/"learning_trajectories.pdf");plt.close(fig)
 out=["# 零点復元と重み収縮：新しいSGD学習実験 0913","",
 f"登録主判定: **{label}**。集計完了{len(CFG['arms'])}条件・各10seed・500task・共通lr=.005。",f"peak/normalの幾何平均free成長比: {verdict['geometric_growth_ratio_peak_over_normal']:.6g}、log差CI={interval}。",f"peak集団の初期からの収縮: **{verdict['population_shrinkage']}**。","",
 ("**途中集計: 完走した条件のみ。残りの条件は実行中。**" if args.available else "全条件の最終集計。"),
 "主比較以外は探索。free重みは入力末尾5bit、全WやMNIST中心化Wと区別。","",
 "## late task451–500、全unit、seed平均","",
 "|arm|free growth|full growth|shrunken units|near-root mean|near-root ALL inputs|v RMS|unit output variance|MSE|",
 "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
 for a in CFG["arms"]:
  name=a["name"];rr=[r for r in rows if r["arm"]==name and r["window"]=="late"]
  keys=["free_growth","full_growth","unit_shrink_fraction","near_mean_fraction","near_all_inputs_fraction","v_rms","unit_output_variance","loss"]
  out.append("|"+name+"|"+"|".join(f'{np.mean([r[k] for r in rr]):.6g}' for k in keys)+"|")
 out+=["","## 初期・early・lateの損失：水準と劣化を分ける","",
 "|arm|initial MSE|early2–20 MSE|late451–500 MSE|late-early|","|---|---:|---:|---:|---:|"]
 for a in CFG["arms"]:
  n=a["name"];ini=values(n,"loss","initial").mean();early=values(n,"loss","early").mean();late=values(n,"loss","late").mean()
  out.append(f"|{n}|{ini:.6g}|{early:.6g}|{late:.6g}|{late-early:+.6g}|")
 out+=["","## 限界","",
 "- 初期W,b,vと入力/教師の乱数列は一致。初期予測と損失は活性化変更で異なる。形状の自然学習効果を測った実験。",
 "- 同じ初期予測を固定補正で揃える介入、CE/Adam、長期射影は未実施。",
 "- 零点近傍unitを最終状態で選んだ群比較は記述。独立標本はseedでありunit数ではない。",
 "- 縮小と出力不参加を分けるため、v・有効ゲート・出力分散を保存。縮小だけで可塑性維持と呼ばない。",
 "- 原点傾き2のSnakeは明示式z+sin(2z)/2による新条件。回収不能な研究室PCの旧実験の忠実再現とは呼ばない。",
 "- プローブの局所復元は他unitとvを固定した測定。自然な全員学習で同じ因果を確定するものではない。",
 "- task終端のcはfree重みの射影だけに定義。N,D,cを含む収支から符号を読む。","",
 "検算: analysis_verification.json。機序測定: mechanism/。再現情報: provenance.json。"]
 (OUT/"summary.md").write_text("\n".join(out)+"\n")
 print(json.dumps(verdict,indent=2),flush=True)
if __name__=="__main__":main()

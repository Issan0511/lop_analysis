"""Descriptive unit tracking on saved Snake trajectories. No causal verdict."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
DATA=Path("/home/issan/Projects/claude/zero_attraction_0913/results/zero_attraction_learning_0913/logs")
OUT=ROOT/"results/unit_fates_0913"
def writecsv(p,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with p.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 flux=[];cohort=[];future=[];population=[];sources=[];traces={};check_error=0
 old=list(csv.DictReader((ROOT/"results/zero_attraction_learning_0913/seed_summary.csv").open()))
 for name in ["SN_normal_q0","SN_peak_q0","SN_valley_q0"]:
  ts=[]
  for seed in range(10):
   p=DATA/f"{name}_seed{seed}.npz"
   sources.append(dict(path=str(p),sha256=hashlib.file_digest(p.open("rb"),"sha256").hexdigest()))
   with np.load(p) as log:
    steps=log["step"];ws=log["layer1_w_free_step"];ix=np.searchsorted(steps,ws)
    assert np.array_equal(ws,np.arange(501)*10000) and np.array_equal(steps[ix],ws)
    u=log["layer1_w_free"].astype(float);mu=log["layer1_zbar"][ix].astype(float);v=log["layer1_v_unit"][ix].astype(float)
   assert u.shape==(501,100,5) and mu.shape==v.shape==(501,100)
   assert all(np.isfinite(a).all() for a in [u,mu,v])
   n2=(u*u).sum(-1);growth=np.sqrt(n2/n2[0]);half=.5*np.abs(u).sum(-1)
   # Final membership is used only for retrospective trajectories.
   for label,sel in [("final_shrink",growth[-1]<1),("final_grow",growth[-1]>=1)]:
    for t in [0,1,5,20,100,500]:
     future.append(dict(arm=name,seed=seed,group=label,task=t,n=int(sel.sum()),median_growth=float(np.median(growth[t,sel])),mean_abs_v=float(np.abs(v[t,sel]).mean()),mean_abs_z=float(np.abs(mu[t,sel]).mean()),free_rms=float(np.sqrt(n2[t,sel].mean()))))
   for eps in [.05,.1,.2]:
    for metric in ["mean","all_inputs"]:
     state=np.abs(mu)<=eps if metric=="mean" else np.abs(mu)+half<=eps
     nn=state.sum(-1);ins=(~state[:-1]&state[1:]).sum(-1);outs=(state[:-1]&~state[1:]).sum(-1)
     assert np.array_equal(np.diff(nn),ins-outs)
     for t in range(501):
      flux.append(dict(arm=name,seed=seed,eps=eps,metric=metric,task=t,n=int(nn[t]),inflow=int(ins[t-1]) if t else 0,outflow=int(outs[t-1]) if t else 0))
     if eps==.1:
      prev=next(x for x in old if x['arm']==name and int(x['seed'])==seed and x['window']=='final')
      key='near_mean_fraction' if metric=='mean' else 'near_all_inputs_fraction'
      err=abs(nn[-1]/100-float(prev[key]));check_error=max(check_error,err);assert err<1e-12
     for start in [20,100]:
      for label,sel in [("inside",state[start]),("outside",~state[start])]:
       count=int(sel.sum());tail=state[start:,sel]
       bits=np.array([[a,b,c,d,e] for a in [-.5,.5] for b in [-.5,.5] for c in [-.5,.5] for d in [-.5,.5] for e in [-.5,.5]])
       z=mu[-1][None]+bits@u[-1].T
       phi=z+np.sin(z)**2 if name=="SN_normal_q0" else z+(1 if name=="SN_peak_q0" else -1)*np.sin(2*z)/2
       ov=(phi*v[-1]).var(0)
       first_switch=[]
       if count:
        for ui in np.flatnonzero(sel):
         loc=np.flatnonzero(state[start+1:,ui]!=state[start,ui]);first_switch.append(start+1+int(loc[0]) if len(loc) else None)
       row=dict(arm=name,seed=seed,eps=eps,metric=metric,start_task=start,group=label,n=count,final_inside=int(state[-1,sel].sum()),final_shrunk=int((growth[-1,sel]<1).sum()),ever_changed=int(sum(x is not None for x in first_switch)),never_changed=int(sum(x is None for x in first_switch)))
       if count:row.update(median_final_growth=float(np.median(growth[-1,sel])),final_v_rms=float(np.sqrt((v[-1,sel]**2).mean())),final_output_variance=float(ov[sel].mean()),mean_occupancy_after_start=float(tail.mean()))
       cohort.append(row)
     if eps==.1 and metric=="mean":
      ts.append(nn);population.append(dict(arm=name,seed=seed,start_count=int(nn[0]),task20_count=int(nn[20]),task100_count=int(nn[100]),final_count=int(nn[-1]),inflow_21_500=int(ins[20:].sum()),outflow_21_500=int(outs[20:].sum()),up_transitions_21_500=int((np.diff(nn)[20:]>0).sum()),down_transitions_21_500=int((np.diff(nn)[20:]<0).sum()),all_monotone_decreasing=bool((np.diff(nn)<=0).all())))
  traces[name]=np.stack(ts)
  print("TRACKED",name,flush=True)
 writecsv(OUT/"flux.csv",flux);writecsv(OUT/"cohorts.csv",cohort);writecsv(OUT/"retrospective_groups.csv",future);writecsv(OUT/"population.csv",population)
 (OUT/"verification.json").write_text(json.dumps(dict(status="PASS",n_logs=30,n_units=3000,n_task_endpoints=501,flow_identity="EXACT",old_final_fraction_max_error=check_error,source_files=sources,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
 fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
 for name,tr in traces.items():
  mean=tr.mean(0);lo,hi=np.quantile(tr,[.1,.9],axis=0)
  axs[0].plot(np.arange(501),mean,label=name);axs[0].fill_between(np.arange(501),lo,hi,alpha=.1)
  for label,style in [("final_shrink","-"),("final_grow","--")]:
   vals=[]
   for t in [0,1,5,20,100,500]:
    rr=[x for x in future if x['arm']==name and x['group']==label and x['task']==t]
    vals.append(np.mean([x['median_growth'] for x in rr]))
   axs[1].plot([0,1,5,20,100,500],vals,style,label=name+" "+label)
 axs[0].set_ylabel("Units with |mean z| <= 0.1, out of 100");axs[0].set_xlabel("Task endpoint");axs[0].legend(fontsize=8)
 axs[1].set_xscale("symlog",linthresh=1);axs[1].set_yscale("log");axs[1].axhline(1,c="black",lw=.6)
 axs[1].set_ylabel("Within-seed median free norm / initial norm");axs[1].set_xlabel("Task endpoint");axs[1].legend(fontsize=6)
 fig.suptitle("Saved Snake trajectories: center occupancy and retrospective final groups")
 fig.savefig(OUT/"unit_fates.png",dpi=160);fig.savefig(OUT/"unit_fates.pdf");plt.close(fig)
 def agg(rows,key):return float(np.mean([float(x[key]) for x in rows if key in x]))
 out=["# 中心群の出入りと発育：事後解析 0913","","3条件・各10seed。中心=unit平均zが±.1。全てtask端点。元のELU動画とは別条件。因果的な分かれ道は未同定。","",
 "|arm|task20中心数|task100|task500|延べ流入21–500|延べ流出21–500|","|---|---:|---:|---:|---:|---:|"]
 for name in traces:
  rr=[x for x in population if x['arm']==name]
  out.append("|"+name+"|"+"|".join(f"{agg(rr,k):.3f}" for k in ['task20_count','task100_count','final_count','inflow_21_500','outflow_21_500'])+"|")
 out += ["","## 固定cohort：task20で中心にいたunit（人数はseed平均、割合は群人数を合算した記述比）","","|arm|task20人数|500でも中心|500で初期より縮小|途中で一度も中心外に出ない|","|---|---:|---:|---:|---:|"]
 for name in traces:
  rr=[x for x in cohort if x['arm']==name and x['eps']==.1 and x['metric']=='mean' and x['start_task']==20 and x['group']=='inside'];n=sum(x['n'] for x in rr)
  out.append(f"|{name}|{n/10:.1f}|"+"|".join(f"{sum(x[k] for x in rr)/n:.3%}" for k in ['final_inside','final_shrunk','never_changed'])+"|")
 out += ["","最終縮小/成長の群分けは事後であり、初期特徴の差は原因とも独立検証済み予測とも呼ばない。中心に残る数と同じ個体の滞留を分ける。流入出は隣接task端点の状態変化のみで、task内の往復を取りこぼす。"]
 (OUT/"summary.md").write_text("\n".join(out)+"\n")
if __name__=="__main__":main()

"""Descriptive distribution and ledger plots; no additional hypothesis labels."""
from pathlib import Path
import argparse,csv,json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results/zero_attraction_learning_0913"
DATA=Path("/home/issan/Projects/claude/zero_attraction_0913/results/zero_attraction_learning_0913")
def mean(rows,key):return float(np.mean([float(r[key]) for r in rows]))
def writecsv(p,rows):
 with p.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--partial",action="store_true");args=ap.parse_args()
 out=BASE/("partial_analysis" if args.partial else "synthesis");out.mkdir(exist_ok=True)
 names=["SN_normal_q0","SN_peak_q0","SN_valley_q0"]
 fig,axs=plt.subplots(2,3,figsize=(13,8),constrained_layout=True);rows=[]
 for j,name in enumerate(names):
  cp=torch.load(DATA/"ckpts"/f"{name}_step5000000.pt",weights_only=True)
  cp0=torch.load(DATA/"ckpts"/f"{name}_step0.pt",weights_only=True)
  u=cp["net"]["W"][:,:,15:].double().numpy();u0=cp0["net"]["W"][:,:,15:].double().numpy()
  n2=(u*u).sum(-1);g=np.sqrt(n2/(u0*u0).sum(-1));v=cp["net"]["v"].double().numpy()
  mu=np.stack([np.load(DATA/"logs"/f"{name}_seed{s}.npz")["layer1_zbar"][-1] for s in range(10)])
  logg=np.log10(np.maximum(g,1e-12));grown=g>=1
  axs[0,j].hist(logg.ravel(),bins=np.linspace(-3,1,65),color=["C0","C1","C2"][j],alpha=.8)
  axs[0,j].axvline(0,c="black",ls="--",lw=1);axs[0,j].set_xlabel("log10(unit free norm / initial norm)")
  axs[0,j].set_ylabel("Units (descriptive, 10 seeds pooled)");axs[0,j].set_title(name)
  sc=axs[1,j].scatter(mu.ravel(),logg.ravel(),c=np.abs(v).ravel(),s=6,alpha=.5,cmap="viridis",vmin=0,vmax=.5)
  axs[1,j].axvspan(-.1,.1,color="C1",alpha=.12);axs[1,j].axhline(0,c="black",ls="--",lw=1)
  axs[1,j].set_xlabel("Final unit mean preactivation");axs[1,j].set_ylabel("log10(free norm growth)")
  for seed in range(10):
   rows.append(dict(arm=name,seed=seed,median_unit_growth=float(np.median(g[seed])),shrink_fraction=float((g[seed]<1).mean()),grown_units_final_energy_fraction=float(n2[seed,grown[seed]].sum()/n2[seed].sum()),near_root_fraction=float((abs(mu[seed])<=.1).mean())))
 fig.colorbar(sc,ax=list(axs[1,:]),label="Absolute readout weight |v|")
 fig.suptitle("Final task 500: shrinkage and growth coexist; unit plots are descriptive")
 fig.savefig(out/"unit_distributions.png",dpi=160);fig.savefig(out/"unit_distributions.pdf");plt.close(fig)
 writecsv(out/"distribution_seed_summary.csv",rows)
 trace=list(csv.DictReader((BASE/"natural_trace/seed_ledger.csv").open()))
 fig,axs=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
 for j,name in enumerate(names):
  for i,task in enumerate([21,101]):
   ax=axs[i,j];steps=[1,20,200,1000,10000]
   for k,label in [("G","Norm squared change G"),("cND","Inward numerator cND"),("K","Path term K")]:
    vals=[mean([r for r in trace if r["arm"]==name and int(r["task"])==task and int(r["updates"])==s],k) for s in steps]
    ax.plot(steps,vals,marker="o",label=label,lw=1.4)
   ax.axhline(0,c="black",lw=.7);ax.set_xscale("log");ax.set_xlabel("Updates after task switch")
   ax.set_title(f"{name}, task {task}");ax.grid(alpha=.2)
 axs[0,0].legend(fontsize=8)
 fig.suptitle("Actual SGD replay: seed mean of all units; initial motion and endpoint differ")
 fig.savefig(out/"transition_ledger.png",dpi=160);fig.savefig(out/"transition_ledger.pdf");plt.close(fig)
 mech=BASE/("mechanism_partial" if args.partial else "mechanism")
 mrows=list(csv.DictReader((mech/"seed_summary.csv").open()));ms=[]
 for name in sorted(set(r["arm"] for r in mrows)):
  for kind in ["mean","width"]:
   rs=[r for r in mrows if r["arm"]==name and int(float(r["step"]))==5000000 and r["kind"]==kind and float(r["eps"])==.05]
   ks=["kappa_total_mean","kappa_self_mean","kappa_rest_mean","kappa_total_positive_fraction","tangent_force_mean","self_tangent_force_mean","rest_tangent_force_mean","effective_v2_gate2_mean"]
   ms.append(dict(arm=name,task=500,kind=kind,eps=.05,**{k:mean(rs,k) for k in ks}))
 writecsv(out/"mechanism_condition_summary.csv",ms)
 note=["# 分布と機序の補助集計", "", "以下は記述的な追加集計。主判定を増やさない。", "",
 "|条件|seed内中央値の平均|収縮unit割合|増大unitが持つ最終二乗和割合|", "|---|---:|---:|---:|"]
 for name in names:
  rs=[r for r in rows if r["arm"]==name]
  note.append(f"|{name}|{mean(rs,'median_unit_growth'):.6g}|{mean(rs,'shrink_fraction'):.6g}|{mean(rs,'grown_units_final_energy_fraction'):.6g}|")
 note += ["", "最終状態で選んだ群の比率であり因果介入ではない。中心への集中と全体ノルムの増大が同時に起きる。", "",
 "局所kappa_total>0は、他unitとvを固定し、学習済み状態から小さくずらした場合の復元を表す。原点への復元とは限らず、全員が更新される自然学習の収縮を直接証明しない。", "",
 "task終端で誤差が小さいとself/restはほぼ打ち消し合う。自然更新でも、収束後の大きいself/restを10000回積むと残差が小さくなる。大きい自己項の積分だけを機序と読まない。", "",
 "cND=I_self+I_rest+I_round+K。G=Q-2(I_self+I_rest+I_round)。cの単純unit平均と、norm変化に入るcND平均は別の量。"]
 (out/"synthesis_notes.md").write_text("\n".join(note)+"\n")
 print("SYNTHESIS DONE",out)
if __name__=="__main__":main()

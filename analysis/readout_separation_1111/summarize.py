"""Paired-seed summaries for preregistered readout_separation_1111; no training."""
from __future__ import annotations
import argparse, hashlib, json, math, os, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
os.environ.setdefault("MPLCONFIGDIR", "/tmp/readout_separation_mpl")
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/readout_separation_1111"
BOOT=10000
METRICS=["U","U_acute","U_late","accuracy_end","ce_auc_mean","T90_first","T90_sustained"]
FACTORS=["family","direction","scale","body_memory","cf","body_gain","jacobian","bank"]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()

def private_seed(key):
    return int.from_bytes(hashlib.sha256(json.dumps(key,sort_keys=True,default=str).encode()).digest()[:8],"little")

def boot_values(x,key):
    x=np.asarray(x,float)
    if x.ndim!=1 or not np.isfinite(x).all():raise ValueError("Nonfinite/unpaired estimate")
    ix=np.random.default_rng(private_seed(key)).integers(0,len(x),(BOOT,len(x)))
    draws=x[ix].mean(1)
    lo,hi=np.quantile(draws,[.025,.975])
    return dict(estimate=float(x.mean()),ci_low=float(lo),ci_high=float(hi),
                verdict="positive" if lo>0 else "negative" if hi<0 else "unresolved",
                n_seed=len(x),n_positive=int((x>0).sum()))

def expected_grid(model,limit):
    each=100 if model=="mlp" else 25
    return sorted(s for s in {0,1,5,25,50,100,199,200,201,limit,*range(0,limit+1,each)} if s<=limit)

def validate(r):
    assert r["complete"] is True
    assert r["model"] in ("mlp","vit")
    assert r["limit"]==(4000 if r["model"]=="mlp" else 1500)
    c=r["curve"]
    s=np.asarray([q["step"] for q in c],int)
    a=np.asarray([q["acc"] for q in c],float)
    ce=np.asarray([q["ce"] for q in c],float)
    assert s.tolist()==expected_grid(r["model"],r["limit"]),(r.get("file"),s.tolist())
    assert np.isfinite(a).all() and ((a>=0)&(a<=1)).all()
    assert np.isfinite(ce).all() and (ce>=0).all()
    assert r["arm"]["family"] in ("physical","surrogate","jacobian")
    assert r["arm"]["body_memory"] in ("keep","clear")
    if r.get('pre_release') is not None:
        pre=r['pre_release']
        assert pre['step']==200 and np.isfinite(pre['acc']) and 0<=pre['acc']<=1
        assert np.isfinite(pre['ce']) and pre['ce']>=0
    return s,a,ce

def metric_row(r):
    s,a,ce=validate(r);horizon=int(r["limit"])
    threshold=float(np.float32(.9)) if r["model"]=="mlp" else .9
    hits=a>=threshold
    first=np.flatnonzero(hits)
    sustained=np.flatnonzero(np.logical_and.accumulate(hits[::-1])[::-1])
    early=s<=200;late=s>=200
    ae=a[early].copy();cee=ce[early].copy()
    if r.get("pre_release"):
        assert int(r["pre_release"]["step"])==200
        ae[-1]=r["pre_release"]["acc"];cee[-1]=r["pre_release"]["ce"]
    arm=r["arm"]
    row={k:r[k] for k in ("model","activation","seed","t_end")}
    row.update(arm_id=arm["id"],**{k:arm.get(k) for k in FACTORS})
    row.update(U=float(np.trapezoid(1-a,s)/horizon),
               U_acute=float(np.trapezoid(1-ae,s[early])/200),
               U_late=float(np.trapezoid(1-a[late],s[late])/(horizon-200)),
               ce_auc_mean=float(np.trapezoid(ce,s)/horizon),
               accuracy_end=float(a[-1]),ce_end=float(ce[-1]),
               T90_first=int(s[first[0]]) if len(first) else None,
               T90_sustained=int(s[sustained[0]]) if len(sustained) else None,
               T90_first_censored=not bool(len(first)),T90_sustained_censored=not bool(len(sustained)),
               sustained_observation_duration=horizon-int(s[sustained[0]]) if len(sustained) else None,
               horizon=horizon,raw_path=r["_path"],raw_sha256=r["_sha256"],
               initial_accuracy=float(a[0]),initial_ce=float(ce[0]),elapsed_s=r.get("elapsed_s"))
    for at,cr in [("start",r["curve"][0]),("pre200",r.get("pre_release") or next(q for q in r["curve"] if q["step"]==200)),
                  ("post200",next(q for q in r["curve"] if q["step"]==200)),("end",r["curve"][-1])]:
        for key,v in cr.items():
            if isinstance(v,(int,float)) and not isinstance(v,bool) and key!="step":
                row[f"{at}_{key}"]=v
        for i,layer in enumerate(cr.get("layers",[])):
            for key,v in layer.items():
                if isinstance(v,(int,float)) and not isinstance(v,bool):row[f"{at}_layer{i}_{key}"]=v
    return row

def load_records(path):
    out=[]
    for p in sorted(Path(path).glob("*.json")):
        r=json.loads(p.read_text())
        if not isinstance(r,dict) or not isinstance(r.get("arm"),dict) or "curve" not in r:continue
        r["_path"]=str(p);r["_sha256"]=sha(p);out.append(r)
    return out

def design_check(records):
    from analysis.readout_separation_1111.ops import arm_plan
    expected=set()
    for model,seeds in [("mlp",range(10)),("vit",range(10252010,10252015))]:
        arms=arm_plan(model)
        for seed in seeds:
            for a in arms:
                aid=a["id"] if isinstance(a,dict) else a.id
                expected.add((model,seed,aid))
    assert len(expected)==280,len(expected)
    actual=[(r["model"],r["seed"],r["arm"]["id"]) for r in records]
    assert len(set(actual))==len(actual),"duplicate arms"
    assert set(actual)==expected,dict(missing=sorted(expected-set(actual)),extra=sorted(set(actual)-expected))
    plans={model:{a['id']:a for a in arm_plan(model)} for model in ('mlp','vit')}
    for r in records:
        validate(r)
        assert r['arm']==plans[r['model']][r['arm']['id']], 'Arm metadata differs from frozen plan'
        assert r['activation']==('ELU' if r['model']=='mlp' else 'SL')
        assert r['t_end']==(10 if r['model']=='mlp' else 100)
    return dict(complete=True,expected=len(expected),actual=len(actual))

def arm_find(meta,**filters):
    x=meta
    for k,v in filters.items():
        if isinstance(v,float):x=x[np.isclose(x[k].astype(float),v)]
        else:x=x[x[k]==v]
    if len(x)!=1:raise ValueError((filters,x.to_dict("records")))
    return x.iloc[0]["arm_id"]

def contrast_plan(meta,model):
    plans=[]
    def add(family,name,coef,**ctx):
        assert len(coef)>0 and abs(sum(coef.values()))<1e-8
        plans.append(dict(kind=family,contrast=name,coefficients=coef,**ctx))
    def phys(d,c,m="keep"):return arm_find(meta,family="physical",direction=d,scale=float(c),body_memory=m)
    def sur(d,c,g):
        if c==1 and g==1:return phys(d,1)
        return arm_find(meta,family="surrogate",direction=d,cf=float(c),body_gain=float(g))
    for d in ("S","P","O"):
        for m in ("keep","clear") if d!="O" else ("keep",):
            add("scale",f"{d}_{m}",{phys(d,1,m):1,phys(d,.1,m):-1},direction=d,body_memory=m)
    for m in ("keep","clear"):
        for c in (1,.1):
            add("direction",f"S-P_c{c}_{m}",{phys("S",c,m):1,phys("P",c,m):-1},scale=c,body_memory=m,target_direction="P")
        add("interaction",f"scale_by_P_{m}",{phys("S",1,m):1,phys("S",.1,m):-1,phys("P",1,m):-1,phys("P",.1,m):1},body_memory=m)
    for c in (1,.1):
        add("direction",f"S-O_c{c}_keep",{phys("S",c):1,phys("O",c):-1},scale=c,body_memory="keep",target_direction="O")
    add("interaction","scale_by_O_keep",{phys("S",1):1,phys("S",.1):-1,phys("O",1):-1,phys("O",.1):1})
    for d in ("S","P"):
        for c in (1,.1):
            add("memory",f"{d}_c{c}",{phys(d,c,"keep"):1,phys(d,c,"clear"):-1},direction=d,scale=c)
        add("interaction",f"scale_by_memory_{d}",{phys(d,1,"keep"):1,phys(d,.1,"keep"):-1,phys(d,1,"clear"):-1,phys(d,.1,"clear"):1},direction=d)
    for c in (1,.1):
        add("interaction",f"direction_by_memory_c{c}",{phys("S",c,"keep"):1,phys("S",c,"clear"):-1,phys("P",c,"keep"):-1,phys("P",c,"clear"):1},scale=c)
    for d in ("S","P"):
        for g in (1,.1):add("forward",f"{d}_gain{g}",{sur(d,1,g):1,sur(d,.1,g):-1},direction=d,body_gain=g)
        for c in (1,.1):add("body_gain",f"{d}_cf{c}",{sur(d,c,1):1,sur(d,c,.1):-1},direction=d,cf=c)
        add("interaction",f"forward_by_gain_{d}",{sur(d,1,1):1,sur(d,.1,1):-1,sur(d,1,.1):-1,sur(d,.1,.1):1},direction=d)
        add("parameterization",f"physical01_minus_TG_{d}",{phys(d,.1):1,sur(d,.1,.1):-1},direction=d)
    for jac in sorted(meta[meta.family=="jacobian"].jacobian.unique()):
        def j(c):return arm_find(meta,family="jacobian",direction="S",scale=float(c),jacobian=jac)
        for c in (1,.1):add("transmission",f"{jac}_c{c}",{j(c):1,phys("S",c):-1},scale=c,jacobian=jac)
        add("interaction",f"scale_by_{jac}",{j(1):1,j(.1):-1,phys("S",1):-1,phys("S",.1):1},jacobian=jac)
    return plans

def contrasts(df):
    rows=[]
    for model,g in df.groupby("model",sort=True):
        meta=g.drop_duplicates("arm_id")[["arm_id",*FACTORS]]
        seeds=sorted(g.seed.unique())
        assert len(seeds)==(10 if model=="mlp" else 5)
        plans=contrast_plan(meta,model)
        for metric in METRICS:
            mat=g.pivot(index="seed",columns="arm_id",values=metric).reindex(seeds)
            for p in plans:
                coef=p["coefficients"];raw=mat[list(coef)].to_numpy(float)
                complete=np.isfinite(raw).all(1)
                row=dict(model=model,metric=metric,**{k:v for k,v in p.items() if k!="coefficients"},
                         coefficients=json.dumps(coef,sort_keys=True),n_complete_pairs=int(complete.sum()),
                         estimable=bool(complete.all()),estimate=None,ci_low=None,ci_high=None,verdict="censored")
                if complete.all():
                    values=raw@np.asarray(list(coef.values()),float)
                    row.update(boot_values(values,["paired_seeds",model,seeds]))
                    row['practical_positive']=bool(row['ci_low']>.01) if metric in ('U','U_acute','U_late') else None
                rows.append(row)
    return pd.DataFrame(rows)

def table(df,columns):
    vals=[]
    for _,r in df.iterrows():
        vals.append("| "+" | ".join(str(r.get(k,"")) for k in columns)+" |")
    return "| "+" | ".join(columns)+" |\n| "+" | ".join(["---"]*len(columns))+" |\n"+"\n".join(vals)

def report_summary(df,c,validation,path):
    out=["# Readout separation: complete paired contrasts",
         "Primary metric U is the period-average fixed-training accuracy deficit. Positive baseline-minus-intervention U means the intervention improves average accuracy. Bootstrap paired seeds,10000 replicates,descriptive unadjusted95% intervals. No additive mediation shares.",
         f"Validated{validation['actual']} curves: ELU10seeds,ViT5seeds. Physicalscale/head direction, body-memory, initial200update surrogate policies, and Jacobian policies are distinct interventions.",
         "Transmission contrasts use altered-Jacobian minus native, so positive U is harm. Parameterization uses physical0.1 minus TG, so positive U favors TG. Interaction signs follow the explicit coefficients in contrasts.csv.",
         "U_acute/U_late are secondary temporal partitions;200pre is used for the acute partition,200post for late. They are not causal mediators.",
         "## Primary U contrasts"]
    for model in ("mlp","vit"):
        out.append(f"### {model}")
        g=c[(c.model==model)&(c.metric=="U")].copy()
        g["effect_pp_95ci"]=g.apply(lambda r:f"{100*r.estimate:+.3f} [{100*r.ci_low:+.3f},{100*r.ci_high:+.3f}]" if r.estimable else "not estimable",axis=1)
        out.append(table(g,["kind","contrast","effect_pp_95ci","verdict","n_seed"]))
    out.append("## Arm endpoints and censoring")
    for (model,arm),g in df.groupby(["model","arm_id"]):
        out.append(f"- {model} {arm}: U={g.U.mean():.6f}; end accuracy={g.accuracy_end.mean():.6f}; T90 reached={(~g.T90_first_censored).sum()}/{len(g)}; sustained={(~g.T90_sustained_censored).sum()}/{len(g)}.")
    path.write_text("\n\n".join(out)+"\n")

def plots(df,c,records,dest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    colors={"S":"#275C9D","P":"#D67C20","O":"#2D9571"}
    fig,axs=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    for ax,model in zip(axs,("mlp","vit")):
        for d in ("S","P","O"):
            for scale in (1.,.1):
                items=[r for r in records if r["model"]==model and r["arm"]["family"]=="physical" and r["arm"]["direction"]==d and r["arm"]["scale"]==scale and r["arm"]["body_memory"]=="keep"]
                steps=np.array([x["step"] for x in items[0]["curve"]])
                curves=np.array([[x["acc"] for x in r["curve"]] for r in items])
                ax.plot(steps,curves.mean(0)*100,color=colors[d],ls="-" if scale==1 else "--",label=f"{d}, scale {scale:g}")
        ax.set(title="ELU, task10" if model=="mlp" else "ViT SL, task100",xlabel="Next-task updates",ylabel="Training accuracy (%)",ylim=(0,102))
        ax.axhline(90,color=".6",lw=.7,ls=":")
        ax.legend(fontsize=8,ncol=2)
    for ext in ("png","pdf"):fig.savefig(dest/f"physical_curves.{ext}",dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    for ax,model in zip(axs,("mlp","vit")):
        g=c[(c.model==model)&(c.metric=="U")&(c.kind=="scale")].copy()
        y=np.arange(len(g))
        est=g.estimate.to_numpy()*100;lo=g.ci_low.to_numpy()*100;hi=g.ci_high.to_numpy()*100
        ax.errorbar(est,y,xerr=np.vstack([est-lo,hi-est]),fmt="o",capsize=3,color="#275C9D")
        ax.set(yticks=y,yticklabels=g.contrast,title="ELU" if model=="mlp" else "ViT SL",xlabel="Benefit of scale1→0.1 (pp,95% CI)")
        ax.axvline(0,color=".5",lw=.8);ax.invert_yaxis()
    for ext in ("png","pdf"):fig.savefig(dest/f"scale_contrasts.{ext}",dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(3,2,figsize=(12,11),constrained_layout=True)
    for col,model in enumerate(('mlp','vit')):
        choices=[('memory',c[(c.model==model)&(c.metric=='U')&(c.kind=='memory')], 'Clearing body Adam'),
                 ('policy',c[(c.model==model)&(c.metric=='U')&(((c.kind=='forward')&(c.body_gain==1))|((c.kind=='body_gain')&(c.cf==1)))], 'Forward and body gain (first 200)'),
                 ('transmission',c[(c.model==model)&(c.metric=='U')&(c.kind=='transmission')], 'Jacobian change (first 200)')]
        for row,(_,g,title) in enumerate(choices):
            ax=axs[row,col];y=np.arange(len(g))
            est=g.estimate.to_numpy()*100;lo=g.ci_low.to_numpy()*100;hi=g.ci_high.to_numpy()*100
            labels=g.contrast.str.replace('ln_remove_radial_mean','LN radial+mean',regex=False).str.replace('ln_remove_radial','LN radial',regex=False).str.replace('elu_unity','ELU derivative=1',regex=False)
            if row==1:
                labels=g.apply(lambda r:r.direction+(': forward scale' if r.kind=='forward' else ': body gain'),axis=1)
            ax.errorbar(est,y,xerr=np.vstack([np.maximum(est-lo,0),np.maximum(hi-est,0)]),fmt='o',capsize=3,color='#275C9D' if model=='mlp' else '#B55E24')
            ax.set(yticks=y,yticklabels=labels,title=('ELU: ' if model=='mlp' else 'ViT SL: ')+title,xlabel=('Harm' if row==2 else 'Benefit')+' (U difference, pp,95% CI)')
            ax.axvline(0,color='.5',lw=.8);ax.invert_yaxis()
    for ext in ('png','pdf'):fig.savefig(dest/f'mechanism_contrasts.{ext}',dpi=180)
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--raw",type=Path,default=OUT/"raw");ap.add_argument("--out",type=Path,default=OUT)
    ap.add_argument("--no-plots",action="store_true");a=ap.parse_args()
    records=load_records(a.raw);v=design_check(records)
    a.out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame([metric_row(r) for r in records]).sort_values(["model","seed","arm_id"])
    c=contrasts(df);df.to_csv(a.out/"seed_metrics.csv",index=False);c.to_csv(a.out/"contrasts.csv",index=False)
    report_summary(df,c,v,a.out/"summary.md")
    if not a.no_plots:plots(df,c,records,a.out)
    checks={**v,"metric_code_sha256":sha(__file__),"git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "input_files":[{"path":r["_path"],"sha256":r["_sha256"]} for r in records],
            "registered_spec_sha256":sha(ROOT/"specs/spec_readout_separation_1111.md")}
    (a.out/"summary_validation.json").write_text(json.dumps(checks,indent=2))
    print(json.dumps({"curves":len(df),"contrasts":len(c),"validated":True}))

if __name__=="__main__":main()

"""Frozen registered inference and report generation for fb_width_seat_0915."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ARMS=["LRoff0_1216","LRwf21_1216","FB21LRoff0_1216","FB21LRwf21_1216","LRwi21_1216","FB21LRwi21_1216"]
FIXED=["LRwf21_1216","FB21LRwf21_1216"]

def boot(x, rng, n=2000):
    x=np.asarray(x,float)
    if len(x)==0:return float("nan"),[float("nan"),float("nan")]
    est=float(x.mean()); b=np.mean(x[rng.integers(0,len(x),(n,len(x)))],axis=1)
    return est,[float(v) for v in np.quantile(b,[.025,.975])]
def win(z,key,a,b,reverse=False,alive=False):
    step=z["step"]; take=np.isin(step,np.arange(a,b+1)*10000)
    if int(take.sum()) != b-a+1: raise ValueError(f"{key}: expected {b-a+1} task ends, got {take.sum()}")
    x=z[key][take].astype(float)
    if alive:
        den=z["layer1_denom"][take]; x=np.where(den>.25,x,np.nan)
    return float(np.nanmedian(np.nanmean(x,axis=0)) if reverse else np.nanmean(np.nanmedian(x,axis=1)))
def load(out,arm,s): return np.load(Path(out)/"logs"/f"{arm}_seed{s}.npz")
def q1_label(ci,stops=True):
    if not stops:return "NOT_DETERMINED_MOVING"
    if ci[0]>=-.15 and ci[1]<=.15:return "SEAT_SAME"
    if ci[0]>0:return "NOISE_LIFTS_SEAT"
    if ci[1]<0:return "NOISE_SINKS_SEAT"
    return "NOT_DETERMINED_WIDE"
def q2_label(fixed,free,m2=True):
    if not m2:return "NOT_DETERMINED","NOT_DETERMINED_NO_GROWTH"
    if all(c[0]>=-.1 and c[1]<=.1 for c in fixed) and all(c[1]<-.1 for c in free):return "STOPS_WITHOUT_GROWTH",""
    if any(c[1]<-.1 for c in fixed):return "KEEPS_SINKING",""
    return "NOT_DETERMINED",""
def compute(out):
    out=Path(out)
    check_path=out/"checks.json"
    checks=json.loads(check_path.read_text()) if check_path.exists() else {"pass":False,"reason":"checks.json missing"}
    missing=[f"{a}_seed{s}" for a in ARMS for s in range(10) if not (out/"logs"/f"{a}_seed{s}.npz").exists()]
    valid={a:[] for a in ARMS}; nonfinite=[]
    for a in ARMS:
        for s in range(10):
            p=out/"logs"/f"{a}_seed{s}.npz"
            if not p.exists():continue
            z=np.load(p); good=(not bool(z.get("numeric_divergence",False))) and all(np.isfinite(z[k]).all() for k in ("layer1_zbar","layer1_zmax","layer1_w_free"))
            (valid[a] if good else nonfinite).append(s if good else f"{a}_seed{s}")
    if any(10-len(valid[a])>3 for a in ARMS[:4]): return dict(Q1="NOT_DETERMINED",Q1_est=float('nan'),Q1_CI=[float('nan')]*2,Q2="NOT_DETERMINED",Q2_reason="NUMERIC_DIVERGENCE_GT3",M2=False,missing=missing,nonfinite=nonfinite,checks=checks,rows=[],raw={})
    rng=np.random.default_rng(20260915); rows=[]; vals={}
    for arm in ARMS:
        q2=[]; latez=[]; growth=[]; kapp=[]
        for s in valid[arm]:
            z=load(out,arm,s); q2.append(win(z,"layer1_zbar",451,500)-win(z,"layer1_zbar",251,300)); latez.append(win(z,"layer1_zmax",451,500))
            W=z["layer1_w_free"]; st=z["layer1_w_free_step"]; i=np.where(st==200000)[0][0]; late=np.isin(st,np.arange(451,501)*10000)
            norm=np.linalg.norm(W,axis=-1); growth.append(float(np.mean(np.median(norm[late],axis=1))/np.median(norm[i])))
            kap=np.sum(np.abs(W),axis=-1)/np.where(norm==0,np.nan,norm); kapp.append([float(np.nanmedian(kap[i])),float(np.nanmean(np.nanmedian(kap[late],axis=1)))])
        vals[arm]={"q2":q2,"zmax":latez,"growth":growth,"kappa":kapp,"q2_ci":boot(q2,rng),"growth_ci":boot(growth,rng)}
    # Q2 arm labels and overall; M2 failure retains both registered phrasings.
    armq={a:("STOPS" if vals[a]["q2_ci"][1][0]>=-.1 and vals[a]["q2_ci"][1][1]<=.1 else "SINKS" if vals[a]["q2_ci"][1][1]<-.1 else "UNSETTLED") for a in FIXED}
    m2=all(vals[a]["growth_ci"][0]>=1.5 for a in (ARMS[0],ARMS[2]))
    stops=all(armq[a]=="STOPS" for a in FIXED); free_sink=all(vals[a]["q2_ci"][1][1]<-.1 for a in (ARMS[0],ARMS[2]))
    q2,q2_reason=q2_label([vals[a]["q2_ci"][1] for a in FIXED],[vals[a]["q2_ci"][1] for a in (ARMS[0],ARMS[2])],m2)
    pair=sorted(set(valid[FIXED[0]])&set(valid[FIXED[1]])); d=np.asarray([vals[FIXED[0]]["zmax"][valid[FIXED[0]].index(s)]-vals[FIXED[1]]["zmax"][valid[FIXED[1]].index(s)] for s in pair]); q1est,q1ci=boot(d,rng)
    q1=q1_label(q1ci,stops)
    for a in ARMS:
        rows.append(dict(arm=a,q2_est=vals[a]["q2_ci"][0],q2_lo=vals[a]["q2_ci"][1][0],q2_hi=vals[a]["q2_ci"][1][1],growth=vals[a]["growth_ci"][0],growth_lo=vals[a]["growth_ci"][1][0],growth_hi=vals[a]["growth_ci"][1][1]))
    if not checks.get("pass"):
        q1=q2="NOT_DETERMINED"; q2_reason="G1_G2_OR_M1_FAILED"
    from analysis.fb_width_seat_0915_report import report_from_directory
    report=report_from_directory(out)
    return dict(Q1=q1,Q1_est=q1est,Q1_CI=q1ci,Q2=q2,Q2_reason=q2_reason,M2=m2,M2_per_seed={a:[x>=1.5 for x in vals[a]["growth"]] for a in (ARMS[0],ARMS[2])},M2_rule="mean of available finite preregistered seed-level growth ratios >= 1.5",arm_Q2=armq,rows=rows,raw=vals,report_only=report,checks=checks,missing=missing,nonfinite=nonfinite)
def synthetic_selftest():
    rng=np.random.default_rng(20260915); same=boot(np.zeros(10),rng)[1]; lift=boot(np.ones(10),rng)[1]
    labels={"same":q1_label(same),"lift":q1_label(lift),"sink":q1_label([-1.,-1.]),"moving":q1_label(same,False),"q2stop":q2_label([[0,0],[0,0]],[[-1,-1],[-1,-1]],True)[0],"q2nogrowth":q2_label([[0,0],[0,0]],[[-1,-1],[-1,-1]],False)[1]}
    return {"pass":same==[0.,0.] and lift[0]>0 and all(v!="bad" for v in labels.values()),"same":same,"lift":lift,"labels":labels,"seed":20260915,"n":2000}
def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path);p.add_argument("--selftest",action="store_true");a=p.parse_args()
    if a.selftest: print(json.dumps(synthetic_selftest(),indent=2));return
    r=compute(a.out); (a.out/"verdict.json").write_text(json.dumps(r,indent=2))
    with (a.out/"verdict.csv").open("w",newline="") as f:
        fields=(list(r["rows"][0]) if r["rows"] else ["arm"]); w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(r["rows"])
    (a.out/"summary.md").write_text(f"# fb_width_seat_0915\n\nQ2: **{r['Q2']}** ({r['Q2_reason']})\n\nQ1: **{r['Q1']}**, Δzmax={r['Q1_est']:.4g}, CI={r['Q1_CI']}\n")
    print(json.dumps({k:r[k] for k in ("Q1","Q1_est","Q1_CI","Q2","Q2_reason","M2")},indent=2))
if __name__=="__main__":main()

"""Frozen registered inference and report generation for fb_width_seat_0915."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ARMS=["LRoff0_1216","LRwf21_1216","FB21LRoff0_1216","FB21LRwf21_1216","LRwi21_1216","FB21LRwi21_1216"]
FIXED=["LRwf21_1216","FB21LRwf21_1216"]

def boot(x, rng, n=2000):
    x=np.asarray(x,float); est=float(x.mean()); b=np.mean(x[rng.integers(0,len(x),(n,len(x)))],axis=1)
    return est,[float(v) for v in np.quantile(b,[.025,.975])]
def win(z,key,a,b):
    step=z["step"]; take=np.isin(step,np.arange(a,b+1)*10000)
    return float(np.mean(np.median(z[key][take],axis=1)))
def load(out,arm,s): return np.load(Path(out)/"logs"/f"{arm}_seed{s}.npz")
def compute(out):
    out=Path(out)
    check_path=out/"checks.json"
    checks=json.loads(check_path.read_text()) if check_path.exists() else {"pass":False,"reason":"checks.json missing"}
    missing=[f"{a}_seed{s}" for a in ARMS for s in range(10) if not (out/"logs"/f"{a}_seed{s}.npz").exists()]
    if missing: return dict(Q1="NOT_DETERMINED",Q2="NOT_DETERMINED",Q2_reason="MISSING_OR_DIVERGED",missing=missing,checks=checks,rows=[],raw={})
    rng=np.random.default_rng(20260915); rows=[]; vals={}
    for arm in ARMS:
        q2=[]; latez=[]; growth=[]; kapp=[]
        for s in range(10):
            z=load(out,arm,s); q2.append(win(z,"layer1_zbar",451,500)-win(z,"layer1_zbar",251,300)); latez.append(win(z,"layer1_zmax",451,500))
            W=z["layer1_w_free"]; st=z["layer1_w_free_step"]; i=np.where(st==200000)[0][0]; late=np.isin(st,np.arange(451,501)*10000)
            norm=np.linalg.norm(W,axis=-1); growth.append(float(np.mean(np.median(norm[late],axis=1))/np.median(norm[i])))
            kap=np.sum(np.abs(W),axis=-1)/np.where(norm==0,np.nan,norm); kapp.append([float(np.nanmedian(kap[i])),float(np.nanmean(np.nanmedian(kap[late],axis=1)))])
        vals[arm]={"q2":q2,"zmax":latez,"growth":growth,"kappa":kapp,"q2_ci":boot(q2,rng),"growth_ci":boot(growth,rng)}
    # Q2 arm labels and overall; M2 failure retains both registered phrasings.
    armq={a:("STOPS" if vals[a]["q2_ci"][1][0]>=-.1 and vals[a]["q2_ci"][1][1]<=.1 else "SINKS" if vals[a]["q2_ci"][1][1]<-.1 else "UNSETTLED") for a in FIXED}
    m2=all(vals[a]["growth_ci"][0]>=1.5 for a in (ARMS[0],ARMS[2]))
    stops=all(armq[a]=="STOPS" for a in FIXED); free_sink=all(vals[a]["q2_ci"][1][1]<-.1 for a in (ARMS[0],ARMS[2]))
    if not m2: q2="NOT_DETERMINED"; q2_reason="NOT_DETERMINED_NO_GROWTH"
    elif stops and free_sink: q2="STOPS_WITHOUT_GROWTH"; q2_reason=""
    elif any(armq[a]=="SINKS" for a in FIXED): q2="KEEPS_SINKING"; q2_reason=""
    else: q2="NOT_DETERMINED"; q2_reason=""
    d=np.asarray(vals[FIXED[0]]["zmax"])-np.asarray(vals[FIXED[1]]["zmax"]); q1est,q1ci=boot(d,rng)
    if not stops: q1="NOT_DETERMINED_MOVING"
    elif q1ci[0]>=-.15 and q1ci[1]<=.15: q1="SEAT_SAME"
    elif q1ci[0]>0:q1="NOISE_LIFTS_SEAT"
    elif q1ci[1]<0:q1="NOISE_SINKS_SEAT"
    else:q1="NOT_DETERMINED_WIDE"
    for a in ARMS:
        rows.append(dict(arm=a,q2_est=vals[a]["q2_ci"][0],q2_lo=vals[a]["q2_ci"][1][0],q2_hi=vals[a]["q2_ci"][1][1],growth=vals[a]["growth_ci"][0],growth_lo=vals[a]["growth_ci"][1][0],growth_hi=vals[a]["growth_ci"][1][1]))
    if not checks.get("pass"):
        q1=q2="NOT_DETERMINED"; q2_reason="G1_G2_OR_M1_FAILED"
    # Registered report-only summaries and sensitivity analyses.
    report={}
    for arm in ARMS:
        neg=[]; vabs=[]; wflip=[]; strict=[]; unfit=[]
        for s in range(10):
            z=load(out,arm,s); take=np.isin(z["step"],np.arange(451,501)*10000)
            neg.append(float(np.mean(z["layer1_zmax"][take]<0))); vabs.append(float(np.mean(np.abs(z["layer1_v_unit"][take])))); wflip.append(float(np.mean(z["layer1_w_flip_norm"][-50:])))
            strict.append(float(np.mean(z["layer1_strict_dead"][take]))); unfit.append(float(np.mean(z["unfit"][take])))
        report[arm]=dict(all_negative_fraction=boot(neg,rng),abs_v=boot(vabs,rng),w_flip_norm=boot(wflip,rng),strict_dead=boot(strict,rng),unfit=boot(unfit,rng),kappa_t20_to_late=vals[arm]["kappa"])
    return dict(Q1=q1,Q1_est=q1est,Q1_CI=q1ci,Q2=q2,Q2_reason=q2_reason,M2=m2,M2_per_seed={a:[x>=1.5 for x in vals[a]["growth"]] for a in (ARMS[0],ARMS[2])},M2_rule="mean of the 10 preregistered seed-level growth ratios >= 1.5",arm_Q2=armq,rows=rows,raw=vals,report_only=report,checks=checks)
def synthetic_selftest():
    rng=np.random.default_rng(20260915); same=boot(np.zeros(10),rng)[1]; lift=boot(np.ones(10),rng)[1]
    labels={"same":("SEAT_SAME" if same[0]>=-.15 and same[1]<=.15 else "bad"),"lift":("NOISE_LIFTS_SEAT" if lift[0]>0 else "bad"),"sink":("NOISE_SINKS_SEAT" if -lift[0]<0 and -lift[1]<0 else "bad")}
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

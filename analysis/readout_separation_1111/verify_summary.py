"""Independent numerical checks for scientific metrics (no checkpoint loading)."""
import copy,json
import numpy as np
import pandas as pd
from analysis.readout_separation_1111 import summarize as S
from analysis.readout_separation_1111.ops import arm_plan

def fake(model,seed,arm):
    horizon=4000 if model=="mlp" else 1500
    accuracy=.4+.2*(1-arm["scale"])+(.05 if arm["direction"]=="P" else .02 if arm["direction"]=="O" else 0)
    accuracy+=(.03 if arm["body_memory"]=="clear" else 0)
    accuracy+=.04*(1-arm["cf"])+.06*(1-arm["body_gain"])
    if arm["family"]=="jacobian":accuracy+=.01
    return dict(model=model,activation="ELU" if model=="mlp" else "SL",seed=seed,t_end=10 if model=="mlp" else 100,
                arm=arm,limit=horizon,complete=True,
                curve=[dict(step=s,acc=accuracy,ce=1.2) for s in S.expected_grid(model,horizon)],
                pre_release=None,_path="synthetic",_sha256="synthetic")

def main():
    records=[fake(model,seed,arm) for model,seeds in [("mlp",range(10)),("vit",range(10252010,10252015))]
             for seed in seeds for arm in arm_plan(model)]
    assert S.design_check(records)["actual"]==280
    df=pd.DataFrame([S.metric_row(r) for r in records])
    assert df.T90_first_censored.all()
    assert df.T90_first.isna().all(),"censor was replaced with horizon"
    c=S.contrasts(df)
    assert not c[c.metric=="T90_first"].estimable.any()
    for model in ("mlp","vit"):
        sel=c[(c.model==model)&(c.metric=="U")&(c.kind=="scale")]
        assert np.allclose(sel.estimate,.18),sel
        assert np.allclose(sel.ci_low,.18) and np.allclose(sel.ci_high,.18)
        mem=c[(c.model==model)&(c.metric=="U")&(c.kind=="memory")]
        assert np.allclose(mem.estimate,.03)
        d=c[(c.model==model)&(c.metric=="U")&(c.kind=="direction")]
        assert np.allclose(d[d.target_direction=="P"].estimate,.05)
        assert np.allclose(d[d.target_direction=="O"].estimate,.02)
        inter=c[(c.model==model)&(c.metric=="U")&(c.kind=="interaction")]
        assert np.allclose(inter.estimate,0,atol=1e-14)
    r=fake("mlp",0,arm_plan("mlp")[0])
    for q in r["curve"]:q["acc"]=.2+.7*q["step"]/4000
    m=S.metric_row(r)
    assert np.isclose(m["U"],.45)
    assert m["T90_first"]==4000 and m["T90_sustained"]==4000
    assert m["sustained_observation_duration"]==0
    # An early isolated90% crossing is distinct from a sustained crossing.
    r["curve"][1]["acc"]=.95
    m=S.metric_row(r)
    assert m["T90_first"]==1 and m["T90_sustained"]==4000
    # The pre-release marker changes only acute U, never full or late U.
    b=S.metric_row(r)
    r["pre_release"]={"step":200,"acc":.8,"ce":.3}
    a=S.metric_row(r)
    assert a["U"]==b["U"] and a["U_late"]==b["U_late"]
    assert a["U_acute"]!=b["U_acute"]
    bad=copy.deepcopy(r);bad["curve"].pop(1)
    try:S.validate(bad)
    except AssertionError:pass
    else:raise AssertionError("missing scheduled measurement accepted")
    bad=copy.deepcopy(r);bad["curve"][0]["acc"]=float("nan")
    try:S.validate(bad)
    except AssertionError:pass
    else:raise AssertionError("nonfinite accepted")
    bad=copy.deepcopy(r);bad['pre_release']['acc']=float('nan')
    try:S.metric_row(bad)
    except AssertionError:pass
    else:raise AssertionError('nonfinite pre-release accepted')
    try:S.design_check(records[:-1])
    except AssertionError:pass
    else:raise AssertionError("incomplete design accepted")
    out=dict(passed=True,synthetic_curves=280,contrasts=len(c),
             checks=["paired_direction_scale_memory_signs","zero_interactions","censor_not_cap",
                     "trapezoid_linear_exact","first_vs_sustained","pre_post_release_separation",
                     "missing_grid_rejected","nonfinite_rejected","nonfinite_pre_release_rejected","incomplete_design_rejected"])
    print(json.dumps(out))

if __name__=="__main__":main()

"""Aggregate frozen-state residual/pullback/transmission assays; seed is the unit."""
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.readout_separation_1111 import summarize as S
KEY=["model","label_channel","forward_bank","backward_bank","jacobian"]
LOCAL_METRICS=[
"residual_norm","g_feature_unreduced_mean_example_norm","g_feature_radial_sum",
"g_pre_norm","g_pre_cos_baseline","g_pre_radial_sum","g_trunk_norm","g_trunk_cos_baseline","g_trunk_cos_native",
"g_gamma_norm","g_beta_norm","adam_keep_trunk_norm","adam_clear_trunk_norm",
"adam_keep_clear_body_cosine","gd_push_mean","gd_push_abs_mean",
"adam_keep_mean_preact_shift","adam_clear_mean_preact_shift",
"normmatch_scale","nm_g_trunk_norm","nm_g_trunk_cos_native",
"nm_adam_keep_trunk_norm","nm_adam_clear_trunk_norm",
"nm_adam_keep_mean_preact_shift","nm_adam_clear_mean_preact_shift"]
LOCAL_METRICS += [f"{prefix}_{group}_{metric}" for prefix in ('adam_keep','adam_clear','nm_adam_keep','nm_adam_clear')
                  for group in ('body','trunk','gamma','beta') for metric in ('cos_native','cos_baseline')]
BANKS={"S1","S01","P1","P01","O1","O01"}

def read_all(directory):
    docs=[]
    for p in sorted(Path(directory).glob("probe_*.json")):
        d=json.loads(p.read_text());assert d["complete"]
        assert d["n_subset"]==128
        assert d['model'] in ('mlp','vit')
        assert d['activation']==('ELU' if d['model']=='mlp' else 'SL')
        assert d['t_end']==(10 if d['model']=='mlp' else 100)
        for row in d['records']:
            assert all(row[k]==d[k] for k in ('model','activation','seed','t_end','n_subset')), 'Probe row/header metadata mismatch'
        assert len(d['fullset'])==6 and {r['bank'] for r in d['fullset']}==BANKS
        assert set(d['real_adam_checks'])=={'keep','clear'}
        assert S.sha(p.parent/d["arrays_file"])==d["arrays_sha256"]
        modes={"native","elu_unity"} if d["model"]=="mlp" else {"native","ln_remove_radial","ln_remove_radial_mean"}
        expected={(label,f,b,j) for label in ("actual","uniform") for f in BANKS for b in BANKS for j in modes}
        actual=[tuple(r[k] for k in ("label_channel","forward_bank","backward_bank","jacobian")) for r in d["records"]]
        assert len(actual)==len(set(actual)) and set(actual)==expected
        assert d["checks"] and all(v["passed"] for v in d["checks"].values())
        assert all(v["passed"] for v in d["real_adam_checks"].values())
        if d["model"]=="vit":assert d["gauge_check"]["passed"]
        for f in BANKS:
            hashes={r["probability_sha256"] for r in d["records"] if r["forward_bank"]==f}
            assert len(hashes)==1,"forward probability not pointwise matched"
        d["_path"]=str(p);d["_sha256"]=S.sha(p);docs.append(d)
    expected={("mlp",seed) for seed in range(10)}|{("vit",seed) for seed in range(10252010,10252015)}
    actual=[(d["model"],d["seed"]) for d in docs]
    assert len(actual)==len(set(actual)) and set(actual)==expected,(expected-set(actual))
    return docs

def summaries(df,keys,metrics):
    rows=[]
    for key,g in df.groupby(keys,dropna=False,sort=True):
        meta=dict(zip(keys,key if isinstance(key,tuple) else (key,)))
        seeds=sorted(g.seed.unique());assert len(seeds)==len(g)
        assert len(seeds)==(10 if meta["model"]=="mlp" else 5)
        g=g.set_index("seed").reindex(seeds)
        for metric in metrics:
            if metric not in g:continue
            vals=g[metric].to_numpy(float)
            valid=np.isfinite(vals)
            r=dict(**meta,metric=metric,n_seed=len(seeds),n_finite=int(valid.sum()),estimable=bool(valid.all()))
            if valid.all():r.update(S.boot_values(vals,["probe_paired",meta["model"],seeds]))
            else:r.update(estimate=None,ci_low=None,ci_high=None,verdict="not_estimable")
            rows.append(r)
    return pd.DataFrame(rows)

def local_contrasts(df):
    rows=[]
    metrics=["g_feature_unreduced_mean_example_norm","g_pre_norm","g_trunk_norm","g_pre_radial_sum",
             "gd_push_mean","adam_keep_trunk_norm","adam_clear_trunk_norm",
             "adam_keep_mean_preact_shift","adam_clear_mean_preact_shift"]
    for model,g in df.groupby("model"):
        seeds=sorted(g.seed.unique())
        for label in ("actual","uniform"):
            base=g[g.label_channel==label]
            def select(f,b,j="native"):
                x=base[(base.forward_bank==f)&(base.backward_bank==b)&(base.jacobian==j)]
                assert set(x.seed)==set(seeds) and len(x)==len(seeds)
                return x.set_index("seed").reindex(seeds)
            plans=[]
            for d in ("S","P","O"):
                plans.extend([(f"{d}_forward_scale",select(d+"1",d+"1"),select(d+"01",d+"1")),
                              (f"{d}_backward_scale",select(d+"1",d+"1"),select(d+"1",d+"01")),
                              (f"{d}_physical_scale",select(d+"1",d+"1"),select(d+"01",d+"01"))])
            for d in ("P","O"):
                plans.extend([(f"{d}_forward_geometry",select("S1","S1"),select(d+"1","S1")),
                              (f"{d}_backward_geometry",select("S1","S1"),select("S1",d+"1")),
                              (f"{d}_joint_geometry",select("S1","S1"),select(d+"1",d+"1"))])
            modes=["elu_unity"] if model=="mlp" else ["ln_remove_radial","ln_remove_radial_mean"]
            for j in modes:
                for b in ("S1","S01","P1","P01"):
                    # altered minus native: sign is not "benefit" for all metrics.
                    plans.append((f"{j}_{b}_minus_native",select(b,b,j),select(b,b)))
            for name,a,b in plans:
                for metric in metrics:
                    if metric not in a or metric not in b:continue
                    vals=a[metric].to_numpy(float)-b[metric].to_numpy(float)
                    ok=np.isfinite(vals).all()
                    r=dict(model=model,label_channel=label,contrast=name,metric=metric,estimable=bool(ok),
                           definition="first named condition minus second; norm difference is not vector decomposition")
                    if ok:r.update(S.boot_values(vals,["probe_paired",model,seeds]))
                    else:r.update(estimate=None,ci_low=None,ci_high=None,verdict="not_estimable")
                    rows.append(r)
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--probes",type=Path,default=S.OUT/"probes");ap.add_argument("--out",type=Path,default=S.OUT)
    a=ap.parse_args();docs=read_all(a.probes);a.out.mkdir(parents=True,exist_ok=True)
    records=[]
    full=[]
    for d in docs:
        records.extend(d["records"])
        full.extend([{**r,"model":d["model"],"activation":d["activation"],"seed":d["seed"],"t_end":d["t_end"]} for r in d["fullset"]])
    df=pd.DataFrame(records);banks=pd.DataFrame(full)
    df.to_csv(a.out/"probe_seed_records.csv",index=False);banks.to_csv(a.out/"probe_fullset_seed_metrics.csv",index=False)
    agg=summaries(df,KEY,LOCAL_METRICS);agg.to_csv(a.out/"probe_estimates.csv",index=False)
    contrast=local_contrasts(df);contrast.to_csv(a.out/"probe_contrasts.csv",index=False)
    bankmetrics=[k for k in banks.select_dtypes(include="number").columns if k not in ("seed","t_end","scale","n")]
    summaries(banks,["model","bank"],bankmetrics).to_csv(a.out/"probe_fullset_estimates.csv",index=False)
    val=dict(complete=True,source_states=len(docs),local_rows=len(df),expected_local_rows=2520,
             source_sha256={d["_path"]:d["_sha256"] for d in docs},checks=sum(len(d["checks"]) for d in docs),
             real_adam_source_checks=sum(len(d["real_adam_checks"]) for d in docs),
             gauge_checks=sum(d.get("gauge_check") is not None for d in docs),
             analysis_sha256=S.sha(__file__),spec_sha256=S.sha(S.ROOT/"specs/spec_readout_separation_1111.md"))
    assert len(df)==2520 and len(banks)==90
    (a.out/"probe_validation.json").write_text(json.dumps(val,indent=2))
    lines=["# Frozen-state separation",
           "All backward-map and transmission comparisons reuse the whole forward probability vector for each fixed forward bank. Actual next labels and uniform-label expectation are distinct channels. Parameter gradients come from128fixed examples; MLP position shifts concern the full1200-example recipient input mean.",
           "Intervals are paired-seed descriptive95% bootstrap intervals. Norm differences do not form an additive decomposition of delay.",
           "## Native physical banks, actual new labels"]
    subset=agg[(agg.label_channel=="actual")&(agg.jacobian=="native")&(agg.forward_bank==agg.backward_bank)&
               agg.metric.isin(["g_feature_unreduced_mean_example_norm","g_trunk_norm","gd_push_mean","adam_keep_mean_preact_shift","adam_keep_trunk_norm"])]
    q=subset.copy();q["value_95ci"]=q.apply(lambda r:f"{r.estimate:.6g} [{r.ci_low:.6g},{r.ci_high:.6g}]" if r.estimable else "not estimable",axis=1)
    lines.append(S.table(q,["model","forward_bank","metric","value_95ci"]))
    lines.append("Full pointwise-matched results: probe_seed_records.csv/probe_estimates.csv/probe_contrasts.csv. Geometry and effective LN-affine scales: probe_fullset_seed_metrics.csv. Gauge and virtual Adam checks: probe_validation.json and raw probe JSON.")
    (a.out/"probe_summary.md").write_text("\n\n".join(lines)+"\n")
    print(json.dumps({"source_states":len(docs),"local_rows":len(df),"contrasts":len(contrast),"validated":True}))

if __name__=="__main__":main()

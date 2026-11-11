#!/usr/bin/env python3
"""Compact tables from planned diagnostics; the 200->500 window is post-hoc."""
import argparse
from pathlib import Path
import pandas as pd
from summarize import md

def main(out):
    out=Path(out);c=pd.read_csv(out/'counterfactual.csv')
    f=c[(c.arm=='V01')&(c.step==4000)&(c.group=='current_top')&c.p_scale.isin([1.,10.])&c.V_scale.isin([1.,10.])]
    f.to_csv(out/'frozen_cross_end.csv',index=False)
    s=pd.read_csv(out/'late_sources.csv');s=s[s.arm.isin(['V1','V01'])&(s.group=='upper')]
    s.to_csv(out/'natural_upper_sources.csv',index=False)
    fs=pd.read_csv(out/'followup_state.csv');w=fs[fs.arm=='C01'].set_index(['task','seed','step'])
    rows=[]
    for t in (10,25):
        for seed in (0,1,2):
            a=w.loc[(t,seed,200)];b=w.loc[(t,seed,500)]
            rows.append(dict(tier='posthoc',task=t,seed=seed,window='200->500',body_delta=b.body-a.body,h_delta=b.h_median-a.h_median))
    window=pd.DataFrame(rows);window.to_csv(out/'simultaneous_window_posthoc.csv',index=False)
    text=['# Mechanism tables','',
          '## Planned frozen-state crossed residual/readout assay','',
          'V01 endpoint, current top5% per unit. Scales relative to the current small V; forward scaling retains per-class mean logits. Same feature state and same receiver mask in each crossing. D<0 requests increasing activity.','',
          md(f[['task','seed','p_scale','V_scale','D','DU','DY','up_magnitude','sign_changed']].round(8)),'',
          'Three-seed means:','',md(f.groupby(['task','p_scale','V_scale'])[['D','DU','DY','up_magnitude','sign_changed']].mean().round(8).reset_index()),'',
          '## Planned natural-trajectory next-step source accounting','',
          'Fixed original upper5%; each seed averages probes200/500/1000/2000. Actual observed Adam denominator. A source is not a mediator fraction.','',
          md(s.pivot(index=['task','seed','arm'],columns='source',values='dz').round(8).reset_index()),'',
          '## Post-hoc simultaneous-motion window','',
          'Exact-gradient constrained C01, interval200->500 selected after plotting. Mean body and median unit maximum move in opposite directions; individual-unit co-occurrence over registered windows is in the primary tables.','',md(window.round(8))]
    (out/'mechanism_summary.md').write_text('\n'.join(text)+'\n')
    print(f.groupby(['task','p_scale','V_scale']).D.agg(['mean','min','max']).to_string());print(window.to_string(index=False))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',required=True);v=a.parse_args();main(v.out)

#!/usr/bin/env python3
"""Descriptive views of already registered fields; no additional training or tests."""
import sys
from pathlib import Path
import pandas as pd

def pick(d,**kw):
    for k,v in kw.items():d=d[d[k].isin(v) if isinstance(v,list) else d[k].eq(v)]
    return d

def main(folder):
    p=Path(folder)
    st,dm,sp,ins,src,resp=[pd.read_csv(p/f'{k}.csv') for k in ['state','demand','step','instant','source','response']]
    keys=['act','seed','task','arm','phase','step']
    a=pick(ins,task=10,labels='new',group='all')
    cols=['act','seed','arm','increase_mag','decrease_mag','increase_fraction','Drms','direct_term_rms','residual_term_rms','sign_changed_fraction','expected_mean_dz_sgd']
    a[cols].to_csv(p/'instant_scalar_t10.csv',index=False)
    a=pick(dm,act='ELU',task=10,phase='fit',arm=['S','Q','QV'],step=[0,500,4000],group=['initial_open','initial_deep'])
    cols=keys+['group','n','Dmean','increase_fraction','increase_mag','decrease_mag','da_since_start','dz_since_start','open_fraction']
    a[cols].to_csv(p/'fixed_groups_elu_t10.csv',index=False)
    a=pick(src,task=10,phase='fit',step=500,arm=['S','Q'],receiver='zero_up')
    cols=keys+['source','n','dz_mean','dz_rms','bias_dz_mean']
    a[cols].to_csv(p/'zero_up_sources_t10.csv',index=False)
    a=pick(sp,act='ELU',task=10,phase='fit',step=[0,500,4000],arm=['S','Q','QH'],optimizer='retained')
    cols=keys+['V_raw_rms','V_update_rms','V_step_cosine','V_grad_step_cosine','V_m_rms','V_v_rms','V_descent','W_descent']
    a[cols].to_csv(p/'elu_head_steps_t10.csv',index=False)
    a=pick(st,phase='reload',step=[0,200]).pivot(index=['act','seed','task','arm'],columns='step',values=['zmean','top_median','Vnorm','units_per_input','ce'])
    b=pd.DataFrame(index=a.index)
    for k in ['zmean','top_median','Vnorm','units_per_input','ce']:b[k+'_change']=a[k][200]-a[k][0]
    b.reset_index().to_csv(p/'reload_first200.csv',index=False)
    print('Five descriptive tables saved; no new tests or training.')

if __name__=='__main__':main(sys.argv[1])

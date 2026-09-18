"""Descriptive heterogeneity check, added after primary outcomes were inspected.
No new primary claims or thresholds: report unit-level increases despite a pooled decrease.
"""
import csv,json
import numpy as np
import report as r

def main():
 rows=[]
 for arm in r.ARMS:
  for age in r.AGES:
   g=np.load(r.RAW/f'{arm}_{age}_geometry.npz');d=np.load(r.RAW/f'{arm}_{age}_float64_B.npz')
   for coord in ['whole','centered']:
    q=g[coord+'_q_A'];bq=g[coord+'_B_q'];ea=np.mean(q*q,0);eb=np.mean(bq*bq,1)
    ma=q.mean(0);mb=bq.mean(1)
    bigger=eb>ea[None]
    measures={
      'unit_flip_fraction_B_RMS_above_A':bigger.mean((0,2)),
      'B_energy_share_units_with_RMS_increase':(eb*bigger).sum((0,2))/eb.sum((0,2)),
      'mean_projection_RMS_B_over_A':np.sqrt((mb*mb).mean((0,2))/(ma*ma).mean(1)),
      'mean_projection_change_RMS':np.sqrt(((mb-ma[None])**2).mean((0,2))),
      'B_minus_A_sign_positive_fraction':(bq-q[None]>0).mean((0,1,3))}
    for branch,k in [('B',0),('control',10)]:
      w0=d['W_Bstart'][k:k+10];w1=d['W'][-1,k:k+10]
      if coord=='centered':w0=w0-w0.mean(-1,keepdims=True);w1=w1-w1.mean(-1,keepdims=True)
      measures[branch+'_norm2_change']=(w1*w1-w0*w0).sum((1,2))
      measures[branch+'_positive_source_event_fraction']=d['positive_source_count'][-1,k:k+10].mean(1)/10000
    for name,value in measures.items():rows.append(dict(arm=arm,age=age,coordinate=coord,metric=name,**r.stats(value)))
 r.write('posthoc_heterogeneity.csv',rows)
 print(json.dumps(rows),flush=True)
if __name__=='__main__':main()

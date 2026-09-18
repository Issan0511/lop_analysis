"""Registered secondary follow-up: actual weight in A's input nullspace."""
import json,subprocess
import numpy as np
from scipy.stats import t as student
import report as r

def followup_stats(x):
 d=r.stats(x)
 if not np.isfinite(x).all():return d
 h=student.ppf(1-.05/24,9)*np.std(x,ddof=1)/np.sqrt(10);m=np.mean(x)
 return dict(mean=float(m),lo=float(m-h),hi=float(m+h),label='POSITIVE' if m-h>0 else 'NEGATIVE' if m+h<0 else 'UNRESOLVED',n=10)

def main():
 rows=[];curves=[];adjusted=[];seeds=[]
 def add(arm,age,name,value):
  rows.append(dict(arm=arm,age=age,metric=name,**r.stats(value)))
  for i,v in enumerate(value):seeds.append(dict(arm=arm,age=age,metric=name,seed=i,value=float(v)))
 for arm in r.ARMS:
  for age in r.AGES:
   g=np.load(r.RAW/f'{arm}_{age}_geometry.npz');d=np.load(r.RAW/f'{arm}_{age}_float64_B.npz')
   P=[]
   for i in range(10):
    _,s,v=np.linalg.svd(g['x_A'][:,i],full_matrices=False);rank=np.sum(s>1e-10*s[0]);P.append(v[:rank].T@v[:rank])
   P=np.stack(P);project=lambda w:np.einsum('rhi,rij->rhj',w,P)
   w0=g['w0'];w1=g['w1'];n=w1-project(w1);n0=w0-project(w0);p=w1-n
   r.check('A_retained_null',n,n0);r.check('not_w0_orthogonal',(n*w0).sum(-1),(n*n).sum(-1))
   qa=np.einsum('rhd,srd->srh',n,g['x_A']);r.check('A_invisibility',qa,0.)
   bq=np.einsum('rhd,fsrd->fsrh',n,g['whole_B_x']);dq=g['whole_B_x']-g['x_A'][None]
   jump=np.einsum('rhd,fsrd->fsrh',w1,dq);pjump=np.einsum('rhd,fsrd->fsrh',p,dq)
   r.check('jump_null_plus_span',jump,bq+pjump)
   r.check('null_B_field_constant_across_patterns',bq,bq.mean(1,keepdims=True))
   den=(n*n).sum((1,2));ja=(jump*jump).mean((0,1,3));nb=(bq*bq).mean((0,1,3));pb=(pjump*pjump).mean((0,1,3));cross=2*(bq*pjump).mean((0,1,3))
   r.check('jump_energy_decomposition',ja,nb+pb+cross)
   for k,v in dict(null_weight_energy_fraction=den/(w1*w1).sum((1,2)),null_B_field_RMS=np.sqrt(nb),total_input_jump_RMS=np.sqrt(ja),null_field_energy_over_jump_energy=nb/ja,span_jump_energy_over_jump_energy=pb/ja,cross_over_jump_energy=cross/ja,null_field_signed_mean=bq.mean((0,1,3)),null_field_positive_fraction=(bq>0).mean((0,1,3)),null_field_positive_energy_fraction=(bq*bq*(bq>0)).sum((0,1,3))/(bq*bq).sum((0,1,3))).items():add(arm,age,k,v)
   q=np.einsum('rhd,srd->srh',n,d['x_B']);z=d['z_Bstart'];masks={'qpos':q>0,'qneg':q<0,'qpos_zpos':(q>0)&(z>0),'qpos_zneg':(q>0)&(z<=0),'qneg_zpos':(q<0)&(z>0),'qneg_zneg':(q<0)&(z<=0)}
   for it,t in enumerate(d['time']):
    ends={}
    for branch,k in [('B',0),('control',10)]:
     dw=d['W'][it,k:k+10]-d['W_Bstart'][k:k+10];dp=project(dw);dn=dw-dp
     dsN=np.einsum('rhd,srd->srh',dn,d['x_B']);dsP=np.einsum('rhd,srd->srh',dp,d['x_B']);ds=d['response_delta'][it,:,k:k+10]
     r.check('B_transport_null_plus_span',ds,dsN+dsP)
     vals=dict(Kn=-(dw*n).sum((1,2))/den,mean_transport_null=dsN.mean((0,2)),mean_transport_span=dsP.mean((0,2)),mean_transport_W=ds.mean((0,2)),mean_transport_bias=(d['b'][it,k:k+10]-d['b_Bstart'][k:k+10]).mean(1),null_weight_norm2_change=((n+dn)**2-n*n).sum((1,2)))
     for name,mask in masks.items():
      vals['Cn_'+name]=r.measure_C(q,dsN,mask);vals['Cwhole_'+name]=r.measure_C(q,ds,mask)
     for name,v in vals.items():
      curves.append(dict(arm=arm,age=age,time=int(t),branch=branch,metric=name,**r.stats(v)))
      if t==10000:add(arm,age,branch+'_'+name,v)
     ends[branch]=vals
    if t==10000:
     for name,v in dict(Kn_diff=ends['B']['Kn']-ends['control']['Kn'],Sn=(ends['B']['Cn_qpos']-ends['B']['Cn_qneg'])-(ends['control']['Cn_qpos']-ends['control']['Cn_qneg'])).items():
      add(arm,age,name,v);adjusted.append(dict(arm=arm,age=age,metric=name,**followup_stats(v)))
   np.savez_compressed(r.RAW/f'{arm}_{age}_retained_fields.npz',null_weight=n,span_weight=p,q_B=q,q_B_all=bq,input_jump_all=jump,span_jump_all=pjump)
   print(arm,age,'retained follow-up complete',flush=True)
 r.write('retained_secondary.csv',rows);r.write('retained_curves.csv',curves);r.write('retained_followup_adjusted.csv',adjusted);r.write('retained_seed_endpoints.csv',seeds)
 result=dict(checks=r.CHECKS,all_pass=all(v['failed']==0 for v in r.CHECKS.values()),git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r.ROOT,text=True).strip())
 (r.OUT/'retained_checks.json').write_text(json.dumps(result,indent=2))
 if not result['all_pass']:raise SystemExit('retained checks failed')
 print(json.dumps(adjusted),flush=True)
if __name__=='__main__':main()

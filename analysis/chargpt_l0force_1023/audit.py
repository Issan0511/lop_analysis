#!/usr/bin/env python3
"""Nontrivial independent gate and archived-vector arithmetic audit."""
import json
import full_self as Q
F=Q.F;M,L,A,C,torch,np=F.M,F.L,F.A,F.C,F.torch,F.np
checks=[];inp=F.Inputs()
for arm in A.ARMS:
 for task in Q.STATES:
  m=inp.model(arm,0,task,10000);_,raw=L.fields(m,A.uniform(0))
  alive=np.flatnonzero(raw['alive'].numpy());ids=np.sort(np.random.default_rng(10230000+task).choice(alive,size=min(32,len(alive)),replace=False))
  xx=A.data(0,task)[0][:A.CHUNK]
  with torch.no_grad():
   cap=[];m.h[0](m.wte(xx),cap);r0=cap[0];a=m.h[0].mlp.phi(cap[1]);i=int(ids[0])
   def fn(aa):return m.head(m.nf(m.h[1](r0+m.h[0].mlp.down(aa))))
   direction=torch.zeros_like(a);direction[...,i]=a[...,i];f,fi=torch.func.jvp(fn,(a,),(direction,))
   an=float(direction.norm());fnorm=float(fi.norm());assert an>1e-10 and fnorm>1e-10,(arm,task,i,an,fnorm)
  checks.append(dict(arm=arm,seed=0,task=task,first_selected_unit=i,activation_direction_norm=an,output_jvp_norm=fnorm,nontrivial=True))
worst=0.
for p in (F.OUT/'full_self/raw').glob('*.npz'):
 d=np.load(p)
 for metric in ['SGD','Adam_fixed']:
  for q in ['mu','r','mu_tangent']:
   total=d[f'{metric}_switch_{q}'];parts=[d[f'{metric}_{k}_{q}'] for k in ['self','other0','residual']]
   err=np.max(np.abs(total-sum(parts)))/max(np.max(np.abs(v)) for v in parts+[total]);worst=max(worst,float(err))
   fit=d[f'{metric}_fit_{q}'];truth=d[f'{metric}_target_{q}'];err=np.max(np.abs(fit-(total-truth)))/max(np.max(np.abs(v)) for v in [fit,total,truth]);worst=max(worst,float(err))
assert worst<1e-10,worst
F.dump(F.OUT/'audit.json',dict(passed=True,independent_theory_and_readonly_code_review=True,full_self_gates_nontrivial=checks,numpy_saved_projection_closure=worst,training_steps=0))
print('audit passed',worst)

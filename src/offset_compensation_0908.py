"""Registered c=-2 compensation experiment; run as python -m src.offset_compensation_0908."""
from pathlib import Path
import csv,hashlib,itertools,json,time,argparse
import numpy as np
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/offset_compensation_0908'
DATA=Path('/home/issan/Projects/obsidian-research-data/act_offset_review_0908/tail')
ARM='LRoffm2_lr0p00125_1216'
CP=DATA/'ckpts'/(ARM+'_step40000000.pt')
CONDS=['all','K','Kc','KU','KUc']
STEPS=[0,1,2,5,10,20,50]+list(range(100,10001,100))
def arr(t):return t.detach().cpu().numpy()
def fw(n,x,y):
 z=torch.einsum('nhd,pnd->pnh',n['W'],x)+n['b']
 h=torch.where(z>0,z,.1*z)-2
 gate=torch.where(z>0,torch.ones_like(z),torch.full_like(z,.1))
 yh=(h*n['v']).sum(-1)+n['c'];r=yh-y
 return z,h,gate,yh,r
def grad(n,x,h,gate,r):
 dz=2*r[:,:,None]*n['v']*gate
 return {'W':torch.einsum('pnh,pnd->nhd',dz,x)/32,'b':dz.mean(0),
         'v':(2*r[:,:,None]*h).mean(0),'c':2*r.mean(0)}
def forces(n,x,h,gate,r):
 gr=grad(n,x,h,gate,r);mu=x.mean(0)
 gz=-gr['b']-torch.einsum('nhd,nd->nh',gr['W'],mu)
 rho=1+torch.einsum('pnd,nd->pn',x,mu)
 selfz=-2*n['v']**2*(h*gate*rho[:,:,None]).mean(0)
 selfb=-2*n['v']**2*(h*gate).mean(0)
 # Independently calculate rest, rather than defining it as the residual.
 resterr=r[:,:,None]-h*n['v']
 restz=-2*n['v']*(resterr*gate*rho[:,:,None]).mean(0)
 restb=-2*n['v']*(resterr*gate).mean(0)
 assert torch.allclose(gz,selfz+restz,rtol=1e-8,atol=1e-8)
 assert torch.allclose(-gr['b'],selfb+restb,rtol=1e-8,atol=1e-8)
 return gr,gz,selfz,restz,selfb
def load():
 ck=torch.load(CP,map_location='cpu',weights_only=True)
 assert ck['activation']=='leaky_off_m2' and ck['act_alpha']==.1
 n={k:v.double().clone() for k,v in ck['net'].items()}
 R,H,D=n['W'].shape
 bits=torch.tensor(list(itertools.product([0.,1.],repeat=5)),dtype=torch.float64)
 raw=torch.cat([ck['env']['flip_state'].double()[None].expand(32,-1,-1),bits[:,None].expand(-1,R,-1)],-1)
 x=raw-(ck['layer_means'][0].double()[None] if ck['centered_layers'][0] else 0)
 t=ck['teacher']
 pre=torch.einsum('rhd,prd->prh',t['W'].double(),raw)+t['b'].double()
 y=((pre>=t['tau'].double()).double()*t['v'].double()).sum(-1)+t['cout'].double()
 return ck,n,x,y
def validate(ck,n,x,y):
 z,h,gate,yh,r=fw(n,x,y);errs=[]
 for ri,run in enumerate(ck['runs']):
  with np.load(DATA/'logs_tail'/(ARM+'_seed'+str(run['seed'])+'.npz')) as log:
   idx=np.flatnonzero(log['step']==40000000);assert len(idx)==1
   ti=int(idx[0])
   pairs={'layer1_zmean':z[:,ri].mean(0),'layer1_zmin':z[:,ri].min(0).values,
          'layer1_zmax':z[:,ri].max(0).values,'layer1_v_unit':n['v'][ri],
          'eval_loss_exact':r[:,ri].square().mean()}
   errors={}
   for key,val in pairs.items():
    errors[key]=float(np.max(abs(arr(val)-log[key][ti])))
    assert np.allclose(arr(val),log[key][ti],rtol=1e-4,atol=2e-5),(ri,key,errors[key])
   errs.append(errors)
 auto={k:v.clone().requires_grad_() for k,v in n.items()}
 zz,hh,gg,pp,rr=fw(auto,x,y);gs=grad(auto,x,hh,gg,rr)
 rr.square().mean(0).sum().backward()
 ae={k:float((gs[k]-auto[k].grad).abs().max()) for k in gs}
 assert all(torch.allclose(gs[k],auto[k].grad,rtol=1e-9,atol=1e-9) for k in gs)
 return {'baseline_errors':errs,'autograd_max_abs_errors':ae}
def writecsv(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
 OUT.mkdir(parents=True,exist_ok=True)
 ck,base,x0,y0=load();checks=validate(ck,base,x0,y0)
 (OUT/'preflight.json').write_text(json.dumps(checks,indent=2))
 print('PREFLIGHT PASS',flush=True)
 if args.preflight:return
 assert not (OUT/'provenance.json').exists()
 start=time.monotonic()
 z,h,gate,yh,r=fw(base,x0,y0);R,H,D=base['W'].shape;B=len(CONDS)
 L=z.max(0).values<0;U=z.min(0).values>0;K=~(L|U)
 assert torch.all((L.int()+U.int()+K.int())==1)
 groups={'L':L,'K':K,'U':U}
 seeds=[int(r['seed']) for r in ck['runs']]
 lr=torch.tensor([float(r['lr']) for r in ck['runs']],dtype=torch.float64)
 assert torch.allclose(lr,torch.full_like(lr,.00125))
 gr,g0,s0,rest0,sb0=forces(base,x0,h,gate,r)
 rms0=r.square().mean(0).sqrt()
 norm0=torch.sqrt((g0.square()*L).sum(-1)/L.sum(-1).clamp(min=1))
 unit_summary=[]
 for ri,seed in enumerate(seeds):
  for group,mask in groups.items():
   vals=z[:,ri,mask[ri]].mean(0)
   unit_summary.append({'seed':seed,'group':group,'n':int(mask[ri].sum()),
      'zmean_median':float(vals.median()) if len(vals) else None,
      'zmean_min':float(vals.min()) if len(vals) else None,'zmean_max':float(vals.max()) if len(vals) else None})
 writecsv('group_counts.csv',unit_summary)
 # Static compensation possibilities at precisely the same initial state.
 oracle_rows=[];oracle_raw={};affine_checks=[]
 for ri,seed in enumerate(seeds):
  nn={k:v[ri:ri+1].clone() for k,v in base.items()};xx=x0[:,ri:ri+1];yy=y0[:,ri:ri+1]
  for name,mask,addc in [('mean_only',torch.zeros(H,dtype=torch.bool),True),('oracle_K',K[ri],False),
                        ('oracle_Kc',K[ri],True),('oracle_KUc',K[ri]|U[ri],True)]:
   nn={k:v[ri:ri+1].clone() for k,v in base.items()}
   A=arr(h[:,ri,mask])
   if addc:A=np.column_stack([A,np.ones(32)])
   beta,_,rank,svals=np.linalg.lstsq(A,-arr(r[:,ri]),rcond=1e-12)
   nn['v'][0,mask]+=torch.from_numpy(beta[:int(mask.sum())])
   if addc:nn['c'][0]+=float(beta[-1])
   zz,hh,gg,pp,rr=fw(nn,xx,yy);gs,gz,ss,rs,sb=forces(nn,xx,hh,gg,rr)
   maskL=L[ri];den=float(norm0[ri])
   oracle_rows.append({'seed':seed,'condition':name,'rank':int(rank),
     'condition_number':float(svals[0]/svals[-1]) if len(svals) and svals[-1]>0 else None,
     'coefficient_change_norm':float(np.linalg.norm(beta)),
     'residual_ratio':float(rr.square().mean().sqrt()/rms0[ri]),
     'mean_residual_ratio':float(rr.mean().abs()/rms0[ri]),
     'net_force_ratio':float(gz[0,maskL].mean()/den) if den>1e-10 and maskL.any() else None,
     'negative_force_rms_ratio':float(gz[0,maskL].clamp(max=0).square().mean().sqrt()/den) if den>1e-10 and maskL.any() else None,
     'force_rms_ratio':float(gz[0,maskL].square().mean().sqrt()/den) if den>1e-10 and maskL.any() else None,
     'self_force_mean':float(ss[0,maskL].mean()) if maskL.any() else None,
     'rest_force_mean':float(rs[0,maskL].mean()) if maskL.any() else None,
     'bias_force_max_abs':float(gs['b'][0,maskL].abs().max()) if maskL.any() else None})
   oracle_raw[str(seed)+'_'+name]=dict(W=arr(nn['W']),b=arr(nn['b']),v=arr(nn['v']),c=arr(nn['c']),
                                     residual=arr(rr),force=arr(gz))
  A=np.column_stack([np.ones(32),arr(xx[:,0])]);rr=arr(r[:,ri])
  residual=rr-A@np.linalg.lstsq(A,rr,rcond=1e-12)[0]
  zz,hh,gg,pp,ee=fw(nn,xx,yy)
  # Use BASE network, not the last oracle state, for this independent affine null.
  nb={k:v[ri:ri+1] for k,v in base.items()}
  zz,hh,gg,pp,ee=fw(nb,xx,yy)
  gg0=grad(nb,xx,hh,gg,torch.from_numpy(residual[:,None]))
  err=max(float(gg0['b'][0,L[ri]].abs().max()) if L[ri].any() else 0,
          float(gg0['W'][0,L[ri]].abs().max()) if L[ri].any() else 0)
  assert err<1e-8,err
  affine_checks.append(err)
 writecsv('oracle_seed_metrics.csv',oracle_rows)
 torch.save(oracle_raw,OUT/'oracle_states.pt')
 n={key:value.repeat((B,)+(1,)*(value.ndim-1)) for key,value in base.items()}
 initial={k:v.clone() for k,v in n.items()}
 x=x0.repeat(1,B,1);y=y0.repeat(1,B)
 masks=torch.stack([torch.ones_like(K),K,K,K|U,K|U]).reshape(B*R,H)
 cmask=torch.tensor([1,0,1,0,1],dtype=torch.bool).repeat_interleave(R)
 lrall=lr.repeat(B)
 active=torch.ones(B*R,dtype=torch.bool);fail=np.full((B,R),-1)
 rec={};rows=[];records=[]
 for step in range(10001):
  if time.monotonic()-start>900:raise RuntimeError('Wall budget exceeded; do not classify incomplete trajectory')
  z,h,gate,yh,r=fw(n,x,y);loss=r.square().mean(0)
  bad=~torch.isfinite(loss)|(loss>1e8)
  for k,v in n.items():bad|=(~torch.isfinite(v).reshape(B*R,-1).all(-1))|(v.reshape(B*R,-1).abs().max(-1).values>1e6)
  newbad=bad&active
  for idx in np.flatnonzero(arr(newbad)):
   fail[idx//R,idx%R]=step;active[idx]=False
   for key in n:n[key][idx]=initial[key][idx]
  if newbad.any():z,h,gate,yh,r=fw(n,x,y);loss=r.square().mean(0)
  gs,gz,ss,rs,sb=forces(n,x,h,gate,r)
  if step in STEPS:
   instantL=z.max(0).values<0
   expected=-.2*n['v']*r.mean(0)[:,None]
   assert torch.allclose((-gs['b'])[instantL],expected[instantL],rtol=1e-9,atol=1e-9)
   for key in ['W','b','v']:
    m=masks if key!='W' else masks[:,:,None].expand_as(n[key])
    assert torch.equal(n[key][~m],initial[key][~m])
   assert torch.equal(n['c'][~cmask],initial['c'][~cmask])
   vals={'zmean':z.mean(0),'zmin':z.min(0).values,'zmax':z.max(0).values,
         'prediction':yh.T,'residual':r.T,'force_z':gz,'force_b':-gs['b'],
         'self_z':ss,'rest_z':rs,'self_b':sb,'active':active,
         'W':n['W'],'b':n['b'],'v':n['v'],'c':n['c'],'M':(h*gate).mean(0)}
   for label,mask in groups.items():
    vals['output_'+label]=(h*n['v']*mask.repeat(B,1)[None]).sum(-1).T
   for key,val in vals.items():
    a=arr(val).copy().reshape((B,R)+tuple(val.shape[1:]));rec.setdefault(key,[]).append(a)
   records.append(step)
   for ci,condition in enumerate(CONDS):
    for ri,seed in enumerate(seeds):
     j=ci*R+ri;ll=L[ri];den=float(norm0[ri]);good=ll.any() and K[ri].any() and rms0[ri]>1e-8 and den>1e-10 and active[j]
     ff=gz[j,ll]
     rows.append({'condition':condition,'seed':seed,'update':step,'valid':bool(good),
       'residual_ratio':float(loss[j].sqrt()/rms0[ri]),'mean_residual_ratio':float(r[:,j].mean().abs()/rms0[ri]),
       'loss':float(loss[j]),'initial_loss':float(rms0[ri]**2),'teacher_variance':float(y0[:,ri].var(unbiased=False)),
       'net_force_ratio':float(ff.mean()/den) if good else None,
       'negative_force_rms_ratio':float(ff.clamp(max=0).square().mean().sqrt()/den) if good else None,
       'force_rms_ratio':float(ff.square().mean().sqrt()/den) if good else None,
       'down_fraction':float((ff< -1e-10).double().mean()) if ll.any() else None,
       'force_mean':float(ff.mean()) if ll.any() else None,
       'self_mean':float(ss[j,ll].mean()) if ll.any() else None,'rest_mean':float(rs[j,ll].mean()) if ll.any() else None})
   if step%2000==0:print('update',step,'elapsed',round(time.monotonic()-start,1),flush=True)
  if step==10000:break
  live=active.double()
  for key in n:
   mm=cmask if key=='c' else masks
   if key=='W':mm=mm[:,:,None]
   shape=(B*R,)+(1,)*(n[key].ndim-1)
   n[key]-=lrall.reshape(shape)*live.reshape(shape)*mm*gs[key]
 # Actual all-parameter one-step release. Mean z is linear in W,b so this check is exact up to roundoff.
 released={k:v-lrall.reshape((B*R,)+(1,)*(v.ndim-1))*gs[k] for k,v in n.items()}
 znext=fw(released,x,y)[0].mean(0)
 err=float(((znext-z.mean(0))-lrall[:,None]*gz).abs().max())
 assert err<1e-9,err
 out={k:np.stack(v) for k,v in rec.items()}
 out.update(steps=np.array(records),conditions=np.array(CONDS),seeds=np.array(seeds),
            group_L=arr(L),group_K=arr(K),group_U=arr(U),initial_force_rms=arr(norm0),
            initial_residual_rms=arr(rms0),x=arr(x0),y=arr(y0),failure_step=fail)
 # Output bookkeeping checks on all saved states.
 assert np.allclose(out['prediction'],out['output_L']+out['output_K']+out['output_U']+out['c'][...,None],rtol=1e-10,atol=1e-10)
 for ci in [1,2,3,4]:
  assert np.array_equal(out['output_L'][:,ci],np.broadcast_to(out['output_L'][0,ci],out['output_L'][:,ci].shape))
 for ci in [1,2]:
  assert np.array_equal(out['output_U'][:,ci],np.broadcast_to(out['output_U'][0,ci],out['output_U'][:,ci].shape))
 np.savez_compressed(OUT/'trajectories.npz',**out)
 writecsv('seed_metrics.csv',rows)
 meta={'checkpoint':str(CP),'checkpoint_sha256':hashlib.sha256(CP.read_bytes()).hexdigest(),
       'spec_sha256':hashlib.sha256((ROOT/'specs/spec_offset_compensation_0908.md').read_bytes()).hexdigest(),
       'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'checks':checks,'affine_null_errors':affine_checks,'release_one_step_error':err,
       'failures':fail.tolist(),'wall_seconds':time.monotonic()-start,'steps':10000,'lr':arr(lr).tolist(),
       'dtype':'float64','mode':'fixed-task, fixed-support, fixed-centering full-batch GD'}
 (OUT/'provenance.json').write_text(json.dumps(meta,indent=2))
 print('FINISHED',meta['wall_seconds'],flush=True)
if __name__=='__main__':main()

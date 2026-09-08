"""Minority-kick collective-response pilot. See specs/spec_collective_target_only_0908.md."""
import argparse,hashlib,itertools,json,sys,time
from pathlib import Path
import numpy as np
import torch
from src.nets import VecMLPL
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/collective_target_only_0908.json').read_text())
OUT=ROOT/'results/collective_target_only_0908'
CONDS=['free_sham','pulse_plus','pulse_minus','alonec_sham','alonec_plus','alonec_minus',
       'alone_sham','alone_plus','alone_minus']
def apg(z,act,a):
 obj=object.__new__(VecMLPL);obj.act=act;obj.act_alpha=a;obj.act_grad_form='alpha_exp'
 p=obj.act_fn(z);return p,obj.act_grad(z,p)
def forward(n,x,y,act,a):
 z=torch.einsum('nhd,pnd->pnh',n['W'],x)+n['b']
 phi,g=apg(z,act,a)
 yh=(phi*n['v']).sum(-1)+n['c']
 err=yh-y
 return z,phi,g,yh,err
def grads(n,x,phi,g,err):
 d=2*err; dz=d[:,:,None]*n['v']*g
 return {'W':torch.einsum('pnh,pnd->nhd',dz,x)/x.shape[0],
         'b':dz.mean(0),'v':(d[:,:,None]*phi).mean(0),'c':d.mean(0)}
def source_grads(n,x,z,phi,err,act,a):
 net=object.__new__(VecMLPL)
 net.L=1;net.act=act;net.act_alpha=a;net.act_grad_form='alpha_exp';net.v=n['v']
 net.Ws=[n['W']];net.bs=[n['b']];net.c=n['c']
 return {k:v.mean(0) for k,v in zip(['W','b','v','c'],net.grads_batch(x,z,phi,err))}
def load(model):
 cp=Path(model['path']);ck=torch.load(cp,map_location='cpu',weights_only=True)
 n={k:v.double().clone() for k,v in ck['net'].items()}
 R,H,D=n['W'].shape
 bits=torch.tensor(list(itertools.product([0.,1.],repeat=5)),dtype=torch.float64)
 raw=torch.cat([ck['env']['flip_state'].double()[None].expand(32,-1,-1),bits[:,None].expand(-1,R,-1)],-1)
 x=raw-(ck['layer_means'][0].double()[None] if ck['centered_layers'][0] else 0)
 t=ck['teacher']
 pre=torch.einsum('rhd,prd->prh',t['W'].double(),raw)+t['b'].double()
 y=((pre>=t['tau'].double()).double()*t['v'].double()).sum(-1)+t['cout'].double()
 act,a=ck['activation'],float(ck['act_alpha'])
 z,p,g,yh,e=forward(n,x,y,act,a)
 for r,run in enumerate(ck['runs']):
  with np.load(Path(model['logdir'])/(model['arm']+'_seed'+str(run['seed'])+'.npz')) as log:
   i=int(np.flatnonzero(log['step']==5000000)[0])
   k='layer1_zmean' if 'layer1_zmean' in log.files else 'layer1_zbar'
   assert np.allclose(z[:,r].mean(0),log[k][i],rtol=2e-5,atol=5e-6)
   assert np.isclose((e[:,r]**2).mean(),log['eval_loss_exact'][i],rtol=1e-4,atol=2e-5)
 return ck,n,x,y,act,a
def validate(model):
 ck,n,x,y,act,a=load(model)
 z,p,g,yh,e=forward(n,x,y,act,a);ours=grads(n,x,p,g,e);ref=source_grads(n,x,z,p,e,act,a)
 errs={}
 for k in ours:
  err=float((ours[k]-ref[k]).abs().max());errs[k]=err
  assert torch.allclose(ours[k],ref[k],rtol=1e-10,atol=1e-10),(model['label'],k,err)
 # An independent small autograd calculation checks loss normalization and gradients.
 gen=torch.Generator().manual_seed(43219)
 small={'W':torch.randn(2,4,3,generator=gen,dtype=torch.float64,requires_grad=True),
        'b':torch.randn(2,4,generator=gen,dtype=torch.float64,requires_grad=True),
        'v':torch.randn(2,4,generator=gen,dtype=torch.float64,requires_grad=True),
        'c':torch.randn(2,generator=gen,dtype=torch.float64,requires_grad=True)}
 xx=torch.randn(7,2,3,generator=gen,dtype=torch.float64);yy=torch.randn(7,2,generator=gen,dtype=torch.float64)
 zz,pp,gg,hh,ee=forward(small,xx,yy,act,a);ana=grads(small,xx,pp,gg,ee)
 (ee.square().mean(0).sum()).backward()
 auto={}
 for k in ana:
  auto[k]=float((ana[k]-small[k].grad).abs().max().detach())
  assert torch.allclose(ana[k],small[k].grad,rtol=1e-9,atol=1e-9),(model['label'],k,auto[k])
 # Identical copied states and identical masks must produce identical updates.
 left={k:v.clone() for k,v in n.items()};right={k:v.clone() for k,v in n.items()}
 for tick in range(3):
  for copy in [left,right]:
   zz,pp,gg,hh,ee=forward(copy,x,y,act,a);gs=grads(copy,x,pp,gg,ee)
   for key in copy:copy[key]-=.001*gs[key]
 assert all(torch.equal(left[k],right[k]) for k in left)
 return {'model':model['label'],'source_max_abs_errors':errs,'autograd_max_abs_errors':auto,
         'baseline_log_match':True,'identical_copy_3step_bit_match':True}
def arr(x):return x.detach().cpu().numpy()
def run_model(model,start_wall):
 ck,base,x0,y0,act,a=load(model)
 R,H,D=base['W'].shape;B=len(CONDS);N=B*R
 seeds=[int(r['seed']) for r in ck['runs']]
 target=torch.zeros(R,H,dtype=torch.bool)
 k=5 if H==100 else 1
 for r,s in enumerate(seeds):
  idx=np.random.default_rng(CFG['selection_seed']+s).choice(H,k,replace=False)
  target[r,idx]=True
 z0=forward(base,x0,y0,act,a)[0]
 sigma=torch.from_numpy(np.maximum(np.median(arr(z0.std(0,unbiased=False)),axis=1),.001))
 n={key:value.repeat((B,)+(1,)*(value.ndim-1)) for key,value in base.items()}
 x=x0.repeat(1,B,1);y=y0.repeat(1,B)
 mask=torch.ones(B,R,H,dtype=torch.float64)
 vmask=torch.ones_like(mask);kick=torch.zeros_like(mask)
 cmask=torch.ones(B,R,dtype=torch.float64)
 for j,c in enumerate(CONDS):
  sign=1 if c.endswith('plus') else -1
  if c.startswith('global'):
   kick[j]=sign*CFG['global_scale']*sigma[:,None]
  elif c.endswith('plus') or c.endswith('minus'):
   kick[j]=sign*CFG['local_scale']*sigma[:,None]*target
  if c.startswith('alone'):
   mask[j]=target.double();vmask[j]=target.double()
   if c.startswith('alone_'):cmask[j].zero_()
  if c.startswith('hold'):
   mask[j]=(~target).double();vmask[j]=(~target).double()
  if c.startswith('readout'):
   mask[j].zero_();vmask[j]=(~target).double()
 n['b']+=kick.reshape(N,H)
 initial={key:val.clone() for key,val in n.items()}
 lr=torch.tensor([float(r['lr']) for r in ck['runs']],dtype=torch.float64).repeat(B)
 targetn=target.repeat(B,1)
 active=torch.ones(N,dtype=torch.bool);fail_step=np.full((B,R),-1,dtype=int)
 rec={};records=[];immediate_errors=[];completed=True
 for step in range(CFG['steps']+1):
  if time.monotonic()-start_wall>CFG['max_wall_seconds']:
   completed=False;break
  z,phi,g,yh,e=forward(n,x,y,act,a)
  loss=e.square().mean(0)
  bad=~torch.isfinite(loss)|(loss>1e8)
  for key,val in n.items():
   bad |= (~torch.isfinite(val).reshape(N,-1).all(-1))|(val.reshape(N,-1).abs().max(-1).values>1e6)
  newbad=bad&active
  if newbad.any():
   for idx in np.flatnonzero(arr(newbad)):
    fail_step[idx//R,idx%R]=step
   active[newbad]=False
   for key in n:n[key][newbad]=initial[key][newbad]
   z,phi,g,yh,e=forward(n,x,y,act,a);loss=e.square().mean(0)
  gr=grads(n,x,phi,g,e)
  if step==0:
   # Before any updates, non-target states are unchanged. Gradient difference is
   # entirely the shared output residual, regardless of self-balance hypotheses.
   for j,c in enumerate(CONDS):
    if c in ['free_sham','hold_sham','readout_sham'] or c.startswith('global'):continue
    d0=yh[:,j*R:(j+1)*R]-yh[:,:R]
    actual=gr['b'][j*R:(j+1)*R]-gr['b'][:R]
    expected=(2*d0[:,:,None]*base['v']*g[:,:R]).mean(0)
    dif=float((actual-expected)[~target].abs().max())
    assert torch.allclose(actual[~target],expected[~target],rtol=1e-10,atol=1e-10)
    actualw=gr['W'][j*R:(j+1)*R]-gr['W'][:R]
    expectedw=torch.einsum('prh,prd->rhd',2*d0[:,:,None]*base['v']*g[:,:R],x0)/32
    difw=float((actualw-expectedw)[~target].abs().max())
    assert torch.allclose(actualw[~target],expectedw[~target],rtol=1e-10,atol=1e-10)
    immediate_errors.append({'condition':c,'b_error':dif,'W_error':difw})
   for direction in ['plus','minus']:
    pi=CONDS.index('pulse_'+direction)
    for prefix in ['alonec','alone']:
     ai=CONDS.index(prefix+'_'+direction)
     for key in ['W','b','v']:
      gg=gr[key].reshape((B,R)+tuple(gr[key].shape[1:]))
      assert torch.allclose(gg[pi][target],gg[ai][target],rtol=1e-12,atol=1e-12)
   assert torch.equal(n['W'],initial['W'])
   for j in range(B):
    if not CONDS[j].startswith('global'):
     assert torch.equal(n['b'].reshape(B,R,H)[j][~target],base['b'][~target])
  # Actual masks are reflected in recorded update directions.
  live=active.double()
  gr['W']*=mask.reshape(N,H,1)*live[:,None,None]
  gr['b']*=mask.reshape(N,H)*live[:,None]
  gr['v']*=vmask.reshape(N,H)*live[:,None]
  gr['c']*=live*cmask.reshape(N)
  if step in CFG['record_steps']:
   q=phi*g;mp=q.clamp(min=0).mean(0);mn=(-q).clamp(min=0).mean(0)
   prod=phi*n['v']
   oo=(prod*(~targetn)[None]).sum(-1)+n['c']
   ot=(prod*targetn[None]).sum(-1)
   vals={'zmean':z.mean(0),'zmin':z.min(0).values,'zmax':z.max(0).values,'zstd':z.std(0,unbiased=False),
         'm_positive':mp,'m_negative_abs':mn,'M':q.mean(0),'gabs_mean':g.abs().mean(0),
         'force_b':-gr['b'],'force_zmean':-gr['b']-torch.einsum('nhd,nd->nh',gr['W'],x.mean(0)),
         'loss':loss,'active':active,'W':n['W'],'b':n['b'],'v':n['v'],'c':n['c'],
         'output_target':ot.T,'output_other':oo.T,'prediction':yh.T,'phi':phi.permute(1,0,2)}
   for key,val in vals.items():
    val=arr(val).copy().reshape((B,R)+tuple(val.shape[1:]))
    rec.setdefault(key,[]).append(val)
   records.append(step)
   # Ensure pinning does not silently unfreeze W/b/v; readout controls preserve all hidden states.
   for j,c in enumerate(CONDS):
    if c.startswith('alone'):
     for key in ['W','b','v']:
      assert torch.equal(n[key].reshape((B,R)+tuple(n[key].shape[1:]))[j][~target],
                         initial[key].reshape((B,R)+tuple(n[key].shape[1:]))[j][~target])
     if c.startswith('alone_'):
      assert torch.equal(n['c'].reshape(B,R)[j],initial['c'].reshape(B,R)[j])
    if c.startswith('hold'):
     for key in ['W','b','v']:
      assert torch.equal(n[key].reshape((B,R)+tuple(n[key].shape[1:]))[j][target],
                         initial[key].reshape((B,R)+tuple(n[key].shape[1:]))[j][target])
    if c.startswith('readout'):
     for key in ['W','b']:
      assert torch.equal(n[key].reshape((B,R)+tuple(n[key].shape[1:]))[j],
                         initial[key].reshape((B,R)+tuple(n[key].shape[1:]))[j])
   if step in [0,500,1000,2000]:
    print(json.dumps({'model':model['label'],'step':step,'active':int(active.sum()),
                      'elapsed_seconds':round(time.monotonic()-start_wall,1)}),flush=True)
  if step==CFG['steps']:break
  for key in n:
   n[key]-=lr.reshape((N,)+(1,)*(n[key].ndim-1))*gr[key]
 result={key:np.stack(value) for key,value in rec.items()}
 result.update(steps=np.array(records),target=arr(target),sigma=arr(sigma),kick=arr(kick),
               seeds=np.array(seeds),conditions=np.array(CONDS),failure_step=fail_step,lr=arr(lr.reshape(B,R)[0]))
 np.savez_compressed(OUT/(model['label']+'.npz'),**result)
 return {'label':model['label'],'checkpoint':model['path'],'checkpoint_sha256':hashlib.sha256(Path(model['path']).read_bytes()).hexdigest(),
         'last_record':records[-1],'completed':completed,'failed_seed_arms':int((fail_step>=0).sum()),
         'target_units':[np.flatnonzero(arr(target[r])).tolist() for r in range(R)],'sigma':arr(sigma).tolist(),
         'immediate_response_checks':immediate_errors,'activation':act,'alpha':a,'seeds':seeds,'width':H}
def main():
 arg=argparse.ArgumentParser();arg.add_argument('--preflight',action='store_true');args=arg.parse_args()
 OUT.mkdir(parents=True,exist_ok=True)
 validation=[validate(m) for m in CFG['models']]
 (OUT/'preflight.json').write_text(json.dumps(validation,indent=2))
 print('PREFLIGHT PASS',flush=True)
 if args.preflight:return
 assert not (OUT/'provenance.json').exists(),'Do not overwrite an existing run'
 start=time.monotonic();reports=[]
 for model in CFG['models']:
  if time.monotonic()-start>CFG['max_wall_seconds']:break
  reports.append(run_model(model,start))
 meta={'models':reports,'wall_seconds':time.monotonic()-start,'source_sha256':hashlib.sha256((ROOT/'src/nets.py').read_bytes()).hexdigest(),
       'spec_sha256':hashlib.sha256((ROOT/'specs/spec_collective_target_only_0908.md').read_bytes()).hexdigest(),
       'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'config':CFG,'dtype':'float64','mode':'frozen-support full-batch GD'}
 (OUT/'provenance.json').write_text(json.dumps(meta,indent=2))
 print('FINISHED',meta['wall_seconds'],flush=True)
if __name__=='__main__':main()

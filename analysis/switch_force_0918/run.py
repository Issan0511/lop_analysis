"""Registered task-switch audit; see specs/spec_switch_force_0918.md."""
from __future__ import annotations
import argparse, ast, csv, hashlib, itertools, json, platform, subprocess, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.nets import VecMLPL
from src.envs import SCREnv

CPROOT = Path('/home/issan/Projects/obsidian-research-data/zero_attraction_0913/training/ckpts')
RAW = Path('/home/issan/Projects/obsidian-research-data/switch_force_0918')
OUT = ROOT/'results/switch_force_0918'
ARMS = ['LR_a0p1_q0', 'LR_a0p7_q0']
STEPS = [200000, 1000000, 5000000]
TIMES = [0,1,2,5,10,20,50,100,200,500,1000,2000,5000,10000]
LR = .005
CHECKS = {}
MUTATIONS = {}
torch.set_num_threads(1)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def check(name, actual, expected, tol=1e-10):
    a,b=torch.broadcast_tensors(torch.as_tensor(actual).double(),torch.as_tensor(expected).double())
    err=(a-b).abs(); bound=torch.as_tensor(tol,dtype=torch.float64)*(1+b.abs())
    check_bound(name,err,bound)

def check_bound(name,err,bound):
    err,bound=torch.broadcast_tensors(err.double(),torch.as_tensor(bound).double())
    bad=(~torch.isfinite(err)) | (err>bound)
    s=CHECKS.setdefault(name,dict(count=0,failed=0,max_abs=0.,max_fraction=0.))
    s['count']+=err.numel(); s['failed']+=int(bad.sum())
    s['max_abs']=max(s['max_abs'],float(err.max()))
    s['max_fraction']=max(s['max_fraction'],float((err/bound.clamp_min(1e-300)).max()))

def mutation(name,wrong,right):
    detected=bool(((wrong-right).abs()>1e-10*(1+right.abs())).any())
    MUTATIONS[name]=MUTATIONS.get(name,False) or detected

def writecsv(p,rows):
    if rows:
        with p.open('w',newline='') as f:
            wr=csv.DictWriter(f,list(rows[0])); wr.writeheader(); wr.writerows(rows)

def load(arm,step):
    return torch.load(CPROOT/f'{arm}_step{step}.pt',map_location='cpu',weights_only=True)

def offset(flip,target):
    k=flip.double().sum(-1)
    return .05*((k+2.5)-((k+2.5)**2-20*(k+1.25-target**2)).sqrt())

def support(cp,flip,dtype):
    bits=torch.tensor(list(itertools.product([0.,1.],repeat=5)))
    raw=torch.cat([flip.float()[None].expand(32,-1,-1),bits[:,None].expand(-1,10,-1)],-1)
    t=cp['teacher']
    pre=torch.einsum('rhd,srd->srh',t['W'],raw)+t['b']
    y=(((pre>=t['tau']).float()*t['v']).sum(-1)+t['cout']).to(dtype)
    off=offset(flip,float(cp['target_mu_norm']))
    x=raw.to(dtype)-off.to(dtype)[None,:,None]
    return x,y

def new_net(cp,dtype,repeats=1):
    n=VecMLPL(10*repeats,[100],20,torch.Generator().manual_seed(0),'cpu',act=cp['activation'],act_alpha=cp['act_alpha'])
    n.load_state({k:v.to(dtype).repeat((repeats,)+(1,)*(v.ndim-1)) for k,v in cp['net'].items()})
    return n

def parts(n,x,y):
    z,a,f=n.forward_batch(x)
    z,a,f=z.double(),a.double(),f.double()
    e=f-y.double(); k=torch.where(z>0,1.,n.act_alpha)
    v=n.v.double()[None]
    h=2*e[...,None]*v*k
    hs=2*v*v*a*k
    hr=2*v*(e[...,None]-v*a)*k
    check('h_self_rest',h,hs+hr)
    return dict(z=z,k=k,e=e,h=h,hs=hs,hr=hr,loss=(e*e).mean(0))

def frozen_metrics(n,x,y,mu):
    p=parts(n,x,y); xd=x.double(); w=n.W.double()
    xm=torch.einsum('srd,rd->sr',xd,mu)[...,None]
    wx=torch.einsum('rhd,srd->srh',w,xd)
    wm=torch.einsum('rhd,rd->rh',w,mu)
    alpha=-LR*p['h']*wx/(w*w).sum(-1)[None]
    total=-LR*p['h']*xm
    d=dict(force=total.mean(0),bias=(-LR*p['h']).mean(0),
           radial=(alpha*wm[None]).mean(0),tangent=(total-alpha*wm[None]).mean(0),
           h=p['h'].mean(0),hself=p['hs'].mean(0),hrest=p['hr'].mean(0),
           common=(-LR*p['h']*mu.square().sum(-1)[None,:,None]).mean(0),
           loss=p['loss'][:,None].expand(-1,100),zbar=(w*mu[:,None]).sum(-1)+n.b.double())
    d['correlation']=d['force']-d['common']
    for source in ['h','hs','hr']:
        for side in ['pos','neg']:
            mask=(p['z']>0) if side=='pos' else (p['z']<=0)
            d[source+'_'+side]=torch.where(mask,-LR*p[source]*xm,0.).mean(0)
    check('frozen_branch',d['force'],d['h_pos']+d['h_neg'])
    check('frozen_source',d['force'],d['hs_pos']+d['hs_neg']+d['hr_pos']+d['hr_neg'])
    return d,p

def autodiff_check(cp,dtype):
    n=new_net(cp,dtype); x,y=support(cp,cp['env']['flip_state'],dtype)
    for i in [0,7,31]:
        xx,yy=x[i],y[i]
        z,a,f=n.forward(xx); grads=n.grads(xx,z,a,f-yy)
        w,b,v,c=[q.detach().clone().requires_grad_() for q in [n.W,n.b,n.v,n.c]]
        zz=torch.einsum('rhd,rd->rh',w,xx)+b
        aa=torch.where(zz>0,zz,n.act_alpha*zz)
        ff=(aa*v).sum(-1)+c
        ag=torch.autograd.grad(((ff-yy)**2).sum(),[w,b,v,c])
        for name,g,ga in zip(['W','b','v','c'],grads,ag):
            check('autograd_'+str(dtype)+'_'+name,g,ga,1e-10 if dtype==torch.float64 else 2e-5)
        if dtype==torch.float64:
            mutation('missing_MSE_factor',grads[0]/2,ag[0])

def preflight():
    old=Path('/home/issan/Projects/claude/zero_attraction_0913')
    records={}
    for filename,cls,names in [('src/nets.py','VecMLPL',['forward_layers','forward_layers_batch','grads_layers','grads_layers_batch','sgd_step_layers']),('src/envs.py','SCREnv',['maybe_flip','step'])]:
        trees=[ast.parse((p/filename).read_text()) for p in [old,ROOT]]
        classes=[next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==cls) for t in trees]
        for name in names:
            nodes=[next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name) for c in classes]
            equal=ast.dump(nodes[0],include_attributes=False)==ast.dump(nodes[1],include_attributes=False)
            records[cls+'.'+name]=equal
            if not equal: raise RuntimeError('source mismatch '+name)
    for arm in ARMS:
        cp=load(arm,200000)
        for dtype in [torch.float64,torch.float32]: autodiff_check(cp,dtype)
        flip=cp['env']['flip_state']
        check('saved_offset',offset(flip,float(cp['target_mu_norm']))[:,None].expand(-1,20),cp['layer_means'][0])
        gen=torch.Generator().manual_seed(20260918)
        idx=torch.randint(0,15,(10,),generator=gen); expected=flip.clone()
        expected[torch.arange(10),idx]=1-expected[torch.arange(10),idx]
        gen2=torch.Generator().manual_seed(1)
        env=SCREnv(10,20,15,torch.full((10,),10000),gen2,'cpu')
        env.load_state(cp['env']); env.gen=torch.Generator().manual_seed(20260918)
        env.maybe_flip(); check('original_env_switch',env.flip_state,expected,0.)
        rnd=torch.randint(0,2,(10,5),generator=gen)
        original_rnd=torch.randint(0,2,(10,5),generator=env.gen)
        check('original_rng_order',rnd,original_rnd,0.)
    (OUT/'source_equivalence.json').write_text(json.dumps(records,indent=2))

def exhaustive(cp,arm,step):
    n=new_net(cp,torch.float64); flip=cp['env']['flip_state']
    xo,yo=support(cp,flip,torch.float64)
    data=defaultdict(list)
    for j in range(15):
        ff=flip.clone(); ff[:,j]=1-ff[:,j]
        xn,yn=support(cp,ff,torch.float64); mu=xn.mean(0)
        ds=[]; ps=[]
        for x,y in [(xo,yo),(xn,yo),(xo,yn),(xn,yn)]:
            d,p=frozen_metrics(n,x,y,mu); ds.append(d); ps.append(p)
        for key in ds[0]: data[key].append(torch.stack([d[key] for d in ds]))
        delta=ps[3]['h']-ps[0]['h']; de=ps[3]['e']-ps[0]['e']; dk=ps[3]['k']-ps[0]['k']
        v=n.v.double()[None]
        pieces=[2*v*de[...,None]*ps[0]['k'],2*v*ps[0]['e'][...,None]*dk,2*v*de[...,None]*dk]
        check('switch_h_product',delta,sum(pieces))
        # Exact force decomposition, including changed x, referenced to new mu.
        xo_mu=torch.einsum('srd,rd->sr',xo,mu)[...,None]
        dx_mu=torch.einsum('srd,rd->sr',xn-xo,mu)[...,None]
        for label,hpiece in zip(['residual','gate','residual_gate_interaction'],pieces):
            data['change_'+label].append((-LR*hpiece*xo_mu).mean(0))
        data['change_input_vector'].append((-LR*ps[3]['h']*dx_mu).mean(0))
        check('switch_force_product',ds[3]['force']-ds[0]['force'],sum(data['change_'+k][-1] for k in ['residual','gate','residual_gate_interaction','input_vector']))
        check('target_only_self_oldx',ps[0]['hs'],ps[2]['hs'],0.)
        check('target_only_self_newx',ps[1]['hs'],ps[3]['hs'],0.)
        inp=.5*((ds[1]['force']-ds[0]['force'])+(ds[3]['force']-ds[2]['force']))
        target=.5*((ds[2]['force']-ds[0]['force'])+(ds[3]['force']-ds[1]['force']))
        check('shapley_force',inp+target,ds[3]['force']-ds[0]['force'])
        data['input_shapley'].append(inp); data['target_shapley'].append(target)
        data['factorial_interaction'].append(ds[3]['force']-ds[1]['force']-ds[2]['force']+ds[0]['force'])
        data['input_jump'].append(torch.einsum('rhd,rd->rh',n.W,mu-xo.mean(0)))
        data['mu_new'].append(mu)
        mutation('omitted_rest',ds[3]['hs_pos']+ds[3]['hs_neg'],ds[3]['force'])
        mutation('omitted_tangent',ds[3]['radial'],ds[3]['force'])
        mutation('omitted_input_jump',torch.zeros_like(data['input_jump'][-1]),data['input_jump'][-1])
    np.savez_compressed(RAW/f'{arm}_{step}_frozen.npz',**{k:torch.stack(v).numpy() for k,v in data.items()})

def trajectory(cp,arm,step,dtype,T=10000,tag=''):
    began=time.time(); name=str(dtype).split('.')[-1]
    gen=torch.Generator().manual_seed(20260918)
    idx=torch.randint(0,15,(10,),generator=gen)
    flip=cp['env']['flip_state'].clone(); newflip=flip.clone()
    newflip[torch.arange(10),idx]=1-newflip[torch.arange(10),idx]
    rnd=torch.randint(0,2,(T,10,5),generator=gen)
    powers=torch.tensor([16,8,4,2,1]); indices=(rnd*powers).sum(-1)
    xo,yo=support(cp,flip,dtype); xn,yn=support(cp,newflip,dtype)
    xall=torch.cat([xn,xo],1); yall=torch.cat([yn,yo],1)
    mun=xn.double().mean(0); muo=xo.double().mean(0)
    mu=mun.repeat(2,1); mun2=mu.square().sum(-1)[:,None]
    n=new_net(cp,dtype,2); lrs=torch.full((20,),LR,dtype=dtype)
    w0=n.W.double().clone(); b0=n.b.double().clone()
    wm0=(w0*mu[:,None]).sum(-1)
    jump=(w0[:10]*(mun-muo)[:,None]).sum(-1)
    initial=dict(W_start=w0.numpy(),b_start=b0.numpy(),mu_new=mun.numpy(),mu_old=muo.numpy(),flip_index=idx.numpy(),support_index=indices.numpy(),input_jump=jump.numpy())
    totals=defaultdict(lambda:torch.zeros(20,100,dtype=torch.float64))
    saved=defaultdict(list); save_times=[]
    ids=torch.arange(20)
    def record(t):
        w=n.W.double(); b=n.b.double(); r=w.norm(dim=-1); r0=w0.norm(dim=-1)
        wm=(w*mu[:,None]).sum(-1); c=wm/r; c0=wm0/r0
        delta=wm-wm0
        check('trajectory_W_closure_'+name,delta,totals['W_actual'])
        check('trajectory_b_closure_'+name,b-b0,totals['b_actual'])
        check('trajectory_radial_tangent_'+name,delta,totals['radial']+totals['tangent'])
        check('trajectory_source_'+name,delta,totals['self_pos']+totals['self_neg']+totals['rest_pos']+totals['rest_neg']+totals['W_round'])
        check('trajectory_down_up_'+name,delta,totals['down']+totals['up'])
        length=(r-r0)*(c+c0)/2; direction=(c-c0)*(r+r0)/2
        check('endpoint_length_direction_'+name,delta,length+direction)
        end_change=(wm[:10]+b[:10])-((w0[:10]*muo[:,None]).sum(-1)+b0[:10])
        check('boundary_plus_learning_'+name,end_change,jump+delta[:10]+b[:10]-b0[:10])
        d,p=frozen_metrics(n,xall,yall,mu)
        state=dict(W=n.W.double(),b=b,v=n.v.double(),cout=n.c.double(),wm=wm,zbar=wm+b,
                   projection_change=delta,length=length,direction=direction,
                   cosine=wm/(r*mu.norm(dim=-1)[:,None]),norm=r,
                   centered_normsq=(w-w.mean(-1,keepdim=True)).square().sum(-1),
                   oldmean_projection_change=((w-w0)*muo.repeat(2,1)[:,None]).sum(-1),loss=p['loss'])
        for k,v in state.items(): saved[k].append(v.detach().numpy().copy())
        for k,v in totals.items(): saved['cum_'+k].append(v.numpy().copy())
        for k,v in d.items(): saved['expected_'+k].append(v.numpy().copy())
        save_times.append(t)
    # Initialize all keys before recording zero.
    keys=['W_actual','b_actual','radial','tangent','centered_transport','rowmean_transport','self_pos','self_neg','rest_pos','rest_neg','bias_self','bias_rest','W_round','b_round','common_mu','input_correlation','pos','neg','down','up','E','R','J','norm_round']
    for k in keys: totals[k]
    record(0)
    for t in range(T):
        ii=indices[t].repeat(2); x=xall[ii,ids]; y=yall[ii,ids]
        w=n.W.double().clone(); b=n.b.double().clone(); v=n.v.double().clone()
        pre,a,f=n.forward(x); gW,gb,gv,gc=n.grads(x,pre,a,f-y)
        k=torch.where(pre.double()>0,1.,n.act_alpha); h=2*(f.double()-y.double())[:,None]*v*k
        hs=2*v*v*a.double()*k; hr=h-hs
        xd=x.double(); xm=(xd*mu).sum(-1)[:,None]
        # Use production gradients and update; diagnostics never mutate these tensors.
        n.sgd_step(lrs,gW,gb,gv,gc)
        dw=n.W.double()-w; db=n.b.double()-b
        actual=(dw*mu[:,None]).sum(-1); nominal=-LR*h*xm
        radial=((w*dw).sum(-1)/w.square().sum(-1))*(w*mu[:,None]).sum(-1)
        wc=w-w.mean(-1,keepdim=True); dwc=dw-dw.mean(-1,keepdim=True)
        xc=xd-xd.mean(-1,keepdim=True)
        lin=-2*LR*h*(wc*xc[:,None]).sum(-1)
        quad=LR**2*h*h*xc.square().sum(-1)[:,None]
        n0=wc.square().sum(-1); n1=(wc+dwc).square().sum(-1)
        err=(n1-n0-lin-quad).abs()
        bound=64*torch.finfo(dtype).eps*(1+n0+n1+lin.abs()+quad)
        check_bound('actual_norm_budget_'+name,err,bound)
        bnd=64*torch.finfo(dtype).eps*(1+(w*mu[:,None]).sum(-1).abs()+((w+dw)*mu[:,None]).sum(-1).abs()+nominal.abs())
        check_bound('actual_transport_'+name,(actual-nominal).abs(),bnd)
        pos=pre>0
        increments=dict(W_actual=actual,b_actual=db,radial=radial,tangent=actual-radial,
            centered_transport=(dwc*mu[:,None]).sum(-1),rowmean_transport=dw.mean(-1)*mu.sum(-1)[:,None],
            self_pos=torch.where(pos,-LR*hs*xm,0.),self_neg=torch.where(~pos,-LR*hs*xm,0.),
            rest_pos=torch.where(pos,-LR*hr*xm,0.),rest_neg=torch.where(~pos,-LR*hr*xm,0.),
            bias_self=-LR*hs,bias_rest=-LR*hr,W_round=actual-nominal,b_round=db+LR*h,
            common_mu=-LR*h*mun2,input_correlation=-LR*h*(xm-mun2),
            pos=torch.where(pos,nominal,0.),neg=torch.where(~pos,nominal,0.),
            down=actual.clamp_max(0),up=actual.clamp_min(0),E=(-lin).clamp_min(0),R=lin.clamp_min(0),J=quad,norm_round=n1-n0-lin-quad)
        for key,val in increments.items(): totals[key]+=val
        if t+1 in TIMES or t+1==T: record(t+1)
    for actual,expected,label in [
        (totals['W_actual'],totals['pos']+totals['neg']+totals['W_round'],'branches'),
        (totals['W_actual'],totals['common_mu']+totals['input_correlation']+totals['W_round'],'common_mu'),
        (totals['W_actual'],totals['centered_transport']+totals['rowmean_transport'],'centering'),
        (totals['b_actual'],totals['bias_self']+totals['bias_rest']+totals['b_round'],'bias_sources')]:
        check('final_'+label+'_'+name,actual,expected)
    if T>=100:
        mutation('omitted_bias',totals['W_actual'],totals['W_actual']+totals['b_actual'])
    payload=dict(initial,**{k:np.stack(v) for k,v in saved.items()},time=np.array(save_times))
    path=RAW/f'{arm}_{step}_{name}{tag}_trajectory.npz'
    np.savez_compressed(path,**payload)
    print(json.dumps(dict(arm=arm,step=step,dtype=name,steps=T,elapsed=round(time.time()-began,2),output=str(path))),flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--preflight',action='store_true'); ap.add_argument('--arm',choices=ARMS); ap.add_argument('--step',type=int); ap.add_argument('--dtype',choices=['float64','float32']); ap.add_argument('--steps',type=int,default=10000)
    args=ap.parse_args(); RAW.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    began=time.time()
    if args.preflight:
        preflight()
        cp=load(ARMS[0],200000)
        for dtype in [torch.float64,torch.float32]: trajectory(cp,ARMS[0],200000,dtype,T=100,tag='_preflight')
        label='preflight'
    else:
        arms=[args.arm] if args.arm else ARMS; steps=[args.step] if args.step else STEPS
        for arm in arms:
            for step in steps:
                cp=load(arm,step)
                if args.dtype in [None,'float64']: exhaustive(cp,arm,step)
                for name in ([args.dtype] if args.dtype else ['float64','float32']):
                    trajectory(cp,arm,step,getattr(torch,name),T=args.steps)
        label='_'.join(str(x) for x in [args.arm,args.step,args.dtype] if x) or 'main'
    result=dict(checks=CHECKS,mutations=MUTATIONS,all_checks_pass=all(s['failed']==0 for s in CHECKS.values()),elapsed=time.time()-began,
        git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),python=sys.version,torch=torch.__version__,platform=platform.platform())
    (OUT/f'checks_{label}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['checks']}),flush=True)
    if not result['all_checks_pass']: raise SystemExit(1)

if __name__=='__main__': main()

"""A orthogonal increment, B exposure, and fixed-receiver counteraction."""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys, time
from collections import defaultdict
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from analysis.switch_force_0918 import run as base

RAW=Path('/home/issan/Projects/obsidian-research-data/orth_reveal_0918')
PARENT=Path('/home/issan/Projects/obsidian-research-data/switch_force_0918')
OUT=ROOT/'results/orth_reveal_0918'
ARMS=base.ARMS; AGES=base.STEPS; TIMES=base.TIMES; LR=.005
torch.set_num_threads(1)

def ctr(x):return x-x.mean(-1,keepdim=True)

def decompose(w0,w1):
    delta=w1-w0
    alpha=(w0*delta).sum(-1)/w0.square().sum(-1)
    p=alpha[...,None]*w0;u=delta-p
    base.check('u_orthogonal_w0',(u*w0).sum(-1),torch.zeros_like(alpha))
    base.check('p_plus_u',p+u,delta)
    base.check('initial_u_coefficient',(w1*u).sum(-1),u.square().sum(-1))
    base.check('norm_budget',w1.square().sum(-1)-w0.square().sum(-1),(2*alpha+alpha.square())*w0.square().sum(-1)+u.square().sum(-1))
    return delta,alpha,p,u

def phi(z,a):return torch.where(z>0,z,a*z)

def prepare(arm,age,dtype):
    name=str(dtype).split('.')[-1]
    cp=base.load(arm,age); parent=np.load(PARENT/f'{arm}_{age}_{name}_trajectory.npz')
    gen=torch.Generator().manual_seed(20260918)
    ia=torch.randint(0,15,(10,),generator=gen)
    ra=torch.randint(0,2,(10000,10,5),generator=gen)
    base.check('A_flip_draw',ia,parent['flip_index'],0.)
    base.check('A_random_stream',(ra*torch.tensor([16,8,4,2,1])).sum(-1),parent['support_index'],0.)
    fa=cp['env']['flip_state'].clone();fa[torch.arange(10),ia]=1-fa[torch.arange(10),ia]
    ib=torch.randint(0,15,(10,),generator=gen)
    rb=torch.randint(0,2,(10000,10,5),generator=gen)
    fb=fa.clone();fb[torch.arange(10),ib]=1-fb[torch.arange(10),ib]
    state={k:torch.as_tensor(parent[q][-1,:10]).to(dtype).clone() for k,q in [('W','W'),('b','b'),('v','v'),('c','cout')]}
    ca=dict(cp,net=state,env=dict(flip_state=fa,t=age+10000))
    xa,ya=base.support(cp,fa,dtype);xb,yb=base.support(cp,fb,dtype)
    w0=torch.as_tensor(parent['W_start'][:10]).double()
    return dict(cp=cp,ca=ca,w0=w0,w1=state['W'].double(),xa=xa,ya=ya,xb=xb,yb=yb,
        flip_A=fa,flip_B=fb,B_index=ib,B_random=rb,B_support_index=(rb*torch.tensor([16,8,4,2,1])).sum(-1))

def span_projector(x):
    ps=[];ranks=[]
    for seed in range(10):
        _,s,vh=np.linalg.svd(x[:,seed].numpy(),full_matrices=False)
        rank=int(np.sum(s>1e-10*s[0]));P=vh[:rank].T@vh[:rank]
        ps.append(P);ranks.append(rank)
    return torch.tensor(np.stack(ps)),np.array(ranks)

def geometry(task,arm,age):
    ca=task['ca'];n=base.new_net(ca,torch.float64)
    xa=task['xa'].double();za=n.forward_batch(xa)[0].double()
    payload=dict(w0=task['w0'].numpy(),w1=task['w1'].numpy(),b1=n.b.numpy(),v1=n.v.numpy(),c1=n.c.numpy(),
        x_A=xa.numpy(),z_A=za.numpy(),flip_A=task['flip_A'].numpy())
    for coord in ['whole','centered']:
        w0=task['w0'] if coord=='whole' else ctr(task['w0'])
        w1=task['w1'] if coord=='whole' else ctr(task['w1'])
        dx,alpha,p,u=decompose(w0,w1)
        xs=xa if coord=='whole' else ctr(xa)
        P,rank=span_projector(xs)
        key=lambda k:coord+'_'+k
        fields=dict(delta=dx,alpha=alpha,p=p,u=u)
        for label,val in [('w0',w0),('delta',dx),('p',p),('u',u)]:
            null=val-torch.einsum('rhi,rij->rhj',val,P)
            fields[label+'_null']=null
        base.check('null_cancellation_'+coord,fields['delta_null'],fields['p_null']+fields['u_null'])
        base.check('span_projector_'+coord,P@P,P)
        base.check('all_A_inputs_in_span_'+coord,torch.einsum('sri,rij->srj',xs,P),xs)
        fields['q_A']=torch.einsum('rhd,srd->srh',u,xa)
        fields['p_A']=torch.einsum('rhd,srd->srh',p,xa)
        fields['delta_A']=torch.einsum('rhd,srd->srh',dx,xa)
        fields['act_A']=phi(za,n.act_alpha)-phi(za-fields['q_A'],n.act_alpha)
        fields['out_A']=fields['act_A']*n.v[None]
        fields['span_projector']=P
        for k,v in fields.items():payload[key(k)]=v.numpy()
        payload[key('span_rank')]=rank
        bb=defaultdict(list)
        for j in range(15):
            fb=task['flip_A'].clone();fb[:,j]=1-fb[:,j]
            xb,yb=base.support(task['cp'],fb,torch.float64)
            zb=n.forward_batch(xb)[0].double()
            q=torch.einsum('rhd,srd->srh',u,xb)
            pp=torch.einsum('rhd,srd->srh',p,xb)
            dd=torch.einsum('rhd,srd->srh',dx,xb)
            un=torch.einsum('rhd,srd->srh',fields['u_null'],xb)
            pn=torch.einsum('rhd,srd->srh',fields['p_null'],xb)
            dn=torch.einsum('rhd,srd->srh',fields['delta_null'],xb)
            base.check('B_projection_closure_'+coord,q+pp,dd)
            base.check('B_null_cancellation_'+coord,un+pn,dn)
            base.check('B_mean_reveal_'+coord,(q-fields['q_A']).mean(0),torch.einsum('rhd,rd->rh',u,xb.mean(0)-xa.mean(0)))
            for k,v in dict(q=q,p=pp,delta=dd,q_null=un,p_null=pn,delta_null=dn,z=zb,
                act=phi(zb,n.act_alpha)-phi(zb-q,n.act_alpha),x=xb).items():bb[k].append(v.numpy())
        for k,v in bb.items():payload[key('B_'+k)]=np.stack(v)
    np.savez_compressed(RAW/f'{arm}_{age}_geometry.npz',**payload)

def run_B(task,arm,age,dtype,T=10000,tag=''):
    started=time.time();name=str(dtype).split('.')[-1]
    n=base.new_net(task['ca'],dtype,2);lrs=torch.full((20,),LR,dtype=dtype)
    xa,ya,xb,yb=[task[k] for k in ['xa','ya','xb','yb']]
    xall=torch.cat([xb,xa],1);yall=torch.cat([yb,ya],1)
    wstart=n.W.double().clone();bstart=n.b.double().clone()
    refs={}
    for coord in ['whole','centered']:
        w0=task['w0'] if coord=='whole' else ctr(task['w0'])
        w1=task['w1'] if coord=='whole' else ctr(task['w1'])
        refs[coord]=decompose(w0,w1)[3]
    zb0=torch.einsum('rhd,srd->srh',task['w1'],xb.double())+task['ca']['net']['b'].double()
    q={coord:torch.einsum('rhd,srd->srh',u,xb.double()) for coord,u in refs.items()}
    fixed=dict(qpos=q['whole']>0,zpos=zb0>0)
    sums={coord:{side:torch.zeros(20,100,dtype=torch.float64) for side in ['positive','negative']} for coord in refs}
    eventpos=torch.zeros(20,100,dtype=torch.int64)
    saved=defaultdict(list); times=[]; ids=torch.arange(20)
    def record(t):
        w=n.W.double();b=n.b.double();delta=w-wstart
        ds=torch.einsum('rhd,srd->srh',delta,xb.double().repeat(1,2,1))
        fields=dict(W=w,b=b,v=n.v.double(),cout=n.c.double(),response_delta=ds,
            response_delta_with_bias=ds+(b-bstart)[None],
            loss=(n.forward_batch(xall)[2].double()-yall.double()).square().mean(0),positive_source_count=eventpos,
            mean_A=torch.einsum('rhd,rd->rh',w,xa.double().mean(0).repeat(2,1))+b,
            mean_B=torch.einsum('rhd,rd->rh',w,xb.double().mean(0).repeat(2,1))+b)
        for coord,u in refs.items():
            inner=(delta*u.repeat(2,1,1)).sum(-1)
            base.check('B_direction_source_'+name+'_'+coord,inner,sums[coord]['positive']+sums[coord]['negative'])
            fields[coord+'_inner']=inner
            for side in ['positive','negative']:fields[coord+'_source_'+side]=sums[coord][side]
        for k,v in fields.items():saved[k].append(v.detach().numpy().copy())
        times.append(t)
    record(0)
    for t in range(T):
        ix=task['B_support_index'][t].repeat(2);x=xall[ix,ids];y=yall[ix,ids]
        old=n.W.double().clone()
        z,a,f=n.forward(x);g=n.grads(x,z,a,f-y)
        n.sgd_step(lrs,*g)
        dw=n.W.double()-old;pos=z>0;eventpos+=pos.long()
        for coord,u in refs.items():
            dot=(dw*u.repeat(2,1,1)).sum(-1)
            sums[coord]['positive']+=torch.where(pos,dot,0.)
            sums[coord]['negative']+=torch.where(~pos,dot,0.)
        if t+1 in TIMES or t+1==T:record(t+1)
    payload={k:np.stack(v) for k,v in saved.items()}
    payload.update(time=np.array(times),W_Bstart=wstart.numpy(),b_Bstart=bstart.numpy(),x_A=xa.double().numpy(),x_B=xb.double().numpy(),
        z_Bstart=zb0.numpy(),B_flip_index=task['B_index'].numpy(),B_support_index=task['B_support_index'][:T].numpy(),
        u_whole=refs['whole'].numpy(),u_centered=refs['centered'].numpy(),q_B=q['whole'].numpy(),q_B_centered=q['centered'].numpy(),
        fixed_q_positive=fixed['qpos'].numpy(),fixed_z_positive=fixed['zpos'].numpy())
    np.savez_compressed(RAW/f'{arm}_{age}_{name}{tag}_B.npz',**payload)
    print(json.dumps(dict(arm=arm,age=age,dtype=name,steps=T,seconds=round(time.time()-started,2))),flush=True)

def preflight():
    w0=torch.tensor([[[1.,0.,0.]]],dtype=torch.float64)
    w1=torch.tensor([[[.8,.3,0.]]],dtype=torch.float64)
    d,a,p,u=decompose(w0,w1)
    base.check('synthetic_alpha',a,torch.tensor([[-.2]],dtype=torch.float64))
    base.check('synthetic_u',u,torch.tensor([[[0.,.3,0.]]],dtype=torch.float64))
    # Mean-zero response need not be invisible over all inputs.
    x=torch.tensor([[1.,1.],[1.,-1.]],dtype=torch.float64);v=torch.tensor([0.,1.],dtype=torch.float64)
    base.mutation('mean_zero_is_not_invisible',torch.zeros(2),x@v)
    # D lies in A input span; decomposition can manufacture opposed null parts.
    w0=torch.tensor([[[1.,1.]]],dtype=torch.float64);w1=w0+torch.tensor([[[-.2,0.]]],dtype=torch.float64)
    d,a,p,u=decompose(w0,w1)
    base.check('synthetic_null_sum',p[...,1]+u[...,1],torch.zeros(1,1))
    base.mutation('omitted_radial_null_cancellation',u[...,1],d[...,1])
    q=torch.tensor([2.,-1.],dtype=torch.float64);ds=torch.tensor([-1.,0.],dtype=torch.float64)
    correct=-(q*ds*(q>0)).sum()/q[q>0].square().sum()
    wrong=-(q*ds*(q<0)).sum()/q[q<0].square().sum()
    base.check('synthetic_Cplus',correct,torch.tensor(.5,dtype=torch.float64))
    base.mutation('reversed_receiver_signs',wrong,correct)
    z0=torch.tensor([1.,-1.]);z1=-z0
    base.mutation('moving_receiver_groups', (z1>0).double(),(z0>0).double())
    for dtype in [torch.float64,torch.float32]:
        task=prepare(ARMS[0],AGES[0],dtype)
        base.autodiff_check(task['ca'],dtype)
        cp=task['cp'];gen=torch.Generator().manual_seed(20260918)
        torch.randint(0,15,(10,),generator=gen);torch.randint(0,2,(10000,10,5),generator=gen)
        env=base.SCREnv(10,20,15,torch.full((10,),10000),torch.Generator().manual_seed(0),'cpu')
        env.load_state(task['ca']['env']);env.gen=gen
        for t in range(3):
            raw=env.step()
            base.check('B_original_env_flip',raw[:,:15],task['flip_B'],0.)
            base.check('B_original_env_stream',raw[:,15:],task['B_random'][t],0.)
        # Nominal gradient must stay in this fixed task's input span.
        if dtype==torch.float64:
            n=base.new_net(task['ca'],dtype);x=task['xa'];y=task['ya'];P,_=span_projector(x)
            z,a,f=n.forward_batch(x);g=n.grads_batch(x,z,a,f-y)[0]
            base.check('nominal_gradient_in_A_span',g,torch.einsum('srhi,rij->srhj',g,P))
        run_B(task,ARMS[0],AGES[0],dtype,T=100,tag='_preflight')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    RAW.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if args.preflight:preflight();label='preflight'
    else:
        for arm in ARMS:
            for age in AGES:
                for dtype in [torch.float64,torch.float32]:
                    task=prepare(arm,age,dtype)
                    if dtype==torch.float64:geometry(task,arm,age)
                    run_B(task,arm,age,dtype)
        label='main'
    result=dict(checks=base.CHECKS,mutations=base.MUTATIONS,all_pass=all(v['failed']==0 for v in base.CHECKS.values()),git_hash=head)
    (OUT/f'checks_{label}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    if not result['all_pass']:raise SystemExit(1)

if __name__=='__main__':main()

"""Replay frozen CondA trajectories and account for each actual SGD update."""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, os, sys, time, subprocess
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src import edge_law_0905 as E
from src.gate_dial_0902 import setup_arm_dial, _arm
from src.gate_dose import forward_gate
from src.elu_swamp import grads_centered_elu
from src.ratchet_log import full_support_ro
torch.set_num_threads(1)
OLD=Path('/home/issan/Projects/claude/proj_004_drift/results/scale_attractor_b_0906')
RAW=Path('/home/issan/Projects/obsidian-research-data/width_budget_0916')
OUT=ROOT/'results/width_budget_0916'
ARMS=['LRa0p35_1216','LRa0p4_1216','LRa0p45_1216','LRa0p55_1216','LRa0p6_1216','LRa0p65_1216','LRa0p8_1216']
GEOMS=['full','centered','variance'];METRICS=['erosion','recovery','injection','pred_linear','pred_injection','self_linear','rest_linear']
PHASES=[0,20,100,1000,10000];PROBES={0,20,100,1000,5000,9999}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def writejson(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False))

def restored(arm,start):
    cfg=E.build_cfg(ROOT/'configs/scale_attractor_b_0906.yaml')
    st=setup_arm_dial(cfg,_arm(cfg,arm),'cpu')
    ckpath=OLD/'ckpts'/f'{arm}_step{start}.pt'
    ck=torch.load(ckpath,weights_only=False,map_location='cpu')
    for k,v in ck['teacher'].items():assert torch.equal(v,st['teacher'].state_dict()[k]),k
    # randint over a segment consumes the same integer stream as individual steps.
    for _ in range(start//10000):st['env'].segment(10000)
    assert st['env'].t==ck['env']['t'] and torch.equal(st['env'].flip_state,ck['env']['flip_state'])
    st['net'].load_state(ck['net']);st['running_mean'].copy_(ck['running_mean'])
    assert torch.equal(st['layer_means'][0],ck['layer_means'][0])
    return st,ckpath

def norm2(w):
    full=w.square().sum(-1)
    return torch.stack([full,full-w.sum(-1).square()/w.shape[-1],w[...,15:].square().sum(-1)/4])

def inner(w,g):
    full=(w*g).sum(-1)
    return torch.stack([full,full-w.sum(-1)*g.sum(-1)/w.shape[-1],(w[...,15:]*g[...,15:]).sum(-1)/4])

def probe(st):
    """Frozen exact-support expectations, separate from the sampled update."""
    net=st['net'];x=full_support_ro(st['env']).double()-st['layer_means'][0].double()[None]
    w=net.Ws[0].double();b=net.bs[0].double();v=net.v.double()
    z=torch.einsum('rhd,prd->prh',w,x)+b
    phi=net.act_fn(z);gate=net.act_grad(z,phi)
    teacher=st['teacher'];raw=x+st['layer_means'][0].double()[None]
    zz=torch.einsum('rhd,prd->prh',teacher.W.double(),raw)+teacher.b.double()
    y=((zz>=teacher.tau.double()).double()*teacher.v.double()).sum(-1)+teacher.cout.double()
    residual=(phi*v).sum(-1)+net.c.double()-y
    q=2*residual[:,:,None]*v*gate
    g=q[:,:,:,None]*x[:,:,None,:];mean_g=g.mean(0)
    esq=norm2(g).mean(1);meansq=norm2(mean_g)
    a=inner(w,mean_g);eta=st['lr'].double()[None,:,None]
    qphi=phi*gate;proj=torch.einsum('rhd,prd->prh',w[...,15:],x[...,15:])
    selfrate=-v.square()*(qphi*proj).mean(0)
    return dict(A=a.numpy(),expected_injection=(eta.square()*esq).numpy(),
        fullbatch_injection=(eta.square()*meansq).numpy(),noise_injection=(eta.square()*(esq-meansq)).numpy(),
        expected_linear=(-2*eta*a).numpy(),q=qphi.mean(0).numpy(),self_variance_rate=selfrate.numpy(),
        variance=norm2(w)[2].numpy(),zmean=z.mean(0).numpy())

def replay(arm,start,steps=100000,tag=''):
    began=time.monotonic();RAW.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    st,path=restored(arm,start);net=st['net'];eta=st['lr'].double()[None,:,None]
    logs=[]
    for seed in range(10):
        with np.load(OLD/'logs'/f'{arm}_seed{seed}.npz') as d:
            logs.append({k:d[k] for k in ['step','layer1_w_norm','layer1_v_unit','layer1_w_free_step','layer1_w_free']})
    nt=(steps+9999)//10000
    sums=torch.zeros((nt,4,3,10,100,len(METRICS)),dtype=torch.float64)
    source_sums=torch.zeros((nt,4,10,5),dtype=torch.float64) # S,H,B,C,A_centered
    endpoints=[];phase_endpoints=[];probe_rows=[];probe_vals={};checks=[]
    max_step_closure=0.;max_formula_error=0.;max_pred_round=0.;max_SHBC_relative=0.
    sign_agree=0;sign_total=0
    w0=net.Ws[0].double();initial=norm2(w0);endpoints.append(initial.numpy())
    phase_endpoints.append(initial.numpy())

    def check_logs(t):
        w=net.Ws[0].double();wn=w.norm(dim=-1).float().numpy();vv=net.v.detach().numpy()
        for seed,d in enumerate(logs):
            ix=int(np.flatnonzero(d['step']==t)[0])
            assert np.array_equal(wn[seed],d['layer1_w_norm'][ix]),(arm,t,seed,'norm')
            assert np.array_equal(vv[seed],d['layer1_v_unit'][ix]),(arm,t,seed,'v')
            if t%10000==0:
                j=int(np.flatnonzero(d['layer1_w_free_step']==t)[0])
                assert np.array_equal(net.Ws[0][seed,:,15:].numpy(),d['layer1_w_free'][j]),(arm,t,seed,'free')
        checks.append(t)

    check_logs(start)
    for step in range(steps):
        task=step//10000;phase=step%10000;binidx=int(np.searchsorted(PHASES,phase,side='right')-1)
        raw=st['env'].step();y=st['teacher'](raw)
        ins,pres,acts,pred=forward_gate(st,raw);delta=pred-y
        grads=grads_centered_elu(net,ins,pres,acts,delta)
        if phase in PROBES:
            pr=probe(st);probe_rows.append([start+step,task,phase])
            for k,v in pr.items():probe_vals.setdefault(k,[]).append(v)
        w=net.Ws[0].double();g=grads[0][0].double();b=net.bs[0].double();v=net.v.double()
        x=ins[0].double();z=pres[0].double();phi=acts[0].double()
        gate=net.act_grad(pres[0],acts[0]).double();u=(2*delta[:,None]*net.v).double()
        q=grads[1][0].double()
        avec=inner(w,g);predlinear=-2*eta*avec;predinj=eta.square()*norm2(g)
        # Nonhomogeneity H includes only native-arithmetic rounding for Leaky.
        ss=(u*phi).sum(-1);hh=(u*(z*gate-phi)).sum(-1)
        bb=(b*u*gate).sum(-1);cc=x.sum(-1)*(w.mean(-1)*u*gate).sum(-1)
        err=abs(ss+hh-bb-cc-avec[1].sum(-1))
        scale=1+ss.abs()+hh.abs()+bb.abs()+cc.abs()+avec[1].abs().sum(-1)
        max_SHBC_relative=max(max_SHBC_relative,float((err/scale).max()))
        source_sums[task,binidx]+=torch.stack([ss,hh,bb,cc,avec[1].sum(-1)],-1)
        # Exact MSE split of the free-width directional contribution up to
        # native float32 gradient-product rounding, which remains in rest.
        selfq=2*v.square()*phi*gate
        psif=(w[...,15:]*x[:,None,15:]).sum(-1)
        selflin=-eta[0]*selfq*psif/2
        net.sgd_step_layers(st['lr'],*grads)
        wp=net.Ws[0].double();dw=wp-w;lin=2*inner(w,dw);inj=norm2(dw)
        before=norm2(w);actual=norm2(wp)-before
        max_step_closure=max(max_step_closure,float(abs(actual-lin-inj).max()))
        max_pred_round=max(max_pred_round,float(abs(actual-predlinear-predinj).max()))
        ok=abs(actual)>1e-10*torch.maximum(before,torch.ones_like(before))
        sign_agree+=int((((predlinear+predinj)<0)==(actual<0))[ok].sum());sign_total+=int(ok.sum())
        s=sums[task,binidx];s[...,0]+=torch.clamp(-lin,min=0);s[...,1]+=torch.clamp(lin,min=0)
        s[...,2]+=inj;s[...,3]+=predlinear;s[...,4]+=predinj
        s[2,...,5]+=selflin;s[2,...,6]+=predlinear[2]-selflin
        if phase+1 in PHASES[1:]:phase_endpoints.append(norm2(wp).numpy())
        if (step+1)%1000==0:check_logs(start+step+1)
        if (step+1)%10000==0:
            endpoints.append(norm2(wp).numpy())
            print(arm,start,'tasks',(step+1)//10000,'seconds',round(time.monotonic()-began,1),flush=True)
    ep=np.array(endpoints);su=sums.numpy();actual_window=norm2(net.Ws[0].double()).numpy()-initial.numpy()
    ledger=(su[...,1]+su[...,2]-su[...,0]).sum((0,1))
    max_window_error=float(abs(actual_window-ledger).max())
    assert max_window_error<1e-8 and max_step_closure<1e-8
    assert max_SHBC_relative<1e-5,max_SHBC_relative
    prefix=f'{tag}{arm}_{start}'
    dest=RAW/f'{prefix}.npz'
    np.savez_compressed(dest,budget=su,endpoints=ep,phase_endpoints=np.array(phase_endpoints),source_terms=source_sums.numpy(),
        probe_index=np.array(probe_rows),**{'probe_'+k:np.array(v) for k,v in probe_vals.items()})
    meta=dict(arm=arm,a=float(st['act_alpha']),start=start,steps=steps,seeds=list(range(10)),
        source_checkpoint=str(path),source_sha256=sha(path),archive=str(dest),archive_sha256=sha(dest),
        phases=PHASES,geometries=GEOMS,metrics=METRICS,source_terms=['S','H','B','C','A_centered'],
        source_log_steps_checked=checks,source_log_bitwise_match=True,max_step_closure=max_step_closure,
        max_window_error=max_window_error,max_SHBC_relative=max_SHBC_relative,max_pred_float_round_error=max_pred_round,
        finite_sign_agreement=sign_agree/max(1,sign_total),finite_sign_count=sign_total,
        git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        elapsed_seconds=time.monotonic()-began)
    writejson(OUT/f'{prefix}.json',meta);return meta

def preflight():
    arm=ARMS[4]
    a,_=restored(arm,0);b,_=restored(arm,0)
    batch=a['env'].segment(10000)
    singles=torch.stack([b['env'].step() for _ in range(10000)])
    assert torch.equal(batch,singles) and torch.equal(a['env'].gen.get_state(),b['env'].gen.get_state())
    result=replay(arm,1000000,steps=2000,tag='preflight_')
    result['segment_stream_bitwise_match']=True
    writejson(OUT/'preflight.json',result);print(json.dumps(result,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','run']);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if a.stage=='preflight':preflight();return
    jobs=[(arm,start) for start in [0,1000000] for arm in ARMS]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs=[pool.submit(replay,*j) for j in jobs]
        for f in concurrent.futures.as_completed(fs):
            r=f.result();print('DONE',r['arm'],r['start'],round(r['elapsed_seconds'],1),flush=True)

if __name__=='__main__':main()

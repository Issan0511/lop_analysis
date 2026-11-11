#!/usr/bin/env python3
"""Registered paired probes; see specs/spec_demand_allocation_1111.md."""
import argparse, concurrent.futures, csv, hashlib, json, math, os, subprocess, time
from pathlib import Path
import numpy as np
import torch
import stage1_ctrl_1111 as E

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
SAVED = Path('/home/issan/Projects/obsidian-research-data/stage1_ctrl_1111/results/stage1_ctrl_1111/raw')
TIMES = (0, 1, 10, 50, 200, 500, 1000, 2000, 4000)
LR = .001

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def rms(v): return float(v.double().square().mean().sqrt()) if v.numel() else float('nan')
def avg(v): return float(v.double().mean()) if v.numel() else float('nan')
def cosine(a,b):
    a,b=a.double().flatten(),b.double().flatten()
    den=a.norm()*b.norm()
    return float(a.dot(b)/den) if den>0 else float('nan')

def write_csv(path, rows):
    if not rows: return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with open(path,'w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def fields(P,X,y,act):
    p=[q.detach() for q in P]
    z=(X@p[0].T+p[1]).detach().requires_grad_(True)
    a=E.activ(z,act)
    G=torch.autograd.grad(a.sum(),z)[0].detach()
    z,a=z.detach(),a.detach()
    ell=a@p[2].T+p[3]
    prob=ell.softmax(-1)
    e=prob-torch.nn.functional.one_hot(y,10)
    D=e@p[2]
    return dict(z=z,a=a,G=G,ell=ell,p=prob,e=e,D=D,delta=D*G,
                ce=float((ell.logsumexp(-1)-ell.gather(1,y[:,None]).squeeze(1)).mean()),
                acc=float((ell.argmax(1)==y).float().mean()))

def masks(F,F0):
    z,G,D=F['z'],F['G'],F['D']; z0=F0['z']
    closed=(z<=0).all(0)[None,:].expand_as(z)
    return dict(all=torch.ones_like(z,dtype=torch.bool),open=z>0,band=(z<=0)&(z>-3),deep=z<=-3,
        closed_units=closed,zero_deriv=G==0,negative_deriv=G<0,
        tiny_nonzero=(G.abs()>0)&(G.abs()<1e-6),active_deriv=G.abs()>=1e-6,
        initial_open=z0>0,initial_band=(z0<=0)&(z0>-3),initial_deep=z0<=-3,
        closed_up=closed&(D<0),zero_up=(G==0)&(D<0))

def snapshot(P,opt,path,**extras):
    sp,so=E.snapshot(P,opt)
    torch.save(dict(P=sp,opt=so,**extras),path)

def train_step(P,opt,X,Y,idx,act,frozen=()):
    xb,yb=X[idx],Y[idx]
    z=xb@P[0].T+P[1];a=E.activ(z,act);ell=a@P[2].T+P[3]
    loss=(ell.logsumexp(-1)-(ell*yb).sum(-1)).mean()
    opt.zero_grad(set_to_none=True);loss.backward()
    for j in frozen: P[j].grad=None
    opt.step()

def stats(P,F,F0,meta,rows,unit):
    z,a,G,D=F['z'],F['a'],F['G'],F['D']; d=F['delta']; sd=z.std(0,unbiased=False)
    ms=masks(F,F0); V=P[2].detach(); Vc=V-V.mean(0)
    common=dict(**meta,ce=F['ce'],acc=F['acc'],zmean=avg(z),sd_median=float(sd.median()),
        r_median=float((z.mean(0)/sd.clamp_min(1e-30)).median()),top_median=float(z.max(0).values.median()),
        q99_median=float(torch.quantile(z,.99,dim=0).median()),
        open_fraction=avg((z>0).double()),units_per_input=avg((z>0).sum(1)),
        closed_unit_fraction=avg((z<=0).all(0).double()),
        small_derivative_unit_fraction=avg((G.abs()<1e-6).all(0).double()),
        zero_derivative_unit_fraction=avg((G==0).all(0).double()),
        Vnorm=float(Vc.norm()),Wnorm=float(P[0].norm()),hidden_bias_norm=float(P[1].norm()),
        output_bias_norm=float(P[3].norm()),logit_centered_rms=rms(F['ell']-F['ell'].mean(1,keepdim=True)),
        residual_rms=rms(F['e']),confidence=avg(F['p'].max(1).values),
        initial_closed_reopened=avg((z[:,(F0['z']<=0).all(0)]>0).any(0).double()),
        analytic_nonzero_implemented_zero=avg(((E.dphi64(z.double(),meta['act'])!=0)&(G==0)).double()))
    rows['state'].append(common)
    for name,m in ms.items():
        dm=D[m]; gm=G[m]
        rows['demand'].append(dict(**meta,group=name,n=int(m.sum()),Dmean=avg(dm),Drms=rms(dm),
            increase_fraction=avg((dm<0).double()),decrease_fraction=avg((dm>0).double()),
            increase_mag=avg((-dm).clamp_min(0)),decrease_mag=avg(dm.clamp_min(0)),
            transmitted_rms=rms(d[m]),Gabs=avg(gm.abs()),Gzero=avg((gm==0).double()),
            zmean=avg(z[m]),amean=avg(a[m]),dz_since_start=avg((z-F0['z'])[m]),
            da_since_start=avg((a-F0['a'])[m]),open_fraction=avg((z[m]>0).double())))
    key=f"{meta['phase']}_{meta['step']}"
    unit[key]=np.stack([z.mean(0).numpy(),sd.numpy(),z.max(0).values.numpy(),(z>0).float().mean(0).numpy(),
        D.mean(0).numpy(),(D<0).float().mean(0).numpy(),(D*(z>0)).mean(0).numpy(),G.abs().mean(0).numpy(),
        (G==0).float().mean(0).numpy(),Vc.norm(dim=0).numpy()])

def logits_allocation(P,Pn,F,Fn,meta,rows,kind):
    a,an=F['a'].double(),Fn['a'].double();V,Vn=P[2].detach().double(),Pn[2].detach().double()
    b,bn=P[3].detach().double(),Pn[3].detach().double(); da,dv=an-a,Vn-V
    parts=dict(body=da@V.T,V=a@dv.T,cross=da@dv.T,bias=(bn-b)[None,:].expand(len(a),-1))
    exact=(an@Vn.T+bn)-(a@V.T+b)
    err=float((sum(parts.values())-exact).abs().max())
    for name,q in parts.items():
        rows['allocation'].append(dict(**meta,kind=kind,part=name,descent=-float((F['e'].double()*q).sum(1).mean()),
            logit_rms=rms(q),account_error=err,exact_ce_decrease=F['ce']-Fn['ce']))
    assert err < 1e-9*max(1,float(exact.abs().max())),err

def noise(P,F,idx64,meta,rows):
    eb=F['e'][idx64].double();ab=F['a'][idx64].double()
    gv=torch.einsum('bnc,bni->bci',eb,ab)/idx64.shape[1]
    if meta['act']=='ELU':
        common=-eb.mean(1)[:,:,None].expand_as(gv); label='elu_minus_one'
    else:
        common=eb.mean(1)[:,:,None]*F['a'].double().mean(0)[None,None,:];label='mean_activation'
    rest=gv-common
    vc=common-common.mean(0);vr=rest-rest.mean(0);vg=gv-gv.mean(0)
    cv=float(vc.square().mean());rv=float(vr.square().mean());cov=2*float((vc*vr).mean());tot=float(vg.square().mean())
    rows['noise'].append(dict(**meta,decomposition=label,common_variance=cv,remainder_variance=rv,
        twice_covariance=cov,total_variance=tot,variance_identity_error=abs(tot-cv-rv-cov),
        common_over_total=cv/tot if tot else float('nan'),mean_gradient_rms=rms(gv.mean(0)),
        sample_gradient_rms=rms(gv),mean_bias_gradient_rms=rms(eb.mean((0,1))),
        bias_gradient_batch_rms=rms(eb.mean(1))))

def probe(P,opt,X,y,idx,act,frozen,F,F0,meta,rows,alternatives=False):
    snap=E.snapshot(P,opt);Pc,oc=E.restore(snap,'adam',LR)
    xb=X[idx];Y=torch.nn.functional.one_hot(y[idx],10).float()
    z=xb@Pc[0].T+Pc[1];a=E.activ(z,act);ell=a@Pc[2].T+Pc[3]
    z.retain_grad();a.retain_grad();ell.retain_grad()
    loss=(ell.logsumexp(-1)-(ell*Y).sum(-1)).mean();loss.backward()
    gradients=[p.grad.detach().clone() for p in Pc]
    batch_delta=z.grad.detach();batch_z=z.detach()
    rawW=batch_delta.T@xb;rawb=batch_delta.sum(0)
    raw_error=max(float((rawW-gradients[0]).abs().max()),float((rawb-gradients[1]).abs().max()))
    assert raw_error <2e-6*max(1,max(float(g.abs().max()) for g in gradients)),raw_error
    old=[{k:v.clone() if torch.is_tensor(v) else v for k,v in oc.state.get(p,{}).items()} for p in Pc]
    for j in frozen: Pc[j].grad=None
    oc.step()
    upd=[(q.detach().double()-p.detach().double()) for p,q in zip(P,Pc)]
    Fn=fields(Pc,X,y,act)
    dz=X.double()@upd[0].T+upd[1]
    dz_native=(Fn['z']-F['z']).double()
    err=float((dz-dz_native).abs().max()); scale=max(1,float(F['z'].abs().max()))
    assert err<2e-5*scale,(err,scale)
    configs=[('retained',upd)]
    if alternatives:
        for kind in ['adam','sgd']:
            Pa=E.new_params(snap[0]);oa=E.make_opt(kind,Pa,LR)
            for j,p in enumerate(Pa):p.grad=None if j in frozen else gradients[j].clone()
            oa.step()
            configs.append(('fresh_'+kind,[(q.detach().double()-p.detach().double()) for p,q in zip(P,Pa)]))
    for label,u in configs:
        uz=X.double()@u[0].T+u[1]
        rows['step'].append(dict(**meta,optimizer=label,body_update_rms=rms(torch.cat([u[0].flatten(),u[1]])),
            V_update_rms=rms(u[2]),bias_update_rms=rms(u[3]),dz_rms=rms(uz),dz_mean=avg(uz),
            hidden_raw_rms=rms(torch.cat([gradients[0].flatten(),gradients[1]])),V_raw_rms=rms(gradients[2]),
            V_step_cosine=cosine(P[2]-P[2].mean(0),u[2]-u[2].mean(0)),
            V_grad_step_cosine=cosine(-gradients[2],u[2]),
            hidden_grad_step_cosine=cosine(-gradients[0],u[0]),
            W_descent=-float((gradients[0].double()*u[0]).sum()),
            hidden_bias_descent=-float((gradients[1].double()*u[1]).sum()),
            V_descent=-float((gradients[2].double()*u[2]).sum()),
            output_bias_descent=-float((gradients[3].double()*u[3]).sum()),
            hidden_m_rms=rms(old[0].get('exp_avg',torch.zeros_like(P[0]))),
            hidden_v_rms=rms(old[0].get('exp_avg_sq',torch.zeros_like(P[0]))),
            V_m_rms=rms(old[2].get('exp_avg',torch.zeros_like(P[2]))),
            V_v_rms=rms(old[2].get('exp_avg_sq',torch.zeros_like(P[2]))),
            raw_formula_error=raw_error,z_rounding_error=err))
    # Exact conditional source accounting: the actual next denominator is shared.
    sources={n:[] for n in ['memory','open','band','deep']}
    bs=dict(open=batch_z>0,band=(batch_z<=0)&(batch_z>-3),deep=batch_z<=-3)
    for j in [0,1]:
        if j in frozen:
            for k in sources:sources[k].append(torch.zeros_like(upd[j]))
            continue
        st=oc.state[Pc[j]];t=float(st['step'])
        denom=(st['exp_avg_sq'].double()/(1-.999**t)).sqrt()+1e-8
        coef=-LR/(1-.9**t)/denom
        mem=old[j].get('exp_avg',torch.zeros_like(P[j])).double()
        sources['memory'].append(coef*.9*mem)
        for name,m in bs.items():
            g=(batch_delta*m).T@xb if j==0 else (batch_delta*m).sum(0)
            sources[name].append(coef*.1*g.double())
    rest=[upd[j]-sum(sources[k][j] for k in sources) for j in [0,1]]
    rounderr=max(float(v.abs().max()) for v in rest)
    assert rounderr<2e-6*max(1,float(P[0].abs().max()),float(P[1].abs().max())),rounderr
    sources['rounding']=rest
    mm=masks(F,F0)
    for source,us in sources.items():
        dzs=X.double()@us[0].T+us[1]
        for rec in ['all','open','band','deep','initial_open','initial_deep','closed_units','zero_deriv','zero_up']:
            m=mm[rec]
            rows['source'].append(dict(**meta,source=source,receiver=rec,n=int(m.sum()),dz_mean=avg(dzs[m]),dz_rms=rms(dzs[m]),
                bias_dz_mean=avg(us[1][None,:].expand_as(dzs)[m]),step_account_error=rounderr))
    for group,m in mm.items():
        rows['response'].append(dict(**meta,group=group,n=int(m.sum()),dz_mean=avg(dz[m]),dz_rms=rms(dz[m]),
            dz_up_fraction=avg((dz[m]>0).double()),native_dz_zero_fraction=avg((dz_native[m]==0).double()),
            da_mean=avg((Fn['a']-F['a'])[m]),
            demand_dz_agree=avg(((-F['D'][m]).double()*dz[m]>0).double()),
            demand_da_agree=avg(((-F['D'][m])*(Fn['a']-F['a'])[m]>0).double())))
    logits_allocation(P,Pc,F,Fn,meta,rows,'one_step')

def immediate(base,X,y,yold,act,Vinit,ctx,meta,rows):
    originals={lab:fields(base[0],X,yy,act) for lab,yy in [('new',y),('old',yold)]}
    for arm in ['S','Q','Craw','Cmean']:
        P,opt=E.restore(base,'adam',LR)
        if arm=='Q':E.intervene('Q',P,opt,ctx,E.own('unused',meta['seed']),Vinit,'adam')
        if arm in ['Craw','Cmean']:
            Vo=P[2].detach().clone();abar=E.mean_act(P,ctx)
            with torch.no_grad():P[2].mul_(.1)
            if arm=='Cmean':E.fix_bias(P,Vo,abar)
        for lab,yy in [('new',y),('old',yold),('uniform',y)]:
            Fs=originals['old' if lab=='old' else 'new'];F=fields(P,X,yy,act)
            es=Fs['p']-.1 if lab=='uniform' else Fs['e'];en=F['p']-.1 if lab=='uniform' else F['e']
            Vs=base[0][2];V=P[2].detach();Ds=es@Vs;Dn=en@V
            direct=es@(V-Vs);residual=(en-es)@V
            err=rms((Dn-Ds)-direct-residual)
            for group,m in masks(F,Fs).items():
                if group not in ['all','open','band','deep','closed_units']:continue
                rows['instant'].append(dict(**meta,arm=arm,labels=lab,group=group,n=int(m.sum()),Drms=rms(Dn[m]),
                    Dmean=avg(Dn[m]),increase_fraction=avg((Dn[m]<0).double()),increase_mag=avg((-Dn[m]).clamp_min(0)),
                    decrease_mag=avg(Dn[m].clamp_min(0)),direct_term_rms=rms(direct[m]),residual_term_rms=rms(residual[m]),
                    sign_changed_fraction=avg((Dn[m]*Ds[m]<0).double()),demand_cosine=cosine(Dn[m],Ds[m]),decomp_error=err,
                    expected_mean_dz_sgd=-LR*avg((en@V)*F['G']*(X@X.mean(0)+1)[:,None]),
                    confidence=avg(F['p'].max(1).values),logit_rms=rms(F['ell']-F['ell'].mean(1,keepdim=True))))

def run_fit(P,opt,X,y,order,act,frozen,F0,meta,rows,out,unit,phase='fit'):
    Y=torch.nn.functional.one_hot(y,10).float()
    idx64=torch.randint(len(X),(64,16),generator=E.own(f"noise|{meta['task']}|{phase}",meta['seed']))
    startP=[q.detach().clone() for q in P];startF=fields(P,X,y,act)
    times=TIMES if phase=='fit' else (0,200,4000)
    for it in range(4001):
        if it in times:
            m=dict(**meta,phase=phase,step=it);F=fields(P,X,y,act)
            stats(P,F,F0,m,rows,unit)
            probe(P,opt,X,y,order[it%4000],act,frozen,F,F0,m,rows,alternatives=(it==0))
            if it in (0,500,4000):noise(P,F,idx64,m,rows)
            if it==4000:logits_allocation(startP,P,startF,F,m,rows,'whole_task')
        if it<4000:train_step(P,opt,X,Y,order[it],act,frozen)
    snapshot(P,opt,out/f"{meta['arm']}_t{meta['task']}_{phase}_end.pt",y=y)

def run_job(act,seed,outstr):
    os.nice(10)
    start=time.monotonic();out=Path(outstr)/f'{act}_s{seed}';out.mkdir(parents=True,exist_ok=False)
    rows={k:[] for k in ['state','demand','step','source','response','allocation','noise','instant']}
    X,subset=E.load_X(seed);ctx=E.Ctx(X,act)
    ini=E.initial(seed,(784,100,10));P=E.new_params(ini);opt=E.make_opt('adam',P,LR)
    gl,gb=E.stream('env_labels_0913',seed),E.stream('env_batch_0913',seed)
    ys=[];orders=[]
    for t in range(26):
        ys.append(torch.randint(10,(1200,),generator=gl));orders.append(E.order_for(gb,4000))
    savedpath=SAVED/f'states_adam_lr0.001_{act}_s{seed}.npz';saved=np.load(savedpath)
    assert np.array_equal(subset,saved['subset'])
    checks=[]
    for task in range(1,26):
        y,order=ys[task-1],orders[task-1]
        assert np.array_equal(y.numpy(),saved['y'][task-1])
        if task in (2,10,25):
            base=E.snapshot(P,opt);F0=fields(P,X,y,act)
            snapshot(P,opt,out/f'baseline_before_t{task}.pt',y=y,yold=ys[task-2],order=order,subset=subset)
            meta=dict(act=act,seed=seed,task=task)
            immediate(base,X,y,ys[task-2],act,ini[2],ctx,meta,rows)
            for arm in ['S','Q','QV','QF','QA']+(['QH'] if act=='ELU' else []):
                Pa,oa=E.restore(base,'adam',LR)
                if arm!='S':E.intervene('Q',Pa,oa,ctx,E.own('unused',seed),ini[2],'adam')
                if arm=='QA':
                    for j in (0,1):oa.state.pop(Pa[j],None)
                if arm=='QH':
                    for j in (2,3):oa.state.pop(Pa[j],None)
                frozen=(2,) if arm=='QV' else (0,1) if arm=='QF' else ()
                unit={};am=dict(**meta,arm=arm)
                run_fit(Pa,oa,X,y,order,act,frozen,F0,am,rows,out,unit)
                if arm=='S':
                    err=E.maxabs(Pa,[saved[f'end{j}'][task-1] for j in range(4)])
                    assert err==0,('branch bit equality',act,seed,task,err)
                if arm in ['S','Q']:
                    Fnext=fields(Pa,X,ys[task],act)
                    run_fit(Pa,oa,X,ys[task],orders[task],act,(),Fnext,am,rows,out,unit,'reload')
                    if arm=='S':
                        err=E.maxabs(Pa,[saved[f'end{j}'][task] for j in range(4)])
                        assert err==0,('reload bit equality',act,seed,task,err)
                np.savez_compressed(out/f'{arm}_t{task}_units.npz',**unit)
            for kind,rs in rows.items():write_csv(out/f'{kind}.csv',rs)
            print(f'{act} seed {seed}: fork task {task} complete, {time.monotonic()-start:.1f}s',flush=True)
        Y=torch.nn.functional.one_hot(y,10).float()
        for idx in order:train_step(P,opt,X,Y,idx,act)
        err=E.maxabs(P,[saved[f'end{j}'][task-1] for j in range(4)])
        checks.append(err);assert err==0,('baseline bit equality',act,seed,task,err)
    meta=dict(act=act,seed=seed,git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256=sha(__file__),engine_sha256=sha(E.__file__),input_sha256=sha(E.DATA/'train-images-idx3-ubyte.gz'),
        saved_state_sha256=sha(savedpath),torch=torch.__version__,baseline_maxabs=checks,
        seconds=time.monotonic()-start,rows={k:len(v) for k,v in rows.items()})
    (out/'provenance.json').write_text(json.dumps(meta,indent=2));return meta

def sanity(out):
    torch.manual_seed(900);errs=[]
    X=torch.randn(19,7,dtype=torch.float64);y=torch.randint(0,10,(19,));Y=torch.nn.functional.one_hot(y,10)
    for act in ['R','ELU','SILU','GELU','LR']:
        P=[torch.randn(*shape,dtype=torch.float64,requires_grad=True) for shape in [(5,7),(5,),(10,5),(10,)]]
        z=X@P[0].T+P[1];a=E.activ(z,act);ell=a@P[2].T+P[3]
        loss=(ell.logsumexp(-1)-(ell*Y).sum(-1)).mean()
        g=torch.autograd.grad(loss,P);e=ell.softmax(-1)-Y;d=(e@P[2])*E.dphi64(z,act)
        manual=[d.T@X/len(X),d.mean(0),e.T@a/len(X),e.mean(0)]
        err=max(float((u-v).abs().max()) for u,v in zip(g,manual));assert err<1e-9
        fd=[]
        for j in range(4):
            direction=torch.randn_like(P[j]);direction/=direction.norm();eps=1e-5
            def value(sign):
                Q=[p.detach().clone() for p in P];Q[j]+=sign*eps*direction
                l=E.activ(X@Q[0].T+Q[1],act)@Q[2].T+Q[3]
                return (l.logsumexp(-1)-(l*Y).sum(-1)).mean()
            fd.append(abs(float((value(1)-value(-1))/(2*eps)-(g[j]*direction).sum())))
        assert max(fd)<1e-7
        errs.append(dict(act=act,gradient_error=err,finite_difference_error=max(fd)))
    # Adam with stale, matched, and fresh moments: independent scalar formula.
    p=torch.tensor([1.,2.,3.],dtype=torch.float64,requires_grad=True)
    opt=torch.optim.Adam([p],lr=.001,betas=(.9,.999),eps=1e-8)
    m=torch.tensor([.03,-.02,.0],dtype=torch.float64);v=torch.tensor([.01,.004,.00001],dtype=torch.float64)
    opt.state[p]={'step':torch.tensor(71.),'exp_avg':m.clone(),'exp_avg_sq':v.clone()}
    grad=torch.tensor([.0003,.2,-.0001],dtype=torch.float64);old=p.detach().clone();p.grad=grad;opt.step()
    manual=-.001*((.9*m+.1*grad)/(1-.9**72))/(((.999*v+.001*grad.square())/(1-.999**72)).sqrt()+1e-8)
    err=float((p.detach()-old-manual).abs().max());assert err<1e-12
    z=torch.tensor([-10.,-16.,-17.,-24.],requires_grad=True);a=E.activ(z,'ELU');g=torch.autograd.grad(a.sum(),z)[0]
    floor=dict(z=z.tolist(),implemented=g.tolist(),analytic=torch.exp(z.double()).tolist())
    # Complete production diagnostic smoke on synthetic data, seed 900, 30 updates.
    X=torch.rand(60,784);P=E.new_params(E.initial(900,(784,100,10)));opt=E.make_opt('adam',P,LR)
    y=torch.randint(10,(60,));Y=torch.nn.functional.one_hot(y,10).float()
    for _ in range(30):train_step(P,opt,X,Y,torch.randint(60,(16,)),'ELU')
    F=fields(P,X,y,'ELU');rows={k:[] for k in ['state','demand','step','source','response','allocation','noise','instant']}
    meta=dict(act='ELU',seed=900,task=1,arm='smoke',phase='fit',step=0)
    stats(P,F,F,meta,rows,{})
    probe(P,opt,X,y,torch.arange(16),'ELU',(),F,F,meta,rows,True)
    noise(P,F,torch.randint(60,(64,16)),meta,rows)
    result=dict(gradient_checks=errs,adam_error=err,elu_floor=floor,smoke_rows={k:len(v) for k,v in rows.items()})
    Path(out).mkdir(parents=True,exist_ok=True);(Path(out)/'sanity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--workers',type=int,default=2)
    a=ap.parse_args()
    if a.smoke:return sanity(a.out)
    assert a.workers in [1,2]
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    (out/'launch.json').write_text(json.dumps(dict(command=os.sys.argv,git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2))
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(run_job,act,seed,str(out)) for act in ['R','ELU','SILU','GELU','LR'] for seed in [0,1,2]]
        results=[f.result() for f in concurrent.futures.as_completed(jobs)]
    (out/'complete.json').write_text(json.dumps(dict(jobs=results,seconds=time.monotonic()-start),indent=2))
    print(f'ALL 15 JOBS COMPLETE in {time.monotonic()-start:.1f}s',flush=True)

if __name__=='__main__':main()

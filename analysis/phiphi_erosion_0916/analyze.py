"""Posthoc, read-only analysis of archived trajectories; no training runs."""
from pathlib import Path
import argparse, hashlib, itertools, json, sys, subprocess, time
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = Path('/home/issan/Projects/claude/proj_004_drift')
ARCHIVE = Path('/home/issan/Projects/obsidian-research-data')
OUT = ROOT / 'results/phiphi_erosion_0916'
RAW = ARCHIVE / 'phiphi_erosion_0916'
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''): h.update(block)
    return h.hexdigest()

def savejson(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))

def corr(x,y):
    if np.std(x)==0 or np.std(y)==0:return 0.
    return float(np.corrcoef(x,y)[0,1])

def spearman(x,y):return corr(pd.Series(x).rank().to_numpy(),pd.Series(y).rank().to_numpy())

def ba(y,p):
    if not np.any(y) or np.all(y):return np.nan
    return .5*(np.mean(p[y])+np.mean(~p[~y]))

def auc(y,x):
    n1=y.sum();n0=len(y)-n1
    if n1==0 or n0==0:return np.nan
    r=pd.Series(x).rank().to_numpy()
    return float((r[y].sum()-n1*(n1+1)/2)/(n1*n0))

def threshold(x,y):
    """Best training balanced accuracy, both orientations, exact unique cutpoints."""
    order=np.argsort(x,kind='stable');xx=x[order];yy=y[order]
    ends=np.r_[np.flatnonzero(xx[1:]!=xx[:-1]),len(xx)-1]
    tpr=np.cumsum(yy)[ends]/max(1,yy.sum())
    fpr=np.cumsum(~yy)[ends]/max(1,(~yy).sum())
    low=.5*(tpr+1-fpr);i=int(np.argmax(np.abs(low-.5)))
    return float(xx[ends[i]]),bool(low[i]>=.5)

def predict(x,t,low):return (x<=t) if low else (x>t)

def algebra():
    # Centered five-bit input: exact finite-step norm/variance and self gradient.
    rng=np.random.default_rng(916);x=np.array(list(itertools.product([-.5,.5],repeat=5)))
    w=rng.normal(size=5);b=-.2;v=.7;a=.6;eta=1e-5
    z=x@w+b;p=np.where(z>0,z,a*z);g=np.where(z>0,1.,a);q=p*g
    dw=-2*eta*v*v*np.mean(q[:,None]*x,axis=0)
    radial=2*w@dw
    expected=-4*eta*v*v*np.mean(q*(z-z.mean()))
    variance_delta=np.var(x@(w+dw)+b)-np.var(z)
    variance_formula=(radial+dw@dw)/4
    eps=1e-6;fd=[]
    for j in range(5):
        e=np.eye(5)[j]*eps
        loss=lambda u:np.mean((v*np.where(x@u+b>0,x@u+b,a*(x@u+b)))**2)
        fd.append((loss(w+e)-loss(w-e))/(2*eps))
    result=dict(radial_identity_error=abs(radial-expected),
                variance_identity_error=abs(variance_delta-variance_formula),
                gradient_fd_maxerr=float(np.max(np.abs(np.array(fd)+dw/eta))))
    assert result['radial_identity_error']<1e-12
    assert result['variance_identity_error']<1e-12
    assert result['gradient_fd_maxerr']<1e-8
    # Same integrated phi*phi', different width contraction: all-negative leaky.
    examples=[]
    for scale in [.1,.2]:
        zz=x@(np.ones(5)*scale)-1
        qq=a*a*zz
        examples.append(dict(scale=scale,qmean=float(qq.mean()),variance=float(zz.var()),
                             self_width_rate=float(-v*v*np.mean(qq*(zz-zz.mean())))))
    assert abs(examples[0]['qmean']-examples[1]['qmean'])<1e-12
    assert abs(examples[1]['self_width_rate']/examples[0]['self_width_rate']-4)<1e-10
    result['same_integral_different_self_contraction']=examples
    savejson('algebra_checks.json',result)

def conda():
    bits=np.array(list(itertools.product([-.5,.5],repeat=5)))
    sources=[];chunks=[];dense=[];guards=[];endpoints=[]
    files=[(p,0.,0) for p in sorted((ORIGINAL/'results/scale_attractor_b_0906/logs').glob('*.npz'))]
    ar=ARCHIVE/'act_offset_review_0908/full/logs'
    for arm,c in [('LRoff0_1216',0.),('LRoffm0p5_1216',-.5),('LRoffp0p5_1216',.5)]:
        files += [(p,c,1) for p in sorted(ar.glob(arm+'_seed*.npz'))]
    for j,(path,c,aux) in enumerate(files):
        with np.load(path,allow_pickle=False) as d:
            a=float(d['act_alpha']);seed=int(d['seed']);arm=str(d['arm'])
            step=d['step'];ws=d['layer1_w_free_step'];ms=d['layer1_moment_step']
            w=d['layer1_w_free'].astype(float);zb=d['layer1_zbar'].astype(float)
            vn=d['layer1_v_unit'].astype(float);wn=d['layer1_w_norm'].astype(float)
            qm=d['layer1_m_phidphi'].astype(float);den=d['layer1_denom'].astype(float)
            zmin=d['layer1_zmin'].astype(float);zmax=d['layer1_zmax'].astype(float)
            idx=np.flatnonzero((ws>=200000)&(ws<5000000))
            si=np.searchsorted(step,ws[idx]);mi=np.searchsorted(ms,ws[idx])
            assert np.all(step[si]==ws[idx]) and np.all(ms[mi]==ws[idx])
            zc=np.einsum('tuj,pj->tup',w[idx],bits);z=zc+zb[si,:,None]
            phi=np.where(z>0,z,a*z)+c;gate=np.where(z>0,1.,a);q=phi*gate
            qmean=q.mean(2);var=np.mean(zc**2,2);cov=np.mean(q*zc,2)
            # Offset makes phi*phi' discontinuous at z=0. Float32 saved weights
            # cannot recover the branch at an exact kink; exclude these unit
            # snapshots instead of treating a .45/32 jump as a physical effect.
            safe=np.ones_like(var,dtype=bool) if c==0 else np.min(abs(z),axis=2)>2e-6
            delta=w[idx+1]-w[idx];rad=2*np.sum(w[idx]*delta,2);inj=np.sum(delta**2,2)
            dn=np.sum(w[idx+1]**2,2)-np.sum(w[idx]**2,2)
            check=dict(file=path.name,moment_maxerr=float(np.max(abs(qmean-qm[mi]))),
                       zmin_maxerr=float(np.max(abs(z.min(2)-zmin[si]))),
                       zmax_maxerr=float(np.max(abs(z.max(2)-zmax[si]))),
                       width_maxerr=float(np.max(abs(np.sqrt(var)-den[si]))),
                       norm_closure=float(np.max(abs(dn-rad-inj))))
            check['kink_ambiguous_fraction']=float(1-safe.mean())
            check['moment_maxerr_away_kink']=float(np.max(abs(qmean-qm[mi])[safe]))
            assert max(check[k] for k in ['moment_maxerr_away_kink','zmin_maxerr','zmax_maxerr','width_maxerr'])<2e-5,check
            assert check['norm_closure']<1e-10
            guards.append(check)
            shape=var.shape
            vals=dict(a=np.full(shape,a),c=np.full(shape,c),seed=np.full(shape,seed),aux=np.full(shape,aux),
                      task=np.broadcast_to(ws[idx,None]/10000,shape),unit=np.broadcast_to(np.arange(100),shape),
                      reconstruction_valid=safe,
                      q=qmean,qabs=np.mean(abs(q),2),qpos=np.mean(np.maximum(q,0),2),
                      qneg=np.mean(np.maximum(-q,0),2),qcov=cov,qcov_v2=cov*vn[si]**2,
                      var=var,vabs=abs(vn[si]),zmean=zb[si],dsigma2=dn/4,radial=rad/4,injection=inj/4,
                      dw2_total=wn[np.searchsorted(step,ws[idx+1])]**2-wn[si]**2,
                      cos=np.sum(w[idx]*delta,2)/(np.linalg.norm(w[idx],axis=2)*np.linalg.norm(delta,axis=2)+1e-30),
                      qnearzero=np.mean(q*(abs(z)<=.5),2),
                      qedges=np.mean(q*(abs(zc)>=np.quantile(abs(zc),.8,axis=2)[:,:,None]),2))
            chunks.append(pd.DataFrame({k:v.ravel().astype(np.float32) for k,v in vals.items()}))
            # Dense last-20-task observations: same measure, predict next 1000 steps.
            ii=np.flatnonzero((ms>=4800000)&(ms<5000000));ss=np.searchsorted(step,ms[ii]);nn=np.searchsorted(step,ms[ii]+1000)
            assert np.all(step[ss]==ms[ii]) and np.all(step[nn]==ms[ii]+1000)
            sh=qm[ii].shape
            dense.append(pd.DataFrame(dict(a=np.full(qm[ii].size,a),c=np.full(qm[ii].size,c),seed=np.full(qm[ii].size,seed),
                       aux=np.full(qm[ii].size,aux),q=qm[ii].ravel(),qabs_mean=abs(qm[ii]).ravel(),
                       dsigma2=(den[nn]**2-den[ss]**2).ravel(),var=(den[ss]**2).ravel(),
                       phase=np.broadcast_to((ms[ii]%10000)[:,None]/1000,sh).ravel())))
            for label,start,end in [('initial_to_final',0,len(ws)-1),('late_100tasks',len(ws)-101,len(ws)-1)]:
                endpoints.append(dict(a=a,c=c,seed=seed,aux=aux,window=label,
                     median_width_ratio=float(np.median(np.linalg.norm(w[end],axis=1)/np.maximum(np.linalg.norm(w[start],axis=1),1e-30))),
                     median_weight_ratio=float(np.median(wn[np.searchsorted(step,ws[end])]/np.maximum(wn[np.searchsorted(step,ws[start])],1e-30)))))
        sources.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
        if j%10==9:print('condA files',j+1,'/',len(files),flush=True)
    frame=pd.concat(chunks,ignore_index=True);dd=pd.concat(dense,ignore_index=True)
    np.savez_compressed(RAW/'conda_units.npz',**{k:frame[k].to_numpy() for k in frame})
    np.savez_compressed(RAW/'conda_dense.npz',**{k:dd[k].to_numpy() for k in dd})
    pd.DataFrame(endpoints).to_csv(OUT/'conda_endpoint_ratios.csv',index=False)
    savejson('conda_sources.json',sources);savejson('conda_checks.json',guards)
    print('condA rows',len(frame),'dense',len(dd),flush=True)

def pm():
    from src import pmnist_boundary_host_0908 as H
    H.DATA_DIR=ORIGINAL/'data/mnist'
    mnist=H.Mnist(torch.device('cpu'));chunks=[];sources=[];checks=[]
    for arm,seed in itertools.product(['LR','SNA'],range(3)):
        path=ARCHIVE/f'boundary_tasks20_40_0909/source/{arm}_none_s{seed}_states.pt'
        saved=torch.load(path,weights_only=False,map_location='cpu');px=mnist.test_x[saved['probe_indices']].double()
        for raw in saved['boundaries']:
            xp=px[:,raw['perm']];states={};oldvar=None
            for step,key in [(0,'before'),(20,'after_20'),(300,'after_300'),(625,'after_625')]:
                ss=raw[key]['state'];w,b=ss['params'][:2];w=w.double();b=b.double()
                z=(xp@w.T+b).numpy();W=w.numpy();wc=W-W.mean(1,keepdims=True)
                mean=z.mean(0);var=z.var(0);center=z-mean
                quant=np.quantile(z,[.05,.95],axis=0)
                if arm=='LR':q=np.where(z>0,z,.1*z)*np.where(z>0,1.,.1)
                else:
                    alpha=ss['alpha'][0]
                    if torch.is_tensor(alpha):alpha=alpha.numpy()
                    q=(z+np.sin(alpha*z)**2/alpha)*(1+np.sin(2*alpha*z))
                states[step]=dict(w=wc,z=z,mean=mean,var=var,upper=quant[1]-mean,lower=mean-quant[0],
                                  q=q.mean(0),qabs=abs(q).mean(0),qcov=(q*center).mean(0))
                # Stored a1 is independently generated from the same saved weights.
                a1=raw[key]['a1'].double().numpy()
                phi=np.where(z>0,z,.1*z) if arm=='LR' else z+np.sin(alpha*z)**2/alpha
                err=float(np.max(abs(a1-phi)));assert err<2e-4,(arm,seed,raw['task'],step,err)
                checks.append(dict(arm=arm,seed=seed,task=raw['task'],step=step,a1_maxerr=err))
            for start,end in [(0,20),(20,300),(300,625),(0,625)]:
                s,e=states[start],states[end];dw=e['w']-s['w'];dz=e['z']-s['z']
                radial=2*np.sum(s['w']*dw,1);inj=np.sum(dw**2,1)
                dwn=np.sum(e['w']**2,1)-np.sum(s['w']**2,1)
                cov=2*np.mean((s['z']-s['mean'])*(dz-dz.mean(0)),0);vi=dz.var(0)
                dvar=e['var']-s['var'];assert np.max(abs(dvar-cov-vi))<1e-10
                assert np.max(abs(dwn-radial-inj))<1e-9
                du=e['upper']-s['upper'];dl=e['lower']-s['lower']
                chunks.append(pd.DataFrame(dict(arm=arm,seed=seed,task=raw['task'],unit=np.arange(100),start=start,end=end,
                    norm2=np.sum(s['w']**2,1),norm_delta=dwn,radial=radial,injection=inj,
                    cos=np.sum(s['w']*dw,1)/(np.linalg.norm(s['w'],axis=1)*np.linalg.norm(dw,axis=1)+1e-30),
                    variance=s['var'],dsigma2=dvar,var_linear=cov,var_quadratic=vi,
                    dmean=e['mean']-s['mean'],dupper=du,dlower=dl,
                    asymmetry=abs(du-dl)/(abs(du)+abs(dl)+1e-30),q=s['q'],qabs=s['qabs'],qcov=s['qcov'])))
        sources.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
        print('PM',arm,seed,'done',flush=True)
    df=pd.concat(chunks,ignore_index=True);df.to_csv(RAW/'pm_units.csv.gz',index=False)
    rows=[]
    for (arm,seed,start,end),g in df.groupby(['arm','seed','start','end']):
        grow=g.dsigma2>1e-6*g.variance.clip(lower=1e-4)
        row=dict(arm=arm,seed=int(seed),start=int(start),end=int(end),n=len(g),
                 growth_fraction=float(grow.mean()),norm_growth_fraction=float((g.norm_delta>0).mean()),
                 median_abs_cos=float(g.cos.abs().median()),
                 symmetry_among_growth=float(g.loc[grow,'asymmetry'].mean()),
                 both_sides_grow_fraction=float(((g.dupper>0)&(g.dlower>0)).mean()))
        for k in ['norm_delta','radial','injection','dsigma2','var_linear','var_quadratic','dmean','dupper','dlower']:
            row[k]=float(g[k].mean())
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'pm_phase_by_seed.csv',index=False)
    savejson('pm_sources.json',sources);savejson('pm_checks.json',checks)

def regression_train(X,y):
    # Balanced logistic regression via Newton updates, training data only.
    X=np.c_[np.ones(len(X)),X];beta=np.zeros(X.shape[1]);weights=np.where(y,.5/max(y.mean(),1e-9),.5/max(1-y.mean(),1e-9))
    for _ in range(30):
        z=np.clip(X@beta,-30,30);p=1/(1+np.exp(-z));h=weights*p*(1-p)
        grad=X.T@(weights*(p-y))/len(y)+1e-3*beta
        hes=X.T@(h[:,None]*X)/len(y)+1e-3*np.eye(X.shape[1])
        step=np.linalg.solve(hes,grad);beta-=step
        if np.linalg.norm(step)<1e-6:break
    return beta

def evaluate():
    with np.load(RAW/'conda_units.npz') as d:allf=pd.DataFrame(dict(d))
    valid=(abs(allf.dsigma2)>1e-6*np.maximum(allf['var'],1e-4))&(allf.reconstruction_valid>0)
    f=allf.loc[valid].copy();f['shrink']=f.dsigma2<0
    primary=f[f.aux==0];train=primary.seed<5;test=~train
    models={};metrics=[];fits=[];transfers=[]
    features=['q','qabs','qpos','qneg','qcov','qcov_v2','var','vabs','qnearzero','qedges']
    for col in features:
        t,lo=threshold(primary.loc[train,col].to_numpy(),primary.loc[train,'shrink'].to_numpy())
        fits.append(dict(feature=col,threshold=t,shrink_if='le' if lo else 'gt'))
        models[col]=(t,lo)
        for (aux,a,c,seed),g in f[f.seed>=5].groupby(['aux','a','c','seed']):
            y=g.shrink.to_numpy();x=g[col].to_numpy();pred=predict(x,t,lo)
            metrics.append(dict(feature=col,aux=aux,a=a,c=c,seed=seed,n=len(g),ba=ba(y,pred),
                                auc=auc(y,-x if lo else x),spearman_erosion=spearman(x,-g.dsigma2.to_numpy())))
    # Exact held-out-a scalar threshold, using separate seeds as well.
    for a in sorted(primary.a.unique()):
        tr=primary[(primary.a!=a)&(primary.seed<5)];te=primary[(primary.a==a)&(primary.seed>=5)]
        for col in ['q','qabs','qcov','qcov_v2']:
            t,lo=threshold(tr[col].to_numpy(),tr.shrink.to_numpy())
            for seed,g in te.groupby('seed'):
                transfers.append(dict(a=a,seed=seed,feature=col,threshold=t,
                     ba=ba(g.shrink.to_numpy(),predict(g[col].to_numpy(),t,lo))))
    # This is a predictive comparison, not a mechanism or a causal alternative.
    cols=['q','qabs','qcov','qcov_v2','var','vabs','a','zmean']
    rng=np.random.default_rng(916);tr=primary[train]
    ids=rng.choice(len(tr),min(len(tr),150000),replace=False);tr=tr.iloc[ids]
    X=np.sign(tr[cols].to_numpy())*np.log1p(abs(tr[cols].to_numpy()))
    mu=X.mean(0);sd=X.std(0);sd[sd<1e-8]=1;X=(X-mu)/sd
    beta=regression_train(X,tr.shrink.to_numpy().astype(float))
    for (aux,a,c,seed),g in f[f.seed>=5].groupby(['aux','a','c','seed']):
        x=np.sign(g[cols].to_numpy())*np.log1p(abs(g[cols].to_numpy()));x=(x-mu)/sd
        score=np.c_[np.ones(len(x)),x]@beta;y=g.shrink.to_numpy()
        metrics.append(dict(feature='multivariable_logistic',aux=aux,a=a,c=c,seed=seed,n=len(g),
                            ba=ba(y,score>0),auc=auc(y,score),spearman_erosion=spearman(score,-g.dsigma2.to_numpy())))
    pd.DataFrame(metrics).to_csv(OUT/'threshold_by_seed_and_slope.csv',index=False)
    pd.DataFrame(fits).to_csv(OUT/'threshold_fits.csv',index=False)
    pd.DataFrame(transfers).to_csv(OUT/'leave_one_slope_out.csv',index=False)
    savejson('prediction_scope.json',dict(rows=len(allf),valid_rows=len(f),excluded_fraction=float(1-valid.mean()),
        primary_train_rows=int(train.sum()),primary_test_rows=int(test.sum()),seed_split='train0..4 test5..9',
        logistic_columns=cols,logistic_coefficients=beta.tolist()))
    # Intratask late-window scalar threshold, independently trained on same train seeds.
    with np.load(RAW/'conda_dense.npz') as d:dd=pd.DataFrame(dict(d))
    dd=dd[(abs(dd.dsigma2)>1e-6*np.maximum(dd['var'],1e-4))&(dd.aux==0)]
    dd['shrink']=dd.dsigma2<0;rows=[]
    for col in ['q','qabs_mean']:
        tr=dd[dd.seed<5];t,lo=threshold(tr[col].to_numpy(),tr.shrink.to_numpy())
        for (a,seed,phase),g in dd[dd.seed>=5].groupby(['a','seed','phase']):
            rows.append(dict(feature=col,a=a,seed=seed,phase=phase,n=len(g),
                ba=ba(g.shrink.to_numpy(),predict(g[col].to_numpy(),t,lo))))
    pd.DataFrame(rows).to_csv(OUT/'dense_thresholds.csv',index=False)
    # Aggregation at seed level, no unit-as-independent p-values.
    agg=f.groupby(['aux','a','c','seed']).agg(n=('dsigma2','size'),shrink_fraction=('shrink','mean'),
        q=('q','mean'),qabs=('qabs','mean'),qcov=('qcov','mean'),qcov_v2=('qcov_v2','mean'),
        radial=('radial','mean'),injection=('injection','mean'),dsigma2=('dsigma2','mean'),cos=('cos','mean')).reset_index()
    agg.to_csv(OUT/'conda_budget_by_seed.csv',index=False)
    # User separates widening injection from erosion. Also examine the signed
    # radial budget, not just the finite net width change. This is a task-level
    # decomposition, not the sum of per-update radial budgets.
    rrows=[]
    for target in ['dsigma2','radial']:
        for col in ['q','qabs','qcov','qcov_v2']:
            tr=primary[train];x=tr[col].to_numpy();y=-tr[target].to_numpy()
            slope=np.mean((x-x.mean())*(y-y.mean()))/(np.var(x)+1e-30)
            intercept=y.mean()-slope*x.mean()
            t,lo=threshold(x,y>0)
            for (a,seed),g in primary[test].groupby(['a','seed']):
                xx=g[col].to_numpy();yy=-g[target].to_numpy();pp=intercept+slope*xx
                rrows.append(dict(target=target,feature=col,a=a,seed=seed,slope=slope,intercept=intercept,
                    r2=1-float(np.sum((yy-pp)**2)/(np.sum((yy-yy.mean())**2)+1e-30)),
                    spearman=spearman(xx,yy),ba=ba(yy>0,predict(xx,t,lo))))
    pd.DataFrame(rrows).to_csv(OUT/'erosion_amount_and_radial.csv',index=False)
    print('prediction completed',flush=True)

def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    met=pd.read_csv(OUT/'threshold_by_seed_and_slope.csv');pmf=pd.read_csv(OUT/'pm_phase_by_seed.csv')
    core=met[met.aux==0]
    perseed=core.groupby(['feature','seed'])[['ba','auc']].mean().reset_index()
    summary=perseed.groupby('feature')[['ba','auc']].median().sort_values('ba',ascending=False)
    summary.to_csv(OUT/'prediction_summary.csv')
    pms=pmf.groupby(['arm','start','end']).median(numeric_only=True).reset_index()
    pms.to_csv(OUT/'pm_phase_summary.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(15,4.6))
    for arm in ['LR','SNA']:
        p=pms[(pms.arm==arm)&~((pms.start==0)&(pms.end==625))]
        axes[0].plot(['0–20','20–300','300–625'],p.dsigma2,marker='o',label=arm)
    axes[0].axhline(0,color='gray',lw=.8);axes[0].set(title='PM: variance change per phase',ylabel='Mean change in within-unit variance',xlabel='Updates after task switch');axes[0].legend()
    sels=['q','qabs','qcov','qcov_v2','var','multivariable_logistic']
    for col in sels:
        g=core[core.feature==col].groupby('a').ba.median()
        axes[1].plot(g.index,g.values,marker='.',label=col)
    axes[1].axhline(.5,color='gray',ls=':');axes[1].axhline(.8,color='gray',ls='--')
    axes[1].set(title='condA: next-task width contraction',xlabel='Leaky negative slope a',ylabel='Held-out balanced accuracy',ylim=(.35,1));axes[1].legend(fontsize=7)
    endpoint=pd.read_csv(OUT/'conda_endpoint_ratios.csv');g=endpoint[(endpoint.aux==0)&(endpoint.window=='initial_to_final')].groupby('a')
    axes[2].plot(g.median_weight_ratio.median(),marker='o',label='Full weight norm')
    axes[2].plot(g.median_width_ratio.median(),marker='o',label='Input width')
    axes[2].axhline(1,color='gray',ls='--');axes[2].set(title='condA: final / initial size',xlabel='Leaky negative slope a',ylabel='Median individual ratio');axes[2].legend()
    fig.tight_layout();fig.savefig(OUT/'verification.png',dpi=170);fig.savefig(OUT/'verification.pdf');plt.close(fig)
    verdict=[]
    for col in ['q','qabs','qcov','qcov_v2']:
        pooled=float(summary.loc[col,'ba']);min_a=float(core[core.feature==col].groupby('a').ba.median().min())
        verdict.append(dict(hypothesis='single_threshold_'+col,label='PREDICTIVE_SUPPORT' if pooled>=.8 and min_a>=.7 else 'NOT_SUFFICIENT',ba=pooled,min_slope_ba=min_a,grade='posthoc'))
    pd.DataFrame(verdict).to_csv(OUT/'verdict.csv',index=False)
    text='# φφ′積分と幅の侵食 — 事後検証\n\n追加学習なし。既存のcondA傾き梯子・PM保存状態を解析。\n\n'
    text+='## 単一閾値の予測\n\n'+summary.to_string()+'\n\n'
    text+='seed0–4で閾値を決め、seed5–9へ固定して適用。BA=.5が無情報。傾きごと・seedごとの値を先に集約。\n'
    text+='q=E[φφ′]、qabs=E|φφ′|、qcov=E[φφ′(z−z̄)]。固定32入力なので平均と合計は閾値の倍率だけが異なる。\n\n'
    text+='## PMの時間順序\n\n'+pms[['arm','start','end','norm_delta','radial','injection','dsigma2','var_linear','var_quadratic','growth_fraction','median_abs_cos','symmetry_among_growth']].to_string(index=False)+'\n\n'
    text+='## 適用範囲\n\n単一閾値の不成立はφφ′を含む自己項の不在を意味しない。condAの1層MSE/SGDとPMの多層CE/Adamを同一の力学として混ぜない。入力上の幅と重みノルムを分けた。平均沈降の検定ではない。\n\n'
    text+='測定済み未来は予測特徴に含めていない。既存研究を読んだ後の事後解析であり、新規の事前登録実験ではない。20更新内の時間順序と因果的な媒介割合は未判定。\n'
    (OUT/'summary.md').write_text(text)
    print(summary.to_string());print(pms[['arm','start','end','dsigma2','norm_delta','median_abs_cos']].to_string(index=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['conda','pm','evaluate','report','algebra']);args=ap.parse_args()
    globals()[args.stage]()

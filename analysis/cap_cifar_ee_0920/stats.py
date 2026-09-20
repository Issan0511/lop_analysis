"""Registered S5 statistics; all thresholds come from spec_cap_cifar_ee_0920."""
import math
import numpy as np
from analysis.resp_cifar_ee_0920.stats import interval as raw_interval,t_quantile
def interval(values,level=.95):
    x=np.asarray(values,dtype=float)
    assert x.shape==(10,) and np.isfinite(x).all()
    if np.all(x==x[0]):
        value=float(x[0]);return dict(mean=value,sd=0.,low=value,high=value,level=level,degenerate_sd=True,sign='+' if value>0 else '-' if value<0 else '0')
    return raw_interval(x,level)


def mean64(values,axis=None):
    x=np.asarray(values,dtype=np.float64)
    if axis is None:return x.flat[0]+(x-x.flat[0]).mean()
    anchor=np.take(x,[0],axis=axis)
    return np.squeeze(anchor,axis=axis)+(x-anchor).mean(axis)


ARMS=('ref','cap1','cap2','cap12','cap12_bfix')


def validate(data):
    assert set(data)==set(ARMS),'arm completeness'
    for arm in ARMS:
        assert set(data[arm])=={'A','G','F'}
        for k,v in data[arm].items():
            a=np.asarray(v,dtype=float);assert a.shape==(10,50) and np.isfinite(a).all(),(arm,k,'shape/finite')
            assert np.all((a>=0)&(a<=1)),(arm,k,'range')
        for k in ('A','G','F'):assert np.array_equal(np.asarray(data[arm][k])[:,0],np.asarray(data['ref'][k])[:,0]),'prefix'
        assert np.array_equal(data[arm]['F'],data['ref']['F']),'labels/floor'


def upper_floor(f):
    f=np.asarray(f,dtype=float);assert f.shape==(10,20) and np.isfinite(f).all()
    # Explicit exact-constant handling avoids a spurious SD from mean rounding.
    mean=mean64(f,1);sd=f.std(1,ddof=1);constant=(f==f[:,:1]).all(1)
    mean=np.where(constant,f[:,0],mean);sd=np.where(constant,0,sd)
    return mean+t_quantile(.95,19)*sd/math.sqrt(20),mean,sd


def floor_state(n):return 'FLOORED' if n==10 else 'ALIVE' if n==0 else 'SPLIT'


def classify(n,a,g):
    state=floor_state(n)
    if state=='FLOORED':return 'COLLAPSED'
    if state=='SPLIT':return 'SPLIT'
    if a=='+':return 'RESCUED' if g=='+' else 'RESCUED_FUNCTION_ONLY'
    return 'ALIVE_UNRESOLVED'


def time_half(a):
    a=np.asarray(a,dtype=float);assert a.shape==(50,) and np.isfinite(a).all()
    if a[0]<=.1:return dict(task=None,censored=False,undefined=True)
    hit=np.flatnonzero((a[1:]-.1)/(a[0]-.1)<.5)
    return dict(task=int(hit[0]+2) if len(hit) else 50,censored=not bool(len(hit)),undefined=False)


def timing(a,b):
    rows=[];signs=[]
    for seed,(aa,bb) in enumerate(zip(a,b)):
        x,y=time_half(aa),time_half(bb);sgn=None
        if not x['undefined'] and not y['undefined']:
            if x['censored'] and y['censored']:sgn=None
            elif x['censored']:sgn=1
            elif y['censored']:sgn=-1
            else:sgn=int(np.sign(x['task']-y['task']))
        if sgn in (-1,1):signs.append(sgn)
        rows.append(dict(seed=seed,arm_time=x,ref_time=y,sign=sgn))
    n=len(signs);later=sum(s==1 for s in signs);earlier=n-later
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(later,earlier)+1))/2**n) if n else None
    lab='TIMING_UNDETERMINED' if n==0 else 'LATER' if p<.05 and later>earlier else 'EARLIER' if p<.05 else 'NO_TIMING_DIFF'
    return dict(label=lab,p=p,comparable=n,later=later,earlier=earlier,rows=rows)


def evaluate(data):
    validate(data);x={a:{k:np.asarray(v,dtype=np.float64) for k,v in d.items()} for a,d in data.items()}
    upper,fmean,fsd=upper_floor(x['ref']['F'][:,30:50]);endpoints={};floors={}
    for a in ARMS:
        endpoints[a]={k:(mean64(x[a][field][:,30:50],1)-x[a][field][:,0]) for k,field in [('E1','A'),('E2','G')]}
        floors[a]=mean64(x[a]['A'][:,30:50],1)<=upper
    applicable=int(floors['ref'].sum())>=9 and interval(endpoints['ref']['E2'])['high']<0
    ref=x['ref'];early_ok=bool((ref['A'][:,1]>ref['F'][:,1]).all());early_mid=(ref['A'][:,1]+ref['F'][:,1])/2
    arms={};per_seed=[];paired=[];secondary=[]
    for a in ARMS:
        deltas={k:endpoints[a][k]-endpoints['ref'][k] for k in ('E1','E2')}
        levels=(.975,.95) if a=='cap12' else (.95,)
        cis={k:interval(deltas[k],levels[0]) for k in ('E1','E2')}
        n=int(floors[a].sum());early_n=int((x[a]['A'][:,1]<early_mid).sum()) if early_ok else None
        lab=classify(n,cis['E1']['sign'],cis['E2']['sign']) if applicable else 'NOT_REPRODUCED'
        if a=='ref' and applicable:lab='REFERENCE_FLOORED' if n==10 else 'REFERENCE_9_OF_10'
        arms[a]=dict(label=lab,floor_count=n,floor_state=floor_state(n),E1=cis['E1'],E2=cis['E2'],early_count=early_n,
                     early='EARLY_FLAG_NOT_ASSESSABLE' if not early_ok else 'IMPAIRED_EARLY' if early_n>=6 else 'NO_EARLY_IMPAIRMENT',
                     online_first=float(x[a]['A'][:,0].mean()),online_late=float(x[a]['A'][:,30:50].mean()),G_first=float(x[a]['G'][:,0].mean()),G_late=float(x[a]['G'][:,30:50].mean()))
        for seed in range(10):
            per_seed.append(dict(arm=a,seed=seed,U=upper[seed],floor_mean=fmean[seed],floor_sd=fsd[seed],A_late=mean64(x[a]['A'][seed,30:50]),A_minus_U=mean64(x[a]['A'][seed,30:50])-upper[seed],AT_FLOOR=bool(floors[a][seed]),
                                 E1=endpoints[a]['E1'][seed],E2=endpoints[a]['E2'][seed],delta_E1=deltas['E1'][seed],delta_E2=deltas['E2'][seed],early_A=x[a]['A'][seed,1],early_mid=early_mid[seed],early_below=bool(x[a]['A'][seed,1]<early_mid[seed]) if early_ok else None))
            for k in ('E1','E2'):paired.append(dict(arm=a,seed=seed,metric=k,arm_value=endpoints[a][k][seed],ref_value=endpoints['ref'][k][seed],difference=deltas[k][seed]))
        for lev in levels:
            for k in ('E1','E2'):secondary.append(dict(arm=a,metric=k,**interval(deltas[k],lev),tier='PRIMARY' if a=='cap12' and lev==.975 else 'REPORT_ONLY'))
        for lo,hi in [(2,10),(11,30),(31,50),(41,50)]:
            for k in ('A','G'):secondary.append(dict(arm=a,metric=k,window=f'{lo}-{hi}',mean=float(x[a][k][:,lo-1:hi].mean()),tier='REPORT_ONLY'))
    interactions={k:interval(endpoints['cap12'][k]-endpoints['cap1'][k]-endpoints['cap2'][k]+endpoints['ref'][k]) for k in ('E1','E2')}
    times={a:timing(x[a]['A'],x['ref']['A']) for a in ARMS if a!='ref'}
    return dict(label=arms['cap12']['label'],applicable=applicable,reference_G_change=interval(endpoints['ref']['E2']),arms=arms,per_seed=per_seed,paired=paired,secondary=secondary,interactions=interactions,timing=times)

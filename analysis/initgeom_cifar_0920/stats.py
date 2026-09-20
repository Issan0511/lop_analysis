"""Exact discrete order-statistic bands under the registered binomial model."""
import math
import numpy as np
from analysis.resp_cifar_ee_0920.stats import t_quantile
Q99=2.3263478740408408
CONDITIONS=('raw','std','C','gamma025','gamma050','gamma075','gamma150','gamma200')
DIAL=('gamma025','gamma050','gamma075','raw','gamma150','gamma200')
ALPHA_CONDITION=.05/8

def probability(r,q=Q99):
    if r is None:return None
    if math.isnan(r) or r<0:raise ValueError('invalid r')
    return 0. if r==0 else 1. if math.isinf(r) else math.erfc(q/(math.sqrt(2)*r))

def binomial_pmf(n,p):
    assert isinstance(n,int) and n>0 and math.isfinite(p) and 0<=p<=1
    a=np.zeros(n+1)
    if p in (0,1):a[int(p)*n]=1;return a
    logs=np.array([math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)+k*math.log(p)+(n-k)*math.log1p(-p) for k in range(n+1)])
    a=np.exp(logs-logs.max());a/=math.fsum(a)
    return a

def poisson_binomial(probabilities):
    a=np.array([1.])
    for p in probabilities:
        assert 0<=p<=1
        b=np.zeros(len(a)+1);b[:-1]+=a*(1-p);b[1:]+=a*p;a=b
    return a

def order_cdf(ps,units=100):
    ps=list(ps);assert len(ps)%2==0 and len(ps)>=2
    Fs=np.stack([np.cumsum(binomial_pmf(units,p)) for p in ps]);Fs=np.clip(Fs,0,1);Fs[:,-1]=1
    js=(len(ps)//2,len(ps)//2+1);out=np.zeros((2,units+1))
    for k in range(units+1):
        counts=poisson_binomial(Fs[:,k])
        for i,j in enumerate(js):out[i,k]=min(1.,math.fsum(counts[j:]))
    out[:,-1]=1
    assert (np.diff(out,axis=1)>=-64*np.finfo(float).eps).all()
    return np.maximum.accumulate(out,axis=1)

def median_band(ps,units=100,alpha=ALPHA_CONDITION):
    assert 0<alpha<1
    cdf=order_cdf(ps,units)
    def quantile(j,a):return int(np.searchsorted(cdf[j],a,side='left'))
    qs=[[quantile(j,p) for j in range(2)] for p in (alpha/4,1-alpha/4)]
    return dict(low=sum(qs[0])/(2*units),high=sum(qs[1])/(2*units),quantiles=qs,alpha=alpha,coverage_at_least=1-alpha,model='independent seed Binomial(unit_count,p_s); conservative union bound for two central order statistics')

def condition_label(low,high,band):
    if low is None or high is None:return 'UNDEFINED'
    if not all(math.isfinite(x) for x in (low,high,band['low'],band['high'])):return 'DIVERGED'
    assert 0<=low<=high<=1
    if low>band['high']:return 'OFF_HIGH'
    if high<band['low']:return 'OFF_LOW'
    if band['low']<=low and high<=band['high']:return 'PREDICTED'
    return 'NUMERIC_UNRESOLVED'

def family(labels,layer):
    if set(labels)!=set(CONDITIONS):return 'INCOMPLETE'
    values=list(labels.values());prefix='L1' if layer==1 else 'L2_CONDITIONAL'
    if any(x in ('INCOMPLETE','CHECK_FAILED','DIVERGED') for x in values):return 'L1_UNRESOLVED' if layer==1 else 'L2_UNRESOLVED'
    if any(x in ('OFF_HIGH','OFF_LOW') for x in values):return prefix+'_MODEL_MISS'
    if all(x=='PREDICTED' for x in values):return prefix+'_ALL_COMPATIBLE'
    return 'L1_UNRESOLVED' if layer==1 else 'L2_UNRESOLVED'

def monotone(intervals):
    assert len(intervals)==6
    if any(intervals[j+1][1]<intervals[j][0] for j in range(5)):return 'NONMONOTONE_SAMPLE_MEDIANS'
    if all(intervals[j+1][0]>=intervals[j][1] for j in range(5)):return 'MONOTONE_SAMPLE_MEDIANS'
    return 'NUMERIC_UNRESOLVED'

def paired_interval(d):
    d=np.asarray(d,float);assert d.shape==(20,) and np.isfinite(d).all()
    mean=float(d.mean());sd=float(d.std(ddof=1));width=t_quantile(.995,19)*sd/math.sqrt(20) if sd else 0.
    lo,hi=mean-width,mean+width
    return dict(mean=mean,sd=sd,low=lo,high=hi,level=.99,df=19,sign='+' if lo>0 else '-' if hi<0 else '0',degenerate=sd==0)

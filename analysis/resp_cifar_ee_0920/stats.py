"""Student t via the registered resp_ee_0917 continued-fraction implementation."""
import math
import numpy as np

def _betacf(a: float, b: float, x: float) -> float:
    tiny, eps = 1e-300, 1e-15
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise RuntimeError("betacf did not converge")

def betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b

def t_cdf(t: float, df: int) -> float:
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail

def t_quantile(p: float, df: int) -> float:
    lo, hi = 0.0, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)

def interval(values, level=.95):
    x=np.asarray(values,dtype=np.float64)
    assert len(x)==10 and np.isfinite(x).all(), 'requires all ten finite seeds'
    mean=float(x.mean()); sd=float(x.std(ddof=1)); width=t_quantile((1+level)/2,9)*sd/math.sqrt(10)
    low,high=mean-width,mean+width
    return dict(mean=mean,sd=sd,low=low,high=high,level=level,degenerate_sd=(sd==0),sign=('+' if low>0 else '-' if high<0 else '0'))

def label(p1,p2):
    a,b=p1['sign'],p2['sign']
    if '-' in (a,b):return 'RESPONSE_REVERSED'
    return {('+','+'):'RESPONSE_BOTH_WAYS',('+','0'):'RESTORE_ONLY',('0','+'):'SINK_ONLY',('0','0'):'RESPONSE_NOT_SHOWN'}[(a,b)]

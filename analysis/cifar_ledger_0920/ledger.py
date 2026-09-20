"""Registered endpoint accounting. No training, no experiment I/O."""
from dataclasses import dataclass
import numpy as np

U = np.finfo(np.float64).eps / 2
TAU = 1e-6
TERMS = ('self', 'upstream', 'cross', 'bias')


def gamma(n, u=U):
    return n * u / (1 - n * u)


def outward(x):
    return np.nextafter(np.asarray(x, dtype=np.float64), np.inf)


@dataclass
class Ball:
    """Value and conservative absolute roundoff/input error, elementwise."""
    v: np.ndarray
    e: np.ndarray | float = 0.0

    def __post_init__(self):
        self.v = np.asarray(self.v, dtype=np.float64)
        self.e = np.broadcast_to(self.e, self.v.shape).astype(np.float64)

    def __add__(self, b):
        b = b if isinstance(b, Ball) else Ball(b)
        v = self.v + b.v
        return Ball(v, outward((self.e + b.e + U * (abs(self.v) + abs(b.v))) / (1 - U)))

    def __neg__(self):
        return Ball(-self.v, self.e)

    def __sub__(self, b):
        return self + (-b if isinstance(b, Ball) else Ball(-np.asarray(b)))

    def __mul__(self, b):
        b = b if isinstance(b, Ball) else Ball(b)
        v = self.v * b.v
        e = abs(self.v)*b.e + abs(b.v)*self.e + self.e*b.e + U*abs(v)/(1-U)
        return Ball(v, outward(e / (1 - gamma(6))))

    def __truediv__(self, b):
        b = b if isinstance(b, Ball) else Ball(b)
        lower = abs(b.v)-b.e
        with np.errstate(divide='ignore', invalid='ignore'):
            v = self.v/b.v
            e = (self.e + abs(v)*b.e)/lower + U*abs(v)/(1-U)
        return Ball(v, np.where(lower > 0, outward(e/(1-gamma(5))), np.inf))

    def sum(self, axis=-1):
        n = self.v.shape[axis]
        v = self.v.sum(axis=axis)
        e = self.e.sum(axis=axis)/(1-gamma(n)) + gamma(n)*abs(self.v).sum(axis=axis)
        return Ball(v, outward(e/(1-gamma(4))))

    def sqrt(self):
        with np.errstate(invalid='ignore', divide='ignore'):
            v = np.sqrt(self.v)
            lo = np.sqrt(np.maximum(0, self.v-self.e))
            hi = np.sqrt(self.v+self.e)
            e = np.maximum(v-lo, hi-v)+U*abs(v)/(1-U)
        return Ball(v, outward(e/(1-gamma(4))))


def dot(w, mu):
    return (w * mu).sum(-1)


def mean_ball(x, axis=0):
    return Ball(x).sum(axis) / x.shape[axis]


def expand(x, axis):
    return Ball(np.expand_dims(x.v, axis), np.expand_dims(x.e, axis))


def norm_direction(x):
    r = (x*x).sum(-1).sqrt()
    return r, x/expand(r, -1)


def components(old, new):
    """States contain w[H,d], b[H], mu[d], mu_error[d]."""
    w0, w1 = Ball(old['w']), Ball(new['w'])
    b0, b1 = Ball(old['b']), Ball(new['b'])
    m0, m1 = Ball(old['mu'], old['mu_error']), Ball(new['mu'], new['mu_error'])
    dw, dm = w1-w0, m1-m0
    out = dict(self=dot(dw,m0), upstream=dot(w0,dm), cross=dot(dw,dm), bias=b1-b0)
    out['delta'] = (dot(w1,m1)+b1)-(dot(w0,m0)+b0)
    total = out['self']+out['upstream']+out['cross']+out['bias']
    out['closure'] = total-out['delta']
    r0,e0=norm_direction(m0); r1,e1=norm_direction(m1)
    out['upstream_growth']=dot(w0, (r1-r0)*((e1+e0)/2))
    out['upstream_rotation']=dot(w0, ((r1+r0)/2)*(e1-e0))
    rw0,ew0=norm_direction(w0); rw1,ew1=norm_direction(w1)
    out['self_growth']=dot(expand(rw1-rw0,-1)*((ew1+ew0)/2),m0)
    out['self_rotation']=dot(expand((rw1+rw0)/2,-1)*(ew1-ew0),m0)
    return out


def response(g):
    g=np.asarray(g)
    return dict(low=(abs(g)<TAU).mean(0), zero=(g==0).mean(0),
                negative=(g<0).mean(0), signed_low=(g<TAU).mean(0),
                mean=g.astype(np.float64).mean(0), abs_mean=abs(g).astype(np.float64).mean(0),
                dead=(abs(g).max(0)<TAU).astype(float))


def depth(m, sd):
    with np.errstate(divide='ignore', invalid='ignore'):
        return m/sd


def first_event(q, online, m, sd, m_error=None, sd_error=None):
    """q[task,layer], states [task,layer,unit]; t00 online is NaN."""
    limit=min(10,len(q)-1)
    times=[]
    for l in range(2):
        ix=np.flatnonzero(q[1:limit+1,l]>.5)
        times.append(int(ix[0]+1) if len(ix) else None)
    online_ix=np.flatnonzero(online[1:limit+1]<.5)
    ta=int(online_ix[0]+1) if len(online_ix) else None
    res=dict(T1=times[0],T2=times[1],T_A=ta,low_from_task1=bool(online[1]<.5))
    finite=[t for t in times if t is not None]
    if finite:
        t=min(finite); ls=[l for l in range(2) if times[l]==t]
        initial=[l+1 for l in ls if q[0,l]>.5 and np.all(q[:t+1,l]>.5)]
        layer='SIMULTANEOUS' if len(ls)==2 else f'L{ls[0]+1}_FIRST'
        label=(f'INITIAL_LOW_{layer}' if initial else layer)
        return dict(res,T_star=t,label=label,layer=layer,initial_low_layers=initial)
    if ta is None:
        return dict(res,T_star=None,label='NO_EVENT_BY_T10',layer=None,initial_low_layers=[])
    a=abs(depth(m[ta],sd[ta])); d=a.mean(-1)
    # Delta-method is avoided: ratio enclosure, with exact sd=0 handled explicitly.
    me=np.zeros_like(m[ta]) if m_error is None else m_error[ta]
    se=np.zeros_like(sd[ta]) if sd_error is None else sd_error[ta]
    b=Ball(abs(m[ta]),me)/Ball(sd[ta],se)
    be=b.e.mean(-1)+gamma(a.shape[-1])*abs(d)
    if np.isnan(a).any() or np.isinf(d).all():
        label='FALLBACK_TIE'
    elif np.isinf(d[0]) and np.isfinite(d[1]):label='FALLBACK_L1'
    elif np.isinf(d[1]) and np.isfinite(d[0]):label='FALLBACK_L2'
    elif d[0]-be[0]>d[1]+be[1]:label='FALLBACK_L1'
    elif d[1]-be[1]>d[0]+be[0]:label='FALLBACK_L2'
    else:label='FALLBACK_TIE'
    return dict(res,T_star=ta,label=label,layer=label,initial_low_layers=[])


def half_side(a,d):
    """a/d >= 1/2 for strictly negative d; return None if unresolved."""
    h=a.v-d.v/2
    if a.e==0 and d.e==0 and h==0:return True
    err=a.e+d.e/2+U*(abs(a.v)+abs(d.v/2))
    if h+err<0:return True
    if h-err>0:return False
    return None


def carrier(totals, delta, event_label='L2_FIRST'):
    d=delta
    if event_label=='NO_EVENT_BY_T10':return 'NO_EVENT',''
    if event_label.startswith('INITIAL_LOW'):return 'INITIAL_LOW_REPORT_ONLY',''
    if d.v-d.e>0:return 'NO_NET_SINK',''
    if d.v+d.e>=0:return 'UNRESOLVED_NET_CHANGE',''
    us=half_side(totals['upstream'],d); ss=half_side(totals['self'],d)
    if us is None or ss is None:return 'UNRESOLVED_SHARE',''
    if us and not ss:return 'UPSTREAM_CARRIES',''
    if ss and not us:return 'SELF_CARRIES',''
    return 'MIXED',('BOTH_GE_HALF' if us else 'NEITHER_GE_HALF')


def majority(labels):
    if len(labels)!=10 or 'INAPPLICABLE' in labels:return 'INAPPLICABLE'
    from collections import Counter
    counts=Counter(labels)
    best,n=counts.most_common(1)[0]
    return best if n>=6 else 'SPLIT'


def window(shards,slot,layer,start,end):
    """Sum time per unit, mean units, then ratios; never pool seeds."""
    out={}
    for k in TERMS+('upstream_growth','upstream_rotation','self_growth','self_rotation'):
        vals=np.stack([shards[t][f'ledger_{k}'][slot,layer] for t in range(start+1,end+1)])
        errs=np.stack([shards[t][f'ledger_{k}_error'][slot,layer] for t in range(start+1,end+1)])
        summed=Ball(vals,errs).sum(0)
        out[k]=summed.sum(-1)/vals.shape[-1]
        out[k+'_median']=float(np.median(summed.v))
    delta=Ball(shards[end]['m_alg'][slot,layer],shards[end]['m_alg_error'][slot,layer])-Ball(shards[start]['m_alg'][slot,layer],shards[start]['m_alg_error'][slot,layer])
    out['delta']=delta.sum(-1)/delta.v.size
    out['delta_median']=float(np.median(delta.v))
    return out

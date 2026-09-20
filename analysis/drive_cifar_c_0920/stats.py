"""Registered A2 seed-level decisions and calibration-only segmentation."""
import math
import numpy as np

def directional(d, positive='CROSS_TASK_SINK_CONDITION_HOLDS',negative='NOT_SUPPORTED'):
    a=np.asarray(d,float)
    if a.shape!=(5,):return dict(label='INCOMPLETE')
    if not np.isfinite(a).all():return dict(label='DIVERGED')
    n=int((a!=0).sum());k=int((a>0).sum())
    p=sum(math.comb(n,j) for j in range(k,n+1))/2**n
    return dict(label=positive if (a>0).all() else negative if (a<0).all() else 'UNRESOLVED',differences=a.tolist(),positive=k,nonzero=n,one_sided_p=p)

def certificate_status(certificates,violations):
    return 'CHECK_FAILED' if violations else 'CERTIFICATE_CONSISTENT' if certificates else 'NO_CERTIFIABLE_EVENTS'

def window(v):
    v=np.asarray(v,float)
    assert v.shape==(400,) and np.isfinite(v).all()
    rss0=float(((v-v.mean())**2).sum())
    if (v==v[0]).all():return dict(label='WINDOW_NOT_IDENTIFIED',reason='constant',break_epoch=None)
    candidates=[]
    for b in range(1,400):
        early,late=float(v[:b].mean()),float(v[b:].mean())
        rss=float(((v[:b]-early)**2).sum()+((v[b:]-late)**2).sum())
        candidates.append((rss,b,early,late))
    rss1,b,early,late=min(candidates)
    bic0=400*math.log(rss0/400)+math.log(400)
    bic1=400*math.log(rss1/400)+3*math.log(400) if rss1 else -math.inf
    identified=bic1<bic0 and early<0 and early<late
    return dict(label='WINDOW_IDENTIFIED' if identified else 'WINDOW_NOT_IDENTIFIED',break_epoch=b if identified else None,boundary_updates=75*b if identified else None,rss0=rss0,rss1=rss1,bic0=bic0,bic1=bic1,early=early,late=late,reason='operational segmentation; not a significance test')

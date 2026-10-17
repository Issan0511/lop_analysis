#!/usr/bin/env python3
"""Registered 15 simultaneous paired intervals across the three native boxes."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import math
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src import resp_step_controls_1018 as R
M=15;EQ=.005
PAIRS=(('R','N'),('LR_G','N'),('LR_U','N'),('LR_G','R'),('LR_U','R'))


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



def interval(d):
    d=np.asarray(d,dtype=np.float64)
    if d.shape!=(10,) or not np.isfinite(d).all():raise ValueError('INCOMPLETE: require 10 finite seed differences')
    mean=float(d.mean());sd=float(d.std(ddof=1));se=sd/np.sqrt(10)
    q=float(t_quantile(1-.05/(2*M),9));q95=float(t_quantile(.975,9))
    return dict(mean=mean,lo=mean-q*se,hi=mean+q*se,lo95=mean-q95*se,hi95=mean+q95*se,
                median=float(np.median(d)),differences=d.tolist(),positive=int((d>0).sum()),
                zero=int((d==0).sum()),negative=int((d<0).sum()),sd=sd,
                status='DEGENERATE_SD' if sd==0 else 'OK')


def box_verdict(rows,box):
    task=29 if box=='F51' else 11
    relevant=[r for r in rows if int(r['task'])==task]
    expected={(s,a) for s in range(10) for a in R.ARMS}
    keys=[(int(r['seed']),r['arm']) for r in relevant]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:return dict(status='INCOMPLETE')
    if any(not r.get('finite',False) for r in relevant):return dict(status='DIVERGED')
    by={k:r for k,r in zip(keys,relevant)}
    # Paired streams and initial state are mandatory; the arms must inherit the same branch.
    for s in range(10):
        for key in ('label_hash','order_hash','initial_state_sha256'):
            vals=[by[s,a].get(key) for a in R.ARMS]
            if key=='initial_state_sha256' and box!='RM':continue
            if any(v is None for v in vals) or len(set(vals))!=1:return dict(status='CHECK_FAILED',reason=f'stream/branch mismatch: {key}, seed{s}')
    effects={f'{a}-{b}':interval([float(by[s,a]['online_acc'])-float(by[s,b]['online_acc']) for s in range(10)]) for a,b in PAIRS}
    if effects['R-N']['lo']<=0:return dict(status='NOT_REPRODUCED',effects=effects)
    controls={}
    for c in ('LR_G','LR_U'):
        cn,cr=effects[f'{c}-N'],effects[f'{c}-R']
        if cn['lo']>0 and cr['lo']>-EQ:controls[c]='REPRODUCES'
        elif cr['hi'] < -EQ:controls[c]='SHORTFALL'
        else:controls[c]='UNRESOLVED'
    if 'REPRODUCES' in controls.values():label='LR_EXPLAINS'
    elif all(v=='SHORTFALL' for v in controls.values()):label='NOT_LR'
    else:label='UNRESOLVED'
    return dict(status='COMPLETE',label=label,controls=controls,effects=effects)


def verdict(inputs):
    boxes={b:box_verdict(inputs.get(b,[]),b) for b in R.BOXES}
    statuses=[v['status'] for v in boxes.values()]
    label=None
    if any(s=='DIVERGED' for s in statuses):status='DIVERGED'
    elif any(s=='CHECK_FAILED' for s in statuses):status='CHECK_FAILED'
    elif any(s=='INCOMPLETE' for s in statuses):status='INCOMPLETE'
    elif any(s=='NOT_REPRODUCED' for s in statuses):status='NOT_REPRODUCED'
    else:
        status='COMPLETE';labels=[b['label'] for b in boxes.values()]
        label='NOT_LR_ALL_THREE' if labels==['NOT_LR']*3 else 'LR_EXPLAINS_ALL_THREE' if labels==['LR_EXPLAINS']*3 else 'MIXED_BY_BOX'
    return dict(run=R.RUN,status=status,label=label,boxes=boxes,m=M,epsilon_eq=EQ,
                prediction='NOT_LR_ALL_THREE',scope='Registered constant layer-specific LR_G and LR_U only; no claim about arbitrary lr or preconditioning')


def main():
    p=argparse.ArgumentParser();p.add_argument('--RC',type=Path);p.add_argument('--RM',type=Path);p.add_argument('--F51',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    inputs={}
    for b in R.BOXES:
        path=getattr(a,b)
        if path and path.exists():inputs[b]=json.loads(path.read_text())
    out=verdict(inputs);R.put(a.out/'verdict.json',out)
    lines=[f"{R.RUN}: {out['status']}",f"Overall label: {out['label']}",'15 Bonferroni paired t intervals; n=10 seed pairs; equivalence margin 0.005.','']
    for b,v in out['boxes'].items():
        lines.append(f"{b}: {v['status']} {v.get('label','')}")
        for pair,e in v.get('effects',{}).items():lines.append(f"  {pair}: {e['mean']*100:.4f} pt; simultaneous CI [{e['lo']*100:.4f}, {e['hi']*100:.4f}] pt")
    (a.out/'summary.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()

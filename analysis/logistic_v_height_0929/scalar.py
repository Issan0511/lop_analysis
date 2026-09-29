"""Frozen readout logistic audit; stable log-moment Adam, no data downloads."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np


OPTIMIZERS = {
    'gd': (0., 0.), 'adam': (.9, .999), 'no_first': (0., .999),
    'no_second': (.9, 0.), 'instant': (0., 0.), 'equal_memory': (.9, .9),
}


def csv_write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def config(group, v, optimizer='adam', eta=.001, eps=0., z0=0., steps=200000):
    b1, b2 = OPTIMIZERS[optimizer]
    return dict(group=group, v=float(v), optimizer=optimizer, eta=float(eta),
                eps=float(eps), z0=float(z0), beta1=b1, beta2=b2, steps=steps)


def sweep_configs():
    rows = []
    for v in np.logspace(-4, 2, 31):
        for opt in OPTIMIZERS:
            for eps in ([0.] if opt == 'gd' else [0., 1e-8, 1e-4]):
                rows.append(config('core', v, opt, eps=eps))
    for v in [.01, .1, 1., 10., 100.]:
        for eta in [1e-4, 1e-3, 1e-2]:
            for z0 in [-2., 0., 2.]:
                for opt in ['gd', 'adam', 'instant']:
                    for eps in ([0.] if opt == 'gd' else [0., 1e-8]):
                        rows.append(config('rate_init', v, opt, eta, eps, z0, 20000))
    for v in [1e-8, 1e-6, 1e3, 1e4, 1e6]:
        for eps in [0., 1e-8, 1e-4]:
            rows.append(config('extreme', v, eps=eps, steps=4000))
    # Small exact-gain equivalence pairs, initialized at identical logits.
    for gain in [.1, 1., 10.]:
        rows.append(config('matched_effective', gain, eta=.001/gain,
                           eps=1e-8*gain, z0=.2/gain, steps=4000))
    return rows


def run_log_sweep(configs, marks):
    """Positive residual/negative gradient: log-domain EMA avoids underflow."""
    n = len(configs)
    v = np.array([c['v'] for c in configs])
    eta = np.array([c['eta'] for c in configs])
    b1 = np.array([c['beta1'] for c in configs])
    b2 = np.array([c['beta2'] for c in configs])
    eps = np.array([c['eps'] for c in configs])
    horizon = np.array([c['steps'] for c in configs])
    gd = np.array([c['optimizer'] == 'gd' for c in configs])
    with np.errstate(divide='ignore'):
        lb1, lb2, leps = np.log(b1), np.log(b2), np.log(eps)
    lm = np.full(n, -np.inf)
    lq = np.full(n, -np.inf)
    lv, leta = np.log(v), np.log(eta)
    z = np.array([c['z0'] for c in configs])
    diverged = np.zeros(n, bool)
    diverged_at = np.zeros(n, int)
    records = []
    checkpoints = set(marks) | set(horizon.tolist())
    old_logexp = np.logaddexp(0., v*z)
    last_growth = np.zeros(n)
    for t in range(1, int(horizon.max())+1):
        active = (~diverged) & (t <= horizon)
        logr = -np.logaddexp(0., v*z)
        logg = lv+logr
        lm_new = np.logaddexp(lb1+lm, np.log1p(-b1)+logg)
        lq_new = np.logaddexp(lb2+lq, np.log1p(-b2)+2*logg)
        lm[active], lq[active] = lm_new[active], lq_new[active]
        # log(1-beta^t), using expm1 for early bias corrections.
        corr1 = np.log(-np.expm1(t*lb1))
        corr2 = np.log(-np.expm1(t*lb2))
        logstep = leta + lm-corr1-np.logaddexp(.5*(lq-corr2), leps)
        logstep[gd] = (leta+logg)[gd]
        # A finite threshold diagnoses explosive no-second-memory controls.
        bad = active & ((logstep > math.log(1e6)) | (~np.isfinite(logstep)))
        # -inf is a legitimate underflowing/zero update in GD.
        bad &= logstep != -np.inf
        diverged[bad], diverged_at[bad] = True, t
        active &= ~bad
        step = np.zeros(n)
        step[active] = np.exp(logstep[active])
        old_z = z.copy()
        z[active] += step[active]
        # Stable increment e^margin; needed for the epsilon asymptotic check.
        dm = v*(z-old_z)
        with np.errstate(over='ignore', invalid='ignore'):
            last_growth = np.exp(v*old_z)*np.expm1(dm)
        if t in checkpoints:
            ids = np.flatnonzero(t <= horizon)
            for k in ids:
                m = float(v[k]*z[k])
                row = dict(config_id=int(k), **configs[k], t=t, z=float(z[k]),
                    margin=m, loss=float(np.logaddexp(0., -m)),
                    log_residual=float(-np.logaddexp(0., m)),
                    step=float(step[k]), step_over_eta=float(step[k]/eta[k]),
                    log_second_rms=float(.5*(lq[k]-corr2[k])),
                    epsilon_over_rms=float(np.exp(min(700., leps[k]-.5*(lq[k]-corr2[k])))),
                    exp_margin_increment=float(last_growth[k]),
                    exp_margin_over_t=float(np.exp(m-math.log(t))) if m-math.log(t)<700 else float('inf'),
                    status='diverged' if diverged[k] else 'ok',
                    diverged_at=int(diverged_at[k]),
                    ordinary_gradient_underflows=bool(lv[k]-np.logaddexp(0.,m)<math.log(np.finfo(float).tiny)))
                records.append(row)
    return records


def flow_margin(v, eta, t, z0=0.):
    m0 = v*z0
    target = m0+math.exp(m0)+eta*v*v*t
    lo, hi = m0, max(m0+1., math.log(max(target, 1.))+1.)
    for _ in range(120):
        mid=(lo+hi)/2
        if mid+math.exp(mid)<target:
            lo=mid
        else:
            hi=mid
    return (lo+hi)/2


def torch_check():
    import torch
    torch.set_num_threads(1)
    rows=[]
    for opt in ['adam', 'no_first', 'instant', 'equal_memory']:
        cfg=[config('torch', v, opt, eps=1e-8, steps=4000) for v in [.1,1.,10.]]
        ref=run_log_sweep(cfg,[4000])
        z=torch.zeros(3,dtype=torch.float64,requires_grad=True)
        v=torch.tensor([.1,1.,10.],dtype=torch.float64)
        optimizer=torch.optim.Adam([z],lr=.001,betas=OPTIMIZERS[opt],eps=1e-8)
        for _ in range(4000):
            optimizer.zero_grad()
            torch.nn.functional.softplus(-v*z).sum().backward()
            optimizer.step()
        for k,r in enumerate(ref):
            actual=float(z.detach()[k])
            rows.append(dict(optimizer=opt,v=float(v[k]),log_domain=r['z'],
                             torch=actual,absolute_error=abs(r['z']-actual)))
    return rows


def basic_checks():
    grad=[]
    for v in [.1,1.,10.]:
        for z in [-2.,.1,2.]:
            h=1e-5
            fd=(np.logaddexp(0.,-v*(z+h))-np.logaddexp(0.,-v*(z-h)))/(2*h)
            analytic=-v*math.exp(-float(np.logaddexp(0.,v*z)))
            grad.append(dict(v=v,z=z,analytic=analytic,finite_difference=float(fd),
                             abs_error=abs(fd-analytic)))
    flow=[]
    for v in [.1,1.,10.]:
        target=flow_margin(v,1.,4.)/v
        for dt in [.001,.0005,.00025]:
            z=0.
            for _ in range(round(4/dt)):
                z+=dt*v/(1+math.exp(v*z))
            flow.append(dict(v=v,dt=dt,n_steps=round(4/dt),numeric=z,analytic=target,
                             abs_error=abs(z-target)))
    return grad,flow


def warmstart():
    rows=[]
    for target in [1.,.8]:
        z=m=q=0.
        t=0
        def advance(z,m,q,t,v):
            margin=v*z
            p=math.exp(-float(np.logaddexp(0.,-margin)))
            g=v*(p-target)
            m=.9*m+.1*g
            q=.999*q+.001*g*g
            t+=1
            mh=m/(1-.9**t)
            qh=q/(1-.999**t)
            z-=.001*mh/(math.sqrt(qh)+1e-8)
            return z,m,q,t
        for _ in range(4000):
            z,m,q,t=advance(z,m,q,t,1.)
        initial=(z,m,q,t)
        for gain in [.1,1.,10.]:
            for policy in ['retained','reset','rescaled']:
                z,m,q,t=initial
                if policy=='reset': m=q=0.;t=0
                if policy=='rescaled': m*=gain;q*=gain*gain
                zstart=z
                minstep=float('inf'); maxstep=-float('inf')
                for n in range(1,20001):
                    old=z
                    z,m,q,t=advance(z,m,q,t,gain)
                    minstep=min(minstep,z-old);maxstep=max(maxstep,z-old)
                    if n in [1,10,100,1000,4000,20000]:
                        margin=gain*z
                        loss=float(np.logaddexp(0.,margin)-target*margin)
                        rows.append(dict(target=target,gain=gain,moment_policy=policy,
                            t_after=n,z_before=zstart,z=z,delta_z=z-zstart,margin=margin,
                            loss=loss,min_step=minstep,max_step=maxstep,
                            finite_optimum_z=math.log(4)/gain if target==.8 else ''))
    # A GD contraction check, initialized at the old exact finite optimum.
    for gain in [.1,1.,10.]:
        z=math.log(4)
        rate=.5/(gain*gain)
        for _ in range(1000):
            p=math.exp(-float(np.logaddexp(0.,-gain*z)))
            z-=rate*gain*(p-.8)
        rows.append(dict(target=.8,gain=gain,moment_policy='gd_exact_optimum_control',
                         t_after=1000,z_before=math.log(4),z=z,delta_z=z-math.log(4),
                         margin=gain*z,finite_optimum_z=math.log(4)/gain,
                         optimum_absolute_error=abs(z-math.log(4)/gain)))
    return rows


def traveling_delta(v,eta=.001,b1=.9,b2=.999):
    cap=min(-math.log(b1),-.5*math.log(b2))
    lo,hi=0.,cap*(1-1e-12)
    for _ in range(120):
        d=(lo+hi)/2
        f=(1-b1)/(1-b1*math.exp(d))*math.sqrt((1-b2*math.exp(2*d))/(1-b2))
        if d-eta*v*f>0:hi=d
        else:lo=d
    return (lo+hi)/2


def write_summary(out, records, checks, warm, torch_rows, asym):
    lines=['# Frozen-v scalar logistic audit','',
        'Replicates the prior three-point toy, then audits broad controls. These are scalar mechanistic models, not RL-MNIST experiments.',
        'All entries are float64; log-domain Adam avoids arithmetic underflow. Unstable controls are flagged, not silently clipped.', '',
        '## Prior example and optimizer controls at 4000 steps', '',
        '| optimizer | epsilon | v | z | final step / eta | loss | status |',
        '|---|---:|---:|---:|---:|---:|---|']
    for r in records:
        if r['group']=='core' and r['t']==4000 and any(np.isclose(r['v'],v) for v in [.1,1,10]) and r['eps'] in [0.,1e-8]:
            lines.append(f"| {r['optimizer']} | {r['eps']:g} | {r['v']:g} | {r['z']:.8g} | {r['step_over_eta']:.6g} | {r['loss']:.6g} | {r['status']} |")
    lines += ['', '## Independent implementation and analytic checks','',
        f"- PyTorch comparison max endpoint error: {max(r['absolute_error'] for r in torch_rows):.4g}.",
        f"- Gradient finite-difference max error: {checks['gradient_max_error']:.4g}.",
        f"- Exact effective-gain matched logit endpoint range: {checks['matched_effective_margin_range']:.4g}.",
        f"- beta1<=beta2 Adam max step/eta: {checks['bounded_step_max']:.12g} (bound 1).",
        f"- GD finite-optimum error max: {checks['soft_target_gd_max_error']:.4g}.",
        '', '## Same-state readout intervention, retained Adam moments','',
        '| target probability | gain | z before | z after 4000 | change |','|---|---:|---:|---:|---:|']
    for r in warm:
        if r['moment_policy']=='retained' and r['t_after']==4000:
            lines.append(f"| {r['target']:g} | {r['gain']:g} | {r['z_before']:.8g} | {r['z']:.8g} | {r['delta_z']:.8g} |")
    lines+=['','The separable target=1 model never moves z downward. The target=.8 finite-optimum control can move either way; its unique margin is log(4). Neither changes normalized shape of the symmetric two-point input.','',
        '## Long-time check (exponential tracking ansatz is not a convergence theorem)','',
        '| v | epsilon | t | observed delta margin | epsilon=0 ansatz | exp(m)/t divided by eta*v^2/eps |',
        '|---:|---:|---:|---:|---:|---:|']
    for r in asym:
        lines.append(f"| {r['v']:g} | {r['eps']:g} | {r['t']} | {r['observed_margin_step']:.8g} | {r['tracking_delta']:.8g} | {r['epsilon_asymptotic_ratio']} |")
    lines+=['','## Limitations','',
        '- Endpoint ordering is not bidirectional response to an intervention.',
        '- At fixed loss, inverse-v height is a conditional identity; different v runs have different losses at equal steps.',
        '- Scalar symmetric points always have max/s=(max-median_midpoint)/s=1; no standardized tail mechanism exists here.',
        '- The actual frozen-W2 intervention changes softmax logits and preserves optimizer history. Its v is a vector norm, not a signed scalar mean.',
        '- No_second can be unstable; finite epsilon and positive moment memory change extreme/long-time limits.',
        '- Parameter sweeps are sensitivity checks, not independent statistical replications.']
    (out/'scalar_summary.md').write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('results/logistic_v_height_0929'));args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    configs=sweep_configs()
    # Record full matrix before executing numerical fits.
    (args.out/'scalar_config.json').write_text(json.dumps(configs,indent=2)+'\n')
    records=run_log_sweep(configs,[1,10,100,1000,4000,20000,100000,200000])
    print(f'Expanded sweep complete: {len(configs)} configurations, {len(records)} records',flush=True)
    torch_rows=torch_check();grad,flow=basic_checks();warm=warmstart()
    print('Independent torch, finite difference, gradient flow, and warm-start controls complete',flush=True)
    longconfigs=[config('long',v,eps=eps,steps=1000000) for v in [.1,1.,10.] for eps in [0.,1e-8,1e-4]]
    longrows=run_log_sweep(longconfigs,[200000,1000000])
    asym=[]
    for r in longrows:
        pred=traveling_delta(r['v'])
        rat=r['exp_margin_over_t']/(r['eta']*r['v']**2/r['eps']) if r['eps']>0 else ''
        asym.append(dict(v=r['v'],eps=r['eps'],t=r['t'],observed_margin_step=r['v']*r['step'],
                         tracking_delta=pred,epsilon_asymptotic_ratio=rat,
                         epsilon_over_rms=r['epsilon_over_rms']))
    effective=[r['margin'] for r in records if r['group']=='matched_effective' and r['t']==4000]
    bounded=[r['step_over_eta'] for r in records if r['optimizer']!='gd' and r['beta1']<=r['beta2'] and r['status']=='ok']
    checks=dict(torch_max_error=max(r['absolute_error'] for r in torch_rows),
                gradient_max_error=float(max(r['abs_error'] for r in grad)),
                matched_effective_margin_range=max(effective)-min(effective),
                bounded_step_max=max(bounded),
                soft_target_gd_max_error=max(r['optimum_absolute_error'] for r in warm if 'optimum_absolute_error' in r))
    assert checks['torch_max_error']<1e-9,checks
    assert checks['gradient_max_error']<1e-7,checks
    assert checks['matched_effective_margin_range']<1e-9,checks
    assert checks['bounded_step_max']<1+1e-9,checks
    assert checks['soft_target_gd_max_error']<1e-10,checks
    for name,rows in [('scalar_sweep.csv',records),('scalar_long.csv',longrows),
                     ('scalar_asymptotic.csv',asym),('scalar_torch_check.csv',torch_rows),
                     ('scalar_gradient_check.csv',grad),('scalar_flow_check.csv',flow),('scalar_warm.csv',warm)]:
        csv_write(args.out/name,rows)
    (args.out/'scalar_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    prov=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              analysis_git_hash=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              numpy=np.__version__,predefined_matrix_count=len(configs),long_matrix_count=len(longconfigs),
              tiers={'prior_example':'replication','expanded':'plan committed before expanded fits','warm':'source/theory-informed supplement'})
    (args.out/'scalar_provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
    write_summary(args.out,records,checks,warm,torch_rows,asym)
    print(json.dumps(checks),flush=True)


if __name__=='__main__':main()

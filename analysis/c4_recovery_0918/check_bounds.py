"""Numerical checks of the scalar Adam recovery theorem, not C1-C3 data.

The proof is in derivation.md. Binary CE, fixed outgoing weight a > 0,
phi(z) = exp(z)-1 on the negative half-line. All calculations are float64.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path


def restoring(z, outgoing=1.0):
    return outgoing * math.exp(z) / (1 + math.exp(outgoing * math.expm1(z)))


def constants(z0, alpha, t0, m0, v0, outgoing=1., b1=.9, b2=.999, eps=1e-8):
    assert z0 < 0 and alpha > 0 and m0 <= 0 and v0 >= 0
    assert 0 < outgoing <= 1 and t0 >= 0 and 0 < b1 < math.sqrt(b2) < 1
    h0 = restoring(z0, outgoing)
    # log(h)' is in [1-a/2, 1] for z <= 0, a <= 1.
    upper_step = alpha / math.sqrt(1-b2)
    if m0:
        upper_step *= max(1., -m0/h0) / (1-b1**(t0+1))
    r = b1 * math.exp(-upper_step)
    return dict(h0=h0, alpha=alpha, t0=t0, v0=v0, b1=b1, b2=b2,
                eps=eps, r=r, A=(1-b1)/(1-r), U=upper_step)


def lower_step(s, c):
    b2s = c['b2']**s
    den = math.sqrt((b2s*c['v0']/c['h0']**2 + 1-b2s)
                    / (1-c['b2']**(c['t0']+s))) + c['eps']/c['h0']
    return c['alpha']*c['A']*(1-c['r']**s)/den


def certificate(z0, target, alpha, t0=0, m0=0., v0=0., outgoing=1.,
                push_fraction=0., limit=5_000_000):
    """push_s = push_fraction*u_s, a predetermined external sequence."""
    c = constants(z0, alpha, t0, m0, v0, outgoing)
    total = 0.
    for s in range(1, limit+1):
        total += (1-push_fraction)*lower_step(s, c)
        if total >= target-z0:
            return s, c
    raise RuntimeError('Certificate scan limit reached; no conclusion reported')


def simulate(z0, target, alpha, t0=0, m0=0., v0=0., outgoing=1.,
             push_fraction=0., limit=5_000_000, bound_constants=None):
    z, m, v = z0, m0, v0
    min_margin = float('inf')
    # No parameter or momentum reset at t0.
    for s in range(1, limit+1):
        g = -restoring(z, outgoing)
        m = .9*m + .1*g
        v = .999*v + .001*g*g
        step = -alpha*(m/(1-.9**(t0+s)))/(math.sqrt(v/(1-.999**(t0+s)))+1e-8)
        lo = lower_step(s, bound_constants) if bound_constants else 0.
        if bound_constants:
            min_margin = min(min_margin, step-lo)
            assert step >= lo - 1e-12*max(1., abs(step))
            assert step <= bound_constants['U'] + 1e-12
        z += step - push_fraction*lo
        if z >= target:
            return dict(actual_steps=s, final_z=z,
                        min_step_margin=min_margin if bound_constants else None)
    raise RuntimeError('Simulation limit reached; right-censored run is not a hit')


def adverse_certificate(z0, target, alpha, t0, m0, v0, outgoing=1.):
    assert m0 > 0 and v0 > 0
    r0 = .9/math.sqrt(.999)
    B = alpha*m0/math.sqrt(v0)*r0/(1-r0)/(1-.9**(t0+1))
    floor = z0-B
    hmin = restoring(floor, outgoing)
    K = math.ceil(math.log(hmin/(2*(m0+hmin)))/math.log(.9))
    # Bound state at K if the threshold has not already been hit.
    vmax = .999**K*v0 + (1-.999**K)*restoring(target, outgoing)**2
    # At the first sign crossing, -m <= h; monotonicity preserves this.
    # An artificial initial -hmin gives the needed generic U, while the
    # lower-bound formula discards all favorable initial first moment.
    N, _ = certificate(floor, target, alpha, t0+K, -hmin, vmax, outgoing)
    return dict(upper_steps=K+N, warmup_steps=K, max_initial_drop=B,
                certified_floor=floor, post_warmup_v_upper=vmax)


def run():
    rows = []
    cases = [
        ('scalar_fresh', .001, 0, 0., 0., 1., 0.),
        ('aligned_105_fresh', .105, 0, 0., 0., 1., 0.),
        ('aligned_105_stale_v', .105, 30000, 0., 1e-4, 1., 0.),
        ('aligned_105_weak_output', .105, 0, 0., 0., 1e-3, 0.),
        ('aligned_105_persistent_push', .105, 0, 0., 0., 1., .5),
        ('aligned_105_favorable_m', .105, 30000, -1e-4, 1e-4, 1., 0.),
    ]
    for name, alpha, t0, m0, v0, outgoing, push in cases:
        N, c = certificate(-16., -1., alpha, t0, m0, v0, outgoing, push)
        sim = simulate(-16., -1., alpha, t0, m0, v0, outgoing, push,
                       limit=N, bound_constants=c)
        assert sim['actual_steps'] <= N
        rows.append(dict(case=name, m0=m0,
                         outgoing=outgoing, push_fraction=push,
                         upper_steps=N, **c, **sim))
    warm = adverse_certificate(-16., -1., .105, 30000, .001, 1e-4)
    sim = simulate(-16., -1., .105, 30000, .001, 1e-4, limit=warm['upper_steps'])
    assert sim['actual_steps'] <= warm['upper_steps']
    rows.append(dict(case='aligned_105_adverse_m', **warm, **sim))
    # Analytic no-uniform-bound counterexample, no simulation truncation:
    # for outgoing <= 1, h(z) <= outgoing/2 on z <= 0.
    weak = dict(outgoing=1e-12, alpha=.105, eps=1e-8, distance=15.)
    weak['necessary_steps'] = math.ceil(2*weak['eps']*weak['distance']
                                       /(weak['alpha']*weak['outgoing']))
    return dict(kind='scalar_theorem_verification_not_C1_C3_results',
                z0=-16., target=-1., beta1=.9, beta2=.999, eps=1e-8,
                cases=rows, nonzero_gradient_slow_counterexample=weak)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    result['git_hash'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    result['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result, indent=2, allow_nan=False))

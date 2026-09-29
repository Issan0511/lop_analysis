"""Verify the solved two-group logistic model; no fitted coefficients."""
from __future__ import annotations
import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.special import expit, logit

OUT = Path('results/two_group_logistic_0929')
D, P, QA, QB, H0, B0, R0 = 4., .02, .8, .2, 1., -12., 2.
DELTA = float(logit(QA)-logit(QB))
X = np.array([[D, 0.], [0., -1.], [0., 0.], [0., 1.]])
MASS = np.array([P, (1-P)/4, (1-P)/2, (1-P)/4])
TARGET = np.array([QA, QB, QB, QB])


def write_csv(name, rows):
    with (OUT/name).open('w', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def field(y, gain, mode):
    z = X @ y[:2] + y[2]
    if mode == 'saturated':
        phi = np.array([z[0], -1., -1., -1.])
        gate = np.array([1., 0., 0., 0.])
    else:
        phi = np.where(z > 0, z, np.expm1(np.minimum(z, 0)))
        gate = np.exp(np.minimum(z, 0))
    logits = y[3] + gain*phi
    e = expit(logits)-TARGET
    gz = MASS*e*gain*gate
    grad = np.r_[X.T @ gz, gz.sum(), (MASS*e).sum()]
    loss = MASS @ (np.logaddexp(0, logits)-TARGET*logits)
    return -grad, z, logits, float(loss)


def stats(z):
    mean = float(MASS @ z)
    var = float(MASS @ (z-mean)**2)
    order = np.argsort(z)
    med = float(z[order[np.searchsorted(np.cumsum(MASS[order]), .5)]])
    return dict(h=float(z[0]), body_median=med,
        body_sd=float(np.sqrt(np.array([.25, .5, .25]) @ (z[1:]-z[2])**2)),
        sd=float(np.sqrt(var)), T=float((z.max()-med)/np.sqrt(var)))


def closed_form(gain, r=R0, mobility=D*D+1):
    h = DELTA/gain-1
    b = B0+(h-H0)/mobility
    w = (h-b)/D
    c = gain+logit(QB)
    return np.array([w, r, b, c])


def rates(gain):
    aa, bb = P*QA*(1-QA), (1-P)*QB*(1-QB)
    k = D*D+1
    mat = np.array([[k*gain*gain*aa, k*gain*aa], [gain*aa, aa+bb]])
    eigen = np.linalg.eigvals(mat)
    assert np.all(eigen > 0)
    return float(eigen.min()), float(eigen.max())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, trace, checks = [], [], []
    for gain in [.1, .3, 1., 2.]:
        y0 = np.array([(H0-B0)/D, R0, B0, 0.])
        target = closed_form(gain)
        slow, fast = rates(gain)
        end = 50/slow
        times = np.r_[0., np.geomspace(.01, end, 260)]
        reduced = solve_ivp(lambda t, hc: np.array([
            -(D*D+1)*P*gain*(expit(hc[1]+gain*hc[0])-QA),
            -P*(expit(hc[1]+gain*hc[0])-QA)
            -(1-P)*(expit(hc[1]-gain)-QB)]),
            [0, end], [H0, 0.], method='DOP853', rtol=1e-10, atol=1e-12,
            t_eval=times)
        assert reduced.success
        for mode in ['saturated', 'elu']:
            sol = solve_ivp(lambda t, y: field(y, gain, mode)[0], [0, end], y0,
                method='Radau', rtol=2e-10, atol=2e-12, t_eval=times)
            assert sol.success
            yf = sol.y[:, -1]
            grad, zf, lf, loss = field(yf, gain, mode)
            analytic_z = X @ target[:2]+target[2]
            endpoint = dict(mode=mode, gain=gain, horizon=end,
                slow_rate=slow, fast_rate=fast, **stats(zf), w1=float(yf[0]),
                r=float(yf[1]), b=float(yf[2]), c=float(yf[3]), loss=loss,
                max_probability_error=float(np.max(np.abs(expit(lf)-TARGET))),
                gradient_norm=float(np.linalg.norm(grad)),
                h_prediction=float(analytic_z[0]), T_prediction=stats(analytic_z)['T'],
                max_parameter_deviation_from_saturated_prediction=float(np.max(np.abs(yf-target))))
            rows.append(endpoint)
            zall = X @ sol.y[:2]+sol.y[2]
            h = zall[0]
            physical = bool(h.min()>0 and zall[1:].max()<0)
            check = dict(mode=mode, gain=gain, success=sol.success,
                physical_branches_valid=physical,
                r_max_change=float(np.max(np.abs(sol.y[1]-R0))),
                invariant_w1_minus_d_b_error=float(np.max(np.abs(
                    sol.y[0]-D*sol.y[2]-(y0[0]-D*y0[2])))),
                full_reduced_h_max_error=float(np.max(np.abs(h-reduced.y[0]))),
                full_reduced_c_max_error=float(np.max(np.abs(sol.y[3]-reduced.y[1]))),
                h_monotonic_wrongway_max=float(max(0., np.max(
                    -np.diff(h) if gain<np.log(4) else np.diff(h)))))
            if mode == 'saturated':
                assert physical
                assert check['r_max_change'] < 1e-11
                assert check['invariant_w1_minus_d_b_error'] < 1e-9
                assert check['full_reduced_h_max_error'] < 2e-7
                assert check['h_monotonic_wrongway_max'] < 1e-7
                assert endpoint['max_parameter_deviation_from_saturated_prediction'] < 2e-7
            checks.append(check)
            for index, time in enumerate(times):
                trace.append(dict(mode=mode,gain=gain,time=float(time),
                    **stats(zall[:, index]),r=float(sol.y[1,index]),b=float(sol.y[2,index])))

    # Algebra checks use independently enumerated weighted observations.
    algebra = []
    for gain in [.1, .3, 1., 2.]:
        y = closed_form(gain)
        z = X @ y[:2]+y[2]
        gap = z[0]-y[2]
        sigma2 = R0*R0/2
        variance = (1-P)*sigma2+P*(1-P)*gap*gap
        derivative = -(D*D/(D*D+1))*(DELTA/gain**2)*(1-P)*sigma2/variance**1.5
        da = gain*1e-5
        def tail(a):
            yy = closed_form(a)
            return stats(X @ yy[:2]+yy[2])['T']
        finite = (tail(gain+da)-tail(gain-da))/(2*da)
        zz = X @ closed_form(gain, r=0)[:2]+closed_form(gain,r=0)[2]
        initial_z = X @ np.array([(H0-B0)/D,R0])+B0
        algebra.append(dict(gain=gain, formula_T=gap/np.sqrt(variance),
            enumerated_T=stats(z)['T'], formula_dT_da=derivative,
            finite_difference_dT_da=finite, derivative_error=abs(derivative-finite),
            zero_body_width_T=stats(zz)['T'], pure_scale_T=stats(initial_z/gain)['T']))
        assert abs(derivative-finite) < 1e-8
        assert abs(stats(z)['T']-gap/np.sqrt(variance)) < 1e-12
    assert np.ptp([r['zero_body_width_T'] for r in algebra]) < 1e-12
    assert np.ptp([r['pure_scale_T'] for r in algebra]) < 1e-12
    write_csv('flow_endpoints.csv', rows)
    write_csv('flow_trace.csv', trace)
    write_csv('flow_algebra.csv', algebra)
    (OUT/'flow_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    provenance=dict(git_hash=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        initial_state='common old saturated equilibrium, a_old=log(4), c0=0',
        solver_primary='Radau, rtol2e-10,atol2e-12',solver_independent='DOP853,rtol1e-10,atol1e-12',
        scope='finite four-atom geometry; no MNIST fit', horizon_rule='50 / analytic slow rate')
    (OUT/'flow_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(dict(endpoints=rows,checks=checks)))


if __name__ == '__main__':
    main()

"""Post-hoc ELU Q audit. Rates use eta=1; saved trajectories use Adam eta=.001.

Run with --data-root pointing to obsidian-research-data. No training is run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def activation(z, act):
    if act == 'LR':
        gate = np.where(z > 0, 1., .1)
        return z * gate, gate
    gate = np.exp(np.minimum(z, 0.))
    return np.where(z > 0, z, np.expm1(np.minimum(z, 0.))), gate


def rates(z, demand, act):
    phi, gate = activation(z, act)
    defect = z * gate - phi
    sv = -2 * np.mean(demand * phi, axis=0)
    sr = -2 * np.mean(demand * z * gate, axis=0)
    q = 2 * np.mean(demand * defect, axis=0)
    qdc = 2 * demand.mean(0)
    return dict(Sv=sv, Sr=sr, Q=q, Qdc=qdc, Qcomp=q-qdc,
                Qdeep=2*np.mean(demand*defect*(z <= -3), axis=0),
                Qband=2*np.mean(demand*defect*((z > -3)&(z <= 0)), axis=0),
                identity_error=sv-sr-q, alive=(z > 0).any(0),
                top=z.max(0), k=(z > 0).sum(0))


def probabilities(logits):
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def loss(logits, y):
    m = logits.max(1)
    return float(np.mean(m + np.log(np.exp(logits-m[:, None]).sum(1))
                         - logits[np.arange(len(y)), y]))


def recover(demand, v, y):
    # demand=e @ v; v contains one column per hidden unit.
    e = demand @ np.linalg.pinv(v)
    onehot = np.eye(v.shape[0])[y]
    raw_p = e + onehot
    p = np.maximum(raw_p, 1e-12)
    p /= p.sum(1, keepdims=True)
    reconstructed = (p-onehot) @ v
    return np.log(p), reconstructed, dict(
        v_condition=float(np.linalg.cond(v)),
        demand_recovery_max=float(np.max(np.abs(e@v-demand))),
        p_min_before_clip=float(raw_p.min()),
        p_row_sum_max_error=float(np.max(np.abs(raw_p.sum(1)-1))),
        clipped_fraction=float(np.mean(raw_p < 1e-12)),
        demand_after_clip_max=float(np.max(np.abs(reconstructed-demand))))


def bias_fit(logits, y):
    classes = logits.shape[1]
    target = np.bincount(y, minlength=classes)/len(y)
    offset = np.zeros(classes)
    for iteration in range(100):
        p = probabilities(logits+offset)
        g = p.mean(0)-target
        if np.max(np.abs(g)) < 1e-11:
            break
        hessian = np.diag(p.mean(0))-p.T@p/len(p)
        step = np.linalg.solve(hessian+np.ones((classes, classes))/classes, g)
        step -= step.mean()
        old = loss(logits+offset, y)
        scale = 1.
        while scale > 1e-8:
            candidate = offset-scale*step
            if loss(logits+candidate, y) <= old - 1e-4*scale*(g@step):
                offset = candidate
                break
            scale *= .5
        else:
            raise RuntimeError('Bias Newton line search failed')
    p = probabilities(logits+offset)
    err = float(np.max(np.abs(p.mean(0)-target)))
    if err > 1e-9:
        raise RuntimeError(f'Bias not balanced: {err}')
    return p-np.eye(classes)[y], dict(iterations=iteration,
        bias_gradient_max=err, loss_before=loss(logits, y),
        loss_after=loss(logits+offset, y), offset_norm=float(np.linalg.norm(offset)))


def slope_checks(logits, y, z, v, act, demand):
    base_phi, _ = activation(z, act)
    a = rates(z, demand, act)
    records = []
    for eps in (1e-4, 1e-5):
        for compensate in ((False, True) if act == 'ELU' else (False,)):
            fd = []
            for i in range(v.shape[1]):
                values = []
                for sign in (-1, 1):
                    s = sign*eps
                    new_phi, _ = activation(np.exp(-s)*z[:, i], act)
                    change = np.exp(s)*new_phi-base_phi[:, i]
                    if compensate:
                        change += np.expm1(s)
                    values.append(loss(logits+change[:, None]*v[:, i], y))
                fd.append((values[1]-values[0])/(2*eps))
            expected = -.5*a['Qcomp' if compensate else 'Q']
            error = np.asarray(fd)-expected
            records.append(dict(eps=eps, compensate=compensate,
                max_absolute_error=float(np.max(np.abs(error))),
                l1_relative_error=float(np.sum(np.abs(error))/max(np.sum(np.abs(expected)), 1e-30)),
                max_expected_slope=float(np.max(np.abs(expected)))))
    return records


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def aggregate(rows):
    ans = []
    keys = sorted(set((r['act'], r['seed'], r['phase']) for r in rows))
    for act, seed, phase in keys:
        block = [r for r in rows if (r['act'], r['seed'], r['phase']) == (act, seed, phase)]
        for population in ('all', 'alive'):
            selected = [r for r in block if population == 'all' or r['alive']]
            if not selected:
                continue
            get = lambda k: np.array([r[k] for r in selected], dtype=float)
            q, dc, comp = get('Q'), get('Qdc'), get('Qcomp')
            sv, sr = get('Sv'), get('Sr')
            signs = lambda a,b: float(np.mean(np.sign(a) == np.sign(b)))
            result = dict(act=act, seed=seed, phase=phase, population=population, n=len(selected),
                Q_median=float(np.median(q)), Q_mean=float(np.mean(q)),
                Q_positive_fraction=float(np.mean(q > 1e-12)),
                Sr_median=float(np.median(sr)), Sv_median=float(np.median(sv)),
                Sr_positive_fraction=float(np.mean(sr > 1e-12)),
                Sv_positive_fraction=float(np.mean(sv > 1e-12)),
                Qcomp_median=float(np.median(comp)), Qcomp_positive_fraction=float(np.mean(comp > 1e-12)),
                Q_Qdc_sign_agreement=signs(q, dc),
                Q_Qcomp_sign_agreement=signs(q, comp),
                Qcomp_to_Q_l1=float(np.sum(np.abs(comp))/max(np.sum(np.abs(q)), 1e-30)),
                Qdeep_fraction_of_component_l1=float(np.sum(np.abs(get('Qdeep')))/max(np.sum(np.abs(get('Qdeep')))+np.sum(np.abs(get('Qband'))), 1e-30)),
                identity_max=float(np.max(np.abs(get('identity_error')))))
            if phase == 'end' and act == 'ELU':
                result['q_median'] = float(np.median(get('ratio')))
                result['raw_qdot_positive_fraction'] = float(np.mean(get('qdot_sgd') > 0))
            ans.append(result)
    return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, checks, states, joins, trajectories, transitions = [], [], [], [], [], []
    source_paths = set()
    for act in ('ELU', 'LR'):
        for seed in (0,1):
            path = args.data_root/f'valley_anchor_1layer_0928/runs/{act}_s{seed}.npz'
            source_paths.add(path)
            with np.load(path) as source:
                data = {k:source[k] for k in source.files}
            norms = None
            if act == 'ELU':
                normpath = args.data_root/f'elu_alpha_ladder_1layer_0925/runs/ELUA1m1_s{seed}_v4.npz'
                source_paths.add(normpath)
                with np.load(normpath) as source:
                    norms = {k:source[k] for k in source.files}
                for task in (10,30):
                    ix = list(norms['snap_tasks']).index(task)
                    for phase in ('sw', '200', 'end'):
                        same = np.array_equal(data[f't{task}_z_{phase}'], norms[f'z_{phase}'][ix])
                        joins.append(dict(seed=seed, task=task, phase=phase, bit_identical=same))
                        if not same:
                            raise RuntimeError('Cannot join unmatched norm trajectory')
            for task in data['snap']:
                task = int(task)
                end_v = data[f't{task}_W2'].astype(float)
                for phase in ('sw', '200', 'end'):
                    z = data[f't{task}_z_{phase}'].astype(float)
                    demand = data[f't{task}_D_{phase}'].astype(float)
                    a = rates(z, demand, act)
                    if phase == 'end' and norms is not None:
                        # Extra per-task checks cover tasks 20,40, without full saved z overlap.
                        idx = task-1
                        error = float(np.max(np.abs(z.max(0)-norms['task_zmax'][idx])))
                        joins.append(dict(seed=seed, task=task, phase='end_top', max_error=error))
                        if error > 1e-4:
                            raise RuntimeError('Per-task maxima mismatch')
                        r = norms['task_wnorm'][idx].astype(float)**2+norms['task_bias'][idx].astype(float)**2
                        v2 = (end_v**2).sum(0)
                        v2centered = ((end_v-end_v.mean(0))**2).sum(0)
                        a.update(R=r, V=v2, ratio=v2/r, ratio_centered=v2centered/r,
                                 qdot_sgd=(a['Sv']-(v2/r)*a['Sr'])/r)
                        for unit in range(len(r)):
                            trajectories.append(dict(seed=seed, task=task, unit=unit,
                                R=r[unit], V=v2[unit], ratio=v2[unit]/r[unit],
                                ratio_centered=v2centered[unit]/r[unit], top=a['top'][unit],
                                alive=bool(a['alive'][unit]), Q=a['Q'][unit],
                                qdot_sgd=a['qdot_sgd'][unit], Sv=a['Sv'][unit], Sr=a['Sr'][unit]))
                    for unit in range(z.shape[1]):
                        row=dict(act=act, seed=seed, task=task, phase=phase, unit=unit)
                        row.update({k:(bool(v[unit]) if k=='alive' else float(v[unit])) for k,v in a.items()})
                        rows.append(row)
                    if phase != 'end':
                        continue
                    y = data[f't{task}_y'].astype(int)
                    logits, recovered, recovery = recover(demand, end_v, y)
                    if recovery['demand_after_clip_max'] > 5e-6:
                        raise RuntimeError(f'Inaccurate probability recovery: {recovery}')
                    for check in slope_checks(logits, y, z, end_v, act, recovered):
                        checks.append(dict(act=act, seed=seed, task=task, **check))
                    state=dict(act=act, seed=seed, task=task, **recovery)
                    if act == 'ELU':
                        residual, bias = bias_fit(logits, y)
                        balanced = rates(z, residual@end_v, act)
                        state.update(bias)
                        state.update(Q_l1_before=float(np.abs(a['Q']).sum()),
                                     Q_l1_after_bias_fit=float(np.abs(balanced['Q']).sum()),
                                     Q_sign_flip_fraction=float(np.mean(np.sign(a['Q'])!=np.sign(balanced['Q']))))
                        for unit in range(z.shape[1]):
                            row=dict(act=act, seed=seed, task=task, phase='bias_fit', unit=unit)
                            row.update({k:(bool(v[unit]) if k=='alive' else float(v[unit])) for k,v in balanced.items()})
                            rows.append(row)
                    states.append(state)
                    print(f'{act} seed={seed} task={task} audited', flush=True)
    for seed in (0,1):
        lookup={(r['task'],r['unit']):r for r in trajectories if r['seed']==seed}
        for task in (10,20,30):
            for unit in range(100):
                a,b=lookup[task,unit],lookup[task+10,unit]
                transitions.append(dict(seed=seed, task=task, unit=unit,
                    alive_start=a['alive'], alive_end=b['alive'], Q_start=a['Q'],
                    qdot_sgd_start=a['qdot_sgd'], Sv_start=a['Sv'], Sr_start=a['Sr'],
                    delta_R=b['R']-a['R'], delta_V=b['V']-a['V'],
                    delta_B=(b['V']-b['R'])-(a['V']-a['R']),
                    delta_ratio=b['ratio']-a['ratio']))
    summary=aggregate(rows)
    for name, content in [('units.csv',rows), ('summary.csv',summary),
                          ('states.csv',states),('finite_difference.csv',checks),
                          ('norm_trajectory.csv',trajectories),('transitions.csv',transitions)]:
        write_csv(args.out/name, content)
    (args.out/'join_checks.json').write_text(json.dumps(joins,indent=2)+'\n')
    provenance=dict(tier='posthoc', data_root=str(args.data_root), numpy=np.__version__,
        analysis_git_hash=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rates_eta=1, trained_optimizer='Adam, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8',
        actual_optimizer_moments_available=False,
        inputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(source_paths)])
    (args.out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    lines=['# ELU Q audit — post-hoc numerical output','',
           'Stored Adam states; gradients/rates below are Euclidean gradient-flow diagnostics with eta=1.',
           'Each seed/phase has four task states (10,20,30,40). No independent-unit statistical claim.', '',
           '| act | seed | phase | population | n | median Q | Q positive | median Sr | median Sv | Qcomp/Q L1 | Q/DC sign agree |',
           '|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['act']} | {r['seed']} | {r['phase']} | {r['population']} | {r['n']} | {r['Q_median']:.6g} | {r['Q_positive_fraction']:.3f} | {r['Sr_median']:.6g} | {r['Sv_median']:.6g} | {r['Qcomp_to_Q_l1']:.4g} | {r['Q_Qdc_sign_agreement']:.3f} |")
    lines += ['', '## Checks', '', f"Maximum Sv-Sr-Q residual: {max(r['identity_max'] for r in summary):.3g}.",
              f"Maximum ELU finite-difference relative L1 error: {max(r['l1_relative_error'] for r in checks if r['act']=='ELU'):.3g}.",
              f"Maximum LR finite-difference absolute error: {max(r['max_absolute_error'] for r in checks if r['act']=='LR'):.3g}.",
              '', '## Limits', '',
              'A local gradient at a stored state is not an integrated Adam update. Endpoint norm differences are actual saved-trajectory differences, but intermediate optimizer moments are missing.',
              'Output-bias fitting and compensated rescaling are local diagnostic interventions at fixed hidden/readout weights, not natural trajectory decompositions.',
              'The end-state norm join matches saved preactivations bit-for-bit at tasks 10 and 30, and per-unit maxima at all four tasks.',
              'Mathematical ELU derivatives are used; the float32 expm1 training implementation has an exact-zero negative-tail derivative.',
              'See interpretation.md for the derivation, selected comparisons, and implications.']
    (args.out/'summary.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()

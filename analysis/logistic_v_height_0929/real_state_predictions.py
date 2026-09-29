"""Read-only, no-fit predictions from reconstructed real RL-MNIST states.

Uses the original source's float32 ELU features/gates, then evaluates the
softmax and Adam one-step response in float64. Native arrays provide an
independent implementation/precision check, not fitted coefficients.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def scalar(data, key):
    a = np.asarray(data[key])
    if a.size != 1:
        raise ValueError(f'{key} must be scalar, received {a.shape}')
    return float(a.item())


def source_features(X, W, b):
    """Match source float32 matmul and expm1+1 gate, including deep zeros."""
    with torch.no_grad():
        xt = torch.as_tensor(X, dtype=torch.float32)
        wt = torch.as_tensor(W, dtype=torch.float32)
        bt = torch.as_tensor(b, dtype=torch.float32)
        z = xt @ wt.T + bt
        neg = torch.expm1(z.clamp(max=0))
        phi = torch.where(z > 0, z, neg)
        gate = torch.where(z > 0, torch.ones_like(z), neg + 1)
    return tuple(t.numpy().astype(np.float64) for t in (z, phi, gate))


def softmax(logits):
    x = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=1, keepdims=True)


def gradients(X, phi, gate, V, c, y, gain):
    """a multiplies every column of the original readout V."""
    h = phi @ V.T
    logits = c + gain * h
    p = softmax(logits)
    e = p.copy()
    e[np.arange(len(y)), y] -= 1
    demand = e @ V
    coeff = gain * gate * demand / len(y)
    gw = coeff.T @ X
    gb = coeff.sum(axis=0)
    gc = e.mean(axis=0)
    hp = p * (h - (p * h).sum(axis=1, keepdims=True))
    demand_prime = hp @ V
    coeff_prime = gate * (demand + gain * demand_prime) / len(y)
    return dict(gw=gw, gb=gb, gc=gc, logits=logits, p=p, demand=demand,
                coeff=coeff, dgw=coeff_prime.T @ X,
                dgb=coeff_prime.sum(axis=0))


def adam(g, M, S, step, lr, b1, b2, eps):
    """Exact real-arithmetic standard Adam step, evaluated in float64."""
    t = int(step) + 1
    ac = 1 - b1 ** t
    bc = 1 - b2 ** t
    mp = b1 * M + (1 - b1) * g
    sp = b2 * S + (1 - b2) * g * g
    mh = mp / ac
    rms = np.sqrt(sp / bc)
    denominator = rms + eps
    zero = denominator == 0
    if np.any(zero & (mh != 0)):
        raise ValueError('nonzero first moment over zero second moment')
    # All-zero coordinates have no update for any epsilon>0. Keep that
    # continuous extension for the epsilon=0 invariance-only diagnostic.
    invden = np.divide(1., denominator, out=np.zeros_like(g), where=~zero)
    update = -lr * mh * invden
    sensitivity = (1 - b1) / ac * invden
    nz = rms > 0
    correction = np.zeros_like(g)
    correction[nz] = (mh[nz] * (1-b2) * g[nz] / (bc*rms[nz])
                      * invden[nz] ** 2)
    sensitivity -= correction
    return dict(update=update, M=mp, S=sp, P=invden, J=sensitivity,
                old_update=-lr*b1/ac*M*invden,
                current_update=-lr*(1-b1)/ac*g*invden, ac=ac, bc=bc)


def tail_metrics(X, W, b, want_q=False):
    """Original source body=lower median; standard deviation uses ddof=1."""
    z = X @ W.T + b
    n = len(X)
    order = np.argsort(z, axis=0, kind='stable')
    top_id = order[-1]
    med_id = order[(n-1)//2]
    unit = np.arange(W.shape[0])
    top = z[top_id, unit]
    body = z[med_id, unit]
    zc = z-z.mean(axis=0, keepdims=True)
    sd = np.sqrt((zc*zc).sum(axis=0)/(n-1))
    if np.any(sd <= 0):
        raise ValueError('tail requires nonzero sample standard deviation')
    T = (top-body)/sd
    result = dict(z=z, top=top, body=body, sd=sd, T=T,
                  top_id=top_id, med_id=med_id, alive=top>0)
    if want_q:
        xc = X-X.mean(axis=0, keepdims=True)
        sigma_w = zc.T @ xc / (n-1)
        result['q'] = ((X[top_id]-X[med_id])/sd[:, None]
                       - (T/sd**2)[:, None]*sigma_w)
    return result


def error_metrics(actual, predicted):
    d = np.asarray(actual, dtype=float)-predicted
    return dict(max_abs=float(np.max(np.abs(d))),
                relative_l1=float(np.abs(d).sum()/max(np.abs(actual).sum(), 1e-300)))


def get_gain(native, prefix):
    key = prefix+'__gain'
    if key in native:
        return scalar(native, key)
    match = re.search(r'_c([0-9]+(?:p[0-9]+)?)$', prefix)
    if not match:
        raise ValueError(f'Cannot recover gain from {prefix}; add __gain')
    return float(match.group(1).replace('p', '.'))


def analyze_seed(seed, state, native):
    torch.set_num_threads(1)
    X = state['X'].astype(float)
    W = state['W1'].astype(float)
    b = state['b1'].astype(float)
    V = state['W2'].astype(float)
    c = state['b2'].astype(float)
    M, S = (state[k].astype(float) for k in ('m_W1', 'q_W1'))
    Mb, Sb = (state[k].astype(float) for k in ('m_b1', 'q_b1'))
    step, stepb = scalar(state, 'step_W1'), scalar(state, 'step_b1')
    lr, eps = scalar(state, 'lr'), scalar(state, 'eps')
    b1, b2 = map(float, state['betas'])
    yfull = state['y_t51'].astype(np.int64)
    ids = state['order_t51'][0].astype(np.int64)
    Xb, y = X[ids], yfull[ids]
    zs, phis, gates = source_features(state['X'], state['W1'], state['b1'])
    zb, phib, gateb = source_features(state['X'][ids], state['W1'], state['b1'])
    base = tail_metrics(X, W, b, want_q=True)
    q0 = base['q']
    ref = gradients(Xb, phib, gateb, V, c, y, 1.)
    ref_aw=adam(ref['gw'],M,S,step,lr,b1,b2,eps)
    ref_ab=adam(ref['gb'],Mb,Sb,stepb,lr,b1,b2,eps)
    ref_post=tail_metrics(X,W+ref_aw['update'],b+ref_ab['update'])
    ref_old_proj=np.einsum('ij,ij->i',q0,ref_aw['old_update'])
    ref_current_proj=np.einsum('ij,ij->i',q0,ref_aw['current_update'])
    prefixes = sorted(k[:-len('__gradient_W1')] for k in native
                      if (k.startswith('A_') or k.startswith('REF_'))
                      and k.endswith('__gradient_W1'))
    if len(prefixes) != 3:
        raise ValueError(f'Expected three A/REF retained arms, got {prefixes}')
    ref_prefix=next(p for p in prefixes if get_gain(native,p)==1.)
    ref_native=tail_metrics(X,native[ref_prefix+'__W1'].astype(float),native[ref_prefix+'__b1'].astype(float))
    unit_rows, input_rows, check_rows = [], [], []
    for prefix in prefixes:
        gain = get_gain(native, prefix)
        actual = gradients(Xb, phib, gateb, V, c, y, gain)
        aw = adam(actual['gw'], M, S, step, lr, b1, b2, eps)
        ab = adam(actual['gb'], Mb, Sb, stepb, lr, b1, b2, eps)
        predicted = tail_metrics(X, W+aw['update'], b+ab['update'], want_q=True)
        nativeW, nativeb = native[prefix+'__W1'].astype(float), native[prefix+'__b1'].astype(float)
        native_m = tail_metrics(X, nativeW, nativeb)
        native_gw = native[prefix+'__gradient_W1'].astype(float)
        native_gb = native[prefix+'__gradient_b1'].astype(float)
        native_aw = adam(native_gw, M, S, step, lr, b1, b2, eps)
        native_ab = adam(native_gb, Mb, Sb, stepb, lr, b1, b2, eps)
        native_gradient_prediction = tail_metrics(X, W+native_aw['update'], b+native_ab['update'])

        scaled = gain*ref['gw']
        scaled_b = gain*ref['gb']
        cf = adam(scaled, M, S, step, lr, b1, b2, eps)
        cfb = adam(scaled_b, Mb, Sb, stepb, lr, b1, b2, eps)
        cf_m = tail_metrics(X, W+cf['update'], b+cfb['update'])

        # Consistent rescaling, epsilon=0: exact baseline-update invariance.
        scaled0 = adam(scaled, gain*M, gain*gain*S, step, lr, b1, b2, 0.)
        ref0 = adam(ref['gw'], M, S, step, lr, b1, b2, 0.)

        # Derivative-only contrast; features remain source phi, gates differ.
        true_gate = np.where(zb>0, 1., np.exp(np.minimum(zb, 0.)))
        exp_grad = gradients(Xb, phib, true_gate, V, c, y, gain)
        exp_aw = adam(exp_grad['gw'], M, S, step, lr, b1, b2, eps)
        exp_ab = adam(exp_grad['gb'], Mb, Sb, stepb, lr, b1, b2, eps)
        exp_m = tail_metrics(X, W+exp_aw['update'], b+exp_ab['update'])

        # Exact infinitesimal response, at the updated tail gradient.
        du_da = -lr*aw['J']*actual['dgw']
        dT_da = np.einsum('ij,ij->i', predicted['q'], du_da)
        h = gain*1e-4
        plus = gradients(Xb, phib, gateb, V, c, y, gain+h)
        minus = gradients(Xb, phib, gateb, V, c, y, gain-h)
        fdgw = (plus['gw']-minus['gw'])/(2*h)
        plus_aw = adam(plus['gw'], M, S, step, lr, b1, b2, eps)
        minus_aw = adam(minus['gw'], M, S, step, lr, b1, b2, eps)
        plus_ab = adam(plus['gb'], Mb, Sb, stepb, lr, b1, b2, eps)
        minus_ab = adam(minus['gb'], Mb, Sb, stepb, lr, b1, b2, eps)
        plus_m = tail_metrics(X, W+plus_aw['update'], b+plus_ab['update'])
        minus_m = tail_metrics(X, W+minus_aw['update'], b+minus_ab['update'])
        fdT = (plus_m['T']-minus_m['T'])/(2*h)

        Pq = aw['P']*q0
        align = Xb @ Pq.T
        current_coeff = -lr*(1-b1)/aw['ac']*actual['coeff']
        input_projection = current_coeff*align
        old_projection = np.einsum('ij,ij->i', q0, aw['old_update'])
        current_projection = input_projection.sum(axis=0)
        linear_projection = np.einsum('ij,ij->i', q0, aw['update'])
        fullp = softmax(c+gain*(phis@V.T))
        fulle = fullp.copy()
        fulle[np.arange(len(X)), yfull] -= 1
        full_demand = fulle @ V
        unit = np.arange(len(W))
        top_ids = base['top_id']
        top_alignment = np.einsum('ij,ij->i', X[top_ids], Pq)
        top_demand = full_demand[top_ids, unit]
        top_gate = gates[top_ids, unit]
        top_drive = -gain*top_gate*top_demand

        # Exact first-batch moments over a fresh iid uniform label draw.
        # These are gradient moments, not an expectation of nonlinear Adam.
        classes=V.shape[0]
        pi=np.full(classes, 1/classes)
        p0=softmax(c[None,:])[0]
        hlog=phib@V.T
        h0h=p0[None,:]*(hlog-(hlog*p0[None,:]).sum(axis=1,keepdims=True))
        g_label_mean=(gain*gateb*((actual['p']-pi)@V)/len(ids)).T@Xb
        g_label_order1=(gateb*((p0-pi)@V)[None,:]/len(ids)).T@Xb
        g_label_order2=(gateb*(h0h@V)/len(ids)).T@Xb
        v_label_var=((V-V.mean(axis=0,keepdims=True))**2).mean(axis=0)
        xnorm2=(Xb*Xb).sum(axis=1)
        label_noise_norm=(gain/len(ids))*np.sqrt(v_label_var*((gateb*gateb)*xnorm2[:,None]).sum(axis=0))
        unprocessed_alignment=Xb@q0.T
        label_tail_gradient=np.einsum('ij,ij->i',q0,g_label_mean)
        label_tail_std=(gain/len(ids))*np.sqrt(v_label_var*((gateb*unprocessed_alignment)**2).sum(axis=0))
        label_series=gain*g_label_order1+gain*gain*g_label_order2

        # Native backprop reconstruction isolates forward/CE rounding from
        # a hidden-gradient formula error.
        reconstructed=None
        if prefix+'__gradient_logits' in native:
            with torch.no_grad():
                eg=torch.as_tensor(native[prefix+'__gradient_logits'],dtype=torch.float32)
                vg=torch.as_tensor(state['W2'],dtype=torch.float32)*gain
                gz=(eg@vg)*torch.as_tensor(gateb,dtype=torch.float32)
                reconstructed=(gz.T@torch.as_tensor(state['X'][ids],dtype=torch.float32)).numpy().astype(float)

        checks = dict(seed=seed, gain=gain, native_prefix=prefix,
                      gate_zero_fraction=float((gateb==0).mean()),
                      gate_zero_but_exp_positive_fraction=float(((gateb==0)&(true_gate>0)).mean()),
                      derivative_only_gradient_relative_l1=error_metrics(actual['gw'], exp_grad['gw'])['relative_l1'],
                      derivative_only_tail_change_max_abs=float(np.max(np.abs(exp_m['T']-predicted['T']))),
                      parameter_max_abs_native=error_metrics(nativeW, W+aw['update'])['max_abs'],
                      parameter_max_abs_with_saved_gradient=error_metrics(nativeW, W+native_aw['update'])['max_abs'],
                      tail_max_abs_native=float(np.max(abs(native_m['T']-predicted['T']))),
                      tail_max_abs_with_saved_gradient=float(np.max(abs(native_m['T']-native_gradient_prediction['T']))),
                      tail_radial_orthogonality_max_abs=float(np.max(abs(np.einsum('ij,ij->i', q0, W)))),
                      projected_sum_max_abs=float(np.max(abs(old_projection+current_projection-linear_projection))),
                      matched_rescale_epsilon0_max_abs=float(np.max(abs(scaled0['update']-ref0['update']))),
                      gradient_derivative_fd_relative_l1=error_metrics(fdgw, actual['dgw'])['relative_l1'],
                      tail_derivative_fd_max_abs=float(np.max(abs(fdT-dT_da))))
        checks['uniform_label_p0_max_deviation']=float(np.max(abs(p0-pi)))
        checks['uniform_label_mean_series_relative_l1']=error_metrics(g_label_mean,label_series)['relative_l1']
        if reconstructed is not None:
            checks['native_backprop_reconstruction_max_abs']=error_metrics(native_gw,reconstructed)['max_abs']
        for saved,observed in [('z_batch',zb),('gate_batch',gateb)]:
            if prefix+'__'+saved in native:
                checks[saved+'_reconstruction_max_abs']=error_metrics(native[prefix+'__'+saved],observed)['max_abs']
        for key, observed, pred in [('gradient', native_gw, actual['gw']),
                                    ('bias_gradient', native_gb, actual['gb'])]:
            checks.update({key+'_'+k: v for k,v in error_metrics(observed, pred).items()})
        if prefix+'__logits' in native:
            checks['logits_max_abs'] = error_metrics(native[prefix+'__logits'], actual['logits'])['max_abs']
        check_rows.append(checks)

        for j in range(len(W)):
            row = dict(seed=seed, gain=gain, unit=j, alive_pre=bool(base['alive'][j]),
                tail_pre=float(base['T'][j]), tail_predicted=float(predicted['T'][j]),
                tail_native=float(native_m['T'][j]),
                delta_tail_actual=float(predicted['T'][j]-base['T'][j]),
                delta_tail_native=float(native_m['T'][j]-base['T'][j]),
                delta_tail_common_scale=float(cf_m['T'][j]-base['T'][j]),
                delta_tail_residual_contrast=float(predicted['T'][j]-cf_m['T'][j]),
                gain_effect_tail=float(predicted['T'][j]-ref_post['T'][j]),
                gain_effect_tail_native=float(native_m['T'][j]-ref_native['T'][j]),
                gain_effect_tail_common_scale=float(cf_m['T'][j]-ref_post['T'][j]),
                gain_effect_old_projection=float(old_projection[j]-ref_old_proj[j]),
                gain_effect_current_projection=float(current_projection[j]-ref_current_proj[j]),
                delta_tail_exp_gate=float(exp_m['T'][j]-base['T'][j]),
                delta_raw_top_actual=float(predicted['top'][j]-base['top'][j]),
                delta_raw_top_native=float(native_m['top'][j]-base['top'][j]),
                delta_body_actual=float(predicted['body'][j]-base['body'][j]),
                old_moment_projection=float(old_projection[j]),
                current_gradient_projection=float(current_projection[j]),
                total_linear_projection=float(linear_projection[j]),
                finite_minus_linear_tail=float(predicted['T'][j]-base['T'][j]-linear_projection[j]),
                current_open_projection=float(input_projection[:,j][zb[:,j]>0].sum()),
                current_band_projection=float(input_projection[:,j][(zb[:,j]<=0)&(zb[:,j]>-8)].sum()),
                current_deep_projection=float(input_projection[:,j][zb[:,j]<=-8].sum()),
                top_input_id=int(top_ids[j]), top_in_batch=bool(np.any(ids==top_ids[j])),
                all_batch_gates_zero=bool(np.all(gateb[:,j]==0)),
                open_inputs_in_batch=int((zb[:,j]>0).sum()),
                top_gate=float(top_gate[j]), top_demand=float(top_demand[j]),
                top_descent_drive=float(top_drive[j]), top_alignment_P=float(top_alignment[j]),
                top_selective_sign_positive=bool(top_drive[j]*top_alignment[j]>0),
                top_positive_drive_and_alignment=bool(top_drive[j]>0 and top_alignment[j]>0),
                d_tail_da=float(dT_da[j]), fd_tail_da=float(fdT[j]),
                uniform_label_mean_gradient_norm=float(np.linalg.norm(g_label_mean[j])),
                uniform_label_noise_rms_norm=float(label_noise_norm[j]),
                uniform_label_mean_tail_gradient=float(label_tail_gradient[j]),
                uniform_label_tail_gradient_std=float(label_tail_std[j]),
                uniform_label_order1_gradient_norm=float(np.linalg.norm(g_label_order1[j])),
                uniform_label_order2_gradient_norm=float(np.linalg.norm(g_label_order2[j])),
                uniform_label_series_error_norm=float(np.linalg.norm(g_label_mean[j]-label_series[j])),
                median_rank_changed=bool(base['med_id'][j]!=predicted['med_id'][j]),
                maximum_rank_changed=bool(base['top_id'][j]!=predicted['top_id'][j]))
            unit_rows.append(row)
            for bi, sample in enumerate(ids):
                input_rows.append(dict(seed=seed, gain=gain, unit=j, batch_pos=bi,
                    sample_id=int(sample), z=float(zb[bi,j]), gate=float(gateb[bi,j]),
                    demand=float(actual['demand'][bi,j]),
                    current_update_coefficient=float(current_coeff[bi,j]),
                    alignment_P=float(align[bi,j]), current_tail_projection=float(input_projection[bi,j]),
                    is_top=bool(sample==top_ids[j]), is_open=bool(zb[bi,j]>0),
                    is_deep=bool(zb[bi,j]<=-8)))
    return unit_rows, input_rows, check_rows


def summarize(units):
    rows=[]
    for seed in sorted({r['seed'] for r in units}):
        for gain in sorted({r['gain'] for r in units}):
            for cohort in ['all_units', 'alive_pre']:
                group=[r for r in units if r['seed']==seed and r['gain']==gain
                       and (cohort=='all_units' or r['alive_pre'])]
                if not group:
                    continue
                a=np.array([r['delta_tail_actual'] for r in group])
                n=np.array([r['delta_tail_native'] for r in group])
                cf=np.array([r['delta_tail_common_scale'] for r in group])
                old=np.array([r['old_moment_projection'] for r in group])
                cur=np.array([r['current_gradient_projection'] for r in group])
                effect=np.array([r['gain_effect_tail'] for r in group])
                effect_native=np.array([r['gain_effect_tail_native'] for r in group])
                effect_cf=np.array([r['gain_effect_tail_common_scale'] for r in group])
                effect_resolved=(np.abs(effect)>1e-8)&(np.abs(effect)>5*np.abs(effect-effect_native))
                significant=(np.abs(a)>1e-8)&(np.abs(a)>5*np.abs(a-n))
                def sign_agreement(x,y):
                    return float(np.mean(np.sign(x[significant])==np.sign(y[significant]))) if np.any(significant) else float('nan')
                row=dict(seed=seed, gain=gain, cohort=cohort, units=len(group),
                    median_delta_tail=float(np.median(a)), median_native_delta_tail=float(np.median(n)),
                    median_common_scale_delta_tail=float(np.median(cf)),
                    actual_positive_fraction=float(np.mean(a>0)),
                    absolute_tail_mass=float(np.abs(a).sum()),
                    residual_contrast_relative_l1=float(np.abs(a-cf).sum()/max(np.abs(a).sum(),1e-300)),
                    common_scale_sign_agreement=sign_agreement(a,cf),
                    native_sign_agreement=sign_agreement(a,n),
                    sign_comparison_precision_resolved_count=int(significant.sum()),
                    old_absolute_projection=float(np.abs(old).sum()),
                    current_absolute_projection=float(np.abs(cur).sum()),
                    old_median_projection=float(np.median(old)), current_median_projection=float(np.median(cur)),
                    top_positive_drive_alignment_fraction=float(np.mean([r['top_positive_drive_and_alignment'] for r in group])),
                    top_in_batch_fraction=float(np.mean([r['top_in_batch'] for r in group])),
                    median_rank_changed_fraction=float(np.mean([r['median_rank_changed'] for r in group])),
                    max_rank_changed_fraction=float(np.mean([r['maximum_rank_changed'] for r in group])))
                row.update(gain_effect_median=float(np.median(effect)),
                    gain_effect_l1=float(np.abs(effect).sum()),
                    gain_effect_residual_contrast_relative_l1=float(np.abs(effect-effect_cf).sum()/np.abs(effect).sum()) if np.abs(effect).sum()>0 else 0.,
                    gain_effect_sign_agreement=float(np.mean(np.sign(effect[effect_resolved])==np.sign(effect_cf[effect_resolved]))) if np.any(effect_resolved) else float('nan'),
                    gain_effect_positive_fraction_resolved=float(np.mean(effect[effect_resolved]>0)) if np.any(effect_resolved) else float('nan'),
                    gain_effect_resolved_count=int(effect_resolved.sum()),
                    gain_effect_old_projection_l1=float(sum(abs(r['gain_effect_old_projection']) for r in group)),
                    gain_effect_current_projection_l1=float(sum(abs(r['gain_effect_current_projection']) for r in group)))
                prior_noise=np.array([r['uniform_label_noise_rms_norm'] for r in group])
                prior_mean=np.array([r['uniform_label_mean_gradient_norm'] for r in group])
                prior_nonzero=prior_noise>0
                order1=sum(r['uniform_label_order1_gradient_norm'] for r in group)
                order2=sum(r['uniform_label_order2_gradient_norm'] for r in group)
                row.update(all_batch_gates_zero_fraction=float(np.mean([r['all_batch_gates_zero'] for r in group])),
                    uniform_label_mean_noise_ratio_median=float(np.median(prior_mean[prior_nonzero]/prior_noise[prior_nonzero])) if np.any(prior_nonzero) else float('nan'),
                    uniform_label_noise_nonzero_count=int(prior_nonzero.sum()),
                    uniform_label_order1_over_order2_norm_sum=float(order1/(gain*order2)) if order2>0 else float('nan'))
                finite_error=np.array([r['finite_minus_linear_tail'] for r in group])
                row.update(finite_projection_error_relative_l1=float(np.abs(finite_error).sum()/max(np.abs(a).sum(),1e-300)),
                           finite_projection_error_max_abs=float(np.max(np.abs(finite_error))))
                for field in ['current_open_projection','current_band_projection','current_deep_projection']:
                    vals=np.array([r[field] for r in group])
                    row[field+'_median']=float(np.median(vals))
                    row[field+'_absolute_sum']=float(np.abs(vals).sum())
                rows.append(row)
    return rows


def report(out, summary, checks):
    lines=['# 実RL-MNIST状態からの一歩の予測と機構対照','',
        'task 50 の同じ重み・Adam状態、task 51 の同じ最初のミニバッチから、倍率0.1/1/10の保持Adam一歩を係数合わせなしで予測した。以下は最初の一歩の識別であり、3課題後の機構の同定ではない。', '',
        'sourceのfloat32前活性・expm1+1ゲートを固定し、softmaxとAdamの式はfloat64で評価した。tailは元の定義（lower median、標本標準偏差）を使う。nativeパラメータもfloat64へ読み直して同じ評価を行い、状態の丸めと統計計算の丸めを区別した。', '',
        '## gain1に対する一歩の介入効果（介入前に開いたunitの固定集合）','',
        '| seed | gain | median effect | effect L1 | 残差対照L1/effect L1 | 共通倍率の符号一致 | 精度基準を満たすunit数 |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['cohort']=='alive_pre' and r['gain']!=1:
            lines.append(f"| {r['seed']} | {r['gain']:g} | {r['gain_effect_median']:.6g} | {r['gain_effect_l1']:.6g} | {r['gain_effect_residual_contrast_relative_l1']:.4g} | {r['gain_effect_sign_agreement']:.4g} | {r['gain_effect_resolved_count']} |")
    lines += ['', '介入効果は同じ状態からgain1で進めた一歩との差。unitごとの符号が混在し、一部は今回のバッチで全gateが0となって倍率効果が0になる。したがって中央値だけで応答がないとは判断せず、固定集合のL1と精度基準を満たすunit数を併記した。', '',
        '## 絶対的な一歩のtail移動（共通の過去driftも含む）','',
        '| seed | gain | median ΔT | native | 共通倍率のみ | 残差対照L1/実応答L1 | 符号一致 |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['cohort']=='alive_pre':
            lines.append(f"| {r['seed']} | {r['gain']:g} | {r['median_delta_tail']:.6g} | {r['median_native_delta_tail']:.6g} | {r['median_common_scale_delta_tail']:.6g} | {r['residual_contrast_relative_l1']:.4g} | {r['common_scale_sign_agreement']:.4g} |")
    lines += ['', '共通倍率対照はgain1の同じ状態の勾配をa倍したもので、保持したモーメントは変えない。「残差対照L1」は実際の予測とこの対照の差の絶対値合計であり、説明分散や媒介割合ではない。差が大きい場合、共通倍率のみの説明では一歩を予測できない。符号比較は|ΔT|>10⁻⁸かつnativeとの値の差の5倍より大きいunitに限定した便宜的精度診断である。', '',
        '## 独立検算と精度','',
        f"- native勾配との最大絶対誤差: {max(r['gradient_max_abs'] for r in checks):.6g}。",
        f"- native更新後W1との最大絶対誤差: {max(r['parameter_max_abs_native'] for r in checks):.6g}。",
        f"- nativeとのtail値の最大差: {max(r['tail_max_abs_native'] for r in checks):.6g}。",
        f"- source勾配を直接与えたAdam公式とnative W1との差: {max(r['parameter_max_abs_with_saved_gradient'] for r in checks):.6g}。",
        f"- 過去moment＋現在入力の射影和と更新全体の射影の最大残差: {max(r['projected_sum_max_abs'] for r in checks):.6g}。",
        f"- 勾配と履歴を同じ倍率に揃えたε=0対照の最大不変性誤差: {max(r['matched_rescale_epsilon0_max_abs'] for r in checks):.6g}。",
        f"- 勾配の倍率微分の中央差分との最大relative L1: {max(r['gradient_derivative_fd_relative_l1'] for r in checks):.6g}。",
        f"- tail倍率微分の中央差分との最大絶対誤差: {max(r['tail_derivative_fd_max_abs'] for r in checks):.6g}。", '',
        '## 射影と選択的整列の解釈','',
        '- 各入力の現在勾配と過去の一次momentを、実際の全勾配で決まる前処理Pとtail勾配qを用いて射影した。open/band/deepの和は現在勾配全体に一致する。削除介入ではPも変わるため、この足し算だけを削除効果と同一視しない。',
        '- current maximumのqᵀP xと正の誤差減少driveを別々に記録した。両方が正でも、その入力が今回のバッチに入っていなければ、今回の一歩の直接寄与ではない。',
        '- centered tailの変化には入力重みの方向変化が必要だが、それだけで原因は同定されない。populationの楕円対称分布では、各方向の標準化した投影分布自体が同じとなるため、共分散だけから有限入力集合の上端突出を予測することもできない。ここでは実際の有限probe入力を使う。',
        '- この解析は同じ状態からの即時応答を扱う。残差が長く残るという説明には、後続の実際の入力寄与・moment・方向の累積を別途確認する必要がある。',
        f"- 重要な近似限界：alive固定集合で、更新前qによる一次射影と有限ΔTの差はL1で{min(r['finite_projection_error_relative_l1'] for r in summary if r['cohort']=='alive_pre'):.1%}〜{max(r['finite_projection_error_relative_l1'] for r in summary if r['cohort']=='alive_pre'):.1%}。lower-median順位は{min(r['median_rank_changed_fraction'] for r in summary if r['cohort']=='alive_pre'):.1%}〜{max(r['median_rank_changed_fraction'] for r in summary if r['cohort']=='alive_pre'):.1%}のunitで変わる。従って入力別・過去/現在の射影を、有限tail変化の厳密な配分として読んではいけない。最初の表は有限Tを直接計算しており、この近似を使っていない。", '',
        '## ELU実装の区別','',
        '元のsourceはfloat32のexpm1(z)+1を微分に使うため、深い負側で厳密な0を作る。exp(z)へ置換した列は微分だけの反事実であり、元の学習の再生ではない。各gainの0の割合、勾配差、一歩のtail差をreal_prediction_checks.jsonへ保存した。ε=0の不変性検算に限り、勾配と全momentが0の座標はεを正から0へ近づけた極限の更新0として扱った。', '',
        '## 新ラベルの事前期待（同じ最初のバッチ、read-only付録）','',
        '新ラベルを独立uniform πから振る直前の期待では、hidden勾配の平均は a mean[gate*x*vᵀ(p(a)−π)]。単一unitのラベル由来共分散は a²/B² Σ gate² xxᵀ [vᵀ(diagπ−ππᵀ)v] となる。平均勾配のノルム、label-noiseのtrace RMS、tail方向への平均と標準偏差をper-unit列へ保存した。',
        'p0=softmax(c)として、小gain展開は G(a)=aG0+a²G1+O(a³)、G0=mean[gate*x*vᵀ(p0−π)]、G1=mean[gate*x*vᵀH(p0)h]。p0=πならG0=0となり、平均がa²、label-noise RMSがaとなる経路はある。しかしp0≠πでは一般にO(a)が残る。実stateのp0とuniformの最大差、各倍率での展開誤差もchecksへ保存した。',
        'このラベル事前期待は、4000歩にわたってラベルを振り直すモデルではない。実際にはtask51のラベルは固定され、各stepの勾配とmomentはそれに適応する。またAdamの期待更新を、この平均勾配とRMSの比だけで置き換えてはいけない。tail方向の平均の符号もcovarianceの半正定値性からは決まらない。', '',
        '| seed | gain | mean-gradient norm / label-noise RMS 中央値 | noise非zero unit数 |',
        '|---:|---:|---:|---:|']
    for r in summary:
        if r['cohort']=='alive_pre':
            lines.append(f"| {r['seed']} | {r['gain']:g} | {r['uniform_label_mean_noise_ratio_median']:.6g} | {r['uniform_label_noise_nonzero_count']} |")
    lines += ['', 'これは現在の勾配のラベル事前期待と分散だけの比較であり、標本抽出雑音の全量でも、保持したAdamの分母でもない。', '',
        'per-unit結果はreal_prediction_units.csv、各入力の射影はreal_prediction_inputs.csv、集計はreal_prediction_summary.csv。数値の精度限界より小さい符号差を機構の証拠としない。']
    (out/'real_prediction_report.md').write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=Path('results/logistic_v_height_0929'))
    ap.add_argument('--out', type=Path, default=Path('results/logistic_v_height_0929'))
    ap.add_argument('--seeds', type=int, nargs='+', default=[0,1])
    args=ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    all_units, all_inputs, all_checks, files=[],[],[],[]
    for seed in args.seeds:
        statepath=args.source/f'real_s{seed}_t50.npz'
        nativepath=args.source/f'real_s{seed}_t51_step1.npz'
        state,native=load_npz(statepath),load_npz(nativepath)
        units,inputs,checks=analyze_seed(seed,state,native)
        all_units.extend(units);all_inputs.extend(inputs);all_checks.extend(checks)
        files.extend([statepath,nativepath])
    summary=summarize(all_units)
    write_csv(args.out/'real_prediction_units.csv',all_units)
    write_csv(args.out/'real_prediction_inputs.csv',all_inputs)
    write_csv(args.out/'real_prediction_summary.csv',summary)
    (args.out/'real_prediction_checks.json').write_text(json.dumps(all_checks,indent=2)+'\n')
    provenance=dict(source_sha256=digest(__file__),
        git_hash=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        inputs=[dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=digest(p)) for p in files],
        numpy=np.__version__,torch=torch.__version__,
        scope='read-only, no fitted coefficients, first batch of task51 only',
        comparison='original float32 source feature/gate, float64 softmax and Adam formula; native float32 checked separately')
    (args.out/'real_prediction_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    report(args.out,summary,all_checks)
    print(json.dumps(dict(seeds=args.seeds,unit_rows=len(all_units),input_rows=len(all_inputs),
                         max_gradient_error=max(r['gradient_max_abs'] for r in all_checks),
                         max_parameter_error=max(r['parameter_max_abs_native'] for r in all_checks))))


if __name__=='__main__':
    main()

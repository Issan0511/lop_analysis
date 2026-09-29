"""Independent algebra checks and cross-file checks, without additional fits.

The response checks verify mathematical identities on arbitrary states. They
do not identify which mechanism operates in the saved RL-MNIST experiments.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import subprocess
from pathlib import Path

import numpy as np

OUT = Path('results/logistic_v_height_0929')
CODE = Path('analysis/logistic_v_height_0929')


def read(name):
    with (OUT / name).open(newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_csv_line_endings():
    """Record a presentation-only transformation; preserve all CSV fields."""
    manifest_path = OUT/'line_endings.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for path in sorted(OUT.glob('*.csv')):
        before = path.read_bytes()
        after = before.replace(b'\r\n', b'\n')
        if before == after:
            continue
        original = list(csv.reader(io.StringIO(before.decode(), newline='')))
        changed = list(csv.reader(io.StringIO(after.decode(), newline='')))
        assert original == changed, path
        manifest[path.name] = dict(original_sha256=hashlib.sha256(before).hexdigest(),
            normalized_sha256=hashlib.sha256(after).hexdigest(),
            original_bytes=len(before), normalized_bytes=len(after),
            transformation='CRLF to LF only; parsed CSV fields verified identical')
        path.write_bytes(after)
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')


def response_checks():
    import torch
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    gen = torch.Generator().manual_seed(1729)
    x = torch.randn(7, 5, generator=gen)
    w = torch.randn(4, 5, generator=gen, requires_grad=True)
    b = torch.randn(4, generator=gen)
    v = torch.randn(3, 4, generator=gen)
    c = torch.randn(3, generator=gen)
    y = torch.tensor([0, 1, 2, 1, 0, 2, 0])
    a = torch.tensor(1.3, requires_grad=True)
    z = x @ w.T + b
    phi = torch.nn.functional.elu(z)
    prime = torch.where(z > 0, torch.ones_like(z), z.exp())
    h = phi @ v.T
    p = torch.softmax(c + a*h, dim=1)
    e = p - torch.nn.functional.one_hot(y, 3)
    hh = p*(h - (p*h).sum(1, keepdim=True))
    direct = (prime*(e @ v)).T @ x / len(x)
    residual_response = a*(prime*(hh @ v)).T @ x / len(x)
    dg = direct + residual_response

    def gradient(gain):
        loss = torch.nn.functional.cross_entropy(c + gain*phi @ v.T, y)
        return torch.autograd.grad(loss, w, create_graph=True)[0]

    g = gradient(a)
    automatic = torch.autograd.functional.jacobian(gradient, a)
    delta = 1e-5
    finite = (gradient(a+delta)-gradient(a-delta))/(2*delta)
    old_m = torch.randn(4, 5, generator=gen)*.03
    old_s = torch.rand(4, 5, generator=gen)*.05 + .001
    beta1, beta2, eta, eps, t = .9, .999, .001, 1e-8, 37
    correction1, correction2 = 1-beta1**t, 1-beta2**t
    new_m = beta1*old_m + (1-beta1)*g
    new_s = beta2*old_s + (1-beta2)*g*g
    rms = torch.sqrt(new_s/correction2)
    den = rms+eps
    jac_update = -eta*((1-beta1)/(correction1*den)
        - (new_m/correction1)*(1-beta2)*g/(correction2*rms*den**2))

    def update(gain):
        grad = gradient(gain)
        m = beta1*old_m+(1-beta1)*grad
        s = beta2*old_s+(1-beta2)*grad*grad
        return -eta*(m/correction1)/(torch.sqrt(s/correction2)+eps)

    automatic_update = torch.autograd.functional.jacobian(update, a)
    # Infinitesimal violation of the saturated-branch scale symmetry by ELU.
    def affine_compensation_logits(gain):
        zz = (z+1)/gain-1
        cc = c+(gain-1)*v.sum(dim=1)
        return cc+gain*torch.nn.functional.elu(zz) @ v.T
    defect_auto = torch.autograd.functional.jacobian(
        affine_compensation_logits, torch.tensor(1., requires_grad=True))
    defect_formula = torch.where(z < 0, -z*z.exp(), torch.zeros_like(z)) @ v.T
    xcenter = x-x.mean(0)
    covariance = xcenter.T @ xcenter / len(x)
    tail_error = []
    radial_error = []
    for unit in range(len(w)):
        zz = z[:, unit]
        sd = zz.std(correction=0)
        tail = (zz.max()-zz.median())/sd
        xmax = x[zz.argmax()]
        xmed = x[torch.argsort(zz)[len(x)//2]]
        formula = (xmax-xmed)/sd-tail*(covariance @ w[unit])/sd**2
        exact = torch.autograd.grad(tail, w, retain_graph=True)[0][unit]
        tail_error.append(float((formula-exact).abs().max().detach()))
        radial_error.append(abs(float((formula @ w[unit]).detach())))
    return {
        'scope': 'algebra verification at a seeded arbitrary state; no fit or actual-network mechanism identification',
        'same_state_hidden_gradient_max_error': float((g-a*direct).abs().max().detach()),
        'readout_derivative_autograd_max_error': float((dg-automatic).abs().max().detach()),
        'readout_derivative_finite_difference_max_error': float((dg-finite).abs().max().detach()),
        'retained_adam_response_max_error': float((jac_update*dg-automatic_update).abs().max().detach()),
        'elu_scale_defect_autograd_max_error': float((defect_auto-defect_formula).abs().max().detach()),
        'centered_tail_gradient_max_error': max(tail_error),
        'centered_tail_radial_orthogonality_max_error': max(radial_error),
    }


def branch_checks():
    rng = np.random.default_rng(1730)
    # Each positive/negative branch stays fixed throughout these gain changes.
    z = np.array([[12, -15, 20], [-13, 11, -25], [14, -17, 16], [-20, -11, -15.]], float)
    v = rng.normal(size=(3, 4))
    c = rng.normal(size=4)
    ph = lambda zz: np.where(zz > 0, zz, -1.)
    baseline = ph(z) @ v+c
    max_error = 0.
    tail_error = 0.
    leak_error = 0.
    for gain in [.1, .5, 2., 10.]:
        zz = (z+1)/gain-1
        assert np.array_equal(zz > 0, z > 0)
        cc = c+(gain-1)*v.sum(0)
        max_error = max(max_error, float(np.max(np.abs(gain*ph(zz) @ v+cc-baseline))))
        original_tail = (z.max(0)-np.median(z, axis=0))/z.std(0)
        changed_tail = (zz.max(0)-np.median(zz, axis=0))/zz.std(0)
        tail_error = max(tail_error, float(np.max(np.abs(original_tail-changed_tail))))
        elu = lambda aa: np.where(aa > 0, aa, np.expm1(aa))
        expected = np.where(z < 0, gain*np.exp(zz)-np.exp(z), 0)
        actual = gain*elu(zz)-elu(z)-(1-gain)
        leak_error = max(leak_error, float(np.max(np.abs(actual-expected))))
    return dict(frozen_branch_logit_identity_max_error=max_error,
                frozen_branch_tail_invariance_max_error=tail_error,
                exact_elu_correction_identity_max_error=leak_error)


def pointwise_compensation_check():
    """A conditional hypothesis, not a network optimum or mechanism claim."""
    cases = []
    for values in [[-3., -2., -1., 1.], [-1., -1., -1., -1., -.01, -.01, 10.]]:
        z = np.array(values)
        zp = np.maximum(z, 0)
        var = z.var()
        maximum, median = z.max(), np.median(z)
        tail = (maximum-median)/np.sqrt(var)
        ratio = np.mean((z-z.mean())*(zp-zp.mean()))/var
        prediction = tail*(ratio-maximum/(maximum-median))
        def transformed_tail(log_gain):
            gain = np.exp(log_gain)
            psi = np.where(z < 0, np.exp(z), z+1)/gain
            zz = np.where(psi < 1, np.log(psi), psi-1)
            return (zz.max()-np.median(zz))/zz.std()
        step = 1e-5
        finite = (transformed_tail(step)-transformed_tail(-step))/(2*step)
        cases.append(dict(z=values, covariance_ratio=float(ratio),
                          geometric_ratio=float(maximum/(maximum-median)),
                          theoretical_derivative=float(prediction),
                          finite_difference_derivative=float(finite),
                          error=float(abs(prediction-finite))))
    assert max(r['error'] for r in cases) < 1e-8
    assert cases[0]['theoretical_derivative'] < 0 < cases[1]['theoretical_derivative']
    return cases


def actual_artifact_checks():
    """Check preserved execution source and checkpoint identities, without fits."""
    result = {}
    def historical_sha(commit, name):
        raw = subprocess.check_output(['git', 'show', f'{commit}:{CODE/name}'])
        return hashlib.sha256(raw).hexdigest()
    execution = historical_sha('008302c', 'real_interventions.py')
    direction = historical_sha('f11cf4d', 'real_direction_control.py')
    # Reporting-only corrections were in the working copy during F and the
    # precision shadow; their exact dependency hash is preserved in this bundle.
    runner = sha(CODE/'real_interventions.py')
    for seed in (0, 1):
        original = json.loads((OUT/f'real_s{seed}_provenance.json').read_text())
        final = json.loads((OUT/f'real_direction_s{seed}_provenance.json').read_text())
        precision = json.loads((OUT/f'real_null_precision_s{seed}_provenance.json').read_text())
        assert original['source_sha256'] == execution
        assert final['source_sha256'] == direction
        assert final['native_runner_sha256'] == runner
        assert final['task50_sha256'] == sha(OUT/f'real_s{seed}_t50.npz')
        assert final['task_plan_sha256'] == sha(OUT/f'real_s{seed}_task_plan.npz')
        assert final['all_length_rounding_bounds_verified']
        assert final['frozen_readouts_unchanged']
        assert all(all(d.values()) for d in final['reference_and_C_bit_equal_to_original'].values())
        assert precision['code_sha256'] == sha(CODE/'real_null_precision.py')
        assert precision['implementation_dependency_sha256'] == runner
        assert precision['source_checkpoint_sha256'] == final['task50_sha256']
        assert precision['source_task_plan_sha256'] == final['task_plan_sha256']
        result[str(seed)] = dict(execution_source_preserved=True, source_checkpoints_match=True,
            direction_reference_and_natural_states_bit_equal=True,
            actual_step_length_rounding_bounds_pass=True)
    prediction = json.loads((OUT/'real_prediction_provenance.json').read_text())
    assert prediction['source_sha256'] == sha(CODE/'real_state_predictions.py')
    for entry in prediction['inputs']:
        assert entry['sha256'] == sha(OUT/Path(entry['path']).name)
    result['first_state_prediction_sources_match'] = True
    result['execution_commits'] = dict(primary='008302c', direction_driver='f11cf4d',
        direction_and_precision_dependency_sha256=runner)
    return result


def main():
    normalize_csv_line_endings()
    sources = {}
    for name, file, key in [
        ('scalar', 'scalar.py', 'source_sha256'),
        ('shared', 'shared_models.py', 'code_sha256'),
        ('saved', 'saved_audit.py', 'analysis_sha256'),
        ('anisotropic', 'anisotropic_toy.py', 'source_sha256')]:
        prov = json.loads((OUT / f'{name}_provenance.json').read_text())
        sources[name] = sha(CODE/file) == prov[key]
    assert all(sources.values()), sources
    pairs = read('shared_paired.csv')
    raw = sum(float(r['unit_mean_raw_max_small_minus_large']) > 0 for r in pairs)
    tail = sum(float(r['unit_mean_centered_max_over_s_small_minus_large']) > 0 for r in pairs)
    assert (len(pairs), raw, tail) == (45, 45, 18)
    rows = read('scalar_sweep.csv')
    assert len({r['config_id'] for r in rows}) == 739
    flow = read('scalar_flow_check.csv')
    refinement = {}
    for v in [.1, 1., 10.]:
        selected = sorted([r for r in flow if float(r['v']) == v], key=lambda r: -float(r['dt']))
        errors = [float(r['abs_error']) for r in selected]
        refinement[str(v)] = [errors[0]/errors[1], errors[1]/errors[2]]
        assert all(1.99 < q < 2.01 for q in refinement[str(v)])
    # Same scalar loss fixes margin exactly; it is a conditional identity.
    loss_rows = []
    for loss in [.01, .1, .5]:
        margin = -math.log(math.expm1(loss))
        for v in [.01, .1, 1., 10., 100.]:
            z = margin/v
            loss_rows.append(dict(loss=loss, v=v, z=z,
                recomputed_loss=float(np.logaddexp(0., -v*z))))
    loss_error = max(abs(r['loss']-r['recomputed_loss']) for r in loss_rows)
    response = response_checks()
    branch = branch_checks()
    for k, val in response.items():
        if k.endswith('max_error'):
            assert val < (1e-7 if 'finite_difference' in k else 1e-11), (k, val)
    assert max(branch.values()) < 1e-11, branch
    checks = dict(source_hash_matches=sources, scalar_configurations=739,
                  shared_pair_count=len(pairs), raw_small_v_higher=raw,
                  centered_tail_small_v_higher=tail,
                  flow_error_refinement_ratios=refinement,
                  matched_loss_identity_max_error=loss_error,
                  matched_loss_cases=loss_rows, exact_response_checks=response,
                  frozen_branch_checks=branch,
                  actual_artifact_checks=actual_artifact_checks(),
                  conditional_pointwise_psi_compensation=pointwise_compensation_check())
    (OUT/'validation.json').write_text(json.dumps(checks, indent=2)+'\n')
    print(json.dumps({k: v for k, v in checks.items() if k != 'matched_loss_cases'}))


if __name__ == '__main__':
    main()

"""Frozen crossed-residual/readout/Jacobian assay; never updates a source state."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / 'specs/spec_readout_separation_1111.md'
BANK_KEYS = ('S1', 'S01', 'P1', 'P01', 'O1', 'O01')
LABELS = ('actual', 'uniform')
GROUPS = ('body', 'trunk', 'gamma', 'beta')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for x in iter(lambda: f.read(1 << 20), b''):
            h.update(x)
    return h.hexdigest()


def tensor_sha(x):
    a = x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str((a.dtype, a.shape)).encode() + a.tobytes()).hexdigest()


def subset_indices(n, model, seed, count=128):
    key = f'readout_separation_1111|probe_subset|{model}|{seed}'
    v = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little') & ((1 << 63) - 1)
    return torch.randperm(n, generator=torch.Generator().manual_seed(v))[:count], key


def norm(xs):
    valid = [x.detach().double().square().sum() for x in xs if x is not None]
    return float(torch.stack(valid).sum().sqrt()) if valid else 0.


def cosine(xs, ys):
    a, b = norm(xs), norm(ys)
    if a == 0. or b == 0.:
        return None
    dots = [torch.sum(x.detach().double() * y.detach().double())
            for x, y in zip(xs, ys) if x is not None and y is not None]
    d = float(torch.stack(dots).sum()) if dots else 0.
    return max(-1., min(1., d / (a * b)))


def assert_close(name, a, b, checks, rtol=2e-5, atol=2e-7):
    a, b = a.detach().double(), b.detach().double()
    ok = bool(torch.allclose(a, b, rtol=rtol, atol=atol))
    checks[name] = dict(passed=ok, max_abs=float((a-b).abs().max()) if a.numel() else 0.,
                        reference_max_abs=float(b.abs().max()) if b.numel() else 0., rtol=rtol, atol=atol)
    if not ok:
        raise AssertionError((name, checks[name]))



def assert_norm_close(name, actual, reference, checks, context):
    """Numerical identity for a vector, robust to near-zero coordinates.

    Different GEMM batch shapes and gauge-rescaled float32 layer norms can
    accumulate rounding/cancellation. Retain the original 2e-5 tolerance on
    relative L2, with no absolute floor. Callers check each relevant sample or
    parameter separately as well as the global vector. Float64 synthetic
    identities use 1e-11 relative and max-absolute bounds.
    """
    xs = [actual] if torch.is_tensor(actual) else list(actual)
    ys = [reference] if torch.is_tensor(reference) else list(reference)
    if len(xs) != len(ys) or any((a is None) != (b is None) for a,b in zip(xs,ys)):
        raise AssertionError((name, 'different unused-parameter structure'))
    pairs = [(a.detach().double(), b.detach().double()) for a,b in zip(xs,ys) if a is not None]
    if any(a.shape != b.shape for a,b in pairs):
        raise AssertionError((name, 'different shapes'))
    is_double = all(a is None or a.dtype == torch.float64 for a in xs+ys)
    rtol = 1e-11 if is_double else 2e-5
    error = norm([a-b for a,b in pairs]); ref = norm([b for _,b in pairs])
    max_abs = max((float((a-b).abs().max()) for a,b in pairs if a.numel()), default=0.)
    rel = error/ref if ref else (0. if error == 0. else None)
    cos = cosine([a for a,_ in pairs], [b for _,b in pairs])
    # If ||a-b||/||b|| <= r, normalized vectors differ by at most 2r;
    # hence 1-cos <= 2r^2. Float64 reduction roundoff needs its own epsilon.
    cos_tol = max(2*rtol**2, 16*torch.finfo(torch.float64).eps)
    finite = all(bool(torch.isfinite(a).all() and torch.isfinite(b).all()) for a,b in pairs)
    ok = finite and ((error == 0.) if ref == 0. else
                     (rel <= rtol and cos is not None and 1-cos <= cos_tol))
    if is_double: ok = ok and max_abs <= 1e-11
    checks[name] = dict(passed=bool(ok), max_abs=max_abs, error_l2=error, reference_l2=ref,
                        relative_l2=rel, cosine=cos, relative_l2_tolerance=rtol,
                        one_minus_cosine_tolerance=cos_tol,
                        absolute_tolerance=1e-11 if is_double else 0.,
                        criterion='relative L2 and cosine per vector; exactly zero reference requires exactly zero alternative',
                        precision='float64_identity' if is_double else 'float32_numerical_'+context)
    if not ok: raise AssertionError((name, checks[name]))



def assert_gauge_close(name, actual, reference, checks):
    assert_norm_close(name, actual, reference, checks, 'gauge')


def assert_feature_consistency(actual, reference, checks):
    """Compare selected features computed in different batch shapes.

    Full-set chunks end with 48 MLP examples (68 ViT); the fixed probe uses
    128. Floating-point GEMM reductions can differ across those shapes. This
    numerical check does not alter either feature tensor or probability bank.
    Every local Jacobian assay still requires exactly the same selected-128
    forward output, checked separately with zero relative/absolute tolerance.
    """
    if actual.ndim != 2 or actual.shape != reference.shape:
        raise AssertionError('Selected/full-set feature shape mismatch')
    name = 'selected_features_fullset_consistency'
    assert_norm_close(name, actual, reference, checks, 'batch_features')
    for i, (a,b) in enumerate(zip(actual,reference)):
        assert_norm_close(name+f'__sample{i}', a, b, checks, 'batch_features')


def body_layout(box):
    from analysis.loading_bridge_1111 import dynamic as D
    ps = list(box.body())
    if isinstance(box, D.MLP):
        names = ['body_weight', 'body_bias']
        groups = dict(body=list(range(len(ps))), trunk=list(range(len(ps))), gamma=[], beta=[])
    else:
        names_by_id = {id(p): name for name, p in box.model.named_parameters()}
        names = [names_by_id[id(p)] for p in ps]
        ln = box.model.layers['norm']
        gamma = [i for i, p in enumerate(ps) if p is ln.weight]
        beta = [i for i, p in enumerate(ps) if p is ln.bias]
        assert len(gamma) == len(beta) == 1
        groups = dict(body=list(range(len(ps))), gamma=gamma, beta=beta,
                      trunk=[i for i in range(len(ps)) if i not in gamma+beta])
    return ps, names, groups


def selected(xs, groups, group):
    return [xs[i] for i in groups[group]]


def optimizer_groups(opt):
    return {id(p): group for group in opt.param_groups for p in group['params']}


@torch.no_grad()
def virtual_adam(ps, grads, opt, clear=False):
    """Float-dtype Adam next parameter displacement, including rounding at p.

    Uses copies of m/v and parameters, never writes source optimizer state. An
    unused parameter (None gradient) is skipped, as in torch.optim.Adam.
    """
    by_id = optimizer_groups(opt)
    result = []
    for p, grad in zip(ps, grads):
        if grad is None:
            result.append(torch.zeros_like(p))
            continue
        group = by_id[id(p)]
        if group.get('differentiable') or group.get('decoupled_weight_decay'):
            raise ValueError('Unsupported Adam mode')
        state = {} if clear else opt.state.get(p, {})
        g = grad.detach().clone()
        if group.get('maximize', False):
            g.neg_()
        wd = float(group.get('weight_decay', 0.))
        if wd:
            g.add_(p, alpha=wd)
        m = state.get('exp_avg', torch.zeros_like(p)).detach().clone()
        v = state.get('exp_avg_sq', torch.zeros_like(p)).detach().clone()
        b1, b2 = group['betas']
        m.lerp_(g, 1-b1)
        v.mul_(b2).addcmul_(g, g, value=1-b2)
        step = float(state.get('step', 0.)) + 1.
        if group.get('amsgrad', False):
            previous = state.get('max_exp_avg_sq', torch.zeros_like(p))
            v = torch.maximum(v, previous)
        den = v.sqrt().div_(math.sqrt(1-b2**step)).add_(float(group['eps']))
        after = p.detach().clone().addcdiv_(m, den, value=-float(group['lr'])/(1-b1**step))
        result.append(after-p.detach())
    return result


def validate_virtual_adam(ps, grads, opt, clear, checks, name):
    """A real torch step on cloned parameters and cloned memories only."""
    by_id = optimizer_groups(opt)
    clones = [torch.nn.Parameter(p.detach().clone()) for p in ps]
    # Preserve original parameter-group hyperparameters, including unused params.
    new_groups = []
    for old_group in opt.param_groups:
        members = [i for i, p in enumerate(ps) if by_id[id(p)] is old_group]
        if not members:
            continue
        kwargs = {k: old_group[k] for k in ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize',
                                                   'foreach', 'fused', 'capturable', 'differentiable') if k in old_group}
        new_groups.append(dict(params=[clones[i] for i in members], **kwargs))
    clone_opt = torch.optim.Adam(new_groups)
    for p, q, g in zip(ps, clones, grads):
        if not clear:
            clone_opt.state[q] = copy.deepcopy(opt.state.get(p, {}))
        q.grad = None if g is None else g.detach().clone()
    predicted = virtual_adam(ps, grads, opt, clear)
    clone_opt.step()
    errs = []
    for i, (p, q, d) in enumerate(zip(ps, clones, predicted)):
        actual = q.detach()-p.detach()
        group = by_id[id(p)]
        fused = bool(group.get('fused', False))
        if fused and p.dtype != torch.float64:
            # Native fused Adam can differ from the analytical unfused formula
            # by parameter-store ulps. Use an elementwise bound, not a tolerance
            # based on the largest (possibly unrelated) entry in this tensor.
            def ulp(t):
                up = torch.nextafter(t, torch.full_like(t, float('inf')))
                down = torch.nextafter(t, torch.full_like(t, -float('inf')))
                return torch.maximum((up-t).abs(), (t-down).abs()).double()
            bound = 5e-10 + 2e-5*actual.double().abs() + 2*torch.maximum(ulp(p.detach()), ulp(q.detach()))
            err = (d.double()-actual.double()).abs()
            ok = bool((err <= bound).all())
            checks[f'{name}_param{i}'] = dict(passed=ok, max_abs=float(err.max()), backend='native_fused',
                                             tolerance='5e-10 + 2e-5*abs(delta) + 2*max(ulp(p),ulp(p_after))',
                                             max_allowed_abs=float(bound.max()))
            if not ok: raise AssertionError((name, i, float((err-bound).max())))
        else:
            assert_close(f'{name}_param{i}', d, actual, checks, rtol=2e-5,
                         atol=2e-14 if p.dtype == torch.float64 else 6e-8)
        errs.append(float((d-actual).abs().max()))
    return dict(passed=True, n_parameters=len(ps), max_abs=max(errs, default=0.),
                backend_groups=[{k:g.get(k) for k in ('foreach','fused','capturable','differentiable')} for g in clone_opt.param_groups])


def match_trunk(grads, native, groups):
    a, b = norm(selected(grads, groups, 'trunk')), norm(selected(native, groups, 'trunk'))
    if a == 0. or b == 0.:
        return None, None, 'zero_alternative' if a == 0. else 'zero_native'
    scale = b/a
    idx = set(groups['trunk'])
    return [None if g is None else g*scale if i in idx else g for i, g in enumerate(grads)], scale, 'matched'


def grouped_stats(row, prefix, values, groups, native=None, baseline=None):
    for group in GROUPS:
        a = selected(values, groups, group)
        row[f'{prefix}_{group}_norm'] = norm(a)
        row[f'{prefix}_{group}_cos_native'] = None if native is None else cosine(a, selected(native, groups, group))
        row[f'{prefix}_{group}_cos_baseline'] = None if baseline is None else cosine(a, selected(baseline, groups, group))


def preact_shift(values, mean_x):
    # For the MLP's affine preactivation this is the exact mean change induced
    # by parameter displacement values, or the signed GD push for gradients.
    w, b = values
    return mean_x.double() @ w.double().T + b.double()


def array_stats(row, prefix, x, arrays, key, baseline=None, native=None):
    x = x.detach().double()
    row[prefix+'_norm'] = norm([x])
    row[prefix+'_mean_vector_norm'] = float(x.mean(0).norm())
    row[prefix+'_mean_example_norm'] = float(x.flatten(1).norm(dim=1).mean())
    row[prefix+'_cos_native'] = None if native is None else cosine([x], [native])
    row[prefix+'_cos_baseline'] = None if baseline is None else cosine([x], [baseline])
    arrays[key+'__'+prefix+'_example_norm'] = x.flatten(1).norm(dim=1).cpu().numpy()
    arrays[key+'__'+prefix+'_mean_vector'] = x.mean(0).cpu().numpy()


def fullset_stats(h, entries, old, new, gamma=None, beta=None):
    h, old, new = h.double().cpu(), old.cpu(), new.cpu()
    mu, hc = h.mean(0), h-h.mean(0)
    result, arrays = [], {}
    for key in BANK_KEYS:
        e = entries[key]; w, b = e['W'].double().cpu(), e['b'].double().cpu()
        z = h@w.T+b; lp = z.log_softmax(1); p = lp.exp(); zc = z-z.mean(1, keepdim=True)
        wc = w-w.mean(0)
        row = dict(bank=key, direction=e['direction'], scale=e['scale'], n=len(h),
                   feature_mean_norm=float(mu.norm()), feature_centered_rms=float(hc.square().mean().sqrt()),
                   head_norm=float(w.norm()), head_centered_norm=float(wc.norm()), bias_norm=float(b.norm()),
                   old_accuracy=float((z.argmax(1)==old).double().mean()),
                   new_accuracy=float((z.argmax(1)==new).double().mean()),
                   old_margin=float(zc.gather(1, old[:, None]).mean()),
                   new_margin=float(zc.gather(1, new[:, None]).mean()),
                   actual_ce=float(F.nll_loss(lp, new)), uniform_ce=float(-lp.mean()),
                   entropy=float(-(p*lp).sum(1).mean()), confidence=float(p.max(1).values.mean()),
                   logit_centered_rms=float(zc.square().mean().sqrt()), mean_logit_rms=float(z.mean(0).square().mean().sqrt()))
        if gamma is not None:
            ga, be = gamma.detach().double().cpu(), beta.detach().double().cpu()
            eff = w*ga[None, :]; eb = w@be+b
            row.update(effective_Wgamma_norm=float(eff.norm()), effective_Wgamma_centered_norm=float((eff-eff.mean(0)).norm()),
                       effective_Wbeta_b_norm=float(eb.norm()), effective_Wbeta_b_centered_norm=float((eb-eb.mean()).norm()))
        arrays[key+'__W'] = w.numpy(); arrays[key+'__b'] = b.numpy()
        arrays[key+'__logits'] = z.numpy(); arrays[key+'__probability'] = p.numpy()
        arrays[key+'__row_norm'] = w.norm(dim=1).numpy(); arrays[key+'__centered_column_norm'] = wc.norm(dim=0).numpy()
        result.append(row)
    return result, arrays


def transmission_vjp(box, x, gh, jacobian, ps):
    from analysis.readout_separation_1111 import ops
    h, aux = ops.features(box, x, jacobian=jacobian, return_aux=True)
    pre = aux.get('preact', aux.get('preln'))
    grads = torch.autograd.grad(h, [pre]+ps, grad_outputs=gh.to(h.dtype), allow_unused=True)
    pregrad = grads[0]
    if 'preln' in aux:
        # The final head consumes only CLS; every other direct pre-LN cotangent
        # is zero. The trunk parameter VJP still includes all token pathways.
        if bool(torch.count_nonzero(pregrad[:, 1:])):
            raise AssertionError('Unexpected final-LN non-CLS cotangent')
        pregrad = pregrad[:, aux['cls_index']]
        prevalue = pre[:, aux['cls_index']].detach()
    else:
        prevalue = pre.detach()
    return h.detach(), prevalue, pregrad.detach(), [None if g is None else g.detach() for g in grads[1:]]


def gauge_check(box, x, y, checks):
    from analysis.readout_separation_1111 import ops
    h, aux = ops.features(box, x, return_aux=True)
    pre, gamma, beta, eps = aux['preln'], aux['gamma'], aux['beta'], aux['eps']
    W, b, rows = box.head(); w, bb = W[rows], b[rows]
    wp = (w.detach()*.1).requires_grad_()
    gp = (gamma.detach()*10).requires_grad_(); bp = (beta.detach()*10).requires_grad_()
    hp = F.layer_norm(pre, (pre.shape[-1],), gp, bp, eps)[:, 0]
    z, zp = F.linear(h, w, bb), F.linear(hp, wp, bb)
    assert_gauge_close('gauge_logits', zp, z, checks)
    residual = (z.detach().softmax(1)-F.one_hot(y, z.shape[1]))/len(y)
    ps, _, groups = body_layout(box); trunk = selected(ps, groups, 'trunk')
    original = torch.autograd.grad(z, [pre, gamma, beta, w]+trunk, residual, retain_graph=True, allow_unused=True)
    changed = torch.autograd.grad(zp, [pre, gp, bp, wp]+trunk, residual, allow_unused=True)
    assert_gauge_close('gauge_preLN_vjp', changed[0], original[0], checks)
    assert_gauge_close('gauge_gamma_coordinate_vjp', changed[1], original[1]*.1, checks)
    assert_gauge_close('gauge_beta_coordinate_vjp', changed[2], original[2]*.1, checks)
    assert_gauge_close('gauge_W_coordinate_vjp', changed[3], original[3]*10, checks)
    for i, (a, b0) in enumerate(zip(changed[4:], original[4:])):
        if a is None or b0 is None:
            assert a is b0
        else:
            assert_gauge_close(f'gauge_trunk_vjp_{i}', a, b0, checks)
    assert_gauge_close('gauge_trunk_vjp_global', changed[4:], original[4:], checks)
    assert_gauge_close('gauge_effective_Wgamma', wp*gp[None], w*gamma[None], checks)
    assert_gauge_close('gauge_effective_Wbeta_b', wp@bp+bb, w@beta+bb, checks)
    return dict(passed=True, same_fixed_residual=True, source_mutated=False,
                numerical_criterion='Per-parameter and global relative L2/cosine; float32 rtol 2e-5, zero absolute floor. Float64 synthetic identity uses rtol/maxabs 1e-11.',
                interpretation='Numerical gauge identity; no raw readout-norm causal claim.')


def source_provenance(box):
    from analysis.loading_bridge_1111 import dynamic as D
    from analysis.readout_separation_1111 import ops
    from src import stage1_ctrl_1111 as M
    files = list(getattr(box, 'sources', [getattr(box, 'source', None)]))
    files = [Path(p) for p in files if p is not None]
    helpers = [ROOT/'analysis/readout_separation_1111'/name for name in ('ops.py', 'run.py', 'probe.py', 'launch.py')]
    helpers += [Path(D.__file__), Path(M.__file__), ROOT/'analysis/loading_bridge_1111/static_probe.py',
                ROOT/'analysis/p_residual_loading_1111/run.py',
                ROOT/'analysis/vit_long_1024/posthoc_2blk/scale_counterfactual.py']
    helpers = [p for p in helpers if p.exists()]
    helper_hashes = {str(p.relative_to(ROOT)):sha(p) for p in helpers}
    if hasattr(box, 'C'): helper_hashes.update(box.C.source_hashes())
    return dict(git_hash=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                script_sha256=sha(__file__), spec_sha256=sha(SPEC), helpers=helper_hashes,
                sources={str(p):sha(p) for p in files}, inputs_sha256=tensor_sha(box.X),
                new_labels_sha256=tensor_sha(box.new), old_labels_sha256=tensor_sha(box.old), torch=torch.__version__, numpy=np.__version__,
                replay_checks=getattr(box, 'replay_checks', []), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))


def assay(box, t_end, output_dir, bank=None):
    """Run a complete registered frozen assay on a restored source box."""
    from analysis.loading_bridge_1111 import dynamic as D
    from analysis.p_residual_loading_1111 import run as R
    from analysis.readout_separation_1111 import ops
    started = time.monotonic(); model = 'mlp' if isinstance(box, D.MLP) else 'vit'
    act = getattr(box, 'act', 'SL'); out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    stem = f'probe_{model}_{act}_s{box.seed}_t{t_end}'
    prov = source_provenance(box); before = R.fingerprints(box)
    if (out/(stem+'.json')).exists():
        previous = json.loads((out/(stem+'.json')).read_text())
        old = previous['provenance']
        assert previous['complete'] and previous['source_fingerprints']==before
        for key in ('git_hash', 'script_sha256', 'spec_sha256', 'helpers', 'sources'):
            assert old[key]==prov[key], f'Incompatible resume: {key}'
        assert sha(out/previous['arrays_file'])==previous['arrays_sha256']
        return previous
    mode = box.model.training if model=='vit' else None
    if model=='vit': box.model.eval()
    bank = ops.make_head_bank(box, t_end) if bank is None else bank
    if model=='vit': box.model.eval()
    ps, names, groups = body_layout(box)
    with torch.no_grad():
        hfull = torch.cat([ops.native_features(box, x).cpu() for x in box.X.split(128)])
    idx_cpu, subset_key = subset_indices(len(box.X), model, box.seed)
    idx = idx_cpu.to(box.X.device); x = box.X[idx]; y = box.new[idx]
    with torch.no_grad(): h0 = ops.native_features(box, x).detach()
    checks = {}; assert_feature_consistency(h0.cpu(), hfull[idx_cpu], checks)
    fullset, arrays = fullset_stats(hfull, bank['entries'], box.old, box.new,
                                  box.model.layers['norm'].weight if model=='vit' else None,
                                  box.model.layers['norm'].bias if model=='vit' else None)
    for key in BANK_KEYS:
        assert_close('fullset_mean_logits_'+key, torch.from_numpy(arrays[key+'__logits']).mean(0),
                     torch.from_numpy(arrays['S1__logits']).mean(0), checks, rtol=2e-5, atol=2e-5)
    arrays.update(indices=idx_cpu.numpy(), selected_features=h0.cpu().numpy(), old_labels=box.old.cpu().numpy(), new_labels=box.new.cpu().numpy())
    local_p = {key:(h0.double()@bank['entries'][key]['W'].double().T+bank['entries'][key]['b'].double()).softmax(1)
               for key in BANK_KEYS}
    probability_hashes = {key:tensor_sha(p) for key,p in local_p.items()}
    mean_x = box.X.mean(0) if model=='mlp' else None
    modes = ('native', 'elu_unity') if model=='mlp' else ('native', 'ln_remove_radial', 'ln_remove_radial_mean')
    records = []; baselines = {}; real_adam_checks = {}
    for label in LABELS:
        for fkey in BANK_KEYS:
            prob = local_p[fkey]
            target = F.one_hot(y, prob.shape[1]).double() if label=='actual' else torch.full_like(prob, 1/prob.shape[1])
            residual = prob-target
            for bkey in BANK_KEYS:
                gh = (residual@bank['entries'][bkey]['W'].double())/len(x)
                native = None; native_pre = None; native_updates = {}
                for jac in modes:
                    key = f'{label}__{fkey}__{bkey}__{jac}'
                    h, preval, gpre, grads = transmission_vjp(box, x, gh, jac, ps)
                    assert_close(key+'__same_forward', h, h0, checks, rtol=0., atol=0.)
                    if jac=='native':
                        native, native_pre = grads, gpre
                        if fkey==bkey=='S1':
                            baselines[label] = dict(grads=grads, pre=gpre, gh=gh, updates={})
                    baseline = baselines[label]
                    row = dict(model=model, activation=act, seed=box.seed, t_end=t_end, forward_bank=fkey,
                               backward_bank=bkey, label_channel=label, jacobian=jac, n_subset=len(x),
                               probability_sha256=probability_hashes[fkey], residual_norm=norm([residual]),
                               g_feature_unreduced_mean_example_norm=float((gh*len(x)).norm(dim=1).mean()))
                    array_stats(row, 'g_feature', gh, arrays, key, baseline=baseline['gh'], native=gh)
                    array_stats(row, 'g_pre', gpre, arrays, key, baseline=baseline['pre'], native=native_pre)
                    row['g_feature_radial_sum'] = float((gh*h0.double()).sum())
                    row['g_pre_radial_sum'] = float((gpre.double()*preval.double()).sum())
                    grouped_stats(row, 'g', grads, groups, native, baseline['grads'])
                    if model=='mlp':
                        push = preact_shift(grads, mean_x)
                        row.update(gd_push_mean=float(push.mean()), gd_push_abs_mean=float(push.abs().mean()))
                        arrays[key+'__gd_mean_preact_push'] = push.cpu().numpy()
                    for memory in ('keep', 'clear'):
                        updates = virtual_adam(ps, grads, box.opt, clear=memory=='clear')
                        if jac=='native':
                            native_updates[memory] = updates
                            if fkey==bkey=='S1': baseline['updates'][memory] = updates
                        grouped_stats(row, 'adam_'+memory, updates, groups, native_updates[memory], baseline['updates'][memory])
                        if memory=='clear':
                            row['adam_keep_clear_body_cosine'] = cosine(kept_updates, updates)
                        else: kept_updates = updates
                        if model=='mlp':
                            shift = preact_shift(updates, mean_x)
                            row['adam_'+memory+'_mean_preact_shift'] = float(shift.mean())
                            row['adam_'+memory+'_abs_mean_preact_shift'] = float(shift.abs().mean())
                            arrays[key+'__adam_'+memory+'_mean_preact_shift'] = shift.cpu().numpy()
                        if label=='actual' and fkey==bkey=='S1' and jac=='native':
                            real_adam_checks[memory] = validate_virtual_adam(ps, grads, box.opt, memory=='clear', checks, 'real_adam_'+memory)
                    row['normmatch_status'] = 'native_reference'; row['normmatch_scale'] = 1.
                    if jac!='native':
                        matched, scale, status = match_trunk(grads, native, groups)
                        row.update(normmatch_status=status, normmatch_scale=scale)
                        if matched is not None:
                            grouped_stats(row, 'nm_g', matched, groups, native, baseline['grads'])
                            for memory in ('keep', 'clear'):
                                d = virtual_adam(ps, matched, box.opt, clear=memory=='clear')
                                grouped_stats(row, 'nm_adam_'+memory, d, groups, native_updates[memory], baseline['updates'][memory])
                                if model=='mlp': row['nm_adam_'+memory+'_mean_preact_shift'] = float(preact_shift(d, mean_x).mean())
                        for group in ('gamma', 'beta'):
                            for i in groups[group]:
                                assert_close(key+'__'+group+'_same_vjp', grads[i], native[i], checks, rtol=0., atol=0.)
                    if any(not math.isfinite(v) for v in row.values() if isinstance(v, float)):
                        raise FloatingPointError(key)
                    records.append(row)
    gauge = gauge_check(box, x, y, checks) if model=='vit' else None
    assert len(records)==(144 if model=='mlp' else 216)
    assert R.fingerprints(box)==before, 'Source parameters or optimizer changed'
    if model=='vit': box.model.train(mode)
    arrays_path = out/(stem+'.npz'); np.savez_compressed(arrays_path, **arrays)
    result = dict(complete=True, model=model, activation=act, seed=box.seed, t_end=t_end, n_subset=len(x),
                  subset_key=subset_key, subset_sha256=tensor_sha(idx_cpu), source_fingerprints=before,
                  provenance=prov, bank_metadata=bank['metadata'], body_parameter_names=names, parameter_groups=groups,
                  fullset=fullset, records=records, checks=checks, real_adam_checks=real_adam_checks, gauge_check=gauge,
                  arrays_file=arrays_path.name, arrays_sha256=sha(arrays_path), elapsed_s=time.monotonic()-started,
                  definitions=dict(gradient='mean loss over the fixed128 examples; g_feature before dtype cast; parameter VJP in native dtype',
                                   local_primary_label='actual', uniform='expectation under independent uniform new labels',
                                   gd_push='positive mean gradient push lowers mean preactivation under gradient descent; full-set input mean and128-example gradient',
                                   adam_shift='signed actual mean preactivation shift from virtual Adam parameter displacement',
                                   cosine_native='same forward bank, backward map, label channel; native transmission',
                                   cosine_baseline='S1 probability and S1 backward map, same label channel, native transmission',
                                   normmatch='scale only upstream trunk parameter gradient to native trunk norm; final affine gradients unchanged'))
    path = out/(stem+'.json'); tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(result, indent=1, allow_nan=False)); tmp.replace(path)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('mlp', 'vit'), required=True)
    parser.add_argument('--seeds', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args()
    from analysis.loading_bridge_1111 import dynamic as D
    torch.set_num_threads(1)
    for seed in map(int, args.seeds.split(',')):
        box = D.MLP(seed, 'ELU', [10]) if args.model=='mlp' else D.ViT(seed, argparse.Namespace(engine='eager'))
        t = 10 if args.model=='mlp' else 100; box.reset(t)
        result = assay(box, t, args.out)
        print(json.dumps(dict(seed=seed, complete=result['complete'], records=len(result['records']), elapsed_s=result['elapsed_s'])), flush=True)


if __name__=='__main__':
    main()

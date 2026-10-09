"""Finite identities for a sparse-support native ten-class CNN.

No training, stationary Monte Carlo, or practical learning-rate certificate.
Exact arithmetic checks a conservative stationary phase margin. Float64
checks native gates, full raw derivatives, and all-data capacity envelopes;
analytic nonemptiness and the finite stochastic theorem are separate proofs.
"""
from argparse import ArgumentParser
from fractions import Fraction as Q
from hashlib import sha256
from pathlib import Path
import json
import math

import torch
from torch.nn import functional as F

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
N, BATCH, EPOCHS, PHASES, CLASSES = 1200, 16, 400, 30000, 10
SHAPES = [(16, 3, 5, 5), (16,), (16, 16, 5, 5), (16,),
          (100, 1024), (100,), (100, 100), (100,), (10, 100), (10,)]
RAW_COUNT = sum(math.prod(s) for s in SHAPES)
assert RAW_COUNT == 121242


def pack(parts):
    return torch.cat([p.reshape(-1) for p in parts])


def unpack(theta, shapes):
    parts, offset = [], 0
    for shape in shapes:
        size = math.prod(shape)
        parts.append(theta[offset:offset+size].reshape(shape))
        offset += size
    assert offset == theta.numel()
    return parts


def hidden(hidden_theta, images, shapes, envelope=False, extra=False):
    w1, b1, w2, b2, w3, b3, w4, b4 = unpack(hidden_theta, shapes[:8])
    z1 = F.conv2d(images, w1, b1, padding=2)
    pool = (lambda z: 4*F.avg_pool2d(z, 2)) if envelope else (
        lambda z: F.max_pool2d(z, 2))
    h1 = pool(F.relu(z1))
    z2 = F.conv2d(h1, w2, b2, padding=2)
    h2 = pool(F.relu(z2))
    z3 = F.linear(h2.flatten(1), w3, b3)
    h3 = F.relu(z3)
    z4 = F.linear(h3, w4, b4)
    h4 = F.relu(z4)
    return (h4, (z1, z2, z3, z4)) if extra else h4


def forward(theta, images, shapes):
    parts = unpack(theta, shapes)
    h = hidden(pack(parts[:8]), images, shapes)
    return F.linear(h, parts[8], parts[9])


def self_state(theta):
    parts = unpack(theta, SHAPES)
    parts[0], parts[1], parts[2] = parts[0][:1], parts[1][:1], parts[2][:, :1]
    return pack(parts), [tuple(p.shape) for p in parts]


def pool_gap(z):
    values = F.unfold(z, 2, stride=2).reshape(z.shape[0], z.shape[1], 4, -1)
    ordered = values.sort(dim=2).values
    return ordered[:, :, -1]-ordered[:, :, -2]


def reference():
    gen = torch.Generator().manual_seed(100910)
    def positive(shape, lo, hi):
        return lo+(hi-lo)*torch.rand(shape, generator=gen)
    base = positive((3, 32, 32), .015, .055)
    images = base[None]+torch.arange(N)[:, None, None, None]*(.02/N)
    images[0, :, 1:6, 1:6] = 1
    parts = [positive(s, .001, .002) for s in SHAPES]
    for index in (1, 3, 5, 7):
        parts[index] = positive(SHAPES[index], .1, .11)
    parts[0][0] = 1
    parts[1][0] = -74.5
    parts[2] = positive(SHAPES[2], .0001, .0002)
    parts[2][:, 0, 2, 2] = positive((16,), 1, 1.1)
    w = torch.full((100,), .01)
    a = torch.tensor([-.4]*6+[.6]*4)
    psi = (hidden(pack(parts[:8]), images[:1], SHAPES)[0]*w).sum()
    parts[8] = a[:, None]*w[None]
    parts[9] = -a*psi
    mean_patch = torch.zeros(75)
    minimum_full_pool_gap = float('inf')
    maximum_inactive_preactivation = -float('inf')
    active_sites = 0
    for start in range(0, N, 32):
        batch = images[start:start+32]
        mean_patch += F.unfold(batch, 5, padding=2).sum((0, 2))/(N*1024)
        _, zs = hidden(pack(parts[:8]), batch, SHAPES, extra=True)
        z1, z2, z3, z4 = zs
        active_sites += int((z1[:, 0] > 0).sum())
        dead = z1[:, 0][z1[:, 0] < 0]
        maximum_inactive_preactivation = max(maximum_inactive_preactivation, float(dead.max()))
        assert z1[:, 1:].min() > 0 and min(z2.min(), z3.min(), z4.min()) > 0
        gaps = [pool_gap(z1[:, 1:]).min(), pool_gap(z2).min()]
        minimum_full_pool_gap = min(minimum_full_pool_gap, *(float(v) for v in gaps))
    assert active_sites == 1 and minimum_full_pool_gap > 0
    directions = [torch.zeros(s) for s in SHAPES]
    directions[0][0] = mean_patch.reshape(3, 5, 5)
    directions[1][0] = 1
    u = pack(directions)
    assert u.min() == 0 and u.sum() >= 1
    return pack(parts), images, u, a, w, dict(
        all_image_count=N, target_active_site_count=active_sites,
        maximum_inactive_target_preactivation=maximum_inactive_preactivation,
        minimum_full_nonconstant_pool_gap=minimum_full_pool_gap,
        mean_direction_sum=float(u.sum()), minimum_input=float(images.min()),
        maximum_nontarget_input=float(images[1:].max()),
    )


def capacity_envelope(theta, images, u, shapes, a, w):
    hp = sum(math.prod(s) for s in shapes[:8])
    v, du = theta[:hp], u[:hp]
    target = images[:1]
    _, zs = hidden(v, target, shapes, extra=True)
    z2 = zs[1]
    windows = F.unfold(z2, 2, stride=2).reshape(1, shapes[2][0], 4, -1)
    is_nonconstant = windows.max(2).values != windows.min(2).values
    gaps = pool_gap(z2)
    assert gaps[is_nonconstant].min() > 0
    equal_windows = int((~is_nonconstant).sum())
    feature_fn = lambda x: hidden(x, target, shapes)[0]
    psi_fn = lambda x: (feature_fn(x)*w).sum()
    h, hu = torch.func.jvp(feature_fn, (v,), (du,))
    d, dp = torch.func.jvp(torch.func.grad(psi_fn), (v,), (du,))
    assert h.min() > 0 and hu.min() > 0 and d.min() >= 0 and dp.min() >= 0
    _, other_hu = torch.func.jvp(lambda x: hidden(x, images[1:3], shapes), (v,), (du,))
    assert other_hu.abs().max() == 0
    env_parts = unpack(v, shapes[:8])
    env_parts[1] = env_parts[1].abs()
    env = pack(env_parts)
    all_ones = torch.ones((1, 3, 32, 32))
    env_fn = lambda x: hidden(x, all_ones, shapes, envelope=True)[0]
    h_env = env_fn(env)
    d_env = torch.func.grad(lambda x: (env_fn(x)*w).sum())(env)
    assert (h <= h_env).all() and (d <= d_env).all()
    hn, dn = math.sqrt(N)*h_env.norm(), math.sqrt(N)*d_env.norm()
    anorm2 = a.square().sum()
    trace_prime = 2*CLASSES*(h*hu).sum()+2*anorm2*(d*dp).sum()
    r_upper = CLASSES*(hn.square()+N)+anorm2*dn.square()
    t_upper = 2*CLASSES*hn*hu.norm()+2*anorm2*dn*dp.norm()
    ridge = 2*r_upper*t_upper/trace_prime
    lower = trace_prime/(2*ridge)-r_upper*t_upper/(2*ridge.square())
    assert trace_prime > 0 and lower > 0
    # Audit all raw logit blocks and the u derivative on three stated images.
    subset = images[:3]
    jac_fn = torch.func.jacrev(lambda x: forward(x, subset, shapes).flatten())
    jac, jac_u = torch.func.jvp(jac_fn, (theta,), (u,))
    feat_fn = lambda x: hidden(x, subset, shapes)
    features, features_u = torch.func.jvp(feat_fn, (v,), (du,))
    grad_fn = torch.func.jacrev(lambda x: feat_fn(x)@w)
    ds, ds_u = torch.func.jvp(grad_fn, (v,), (du,))
    assert ds_u[1:].abs().max() == 0 and jac_u[CLASSES:].abs().max() == 0
    identity = torch.eye(CLASSES)
    aa = a[:, None]*a[None]
    expected_k = torch.kron(features@features.T+torch.ones((3, 3)), identity)+torch.kron(ds@ds.T, aa)
    expected_kp = torch.kron(features_u@features.T+features@features_u.T, identity)+torch.kron(ds_u@ds.T+ds@ds_u.T, aa)
    actual_k = jac@jac.T
    actual_kp = jac_u@jac.T+jac@jac_u.T
    k_error = float((actual_k-expected_k).abs().max())
    kp_error = float((actual_kp-expected_kp).abs().max())
    assert k_error < 1e-9 and kp_error < 1e-9
    kappa = d[:math.prod(shapes[0])].reshape(shapes[0])[0].flatten()
    assert (kappa-kappa[0]).abs().max() < 1e-14
    b1_offset = math.prod(shapes[0])
    assert abs(float(d[b1_offset]-kappa[0])) < 1e-14
    return dict(
        raw_parameter_count=theta.numel(), target_kappa=float(kappa[0]),
        target_second_pool_nonconstant_gap=float(gaps[is_nonconstant].min()),
        target_second_pool_identical_bias_only_windows=equal_windows,
        all_data_trace_derivative=float(trace_prime),
        all_data_kernel_trace_upper=float(r_upper),
        all_data_derivative_nuclear_upper=float(t_upper),
        sufficient_ridge=float(ridge),
        all_data_capacity_derivative_lower_at_own_ridge=float(lower),
        subset_images=[0, 1, 2], raw_kernel_identity_max_error=k_error,
        raw_kernel_derivative_identity_max_error=kp_error,
        nontarget_feature_directional_derivative_exactly_zero=True,
    )


def exact_margin(kappa_floor=Q(1, 1000)):
    beta1, beta2, eps, q = Q(9, 10), Q(999, 1000), Q(1, 10**8), Q(3, 5)
    overlap = (1-beta1)*(1-beta2)*(beta1*beta2)**149
    e = 16*eps/kappa_floor
    coefficient = (1-q)*(2*q-1)/(2*(q+e)**2)
    delta = coefficient*overlap
    def record(x):
        return dict(fraction=str(x), approximate=float(x))
    assert delta > 0
    return dict(
        condition='kappa >= 1/1000; rational bound, with native value checked separately in float64',
        latest_visit_lag_upper=149, kappa_floor=record(kappa_floor),
        overlap_lower=record(overlap), phase_quotient_lower=record(delta),
        task_mean_field_lower_using_U_ge_1=record(PHASES*delta),
    )


def positive_mean_geometry():
    total_sites = N*32**2
    b0, amplitude = Q(1, 10), Q(9, 10)
    tau = amplitude/(4*total_sites)
    mean_lower = amplitude/total_sites-2*tau
    patch_floor = b0-2*tau
    assert tau > 0 and mean_lower > 0 and patch_floor > 0
    return dict(
        scope='Exact scalar geometry for the separate analytical positive-mean reference; not a second native Jacobian or routing run.',
        total_image_spatial_sites=total_sites,
        inactive_preactivation_interval=[str(-2*tau), str(-tau)],
        positive_mean_lower=dict(fraction=str(mean_lower), approximate=float(mean_lower)),
        target_patch_coefficient_floor=dict(fraction=str(patch_floor), approximate=float(patch_floor)),
    )


def run():
    theta, images, u, a, w, geometry = reference()
    full = capacity_envelope(theta, images, u, SHAPES, a, w)
    self_theta, self_shapes = self_state(theta)
    self_u, _ = self_state(u)
    isolated = capacity_envelope(self_theta, images, self_u, self_shapes, a, w)
    assert full['target_kappa'] > .001
    # All ten labels: native batch CE gradients on the entire target raw block.
    block = torch.cat((torch.arange(75), torch.tensor([1200])))
    logits = forward(theta, images[:16], SHAPES)
    errors = []
    for label in range(CLASSES):
        labels = torch.arange(16) % CLASSES
        labels[0] = label
        gradient = torch.func.grad(lambda x: F.cross_entropy(forward(x, images[:16], SHAPES), labels))(theta)
        expected = -full['target_kappa']*float(a[label])/16
        errors.append(float((gradient[block]-expected).abs().max()))
    # A batch with no target has identically zero target-block gradients.
    labels = torch.arange(16) % CLASSES
    absent = torch.func.grad(lambda x: F.cross_entropy(forward(x, images[1:17], SHAPES), labels))(theta)
    assert absent[block].abs().max() == 0
    assert max(errors) < 1e-13 and logits[0].abs().max() < 1e-14
    common_ridge = max(full['sufficient_ridge'], isolated['sufficient_ridge'])
    for report in (full, isolated):
        tau, r, t = (report[k] for k in ('all_data_trace_derivative', 'all_data_kernel_trace_upper', 'all_data_derivative_nuclear_upper'))
        report['capacity_lower_at_common_ridge'] = tau/(2*common_ridge)-r*t/(2*common_ridge**2)
        assert report['capacity_lower_at_common_ridge'] > 0
    return dict(
        scope='Finite native identities and analytic-envelope quantities; no training, stochastic trajectory, full-data NTK formation, or certified practical rate.',
        settings=dict(N=N, batch=BATCH, epochs=EPOCHS, phases=PHASES, classes=CLASSES),
        geometry=geometry, full=full, literal_self=isolated, common_ridge=common_ridge,
        batch_ce_target_block_max_error=max(errors),
        absent_target_batch_gradient_exactly_zero=True,
        exact_stationary_margin=exact_margin(),
        analytical_positive_mean_variant=positive_mean_geometry(),
        numeric_status='Native quantities are float64 checks, not outward-rounded interval certificates. Exact Fraction phase bound is conditional on its stated kappa floor. Analytic nonemptiness and stochastic conclusion are in adam_tenclass_finite_drift.md.',
        source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/cnn_drive_1009/adam_tenclass_finite_drift.json'))
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('geometry', 'common_ridge', 'batch_ce_target_block_max_error')}, ensure_ascii=False, indent=2))
    print('output:', args.output)

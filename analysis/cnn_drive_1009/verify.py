"""Deterministic checks of theorems, including actual Conv/MaxPool autograd.

These are constructed mathematical examples, not RL-CIFAR training runs.
Run with the project's Python environment. No datasets, network, or GPU needed.
"""
from __future__ import annotations

import itertools
import json
import math
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as nnf

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "cnn_drive_1009"


def inv2(a):
    determinant = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    return [[a[1][1] / determinant, -a[0][1] / determinant],
            [-a[1][0] / determinant, a[0][0] / determinant]]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def transpose(a):
    return list(map(list, zip(*a)))


def mm(a, b):
    return [[dot(row, col) for col in transpose(b)] for row in a]


def mv(a, b):
    return [dot(row, b) for row in a]


def rational_fit(h, r, lam):
    k = mm(h, transpose(h))
    p = inv2([[k[i][j] + lam * (i == j) for j in range(2)] for i in range(2)])
    q = mm(k, mm(p, p))
    exact = dot(r, mv(q, [row[0] for row in h]))
    enumerated = F(0)
    labels = list(itertools.product([F(-1), F(1)], repeat=2))
    for old in labels:
        v = mv(transpose(h), mv(p, old))
        pred = mv(h, v)
        for new in labels:
            enumerated += v[0] * dot(r, [a - b for a, b in zip(pred, new)]) / 16
    assert exact == enumerated
    return exact


def fitted_conv_example(kind):
    x = torch.zeros(2, 3, 6, 6)
    w = torch.zeros(2, 3, 5, 5)
    b = torch.zeros(2)
    if kind == "certificate":
        x[0, 0, :5, :5] = 1
        x[1, 1, :5, :5] = 1
        w[0, 0], w[0, 1] = 1 / 25, 1 / 250
        w[1, 0], w[1, 1] = 1 / 250, 1 / 25
        hf = [[F(1), F(1, 10), F(1)], [F(1, 10), F(1), F(1)]]
        rf = [F(89, 8), F(89, 8)]
    else:
        x[0, 0, :5, :5] = .1
        x[1, 0, :5, :5] = 1
        w[0, 0], w[1, 0], b[1] = 2 / 5, 86 / 75, 4 / 3
        hf = [[F(1), F(21, 5), F(1)], [F(10), F(30), F(1)]]
        rf = [F(1691, 800), F(971, 80)]
    rational = rational_fit(hf, rf, F(1, 10))
    isolated = rational_fit([[row[0], row[-1]] for row in hf], rf, F(1, 10))
    assert isolated > 0  # Literal isolated-channel fitted head, including trained bias.
    w.requires_grad_()
    b.requires_grad_()
    z = nnf.conv2d(x, w, b)
    order = z.flatten(2).sort(dim=2, descending=True).values
    gap = order[:, :, 0] - order[:, :, 1]
    assert float(gap.detach().min()) > 0
    h0 = nnf.max_pool2d(nnf.relu(z), 2).flatten(1)
    h = torch.cat([h0, torch.ones(2, 1)], dim=1)
    mean_grad = torch.autograd.grad(z[:, 0].mean(), [w, b], retain_graph=True)
    r = []
    for n in range(2):
        gh = torch.autograd.grad(h[n, 0], [w, b], retain_graph=True)
        r.append(sum((a * c).sum() for a, c in zip(gh, mean_grad)))
    r = torch.stack(r)
    assert torch.allclose(r, torch.tensor(list(map(float, rf))), atol=1e-12, rtol=0)
    assert torch.allclose(h, torch.tensor([[float(v) for v in row] for row in hf]),
                          atol=1e-12, rtol=0)
    hd = h.detach()
    k = hd @ hd.T
    p = torch.linalg.inv(k + .1 * torch.eye(2))
    q = k @ p @ p
    formula = float(r @ q @ hd[:, 0])
    values = []
    labels = list(itertools.product([-1., 1.], repeat=2))
    for old in labels:
        # Deliberately detach: the actual new-task gradient holds the fitted head fixed.
        v = (hd.T @ p @ torch.tensor(old)).detach()
        for new in labels:
            loss = .5 * ((h @ v - torch.tensor(new)) ** 2).sum()
            grad = torch.autograd.grad(loss, [w, b], retain_graph=True)
            values.append(sum(float((a * c).sum()) for a, c in zip(grad, mean_grad)))
    enumerated = sum(values) / len(values)
    assert abs(enumerated - formula) < 1e-10
    assert abs(enumerated - float(rational)) < 1e-10
    diagonal = q.diag()
    normalized = q / torch.sqrt(diagonal[:, None] * diagonal[None, :])
    epsilon = float(torch.linalg.matrix_norm(normalized - torch.eye(2), ord=2))
    a, c = r * diagonal.sqrt(), hd[:, 0] * diagonal.sqrt()
    cosine = float(a @ c / (a.norm() * c.norm()))
    ratios = hd[:, 0] / r
    spread = float(ratios.max() / ratios.min())
    range_bound = 2 * math.sqrt(spread) / (1 + spread)
    if kind == "certificate":
        assert epsilon < range_bound <= cosine + 1e-12
        assert enumerated > 0
    else:
        assert enumerated < 0
        assert float(r @ hd[:, 0]) > 0
    return dict(kind=kind, H=hd.tolist(), mean_direction=r.tolist(),
                minimum_winner_gap=float(gap.detach().min()), rational=str(rational),
                isolated_with_output_bias_rational=str(isolated),
                isolated_with_output_bias_gradient=float(isolated),
                formula_gradient_sum_loss=formula, enumerated_gradient_sum_loss=enumerated,
                gradient_mean_loss=enumerated / 2,
                Q_eigenvalues=torch.linalg.eigvalsh(q).tolist(),
                K_eigenvalues=torch.linalg.eigvalsh(k).tolist(),
                epsilon=epsilon, self_angle=cosine, range_angle_lower_bound=range_bound,
                certificate_pass=epsilon < cosine)


def inverse_root(a):
    eig, u = np.linalg.eigh(a)
    assert eig.min() > 0
    return (u * (1 / np.sqrt(eig))) @ u.T


def capacity_certificate():
    # Actual 1x1 Conv on 6-channel, 2x2 images, followed by MaxPool(2).
    n = 6
    amplitudes = np.array([1, 1.05, .9, 1.1, 1.2, .8])
    patches = np.empty((n, 4, n))
    for i in range(n):
        for s in range(4):
            patches[i, s] = np.ones(n) + amplitudes[i] * np.eye(n)[i] + .1 * s
    x = torch.tensor(patches.transpose(0, 2, 1).reshape(n, n, 2, 2))
    mu = patches.mean(axis=(0, 1))
    w = np.vstack([np.array([1, 1, -1, -1, -1, -1]), np.eye(n)])
    biases = np.r_[2., np.full(n, -1.7)]
    v = np.ones(n + 1)
    for c in range(n):
        selected = np.r_[patches[c, 3], 1.]
        v[c + 1] = math.sqrt((2 - (amplitudes[c] - .4) ** 2) / (selected @ selected))
    theta = torch.tensor(np.c_[w, biases].ravel(), requires_grad=True)
    v_t = torch.tensor(v, requires_grad=True)
    # Include bias in the Euclidean mean-preactivation direction, unlike a bias-only shift.
    direction = torch.zeros_like(theta)
    direction[:n + 1] = torch.tensor(np.r_[mu, 1.])

    def features(th):
        pars = th.reshape(n + 1, n + 1)
        z = nnf.conv2d(x, pars[:, :n, None, None], pars[:, n])
        return nnf.max_pool2d(nnf.relu(z), 2).flatten(1)

    def output(th, head):
        return features(th) @ head

    h = features(theta)
    jac_conv = torch.autograd.functional.jacobian(lambda th: output(th, v_t), theta,
                                                 create_graph=True)
    ht = h[:, :1]
    _, r_all = torch.autograd.functional.jvp(features, theta, direction, create_graph=True)
    rt = r_all[:, :1]
    gate_target = jac_conv[:, :n + 1]
    gate_other = jac_conv[:, n + 1:]
    eye = torch.eye(n)
    # A finite output-bias kernel is retained. No unpenalized centering substitution.
    fixed = gate_target @ gate_target.T + torch.ones(n, n)
    other = gate_other @ gate_other.T + h[:, 1:] @ h[:, 1:].T
    assert torch.allclose(other, 2 * eye, atol=1e-11, rtol=0)
    a0 = .01 * eye + fixed + ht @ ht.T
    full = a0 + other
    phi = .5 * torch.linalg.slogdet(full)[1]
    autod = float(torch.autograd.grad(phi, theta)[0] @ direction)
    form = float((rt * torch.linalg.solve(full, ht)).sum())
    self_force = float((rt * torch.linalg.solve(a0, ht)).sum())
    assert abs(autod - form) < 1e-10
    ai = inverse_root(a0.detach().numpy())
    xx, xp = ai @ ht.detach().numpy(), ai @ rt.detach().numpy()
    bb = ai @ other.detach().numpy() @ ai
    beta = np.linalg.eigvalsh(bb)
    epsilon = float((beta[-1] - beta[0]) / (2 + beta[-1] + beta[0]))
    angle = abs(self_force) / (np.linalg.norm(xx) * np.linalg.norm(xp))
    # If this general sufficient certificate is conservative, report that honestly.
    passed = bool(epsilon < angle)
    assert passed
    assert form > 0 and self_force > 0
    return dict(N=n, channels=n + 1, windows=1, mean_direction_includes_bias=True,
                target_preactivation=[1., 1.05, -.9, -1.1, -1.2, -.8],
                other_kernel_is_2I_error=float(torch.abs(other - 2 * eye).max()),
                full_capacity_autograd=autod, full_capacity_formula=form,
                isolated_capacity_derivative=self_force,
                whitened_background_eigenvalues=beta.tolist(),
                epsilon_B=epsilon, self_angle=float(angle), certificate_pass=passed)


def multiple_window_check():
    # Random positive images and filters, unique pool winners with probability one.
    # One label per IMAGE. All windows share the same filter parameters.
    torch.manual_seed(1009)
    x = torch.rand(3, 2, 4, 6)
    theta = (.2 + torch.rand(2, 3)).flatten().requires_grad_()
    head = torch.randn(12)  # Both signs: six pooled windows per channel.
    direction = torch.zeros_like(theta)
    direction[:3] = torch.cat([x.mean(dim=(0, 2, 3)), torch.ones(1)])

    def features(th):
        p = th.reshape(2, 3)
        z = nnf.conv2d(x, p[:, :2, None, None], p[:, 2])
        return nnf.max_pool2d(nnf.relu(z), 2).flatten(1)

    def capacity(th):
        h = features(th)
        j = torch.autograd.functional.jacobian(lambda t: features(t) @ head,
                                               th, create_graph=True)
        k = h @ h.T + j @ j.T + torch.ones(3, 3)
        return .5 * torch.linalg.slogdet(k + .2 * torch.eye(3))[1], k

    phi, k = capacity(theta)
    grad = torch.autograd.grad(phi, theta)[0]
    h = features(theta)
    _, r = torch.autograd.functional.jvp(features, theta, direction)
    formula = float((r * torch.linalg.solve(k.detach() + .2 * torch.eye(3), h)).sum())
    actual = float(grad @ direction)
    epsilon = 1e-6
    fd = float((capacity(theta + epsilon * direction)[0] -
                capacity(theta - epsilon * direction)[0]) / (2 * epsilon))
    assert abs(actual - formula) < 1e-10
    assert abs(actual - fd) < 1e-7
    # F1 with full multiclass labels, all 2^3 old scalar labels; new labels analytically averaged.
    hd, rd = h.detach(), r.detach()
    kh = hd @ hd.T
    p = torch.linalg.inv(kh + .3 * torch.eye(3))
    q = kh @ p @ p
    expected = float((rd * (q @ hd)).sum())
    enumeration = 0.
    for labels in itertools.product([-1., 1.], repeat=3):
        fitted = (hd.T @ p @ torch.tensor(labels)).detach()
        loss = .5 * (features(theta) @ fitted).square().sum()
        g = torch.autograd.grad(loss, theta)[0]
        enumeration += float(g @ direction) / 8
    assert abs(enumeration - expected) < 1e-10
    return dict(N=3, channels=2, windows_per_channel=6, mixed_sign_head=True,
                capacity_formula=formula, capacity_autograd=actual, capacity_finite_difference=fd,
                fitted_head_formula=expected, fitted_head_label_enumeration=enumeration)


def rank_deficient_check():
    h = np.array([[1., .1, 1.], [.1, 1., 1.]] * 2)
    target = h[:, :1]
    r = np.full_like(target, 89 / 8)
    k = h @ h.T
    eig, u = np.linalg.eigh(k)
    positive = eig > 1e-12
    up = u[:, positive]
    vals = eig[positive] / (eig[positive] + .1) ** 2
    q = (up * vals) @ up.T
    projected = up @ up.T @ r
    angle = float(np.sum(r * target) / (np.linalg.norm(projected) * np.linalg.norm(target)))
    epsilon_range = float((vals.max() - vals.min()) / (vals.max() + vals.min()))
    diagonal = np.diag(q)
    epsilon_full = float(np.linalg.norm(q / np.sqrt(diagonal[:, None] * diagonal[None, :])
                                         - np.eye(4), 2))
    assert positive.sum() == 2
    assert epsilon_full >= 1 - 1e-12
    assert epsilon_range < angle
    force = float(np.sum(r * (q @ target)))
    assert force > 0
    return dict(N=4, rank=2, epsilon_full=epsilon_full,
                epsilon_range=epsilon_range, projected_self_angle=angle,
                gradient_sum_loss=force, F2_pass=False, F3_pass=True)


def ce_one_step_and_late_counterexample():
    h = torch.tensor([[1., 2.], [2., 1.]])
    r = torch.tensor([[9., 0.], [1.5, 0.]])
    y_uniform = torch.full((2, 2), .5)
    results = []
    for steps in [1, 2, 10, 100, 1000]:
        gradients = []
        for labels in itertools.product(range(2), repeat=2):
            y = torch.tensor(labels)
            v = torch.zeros(2, 2, requires_grad=True)
            for _ in range(steps):
                old_loss = nnf.cross_entropy(h @ v, y)
                grad = torch.autograd.grad(old_loss, v)[0]
                v = (v - grad).detach().requires_grad_()
            # New labels are averaged exactly as uniform probabilities.
            logits = h @ v.detach()
            g = float((r * ((logits.softmax(dim=1) - y_uniform) @ v.detach().T / 2)).sum())
            gradients.append(g)
        mean = sum(gradients) / 4
        results.append(dict(steps=steps, expected_gradient=mean))
    assert results[0]["expected_gradient"] > 0
    assert results[-1]["expected_gradient"] < 0
    return dict(H=h.tolist(), R=r.tolist(), learning_rate=1.,
                one_step_positive=True, fully_learned_sign_not_guaranteed=True, trajectory=results)


def main():
    checks = {
        "scope": "Constructed finite CNN certificates and counterexamples, not an RL-CIFAR trajectory result",
        "fitted_head": [fitted_conv_example(k) for k in ["certificate", "counterexample"]],
        "capacity_self_vs_full": capacity_certificate(),
        "multiple_windows": multiple_window_check(),
        "rank_deficient": rank_deficient_check(),
        "ce_head_training": ce_one_step_and_late_counterexample(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()

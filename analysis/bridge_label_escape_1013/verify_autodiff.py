"""Independent float64 checks of analytic statements; no training trajectories.

The exact counterexample and the uniform CE error bound are proved separately.
Here the capacity derivative is obtained by differentiating logdet, and loss
gradients by differentiating the actual forward network, not by copying the
hand-coded update in verify_exact.py.
"""
from itertools import product
import json
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
N, H = 5, 3
X = torch.eye(N)
THETA = X @ X.T + torch.ones(N, N)
XBAR = torch.cat((X.mean(dim=0), torch.ones(1)))
Z = torch.tensor([[-4., -4., -4.], [-7., -4., -9.], [2., 4., 1.],
                  [-8., -3., -9.], [8., 5., 6.]])
INITIAL = torch.cat((Z.T.flatten(), torch.zeros(H), torch.ones(H), torch.zeros(1)))
TARGET = 2


def unpack(theta):
    return theta[:15].reshape(H, N), theta[15:18], theta[18:21], theta[21]


def forward(theta):
    w, b, v, c = unpack(theta)
    z = X @ w.T + b
    return torch.relu(z) @ v + c


def capacity(theta, position, alone=False):
    w, b, v, _ = unpack(theta)
    selector = torch.nn.functional.one_hot(torch.tensor(TARGET), H).to(theta.dtype)
    w = w + position * selector[:, None] * XBAR[:-1]
    b = b + position * selector * XBAR[-1]
    z = X @ w.T + b
    h = torch.relu(z)
    d = (z > 0).to(theta.dtype)
    k = torch.ones(N, N)
    for j in ([TARGET] if alone else range(H)):
        k = k + v[j] ** 2 * torch.outer(d[:, j], d[:, j]) * THETA
        k = k + torch.outer(h[:, j], h[:, j])
    return .5 * torch.linalg.slogdet(torch.eye(N) + k).logabsdet


def force(theta, alone=False):
    t = torch.zeros((), requires_grad=True)
    return torch.autograd.grad(capacity(theta, t, alone), t, create_graph=True)[0]


def actual_step(theta, labels, eta, kind):
    pred = forward(theta)
    if kind == "mse":
        loss = .5 * ((pred - labels) ** 2).mean()
    else:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, labels)
    grad = torch.autograd.grad(loss, theta)[0]
    return theta.detach() - eta * grad.detach(), -grad.detach()


theta = INITIAL.clone().requires_grad_()
a0, s0 = force(theta), force(theta, True)
assert abs(a0.item() + 36 / 3377) < 1e-13
assert abs(s0.item() - 336 / 2285) < 1e-13

# Independent difference of logdet in the physical position direction.
dt = torch.tensor(1e-4)
central = (capacity(theta, dt) - capacity(theta, -dt)) / (2 * dt)
assert abs(central.item() - a0.item()) < 1e-8

grad_a = torch.autograd.grad(a0, theta)[0]
j = torch.autograd.functional.jacobian(forward, theta)
sens = j @ grad_a
p = torch.sigmoid(forward(theta)).detach()
dmax = (torch.clamp_min(sens, 0) - p * sens).mean().item()
dmin = (torch.clamp_max(sens, 0) - p * sens).mean().item()
ce_derivatives, rows = [], []
for kind, labels_set, eta in (("mse", (-1., 1.), 1e-3), ("bce", (0., 1.), 1e-6)):
    for labels in product(labels_set, repeat=N):
        th = INITIAL.clone().requires_grad_()
        y = torch.tensor(labels)
        updated, direction = actual_step(th, y, eta, kind)
        aa, ss = force(updated).item(), force(updated, True).item()
        assert aa < 0 < ss
        if kind == "bce":
            derivative = (grad_a @ direction).item()
            formula = ((y - p) * sens).mean().item()
            assert abs(derivative - formula) < 1e-13
            ce_derivatives.append(derivative)
        rows.append(dict(loss=kind, labels=labels, A=aa, self=ss))

assert abs(max(ce_derivatives) - dmax) < 1e-13
assert abs(min(ce_derivatives) - dmin) < 1e-13
assert dmax > 0 > dmin

# Degenerate but genuinely held state: capacity is even in v, while the
# hidden-logit Jacobian vanishes at v=0. This checks their product, not just A.
zero = INITIAL.clone()
zero[18:21] = 0
zero.requires_grad_()
az, sz = force(zero), force(zero, True)
gaz = torch.autograd.grad(az, zero)[0]
jz = torch.autograd.functional.jacobian(forward, zero)
assert abs(az.item() + 27 / 640) < 1e-13
assert abs(sz.item() - 168 / 895) < 1e-13
assert torch.count_nonzero(jz @ gaz).item() == 0

# Algebraic invariant of the separate one-input binary-CE example.
for z, v, eta in ((2., .5, .2), (1.1, -1., .9), (3., 0., .7)):
    for label in (0., 1.):
        r = torch.sigmoid(torch.tensor(v * z)).item() - label
        zn, vn = z - eta * r * v, v - eta * r * z
        rhs = (1 - eta ** 2 * r ** 2) * (z ** 2 - v ** 2)
        assert abs(zn ** 2 - vn ** 2 - rhs) < 1e-12
        assert zn > abs(vn)

report = dict(
    verification="independent autodiff consistency check, not a proof by simulation",
    initial=dict(A=a0.item(), self=s0.item()),
    position_finite_difference_error=abs(central.item() - a0.item()),
    ce_direction_extrema=dict(maximum=dmax, minimum=dmin),
    all_updates_strictly_held=len(rows),
    zero_readout=dict(A=az.item(), self=sz.item(), transverse_product_norm=0.),
    rows=rows,
)
out = Path(__file__).with_name("autodiff_checks.json")
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))

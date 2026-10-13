"""Independent autodiff / finite-label checks, not empirical training evidence.

The CE check enumerates all 3**3 fresh label tasks, holding the task-start
state fixed. The regression check runs an entire fixed-label task and compares
with the exact frozen-Jacobian task map. No trajectory or large simulation.
"""
import itertools
import json
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
N, C, H = 3, 3, 2
X = torch.cat((torch.eye(N), torch.ones(N, 1)), dim=1)
LEAK, SCALE, ETA = .2, .7, .013
W = torch.tensor([[-1., 1., 2., 0.], [2., 3., 4., 0.]])
V = torch.tensor([[1., -.3, .2], [.4, .7, -.5]])
INITIAL = torch.cat((W.flatten(), V.flatten(), torch.tensor([.1, -.2, .05])))
P = torch.eye(C) - torch.ones(C, C) / C


def unpack(theta):
    return theta[:8].reshape(H, 4), theta[8:14].reshape(H, C), theta[14:]


def logits(theta):
    w, v, b = unpack(theta)
    z = X @ w.T
    h = torch.where(z > 0, z, LEAK * z)
    return h @ v + b


POSITION = torch.zeros_like(INITIAL)
POSITION[:4] = X.mean(dim=0)
OWN_COLUMNS = [0, 1, 2, 3, 8, 9, 10, 14, 15, 16]


def capacity(theta, alone=False):
    jac = torch.autograd.functional.jacobian(logits, theta, create_graph=True).reshape(N*C, -1)
    if alone:
        jac = jac[:, OWN_COLUMNS]
    kernel = jac @ jac.T
    return .5 * torch.linalg.slogdet(torch.eye(N*C) + kernel).logabsdet


def force(theta, alone=False):
    grad = torch.autograd.grad(capacity(theta, alone), theta, create_graph=True)[0]
    return grad @ POSITION


theta = INITIAL.clone().requires_grad_()
aa, ss = force(theta), force(theta, True)
chi = aa / ss
normal = torch.autograd.grad(chi, theta)[0].detach()
pred = logits(theta).detach()
prob = torch.softmax(pred, dim=1)
jac = torch.autograd.functional.jacobian(logits, theta).detach()
sens = torch.einsum('ncp,p->nc', jac, normal)
mean_formula = ((sens.mean(dim=1)) - (prob * sens).sum(dim=1)).mean()
var_formula = ((sens @ P)**2).sum() / (N*N*C)
expected_direction = torch.einsum('ncp,nc->p', jac, torch.ones(N,C)/C-prob)/N
derivatives, relative_steps, directions = [], [], []
for labels in itertools.product(range(C), repeat=N):
    t = INITIAL.clone().requires_grad_()
    loss = torch.nn.functional.cross_entropy(logits(t), torch.tensor(labels))
    direction = -torch.autograd.grad(loss, t)[0]
    derivatives.append(normal @ direction)
    directions.append(direction)
    delta_w = (ETA*direction[:8]).reshape(H, 4)
    relative_steps.append(delta_w[1] - SCALE*delta_w[0])
derivatives = torch.stack(derivatives)
directions = torch.stack(directions)
relative_steps = torch.stack(relative_steps)
assert torch.allclose(directions.mean(0), expected_direction, atol=1e-13, rtol=1e-12)
assert abs(derivatives.mean()-mean_formula) < 1e-12
assert abs(derivatives.var(unbiased=False)-var_formula) < 1e-12

d = torch.where(X @ W.T > 0, 1., LEAK)
contrast = (d[:, 1, None]*V[1] - SCALE*d[:, 0, None]*V[0]) @ P
cov_formula = ETA**2/(N*N*C) * torch.einsum('n,ni,nj->ij', (contrast**2).sum(1), X, X)
centered = relative_steps - relative_steps.mean(0)
cov_enum = centered.T @ centered / len(relative_steps)
assert torch.allclose(cov_enum, cov_formula, atol=1e-14, rtol=1e-12)
lower = ETA**2 * SCALE**2 * (1-LEAK)**2 * ((V[0] @ P)**2).sum()/(2*N*N*C)
assert torch.trace(cov_formula) >= lower
projection_variances = torch.einsum('ni,ij,nj->n', X, cov_formula, X)
assert (projection_variances >= lower).all()
dstar = abs(SCALE)*(1-LEAK)*torch.linalg.vector_norm(V[0] @ P)
motion_threshold = ETA*dstar/(2*N*C**.5)
motion_probability = (torch.linalg.vector_norm(relative_steps, dim=1) >= motion_threshold).double().mean()
assert motion_probability >= 1/C

# A whole fixed-label task, joint first-order linearization in all parameters.
j = jac.reshape(N*C, -1)
k = j @ j.T
ridge = .3
steps = 17
alpha = .5 / (torch.linalg.eigvalsh(k).max()+ridge)
gram = k + ridge*torch.eye(N*C)
bmap = j.T @ torch.linalg.solve(gram, torch.eye(N*C)-torch.linalg.matrix_power(torch.eye(N*C)-alpha*gram, steps))
flat_pred = pred.flatten()


def frozen_task(labels):
    delta = torch.zeros_like(INITIAL)
    for _ in range(steps):
        delta = delta + alpha * (j.T @ (labels-flat_pred-j@delta)-ridge*delta)
    return delta


# Symmetric basis quadrature has exactly mean zero and covariance sigma^2 I.
# This verifies the moments, not the Gaussian tail by Monte Carlo.
sigma = .8
labels = torch.cat((torch.eye(N*C), -torch.eye(N*C))) * (N*C)**.5 * sigma
deltas = torch.stack([frozen_task(y) for y in labels])
closed = (labels-flat_pred) @ bmap.T
assert torch.allclose(deltas, closed, atol=1e-12, rtol=1e-11)
mean_delta = -bmap @ flat_pred
cov_delta = sigma**2 * bmap @ bmap.T
assert torch.allclose(deltas.mean(0), mean_delta, atol=1e-12, rtol=1e-11)
dc = deltas-deltas.mean(0)
assert torch.allclose(dc.T@dc/len(deltas), cov_delta, atol=1e-12, rtol=1e-11)
margin_mean = chi.detach() + normal @ mean_delta
margin_var = normal @ cov_delta @ normal
assert margin_var > 0
exit_proxy = torch.special.ndtr(margin_mean/torch.sqrt(margin_var))

report = {
    'kind': 'finite algebra checks, not empirical evidence for bridge lifetime',
    'ce_all_label_vectors': len(derivatives),
    'ce_chi_mean_formula': mean_formula.item(),
    'ce_chi_mean_enumerated': derivatives.mean().item(),
    'ce_chi_variance_formula': var_formula.item(),
    'ce_chi_variance_enumerated': derivatives.var(unbiased=False).item(),
    'relative_covariance_max_error': (cov_enum-cov_formula).abs().max().item(),
    'relative_covariance_trace': torch.trace(cov_formula).item(),
    'straddling_lower_bound': lower.item(),
    'relative_preactivation_variances': projection_variances.tolist(),
    'relative_motion_threshold': motion_threshold.item(),
    'relative_motion_probability_enumerated': motion_probability.item(),
    'relative_motion_probability_lower_bound': 1/C,
    'fixed_label_task_steps': steps,
    'fixed_label_task_max_error': (deltas-closed).abs().max().item(),
    'linearized_margin_mean': margin_mean.item(),
    'linearized_margin_variance': margin_var.item(),
    'gaussian_linearized_positive_margin_probability': exit_proxy.item(),
    'initial_chi': chi.item(),
    'note': 'The identity-check state is not claimed to be held or task-trained.',
}
Path(__file__).with_name('identity_checks.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))

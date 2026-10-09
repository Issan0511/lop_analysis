"""Finite algebra/autograd check; infinite-time claims require the analytic proof."""
from pathlib import Path
import json
import math
from fractions import Fraction as Q
import torch
from torch import nn
from torch.nn import functional as F

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
RHO = torch.tensor([.4, .8])
TILE = torch.tensor([[1., .2, .7, .3], [.4, .1, .2, .1],
                     [.6, .3, .8, .4], [.2, .1, .5, .2]])
PATTERN = torch.cat([TILE, TILE], dim=1)
IMAGES = RHO[:, None, None, None] * PATTERN[None, None].repeat(1, 3, 1, 1)
MU = IMAGES.mean().item()
B1, B2, EPS, K = .9, .999, 1e-8, 73.
D0, POWER, TASKS, H = .01, .75, 60, 4


def rational_probability_certificate():
    """A conservative exact upper bound for the analytic averaging error.

    For r=3/4 use sum n^-3/2 <=3 and sum log(n)n^-3/2 <=5.
    Replace -log(rho) by its lower bound 1-rho, log_+(x) by x,
    delta*log(1/delta) by 1. All operations below are rational.
    """
    a0, cap, eps, kk, hh = Q(1, 5), Q(2, 5), Q(1, 10**8), Q(73), Q(4)
    b1, b2 = Q(9, 10), Q(999, 1000)
    zeta = Q(1999, 2000)  # Strictly exceeds sqrt(beta2).
    assert zeta*zeta > b2 and kk*kk*(1-b2)*(1-b1*b1/b2) > 1
    rho = zeta**4
    avg_rho, avg_rho2 = Q(3, 5), Q(2, 5)
    gmax = 12*cap**2*avg_rho
    lip = 24*cap*avg_rho + 1296*cap**4*avg_rho2
    hmax = lh = 1/kk
    move0 = lip*hh*hmax*kk*(1+kk)/eps
    init0 = gmax/eps*(1+1/(1-b1)+2*kk)
    move = hmax*move0 + hh*lh*hmax*kk**2
    init = hmax*init0
    bf = bb = hh*hmax*kk
    lf = hh*(hmax*(1+kk)*lip/eps + lh*kk)
    cq = 2*gmax*(1+kk)/eps
    vv = max(2*bf, hh*hmax*cq/rho)
    uu = vv/(1-rho)
    s2, slog, exponent, failure = Q(3), Q(5), Q(3, 4), Q(1, 10)
    assert Q(6)**2 >= s2/failure
    ctrack = init*zeta/(1-zeta) + move*hh/(1-zeta)*s2
    xx = vv/(lf*bb)
    crho = 1+1/(1-rho)
    ccorr = 2*uu + 2*lf*bb*(crho*s2 + (s2*xx+s2+exponent*slog)/(1-rho))
    cnoise = 2*uu*6
    ctotal = ctrack+ccorr+cnoise
    delta = min(Q(1, 100), a0/(8*hh), a0/(4*ctotal))
    error_upper = ctotal*delta
    assert delta > 0 and error_upper <= a0/4
    assert hh*delta < a0/4 and b1/(1-delta)**2 < 1
    def rep(x):
        return {"fraction": str(x), "approximate": float(x)}
    return {"scope": "Exact rational analytic certificate, not an empirical success estimate.",
            "failure_probability": rep(failure), "initial_amplitude": rep(a0), "upper_barrier": rep(cap),
            "reuse_H": int(hh), "exponent": rep(exponent), "G": rep(gmax), "L": rep(lip),
            "linear_error_coefficient": rep(ctotal), "admissible_delta0": rep(delta),
            "uniform_error_upper": rep(error_upper),
            "finite_raw_check_delta0_is_different": True}


class CNN(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.c1 = nn.Conv2d(3, 4, 1, bias=False)
        self.c2 = nn.Conv2d(4, 3, 1, bias=False)
        self.head = nn.Linear(6, 2, bias=False)
        with torch.no_grad():
            self.c1.weight.fill_(a)
            self.c2.weight.fill_(a)
            self.head.weight[0].fill_(a)
            self.head.weight[1].fill_(-a)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.head(x.flatten(1))


def forward_flat(t, width):
    n = 3 * width
    w1, w2, vh = t[:n].reshape(width, 3, 1, 1), t[n:2*n].reshape(3, width, 1, 1), t[2*n:].reshape(2, 6)
    x = F.max_pool2d(F.relu(F.conv2d(IMAGES, w1)), 2)
    x = F.max_pool2d(F.relu(F.conv2d(x, w2)), 2)
    return (x.flatten(1) @ vh.T).flatten()


def kernel_check(a, width):
    # width=1 deletes other first channels and second-layer input columns.
    # All remaining parameter values are unchanged; nothing is refitted.
    t = torch.full((6 * width + 12,), a)
    t[-6:] = -a
    direction = torch.zeros_like(t)
    direction[:3] = MU
    fn = torch.func.jacrev(lambda p: forward_flat(p, width))
    j, jp = torch.func.jvp(fn, (t,), (direction,))
    ker, kp = j @ j.T, jp @ j.T + j @ jp.T
    z = RHO[:, None] @ RHO[None, :]
    vv, eye = torch.tensor([[1., -1.], [-1., 1.]]), torch.eye(2)
    if width == 4:
        ref = 864 * a**4 * torch.kron(z, vv + eye)
        refp = 216 * a**3 * MU * torch.kron(z, vv + 2 * eye)
    else:
        ref = a**4 * torch.kron(z, 216 * vv + 54 * eye)
        refp = a**3 * MU * torch.kron(z, 216 * vv + 108 * eye)
    slope = torch.trace(torch.linalg.solve(torch.eye(4) + ker, kp)).item()
    assert slope > 0
    assert torch.allclose(ker, ref, atol=2e-12, rtol=2e-12)
    assert torch.allclose(kp, refp, atol=2e-12, rtol=2e-12)
    return {"raw_parameters": t.numel(), "kernel_error": (ker-ref).abs().max().item(),
            "derivative_error": (kp-refp).abs().max().item(),
            "capacity_derivative_lambda_1": slope,
            "derivative_eigenvalues": torch.linalg.eigvalsh(kp).tolist()}


def main():
    a = a0 = .2
    net = CNN(a)
    opt = torch.optim.Adam(net.parameters(), lr=1., betas=(B1, B2), eps=EPS)
    rng = torch.Generator().manual_seed(1009)
    signs = torch.ones(36)
    signs[-6:] = -1
    m = v = 0.
    errors = {"raw_recurrence": 0., "raw_gradient": 0., "moment": 0., "logits": 0.}
    max_q = lr_sum = 0.
    rows = []
    kernels0 = {"full": kernel_check(a, 4), "literal_self": kernel_check(a, 1)}
    for task in range(TASKS):
        labels = torch.randint(2, (2,), generator=rng)
        ys = 1. - 2. * labels.to(torch.float64)
        start = a
        delta = D0 / (task + 1)**POWER
        for inner in range(H):
            step = task * H + inner + 1
            lr = delta * a / (K * (1. + a))
            opt.param_groups[0]["lr"] = lr
            opt.zero_grad()
            logits = net(IMAGES)
            expected = (72 * a**3 * RHO)[:, None] * torch.tensor([1., -1.])[None, :]
            errors["logits"] = max(errors["logits"], (logits-expected).abs().max().item())
            F.cross_entropy(logits, labels).backward()
            # Each RAW coordinate gradient; differentiating a tied network
            # with respect to a would instead introduce a factor of 36.
            g = (6 * a**2 * (RHO * (torch.tanh(72 * a**3 * RHO)-ys)).mean()).item()
            grads = torch.cat([p.grad.flatten() for p in net.parameters()])
            errors["raw_gradient"] = max(errors["raw_gradient"], (grads-signs*g).abs().max().item())
            m, v = B1*m+(1-B1)*g, B2*v+(1-B2)*g*g
            q = (m/(1-B1**step))/(math.sqrt(v/(1-B2**step))+EPS)
            nxt = a-lr*q
            assert abs(q) <= K and nxt > 0 and nxt >= a*(1-delta)
            opt.step()
            raw = torch.cat([p.detach().flatten() for p in net.parameters()])
            rm = torch.cat([opt.state[p]["exp_avg"].flatten() for p in net.parameters()])
            rv = torch.cat([opt.state[p]["exp_avg_sq"].flatten() for p in net.parameters()])
            errors["raw_recurrence"] = max(errors["raw_recurrence"], (raw-signs*nxt).abs().max().item())
            errors["moment"] = max(errors["moment"], (rm-signs*m).abs().max().item(), (rv-v).abs().max().item())
            max_q, lr_sum, a = max(max_q, abs(q)), lr_sum+lr, nxt
        rows.append({"task": task, "labels": labels.tolist(), "a_start": start, "a_end": a})
    assert max(errors.values()) < 1e-12
    result = {"scope": "Finite identity checks only, not a probability or infinite-limit estimate.",
              "architecture": "RGB3->Conv1x1(4)->ReLU->Pool2->Conv1x1(3)->ReLU->Pool2->free head6->2",
              "all_raw_parameters_train": 36, "biases": False,
              "image_shape": list(IMAGES.shape), "pooled_spatial_cells": 2, "image_feature_rank": 1,
              "all_conv_channels_overlap": True, "independent_labels_per_task": True,
              "task_count": TASKS, "repeated_full_batch_updates": H, "batch_size": 2,
              "beta1": B1, "beta2": B2, "epsilon": EPS, "moment_reset": False,
              "delta0": D0, "decay_power": POWER, "adam_bound": K,
              "max_errors": errors, "a_initial": a0, "a_final_finite_run": a,
              "mean_patch": MU, "first_conv_mean_initial": 3*a0*MU,
              "first_conv_mean_final_finite_run": 3*a*MU,
              "largest_normalized_update": max_q, "finite_sum_learning_rates": lr_sum,
              "kernel_initial": kernels0,
              "kernel_final": {"full": kernel_check(a, 4), "literal_self": kernel_check(a, 1)},
              "analytic_probability_certificate": rational_probability_certificate(),
              "task_rows": rows}
    out = Path(__file__).resolve().parents[2]/"results/cnn_drive_1009/adam_iid_longtime.json"
    out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "task_rows"}, indent=2))


if __name__ == "__main__":
    main()

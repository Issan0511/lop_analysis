"""An explicitly defined channel-diagonal CE self term for the native LR CNN.

This is a local-affine operational decomposition, not a unique causal attribution.
For each image, let A_j be the logits' Jacobian with respect to c1 activation
channel j, v_j=A_j a_j, and Q=integral_0^1 Hessian(logsumexp)(t*f) dt.
Then self_j=A_j.T Q v_j, cross_bias_j=A_j.T Q(f-v_j), and
label_j=A_j.T (uniform-onehot(y)).  The identity Q*f=softmax(f)-uniform
makes their sum the actual CE cotangent.  All A, Q, and cotangents are detached:
we do not differentiate the attribution itself or change pool winners.

The original training forward and cross_entropy are used for returned `grads`.
Nothing here mutates parameters, optimizer state, RNG, or activation statistics.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from src import sna_cnn_cause_1009 as E


@lru_cache(maxsize=8)
def _legendre(n: int):
    x, w = np.polynomial.legendre.leggauss(n)
    return (x + 1) * 0.5, w * 0.5


@torch.no_grad()
def ce_metric(logits: torch.Tensor, quadrature: int = 64,
              tolerance: float = 1e-7,
              matrix_tolerance: float = 1e-9) -> tuple[torch.Tensor, dict]:
    """Float64 radial CE metric with adaptive Gauss--Legendre quadrature.

    Logit centering improves conditioning without changing either Q or Q*f.
    Failure of the numerical identity is reported, never folded into self.
    """
    if quadrature < 1 or quadrature > 512:
        raise ValueError("quadrature must be between 1 and 512")
    f = logits.detach().double()
    f = f - f.mean(-1, keepdim=True)
    target = f.softmax(-1) - 1 / f.shape[-1]
    attempts = []
    # Q*f alone only checks one direction.  The self term acts on v_j,
    # so always compare the full matrix at successive quadrature orders.
    n = min(max(64, int(quadrature)), 256)
    previous_q = None
    q_change = float("nan")
    while True:
        xn, wn = _legendre(n)
        t = torch.as_tensor(xn, device=f.device, dtype=torch.float64)
        w = torch.as_tensor(wn, device=f.device, dtype=torch.float64)
        probs = (t.view(-1, *([1] * f.ndim)) * f.unsqueeze(0)).softmax(-1)
        mean_p = torch.einsum("q,q...c->...c", w, probs)
        outer = torch.einsum("q,q...c,q...d->...cd", w, probs, probs)
        q = torch.diag_embed(mean_p) - outer
        error = torch.einsum("...cd,...d->...c", q, f) - target
        max_error = float(error.abs().max())
        attempts.append({"nodes": n, "qf_error_absmax": max_error})
        if previous_q is not None:
            q_change = float((q - previous_q).abs().max())
        converged = max_error <= tolerance and q_change <= matrix_tolerance
        if converged or n == 512:
            break
        previous_q = q
        n = min(2 * n, 512)
    return q, {
        "quadrature_nodes": n,
        "quadrature_converged": converged,
        "qf_error_absmax": max_error,
        "qf_error_per_run": error.abs().flatten(1).amax(1).detach(),
        "q_change_absmax": q_change,
        "quadrature_attempts": attempts,
        "q_row_sum_absmax": float(q.sum(-1).abs().max()),
        "logit_absmax": float(logits.detach().abs().max()),
    }


def decompose_batch(B, xb: torch.Tensor, yb: torch.Tensor,
                    quadrature: int = 64) -> dict:
    """Return native grads and c1 gradient components for a bundled minibatch.

    xb: (runs,batch,3,32,32), yb: (runs,batch).  Loss averages over batch
    and sums over runs, matching E.Bundle.step exactly.  Component tensors
    have the same shapes as B.P[0] and B.P[1].  `numerical_residual` is an
    explicit closure term including quadrature and float32 roundoff.
    Ten class backprops materialize a float32 Jacobian (~100 MiB for 10
    runs of 16 images); three additional c1 backprops produce components.
    All returned tensors are detached.  Training graph is released on exit.
    """
    if xb.ndim != 5 or yb.shape != xb.shape[:2]:
        raise ValueError("expected xb (R,B,3,32,32), yb (R,B)")
    runs, batch = xb.shape[:2]
    out = E.forward(B.P, xb, B.act)
    a1, logits = out[1], out[8]
    classes = logits.shape[-1]
    channels = a1.shape[1] // runs
    loss_per_run = F.cross_entropy(
        logits.reshape(-1, classes), yb.reshape(-1), reduction="none"
    ).view(runs, batch).mean(1)
    grads = [g.detach() for g in torch.autograd.grad(
        loss_per_run.sum(), B.P, retain_graph=True)]
    q, diagnostics = ce_metric(logits, quadrature)
    # Samples and runs do not interact; summing each class's logits gives all
    # per-image Jacobians in one reverse pass, preserving native pool ties.
    jac = []
    contributions = []
    a = a1.detach().view(batch, runs, channels, *a1.shape[-2:])
    for c in range(classes):
        gc = torch.autograd.grad(logits[..., c].sum(), a1, retain_graph=True)[0].detach()
        gc = gc.view_as(a)
        jac.append(gc)
        contributions.append((gc.double() * a.double()).sum((-2, -1)))
    # J: (class,batch,run,channel,h,w), v: (run,batch,channel,class).
    jac = torch.stack(jac)
    v = torch.stack(contributions, dim=-1).permute(1, 0, 2, 3)
    with torch.no_grad():
        f = logits.detach().double()
        sv = torch.einsum("rbcq,rbjq->rbjc", q, v)
        cv = torch.einsum("rbcq,rbjq->rbjc", q, f[:, :, None, :] - v)
        lv = torch.full_like(f, 1 / classes)
        lv.scatter_add_(-1, yb[..., None], -torch.ones_like(yb[..., None], dtype=f.dtype))
        # Each channel gets its own CE metric vector for self/cross.  Label
        # vector is shared.  Cast at the native float32 Jacobian boundary.
        cotangents = {
            "self": torch.einsum("cbrjhw,rbjc->brjhw", jac, sv.to(jac.dtype)),
            "cross_bias": torch.einsum("cbrjhw,rbjc->brjhw", jac, cv.to(jac.dtype)),
            "label": torch.einsum("cbrjhw,rbc->brjhw", jac, lv.to(jac.dtype)),
        }
    components = {}
    for index, (name, cotangent) in enumerate(cotangents.items()):
        components[name] = [g.detach() for g in torch.autograd.grad(
            a1, B.P[:2], grad_outputs=cotangent.reshape_as(a1) / batch,
            retain_graph=index < len(cotangents) - 1)]
    components["numerical_residual"] = [
        grads[i] - sum(components[name][i] for name in ("self", "cross_bias", "label"))
        for i in range(2)
    ]
    # Preserve raw errors both absolutely and relative to the actual c1 grad;
    # when the latter cancels, relative values can be large and are diagnostic.
    residual = components["numerical_residual"]
    # Independent homogeneity check: leaky ReLU implies
    # <(W_j,b_j),grad_self_j> = mean_batch(v_j.T Q v_j) >= 0.
    radial_param = ((B.P[0].detach().double() * components["self"][0].double())
                    .flatten(2).sum(-1)
                    + B.P[1].detach().double() * components["self"][1].double())
    radial_logit = (v * sv).sum(-1).mean(1)
    radial_error = radial_param - radial_logit
    diagnostics.update({
        "loss_per_run": loss_per_run.detach(),
        "c1_residual_absmax": max(float(g.abs().max()) for g in residual),
        "c1_actual_grad_absmax": max(float(g.abs().max()) for g in grads[:2]),
        "component_absmax": {name: max(float(g.abs().max()) for g in values)
                             for name, values in components.items()},
        "c1_residual_norm_per_run": torch.sqrt(sum(
            g.double().flatten(1).square().sum(1) for g in residual)),
        "c1_actual_grad_norm_per_run": torch.sqrt(sum(
            g.double().flatten(1).square().sum(1) for g in grads[:2])),
        "jacobian_class_sum_absmax": float(jac.sum(0).abs().max()),
        "v_channel_sum_absmax": float(v.sum(2).abs().max()),
        "affine_remainder_absmax": float((f - v.sum(2)).abs().max()),
        "radial_param": radial_param.detach(),
        "radial_logit": radial_logit.detach(),
        "radial_error_absmax": float(radial_error.abs().max()),
        "radial_error_relative_absmax": float(radial_error.abs().max()
                                              / radial_logit.abs().max().clamp_min(1e-30)),
        "radial_param_negative_count": int((radial_param < -1e-10).sum()),
        "radial_logit_negative_count": int((radial_logit < -1e-10).sum()),
    })
    diagnostics["c1_residual_relative_absmax"] = (
        diagnostics["c1_residual_absmax"] / max(diagnostics["c1_actual_grad_absmax"], 1e-30))
    return {"grads": grads, "components": components, "diagnostics": diagnostics,
            "zs": [out[i].detach() for i in (0, 2, 4, 6)],
            "lossv": loss_per_run.detach(), "logits": logits.detach()}


def _self_check(checkpoint, cifar_dir, output):
    """Bounded reproducible checks on one saved LR run; never updates it."""
    import json
    import time
    from pathlib import Path
    from types import SimpleNamespace
    from src import cnn_drive_verify_1009_lr as LR
    from src import rlcifar_cnn_0908 as CN
    from src import pmnist_0905 as H
    from src import pmnist_rlcifar_0907 as RC

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RC.DATA_DIR = Path(cifar_dir)
    st = torch.load(checkpoint, map_location="cpu", weights_only=False)
    params = [p[None].to(device).requires_grad_(True) for p in st["P"]]
    bundle = SimpleNamespace(P=params, act=LR.BundleLeaky(1, 0.1, device))
    cifar = RC.Cifar10()
    x = CN.images(cifar, CN.subset_idx(st["seed"]), device)
    # Labels for the first task after the saved checkpoint, with a fixed batch.
    g = H.stream("rlcc_labels", st["seed"])
    for _ in range(round(float(st["tc"]) / 30000) + 1):
        labels = CN.task_labels(g)
    xb, yb = x[None, :16], labels[None, :16].to(device)
    state_before = [p.detach().clone() for p in params]
    t0 = time.perf_counter()
    result = decompose_batch(bundle, xb, yb)
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    native = E.forward(params, xb, bundle.act)
    native_loss = F.cross_entropy(native[8].reshape(-1, 10), yb.reshape(-1),
                                  reduction="none").view(1, -1).mean(1).sum()
    native_grads = torch.autograd.grad(native_loss, params)
    native_error = max(float((a - b).abs().max()) for a, b in zip(native_grads, result["grads"]))
    # A central finite-difference check in double precision avoids conflating
    # float32 cancellation with an algebraic scaling/layout error.
    dp = [p.detach().double().requires_grad_(True) for p in params]
    dx = xb[:, :2].double()
    dy = yb[:, :2]
    do = E.forward(dp, dx, bundle.act)
    dl = F.cross_entropy(do[8].reshape(-1, 10), dy.reshape(-1))
    dg = torch.autograd.grad(dl, dp[:2])
    rg = torch.Generator(device="cpu").manual_seed(1011)
    direction = [torch.randn(p.shape, generator=rg, dtype=torch.float64).to(device) for p in dp[:2]]
    norm = torch.sqrt(sum(t.square().sum() for t in direction))
    direction = [t / norm for t in direction]
    exact = float(sum((g * d).sum() for g, d in zip(dg, direction)))
    eps = 1e-6
    losses = []
    with torch.no_grad():
        for sign in (-1, 1):
            shifted = [p + sign * eps * direction[i] if i < 2 else p
                       for i, p in enumerate(dp)]
            ff = E.forward(shifted, dx, bundle.act)[8]
            losses.append(float(F.cross_entropy(ff.reshape(-1, 10), dy.reshape(-1))))
    finite = (losses[1] - losses[0]) / (2 * eps)
    fd_rel = abs(finite - exact) / max(abs(exact), 1e-12)
    double_bundle = SimpleNamespace(P=dp, act=bundle.act)
    double_result = decompose_batch(double_bundle, dx, dy)
    q, qdiag = ce_metric(result["logits"], quadrature=64)
    q128, q128diag = ce_metric(result["logits"], quadrature=128)
    check = {
        "checkpoint": str(checkpoint), "seed": st["seed"], "device": str(device),
        "torch_version": torch.__version__, "batch_size": 16,
        "wall_seconds": wall,
        "native_gradient_absmax_difference": native_error,
        "parameters_unchanged": all(torch.equal(a, b) for a, b in zip(params, state_before)),
        "q_eigenvalue_min": float(torch.linalg.eigvalsh(q).min()),
        "q_extra_refinement": {"first_nodes": qdiag["quadrature_nodes"],
                               "second_nodes": q128diag["quadrature_nodes"],
                               "absmax_difference": float((q - q128).abs().max())},
        "finite_difference": {"epsilon": eps, "analytic": exact, "finite": finite,
                              "relative_error": fd_rel},
        "diagnostics": result["diagnostics"],
        "float64_two_image_algebra_check": double_result["diagnostics"],
        "note": "Float32 closure is recorded separately; algebra is checked in float64. "
                "Deterministic CUDA convolution weight backward amplifies cotangent roundoff.",
    }
    check["pass"] = (native_error == 0 and check["parameters_unchanged"]
                     and result["diagnostics"]["quadrature_converged"]
                     and double_result["diagnostics"]["c1_residual_relative_absmax"] < 1e-10
                     and double_result["diagnostics"]["radial_error_relative_absmax"] < 1e-10
                     and double_result["diagnostics"]["radial_param_negative_count"] == 0
                     and check["q_eigenvalue_min"] > -1e-12 and fd_rel < 1e-4)
    def convert(value):
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    Path(output).write_text(json.dumps(convert(check), indent=2) + "\n")
    print(json.dumps(convert(check), indent=2))
    if not check["pass"]:
        raise SystemExit("CE self kernel check failed")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cifar-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    _self_check(args.checkpoint, args.cifar_dir, args.output)

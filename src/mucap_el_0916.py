#!/usr/bin/env python3
"""mu-direction component caps for the RL box (H2 design note section 10.4).

The row of a hidden unit is split along the fixed mean image of that seed's training set:

    e = mu / ||mu||,   q_i = w_i . e,   v_i = w_i - q_i e   (v_i . e = 0),

so that, on that same fixed set,  zbar_i = w_i . mu + b_i = q_i ||mu|| + b_i  exactly -- the parallel
component alone sets the mean preactivation, the orthogonal component alone sets the spread around it.

Three caps, each applied right after an Adam update, each a cap and not an equality:
    cap_parallel_  |q_i| <= q_cap_i          v untouched
    cap_perp_      ||v_i|| <= v_cap_i        q untouched, direction of v kept
    cap_both_      both, parallel first (the two orders agree; S3 checks it)

Rows at or below their cap keep every bit.  Bias, the other tensors and the Adam moments are never
touched here.  This is NOT the row-mean / centered-weight split of wcap_rlmnist_0914: e is the unit
vector along the mean image, not the uniform direction 1/sqrt(d).  Both are logged by components().
"""

from __future__ import annotations

import torch

__all__ = ["mu_basis", "split_mu", "cap_parallel_", "cap_perp_", "cap_both_",
           "parallel_cap", "perp_cap", "components"]


@torch.no_grad()
def mu_basis(x: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, float]:
    """(e, e64, ||mu||) for the evaluation set x (n, d), computed once in float64.

    e is in x's dtype and is what the caps project with inside the training loop; e64 is the same
    direction in float64 and is what the ledger and the identity zbar = q||mu|| + b use.  Casting e to
    float32 tilts it by ~eps32, so the identity only holds to that precision along e -- keep both and
    do not recompute q in float64 from the float32 e.  Returns (None, None, 0.0) when mu = 0, which the
    caller records as 'not applicable'."""
    mu = x.double().mean(0)
    nrm = float(torch.linalg.vector_norm(mu))
    if nrm == 0.0:
        return None, None, 0.0
    e64 = mu / nrm
    return e64.to(x.dtype), e64, nrm


@torch.no_grad()
def split_mu(W: torch.Tensor, e: torch.Tensor):
    """(q (rows, 1), v (rows, d)) in W's dtype.  The cap radii and both projections go through here."""
    q = (W @ e).unsqueeze(1)
    v = W - q * e
    return q, v


@torch.no_grad()
def parallel_cap(W: torch.Tensor, e: torch.Tensor, mult: float = 1.0) -> torch.Tensor:
    """|q_i| at this state, times mult: the radius cap_parallel_ is later called with."""
    return (mult * split_mu(W, e)[0].abs()).detach().clone()


@torch.no_grad()
def perp_cap(W: torch.Tensor, e: torch.Tensor, mult: float = 1.0) -> torch.Tensor:
    """||v_i|| at this state, times mult."""
    v = split_mu(W, e)[1]
    return (mult * torch.linalg.vector_norm(v, dim=1, keepdim=True)).detach().clone()


@torch.no_grad()
def cap_parallel_(W: torch.Tensor, e: torch.Tensor, q_cap: torch.Tensor):
    """In place: rows with |q| > q_cap move along e until |q| = q_cap, keeping the sign q has now;
    a sign change inside the cap is left alone.  Rows at or below the cap keep every bit.
    Returns (rows written, sum of |q| - q_cap over them)."""
    q, _ = split_mu(W, e)
    over = q.abs() > q_cap
    q_new = torch.clamp(q, -q_cap, q_cap)
    W.copy_(torch.where(over, W + (q_new - q) * e, W))
    return over.sum(), torch.where(over, q.abs() - q_cap, torch.zeros_like(q)).sum()


@torch.no_grad()
def cap_perp_(W: torch.Tensor, e: torch.Tensor, v_cap: torch.Tensor):
    """In place: rows with ||v|| > v_cap keep q and the direction of v and get ||v|| = v_cap.
    Zero-norm rows are never over a non-negative cap, so they are left alone.
    Returns (rows written, sum of ||v|| - v_cap over them)."""
    q, v = split_mu(W, e)
    n = torch.linalg.vector_norm(v, dim=1, keepdim=True)
    over = n > v_cap
    W.copy_(torch.where(over, q * e + v * (v_cap / n), W))
    return over.sum(), torch.where(over, n - v_cap, torch.zeros_like(n)).sum()


@torch.no_grad()
def cap_both_(W: torch.Tensor, e: torch.Tensor, q_cap: torch.Tensor, v_cap: torch.Tensor):
    """cap_parallel_ then cap_perp_.  The parallel step does not change ||v|| and the perpendicular
    step does not change q, so the other order gives the same row up to rounding (S3)."""
    a = cap_parallel_(W, e, q_cap)
    b = cap_perp_(W, e, v_cap)
    return a, b


@torch.no_grad()
def components(W: torch.Tensor, e64: torch.Tensor) -> dict:
    """Both coordinate systems of one layer, float64, for the ledger: the mu split (q, ||v||) and the
    row-mean split (m, ||W~||).  They are different bases and are reported side by side.  Pass the
    float64 e64 from mu_basis, not the cast-down e the caps use."""
    W64 = W.detach().double()
    e64 = e64.double()
    q = W64 @ e64
    v = W64 - q.unsqueeze(1) * e64
    m = W64.mean(dim=1)
    Wt = W64 - m.unsqueeze(1)
    return {"q": q.numpy(), "v_norm": torch.linalg.vector_norm(v, dim=1).numpy(),
            "row_mean": m.numpy(), "wt_norm": torch.linalg.vector_norm(Wt, dim=1).numpy(),
            "row_norm": torch.linalg.vector_norm(W64, dim=1).numpy()}

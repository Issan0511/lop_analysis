#!/usr/bin/env python3
"""Row-norm cap for the second hidden layer (l2cap_ee_0917; H2 design note section 10.4's old EE plan).

    cap_row_norm_(W, r)   ||w_i|| <= r_i, applied right after an Adam update

Rows at or below their radius keep every bit; a row above it is scaled onto the sphere, direction kept
(up to float32 rounding).  A cap, not an equality: small rows are never inflated.  Bias, the other
tensors and the Adam moments are not touched here.  The radius is the row's own norm at the end of
task 1 (row_norms), so a zero-radius row can only be a row that was zero then.
"""
from __future__ import annotations

import torch

__all__ = ["row_norms", "cap_row_norm_"]


@torch.no_grad()
def row_norms(W: torch.Tensor) -> torch.Tensor:
    """(rows, 1) norms of W's rows at this state, in W's dtype: the radii cap_row_norm_ is called with."""
    return torch.linalg.vector_norm(W, dim=1, keepdim=True).detach().clone()


@torch.no_grad()
def cap_row_norm_(W: torch.Tensor, r: torch.Tensor) -> tuple[int, float]:
    """In place: rows with ||w_i|| > r_i become w_i * r_i / ||w_i||.  Returns (rows written, sum of the
    norm removed).  A zero row is never 'over' (its norm is 0 <= r), so the division never reaches it."""
    n = torch.linalg.vector_norm(W, dim=1, keepdim=True)
    over = n > r
    safe = torch.where(over, n, torch.ones_like(n))
    W.copy_(torch.where(over, W * (r / safe), W))
    return int(over.sum()), float((n - r).clamp(min=0).sum())

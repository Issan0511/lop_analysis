"""drive_rlmnist_1007 numerics: A2's measure on the Random-Label MNIST ELU->ELU host.

Imported unchanged from A2 (analysis/drive_cifar_c_0920/numerics.py, the code of record): measure, pack,
KEYS, gamma, dot, augmented and the unit roundoffs.  Ported here (spec section 1.2):

  full_features  the host's forward2 on the fixed 1200 images (A2: baddbmm + clamp on a CUDA R-stack);
                 the native-mean error bound is A2's, with the image mean taken before the |W2| product
                 (sum_j mean_s|a1_sj| |W2_ij| is the same number as mean_s sum_j |a1_sj||W2_ij|).
  decompose      A2's d/e decomposition with the training derivative phi'(z) of the host ELU (autograd of
                 the host phi on the same float32 z: 1 for z > 0, fl(expm1(z) + 1) for z <= 0, exactly 0
                 for z < ln 2^-24) instead of the ReLU gate (z >= 0).  phi' >= 0, so A2's bound
                 |K phi' e| <= |K| phi' eps L is unchanged.
  layer1_S       first-layer self movement S1 = dW1 . xbar + db1 (REPORT_ONLY; U1 = 0 because the inputs
                 are fixed).
  bracket        REPORT_ONLY implementation check (not registered): the actual S lies in A2's interval
                 [-lr/c1 Qplus - move_error - err(S), -lr/c1 Qminus + move_error + err(S)].

All tensors carry A2's leading replica axis R = 1.  Nothing here writes to its arguments.
"""
import torch

from analysis.drive_cifar_c_0920.numerics import (  # noqa: F401  (re-exported, code of record)
    U32, U64, TINY32, TINY64, gamma, augmented, dot, measure, pack,
    KEYS, COUNT_KEYS, TRANSPORT_KEYS, DIAG_KEYS)

L1_KEYS = ("S1_sum", "S1_positive", "S1_negative", "S1_positive_count", "S1_negative_count")
AUX_KEYS = ("bracket_violation",)


def train_gate(act, z):
    """The derivative the optimizer uses: autograd of the host's phi at the same float32 z."""
    with torch.enable_grad():
        zz = z.detach().clone().requires_grad_(True)
        g, = torch.autograd.grad(act.phi(zz).sum(), zz)
    return g.detach()


def full_features(params, x, act1, act2, forward2):
    """(mu, mu_error, native mean z2, native error), each (1, 100), on the fixed images x."""
    with torch.no_grad():
        _, a1, z2, _, _ = forward2(params, x, act1, act2)
    hd = a1.double()
    n, width = hd.shape
    mu = hd.mean(0)
    mu_error = gamma(n + 2) * hd.abs().mean(0) + TINY64
    native_error = gamma(width + 3, U32) * (hd.abs().mean(0) @ params[2].detach().double().abs().T
                                             + params[3].detach().double().abs())
    native_error = native_error + gamma(n + 2) * z2.double().abs().mean(0) + TINY32 * (width + 3)
    return mu[None], mu_error[None], z2.double().mean(0)[None], native_error[None]


def decompose(h, z, logits, J, old, new, gate):
    """A2's decompose for one replica.  h, z, gate: (B, I) float32 native; logits (B, C); J = W3 (C, I)
    before the update; old/new: (B,) labels.  Returns A2's dict with a leading R = 1 axis."""
    h = h.detach().double()[None]
    logits = logits.detach().double()[None]
    J = J.detach().double()[None]
    gate = gate.detach().double()[None]
    old, new = old[None], new[None]
    R, B, _ = h.shape
    ar = torch.arange(R, device=h.device)[:, None]
    jo, jn = J[ar, old], J[ar, new]
    p = torch.softmax(logits, -1)
    d = jo - jn
    e = torch.bmm(p, J) - jo
    eps = 1 - p.gather(-1, old[..., None]).squeeze(-1)
    L = (J[:, None, :, :] - jo[:, :, None, :]).abs().amax(2)
    ht = torch.cat((h, torch.ones(R, B, 1, device=h.device, dtype=h.dtype)), -1)
    g = torch.bmm(((d + e) * gate).transpose(1, 2), ht) / B
    uconf = torch.bmm(p - 1 / J.shape[1], J)
    gc = torch.bmm((uconf * gate).transpose(1, 2), ht) / B
    absu = torch.bmm(p, J.abs()) + jn.abs()
    gb = gamma(128, U32) * torch.bmm((absu * gate).transpose(1, 2), ht.abs()) / B + 128 * TINY32
    eb = gamma(64) * (torch.bmm(p, J.abs()) + jo.abs() + eps[..., None] * L + L) + TINY64
    return dict(ht=ht, gate=gate, d=d, e=e, eps=eps, L=L, g=g, gconf=gc, gb=gb, eb=eb)


def layer1_S(W1_before, b1_before, W1_after, b1_after, xbar):
    """(100,) float64 first-layer self movement on the fixed images' mean xbar (float64)."""
    return (W1_after.double() - W1_before) @ xbar + (b1_after.double() - b1_before)


def pack_l1(S1):
    return torch.stack((S1, S1.clamp(min=0), S1.clamp(max=0), (S1 > 0).double(), (S1 < 0).double()), -1)


def bracket(d, inv1):
    """(R, I) 1.0 where the actual S leaves A2's theory interval (REPORT_ONLY implementation check)."""
    i1 = inv1.double()
    hi = -.001 * i1 * d["Qminus"] + d["move_error"] + d["S_error"]
    lo = -.001 * i1 * d["Qplus"] - d["move_error"] - d["S_error"]
    return ((d["S"] > hi) | (d["S"] < lo)).double()

"""POSTHOC (registered=0, analysis_grade=posthoc_not_preregistered), written 2026-09-16 before spec_l2_wall_0916
was registered; its outputs are listed in the spec §0. Inputs are read from absolute paths on white-san:
checkpoints copied to ~/Projects/obsidian-research-data/elu_three_studies_0913/ and, for h3_mucap_ref_l2.py,
the ref arm of the running mucap_el_0916 (cap arms not read). Run from the repo root of proj_004_drift or
this worktree; outputs are in results/l2_wall_0916/posthoc/."""
"""Exact second-layer wall quantities from elu_reserve_0913 checkpoints (PM box A, 625 upd/task,
ELU1 / LR, t20 / t100, seeds 0-2). Posthoc, unregistered. Evaluation set: MNIST test permuted by the
checkpoint's last task permutation (same as the probe convention)."""
import sys, numpy as np, torch
sys.path.insert(0, ".")
from src import width_sink_clamp_0909 as C
from src import transport_common_0910 as T
H = C.H
T.data_dir(); mnist = H.Mnist(torch.device("cpu"))
xt = mnist.test_x.double()
root = "/home/issan/Projects/obsidian-research-data/elu_three_studies_0913/files/results/elu_reserve_0913"
def act_fn(z, act):
    if act == "ELU1": return torch.where(z > 0, z, torch.expm1(z.clamp_max(0)))
    return torch.where(z > 0, z, 0.1 * z)
def med(a): return float(np.nanmedian(a))
print("act seed t | L1: d1 rho1S c1S | L2: med d2  zb2  sig2  p+2  U2<0 | mu-dir: |mu2| c2 kappa2 min(e2'a1) c2>k2 | S-dir: S2mean sd c2S kappa2S | med rho_mu  med rho_S  cov-share | corr(d2,-c2*sqrt(rho_mu)+b/sig)")
for act in ("ELU1", "LR"):
    for seed in (0, 1, 2):
        for t in (20, 100):
            ck = torch.load(f"{root}/{act}_s{seed}_t{t}_checkpoint.pt", map_location="cpu", weights_only=False)
            snap, perm = ck["snapshot"], ck["perm"]
            p = snap["p"]
            p = [torch.as_tensor(q).detach().double() for q in p]
            if t == 20 and seed == 0: print("param shapes:", [tuple(q.shape) for q in p])
            x = xt[:, perm]
            z1 = x @ p[0].T + p[1]; a1 = act_fn(z1, act)
            z2 = a1 @ p[2].T + p[3]
            # layer 1 in the all-ones direction (stage-0 quantity)
            S1 = x.sum(1); m1 = p[0].mean(1); sig1 = z1.std(0, unbiased=False); zb1 = z1.mean(0)
            rho1S = (S1.var(unbiased=False) * m1 ** 2 / sig1 ** 2).numpy(); d1 = (zb1 / sig1).numpy()
            c1S = float(S1.mean() / S1.std(unbiased=False))
            # layer 2 support geometry
            zb2 = z2.mean(0); sig2 = z2.std(0, unbiased=False); U2 = z2.amax(0); pp2 = (z2 > 0).double().mean(0)
            d2 = (zb2 / sig2).numpy()
            mu2 = a1.mean(0); mu2n = float(mu2.norm()); e2 = mu2 / mu2n
            proj = a1 @ e2; sdp = float(proj.std(unbiased=False))
            c2 = mu2n / sdp; k2 = float((proj.mean() - proj.min()) / sdp); pmin = float(proj.min())
            S2 = a1.sum(1); c2S = float(S2.mean() / S2.std(unbiased=False)); k2S = float((S2.mean() - S2.min()) / S2.std(unbiased=False))
            q2 = p[2] @ e2; v2 = p[2] - q2[:, None] * e2[None, :]
            va = a1 @ v2.T                      # pattern-channel preactivation per unit
            var_q = (q2 ** 2) * proj.var(unbiased=False)
            var_v = va.var(0, unbiased=False)
            cov = 2 * q2 * ((proj[:, None] - proj.mean()) * (va - va.mean(0))).mean(0)
            tot = var_q + var_v + cov
            rho_mu = (var_q / tot).numpy(); covshare = (cov / tot).numpy()
            m2 = p[2].mean(1); rho_S = (S2.var(unbiased=False) * m2 ** 2 / sig2 ** 2).numpy()
            pred = (-c2 * np.sqrt(np.clip(rho_mu, 0, None)) * np.sign(q2.numpy()) * -1) + (p[3] / sig2).numpy()
            # d2 = (q2 |mu2| + b2)/sig2 exactly; with rho_mu = q2^2 Var(proj)/sig2^2 -> q2|mu2|/sig2 = sign(q2) c2 sqrt(rho_mu)
            exact = (np.sign(q2.numpy()) * c2 * np.sqrt(np.clip(rho_mu, 0, None)) + (p[3] / sig2).numpy())
            err = np.abs(exact - d2).max()
            print(f"{act:4s} {seed} {t:3d} | {med(d1):+.2f} {med(rho1S):.3f} {c1S:.2f} | {med(d2):+.2f} {med(zb2.numpy()):+6.2f} {med(sig2.numpy()):5.2f} {med(pp2.numpy()):.3f} {float((U2 < 0).double().mean()):.2f} | {mu2n:6.2f} {c2:5.2f} {k2:5.2f} {pmin:+6.2f} {c2 > k2} | {float(S2.mean()):+7.2f} {float(S2.std(unbiased=False)):6.2f} {c2S:+5.2f} {k2S:5.2f} | {med(rho_mu):.3f} {med(rho_S):.3f} {med(covshare):+.3f} | identity err {err:.1e}; frac q2<0 {float((q2 < 0).double().mean()):.2f}")

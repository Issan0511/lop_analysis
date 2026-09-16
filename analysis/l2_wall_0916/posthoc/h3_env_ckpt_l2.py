"""POSTHOC (registered=0, analysis_grade=posthoc_not_preregistered), written 2026-09-16 before spec_l2_wall_0916
was registered; its outputs are listed in the spec §0. Inputs are read from absolute paths on white-san:
checkpoints copied to ~/Projects/obsidian-research-data/elu_three_studies_0913/ and, for h3_mucap_ref_l2.py,
the ref arm of the running mucap_el_0916 (cap arms not read). Run from the repo root of proj_004_drift or
this worktree; outputs are in results/l2_wall_0916/posthoc/."""
"""Second-layer wall quantities at task 50 from elu_environment_0913/checkpoint.pt (RL models: fixed
1200 train images per seed, no permutation). Posthoc, unregistered."""
import sys, numpy as np, torch
sys.path.insert(0, "/home/issan/Projects/claude/proj_004_drift")
from src import elu_environment_0913 as E
ck = torch.load("/home/issan/Projects/obsidian-research-data/elu_three_studies_0913/files/results/elu_environment_0913/checkpoint.pt", map_location="cpu", weights_only=False)
print("adam step t =", float(ck["t"]))
ax = E.read_idx(E.DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255
subset = {s: torch.randperm(len(ax), generator=E.stream("rl_subset", s))[:E.N].numpy() for s in range(3)}
def act_fn(z, act):
    if act == "ELU1": return torch.where(z > 0, z, torch.expm1(z.clamp_max(0)))
    return torch.where(z > 0, z, 0.1 * z)
def gate_fn(z, act):
    if act == "ELU1": return torch.where(z > 0, torch.ones_like(z), z.clamp_max(0).exp())
    return torch.where(z > 0, torch.ones_like(z), torch.full_like(z, 0.1))
def med(a): return float(np.nanmedian(a))
P = [q.double() for q in ck["parameters"]]
print("env act iv seed | L1 med d1  zb1  gate1  U1 | L2 med d2  zb2  sig2  gate2  p+2  U2<0 | |mu2|  c2  kappa2  min e2'a1  c2>k2 | med rho_mu  q2<0  med |b2|/sig2  | pred d2 from cone (-c2*sqrt(rho))  vs med d2")
for j, m in enumerate(ck["models"]):
    if m["env"] != "RL": continue
    s = m["seed"]; act = m["act"]
    x = torch.from_numpy(ax[subset[s]]).double()
    p = [q[j] for q in P]
    z1 = x @ p[0].T + p[1]; a1 = act_fn(z1, act); z2 = a1 @ p[2].T + p[3]
    g1, g2 = gate_fn(z1, act), gate_fn(z2, act)
    zb1, sg1, U1 = z1.mean(0), z1.std(0, unbiased=False), z1.amax(0)
    zb2, sg2, U2 = z2.mean(0), z2.std(0, unbiased=False), z2.amax(0)
    d1 = (zb1 / sg1).numpy(); d2 = (zb2 / sg2).numpy()
    mu2 = a1.mean(0); mu2n = float(mu2.norm()); e2 = mu2 / mu2n
    proj = a1 @ e2; sdp = float(proj.std(unbiased=False))
    c2 = mu2n / sdp; k2 = float((proj.mean() - proj.min()) / sdp); pmin = float(proj.min())
    q2 = p[2] @ e2
    rho = ((q2 ** 2) * proj.var(unbiased=False) / sg2 ** 2).numpy()
    pred = np.sign(q2.numpy()) * c2 * np.sqrt(rho)
    print(f"{m['env']} {act:4s} {m['iv']:6s} {s} | {med(d1):+.2f} {med(zb1.numpy()):+7.2f} {med(g1.mean(0).numpy()):.3f} {med(U1.numpy()):+6.1f} | {med(d2):+.2f} {med(zb2.numpy()):+8.2f} {med(sg2.numpy()):6.2f} {med(g2.mean(0).numpy()):.1e} {med((z2 > 0).double().mean(0).numpy()):.3f} {float((U2 < 0).double().mean()):.2f} | {mu2n:6.2f} {c2:5.2f} {k2:5.2f} {pmin:+7.2f} {c2 > k2} | {med(rho):.3f} {float((q2 < 0).double().mean()):.2f} {med(np.abs(p[3].numpy()) / sg2.numpy()):.3f} | {med(pred):+.2f} vs {med(d2):+.2f}")
    if m["iv"] == "ref" and act == "ELU1":
        # distribution across units of rho and d2, and the three-term sigma2 split
        v2 = p[2] - q2[:, None] * e2[None, :]; va = a1 @ v2.T
        var_q = (q2 ** 2) * proj.var(unbiased=False); var_v = va.var(0, unbiased=False)
        cov = 2 * q2 * ((proj[:, None] - proj.mean()) * (va - va.mean(0))).mean(0); tot = var_q + var_v + cov
        print(f"      rho quantiles 10/50/90: {np.percentile(rho,10):.2f} {np.percentile(rho,50):.2f} {np.percentile(rho,90):.2f} | d2 quantiles: {np.percentile(d2,10):+.2f} {np.percentile(d2,50):+.2f} {np.percentile(d2,90):+.2f} | sigma2 split med: q-term {med((var_q/tot).numpy()):.2f} v-term {med((var_v/tot).numpy()):.2f} cov {med((cov/tot).numpy()):+.2f} | a1: frac coords with mean<-0.9: {float((mu2 < -0.9).double().mean()):.2f}, max a1 {float(a1.max()):.1f}, S2 mean {float(a1.sum(1).mean()):+.1f} sd {float(a1.sum(1).std(unbiased=False)):.1f}")

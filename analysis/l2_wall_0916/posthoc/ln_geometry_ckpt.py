"""POSTHOC (registered=0, analysis_grade=posthoc_not_preregistered), written 2026-09-16 before spec_l2_wall_0916
was registered; its outputs are listed in the spec §0. Inputs are read from absolute paths on white-san:
checkpoints copied to ~/Projects/obsidian-research-data/elu_three_studies_0913/ and, for h3_mucap_ref_l2.py,
the ref arm of the running mucap_el_0916 (cap arms not read). Run from the repo root of proj_004_drift or
this worktree; outputs are in results/l2_wall_0916/posthoc/."""
"""Geometry-only check (posthoc, no training): what LayerNorm would do to the wall directions of the
representations stored in elu_environment_0913/checkpoint.pt (task 50). Weights are NOT re-trained with LN;
this only asks whether a one-sided mean direction survives the LN map for these inputs."""
import sys, numpy as np, torch
sys.path.insert(0, "/home/issan/Projects/claude/proj_004_drift")
from src import elu_environment_0913 as E
ck = torch.load("/home/issan/Projects/obsidian-research-data/elu_three_studies_0913/files/results/elu_environment_0913/checkpoint.pt", map_location="cpu", weights_only=False)
ax = E.read_idx(E.DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255
subset = {s: torch.randperm(len(ax), generator=E.stream("rl_subset", s))[:E.N].numpy() for s in range(3)}
# task-50 permutation per seed (the perm generator is consumed once per task)
gperm = {s: E.stream("env_perm_0913", s) for s in range(3)}
perm50 = {}
for s in range(3):
    for t in range(1, 51): pp = torch.randperm(784, generator=gperm[s])
    perm50[s] = pp
def act(z, a):
    return torch.where(z > 0, z, torch.expm1(z.clamp_max(0))) if a == "ELU1" else torch.where(z > 0, z, 0.1 * z)
def LN(a, eps=1e-5):
    return (a - a.mean(-1, keepdim=True)) / (a.var(-1, unbiased=False, keepdim=True) + eps).sqrt()
def wall(u):
    """mean-direction wall constants of an input cloud u [N,n]: |mu|, cos(mu,1), c, kappa, min proj, and the
    all-ones channel S=1'u: mean, sd, c_1 = mean/sd, min."""
    mu = u.mean(0); r = mu.norm(); e = mu / r; p = u @ e
    n = u.shape[1]; S = u.sum(1)
    c = float(r / p.std(unbiased=False)); k = float((p.mean() - p.min()) / p.std(unbiased=False))
    cos1 = float(mu.sum() / (r * n ** 0.5))
    sdS = float(S.std(unbiased=False))
    return dict(r=float(r), cos1=cos1, c=c, k=k, pmin=float(p.min()), onesided=bool(p.min() > 0),
                S_mean=float(S.mean()), S_sd=sdS, c1=(float(S.mean()) / sdS) if sdS > 1e-9 else float("inf"), S_min=float(S.min()))
def fmt(w):
    c1 = "inf" if w["c1"] == float("inf") else f"{w['c1']:+.2f}"
    return f"|mu| {w['r']:7.2f} cos(mu,1) {w['cos1']:+.2f} | mu-dir c {w['c']:6.2f} k {w['k']:5.2f} min {w['pmin']:+8.2f} {'ONE-SIDED' if w['onesided'] else 'two-sided'} | 1-dir mean {w['S_mean']:+8.2f} sd {w['S_sd']:7.2f} c1 {c1}"
P = [q.double() for q in ck["parameters"]]
for j, m in enumerate(ck["models"]):
    if m["iv"] != "ref": continue
    s, a = m["seed"], m["act"]
    x = torch.from_numpy(ax[subset[s]]).double()
    if m["env"] == "PM": x = x[:, perm50[s]]
    p = [q[j] for q in P]
    z1 = x @ p[0].T + p[1]; a1 = act(z1, a)
    print(f"\n== {m['env']} {a} seed {s} (task 50)")
    if s == 0:
        print("  L1 input x           :", fmt(wall(x)))
        print("  L1 input LN(x)       :", fmt(wall(LN(x))))
    print("  L2 input a1          :", fmt(wall(a1)))
    print("  L2 input LN(a1)      :", fmt(wall(LN(a1))))
    print("  L2 input ReLU(LN(z1)):", fmt(wall(torch.relu(LN(z1)))))
    print("  L2 input ELU(LN(z1)) :", fmt(wall(act(LN(z1), 'ELU1'))))
    # pre-activation LN at layer 2 itself: per-sample mean over units of LN(z2) is 0 -> how many units positive per sample
    z2 = a1 @ p[2].T + p[3]
    n2 = LN(z2)
    print(f"  L2 pre-act LN(z2): per-sample share of units >0: min {float((n2 > 0).double().mean(1).min()):.2f} median {float((n2 > 0).double().mean(1).median()):.2f} | raw z2: share of samples with ANY unit >0: {float((z2 > 0).any(1).double().mean()):.2f}")
    # per-unit sign consistency after LN at layer-2 input: units whose LN'd value is below the per-sample mean for all samples
    u = LN(a1)
    print(f"  LN(a1): coords with same sign over all samples: neg {int((u < 0).all(0).sum())} pos {int((u > 0).all(0).sum())} of {u.shape[1]} | a1 coords with mean<-0.9: {int((a1.mean(0) < -0.9).sum())}")

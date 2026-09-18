"""phi2_projection_0918: does the φ′² projection-contraction (AQ note 0918) hold on the actual CondA network?

Post-hoc verification of a theory memo (specs/spec_phi2_projection_0918.md), no new long training.
float64 re-implementation of the production batch-1 SGD (same path as the eta-ladder script):
W/b/v/c all updated, shared input stream across paired arms.

  T1/T2  `perturb`: two copies of the checkpoint network, one with w_i += eps*muhat for one unit
         per seed; both follow the same stream for T updates. Records the difference per block.
  T3     `switch` : one task switch (1 flip bit, fixed-dose offset recomputed) then 10^4 updates;
         per-unit decomposition of the task increment D into muhat / S_A⊖muhat / N_A.
"""
import sys, json, itertools, argparse, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.nets import VecMLPL  # noqa: E402

torch.set_num_threads(2)
CK = Path("/home/issan/Projects/obsidian-research-data/zero_attraction_0913/training/ckpts")
OUT = ROOT / "results" / "phi2_projection_0918"
TARGET = 3.041          # target_mu_norm of the fixed-dose arms (checkpoint field)
LR = 0.005
H = 100
ARMS = ["LR_a0p1_q0", "LR_a0p7_q0"]
STEPS = [200000, 5000000]
REC_T = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]


def gamma_for_k(k):
    """fixed-dose offset: off = gamma/2 makes ||mu|| = TARGET exactly for k ones in the flip block."""
    disc = (k + 2.5) ** 2 - 20.0 * (k + 1.25 - TARGET ** 2)
    return ((k + 2.5) - disc.sqrt()) / 10.0


def teacher_y(t, raw):
    pre = torch.einsum("rhd,srd->srh", t["W"].float(), raw.float()) + t["b"].float()
    return (((pre >= t["tau"].float()).float() * t["v"].float()).sum(-1) + t["cout"].float()).double()


def load(arm, step):
    cp = torch.load(CK / f"{arm}_step{step}.pt", map_location="cpu", weights_only=True)
    R = cp["net"]["W"].shape[0]
    f = cp["env"]["flip_state"].shape[1]
    m = cp["net"]["W"].shape[2]
    return cp, R, f, m


def make_net(cp, R, m):
    n = VecMLPL(R, [H], m, torch.Generator().manual_seed(0), "cpu",
                act=cp["activation"], act_alpha=cp["act_alpha"])
    n.load_state({k: v.double().clone() for k, v in cp["net"].items()})
    return n


def support(cp, flip, R, m, f):
    """32-pattern support of the task with flip state `flip` [R,f]: raw, x=raw-off, y, mu, P_A, N_A."""
    off = 0.5 * gamma_for_k(flip.sum(1))                                   # [R]
    bits = torch.tensor(list(itertools.product([0., 1.], repeat=m - f))).double()
    raw = torch.cat([flip[None].expand(32, -1, -1), bits[:, None].expand(-1, R, -1)], -1)  # [32,R,m]
    y = teacher_y(cp["teacher"], raw)                                      # [32,R]
    x = raw - off[None, :, None]
    mu = x.mean(0)                                                          # [R,m]
    P_A = torch.zeros(R, m, m, dtype=torch.float64)
    rank = []
    for r in range(R):
        U, S, Vh = torch.linalg.svd(x[:, r, :], full_matrices=False)
        k = int((S > S[0] * 1e-10).sum())
        rank.append(k)
        B = Vh[:k].T                                                        # [m,k]
        P_A[r] = B @ B.T
    N_A = torch.eye(m, dtype=torch.float64)[None] - P_A
    return dict(off=off, raw=raw, x=x, y=y, mu=mu, muhat=mu / mu.norm(dim=1, keepdim=True),
                P_A=P_A, N_A=N_A, rank=rank, bits=bits)


def sup_forward(n, x_sup):
    pre = torch.einsum("rhd,srd->srh", n.W, x_sup) + n.b
    a = torch.where(pre > 0, pre, n.act_alpha * pre)
    yh = (a * n.v).sum(-1) + n.c
    return pre, a, yh


def branch_groups(pre_sup, margin):
    """pre_sup [32,R,H] -> 'pos' if all 32 patterns > margin, 'neg' if all < -margin, else 'mix'."""
    pos = (pre_sup > margin).all(0)
    neg = (pre_sup < -margin).all(0)
    return pos, neg


def gn_endpoint(n, sup, u, q):
    """Linear Gauss-Newton endpoint of a parameter difference that starts as `q` in the w_u block:
    Δθ_∞ = (I - P_J) q, P_J = orthogonal projector onto the row space of the 32-pattern Jacobian J.
    Returns the w_u block of Δθ_∞ [R,m], plus the per-seed NTK diagnostics."""
    R, m = n.W.shape[0], n.W.shape[2]
    pre, a, _ = sup_forward(n, sup["x"])                                    # [32,R,H]
    k = torch.where(pre > 0, torch.ones_like(pre), torch.full_like(pre, n.act_alpha))
    x = sup["x"]                                                            # [32,R,m]
    out = torch.zeros(R, m, dtype=torch.float64)
    diag = []
    for r in range(R):
        vk = n.v[r][None, :] * k[:, r, :]                                   # [32,H]
        JW = (vk[:, :, None] * x[:, r, None, :]).reshape(32, H * m)         # w blocks
        Jb = vk                                                             # b blocks
        Jv = a[:, r, :]                                                     # v blocks
        Jc = torch.ones(32, 1, dtype=torch.float64)
        J = torch.cat([JW, Jb, Jv, Jc], 1)                                  # [32, P]
        K = J @ J.T
        ui = int(u[r])
        Jq = vk[:, ui] * (x[:, r, :] @ q[r])                                # [32]
        alpha = torch.linalg.pinv(K) @ Jq
        corr = (alpha[:, None] * vk[:, ui, None] * x[:, r, :]).sum(0)      # w_u block of P_J q
        out[r] = q[r] - corr
        nJ2 = (J ** 2).sum(1)                                               # ||J_p||^2 per pattern
        share = (n.v[r, ui] ** 2 * k[:, r, ui] ** 2 * ((x[:, r, :] ** 2).sum(1) + 1.0)) / nJ2
        diag.append(dict(J2_mean=nJ2.mean().item(), J2_max=nJ2.max().item(),
                         Jv2_mean=(Jv ** 2).sum(1).mean().item(), JW2_mean=(JW ** 2).sum(1).mean().item(),
                         share_mean=share.mean().item(), K_cond=torch.linalg.cond(K).item()))
    return out, diag


def run_perturb(arm, step, eps, T, n_pos, n_neg, margin, seed, out_dir, qdir="mu", freeze_others=False):
    """qdir: 'mu' -> q = eps*muhat (visible only); 'munull' -> q = eps*(muhat + nhat)/sqrt2 with nhat a random
    unit vector of N_A (reviewer's non-vacuous control for N_A preservation).
    freeze_others: both copies update ONLY the perturbed unit's incoming w (the single-unit premise of the AQ note;
    positive control: the visible part must then be erased completely)."""
    cp, R, f, m = load(arm, step)
    flip = cp["env"]["flip_state"].clone().double()      # continue the CURRENT task (no flip)
    sup = support(cp, flip, R, m, f)
    n0 = make_net(cp, R, m)
    pre_sup, a_sup, _ = sup_forward(n0, sup["x"])
    pos, neg = branch_groups(pre_sup, margin)
    gq = torch.Generator().manual_seed(seed + 1)
    nhat = torch.einsum("rij,rj->ri", sup["N_A"], torch.randn(R, m, generator=gq, dtype=torch.float64))
    nhat = nhat / nhat.norm(dim=1, keepdim=True)
    qunit = sup["muhat"] if qdir == "mu" else (sup["muhat"] + nhat) / np.sqrt(2.0)
    # unit choice per seed: most robust positive / negative units by kink margin, then by |v| descending
    rows = []
    stream = torch.randint(0, 2, (T, R, m - f), generator=torch.Generator().manual_seed(seed)).double()
    for grp, mask, cnt in (("pos", pos, n_pos), ("neg", neg, n_neg)):
        marg = torch.where(mask, pre_sup.abs().min(0).values, torch.full_like(n0.v, -1.0))
        order = torch.argsort(marg, dim=1, descending=True)                 # [R,H]
        for j in range(cnt):
            u = order[:, j]
            ok = mask[torch.arange(R), u]
            rows.append((grp, j, u, ok))
    records, onestep, endpoints = [], [], []
    for grp, j, u, ok in rows:
        A = make_net(cp, R, m)
        B = make_net(cp, R, m)
        q0 = eps * qunit                                                    # [R,m]
        B.W[torch.arange(R), u, :] += q0
        rowmask = torch.zeros(R, H, 1, dtype=torch.float64); rowmask[torch.arange(R), u, 0] = 1.0
        gn_w, gn_diag = gn_endpoint(A, sup, u, q0)
        qNA0 = torch.einsum("rij,rj->ri", sup["N_A"], q0)
        lrv = torch.full((R,), LR, dtype=torch.float64)
        gate_mismatch_stream = torch.zeros(R, dtype=torch.int64)
        v0 = A.v.clone()
        for t in range(1, T + 1):
            raw = torch.cat([flip, stream[t - 1]], 1)
            y = teacher_y(cp["teacher"], raw[None])[0]
            x = raw - sup["off"][:, None]
            preA, aA, yA = A.forward(x)
            preB, aB, yB = B.forward(x)
            gate_mismatch_stream += ((preA > 0) != (preB > 0)).sum(1)
            if t == 1:
                # exact one-step prediction (memo §4) from the unperturbed state
                kA = torch.where(preA > 0, torch.ones_like(preA), torch.full_like(preA, A.act_alpha))
                eA = yA - y
                qx = (q0 * x).sum(1)                                        # [R]
                vk_u = A.v[torch.arange(R), u] * kA[torch.arange(R), u]
                de = vk_u * qx                                              # δŷ
                pred_qu = q0 - 2 * LR * vk_u[:, None] ** 2 * qx[:, None] * x
                pred_dW = -2 * LR * (de[:, None, None] * (A.v * kA)[:, :, None] * x[:, None, :])
                pred_dW[torch.arange(R), u, :] = pred_qu
                pred_db = -2 * LR * de[:, None] * A.v * kA
                pred_dv = -2 * LR * de[:, None] * aA
                pred_dv[torch.arange(R), u] += -2 * LR * eA * kA[torch.arange(R), u] * qx
                pred_dc = -2 * LR * de
            gA = A.grads(x, preA, aA, yA - y)
            gB = B.grads(x, preB, aB, yB - y)
            if freeze_others:   # single-unit premise: only w_u moves, in both copies
                gA = (gA[0] * rowmask, torch.zeros_like(gA[1]), torch.zeros_like(gA[2]), torch.zeros_like(gA[3]))
                gB = (gB[0] * rowmask, torch.zeros_like(gB[1]), torch.zeros_like(gB[2]), torch.zeros_like(gB[3]))
            A.sgd_step(lrv, *gA)
            B.sgd_step(lrv, *gB)
            if t == 1:
                dW, db, dv, dc = B.W - A.W, B.b - A.b, B.v - A.v, B.c - A.c
                rel = lambda a_, b_: ((a_ - b_).norm() / b_.norm().clamp_min(1e-300)).item()
                for r in range(R):
                    onestep.append(dict(arm=arm, step=step, group=grp, rank=j, seed=r, unit=int(u[r]),
                                        ok=bool(ok[r]), eps=eps,
                                        rel_err_wu=rel(dW[r, u[r]], pred_dW[r, u[r]]),
                                        rel_err_W_other=rel(dW[r], pred_dW[r]),
                                        rel_err_b=rel(db[r], pred_db[r]), rel_err_v=rel(dv[r], pred_dv[r]),
                                        rel_err_c=abs((dc[r] - pred_dc[r]).item()) / max(abs(pred_dc[r].item()), 1e-300),
                                        lambda_x=(2 * LR * vk_u[r] ** 2 * (x[r] ** 2).sum()).item(),
                                        dyhat1=de[r].item()))
            if t in REC_T:
                dW, db, dv, dc = B.W - A.W, B.b - A.b, B.v - A.v, B.c - A.c
                qi = dW[torch.arange(R), u, :]
                qPA = torch.einsum("rij,rj->ri", sup["P_A"], qi)
                qNA = torch.einsum("rij,rj->ri", sup["N_A"], qi)
                dW_other = dW.clone(); dW_other[torch.arange(R), u, :] = 0
                _, _, ysA = sup_forward(A, sup["x"]); preSA, _, ysB = sup_forward(B, sup["x"])
                preSB = torch.einsum("rhd,srd->srh", B.W, sup["x"]) + B.b
                dys = ((ysB - ysA) ** 2).mean(0).sqrt()                     # [R] RMS over 32 patterns
                gm_sup = ((preSA > 0) != (preSB > 0)).sum(0).sum(1)         # [R]
                for r in range(R):
                    records.append(dict(arm=arm, step=step, group=grp, rank=j, seed=r, unit=int(u[r]),
                                        ok=bool(ok[r]), eps=eps, qdir=qdir, freeze_others=bool(freeze_others), t=t,
                                        q_NA0=qNA0[r].norm().item(),
                                        q_mu=(qi[r] @ sup["muhat"][r]).item(), q_mu0=(q0[r] @ sup["muhat"][r]).item(),
                                        q_PA=qPA[r].norm().item(), q_PA0=torch.einsum("ij,j->i", sup["P_A"][r], q0[r]).norm().item(),
                                        q_NA_dev=(qNA[r] - qNA0[r]).norm().item(),
                                        dW_other=dW_other[r].norm().item(), db=db[r].norm().item(),
                                        dv=dv[r].norm().item(), dv_u=dv[r, u[r]].item(), dc=dc[r].item(),
                                        dyhat_rms=dys[r].item(), dyhat_rms0=abs(de[r].item()) if t == 1 else float("nan"),
                                        gate_mismatch_sup=int(gm_sup[r]), gate_mismatch_stream=int(gate_mismatch_stream[r]),
                                        gn_q_mu=(gn_w[r] @ sup["muhat"][r]).item(),
                                        gn_q_PA=torch.einsum("ij,j->i", sup["P_A"][r], gn_w[r]).norm().item(),
                                        v_u=v0[r, u[r]].item(), k_u=float(1.0 if grp == "pos" else A.act_alpha),
                                        **{f"ntk_{kk}": vv for kk, vv in gn_diag[r].items()}))
        print(f"  {arm} {step} {grp}{j} done", flush=True)
    return records, onestep


def run_kick(arm, step, delta, T, n_units, zmax, seed):
    """T4 (post hoc, suggested by the round-1 review): symmetric ± bias kicks on kink-straddling units.
    Three copies share the stream: A (none), P (b_u += delta), M (b_u -= delta). Records the surviving kick
    s_± = (zbar_±(t) - zbar_A(t))/(±delta) and the pair mean (zbar_+ + zbar_- - 2 zbar_A)/2 (ratchet signature)."""
    cp, R, f, m = load(arm, step)
    flip = cp["env"]["flip_state"].clone().double()
    sup = support(cp, flip, R, m, f)
    n0 = make_net(cp, R, m)
    pre_sup, _, _ = sup_forward(n0, sup["x"])
    zbar_sup = pre_sup.mean(0)                                              # [R,H]
    mixed = (pre_sup > 0).any(0) & (pre_sup <= 0).any(0)
    score = torch.where(mixed & (zbar_sup.abs() < zmax), zbar_sup.abs(), torch.full_like(zbar_sup, 1e9))
    order = torch.argsort(score, dim=1)
    stream = torch.randint(0, 2, (T, R, m - f), generator=torch.Generator().manual_seed(seed)).double()
    recs = []
    for j in range(n_units):
        u = order[:, j]
        ok = score[torch.arange(R), u] < 1e8
        nets = {s: make_net(cp, R, m) for s in ("A", "P", "M")}
        nets["P"].b[torch.arange(R), u] += delta
        nets["M"].b[torch.arange(R), u] -= delta
        lrv = torch.full((R,), LR, dtype=torch.float64)
        npos0 = (pre_sup[:, torch.arange(R), u] > 0).sum(0)
        for t in range(1, T + 1):
            raw = torch.cat([flip, stream[t - 1]], 1)
            y = teacher_y(cp["teacher"], raw[None])[0]
            x = raw - sup["off"][:, None]
            for s, n in nets.items():
                pre, a, yh = n.forward(x)
                n.sgd_step(lrv, *n.grads(x, pre, a, yh - y))
            if t in REC_T:
                zb = {s: (torch.einsum("rhd,rd->rh", n.W, sup["mu"]) + n.b)[torch.arange(R), u] for s, n in nets.items()}
                ys = {s: sup_forward(n, sup["x"])[2] for s, n in nets.items()}
                pres = {s: sup_forward(n, sup["x"])[0][:, torch.arange(R), u] for s, n in nets.items()}
                for r in range(R):
                    recs.append(dict(arm=arm, step=step, rank=j, seed=r, unit=int(u[r]), ok=bool(ok[r]), delta=delta, t=t,
                                     zbar0=zbar_sup[r, u[r]].item(), npos0=int(npos0[r]),
                                     s_plus=((zb["P"][r] - zb["A"][r]) / delta).item(), s_minus=((zb["M"][r] - zb["A"][r]) / (-delta)).item(),
                                     pair_mean=((zb["P"][r] + zb["M"][r] - 2 * zb["A"][r]) / 2).item(),
                                     db_plus=(nets["P"].b[r, u[r]] - nets["A"].b[r, u[r]]).item(), db_minus=(nets["M"].b[r, u[r]] - nets["A"].b[r, u[r]]).item(),
                                     dwmu_plus=((nets["P"].W[r, u[r]] - nets["A"].W[r, u[r]]) @ sup["mu"][r]).item(),
                                     dwmu_minus=((nets["M"].W[r, u[r]] - nets["A"].W[r, u[r]]) @ sup["mu"][r]).item(),
                                     dyhat_plus=((ys["P"][:, r] - ys["A"][:, r]) ** 2).mean().sqrt().item(),
                                     dyhat_minus=((ys["M"][:, r] - ys["A"][:, r]) ** 2).mean().sqrt().item(),
                                     npos_A=int((pres["A"][:, r] > 0).sum()), npos_P=int((pres["P"][:, r] > 0).sum()), npos_M=int((pres["M"][:, r] > 0).sum()),
                                     dW_other_plus=(nets["P"].W - nets["A"].W)[r].norm().item(), dv_plus=(nets["P"].v - nets["A"].v)[r].norm().item(),
                                     v_u=cp["net"]["v"][r, u[r]].item()))
        print(f"  kick {arm} {step} unit-rank {j} done", flush=True)
    return recs


def run_switch(arm, step, T, seed, margin):
    cp, R, f, m = load(arm, step)
    g = torch.Generator().manual_seed(seed)
    flip = cp["env"]["flip_state"].clone().double()
    idx = torch.randint(0, f, (R,), generator=g)
    flip[torch.arange(R), idx] = 1 - flip[torch.arange(R), idx]              # task switch: one bit per seed
    sup = support(cp, flip, R, m, f)
    stream = torch.randint(0, 2, (T, R, m - f), generator=g).double()
    n = make_net(cp, R, m)
    pre0, _, _ = sup_forward(n, sup["x"])
    pos, neg = branch_groups(pre0, margin)
    W0, b0 = n.W.clone(), n.b.clone()
    zbar0 = torch.einsum("rhd,rd->rh", W0, sup["mu"]) + b0
    lrv = torch.full((R,), LR, dtype=torch.float64)
    dz_pos = torch.zeros(R, H, dtype=torch.float64); dz_neg = torch.zeros_like(dz_pos)
    for t in range(T):
        raw = torch.cat([flip, stream[t]], 1)
        y = teacher_y(cp["teacher"], raw[None])[0]
        x = raw - sup["off"][:, None]
        pre, a, yh = n.forward(x)
        e = yh - y
        k = torch.where(pre > 0, torch.ones_like(pre), torch.full_like(pre, n.act_alpha))
        dzb = -LR * (2 * e[:, None] * n.v * k) * ((x * sup["mu"]).sum(-1) + 1)[:, None]
        dz_pos += torch.where(pre > 0, dzb, torch.zeros_like(dzb)); dz_neg += torch.where(pre <= 0, dzb, torch.zeros_like(dzb))
        n.sgd_step(lrv, *n.grads(x, pre, a, e))
    W1, b1 = n.W, n.b
    D = W1 - W0
    mh = sup["muhat"]
    D_mu = (D * mh[:, None, :]).sum(-1, keepdim=True) * mh[:, None, :]
    D_PA = torch.einsum("rij,rhj->rhi", sup["P_A"], D)
    D_perp = D_PA - D_mu
    D_N = D - D_PA
    part = lambda Dp: 2 * (W0 * Dp).sum(-1) + (Dp ** 2).sum(-1)
    dnorm = (W1 ** 2).sum(-1) - (W0 ** 2).sum(-1)
    p_mu, p_perp, p_N = part(D_mu), part(D_perp), part(D_N)
    zbar1 = torch.einsum("rhd,rd->rh", W1, sup["mu"]) + b1
    cos = lambda Wm: (Wm * mh[:, None, :]).sum(-1) / Wm.norm(dim=-1)
    rows = []
    for r in range(R):
        for i in range(H):
            rows.append(dict(arm=arm, step=step, seed=r, unit=i,
                             group="pos" if pos[r, i] else ("neg" if neg[r, i] else "mix"),
                             zbar0=zbar0[r, i].item(), zbar1=zbar1[r, i].item(), dzbar=(zbar1 - zbar0)[r, i].item(),
                             dz_pos_samples=dz_pos[r, i].item(), dz_neg_samples=dz_neg[r, i].item(),
                             db=(b1 - b0)[r, i].item(),
                             dnorm2=dnorm[r, i].item(), part_mu=p_mu[r, i].item(), part_perp=p_perp[r, i].item(),
                             part_N=p_N[r, i].item(),
                             closure=(dnorm[r, i] - p_mu[r, i] - p_perp[r, i] - p_N[r, i]).abs().item(),
                             D_norm=D[r, i].norm().item(), D_mu=D_mu[r, i].norm().item(),
                             D_perp=D_perp[r, i].norm().item(), D_N=D_N[r, i].norm().item(),
                             w0_norm=W0[r, i].norm().item(), w1_norm=W1[r, i].norm().item(),
                             w0_mu=(W0[r, i] @ mh[r]).item(), w1_mu=(W1[r, i] @ mh[r]).item(),
                             cos0=cos(W0)[r, i].item(), cos1=cos(W1)[r, i].item(),
                             v0=cp["net"]["v"][r, i].item(), rank_SA=sup["rank"][r]))
    print(f"  switch {arm} {step} done", flush=True)
    return rows


def write_csv(path, rows):
    import csv
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["perturb", "switch", "kick"])
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--qdir", default="mu", choices=["mu", "munull"])
    ap.add_argument("--freeze_others", action="store_true")
    ap.add_argument("--delta", type=float, default=0.2)
    ap.add_argument("--n_units", type=int, default=5)
    ap.add_argument("--zmax", type=float, default=0.3)
    ap.add_argument("--T", type=int, default=10000)
    ap.add_argument("--n_pos", type=int, default=3)
    ap.add_argument("--n_neg", type=int, default=3)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--steps", default=",".join(map(str, STEPS)))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    arms = args.arms.split(","); steps = [int(s) for s in args.steps.split(",")]
    if args.mode == "perturb":
        recs, ones = [], []
        for arm in arms:
            for st in steps:
                r_, o_ = run_perturb(arm, st, args.eps, args.T, args.n_pos, args.n_neg, args.margin, args.seed, OUT,
                                     qdir=args.qdir, freeze_others=args.freeze_others)
                recs += r_; ones += o_
        write_csv(OUT / f"perturb{args.tag}_eps{args.eps:g}.csv", recs)
        write_csv(OUT / f"onestep{args.tag}_eps{args.eps:g}.csv", ones)
    elif args.mode == "kick":
        rows = []
        for arm in arms:
            for st in steps:
                rows += run_kick(arm, st, args.delta, args.T, args.n_units, args.zmax, args.seed)
        write_csv(OUT / f"kick{args.tag}_delta{args.delta:g}.csv", rows)
    else:
        rows = []
        for arm in arms:
            for st in steps:
                rows += run_switch(arm, st, args.T, args.seed, args.margin)
        write_csv(OUT / f"switch{args.tag}.csv", rows)
    meta = dict(mode=args.mode, args=vars(args), lr=LR, target=TARGET, rec_t=REC_T, elapsed_s=time.time() - t0,
                torch=torch.__version__, numpy=np.__version__)
    (OUT / f"meta_{args.mode}{args.tag}.json").write_text(json.dumps(meta, indent=1))
    print("elapsed", round(time.time() - t0, 1), "s")

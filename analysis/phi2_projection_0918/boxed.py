"""Direct test of the boxed task-averaged formula of the AQ note (Issa's primary target):

    q+ = (I - 2 eta v^2 E_A[phi'(z)^2 x x^T]) q      (H_A = 2 v^2 E_A[k^2 x x^T], exact expectation over the 32 patterns)

(a) full-batch GD on w_u only (both copies): q_t must equal (I - eta H_A)^t q_0 to machine precision while
    both copies stay in the same branches.
(b) batch-1 SGD on w_u only (the note's conditional-mean case): q_t vs (I - eta H_A)^t q_0, approximate.
(c) natural full-network SGD (all W/b/v/c): same comparison; the formula is expected to fail (NTK share).
All in float64, same units as T2 (pure pos/neg, |z|>0.03 on all 32 patterns), q0 = eps*muhat, shared stream."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import load, make_net, support, sup_forward, branch_groups, teacher_y, OUT, H, ARMS, STEPS, LR  # noqa: E402

torch.set_num_threads(2)
REC = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]


def H_A(n, sup, u):
    """2 v_u^2 E_A[k^2 x x^T] per seed, [R,m,m], using the current (unperturbed) branches."""
    pre, _, _ = sup_forward(n, sup["x"])
    k = torch.where(pre > 0, torch.ones_like(pre), torch.full_like(pre, n.act_alpha))
    R = n.W.shape[0]
    ku = k[:, torch.arange(R), u]                                            # [32,R]
    x = sup["x"]                                                             # [32,R,m]
    E = torch.einsum("pr,pri,prj->rij", ku ** 2, x, x) / 32.0
    return 2.0 * (n.v[torch.arange(R), u] ** 2)[:, None, None] * E


def run_cond(arm, step, eps, T, n_pos, n_neg, margin, seed):
    cp, R, f, m = load(arm, step)
    flip = cp["env"]["flip_state"].clone().double()
    sup = support(cp, flip, R, m, f)
    n0 = make_net(cp, R, m)
    pre_sup, _, _ = sup_forward(n0, sup["x"])
    pos, neg = branch_groups(pre_sup, margin)
    stream = torch.randint(0, 2, (T, R, m - f), generator=torch.Generator().manual_seed(seed)).double()
    y_sup = sup["y"]                                                          # [32,R]
    rows = []
    for grp, mask, cnt in (("pos", pos, n_pos), ("neg", neg, n_neg)):
        marg = torch.where(mask, pre_sup.abs().min(0).values, torch.full_like(n0.v, -1.0))
        order = torch.argsort(marg, dim=1, descending=True)
        for j in range(cnt):
            u = order[:, j]
            ok = mask[torch.arange(R), u]
            if not bool(ok.any()):
                continue
            q0 = eps * sup["muhat"]
            HA = H_A(n0, sup, u)                                              # [R,m,m]
            M = torch.eye(m, dtype=torch.float64)[None] - LR * HA               # one-step mean operator
            ev = torch.linalg.eigvalsh(LR * HA)                                 # eta*lambda, in [0,2) for contraction
            rowmask = torch.zeros(R, H, 1, dtype=torch.float64); rowmask[torch.arange(R), u, 0] = 1.0
            for mode in ("fullbatch_wonly", "sgd_wonly", "natural"):
                A = make_net(cp, R, m); B = make_net(cp, R, m)
                B.W[torch.arange(R), u, :] += q0
                Mt = torch.eye(m, dtype=torch.float64)[None].expand(R, -1, -1).clone()
                lrv = torch.full((R,), LR, dtype=torch.float64)
                mism = torch.zeros(R, dtype=torch.int64)
                for t in range(1, T + 1):
                    if mode == "fullbatch_wonly":
                        for net in (A, B):
                            pre, a, yh = sup_forward(net, sup["x"])                 # [32,R,·]
                            e = yh - y_sup                                          # [32,R]
                            k = torch.where(pre > 0, torch.ones_like(pre), torch.full_like(pre, net.act_alpha))
                            gW = torch.einsum("pr,prh,pri->rhi", 2.0 * e, net.v[None] * k, sup["x"]) / 32.0
                            net.W -= lrv[:, None, None] * (gW * rowmask)
                        pA, _, _ = sup_forward(A, sup["x"]); pB, _, _ = sup_forward(B, sup["x"])
                        mism += ((pA > 0) != (pB > 0))[:, torch.arange(R), u].sum(0)
                    else:
                        raw = torch.cat([flip, stream[t - 1]], 1)
                        y = teacher_y(cp["teacher"], raw[None])[0]
                        x = raw - sup["off"][:, None]
                        preA, aA, yA = A.forward(x); preB, aB, yB = B.forward(x)
                        mism += ((preA > 0) != (preB > 0))[torch.arange(R), u]
                        gA = A.grads(x, preA, aA, yA - y); gB = B.grads(x, preB, aB, yB - y)
                        if mode == "sgd_wonly":
                            gA = (gA[0] * rowmask, 0 * gA[1], 0 * gA[2], 0 * gA[3])
                            gB = (gB[0] * rowmask, 0 * gB[1], 0 * gB[2], 0 * gB[3])
                        A.sgd_step(lrv, *gA); B.sgd_step(lrv, *gB)
                    Mt = torch.bmm(M, Mt)                                        # (I - eta H_A)^t
                    if t in REC:
                        q_emp = (B.W - A.W)[torch.arange(R), u, :]
                        q_pred = torch.einsum("rij,rj->ri", Mt, q0)
                        for r in range(R):
                            if not bool(ok[r]):
                                continue
                            qe, qp = q_emp[r], q_pred[r]
                            rows.append(dict(arm=arm, step=step, group=grp, rank=j, seed=r, unit=int(u[r]), mode=mode, t=t,
                                             rel_err=float((qe - qp).norm() / q0[r].norm()),
                                             rel_err_vs_pred=float((qe - qp).norm() / qp.norm().clamp_min(1e-300)),
                                             cos=float((qe @ qp) / (qe.norm() * qp.norm()).clamp_min(1e-300)),
                                             surv_emp=float(qe.norm() / q0[r].norm()), surv_pred=float(qp.norm() / q0[r].norm()),
                                             surv_PA_emp=float(torch.einsum("ij,j->i", sup["P_A"][r], qe).norm() / q0[r].norm()),
                                             surv_PA_pred=float(torch.einsum("ij,j->i", sup["P_A"][r], qp).norm() / q0[r].norm()),
                                             eta_lam_max=float(ev[r].max()), eta_lam_min_pos=float(ev[r][ev[r] > 1e-12].min()) if bool((ev[r] > 1e-12).any()) else 0.0,
                                             rank_HA=int((ev[r] > 1e-12).sum()), gate_mismatch=int(mism[r]), v_u=float(n0.v[r, u[r]]),
                                             k_u=1.0 if grp == "pos" else float(n0.act_alpha)))
            print(f"  boxed {arm} {step} {grp}{j} done", flush=True)
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=2000); ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--n_pos", type=int, default=5); ap.add_argument("--n_neg", type=int, default=5)
    ap.add_argument("--margin", type=float, default=0.03); ap.add_argument("--seed", type=int, default=20260918)
    args = ap.parse_args()
    rows = []
    for arm in ARMS:
        for st in STEPS:
            rows += run_cond(arm, st, args.eps, args.T, args.n_pos, args.n_neg, args.margin, args.seed)
    df = pd.DataFrame(rows); df.to_csv(OUT / "boxed.csv", index=False)
    pd.set_option("display.width", 250)
    for mode in ("fullbatch_wonly", "sgd_wonly", "natural"):
        d = df[df["mode"] == mode]
        print(f"\n== {mode}: max rel_err (vs q0) by t; median survival emp vs pred ==")
        print(d.groupby("t").agg(max_rel_err=("rel_err", "max"), med_rel_err=("rel_err", "median"), med_cos=("cos", "median"),
                                 surv_emp=("surv_PA_emp", "median"), surv_pred=("surv_PA_pred", "median"),
                                 gate_mismatch_max=("gate_mismatch", "max"), n=("unit", "size")).round(8).to_string())
    d = df[(df["mode"] == "fullbatch_wonly") & (df.t == 1)]
    print("\neta*lambda_max by group (median/max):"); print(d.groupby(["arm", "step", "group"]).eta_lam_max.agg(["median", "max"]).round(4).to_string())

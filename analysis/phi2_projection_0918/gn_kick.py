"""Linear GN endpoint for the bias kicks of run_kick (post hoc): how much of a bias kick on a kink-straddling unit
can the REST of the network imitate?  s_inf = 1 - ([P_J q]_{b_u} + [P_J q]_{w_u}·mu)/delta for q = delta*e_{b_u}.
Compared with the observed survival s_± from kick*_delta*.csv."""
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import load, make_net, support, sup_forward, OUT, H, ARMS, STEPS  # noqa: E402

torch.set_num_threads(2)


def gn_bias_share(n, sup, u):
    R, m = n.W.shape[0], n.W.shape[2]
    pre, a, _ = sup_forward(n, sup["x"])
    k = torch.where(pre > 0, torch.ones_like(pre), torch.full_like(pre, n.act_alpha))
    x = sup["x"]
    rows = []
    for r in range(R):
        vk = n.v[r][None, :] * k[:, r, :]
        JW = (vk[:, :, None] * x[:, r, None, :]).reshape(32, H * m)
        J = torch.cat([JW, vk, a[:, r, :], torch.ones(32, 1, dtype=torch.float64)], 1)
        K = J @ J.T
        ui = int(u[r])
        Jq = vk[:, ui]                                       # per unit delta (delta=1)
        alpha = torch.linalg.pinv(K) @ Jq
        cb = (alpha * vk[:, ui]).sum()                       # b_u block of P_J q
        cw = ((alpha[:, None] * vk[:, ui, None] * x[:, r, :]).sum(0)) @ sup["mu"][r]   # w_u block · mu
        # how much of the output-error vector is imitable by everybody else (drop unit u's blocks)
        keep = torch.ones(J.shape[1], dtype=torch.bool)
        keep[ui * m:(ui + 1) * m] = False; keep[H * m + ui] = False; keep[H * m + H + ui] = False
        Jo = J[:, keep]; Ko = Jo @ Jo.T
        resid = Jq - Ko @ (torch.linalg.pinv(Ko) @ Jq)       # part of the output-error vector outside col(Jo)
        rows.append(dict(seed=r, unit=ui, s_inf=float(1 - cb - cw), share_b=float(cb), share_wmu=float(cw),
                         nonimitable_frac=float(resid.norm() ** 2 / Jq.norm() ** 2), v_u=float(n.v[r, ui]),
                         npos=int((pre[:, r, ui] > 0).sum())))
    return rows


if __name__ == "__main__":
    out = []
    for arm in ARMS:
        for st in STEPS:
            cp, R, f, m = load(arm, st)
            flip = cp["env"]["flip_state"].clone().double()
            sup = support(cp, flip, R, m, f)
            n = make_net(cp, R, m)
            pre_sup, _, _ = sup_forward(n, sup["x"])
            zbar = pre_sup.mean(0)
            mixed = (pre_sup > 0).any(0) & (pre_sup <= 0).any(0)
            score = torch.where(mixed & (zbar.abs() < 0.3), zbar.abs(), torch.full_like(zbar, 1e9))
            order = torch.argsort(score, dim=1)
            for j in range(5):
                u = order[:, j]
                for row in gn_bias_share(n, sup, u):
                    row.update(arm=arm, step=st, rank=j, ok=bool(score[row["seed"], u[row["seed"]]] < 1e8))
                    out.append(row)
    df = pd.DataFrame(out); df.to_csv(OUT / "gn_kick.csv", index=False)
    d = df[df.ok]
    obs = []
    for tag, dl in (("_d0005", 0.005), ("", 0.2)):
        k = pd.read_csv(OUT / f"kick{tag}_delta{dl:g}.csv"); k = k[(k.ok) & (k.t == 2000)]
        k = k.merge(d[["arm", "step", "seed", "unit", "s_inf", "nonimitable_frac"]], on=["arm", "step", "seed", "unit"])
        g = k.groupby(["arm", "step"]).agg(s_inf=("s_inf", "median"), nonimit=("nonimitable_frac", "median"),
                                           s_plus=("s_plus", "median"), s_minus=("s_minus", "median"), n=("unit", "size")).reset_index()
        g["delta"] = dl; obs.append(g)
    print(pd.concat(obs).sort_values(["arm", "step", "delta"]).round(3).to_string())
    print("\nper-unit correlation (delta=0.005, t=2000) of observed mean survival with GN s_inf:")
    k = pd.read_csv(OUT / "kick_d0005_delta0.005.csv"); k = k[(k.ok) & (k.t == 2000)]
    k = k.merge(d[["arm", "step", "seed", "unit", "s_inf"]], on=["arm", "step", "seed", "unit"])
    k["s_mean"] = (k.s_plus + k.s_minus) / 2
    for (arm, st), g in k.groupby(["arm", "step"]):
        print(arm, st, "spearman", round(g[["s_mean", "s_inf"]].corr(method="spearman").iloc[0, 1], 3), "n", len(g))

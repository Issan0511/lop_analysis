#!/usr/bin/env python3
"""reload_dial_1020 の解析: checkpoint ごとの自信・κ・κ の下界・期待の押しと、draw ごとの実際の切替（GD 一歩・Adam 1 歩・200 更新）。
float64。関数は analysis/kappa_reload_1020/measure.py を使う（閉じた式は autograd と照合済み）。
"""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kappa_reload_1020"))
import measure as M

RAW = Path("results/reload_dial_1020/raw"); OUT = Path("results/reload_dial_1020")
ACTS = ["LR", "R", "ELU", "SILU", "GELU"]; SEEDS = [0, 1]
CKS = ["init", "a20", "a50", "a80", "a95", "end"]
K = 10; LOGK = math.log(K)


def kappa_parts(e, Y, b2, N):
    """κ・揃い A・‖F‖ と、恒等式 A = Σ[(ℓ_y − ℓ̄) − b^c_y]、下界 A ≥ N(log K − L) − B。"""
    F = (Y - 1.0 / K).T @ e["a"]; ff = float((F * F).sum()); A = float((e["W2c"] * F).sum())
    y = Y.argmax(1); n = Y.sum(0)
    logits = e["a"] @ e["W2"].T + b2
    b2c = b2 - b2.mean()
    A_id = float(((logits[np.arange(len(y)), y] - logits.mean(1)) - b2c[y]).sum())
    L = float(-np.log(e["pi"][np.arange(len(y)), y]).mean())
    B = float(((n - len(y) / K) * b2c).sum())
    A_lb = N * (LOGK - L) - B
    normV = float(np.sqrt((e["W2c"] ** 2).sum()))
    return dict(kappa=A / ff if ff > 0 else float("nan"), A=A, A_id_relerr=abs(A - A_id) / max(abs(A), 1e-300),
                F_norm=math.sqrt(ff), normV=normV, L=L, B=B, A_lb=A_lb,
                kappa_lb=A_lb / ff if ff > 0 else float("nan"),
                kappa_ub=normV / math.sqrt(ff) if ff > 0 else float("nan"))


def analyze_run(act, seed):
    saved = np.load(M.STATES / f"{act}_K10_s{seed}.npz")
    X = M.load_X(saved["subset"]); N = X.shape[0]; xbar = X.mean(0); Kn = X @ xbar + 1.0
    Y1 = M.onehot(saved["y"][0].astype(int), K)
    sw = np.load(RAW / "switch" / f"switch_{act}_s{seed}.npz")
    steps = list(sw["save_steps"]); i1, i200 = steps.index(1), steps.index(200)
    rows = []
    for name in CKS:
        ck = np.load(RAW / "ckpt" / f"ckpt_{act}_s{seed}_{name}.npz")
        P = [ck[f"p{q}"].astype(np.float64) for q in range(4)]
        e = M.evaluate(P, act, X, xbar)
        y1 = Y1.argmax(1)
        kp = kappa_parts(e, Y1, P[3], N)
        alive = e["kopen"] > 0
        S_exact = (Kn[:, None] * e["g"] * (e["pi"] @ e["W2c"])).sum(0)
        S_self = kp["kappa"] * (1 - 1.0 / K) * (Kn[:, None] * e["g"] * e["a"]).sum(0)
        a_se, _ = M.agree(S_self, S_exact, alive)
        draws = []
        j = 0
        while f"{name}_d{j}_m" in sw.files:
            ysw = sw[f"{name}_d{j}_y"].astype(int); Ysw = M.onehot(ysw, K)
            delta = e["g"] * ((e["pi"] - Ysw) @ e["W2"])
            dm_gd = -(1e-3 / N) * (Kn[:, None] * delta).sum(0)
            mtraj = sw[f"{name}_d{j}_m"]
            dm1 = mtraj[i1] - e["m"]; dm200 = mtraj[i200] - e["m"]
            draws.append(dict(
                frac_sink_gd=float((dm_gd[alive] < 0).mean()), frac_sink_adam1=float((dm1[alive] < 0).mean()),
                frac_sink_adam200=float((dm200[alive] < 0).mean()),
                agree_exact_gd=M.agree(-S_exact, dm_gd, alive)[0], agree_exact_adam1=M.agree(-S_exact, dm1, alive)[0],
                agree_exact_adam200=M.agree(-S_exact, dm200, alive)[0],
                sink200_med=float(np.median(-dm200[alive])), sink1_med=float(np.median(-dm1[alive])),
                p_up_200=int(((e["pplus"] < 0.5) & (sw[f"{name}_d{j}_p"][i200] >= 0.5)).sum()),
                p_down_200=int(((e["pplus"] >= 0.5) & (sw[f"{name}_d{j}_p"][i200] < 0.5)).sum())))
            j += 1
        agg = {k: (float(np.mean([d[k] for d in draws])), float(np.std([d[k] for d in draws]))) for k in draws[0]}
        rows.append(dict(act=act, seed=seed, ck=name, update=int(ck["update"]),
                         acc=float((e["pi"].argmax(1) == y1).mean()),
                         conf_y=float(e["pi"][np.arange(N), y1].mean()), conf_max=float(e["pi"].max(1).mean()),
                         gain=LOGK - kp["L"], n_alive=int(alive.sum()), **kp,
                         S_exact_medN=float(np.median(S_exact[alive]) / N),
                         S_self_medN=float(np.median(S_self[alive]) / N),
                         frac_sink_exact=float((S_exact[alive] > 0).mean()), frac_sink_self=float((S_self[alive] > 0).mean()),
                         agree_self_exact=a_se, n_draws=len(draws), draws=agg,
                         p_below=int((e["pplus"] < 0.5).sum())))
    return rows


def bound_on_saved_tasks(act, seed):
    """kappa_reload_1020 と同じ 30 課題の課題末で、恒等式と下界を確かめる。"""
    d = np.load(M.STATES / f"{act}_K10_s{seed}.npz")
    X = M.load_X(d["subset"]); N = X.shape[0]; xbar = X.mean(0)
    out = []
    for t in range(d["end0"].shape[0]):
        P = [d[f"end{q}"][t].astype(np.float64) for q in range(4)]
        e = M.evaluate(P, act, X, xbar)
        kp = kappa_parts(e, M.onehot(d["y"][t].astype(int), K), P[3], N)
        if not math.isfinite(kp["kappa"]): continue
        out.append(dict(task=t + 1, **kp))
    return out


def main():
    rows = []; bound = []
    for act in ACTS:
        for s in SEEDS:
            rows += analyze_run(act, s)
            bound += [dict(act=act, seed=s, **b) for b in bound_on_saved_tasks(act, s)]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    (OUT / "bound_30tasks.json").write_text(json.dumps(bound, ensure_ascii=False, indent=1))
    worst_id = max(max(r["A_id_relerr"] for r in rows), max(b["A_id_relerr"] for b in bound))
    viol = [r for r in rows if r["kappa"] < r["kappa_lb"] - 1e-12] + [b for b in bound if b["kappa"] < b["kappa_lb"] - 1e-12]
    print(f"rows {len(rows)}  bound rows {len(bound)}  identity worst rel.err {worst_id:.2e}  lower-bound violations {len(viol)}")


if __name__ == "__main__":
    main()

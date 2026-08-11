"""B2 — c_t = Cov(f_t(x), x) との照合（H-cov の検証、仕様書 §4）。

  .venv/bin/python -m analysis.june_reanalysis.b2_cov_check

仕様書は c_t を教師 f_t の共分散として定義しているが、H-cov の文面は
「共通力は Cov(δ,x)（弱学習器の構造的近似誤差）」である。両者は別物なので
**両方**（および E[δ·x] の生の期待値）を v̂ と照合する。

E[g_{W_i}] = 2 v_i E[δ·1[pre_i>0]·x] という厳密な恒等式（nets.py:44-52）があるので、
ゲートを外した E[δ·x] は「全ニューロン共通の力」の第一候補である。
"""
import collections
import os

import numpy as np
import torch

from . import common as C
from . import measure as Ms

STEPS = [100000, 1000000]
N_MC = 100000
CHUNK = 5000


# ------------------------------------------------- 教師の c_t（高精度クロスチェック）

def teacher_cov_condB(ck, device="cpu"):
    """条件B: Monte Carlo n=100,000 と Stein 等式 Σ·E[∇f] の両方で c を計算。"""
    W, b, v, c0 = (ck["teacher"][k] for k in ("W", "b", "v", "c"))
    R, h, d = W.shape
    runs = ck["runs"]
    mu = torch.tensor([[r["c"] / np.sqrt(d)] * d for r in runs], dtype=torch.float32)
    g = torch.Generator(device=device).manual_seed(C.SEED)
    s_y = torch.zeros(R, dtype=torch.float64)
    s_yx = torch.zeros(R, d, dtype=torch.float64)
    s_x = torch.zeros(R, d, dtype=torch.float64)
    s_grad = torch.zeros(R, d, dtype=torch.float64)
    n = 0
    while n < N_MC:
        m = min(CHUNK, N_MC - n)
        x = mu[None] + torch.randn(m, R, d, generator=g)
        pre = torch.einsum("rhd,nrd->nrh", W, x) + b
        y = (torch.relu(pre) * v).sum(-1) + c0
        gate = (pre > 0).float()
        grad = torch.einsum("nrh,rh,rhd->nrd", gate, v, W)     # ∇f(x)
        s_y += y.double().sum(0)
        s_yx += torch.einsum("nr,nrd->rd", y.double(), x.double())
        s_x += x.double().sum(0)
        s_grad += grad.double().sum(0)
        n += m
    Ey, Ex, Eyx = s_y / N_MC, s_x / N_MC, s_yx / N_MC
    c_mc = (Eyx - Ey[:, None] * Ex).numpy()
    c_stein = (s_grad / N_MC).numpy()          # Σ = I なので Σ·E[∇f] = E[∇f]
    return c_mc, c_stein


def teacher_cov_condA(ck, device="cpu"):
    """条件A: 周期内の入力は 5 ビットの U{0,1} のみ（flip 15 ビットは定数）。
    2^5 = 32 通りを**完全列挙**して c を厳密に計算する（MC 誤差ゼロ）。"""
    from src.envs import LTUTarget
    W, b, v, cout, tau = (ck["teacher"][k] for k in ("W", "b", "v", "cout", "tau"))
    R = W.shape[0]
    d = W.shape[2]
    flip = ck["env"]["flip_state"]                    # [R, f]
    f = flip.shape[1]
    nfree = d - f
    bits = torch.tensor([[(k >> j) & 1 for j in range(nfree)] for k in range(2 ** nfree)],
                        dtype=torch.float32)          # [32, nfree]
    x = torch.cat([flip[None].expand(len(bits), -1, -1),
                   bits[:, None, :].expand(-1, R, -1)], dim=2)      # [32,R,d]
    t = LTUTarget.__new__(LTUTarget)
    t.W, t.b, t.v, t.cout, t.tau = W, b, v, cout, tau
    y = t(x)                                                        # [32,R]
    p = 1.0 / len(bits)
    Ey = (y * p).sum(0).double()
    Ex = (x * p).sum(0).double()
    Eyx = torch.einsum("nr,nrd->rd", y.double() * p, x.double())
    return (Eyx - Ey[:, None] * Ex).numpy(), None


def teacher_cov(exp, width, step):
    path = os.path.join(C.SRC_RESULTS, "ckpts", f"{exp}_w{width}_step{step}.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return teacher_cov_condB(ck) if exp == "B" else teacher_cov_condA(ck)


# ------------------------------------------------------------------ 照合

def perp(x, mu):
    m = C.unit(mu)
    return x - (x @ m) * m


def acos(a, b):
    """|cos| と signed cos。"""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-30 or nb < 1e-30:
        return np.nan, np.nan
    s = float(a @ b / (na * nb))
    return abs(s), s


def vhat_from_npz(z, i, mu):
    """B1 と同じ手順で第二共通方向 v̂ を作る（µ は引数で指定 = x_in 基準に統一）。"""
    M, sv = C.get_matrix(z, i, "Eg", alive_only=True)
    if len(M) < 3:
        return None, None
    Mp = M - (M @ C.unit(mu))[:, None] * C.unit(mu)[None, :]
    nrm = np.linalg.norm(Mp, axis=1)
    keep = nrm > 1e-10 * max(np.linalg.norm(M, axis=1).max(), 1e-30)
    if keep.sum() < 3:
        return None, None
    X = C.unit(Mp[keep], axis=1) * sv[keep][:, None]
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[0], float(s[0] ** 2 / max((s ** 2).sum(), 1e-30))


def main():
    runs = C.load_runs()
    rows = collections.defaultdict(list)
    xcheck = {}

    for exp, width in C.GROUPS:
        for step in STEPS:
            m = Ms.get(exp, width, step)
            z = C.load_npz(exp, width, step)
            if m is None or z is None:
                continue
            # 高精度クロスチェック（ckpt 時点の教師、全 run 一括）
            c_hi, c_stein = teacher_cov(exp, width, step)

            for i, rid in enumerate(z["run_ids"]):
                if not z["finite"][i]:
                    continue
                r = runs[rid]
                mu_in = m["mu_in"][i]
                vh, sv1 = vhat_from_npz(z, i, mu_in)
                if vh is None:
                    continue
                d = len(mu_in)
                floor = C.chance_floor(d)

                cand = {
                    "Cov_delta_x": m["Cov_delta_x"][i],       # H-cov の主対象
                    "Edx": m["Edx"][i],                       # E[δ x]（ゲート無しの共通力）
                    "Cov_y_x": m["Cov_y_x"][i],               # 教師 c̄（仕様書 §4 の定義）
                    "Cov_yhat_x": m["Cov_yhat_x"][i],
                    "c_teacher_hi": c_hi[i],                  # 高精度 MC / 厳密列挙
                }
                e = dict(run_id=rid, step=step, sv1_frac=sv1, floor=floor,
                         n_alive=int((~z["dead"][i]).sum()))
                for name, c in cand.items():
                    cp = perp(np.asarray(c, dtype=float), mu_in)
                    a, s = acos(cp, vh)
                    e[f"absCos_{name}_perp_vhat"] = a
                    e[f"cos_{name}_perp_vhat"] = s
                    a2, _ = acos(np.asarray(c, dtype=float), vh)
                    e[f"absCos_{name}_raw_vhat"] = a2
                    e[f"perpfrac_{name}"] = float(
                        np.linalg.norm(cp) / max(np.linalg.norm(c), 1e-30))
                # 窓平均 vs 高精度教師 c の一致（測定の健全性チェック）
                e["absCos_Cov_y_x_vs_hi"] = acos(m["Cov_y_x"][i], c_hi[i])[0]
                if c_stein is not None:
                    e["absCos_stein_vs_mc"] = acos(c_stein[i], c_hi[i])[0]

                # --- period 間のばらつき（H-cov が「静的方向」を作れるか）
                ok = m["period_ok"][i].astype(bool)
                for tag, arr in [("y", m["period_cov_y_x"][i]), ("delta", m["period_cov_delta_x"][i])]:
                    P = arr[ok]
                    if len(P) >= 2:
                        Pp = np.stack([perp(p, mu_in) for p in P])
                        Up = C.unit(Pp, axis=1)
                        G = Up @ Up.T
                        iu = np.triu_indices(len(Up), 1)
                        e[f"period_{tag}_pairwise_signed"] = float(G[iu].mean())
                        e[f"period_{tag}_pairwise_abs"] = float(np.abs(G[iu]).mean())
                        e[f"period_{tag}_n"] = int(len(P))
                        # 各 period の c_t⊥ が v̂ にどれだけ向くか
                        e[f"period_{tag}_absCos_vhat_mean"] = float(np.abs(Up @ vh).mean())
                rows[(step, C.cond_label(r))].append(e)

    KEYS = sorted({k for v in rows.values() for e in v for k in e
                   if isinstance(e[k], float) or isinstance(e[k], int)} - {"run_id"})
    summary = {}
    for (step, cond), lst in sorted(rows.items()):
        summary[f"step{step}|{cond}"] = {
            k: C.agg_seeds([e.get(k, np.nan) for e in lst]) for k in KEYS}
        summary[f"step{step}|{cond}"]["n_runs"] = len(lst)

    C.save_json(dict(steps=STEPS, n_mc=N_MC, summary=summary), "B2", "b2.json")
    make_figs(summary)
    write_verdict(summary)
    return summary


def _sel(summary, step, w100=True):
    out = []
    for k, v in summary.items():
        s, cond = k.split("|")
        if s != f"step{step}":
            continue
        if w100 and "_w100_" not in cond:
            continue
        out.append((cond, v))
    return sorted(out)


def make_figs(summary):
    plt = C.mpl()
    cands = [("Edx", "E[δ·x]  (ungated common force)", "tab:red"),
             ("Cov_delta_x", "Cov(δ, x)  [H-cov]", "tab:orange"),
             ("Cov_y_x", "Cov(y, x)  [teacher, spec §4]", "tab:blue"),
             ("c_teacher_hi", "c_t high-precision (MC/exact)", "tab:cyan")]
    for step in STEPS:
        items = _sel(summary, step)
        if not items:
            continue
        fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(items)), 5))
        x = np.arange(len(items))
        w = 0.8 / len(cands)
        for j, (key, lab, col) in enumerate(cands):
            m = [it[1][f"absCos_{key}_perp_vhat"]["mean"] for it in items]
            sd = [it[1][f"absCos_{key}_perp_vhat"]["std"] for it in items]
            ax.bar(x + (j - (len(cands) - 1) / 2) * w, m, w, yerr=sd, capsize=2,
                   label=lab, color=col)
        fl = items[0][1]["floor"]["mean"]
        ax.axhline(fl, ls=":", color="k", lw=1.2)
        ax.text(0.01, fl + 0.01, f"random floor = {fl:.3f}", fontsize=8,
                transform=ax.get_yaxis_transform())
        ax.axhline(0.4, ls="--", color="green", lw=1)
        ax.text(0.01, 0.41, "spec threshold 0.4", fontsize=8, color="green",
                transform=ax.get_yaxis_transform())
        ax.set_xticks(x)
        ax.set_xticklabels([it[0].replace("_lr0.01", "") for it in items],
                           rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("|cos(candidate⊥, v̂)|")
        ax.set_ylim(0, 1)
        ax.set_title(f"B2: does a covariance direction explain v̂?  step={step:g}\n"
                     f"(width=100, µ̂ = mean of x_in; mean±std over seeds)", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(C.figpath("B2", f"fig_b2_vhat_match_step{step}.png"), dpi=140)
        plt.close(fig)

    # period 間の一貫性
    for step in STEPS:
        items = _sel(summary, step)
        if not items:
            continue
        fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(items)), 4.6))
        x = np.arange(len(items))
        for j, (key, lab, col) in enumerate([("y", "c_t = Cov(y,x)", "tab:blue"),
                                             ("delta", "c_t = Cov(δ,x)", "tab:orange")]):
            k = f"period_{key}_pairwise_abs"
            m = [it[1].get(k, {"mean": np.nan})["mean"] for it in items]
            sd = [it[1].get(k, {"std": np.nan})["std"] for it in items]
            ax.bar(x + (j - 0.5) * 0.38, m, 0.38, yerr=sd, capsize=2, label=lab, color=col)
        ax.axhline(items[0][1]["floor"]["mean"], ls=":", color="k", lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels([it[0].replace("_lr0.01", "") for it in items],
                           rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("mean |cos| between periods")
        ax.set_ylim(0, 1)
        ax.set_title(f"B2: is c_t⊥ static across periods?  step={step:g}", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(C.figpath("B2", f"fig_b2_period_consistency_step{step}.png"), dpi=140)
        plt.close(fig)
    print("  wrote B2 figures")


def write_verdict(summary):
    items = _sel(summary, 1000000)
    L = []
    for key in ["Edx", "Cov_delta_x", "Cov_y_x"]:
        v = np.nanmean([it[1][f"absCos_{key}_perp_vhat"]["mean"] for it in items])
        L.append(f"|cos({key}⊥, v̂)| = {v:.3f}")
    pv = np.nanmean([it[1].get("period_y_pairwise_abs", {"mean": np.nan})["mean"]
                     for it in items])
    pd_ = np.nanmean([it[1].get("period_delta_pairwise_abs", {"mean": np.nan})["mean"]
                      for it in items])
    fl = items[0][1]["floor"]["mean"]
    st = np.nanmean([it[1].get("absCos_stein_vs_mc", {"mean": np.nan})["mean"]
                     for it in items])
    C.verdict("B2", "verdict.txt",
              f"step=1e6, width=100 平均: " + ", ".join(L) +
              f" | period 間 |cos|: Cov(y,x) {pv:.3f}, Cov(δ,x) {pd_:.3f} (床 {fl:.3f})"
              f" | Stein vs MC (cond B) = {st:.4f}")


if __name__ == "__main__":
    main()

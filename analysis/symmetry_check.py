"""mu 符号反転チェック — gamma(1次項) がアーティファクトか力学かの判別。

  python3 analysis/symmetry_check.py results/mu_sweep_norm_signed_0812

入力は src/mu_sweep_norm_analysis.py が書いた norm_sweep_per_run.csv
(発散除外・tail 平均・invD 算出はすべてそちらで済んでいる)。
このスクリプトは src/ を一切変更せず、その出力を読むだけ。

判別:
  (S1) 各 |c| で Delta = invD(-c) - invD(+c)。全 |c| で 95%CI が 0 を含めば対称。
  (S2) 奇成分 odd(|c|) = (invD(+c) - invD(-c))/2 を |c| に原点通過で回帰 -> gamma_odd。
  (S3) 符号込み二次フィット invD ~ 1 + c^2 + c -> gamma は beta と分離同定される。
       正側のみ / 負側のみ の gamma と並べる。正側のみで gamma != 0 なのに
       符号込みで gamma ~ 0 なら、gamma は片側設計の共線性アーティファクト。
CI は全て seed 復元抽出 (各 c セル内で独立、N_BOOT=10000、RNG seed 0)。
本体スクリプトの慣習に合わせ「95%CI が 0 を含むか」で判定する。
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(0)
N_BOOT = 10000
ARMS = ["fixed", "free"]        # fixed が主判定
TOL = 1e-9                      # c の突き合わせ許容


def cells(per):
    """c -> invD 配列 (seed 方向)。"""
    return {float(c): g.invD.values for c, g in per.groupby("c")}


def boot_indices(cell):
    """各 c セル内で seed を復元抽出した invD 辞書を N_BOOT 個生成。"""
    keys = list(cell)
    for _ in range(N_BOOT):
        yield {k: RNG.choice(cell[k], size=len(cell[k]), replace=True) for k in keys}


def ci(samples):
    a = np.asarray([s for s in samples if np.all(np.isfinite(s))])
    if a.size == 0:
        return np.nan, np.nan
    return np.percentile(a, 2.5, axis=0), np.percentile(a, 97.5, axis=0)


def match_pairs(cell):
    """(|c|, +c キー, -c キー) の一覧。|c|>0 のみ。"""
    out = []
    for k in sorted(cell):
        if k <= TOL:
            continue
        neg = [q for q in cell if abs(q + k) < TOL]
        if neg:
            out.append((k, k, neg[0]))
    return out


def quad_signed(cs, ys, ws):
    """invD ~ 1 + c^2 + c の WLS。戻り: (alpha, beta, gamma)。"""
    X = np.stack([np.ones(len(cs)), cs ** 2, cs], axis=1)
    Wr = np.sqrt(ws)[:, None]
    b, *_ = np.linalg.lstsq(X * Wr, ys * np.sqrt(ws), rcond=None)
    return b[0], b[1], b[2]


def cell_stats(cell):
    """c -> (mean, sem, n)。"""
    return {k: (v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan, len(v))
            for k, v in cell.items()}


def fit_from_cell(cell, keys):
    """指定キーの c セル平均で重み付き二次フィット。重みは 1/sem^2。"""
    st = cell_stats({k: cell[k] for k in keys})
    cs = np.array(keys, dtype=float)
    ys = np.array([st[k][0] for k in keys])
    sem = np.array([st[k][1] for k in keys])
    ws = 1.0 / np.clip(sem, 1e-9, None) ** 2
    if len(keys) < 3:
        return (np.nan, np.nan, np.nan)
    return quad_signed(cs, ys, ws)


def odd_slope(cell, pairs):
    """odd(|c|) = (invD(+c)-invD(-c))/2 を |c| に原点通過回帰 -> gamma_odd。"""
    a = np.array([abs(p[0]) for p in pairs])
    o = np.array([(cell[p[1]].mean() - cell[p[2]].mean()) / 2.0 for p in pairs])
    if len(a) == 0 or np.all(a == 0):
        return np.nan
    return float((a @ o) / (a @ a))


def analyze(per, arm):
    cell = cells(per)
    pairs = match_pairs(cell)
    st = cell_stats(cell)
    pos = sorted([k for k in cell if k >= -TOL])
    neg = sorted([k for k in cell if k <= TOL])

    d_pt = np.array([cell[p[2]].mean() - cell[p[1]].mean() for p in pairs])
    g_odd_pt = odd_slope(cell, pairs)
    g_sign_pt = fit_from_cell(cell, sorted(cell))
    g_pos_pt = fit_from_cell(cell, pos)
    g_neg_pt = fit_from_cell(cell, neg)

    d_bs, o_bs, s_bs, p_bs, n_bs = [], [], [], [], []
    for bc in boot_indices(cell):
        d_bs.append([bc[p[2]].mean() - bc[p[1]].mean() for p in pairs])
        o_bs.append(odd_slope(bc, pairs))
        s_bs.append(fit_from_cell(bc, sorted(bc)))
        p_bs.append(fit_from_cell(bc, pos))
        n_bs.append(fit_from_cell(bc, neg))

    d_lo, d_hi = ci(d_bs)
    o_lo, o_hi = ci(o_bs)
    s_lo, s_hi = ci(s_bs)
    p_lo, p_hi = ci(p_bs)
    n_lo, n_hi = ci(n_bs)

    rows = []
    for i, (a, kp, kn) in enumerate(pairs):
        rows.append(dict(absc=a, mean_pos=st[kp][0], sem_pos=st[kp][1], n_pos=st[kp][2],
                         mean_neg=st[kn][0], sem_neg=st[kn][1], n_neg=st[kn][2],
                         delta=d_pt[i], lo=d_lo[i], hi=d_hi[i],
                         contains0=bool(d_lo[i] <= 0 <= d_hi[i])))
    tab = pd.DataFrame(rows)
    return dict(arm=arm, cell=cell, stats=st, pairs=pairs, tab=tab,
                g_odd=(g_odd_pt, o_lo, o_hi),
                g_signed=(g_sign_pt[2], s_lo[2], s_hi[2]),
                b_signed=(g_sign_pt[1], s_lo[1], s_hi[1]),
                a_signed=(g_sign_pt[0], s_lo[0], s_hi[0]),
                g_pos=(g_pos_pt[2], p_lo[2], p_hi[2]),
                g_neg=(g_neg_pt[2], n_lo[2], n_hi[2]),
                all_sym=bool(tab.contains0.all()) if len(tab) else False,
                odd_zero=bool(o_lo <= 0 <= o_hi),
                gsign_zero=bool(s_lo[2] <= 0 <= s_hi[2]))


def fmt(t, nd=4):
    v, lo, hi = t
    return f"{v:.{nd}g} [{lo:.{nd}g}, {hi:.{nd}g}]"


def figure(res, path):
    fig, axes = plt.subplots(1, len(res), figsize=(6 * len(res), 4.4), squeeze=False)
    for ax, r in zip(axes[0], res):
        st, pairs = r["stats"], r["pairs"]
        a = [p[0] for p in pairs]
        ax.errorbar(a, [st[p[1]][0] for p in pairs], yerr=[st[p[1]][1] for p in pairs],
                    marker="o", capsize=3, label="+c")
        ax.errorbar(a, [st[p[2]][0] for p in pairs], yerr=[st[p[2]][1] for p in pairs],
                    marker="s", ls="--", capsize=3, label="-c")
        ax.set_xlabel("|c|"); ax.set_ylabel("1/D*")
        ax.set_title(f"{r['arm']}: invD(+c) vs invD(-c)")
        ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    args = ap.parse_args()
    R = args.outdir.rstrip("/")
    per = pd.read_csv(os.path.join(R, "norm_sweep_per_run.csv"))

    res = [analyze(per[per.norm == arm].reset_index(drop=True), arm)
           for arm in ARMS if (per.norm == arm).any()]
    figdir = os.path.join(R, "figures"); os.makedirs(figdir, exist_ok=True)
    figure(res, os.path.join(figdir, "fig_s1_symmetry.png"))

    L = ["# mu 符号反転チェック (gamma アーティファクト判別)", "",
         f"入力: `{os.path.join(R, 'norm_sweep_per_run.csv')}`  "
         f"(N_BOOT={N_BOOT}, seed 復元抽出, 95%CI)", ""]
    for r in res:
        L += [f"## アーム: {r['arm']}" + ("（主判定）" if r["arm"] == "fixed" else ""), "",
              "### (S1) 各 |c| の Delta = invD(-c) - invD(+c)", "",
              "| \\|c\\| | invD(+c) | invD(-c) | Delta | 95% CI | 0 を含む |",
              "|---|---|---|---|---|---|"]
        for _, x in r["tab"].iterrows():
            L.append(f"| {x.absc:g} | {x.mean_pos:.5g} ± {x.sem_pos:.2g} "
                     f"| {x.mean_neg:.5g} ± {x.sem_neg:.2g} | {x.delta:+.4g} "
                     f"| [{x.lo:+.4g}, {x.hi:+.4g}] | {'YES' if x.contains0 else 'NO'} |")
        L += ["",
              "### (S2)(S3) 奇成分", "",
              "| 量 | 値 | 95% CI |", "|---|---|---|",
              f"| gamma_odd (奇成分の傾き) | {r['g_odd'][0]:.4g} "
              f"| [{r['g_odd'][1]:.4g}, {r['g_odd'][2]:.4g}] |",
              f"| gamma (符号込みフィット) | {r['g_signed'][0]:.4g} "
              f"| [{r['g_signed'][1]:.4g}, {r['g_signed'][2]:.4g}] |",
              f"| gamma (正側のみ) | {r['g_pos'][0]:.4g} "
              f"| [{r['g_pos'][1]:.4g}, {r['g_pos'][2]:.4g}] |",
              f"| gamma (負側のみ) | {r['g_neg'][0]:.4g} "
              f"| [{r['g_neg'][1]:.4g}, {r['g_neg'][2]:.4g}] |",
              f"| beta (符号込みフィット) | {r['b_signed'][0]:.4g} "
              f"| [{r['b_signed'][1]:.4g}, {r['b_signed'][2]:.4g}] |",
              f"| alpha (符号込みフィット) | {r['a_signed'][0]:.4g} "
              f"| [{r['a_signed'][1]:.4g}, {r['a_signed'][2]:.4g}] |",
              "",
              f"S1 全 |c| で対称 (Delta の CI が 0 を含む): "
              f"{'YES' if r['all_sym'] else 'NO'}",
              f"S2 gamma_odd の CI が 0 を含む: {'YES' if r['odd_zero'] else 'NO'}",
              f"S3 符号込み gamma の CI が 0 を含む: {'YES' if r['gsign_zero'] else 'NO'}",
              ""]

    L += ["## 報告用テンプレ（この節を丸ごと貼り付けて報告する）", "", "```",
          "[mu_sweep_norm_signed_0812 対称性チェック]"]
    for r in res:
        L.append(f"{r['arm']}: S1対称={'YES' if r['all_sym'] else 'NO'} "
                 f"S2 gamma_odd={fmt(r['g_odd'])} S3 gamma_signed={fmt(r['g_signed'])} "
                 f"gamma_posonly={fmt(r['g_pos'])} gamma_negonly={fmt(r['g_neg'])} "
                 f"beta_signed={fmt(r['b_signed'])}")
    L += ["```", ""]

    out = os.path.join(R, "symmetry_report.md")
    with open(out, "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote: {out}")


if __name__ == "__main__":
    main()

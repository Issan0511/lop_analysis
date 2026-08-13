"""coupling_ab_0813 (実験(5): Path B → Path A 結合ダイナミクス) の成果物図と順序統計。

  python -m src.figures_coupling results/coupling_ab_0813

出力:
  fig_cp_trend.png     — 4指標 (srank低下度 / 切替直後誤差 / tr C / dead_frac) を
                          正規化して同一時間軸に重ねたマクロトレンド (仕様のメイン図)
  fig_cp_event_<m>.png — タスク境界に整列した平均応答 (学習の早期/中期/後期で色分け)
  coupling_stats.csv   — 各指標の半立ち上がり時刻 t50 (run 別) とブートストラップ順序確率
  stats_summary.md     — 順序検定の要約

指標の対応 (仕様①〜④):
  ① srank(W)   = stable_rank_W (無ければ eff_rank_W)。低下方向なので「低下度」に反転
  ② E[e^2]     = postswitch_err (切替直後 postswitch_n ステップの online 二乗誤差平均)
  ③ tr C(w)    = trC_W (eval バッチのミニバッチ勾配分散、log10 で正規化)
  ④ dead_frac
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MET_LABEL = {"srank_drop": "1) srank(W) drop", "srank_alive_drop": "1b) srank(alive W) drop",
             "post_err": "2) post-switch E[e^2]",
             "trC": "3) tr C(w) (log)", "dead": "4) dead_frac"}
MET_ORDER = ["srank_drop", "srank_alive_drop", "post_err", "trC", "dead"]
MET_COLOR = {"srank_drop": "tab:blue", "srank_alive_drop": "tab:purple",
             "post_err": "tab:orange", "trC": "tab:green", "dead": "tab:red"}


def load_all(resdir):
    runs = pd.read_csv(os.path.join(resdir, "runs.csv")).set_index("run_id")
    # batch 列は int と "full" の混在で dtype が揺れるため文字列に正規化
    runs["batch"] = runs["batch"].astype(str)

    def cat(prefix):
        fs = sorted(glob.glob(os.path.join(resdir, f"{prefix}_*.csv")))
        if not fs:
            return pd.DataFrame()
        df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        return df.join(runs, on="run_id", how="inner")

    return dict(runs=runs, lop=cat("lop_metrics"), post=cat("postswitch_err"))


def cells(lop):
    """パネル単位 = (exp, width, batch, c)。A は c=NaN。batch は文字列 ("1"/"full")。"""
    out = []
    for (exp, width, batch), sub in lop.groupby(["exp", "width", "batch"]):
        if exp == "A":
            out.append((exp, width, batch, np.nan))
        else:
            out += [(exp, width, batch, c) for c in sorted(sub.c.dropna().unique())]
    return out


def cell_mask(df, cell):
    exp, width, batch, c = cell
    m = (df.exp == exp) & (df.width == width) & (df.batch == batch)
    if not np.isnan(c):
        m &= df.c == c
    return m


def cell_title(cell):
    exp, width, batch, c = cell
    return (f"cond {exp}, w={width}" + ("" if np.isnan(c) else f", c={c:g}")
            + ("" if batch == "1" else f", batch={batch}"))


def series_for_run(lop_r, post_r, srank_col):
    """1 run の各指標の (step, 生値) 系列 dict。srank 系は低下 -> 上昇に反転。"""
    lop_r = lop_r.sort_values("step")
    post_r = post_r.sort_values("switch_step")
    out = {
        "srank_drop": (lop_r.step.values, -lop_r[srank_col].values),
        "post_err": (post_r.switch_step.values, post_r.post_err.values),
        "trC": (lop_r.step.values, np.log10(np.maximum(lop_r.trC_W.values, 1e-30))),
        "dead": (lop_r.step.values, lop_r.dead_frac.values),
    }
    if "stable_rank_W_alive" in lop_r.columns:
        out["srank_alive_drop"] = (lop_r.step.values, -lop_r.stable_rank_W_alive.values)
    return out


def smooth(y, k=5):
    if len(y) < k:
        return y
    return pd.Series(y).rolling(k, center=True, min_periods=1).mean().values


def norm01(y):
    lo, hi = np.nanmin(y), np.nanmax(y)
    if not np.isfinite(hi - lo) or hi - lo < 1e-12:
        return np.full_like(y, np.nan)
    return (y - lo) / (hi - lo)


def t_half(x, y, min_change):
    """半立ち上がり時刻: smoothed y が 初期値 + 0.5*(最大-初期) を初めて超える step。
    変化量が min_change (生値スケール) 未満なら NaN (立ち上がり無しと判定)。"""
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(y) < 5:
        return np.nan
    ys = smooth(y)
    y0 = np.nanmean(ys[:3])
    ymax = np.nanmax(ys)
    if ymax - y0 < min_change:
        return np.nan
    thr = y0 + 0.5 * (ymax - y0)
    idx = np.argmax(ys >= thr)
    return float(x[idx]) if ys[idx] >= thr else np.nan


# 生値スケールでの最小変化量 (これ未満は「その Path が発現していない」として除外)
MIN_CHANGE = {"srank_drop": 0.5, "srank_alive_drop": 0.5,
              "post_err": 0.05, "trC": 0.3, "dead": 0.05}


def fig_trend(d, srank_col, figdir):
    lop, post = d["lop"], d["post"]
    cl = cells(lop)
    ncol = 2
    nrow = int(np.ceil(len(cl) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 3.6 * nrow), squeeze=False)
    for k, cell in enumerate(cl):
        ax = axes[k // ncol][k % ncol]
        lsub, psub = lop[cell_mask(lop, cell)], post[cell_mask(post, cell)]
        for met in MET_ORDER:
            xs_all, ys_all = [], []
            for rid, lr in lsub.groupby("run_id"):
                pr = psub[psub.run_id == rid] if "run_id" in psub.columns else psub.iloc[0:0]
                ss = series_for_run(lr, pr, srank_col)
                if met not in ss:
                    continue
                x, y = ss[met]
                if len(x):
                    xs_all.append(pd.Series(y, index=x))
            if not xs_all:
                continue
            mean = pd.concat(xs_all, axis=1).mean(axis=1)   # seed 平均 (step で整列)
            ax.plot(mean.index, norm01(smooth(mean.values)), lw=1.4,
                    color=MET_COLOR[met], label=MET_LABEL[met])
        ax.set_title(cell_title(cell))
        ax.set_xlabel("step")
        ax.set_ylabel("normalized [0,1]")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=7)
    for k in range(len(cl), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Path B → Path A coupling: 4 indicators on one time axis "
                 "(seed-mean, min-max normalized; trC log10)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_cp_trend.png"), dpi=150)
    plt.close(fig)


def fig_event(d, cfg_coupling, period, figdir):
    """タスク境界整列平均。切替 s0 を学習の早期/中期/後期の3相に分け、
    相ごとに offset (step - s0) 平均を描く (境界応答が学習進行でどう変わるか)。"""
    lop = d["lop"]
    pre, post_w = cfg_coupling.get("pre_window", 0), cfg_coupling.get("post_window", 0)
    if not (pre or post_w):
        return
    lop = lop.copy()
    s0 = (lop.step / period).round() * period
    lop["offset"] = lop.step - s0
    lop["s0"] = s0
    win = lop[(lop.offset >= -pre) & (lop.offset <= post_w) & (lop.s0 > 0)]
    total = lop.step.max()
    phase = pd.cut(win.s0, [0, total / 3, 2 * total / 3, total + 1],
                   labels=["early", "mid", "late"])
    win = win.assign(phase=phase)

    for met, col, tf in [("trC", "trC_W", lambda v: np.log10(np.maximum(v, 1e-30))),
                         ("dead", "dead_frac", lambda v: v),
                         ("srank", None, lambda v: v),
                         ("srank_alive", "stable_rank_W_alive", lambda v: v)]:
        if col is not None and col not in lop.columns:
            continue
        cl = cells(lop)
        ncol = 2
        nrow = int(np.ceil(len(cl) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 3.4 * nrow), squeeze=False)
        for k, cell in enumerate(cl):
            ax = axes[k // ncol][k % ncol]
            sub = win[cell_mask(win, cell)]
            use_col = col or ("stable_rank_W" if "stable_rank_W" in sub.columns else "eff_rank_W")
            for ph, color in [("early", "tab:blue"), ("mid", "tab:orange"), ("late", "tab:red")]:
                s = sub[sub.phase == ph]
                if s.empty:
                    continue
                v = tf(s[use_col].values)
                g = pd.DataFrame({"offset": s.offset.values, "v": v}) \
                    .groupby("offset")["v"]
                base = g.mean()
                pre_mean = base[base.index <= 0].mean()   # 切替前基準からの偏差
                ax.plot(base.index, base.values - pre_mean, marker="o", ms=2.5,
                        lw=1.2, color=color, label=ph)
            ax.axvline(0, color="gray", lw=0.8, ls="--")
            ax.set_title(cell_title(cell))
            ax.set_xlabel("step - switch")
            ax.grid(alpha=0.3)
            if k == 0:
                ax.set_ylabel(f"Δ{met} (vs pre-switch)")
                ax.legend(fontsize=7)
        for k in range(len(cl), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        fig.suptitle(f"event-aligned {met} around task switches (mean over switches+seeds)")
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, f"fig_cp_event_{met}.png"), dpi=150)
        plt.close(fig)


def order_stats(d, srank_col, resdir, n_boot=4000, rng_seed=0):
    """run 別 t50 → セル別に**全ペア**の順序を seed ブートストラップで検定
    (隣接ペアのみの連鎖推論は post_err のような非単調指標を挟むと壊れるため)。
    P(t50_X < t50_Y) と平均ラグを出す。"""
    from itertools import combinations
    lop, post = d["lop"], d["post"]
    rng = np.random.default_rng(rng_seed)
    rows, pair_rows = [], []
    for cell in cells(lop):
        lsub, psub = lop[cell_mask(lop, cell)], post[cell_mask(post, cell)]
        mets = [m for m in MET_ORDER
                if m != "srank_alive_drop" or "stable_rank_W_alive" in lop.columns]
        t50 = {m: {} for m in mets}
        for rid, lr in lsub.groupby("run_id"):
            pr = psub[psub.run_id == rid] if len(psub) else psub
            ss = series_for_run(lr, pr, srank_col)
            for met in mets:
                x, y = ss[met]
                t = t_half(np.asarray(x, float), np.asarray(y, float), MIN_CHANGE[met])
                t50[met][rid] = t
                rows.append(dict(exp=cell[0], width=cell[1], batch=cell[2], c=cell[3],
                                 run_id=rid, metric=met, t50=t))
        for a, b in combinations(mets, 2):
            ids = [r for r in t50[a] if np.isfinite(t50[a][r]) and np.isfinite(t50[b].get(r, np.nan))]
            if len(ids) < 3:
                pair_rows.append(dict(exp=cell[0], width=cell[1], batch=cell[2], c=cell[3],
                                      pair=f"{a}<{b}",
                                      n=len(ids), p_order=np.nan, lag_mean=np.nan,
                                      lag_lo=np.nan, lag_hi=np.nan))
                continue
            la = np.array([t50[a][r] for r in ids])
            lb = np.array([t50[b][r] for r in ids])
            diff = lb - la
            bs = rng.choice(len(ids), (n_boot, len(ids)), replace=True)
            bmean = diff[bs].mean(axis=1)
            pair_rows.append(dict(exp=cell[0], width=cell[1], batch=cell[2], c=cell[3],
                                  pair=f"{a}<{b}",
                                  n=len(ids), p_order=float((bmean > 0).mean()),
                                  lag_mean=float(diff.mean()),
                                  lag_lo=float(np.quantile(bmean, 0.025)),
                                  lag_hi=float(np.quantile(bmean, 0.975))))
    pd.DataFrame(rows).to_csv(os.path.join(resdir, "coupling_stats.csv"), index=False)
    pdf = pd.DataFrame(pair_rows)
    with open(os.path.join(resdir, "stats_summary.md"), "w") as fh:
        fh.write("# coupling_ab 順序検定 (t50 = 半立ち上がり時刻, seed bootstrap)\n\n")
        fh.write("pair X<Y: 仕様の因果順で X が Y に先行するか。p_order = P(lag>0), "
                 "lag = t50_Y - t50_X (step)。n は両指標が発現した seed 数。\n\n")
        fh.write(pdf.to_string(index=False))
        fh.write("\n")
    return pdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    args = ap.parse_args()
    figdir = os.path.join(args.results, "figures")
    os.makedirs(figdir, exist_ok=True)
    d = load_all(args.results)
    srank_col = "stable_rank_W" if "stable_rank_W" in d["lop"].columns else "eff_rank_W"

    import yaml
    with open(os.path.join(args.results, "config_used.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    period = int(d["runs"].period.iloc[0])

    fig_trend(d, srank_col, figdir)
    fig_event(d, cfg.get("coupling", {}), period, figdir)
    pdf = order_stats(d, srank_col, args.results)
    print(pdf.to_string(index=False))
    print(f"figures -> {figdir}")


if __name__ == "__main__":
    main()

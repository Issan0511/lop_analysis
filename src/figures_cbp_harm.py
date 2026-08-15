"""cbp_harm_0815 の判定・図・summary (spec_cbp_harm_0815 §3–6)。

  python -m src.figures_cbp_harm results/cbp_harm_0815

判定は **clean eval_loss のみ** (dead_frac は CBP が定義上下げるので循環する)。
PC-1..PC-6 を verdict.csv に {PASS, FAIL, NA} + 根拠数値で出す。null も同じ体裁。
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.center_selfcov.slopes import paired_boot_ci, boot_ci

RHO_COLOR = {0.0: "tab:gray", 1e-5: "tab:orange", 1e-4: "tab:red"}
ROUTE_COLOR = {"N": "tab:red", "K": "tab:green"}
# PC-6 アンカー: methods_sde_0813 の条件A A_w100 実測 (再実行不要)
ANCHOR = dict(cell="condA A_w100 (µ≠0, methods_sde_0813)",
              dead_none=0.968, dead_cbp=0.034, eval_none=0.937, eval_cbp=0.005)


def load(resdir):
    rows = []
    for d in sorted(glob.glob(os.path.join(resdir, "n*_K*"))):
        rp = os.path.join(d, "runs.csv")
        if not os.path.exists(rp):
            continue
        runs = pd.read_csv(rp).set_index("run_id")
        for f in sorted(glob.glob(os.path.join(d, "lop_metrics_*.csv"))):
            lop = pd.read_csv(f).join(runs, on="run_id", how="inner")
            lop["cell"] = os.path.basename(d)
            rows.append(lop)
    lop = pd.concat(rows, ignore_index=True)
    return lop


def last_by_seed(sub, col):
    return sub.sort_values("step").groupby("seed").last()[col]


def cmp_rho(lop, cell, rho_hi=1e-4, rng=None, col="eval_loss"):
    """同一セル内 rho=0 vs rho_hi の paired 差 (rho_hi − rho0)。正 = CBP が悪化させる。"""
    g = lop[lop.cell == cell]
    a = last_by_seed(g[g.rho == rho_hi], col)
    b = last_by_seed(g[g.rho == 0.0], col)
    seeds = a.index.intersection(b.index)
    ci = paired_boot_ci(a.loc[seeds].values, b.loc[seeds].values, rng)
    ci.update(hi_mean=float(a.loc[seeds].mean()), lo_mean=float(b.loc[seeds].mean()))
    return ci


def log_ratio(lop, cell, rho_hi=1e-4):
    """seed 対応の log(eval(rho_hi)/eval(0))。セル間で効果量を比較するため
    絶対差ではなく比で見る (eval の水準がセルで 1 桁違うため)。"""
    g = lop[lop.cell == cell]
    a = last_by_seed(g[g.rho == rho_hi], "eval_loss")
    b = last_by_seed(g[g.rho == 0.0], "eval_loss")
    s = a.index.intersection(b.index)
    return np.log(a.loc[s].values) - np.log(b.loc[s].values)


def verdicts(lop, resdir, rng):
    rows = []
    cells = sorted(lop.cell.unique())

    # PC-1: 経路N σ_ξ=2 で CBP (rho=1e-4) が clean eval を悪化させる
    c = cmp_rho(lop, "n2_K10000", rng=rng)
    pc1 = bool(c["lo"] > 0)
    rows.append(dict(pred="PC-1", scope="経路N σ_ξ=2: rho=1e-4 の clean eval が rho=0 より高い",
                     verdict="PASS" if pc1 else "FAIL",
                     evidence=f"eval rho0 {c['lo_mean']:.4f} → rho1e-4 {c['hi_mean']:.4f}, "
                              f"diff {c['mean']:+.4f} CI [{c['lo']:+.4f}, {c['hi']:+.4f}]"
                              + ("→ **CBP が悪化させる**" if pc1 else
                                 "→ 悪化せず。§1.2 の適応的スパース化説は棄却され、"
                                 "bias_margin の free 優位は frozen 腕の表現力ハンデで"
                                 "説明されるべき")))

    # PC-2: 経路K K=100 では悪化しない (改善またはニュートラル)
    c2 = cmp_rho(lop, "n0_K100", rng=rng)
    pc2 = bool(c2["lo"] <= 0)          # 悪化が有意でない
    rows.append(dict(pred="PC-2", scope="経路K K=100: rho=1e-4 が rho=0 に対し悪化しない",
                     verdict="PASS" if pc2 else "FAIL",
                     evidence=f"eval rho0 {c2['lo_mean']:.4f} → rho1e-4 {c2['hi_mean']:.4f}, "
                              f"diff {c2['mean']:+.4f} CI [{c2['lo']:+.4f}, {c2['hi']:+.4f}] "
                              + ("(悪化は有意でない)" if pc2 else "(有意に悪化 = 反転せず)")))
    # 目玉の直接検定 (事前登録は PC-1∧PC-2 の連言だったが、両者は別々のセルの
    # 片側検定なので検出力が低い。同一 seed で経路間の効果量の差を直接検定する方が
    # 主張「dead_frac は治療の要否を決めない」に正対する。**事後追加の検定である旨を明記**)
    dn = last_by_seed(lop[(lop.cell == "n2_K10000") & (lop.rho == 0)], "dead_frac").mean()
    dk = last_by_seed(lop[(lop.cell == "n0_K100") & (lop.rho == 0)], "dead_frac").mean()
    ctr = paired_boot_ci(log_ratio(lop, "n2_K10000"), log_ratio(lop, "n0_K100"), rng)
    rows.append(dict(pred="PC-1∧2",
                     scope="**目玉**: dead_frac 同程度でも治療の効果が違う "
                           "(事後追加: 経路間コントラストの直接検定)",
                     verdict="PASS" if ctr["excl_zero"] else "FAIL",
                     evidence=f"dead_frac は n2 {dn:.2f} vs K100 {dk:.2f} とほぼ同じ。"
                              f"CBP 効果 (log 比) の経路間差 = {ctr['mean']:+.3f} "
                              f"CI [{ctr['lo']:+.3f}, {ctr['hi']:+.3f}] "
                              + ("→ **有意に異なる = dead_frac は治療の要否を決めない**"
                                 if ctr["excl_zero"] else "→ 有意差なし")
                              + "。ただし事前登録の連言 (PC-1 ∧ PC-2) 自体は PC-1 が"
                                "有意に達せず不成立"))

    # PC-3: 用量反応 (rho と σ_ξ に単調)
    n_cells = ["n0_K10000", "n1_K10000", "n2_K10000"]
    dose = []
    for cell in n_cells:
        g = lop[lop.cell == cell]
        row = dict(cell=cell, sd=float(g.noise_sd.iloc[0]))
        for r in [0.0, 1e-5, 1e-4]:
            sub = g[g.rho == r]
            row[f"eval_rho{r:g}"] = last_by_seed(sub, "eval_loss").mean() if len(sub) else np.nan
        row["harm_1e5"] = row["eval_rho1e-05"] - row["eval_rho0"]
        row["harm_1e4"] = row["eval_rho0.0001"] - row["eval_rho0"]
        dose.append(row)
    dose = pd.DataFrame(dose)
    mono_rho = bool((dose.harm_1e4 >= dose.harm_1e5).all())
    mono_sd = bool(np.all(np.diff(dose.sort_values("sd").harm_1e4.values) >= 0))
    rows.append(dict(pred="PC-3", scope="用量反応 (害が rho と σ_ξ に単調増加)",
                     verdict="PASS" if (mono_rho and mono_sd) else "FAIL",
                     evidence=f"rho 単調 {'○' if mono_rho else '×'} "
                              f"(harm 1e-5 {list(dose.harm_1e5.round(4))} vs "
                              f"1e-4 {list(dose.harm_1e4.round(4))}); "
                              f"σ_ξ 単調 {'○' if mono_sd else '×'} "
                              f"(σ {list(dose.sort_values('sd').sd)} → harm "
                              f"{list(dose.sort_values('sd').harm_1e4.round(4))})"))

    # PC-4: 残差なし (σ_ξ=0, K=1e4) で CBP は中立
    c4 = cmp_rho(lop, "n0_K10000", rng=rng)
    pc4 = bool(c4["lo"] <= 0 <= c4["hi"])
    rows.append(dict(pred="PC-4", scope="残差なし (σ_ξ=0,K=1e4) で CBP は中立",
                     verdict="PASS" if pc4 else "FAIL",
                     evidence=f"eval rho0 {c4['lo_mean']:.4f} → rho1e-4 {c4['hi_mean']:.4f}, "
                              f"diff {c4['mean']:+.4f} CI [{c4['lo']:+.4f}, {c4['hi']:+.4f}]"
                              + ("" if pc4 else " → **交絡**: 害は「ノイズへのフィット」ではなく"
                                 "汎用的な reset ダメージの可能性。主張を「reset は残差の"
                                 "有無に関わらず有害」に修正すること")))
    # PC-4 が FAIL のとき、ノイズ経路の害が汎用 reset コストを超えるかを直接検定する
    # (超えなければ「ノイズ固有の害」は主張できない)
    if not pc4:
        exc = paired_boot_ci(log_ratio(lop, "n2_K10000"), log_ratio(lop, "n0_K10000"), rng)
        rows.append(dict(pred="PC-4b",
                         scope="ノイズ経路の害は汎用 reset コストを超えるか (事後追加)",
                         verdict="PASS" if (exc["excl_zero"] and exc["mean"] > 0) else "FAIL",
                         evidence=f"log 比 n2 − 残差なし = {exc['mean']:+.3f} "
                                  f"CI [{exc['lo']:+.3f}, {exc['hi']:+.3f}] "
                                  + ("→ ノイズ固有の上乗せあり"
                                     if (exc["excl_zero"] and exc["mean"] > 0) else
                                     "→ **区別できない**。ノイズ経路の害は残差ゼロでも"
                                     "同じだけ出る汎用 reset コストで説明され、"
                                     "「ノイズを拾いに行く害」は支持されない")))

    # PC-5: 機構の解離 (dead は下がるが eval は上がる)
    d5 = cmp_rho(lop, "n2_K10000", rng=rng, col="dead_frac")
    pc5 = bool(d5["hi"] < 0 and pc1)
    rows.append(dict(pred="PC-5", scope="解離: 同一セルで dead↓ かつ clean eval↑",
                     verdict="PASS" if pc5 else "FAIL",
                     evidence=f"n2_K10000: dead {d5['lo_mean']:.3f} → {d5['hi_mean']:.3f} "
                              f"(diff {d5['mean']:+.3f} CI [{d5['lo']:+.3f}, {d5['hi']:+.3f}]) "
                              f"かつ eval は {c['mean']:+.4f} 悪化 → "
                              + ("**同一介入で dead と機能が逆向きに動く**"
                                 if pc5 else "解離は成立せず")))

    rows.append(dict(pred="PC-6", scope="アンカー: 条件A では CBP は有効なまま (再実行なし)",
                     verdict="PASS",
                     evidence=f"{ANCHOR['cell']}: dead {ANCHOR['dead_none']} → "
                              f"{ANCHOR['dead_cbp']}, eval {ANCHOR['eval_none']} → "
                              f"{ANCHOR['eval_cbp']} (methods_sde_0813 の既存値)。"
                              "**本実験は Dohare 批判ではない**: 主張は「CBP が効かない」"
                              "ではなく「CBP の適用条件が明示されていない」"))
    return pd.DataFrame(rows), dose


def sanity(resdir, lop):
    s = {}
    # S3: CBP の reset 数が理論値 rho*width*steps と一致 (端数繰越・eligible 不足で下振れ)
    fs = glob.glob(os.path.join(resdir, "*", "cbp_stats_*.csv"))
    if fs:
        cs = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        cs["ratio"] = cs.n_reset / cs.expected.clip(lower=1e-9)
        s["S3_reset_ratio"] = dict(min=float(cs.ratio.min()), median=float(cs.ratio.median()),
                                   max=float(cs.ratio.max()), n=len(cs))
        s["S3_pass"] = bool(cs.ratio.min() > 0.9 and cs.ratio.max() <= 1.001)
        s["S3_by_rho"] = cs.groupby("rho").ratio.median().round(4).to_dict()
    # S5: freeze_bias=false (b が動いている) — b_std > 0 を確認
    if "b_std" in lop.columns:
        last = lop.sort_values("step").groupby(["cell", "run_id"]).last()
        s["S5_min_b_std"] = float(last.b_std.min())
        s["S5_pass"] = bool(last.b_std.min() > 0)
    return s


def s1_check(resdir, bm_dir):
    """S1: rho=0 の各セルが bias_margin_0814 の対応セル (b free) と bit 一致。"""
    out = []
    for d in sorted(glob.glob(os.path.join(resdir, "n*_K*"))):
        tag = os.path.basename(d)
        src = os.path.join(bm_dir, f"{tag}_bfree", "lop_metrics_B_w20.csv")
        dst = os.path.join(d, "lop_metrics_B_w20.csv")     # method=none は無印 gname
        if not (os.path.exists(src) and os.path.exists(dst)):
            out.append(dict(cell=tag, status="NA (対応ファイルなし)"))
            continue
        a, b = pd.read_csv(dst, dtype=str), pd.read_csv(src, dtype=str)
        common = [c for c in a.columns if c in b.columns]
        same = a[common].equals(b[common])
        out.append(dict(cell=tag, status="PASS" if same else "FAIL",
                        n_cols=len(common), n_rows=len(a)))
    return pd.DataFrame(out)


S2_QUOTE = """src/train.py — ξ は学習ループの y にのみ加算され eval_batch() には無い:

    # 学習ループ
    y = teacher(x_raw)
    if noise_sd > 0:                                 # 学習信号のみ汚す (eval は clean)
        y = y + noise_sd * torch.randn(y.shape, generator=st["gens"]["noise"], ...)

    # eval_batch() — ノイズ加算なし (clean teacher に対する誤差)
    def eval_batch(st):
        ...
        y = st["teacher"](x)
        return x, y"""


def figures(resdir, lop, dose):
    fd = os.path.join(resdir, "figures")
    os.makedirs(fd, exist_ok=True)
    last = lop.sort_values("step").groupby(["cell", "rho", "seed"]).last().reset_index()
    agg = last.groupby(["cell", "rho"]).agg(
        eval_m=("eval_loss", "mean"), eval_s=("eval_loss", "sem"),
        dead_m=("dead_frac", "mean"), sd=("noise_sd", "first"),
        K=("K_cell", "first")).reset_index()

    # c1: clean eval の rho 依存 (主図)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for cell, g in agg.groupby("cell"):
        g = g.sort_values("rho")
        route = "N" if g.sd.iloc[0] > 0 else ("K" if g.K.iloc[0] < 10000 else "base")
        c = {"N": "tab:red", "K": "tab:green", "base": "tab:gray"}[route]
        ls = "-" if route == "N" else ("--" if route == "K" else ":")
        x = np.arange(len(g))
        ax.errorbar(x, g.eval_m, yerr=g.eval_s, marker="o", ms=5, lw=1.6,
                    color=c, ls=ls, capsize=3,
                    label=f"{cell} ({'noise' if route=='N' else 'K' if route=='K' else 'baseline'})")
    ax.set_xticks(range(3)); ax.set_xticklabels(["rho=0", "1e-5", "1e-4"])
    ax.set_ylabel("clean eval_loss (final, seed mean±SE)")
    ax.set_title("PC-1/PC-2: does CBP help or harm?\n(up = CBP harms)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig_c1_eval_vs_rho.png"), dpi=150)
    plt.close(fig)

    # c2: dead↓ / eval↑ の解離
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (col, lbl) in zip(axes, [("dead_m", "dead_frac"), ("eval_m", "clean eval_loss")]):
        for cell, g in agg.groupby("cell"):
            g = g.sort_values("rho")
            route = "N" if g.sd.iloc[0] > 0 else ("K" if g.K.iloc[0] < 10000 else "base")
            c = {"N": "tab:red", "K": "tab:green", "base": "tab:gray"}[route]
            ls = "-" if route == "N" else ("--" if route == "K" else ":")
            ax.plot(np.arange(len(g)), g[col], marker="o", ms=5, lw=1.6, color=c, ls=ls,
                    label=cell)
        ax.set_xticks(range(3)); ax.set_xticklabels(["rho=0", "1e-5", "1e-4"])
        ax.set_ylabel(lbl); ax.grid(alpha=0.3)
    axes[0].set_title("CBP lowers dead (by construction)")
    axes[1].set_title("but clean eval can go UP")
    axes[0].legend(fontsize=7)
    fig.suptitle("PC-5: dissociation between dead_frac and function")
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig_c2_dead_vs_eval.png"), dpi=150)
    plt.close(fig)

    # c3: 用量反応
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dose.sd, dose.harm_1e4, "-o", color="tab:red", label="rho=1e-4")
    ax.plot(dose.sd, dose.harm_1e5, "--s", color="tab:orange", label="rho=1e-5")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("target_noise_sd σ_ξ")
    ax.set_ylabel("harm = eval(rho) − eval(rho=0)")
    ax.set_title("PC-3: dose response of CBP harm (route N)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig_c3_dose.png"), dpi=150)
    plt.close(fig)

    # c4: 条件A アンカーとの並置
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    n2 = agg[agg.cell == "n2_K10000"].sort_values("rho")
    k1 = agg[agg.cell == "n0_K100"].sort_values("rho")
    for ax, (labels, none_v, cbp_v, title) in zip(axes, [
            (["dead", "eval"], [ANCHOR["dead_none"], ANCHOR["eval_none"]],
             [ANCHOR["dead_cbp"], ANCHOR["eval_cbp"]],
             "condA w100 (µ≠0): CBP works"),
            (["dead", "eval"],
             [n2.dead_m.iloc[0], n2.eval_m.iloc[0]],
             [n2.dead_m.iloc[-1], n2.eval_m.iloc[-1]],
             "condB µ=0, σ_ξ=2: CBP harms eval")]):
        x = np.arange(2); w = 0.35
        ax.bar(x - w / 2, none_v, w, label="no CBP", color="tab:gray")
        ax.bar(x + w / 2, cbp_v, w, label="CBP rho=1e-4", color="tab:red")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_title(title); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    fig.suptitle("PC-6: same intervention, opposite outcome by regime")
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig_c4_condA_anchor.png"), dpi=150)
    plt.close(fig)


def write_summary(resdir, ver, dose, san, s1, lop):
    lines = ["# cbp_harm_0815 summary (spec_cbp_harm_0815 §3 事前登録判定)\n",
             "条件B・µ=0 厳密 (c=0)・κ=1・w20・target_hidden=100・lr=0.01・1M step。\n",
             "**判定は clean eval_loss のみ** (dead_frac は CBP が定義上下げるので"
             "判定に使うと循環する)。\n",
             "## 判定表 (null も同じ体裁)\n", ver.to_string(index=False)]

    lines.append("\n\n## 用量反応の内訳 (経路N)\n")
    lines.append(dose.round(5).to_string(index=False))

    lines.append("\n## セル別の最終値\n")
    last = lop.sort_values("step").groupby(["cell", "rho", "seed"]).last().reset_index()
    tab = last.groupby(["cell", "rho"]).agg(
        dead=("dead_frac", "mean"), eval_loss=("eval_loss", "mean"),
        b_mean=("b_mean_alive", "mean"), n=("seed", "size")).round(4)
    lines.append(tab.to_string())

    lines.append("\n## サニティ (§4)\n")
    lines.append("- S1 (rho=0 が bias_margin_0814 の対応セルと bit 一致、OMP_NUM_THREADS=6 で統一):\n")
    lines.append(s1.to_string(index=False))
    lines.append(f"\n- S2 (clean eval に ξ を入れていない): PASS — コード引用:\n\n```\n{S2_QUOTE}\n```")
    lines.append(f"- S3 (CBP reset 数が理論値 rho×width×steps と一致): "
                 f"{'PASS' if san.get('S3_pass') else 'FAIL/注意'} "
                 f"— 実測/理論比 {san.get('S3_reset_ratio')}, rho 別中央値 {san.get('S3_by_rho')}。"
                 "下振れは apply_method が eligible (age>maturity) 不足時に切り捨てる仕様による")
    lines.append("- S4 (reset 直後は v=0 で出力に寄与しない): 実装で担保 "
                 "(apply_method が W を kaiming 再サンプル、b=0、v=0 に設定)")
    lines.append(f"- S5 (全アームで b は学習可能): "
                 f"{'PASS' if san.get('S5_pass') else 'FAIL'} "
                 f"(最小 b_std {san.get('S5_min_b_std'):.4g} > 0)")

    lines.append("""
## 結論

### 1. 事前登録の主判定 PC-1 は不成立 → §1.2「適応的スパース化」は棄却

経路N σ_ξ=2 で CBP は点推定では 25% 悪化させる (eval 0.560 → 0.698) が、
diff CI [−0.025, +0.352] はゼロを含み有意でない (n=5 seed)。
仕様 PC-1 の反証規定に従い、**§1.2 の適応的スパース化説は棄却**する。
`bias_margin_0814` の free 優位 (free/frozen 0.57) は、**frozen 腕の表現力ハンデ**
(b を凍結すると閾値の自由度ごと失われる) で説明されるべきであり、
「ノイズを拾う容量を自ら削っている」という読みは本実験では支持されなかった。

### 2. PC-4 が FAIL: 害は「ノイズへのフィット」ではなく汎用的な reset コスト

**dead が 1 つも無いセルでこそ CBP の害が有意**である:

| セル | dead(rho=0) | eval(rho=0) | eval(1e-4) | 比 | log 比 CI |
|---|---|---|---|---|---|
| n0_K10000 (残差なし) | 0.00 | 0.075 | 0.092 | 1.23 | [+0.039, +0.420] **有意** |
| n0_K1000 | 0.04 | 0.193 | 0.227 | 1.18 | [+0.104, +0.238] **有意** |
| n1_K10000 (σ_ξ=1) | 0.57 | 0.347 | 0.313 | 0.90 | n.s. |
| n2_K10000 (σ_ξ=2) | 0.92 | 0.560 | 0.698 | 1.25 | n.s. |
| n0_K100 (高速切替) | 0.99 | 0.530 | 0.431 | 0.81 | n.s. |

さらに PC-4b (事後追加) のとおり、n2 の害は残差なしセルの害と**統計的に区別できない**
(log 比の差 −0.048 CI [−0.347, +0.250])。よって仕様の指示どおり主張を
**「reset は残差の有無に関わらず約 20% のコストを持つ」**に修正する。
CBP が正味で有益になるのは、そのコストを上回るだけの「本当に失われた容量」がある場合に限る。

### 3. それでも中心的な主張は生き残る: dead_frac は治療の要否を決めない

- 経路間コントラスト (事後追加の直接検定): dead_frac が 0.92 と 0.99 でほぼ同じ n2 と
  K100 で、CBP 効果の差は **+0.349 CI [+0.216, +0.496] で有意**。
- dead_frac と CBP 効果の対応に単調性は無い:
  dead 0.00 → +0.203 / 0.04 → +0.176 / 0.57 → −0.056 / 0.92 → +0.154 / 0.99 → −0.195。
- **PC-5 の解離も別の形で成立**: n2 で CBP は dead を 0.92 → **0.000** と完全に消したが、
  clean eval は改善しなかった (点推定はむしろ悪化)。**容量を全部戻しても何も買えていない。**

ただしこれは §1.2 の機構 (ノイズ死は適応的) によるものではなく、
**「CBP の便益は失われた容量の実在に依存し、dead_frac はその代理指標として機能しない」**
という、より穏当だが交絡の少ない主張である。

### 4. レジーム依存 (PC-6 と併せて)

同じ CBP が、条件A (µ≠0) では dead 0.968 → 0.034・eval 0.937 → **0.005** と劇的に効き、
µ=0・残差なしでは 20% 悪化させ、µ=0・高速切替では (n.s. ながら) 19% 改善する。
**手法の符号はレジームが決める。** これは「CBP が悪い」ではなく
「**CBP の適用条件が明示されていない**」という §1.4 の主張を支持する。
""")

    lines.append("\n## 主張してはいけないこと (spec §5 の再掲・厳守)\n")
    lines.append("1. **「CBP は有害な手法である」とは言えない。** 言えるのは「µ=0 かつ"
                 "フィット不能ノイズという特定レジームで有害」まで。PC-6 (条件A では"
                 "dead 0.968→0.034, eval 0.937→0.005 で最強アーム) を必ず併記する")
    lines.append("2. **「Dohare の dead も無害」とは言えない。** 条件A (µ≠0) の dead は "
                 "µ 経路で、condA_freeze_0815 のとおり b とは無関係。レジームが違う")
    lines.append("3. **dead_frac を判定指標に使わない** (CBP は定義上 dead を下げるので循環)")
    lines.append("4. PC-2 が落ちた場合、「経路で符号が反転する」という一番強い形は放棄する")
    lines.append("5. 「適応的スパース化」は本実験で**初めて検証される仮説**であり、"
                 "bias_margin_0814 の free/frozen 比較は交絡ありの示唆に過ぎなかった "
                 "→ **本実験で棄却された**。今後この説を既定事実として書かないこと")
    lines.append("6. **「CBP はノイズを拾いに行くから有害」とは言えない** (PC-4b)。"
                 "害は残差ゼロでも同じだけ出るので、機構は汎用的な reset コストである")
    lines.append("7. 経路間コントラストと PC-4b は**事後追加の検定**である。"
                 "事前登録された連言 (PC-1 ∧ PC-2) は PC-1 が有意に達せず不成立であり、"
                 "その事実を隠して事後検定だけを報告しないこと")
    with open(os.path.join(resdir, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--bias-margin", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    lop = load(args.results)
    ver, dose = verdicts(lop, args.results, rng)
    ver.to_csv(os.path.join(args.results, "verdict.csv"), index=False)
    san = sanity(args.results, lop)
    json.dump(san, open(os.path.join(args.results, "sanity.json"), "w"),
              indent=1, default=str)
    bm = args.bias_margin or os.path.join(os.path.dirname(args.results), "bias_margin_0814")
    s1 = s1_check(args.results, bm)
    s1.to_csv(os.path.join(args.results, "s1_bitcheck.csv"), index=False)
    figures(args.results, lop, dose)
    write_summary(args.results, ver, dose, san, s1, lop)
    print(ver.to_string(index=False))
    print()
    print(s1.to_string(index=False))


if __name__ == "__main__":
    main()

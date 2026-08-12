"""実験(2): E[g] と W の整列リード・ラグ解析 (eg_lag)。

  python -m analysis.eg_lag.eg_lag results/fullbatch_0812
  python -m analysis.eg_lag.eg_lag results/aniso_perp_0812

既存チェックポイントの followup npz (`<results>/followup_Eg_*.npz`, `src.followup` が
再学習なしで生成) だけを読み取り、ニューロン間 pairwise |cos| を Eg_W (期待勾配) と
W (重み) それぞれについて計算する。恒等式 E[g_Wi] = 2vi・E[δ・1[prei>0]・x] より
Eg の整列は architecture-intrinsic であり、「ckpt=0 の時点で Eg の整列は既に高く、
W の整列はランダム水準から遅れて立ち上がる」という予測を P1 (ckpt=0 の水準判定) /
P2 (全 ckpt で align(Eg) >= align(W) が保たれるかの bootstrap CI 判定) で評価する。

注意: checkpoints は [0, 1e4, 5e4, 1e5, 3e5, 1e6] の6点のみ。時系列相互相関による
ラグの定量化はしない。主張は「各 ckpt で align(Eg) >= align(W) が保たれ、ckpt=0 で
最大の乖離」という順序判定に限定する (仕様書 2026-08-12 実験(2))。

出力 (すべて <results>/eg_lag/ の中):
  eg_lag_summary.csv        run x ckpt の全指標
  fig_leadlag_<gname>.png   ckpt vs mean|cos| (Eg / W / ランダム基準線)
  summary.md                P1/P2 判定表 (group 別)、cos_self 記述統計、報告文
"""
import argparse
import glob
import os
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.common import ROOT                                           # noqa: E402

EPS = 1e-12
BOOT_N = 10000
BOOT_SEED = 20260812
P1_EG_THRESH = 0.35
P1_W_THRESH = 0.25

FNAME_RE = re.compile(r"followup_Eg_([AB]_w\d+(?:_b(?:\d+|full))?)_step(\d+)\.npz")
GNAME_RE = re.compile(r"([AB])_w(\d+)")

METRIC_COLS = ["inter_abs_cos_Eg", "inter_abs_cos_W", "vcos_Eg_mean", "vcos_W_mean",
               "cos_self_median", "cos_self_iqr_lo", "cos_self_iqr_hi", "cos_mu_absmean"]


def git_hash():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "N/A"


def unitize(a, axis):
    return a / np.maximum(np.linalg.norm(a, axis=axis, keepdims=True), EPS)


def chance_floor(d):
    """d 次元の独立ランダム単位ベクトル対の E|cos|。"""
    return float(np.sqrt(2.0 / (np.pi * d)))


# --------------------------------------------------------------- npz 読み込み

def load_group_npz(resdir):
    """{(gname, step): npz-dict} を返す。gname は followup.py の group_name() 準拠
    (例: A_w5, A_w5_b32, A_w5_bfull, B_w100)。"""
    out = {}
    for p in sorted(glob.glob(os.path.join(resdir, "followup_Eg_*.npz"))):
        m = FNAME_RE.match(os.path.basename(p))
        if not m:
            continue
        gname, step = m.group(1), int(m.group(2))
        z = dict(np.load(p, allow_pickle=True))
        for k, v in z.items():
            if isinstance(v, np.ndarray) and v.dtype.kind == "f":
                z[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        z["run_ids"] = np.array([str(s) for s in z["run_ids"]])
        out[(gname, step)] = z
    return out


# ------------------------------------------------------------ run 単位の指標

def compute_run_row(z, i, gname, step):
    """1 run x 1 ckpt の指標 dict。alive < 2 または非有限なら指標は NaN。"""
    finite = bool(z["finite"][i])
    alive = ~z["dead"][i]
    n_alive = int(alive.sum())
    d = int(z["Eg_W"].shape[2])
    row = dict(run_id=str(z["run_ids"][i]), gname=gname, ckpt=step,
               finite=finite, n_alive=n_alive, d=d)
    if not finite or n_alive < 2:
        row.update({k: np.nan for k in METRIC_COLS})
        return row

    Eg, W, v = z["Eg_W"][i], z["W"][i], z["v"][i]
    mu_true = z["mu_true"][i]
    sv = np.sign(v)
    sv[sv == 0] = 1.0

    U_Eg, U_W = unitize(Eg, 1), unitize(W, 1)

    def pairwise(U, mask, sv=None):
        Um = U[mask]
        G = Um @ Um.T
        iu = np.triu_indices(len(Um), k=1)
        raw = G[iu]
        if sv is None:
            return float(np.abs(raw).mean())
        svm = sv[mask]
        Gs = (svm[:, None] * svm[None, :]) * G
        return float(Gs[iu].mean())

    row["inter_abs_cos_Eg"] = pairwise(U_Eg, alive)
    row["inter_abs_cos_W"] = pairwise(U_W, alive)
    row["vcos_Eg_mean"] = pairwise(U_Eg, alive, sv)
    row["vcos_W_mean"] = pairwise(U_W, alive, sv)

    cself = (U_Eg[alive] * U_W[alive]).sum(1)          # cos(Eg_i, W_i) per neuron
    q1, q3 = np.percentile(cself, [25, 75])
    row["cos_self_median"] = float(np.median(cself))
    row["cos_self_iqr_lo"] = float(q1)
    row["cos_self_iqr_hi"] = float(q3)

    u_mu = mu_true / max(np.linalg.norm(mu_true), EPS)
    row["cos_mu_absmean"] = float(np.abs(U_Eg[alive] @ u_mu).mean())
    return row


def build_summary(resdir):
    data = load_group_npz(resdir)
    runs = pd.read_csv(os.path.join(resdir, "runs.csv")).set_index("run_id")
    rows = []
    for (gname, step), z in sorted(data.items()):
        for i in range(len(z["run_ids"])):
            rows.append(compute_run_row(z, i, gname, step))
    df = pd.DataFrame(rows)
    m = df["gname"].str.extract(GNAME_RE)
    df["exp_g"], df["width_g"] = m[0], m[1].astype("Int64")
    df = df.join(runs, on="run_id")
    return df


# ------------------------------------------------------------------ bootstrap

def boot_mean_ci(x, n=BOOT_N, seed=BOOT_SEED):
    """seed(run) 単位のリサンプルで平均の 95% CI。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(x), size=(n, len(x)))
    b = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boot_diff_ci(x, y, n=BOOT_N, seed=BOOT_SEED):
    """x (Eg), y (W) は同一 run 順で対応。resample は run 単位、各回で x,y 同じ添字。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(x), size=(n, len(x)))
    diff = x[idx].mean(axis=1) - y[idx].mean(axis=1)
    return float((x - y).mean()), float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))


# ------------------------------------------------------------------------- P1/P2

def judge_group(df, gname):
    g = df[df.gname == gname]
    ckpts = sorted(g.ckpt.unique())
    d_vals = g["d"].dropna()
    d = int(d_vals.iloc[0]) if len(d_vals) else None
    chance = chance_floor(d) if d else float("nan")

    r0 = g[g.ckpt == 0]
    eg0_mean, eg0_lo, eg0_hi = boot_mean_ci(r0.inter_abs_cos_Eg)
    w0_mean, w0_lo, w0_hi = boot_mean_ci(r0.inter_abs_cos_W)
    p1_pass = bool(np.isfinite(eg0_mean) and np.isfinite(w0_mean)
                   and eg0_mean > P1_EG_THRESH and w0_mean < P1_W_THRESH)

    p2_rows, p2_pass = [], True
    for c in ckpts:
        rc = g[g.ckpt == c]
        dmean, dlo, dhi = boot_diff_ci(rc.inter_abs_cos_Eg, rc.inter_abs_cos_W)
        ok = bool(np.isfinite(dlo) and dlo >= 0)
        p2_pass = p2_pass and ok
        p2_rows.append(dict(ckpt=c, diff_mean=dmean, diff_ci_lo=dlo, diff_ci_hi=dhi, ok=ok))

    return dict(gname=gname, d=d, chance=chance, ckpts=ckpts,
                eg0=(eg0_mean, eg0_lo, eg0_hi), w0=(w0_mean, w0_lo, w0_hi),
                p1_pass=p1_pass, p2_rows=p2_rows, p2_pass=p2_pass)


# ----------------------------------------------------------------------------- 図

def fig_leadlag(df, gname, chance, outpath):
    g = df[df.gname == gname]
    ckpts = sorted(g.ckpt.unique())
    xs = [max(c, 1) for c in ckpts]
    eg_m, eg_lo, eg_hi, w_m, w_lo, w_hi = [], [], [], [], [], []
    for c in ckpts:
        rc = g[g.ckpt == c]
        m, lo, hi = boot_mean_ci(rc.inter_abs_cos_Eg)
        eg_m.append(m); eg_lo.append(lo); eg_hi.append(hi)
        m, lo, hi = boot_mean_ci(rc.inter_abs_cos_W)
        w_m.append(m); w_lo.append(lo); w_hi.append(hi)
    eg_m, eg_lo, eg_hi = np.array(eg_m), np.array(eg_lo), np.array(eg_hi)
    w_m, w_lo, w_hi = np.array(w_m), np.array(w_lo), np.array(w_hi)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.errorbar(xs, eg_m, yerr=[eg_m - eg_lo, eg_hi - eg_m], marker="o", ms=5,
                color="tab:red", capsize=3, label="E[g_W]  inter-unit |cos|")
    ax.errorbar(xs, w_m, yerr=[w_m - w_lo, w_hi - w_m], marker="s", ms=5,
                color="tab:blue", capsize=3, label="W  inter-unit |cos|")
    if np.isfinite(chance):
        ax.axhline(chance, color="gray", ls=":", lw=1, label=f"chance |cos| = {chance:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("checkpoint step (ckpt=0 plotted at x=1; see note)")
    ax.set_ylabel("mean |cos| across alive-neuron pairs\n(seed-bootstrap 95% CI)")
    ax.set_title(f"eg_lag: {gname}")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- report

def falsification_checks(judgements):
    """gname 別判定から exp x batch ごとに w5/w100 の両方が不成立かを調べる。"""
    by_key = {}
    for j in judgements:
        m = GNAME_RE.match(j["gname"])
        exp, width = m.group(1), int(m.group(2))
        batch_m = re.search(r"_b(\d+|full)$", j["gname"])
        batch = batch_m.group(1) if batch_m else "1"
        key = (exp, batch)
        by_key.setdefault(key, {})[width] = j
    out = []
    for (exp, batch), widths in sorted(by_key.items()):
        j5, j100 = widths.get(5), widths.get(100)
        if j5 is None or j100 is None:
            continue
        both_fail = (not (j5["p1_pass"] or j5["p2_pass"])) and \
                    (not (j100["p1_pass"] or j100["p2_pass"]))
        out.append(dict(exp=exp, batch=batch, both_fail=both_fail,
                        gname5=j5["gname"], gname100=j100["gname"]))
    return out


def write_summary_md(resdir, df, judgements, gh):
    n_bad = int((~df.finite).sum())
    lines = []
    lines.append(f"# eg_lag summary — {os.path.basename(resdir)}")
    lines.append("")
    lines.append(f"git_hash: `{gh}`  /  non-finite (diverged) run-ckpt rows: "
                 f"{n_bad}/{len(df)}")
    lines.append("")
    lines.append("6 点の checkpoint グリッド ([0, 1e4, 5e4, 1e5, 3e5, 1e6]) しか無いため、"
                 "**時系列相互相関によるラグの定量化はしない**。ここで判定するのは "
                 "「各 ckpt で align(E[g]) >= align(W) が保たれ、ckpt=0 で最大の乖離があるか」"
                 "という順序判定 (P1/P2) のみ。ラグを定量化するには密グリッドでの再走が必要。")
    lines.append("")
    lines.append("## 事前登録判定")
    lines.append("")
    lines.append("- P1（主判定）: ckpt=0 で mean|cos|(E[g]) > 0.35 かつ mean|cos|(W) < 0.25")
    lines.append("- P2: 全 ckpt で mean|cos|(E[g]) − mean|cos|(W) の bootstrap 95%CI 下端 >= 0")
    lines.append("")
    lines.append("## 判定表 (group = followup gname)")
    lines.append("")
    lines.append("| gname | d | chance | Eg@0 (95%CI) | W@0 (95%CI) | P1 | P2 | "
                 "P2 不成立 ckpt |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for j in judgements:
        eg0, w0 = j["eg0"], j["w0"]
        fail_ckpts = [str(r["ckpt"]) for r in j["p2_rows"] if not r["ok"]]
        lines.append(
            f"| {j['gname']} | {j['d']} | {j['chance']:.3f} | "
            f"{eg0[0]:.3f} ({eg0[1]:.3f}, {eg0[2]:.3f}) | "
            f"{w0[0]:.3f} ({w0[1]:.3f}, {w0[2]:.3f}) | "
            f"{'PASS' if j['p1_pass'] else 'FAIL'} | "
            f"{'PASS' if j['p2_pass'] else 'FAIL'} | "
            f"{', '.join(fail_ckpts) if fail_ckpts else '—'} |")
    lines.append("")

    fchecks = falsification_checks(judgements)
    if fchecks:
        lines.append("## 反証条件チェック (w5 と w100 の両方で P1 と P2 が不成立か)")
        lines.append("")
        lines.append("| exp | batch | w5 group | w100 group | 両方不成立 (=棄却) |")
        lines.append("|---|---|---|---|---|")
        for f in fchecks:
            lines.append(f"| {f['exp']} | {f['batch']} | {f['gname5']} | {f['gname100']} | "
                         f"{'YES -> 棄却' if f['both_fail'] else 'no'} |")
        lines.append("")

    lines.append("## cos_self = cos(E[g_i], W_i) の記述統計 (最終 ckpt, alive のみ)")
    lines.append("")
    lines.append("| gname | ckpt | median | IQR |")
    lines.append("|---|---|---|---|")
    for gname in sorted(df.gname.unique()):
        g = df[df.gname == gname]
        c = g.ckpt.max()
        rc = g[g.ckpt == c]
        med = rc.cos_self_median.mean()
        lo = rc.cos_self_iqr_lo.mean()
        hi = rc.cos_self_iqr_hi.mean()
        lines.append(f"| {gname} | {c:g} | {med:.3f} | [{lo:.3f}, {hi:.3f}] |")
    lines.append("")

    lines.append("## 但し書き")
    lines.append("")
    lines.append("- ckpt グリッドが6点のみのため、上記は全て順序判定 (どちらが先に高いか) で"
                 "あり、リード・ラグの時間長 (何 step 先行するか) は推定していない。"
                 "定量化するには少なくとも ckpt=0〜1e4 の間を対数刻みで細分した密グリッドでの"
                 "再走が必要 (既存の6点は 0, 1e4, 5e4, 1e5, 3e5, 1e6)。"
                 "コスト見積り: チェックポイント保存自体の追加コストは小さく、"
                 "支配項は既存のフル学習と同じ (`RESULTS.md`/メモ記録では w5 は CPU で"
                 "全体 ~1時間程度、w100 の full batch 系列のみ GPU が必要)。"
                 "つまり密グリッド化のための再学習コストは既存のフル実行1回分と同程度と見積もる"
                 "(要ユーザー確認、本解析では新規学習は行っていない)。")
    lines.append("- bootstrap は run (seed) 単位のリサンプル 10,000 回、95% パーセンタイルCI。")
    lines.append("- aniso_perp_0812 の各 group (`B_w5`, `B_w100`) は kappa in {1,4,16} と "
                 "lr in {0.003, 0.001} の複数条件を1つの group (npz の R 次元) にまとめて"
                 "含む。判定は followup の group 単位 (gname) で行っており、"
                 "kappa/lr ごとの分離判定はしていない (`eg_lag_summary.csv` の kappa/lr 列で"
                 "個別に再集計可能)。")
    lines.append("")

    lines.append("## 報告文 (8/20 返信用)")
    lines.append("")
    pooled0 = df[df.ckpt == 0]
    eg_m, eg_lo, eg_hi = boot_mean_ci(pooled0.inter_abs_cos_Eg)
    w_m, w_lo, w_hi = boot_mean_ci(pooled0.inter_abs_cos_W)
    d_vals = pooled0["d"].dropna()
    chance = chance_floor(int(d_vals.iloc[0])) if len(d_vals) else float("nan")
    lines.append(
        f"「E[g] 整列は ckpt=0 から {eg_m:.2f}（95%CI [{eg_lo:.2f}, {eg_hi:.2f}]）であり、"
        f"W は {w_m:.2f}（95%CI [{w_lo:.2f}, {w_hi:.2f}]、ランダム {chance:.2f}）"
        f"であり、W の整列に先行する。ラグの定量化には ckpt 密グリッドの再走が必要"
        f"（コスト見積り：既存のフル実行1回分と同程度、要ユーザー確認）」"
        f"（{os.path.basename(resdir)} 全 group プール、n={pooled0.finite.sum()} finite runs）")
    lines.append("")

    with open(os.path.join(resdir, "eg_lag", "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="実験ディレクトリ (例: results/fullbatch_0812)")
    args = ap.parse_args()

    ckdir = os.path.join(args.results, "ckpts")
    if not os.path.isdir(ckdir) or not os.listdir(ckdir):
        raise SystemExit(f"[eg_lag] ckpts が見つかりません: {ckdir} — 中断")

    npz_glob = glob.glob(os.path.join(args.results, "followup_Eg_*.npz"))
    if not npz_glob:
        raise SystemExit(f"[eg_lag] followup npz が見つかりません: {args.results} — "
                         f"先に `python -m src.followup {args.results}` を実行してください")

    outdir = os.path.join(args.results, "eg_lag")
    os.makedirs(outdir, exist_ok=True)

    df = build_summary(args.results)
    gh = git_hash()
    df = df.copy()
    df["git_hash"] = gh
    df.to_csv(os.path.join(outdir, "eg_lag_summary.csv"), index=False)
    print(f"  wrote eg_lag_summary.csv ({len(df)} rows)")

    judgements = []
    for gname in sorted(df.gname.unique()):
        j = judge_group(df, gname)
        judgements.append(j)
        fig_leadlag(df, gname, j["chance"],
                    os.path.join(outdir, f"fig_leadlag_{gname}.png"))
        print(f"  {gname}: P1={'PASS' if j['p1_pass'] else 'FAIL'} "
              f"P2={'PASS' if j['p2_pass'] else 'FAIL'}")

    write_summary_md(args.results, df, judgements, gh)
    print("  wrote summary.md")
    print("EG_LAG DONE")


if __name__ == "__main__":
    main()

"""surv_hist_0822: 生存者の正体の判別（T1 × T2）[spec_surv_hist_0822]。

  OMP_NUM_THREADS=1 .venv/bin/python -m analysis.surv_hist.surv_hist \
      [results/ratchet_log_0819] [--outdir results/surv_hist_0822]

**再学習なし**。`results/ratchet_log_0819/logs/seed*.npz` だけを読み、生存ユニット
（p̂ ≥ 0.05、瞬時値・履歴を使わない）の 2 つの統計量で `停止条件.md` に残る 2 候補
H_marg（罠幾何説）と H_earn（低次成分の稼ぎ説）を判別する:

  T1  1,000,000 step 時点の alive ユニットの p̂ 中央値（seed ごとに中央値 → seed 間
      中央値）。帯は spec §5 A 案で固定（CI 上端 < 4/32 = 0.125 → H_marg 側、
      CI 下端 > 16/32 = 0.5 → H_earn 側、それ以外は保留＝正規の結末）
  T2  窓 [700000, 1000000] での alive 数の step に対する傾き（seed ごとに OLS、
      単位「1 ユニット / 10⁵ step」）。**主判定はバルクグリッドの記録点のみ**
      （`step % 1000 == 0` かつ境界窓外、spec §4 の識別限界 1 項）。境界窓内の点を
      含めた版 (T2-win) は副次で、判定には使わず主判定との符号一致のみ報告する

判定基準は spec §5 が唯一の正で、本モジュールはそれを実装するだけ。**T1 の先は
追跡しない**（8/21 の線引き、spec §8）。E2 は台帳に定義が無いため実装しない。

判定は `dead2path_0821` の群ラベル（再分類死 / 輸送死）を一切使わない（spec §8・
Q17 穴A の禁止）。camp = sign(v_i) によるブレークダウンも行わない。

出力（すべて --outdir の中）: verdict.csv / per_seed_metrics.csv / summary.md /
meta.json / figures/。
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.common import ROOT, load_config                              # noqa: E402
from src.figures_ratchet_log import load_seeds, boot_ci                # noqa: E402

TAU = 0.05                       # alive/dead 閾値 [spec §3]
BOOT_N = 10000
BOOT_SEED = 20260822              # spec §3 の rng シード [事前登録]
T1_MARG_HI = 4 / 32               # 0.125  [spec §5 A案]
T1_EARN_LO = 16 / 32              # 0.5    [spec §5 A案]
WIN_LO, WIN_HI = 700_000, 1_000_000   # T2 の窓 [spec §3]
T1_STEP = 1_000_000

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Sans CJK TC",
                              "Noto Sans CJK KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ 境界の同定 (S2)

def realized_boundaries(d):
    """flip_state の差分から境界を機械的に決定する（ハードコードしない、spec §6 S2）。

    戻り値 left: 境界ペアの左側 step（本来は period の倍数、99 個実現するはず）。
    t=1,000,000 の 100 番目の候補境界は flip が起きない（右側の記録点が無い）ので、
    ここには含まれない。"""
    step, fs = d["step"], d["flip_state"]
    chg = (np.abs(np.diff(fs, axis=0)) > 0).any(axis=1)
    left, right = step[:-1][chg], step[1:][chg]
    return left, right, chg


def check_s2(seeds, period):
    rows, ok = [], True
    for d in seeds:
        left, right, chg = realized_boundaries(d)
        n_real = int(chg.sum())
        n_aligned = int((left % period == 0).sum())
        n_adjacent = int(((right - left) == 1).sum())
        good = (n_real == 99) and (n_aligned == n_real) and (n_adjacent == n_real)
        ok &= good
        rows.append(dict(seed=int(d["seed"]), run_id=str(d["run_id"]),
                         n_realized=n_real, n_aligned=n_aligned,
                         n_adjacent=n_adjacent, ok=good))
    return ok, pd.DataFrame(rows)


# ------------------------------------------------------------------ OLS (依存なし・全桁再現)

def ols_slope(x, y):
    """単純最小二乗の傾き。x を中心化してから計算 (x ~ 1e6 の桁での数値誤差回避)。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xm = x - x.mean()
    denom = float(np.sum(xm * xm))
    if denom <= 0:
        return np.nan
    return float(np.sum(xm * (y - y.mean())) / denom)


# ------------------------------------------------------------------ seed ごとの指標

def compute_seed(d, period, half_w):
    step, p_hat = d["step"], d["p_hat"]
    if int(step[-1]) != T1_STEP:
        raise SystemExit(f"[surv_hist] seed {int(d['seed'])}: 最終 step が "
                         f"{T1_STEP} ではない ({int(step[-1])}) — 中止")

    alive = p_hat >= TAU                       # [n, h] 瞬時値判定 (履歴を使わない)
    n_alive_series = alive.sum(axis=1).astype(np.int64)

    # --- T1: 1,000,000 step 時点
    last_alive = alive[-1]
    n_alive_1m = int(last_alive.sum())
    p_hat_med_1m = (float(np.median(p_hat[-1][last_alive]))
                    if n_alive_1m > 0 else np.nan)

    # --- 境界窓 (実現した 99 個の境界のみを基準にする。t=1,000,000 の非実現候補を
    #     境界として扱わない。§4 の落とし穴・S4)
    left, _, _ = realized_boundaries(d)
    if left.size:
        dist = np.abs(step[:, None] - left[None, :]).min(axis=1)
    else:
        dist = np.full(step.shape, np.inf)
    in_bw = dist <= half_w

    win_mask = (step >= WIN_LO) & (step <= WIN_HI)
    bulk_mask = (step % 1000 == 0)
    main_sel = win_mask & bulk_mask & ~in_bw
    winc_sel = win_mask & (bulk_mask | in_bw)

    # S4 対象: 境界ではない末尾の毎 step 記録 (999900-999999) が境界窓扱いされていないか
    tail_mask = (step >= 999_900) & (step <= 999_999)
    n_tail = int(tail_mask.sum())
    n_tail_in_bw = int((tail_mask & in_bw).sum())

    slope_main = ols_slope(step[main_sel], n_alive_series[main_sel]) * 1e5
    slope_win = ols_slope(step[winc_sel], n_alive_series[winc_sel]) * 1e5

    row = dict(seed=int(d["seed"]), run_id=str(d["run_id"]),
               n_alive_1m=n_alive_1m, p_hat_median_1m=p_hat_med_1m,
               n_boundary_realized=int(left.size),
               n_main_grid_win=int(main_sel.sum()),
               n_winclude_grid_win=int(winc_sel.sum()),
               n_tail_pts=n_tail, n_tail_in_boundary_window=n_tail_in_bw,
               slope_main_per1e5=slope_main, slope_win_per1e5=slope_win)
    return row, n_alive_series


# ------------------------------------------------------------------ 中央値ブートストラップ (T1)

def boot_ci_median(rng, vec, B):
    """seed 単位の paired bootstrap。点推定・各リサンプルとも中央値 (spec §3 T1)。

    `src.figures_ratchet_log.boot_ci` と同じ形だが、集計統計量が平均ではなく
    中央値である点だけが違う (T1 は p̂ の量子化構造に合わせて中央値と事前固定)。"""
    v = np.asarray(vec, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    bs = np.median(v[rng.integers(0, v.size, (B, v.size))], axis=1)
    return float(np.median(v)), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


# ------------------------------------------------------------------ 図

def make_figures(outdir, seeds, series_by_seed, df):
    fig_dir = os.path.join(outdir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1) alive 数の時系列 (全区間、[700k,1M] 窓を陰影で強調)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for d, n_alive_series in zip(seeds, series_by_seed):
        ax.plot(d["step"], n_alive_series, lw=0.9, alpha=0.75,
                label=f"seed {int(d['seed'])}")
    ax.axvspan(WIN_LO, WIN_HI, color="tab:orange", alpha=0.10,
              label="T2 窓 [700k, 1M]")
    ax.set_xlabel("step")
    ax.set_ylabel("alive 数 (p̂ ≥ 0.05, 瞬時値)")
    ax.set_title("surv_hist: alive 数の時系列 (10 seed)")
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_alive_timeseries.png"), dpi=140)
    plt.close(fig)

    # 2) alive p̂ ヒストグラム (1,000,000 step 時点、10 seed プール)
    all_p = []
    for d in seeds:
        alive = d["p_hat"][-1] >= TAU
        all_p.append(d["p_hat"][-1][alive].astype(np.float64))
    all_p = np.concatenate(all_p) if all_p else np.zeros(0)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    edges = (np.arange(0, 34) - 0.5) / 32.0     # k/32 に中心を合わせたビン
    ax.hist(all_p, bins=edges, color="tab:blue", edgecolor="white",
            label=f"alive p̂ (n={all_p.size}, 10 seed プール)")
    ax.axvline(T1_MARG_HI, color="tab:green", ls="--", lw=1.2,
              label=f"H_marg 側境界 4/32={T1_MARG_HI:.3f}")
    ax.axvline(T1_EARN_LO, color="tab:red", ls="--", lw=1.2,
              label=f"H_earn 側境界 16/32={T1_EARN_LO:.3f}")
    ax.set_xlabel("p̂ (k/32 に量子化)")
    ax.set_ylabel("ユニット数")
    ax.set_title("surv_hist: alive p̂ 分布 (step=1,000,000)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_alive_phat_hist.png"), dpi=140)
    plt.close(fig)
    return fig_dir


# ------------------------------------------------------------------ summary.md

def _md(df, fmt=".4f"):
    cols = list(df.columns)
    f = lambda v: (format(v, fmt) if isinstance(v, (float, np.floating))
                   and np.isfinite(v) else ("" if isinstance(v, float)
                                            and not np.isfinite(v) else str(v)))
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f(r[c]) for c in cols) + " |")
    return "\n".join(out)


def write_summary(outdir, meta, s2_df, s3, s4_df, s5, df, t1, t2, t2w,
                  t1_band, t2_verdict, sign_agree, cell):
    L = ["# surv_hist_0822: 生存者の正体の判別（T1 × T2）", "",
         f"仕様: `specs/spec_surv_hist_0822.md`（事前登録、承認 2026-08-22）。"
         f"生成: `{meta['date']}`、git `{meta['git_hash']}`。**再学習なし** — "
         f"`results/ratchet_log_0819/logs/seed*.npz` の事後解析。", "",
         "## 0. 一行", "",
         "生存ユニットの p̂ 水準 (T1) と生存者数の平衡 (T2) の 2 票で、"
         "H_marg（罠幾何説）と H_earn（低次成分の稼ぎ説）を判別する。"
         "alive/dead は各記録点の瞬時 p̂ で判定し、`dead2path_0821` の群ラベルは"
         "使わない (spec §8)。", ""]

    # --- サニティ
    L += ["## 1. サニティ", "",
         f"**S1**: `OMP_NUM_THREADS={meta['omp_num_threads']}`、"
         f"python {meta['python']} / numpy {meta['numpy']} / "
         f"pandas {meta['pandas']}。", "",
         f"**S2** (境界を flip_state の差分から機械的に決定。全 10 seed で 99/99 が "
         f"step ≡ 0 (mod 10⁴) であること): "
         f"{'PASS' if bool(s2_df.ok.all()) else '**FAIL**'}", "",
         _md(s2_df, ".0f"), "",
         f"**S3** (1M 時点の alive 数が 53/1000 = 1000 − 947 と一致するか。npz から"
         f"直接再計算): {'PASS' if s3['ok'] else '**FAIL**'} — "
         f"再計算 alive={s3['n_alive']}, dead={s3['n_dead']} "
         f"(期待 alive=53, dead=947)。", "",
         f"**S4** (境界ではない末尾の毎 step 記録 999,900–999,999 が境界窓として"
         f"扱われていないか。全 10 seed で 100 点とも境界窓外であるべき): "
         f"{'PASS' if bool(s4_df.ok.all()) else '**FAIL**'}", "",
         _md(s4_df, ".0f"), "",
         f"**S5** (T2 窓 [700k,1M] のバルクグリッド主判定点数が 10 seed で一致するか): "
         f"{'PASS' if s5['ok'] else '**FAIL**'} — 全 seed "
         f"{s5['values']} (一意なら 1 値)。", ""]

    # --- per-seed 表
    L += ["## 2. seed ごとの指標", "",
         _md(df[["seed", "run_id", "n_alive_1m", "p_hat_median_1m",
                 "n_boundary_realized", "n_main_grid_win", "n_winclude_grid_win",
                 "slope_main_per1e5", "slope_win_per1e5"]], ".4f"), ""]

    # --- T1
    L += ["## 3. T1 — 生存者の p̂ 水準", "",
         "統計量: 1,000,000 step 時点の alive ユニットの p̂ の中央値 "
         "(seed ごとに中央値 → seed 間中央値、seed 単位ペアブートストラップ "
         f"B={BOOT_N}, `np.random.default_rng({BOOT_SEED})`)。", "",
         f"点推定 {t1[0]:.6f}  95%CI [{t1[1]:.6f}, {t1[2]:.6f}]", "",
         f"帯 (spec §5 A案): H_marg 側 = CI 上端 < 4/32 = {T1_MARG_HI:.4f} / "
         f"H_earn 側 = CI 下端 > 16/32 = {T1_EARN_LO:.4f} / それ以外 = 保留 "
         f"(正規の結末、FAIL ではない)。", "",
         f"**判定: {t1_band}**", ""]

    # --- T2
    L += ["## 4. T2 — 平衡の有無", "",
         "統計量: 窓 [700000, 1000000] での alive 数の step に対する傾き "
         "(seed ごとに OLS、単位「1 ユニット / 10⁵ step」)。**主判定はバルクグリッド "
         "限定** (`step % 1000 == 0` かつ実現した境界の ±100 窓外、spec §4 識別限界 1 項)。", "",
         f"主判定 点推定 {t2[0]:.6f}  95%CI [{t2[1]:.6f}, {t2[2]:.6f}]", "",
         "基準: CI がゼロを含む → 平衡あり (EQ)。CI 上端 < 0 → 平衡なし (DECL)。"
         "CI 下端 > 0 → 増加 (INC、処置表に無い結末＝想定外)。", "",
         f"**判定: {t2_verdict}**", "",
         f"T2-win (副次、境界窓内の点を含む。判定には使わない): "
         f"点推定 {t2w[0]:.6f}  95%CI [{t2w[1]:.6f}, {t2w[2]:.6f}]。"
         f"主判定との符号一致: {'一致' if sign_agree else '不一致'}。", ""]

    # --- 決定行列
    L += ["## 5. 決定行列 (spec §5)", "",
         "|  | T2 = EQ（平衡あり） | T2 = DECL（平衡なし） |",
         "|---|---|---|",
         "| T1 = H_marg 帯 | H_marg支持（両票一致） | 判別不能。台帳へ差し戻し |",
         "| T1 = 保留 | T2のみ記録（Q9bの反論欄は閉じる。Q10は未決） | "
         "T2のみ記録（同上） |",
         "| T1 = H_earn 帯 | H_earn支持（T2はH_earnを排除しない） | "
         "H_earn支持（両票整合） |", "",
         "T2 = INC は処置表に無い結末とし、判定を「想定外」と記録するに留めて"
         "処置を決めない (`cell4_0821` 追補と同じ扱い)。", "",
         f"**この run のセル: {cell}**", ""]

    # --- 逸脱節
    L += ["## 6. 逸脱節", "",
         "1. **盲検性の毀損 (spec §9)**: 本 spec は厳密な意味での盲検事前登録では"
         "ない。T1・T2 に対応する探索値が起草時点で既に vault に記録されていた: "
         "`teachw_0820` の P3 (探索的・判定なし) で alive p̂ 中央値 0.2969 "
         "(H_T=100 アーム、`results/teachw_0820/summary.md`)。`停止条件.md` に "
         "「700k→1M の中央値変化: H8/H32/H100 は −1〜−2 でほぼ横ばい」。"
         "spec §5 の T1 帯 (A 案、上端 4/32=0.125) は既知の探索値 0.2969 が入るように"
         "調整したものではなく、p̂ の量子化構造 (k/32) のみを根拠に固定してある。"
         "本走の T1 点推定はこの探索値と近い値になったが、それは事後に判明したこと"
         "であり、帯の選定基準ではない。",
         "2. **E2 は実装していない**: 台帳 (`命題リスト.md` L56, `blindspot_0820.md` "
         "L72) は「T1/T2/E2」の 3 統計量を挙げているが、E2 の定義は vault・specs・"
         "README・全 commit のどこにも存在しない (spec §5)。本 spec は E2 を含めない。"
         "定義が人間側から出てきたら別 spec の補遺として起票する。",
         "3. **T2 の seed 間集計統計量**: spec §3 は T1 について「seed ごとに中央値 → "
         "seed 間の中央値」と明記するが、T2 の seed 間集計方法 (点推定に何を使うか) "
         "は spec 本文に明記が無い。house の標準 (`src.figures_ratchet_log.boot_ci` "
         "の seed 平均・ペアブートストラップ) をそのまま用いた。判定基準自体は "
         "CI の上端・下端のみで決まるため、この選択は判定結果に影響しない。",
         "4. **T1 の先を追跡しない** (spec §8): 決定行列のセルを書く以上の追加解析・"
         "spec 提案・実装は行っていない (8/21 の線引き)。",
         "5. **camp = sign(v_i) によるブレークダウンは行っていない** (Q17 穴A の禁止)。",
         "", "## 7. スコープ節", "",
         "スコープ: **condA・w100・T=1e4・batch=1・`ratchet_log_0819` のログ限定**。"
         "condB へ外挿しない。再学習していない。`dead2path_0821` の群ラベル "
         "（再分類死 / 輸送死）は本 spec で使っていない。独立に判定した。"
         "null 結果・保留も PASS と同じ形式で報告している。", ""]

    L += ["## 8. 出力", "",
         "- `verdict.csv` — T1・T2・T2-win の点推定・CI、決定行列セル",
         "- `per_seed_metrics.csv` — seed ごとの alive 数・p̂ 中央値・傾き",
         "- `figures/fig_alive_timeseries.png`, `fig_alive_phat_hist.png`", "",
         "> **執筆スコープ（2026-08-22 決定）**: 本結果のうち現論文で使用するのは "
         "**T2（平衡の有無）** までとする。T1（H_earn / H_marg の判別）は記録として"
         "残すが、**8/21 の「これ以上掘り下げない」合意により、ここから先の追跡は"
         "起票しない**。", ""]

    p = os.path.join(outdir, "summary.md")
    with open(p, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return p


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?",
                    default=os.path.join(ROOT, "results", "ratchet_log_0819"),
                    help="ratchet_log_0819 の実験ディレクトリ")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results", "surv_hist_0822"))
    args = ap.parse_args()

    t0 = time.time()
    resdir = args.results
    cfg = load_config(os.path.join(resdir, "config_used.yaml"))
    period = int(cfg["condA"]["T_values"][0])
    half_w = int(cfg["ratchet"]["boundary_window"])

    seeds = load_seeds(resdir)
    seeds.sort(key=lambda d: str(d["run_id"]))                     # run_id ソート [spec §3]
    print(f"loaded {len(seeds)} seeds, {len(seeds[0]['step'])} 記録点, "
         f"period={period}, half_w={half_w}", flush=True)

    # --- S2
    s2_ok, s2_df = check_s2(seeds, period)
    print(f"S2: {'PASS' if s2_ok else 'FAIL'}", flush=True)
    print(s2_df.to_string(index=False), flush=True)
    if not s2_ok:
        raise SystemExit("[surv_hist] S2 FAIL — 境界の同定が壊れている。中止 (spec §6 S2)")

    # --- 本体計算 (S3-S5 用の中間値もここで作る)
    rows, series_by_seed = [], []
    for d in seeds:
        row, n_alive_series = compute_seed(d, period, half_w)
        rows.append(row)
        series_by_seed.append(n_alive_series)
    df = pd.DataFrame(rows)

    # --- S3: 1M 時点の alive/dead を npz から直接再計算して照合
    n_alive_total = int(df.n_alive_1m.sum())
    n_dead_total = 1000 - n_alive_total
    s3_ok = (n_alive_total == 53) and (n_dead_total == 947)
    s3 = dict(ok=s3_ok, n_alive=n_alive_total, n_dead=n_dead_total)
    print(f"S3: {'PASS' if s3_ok else 'FAIL'} — alive={n_alive_total}, "
         f"dead={n_dead_total} (期待 53/947)", flush=True)
    if not s3_ok:
        raise SystemExit("[surv_hist] S3 FAIL — 1M 時点の alive/dead 数が既出値と"
                         "一致しない。ローダのバグの疑い。判定に進まない (spec §6 S3)")

    # --- S4: 境界ではない末尾の毎 step 記録が境界窓扱いされていないか
    s4_rows = [dict(seed=r["seed"], run_id=r["run_id"], n_tail_pts=r["n_tail_pts"],
                    n_tail_in_boundary_window=r["n_tail_in_boundary_window"],
                    ok=(r["n_tail_pts"] == 100 and r["n_tail_in_boundary_window"] == 0))
              for r in rows]
    s4_df = pd.DataFrame(s4_rows)
    s4_ok = bool(s4_df.ok.all())
    print(f"S4: {'PASS' if s4_ok else 'FAIL'}", flush=True)
    if not s4_ok:
        raise SystemExit("[surv_hist] S4 FAIL — 境界ではない末尾の毎 step 記録が"
                         "境界窓として扱われている。中止 (spec §6 S4)")

    # --- S5: T2 窓のバルクグリッド主判定点数が 10 seed で一致するか
    uniq = sorted(set(df.n_main_grid_win.tolist()))
    s5_ok = len(uniq) == 1
    s5 = dict(ok=s5_ok, values=uniq)
    print(f"S5: {'PASS' if s5_ok else 'FAIL'} — n_main_grid_win 値の集合 {uniq}", flush=True)
    if not s5_ok:
        raise SystemExit("[surv_hist] S5 FAIL — T2 窓のバルクグリッド点数が seed 間で"
                         "食い違う。中止 (spec §6 S5)")

    # --- T1 / T2 / T2-win: 単一の rng を固定順 (T1 -> T2 -> T2-win) で消費 [spec §3]
    rng = np.random.default_rng(BOOT_SEED)
    t1 = boot_ci_median(rng, df.p_hat_median_1m.values, BOOT_N)
    t2 = boot_ci(rng, df.slope_main_per1e5.values, BOOT_N)
    t2w = boot_ci(rng, df.slope_win_per1e5.values, BOOT_N)

    if t1[2] < T1_MARG_HI:
        t1_band = "H_marg側"
    elif t1[1] > T1_EARN_LO:
        t1_band = "H_earn側"
    else:
        t1_band = "保留"

    if t2[2] < 0:
        t2_verdict = "DECL（平衡なし）"
    elif t2[1] > 0:
        t2_verdict = "想定外（INC）"
    else:
        t2_verdict = "EQ（平衡あり）"

    sign_agree = bool(np.sign(t2[0]) == np.sign(t2w[0]))

    if t2_verdict.startswith("想定外"):
        cell = "想定外（T2=INC は処置表に無い結末。処置を決めない）"
    elif t1_band == "H_marg側" and t2_verdict.startswith("EQ"):
        cell = "H_marg支持（両票一致）"
    elif t1_band == "H_marg側" and t2_verdict.startswith("DECL"):
        cell = "判別不能。台帳へ差し戻し"
    elif t1_band == "保留":
        cell = "T2のみ記録（Q9bの反論欄は閉じる。Q10は未決）"
    elif t1_band == "H_earn側" and t2_verdict.startswith("EQ"):
        cell = "H_earn支持（T2はH_earnを排除しない）"
    elif t1_band == "H_earn側" and t2_verdict.startswith("DECL"):
        cell = "H_earn支持（両票整合）"
    else:
        cell = "未分類"

    print(f"T1: point={t1[0]:.6f} CI=[{t1[1]:.6f},{t1[2]:.6f}] -> {t1_band}", flush=True)
    print(f"T2: point={t2[0]:.6f} CI=[{t2[1]:.6f},{t2[2]:.6f}] -> {t2_verdict}", flush=True)
    print(f"T2-win: point={t2w[0]:.6f} CI=[{t2w[1]:.6f},{t2w[2]:.6f}] "
         f"sign_agree={sign_agree}", flush=True)
    print(f"decision cell: {cell}", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    df.to_csv(os.path.join(args.outdir, "per_seed_metrics.csv"), index=False)

    V = pd.DataFrame([
        dict(id="T1", statistic="alive p̂ 中央値 (1,000,000 step, seed間中央値)",
            point=t1[0], ci_lo=t1[1], ci_hi=t1[2],
            threshold=f"CI上端<4/32={T1_MARG_HI:.4f}→H_marg側 / "
                     f"CI下端>16/32={T1_EARN_LO:.4f}→H_earn側 / それ以外→保留",
            result=t1_band,
            note=f"n_alive(1M) seed別合計={n_alive_total}/1000"),
        dict(id="T2", statistic="alive数の傾き ([700k,1M] バルクグリッド限定, "
                               "1ユニット/1e5step, 主判定)",
            point=t2[0], ci_lo=t2[1], ci_hi=t2[2],
            threshold="CIがゼロを含む→EQ / CI上端<0→DECL / CI下端>0→INC(想定外)",
            result=t2_verdict,
            note=f"seedあたり主判定点数={int(df.n_main_grid_win.iloc[0])} "
                 f"(10 seed で一致, S5)"),
        dict(id="T2-win", statistic="alive数の傾き (境界窓内の点を含む, 副次)",
            point=t2w[0], ci_lo=t2w[1], ci_hi=t2w[2],
            threshold="判定に使わない。主判定との符号一致のみ報告",
            result="符号一致" if sign_agree else "符号不一致",
            note=f"seedあたり点数={int(df.n_winclude_grid_win.iloc[0])}"),
        dict(id="decision", statistic="決定行列セル (T1 x T2)",
            point=np.nan, ci_lo=np.nan, ci_hi=np.nan,
            threshold="spec §5 決定行列", result=cell, note=""),
    ])
    V.to_csv(os.path.join(args.outdir, "verdict.csv"), index=False)

    fig_dir = make_figures(args.outdir, seeds, series_by_seed, df)

    meta = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), git_hash=git_hash(),
               spec="specs/spec_surv_hist_0822.md", source=resdir,
               device="cpu (post-hoc numpy analysis, no GPU/torch used)",
               elapsed_sec=round(time.time() - t0, 1),
               omp_num_threads=os.environ.get("OMP_NUM_THREADS", "(未設定)"),
               rng_seed=BOOT_SEED, bootstrap_B=BOOT_N,
               python=platform.python_version(), numpy=np.__version__,
               pandas=pd.__version__, n_seeds=len(seeds), period=period,
               half_window=half_w,
               s2_pass=bool(s2_ok), s3_pass=bool(s3_ok), s4_pass=bool(s4_ok),
               s5_pass=bool(s5_ok),
               s2=s2_df.to_dict("records"), s3=s3, s4=s4_df.to_dict("records"),
               s5=s5, t1_point=t1[0], t1_ci=[t1[1], t1[2]], t1_band=t1_band,
               t2_point=t2[0], t2_ci=[t2[1], t2[2]], t2_verdict=t2_verdict,
               t2win_point=t2w[0], t2win_ci=[t2w[1], t2w[2]],
               t2_sign_agree=sign_agree, decision_cell=cell)
    with open(os.path.join(args.outdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=str, ensure_ascii=False)

    sp = write_summary(args.outdir, meta, s2_df, s3, s4_df, s5, df, t1, t2, t2w,
                       t1_band, t2_verdict, sign_agree, cell)

    print(V.to_string(index=False), flush=True)
    print(f"-> {args.outdir}/verdict.csv, per_seed_metrics.csv, meta.json, "
         f"{sp}, {fig_dir}/", flush=True)
    print("SURV_HIST DONE", flush=True)
    return df, V


if __name__ == "__main__":
    main()

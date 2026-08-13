"""rank_int_0814 Phase 0: 既存データ (coupling_fbw_0813) から t_int の妥当性確認と
回復/予防ラベル付け、B アームの目標ランク srank_target を算出する (再学習なし)。

  python -m src.rank_int_phase0

仕様 spec_rank_int_0814 §2:
  1. t_int ∈ {150k(主), 300k(副)} で (a) srank_alive が t50 通過済み (b) dead ≤ 0.15
  2. eval_loss 離陸時刻 = 「初期値と 1M 時点値の半値を最初に上抜く step」。
     eval_loss は lop_every=1000 の格子で計測済みなので、格子点そのままが
     1k 窓移動平均に相当する (格子間隔 = 窓幅。境界密測定点は格子外として除く)。
  3. srank_target = step=0 の stable_rank_W_alive (seed 別)
"""
import os

import numpy as np
import pandas as pd

from .common import ROOT

SRC = os.path.join(ROOT, "results", "coupling_fbw_0813")
OUT = os.path.join(ROOT, "results", "rank_int_0814")
T_INTS = [150_000, 300_000]
T_MAIN = 150_000
DEAD_MAX = 0.15


def takeoff_step(steps, vals):
    """仕様字義: 初期値 + 0.5*(1M時点値 − 初期値) を最初に上抜く step。未離陸は NaN。

    注意: full-batch アームは v0 >> v_1M (高い初期損失 → ほぼ 0 まで降下 → LoP で再上昇)
    のため、この定義は閾値が初期値未満になり step=0 で自明に「上抜き」となって退化する。"""
    v0, v1 = vals[0], vals[-1]
    thr = v0 + 0.5 * (v1 - v0)
    above = vals > thr
    return float(steps[np.argmax(above)]) if above.any() else np.nan


def takeoff_step_robust(steps, vals):
    """ロバスト定義 (仕様逸脱、要先生確認): 全区間最小値の位置以降で、
    min + 0.5*(v_1M − min) を最初に上抜く step。「低い平坦部からの離陸」を捉える。"""
    imin = int(np.argmin(vals))
    vmin, v1 = vals[imin], vals[-1]
    if v1 <= vmin:
        return np.nan
    thr = vmin + 0.5 * (v1 - vmin)
    above = vals[imin:] > thr
    return float(steps[imin:][np.argmax(above)]) if above.any() else np.nan


def main():
    os.makedirs(OUT, exist_ok=True)
    t50 = pd.read_csv(os.path.join(SRC, "t50_runs.csv"))
    t50 = t50[(t50.exp == "A") & (t50.batch == "full") & t50.width.isin([10, 20])]
    t50 = t50.set_index(["width", "seed"])

    rows = []
    for width in [10, 20]:
        lop = pd.read_csv(os.path.join(SRC, f"lop_metrics_A_w{width}_bfull.csv"))
        runs = pd.read_csv(os.path.join(SRC, "runs.csv")).set_index("run_id")
        lop = lop.join(runs[["seed"]], on="run_id")
        lop = lop[lop.step % 1000 == 0]          # 1k 格子のみ (境界密測定点を除く)
        for seed, g in lop.groupby("seed"):
            g = g.sort_values("step")
            srank_t50 = t50.loc[(width, seed), "t50_srank_alive_drop"]
            dead_at = {t: float(np.interp(t, g.step.values, g.dead_frac.values))
                       for t in T_INTS}
            v = g.eval_loss.values
            toff = takeoff_step(g.step.values, v)
            toff_r = takeoff_step_robust(g.step.values, v)
            label_spec = "回復" if (np.isfinite(toff) and toff <= T_MAIN) else "予防"
            label = "回復" if (np.isfinite(toff_r) and toff_r <= T_MAIN) else "予防"
            rows.append(dict(
                seed=seed, width=width,
                srank_t50=srank_t50,
                srank_t50_passed_150k=bool(srank_t50 <= 150_000),
                srank_t50_passed_300k=bool(srank_t50 <= 300_000),
                dead_at_tint=dead_at[T_MAIN],
                dead_at_tint_ok=bool(dead_at[T_MAIN] <= DEAD_MAX),
                dead_at_300k=dead_at[300_000],
                dead_at_300k_ok=bool(dead_at[300_000] <= DEAD_MAX),
                evalloss_takeoff=toff,
                evalloss_takeoff_robust=toff_r,
                label_spec=label_spec,
                label=label,
                srank_target=float(g[g.step == 0].stable_rank_W_alive.iloc[0]),
                evalloss_1M=float(g.eval_loss.iloc[-1]),
            ))
    df = pd.DataFrame(rows).sort_values(["width", "seed"])
    df.to_csv(os.path.join(OUT, "phase0_targets.csv"), index=False)

    lines = ["# rank_int_0814 Phase 0 (spec §2): t_int 妥当性・回復/予防ラベル・srank_target\n",
             "入力: results/coupling_fbw_0813 の A_w10_bfull / A_w20_bfull (seed 0–4)。\n",
             "**仕様逸脱 (要先生確認)**: 離陸時刻の字義定義 (初期値と 1M 値の半値) は、"
             "full-batch アームでは v0 >> v_1M (初期損失が高くほぼ 0 まで降下後に再上昇) の"
             "ため閾値が初期値を下回り、全 seed が step=0 で自明に「離陸」して退化する "
             "(evalloss_takeoff 列)。ラベルには意図 (低平坦部からの離陸) に沿うロバスト定義 "
             "「argmin 以降で min + 0.5*(v_1M − min) を最初に上抜く step」"
             "(evalloss_takeoff_robust 列) を用いた。label_spec は字義定義によるラベル。\n",
             df.to_string(index=False), ""]
    n_ok = (df.srank_t50_passed_150k & df.dead_at_tint_ok).sum()
    lines.append(f"\n- t_int=150k で (a) srank t50 通過済み ∧ (b) dead ≤ {DEAD_MAX}: "
                 f"{n_ok}/{len(df)} seed が適格 (不適格 seed も除外せず層別報告)")
    bad = df[~(df.srank_t50_passed_150k & df.dead_at_tint_ok)]
    if len(bad):
        lines.append("- 不適格 seed: "
                     + ", ".join(f"w{r.width}/s{r.seed}"
                                 f"(srank_t50={r.srank_t50:.0f}, dead={r.dead_at_tint:.3f})"
                                 for r in bad.itertuples()))
    lines.append(f"- ラベル分布: {df.groupby(['width', 'label']).size().to_dict()}")
    lines.append(f"- srank_target (step0 stable_rank_W_alive): "
                 f"{df.groupby('width').srank_target.mean().round(3).to_dict()} (幅別平均)")
    with open(os.path.join(OUT, "phase0_summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(df.to_string(index=False))
    print(f"\n-> {OUT}/phase0_targets.csv, phase0_summary.md")


if __name__ == "__main__":
    main()

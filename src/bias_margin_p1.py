"""bias_margin_0814 Phase 1 (spec §3): µ=0 を保ったまま dead_frac > 0 が出るレジーム探索。

  python -m src.bias_margin_p1 --config configs/bias_margin_p1_0814.yaml [--device cpu]

noise_sd と K は run 軸ではないので、値ごとに config を差し替えて学習し
results/bias_margin_0814/phase1/<cell>/ に分ける。選定基準 (§3) を順に適用して採用セルを
決める。基準2 を満たすセルが皆無なら Phase 2 に進まず停止 (事前登録された null)。
"""
import argparse
import copy
import glob
import os
import time

import numpy as np
import pandas as pd

from .common import ROOT, load_config, pick_device, build_runs, group_runs, group_name
from .train import train_group

OUT = os.path.join(ROOT, "results", "bias_margin_0814", "phase1")


def cells_of(cfg):
    """(route, noise_sd, K, tag) の一覧。route1 の noise=0 と route2 の K=1e4 は同一セル
    なので後者を落として重複実行を避ける。"""
    P = cfg["bias_margin_p1"]
    out = []
    for sd in P["route1_noise_sd"]:
        out.append(("route1", float(sd), 10000, f"n{sd:g}_K10000"))
    for K in P["route2_K"]:
        if K == 10000:
            continue                      # route1 の n0_K10000 と同一 (baseline を共有)
        out.append(("route2", 0.0, int(K), f"n0_K{K}"))
    return out


def run_cell(cfg, sd, K, tag, device):
    cfg = copy.deepcopy(cfg)
    cfg["condB"]["target_noise_sd"] = sd
    cfg["condB"]["K_values"] = [K]
    outdir = os.path.join(OUT, tag)
    os.makedirs(outdir, exist_ok=True)
    runs = build_runs(cfg)
    pd.DataFrame(runs).to_csv(os.path.join(outdir, "runs.csv"), index=False)
    import yaml
    with open(os.path.join(outdir, "config_used.yaml"), "w") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True)
    for gkey, gruns in group_runs(runs).items():
        t0 = time.time()
        train_group(gkey, gruns, cfg, device, outdir)
        print(f"    [{tag}] {group_name(gkey)}: R={len(gruns)} {time.time()-t0:.0f}s",
              flush=True)


def collect(cfg):
    rows = []
    for route, sd, K, tag in cells_of(cfg):
        d = os.path.join(OUT, tag)
        if not os.path.isdir(d):
            continue
        runs = pd.read_csv(os.path.join(d, "runs.csv")).set_index("run_id")
        for f in sorted(glob.glob(os.path.join(d, "lop_metrics_*.csv"))):
            lop = pd.read_csv(f).join(runs, on="run_id")
            for rid, g in lop.groupby("run_id"):
                g = g.sort_values("step")
                first, last = g.iloc[0], g.iloc[-1]
                rows.append(dict(
                    route=route, noise_sd=sd, K=K, tag=tag,
                    width=int(g.width.iloc[0]), seed=int(g.seed.iloc[0]), run_id=rid,
                    dead_end=last.dead_frac, dead_max=g.dead_frac.max(),
                    eval0=first.eval_loss, eval_end=last.eval_loss,
                    eval_ratio=last.eval_loss / first.eval_loss if first.eval_loss > 0 else np.nan,
                    b_mean_end=last.b_mean_alive, b_std_end=last.b_std,
                    b_min_end=last.b_min, beta_mean_end=last.beta_mean,
                    p_min_end=last.p_min,
                    srank_end=last.stable_rank_W_alive,
                    finite=bool(np.isfinite(last.eval_loss))))
    return pd.DataFrame(rows)


def select(df, cfg):
    P = cfg["bias_margin_p1"]
    cell = df.groupby(["route", "tag", "noise_sd", "K", "width"]).agg(
        dead_end=("dead_end", "mean"), dead_max=("dead_max", "mean"),
        eval_ratio=("eval_ratio", "mean"), eval_end=("eval_end", "mean"),
        b_mean_end=("b_mean_end", "mean"), b_std_end=("b_std_end", "mean"),
        b_min_end=("b_min_end", "mean"), beta_mean_end=("beta_mean_end", "mean"),
        p_min_end=("p_min_end", "mean"), finite=("finite", "all"),
        n=("seed", "size")).reset_index()
    cell["c1_finite"] = cell.finite
    cell["c2_dead"] = cell.c1_finite & (cell.dead_end >= P["dead_min"])
    cell["c3_lop"] = cell.c2_dead & (cell.eval_ratio >= P["evalloss_ratio_min"])
    return cell.sort_values("dead_end", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bias_margin_p1_0814.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["common"]["device"] = args.device
    device = pick_device(cfg)
    os.makedirs(OUT, exist_ok=True)

    if not args.skip_train:
        for route, sd, K, tag in cells_of(cfg):
            run_cell(cfg, sd, K, tag, device)

    df = collect(cfg)
    df.to_csv(os.path.join(OUT, "phase1_runs.csv"), index=False)
    cell = select(df, cfg)
    cell.to_csv(os.path.join(OUT, "selection.csv"), index=False)

    passing = cell[cell.c3_lop]
    fallback = cell[cell.c2_dead]
    chosen = passing.iloc[0] if len(passing) else (
        fallback.iloc[0] if len(fallback) else None)

    lines = ["# bias_margin_0814 Phase 1: レジーム探索 (spec §3)\n",
             f"µ=0 (c=0, κ=1) を厳密に保った条件B。width {cfg['condB']['widths']}, "
             f"target_hidden {cfg['condB']['target_hidden']}, lr {cfg['condB']['lr_values']}, "
             f"seeds {cfg['common']['seeds']}, {cfg['common']['total_steps']} step。\n",
             "経路1 = target_noise_sd スイープ (K=1e4)、経路2 = K 短縮 (noise=0)。\n",
             "## セル別 (seed 平均、dead_end 降順)\n",
             cell.round(4).to_string(index=False), ""]
    n2 = int(cell.c2_dead.sum())
    lines.append(f"\n- 基準1 (発散なし): {int(cell.c1_finite.sum())}/{len(cell)}")
    lines.append(f"- 基準2 (dead_frac ≥ {cfg['bias_margin_p1']['dead_min']}): {n2}/{len(cell)}")
    lines.append(f"- 基準3 (eval_loss ≥ {cfg['bias_margin_p1']['evalloss_ratio_min']}倍): "
                 f"{int(cell.c3_lop.sum())}/{len(cell)}")

    if n2 == 0:
        lines.append("\n## 判定: **Phase 2 に進まず停止** (事前登録された null)\n")
        lines.append("基準2 (dead_frac ≥ 0.1) を満たすセルが皆無。仕様 §3 の停止規則により"
                     "グリッドを広げずに停止し、「**µ=0 下では b が沈まず Path A は発現しない**」"
                     "と記録する。b 経路の棄却として報告可能。")
        status = "STOP_NO_DEAD"
    else:
        lines.append("\n## 採用セル\n")
        lines.append(chosen.to_frame().T.to_string(index=False))
        if not len(passing):
            lines.append("\n注: 基準3 (LoP 症状) を満たすセルが無いため、基準2 までを満たす"
                         "セルのうち dead_frac 最大を採用した (仕様に規定の無いケース。"
                         "summary に明記)。")
        status = "OK" if len(passing) else "OK_NO_LOP"
    lines.append(f"\n- status: {status}")
    with open(os.path.join(OUT, "phase1_report.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(cell.round(4).to_string(index=False))
    print(f"\nstatus: {status}")


if __name__ == "__main__":
    main()

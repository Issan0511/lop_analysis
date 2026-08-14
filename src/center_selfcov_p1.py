"""center_selfcov_0814 Phase 1 (spec §4): レジーム探索スモーク。

  python -m src.center_selfcov_p1 --config configs/center_selfcov_p1_0814.yaml [--device cpu]

target_hidden は run 軸ではなく教師の構造なので、値ごとに config を差し替えて
run_all 相当の学習を回し、結果を results/center_selfcov_0814/phase1/th{値}/ に分ける。
選定基準 (§4) を順に適用して採用セルを1つ決め、selection.csv / phase1_report.md を出す。
全セルが基準2 (Path B 進行) を満たさなければ **Phase 2 に進まず停止** し、その旨を記録する
(事前登録された null 結果。グリッドは広げない)。
"""
import argparse
import copy
import glob
import os
import time

import numpy as np
import pandas as pd
import torch

from .common import ROOT, load_config, pick_device, build_runs, group_runs, group_name
from .train import train_group

OUT = os.path.join(ROOT, "results", "center_selfcov_0814", "phase1")


def th_tag(th):
    return "same" if th is None else str(th)


def run_all_for_th(cfg, th, device, outdir):
    cfg = copy.deepcopy(cfg)
    cfg["condB"]["target_hidden"] = th
    os.makedirs(outdir, exist_ok=True)
    runs = build_runs(cfg)
    pd.DataFrame(runs).to_csv(os.path.join(outdir, "runs.csv"), index=False)
    import yaml
    with open(os.path.join(outdir, "config_used.yaml"), "w") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True)
    for gkey, gruns in group_runs(runs).items():
        gname = group_name(gkey)
        print(f"    [th={th_tag(th)}] {gname}: R={len(gruns)}", flush=True)
        t0 = time.time()
        train_group(gkey, gruns, cfg, device, outdir)
        print(f"      {time.time()-t0:.0f}s", flush=True)


def collect(cfg):
    """全 th ディレクトリの lop_metrics を読み、セル別の選定指標を作る。"""
    P = cfg["center_selfcov_p1"]
    rows = []
    for th in P["target_hidden_values"]:
        d = os.path.join(OUT, f"th{th_tag(th)}")
        runs = pd.read_csv(os.path.join(d, "runs.csv")).set_index("run_id")
        for f in sorted(glob.glob(os.path.join(d, "lop_metrics_*.csv"))):
            lop = pd.read_csv(f).join(runs, on="run_id")
            for rid, g in lop.groupby("run_id"):
                g = g.sort_values("step")
                r = g.iloc[0]
                first, last = g.iloc[0], g.iloc[-1]
                s0, s1 = first.stable_rank_W_alive, last.stable_rank_W_alive
                e0, e1 = first.eval_loss, last.eval_loss
                rows.append(dict(
                    th=th_tag(th), width=int(r.width), kappa=int(r.kappa),
                    lr=float(r.lr), seed=int(r.seed), run_id=rid,
                    srank0=s0, srank_end=s1,
                    srank_ratio=s1 / s0 if s0 > 0 else np.nan,
                    dead_end=last.dead_frac, eval0=e0, eval_end=e1,
                    eval_ratio=e1 / e0 if e0 > 0 else np.nan,
                    wnorm0=first.w_norm_mean, wnorm_end=last.w_norm_mean,
                    finite=bool(np.isfinite(e1) and np.isfinite(s1)),
                    cos_e1W_end=last.cos_e1W_e1Sig))
    return pd.DataFrame(rows)


def select(df, cfg):
    """§4 の選定基準を順に適用。セル = (th, width, kappa, lr) の seed 平均。"""
    P = cfg["center_selfcov_p1"]
    cell = df.groupby(["th", "width", "kappa", "lr"]).agg(
        srank_ratio=("srank_ratio", "mean"), srank0=("srank0", "mean"),
        srank_end=("srank_end", "mean"), dead_end=("dead_end", "mean"),
        eval_ratio=("eval_ratio", "mean"), eval_end=("eval_end", "mean"),
        cos_e1W_end=("cos_e1W_end", "mean"),
        wnorm_ratio=("wnorm_end", "mean"), finite=("finite", "all"),
        n=("seed", "size")).reset_index()
    cell["wnorm_ratio"] = cell.wnorm_ratio / df.groupby(
        ["th", "width", "kappa", "lr"]).wnorm0.mean().values
    cell["c1_finite"] = cell.finite & (cell.wnorm_ratio < P["wnorm_blowup"])
    cell["c2_pathB"] = cell.c1_finite & (cell.srank_ratio <= P["srank_ratio_max"])
    cell["c3_notA"] = cell.c2_pathB & (cell.dead_end < P["dead_max"])
    cell["c4_lop"] = cell.c3_notA & (cell.eval_ratio >= P["evalloss_ratio_min"])
    return cell.sort_values("srank_ratio")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/center_selfcov_p1_0814.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-train", action="store_true", help="集計のみ再実行")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["common"]["device"] = args.device
    device = pick_device(cfg)
    os.makedirs(OUT, exist_ok=True)

    if not args.skip_train:
        for th in cfg["center_selfcov_p1"]["target_hidden_values"]:
            run_all_for_th(cfg, th, device, os.path.join(OUT, f"th{th_tag(th)}"))

    df = collect(cfg)
    df.to_csv(os.path.join(OUT, "phase1_runs.csv"), index=False)
    cell = select(df, cfg)
    cell.to_csv(os.path.join(OUT, "selection.csv"), index=False)

    passing = cell[cell.c4_lop]
    fallback = cell[cell.c3_notA]
    chosen = None
    if len(passing):
        chosen = passing.iloc[0]           # srank_ratio 昇順 = 低下最大
    elif len(fallback):
        chosen = fallback.iloc[0]

    lines = ["# center_selfcov_0814 Phase 1: レジーム探索 (spec §4)\n",
             f"グリッド: th {cfg['center_selfcov_p1']['target_hidden_values']} × "
             f"width {cfg['condB']['widths']} × kappa {cfg['condB']['kappa_values']} × "
             f"lr {cfg['condB']['lr_values']}, c=0.0, seeds {cfg['common']['seeds']}, "
             f"{cfg['common']['total_steps']} step\n",
             "## セル別の選定指標 (seed 平均、srank_ratio 昇順)\n",
             cell.round(4).to_string(index=False), ""]
    n2 = int(cell.c2_pathB.sum())
    lines.append(f"\n- 基準1 (発散なし) 通過: {int(cell.c1_finite.sum())}/{len(cell)}")
    lines.append(f"- 基準2 (srank ≤ 80%) 通過: {n2}/{len(cell)}")
    lines.append(f"- 基準3 (dead < 0.5) 通過: {int(cell.c3_notA.sum())}/{len(cell)}")
    lines.append(f"- 基準4 (eval_loss 2倍以上) 通過: {int(cell.c4_lop.sum())}/{len(cell)}")

    if n2 == 0:
        lines.append("\n## 判定: **Phase 2 に進まず停止** (事前登録された null 結果)\n")
        lines.append("全セルが基準2 (Path B の進行) を満たさない。仕様 §4 の規定により"
                     "グリッドを広げずに停止し、「c=0 では Path B が進行しない」"
                     "= µ は増幅因子ではなく必要条件に近い、と記録する。")
        status = "STOP_NO_PATHB"
    else:
        lines.append(f"\n## 採用セル\n")
        lines.append(chosen.to_frame().T.to_string(index=False))
        if not len(passing):
            lines.append("\n注: 基準4 (LoP 発現) を満たすセルが無いため、基準3 までを"
                         "満たすセルのうち srank 低下最大を採用した (要 summary 明記)。")
        status = "OK" if len(passing) else "OK_NO_LOP"
    lines.append(f"\n- status: {status}")
    with open(os.path.join(OUT, "phase1_report.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(cell.round(4).to_string(index=False))
    print(f"\nstatus: {status}")
    if chosen is not None:
        print("chosen:", dict(chosen[["th", "width", "kappa", "lr", "srank_ratio",
                                      "dead_end", "eval_ratio"]]))


if __name__ == "__main__":
    main()

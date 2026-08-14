"""bias_margin_0814 Phase 2 (spec §3): 本実験ランナー。

  python -m src.bias_margin --config configs/bias_margin_0814.yaml [--device cpu]
                            [--cells n2.0_K10000_bfree ...]

freeze_bias / target_noise_sd / K は run 軸ではなく学習設定なので、セルごとに config を
差し替えて results/bias_margin_0814/<cell>/ に出力する。A3 (checkpoint 保存) は A1 の
該当セルに checkpoints を付けて兼用する。
"""
import argparse
import copy
import os
import time

import pandas as pd

from .common import ROOT, load_config, pick_device, build_runs, group_runs, group_name
from .train import train_group

OUT = os.path.join(ROOT, "results", "bias_margin_0814")


def cell_tag(sd, K, freeze):
    return f"n{sd:g}_K{K}_{'bfrozen' if freeze else 'bfree'}"


def cells_of(cfg):
    """(arm, noise_sd, K, freeze, tag) の一覧。A1 と A2 の重複は無い (A2 は noise=0 かつ
    K≠1e4)。"""
    P = cfg["bias_margin"]
    out = []
    for fz in P["freeze_values"]:
        for sd in P["a1_noise_sd"]:
            out.append(("A1", float(sd), 10000, bool(fz), cell_tag(float(sd), 10000, fz)))
        for K in P["a2_K"]:
            out.append(("A2", 0.0, int(K), bool(fz), cell_tag(0.0, int(K), fz)))
    return out


def run_cell(cfg, arm, sd, K, freeze, tag, device):
    P = cfg["bias_margin"]
    cfg = copy.deepcopy(cfg)
    cfg["condB"].update(target_noise_sd=sd, K_values=[K], freeze_bias=freeze)
    a3 = P["a3_cell"]
    is_a3 = (sd == float(a3["noise_sd"]) and K == int(a3["K"])
             and freeze == bool(a3["freeze_bias"]))
    if is_a3:
        cfg["common"]["checkpoints"] = P["a3_checkpoints"]
    outdir = os.path.join(OUT, tag)
    os.makedirs(outdir, exist_ok=True)
    runs = build_runs(cfg)
    for r in runs:
        r["arm"], r["noise_sd"], r["K_cell"], r["freeze_bias"] = arm, sd, K, freeze
    pd.DataFrame(runs).to_csv(os.path.join(outdir, "runs.csv"), index=False)
    import yaml
    with open(os.path.join(outdir, "config_used.yaml"), "w") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True)
    for gkey, gruns in group_runs(runs).items():
        t0 = time.time()
        st, _ = train_group(gkey, gruns, cfg, device, outdir)
        print(f"    [{tag}] {group_name(gkey)}: R={len(gruns)} "
              f"{time.time()-t0:.0f}s{' (A3: ckpts)' if is_a3 else ''}", flush=True)
    return is_a3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bias_margin_0814.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--cells", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["common"]["device"] = args.device
    device = pick_device(cfg)
    os.makedirs(OUT, exist_ok=True)

    t0 = time.time()
    done = []
    for arm, sd, K, fz, tag in cells_of(cfg):
        if args.cells and tag not in args.cells:
            continue
        run_cell(cfg, arm, sd, K, fz, tag, device)
        done.append(tag)
    print(f"ALL DONE ({len(done)} cells, {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

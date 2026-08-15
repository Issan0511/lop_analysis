"""cbp_harm_0815: CBP が有害になるレジームの判定 (spec_cbp_harm_0815)。

  python -m src.cbp_harm --config configs/cbp_harm_0815.yaml [--device cpu] [--cells ...]

target_noise_sd / K は run 軸ではないのでセルごとに config を差し替え、
rho は methods 軸で回す (1 セル = 3 group: none / cbp1e-5 / cbp1e-4)。
出力は results/cbp_harm_0815/<cell>/。
"""
import argparse
import copy
import os
import time

import pandas as pd

from .common import ROOT, load_config, pick_device, build_runs, group_runs, group_name
from .train import train_group

OUT = os.path.join(ROOT, "results", "cbp_harm_0815")


def cell_tag(sd, K):
    return f"n{sd:g}_K{K}"


def cells_of(cfg):
    """(route, noise_sd, K, tag)。(σ_ξ=0, K=1e4) は両経路の共有セルなので 1 回だけ。"""
    P = cfg["cbp_harm"]
    out = [("N", float(sd), 10000, cell_tag(float(sd), 10000))
           for sd in P["routeN_noise_sd"]]
    out += [("K", 0.0, int(K), cell_tag(0.0, int(K)))
            for K in P["routeK_K"] if int(K) != 10000]
    return out


def run_cell(cfg, route, sd, K, tag, device):
    cfg = copy.deepcopy(cfg)
    cfg["condB"].update(target_noise_sd=sd, K_values=[K])
    outdir = os.path.join(OUT, tag)
    os.makedirs(outdir, exist_ok=True)
    runs = build_runs(cfg)
    for r in runs:
        r["route"], r["noise_sd"], r["K_cell"] = route, sd, K
        m = r.get("method_cfg", {})
        r["rho"] = float(m.get("rho", 0.0))
    pd.DataFrame(runs).to_csv(os.path.join(outdir, "runs.csv"), index=False)
    import yaml
    with open(os.path.join(outdir, "config_used.yaml"), "w") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True)
    for gkey, gruns in group_runs(runs).items():
        t0 = time.time()
        train_group(gkey, gruns, cfg, device, outdir)
        print(f"    [{tag}] {group_name(gkey)}: R={len(gruns)} {time.time()-t0:.0f}s",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cbp_harm_0815.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--cells", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["common"]["device"] = args.device
    device = pick_device(cfg)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    for route, sd, K, tag in cells_of(cfg):
        if args.cells and tag not in args.cells:
            continue
        run_cell(cfg, route, sd, K, tag, device)
    print(f"ALL DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

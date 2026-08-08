"""実験一括実行 CLI。

  python -m src.run_all --smoke          # 短縮スモークテスト (パイプライン検証)
  python -m src.run_all                  # フル実行 (config.yaml どおり)
  python -m src.run_all --groups A_w5    # グループ指定
"""
import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time

import torch

from .common import ROOT, load_config, pick_device, build_runs, group_runs
from .train import train_group
from .freeze import run_freeze_all


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "N/A"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="短縮実行でパイプライン検証")
    ap.add_argument("--groups", nargs="*", default=None,
                    help="実行グループ (例: A_w5 A_w100 B_w5 B_w100)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    if args.device:
        cfg["common"]["device"] = args.device
    device = pick_device(cfg)
    torch.backends.cudnn.deterministic = True     # [J]
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True  # 仕様書 §8: TF32 可
    torch.backends.cudnn.allow_tf32 = True

    outdir = args.outdir or os.path.join(ROOT, "results" + ("_smoke" if args.smoke else ""))
    os.makedirs(outdir, exist_ok=True)

    total_steps, ckpts = None, None
    if args.smoke:
        cfg = copy.deepcopy(cfg)
        total_steps = args.steps or 3000
        ckpts = [0, 1000, total_steps]
        cfg["common"]["loss_bin"] = 100
        cfg["common"]["lop_every"] = 1000
        cfg["common"]["freeze_min_periods"] = 5
        cfg["common"]["freeze_M_cap"] = 3000
        cfg["common"]["checkpoints"] = ckpts
    elif args.steps:
        total_steps = args.steps

    runs = build_runs(cfg)
    groups = group_runs(runs)
    sel = args.groups
    with open(os.path.join(outdir, "runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(runs[0].keys()))
        w.writeheader()
        w.writerows(runs)
    with open(os.path.join(outdir, "meta.json"), "w") as fh:
        json.dump(dict(git_hash=git_hash(), device=device,
                       torch=torch.__version__, argv=sys.argv[1:],
                       date=time.strftime("%Y-%m-%d %H:%M:%S")), fh, indent=1)
    import yaml
    with open(os.path.join(outdir, "config_used.yaml"), "w") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True)

    for gkey, gruns in groups.items():
        gname = f"{gkey[0]}_w{gkey[1]}"
        if sel and gname not in sel:
            continue
        print(f"=== train {gname}: R={len(gruns)} device={device}", flush=True)
        t0 = time.time()
        st, elapsed = train_group(gkey, gruns, cfg, device, outdir,
                                  total_steps=total_steps, ckpts=ckpts)
        print(f"    train done in {elapsed:.1f}s "
              f"({(total_steps or cfg['common']['total_steps'])/elapsed:.0f} steps/s)", flush=True)
        ng, nn = run_freeze_all(gkey, cfg, device, outdir, ckpt_steps=ckpts)
        print(f"    freeze done: {ng} global rows, {nn} neuron rows "
              f"(+{time.time()-t0-elapsed:.1f}s)", flush=True)

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

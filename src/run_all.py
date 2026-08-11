"""実験一括実行 CLI。

  python -m src.run_all --config configs/drift_0809.yaml           # フル実行
  python -m src.run_all --config configs/drift_0809.yaml --smoke   # 短縮スモークテスト
  python -m src.run_all --config configs/drift_0809.yaml --groups A_w5

出力先は config のファイル名から決まる (configs/drift_0809.yaml -> results/drift_0809/)。
--smoke のときは常に results/_smoke/ (.gitignore 済み、上書き前提の捨て場)。
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

from .common import (ROOT, load_config, pick_device, build_runs, group_runs,
                     config_title, resolve_outdir)
from .train import train_group
from .freeze import run_freeze_all


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "N/A"


def write_meta(outdir, meta):
    with open(os.path.join(outdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="実験 config (例: configs/drift_0809.yaml)。出力先はこの名前で決まる")
    ap.add_argument("--smoke", action="store_true", help="短縮実行でパイプライン検証")
    ap.add_argument("--groups", nargs="*", default=None,
                    help="実行グループ (例: A_w5 A_w100 B_w5 B_w100)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--outdir", default=None, help="出力先の明示指定 (通常は不要)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--skip-freeze", action="store_true",
                    help="学習後の凍結測定 (E[g] 測定) をスキップ。D 計測のみの実験用")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["common"]["device"] = args.device
    device = pick_device(cfg)
    torch.backends.cudnn.deterministic = True     # [J]
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True  # 仕様書 §8: TF32 可
    torch.backends.cudnn.allow_tf32 = True

    outdir = resolve_outdir(args.config, smoke=args.smoke, outdir=args.outdir)
    os.makedirs(outdir, exist_ok=True)
    print(f"outdir: {outdir}", flush=True)

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
    meta = dict(title=config_title(args.config), date=time.strftime("%Y-%m-%d"),
                git_hash=git_hash(), device=device, smoke=args.smoke,
                elapsed_sec=None, torch=torch.__version__,
                config=os.path.relpath(os.path.abspath(args.config), ROOT),
                started=time.strftime("%Y-%m-%d %H:%M:%S"), argv=sys.argv[1:])
    write_meta(outdir, meta)   # 途中で落ちても「どの状態で走ったか」は残す
    import yaml
    with open(os.path.join(outdir, "config_used.yaml"), "w") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True)

    t_start = time.time()
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
        if args.skip_freeze:
            print("    freeze skipped (--skip-freeze)", flush=True)
        else:
            ng, nn = run_freeze_all(gkey, cfg, device, outdir, ckpt_steps=ckpts)
            print(f"    freeze done: {ng} global rows, {nn} neuron rows "
                  f"(+{time.time()-t0-elapsed:.1f}s)", flush=True)

    meta["elapsed_sec"] = round(time.time() - t_start, 1)
    write_meta(outdir, meta)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

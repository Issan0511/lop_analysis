"""Phase 0: データ発見。事実を集めて data_inventory.md に必要な数値を吐く。

  python -m analysis.june_reanalysis.phase0_inventory
"""
import collections
import glob
import os

import numpy as np

from . import common as C


def main():
    runs = C.load_runs()
    print(f"runs.csv: {len(runs)} runs")
    by = collections.Counter()
    for r in runs.values():
        by[(r["exp"], r["width"], r["period"], r["enc"], r["c"], r["lr"])] += 1
    print(f"conditions (seed を除く): {len(by)}")
    for k, n in sorted(by.items(), key=lambda kv: str(kv[0])):
        print(f"  exp={k[0]} w={k[1]:3d} period={k[2]:6d} enc={k[3]:8s} c={k[4]} lr={k[5]}: {n} seeds")

    print("\nnpz files:")
    fs = sorted(glob.glob(os.path.join(C.SRC_RESULTS, "followup_Eg_*.npz")))
    print(f"  {len(fs)} files")
    ck = sorted(glob.glob(os.path.join(C.SRC_RESULTS, "ckpts", "*.pt")))
    print(f"  ckpts: {len(ck)} files")

    print("\ndead 率と alive 数 (最終 ckpt 1e6 / 1e5):")
    for step in [100000, 1000000]:
        for exp, width in C.GROUPS:
            z = C.load_npz(exp, width, step)
            if z is None:
                continue
            na = (~z["dead"]).sum(axis=1)
            nf = int(z["finite"].sum())
            print(f"  step={step:7d} {exp}_w{width:3d}: alive/neuron min={na.min():3d} "
                  f"med={int(np.median(na)):3d} max={na.max():3d}  finite={nf}/{len(na)}")

    print("\n既報 A1 (vpcos_alive_mean) の再現 — 条件別 seed 平均, signed, v符号補正, alive のみ:")
    for step in [100000, 1000000]:
        for exp, width in C.GROUPS:
            z = C.load_npz(exp, width, step)
            if z is None:
                continue
            acc = collections.defaultdict(list)
            for i, rid in enumerate(z["run_ids"]):
                r = runs[rid]
                if not z["finite"][i]:
                    continue
                for obj in ["W", "Eg"]:
                    M, sv = C.get_matrix(z, i, obj, alive_only=True)
                    acc[(C.cond_label(r), obj)].append(np.mean(C.pair_cos(M, sv))
                                                       if len(M) >= 2 else np.nan)
            for (lab, obj), vs in sorted(acc.items()):
                if width != 100 or "T10000" not in lab and "K10000" not in lab:
                    continue
                a = C.agg_seeds(vs)
                print(f"  step={step:7d} {lab:34s} obj={obj:2s} "
                      f"vpcos={a['mean']:+.3f}±{a['std']:.3f} (n={a['n_seeds']})")

    print("\n入力次元と床:")
    for exp, d in [("A", 20), ("B", 21)]:
        print(f"  cond {exp}: d={d}, chance E|cos| = {C.chance_floor(d):.4f}")

    print("\nΣ (解析値):")
    print("  cond B: x = mu + z, z~N(0,I_d) -> Σ = I_21 (完全等方)")
    print("  cond A: x = [flip(15 bits, 周期内固定), U{0,1}(5 bits)]")
    print("          周期内 Σ = diag(0×15, 0.25×5);  測定窓 (50周期) では flip 側にも分散")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Post hoc (report only; requested by the parent session after the run): per task end and layer, whether the layer
has escaped to the positive side (linearised), is closed, or still gates, and which gating layer is nearest the readout.

    python analysis/depth_silu_rlmnist_1020/gating_layers.py --units /home/issan/Projects/obsidian-research-data/depth_silu_rlmnist_1020/depth_silu_rlmnist_1020/units --out results/depth_silu_rlmnist_1020

Layer states (unit summaries at the task end):
  L (linearised, escaped to the positive side): median r > 0 and mean phi' >= 0.75.  0.75 is half way between the
    gate's value at z = 0 (SiLU phi'(0) = 0.5) and the linear slope 1; SiLU phi'(0.5) = 0.74.  For ReLU mean phi' = p.
  C (closed): mean open fraction p < 0.01.
  G (gating): neither.
The readout-nearest gating layer is the highest-index G layer ('-' if none).
"""
import argparse, csv
from collections import Counter
from pathlib import Path
import numpy as np

ARMS = {("SILU", 1, "none"): "A1", ("SILU", 2, "none"): "A2", ("SILU", 4, "none"): "A4",
        ("R", 1, "none"): "R1", ("R", 4, "none"): "R4", ("SILU", 4, "C"): "A4C"}
ORDER = ["A1", "A2", "A4", "A4C", "R1", "R4"]


def state(u, l, k):
    r = np.nanmedian(u[f"end_r_l{l}"][k]) if np.isfinite(u[f"end_r_l{l}"][k]).any() else np.nan
    g = float(np.mean(u[f"end_gmean_l{l}"][k])); p = float(np.mean(u[f"end_p_l{l}"][k]))
    if p < 0.01:
        return "C", r, g, p
    if r > 0 and g >= 0.75:
        return "L", r, g, p
    return "G", r, g, p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--units", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, seq = [], {}
    for f in sorted(Path(a.units).glob("units_*.npz")):
        act, L, door, s = f.stem.replace("units_", "").split("_")
        arm, L, s = ARMS[(act, int(L[1:]), door)], int(L[1:]), int(s[1:])
        u = np.load(f); T = u["end_p_l1"].shape[0]
        for k in range(T):
            st = [state(u, l, k) for l in range(1, L + 1)]
            gates = [l + 1 for l, x in enumerate(st) if x[0] == "G"]
            last = max(gates) if gates else 0
            seq[(arm, s, k + 1)] = ("".join(x[0] for x in st), last)
            for l, (c, r, g, p) in enumerate(st, start=1):
                rows.append({"arm": arm, "seed": s, "task": k + 1, "layer": l, "state": c, "r_med": r, "gmean": g, "p_mean": p,
                             "last_gating_layer": last})
    out = Path(a.out)
    with open(out / "gating_layers.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    md = ["# 層の状態と読み出しに最も近いゲート層（事後・報告のみ）", "",
          "状態: G = ゲート（どちらでもない）、L = 正側へ逃げた（r の中央値 > 0 かつ φ′ の平均 ≥ 0.75）、C = 閉じた（開く割合 p の平均 < 0.01）。",
          "文字列は第 1 層から順。セルは 10 seed（0〜9）の度数。「最も近いゲート層」は G の層のうち番号が最大のもの（無ければ —）。", ""]
    tasks = [1, 2, 3, 5, 8, 10, 20, 30, 40, 50]
    for arm in ORDER:
        md.append(f"## {arm}\n")
        md.append("| 課題 | 層の状態（度数） | 読み出しに最も近いゲート層（度数） |")
        md.append("|---|---|---|")
        for t in tasks:
            ss = [seq[(arm, s, t)] for s in range(10) if (arm, s, t) in seq]
            c1 = Counter(x[0] for x in ss); c2 = Counter(x[1] for x in ss)
            md.append(f"| {t} | " + "・".join(f"{k} ×{v}" for k, v in c1.most_common()) + " | "
                      + "・".join(f"{('第' + str(k) + '層') if k else '—'} ×{v}" for k, v in sorted(c2.items(), reverse=True)) + " |")
        md.append("")
    (out / "gating_layers.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()

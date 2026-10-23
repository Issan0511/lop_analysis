#!/usr/bin/env python3
"""Post-hoc (not registered) descriptions of depth_silu_rlmnist_1020 from the per-unit arrays.

    python analysis/depth_silu_rlmnist_1020/posthoc.py --units raw/depth_silu_rlmnist_1020/units --out results/depth_silu_rlmnist_1020

(1) always-on units (p > 0.99) per layer and task end: the 'escape to the linear side' of a returning unit;
(2) the returned units: their p before and after the first return, and whether they ended always-on;
(3) figure: median r per layer and the online accuracy over tasks (seed 0-2 means).
"""
import argparse, json
from pathlib import Path
import numpy as np

ARMS = {("SILU", 1, "none"): "A1", ("SILU", 2, "none"): "A2", ("SILU", 4, "none"): "A4",
        ("R", 1, "none"): "R1", ("R", 4, "none"): "R4", ("SILU", 4, "C"): "A4C"}
LAYERS = {"A1": 1, "A2": 2, "A4": 4, "R1": 1, "R4": 4, "A4C": 4}


def parse(name):
    act, L, door, s = name.replace("units_", "").split("_")
    return ARMS[(act, int(L[1:]), door)], int(s[1:])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--units", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--runs", required=True); ap.add_argument("--root", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    rec = {}
    for f in sorted(Path(a.units).glob("units_*.npz")):
        arm, s = parse(f.stem)
        u = np.load(f)
        r = {}
        for l in range(1, LAYERS[arm] + 1):
            p = u[f"end_p_l{l}"]                                  # [T, 100]
            T = p.shape[0]
            on = (p > 0.99).mean(1)
            below = np.maximum.accumulate(p < 0.5, axis=0)
            first = []
            for i in range(p.shape[1]):
                ks = [k for k in range(1, T) if below[k - 1, i] and p[k, i] >= 0.5]
                if ks:
                    k = ks[0]
                    first.append({"unit": i, "task_return": k + 1, "p_before": float(p[k - 1, i]), "p_after": float(p[k, i]),
                                  "min_p_before": float(p[:k, i].min()), "p_t30": float(p[min(29, T - 1), i]), "p_last": float(p[-1, i]),
                                  "always_on_last": bool(p[-1, i] > 0.99), "r_after": float(u[f"end_r_l{l}"][k, i])})
            r[str(l)] = {"always_on_frac": on.tolist(), "below_half_frac": (p < 0.5).mean(1).tolist(), "returned": first,
                         "r_med": np.nanmedian(u[f"end_r_l{l}"], axis=1).tolist(), "s_med": np.nanmedian(u[f"end_s_l{l}"], axis=1).tolist(),
                         "gabs": u[f"end_gabs_l{l}"].mean(1).tolist(), "neff": u[f"end_neff_l{l}"].mean(1).tolist()}
        # (4) the readout-adjacent layer: first task end with mean p < 0.01 and whether it reopens (mean p > 0.1 later)
        L = LAYERS[arm]
        pL = u[f"end_p_l{L}"].mean(1)
        ks = np.where(pL < 0.01)[0]
        ev = None
        if ks.size:
            k = int(ks[0])
            ev = {"task": k + 1, "A_before": float(u["A"][k - 1]) if k > 0 else None, "A_at": float(u["A"][k]),
                  "reopened_task": (int(np.where(pL[k:] > 0.1)[0][0]) + k + 1) if (pL[k:] > 0.1).any() else None,
                  "closed_tasks": int((pL < 0.01).sum())}
        # (5) who carries the next layer's input mean: share of |hbar_{l}|^2 from always-on units and from the top-2 units
        carry = {}
        runf = Path(a.runs) / (f.stem.replace("units_", "") + ".npz")
        if runf.exists() and L > 1:
            import sys
            sys.path.insert(0, str(Path(a.root))); sys.path.insert(0, str(Path(a.root) / "analysis" / "depth_silu_rlmnist_1020"))
            import measure as Mm
            from src import depth_silu_rlmnist_1020 as E
            d = np.load(runf)
            x64 = E.load_inputs(int(d["seed"]), str(d["door"]))[2].double().numpy()
            for t in (1, 5, 10, 20, 30, 50):
                P = [d[f"end{j}"][t - 1].astype(np.float64) for j in range(2 * L + 2)]
                hs, zs = Mm.forward64(P, x64, str(d["act"]), L)
                for l in range(1, L):                              # layer l's output = layer l+1's input
                    h = Mm.phi(zs[l - 1], str(d["act"])); hb = h.mean(0); tot = float((hb ** 2).sum())
                    on = (zs[l - 1] > 0).mean(0) > 0.99
                    top2 = np.sort(hb ** 2)[-2:].sum()
                    carry[f"t{t}_l{l}"] = {"hbar_norm": float(np.sqrt(tot)), "n_always_on": int(on.sum()),
                                           "share_always_on": float((hb[on] ** 2).sum() / tot) if tot > 0 else None,
                                           "share_top2": float(top2 / tot) if tot > 0 else None,
                                           "hbar_neg_share": float((hb[hb < 0] ** 2).sum() / tot) if tot > 0 else None}
        rec[f"{arm}_s{s}"] = {"arm": arm, "seed": s, "layers": r, "A": u["A"].tolist(), "readout_layer_close": ev, "carry": carry}
    (out / "posthoc.json").write_text(json.dumps(rec))
    # figure
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Sans CJK TC", "DejaVu Sans"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for j, arm in enumerate(["A1", "A2", "A4", "A4C", "R4"]):
        axx = ax.flat[j]
        for l in range(1, LAYERS[arm] + 1):
            ys = [rec[f"{arm}_s{s}"]["layers"][str(l)]["r_med"] for s in (0, 1, 2) if f"{arm}_s{s}" in rec]
            if not ys:
                continue
            ys = np.array(ys)
            t = np.arange(1, ys.shape[1] + 1)
            for y in ys:
                axx.plot(t, y, color=cols[l - 1], alpha=0.35, lw=0.8)
            axx.plot(t, np.nanmean(ys, 0), color=cols[l - 1], lw=2, label=f"第 {l} 層")
        axx.axhline(0, color="k", lw=0.5); axx.set_title(f"{arm}: 課題末の位置 r の中央値"); axx.legend(fontsize=8)
        axx.set_ylim(-4.5, 2.5)
    axx = ax.flat[5]
    for j, arm in enumerate(["A1", "A2", "A4", "A4C", "R1", "R4"]):
        ys = np.array([rec[f"{arm}_s{s}"]["A"] for s in (0, 1, 2) if f"{arm}_s{s}" in rec])
        if ys.size:
            axx.plot(np.arange(1, ys.shape[1] + 1), ys.mean(0), lw=2, label=arm)
    axx.set_title("オンライン精度（3 seed 平均）"); axx.legend(fontsize=8); axx.set_ylim(0, 1)
    for axx in ax[1]:
        axx.set_xlabel("課題")
    fig.tight_layout(); fig.savefig(out / "depth_layers.png", dpi=110)
    print("wrote", out / "posthoc.json", out / "depth_layers.png")


if __name__ == "__main__":
    main()

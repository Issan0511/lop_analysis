#!/usr/bin/env python3
"""Round 1b (spec_sink_roots_0930_round1b.md): the W1-cap follow-ups against the R3 base and round 1's R1 cap.
Ratio = R3's rho_med (units open at the switch; push m(200)-m(0), return m(T)-m(200)); accuracy = mean task-end accuracy
(all 1,200 images); all-closed = share of units with k = 0 at the task end.  Windows: tasks 5-30, and for the 200-task
arms also 5-200 and 151-200.  Descriptive per task (tasks 5-30, open units at the switch, medians): width change ds,
top change dtop, mean change dm."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_r3 as A

RAW, RES = A.RAW, A.RES


def rho(m, k, grid, tasks):
    g = list(grid); i0, i200, iT = g.index(0), g.index(200), len(g) - 1
    Pm, Rm = [], []
    for t in tasks:
        ti = t - 1; sel = k[ti, i0] > 0
        if not sel.any():
            continue
        Pm.append(np.median((m[ti, i200] - m[ti, i0])[sel])); Rm.append(np.median((m[ti, iT] - m[ti, i200])[sel]))
    if not Pm:
        return float("nan")
    mp, mr = np.median(Pm), np.median(Rm)
    return float(-mr / mp) if mp != 0 else float("nan")


def desc(name):
    a = np.load(RAW / name / "arrays.npz")
    m, k, s, top, g = a["u_m1"], a["u_k1"], a["u_s1"], a["u_top1"], a["grid"]
    nt = m.shape[0]
    out = {"tasks": nt}
    wins = [(5, 30)] + ([(5, 200), (151, 200)] if nt >= 200 else [])
    for lo, hi in wins:
        ti = [t - 1 for t in range(lo, hi + 1)]
        out[f"rho_{lo}_{hi}"] = rho(m, k, g, range(lo, hi + 1)) if 200 in list(g) else float("nan")
        out[f"acc_{lo}_{hi}"] = float(np.mean(a["s_acc"][ti, -1]))
        out[f"allclosed_{lo}_{hi}"] = float(np.mean(k[ti, -1] == 0))
    i0 = list(g).index(0)
    ds, dtop, dm = [], [], []
    for t in range(5, 31):
        ti = t - 1; sel = k[ti, i0] > 0
        ds.append(np.median((s[ti, -1] - s[ti, i0])[sel])); dtop.append(np.median((top[ti, -1] - top[ti, i0])[sel]))
        dm.append(np.median((m[ti, -1] - m[ti, i0])[sel]))
    out.update(ds_task=float(np.median(ds)), dtop_task=float(np.median(dtop)), dm_task=float(np.median(dm)))
    return out


GROUPS = [("R3_base", "GELU"), ("R1b1_wcap", "GELU"), ("R3_base", "LR"), ("R1b1_wcap", "LR"),
          ("R3_base", "ELU"), ("R1wcap", "ELU"), ("R1b3_par", "ELU"), ("R1b3_perp", "ELU"),
          ("R1b4_scale0.5", "ELU"), ("R1b4_scale2", "ELU"), ("R1b4_scale4", "ELU"), ("R1b2_long", "ELU"), ("R5_main", "ELU"),
          ("R3_base", "SILU"), ("R1wcap", "SILU"), ("R1b3_par", "SILU"), ("R1b3_perp", "SILU"), ("R1b2_long", "SILU")]


def main():
    out = {}
    for tag, act in GROUPS:
        rows = []
        for s in (0, 1, 2):
            n = f"{tag}_{act}_s{s}"
            if (RAW / n / "provenance.json").exists():
                out[n] = desc(n); rows.append(out[n])
        if not rows:
            continue
        keys = [k for k in rows[0] if k != "tasks"]
        f = lambda k: " ".join(f"{r[k]:.3f}" for r in rows)
        print(f"{act:4s} {tag:14s} n{len(rows)} | " + " | ".join(f"{k} {f(k)}" for k in keys if k.startswith(("rho", "acc", "allclosed")))
              + f" | ds {f('ds_task')} dtop {f('dtop_task')} dm {f('dm_task')}")
    (RES / "round1b_wcap.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

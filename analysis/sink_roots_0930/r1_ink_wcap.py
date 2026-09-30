#!/usr/bin/env python3
"""Round 1 return R6 (ink normalize) and R1 (W1 row-norm cap from task 5): the R3 ratio (rho_med, open units, tasks 5-30)
against the R3 base, plus descriptive width and top changes per task (open units at the switch, medians)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_r3 import ratios, RAW, TASKS

RES = Path(__file__).resolve().parents[2] / "results" / "sink_roots_0930"


def desc(name):
    a = np.load(RAW / name / "arrays.npz")
    m, k, s, top, g = a["u_m1"], a["u_k1"], a["u_s1"], a["u_top1"], list(a["grid"])
    i0, i200 = g.index(0), g.index(200)
    r = ratios(m, k, a["grid"], len(g) - 1)
    ds, dtop, dm, allc = [], [], [], []
    for t in TASKS:
        ti = t - 1; sel = k[ti, i0] > 0
        ds.append(np.median((s[ti, -1] - s[ti, i0])[sel])); dtop.append(np.median((top[ti, -1] - top[ti, i0])[sel]))
        dm.append(np.median((m[ti, -1] - m[ti, i0])[sel])); allc.append(np.mean(k[ti, -1] == 0))
    r.update({"ds_task": float(np.median(ds)), "dtop_task": float(np.median(dtop)), "dm_task": float(np.median(dm)),
              "allclosed_end": float(np.mean(allc)), "acc_end": float(np.mean(a["s_acc"][4:30, -1]))})
    return r


def main():
    out = {}
    for tag, acts in (("R3_base", ("ELU", "LR", "SILU")), ("R6ink", ("ELU", "LR", "SILU")), ("R1wcap", ("ELU", "SILU"))):
        for act in acts:
            for s in (0, 1, 2):
                n = f"{tag}_{act}_s{s}"
                if (RAW / n / "provenance.json").exists():
                    out[n] = desc(n)
    for act in ("ELU", "LR", "SILU"):
        for tag in ("R3_base", "R6ink", "R1wcap"):
            rr = [out.get(f"{tag}_{act}_s{s}") for s in (0, 1, 2)]
            if not all(rr):
                continue
            f = lambda k: np.round([r[k] for r in rr], 3).tolist()
            print(f"{act:4s} {tag:8s} rho_med(open) {f('rho_med_open')} | push {f('push_open')} ret {f('ret_open')} | "
                  f"ds/task {f('ds_task')} dtop/task {f('dtop_task')} dm/task {f('dm_task')} | allclosed {f('allclosed_end')} acc {f('acc_end')}")
    (RES / "round1_R6_R1.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

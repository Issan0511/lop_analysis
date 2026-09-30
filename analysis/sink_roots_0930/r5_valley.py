#!/usr/bin/env python3
"""Round 5 R2e_valley_long_eps (spec_sink_roots_0930_round5.md §2), with round 2's R2_valley_long (eps1 1e-8) alongside.
Closed = the unit's top at the task end <= 0.  A closed period = a maximal run of closed task ends of one unit; it ends
(an event) when the unit is open at a later task end, and is censored at the last task.  Wall = the eps arm's lowest closed
top measured in round 1's R_eps_wall (1e-6 -6.31, 1e-8 -6.97, 1e-12 -8.13); a period 'touches the wall' if its top gets
within 0.25 of it (decided before these runs finished).  S_C(10) = Kaplan-Meier probability of still being closed 10 tasks
after the start; the hazard at age >= 100 = reopenings / unit-tasks at risk among periods that reached age 100."""
import glob, re
from pathlib import Path
import numpy as np

RAW = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930")
RES = Path(__file__).resolve().parents[2] / "results" / "sink_roots_0930"
WALL = {"1e-6": -6.31, "1e-8": -6.97, "1e-12": -8.13}


def periods(top):
    """(unit, start task index, length, event, min top) for every closed period."""
    out = []
    T, H = top.shape
    for u in range(H):
        t = 0
        while t < T:
            if top[t, u] <= 0:
                s = t
                while t < T and top[t, u] <= 0:
                    t += 1
                out.append((u, s, t - s, t < T, float(top[s:t, u].min())))
            else:
                t += 1
    return out


def km_at(lengths, events, x):
    lengths, events = np.asarray(lengths), np.asarray(events, bool)
    S = 1.0
    for d in range(1, x + 1):
        at_risk = np.sum(lengths >= d)
        ev = np.sum((lengths == d) & events)
        if at_risk:
            S *= 1 - ev / at_risk
    return S


def main():
    L = []
    files = {("1e-8", s): RAW / "round2" / "valley_long" / f"GELU_s{s}_v4.npz" for s in (0, 1)}
    files.update({(e, s): RAW / "round5" / "R2e" / f"GELU_eps{e}_s{s}_v4.npz" for e in ("1e-6", "1e-12") for s in (0, 1)})
    for (eps, s), f in sorted(files.items()):
        if not f.exists():
            continue
        top = np.load(f)["task_zmax"].astype(float); T = top.shape[0]
        ac = lambda lo, hi: float((top[lo - 1:hi] <= 0).mean())
        rates = []
        for lo in range(1, T, 200):
            hi = min(lo + 199, T); oc = oo = co = cc = 0
            for t in range(lo - 1, min(hi, T - 1)):
                x, y = top[t] > 0, top[t + 1] > 0
                oc += (x & ~y).sum(); oo += x.sum(); co += (~x & y).sum(); cc += (~x).sum()
            rates.append((lo, hi, ac(lo, hi), oc / max(oo, 1), co / max(cc, 1)))
        P = periods(top); wall = WALL[eps]
        L.append(f"## GELU eps1 {eps} seed {s} ({T} tasks): closed periods {len(P)}; lowest closed top {top[top <= 0].min():.2f}")
        L.append("   windows: " + " | ".join(f"t{lo}-{hi} all-closed {a:.3f} mu+ {mp:.3f} mu- {mm:.3f}" for lo, hi, a, mp, mm in rates))
        if T >= 1000:
            L.append(f"   all-closed 151-200 {ac(151, 200):.3f}  801-1000 {ac(801, 1000):.3f}  (difference {ac(801, 1000) - ac(151, 200):+.3f})")
        for touched in (False, True):
            row = []
            for e0 in range(0, T, 100):
                sel = [(l, ev) for u, st, l, ev, mn in P if e0 <= st < e0 + 100 and ((mn <= wall + 0.25) == touched)]
                if len(sel) >= 20:
                    row.append(f"start {e0 + 1}-{e0 + 100}: S_C(10) {km_at([l for l, _ in sel], [ev for _, ev in sel], 10):.3f} (n {len(sel)})")
            L.append(f"   {'touching' if touched else 'not touching'} the wall: " + " | ".join(row))
            if not touched and len(row) >= 3:                       # (c): rank correlation of the start bin and S_C(10)
                sc = [float(x.split("S_C(10) ")[1].split(" ")[0]) for x in row]
                rk = lambda v: np.argsort(np.argsort(v)).astype(float)
                rho = float(np.corrcoef(rk(np.arange(len(sc))), rk(np.array(sc)))[0, 1])
                L.append(f"      (c) Spearman(start bin, S_C(10)) over {len(sc)} bins {rho:+.2f}; first bin {sc[0]:.3f}, last bin {sc[-1]:.3f}")
        old = [(l, ev) for u, st, l, ev, mn in P if l >= 100]
        at_risk = sum(l - 100 for l, ev in old) + sum(1 for l, ev in old if ev)
        events = sum(1 for l, ev in old if ev)
        L.append(f"   age >= 100: periods {len(old)}, reopenings {events}, hazard per task {events / max(at_risk, 1):.4f}" + ("" if events >= 30 else " (fewer than 30 events: not reported as a rate)"))
    txt = "\n".join(L); print(txt)
    (RES / "round5_R2e.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()

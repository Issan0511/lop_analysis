#!/usr/bin/env python3
"""Registered verdicts for layer_chimera_cifar_0921 (spec §5).

    python3 analysis/layer_chimera_cifar_0921/report.py            # -> verdict.json, summary.md

Reads results/<run>/<CELL>/per_task.csv and results/<run>/measure.csv only.  Every threshold
comes from the spec: 0.5 is the parent battle's registered collapse rule, the floor is the
label draw's best constant predictor + 3 s/sqrt(20), the bands are seed ranges of a parent cell.
"""
from __future__ import annotations

import argparse
import json
import sys
from math import comb, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import layer_chimera_cifar_0921 as C
from src import pmnist_0905 as H
from src import pmnist_rlcifar_0907 as RC

WIN, EARLY, LATE = (31, 50), (1, 10), (41, 50)
SEEDS = list(range(10))
CELLS = list(C.CELL_ORDER)
MAIN = ["LL", "EL", "GL", "LE", "EE", "GE"]                     # GG is the reference (spec §3)


def sign(diff) -> tuple[int, int, float]:
    """Exact two-sided sign test, zeros dropped (the parent battle's rule)."""
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    nz = d[d != 0]
    n = len(nz)
    pos = int((nz > 0).sum())
    k = min(pos, n - pos)
    p = min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0) if n else float("nan")
    return pos, n, p


def call(diff) -> dict:
    pos, n, p = sign(diff)
    win = (n - pos) <= 1 and p < 0.05 and n > 0
    lose = pos <= 1 and p < 0.05 and n > 0
    return {"median": float(np.median(diff)), "pos": pos, "n": n, "p": p,
            "call": "A_WINS" if win else "B_WINS" if lose else "TIE"}


def floors() -> dict:
    """spec §5.1: thr_s = F_s + 3 s_s / sqrt(20) over the window's own label draws."""
    out = {}
    for s in SEEDS:
        g = H.stream("rlc_labels", s)
        f = []
        for t in range(1, WIN[1] + 1):
            y = RC.task_labels(g)
            f.append(float(torch.bincount(y, minlength=RC.N_CLASSES).max()) / RC.N_IMAGES)
        w = np.array(f[WIN[0] - 1:WIN[1]])
        out[s] = {"F": float(w.mean()), "sd": float(w.std(ddof=1)),
                  "thr": float(w.mean() + 3 * w.std(ddof=1) / sqrt(len(w)))}
    return out


def per_run(src: Path) -> pd.DataFrame:
    rows = []
    for cell in CELLS:
        f = src / cell / "per_task.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, float_precision="round_trip")
        d = d[d.cond == "raw"]
        for seed, g in d.groupby("seed"):
            ok = g[g.online_acc.notna()]
            def mean(lo, hi):
                v = ok[(ok.task >= lo) & (ok.task <= hi)].online_acc
                return float(v.mean()) if len(v) else float("nan")
            t1 = ok[ok.task == 1].online_acc
            below = ok[ok.online_acc < 0.5].task
            rows.append({"cell": cell, "seed": int(seed), "window": mean(*WIN),
                         "early": mean(*EARLY), "late": mean(*LATE),
                         "t1_online": float(t1.iloc[0]) if len(t1) else float("nan"),
                         "T_A": int(below.min()) if len(below) else 0,
                         "tasks": int(ok.task.max()), "diverged": bool(g.online_acc.isna().any())})
    r = pd.DataFrame(rows)
    r["drop"] = r.early - r.late
    return r


def stage_and_type(r: pd.DataFrame, fl: dict) -> pd.DataFrame:
    thr = r.seed.map(lambda s: fl[s]["thr"])
    r = r.copy()
    r["thr"] = thr
    r["margin"] = r.window - thr
    r["stage"] = np.where(r.window >= 0.5, "ALIVE",
                          np.where(r.window <= thr, "AT_FLOOR", "PARTIAL"))
    learns = r.t1_online >= 0.5
    r["type"] = [
        {"ALIVE": "SLOW_SINK" if L else "LATE_START",
         "PARTIAL": "LEARNS_THEN_PARTIAL" if L else "LOW_FROM_T1_PARTIAL",
         "AT_FLOOR": "LEARNS_THEN_DIES" if L else "DEAD_FROM_T1"}[st]
        for st, L in zip(r.stage, learns)]
    return r


def majority(labels: list[str], k: int = 6) -> str:
    vals, cnt = np.unique(labels, return_counts=True)
    i = int(cnt.argmax())
    return str(vals[i]) if cnt[i] >= k else "SPLIT"


def q1(types: dict) -> dict:
    """spec §5.2, applied in order."""
    t = types
    rows = {a1: (t[a1 + "L"], t[a1 + "E"]) for a1 in "LEG"}
    if len(set(t[c] for c in MAIN)) == 1:
        lab = "NO_SPLIT"
    elif all(x == y for x, y in rows.values()):
        lab = "L1_DECIDES"
    elif (len({t["LL"], t["EL"], t["GL"]}) == 1 and len({t["LE"], t["EE"], t["GE"]}) == 1
          and t["LL"] != t["LE"]):
        lab = "L2_DECIDES"
    elif (t["LL"] == t["EL"] == "SLOW_SINK"
          and t["LE"] in ("LEARNS_THEN_DIES", "LEARNS_THEN_PARTIAL")
          and t["EE"] in ("LEARNS_THEN_DIES", "LEARNS_THEN_PARTIAL")
          and t["GL"] == t["GE"] == "DEAD_FROM_T1"):
        lab = "BOTH_LAYERS_THREE_TYPES"
    else:
        lab = "MIXED"
    return {"label": lab, "types": t}


def paired(r: pd.DataFrame, a: str, b: str, col="window") -> dict:
    x = r[r.cell == a].set_index("seed")[col]
    y = r[r.cell == b].set_index("seed")[col]
    common = sorted(set(x.index) & set(y.index))
    return {"pair": f"{a} - {b}", **call([x[s] - y[s] for s in common])}


def q2(r: pd.DataFrame, m: pd.DataFrame) -> dict:
    """spec §5.3a: at the first task with online < 0.5, which layer is deeper (D_abs)."""
    out = {}
    for cell in CELLS:
        labs, detail = [], []
        for _, row in r[r.cell == cell].iterrows():
            if not row.T_A:
                labs.append("NO_LOSS"); detail.append({"seed": int(row.seed), "T_A": None}); continue
            g = m[(m.cell == cell) & (m.seed == row.seed) & (m.task == row.T_A)]
            d1 = float(g[g.layer == 1].D_abs.iloc[0]); d2 = float(g[g.layer == 2].D_abs.iloc[0])
            lab = "DEEPER_TIE" if d1 == d2 else ("DEEPER_L1_AT_LOSS" if d1 > d2 else "DEEPER_L2_AT_LOSS")
            labs.append(lab)
            detail.append({"seed": int(row.seed), "T_A": int(row.T_A), "D1": d1, "D2": d2, "label": lab})
        out[cell] = {"label": majority(labs), "seeds": detail}
    return out


def q3(m: pd.DataFrame) -> dict:
    mu = m[(m.layer == 2)].pivot_table(index=["cell", "seed"], columns="task", values="mu_norm")
    def at(cell, t):
        if t not in mu.columns:
            return {}
        v = {s: float(mu.loc[(cell, s), t]) for s in SEEDS if (cell, s) in mu.index}
        return {s: x for s, x in v.items() if np.isfinite(x)}
    ee, el = at("EE", 10), at("EL", 10)
    common = sorted(set(ee) & set(el))
    p3a = call([ee[s] - el[s] for s in common])
    p3a["non_overlapping"] = bool(common and max(el[s] for s in common) < min(ee[s] for s in common))
    below = {}
    for cell in ("GL", "GE", "GG"):
        t10, t0 = at(cell, 10), at(cell, 0)
        ok = [s for s in sorted(set(t10) & set(t0)) if t10[s] <= t0[s]]
        below[cell] = {"seeds_below_init": len(ok), "n": len(t10),
                       "label": "MU2_BELOW_INIT" if len(ok) >= 9 else "NOT_BELOW_INIT",
                       "median_t10": float(np.median(list(t10.values()))) if t10 else None,
                       "median_t00": float(np.median(list(t0.values()))) if t0 else None}
    levels = {cell: {f"t{t}": float(np.median(list(at(cell, t).values()) or [np.nan]))
                     for t in (0, 1, 2, 10, 50)} for cell in CELLS
              if any((cell, s) in mu.index for s in SEEDS)}
    return {"P3a_EE_minus_EL_t10": p3a, "P3b": below, "mu2_median_levels": levels}


def q4(m: pd.DataFrame) -> dict:
    """spec §5.5: is layer 1 the same whatever sits downstream?  Bands are the parent's seed range."""
    out = {}
    for col, (a, b, ref) in {"GELU": ("GL", "GE", "GG"), "ELU": ("EL", "EE", "EE")}.items():
        det, verdict = {}, "L1_INDEPENDENT_OF_L2"
        for x in ("p_pos", "d_signed", "Q"):
            for t in (1, 2, 10):
                def vals(cell):
                    g = m[(m.cell == cell) & (m.layer == 1) & (m.task == t)]
                    return {int(s): float(v) for s, v in zip(g.seed, g[x])}
                va, vb, vr = vals(a), vals(b), vals(ref)
                if not (va and vb and vr):
                    continue
                band = max(vr.values()) - min(vr.values())
                common = sorted(set(va) & set(vb))
                d_ab = float(np.median([va[s] - vb[s] for s in common]))
                d_ar = float(np.median([va[s] - vr[s] for s in sorted(set(va) & set(vr))]))
                ok = abs(d_ab) <= band and abs(d_ar) <= band
                det[f"{x}@t{t}"] = {"band": band, f"med({a}-{b})": d_ab, f"med({a}-{ref})": d_ar,
                                    "within": bool(ok)}
                if not ok:
                    verdict = "L1_DEPENDS_ON_L2"
        out[col] = {"label": verdict, "detail": det, "reference": ref}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(C.OUT_ROOT))
    ap.add_argument("--out", default=str(C.OUT_ROOT))
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    fl = floors()
    r = stage_and_type(per_run(src), fl)
    m = pd.read_csv(out / "measure.csv", float_precision="round_trip")
    m = m[m.cond == "raw"]
    types = {c: majority(list(r[r.cell == c].type)) for c in CELLS if (r.cell == c).any()}
    stages = {c: majority(list(r[r.cell == c].stage)) for c in types}
    v = {"run_id": C.EXPERIMENT, "window": WIN, "floors": fl,
         "cell_window_median": {c: float(r[r.cell == c].window.median()) for c in types},
         "cell_drop_median": {c: float(r[r.cell == c]["drop"].median()) for c in types},
         "cell_t1_median": {c: float(r[r.cell == c].t1_online.median()) for c in types},
         "stage": stages, "type": types,
         "Q1": q1(types) if all(c in types for c in MAIN) else {"label": "INAPPLICABLE"},
         "C1": [paired(r, x, y) for x, y in (("EL", "EE"), ("GL", "GE"), ("LL", "LE"))],
         "C2": [paired(r, x, y) for x, y in (("GL", "LL"), ("GE", "LE"), ("EL", "LL"), ("EE", "LE"))],
         "C3": [paired(r, x, y) for x, y in (("LL", "EL"), ("EL", "GL"))],
         "Q2_deeper_at_loss": q2(r, m), "Q3": q3(m), "Q4": q4(m),
         "Q5_timing": {c: {"T_A_median": float(np.median([t for t in r[r.cell == c].T_A if t])
                                                or np.nan) if (r[r.cell == c].T_A > 0).any() else None}
                       for c in types}}
    (out / "verdict.json").write_text(json.dumps(v, indent=2, default=float))
    r.to_csv(out / "per_seed.csv", index=False)
    lines = ["# layer_chimera_cifar_0921 -- 層別キメラ (S-A) の登録判定", "",
             f"窓 = t{WIN[0]}-{WIN[1]} の online 平均。raw のみ・R=10・seed 0-9。", "",
             "| cell | act1 | act2 | 窓(中央値) | t1 online | 低下 | 段 | 型 | T_A 中央値 |",
             "|---|---|---|---:|---:|---:|---|---|---:|"]
    for c in CELLS:
        if c not in types:
            continue
        ta = v["Q5_timing"][c]["T_A_median"]
        lines.append(f"| `{c}` | {C.CELLS[c][0]} | {C.CELLS[c][1]} | {v['cell_window_median'][c]:.4f} | "
                     f"{v['cell_t1_median'][c]:.4f} | {v['cell_drop_median'][c]:+.4f} | {stages[c]} | "
                     f"{types[c]} | {'-' if ta is None else f'{ta:.0f}'} |")
    lines += ["", f"**Q1 主判定: `{v['Q1']['label']}`**", ""]
    for key in ("C1", "C2", "C3"):
        lines.append(f"## {key}")
        for q in v[key]:
            lines.append(f"- {q['pair']}: median {q['median']:+.4f}, {q['pos']}/{q['n']} seeds, "
                         f"p={q['p']:.4f} -> {q['call']}")
        lines.append("")
    lines.append("## Q2 機能喪失時点で深い層")
    for c, q in v["Q2_deeper_at_loss"].items():
        lines.append(f"- `{c}`: {q['label']}")
    lines += ["", "## Q3 担い手 ||mu2||",
              f"- EE - EL @t10: median {v['Q3']['P3a_EE_minus_EL_t10']['median']:+.3f}, "
              f"{v['Q3']['P3a_EE_minus_EL_t10']['pos']}/{v['Q3']['P3a_EE_minus_EL_t10']['n']}, "
              f"{v['Q3']['P3a_EE_minus_EL_t10']['call']}, "
              f"non_overlapping={v['Q3']['P3a_EE_minus_EL_t10']['non_overlapping']}"]
    for c, q in v["Q3"]["P3b"].items():
        lines.append(f"- `{c}`: {q['label']} ({q['seeds_below_init']}/{q['n']}, "
                     f"t00 {q['median_t00']:.3f} -> t10 {q['median_t10']:.3f})")
    lines += ["", "## Q4 第1層は下流に依らないか"]
    for col, q in v["Q4"].items():
        lines.append(f"- {col} 列: **{q['label']}**（帯は `{q['reference']}` の seed 範囲）")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

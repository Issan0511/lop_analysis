#!/usr/bin/env python3
"""R3 (spec §2): push / return per run, the ratios rho_med and rho_unit, and the registered labels."""
import csv, json, sys
from pathlib import Path
import numpy as np

RAW = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930/mnist")
RES = Path(__file__).resolve().parents[2] / "results" / "sink_roots_0930"
ARMS = ["base", "b2_099", "b2_09", "T1k", "T16k", "adamreset", "vrestore", "ls01", "sq003", "bwfloor", "bwabs", "bwrelu"]   # bwrelu: round 2
ACTS = ["ELU", "GELU", "SILU", "LR"]
SEEDS = [0, 1, 2]
TASKS = range(5, 31)


def ratios(m, k, grid, iT):
    g = list(grid)
    i0, i200 = g.index(0), g.index(200)
    out = {}
    for uset in ("open", "all"):
        Pm, Rm, pu = [], [], []
        for t in TASKS:
            ti = t - 1
            P = m[ti, i200] - m[ti, i0]
            R = m[ti, iT] - m[ti, i200]
            sel = k[ti, i0] > 0 if uset == "open" else np.ones_like(P, dtype=bool)
            if not sel.any():
                continue
            Pm.append(np.median(P[sel])); Rm.append(np.median(R[sel]))
            s2 = sel & (P < 0)
            pu.extend(list(-R[s2] / P[s2]))
        mp, mr = np.median(Pm), np.median(Rm)
        out[f"rho_med_{uset}"] = float(-mr / mp) if mp != 0 else np.nan
        out[f"rho_unit_{uset}"] = float(np.median(pu)) if pu else np.nan
        out[f"push_{uset}"] = float(mp)
        out[f"ret_{uset}"] = float(mr)
    return out


def summarize(name):
    d = RAW / name
    if not (d / "provenance.json").exists():
        return None
    a = np.load(d / "arrays.npz")
    m, k, grid = a["u_m1"], a["u_k1"], a["grid"]
    if m.shape[0] < 30:
        return None
    g = list(grid)
    row = {"name": name}
    row.update(ratios(m, k, grid, len(g) - 1))
    if g[-1] == 16000:
        r4 = ratios(m, k, grid, g.index(4000))
        row.update({f"{kk}_at4000": v for kk, v in r4.items()})
    ti = [t - 1 for t in TASKS]
    i0 = g.index(0)
    row["net_med"] = float(np.median([np.median(m[t, -1] - m[t, i0]) for t in ti]))
    row["p_old_sw"] = float(np.nanmean(a["s_p_old"][ti, i0]))
    row["margin_old_sw"] = float(np.nanmean(a["s_margin_old_med"][ti, i0]))
    row["pmax_end"] = float(np.mean(a["s_pmax"][ti, -1]))
    row["acc_end"] = float(np.mean(a["s_acc"][ti, -1]))
    row["pplus_t30"] = float(k[29, -1].mean() / 1200)
    row["allclosed_t30"] = float((k[29, -1] == 0).mean())
    row["top_t30"] = float(np.median(a["u_top1"][29, -1]))
    return row


def label(deltas):
    deltas = [x for x in deltas if x is not None and np.isfinite(x)]
    if len(deltas) < 3:
        return "INCOMPLETE"
    med = float(np.median(deltas))
    if all(x > 0 for x in deltas) and med >= 0.10:
        return "UP"
    if all(x < 0 for x in deltas) and med <= -0.10:
        return "DOWN"
    if abs(med) < 0.05:
        return "FLAT"
    return "MIXED"


def main():
    rows = {}
    for arm in ARMS:
        for act in ACTS:
            for s in SEEDS:
                r = summarize(f"R3_{arm}_{act}_s{s}")
                if r:
                    r.update({"arm": arm, "act": act, "seed": s})
                    rows[(arm, act, s)] = r
    RES.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(kk for r in rows.values() for kk in r))
    with open(RES / "R3_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows.values())
    labels = {}
    print(f"{'act':5s} {'arm':10s} rho_med(open) per seed         | rho_unit(open) | rho_med(all) | push   ret    | label")
    for act in ACTS:
        for arm in ARMS:
            rr = [rows.get((arm, act, s)) for s in SEEDS]
            if not any(rr):
                continue
            base = [rows.get(("base", act, s)) for s in SEEDS]
            rm = [r["rho_med_open"] if r else np.nan for r in rr]
            deltas = [(r["rho_med_open"] - b["rho_med_open"]) if (r and b) else None for r, b in zip(rr, base)]
            lab = label(deltas) if arm != "base" else "-"
            extra = ""
            if arm == "T16k" and all(rr):
                dd = [r["rho_med_open"] - r["rho_med_open_at4000"] for r in rr]
                tl = "TIMEOUT" if all(x >= 0.10 for x in dd) else "no"
                extra = f" | rho(16k)-rho(4k) {np.round(dd, 3)} {tl}"
                labels[f"{act}|T16k_within"] = tl
            labels[f"{act}|{arm}"] = lab
            ru = [r["rho_unit_open"] if r else np.nan for r in rr]
            ra = [r["rho_med_all"] if r else np.nan for r in rr]
            pu = np.nanmedian([r["push_open"] for r in rr if r]); re_ = np.nanmedian([r["ret_open"] for r in rr if r])
            print(f"{act:5s} {arm:10s} {np.round(rm, 3)} | {np.round(ru, 3)} | {np.round(ra, 3)} | {pu:+.2f} {re_:+.2f} | {lab}{extra}")
    (RES / "R3_labels.json").write_text(json.dumps(labels, indent=1))
    # time shape of the return (descriptive): -median_t median_i (m(s) - m(200)) / median_t median_i P, units open at switch
    shape_rows = []
    print("\ntime shape: return fraction -R(s)/P at s (ratio of medians over tasks 5-30, open units)")
    for act in ACTS:
        for arm in ("base", "T16k", "b2_09", "vrestore", "ls01", "sq003", "adamreset"):
            for s_ in SEEDS:
                d = RAW / f"R3_{arm}_{act}_s{s_}"
                if not (d / "provenance.json").exists():
                    continue
                a = np.load(d / "arrays.npz")
                m, k, g = a["u_m1"], a["u_k1"], list(a["grid"])
                i0, i200 = g.index(0), g.index(200)
                pts = [x for x in (500, 1000, 2000, 3000, 4000, 8000, 12000, 16000) if x in g]
                Pm = np.median([np.median((m[t - 1, i200] - m[t - 1, i0])[k[t - 1, i0] > 0]) for t in TASKS])
                row = {"act": act, "arm": arm, "seed": s_}
                for x in pts:
                    Rm = np.median([np.median((m[t - 1, g.index(x)] - m[t - 1, i200])[k[t - 1, i0] > 0]) for t in TASKS])
                    row[f"frac_{x}"] = float(-Rm / Pm)
                shape_rows.append(row)
            rr = [r for r in shape_rows if r["act"] == act and r["arm"] == arm]
            if rr:
                keys = [kk for kk in rr[0] if kk.startswith("frac_")]
                print(f"{act:5s} {arm:10s}", {kk[5:]: [round(r[kk], 2) for r in rr] for kk in keys})
    keys = list(dict.fromkeys(kk for r in shape_rows for kk in r))
    with open(RES / "R3_time_shape.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(shape_rows)


if __name__ == "__main__":
    main()

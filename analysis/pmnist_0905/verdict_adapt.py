"""pmnist_adapt_0905 — spec §3 judgements Q1-Q4, per box and combined.

Usage: verdict_adapt.py [boxA] [boxB]   (omit a box to skip it)
Rows from */l2init/ and */cbp/ are relabelled R+l2init / R+cbp (run_one does not
write the --iv flag into per_task.csv; the directory and provenance.json carry it).
"""
import sys, glob, json
from math import comb
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("results/pmnist_adapt_0905"); WIN = (151, 200); BAND = (0.50, 0.85)
OUT_OF_BAND = {"boxA": "SN1", "boxB": "SN3"}; IN_BAND = {"boxA": "SN02", "boxB": "SN1"}


def load(box):
    parts = []
    for f in glob.glob(str(ROOT / box / "**" / "per_task.csv"), recursive=True):
        d = pd.read_csv(f); sub = Path(f).parent.name
        # run_one does not write --iv into per_task.csv; the directory carries it.
        relabel = {"l2init": "R+l2init", "cbp": "R+cbp",
                   "sna_l2": "SNA+l2", "r_l2": "R+l2", "sna_l2init": "SNA+l2init"}
        if sub in relabel: d["arm"] = relabel[sub]
        parts.append(d)
    d = pd.concat(parts); return d[d.acc.notna()]


def sign(diff):
    d = np.asarray(diff, float); nz = d[d != 0]; n = len(nz); pos = int((nz > 0).sum())
    k = min(pos, n - pos); p = min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0) if n else np.nan
    return pos, n, p


def win(d, arm, col="acc"):
    g = d[(d.arm == arm) & (d.task >= WIN[0]) & (d.task <= WIN[1])]
    full = [s for s, gg in g.groupby("seed") if len(gg) == WIN[1] - WIN[0] + 1]
    return g[g.seed.isin(full)].groupby("seed")[col].mean()


def pair(d, a, b):
    x, y = win(d, a), win(d, b); s = sorted(set(x.index) & set(y.index))
    if not s: return None
    diff = (x[s] - y[s]).values; pos, n, p = sign(diff)
    return dict(mean=diff.mean(), se=diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.nan,
                wins=pos, n=n, p=p, sig_win=(pos >= 9 and p < 0.05), sig_loss=(pos <= 1 and p < 0.05 and n >= 10))


def box_report(box):
    d = load(box); L = []; V = {}
    arms = [a for a in ("SNA", "SN02", "SN1", "SN3", "LR", "R", "LIN", "R+l2init", "R+cbp",
                        "SNA+l2", "R+l2", "SNA+l2init") if a in set(d.arm)]
    L.append(f"### {box}\n")
    L.append(f"| arm | n | acc 151-200 | deg (21-70 − 151-200) | mob L1 | zbar L1 | 2αW L1 | clip% |")
    L.append("|---|---|---|---|---|---|---|---|")
    for a in arms:
        rows = [g.sort_values("task") for _, g in d[d.arm == a].groupby("seed") if g.task.max() == 200]
        if not rows: continue
        m = np.median
        acc = m([r[(r.task >= 151)].acc.mean() for r in rows])
        deg = m([r[(r.task >= 21) & (r.task <= 70)].acc.mean() - r[(r.task >= 151)].acc.mean() for r in rows])
        mob = m([r[r.task >= 151].mob_l1.mean() for r in rows]); zb = m([r[r.task >= 151].zbar_l1.mean() for r in rows])
        taw = m([r[r.task >= 151].two_alpha_W_med_l1.mean() for r in rows]) if "two_alpha_W_med_l1" in rows[0] and rows[0].two_alpha_W_med_l1.notna().any() else np.nan
        clip = m([r[r.task >= 151].alpha_clip_frac_l1.mean() for r in rows]) if "alpha_clip_frac_l1" in rows[0] and rows[0].alpha_clip_frac_l1.notna().any() else np.nan
        V[a] = dict(acc=acc, deg=deg, mob=mob, zbar=zb, taw=taw, clip=clip, n=len(rows))
        L.append(f"| {a} | {len(rows)} | {acc:.4f} | {deg:+.4f} | {mob:.4f} | {zb:+.3f} | {taw if np.isnan(taw) else f'{taw:.2f}'} | {'' if np.isnan(clip) else f'{100*clip:.1f}'} |")
    # Q1 pieces
    q1 = None
    if "SNA" in V:
        q1 = dict(in_band=BAND[0] <= V["SNA"]["mob"] <= BAND[1], clip_ok=(V["SNA"]["clip"] < 0.05))
    # pairs
    P = {}
    for opp in ("LR", "R", "LIN", "SN02", "SN1", "SN3", "R+l2init", "R+cbp", "SNA+l2", "R+l2", "SNA+l2init"):
        if opp in V and "SNA" in V and opp != "SNA": P[opp] = pair(d, "SNA", opp)
    L.append(f"\nSNA − opponent, tasks 151-200, paired by seed:\n")
    L.append("| vs | mean ± SE | SNA wins | p |"); L.append("|---|---|---|---|")
    for opp, r in P.items():
        if r: L.append(f"| {opp} | {r['mean']:+.4f} ± {r['se']:.4f} | {r['wins']}/{r['n']} | {r['p']:.4f} |")
    return d, V, P, q1, "\n".join(L)


def main():
    boxes = [b for b in (sys.argv[1:] or ["boxA", "boxB"]) if (ROOT / b).exists()]
    R = {b: box_report(b) for b in boxes}
    out = ["# pmnist_adapt_0905 — verdict\n"]
    for b in boxes: out.append(R[b][4]); out.append("")
    labels = {}
    # Q1
    q1s = {b: R[b][3] for b in boxes if R[b][3]}
    if len(q1s) == 2:
        ok = [v["in_band"] and v["clip_ok"] for v in q1s.values()]
        labels["Q1"] = "ALPHA_SELF_SETS" if all(ok) else ("ALPHA_BOX_DEPENDENT" if any(ok) else "ALPHA_NOT_SELF_SETTING")
    # Q2
    if len(boxes) == 2 and all(OUT_OF_BAND[b] in R[b][2] and IN_BAND[b] in R[b][2] for b in boxes):
        beats_out = all(R[b][2][OUT_OF_BAND[b]]["sig_win"] for b in boxes)
        loses_in = any(R[b][2][IN_BAND[b]]["sig_loss"] for b in boxes)
        labels["Q2"] = ("GATE_BAND_CAUSAL_SUPPORTED" if beats_out and not loses_in else
                        "ADAPTIVE_COSTS" if beats_out else "GATE_BAND_NOT_CAUSAL")
    # Q3
    if len(boxes) == 2 and all("LR" in R[b][2] for b in boxes):
        w = [R[b][2]["LR"]["sig_win"] for b in boxes]
        labels["Q3"] = ("SNAKE_RETAINS_BETTER_BOTH" if all(w) else
                        f"SNAKE_RETAINS_BETTER_ONE ({[b for b, x in zip(boxes, w) if x][0]})" if any(w) else "SNAKE_NOT_BETTER")
    # Q4 per box
    for b in boxes:
        cs = [R[b][2].get(k) for k in ("R+l2init", "R+cbp")]; cs = [c for c in cs if c]
        if cs:
            labels[f"Q4_{b}"] = ("B_ABOVE_C" if all(c["sig_win"] for c in cs) else
                                 "B_BELOW_C" if any(c["sig_loss"] for c in cs) else "B_WITHIN_C")
    out.append("## Labels\n"); out += [f"- **{k}** = `{v}`" for k, v in labels.items()]
    txt = "\n".join(out); print(txt)
    (ROOT / "summary.md").write_text(txt + "\n")
    (ROOT / "verdict.json").write_text(json.dumps({"labels": labels,
        "pairs": {b: {k: {kk: (None if (isinstance(vv, float) and np.isnan(vv)) else (bool(vv) if isinstance(vv, (bool, np.bool_)) else float(vv))) for kk, vv in v.items()} for k, v in R[b][2].items() if v} for b in boxes}}, indent=2))


if __name__ == "__main__":
    main()

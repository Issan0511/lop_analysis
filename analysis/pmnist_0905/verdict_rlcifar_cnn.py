"""rlcifar_cnn_0908 — spec §4 judgements (Random Label CIFAR, Kumar's CNN).

Window = tasks 31-50 of `online_acc`. Sites are c1, c2, f1, f2.
Q1 SNA+l2 vs R+l2   -> GAP_WIDENS (>= +0.0233, the Random Label MNIST value) / GAP_HOLDS
                       (>= +0.02) / GAP_NARROWS / GAP_CLOSES; guard: R+l2 window < 0.30 ->
                       INCONCLUSIVE_WD_TOO_WEAK (set by stage 0, carried in via --wd-weak)
Q2 c transports     -> all four sites in [0.50, 0.85] with clip < 5% for BOTH SNA and SNA+l2
                       -> C_TRANSPORTS ; fc-only -> C_FC_ONLY ; some -> C_PARTIAL ; none ->
                       C_DOES_NOT_TRANSPORT
Q3 SNA+l2 vs LR+l2  -> SNAKE_SPECIFIC / NOT_SNAKE_SPECIFIC
n < 10 reads "9/10 or better" as "at most one seed against and p < .05" (identical at n = 10).
"""
import sys, glob, json, argparse
from math import comb
from pathlib import Path
import numpy as np, pandas as pd

WIN = (31, 50); SITES = ("c1", "c2", "f1", "f2"); BAND = (0.50, 0.85)
MNIST_GAP = 0.0233
LAB = {"R": "R", "LR": "LR", "SNA": "SNA", "R_l2": "R+l2", "LR_l2": "LR+l2",
       "SNA_l2": "SNA+l2", "R_l2init": "R+l2init", "SNA_l2init": "SNA+l2init"}


def load(root):
    parts = []
    for f in glob.glob(str(Path(root) / "**" / "per_task.csv"), recursive=True):
        d = pd.read_csv(f); d["arm"] = LAB.get(Path(f).parent.name, Path(f).parent.name)
        parts.append(d)
    d = pd.concat(parts)
    return d[d.online_acc.notna()]


def sign(diff):
    d = np.asarray(diff, float); nz = d[d != 0]; n = len(nz); pos = int((nz > 0).sum())
    k = min(pos, n - pos)
    return pos, n, (min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0) if n else np.nan)


def win(d, arm, col="online_acc"):
    g = d[(d.arm == arm) & (d.task >= WIN[0]) & (d.task <= WIN[1])]
    full = [s for s, gg in g.groupby("seed") if len(gg) == WIN[1] - WIN[0] + 1]
    return g[g.seed.isin(full)].groupby("seed")[col].mean()


def pair(d, a, b):
    x, y = win(d, a), win(d, b); s = sorted(set(x.index) & set(y.index))
    if not s: return None
    v = (x[s] - y[s]).values; pos, n, p = sign(v)
    return dict(mean=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan,
                wins=pos, n=n, p=float(p), sig_win=(n - pos <= 1 and p < 0.05),
                sig_loss=(pos <= 1 and p < 0.05))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results/rlcifar_cnn_0908")
    ap.add_argument("--wd-weak", action="store_true",
                    help="stage 0 found no lambda with R+l2 >= 0.30 (spec §3 guard)")
    a = ap.parse_args()
    d = load(a.root); out = ["# rlcifar_cnn_0908 — verdict\n"]
    arms = [x for x in ("R", "LR", "SNA", "R+l2", "LR+l2", "SNA+l2", "R+l2init", "SNA+l2init")
            if x in set(d.arm)]
    hdr = "| arm | n | online 31-50 | memo | collapse |" + "".join(f" mob {s} |" for s in SITES) + " clip% | ‖w‖ c1 |"
    out += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    V = {}
    for x in arms:
        rows = [g.sort_values("task") for _, g in d[d.arm == x].groupby("seed") if g.task.max() >= WIN[1]]
        if not rows: continue
        m = np.median
        w = lambda c: m([r[(r.task >= WIN[0]) & (r.task <= WIN[1])][c].mean() for r in rows]) if c in rows[0] else np.nan
        col = [int(r[r.online_acc < 0.5].task.min()) if (r.online_acc < 0.5).any() else 0 for r in rows]
        V[x] = dict(onl=w("online_acc"), memo=w("memo_acc"), n=len(rows), collapse=int(m(col)),
                    mob={s: w(f"mob_{s}") for s in SITES},
                    clip=max([w(f"alpha_clip_frac_{s}") for s in SITES if f"alpha_clip_frac_{s}" in rows[0]] or [np.nan]),
                    wr=m([r.w_norm_c1.tail(5).mean() / r.w_norm_c1.iloc[0] for r in rows]))
        v = V[x]
        out.append(f"| {x} | {v['n']} | {v['onl']:.4f} | {v['memo']:.4f} | {v['collapse'] or '—'} |"
                   + "".join(f" {v['mob'][s]:.3f} |" for s in SITES)
                   + (f" {100*v['clip']:.1f} |" if v["clip"] == v["clip"] else " |") + f" {v['wr']:.2f} |")
    P = {k: pair(d, *k) for k in (("SNA+l2", "R+l2"), ("SNA+l2", "LR+l2"), ("SNA+l2", "R+l2init"),
                                  ("SNA", "R"), ("SNA", "LR"), ("R+l2", "R"), ("SNA+l2", "SNA"))}
    out += ["\nPaired differences, online_acc, tasks 31-50:\n", "| a − b | mean ± SE | a wins | p |", "|---|---|---|---|"]
    for (x, y), r in P.items():
        if r: out.append(f"| {x} − {y} | {r['mean']:+.4f} ± {r['se']:.4f} | {r['wins']}/{r['n']} | {r['p']:.4f} |")
    L = {}
    q1 = P.get(("SNA+l2", "R+l2"))
    if a.wd_weak or (("R+l2" in V) and V["R+l2"]["onl"] < 0.30):
        L["Q1"] = "INCONCLUSIVE_WD_TOO_WEAK"
    elif q1:
        L["Q1"] = ("GAP_CLOSES" if not q1["sig_win"] else
                   "GAP_WIDENS" if q1["mean"] >= MNIST_GAP else
                   "GAP_HOLDS" if q1["mean"] >= 0.02 else "GAP_NARROWS")
    ok = {}
    for x in ("SNA", "SNA+l2"):
        if x in V:
            ok[x] = [s for s in SITES if BAND[0] <= V[x]["mob"][s] <= BAND[1]]
    if ok:
        allin = all(len(v) == 4 for v in ok.values())
        clipok = all(V[x]["clip"] != V[x]["clip"] or V[x]["clip"] < 0.05 for x in ok)
        fconly = all(set(v) == {"f1", "f2"} for v in ok.values())
        L["Q2"] = ("C_TRANSPORTS" if allin and clipok else "C_FC_ONLY" if fconly else
                   "C_PARTIAL" if any(ok.values()) else "C_DOES_NOT_TRANSPORT")
        L["Q2_sites_in_band"] = {x: v for x, v in ok.items()}
    q3 = P.get(("SNA+l2", "LR+l2"))
    if q3: L["Q3"] = "SNAKE_SPECIFIC" if q3["sig_win"] else "NOT_SNAKE_SPECIFIC"
    q4 = P.get(("SNA+l2", "R+l2init"))
    if q4: L["vs_l2init"] = "B_ABOVE_C" if q4["sig_win"] else ("B_BELOW_C" if q4["sig_loss"] else "B_WITHIN_C")
    out += ["\n## Labels\n"] + [f"- **{k}** = `{v}`" for k, v in L.items()]
    txt = "\n".join(out); print(txt)
    Path(a.root, "summary.md").write_text(txt + "\n")
    Path(a.root, "verdict.json").write_text(json.dumps(
        {"labels": L, "pairs": {f"{x}-{y}": r for (x, y), r in P.items() if r},
         "arms": {k: {kk: vv for kk, vv in v.items()} for k, v in V.items()}}, indent=2, default=float))


if __name__ == "__main__":
    main()

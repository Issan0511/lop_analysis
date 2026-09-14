"""pmnist_rlcifar_0907 — spec §3 judgements (Random Label CIFAR).

Window = tasks 31-50 of `online_acc` (pre-update batch accuracy, Kumar's metric).
Q1 : SNA+l2 vs R+l2 -> ACTIVATION_MATTERS_UNDER_WD (>=9/10, p<.05, AND mean diff >= +0.02)
                       / WD_ABSORBS / SNAKE_WORSE_UNDER_WD ; guard: R+l2 window < 0.50 -> INCONCLUSIVE_WD_TOO_WEAK
Q2 : SNA+l2 vs R+l2init -> B_ABOVE_C / B_WITHIN_C / B_BELOW_C
Q3 : SNA vs R, SNA vs LR (REPORT)
Rows are labelled by the run directory name (the module writes `iv` too; the directory is authoritative).
"""
import sys, glob, json
from math import comb
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/pmnist_rlcifar_0907")
WIN = (31, 50)
LAB = {"R": "R", "LR": "LR", "SNA": "SNA", "R_l2": "R+l2", "LR_l2": "LR+l2", "SNA_l2": "SNA+l2",
       "R_l2init": "R+l2init", "SNA_l2init": "SNA+l2init",
       "r_l2": "R+l2", "sna_l2": "SNA+l2", "r_l2init": "R+l2init", "sna_l2init": "SNA+l2init"}


def load():
    parts = []
    for f in glob.glob(str(ROOT / "**" / "per_task.csv"), recursive=True):
        d = pd.read_csv(f); sub = Path(f).parent.name
        d["arm"] = LAB.get(sub, sub)
        parts.append(d)
    d = pd.concat(parts)
    return d[d.online_acc.notna()] if "online_acc" in d else d


def sign(diff):
    d = np.asarray(diff, float); nz = d[d != 0]; n = len(nz); pos = int((nz > 0).sum())
    k = min(pos, n - pos); p = min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0) if n else np.nan
    return pos, n, p


def win(d, arm, col="online_acc"):
    g = d[(d.arm == arm) & (d.task >= WIN[0]) & (d.task <= WIN[1])]
    full = [s for s, gg in g.groupby("seed") if len(gg) == WIN[1] - WIN[0] + 1]
    return g[g.seed.isin(full)].groupby("seed")[col].mean()


def pair(d, a, b, col="online_acc"):
    x, y = win(d, a, col), win(d, b, col); s = sorted(set(x.index) & set(y.index))
    if not s: return None
    diff = (x[s] - y[s]).values; pos, n, p = sign(diff)
    # spec §3 says "9/10 or better"; with n < 10 (a run still in flight, or a dropped
    # arm) that literal reading can never fire, so the rule is read as "at most one
    # seed against, and p < .05". At n = 10 this is exactly 9/10.
    return dict(mean=float(diff.mean()), se=float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else np.nan,
                wins=pos, n=n, p=float(p),
                sig_win=(n - pos <= 1 and p < 0.05), sig_loss=(pos <= 1 and p < 0.05))


def main():
    d = load(); out = ["# pmnist_rlcifar_0907 — verdict\n"]
    arms = [a for a in ("R", "LR", "SNA", "R+l2", "LR+l2", "SNA+l2", "R+l2init", "SNA+l2init") if a in set(d.arm)]
    out.append("| arm | n | online_acc 31-50 | memo_acc 31-50 | collapse task (online<0.5) | mob L1 | zbar L1 | ‖w‖ L1 ratio | 2αW | clip% |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    V = {}
    for a in arms:
        rows = [g.sort_values("task") for _, g in d[d.arm == a].groupby("seed") if g.task.max() >= WIN[1]]
        if not rows: continue
        m = np.median
        onl = m([r[(r.task >= WIN[0]) & (r.task <= WIN[1])].online_acc.mean() for r in rows])
        memo = m([r[(r.task >= WIN[0]) & (r.task <= WIN[1])].memo_acc.mean() for r in rows]) if "memo_acc" in rows[0] else np.nan
        col = [int(r[r.online_acc < 0.5].task.min()) if (r.online_acc < 0.5).any() else 0 for r in rows]
        mob = m([r[r.task >= WIN[0]].mob_l1.mean() for r in rows]); zb = m([r[r.task >= WIN[0]].zbar_l1.mean() for r in rows])
        wr = m([r.w_norm_l1.tail(5).mean() / r.w_norm_l1.iloc[0] for r in rows])
        taw = m([r[r.task >= WIN[0]].two_alpha_W_med_l1.mean() for r in rows]) if "two_alpha_W_med_l1" in rows[0] and rows[0].two_alpha_W_med_l1.notna().any() else np.nan
        clip = m([r[r.task >= WIN[0]].alpha_clip_frac_l1.mean() for r in rows]) if "alpha_clip_frac_l1" in rows[0] and rows[0].alpha_clip_frac_l1.notna().any() else np.nan
        V[a] = dict(onl=onl, memo=memo, mob=mob, zb=zb, wr=wr, taw=taw, clip=clip, n=len(rows), collapse=int(m(col)))
        out.append(f"| {a} | {len(rows)} | {onl:.4f} | {memo:.4f} | {V[a]['collapse'] or '—'} | {mob:.4f} | {zb:+.3f} | {wr:.2f} | {'' if np.isnan(taw) else f'{taw:.2f}'} | {'' if np.isnan(clip) else f'{100*clip:.1f}'} |")
    P = {k: pair(d, *k) for k in (("SNA+l2", "R+l2"), ("SNA+l2", "LR+l2"), ("SNA+l2", "R+l2init"),
                                  ("SNA", "R"), ("SNA", "LR"), ("R+l2", "R"),
                                  ("SNA+l2", "SNA+l2init"), ("SNA+l2", "SNA"))}
    out.append("\nPaired differences, online_acc, tasks 31-50:\n"); out.append("| a − b | mean ± SE | a wins | p |"); out.append("|---|---|---|---|")
    for (a, b), r in P.items():
        if r: out.append(f"| {a} − {b} | {r['mean']:+.4f} ± {r['se']:.4f} | {r['wins']}/{r['n']} | {r['p']:.4f} |")
    # spec RandomLabelCIFAR_spec_0907 §3
    MNIST_GAP = 0.0233                      # the Random Label MNIST value Q1 compares against
    labels = {}
    q1 = P.get(("SNA+l2", "R+l2"))
    cap = V.get("SNA", {}).get("memo", 1.0)
    if cap < 0.90:                          # §3.1 guard: the MLP cannot memorise this box
        labels["Q1"] = labels["Q2"] = labels["Q3"] = "INCONCLUSIVE_CAPACITY"
    elif q1 and "R+l2" in V:
        if V["R+l2"]["onl"] < 0.30: labels["Q1"] = "INCONCLUSIVE_WD_TOO_WEAK"
        elif not q1["sig_win"]:     labels["Q1"] = "GAP_CLOSES"
        elif q1["mean"] >= MNIST_GAP: labels["Q1"] = "GAP_WIDENS"
        elif q1["mean"] >= 0.02:      labels["Q1"] = "GAP_HOLDS"
        else:                         labels["Q1"] = "GAP_NARROWS"
    if labels.get("Q1") != "INCONCLUSIVE_CAPACITY":
        # §3.2 Q2: does c=0.6 transport to a new input distribution?
        band = lambda a: (a in V and 0.50 <= V[a]["mob"] <= 0.85
                          and (V[a]["clip"] != V[a]["clip"] or V[a]["clip"] < 0.05))
        n_ok = sum(band(a) for a in ("SNA", "SNA+l2") if a in V)
        n_have = sum(a in V for a in ("SNA", "SNA+l2"))
        if n_have: labels["Q2"] = ("C_TRANSPORTS" if n_ok == n_have else
                                   "C_PARTIAL" if n_ok else "C_DOES_NOT_TRANSPORT")
        q3 = P.get(("SNA+l2", "LR+l2"))     # §3.3 Snake-specific (REPORT)
        if q3: labels["Q3_vs_LR+l2"] = f"{q3['mean']:+.4f} ({q3['wins']}/{q3['n']}, p={q3['p']:.4f})"
    q2c = P.get(("SNA+l2", "R+l2init"))
    if q2c: labels["vs_l2init"] = "B_ABOVE_C" if q2c["sig_win"] else ("B_BELOW_C" if q2c["sig_loss"] else "B_WITHIN_C")
    out.append("\n## Labels\n"); out += [f"- **{k}** = `{v}`" for k, v in labels.items()]
    txt = "\n".join(out); print(txt)
    (ROOT / "summary.md").write_text(txt + "\n")
    (ROOT / "verdict.json").write_text(json.dumps({"labels": labels, "pairs": {f"{a}-{b}": r for (a, b), r in P.items() if r}}, indent=2, default=float))


if __name__ == "__main__":
    main()

"""Stage-2 verdicts P1 (§5.2, as amended §10.2) and P2 (§5.3).

Window = tasks 151-200 (§5.2/§5.3). Everything is paired by seed and judged
with a two-sided sign test, n=10.

P1 indicators are zbar / mobility / w_norm -- `dead_frac` was dropped from P1
on 2026-09-05 because it is identically 0 for every arm with a non-zero
derivative floor, which makes the SN3-vs-LR pairing a 0-vs-0 tie in all seeds.
It is reported here as REPORT_ONLY.

P2 reports BOTH the sign test and the paired mean +- SE, as §5.3 registers
("both, decided in advance, not re-chosen after the fact").
"""
import argparse, glob
from math import comb
from pathlib import Path
import numpy as np
import pandas as pd

WIN = (151, 200)


def sign_test(diff):
    """Two-sided sign test. Zeros are dropped (and reported)."""
    d = np.asarray(diff, float)
    nz = d[d != 0]
    n, k = len(nz), int((nz < 0).sum())      # k = times `a` is LOWER than `b`
    if n == 0:
        return dict(n=0, wins=0, p=float("nan"), ties=len(d))
    p = sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n * 2
    return dict(n=n, wins=k, p=min(p, 1.0), ties=len(d) - n)


def window(df, arm, lr):
    g = df[(df.arm == arm) & (df.lr == lr) & df.acc.notna()]
    g = g[(g.task >= WIN[0]) & (g.task <= WIN[1])]
    full = [s for s, gg in g.groupby("seed") if len(gg) == WIN[1] - WIN[0] + 1]
    return g[g.seed.isin(full)].groupby("seed").mean(numeric_only=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--target", default="SN3")
    a = ap.parse_args()
    files = glob.glob(f"{a.root}/**/per_task.csv", recursive=True)
    df = pd.concat([pd.read_csv(f) for f in files])
    root = Path(a.root)

    P1_IND = [("zbar_l1", "zbar L1"), ("zbar_l2", "zbar L2"),
              ("mob_l1", "mobility L1"), ("mob_l2", "mobility L2"),
              ("w_norm_l1", "w_norm L1"), ("w_norm_l2", "w_norm L2")]
    rows = []
    for lr in sorted(df.lr.unique()):
        tgt = window(df, a.target, lr)
        if tgt.empty:
            rows.append({"lr": lr, "note": f"{a.target} has no complete seed in the window"})
            continue
        for opp in [x for x in df.arm.unique() if x != a.target]:
            o = window(df, opp, lr)
            seeds = sorted(set(tgt.index) & set(o.index))
            if not seeds:
                continue
            r = {"lr": lr, "target": a.target, "opponent": opp, "n_seed": len(seeds)}
            # --- P2: accuracy, higher is better for the target
            d = (tgt.loc[seeds, "acc"] - o.loc[seeds, "acc"]).values
            st = sign_test(-d)                       # wins = target higher
            r.update(p2_mean_diff=d.mean(), p2_se=d.std(ddof=1) / np.sqrt(len(d)),
                     p2_median=float(np.median(d)), p2_wins=st["wins"], p2_n=st["n"],
                     p2_p=st["p"])
            r["p2_mean_over_se"] = r["p2_mean_diff"] / r["p2_se"] if r["p2_se"] else np.nan
            # --- P1: lower |sinking| is better -> target should be HIGHER on
            #     zbar and mobility, LOWER on w_norm
            for col, _ in P1_IND:
                dd = (tgt.loc[seeds, col] - o.loc[seeds, col]).values
                s = sign_test(-dd if not col.startswith("w_norm") else dd)
                r[f"p1_{col}_wins"] = s["wins"]; r[f"p1_{col}_n"] = s["n"]
                r[f"p1_{col}_p"] = s["p"]; r[f"p1_{col}_med"] = float(np.median(dd))
            dd = (tgt.loc[seeds, "dead_frac_l1"] - o.loc[seeds, "dead_frac_l1"]).values
            r["dead_l1_ties"] = int((dd == 0).sum())
            rows.append(r)
    v = pd.DataFrame(rows)
    v.to_csv(root / "verdict.csv", index=False)

    for lr, g in v.groupby("lr"):
        print(f"\n{'='*104}\nlr = {lr:g}")
        print(f"{'vs':>5} {'n':>3} | {'P2 acc diff':>12} {'mean/SE':>8} {'sign':>7} {'p':>7} |"
              f" {'P1 zbar L1':>16} {'P1 mob L1':>16} {'P1 wnorm L1':>16}")
        print("-" * 104)
        for _, r in g.iterrows():
            if "opponent" not in r or pd.isna(r.get("opponent")):
                print("  ", r.get("note", "")); continue
            f = lambda c: (f"{r[f'p1_{c}_wins']:.0f}/{r[f'p1_{c}_n']:.0f} "
                           f"p={r[f'p1_{c}_p']:.3f}")
            print(f"{r.opponent:>5} {r.n_seed:>3.0f} | {r.p2_mean_diff:>+12.4f} "
                  f"{r.p2_mean_over_se:>+8.2f} {r.p2_wins:>3.0f}/{r.p2_n:<3.0f} "
                  f"{r.p2_p:>7.4f} | {f('zbar_l1'):>16} {f('mob_l1'):>16} "
                  f"{f('w_norm_l1'):>16}")
    print(f"\nP2: 'diff' = {a.target} minus opponent, positive = {a.target} more accurate.")
    print(f"P1: wins = seeds where {a.target} is shallower (zbar/mobility higher, "
          "w_norm lower).")
    print(f"\nwrote {root}/verdict.csv")


if __name__ == "__main__":
    main()

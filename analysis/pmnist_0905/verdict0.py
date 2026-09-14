"""Stage-0 verdict (spec §5.1) -> verdict.csv + summary.md.

Reports the M0 label under both anchors:
  registered : accuracy compared with task 1        (§5.1 literal)
  peak       : accuracy compared with the smoothed peak of the run
The second exists because below lr~0.05 task 1 is still in the warm-up phase,
so the registered anchor measures how good task 1 was rather than what was lost.
Both are printed; the registered one is the label.
"""
import argparse, glob
from pathlib import Path
import numpy as np
import pandas as pd

W = 10


def stats(g):
    g = g.sort_values("task")
    roll = g.acc.rolling(W, min_periods=W).mean()
    first, last = g.iloc[0], g.tail(W).mean(numeric_only=True)
    d = {"n_tasks": int(g.task.max()), "acc_t1": first.acc,
         "acc_peak": float(roll.max()), "peak_task": int(g.task.iloc[int(roll.argmax())]),
         "acc_final": float(g.acc.tail(W).mean())}
    d["drop_registered"] = d["acc_t1"] - d["acc_final"]
    d["drop_peak"] = d["acc_peak"] - d["acc_final"]
    for L in ("l1", "l2"):
        d[f"dead_rise_{L}"] = last[f"dead_frac_{L}"] - first[f"dead_frac_{L}"]
        d[f"dead_final_{L}"] = last[f"dead_frac_{L}"]
        d[f"mob_final_{L}"] = last[f"mob_{L}"]
        d[f"zbar_final_{L}"] = last[f"zbar_{L}"]
    for L in ("l1", "l2", "l3"):
        d[f"wratio_{L}"] = last[f"w_norm_{L}"] / first[f"w_norm_{L}"]
    return d


def label(m, anchor):
    acc = m[f"drop_{anchor}"] >= 0.02
    dead = max(m.dead_rise_l1, m.dead_rise_l2) >= 0.10
    w = max(m.wratio_l1, m.wratio_l2) >= 1.2
    if acc and dead and w:
        return "MECHANISM_PRESENT", (acc, dead, w)
    if acc:
        return "LOP_BY_OTHER_MECHANISM", (acc, dead, w)
    return "NO_LOP_IN_THIS_SETUP", (acc, dead, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--cut", type=int, default=None, help="truncate to this many tasks")
    a = ap.parse_args()
    files = sorted(glob.glob(f"{a.root}/**/per_task.csv", recursive=True))
    df = pd.concat([pd.read_csv(f) for f in files])
    df = df[df.acc.notna()]
    if a.cut:
        df = df[df.task <= a.cut]

    rows = []
    for (lr, arm), g in df.groupby(["lr", "arm"]):
        per = pd.DataFrame([{"seed": s, **stats(gg)} for s, gg in g.groupby("seed")])
        m = per.median(numeric_only=True)
        lab_r, f_r = label(m, "registered")
        lab_p, f_p = label(m, "peak")
        rows.append({"lr": lr, "arm": arm, "n_seed": len(per), **m.to_dict(),
                     "label_registered": lab_r, "label_peak": lab_p,
                     "flags_registered": "".join("*" if x else "." for x in f_r),
                     "flags_peak": "".join("*" if x else "." for x in f_p)})
    v = pd.DataFrame(rows).sort_values(["lr", "arm"])
    out = Path(a.root)
    v.to_csv(out / "verdict.csv", index=False)

    hdr = (f"{'lr':>6} {'arm':>4} {'n':>3} | {'t1':>5} {'peak':>5} {'@':>4} {'fin':>5} | "
           f"{'reg':>6} {'pk':>6} | {'d_rise':>6} {'d_fin':>5} {'wr1':>5} | {'mobN1':>7} {'zbN1':>7} | "
           f"{'M0 (registered)':<22} {'M0 (peak anchor)'}")
    lines = [hdr, "-" * len(hdr)]
    for _, r in v.iterrows():
        lines.append(
            f"{r.lr:>6g} {r.arm:>4} {int(r.n_seed):>3} | {r.acc_t1:>5.3f} {r.acc_peak:>5.3f} "
            f"{int(r.peak_task):>4} {r.acc_final:>5.3f} | {r.drop_registered:>+6.3f} "
            f"{r.drop_peak:>+6.3f} | {max(r.dead_rise_l1, r.dead_rise_l2):>+6.3f} {max(r.dead_final_l1, r.dead_final_l2):>5.3f} "
            f"{r.wratio_l1:>5.2f} | {r.mob_final_l1:>7.4f} {r.zbar_final_l1:>7.3f} | "
            f"{r.flags_registered} {r.label_registered:<20} {r.flags_peak} {r.label_peak}")
    txt = "\n".join(lines)
    print(txt)
    (out / "summary.md").write_text(
        "# pmnist_probe_0905 — stage 0 (M0)\n\n"
        f"tasks: {int(df.task.max())} | seeds: {sorted(df.seed.unique())}\n\n"
        "```\n" + txt + "\n```\n\n"
        "flags = (acc drop>=0.02)(dead rise>=0.10)(w_norm ratio>=1.2); "
        "dead/w_norm take the max over the two hidden layers.\n"
        "`reg` = drop from task 1 (§5.1 literal). `pk` = drop from the "
        f"{W}-task rolling peak. `fin` = mean of the last {W} tasks.\n")
    print(f"\nwrote {out}/verdict.csv and {out}/summary.md")


if __name__ == "__main__":
    main()

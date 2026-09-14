"""M0 judgement (spec §5.1) from a per_task.csv produced by src/pmnist_0905.py.

Registered rule (§5.1), all on the 10-seed median:
  acc(task 1) - acc(task N)      >= 0.02      AND
  dead_frac(task N) - dead_frac(task 1) >= 0.10   AND
  w_norm(task N) / w_norm(task 1)       >= 1.2
    -> MECHANISM_PRESENT
  accuracy falls but not the other two   -> LOP_BY_OTHER_MECHANISM
  accuracy does not fall                 -> NO_LOP_IN_THIS_SETUP

The spec does not say which layer the dead_frac / w_norm conditions apply to,
so this reports every layer and takes "holds in at least one hidden layer" as
the primary reading. The per-layer table is printed so the label can be
re-derived under any other rule.
"""
import argparse
import numpy as np
import pandas as pd


def per_seed(df, arm, lr, last_k=1):
    d = df[(df.arm == arm) & (df.lr == lr) & df.acc.notna()]
    out = {}
    for seed, g in d.groupby("seed"):
        g = g.sort_values("task")
        first, last = g.iloc[0], g.tail(last_k).mean(numeric_only=True)
        row = {"n_tasks": int(g.task.max()),
               "acc_first": first.acc, "acc_last": last.acc,
               "acc_drop": first.acc - last.acc}
        for L in ("l1", "l2"):
            row[f"dead_rise_{L}"] = last[f"dead_frac_{L}"] - first[f"dead_frac_{L}"]
            row[f"dead_last_{L}"] = last[f"dead_frac_{L}"]
            row[f"mob_first_{L}"] = first[f"mob_{L}"]
            row[f"mob_last_{L}"] = last[f"mob_{L}"]
            row[f"zbar_last_{L}"] = last[f"zbar_{L}"]
        for L in ("l1", "l2", "l3"):
            row[f"wratio_{L}"] = last[f"w_norm_{L}"] / first[f"w_norm_{L}"]
        out[seed] = row
    return pd.DataFrame(out).T


def judge(t):
    med = t.median()
    acc_ok = med.acc_drop >= 0.02
    dead_ok = max(med.dead_rise_l1, med.dead_rise_l2) >= 0.10
    w_ok = max(med.wratio_l1, med.wratio_l2) >= 1.2
    if acc_ok and dead_ok and w_ok:
        label = "MECHANISM_PRESENT"
    elif acc_ok:
        label = "LOP_BY_OTHER_MECHANISM"
    else:
        label = "NO_LOP_IN_THIS_SETUP"
    return label, med, (acc_ok, dead_ok, w_ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--arm", default="R")
    ap.add_argument("--last-k", type=int, default=1,
                    help="average the final k tasks instead of the endpoint")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)

    print(f"{'lr':>7} {'n':>3} {'ntask':>5} | {'acc1':>6} {'accN':>6} {'drop':>7} | "
          f"{'dead1':>6} {'deadN':>6} | {'wr_l1':>6} {'wr_l2':>6} | "
          f"{'mob1_l1':>7} {'mobN_l1':>7} | {'zbarN_l1':>8} | label")
    print("-" * 128)
    for lr in sorted(df.lr.unique()):
        t = per_seed(df, a.arm, lr, a.last_k)
        if t.empty:
            continue
        label, m, flags = judge(t)
        mark = "".join("*" if f else "." for f in flags)
        print(f"{lr:>7g} {len(t):>3d} {int(m.n_tasks):>5d} | {m.acc_first:>6.3f} "
              f"{m.acc_last:>6.3f} {m.acc_drop:>+7.3f} | "
              f"{max(m.dead_last_l1, m.dead_last_l2):>6.3f} "
              f"{max(m.dead_rise_l1, m.dead_rise_l2):>+6.3f} | "
              f"{m.wratio_l1:>6.3f} {m.wratio_l2:>6.3f} | "
              f"{m.mob_first_l1:>7.3f} {m.mob_last_l1:>7.3f} | "
              f"{m.zbar_last_l1:>8.3f} | {mark} {label}")
    print("\n  flags = (acc drop>=0.02)(dead rise>=0.10)(w ratio>=1.2), '*' = met")
    print("  dead1/deadN and w ratio: max over the two hidden layers")


if __name__ == "__main__":
    main()

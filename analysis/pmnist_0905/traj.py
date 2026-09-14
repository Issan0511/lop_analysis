"""Peak-vs-final accuracy, to separate warm-up from actual plasticity loss.

§5.1 anchors the accuracy condition on task 1. Below lr~0.05 task 1 is still in
the warm-up phase, so "accuracy fell from task 1" measures how good task 1 was,
not how much was lost. This reports the smoothed peak instead.
"""
import sys
import pandas as pd

df = pd.concat([pd.read_csv(p) for p in sys.argv[1:]])
W = 10
print(f"{'lr':>7} | {'acc t1':>6} {'peak':>6} {'@task':>5} {'final':>6} | "
      f"{'peak-fin':>8} {'per-seed':>9} | {'mobN_l1':>7} {'mobN_l2':>7} "
      f"{'nearoff':>7} {'dead':>6} | {'wr_l1':>6}")
print("-" * 108)
for lr, gl in df.groupby("lr"):
    peaks, finals, drops, t1s, ats = [], [], [], [], []
    for seed, g in gl.groupby("seed"):
        g = g.sort_values("task")
        r = g.acc.rolling(W, min_periods=W).mean()
        peaks.append(r.max()); ats.append(int(g.task.iloc[r.argmax()]))
        finals.append(g.acc.tail(W).mean()); t1s.append(g.acc.iloc[0])
        drops.append(r.max() - g.acc.tail(W).mean())
    m = gl[gl.task > gl.task.max() - W].median(numeric_only=True)
    med = lambda v: sorted(v)[len(v) // 2]
    print(f"{lr:>7g} | {med(t1s):>6.3f} {med(peaks):>6.3f} {med(ats):>5d} "
          f"{med(finals):>6.3f} | {med(peaks)-med(finals):>+8.3f} {med(drops):>+9.3f} | "
          f"{m.mob_l1:>7.4f} {m.mob_l2:>7.4f} "
          f"{'':>7} {max(m.dead_frac_l1, m.dead_frac_l2):>6.3f} | "
          f"{m.w_norm_l1 / gl[gl.task==1].w_norm_l1.median():>6.3f}")
print(f"\n  peak = max of a {W}-task rolling mean; final = mean of last {W} tasks")
print("  'per-seed' = median over seeds of each seed's own (peak - final)")

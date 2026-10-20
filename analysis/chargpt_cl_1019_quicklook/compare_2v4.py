import csv, numpy as np
from pathlib import Path
Q = Path("results/chargpt_cl_1019/quicklook")
def on(d):
    return {int(r["task"]): float(r["online_all"]) for r in csv.DictReader(open(d / "per_task.csv"))}
def merged(arm):
    o = on(Q / "M" / arm / "seed1000"); c = Q / "M" / (arm + "_cont") / "seed1000"
    if c.exists(): o.update(on(c))
    return o
runs = {"C3-2": on(Q / "M2/C3/seed1000"), "C3-4": merged("C3"), "SiLU-2": on(Q / "M2/SiLU/seed1000"), "SiLU-4": merged("SiLU")}
print("10 課題ごとのオンライン超過損失（同じ課題列）")
print("腕     | " + " | ".join("%d-%d" % (a, a + 9) for a in range(1, 100, 10)))
for k, o in runs.items():
    cells = []
    for a in range(1, 100, 10):
        ts = range(a, a + 10)
        cells.append("%.3f" % np.mean([o[t] for t in ts]) if all(t in o for t in ts) else "  -  ")
    print("%-6s | " % k + " | ".join(cells))
for k, d in (("C3-2", Q / "M2/C3/seed1000"), ("SiLU-2", Q / "M2/SiLU/seed1000")):
    E = {}
    for r in csv.DictReader(open(d / "layer_stats.csv")):
        if r["probe"] == "chain" and int(r["u"]) == 10000: E[(int(r["task"]), int(r["layer"]))] = r
    tt = (1, 5, 10, 25, 50, 75, 100)
    print(k, "μ/σ 中央 L0/L1:", " ".join("t%d %+.2f/%+.2f" % (t, float(E[(t, 0)]["r_med"]), float(E[(t, 1)]["r_med"])) for t in tt))
    print(k, "死 L0/L1:", " ".join("t%d %.2f/%.2f" % (t, float(E[(t, 0)]["dead"]), float(E[(t, 1)]["dead"])) for t in tt))
    print(k, "fresh:", " ".join("%s: %.3f" % (r["task"], float(r["online_all"])) for r in csv.DictReader(open(d / "fresh.csv"))))

"""事後の読み直し（追補 C）: 第 1 層の実効的な幅（前活性の標準偏差 zsd_l1）と重みのノルム（w_norm_l1 = 行ノルムの RMS）の時系列。

- PM: pm_ledger_1108 / pm_ledger_add_1108 の走のログ（per_task.csv、10 seed、500 課題）。
- RL-CIFAR: redo_chb_rlcifar_1018 の CHB 腕（ReLU raw、lr 0.001、seed 10〜19、200 課題）。本稿 §4 の箱。
数は seed ごとの窓平均をとってから seed の中央値（[最小, 最大]）。丸めは表示の 1 回だけ。
使い方: python analysis/pm_ledger_add_1108/width_posthoc.py > results/pm_ledger_add_1108/width_posthoc.md
"""
import glob, numpy as np, pandas as pd
D = "/home/issan/Projects/obsidian-research-data"
RL = f"{D}/redo_chb_rlcifar_1018/host_gpu_20261018T090515Z/results/redo_chb_rlcifar_1018/arms/CHB/per_task.csv"
ARMS = ["none", "l2init", "CHB", "KKT1", "CHB+INV0", "CHB+ISO", "L2I+ADD", "KKT1+INV0", "none+INV0"]
WIN = {"w1 6–50": (6, 50), "w2 101–150": (101, 150), "w3 251–300": (251, 300), "w4 451–500": (451, 500)}

fs = glob.glob(f"{D}/pm_ledger_1108/runs/*/seed*/per_task.csv") + glob.glob(f"{D}/pm_ledger_add_1108/runs/*/seed*/per_task.csv")
pm = pd.concat([pd.read_csv(f) for f in fs])
rl = pd.read_csv(RL, usecols=["arm", "seed", "task", "zsd_l1", "w_norm_l1", "zbar_l1", "dead_frac_l1", "online_acc"])

def wmean(df, a, b, cols):
    return df[(df.task >= a) & (df.task <= b)].groupby("seed")[cols].mean()

def fmt(s):  # median [min, max]
    return f"{s.median():.2f} [{s.min():.2f}, {s.max():.2f}]"

print("# 追補 C（事後）: 第 1 層の実効的な幅と重みのノルム\n")
print("幅 = 課題末の第 1 層の前活性の標準偏差（その課題の入力で測る）。ノルム = 第 1 層の行ノルムの RMS。seed ごとの窓平均 → seed の中央値。\n")
print("## 1. PM: 腕 × 窓（10 seed の中央値）\n")
print("| 腕 | 窓 | 幅 l1 | 平均 l1 | ノルム l1 | 死 l1 | 幅 l2 | 平均 l2 | 死 l2 |")
print("|---|---|---|---|---|---|---|---|---|")
cols = ["zsd_l1", "zbar_l1", "w_norm_l1", "dead_frac_l1", "zsd_l2", "zbar_l2", "dead_frac_l2"]
for arm in ARMS:
    g = pm[pm.arm == arm]
    for wn, (a, b) in WIN.items():
        m = wmean(g, a, b, cols).median()
        print(f"| {arm} | {wn} | {m.zsd_l1:.2f} | {m.zbar_l1:.2f} | {m.w_norm_l1:.2f} | {m.dead_frac_l1:.2f} | {m.zsd_l2:.2f} | {m.zbar_l2:.2f} | {m.dead_frac_l2:.2f} |")

print("\n## 2. 伸びの比（seed ごとの比の中央値 [最小, 最大]）\n")
print("| 箱 | 腕 | 窓 | 幅の比 | ノルムの比 | 早い窓の幅 | 遅い窓の幅 | 早い窓のノルム | 遅い窓のノルム |")
print("|---|---|---|---|---|---|---|---|---|")
for arm in ARMS:
    g = pm[pm.arm == arm]
    e, l = wmean(g, 6, 50, ["zsd_l1", "w_norm_l1"]), wmean(g, 451, 500, ["zsd_l1", "w_norm_l1"])
    r = (l / e).replace([np.inf, -np.inf], np.nan).dropna()
    print(f"| PM | {arm} | w1→w4 | {fmt(r.zsd_l1)} | {fmt(r.w_norm_l1)} | {e.zsd_l1.median():.2f} | {l.zsd_l1.median():.2f} | {e.w_norm_l1.median():.2f} | {l.w_norm_l1.median():.2f} |")
for (a, b), lab in [((1, 10), "t1–10→t181–200"), ((11, 30), "t11–30→t181–200")]:
    e, l = wmean(rl, a, b, ["zsd_l1", "w_norm_l1"]), wmean(rl, 181, 200, ["zsd_l1", "w_norm_l1"])
    r = (l / e)
    print(f"| RL-CIFAR | CHB | {lab} | {fmt(r.zsd_l1)} | {fmt(r.w_norm_l1)} | {e.zsd_l1.median():.2f} | {l.zsd_l1.median():.2f} | {e.w_norm_l1.median():.2f} | {l.w_norm_l1.median():.2f} |")

print("\n## 3. CHB の時系列（seed の中央値）\n")
rl["blk"] = ((rl.task - 1) // 20) * 20 + 1
t = rl.groupby("blk")[["zsd_l1", "w_norm_l1", "dead_frac_l1", "online_acc"]].median().round(2)
print("RL-CIFAR CHB、20 課題ごと（課題の頭）:\n")
print("| 量 | " + " | ".join(str(i) for i in t.index) + " |"); print("|---|" + "---|" * len(t.index))
for c, lab in [("zsd_l1", "幅 l1"), ("w_norm_l1", "ノルム l1"), ("dead_frac_l1", "死 l1"), ("online_acc", "online")]:
    print(f"| {lab} | " + " | ".join(f"{v:.2f}" for v in t[c]) + " |")
c_ = pm[pm.arm == "CHB"].copy(); c_["blk"] = ((c_.task - 1) // 50) * 50 + 1
t = c_.groupby("blk")[["zsd_l1", "w_norm_l1", "dead_frac_l1", "online"]].median().round(2)
print("\nPM CHB、50 課題ごと（課題の頭）:\n")
print("| 量 | " + " | ".join(str(i) for i in t.index) + " |"); print("|---|" + "---|" * len(t.index))
for c, lab in [("zsd_l1", "幅 l1"), ("w_norm_l1", "ノルム l1"), ("dead_frac_l1", "死 l1"), ("online", "online")]:
    print(f"| {lab} | " + " | ".join(f"{v:.2f}" for v in t[c]) + " |")

print("\n## 4. PM: 幅（w4）と喪失 G の関係（死の無い ReLU の 5 腕、seed の対）\n")
ps = pd.read_csv("/home/issan/Projects/claude/proj_004_drift/results/pm_ledger_add_1108/per_seed.csv")
key = {"l2init": "G_l2init", "CHB+ISO": "G_CHB_ISO", "CHB+INV0": "G_CHB_INV0", "L2I+ADD": "G_L2I_ADD", "CHB": "G_CHB"}
w4 = wmean(pm[pm.arm.isin(key)].assign(seed=lambda d: d.seed), 451, 500, ["zsd_l1"])  # placeholder (per arm below)
rows = []
for arm, gcol in key.items():
    w = wmean(pm[pm.arm == arm], 451, 500, ["zsd_l1", "w_norm_l1"])
    for s, r in w.iterrows():
        gv = ps.loc[ps.seed == s, gcol]
        if len(gv): rows.append(dict(arm=arm, seed=s, zsd=r.zsd_l1, wn=r.w_norm_l1, G=float(gv.iloc[0])))
rel = pd.DataFrame(rows)
print("| 腕 | 幅 l1（w4、中央値） | G（平均） |"); print("|---|---|---|")
for arm in key:
    g = rel[rel.arm == arm]; print(f"| {arm} | {g.zsd.median():.2f} | {g.G.mean():+.2f} |")
per_seed = rel.groupby("seed").apply(lambda g: g[["zsd", "G"]].corr(method="spearman").iloc[0, 1])
pooled = rel[["zsd", "G"]].corr(method="spearman").iloc[0, 1]
order_ok = rel.groupby("seed").apply(lambda g: (g.sort_values("zsd").G.diff().dropna() > 0).all())
print(f"\nseed ごとの Spearman（5 腕）: 中央値 {per_seed.median():.2f}、最小 {per_seed.min():.2f}、最大 {per_seed.max():.2f}（n = {len(per_seed)}）。")
print(f"幅の順に G が単調に増える seed: {int(order_ok.sum())}/{len(order_ok)}。全 50 点の Spearman: {pooled:.2f}。")
print("\n注: none は第 2 層の死（77%）が重なるので外した。KKT1 系は活性化が違い、新品の基準も違うので外した。事後の読みで、本稿の規則（付録 J）の外。")

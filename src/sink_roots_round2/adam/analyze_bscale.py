"""branch_scale_probe.py の出力（branches_scale/*.npz）の集計。PREDICTIONS.md の PS-a〜d を採点する。
深さ = 課題全体（10 更新ごと）の軌道の最小（unit ごと）の中央値、転回点 = 最小の時刻の中央値、戻り ÷ 深さ = 中央値どうし。
取り消しの時刻 = 旧ラベルの余裕が切替時の 2 割に落ちた最初の時刻。"""
import glob, sys
import numpy as np
files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "branches_scale/*.npz"))
score = {"a": [], "b": [], "c": [], "d": []}
for fn in files:
    d = np.load(fn); print(f"==== {fn.split('/')[-1]}")
    sws = sorted({int(k.split("_")[0][1:]) for k in d.files})
    for t in sws:
        live = d[f"t{t}_k"] > 0; res = {}
        for arm in ("base", "ro0p5", "ro2", "basef", "ro0p5f", "ro2f"):
            if f"t{t}_{arm}_tr" not in d: continue
            tr = d[f"t{t}_{arm}_tr"][:, live].astype(float); mg = d[f"t{t}_{arm}_mg"]; sd = d[f"t{t}_{arm}_sd"]; acc = d[f"t{t}_{arm}_acc"]
            mn = tr.min(0); tmin = (tr.argmin(0) + 1) * 10
            grid = np.arange(len(mg)) * 10
            idx = np.where(mg <= 0.2 * mg[0])[0]; te = grid[idx[0]] if len(idx) and mg[0] > 0 else np.nan
            res[arm] = dict(depth=float(np.median(mn)), tmin=float(np.median(tmin)), ret=float(np.median(tr[-1] - mn)), te=te, mg0=float(mg[0]), sd0=float(sd[0]), acc=float(acc[-1]))
        print(f" 切替 {t}（生きた unit {live.sum()}）")
        for arm, r in res.items():
            ref = res["basef"] if arm.endswith("f") else res["base"]
            print(f"   {arm:7s} 深さ {r['depth']:+.3f}（比 {r['depth']/ref['depth']:.2f}）  転回 {r['tmin']:5.0f}（比 {r['tmin']/ref['tmin']:.2f}）  戻り÷深さ {r['ret']/-r['depth']:.2f}  "
                  f"余裕 {r['mg0']:+.2f}・2 割の時刻 {r['te']:5.0f}・転回÷その時刻 {r['tmin']/r['te'] if r['te'] > 0 else np.nan:.2f}  logitSD {r['sd0']:.2f}  課題末の正解率 {r['acc']:.3f}")
        if all(a in res for a in ("base", "ro0p5", "ro2")):
            score["a"].append(1.15 <= res["ro2"]["depth"] / res["base"]["depth"] <= 2.0 and 0.5 <= res["ro0p5"]["depth"] / res["base"]["depth"] <= 0.87)
            score["b"].append(res["ro2"]["tmin"] > res["base"]["tmin"] and res["ro0p5"]["tmin"] < res["base"]["tmin"])
        if all(a in res for a in ("basef", "ro0p5f", "ro2f")):
            score["c"].append(all(0.8 <= res[a]["depth"] / res["basef"]["depth"] <= 1.25 for a in ("ro0p5f", "ro2f")))
        score["d"].append(all(np.isfinite(r["te"]) and r["te"] > 0 and 0.67 <= r["tmin"] / r["te"] <= 1.5 for r in res.values()))
print("\n採点（PREDICTIONS.md）:", {k: f"{sum(v)}/{len(v)}" for k, v in score.items()})

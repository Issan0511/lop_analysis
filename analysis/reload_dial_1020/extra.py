#!/usr/bin/env python3
"""reload_dial_1020 の追加の集計: 単調性、押し / 損失の下がり幅、下界が 0 以下の状態、Adam の沈みの時間変化。→ extra.json・extra.md"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kappa_reload_1020"))
import measure as M

OUT = Path("results/reload_dial_1020"); RAW = OUT / "raw"
rows = json.loads((OUT / "rows.json").read_text()); bound = json.loads((OUT / "bound_30tasks.json").read_text())
ACTS = ["LR", "R", "ELU", "SILU", "GELU"]; NAME = {"LR": "leaky 0.1", "R": "ReLU", "ELU": "ELU", "SILU": "SiLU", "GELU": "GELU"}
CKS = ["init", "a20", "a50", "a80", "a95", "end"]
ex = {"runs": {}, "time_course": {}}
ratios = []
for act in ACTS:
    for s in (0, 1):
        rr = [next(x for x in rows if x["act"] == act and x["seed"] == s and x["ck"] == c) for c in CKS]
        push = [x["S_exact_medN"] for x in rr]; A = [x["A"] for x in rr]; conf = [x["conf_y"] for x in rr]; gain = [x["gain"] for x in rr]
        rg = [p / g for p, g in zip(push[2:], gain[2:])]; ratios += rg
        ex["runs"][f"{act}_s{s}"] = dict(push_monotone=bool(np.all(np.diff(push) > 0)), A_monotone=bool(np.all(np.diff(A) > 0)),
                                         conf_monotone=bool(np.all(np.diff(conf) > 0)), push_over_gain_a50_end=rg)
ex["push_over_gain_quantiles"] = [float(v) for v in np.percentile(ratios, [0, 25, 50, 75, 100])]
ex["nonpositive_bound_states"] = [dict(act=b["act"], seed=b["seed"], task=b["task"], L=b["L"], B=b["B"], kappa=b["kappa"], kappa_lb=b["kappa_lb"])
                                  for b in bound if b["kappa_lb"] <= 0]
ex["B_over_N_quantiles"] = [float(v) for v in np.percentile([abs(x["B"]) / 1200 for x in rows + bound], [50, 90, 100])]
for act in ACTS:
    for ck in ("a50", "end"):
        fr = []
        for s in (0, 1):
            saved = np.load(M.STATES / f"{act}_K10_s{s}.npz"); X = M.load_X(saved["subset"]); xbar = X.mean(0)
            c = np.load(RAW / "ckpt" / f"ckpt_{act}_s{s}_{ck}.npz"); P = [c[f"p{q}"].astype(np.float64) for q in range(4)]
            e = M.evaluate(P, act, X, xbar); al = e["kopen"] > 0
            sw = np.load(RAW / "switch" / f"switch_{act}_s{s}.npz")
            j = 0
            while f"{ck}_d{j}_m" in sw.files:
                mt = sw[f"{ck}_d{j}_m"]; fr.append([float(((mt[t] - e["m"])[al] < 0).mean()) for t in range(mt.shape[0])]); j += 1
        ex["time_course"][f"{act}_{ck}"] = [float(v) for v in np.mean(fr, 0)]
ex["time_course_steps"] = [int(v) for v in np.load(RAW / "switch" / "switch_LR_s0.npz")["save_steps"]]
(OUT / "extra.json").write_text(json.dumps(ex, ensure_ascii=False, indent=1))
L = ["## 4 追加の集計（extra.py）\n",
     f"- 期待の押しの中央値が 6 つの checkpoint で単調に増えた走: {sum(v['push_monotone'] for v in ex['runs'].values())} / 10。揃い A は {sum(v['A_monotone'] for v in ex['runs'].values())} / 10、自信は {sum(v['conf_monotone'] for v in ex['runs'].values())} / 10。",
     "- 期待の押し / 損失の下がり幅（精度 50% 以降）: 最小・四分位・最大 = " + " / ".join(f"{v:.3f}" for v in ex["push_over_gain_quantiles"]),
     "- 下界が 0 以下だった状態: " + "、".join(f"{NAME[b['act']]} seed {b['seed']} 課題 {b['task']}（L = {b['L']:.4f}）" for b in ex["nonpositive_bound_states"]) + "。どれも全滅の直前で、損失が偶然 log K = 2.3026 とほぼ同じ。",
     "- |B|/N の中央値・90%・最大: " + " / ".join(f"{v:.4f}" for v in ex["B_over_N_quantiles"]),
     "", "Adam で沈んだ unit の割合の時間変化（5 draw × 2 seed の平均、生きた unit）:\n",
     "| 活性化 | checkpoint | " + " | ".join(f"{t} 更新" for t in ex["time_course_steps"]) + " |",
     "|---|---|" + "---|" * len(ex["time_course_steps"])]
for act in ACTS:
    for ck in ("a50", "end"):
        L.append(f"| {NAME[act]} | {'精度 50%' if ck == 'a50' else '課題 1 の末'} | " + " | ".join(f"{round(100 * v):d}%" for v in ex["time_course"][f"{act}_{ck}"]) + " |")
(OUT / "extra.md").write_text("\n".join(L) + "\n")
print("\n".join(L))

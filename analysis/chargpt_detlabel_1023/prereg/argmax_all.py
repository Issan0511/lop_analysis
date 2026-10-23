"""却下した代案 (a)「最頻の記号だけを出す連鎖」の閉路の分布（α = 0.01 の箱の遷移表、seed 0〜9 × 課題 1〜100 = 1,000 写像）。
遷移表は学習コードと同じ stream（np_gen(seed, "rule", task) の dirichlet）。π は使わない（最頻の写像だけ）。
使い方: python -I argmax_all.py <out_json>（学習コードの場所は chain_stats.py と同じ）
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import chain_stats as CS
C = CS.C
V = C.V
rows = []
for s in range(10):
    for t in range(1, 101):
        P2 = C.np_gen(s, "rule", t).dirichlet(0.01 * np.ones(V), size=V * V)
        ac = CS.argmax_cycles(P2)
        w, Lk = ac["basin"], ac["cycles_raw"]
        rows.append(dict(seed=s, task=t, n_cycles=len(Lk), n_cyclic=int(ac["on_cycle"].sum()),
                         rho_mean=float(ac["rho"].mean()), cyc_mean=float(ac["cyc_len"].mean()), tail_mean=float(ac["dist"].mean()),
                         rho=np.bincount(ac["rho"], minlength=600)[:600].tolist(), cyc=np.bincount(ac["cyc_len"], minlength=600)[:600].tolist(),
                         period_basin=float((w * Lk).sum()), copy_basin=float((w * np.maximum(0, 127 - Lk) / 127).sum()),
                         max_cycle=int(Lk.max()), basin_of_max=float(w[np.argmax(Lk)]),
                         basin_cyc_lengths=[int(x) for x in Lk], basin_weights=[float(x) for x in w]))
rho = np.sum([r["rho"] for r in rows], 0); cyc = np.sum([r["cyc"] for r in rows], 0)
json.dump(dict(rho_hist=rho.tolist(), cyc_hist=cyc.tolist(), maps=[{k: v for k, v in r.items() if k not in ("rho", "cyc")} for r in rows]), open(sys.argv[1], "w"))
def wq(h, ps):
    c = np.cumsum(h) / h.sum(); return [int(np.searchsorted(c, p)) for p in ps]
ps = (0.1, 0.25, 0.5, 0.75, 0.9)
print("maps", len(rows), "starts", int(rho.sum()))
print("rho（一様な出発点から最初の繰り返しまでの状態数）: 平均 %.1f 分位 %s" % ((np.arange(600) * rho).sum() / rho.sum(), wq(rho, ps)))
print("閉路長（一様な出発点が行き着く閉路）: 平均 %.1f 分位 %s" % ((np.arange(600) * cyc).sum() / cyc.sum(), wq(cyc, ps)))
nc = np.array([r["n_cyclic"] for r in rows]); ncy = np.array([r["n_cycles"] for r in rows])
print("閉路上の状態の数: 平均 %.1f 分位 %s" % (nc.mean(), np.quantile(nc, ps).tolist()))
print("閉路の数: 平均 %.2f 分位 %s" % (ncy.mean(), np.quantile(ncy, ps).tolist()))
pb = np.array([r["period_basin"] for r in rows]); cb = np.array([r["copy_basin"] for r in rows])
print("学習コードの π（流域の重み）での系列の周期: 平均 %.1f 分位 %s" % (pb.mean(), np.quantile(pb, ps).round(1).tolist()))
print("同じく、文脈の中で前の周期を写せば当たる目標の割合: 平均 %.3f 分位 %s" % (cb.mean(), np.quantile(cb, ps).round(3).tolist()))
# 一つの系列（周期 L の閉路の上）に出る異なる文脈の数 = min(L, 127)
nctx = np.array([sum(wi * min(Li, 127) for wi, Li in zip(r["basin_weights"], r["basin_cyc_lengths"])) for r in rows])
print("1 本の系列の異なる文脈の数（流域の重み）: 平均 %.1f" % nctx.mean())
print("理論（一様な乱写像 N=4225）: rho %.2f 閉路長 %.2f 閉路上の状態 %.2f 閉路の数 %.2f" % (np.sqrt(np.pi*4225/2), np.sqrt(np.pi*4225/8), np.sqrt(np.pi*4225/2), 0.5*np.log(2*4225)+0.5772/2))

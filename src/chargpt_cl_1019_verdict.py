#!/usr/bin/env python3
"""chargpt_cl_1019 の判定（spec §5、量の定め方は追補 0 §F）。本走の数値を見る前に書いた。

  python src/chargpt_cl_1019_verdict.py --root results/chargpt_cl_1019 --seeds 0 1 2 3 4 5 6 7 8 9

入力: <root>/<box>/<arm>/seed<k>/{per_task.csv, layer_stats.csv, fresh.csv, unit_stats.npz}
出力: <root>/verdict.csv と <root>/summary.md
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np

ARMS = ("C3", "C4", "B3", "N4")
PRIMARY = dict(M="chain", W="windows")
EARLY, LATE = range(1, 11), range(41, 51)


def rd(p):
    return list(csv.DictReader(open(p)))


def sign_test(d):
    d = [x for x in d if x != 0 and not math.isnan(x)]
    n, k = len(d), sum(1 for x in d if x > 0)
    if n == 0:
        return float("nan")
    cdf = lambda j: sum(math.comb(n, i) for i in range(j + 1)) / 2 ** n
    return min(1.0, 2 * min(cdf(k), 1 - cdf(k - 1)))


def mse(d):
    d = np.asarray([x for x in d if not math.isnan(x)], float)
    return (float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))) if len(d) > 1 else (float("nan"), float("nan"))


def seed_stats(d, box):
    pt = rd(d / "per_task.csv")
    ls = rd(d / "layer_stats.csv")
    n = int(pt[0]["n"])
    prim = PRIMARY[box]
    end = {}
    for r in ls:
        if r["probe"] == prim and int(r["u"]) == n:
            end.setdefault(int(r["task"]), []).append(r)
    lm = lambda t, k: float(np.mean([float(r[k]) for r in end[t]]))
    online = {int(r["task"]): float(r["online_all"]) for r in pt}
    fr = {int(r["task"]): float(r["online_all"]) for r in rd(d / "fresh.csv")} if (d / "fresh.csv").exists() else {}
    U = np.load(d / "unit_stats.npz")
    push = []
    for ti in range(1, len(pt)):                     # 課題 2〜50
        dm = (U[f"{prim}_mu_u100"][ti] - U[f"{prim}_mu_u0"][ti]) / (U[f"{prim}_sd_u0"][ti] + 1e-12)
        push.append(float(np.mean(np.median(dm, axis=1))))
    s = dict(neg10=lm(10, "neg_frac"), unitsneg10=lm(10, "units_mean_neg"),
             rmed_early=float(np.mean([lm(t, "r_med") for t in EARLY])),
             rmed_late=float(np.mean([lm(t, "r_med") for t in LATE])),
             on_early=float(np.mean([online[t] for t in EARLY])), on_late=float(np.mean([online[t] for t in LATE])),
             push_mean=float(np.mean(push)) if push else float("nan"))
    s["d_rmed"] = s["rmed_late"] - s["rmed_early"]
    # 成績が悪くなる向きを正に: M は超過損失が増える、W は精度が下がる
    s["on_worse"] = (s["on_late"] - s["on_early"]) if box == "M" else (s["on_early"] - s["on_late"])
    if 50 in fr:
        s["fresh_gap50"] = (online[50] - fr[50]) if box == "M" else (fr[50] - online[50])
    else:
        s["fresh_gap50"] = float("nan")
    # Q6（B3）: 課題 50 の末の |b|/|μ| と |w·β|/|μ| の unit 中央値
    mu = U[f"{prim}_mu"][-1].astype(float)
    s["bias_share"] = float(np.median(np.abs(U[f"{prim}_t_b"][-1]) / (np.abs(mu) + 1e-12)))
    s["beta_share"] = float(np.median(np.abs(U[f"{prim}_t_wbeta"][-1]) / (np.abs(mu) + 1e-12)))
    return s


def q1(ss):
    sink = sum(1 for s in ss if s["neg10"] >= 0.65 and s["unitsneg10"] >= 0.8)
    nos = sum(1 for s in ss if s["neg10"] < 0.58)
    return ("SINKS" if sink >= 8 else "NO_SINK" if nos >= 8 else "MIXED"), sink, nos


def q2(ss):
    d = [s["d_rmed"] for s in ss]
    if sum(x <= -0.1 for x in d) >= 8:
        return "RATCHET"
    if sum(abs(x) < 0.1 for x in d) >= 8:
        return "SATURATED"
    if sum(x >= 0.1 for x in d) >= 8:
        return "RECOVER"
    return "MIXED"


def q3(ss):
    if sum(1 for s in ss if s["on_worse"] > 0 and s["fresh_gap50"] > 0) >= 8:
        return "LOSS"
    if sum(1 for s in ss if s["fresh_gap50"] <= 0) >= 8:
        return "NO_LOSS"
    return "MIXED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    a = ap.parse_args()
    root = Path(a.root)
    rows, verdict = [], {}
    for box in ("M", "W"):
        for arm in ARMS:
            ss = [seed_stats(root / box / arm / f"seed{k}", box) for k in a.seeds
                  if (root / box / arm / f"seed{k}" / "per_task.csv").exists()]
            if len(ss) < len(a.seeds):
                continue
            v1, nsink, nnos = q1(ss)
            r = dict(box=box, arm=arm, n=len(ss), Q1=v1, Q1_sink_seeds=nsink, Q1_nosink_seeds=nnos,
                     Q2=q2(ss), Q3=q3(ss))
            for k in ("neg10", "unitsneg10", "d_rmed", "on_worse", "fresh_gap50", "push_mean", "bias_share", "beta_share"):
                m, se = mse([s[k] for s in ss])
                r[f"{k}_mean"], r[f"{k}_se"] = m, se
            for k in ("d_rmed", "on_worse", "fresh_gap50"):
                r[f"{k}_sign_p"] = sign_test([s[k] for s in ss])
            rows.append(r)
            verdict[(box, arm)] = r
    for box in ("M", "W"):
        g = lambda arm: verdict.get((box, arm), {}).get("Q1")
        if g("C3") and g("C4"):
            q4 = "LR_GATES_SINKING" if g("C3") == "SINKS" and g("C4") == "NO_SINK" else f"C3={g('C3')},C4={g('C4')}"
            if g("N4"):
                if g("N4") == "SINKS" and g("C4") == "NO_SINK":
                    q5 = "LN_SLOWS_AT_LOW_LR"
                elif g("N4") == "NO_SINK" and g("C4") == "NO_SINK":
                    q5 = "LR_ALONE"
                elif g("N4") == "SINKS" and g("C4") == "SINKS":
                    q5 = "BOTH_SINK(パイロット B と食い違い)"
                else:
                    q5 = f"N4={g('N4')},C4={g('C4')}"
            else:
                q5 = "N/A"
            rows.append(dict(box=box, arm="ALL", Q4=q4, Q5=q5))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(root / "verdict.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    lines = ["# chargpt_cl_1019 判定", "", "| 箱 | 腕 | Q1 | Q2 | Q3 | 負の割合（課題 10） | Δμ/σ 中央（後期−早期） | fresh gap（課題 50） |",
             "| -- | -- | -- | -- | -- | -- | -- | -- |"]
    for r in rows:
        if r.get("arm") == "ALL":
            continue
        lines.append(f"| {r['box']} | {r['arm']} | {r['Q1']} | {r['Q2']} | {r['Q3']} | "
                     f"{r['neg10_mean']:.3f} ± {r['neg10_se']:.3f} | {r['d_rmed_mean']:+.3f} ± {r['d_rmed_se']:.3f} | "
                     f"{r['fresh_gap50_mean']:+.4f} ± {r['fresh_gap50_se']:.4f} |")
    lines += [""] + [f"- 箱 {r['box']}: Q4 {r['Q4']}、Q5 {r['Q5']}" for r in rows if r.get("arm") == "ALL"]
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

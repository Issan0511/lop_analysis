#!/usr/bin/env python3
"""追補 8 の集計: 3 腕 × 7 課題 × 10 seed の L1 の変位（上流の分・合計・減りの割合）と予測の採点（事後・未登録）。
  python analysis/chargpt_cl_1019_quicklook/carrier_fix_summary.py
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
O = ROOT / "results/chargpt_cl_1019/m2_carrier_fix"
TASKS = (2, 3, 5, 10, 41, 45, 50); EARLY, LATE = (2, 3, 5, 10), (41, 45, 50); SEEDS = range(10)


def main():
    V = {}            # (seed, arm, t, sel) -> dict(up, total, own)
    loss, checks = {}, []
    for s in SEEDS:
        d = O / f"seed{s}"; A = np.load(d / "units.npz"); M = json.loads((d / "meta.json").read_text())["runs"]
        for t in TASKS:
            sink = A[f"control_t{t}_total"] < 0
            for arm in ("control", "ffn0", "attn1"):
                up = A[f"{arm}_t{t}_up"] + A[f"{arm}_t{t}_cross"]; tot = A[f"{arm}_t{t}_total"]; own = A[f"{arm}_t{t}_own"]
                for sel, m in (("sink", sink), ("all", np.ones_like(sink))):
                    V[(s, arm, t, sel)] = dict(up=float(up[m].mean()) if m.any() else np.nan, total=float(tot[m].mean()) if m.any() else np.nan,
                                               own=float(own[m].mean()) if m.any() else np.nan, n=int(m.sum()))
                loss[(s, arm, t)] = M[f"{arm}_t{t}"]["loss_last10"]
                checks.append((s, arm, t, M[f"{arm}_t{t}"]))
    red = lambda a, c: 1.0 - a / c if c != 0 else np.nan
    grp = lambda s, arm, ts, key, sel: float(np.nanmean([V[(s, arm, t, sel)][key] for t in ts]))
    L = ["# 追補 8：2 ブロックの L1 の担い手を、切替の 100 更新で上流の共通成分を固定して確かめる（事後・未登録、C3-2 seed 0〜9）", ""]
    ctl = [c for c in checks if c[1] == "control"]
    L.append(f"- 検査：対照 {len(ctl)} 本のうち、損失のビット一致 {sum(c[3]['loss_bitwise'] for c in ctl)}・更新 100 の重みのビット一致 {sum(c[3]['u100_bitwise'] for c in ctl)}。"
             f"恒等式の残差は最大 {max(c[3]['resid'] for c in checks):.1e}。固定した共通成分の窓の末のずれは最大 {max(c[3].get('fix_err', 0) for c in checks):.1e}（相対）。")
    for sel, title in (("sink", "対照で沈む unit の平均（主）"), ("all", "全 unit の平均（副）")):
        L += ["", f"## {title}", "", "| 課題 | 対照 上流の分 / 合計 | FFN0 固定 上流の分 / 合計（減り） | 注意 1 固定 上流の分 / 合計（減り） | 損失の差（FFN0 / 注意 1） |", "| -- | -- | -- | -- | -- |"]
        for t in TASKS:
            c = [V[(s, "control", t, sel)] for s in SEEDS]; f = [V[(s, "ffn0", t, sel)] for s in SEEDS]; a = [V[(s, "attn1", t, sel)] for s in SEEDS]
            m = lambda xs, k: float(np.nanmedian([x[k] for x in xs]))
            rf = np.nanmedian([red(f[i]["up"], c[i]["up"]) for i in range(10)]); ra = np.nanmedian([red(a[i]["up"], c[i]["up"]) for i in range(10)])
            rft = np.nanmedian([red(f[i]["total"], c[i]["total"]) for i in range(10)]); rat = np.nanmedian([red(a[i]["total"], c[i]["total"]) for i in range(10)])
            dl_f = np.median([loss[(s, "ffn0", t)] - loss[(s, "control", t)] for s in SEEDS]); dl_a = np.median([loss[(s, "attn1", t)] - loss[(s, "control", t)] for s in SEEDS])
            L.append(f"| {t} | {m(c, 'up'):+.3f} / {m(c, 'total'):+.3f} | {m(f, 'up'):+.3f} / {m(f, 'total'):+.3f}（{rf:.2f} / {rft:.2f}） | {m(a, 'up'):+.3f} / {m(a, 'total'):+.3f}（{ra:.2f} / {rat:.2f}） | {dl_f:+.3f} / {dl_a:+.3f} |")
        L.append("")
        if sel == "sink":
            miss = [(s, t) for s in SEEDS for t in TASKS if V[(s, "control", t, "sink")]["n"] == 0]
            L.append(f"対照で沈む unit が 0 個の (seed, 課題) は欠測: {miss}。中央値と塊の平均は欠測を除いて取る（spec に無い扱い、走ったあとで決めた）。")
        L.append("値は 10 seed の中央値。減りは seed ごとの 1 − 腕 / 対照 の中央値（上流の分 / 合計）。損失の差は更新 91〜100 の平均の、対照との差の中央値。")
    # seed ごとの塊の減りと採点（主の集計）
    rows = []
    for s in SEEDS:
        r = dict(seed=s)
        for arm in ("ffn0", "attn1"):
            for g, ts in (("early", EARLY), ("late", LATE)):
                for key in ("up", "total"):
                    r[f"{arm}_{g}_{key}"] = red(grp(s, arm, ts, key, "sink"), grp(s, "control", ts, key, "sink"))
        r["ctl_early_up_share"] = grp(s, "control", EARLY, "up", "sink") / grp(s, "control", EARLY, "total", "sink")
        r["ctl_late_up_share"] = grp(s, "control", LATE, "up", "sink") / grp(s, "control", LATE, "total", "sink")
        rows.append(r)
    L += ["", "## seed ごとの塊の減り（主の集計。課題 2〜10 は t = 2・3・5・10 の平均、41〜50 は t = 41・45・50 の平均を 1 つの値として比べる）", "",
          "| seed | 対照の上流の割合 2〜10 / 41〜50 | FFN0 固定 2〜10 上流 / 合計 | FFN0 固定 41〜50 上流 / 合計 | 注意 1 固定 2〜10 上流 / 合計 | 注意 1 固定 41〜50 上流 / 合計 |", "| -- | -- | -- | -- | -- | -- |"]
    for r in rows:
        L.append(f"| {r['seed']} | {r['ctl_early_up_share']:.2f} / {r['ctl_late_up_share']:.2f} | {r['ffn0_early_up']:.2f} / {r['ffn0_early_total']:.2f} | {r['ffn0_late_up']:.2f} / {r['ffn0_late_total']:.2f} | {r['attn1_early_up']:.2f} / {r['attn1_early_total']:.2f} | {r['attn1_late_up']:.2f} / {r['attn1_late_total']:.2f} |")
    n1 = sum(r["ffn0_early_up"] >= 0.5 for r in rows); n2 = sum(r["ffn0_early_total"] >= 0.4 for r in rows)
    n3a = sum(r["ffn0_late_up"] < 0.3 for r in rows); n3a_t = sum(r["ffn0_late_total"] < 0.3 for r in rows)
    n3b = sum(r["attn1_late_up"] > r["attn1_early_up"] for r in rows); n3b_t = sum(r["attn1_late_total"] > r["attn1_early_total"] for r in rows)
    v = lambda k: "当たり" if k >= 8 else "外れ"
    L += ["", "## 予測の採点（8/10 以上で当たり）", "", "| 予測 | seed の数 | 結果 |", "| -- | -- | -- |",
          f"| P-C1 FFN0 固定で課題 2〜10 の上流の分が 5 割以上減る | {n1}/10 | {v(n1)} |",
          f"| P-C2 FFN0 固定で課題 2〜10 の合計が 4 割以上減る | {n2}/10 | {v(n2)} |",
          f"| P-C3a FFN0 固定の減りは課題 41〜50 で 3 割未満（上流の分。合計では {n3a_t}/10） | {n3a}/10 | {v(n3a)} |",
          f"| P-C3b 注意 1 固定の減りは課題 41〜50 のほうが大きい（上流の分。合計では {n3b_t}/10） | {n3b}/10 | {v(n3b)} |"]
    (O / "carrier_fix_summary.md").write_text("\n".join(L)); (O / "carrier_fix_summary.json").write_text(json.dumps(dict(rows=rows, scores=dict(P_C1=n1, P_C2=n2, P_C3a=n3a, P_C3b=n3b)), indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()

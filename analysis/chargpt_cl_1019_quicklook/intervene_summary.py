#!/usr/bin/env python3
"""L1 の生き返りの介入（intervene_revival.py の出力）を腕ごとに並べる（事後・判定外）。
  python analysis/chargpt_cl_1019_quicklook/intervene_summary.py
"""
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
O = ROOT / "results/chargpt_cl_1019/quicklook/M2/intervene"
SRC = ROOT / "results/chargpt_cl_1019/quicklook/M2/C3/seed1000"
ARMS = [("control_from40", "対照（課題 40 から）"), ("lnfix_from40", "分母を定数に固定"), ("lnex1_from40", "分母から FFN1 を除く"),
        ("nocommon_from40", "FFN1 の共通成分を除く"), ("l0freeze_from40", "L0 を凍結"),
        ("control_from25", "対照（課題 25 から）"), ("gamma03_from25", "γ_f を 0.3 倍"), ("gamma03f_from25", "γ_f を 0.3 倍で固定")]


def layer(d, probe="chain"):
    out = {}
    for r in csv.DictReader(open(d / "layer_stats.csv")):
        if r["probe"] == probe: out[(int(r["task"]), int(r["layer"]))] = r
    return out


def three_phase(loss, H):
    ex = loss.astype(np.float64) - H; ma = np.convolve(ex, np.ones(50) / 50, "valid")
    hit = np.nonzero(ma <= 0.15)[0]
    return (int(hit[0]) + 50 if len(hit) else 10000), float(ex[950:1050].mean()), float(ex[-1000:].mean())


def main():
    lines = ["# L1 の生き返りを介入で分ける（事後・判定外・未登録、C3-2 seed 1000）", ""]
    orig = layer(SRC); orig_pt = {int(r["task"]): r for r in csv.DictReader(open(SRC / "per_task.csv"))}
    res = {}
    for name, label in ARMS:
        d = O / name
        if not (d / "provenance.json").exists(): continue
        pv = json.loads((d / "provenance.json").read_text()); Ls = layer(d)
        ex = {int(r["task"]): r for r in csv.DictReader(open(d / "extras.csv"))}
        pt = {int(r["task"]): r for r in csv.DictReader(open(d / "per_task.csv"))}
        loss = np.load(d / "online.npz")["loss"]; a, b = pv["from_task"], pv["to_task"]
        lines += [f"## {label}（{name}）", ""]
        if pv.get("control_bitwise_match") is not None: lines.append(f"- 元の走の online とビット一致: {pv['control_bitwise_match']}")
        if pv.get("freeze_max_abs_diff") is not None: lines.append(f"- 固定した重みの最大のずれ: {pv['freeze_max_abs_diff']:.1e}")
        lines += ["", "| 課題 | L1 の死 | L0 の死 | μ/σ L1 / L0 | 最終 LN の入力 rms | 使った分母 | FFN1 rms（共通成分） | FFN0 rms | 注意 1 rms | L1 の有効な読み出し | γ_f 中央 | 超過損失（課題の平均） | 元の走の L1 の死 |",
                  "| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |"]
        rows = []
        for t in range(a, b + 1):
            if (t, 1) not in Ls or t not in ex: continue
            l1, l0, e = Ls[(t, 1)], Ls[(t, 0)], ex[t]
            on = float(pt[t]["online_all"]) if t in pt else float("nan")
            o1 = float(orig[(t, 1)]["frac_p_lt005"]) if (t, 1) in orig else float("nan")
            row = dict(task=t, d1=float(l1["frac_p_lt005"]), d0=float(l0["frac_p_lt005"]), r1=float(l1["r_med"]), r0=float(l0["r_med"]), nf=float(e["nf_in_rms"]), den=float(e["denom_used"]),
                       f1=float(e["ffn1_rms"]), c1=float(e["ffn1_common_norm"]), f0=float(e["ffn0_rms"]), a1=float(e["attn1_rms"]), v1=float(e["veff1_med"]), g=float(e["gamma_f_med"]), on=on, o1=o1)
            rows.append(row)
            lines.append(f"| {t} | {row['d1']:.2f} | {row['d0']:.2f} | {row['r1']:+.2f} / {row['r0']:+.2f} | {row['nf']:.1f} | {row['den']:.1f} | {row['f1']:.1f}（{row['c1']:.0f}） | {row['f0']:.1f} | {row['a1']:.1f} | {row['v1']:.3f} | {row['g']:.3f} | {on:.3f} | {o1:.2f} |")
        tp = [three_phase(loss[i], float(pt[t]["H"])) for i, t in enumerate(range(a + 1, b + 1)) if t in pt]
        last = list(range(a + 1, b + 1))[-5:]
        tpl = [three_phase(loss[t - a - 1], float(pt[t]["H"])) for t in last]
        lines += ["", f"- 最後の 5 課題（{last[0]}〜{last[-1]}）の 3 相の平均: T015 {np.mean([x[0] for x in tpl]):.0f}・E1000 {np.mean([x[1] for x in tpl]):.3f}・ELast {np.mean([x[2] for x in tpl]):.3f}", ""]
        res[name] = dict(rows=rows, three_phase_last5=tpl, provenance=pv)
    (O / "intervene_summary.md").write_text("\n".join(lines)); (O / "intervene_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

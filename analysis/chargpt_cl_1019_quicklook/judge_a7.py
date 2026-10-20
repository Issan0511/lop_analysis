#!/usr/bin/env python3
"""spec 追補 7 の判定（登録走、2 ブロック、C3-2 対 ReLUcap-2、seed 0〜9、100 課題）。規則は追補 7 §3 のまま。
  python analysis/chargpt_cl_1019_quicklook/judge_a7.py --out results/chargpt_cl_1019/m2_judge7
"""
import argparse, csv, json, math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(Path(__file__).resolve().parent))
from three_phase import phases  # noqa: E402
R = ROOT / "results/chargpt_cl_1019"


def sign_test(d):
    d = [x for x in d if x != 0 and not math.isnan(x)]; n, k = len(d), sum(1 for x in d if x > 0)
    if n == 0: return float("nan")
    cdf = lambda j: sum(math.comb(n, i) for i in range(j + 1)) / 2 ** n
    return min(1.0, 2 * min(cdf(k), 1 - cdf(k - 1)))


def run_stats(arm, s):
    d = R / "M2" / arm / f"seed{s}"
    on = np.load(d / "online.npz")["loss"]; pt = list(csv.DictReader(open(d / "per_task.csv"))); H = [float(r["H"]) for r in pt]
    P = np.array([phases(on[t], H[t]) for t in range(90, 100)]).mean(0)
    fr = phases(np.load(d / "fresh_online_t100.npz")["loss"], H[99])
    last = phases(on[99], H[99])
    E = {}
    for r in csv.DictReader(open(d / "layer_stats.csv")):
        if r["probe"] == "uniform" and int(r["u"]) == 10000: E[(int(r["task"]), int(r["layer"]))] = r
    lay = {t: {l: (float(E[(t, l)]["frac_p_lt005"]), float(E[(t, l)]["frac_p_ge099"])) for l in (0, 1)} for t in (10, 50, 100)}
    online = [float(r["online_all"]) for r in pt]
    return dict(P=P, fresh=fr, last100=last, layers=lay, online=online)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(R / "m2_judge7")); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    S = {arm: {s: run_stats(arm, s) for s in range(10)} for arm in ("C3", "ReLUcap")}
    names = ("T015", "E1000", "ELast"); lines = ["# 追補 7 の判定（登録走、2 ブロック、seed 0〜9、100 課題）", ""]
    better = np.array([[S["ReLUcap"][s]["P"][i] < S["C3"][s]["P"][i] for i in range(3)] for s in range(10)])
    worse = np.array([[S["ReLUcap"][s]["P"][i] > S["C3"][s]["P"][i] for i in range(3)] for s in range(10)])
    all_b, all_w = int(better.all(1).sum()), int(worse.all(1).sum())
    verdict = "CAP_RESCUES" if all_b >= 8 else ("CAP_HARMS" if all_w >= 8 else "MIXED")
    lines += ["## 主（課題 91〜100 の平均、seed ごと）", "", "| seed | C3-2 T015 / E1000 / ELast | ReLUcap-2 T015 / E1000 / ELast | 3 相とも良い |", "| -- | -- | -- | -- |"]
    for s in range(10):
        c, r = S["C3"][s]["P"], S["ReLUcap"][s]["P"]
        lines.append(f"| {s} | {c[0]:.0f} / {c[1]:.3f} / {c[2]:.3f} | {r[0]:.0f} / {r[1]:.3f} / {r[2]:.3f} | {'はい' if better[s].all() else 'いいえ'} |")
    lines.append("")
    per = []
    for i, nm in enumerate(names):
        diff = [S["ReLUcap"][s]["P"][i] - S["C3"][s]["P"][i] for s in range(10)]
        per.append(dict(phase=nm, better_seeds=int(better[:, i].sum()), sign_p=sign_test(diff), median_diff=float(np.median(diff))))
        lines.append(f"- {nm}: cap が良い seed {int(better[:, i].sum())}/10、符号検定 p = {sign_test(diff):.4f}、対の差（ReLUcap − C3）の中央値 {np.median(diff):+.3f}")
    lines += [f"- 3 相ともに良い seed {all_b}/10、3 相ともに悪い seed {all_w}/10 → **{verdict}**", ""]
    # 副 1
    j6 = {s: json.load(open(R / "m2_judge6" / f"seed{s}" / "judge_m2.json"))["C3"] for s in range(10)}
    c1 = [j6[s]["cond1"]["pass"] for s in range(10)]; c2 = [j6[s]["cond2"]["pass"] for s in range(10)]; c3 = [j6[s]["cond3"]["pass"] for s in range(10)]
    allp = [x and y and z for x, y, z in zip(c1, c2, c3)]
    lines += ["## 副 1（C3-2 が追補 6 の 3 条件をすべて満たす seed）", "", f"- 条件 1 {sum(c1)}/10、条件 2 {sum(c2)}/10、条件 3 {sum(c3)}/10 → すべて {sum(allp)}/10（外れた seed: " + ", ".join(str(s) for s in range(10) if not allp[s]) + "）", ""]
    # 副 2
    lines += ["## 副 2（届かない型: 課題 91〜100 の ELast が同じ seed の新品（課題 100）を上回る）", ""]
    sec2 = {}
    for arm in ("ReLUcap", "C3"):
        cnt = sum(1 for s in range(10) if S[arm][s]["P"][2] > S[arm][s]["fresh"][2])
        cnt100 = sum(1 for s in range(10) if S[arm][s]["last100"][2] > S[arm][s]["fresh"][2])
        sec2[arm] = dict(registered=cnt, same_task100=cnt100)
        lines.append(f"- {arm}-2: {cnt}/10（登録の定義）。参考: 課題 100 どうし（本体の課題 100 の ELast 対 新品）では {cnt100}/10")
    lines.append("- seed ごと（ReLUcap-2 の ELast 91〜100 / 課題 100 / 新品）: " + "; ".join(f"{s}: {S['ReLUcap'][s]['P'][2]:.3f}/{S['ReLUcap'][s]['last100'][2]:.3f}/{S['ReLUcap'][s]['fresh'][2]:.3f}" for s in range(10)))
    lines.append("")
    # 記録: 層ごとの死と正側の片側化
    lines += ["## 記録（固定評価集合: 死 p < 0.005 / 正側の片側化 p ≥ 0.99、seed の平均）", ""]
    for arm in ("C3", "ReLUcap"):
        row = []
        for t in (10, 50, 100):
            for l in (0, 1):
                dd = np.mean([S[arm][s]["layers"][t][l][0] for s in range(10)]); oo = np.mean([S[arm][s]["layers"][t][l][1] for s in range(10)])
                row.append(f"t{t} L{l} {dd:.2f}/{oo:.2f}")
        lines.append(f"- {arm}-2: " + " | ".join(row))
    # 2 対 4
    lines += ["", "## 同じ課題列での 2 対 4（10 課題ごとのオンライン超過損失、seed 0〜9 の平均）", ""]
    four = []
    for s in range(10):
        four.append([float(r["online_all"]) for r in csv.DictReader(open(R / "M" / "C3" / f"seed{s}" / "per_task.csv"))])
    for nm, arr in (("C3-4（本走）", np.array(four)), ("C3-2", np.array([S["C3"][s]["online"] for s in range(10)])), ("ReLUcap-2", np.array([S["ReLUcap"][s]["online"] for s in range(10)]))):
        n = arr.shape[1]
        lines.append(f"- {nm}: " + " ".join(f"{a_}-{a_+9}: {arr[:, a_-1:a_+9].mean():.3f}" for a_ in range(1, n, 10)))
    res = dict(verdict=verdict, all_better=all_b, all_worse=all_w, per_phase=per, sec1=dict(cond1=sum(c1), cond2=sum(c2), cond3=sum(c3), all=sum(allp)), sec2=sec2)
    (out / "judge_a7.json").write_text(json.dumps(res, indent=1, ensure_ascii=False)); (out / "judge_a7.md").write_text("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main()

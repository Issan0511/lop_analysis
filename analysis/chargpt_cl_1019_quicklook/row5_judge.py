#!/usr/bin/env python3
"""spec 追補 9 の判定（9.A §5 の 4 比較の族）と 9.C の ELR の判定、副の記述量。結果を見る前に書いた。
  CHARGPT_NLAYER=2 python analysis/chargpt_cl_1019_quicklook/row5_judge.py
"""
import csv, json, math, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import chargpt_cl_1019_kune as K  # noqa: E402
C = K.C
R5 = ROOT / "results/chargpt_cl_1019/row5"; M2 = ROOT / "results/chargpt_cl_1019/M2"
SEEDS = range(10); P_STAR = 0.3321
FAMILY = [("C3self", "betaFA"), ("C3self", "Fswap"), ("C3self", "Fswapscale"), ("Frev", "capself")]


def sign_test(d):
    d = [x for x in d if np.isfinite(x)]; nz = [x for x in d if x != 0]; n = len(nz); k = sum(x > 0 for x in nz)
    if n == 0: return 1.0, 0, 0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail), k, n


def boot(d, seed=1021, B=20000):
    d = np.array([x for x in d if np.isfinite(x)]); rng = np.random.default_rng(seed)
    if len(d) == 0: return (np.nan, np.nan)
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1); return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def rows(sub, arm, s):
    f = R5 / sub / arm / f"seed{s}" / "per_task.csv"
    return [dict((k, float(v) if v not in ("True", "False") and k != "task" else (v == "True" if v in ("True", "False") else int(v))) for k, v in r.items()) for r in csv.DictReader(open(f))] if f.exists() else None


def prov(sub, arm, s):
    f = R5 / sub / arm / f"seed{s}" / "provenance.json"; return json.loads(f.read_text()) if f.exists() else None


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = [0.0] * m; run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


def main_family(lines, out):
    lines += ["## 主：確認的な族（課題 91〜100 の seed 内平均 E1000、両側 exact 符号検定、Holm）", ""]
    E = {}
    for arm in ("C3self", "capself", "betaFA", "Fswap", "Frev", "Fswapscale", "betaF"):
        for s in SEEDS:
            r = rows("main", arm, s)
            if r is None: E[(arm, s)] = np.nan; continue
            fin = all(x["finite"] for x in r); E[(arm, s)] = float(np.mean([x["E1000"] for x in r])) if fin and len(r) == 10 else np.nan
    res = []
    for a, b in FAMILY:
        d = [E[(a, s)] - E[(b, s)] for s in SEEDS]; p, k, n = sign_test(d); res.append(dict(cmp=f"{a} − {b}", diffs=d, p=p, k_pos=k, n_nonzero=n,
                                                                                       mean=float(np.nanmean(d)), median=float(np.nanmedian(d)), ci=boot(d), n_nonfinite=int(sum(not np.isfinite(x) for x in d))))
    adj = holm([r["p"] for r in res])
    for r, pa in zip(res, adj):
        r["p_holm"] = pa; maj = r["k_pos"] > r["n_nonzero"] - r["k_pos"]; mpos = r["mean"] > 0
        if r["n_nonzero"] == 0: lab = "効果を検出せず"
        elif pa < 0.05 and maj and mpos: lab = "予測方向の効果を検出"
        elif pa < 0.05 and (not maj) and (not mpos): lab = "逆方向を検出"
        elif maj != mpos: lab = "向きが不均一"
        else: lab = "効果を検出せず"
        r["label"] = lab
        lines.append(f"- {r['cmp']}：正の seed {r['k_pos']}/{r['n_nonzero']}（非有限 {r['n_nonfinite']}）、p = {r['p']:.4f}、Holm 後 {pa:.4f}、平均 {r['mean']:+.3f}（区間 {r['ci'][0]:+.3f}〜{r['ci'][1]:+.3f}）、中央 {r['median']:+.3f} → **{lab}**")
        lines.append("  - seed ごとの差: " + " ".join(f"{x:+.3f}" for x in r["diffs"]))
    lines += ["", "| seed | " + " | ".join(("C3self", "capself", "betaFA", "Fswap", "Frev", "Fswapscale", "betaF")) + " |", "| -- " * 8 + "|"]
    for s in SEEDS:
        lines.append(f"| {s} | " + " | ".join(f"{E[(a, s)]:.3f}" for a in ("C3self", "capself", "betaFA", "Fswap", "Frev", "Fswapscale", "betaF")) + " |")
    out["family"] = res; out["E1000"] = {f"{a}_{s}": v for (a, s), v in E.items()}
    return E


def operation(lines, out):
    lines += ["", "## 操作の成立", ""]
    op = {}
    for sub in ("main", "side"):
        b = [prov(sub, "betaFA", s) for s in SEEDS]; b = [x for x in b if x]
        withD = [x for x in b if x["n_D"] > 0]
        ok = [x["valid_D_eff_open_ge_half_pstar"] >= 0.5 for x in withD]       # seed ごと: D の unit の半分以上が p*/2 以上で開く
        mean_open = [x["valid_D_eff_open_mean"] for x in withD]
        op[f"{sub}_betaFA"] = dict(n_seed_withD=len(withD), n_ok=int(sum(ok)), D_sizes=[x["n_D"] for x in b], D_eff_open_mean=mean_open)
        lines.append(f"- {sub} β-FA：D が空でない seed {len(withD)} のうち、検証集合で D の unit の過半が p*/2 以上で開いた seed {sum(ok)}。D の大きさ {[x['n_D'] for x in b]}。"
                     f"D の実効の開く割合の平均 {[round(v, 3) for v in mean_open]}")
        for arm in ("Fswap", "Frev", "Fswapscale"):
            pv = [prov(sub, arm, s) for s in SEEDS]; pv = [x for x in pv if x]
            if not pv: continue
            mv = [abs(x["valid_eff_open_mean"] - x["valid_donor_open_mean"]) < abs(x["valid_nat_open_mean"] - x["valid_donor_open_mean"]) for x in pv]
            op[f"{sub}_{arm}"] = dict(n=len(pv), n_toward_donor=int(sum(mv)), eff=[x["valid_eff_open_mean"] for x in pv], nat=[x["valid_nat_open_mean"] for x in pv], donor=[x["valid_donor_open_mean"] for x in pv])
            lines.append(f"- {sub} {arm}：実効ゲートの開く割合が供与者側へ動いた seed {sum(mv)}/{len(pv)}（実効 {np.mean(op[f'{sub}_{arm}']['eff']):.3f}・自然 {np.mean(op[f'{sub}_{arm}']['nat']):.3f}・供与者 {np.mean(op[f'{sub}_{arm}']['donor']):.3f} の平均）")
    # 持続: β-FA の D の実効の開く割合（検証集合）の課題末の推移
    per = []
    for s in SEEDS:
        r = rows("main", "betaFA", s)
        if r and not math.isnan(r[0]["D_eff_open_valid"]): per.append([x["D_eff_open_ge_half_pstar_valid"] for x in r])
    if per:
        per = np.array(per); op["betaFA_persistence"] = per.mean(0).tolist()
        lines.append(f"- 持続（主 β-FA、D の unit のうち p*/2 以上で開く割合、課題 91〜100 の末、seed 平均）：{' '.join(f'{v:.2f}' for v in per.mean(0))}")
    out["operation"] = op


def secondary(lines, out, E):
    lines += ["", "## 副（記述のみ）", ""]
    arms = ("C3self", "capself", "betaFA", "Fswap", "Frev", "Fswapscale", "betaF")
    for sub, ar in (("main", arms), ("side", ("C3self", "capself", "betaFA", "Fswap"))):
        lines += [f"### {sub}（{'課題 91〜100' if sub == 'main' else '課題 51〜60'}、10 seed の中央値）", "", "| 腕 | T*（上限 10,000） | 未到達率 | E1000 | ELast | 最初の課題の E1000 | 有限 |", "| -- | -- | -- | -- | -- | -- | -- |"]
        for arm in ar:
            rr = [rows(sub, arm, s) for s in SEEDS]; rr = [r for r in rr if r]
            if not rr: continue
            T = [np.mean([x["T015"] for x in r]) for r in rr]; un = [np.mean([not x["reached"] for x in r]) for r in rr]
            e = [np.mean([x["E1000"] for x in r]) for r in rr]; el = [np.mean([x["ELast"] for x in r]) for r in rr]; e1 = [r[0]["E1000"] for r in rr]
            fin = sum(all(x["finite"] for x in r) for r in rr)
            lines.append(f"| {arm} | {np.median(T):.0f} | {np.median(un):.2f} | {np.median(e):.3f} | {np.median(el):.3f} | {np.median(e1):.3f} | {fin}/{len(rr)} |")
        lines.append("")
    G = []
    for s in SEEDS:
        den = E[("C3self", s)] - E[("capself", s)]
        G.append({a: (E[("C3self", s)] - E[(a, s)]) / den if np.isfinite(den) and den != 0 else np.nan for a in ("betaFA", "Fswap", "Fswapscale")})
    lines.append("- 対照間差の回復比 G = (C3self − 腕) / (C3self − capself)（媒介率ではない）の中央値：" + "、".join(f"{a} {np.nanmedian([g[a] for g in G]):.2f}" for a in ("betaFA", "Fswap", "Fswapscale")))
    d = [E[("Fswap", s)] - E[("Fswapscale", s)] for s in SEEDS]
    lines.append(f"- F-swap − F-swap-scale の直接差：平均 {np.nanmean(d):+.3f}・中央 {np.nanmedian(d):+.3f}（正の seed {sum(x > 0 for x in d if np.isfinite(x))}/10）")
    out["G"] = G


@torch.no_grad()
def c3_reference(s):
    """C3-2（追補 7）の L0 の課題ごとの回転 1 − cos の中央値と、L0 の出力の RMS（課題 t の鎖の probe）。課題 40〜50・90〜100。"""
    d = M2 / "C3" / f"seed{s}" / "ckpts"; out = {}
    prev = None
    for t in list(range(39, 51)) + list(range(89, 101)):
        m = C.build_model("C3", "M", s, "cpu"); m.load_state_dict(torch.load(d / f"task{t}.pt", map_location="cpu")); W = m.h[0].mlp.up.weight.clone()
        if prev is not None and prev[0] == t - 1:
            got = {}; hk = m.h[0].mlp.register_forward_hook(lambda mm, i, o: got.__setitem__("f0", o)); m.body(C.task_cpu_M(s, t, 1, 0.01)["probe"]); hk.remove()
            out[t] = dict(rot=float((1 - (F.normalize(W, dim=1) * F.normalize(prev[1], dim=1)).sum(1)).median()), rms=float(got["f0"].double().pow(2).mean().sqrt()))
        prev = (t, W)
    return out


def elr(lines, out):
    lines += ["", "## ELR（9.C、別の確認的比較）", ""]
    better = worse = 0; diffs = {k: [] for k in ("T015", "E1000", "ELast")}; capd = {k: [] for k in diffs}; opok_rot = opok_rms = 0; death100 = []; n_have = 0
    for s in SEEDS:
        f = R5 / "elr" / "elr" / f"seed{s}" / "per_task.csv"
        if not f.exists(): continue
        rr = {int(r["task"]): r for r in csv.DictReader(open(f))}
        if not all(t in rr for t in range(91, 101)): continue
        n_have += 1
        mine = {k: np.mean([float(rr[t][k]) for t in range(91, 101)]) for k in diffs}
        ref = {}; refc = {}
        for arm, dst in (("C3", ref), ("ReLUcap", refc)):
            on = np.load(M2 / arm / f"seed{s}" / "online.npz")["loss"]; pt = {int(r["task"]): float(r["H"]) for r in csv.DictReader(open(M2 / arm / f"seed{s}" / "per_task.csv"))}
            ph = []
            for t in range(91, 101):
                ex = on[t - 1].astype(np.float64) - pt[t]; ma = np.convolve(ex, np.ones(50) / 50, "valid"); hit = np.nonzero(ma <= 0.15)[0]
                ph.append(((int(hit[0]) + 50) if len(hit) else 10000, ex[950:1050].mean(), ex[-1000:].mean()))
            dst.update(T015=np.mean([x[0] for x in ph]), E1000=np.mean([x[1] for x in ph]), ELast=np.mean([x[2] for x in ph]))
        b = all(mine[k] < ref[k] for k in diffs); w = all(mine[k] > ref[k] for k in diffs); better += b; worse += w
        for k in diffs: diffs[k].append(mine[k] - ref[k]); capd[k].append(mine[k] - refc[k])
        cr = c3_reference(s)
        rot_ok = np.median([float(rr[t]["rot0_med"]) / cr[t]["rot"] for t in cr if t in rr]) >= 10
        rms_ok = all(abs(float(rr[t]["ffn0_out_rms"]) / cr[t]["rms"] - 1) <= 0.2 for t in cr if t in rr)
        opok_rot += rot_ok; opok_rms += rms_ok
        ls = [r for r in csv.DictReader(open(R5 / "elr" / "elr" / f"seed{s}" / "layer_stats.csv")) if int(r["task"]) == 100 and r["probe"] == "chain" and int(r["layer"]) == 0]
        death100.append(float(ls[0]["frac_p_lt005"]) if ls else np.nan)
    lab = "ELR_RESCUES" if better >= 8 else ("ELR_HARMS" if worse >= 8 else "MIXED")
    lines.append(f"- seed {n_have}/10 がそろった。3 相すべてで C3-2 より良い seed {better}、すべて悪い seed {worse} → **{lab}**")
    for k in diffs:
        p, kp, n = sign_test(diffs[k]); lines.append(f"  - {k}：ELR − C3-2 の中央 {np.median(diffs[k]):+.3f}（正 {kp}/{n}、p = {p:.4f}）、ELR − ReLUcap-2 の中央 {np.median(capd[k]):+.3f}")
    lines.append(f"- 操作の成立：回転が C3-2 の 10 倍以上 {opok_rot}/{n_have}、L0 の出力の RMS が C3-2 の ±20% 以内 {opok_rms}/{n_have}")
    lines.append(f"- 課題 100 の L0 の死：{[round(x, 2) for x in death100]}（0.1 以下 {sum(x <= 0.1 for x in death100 if np.isfinite(x))}/{len(death100)}）")
    out["elr"] = dict(label=lab, better=better, worse=worse, diffs=diffs, vs_cap=capd, op_rot=opok_rot, op_rms=opok_rms, death100=death100)
    f = R5 / "elr" / "literal" / "seed0" / "per_task.csv"
    if f.exists():
        rr = list(csv.DictReader(open(f)))
        lines.append("- 文字どおりの形の診断（seed 0）：L0 の ‖w‖ の中央値の、課題 1 末に対する比 = " + " ".join(f"t{r['task']}:{float(r['w0_norm_over_n1_med']):.3g}" for r in rr)
                     + (f"（課題 {json.loads((f.parent / 'provenance.json').read_text()).get('stopped_nonfinite_at')} で非有限）" if (f.parent / "provenance.json").exists() and json.loads((f.parent / "provenance.json").read_text()).get("stopped_nonfinite_at") else ""))


def main():
    lines = ["# 追補 9 の判定（2 ブロックで L0 の応答だけを操作する介入）", ""]; out = {}
    E = main_family(lines, out); operation(lines, out); secondary(lines, out, E); elr(lines, out)
    (R5 / "judge.md").write_text("\n".join(lines)); (R5 / "judge.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float)); print("\n".join(lines))


if __name__ == "__main__":
    main()

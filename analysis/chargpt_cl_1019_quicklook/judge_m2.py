#!/usr/bin/env python3
"""spec 追補 6 の 3 条件と記録だけの項目（2 ブロックの箱、下見・未登録）。
  CHARGPT_NLAYER=2 python analysis/chargpt_cl_1019_quicklook/judge_m2.py --root results/chargpt_cl_1019/quicklook/M2 --decomp results/chargpt_cl_1019/quicklook/decomp_dmu_M2 --out results/chargpt_cl_1019/quicklook/judge_m2
閾値は追補 6 §4 で固定した値（後から動かさない）。L_top = 読み出しに隣接する FFN（2 ブロックでは L1）。
"""
import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import chargpt_cl_1019_kune as K  # noqa: E402
C = K.C
SEED, ALPHA = 1000, 0.01
DELTA = 3 * 0.0084                 # 喪失の幅 δ（本走 10 seed の C3 の fresh gap の SE の 3 倍）
PUSH_TH, NEGFRAC_TH = -0.206, 0.75  # 切替の押し（参照 −0.412 の半分）、負の課題の割合
KAPPA_TH = 0.8 * (0.606 - 0.25) / 0.75   # 一様からの超過で比べた集中度の閾値（κ_ref = 0.475 の 0.8 倍 = 0.380）
UPSHARE_TH = 0.8 * 0.770           # L_top の負の量のうち上流の割合
RESP_DELTA, LEAD = 0.10, 5         # 応答しない割合の増え、先行・同時の許容（課題）


def rd(p): return list(csv.DictReader(open(p)))


def learning(d):
    pt = rd(d / "per_task.csv"); n = len(pt)
    on = np.array([float(r["online_all"]) for r in pt]); H = np.array([float(r["H"]) for r in pt])
    loss = np.load(d / "online.npz")["loss"].astype(np.float64); head = (loss[:, :100] - H[:, None]).mean(1)
    fr = {int(r["task"]): float(r["online_all"]) for r in rd(d / "fresh.csv")}
    gap = {t: on[t - 1] - fr[t] for t in fr if t <= n}
    end1 = float(pt[0]["end_metric"])
    early, late = on[1:10].mean(), (on[90:100].mean() if n >= 100 else float("nan"))
    return dict(n=n, on=on, head=head, gap=gap, end1=end1, early=early, late=late,
                head_early=head[1:10].mean(), head_late=(head[90:100].mean() if n >= 100 else float("nan")))


def layer_end(d, n, probe):
    E = {}
    for r in rd(d / "layer_stats.csv"):
        if r["probe"] == probe and int(r["u"]) == 10000:
            E[(int(r["task"]), int(r["layer"]))] = r
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "results/chargpt_cl_1019/quicklook/M2"))
    ap.add_argument("--decomp", default=str(ROOT / "results/chargpt_cl_1019/quicklook/decomp_dmu_M2"))
    ap.add_argument("--out", default=str(ROOT / "results/chargpt_cl_1019/quicklook/judge_m2"))
    ap.add_argument("--arms", nargs="+", default=["C3", "SiLU"])
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()
    global SEED; SEED = a.seed; root, out = Path(a.root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    nl = C.L; top = nl - 1; res = {"n_layer": nl, "thresholds": dict(delta=DELTA, push=PUSH_TH, negfrac=NEGFRAC_TH, kappa=KAPPA_TH, c_min=1 / nl + KAPPA_TH * (1 - 1 / nl), upshare=UPSHARE_TH)}
    lines = [f"# 追補 6 の判定（n_layer = {nl}、seed {SEED}、下見・未登録）", ""]
    for arm in a.arms:
        d = root / arm / f"seed{SEED}"
        if not (d / "per_task.csv").exists():
            lines.append(f"- {arm}: 結果なし"); continue
        L = learning(d); n = L["n"]; R = {}
        # ---- 条件 1
        c1 = dict(end1=L["end1"], end1_ok=L["end1"] <= 0.1, early=L["early"], late=L["late"], worse=L["late"] - L["early"],
                  gap={str(k): v for k, v in sorted(L["gap"].items())}, head_early=L["head_early"], head_late=L["head_late"])
        if 10 in L["gap"] and 100 in L["gap"]:
            c1["gap_shift"] = L["gap"][100] - L["gap"][10]
        if arm == "C3":
            c1["pass"] = bool(c1["end1_ok"] and c1["worse"] >= DELTA and c1.get("gap_shift", -1) >= DELTA and L["head_late"] > L["head_early"])
        else:
            c1["pass"] = bool(c1["end1_ok"])                    # SiLU は失わなくてよい
        R["cond1"] = c1
        # ---- 条件 2（切替の押し・応答の喪失）
        U = np.load(d / "unit_stats.npz")
        s = np.median((U["chain_mu_u100"].astype(float) - U["chain_mu_u0"].astype(float)) / (U["chain_sd_u0"].astype(float) + 1e-12), axis=2)   # [課題, 層]
        pu = 1 - U["uniform_q"].astype(float)                    # 固定評価集合での p
        Rt = ((pu < 0.005) | (pu >= 0.99)).mean(2)               # [課題, 層]
        onset_resp = next((t for t in range(2, n + 1) if Rt[t - 1, top] - Rt[0, top] >= RESP_DELTA), None)
        onset_loss = next((t for t in range(11, n + 1) if L["on"][t - 5:t].mean() - L["early"] >= DELTA), None)
        c2 = dict(push_mean_2_50=float(s[1:50, top].mean()), negfrac_2_50=float((s[1:50, top] < 0).mean()),
                  push_mean_51_100=float(s[50:100, top].mean()) if n >= 100 else None, negfrac_51_100=float((s[50:100, top] < 0).mean()) if n >= 100 else None,
                  push_by_layer_2_50=[float(s[1:50, l].mean()) for l in range(nl)], onset_resp=onset_resp, onset_loss=onset_loss,
                  R_top={str(t): float(Rt[t - 1, top]) for t in (1, 5, 10, 25, 50, 75, 100) if t <= n})
        lead_ok = None if onset_loss is None else (onset_resp is not None and onset_resp <= onset_loss + LEAD)
        c2["lead_ok"] = lead_ok
        c2["pass"] = bool(c2["push_mean_2_50"] <= PUSH_TH and c2["negfrac_2_50"] >= NEGFRAC_TH and (lead_ok is True)) if arm == "C3" else None
        R["cond2"] = c2
        # p の分布と Var_x(φ′)（固定評価集合、L_top）
        Eu = layer_end(d, n, "uniform"); Ec = layer_end(d, n, "chain")
        bins = [(0, 0.005), (0.005, 0.02), (0.02, 0.1), (0.1, 0.3), (0.3, 0.99), (0.99, 1.01)]
        R["p_dist_top"] = {str(t): [float(((pu[t - 1, top] >= lo) & (pu[t - 1, top] < hi)).mean()) for lo, hi in bins] for t in (1, 5, 10, 25, 50, 75, 100) if t <= n}
        R["phi1_var_med_top"] = {str(t): float(Eu[(t, top)]["phi1_var_med"]) for t in (1, 5, 10, 25, 50, 75, 100) if t <= n and "phi1_var_med" in Eu[(t, top)]}
        # ---- 条件 3（分解）
        dd = Path(a.decomp) / "decomp_dmu.csv"
        if dd.exists():
            rows = [r for r in rd(dd) if r["arm"] == arm]
            def blk(t0, t1):
                sel = [r for r in rows if r["window"] == "i" and t0 <= int(r["task"]) <= t1]
                if not sel: return None
                nd = [np.mean([float(r["neg_dmu"]) for r in sel if int(r["layer"]) == l]) for l in range(nl)]
                nu = np.mean([float(r["neg_up"]) for r in sel if int(r["layer"]) == top]); no = np.mean([float(r["neg_own"]) for r in sel if int(r["layer"]) == top])
                c = nd[top] / sum(nd); kappa = (c - 1 / nl) / (1 - 1 / nl)
                return dict(neg_by_layer=[float(x) for x in nd], c=float(c), kappa=float(kappa), upshare=float(nu / (nu + no)), cancel=float(nu + no - nd[top]),
                            resid_max=max(float(r["resid"]) for r in sel))
            c3 = {"t2-10": blk(2, 10), "t40-50": blk(40, 50), "t91-100": blk(91, 100)}
            b = c3["t2-10"]
            c3["pass"] = bool(b and b["resid_max"] <= 1e-8 and b["kappa"] >= KAPPA_TH and b["upshare"] >= UPSHARE_TH) if arm == "C3" else None
            R["cond3"] = c3
        # ---- L_top の担い手の出所（埋め込み・注意・FFN の項ごと、checkpoint の課題）
        orig = {}
        for t in (1, 10, 50, 100):
            ck = d / "ckpts" / f"task{t}.pt"
            if not ck.exists(): continue
            m = C.build_model(arm, "M", SEED, "cpu"); m.load_state_dict(torch.load(ck, map_location="cpu")); m.eval()
            probe = C.task_cpu_M(SEED, t, 10, ALPHA)["probe"]
            with torch.no_grad():
                h = m.wte(probe); srcs = [("emb", h)]
                for li, bk in enumerate(m.h):
                    at = bk.sc * bk.attn(bk.n1(h)); h = h + at; srcs.append((f"attn{li}", at))
                    if li == top:
                        X = h.reshape(-1, C.D).double(); xb = X.mean(0); xb2 = float(xb.pow(2).sum())
                        orig[str(t)] = {k: float((v.reshape(-1, C.D).double().mean(0) @ xb) / xb2) for k, v in srcs}
                        orig[str(t)]["attn0_rms"] = float(srcs[1][1].double().pow(2).mean().sqrt()); orig[str(t)]["ln1_gamma0"] = float(m.h[0].n1.weight.pow(2).mean().sqrt())
                        break
                    f = bk.mlp(bk.n2(h)); h = h + f; srcs.append((f"ffn{li}", f))
        R["carrier_origin_top"] = orig
        # ---- 記録だけ: ゲート層、役割の移動
        gate = {}
        for probe, E in (("uniform", Eu), ("chain", Ec)):
            seq = []
            for t in range(1, n + 1):
                st = []
                for l in range(nl):
                    r = E[(t, l)]; pm = float(r["p_mean"]) if "p_mean" in r else 1 - float(r["neg_frac"])
                    st.append("C" if pm < 0.01 else ("L" if float(r["r_med"]) > 0 and float(r["phi1_mean"]) >= 0.75 else "G"))
                g = [l for l in range(nl) if st[l] == "G"]
                seq.append(("".join(st), (max(g) if g else None)))
            gate[probe] = seq
        R["gating"] = {p: {str(t): {"states": s_[0], "nearest_gate": s_[1]} for t, s_ in enumerate(v, 1) if t in (1, 2, 3, 5, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)} for p, v in gate.items()}
        rr = [float(Ec[(t, top)]["r_med"]) for t in range(1, n + 1)]; ph = [float(Ec[(t, top)]["phi1_mean"]) for t in range(1, n + 1)]
        esc = next((t for t in range(1, n - 3) if all(rr[k - 1] > 1 and ph[k - 1] >= 0.75 for k in range(t, t + 5))), None)
        r0 = [float(Ec[(t, 0)]["r_med"]) for t in range(1, n + 1)]
        drop = None
        if esc is not None:
            drop = next((t for t in range(max(esc, 6), n + 1) if r0[t - 1] <= np.mean(r0[t - 6:t - 1]) - 1.0), None)
        R["role_shift"] = dict(top_escape_task=esc, l0_drop_task=drop, r_top={str(t): rr[t - 1] for t in (1, 10, 25, 50, 75, 100) if t <= n},
                               r_l0={str(t): r0[t - 1] for t in (1, 10, 25, 50, 75, 100) if t <= n})
        res[arm] = R
        # ---- 表
        lines += [f"## {arm}", "",
                  f"- 条件 1: 課題 1 末 {L['end1']:.3f}（≤ 0.1: {c1['end1_ok']}）| 超過損失 2〜10 {L['early']:.4f} → 91〜100 {L['late']:.4f}（悪化 {c1['worse']:+.4f}、δ {DELTA:.3f}）"
                  f" | fresh gap " + " / ".join(f"{k}: {v:+.3f}" for k, v in sorted(L['gap'].items())) + f"（100 − 10 = {c1.get('gap_shift', float('nan')):+.3f}）"
                  f" | 頭 100 更新 {L['head_early']:.2f} → {L['head_late']:.2f} | 合否 {c1['pass']}",
                  f"- 条件 2: L{top} の切替の押し 課題 2〜50 平均 {c2['push_mean_2_50']:+.3f}（閾値 {PUSH_TH}）負の課題 {c2['negfrac_2_50']:.2f}（≥ {NEGFRAC_TH}）| 51〜100 {c2['push_mean_51_100']} / {c2['negfrac_51_100']}"
                  f" | 層ごと（2〜50）" + " ".join(f"L{l} {v:+.3f}" for l, v in enumerate(c2['push_by_layer_2_50'])) + f" | onset_resp {onset_resp}・onset_loss {onset_loss}（先行・同時 {lead_ok}）| 合否 {c2['pass']}",
                  f"- L{top} の応答しない割合 R_t: " + " ".join(f"t{k} {v:.2f}" for k, v in c2['R_top'].items()),
                  f"- L{top} の p の分布（固定評価集合、[<.005, .005-.02, .02-.1, .1-.3, .3-.99, ≥.99]）: " + " | ".join(f"t{k} " + "/".join(f"{x:.2f}" for x in v) for k, v in R['p_dist_top'].items()),
                  f"- L{top} の Var_x(φ′) 中央: " + " ".join(f"t{k} {v:.3f}" for k, v in R['phi1_var_med_top'].items())]
        if "cond3" in R:
            for k in ("t2-10", "t40-50", "t91-100"):
                b = R["cond3"][k]
                if b: lines.append(f"- 条件 3 {k}: 負の量 " + " ".join(f"L{l} {v:.3f}" for l, v in enumerate(b['neg_by_layer'])) + f" | L{top} の割合 c {b['c']:.3f}（κ {b['kappa']:.3f}、閾値 κ ≥ {KAPPA_TH:.3f}）| 上流の割合 {b['upshare']:.3f}（≥ {UPSHARE_TH:.3f}）| 相殺 {b['cancel']:.3f} | 残差 {b['resid_max']:.1e}")
            lines.append(f"- 条件 3 合否: {R['cond3']['pass']}")
        lines.append(f"- L{top} の担い手の出所: " + " | ".join(f"t{t} " + " ".join(f"{k} {v:+.2f}" for k, v in o.items() if not k.endswith('rms') and not k.startswith('ln1')) + f"（注意 0 の rms {o['attn0_rms']:.3f}、LN1 γ {o['ln1_gamma0']:.3f}）" for t, o in orig.items()))
        lines.append(f"- ゲート層（固定評価集合）: " + " ".join(f"t{t} {v['states']}→{v['nearest_gate']}" for t, v in R['gating']['uniform'].items()))
        lines.append(f"- ゲート層（chain probe）: " + " ".join(f"t{t} {v['states']}→{v['nearest_gate']}" for t, v in R['gating']['chain'].items()))
        lines.append(f"- 役割の移動: L{top} が正側へ出た課題 {esc}、そのあと L0 が下がった課題 {drop} | μ/σ L{top} " + " ".join(f"t{k} {v:+.2f}" for k, v in R['role_shift']['r_top'].items()) + " | L0 " + " ".join(f"t{k} {v:+.2f}" for k, v in R['role_shift']['r_l0'].items()))
        lines.append("")
    (out / "judge_m2.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    (out / "judge_m2.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

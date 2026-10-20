#!/usr/bin/env python3
"""ρ の分割と κ_R の事後解析から、依頼の Q1〜Q4 と予測 P1〜P7 に答える表を出す（事後・判定外、seed 1000、4 ブロック）。
入力: rho/out/<arm>/t{t}_{a|b}_chain.json（rho_split.py）、rho/kappa/<arm>/t{t}.json（kappa_split.py）、rho/summary/rho_summary.json。
  python analysis/chargpt_cl_1019_quicklook/rho_kappa_answers.py
"""
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
Q = ROOT / "results/chargpt_cl_1019/quicklook"
OUT = Q / "rho/summary"
BLOCKS = {"C3": {"t2-10": range(2, 11), "t40-50": range(40, 51)},
          "SiLU": {"t2-10": range(2, 11), "t11-15": range(11, 16), "t40-50": range(40, 51), "t60-100": range(60, 101, 5)}}
COMP = ("own", "self_up", "others_same", "others_path")
NAME = {"own": "①", "self_up": "②", "others_same": "③a", "others_path": "③b"}


def rho_units(arm, ts, st, l, mt):
    acc = {k: [] for k in COMP + ("alive",)}
    for t in ts:
        f = Q / "rho/out" / arm / f"t{t}_{st}_chain.json"
        if not f.exists(): continue
        L = json.loads(f.read_text())["layers"].get(str(l))
        if L is None or mt not in L: continue
        for k in COMP: acc[k] += L[mt][k]
        acc["alive"] += L["alive"]
    return {k: np.array(v, dtype=float) for k, v in acc.items()}


def kappa(arm, t):
    f = Q / "rho/kappa" / arm / f"t{t}.json"
    return json.loads(f.read_text()) if f.exists() else None


def carriers(u):
    """層の平均の押し（unit の平均）を成分に割る。浮く unit・沈む unit（①+②+③ の符号）ごとの成分の和の割合も。"""
    tot = sum(u[k] for k in COMP); nz = (np.abs(u["own"]) + np.abs(u["self_up"])) > 0
    m = {k: float(u[k].mean()) for k in COMP}; mt = float(tot.mean())
    mass = {k: float(np.abs(u[k]).sum()) for k in COMP}; ms = sum(mass.values())
    up, dn = tot > 0, tot < 0
    sh = lambda sel: {k: float(u[k][sel].sum() / (tot[sel].sum() + 1e-300)) for k in COMP}
    opp = float((np.sign(u["others_same"] + u["others_path"]) != np.sign(u["own"]))[nz].mean()) if nz.any() else np.nan
    return dict(mean=m, mean_tot=mt, mass_share={k: mass[k] / ms for k in COMP}, share_up=sh(up), share_dn=sh(dn),
                frac_up=float(up.mean()), oth_opp_own=opp, n=int(len(tot)))


def fmt_c(c):
    return (f"平均 ① {c['mean']['own']:+.2e} ② {c['mean']['self_up']:+.2e} ③a {c['mean']['others_same']:+.2e} ③b {c['mean']['others_path']:+.2e}（計 {c['mean_tot']:+.2e}）"
            f" | 量の割合 {'/'.join(f'{c['mass_share'][k]:.2f}' for k in COMP)}"
            f" | 浮く unit {c['frac_up']:.2f}: 和の内訳 {'/'.join(f'{c['share_up'][k]:+.2f}' for k in COMP)}"
            f" | 沈む unit: {'/'.join(f'{c['share_dn'][k]:+.2f}' for k in COMP)} | ③ が ① と逆 {c['oth_opp_own']:.2f}")


def mu_traj(arm, layer):
    """課題末の μ/σ の中央値（layer_stats の r_med、chain probe、課題末の更新）。"""
    out = {}
    for f in [Q / "M" / arm / "seed1000/layer_stats.csv"] + sorted((Q / "rho/cont").glob(f"{arm}_*/layer_stats.csv")):
        if not f.exists(): continue
        rows = [r for r in csv.DictReader(open(f)) if r["probe"] == "chain" and int(r["layer"]) == layer]
        last = {}
        for r in rows:
            t, u = int(r["task"]), int(r["u"])
            if t not in last or u > last[t][0]: last[t] = (u, float(r["r_med"]), float(r.get("dead", "nan") or "nan"))
        for t, v in last.items(): out.setdefault(t, v)
    return dict(sorted(out.items()))


def main():
    L = ["# ρ の分割と κ_R：Q1〜Q4 と P1〜P7（事後・判定外、seed 1000、4 ブロック）", ""]
    res = {}
    # ---- Q1 / P2: C3 の L3 の切替の押し
    L += ["## Q1・P2　C3 の L3（切替の瞬間 = 状態 a と、更新 100 = 状態 b）", ""]
    for b in ("t2-10", "t40-50"):
        for st in ("a", "b"):
            for mt in ("A", "D"):
                u = rho_units("C3", BLOCKS["C3"][b], st, 3, mt); c = carriers(u)
                oth = u["others_same"] + u["others_path"]; nz = (np.abs(u["own"]) + np.abs(u["self_up"])) > 0
                rho = np.abs(u["own"] + u["self_up"]) / (np.abs(oth) + 1e-300)
                c.update(rho_med=float(np.median(rho[nz])), oth_same_sign_own=float((np.sign(oth) == np.sign(u["own"]))[nz].mean()))
                res[f"Q1_C3_L3_{b}_{st}_{mt}"] = c
                L.append(f"- {b} 状態{st} {mt}: {fmt_c(c)} | ρ 中央 {c['rho_med']:.2f} | ③ の符号が ① と同じ {c['oth_same_sign_own']:.2f}")
    # ---- Q2 / P3: SiLU の L3 が課題 10〜15 に正側へ出る
    L += ["", "## Q2・P3　SiLU の L3 が正側へ出る課題 11〜15（と課題 40〜50 の上向き）", ""]
    for b in ("t11-15", "t40-50", "t2-10"):
        for st in ("a", "b"):
            for mt in ("A", "D"):
                u = rho_units("SiLU", BLOCKS["SiLU"][b], st, 3, mt); c = carriers(u); res[f"Q2_SiLU_L3_{b}_{st}_{mt}"] = c
                L.append(f"- {b} 状態{st} {mt}: {fmt_c(c)}")
    # ---- Q3: SiLU の L2 が沈んで死ぬ課題 70〜100
    L += ["", "## Q3　SiLU の L2（課題 60〜100、5 課題おき）", ""]
    for name, ts in (("t60-65", (60, 65)), ("t70-100", range(70, 101, 5))):
        for st in ("a", "b"):
            for mt in ("A", "D"):
                u = rho_units("SiLU", ts, st, 2, mt); c = carriers(u)
                oth = u["others_same"] + u["others_path"]; nz = (np.abs(u["own"]) + np.abs(u["self_up"])) > 0
                rho = np.abs(u["own"] + u["self_up"]) / (np.abs(oth) + 1e-300); al = nz & (u["alive"] > 0)
                agree = float((np.sign(u["own"]) == np.sign(u["self_up"]))[al].mean()) if al.any() else np.nan
                c.update(rho_q=[float(v) for v in np.percentile(rho[nz], [25, 50, 75])] if nz.any() else [np.nan] * 3, agree_alive=agree, n_alive=int(al.sum()))
                res[f"Q3_SiLU_L2_{name}_{st}_{mt}"] = c
                L.append(f"- {name} 状態{st} {mt}: {fmt_c(c)} | ρ 四分位 {'/'.join(f'{v:.2f}' for v in c['rho_q'])} | 生きた unit {c['n_alive']} で ①② 一致 {agree:.2f}")
    # ---- P1: ①②の符号一致（生きた unit）
    L += ["", "## P1　①と②の符号一致（生きた unit = probe で開く unit、①・② が厳密に 0 の unit は除く）", ""]
    cells = []
    for arm, bl in BLOCKS.items():
        for b, ts in bl.items():
            for st in ("a", "b"):
                for mt in ("A", "D"):
                    for l in range(4):
                        u = rho_units(arm, ts, st, l, mt)
                        if not len(u["own"]): continue
                        al = (u["alive"] > 0) & ((np.abs(u["own"]) + np.abs(u["self_up"])) > 0)
                        if not al.any(): continue
                        ag = float((np.sign(u["own"]) == np.sign(u["self_up"]))[al].mean())
                        mag = np.abs(u["own"][al]) + np.abs(u["self_up"][al])
                        agw = float((mag * (np.sign(u["own"]) == np.sign(u["self_up"]))[al]).sum() / mag.sum())
                        cells.append((arm, b, st, mt, l, ag, agw, int(al.sum())))
    ags = np.array([c[5] for c in cells]); agws = np.array([c[6] for c in cells])
    L.append(f"- マス {len(cells)}: 一致の中央 {np.median(ags):.2f}、最小 {ags.min():.2f}、0.80 以上のマス {(ags >= 0.8).mean():.2f}。量で重みづけると中央 {np.median(agws):.2f}、最小 {agws.min():.2f}")
    for c in sorted(cells, key=lambda c: c[5])[:8]:
        L.append(f"  - 低い順: {c[0]} {c[1]} 状態{c[2]} {c[3]} L{c[4]}: 一致 {c[5]:.2f}（量で重みづけ {c[6]:.2f}、生きた unit {c[7]}）")
    res["P1_cells"] = cells
    # ---- P4: 伸びと回転
    sm = json.loads((OUT / "rho_summary.json").read_text())
    sr = {(s["arm"], s["block"], s["layer"]): s for s in sm["stretch_rotation"]}
    L += ["", "## Q4・P4　伸びと回転（実際の窓の w·Δȳ を σ で割る）", ""]
    for key in (("C3", "t2-10", 3), ("C3", "t40-50", 3), ("SiLU", "t11-15", 3), ("SiLU", "t40-50", 3), ("SiLU", "t60-100", 2)):
        s = sr.get(key)
        if s: L.append(f"- {key[0]} {key[1]} L{key[2]}: 伸びの割合 全 {s['stretch_share']:.2f}、沈む unit {s['stretch_share_neg']:.2f}、浮く unit {s['stretch_share_pos']:.2f}（中央 伸び {s['med_stretch']:+.3f}・回転 {s['med_rot']:+.3f}）")
    # ---- P5〜P7: κ_R
    L += ["", "## P5　unit ごとの κ_R（κ_R > 0 が沈める側。生きた unit = probe で開く unit）", ""]
    L.append("| 腕 | 課題 | 層 | κ_R 四分位 | κ_R < 0（生きた unit） | sign(κ_R) = −sign(実際の Δm) | sign(κ_R) = −sign(①, A) | 恒等式の誤差 |")
    L.append("| -- | -- | -- | -- | -- | -- | -- | -- |")
    krows = []
    for arm, ts in (("C3", list(range(2, 11)) + list(range(40, 51))), ("SiLU", list(range(2, 16)) + [20, 25, 30, 35] + list(range(40, 51)) + list(range(60, 101, 5)))):
        for t in ts:
            k = kappa(arm, t)
            if k is None: continue
            ra = Q / "rho/out" / arm / f"t{t}_a_chain.json"; ro = json.loads(ra.read_text())["layers"] if ra.exists() else None
            for l in (2, 3):
                Ld = k["layers"][str(l)]; kr = np.array(Ld["kappa_R"]); al = np.array(Ld["alive"], bool)
                dm = np.array(Ld["dmu_window"]) if "dmu_window" in Ld else None
                ag_dm = float((np.sign(kr) == -np.sign(dm))[al].mean()) if dm is not None and al.any() else np.nan
                ag_own = np.nan
                if ro is not None:
                    own = np.array(ro[str(l)]["A"]["own"]); sel = al & (own != 0)
                    ag_own = float((np.sign(kr) == -np.sign(own))[sel].mean()) if sel.any() else np.nan
                q = np.percentile(kr[al], [25, 50, 75]) if al.any() else [np.nan] * 3
                r = dict(arm=arm, task=t, layer=l, q=[float(v) for v in q], neg=float((kr[al] < 0).mean()) if al.any() else np.nan, n_alive=int(al.sum()),
                         ag_dm=ag_dm, ag_own=ag_own, chk=Ld["check_identity"])
                krows.append(r)
                L.append(f"| {arm} | {t} | L{l} | {q[0]:+.2e} / {q[1]:+.2e} / {q[2]:+.2e} | {r['neg']:.2f}（{r['n_alive']}） | {ag_dm:.2f} | {ag_own:.2f} | {r['chk']:.0e} |")
    res["kappa_rows"] = krows
    L += ["", "## P6・P7　SiLU の L2 の読み出しの割り（直通・L3 経由）と L3 の非線形の残り", ""]
    mt2, mt3 = mu_traj("SiLU", 2), mu_traj("SiLU", 3)
    L.append("| 課題 | ‖s_viaL3‖/‖s_direct‖ の中央 | κ_R 中央（全体 / 直通 / L3 経由） | L2 の μ/σ 中央・死 | L3 の μ/σ 中央 | L3 の非線形の残り |")
    L.append("| -- | -- | -- | -- | -- | -- |")
    p6 = []
    for t in list(range(2, 16)) + [20, 25, 26, 30, 35] + list(range(40, 51)) + list(range(60, 101, 5)):
        k = kappa("SiLU", t)
        if k is None or "L2_split" not in k: continue
        s = k["L2_split"]; al = np.array(k["layers"]["2"]["alive"], bool)
        ratio = np.array(s["s_norm_viaL3"]) / (np.array(s["s_norm_direct"]) + 1e-300)
        kall = np.array(k["layers"]["2"]["kappa_R"]); kd = np.array(s["kappa_R_direct"]); kv = np.array(s["kappa_R_viaL3"])
        sel = al if al.any() else np.ones_like(al)
        m2 = mt2.get(t - 1, (None, np.nan, np.nan)); m3 = mt3.get(t - 1, (None, np.nan, np.nan))
        row = dict(task=t, ratio=float(np.median(ratio[sel])), k_all=float(np.median(kall[sel])), k_dir=float(np.median(kd[sel])), k_via=float(np.median(kv[sel])),
                   mu2=m2[1], dead2=m2[2], mu3=m3[1], resid=k.get("L3_nonlin_resid", np.nan), n_alive=int(al.sum()))
        p6.append(row)
        L.append(f"| {t} | {row['ratio']:.2f} | {row['k_all']:+.2e} / {row['k_dir']:+.2e} / {row['k_via']:+.2e} | {row['mu2']:+.2f}・{row['dead2']:.2f} | {row['mu3']:+.2f} | {row['resid']:.3f} |")
    res["p6_rows"] = p6
    L += ["", "μ/σ と死は課題 k の末（窓の直前、t = k + 1 の行に置く）。生きた unit が 0 の課題は全 unit で中央を取る。"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rho_kappa_answers.md").write_text("\n".join(L)); (OUT / "rho_kappa_answers.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()

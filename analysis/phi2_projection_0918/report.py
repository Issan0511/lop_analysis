"""Evaluate the pre-registered predictions P1-P9 of specs/spec_phi2_projection_0918.md from the raw CSVs.
Writes results/phi2_projection_0918/summary.md, verdict.json and two figures. No new dynamics here."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phi2_projection_0918"
A = {"LR_a0p1_q0": 0.1, "LR_a0p7_q0": 0.7}
T_STAR = 100          # pre-registered: "そのとき" = after δŷ has fallen (P3 says t<=100)


def load():
    one = pd.read_csv(OUT / "onestep_eps0.01.csv")
    per = pd.read_csv(OUT / "perturb_eps0.01.csv")
    lin1 = pd.read_csv(OUT / "onestep_lin_eps0.001.csv") if (OUT / "onestep_lin_eps0.001.csv").exists() else None
    lin = pd.read_csv(OUT / "perturb_lin_eps0.001.csv") if (OUT / "perturb_lin_eps0.001.csv").exists() else None
    sw = pd.read_csv(OUT / "switch.csv")
    for d in (one, per, lin1, lin):
        if d is not None:
            d["a"] = d.arm.map(A)
    sw["a"] = sw.arm.map(A)
    return one, per, lin1, lin, sw


def fmt(x, n=3):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}g}"


def main():
    one, per, lin1, lin, sw = load()
    V, L = {}, []
    ok1 = one[one.ok]
    okp = per[per.ok].copy()
    okp["cond"] = okp.arm + "_" + okp.step.astype(str)
    okp["f_mu"] = 1 - okp.q_mu / okp.q_mu0            # erased fraction of the muhat projection
    okp["f_PA"] = 1 - okp.q_PA / okp.q_PA0            # erased fraction of the S_A norm
    okp["f_gn_mu"] = 1 - okp.gn_q_mu / okp.q_mu0
    okp["f_gn_PA"] = 1 - okp.gn_q_PA / okp.q_PA0
    key = ["cond", "group", "seed", "unit"]

    # ---------------- P0: Issa's primary target — the boxed task-averaged formula, three regimes
    fbox = OUT / "boxed.csv"
    if fbox.exists():
        bx = pd.read_csv(fbox)
        L.append("## P0 箱の式 $q^+=(I-2\\eta v^2\\mathbb E_A[\\phi'(z)^2xx^\\top])q$ の直接検証（Issa 指定の主対象・事後）\n")
        L.append("同じ純正側/純負側 unit に q=εμ̂ を置き、$H_A=2v_u^2\\,\\frac1{32}\\sum_pk_{u,p}^2x_px_p^\\top$ を 32 パターンで厳密に計算して $(I-\\eta H_A)^tq_0$ と実測の差 $q_t$ を比べる。(a) w_u だけを full-batch GD（期待値そのもの）、(b) w_u だけを batch-1 SGD（条件付き平均の形）、(c) 全 W/b/v/c を自然更新。誤差は ‖q_t − 予測‖/‖q_0‖。\n")
        L.append("| 領域 | t | n | 誤差 最大 | 誤差 中央値 | cos(実測,予測) 中央値 | 可視成分の生存率 実測 | 同 予測 | 枝不一致 最大 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for mode, label in (("fullbatch_wonly", "(a) full-batch・w_u のみ"), ("sgd_wonly", "(b) batch-1・w_u のみ"), ("natural", "(c) 自然学習・全パラメータ")):
            d = bx[bx["mode"] == mode]
            for t in (1, 10, 100, 1000, 2000):
                dd = d[d.t == t]
                if len(dd) == 0:
                    continue
                L.append(f"| {label} | {t} | {len(dd)} | {fmt(dd.rel_err.max())} | {fmt(dd.rel_err.median())} | {dd['cos'].median():.4f} | {fmt(dd.surv_PA_emp.median())} | {fmt(dd.surv_PA_pred.median())} | {int(dd.gate_mismatch.max())} |")
        d1 = bx[(bx["mode"] == "fullbatch_wonly") & (bx.t == 1)]
        lam = d1.groupby(["arm", "group"]).eta_lam_max.agg(["median", "max"]).reset_index()
        L.append("\n$\\eta\\lambda_{\\max}(H_A)$（安定な歩幅の条件は <2）: " + "; ".join(f"{r.arm}/{r.group} 中央値 {r['median']:.3g}・最大 {r['max']:.3g}" for _, r in lam.iterrows()) + "。")
        fa = bx[(bx["mode"] == "fullbatch_wonly") & (bx.gate_mismatch == 0)]
        fb = bx[(bx["mode"] == "sgd_wonly") & (bx.gate_mismatch == 0)]
        V["P0"] = dict(fullbatch_max_err_same_branch=float(fa.rel_err.max()), fullbatch_n=int(len(fa)),
                       sgd_median_err_t2000=float(fb[fb.t == 2000].rel_err.median()), sgd_cos_median_t2000=float(fb[fb.t == 2000]["cos"].median()),
                       natural_surv_emp_t2000=float(bx[(bx["mode"] == "natural") & (bx.t == 2000)].surv_PA_emp.median()),
                       natural_surv_pred_t2000=float(bx[(bx["mode"] == "natural") & (bx.t == 2000)].surv_PA_pred.median()),
                       eta_lam_max=lam.to_dict(orient="records"))
        L.append(f"\n**読み**: (a) 枝が変わらない限り full-batch では箱の式が機械精度で成立（最大誤差 {fmt(fa.rel_err.max())}・n={len(fa)}）。(b) batch-1 では標本ごとの作用素の積が平均作用素のべきの周りに散るが向きは一致（t=2000 で cos 中央値 {fb[fb.t == 2000]['cos'].median():.3f}・誤差中央値 {fmt(fb[fb.t == 2000].rel_err.median())}）。(c) 全パラメータが動くと式は破れる: 可視成分の生存率は予測 {fmt(V['P0']['natural_surv_pred_t2000'])} に対し実測 {fmt(V['P0']['natural_surv_emp_t2000'])}（t=2000・中央値）。箱の式は「他を固定した単一 unit」の式としては正しく、自然学習の unit の重みには当てはまらない。\n")

    # ---------------- P1: one-step exactness
    mx = {c: ok1[c].max() for c in ["rel_err_wu", "rel_err_W_other", "rel_err_b", "rel_err_c"]}
    p1 = all(v < 1e-10 for v in mx.values())
    v_err = ok1.rel_err_v.median()
    v_err_lin = lin1[lin1.ok].rel_err_v.median() if lin1 is not None else float("nan")
    V["P1"] = {"pass_": bool(p1), "max_rel_err": mx, "rel_err_v_median_eps1e-2": v_err,
               "rel_err_v_median_eps1e-3": v_err_lin, "n_units": int(len(ok1))}
    corr_b = float(np.corrcoef(np.log(ok1.rel_err_b.clip(1e-16)), np.log(ok1.dyhat1.abs().clip(1e-16)))[0, 1])
    V["P1"]["corr_log_relerr_b_vs_log_dyhat"] = corr_b
    V["P1"]["wu_block_pass"] = bool(mx["rel_err_wu"] < 1e-10 and mx["rel_err_W_other"] < 1e-10)
    L.append(f"## P1 一歩の厳密性: **{'PASS' if p1 else 'FAIL'}**（登録基準: 全ブロック相対誤差 <1e−10）— w_u ブロック最大相対誤差 {fmt(mx['rel_err_wu'])}、他 unit の W {fmt(mx['rel_err_W_other'])}、"
             f"b {fmt(mx['rel_err_b'])}、c {fmt(mx['rel_err_c'])}（ok unit {len(ok1)} 個）。v_u ブロックは O(ε) 項を含み中央値 {fmt(v_err)}（ε=1e−3 では {fmt(v_err_lin)}、ε に比例）。"
             f"\n\n**読み**: 主張の対象である w_u ブロックと他 unit の W は 1e−13 で一致（{'PASS' if V['P1']['wu_block_pass'] else 'FAIL'}）。b・c の 1e−10 は、差の予測値が小さい unit（δŷ が 1e−5 級）で b（0.2〜0.7）の丸め 1e−16 が相対で拡大したもの"
             f"（log 相対誤差と log|δŷ| の相関 {corr_b:.2f}）。登録した閾値を算術から導かなかった私の誤り（閾値は eps_mach·|b|/|Δb 予測| で置くべきだった）。式の予測は外れていない。")

    # ---------------- P2: N_A preserved
    p2v = per.q_NA_dev.max()
    V["P2"] = dict(pass_=bool(p2v < 1e-12), max_dev=p2v)
    L.append(f"## P2 N_A 成分の保存: **{'PASS' if p2v < 1e-12 else 'FAIL'}** — 全時刻・全 unit の最大偏差 {fmt(p2v)}。")

    # ---------------- P3: δŷ killed by t<=100
    piv = okp.pivot_table(index=key, columns="t", values="dyhat_rms")
    ratio = piv.div(piv[1], axis=0)
    frac100 = (ratio[100] < 0.1).mean()
    # first recorded t at which ratio < 0.1
    ts = sorted(c for c in ratio.columns)
    first = ratio.apply(lambda r: next((t for t in ts if r[t] < 0.1), np.nan), axis=1)
    V["P3"] = dict(pass_=bool(frac100 > 0.5), frac_units_below_0p1_at_100=frac100,
                   median_first_t=float(np.nanmedian(first)), ratio_median_by_t={int(t): float(ratio[t].median()) for t in ts})
    L.append(f"## P3 出力差の消失: **{'PASS' if frac100 > 0.5 else 'FAIL'}** — t=100 で δŷ の RMS が t=1 の 10% 未満になった unit の割合 {frac100:.2f}、"
             f"10% を切る最初の記録時刻の中央値 {fmt(np.nanmedian(first))}。中央値の推移: " + ", ".join(f"t{t}:{ratio[t].median():.3f}" for t in ts) + "。")

    # ---------------- P4: GN endpoint vs single-unit endpoint
    at = okp[okp.t == T_STAR].copy()
    at["r_mu"] = at.f_mu / at.f_gn_mu
    at["r_PA"] = at.f_PA / at.f_gn_PA
    within = ((at.r_PA > 0.5) & (at.r_PA < 2)).mean()
    within_mu = ((at.r_mu > 0.5) & (at.r_mu < 2)).mean()
    med_f = at.f_PA.median(); max_f = at.f_PA.max()
    V["P4"] = dict(pass_within2x_PA=bool(within > 0.5), frac_within_2x_PA=within, frac_within_2x_mu=within_mu,
                   f_PA_median=med_f, f_PA_max=max_f, f_mu_median=at.f_mu.median(),
                   frac_f_below_1e_2=float((at.f_PA < 1e-2).mean()),
                   single_unit_prediction_f=1.0, n=int(len(at)))
    L.append(f"## P4 全網の終点（t={T_STAR}）: 消去率が GN 予測の 1/2〜2 倍に入る unit の割合 **{within:.2f}**（‖P_A q‖ 版）／{within_mu:.2f}（μ̂ 射影版）→ "
             f"**{'PASS' if within > 0.5 else 'FAIL'}**。消去率の中央値 {fmt(med_f)}・最大 {fmt(max_f)}・1e−2 未満の割合 {(at.f_PA < 1e-2).mean():.2f}。"
             f"単一 unit 式の予測（消去率 1）は全 unit で外れる。")
    tab = at.groupby(["cond", "group"]).agg(n=("f_PA", "size"), f_emp_med=("f_PA", "median"), f_gn_med=("f_gn_PA", "median"),
                                              ratio_med=("r_PA", "median"), share_med=("ntk_share_mean", "median")).reset_index()
    L.append("\n| 条件 | 群 | n | 消去率 実測 中央値 | GN 予測 中央値 | 実測/予測 中央値 | NTK 分担（粗）中央値 |\n|---|---|---:|---:|---:|---:|---:|")
    for _, r in tab.iterrows():
        L.append(f"| {r.cond} | {r.group} | {r.n} | {fmt(r.f_emp_med)} | {fmt(r.f_gn_med)} | {fmt(r.ratio_med)} | {fmt(r.share_med)} |")
    # later times (report only)
    late = okp[okp.t.isin([200, 500, 1000, 2000, 5000, 10000])].groupby(["t"]).agg(f_PA_med=("f_PA", "median"), f_mu_med=("f_mu", "median"),
                                                                                      r_PA_med=("f_PA", lambda s: np.nan), dyhat=("dyhat_rms", "median"),
                                                                                      dW_other=("dW_other", "median"), dv=("dv", "median"),
                                                                                      gate_sup=("gate_mismatch_sup", "mean"), gate_stream=("gate_mismatch_stream", "median"))
    L.append("\n報告のみ（t>100 の推移・中央値）:\n\n| t | 消去率 ‖P_A‖ | 消去率 μ̂ | δŷ RMS | 他 unit の ΔW | Δv | 支持での枝不一致（平均個数） | 標本列での枝不一致累計 |\n|---:|---:|---:|---:|---:|---:|---:|---:|")
    for t, r in late.iterrows():
        L.append(f"| {t} | {fmt(r.f_PA_med)} | {fmt(r.f_mu_med)} | {fmt(r.dyhat)} | {fmt(r.dW_other)} | {fmt(r.dv)} | {fmt(r.gate_sup)} | {fmt(r.gate_stream)} |")
    V["late"] = {int(t): {k: float(v) for k, v in r.items()} for t, r in late.iterrows()}

    # ---------------- P5: share ratio pos/neg normalised by v^2
    p5 = {}
    L.append(f"\n## P5 分担の比（t={T_STAR}・消去率/v_u²・正側 vs 負側）")
    L.append("\n| 腕 | 1/a² | 実測 比（中央値の比） | GN 予測 比 | n_pos / n_neg | 判定 |\n|---|---:|---:|---:|---:|---|")
    all_pass = True
    for arm, a in A.items():
        s = at[at.arm == arm]
        pos, neg = s[s.group == "pos"], s[s.group == "neg"]
        if len(pos) == 0 or len(neg) == 0:
            p5[arm] = dict(pass_=None, n_pos=len(pos), n_neg=len(neg)); L.append(f"| {arm} | {1/a**2:.3g} | – | – | {len(pos)}/{len(neg)} | 判定不能 |"); continue
        r_emp = (pos.f_PA / pos.v_u ** 2).median() / (neg.f_PA / neg.v_u ** 2).median()
        r_gn = (pos.f_gn_PA / pos.v_u ** 2).median() / (neg.f_gn_PA / neg.v_u ** 2).median()
        ok = 0.5 / a ** 2 < r_emp < 2 / a ** 2
        all_pass &= ok
        p5[arm] = dict(pass_=bool(ok), ratio_emp=r_emp, ratio_gn=r_gn, target=1 / a ** 2, n_pos=int(len(pos)), n_neg=int(len(neg)))
        L.append(f"| {arm} | {1/a**2:.3g} | {r_emp:.3g} | {r_gn:.3g} | {len(pos)}/{len(neg)} | {'PASS' if ok else 'FAIL'} |")
    V["P5"] = dict(pass_=bool(all_pass), per_arm=p5)
    # supplementary (not pre-registered): pooled log-regression of erased fraction on log v^2 and log k^2
    s = at[(at.f_PA > 0)].copy()
    X = np.c_[np.ones(len(s)), np.log(s.v_u ** 2), np.log(s.k_u ** 2)]
    beta = np.linalg.lstsq(X, np.log(s.f_PA), rcond=None)[0]
    beta_gn = np.linalg.lstsq(X, np.log(s.f_gn_PA.clip(lower=1e-300)), rcond=None)[0]
    V["P5_supplement"] = dict(note="事後・未登録: log 消去率 = c + β1 log v² + β2 log k²（φ′² 則は β2=1、φ′ 則は β2=0.5）",
                              beta_emp=[float(b) for b in beta], beta_gn=[float(b) for b in beta_gn], n=int(len(s)))
    L.append(f"\n補足（事後・未登録）: 全 unit をプールした回帰 log f = c + β₁ log v² + β₂ log k²。実測 β₁={beta[1]:.2f}, β₂={beta[2]:.2f}（GN 予測では β₁={beta_gn[1]:.2f}, β₂={beta_gn[2]:.2f}）。φ′² 則は β₂=1、φ′ の一次則は β₂=0.5。")

    # ---------------- linearity (report)
    if lin is not None:
        l = lin[lin.ok].copy(); l["cond"] = l.arm + "_" + l.step.astype(str)
        l["f_PA"] = 1 - l.q_PA / l.q_PA0
        m = at.merge(l[l.t == T_STAR][key + ["f_PA"]], on=key, suffixes=("", "_lin"))
        rr = (m.f_PA / m.f_PA_lin)
        V["linearity"] = {"ratio_f_eps1e-2_over_eps1e-3_median": float(rr.median()), "q05": float(rr.quantile(0.05)), "q95": float(rr.quantile(0.95))}
        L.append(f"\n線形性（報告）: 同じ unit の消去率 ε=1e−2 / ε=1e−3 の比: 中央値 {rr.median():.3f}（5–95% {rr.quantile(0.05):.3f}–{rr.quantile(0.95):.3f}）。1 なら線形。")

    # ---------------- switch: P6-P9
    clos = sw.closure.max()
    p6v = sw.D_N.max()
    V["P6"] = dict(pass_=bool(p6v < 1e-12), max_D_N=p6v, closure_max=clos)
    L.append(f"\n## P6 切替後 10⁴ 更新で N_A D = 0: **{'PASS' if p6v < 1e-12 else 'FAIL'}** — 最大 ‖N_A D‖ {fmt(p6v)}。ノルム収支の閉包誤差最大 {fmt(clos)}。")
    pos = sw[sw.group == "pos"]
    sub = pos[pos.dzbar < 0]
    fr = (sub.dnorm2 > 0).mean()
    V["P7"] = dict(pass_=bool(fr > 0.5), frac_norm_grows_given_zbar_falls=fr, n=int(len(sub)), n_pos=int(len(pos)))
    L.append(f"## P7 正側で z̄ が下がった unit のうちノルムが伸びた割合: **{fr:.2f}** (n={len(sub)}/{len(pos)}) → **{'PASS' if fr > 0.5 else 'FAIL'}**。")
    g = sw.groupby(["arm", "step", "group"]).agg(n=("unit", "size"), dzbar=("dzbar", "mean"), dz_pos=("dz_pos_samples", "mean"), dz_neg=("dz_neg_samples", "mean"),
                                                dnorm2=("dnorm2", "mean"), part_mu=("part_mu", "mean"), part_perp=("part_perp", "mean"), part_N=("part_N", "mean"),
                                                abs_mu=("part_mu", lambda s: s.abs().mean()), abs_perp=("part_perp", lambda s: s.abs().mean()),
                                                D_mu=("D_mu", "mean"), D_perp=("D_perp", "mean"), dcos=("cos1", "mean"), cos0=("cos0", "mean"),
                                                frac_grow=("dnorm2", lambda s: (s > 0).mean()), frac_zdown=("dzbar", lambda s: (s < 0).mean())).reset_index()
    g["dcos"] = g.dcos - g.cos0
    L.append("\n| 腕 | 時点 | 群 | n | Δz̄ | 正側標本由来 | 負側標本由来 | Δ‖w‖² | μ̂ 成分 | S_A⊖μ̂ 成分 | N_A 成分 | \\|μ̂\\| 平均 | \\|⊥\\| 平均 | ‖D_μ‖ | ‖D_⊥‖ | ノルム増の割合 | z̄ 降の割合 | Δcos(w,μ̂) |")
    L.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in g.iterrows():
        L.append(f"| {r.arm} | {r.step} | {r.group} | {r.n} | {r.dzbar:+.4f} | {r.dz_pos:+.4f} | {r.dz_neg:+.4f} | {r.dnorm2:+.3f} | {r.part_mu:+.3f} | {r.part_perp:+.3f} | {r.part_N:+.1e} | {r.abs_mu:.3f} | {r.abs_perp:.3f} | {r.D_mu:.3f} | {r.D_perp:.3f} | {r.frac_grow:.2f} | {r.frac_zdown:.2f} | {r.dcos:+.4f} |")
    gp = g[g.group == "pos"]
    p8 = bool((gp.abs_perp > gp.abs_mu).all()); p9 = bool((gp.dcos < 0).all())
    V["P8"] = dict(pass_=p8, per_cond={f"{r.arm}_{r.step}": dict(abs_perp=r.abs_perp, abs_mu=r.abs_mu) for _, r in gp.iterrows()})
    V["P9"] = dict(pass_=p9, per_cond={f"{r.arm}_{r.step}": r.dcos for _, r in gp.iterrows()})
    L.append(f"\n## P8 正側群の Δ‖w‖² 内訳で S_A⊖μ̂ 成分 > μ̂ 成分（絶対値平均・純正側群のある {len(gp)} 条件）: **{'PASS' if p8 else 'FAIL'}**")
    L.append(f"## P9 正側群で cos(w, μ̂) が平均で減る（純正側群のある {len(gp)} 条件）: **{'PASS' if p9 else 'FAIL'}**")
    V["switch_table"] = g.to_dict(orient="records")

    # ---------------- additional diagnostics (post hoc, not pre-registered)
    L.append("\n## 追加診断（事後・未登録）")
    cnt = per[per.t == 1].groupby(["arm", "step", "group"]).ok.agg(["sum", "size"]).reset_index()
    L.append("\n### ok unit の数（|z|>0.03 で純正側/純負側だった unit・各 seed 5 個の枠）\n\n| 腕 | 時点 | 群 | ok | 枠 |\n|---|---:|---|---:|---:|")
    for _, r in cnt.iterrows():
        L.append(f"| {r.arm} | {r.step} | {r.group} | {int(r['sum'])} | {int(r['size'])} |")
    V["ok_counts"] = cnt.to_dict(orient="records")
    s = sw[sw.dzbar < 0].copy(); s["w0mu"] = np.where(s.w0_mu > 0, "w0·μ̂>0", "w0·μ̂<=0")
    ss = s.groupby(["group", "w0mu"]).agg(n=("unit", "size"), frac_grow=("dnorm2", lambda x: (x > 0).mean()), dnorm2=("dnorm2", "mean"),
                                          part_mu=("part_mu", "mean"), part_perp=("part_perp", "mean"), w0_mu=("w0_mu", "mean"), zbar0=("zbar0", "mean")).reset_index()
    L.append("\n### P7 の内訳: z̄ が下がった unit を w₀·μ̂ の符号で分ける（4 条件プール）\n\n算術: μ̂ 成分の寄与は 2(w₀·μ̂)(D·μ̂)+‖D_μ‖² なので、侵食（D·μ̂<0）でノルムが縮むのは w₀·μ̂>0 のときだけ。\n\n| 群 | w₀·μ̂ | n | ノルム増の割合 | Δ‖w‖² | μ̂ 成分 | ⊥ 成分 | w₀·μ̂ 平均 | z̄₀ 平均 |\n|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in ss.iterrows():
        L.append(f"| {r.group} | {r.w0mu} | {r.n} | {r.frac_grow:.2f} | {r.dnorm2:+.4f} | {r.part_mu:+.4f} | {r.part_perp:+.4f} | {r.w0_mu:+.3f} | {r.zbar0:+.3f} |")
    V["P7_split"] = ss.to_dict(orient="records")
    sw2 = sw.copy(); sw2["frest"] = -sw2.dzbar / sw2.zbar0
    ge = sw2.groupby(["arm", "step", "group"]).agg(n=("unit", "size"), zbar0=("zbar0", "mean"), dzbar=("dzbar", "mean"), v2=("v0", lambda x: (x ** 2).mean()),
                                                   frest_med=("frest", "median")).reset_index()
    L.append("\n### 群ごとの深さと 1 課題での復元率 −Δz̄/z̄₀（中央値）\n\n| 腕 | 時点 | 群 | n | z̄₀ | Δz̄ | v² 平均 | 復元率 中央値 |\n|---|---:|---|---:|---:|---:|---:|---:|")
    for _, r in ge.iterrows():
        L.append(f"| {r.arm} | {r.step} | {r.group} | {r.n} | {r.zbar0:+.3f} | {r.dzbar:+.4f} | {r.v2:.4f} | {r.frest_med:+.3f} |")
    V["restoration"] = ge.to_dict(orient="records")
    L.append("\n（正側/負側の復元率の比は v²・深さ・整列が群で違うので、φ′ の一次と二次を判別する検査にはならない。観察として置く。）")
    trj = okp.groupby(["cond", "group", "t"]).f_PA.median().unstack().reset_index()
    gnm = okp[okp.t == 1].groupby(["cond", "group"]).f_gn_PA.median().reset_index()
    trj = trj.merge(gnm, on=["cond", "group"])
    cols = [c for c in [1, 10, 100, 1000, 10000] if c in trj.columns]
    L.append("\n### 消去率 ‖P_A‖ 版の推移（条件・群ごとの中央値）と GN 終点\n\n| 条件 | 群 | " + " | ".join(f"t={c}" for c in cols) + " | GN 終点 |\n|---|---|" + "---:|" * (len(cols) + 1))
    for _, r in trj.iterrows():
        L.append(f"| {r.cond} | {r.group} | " + " | ".join(fmt(r[c]) for c in cols) + f" | {fmt(r.f_gn_PA)} |")
    V["erasure_trajectory"] = trj.to_dict(orient="records")
    L.append("\n読み: a=0.1（適合が良く残差が小さい）では消去率は GN 終点の水準に留まり 1 課題を通して安定。a=0.7・5M（late MSE ≈ 0.47 で残差が大きい）では GN 分担は 1e−6 なのに、残差に比例する二次項（2e∇²ŷ・読み出しの補償 δv が残差を通して w に戻る）が 1 課題で 3〜4 割を消す。単一 unit 式の「λ_x の率で全部消える」はどちらの領域でも成り立たない。")

    # ---------------- reviewer-requested controls (post hoc): null-direction, frozen-others, ± kicks
    L.append("\n## レビュー（GPT6Astra 第 1 回）で足した対照（事後・未登録）")
    fnull = OUT / "perturb_null_eps0.01.csv"
    if fnull.exists():
        nl = pd.read_csv(fnull); nlo = nl[nl.ok].copy(); nlo["f_PA"] = 1 - nlo.q_PA / nlo.q_PA0
        mdev = nlo.q_NA_dev.max(); mna0 = nlo.q_NA0.min()
        V["P2_null_control"] = dict(pass_=bool(mdev < 1e-12 and mna0 > 0), max_dev=float(mdev), min_q_NA0=float(mna0),
                                   f_PA_t100_median=float(nlo[nlo.t == 100].f_PA.median()), n=int(nlo.unit.nunique()))
        L.append(f"\n### N_A 成分を持つ摂動 q=ε(μ̂+n̂)/√2（P2 の空虚さを直す対照）\n\n‖N_A q‖ = {fmt(mna0)} > 0 の摂動で、全時刻の N_A 成分の偏差の最大 {fmt(mdev)} → **{'PASS' if mdev < 1e-12 else 'FAIL'}**（非空虚）。可視部分の消去率（t=100・中央値）{fmt(nlo[nlo.t == 100].f_PA.median())}（主走 {fmt(at.f_PA.median())}）。")
    ffrz = OUT / "perturb_frz_eps0.01.csv"
    if ffrz.exists():
        fz = pd.read_csv(ffrz); fzo = fz[fz.ok].copy(); fzo["f_PA"] = 1 - fzo.q_PA / fzo.q_PA0
        fzo["cond"] = fzo.arm + "_" + fzo.step.astype(str)
        tab = fzo[fzo.t.isin([10, 100, 1000, 2000])].pivot_table(index=["cond", "group"], columns="t", values="f_PA", aggfunc="median").reset_index()
        L.append("\n### 他パラメータを凍結した陽性対照（単一 unit 式の前提そのもの）— 可視成分の消去率の中央値\n\n| 条件 | 群 | t=10 | t=100 | t=1000 | t=2000 | 主走 t=2000（全網） |\n|---|---|---:|---:|---:|---:|---:|")
        main2000 = okp[okp.t == 2000].groupby(["cond", "group"]).f_PA.median()
        for _, r in tab.iterrows():
            L.append(f"| {r.cond} | {r.group} | {fmt(r[10])} | {fmt(r[100])} | {fmt(r[1000])} | {fmt(r[2000])} | {fmt(main2000.get((r.cond, r.group), float('nan')))} |")
        # rate test of the φ'^2 law under its own premise: -log(1-f)/t at t=100 vs v^2 k^2
        s100 = fzo[(fzo.t == 100) & (fzo.f_PA < 0.999) & (fzo.f_PA > 0)].copy()
        s100["rate"] = -np.log(1 - s100.f_PA) / 100.0
        X = np.c_[np.ones(len(s100)), np.log(s100.v_u ** 2), np.log(s100.k_u ** 2)]
        bf = np.linalg.lstsq(X, np.log(s100.rate), rcond=None)[0]
        V["freeze_control"] = dict(table=tab.to_dict(orient="records"), rate_regression_beta=[float(b) for b in bf], n=int(len(s100)),
                                   pos_f2000_median=float(fzo[(fzo.t == 2000) & (fzo.group == "pos")].f_PA.median()),
                                   neg_f2000_median=float(fzo[(fzo.t == 2000) & (fzo.group == "neg")].f_PA.median()))
        L.append(f"\n凍結下では正側の可視成分は課題の途中で消え（t=2000 中央値 {fmt(V['freeze_control']['pos_f2000_median'])}）、負側は {fmt(V['freeze_control']['neg_f2000_median'])}。"
                 f"t=100 の実効率 −log(1−f)/t を log v²・log k² に回帰: β₁={bf[1]:.2f}, β₂={bf[2]:.2f}（φ′² 則は β₂=1・β₁=1、n={len(s100)}）。"
                 f"同じ unit の全網の主走では t=2000 でも消去率は右端の列。単一 unit 式は前提を満たせば成り立ち、全網では分担に置き換わる。")
    fkick = OUT / "kick_delta0.2.csv"
    if fkick.exists():
        kk = pd.read_csv(fkick); ko = kk[kk.ok].copy(); ko["cond"] = ko.arm + "_" + ko.step.astype(str)
        ko["pm"] = ko.pair_mean / ko.delta; ko["asym"] = ko.s_plus - ko.s_minus
        kt = ko[ko.t.isin([10, 100, 1000, 2000])].groupby(["cond", "t"]).agg(n=("unit", "size"), s_plus=("s_plus", "median"), s_minus=("s_minus", "median"),
                                                                            pm=("pm", "median"), asym=("asym", "median"), frac_survive=("s_plus", lambda s: (s > 0.9).mean()),
                                                                            dnpos_P=("npos_P", "median"), dnpos_M=("npos_M", "median"), npos_A=("npos_A", "median")).reset_index()
        L.append("\n### 折れ目を跨ぐ unit への対称な ± 蹴り（b_u ± 0.2）— 蹴りの生存率 s_± と対の平均輸送（δ 単位）\n\n単一 unit の復元（自己項）なら上向きの蹴りは k=1 で速く戻り s₊≪s₋、対の平均は負（整流）。全網の補償なら s_±≈1 で対の平均 ≈ 0。\n\n| 条件 | t | n | s₊ | s₋ | 対の平均/δ | s₊−s₋ | s₊>0.9 の割合 | 正パターン数 A/P/M |\n|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for _, r in kt.iterrows():
            L.append(f"| {r.cond} | {r.t} | {r.n} | {r.s_plus:.3f} | {r.s_minus:.3f} | {r.pm:+.4f} | {r.asym:+.4f} | {r.frac_survive:.2f} | {r.npos_A:.0f}/{r.dnpos_P:.0f}/{r.dnpos_M:.0f} |")
        V["kick"] = kt.to_dict(orient="records")
        # dose: does the decay of the kick vanish when the kick no longer flips branches?
        dose = []
        for tag, d in (("_d0005", 0.005), ("_d005", 0.05), ("", 0.2)):
            fp = OUT / f"kick{tag}_delta{d:g}.csv"
            if fp.exists():
                kd = pd.read_csv(fp); kd = kd[kd.ok].copy(); kd["cond"] = kd.arm + "_" + kd.step.astype(str)
                kd["pm"] = kd.pair_mean / kd.delta; kd["flipP"] = kd.npos_P - kd.npos_A; kd["flipM"] = kd.npos_A - kd.npos_M
                g2 = kd[kd.t == 2000].groupby("cond").agg(s_plus=("s_plus", "median"), s_minus=("s_minus", "median"), pm=("pm", "median"),
                                                          flipP=("flipP", "median"), flipM=("flipM", "median"), n=("unit", "size")).reset_index()
                g2["delta"] = d; dose.append(g2)
        fgn = OUT / "gn_kick.csv"
        if fgn.exists():
            gk = pd.read_csv(fgn); gk = gk[gk.ok]
            k5 = pd.read_csv(OUT / "kick_d0005_delta0.005.csv"); k5 = k5[(k5.ok) & (k5.t == 2000)]
            k5 = k5.merge(gk[["arm", "step", "seed", "unit", "s_inf", "nonimitable_frac", "share_b", "share_wmu"]], on=["arm", "step", "seed", "unit"])
            k5["s_mean"] = (k5.s_plus + k5.s_minus) / 2
            gg = k5.groupby(["arm", "step"]).agg(n=("unit", "size"), s_inf=("s_inf", "median"), nonimit=("nonimitable_frac", "median"),
                                                 s_obs=("s_mean", "median"),
                                                 rho=("s_mean", lambda s: float(np.corrcoef(pd.Series(s).rank(), k5.loc[s.index, "s_inf"].rank())[0, 1]))).reset_index()
            L.append("\n### 蹴りの GN 終点 — 折れ目 unit の bias 摂動を線形 GN はどれだけ戻すと言うか（δ=0.005・t=2000 と比較）\n\n| 腕 | 時点 | n | GN 予測の生存率 s∞ | 実測 (s₊+s₋)/2 | 順位相関 |\n|---|---:|---:|---:|---:|---:|")
            for _, r in gg.iterrows():
                L.append(f"| {r.arm} | {r.step} | {r.n} | {r.s_inf:.3f} | {r.s_obs:.3f} | {r.rho:+.2f} |")
            V["kick_gn"] = gg.to_dict(orient="records")
            L.append("\n線形 GN は a=0.1 の折れ目 unit には大きな自己復元（消去 0.43／0.71。純正側 unit の 0.001〜0.05 と違い、k_{u,p} の段差を持つ Jacobian 行は他の unit と重ならないため）を予測するが、a=0.7 では v が小さく分担はほぼ 0（s∞ 0.98／1.00）。実測の復元は 4 条件とも GN より強い（順位相関 +0.18〜+0.59）。折れ目に座る unit は基準軌道そのものが枝を跨ぎ続けるので、固定曲率の線形化が当てはまらない。機構は未同定。")
        if dose:
            dd = pd.concat(dose).sort_values(["cond", "delta"])
            L.append("\n### 蹴りの用量（t=2000）— 枝を跨がない小さな蹴りで生存率は 1 に戻るか\n\n| 条件 | δ | n | s₊ | s₋ | 対の平均/δ | 跨いだパターン数 +/− |\n|---|---:|---:|---:|---:|---:|---|")
            for _, r in dd.iterrows():
                L.append(f"| {r.cond} | {r.delta:g} | {r.n} | {r.s_plus:.3f} | {r.s_minus:.3f} | {r.pm:+.4f} | {r.flipP:.0f}/{r.flipM:.0f} |")
            V["kick_dose"] = dd.to_dict(orient="records")

    # ---------------- figures
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
        for ax, (cond, d) in zip(axes, okp.groupby("cond")):
            for (grp, sd, u), dd in d.groupby(["group", "seed", "unit"]):
                dd = dd.sort_values("t")
                ax.plot(dd.t, 1 - dd.q_PA / dd.q_PA0, color="C3" if grp == "pos" else "C0", alpha=0.5, lw=0.8)
                ax.plot([dd.t.max()], [1 - dd.gn_q_PA.iloc[0] / dd.q_PA0.iloc[0]], marker="_", color="k", ms=8, alpha=0.6)
            ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1e-4); ax.set_title(cond); ax.set_xlabel("t"); ax.axvline(T_STAR, color="gray", ls=":")
        axes[0].set_ylabel("erased fraction of S_A component of q_i\n(red pos, blue neg, black tick = GN endpoint)")
        fig.tight_layout(); fig.savefig(OUT / "perturb_erasure.png", dpi=130); plt.close(fig)
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
        for ax, ((arm, st), d) in zip(axes, sw.groupby(["arm", "step"])):
            for grp, c in (("pos", "C3"), ("neg", "C0"), ("mix", "gray")):
                dd = d[d.group == grp]; ax.scatter(dd.dzbar, dd.dnorm2, s=6, color=c, alpha=0.5, label=f"{grp} n={len(dd)}")
            ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5); ax.set_title(f"{arm} step{st}"); ax.set_xlabel("Δz̄ (new task)"); ax.set_ylabel("Δ‖w‖²"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(OUT / "switch_geometry.png", dpi=130); plt.close(fig)
    except Exception as ex:  # figures are optional
        L.append(f"\n(図の生成失敗: {ex})")

    passes = {k: V[k]["pass_"] for k in ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"] if "pass_" in V[k]}
    V["P4"]["pass_"] = V["P4"]["pass_within2x_PA"]
    p0line = ""
    if "P0" in V:
        p0line = (f"P0a(full-batch・他固定)={'PASS' if V['P0']['fullbatch_max_err_same_branch'] < 1e-10 else 'FAIL'}, "
                  f"P0b(batch-1・他固定)={'PASS' if V['P0']['sgd_cos_median_t2000'] > 0.999 else 'FAIL'}, "
                  f"P0c(自然学習)={'FAIL（予測どおり破れる）' if V['P0']['natural_surv_emp_t2000'] > 0.9 else 'PASS'}, ")
    head = ["# phi2_projection_0918 — 結果（事後・理論検証）", "",
            "判定: " + p0line + ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in {**passes, 'P4': V['P4']['pass_']}.items()), "",
            f"設定: CondA checkpoint（zero_attraction_0913、leaky a=0.1/0.7、step 200000/5000000、10 seed）、float64、lr 0.005、W/b/v/c 全更新。摂動 q=εμ̂（ε=1e−2、線形性は 1e−3）、unit は各 seed で純正側/純負側（32 パターン全部で |z|>0.03）を折れ目からの余裕順に 5 個ずつ（足りない seed は ok=False で除外）。切替は η梯子と同じ（1 bit 反転・用量固定 off 再計算）。", ""]
    (OUT / "summary.md").write_text("\n".join(head + L) + "\n")
    json.dump(V, open(OUT / "verdict.json", "w"), indent=1, default=float)
    print("\n".join(head + L))


if __name__ == "__main__":
    main()

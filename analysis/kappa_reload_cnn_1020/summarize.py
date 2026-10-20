#!/usr/bin/env python3
"""kappa_reload_cnn_1020: Markdown tables T1-T7 (+ g2, diagnostics) and verdict.json from rows.json and raw/.

    python analysis/kappa_reload_cnn_1020/summarize.py [--only kappa]

Scoring follows spec addendum 1, A1.3 (fixed before any computation).  Seed aggregates are the mean (SD) over
the arm's seeds of per-seed values; 95% intervals use t with 9 degrees of freedom; R is one seed (no SD).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "kappa_reload_cnn_1020"
RAW = OUT / "raw"
ARMS = ("LR", "SNA", "SNAc3", "CV06FC3", "CV3FC06", "R")
SNAKE = ("SNA", "SNAc3", "CV06FC3", "CV3FC06")
ARM_JA = {"LR": "LR（leaky 0.1）", "R": "R（ReLU、1 seed）", "SNA": "SNA", "SNAc3": "SNAc3",
          "CV06FC3": "CV06FC3", "CV3FC06": "CV3FC06"}
LAYERS = ("c1", "c2", "f1", "f2")
OFF = {"c1": slice(0, 16), "c2": slice(16, 32), "f1": slice(32, 132), "f2": slice(132, 232)}
T975 = 2.2621571627409915                      # t_{0.975, 9}
INTERVAL_JA = {"init_to_task1": "初期値→課題 1", "replay_with_saved_end": "再生（終端を保存値と照合可）",
               "replay_without_saved_end": "同一エンジンによる再生、終端の直接照合なし",
               "continuation_beyond_original_run": "同一エンジンによる継続、元の走との照合なし"}


def fin(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def fmt(x, d=3):
    if not fin(x):
        return "—"
    ax = abs(x)
    if ax != 0 and (ax < 10 ** (-d) or ax >= 1e5):
        return f"{x:.2e}"
    return f"{x:.{d}f}"


def stats(vals):
    v = np.array([x for x in vals if fin(x)], dtype=np.float64)
    if len(v) == 0:
        return dict(n=0, mean=float("nan"), sd=float("nan"), lo=float("nan"), hi=float("nan"), med=float("nan"))
    sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
    hw = T975 * sd / math.sqrt(len(v)) if len(v) == 10 else float("nan")
    return dict(n=len(v), mean=float(v.mean()), sd=sd, lo=float(v.mean() - hw), hi=float(v.mean() + hw),
                med=float(np.median(v)), min=float(v.min()), max=float(v.max()))


def msd(vals, d=2):
    s = stats(vals)
    if s["n"] == 0:
        return "—"
    if s["n"] == 1 or not fin(s["sd"]):
        return fmt(s["mean"], d)
    return f"{fmt(s['mean'], d)}（{fmt(s['sd'], d)}）"


def mci(vals, d=2):
    s = stats(vals)
    if s["n"] == 0:
        return "—"
    if not fin(s["lo"]):
        return fmt(s["mean"], d)
    return f"{fmt(s['mean'], d)} [{fmt(s['lo'], d)}, {fmt(s['hi'], d)}]"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def load():
    rows = json.loads((OUT / "rows.json").read_text())
    by = {}
    for r in rows:
        by.setdefault((r["arm"], r["t"]), []).append(r)
    for k in by:
        by[k].sort(key=lambda r: r["seed"])
    fl = json.loads((OUT / "floors.json").read_text())
    return rows, by, fl


def arms_present(by):
    return [a for a in ARMS if any(k[0] == a for k in by)]


def tasks_of(by, arm, with0=False):
    return sorted(t for a, t in by if a == arm and (with0 or t > 0))


def col(by, arm, t, key):
    return [r.get(key) for r in by.get((arm, t), [])]


def q_ratio(by, arm, t):
    """post hoc (not registered): seed-wise Q_t / Q_1, Q = kappa ||F|| (projection norm)."""
    q1 = {r["seed"]: r["Q_proj"] for r in by.get((arm, 1), [])}
    return [r["Q_proj"] / q1[r["seed"]] if fin(q1.get(r["seed"])) and q1[r["seed"]] > 0 and fin(r["Q_proj"]) else float("nan")
            for r in by.get((arm, t), [])]


def kappa_ratio(by, arm, t, fl):
    """seed-wise kappa_t / kappa_1 (undefined if kappa_1 is not above the floor)."""
    k1 = {r["seed"]: r["kappa"] for r in by.get((arm, 1), [])}
    floor = fl["atol"].get("kappa", 0.0)
    out = []
    for r in by.get((arm, t), []):
        a = k1.get(r["seed"])
        out.append(r["kappa"] / a if (fin(a) and a > floor and fin(r["kappa"])) else float("nan"))
    return out


# --------------------------------------------------------------------------
# T1 / T2
# --------------------------------------------------------------------------

def t1_t2(by, fl):
    md = ["# kappa_reload_cnn_1020 — 表 T1（κ）と T2（補題）", "",
          "値は腕 × 保存境界ごとに seed の平均（SD）。κ は v^c の F 方向への原点を通る最小二乗射影係数（A1.1(8)）。"
          "Q = κ‖F‖ は射影ノルム、‖κF‖/√H は現課題の F 方向の読み出し射影の大きさ、射影割合は中心化 head の二乗ノルムに"
          "対する射影の割合（予測の説明率ではない）。κ の帰無分位点は head の中心化行をクラスで置換した 1,000 標本"
          "（特徴と F は固定）の中での実測 κ の位置。初期値（t = 0）の κ は課題 1 のラベルで作った偶然の値（MLP では NaN"
          "だった追加測定）。", ""]
    head = ["腕", "t", "κ 平均（SD）", "κ_k/κ₁ 中央値", "κ_k/κ₁ ≥ 0.8 の seed", "κ の帰無分位点（中央値）",
            "κ / 帰無 SD（中央値）", "Q = κ‖F‖", "‖κF‖/√H", "射影割合", "‖F‖", "‖v^c‖", "A/N",
            "旧ラベルの正答率", "正解クラスの予測確率"]
    rows = []
    for arm in arms_present(by):
        for t in tasks_of(by, arm, with0=True):
            rs = by[(arm, t)]
            kr = kappa_ratio(by, arm, t, fl) if t > 0 else []
            nge = sum(1 for v in kr if fin(v) and v >= 0.8)
            ndef = sum(1 for v in kr if fin(v))
            zs = [r["kappa"] / r["kappa_null_sd"] if fin(r.get("kappa_null_sd")) and r["kappa_null_sd"] > 0 else float("nan")
                  for r in rs]
            rows.append([ARM_JA[arm], "0（初期値）" if t == 0 else t, msd(col(by, arm, t, "kappa"), 5),
                         fmt(stats(kr)["med"], 3) if t > 1 else ("1" if t == 1 else "—"),
                         f"{nge}/{ndef}" if t > 1 else "—",
                         fmt(stats(col(by, arm, t, "kappa_null_quantile"))["med"], 3), fmt(stats(zs)["med"], 1),
                         msd(col(by, arm, t, "Q_proj"), 2), msd(col(by, arm, t, "learned_norm"), 2),
                         msd(col(by, arm, t, "expl_total"), 3), msd(col(by, arm, t, "F_norm"), 0),
                         msd(col(by, arm, t, "normV"), 2), msd([r["A"] / 1200 for r in rs], 2),
                         msd(col(by, arm, t, "acc_old"), 3), msd(col(by, arm, t, "conf_old"), 3)])
    md += ["## T1a κ・射影ノルム・読み出しの大きさ", "", table(head, rows), ""]
    # post hoc columns (parent's request after the interim report; not registered)
    head = ["腕", "t", "Q_k/Q₁ 中央値", "Q_k/Q₁ ≥ 0.8 の seed", "射影割合 cos²（seed 平均）", "κ_k/κ₁ ≥ 0.8 の seed（登録の P1）"]
    rows = []
    for arm in arms_present(by):
        for t in tasks_of(by, arm):
            if t < 2:
                continue
            qr = q_ratio(by, arm, t); kr = kappa_ratio(by, arm, t, fl)
            rows.append([ARM_JA[arm], t, fmt(stats(qr)["med"], 3), f"{sum(1 for v in qr if fin(v) and v >= 0.8)}/{sum(1 for v in qr if fin(v))}",
                         msd(col(by, arm, t, "expl_total"), 3), f"{sum(1 for v in kr if fin(v) and v >= 0.8)}/{sum(1 for v in kr if fin(v))}"])
    md += ["### T1a 補（事後・未登録）射影ノルム Q = κ‖F‖ の保持と射影割合", "",
           "κ は F を a 倍・head を 1/a 倍する再パラメータ化で 1/a² 倍になり、関数の不変量ではない（A1.1(9)）。"
           "登録の P1 は κ 比で採点する。この表は親の依頼で計算後に足した補助列で、予測の採点には使わない。", "",
           table(head, rows), ""]
    for span_one in (True, False):
        title = ("## T1b 課題ごとの再装填の内訳（span = 1 の射影帳簿）" if span_one else
                 "## T1c 保存境界間の射影帳簿（span > 1、複数課題をまたぐ）")
        md += [title, "", "割合は κ に対する各項（seed 平均（SD））。偶然整列項 = 参照状態と現課題ラベルの偶然整列、"
               "特徴差項 = 参照 head による特徴差、head 差項 = 現特徴による head 差 [恒等式で和 = κ]。負や 100% 超もありうる。", ""]
        rows = []
        for arm in arms_present(by):
            for t in tasks_of(by, arm):
                rs = by[(arm, t)]
                if (rs[0]["span"] == 1) != span_one:
                    continue
                ok = [r for r in rs if fin(r["kappa"]) and abs(r["kappa"]) > fl["atol"].get("kappa", 0.0)]
                rows.append([ARM_JA[arm], t, rs[0]["ref_task"], rs[0]["span"],
                             msd([r["kap_chance"] / r["kappa"] for r in ok], 3),
                             msd([r["kap_feat"] / r["kappa"] for r in ok], 3),
                             msd([r["kap_read"] / r["kappa"] for r in ok], 3),
                             fmt(max([r["decomp_absdiff"] / max(abs(r["kappa"]), 1e-300) for r in ok] or [float("nan")]), 1)])
        md += [table(["腕", "t", "参照 ref_task", "span", "偶然整列項 / κ", "特徴差項 / κ", "head 差項 / κ",
                      "和と κ の最大相対差"], rows), ""]
    # T2
    md += ["## T2 補題（A = Σ[(ℓ_y − ℓ̄) − b^c_y]、A = D + J、κ ≥ κ_lb = D/‖F‖²）", "",
           "D = N(log K − L) − B、J = Σ_n[logsumexp − mean − log K] ≥ 0、κ/κ_lb = 1 + J/D（D > 床の状態だけ）。"
           f"破れは J < −atol（atol_J = {fmt(fl['atol'].get('J', 0.0), 2)}）で数える。初期値の行は課題 1 のラベル。", ""]
    head = ["腕", "状態", "状態数", "A の 2 式の最大相対誤差", "A = D + J の最大相対誤差", "破れ（J < −atol）",
            "D ≤ 床", "κ 未定義（‖F‖² ≤ 床）", "κ/κ_lb の四分位", "κ/κ_lb の最小"]
    rows = []
    for arm in arms_present(by):
        for lab, sel in (("t ≥ 1", lambda t: t > 0), ("初期値", lambda t: t == 0)):
            rs = [r for t in tasks_of(by, arm, with0=True) if sel(t) for r in by[(arm, t)]]
            if not rs:
                continue
            e1 = max(r["A_id_absdiff"] / max(abs(r["A"]), 1e-300) for r in rs)
            e2 = max(r["A_DJ_absdiff"] / max(abs(r["A"]), abs(r["D"]) + abs(r["J"]), 1e-300) for r in rs)
            ratios = [r["kappa_over_lb"] for r in rs if fin(r.get("kappa_over_lb"))]
            q = np.percentile(ratios, [25, 50, 75]) if ratios else [float("nan")] * 3
            rows.append([ARM_JA[arm], lab, len(rs), fmt(e1, 1), fmt(e2, 1), sum(bool(r["lemma_violation"]) for r in rs),
                         sum(not r["D_positive"] for r in rs), sum(not r["kappa_defined"] for r in rs),
                         " / ".join(fmt(v, 2) for v in q), fmt(min(ratios) if ratios else float("nan"), 2)])
    md += [table(head, rows), ""]
    head = ["腕"] + [f"t{t}" for t in (1, 2, 3, 5, 10, 20, 30, 40, 50)]
    rows = []
    for arm in arms_present(by):
        rows.append([ARM_JA[arm]] + [fmt(stats(col(by, arm, t, "kappa_over_lb"))["med"], 2) if (arm, t) in by else "·"
                                     for t in (1, 2, 3, 5, 10, 20, 30, 40, 50)])
    md += ["κ/κ_lb の seed 間中央値（腕 × 境界）:", "", table(head, rows), ""]
    dneg = []
    for fn in sorted(OUT.glob("ref_*.json")):
        dneg += json.loads(fn.read_text()).get("D_nonpositive", [])
    rs_neg = [r for k, v in by.items() for r in v if k[1] > 0 and not r["D_positive"]]
    if rs_neg:
        dd = {(d["arm"], d["seed"], d["t"]): d for d in dneg}
        rows = []
        for r in sorted(rs_neg, key=lambda r: (r["arm"], r["t"], r["seed"])):
            d = dd.get((r["arm"], r["seed"], r["t"]), {})
            rows.append([ARM_JA[r["arm"]], r["t"], r["seed"], fmt(r["L_old"], 3), fmt(r["B"], 2), fmt(r["D"], 1), fmt(r["J"], 1),
                         r.get("n_wrong_old", "—"), fmt(d.get("ce_sum_wrong", float("nan")) / 1200, 3),
                         fmt(d.get("ce_sum_right", float("nan")) / 1200, 3), fmt(d.get("ce_max", float("nan")), 1),
                         fmt(d.get("logit_absmax", float("nan")), 1)])
        md += ["D ≤ 床の状態（t ≥ 1）: 旧ラベルの平均 CE が log K − B/N を超え、下界が正にならない。L の内訳（誤答と正答の CE の和 / N）:", "",
               table(["腕", "t", "seed", "L", "B", "D", "J", "誤答数", "誤答の CE の和 / N", "正答の CE の和 / N", "CE の最大", "|logit| の最大"], rows), ""]
    return "\n".join(md)


# --------------------------------------------------------------------------
# T3-T7
# --------------------------------------------------------------------------

def t3(by, fl):
    md = ["## T3 層ごとの期待の押し（固定状態、新ラベルの一様期待）", "",
          "G > 0 が沈める側。主表は生きた unit（z > 0 の入力が 1 つ以上）で、|G| が層ごとの数値床"
          f"（{', '.join(f'{k} {fmt(v, 1)}' for k, v in fl['G'].items())}）以下は不定として除く。seed 平均 [95% 区間]。"
          "G_up はブロック内積の和から独立に出した上流ブロックの応答（適応 α の更新は含まない）。分担 = |G_up|/(|G_own|+|G_up|)、"
          "相殺 = |G_full|/(|G_own|+|G_up|) の中央値。", ""]
    head = ["腕", "t", "生きた unit c1/c2/f1/f2", "c1 G_full>0", "c2 G_full>0", "f1 G_full>0", "f2 G_full>0",
            "c2 G_own>0", "f1 G_own>0", "f2 G_own>0", "分担 c2/f1/f2", "相殺 c2/f1/f2", "不定 c1/c2/f1/f2"]
    rows = []
    for arm in arms_present(by):
        for t in tasks_of(by, arm, with0=True):
            rows.append([ARM_JA[arm], t,
                         "/".join(fmt(stats(col(by, arm, t, f"{L}_n_alive"))["mean"], 1) for L in LAYERS)]
                        + [mci(col(by, arm, t, f"{L}_frac_Gfull_pos"), 2) for L in LAYERS]
                        + [mci(col(by, arm, t, f"{L}_frac_Gown_pos"), 2) for L in ("c2", "f1", "f2")]
                        + ["/".join(fmt(stats(col(by, arm, t, f"{L}_up_share_med"))["mean"], 2) for L in ("c2", "f1", "f2")),
                           "/".join(fmt(stats(col(by, arm, t, f"{L}_cancel_med"))["mean"], 2) for L in ("c2", "f1", "f2")),
                           "/".join(fmt(stats(col(by, arm, t, f"{L}_n_indet_full"))["mean"], 1) for L in LAYERS)])
    md += [table(head, rows), ""]
    md += ["補表（全 unit。生きていない unit も含む）: G_full > 0 の割合の seed 平均", ""]
    head = ["腕", "t"] + [f"{L}（全 {16 if L[0] == 'c' else 100}）" for L in LAYERS]
    rows = [[ARM_JA[arm], t] + [msd(col(by, arm, t, f"{L}_all_frac_Gfull_pos"), 2) for L in LAYERS]
            for arm in arms_present(by) for t in tasks_of(by, arm, with0=True)]
    md += [table(head, rows), ""]
    md += ["### T3b f2 の自己形と射影帳簿", "",
           "S_exact = N·G_own(f2)。S_self = κ(1 − 1/K)Σ K_n g φ、S_oh = Σ K_n g v^c_{y_n}、S_proj = Σ K_n g (κF)_{y_n}。"
           "一致は生きた unit で両方が床を超えるもの。偶然一致 = p_self p_own + (1 − p_self)(1 − p_own)。"
           "帳簿の各段は |段| / |S_exact| の中央値。", ""]
    head = ["腕", "t", "S_self > 0", "S_self と S_exact の一致（偶然）", "S_self/S_exact 四分位（中央の seed）",
            "S_oh 一致", "S_proj 一致", "Spearman(S_self, S_exact)", "Spearman(S_oh, S_exact)",
            "|S_exact−S_oh|/|S_exact|", "|S_oh−S_proj|/|S_exact|", "|S_proj−S_self|/|S_exact|", "S_self<0 の unit（G_own<0 の割合）",
            "K_n 最小（負の割合）"]
    rows = []
    for arm in arms_present(by):
        for t in tasks_of(by, arm):
            rs = by[(arm, t)]
            qmid = [r["f2_ratio_self_exact_q"][1] for r in rs]
            nneg = sum(r["f2_n_Sself_neg"] for r in rs)
            rows.append([ARM_JA[arm], t, msd(col(by, arm, t, "f2_frac_Sself_pos"), 2),
                         f"{msd(col(by, arm, t, 'f2_agree_self_exact'), 2)}（{fmt(stats(col(by, arm, t, 'f2_agree_self_exact_chance'))['mean'], 2)}）",
                         fmt(stats(qmid)["med"], 3), msd(col(by, arm, t, "f2_agree_oh_exact"), 2),
                         msd(col(by, arm, t, "f2_agree_proj_exact"), 2), msd(col(by, arm, t, "f2_sp_self_exact"), 2),
                         msd(col(by, arm, t, "f2_sp_oh_exact"), 2),
                         fmt(stats(col(by, arm, t, "f2_ledger_exact_minus_oh_medabs_rel"))["med"], 3),
                         fmt(stats(col(by, arm, t, "f2_ledger_oh_minus_proj_medabs_rel"))["med"], 3),
                         fmt(stats(col(by, arm, t, "f2_ledger_proj_minus_self_medabs_rel"))["med"], 3),
                         f"{nneg}（{fmt(stats(col(by, arm, t, 'f2_frac_Gown_neg_given_Sself_neg'))['mean'], 2)}、"
                         f"{sum(1 for r in rs if r['f2_n_Sself_neg'] > 0)} seed）",
                         f"{fmt(min(col(by, arm, t, 'K_min')), 2)}（{fmt(stats(col(by, arm, t, 'K_negfrac'))['mean'], 3)}）"])
    md += [table(head, rows), ""]
    sn = [a for a in arms_present(by) if a in SNAKE]
    if sn:
        md += ["### T3d Snake の谷の則（seat = 2αm、G_full > 0 の AUC。正負両方ある状態だけ）", ""]
        head = ["腕", "t", "c2 AUC", "c2 谷の上で沈める側", "c2 谷の下で沈める側", "c2 谷の下の数", "f2 AUC"]
        rows = [[a, t, msd(col(by, a, t, "c2_seat_auc"), 2), msd(col(by, a, t, "c2_frac_sink_above_valley"), 2),
                 msd(col(by, a, t, "c2_frac_sink_below_valley"), 2), fmt(stats(col(by, a, t, "c2_n_below_valley"))["mean"], 1),
                 msd(col(by, a, t, "f2_seat_auc"), 2)] for a in sn for t in tasks_of(by, a)]
        md += [table(head, rows), ""]
    return "\n".join(md)


def t3c():
    refs = []
    for fn in sorted(OUT.glob("ref_*.json")):
        refs += json.loads(fn.read_text())["rows"]
    if not refs:
        return ""
    md = ["### T3c 旧ラベルへの整列を壊した head での f2 の押し（事後・未登録、親の依頼）", "",
          "特徴（a3・z4・a4、K_n・g）は課題末のまま固定し、head だけを替えて π を計算し直した。置換 = 中心化 head の行をクラスで"
          "置換（200 通り）。乱数（学習済みノルム）= 中心化した正規乱数を ‖v^c‖ に合わせ bias はそのまま。乱数（初期化尺度）= "
          "‖v^c‖ をその seed の t0 の値に、bias を t0 の値に合わせた正規乱数。値は 200 通りの中央値（括弧は四分位の幅）。"
          "自己形の一致 = 替えた head で κ を取り直した S_self と S_exact の符号一致。", ""]
    head = ["腕", "t", "seed", "head", "f2 沈める側", "med|S_exact| 替えた/実際", "自己形と S_exact の一致", "κ（替えた head）",
            "平均の自信 max π", "κ > 0 の割合"]
    rows = []
    for r in refs:
        rows.append([ARM_JA[r["arm"]], r["t"], r["seed"], "実際", fmt(r["real"]["frac_sink"], 2), "1", fmt(r["real"]["agree_self"], 2),
                     fmt(r["real"]["kappa"], 5), fmt(r["real"]["conf"], 3), "—"])
        for k, lab in (("perm", "置換"), ("rand_trained_norm", "乱数（学習済みノルム）"), ("rand_init_scale", "乱数（初期化尺度）")):
            if k not in r:
                continue
            v = r[k]
            q = lambda kk, d=2: f"{fmt(v[kk][1], d)}（{fmt(v[kk][0], d)}〜{fmt(v[kk][2], d)}）"
            rows.append(["", "", "", lab, q("frac_sink"), q("medabs_ratio_q", 3), q("agree_self"), q("kappa", 5), q("conf", 3),
                         fmt(v["frac_draws_kappa_pos"], 2)])
    md += [table(head, rows), ""]
    return "\n".join(md)


def t4(by, fl):
    md = ["## T4 実際の切替（再生、最初の 200 更新）", "",
          "step 0 で生きていた unit。沈める側 = Δm < 0（|Δm| ≤ 再評価差の床は不定）。一致は −G_full（f2 は −G_own も）と Δm の符号。"
          "沈み = −Δm_200 の中央値、沈み/幅 = −Δm_200/s(0) の中央値。seed 平均（SD）。", ""]
    head = ["腕", "t", "区間", "c1 1 歩/200", "c2 1 歩/200", "f1 1 歩/200", "f2 1 歩/200",
            "f2 −G_own と Δm_1 / Δm_200 の一致", "f2 −G_full と Δm_1 / Δm_200", "c2 −G_full と Δm_200", "f2 沈み（沈み/幅）",
            "f2 p⁺: 1/2 を上へ / 下へ（200 更新、seed 合計）"]
    rows = []
    for arm in arms_present(by):
        for t in tasks_of(by, arm, with0=True):
            rs = by[(arm, t)]
            if not any("rp_f2_frac_sink200" in r for r in rs):
                continue
            rows.append([ARM_JA[arm], t, INTERVAL_JA.get(rs[0].get("rp_interval"), "—")]
                        + [f"{msd(col(by, arm, t, f'rp_{L}_frac_sink1'), 2)} / {msd(col(by, arm, t, f'rp_{L}_frac_sink200'), 2)}"
                           for L in LAYERS]
                        + [f"{msd(col(by, arm, t, 'rp_f2_agree_Gown_1'), 2)} / {msd(col(by, arm, t, 'rp_f2_agree_Gown_200'), 2)}",
                           f"{msd(col(by, arm, t, 'rp_f2_agree_Gfull_1'), 2)} / {msd(col(by, arm, t, 'rp_f2_agree_Gfull_200'), 2)}",
                           msd(col(by, arm, t, "rp_c2_agree_Gfull_200"), 2),
                           f"{fmt(stats(col(by, arm, t, 'rp_f2_sink200_med'))['mean'], 3)}（{fmt(stats(col(by, arm, t, 'rp_f2_sink200_rel_med'))['mean'], 3)}）",
                           f"{sum(r.get('rp_f2_p_up_200', 0) for r in rs)} / {sum(r.get('rp_f2_p_down_200', 0) for r in rs)}"])
    md += [table(head, rows), ""]
    # g2 (registered for LR only; Snake t1 comes from the K5 check run and is shown apart)
    rows, rows_sn = [], []
    for arm in arms_present(by):
        for t in tasks_of(by, arm):
            rs = by[(arm, t)]
            if not any("rp_f2_return_agg" in r for r in rs):
                continue
            (rows if arm == "LR" else rows_sn).append([ARM_JA[arm], t, INTERVAL_JA.get(rs[0].get("rp_interval"), "—")]
                        + [f"{msd(col(by, arm, t, f'rp_{L}_return_agg'), 2)}（{fmt(stats(col(by, arm, t, f'rp_{L}_n_sank200'))['mean'], 1)}）"
                           for L in LAYERS]
                        + [msd(col(by, arm, t, "rp_f2_frac_rest_pos"), 2), msd(col(by, arm, t, "rp_f2_frac_return_ge1"), 2),
                           msd(col(by, arm, t, "rp_f2_frac_sink_end"), 2), msd(col(by, arm, t, "rp_f2_agree_Gfull_end"), 2)])
    if rows:
        md += ["### g2 課題の中の戻り（LR、課題の全 30,000 更新を再生）", "",
               "Δm_rest = m(30000) − m(200)。対象は step 0 で生きていて step 200 までに沈んだ unit。戻りの比 = ΣΔm_rest / Σ|Δm_200|"
               "（seed ごとに出してから平均）。括弧は対象 unit 数の seed 平均。", "",
               table(["腕", "t", "区間", "c1 戻りの比", "c2", "f1", "f2", "f2 残りで上がる割合", "f2 全部戻る割合",
                      "f2 課題全体で下がる割合", "f2 −G_full と課題全体の変位の一致"], rows), ""]
    if rows_sn:
        md += ["補表: Snake 4 腕と R の t1 → t2（K5 の忠実度検査で回した全課題の再生の副産物。g2 の登録対象外）", "",
               table(["腕", "t", "区間", "c1 戻りの比", "c2", "f1", "f2", "f2 残りで上がる割合", "f2 全部戻る割合",
                      "f2 課題全体で下がる割合", "f2 −G_full と課題全体の変位の一致"], rows_sn), ""]
    return "\n".join(md)


def keep_ratio(by, arm, t, key):
    base = {r["seed"]: r.get(key) for r in by.get((arm, 1), [])}
    return [r.get(key) / base[r["seed"]] if fin(r.get(key)) and fin(base.get(r["seed"])) and base[r["seed"]] > 0
            else float("nan") for r in by.get((arm, t), [])]


def t5(by, fl):
    md = ["## T5 Keep（押しの大きさの比 t / t1、seed ごとの比の seed 間中央値）", "",
          "med|G| は生きた unit の中央値。f2 は G_own（= S_exact/N）も。κ_k/κ₁ を並べる。", ""]
    head = ["腕", "t", "c1 med|G_full|", "c2", "f1", "f2", "f2 med|G_own|", "κ_k/κ₁", "f2 の生きた unit"]
    rows = []
    for arm in arms_present(by):
        for t in tasks_of(by, arm):
            rows.append([ARM_JA[arm], t] + [fmt(stats(keep_ratio(by, arm, t, f"{L}_medabs_Gfull"))["med"], 2) for L in LAYERS]
                        + [fmt(stats(keep_ratio(by, arm, t, "f2_medabs_Gown"))["med"], 2),
                           fmt(stats(kappa_ratio(by, arm, t, fl))["med"], 2),
                           fmt(stats(col(by, arm, t, "f2_n_alive"))["mean"], 1)])
    md += [table(head, rows), ""]
    return "\n".join(md)


def common_alive_ratio(arm, seed):
    fn = RAW / f"units_{arm}_s{seed}.npz"
    if not fn.exists():
        return float("nan")
    U = np.load(fn)
    ts = list(U["t"])
    if 0 not in ts or 1 not in ts:
        return float("nan")
    a = (U["alive"][ts.index(0)][OFF["f2"]] > 0) & (U["alive"][ts.index(1)][OFF["f2"]] > 0)
    if not a.any():
        return float("nan")
    s0 = np.median(np.abs(U["S_exact"][ts.index(0)][a])); s1 = np.median(np.abs(U["S_exact"][ts.index(1)][a]))
    return float(s0 / s1) if s1 > 0 else float("nan")


def t6(by, fl):
    md = ["## T6 初期化（t = 0、乱数 head）の押しと課題 1 末の押し", "",
          "比は seed ごとに出して seed 間中央値。主列は状態ごとの生きた集合（先行比較用）、補列は全 100 unit と両状態に共通の生きた集合。", ""]
    head = ["腕", "f2 med|S_exact| t0/t1", "全 100 unit", "共通の生きた集合", "c1 med|G_full| t0/t1", "c2", "f1", "f2",
            "t0 の期待の押し G_full>0 c1/c2/f1/f2", "t0 の 200 更新で沈んだ c1/c2/f1/f2", "t0 の κ（偶然、課題 1 のラベル）"]
    rows = []
    for arm in arms_present(by):
        if (arm, 0) not in by or (arm, 1) not in by:
            continue
        r0 = {r["seed"]: r for r in by[(arm, 0)]}; r1 = {r["seed"]: r for r in by[(arm, 1)]}
        seeds = sorted(set(r0) & set(r1))

        def ratio(key):
            return [r0[s][key] / r1[s][key] if fin(r0[s].get(key)) and fin(r1[s].get(key)) and r1[s][key] > 0
                    else float("nan") for s in seeds]
        rows.append([ARM_JA[arm], fmt(stats(ratio("f2_medabs_Sexact_N"))["med"], 4),
                     fmt(stats(ratio("f2_medabs_Sexact_N_all"))["med"], 4),
                     fmt(stats([common_alive_ratio(arm, s) for s in seeds])["med"], 4)]
                    + [fmt(stats(ratio(f"{L}_medabs_Gfull"))["med"], 4) for L in LAYERS]
                    + ["/".join(fmt(stats(col(by, arm, 0, f"{L}_frac_Gfull_pos"))["mean"], 2) for L in LAYERS),
                       "/".join(fmt(stats(col(by, arm, 0, f"rp_{L}_frac_sink200"))["mean"], 2) for L in LAYERS),
                       msd(col(by, arm, 0, "kappa"), 6)])
    md += [table(head, rows), ""]
    return "\n".join(md)


def returns(by):
    """g1: per arm x layer over the saved boundaries t >= 1: unique seed-units that went below and came back."""
    res = {}
    who = []
    for arm in arms_present(by):
        seeds = sorted({r["seed"] for r in by.get((arm, 1), [])})
        for L in LAYERS:
            for crit in ("p", "m"):
                tot = ent = risk = ret = 0
                for s in seeds:
                    fn = RAW / f"units_{arm}_s{s}.npz"
                    if not fn.exists():
                        continue
                    U = np.load(fn)
                    ts = list(U["t"])
                    idx = [i for i, t in enumerate(ts) if t > 0]
                    X = (U["pplus"] if crit == "p" else U["m"])[idx][:, OFF[L]]
                    below = X < 0.5 if crit == "p" else X < 0
                    for j in range(below.shape[1]):
                        tot += 1
                        b = np.flatnonzero(below[:, j])
                        if len(b) == 0:
                            continue
                        ent += 1
                        if b[0] < below.shape[0] - 1:
                            risk += 1
                        ups = [k for k in range(b[0] + 1, below.shape[0]) if not below[k, j]]
                        if ups:
                            ret += 1
                            if crit == "p":
                                who.append(dict(arm=arm, seed=s, layer=L, unit=j, first_below_t=int(ts[idx[b[0]]]),
                                                back_t=[int(ts[idx[k]]) for k in ups],
                                                seq=[round(float(v), 3) for v in X[:, j]]))
                res[(arm, L, crit)] = dict(total=tot, entered=ent, risk=risk, returned=ret)
    return res, who


def t7(by):
    res, who = returns(by)
    md = ["## T7 戻り（g1、保存境界の列 t ≥ 1 で一度閾値未満に入った unit が後の境界で閾値以上に戻った数）", "",
          "一意な seed-unit で数える。リスク集合 = 一度未満になり後続の観測がある unit。疎な観測境界での戻りに限る。", ""]
    head = ["腕", "層", "seed-unit 数", "p⁺<1/2 に入った", "リスク集合", "p⁺≥1/2 に戻った", "戻り / 全", "戻り / リスク",
            "m<0 に入った", "リスク集合", "m≥0 に戻った", "戻り / 全"]
    rows = []
    for arm in arms_present(by):
        for L in LAYERS:
            p, m = res.get((arm, L, "p")), res.get((arm, L, "m"))
            if not p or not p["total"]:
                continue
            rows.append([ARM_JA[arm], L, p["total"], p["entered"], p["risk"], p["returned"],
                         fmt(p["returned"] / p["total"], 4), fmt(p["returned"] / p["risk"], 4) if p["risk"] else "—",
                         m["entered"], m["risk"], m["returned"], fmt(m["returned"] / m["total"], 4)])
    md += [table(head, rows), ""]
    if who:
        md += ["戻った unit の内訳（p⁺ 基準、最大 40 行）:", "",
               table(["腕", "seed", "層", "unit", "初めて p⁺<1/2（t）", "p⁺≥1/2 に戻った t", "p⁺ の列（保存境界）"],
                     [[w["arm"], w["seed"], w["layer"], w["unit"], w["first_below_t"], ",".join(map(str, w["back_t"])),
                       " ".join(f"{v:.2f}" for v in w["seq"])] for w in who[:40]]), ""]
    return "\n".join(md), res, who


def diag_md(by, fl):
    out = []
    fn = RAW / "diag_snake_t01_first_step.npz"
    if fn.exists():
        d = np.load(fn)
        slots = [str(s) for s in d["slots"]]
        dmP = d["m_P1_a0"] - d["m0"]; dma = d["m_P1_a1"] - d["m_P1_a0"]; dmt = d["m_P1_a1"] - d["m0"]
        rows = []
        for arm in SNAKE:
            fr = {k: [] for k in ("P", "a", "t", "aP", "at")}
            for r, sl_ in enumerate(slots):
                a, s = sl_.rsplit("_", 1)
                if a != arm:
                    continue
                U = np.load(RAW / f"units_{arm}_s{s}.npz")
                ti = list(U["t"]).index(1)
                al = U["alive"][ti][OFF["f2"]] > 0
                G = U["G_full"][ti][OFF["f2"]]
                q = OFF["f2"]
                fr["P"].append(float((dmP[r][q][al] < 0).mean())); fr["a"].append(float((dma[r][q][al] < 0).mean()))
                fr["t"].append(float((dmt[r][q][al] < 0).mean()))
                fr["aP"].append(float((np.sign(-G[al]) == np.sign(dmP[r][q][al])).mean()))
                fr["at"].append(float((np.sign(-G[al]) == np.sign(dmt[r][q][al])).mean()))
            rows.append([arm, msd(fr["P"], 2), msd(fr["a"], 2), msd(fr["t"], 2), msd(fr["aP"], 2), msd(fr["at"], 2)])
        out += ["### 診断 1: Snake t1 の最初の 1 歩の分解（f2、生きた unit）", "",
                "Δm_P = m(P₁, α₀) − m(P₀, α₀)、Δm_α = m(P₁, α₁) − m(P₁, α₀)、Δm_total = 両者の和。", "",
                table(["腕", "Δm_P < 0", "Δm_α < 0", "Δm_total < 0", "−G_full と Δm_P の一致", "−G_full と Δm_total の一致"], rows), ""]
    f0, f1 = RAW / "diag_LR_t01_first_step.npz", RAW / "diag_LR_t01_first_step_m0.npz"
    if f0.exists() and f1.exists():
        a, b = np.load(f0), np.load(f1)
        rows = []
        for L in LAYERS:
            v0, v1 = [], []
            for r, s in enumerate(a["seeds"]):
                U = np.load(RAW / f"units_LR_s{int(s)}.npz")
                ti = list(U["t"]).index(1)
                al = U["alive"][ti][OFF[L]] > 0
                v0.append(float(((a["m1"][r] - a["m0"][r])[OFF[L]][al] < 0).mean()))
                v1.append(float(((b["m1"][r] - b["m0"][r])[OFF[L]][al] < 0).mean()))
            rows.append([L, msd(v0, 2), msd(v1, 2)])
        out += ["### 診断 2: LR t1 の最初の 1 歩、旧い一次モーメント m を 0 にした複製（実再生とは混ぜない）", "",
                table(["層", "実際の 1 歩で沈んだ割合", "m = 0 の複製で沈んだ割合"], rows), ""]
    return "\n".join(out)


def t_kappaR():
    """A1.5 (optional, no registered prediction): own-direction sensitivity alignment kappa_R per layer."""
    fns = sorted(RAW.glob("kappaR_*.npz"))
    if not fns:
        return "## A1.5 own 方向の感度の旧ラベル整列 κ_R\n\n無し（計算していない）。\n"
    agg = {}
    checks = {"G_own": 0.0, "Soh": 0.0, "ledger": 0.0}
    for fn in fns:
        arm, s_ = fn.stem.split("_")[1], int(fn.stem.split("_s")[-1])
        Rr = np.load(fn)
        U = np.load(RAW / f"units_{arm}_s{s_}.npz")
        ut = list(U["t"])
        for i, t in enumerate(Rr["t"]):
            t = int(t)
            ti = ut.index(t)
            checks["G_own"] = max(checks["G_own"], float(Rr["check_G_own_absdiff"][i] / max(Rr["check_G_own_scale"][i], 1e-300)))
            checks["Soh"] = max(checks["Soh"], float(Rr["check_f2_Soh_absdiff"][i] / max(Rr["check_f2_Soh_scale"][i], 1e-300)))
            checks["ledger"] = max(checks["ledger"], float(Rr["check_ledger_absdiff"][i] / max(Rr["check_G_own_scale"][i], 1e-300)))
            for L in LAYERS:
                q = OFF[L]
                al = U["alive"][ti][q] > 0
                kR, eps, Go, marg, cR = (Rr["kappa_R"][i][q], Rr["eps"][i][q], U["G_own"][ti][q], Rr["margin"][i][q], Rr["c_R"][i][q])
                if not al.any():
                    continue
                d = agg.setdefault((arm, t, L), {k: [] for k in ("pos", "agree", "margin_pos", "cR", "share", "n")})
                d["pos"].append(float((kR[al] > 0).mean()))
                d["agree"].append(float((np.sign(kR[al]) == np.sign(Go[al])).mean()))
                d["margin_pos"].append(float((marg[al] > 0).mean()))
                d["cR"].append(float(np.median(cR[al])))
                part = np.abs(0.9 * kR[al]) / np.maximum(np.abs(0.9 * kR[al]) + np.abs(eps[al]), 1e-300)
                d["share"].append(float(np.median(part)))
                d["n"].append(int(al.sum()))
    md = ["## A1.5 own 方向の感度の旧ラベル整列 κ_R（任意・登録予測なし）", "",
          "R^own_nj = Df_n[u_j^own]（unit j の own 方向に動かしたときの画像 n の logit の変化）、κ_R,j = D⁻¹Σ_n⟨R^c_nj, y_n − e⟩"
          "（D = N(1 − 1/K)）、G_own = (1 − 1/K)κ_R + ε [恒等式]、余裕 = (1 − 1/K)κ_R − ρ e_π > 0 は G_own > 0 の十分条件。"
          f"検算: G_own の再構成の最大相対差 {fmt(checks['G_own'], 1)}、f2 の D·κ_R = S_oh {fmt(checks['Soh'], 1)}、帳簿 {fmt(checks['ledger'], 1)}。"
          "分担 = |(1 − 1/K)κ_R| / (|(1 − 1/K)κ_R| + |ε|) の中央値。seed 平均。", ""]
    head = ["腕", "t", "層", "κ_R > 0", "sign(κ_R) = sign(G_own)", "十分条件の余裕 > 0", "正規化した整列 c_R（中央値）", "κ_R の分担", "生きた unit"]
    rows = []
    for (arm, t, L) in sorted(agg, key=lambda k: (ARMS.index(k[0]), k[1], LAYERS.index(k[2]))):
        d = agg[(arm, t, L)]
        rows.append([ARM_JA[arm], t, L, msd(d["pos"], 2), msd(d["agree"], 2), msd(d["margin_pos"], 2), msd(d["cR"], 3),
                     msd(d["share"], 2), fmt(float(np.mean(d["n"])), 1)])
    md += [table(head, rows), ""]
    return "\n".join(md)


# --------------------------------------------------------------------------
# verdict (A1.3)
# --------------------------------------------------------------------------

def verdict(by, fl, ret):
    V = {}
    # P1
    p1 = {}
    for arm in [a for a in arms_present(by) if a != "R"]:
        need = 10 if arm == "LR" else 8
        per = {}
        for t in tasks_of(by, arm):
            if t < 2:
                continue
            kr = kappa_ratio(by, arm, t, fl)
            per[t] = dict(n_ge_08=sum(1 for v in kr if fin(v) and v >= 0.8), n_defined=sum(1 for v in kr if fin(v)))
        undetermined = any(v["n_defined"] < need for v in per.values())
        ok = all(v["n_ge_08"] >= need for v in per.values())
        qaux = {t: dict(n_Q_ge_08=sum(1 for v in q_ratio(by, arm, t) if fin(v) and v >= 0.8),
                        cos2_mean=stats(col(by, arm, t, "expl_total"))["mean"]) for t in tasks_of(by, arm) if t >= 2}
        p1[arm] = dict(per_boundary=per, need=need, verdict="判定不能" if undetermined and not ok else ("支持" if ok else "不支持"),
                       post_hoc_unregistered_Q_ratio_and_cos2=qaux)
    V["P1"] = p1
    # P2
    rs_all = [r for k, v in by.items() for r in v]
    viol = sum(bool(r["lemma_violation"]) for r in rs_all)
    p2r = {}
    for arm in [a for a in arms_present(by) if a != "R"]:
        per = {t: stats(col(by, arm, t, "kappa_over_lb"))["med"] for t in tasks_of(by, arm)}
        nD = sum(1 for t in tasks_of(by, arm) for r in by[(arm, t)] if not r["D_positive"])
        p2r[arm] = dict(seed_median_per_boundary=per, n_D_below_floor=nD,
                        verdict="支持" if all(fin(v) and v >= 2 for v in per.values()) else "不支持")
    V["P2"] = dict(violations=viol, n_states=len(rs_all), violation_verdict="支持" if viol == 0 else "不支持", ratio=p2r)
    # P3
    p3 = {}
    for arm in [a for a in arms_present(by) if a != "R"]:
        ts = tasks_of(by, arm)
        m = {L: {t: stats(col(by, arm, t, f"{L}_frac_Gfull_pos"))["mean"] for t in ts} for L in LAYERS}
        rec = dict(f2=dict(values=m["f2"], verdict="支持" if all(fin(v) and v >= 0.8 for v in m["f2"].values()) else "不支持"),
                   f1=dict(values=m["f1"], verdict="支持" if all(fin(v) and v >= 0.7 for v in m["f1"].values()) else "不支持"))
        if arm == "LR":
            rec["c2_reproduction"] = dict(values=m["c2"], verdict="支持" if all(fin(v) and v >= 0.7 for v in m["c2"].values()) else "不支持")
            rec["c1_reproduction"] = dict(values=m["c1"], verdict="支持" if all(fin(v) and v <= 0.5 for v in m["c1"].values()) else "不支持")
        else:
            au = {t: stats(col(by, arm, t, "c2_seat_auc"))["mean"] for t in ts}
            rec["c2_valley_auc"] = dict(values=au, verdict="支持" if all(fin(v) and v >= 0.8 for v in au.values() if fin(v))
                                        and any(fin(v) for v in au.values()) else "不支持",
                                        n_states_undefined=sum(1 for t in ts for v in col(by, arm, t, "c2_seat_auc") if not fin(v)))
        p3[arm] = rec
    if any(k[0] == "R" for k in by):
        p3["R_reference"] = {L: {t: stats(col(by, "R", t, f"{L}_frac_Gfull_pos"))["mean"] for t in tasks_of(by, "R")} for L in LAYERS}
    V["P3"] = p3
    # P4
    p4 = {}
    for arm in arms_present(by):
        per = {}
        for t in tasks_of(by, arm):
            if not any("rp_f2_frac_sink1" in r for r in by[(arm, t)]):
                continue
            s1 = stats(col(by, arm, t, "rp_f2_frac_sink1"))["mean"]; s200 = stats(col(by, arm, t, "rp_f2_frac_sink200"))["mean"]
            per[t] = dict(step1=s1, step200=s200, step1_ge_07=bool(fin(s1) and s1 >= 0.7),
                          step200_ge_08=bool(fin(s200) and s200 >= 0.8), step1_in_old_band=bool(fin(s1) and 0.5 <= s1 <= 0.65))
        if per:
            p4[arm] = dict(per_boundary=per,
                           step1_verdict="支持" if all(v["step1_ge_07"] for v in per.values()) else "不支持",
                           step200_verdict="支持" if all(v["step200_ge_08"] for v in per.values()) else "不支持",
                           old_band_verdict="支持" if all(v["step1_in_old_band"] for v in per.values()) else "不支持")
    V["P4"] = p4
    # P5
    p5 = {}
    for arm in [a for a in arms_present(by) if a != "R"]:
        if (arm, 0) not in by:
            continue
        r0 = {r["seed"]: r for r in by[(arm, 0)]}; r1 = {r["seed"]: r for r in by[(arm, 1)]}
        rat = [r0[s]["f2_medabs_Sexact_N"] / r1[s]["f2_medabs_Sexact_N"] for s in sorted(set(r0) & set(r1))
               if fin(r1[s]["f2_medabs_Sexact_N"]) and r1[s]["f2_medabs_Sexact_N"] > 0]
        rat_all = [r0[s]["f2_medabs_Sexact_N_all"] / r1[s]["f2_medabs_Sexact_N_all"] for s in sorted(set(r0) & set(r1))]
        med_ = stats(rat)["med"]
        p5[arm] = dict(seed_median=med_, all_units_seed_median=stats(rat_all)["med"],
                       common_alive_seed_median=stats([common_alive_ratio(arm, s) for s in sorted(set(r0) & set(r1))])["med"],
                       verdict="支持" if fin(med_) and med_ <= 1 / 30 else "不支持")
    V["P5"] = p5
    # P6
    p6 = {}
    for (arm, L, crit), v in ret.items():
        rate = v["returned"] / v["total"] if v["total"] else float("nan")
        p6.setdefault(arm, {})[f"{L}_{crit}"] = dict(**v, rate=rate, risk_rate=v["returned"] / v["risk"] if v["risk"] else float("nan"),
                                                     verdict="支持" if fin(rate) and rate <= 0.01 else "不支持")
    V["P6"] = p6
    V["P7"] = "未実施・採点対象外"
    # P8
    p8 = {}
    for arm in [a for a in arms_present(by) if a in SNAKE]:
        per = {}
        for t in tasks_of(by, arm):
            ag = stats(col(by, arm, t, "f2_agree_self_exact"))["mean"]
            ch = stats(col(by, arm, t, "f2_agree_self_exact_chance"))["mean"]
            neg = [r for r in by[(arm, t)] if r["f2_n_Sself_neg"] > 0]
            fneg = stats([r["f2_frac_Gown_neg_given_Sself_neg"] for r in neg])["mean"]
            per[t] = dict(agree_mean=ag, chance=ch, n_Sself_neg=sum(r["f2_n_Sself_neg"] for r in by[(arm, t)]),
                          n_seeds_with_neg=len(neg), frac_Gown_neg_given_Sself_neg=fneg,
                          agree_ok=bool(fin(ag) and ag >= 0.8), neg_ok=bool(len(neg) > 0 and fin(fneg) and fneg > 0.5))
        p1ok = V["P1"].get(arm, {}).get("verdict") == "支持"
        with_t = [v for v in per.values() if v["n_seeds_with_neg"] > 0]
        neg_where = all(v["neg_ok"] for v in with_t) and bool(with_t)
        n_without = sum(1 for v in per.values() if v["n_seeds_with_neg"] == 0)
        agree_all = all(v["agree_ok"] for v in per.values())
        if p1ok and agree_all and neg_where and n_without == 0:
            verd = "支持"
        elif p1ok and agree_all and neg_where:
            verd = f"対象のある境界では支持（S_self<0 の対象なしの境界 {n_without} 個は支持に数えない）"
        else:
            verd = "不支持"
        p8[arm] = dict(per_boundary=per, P1=V["P1"].get(arm, {}).get("verdict"),
                       agree_verdict="支持" if agree_all else "不支持",
                       neg_subset_verdict=("支持" if neg_where else "不支持") + (f"（対象なしの境界 {n_without}）" if n_without else ""),
                       verdict=verd)
    V["P8"] = p8
    # P9
    p9 = {}
    for arm in [a for a in arms_present(by) if a != "R"]:
        per = {t: dict(f2_Gfull=stats(keep_ratio(by, arm, t, "f2_medabs_Gfull"))["med"],
                       f2_Gown=stats(keep_ratio(by, arm, t, "f2_medabs_Gown"))["med"],
                       c1_Gfull=stats(keep_ratio(by, arm, t, "c1_medabs_Gfull"))["med"]) for t in tasks_of(by, arm) if t > 1}
        rec = dict(per_boundary=per, f2_verdict="支持" if all(fin(v["f2_Gfull"]) and v["f2_Gfull"] >= 0.3 for v in per.values())
                   else "不支持", f2_own_verdict="支持" if all(fin(v["f2_Gown"]) and v["f2_Gown"] >= 0.3 for v in per.values()) else "不支持")
        if arm == "LR":
            rec["c1_below_1_somewhere"] = "支持" if any(fin(v["c1_Gfull"]) and v["c1_Gfull"] < 1 for v in per.values()) else "不支持"
        p9[arm] = rec
    V["P9"] = p9
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    rows, by, fl = load()
    (OUT / "tables_kappa.md").write_text(t1_t2(by, fl) + "\n")
    print("wrote tables_kappa.md")
    if a.only == "kappa":
        return
    t7md, ret, who = t7(by)
    parts = ["# kappa_reload_cnn_1020 — 表 T3〜T7", "", t3(by, fl), t3c(), t4(by, fl), t5(by, fl), t6(by, fl), t7md, diag_md(by, fl),
             t_kappaR()]
    (OUT / "tables_push.md").write_text("\n".join(parts) + "\n")
    V = verdict(by, fl, ret)
    V["returners_p"] = f"raw/returners_g1.json ({len(who)} seed-units, p+ criterion)"
    (RAW / "returners_g1.json").write_text(json.dumps(who, ensure_ascii=False))
    (OUT / "verdict.json").write_text(json.dumps(V, indent=1, ensure_ascii=False, default=float))
    print("wrote tables_push.md, verdict.json")


if __name__ == "__main__":
    main()

"""head0_mlp_1108 追補 C（Adam の v の記憶の向き）の集計・判定・図（spec C.2〜C.6）。

実行:
  cd ~/Projects/claude/wt/head0_cap_1108
  PYTHONPATH=. /home/issan/Projects/claude/proj_004_drift/.venv/bin/python -m analysis.head0_mlp_1108.judge_adamv \
      [--raw results/head0_mlp_1108/adamv/raw] [--vit <posthoc_scale dir>] [--out results/head0_mlp_1108/adamv]

入力は src/adamv_1108.py の raw npz（{ACT}_s{seed}.npz と同名の .json）と、ViT の既存の記録
scale_SL_T100_{base_track, head0_track, adam0at25, bodyadam0at25, head0at25+adam0at25}.json（読むだけ）。
欠けた走・課題は飛ばし、judge_adamv.md の「欠け」に書く。計算はすべて float64。
窓の平均は更新ごとの R・D の算術平均（R が inf の更新を含む窓は inf）。中央値は課題の窓の中の切替について取る。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ACTS = ["LR", "R", "ELU", "SILU", "GELU"]
ACT_JA = {"LR": "leaky 0.1", "R": "ReLU", "ELU": "ELU", "SILU": "SiLU", "GELU": "GELU"}
SEEDS = [0, 1]
ARMS = ["S", "H0", "H0A"]
COL = {"rms_g": 0, "rms_v": 1, "R": 2, "med": 3, "D": 4, "rms_g_head": 5, "loss": 6}
EARLY = list(range(2, 11))
LATE = list(range(21, 31))
DEAD = 0.005
F_MIN, N1_MAX = 0.8, 200            # J C4 の目安（spec C.4 に明記の固定値）
C_S, C_H0, C_H0A = "#000000", "#D55E00", "#009E73"
C_T3, C_T25 = "#0072B2", "#E69F00"
VIT_MODES = ["base_track", "head0_track", "adam0at25", "bodyadam0at25", "head0at25+adam0at25"]
VIT_DIR = Path("/home/issan/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/posthoc_scale")
VIT_N_BODY = 6_011_520              # build_arm_model('SL', 10252010, 'cpu', 0.0, 2, 8.0) の head（layers.out）以外の要素数
VIT_LR = 1e-4                       # analysis/vit_long_1024/common.py の LR（build_optimizer の Adam lr）
VIT_COL = {"base_track": "#000000", "head0_track": "#D55E00", "adam0at25": "#0072B2",
           "bodyadam0at25": "#009E73", "head0at25+adam0at25": "#CC79A7"}


# ---------------------------------------------------------------- 読み込み
def load_raw(raw: Path):
    runs, notes = {}, []
    for act in ACTS:
        for s in SEEDS:
            f = raw / f"{act}_s{s}.npz"
            if not f.exists():
                notes.append(f"{act}_s{s}.npz が無い")
                continue
            z = np.load(f, allow_pickle=False)
            d = {k: z[k].astype(np.float64) if z[k].dtype.kind == "f" else z[k] for k in z.files}
            meta = {}
            js = raw / f"{act}_s{s}.json"
            if js.exists():
                try:
                    meta = json.loads(js.read_text())
                except Exception as e:  # noqa: BLE001
                    notes.append(f"{js.name} を読めない: {e}")
            runs[(act, s)] = (d, meta)
    return runs, notes


def wmean(x, a, b, nan_ok=False):
    """更新 a..b（1 始まり、両端を含む）の平均。系列が b に届かなければ NaN。"""
    if len(x) < b:
        return float("nan")
    seg = np.asarray(x[a - 1:b], dtype=np.float64)
    if nan_ok:
        return float(np.nanmean(seg)) if np.isfinite(seg).any() or np.isinf(seg).any() else float("nan")
    return float(np.mean(seg))


def wnanmean(x, a, b):
    """更新 a..b の NaN を除いた平均（Dn 用: nz = 0 の更新は NaN）。全部 NaN か系列が届かなければ NaN。"""
    if len(x) < b:
        return float("nan")
    seg = np.asarray(x[a - 1:b], dtype=np.float64)
    seg = seg[~np.isnan(seg)]
    return float(np.mean(seg)) if seg.size else float("nan")


def all_zero(nz, a, b):
    """更新 a..b の nz がすべて 0（体の勾配が全部 0）。nz が無いか届かなければ False。"""
    if nz is None or len(nz) < b:
        return False
    seg = np.asarray(nz[a - 1:b], dtype=np.float64)
    return bool(np.all(seg == 0))


def first_below(R, start=2):
    idx = np.nonzero(np.asarray(R[start - 1:]) < 1.0)[0]
    return float(idx[0] + start) if idx.size else float("nan")


def frac_below(R, a=200, b=1000):
    if len(R) < a:
        return float("nan")
    seg = np.asarray(R[a - 1:min(b, len(R))])
    return float(np.mean(seg < 1.0))


def dead_counts(pp0, pp1):
    if pp1 is None or not np.isfinite(pp1).all() or not np.isfinite(pp0).all():
        return float("nan"), float("nan")
    alive0 = pp0 >= DEAD
    return float(np.sum(alive0 & (pp1 < DEAD))), float(np.sum(~alive0 & (pp1 >= DEAD)))


def series_of(d, arm, t):
    if arm == "S":
        k = t - 1
        if k >= d["S_series"].shape[0]:
            return None
        return dict(ser=d["S_series"][k], nz=d.get("S_nz", None)[k] if "S_nz" in d else None,
                    pp=d["pplus_S"][k], acc=d["acc_S"][k])
    k = t - 2
    if k < 0 or k >= d[f"{arm}_series"].shape[0]:
        return None
    return dict(ser=d[f"{arm}_series"][k], nz=d[f"{arm}_nz"][k] if f"{arm}_nz" in d else None,
                pp=d[f"pplus_{arm}"][k], acc=d[f"acc_{arm}"][k])


def build_rows(runs):
    rows, missing = [], []
    for (act, s), (d, meta) in runs.items():
        T = d["S_series"].shape[0]
        S = d["S_series"]
        for arm in ARMS:
            ts = range(1, T + 1) if arm == "S" else range(2, T + 1)
            miss = []
            for t in ts:
                x = series_of(d, arm, t)
                if x is None or not np.isfinite(x["ser"][:, COL["rms_g"]]).any():
                    miss.append(t)
                    continue
                ser = x["ser"]
                R, D, med = ser[:, COL["R"]], ser[:, COL["D"]], ser[:, COL["med"]].copy()
                med[ser[:, COL["rms_v"]] == 0] = np.nan          # v̂ = 0 の更新は状態が無いのと同じ（spec C.2）
                L = ser.shape[0]
                nz = np.asarray(x["nz"], dtype=np.float64) if x["nz"] is not None else np.full(L, np.nan)
                with np.errstate(divide="ignore", invalid="ignore"):
                    Dn = np.where(nz > 0, D / np.sqrt(np.where(nz > 0, nz, 1.0)), np.nan)   # nz = 0 か不明なら NaN
                r = {"act": act, "seed": s, "task": t, "arm": arm, "n_updates": L,
                     "R1": float(R[0]), "R1_5": wmean(R, 1, 5), "R2_5": wmean(R, 2, 5), "R3_5": wmean(R, 3, 5),
                     "D1": float(D[0]), "D1_5": wmean(D, 1, 5), "D6_100": wmean(D, 6, 100), "D101_1000": wmean(D, 101, 1000),
                     "rms_g1": float(ser[0, COL["rms_g"]]), "rms_v1": float(ser[0, COL["rms_v"]]),
                     "Dn1": float(Dn[0]), "Dn1_5": wnanmean(Dn, 1, 5), "Dn6_100": wnanmean(Dn, 6, 100),
                     "Dn101_1000": wnanmean(Dn, 101, 1000),
                     "med1_5": wmean(med, 1, 5, nan_ok=True),
                     "nz1": float(nz[0]), "nz1_5": wmean(nz, 1, 5),
                     "nograd1_5": all_zero(x["nz"], 1, 5), "nograd1_1000": all_zero(x["nz"], 1, min(1000, L)),
                     "loss1_5": wmean(ser[:, COL["loss"]], 1, 5),
                     "n1": first_below(R), "f200_1000": frac_below(R),
                     "acc100": float(x["acc"][0]), "acc1000": float(x["acc"][1]), "acc_end": float(x["acc"][2])}
                pp = x["pp"]
                r["ddead100"], r["reopen100"] = dead_counts(pp[0], pp[1])
                r["ddead1000"], r["reopen1000"] = dead_counts(pp[0], pp[2] if L >= 1000 else None)
                r["dead0"] = float(np.sum(pp[0] < DEAD))
                if t >= 2:
                    g_prev = float(np.mean(S[t - 2, -5:, COL["rms_g"]]))
                    rgS1 = float(S[t - 1, 0, COL["rms_g"]])
                    r["g_prev"] = g_prev
                    r["A_prev"] = float(d["acc_S"][t - 2, 2])
                    r["R_prev"] = g_prev / rgS1 if rgS1 > 0 else float("inf")
                    r["v1_over_gprev"] = float(S[t - 1, 0, COL["rms_v"]]) / g_prev if g_prev > 0 else float("nan")
                else:
                    r.update(g_prev=float("nan"), A_prev=float("nan"), R_prev=float("nan"), v1_over_gprev=float("nan"))
                rows.append(r)
            if miss:
                missing.append((act, s, arm, miss))
    return rows, missing


ROW_KEYS = ["act", "seed", "task", "arm", "n_updates", "R1", "R1_5", "R2_5", "R3_5", "D1", "D1_5", "D6_100", "D101_1000",
            "Dn1", "Dn1_5", "Dn6_100", "Dn101_1000", "rms_g1", "rms_v1", "med1_5", "nz1", "nz1_5", "nograd1_5", "nograd1_1000", "loss1_5", "g_prev", "A_prev", "R_prev", "v1_over_gprev",
            "n1", "f200_1000", "dead0", "ddead100", "reopen100", "ddead1000", "reopen1000", "acc100", "acc1000", "acc_end"]


# ---------------------------------------------------------------- 判定
def med(vals, inf_for_nan=False):
    a = np.asarray(vals, dtype=np.float64)
    if inf_for_nan:
        a = np.where(np.isnan(a), np.inf, a)
    a = a[~np.isnan(a)]
    return float(np.median(a)) if a.size else float("nan")


def side(x):
    if not np.isfinite(x) and not np.isinf(x):
        return "—"
    return ">1" if x > 1 else ("<1" if x < 1 else "=1")


def get_rows(rows, act, s, arm, tasks):
    tset = set(tasks)
    return [r for r in rows if r["act"] == act and r["seed"] == s and r["arm"] == arm and r["task"] in tset]


def judge_run(rows, act, s):
    J = {}
    # J C1
    e = get_rows(rows, act, s, "S", EARLY)
    ev = [r for r in e if not r["nograd1_5"]]          # 窓の全更新で nz = 0 の切替は R も Dn も意味が無いので除き、数を出す
    mR, mD = med([r["R1_5"] for r in ev]), med([r["Dn1_5"] for r in ev])
    if not e:
        lab = "NOT_AVAILABLE"
    elif not ev:
        lab = "NO_GRADIENT"
    elif not (np.isfinite(mR) or np.isinf(mR)) or not np.isfinite(mD):
        lab = "NOT_AVAILABLE"
    else:
        lab = "LIT_EARLY" if (mR > 1 and mD > 1) else ("OURS_EARLY" if (mR < 1 and mD < 1) else "MIXED")
    J["C1"] = dict(tasks=[r["task"] for r in e], n_nograd=len(e) - len(ev), R1_5_med=mR, Dn1_5_med=mD,
                   D1_5_med=med([r["D1_5"] for r in ev]), nz1_5_med=med([r["nz1_5"] for r in ev]), label=lab,
                   R_prev_med=med([r["R_prev"] for r in e]), g_prev_med=med([r["g_prev"] for r in e]),
                   A_prev_med=med([r["A_prev"] for r in e]))
    # J C2
    h = get_rows(rows, act, s, "H0", LATE)
    hv = [r for r in h if not r["nograd1_5"]]
    mR, mD = med([r["R2_5"] for r in hv]), med([r["Dn1_5"] for r in hv])
    if not h:
        lab = "NOT_AVAILABLE"
    elif not hv:
        lab = "NO_GRADIENT"
    elif not np.isfinite(mR) or not np.isfinite(mD):
        lab = "NOT_AVAILABLE"
    else:
        lab = "OURS_LATE_H0" if (mR < 1 and mD < 1) else ("LIT_LATE_H0" if (mR > 1 and mD > 1) else "MIXED")
    J["C2"] = dict(tasks=[r["task"] for r in h], n_nograd=len(h) - len(hv), R2_5_med=mR, Dn1_5_med=mD,
                   D1_5_med=med([r["D1_5"] for r in hv]), nz1_5_med=med([r["nz1_5"] for r in hv]), label=lab,
                   R1_med=med([r["R1"] for r in hv]), D1_med=med([r["D1"] for r in hv]))
    # J C3
    l_ = get_rows(rows, act, s, "S", LATE)
    lv = [r for r in l_ if not r["nograd1_5"]]
    mR, mD = med([r["R1_5"] for r in lv]), med([r["Dn1_5"] for r in lv])
    J["C3"] = dict(tasks=[r["task"] for r in l_], n_nograd=len(l_) - len(lv), R1_5_med=mR, Dn1_5_med=mD,
                   D1_5_med=med([r["D1_5"] for r in lv]), nz1_5_med=med([r["nz1_5"] for r in lv]),
                   label=("NOT_AVAILABLE" if not l_ else ("NO_GRADIENT" if not lv else f"R{side(mR)}・Dn{side(mD)}")),
                   R_prev_med=med([r["R_prev"] for r in l_]), g_prev_med=med([r["g_prev"] for r in l_]),
                   A_prev_med=med([r["A_prev"] for r in l_]))
    # J C4（遅い切替で判定、早い切替も同じ量を記録）
    for key, rr0 in (("C4_late", l_), ("C4_early", e)):
        rr = [r for r in rr0 if not r["nograd1_1000"]]   # 更新 1〜1,000 の全部で nz = 0 の切替は R = 0 が無意味なので除く
        n1 = [r["n1"] for r in rr]
        fs = [r["f200_1000"] for r in rr]
        mn1, mf = med(n1, inf_for_nan=True), med(fs)
        if not rr0:
            lab = "NOT_AVAILABLE"
        elif not rr:
            lab = "NO_GRADIENT"
        elif not np.isfinite(mf):
            lab = "NOT_AVAILABLE"
        else:
            lab = "SPIKE_SUPPRESSES" if (mn1 <= N1_MAX and mf >= F_MIN) else "NOT_SUPPRESSED"
        J[key] = dict(tasks=[r["task"] for r in rr0], n_nograd=len(rr0) - len(rr), n1_med=mn1, f_med=mf, n1_never=int(sum(np.isnan(n1))),
                      n1_list=n1, f_list=fs, label=lab)
    # J C5（全切替、S）
    a_ = [r for r in rows if r["act"] == act and r["seed"] == s and r["arm"] == "S" and r["task"] >= 2
          and np.isfinite(r["ddead100"]) and not np.isnan(r["R1_5"])]
    dd = np.array([r["ddead100"] for r in a_], dtype=np.float64)
    big = np.array([r["R1_5"] > 1 for r in a_], dtype=bool)
    tot = float(dd.sum()) if dd.size else 0.0
    base = float(big.mean()) if big.size else float("nan")
    share = float(dd[big].sum() / tot) if tot > 0 else float("nan")
    if not a_:
        lab = "NOT_AVAILABLE"
    elif tot < 5:
        lab = "NOT_TESTABLE"
    else:
        lab = "DEATHS_AT_LARGE_STEPS" if share > base else "NOT_AT_LARGE_STEPS"
    early_dd = float(sum(r["ddead100"] for r in a_ if r["task"] in EARLY))
    J["C5"] = dict(n_switches=len(a_), ddead100_total=tot, share=share, base=base, label=lab,
                   ddead100_early=early_dd, early_share=(early_dd / tot if tot > 0 else float("nan")),
                   n_big=int(big.sum()), ddead1000_total=float(np.nansum([r["ddead1000"] for r in a_])),
                   reopen100_total=float(np.nansum([r["reopen100"] for r in a_])))
    # 読み 1 の前提
    ratios = [r["v1_over_gprev"] for r in a_]
    m = med(ratios)
    J["premise"] = dict(median=m, n=int(np.isfinite(ratios).sum()) if ratios else 0,
                        early_median=med([r["v1_over_gprev"] for r in e]), late_median=med([r["v1_over_gprev"] for r in l_]),
                        label=("—" if not np.isfinite(m) else ("入る" if 0.5 <= m <= 2 else "入らない")))
    # 腕ごとの窓の中央値（記録）
    arms = {}
    for arm in ARMS:
        for wname, ts in (("early", EARLY), ("late", LATE)):
            rr = get_rows(rows, act, s, arm, ts)
            arms[f"{arm}_{wname}"] = {q: med([r[q] for r in rr]) for q in
                                      ("R1", "R1_5", "R2_5", "R3_5", "D1", "D1_5", "D6_100", "D101_1000", "Dn1", "Dn1_5",
                                       "Dn6_100", "Dn101_1000", "rms_g1", "rms_v1", "med1_5", "nz1", "nz1_5", "ddead100", "ddead1000", "reopen100", "acc100", "acc1000", "acc_end")}
            arms[f"{arm}_{wname}"]["n"] = len(rr)
    J["arms"] = arms
    return J


def seed_agree(labels):
    labs = [l for l in labels if l is not None]
    if len(labs) < 2:
        return "未走（seed 欠け）" if not labs else f"seed 1 本のみ {labs[0]}"
    return f"確定 {labs[0]}" if all(l == labs[0] for l in labs) else "割れ"


def score(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    tag = "（seed 1 本）" if len(vals) == 1 else ""
    if all(vals):
        return "当たり" + tag
    if not any(vals):
        return "外れ" + tag
    return "割れ"


def inr(x, lo, hi):
    return None if not np.isfinite(x) else bool(lo <= x <= hi)


# ---------------------------------------------------------------- ViT
def vit_read(vdir: Path):
    out, notes = {}, []
    for mode in VIT_MODES:
        f = vdir / f"scale_SL_T100_{mode}.json"
        if not f.exists():
            notes.append(f"ViT {f.name} が無い")
            continue
        tr = json.loads(f.read_text())["track"]
        step = np.asarray(tr["step"], dtype=np.int64)
        g = np.asarray(tr["g_body"], dtype=np.float64) / math.sqrt(VIT_N_BODY)
        v = np.asarray(tr["v_body"], dtype=np.float64)
        dlt = np.asarray(tr["d_body"], dtype=np.float64) / math.sqrt(VIT_N_BODY)
        with np.errstate(divide="ignore", invalid="ignore"):
            R = g / v
            Rlag = np.full_like(R, np.nan); Rlag[1:] = g[1:] / v[:-1]
        D = dlt / VIT_LR
        def at(a, n):
            i = np.nonzero(step == n)[0]
            return float(a[i[0]]) if i.size else float("nan")
        below = np.nonzero(Rlag < 1)[0]                 # 判定の量は R_lag（step の前の v に近い）
        n_last = int(step[-1])
        out[mode] = dict(step=step, R=R, Rlag=Rlag, D=D, loss=np.asarray(tr["loss"], dtype=np.float64),
                         first_R_below1=(int(step[below[0]]) if below.size else None),
                         R1=at(R, 1), R25=at(R, 25), R26=at(R, 26), R100=at(R, 100), R500=at(Rlag, 500),
                         Rpost500=at(R, 500), Rlag2=at(Rlag, 2), Rlag100=at(Rlag, 100),
                         R_last=float(Rlag[-1]), n_last=n_last,
                         R_med_401_500=float(np.median(Rlag[(step >= 401) & (step <= 500)])) if n_last >= 401 else float("nan"),
                         Rlag26=at(Rlag, 26), Rlag500=at(Rlag, 500), first_Rpost_below1=(int(step[np.nonzero(R < 1)[0][0]]) if (R < 1).any() else None),
                         D1=at(D, 1), D25=at(D, 25), D26=at(D, 26), D500=at(D, 500),
                         D_med_401_500=float(np.median(D[(step >= 401) & (step <= 500)])) if n_last >= 401 else float("nan"))
    return out, notes


# ---------------------------------------------------------------- 書式
def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, str):
        return x
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if np.isnan(x):
        return "—"
    if np.isinf(x):
        return "∞" if x > 0 else "−∞"
    if x == 0:
        return "0"
    if abs(x) < 10 ** (-nd) or abs(x) >= 1e5:
        return f"{x:.2e}"
    if abs(x) >= 100:
        return f"{x:.0f}" if float(x).is_integer() else f"{x:.1f}"
    return f"{x:.{nd}f}"


def tf(v):
    return "—" if v is None else ("○" if v else "×")


# ---------------------------------------------------------------- 採点
def scoring(J, V):
    """送り手 5 つと受け手の表。seed ごとの真偽 → 活性化ごと（当たり／外れ／割れ）。"""
    S_ = {}
    def per(act, fn):
        vals = [fn(J[f"{act}_s{s}"]) if f"{act}_s{s}" in J else None for s in SEEDS]
        return vals, score(vals)

    def p1(j):
        if j["C1"]["label"] in ("NOT_AVAILABLE", "NO_GRADIENT"):
            return None
        if j["C1"]["label"] != "LIT_EARLY":
            return False
        es = j["C5"]["early_share"]
        return None if not np.isfinite(es) else bool(es >= 0.5)
    def p2(j):
        r, d = j["C2"]["R2_5_med"], j["C2"]["Dn1_5_med"]
        return None if not (np.isfinite(r) and np.isfinite(d)) else bool(r < 0.1 and d < 0.3)
    def p3(j):
        return inr(j["C3"]["R1_5_med"], 0.3, 3)
    def p4(j):
        return None if j["C4_late"]["label"] in ("NOT_AVAILABLE", "NO_GRADIENT") else j["C4_late"]["label"] == "SPIKE_SUPPRESSES"
    sender = {}
    for act in ACTS:
        sender[act] = {k: per(act, f) for k, f in (("1", p1), ("2", p2), ("3", p3), ("4", p4))}
    b = V.get("base_track")
    # 突発は R_lag(2) > 1（R_lag(1) は前の記録が無い）、1 を割る更新と R(500) も R_lag で
    v5 = None if b is None else bool(b["Rlag2"] > 1 and b["first_R_below1"] is not None and np.isfinite(b["R500"]) and b["R500"] < 1)
    S_["sender"] = sender
    S_["sender_vit"] = v5

    recv = {}
    for act in ACTS:
        recv[act] = {
            "C1_label": per(act, lambda j: None if j["C1"]["label"] in ("NOT_AVAILABLE", "NO_GRADIENT") else j["C1"]["label"] == "LIT_EARLY"),
            "C1_R": per(act, lambda j: inr(j["C1"]["R1_5_med"], 3, 30)),
            "C1_D": per(act, lambda j: inr(j["C1"]["Dn1_5_med"], 1.5, 3)),
            "C2_label": per(act, lambda j: None if j["C2"]["label"] in ("NOT_AVAILABLE", "NO_GRADIENT") else j["C2"]["label"] == "OURS_LATE_H0"),
            "C2_R": per(act, lambda j: inr(j["C2"]["R2_5_med"], 0.005, 0.05)),
            "C2_D": per(act, lambda j: inr(j["C2"]["Dn1_5_med"], 0.05, 0.3)),
            "C3_R": per(act, lambda j: None if not np.isfinite(j["C3"]["R1_5_med"]) else bool(1 < j["C3"]["R1_5_med"] <= 3)),
            "C3_D": per(act, lambda j: inr(j["C3"]["Dn1_5_med"], 0.5, 1.5)),
            "C4_late": per(act, lambda j: None if j["C4_late"]["label"] in ("NOT_AVAILABLE", "NO_GRADIENT") else j["C4_late"]["label"] == "SPIKE_SUPPRESSES"),
            "C4_early": per(act, lambda j: None if j["C4_early"]["label"] in ("NOT_AVAILABLE", "NO_GRADIENT") else j["C4_early"]["label"] == "SPIKE_SUPPRESSES"),
            "C4_n1": per(act, lambda j: inr(j["C4_late"]["n1_med"], 50, 200)),
            "C4_f": per(act, lambda j: inr(j["C4_late"]["f_med"], 0.8, 1.0)),
            "C5_label": per(act, lambda j: None if j["C5"]["label"] in ("NOT_AVAILABLE", "NOT_TESTABLE") else j["C5"]["label"] == "DEATHS_AT_LARGE_STEPS"),
            "C5_share": per(act, lambda j: None if not np.isfinite(j["C5"]["share"]) else bool(j["C5"]["share"] >= 0.8)),
            "premise": per(act, lambda j: inr(j["premise"]["median"], 0.7, 1.5)),
        }
    S_["receiver"] = recv
    a0 = V.get("adam0at25")
    S_["receiver_vit"] = {
        "base_first_below_20_60": None if b is None else (b["first_R_below1"] is not None and 20 <= b["first_R_below1"] <= 60),
        "base_R500_below1": None if b is None or not np.isfinite(b["R500"]) else bool(b["R500"] < 1),
        "adam0at25_D26_approx1": None if a0 is None else inr(a0["D26"], 0.8, 1.25),
    }
    return S_


# ---------------------------------------------------------------- md
def write_md(path, J, checks, missing, notes, V, SC, act_summary):
    L = []
    w = L.append
    w("# head0_mlp_1108 追補 C 判定（Adam の v の記憶の向き、spec C.4）\n")
    w("早い切替 = 課題 2〜10、遅い切替 = 課題 21〜30。R(1〜5) などは更新ごとの値の窓の算術平均、表の値は窓の中の切替についての中央値。"
      "H0 は更新 1 の体の勾配が厳密に 0 なので R は 2〜5 で見る。H0A は更新 1 の体の勾配が 0（head が 0）で v̂ が更新 2 の前も 0 なので、R は 3〜5 で見る（図）。\n")
    w("歩幅の判定の量は Dn(n) = D(n)/√nz(n)（nz = 体の勾配が 0 でない座標の割合、Adam 状態が無い一歩で厳密に 1）。nz = 0 の更新の Dn は NaN とし、窓の平均は NaN を除いて取る"
      "（H0 は更新 1 の nz が 0 なので Dn(1〜5) は実質 2〜5 の平均）。窓（R・Dn は 1〜5、J C4 は 1〜1,000）の全更新で nz = 0 の切替は「勾配が無い」として中央値から除き、"
      "表の「勾配なし」に数を出す。窓の切替が全部そうなら NO_GRADIENT。D は記録として残す。\n")
    w("## 検査\n")
    w("| 活性化 | seed | 課題末 θ の差の最大（S、全課題） | 課題数 | H0 R(1)（t2） | H0 D(1)（t2） | H0A rms_v(1)（t2） | H0A D(1)（t2） | S D(1)（t1） | √(sign の一歩の予想)（t1） | torch | git | 秒 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for act in ACTS:
        for s in SEEDS:
            c = checks.get(f"{act}_s{s}")
            if c is None:
                continue
            w(f"| {ACT_JA[act]} | {s} | {fmt(c['end_max'], 4)} | {c['n_end']} | {fmt(c.get('H0_t02_R1'))} | {fmt(c.get('H0_t02_D1'), 4)} | "
              f"{fmt(c.get('H0A_t02_rms_v1'))} | {fmt(c.get('H0A_t02_D1'), 4)} | {fmt(c.get('S_t01_D1'), 6)} | {fmt(c.get('S_t01_D1_exp'), 6)} | "
              f"{c.get('torch', '—')} | {str(c.get('git', '—'))[:10]} | {fmt(c.get('elapsed'), 1)} |")
    w("\nAdam の状態が無い一歩の D は √mean((g/(|g|+ε))²) で、勾配が 0 でない座標の割合の平方根にほぼ等しい（1 ではない）。"
      "1,200 枚すべてで 0 の画素が 784 のうち約 21%、ミニバッチ 16 枚で 0 の画素はさらに多い。表の「√(sign の一歩の予想)」がその値。\n")
    if missing or notes:
        w("## 欠け\n")
        for a_, s, arm, ts in missing:
            w(f"- {ACT_JA[a_]} seed {s} {arm}: 課題 {ts}")
        for n in notes:
            w(f"- {n}")
        w("")

    def J_(act, s):
        return J.get(f"{act}_s{s}")

    w("## J C1 早い切替の S（課題 2〜10）\n")
    w("| 活性化 | seed | R(1〜5) 中央値 | Dn(1〜5) 中央値 | D(1〜5) 中央値 | nz(1〜5) 中央値 | R_prev 中央値 | g_prev 中央値 | A_prev 中央値 | 切替数 | 勾配なし | ラベル |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for act in ACTS:
        for s in SEEDS:
            j = J_(act, s)
            if j is None: continue
            c = j["C1"]
            w(f"| {ACT_JA[act]} | {s} | {fmt(c['R1_5_med'])} | {fmt(c['Dn1_5_med'])} | {fmt(c['D1_5_med'])} | {fmt(c['nz1_5_med'])} | "
              f"{fmt(c['R_prev_med'])} | {fmt(c['g_prev_med'], 4)} | {fmt(c['A_prev_med'])} | {len(c['tasks'])} | {c['n_nograd']} | {c['label']} |")
    w("\n## J C2 遅い切替の H0（課題 21〜30）\n")
    w("| 活性化 | seed | R(2〜5) 中央値 | Dn(1〜5) 中央値 | D(1〜5) 中央値 | nz(1〜5) 中央値 | R(1) 中央値 | D(1) 中央値 | 切替数 | 勾配なし | ラベル |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for act in ACTS:
        for s in SEEDS:
            j = J_(act, s)
            if j is None: continue
            c = j["C2"]
            w(f"| {ACT_JA[act]} | {s} | {fmt(c['R2_5_med'], 4)} | {fmt(c['Dn1_5_med'])} | {fmt(c['D1_5_med'])} | {fmt(c['nz1_5_med'])} | "
              f"{fmt(c['R1_med'])} | {fmt(c['D1_med'])} | {len(c['tasks'])} | {c['n_nograd']} | {c['label']} |")
    w("\n## J C3 遅い切替の S（課題 21〜30、記録）\n")
    w("| 活性化 | seed | R(1〜5) 中央値 | Dn(1〜5) 中央値 | D(1〜5) 中央値 | nz(1〜5) 中央値 | R_prev 中央値 | g_prev 中央値 | A_prev 中央値 | 切替数 | 勾配なし | 側 |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for act in ACTS:
        for s in SEEDS:
            j = J_(act, s)
            if j is None: continue
            c = j["C3"]
            w(f"| {ACT_JA[act]} | {s} | {fmt(c['R1_5_med'])} | {fmt(c['Dn1_5_med'])} | {fmt(c['D1_5_med'])} | {fmt(c['nz1_5_med'])} | "
              f"{fmt(c['R_prev_med'])} | {fmt(c['g_prev_med'], 4)} | {fmt(c['A_prev_med'])} | {len(c['tasks'])} | {c['n_nograd']} | {c['label']} |")
    w("\n## J C4 読み 2（S の課題の中）\n")
    w(f"n₁ = R が初めて 1 を割る更新（更新 2 以降、4,000 更新で割らなければ中央値では ∞ として数える）、f = 更新 200〜1,000 で R < 1 の割合。"
      f"中央値 n₁ ≤ {N1_MAX} かつ中央値 f ≥ {F_MIN} → SPIKE_SUPPRESSES（目安の固定値、spec C.4）。\n")
    w("| 活性化 | seed | 遅い n₁ 中央値 | 遅い f 中央値 | 遅い 割らない数 | 遅い 勾配なし | 遅い ラベル | 早い n₁ 中央値 | 早い f 中央値 | 早い 割らない数 | 早い 勾配なし | 早い ラベル |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for act in ACTS:
        for s in SEEDS:
            j = J_(act, s)
            if j is None: continue
            a, b = j["C4_late"], j["C4_early"]
            w(f"| {ACT_JA[act]} | {s} | {fmt(a['n1_med'])} | {fmt(a['f_med'])} | {a['n1_never']}/{len(a['tasks'])} | {a['n_nograd']} | {a['label']} | "
              f"{fmt(b['n1_med'])} | {fmt(b['f_med'])} | {b['n1_never']}/{len(b['tasks'])} | {b['n_nograd']} | {b['label']} |")
    w("\n## J C5 死は大きな歩幅の切替に集まるか（S、全切替）\n")
    w("| 活性化 | seed | Δdead_100 合計 | R(1〜5) > 1 の切替の share | base（R > 1 の切替の割合） | R > 1 の切替数 / 切替数 | 早い切替の割合 | Δdead_1000 合計 | 開き直し（100）合計 | ラベル |\n|---|---|---|---|---|---|---|---|---|---|")
    for act in ACTS:
        for s in SEEDS:
            j = J_(act, s)
            if j is None: continue
            c = j["C5"]
            w(f"| {ACT_JA[act]} | {s} | {fmt(c['ddead100_total'])} | {fmt(c['share'])} | {fmt(c['base'])} | {c['n_big']}/{c['n_switches']} | "
              f"{fmt(c['early_share'])} | {fmt(c['ddead1000_total'])} | {fmt(c['reopen100_total'])} | {c['label']} |")
    w("\n## 読み 1 の前提（RMS_v(1) / g_prev）\n")
    w("| 活性化 | seed | 全切替の中央値 | 早い中央値 | 遅い中央値 | 切替数 | 0.5〜2 |\n|---|---|---|---|---|---|---|")
    for act in ACTS:
        for s in SEEDS:
            j = J_(act, s)
            if j is None: continue
            c = j["premise"]
            w(f"| {ACT_JA[act]} | {s} | {fmt(c['median'])} | {fmt(c['early_median'])} | {fmt(c['late_median'])} | {c['n']} | {c['label']} |")

    w("\n## 腕ごとの窓の中央値（記録）\n")
    w("| 活性化 | seed | 腕 | 窓 | R(1) | R(1〜5) | R(2〜5) | R(3〜5) | D(1) | D(1〜5) | Dn(1〜5) | Dn(6〜100) | Dn(101〜1,000) | D(101〜1,000) | med(1〜5) | nz(1〜5) | Δdead_100 | Δdead_1000 | 精度 100 | 精度 1,000 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for act in ACTS:
        for s in SEEDS:
            j = J_(act, s)
            if j is None: continue
            for arm in ARMS:
                for wn, wj in (("early", "早い"), ("late", "遅い")):
                    a = j["arms"][f"{arm}_{wn}"]
                    if a["n"] == 0: continue
                    w(f"| {ACT_JA[act]} | {s} | {arm} | {wj} | {fmt(a['R1'])} | {fmt(a['R1_5'])} | {fmt(a['R2_5'])} | {fmt(a['R3_5'])} | "
                      f"{fmt(a['D1'])} | {fmt(a['D1_5'])} | {fmt(a['Dn1_5'])} | {fmt(a['Dn6_100'])} | {fmt(a['Dn101_1000'])} | {fmt(a['D101_1000'])} | "
                      f"{fmt(a['med1_5'])} | {fmt(a['nz1_5'])} | "
                      f"{fmt(a['ddead100'])} | {fmt(a['ddead1000'])} | {fmt(a['acc100'])} | {fmt(a['acc1000'])} |")

    w("\n## ViT（SL seed 10252010、課題 100 の末から、既存の記録）\n")
    w(f"R(n) = (‖g_body‖/√n_body) / v_body、D(n) = (‖Δθ_body‖/√n_body) / lr。n_body = {VIT_N_BODY:,}、lr = {VIT_LR:g}。"
      "記録の v_body は更新 n の step の後の √mean(v) で bias 補正なし（MLP の R は step の前の v̂）。R_lag は v を更新 n−1 の後の値に替えたもの（step の前の v に近い）。**「R が 1 を割る最初の更新」・R(500)・401〜500 の中央値は R_lag で取る**（step の後の v の R は R_post として並べる）。"
      "ViT の記録には nz が無いので D は正規化していない。"
      "adam0at25 などは更新 25 の後に状態を消すので、更新 26 の v は (1−β₂)g² で bias 補正が無く、R(26) は約 √1000 ≈ 32 倍に出る。\n")
    w("| モード | R_lag が 1 を割る最初の更新 | R_post が 1 を割る最初の更新 | R_post(1) | R_lag(2) | R_post(25) | R_post(26) | R_lag(26) | R_lag(100) | R_lag(500) | R_post(500) | R_lag の中央値（401〜500） | D(1) | D(25) | D(26) | D(500) | D の中央値（401〜500） | 記録の長さ |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for mode in VIT_MODES:
        v = V.get(mode)
        if v is None: continue
        w(f"| {mode} | {fmt(v['first_R_below1'])} | {fmt(v['first_Rpost_below1'])} | {fmt(v['R1'])} | {fmt(v['Rlag2'])} | {fmt(v['R25'])} | {fmt(v['R26'])} | "
          f"{fmt(v['Rlag26'])} | {fmt(v['Rlag100'])} | {fmt(v['R500'])} | {fmt(v['Rpost500'])} | {fmt(v['R_med_401_500'])} | {fmt(v['D1'])} | {fmt(v['D25'])} | {fmt(v['D26'])} | "
          f"{fmt(v['D500'])} | {fmt(v['D_med_401_500'])} | {v['n_last']} |")

    w("\n## 判定のまとめ（活性化ごと、2 seed が同じなら確定、違えば割れ）\n")
    w("| 活性化 | J C1 | J C2 | J C3 | J C4（遅い） | J C4（早い） | J C5 | 読み 1 の前提 |\n|---|---|---|---|---|---|---|---|")
    for act in ACTS:
        a = act_summary[act]
        w(f"| {ACT_JA[act]} | {a['C1']} | {a['C2']} | {a['C3']} | {a['C4_late']} | {a['C4_early']} | {a['C5']} | {a['premise']} |")

    w("\n## 予測の採点\n")
    w("seed ごとに ○（当たり）× （外れ）を出し、2 seed とも ○ → 当たり、2 seed とも × → 外れ、それ以外 → 割れ。\n")
    w("### 送り手（Fable 5.1、Vit実験の振り返り）\n")
    w("採点の基準（予測の D は Dn と読む、spec の 1108 16:00 の修正）: 1 = J C1 が LIT_EARLY かつ Δdead_100 の半分以上が早い切替（課題 2〜10）、2 = H0 遅い R(2〜5) 中央値 < 0.1 かつ Dn(1〜5) 中央値 < 0.3、"
      "3 = S 遅い R(1〜5) 中央値が 0.3〜3、4 = J C4（遅い）が SPIKE_SUPPRESSES、5 = ViT base_track で R_lag(2) > 1・R_lag が 1 を割る更新がある・R_lag(500) < 1。NO_GRADIENT は採点しない（—）。\n")
    w("| 活性化 | 1 seed0/seed1 | 1 | 2 seed0/seed1 | 2 | 3 seed0/seed1 | 3 | 4 seed0/seed1 | 4 |\n|---|---|---|---|---|---|---|---|---|")
    for act in ACTS:
        sd = SC["sender"][act]
        w(f"| {ACT_JA[act]} | " + " | ".join(f"{tf(sd[k][0][0])}/{tf(sd[k][0][1])} | {sd[k][1]}" for k in ("1", "2", "3", "4")) + " |")
    v5 = SC["sender_vit"]
    w(f"\n予測 5（ViT）: {'—' if v5 is None else ('当たり' if v5 else '外れ')}\n")
    w("### 受け手（Fable 5.1、このセッション）\n")
    rows_ = [("C1_label", "J C1 = LIT_EARLY"), ("C1_R", "J C1 R(1〜5) 3〜30"), ("C1_D", "J C1 Dn(1〜5) 1.5〜3"),
             ("C2_label", "J C2 = OURS_LATE_H0"), ("C2_R", "J C2 R(2〜5) 0.005〜0.05"), ("C2_D", "J C2 Dn(1〜5) 0.05〜0.3"),
             ("C3_R", "J C3 R(1〜5) 1 より上で 3 以下"), ("C3_D", "J C3 Dn(1〜5) 0.5〜1.5"),
             ("C4_late", "J C4 遅い = SPIKE_SUPPRESSES"), ("C4_early", "J C4 早いも同じ形"), ("C4_n1", "J C4 n₁ 50〜200"), ("C4_f", "J C4 f 0.8〜1.0"),
             ("C5_label", "J C5 = DEATHS_AT_LARGE_STEPS"), ("C5_share", "J C5 share ≥ 0.8"), ("premise", "読み 1 の前提 0.7〜1.5")]
    w("| 予測 | " + " | ".join(ACT_JA[a] for a in ACTS) + " |\n|---|" + "---|" * len(ACTS))
    for k, name in rows_:
        cells = []
        for act in ACTS:
            vals, sc = SC["receiver"][act][k]
            cells.append(f"{sc}（{tf(vals[0])}/{tf(vals[1])}）")
        w(f"| {name} | " + " | ".join(cells) + " |")
    rv = SC["receiver_vit"]
    w("\n| ViT の予測 | 採点 |\n|---|---|")
    for k, name in (("base_first_below_20_60", "base は更新 20〜60 で R < 1（R_lag）"), ("base_R500_below1", "base は 500 更新で戻らない（R_lag(500) < 1）"),
                    ("adam0at25_D26_approx1", "adam0at25 の更新 26 の D ≈ 1（0.8〜1.25 で採点）")):
        v = rv[k]
        w(f"| {name} | {'—' if v is None else ('当たり' if v else '外れ')} |")
    w("")
    path.write_text("\n".join(L))


def write_summary(path, J, V, act_summary):
    L = ["# 追補 C の数（解釈なし）\n"]
    w = L.append
    w("| 活性化 | seed | C1 R(1〜5) | C1 Dn(1〜5) | C2 R(2〜5) | C2 Dn(1〜5) | C3 R(1〜5) | C3 Dn(1〜5) | C4 遅い n₁ | C4 遅い f | C4 早い n₁ | C4 早い f | C5 Δdead_100 | C5 share | C5 base | 前提 v/g_prev | 勾配なし C1/C2/C3/C4 遅い |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for act in ACTS:
        for s in SEEDS:
            j = J.get(f"{act}_s{s}")
            if j is None: continue
            w(f"| {ACT_JA[act]} | {s} | {fmt(j['C1']['R1_5_med'])} | {fmt(j['C1']['Dn1_5_med'])} | {fmt(j['C2']['R2_5_med'], 4)} | {fmt(j['C2']['Dn1_5_med'])} | "
              f"{fmt(j['C3']['R1_5_med'])} | {fmt(j['C3']['Dn1_5_med'])} | {fmt(j['C4_late']['n1_med'])} | {fmt(j['C4_late']['f_med'])} | "
              f"{fmt(j['C4_early']['n1_med'])} | {fmt(j['C4_early']['f_med'])} | {fmt(j['C5']['ddead100_total'])} | {fmt(j['C5']['share'])} | "
              f"{fmt(j['C5']['base'])} | {fmt(j['premise']['median'])} | "
              f"{j['C1']['n_nograd']}/{j['C2']['n_nograd']}/{j['C3']['n_nograd']}/{j['C4_late']['n_nograd']} |")
    w("\n| 活性化 | J C1 | J C2 | J C3 | J C4 遅い | J C4 早い | J C5 | 前提 |\n|---|---|---|---|---|---|---|---|")
    for act in ACTS:
        a = act_summary[act]
        w(f"| {ACT_JA[act]} | {a['C1']} | {a['C2']} | {a['C3']} | {a['C4_late']} | {a['C4_early']} | {a['C5']} | {a['premise']} |")
    w("\n| ViT モード | R_lag < 1 の最初の更新 | R_lag(500) | D(26) |\n|---|---|---|---|")
    for mode in VIT_MODES:
        v = V.get(mode)
        if v is None: continue
        w(f"| {mode} | {fmt(v['first_R_below1'])} | {fmt(v['R500'])} | {fmt(v['D26'])} |")
    w("")
    path.write_text("\n".join(L))


# ---------------------------------------------------------------- 図
def _font():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"] if "Noto Sans CJK JP" in names else ["DejaVu Sans"]
    return plt


def _pos(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where((x > 0) & np.isfinite(x), x, np.nan)


def make_fig_mlp(path, rows, runs):
    plt = _font()
    fig, axes = plt.subplots(3, len(ACTS), figsize=(4.0 * len(ACTS), 10.5), squeeze=False)
    for c, act in enumerate(ACTS):
        # 上: R・D の切替ごとの値（2 seed の中央値）
        ax = axes[0, c]
        for arm, col, rq in (("S", C_S, "R1_5"), ("H0", C_H0, "R2_5"), ("H0A", C_H0A, "R3_5")):
            ts = sorted({r["task"] for r in rows if r["act"] == act and r["arm"] == arm})
            Rm = [med([r[rq] for r in rows if r["act"] == act and r["arm"] == arm and r["task"] == t]) for t in ts]
            Dm = [med([r["Dn1_5"] for r in rows if r["act"] == act and r["arm"] == arm and r["task"] == t]) for t in ts]
            rl = {"R1_5": "R(1〜5)", "R2_5": "R(2〜5)", "R3_5": "R(3〜5)"}[rq]
            ax.plot(ts, _pos(Rm), color=col, marker="o", ms=3, lw=1.2, label=f"{arm} {rl}")
            ax.plot(ts, _pos(Dm), color=col, ls="--", lw=1.0, label=f"{arm} Dn(1〜5)")
        ax.axhline(1, color="0.5", lw=0.8)
        ax.axvspan(1.5, 10.5, color="0.95", lw=0, zorder=0); ax.axvspan(20.5, 30.5, color="0.9", lw=0, zorder=0)
        ax.set_yscale("log"); ax.set_xlabel("課題"); ax.set_title(ACT_JA[act])
        if c == 0: ax.set_ylabel("R・Dn（2 seed の中央値）")
        # 中: S の課題 3 と 25 の時系列（seed 0）
        ax = axes[1, c]
        key = (act, 0) if (act, 0) in runs else ((act, 1) if (act, 1) in runs else None)
        if key is not None:
            S = runs[key][0]["S_series"]
            Snz = runs[key][0].get("S_nz")
            for t, col in ((3, C_T3), (25, C_T25)):
                if t - 1 < S.shape[0]:
                    n = np.arange(1, 2001)
                    ax.plot(n, _pos(S[t - 1, :2000, COL["R"]]), color=col, lw=0.6, label=f"課題 {t} R")
                    Dt = S[t - 1, :2000, COL["D"]]
                    if Snz is not None:
                        nzt = Snz[t - 1, :2000]
                        Dt = np.where(nzt > 0, Dt / np.sqrt(np.where(nzt > 0, nzt, 1.0)), np.nan)
                    ax.plot(n, _pos(Dt), color=col, lw=0.6, ls=":", label=f"課題 {t} Dn")
            ax.set_title(f"{ACT_JA[act]} S（seed {key[1]}）", fontsize=9)
        ax.axhline(1, color="0.5", lw=0.8)
        ax.set_yscale("log"); ax.set_xlim(0, 2000); ax.set_xlabel("課題の中の更新")
        if c == 0: ax.set_ylabel("R・Dn（更新ごと）")
        # 下: Δdead_100 対 R(1〜5)
        ax = axes[2, c]
        for s, fc in ((0, C_S), (1, "none")):
            rr = [r for r in rows if r["act"] == act and r["seed"] == s and r["arm"] == "S" and r["task"] >= 2]
            ax.scatter(_pos([r["R1_5"] for r in rr]), [r["ddead100"] for r in rr], s=18, facecolors=fc, edgecolors=C_S,
                       label=f"seed {s}")
        ax.axvline(1, color="0.5", lw=0.8)
        ax.set_xscale("log"); ax.set_xlabel("S の R(1〜5)")
        if c == 0: ax.set_ylabel("Δdead_100（unit 数）")
    for row in range(3):
        h, l = axes[row, -1].get_legend_handles_labels()
        if h:
            axes[row, -1].legend(h, l, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
    axes[0, -1].text(1.02, 0.02, "薄い灰: 課題 2〜10、濃い灰: 21〜30\nH0 は R(2〜5)、H0A は R(3〜5)", transform=axes[0, -1].transAxes,
                     fontsize=8, va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_fig_vit(path, V):
    plt = _font()
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for mode in VIT_MODES:
        v = V.get(mode)
        if v is None: continue
        axes[0].plot(v["step"], _pos(v["Rlag"]), color=VIT_COL[mode], lw=0.8, label=mode)
        axes[1].plot(v["step"], _pos(v["D"]), color=VIT_COL[mode], lw=0.8, label=mode)
    for ax, lab in zip(axes, ("R_lag = (‖g_n‖/√n_body) / √mean(v_{n−1})", "D = (‖Δθ_body‖/√n_body) / lr")):
        ax.axhline(1, color="0.5", lw=0.8); ax.axvline(25, color="0.5", lw=0.8, ls=":")
        ax.set_yscale("log"); ax.set_ylabel(lab)
    axes[1].set_xlabel("課題 101 の更新（ViT SL、課題 100 の末から）")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
    axes[0].set_title("ViT: R は 1 つ前の更新の後の v で割る R_lag（bias 補正なし、点線 = 更新 25）", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        o = float(o)
        if np.isnan(o):
            return None
        return o if np.isfinite(o) else ("inf" if o > 0 else "-inf")
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def checks_for(d, meta):
    chk = meta.get("check", {}) if isinstance(meta, dict) else {}
    end = np.asarray(d["check_end_maxabs"], dtype=np.float64)
    out = dict(end_max=float(np.nanmax(end)) if np.isfinite(end).any() else float("nan"), n_end=int(np.isfinite(end).sum()),
               torch=meta.get("torch"), git=meta.get("git_head"), elapsed=meta.get("elapsed_s"))
    for k in ("H0_t02_R1", "H0_t02_D1", "H0A_t02_rms_v1", "H0A_t02_D1", "S_t01_D1"):
        if k in chk: out[k] = chk[k]
    if "S_t01_first" in chk: out["S_t01_D1_exp"] = chk["S_t01_first"].get("D1_expected_if_no_state")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/head0_mlp_1108/adamv/raw")
    ap.add_argument("--vit", default=str(VIT_DIR))
    ap.add_argument("--out", default="results/head0_mlp_1108/adamv")
    a = ap.parse_args(argv)
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs, notes = load_raw(raw)
    if not runs:
        raise SystemExit(f"raw が無い: {raw}")
    rows, missing = build_rows(runs)
    order = {x: i for i, x in enumerate(ACTS)}; arm_o = {x: i for i, x in enumerate(ARMS)}
    rows.sort(key=lambda r: (order[r["act"]], r["seed"], r["task"], arm_o[r["arm"]]))
    with open(out / "rows_adamv.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=ROW_KEYS, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (repr(float(r[k])) if isinstance(r.get(k), float) else r.get(k, "")) for k in ROW_KEYS})
    J, checks = {}, {}
    for (act, s), (d, meta) in runs.items():
        J[f"{act}_s{s}"] = judge_run(rows, act, s)
        checks[f"{act}_s{s}"] = checks_for(d, meta)
    act_summary = {}
    for act in ACTS:
        act_summary[act] = {k: seed_agree([J[f"{act}_s{s}"][k]["label"] if f"{act}_s{s}" in J else None for s in SEEDS])
                            for k in ("C1", "C2", "C3", "C4_late", "C4_early", "C5", "premise")}
    V, vnotes = vit_read(Path(a.vit))
    notes += vnotes
    SC = scoring(J, V)
    write_md(out / "judge_adamv.md", J, checks, missing, notes, V, SC, act_summary)
    write_summary(out / "summary_adamv.md", J, V, act_summary)
    Vj = {m: {k: v for k, v in x.items() if k not in ("step", "R", "Rlag", "D", "loss")} for m, x in V.items()}
    (out / "judge_adamv.json").write_text(json.dumps(_jsonable({
        "spec": "specs/spec_head0_mlp_1108.md 追補 C", "early_tasks": EARLY, "late_tasks": LATE,
        "fixed_values": {"J_C4_f_min": F_MIN, "J_C4_n1_max": N1_MAX, "dead_threshold": DEAD},
        "vit_settings": {"n_body": VIT_N_BODY, "lr": VIT_LR, "dir": str(a.vit)},
        "runs": J, "checks": checks, "summary_by_activation": act_summary, "vit": Vj, "scoring": SC,
        "missing": [{"act": x, "seed": s, "arm": arm, "tasks": ts} for x, s, arm, ts in missing], "notes": notes}),
        ensure_ascii=False, indent=1))
    make_fig_mlp(out / "fig_adamv_mlp.png", rows, runs)
    if V:
        make_fig_vit(out / "fig_adamv_vit.png", V)
    for act in ACTS:
        for s in SEEDS:
            j = J.get(f"{act}_s{s}")
            if j is None: continue
            print(f"{act:5s} s{s} C1 {j['C1']['label']:14s} R {fmt(j['C1']['R1_5_med']):>7s} Dn {fmt(j['C1']['Dn1_5_med']):>6s} | "
                  f"C2 {j['C2']['label']:13s} | C3 {j['C3']['label']:10s} | C4 {j['C4_late']['label']:16s} | C5 {j['C5']['label']:22s} | "
                  f"premise {fmt(j['premise']['median'])}")
    print(f"wrote {out}/rows_adamv.csv, judge_adamv.md, judge_adamv.json, summary_adamv.md, fig_adamv_mlp.png, fig_adamv_vit.png")


if __name__ == "__main__":
    main()

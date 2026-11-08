"""head0_cap_1108（spec 追補 A）の集計・判定・図: 残った網の容量を同じ幅の新品と並べる。

実行:
  cd ~/Projects/claude/wt/head0_cap_1108
  PYTHONPATH=. /home/issan/Projects/claude/proj_004_drift/.venv/bin/python -m analysis.head0_mlp_1108.judge_cap \
      [--raw results/head0_mlp_1108/cap/raw] \
      [--prev /home/issan/Projects/obsidian-research-data/head0_mlp_1108/results/head0_mlp_1108/raw] \
      [--out results/head0_mlp_1108/cap]

入力: src/head0_cap_1108.py の raw（{ACT}_s{seed}.npz と同名の .json）と、本体 head0_mlp_1108 の raw（読むだけ）。
欠けた走・課題・k は飛ばし、judge_cap.md の「欠け・注記」に書く。計算はすべて float64。

約束（spec A.3・A.4 の読みで決めたもの。judge_cap.md の冒頭にも書く）:
- 漸近 A_end = 末の 50 評価点（更新 3,510〜4,000）の訓練精度の平均。A1000 = 更新 1,000 の精度。T90 = 精度が初めて
  0.9 以上の更新、届かなければ 10 × 評価点数（401 点なら 4,010）。評価点数は配列から取る。
- k = 0 の点は最頻クラスの割合 maj_t（A_end・A1000 とも）。曲線は k = 0 ∪ 走った k の折れ線。
- k_equiv: 折れ線を k の小さい側から走査し、A_H0A と最初に交わる点。A_H0A < A(0) → 0、A_H0A > 曲線の最大 → ∞。
  傾きはその点を含む区間の両端の差分（端点上なら左の区間。k_equiv = 0 は最初の区間）。δ = 2s/|傾き|、傾き 0 か k_equiv = ∞ → δ = ∞。
- SATURATED: A_H0A ≥ A_fresh(100) − 2s（k = 100 が無ければ走った最大の k）。「中」に数える。判定は SAT を先に見る。
- σ_f²: 一致させた k（k_start・k_end、0 を除く）の (a, b) 対の差の二乗の平均の半分。対が 5 未満なら全ての k の対で。
  σ_H²: (本体 H0A, H0Ab) の同じ量。s² = σ_f²/2 + σ_H²/2。
- ラベルは 10 課題のうち 8 以上。課題が 10 未満なら 80%（切り上げ）。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

ACTS = ["LR", "R", "ELU", "SILU", "GELU"]
ACT_JA = {"LR": "leaky 0.1", "R": "ReLU", "ELU": "ELU", "SILU": "SiLU", "GELU": "GELU"}
SEEDS = [0, 1]
GRID = [3, 6, 13, 25, 45, 74, 100]
CURVE_K = [0] + GRID
EVAL_STEP = 10
N_END = 50
ALIVE, OPEN05 = 0.005, 0.05
MIN_MATCHED_PAIRS = 5
DEFAULT_PREV = "/home/issan/Projects/obsidian-research-data/head0_mlp_1108/results/head0_mlp_1108/raw"

C_VERM, C_BLUE = "#D55E00", "#0072B2"
FRESH_RE = re.compile(r"^fresh_k(\d{3})([ab])_t(\d{2})_eval$")

# ---- 予測（spec A.5、走る前に固定）
SEND_A2 = {a: "WITHIN_005" for a in ACTS}
SEND_POINT = {"LR": (45, 0.95, None), "SILU": (13, 0.5, 0.6), "GELU": (6, 0.3, 0.4)}   # (k, 下限, 上限)
RECV_A1 = {"LR": "CAPACITY_MATCH", "R": "CAPACITY_MATCH", "ELU": "BELOW", "SILU": "CAPACITY_MATCH", "GELU": "CAPACITY_MATCH"}
RECV_A2 = {"LR": "OUTSIDE_005", "R": "WITHIN_005", "ELU": "OUTSIDE_005", "SILU": "WITHIN_005", "GELU": "WITHIN_005"}
RECV_KEQ = {"LR": (70, 100), "R": (0, 0), "ELU": (30, 50), "SILU": (13, 25), "GELU": (5, 12)}
RECV_K90 = (25, 45)
RECV_POINT = {"LR": (45, 0.90, 0.97), "SILU": (13, 0.45, 0.6), "GELU": (6, 0.25, 0.4)}


# ---------------------------------------------------------------- 小道具
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
    if not np.isfinite(x):
        return "—" if np.isnan(x) else ("∞" if x > 0 else "−∞")
    if x == 0:
        return "0"
    if abs(x) < 10 ** (-nd) or abs(x) >= 1e5:
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def fmt_k(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if np.isinf(x):
        return "> 100"
    return f"{x:.1f}"


def nanmean(v):
    v = np.asarray([x for x in v if x is not None], dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        o = float(o)
        if np.isnan(o):
            return None
        return o if np.isfinite(o) else ("inf" if o > 0 else "-inf")
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def eval_metrics(ev):
    ev = np.asarray(ev, dtype=np.float64)
    n = ev.shape[0]
    acc, loss = ev[:, 0], ev[:, 1]
    k = min(N_END, n)
    hit = np.nonzero(acc >= 0.9)[0]
    return {"n_eval": n, "A_end": float(acc[-k:].mean()), "L_end": float(loss[-k:].mean()),
            "A1000": float(acc[100]) if n > 100 else float("nan"),
            "T90": int(hit[0] * EVAL_STEP) if hit.size else int(EVAL_STEP * n),
            "T90_cens": 0 if hit.size else 1}


def unit_metrics(unit, head=None):
    out = {}
    if unit is not None:
        u = np.asarray(unit, dtype=np.float64)
        out.update(pplus_start=float(u[0, 1].mean()), pplus_end=float(u[1, 1].mean()),
                   dead_end=float((u[1, 1] < ALIVE).mean()),
                   k_alive_end=int((u[1, 1] >= ALIVE).sum()))
    if head is not None:
        h = np.asarray(head, dtype=np.float64).ravel()
        out.update(head_start=float(h[0]), head_end=float(h[-1]))
    return out


# ---------------------------------------------------------------- 読み込み
def load_npz(f):
    z = np.load(f, allow_pickle=False)
    return {k: z[k] for k in z.files}


def load_runs(raw: Path, prev: Path):
    runs, notes = {}, []
    for act in ACTS:
        for s in SEEDS:
            f = raw / f"{act}_s{s}.npz"
            if not f.exists():
                notes.append(f"{f.name} が無い（新しい raw）")
                continue
            d = load_npz(f)
            meta = {}
            js = raw / f"{act}_s{s}.json"
            if js.exists():
                try:
                    meta = json.loads(js.read_text())
                except Exception as e:  # noqa: BLE001
                    notes.append(f"{js.name} を読めない: {e}")
            else:
                notes.append(f"{js.name} が無い")
            pf = prev / f"{act}_s{s}.npz"
            p = None
            if pf.exists():
                p = load_npz(pf)
            else:
                notes.append(f"本体 raw {pf} が無い（A_H0A は H0Ab だけ・p⁺ の記録なし）")
            runs[(act, s)] = {"d": d, "meta": meta, "prev": p}
    return runs, notes


def tasks_of(d):
    if "tasks" in d and np.size(d["tasks"]):
        return [int(t) for t in np.ravel(d["tasks"])]
    ts = set()
    for k in d:
        m = re.match(r"^H0Ab_t(\d{2})_eval$", k) or FRESH_RE.match(k)
        if m:
            ts.add(int(m.groups()[-1]))
    return sorted(ts)


# ---------------------------------------------------------------- 曲線
def curve_points(fresh_mean, maj):
    """[(k, A)] を k で並べる。k = 0 は maj。"""
    pts = {0: maj}
    for k, v in fresh_mean.items():
        if k != 0 and np.isfinite(v):
            pts[k] = v
    ks = sorted(pts)
    return np.array(ks, dtype=np.float64), np.array([pts[k] for k in ks], dtype=np.float64)


def interp(ks, As, k):
    if k is None or not np.isfinite(k) or ks.size == 0:
        return float("nan")
    return float(np.interp(k, ks, As))


def k_equiv(ks, As, target):
    """戻り値: (k_equiv, 区間の傾き, 区間 (k_lo, k_hi), 注記, 交点の数)。"""
    info = {"nonmonotone": bool(np.any(np.diff(As) < 0)), "n_cross": 0}
    if not np.isfinite(target) or ks.size < 2:
        return float("nan"), float("nan"), None, info
    slopes = np.diff(As) / np.diff(ks)
    if target < As[0]:
        info["case"] = "below_k0"
        return 0.0, float(slopes[0]), (ks[0], ks[1]), info
    if target > As.max():
        info["case"] = "above_max"
        return float("inf"), float("nan"), None, info
    found = None
    for i in range(ks.size - 1):
        a0, a1 = As[i], As[i + 1]
        if min(a0, a1) <= target <= max(a0, a1):
            if a1 == a0:
                kx = float(ks[i])
            else:
                kx = float(ks[i] + (target - a0) * (ks[i + 1] - ks[i]) / (a1 - a0))
            # 区間の端点の重複（次の区間の始点と同じ交点）は数えない
            if found is not None and kx == found[0]:
                continue
            info["n_cross"] += 1
            if found is None:
                found = (kx, float(slopes[i]), (float(ks[i]), float(ks[i + 1])))
    info["case"] = "crossing"
    return found[0], found[1], found[2], info


# ---------------------------------------------------------------- 集計
def build(runs):
    rows, R = [], {}
    for (act, s), run in runs.items():
        d, p = run["d"], run["prev"]
        tasks = tasks_of(d)
        ksa = [int(x) for x in np.ravel(d.get("k_start", []))]
        kea = [int(x) for x in np.ravel(d.get("k_end", []))]
        TT = {}
        for i, t in enumerate(tasks):
            T = {"task": t, "notes": []}
            T["k_start"] = ksa[i] if i < len(ksa) else None
            T["k_end"] = kea[i] if i < len(kea) else None
            T["maj"] = float(d[f"maj_t{t:02d}"]) if f"maj_t{t:02d}" in d else float("nan")
            # 新品
            fresh = {}
            for key in d:
                m = FRESH_RE.match(key)
                if not m or int(m.group(3)) != t:
                    continue
                k, rep = int(m.group(1)), m.group(2)
                r = eval_metrics(d[key])
                pre = key[:-5]
                r.update(unit_metrics(d.get(pre + "_unit"), d.get(pre + "_head")))
                fresh.setdefault(k, {})[rep] = r
                rows.append({"act": act, "seed": s, "task": t, "k": k, "rep": rep, **r})
            T["fresh"] = fresh
            for q in ("A_end", "A1000"):
                T[f"{q}_fresh_mean"] = {k: nanmean([fresh[k][r][q] for r in fresh[k]]) for k in fresh}
            # H0A（本体）と H0Ab（新）
            hb = eval_metrics(d[f"H0Ab_t{t:02d}_eval"]) if f"H0Ab_t{t:02d}_eval" in d else None
            if hb is not None:
                hb.update(unit_metrics(d.get(f"H0Ab_t{t:02d}_unit")))
                rows.append({"act": act, "seed": s, "task": t, "k": "H0Ab", "rep": "b", **hb})
            ha = None
            ref = {}
            if p is not None:
                if f"H0A_t{t:02d}_eval" in p:
                    ha = eval_metrics(p[f"H0A_t{t:02d}_eval"])
                    ha.update(unit_metrics(p.get(f"H0A_t{t:02d}_unit"), p.get(f"H0A_t{t:02d}_head")))
                    rows.append({"act": act, "seed": s, "task": t, "k": "H0A_prev", "rep": "a", **ha})
                for arm in ("H0", "fa", "fb", "S"):
                    if f"{arm}_t{t:02d}_eval" in p:
                        ref[f"A_end_{arm}_prev"] = eval_metrics(p[f"{arm}_t{t:02d}_eval"])["A_end"]
                # p⁺ の記録（課題の頭は S と H0A で同じ体。H0A の頭が無ければ S の頭）
                ua = p.get(f"H0A_t{t:02d}_unit")
                us = p.get(f"S_t{t:02d}_unit")
                u0 = ua if ua is not None else us
                if u0 is not None:
                    u0 = np.asarray(u0, dtype=np.float64)
                    ref["k05_start"] = int((u0[0, 1] >= OPEN05).sum())
                    ref["sum_pplus_start"] = float(u0[0, 1].sum())
                    ref["k005_start_prev"] = int((u0[0, 1] >= ALIVE).sum())
                if ua is not None:
                    ua = np.asarray(ua, dtype=np.float64)
                    ref["k05_end_H0A"] = int((ua[1, 1] >= OPEN05).sum())
                    ref["sum_pplus_end_H0A"] = float(ua[1, 1].sum())
                    ref["k005_end_prev"] = int((ua[1, 1] >= ALIVE).sum())
            T["H0Ab"], T["H0A_prev"], T["ref"] = hb, ha, ref
            for q in ("A_end", "A1000"):
                T[f"{q}_H0A"] = nanmean([x[q] for x in (ha, hb) if x is not None])
            if ha is None:
                T["notes"].append("本体の H0A が無い（H0Ab だけで A_H0A）")
            if hb is None:
                T["notes"].append("H0Ab が無い（本体 H0A だけで A_H0A）")
            # 照合: npz の k_start/k_end と本体 raw から数え直した値
            for a_, b_ in (("k_start", "k005_start_prev"), ("k_end", "k005_end_prev")):
                if T[a_] is not None and b_ in ref and T[a_] != ref[b_]:
                    T["notes"].append(f"{a_}={T[a_]} が本体 raw からの数え直し {ref[b_]} と違う")
            ks, As = curve_points(T["A_end_fresh_mean"], T["maj"])
            T["curve_k"], T["curve_A"] = ks, As
            T["A_fresh_kstart"] = interp(ks, As, T["k_start"])
            T["A_fresh_kend"] = interp(ks, As, T["k_end"])
            if T["k_start"] is not None and T["k_start"] != 0 and T["k_start"] not in T["A_end_fresh_mean"]:
                T["notes"].append("k_start の新品が無いので折れ線で補間")
            kmax_ref = 100 if 100 in T["A_end_fresh_mean"] else (int(ks[-1]) if ks.size else None)
            T["k_ref_sat"] = kmax_ref
            T["A_fresh_100"] = interp(ks, As, kmax_ref) if kmax_ref is not None else float("nan")
            if kmax_ref is not None and kmax_ref != 100:
                T["notes"].append(f"k = 100 が無いので SAT は k = {kmax_ref} で")
            ke, slope, seg, info = k_equiv(ks, As, T["A_end_H0A"])
            T.update(k_equiv=ke, slope=slope, segment=seg, keq_info=info)
            if info.get("nonmonotone"):
                if info.get("case") == "crossing":
                    T["notes"].append(f"曲線が単調でない（交点 {info.get('n_cross', 0)} 個、最小の k を採用）")
                else:
                    T["notes"].append(f"曲線が単調でない（交点なし: {'A_H0A が曲線の最大を超える → > 100' if info.get('case') == 'above_max' else 'A_H0A < A(0) → 0'}）")
            TT[t] = T
        R[(act, s)] = {"tasks": TT, "order": tasks}
    return rows, R


def noise(Rrun):
    TT = Rrun["tasks"]
    m_pairs, all_pairs, h_pairs = [], [], []
    for t, T in TT.items():
        fr = T["fresh"]
        matched = {k for k in (T["k_start"], T["k_end"]) if k is not None and k != 0}
        for k, reps in fr.items():
            if "a" in reps and "b" in reps:
                dd = reps["a"]["A_end"] - reps["b"]["A_end"]
                all_pairs.append(dd)
                if k in matched:
                    m_pairs.append(dd)
        if T["H0A_prev"] is not None and T["H0Ab"] is not None:
            h_pairs.append(T["H0A_prev"]["A_end"] - T["H0Ab"]["A_end"])
    use_all = len(m_pairs) < MIN_MATCHED_PAIRS
    fp = np.array(all_pairs if use_all else m_pairs, dtype=np.float64)
    hp = np.array(h_pairs, dtype=np.float64)
    sf2 = float(np.mean(fp ** 2) / 2) if fp.size else float("nan")
    sh2 = float(np.mean(hp ** 2) / 2) if hp.size else float("nan")
    parts = [x for x in (sf2, sh2) if np.isfinite(x)]
    s2 = (sf2 / 2 if np.isfinite(sf2) else 0.0) + (sh2 / 2 if np.isfinite(sh2) else 0.0) if parts else float("nan")
    return {"sigma_f2": sf2, "sigma_H2": sh2, "s2": s2, "sigma_f": math.sqrt(sf2) if np.isfinite(sf2) else float("nan"),
            "sigma_H": math.sqrt(sh2) if np.isfinite(sh2) else float("nan"), "s": math.sqrt(s2) if np.isfinite(s2) else float("nan"),
            "n_f_pairs": int(fp.size), "f_pairs_source": "全ての k の対" if use_all else "一致させた k（k_start・k_end）の対",
            "n_matched_pairs": len(m_pairs), "n_H_pairs": int(hp.size),
            "sigma_H_missing": not np.isfinite(sh2), "sigma_f_missing": not np.isfinite(sf2)}


def need_count(n):
    return 8 if n == 10 else int(math.ceil(0.8 * n))


def judge(R):
    J = {}
    for (act, s), Rrun in R.items():
        nz = noise(Rrun)
        sc = nz["s"]
        per = {}
        for t in Rrun["order"]:
            T = Rrun["tasks"][t]
            ke, sl = T["k_equiv"], T["slope"]
            if not np.isfinite(ke) or not np.isfinite(sl) or sl == 0 or not np.isfinite(sc):
                delta = float("inf")
            else:
                delta = 2 * sc / abs(sl)
            T["delta"] = delta
            A, Af100 = T["A_end_H0A"], T["A_fresh_100"]
            sat = bool(np.isfinite(A) and np.isfinite(Af100) and np.isfinite(sc) and A >= Af100 - 2 * sc)
            ks_, ke_ = T["k_start"], T["k_end"]
            if np.isnan(ke) or ks_ is None or ke_ is None:
                lab = "—"
            elif sat:
                lab = "SAT"
            elif ke > ke_ + delta:
                lab = "上"
            elif ke < ks_ - delta:
                lab = "下"
            else:
                lab = "中"
            T["A1"] = lab
            T["delta_inf_mid"] = bool(lab == "中" and np.isinf(delta))
            dA = A - T["A_fresh_kstart"]
            T["dA_kstart"] = dA
            T["A2_within"] = bool(np.isfinite(dA) and abs(dA) <= 0.05) if np.isfinite(dA) else None
            per[t] = lab
        n = len([l for l in per.values() if l != "—"])
        cnt = {k: sum(1 for l in per.values() if l == k) for k in ("中", "上", "下", "SAT")}
        need = need_count(n) if n else None
        mid = cnt["中"] + cnt["SAT"]
        if not n:
            l1 = "NOT_AVAILABLE"
        elif mid >= need:
            l1 = "CAPACITY_MATCH"
        elif cnt["上"] >= need:
            l1 = "ABOVE"
        elif cnt["下"] >= need:
            l1 = "BELOW"
        else:
            l1 = "MIXED"
        a2 = [Rrun["tasks"][t]["A2_within"] for t in Rrun["order"] if Rrun["tasks"][t]["A2_within"] is not None]
        n2 = len(a2)
        l2 = "NOT_AVAILABLE" if not n2 else ("WITHIN_005" if sum(a2) >= need_count(n2) else "OUTSIDE_005")
        J[f"{act}_s{s}"] = {"act": act, "seed": s, "noise": nz, "counts": cnt, "n_tasks": n, "need": need,
                            "A1": l1, "A2": l2, "A2_within_count": int(sum(a2)), "n_A2": n2,
                            "under10_rule": bool(n and n < 10)}
    return J


def records(R, J):
    """k90・A_fresh at 6/13/45・曲線の格子表の値。"""
    for (act, s), Rrun in R.items():
        TT = Rrun["tasks"]
        grid_present = sorted({k for T in TT.values() for k in T["A_end_fresh_mean"] if k in GRID})
        mean_at = {}
        for k in grid_present:
            mean_at[k] = nanmean([T["A_end_fresh_mean"].get(k) for T in TT.values()])
        k90 = next((k for k in grid_present if mean_at[k] >= 0.9), None)
        k90_task = {}
        for t, T in TT.items():
            k90_task[t] = next((k for k in sorted(T["A_end_fresh_mean"]) if k in GRID and T["A_end_fresh_mean"][k] >= 0.9), None)
        rec = J[f"{act}_s{s}"]
        rec["k90"] = k90
        rec["k90_per_task"] = k90_task
        rec["A_fresh_grid_mean"] = mean_at
        rec["A_fresh_at"] = {k: mean_at.get(k, float("nan")) for k in (6, 13, 45)}
        keqs = [TT[t]["k_equiv"] for t in Rrun["order"]]
        keqs = [x for x in keqs if not np.isnan(x)]
        rec["k_equiv_median"] = float(np.median(keqs)) if keqs else float("nan")


# ---------------------------------------------------------------- 検査
def checks_for(run):
    d, meta = run["d"], run["meta"]
    chk = meta.get("check", {}) if isinstance(meta, dict) else {}
    out = []

    def mx(v):
        if v is None:
            return float("nan")
        a = np.asarray([np.nan if x is None else x for x in np.ravel(v)], dtype=np.float64)
        return float(np.nanmax(a)) if np.isfinite(a).any() else float("nan")

    def nfin(v):
        if v is None:
            return 0
        a = np.asarray([np.nan if x is None else x for x in np.ravel(v)], dtype=np.float64)
        return int(np.isfinite(a).sum())

    end = d.get("check_end_maxabs", chk.get("check_end_maxabs"))
    pp = d.get("check_pplus_start_maxabs", chk.get("check_pplus_start_maxabs"))
    init = chk.get("init_maxabs")
    lab = chk.get("labels_equal_saved")
    out.append(("C1 本線 S の課題末 θ と保存済み end の差の絶対値の最大", mx(end), "== 0", nfin(end)))
    out.append(("課題の頭の p⁺ と本体 S の頭の p⁺ の差の絶対値の最大", mx(pp), "== 0", nfin(pp)))
    out.append(("engine の初期値との差の絶対値の最大", float(init) if isinstance(init, (int, float)) else float("nan"), "== 0", 1 if init is not None else 0))
    table = []
    for name, v, cond, n in out:
        ok = None if not np.isfinite(v) else (v == 0)
        table.append({"name": name, "value": v, "condition": cond, "n": n, "pass": ok})
    table.append({"name": "ラベルが保存済みと一致", "value": lab, "condition": "True", "n": 1 if lab is not None else 0,
                  "pass": None if lab is None else bool(lab)})
    info = {k: meta.get(k) for k in ("torch", "git_head", "smoke", "steps", "tasks_from", "tasks_to", "grid", "elapsed_s")
            if isinstance(meta, dict)}
    return {"table": table, "info": info}


# ---------------------------------------------------------------- 出力
def score(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return "—"
    tag = "" if len(v) == 2 else "（1 seed のみ）"
    if all(v):
        return "当たり" + tag
    if not any(v):
        return "外れ" + tag
    return "割れ"


def seed_agree(labels):
    labs = [l for l in labels if l is not None]
    if len(labs) < 2:
        return f"未確定（seed 欠け: {labs[0] if labs else '—'}）"
    return f"確定 {labs[0]}" if all(l == labs[0] for l in labs) else "割れ"


def t90_cell(vals, cens):
    if not vals:
        return "—"
    med = float(np.median(vals))
    nc, n = int(sum(cens)), len(vals)
    cens_val = max(vals) if nc else None
    if nc and cens_val is not None and med >= cens_val:
        return f"届かず（{nc}/{n}）"
    return f"{med:,.0f}（届かず {nc}/{n}）"


def write_curve(path, R, J):
    L = ["# 新品 784-k-10 の曲線（課題 21〜30・2 本の平均、spec 追補 A）\n",
         "A_end は更新 3,510〜4,000 の訓練精度の平均。課題と取り直し a・b の平均。k = 0 は最頻クラスの割合（走らない）。"
         "T90 は課題 × 取り直しの中央値（届かない走は 4,010 と置いて中央値に入れる）。\n"]
    for act in ACTS:
        if not any((act, s) in R for s in SEEDS):
            continue
        L.append(f"## {ACT_JA[act]}\n")
        L.append("| k | A_end seed 0 | A_end seed 1 | A1000 seed 0 | A1000 seed 1 | T90 seed 0 | T90 seed 1 |")
        L.append("|---|---|---|---|---|---|---|")
        for k in CURVE_K:
            cells = {"A_end": [], "A1000": [], "T90": []}
            for s in SEEDS:
                Rr = R.get((act, s))
                if Rr is None:
                    for q in cells:
                        cells[q].append("—")
                    continue
                TT = Rr["tasks"]
                if k == 0:
                    m = nanmean([T["maj"] for T in TT.values()])
                    cells["A_end"].append(fmt(m))
                    cells["A1000"].append(fmt(m))
                    cells["T90"].append("—（走らない）")
                    continue
                runs_k = [r for T in TT.values() for r in T["fresh"].get(k, {}).values()]
                cells["A_end"].append(fmt(nanmean([r["A_end"] for r in runs_k])) if runs_k else "—")
                cells["A1000"].append(fmt(nanmean([r["A1000"] for r in runs_k])) if runs_k else "—")
                cells["T90"].append(t90_cell([r["T90"] for r in runs_k], [r["T90_cens"] for r in runs_k]))
            L.append(f"| {k} | " + " | ".join(cells["A_end"] + cells["A1000"] + cells["T90"]) + " |")
        k90s = [J[f"{act}_s{s}"]["k90"] if f"{act}_s{s}" in J else None for s in SEEDS]
        L.append("\nk90（課題平均の A_end ≥ 0.9 になる最小の格子 k）: "
                 + "、".join(f"seed {s} = {('> 100' if v is None else v) if f'{act}_s{s}' in J else '—'}" for s, v in zip(SEEDS, k90s)) + "\n")
    Path(path).write_text("\n".join(L) + "\n")


def write_md(path, R, J, checks, notes):
    L = []
    w = L.append
    w("# head0_cap_1108 判定（spec 追補 A.4）\n")
    w("残った網（本体 H0A：体の重みだけ引き継ぎ、head と Adam を 0）の漸近を、同じ幅 k の新品 784-k-10 の漸近と並べる。"
      "課題 21〜30。数は表に置き、読みは書かない。\n")
    w("約束: A_H0A は本体 H0A と H0Ab の平均。曲線は k = 0（最頻クラスの割合）と走った k の折れ線。k_equiv は折れ線が A_H0A と"
      "最初に交わる k（A_H0A < A(0) なら 0、曲線の最大を超えれば > 100）。δ = 2s / |k_equiv を含む区間の傾き|。"
      "SAT は A_H0A ≥ A_fresh(100) − 2s で「中」に数える。判は SAT を先に見る。|Δ| = |A_H0A − A_fresh(k_start)|。\n")
    n_eval = sorted({r["n_eval"] for Rr in R.values() for T in Rr["tasks"].values() for k in T["fresh"].values() for r in k.values()})
    w(f"評価点の数: {n_eval}（401 点が本測定）。\n")

    w("## 検査\n")
    w("| 活性化 | seed | 項目 | 値 | 条件 | 数 | 合否 |\n|---|---|---|---|---|---|---|")
    for act in ACTS:
        for s in SEEDS:
            c = checks.get(f"{act}_s{s}")
            if c is None:
                continue
            for row in c["table"]:
                ok = "—" if row["pass"] is None else ("通過" if row["pass"] else "不合格")
                w(f"| {ACT_JA[act]} | {s} | {row['name']} | {fmt(row['value'], 4)} | {row['condition']} | {row['n']} | {ok} |")
    w("\n| 活性化 | seed | 課題 | torch | git | smoke |\n|---|---|---|---|---|---|")
    for act in ACTS:
        for s in SEEDS:
            c = checks.get(f"{act}_s{s}")
            if c is None:
                continue
            i = c["info"]
            w(f"| {ACT_JA[act]} | {s} | {i.get('tasks_from', '—')}〜{i.get('tasks_to', '—')} | {i.get('torch', '—')} | "
              f"{str(i.get('git_head', '—'))[:10]} | {i.get('smoke', '—')} |")
    w("")

    w("## 課題ごと\n")
    w("k(p⁺≥0.05)・Σp⁺ は本体 raw の課題の頭（左）と H0A の末（右）。判: 中・上・下・SAT。\n")
    for act in ACTS:
        for s in SEEDS:
            Rr = R.get((act, s))
            if Rr is None:
                continue
            w(f"### {ACT_JA[act]} seed {s}\n")
            w("| 課題 | k_start | k_end | k(p⁺≥0.05) 頭/末 | Σp⁺ 頭/末 | A_H0A | H0A / H0Ab | A_fresh(k_start) | A_fresh(k_end) | A_fresh(100) | k_equiv | δ | 判 | |Δ| at k_start |")
            w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for t in Rr["order"]:
                T = Rr["tasks"][t]
                rf = T["ref"]
                hp = T["H0A_prev"]["A_end"] if T["H0A_prev"] else float("nan")
                hb = T["H0Ab"]["A_end"] if T["H0Ab"] else float("nan")
                w(f"| {t} | {fmt(T['k_start'])} | {fmt(T['k_end'])} | {fmt(rf.get('k05_start'))} / {fmt(rf.get('k05_end_H0A'))} | "
                  f"{fmt(rf.get('sum_pplus_start'), 2)} / {fmt(rf.get('sum_pplus_end_H0A'), 2)} | {fmt(T['A_end_H0A'])} | "
                  f"{fmt(hp)} / {fmt(hb)} | {fmt(T['A_fresh_kstart'])} | {fmt(T['A_fresh_kend'])} | {fmt(T['A_fresh_100'])} | "
                  f"{fmt_k(T['k_equiv'])} | {fmt_k(T['delta'])} | {T['A1']} | {fmt(abs(T['dA_kstart']) if np.isfinite(T['dA_kstart']) else float('nan'))} |")
            w("")

    w("## 活性化 × seed のまとめ\n")
    w("| 活性化 | seed | 課題数 | 中 | 上 | 下 | SAT | 中（δ = ∞） | 必要数 | J A1 | |Δ| ≤ 0.05 の数 | J A2 | σ_f | σ_H | s | σ_f の対 | k_equiv 中央値 | k90 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for act in ACTS:
        for s in SEEDS:
            r = J.get(f"{act}_s{s}")
            if r is None:
                continue
            c, nz = r["counts"], r["noise"]
            ninf = sum(1 for T in R[(act, s)]["tasks"].values() if T.get("delta_inf_mid"))
            w(f"| {ACT_JA[act]} | {s} | {r['n_tasks']} | {c['中']} | {c['上']} | {c['下']} | {c['SAT']} | {ninf} | "
              f"{r['need']}{'（80%）' if r['under10_rule'] else ''} | {r['A1']} | {r['A2_within_count']}/{r['n_A2']} | {r['A2']} | "
              f"{fmt(nz['sigma_f'], 4)} | {fmt(nz['sigma_H'], 4)} | {fmt(nz['s'], 4)} | {nz['f_pairs_source']}（{nz['n_f_pairs']}） | "
              f"{fmt_k(r['k_equiv_median'])} | {('> 100' if r['k90'] is None else r['k90'])} |")
    w("")
    w("## 活性化の判定（2 seed が同じときだけ確定）\n")
    w("| 活性化 | J A1 seed 0 | J A1 seed 1 | J A1 | J A2 seed 0 | J A2 seed 1 | J A2 |\n|---|---|---|---|---|---|---|")
    summary = {}
    for act in ACTS:
        l1 = [J[f"{act}_s{s}"]["A1"] if f"{act}_s{s}" in J else None for s in SEEDS]
        l2 = [J[f"{act}_s{s}"]["A2"] if f"{act}_s{s}" in J else None for s in SEEDS]
        if all(x is None for x in l1):
            continue
        summary[act] = {"A1": seed_agree(l1), "A2": seed_agree(l2)}
        w(f"| {ACT_JA[act]} | {l1[0] or '—'} | {l1[1] or '—'} | {summary[act]['A1']} | {l2[0] or '—'} | {l2[1] or '—'} | {summary[act]['A2']} |")
    w("")

    # ---- 予測の採点
    w("## 予測の採点（spec A.5）\n")
    w("seed ごとに当否を出し、2 seed とも当たり → 当たり、2 seed とも外れ → 外れ、それ以外 → 割れ。"
      "A_fresh(k) は課題 21〜30 と取り直し 2 本の平均。k_equiv の範囲は課題 21〜30 の k_equiv の中央値で採点する"
      "（> 100 は範囲の外として数える）。範囲は両端を含む。点の予測（ReLU の k_equiv = 0）は δ の中央値まで許す。\n")
    scores = {"sender": [], "receiver": []}

    def get(act, s):
        return J.get(f"{act}_s{s}")

    def line(who, act, name, pred, vals, hits):
        sc = score(hits)
        scores[who].append({"act": act, "item": name, "pred": pred, "values": vals, "hits": hits, "score": sc})
        w(f"| {ACT_JA[act]} | {name} | {pred} | {vals[0]} | {vals[1]} | {sc} |")

    w("### 送り手\n")
    w("| 活性化 | 量 | 予測 | seed 0 | seed 1 | 採点 |\n|---|---|---|---|---|---|")
    for act in ACTS:
        if not any(get(act, s) for s in SEEDS):
            continue
        vals, hits = [], []
        for s in SEEDS:
            r = get(act, s)
            vals.append(f"{r['A2']}（{r['A2_within_count']}/{r['n_A2']}）" if r else "—")
            hits.append(None if r is None or r["A2"] == "NOT_AVAILABLE" else r["A2"] == SEND_A2[act])
        line("sender", act, "J A2（|Δ| ≤ 0.05）", SEND_A2[act], vals, hits)
        if act in SEND_POINT:
            k, lo, hi = SEND_POINT[act]
            vals, hits = [], []
            for s in SEEDS:
                r = get(act, s)
                v = r["A_fresh_at"].get(k) if r else None
                vals.append(fmt(v))
                hits.append(None if v is None or not np.isfinite(v) else (v >= lo and (hi is None or v <= hi)))
            line("sender", act, f"A_fresh(k = {k})", f"≥ {lo}" if hi is None else f"{lo}〜{hi}", vals, hits)
    w("")
    w("### 受け手\n")
    w("| 活性化 | 量 | 予測 | seed 0 | seed 1 | 採点 |\n|---|---|---|---|---|---|")
    for act in ACTS:
        if not any(get(act, s) for s in SEEDS):
            continue
        for key, pred_d, name in (("A1", RECV_A1, "J A1"), ("A2", RECV_A2, "J A2")):
            vals = [get(act, s)[key] if get(act, s) else "—" for s in SEEDS]
            hits = [None if get(act, s) is None or get(act, s)[key] == "NOT_AVAILABLE" else get(act, s)[key] == pred_d[act] for s in SEEDS]
            line("receiver", act, name, pred_d[act], vals, hits)
        lo, hi = RECV_KEQ[act]
        vals, hits = [], []
        for s in SEEDS:
            r = get(act, s)
            v = r["k_equiv_median"] if r else None
            tol = 0.0
            if r is not None and lo == hi:
                # 点の予測（ReLU の 0）は幅の揺れ δ の中央値まで許す
                ds = [T["delta"] for T in R[(act, s)]["tasks"].values() if np.isfinite(T.get("delta", np.nan))]
                tol = float(np.median(ds)) if ds else 0.0
            vals.append((fmt_k(v) + (f"（許容 ±{tol:.2f}）" if lo == hi else "")) if r else "—")
            hits.append(None if v is None or np.isnan(v) else (lo - tol <= v <= hi + tol))
        line("receiver", act, "k_equiv（中央値）", f"{lo}〜{hi}", vals, hits)
        vals, hits = [], []
        for s in SEEDS:
            r = get(act, s)
            if r is None:
                vals.append("—"); hits.append(None); continue
            v = r["k90"]
            vals.append("> 100" if v is None else str(v))
            hits.append(v is not None and RECV_K90[0] <= v <= RECV_K90[1])
        line("receiver", act, "k90", f"{RECV_K90[0]}〜{RECV_K90[1]}", vals, hits)
        if act in RECV_POINT:
            k, lo, hi = RECV_POINT[act]
            vals, hits = [], []
            for s in SEEDS:
                r = get(act, s)
                v = r["A_fresh_at"].get(k) if r else None
                vals.append(fmt(v))
                hits.append(None if v is None or not np.isfinite(v) else (lo <= v <= hi))
            line("receiver", act, f"A_fresh(k = {k})", f"{lo}〜{hi}", vals, hits)
    w("")
    w("## 欠け・注記\n")
    any_note = False
    for n in notes:
        w(f"- {n}")
        any_note = True
    for (act, s), Rr in R.items():
        for t in Rr["order"]:
            for n in Rr["tasks"][t]["notes"]:
                w(f"- {ACT_JA[act]} seed {s} 課題 {t}: {n}")
                any_note = True
        nz = J[f"{act}_s{s}"]["noise"]
        if nz["f_pairs_source"].startswith("全"):
            w(f"- {ACT_JA[act]} seed {s}: 一致させた k の対が {nz['n_matched_pairs']} 個（< {MIN_MATCHED_PAIRS}）なので σ_f は全ての k の対で")
            any_note = True
        if nz["sigma_H_missing"]:
            w(f"- {ACT_JA[act]} seed {s}: σ_H を出せない（H0A と H0Ab の対が無い）。s は σ_f だけ")
            any_note = True
    if not any_note:
        w("- なし")
    w("")
    Path(path).write_text("\n".join(L) + "\n")
    return summary, scores


def make_fig(path, R, J):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    names = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = ["Noto Sans CJK JP"] if "Noto Sans CJK JP" in names else ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2), sharey=True)
    for j, act in enumerate(ACTS):
        ax = axes[j]
        ax.axhline(0.9, color="0.4", lw=0.8, ls=":", zorder=1)
        allk = {}
        for s in SEEDS:
            Rr = R.get((act, s))
            if Rr is None:
                continue
            for t in Rr["order"]:
                T = Rr["tasks"][t]
                ax.plot(T["curve_k"], T["curve_A"], color="0.75", lw=0.6, zorder=2)
                for k, reps in T["fresh"].items():
                    for r in reps.values():
                        ax.plot(k, r["A_end"], ".", color="0.7", ms=2.5, zorder=2)
                allk.setdefault(0, []).append(T["maj"])
                for k, v in T["A_end_fresh_mean"].items():
                    if k in GRID:
                        allk.setdefault(k, []).append(v)
        if allk:
            kk = sorted(allk)
            ax.plot(kk, [nanmean(allk[k]) for k in kk], "-o", color="black", lw=1.8, ms=4, zorder=4)
        for s, ls in zip(SEEDS, ("-", "--")):
            Rr = R.get((act, s))
            if Rr is None:
                continue
            for t in Rr["order"]:
                T = Rr["tasks"][t]
                if T["k_start"] is None or not np.isfinite(T["A_end_H0A"]):
                    continue
                x0, x1 = T["k_start"], T["k_end"]
                if x0 == x1:
                    x0, x1 = x0 - 0.8, x1 + 0.8
                ax.plot([x0, x1], [T["A_end_H0A"]] * 2, color=C_VERM, lw=1.2, ls=ls, alpha=0.85, zorder=5)
                if np.isfinite(T["k_equiv"]):
                    ax.plot(T["k_equiv"], T["A_end_H0A"], "o", color=C_BLUE, ms=3.5, zorder=6)
        ax.set_xlim(-3, 103)
        ax.set_ylim(0, 1.02)
        ax.set_title(ACT_JA[act])
        ax.set_xlabel("幅 k")
        ax.grid(alpha=0.25, lw=0.5)
        if j == 0:
            ax.set_ylabel("漸近の訓練精度")
    handles = [
        Line2D([], [], color="0.75", lw=0.8, marker=".", ms=4, mfc="0.7", mec="0.7", label="新品 784-k-10（課題ごと、点は取り直し a・b）"),
        Line2D([], [], color="black", lw=1.8, marker="o", ms=4, label="新品（課題 21〜30 の平均）"),
        Line2D([], [], color=C_VERM, lw=1.2, ls="-", label="H0A の漸近（k_start〜k_end、seed 0）"),
        Line2D([], [], color=C_VERM, lw=1.2, ls="--", label="H0A の漸近（seed 1）"),
        Line2D([], [], color=C_BLUE, ls="none", marker="o", ms=4, label="同じ幅 k_equiv"),
        Line2D([], [], color="0.4", lw=0.8, ls=":", label="精度 0.9"),
    ]
    axes[-1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/head0_mlp_1108/cap/raw")
    ap.add_argument("--prev", default=DEFAULT_PREV)
    ap.add_argument("--out", default="results/head0_mlp_1108/cap")
    a = ap.parse_args(argv)
    raw, prev, out = Path(a.raw), Path(a.prev), Path(a.out)
    runs, notes = load_runs(raw, prev)
    if not runs:
        raise SystemExit(f"raw が無い: {raw}")
    out.mkdir(parents=True, exist_ok=True)
    rows, R = build(runs)
    J = judge(R)
    records(R, J)
    checks = {f"{act}_s{s}": checks_for(run) for (act, s), run in runs.items()}

    keys = ["act", "seed", "task", "k", "rep", "A_end", "L_end", "A1000", "T90", "T90_cens", "pplus_start", "pplus_end",
            "dead_end", "k_alive_end", "head_start", "head_end", "n_eval"]
    order = {x: i for i, x in enumerate(ACTS)}
    rows.sort(key=lambda r: (order[r["act"]], r["seed"], r["task"], (0, r["k"]) if isinstance(r["k"], int) else (1, 0), r["rep"]))
    with open(out / "rows_cap.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (repr(r[k]) if isinstance(r.get(k), float) else r.get(k, "")) for k in keys})

    write_curve(out / "curve.md", R, J)
    summary, scores = write_md(out / "judge_cap.md", R, J, checks, notes)

    per_task = {}
    for (act, s), Rr in R.items():
        per_task[f"{act}_s{s}"] = {t: {k: v for k, v in Rr["tasks"][t].items() if k not in ("fresh",)} for t in Rr["order"]}
    (out / "judge_cap.json").write_text(json.dumps(_jsonable({
        "spec": "specs/spec_head0_mlp_1108.md 追補 A", "raw": str(raw), "prev": str(prev),
        "runs": J, "per_task": per_task, "checks": checks, "summary_by_activation": summary,
        "prediction_scores": scores, "notes": notes,
        "conventions": {"A_end_window": N_END, "alive": ALIVE, "open05": OPEN05, "min_matched_pairs": MIN_MATCHED_PAIRS,
                        "k_equiv_median_inf_counts_out_of_range": True}}), ensure_ascii=False, indent=1))
    make_fig(out / "fig_cap.png", R, J)

    print(f"{'act':5s} {'s':>2s} {'n':>3s} {'中':>3s} {'上':>3s} {'下':>3s} {'SAT':>4s} {'J_A1':15s} {'|Δ|≤.05':>8s} {'J_A2':12s} "
          f"{'s':>7s} {'keq_med':>8s} {'k90':>5s} {'A(6)':>6s} {'A(13)':>6s} {'A(45)':>6s}")
    for act in ACTS:
        for s in SEEDS:
            r = J.get(f"{act}_s{s}")
            if r is None:
                continue
            c = r["counts"]
            print(f"{act:5s} {s:2d} {r['n_tasks']:3d} {c['中']:3d} {c['上']:3d} {c['下']:3d} {c['SAT']:4d} {r['A1']:15s} "
                  f"{r['A2_within_count']:>4d}/{r['n_A2']:<3d} {r['A2']:12s} {fmt(r['noise']['s'], 4):>7s} {fmt_k(r['k_equiv_median']):>8s} "
                  f"{('>100' if r['k90'] is None else str(r['k90'])):>5s} {fmt(r['A_fresh_at'][6]):>6s} {fmt(r['A_fresh_at'][13]):>6s} {fmt(r['A_fresh_at'][45]):>6s}")
    for act, v in summary.items():
        print(f"  {ACT_JA[act]}: J A1 {v['A1']} / J A2 {v['A2']}")
    print(f"wrote {out}/rows_cap.csv, curve.md, judge_cap.md, judge_cap.json, fig_cap.png")


if __name__ == "__main__":
    main()

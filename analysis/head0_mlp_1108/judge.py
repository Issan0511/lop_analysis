"""head0_mlp_1108 の集計・判定・図（spec §4・§5・§8）。

実行:
  cd ~/Projects/claude/wt/head0_mlp_1108
  PYTHONPATH=. .venv/bin/python -m analysis.head0_mlp_1108.judge \
      [--raw results/head0_mlp_1108/raw] [--out results/head0_mlp_1108]

入力は src/head0_mlp_1108.py の raw npz（{ACT}_s{seed}.npz と同名の .json）。
欠けた腕・課題は飛ばし、judge.md の「欠け」に書く。計算はすべて float64。
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
ARMS = ["S", "Sb", "H0", "H0b", "H0k", "H0A", "fa", "fb"]
ARM_TASKS = {
    "S": range(1, 31), "fa": range(1, 31),
    "H0": range(2, 31), "H0A": range(2, 31),
    "Sb": range(21, 31), "H0b": range(21, 31), "H0k": range(21, 31), "fb": range(21, 31),
}
JUDGE_TASKS = list(range(21, 31))
PAIRS = {"S": "Sb", "H0": "H0b", "fa": "fb"}
EVAL_STEP = 10           # 評価は 10 更新ごと（更新 0, 10, …）
N_END = 50               # 漸近の窓: 末の 50 点（更新 3,510〜4,000）
THRS = {"T20": 0.2, "T50": 0.5, "T90": 0.9}

# Okabe–Ito
C_S, C_H0, C_F, C_RA, C_R1 = "#000000", "#D55E00", "#009E73", "#0072B2", "#E69F00"

# 予測（spec §7、走る前に固定されたもの）
RECV_RHO_A = {"LR": (0.7, 0.9), "R": (0.95, 1.05), "ELU": (0.85, 1.0), "SILU": (0.85, 1.0), "GELU": (0.85, 1.0)}
RECV_RHO_1000 = {"LR": (0.5, 0.8), "R": (0.95, 1.05), "ELU": (0.8, 1.0), "SILU": (0.8, 1.0), "GELU": (0.8, 1.0)}
RECV_J1 = {"LR": "BODY_SIDE", "R": "BODY_SIDE+FULL_BODY", "ELU": "BODY_SIDE", "SILU": "BODY_SIDE", "GELU": "BODY_SIDE"}


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
            d = {k: z[k] for k in z.files}
            js = raw / f"{act}_s{s}.json"
            meta = {}
            if js.exists():
                try:
                    meta = json.loads(js.read_text())
                except Exception as e:  # noqa: BLE001
                    notes.append(f"{js.name} を読めない: {e}")
            runs[(act, s)] = (d, meta)
    return runs, notes


def run_metrics(online, ev):
    ev = np.asarray(ev, dtype=np.float64)
    online = np.asarray(online, dtype=np.float64)
    n = ev.shape[0]
    acc, loss = ev[:, 0], ev[:, 1]
    cens_val = EVAL_STEP * n  # 4,010（401 点のとき）
    out = {"n_eval": n, "n_online": online.shape[0]}
    out["spk"] = float(online[:5].mean()) if online.shape[0] >= 5 else float("nan")
    out["A100"] = float(acc[10]) if n > 10 else float("nan")
    out["A1000"] = float(acc[100]) if n > 100 else float("nan")
    for name, thr in THRS.items():
        hit = np.nonzero(acc >= thr)[0]
        if hit.size:
            out[name], out[name + "_cens"] = int(hit[0] * EVAL_STEP), 0
        else:
            out[name], out[name + "_cens"] = int(cens_val), 1
    k = min(N_END, n)
    out["A_end"] = float(acc[-k:].mean())
    out["L_end"] = float(loss[-k:].mean())
    return out


def unit_metrics(unit, head):
    out = {}
    if unit is not None:
        u = np.asarray(unit, dtype=np.float64)
        out["m_start"], out["m_end"] = float(u[0, 0].mean()), float(u[1, 0].mean())
        out["pplus_start"], out["pplus_end"] = float(u[0, 1].mean()), float(u[1, 1].mean())
        out["sigma_start"], out["sigma_end"] = float(u[0, 2].mean()), float(u[1, 2].mean())
        out["dead_end"] = float((u[1, 1] < 0.005).mean())
    if head is not None:
        h = np.asarray(head, dtype=np.float64)
        out["head_start"], out["head_end"] = float(h[0]), float(h[1])
    return out


LOGK = math.log(10.0)


def loading_prev(d, t):
    """log K − L_prev: 本線 S の課題 t−1 の末（更新 4,000、eval の最後の点）の全 1,200 枚 CE 損失から。"""
    k = f"S_t{t - 1:02d}_eval"
    if t < 2 or k not in d:
        return float("nan")
    return float(LOGK - np.asarray(d[k], dtype=np.float64)[-1, 1])


def build_rows(runs):
    rows, missing = [], []
    for (act, s), (d, _) in runs.items():
        for arm in ARMS:
            miss_t = []
            for t in ARM_TASKS[arm]:
                pre = f"{arm}_t{t:02d}"
                if pre + "_eval" not in d or pre + "_online" not in d:
                    miss_t.append(t)
                    continue
                r = {"act": act, "seed": s, "task": t, "arm": arm, "logK_minus_Lprev": loading_prev(d, t)}
                r.update(run_metrics(d[pre + "_online"], d[pre + "_eval"]))
                r.update(unit_metrics(d.get(pre + "_unit"), d.get(pre + "_head")))
                rows.append(r)
            if miss_t:
                missing.append((act, s, arm, miss_t))
    return rows, missing


# ---------------------------------------------------------------- 判定
class Table:
    def __init__(self, rows):
        self.v = {(r["act"], r["seed"], r["task"], r["arm"]): r for r in rows}

    def get(self, act, s, t, arm, q):
        r = self.v.get((act, s, t, arm))
        return float(r[q]) if r is not None else float("nan")

    def avg(self, act, s, t, arm, q):
        """課題 21〜30 は取り直しの対の平均（片方が欠ければ在る方）、それ以外は単走。"""
        a = self.get(act, s, t, arm, q)
        if arm in PAIRS and t in JUDGE_TASKS:
            b = self.get(act, s, t, PAIRS[arm], q)
            if np.isfinite(a) and np.isfinite(b):
                return 0.5 * (a + b)
            return a if np.isfinite(a) else b
        return a


def per_task_ratio(tb, act, s, q):
    out = {}
    for t in range(2, 31):
        f, h, S = (tb.avg(act, s, t, a, q) for a in ("fa", "H0", "S"))
        D = f - S
        rho = (f - h) / D if (np.isfinite(D) and D != 0 and np.isfinite(h)) else float("nan")
        out[t] = {"rho": rho, "D": D, "f": f, "H0": h, "S": S}
    return out


def sigma2(tb, act, s, arm, q, tasks):
    d = [tb.get(act, s, t, arm, q) - tb.get(act, s, t, PAIRS[arm], q) for t in tasks]
    d = np.array([x for x in d if np.isfinite(x)])
    return (float(np.mean(d ** 2) / 2.0), int(d.size)) if d.size else (float("nan"), 0)


def judge_ratio(tb, act, s, q, suffix=""):
    pt = per_task_ratio(tb, act, s, q)
    full = [t for t in JUDGE_TASKS
            if all(np.isfinite(tb.get(act, s, t, a, q)) for a in ("S", "Sb", "H0", "H0b", "fa", "fb"))]
    res = {"quantity": q, "tasks_used": full,
           "rho_per_task": {t: pt[t]["rho"] for t in range(2, 31)},
           "D_per_task": {t: pt[t]["D"] for t in range(2, 31)}}
    if len(full) < 2:
        res.update(label="NOT_AVAILABLE" + suffix, reason="課題 21〜30 で 6 腕がそろう課題が 2 未満")
        return res
    sf2, _ = sigma2(tb, act, s, "fa", q, full)
    sh2, _ = sigma2(tb, act, s, "H0", q, full)
    ss2, _ = sigma2(tb, act, s, "S", q, full)
    rhos = np.array([pt[t]["rho"] for t in full])
    rt = float(np.nanmedian(rhos)) if np.isfinite(rhos).any() else float("nan")
    Dbar = float(np.mean([pt[t]["D"] for t in full]))
    n = len(full)
    s_rho2 = (sh2 / 2 + sf2 / 2 + rt ** 2 * (ss2 / 2 + sf2 / 2)) / Dbar ** 2 if Dbar != 0 else float("inf")
    s_rho = math.sqrt(s_rho2)
    SE = 1.25 * s_rho / math.sqrt(n)
    lo, hi = rt - 2 * SE, rt + 2 * SE
    floor = 10 * max(math.sqrt(ss2), math.sqrt(sf2))
    testable = bool(Dbar > floor)
    if not testable:
        label = "NOT_TESTABLE"
    else:
        base = "BODY_SIDE" if lo > 0.5 else ("LOADING_SIDE" if hi < 0.5 else "MIXED")
        label = base + suffix
        if (lo <= 1.0 <= hi) or rt > 1.0:
            label += "+FULL_BODY"
    res.update(sigma_f2=sf2, sigma_H2=sh2, sigma_S2=ss2,
               sigma_f=math.sqrt(sf2), sigma_H=math.sqrt(sh2), sigma_S=math.sqrt(ss2),
               rho_med=rt, Dbar=Dbar, s_rho2=s_rho2, s_rho=s_rho, n=n, SE=SE,
               band=[lo, hi], floor=floor, testable=testable, label=label)
    return res


def med_diff(tb, act, s, a, b, q, tasks=JUDGE_TASKS):
    d = np.array([tb.get(act, s, t, a, q) - tb.get(act, s, t, b, q) for t in tasks])
    d = d[np.isfinite(d)]
    return (float(np.median(d)) if d.size else float("nan")), int(d.size)


def judge_run(tb, act, s):
    J = {"J1": judge_ratio(tb, act, s, "A_end"), "J2": judge_ratio(tb, act, s, "A1000", "_SPEED")}
    # J3
    ssp2, nsp = sigma2(tb, act, s, "fa", "spk", JUDGE_TASKS)
    mH, nH = med_diff(tb, act, s, "H0", "fa", "spk")
    mS, _ = med_diff(tb, act, s, "S", "fa", "spk")
    fa_spk = [tb.get(act, s, t, "fa", "spk") for t in JUDGE_TASKS]
    fa_spk = float(np.nanmedian(fa_spk)) if np.isfinite(fa_spk).any() else float("nan")
    sig_spk = math.sqrt(ssp2) if np.isfinite(ssp2) else float("nan")
    if not (np.isfinite(mH) and np.isfinite(sig_spk)):
        lab3 = "NOT_AVAILABLE"
    else:
        lab3 = "SPIKE_GONE" if abs(mH) <= 2 * sig_spk else "SPIKE_REMAINS"
    J["J3"] = {"spk_fa_med": fa_spk, "med_H0_minus_fa": mH, "med_S_minus_fa": mS,
               "sigma_spk2": ssp2, "sigma_spk": sig_spk, "two_sigma": 2 * sig_spk,
               "n": nH, "label": lab3}
    # J4・J5（物差しは同じ量の σ_H）
    for key, arm in (("J4", "H0A"), ("J5", "H0k")):
        rec = {}
        for q, jk in (("A_end", "J1"), ("A1000", "J2")):
            m, n = med_diff(tb, act, s, arm, "H0", q)
            sh = J[jk].get("sigma_H", float("nan"))
            lab = ("NOT_AVAILABLE" if not (np.isfinite(m) and np.isfinite(sh))
                   else ("差 ≤ 2σ" if abs(m) <= 2 * sh else "差 > 2σ"))
            rec[q] = {"median": m, "n": n, "sigma_H": sh, "two_sigma_H": 2 * sh, "label": lab}
        J[key] = rec
    # T90 の打ち切り
    cens = {}
    for arm in ("S", "H0", "fa", "Sb", "H0b", "fb"):
        v = [tb.v[(act, s, t, arm)]["T90_cens"] for t in JUDGE_TASKS if (act, s, t, arm) in tb.v]
        cens[arm] = {"censored": int(sum(v)), "n": len(v)}
    J["T90"] = cens
    # 送り手の予測 1 用: ReLU 等の課題 ≥ 15 の単走
    c15 = {}
    for arm in ("S", "H0", "fa"):
        v = [tb.v[(act, s, t, arm)]["T90_cens"] for t in range(15, 31) if (act, s, t, arm) in tb.v]
        c15[arm] = {"censored": int(sum(v)), "n": len(v)}
    J["T90_ge15"] = c15
    J["logK_minus_Lprev"] = {t: _lp(tb, act, s, t) for t in range(2, 31)}
    return J


def _lp(tb, act, s, t):
    for arm in ARMS:
        r = tb.v.get((act, s, t, arm))
        if r is not None:
            return r["logK_minus_Lprev"]
    return float("nan")


# ---------------------------------------------------------------- 検査
def _flatten(d, pre=""):
    out = {}
    for k, v in d.items():
        kk = f"{pre}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, kk + "."))
        else:
            out[kk] = v
    return out


def _nanmax(a):
    a = np.asarray([np.nan if x is None else x for x in np.ravel(a)], dtype=np.float64)
    return float(np.nanmax(a)) if np.isfinite(a).any() else float("nan")


# (表示名, 値の出どころ, 合格条件)。出どころ "npz:<鍵>" は npz、それ以外は json の check 下の鍵。
CHECK_SPEC = [
    ("C1 S-replay: 課題末 θ の差の絶対値の最大（30 課題）", "C1_end", "== 0"),
    ("C2 S-switch: 課題 2 の 200 更新後と mid[1] の差の絶対値の最大", "C2_mid1", "== 0"),
    ("C2 変異対照: Adam 状態を捨てたときの差の絶対値の最大", "mid1_mutant_maxabs", "> 0"),
    ("C3 fa の初期値と engine の初期値の差（パラメタごとの差の絶対値の最大の、パラメタ間の最小）", "fresh_a_t01_vs_engine_init_min_over_params", "> 0"),
    ("C3 fa と fb の初期値の差（同上）", "fresh_a_vs_b_min_over_params", "> 0"),
    ("C4 H0 の最初の 1 更新の体の勾配の絶対値の最大", "h0_first_body_grad", "== 0"),
]


def checks_for(d, meta):
    chk = meta.get("check", {}) if isinstance(meta, dict) else {}
    vals = {}
    end = d["check_end_maxabs"] if "check_end_maxabs" in d else chk.get("check_end_maxabs")
    vals["C1_end"] = _nanmax(end) if end is not None else float("nan")
    vals["C1_n_tasks"] = int(np.isfinite(np.asarray([np.nan if x is None else x for x in np.ravel(end)], dtype=np.float64)).sum()) if end is not None else 0
    mid = d["check_mid_maxabs"] if "check_mid_maxabs" in d else chk.get("check_mid_maxabs")
    vals["C2_mid1"] = float(np.ravel(mid)[0]) if mid is not None and len(np.ravel(mid)) and np.ravel(mid)[0] is not None else float("nan")
    for _, k, _ in CHECK_SPEC:
        if k not in vals:
            v = chk.get(k)
            vals[k] = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else float("nan")
    table = []
    for name, k, cond in CHECK_SPEC:
        v = vals[k]
        ok = None if not np.isfinite(v) else (v == 0 if cond == "== 0" else v > 0)
        table.append({"name": name, "key": k, "value": v, "condition": cond, "pass": ok})
    meta_flat = _flatten(meta) if isinstance(meta, dict) else {}
    info = {k: v for k, v in meta_flat.items() if k in ("torch", "git_head", "engine_sha256", "smoke", "steps", "tasks",
                                                       "check.end_check_applies", "check.init_equal")}
    return {"table": table, "C1_n_tasks": vals["C1_n_tasks"], "info": info,
            "json_check_raw": {k: v for k, v in chk.items()} if isinstance(chk, dict) else {}}


# ---------------------------------------------------------------- 出力
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


def base_label(lab):
    return lab.split("+")[0].replace("_SPEED", "")


def seed_agree(labels):
    labs = [l for l in labels if l is not None]
    if len(labs) < 2:
        return "未走（seed 欠け）"
    return f"確定 {labs[0]}" if all(l == labs[0] for l in labs) else "割れ"


def score_cell(vals, pred):
    """vals: seed ごとの真偽。両方当たり→当たり、両方外れ→外れ、それ以外→割れ。"""
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    if all(vals):
        return "当たり"
    if not any(vals):
        return "外れ"
    return "割れ"


def write_md(path, J, checks, missing, notes, n_eval_set):
    L = []
    w = L.append
    w("# head0_mlp_1108 判定（spec §5）\n")
    w("課題 21〜30 の 10 切替で判定した。課題 2〜20 の ρ は図と judge.json に出し、判定に使わない。\n")
    w(f"評価点の数: {sorted(n_eval_set)}（401 点が本測定）。\n")

    w("## 検査\n")
    w("| 活性化 | seed | 項目 | 値 | 条件 | 合否 |\n|---|---|---|---|---|---|")
    for act in ACTS:
        for s in SEEDS:
            c = checks.get(f"{act}_s{s}")
            if c is None:
                continue
            for row in c["table"]:
                ok = "—" if row["pass"] is None else ("通過" if row["pass"] else "不合格")
                w(f"| {ACT_JA[act]} | {s} | {row['name']} | {fmt(row['value'], 4)} | {row['condition']} | {ok} |")
    w("\n| 活性化 | seed | C1 の課題数 | torch | git | engine sha256 |\n|---|---|---|---|---|---|")
    for act in ACTS:
        for s in SEEDS:
            c = checks.get(f"{act}_s{s}")
            if c is None:
                continue
            i = c["info"]
            w(f"| {ACT_JA[act]} | {s} | {c['C1_n_tasks']} | {i.get('torch', '—')} | {str(i.get('git_head', '—'))[:10]} | "
              f"{str(i.get('engine_sha256', '—'))[:12]} |")
    w("")

    def ratio_table(key, title):
        w(f"## {title}\n")
        w("| 活性化 | seed | ρ̃ | 帯の下端 | 帯の上端 | D̄ | 床 10·max(σ_S, σ_f) | σ_f | σ_H | σ_S | s_ρ | SE | 課題数 | ラベル |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for act in ACTS:
            for s in SEEDS:
                r = J.get(f"{act}_s{s}", {}).get(key)
                if r is None:
                    continue
                if "rho_med" not in r:
                    w(f"| {ACT_JA[act]} | {s} | — | — | — | — | — | — | — | — | — | — | {len(r['tasks_used'])} | {r['label']} |")
                    continue
                w(f"| {ACT_JA[act]} | {s} | {fmt(r['rho_med'])} | {fmt(r['band'][0])} | {fmt(r['band'][1])} | "
                  f"{fmt(r['Dbar'])} | {fmt(r['floor'])} | {fmt(r['sigma_f'])} | {fmt(r['sigma_H'])} | {fmt(r['sigma_S'])} | "
                  f"{fmt(r['s_rho'])} | {fmt(r['SE'])} | {r['n']} | {r['label']} |")
        w("")

    ratio_table("J1", "J1 漸近（ρ_A、A_end）")
    ratio_table("J2", "J2 速さ（ρ_1000、A1000）")

    w("## J3 スパイク（更新 0〜4 のミニバッチ損失の平均）\n")
    w("| 活性化 | seed | fa の spk（中央値） | H0 − fa（中央値） | S − fa（中央値） | 2σ_spk | ラベル |")
    w("|---|---|---|---|---|---|---|")
    for act in ACTS:
        for s in SEEDS:
            r = J.get(f"{act}_s{s}", {}).get("J3")
            if r is None:
                continue
            w(f"| {ACT_JA[act]} | {s} | {fmt(r['spk_fa_med'])} | {fmt(r['med_H0_minus_fa'], 4)} | "
              f"{fmt(r['med_S_minus_fa'], 4)} | {fmt(r['two_sigma'], 4)} | {r['label']} |")
    w("")

    w("## J4（H0A − H0）・J5（H0k − H0）\n")
    w("| 活性化 | seed | J4 A_end 差 | J4 A1000 差 | J5 A_end 差 | J5 A1000 差 | 2σ_H（A_end） | 2σ_H（A1000） | J4 A_end | J4 A1000 | J5 A_end | J5 A1000 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for act in ACTS:
        for s in SEEDS:
            r = J.get(f"{act}_s{s}")
            if r is None:
                continue
            j4, j5 = r["J4"], r["J5"]
            w(f"| {ACT_JA[act]} | {s} | {fmt(j4['A_end']['median'], 4)} | {fmt(j4['A1000']['median'], 4)} | "
              f"{fmt(j5['A_end']['median'], 4)} | {fmt(j5['A1000']['median'], 4)} | "
              f"{fmt(j4['A_end']['two_sigma_H'], 4)} | {fmt(j4['A1000']['two_sigma_H'], 4)} | "
              f"{j4['A_end']['label']} | {j4['A1000']['label']} | {j5['A_end']['label']} | {j5['A1000']['label']} |")
    w("\n差は課題 21〜30 の中央値。\n")

    w("## T90 の打ち切り（課題 21〜30、0.9 に届かなかった課題の数 / 課題数）\n")
    w("| 活性化 | seed | S | H0 | fa | Sb | H0b | fb |\n|---|---|---|---|---|---|---|---|")
    for act in ACTS:
        for s in SEEDS:
            r = J.get(f"{act}_s{s}")
            if r is None:
                continue
            c = r["T90"]
            w(f"| {ACT_JA[act]} | {s} | " + " | ".join(f"{c[a]['censored']}/{c[a]['n']}" for a in ("S", "H0", "fa", "Sb", "H0b", "fb")) + " |")
    w("")

    w("## 判定のまとめ\n")
    w("| 活性化 | J1 seed 0 | J1 seed 1 | J1 | J2 | J3 |\n|---|---|---|---|---|---|")
    summary = {}
    for act in ACTS:
        labs = {k: [J.get(f"{act}_s{s}", {}).get(k, {}).get("label") for s in SEEDS] for k in ("J1", "J2", "J3")}
        summary[act] = {k: seed_agree(v) for k, v in labs.items()}
        w(f"| {ACT_JA[act]} | {labs['J1'][0] or '—'} | {labs['J1'][1] or '—'} | {summary[act]['J1']} | "
          f"{summary[act]['J2']} | {summary[act]['J3']} |")
    w("")

    # ---- 予測の採点
    w("## 予測の採点\n")
    w("seed ごとに当否を出し、2 seed とも当たり → 当たり、2 seed とも外れ → 外れ、それ以外 → 割れ。\n")
    w("### 受け手の予測（spec §7 の表）\n")
    w("| 活性化 | 量 | 予測 | seed 0 | seed 1 | 採点 |\n|---|---|---|---|---|---|")

    def gr(act, s):
        return J.get(f"{act}_s{s}")

    for act in ACTS:
        # ρ_A
        for key, rng, name in (("J1", RECV_RHO_A, "ρ̃_A"), ("J2", RECV_RHO_1000, "ρ̃_1000")):
            lo, hi = rng[act]
            vals, hits = [], []
            for s in SEEDS:
                r = gr(act, s)
                v = r[key].get("rho_med") if r else None
                vals.append(v)
                hits.append(None if v is None or not np.isfinite(v) else (lo <= v <= hi))
            w(f"| {ACT_JA[act]} | {name} | {lo}〜{hi} | {fmt(vals[0])} | {fmt(vals[1])} | {score_cell(hits, None)} |")
        # J1
        labs = [gr(act, s)["J1"]["label"] if gr(act, s) else None for s in SEEDS]
        pred = RECV_J1[act]
        if act == "R":
            hits = [None if l is None else (l == pred) for l in labs]
        else:
            hits = [None if l is None else (base_label(l) == pred) for l in labs]
        w(f"| {ACT_JA[act]} | J1 | {pred} | {labs[0] or '—'} | {labs[1] or '—'} | {score_cell(hits, None)} |")
        # J3
        labs = [gr(act, s)["J3"]["label"] if gr(act, s) else None for s in SEEDS]
        hits = [None if l is None or l == "NOT_AVAILABLE" else (l == "SPIKE_GONE") for l in labs]
        w(f"| {ACT_JA[act]} | J3 | SPIKE_GONE | {labs[0] or '—'} | {labs[1] or '—'} | {score_cell(hits, None)} |")
        # J4・J5
        for key in ("J4", "J5"):
            cells, hits = [], []
            for s in SEEDS:
                r = gr(act, s)
                if r is None:
                    cells.append("—"); hits.append(None); continue
                la, lb = r[key]["A_end"]["label"], r[key]["A1000"]["label"]
                cells.append(f"{la} / {lb}")
                hits.append(None if "NOT_AVAILABLE" in (la, lb) else (la == "差 ≤ 2σ" and lb == "差 ≤ 2σ"))
            w(f"| {ACT_JA[act]} | {key}（A_end / A1000） | 差 ≤ 2σ | {cells[0]} | {cells[1]} | {score_cell(hits, None)} |")
        # T90
        cells, hits = [], []
        for s in SEEDS:
            r = gr(act, s)
            if r is None:
                cells.append("—"); hits.append(None); continue
            c = r["T90"]
            cells.append(f"S {c['S']['censored']}/{c['S']['n']}・H0 {c['H0']['censored']}/{c['H0']['n']}・fa {c['fa']['censored']}/{c['fa']['n']}")
            ok = (c["S"]["n"] > 0 and c["S"]["censored"] == c["S"]["n"] and c["H0"]["censored"] == c["H0"]["n"]
                  and c["fa"]["censored"] == 0)
            hits.append(ok if c["S"]["n"] and c["H0"]["n"] and c["fa"]["n"] else None)
        w(f"| {ACT_JA[act]} | T90 打ち切り（課題 21〜30） | S・H0 全課題で届かず、fa 全課題で届く | {cells[0]} | {cells[1]} | {score_cell(hits, None)} |")
    w("")

    w("### 送り手の予測（spec §7 の 6 つ）\n")
    w("| 番号 | 予測 | 測定 seed 0 | 測定 seed 1 | 採点 |\n|---|---|---|---|---|")
    # 1
    cells, hits = [], []
    for s in SEEDS:
        r = gr("R", s)
        if r is None:
            cells.append("—"); hits.append(None); continue
        c = r["T90_ge15"]
        cells.append(f"S {c['S']['censored']}/{c['S']['n']}・H0 {c['H0']['censored']}/{c['H0']['n']}・fa {c['fa']['censored']}/{c['fa']['n']}")
        hits.append((c["S"]["censored"] == c["S"]["n"] and c["H0"]["censored"] == c["H0"]["n"] and c["fa"]["censored"] == 0)
                    if c["S"]["n"] and c["H0"]["n"] and c["fa"]["n"] else None)
    w(f"| 1 | ReLU・課題 ≥ 15（単走）: S も H0 も 0.9 に届かず、fa は届く（T90 打ち切りの数） | {cells[0]} | {cells[1]} | {score_cell(hits, None)} |")

    def rho_of(act, s):
        r = gr(act, s)
        v = r["J1"].get("rho_med") if r else None
        return v if v is not None and np.isfinite(v) else None

    # 2
    vals = [rho_of("ELU", s) for s in SEEDS]
    hits = [None if v is None else v >= 0.7 for v in vals]
    w(f"| 2 | ELU の ρ̃_A ≥ 0.7 | {fmt(vals[0])} | {fmt(vals[1])} | {score_cell(hits, None)} |")
    # 3
    vals = [rho_of("LR", s) for s in SEEDS]
    hits = [None if v is None else 0.5 <= v <= 0.8 for v in vals]
    w(f"| 3 | leaky の ρ̃_A が 0.5〜0.8 | {fmt(vals[0])} | {fmt(vals[1])} | {score_cell(hits, None)} |")
    # 4
    cells, hits = [], []
    for s in SEEDS:
        e, l = rho_of("ELU", s), rho_of("LR", s)
        si, ge = rho_of("SILU", s), rho_of("GELU", s)
        cells.append(f"ELU {fmt(e)}・leaky {fmt(l)}・SiLU {fmt(si)}・GELU {fmt(ge)}")
        if None in (e, l, si, ge):
            hits.append(None)
        else:
            a, b = min(e, l), max(e, l)
            hits.append(a <= si <= b and a <= ge <= b)
    w(f"| 4 | SiLU・GELU の ρ̃_A が ELU と leaky の間 | {cells[0]} | {cells[1]} | {score_cell(hits, None)} |")
    # 5
    cells, hits = [], []
    for s in SEEDS:
        labs = [gr(a, s)["J3"]["label"] if gr(a, s) else None for a in ACTS]
        cells.append("・".join(f"{ACT_JA[a]} {l or '—'}" for a, l in zip(ACTS, labs)))
        ok = [l for l in labs if l not in (None, "NOT_AVAILABLE")]
        hits.append(all(l == "SPIKE_GONE" for l in ok) if ok else None)
    w(f"| 5 | スパイクは H0 で fresh 並みに消える（J3 が 5 活性化とも SPIKE_GONE） | {cells[0]} | {cells[1]} | {score_cell(hits, None)} |")
    # 6
    cells, hits = [], []
    for s in SEEDS:
        parts, ok = [], []
        for a in ACTS:
            r = gr(a, s)
            if r is None:
                continue
            j = r["J4"]["A1000"]
            parts.append(f"{ACT_JA[a]} {fmt(j['median'], 4)}（2σ_H {fmt(j['two_sigma_H'], 4)}）")
            if np.isfinite(j["median"]) and np.isfinite(j["two_sigma_H"]):
                ok.append(j["median"] <= j["two_sigma_H"])
        cells.append("・".join(parts) if parts else "—")
        hits.append(all(ok) if ok else None)
    w(f"| 6 | H0A は H0 より速くならない（J4 の A1000 差の中央値 ≤ 2σ_H、5 活性化とも） | {cells[0]} | {cells[1]} | {score_cell(hits, None)} |")
    w("")

    if missing or notes:
        w("## 欠け\n")
        if missing:
            w("| 活性化 | seed | 腕 | 欠けた課題 |\n|---|---|---|---|")
        for act, s, arm, ts in missing:
            w(f"| {ACT_JA[act]} | {s} | {arm} | {', '.join(map(str, ts)) if len(ts) < 12 else f'{ts[0]}〜{ts[-1]}（{len(ts)} 課題）'} |")
        w("")
        for n in notes:
            w(f"- {n}")
        w("")
    path.write_text("\n".join(L) + "\n")
    return summary


def write_early(path, tb, J):
    L = ["# 早い切替（課題 2〜10）の記録\n",
         "取り直しが無いので帯もラベルも出さない（spec §4・§5）。値は単走。"
         "log K − L_prev は本線 S の課題 t−1 の末の全 1,200 枚 CE 損失 L_prev から、K = 10。"
         "D = A_end[fa] − A_end[S]。\n",
         "| 活性化 | seed | 課題 | ρ_A | ρ_1000 | log K − L_prev | D | A_end S | A_end H0 | A_end fa | A1000 S | A1000 H0 | A1000 fa |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for act in ACTS:
        for s in SEEDS:
            r = J.get(f"{act}_s{s}")
            if r is None:
                continue
            for t in range(2, 11):
                g = lambda arm, q: fmt(tb.get(act, s, t, arm, q))  # noqa: E731
                L.append(f"| {ACT_JA[act]} | {s} | {t} | {fmt(r['J1']['rho_per_task'][t])} | {fmt(r['J2']['rho_per_task'][t])} | "
                         f"{fmt(r['logK_minus_Lprev'][t])} | {fmt(r['J1']['D_per_task'][t])} | "
                         f"{g('S', 'A_end')} | {g('H0', 'A_end')} | {g('fa', 'A_end')} | "
                         f"{g('S', 'A1000')} | {g('H0', 'A1000')} | {g('fa', 'A1000')} |")
    path.write_text("\n".join(L) + "\n")


def make_fig(path, tb, J):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = ["Noto Sans CJK JP"] if "Noto Sans CJK JP" in names else ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(3, 5, figsize=(17, 10), sharey="row")
    lw, ms = 1.2, 4.5
    for j, act in enumerate(ACTS):
        ax = axes[0, j]
        for s, alpha in zip(SEEDS, (1.0, 0.5)):
            for arm, col, ls, lab in (("S", C_S, "-", "継続 S"), ("H0", C_H0, "--", "head を 0 に H0"), ("fa", C_F, ":", "新品 fa")):
                ts = [t for t in range(1, 31) if (act, s, t, arm) in tb.v]
                if not ts:
                    continue
                ax.plot(ts, [tb.get(act, s, t, arm, "A_end") for t in ts], color=col, ls=ls, lw=lw, alpha=alpha,
                        label=f"{lab}（seed {s}）")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0.5, 30.5)
        ax.set_title(ACT_JA[act])
        ax.set_xlabel("課題")
        ax.grid(alpha=0.25, lw=0.5)
        if j == 0:
            ax.set_ylabel("漸近の訓練精度")
        ax = axes[1, j]
        ax.axvspan(20.5, 30.5, color="0.9", zorder=0, lw=0)
        for y in (0, 0.5, 1):
            ax.axhline(y, color="0.5", lw=0.7, ls="-" if y != 0.5 else "--", zorder=1)
        for s, alpha in zip(SEEDS, (1.0, 0.5)):
            r = J.get(f"{act}_s{s}")
            if r is None:
                continue
            for key, col, mk, ls, lab in (("J1", C_RA, "o", "-", "ρ_A"), ("J2", C_R1, "^", "--", "ρ_1000")):
                rp = r[key]["rho_per_task"]
                ts = [t for t in range(2, 31) if np.isfinite(rp[t])]
                ys = np.clip([rp[t] for t in ts], -0.5, 1.5)
                ax.plot(ts, ys, color=col, ls=ls, marker=mk, ms=ms, lw=lw, alpha=alpha, label=f"{lab}（seed {s}）", zorder=3)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlim(0.5, 30.5)
        ax.set_xlabel("課題")
        ax.grid(alpha=0.25, lw=0.5)
        if j == 0:
            ax.set_ylabel("比 ρ")
        # 下段: ρ を前の課題への装填 log K − L_prev に対して
        ax = axes[2, j]
        for y in (0, 0.5, 1):
            ax.axhline(y, color="0.5", lw=0.7, ls="-" if y != 0.5 else "--", zorder=1)
        for s, alpha in zip(SEEDS, (1.0, 0.5)):
            r = J.get(f"{act}_s{s}")
            if r is None:
                continue
            lp = r["logK_minus_Lprev"]
            for key, col, mk, lab in (("J1", C_RA, "o", "ρ_A"), ("J2", C_R1, "^", "ρ_1000")):
                rp = r[key]["rho_per_task"]
                for win, fill, wl in ((range(2, 21), "none", "課題 2〜20"), (range(21, 31), col, "課題 21〜30")):
                    ts = [t for t in win if np.isfinite(rp[t]) and np.isfinite(lp[t])]
                    if not ts:
                        continue
                    ax.plot([lp[t] for t in ts], np.clip([rp[t] for t in ts], -0.5, 1.5), ls="none", marker=mk, ms=ms + 0.5,
                            mec=col, mfc=fill, mew=1.0, alpha=alpha, zorder=3, label=f"{lab}・{wl}（seed {s}）")
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel("前の課題への装填 log K − L")
        ax.grid(alpha=0.25, lw=0.5)
        if j == 0:
            ax.set_ylabel("比 ρ")
    for row in (0, 1, 2):
        h, l = axes[row, 0].get_legend_handles_labels()
        if not h:
            for ax in axes[row]:
                h, l = ax.get_legend_handles_labels()
                if h:
                    break
        if h:
            axes[row, -1].legend(h, l, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
    axes[1, -1].text(1.02, 0.02, "灰色の帯: 判定の窓（課題 21〜30）", transform=axes[1, -1].transAxes, fontsize=8, va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        o = float(o)
        return o if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/head0_mlp_1108/raw")
    ap.add_argument("--out", default="results/head0_mlp_1108")
    a = ap.parse_args(argv)
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs, notes = load_raw(raw)
    if not runs:
        raise SystemExit(f"raw が無い: {raw}")
    rows, missing = build_rows(runs)
    keys = ["act", "seed", "task", "arm", "logK_minus_Lprev", "spk", "A100", "A1000", "T20", "T20_cens", "T50", "T50_cens", "T90", "T90_cens",
            "A_end", "L_end", "m_start", "m_end", "pplus_start", "pplus_end", "dead_end", "sigma_start", "sigma_end",
            "head_start", "head_end", "n_eval", "n_online"]
    order = {a_: i for i, a_ in enumerate(ACTS)}
    arm_o = {a_: i for i, a_ in enumerate(ARMS)}
    rows.sort(key=lambda r: (order[r["act"]], r["seed"], r["task"], arm_o[r["arm"]]))
    with open(out / "rows.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (repr(r[k]) if isinstance(r.get(k), float) else r.get(k, "")) for k in keys})
    tb = Table(rows)
    J, checks = {}, {}
    for (act, s), (d, meta) in runs.items():
        J[f"{act}_s{s}"] = judge_run(tb, act, s)
        checks[f"{act}_s{s}"] = checks_for(d, meta)
    n_eval_set = {r["n_eval"] for r in rows}
    summary = write_md(out / "judge.md", J, checks, missing, notes, n_eval_set)
    (out / "judge.json").write_text(json.dumps(_jsonable({
        "spec": "specs/spec_head0_mlp_1108.md", "judge_tasks": JUDGE_TASKS,
        "runs": J, "checks": checks, "summary_by_activation": summary,
        "missing": [{"act": a_, "seed": s, "arm": arm, "tasks": ts} for a_, s, arm, ts in missing],
        "notes": notes}), ensure_ascii=False, indent=1))
    write_early(out / "early_switches.md", tb, J)
    make_fig(out / "fig_head0_mlp_1108.png", tb, J)
    print(f"{'act':5s} {'s':>2s} {'rho_A':>7s} {'band':>17s} {'Dbar':>7s} {'J1':28s} {'rho1000':>7s} {'J2':28s} J3")
    for act in ACTS:
        for s in SEEDS:
            r = J.get(f"{act}_s{s}")
            if r is None:
                continue
            j1, j2 = r["J1"], r["J2"]
            band = f"[{fmt(j1['band'][0])},{fmt(j1['band'][1])}]" if "band" in j1 else "—"
            print(f"{act:5s} {s:2d} {fmt(j1.get('rho_med')):>7s} {band:>17s} {fmt(j1.get('Dbar')):>7s} {j1['label']:28s} "
                  f"{fmt(j2.get('rho_med')):>7s} {j2['label']:28s} {r['J3']['label']}")
    print(f"wrote {out}/rows.csv, judge.md, judge.json, early_switches.md, fig_head0_mlp_1108.png")


if __name__ == "__main__":
    main()

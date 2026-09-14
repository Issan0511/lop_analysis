#!/usr/bin/env python3
"""Aggregation and registered labels for wcap_rlmnist_0914 (specs/spec_wcap_rlmnist_0914.md sections 4-5).

    python3 analysis/wcap_rlmnist_0914/verdict.py                       # the real run
    python3 analysis/wcap_rlmnist_0914/verdict.py --src results/_smoke_wcap_rlmnist_0914/runs \
        --out results/_smoke_wcap_rlmnist_0914                         # smoke: always pass --src explicitly

Shards are enumerated by name (act x arm x seed), never globbed.  Writes per_task.csv, layer_metrics.csv,
verdict.csv, summary.md, provenance.json (and copies checks.json) into --out.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUN_ID = "wcap_rlmnist_0914"
ARMS8 = [("LR", "ref"), ("LR", "l2"), ("LR", "l2init"), ("LR", "capT1"), ("LR", "cap2"),
         ("R", "ref"), ("R", "l2"), ("R", "cap2")]
SEEDS = tuple(range(10))
N_TASKS = 50
WIN = (31, 50)                 # G = A(1) - A(WIN)
P_WINS = ((11, 20), (41, 50))  # P = A(11-20) - A(41-50)
D_WIN = (2, 6)                 # D = A(2-6) - A(WIN)
SLOPE_WIN = (11, 50)
BAND_LO, BAND_HI = 0.1, 0.9    # spec 5.1
DELTA_BAND = 0.1               # spec 5.2 Q2
GUARD = 2.0                    # gamma, kappa >= 2
MIN_SEEDS = 8
N_BOOT = 10_000
RNG_SEED = 20260914
STREAM_KEYS = ("init_sha256", "subset_idx_sha256", "labels_sha256", "batch_sha256")
TRAJ_TASKS = (1, 2, 3, 5, 10, 20, 30, 40, 50)


def shard_name(act: str, arm: str, seed: int) -> str:
    return f"{act}_{arm}_s{seed}"


# --------------------------------------------------------------------------
# loading and validity
# --------------------------------------------------------------------------

def load(src: Path):
    pts, lms, provs, missing = [], [], {}, []
    for act, arm in ARMS8:
        for seed in SEEDS:
            d = src / shard_name(act, arm, seed)
            if not (d / "provenance.json").exists():
                missing.append(shard_name(act, arm, seed))
                continue
            provs[(act, arm, seed)] = json.loads((d / "provenance.json").read_text())
            pts.append(pd.read_csv(d / "per_task.csv"))
            if (d / "layer_metrics.csv").exists():
                lms.append(pd.read_csv(d / "layer_metrics.csv"))
    pt = pd.concat(pts, ignore_index=True) if pts else pd.DataFrame()
    lm = pd.concat(lms, ignore_index=True) if lms else pd.DataFrame()
    return pt, lm, provs, missing


def valid_seeds(pt: pd.DataFrame, provs: dict) -> tuple[list[int], dict]:
    """spec 2.4 / 5.2: a seed counts only if all 8 arms finished 50 tasks without divergence and consumed
    bit-identical streams (init, 1200 images, labels, batch orders)."""
    ok, why = [], {}
    for seed in SEEDS:
        keys = [(a, m, seed) for a, m in ARMS8]
        if any(k not in provs for k in keys):
            why[seed] = "missing shard"
            continue
        runs = [provs[k]["runs"][str(seed)] for k in keys]
        if any(r["divergence"]["diverged"] or r["tasks_completed"] != N_TASKS for r in runs):
            why[seed] = "diverged or incomplete"
            continue
        n_rows = [len(pt[(pt.act == a) & (pt.arm == m) & (pt.seed == seed) & pt.online_acc.notna()]) for a, m, _ in keys]
        if any(n != N_TASKS for n in n_rows):
            why[seed] = f"per_task rows {n_rows}"
            continue
        if len({json.dumps({f: r[f] for f in STREAM_KEYS}, sort_keys=True) for r in runs}) != 1:
            why[seed] = "stream sha256 differ across arms"
            continue
        ok.append(seed)
    return ok, why


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------

def per_seed(pt: pd.DataFrame, provs: dict, act: str, arm: str, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for s in seeds:
        g = pt[(pt.act == act) & (pt.arm == arm) & (pt.seed == s)].set_index("task").sort_index()
        A = g.online_acc
        win = A.loc[WIN[0]:WIN[1]].mean()
        h = A.loc[SLOPE_WIN[0]:SLOPE_WIN[1]]
        init_med = provs[(act, arm, s)]["runs"][str(s)]["init_centered"]["wt_med_l1"]
        rows.append(dict(
            seed=s, A1=A.loc[1], Awin=win, G=A.loc[1] - win,
            P=A.loc[P_WINS[0][0]:P_WINS[0][1]].mean() - A.loc[P_WINS[1][0]:P_WINS[1][1]].mean(),
            D=A.loc[D_WIN[0]:D_WIN[1]].mean() - win,
            slope=float(np.polyfit(h.index.to_numpy(float), h.to_numpy(float), 1)[0]) * 10,
            memo1=g.memo_acc.loc[1], memo_win=g.memo_acc.loc[WIN[0]:WIN[1]].mean(),
            wt1_t0=init_med, wt1_t1=g.wt_med_l1.loc[1], wt1_t10=g.wt_med_l1.loc[10], wt1_t50=g.wt_med_l1.loc[N_TASKS],
            wt1_win=g.wt_med_l1.loc[WIN[0]:WIN[1]].mean(),
            wt2_t0=provs[(act, arm, s)]["runs"][str(s)]["init_centered"]["wt_med_l2"],
            wt2_t1=g.wt_med_l2.loc[1], wt2_t50=g.wt_med_l2.loc[N_TASKS], wt2_win=g.wt_med_l2.loc[WIN[0]:WIN[1]].mean(),
            gamma=g.wt_med_l1.loc[N_TASKS] / g.wt_med_l1.loc[1],
            kappa=g.wt_med_l1.loc[WIN[0]:WIN[1]].mean() / (2.0 * init_med)))
    return pd.DataFrame(rows).set_index("seed")


def boot_idx(n: int) -> np.ndarray:
    return np.random.default_rng(RNG_SEED).integers(0, n, (N_BOOT, n))


def ci(boot: np.ndarray) -> dict:
    lo95, hi95 = np.percentile(boot, [2.5, 97.5])
    lo975, hi975 = np.percentile(boot, [1.25, 98.75])
    return dict(lo95=float(lo95), hi95=float(hi95), lo975=float(lo975), hi975=float(hi975))


def removal(Gm: np.ndarray, Gref: np.ndarray, idx: np.ndarray) -> dict:
    """rho = 1 - mean(G_m) / mean(G_ref), seed-paired bootstrap (same index for numerator and denominator)."""
    est = 1.0 - Gm.mean() / Gref.mean()
    boot = 1.0 - Gm[idx].mean(axis=1) / Gref[idx].mean(axis=1)
    return dict(est=float(est), **ci(boot), per_seed=[float(v) for v in 1.0 - Gm / Gref])


def delta_removal(Gl2: np.ndarray, Gcap: np.ndarray, Gref: np.ndarray, idx: np.ndarray) -> dict:
    """Delta = rho_cap - rho_l2 = (mean G_l2 - mean G_cap) / mean G_ref."""
    est = (Gl2.mean() - Gcap.mean()) / Gref.mean()
    boot = (Gl2[idx].mean(axis=1) - Gcap[idx].mean(axis=1)) / Gref[idx].mean(axis=1)
    return dict(est=float(est), **ci(boot))


def mean_ci(x: np.ndarray, idx: np.ndarray) -> dict:
    return dict(est=float(x.mean()), **ci(x[idx].mean(axis=1)))


def band_label(st: dict, lvl: str, names=("SUFFICIENT", "NOT_LEVER", "PARTIAL")) -> str:
    lo, hi = st[f"lo{lvl}"], st[f"hi{lvl}"]
    if lo >= BAND_HI:
        return names[0]
    if hi <= BAND_LO:
        return names[1]
    if lo > BAND_LO and hi < BAND_HI:
        return names[2]
    return "UNRESOLVED"


def delta_label(st: dict, lvl: str) -> str:
    lo, hi = st[f"lo{lvl}"], st[f"hi{lvl}"]
    if lo >= -DELTA_BAND and hi <= DELTA_BAND:
        return "SIZE_EXPLAINS_L2"
    if hi < -DELTA_BAND:
        return "L2_BEYOND_SIZE"
    if lo > DELTA_BAND:
        return "CAP_BEYOND_L2"
    return "UNRESOLVED"


def analyze(pt: pd.DataFrame, provs: dict) -> dict:
    seeds, why = valid_seeds(pt, provs) if len(pt) else ([], {s: "no data" for s in SEEDS})
    res = {"valid_seeds": seeds, "invalid_seeds": {str(k): v for k, v in why.items()}, "n_valid": len(seeds),
           "labels": {}, "stats": {}, "tables": {}}
    if len(seeds) < MIN_SEEDS:
        for q in ("Q1", "Q2", "Q2_rho_cap2", "Q3"):
            res["labels"][q] = {"95": ("INCOMPLETE", f"{len(seeds)} valid seeds < {MIN_SEEDS}"),
                                "97.5": ("INCOMPLETE", f"{len(seeds)} valid seeds < {MIN_SEEDS}")}
        return res
    idx = boot_idx(len(seeds))
    T = {f"{a}_{m}": per_seed(pt, provs, a, m, seeds) for a, m in ARMS8}
    res["tables"] = T
    st = res["stats"]
    for act in ("LR", "R"):
        Gref = T[f"{act}_ref"].G.to_numpy()
        for arm in [m for a, m in ARMS8 if a == act and m != "ref"]:
            q = T[f"{act}_{arm}"]
            st[f"rho|{act}_{arm}"] = removal(q.G.to_numpy(), Gref, idx)
            Pref = T[f"{act}_ref"].P.to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):   # REPORT only; P_ref ~ 0 in a collapsed ref
                st[f"pi|{act}_{arm}"] = {"est": float(1 - q.P.mean() / Pref.mean()),
                                         **ci(1 - q.P.to_numpy()[idx].mean(1) / Pref[idx].mean(1))}
        for arm in [m for a, m in ARMS8 if a == act]:
            q = T[f"{act}_{arm}"]
            for col in ("G", "P", "D", "slope", "Awin", "A1"):
                st[f"{col}|{act}_{arm}"] = mean_ci(q[col].to_numpy(), idx)
        st[f"gamma|{act}_ref"] = float(T[f"{act}_ref"].gamma.median())
        st[f"kappa|{act}_ref"] = float(T[f"{act}_ref"].kappa.median())
    GLR = T["LR_ref"].G.to_numpy()
    st["delta|LR_cap2-l2"] = delta_removal(T["LR_l2"].G.to_numpy(), T["LR_cap2"].G.to_numpy(), GLR, idx)
    st["delta|R_cap2-l2"] = delta_removal(T["R_l2"].G.to_numpy(), T["R_cap2"].G.to_numpy(), T["R_ref"].G.to_numpy(), idx)
    a, b = T["LR_l2init"], T["LR_l2"]
    st["decomp|LR_l2init-l2"] = {"win": mean_ci((a.Awin - b.Awin).to_numpy(), idx),
                                 "fresh": mean_ci((a.A1 - b.A1).to_numpy(), idx),
                                 "temporal": mean_ci((b.G - a.G).to_numpy(), idx)}
    st["tax|LR_cap2_A1"] = mean_ci((T["LR_cap2"].A1 - T["LR_ref"].A1).to_numpy(), idx)
    st["tax|R_cap2_A1"] = mean_ci((T["R_cap2"].A1 - T["R_ref"].A1).to_numpy(), idx)

    for lvl in ("95", "97.5"):
        k = lvl.replace(".", "")
        # Q1 (primary): LR capT1
        if st["gamma|LR_ref"] < GUARD:
            q1 = ("WEAK_MANIPULATION", f"gamma = {st['gamma|LR_ref']:.3f} < {GUARD}")
        else:
            lab = band_label(st["rho|LR_capT1"], k, ("ACCUMULATION_SUFFICIENT", "ACCUMULATION_NOT_LEVER",
                                                      "ACCUMULATION_PARTIAL"))
            q1 = (lab, f"rho = {st['rho|LR_capT1']['est']:.4f} [{st['rho|LR_capT1'][f'lo{k}']:.4f}, "
                       f"{st['rho|LR_capT1'][f'hi{k}']:.4f}]")
        # Q2: LR cap2 vs l2, and rho_cap2 itself
        if st["kappa|LR_ref"] < GUARD:
            q2 = ("WEAK_MANIPULATION", f"kappa = {st['kappa|LR_ref']:.3f} < {GUARD}")
            q2r = q2
        else:
            d = st["delta|LR_cap2-l2"]
            q2 = (delta_label(d, k), f"delta = {d['est']:+.4f} [{d[f'lo{k}']:+.4f}, {d[f'hi{k}']:+.4f}]")
            r = st["rho|LR_cap2"]
            q2r = (band_label(r, k), f"rho = {r['est']:.4f} [{r[f'lo{k}']:.4f}, {r[f'hi{k}']:.4f}]")
        # Q3: R cap2
        if st["kappa|R_ref"] < GUARD:
            q3 = ("WEAK_MANIPULATION", f"kappa_R = {st['kappa|R_ref']:.3f} < {GUARD}")
        else:
            r = st["rho|R_cap2"]
            q3 = (band_label(r, k, ("SIZE_RESCUES_RELU", "SIZE_NOT_LEVER", "SIZE_PARTIAL")),
                  f"rho = {r['est']:.4f} [{r[f'lo{k}']:.4f}, {r[f'hi{k}']:.4f}]")
        for q, v in (("Q1", q1), ("Q2", q2), ("Q2_rho_cap2", q2r), ("Q3", q3)):
            res["labels"].setdefault(q, {})[lvl] = v

    # task-1 identity capT1 vs ref (REPORT)
    same = [provs[("LR", "capT1", s)]["runs"][str(s)].get("task1_end_state_sha256")
            == provs[("LR", "ref", s)]["runs"][str(s)].get("task1_end_state_sha256") for s in seeds]
    acc_same = [float(T["LR_capT1"].A1.loc[s]) == float(T["LR_ref"].A1.loc[s]) for s in seeds]
    res["task1_identity"] = {"state_sha_equal": int(sum(same)), "online_acc_equal": int(sum(acc_same)), "n": len(seeds)}
    return res


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------

def verdict_rows(res: dict) -> list[dict]:
    rows = []
    for q, v in res["labels"].items():
        for lvl, (lab, reason) in v.items():
            rows.append({"kind": "label", "question": q, "level": lvl, "label": lab, "reason": reason,
                         "n_valid_seeds": res["n_valid"]})
    for key, s in res["stats"].items():
        if isinstance(s, dict) and "est" in s:
            rows.append({"kind": "stat", "question": key, "est": s["est"],
                         **{k: s.get(k) for k in ("lo95", "hi95", "lo975", "hi975")}, "n_valid_seeds": res["n_valid"]})
        elif isinstance(s, dict):
            for sub, t in s.items():
                rows.append({"kind": "stat", "question": f"{key}|{sub}", "est": t["est"],
                             **{k: t.get(k) for k in ("lo95", "hi95", "lo975", "hi975")},
                             "n_valid_seeds": res["n_valid"]})
        else:
            rows.append({"kind": "stat", "question": key, "est": s, "n_valid_seeds": res["n_valid"]})
    return rows


def pct(s: dict, sign: bool = True, k: str = "95") -> str:
    f = "{:+.2f}" if sign else "{:.2f}"
    return f"{f.format(100 * s['est'])} [{f.format(100 * s['lo' + k])}, {f.format(100 * s['hi' + k])}]"


def frac(s: dict, k: str = "95") -> str:
    return f"{s['est']:.4f} [{s['lo' + k]:.4f}, {s['hi' + k]:.4f}]"


def summary_md(res: dict, pt: pd.DataFrame, provs: dict, missing: list, checks: dict | None, env: dict) -> str:
    L = [f"# {RUN_ID} — 判定（Random Label MNIST・隠れ層のユニット別中心化ノルムの上限）", "",
         "> 自動生成: `analysis/wcap_rlmnist_0914/verdict.py`。spec: `specs/spec_wcap_rlmnist_0914.md`。数値はこのファイルと `verdict.csv` から転記する。", "",
         "## 0. 実行したもの", ""]
    hashes = sorted({p["git_hash"] for p in provs.values()})
    dirty = sorted({str(p.get("git_dirty_code")) for p in provs.values()})
    walls = [p["wall_clock_s"] for p in provs.values()]
    rss = [p.get("peak_rss_kb", 0) for p in provs.values()]
    L += [f"- run の commit: {', '.join('`' + h + '`' for h in hashes)}（run 時の code の未 commit 変更: {', '.join(dirty)}）",
          f"- 集計時の HEAD: `{env.get('head', '?')}`",
          f"- 環境: host {', '.join(sorted({p['hostname'] for p in provs.values()}))}・torch {', '.join(sorted({p['torch'] for p in provs.values()}))}・"
          f"python {', '.join(sorted({p['python'] for p in provs.values()}))}・スレッド {sorted({p['torch_num_threads'] for p in provs.values()})}",
          f"- shard: {len(provs)}/80・欠損 {len(missing)}・有効 seed {res['n_valid']}（無効: {res['invalid_seeds'] or 'なし'}）",
          f"- 1 run の壁時計 中央値 {np.median(walls) / 60:.1f} 分（最大 {max(walls) / 60:.1f} 分）・peak RSS 中央値 {np.median(rss) / 1024:.0f} MiB" if walls else "- 壁時計: なし",
          f"- 検査: `checks.json` all_pass = **{checks.get('all_pass') if checks else '（なし）'}**", ""]
    L += ["## 1. 結論（spec §5.2 の表を上から適用）", "", "| 問い | ラベル（95%） | 理由 | Bonferroni 版（97.5%） |", "|---|---|---|---|"]
    names = {"Q1": "Q1（主）LR capT1", "Q2": "Q2 LR cap2 対 l2", "Q2_rho_cap2": "Q2 付記 LR cap2 の ρ", "Q3": "Q3 R cap2"}
    for q, v in res["labels"].items():
        L.append(f"| {names[q]} | **{v['95'][0]}** | {v['95'][1]} | {v['97.5'][0]} |")
    L += ["", "限定: Random Label MNIST・784–100–100–10・Adam lr=1e−3・400 epoch・50 タスク・seed 0–9・white-san CPU。G は画像固定による正の転移を含む正味。", ""]
    if res["n_valid"] < MIN_SEEDS:
        return "\n".join(L) + "\n"
    st, T = res["stats"], res["tables"]
    L += ["## 2. 時間劣化と除去率（pt・[ ] は seed 対応 bootstrap 95% CI）", "",
          "| 腕 | A(1) | A(31–50) | G = A(1)−A(31–50) | ρ = 1 − G/G_ref | P = A(11–20)−A(41–50) | π = 1 − P/P_ref | D = A(2–6)−A(31–50) | 傾き 11–50 /10 タスク |",
          "|---|---|---|---|---|---|---|---|---|"]
    for act, arm in ARMS8:
        k = f"{act}_{arm}"
        rho = frac(st[f"rho|{k}"]) if f"rho|{k}" in st else "—"
        pi = frac(st[f"pi|{k}"]) if f"pi|{k}" in st else "—"
        L.append(f"| {k} | {pct(st[f'A1|{k}'], False)} | {pct(st[f'Awin|{k}'], False)} | {pct(st[f'G|{k}'])} | {rho} | "
                 f"{pct(st[f'P|{k}'])} | {pi} | {pct(st[f'D|{k}'])} | {pct(st[f'slope|{k}'])} |")
    d = st["delta|LR_cap2-l2"]
    dr = st["delta|R_cap2-l2"]
    L += ["", f"- Δ = ρ_cap2 − ρ_l2（LR）= {frac(d)}・97.5% [{d['lo975']:.4f}, {d['hi975']:.4f}]",
          f"- Δ_R = ρ_cap2 − ρ_l2（R・REPORT）= {frac(dr)}",
          f"- ガード: γ（LR ref の median‖W̃1‖ t50/t1 の seed 中央値）= **{st['gamma|LR_ref']:.3f}**・κ（LR）= **{st['kappa|LR_ref']:.3f}**・κ（R）= **{st['kappa|R_ref']:.3f}**（いずれも ≥ 2 が必要）",
          f"- 上限腕の fresh の税 A(1) − A_ref(1): LR cap2 {pct(st['tax|LR_cap2_A1'])}・R cap2 {pct(st['tax|R_cap2_A1'])}",
          f"- REPORT LR l2init − l2: 窓 31–50 の差 {pct(st['decomp|LR_l2init-l2']['win'])} = fresh 分 {pct(st['decomp|LR_l2init-l2']['fresh'])} + 時間分 {pct(st['decomp|LR_l2init-l2']['temporal'])}",
          f"- task 1 の同一性（LR capT1 対 ref・REPORT）: 状態 sha256 一致 {res['task1_identity']['state_sha_equal']}/{res['task1_identity']['n']}・A(1) 一致 {res['task1_identity']['online_acc_equal']}/{res['task1_identity']['n']}", ""]
    L += ["### 2.1 ρ の seed 別の値", "", "| 腕 | " + " | ".join(f"s{s}" for s in res["valid_seeds"]) + " |",
          "|---|" + "---|" * res["n_valid"]]
    for act, arm in ARMS8:
        if arm != "ref":
            L.append(f"| {act}_{arm} | " + " | ".join(f"{v:.3f}" for v in st[f"rho|{act}_{arm}"]["per_seed"]) + " |")
    L += ["", "## 3. 軌道（online_acc の seed 平均・pt）", "", "| 腕 | " + " | ".join(f"t{t}" for t in TRAJ_TASKS) + " |",
          "|---|" + "---|" * len(TRAJ_TASKS)]
    ptv = pt[pt.seed.isin(res["valid_seeds"])]
    for act, arm in ARMS8:
        q = ptv[(ptv.act == act) & (ptv.arm == arm)].groupby("task").online_acc.mean()
        L.append(f"| {act}_{arm} | " + " | ".join(f"{100 * q.loc[t]:.2f}" for t in TRAJ_TASKS) + " |")
    L += ["", "## 4. 状態（seed 中央値）", "",
          "| 腕 | median‖W̃1_i‖ t0 → t1 → t10 → t50 | 窓 31–50 | median‖W̃2_i‖ t0 → t1 → t50 | bind_frac l1/l2（t50） | proj_rows l1/l2（t50 のタスク） | mob1 t1 → t50 | zbar1 t1 → t50 | dead1 t1 → t50 | memo 窓 31–50 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for act, arm in ARMS8:
        q = T[f"{act}_{arm}"].median()
        g = ptv[(ptv.act == act) & (ptv.arm == arm)]
        g1, g50 = g[g.task == 1].median(numeric_only=True), g[g.task == N_TASKS].median(numeric_only=True)
        bind = f"{g50.bind_frac_l1:.3f} / {g50.bind_frac_l2:.3f}" if arm.startswith("cap") else "—"
        proj = f"{g50.proj_rows_l1:.0f} / {g50.proj_rows_l2:.0f}" if arm.startswith("cap") else "—"
        L.append(f"| {act}_{arm} | {q.wt1_t0:.3f} → {q.wt1_t1:.3f} → {q.wt1_t10:.3f} → {q.wt1_t50:.3f} | {q.wt1_win:.3f} | "
                 f"{q.wt2_t0:.3f} → {q.wt2_t1:.3f} → {q.wt2_t50:.3f} | {bind} | {proj} | {g1.mob_l1:.3f} → {g50.mob_l1:.3f} | "
                 f"{g1.zbar_l1:.2f} → {g50.zbar_l1:.2f} | {g1.dead_frac_l1:.3f} → {g50.dead_frac_l1:.3f} | {100 * q.memo_win:.2f} |")
    ref = env.get("ref_0906", {})
    if ref:
        L += ["", "## 5. 0906（CUDA・別の走）の窓 31–50 平均 — 参考。bit 一致は求めず判定に使わない", "", "| 腕 | 0906 | 本走 |", "|---|---|---|"]
        for k, v in ref.items():
            L.append(f"| {k} | {100 * v:.2f} | {pct(st['Awin|' + k], False)} |")
    if checks:
        L += ["", "## 6. 検査（`checks.json`）", ""]
        for k, v in checks.items():
            if isinstance(v, dict) and "pass" in v:
                muts = v.get("mutations", [])
                ms = f"・mutation {sum(m['detected'] for m in muts)}/{len(muts)} 検出" if muts else ""
                L.append(f"- **{k}**: {'PASS' if v['pass'] else 'FAIL'}{ms}")
    return "\n".join(L) + "\n"


def ref_0906() -> dict:
    out = {}
    for k, arm in (("LR_ref", "LR"), ("LR_l2", "LR_l2"), ("R_ref", "R"), ("R_l2", "R_l2")):
        p = REPO / "results" / "pmnist_rlmnist_0906" / arm / "per_task.csv"
        if p.exists():
            d = pd.read_csv(p)
            out[k] = float(d[(d.task >= WIN[0]) & (d.task <= WIN[1])].groupby("seed").online_acc.mean().mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "results" / RUN_ID / "runs"))
    ap.add_argument("--out", default=str(REPO / "results" / RUN_ID))
    ap.add_argument("--checks", default=str(REPO / "results" / RUN_ID / "checks.json"))
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    real = src.resolve() == (REPO / "results" / RUN_ID / "runs").resolve()
    pt, lm, provs, missing = load(src)
    if real and missing:
        sys.exit(f"REFUSE: {len(missing)} shards missing in the real run directory; aggregate only after all 80 finish")
    res = analyze(pt, provs)
    out.mkdir(parents=True, exist_ok=True)
    pt.to_csv(out / "per_task.csv", index=False)
    lm.to_csv(out / "layer_metrics.csv", index=False)
    pd.DataFrame(verdict_rows(res)).to_csv(out / "verdict.csv", index=False)
    checks = json.loads(Path(args.checks).read_text()) if Path(args.checks).exists() else None
    if checks and Path(args.checks).resolve() != (out / "checks.json").resolve():
        shutil.copy(args.checks, out / "checks.json")
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    env = {"head": head, "ref_0906": ref_0906()}
    (out / "summary.md").write_text(summary_md(res, pt, provs, missing, checks, env))
    agg = {"run_id": RUN_ID, "aggregated_at": dt.datetime.now().astimezone().isoformat(), "head": head,
           "src": str(src), "n_shards": len(provs), "missing": missing, "valid_seeds": res["valid_seeds"],
           "invalid_seeds": res["invalid_seeds"],
           "judgement": {"window": list(WIN), "bands": [BAND_LO, BAND_HI], "delta_band": DELTA_BAND, "guard": GUARD,
                         "n_boot": N_BOOT, "rng_seed": RNG_SEED, "min_seeds": MIN_SEEDS},
           "labels": res["labels"],
           "shards": {shard_name(*k): {"git_hash": v["git_hash"], "hostname": v["hostname"],
                                       "wall_clock_s": v["wall_clock_s"], "peak_rss_kb": v.get("peak_rss_kb"),
                                       "code_sha256": v["code_sha256"], "git_dirty_code": v.get("git_dirty_code")}
                      for k, v in provs.items()}}
    (out / "provenance.json").write_text(json.dumps(agg, indent=2, default=str))
    for q, v in res["labels"].items():
        print(f"{q}: {v['95'][0]}  ({v['95'][1]})  | 97.5%: {v['97.5'][0]}")


if __name__ == "__main__":
    main()

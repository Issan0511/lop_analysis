#!/usr/bin/env python3
"""Registered judgements of mlp_center_x10_1020 (spec §6, §8, §7 S-reuse).  Fixed before the production run.

    python3 analysis/mlp_center_x10_1020/verdict.py --root results/mlp_center_x10_1020

The per-run windows, the exact two-sided sign test (zeros dropped), Holm and the A_WINS / B_WINS / TIE rule
are imported from the 0920 report, not re-written.  Writes verdict.json, paired_tests.csv and summary.md
into --root.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import cifar5p1_mlp_0920_report as RPT    # noqa: E402
from src import mlp_center_x10_1020 as M           # noqa: E402

ARMS = M.ARMS
FAMILY_M = (("CH", "R"), ("CHB0", "R"))
EARLY_HARD = (1, 3, 5, 7, 9)                         # "課題 1〜10" of spec §6.2 O
LATE_HARD = tuple(RPT.LATE)                          # 21, 23, 25, 27, 29
T_WIDTH = 30                                         # §6.2 W: end of task 30
T_RANK = 29                                          # §6.2 K: end of task 29
NEARLY_ZERO = 0.01                                   # §8 P3: one unit of 100 per layer


def cell(arm: str) -> str:
    return M.cell_name(arm)


def label(q: float, med: float) -> str:
    return "A_WINS" if q < 0.05 and med > 0 else "B_WINS" if q < 0.05 and med < 0 else "TIE"


def sign_row(name: str, v: np.ndarray) -> dict:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    pos, n, p = RPT.sign(v)
    return {"name": name, "n": n, "pos": pos, "neg": n - pos, "median": float(np.median(v)) if len(v) else float("nan"),
            "p": p}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load(root: Path) -> dict:
    out = {}
    for arm in ARMS:
        d = root / cell(arm)
        if not (d / "provenance.json").exists():
            raise SystemExit(f"{d} is unfinished (no provenance.json)")
        t = pd.read_csv(d / "per_task.csv")
        t["cell"], t["hidden"] = arm, 100
        z = np.load(d / "diag_units.npz")
        out[arm] = {"tasks": t, "fresh": pd.read_csv(d / "fresh_control.csv"),
                    "npz": {k: z[k] for k in z.files}, "diag": pd.read_csv(d / "diag_per_task.csv"),
                    "prov": json.loads((d / "provenance.json").read_text())}
    seeds = [list(map(int, out[a]["npz"]["seeds"])) for a in ARMS]
    if any(s != seeds[0] for s in seeds):
        raise SystemExit(f"the three cells hold different seeds: {seeds}")
    return out


def by_seed(D: dict, arm: str, fn) -> dict[int, float]:
    """{seed: fn(npz, slot)} over the arm's slots."""
    z = D[arm]["npz"]
    return {int(s): float(fn(z, r)) for r, s in enumerate(z["seeds"])}


def paired(a: dict, b: dict) -> np.ndarray:
    return np.array([a[s] - b[s] for s in sorted(a) if s in b and np.isfinite(a[s]) and np.isfinite(b[s])])


# --------------------------------------------------------------------------
# the quantities of spec §6.2
# --------------------------------------------------------------------------

def width_l2(z, r, t=T_WIDTH, src="probe"):
    return np.median(z[f"{src}_zsd_l2"][t - 1, r]) if z["live"][t - 1, r] else np.nan


def sparse_frac(z, r, tasks=LATE_HARD, layers=(1, 2), src="probe"):
    n = int(z[f"n_{src}"][tasks[0] - 1])
    vals = []
    for t in tasks:
        if not z["live"][t - 1, r]:
            return np.nan
        m = np.concatenate([M.pplus_bins(z[f"{src}_open_l{li}"][t - 1, r], n)["sparse"] for li in layers])
        vals.append(m.mean())
    return float(np.mean(vals))


def bin_fracs(z, r, tasks, li, src="probe"):
    n = int(z[f"n_{src}"][tasks[0] - 1])
    acc = {b: [] for b in M.BINS}
    for t in tasks:
        for b, m in M.pplus_bins(z[f"{src}_open_l{li}"][t - 1, r], n).items():
            acc[b].append(m.mean())
    return {b: float(np.mean(v)) for b, v in acc.items()}


def first_k(z, r, t, k):
    steps = int(z["steps"][t - 1])
    if not z["live"][t - 1, r] or k > steps:
        return np.nan
    if k == steps:
        return z["curve_final"][t - 1, r] / k
    return z["curve_cum"][t - 1, r, k // M.SNAP - 1] / k


def rank_l(z, r, t=T_RANK, li=2, src="probe"):
    return z[f"{src}_eff_rank_l{li}"][t - 1, r] if z["live"][t - 1, r] else np.nan


# --------------------------------------------------------------------------
# S-reuse (spec §7)
# --------------------------------------------------------------------------

def _rows(p: Path) -> list[dict]:
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def reuse(root: Path, host: Path, arm: str, npz: dict) -> dict:
    new_d, old_d = root / cell(arm), host / cell(arm)
    if not (old_d / "per_task.csv").exists():
        return {"label": "NO_HOST", "host": str(old_d)}
    res = {f: (new_d / f).read_bytes() == (old_d / f).read_bytes() for f in ("per_task.csv", "fresh_control.csv")}
    key = lambda q: (int(q["seed"]), int(q["task"]))
    got, want = sorted(_rows(new_d / "per_task.csv"), key=key), sorted(_rows(old_d / "per_task.csv"), key=key)
    same_keys = [key(q) for q in got] == [key(q) for q in want]
    cols = sorted({k for q in want for k in q} | {k for q in got for k in q})
    exact = [c for c in cols if not c.startswith("eff_rank")]
    mism = [(key(w), c) for g, w in zip(got, want) for c in exact if g.get(c) != w.get(c)] if same_keys else ["keys"]
    traj = same_keys and not mism and res["fresh_control.csv"]
    slot = {int(s): r for r, s in enumerate(npz["seeds"])}
    er_out, er_max_rel, n_er = 0, 0.0, 0
    if same_keys:
        for g, w in zip(got, want):
            s, t = key(w)
            for li in (1, 2):
                c = f"eff_rank_l{li}"
                if g.get(c) in (None, "") or w.get(c) in (None, ""):
                    continue
                a, b = float(g[c]), float(w[c])
                bound = npz[f"task_eff_rank_b64_l{li}"][t - 1, slot[s]] + M.print_halfunit(a) + M.print_halfunit(b)
                n_er += 1
                er_out += int(abs(a - b) > bound)
                er_max_rel = max(er_max_rel, abs(a - b) / max(abs(b), 1e-300))
    first = mism[0] if mism else None
    return {"label": "PASS" if traj and er_out == 0 else "FAIL", "bytes_equal": res, "trajectory_equal": traj,
            "n_mismatched_cells_without_eff_rank": len(mism) if same_keys else None,
            "first_mismatch": list(first) if isinstance(first, tuple) else first,
            "first_mismatch_task": (first[0][1] if isinstance(first, tuple) else None),
            "eff_rank_rows": n_er, "eff_rank_outside_b64": er_out, "eff_rank_max_rel_diff": er_max_rel,
            "host": str(old_d)}


def host_window(host: Path, arm: str) -> pd.DataFrame | None:
    f = host / cell(arm) / "per_task.csv"
    if not f.exists():
        return None
    t = pd.read_csv(f)
    t["cell"], t["hidden"] = arm, 100
    return RPT.per_run(t)


# --------------------------------------------------------------------------
# compute
# --------------------------------------------------------------------------

def compute(root: Path, host: Path | None = None) -> dict:
    D = load(root)
    r = pd.concat([RPT.per_run(D[a]["tasks"]) for a in ARMS], ignore_index=True)
    gap = {a: D[a]["fresh"].set_index("seed").fresh_gap for a in ARMS}
    r["fresh_gap"] = [gap[c].get(s, np.nan) for c, s in zip(r.cell, r.seed)]

    # ---- §6.1 family M
    fam = [RPT.pair(r, a, b) for a, b in FAMILY_M]
    for t_, q in zip(fam, RPT.holm([t_["p"] for t_ in fam])):
        t_["p_holm"], t_["verdict"] = q, label(q, t_["median_diff"])
    ch_chb0 = RPT.pair(r, "CH", "CHB0")
    lab = {f"{t_['a']}-{t_['b']}": t_["verdict"] for t_ in fam}
    worse = [v == "B_WINS" for v in lab.values()]
    overall = ("DOSE" if all(worse) and ch_chb0["median_diff"] > 0 else "BOTH_WORSE" if all(worse) else
               "ONE_WORSE" if sum(worse) == 1 else "NOT_WORSE")

    # ---- §6.2 W
    W = {a: by_seed(D, a, width_l2) for a in ARMS}
    W_task = {a: by_seed(D, a, lambda z, q: width_l2(z, q, src="task")) for a in ARMS}
    w_rep = {"seed_median_probe": {a: float(np.nanmedian(list(W[a].values()))) for a in ARMS},
             "seed_median_task": {a: float(np.nanmedian(list(W_task[a].values()))) for a in ARMS},
             "CHB0_ge_2R": int(sum(W["CHB0"][s] >= 2 * W["R"][s] for s in W["R"])),
             "CH_ge_2R": int(sum(W["CH"][s] >= 2 * W["R"][s] for s in W["R"])),
             "n": len(W["R"]), "per_seed": W}

    # ---- §6.2 D
    Dsp = {a: by_seed(D, a, sparse_frac) for a in ARMS}
    d_test = sign_row("sparse R - CHB0", paired(Dsp["R"], Dsp["CHB0"]))
    d_rep = {"seed_median": {a: float(np.nanmedian(list(Dsp[a].values()))) for a in ARMS},
             "R_gt_CHB0": int(sum(Dsp["R"][s] > Dsp["CHB0"][s] for s in Dsp["R"])), "test_R_minus_CHB0": d_test,
             "test_R_minus_CH": sign_row("sparse R - CH", paired(Dsp["R"], Dsp["CH"])),
             "per_layer_late": {a: {f"l{li}": float(np.nanmedian(list(by_seed(
                 D, a, lambda z, q, li=li: sparse_frac(z, q, layers=(li,))).values()))) for li in (1, 2)} for a in ARMS},
             "t30": {a: float(np.nanmedian(list(by_seed(D, a, lambda z, q: sparse_frac(z, q, tasks=(30,))).values())))
                     for a in ARMS},
             "task_data_late": {a: float(np.nanmedian(list(by_seed(
                 D, a, lambda z, q: sparse_frac(z, q, src="task")).values()))) for a in ARMS},
             "bins_late_probe": {a: {f"l{li}": {b: float(np.nanmedian([bin_fracs(D[a]["npz"], q, LATE_HARD, li)[b]
                                                                     for q in range(len(D[a]["npz"]["seeds"]))]))
                                                for b in M.BINS} for li in (1, 2)} for a in ARMS},
             "per_seed": Dsp}

    # ---- §6.2 O
    F780 = {a: {t: by_seed(D, a, lambda z, q, t=t: first_k(z, q, t, 780)) for t in range(1, 31)} for a in ARMS}

    def block(a, b, tasks):
        seeds = sorted(F780[a][tasks[0]])
        return {s: float(np.mean([F780[a][t][s] - F780[b][t][s] for t in tasks])) for s in seeds}

    o_rep = {}
    for a in ("CH", "CHB0"):
        e, l_ = block(a, "R", EARLY_HARD), block(a, "R", LATE_HARD)
        delta = paired(l_, e)
        t_d = sign_row(f"{a}-R first780: late - early", delta)
        lbl = ("OPENS" if t_d["p"] < 0.05 and t_d["median"] < 0 else
               "CLOSES" if t_d["p"] < 0.05 and t_d["median"] > 0 else "FLAT")
        o_rep[f"{a}-R"] = {"label": lbl, "test_delta": t_d,
                           "test_early": sign_row(f"{a}-R first780 early", np.array(list(e.values()))),
                           "test_late": sign_row(f"{a}-R first780 late", np.array(list(l_.values()))),
                           "early_per_seed": e, "late_per_seed": l_}
    curves = {a: {k: {t: float(np.nanmedian(list(by_seed(D, a, lambda z, q, t=t, k=k: first_k(z, q, t, k)).values())))
                      for t in range(1, 31)} for k in (100, 780)} for a in ARMS}

    # ---- §6.2 K
    K = {a: by_seed(D, a, rank_l) for a in ARMS}
    k_rep = {"seed_median": {a: float(np.nanmedian(list(K[a].values()))) for a in ARMS},
             "test_CH_minus_R": sign_row("eff_rank_l2 probe t29 CH - R", paired(K["CH"], K["R"])),
             "test_CHB0_minus_R": sign_row("eff_rank_l2 probe t29 CHB0 - R", paired(K["CHB0"], K["R"])),
             "l1_t29": {a: float(np.nanmedian(list(by_seed(D, a, lambda z, q: rank_l(z, q, li=1)).values())))
                        for a in ARMS},
             "l2_t30": {a: float(np.nanmedian(list(by_seed(D, a, lambda z, q: rank_l(z, q, t=30)).values())))
                        for a in ARMS}}

    # ---- descriptive table
    table = []
    for a in ARMS:
        g = r[r.cell == a]
        z = D[a]["npz"]
        table.append({"arm": a, "window": float(g.window.median()), "early": float(g.early.median()),
                      "drop": float(g["drop"].median()), "train_end": float(g.train_end.median()),
                      "test_end": float(g.test_end.median()), "fresh_gap": float(g.fresh_gap.median()),
                      "fresh_gap_pos": int((g.fresh_gap > 0).sum()), "diverged": int(g.diverged.sum()),
                      "first100_late": float(np.nanmedian([np.mean([first_k(z, q, t, 100) for t in LATE_HARD])
                                                           for q in range(len(z["seeds"]))])),
                      "first780_late": float(np.nanmedian([np.mean([first_k(z, q, t, 780) for t in LATE_HARD])
                                                           for q in range(len(z["seeds"]))])),
                      "b2_absmed_t30": float(np.nanmedian([np.median(np.abs(z["b2"][29, q])) for q in range(len(z["seeds"]))])),
                      "wnorm2_med_t30": float(np.nanmedian([np.median(z["wnorm2"][29, q]) for q in range(len(z["seeds"]))]))})
    sigma_traj = {a: {li: [float(np.nanmedian([np.median(D[a]["npz"][f"probe_zsd_l{li}"][t, q])
                                               for q in range(len(D[a]["npz"]["seeds"]))])) for t in range(30)]
                      for li in (1, 2)} for a in ARMS}

    # ---- §8 predictions
    p1_ch = paired({s: v for s, v in zip(r[r.cell == "CH"].seed, r[r.cell == "CH"].window)},
                   {s: v for s, v in zip(r[r.cell == "R"].seed, r[r.cell == "R"].window)})
    p1_chb0 = paired({s: v for s, v in zip(r[r.cell == "CHB0"].seed, r[r.cell == "CHB0"].window)},
                     {s: v for s, v in zip(r[r.cell == "R"].seed, r[r.cell == "R"].window)})
    wm = w_rep["seed_median_probe"]
    preds = {
        "P1": bool((p1_chb0 < 0).sum() == 10 and (p1_ch < 0).sum() >= 9 and ch_chb0["median_diff"] > 0),
        "P2": bool(wm["CHB0"] > wm["CH"] > wm["R"] and w_rep["CHB0_ge_2R"] >= 8),
        "P3": bool(d_test["p"] < 0.05 and d_test["median"] > 0 and d_rep["seed_median"]["CHB0"] <= NEARLY_ZERO),
        "P4": bool(o_rep["CH-R"]["label"] == "OPENS" and o_rep["CH-R"]["test_late"]["p"] < 0.05
                   and o_rep["CH-R"]["test_late"]["median"] < 0),
        "P5": bool(k_rep["seed_median"]["CH"] > k_rep["seed_median"]["R"]
                   and k_rep["seed_median"]["CHB0"] > k_rep["seed_median"]["R"])}

    out = {"overall": overall, "family_M": fam, "CH_minus_CHB0": ch_chb0, "W": w_rep, "D": d_rep, "O": o_rep,
           "K": k_rep, "table": table, "curves_first100_first780": curves, "sigma_probe_traj": sigma_traj,
           "predictions_claude_fable": preds,
           "torch": {a: D[a]["prov"].get("torch") for a in ARMS},
           "git": {a: D[a]["prov"].get("git_hash") for a in ARMS}}
    if host is not None:
        out["S_reuse"] = {a: reuse(root, host, a, D[a]["npz"]) for a in ("R", "CH")}
        old = {a: host_window(host, a) for a in ("R", "CH")}
        if old["R"] is not None and old["CH"] is not None:
            both = pd.concat([old["R"], old["CH"]], ignore_index=True)
            out["old_x10_CH_minus_R"] = RPT.pair(both, "CH", "R")
    return out


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

def f(v, spec=".4f"):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, spec)


def summary(V: dict) -> str:
    L = ["# mlp_center_x10_1020 の判定", "",
         "spec: `specs/spec_mlp_center_x10_1020.md`。5+1 CIFAR MLP、std、lr 1e−4、課題あたり 7,800 更新、seed 0〜9。", "",
         f"## 総合: `{V['overall']}`", "",
         "| 比較（後期窓） | 中央値の差 | a が上の seed | n | p | p_holm | 判定 |", "|---|---|---|---|---|---|---|"]
    for t in V["family_M"]:
        L.append(f"| {t['a']} − {t['b']} | {f(t['median_diff'], '+.4f')} | {t['a_wins']} | {t['n']} | "
                 f"{f(t['p'])} | {f(t['p_holm'])} | {t['verdict']} |")
    c = V["CH_minus_CHB0"]
    L += [f"| CH − CHB0（記述） | {f(c['median_diff'], '+.4f')} | {c['a_wins']} | {c['n']} | {f(c['p'])} | — | — |", "",
          "## 腕ごと（seed の中央値）", "",
          "| 腕 | 後期窓 | 早期窓 | 低下 | train 末 | test 末 | fresh gap | gap > 0 | 最初の 100（後期） | 最初の 780（後期） | |b2| 中央値 t30 | W2 行ノルム t30 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in V["table"]:
        L.append(f"| {t['arm']} | {f(t['window'])} | {f(t['early'])} | {f(t['drop'], '+.4f')} | {f(t['train_end'])} | "
                 f"{f(t['test_end'])} | {f(t['fresh_gap'], '+.4f')} | {t['fresh_gap_pos']}/10 | {f(t['first100_late'])} | "
                 f"{f(t['first780_late'])} | {f(t['b2_absmed_t30'], '.3f')} | {f(t['wnorm2_med_t30'], '.3f')} |")
    W, Dd, O, K = V["W"], V["D"], V["O"], V["K"]
    L += ["", "## 副の判定", "",
          f"**W（第 2 層の幅、probe、課題 30 の末）**: seed 中央値 R {f(W['seed_median_probe']['R'], '.2f')}・"
          f"CH {f(W['seed_median_probe']['CH'], '.2f')}・CHB0 {f(W['seed_median_probe']['CHB0'], '.2f')}。"
          f"CHB0 ≥ 2 × R は {W['CHB0_ge_2R']}/{W['n']} seed、CH ≥ 2 × R は {W['CH_ge_2R']}/{W['n']} seed。"
          f"課題の訓練画像の上では R {f(W['seed_median_task']['R'], '.2f')}・CH {f(W['seed_median_task']['CH'], '.2f')}・"
          f"CHB0 {f(W['seed_median_task']['CHB0'], '.2f')}。", "",
          f"**D（まばらな検出器の割合、probe、2 層まとめ、後期窓の課題末の平均）**: seed 中央値 "
          f"R {f(Dd['seed_median']['R'], '.3f')}・CH {f(Dd['seed_median']['CH'], '.3f')}・CHB0 {f(Dd['seed_median']['CHB0'], '.3f')}。"
          f"R > CHB0 は {Dd['R_gt_CHB0']}/10 seed（符号検定 p = {f(Dd['test_R_minus_CHB0']['p'])}）。", ""]
    for k, o in O.items():
        L.append(f"**O（最初の 780 更新の online の差 {k}）**: 早期 {f(o['test_early']['median'], '+.4f')}、"
                 f"後期 {f(o['test_late']['median'], '+.4f')}（後期の符号検定 p = {f(o['test_late']['p'])}）、"
                 f"後期 − 早期 {f(o['test_delta']['median'], '+.4f')}（p = {f(o['test_delta']['p'])}）→ `{o['label']}`。")
    L += ["", f"**K（実効階数、probe、第 2 層、課題 29 の末）**: seed 中央値 R {f(K['seed_median']['R'], '.2f')}・"
          f"CH {f(K['seed_median']['CH'], '.2f')}・CHB0 {f(K['seed_median']['CHB0'], '.2f')}。"
          f"CH − R の符号検定 p = {f(K['test_CH_minus_R']['p'])}（CH が上 {K['test_CH_minus_R']['pos']}/{K['test_CH_minus_R']['n']}）、"
          f"CHB0 − R p = {f(K['test_CHB0_minus_R']['p'])}（CHB0 が上 {K['test_CHB0_minus_R']['pos']}/{K['test_CHB0_minus_R']['n']}）。",
          "", "## 予測の採点（Claude（Fable）、§8 の規則）", ""]
    for k, v in V["predictions_claude_fable"].items():
        L.append(f"- {k}: {'的中' if v else '外れ'}")
    if "S_reuse" in V:
        L += ["", "## S-reuse（旧 10 倍走との照合）", ""]
        for a, s in V["S_reuse"].items():
            L.append(f"- {a}: `{s['label']}`。軌道（eff_rank 以外）一致 {s.get('trajectory_equal')}、"
                     f"最初の食い違い {s.get('first_mismatch')}、eff_rank の上界外 {s.get('eff_rank_outside_b64')}/"
                     f"{s.get('eff_rank_rows')}、相対差の最大 {f(s.get('eff_rank_max_rel_diff'), '.2e')}。")
        if "old_x10_CH_minus_R" in V:
            o = V["old_x10_CH_minus_R"]
            L.append(f"- 旧 10 倍走の CH − R（後期窓）: {f(o['median_diff'], '+.4f')}、CH が上 {o['a_wins']}/{o['n']}、p = {f(o['p'])}。")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "results" / M.EXPERIMENT))
    ap.add_argument("--host", default=str(ROOT / "results" / "cifar5p1_mlp_0920"))
    ap.add_argument("--no-host", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    V = compute(root, None if a.no_host else Path(a.host))
    (root / "verdict.json").write_text(json.dumps(V, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    rows = [dict(t) for t in V["family_M"]] + [dict(V["CH_minus_CHB0"], verdict="descriptive")]
    pd.DataFrame(rows).to_csv(root / "paired_tests.csv", index=False)
    (root / "summary.md").write_text(summary(V))
    print(summary(V))


if __name__ == "__main__":
    main()

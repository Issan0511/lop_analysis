#!/usr/bin/env python3
"""Verdict for neff_pred_0917 (specs/spec_neff_pred_0917.md section 5).

    python3 analysis/neff_pred_0917/verdict.py                          # the registered run
    python3 analysis/neff_pred_0917/verdict.py --src results/_checks_neff_pred_0917/synth_x --out /tmp/x

Reads <src>/gpu/{rows.csv, provenance.json}, <src>/ee/s<seed>/{prefix.csv, provenance.json} for the listed
seeds only, and the registered calibration (results/neff_pred_0917/calibration.json, whose sha256 the spec
records).  Refuses a run whose shards are missing or were not started at the registration commit.

A point is (box, seed, branch t): x = the box's predictor at the start of task t+1, E = task t+1's online
accuracy, E0 = the same seed's E at t = 2, F = clip((E - 0.1) / (E0 - 0.1), 0, 1).  Inside = F within the
calibration band at x.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis" / "neff_pred_0917"))
import calibrate as C                                              # noqa: E402

RUN_ID = "neff_pred_0917"
MAIN_SRC = REPO / "results" / RUN_ID
CAL_PATH = MAIN_SRC / "calibration.json"
EE_SEEDS = (10, 11, 12)
GPU_SEEDS = (0, 1, 2)
GPU_TASKS = 150
EE_TASKS = 51
BRANCH = {"RL": (2, 5, 10, 20, 50), "PM": (2, 5, 10, 20, 50, 100)}
CLASS = ("ELU1", "R", "GELU", "SILU")          # negative-side derivative vanishes; leaky (LR) is outside
PASS_SHARE = 0.8
CHANCE = 0.1
PRIMARY = {                                      # box -> (engine, env, act1, act2); "ee" is the CPU box
    "RL_EE_cpu": ("ee", "RL", "ELU1", "ELU1"),
    "RL_GELU": ("B", "RL", "GELU", "GELU"),
    "PM_GELU": ("B", "PM", "GELU", "GELU"),
    "RL_EL": ("L", "RL", "ELU1", "LR"),
}
OUT_OF_BOX = ("RL_GELU", "PM_GELU", "RL_EL")
EXT150 = ("fca473d", "results/relu_gelu_silu_rl_ext150_0914/rows.csv", 150)
CHIMERA = ("58c1819", "results/layer_chimera_rl_0914/rows.csv", 50)
LINK_COLS = ("online_acc", "online_ce", "train_acc", "train_ce")
PRED = {                                         # spec section 4.1 (Claude, this session)
    "label": ("BOX_SPECIFIC", 0.80),
    "pass": {"RL_EE_cpu": 0.85, "RL_GELU": 0.15, "PM_GELU": 0.55, "RL_EL": 0.45},
    "b": 0.45,
    "gelu_t50_below": 0.80,          # RL_GELU: F < band at t50 in >= 2 of 3 seeds
    "el_t50_above": 0.40,            # RL_EL:   F > band at t50 in >= 2 of 3 seeds
    "pm_timing_close": 0.55,         # PM_GELU: |T_x - T_F| <= 5 in >= 2 of 3 seeds
}
PRED_DESIGN = {"RL_EE_cpu": 0.85, "RL_GELU": 0.60, "PM_GELU": 0.55, "RL_EL": 0.40}   # the design note's


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def primary_layer(a1: str, a2: str) -> int | None:
    """The deepest hidden layer whose activation is in the class (design: EL -> layer 1)."""
    return 2 if a2 in CLASS else (1 if a1 in CLASS else None)


def need(n: int) -> int:
    return math.ceil(PASS_SHARE * n - 1e-12)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, float_precision="round_trip")


def load(src: Path, prereg: str | None) -> dict:
    """Shards and their provenance; problems are listed, not raised (the caller decides)."""
    problems = []
    g = src / "gpu"
    gpu = read_csv(g / "rows.csv") if (g / "rows.csv").exists() else None
    gprov = json.loads((g / "provenance.json").read_text()) if (g / "provenance.json").exists() else None
    if gpu is None or gprov is None:
        problems.append("gpu shard missing")
    else:
        if prereg is not None and gprov.get("prereg_commit") != prereg:
            problems.append(f"gpu prereg {gprov.get('prereg_commit')} != {prereg}")
        if gprov.get("git_dirty_code"):
            problems.append("gpu run started with uncommitted code")
        if int(gprov.get("tasks", 0)) != GPU_TASKS:
            problems.append(f"gpu tasks {gprov.get('tasks')} != {GPU_TASKS}")
    ee = {}
    for s in EE_SEEDS:
        d = src / "ee" / f"s{s}"
        if not ((d / "prefix.csv").exists() and (d / "provenance.json").exists()):
            problems.append(f"ee seed {s} missing")
            continue
        p = json.loads((d / "provenance.json").read_text())
        if prereg is not None and p.get("prereg_commit") != prereg:
            problems.append(f"ee s{s} prereg {p.get('prereg_commit')} != {prereg}")
        if p.get("git_dirty_code"):
            problems.append(f"ee s{s} started with uncommitted code")
        if int(p.get("tasks", 0)) != EE_TASKS or int(p.get("seed", -1)) != s:
            problems.append(f"ee s{s} tasks/seed mismatch")
        ee[s] = read_csv(d / "prefix.csv")
    return {"gpu": gpu, "gprov": gprov, "ee": ee, "problems": problems}


# --------------------------------------------------------------------------
# points
# --------------------------------------------------------------------------

def _clipF(e: float, e0: float) -> float:
    return min(max((e - CHANCE) / (e0 - CHANCE), 0.0), 1.0)


def gpu_points(gpu: pd.DataFrame, engine: str, env: str, a1: str, a2: str) -> pd.DataFrame:
    """All branch points t = 1..T-1 of one GPU box (every seed)."""
    d = gpu[(gpu.engine == engine) & (gpu.env == env) & (gpu.act1 == a1) & (gpu.act2 == a2)]
    rows = []
    for s, ds in d.groupby("seed"):
        ds = ds.set_index("task").sort_index()
        e0 = float(ds.loc[3, "online_acc"])
        for t in ds.index[:-1]:
            r = ds.loc[t + 1]
            row = {"seed": int(s), "t": int(t), "E": float(r["online_acc"]), "E0": e0,
                   "F": _clipF(float(r["online_acc"]), e0), "finite": bool(r["finite"])}
            for li in (1, 2):
                for k in ("neff_abs", "neff_sgn", "gabs", "zero", "tiny", "neg"):
                    row[f"{k}_l{li}"] = float(r[f"start_{k}_l{li}"])
            rows.append(row)
    return pd.DataFrame(rows)


def ee_points(ee: dict) -> pd.DataFrame:
    """The CPU box: resp_ee's prefix columns; RL, so the end of t is the start of t+1."""
    rows = []
    for s, p in ee.items():
        p = p.set_index("task").sort_index()
        e0 = float(p.loc[3, "online_acc"])
        for t in p.index[:-1]:
            e = float(p.loc[t + 1, "online_acc"])
            row = {"seed": int(s), "t": int(t), "E": e, "E0": e0, "F": _clipF(e, e0),
                   "finite": bool(np.isfinite(p.loc[t + 1, "online_ce"]))}
            for li in (1, 2):
                row[f"neff_abs_l{li}"] = float(p.loc[t, f"neffT{li}_end"])
                row[f"neff_sgn_l{li}"] = float(p.loc[t, f"neffT{li}_end"])      # ELU: g >= 0
                row[f"gabs_l{li}"] = float(p.loc[t, f"gtr{li}_end"])
                row[f"zero_l{li}"] = float(p.loc[t, f"zero{li}_end"])
            rows.append(row)
    return pd.DataFrame(rows)


def with_prediction(pts: pd.DataFrame, x: np.ndarray, cal: dict) -> pd.DataFrame:
    f, lo, hi = C.predict(x, cal)
    out = pts.copy()
    out["x"], out["f"], out["lo"], out["hi"] = x, f, lo, hi
    out["inside"] = (out["F"] >= out["lo"]) & (out["F"] <= out["hi"])
    out["side"] = np.where(out["inside"], "in", np.where(out["F"] < out["lo"], "below", "above"))
    return out


def predictor(pts: pd.DataFrame, rule: str, layer: int | None, x2_cal: float | None = None) -> np.ndarray:
    if rule == "primary":
        return pts[f"neff_abs_l{layer}"].to_numpy()
    if rule == "rel":                             # the seed's own t2 level mapped to the calibration's
        col = f"neff_abs_l{layer}"
        x2 = pts[pts.t == 2].set_index("seed")[col]
        return (pts[col] / pts["seed"].map(x2) * x2_cal).to_numpy()
    if rule == "min":
        return np.minimum(pts["neff_abs_l1"], pts["neff_abs_l2"]).to_numpy()
    if rule == "L2":
        return pts["neff_abs_l2"].to_numpy()
    if rule == "signed":
        return pts[f"neff_sgn_l{layer}"].to_numpy()
    raise ValueError(rule)


RULE_CAL = {"primary": "L2", "min": "min", "L2": "L2", "signed": "L2", "rel": "L2", "arms26": "arms26"}
LAYER_RULES = ("primary", "signed", "rel", "arms26")


def cal_x2(cal: dict) -> float:
    p = pd.DataFrame(cal["points"]["L2"])
    return float(p.loc[p.t == 2, "x"].mean())


def box_table(pts: pd.DataFrame, env: str, a1: str, a2: str, cal: dict, rule: str = "primary",
              branch_only: bool = True) -> pd.DataFrame | None:
    layer = primary_layer(a1, a2)
    if rule in LAYER_RULES and layer is None:
        return None
    use = pts[pts.t.isin(BRANCH[env])] if branch_only else pts
    x = predictor(use, "primary" if rule == "arms26" else rule, layer,
                  cal_x2(cal) if rule == "rel" else None)
    t = with_prediction(use, x, cal[RULE_CAL[rule]])
    t["layer"] = layer if rule in LAYER_RULES else rule
    if layer is not None:
        t["g_sel"] = t[f"gabs_l{layer}"]
        t["zero_sel"] = t[f"zero_l{layer}"]
    return t


def timing(full: pd.DataFrame) -> dict:
    """Per seed: first branch t with f(x) < 0.5 and first with F < 0.5 (report only)."""
    out = {}
    for s, d in full.groupby("seed"):
        d = d.sort_values("t")
        tx = d.loc[d.f < 0.5, "t"]
        tf = d.loc[d.F < 0.5, "t"]
        out[str(s)] = {"T_x": int(tx.iloc[0]) if len(tx) else None,
                       "T_F": int(tf.iloc[0]) if len(tf) else None,
                       "last_t": int(d.t.max())}
    return out


# --------------------------------------------------------------------------
# the registered analysis
# --------------------------------------------------------------------------

def analyze(sh: dict, cal: dict) -> dict:
    res = {"problems": list(sh["problems"])}
    boxes, tables, full = {}, {}, {}
    for name, (eng, env, a1, a2) in PRIMARY.items():
        if eng == "ee":
            pts = ee_points(sh["ee"]) if len(sh["ee"]) == len(EE_SEEDS) else None
        else:
            pts = gpu_points(sh["gpu"], eng, env, a1, a2) if sh["gpu"] is not None else None
        if pts is None or pts.empty:
            res["problems"].append(f"{name}: no points")
            continue
        tb = box_table(pts, env, a1, a2, cal)
        want = len(BRANCH[env]) * (len(EE_SEEDS) if eng == "ee" else len(GPU_SEEDS))
        if len(tb) != want:
            res["problems"].append(f"{name}: {len(tb)} branch points, expected {want}")
        if not tb["finite"].all():
            res["problems"].append(f"{name}: non-finite continuation at a branch point")
        n_in = int(tb["inside"].sum())
        boxes[name] = {"engine": eng, "env": env, "act1": a1, "act2": a2, "layer": primary_layer(a1, a2),
                       "n": int(len(tb)), "inside": n_in, "need": need(len(tb)),
                       "pass": n_in >= need(len(tb)),
                       "below": int((tb.side == "below").sum()), "above": int((tb.side == "above").sum())}
        tables[name] = tb.assign(box=name)
        full[name] = box_table(pts, env, a1, a2, cal, branch_only=False)
    res["boxes"] = boxes
    if res["problems"]:
        res["label"] = "INVALID"
        return res | {"tables": tables, "full": full}
    pooled = pd.concat([tables[b] for b in OUT_OF_BOX])
    rho = {"neff": spearman(pooled["F"], pooled["x"]), "gmean": spearman(pooled["F"], pooled["g_sel"]),
           "zero": spearman(pooled["F"], -pooled["zero_sel"])}
    beats = [not np.isfinite(rho[k]) or rho["neff"] > rho[k] for k in ("gmean", "zero")]
    res["b"] = {"rho": rho, "n": int(len(pooled)), "pass": bool(np.isfinite(rho["neff"]) and all(beats))}
    if not boxes["RL_EE_cpu"]["pass"]:
        res["label"] = "NOT_PREDICTIVE"
    elif all(boxes[b]["pass"] for b in OUT_OF_BOX):
        res["label"] = "GENERALIZES" if res["b"]["pass"] else "PREDICTS_NOT_BEST"
    else:
        res["label"] = "BOX_SPECIFIC"
    res["failing_boxes"] = [b for b in OUT_OF_BOX if not boxes[b]["pass"]]
    res["timing"] = {b: timing(full[b]) for b in PRIMARY}
    res["tables"], res["full"] = tables, full
    res["predictions"] = score_predictions(res, tables)
    return res


def score_predictions(res: dict, tables: dict) -> dict:
    sc = {}
    lab, p = PRED["label"]
    sc["label"] = {"pred": lab, "prob": p, "obs": res["label"], "hit": res["label"] == lab}
    for b, pb in PRED["pass"].items():
        obs = res["boxes"][b]["pass"]
        sc[f"pass_{b}"] = {"prob_pass": pb, "obs": obs, "hit": obs == (pb >= 0.5)}
    sc["b"] = {"prob_pass": PRED["b"], "obs": res["b"]["pass"], "hit": res["b"]["pass"] == (PRED["b"] >= 0.5)}
    g = tables["RL_GELU"]
    n_below = int(((g.t == 50) & (g.side == "below")).sum())
    sc["gelu_t50_below"] = {"prob": PRED["gelu_t50_below"], "obs": n_below,
                            "hit": (n_below >= 2) == (PRED["gelu_t50_below"] >= 0.5)}
    e = tables["RL_EL"]
    n_above = int(((e.t == 50) & (e.side == "above")).sum())
    sc["el_t50_above"] = {"prob": PRED["el_t50_above"], "obs": n_above,
                          "hit": (n_above >= 2) == (PRED["el_t50_above"] >= 0.5)}
    tm = res["timing"]["PM_GELU"]
    close = sum(1 for v in tm.values() if v["T_x"] is not None and v["T_F"] is not None
                and abs(v["T_x"] - v["T_F"]) <= 5)
    sc["pm_timing_close"] = {"prob": PRED["pm_timing_close"], "obs": close,
                             "hit": (close >= 2) == (PRED["pm_timing_close"] >= 0.5)}
    sc["design_note"] = {b: {"prob_pass": pb, "obs": res["boxes"][b]["pass"],
                             "hit": res["boxes"][b]["pass"] == (pb >= 0.5)} for b, pb in PRED_DESIGN.items()}
    sc["n_hit"] = sum(v["hit"] for k, v in sc.items() if k not in ("design_note", "n_hit", "n_total"))
    sc["n_total"] = sum(1 for k in sc if k not in ("design_note", "n_hit", "n_total"))
    return sc


# --------------------------------------------------------------------------
# report only
# --------------------------------------------------------------------------

def secondary(sh: dict, cal: dict) -> pd.DataFrame:
    """Every box x every rule: branch-point hit count, full-curve hit share, within-box Spearman."""
    rows = []
    combos = [("ee", "RL", "ELU1", "ELU1")]
    if sh["gpu"] is not None:
        combos += sorted({(r.engine, r.env, r.act1, r.act2) for r in
                          sh["gpu"][["engine", "env", "act1", "act2"]].drop_duplicates().itertuples()})
    for eng, env, a1, a2 in combos:
        pts = ee_points(sh["ee"]) if eng == "ee" else gpu_points(sh["gpu"], eng, env, a1, a2)
        if pts.empty:
            continue
        for rule in ("primary", "min", "L2", "signed", "rel", "arms26"):
            tb = box_table(pts, env, a1, a2, cal, rule)
            if tb is None:
                continue
            fb = box_table(pts, env, a1, a2, cal, rule, branch_only=False)
            rows.append({"engine": eng, "env": env, "act1": a1, "act2": a2, "rule": rule,
                         "layer": primary_layer(a1, a2), "n": len(tb), "inside": int(tb.inside.sum()),
                         "pass": int(tb.inside.sum()) >= need(len(tb)),
                         "below": int((tb.side == "below").sum()), "above": int((tb.side == "above").sum()),
                         "full_inside_share": float(fb.inside.mean()),
                         "rho_full_x": spearman(fb["F"], fb["x"]),
                         "F_t_last_branch_mean": float(tb[tb.t == max(BRANCH[env])]["F"].mean()),
                         "x_t2_mean": float(tb[tb.t == 2]["x"].mean()),
                         "x_t_last_branch_mean": float(tb[tb.t == max(BRANCH[env])]["x"].mean())})
    return pd.DataFrame(rows)


def link_check(gpu: pd.DataFrame) -> dict:
    """The GPU shard against the records it re-ran (report: LINKED / DIFFERS per engine)."""
    out = {}
    for eng, (commit, path, T) in (("B", EXT150), ("L", CHIMERA)):
        try:
            rec = pd.read_csv(io.StringIO(subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=REPO,
                                                                  text=True)), float_precision="round_trip")
        except Exception as e:                                     # noqa: BLE001
            out[eng] = {"flag": "NO_RECORD", "error": repr(e)[:200]}
            continue
        new = gpu[(gpu.engine == eng) & (gpu.task <= T)].copy()
        if eng == "B":
            rec = rec.rename(columns={"act": "act1"})
            rec["act2"] = rec["act1"]
        m = new.merge(rec, on=["seed", "env", "act1", "act2", "task"], suffixes=("", "_rec"))
        cols = [c for c in rec.columns if c not in ("seed", "env", "act1", "act2", "task") and c in new.columns]
        diff = {c: int((m[c] != m[c + "_rec"]).sum()) for c in cols}
        exp_n = len(rec[rec.task <= T])
        ok = len(m) == exp_n and all(v == 0 for v in diff.values())
        out[eng] = {"flag": "LINKED" if ok else "DIFFERS", "compared_rows": int(len(m)), "expected_rows": exp_n,
                    "columns": len(cols), "mismatches": {k: v for k, v in diff.items() if v}, "tasks": T}
    return out


def figure(res: dict, cal_all: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cal = cal_all["L2"]
    u = np.linspace(-6, 0.1, 400)
    f = C.logistic(u, cal["c"], cal["w"])
    q = np.asarray(cal["q"])[C.bin_of(f)]
    pts = pd.DataFrame(cal_all["points"]["L2"])
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for ax, b in zip(axes, PRIMARY):
        ax.fill_between(u, f - q, f + q, color="0.85", lw=0)
        ax.plot(u, f, color="0.3", lw=1)
        ax.scatter(C.u_of(pts.x), pts.F, s=5, color="0.6", label="resp_ee (fit)")
        if b in res["tables"]:
            fu = res["full"][b]
            for s, d in fu.groupby("seed"):
                ax.plot(C.u_of(d.x), d.F, lw=0.6, alpha=0.6)
            tb = res["tables"][b]
            ax.scatter(C.u_of(tb.x), tb.F, s=22, c=np.where(tb.inside, "tab:blue", "tab:red"), zorder=3)
        ax.set_title(f"{b} (layer {res['boxes'].get(b, {}).get('layer')})")
        ax.set_xlabel("log10 n_eff")
        ax.set_xlim(-6.2, 0.2)
    axes[0].set_ylabel("F")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summary_md(res: dict, link: dict | None, sec: pd.DataFrame | None, cal_all: dict) -> str:
    L = [f"# {RUN_ID} — 判定", ""]
    if res["label"] == "INVALID":
        return "\n".join(L + ["**INVALID**", ""] + [f"- {p}" for p in res["problems"]])
    cal = cal_all["L2"]
    L += [f"## 主判定: **{res['label']}**", "",
          f"較正（resp_ee の自然軌道・第2層）: 半減点 n_eff = {cal['x_half']:.4f}・w = {cal['w']:.4f}・"
          f"帯 ±{cal['q'][0]:.3f} / ±{cal['q'][1]:.3f} / ±{cal['q'][2]:.3f}", "",
          "| 箱 | 層 | 帯の中 | 必要 | 下に外れ | 上に外れ | 合否 |", "|---|---|---|---|---|---|---|"]
    for b, v in res["boxes"].items():
        L.append(f"| {b} | L{v['layer']} | {v['inside']}/{v['n']} | {v['need']} | {v['below']} | {v['above']} | "
                 f"{'PASS' if v['pass'] else 'FAIL'} |")
    r = res["b"]["rho"]
    L += ["", f"(b) 箱の外 {res['b']['n']} 点の順位相関: n_eff {r['neff']:+.3f}・微分の平均 {r['gmean']:+.3f}・"
          f"0 の割合（符号反転）{r['zero']:+.3f} → {'n_eff が最大' if res['b']['pass'] else 'n_eff は最大でない'}", ""]
    L += ["## 分岐点ごと（主の予測子）", "", "| 箱 | seed | t | x | f | 帯 | F | E | 判定 |", "|---|---|---|---|---|---|---|---|---|"]
    for b, tb in res["tables"].items():
        for rr in tb.sort_values(["seed", "t"]).itertuples():
            L.append(f"| {b} | {rr.seed} | {rr.t} | {rr.x:.3g} | {rr.f:.3f} | [{rr.lo:.3f}, {rr.hi:.3f}] | "
                     f"{rr.F:.3f} | {rr.E:.3f} | {rr.side} |")
    L += ["", "## 時刻（報告のみ）: f(x) < 0.5 と F < 0.5 の最初の分岐点", "",
          "| 箱 | seed | T_x | T_F |", "|---|---|---|---|"]
    for b, tm in res["timing"].items():
        for s, v in tm.items():
            L.append(f"| {b} | {s} | {v['T_x']} | {v['T_F']} |")
    sc = res["predictions"]
    L += ["", f"## 予測の照合（Claude・このセッション）: {sc['n_hit']}/{sc['n_total']}", ""]
    for k, v in sc.items():
        if k in ("n_hit", "n_total", "design_note"):
            continue
        L.append(f"- {k}: {json.dumps(v, ensure_ascii=False)}")
    L += ["", "設計案の予測（別セッション）: " + json.dumps(sc["design_note"], ensure_ascii=False), ""]
    if link is not None:
        L += ["## 記録との一致（報告）", ""] + [f"- {k}: {json.dumps(v, ensure_ascii=False)}" for k, v in link.items()] + [""]
    if sec is not None and len(sec):
        L += ["## 副（報告のみ）: 箱 × 予測子", "",
              "| engine | env | act1 | act2 | rule | 帯の中 | 合否 | 下 | 上 | 全 t の帯の中 | ρ(全 t) | x@t2 | x@最後 | F@最後 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for rr in sec.itertuples():
            L.append(f"| {rr.engine} | {rr.env} | {rr.act1} | {rr.act2} | {rr.rule} | {rr.inside}/{rr.n} | "
                     f"{'PASS' if rr.pass_ else 'FAIL'} | {rr.below} | {rr.above} | {rr.full_inside_share:.2f} | "
                     f"{rr.rho_full_x:+.2f} | {rr.x_t2_mean:.3g} | {rr.x_t_last_branch_mean:.3g} | "
                     f"{rr.F_t_last_branch_mean:.3f} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(MAIN_SRC))
    ap.add_argument("--out", default=None)
    ap.add_argument("--cal", default=str(CAL_PATH))
    ap.add_argument("--cal-sha256", default=None, help="required sha256 of the calibration (default: the spec's)")
    ap.add_argument("--prereg", default=None, help="required prereg_commit (default: the runner's)")
    ap.add_argument("--no-link", action="store_true")
    args = ap.parse_args(argv)
    src = Path(args.src)
    out = Path(args.out) if args.out else src
    prereg = args.prereg
    if prereg is None:
        txt = (REPO / "src" / "neff_pred_0917.py").read_text()
        prereg = next((ln.split('"')[1] for ln in txt.splitlines() if ln.startswith('PREREG_COMMIT = "')), None)
        if prereg is None:
            raise SystemExit("PREREG_COMMIT is not set in the runner (pass --prereg for synthetic runs)")
    prereg = None if prereg == "ANY" else prereg
    want_sha = args.cal_sha256
    if want_sha is None:
        spec = (REPO / "specs" / "spec_neff_pred_0917.md").read_text()
        want_sha = next((ln.split("`")[1] for ln in spec.splitlines() if ln.startswith("calibration.json sha256:")), None)
    cal_all = json.loads(Path(args.cal).read_text())
    if want_sha != "ANY" and sha256(Path(args.cal)) != want_sha:
        raise SystemExit(f"calibration sha256 {sha256(Path(args.cal))} != registered {want_sha}")
    sh = load(src, prereg)
    res = analyze(sh, cal_all)
    link = None if (args.no_link or sh["gpu"] is None) else link_check(sh["gpu"])
    sec = secondary(sh, cal_all) if res["label"] != "INVALID" else None
    if sec is not None:
        sec = sec.rename(columns={"pass": "pass_"})
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(summary_md(res, link, sec, cal_all))
    if res["label"] != "INVALID":
        pd.concat(res["tables"].values()).to_csv(out / "points.csv", index=False)
        pd.concat([v.assign(box=k) for k, v in res["full"].items()]).to_csv(out / "curves.csv", index=False)
        sec.rename(columns={"pass_": "pass"}).to_csv(out / "secondary.csv", index=False)
        pd.DataFrame([{"box": k, **v} for k, v in res["boxes"].items()]).to_csv(out / "verdict.csv", index=False)
        figure(res, cal_all, out / "fig_neff_pred_0917.png")
    slim = {k: v for k, v in res.items() if k not in ("tables", "full")}
    slim["link"] = link
    slim["calibration_sha256"] = sha256(Path(args.cal))
    (out / "verdict.json").write_text(json.dumps(slim, indent=2, default=str))
    print((out / "summary.md").read_text()[:3000])


if __name__ == "__main__":
    main()

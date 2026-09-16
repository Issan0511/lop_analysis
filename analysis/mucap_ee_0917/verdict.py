#!/usr/bin/env python3
"""Verdict for mucap_ee_0917 (specs/spec_mucap_ee_0917.md sections 4-5).

    python3 analysis/mucap_ee_0917/verdict.py [--src results/mucap_ee_0917/runs]

Reads the 40 shards, applies the registered endpoints and labels, and writes
results/mucap_ee_0917/{paired.csv, verdict.csv, secondary.csv, timing.csv, summary.md, provenance.json}.
analyze() is pure (shards in, results out) so that checks.py S8 can run it on synthetic shards.

Endpoints (spec 4.1), per seed, each minus its own task-1 value:
  E1 = online accuracy, arithmetic mean over the window's tasks;
  E2 = mean phi' of the SECOND layer: image mean -> arithmetic mean over all 100 units -> arithmetic
       mean over the window's task ends.
The arm effect is the within-seed difference to ref (task 1 is shared bit for bit, so it is also the
difference of the window levels).  Floor (spec 4.1, the parent layer_chimera_rl_0914 rule): a run is
AT_FLOOR when its window online accuracy is <= F + 3 s / sqrt(n), F and s (ddof 1) being the mean and SD
over the window's n tasks of the best constant predictor's accuracy on that task's labels.
The Student-t machinery (paired, t_quantile) is mucap_el_0916's, imported unchanged.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("_mucap_el_verdict", REPO / "analysis" / "mucap_el_0916" / "verdict.py")
ELV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ELV)
paired, t_quantile, t_cdf, task_ends, unit_mean = ELV.paired, ELV.t_quantile, ELV.t_cdf, ELV.task_ends, ELV.unit_mean

RUN_ID = "mucap_ee_0917"
ARMS = ("ref", "cap_par", "cap_perp", "cap_both")
CAP_ARMS = ARMS[1:]
MAIN_ARM = "cap_perp"                       # spec 5.3: the primary comparison is cap_perp - ref
CAPPED = {"cap_par": ("rows_par",), "cap_perp": ("rows_perp",), "cap_both": ("rows_par", "rows_perp")}
SEEDS = tuple(range(10))
N_TASKS = 100
SPT = 6000                                  # updates per task: the task-end diagnostic step
MAIN_WIN = (51, 100)                        # spec 4.1
FORM_WINS = ((2, 10), (11, 50))             # spec 4.1, reported only
CAP_WIN = (2, 10)                           # spec 5.2 (C): the cap must write in every task of this window
MIN_SEEDS = 8                               # spec 5.2 (A)
REF_FLOOR_FRAC = 0.8                        # spec 5.2 (B): ref AT_FLOOR in at least this share of valid seeds
FLOOR_K = 3.0                               # spec 4.1: F + 3 s / sqrt(n)
CHANCE = 0.1                                # 10 uniform labels: the fit fraction's zero (spec 4.2)
HALF = 0.5                                  # spec 4.2-1: T_half = first task whose fit fraction is < 1/2
SIGN_ALPHA = 0.05                           # spec 5.3 timing: two-sided exact sign test
EARLY = (2, 5)                              # spec 5.2 IMPAIRED
ABSORB_RUN = 5                              # spec 4.2: consecutive task ends with U < 0
MAIN_LEVEL = 0.975                          # two-sided, two main endpoints (Bonferroni)
COMP_LEVEL = 0.95                           # cap_par and cap_both
STREAM_KEYS = ("init_sha256", "subset_idx_sha256", "labels_sha256", "batch_sha256",
               "task1_end_state_sha256")
E2_KEY = "gate_mean_l2"
UNIT_Q = {"E2": E2_KEY, "g1": "gate_mean_l1", "pplus_l2": "pplus_l2", "zbar_l2": "zbar_l2",
          "d_l2": "d_l2", "sigma_l2": "sigma_l2", "d_l1": "d_l1", "sigma_l1": "sigma_l1",
          "v_norm_l1": "v_norm_l1", "q_l1": "q_l1", "b2": "b2", "row_norm_l2": "row_norm_l2"}


def shard_name(arm: str, seed: int) -> str:
    return f"{arm}_s{seed}"


# --------------------------------------------------------------------------
# per shard
# --------------------------------------------------------------------------

def online(pt: pd.DataFrame) -> pd.Series:
    return pt.set_index("task")["online_acc"].astype(float)


def e1_delta(pt: pd.DataFrame, win: tuple[int, int]) -> float:
    a = online(pt)
    return float(np.mean([a.loc[t] for t in range(win[0], win[1] + 1)])) - float(a.loc[1])


def unit_window_delta(u: dict, key: str, win: tuple[int, int]) -> tuple[float, int]:
    ends = task_ends(u)
    base, nans = unit_mean(u[key][ends[1]])
    vals = []
    for t in range(win[0], win[1] + 1):
        m, k = unit_mean(u[key][ends[t]])
        vals.append(m)
        nans += k
    return float(np.mean(vals)) - base, nans


def scalar_window(u: dict, key: str, win: tuple[int, int]) -> float:
    ends = task_ends(u)
    return float(np.mean([float(np.asarray(u[key][ends[t]]).ravel()[0]) for t in range(win[0], win[1] + 1)]))


def floor_call(pt: pd.DataFrame, win: tuple[int, int]) -> dict:
    p = pt.set_index("task")
    rows = p.loc[win[0]:win[1]]
    f = rows["major_frac"].astype(float).to_numpy()
    n = len(f)
    F, s = float(f.mean()), float(f.std(ddof=1))
    thr = F + FLOOR_K * s / math.sqrt(n)
    level = float(rows["online_acc"].astype(float).mean())
    return {"F": F, "s": s, "thr": thr, "level": level, "at_floor": level <= thr}


def fit(pt: pd.DataFrame) -> pd.Series:
    a = online(pt)
    return (a - CHANCE) / (a.loc[1] - CHANCE)


def t_half(pt: pd.DataFrame) -> int | None:
    """First task >= 2 whose fit fraction is below HALF; None (right-censored) if none in the run."""
    f = fit(pt)
    for t in range(2, N_TASKS + 1):
        if f.loc[t] < HALF:
            return t
    return None


def fit_early(pt: pd.DataFrame) -> float:
    f = fit(pt)
    return float(np.mean([f.loc[t] for t in range(EARLY[0], EARLY[1] + 1)]))


def sign_test(k: int, n: int) -> float:
    """Two-sided exact binomial(n, 1/2) p for the larger count k."""
    if n == 0:
        return 1.0
    k = max(k, n - k)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def compare_times(ta: int | None, tr: int | None) -> str:
    """The arm's T_half against ref's in the same seed; a censored time is later than any observed one."""
    if ta is None and tr is None:
        return "tie"
    if ta is None:
        return "later"
    if tr is None:
        return "earlier"
    return "earlier" if ta < tr else "later" if ta > tr else "tie"


def absorption(u: dict, key: str) -> dict:
    ends = task_ends(u)
    U = np.stack([np.asarray(u[key][ends[t]], dtype=np.float64) for t in range(1, N_TASKS + 1)])
    neg = U < 0
    first_run = []
    for i in range(U.shape[1]):
        col = neg[:, i]
        fr = None
        for s0 in range(0, N_TASKS - ABSORB_RUN + 1):
            if col[s0:s0 + ABSORB_RUN].all():
                fr = s0 + 1
                break
        first_run.append(fr)
    times = [x for x in first_run if x is not None]
    return {"frac_absorbed": len(times) / U.shape[1],
            "median_absorb_task": float(np.median(times)) if times else float("nan"),
            "frac_neg_at_end": float(neg[-1].mean()),
            "frac_ever_neg": float(neg.any(axis=0).mean())}


# --------------------------------------------------------------------------
# loading and validity
# --------------------------------------------------------------------------

def load(src: Path) -> tuple[dict, list]:
    shards, missing = {}, []
    for arm in ARMS:
        for seed in SEEDS:
            d = src / shard_name(arm, seed)
            if not (d / "provenance.json").exists():
                missing.append(shard_name(arm, seed))
                continue
            pre = f"s{seed}_"
            with np.load(d / "units.npz") as z:
                units = {k[len(pre):]: z[k] for k in z.files if k.startswith(pre)}
            ledger = {}
            if (d / "ledger.npz").exists():
                with np.load(d / "ledger.npz") as z:
                    ledger = {k[len(pre):]: z[k] for k in z.files if k.startswith(pre)}
            shards[(arm, seed)] = {"units": units, "per_task": pd.read_csv(d / "per_task.csv"),
                                   "prov": json.loads((d / "provenance.json").read_text()), "ledger": ledger}
    return shards, missing


def seed_validity(shards: dict, seed: int) -> str | None:
    """None if the seed is valid for spec 5.2 (A), else the reason."""
    infos = {}
    for arm in ARMS:
        sh = shards.get((arm, seed))
        if sh is None:
            return f"{arm} missing"
        info = sh["prov"]["per_seed"][str(seed)]
        if info.get("tasks_completed") != N_TASKS or info.get("divergence", {}).get("diverged"):
            return f"{arm} incomplete or diverged"
        pt = sh["per_task"]
        if sorted(pt["task"].astype(int)) != list(range(1, N_TASKS + 1)) or \
                not np.isfinite(pt["online_acc"].astype(float)).all():
            return f"{arm} per-task rows missing or non-finite"
        ends = task_ends(sh["units"])
        if sorted(ends) != list(range(1, N_TASKS + 1)):
            return f"{arm} task ends missing"
        g = np.stack([sh["units"][k][ends[t]] for t in ends for k in ("gate_mean_l1", E2_KEY)])
        if not np.isfinite(g).all():
            return f"{arm} non-finite response"
        infos[arm] = tuple(info.get(k) for k in STREAM_KEYS)
    if len(set(infos.values())) != 1:
        return "streams or task-1 state differ across arms"
    return None


# --------------------------------------------------------------------------

def analyze(shards: dict) -> dict:
    invalid = {s: r for s in SEEDS if (r := seed_validity(shards, s)) is not None}
    valid = [s for s in SEEDS if s not in invalid]
    res = {"valid_seeds": valid, "invalid_seeds": invalid, "rows": [], "secondary": [], "timing": [],
           "applicability": {}}

    D = {}
    for arm in ARMS:
        for s in valid:
            sh = shards[(arm, s)]
            for win in (MAIN_WIN, *FORM_WINS):
                D[(arm, s, "E1", win)] = (e1_delta(sh["per_task"], win), 0)
                D[(arm, s, "E2", win)] = unit_window_delta(sh["units"], E2_KEY, win)

    def deltas(arm, ep, win=MAIN_WIN):
        return np.array([D[(arm, s, ep, win)][0] for s in valid])

    def diffs(arm, ep, win=MAIN_WIN):
        return deltas(arm, ep, win) - deltas("ref", ep, win)

    floors = {(arm, s): floor_call(shards[(arm, s)]["per_task"], MAIN_WIN) for arm in ARMS for s in valid}
    state = {}
    for arm in ARMS:
        at = [floors[(arm, s)]["at_floor"] for s in valid]
        state[arm] = "FLOORED" if at and all(at) else "ALIVE" if at and not any(at) else "SPLIT"

    # (A) (B) (C)
    A_ok = len(valid) >= MIN_SEEDS
    ref_at = sum(floors[("ref", s)]["at_floor"] for s in valid)
    B_E2 = paired(deltas("ref", "E2"), 0.95)
    B_ok = A_ok and ref_at >= math.ceil(REF_FLOOR_FRAC * len(valid)) and B_E2["hi"] < 0
    C = {}
    for arm in CAP_ARMS:
        ok = True
        for s in valid:
            pt = shards[(arm, s)]["per_task"].set_index("task")
            w = pt.loc[CAP_WIN[0]:CAP_WIN[1]]
            for col in CAPPED[arm]:
                ok &= bool(len(w) == CAP_WIN[1] - CAP_WIN[0] + 1 and (w[col] > 0).all())
        C[arm] = ok
    res["applicability"] = {"A_valid_seeds": len(valid), "A_ok": A_ok, "B_ref_at_floor": ref_at,
                            "B_ref_E2": {k: B_E2[k] for k in ("mean", "lo", "hi")}, "B_ok": B_ok, "C": C}
    res["floor_state"] = state
    res["floors"] = {f"{a}|{s}": v for (a, s), v in floors.items()}

    impaired = {}
    for arm in CAP_ARMS:
        hits = 0
        for s in valid:
            f_ref = fit_early(shards[("ref", s)]["per_task"])
            f_arm = fit_early(shards[(arm, s)]["per_task"])
            hits += f_arm < f_ref - (1.0 - f_ref)
        impaired[arm] = {"seeds_below": hits, "flag": hits > len(valid) / 2}
    res["impaired"] = impaired

    def label(arm, st):
        if not A_ok or not C[arm]:
            return "INAPPLICABLE"
        if not B_ok:
            return "NOT_REPRODUCED"
        if state[arm] == "FLOORED":
            return "COLLAPSED"
        if state[arm] == "SPLIT":
            return "SPLIT"
        if st["E1"]["sign"] == "+":
            return "RESCUED" if st["E2"]["sign"] == "+" else "RESCUED_FUNCTION_ONLY"
        return "ALIVE_UNRESOLVED"

    labels = {}
    for arm in CAP_ARMS:
        for level in (MAIN_LEVEL, COMP_LEVEL):
            st = {ep: paired(diffs(arm, ep), level) for ep in ("E1", "E2")}
            lab = label(arm, st)
            labels[(arm, level)] = lab
            for ep, x in st.items():
                res["rows"].append({"arm": arm, "endpoint": ep, "window": f"{MAIN_WIN[0]}-{MAIN_WIN[1]}",
                                    "level": level, **{k: x[k] for k in ("n", "mean", "sd", "lo", "hi",
                                                                         "sign", "degenerate")},
                                    "floor_state": state[arm], "label": lab,
                                    "role": "main" if (arm == MAIN_ARM and level == MAIN_LEVEL) else
                                    ("component" if level == COMP_LEVEL and arm != MAIN_ARM else "report")})

    # timing (registered, per capped arm)
    th = {(arm, s): t_half(shards[(arm, s)]["per_task"]) for arm in ARMS for s in valid}
    timing = {}
    for arm in CAP_ARMS:
        cmp_ = [compare_times(th[(arm, s)], th[("ref", s)]) for s in valid]
        e, l_ = cmp_.count("earlier"), cmp_.count("later")
        p = sign_test(max(e, l_), e + l_)
        lab = ("EARLIER" if e > l_ else "LATER") if (e + l_ > 0 and p < SIGN_ALPHA) else "NO_TIMING_DIFF"
        if not A_ok or not C[arm]:
            lab = "INAPPLICABLE"
        timing[arm] = {"label": lab, "earlier": e, "later": l_, "ties": cmp_.count("tie"), "p": p}
        for s, c in zip(valid, cmp_):
            res["timing"].append({"arm": arm, "seed": s, "t_half_arm": th[(arm, s)],
                                  "t_half_ref": th[("ref", s)], "compare": c})
    res["timing_labels"] = timing
    res["t_half"] = {f"{a}|{s}": v for (a, s), v in th.items()}

    res["labels"] = {"main": labels[(MAIN_ARM, MAIN_LEVEL)],
                     "cap_perp_95": labels[(MAIN_ARM, COMP_LEVEL)],
                     "cap_par": labels[("cap_par", COMP_LEVEL)],
                     "cap_both": labels[("cap_both", COMP_LEVEL)],
                     **{f"timing_{a}": timing[a]["label"] for a in CAP_ARMS}}
    res["per_seed"] = {arm: {ep: diffs(arm, ep).tolist() for ep in ("E1", "E2")} for arm in CAP_ARMS}
    res["ref_delta"] = {ep: deltas("ref", ep).tolist() for ep in ("E1", "E2")}
    res["interaction"] = {ep: paired(diffs("cap_both", ep) - diffs("cap_par", ep) - diffs("cap_perp", ep),
                                     COMP_LEVEL) for ep in ("E1", "E2")}

    # secondary (never a verdict)
    wins = (MAIN_WIN, *FORM_WINS)
    for arm in ARMS:
        for s_ep in ("E1", "E2"):
            for win in wins:
                v = deltas(arm, s_ep, win)
                row = {"arm": arm, "quantity": f"delta_{s_ep}", "window": f"{win[0]}-{win[1]}",
                       "mean": float(v.mean()) if v.size else float("nan")}
                if arm != "ref":
                    p = paired(diffs(arm, s_ep, win), COMP_LEVEL)
                    row |= {"diff_mean": p["mean"], "diff_lo95": p["lo"], "diff_hi95": p["hi"],
                            "diff_pos_seeds": int((diffs(arm, s_ep, win) > 0).sum())}
                res["secondary"].append(row)
        for name, key in UNIT_Q.items():
            if name == "E2":
                continue
            for win in wins:
                lv = []
                for s in valid:
                    u = shards[(arm, s)]["units"]
                    if key in u:
                        ends = task_ends(u)
                        lv.append(float(np.mean([unit_mean(u[key][ends[t]])[0]
                                                 for t in range(win[0], win[1] + 1)])))
                if lv:
                    res["secondary"].append({"arm": arm, "quantity": f"level_{name}",
                                             "window": f"{win[0]}-{win[1]}", "mean": float(np.mean(lv))})
        for win in wins:                                   # who carries ||mu2||^2 (spec 4.2-4)
            parts = {"frac_q_pos": [], "frac_always_on": [], "mu2sq_share_q_pos": [], "mu2sq_share_always_on": []}
            for s in valid:
                u = shards[(arm, s)]["units"]
                if "a1mean_l1" not in u:
                    continue
                ends = task_ends(u)
                acc_ = {k: [] for k in parts}
                for t in range(win[0], win[1] + 1):
                    q, pp, m = (np.asarray(u[k][ends[t]], float) for k in ("q_l1", "pplus_l1", "a1mean_l1"))
                    tot = float((m ** 2).sum())
                    acc_["frac_q_pos"].append(float((q > 0).mean()))
                    acc_["frac_always_on"].append(float((pp == 1.0).mean()))
                    acc_["mu2sq_share_q_pos"].append(float((m[q > 0] ** 2).sum()) / tot if tot > 0 else float("nan"))
                    acc_["mu2sq_share_always_on"].append(float((m[pp == 1.0] ** 2).sum()) / tot if tot > 0 else float("nan"))
                for k in parts:
                    parts[k].append(float(np.nanmean(acc_[k])))
            for k, v in parts.items():
                if v:
                    res["secondary"].append({"arm": arm, "quantity": f"level_{k}", "window": f"{win[0]}-{win[1]}",
                                             "mean": float(np.nanmean(v))})
        for key in ("mu2_norm", "mu2_proj_sd", "s2_mean", "s2_sd"):
            for win in wins:
                lv = [scalar_window(shards[(arm, s)]["units"], key, win) for s in valid
                      if key in shards[(arm, s)]["units"]]
                if lv:
                    res["secondary"].append({"arm": arm, "quantity": f"level_{key}",
                                             "window": f"{win[0]}-{win[1]}", "mean": float(np.mean(lv))})
        for col in ("memo_acc", "online_acc"):
            for win in wins:
                lv = [float(shards[(arm, s)]["per_task"].set_index("task").loc[win[0]:win[1], col].mean())
                      for s in valid]
                res["secondary"].append({"arm": arm, "quantity": f"level_{col}",
                                         "window": f"{win[0]}-{win[1]}", "mean": float(np.mean(lv))})
        for layer, key in (("l1", "U_l1"), ("l2", "U_l2")):
            ab = [absorption(shards[(arm, s)]["units"], key) for s in valid]
            for k in ab[0] if ab else ():
                vals = np.array([a[k] for a in ab], dtype=np.float64)
                res["secondary"].append({"arm": arm, "quantity": f"absorb_{layer}_{k}", "window": f"1-{N_TASKS}",
                                         "mean": float(np.nanmean(vals)) if np.isfinite(vals).any()
                                         else float("nan")})
        ths = [th[(arm, s)] for s in valid]
        obs = [x for x in ths if x is not None]
        res["secondary"].append({"arm": arm, "quantity": "t_half_median_observed", "window": f"2-{N_TASKS}",
                                 "mean": float(np.median(obs)) if obs else float("nan"),
                                 "n_censored": len(ths) - len(obs)})
        res["secondary"].append({"arm": arm, "quantity": "seeds_at_floor", "window": f"{MAIN_WIN[0]}-{MAIN_WIN[1]}",
                                 "mean": float(sum(floors[(arm, s)]["at_floor"] for s in valid))})
        # a ledger-off shard still writes ledger.npz with its task column only (fixed after the run,
        # 0917: this line read such a file as a ledger and raised; labels never read the ledger)
        led = [shards[(arm, s)]["ledger"] for s in valid if "adam_q_align" in shards[(arm, s)]["ledger"]]
        for comp in ("q", "v2") if led else ():
            for part in ("adam", "proj"):
                for term in ("align", "sq"):
                    k = f"{part}_{comp}_{term}"
                    for win in wins:
                        tot = []
                        for L in led:
                            r_ = (L["task"] >= win[0]) & (L["task"] <= win[1])
                            tot.append(float(L[k][r_].mean(axis=1).sum()))
                        res["secondary"].append({"arm": arm, "quantity": f"ledger_{k}",
                                                 "window": f"{win[0]}-{win[1]}", "mean": float(np.mean(tot))})
    return res


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------

def _f(x, nd=4):
    return "nan" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:+.{nd}f}"


def summary_md(res: dict, missing: list, env: dict) -> str:
    ap = res["applicability"]
    n = len(res["valid_seeds"])
    L = [f"# {RUN_ID} — 判定（第1層の成分上限・ELU→ELU・{N_TASKS} タスク）", "",
         "> 自動生成: `analysis/mucap_ee_0917/verdict.py`。spec: `specs/spec_mucap_ee_0917.md`。数値は `verdict.csv`・"
         "`paired.csv`・`timing.csv`・`secondary.csv` から。", "",
         "## 0. 実行したもの", "",
         f"- 集計時の HEAD: `{env['head']}`・run の commit: {env['run_commits']}",
         f"- shard: {env['n_shards']}/40・欠損 {len(missing)}・有効 seed {n}（無効: {res['invalid_seeds'] or 'なし'}）", "",
         "## 1. 適用条件（spec §5.2）", "",
         f"- (A) 有効 seed {ap['A_valid_seeds']} ≥ {MIN_SEEDS}: **{ap['A_ok']}**",
         f"- (B) ref が主窓で床: {ap['B_ref_at_floor']}/{n} seed（要 ≥ {math.ceil(REF_FLOOR_FRAC * n)}）、"
         f"ref の第2層 ΔE2 {_f(ap['B_ref_E2']['mean'])} [{_f(ap['B_ref_E2']['lo'])}, {_f(ap['B_ref_E2']['hi'])}]"
         f" → **{ap['B_ok']}**",
         "- (C) 上限が task 2–10 の全タスクで書いた: " + "、".join(f"{a} **{v}**" for a, v in ap["C"].items()),
         "- IMPAIRED（読みのフラグ）: " + "、".join(
             f"{a} {v['seeds_below']}/{n} seed → **{v['flag']}**" for a, v in res["impaired"].items()),
         "- 主窓の床の状態: " + "、".join(f"{a} {v}" for a, v in res["floor_state"].items()),
         "", "## 2. 判定", "",
         "| 比較 | 水準 | ラベル | 時間（T_half） |", "|---|---|---|---|",
         f"| **主: cap_perp − ref** | 97.5% | **{res['labels']['main']}** | {res['labels']['timing_cap_perp']} |",
         f"| cap_perp − ref（参考） | 95% | {res['labels']['cap_perp_95']} | |",
         f"| cap_par − ref | 95% | {res['labels']['cap_par']} | {res['labels']['timing_cap_par']} |",
         f"| cap_both − ref | 95% | {res['labels']['cap_both']} | {res['labels']['timing_cap_both']} |", "",
         f"## 3. 腕間差（主窓 task {MAIN_WIN[0]}–{MAIN_WIN[1]}・seed 内対応差）", "",
         "| 腕 | endpoint | 水準 | 平均 | SD | 区間 | 符号 |", "|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        L.append(f"| {r['arm']} | {r['endpoint']} | {r['level']} | {_f(r['mean'])} | {_f(r['sd'])} | "
                 f"[{_f(r['lo'])}, {_f(r['hi'])}]{' (退化)' if r['degenerate'] else ''} | {r['sign']} |")
    L += ["", "交互作用 δ_both − δ_par − δ_perp（95%）: " + "、".join(
        f"{ep} {_f(x['mean'])} [{_f(x['lo'])}, {_f(x['hi'])}]" for ep, x in res["interaction"].items()), "",
          "### 3.1 seed 別の差と床", "", "| 腕 | endpoint | " + " | ".join(f"s{s}" for s in res["valid_seeds"]) + " |",
          "|---|---|" + "---|" * n]
    for arm, eps in res["per_seed"].items():
        for ep, v in eps.items():
            L.append(f"| {arm} | {ep} | " + " | ".join(_f(x, 3) for x in v) + " |")
    for arm in ARMS:
        L.append(f"| {arm} | 主窓 online（閾値） | " + " | ".join(
            f"{res['floors'][f'{arm}|{s}']['level']:.3f} ({res['floors'][f'{arm}|{s}']['thr']:.3f})"
            f"{' 床' if res['floors'][f'{arm}|{s}']['at_floor'] else ''}" for s in res["valid_seeds"]) + " |")
    L += ["", f"### 3.2 T_half（適合率 F が初めて 1/2 を下回るタスク・— は {N_TASKS} まで未到達）", "",
          "| 腕 | " + " | ".join(f"s{s}" for s in res["valid_seeds"]) + " | 早/遅/同 | p |", "|---|" + "---|" * (n + 2)]
    for arm in ARMS:
        tl = res["timing_labels"].get(arm)
        L.append(f"| {arm} | " + " | ".join(str(res['t_half'][f'{arm}|{s}'] or '—') for s in res["valid_seeds"])
                 + (f" | {tl['earlier']}/{tl['later']}/{tl['ties']} | {tl['p']:.3g} |" if tl else " | | |"))
    L += ["", "## 4. 副 endpoint（判定に使わない）", "", "| 腕 | 量 | 窓 | 値 | 差 [95%] |", "|---|---|---|---|---|"]
    for r in res["secondary"]:
        diff = (f"{_f(r['diff_mean'])} [{_f(r['diff_lo95'])}, {_f(r['diff_hi95'])}] ({r['diff_pos_seeds']}/{n} 正)"
                if "diff_mean" in r else "")
        L.append(f"| {r['arm']} | {r['quantity']} | {r['window']} | {_f(r['mean'])} | {diff} |")
    L += ["", "## 5. 読み方（spec §5.4 のまま）", "",
          "- COLLAPSED の腕では第1層も下流の停止で凍るので、第1層の水準を機構の比較に使わない。",
          f"- RESCUED は「この箱の {N_TASKS} タスクで床に落ちなかった」であり、無限時間の非崩壊ではない。",
          "- IMPAIRED の腕では「可塑性を守った」と書かない。",
          "- 機構（‖µ₂‖・c₂・z̄₂）は報告のみ。腕の順位の一致は因果の証明ではない。"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "results" / RUN_ID / "runs"))
    ap.add_argument("--out", default=str(REPO / "results" / RUN_ID))
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    shards, missing = load(src)
    res = analyze(shards)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    run_commits = sorted({sh["prov"].get("git_hash") for sh in shards.values()})
    env = {"head": head, "run_commits": run_commits, "n_shards": len(shards)}
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res["rows"]).to_csv(out / "verdict.csv", index=False)
    pd.DataFrame([{"arm": a, "endpoint": ep, "seed": s, "delta_minus_ref": v}
                  for a, eps in res["per_seed"].items() for ep, vals in eps.items()
                  for s, v in zip(res["valid_seeds"], vals)]).to_csv(out / "paired.csv", index=False)
    pd.DataFrame(res["timing"]).to_csv(out / "timing.csv", index=False)
    pd.DataFrame(res["secondary"]).to_csv(out / "secondary.csv", index=False)
    (out / "summary.md").write_text(summary_md(res, missing, env))
    (out / "provenance.json").write_text(json.dumps({
        "run_id": RUN_ID, "aggregated_at": dt.datetime.now().astimezone().isoformat(), "head": head,
        "run_commits": run_commits, "src": str(src), "n_shards": len(shards), "missing": missing,
        "valid_seeds": res["valid_seeds"], "invalid_seeds": res["invalid_seeds"],
        "applicability": res["applicability"], "floor_state": res["floor_state"], "floors": res["floors"],
        "impaired": res["impaired"], "labels": res["labels"], "timing": res["timing_labels"],
        "verdict_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "el_verdict_sha256": hashlib.sha256((REPO / "analysis" / "mucap_el_0916" / "verdict.py").read_bytes()).hexdigest(),
        "judgement": {"main_arm": MAIN_ARM, "main_window": MAIN_WIN, "cap_window": CAP_WIN,
                      "main_level": MAIN_LEVEL, "comp_level": COMP_LEVEL, "min_seeds": MIN_SEEDS,
                      "ref_floor_frac": REF_FLOOR_FRAC, "floor_k": FLOOR_K, "half": HALF,
                      "sign_alpha": SIGN_ALPHA, "early": EARLY, "chance": CHANCE, "absorb_run": ABSORB_RUN},
    }, indent=2, default=str))
    print(f"labels: {res['labels']}  valid seeds {len(res['valid_seeds'])}  missing {len(missing)}")


if __name__ == "__main__":
    main()

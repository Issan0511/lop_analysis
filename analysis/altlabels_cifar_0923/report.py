#!/usr/bin/env python3
"""report.py -- altlabels_cifar_0923 §4 and §5.

Reads the csv files ledger.py wrote and produces

    results/altlabels_cifar_0923/verdict.json
    results/altlabels_cifar_0923/summary.md

with every Q1-Q9 label spelled exactly as spec §4 names it, the seed counts behind each
one, the P1-P17 scoring table, and the seed-level tables.

Rules it implements literally:
  * every verdict is the SEED MEDIAN's interval, and carries the number of seeds in each
    interval.  When fewer than ceil(0.8 n) seeds fall in the median's own interval the
    label gets a `_MIXED` suffix (spec §3: 8/10 is a consistency flag, not a 5% test).
  * Q4 is a COUNT (A earlier than C in >= 8/10 seeds at t = 48), with the censoring rules
    of §3: an unreached hit999 is right-censored at 30,000 and only pairs whose order is
    determined are counted.  hit_full and hit99 are reported as robustness variants.
  * Q7 reads the 12-task ABC arm in its own window t4-12.
  * a verdict whose window is not yet complete is marked `provisional` and says why.  The
    script runs on partial data by design; nothing is silently dropped.

Usage: report.py [--out DIR] [--ledger DIR] [--suffix S]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

NT = str(min(8, int(os.environ.get("ALTLAB_THREADS", "8"))))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, NT)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ledger as LG                                                      # noqa: E402

MIXED_FRAC = 0.8
CENSOR = LG.CENSOR

TH = {   # spec §4.  "operational" = a working midpoint, not derived from a model.
    "Q1_rho_lo": (0.5, "midpoint of 0 and 1"),
    "Q1_rho_hi": (1.5, "operational: 50% faster"),
    "Q1_comp_abs_slope": (2.1, "midpoint of 0 and the fresh-leak 4.2 per task"),
    "Q1_beta_flag": (1.5, "descriptive flag EXP_GT_1.5"),
    "Q2_mid2_C2": (0.0774, "midpoint of OU-F -0.005 and periodic mixture +0.16"),
    "Q2_low_C2": (0.059, "sub"),
    "Q2_top_C2": (0.13, "sub"),
    "Q3_r_top": (0.0636, "midpoint of the exact return 1/45 and the IID 0.105"),
    "Q4_seeds": (8, "consistency flag out of 10 (one-sided sign test p = 0.055)"),
    "Q4_ratio": (0.5, "operational"),
    "Q4_fast": (1800, "the evaluation point after a fresh net's 1767"),
    "Q5_sigma50": (60.08, "midpoint of the fixed-return 20.3 and the IID 82.5"),
    "Q5_sigma25_record": (47.7, "record only (64.3 is an extrapolated proxy)"),
    "Q6_N_ratio": (0.80, "midpoint of the Adam-only null 1.09 and the PMNIST transplant 0.52"),
    "Q7_C3": (0.08, "mixture (-0.08,-0.08,+0.16) vs pure return (-0.5,-0.5,+1)"),
    "Q7_C2": (0.08, "same"),
    "Q8_sigma50": (29.36, "midpoint of the fixed-reuse 20.7 and the IID equilibrium 36"),
    "Q9_lo": (0.5, "operational"),
    "Q9_hi": (1.5, "operational"),
    "P16_Ntop_ratio": (1.10, "N_top(49) / N_top(25)"),
}

PREDICTIONS = [   # spec §5; the Issa column was left empty (asleep), so it is not scored
    ("P1", "Q1 = IID_SCALE_GROWTH", 0.50, 0.50),
    ("P2", "Q1 = FASTER_GROWTH", 0.30, 0.15),
    ("P3", "Q1 = REDUCED_GROWTH", 0.20, 0.35),
    ("P4", "EXP_GT_1.5 (beta_all > 1.5)", 0.15, 0.15),
    ("P5", "Q2 = PERIODIC (mid2 C2 >= 0.077)", 0.60, 0.85),
    ("P6", "Q3 = LOW_NET_DISPLACEMENT (r_top < 0.0636)", 0.70, 0.70),
    ("P7", "Q3b = HIGH_DIRECTIONAL_RETURN", 0.40, 0.75),
    ("P8", "Q4 = SAVINGS (t48, A < C in >= 8/10)", 0.75, 0.75),
    ("P9", "Q4 = STRONG_SAVINGS", 0.35, 0.25),
    ("P10", "Q5 = LOWER_WIDTH_SIDE (sigma(50) < 60.1)", 0.45, 0.50),
    ("P11", "Q6 = CONTINUOUS_FIT_GE_80PCT_IID", 0.45, 0.45),
    ("P12", "Q7 = PERIOD3_PATTERN (12 tasks)", 0.50, 0.80),
    ("P13", "Q8 = LOWER_SNAKE_WIDTH", 0.35, 0.55),
    ("P14", "Q9 N ratio < 0.5", 0.30, 0.40),
    ("P15", "Q9 N ratio in [0.5, 1.5]", 0.50, 0.55),
    ("P16", "r_top < 0.0636 and N_top(49)/N_top(25) > 1.10", 0.50, 0.55),
    ("P17", "comp complete-cycle slope > 2.1 per task", 0.60, 0.60),
]


# ==========================================================================
# the median rule
# ==========================================================================

def consistency_need(n: int) -> int:
    return int(math.ceil(MIXED_FRAC * n - 1e-9))


def label_of(x: float, buckets) -> str:
    for lab, pred in buckets:
        if pred(x):
            return lab
    return "UNCLASSIFIED"


def verdict_median(stat: str, per_seed: dict, buckets, provisional=False,
                   reason: str = "", extra: dict = None) -> dict:
    v = {int(s): float(x) for s, x in per_seed.items() if np.isfinite(x)}
    miss = sorted(set(int(s) for s in per_seed) - set(v))
    out = {"statistic": stat, "n_seeds": len(v), "seeds_used": sorted(v),
           "seeds_missing_or_nonfinite": miss,
           "per_seed": {str(k): v[k] for k in sorted(v)},
           "provisional": bool(provisional), "provisional_reason": reason}
    if not v:
        out.update({"label": None, "base_label": None, "median": None, "mixed": None,
                    "counts": {}, "status": "no_data"})
        return out
    m = float(np.median(list(v.values())))
    base = label_of(m, buckets)
    counts = Counter(label_of(x, buckets) for x in v.values())
    need = consistency_need(len(v))
    mixed = counts[base] < need
    out.update({"median": m, "base_label": base,
                "label": base + ("_MIXED" if mixed else ""),
                "mixed": bool(mixed), "counts": dict(counts),
                "consistency_need": need, "in_median_interval": counts[base],
                "status": "ok"})
    if extra:
        out.update(extra)
    return out


# ==========================================================================
# pulling statistics out of the ledger csvs
# ==========================================================================

class Data:
    def __init__(self, d: Path, sfx: str = ""):
        self.dir = d
        self.sfx = sfx
        self.t = {}
        for n in ("ledger", "cycle", "lags", "growth", "stop", "forks_scored",
                  "forks_bands", "trace_phases"):
            p = d / f"{n}{sfx}.csv"
            self.t[n] = pd.read_csv(p) if p.exists() and p.stat().st_size > 2 \
                else pd.DataFrame()
        p = d / f"ledger_provenance{sfx}.json"
        self.prov = json.loads(p.read_text()) if p.exists() else {}

    def arms(self) -> set:
        g = self.t["growth"]
        return set(g["arm"]) if len(g) else set()

    def seeds(self, arm: str) -> list:
        g = self.t["growth"]
        return sorted(set(g[g["arm"] == arm]["seed"].astype(int))) if len(g) else []

    def growth(self, arm: str, band: str, col: str) -> dict:
        g = self.t["growth"]
        if not len(g):
            return {}
        s = g[(g["arm"] == arm) & (g["band"] == band)]
        return {int(r["seed"]): float(r[col]) for _, r in s.iterrows()} if col in s else {}

    def growth_flag(self, arm: str, band: str, col: str) -> dict:
        g = self.t["growth"]
        if not len(g) or col not in g.columns:
            return {}
        s = g[(g["arm"] == arm) & (g["band"] == band)]
        return {int(r["seed"]): r[col] for _, r in s.iterrows()}

    def lag(self, arm: str, band: str, window: str, col: str) -> dict:
        l = self.t["lags"]
        if not len(l) or col not in l.columns:
            return {}
        s = l[(l["arm"] == arm) & (l["band"] == band) & (l["window"] == window)]
        return {int(r["seed"]): float(r[col]) for _, r in s.iterrows()}

    def cycle_cos(self, arm: str, band: str, tag: str, t_lo: int, t_hi: int) -> dict:
        """The seed median of cos(W_t, W_{t-lag}) over the cycle slots whose task is in
        [t_lo, t_hi] (tag 'odd' = the A-parity task of each ABAB period)."""
        c = self.t["cycle"]
        if not len(c) or f"cos_{tag}" not in c.columns:
            return {}
        s = c[(c["arm"] == arm) & (c["band"] == band)
              & (c[f"ret_{tag}_t"] >= t_lo) & (c[f"ret_{tag}_t"] <= t_hi)]
        out = {}
        for sd, g in s.groupby("seed"):
            v = g[f"cos_{tag}"].astype(float).values
            v = v[np.isfinite(v)]
            out[int(sd)] = float(np.median(v)) if v.size else np.nan
        return out

    def ratio(self, arm: str, ref: str, band: str, col: str) -> dict:
        a, b = self.growth(arm, band, col), self.growth(ref, band, col)
        return {s: (a[s] / b[s] if s in b and np.isfinite(b[s]) and b[s] != 0 else np.nan)
                for s in a}

    def growth_applies(self, arm: str) -> bool:
        """False for the 12-task ABC arm: it has no j = 14..25 window by design."""
        ga = self.prov.get("arms", {}).get(arm, {}).get("growth_tasks")
        return not (ga is not None and ga[0] in (None, "None"))

    def complete(self, arm: str) -> tuple:
        """(is the registered complete-cycle window there, a sentence saying why not)."""
        g = self.t["growth"]
        if not len(g) or arm not in set(g["arm"]):
            return False, f"{arm}: no data on disk"
        if not self.growth_applies(arm):
            return True, ""
        s = g[g["arm"] == arm]
        if "growth_complete" in s.columns and not bool(s["growth_complete"].all()):
            tp = int(s["T_present"].min())
            return False, f"{arm}: only {tp} tasks present, the j = 14..25 window needs 50"
        return True, ""

    def window_complete(self, arm: str, window: str) -> tuple:
        """Whether the lag window this verdict reads actually reached its last task."""
        l = self.t["lags"]
        if not len(l) or "complete" not in l.columns:
            return False, f"{arm}: no lags.csv"
        s = l[(l["arm"] == arm) & (l["window"] == window)]
        if not len(s):
            return False, f"{arm}: no {window} window on disk"
        if not bool(s["complete"].all()):
            hi = int(s["window_hi"].iloc[0])
            used = int(s["window_hi_used"].max())
            return False, f"{arm}: the {window} window wants t{hi}, only t{used} is on disk"
        return True, ""


# ==========================================================================
# the nine verdicts
# ==========================================================================

def rho_buckets():
    lo, hi = TH["Q1_rho_lo"][0], TH["Q1_rho_hi"][0]
    return [("REDUCED_GROWTH", lambda x: x < lo),
            ("IID_SCALE_GROWTH", lambda x: lo <= x <= hi),
            ("FASTER_GROWTH", lambda x: x > hi)]


def two(lab_lo, lab_hi, thr, strict_lt=True):
    return [(lab_lo, (lambda x: x < thr) if strict_lt else (lambda x: x <= thr)),
            (lab_hi, lambda x: True)]


def build_verdicts(D: Data) -> dict:
    V = {}
    have = D.arms()

    # ---------------- Q1 (primary): mid2 growth of LR_abab against LR_iid
    ok_a, why_a = D.complete("LR_abab")
    ok_i, why_i = D.complete("LR_iid")
    prov = (not ok_a) or (not ok_i)
    why = "; ".join(x for x in (why_a, why_i) if x)
    V["Q1"] = verdict_median("rho_mid2 = slope(E_mid2, j=14..25) / same for LR_iid, per seed",
                             D.growth("LR_abab", "mid2", "rho"), rho_buckets(), prov, why,
                             extra={"question": "mid2 growth within 50 tasks (LR_abab vs LR_iid)",
                                    "rho_status": {str(k): str(v) for k, v in
                                                   D.growth_flag("LR_abab", "mid2",
                                                                 "rho_status").items()},
                                    "slope_abab": D.growth("LR_abab", "mid2", "slope_cycle"),
                                    "slope_iid": D.growth("LR_iid", "mid2", "slope_cycle"),
                                    "secondary_rho_per_task":
                                        D.growth("LR_abab", "mid2", "rho_task")})
    for band in ("low", "all", "comp", "top", "mid1"):
        V[f"Q1_sub_{band}"] = verdict_median(
            f"rho_{band} (registered rule, recorded only)",
            D.growth("LR_abab", band, "rho"), rho_buckets(), prov, why)
    V["Q1_sub_comp_absolute"] = verdict_median(
        "comp complete-cycle slope of LR_abab, per task (threshold 2.1; LR_iid measures ~3.9)",
        D.growth("LR_abab", "comp", "slope_cycle"),
        two("COMP_SLOPE_LE_2.1", "COMP_SLOPE_GT_2.1", TH["Q1_comp_abs_slope"][0],
            strict_lt=False), prov, why,
        extra={"comp_slope_LR_iid": D.growth("LR_iid", "comp", "slope_cycle")})
    V["EXP_GT_1.5"] = verdict_median(
        "beta_all = log-log slope of N_all vs t over t6-50 (descriptive flag)",
        D.growth("LR_abab", "all", "beta_loglog"),
        [("EXP_GT_1.5", lambda x: x > TH["Q1_beta_flag"][0]),
         ("EXP_LE_1.5", lambda x: True)], prov, why)
    # the spec asks that the centre's move and the diameter's growth be reported apart
    V["Q1_center_vs_diameter"] = {
        "statistic": "complete-cycle slope of ||M_j||^2 and of D_j^2 (mid2), LR_abab vs LR_iid",
        "note": "reported apart so a growing E is not read as a moving centre",
        "per_arm": {a: {"M_sq_slope": cycle_component_slope(D, a, "mid2", "M_sq"),
                        "D_sq_slope": cycle_component_slope(D, a, "mid2", "D_sq")}
                    for a in ("LR_abab", "LR_iid") if a in have},
    }

    # ---------------- Q2 periodicity
    ok, why = D.window_complete("LR_abab", "late")
    V["Q2"] = verdict_median(
        "mid2 C2 (arithmetic mean of Frobenius cosines), LR_abab, late window t26-50",
        D.lag("LR_abab", "mid2", "late", "C2_cos"),
        two("NOT_PERIODIC", "PERIODIC", TH["Q2_mid2_C2"][0]), not ok, why,
        extra={"C2_minus_C13": D.lag("LR_abab", "mid2", "late", "C2_minus_C13_cos"),
               "C2_energy": D.lag("LR_abab", "mid2", "late", "C2_energy"),
               "even_odd_pattern": even_odd(D, "LR_abab", "mid2", "late"),
               "iid_reference_C2": D.lag("LR_iid", "mid2", "late", "C2_cos")})
    for band, key in (("low", "Q2_low_C2"), ("top", "Q2_top_C2")):
        V[f"Q2_sub_{band}"] = verdict_median(
            f"{band} C2 (late), threshold {TH[key][0]}",
            D.lag("LR_abab", band, "late", "C2_cos"),
            two(f"{band.upper()}_NOT_PERIODIC", f"{band.upper()}_PERIODIC", TH[key][0]),
            not ok, why)

    # ---------------- Q3 net displacement of the top band
    ok_r, why_r = D.window_complete("LR_abab", "all")
    V["Q3"] = verdict_median(
        "r_top = ||sum_{t=6..50} dW_t||^2_top / sum ||dW_t||^2_top, LR_abab",
        D.lag("LR_abab", "top", "all", "r_b"),
        two("LOW_NET_DISPLACEMENT", "HIGH_NET_DISPLACEMENT", TH["Q3_r_top"][0]), not ok_r, why_r,
        extra={"exact_return_floor": 1 / 45, "iid_reference_r_top":
               D.lag("LR_iid", "top", "all", "r_b")})

    # Q3b: the A-parity lag-2 cosine against the per-seed threshold (c_IID + 1)/2
    ab = D.cycle_cos("LR_abab", "top", "odd", 26, 50)
    iid = D.cycle_cos("LR_iid", "top", "odd", 26, 50)
    margin = {s: ab[s] - (iid[s] + 1.0) / 2.0 for s in ab if s in iid}
    V["Q3b"] = verdict_median(
        "median over late odd t of cos(W_t, W_{t-2})_top, minus the seed's own "
        "(c_IID + 1)/2 threshold",
        margin, two("LOW_DIRECTIONAL_RETURN", "HIGH_DIRECTIONAL_RETURN", 0.0), not ok, why,
        extra={"cos_abab": ab, "cos_iid": iid,
               "threshold_per_seed": {str(s): (iid[s] + 1.0) / 2.0 for s in iid},
               "note": "a_t and the residual are in cycle.csv; this is not a claim that the "
                       "solution is the same"})

    # ---------------- Q4 the revisit saving (t = 48, A vs C)
    V["Q4"] = fork_verdict(D, "LR_abab_fork", 48)
    for t in (2, 10, 20, 30, 40):
        v = fork_verdict(D, "LR_abab_fork", t)
        if v["status"] != "no_data":
            V[f"Q4_sub_t{t}"] = v
    v = fork_verdict(D, "LR_iid_fork", 48, lhs="next", tag="nextC")
    if v["status"] != "no_data":
        V["Q4_sub_iid_next_vs_C"] = v

    # ---------------- Q5 width
    ok, why = D.complete("LR_abab")
    V["Q5"] = verdict_median(
        "sigma_med(50), LR_abab (numpy median over the 100 units of sd(z1_i))",
        D.growth("LR_abab", "all", "sigma_med_50"),
        two("LOWER_WIDTH_SIDE", "IID_WIDTH_SIDE", TH["Q5_sigma50"][0]), not ok, why,
        extra={"sigma_med_25": D.growth("LR_abab", "all", "sigma_med_25"),
               "sigma_med_25_record_threshold": TH["Q5_sigma25_record"][0],
               "sigma_med_50_LR_iid": D.growth("LR_iid", "all", "sigma_med_50"),
               "V_top_50_ratio": D.ratio("LR_abab", "LR_iid", "top", "V_top_50"),
               "V_mid1_50": D.growth("LR_abab", "mid1", "V_50"),
               "V_mid1_50_LR_iid": D.growth("LR_iid", "mid1", "V_50")})

    # ---------------- Q6 AAAA
    ok_a, why_a = D.complete("LR_aaaa")
    ok_i, why_i = D.complete("LR_iid")
    V["Q6"] = verdict_median(
        "N_all(50) of LR_aaaa over LR_iid, same seed",
        D.ratio("LR_aaaa", "LR_iid", "all", "N_50"),
        [("CONTINUOUS_FIT_GE_80PCT_IID", lambda x: x >= TH["Q6_N_ratio"][0]),
         ("CONTINUOUS_FIT_LT_80PCT_IID", lambda x: True)],
        (not ok_a) or (not ok_i), "; ".join(x for x in (why_a, why_i) if x),
        extra={"N_all_50_aaaa": D.growth("LR_aaaa", "all", "N_50"),
               "N_all_50_iid": D.growth("LR_iid", "all", "N_50"),
               "axis_note": "1.5 M updates of one label; the time axis is cumulative updates"})

    # ---------------- Q7 ABC
    ok, why = D.window_complete("LR_abc", "late")
    if "LR_abc" in have:
        c1 = D.lag("LR_abc", "mid2", "late", "C1_cos")
        c2 = D.lag("LR_abc", "mid2", "late", "C2_cos")
        c3 = D.lag("LR_abc", "mid2", "late", "C3_cos")
        lab = {}
        for s in c3:
            if c3[s] >= TH["Q7_C3"][0] and c2.get(s, np.nan) < TH["Q7_C2"][0]:
                lab[s] = "PERIOD3_PATTERN"
            elif c3[s] >= TH["Q7_C3"][0]:
                lab[s] = "GENERIC_PERSISTENCE"
            else:
                lab[s] = "NONE"
        counts = Counter(lab.values())
        # the registered classification is on the seed medians of C2 and C3
        m2, m3 = med(c2), med(c3)
        base = ("PERIOD3_PATTERN" if (m3 >= TH["Q7_C3"][0] and m2 < TH["Q7_C2"][0]) else
                "GENERIC_PERSISTENCE" if m3 >= TH["Q7_C3"][0] else "NONE")
        need = consistency_need(len(lab)) if lab else 0
        V["Q7"] = {"statistic": "mid2 (C1, C2, C3) cosines over the ABC window t4-12",
                   "window": "t4-12", "median_C1": med(c1), "median_C2": m2, "median_C3": m3,
                   "median_C3_minus_C2": med({s: c3[s] - c2[s] for s in c3 if s in c2}),
                   "base_label": base,
                   "label": base + ("_MIXED" if counts.get(base, 0) < need else ""),
                   "mixed": bool(counts.get(base, 0) < need),
                   "counts": dict(counts), "consistency_need": need,
                   "n_seeds": len(lab), "per_seed": {str(s): lab[s] for s in sorted(lab)},
                   "per_seed_C": {str(s): [c1.get(s), c2.get(s), c3.get(s)]
                                  for s in sorted(c3)},
                   "reference_mixture": [-0.08, -0.08, 0.16],
                   "reference_pure_return": [-0.5, -0.5, 1.0],
                   "provisional": not ok, "provisional_reason": why,
                   "status": "ok" if lab else "no_data"}
    else:
        V["Q7"] = {"statistic": "mid2 (C1, C2, C3), ABC window t4-12", "status": "no_data",
                   "label": None, "provisional": True,
                   "provisional_reason": "LR_abc has no snapshots yet"}

    # ---------------- Q8 Snake
    ok, why = D.complete("SNA_abab")
    V["Q8"] = verdict_median(
        "sigma_med(50), SNA_abab",
        D.growth("SNA_abab", "all", "sigma_med_50"),
        two("LOWER_SNAKE_WIDTH", "IID_SNAKE_WIDTH", TH["Q8_sigma50"][0]), not ok, why,
        extra={"sigma_med_50_SNA_iid_ext": D.growth("SNA_iid_ext", "all", "sigma_med_50"),
               "note": "the SNA IID comparison is the archived R=20 run: a different layout, "
                       "recorded only"})
    V["Q8_sub_rho_mid2_vs_ext"] = verdict_median(
        "SNA_abab mid2 complete-cycle slope over the archived SNA R=20 IID slope (layout "
        "differs; recorded only)",
        D.ratio("SNA_abab", "SNA_iid_ext", "mid2", "slope_cycle"), rho_buckets(), not ok, why)

    # ---------------- Q9 the stop chains
    ok_a, why_a = D.complete("LR_abab_stop")
    ok_i, why_i = D.complete("LR_iid_stop")
    rat = D.ratio("LR_abab_stop", "LR_iid_stop", "all", "N_50")
    V["Q9"] = verdict_median(
        "N_all(50) of LR_abab_stop over LR_iid_stop, same seed (seeds 0-4, recorded only)",
        rat, [("STOP_N_RATIO_LT_0.5", lambda x: x < TH["Q9_lo"][0]),
              ("STOP_N_RATIO_0.5_1.5", lambda x: TH["Q9_lo"][0] <= x <= TH["Q9_hi"][0]),
              ("STOP_N_RATIO_GT_1.5", lambda x: x > TH["Q9_hi"][0])],
        (not ok_a) or (not ok_i), "; ".join(x for x in (why_a, why_i) if x),
        extra={"slope_cycle_ratio_mid2": D.ratio("LR_abab_stop", "LR_iid_stop", "mid2",
                                                 "slope_cycle"),
               "slope_cycle_ratio_all": D.ratio("LR_abab_stop", "LR_iid_stop", "all",
                                                "slope_cycle"),
               "cumulative_updates": stop_updates(D),
               "note": "this compares two learning policies that use the same reaching "
                       "criterion per task; the difference from the full runs is NOT called "
                       "a post-convergence effect (spec Q9)"})
    return V


def med(d: dict) -> float:
    v = np.asarray([x for x in d.values() if np.isfinite(x)], float)
    return float(np.median(v)) if v.size else np.nan


def even_odd(D: Data, arm: str, band: str, window: str) -> dict:
    l = D.t["lags"]
    if not len(l):
        return {}
    s = l[(l["arm"] == arm) & (l["band"] == band) & (l["window"] == window)]
    return {f"C{k}": med({int(r["seed"]): float(r[f"C{k}_cos"]) for _, r in s.iterrows()})
            for k in range(1, 7)}


def cycle_component_slope(D: Data, arm: str, band: str, col: str) -> dict:
    """OLS slope of the cycle component against x_j, over the registered j = 14..25."""
    c = D.t["cycle"]
    if not len(c) or col not in c.columns:
        return {}
    s = c[(c["arm"] == arm) & (c["band"] == band) & (c["j"] >= 14) & (c["j"] <= 25)]
    out = {}
    for sd, g in s.groupby("seed"):
        sl, _, n, _ = LG.ols(g["x_j"].values, g[col].values)
        out[str(int(sd))] = sl if n >= 3 else np.nan
    return out


def stop_updates(D: Data) -> dict:
    st = D.t["stop"]
    if not len(st):
        return {}
    out = {}
    for arm, g in st[st["band"] == "all"].groupby("arm"):
        gg = g[g["t"] == g["t"].max()]
        out[str(arm)] = {"t_max": int(g["t"].max()),
                         "tc_at_t_max": {str(int(r["seed"])): (float(r["tc"])
                                                               if np.isfinite(r["tc"]) else None)
                                         for _, r in gg.iterrows()}}
    return out


def fork_verdict(D: Data, fset: str, t: int, lhs: str = "A", tag: str = "AC") -> dict:
    f = D.t["forks_scored"]
    base = {"statistic": f"{fset} t={t}: {lhs} vs C, hit999 with the spec's censoring",
            "fork_set": fset, "t": t, "pair": f"{lhs} vs C"}
    if not len(f) or f"sign_{tag}_hit999" not in f.columns:
        return {**base, "status": "no_data", "label": None, "provisional": True,
                "provisional_reason": f"{fset} has not run yet (forks_scored.csv is empty)"}
    s = f[(f["fork_set"] == fset) & (f["t"] == t)]
    if not len(s):
        return {**base, "status": "no_data", "label": None, "provisional": True,
                "provisional_reason": f"{fset} has no rows at t = {t}"}
    out = {**base}
    for key in ("hit999", "hit_full", "hit99"):
        sg = s[f"sign_{tag}_{key}"].astype(float).values
        seeds = s["seed"].astype(int).values
        n = len(sg)
        n_lhs_first = int((sg == -1).sum())
        n_tie = int((sg == 0).sum())
        n_rhs_first = int((sg == +1).sum())
        n_undet = int(np.isnan(sg).sum())
        lo = s[f"ratio_{tag}_{key}_lo"].astype(float).values
        hi = s[f"ratio_{tag}_{key}_hi"].astype(float).values
        pt = s[f"ratio_{tag}_{key}"].astype(float).values
        med_pt = float(np.nanmedian(pt)) if np.isfinite(pt).any() else np.nan
        med_hi = float(np.nanmedian(hi)) if np.isfinite(hi).any() else np.nan
        lhs_h = s[f"hit999_{lhs}"].astype(float).values if key == "hit999" \
            else s[f"{key}_{lhs}"].astype(float).values
        lhs_obs = lhs_h[lhs_h >= 0]
        med_lhs = float(np.median(lhs_obs)) if lhs_obs.size else np.nan
        need = TH["Q4_seeds"][0] if n >= 10 else consistency_need(n)
        savings = n_lhs_first >= need
        strong = bool(savings and np.isfinite(med_hi) and med_hi < TH["Q4_ratio"][0])
        fast = bool(savings and np.isfinite(med_lhs) and med_lhs <= TH["Q4_fast"][0])
        lab = "SAVINGS" if savings else "SAVINGS_NOT_ESTABLISHED"
        out[key] = {
            "label": lab, "STRONG_SAVINGS": strong, "FAST_REVISIT": fast,
            "n_seeds": n, "need": need,
            "counts": {f"{lhs}_first": n_lhs_first, "tie": n_tie, "C_first": n_rhs_first,
                       "undetermined": n_undet},
            "median_ratio_point": med_pt,
            "median_ratio_upper_bound": med_hi,
            "median_ratio_lower_bound": float(np.nanmedian(lo)) if np.isfinite(lo).any()
            else np.nan,
            f"median_{key}_{lhs}_observed": med_lhs,
            "censoring": {str(int(sd)): str(c) for sd, c in
                          zip(seeds, s[f"censoring_{tag}_{key}"].values)},
            "per_seed": {str(int(sd)): {"lhs": float(a), "C": float(b), "sign": float(g_)}
                         for sd, a, b, g_ in zip(
                             seeds, lhs_h,
                             s["hit999_C" if key == "hit999"
                               else f"{key}_C"].astype(float).values,
                             sg)},
        }
    prim = out["hit999"]
    out.update({"label": prim["label"] + ("" if prim["n_seeds"] >= 10 else "_PARTIAL_SEEDS"),
                "STRONG_SAVINGS": prim["STRONG_SAVINGS"], "FAST_REVISIT": prim["FAST_REVISIT"],
                "status": "ok",
                "provisional": bool(prim["n_seeds"] < 10),
                "provisional_reason": (f"only {prim['n_seeds']}/10 seeds present"
                                       if prim["n_seeds"] < 10 else ""),
                "censor_at": CENSOR,
                "censoring_note": "hit999 is right-censored at 30,000; hit_full and hit99 at the "
                                  "bundle's own last step.  A pair counts only when its order is "
                                  "determined."})
    return out


# ==========================================================================
# P1-P17
# ==========================================================================

def base(lab):
    return None if lab is None else str(lab).replace("_MIXED", "").replace("_PARTIAL_SEEDS", "")


def score(V: dict, D: Data) -> list:
    def q(k):
        return V.get(k, {})

    def lab(k):
        return base(q(k).get("label"))

    def ok(k):
        return q(k).get("status") == "ok" and q(k).get("label") is not None

    rows = []

    def put(pid, pred, cl, cx, outcome, measured, note=""):
        rows.append({"id": pid, "predicate": pred, "claude": cl, "codex": cx,
                     "outcome": outcome, "measured": measured, "note": note})

    m = {p[0]: p for p in PREDICTIONS}
    U = "△"   # undecidable
    Y = "○"
    N = "×"

    for pid, target in (("P1", "IID_SCALE_GROWTH"), ("P2", "FASTER_GROWTH"),
                        ("P3", "REDUCED_GROWTH")):
        _, pred, cl, cx = m[pid]
        o = U if not ok("Q1") else (Y if lab("Q1") == target else N)
        put(pid, pred, cl, cx, o,
            f"rho_mid2 median = {fmt(q('Q1').get('median'))} -> {q('Q1').get('label')}",
            "provisional" if q("Q1").get("provisional") else "")

    _, pred, cl, cx = m["P4"]
    o = U if not ok("EXP_GT_1.5") else (Y if lab("EXP_GT_1.5") == "EXP_GT_1.5" else N)
    put("P4", pred, cl, cx, o,
        f"beta_all median = {fmt(q('EXP_GT_1.5').get('median'))}",
        "provisional" if q("EXP_GT_1.5").get("provisional") else "")

    _, pred, cl, cx = m["P5"]
    o = U if not ok("Q2") else (Y if lab("Q2") == "PERIODIC" else N)
    put("P5", pred, cl, cx, o, f"mid2 C2 (late) median = {fmt(q('Q2').get('median'))}",
        "provisional" if q("Q2").get("provisional") else "")

    _, pred, cl, cx = m["P6"]
    o = U if not ok("Q3") else (Y if lab("Q3") == "LOW_NET_DISPLACEMENT" else N)
    put("P6", pred, cl, cx, o, f"r_top median = {fmt(q('Q3').get('median'))}",
        "provisional" if q("Q3").get("provisional") else "")

    _, pred, cl, cx = m["P7"]
    o = U if not ok("Q3b") else (Y if lab("Q3b") == "HIGH_DIRECTIONAL_RETURN" else N)
    put("P7", pred, cl, cx, o,
        f"median margin over (c_IID+1)/2 = {fmt(q('Q3b').get('median'))}",
        "provisional" if q("Q3b").get("provisional") else "")

    q4 = q("Q4")
    _, pred, cl, cx = m["P8"]
    o = U if q4.get("status") != "ok" else (Y if base(q4.get("label")) == "SAVINGS" else N)
    put("P8", pred, cl, cx, o,
        f"A earlier than C in {q4.get('hit999', {}).get('counts', {}).get('A_first', '-')}"
        f"/{q4.get('hit999', {}).get('n_seeds', '-')} seeds",
        "provisional" if q4.get("provisional") else "")
    _, pred, cl, cx = m["P9"]
    o = U if q4.get("status") != "ok" else (Y if q4.get("STRONG_SAVINGS") else N)
    put("P9", pred, cl, cx, o,
        f"median A/C (upper bound) = "
        f"{fmt(q4.get('hit999', {}).get('median_ratio_upper_bound'))}",
        "provisional" if q4.get("provisional") else "")

    _, pred, cl, cx = m["P10"]
    o = U if not ok("Q5") else (Y if lab("Q5") == "LOWER_WIDTH_SIDE" else N)
    put("P10", pred, cl, cx, o, f"sigma_med(50) median = {fmt(q('Q5').get('median'))}",
        "provisional" if q("Q5").get("provisional") else "")

    _, pred, cl, cx = m["P11"]
    o = U if not ok("Q6") else (Y if lab("Q6") == "CONTINUOUS_FIT_GE_80PCT_IID" else N)
    put("P11", pred, cl, cx, o, f"N_all(50) ratio median = {fmt(q('Q6').get('median'))}",
        "provisional" if q("Q6").get("provisional") else "")

    _, pred, cl, cx = m["P12"]
    q7 = q("Q7")
    o = U if q7.get("status") != "ok" else (Y if base(q7.get("label")) == "PERIOD3_PATTERN" else N)
    put("P12", pred, cl, cx, o,
        f"(C1, C2, C3) = ({fmt(q7.get('median_C1'))}, {fmt(q7.get('median_C2'))}, "
        f"{fmt(q7.get('median_C3'))})", "provisional" if q7.get("provisional") else "")

    _, pred, cl, cx = m["P13"]
    o = U if not ok("Q8") else (Y if lab("Q8") == "LOWER_SNAKE_WIDTH" else N)
    put("P13", pred, cl, cx, o, f"SNA sigma_med(50) median = {fmt(q('Q8').get('median'))}",
        "provisional" if q("Q8").get("provisional") else "")

    q9 = q("Q9")
    _, pred, cl, cx = m["P14"]
    o = U if not ok("Q9") else (Y if lab("Q9") == "STOP_N_RATIO_LT_0.5" else N)
    put("P14", pred, cl, cx, o, f"stop N ratio median = {fmt(q9.get('median'))}",
        "provisional" if q9.get("provisional") else "")
    _, pred, cl, cx = m["P15"]
    o = U if not ok("Q9") else (Y if lab("Q9") == "STOP_N_RATIO_0.5_1.5" else N)
    put("P15", pred, cl, cx, o, f"stop N ratio median = {fmt(q9.get('median'))}",
        "provisional" if q9.get("provisional") else "")

    # P16: a conjunction, on the seed medians
    _, pred, cl, cx = m["P16"]
    n49 = D.growth("LR_abab", "top", "N_49")
    n25 = D.growth("LR_abab", "top", "N_25")
    rat = {s: n49[s] / n25[s] for s in n49 if s in n25 and np.isfinite(n25[s]) and n25[s] > 0}
    mr = med(rat)
    rt = q("Q3").get("median")
    if not ok("Q3") or not np.isfinite(mr):
        o = U
    else:
        o = Y if (rt < TH["Q3_r_top"][0] and mr > TH["P16_Ntop_ratio"][0]) else N
    put("P16", pred, cl, cx, o,
        f"r_top = {fmt(rt)}, N_top(49)/N_top(25) median = {fmt(mr)}",
        "provisional" if q("Q3").get("provisional") else "")

    _, pred, cl, cx = m["P17"]
    vv = q("Q1_sub_comp_absolute")
    o = U if vv.get("status") != "ok" else (
        Y if float(vv.get("median", np.nan)) > TH["Q1_comp_abs_slope"][0] else N)
    put("P17", pred, cl, cx, o, f"comp cycle slope median = {fmt(vv.get('median'))}",
        "provisional" if vv.get("provisional") else "")
    return rows


def fmt(x, p=4):
    if x is None:
        return "n/a"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.{p}g}"


def brier(rows, col):
    v = [(r[col], 1.0 if r["outcome"] == "○" else 0.0)
         for r in rows if r["outcome"] in ("○", "×")]
    return float(np.mean([(p - o) ** 2 for p, o in v])) if v else None


# ==========================================================================
# summary.md
# ==========================================================================

def write_summary(V: dict, P: list, D: Data, meta: dict, out: Path, sfx: str) -> None:
    L = []
    w = L.append
    w("# altlabels_cifar_0923 — 2 ラベル交互（ABAB）の第1層 W と幅")
    w("")
    w(f"`report.py` が書いた。生成 {meta['written']}。spec sha256 `{meta['spec_sha256']}`"
      f"（{meta['spec_commit']}）。閾値は spec §4 のもの（下の表）。")
    w("")
    w(f"**数値は seed 中央値 [最小, 最大]（seed の範囲であって信頼区間ではない）。**"
      f" 判定は中央値の区間で、中央値の区間に入る seed が {int(MIXED_FRAC*100)}% 未満なら"
      f" `_MIXED` を付ける。8/10 は整合性の旗で 5% の検定ではない（spec §3）。")
    w("")

    # ---- data state
    w("## 0. いまあるデータ")
    w("")
    w("| 腕 | seed | 課題 t の最大 | 登録窓（j=14..25 / t27–50）| 状態 |")
    w("|---|---|---|---|---|")
    g = D.t["growth"]
    for arm in sorted(D.arms()):
        s = g[g["arm"] == arm]
        sd = sorted(set(s["seed"].astype(int)))
        tp = sorted(set(s["T_present"].astype(int)))
        comp, why = D.complete(arm)
        cell = ("—（12 課題の腕なので Q1 の窓は無い）" if not D.growth_applies(arm)
                else "完" if comp else "未完")
        w(f"| {arm} | {len(sd)} ({min(sd)}–{max(sd)}) | {min(tp)}–{max(tp)} | {cell} | "
          f"{'確定' if comp else '暫定'} |")
    miss = meta.get("arms_absent", {})
    if miss:
        w("")
        w("未着手/未検出: " + ", ".join(f"`{k}`" for k in miss))
    nop = D.prov.get("arms_without_provenance") or []
    if nop:
        w("")
        w("**provenance.json がまだ無い腕**（エンジンは走り終わりに書くので、走行中）: "
          + ", ".join(f"`{k}`" for k in nop)
          + "。これらの数値はすべて暫定で、走が終わってから出し直す必要がある。"
            "tc は trace の最終行（課題終端の時計）から復元しており、"
            "step 0 の照合はフル走の形が確認できる場合だけ行っている。")
    w("")
    prov = [k for k, v in V.items() if isinstance(v, dict) and v.get("provisional")]
    if prov:
        w(f"**暫定の判定**（窓が未完・データ不足）: {', '.join('`'+k+'`' for k in prov)}。"
          f"腕が走り終えたら同じコマンドで出し直す。")
        w("")

    # ---- verdicts
    w("## 1. 登録判定（§4）")
    w("")
    w("| Q | 統計量 | 中央値 | 判定 | 区間ごとの seed 数 | 状態 |")
    w("|---|---|---|---|---|---|")
    for k in ("Q1", "Q2", "Q3", "Q3b", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9"):
        v = V.get(k, {})
        if k == "Q4":
            h = v.get("hit999", {})
            c = h.get("counts", {})
            w(f"| Q4 | {v.get('statistic','')} | A/C 中央値 "
              f"{fmt(h.get('median_ratio_point'))} | **{v.get('label')}**"
              f"{' + STRONG_SAVINGS' if v.get('STRONG_SAVINGS') else ''}"
              f"{' + FAST_REVISIT' if v.get('FAST_REVISIT') else ''} | "
              f"A先 {c.get('A_first','-')} / C先 {c.get('C_first','-')} / "
              f"同 {c.get('tie','-')} / 未確定 {c.get('undetermined','-')} | "
              f"{'暫定' if v.get('provisional') else '確定'} |")
            continue
        if k == "Q7":
            w(f"| Q7 | mid2 (C1,C2,C3) t4–12 | ({fmt(v.get('median_C1'))}, "
              f"{fmt(v.get('median_C2'))}, {fmt(v.get('median_C3'))}) | "
              f"**{v.get('label')}** | {v.get('counts', {})} | "
              f"{'暫定' if v.get('provisional') else '確定'} |")
            continue
        w(f"| {k} | {v.get('statistic','')} | {fmt(v.get('median'))} | "
          f"**{v.get('label')}** | {v.get('counts', {})} | "
          f"{'暫定' if v.get('provisional') else '確定'} |")
    w("")
    w("副次（登録外の記録）")
    w("")
    w("| 名前 | 統計量 | 中央値 | 判定 |")
    w("|---|---|---|---|")
    for k, v in V.items():
        if k in ("Q1", "Q2", "Q3", "Q3b", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9"):
            continue
        if not isinstance(v, dict):
            continue
        if k == "Q1_center_vs_diameter":
            continue
        if k.startswith("Q4_sub"):
            h = v.get("hit999", {})
            w(f"| {k} | {v.get('statistic','')} | {v.get('pair','A/C')} = "
              f"{fmt(h.get('median_ratio_point'))} | {v.get('label')} |")
            continue
        w(f"| {k} | {v.get('statistic','')} | {fmt(v.get('median'))} | {v.get('label')} |")
    w("")
    cv = V.get("Q1_center_vs_diameter", {}).get("per_arm", {})
    if cv:
        w("周期の中心の移動と直径の増大（mid2、j=14..25 の傾き、seed 中央値）")
        w("")
        w("| 腕 | ‖M_j‖² の傾き | D_j² の傾き |")
        w("|---|---|---|")
        for arm, d in cv.items():
            w(f"| {arm} | {fmt(med({int(k2): v2 for k2, v2 in d['M_sq_slope'].items()}))} | "
              f"{fmt(med({int(k2): v2 for k2, v2 in d['D_sq_slope'].items()}))} |")
        w("")

    # ---- Q4 robustness
    q4 = V.get("Q4", {})
    if q4.get("status") == "ok":
        w("### Q4 の頑健性（同じ判定を hit_full と hit99 でも）")
        w("")
        w("| 基準 | A 先 / C 先 / 同 / 未確定 | 判定 | A/C 中央値（点 / 上界）| A の中央値 |")
        w("|---|---|---|---|---|")
        for key, nm in (("hit999", "hit999（登録）"), ("hit_full", "hit_full (1200/1200)"),
                        ("hit99", "hit99")):
            h = q4.get(key, {})
            c = h.get("counts", {})
            w(f"| {nm} | {c.get('A_first','-')} / {c.get('C_first','-')} / "
              f"{c.get('tie','-')} / {c.get('undetermined','-')} | {h.get('label')}"
              f"{' + STRONG' if h.get('STRONG_SAVINGS') else ''}"
              f"{' + FAST' if h.get('FAST_REVISIT') else ''} | "
              f"{fmt(h.get('median_ratio_point'))} / {fmt(h.get('median_ratio_upper_bound'))} | "
              f"{fmt(h.get(f'median_{key}_A_observed'))} |")
        w("")
        w(f"打ち切り: hit999 は 30,000 で右打ち切り。hit_full / hit99 は束自身の最後の"
          f"評価点で打ち切り（束は全スロットが hit999+500 を過ぎた時点で終わるので "
          f"30,000 に届かないことがある）。大小が確定する対だけ数える。")
        w("")

    # ---- predictions
    w("## 2. 予測の採点（§5）")
    w("")
    w("○ = 起きた、× = 起きなかった、△ = まだ判定できない。Issa の列は空のまま（就寝中）。")
    w("")
    w("| # | 述語 | Claude | Codex | 結果 | 実測 |")
    w("|---|---|---|---|---|---|")
    for r in P:
        w(f"| {r['id']} | {r['predicate']} | {r['claude']:.2f} | {r['codex']:.2f} | "
          f"{r['outcome']}{'*' if r['note'] else ''} | {r['measured']} |")
    w("")
    nd = sum(1 for r in P if r["outcome"] == "△")
    star = sum(1 for r in P if r["note"])
    w(f"未判定 {nd}/17。* = 窓が未完のままの暫定（{star} 件）。")
    bc, bx = meta["brier_claude"], meta["brier_codex"]
    if bc is not None:
        nhit_c = sum(1 for r in P if r["outcome"] == "○")
        w("")
        w(f"決着した {17-nd} 件の Brier 得点: Claude {bc:.4f}、Codex {bx:.4f}"
          f"（小さいほど良い。○ は {nhit_c} 件）。")
    w("")

    # ---- seed tables
    w("## 3. seed ごとの表")
    w("")
    for k in ("Q1", "Q2", "Q3", "Q3b", "Q5", "Q6", "Q8", "Q9"):
        v = V.get(k, {})
        ps = v.get("per_seed", {})
        if not ps:
            continue
        w(f"**{k}** — {v.get('statistic','')}")
        w("")
        ks = sorted(ps, key=int)
        w("| seed | " + " | ".join(ks) + " | 中央値 |")
        w("|---|" + "---|" * (len(ks) + 1))
        w("| 値 | " + " | ".join(fmt(ps[s]) for s in ks) + f" | {fmt(v.get('median'))} |")
        w("")
    q4 = V.get("Q4", {})
    if q4.get("status") == "ok":
        h = q4.get("hit999", {})
        ps = h.get("per_seed", {})
        w("**Q4** — fork t=48、hit999（A / C / 符号）")
        w("")
        ks = sorted(ps, key=int)
        w("| seed | " + " | ".join(ks) + " |")
        w("|---|" + "---|" * len(ks))
        w("| A | " + " | ".join(fmt(ps[s]["lhs"], 5) for s in ks) + " |")
        w("| C | " + " | ".join(fmt(ps[s]["C"], 5) for s in ks) + " |")
        w("| A<C | " + " | ".join(
            {-1.0: "○", 0.0: "=", 1.0: "×"}.get(ps[s]["sign"], "?") for s in ks) + " |")
        w("")

    # ---- thresholds and conventions
    w("## 4. 使った閾値と規約")
    w("")
    w("| 名前 | 値 | 由来 |")
    w("|---|---|---|")
    for k, (v, why) in TH.items():
        w(f"| {k} | {v} | {why} |")
    w("")
    conv = D.prov.get("conventions", {})
    if conv:
        w("解析側の規約（`ledger_provenance.json` にも入っている）")
        w("")
        for k, v in conv.items():
            w(f"- `{k}`: {v}")
        w("")
    ch = D.prov.get("checks", {})
    if ch:
        w("帳簿の恒等式（S7）: "
          f"ΔN = d + e の最大相対残差 {fmt(ch.get('max_identity_rel'), 3)}"
          f"（許容 {D.prov.get('identity_rtol')}）、帯の和 {fmt(ch.get('max_bandsum_rel'), 3)}、"
          f"V の 4 帯の和 {fmt(ch.get('max_V_sum_rel'), 3)}、"
          f"V_mu = {ch.get('V_mu_max')}、V_comp = {ch.get('V_comp_max')}、"
          f"σ と走の trace の差 {fmt(ch.get('sigma_med_vs_trace_max_abs'), 3)}。")
        w("")
    sc = meta.get("synthetic_checks")
    if sc:
        w(f"合成検査（`synthetic_checks.json`）: all_pass = {sc}。")
        w("")
    w("## 5. 読み方の注意")
    w("")
    w("- どの判定も **50 課題内の分類**で、有界性や漸近次数の判定ではない（spec §0）。")
    w("- 「同じ」の主張には同等性の議論が要る。差が出ないことでは足りない（spec §3）。")
    w("- 外部参照（`LR_iid_ext` / `SNA_iid_ext`）は親の R=20 の走で、配置が違うので "
      "bit 一致は仮定していない。登録の対照は同配置の `LR_iid`。")
    w("- fork 点 t = 48 だけが登録で、他の fork 点は独立標本として合算しない（spec §4 Q4）。")
    (out / f"summary{sfx}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out/('summary'+sfx+'.md')}")


# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(LG.OUT_DEFAULT))
    ap.add_argument("--ledger", default=None, help="where the csvs are (default = --out)")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = Data(Path(a.ledger) if a.ledger else out, a.suffix)
    if not len(D.t["growth"]):
        raise SystemExit(f"no growth{a.suffix}.csv in {D.dir}: run ledger.py first")

    V = build_verdicts(D)
    P = score(V, D)

    spec_sha, spec_commit = "unknown", "unknown"
    p = out / "PREDICTIONS.sha256"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith("sha256:"):
                spec_sha = line.split(":", 1)[1].strip()
            if line.startswith("commit:"):
                spec_commit = line.split(":", 1)[1].strip()
    sc = None
    p = out / "synthetic_checks.json"
    if p.exists():
        sc = json.loads(p.read_text()).get("_meta", {}).get("all_pass")

    meta = {"run_id": LG.EXPERIMENT, "stage": "report", "written": time.strftime("%FT%T%z"),
            **LG.git_state(), "spec_sha256": spec_sha, "spec_commit": spec_commit,
            "ledger_dir": str(D.dir), "ledger_provenance": D.prov.get("checks"),
            "arms_present": sorted(D.arms()), "arms_absent": D.prov.get("arms_absent", {}),
            "mixed_fraction": MIXED_FRAC, "censor_at": CENSOR,
            "synthetic_checks": sc,
            "brier_claude": brier(P, "claude"), "brier_codex": brier(P, "codex"),
            "n_undecided": sum(1 for r in P if r["outcome"] == "△"),
            "provisional_verdicts": sorted(k for k, v in V.items()
                                           if isinstance(v, dict) and v.get("provisional")),
            "scoring_note": "a _MIXED / _PARTIAL_SEEDS suffix does not change which interval "
                            "the median fell in, so the predictions are scored on the base "
                            "label and the suffix is reported next to it",
            }
    doc = {"_meta": meta, "thresholds": {k: {"value": v, "origin": w} for k, (v, w) in TH.items()},
           "verdicts": V, "predictions": P}
    (out / f"verdict{a.suffix}.json").write_text(json.dumps(doc, indent=1, default=str))
    print(f"wrote {out/('verdict'+a.suffix+'.json')}")
    write_summary(V, P, D, meta, out, a.suffix)

    for k in ("Q1", "Q2", "Q3", "Q3b", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9"):
        v = V.get(k, {})
        print(f"  {k:4s} {str(v.get('label')):34s} "
              f"{'(provisional)' if v.get('provisional') else ''}")
    print(f"  undecided predictions: {meta['n_undecided']}/17")


if __name__ == "__main__":
    main()

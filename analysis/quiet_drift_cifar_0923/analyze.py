#!/usr/bin/env python3
"""quiet_drift_cifar_0923 -- score the seven replays against PREDICTIONS.md (V1-V6, P1-P9).

Reads results/quiet_drift_cifar_0923/<bundle>/probe.npz and writes, next to them,
  per_seed.csv   one row per (bundle, seed): hit999, rest / burst / quiet growth, the pre-burst
                 window, its decomposition, bands, coherence, step size, K_eff, CE, Adam state
  profile.csv    per (bundle, 2,000-step block) seed medians of the drift and its probes
  verdict.json   V1-V6, the check, P1-P9 scored, the main-trace reproduction
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "quiet_drift_cifar_0923"
MAIN = Path("/home/issan/Projects/obsidian-research-data/altlabels_cifar_0923/results/"
            "altlabels_cifar_0923")
BUNDLES = ("I_next", "I_C", "I_rev", "I_same", "A_own", "A_C", "A_same")
REPRO = {"I_next": "LR_iid", "A_own": "LR_abab"}
BANDN = ("top", "mid1", "mid2", "low", "mu", "comp")
NEED, EP_MIN, PLUS, PROBE = 1199, 1190, 500, 100
SEEDS = range(10)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    rx = pd.Series(x[ok]).rank().to_numpy()
    ry = pd.Series(y[ok]).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1]), int(ok.sum())


def episodes(correct: np.ndarray, k0: int):
    """Burst episodes among probes k0..end: runs of correct < NEED whose minimum is < EP_MIN."""
    eps, k, K = [], k0, len(correct)
    while k < K:
        if correct[k] < NEED:
            j = k
            while j + 1 < K and correct[j + 1] < NEED:
                j += 1
            if correct[k:j + 1].min() < EP_MIN:
                eps.append((k, j))
            k = j + 1
        else:
            k += 1
    return eps


def slot(p: dict, s: int) -> dict:
    corr = p["correct"][:, s].astype(int)
    n1 = p["n1"][:, s]
    steps = p["step"]
    K = len(steps)
    H = int(p["hit999"][s])
    row = {"hit999": H}
    if H < 0:
        row["status"] = "no_hit"
        return row
    kh = (H + PLUS) // PROBE
    if kh >= K:
        row["status"] = "hit_too_late"
        return row
    eps = episodes(corr, kh)
    burst_iv = set()                         # interval index k means (k-1, k]
    for a, b in eps:
        burst_iv.update(range(a, b + 1))
        if b + 1 < K:
            burst_iv.add(b + 1)
    iv = [k for k in range(kh + 1, K)]
    dn = np.diff(n1)                         # dn[k-1] = interval (k-1, k]
    quiet = [k for k in iv if k not in burst_iv]
    burst = [k for k in iv if k in burst_iv]
    row.update(status="ok", rest=float(n1[-1] - n1[kh]), quiet=float(sum(dn[k - 1] for k in quiet)),
               burst=float(sum(dn[k - 1] for k in burst)), n_bursts=len(eps),
               first_burst_step=int(steps[eps[0][0]]) if eps else -1,
               min_correct_rest=int(corr[kh:].min()))
    # pre-burst quiet window: kh -> the start of the first burst interval
    ke = (eps[0][0] - 1) if eps else K - 1
    ke = max(ke, kh)
    row.update(pre_start=int(steps[kh]), pre_end=int(steps[ke]), pre_len=int(steps[ke] - steps[kh]),
               pre_growth=float(n1[ke] - n1[kh]))
    row["pre_rate"] = row["pre_growth"] / row["pre_len"] if row["pre_len"] > 0 else np.nan
    # decomposition at the end of the pre-burst window
    rad = p["rad_h"][ke, s]                  # <W_h, D> per band
    new = p["new_h"][ke, s]                  # ||D||^2 per band
    fd = p["fitdir_h"][ke, s]                # <F, D> per band
    row.update(radWh=float(2 * rad.sum()), newD=float(new.sum()), fitF=float(2 * fd.sum()),
               oldW0=float(2 * (rad.sum() - fd.sum())), fit_sq=float(p["fit_sq"][ke, s].sum()))
    tot = row["radWh"] + row["newD"]
    row["share_new"] = row["newD"] / tot if tot != 0 else np.nan
    row["decomp_check"] = tot - row["pre_growth"]
    for b, nm in enumerate(BANDN):
        row[f"pre_band_{nm}"] = float(2 * rad[b] + new[b])
    # the quiet drift split into bands (per interval: 2<W_prev, D100> + ||D100||^2 per band)
    r100, d100 = p["rad100"][:, s, :], p["band_D100"][:, s, :]
    for b, nm in enumerate(BANDN):
        row[f"quiet_band_{nm}"] = float(sum(2 * r100[k, b] + d100[k, b] for k in quiet))
    # 1,000-step windows fully inside the pre-burst window
    uu = p["ps_uu"][:, s]
    dwu = p["ps_dot_Wu"][:, s]
    kap = p["kappa_u1k"][:, s]
    sk, rms, cos, wn, kk = [], [], [], [], []
    for k in range(10, K, 10):
        t = int(steps[k])
        if t - 1000 < steps[kh] or t > steps[ke] or not np.isfinite(kap[k]):
            continue
        seg = slice(t - 1000, t)
        suu = uu[seg].sum()
        dist = np.sqrt(max(kap[k], 0) * 1000 * suu)
        sk.append(np.sqrt(max(kap[k], 0)))
        rms.append(np.sqrt(uu[seg].mean()))
        wn.append(np.sqrt(n1[k - 10]))
        cos.append(-dwu[seg].sum() / (np.sqrt(n1[k - 10]) * dist) if dist > 0 else np.nan)
        kk.append(kap[k])
    row.update(n_1k=len(sk), sqrt_kappa=float(np.median(sk)) if sk else np.nan,
               kappa=float(np.median(kk)) if kk else np.nan,
               rms_u=float(np.median(rms)) if rms else np.nan,
               wnorm=float(np.median(wn)) if wn else np.nan,
               cos_1k=float(np.nanmedian(cos)) if cos else np.nan)
    # probes in the pre-burst window
    sel = slice(kh, ke + 1)
    ke_ = p["k_eff"][sel, s]
    ce = p["ce32"][sel, s]
    ce64 = p["resid_sum"][sel, s] / 1200.0
    row.update(keff_med=float(np.median(ke_)), log_keff_med=float(np.median(np.log(ke_))),
               ce32_med=float(np.median(ce)),
               log_ce32_med=float(np.median(np.log(np.where(ce > 0, ce, np.nan)))) if (ce > 0).any()
               else -np.inf, ce32_zero_frac=float((ce == 0).mean()),
               log_resid_mean_med=float(np.median(np.log(ce64))),
               keff32_med=float(np.median(p["k_eff32"][sel, s])),
               nact2_med=float(np.median(p["n_act"][sel, s, 1])),
               margin_med_start=float(np.median(p["margins"][kh, s])),
               margin_min_start=float(np.min(p["margins"][kh, s])),
               margin_q10_start=float(np.quantile(p["margins"][kh, s], 0.1)),
               sqrtv_med=float(np.median(p["sqrt_v_q"][sel, s, 2])),
               frac_sqrtv_lt_eps=float(np.median(p["frac_sqrtv_lt_eps"][sel, s, 0])),
               frac_sqrtv_lt_10eps=float(np.median(p["frac_sqrtv_lt_eps"][sel, s, 1])),
               ratio_mean=float(np.median(p["ratio_mean"][sel, s])),
               kappa100=float(np.median(p["kappa_u"][kh + 1:ke + 1, s])) if ke > kh else np.nan,
               kappa_g100=float(np.median(p["kappa_g"][kh + 1:ke + 1, s])) if ke > kh else np.nan,
               n2_rest=float(p["n2"][-1, s] - p["n2"][kh, s]),
               n3_rest=float(p["n3"][-1, s] - p["n3"][kh, s]),
               n1_start=float(n1[0]), n1_h=float(n1[kh]), n1_end=float(n1[-1]))
    return row


def main() -> None:
    per, prof, repro = [], [], {}
    P = {}
    for b in BUNDLES:
        f = OUT / b / "probe.npz"
        if not f.exists():
            raise SystemExit(f"missing {f}")
        with np.load(f) as d:
            P[b] = {k: d[k] for k in d.files}
    for b in BUNDLES:
        p = P[b]
        for s in SEEDS:
            r = slot(p, s)
            r.update(bundle=b, seed=s)
            per.append(r)
        # 2,000-step blocks: seed medians
        n1 = p["n1"]
        for k0 in range(0, 300, 20):
            k1 = k0 + 20
            blk = {"bundle": b, "step_from": k0 * 100, "step_to": k1 * 100,
                   "dn1": float(np.median(n1[k1] - n1[k0])),
                   "ce32": float(np.median(p["ce32"][k1])),
                   "correct_min": int(p["correct"][k0 + 1:k1 + 1].min()),
                   "k_eff": float(np.median(p["k_eff"][k1])),
                   "kappa_u1k": float(np.nanmedian(p["kappa_u1k"][k1])),
                   "rms_u": float(np.median(np.sqrt(p["ps_uu"][k0 * 100:k1 * 100].mean(0)))),
                   "sqrtv_med": float(np.median(p["sqrt_v_q"][k1, :, 2])),
                   "frac_lt_eps": float(np.median(p["frac_sqrtv_lt_eps"][k1, :, 0])),
                   "margin_med": float(np.median(p["margin_med32"][k1]))}
            prof.append(blk)
        if b in REPRO:
            worst = {}
            for s in SEEDS:
                z = np.load(MAIN / REPRO[b] / "trace" / f"LR_std_seed{s}.npz")
                m = z["task"] == 49
                if not np.array_equal(z["step"][m], p["step"]):
                    worst["step_grid"] = "differs"
                    continue
                for kp, kt in (("n1", "n1"), ("ce32", "ce"), ("correct", "correct"),
                               ("margin_med32", "margin_med"), ("n2", "n2"), ("n3", "n3")):
                    dd = float(np.abs(p[kp][:, s].astype(float) - z[kt][m].astype(float)).max())
                    worst[kp] = max(worst.get(kp, 0.0), dd)
            repro[b] = {"max_abs_diff": worst,
                        "bit_identical": all(v == 0.0 for v in worst.values()
                                             if not isinstance(v, str))}
    df = pd.DataFrame(per)
    df.to_csv(OUT / "per_seed.csv", index=False, float_format="%.8g")
    pd.DataFrame(prof).to_csv(OUT / "profile.csv", index=False, float_format="%.6g")

    Q = df.pivot(index="seed", columns="bundle", values="quiet")
    med = lambda x: float(np.nanmedian(x))                                   # noqa: E731
    v = {"bundle_medians": {b: {c: med(df[df.bundle == b][c]) for c in
                                ("hit999", "rest", "quiet", "burst", "n_bursts", "first_burst_step",
                                 "pre_len", "pre_growth", "pre_rate", "share_new", "radWh", "newD",
                                 "fitF", "oldW0", "sqrt_kappa", "kappa", "rms_u", "wnorm", "cos_1k",
                                 "keff_med", "ce32_med", "sqrtv_med", "frac_sqrtv_lt_eps",
                                 "ratio_mean", "kappa100", "kappa_g100", "n2_rest", "n3_rest",
                                 "margin_med_start", "margin_min_start", "margin_q10_start")}
                            for b in BUNDLES},
         "reproduction": repro}
    # V1
    den = Q["I_C"] - Q["A_own"]
    lam = ((Q["A_C"] - Q["A_own"]) / den).where(den > 0)
    lm = med(lam)
    v["V1"] = {"lambda_per_seed": lam.round(4).tolist(), "excluded": int((den <= 0).sum()),
               "lambda_median": lm,
               "lambda_from_medians": (med(Q["A_C"]) - med(Q["A_own"])) / (med(Q["I_C"]) - med(Q["A_own"])),
               "verdict": "LABEL_DOMINANT" if lm >= 2 / 3 else
               "NETWORK_DOMINANT" if lm <= 1 / 3 else "MIXED"}
    # V2
    rr = Q["I_rev"] / Q["I_C"]
    v["V2"] = {"rho_rev_per_seed": rr.round(4).tolist(), "rho_rev_median": med(rr),
               "verdict": "REVISIT_HALVES" if med(rr) < 0.5 else "REVISIT_LESS_THAN_HALVES"}
    # V3
    ri, ra = Q["I_same"] / Q["I_C"], Q["A_same"] / Q["A_C"]
    v["V3"] = {"rho_same_I": med(ri), "rho_same_A": med(ra),
               "verdict": "NO_SWITCH_SMALL" if (med(ri) < 0.5 and med(ra) < 0.5)
               else "NO_SWITCH_NOT_SMALL"}
    # V4
    ic = df[df.bundle == "I_C"]
    v["V4"] = {"fitF_median": med(ic.fitF), "oldW0_median": med(ic.oldW0),
               "share_new_median": med(ic.share_new),
               "verdict": "FIT_DIRECTION" if med(ic.fitF) > med(ic.oldW0) else "OLD_WEIGHTS"}
    # V5
    ao = df[df.bundle == "A_own"]
    lr = {k: float(np.log(med(ic[k]) / med(ao[k]))) for k in ("sqrt_kappa", "rms_u", "wnorm", "cos_1k")
          if med(ao[k]) > 0 and med(ic[k]) > 0}
    v["V5"] = {"log_ratio_I_C_over_A_own": lr,
               "ratio_I_C_over_A_own": {k: float(np.exp(x)) for k, x in lr.items()},
               "pre_rate_ratio": med(ic.pre_rate) / med(ao.pre_rate),
               "verdict": "COHERENCE" if abs(lr.get("sqrt_kappa", 0)) > abs(lr.get("rms_u", 0))
               else "STEP_SIZE"}
    # V6
    ok = df[df.status == "ok"]
    rk = spearman(ok.pre_rate, ok.log_keff_med)
    rc = spearman(ok.pre_rate, ok.log_ce32_med)
    rc64 = spearman(ok.pre_rate, ok.log_resid_mean_med)
    v["V6"] = {"spearman_rate_logKeff": rk, "spearman_rate_logCE32": rc,
               "spearman_rate_logCE64_record": rc64,
               "verdict": "KEFF_BETTER" if abs(rk[0]) > abs(rc[0]) else "CE_BETTER"}
    chk = Q["I_C"] / Q["I_next"]
    v["check"] = {"Q_IC_over_Q_Inext_median": med(chk),
                  "ok": bool(0.5 <= med(chk) <= 2.0)}
    pred = [("P1", "V1", "LABEL_DOMINANT", 0.40), ("P2", "V1", "MIXED", 0.45),
            ("P3", "V1", "NETWORK_DOMINANT", 0.15), ("P4", "V2", "REVISIT_HALVES", 0.35),
            ("P5", "V3", "NO_SWITCH_SMALL", 0.70), ("P6", "V4", "FIT_DIRECTION", 0.60),
            ("P7", "V5", "COHERENCE", 0.55), ("P8", "V6", "KEFF_BETTER", 0.60)]
    sc = []
    for name, q, lab, pr in pred:
        hit = v[q]["verdict"] == lab
        sc.append({"P": name, "predicate": f"{q} = {lab}", "p": pr, "outcome": hit,
                   "brier": (pr - float(hit)) ** 2})
    hit = v["check"]["ok"]
    sc.append({"P": "P9", "predicate": "Q(I_C)/Q(I_next) in [1/2, 2]", "p": 0.90, "outcome": hit,
               "brier": (0.90 - float(hit)) ** 2})
    v["scores"] = sc
    v["brier_mean"] = float(np.mean([q["brier"] for q in sc]))
    (OUT / "verdict.json").write_text(json.dumps(v, indent=1, default=float))
    for k in ("V1", "V2", "V3", "V4", "V5", "V6"):
        print(k, v[k]["verdict"])
    print("check", v["check"], "repro", {b: r["bit_identical"] for b, r in repro.items()})
    print("brier", round(v["brier_mean"], 3), [(q["P"], q["outcome"]) for q in sc])


if __name__ == "__main__":
    sys.exit(main())

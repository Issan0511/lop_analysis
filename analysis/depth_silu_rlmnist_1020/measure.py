#!/usr/bin/env python3
"""Per-layer measurements of depth_silu_rlmnist_1020 (spec section 5), float64, from the saved weights.

    OMP_NUM_THREADS=1 python analysis/depth_silu_rlmnist_1020/measure.py --runs raw/depth_silu_rlmnist_1020/runs --out raw/depth_silu_rlmnist_1020/units --jobs 8

For each run file <name>.npz it writes <out>/units_<name>.npz (per-unit arrays) and <out>/rows_<name>.json (the
per-layer, per-task summaries that summarize.py aggregates).  Points: init, mid_k (+200 updates of task k), end_k.
Windows: switch = end_{k-1} (init for k = 1) -> mid_k; rest = mid_k -> end_k.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
U64 = 2.0 ** -53


# ----------------------------------------------------------------------------- float64 model
def sigmoid(z):
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def phi(z, act):
    if act == "SILU": return z * sigmoid(z)
    if act == "R": return np.maximum(z, 0.0)
    if act == "LR": return np.where(z > 0, z, 0.1 * z)
    if act == "ELU": return np.where(z > 0, z, np.expm1(np.minimum(z, 0.0)))
    raise ValueError(act)


def dphi(z, act):
    if act == "SILU":
        s = sigmoid(z); return s * (1.0 + z * (1.0 - s))
    if act == "R": return (z > 0).astype(np.float64)
    if act == "LR": return np.where(z > 0, 1.0, 0.1)
    if act == "ELU": return np.where(z > 0, 1.0, np.exp(np.minimum(z, 0.0)))
    raise ValueError(act)


def forward64(P, x64, act, L):
    """Lists of the layer inputs h_{l-1} and preactivations z_l (float64), l = 1..L."""
    hs, zs = [], []
    h = x64
    for l in range(L):
        z = h @ P[2 * l].T + P[2 * l + 1]
        hs.append(h); zs.append(z)
        h = phi(z, act)
    return hs, zs


def gamma(k, u=U64):
    return k * u / (1.0 - k * u)


# ----------------------------------------------------------------------------- per-layer statistics
def layer_stats(h_in, z, act):
    """Per-unit position / response arrays and the layer's carrier numbers."""
    N = z.shape[0]
    mu = z.mean(0); s = z.std(0)
    r = np.where(s > 0, mu / np.where(s > 0, s, 1.0), np.nan)
    p = (z > 0).mean(0)
    g = dphi(z, act)
    S1 = np.abs(g).sum(0); S2 = (g * g).sum(0)
    neff = np.where(S2 > 0, S1 * S1 / (N * np.where(S2 > 0, S2, 1.0)), 0.0)
    hbar = h_in.mean(0)
    K = h_in @ hbar + 1.0
    Km, Ks = float(K.mean()), float(K.std())
    vin = float(((h_in - hbar) ** 2).mean(0).sum())
    return {"mu": mu, "s": s, "r": r, "p": p, "gmean": g.mean(0), "gabs": np.abs(g).mean(0), "neff": neff}, \
           {"hbar_norm": float(np.linalg.norm(hbar)), "Kmean": Km, "Ksd": Ks,
            "c": (Km / Ks) if Ks > 0 else float("inf"), "vin": vin}


def decompose(Wa, ba, Wb, bb, hbar_a, hbar_b, drop_cross=False):
    """Delta mu = (dW.hbar_a + db) + W_a.dhbar + dW.dhbar, exact up to rounding; the residual and its bound
    4 gamma_{d+2} (sum of the absolute products) per unit (spec S-ident)."""
    d = Wa.shape[1]
    dW, db, dh = Wb - Wa, bb - ba, hbar_b - hbar_a
    mu_a = Wa @ hbar_a + ba; mu_b = Wb @ hbar_b + bb
    own = dW @ hbar_a + db; up = Wa @ dh; cross = dW @ dh
    dmu = mu_b - mu_a
    resid = dmu - (own + up + (0.0 if drop_cross else cross))
    aWa, aWb, adW, ahA, ahB, adh = np.abs(Wa), np.abs(Wb), np.abs(dW), np.abs(hbar_a), np.abs(hbar_b), np.abs(dh)
    bound = 4.0 * gamma(d + 2) * (aWa @ ahA + aWb @ ahB + adW @ ahA + aWa @ adh + adW @ adh
                                 + np.abs(ba) + np.abs(bb) + np.abs(db))
    return {"dmu": dmu, "own": own, "up": up, "cross": cross, "resid": resid, "bound": bound}


def selfform_projection(h_a, z_a, act, dW, db, hbar_a):
    """The switch window's own term split as in chargpt decomp_dmu: d_i = sum_n K_n phi phi'(z_in) [h_n; 1],
    self_i = (dtheta_i . dhat_i)(dhat_i . [hbar_a; 1])."""
    K = h_a @ hbar_a + 1.0
    pp = phi(z_a, act) * dphi(z_a, act)                        # [N, U]
    ha1 = np.concatenate([h_a, np.ones((h_a.shape[0], 1))], 1)  # [N, d+1]
    D = (pp * K[:, None]).T @ ha1                              # [U, d+1]
    nrm = np.linalg.norm(D, axis=1)
    Dh = D / np.where(nrm > 0, nrm, 1.0)[:, None]
    dth = np.concatenate([dW, db[:, None]], 1)
    hb1 = np.concatenate([hbar_a, [1.0]])
    proj = (dth * Dh).sum(1) * (Dh @ hb1)
    cos = (dth * Dh).sum(1) / np.where(np.linalg.norm(dth, axis=1) > 0, np.linalg.norm(dth, axis=1), 1.0)
    return np.where(nrm > 0, proj, 0.0), np.where(nrm > 0, cos, np.nan)


# ----------------------------------------------------------------------------- returns and learning
def return_counts(pT, upto):
    """pT [T, U] task-end open fractions.  plain: p < 1/2 at some task end, p >= 1/2 at a later one (units);
    events: consecutive task ends with p < 1/2 then p >= 1/2; deep: p <= 0.1 then later p >= 1/2 (units)."""
    p = pT[:upto]
    below = np.maximum.accumulate(p < 0.5, axis=0)
    deep = np.maximum.accumulate(p <= 0.1, axis=0)
    later_up = p[1:] >= 0.5
    plain_units = (below[:-1] & later_up).any(0)
    deep_units = (deep[:-1] & later_up).any(0)
    events = int(((p[:-1] < 0.5) & (p[1:] >= 0.5)).sum())
    ever_below = (p < 0.5).any(0); ever_deep = (p <= 0.1).any(0)
    return {"plain_units": int(plain_units.sum()), "deep_units": int(deep_units.sum()), "events": events,
            "ever_below": int(ever_below.sum()), "ever_deep": int(ever_deep.sum())}, plain_units, deep_units


def learning(d):
    hits = d["hits"].astype(np.float64); steps = hits.shape[1]
    A = hits.sum(1) / (steps * 16); Hh = hits[:, :200].sum(1) / (200 * 16); A_last = hits[:, -1000:].sum(1) / (1000 * 16)
    F = np.array([np.bincount(y, minlength=10).max() / y.size for y in d["y"]])
    return A, Hh, A_last, F


def gap(A, lo, hi):
    T = len(A)
    if T < hi:
        return float("nan")
    return float(A[0] - A[lo - 1:hi].mean())


def floored(A, F, lo, hi):
    if len(A) < hi:
        return None
    w = slice(lo - 1, hi)
    thr = F[w].mean() + 3.0 * F[w].std(ddof=1) / math.sqrt(hi - lo + 1)
    return bool(A[w].mean() <= thr), float(A[w].mean()), float(thr)


# ----------------------------------------------------------------------------- one run
def nanmed(a):
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


def measure_run(path: Path, out: Path):
    from src import depth_silu_rlmnist_1020 as E
    d = np.load(path)
    act, L, seed, door = str(d["act"]), int(d["L"]), int(d["seed"]), str(d["door"])
    name = path.stem
    T = d["end0"].shape[0]
    _, _, X = E.load_inputs(seed, door)
    if E.sha(X) != str(d["x_sha256"]):
        raise RuntimeError(f"{name}: the x of the analysis is not the x of the run")
    x64 = X.double().numpy()
    npar = 2 * L + 2
    getP = lambda key, k: [d[f"{key}{j}"][k].astype(np.float64) for j in range(npar)]
    P_init = [d[f"init{j}"].astype(np.float64) for j in range(npar)]
    U = {}
    keys_u = ("mu", "s", "r", "p", "gmean", "gabs", "neff")
    for pt in ("end", "mid"):
        for l in range(1, L + 1):
            for k in keys_u:
                U[f"{pt}_{k}_l{l}"] = np.full((T, 100), np.nan)
            for k in ("hbar_norm", "Kmean", "Ksd", "c", "vin"):
                U[f"{pt}_{k}_l{l}"] = np.full(T, np.nan)
    for w in ("sw", "rest"):
        for l in range(1, L + 1):
            for k in ("dmu", "own", "up", "cross", "resid", "bound", "s_a", "p_a", "self", "cos_self"):
                U[f"{w}_{k}_l{l}"] = np.full((T, 100), np.nan)
    init_layer = {}
    hs_i, zs_i = forward64(P_init, x64, act, L)
    for l in range(L):
        u, lay = layer_stats(hs_i[l], zs_i[l], act)
        init_layer[l + 1] = {"u": u, "lay": lay}
    prev = (P_init, hs_i, zs_i)                 # the state at the start of task k's switch window
    for k in range(T):
        Pm, Pe = getP("mid", k), getP("end", k)
        hs_m, zs_m = forward64(Pm, x64, act, L)
        hs_e, zs_e = forward64(Pe, x64, act, L)
        for pt, hs, zs in (("mid", hs_m, zs_m), ("end", hs_e, zs_e)):
            for l in range(L):
                u, lay = layer_stats(hs[l], zs[l], act)
                for key in keys_u:
                    U[f"{pt}_{key}_l{l + 1}"][k] = u[key]
                for key, v in lay.items():
                    U[f"{pt}_{key}_l{l + 1}"][k] = v
        for w, (Pa, ha, za), (Pb, hb, zb) in (("sw", prev, (Pm, hs_m, zs_m)), ("rest", (Pm, hs_m, zs_m), (Pe, hs_e, zs_e))):
            for l in range(L):
                Wa, ba, Wb, bb = Pa[2 * l], Pa[2 * l + 1], Pb[2 * l], Pb[2 * l + 1]
                hbar_a, hbar_b = ha[l].mean(0), hb[l].mean(0)
                dec = decompose(Wa, ba, Wb, bb, hbar_a, hbar_b)
                for key, v in dec.items():
                    U[f"{w}_{key}_l{l + 1}"][k] = v
                U[f"{w}_s_a_l{l + 1}"][k] = za[l].std(0)
                U[f"{w}_p_a_l{l + 1}"][k] = (za[l] > 0).mean(0)
                if w == "sw":
                    sf, cs = selfform_projection(ha[l], za[l], act, Wb - Wa, bb - ba, hbar_a)
                    U[f"sw_self_l{l + 1}"][k] = sf; U[f"sw_cos_self_l{l + 1}"][k] = cs
        prev = (Pe, hs_e, zs_e)
    A, Hh, A_last, F = learning(d)
    U |= {"A": A, "H200": Hh, "A_last1000": A_last, "F": F, "acc_end": d["acc"]}
    for l in range(1, L + 1):
        for key, v in init_layer[l]["u"].items():
            U[f"init_{key}_l{l}"] = v
        for key, v in init_layer[l]["lay"].items():
            U[f"init_{key}_l{l}"] = np.array(v)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"units_{name}.npz", **U)
    rows = summarize_run(U, act, L, seed, door, name, T)
    (out / f"rows_{name}.json").write_text(json.dumps(rows))
    return name, rows["ident"]


def summarize_run(U, act, L, seed, door, name, T):
    """Per-layer, per-task summaries plus the run-level counts (spec section 5)."""
    per = []
    for k in range(T):
        for l in range(1, L + 1):
            row = {"task": k + 1, "layer": l}
            r = U[f"end_r_l{l}"][k]; s = U[f"end_s_l{l}"][k]; p = U[f"end_p_l{l}"][k]
            row |= {"r_med": nanmed(r), "mu_med": nanmed(U[f"end_mu_l{l}"][k]), "s_med": nanmed(s), "p_mean": float(np.mean(p)),
                    "dead": float(np.mean(p == 0)), "g_mean": float(np.mean(U[f"end_gmean_l{l}"][k])),
                    "gabs_mean": float(np.mean(U[f"end_gabs_l{l}"][k])), "neff_mean": float(np.mean(U[f"end_neff_l{l}"][k])),
                    "below_half": float(np.mean(p < 0.5))}
            for key in ("hbar_norm", "Kmean", "Ksd", "c", "vin"):
                row[key] = float(U[f"end_{key}_l{l}"][k])
            for w in ("sw", "rest"):
                sa = U[f"{w}_s_a_l{l}"][k]; ok = sa > 0
                if not ok.any():
                    continue
                nz = lambda key: U[f"{w}_{key}_l{l}"][k][ok] / sa[ok]
                dmu, own, up, cross = nz("dmu"), nz("own"), nz("up"), nz("cross")
                upc = up + cross
                den = np.abs(own).sum() + np.abs(upc).sum()
                live = U[f"{w}_p_a_l{l}"][k] > 0
                row |= {f"{w}_dmu_mean": float(dmu.mean()), f"{w}_dmu_med": float(np.median(dmu)),
                        f"{w}_own_mean": float(own.mean()), f"{w}_up_mean": float(up.mean()), f"{w}_cross_mean": float(cross.mean()),
                        f"{w}_U_abs": float(np.abs(upc).sum() / den) if den > 0 else float("nan"),
                        f"{w}_U_signed": float(upc.mean() / dmu.mean()) if dmu.mean() != 0 else float("nan"),
                        f"{w}_up_dominant": float(np.mean(np.abs(upc) > np.abs(own))),
                        f"{w}_down_live": float(np.mean(U[f"{w}_dmu_l{l}"][k][live] < 0)) if live.any() else float("nan"),
                        f"{w}_n_units": int(ok.sum())}
                if w == "sw":
                    sf = U[f"sw_self_l{l}"][k][ok] / sa[ok]
                    row |= {"sw_self_mean": float(sf.mean()), "sw_other_mean": float((own - sf).mean()),
                            "sw_cos_self_med": nanmed(U[f"sw_cos_self_l{l}"][k][ok])}
            per.append(row)
    ret = {}
    for l in range(1, L + 1):
        pT = U[f"end_p_l{l}"]
        for upto in (30, 50):
            if T >= upto:
                c, _, _ = return_counts(pT, upto)
                ret[f"l{l}_t{upto}"] = c
    # r_net: all hidden units pooled at task ends 30 and 50
    rnet = {}
    for t in (1, 5, 10, 20, 30, 50):
        if T >= t:
            rnet[str(t)] = nanmed(np.concatenate([U[f"end_r_l{l}"][t - 1] for l in range(1, L + 1)]))
    # the decomposition identity over the whole run
    worst, n_bad, n_tot, l1_up_max, l1_cross_max = 0.0, 0, 0, 0.0, 0.0
    for w in ("sw", "rest"):
        for l in range(1, L + 1):
            res = np.abs(U[f"{w}_resid_l{l}"]); bd = U[f"{w}_bound_l{l}"]
            n_bad += int((res > bd).sum()); n_tot += int(res.size)
            worst = max(worst, float(np.nanmax(res / np.where(bd > 0, bd, np.inf))))
            if l == 1:
                l1_up_max = max(l1_up_max, float(np.abs(U[f"{w}_up_l1"]).max()))
                l1_cross_max = max(l1_cross_max, float(np.abs(U[f"{w}_cross_l1"]).max()))
    A, F = U["A"], U["F"]
    learn = {"A": A.tolist(), "H200": U["H200"].tolist(), "A_last1000": U["A_last1000"].tolist(), "F": F.tolist(),
             "G30": gap(A, 21, 30), "G50": gap(A, 41, 50), "floor_21_30": floored(A, F, 21, 30), "floor_41_50": floored(A, F, 41, 50)}
    return {"name": name, "act": act, "L": L, "seed": seed, "door": door, "T": T, "per": per, "returns": ret, "r_net": rnet,
            "learn": learn, "ident": {"n_resid_over_bound": n_bad, "n_total": n_tot, "max_resid_over_bound": worst,
                                      "l1_up_absmax": l1_up_max, "l1_cross_absmax": l1_cross_max}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    files = sorted(p for p in Path(a.runs).glob("*.npz") if not p.name.endswith(".tmp.npz"))
    if a.only:
        files = [p for p in files if p.stem in a.only]
    out = Path(a.out)
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for name, ident in ex.map(measure_run, files, [out] * len(files)):
            print(name, ident, flush=True)


if __name__ == "__main__":
    main()

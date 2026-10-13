"""TASK V (cobirth_seeds_1013, verifier, 1014): state-matched n_beta at EVERY phase-2 state for seed 1 (a0.1/0.3/0.5) and a0.2 (s0/s1),
from the verifier's sweep of the ORIGINAL p172c logic (verify/p172c_sweep/, 1 thread, slim) plus the agents' already computed states
(seed1/p172c/*_t*_real.json, slopes/p172c/*_T{39,99}_real.json; their copies were checked bit-identical to the original logic).
For comparison the same quantities are computed for seed 0 from nbeta_time (D) files.
Convention: M1, T = 1..100, state end0[T-1]; p172c 't' = T + 1.  mu_i, s_i^2 (ddof=1) from the same 90-task N_i series [in-sample].
Outputs per condition: n_beta distribution over (i, T in 11..100 alive); (N*') gaps; state-matched Cantelli per (i,T) max, per-target
time-mean max and Sigma_i; empirical N >= n_beta; per-target-min form of (B')+(N*') with Fbar = measured max F_i, 3, 6.4;
holds (A <= 0 < self) count; min margin n_beta(i,T) - break-even n*_i (14% and the arm threshold); adjacent-state rank correlation
of n_beta (median over pairs T, T+1).
usage: verify_sweep_cantelli.py"""
import os, json, glob, re, math
import numpy as np
V = "/home/issan/Projects/obsidian-research-data/cobirth_seeds_1013"
THR = {"a01": 0.14, "a03": 0.11, "a05": 0.08, "a02": None}
def rd(path):
    j = json.load(open(path)); return {int(r["i"]): (float(r["n_b"]), float(r["A"]), float(r["self"])) for r in j["real"]}
def gather(tag):
    NB = {}
    if tag.endswith("s0") and tag[:3] in ("a01", "a03", "a05"):
        for f in glob.glob(f"{V}/nbeta_time/grid/p172c_{tag[:3]}_T*_real.json") + glob.glob(f"{V}/nbeta_time/full/p172c_{tag[:3]}_T*_real_slim.json"):
            NB[int(re.search(r"_T(\d+)_", f).group(1))] = rd(f)
        return NB
    for f in glob.glob(f"{V}/verify/p172c_sweep/p172c_{tag}_t*_real.json"): NB[int(re.search(r"_t(\d+)_", f).group(1)) - 1] = rd(f)
    if tag in ("a01s1", "a03s1", "a05s1"):
        for f in glob.glob(f"{V}/seed1/p172c/p172c_{tag}_t*_real.json"): NB[int(re.search(r"_t(\d+)_", f).group(1)) - 1] = rd(f)
    else:
        m = {"a02s0": "a0.2_s0", "a02s1": "a0.2_s1"}[tag]
        for f in glob.glob(f"{V}/slopes/p172c/p172c_{m}_T*_real.json"): NB[int(re.search(r"_T(\d+)_", f).group(1))] = rd(f)
    return NB
def rank(x): return np.argsort(np.argsort(x)).astype(float)
res = {}
for tag in ["a01s0", "a03s0", "a05s0", "a01s1", "a03s1", "a05s1", "a02s0", "a02s1"]:
    NB = gather(tag); have = sorted(T for T in NB if 11 <= T <= 100); missing = [T for T in range(11, 101) if T not in NB]
    st = json.load(open(f"{V}/verify/vstats_{tag}.json")); ids = st["per_target"]["i"]; mu = dict(zip(ids, st["per_target"]["mu"])); var = dict(zip(ids, st["per_target"]["var"]))
    N = np.load(f"{V}/verify/vcounts_{tag}.npz")["N"]; Fbar = max(var[i] / mu[i] for i in ids if mu[i] > 0)
    nbs = []; gaps = []; cmax = (0.0, None); tmean = {}; emp = 0; rmax = 0.0; holds = []; nbmin = {}; marg14 = (np.inf, None); margA = (np.inf, None)
    thr = THR[tag[:3]]
    for T in have:
        for i, (nb, A, s) in NB[T].items():
            if A <= 0 < s: holds.append((T, i, A / s))
            if i not in mu or N[T - 1, i] < 0: continue
            n = N[T - 1, i]; nbs.append(nb); gaps.append((nb - mu[i], i, T)); nbmin[i] = min(nbmin.get(i, np.inf), nb)
            c = var[i] / (var[i] + (nb - mu[i]) ** 2) if nb > mu[i] else 1.0; tmean.setdefault(i, []).append(c)
            if c > cmax[0]: cmax = (c, (i, T, round(nb, 3), round(mu[i], 3)))
            emp += int(n >= nb); rmax = max(rmax, n / nb)
            be14 = mu[i] + math.sqrt(var[i]) * math.sqrt(0.86 / 0.14)
            if nb - be14 < marg14[0]: marg14 = (nb - be14, (i, T))
            if thr is not None:
                beA = mu[i] + math.sqrt(var[i]) * math.sqrt((1 - thr) / thr)
                if nb - beA < margA[0]: margA = (nb - beA, (i, T))
    nbs = np.array(nbs); tm = {i: float(np.mean(v)) for i, v in tmean.items()}; itm = max(tm, key=tm.get)
    def form(Fb):
        best = (0.0, None)
        for i in nbmin:
            v = (Fb * mu[i]) if Fb is not None else var[i]; c = v / (v + (nbmin[i] - mu[i]) ** 2) if nbmin[i] > mu[i] else 1.0
            if c > best[0]: best = (c, i)
        return best
    # adjacent-state rank correlation of n_beta over units alive at both
    rc = []
    for T in have:
        if T + 1 in NB and 11 <= T + 1 <= 100:
            common = sorted(set(NB[T]) & set(NB[T + 1]))
            if len(common) > 10: rc.append(np.corrcoef(rank([NB[T][i][0] for i in common]), rank([NB[T + 1][i][0] for i in common]))[0, 1])
    g = min(gaps)
    res[tag] = dict(n_states=len(have), missing_states=missing, n_target_tasks=int(len(nbs)), nbeta_min=float(nbs.min()), nbeta_q10=float(np.quantile(nbs, .1)), nbeta_median=float(np.median(nbs)),
                    n_nbeta_lt4=int((nbs < 4).sum()), n_nbeta_lt10=int((nbs < 10).sum()), Nstar_min_gap=[float(g[0]), int(g[1]), int(g[2])],
                    cantelli_per_iT_max=[float(cmax[0]), cmax[1]], cantelli_time_mean_max=[tm[itm], int(itm)], cantelli_time_mean_sigma=float(sum(tm.values())),
                    emp_N_ge_nbeta=emp, max_N_over_nbeta=float(rmax), form_pertarget_min=dict(F_own=form(None), Fbar_meas=(Fbar, form(Fbar)), Fbar_3=form(3.0), Fbar_6p4=form(6.4)),
                    holds=holds, min_margin_nbeta_minus_breakeven14=[float(marg14[0]), marg14[1]], min_margin_nbeta_minus_breakeven_armthr=[float(margA[0]), margA[1]] if thr else None,
                    adjacent_rank_corr_median=float(np.median(rc)) if rc else None, adjacent_rank_corr_range=[float(np.min(rc)), float(np.max(rc))] if rc else None)
    r = res[tag]
    print(f"{tag}: states {len(have)} (missing {len(missing)}) | n_beta min {r['nbeta_min']:.3f} q10 {r['nbeta_q10']:.2f} med {r['nbeta_median']:.2f} | <4 {r['n_nbeta_lt4']} <10 {r['n_nbeta_lt10']} | N*' min gap {g[0]:.3f} (u{g[1]} T{g[2]}) | "
          f"Cantelli per-(i,T) max {cmax[0]:.4f} {cmax[1]} | time-mean max {tm[itm]:.4f} (u{itm}) Sigma {sum(tm.values()):.3f} | emp N>=n_beta {emp} max N/n_beta {rmax:.3f} | "
          f"form(per-target min): own F {r['form_pertarget_min']['F_own'][0]:.4f}, Fbar meas {Fbar:.2f} {r['form_pertarget_min']['Fbar_meas'][1][0]:.4f}, Fbar 3 {r['form_pertarget_min']['Fbar_3'][0]:.4f}, Fbar 6.4 {r['form_pertarget_min']['Fbar_6p4'][0]:.4f} | "
          f"holds {holds} | margin n_beta - n*_be14 {marg14[0]:.3f} {marg14[1]} armthr {margA[0] if thr else float('nan'):.3f} | adj rank corr med {r['adjacent_rank_corr_median']}", flush=True)
json.dump(res, open(f"{V}/verify/verify_sweep_cantelli.json", "w"), indent=1, default=str)

# ---- addendum: time-average form with a uniform Fbar (sigma^2 = Fbar mu), and the largest N/n_beta case ----
add = {}
for tag in ["a01s0", "a03s0", "a05s0", "a01s1", "a03s1", "a05s1", "a02s0", "a02s1"]:
    NB = gather(tag); st = json.load(open(f"{V}/verify/vstats_{tag}.json")); ids = st["per_target"]["i"]; mu = dict(zip(ids, st["per_target"]["mu"]))
    N = np.load(f"{V}/verify/vcounts_{tag}.npz")["N"]; out = {}
    for Fb in (3.0, 6.4):
        tm = {}; pmax = 0.0
        for T in range(11, 101):
            for i, (nb, A, s) in NB[T].items():
                if i not in mu or N[T - 1, i] < 0: continue
                v = Fb * mu[i]; c = v / (v + (nb - mu[i]) ** 2) if nb > mu[i] else 1.0; tm.setdefault(i, []).append(c); pmax = max(pmax, c)
        tmv = {i: float(np.mean(x)) for i, x in tm.items()}; im = max(tmv, key=tmv.get)
        out[f"Fbar{Fb:g}"] = dict(per_task_max=pmax, time_mean_max=tmv[im], target=int(im), sigma=float(sum(tmv.values())))
    best = (0, None)
    for T in range(11, 101):
        for i, (nb, A, s) in NB[T].items():
            if i in mu and N[T - 1, i] >= 0 and N[T - 1, i] / nb > best[0]: best = (N[T - 1, i] / nb, (int(i), T, int(N[T - 1, i]), round(nb, 3)))
    out["max_N_over_nbeta_case"] = best; add[tag] = out
    print(tag, {k: (round(v["per_task_max"], 4), round(v["time_mean_max"], 4), v["target"], round(v["sigma"], 3)) if isinstance(v, dict) else v for k, v in out.items()}, flush=True)
res["addendum_uniform_Fbar_time_mean"] = add
json.dump(res, open(f"{V}/verify/verify_sweep_cantelli.json", "w"), indent=1, default=str)

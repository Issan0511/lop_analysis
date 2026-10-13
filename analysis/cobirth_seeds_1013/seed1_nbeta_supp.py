"""seed1_nbeta_supp.py (cobirth_seeds_1013, TASK A 'seed1'): (i) p172c anchor check (seed 0 a0.3 T 99 n_beta, generalized copy vs the saved json);
(ii) seed-1 n_beta across the computed p172c states: rank correlation of per-target n_beta between states, Spearman(N_i(T), n_beta(i,T)) at the states,
and an independence-pairing estimate of P(N >= n_beta) [assumption: n_beta(i,T) drawn from the pooled n_beta of the arm independently of N_i(T)].
M1 convention: T = 1..100, state end0[T-1]; p172c file t = T + 1.  Writes RAW/p172c_anchor_check.json and RAW/nbeta_pairing_supp.json."""
import os, json
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "1"
import numpy as np
RAW = "/home/issan/Projects/obsidian-research-data/cobirth_seeds_1013/seed1"; P1 = RAW + "/p172c"
B0 = "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930/review/stage4_1002/block_1003/stage14_1007_h4/agent_bound"
def sp(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float); return float(np.corrcoef(rx, ry)[0, 1])
# (i) anchor
a = json.load(open(f"{B0}/p172c_a03_t100_real.json")); b = json.load(open(f"{P1}/p172c_a03s0anchor_t100_real.json"))
na = {r["i"]: r["n_b"] for r in a["real"]}; nb = {r["i"]: r["n_b"] for r in b["real"]}
rel = max(abs(na[i] - nb[i]) / abs(na[i]) for i in na)
anc = dict(saved=f"{B0}/p172c_a03_t100_real.json", recomputed=f"{P1}/p172c_a03s0anchor_t100_real.json", same_units=set(na) == set(nb), n_units=len(na), max_rel_diff_n_beta=rel,
           n_bitwise_equal=int(sum(na[i] == nb[i] for i in na)), min_saved=min(na.values()), min_recomputed=min(nb.values()), median_saved=float(np.median(list(na.values()))), median_recomputed=float(np.median(list(nb.values()))),
           match=bool(set(na) == set(nb) and rel < 1e-12), note="generalized copy runs 1 BLAS thread (original 2); differences are float rounding")
json.dump(anc, open(f"{RAW}/p172c_anchor_check.json", "w"), indent=1); print("p172c anchor:", anc)
# (ii) seed-1 n_beta across states
out = {}
for tag, Ts in (("a01s1", (20, 39, 50, 70)), ("a03s1", (20, 50, 70, 99)), ("a05s1", (20, 50, 70, 99))):
    z = np.load(f"{RAW}/counts_{tag}.npz"); N = z["N"]; k = z["k"]; nbt = {}
    for T in Ts: nbt[T] = {r["i"]: r["n_b"] for r in json.load(open(f"{P1}/p172c_{tag}_t{T + 1}_real.json"))["real"]}
    units = sorted(set().union(*[set(nbt[T]) for T in Ts])); M = np.array([[nbt[T].get(i, np.nan) for T in Ts] for i in units])
    rc = {}
    for x in range(len(Ts)):
        for y in range(x + 1, len(Ts)):
            ok = np.isfinite(M[:, x]) & np.isfinite(M[:, y]); rc[f"T{Ts[x]}-T{Ts[y]}"] = sp(M[ok, x], M[ok, y])
    xs, ys, ks = [], [], []
    for T in Ts:
        for i, v in nbt[T].items():
            if N[T - 1, i] >= 0: xs.append(N[T - 1, i]); ys.append(v); ks.append(k[T - 1, i])
    xs = np.array(xs, float); ys = np.array(ys); ks = np.array(ks, float); pool = np.sort(np.array([v for T in Ts for v in nbt[T].values()]))
    Nph = N[10:100]; vals = Nph[Nph >= 0].astype(float); frac = np.searchsorted(pool, vals, side="right") / len(pool)
    pt = np.array([(np.searchsorted(pool, Nph[:, i][Nph[:, i] >= 0].astype(float), side="right") / len(pool)).mean() for i in range(100) if (Nph[:, i] >= 0).sum() >= 30])
    mn = np.nanmin(M, 1)
    out[tag] = dict(states_T=list(Ts), rank_corr_between_states=rc, per_target_min_over_states=dict(min=float(mn.min()), median=float(np.median(mn))), per_target_max_over_min_median=float(np.nanmedian(np.nanmax(M, 1) / mn)),
                    spearman_N_nbeta_at_states=sp(xs, ys), spearman_k_nbeta_at_states=sp(ks, ys), n_pairs_at_states=int(len(xs)), n_N_ge_nbeta_at_states=int((xs >= ys).sum()), max_N_over_nbeta_at_states=float((xs / ys).max()),
                    pooled_nbeta=dict(n=int(len(pool)), min=float(pool.min()), q01=float(np.quantile(pool, .01)), q05=float(np.quantile(pool, .05)), q10=float(np.quantile(pool, .1)), median=float(np.median(pool)),
                                      frac_lt={str(c): float((pool < c).mean()) for c in (4, 5, 6, 8, 10)}),
                    independence_pairing=dict(mean_P_nbeta_le_N=float(frac.mean()), max_target_time_avg=float(pt.max()), sum_over_targets_per_task=float(pt.sum()), note="assumption: n_beta(i,T) ~ pooled arm distribution, independent of N_i(T)"))
    print(tag, json.dumps(out[tag]))
json.dump(out, open(f"{RAW}/nbeta_pairing_supp.json", "w"), indent=1)
# (iii) the (B')+(N*') form of cobirth_1013 README 4.1 on each condition: P(N_i >= n*_i) <= F mu_i / (F mu_i + (n*_i - mu_i)^2) with F = observed F_i, 3, 6.4;
#       n*_i = n_beta at the single reference state (M1 layout; missing -> median) and the per-target minimum over computed states (seed 1).
bp = {}
for tag in ("a01s1", "a03s1", "a05s1", "a01s0", "a03s0", "a05s0"):
    d = json.load(open(f"{RAW}/{tag}.json")); pt = d["presence"]["per_target"]; mu = np.array(pt["mu"]); F = np.array(pt["fano"]); spec = d["nbeta_spec"]; Tref = str(d["nbeta_ref_T"])
    nbs = {T: {r["i"]: r["n_b"] for r in json.load(open(p))["real"]} for T, p in spec.items()}
    v = np.array([nbs[Tref].get(i, np.nan) for i in pt["i"]]); single = np.where(np.isfinite(v), v, np.nanmedian(v))
    vmin = np.array([min([nbs[T][i] for T in nbs if i in nbs[T]], default=np.nan) for i in pt["i"]])
    rec = {}
    for nm, ns in (("single_state_T" + Tref, single), ("min_over_states", vmin if np.isfinite(vmin).all() else None)):
        if ns is None: continue
        gap = ns - mu; r_ = dict(min_gap_nstar_minus_mu=float(gap.min()), max_mu=float(mu.max()), n_states=len(nbs))
        for Fb in (None, 3.0, 6.4):
            FF = F if Fb is None else np.full_like(F, Fb); b = FF * mu / (FF * mu + gap ** 2)
            r_[f"F_{'obs' if Fb is None else Fb}"] = dict(max=float(b.max()), argmax_target=int(pt["i"][int(np.argmax(b))]), sum=float(b.sum()))
        rec[nm] = r_
    bp[tag] = rec; print(tag, json.dumps(rec))
sup = json.load(open(f"{RAW}/nbeta_pairing_supp.json")); sup["Bprime_Nstar"] = bp; json.dump(sup, open(f"{RAW}/nbeta_pairing_supp.json", "w"), indent=1)

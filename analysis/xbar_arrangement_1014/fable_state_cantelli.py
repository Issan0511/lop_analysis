"""Fable (1014): per-state Cantelli bound for the hold fraction with MODEL moments (THEORY_XBAR.md, 定理).
Given the brightness-stratified surrogate mean c_i(T) = N̄^strat_i(T) (agent B, = E[N_i | T_i(T), O_i(T)] under model M1+M2,
Poisson-binomial so Var ≤ mean) and the per-state necessary number n_β(i, T) (cobirth_seeds_1013, p172c), compute
    rho_model(i) = mean_T  c(T) / (c(T) + (n_β(i,T) − c(T))²)          [c(T) < n_β(i,T); else term = 1]
and compare with (a) the actual-marginal per-state form of cobirth_seeds (σ_i²/(σ_i² + (n_β(i,T) − μ_i)²), μ, σ² = marginal
sample moments of the actual N_i), (b) the thresholds 14/11/8 % (c = 2.8; a0.1/a0.3/a0.5, seed 0; a0.2 unknown → 14 % proxy).
Also: Spearman(c(T), n_β(i,T)) per target (does the model mean rise where n_β is small?), c in states with n_β < 10 vs all,
and the Monte-Carlo variance of the surrogate draws vs c (checks Var ≤ mean).
usage: fable_state_cantelli.py <cond> [iso|isoc]   with cond in a01s0 a03s0 a05s0 a01s1 a02s0 a02s1 a03s1 a05s1; the surrogate (iso = isotropic directions
with actual m_k, s⊥_k; isoc = iso + the actual common pattern component) supplies c_i(T) = surrogate mean of N_i(T) (agent B arrays).
"""
import os, sys, json, glob, re
import numpy as np
OBS = "/home/issan/Projects/obsidian-research-data"
NB_FULL = os.path.join(OBS, "cobirth_seeds_1013/nbeta_time/full")          # seed 0: p172c_{a01,a03,a05}_T{T}_real_slim.json (T = M1 task)
NB_SWEEP = os.path.join(OBS, "cobirth_seeds_1013/verify/p172c_sweep")      # others: p172c_{a01s1,...}_t{t}_real.json (t = T + 1)
B_DIR = os.path.join(OBS, "xbar_arrangement_1014/agent_b")
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "..", "results", "xbar_arrangement_1014")
THR = {"a01": 0.14, "a03": 0.11, "a05": 0.08, "a02": 0.14, "a003": 0.14}

def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0: return float("nan")
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y)); return float(np.corrcoef(rx, ry)[0, 1])

def load_nbeta(cond):
    """returns dict T -> dict unit -> (n_b, A, self)"""
    out = {}
    if cond.endswith("s0") and cond[:-2] in ("a01", "a03", "a05"):
        # seed 0: nbeta_time/full/ holds 81 states per arm; the other 9 (T = 10, 20, ..., 100) are in nbeta_time/grid/ (verifier V3, 1014)
        files = glob.glob(os.path.join(NB_FULL, f"p172c_{cond[:-2]}_T*_real_slim.json")) + glob.glob(os.path.join(NB_FULL, "..", "grid", f"p172c_{cond[:-2]}_T*_real*.json"))
        for f in files:
            T = int(re.search(r"_T(\d+)_", f).group(1)); d = json.load(open(f))
            if T in out: continue
            out[T] = {int(r["i"]): (float(r["n_b"]), float(r["A"]), float(r["self"])) for r in d["real"]}
    else:
        for f in glob.glob(os.path.join(NB_SWEEP, f"p172c_{cond}_t*_real.json")):
            t = int(re.search(r"_t(\d+)_", f).group(1)); T = t - 1; d = json.load(open(f))
            out[T] = {int(r["i"]): (float(r["n_b"]), float(r["A"]), float(r["self"])) for r in d["real"]}
    return out

def main(cond, surr="iso"):
    nb = load_nbeta(cond); assert nb, f"no n_beta files for {cond}"
    bf = os.path.join(B_DIR, f"b_arrays_{surr}_{cond}.npz") if surr != "strat" else os.path.join(B_DIR, f"b_arrays_{cond}.npz")
    assert os.path.exists(bf), f"no agent-B arrays {bf}"
    key = {"iso": "Niso", "isoc": "Nisoc", "strat": "Nstrat"}[surr]
    b = np.load(bf); N = b["N"]; Cs = b[key + "_mean"]; draws = b[key + "_draws"] if key + "_draws" in b.files else None; T_list = [int(t) for t in b["T_list"]]
    thr = THR[re.match(r"(a\d+)", cond).group(1)]
    rows = []
    for i in range(N.shape[1]):
        Ts = [T for T in T_list if T in nb and i in nb[T] and N[T_list.index(T), i] >= 0 and np.isfinite(nb[T][i][0])]
        if len(Ts) < 30: continue
        idx = [T_list.index(T) for T in Ts]
        n_act = N[idx, i].astype(float); c = Cs[idx, i].astype(float); nbeta = np.array([nb[T][i][0] for T in Ts]); Aself = np.array([nb[T][i][1] / nb[T][i][2] for T in Ts])
        mu, s2 = n_act.mean(), n_act.var(ddof=1)
        # model per-state Cantelli (Var <= mean for the Poisson-binomial given the state)
        term_model = np.where(c < nbeta, c / (c + (nbeta - c) ** 2), 1.0)
        # actual-marginal per-state form (cobirth_seeds 状態ごと)
        term_marg = np.where(mu < nbeta, s2 / (s2 + (nbeta - mu) ** 2), 1.0) if s2 > 0 else np.where(mu < nbeta, 0.0, 1.0)
        mc_var = draws[:, idx, i].astype(float).var(axis=0, ddof=1).mean() if draws is not None else float("nan")
        lo = nbeta < 10
        rows.append(dict(unit=i, n_states=len(Ts), mu_actual=mu, var_actual=s2, c_mean=float(c.mean()), c_max=float(c.max()), mc_var_mean=float(mc_var),
                         rho_model=float(term_model.mean()), rho_model_max_state=float(term_model.max()),
                         rho_marg=float(term_marg.mean()), rho_marg_max_state=float(term_marg.max()),
                         spearman_c_nbeta=spearman(c, nbeta), spearman_Nact_c=spearman(n_act, c), spearman_Aself_c=spearman(Aself, c),
                         nbeta_min=float(nbeta.min()), n_lt10=int(lo.sum()), c_lt10=float(c[lo].mean()) if lo.any() else float("nan"), Nact_lt10=float(n_act[lo].mean()) if lo.any() else float("nan"),
                         max_N_over_nbeta=float((n_act / nbeta).max()), max_c_over_nbeta=float((c / nbeta).max())))
    f = lambda k: np.array([r[k] for r in rows], float)
    summ = dict(cond=cond, surrogate=surr, threshold=thr, n_targets=len(rows), n_beta_states=len(nb), B_file=bf[0],
                rho_model=dict(median=float(np.median(f("rho_model"))), q90=float(np.quantile(f("rho_model"), .9)), max=float(f("rho_model").max()), n_above_thr=int((f("rho_model") > thr).sum()), max_state=float(f("rho_model_max_state").max())),
                rho_marg=dict(median=float(np.median(f("rho_marg"))), q90=float(np.quantile(f("rho_marg"), .9)), max=float(f("rho_marg").max()), n_above_thr=int((f("rho_marg") > thr).sum()), max_state=float(f("rho_marg_max_state").max())),
                spearman_c_nbeta=dict(median=float(np.nanmedian(f("spearman_c_nbeta"))), q10=float(np.nanquantile(f("spearman_c_nbeta"), .1)), q90=float(np.nanquantile(f("spearman_c_nbeta"), .9))),
                spearman_Nact_c=dict(median=float(np.nanmedian(f("spearman_Nact_c"))), q10=float(np.nanquantile(f("spearman_Nact_c"), .1)), q90=float(np.nanquantile(f("spearman_Nact_c"), .9))),
                spearman_Aself_c=dict(median=float(np.nanmedian(f("spearman_Aself_c")))),
                mc_var_over_c=dict(median=float(np.nanmedian(f("mc_var_mean") / f("c_mean")))),
                c_mean_median=float(np.median(f("c_mean"))), mu_actual_median=float(np.median(f("mu_actual"))),
                lt10=dict(n_targets_with=int((f("n_lt10") > 0).sum()), c_lt10_over_c=float(np.nanmedian(f("c_lt10") / f("c_mean"))), Nact_lt10_over_mu=float(np.nanmedian(f("Nact_lt10") / f("mu_actual")))),
                max_N_over_nbeta=float(f("max_N_over_nbeta").max()), max_c_over_nbeta=float(f("max_c_over_nbeta").max()))
    os.makedirs(RES, exist_ok=True)
    json.dump(dict(summary=summ, targets=rows), open(os.path.join(RES, f"fable_state_cantelli_{surr}_{cond}.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1, ensure_ascii=False))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "iso")

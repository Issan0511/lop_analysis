"""slopes_exceedance_check.py (cobirth_seeds_1013, TASK B 'slopes'): follow-up to slopes_counts.py.
(1) For the target-tasks where N_i(T) >= n*_i with n*_i taken from ANOTHER state (cross-state exceedances in the Cantelli blocks), p172c was rerun at
    that very state (slopes_p172c.py --T <T>): a0.03_s0 T54 (u28), a0.03_s1 T16 (u18), T21 (u96), T34 (u58). Report N, n_beta, A, self, hold at that state.
(2) Same-state check at every state with a p172c file: #targets with N_i(T) >= n_beta,i(T), #A <= 0, #hold (A <= 0 < self).
(3) Per-target n_beta stability between T39 and T99 (log correlation, ratio quantiles).
Writes RAW/exceedance_check.json.  Task convention M1 (T = state end0[T-1]); p172c's 'task' = T + 1."""
import os, sys, json, glob
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "1"
import numpy as np
L = "/home/issan/Projects/obsidian-research-data/cobirth_seeds_1013/slopes"
cases = [("a0.03_s0", 54, [28]), ("a0.03_s1", 16, [18]), ("a0.03_s1", 21, [96]), ("a0.03_s1", 34, [58])]
out = []
for c, T, us in cases:
    d = json.load(open(f"{L}/p172c/p172c_{c}_T{T}_real.json")); rec = {r["i"]: r for r in d["real"]}
    cnt = np.load(f"{L}/counts_{c}.npz"); N = cnt["N"][T - 1]; k = cnt["k"][T - 1]
    for u in us:
        r = rec[u]
        out.append(dict(cond=c, T=T, unit=u, N=int(N[u]), k=int(k[u]), n_b=r["n_b"], need_b=r["need_b"], sup_b=r["sup_b"], A=r["A"], Ag=r["Ag"], Ar=r["Ar"], self=r["self"], hold=bool(r["A"] <= 0 < r["self"])))
rows = []
for f in sorted(glob.glob(f"{L}/p172c/p172c_a0.*_real.json")):
    b = os.path.basename(f); c = b.split("_T")[0].replace("p172c_", ""); T = int(b.split("_T")[1].split("_")[0])
    d = json.load(open(f)); cnt = np.load(f"{L}/counts_{c}.npz"); N = cnt["N"][T - 1]
    nb = np.array([r["n_b"] for r in d["real"]]); A = np.array([r["A"] for r in d["real"]]); S = np.array([r["self"] for r in d["real"]]); ii = np.array([r["i"] for r in d["real"]]); Nn = N[ii]
    rows.append(dict(cond=c, T=T, alive=d["alive"], nb_min=float(nb.min()), nb_q10=float(np.quantile(nb, .1)), nb_median=float(np.median(nb)), A_le0=int((A <= 0).sum()), hold=int(((A <= 0) & (S > 0)).sum()),
                     N_max=int(Nn.max()), N_ge_nb=int((Nn >= nb).sum()), min_nb_minus_N=float((nb - Nn).min()), self_neg=int((S < 0).sum()), A_over_self_min=float((A / S).min()), A_over_self_median=float(np.median(A / S))))
stab = {}
for c in ["a0.03_s0", "a0.03_s1", "a0.2_s0", "a0.2_s1"]:
    a = {r["i"]: r["n_b"] for r in json.load(open(f"{L}/p172c/p172c_{c}_T39_real.json"))["real"]}; b = {r["i"]: r["n_b"] for r in json.load(open(f"{L}/p172c/p172c_{c}_T99_real.json"))["real"]}
    com = sorted(set(a) & set(b)); x = np.array([a[i] for i in com]); y = np.array([b[i] for i in com]); r = np.log(y / x)
    stab[c] = dict(n_common=len(com), corr_log=float(np.corrcoef(np.log(x), np.log(y))[0, 1]), ratio_T99_over_T39_q10_50_90=[float(v) for v in np.exp(np.quantile(r, [.1, .5, .9]))], only_T39=len(set(a) - set(b)), only_T99=len(set(b) - set(a)))
json.dump(dict(exceedances=out, same_state=rows, nbeta_stability_T39_T99=stab), open(f"{L}/exceedance_check.json", "w"), indent=1)
for x in out: print(x)
for x in rows: print(x)
for k, v in stab.items(): print(k, v)

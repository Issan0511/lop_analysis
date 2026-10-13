"""V3 (verifier): re-derive the parent's per-state Cantelli rho_model(i) = mean_T c(T)/(c(T) + (n_beta(i,T) - c(T))^2) (term = 1 if c >= n_beta)
from agent B's arrays (Niso_mean) and n_beta files; check the T = t - 1 convention of the sweep files against the weights.
Seed 0: nbeta_time/full/p172c_<a01|a03|a05>_T<T>_real_slim.json (T = M1 task; as the parent) and, as a variant, + nbeta_time/grid/ (T 20..100).
Others: verify/p172c_sweep/p172c_<cond>_t<t>_real.json with T = t - 1."""
import json, os, re, glob, sys
import numpy as np
from v_common import load, weights, CONDS, RES, OBS
NB = os.path.join(OBS, "cobirth_seeds_1013")
THR = {"a01": 0.14, "a03": 0.11, "a05": 0.08}

def nbeta_files(cond, with_grid):
    out = {}
    if cond.endswith("s0"):
        tag = cond[:-2]
        for f in glob.glob(os.path.join(NB, "nbeta_time/full", f"p172c_{tag}_T*_real_slim.json")):
            out[int(re.search(r"_T(\d+)_", f).group(1))] = f
        if with_grid:
            for f in glob.glob(os.path.join(NB, "nbeta_time/grid", f"p172c_{tag}_T*_real.json")):
                T = int(re.search(r"_T(\d+)_", f).group(1))
                if T >= 11: out[T] = f
    else:
        for f in glob.glob(os.path.join(NB, "verify/p172c_sweep", f"p172c_{cond}_t*_real.json")):
            out[int(re.search(r"_t(\d+)_", f).group(1)) - 1] = f
    return out

def run(cond, with_grid):
    files = nbeta_files(cond, with_grid)
    b = np.load(os.path.join(OBS, "xbar_arrangement_1014/agent_b", f"b_arrays_iso_{cond}.npz"))
    N = b["N"]; C = b["Niso_mean"]; TL = [int(t) for t in b["T_list"]]
    nb = {}; conv = []
    d, Xt, xbar, K, Kbar = load(cond)
    for T, f in sorted(files.items()):
        j = json.load(open(f)); nb[T] = {int(r["i"]): float(r["n_b"]) for r in j["real"]}
        # convention check: the file's 'task' t must equal T + 1 and its alive count / unit set must match end0[T-1]
        if T in (11, 50, 99) or len(conv) < 3:
            Z = Xt @ weights(d, T).T; alive = set(np.flatnonzero((Z > 0).sum(0) > 0).tolist())
            Z2 = Xt @ weights(d, T + 1).T if T < 100 else None; alive2 = set(np.flatnonzero((Z2 > 0).sum(0) > 0).tolist()) if Z2 is not None else None
            conv.append(dict(T=T, file_task=j.get("task"), file_alive=j.get("alive"), alive_end0_Tm1=len(alive), alive_end0_T=len(alive2) if alive2 else None,
                             units_in_file_subset_of_alive_Tm1=set(nb[T]).issubset(alive), units_in_file_subset_of_alive_T=(set(nb[T]).issubset(alive2) if alive2 else None),
                             n_units_file=len(nb[T])))
    rows = []
    for i in range(N.shape[1]):
        Ts = [T for T in TL if T in nb and i in nb[T] and N[TL.index(T), i] >= 0 and np.isfinite(nb[T][i])]
        if len(Ts) < 30: continue
        idx = [TL.index(T) for T in Ts]; c = C[idx, i].astype(float); nbe = np.array([nb[T][i] for T in Ts]); n_act = N[idx, i].astype(float)
        term = np.where(c < nbe, c / (c + (nbe - c) ** 2), 1.0)
        rc = np.argsort(np.argsort(c)); rn = np.argsort(np.argsort(nbe))
        sp = float(np.corrcoef(rc, rn)[0, 1]) if c.std() > 0 and nbe.std() > 0 else np.nan
        rows.append(dict(i=i, n=len(Ts), rho=float(term.mean()), maxterm=float(term.max()), sp_c_nbeta=sp, nbeta_min=float(nbe.min()),
                         n_ge_nbeta_actual=int((n_act >= nbe).sum())))
    rho = np.array([r["rho"] for r in rows]); thr = THR[re.match(r"(a\d+)", cond).group(1)]
    return dict(cond=cond, with_grid=with_grid, n_states=len(files), n_targets=len(rows), states_per_target_median=float(np.median([r["n"] for r in rows])),
                rho_median=float(np.median(rho)), rho_q90=float(np.quantile(rho, .9)), rho_max=float(rho.max()), n_above_thr=int((rho > thr).sum()), thr=thr,
                max_state_term=float(max(r["maxterm"] for r in rows)), sp_c_nbeta_median=float(np.nanmedian([r["sp_c_nbeta"] for r in rows])),
                actual_N_ge_nbeta=int(sum(r["n_ge_nbeta_actual"] for r in rows)), convention_checks=conv)

if __name__ == "__main__":
    out = {}
    for cond in ("a01s0", "a05s1"):
        for g in ((False, True) if cond.endswith("s0") else (False,)):
            r = run(cond, g); out[f"{cond}_grid{int(g)}"] = r
            p = json.load(open(os.path.join(RES, f"fable_state_cantelli_iso_{cond}.json")))["summary"]
            r["parent"] = dict(rho_model=p["rho_model"], n_targets=p["n_targets"], n_beta_states=p["n_beta_states"])
    json.dump(out, open(os.path.join(RES, "v3_cantelli.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))

"""V2 (verifier): independent re-implementation of the iso surrogate (THEORY §3 命題 4 / REPORT_B §8) and a fair variant.
iso : per phase-2 state T and alive unit k: g ~ N(0, I_785), g <- g - xbar (xbar.g)/Kbar, u = Xt g, u <- u * sperp_k / sd_n(u),
      z = m_k K_n/Kbar + u (actual m_k); candidate gates 1[z > 0]. Target's own T_i, O_i actual. Candidate eligibility = actual alive.
fair: same u, but the position is shifted per (unit, draw) so the realized open count equals the actual k (open = top-k of u_n/K_n,
      i.e. z = m'_k K_n/Kbar + u with m' chosen between the k-th and (k+1)-th value of -Kbar u_n/K_n).
R = 10 draws, rng default_rng(91400 + r) (independent of agent B's 2000 + r).
usage: v2_iso.py <cond>   (a01s0 | a03s0 | ELU_s0_E1)"""
import json, os, sys, time
import numpy as np
from v_common import load, weights, thresholds, bridge_counts, phase2_stats, CONDS, PH2, RES, OBS

def main(cond, R=10):
    t0 = time.time(); npz, gate = CONDS[cond]; d, Xt, xbar, K, Kbar = load(cond); N, D = Xt.shape; H = d["end0"].shape[1]
    NA = np.full((90, H), -1); CA = np.full((90, H), -1)
    NI = np.full((R, 90, H), -1); CI = np.full((R, 90, H), -1); NF = np.full((R, 90, H), -1); CF = np.full((R, 90, H), -1)
    p_act, p_iso = [], []; k_act_all = []
    rngs = [np.random.default_rng(91400 + r) for r in range(R)]
    for ti, T in enumerate(PH2):
        W = weights(d, T); Z = Xt @ W.T; O = Z > 0; k = O.sum(0); alive = k > 0
        T3 = thresholds(Z, K, gate); m = W @ xbar; U = Z - np.outer(K / Kbar, m); sp = U.std(0)
        NA[ti], CA[ti] = bridge_counts(O, T3, O, alive)
        p_act.append(k[alive] / N)
        for r in range(R):
            G = rngs[r].standard_normal((D, H)); G -= np.outer(xbar, xbar @ G) / Kbar
            Ui = Xt @ G; Ui *= (sp / Ui.std(0))[None, :]
            Oi = (np.outer(K / Kbar, m) + Ui) > 0
            NI[r, ti], CI[r, ti] = bridge_counts(O, T3, Oi, alive)
            if r == 0: p_iso.append(Oi.sum(0)[alive] / N)
            else: p_iso[-1] = np.r_[p_iso[-1], Oi.sum(0)[alive] / N] if False else p_iso[-1]
            # fair: open = top-k_actual of u/K per unit
            rat = Ui / K[:, None]; Of = np.zeros_like(O)
            for j in np.flatnonzero(alive):
                kj = int(k[j]); idx = np.argpartition(-rat[:, j], kj - 1)[:kj]; Of[idx, j] = True
            NF[r, ti], CF[r, ti] = bridge_counts(O, T3, Of, alive)
    # reproduce agent B's actual counts
    bB = np.load(os.path.join(OBS, "xbar_arrangement_1014/agent_b", f"b_arrays_iso_{cond}.npz"))
    match_actual = bool((bB["N"] == NA).all()) and bool((bB["C"][NA >= 0] == CA[NA >= 0]).all())
    act = dict(N=phase2_stats(NA), C=phase2_stats(CA))
    def summ(arrs):
        S = [phase2_stats(a) for a in arrs]
        return {k: dict(median=float(np.median([s[k] for s in S])), min=float(np.min([s[k] for s in S])), max=float(np.max([s[k] for s in S])))
                for k in ("mu_median", "pooled_mean", "pooled_fano", "max", "n_ge10")}
    out = dict(cond=cond, R=R, match_agentB_actual_counts=match_actual, actual=act,
               iso=dict(N=summ(NI), C=summ(CI)), fair=dict(N=summ(NF), C=summ(CF)),
               width=dict(p_actual_mean=float(np.concatenate(p_act).mean()), p_iso_mean_r0=float(np.concatenate(p_iso).mean()),
                          ratio_iso_over_actual=float(np.concatenate(p_iso).mean() / np.concatenate(p_act).mean())))
    for nm in ("iso", "fair"):
        for q in ("N", "C"):
            out[nm][q]["ratio_pooled_mean_to_actual"] = out[nm][q]["pooled_mean"]["median"] / act[q]["pooled_mean"]
            out[nm][q]["ratio_mu_median_to_actual"] = out[nm][q]["mu_median"]["median"] / act[q]["mu_median"]
    # agent B's reported iso numbers for comparison
    jb = json.load(open(os.path.join(RES, f"b_iso_{cond}.json")))
    out["agentB_iso"] = dict(N=dict(mu_median=jb["N"]["iso"]["summary"]["mu_median"] if "mu_median" in jb["N"]["iso"]["summary"] else None,
                                    summary_keys=list(jb["N"]["iso"]["summary"].keys())), width=jb["widths"]["ratio_iso_over_actual"])
    for q in ("N", "C"):
        s = jb[q]["iso"]["summary"]
        out["agentB_iso"][q] = {k: s[k]["median"] for k in s if isinstance(s[k], dict) and "median" in s[k]}
    out["elapsed_s"] = time.time() - t0
    np.savez_compressed(os.path.join("/tmp/claude-1000/-home-issan-Projects-claude/2fa60fc8-0b28-4ab8-880d-3470ff4b67ed/scratchpad", f"v2_{cond}.npz"), NA=NA, CA=CA, NI=NI, CI=CI, NF=NF, CF=CF)
    json.dump(out, open(os.path.join(RES, f"v2_iso_{cond}.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main(sys.argv[1])

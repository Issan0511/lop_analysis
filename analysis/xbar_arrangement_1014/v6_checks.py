"""V6 (verifier): (i) per-state sd of the pairwise pattern correlation (alive units) vs an iso null with the same H and sperp_k
(T = 11, 21, ..., 91, 100; 10 null draws per state), plus the pooled sd as agent A computes it;
(ii) phase-2 states with an input of occupancy f >= 1/2: number of alive units with m_k > 0 in that state, and whether the
f >= 1/2 inputs are open in those m>0 units; model value F(n) = (1/H) sum_k P(open) from 20 iso draws (actual m_k, sperp_k)."""
import json, math, os, sys
import numpy as np
from v_common import load, weights, RES, PH2
FILES = {"LR0.03_s0": "review/opus_lrslope/out/LRa0.03_s0.npz", "LR0.1_s0": "review/opus_bwdfloor/out/LR_ref_s0.npz",
         "LR0.3_s0": "review/opus_lrslope/out/LRa0.3_s0.npz", "LR0.5_s0": "review/opus_lrslope/out/LRa0.5_s0.npz",
         "LR0.5_s1": "review/opus_lrslope/out/LRa0.5_s1.npz", "ELU_s0": "review/opus_bwdfloor/out/ELU_ref_s0.npz",
         "ELU_s1": "review/opus_bwdfloor/out/ELU_ref_s1.npz"}
TS = [11, 21, 31, 41, 51, 61, 71, 81, 91, 100]

def part_i(tag, rng):
    d, Xt, xbar, K, Kbar = load(FILES[tag]); D = Xt.shape[1]; rows = []; pool_a, pool_n = [], []
    for T in TS:
        W = weights(d, T); Z = Xt @ W.T; m = W @ xbar; U = Z - np.outer(K / Kbar, m); al = np.flatnonzero((Z > 0).sum(0) > 0); H = len(al)
        Ua = U[:, al]; sp = Ua.std(0); iu = np.triu_indices(H, 1); c = np.corrcoef(Ua.T)[iu]; pool_a.append(c)
        nsd = []; nmean = []
        for r in range(10):
            G = rng.standard_normal((D, H)); G -= np.outer(xbar, xbar @ G) / Kbar; Ui = Xt @ G
            cn = np.corrcoef(Ui.T)[iu]; nsd.append(cn.std()); nmean.append(cn.mean())
            if r == 0: pool_n.append(cn)
        rows.append(dict(T=T, H=H, sd=float(c.std()), mean=float(c.mean()), null_sd_mean=float(np.mean(nsd)), null_sd_min=float(np.min(nsd)), null_sd_max=float(np.max(nsd)),
                         z=float((c.std() - np.mean(nsd)) / np.std(nsd, ddof=1))))
    pa = np.concatenate(pool_a); pn = np.concatenate(pool_n)
    return dict(per_state=rows, pooled_sd=float(pa.std()), pooled_null_sd=float(pn.std()), n_states_below_null_min=sum(r["sd"] < r["null_sd_min"] for r in rows),
                n_states_above_null_max=sum(r["sd"] > r["null_sd_max"] for r in rows), median_within_state_sd=float(np.median([r["sd"] for r in rows])),
                median_null_sd=float(np.median([r["null_sd_mean"] for r in rows])))

def part_ii(tag, rng):
    d, Xt, xbar, K, Kbar = load(FILES[tag]); D = Xt.shape[1]; rows = []
    for T in PH2:
        W = weights(d, T); Z = Xt @ W.T; O = Z > 0; al = np.flatnonzero(O.sum(0) > 0); f = O[:, al].mean(1); hub = np.flatnonzero(f >= 0.5)
        if hub.size == 0: continue
        m = W @ xbar; U = Z - np.outer(K / Kbar, m); sp = U[:, al].std(0); pos = al[m[al] > 0]
        F = np.zeros(len(K))
        for r in range(20):
            G = rng.standard_normal((D, len(al))); G -= np.outer(xbar, xbar @ G) / Kbar; Ui = Xt @ G; Ui *= (sp / Ui.std(0))[None, :]
            F += ((np.outer(K / Kbar, m[al]) + Ui) > 0).mean(1) / 20
        rows.append(dict(T=T, H=int(len(al)), n_hub_inputs=int(hub.size), f_max=float(f.max()), n_m_pos=int(len(pos)),
                         m_pos_values=[float(m[k]) for k in pos], hub_open_in_mpos=[int(O[n, pos].sum()) for n in hub],
                         hub_n_open=[int(O[n, al].sum()) for n in hub], hub_K=[float(K[n]) for n in hub], model_F_at_hub=[float(F[n]) for n in hub], model_F_max=float(F.max())))
    allm = []
    for T in PH2:
        W = weights(d, T); Z = Xt @ W.T; al = np.flatnonzero((Z > 0).sum(0) > 0); m = W @ xbar; allm.append(int((m[al] > 0).sum()))
    return dict(states_with_hub=len(rows), states_with_hub_and_no_mpos=sum(r["n_m_pos"] == 0 for r in rows),
                states_with_mpos_any=int(sum(x > 0 for x in allm)), n_mpos_distribution_all_phase2={str(k): allm.count(k) for k in sorted(set(allm))}, rows=rows)

if __name__ == "__main__":
    rng = np.random.default_rng(14600); out = dict(part_i={}, part_ii={})
    for tag in ("LR0.03_s0", "LR0.1_s0", "LR0.3_s0", "LR0.5_s0", "ELU_s0"):
        out["part_i"][tag] = part_i(tag, rng); r = out["part_i"][tag]
        print(tag, "pooled sd %.3f null %.3f | within-state median %.3f null %.3f | below null min %d/10 above max %d/10" % (r["pooled_sd"], r["pooled_null_sd"], r["median_within_state_sd"], r["median_null_sd"], r["n_states_below_null_min"], r["n_states_above_null_max"]), flush=True)
    for tag in ("LR0.5_s0", "LR0.5_s1", "ELU_s1"):
        out["part_ii"][tag] = part_ii(tag, rng); r = out["part_ii"][tag]
        print(tag, {k: v for k, v in r.items() if k != "rows"}, flush=True)
        for row in r["rows"]: print("   ", row, flush=True)
    json.dump(out, open(os.path.join(RES, "v6_checks.json"), "w"), indent=1)

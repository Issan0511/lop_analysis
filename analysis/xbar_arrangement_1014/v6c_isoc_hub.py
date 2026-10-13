"""V6c (verifier): at the a0.5 / ELU s1 phase-2 states with an input f >= 1/2, the model occupancy at that input under
iso (actual m_k, sperp_k) and under iso+common (actual ubar_n added, residual sd sqrt(max(sperp^2 - sd(ubar)^2, 0.25 sperp^2)), as REPORT_B §8);
20 draws per state; also ubar_n / (|mbar| K_n/Kbar) at the hub input (A2's rho)."""
import json, os
import numpy as np
from v_common import load, weights, RES, PH2
FILES = {"LR0.5_s0": "review/opus_lrslope/out/LRa0.5_s0.npz", "LR0.5_s1": "review/opus_lrslope/out/LRa0.5_s1.npz", "ELU_s1": "review/opus_bwdfloor/out/ELU_ref_s1.npz"}

def main():
    rng = np.random.default_rng(14700); out = {}
    for tag, npz in FILES.items():
        d, Xt, xbar, K, Kbar = load(npz); D = Xt.shape[1]; rows = []
        for T in PH2:
            W = weights(d, T); Z = Xt @ W.T; O = Z > 0; al = np.flatnonzero(O.sum(0) > 0); f = O[:, al].mean(1); hub = np.flatnonzero(f >= 0.5)
            if hub.size == 0: continue
            m = (W @ xbar)[al]; U = (Z - np.outer(K / Kbar, W @ xbar))[:, al]; sp = U.std(0); ub = U.mean(1); sres = np.sqrt(np.maximum(sp ** 2 - ub.std() ** 2, 0.25 * sp ** 2))
            Fi = np.zeros(len(K)); Fc = np.zeros(len(K))
            for r in range(20):
                G = rng.standard_normal((D, len(al))); G -= np.outer(xbar, xbar @ G) / Kbar; Ui = Xt @ G
                Fi += ((np.outer(K / Kbar, m) + Ui * (sp / Ui.std(0))[None, :]) > 0).mean(1) / 20
                Fc += ((np.outer(K / Kbar, m) + ub[:, None] + Ui * (sres / Ui.std(0))[None, :]) > 0).mean(1) / 20
            rho = ub / (abs(m.mean()) * K / Kbar)
            rows.append(dict(T=T, f_hub=[float(f[n]) for n in hub], F_iso_hub=[float(Fi[n]) for n in hub], F_isoc_hub=[float(Fc[n]) for n in hub], rho_hub=[float(rho[n]) for n in hub],
                             F_isoc_max=float(Fc.max()), n_isoc_ge_half=int((Fc >= 0.5).sum())))
        fh = np.concatenate([r["f_hub"] for r in rows]); fi = np.concatenate([r["F_iso_hub"] for r in rows]); fc = np.concatenate([r["F_isoc_hub"] for r in rows]); rh = np.concatenate([r["rho_hub"] for r in rows])
        out[tag] = dict(n_states=len(rows), n_hub_inputs=int(fh.size), f_hub_range=[float(fh.min()), float(fh.max())], F_iso_hub_range=[float(fi.min()), float(fi.max())],
                        F_isoc_hub_range=[float(fc.min()), float(fc.max())], n_hub_with_F_isoc_ge_half=int((fc >= 0.5).sum()), rho_hub_range=[float(rh.min()), float(rh.max())], rows=rows)
        print(tag, {k: v for k, v in out[tag].items() if k != "rows"}, flush=True)
    json.dump(out, open(os.path.join(RES, "v6c_isoc_hub.json"), "w"), indent=1)

if __name__ == "__main__":
    main()

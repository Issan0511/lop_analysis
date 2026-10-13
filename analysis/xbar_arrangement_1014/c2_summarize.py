"""C2 (xbar_arrangement_1014): summarize c2_iso_expect_<tag>_s<seed>.json -> results/.../c2_tables.md, c2_summary.json.
Cross-check: actual and E_full equal the c_strat_expect (P149-reproducing) values. Conditional rows (if per-draw values present): draws with p_iso > 0, and draws with p_iso in [p_act/2, 2 p_act]."""
import os, json
import numpy as np
RES = "/home/issan/Projects/claude/wt/xbar_arrangement_1014/results/xbar_arrangement_1014"
q = lambda x, p: float(np.percentile(x, p))
ST = [s for s in ("a01_s0", "a03_s0", "a05_s0", "a01_s1", "a03_s1", "a05_s1") if os.path.exists(os.path.join(RES, f"c2_iso_expect_{s}.json"))]
L = []; SUM = {}
L.append("| state | n | max rel diff of actual/E_full vs c_strat_expect | p_act median | p_iso/p_act median [10,90] | p_iso median | draws with p_iso = 0 (mean frac) | (d-a) mass iso/act median [10,90] | e' mass iso/act median [10,90] |")
L.append("|---|---|---|---|---|---|---|---|---|")
for st in ST:
    J = json.load(open(os.path.join(RES, f"c2_iso_expect_{st}.json"))); R = J["rows"]; C = {r["unit"]: r for r in json.load(open(os.path.join(RES, f"c_strat_expect_{st}.json")))["rows"]}
    f_ = lambda k: np.array([r[k] for r in R], dtype=float); n = len(R)
    rel = max(max(abs(r[a_] - C[r["unit"]][b_]) / abs(C[r["unit"]][b_]) for r in R) for a_, b_ in (("A_f", "Ag"), ("A_b", "Aw"), ("A_r", "Ar"), ("Efull_f", "Ef"), ("Efull_b", "Eb"), ("Efull_r", "Er")))
    pr = f_("p_iso_mean") / f_("p_act"); mr = f_("mass_iso_mean") / f_("mass_act"); er = f_("emass_iso_mean") / f_("emass_act")
    L.append(f"| {st} | {n} | {rel:.1e} | {np.median(f_('p_act')):.4f} | {np.median(pr):.2f} [{q(pr, 10):.2f}, {q(pr, 90):.2f}] | {np.median(f_('p_iso_mean')):.4f} | {f_('p_iso_zero').mean():.3f} | {np.median(mr):.2f} [{q(mr, 10):.2f}, {q(mr, 90):.2f}] | {np.median(er):.2f} [{q(er, 10):.2f}, {q(er, 90):.2f}] |")
    SUM[st] = dict(n=n, rel_check=rel, p_ratio=[float(np.median(pr)), q(pr, 10), q(pr, 90)], p_zero=float(f_("p_iso_zero").mean()), mass_ratio=float(np.median(mr)), emass_ratio=float(np.median(er)), M={})
    for k in ("f", "b", "r"):
        A = f_(f"A_{k}"); E0 = f_(f"Efull_{k}"); Ei = f_(f"Eiso_{k}"); si = f_(f"sdiso_{k}"); z = (A - Ei) / si; den = E0 - A; ok = den > 0; gap = (E0 - Ei)[ok] / den[ok]; ra = A / Ei
        band = (A >= f_(f"q025_{k}")) & (A <= f_(f"q975_{k}"))
        x = dict(ratio=[float(np.median(ra)), q(ra, 10), q(ra, 90)], z=[float(np.median(z)), q(z, 10), q(z, 90)], lt=int((A < Ei).sum()), gap=[float(np.median(gap)), q(gap, 10), q(gap, 90)], den_le0=int((~ok).sum()),
                 Epos=int((Ei > 0).sum()), z2=int((np.abs(z) <= 2).sum()), band95=int(band.sum()), rank_med=float(np.median(f_(f"rank_{k}"))))
        if "draws" in R[0]:
            for cname, sel in (("p>0", lambda r: np.array(r["draws"]["p"]) > 0), ("p in [p/2, 2p]", lambda r: (np.array(r["draws"]["p"]) >= r["p_act"] / 2) & (np.array(r["draws"]["p"]) <= 2 * r["p_act"]))):
                Ec = []; sc = []; nd = []
                for r in R:
                    s = sel(r); v = np.array(r["draws"][k])[s]; nd.append(int(s.sum())); Ec.append(v.mean() if s.sum() >= 2 else np.nan); sc.append(v.std() if s.sum() >= 2 else np.nan)
                Ec = np.array(Ec); sc = np.array(sc); okc = ok & np.isfinite(Ec); zc = ((A - Ec) / sc)[np.isfinite(Ec)]; gc = (E0 - Ec)[okc] / den[okc]
                x[cname] = dict(gap=float(np.median(gc)), z=float(np.median(zc)), lt=int((A < Ec)[np.isfinite(Ec)].sum()), nunits=int(np.isfinite(Ec).sum()), ndraw_med=float(np.median(nd)), z2=int((np.abs(zc) <= 2).sum()))
        SUM[st]["M"][k] = x
L.append("")
for k, nm in (("b", "probe M"), ("f", "full M"), ("r", "readout (u = K d_iso, f_iso = F(z_iso), Ebar)")):
    L.append(f"**{nm}**\n")
    L.append("| state | actual/E_iso med [10,90] | z_iso med [10,90] | actual < E_iso | gap closed (E_full-E_iso)/(E_full-actual) med [10,90] | E_iso > 0 | abs(z_iso) <= 2 | actual in iso 2.5-97.5% band | rank of actual in iso draws (med) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for st in ST:
        x = SUM[st]["M"][k]; n = SUM[st]["n"]
        L.append(f"| {st} | {x['ratio'][0]:.2f} [{x['ratio'][1]:.2f}, {x['ratio'][2]:.2f}] | {x['z'][0]:+.1f} [{x['z'][1]:+.1f}, {x['z'][2]:+.1f}] | {x['lt']}/{n} ({x['lt'] / n:.0%}) | {x['gap'][0]:+.2f} [{x['gap'][1]:+.2f}, {x['gap'][2]:+.2f}] (den<=0: {x['den_le0']}) | {x['Epos']}/{n} | {x['z2']}/{n} ({x['z2'] / n:.0%}) | {x['band95']}/{n} ({x['band95'] / n:.0%}) | {x['rank_med']:.2f} |")
    L.append("")
if any("p>0" in SUM[s]["M"]["b"] for s in ST):
    L.append("**Conditional iso expectation (sensitivity): restrict the 150 draws to p_iso > 0, or to p_iso in [p_act/2, 2 p_act]**\n")
    L.append("| state | M | gap closed (p>0) | z_iso med (p>0) | actual < E (p>0) | gap closed (p-matched) | z_iso med (p-matched) | actual < E (p-matched) | abs(z) <= 2 (p-matched) | draws used per unit (p-matched, med) | units |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for st in ST:
        for k in ("b", "f", "r"):
            x = SUM[st]["M"][k]
            if "p>0" not in x: continue
            a_, b_ = x["p>0"], x["p in [p/2, 2p]"]
            L.append(f"| {st} | {dict(b='probe', f='full', r='readout')[k]} | {a_['gap']:+.2f} | {a_['z']:+.1f} | {a_['lt']}/{a_['nunits']} | {b_['gap']:+.2f} | {b_['z']:+.1f} | {b_['lt']}/{b_['nunits']} | {b_['z2']}/{b_['nunits']} | {b_['ndraw_med']:.0f} | {b_['nunits']} |")
open(os.path.join(RES, "c2_tables.md"), "w").write("\n".join(L) + "\n"); json.dump(SUM, open(os.path.join(RES, "c2_summary.json"), "w"), indent=1); print("\n".join(L))

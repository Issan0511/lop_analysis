"""Within-task ledger of the position push from the bit-exact replay (replay_adam.py output), phase 2 (tasks 11..100), three arms.

Per task T (start state = end of task T-1) and hidden unit i (sinking-positive sign convention everywhere):
  D      = m_i(0) - m_i(4000)                      actual Adam displacement of the position over the task
  push_u = m_i(0) - min_s m_i(s), ret_u = min_s m_i(s) - m_i(4000)   with the per-unit bottom step s*_i (over the 851 check steps)
  G      = sum_s F_i^{Y}(theta_s; B_s)              path integral along the Adam trajectory of the batch-mean CE position gradient
           (eta*G = the displacement plain SGD would have produced along this same path; F^Y = F^U - F^G exactly)
  GU, GG = the confidence part sum_s F^U and the label part sum_s F^{G,Y} (G = GU - GG); at s = 0, E_Y F^G = 0 and F^U = S_i^B
  G0     = sum_s F^{Y} with the gate d_in and readout v_i frozen at the task start (actual residuals); Gad = G - G0 (adaptation part)
  g1     = first-batch position gradient at the task start (SGD first step = eta*g1), dm1 = -(first Adam step of the position)
Cumulative sums over the 90 phase-2 tasks per unit, sign statistics, and within-task profiles.
Output: results/cumulative_push_1013/ledger_replay_{tag}.json (+ per-unit arrays in ledger_replay_{tag}.npz)
"""
import os, sys, json, glob
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "1"
import numpy as np
ROOT = "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930"
REPLAY = "/home/issan/Projects/obsidian-research-data/cumulative_push_1013/replay"
sys.path.insert(0, ROOT); from common import load_X
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "..", "results", "cumulative_push_1013"); os.makedirs(OUT, exist_ok=True)
ARMS = {"a01": ("review/opus_bwdfloor/out/LR_ref_s0.npz", 0.1, "a0.1"), "a03": ("review/opus_lrslope/out/LRa0.3_s0.npz", 0.3, "a0.3"), "a05": ("review/opus_lrslope/out/LRa0.5_s0.npz", 0.5, "a0.5")}
ETA = 1e-3; T0, T1 = 11, 100; PROF = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 1000, 2000, 3000, 4000]
tags = sys.argv[1:] or list(ARMS)

def q(x, qs=(0.1, 0.5, 0.9)): return [float(np.quantile(x, v)) for v in qs]

for tag in tags:
    rel, a, rdir = ARMS[tag]; d = np.load(os.path.join(ROOT, rel)); X = load_X(d["subset"]); mu = X.mean(0)
    tal = np.load(os.path.join(OUT, f"tally_{tag}.npz")); alive_end = tal["alive"]            # [100 tasks, H] alive at task ends
    recs = []
    for T in range(T0, T1 + 1):
        f = os.path.join(REPLAY, rdir, f"task_{T:03d}.npz")
        if not os.path.exists(f): break
        z = np.load(f)
        if not bool(z["bit_ok"]): print(f"{tag}: task {T} bit_ok False -> stop"); break
        m = z["m_series"]; s = z["s_series"]; cs = z["check_steps"]
        live = alive_end[T - 2] & alive_end[T - 1]                                            # alive at both ends of task T
        imin = m.argmin(0); mmin = m.min(0)
        g1 = z["g1_0"].astype(np.float64) @ mu + z["g1_1"].astype(np.float64)
        dm1 = -(z["dtheta1_0"].astype(np.float64) @ mu + z["dtheta1_1"].astype(np.float64))
        pidx = [int(np.where(cs == p_)[0][0]) for p_ in PROF]
        recs.append(dict(T=T, live=live, s0=s[0], D=m[0] - m[-1], push=m[0] - mmin, ret=mmin - m[-1], bottom_step=cs[imin],
                         G=z["accF_series"][-1], GU=z["accFU_series"][-1], GG=z["accFG_series"][-1], G0=z["accF0_series"][-1],
                         g1=g1, dm1=dm1,
                         prof_drop=(m[0][None, :] - m[pidx]) / s[0][None, :],                 # [len(PROF), H] in width units
                         prof_F=z["accF_series"][pidx], prof_FU=z["accFU_series"][pidx], prof_FG=z["accFG_series"][pidx], prof_F0=z["accF0_series"][pidx]))
    if not recs: print(f"{tag}: no tasks"); continue
    nT = len(recs); H = recs[0]["D"].shape[0]
    L = np.stack([r["live"] for r in recs]); la = L.sum(0) >= 0.9 * nT
    def arr(k): return np.stack([r[k] for r in recs])
    D, push, ret, G, GU, GG, G0, g1, dm1, s0 = [arr(k) for k in ("D", "push", "ret", "G", "GU", "GG", "G0", "g1", "dm1", "s0")]
    Gad = G - G0; bstep = arr("bottom_step")
    def cum(x): return np.where(L, x, 0.0).sum(0)
    def frac(x): return float((x[L] > 0).mean())
    def agree(x, y): return float(((x > 0) == (y > 0))[L].mean())
    res = dict(tag=tag, a=a, n_tasks=nT, units_live_ge90pct=int(la.sum()), n_unit_tasks=int(L.sum()),
        # actual Adam displacement
        frac_D_sink=frac(D), units_cumD_sink=int((cum(D)[la] > 0).sum()), cumD_quantiles=q(cum(D)[la]),
        push_u_over_s0_quantiles=q((push / s0)[L]), ret_u_over_s0_quantiles=q((ret / s0)[L]), D_over_s0_quantiles=q((D / s0)[L]),
        cum_ret_over_cum_push_quantiles=q((cum(ret) / cum(push))[la]), bottom_step_quantiles=q(bstep[L].astype(float)),
        # SGD-equivalent gradient path integral along the same path
        frac_G_sink=frac(G), units_cumG_sink=int((cum(G)[la] > 0).sum()), sign_agree_D_G=agree(D, G),
        etaG_over_D_abs_median=float(np.median((ETA * np.abs(G) / (np.abs(D) + 1e-30))[L])),
        frac_GU_pos=frac(GU), frac_GG_pos=frac(GG), G_over_GU_abs_quantiles=q((np.abs(G) / (np.abs(GU) + 1e-30))[L]),
        frac_G0_sink=frac(G0), frac_Gad_sink=frac(Gad), units_cumG0_sink=int((cum(G0)[la] > 0).sum()), units_cumGad_sink=int((cum(Gad)[la] > 0).sum()),
        G0_over_G_quantiles=q((G0 / (G + 1e-30))[L & (np.abs(G) > 1e-6)]), sign_agree_G0_G=agree(G0, G), sign_agree_G0_D=agree(G0, D),
        # first step
        frac_g1_sink=frac(g1), frac_dm1_sink=frac(dm1), sign_agree_g1_dm1=agree(g1, dm1), units_cum_g1_sink=int((cum(g1)[la] > 0).sum()), units_cum_dm1_sink=int((cum(dm1)[la] > 0).sum()),
        dm1_over_push_quantiles=q((dm1 / (push + 1e-30))[L & (push > 1e-6)]),
        # profiles (median over live unit-tasks)
        PROF=PROF,
        prof_drop_median=[float(np.median(arr("prof_drop")[:, j, :][L])) for j in range(len(PROF))],
        prof_drop_q10=[float(np.quantile(arr("prof_drop")[:, j, :][L], 0.1)) for j in range(len(PROF))],
        prof_drop_q90=[float(np.quantile(arr("prof_drop")[:, j, :][L], 0.9)) for j in range(len(PROF))],
        prof_F_over_absGmax_median=[float(np.median((arr("prof_F")[:, j, :] / (np.abs(arr("prof_F")).max(1) + 1e-30))[L])) for j in range(len(PROF))],
        prof_FU_over_end_median=[float(np.median((arr("prof_FU")[:, j, :] / (GU + 1e-30))[L])) for j in range(len(PROF))],
        prof_FG_over_end_median=[float(np.median((arr("prof_FG")[:, j, :] / (GG + 1e-30))[L])) for j in range(len(PROF))],
        prof_F0_over_end_median=[float(np.median((arr("prof_F0")[:, j, :] / (G0 + 1e-30))[L])) for j in range(len(PROF))],
        prof_F_sink_frac=[frac(arr("prof_F")[:, j, :]) for j in range(len(PROF))],
    )
    print(f"=== {tag} (a={a}) tasks {recs[0]['T']}..{recs[-1]['T']} ({nT}) units live>=90% {la.sum()}/{H} ===")
    for k_, v in res.items():
        if k_ in ("tag", "a"): continue
        print(f"  {k_}: {v}")
    json.dump(res, open(os.path.join(OUT, f"ledger_replay_{tag}.json"), "w"), indent=1)
    np.savez(os.path.join(OUT, f"ledger_replay_{tag}.npz"), T=np.array([r["T"] for r in recs]), live=L, D=D, push=push, ret=ret, G=G, GU=GU, GG=GG, G0=G0, g1=g1, dm1=dm1, s0=s0, bottom_step=bstep)

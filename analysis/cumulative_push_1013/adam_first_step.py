"""Exact-distribution expectation of Adam's first step at the switch, for the position of each hidden unit, at the real
task-end states, using the replayed Adam state (analysis/cumulative_push_1013/replay_adam.py output).

At the start of task T+1 (state theta = end of task T, Adam moments m0, v0, step count k), the first batch B (16 inputs,
indices from the replay) gets fresh labels Y_n iid uniform on 10 classes. For unit i and input coordinate j (785 = 784 pixels + bias),
the batch-mean gradient is
    gamma_j = mu_j - xi_j,   mu_j = (1/16) sum_{n in B} x~_nj d_in <v_i, p_n - u>,   xi_j = (1/16) sum_{n in B} x~_nj d_in eps_n,
    eps_n = v_{i,Y_n} - vbar_i   (iid over n, 10 atoms, mean 0; v_i = readout column of unit i).
PyTorch Adam (no weight decay): m1 = b1 m0 + (1-b1) gamma, v1 = b2 v0 + (1-b2) gamma^2,
    dw_j = -(lr/(1-b1^k)) m1 / (sqrt(v1)/sqrt(1-b2^k) + eps).
Position change of the first step: dm_i = sum_j xbar~_j dw_j (xbar~ = mean input with bias coordinate 1).
We compute E_Y[dm_i] (and its label-noise SD) by Monte Carlo over the 16 labels with R draws (common random numbers across
units), together with the SGD expectation -lr * S_i^B where S_i^B = sum_j xbar~_j mu_j = (1/16) sum_{n in B} K_n d_in <v_i, p_n - u>,
the realized first step from the replay (dtheta1 projected on xbar~), and a decomposition of the Adam expectation into the
confidence-only step (labels replaced by their mean: gamma = mu) and the rest.
Phase-2 states: task-end indices t = 9..98 (start of tasks 11..100). Output: results/cumulative_push_1013/adam_first_step_{tag}.json
"""
import os, sys, json, glob, time
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "1"
import numpy as np
ROOT = "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930"
REPLAY = "/home/issan/Projects/obsidian-research-data/cumulative_push_1013/replay"
sys.path.insert(0, ROOT); from common import load_X
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "..", "results", "cumulative_push_1013"); os.makedirs(OUT, exist_ok=True)
ARMS = {"a01": ("review/opus_bwdfloor/out/LR_ref_s0.npz", 0.1, "a0.1"), "a03": ("review/opus_lrslope/out/LRa0.3_s0.npz", 0.3, "a0.3"), "a05": ("review/opus_lrslope/out/LRa0.5_s0.npz", 0.5, "a0.5")}
K, LR, B1, B2, EPS = 10, 1e-3, 0.9, 0.999, 1e-8
R = int(os.environ.get("ADAM_MC_DRAWS", "4000")); T0, T1 = 11, 100
tags = sys.argv[1:] or list(ARMS)

def softmax(l):
    l = l - l.max(1, keepdims=True); e = np.exp(l); return e / e.sum(1, keepdims=True)

def adam_step(gamma, m0, v0, k):
    m1 = B1 * m0 + (1 - B1) * gamma; v1 = B2 * v0 + (1 - B2) * gamma * gamma
    return -(LR / (1 - B1 ** k)) * m1 / (np.sqrt(v1) / np.sqrt(1 - B2 ** k) + EPS)

rng = np.random.default_rng(20261013)
for tag in tags:
    rel, a, rdir = ARMS[tag]; d = np.load(os.path.join(ROOT, rel)); X = load_X(d["subset"]); N = X.shape[0]
    Xt = np.concatenate([X, np.ones((N, 1))], 1); xbar = Xt.mean(0); Kn = Xt @ xbar
    labels_mc = rng.integers(0, K, size=(R, 16))                      # common random labels for all states/units
    rows = []; t_start = time.time()
    for t in range(T0 - 2, T1 - 1):                                   # end-state index t = start of task t+2
        f = os.path.join(REPLAY, rdir, f"task_{t + 2:03d}.npz")
        if not os.path.exists(f): print(f"{tag}: missing {f}, stopping at t={t}"); break
        z = np.load(f)
        if not bool(z["bit_ok"]): print(f"{tag}: task {t + 2} bit_ok False"); break
        W1, b1, W2, b3 = [d[f"end{j}"][t].astype(np.float64) for j in range(4)]
        W = np.concatenate([W1, b1[:, None]], 1); H = W.shape[0]
        Z = Xt @ W.T; opn = Z > 0; D = np.where(opn, 1.0, a); phi = D * Z; prob = softmax(phi @ W2.T + b3)
        V = W2.T; Vc = V - V.mean(1, keepdims=True)                     # [H,K]
        idx = z["first_batch_idx"].astype(int)                         # 16 inputs of the first batch
        # Adam state at the end of the previous task (= task t+1) is stored in that task's file
        zp = np.load(os.path.join(REPLAY, rdir, f"task_{t + 1:03d}.npz"))
        m0 = np.concatenate([zp["adam_m_0"].astype(np.float64), zp["adam_m_1"].astype(np.float64)[:, None]], 1)   # [H,785]
        v0 = np.concatenate([zp["adam_v_0"].astype(np.float64), zp["adam_v_1"].astype(np.float64)[:, None]], 1)
        k = int(zp["adam_step"]) + 1
        # realized first step from the replay (float32 dtheta1), projected on xbar~
        dth = np.concatenate([z["dtheta1_0"].astype(np.float64), z["dtheta1_1"].astype(np.float64)[:, None]], 1)
        dm_real = dth @ xbar
        # per-unit quantities
        xb = Xt[idx]                                                   # [16,785]
        c_conf = (prob[idx] - 1.0 / K) @ Vc.T                          # [16,H]  <v_i^c, p_n - u>
        Db = D[idx]                                                    # [16,H]
        coef = Db * Kn[idx][:, None] / 16.0                            # [16,H]  K_n d_in /16
        S_B = (coef * c_conf).sum(0)                                   # SGD expectation of the position gradient on this batch
        # label noise: eps_n^(r) = Vc[i, Y_n^(r)] for draw r -> [R,16,H]; gamma_j = mu_j - (1/16) sum_n x~_nj d_in eps_n
        E_dm = np.zeros(H); SD_dm = np.zeros(H); E_dm_conf = np.zeros(H); P_sink = np.zeros(H); E_dm_sgd_mc = np.zeros(H)
        for i in range(H):
            mu = (xb * (Db[:, i] * c_conf[:, i])[:, None]).sum(0) / 16.0            # [785]
            eps = Vc[i][labels_mc]                                                  # [R,16]
            xi = (eps * Db[:, i][None, :]) @ xb / 16.0                              # [R,785]
            gamma = mu[None, :] - xi
            dw = adam_step(gamma, m0[i][None, :], v0[i][None, :], k)                # [R,785]
            dm = dw @ xbar
            E_dm[i] = dm.mean(); SD_dm[i] = dm.std(); P_sink[i] = (dm < 0).mean()
            E_dm_conf[i] = adam_step(mu[None, :], m0[i][None, :], v0[i][None, :], k)[0] @ xbar
            E_dm_sgd_mc[i] = (-LR * (gamma @ xbar)).mean()                          # = -LR*S_B up to MC error (check)
        dm_mom = adam_step(np.zeros((H, 785)), m0, v0, k) @ xbar                    # carried-momentum-only step (gamma = 0)
        rows.append(dict(task=t + 2, alive=opn.any(0).tolist(), k=k, E_dm_adam=E_dm.tolist(), SD_dm_adam=SD_dm.tolist(), P_sink_adam=P_sink.tolist(),
                         E_dm_adam_confonly=E_dm_conf.tolist(), dm_adam_momentum_only=dm_mom.tolist(), E_dm_sgd=(-LR * S_B).tolist(),
                         E_dm_sgd_mc=E_dm_sgd_mc.tolist(), dm_realized=dm_real.tolist(), S_B=S_B.tolist(),
                         sqrt_v0_over_absmu_median=float(np.median(np.sqrt(v0) / (np.abs((xb[:, None, :] * (Db * c_conf)[:, :, None]).sum(0) / 16.0) + 1e-30)))))
        al = opn.any(0)
        print(f"{tag} task {t + 2:3d} k {k}: sign agree adam/sgd {((E_dm < 0) == (S_B > 0))[al].mean():.3f}  frac E_dm_adam<0 (sink) {(E_dm[al] < 0).mean():.3f}  frac SGD sink {(S_B[al] > 0).mean():.3f}  median |E_adam|/|E_sgd| {np.median(np.abs(E_dm[al]) / (LR * np.abs(S_B[al]) + 1e-30)):.2f}  realized sink {(dm_real[al] < 0).mean():.3f} ({time.time() - t_start:.0f}s)", flush=True)
    json.dump(dict(tag=tag, a=a, R=R, rows=rows), open(os.path.join(OUT, f"adam_first_step_{tag}.json"), "w"))
    # cumulative summary over phase 2 for units alive in >= 80 tasks
    if rows:
        al = np.array([r["alive"] for r in rows]); la = al.sum(0) >= 80
        Ead = np.array([r["E_dm_adam"] for r in rows]); Esg = np.array([r["E_dm_sgd"] for r in rows]); Dre = np.array([r["dm_realized"] for r in rows]); Ecf = np.array([r["E_dm_adam_confonly"] for r in rows])
        def cum(x): return np.where(al, x, 0).sum(0)
        summ = dict(tag=tag, n_tasks=len(rows), units_ge80=int(la.sum()),
                    frac_unit_tasks_adam_expect_sink=float((Ead[al] < 0).mean()), frac_unit_tasks_sgd_expect_sink=float((Esg[al] < 0).mean()),
                    sign_agree_adam_sgd=float(((Ead < 0) == (Esg < 0))[al].mean()), frac_realized_sink=float((Dre[al] < 0).mean()),
                    units_cum_adam_expect_sink=int((cum(Ead)[la] < 0).sum()), units_cum_sgd_expect_sink=int((cum(Esg)[la] < 0).sum()), units_cum_realized_sink=int((cum(Dre)[la] < 0).sum()),
                    cum_adam_over_cum_sgd_median=float(np.median((cum(Ead) / cum(Esg))[la])), cum_confonly_over_cum_adam_median=float(np.median((cum(Ecf) / cum(Ead))[la])),
                    median_abs_ratio_adam_sgd=float(np.median((np.abs(Ead) / (np.abs(Esg) + 1e-30))[al])))
        print(json.dumps(summ, indent=1)); json.dump(summ, open(os.path.join(OUT, f"adam_first_step_{tag}_summary.json"), "w"), indent=1)

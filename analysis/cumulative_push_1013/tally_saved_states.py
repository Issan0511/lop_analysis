"""Cumulative accounting of the position push along the three saved 100-task trajectories (leaky a = 0.1, 0.3, 0.5, seed 0).

Inputs (read only): the task-end and task-bottom weights saved by review/opus_bwdfloor/run_bwd.py (a = 0.1) and
review/opus_lrslope/run_slope.py (a = 0.3, 0.5), and the per-checkpoint capacity push / self term of P170
(stage14_1007_h4/p170_{a01,a03,a05}.json, 1-output both-layer kernel, P128 conventions: tau = 0.1 sd(z), lambda = 1e-4 lambda_max).

Per unit i and task-end state theta (index t = 0..99, task number T = t + 1):
  m_i = mean_n z_in, s_i = std_n z_in (population), r_i = m_i / s_i, p_i = open fraction, alive = any open
  S_i = sum_n K_n d_in <v_i^c, p_n - u>  (switching push at this state = exact expectation over fresh labels of the
        first full-batch SGD position gradient; positive = sinking side), split into open side (gate 1) and closed side (gate a)
Per task T+1 (start = end[t], bottom = bot[t+1] (layer-mean bottom step), end = end[t+1]):
  push = m(start) - m(bottom), ret = m(bottom) - m(end), net = m(start) - m(end)   (positive = sinking)
  net = -(r_t ds + s_{t+1} dr) with ds = s_{t+1} - s_t, dr = r_{t+1} - r_t  (exact)
Phase 2 window: task numbers 11..100 (start indices t = 9..98), 90 tasks. Cumulative sums run over unit-tasks with the unit alive
(at least one open input) at both ends of the task; per-unit cumulative statistics are reported for units alive in >= 80 of the 90 tasks.
Output: results/cumulative_push_1013/tally_saved_states.json and a printed summary.
"""
import os, sys, json
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "1"
import numpy as np
ROOT = "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930"
sys.path.insert(0, ROOT)
from common import load_X
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "results", "cumulative_push_1013"); os.makedirs(OUT, exist_ok=True)
ARMS = {"a01": ("review/opus_bwdfloor/out/LR_ref_s0.npz", 0.1), "a03": ("review/opus_lrslope/out/LRa0.3_s0.npz", 0.3), "a05": ("review/opus_lrslope/out/LRa0.5_s0.npz", 0.5)}
P170 = os.path.join(ROOT, "review/stage4_1002/block_1003/stage14_1007_h4/p170_{tag}.json")
K = 10; T0, T1 = 11, 100            # phase-2 task numbers (inclusive)

def softmax(l):
    l = l - l.max(1, keepdims=True); e = np.exp(l); return e / e.sum(1, keepdims=True)

def state_quantities(X, Kn, a, W1, b1, W2, b3, y=None):
    z = X @ W1.T + b1; opn = z > 0; d = np.where(opn, 1.0, a); phi = d * z
    prob = softmax(phi @ W2.T + b3); Vc = W2.T - W2.T.mean(1, keepdims=True)
    c = (prob - 1.0 / K) @ Vc.T                               # [N,H] <v_i^c, p_n - u>
    S_open = (Kn[:, None] * opn * c).sum(0); S_closed = a * (Kn[:, None] * (~opn) * c).sum(0)
    out = dict(m=z.mean(0), s=z.std(0), p=opn.mean(0), alive=opn.any(0), S=S_open + S_closed, S_open=S_open, S_closed=S_closed,
                pmax=prob.max(1).mean())
    if y is not None:                                          # fit residual of this task in the position direction: F^Y = sum_n K_n d_in <v_i, p_n - e_{y_n}>
        cy = (prob - np.eye(K)[y]) @ Vc.T; out["FY"] = (Kn[:, None] * d * cy).sum(0); out["acc"] = float((prob.argmax(1) == y).mean())
    return out

def q(x, qs=(0.1, 0.5, 0.9)): return [float(np.quantile(x, v)) for v in qs]

summary = {}
for tag, (rel, a) in ARMS.items():
    d = np.load(os.path.join(ROOT, rel)); X = load_X(d["subset"]); mu = X.mean(0); Kn = X @ mu + 1.0
    nT = d["end0"].shape[0]
    E = [state_quantities(X, Kn, a, *[d[f"end{j}"][t].astype(np.float64) for j in range(4)], y=d["y"][t].astype(int)) for t in range(nT)]
    FY_end = np.stack([e["FY"] for e in E]); acc_end = np.array([e["acc"] for e in E])
    m_end = np.stack([e["m"] for e in E]); s_end = np.stack([e["s"] for e in E]); r_end = m_end / s_end
    p_end = np.stack([e["p"] for e in E]); alive = np.stack([e["alive"] for e in E]); S_end = np.stack([e["S"] for e in E])
    So_end = np.stack([e["S_open"] for e in E]); Sc_end = np.stack([e["S_closed"] for e in E]); pmax = np.array([e["pmax"] for e in E])
    m_bot = np.stack([(X @ d["bot0"][t].astype(np.float64).T + d["bot1"][t].astype(np.float64)).mean(0) for t in range(nT)])
    # P170 capacity push A and self term Aal at the same end states: row task k <-> end index k-2
    rows = {r["task"]: r for r in json.load(open(P170.format(tag=tag)))["rows"]}
    A_end = np.full_like(m_end, np.nan); Aal_end = np.full_like(m_end, np.nan)
    for k, r in rows.items():
        if 0 <= k - 2 < nT: A_end[k - 2] = r["A"]; Aal_end[k - 2] = r["Aal"]
    # per-task transitions T+1 = t+2 for start index t
    st = np.arange(T0 - 2, T1 - 1)                             # start indices 9..98
    en = st + 1
    push = m_end[st] - m_bot[en]; ret = m_bot[en] - m_end[en]; net = m_end[st] - m_end[en]
    ds = s_end[en] - s_end[st]; dr = r_end[en] - r_end[st]
    rds = r_end[st] * ds; sdr = s_end[en] * dr                  # dm = r_t ds + s_{t+1} dr exactly
    assert np.allclose(-(rds + sdr), net, atol=1e-9)
    live = alive[st] & alive[en]                               # units alive at both ends of the task
    n_live = live.sum(0)                                        # tasks (of 90) in which the unit is alive at both ends
    live_all = n_live >= 80                                     # units alive in at least 80 of the 90 phase-2 tasks
    H = m_end.shape[1]
    def cum(x, mask): return np.where(mask, x, 0.0).sum(0)
    cum_push, cum_ret, cum_net = cum(push, live), cum(ret, live), cum(net, live)
    cum_rds, cum_sdr = cum(rds, live), cum(sdr, live)
    # capacity push along phase-2 start states; actual switching push S at the same states
    A_ph, Aal_ph, S_ph = A_end[st], Aal_end[st], S_end[st]
    cum_A, cum_Aal, cum_S = cum(A_ph, alive[st]), cum(Aal_ph, alive[st]), cum(S_ph, alive[st])
    cum_So, cum_Sc = cum(So_end[st], alive[st]), cum(Sc_end[st], alive[st])
    la = live_all
    res = dict(a=a, n_tasks_phase2=int(len(st)), units_alive_ge80_tasks=int(la.sum()), units_alive_all_tasks=int((n_live == len(st)).sum()),
        alive_per_task_min_med_max=[int(alive[st].sum(1).min()), float(np.median(alive[st].sum(1))), int(alive[st].sum(1).max())], H=H,
        # per unit-task (alive at both ends)
        frac_sinking_unit_tasks=float((net[live] > 0).mean()), n_unit_tasks=int(live.sum()),
        net_over_s_quantiles=q(net[live] / s_end[st][live]), push_over_s_quantiles=q(push[live] / s_end[st][live]), ret_over_s_quantiles=q(ret[live] / s_end[st][live]),
        # cumulative over phase 2, units alive throughout
        cum_net_quantiles=q(cum_net[la]), cum_push_quantiles=q(cum_push[la]), cum_ret_quantiles=q(cum_ret[la]),
        units_cum_net_positive=int((cum_net[la] > 0).sum()), ret_over_push_cum_quantiles=q(cum_ret[la] / cum_push[la]),
        cum_rds_quantiles=q(-cum_rds[la]), cum_sdr_quantiles=q(-cum_sdr[la]),            # sinking-positive parts of cum_net = -(sum r ds) - (sum s dr)
        # two-point decomposition m_B - m_A = r_A (s_B - s_A) + s_B (r_B - r_A), A = end of task 10, B = end of task 100 (sinking-positive)
        twopoint_total_sink_quantiles=q(-(m_end[T1 - 1] - m_end[T0 - 2])[la]), twopoint_width_part_quantiles=q(-(r_end[T0 - 2] * (s_end[T1 - 1] - s_end[T0 - 2]))[la]),
        twopoint_r_part_quantiles=q(-(s_end[T1 - 1] * (r_end[T1 - 1] - r_end[T0 - 2]))[la]),
        layer_twopoint_total=float(-(m_end[T1 - 1] - m_end[T0 - 2])[la].sum()), layer_twopoint_width=float(-(r_end[T0 - 2] * (s_end[T1 - 1] - s_end[T0 - 2]))[la].sum()), layer_twopoint_r=float(-(s_end[T1 - 1] * (r_end[T1 - 1] - r_end[T0 - 2]))[la].sum()),
        cov_r_ds_over_rbar_sumds_median=float(np.median([(np.where(live[:, i], (r_end[st][:, i] - r_end[st][:, i].mean()) * ds[:, i], 0).sum()) / (r_end[st][:, i].mean() * np.where(live[:, i], ds[:, i], 0).sum() + 1e-30) for i in np.where(la)[0]])),
        # chain identity at task ends: F^{Y_t}(theta_{t+1}) (fit residual, position direction) vs S(theta_{t+1}) (next switching push); same units (sum over N inputs)
        fit_residual_over_S_abs_quantiles=q((np.abs(FY_end[en]) / (np.abs(S_end[en]) + 1e-30))[live]), fit_residual_sign_frac_pos=float((FY_end[en][live] > 0).mean()),
        acc_end_phase2_quantiles=q(acc_end[en]),
        layer_cum_net=float(cum_net[la].sum()), layer_cum_rds=float(-cum_rds[la].sum()), layer_cum_sdr=float(-cum_sdr[la].sum()),
        layer_cum_push=float(cum_push[la].sum()), layer_cum_ret=float(cum_ret[la].sum()),
        dr_per_task_mean=float(np.mean(dr[:, la])), ds_per_task_mean=float(np.mean(ds[:, la])), dr_per_task_sd=float(np.std(dr[:, la])),
        r_task10_quantiles=q(r_end[T0 - 2][la]), r_task100_quantiles=q(r_end[T1 - 1][la]), s_task10_quantiles=q(s_end[T0 - 2][la]), s_task100_quantiles=q(s_end[T1 - 1][la]),
        dr_per_task_median=float(np.median(dr[:, la])), ds_per_task_median=float(np.median(ds[:, la])),
        # capacity push and self term (P170 version) along phase 2
        A_neg_unit_tasks=int(((A_ph <= 0) & alive[st]).sum()), Aal_neg_unit_tasks=int(((Aal_ph <= 0) & alive[st]).sum()),
        cum_A_over_cum_Aal_quantiles=q(cum_A[la] / cum_Aal[la]), units_cum_A_nonpositive=int((cum_A[la] <= 0).sum()),
        per_task_A_over_Aal_quantiles=q((A_ph / Aal_ph)[alive[st]]),
        # actual switching push S at phase-2 states
        frac_S_positive_unit_tasks=float((S_ph[alive[st]] > 0).mean()), units_cum_S_positive=int((cum_S[la] > 0).sum()),
        cum_S_quantiles=q(cum_S[la]), cum_S_open_quantiles=q(cum_So[la]), cum_S_closed_quantiles=q(cum_Sc[la]),
        S_open_positive_frac=float((So_end[st][alive[st]] > 0).mean()), S_closed_negative_frac=float((Sc_end[st][alive[st]] < 0).mean()),
        pmax_phase2_quantiles=q(pmax[st]),
        # does the model push A predict the actual switching push S at the same state?
        spearman_A_S_per_task_median=float(np.median([np.corrcoef(np.argsort(np.argsort(A_ph[k][alive[st][k]])), np.argsort(np.argsort(S_ph[k][alive[st][k]])))[0, 1] for k in range(len(st))])),
        sign_agree_A_S=float(((A_ph > 0) == (S_ph > 0))[alive[st]].mean()),
        # running cumulative ratio stability: cum A / cum Aal after 30, 60, 90 tasks (median over units alive throughout)
        running_cumA_ratio_median=[float(np.median((np.where(alive[st][:n], A_ph[:n], 0).sum(0) / np.where(alive[st][:n], Aal_ph[:n], 0).sum(0))[la])) for n in (30, 60, 90)],
    )
    summary[tag] = res
    print(f"=== {tag} (a={a}) phase 2 tasks {T0}..{T1}: units alive in >=80 tasks {la.sum()}/{H} ===")
    for k_, v in res.items():
        if k_ in ("a", "H"): continue
        print(f"  {k_}: {v}")
    # per-unit arrays for later figures
    np.savez(os.path.join(OUT, f"tally_{tag}.npz"), m_end=m_end, s_end=s_end, r_end=r_end, p_end=p_end, alive=alive, S_end=S_end, S_open=So_end, S_closed=Sc_end,
             m_bot=m_bot, A_end=A_end, Aal_end=Aal_end, start_idx=st, push=push, ret=ret, net=net, rds=rds, sdr=sdr, live=live, live_all=la)
json.dump(summary, open(os.path.join(OUT, "tally_saved_states.json"), "w"), indent=1)
print("saved", os.path.join(OUT, "tally_saved_states.json"))

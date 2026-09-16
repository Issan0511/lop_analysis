"""POSTHOC (registered=0, analysis_grade=posthoc_not_preregistered), written 2026-09-16 before spec_l2_wall_0916
was registered; its outputs are listed in the spec §0. Inputs are read from absolute paths on white-san:
checkpoints copied to ~/Projects/obsidian-research-data/elu_three_studies_0913/ and, for h3_mucap_ref_l2.py,
the ref arm of the running mucap_el_0916 (cap arms not read). Run from the repo root of proj_004_drift or
this worktree; outputs are in results/l2_wall_0916/posthoc/."""
"""H3 x rho/mu-wall: what the running mucap_el_0916 *ref* arm (RL, ELU1 -> leaky, 150 tasks) already
says about the second layer. Posthoc, unregistered, ref arm only (cap arms are not read here)."""
import numpy as np
base = "/home/issan/Projects/claude/wt/mucap_el_0916/results/mucap_el_0916/runs"
TASKS = (1, 2, 5, 10, 20, 50, 100, 150)
np.set_printoptions(precision=3, suppress=True)

def med(a): return float(np.nanmedian(a))

for s in (0, 1, 2):
    d = np.load(f"{base}/ref_s{s}/units.npz")
    P = f"s{s}_"
    task = d[P + "task"].astype(int); step = d[P + "step"].astype(int)
    spt = step.max(); end = step == spt
    T = task[end]
    g = lambda k: d[P + k][end]
    zb2, sg2, d2, b2, q2, v2, m2, wt2 = (g(k) for k in ("zbar_l2", "sigma_l2", "d_l2", "b2", "q_l2", "v_norm_l2", "row_mean_l2", "wt_norm_l2"))
    U2, pp2, K2, gate2 = (g(k) for k in ("U_l2", "pplus_l2", "K_l2", "gate_mean_l2"))
    s2m, s2sd, mu2 = g("s2_mean")[:, 0], g("s2_sd")[:, 0], g("mu2_norm")[:, 0]
    zb1, d1, q1, b1, gate1, sg1, U1 = (g(k) for k in ("zbar_l1", "d_l1", "q_l1", "b1", "gate_mean_l1", "sigma_l1", "U_l1"))
    print(f"\n=== ref seed {s} ===  (task-end points, {len(T)} tasks, fixed 1200 train images)")
    print("t | L1: med d1  med zb1  gate1 | L2: med d2  med zb2  med sig2  gate2  p+2  U2<0 | S2 mean  sd  c2S=mean/sd  |mu2| | med m2  med rhoS  med |b2|/sig2 | corr(zb2, m2*S2+b2)  rel.resid | share q2|mu2| of zb2")
    for t in TASKS:
        i = np.where(T == t)[0][0]
        c2S = s2m[i] / s2sd[i]
        rhoS = (s2sd[i] ** 2 * m2[i] ** 2) / sg2[i] ** 2
        pred1 = m2[i] * s2m[i] + b2[i]
        resid = np.linalg.norm(zb2[i] - pred1) / np.linalg.norm(zb2[i])
        corr = np.corrcoef(zb2[i], pred1)[0, 1]
        share = np.nanmedian((q2[i] * mu2[i]) / np.where(np.abs(zb2[i]) > 1e-9, zb2[i], np.nan))
        print(f"{t:3d} | {med(d1[i]):+.2f} {med(zb1[i]):+7.2f} {med(gate1[i]):.3f} | {med(d2[i]):+.2f} {med(zb2[i]):+7.2f} {med(sg2[i]):6.2f} {med(gate2[i]):.3f} {med(pp2[i]):.3f} {np.mean(U2[i] < 0):.2f} | {s2m[i]:+7.2f} {s2sd[i]:6.2f} {c2S:+6.2f} {mu2[i]:6.2f} | {med(m2[i]):+.4f} {med(rhoS):.3f} {med(np.abs(b2[i]) / sg2[i]):.3f} | {corr:.3f} {resid:.3f} | {share:.2f}")
    # Delta zbar2 ledger in the mu basis, task-end to task-end (q2 measured against the *current* e2,
    # so Delta q2 mixes own update and rotation of e2; only |mu2| growth is cleanly 'upstream').
    print("window | mean over units: dzb2 = dq2*|mu2|_old + q2_old*d|mu2| + dq2*d|mu2| + db2   (closure check)")
    for (ta, tb) in ((1, 10), (10, 50), (50, 150), (1, 150)):
        a = np.where(T == ta)[0][0]; b = np.where(T == tb)[0][0]
        dq = q2[b] - q2[a]; dmu = mu2[b] - mu2[a]; db = b2[b] - b2[a]
        t_self = dq * mu2[a]; t_up = q2[a] * dmu; t_x = dq * dmu
        tot = zb2[b] - zb2[a]
        print(f"t{ta}->t{tb} | {tot.mean():+8.3f} = {t_self.mean():+8.3f} + {t_up.mean():+8.3f} + {t_x.mean():+8.3f} + {db.mean():+8.3f}   closure {np.abs(tot - (t_self + t_up + t_x + db)).max():.1e}")
    # sigma2 vs components: is sigma2 growth carried by q2 (mean-direction) or v2 (pattern)?
    print("t | med |q2|  med |v2|  med sig2  sig2/|w2|  ;  L1 med |q1|  med sig1  U1 med")
    for t in (1, 10, 50, 150):
        i = np.where(T == t)[0][0]
        w2n = np.sqrt(q2[i] ** 2 + v2[i] ** 2)
        print(f"{t:3d} | {med(np.abs(q2[i])):.3f} {med(v2[i]):.3f} {med(sg2[i]):.3f} {med(sg2[i] / w2n):.3f} ; {med(np.abs(q1[i])):.3f} {med(sg1[i]):.3f} {med(U1[i]):+.2f}")

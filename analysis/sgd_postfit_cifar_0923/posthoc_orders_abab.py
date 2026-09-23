#!/usr/bin/env python3
"""POST HOC (0923 night, not registered).  (1) Over the switch interval s_sw(t) -> s_sw(t+1) of LR_iid (crossover-fork
checkpoints, t in {10,20,30,40,48} x 10 seeds), split the W1 norm change into the first-order term 2<W_sw, D> and the
squared displacement |D|^2, with the push (D = P + R) and without it (D = R0, the (Wsw,Ssw) cell).  (2) ABAB: are the
successive switch displacements R_AB, R_BA reverses of each other?  Adam LR_abab (altlabels archive) vs pure-SGD SA_abab.
    python3 analysis/sgd_postfit_cifar_0923/posthoc_orders_abab.py"""
import numpy as np, os, pandas as pd
n = lambda a: float((a * a).sum()); ip = lambda a, b: float((a * b).sum())
def w1(p): return np.load(p)["W1"].astype(np.float64)
# 1) first-/second-order split over the switch interval s_sw(t) -> s_sw(t+1), with and without the push
R = os.path.expanduser("~/Projects/obsidian-research-data/sgd_postfit_cifar_0923/results/sgd_postfit_cifar_0923")
rows = []
for t in (10, 20, 30, 40, 48):
    for s in range(10):
        sd = f"LR_std_seed{s}"
        Wsw = w1(f"{R}/xfork/t{t}_sw/snap/{sd}/t01.npz"); Wend = w1(f"{R}/A/snap/{sd}/t{t:02d}.npz")
        Wnext = w1(f"{R}/xfork/t{t}_Wend_Send/snap/{sd}/t01.npz"); Wnext0 = w1(f"{R}/xfork/t{t}_Wsw_Ssw/snap/{sd}/t01.npz")
        P, Rt, R0 = Wend - Wsw, Wnext - Wend, Wnext0 - Wsw
        D = Wnext - Wsw                                   # whole interval displacement with push
        rows.append(dict(t=t, s=s, first_with=2 * ip(Wsw, D), second_with=n(D), net_with=n(Wnext) - n(Wsw),
                         first_no=2 * ip(Wsw, R0), second_no=n(R0), net_no=n(Wnext0) - n(Wsw),
                         cos_R_R0=ip(Rt, R0) / np.sqrt(n(Rt) * n(R0)), normR=np.sqrt(n(Rt)), normR0=np.sqrt(n(R0)), normW=np.sqrt(n(Wsw))))
d = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("switch interval s_sw(t)->s_sw(t+1), W1, medians per t:")
print(d.groupby("t")[["first_with", "second_with", "net_with", "first_no", "second_no", "net_no", "cos_R_R0", "normR", "normR0", "normW"]].median().round(1).T)
# 2) ABAB (Adam LR_abab and pure-SGD SA_abab): successive switch displacements and the center's motion vs D
A = os.path.expanduser("~/Projects/obsidian-research-data/altlabels_cifar_0923/results/altlabels_cifar_0923/LR_abab/snap")
S = f"{R}/SA_abab/snap"
for name, base in (("Adam LR_abab", A), ("SGD SA_abab", S)):
    out = []
    for s in range(10):
        W = {t: w1(f"{base}/LR_std_seed{s}/t{t:02d}.npz") for t in range(1, 51)}
        for j in range(2, 25):                              # cycle j = tasks (2j-1, 2j)
            Rab = W[2 * j] - W[2 * j - 1]; Rba = W[2 * j + 1] - W[2 * j]
            M_prev = (W[2 * j - 3] + W[2 * j - 2]) / 2; M = (W[2 * j - 1] + W[2 * j]) / 2; Dj = (W[2 * j] - W[2 * j - 1]) / 2
            dM = M - M_prev
            out.append(dict(j=j, cos_ab_ba=ip(Rab, Rba) / np.sqrt(n(Rab) * n(Rba)), cos_dM_D=ip(dM, Dj) / np.sqrt(n(dM) * n(Dj)),
                            cos_dM_M=ip(dM, M_prev) / np.sqrt(n(dM) * n(M_prev)), dM_sq=n(dM), radial=2 * ip(M_prev, dM),
                            sum_sq=n(Rab + Rba), Rab_sq=n(Rab), Rba_sq=n(Rba)))
    o = pd.DataFrame(out)
    late = o[o.j >= 14]
    print(f"\n{name} (cycles 14-24, median): cos(R_AB, R_BA) {late.cos_ab_ba.median():.3f}  |R_AB+R_BA|^2/|R_AB|^2 {(late.sum_sq/late.Rab_sq).median():.3f}  "
          f"cos(dM, D) {late.cos_dM_D.median():.3f}  cos(dM, M) {late.cos_dM_M.median():.3f}  per cycle: |dM|^2 {late.dM_sq.median():.1f} radial {late.radial.median():.1f}")

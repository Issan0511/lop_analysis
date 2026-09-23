#!/usr/bin/env python3
"""POST HOC (0923 night).
(2) crossover fork: effective (input-weighted) displacement of the push P and the refit R (with push) / R0 (without)."""
import numpy as np, os, pandas as pd
SP = os.path.expanduser("~/Projects/obsidian-research-data/sgd_postfit_cifar_0923/results/sgd_postfit_cifar_0923")
B = os.path.expanduser("~/Projects/obsidian-research-data/sgd_postfit_cifar_0923/basis_cache")
QL = {}
for s in range(10):
    z = np.load(f"{B}/basis_std_s{s}.npz"); QL[s] = (z["Q"][:, :1199].astype(np.float64), z["lam"].astype(np.float64))
def w1(p): return np.load(p)["W1"].astype(np.float64)
def sn(A, s): Q, lam = QL[s]; C = A @ Q; return float((lam * C * C).sum())
def sip(A, Bm, s): Q, lam = QL[s]; return float((lam * (A @ Q) * (Bm @ Q)).sum())
rows = []
for arm in ("A", "A_ce", "F", "S_hi", "SA_iid"):
    for s in range(10):
        Ws = {t: w1(f"{SP}/{arm}/snap/LR_std_seed{s}/t{t:02d}.npz") for t in range(40, 51)}
        acc = []
        for t in range(41, 51):
            W, D = Ws[t - 1], Ws[t] - Ws[t - 1]; V, dd, wd = sn(W, s), sn(D, s), sip(W, D, s)
            acc.append((np.sqrt(V), np.sqrt(dd), wd / np.sqrt(V * dd), np.median(np.sqrt((QL[s][1] * (Ws[t] @ QL[s][0]) ** 2).sum(1)))))
        a = np.array(acc).mean(0)
        rows.append(dict(arm=arm, seed=s, W_S=a[0], D_S=a[1], c_S=a[2], sigma=a[3], W_S_star=a[1] / (2 * abs(a[2]))))
d = pd.DataFrame(rows).groupby("arm").median()
d["obs/pred"] = d.W_S / d.W_S_star
print("steady-state check (t41-50, median over seeds):"); print(d[["W_S", "D_S", "c_S", "W_S_star", "obs/pred", "sigma"]].round(3).loc[["A", "A_ce", "F", "S_hi", "SA_iid"]].to_string())
rows = []
for t in (10, 20, 30, 40, 48):
    for s in range(10):
        sd = f"LR_std_seed{s}"
        Wsw = w1(f"{SP}/xfork/t{t}_sw/snap/{sd}/t01.npz"); Wend = w1(f"{SP}/A/snap/{sd}/t{t:02d}.npz")
        Wnx = w1(f"{SP}/xfork/t{t}_Wend_Send/snap/{sd}/t01.npz"); Wnx0 = w1(f"{SP}/xfork/t{t}_Wsw_Ssw/snap/{sd}/t01.npz")
        P, R, R0 = Wend - Wsw, Wnx - Wend, Wnx0 - Wsw
        rows.append(dict(t=t, s=s, V_sw=sn(Wsw, s), dV_push=sn(Wend, s) - sn(Wsw, s), P_S=np.sqrt(sn(P, s)), cP=sip(Wsw, P, s) / np.sqrt(sn(Wsw, s) * sn(P, s)),
                         dV_ret=sn(Wnx, s) - sn(Wend, s), R_S=np.sqrt(sn(R, s)), cR=sip(Wend, R, s) / np.sqrt(sn(Wend, s) * sn(R, s)),
                         dV_ret0=sn(Wnx0, s) - sn(Wsw, s), R0_S=np.sqrt(sn(R0, s)), keep=sn(Wnx, s) - sn(Wnx0, s),
                         P_N=np.linalg.norm(P), R_N=np.linalg.norm(R), R0_N=np.linalg.norm(R0)))
e = pd.DataFrame(rows); pd.set_option("display.width", 220)
print("\ncrossover fork, effective metric (median over 50): V_sw, dV over push / refit / refit-without-push, displacement sizes |P|_S |R|_S |R0|_S, radial cos, keep = V(next) - V(next0)")
print(e.drop(columns=["t", "s"]).median().round(3).to_string())
print("\nby fork task:"); print(e.groupby("t")[["V_sw", "dV_push", "P_S", "dV_ret", "R_S", "R0_S", "dV_ret0", "keep"]].median().round(0).to_string())

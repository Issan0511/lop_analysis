#!/usr/bin/env python3
"""POST HOC (0923 night, not registered).  Is the next task's "return" the reverse of the post-fit "push"?

Uses the crossover-fork checkpoints of LR_iid (fork tasks t in {10, 20, 30, 40, 48}, seeds 0-9), all W1 in float64:
  W_sw(t)      xfork/t{t}_sw/snap/<seed>/t01.npz          weights at s_sw = hit999+500 of task t
  W_end(t)     A/snap/<seed>/t{t}.npz                      task-t end (A == LR_iid bit for bit)
  W_next(t+1)  xfork/t{t}_Wend_Send/snap/<seed>/t01.npz    task t+1 run from W_end, stopped at its own s_sw
  W_next0      xfork/t{t}_Wsw_Ssw/snap/<seed>/t01.npz      the same without the push (from W_sw, state at sw)
P = W_end - W_sw (push), R = W_next - W_end (return = shock + fit + hold of task t+1).
  push = |W_end|^2 - |W_sw|^2 = 2<W_sw,P> + |P|^2 ;  ret = 2<W_sw,R> + 2<P,R> + |R|^2
  undo_frac = -<P,R>/|P|^2 ;  keep_after_fit = |W_next|^2 - |W_next0|^2 (what the push still adds after the refit).
Band version: rows of P, R, W projected on the seed's input-PCA basis Q (1199 PCs + mu-perp; comp = the rest).
    python3 analysis/sgd_postfit_cifar_0923/posthoc_push_return.py
"""
from __future__ import annotations
import os, numpy as np, pandas as pd
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
ARCH = os.path.expanduser("~/Projects/obsidian-research-data/sgd_postfit_cifar_0923")
R = f"{ARCH}/results/sgd_postfit_cifar_0923"; B = f"{ARCH}/basis_cache"
OUT = REPO / "results/sgd_postfit_cifar_0923"
BANDS = (("top", 0, 10), ("mid1", 10, 100), ("mid2", 100, 439), ("low", 439, 1199), ("mu", 1199, 1200))
n = lambda a: float((a * a).sum()); ip = lambda a, b: float((a * b).sum())
def w1(p): return np.load(p)["W1"].astype(np.float64)

rows, brows = [], []
for s in range(10):
    Q = np.load(f"{B}/basis_std_s{s}.npz")["Q"].astype(np.float64); sd = f"LR_std_seed{s}"
    for t in (10, 20, 30, 40, 48):
        Wsw = w1(f"{R}/xfork/t{t}_sw/snap/{sd}/t01.npz"); Wend = w1(f"{R}/A/snap/{sd}/t{t:02d}.npz")
        assert np.array_equal(Wend, w1(f"{R}/xfork/t{t}_Wend_Send/snap/{sd}/t00.npz"))
        Wnext = w1(f"{R}/xfork/t{t}_Wend_Send/snap/{sd}/t01.npz"); Wnext_ws = w1(f"{R}/xfork/t{t}_Wend_Ssw/snap/{sd}/t01.npz")
        Wnext0 = w1(f"{R}/xfork/t{t}_Wsw_Ssw/snap/{sd}/t01.npz")
        P, Rt = Wend - Wsw, Wnext - Wend
        Prad = ip(P, Wsw) / n(Wsw) * Wsw
        push = n(Wend) - n(Wsw)
        rows.append(dict(t=t, s=s, N_sw=n(Wsw), push=push, push_1st=2 * ip(Wsw, P), push_2nd=n(P), P_rad_sq=n(Prad),
                         P_perp_sq=n(P - Prad), cos_P_Wsw=ip(P, Wsw) / np.sqrt(n(P) * n(Wsw)),
                         ret=n(Wnext) - n(Wend), ret_1st_old=2 * ip(Wsw, Rt), ret_1st_push=2 * ip(P, Rt), ret_2nd=n(Rt),
                         undo_frac=-ip(P, Rt) / n(P), cos_PR=ip(P, Rt) / np.sqrt(n(P) * n(Rt)),
                         cos_R_Wend=ip(Rt, Wend) / np.sqrt(n(Rt) * n(Wend)), ret0=n(Wnext0) - n(Wsw),
                         keep_after_fit=n(Wnext) - n(Wnext0), keep_ratio=(n(Wnext) - n(Wnext0)) / push,
                         keep_ratio_ws=(n(Wnext_ws) - n(Wnext0)) / push, net_interval=push + n(Wnext) - n(Wend)))
        M = dict(Wsw=Wsw, Wend=Wend, Wnext=Wnext, Wnext0=Wnext0, P=P, R=Rt)
        C = {k: v @ Q for k, v in M.items()}; comp = {k: v - C[k] @ Q.T for k, v in M.items()}
        for name, lo, hi in list(BANDS) + [("comp", None, None)]:
            g = (lambda k: comp[k]) if lo is None else (lambda k: C[k][:, lo:hi])
            pb = n(g("Wend")) - n(g("Wsw")); kb = n(g("Wnext")) - n(g("Wnext0"))
            brows.append(dict(t=t, s=s, band=name, N_sw=n(g("Wsw")), push=pb, push_share=pb / push, P_sq=n(g("P")), R_sq=n(g("R")),
                              undo=-ip(g("P"), g("R")) / n(g("P")), keep=kb, keep_frac=kb / pb,
                              ret_1st_old=2 * ip(g("Wsw"), g("R")), ret=n(g("Wnext")) - n(g("Wend"))))
df, db = pd.DataFrame(rows), pd.DataFrame(brows)
df.to_csv(OUT / "posthoc_push_return.csv", index=False); db.to_csv(OUT / "posthoc_push_return_bands.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
cols = ["N_sw", "push", "push_1st", "push_2nd", "P_rad_sq", "P_perp_sq", "cos_P_Wsw", "ret", "ret_1st_old", "ret_1st_push",
        "ret_2nd", "undo_frac", "cos_PR", "cos_R_Wend", "ret0", "keep_after_fit", "keep_ratio", "keep_ratio_ws", "net_interval"]
with open(OUT / "posthoc_push_return.txt", "w") as fh:
    print("W1, median over seeds per fork task t (columns), and over all 50 cases (last column)", file=fh)
    m = df.groupby("t")[cols].median().T; m["all"] = df[cols].median(); print(m.round(3), file=fh)
    print(f"\nkeep_after_fit > 0 in {int((df.keep_after_fit > 0).sum())}/{len(df)} cases", file=fh)
    print("\nbands, median over all 50 cases", file=fh)
    print(db.groupby("band")[["N_sw", "push", "push_share", "P_sq", "R_sq", "undo", "keep_frac", "ret_1st_old", "ret"]]
          .median().loc[["top", "mid1", "mid2", "low", "mu", "comp"]].round(3), file=fh)
    for v in ("undo", "keep_frac"):
        print(f"\n{v} by fork task (median over seeds)", file=fh)
        print(db.pivot_table(index="band", columns="t", values=v, aggfunc="median").loc[["top", "mid1", "mid2", "low", "comp"]].round(2), file=fh)
print((OUT / "posthoc_push_return.txt").read_text())

import sys, numpy as np
"""Tables from posthoc_unit_state_mlp.py records (post-hoc, not registered).
usage: posthoc_read_unit_state.py [RUN ...]   (reads results/swish_battle_0917/posthoc_mlp/<RUN>.npz)
columns: u = median zbar/sd, g = median mean gate, neg = share of inputs with gate < 0,
off = median |mean output| / sd, Emu = share of activation energy in the mean vector,
rk u/c = effective rank uncentered / centered, c2 = |mu1| / sd of the layer-1 output along
mu-hat, rho = share of z2 variance along mu-hat, m2/sd and b2/sd = the two parts of layer-2
depth, wneg = share of layer-2 units with negative weight on mu-hat, down = share of units
whose mean preactivation plain GD pushes down, aW = alpha*W, loss = task-end CE on 1200 images."""
from pathlib import Path
S = str(Path(__file__).resolve().parents[2] / "results" / "swish_battle_0917" / "posthoc_mlp")
runs = sys.argv[1:] or ["SW1_s0", "SWA1_s0", "SWA1u_s0", "SWA3_s0", "SWA3_s2"]
T = [1, 2, 3, 5, 7, 8, 9, 10, 12, 14, 20, 30, 50]
med = lambda a: float(np.median(a))
for r in runs:
    d = dict(np.load(f"{S}/{r}.npz"))
    print(f"== {r}  repro mismatched cells {int(d['repro_mismatch'])}  rows {int(d['n_rows'])}")
    print(" t |  u1   g1   neg1  off1  Emu1 rk1u rk1c |  u2   g2   neg2  off2  Emu2 rk2u rk2c |  c2   rho  m2/sd  b2/sd wneg | down1 down2 | aW1  aW2 | loss")
    for t in T:
        i = t - 1
        if i >= len(d["u1"]):
            continue
        off = lambda l: med(np.abs(d[f"abar{l}"][i]) / np.maximum(d[f"asd{l}"][i], 1e-30))
        down = lambda l: float((d[f"push{l}"][i] < 0).mean())
        print(f"{t:2d} | {med(d['u1'][i]):5.2f} {med(d['gmean1'][i]):5.2f} {d['gneg1'][i].mean():5.2f} {off(1):5.2f} {float(d['mean_energy1'][i]):4.2f} {float(d['erank_unc1'][i]):5.1f} {float(d['erank_cen1'][i]):5.1f}"
              f" | {med(d['u2'][i]):5.2f} {med(d['gmean2'][i]):5.2f} {d['gneg2'][i].mean():5.2f} {off(2):5.2f} {float(d['mean_energy2'][i]):4.2f} {float(d['erank_unc2'][i]):5.1f} {float(d['erank_cen2'][i]):5.1f}"
              f" | {float(d['c2'][i]):5.2f} {med(d['rho_mu'][i]):4.2f} {med(d['m2_over_sd'][i]):6.2f} {med(d['b2_over_sd'][i]):6.2f} {float((d['w_mu'][i] < 0).mean()):4.2f}"
              f" | {down(1):4.2f} {down(2):4.2f} | {med(d['alphaW1'][i]):4.1f} {med(d['alphaW2'][i]):4.1f} | {float(d['loss'][i]):.3f}")

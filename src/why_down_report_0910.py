"""Tables for why_down_posthoc_0910: where does the activation error e_i(x) come from?"""
import csv, json
import numpy as np
from src import boundary_gradient_0908 as G

OUT = G.ROOT / 'results/why_down_posthoc_0910'
WIN = ['t21-25', 't96-100']; PH = ['1-20', '21-100', '101-625']; SIDE = ['z<=0', 'z>0']; INK = ['dark Q1', 'Q2', 'Q3', 'bright Q4']


def main(arm):
    A = np.load(OUT / f'{arm}_s0_cells.npy')       # [win, phase, side, ink, correct, (sum e, sum eS, n)]
    prov = json.load(open(OUT / f'{arm}_s0_provenance.json'))
    print(f'===== {arm}   trajectory maxabs {prov["trajectory_maxabs"]}   ink quartiles {np.round(prov["ink_quartiles"], 1)}')
    print('mean e_i(x) = dL/da_i per (unit,sample) cell;  >0 : the loss wants LESS activation on that sample')
    for w in range(2):
        print(f'\n--- window {WIN[w]} ---')
        print(f"{'phase':8s} {'side':6s} | {'mean e (all)':>13s} {'n':>9s} | " + ' '.join(f'{k:>12s}' for k in INK) + f" | {'correct':>10s} {'wrong':>10s} | {'share of dL/dm from this side':>30s}")
        for ph in range(3):
            tot_eS = A[w, ph, :, :, :, 1].sum()
            for s in (0, 1):
                c = A[w, ph, s]; n = c[:, :, 2].sum(); me = c[:, :, 0].sum() / max(n, 1)
                byink = [c[q, :, 0].sum() / max(c[q, :, 2].sum(), 1) for q in range(4)]
                bycor = [c[:, k, 0].sum() / max(c[:, k, 2].sum(), 1) for k in (1, 0)]
                share = c[:, :, 1].sum() / tot_eS if abs(tot_eS) > 0 else np.nan
                print(f"{PH[ph]:8s} {SIDE[s]:6s} | {me:+13.3e} {int(n):9d} | " + ' '.join(f'{v:+12.3e}' for v in byink) + f" | {bycor[0]:+10.3e} {bycor[1]:+10.3e} | {share:+30.2f}")
        # fraction of samples predicted correctly, by phase
        for ph in range(3):
            n_c = A[w, ph, :, :, 1, 2].sum(); n_all = A[w, ph, :, :, :, 2].sum()
            print(f"  accuracy of the running prediction in {PH[ph]}: {n_c / n_all:.3f}")
    # per unit: mean e on each side vs occupancy at the switch
    U = list(csv.DictReader(open(OUT / f'{arm}_s0_units.csv')))
    p0 = np.array([float(u['pos0']) for u in U])
    edges = [0, .005, .02, .05, .1, .2, .4, 1.01]
    print(f"\nper unit (both windows, {len(U)} unit-tasks): mean e on the positive side / negative side, by occupancy at the switch")
    print(f"{'pos0 bin':14s} {'n':>4s} | {'e+ (1-20)':>11s} {'e+ (21-625)':>12s} | {'e- (1-20)':>11s} {'e- (21-625)':>12s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p0 >= lo) & (p0 < hi)
        if m.sum() < 10: continue
        def mean_e(side, phases):
            e = sum(float(u[f'e_{side}_{ph}']) for u in np.array(U)[m] for ph in phases); n = sum(float(u[f'n_{side}_{ph}']) for u in np.array(U)[m] for ph in phases)
            return e / n if n > 0 else np.nan
        print(f"[{lo:.3f},{hi:.2f})   {m.sum():4d} | {mean_e(1, [0]):+11.3e} {mean_e(1, [1, 2]):+12.3e} | {mean_e(0, [0]):+11.3e} {mean_e(0, [1, 2]):+12.3e}")


if __name__ == '__main__':
    import sys
    for arm in (sys.argv[1:] or ['LR', 'SNA']):
        main(arm)

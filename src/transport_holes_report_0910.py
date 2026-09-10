"""Verdicts for spec_transport_holes_0910 (sub-runs A/B/C/D).

Written and smoke-verified BEFORE the run, as the parent spec's judgment module
was.  The rho statistic, the bands, the gates and the A/B/C/D/E labels are NOT
re-derived here: they are imported from `clamp_horizon_report_0910`, whose
implementation is the live form of spec_clamp_horizon_0910 §2 as amended by
追補 1.  Only the arm list, the output directory and the arbitration change,
plus the three labels this spec adds:

  A1  INVERSION_ACCUMULATES / INVERSION_SATURATES / PARTIAL   (sub-run A)
  F   DEPTH_DRIVES_DEATH / DEATH_NOT_FROM_DEPTH / PARTIAL     (sub-run B, ReLU)
  G   DEPTH_DRIVES_INVERSION / INVERSION_NOT_FROM_DEPTH / PARTIAL  (sub-run C)
  D   VALLEY_SIGN_FLIPS / VALLEY_SIGN_HOLDS / VALLEY_SIGN_MIXED (sub-run D)
"""
from pathlib import Path
import csv, json

import numpy as np

from src import clamp_horizon_report_0910 as R
from src import boundary_gradient_0908 as G

ROOT = G.ROOT
CLAMP_OUT = ROOT / 'results/clamp_horizon_acts_0910'
LONG_OUT = ROOT / 'results/long_horizon_acts_0910'
WHY_OUT = ROOT / 'results/why_down_acts_0910'
REPORT = ROOT / 'results/transport_holes_0910'
CLAMP_ARMS = ['R', 'GELU', 'SILU']
LONG_ARMS = ['GELU', 'SILU']
# LR is a validation arm, not a registered one: its `cells` array must reproduce
# the committed why_down_posthoc_0910 exactly, which is what shows this module
# measures the same decomposition the published result did.
WHY_ARMS = ['R', 'LR', 'ELU1', 'GELU', 'SILU']
SEEDS = [0, 1, 2]
LATE, BASE = R.LATE, R.BASE

A1_HI, A1_LO = 20, 10          # inv_units(t400) level  (registered, low-information)
A1P_HI, A1P_LO = 15, 5         # inv_units(t400) - inv_units(t20)  (追補 1: the accumulation)
FG_HI, FG_LO = 20, 5           # how much dclamp must reduce the counter to "drive" it
A1PP_HI, A1PP_LO = 15, 3       # frozen_units(t400) - frozen_units(t20)  (追補 2)
A2_TOL = 0.3                   # pt/100task, for VALLEY_FALLS_FASTER
A3_TIE = 0.5                   # pt: cumulative losses closer than this are a tie
LONG_REF = ROOT / 'results/long_horizon_0910'
REF_ARMS = ['LR', 'ELU1', 'SNA']


def rows_of(out, arm, seed, name='{arm}_none_s{seed}_rows.csv'):
    p = out / name.format(arm=arm, seed=seed)
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        for k, v in list(r.items()):
            if k in ('arm', 'iv', 'clamp'):
                continue
            r[k] = float(v) if v not in ('', None) else np.nan
    return rows


def med(vals):
    v = [x for x in vals if np.isfinite(x)]
    return float(np.median(v)) if v else np.nan


def agree3(vals, hi, lo, hi_label, lo_label, mid='PARTIAL', harm_label=None):
    """3/3 agreement or nothing -- the house rule, with the effect-size floor
    written into the call rather than left implicit.

    The harm band is cut FIRST.  Without it, "the intervention made the counter
    much worse" falls into the null label, which is the parent addendum's R3
    defect: a reversal is an active finding, not an absence of one."""
    if any(not np.isfinite(v) for v in vals):
        return 'NOT_TESTABLE'
    if harm_label is not None and all(v <= -abs(lo) for v in vals):
        return harm_label
    if all(v >= hi for v in vals):
        return hi_label
    if all(-abs(lo) < v <= lo for v in vals):
        return lo_label
    return mid


def _long_horizon_reference():
    """LR / ELU1 / SNA late slope, cumulative loss and width from the committed
    long_horizon_0910, read with THIS spec's LATE and BASE windows so A2 and A3
    compare like with like."""
    out = {}
    for arm in REF_ARMS:
        try:
            rows = rows_of(LONG_REF, arm, 0)
        except FileNotFoundError:
            return {}
        late = [r for r in rows if LATE[0] <= r['task'] <= LATE[1]]
        if len(late) < 2:
            return {}
        t = np.array([r['task'] for r in late])
        a = np.array([r['acc'] for r in late])
        base = med([r['acc'] for r in rows if BASE[0] <= r['task'] <= BASE[1]])
        out[arm] = dict(slope=float(np.polyfit(t, a, 1)[0]) * 100 * 100,
                        cum=100 * (base - med([r['acc'] for r in late])),
                        cnorm=late[-1]['cnorm'])
    return out


def _width_order(ref, arm, cum, cnorm):
    """long_horizon_0910 found cumulative loss ordered by ||W~||.  Add this arm and
    ask whether the ordering still holds over all the points.  A pair whose losses
    are within A3_TIE pt is not counted either way."""
    if not ref:
        return 'NOT_TESTABLE'
    pts = [(v['cnorm'], v['cum'], a) for a, v in ref.items()] + [(cnorm, cum, arm)]
    if any(not np.isfinite(x) for p_ in pts for x in p_[:2]):
        return 'NOT_TESTABLE'
    bad = ties = 0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dw = pts[i][0] - pts[j][0]
            dl = pts[i][1] - pts[j][1]
            if abs(dl) < A3_TIE:
                ties += 1
            elif dw * dl < 0:
                bad += 1
    return 'WIDTH_ORDER_HOLDS' if bad == 0 else 'WIDTH_ORDER_BREAKS'


# --------------------------------------------------------------------- sub-run A
def report_A():
    out, rows = [], []
    for arm in LONG_ARMS:
        try:
            per = {s: rows_of(LONG_OUT, arm, s) for s in SEEDS}
        except FileNotFoundError as e:
            print('A: missing', e)
            continue
        inv = [per[s][-1]['inv_units'] for s in SEEDS]
        inv20 = [med([r['inv_units'] for r in per[s] if r['task'] == 20]) for s in SEEDS]
        dinv = [a - b for a, b in zip(inv, inv20)]
        froz = [per[s][-1].get('frozen_units', np.nan) for s in SEEDS]
        froz20 = [med([r.get('frozen_units', np.nan) for r in per[s] if r['task'] == 20])
                  for s in SEEDS]
        dfroz = [a - b for a, b in zip(froz, froz20)]
        dead = [per[s][-1]['dead_hard'] for s in SEEDS]
        immob = [per[s][-1]['immobile_units'] for s in SEEDS]
        beyond = [per[s][-1]['beyond_frac'] for s in SEEDS]
        slope, cum = [], []
        for s in SEEDS:
            late = [r for r in per[s] if LATE[0] <= r['task'] <= LATE[1]]
            if len(late) < 2:               # a run that stopped short: report, do not crash
                slope.append(np.nan)
                cum.append(np.nan)
                continue
            t = np.array([r['task'] for r in late])
            a = np.array([r['acc'] for r in late])
            slope.append(float(np.polyfit(t, a, 1)[0]) * 100 * 100)
            base = med([r['acc'] for r in per[s] if BASE[0] <= r['task'] <= BASE[1]])
            cum.append(100 * (base - med([r['acc'] for r in late])))
        A1 = agree3(inv, A1_HI, A1_LO, 'INVERSION_ACCUMULATES', 'INVERSION_SATURATES')
        # 追補 1: A1 measures the LEVEL, whose baseline is already ~68 by t8, so it
        # is near-decided before the run.  A1' measures the accumulation from the
        # branch point, which is what the derivation actually predicts.
        A1p = agree3(dinv, A1P_HI, A1P_LO, 'INVERSION_ACCUMULATES2', 'INVERSION_SATURATES2')
        # 追補 2: inv_units is a count out of 100 that is already ~68 by t8, so
        # A1' has a CEILING where A1 had a floor.  frozen_units starts at 0 and is
        # the quantity the derivation actually names -- past the valley a unit
        # escapes until phi' -> 0 and freezes.
        A1pp = agree3(dfroz, A1PP_HI, A1PP_LO, 'ESCAPE_FREEZES', 'ESCAPE_DOES_NOT_FREEZE')
        # A2: is the valley arm's late slope at least A2_TOL pt/100task MORE
        # negative than ELU1's?  ELU1's own late slope comes from the committed
        # long_horizon_0910, read with the same LATE window.
        REF = _long_horizon_reference()
        A2 = 'NOT_TESTABLE'
        if REF and np.isfinite(REF['ELU1']['slope']) and all(np.isfinite(x) for x in slope):
            d = [REF['ELU1']['slope'] - x for x in slope]      # >0 means the valley falls faster
            A2 = agree3(d, A2_TOL, A2_TOL - 1e-12,
                        'VALLEY_FALLS_FASTER', 'VALLEY_SAME_BAND')
            if A2 == 'VALLEY_SAME_BAND' and not all(abs(x) < A2_TOL for x in d):
                A2 = 'PARTIAL'
        # A3: does "cumulative loss follows the order of ||W~||" survive the two
        # extra points?  Five arms: LR, ELU1, SNA from long_horizon_0910 plus this one.
        A3 = _width_order(REF, arm, med(cum), med([per[s][-1]['cnorm'] for s in SEEDS]))
        out.append(dict(arm=arm, A1_inversion=A1, A1p_accumulation=A1p,
                        A1pp_freezing=A1pp, A2_slope=A2, A3_width_order=A3,
                        frozen_units_t20=';'.join(f'{x:.0f}' for x in froz20),
                        frozen_units_t400=';'.join(f'{x:.0f}' for x in froz),
                        delta_frozen_units=';'.join(f'{x:+.0f}' for x in dfroz),
                        inv_units_t400=';'.join(f'{x:.0f}' for x in inv),
                        inv_units_t20=';'.join(f'{x:.0f}' for x in inv20),
                        delta_inv_units=';'.join(f'{x:+.0f}' for x in dinv),
                        dead_hard_t400=';'.join(f'{x:.0f}' for x in dead),
                        immobile_units_t400=';'.join(f'{x:.0f}' for x in immob),
                        beyond_frac_t400=';'.join(f'{x:.3f}' for x in beyond),
                        slope_late=';'.join(f'{x:+.2f}' for x in slope),
                        cum_loss_pt=';'.join(f'{x:.2f}' for x in cum),
                        cnorm_t400=';'.join(f"{per[s][-1]['cnorm']:.2f}" for s in SEEDS),
                        zbar_t400=';'.join(f"{per[s][-1]['zbar_inv']:+.2f}" for s in SEEDS)))
        rows += [dict(arm=arm, seed=s, **{k: per[s][-1][k] for k in
                                          ('acc', 'cnorm', 'zbar_inv', 'inv_units', 'dead_hard',
                                           'immobile_units', 'gate_neg_frac', 'beyond_frac', 'sat')})
                 for s in SEEDS]
    return out, rows


# ------------------------------------------------------------------ sub-runs B/C
def report_BC():
    """Reuse clamp_horizon_report_0910's per_seed / labels against our OUT dir."""
    R.OUT = CLAMP_OUT                  # module-global read by R.read / R.per_seed
    R.ARMS = CLAMP_ARMS
    verdicts, seedrows = [], []
    for arm in CLAMP_ARMS:
        try:
            P = {s: R.per_seed(arm, s) for s in SEEDS}
        except FileNotFoundError as e:
            print('BC: missing', e)
            continue
        except (TypeError, ValueError) as e:
            # R.per_seed fits a line over the late window; a run that stopped short
            # gives an empty vector.  Report the arm as missing rather than crashing
            # the whole judgment for the arms that did finish.
            print('BC: incomplete', arm, e)
            continue
        testable = [P[s]['ref']['L_ref'] >= R.G0_MIN_LOSS for s in SEEDS]
        g3 = {c: [P[s][c]['ce_frac'] >= R.G3_CE_FRAC for s in SEEDS] for c in R.CLAMPS}
        g4 = [abs(P[s]['dclamp']['zbar_late'] - P[s]['dclamp']['z20'])
              < R.G4_HOLD * abs(P[s]['ref']['zbar_late'] - P[s]['ref']['z20']) for s in SEEDS]
        g5 = [P[s]['wclamp']['zbar_late'] <= P[s]['ref']['zbar_late'] + R.G5_DEPTH_TOL
              for s in SEEDS]

        def pairs(c):
            o = []
            for s in SEEDS:
                ok = testable[s] and g3[c][s]
                o.append((P[s][c]['rho'] if ok else np.nan,
                          P[s][c]['slope_late'] - P[s]['ref']['slope_late'] if ok else np.nan))
            return o

        pA, pB, pE = pairs('dclamp'), pairs('wclamp'), pairs('wcap2')
        rA, rB, rE = [x[0] for x in pA], [x[0] for x in pB], [x[0] for x in pE]

        def gated(base, clamp, extra=None, name=None):
            if not all(testable):
                return 'NOT_TESTABLE_NO_LOSS'
            if not all(g3[clamp]):
                return 'NOT_TESTABLE_LEARNING_BROKEN'
            if extra is not None and not all(extra):
                return name
            return base

        def pref(tag, lab):
            return f'{tag}_{lab}'.replace('REMOVES', 'REMOVES_LOSS') \
                if lab in ('REMOVES', 'NO_EFFECT', 'PARTIAL', 'HARMS', 'LEVEL_SHIFT_ONLY') else lab

        A = gated(pref('DEPTH', R.label3(pA, R.rule_rho)), 'dclamp', g4,
                  'NOT_TESTABLE_DEPTH_NOT_HELD')
        B = gated(pref('WIDTH', R.label3(pB, R.rule_rho)), 'wclamp')
        Cb = 'WIDTH_WITHOUT_DEPTH' if all(R.rule_rho(*x) == 'REMOVES' for x in pB) else 'NOT_SHOWN'
        Cl = gated(Cb, 'wclamp', g5, 'NOT_TESTABLE_DEPTH_ALSO_STOPPED')
        sr = [P[s]['dclamp']['slope_sigma_late'] / P[s]['ref']['slope_sigma_late']
              if P[s]['ref']['slope_sigma_late'] not in (0, None) else np.nan for s in SEEDS]
        Db = 'DEPTH_WITHOUT_WIDTH' if (all(R.rule_rho(*x) == 'REMOVES' for x in pA)
                                       and all(np.isfinite(x) and x >= R.D_SLOPE_MIN
                                               for x in sr)) else 'NOT_SHOWN'
        Dl = gated(Db, 'dclamp', g4, 'NOT_TESTABLE_DEPTH_NOT_HELD')
        if B != 'WIDTH_REMOVES_LOSS':
            El = 'NOT_TESTABLE_NO_WIDTH_EFFECT'
        elif any(not (np.isfinite(a) and np.isfinite(b)) for a, b in zip(rE, rB)):
            El = gated('NOT_SHOWN', 'wcap2')
        elif all(a > b + .1 for a, b in zip(rE, rB)):
            # tested FIRST: E is only reached when rho_wclamp >= 0.5, so b + 0.1 >
            # 0.5b - 0.1 always and DOSE_GRADED would otherwise swallow this branch
            El = 'CAP_BEATS_CLAMP'
        elif all(a >= .5 * b - .1 for a, b in zip(rE, rB)):
            El = 'DOSE_GRADED'
        elif all(a <= .2 * b for a, b in zip(rE, rB)):
            El = 'DOSE_THRESHOLD'
        else:
            El = 'DOSE_PARTIAL'

        # F / G: does stopping the sinking reduce the death / inversion counter?
        counter = 'dead_hard' if arm == 'R' else 'inv_units'
        drop = []
        for s in SEEDS:
            rows = rows_of(CLAMP_OUT, arm, s)
            def last(c):
                e = [r for r in rows if r['clamp'] == c]
                return e[-1][counter] if e else np.nan
            drop.append(last('ref') - last('dclamp'))
        if arm == 'R':
            FG = agree3(drop, FG_HI, FG_LO, 'DEPTH_DRIVES_DEATH', 'DEATH_NOT_FROM_DEPTH',
                        harm_label='DEPTH_RAISES_DEATH')
        else:
            FG = agree3(drop, FG_HI, FG_LO, 'DEPTH_DRIVES_INVERSION', 'INVERSION_NOT_FROM_DEPTH',
                        harm_label='DEPTH_RAISES_INVERSION')
        if not all(g4):
            FG = 'NOT_TESTABLE_DEPTH_NOT_HELD'

        j = lambda v, f='{:+.3f}': ';'.join(f.format(x) if np.isfinite(x) else 'NA' for x in v)
        verdicts.append(dict(arm=arm, A_depth=A, B_width=B, C_dissoc=Cl, D_dissoc=Dl, E_dose=El,
                             FG_counter=FG, counter_name=counter,
                             counter_drop_dclamp=j(drop, '{:+.0f}'),
                             rho_dclamp=j(rA), rho_wclamp=j(rB), rho_wcap2=j(rE),
                             dslope_dclamp=j([x[1] for x in pA], '{:+.2f}'),
                             dslope_wclamp=j([x[1] for x in pB], '{:+.2f}'),
                             sigma_slope_ratio_dclamp=j(sr),
                             L_ref_pt=j([100 * P[s]['ref']['L_ref'] for s in SEEDS], '{:.2f}'),
                             zbar_late=';'.join(
                                 f"{P[s]['ref']['zbar_late']:+.2f}/{P[s]['wclamp']['zbar_late']:+.2f}"
                                 f"/{P[s]['dclamp']['zbar_late']:+.2f}" for s in SEEDS),
                             g0=''.join('1' if x else '0' for x in testable),
                             g3=''.join('1' if all(g3[c]) else '0' for c in R.CLAMPS),
                             g4=''.join('1' if x else '0' for x in g4),
                             g5=''.join('1' if x else '0' for x in g5),
                             joint=_joint(arm, A, B)))
        for s in SEEDS:
            for c in R.CLAMPS:
                seedrows.append(dict(arm=arm, seed=s, clamp=c, **P[s][c]))
    return verdicts, seedrows


# --------------------------------------------------------------------- sub-run D
def report_D():
    out = []
    for arm in WHY_ARMS:
        p = WHY_OUT / f'{arm}_s0_provenance.json'
        if not p.exists():
            print('D: missing', p)
            continue
        ck = json.loads(p.read_text())['checks']
        f = np.array(ck['pos_side_force_by_window_phase'])           # window x phase
        n = np.array(ck['neg_side_force_by_window_phase'])
        cells = n[:, 1:]                                             # phases 21-100, 101-625
        # 追補 2: the PRIMARY statistic for a valley arm is the inverted region
        # z < z_c, not the registered z <= 0 -- the latter also contains the
        # restoring shoulder (z_c, 0], whose gate is positive and up to 4x larger.
        # The z <= 0 sum is kept as the registered secondary.
        inv3 = np.array(ck.get('force3_by_window_phase_bin', []))
        prim = inv3[:, 1:, 2] if inv3.size else cells
        if arm == 'R':
            lab = 'RELU_ZERO' if float(np.abs(cells).max()) == 0. else 'RELU_NONZERO_BUG'
        elif arm in ('GELU', 'SILU'):
            lab = ('VALLEY_SIGN_FLIPS' if (prim > 0).all()
                   else 'VALLEY_SIGN_HOLDS' if (prim < 0).all() else 'VALLEY_SIGN_MIXED')
        else:
            lab = ('SIGN_SINKS' if (cells > 0).all()
                   else 'SIGN_LIFTS' if (cells < 0).all() else 'SIGN_MIXED')
        out.append(dict(arm=arm, D_sign=lab,
                        inverted_force=f"{ck.get('inverted_force', float('nan')):+.4g}",
                        shoulder_force=f"{ck.get('shoulder_force', float('nan')):+.4g}",
                        neg_force_total=f'{float(cells.sum()):+.4g}',
                        neg_by_window_phase=';'.join(f'{x:+.3g}' for x in cells.ravel()),
                        pos_force_total=f'{float(f[:, 1:].sum()):+.4g}',
                        beyond_force=';'.join(
                            f'{x:+.3g}' for x in np.array(ck['beyond_force_by_window_phase'])[:, 1:].ravel()),
                        gate_shift_control=f"{ck.get('neg_side_force_gate_shifted', float('nan')):+.4g}",
                        trajectory_maxabs=ck['trajectory_maxabs'],
                        trajectory_compared=ck['trajectory_compared']))
    return out


def _joint(arm, A, B):
    """This spec registered the SAME prediction from both people for every arm
    (spec §5 and 追補 1), so the parent's ISSA_SINKING / CLAUDE_WIDTH table does not
    apply -- reusing it would print a winner in a contest nobody entered.  Score the
    JOINT prediction instead: R was predicted DEPTH_NO_EFFECT + WIDTH_REMOVES_LOSS,
    the valley arms DEPTH_REMOVES_LOSS."""
    if A.startswith('NOT_TESTABLE') or B.startswith('NOT_TESTABLE'):
        return 'NOT_TESTABLE'
    if arm == 'R':
        want = (A == 'DEPTH_NO_EFFECT' and B == 'WIDTH_REMOVES_LOSS')
        wrong = (A == 'DEPTH_REMOVES_LOSS')
    else:
        want = (A == 'DEPTH_REMOVES_LOSS')
        wrong = (A in ('DEPTH_NO_EFFECT', 'DEPTH_HARMS'))
    if want:
        return 'BOTH_RIGHT'
    if wrong:
        return 'BOTH_WRONG'
    return 'UNDECIDED'


def md(rows, cols):
    return '\n'.join(['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)] +
                     ['| ' + ' | '.join(str(r.get(c, '')) for c in cols) + ' |' for r in rows])


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    A, Arows = report_A()
    BC, BCrows = report_BC()
    D = report_D()
    for name, rr in [('verdict_A.csv', A), ('verdict_BC.csv', BC), ('verdict_D.csv', D),
                     ('seed_A.csv', Arows), ('seed_BC.csv', BCrows)]:
        if rr:
            keys = []
            [keys.append(k) for r in rr for k in r if k not in keys]
            G.B.csvwrite(REPORT / name, [{k: r.get(k) for k in keys} for r in rr])
    S = ['# transport_holes_0910 summary', '',
         'spec `specs/spec_transport_holes_0910.md`（単独 commit）。V9 §11 の穴 1〜3 を箱 B で塞ぐ 4 副走。', '',
         '## 副走 A — 谷越え型の参照軌道 t1–400', '',
         md(A, ['arm', 'A1_inversion', 'A1p_accumulation', 'A1pp_freezing', 'A2_slope',
                'A3_width_order']), '',
         md(A, ['arm', 'inv_units_t20', 'inv_units_t400', 'delta_inv_units',
                'frozen_units_t20', 'frozen_units_t400', 'delta_frozen_units',
                'dead_hard_t400', 'immobile_units_t400', 'beyond_frac_t400',
                'slope_late', 'cum_loss_pt', 'cnorm_t400', 'zbar_t400']), '',
         '`inv_units` = プローブ標本の 50% 超で φ′<0 のユニット数（**上限 100 で t20 で既に飽和気味**）。'
         '`frozen_units` = 谷の向こうに居てゲートが float32 で 0 に潰れたユニット数（**0 から始まる**・'
         '導出が名指しする終状態）。`dead_hard` は符号付きゲートの counter なので谷越え型では反転を死と'
         '読み違える。`immobile_units` は |φ′| で測った本当の不動。', '',
         '## 副走 B・C — クランプ', '',
         md(BC, ['arm', 'A_depth', 'B_width', 'C_dissoc', 'D_dissoc', 'E_dose', 'FG_counter',
                 'joint']), '',
         md(BC, ['arm', 'rho_dclamp', 'rho_wclamp', 'rho_wcap2', 'L_ref_pt',
                 'counter_drop_dclamp', 'g0', 'g3', 'g4', 'g5']), '',
         '## 副走 D — 駆動源の符号', '',
         md(D, ['arm', 'D_sign', 'inverted_force', 'shoulder_force', 'neg_force_total',
                'pos_force_total', 'gate_shift_control', 'trajectory_maxabs']), '',
         md(D, ['arm', 'neg_by_window_phase', 'beyond_force']), '',
         '負側の Σφ′eS が正なら沈み、負なら浮き。ReLU は φ′(z≤0)=0 で厳密に 0（`gate_shift_control` が'
         '同じ和をゲート +1e−12 で取ったもので、0 でないことがこの 0 の非空虚性）。', '']
    (REPORT / 'summary.md').write_text('\n'.join(S))
    for v in A:
        print('A', v['arm'], v['A1_inversion'], '|', v['A1p_accumulation'], '|',
              v['A1pp_freezing'], '|', v['A2_slope'], '|', v['A3_width_order'])
        print('    inv t20->t400', v['inv_units_t20'], '->', v['inv_units_t400'],
              'delta', v['delta_inv_units'],
              '| frozen', v['frozen_units_t20'], '->', v['frozen_units_t400'],
              'delta', v['delta_frozen_units'])
    for v in BC:
        print('BC', v['arm'], v['A_depth'], '|', v['B_width'], '|', v['C_dissoc'], '|',
              v['E_dose'], '|', v['FG_counter'], '=>', v['joint'])
        print('    rho d/w/cap:', v['rho_dclamp'], '/', v['rho_wclamp'], '/', v['rho_wcap2'],
              ' L_ref', v['L_ref_pt'], 'pt')
    for v in D:
        print('D', v['arm'], v['D_sign'], 'inverted', v['inverted_force'],
              'shoulder', v['shoulder_force'], 'z<=0', v['neg_force_total'])


if __name__ == '__main__':
    main()

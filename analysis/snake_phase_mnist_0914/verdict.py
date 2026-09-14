"""snake_phase_mnist_0914 verdict (spec §6-§7).  Reads only rows.csv / units.npz / provenance.json.

    python3 analysis/snake_phase_mnist_0914/verdict.py --runs results/snake_phase_mnist_0914/runs --out results/snake_phase_mnist_0914
    python3 analysis/snake_phase_mnist_0914/verdict.py --selftest

Interpretations fixed before any main-run result was read (also printed in summary.md):
  * M1 labels that gate wording and STRUCTURE use layer 1; layer 2 is reported.
  * CI = inversion of the exact sign-flip test in the location shift, bisection to 1e-4 pt,
    reporting the outer end.  Sign-flip statistic = sum of paired differences, two-sided.
  * Holm uses the family's registered m even when contrasts are INCOMPLETE.
"""
import argparse, csv, json, math, os, sys, tempfile, zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ALPHA = 0.05
BASE, LATE = (16, 30), (101, 120)
CT_MARGIN = 0.09877
FAMILIES = {
    'F1': (6, {'C1': ('P06', 'N06'), 'C2': ('V06', 'N06'), 'C3': ('P06c', 'N06'), 'C4': ('P06i', 'N06'),
               'C5': ('V06c', 'N06'), 'C6': ('V06i', 'N06')}),
    'F1x': (3, {'I_P': (('P06', 'P06i'), ('P06c', 'N06')), 'I_V': (('V06', 'V06i'), ('V06c', 'N06')),
                'B_P': ('P06c_k1', 'P06c')}),
    'F2': (4, {'C7': ('SNAP', 'SNA'), 'C8': ('SNAV', 'SNA'), 'C7s': ('SNAP', 'SNAi_P'), 'C8s': ('SNAV', 'SNAi_V')}),
    'F3': (2, {'C9': ('LR_qKp', 'LR'), 'C9n': ('LR_qKpn', 'LR')}),
}
M1_PAIRS = {'C1': ('P06', 'N06', math.pi / 2), 'C2': ('V06', 'N06', math.pi / 2), 'C7': ('SNAP', 'SNA', math.pi / 2),
            'C8': ('SNAV', 'SNA', math.pi / 2), 'C7s': ('SNAP', 'SNAi_P', math.pi / 2),
            'C8s': ('SNAV', 'SNAi_V', math.pi / 2)}
ENDPOINTS = ('D', 'Alate', 'Abase', 'Gap', 'Afresh')


# ------------------------------------------------------------------ statistics
def n_min(m):
    return math.ceil(math.log2(2 * m / ALPHA))


def signflip_p(d):
    d = np.asarray(d, dtype=np.float64); n = len(d)
    if n == 0:
        return 1.0
    T = abs(d.sum()); tol = 1e-9 * (1 + T)
    if n <= 20:
        h = n // 2
        def sums(v):
            s = np.zeros(1)
            for x in v:
                s = np.concatenate([s + x, s - x])
            return s
        s1, s2 = sums(d[:h]), np.sort(sums(d[h:]))
        ge = len(s2) - np.searchsorted(s2, T - tol - s1, side='left')
        le = np.searchsorted(s2, -T + tol - s1, side='right')
        cnt = (ge + le).sum()
        # a sum counted in both tails only when T - tol <= s <= -T + tol, i.e. T ~ 0
        if T <= tol:
            cnt = 2 ** n
        return float(min(1.0, cnt / 2 ** n))
    rng = np.random.default_rng([20260914, n])
    M = 10 ** 6
    b = 0
    for _ in range(10):
        s = rng.choice([-1.0, 1.0], size=(M // 10, n)) @ d
        b += int((np.abs(s) >= T - tol).sum())
    return (b + 1) / (M + 1)


def signflip_ci(d, level):
    d = np.asarray(d, dtype=np.float64)
    if len(d) < 2:
        return (float('nan'), float('nan'))
    a = 1 - level; mu = float(d.mean()); span = float(d.max() - d.min()) + 1e-9
    def bound(direction):
        inside, outside = mu, mu + direction * (span + 1e-6)
        if signflip_p(d - outside) > a:           # the test cannot reject anywhere (n too small)
            return float(direction * math.inf)
        while abs(outside - inside) > 1e-4:
            mid = 0.5 * (inside + outside)
            if signflip_p(d - mid) > a:
                inside = mid
            else:
                outside = mid
        return outside
    return (bound(-1), bound(+1))


def holm(pvals, m, mutate_no_holm=False):
    """pvals: dict name -> p.  Returns dict name -> adjusted p (step-down with registered m)."""
    if mutate_no_holm:
        return dict(pvals)
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    adj, run = {}, 0.0
    for j, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - j) * p))
        adj[k] = run
    return adj


def test(d, m):
    d = np.asarray(d, dtype=np.float64)
    return dict(n=len(d), est=float(d.mean()) if len(d) else float('nan'), p=signflip_p(d),
                ci95=signflip_ci(d, .95), ci_m=signflip_ci(d, 1 - 2 * ALPHA / m))


def within(ci, h):
    return (not math.isnan(ci[0])) and -h <= ci[0] and ci[1] <= h


# ------------------------------------------------------------------ data
def load_runs(runs):
    runs = Path(runs); data = {}
    for d in sorted(runs.glob('*_s*')):
        pj = d / 'provenance.json'
        if not pj.exists():
            continue
        prov = json.loads(pj.read_text())
        rows = list(csv.DictReader(open(d / 'rows.csv')))
        u = np.load(d / 'units.npz')
        rec = dict(status=prov['status'],
                   acc=np.array([int(r['correct_seq']) for r in rows], dtype=np.float64) / 100.0,   # pt
                   fresh={int(r['task']): int(r['correct_fresh']) / 100.0 for r in rows if r.get('correct_fresh', '') != ''},
                   ce0=np.array([float(r['ce0']) for r in rows]), ce20=np.array([float(r['ce20']) for r in rows]),
                   units={k: u[k] for k in u.files if k.startswith('L')})
        data[(prov['arm'], int(prov['seed']))] = rec
    return data


def usable(rec, tasks=120):
    if rec is None or rec['status'] != 'COMPLETE' or len(rec['acc']) < tasks:
        return False
    return True


def broken(rec):
    ce0, ce20 = rec['ce0'][LATE[0] - 1:LATE[1]], rec['ce20'][LATE[0] - 1:LATE[1]]
    return int((ce20 < ce0).sum()) < 15


def wmean(x, w):
    return float(np.mean(x[w[0] - 1:w[1]]))


def seed_metrics(rec):
    acc = rec['acc']
    fr = rec['fresh']
    late_f = [fr.get(t, np.nan) for t in range(LATE[0], LATE[1] + 1)]
    return dict(D=wmean(acc, BASE) - wmean(acc, LATE), Alate=wmean(acc, LATE), Abase=wmean(acc, BASE),
                Gap=float(np.mean(np.array(late_f) - acc[LATE[0] - 1:LATE[1]])), Afresh=float(np.mean(late_f)))


def logN_window(cnorm, t_from, t_to):
    """units arrays are indexed [t] with t=0 the initial state."""
    return 0.5 * math.log(float((cnorm[t_from:t_to + 1] ** 2).mean()))


class Verdict:
    def __init__(self, data, seeds=range(20), mut=None):
        self.data, self.seeds, self.mut = data, list(seeds), mut or {}
        self.ok = {}
        for (arm, s), rec in data.items():
            self.ok[(arm, s)] = usable(rec) and not broken(rec)
        self.metric = {k: seed_metrics(v) for k, v in data.items() if usable(v)}

    def arm_seeds(self, arm):
        return [s for s in self.seeds if self.ok.get((arm, s))]

    def paired(self, a, b, ep):
        ss = [s for s in self.seeds if self.ok.get((a, s)) and self.ok.get((b, s))]
        if self.mut.get('swap'):
            a, b = b, a
        va = np.array([self.metric[(a, s)][ep] for s in ss]); vb = np.array([self.metric[(b, s)][ep] for s in ss])
        if self.mut.get('unpair') and len(vb) > 1:
            vb = np.roll(vb, 1)
        return va - vb, ss

    def contrast_values(self, spec, ep):
        if isinstance(spec[0], tuple):                    # interaction (a-b) - (c-d)
            (a, b), (c, d) = spec
            ss = [s for s in self.seeds if all(self.ok.get((x, s)) for x in (a, b, c, d))]
            v = np.array([(self.metric[(a, s)][ep] - self.metric[(b, s)][ep]) - (self.metric[(c, s)][ep] - self.metric[(d, s)][ep]) for s in ss])
            return v, ss
        return self.paired(spec[0], spec[1], ep)

    # ---- resolution margins (reference pairs only)
    def margins(self):
        if 'h_const' in self.mut:
            c = self.mut['h_const']; return dict(D=c, Alate=c, Abase=c, Gap=c, Afresh=c)
        se2 = {'D': [], 'Alate': [], 'Abase': [], 'Gap': []}
        def resvar(y):
            t = np.arange(len(y), dtype=np.float64)
            X = np.stack([np.ones_like(t), t], 1)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            r = y - X @ beta
            return float((r ** 2).sum() / (len(y) - 2))
        for a, b in (('N06', 'LIN'), ('LR', 'LIN')):
            for s in self.seeds:
                if not (self.ok.get((a, s)) and self.ok.get((b, s))):
                    continue
                d = self.data[(a, s)]['acc'] - self.data[(b, s)]['acc']
                vb = resvar(d[BASE[0] - 1:BASE[1]]); vl = resvar(d[LATE[0] - 1:LATE[1]])
                se2['D'].append(vb / 15 + vl / 20); se2['Alate'].append(vl / 20); se2['Abase'].append(vb / 15)
                ga = np.array([self.data[(a, s)]['fresh'][t] for t in range(101, 121)]) - self.data[(a, s)]['acc'][100:120]
                gb = np.array([self.data[(b, s)]['fresh'][t] for t in range(101, 121)]) - self.data[(b, s)]['acc'][100:120]
                se2['Gap'].append(resvar(ga - gb) / 20)
        h = {k: math.sqrt(float(np.mean(v))) if v else float('nan') for k, v in se2.items()}
        h['Afresh'] = h['Gap']
        return h

    # ---- labels
    def e_label(self, ep, t, adjp, m, h, testable):
        pos, neg = {'D': ('MORE_DECLINE', 'LESS_DECLINE'), 'Alate': ('HIGHER', 'LOWER'), 'Abase': ('HIGHER', 'LOWER'),
                    'Gap': ('GAP_LARGER', 'GAP_SMALLER'), 'Afresh': ('HIGHER', 'LOWER')}[ep]
        if t['n'] < n_min(m):
            return 'INCOMPLETE'
        if not testable:
            return 'NOT_TESTABLE_REF_FLAT'
        if adjp <= ALPHA:
            lab = pos if t['est'] > 0 else neg
            if within(t['ci_m'], h):
                lab += '_WITHIN_RESOLUTION'
            return lab
        if within(t['ci_m'], h):
            return 'EQUIVALENT'
        return 'INCONCLUSIVE'

    def run(self):
        self.h = self.margins()
        out = dict(h=self.h, contrasts={}, testable={}, m1={}, ct1={}, classes={}, answers={}, report={})
        for nm, (a, b) in (('TESTABLE_FIXED', ('N06', 'LIN')), ('TESTABLE_LEAKY', ('LR', 'LIN'))):
            v, ss = self.paired(a, b, 'D')
            ci = signflip_ci(v, .95)
            out['testable'][nm] = dict(n=len(v), est=float(v.mean()) if len(v) else float('nan'), ci95=ci,
                                       holds=bool(len(v) >= 2 and ci[0] > 0))
        for fam, (m, cons) in FAMILIES.items():
            testable = True
            if fam in ('F1', 'F1x'):
                testable = out['testable']['TESTABLE_FIXED']['holds']
            if fam == 'F3':
                testable = out['testable']['TESTABLE_LEAKY']['holds']
            for ep in ENDPOINTS:
                tests = {c: test(self.contrast_values(spec, ep)[0], m) for c, spec in cons.items()}
                adj = holm({c: t['p'] for c, t in tests.items() if t['n'] >= 1}, m, self.mut.get('no_holm'))
                for c, t in tests.items():
                    t['p_holm'] = adj.get(c, 1.0)
                    if fam == 'F1x' and c.startswith('I_'):
                        if t['n'] < n_min(m):
                            lab = 'INCOMPLETE'
                        elif not testable:
                            lab = 'NOT_TESTABLE_REF_FLAT'
                        elif t['p_holm'] <= ALPHA:
                            lab = 'NON_ADDITIVE'
                        elif within(t['ci_m'], self.h['D' if ep == 'D' else ep]):
                            lab = 'ADDITIVE'
                        else:
                            lab = 'ADDITIVE_INCONCLUSIVE'
                    else:
                        lab = self.e_label(ep, t, t['p_holm'], m, self.h[ep], testable)
                    t['label'] = lab
                    out['contrasts'].setdefault(c, dict(family=fam, m=m))[ep] = t
        # FRESH_LEVEL_CONFOUNDED modifier
        for c, r in out['contrasts'].items():
            af = r.get('Afresh')
            r['modifiers'] = []
            if af and af['n'] >= 1 and af['p_holm'] <= ALPHA and not c.startswith('I_'):
                r['modifiers'].append('FRESH_LEVEL_CONFOUNDED')
            for arm in self._arms_of(c):
                bad = [s for s in self.seeds if (arm, s) in self.data and not self.ok[(arm, s)]]
                div = [s for s in bad if self.data[(arm, s)]['status'] == 'DIVERGED']
                brk = [s for s in bad if s not in div and usable(self.data[(arm, s)])]
                if div:
                    r['modifiers'].append(f'DIVERGED_{len(div)}/{len(self.seeds)}({arm})')
                if brk:
                    r['modifiers'].append(f'BROKEN_{len(brk)}/{len(self.seeds)}({arm})')
        for c, r in out['contrasts'].items():
            if not c.startswith('I_') and c != 'B_P':
                out['classes'][c] = composite(r, self.mut.get('class_order'))
        out['m1'] = self.m1()
        out['ct1'] = self.ct1()
        out['answers'] = answers(out['contrasts'], out['m1'])
        out['report'] = self.report()
        return out

    def _arms_of(self, c):
        for fam, (m, cons) in FAMILIES.items():
            if c in cons:
                spec = cons[c]
                return [x for pair in spec for x in pair] if isinstance(spec[0], tuple) else list(spec)
        return []

    def m1(self):
        res = {}
        for c, (th, ref, dth) in M1_PAIRS.items():
            coef = 2 * abs(math.sin(dth / 2))
            for layer in (1, 2):
                for wn, w in (('base', BASE), ('late', LATE)):
                    vals = []
                    for s in self.seeds:
                        if not (self.ok.get((th, s)) and self.ok.get((ref, s))):
                            continue
                        Au = self.data[(th, s)]['units'][f'L{layer}_A'][w[0]:w[1] + 1].mean(0)
                        gb = self.data[(ref, s)]['units'][f'L{layer}_gbar'][w[0]:w[1] + 1].mean(0)
                        q75, q25 = np.percentile(gb, [75, 25])
                        vals.append(coef * float(np.median(Au)) - float(q75 - q25))
                    v = np.array(vals); ci = signflip_ci(v, .95)
                    lab = 'PHASE_ACTIVE' if ci[0] > 0 else ('PHASE_WASHED' if ci[1] < 0 else 'PHASE_BORDERLINE')
                    if len(v) < 2:
                        lab = 'INCOMPLETE'
                    res[f'{c}_L{layer}_{wn}'] = dict(n=len(v), est=float(v.mean()) if len(v) else float('nan'), ci95=ci, label=lab)
        return res

    def ct1(self):
        res = {}
        for layer in (1, 2):
            tests = {}
            for c in ('C1', 'C2', 'C7', 'C8'):
                a, b = FAMILIES['F1'][1].get(c) or FAMILIES['F2'][1][c]
                ss = [s for s in self.seeds if self.ok.get((a, s)) and self.ok.get((b, s))]
                v = np.array([logN_window(self.data[(a, s)]['units'][f'L{layer}_cnorm'], *LATE)
                              - logN_window(self.data[(b, s)]['units'][f'L{layer}_cnorm'], *LATE) for s in ss])
                tests[c] = test(v, 4)
            adj = holm({c: t['p'] for c, t in tests.items() if t['n'] >= 1}, 4, self.mut.get('no_holm'))
            for c, t in tests.items():
                t['p_holm'] = adj.get(c, 1.0)
                if t['n'] < n_min(4):
                    lab = 'INCOMPLETE'
                elif t['p_holm'] <= ALPHA:
                    lab = 'GROWTH_ENHANCED' if t['est'] > 0 else 'GROWTH_SUPPRESSED'
                    if within(t['ci_m'], CT_MARGIN):
                        lab += '_BELOW_CONDA_SCALE'
                elif within(t['ci_m'], CT_MARGIN):
                    lab = 'GROWTH_BELOW_CONDA_SCALE'
                else:
                    lab = 'INCONCLUSIVE'
                t['label'] = lab
                res[f'{c}_L{layer}'] = t
        return res

    def report(self):
        rep = {}
        arms = sorted({a for a, _ in self.data})
        for arm in arms:
            ss = self.arm_seeds(arm)
            if not ss:
                continue
            r = {}
            L = []
            for s in ss:
                acc = self.data[(arm, s)]['acc']
                L.append(float(np.median(acc[15:20]) - np.median(acc[100:120])))
            r['L_compat_median'] = float(np.median(L))
            r['D_mean'] = float(np.mean([self.metric[(arm, s)]['D'] for s in ss]))
            r['Alate_mean'] = float(np.mean([self.metric[(arm, s)]['Alate'] for s in ss]))
            r['Gap_mean'] = float(np.mean([self.metric[(arm, s)]['Gap'] for s in ss]))
            u = [self.data[(arm, s)]['units'] for s in ss]
            lg = [np.log(x['L1_cnorm'][120] / x['L1_cnorm'][0]) for x in u]
            r['L1_logratio_sd_mean'] = float(np.mean([np.std(v, ddof=1) for v in lg]))
            r['L1_strict_shrink_frac'] = float(np.mean([(x['L1_cnorm'][120] ** 2 < x['L1_cnorm'][0] ** 2).mean() for x in u]))
            def spear(a, b):
                ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
                return float(np.corrcoef(ra, rb)[0, 1])
            r['L1_spearman_c1_c120'] = float(np.mean([spear(x['L1_cnorm'][1], x['L1_cnorm'][120]) for x in u]))
            for layer in (1, 2):
                r[f'L{layer}_logN_late_mean'] = float(np.mean([logN_window(x[f'L{layer}_cnorm'], *LATE) for x in u]))
            r['n'] = len(ss)
            rep[arm] = r
        # main effects (REPORT, 95% CI)
        for ph, (P, Pi, Pc) in (('peak', ('P06', 'P06i', 'P06c')), ('valley', ('V06', 'V06i', 'V06c'))):
            ss = [s for s in self.seeds if all(self.ok.get((x, s)) for x in (P, Pi, Pc, 'N06'))]
            for ep in ('D', 'Alate'):
                g = lambda a, s: self.metric[(a, s)][ep]
                S = np.array([0.5 * ((g(Pi, s) - g('N06', s)) + (g(P, s) - g(Pc, s))) for s in ss])
                K = np.array([0.5 * ((g(Pc, s) - g('N06', s)) + (g(P, s) - g(Pi, s))) for s in ss])
                rep[f'main_S_{ph}_{ep}'] = dict(n=len(S), est=float(S.mean()) if len(S) else float('nan'), ci95=signflip_ci(S, .95))
                rep[f'main_K_{ph}_{ep}'] = dict(n=len(K), est=float(K.mean()) if len(K) else float('nan'), ci95=signflip_ci(K, .95))
        # CT1 of the decomposition contrasts (REPORT)
        for c in ('C3', 'C4', 'C5', 'C6'):
            a, b = FAMILIES['F1'][1][c]
            ss = [s for s in self.seeds if self.ok.get((a, s)) and self.ok.get((b, s))]
            for layer in (1, 2):
                v = np.array([logN_window(self.data[(a, s)]['units'][f'L{layer}_cnorm'], *LATE) - logN_window(self.data[(b, s)]['units'][f'L{layer}_cnorm'], *LATE) for s in ss])
                rep[f'dlogN_{c}_L{layer}'] = dict(n=len(v), est=float(v.mean()) if len(v) else float('nan'), ci95=signflip_ci(v, .95))
        # r_a (M2 type) and Delta-u transient (M3 type), REPORT
        for arm, ref, q in (('P06c', 'N06', None), ('P06c_k1', 'N06', None), ('V06c', 'N06', None), ('LR_qKp', 'LR', None), ('LR_qKpn', 'LR', None)):
            qv = {'P06c': -2.1423302723290805, 'P06c_k1': 3.0936574836539084, 'V06c': 0.4756636056624139,
                  'LR_qKp': -2.1423302723290805, 'LR_qKpn': 2.1423302723290805}[arm]
            ss = [s for s in self.seeds if self.ok.get((arm, s)) and self.ok.get((ref, s))]
            for layer in (1, 2):
                for wn, w in (('t1', (1, 1)), ('base', BASE), ('late', LATE)):
                    v = np.array([(self.data[(arm, s)]['units'][f'L{layer}_abar'][w[0]:w[1] + 1].mean() -
                                   self.data[(ref, s)]['units'][f'L{layer}_abar'][w[0]:w[1] + 1].mean()) / qv for s in ss])
                    rep[f'r_a_{arm}_L{layer}_{wn}'] = dict(n=len(v), est=float(v.mean()) if len(v) else float('nan'), ci95=signflip_ci(v, .95))
        S_SHIFT = 1.3089969389957472
        for c, a, b, sa in (('P06i-N06', 'P06i', 'N06', 0.0), ('P06-P06c', 'P06', 'P06c', +S_SHIFT),
                            ('V06i-N06', 'V06i', 'N06', 0.0), ('V06-V06c', 'V06', 'V06c', -S_SHIFT)):
            ss = [s for s in self.seeds if self.ok.get((a, s)) and self.ok.get((b, s))]
            e = []
            for s in ss:
                du = np.median(self.data[(a, s)]['units']['L1_zcur'] + sa, axis=1) - np.median(self.data[(b, s)]['units']['L1_zcur'], axis=1)
                e.append(float(du[BASE[0]:BASE[1] + 1].mean() - du[LATE[0]:LATE[1] + 1].mean()))
            e = np.array(e)
            rep[f'transient_e_{c}'] = dict(n=len(e), est=float(e.mean()) if len(e) else float('nan'), ci95=signflip_ci(e, .95))
        return rep


def sig(lab):
    return lab.startswith(('MORE_DECLINE', 'LESS_DECLINE', 'HIGHER', 'LOWER', 'GAP_LARGER', 'GAP_SMALLER'))


def base_lab(lab):
    return lab.replace('_WITHIN_RESOLUTION', '')


def composite(r, mut_order=False):
    e1, al, ab, e3 = (r[k]['label'] for k in ('D', 'Alate', 'Abase', 'Gap'))
    if e1 in ('INCOMPLETE', 'NOT_TESTABLE_REF_FLAT'):
        return None
    suf = '_WITHIN_RESOLUTION' if e1.endswith('_WITHIN_RESOLUTION') else ''
    b1 = base_lab(e1); bl = base_lab(al); bb = base_lab(ab); b3 = base_lab(e3)
    rules = [
        ('CONFLICT', (b1 == 'MORE_DECLINE' and b3 == 'GAP_SMALLER') or (b1 == 'LESS_DECLINE' and b3 == 'GAP_LARGER')),
        ('TEMPORAL_LOP_MORE', b1 == 'MORE_DECLINE' and (bl == 'LOWER' or b3 == 'GAP_LARGER')),
        ('TEMPORAL_LOP_LESS', b1 == 'LESS_DECLINE' and (bl == 'HIGHER' or b3 == 'GAP_SMALLER')),
        ('BASE_SHIFT', b1 in ('MORE_DECLINE', 'LESS_DECLINE') and ((b1 == 'MORE_DECLINE' and bb == 'HIGHER') or (b1 == 'LESS_DECLINE' and bb == 'LOWER'))
         and not sig(al) and not sig(e3)),
        ('NO_DIFFERENCE_RESOLVED', b1 == 'EQUIVALENT' and bl == 'EQUIVALENT' and b3 == 'EQUIVALENT'),
        ('LEVEL_ONLY', b1 == 'EQUIVALENT' and sig(al)),
        ('LEVEL_DIFF_DECLINE_INCONCLUSIVE', b1 == 'INCONCLUSIVE' and sig(al)),
    ]
    if mut_order:
        rules = rules[1:] + rules[:1]
    for name, cond in rules:
        if cond:
            return name + (suf if name in ('CONFLICT', 'TEMPORAL_LOP_MORE', 'TEMPORAL_LOP_LESS', 'BASE_SHIFT') else '')
    return 'INCONCLUSIVE'


def s_counts(r):
    """S simple effect counts as MORE/LESS only if dA_late or E3 agrees in direction (spec §6.6)."""
    b1 = base_lab(r['D']['label'])
    if b1 == 'MORE_DECLINE':
        return base_lab(r['Alate']['label']) == 'LOWER' or base_lab(r['Gap']['label']) == 'GAP_LARGER'
    if b1 == 'LESS_DECLINE':
        return base_lab(r['Alate']['label']) == 'HIGHER' or base_lab(r['Gap']['label']) == 'GAP_SMALLER'
    return False


def answer_row(total, K, S, I, B=None):
    """total/K/S: contrast records with ['D']['label']; I: interaction label; B: branch label or None."""
    tl = total['D']['label']
    if tl in ('INCOMPLETE', 'NOT_TESTABLE_REF_FLAT'):
        return 'NOT_ANSWERABLE'
    if I == 'NON_ADDITIVE':
        return 'NON_ADDITIVE'
    ks = base_lab(K['D']['label']) in ('MORE_DECLINE', 'LESS_DECLINE')
    ss = s_counts(S)
    keq = K['D']['label'] == 'EQUIVALENT'; seq = S['D']['label'] == 'EQUIVALENT'
    if ks and ss:
        return 'BOTH'
    if ks and seq:
        if B is None:
            return 'LEVEL'
        bb = base_lab(B)
        return 'LEVEL_ROOT_GEOMETRY' if B == 'EQUIVALENT' else ('LEVEL_ACTIVATION_VALUE' if bb in ('MORE_DECLINE', 'LESS_DECLINE') else 'LEVEL')
    if ss and keq:
        return 'ORIGIN'
    if tl == 'EQUIVALENT' and keq and seq:
        return 'NONE'
    return 'INCONCLUSIVE'


def structure(r, m1, c, layer=1):
    lab = r['D']['label']
    active = all(m1.get(f'{c}_L{layer}_{w}', {}).get('label') == 'PHASE_ACTIVE' for w in ('base', 'late'))
    if base_lab(lab) in ('MORE_DECLINE', 'LESS_DECLINE') and active:
        return 'STRUCTURE'
    if lab == 'EQUIVALENT':
        return 'STRUCTURE_NOT_SHOWN'
    return 'STRUCTURE_INCONCLUSIVE'


def answers(con, m1):
    out = {}
    need = ['C1', 'C3', 'C4', 'I_P', 'B_P']
    if all(k in con for k in need):
        out['peak'] = answer_row(con['C1'], con['C3'], con['C4'], con['I_P']['D']['label'], con['B_P']['D']['label'])
    if all(k in con for k in ('C2', 'C5', 'C6', 'I_V')):
        out['valley'] = answer_row(con['C2'], con['C5'], con['C6'], con['I_V']['D']['label'])
    for c in ('C7s', 'C8s'):
        if c in con:
            out[f'structure_{c}'] = structure(con[c], m1, c)
    return out


# ------------------------------------------------------------------ output
def fmt(x, nd=3):
    if isinstance(x, (tuple, list)):
        return '[' + ', '.join(fmt(v, nd) for v in x) + ']'
    if x is None:
        return ''
    if isinstance(x, float):
        return 'nan' if math.isnan(x) else (f'{x:+.{nd}f}' if abs(x) < 1e4 else f'{x:.3e}')
    return str(x)


def write_outputs(res, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    (out / 'verdict.json').write_text(json.dumps(res, indent=1, default=float))
    with open(out / 'verdict.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['block', 'name', 'endpoint', 'window', 'label', 'n', 'estimate_pt_or_log', 'ci95_lo', 'ci95_hi',
                    'ci_m_lo', 'ci_m_hi', 'p', 'p_holm', 'margin'])
        win = {'D': 'mean t16-30 - mean t101-120', 'Alate': 't101-120', 'Abase': 't16-30', 'Gap': 'fresh-seq t101-120', 'Afresh': 'fresh t101-120'}
        for c, r in res['contrasts'].items():
            for ep in ENDPOINTS:
                t = r[ep]
                w.writerow([r['family'], c, ep, win[ep], t['label'], t['n'], t['est'], *t['ci95'], *t['ci_m'], t['p'], t['p_holm'], res['h'][ep]])
        for k, t in res['testable'].items():
            w.writerow(['testable', k, 'D', win['D'], 'HOLDS' if t['holds'] else 'FAILS', t['n'], t['est'], *t['ci95'], '', '', '', '', ''])
        for k, t in res['m1'].items():
            w.writerow(['M1', k, 'Phi', k.rsplit('_', 1)[1], t['label'], t['n'], t['est'], *t['ci95'], '', '', '', '', ''])
        for k, t in res['ct1'].items():
            w.writerow(['CT1', k, 'dlogN', 't101-120', t['label'], t['n'], t['est'], *t['ci95'], *t['ci_m'], t['p'], t['p_holm'], CT_MARGIN])
        for k, v in res['classes'].items():
            w.writerow(['class_6.4', k, '', '', v, '', '', '', '', '', '', '', '', ''])
        for k, v in res['answers'].items():
            w.writerow(['answer_6.6', k, '', '', v, '', '', '', '', '', '', '', '', ''])
    L = ['# snake_phase_mnist_0914 summary', '',
         'spec: `specs/spec_snake_phase_mnist_0914.md`（事前登録 aa78461）。数値はすべて verdict.csv / verdict.json と同じ計算から出力。単位は pt（精度 ×100）、CT1 は log。',
         '解釈（実装時に固定・本走の結果を見る前）: M1 のラベル（書き方の規則と STRUCTURE）は層 1 で判定し層 2 は併記。CI は符号反転検定の反転（二分法 1e−4 pt、外側の端）。Holm は登録の m を固定。', '',
         f"分解能マージン h（基準対 N06−LIN・LR−LIN のみ）: D {fmt(res['h']['D'])}・A_late {fmt(res['h']['Alate'])}・A_base {fmt(res['h']['Abase'])}・Gap {fmt(res['h']['Gap'])}", '',
         '## 可検定性', '', '| | n | D_pair 平均 | 95% CI | 判定 |', '|---|---:|---:|---|---|']
    for k, t in res['testable'].items():
        L.append(f"| {k} | {t['n']} | {fmt(t['est'])} | {fmt(t['ci95'])} | {'成立' if t['holds'] else '不成立'} |")
    L += ['', '## E1 時間劣化（確認的見出し）・E2 水準・E3 fresh gap', '',
          '| 族 | 対比 | E1 D（mean t16–30 − mean t101–120） | 95% CI | p_Holm | E1 | ΔA_late (t101–120) | E2 late | ΔA_base (t16–30) | E2 base | ΔGap (t101–120) | E3 | 修飾 | §6.4 |',
          '|---|---|---:|---|---:|---|---:|---|---:|---|---:|---|---|---|']
    for c, r in res['contrasts'].items():
        L.append(f"| {r['family']} | {c} | {fmt(r['D']['est'])} (n={r['D']['n']}) | {fmt(r['D']['ci95'])} | {fmt(r['D']['p_holm'])} | {r['D']['label']} | {fmt(r['Alate']['est'])} | {r['Alate']['label']} | {fmt(r['Abase']['est'])} | {r['Abase']['label']} | {fmt(r['Gap']['est'])} | {r['Gap']['label']} | {' '.join(r['modifiers'])} | {res['classes'].get(c, '')} |")
    L += ['', '## §6.6 答え', '']
    for k, v in res['answers'].items():
        L.append(f'- {k}: `{v}`')
    L += ['', '## M1 位相の効き目（Φ = 2|sin(Δθ/2)|·medianᵢ Aᵢ − IQRᵢ ḡᵢ^ref、窓内タスク平均）', '', '| 対・層・窓 | n | Φ 平均 | 95% CI | ラベル |', '|---|---:|---:|---|---|']
    for k, t in res['m1'].items():
        L.append(f"| {k} | {t['n']} | {fmt(t['est'])} | {fmt(t['ci95'])} | {t['label']} |")
    L += ['', '## CT1 成長（ΔlogN、t101–120、マージン ±0.09877）', '', '| 対比・層 | n | ΔlogN | 95% CI | p_Holm | ラベル |', '|---|---:|---:|---|---:|---|']
    for k, t in res['ct1'].items():
        L.append(f"| {k} | {t['n']} | {fmt(t['est'])} | {fmt(t['ci95'])} | {fmt(t['p_holm'])} | {t['label']} |")
    L += ['', '## REPORT（ラベルなし）', '', '### 腕ごと', '',
          '| 腕 | n | L 互換（中央値 t16–20 − 中央値 t101–120、seed 中央値） | D 平均 | A_late 平均 | Gap 平均 | 層1 log成長比 SD（t120/t0） | 層1 strict 縮小割合 | Spearman(c1,c120) | logN1 late | logN2 late |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm, r in res['report'].items():
        if isinstance(r, dict) and 'L_compat_median' in r:
            L.append(f"| {arm} | {r['n']} | {fmt(r['L_compat_median'])} | {fmt(r['D_mean'])} | {fmt(r['Alate_mean'])} | {fmt(r['Gap_mean'])} | {fmt(r['L1_logratio_sd_mean'])} | {fmt(r['L1_strict_shrink_frac'])} | {fmt(r['L1_spearman_c1_c120'])} | {fmt(r['L1_logN_late_mean'])} | {fmt(r['L2_logN_late_mean'])} |")
    L += ['', '### 主効果・分解対比の ΔlogN・r_a・起点の過渡（REPORT、95% CI）', '']
    for k, r in res['report'].items():
        if isinstance(r, dict) and 'ci95' in r:
            L.append(f"- {k}: {fmt(r['est'])} {fmt(r['ci95'])} (n={r['n']})")
    L += ['', '腕間の幅の差（CT1・logN）は W 病理の証拠ではない（spec §6.7・§14）。EQUIVALENT は「単走のタスク標本の分解能より小さい」の意味。']
    (out / 'summary.md').write_text('\n'.join(L) + '\n')


# ------------------------------------------------------------------ selftest (S14)
ARMS = ['N06', 'P06', 'V06', 'P06c', 'P06c_k1', 'P06i', 'V06c', 'V06i', 'SNA', 'SNAP', 'SNAV', 'SNAi_P', 'SNAi_V', 'LIN', 'LR', 'LR_qKp', 'LR_qKpn']


def synth(root, plant, n=20, skip=None, lr_flat=False):
    """Write synthetic run dirs.  plant[arm] = dict(decline, level, late_shift, base_shift, gap, noise_seed_sd, shared)."""
    rng = np.random.default_rng(7)
    shared = rng.normal(0, 2.0, size=n)                 # per-seed offset in D shared by all arms (pairing matters)
    tnoise = {}
    for arm in ARMS:
        pl = plant.get(arm, {})
        for s in range(n):
            if skip and arm in skip and s >= skip[arm]:
                continue
            d = Path(root) / f'{arm}_s{s}'; d.mkdir(parents=True, exist_ok=True)
            r2 = np.random.default_rng([zlib.crc32(arm.encode()), s])
            t = np.arange(1, 121)
            decl = pl.get('decline', 2.0 if arm not in ('LIN',) else 0.0)
            if lr_flat and arm.startswith('LR'):
                decl = 0.0
            seed_d = pl.get('seed_sd', 0.05) * r2.normal() + (shared[s] if pl.get('shared', True) else 0.0)
            acc = 95.0 + pl.get('level', 0.0) - (decl + seed_d) * np.clip((t - 16) / 104, 0, 1) \
                + pl.get('late_shift', 0.0) * (t >= 101) + pl.get('base_shift', 0.0) * ((t >= 16) & (t <= 30)) \
                + r2.normal(0, pl.get('task_sd', 0.3), size=120)
            fresh = acc + 1.0 + pl.get('gap', 0.0) + r2.normal(0, 0.2, size=120)
            corr = np.clip(np.round(acc * 100), 0, 10000).astype(int)
            corrf = np.clip(np.round(fresh * 100), 0, 10000).astype(int)
            ce0 = np.full(120, 2.0); ce20 = np.full(120, 1.0)
            if pl.get('broken_seed') == s:
                ce20[:] = 3.0
            with open(d / 'rows.csv', 'w') as f:
                f.write('task,correct_seq,correct_fresh,ce0,ce20\n')
                for i in range(120):
                    cf = corrf[i] if (i + 1) in [1] + list(range(16, 21)) + list(range(101, 121)) else ''
                    f.write(f'{i + 1},{corr[i]},{cf},{ce0[i]},{ce20[i]}\n')
            U = {}
            for l in (1, 2):
                c0 = np.full(100, 0.577)
                growth = pl.get('growth', 1.0)
                U[f'L{l}_cnorm'] = np.stack([c0 * (1 + (growth * 2.0 - 1) * min(tt, 120) / 120) for tt in range(121)]) * np.exp(r2.normal(0, 0.01))
                U[f'L{l}_A'] = np.full((121, 100), pl.get('A', 0.1)) + r2.normal(0, 0.005, size=(121, 100))
                U[f'L{l}_gbar'] = 1.0 + np.tile(np.linspace(-0.05, 0.05, 100), (121, 1)) * pl.get('iqr_scale', 1.0)
                U[f'L{l}_zcur'] = np.zeros((121, 100)); U[f'L{l}_abar'] = np.zeros((121, 100))
            np.savez(d / 'units.npz', **U)
            status = 'DIVERGED' if pl.get('diverged_seed') == s else 'COMPLETE'
            (d / 'provenance.json').write_text(json.dumps(dict(arm=arm, seed=s, status=status)))


def selftest():
    cases, muts = [], []
    def case(name, got, exp):
        cases.append(dict(name=name, got=got, expected=exp, ok=got == exp))
    # --- statistics against brute force
    rng = np.random.default_rng(1)
    for n in (6, 9, 12):
        d = rng.normal(0.3, 1, size=n)
        signs = np.array(np.meshgrid(*[[-1, 1]] * n)).reshape(n, -1).T
        bf = float((np.abs(signs @ d) >= abs(d.sum()) - 1e-9 * (1 + abs(d.sum()))).mean())
        case(f'signflip_bruteforce_n{n}', round(signflip_p(d), 12), round(bf, 12))
    case('signflip_all_positive_n10', signflip_p(np.arange(1, 11.)), 2 / 1024)
    ci = signflip_ci(np.full(20, 1.0) + np.linspace(-0.01, 0.01, 20), .95)
    case('ci_contains_true', bool(ci[0] <= 1.0 <= ci[1]), True)
    case('n_min', [n_min(6), n_min(4), n_min(3), n_min(2)], [8, 8, 7, 7])
    hp = holm({'a': 0.02, 'b': 0.5}, 6)
    case('holm_m6_0.02_not_sig', hp['a'] <= ALPHA, False)
    muts.append(dict(name='no_holm', detected=holm({'a': 0.02, 'b': 0.5}, 6, mutate_no_holm=True)['a'] <= ALPHA))
    # --- §6.6 table and STRUCTURE on hand-built labels
    R = lambda d, al='INCONCLUSIVE', g='INCONCLUSIVE': {'D': {'label': d}, 'Alate': {'label': al}, 'Gap': {'label': g}}
    case('6.6_not_answerable', answer_row(R('INCOMPLETE'), R('MORE_DECLINE'), R('MORE_DECLINE', 'LOWER'), 'ADDITIVE'), 'NOT_ANSWERABLE')
    case('6.6_non_additive', answer_row(R('MORE_DECLINE'), R('MORE_DECLINE'), R('EQUIVALENT'), 'NON_ADDITIVE'), 'NON_ADDITIVE')
    case('6.6_both', answer_row(R('MORE_DECLINE'), R('LESS_DECLINE'), R('MORE_DECLINE', 'LOWER'), 'ADDITIVE_INCONCLUSIVE'), 'BOTH')
    case('6.6_S_not_counted_without_level', answer_row(R('MORE_DECLINE'), R('LESS_DECLINE'), R('MORE_DECLINE', 'INCONCLUSIVE', 'INCONCLUSIVE'), 'ADDITIVE_INCONCLUSIVE'), 'INCONCLUSIVE')
    case('6.6_level_root', answer_row(R('MORE_DECLINE'), R('MORE_DECLINE'), R('EQUIVALENT'), 'ADDITIVE', 'EQUIVALENT'), 'LEVEL_ROOT_GEOMETRY')
    case('6.6_level_value', answer_row(R('MORE_DECLINE'), R('MORE_DECLINE'), R('EQUIVALENT'), 'ADDITIVE', 'LESS_DECLINE'), 'LEVEL_ACTIVATION_VALUE')
    case('6.6_level_plain', answer_row(R('MORE_DECLINE'), R('MORE_DECLINE'), R('EQUIVALENT'), 'ADDITIVE'), 'LEVEL')
    case('6.6_origin', answer_row(R('MORE_DECLINE'), R('EQUIVALENT'), R('LESS_DECLINE', 'HIGHER'), 'ADDITIVE'), 'ORIGIN')
    case('6.6_none', answer_row(R('EQUIVALENT'), R('EQUIVALENT'), R('EQUIVALENT'), 'ADDITIVE_INCONCLUSIVE'), 'NONE')
    case('6.6_inconclusive', answer_row(R('INCONCLUSIVE'), R('INCONCLUSIVE'), R('EQUIVALENT'), 'ADDITIVE_INCONCLUSIVE'), 'INCONCLUSIVE')
    act = {f'C7s_L1_{w}': {'label': 'PHASE_ACTIVE'} for w in ('base', 'late')}
    case('structure', structure(R('MORE_DECLINE'), act, 'C7s'), 'STRUCTURE')
    case('structure_needs_m1', structure(R('MORE_DECLINE'), {'C7s_L1_base': {'label': 'PHASE_ACTIVE'}, 'C7s_L1_late': {'label': 'PHASE_WASHED'}}, 'C7s'), 'STRUCTURE_INCONCLUSIVE')
    case('structure_not_shown', structure(R('EQUIVALENT'), act, 'C7s'), 'STRUCTURE_NOT_SHOWN')
    # --- §6.4 classes on hand-built labels (all 8 + overlap precedence)
    Q = lambda d, al, ab, g: {'D': {'label': d}, 'Alate': {'label': al}, 'Abase': {'label': ab}, 'Gap': {'label': g}}
    cl = [('CONFLICT', Q('MORE_DECLINE', 'LOWER', 'INCONCLUSIVE', 'GAP_SMALLER')),
          ('TEMPORAL_LOP_MORE', Q('MORE_DECLINE', 'LOWER', 'INCONCLUSIVE', 'INCONCLUSIVE')),
          ('TEMPORAL_LOP_LESS', Q('LESS_DECLINE', 'INCONCLUSIVE', 'INCONCLUSIVE', 'GAP_SMALLER')),
          ('BASE_SHIFT', Q('MORE_DECLINE', 'INCONCLUSIVE', 'HIGHER', 'INCONCLUSIVE')),
          ('NO_DIFFERENCE_RESOLVED', Q('EQUIVALENT', 'EQUIVALENT', 'INCONCLUSIVE', 'EQUIVALENT')),
          ('LEVEL_ONLY', Q('EQUIVALENT', 'LOWER', 'INCONCLUSIVE', 'INCONCLUSIVE')),
          ('LEVEL_DIFF_DECLINE_INCONCLUSIVE', Q('INCONCLUSIVE', 'HIGHER', 'INCONCLUSIVE', 'INCONCLUSIVE')),
          ('INCONCLUSIVE', Q('INCONCLUSIVE', 'INCONCLUSIVE', 'INCONCLUSIVE', 'INCONCLUSIVE')),
          ('TEMPORAL_LOP_MORE_WITHIN_RESOLUTION', Q('MORE_DECLINE_WITHIN_RESOLUTION', 'LOWER', 'INCONCLUSIVE', 'INCONCLUSIVE'))]
    for exp, r in cl:
        case(f'class_{exp}', composite(r), exp)
    muts.append(dict(name='class_order_conflict_last', detected=composite(cl[0][1], mut_order=True) != 'CONFLICT'))
    # --- full pipeline on synthetic runs
    plant = {
        'LIN': dict(decline=0.0),
        'P06': dict(decline=3.0, late_shift=-1.0, A=0.01),          # C1 MORE_DECLINE + LOWER -> TEMPORAL_LOP_MORE; M1 washed
        'V06': dict(decline=2.0, seed_sd=0.002, task_sd=0.3),       # C2 ~ N06
        'P06c': dict(decline=1.0, late_shift=1.0),                  # C3 LESS + HIGHER
        'P06i': dict(decline=2.0, seed_sd=1.0),                     # C4 INCONCLUSIVE (noisy)
        'V06c': dict(decline=2.0, base_shift=1.5),                  # C5 MORE via base -> BASE_SHIFT
        'V06i': dict(decline=2.0, level=-1.0),                      # C6 level only
        'SNAP': dict(decline=3.0, gap=-1.0),                        # C7 MORE + GAP_SMALLER -> CONFLICT
        'SNAV': dict(decline=2.0),                                  # C8 INCOMPLETE (seeds skipped)
        'LR': dict(decline=2.0, seed_sd=0.005, task_sd=0.02),
        'LR_qKp': dict(decline=2.08, seed_sd=0.005, task_sd=0.02),  # C9 small significant effect within resolution
        'LR_qKpn': dict(decline=2.0, level=-2.0, seed_sd=1.0),      # C9n level diff, decline inconclusive
        'P06c_k1': dict(decline=2.0, broken_seed=3),
        'SNAi_P': dict(decline=2.0, diverged_seed=4),
    }
    with tempfile.TemporaryDirectory() as tmp:
        synth(tmp, plant, skip={'SNAV': 5})
        data = load_runs(tmp)
        res = Verdict(data).run()
        con, cls = res['contrasts'], res['classes']
        case('pipe_testable_fixed', res['testable']['TESTABLE_FIXED']['holds'], True)
        case('pipe_C1_E1', con['C1']['D']['label'], 'MORE_DECLINE')
        case('pipe_C1_class', cls['C1'], 'TEMPORAL_LOP_MORE')
        case('pipe_C2_E1', con['C2']['D']['label'], 'EQUIVALENT')
        case('pipe_C3_E1', con['C3']['D']['label'], 'LESS_DECLINE')
        case('pipe_C3_class', cls['C3'], 'TEMPORAL_LOP_LESS')
        case('pipe_C4_E1', con['C4']['D']['label'], 'INCONCLUSIVE')
        case('pipe_C5_class', cls['C5'], 'BASE_SHIFT')
        case('pipe_C6_class', cls['C6'], 'LEVEL_ONLY')
        case('pipe_C7_class', cls['C7'], 'CONFLICT')
        case('pipe_C8_E1', con['C8']['D']['label'], 'INCOMPLETE')
        case('pipe_C9_E1', con['C9']['D']['label'], 'MORE_DECLINE_WITHIN_RESOLUTION')
        case('pipe_C9n_class', cls['C9n'], 'LEVEL_DIFF_DECLINE_INCONCLUSIVE')
        case('pipe_broken_modifier', any(m.startswith('BROKEN_1/20') for m in con['B_P']['modifiers']), True)
        case('pipe_broken_n', con['B_P']['D']['n'], 19)
        case('pipe_diverged_modifier', any(m.startswith('DIVERGED_1/20') for m in con['C7s']['modifiers']), True)
        case('pipe_ct1_label_suppressed_or_scale', res['ct1']['C1_L1']['label'] in ('GROWTH_BELOW_CONDA_SCALE', 'INCONCLUSIVE'), True)
        case('pipe_m1_active', res['m1']['C1_L1_late']['label'], 'PHASE_WASHED')
        # mutations on the pipeline
        r_sw = Verdict(data, mut={'swap': True}).run()
        # swapping arms flips every contrast sign; the reference pair N06-LIN then fails TESTABLE
        muts.append(dict(name='swap_arms', detected=r_sw['contrasts']['C1']['D']['est'] < 0
                         and r_sw['contrasts']['C1']['D']['label'] != 'MORE_DECLINE'))
        r_up = Verdict(data, mut={'unpair': True}).run()
        w = lambda r, c: r['contrasts'][c]['D']['ci95'][1] - r['contrasts'][c]['D']['ci95'][0]
        muts.append(dict(name='unpair_seeds', detected=w(r_up, 'C3') > w(res, 'C3')
                         and r_up['contrasts']['C2']['D']['label'] != 'EQUIVALENT'))
        r_h = Verdict(data, mut={'h_const': 5.0}).run()
        muts.append(dict(name='h_const', detected=r_h['contrasts']['C4']['D']['label'] != 'INCONCLUSIVE'))
    with tempfile.TemporaryDirectory() as tmp:
        synth(tmp, {'LIN': dict(decline=0.0)}, lr_flat=True)
        res2 = Verdict(load_runs(tmp)).run()
        case('pipe_F3_not_testable', res2['contrasts']['C9']['D']['label'], 'NOT_TESTABLE_REF_FLAT')
        # M1 active case: large A in theta arm, tiny IQR in ref
    with tempfile.TemporaryDirectory() as tmp:
        synth(tmp, {'LIN': dict(decline=0.0), 'SNAP': dict(A=0.6), 'SNA': dict(iqr_scale=0.1)})
        res3 = Verdict(load_runs(tmp)).run()
        case('pipe_m1_phase_active', res3['m1']['C7_L1_late']['label'], 'PHASE_ACTIVE')
    with tempfile.TemporaryDirectory() as tmp:
        synth(tmp, {'LIN': dict(decline=0.0), 'P06': dict(growth=1.5)})
        res4 = Verdict(load_runs(tmp)).run()
        case('pipe_ct1_growth_enhanced', res4['ct1']['C1_L1']['label'].startswith('GROWTH_ENHANCED'), True)
    allp = all(c['ok'] for c in cases) and all(m['detected'] for m in muts)
    j = dict(all_pass=allp, cases=cases, mutations=muts)
    (HERE / 'verdict_selftest.json').write_text(json.dumps(j, indent=1, default=str))
    for c in cases:
        if not c['ok']:
            print('CASE FAIL', c)
    for m in muts:
        if not m['detected']:
            print('MUTATION NOT DETECTED', m)
    print('selftest all_pass', allp)
    return allp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs'); ap.add_argument('--out'); ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    data = load_runs(a.runs)
    res = Verdict(data).run()
    write_outputs(res, a.out)
    print('wrote', a.out)


if __name__ == '__main__':
    main()

"""sgd_bridge_mnist_0914 verdict (spec §4).  Statistics helpers are imported from the registered
snake_phase_mnist_0914 verdict (sign-flip test, inversion CI, Holm, n_min, E labels, composite classes).

    python3 analysis/sgd_bridge_mnist_0914/verdict.py --runs results/sgd_bridge_mnist_0914/runs --out results/sgd_bridge_mnist_0914
    python3 analysis/sgd_bridge_mnist_0914/verdict.py --selftest
"""
import argparse, csv, json, math, sys, tempfile, zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'analysis/snake_phase_mnist_0914'))
import verdict as V   # noqa: E402

ALPHA, MARGIN = 0.05, 0.09877
SEEDS = list(range(10))
CONS = {'C1': ('P06', 'N06'), 'C2': ('V06', 'N06')}
ADAM_REF = ROOT / 'results/sgd_bridge_mnist_0914/adam_reference.json'
ADAM_VERDICT = ROOT / 'results/snake_phase_mnist_0914/verdict.json'


def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def growth_label(t, mut=None):
    if t['n'] < V.n_min(4):
        return 'INCOMPLETE'
    if t['p_holm'] <= ALPHA:
        lab = 'GROWTH_ENHANCED' if t['est'] > 0 else 'GROWTH_SUPPRESSED'
        return lab + ('_BELOW_CONDA_SCALE' if V.within(t['ci_m'], MARGIN) else '')
    return 'GROWTH_BELOW_CONDA_SCALE' if V.within(t['ci_m'], MARGIN) else 'INCONCLUSIVE'


def x_label(t):
    if t['n'] < V.n_min(4):
        return 'INCOMPLETE'
    if t['p_holm'] <= ALPHA:
        return 'X_POSITIVE' if t['est'] > 0 else 'X_NEGATIVE'
    return 'X_EQUIVALENT' if V.within(t['ci_m'], MARGIN) else 'INCONCLUSIVE'


def direction(label):
    if label.startswith('GROWTH_ENHANCED'):
        return +1
    if label.startswith('GROWTH_SUPPRESSED'):
        return -1
    return 0


def fork_stats(cn, mut=None):
    """cn: [t, unit] array.  Returns (f, excess)."""
    c0, c1 = cn[0], cn[120]
    shrink = (c1 <= 0.95 * c0) if mut == 'tolerance' else (c1 ** 2 < c0 ** 2)
    f = float(shrink.mean())
    lg = np.log(c1 / c0)
    mu, sd = float(lg.mean()), float(lg.std(ddof=1))
    f0 = Phi(-mu / sd) if sd > 0 else (1.0 if mu < 0 else 0.0)
    return f, f - f0


def fork_label(fs, es):
    cf = V.signflip_ci(np.array(fs), .95); ce = V.signflip_ci(np.array(es), .95)
    if cf[0] > 0.01 and ce[0] > 0:
        return 'FORK_PRESENT', cf, ce
    if cf[1] < 0.01:
        return 'FORK_ABSENT', cf, ce
    return 'FORK_INCONCLUSIVE', cf, ce


def analyze(data, adam_ref, adam_labels, mut=None):
    mut = mut or {}
    v = V.Verdict(data, seeds=SEEDS)
    ok = v.ok
    out = dict(G={}, X={}, F={}, E={}, testable={}, report={}, h=None)
    logN = lambda arm, s, l: V.logN_window(data[(arm, s)]['units'][f'L{l}_cnorm'], 101, 120)
    # ---- G and X (one family of 4 each)
    gt, xt = {}, {}
    for c, (a, b) in CONS.items():
        if mut.get('swap'):
            a, b = b, a
        for l in (1, 2):
            ss = [s for s in SEEDS if ok.get((a, s)) and ok.get((b, s))]
            d = np.array([logN(a, s, l) - logN(b, s, l) for s in ss])
            gt[f'{c}_L{l}'] = V.test(d, 4)
            ad = np.array([adam_ref[f'{CONS[c][0]}_s{s}'][f'logN{l}_late'] - adam_ref[f'{CONS[c][1]}_s{s}'][f'logN{l}_late'] for s in ss])
            if mut.get('unpair') and len(ad) > 1:
                ad = np.roll(ad, 1)
            xt[f'{c}_L{l}'] = V.test(d - ad, 4)
            xt[f'{c}_L{l}']['adam_est'] = float(ad.mean()) if len(ad) else float('nan')
    for T, lab in ((gt, growth_label), (xt, x_label)):
        adj = V.holm({k: t['p'] for k, t in T.items() if t['n'] >= 1}, 4)
        for k, t in T.items():
            t['p_holm'] = adj.get(k, 1.0); t['label'] = lab(t)
    for k, t in xt.items():
        t['modifiers'] = []
        al = adam_labels.get(k, '')
        if direction(al) != 0 and direction(gt[k]['label']) == -direction(al):
            t['modifiers'].append('SIGN_REVERSED')
        t['adam_label'] = al; t['sgd_label'] = gt[k]['label']
    out['G'], out['X'] = gt, xt
    # ---- F
    for arm in ('N06', 'P06', 'V06', 'LIN', 'LR'):
        for l in (1, 2):
            fs, es = [], []
            for s in SEEDS:
                if ok.get((arm, s)):
                    f, e = fork_stats(data[(arm, s)]['units'][f'L{l}_cnorm'], mut.get('fork'))
                    fs.append(f); es.append(e)
            if len(fs) < 2:
                out['F'][f'{arm}_L{l}'] = dict(label='INCOMPLETE', n=len(fs)); continue
            lab, cf, ce = fork_label(fs, es)
            out['F'][f'{arm}_L{l}'] = dict(label=lab, n=len(fs), f_mean=float(np.mean(fs)), f_ci95=cf,
                                           excess_mean=float(np.mean(es)), excess_ci95=ce)
    # ---- E (family FS, m=2), testable
    h = v.margins(); out['h'] = h
    d, _ = v.paired('N06', 'LIN', 'D'); ci = V.signflip_ci(d, .95)
    testable = bool(len(d) >= 2 and ci[0] > 0)
    out['testable'] = dict(n=len(d), est=float(d.mean()) if len(d) else float('nan'), ci95=ci, holds=testable)
    E = {}
    for ep in V.ENDPOINTS:
        tests = {c: V.test(v.paired(a, b, ep)[0], 2) for c, (a, b) in CONS.items()}
        adj = V.holm({c: t['p'] for c, t in tests.items() if t['n'] >= 1}, 2)
        for c, t in tests.items():
            t['p_holm'] = adj.get(c, 1.0)
            t['label'] = v.e_label(ep, t, t['p_holm'], 2, h[ep], testable)
            E.setdefault(c, {})[ep] = t
    for c, r in E.items():
        r['class'] = V.composite(r)
        af = r['Afresh']
        r['modifiers'] = ['FRESH_LEVEL_CONFOUNDED'] if af['n'] >= 1 and af['p_holm'] <= ALPHA else []
    out['E'] = E
    # ---- REPORT
    rep = {}
    for arm in ('N06', 'P06', 'V06', 'LIN', 'LR'):
        ss = [s for s in SEEDS if ok.get((arm, s))]
        if not ss:
            continue
        u = [data[(arm, s)]['units'] for s in ss]
        r = dict(n=len(ss), D=float(np.mean([v.metric[(arm, s)]['D'] for s in ss])),
                 Alate=float(np.mean([v.metric[(arm, s)]['Alate'] for s in ss])),
                 Gap=float(np.mean([v.metric[(arm, s)]['Gap'] for s in ss])))
        for l in (1, 2):
            late = [V.logN_window(x[f'L{l}_cnorm'], 101, 120) for x in u]
            t0 = [0.5 * math.log(float((x[f'L{l}_cnorm'][0] ** 2).mean())) for x in u]
            r[f'logN{l}_late'] = float(np.mean(late))
            r[f'growth{l}_x'] = float(np.exp(np.mean(np.array(late) - np.array(t0))))
            ad = [adam_ref.get(f'{arm}_s{s}', {}).get(f'logN{l}_late') for s in ss]
            if all(x is not None for x in ad):
                r[f'logN{l}_late_minus_adam'] = float(np.mean(np.array(late) - np.array(ad)))
            r[f'logratio_sd{l}'] = float(np.mean([np.std(np.log(x[f'L{l}_cnorm'][120] / x[f'L{l}_cnorm'][0]), ddof=1) for x in u]))
            r[f'min_ratio{l}'] = float(min(np.min(x[f'L{l}_cnorm'][120] / x[f'L{l}_cnorm'][0]) for x in u))
        rep[arm] = r
    for c, (a, b) in CONS.items():
        for l in (1, 2):
            for wn, w in (('base', (16, 30)), ('late', (101, 120))):
                vals = []
                for s in SEEDS:
                    if not (ok.get((a, s)) and ok.get((b, s))):
                        continue
                    Au = data[(a, s)]['units'][f'L{l}_A'][w[0]:w[1] + 1].mean(0)
                    gb = data[(b, s)]['units'][f'L{l}_gbar'][w[0]:w[1] + 1].mean(0)
                    q75, q25 = np.percentile(gb, [75, 25])
                    vals.append(math.sqrt(2) * float(np.median(Au)) - float(q75 - q25))
                vv = np.array(vals); cc = V.signflip_ci(vv, .95)
                rep[f'M1_{c}_L{l}_{wn}'] = dict(n=len(vv), est=float(vv.mean()) if len(vv) else float('nan'), ci95=cc,
                                               label='PHASE_ACTIVE' if cc[0] > 0 else ('PHASE_WASHED' if cc[1] < 0 else 'PHASE_BORDERLINE'))
    out['report'] = rep
    return out


def fmt(x):
    return V.fmt(x)


def write(res, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    (out / 'verdict.json').write_text(json.dumps(res, indent=1, default=float))
    with open(out / 'verdict.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['block', 'name', 'label', 'modifiers', 'n', 'estimate', 'ci95_lo', 'ci95_hi', 'ci_m_lo', 'ci_m_hi', 'p', 'p_holm', 'extra'])
        for k, t in res['X'].items():
            w.writerow(['X', k, t['label'], ' '.join(t['modifiers']), t['n'], t['est'], *t['ci95'], *t['ci_m'], t['p'], t['p_holm'], f"adam_est={t['adam_est']} adam_label={t['adam_label']} sgd_label={t['sgd_label']}"])
        for k, t in res['G'].items():
            w.writerow(['G', k, t['label'], '', t['n'], t['est'], *t['ci95'], *t['ci_m'], t['p'], t['p_holm'], ''])
        for k, t in res['F'].items():
            w.writerow(['F', k, t['label'], '', t['n'], t.get('f_mean'), *(t.get('f_ci95') or ('', '')), '', '', '', '', f"excess={t.get('excess_mean')} {t.get('excess_ci95')}"])
        w.writerow(['testable', 'TESTABLE_FIXED', 'HOLDS' if res['testable']['holds'] else 'FAILS', '', res['testable']['n'], res['testable']['est'], *res['testable']['ci95'], '', '', '', '', ''])
        for c, r in res['E'].items():
            for ep in V.ENDPOINTS:
                t = r[ep]
                w.writerow(['E', f'{c}_{ep}', t['label'], ' '.join(r['modifiers']), t['n'], t['est'], *t['ci95'], *t['ci_m'], t['p'], t['p_holm'], f"class={r['class']}"])
    L = ['# sgd_bridge_mnist_0914 summary', '',
         'spec: `specs/spec_sgd_bridge_mnist_0914.md`。箱 B（0914 と同一）の optimizer だけを plain SGD（lr 0.02・2,500 更新/タスク）に替えた 5 腕 × seed 0–9。ΔlogN は log（t101–120 の二乗平均平方根の中心化ノルム）、E は pt。', '',
         '## X：optimizer × 位相（見出し）— ΔlogN(SGD) − ΔlogN(Adam 0914)、seed 対', '',
         '| 対比・層 | n | X | 95% CI | p_Holm | ラベル | 修飾 | ΔlogN Adam（seed 0–9） | Adam の登録ラベル（0914, 20 seed） | SGD の G ラベル |', '|---|---:|---:|---|---:|---|---|---:|---|---|']
    for k, t in res['X'].items():
        L.append(f"| {k} | {t['n']} | {fmt(t['est'])} | {fmt(t['ci95'])} | {fmt(t['p_holm'])} | {t['label']} | {' '.join(t['modifiers'])} | {fmt(t['adam_est'])} | {t['adam_label']} | {t['sgd_label']} |")
    L += ['', '## G：SGD 下の成長差（ΔlogN、t101–120、マージン ±0.09877）', '', '| 対比・層 | n | ΔlogN | 95% CI | p_Holm | ラベル |', '|---|---:|---:|---|---:|---|']
    for k, t in res['G'].items():
        L.append(f"| {k} | {t['n']} | {fmt(t['est'])} | {fmt(t['ci95'])} | {fmt(t['p_holm'])} | {t['label']} |")
    L += ['', '## F：個体の分岐（strict 縮小の割合 f、単峰近似からの超過 e）', '', '| 腕・層 | n | f 平均 | f 95% CI | e 平均 | e 95% CI | ラベル |', '|---|---:|---:|---|---:|---|---|']
    for k, t in res['F'].items():
        L.append(f"| {k} | {t['n']} | {fmt(t.get('f_mean'))} | {fmt(t.get('f_ci95'))} | {fmt(t.get('excess_mean'))} | {fmt(t.get('excess_ci95'))} | {t['label']} |")
    tb = res['testable']
    L += ['', f"## E：時間劣化・水準・fresh gap（族 FS、m=2）", '', f"TESTABLE_FIXED（D_pair N06−LIN）: {fmt(tb['est'])} {fmt(tb['ci95'])} → {'成立' if tb['holds'] else '不成立'}。分解能 h: D {fmt(res['h']['D'])}・A_late {fmt(res['h']['Alate'])}・Gap {fmt(res['h']['Gap'])}", '',
          '| 対比 | E1 D（t16–30 − t101–120） | 95% CI | E1 | ΔA_late | E2 late | ΔA_base | ΔGap | E3 | ΔA_fresh | 修飾 | §6.4 |', '|---|---:|---|---|---:|---|---:|---:|---|---:|---|---|']
    for c, r in res['E'].items():
        L.append(f"| {c} | {fmt(r['D']['est'])} | {fmt(r['D']['ci95'])} | {r['D']['label']} | {fmt(r['Alate']['est'])} | {r['Alate']['label']} | {fmt(r['Abase']['est'])} | {fmt(r['Gap']['est'])} | {r['Gap']['label']} | {fmt(r['Afresh']['est'])} | {' '.join(r['modifiers'])} | {r['class']} |")
    L += ['', '## REPORT', '', '| 腕 | n | D | A_late | Gap | 層1 成長倍率 | 層2 成長倍率 | logN1 − Adam | logN2 − Adam | log成長比 SD 層1 | 層2 | 最小倍率 層1 | 層2 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm, r in res['report'].items():
        if 'D' in r:
            L.append(f"| {arm} | {r['n']} | {fmt(r['D'])} | {fmt(r['Alate'])} | {fmt(r['Gap'])} | {fmt(r['growth1_x'])} | {fmt(r['growth2_x'])} | {fmt(r.get('logN1_late_minus_adam'))} | {fmt(r.get('logN2_late_minus_adam'))} | {fmt(r['logratio_sd1'])} | {fmt(r['logratio_sd2'])} | {fmt(r['min_ratio1'])} | {fmt(r['min_ratio2'])} |")
    L += ['']
    for k, r in res['report'].items():
        if k.startswith('M1_'):
            L.append(f"- {k}: {fmt(r['est'])} {fmt(r['ci95'])} {r['label']}")
    L += ['', 'X は optimizer と更新数（625→2,500）を合わせて替えた効果。腕間の幅の差は W 病理の証拠ではない。']
    (out / 'summary.md').write_text('\n'.join(L) + '\n')


# ------------------------------------------------------------------ selftest (S14s)
def synth(root, seeds=SEEDS):
    rng_shared = np.random.default_rng(3).normal(0, 0.3, size=len(seeds))
    adam_ref = {}
    plan = {'N06': dict(dec=2.0), 'P06': dict(dec=1.0), 'V06': dict(dec=2.0), 'LIN': dict(dec=0.0), 'LR': dict(dec=2.0)}
    for arm, pl in plan.items():
        for s in seeds:
            r = np.random.default_rng([zlib.crc32(arm.encode()), s])
            d = Path(root) / f'{arm}_s{s}'; d.mkdir(parents=True, exist_ok=True)
            t = np.arange(1, 121)
            acc = 95 - pl['dec'] * np.clip((t - 16) / 104, 0, 1) + r.normal(0, 0.3, 120)
            fresh = acc + 1 + r.normal(0, 0.2, 120)
            with open(d / 'rows.csv', 'w') as f:
                f.write('task,correct_seq,correct_fresh,ce0,ce20\n')
                for i in range(120):
                    cf = int(round(fresh[i] * 100)) if (i + 1) in [1] + list(range(16, 21)) + list(range(101, 121)) else ''
                    f.write(f'{i + 1},{int(round(acc[i] * 100))},{cf},2.0,1.0\n')
            U = {}
            for l in (1, 2):
                c0 = 0.577 * np.exp(r.normal(0, 0.02, 100))
                if l == 1 and arm == 'P06':                                    # minority shrinks beyond the normal tail; suppressed growth
                    g = np.where(np.arange(100) < 25, 0.9, 2.0) * np.exp(r.normal(0, 0.02, 100))
                elif l == 1 and arm == 'N06':
                    g = np.exp(r.normal(1.2, 0.05, 100))
                elif l == 1 and arm == 'V06':                                  # unimodal, many shrink, no excess
                    g = np.exp(r.normal(0.05, 0.10, 100))
                elif l == 2 and arm == 'N06':                                  # units at ratio 0.97 (strict counts them)
                    g = np.where(np.arange(100) < 5, 0.97, np.exp(r.normal(0.7, 0.05, 100)))
                else:
                    g = np.exp(r.normal(0.7, 0.05, 100) + (rng_shared[s] if l == 2 else 0.0))
                ts = np.linspace(0, 1, 121)[:, None]
                U[f'L{l}_cnorm'] = c0[None, :] * (1 + (g[None, :] - 1) * ts)
                U[f'L{l}_A'] = np.full((121, 100), 0.1); U[f'L{l}_gbar'] = 1 + np.tile(np.linspace(-.05, .05, 100), (121, 1))
                U[f'L{l}_zcur'] = np.zeros((121, 100)); U[f'L{l}_abar'] = np.zeros((121, 100))
            np.savez(d / 'units.npz', **U)
            (d / 'provenance.json').write_text(json.dumps(dict(arm=arm, seed=s, status='COMPLETE')))
            rec = {}
            for l in (1, 2):
                late = V.logN_window(U[f'L{l}_cnorm'], 101, 120)
                bump = 0.12 if (arm == 'P06' and l == 1) else 0.0
                # layer 2: Adam reference equals SGD plus tiny noise, so the pairing matters (shared seed term)
                rec[f'logN{l}_late'] = late + bump + (r.normal(0, 0.002) if l == 2 else 0.0)
                if arm == 'P06' and l == 1:
                    rec[f'logN{l}_late'] = V.logN_window(np.load(Path(root) / f'N06_s{s}/units.npz')['L1_cnorm'], 101, 120) + 0.12 if (Path(root) / f'N06_s{s}/units.npz').exists() else late
            adam_ref[f'{arm}_s{s}'] = rec
    return adam_ref


def selftest():
    cases, muts = [], []
    def case(n, got, exp):
        cases.append(dict(name=n, got=got, expected=exp, ok=got == exp))
    with tempfile.TemporaryDirectory() as tmp:
        ref = synth(tmp)
        data = V.load_runs(tmp)
        labels = {'C1_L1': 'GROWTH_ENHANCED', 'C2_L1': 'GROWTH_BELOW_CONDA_SCALE', 'C1_L2': 'GROWTH_ENHANCED_BELOW_CONDA_SCALE', 'C2_L2': 'GROWTH_ENHANCED'}
        r = analyze(data, ref, labels)
        case('G_C1_L1_suppressed', r['G']['C1_L1']['label'].startswith('GROWTH_SUPPRESSED'), True)
        case('X_C1_L1_negative', r['X']['C1_L1']['label'], 'X_NEGATIVE')
        case('X_C1_L1_sign_reversed', 'SIGN_REVERSED' in r['X']['C1_L1']['modifiers'], True)
        case('X_C1_L2_equivalent', r['X']['C1_L2']['label'], 'X_EQUIVALENT')
        case('X_C1_L2_not_reversed', 'SIGN_REVERSED' in r['X']['C1_L2']['modifiers'], False)
        case('F_P06_L1_present', r['F']['P06_L1']['label'], 'FORK_PRESENT')
        case('F_V06_L1_inconclusive_no_excess', r['F']['V06_L1']['label'], 'FORK_INCONCLUSIVE')
        case('F_LIN_L1_absent', r['F']['LIN_L1']['label'], 'FORK_ABSENT')
        case('E_testable', r['testable']['holds'], True)
        case('E_C1_less', r['E']['C1']['D']['label'], 'LESS_DECLINE')
        case('fork_stats_strict', round(fork_stats(np.array([[1.0, 1.0]] + [[0, 0]] * 119 + [[0.97, 1.2]]))[0], 3), 0.5)
        case('Phi', round(Phi(0.0), 6), 0.5)
        rs = analyze(data, ref, labels, mut={'swap': True})
        muts.append(dict(name='swap_arms', detected=rs['G']['C1_L1']['label'].startswith('GROWTH_ENHANCED')))
        ru = analyze(data, ref, labels, mut={'unpair': True})
        wd = lambda x: x['X']['C1_L2']['ci95'][1] - x['X']['C1_L2']['ci95'][0]
        muts.append(dict(name='unpair_adam_reference', detected=wd(ru) > wd(r) and ru['X']['C1_L2']['label'] != 'X_EQUIVALENT'))
        rt = analyze(data, ref, labels, mut={'fork': 'tolerance'})
        muts.append(dict(name='fork_tolerance', detected=rt['F']['N06_L2']['f_mean'] != r['F']['N06_L2']['f_mean']))
    allp = all(c['ok'] for c in cases) and all(m['detected'] for m in muts)
    (HERE / 'verdict_selftest.json').write_text(json.dumps(dict(all_pass=allp, cases=cases, mutations=muts), indent=1, default=str))
    for c in cases:
        if not c['ok']:
            print('CASE FAIL', c)
    for m in muts:
        if not m['detected']:
            print('MUT NOT DETECTED', m)
    print('selftest all_pass', allp)
    return allp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs'); ap.add_argument('--out'); ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    data = V.load_runs(a.runs)
    adam_ref = json.loads(ADAM_REF.read_text())['runs']
    av = json.loads(ADAM_VERDICT.read_text())['ct1']
    labels = {k: av[k]['label'] for k in ('C1_L1', 'C2_L1', 'C1_L2', 'C2_L2')}
    write(analyze(data, adam_ref, labels), a.out)
    print('wrote', a.out)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""bwdswap_1111 readout skeleton (brief item 6).  Explicit --src; refuses incomplete runs unless --allow-partial.

For each setting (RC, RM, F51) and direction (restore / reverse), on E = the next task's online accuracy
(RM: the first of its two continuation tasks; the second is reported as secondary):
  - the seed means of N, X, B (and E.1's N and X means, recorded beside them)
  - the paired differences X - N, B - N, B - X: mean, SD, 97.5% t interval (df = n - 1), 95% as well
  - per seed the sign of each difference, and the counts
Completeness gates first: every arm of every seed present and finite, and the runner's runtime checks
(branch probes, preflights, B's forward monitor, natural continuation) all passed.  No labels are assigned
here; the registered judgment is added when the predictions are registered.

    python analysis/bwdswap_1111/judge.py --src results/bwdswap_1111
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

from analysis.resp_cifar_ee_0920.stats import t_quantile

RUN = 'bwdswap_1111'
SETTINGS = {   # mirrors src/bwdswap_1111.py SETTINGS (asserted when the runner is importable)
    'RC': {'restore': dict(N='N10', X='R1_10', B='R1_10_bwd', task=11),
           'reverse': dict(N='N1r', X='S1_10r', B='S1_10r_bwd', task=2)},
    'RM': {'restore': dict(N='N10', X='R2_10', B='R2_10_bwd', task=11),
           'reverse': dict(N='N2r', X='S2_10r', B='S2_10r_bwd', task=3)},
    'F51': {'restore': dict(N='N_c', X='R_ch', B='R_ch_bwd', task=29),
            'reverse': dict(N='N_hr', X='S_hcr', B='S_hcr_bwd', task=3)},
}
E1_RESULTS = {'RC': 'resp_cifar_ee_0920', 'RM': 'resp_ee_0917', 'F51': 'resp_cifar5p1_1007'}
COMPARISONS = (('X-N', 'X', 'N'), ('B-N', 'B', 'N'), ('B-X', 'B', 'X'))
LABEL = {'restore': '戻す（劣化した網へ早期の微分）', 'reverse': '渡す（早期の網へ劣化した微分）'}


def interval(values, level):
    """Paired t interval over seeds (df = n - 1).  SD = 0 gives the point [mean, mean] (degenerate_sd)."""
    x = np.asarray(values, dtype=np.float64)
    n = len(x)
    assert n >= 2 and np.isfinite(x).all(), 'needs >= 2 finite seeds'
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    half = t_quantile((1 + level) / 2, n - 1) * sd / math.sqrt(n)
    low, high = mean - half, mean + half
    return dict(mean=mean, sd=sd, low=low, high=high, level=level, n=n, degenerate_sd=sd == 0,
                sign='+' if low > 0 else '-' if high < 0 else '0')


def seed_signs(values):
    return ['+' if v > 0 else '-' if v < 0 else '0' for v in values]


def read_csv(path):
    with Path(path).open() as fh:
        return list(csv.DictReader(fh))


def load_setting(src, setting, seeds):
    """{direction: {kind: {seed: E}}} (RM also k=2), plus completeness notes and the runtime checks."""
    out, notes, checks, e1 = {}, [], {}, {}
    if setting == 'RM':
        k2 = {}
        for s in seeds:
            d = src / 'RM' / f's{s}'
            if not (d / 'done.json').exists():
                notes.append(f'RM seed {s}: not complete')
                continue
            rc = json.loads((d / 'run_checks.json').read_text())
            checks[f's{s}'] = rc['all_pass']
            if (d / 'e1_check.json').exists():
                e1[f's{s}'] = json.loads((d / 'e1_check.json').read_text())['all_equal']
            rows = read_csv(d / 'arms.csv')
            for direction, spec in SETTINGS['RM'].items():
                for kind in 'NXB':
                    for r in rows:
                        if r['arm'] == spec[kind]:
                            ok = r['finite'] == 'True'
                            if not ok:
                                notes.append(f'RM seed {s} {spec[kind]} k{r["k"]}: not finite')
                            tgt = out if r['k'] == '1' else k2
                            tgt.setdefault(direction, {}).setdefault(kind, {})[s] = float(r['online_acc']) if ok else math.nan
        return out, notes, checks, e1, k2
    base = src / setting
    if not (base / 'run_checks.json').exists():
        notes.append(f'{setting}: no run_checks.json (run not complete)')
    else:
        rc = json.loads((base / 'run_checks.json').read_text())
        checks['run'] = rc['all_pass']
    if (base / 'e1_comparison.json').exists():
        e1 = json.loads((base / 'e1_comparison.json').read_text())
    for direction, spec in SETTINGS[setting].items():
        for kind in 'NXB':
            f = base / 'arms' / spec[kind] / 'rows.json'
            if not (f.parent / 'done.json').exists():
                notes.append(f'{setting} {spec[kind]}: not complete')
                continue
            for r in json.loads(f.read_text()):
                if r['seed'] in seeds:
                    out.setdefault(direction, {}).setdefault(kind, {})[r['seed']] = \
                        float(r['online_acc']) if r['finite'] else math.nan
    return out, notes, checks, e1, None


def e1_means(setting, direction):
    """E.1's N and X seed means of the same next task, from the committed E.1 results (recorded beside)."""
    spec = SETTINGS[setting][direction]
    res = ROOT / 'results' / E1_RESULTS[setting]
    vals = {}
    for kind in 'NX':
        if setting == 'RM':
            v = [float(r['online_acc']) for s in range(10)
                 for r in read_csv(res / 'runs' / f's{s}' / 'arms.csv') if r['arm'] == spec[kind] and r['k'] == '1']
        else:
            v = [r['online_acc'] for r in json.loads((res / 'arms' / spec[kind] / 'rows.json').read_text())]
        vals[kind] = float(np.mean(v)) if len(v) == 10 else None
    return vals


def analyse(E, seeds, allow_partial):
    """E: {kind: {seed: value}} for one setting and direction."""
    have = [s for s in seeds if all(s in E.get(k, {}) and math.isfinite(E[k][s]) for k in 'NXB')]
    complete = len(have) == len(seeds)
    if not complete and not (allow_partial and len(have) >= 2):
        return dict(complete=False, seeds_complete=have)
    res = dict(complete=complete, seeds=have, means={k: float(np.mean([E[k][s] for s in have])) for k in 'NXB'},
               per_seed=[dict(seed=s, **{k: E[k][s] for k in 'NXB'}) for s in have], paired={})
    for name, a, b in COMPARISONS:
        diff = [E[a][s] - E[b][s] for s in have]
        signs = seed_signs(diff)
        res['paired'][name] = dict(ci975=interval(diff, .975), ci95=interval(diff, .95), per_seed=diff,
                                   signs=signs, n_pos=signs.count('+'), n_neg=signs.count('-'), n_zero=signs.count('0'))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='results root of the run (contains RC/, F51/, RM/s<seed>/)')
    ap.add_argument('--seeds', default='0-9')
    ap.add_argument('--settings', default='RC,RM,F51')
    ap.add_argument('--allow-partial', action='store_true', help='smoke only: fewer seeds / missing settings')
    ap.add_argument('--out', help='where to write (default: --src)')
    a = ap.parse_args()
    src = Path(a.src).resolve()
    lo, _, hi = a.seeds.partition('-')
    seeds = list(range(int(lo), int(hi) + 1)) if hi else [int(q) for q in a.seeds.split(',')]
    out = Path(a.out).resolve() if a.out else src
    try:
        from src.bwdswap_1111 import SETTINGS as RUNNER
        for st, dirs in SETTINGS.items():
            for d, spec in dirs.items():
                assert all(RUNNER[st][d][k] == v for k, v in spec.items()), ('arm map differs from the runner', st, d)
    except ImportError:
        pass
    verdict = dict(run=RUN, src=str(src), seeds=seeds, settings={})
    per_seed_rows, paired_rows = [], []
    lines = [f'# {RUN} — 逆伝播の微分だけを入れ替える腕（B）の集計', '',
             f'src: `{src}`  seeds: {seeds[0]}–{seeds[-1]}（{len(seeds)}）。E = 次の課題の online 精度（RM は継続 2 課題のうち 1 課題目）。',
             '区間は対応差の t 区間（df = n−1）。97.5% が主、95% を併記。符号は下端 > 0 で +、上端 < 0 で −、他は 0。', '']
    complete_all = True
    for setting in a.settings.split(','):
        E, notes, checks, e1, k2 = load_setting(src, setting, seeds)
        ok_checks = bool(checks) and all(v is True for v in checks.values())
        sv = dict(notes=notes, runtime_checks=checks, runtime_checks_pass=ok_checks, e1_comparison=e1, directions={})
        lines += [f'## {setting}', '']
        if notes:
            lines += ['未完了・注記: ' + '; '.join(notes), '']
        lines += [f'実行時の検査（分岐の probe・preflight・B の順伝播の毎歩照合・自然な継続）: {"すべて通過" if ok_checks else "未通過または欠け"}', '']
        for direction in ('restore', 'reverse'):
            res = analyse(E.get(direction, {}), seeds, a.allow_partial)
            res['e1_means'] = e1_means(setting, direction)
            if k2 is not None:
                res['secondary_k2'] = analyse(k2.get(direction, {}), seeds, a.allow_partial)
            sv['directions'][direction] = res
            spec = SETTINGS[setting][direction]
            lines += [f'### {direction}: {LABEL[direction]}（次の課題 {spec["task"]}）', '']
            if 'means' not in res:
                complete_all = False
                lines += ['未完了のため集計しない。', '']
                continue
            complete_all &= res['complete']
            em = res['e1_means']
            lines += ['| 腕 | 平均 E | E.1 の平均 |', '|---|---:|---:|']
            for k in 'NXB':
                ref = em.get(k)
                lines.append(f'| {k}（{spec[k]}） | {res["means"][k]:.4f} | {"" if ref is None else f"{ref:.4f}"} |')
            lines += ['', '| 対応差 | 平均 | 97.5% 区間 | 95% 区間 | 符号 | seed（+/−/0） |', '|---|---:|---|---|---|---|']
            for name, _, _ in COMPARISONS:
                p = res['paired'][name]
                c, c95 = p['ci975'], p['ci95']
                lines.append(f'| {name} | {c["mean"]:+.4f} | [{c["low"]:+.4f}, {c["high"]:+.4f}] | '
                             f'[{c95["low"]:+.4f}, {c95["high"]:+.4f}] | {c["sign"]} | {p["n_pos"]}/{p["n_neg"]}/{p["n_zero"]} |')
                paired_rows.append(dict(setting=setting, direction=direction, comparison=name, n=c['n'], mean=c['mean'],
                                        sd=c['sd'], low975=c['low'], high975=c['high'], low95=c95['low'], high95=c95['high'],
                                        sign975=c['sign'], n_pos=p['n_pos'], n_neg=p['n_neg'], n_zero=p['n_zero']))
            lines.append('')
            lines.append('seed ごとの向き: ' + '; '.join(
                f'{name} ' + ''.join(res['paired'][name]['signs']) for name, _, _ in COMPARISONS))
            lines.append('')
            for row in res['per_seed']:
                per_seed_rows.append(dict(setting=setting, direction=direction, seed=row['seed'],
                                          E_N=row['N'], E_X=row['X'], E_B=row['B'],
                                          **{name: row[a_] - row[b_] for name, a_, b_ in COMPARISONS}))
        verdict['settings'][setting] = sv
        complete_all &= ok_checks and not notes
        if e1:
            lines += [f'E.1 との照合（記録）: `{json.dumps(e1 if setting == "RM" else {k: v for k, v in e1.items() if k != "arms"}, ensure_ascii=False)[:600]}`', '']
    verdict['complete'] = bool(complete_all)
    lines.insert(4, f'**完全性: {"COMPLETE" if complete_all else "INCOMPLETE"}**' + ('（--allow-partial）' if a.allow_partial else ''))
    lines.insert(5, '')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'verdict.json').write_text(json.dumps(verdict, ensure_ascii=False, indent=2, default=str) + '\n')
    for name, rows in (('per_seed.csv', per_seed_rows), ('paired.csv', paired_rows)):
        if rows:
            with (out / name).open('w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    if not complete_all and not a.allow_partial:
        sys.exit(2)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Registered counts and descriptive ranges; no inferential significance claims."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/chargpt_h1audit_1023'


def main():
    rows = list(csv.DictReader((OUT/'windows.csv').open()))
    rows = [{k:(v if k in ['arm', 'window'] else float(v)) for k, v in r.items()} for r in rows]
    index = {(r['arm'], int(r['seed']), r['window']):r for r in rows}
    judge = {}
    for arm in ['C3', 'ReLUcap']:
        j = {}
        for en in ['oracle', 'truth']:
            late = [index[arm, s, 'late'] for s in range(10)]
            count = sum(r[f's_{en}_load_exact']>=.75 and r[f's_{en}_load_exact']-r['s_actual_load_exact']>=.20 for r in late)
            j[f'Q1_{en}'] = dict(n_restored=count, passes=count>=8)
        for suffix in ['direct', 'direct_absK']:
            count = sum(index[arm, s, 'late'][f's_diag_self_{suffix}']>=.95 for s in range(10))
            j[f'Q2_{suffix}'] = dict(n_positive=count, passes=count>=8)
        count = sum(index[arm, s, 'early']['share_off_self_direct']>.5 for s in range(10))
        j['Q3'] = dict(n_offdiagonal_larger_early=count, passes=count>=8)
        count = sum(index[arm, s, 'early']['s_oracle_other_direct']-index[arm, s, 'late']['s_oracle_other_direct']>=.20 for s in range(10))
        j['Q4'] = dict(n_oracle_other_declines=count, passes=count>=8)
        j['Q5'] = {w:dict(n_in_40_60=sum(.4<=index[arm, s, w]['s_oracle_unitrest_direct']<=.6 for s in range(10))) for w in ['early', 'middle', 'late']}
        judge[arm] = j
    (OUT/'judge.json').write_text(json.dumps(dict(exploratory_mechanism_diagnostic=True, judges=judge), indent=2)+'\n')
    lines = ['# H1 audit: registered diagnostics', '', 'All counts are out of 10 seeds; window medians first. No training, no causal or significance test.', '']
    for arm, j in judge.items():
        lines += [f'## {arm}', '', '```json', json.dumps(j, indent=2), '```', '']
    (OUT/'judge.md').write_text('\n'.join(lines))
    fields = [
        's_actual_load_exact', 's_oracle_load_exact', 's_truth_load_exact',
        's_actual_load_direct', 's_oracle_load_direct', 's_truth_load_direct',
        's_actual_self_direct', 's_oracle_self_direct', 's_truth_self_direct',
        's_actual_self_exact', 's_oracle_self_exact',
        's_actual_other_direct', 's_oracle_other_direct', 's_truth_other_direct',
        's_actual_other_exact', 's_oracle_other_exact',
        's_actual_residual_direct', 's_oracle_residual_direct',
        's_actual_headperp_direct', 's_oracle_headperp_direct',
        's_oracle_unitrest_direct', 'share_oracle_unitrest_direct',
        's_diag_self_direct', 's_diag_self_direct_absK', 's_diag_self_exact',
        's_diag_load_direct', 's_diag_load_exact',
        's_off_self_direct', 's_off_self_exact', 's_off_load_direct', 's_off_load_exact',
        'share_off_self_direct', 'share_off_load_direct', 'share_off_load_exact',
        'share_fit_self_direct', 'share_fit_load_direct', 'share_fit_load_exact',
        's_fit_load_direct', 's_fit_load_exact',
        'oracle_direct_other_flips', 'actual_direct_other_flips',
        'K_positive', 'n_alive', 'kappa', 'cos2',
        'accuracy', 'mean_p_true', 'true_entropy', 'KL_true_pred',
        'input_rank', 'input_centered_rank', 'down_rank', 'down_cos2']
    lines = ['# H1 audit: descriptive ranges', '',
        'Matched current-task probe only. Early task 1–9, middle 39–50, late 89–100. Each cell is the range of 10 seed window medians.',
        'Positive S means sinking. `self` is the own-unit loading source; `diag` is m=n in the input-label double sum. They are different indices.',
        '`oracle` uses sampled old one-hot labels as the fixed error signal. `truth` uses the true conditional distribution. Neither changes the forward computation or parameters.', '']
    ranges = []
    for arm in ['C3', 'ReLUcap']:
        lines += [f'## {arm}', '', '| metric | early | middle | late |', '|---|---:|---:|---:|']
        for field in fields:
            cells = []
            for win in ['early', 'middle', 'late']:
                vals = [index[arm, s, win][field] for s in range(10)]
                lo, hi = min(vals), max(vals)
                cells.append(f'{lo:.6g}–{hi:.6g}')
                ranges.append(dict(arm=arm, window=win, metric=field, min=lo, max=hi))
            lines.append(f'| {field} | '+' | '.join(cells)+' |')
        lines += ['', '### Seed-specific early → late', '',
            '| seed | actual exact | oracle exact | true exact | oracle other direct | oracle unitrest direct | off self share |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for seed in range(10):
            e, l = index[arm, seed, 'early'], index[arm, seed, 'late']
            fs = ['s_actual_load_exact', 's_oracle_load_exact', 's_truth_load_exact', 's_oracle_other_direct', 's_oracle_unitrest_direct', 'share_off_self_direct']
            lines.append(f'| {seed} | '+' | '.join(f'{e[f]:.4f} → {l[f]:.4f}' for f in fs)+' |')
        lines += ['']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    with (OUT/'ranges.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(ranges[0])); w.writeheader(); w.writerows(ranges)
    print(json.dumps(judge, indent=2))


if __name__ == '__main__': main()

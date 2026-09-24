"""Recompute the user-requested late-window normalizations, without training."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parent


def read(name):
    with (ROOT / name).open() as f:
        return list(csv.DictReader(f))


cov = {(r['geometry'], int(r['seed'])): r for r in read('covstats.csv')}
values = {}
residuals = []
for row in read('transitions.csv'):
    if not 41 <= int(row['task']) <= 50:
        continue
    geometry, optimizer, seed = row['geometry'], row['optimizer'], int(row['seed'])
    c = cov[geometry, seed]
    d, trace = float(c['dimension']), float(c['cov_trace'])
    R, rms = float(row['R']), float(row['raw_W_RMS'])
    factor = R / (rms * math.sqrt(100 * trace))
    residuals.append(abs(R * R - rms * rms * 100 * trace * factor * factor))
    for field, value in {
        'R': R,
        'R_div_sqrt_d': R / math.sqrt(d),
        'R_div_sqrt_input_trace': R / math.sqrt(trace),
        'raw_W_RMS': rms,
        'width_alignment_factor': factor,
    }.items():
        values.setdefault((geometry, optimizer, seed, field), []).append(value)

paired = []
for optimizer in ['adam', 'sgd']:
    for seed in range(300, 305):
        for field in ['R', 'R_div_sqrt_d', 'R_div_sqrt_input_trace', 'raw_W_RMS', 'width_alignment_factor']:
            a = values['rgb32', optimizer, seed, field]
            b = values['avg16', optimizer, seed, field]
            assert len(a) == len(b) == 10
            av, bv = statistics.median(a), statistics.median(b)
            paired.append(dict(optimizer=optimizer, seed=seed, window='late41-50', field=field,
                               rgb32_value=av, avg16_value=bv, paired_ratio=bv / av))

# Check the previously reported CSV instead of silently replacing it.
old = read('width_normalization_posthoc.csv')
index = {(r['optimizer'], int(r['seed']), r['field']): r for r in old}
maximum_error = 0.0
for row in paired:
    prior = index[row['optimizer'], row['seed'], row['field']]
    for field in ['rgb32_value', 'avg16_value', 'paired_ratio']:
        error = abs(row[field] - float(prior[field]))
        maximum_error = max(maximum_error, error)
        assert math.isclose(row[field], float(prior[field]), rel_tol=2e-14, abs_tol=2e-14)

groups = read('width_normalization_posthoc_groups.csv')
for group in groups:
    ratios = [r['paired_ratio'] for r in paired if r['optimizer'] == group['optimizer'] and r['field'] == group['field']]
    assert len(ratios) == 5
    for field, v in [('median', statistics.median(ratios)), ('min', min(ratios)), ('max', max(ratios))]:
        assert math.isclose(v, float(group[field]), rel_tol=2e-14, abs_tol=2e-14)

audit = dict(status='PASS', maximum_reproduction_abs_error=maximum_error,
             maximum_factorization_abs_residual=max(residuals), tasks_per_series=10,
             paired_rows=len(paired), group_rows=len(groups),
             inputs={n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in [
                 'transitions.csv', 'covstats.csv', 'width_normalization_posthoc.csv',
                 'width_normalization_posthoc_groups.csv', 'width_normalization_posthoc_definitions.json']},
             script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(ROOT/'width_normalization_posthoc_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
print(json.dumps(audit, indent=2))

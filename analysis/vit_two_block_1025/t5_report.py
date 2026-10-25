"""Summarize only completed T5 cases, preserving replica Monte Carlo uncertainty."""
import csv
import json
from pathlib import Path
import common as T
from t5_launch import completed
np = T.np


def summarize(path):
    assert completed(path)
    info = json.loads((path / 'complete.json').read_text())['config']
    rows = []
    with np.load(path / 'measurements.npz') as d:
        for step in info['points'][1:]:
            valid = (d['s0_valid'] & d[f's{step}_valid']).all(0)
            row = {k: info[k] for k in ('seed', 'task', 'start', 'layer', 'mode', 'replicas')}
            row.update(step=step, common_valid_units=int(valid.sum()), directory=str(path))
            for field in ('mu', 'r'):
                delta = d[f's{step}_{field}'] - d[f's0_{field}']
                if field == 'r':
                    delta = delta[:, valid]
                delta = delta.mean(1)
                full, reps = delta[0], delta[1:]
                row.update({f'{field}_full_delta': float(full), f'{field}_replica_delta': float(reps.mean()),
                    f'{field}_noise_effect': float(reps.mean() - full),
                    f'{field}_noise_effect_se': float(reps.std(ddof=1) / np.sqrt(len(reps))),
                    f'{field}_replica_negative_fraction': float((reps < 0).mean())})
            for field in ('total', 'mean_field', 'gate_covariance', 'covariance_half0', 'covariance_half1',
                          'gate_variance', 'activation_gate_covariance'):
                row[f'next_fixed_{field}'] = float(d[f's{step}_{field}'].mean())
            total = row['next_fixed_total']
            cov = row['next_fixed_gate_covariance']
            mf = row['next_fixed_mean_field']
            row['next_fixed_gate_signed_ratio'] = cov / total if abs(total) > .1 * (abs(cov) + abs(mf)) else None
            row['ce_full'] = float(d[f's{step}_ce'][0])
            row['ce_replica'] = float(d[f's{step}_ce'][1:].mean())
            rows.append(row)
    return rows


def main():
    rows = []
    for marker in sorted((T.ARCHIVE / 't5').glob('*/complete.json')):
        rows.extend(summarize(marker.parent))
    out = T.ROOT / 'results/vit_two_block_1025'
    out.mkdir(exist_ok=True)
    with (out / 't5_completed_cases.csv').open('w', newline='') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    case_count = len({r['directory'] for r in rows})
    text = ['# T5複製実験の途中結果', '',
        f'完了した条件のみ: {case_count}/80。各条件32複製とfull-batch。未完了条件を0として集計しない。', '',
        '入力とfc1以外を固定した局所介入。μ・rはunit平均、標本抽出ありはさらに32複製で平均する。',
        '雑音効果は「複製平均−full-batch」、SEは固定した1初期状態での複製間Monte Carlo標準誤差。seed間の再現性ではない。',
        '次のゲート共分散項は共通固定計量での期待Δμであり、有限50更新の寄与率でも、実Adamの変位分解でもない。', '',
        '| seed | 課題 | 起点 | block | optimizer | full Δμ | 複製 Δμ | 雑音効果 ± SE | 次の共分散項 | 次の全体 |',
        '|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['step'] == 50:
            text.append(f"| {r['seed']} | {r['task']} | {r['start']} | {r['layer']} | {r['mode']} | "
                f"{r['mu_full_delta']:.7g} | {r['mu_replica_delta']:.7g} | {r['mu_noise_effect']:.3g} ± {r['mu_noise_effect_se']:.3g} | "
                f"{r['next_fixed_gate_covariance']:.4g} | {r['next_fixed_total']:.4g} |")
    text += ['', '全時点とrの変化は[t5_completed_cases.csv](t5_completed_cases.csv)。',
             '解析コード・初期checkpoint・生の複製・検算は各directoryのprovenance.json、manifest.json、complete.jsonから追跡できる。', '']
    (out / 't5_partial.md').write_text('\n'.join(text))
    print(json.dumps(dict(completed_cases=case_count, output=str(out / 't5_partial.md'))))


if __name__ == '__main__':
    main()

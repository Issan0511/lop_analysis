"""Audit corrected evaluation against preserved strict-gate measurements."""
import json
from pathlib import Path
import pandas as pd
import common as T
from t5_launch import completed
from t5_compat import legacy_sources
np = T.np


def main():
    rows, inputs, legacy_cases = [], [], set()
    corrected_count = 0
    for root in sorted((T.ARCHIVE / 't5').glob('seed*')):
        old = root / 'before_clamp_boundary_fix'
        marker = root / 'complete.json'
        if (old / 'measurements.npz').exists():
            legacy_cases.add(root.name)
        elif marker.exists() and legacy_sources(json.loads(marker.read_text())['config']['source_hashes']):
            legacy_cases.add(root.name)
        if not completed(root):
            continue
        corrected_count += 1
        if not (old / 'measurements.npz').exists():
            continue
        prior_manifest = json.loads((old / 'manifest.json').read_text())
        current_manifest = json.loads((root / 'manifest.json').read_text())
        before = {x['path']: x['sha256'] for x in prior_manifest if x['path'].endswith('.pt')}
        after = {x['path']: x['sha256'] for x in current_manifest if x['path'].endswith('.pt')}
        assert before == after and len(before) == 33, root
        cfg = json.loads(marker.read_text())['config']
        with np.load(old / 'measurements.npz') as prior, np.load(root / 'measurements.npz') as current:
            for step in cfg['points']:
                for field in ('mu', 'sd', 'r', 'valid', 'opened', 'ce', 'accuracy', 'direct'):
                    key = f's{step}_{field}'
                    assert np.array_equal(prior[key], current[key]), (root, key)
                for field in ('total', 'mean_field', 'gate_covariance'):
                    assert np.array_equal(prior[f's{step}_{field}'], current[f's{step}_positive_{field}']), (root, step, field)
                boundary = current[f's{step}_zero_boundary']
                error = current[f's{step}_total'] - prior[f's{step}_total'] - boundary
                assert np.abs(error).max() < 1e-11
                row = {k: cfg[k] for k in ('seed', 'task', 'start', 'layer', 'mode')}
                row.update(step=step, raw_replicas_unchanged=True, observables_bit_equal=True,
                    strict_gate_ledger_bit_equal=True,
                    old_covariance=float(prior[f's{step}_gate_covariance'].mean()),
                    corrected_covariance=float(current[f's{step}_gate_covariance'].mean()),
                    boundary_mean=float(boundary.mean()), boundary_max_abs=float(np.abs(boundary).max()),
                    boundary_identity_error=float(np.abs(error).max()),
                    derivative_zero_fraction=float(current[f's{step}_zero_fraction'].mean()))
                rows.append(row)
        inputs.append(dict(case=root.name, raw_replicas_unchanged=33,
            original_arrays=str(old / 'measurements.npz'), original_sha256=T.sha256(old / 'measurements.npz'),
            corrected_arrays=str(root / 'measurements.npz'), corrected_sha256=T.sha256(root / 'measurements.npz')))
    output = T.ROOT / 'results/vit_two_block_1025'
    pd.DataFrame(rows).to_csv(output / 't5_correction_comparison.csv', index=False)
    report = dict(corrected_completed=corrected_count, legacy_conditions=len(legacy_cases),
        legacy_revalidated=len(inputs), source_sha256=T.sha256(Path(__file__)), inputs=inputs)
    T.atomic_json(output / 't5_correction_inputs.json', report)
    lines = ['# 境界訂正前後の一致を確認', '',
        f'訂正版の完了は{corrected_count}/80条件。旧測定が完了していた{len(legacy_cases)}条件のうち、{len(inputs)}条件の再評価を照合した。', '',
        '再評価済みの各条件で次を確認した:', '',
        '- full-batchと32複製、計33個の学習記録のSHAが不変。',
        '- 全観測時点のμ・σ・μ/σ・開口率・CE・精度・直接逆伝播の配列が旧測定とbit一致。',
        '- 訂正版に併記した厳密な開口gate 1[z>0]の台帳は、旧測定の合計・平均場項・共分散項とbit一致。',
        '- 訂正版の実装gate 1[z>=0]による合計と旧合計の差は、z=0の境界項に一致。', '',
        'これは再評価済み条件の検算であり、未再評価条件を完了へ数えない。全時点の差はt5_correction_comparison.csv。', '',
        '旧台帳の恒等式が壊れていたのではなく、元モデルが採るz=0の微分と異なっていた。旧台帳も削除せず、実装gateの台帳と区別して保存する。', '']
    (output / 't5_correction_comparison.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in report.items() if k != 'inputs'}))


if __name__ == '__main__':
    main()

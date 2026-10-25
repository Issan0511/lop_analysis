"""Summarize only completed T5 cases, preserving replica Monte Carlo uncertainty."""
import csv
import json
from pathlib import Path
import common as T
from t5_launch import completed
np = T.np
FINITE_FIELDS = ('mu', 'r', 'r_mean_component', 'r_width_component')


def seed_groups(rows):
    """Keep seed replication distinct from replicas sharing one saved state."""
    keys = ('task', 'start', 'layer', 'mode', 'step')
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    results = []
    for values, group in sorted(groups.items()):
        seeds = [r['seed'] for r in group]
        assert len(set(seeds)) == len(seeds), (values, seeds)
        assert set(seeds) <= set(T.SEEDS)
        row = dict(zip(keys, values))
        row.update(seeds=len(seeds), complete_five_seeds=set(seeds) == set(T.SEEDS))
        for field in FINITE_FIELDS:
            full, replicas, noise = [np.array([r[f'{field}_{suffix}'] for r in group])
                                    for suffix in ('full_delta', 'replica_delta', 'noise_effect')]
            assert np.max(np.abs(replicas - full - noise)) < 1e-12
            row.update({f'{field}_full_mean': float(full.mean()),
                f'{field}_replica_mean': float(replicas.mean()),
                f'{field}_noise_mean': float(noise.mean()),
                f'{field}_noise_between_seed_se': float(noise.std(ddof=1) / np.sqrt(len(noise))) if len(noise) > 1 else None,
                f'{field}_noise_mc_se_of_mean': float(np.sqrt(sum(r[f'{field}_noise_effect_se']**2 for r in group)) / len(group)),
                f'{field}_full_down_seeds': int((full < 0).sum()),
                f'{field}_replica_down_seeds': int((replicas < 0).sum()),
                f'{field}_noise_down_seeds': int((noise < 0).sum())})
        arrays = {name: np.array([r[f'next_fixed_{name}'] for r in group]) for name in
                  ('total', 'mean_field', 'gate_covariance', 'covariance_half0', 'covariance_half1')}
        assert np.max(np.abs(arrays['total'] - arrays['mean_field'] - arrays['gate_covariance'])) < 1e-11
        for name, array in arrays.items():
            row[f'next_fixed_{name}_mean'] = float(array.mean())
            row[f'next_fixed_{name}_down_seeds'] = int((array < 0).sum())
        row['next_fixed_covariance_both_halves_down_seeds'] = int(
            ((arrays['covariance_half0'] < 0) & (arrays['covariance_half1'] < 0)).sum())
        results.append(row)
    return results


def summarize(path):
    assert completed(path)
    info = json.loads((path / 'complete.json').read_text())['config']
    rows = []
    with np.load(path / 'measurements.npz') as d:
        for step in info['points'][1:]:
            valid = (d['s0_valid'] & d[f's{step}_valid']).all(0)
            assert valid.any()
            row = {k: info[k] for k in ('seed', 'task', 'start', 'layer', 'mode', 'replicas')}
            row.update(step=step, common_valid_units=int(valid.sum()), directory=str(path))
            mu0, mu1 = d['s0_mu'][:, valid], d[f's{step}_mu'][:, valid]
            q0, q1 = 1 / (d['s0_sd'][:, valid] + 1e-12), 1 / (d[f's{step}_sd'][:, valid] + 1e-12)
            mean_component = (mu1 - mu0) * (q0 + q1) / 2
            width_component = (mu0 + mu1) * (q1 - q0) / 2
            r_delta = (d[f's{step}_r'] - d['s0_r'])[:, valid]
            error = float(np.abs(mean_component + width_component - r_delta).max())
            assert error < 1e-11, (path, step, error)
            row['r_endpoint_identity_error'] = error
            finite = dict(mu=(d[f's{step}_mu'] - d['s0_mu']).mean(1), r=r_delta.mean(1),
                r_mean_component=mean_component.mean(1), r_width_component=width_component.mean(1))
            for field, delta in finite.items():
                full, reps = delta[0], delta[1:]
                row.update({f'{field}_full_delta': float(full), f'{field}_replica_delta': float(reps.mean()),
                    f'{field}_noise_effect': float(reps.mean() - full),
                    f'{field}_noise_effect_se': float(reps.std(ddof=1) / np.sqrt(len(reps))),
                    f'{field}_replica_negative_fraction': float((reps < 0).mean())})
            for field in ('total', 'mean_field', 'gate_covariance', 'covariance_half0', 'covariance_half1',
                          'gate_variance', 'activation_gate_covariance', 'positive_total', 'positive_mean_field',
                          'positive_gate_covariance', 'zero_boundary', 'zero_fraction'):
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
        if not completed(marker.parent):
            continue
        rows.extend(summarize(marker.parent))
    out = T.ROOT / 'results/vit_two_block_1025'
    out.mkdir(exist_ok=True)
    with (out / 't5_completed_cases.csv').open('w', newline='') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    groups = seed_groups(rows)
    with (out / 't5_seed_groups.csv').open('w', newline='') as stream:
        if groups:
            writer = csv.DictWriter(stream, fieldnames=list(groups[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(groups)
    case_count = len({r['directory'] for r in rows})
    text = ['# T5複製実験の途中結果', '',
        f'完了した条件のみ: {case_count}/80。各条件32複製とfull-batch。未完了条件を0として集計しない。', '',
        '入力とfc1以外を固定した局所介入。μ・rはunit平均、標本抽出ありはさらに32複製で平均する。',
        '雑音効果は「複製平均−full-batch」、SEは固定した1初期状態での複製間Monte Carlo標準誤差。seed間の再現性ではない。',
        '次のゲート共分散項は共通固定計量での期待Δμであり、有限50更新の寄与率でも、実Adamの変位分解でもない。', '',
        '境界微分の訂正版。主台帳は元モデルのclampの微分1[z>=0]を使う。厳密な開口1[z>0]の台帳と、z=0だけの境界項もCSVへ保存した。旧測定は各条件のbefore_clamp_boundary_fix/に保全している。', '',
        '| seed | 課題 | 起点 | block | optimizer | full Δμ | 複製 Δμ | 雑音効果 ± SE | 次の共分散項 | 次の全体 |',
        '|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['step'] == 50:
            text.append(f"| {r['seed']} | {r['task']} | {r['start']} | {r['layer']} | {r['mode']} | "
                f"{r['mu_full_delta']:.7g} | {r['mu_replica_delta']:.7g} | {r['mu_noise_effect']:.3g} ± {r['mu_noise_effect_se']:.3g} | "
                f"{r['next_fixed_gate_covariance']:.4g} | {r['next_fixed_total']:.4g} |")
    text += ['', '## 全5seedの再現性と複製分割', '',
        '次は同じ課題・起点・block・optimizerで全5seedが揃った50更新後だけ。共分散項はunit平均→seed平均。',
        '分割欄は前半16複製・後半16複製の共分散項のunit平均が、両方とも下向きだったseed数。',
        'これは符号の安定性の記述であり、有意差検定ではない。実Adam条件の共分散項も固定計量による診断である。', '',
        '| 課題 | 起点 | block | optimizer | 次の共分散項平均 | 下向きseed | 両分割とも下向き | 次の全体平均 |',
        '|---:|---:|---:|---|---:|---:|---:|---:|']
    for r in groups:
        if r['step'] == 50 and r['complete_five_seeds']:
            text.append(f"| {r['task']} | {r['start']} | {r['layer']} | {r['mode']} | "
                f"{r['next_fixed_gate_covariance_mean']:.6g} | {r['next_fixed_gate_covariance_down_seeds']}/5 | "
                f"{r['next_fixed_covariance_both_halves_down_seeds']}/5 | {r['next_fixed_total_mean']:.6g} |")
    text += ['', '有限変位についての5seed平均とseed間SEは[t5_seed_groups.csv](t5_seed_groups.csv)。',
        'そのSEはseed別効果の標準偏差/√5であり、固定した1状態での複製間Monte Carlo SEとは区別する。',
        '部分的なseed集合もCSVには残すが、complete_five_seeds=falseの群を5seedの結果に数えない。', '']
    text += ['## μの押しとμ/σの変位を区別する', '',
        'q=1/(σ+1e−12)として、各unit・複製の端点でΔr=Δμ(q₀+q₁)/2+(μ₀+μ₁)Δq/2を検算した。',
        '以下はその平均項と幅項をunit平均→複製平均→5seed平均した値。両項はΔrへ加算できるが、原因別の因果寄与率ではない。',
        'rは全複製の始点・終点で共通にvalidなunitを使う。生のμの平均は全unitであり、母数の違いもCSVで記録する。', '',
        '| 課題 | 起点 | block | optimizer | full 平均項 | full 幅項 | 抽出あり 平均項 | 抽出あり 幅項 |',
        '|---:|---:|---:|---|---:|---:|---:|---:|']
    for r in groups:
        if r['step'] == 50 and r['complete_five_seeds']:
            for arm in ('full', 'replica'):
                assert abs(r[f'r_mean_component_{arm}_mean'] + r[f'r_width_component_{arm}_mean'] - r[f'r_{arm}_mean']) < 1e-11
            text.append(f"| {r['task']} | {r['start']} | {r['layer']} | {r['mode']} | "
                f"{r['r_mean_component_full_mean']:.6g} | {r['r_width_component_full_mean']:.6g} | "
                f"{r['r_mean_component_replica_mean']:.6g} | {r['r_width_component_replica_mean']:.6g} |")
    text += ['', '全時点とrの変化は[t5_completed_cases.csv](t5_completed_cases.csv)。',
             '解析コード・初期checkpoint・生の複製・検算は各directoryのprovenance.json、manifest.json、complete.jsonから追跡できる。', '']
    (out / 't5_partial.md').write_text('\n'.join(text))
    print(json.dumps(dict(completed_cases=case_count, output=str(out / 't5_partial.md'))))


if __name__ == '__main__':
    main()

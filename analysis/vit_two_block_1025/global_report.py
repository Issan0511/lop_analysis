"""Full-model noise effects; signed counterfactual ratios are not T5 attribution."""
import json
import pandas as pd
import common as T
from global_noise import hashes
np = T.np
FINITE_FIELDS = ('mu', 'r', 'r_mean_component', 'r_width_component')


def endpoint_components(data, step, layer):
    """Exact endpoint accounting on the original common-unit mask, per replica."""
    a, b = f's0_L{layer}', f's{step}_L{layer}'
    valid = (data[f'{a}_valid'] & data[f'{b}_valid']).all(0)
    assert valid.any()
    mu0, mu1 = data[f'{a}_mu'][:, valid], data[f'{b}_mu'][:, valid]
    q0, q1 = 1 / (data[f'{a}_sd'][:, valid] + 1e-12), 1 / (data[f'{b}_sd'][:, valid] + 1e-12)
    mean = (mu1 - mu0) * (q0 + q1) / 2
    width = (mu0 + mu1) * (q1 - q0) / 2
    delta = (data[f'{b}_r'] - data[f'{a}_r'])[:, valid]
    error = float(np.abs(mean + width - delta).max())
    assert error < 1e-11, (step, layer, error)
    result = dict(common_valid_units=int(valid.sum()), r_endpoint_identity_error=error)
    arrays = dict(mu=(data[f'{b}_mu'] - data[f'{a}_mu']).mean(1), r=delta.mean(1),
                  r_mean_component=mean.mean(1), r_width_component=width.mean(1))
    for field, values in arrays.items():
        full, reps = values[0], values[1:]
        result.update({f'{field}_full_delta': float(full), f'{field}_replica_delta': float(reps.mean()),
            f'{field}_noise_effect': float(reps.mean() - full),
            f'{field}_noise_effect_se': float(reps.std(ddof=1) / np.sqrt(len(reps)))})
    return result


def main():
    rows, cases = [], []
    for marker in sorted((T.ARCHIVE / 'global_noise').glob('*/complete.json')):
        root = marker.parent
        result = json.loads(marker.read_text())
        cfg = result['config']
        assert result['status'] == 'complete' and cfg['source_hashes'] == hashes()
        manifest = json.loads((root / 'manifest.json').read_text())
        # Validate the consumed measurement files. The controller checks all replica files.
        for item in manifest:
            if item['path'] in ('summary.csv', 'measurements.npz', 'input.json', 'provenance.json'):
                path = root / item['path']
                assert path.stat().st_size == item['bytes'] and T.sha256(path) == item['sha256'], path
        with T.np.load(root / 'measurements.npz') as d:
            ce0 = float(d['s0_L1_ce'][0])
            accuracy0 = float(d['s0_L1_accuracy'][0])
            components = {(step, layer): endpoint_components(d, step, layer)
                          for step in cfg['points'][1:] for layer in (1, 2)}
        for row in pd.read_csv(root / 'summary.csv').to_dict('records'):
            added = components[(row['step'], row['layer'])]
            # Reconstruct every original finite mu/r statistic independently from
            # the saved arrays before adding the new endpoint components.
            for key in added.keys() & row.keys():
                assert abs(added[key] - row[key]) < 1e-12, (root, key)
            row.update(added)
            row.update({k: cfg[k] for k in ('seed', 'task', 'start', 'replicas')})
            row.update(initial_ce=ce0, initial_accuracy=accuracy0, directory=str(root))
            for field in ('mu', 'r'):
                delta, se = row[f'{field}_replica_delta'], row[f'{field}_noise_effect_se']
                # Do not give a sinking fraction when the mean does not reliably sink.
                row[f'{field}_noise_signed_ratio'] = row[f'{field}_noise_effect'] / delta if delta < -max(1e-8, 4 * se) else None
            rows.append(row)
        cases.append(root.name)
    output = T.ROOT / 'results/vit_two_block_1025'
    output.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'global_noise_completed_cases.csv', index=False)
    groups = []
    if len(frame):
        for keys, g in frame.groupby(['task', 'start', 'layer', 'step']):
            assert g.seed.nunique() == len(g)
            row = dict(zip(('task', 'start', 'layer', 'step'), keys))
            row.update(seeds=len(g), complete_five_seeds=set(g.seed) == set(T.SEEDS),
                       initial_accuracy_min=float(g.initial_accuracy.min()),
                       initial_accuracy_median=float(g.initial_accuracy.median()),
                       initial_accuracy_max=float(g.initial_accuracy.max()))
            for field in FINITE_FIELDS:
                full, stochastic, effect = (g[f'{field}_{key}'] for key in ('full_delta', 'replica_delta', 'noise_effect'))
                assert abs(float(stochastic.mean() - full.mean() - effect.mean())) < 1e-12
                row.update({f'{field}_full_mean': float(full.mean()),
                    f'{field}_stochastic_mean': float(stochastic.mean()),
                    f'{field}_noise_mean': float(effect.mean()),
                    f'{field}_noise_between_seed_se': float(effect.std(ddof=1) / len(g)**.5) if len(g) > 1 else None,
                    f'{field}_noise_mc_se_of_mean': float((g[f'{field}_noise_effect_se']**2).sum()**.5 / len(g)),
                    f'{field}_full_sinking_seeds': int((full < 0).sum()),
                    f'{field}_stochastic_sinking_seeds': int((stochastic < 0).sum()),
                    f'{field}_noise_downward_seeds': int((effect < 0).sum())})
            groups.append(row)
    pd.DataFrame(groups).to_csv(output / 'global_noise_seed_groups.csv', index=False)
    text = ['# 網全体のAdam更新：標本雑音の総効果', '',
        f'完了 {len(cases)}/30状態。全parameterを更新し、各状態32複製とfull-batchを比較した。', '',
        '各値はunit平均の有限Δr=Δ(μ/σ)。標本抽出ありはさらに32複製で平均する。',
        '雑音効果は複製平均−full-batch。負なら新たな抽出雑音が沈降を増す。SEは同じ初期状態でのMonte Carlo標準誤差。',
        'この総効果にはゲート・上流輸送・Adamの非線形性が含まれ、全量をT5へ帰属しない。過去のAdamの履歴は両条件に残る。', '',
        '| seed | 課題 | 起点 | 初期精度 | block | full-batch Δr | 抽出あり Δr | 雑音効果 ± SE |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['step'] == 50:
            text.append(f"| {r['seed']} | {r['task']} | {r['start']} | {r['initial_accuracy']:.1%} | {r['layer']} | "
                f"{r['r_full_delta']:.6g} | {r['r_replica_delta']:.6g} | {r['r_noise_effect']:.3g} ± {r['r_noise_effect_se']:.3g} |")
    text += ['', '## 全5seedが揃った条件', '',
        '以下は50更新の変位について、unit平均→複製平均→5seed平均の順で集計した。中央値ではない。',
        '効果の±SEは5個のseed別効果の標準偏差/√5。上の表の固定状態における32複製のMonte Carlo SEとは別であり、seed・課題の違いを含む。',
        '未完了seedがある条件は、この5seed表には含めない。個々の完了値は上の表とCSVに保持する。', '',
        '| 課題 | 起点 | block | full-batch Δr平均 | 抽出あり Δr平均 | 雑音効果平均 ± seed SE | full-batchで沈むseed | 抽出ありで沈むseed |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in groups:
        if r['step'] == 50 and r['complete_five_seeds']:
            text.append(f"| {r['task']} | {r['start']} | {r['layer']} | {r['r_full_mean']:.6g} | "
                f"{r['r_stochastic_mean']:.6g} | {r['r_noise_mean']:.3g} ± {r['r_noise_between_seed_se']:.3g} | "
                f"{r['r_full_sinking_seeds']}/5 | {r['r_stochastic_sinking_seeds']}/5 |")
    text += ['', 'この集計は効果の記述であり、新しい有意差判定や成功基準を追加したものではない。', '',
        '20更新起点は切替後の学び直しを含み、適合後と同一視しない。',
        'CSVのnoise_signed_ratioは「抽出を加えた総効果/抽出ありの平均変位」。沈降が数値的に小さい場合と4SE以下では欠測にし、T5の割合とは呼ばない。',
        '負の比は雑音が沈降を緩和した場合。100%を超える比はfull-batchが逆向きに動くなど相殺のある場合で、0〜100%へ切り詰めない。', '']
    text += ['## 雑音による位置の差を平均項と幅項へ分ける', '',
        '課題60・seed2・20更新起点の差を見た後に追加した探索的解析。全状態・全時点・両blockへ同じ式を適用し、元の集計を置き換えない。',
        'q=1/(σ+1e−12)としてΔr=Δμ(q₀+q₁)/2+(μ₀+μ₁)Δq/2。前者を平均項、後者を幅項とする。',
        '全unit・全複製でこの恒等式を検算し、元の共通valid集合とμ/rの全統計を再現した。幅を固定した因果介入ではなく、両項にも端点の重みの変化が含まれる。',
        '以下は50更新・全5seedが揃った群の、抽出あり−full-batchのseed平均。SEは各項にもCSVで保存し、両項のSEを単純に足さない。', '',
        '| 課題 | 起点 | block | 雑音効果の平均項 | 雑音効果の幅項 | 雑音効果の合計Δr |',
        '|---:|---:|---:|---:|---:|---:|']
    for r in groups:
        if r['step'] == 50 and r['complete_five_seeds']:
            for arm in ('full_mean', 'stochastic_mean', 'noise_mean'):
                assert abs(r[f'r_mean_component_{arm}'] + r[f'r_width_component_{arm}'] - r[f'r_{arm}']) < 1e-11
            text.append(f"| {r['task']} | {r['start']} | {r['layer']} | {r['r_mean_component_noise_mean']:.6g} | "
                f"{r['r_width_component_noise_mean']:.6g} | {r['r_noise_mean']:.6g} |")
    text += ['', '各条件・各時点・両腕の成分と恒等式の最大誤差はglobal_noise_completed_cases.csv、seed間集計はglobal_noise_seed_groups.csv。', '']
    (output / 'global_noise_partial.md').write_text('\n'.join(text))
    print(json.dumps(dict(completed_states=len(cases), output=str(output / 'global_noise_partial.md'))))


if __name__ == '__main__':
    main()

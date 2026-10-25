"""Full-model noise effects; signed counterfactual ratios are not T5 attribution."""
import json
import pandas as pd
import common as T
from global_noise import hashes


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
        for row in pd.read_csv(root / 'summary.csv').to_dict('records'):
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
            for field in ('mu', 'r'):
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
    (output / 'global_noise_partial.md').write_text('\n'.join(text))
    print(json.dumps(dict(completed_states=len(cases), output=str(output / 'global_noise_partial.md'))))


if __name__ == '__main__':
    main()

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
    pd.DataFrame(rows).to_csv(output / 'global_noise_completed_cases.csv', index=False)
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
    text += ['', '20更新起点は切替後の学び直しを含み、適合後と同一視しない。',
        'CSVのnoise_signed_ratioは「抽出を加えた総効果/抽出ありの平均変位」。沈降が数値的に小さい場合と4SE以下では欠測にし、T5の割合とは呼ばない。',
        '負の比は雑音が沈降を緩和した場合。100%を超える比はfull-batchが逆向きに動くなど相殺のある場合で、0〜100%へ切り詰めない。', '']
    (output / 'global_noise_partial.md').write_text('\n'.join(text))
    print(json.dumps(dict(completed_states=len(cases), output=str(output / 'global_noise_partial.md'))))


if __name__ == '__main__':
    main()

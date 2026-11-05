"""Aggregate complete one-step diagnostics without pooling units as seeds."""
import json
from pathlib import Path
import pandas as pd
import common as T


def main():
    output = T.ROOT / 'results/vit_two_block_1025'
    parts, manifests = [], []
    for path in sorted((T.ARCHIVE/'ordinary_drive').glob('*/complete.json')):
        info = json.loads(path.read_text())
        assert info['status'] == 'complete'
        assert T.sha256(path.parent/'arrays.npz') == info['arrays_sha256']
        parts.append(pd.read_csv(path.parent/'rows.csv'))
        manifests.append(dict(path=str(path), sha256=T.sha256(path), checks=info['checks']))
    data = pd.concat(parts, ignore_index=True)
    data.to_csv(output/'ordinary_drive_cases.csv', index=False)
    rows = []
    for keys, g in data.groupby(['task','start','layer','arm','mode','kind']):
        row = dict(zip(['task','start','layer','arm','mode','kind'], keys))
        row.update(seeds=len(g), r_mean=float(g.r_mean.mean()), r_down=int((g.r_mean<0).sum()),
                   mu_mean=float(g.mu_mean.mean()), mu_down=int((g.mu_mean<0).sum()))
        if keys[-1] == 'finite':
            row.update(ce_delta_mean=float(g.ce_delta.mean()), ce_down=int((g.ce_delta<0).sum()))
        rows.append(row)
    groups = pd.DataFrame(rows)
    groups.to_csv(output/'ordinary_drive_groups.csv', index=False)
    T.atomic_json(output/'ordinary_drive_inputs.json', dict(cases=len(parts), inputs=manifests,
        report_sha256=T.sha256(Path(__file__))))
    lines = ['# 通常学習の逆伝播ゲート診断', '', f'完了 {len(parts)}/40 状態。事後の問いから固定した全比較。', '',
             '入力とfc1以外は固定。全画像のCEを用いた一歩。起点のforwardは全腕同じ。',
             'all_openは逆伝播ゲートを1、rate_gateはtoken/unit別の開口率、no_image_pairingは画像間で平均した活性化への誤差。',
             '_normは通常腕とfc1勾配のnormを揃えたもの。画像間の対応の介入であってT5除去ではない。', '',
             '固定計量は保存Adam分母を使いmomentumなし。Adamは保存履歴を保つ。有限一歩のΔrとCE。数値はunit平均→5seed平均。', '',
             '| 課題 | 起点 | block | optimizer | 腕 | seed数 | Δr | 沈降seed | ΔCE | CE低下seed |',
             '|---:|---:|---:|---|---|---:|---:|---:|---:|---:|']
    for row in groups[(groups.kind=='finite') & groups.arm.isin(['relu','all_open_norm','rate_gate_norm','no_image_pairing_norm'])].to_dict('records'):
        lines.append(f"| {row['task']} | {row['start']} | {row['layer']} | {row['mode']} | {row['arm']} | {row['seeds']} | {row['r_mean']:.8g} | {row['r_down']} | {row['ce_delta_mean']:.8g} | {int(row['ce_down'])} |")
    lines += ['', '## 限界', '',
              '- 逆伝播を変えた一歩の診断。長期に沈降を除いた実験ではなく、学習を維持できる保証もない。',
              '- 正負の要求の分解、画像対応の差は加法的な帳簿。唯一の因果寄与率にはしない。',
              '- μとμ/σは別列。深いblockのμ低下とr上昇は両立しうる。',
              '- チェック値・各状態はCSVとinputs.json、全unitの値と勾配はarchive。']
    (output/'ordinary_drive_summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(cases=len(parts), groups=len(groups))))


if __name__ == '__main__':
    main()

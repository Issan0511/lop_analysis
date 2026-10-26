"""switch25_units.py の結果の表（Claude、1026）: 切替後 25 更新で沈んだ unit の割合と、沈みの自分の重みの分と入力の分。

    python analysis/vit_long_1024/posthoc_2blk/switch25_table.py <JSON の置き場> <表の置き場>
"""
import json
import sys
from pathlib import Path


def main(src, out):
    lines = ['切替後 25 更新での最終 FFN の unit ごとの CLS 位置の平均前活性の変化 Δz̄_j（probe は各クラス先頭 25 枚）。'
             '「自分の重み」は Δw_j·ȳ + Δb_j、「入力」は w_j·Δȳ（上流の変化）。rms は生きた unit の二乗平均平方根。', '',
             '| 腕 | 切替 | 沈んだ unit の割合（全体 / 自分の重み / 入力） | Δz̄ の中央値（全体 / 自分 / 入力） | rms（全体 / 自分 / 入力） | ȳ の移動 ÷ ‖ȳ‖ | S_load の符号との一致（自分 / 全体） |',
             '|---|---|---|---|---|---|---|']
    for arm in ('SL', 'R0L', 'S', 'R0'):
        p = src / f'switch25_{arm}.json'
        if not p.exists():
            continue
        for r in json.loads(p.read_text())['results']:
            lines.append(f"| {arm} | {r['task']}→{r['task'] + 1} | {r['frac_total_sinks']:.2f} / {r['frac_own_sinks']:.2f} / {r['frac_input_sinks']:.2f}"
                         f" | {r['median_dz']:+.2f} / {r['median_d_own']:+.3f} / {r['median_d_input']:+.2f}"
                         f" | {r['rms_dz']:.2f} / {r['rms_d_own']:.2f} / {r['rms_d_input']:.2f} | {r['ybar_shift_norm'] / r['ybar_norm']:.2f}"
                         f" | {r.get('agree_S_own', float('nan')):.2f} / {r.get('agree_S_total', float('nan')):.2f} |")
    text = '\n'.join(lines)
    (out / 'switch25_table.md').write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))

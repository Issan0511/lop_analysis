"""Label-removal signal at a matched state, distinct from head rotation or T5.

Post-hoc analysis of saved probes. The gradient identity is algebra, whereas
the layer signs and finite Adam contrasts are measured properties of the ViT.
"""
import json
from pathlib import Path
import pandas as pd
import common as T
from alignment import hashes
from alignment_report import pair
np = T.np


def main():
    rows, overlaps, finite, inputs = [], [], [], []
    for marker in sorted((T.ARCHIVE / 'alignment').glob('*/complete.json')):
        root = marker.parent
        cfg = json.loads(marker.read_text())['config']
        assert cfg['source_hashes'] == hashes()
        for item in json.loads((root / 'manifest.json').read_text()):
            path = root / item['path']
            assert path.stat().st_size == item['bytes'] and T.sha256(path) == item['sha256'], path
        context = {k: cfg[k] for k in ('seed', 'task', 'start')}
        inputs.append(dict(path=str(root / 'arrays.npz'), sha256=T.sha256(root / 'arrays.npz')))
        with np.load(root / 'arrays.npz') as d:
            for layer in (1, 2):
                valid = d[f'initial_L{layer}_valid']
                for metric in ('SGD', 'Adam_fixed'):
                    for route in ('full', 'local', 'upstream'):
                        values = {}
                        for moment in ('mu', 'sd', 'r'):
                            def get(signal):
                                full = d[f'old_{metric}_full_{signal}_{moment}'][layer - 1]
                                local = d[f'old_{metric}_local_{signal}_{moment}'][layer - 1]
                                return dict(full=full, local=local, upstream=full - local)[route]
                            switch, fit = get('switch'), get('fit')
                            target = switch - fit
                            values[moment] = dict(switch=switch, fit=fit, target=target)
                            rows.append(dict(**context, layer=layer, metric=metric, route=route, moment=moment,
                                valid_units=int(valid.sum()), switch=float(switch[valid].mean()),
                                fit=float(fit[valid].mean()), target=float(target[valid].mean()),
                                target_negative_units=float((target[valid] < 0).mean())))
                        mu, sd = d[f'initial_L{layer}_mu'], d[f'initial_L{layer}_sd']
                        predicted_r = values['mu']['target'] / (sd + 1e-12) - mu * values['sd']['target'] / (sd + 1e-12)**2
                        assert np.max(np.abs((predicted_r - values['r']['target'])[valid])) < 1e-8
                        # Reuse numerical-zero handling from the original overlap report.
                        stats = pair(values['r']['target'], values['r']['fit'], valid)
                        stats = {k.replace('load', 'target'): v for k, v in stats.items()}
                        overlaps.append(dict(**context, layer=layer, metric=metric, route=route, **stats))
                for moment in ('mu', 'sd', 'r'):
                    switch = d[f'old_finite_switch_L{layer}_{moment}']
                    fit = d[f'old_finite_fit_L{layer}_{moment}']
                    finite.append(dict(**context, layer=layer, moment=moment,
                        switch=float(switch[valid].mean()), fit=float(fit[valid].mean()),
                        switch_minus_fit=float((switch[valid] - fit[valid]).mean())))
    output = T.ROOT / 'results/vit_two_block_1025'
    for name, data in [('target_signal_channels', rows), ('target_signal_overlap', overlaps), ('target_signal_finite', finite)]:
        pd.DataFrame(data).to_csv(output / f'{name}.csv', index=False)
    T.atomic_json(output / 'target_signal_inputs.json', dict(source_sha256=T.sha256(Path(__file__)), inputs=inputs))
    text = ['# 旧正解信号を外すことと、各blockの位置変化', '',
        f'完了した整列測定{len(inputs)}/20状態の保存配列を使った事後解析。新たな学習や介入はしていない。', '',
        '同じ画像、同じモデル、同じ読み出し、同じoptimizer状態で比較する。旧正解q、予測p、一様分布uに対し、',
        '`e_switch=p−u=(p−q)+(q−u)=e_fit+e_target`。',
        '固定計量での一次Δμ・Δσ・Δrは勾配に線形なので、targetをswitch−fitとして求める。恒等式そのものは実証ではなく、層ごとの符号と大きさが測定結果である。',
        'targetは旧正解を支えていた信号を外す効果。最終headの射影成分load、T5、あるいはfc2の整列だけを除く介入とは同一ではない。', '',
        '以下は各unitの平均を取り、さらにseed中央値。別々の中央値の和は一致するとは限らない。', '',
        '| 課題 | 起点 | block | 経路 | n seed | switch Δr | fit Δr | target Δr | targetが沈むseed |',
        '|---:|---:|---:|---|---:|---:|---:|---:|---:|']
    frame = pd.DataFrame(rows)
    if len(frame):
        subset = frame[(frame.metric == 'Adam_fixed') & (frame.moment == 'r')]
        for keys, g in subset.groupby(['task', 'start', 'layer', 'route']):
            task, start, layer, route = keys
            text.append(f'| {task} | {start} | {layer} | {route} | {len(g)}/5 | {g.switch.median():.6g} | '
                        f'{g.fit.median():.6g} | {g.target.median():.6g} | {int((g.target < 0).sum())}/{len(g)} |')
    text += ['', '実Adamの1歩は非線形なのでtargetという加法成分にはしない。同じ起点からswitch損失またはfit損失で1歩進めた、対照間の実変位差を示す。', '',
             '| 課題 | 起点 | block | n seed | 実switch Δr中央値 | 実fit Δr中央値 | switch−fit中央値 | 差が下向きのseed |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|']
    ff = pd.DataFrame(finite)
    if len(ff):
        for keys, g in ff[ff.moment == 'r'].groupby(['task', 'start', 'layer']):
            task, start, layer = keys
            text.append(f'| {task} | {start} | {layer} | {len(g)}/5 | {g.switch.median():.6g} | {g.fit.median():.6g} | '
                        f'{g.switch_minus_fit.median():.6g} | {int((g.switch_minus_fit < 0).sum())}/{len(g)} |')
    text += ['', '全seed・両計量・μとσの値はtarget_signal_channels.csv、unitごとの対応はtarget_signal_overlap.csv、有限Adamはtarget_signal_finite.csv。',
             'この比較は同じ画像上でのラベル切替の期待値。実際の新しい画像・有限個の新ラベルでの学習全期間を、そのまま置き換えるものではない。', '']
    (output / 'target_signal_partial.md').write_text('\n'.join(text))
    print(json.dumps(dict(completed_states=len(inputs), output=str(output / 'target_signal_partial.md'))))


if __name__ == '__main__':
    main()

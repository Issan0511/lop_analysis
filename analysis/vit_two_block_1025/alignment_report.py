"""Completed alignment probes: seed-level intervention effects and unit overlap."""
import json
import pandas as pd
import common as T
from alignment import hashes
np = T.np


def pair(a, b, valid):
    ta = max(1e-12, 1e-6 * float(np.abs(a).max()))
    tb = max(1e-12, 1e-6 * float(np.abs(b).max()))
    keep = valid & (np.abs(a) > ta) & (np.abs(b) > tb)
    a, b = a[keep], b[keep]
    if not len(a):
        return dict(n=0, excluded=int(len(valid)))
    sa, sb = a < 0, b < 0
    expected = sa.mean() * (1 - sb.mean()) + (1 - sa.mean()) * sb.mean()
    return dict(n=len(a), excluded=int(len(valid) - len(a)), load_sink_fraction=float(sa.mean()),
                fit_sink_fraction=float(sb.mean()), both_sink=float((sa & sb).mean()),
                load_only=float((sa & ~sb).mean()), fit_only=float((~sa & sb).mean()),
                both_float=float((~sa & ~sb).mean()), opposite_fraction=float((sa != sb).mean()),
                independent_opposite=float(expected), excess_opposite=float((sa != sb).mean() - expected),
                cosine=float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))),
                spearman=float(pd.Series(a).rank().corr(pd.Series(b).rank())))


def main():
    effects, overlaps, channels, terms, doses = [], [], [], [], []
    count = 0
    for marker in sorted((T.ARCHIVE / 'alignment').glob('*/complete.json')):
        root = marker.parent
        result = json.loads(marker.read_text())
        cfg = result['config']
        assert result['status'] == 'complete' and cfg['source_hashes'] == hashes()
        for item in json.loads((root / 'manifest.json').read_text()):
            path = root / item['path']
            assert path.stat().st_size == item['bytes'] and T.sha256(path) == item['sha256']
        context = {k: cfg[k] for k in ('seed', 'task', 'start')}
        position = pd.read_csv(root / 'positions.csv')
        finite = pd.read_csv(root / 'finite.csv')
        heads = pd.read_csv(root / 'heads.csv')
        channels.extend(dict(**context, **r) for r in position.to_dict('records'))
        with np.load(root / 'arrays.npz') as d:
            for arm in ('old', 'next'):
                for layer in (1, 2):
                    pp = position[(position.dataset == arm) & (position.layer == layer) & (position.metric == 'Adam_fixed') & (position.route == 'full') & (position.moment == 'r')].set_index('signal')
                    ff = finite[(finite.dataset == arm) & (finite.layer == layer) & (finite.moment == 'r')].set_index('variant')
                    controls = [f'perm{i}' for i in range(4)]
                    hh = heads[heads.dataset == arm].set_index('variant')
                    base, perm = float(pp.loc['switch', 'mean']), float(pp.loc[controls, 'mean'].mean())
                    fbase, fperm = float(ff.loc['switch', 'mean']), float(ff.loc[controls, 'mean'].mean())
                    effects.append(dict(**context, dataset=arm, layer=layer, base_fixed_r=base, perm_fixed_r=perm,
                        fixed_difference=base - perm, registered_positive=base < 0 and base < perm,
                        base_finite_r=fbase, perm_finite_r=fperm, finite_difference=fbase - fperm,
                        base_pmax=float(hh.loc['switch', 'pmax']), perm_pmax=float(hh.loc[controls, 'pmax'].mean()),
                        finite_positive=fbase < 0 and fbase < fperm))
                    for variant in ('switch', 'half', 'zero', 'norm_unload', *controls):
                        doses.append(dict(**context, dataset=arm, layer=layer, variant=variant,
                            fixed_r=float(pp.loc[variant, 'mean']), finite_r=float(ff.loc[variant, 'mean']),
                            norm_ratio=float(hh.loc[variant, 'norm_ratio']),
                            alignment_coefficient=float(hh.loc[variant, 'alignment_coefficient']),
                            pmax=float(hh.loc[variant, 'pmax']), old_ce=float(hh.loc[variant, 'old_ce'])))
                    valid = d[f'initial_L{layer}_valid']
                    for metric in ('SGD', 'Adam_fixed'):
                        for route in ('full', 'local'):
                            a = d[f'{arm}_{metric}_{route}_load_r'][layer - 1]
                            b = d[f'old_{metric}_{route}_fit_r'][layer - 1]
                            overlaps.append(dict(**context, dataset=arm, layer=layer, metric=metric, route=route, **pair(a, b, valid)))
                        for signal in ('switch', 'load') + (('fit',) if arm == 'old' else ()):
                            mu, sd = d[f'initial_L{layer}_mu'], d[f'initial_L{layer}_sd']
                            for route in ('full', 'local', 'upstream'):
                                def field(moment):
                                    full = d[f'{arm}_{metric}_full_{signal}_{moment}'][layer - 1]
                                    local = d[f'{arm}_{metric}_local_{signal}_{moment}'][layer - 1]
                                    return {'full': full, 'local': local, 'upstream': full - local}[route]
                                dm, ds, dr = field('mu'), field('sd'), field('r')
                                mean_term = dm / (sd + 1e-12)
                                width_term = -mu * ds / (sd + 1e-12) ** 2
                                assert np.max(np.abs(mean_term + width_term - dr)) < 1e-8
                                terms.append(dict(**context, dataset=arm, layer=layer, metric=metric,
                                    route=route, signal=signal, raw_mu=float(dm[valid].mean()), raw_sd=float(ds[valid].mean()),
                                    mean_term=float(mean_term[valid].mean()), width_term=float(width_term[valid].mean()),
                                    total_r=float(dr[valid].mean())))
        count += 1
    output = T.ROOT / 'results/vit_two_block_1025'
    output.mkdir(exist_ok=True)
    for name, rows in [('alignment_effects', effects), ('alignment_overlap', overlaps),
                       ('alignment_channels', channels), ('alignment_terms', terms), ('alignment_head_doses', doses)]:
        pd.DataFrame(rows).to_csv(output / f'{name}.csv', index=False)
    text = ['# 2blockの整列介入・途中結果', '', f'完了状態 {count}/20。未完了seedを除いて成功基準を緩めない。', '',
        '最終headの旧ラベル整列を、同じnormの4方向へ置き換えた比較。負のΔrが沈降。',
        '登録判定は全5seed中4以上で「baseが沈み、4対照平均より下向き」。有限Adamの1歩は別の判定として併記する。', '',
        '| 課題 | 起点 | 入力 | block | 完了seed | 固定計量の成立 | 実Adamの成立 | base−対照の固定計量Δr中央値 |',
        '|---:|---:|---|---:|---:|---:|---:|---:|']
    if effects:
        df = pd.DataFrame(effects)
        for (task, start, arm, layer), g in df.groupby(['task', 'start', 'dataset', 'layer']):
            text.append(f'| {task} | {start} | {arm} | {layer} | {len(g)}/5 | {int(g.registered_positive.sum())}/{len(g)} | '
                        f'{int(g.finite_positive.sum())}/{len(g)} | {g.fixed_difference.median():.6g} |')
    text += ['', '確率を固定した整列成分loadと、確率も再計算するhead介入は別である。',
             'effects.csvのbase_pmax・perm_pmaxは介入後の平均最大予測確率。headのnormは合わせているが、予測確率や逆伝播方向は同じではない。',
             'unit対応はalignment_overlap.csv、平均・幅・位置および全経路/自fc1の内訳はalignment_channels.csv。',
             'rへの平均・幅の寄与と上流/自fc1の内訳はalignment_terms.csv。平均・幅のseed中央値を別々に取ると、その和はrのseed中央値と一致するとは限らない。', '']
    text += ['', '## 整列成分の量とhead normの対照', '',
        '全5seedが揃った条件の実Adam1歩のΔr。各値はunit平均→seed平均、括弧は沈むseed数。上の登録判定表の中央値とは異なる。',
        'half/zeroは整列成分Lを半分/ゼロへ変え、norm_unloadは整列を除いた残差Rを元headと同じnormへ拡大する。置換は各seedで4方向を平均してからseed間で集計する。',
        '全variantで予測確率を計算し直した。これは新しい成功基準ではなく、登録した量の操作とnorm対照の記述。未完了条件もalignment_head_doses.csvには保持する。', '',
        '| 課題 | 起点 | 入力 | block | base | half | zero | norm_unload | norm一致の置換 |',
        '|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    dd = pd.DataFrame(doses)
    if len(dd):
        for keys, group in dd.groupby(['task', 'start', 'dataset', 'layer']):
            if set(group.seed) != set(T.SEEDS):
                continue
            task, start, arm, layer = keys
            assert len(group) == len(T.SEEDS) * 8
            wide = group.pivot(index='seed', columns='variant', values='finite_r')
            values = [wide[k] for k in ('switch', 'half', 'zero', 'norm_unload')]
            values.append(wide[[f'perm{i}' for i in range(4)]].mean(axis=1))
            cells = [f'{v.mean():.6g} ({int((v < 0).sum())}/5)' for v in values]
            text.append(f'| {task} | {start} | {arm} | {layer} | ' + ' | '.join(cells) + ' |')
    (output / 'alignment_partial.md').write_text('\n'.join(text))
    print(json.dumps(dict(completed_states=count, output=str(output / 'alignment_partial.md'))))


if __name__ == '__main__':
    main()

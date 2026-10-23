#!/usr/bin/env python3
"""Deterministic judge of preregistered windows; numpy only, no model execution."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/chargpt_latepush_1023'
WINDOWS = {'early': range(1, 10), 'middle': range(39, 51), 'late': range(89, 101)}
METRICS = ['s_full', 's_load', 's_res', 'share_load', 'n_alive', 'kappa', 'cos2',
           'abs_full_median', 'abs_load_median', 'abs_res_median', 'kn_positive', 'kn_negative']


def ranks(x):
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return (ends-counts+(counts+1)/2)[inv]


def spearman(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    a, b = ranks(x), ranks(y)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def finite_median(rows, col):
    x = [r[col] for r in rows]
    return float(np.median(x)) if x and np.isfinite(x).all() else None


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)


def table(headers, rows):
    esc = lambda c: str(c).replace('|', '\\|')
    return '\n'.join(['| ' + ' | '.join(esc(c) for c in headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(esc(c) for c in r) + ' |' for r in rows])


def main():
    assert np.allclose(ranks([1, 2, 2, 4]), [1, 2.5, 2.5, 4])
    assert abs(spearman([1, 2, 2, 4], [4, 3, 3, 1])+1) < 1e-14
    assert spearman([1, 1], [1, 2]) is None
    rows = []
    for r in csv.DictReader((OUT/'states.csv').open()):
        rows.append({k: (v if k == 'arm' else float(v)) for k, v in r.items()})
    assert len(rows) == 1440
    keys = [(r['arm'], r['seed'], r['task'], r['layer']) for r in rows]
    assert len(set(keys)) == 1440
    check = json.loads((OUT/'reproduction_check.json').read_text())
    assert check['passed']
    windows, correlations, judged = [], [], {}
    for arm in ['C3', 'ReLUcap']:
        judged[arm] = {}
        for layer in [0, 1]:
            records = []
            for seed in range(10):
                rr = sorted([r for r in rows if r['arm'] == arm and r['layer'] == layer and r['seed'] == seed],
                            key=lambda r: r['task'])
                assert len(rr) == 36
                ww = {}
                for name, tasks in WINDOWS.items():
                    rs = [r for r in rr if r['task'] in tasks]
                    assert len(rs) == len(tasks)
                    ww[name] = {col: finite_median(rs, col) for col in METRICS}
                    ww[name]['n_alive_min'] = min(r['n_alive'] for r in rs)
                    windows.append(dict(arm=arm, layer=layer, seed=seed, window=name, n_states=len(rs), **ww[name]))
                ends = [r for r in rr if r['task'] > 0]
                rho = spearman([r['s_full']-.5 for r in ends], [r['cos2'] for r in ends])
                rho36 = spearman([r['s_full']-.5 for r in rr], [r['cos2'] for r in rr])
                correlations.append(dict(arm=arm, layer=layer, seed=seed, n_task_ends=len(ends), rho=rho,
                                         n_with_init=len(rr), rho_with_init=rho36))
                e, l = ww['early'], ww['late']
                records.append(dict(seed=seed, late_s_load=l['s_load'], late_s_res=l['s_res'],
                                    early_share_load=e['share_load'], late_share_load=l['share_load'], rho=rho,
                                    rho_with_init=rho36,
                                    loading=l['s_load'] is not None and l['s_load'] >= .75,
                                    residual=l['s_res'] is not None and .40 <= l['s_res'] <= .60,
                                    share_decreases=e['share_load'] is not None and l['share_load'] is not None and l['share_load'] < e['share_load'],
                                    tracks=rho is not None and rho > .5))
            counts = {k: sum(r[k] for r in records) for k in ['loading', 'residual', 'share_decreases', 'tracks']}
            counts['share_evaluable'] = sum(r['early_share_load'] is not None and r['late_share_load'] is not None for r in records)
            res, shr = counts['residual'] >= 8, counts['share_decreases'] >= 8
            judged[arm][str(layer)] = dict(
                J1='LOADING_SINKS_LATE' if counts['loading'] >= 8 else 'NOT_LOADING',
                J2='RESIDUAL_DILUTES' if res and shr else ('PARTIAL' if res or shr else 'NOT_RESIDUAL_DILUTES'),
                J3='TRACKS_COS2' if counts['tracks'] >= 8 else 'NOT_TRACKING_COS2',
                counts=counts, seeds=records)
    write_csv(OUT/'windows.csv', windows); write_csv(OUT/'correlations.csv', correlations)
    sanity_cols = ['orthogonality_error', 'readout_additivity_error', 'push_additivity_error']
    sanity = {k: max(r[k] for r in rows) for k in sanity_cols}
    sanity.update(n_zero_alive=sum(r['n_alive'] == 0 for r in rows),
                  zero_share_denominators=sum(int(r['share_zero_denominator']) for r in rows),
                  nonfinite_primary_values=sum(not np.isfinite(r[k]) for r in rows for k in ['s_full','s_load','s_res','share_load']))
    report = dict(main=judged['C3']['1'], records=judged, sanity=sanity, reproduction=check,
                  task_end_points_for_J3=35, supplementary_points_with_init=36,
                  threshold_semantics='J1 >= .75, J2 [.40,.60] and strict share decrease, J3 > .5; each >= 8 seeds')
    (OUT/'judge.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    jrows = []
    for arm in judged:
        for layer, j in judged[arm].items():
            c = j['counts']
            jrows.append([arm, f'L{layer}', j['J1'], c['loading'], j['J2'], c['residual'], c['share_decreases'], c['share_evaluable'], j['J3'], c['tracks']])
    jtext = table(['arm','layer','J1','n J1','J2','n residual','n share↓','n share evaluable','J3','n J3'], jrows)
    (OUT/'judge.md').write_text('# 判定（登録 spec §4・§8）\n\n'+jtext+
        '\n\n主判定は C3 L1。各条件は10 seed 中8以上。L0・ReLUcap は記録。J3 は課題末35点、init 込み36点は補助。\n')

    def span(arm, layer, win, metric, digits=4):
        vals = [w[metric] for w in windows if w['arm']==arm and w['layer']==layer and w['window']==win]
        finite = [x for x in vals if x is not None]
        if not finite: return '欠測（0/10）'
        value = f'{min(finite):.{digits}f}–{max(finite):.{digits}f}'
        return value if len(finite)==10 else value+f'（{len(finite)}/10のみ）'

    lines = ['# 後期の押しを装填成分と残りに分ける（chargpt_latepush_1023）',
             '', '保存状態のみ。新しい学習・API ワーカー・サブエージェントなし。', '', jtext,
             '', f"主 C3 L1: 装填成分の後期 s_load ≥0.75 は {judged['C3']['1']['counts']['loading']}/10。『後期も装填の押しは沈める側にそろう』という登録予測は支持されない。κ>0 と unit ごとの押しの向きは別の量。",
             '', '## 窓ごとの10 seed の範囲', '',
             '各 seed で窓内の状態中央値を取り、その10 seed 間の最小–最大。share_load は各状態の生きた unit の絶対値分担の中央値。', '']
    wrows = []
    for arm in ['C3','ReLUcap']:
        for layer in [1,0]:
            for win in WINDOWS:
                wrows.append([arm, f'L{layer}', win]+[span(arm, layer, win, k) for k in ['s_full','s_load','s_res','share_load','n_alive','cos2']])
    lines += [table(['arm','layer','window','s_full','s_load','s_res','share_load','n_alive med','cos²'],wrows),
              '', '## Spearman（seed 別）', '', '同順位は平均順位。s_full−0.5 と cos²。課題末35点を主判定に使用。', '']
    crows=[]
    for r in correlations:
        crows.append([r['arm'],f"L{r['layer']}",r['seed'],f"{r['rho']:.6f}" if r['rho'] is not None else '欠測',
                      f"{r['rho_with_init']:.6f}" if r['rho_with_init'] is not None else '欠測'])
    lines += [table(['arm','layer','seed','ρ (35 ends)','ρ (36 with init)'],crows),
              '', '## L0・少数生存・絶対量', '']
    for arm in ['C3','ReLUcap']:
        l0=[w for w in windows if w['arm']==arm and w['layer']==0 and w['window']=='early']
        below=sum(w['s_load'] is not None and w['s_load'] < .5 for w in l0)
        late=[r for r in rows if r['arm']==arm and r['layer']==1 and r['task'] in WINDOWS['late']]
        lines.append(f"- {arm} L0: 早期 s_load < 0.5 は {below}/10。早期の元指標 s_full と、新しい s_load を同一視しない。")
        lines.append(f"- {arm} L1 後期 n_alive: 全状態の最小–最大 {min(r['n_alive'] for r in late):.0f}–{max(r['n_alive'] for r in late):.0f}/512。窓中央値の10 seed 範囲は表に併記。")
    arows=[]
    for arm in ['C3','ReLUcap']:
        for win in WINDOWS:
            arows.append([arm,win]+[span(arm,1,win,k,7) for k in ['abs_full_median','abs_load_median','abs_res_median']])
    lines += ['',table(['arm','L1 window','median |S_full|','median |S_load|','median |S_res|'],arows)]
    for arm in ['C3','ReLUcap']:
        if judged[arm]['1']['J1']=='NOT_LOADING':
            krows=[]
            for task in sorted(set(r['task'] for r in rows if r['task'] > 0)):
                rr=[r for r in rows if r['arm']==arm and r['layer']==1 and r['task']==task]
                krows.append([int(task)]+[f"{min(r[k] for r in rr):.6f}–{max(r[k] for r in rr):.6f}" for k in ['kn_positive','kn_negative','kn_zero']])
            lines += ['',f'## J1 不成立時の K_n 符号（{arm} L1、課題末別、10 seed の範囲）','',table(['task','K_n > 0','K_n < 0','K_n = 0'],krows)]
    lines += ['', '## 検査・出所・限界', '',
              f"- 1 seed 一致確認: seed 0、両腕、init＋課題末35点、両層（{check['layer_rows']}行）。PASS。最大誤差 {json.dumps(check['worst'],ensure_ascii=False)}。",
              f'- 全状態 sanity: {json.dumps(sanity,ensure_ascii=False)}。',
              '- share_load は両成分の押しが0の生存 unit で0/0になり、その状態と窓中央値を欠測にする（登録どおり）。C3 L1 の早期→後期の有効な seed ペアは0/10。J2 の share低下0/10は「低下しない」と同義ではない。事後のゼロ除外補助値は別節に記録。',
              '- F・κ・cos²・probe・alive・押しの定義は analysis/chargpt_seed10_1022/load10.py。F は固定 probe の実現ラベルで計算。',
              '- softmax 誤差は元の状態に固定し、body への逆伝播の読み出しだけを分ける。Adam の分母は元解析と同じく不使用。',
              '- 35課題末×10 seed×2腕=700状態。init20状態も保存、両層で states.csv は1440行。',
              '- J1は符号の偏りであり絶対的な押しの強さを判定しない。s_res が半々でも乱数性・独立性を証明しない。',
              '- v⊥ は現課題の F に直交する残差。この解析だけでは旧課題由来と同定できない。因果寄与率・100更新の分担ではない。',
              '- 事前 spec §8 は点数、固定softmax、元 src の git blob 復元などを実装前に確定。判定・予測・窓の事後変更なし。',
              '- 参照: states.csv、windows.csv、correlations.csv、judge.json、reproduction_check.json、provenance.json。']
    full_check = json.loads((OUT/'full_reproduction_check.json').read_text()) if (OUT/'full_reproduction_check.json').exists() else None
    if full_check:
        lines += [f"- 事後の全10 seed照合: 全{full_check['layer_rows']}層行で元CSVを再現。最大誤差 {json.dumps(full_check['worst'],ensure_ascii=False)}。出典 full_reproduction_check.json。"]
    lines += ['', '## 登録予測の採点', '',
              table(['項目','Fableの事前確率','結果'],[
                  ['J1', '.65', judged['C3']['1']['J1']],
                  ['J2', '.60', judged['C3']['1']['J2']+'（shareは欠測で判定条件不成立）'],
                  ['J3', '.55', judged['C3']['1']['J3']],
                  ['L0 early s_load < .5', '.70', str(sum(w['s_load'] is not None and w['s_load'] < .5 for w in windows if w['arm']=='C3' and w['layer']==0 and w['window']=='early'))+'/10']]),
              '', 'Issa の予測欄は空欄のまま。']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    print(jtext)


if __name__ == '__main__':
    main()

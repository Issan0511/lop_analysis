#!/usr/bin/env python3
"""Explicit seed-level aggregation and grading; no changes to registered spec.

Primary choices, made before the 20-run result was inspected:
A1: ratio of pooled negative contributions (all units), cross kept separate.
A2: median of window-wise unit medians, i+ii pooled; show i/ii separately.
A3: per-seed task mean of ii own_mean, then median paired cap-C3 difference.
A4: per-seed median of end-to-end unit-median 1-cos, then cap/C3 ratio.
B: per-seed task median, then count seeds and report median across seeds.
Phases/windows/layers omitted by a prediction are shown separately as sensitivity.
"""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/chargpt_l0_carrier_1021'
PHASES = {'1-10': (1, 10), '40-50': (40, 50), '90-100': (90, 100)}
ARMS = ['C3', 'ReLUcap']


def md(headers, rows):
    def fmt(x):
        if isinstance(x, (float, np.floating)):
            return f'{x:.4g}'
        return str(x).replace('|', r'\|').replace('\n', '<br>')
    return '\n'.join(['| ' + ' | '.join(fmt(x) for x in headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(fmt(x) for x in row) + ' |' for row in rows])


def main(pa5):
    checks = json.loads((OUT / 'checks.json').read_text())
    assert checks['status'] == 'complete'
    source = Path('/home/issan/Projects/claude/wt/chargpt_cl_1019')
    assert all(hashlib.sha256((source / p).read_bytes()).hexdigest() == digest
               for p, digest in checks['source_sha256'].items())
    assert hashlib.sha256((ROOT / 'specs/spec_chargpt_l0_carrier_1021.md').read_bytes()).hexdigest() == checks['spec_sha256']
    d, s, shift, b = [pd.read_csv(OUT / f'table_{k}.csv') for k in ('decomp', 'scale', 'data_shift', 'carrier_source')]
    frames = {'decomp': d, 'scale': s, 'data_shift': shift, 'carrier_source': b}
    expected = {'decomp': 2640, 'scale': 3880, 'data_shift': 1400, 'carrier_source': 640}
    for k, frame in frames.items():
        assert len(frame) == expected[k]
        keys = ['arm', 'seed', 'layer', 'task'] + (['window'] if k in ('decomp', 'scale') else [])
        assert not frame.duplicated(keys).any()
    assert np.allclose(d.dmu_mean, d.own_mean+d.up_mean+d.cross_mean, atol=1e-10, rtol=0)
    assert len(list((OUT / 'units').glob('*.npz'))) == 20
    units_size = sum(p.stat().st_size for p in (OUT / 'units').glob('*.npz'))
    archive_error = 0.0
    for (arm, seed), group in d.groupby(['arm', 'seed']):
        with np.load(OUT/'units'/f'{arm}_seed{seed}.npz') as archive:
            for row in group.itertuples():
                prefix = f't{row.task}_{row.window}_L{row.layer}_'
                for term in ('dmu', 'own', 'up', 'cross'):
                    arr = archive[prefix+term]
                    assert arr.shape == (512,) and arr.dtype == np.float32
                    assert np.isfinite(arr).all()
                    err = abs(float(arr.astype(float).mean()) - getattr(row, term+'_mean'))
                    archive_error = max(archive_error, err)
                    assert err <= max(1.0, float(np.abs(arr).max())) * np.finfo(np.float32).eps * 2
                assert int((archive[prefix+'dmu'] < 0).sum()) == row.n_sinking
    checks['output_validation'] = dict(pass_=True, rows=expected, unique_keys=True,
                                     aggregate_identity_max=float((d.dmu_mean-d.own_mean-d.up_mean-d.cross_mean).abs().max()),
                                     npz_files=20, npz_bytes=units_size)
    checks['output_validation']['npz_mean_max_rounding_error'] = archive_error
    checks['output_validation']['source_and_spec_hashes_unchanged'] = True
    # A row is one independent seed's estimate; all threshold counts use these.
    sm = []
    def put(pred, arm, layer, window, phase, seed, metric, value):
        sm.append(dict(prediction=pred, arm=arm, layer=layer, window=window, phase=phase,
                       seed=seed, metric=metric, value=float(value)))
    decseed = []
    for (arm, seed, li, win), group in d.groupby(['arm', 'seed', 'layer', 'window']):
        ps = PHASES if win != 'coarse' else {'10-25': (25,25), '25-39': (39,39)}
        for phase, (lo, hi) in ps.items():
            g = group[group.task.between(lo, hi)]
            if g.empty:
                continue
            r = dict(arm=arm, seed=seed, layer=li, window=win, phase=phase, n_tasks=len(g),
                     sinking_unit_windows=int(g.n_sinking.sum()))
            for col in ('dmu_mean', 'own_mean', 'up_mean', 'cross_mean', 'sinking_fraction'):
                r[col] = g[col].mean()
            for col in ('dmu_mean', 'own_mean', 'up_mean', 'cross_mean'):
                r[col+'_sinking'] = np.average(g[col+'_sinking'].fillna(0), weights=g.n_sinking) if g.n_sinking.sum() else np.nan
            for suffix in ('', '_sinking'):
                weights = np.ones(len(g)) if not suffix else g.n_sinking.to_numpy()
                negs = {k: np.nansum(g['neg_'+k+suffix].to_numpy()*weights) for k in ('own', 'up', 'cross', 'up_cross')}
                denom = negs['own']+negs['up']+negs['cross']
                legacy_denom = negs['own']+negs['up_cross']
                r['up_share'+suffix] = negs['up']/denom if denom else np.nan
                r['legacy_up_share'+suffix] = negs['up_cross']/legacy_denom if legacy_denom else np.nan
            decseed.append(r)
    ds = pd.DataFrame(decseed)
    ds.to_csv(OUT/'table_decomp_seed.csv', index=False)
    for row in ds.itertuples():
        if row.arm == 'C3' and row.layer == 0 and row.window == 'ii' and row.phase == '40-50':
            for metric in ('up_share', 'up_share_sinking', 'legacy_up_share', 'legacy_up_share_sinking'):
                put('P-A1', row.arm, 0, 'ii', row.phase, row.seed, metric, getattr(row, metric))
    ss = []
    windows = s[s.window.isin(['i','ii'])]
    for (arm, seed, li), group in windows.groupby(['arm', 'seed', 'layer']):
        for win in ('i', 'ii', 'i+ii'):
            g = group if win == 'i+ii' else group[group.window == win]
            for phase, (lo, hi) in PHASES.items():
                z = g[g.task.between(lo,hi)]
                ss.append(dict(arm=arm, seed=seed, layer=li, window=win, phase=phase,
                               **{c: z[c].median() for c in ('up_abs_median','w_norm_median','dy_norm','alignment_median','lr_over_w_median')}))
    ss = pd.DataFrame(ss)
    ss.to_csv(OUT/'table_scale_seed.csv', index=False)
    for (arm, seed, li, win), group in ss.groupby(['arm','seed','layer','window']):
        z = group.set_index('phase')
        for metric in ('up_abs_median','w_norm_median','dy_norm','alignment_median'):
            put('P-A2',arm,li,win,'late/early',seed,metric+'_ratio',z.loc['90-100',metric]/z.loc['1-10',metric])
    for (seed, li, win, phase), group in ss.groupby(['seed','layer','window','phase']):
        z = group.set_index('arm')
        ratio = z.loc['ReLUcap','dy_norm']/z.loc['C3','dy_norm']
        put('P-A2','paired',li,win,phase,seed,'dy_cap_over_C3',ratio)
        put('P-A2','paired',li,win,phase,seed,'dy_symmetric_fold',max(ratio,1/ratio))
    for li in (0,1):
        for phase, (lo,hi) in dict(PHASES, all=(1,100)).items():
            g = d[(d.window=='ii') & (d.layer==li) & d.task.between(lo,hi)]
            p = g.groupby(['seed','arm']).own_mean.mean().unstack()
            for seed,row in p.iterrows():
                put('P-A3','paired',li,'ii',phase,seed,'own_cap_minus_C3',row.ReLUcap-row.C3)
    rotations = []
    for li in (0,1):
        for phase,(lo,hi) in dict(PHASES,all=(1,100)).items():
            g = s[(s.window=='end_to_end') & (s.layer==li) & s.task.between(lo,hi)]
            p = g.groupby(['seed','arm']).one_minus_cos_median.median().unstack()
            for seed,row in p.iterrows():
                put('P-A4','paired',li,'end_to_end',phase,seed,'rotation_cap_over_C3',row.ReLUcap/row.C3)
            for (arm,seed),z in g.groupby(['arm','seed']):
                rotations.append(dict(arm=arm,seed=seed,layer=li,phase=phase,
                                      one_minus_cos=z.one_minus_cos_median.median(),
                                      cos=z.rotation_cos_median.median(), lr_over_w=z.lr_over_w_median.median()))
    rotations = pd.DataFrame(rotations)
    rotations.to_csv(OUT/'table_rotation_seed.csv',index=False)
    for (arm,seed,li),g in shift.groupby(['arm','seed','layer']):
        put('P-A5',arm,li,'data_only','all_saved',seed,'abs_shift_median',g.abs_median.median())
        put('P-A5',arm,li,'data_only','all_saved',seed,'abs_shift_max',g.abs_median.max())
    bs = []
    for (arm,seed),g in b.groupby(['arm','seed']):
        for phase,(lo,hi) in PHASES.items():
            z = g[g.task.between(lo,hi)]
            r=dict(arm=arm,seed=seed,phase=phase,**{c:z[c].median() for c in ('emb','attn0','ffn0','attn1','ffn0_preln')})
            bs.append(r)
            put('P-B1' if phase=='1-10' else 'P-B2',arm,1,'end',phase,seed,'ffn0',r['ffn0'])
    bs = pd.DataFrame(bs)
    bs.to_csv(OUT/'table_carrier_seed.csv', index=False)
    sm = pd.DataFrame(sm)
    sm.to_csv(OUT/'table_seed_metrics.csv',index=False)
    def vals(pred,metric,arm=None,layer=0,window=None,phase=None):
        g=sm[(sm.prediction==pred)&(sm.metric==metric)&(sm.layer==layer)]
        if arm is not None: g=g[g.arm==arm]
        if window is not None: g=g[g.window==window]
        if phase is not None: g=g[g.phase==phase]
        return g.value.to_numpy()
    scores=[]
    def score(pred, verdict, reason):
        scores.append(dict(prediction=pred,verdict=verdict,reason=reason))
    a1=vals('P-A1','up_share','C3')
    a1s=vals('P-A1','up_share_sinking','C3')
    score('P-A1','○' if (a1>=.6).sum()>=8 else '✗',f'上流割合≥0.6: {sum(a1>=.6)}/10。中央値 {np.median(a1):.4f}。沈むunit限定 {sum(a1s>=.6)}/10。')
    a2c=vals('P-A2','up_abs_median_ratio','C3',window='i+ii',phase='late/early')
    a2r=vals('P-A2','up_abs_median_ratio','ReLUcap',window='i+ii',phase='late/early')
    dy=sm[(sm.prediction=='P-A2')&(sm.metric=='dy_symmetric_fold')&(sm.layer==0)&(sm.window=='i+ii')]
    dymax=dy.groupby('seed').value.max().to_numpy()
    a2ok=(a2c>=2).sum()>=8 and (a2r<=1.3).sum()>=8 and (dymax<=1.5).sum()>=8
    score('P-A2','○' if a2ok else '✗',f'C3伸び≥2: {sum(a2c>=2)}/10（中央 {np.median(a2c):.4f}）。cap伸び≤1.3: {sum(a2r<=1.3)}/10（中央 {np.median(a2r):.4f}）。入力変化の腕間差≤1.5倍が全3期: {sum(dymax<=1.5)}/10。')
    a3=vals('P-A3','own_cap_minus_C3',phase='all')
    score('P-A3','○' if np.median(a3)<=.05 else '✗',f'L0窓ii・全対象課題の自分の項の差: 中央 {np.median(a3):+.4f}（閾値≤+0.05）。')
    a4=vals('P-A4','rotation_cap_over_C3',phase='all')
    score('P-A4','○' if (a4>=1.2).sum()<=5 else '✗',f'L0回転量のcap/C3≥1.2: {sum(a4>=1.2)}/10（閾値≤5/10）。比の中央 {np.median(a4):.4f}。')
    violations=int((shift.abs_median>.1).sum())
    # A single observed violation disproves a universal bound even though only
    # saved checkpoints are available. Passing all saved ends cannot prove all 100.
    pooled = shift.groupby(['arm','seed','layer']).abs_median.median()
    pa5_verdict = ('✗' if violations else '判定不能') if pa5 == 'per-end' else ('○' if (pooled<=.1).all() else '✗') if pa5 == 'pooled' else '判定不能'
    score('P-A5',pa5_verdict,f'各課題末の閾値なら✗: {violations}/{len(shift)} 行が>0.1。課題も中央値集約なら○: 両腕・両層で各10/10 seedが≤0.1。集約の指定: {pa5}。保存済み35課題末の範囲。')
    for pred,phase,th,nreq,relation in [('P-B1','1-10',.6,8,'ge'),('P-B2','90-100',.5,6,'lt')]:
        parts=[]; passes=[]
        for arm in ARMS:
            v=bs[(bs.arm==arm)&(bs.phase==phase)].ffn0.to_numpy()
            n=int((v>=th).sum() if relation=='ge' else (v<th).sum())
            parts.append(f'{arm}: {n}/10（中央 {np.median(v):.4f}）')
            passes.append(n>=nreq)
        score(pred,'○' if all(passes) else '✗','、'.join(parts)+'。両腕で条件成立を要求。')
    pd.DataFrame(scores).to_csv(OUT/'table_scores.csv',index=False)

    lines=['# chargpt_l0_carrier_1021 — (A)(B) の計算結果','',
           '固定済み checkpoint の分解と射影を計算した。',
           '(C) の学習再生は実施していない。',
           'spec の予測は変更していない。',
           'commit は実施していない。','',
           '## 定義と集計','',
           '対象は両腕・seed 0〜9・L0/L1 である。',
           '窓 i は前課題末から更新100までである。',
           '課題1の窓 i は init から始まる。',
           '窓 ii は更新100から課題末までである。',
           '主対象は課題1〜10・40〜50・90〜100である。',
           '粗い窓10→25・25→39は終点課題のprobeで評価した。',
           '両端は同じ `probes_M(ChainM(seed,t,0.01),seed,t)` のchain probeで評価した。','',
           '位置は m_i=w_i·ȳ とした。',
           'ȳ は LN2 後の FFN 入力のtoken平均である。',
           'Δm は Δw·ȳ_a（自分）、w_a·Δȳ（上流）、Δw·Δȳ（交差）に分けた。',
           '各項は始点の前活性の標本標準偏差 σ_a（ddof=1）で割った。',
           'forward は保存モデルと同じ float32 である。',
           '平均・標準偏差・分解の演算は float64 である。',
           'unit 配列の保存だけ float32 とした。',
           '採点は丸める前のCSV値で行った。','',
           '上流割合の主定義は N_up/(N_own+N_up+N_cross) である。',
           'N_j は各unit・窓で max(−項_j/σ_a,0) を取り、対象窓で合計した量である。',
           'P-A1 の主集計は全unitである。',
           '沈むunit限定は Δm<0 のunitだけで同じ計算を行う。',
           '沈むunit限定の窓をまとめる際は沈むunit数で重み付けした。',
           '旧定義は交差を上流へまとめてから負の量を取り、再現検査と感度分析にだけ用いた。','',
           '表は特記しない限り seed 内集約値の10 seed中央値である。',
           '8/10 は10 seedそれぞれで閾値を評価してから成功seedを数える。',
           'unitや課題を独立なseedとして数えない。',
           '分解の符号付き平均はseed内で窓の算術平均を取る。',
           '尺度は各窓のunit中央値を取り、その後seed内で課題・窓の中央値を取る。',
           'P-A2の主集計は窓i・iiをまとめた中央値である。',
           'P-A3の主集計はL0・窓iiの全対象課題平均をseed内で取り、対応するcap−C3の10 seed中央値を取る。',
           'P-A3は記載された片側閾値≤+0.05で採点した。',
           'P-A4は連続課題末のunit中央1−cosをseed内で中央値にしてcap/C3を取る。',
           '課題1の回転と保存間隔が空いた回転は含めない。',
           'P-A2〜A4はL0を主判定とした。',
           '時期別・窓別・L1の値は感度分析として残した。',
           'これらの集約順序はspecで一意に定まっていないため、実装時の選択である。','',
           'データ差は同じ課題t末の重みで probe(t)→probe(t+1) を比較し、probe(t)のσで割った。',
           'このデータ差には有限probeの標本差も含まれる。',
           'データ差は保存済み35課題末を対象とした。',
           '課題100末では未学習の課題101の規則からprobeを生成した。',
           '保存されていない課題末の同重み比較は計算できない。','',
           'P-A5は実装時に「各課題末のunit中央値」を主解釈に選んだ。',
           'その解釈での採点と「課題もまとめる中央値」での採点は異なった。',
           'specにはこの集約順序が明記されていない。',
           '集約の回答が未指定の場合、最終採点は判定不能とし、両解釈の結果を残す。','',
           '担い手の分解は各tokenで残差各項の特徴平均を引き、全残差に共通のLN標準偏差で割り、γを掛けた。',
           'そのtoken平均を全体のȳ₁へ射影し、‖ȳ₁‖²で割った。',
           '分母を各成分ごとに作り直すLN(成分)は用いていない。',
           'この割合は符号付きであり、負値や1超を許す。',
           '既存quicklookのcarrier_source.py・judge_m2.pyはLN前のx̄への射影だった。',
           '指定に従いLN後を主表とし、LN前の射影を `_preln` 列に残した。',
           'Bのseed内集約は対象課題の中央値である。',
           'Bの採点は両腕で条件が成立することを要求した。','',
           '## 検査','']
    ck=checks
    lines += [md(['検査','結果','数値'],[
        ['恒等式・全窓','PASS',f"raw {ck['full_identity_max']:.3e}; /σ {ck['full_normalized_identity_max']:.3e}; 閾値1e-10"],
        ['seed1000 上流割合','PASS',ck['seed1000']['upstream_share']],
        ['seed1000 L1集中','PASS',ck['seed1000']['concentration']],
        ['importした旧windowとの一致','PASS',ck['seed1000']['imported_window_max_difference']],
        ['cap seed0 L0/L1・66 checkpoint','PASS',str(ck['cap_seed0']['max_tracked_s_ratio_by_layer'])],
        ['Δw=0の変異','PASS',ck['mutations']['zero_dW_max_error']],
        ['データprobe交換の変異','PASS',ck['mutations']['probe_swap_identity_max_error']],
        ['LN後の成分和の相対誤差・全件','PASS',ck['full_carrier_closure_max']],
        ['出力行数・一意キー・平均の恒等式','PASS',str(expected)],
        ['unit npz','20ファイル',f'{units_size/1024**2:.2f} MiB'],
    ]),'',
    'cap の検査対象は追跡幅 s_i=√V_i である。',
    'float32保存に伴う丸めを相対許容5e−7とした。',
    'cap の上限配列は課題1末の追跡幅と完全一致した。',
    '重みノルムやprobe上のσがこの上限に収まることは要求していない。',
    'probe交換の変異検査では主分解のprobeを固定し、追加のデータ差だけのprobeを替えた。',
    '主分解自体のprobeを替えれば自分・上流・交差も変わり得る。',
    '担い手の成分和には元forwardのfloat32残差加算の丸めが残る。','',
    '## 予測の採点','',md(['予測','採点','根拠'],[[r['prediction'],r['verdict'],r['reason']] for r in scores]),'',
    '## 分解','',
    '各成分の単位はΔm/σ_aである。',
    '以下の各列は別々に10 seed中央値を取るため、表示列の和は合計列と厳密には一致しない。','']
    dr=[]
    for (arm,li,win,phase),g in ds.groupby(['arm','layer','window','phase'],sort=False):
        if win=='coarse': continue
        v=g.median(numeric_only=True)
        dr.append([arm,f'L{li}',win,phase,v.dmu_mean,v.own_mean,v.up_mean,v.cross_mean,v.up_share,v.up_share_sinking])
    lines += [md(['腕','層','窓','課題','合計','自分','上流','交差','上流割合・全体','上流割合・沈むunit'],dr),'',
              '沈むunitが存在しないseed・窓群の限定平均と割合は未定義とした。',
              'C3・seed8・L1・窓ii・課題40〜50がその群に当たる。',
              'この群の沈むunit限定のseed中央値だけは定義できる9 seedから計算した。','',
              '沈むunitに限った符号付き平均は次のとおりである。','']
    sinking=[]
    for (arm,li,win,phase),g in ds[ds.window!='coarse'].groupby(['arm','layer','window','phase'],sort=False):
        v=g.median(numeric_only=True)
        sinking.append([arm,f'L{li}',win,phase,v.dmu_mean_sinking,v.own_mean_sinking,v.up_mean_sinking,v.cross_mean_sinking,int(g.up_share_sinking.notna().sum())])
    lines += [md(['腕','層','窓','課題','合計','自分','上流','交差','定義可能seed'],sinking),'',
              '## L0 の尺度','']
    sr=[]
    for (arm,win,phase),g in ss[(ss.layer==0)&(ss.window!='i+ii')].groupby(['arm','window','phase'],sort=False):
        v=g.median(numeric_only=True)
        sr.append([arm,win,phase,v.up_abs_median,v.w_norm_median,v.dy_norm,v.alignment_median])
    lines += [md(['腕','窓','課題','median|w·Δȳ|','median‖w‖','‖Δȳ‖','揃い'],sr),'',
              '揃いはunitごとの |w·Δȳ|/(‖w‖‖Δȳ‖) の中央値である。','']
    ratios=[]
    for arm in ARMS:
        for win in ('i','ii','i+ii'):
            v=vals('P-A2','up_abs_median_ratio',arm,window=win,phase='late/early')
            wn=vals('P-A2','w_norm_median_ratio',arm,window=win,phase='late/early')
            count=int((v>=2).sum() if arm=='C3' else (v<=1.3).sum())
            ratios.append([arm,win,np.median(v),np.median(wn),f'{count}/10'])
    lines += [md(['腕','窓','|w·Δȳ|の終盤/序盤','‖w‖の終盤/序盤','P-A2の伸び閾値を満たすseed'],ratios),'']
    dyrows=[]
    for win in ('i','ii','i+ii'):
        for phase in PHASES:
            r=vals('P-A2','dy_cap_over_C3','paired',window=win,phase=phase)
            dyrows.append([win,phase,np.median(r),f'{sum((r>=1/1.5)&(r<=1.5))}/10'])
    lines += [md(['窓','課題','‖Δȳ‖のcap/C3','1.5倍以内'],dyrows),'',
              '## 自分の項と回転','']
    a34=[]
    for li in (0,1):
        for phase in list(PHASES)+['all']:
            o=vals('P-A3','own_cap_minus_C3',layer=li,phase=phase)
            r=vals('P-A4','rotation_cap_over_C3',layer=li,phase=phase)
            a34.append([f'L{li}',phase,np.median(o),np.median(r),f'{sum(r>=1.2)}/10'])
    lines += [md(['層','課題','自分の項 cap−C3','回転量 cap/C3','回転量比≥1.2'],a34),'']
    rr=[]
    for (arm,li,phase),g in rotations.groupby(['arm','layer','phase'],sort=False):
        if phase=='all': continue
        v=g.median(numeric_only=True)
        rr.append([arm,f'L{li}',phase,v.cos,v.one_minus_cos,v.lr_over_w])
    lines += [md(['腕','層','課題','cosの中央値','1−cosの中央値','lr/‖w‖'],rr),'',
              'lr/‖w‖ は基準学習率1e−3を用いた代理量である。',
              '課題末ではwarmupが終了している。',
              'この代理量はAdamの前処理後の実歩幅を表さない。','',
              '## データの差','']
    shrows=[]
    for (arm,li),g in shift.groupby(['arm','layer']):
        v=g.groupby('seed').abs_median.median()
        worst=g.loc[g.abs_median.idxmax()]
        shrows.append([arm,f'L{li}',v.median(),worst.abs_median,int(worst.seed),int(worst.task),f'{int((g.abs_median>.1).sum())}/{len(g)}'])
    lines += [md(['腕','層','seed中央値','最大の課題末unit中央値','seed','課題','>0.1の行数'],shrows),'',
              '課題100末を除いても各課題末の閾値を超える行が残る。',
              '課題で集約する解釈では、各seedのunit中央値を課題末で中央値にして評価した。','',
              md(['腕','層','課題集約の閾値内seed','課題100を除いた>0.1行数'],
                 [[arm,f'L{li}',f'{int((g.groupby("seed").abs_median.median()<=.1).sum())}/10',int(((g.task<100)&(g.abs_median>.1)).sum())]
                  for (arm,li),g in shift.groupby(['arm','layer'])]),'',
              '## L1 の担い手の出所','']
    br=[]
    for (arm,phase),g in bs.groupby(['arm','phase'],sort=False):
        v=g.median(numeric_only=True)
        br.append([arm,phase,v.emb,v.attn0,v.ffn0,v.attn1,v.ffn0_preln])
    lines += [md(['腕','課題','埋め込み','注意0','FFN0','注意1','参考: LN前FFN0'],br),'',
              '## 粗い窓（報告のみ）','']
    coarse=[]
    for (arm,li,phase),g in ds[ds.window=='coarse'].groupby(['arm','layer','phase'],sort=False):
        v=g.median(numeric_only=True)
        coarse.append([arm,f'L{li}',phase,v.dmu_mean,v.own_mean,v.up_mean,v.cross_mean,v.up_share])
    lines += [md(['腕','層','窓','合計','自分','上流','交差','上流割合'],coarse),'',
              '粗い窓は複数課題にまたがるため予測採点には用いていない。','',
              '## 言えること／言えないこと','',
              'L0の中期の課題内窓は、上流が負の量の大半を占めるという予測に合わなかった。',
              'L0の切替窓では、対照の中期・終盤で上流の割合が大きかった。',
              '上流項の絶対量の伸びは、対照とcapで異なる向きを示した。',
              '入力変化の大きさが両腕で同程度という予測は支持されなかった。',
              'capではL0の自分の項が対照より上向きだった。',
              'capではL0の課題末どうしの回転が対照より大きかった。',
              'したがって帳簿上は、上流項の尺度差と自分の項・回転の差の両方が現れた。',
              '序盤のL1担い手は両腕でFFN0の射影が大きかった。',
              '終盤のL1担い手は対照で注意1の射影が大きかった。',
              '終盤のcapではFFN0の射影が高いまま残った。',
              'この解析は保存状態の位置変化を加法的に配分する。',
              '担い手の表は指定状態の共通入力への射影を配分する。',
              '各成分を操作した際の効果は測定していない。',
              'capによる救済の因果経路を、この分解だけでは特定できない。',
              '重みノルムだけから上流の変位は決まらない。',
              '上流の変位には入力変化の大きさと揃いも関わる。',
              '学習可能なγを含むLNはȳの大きさを課題を通じて一定に保つ制約ではない。',
              '射影割合の変化だけからFFN0または注意1の因果的寄与は断定できない。',
              'P-A3の片側閾値を満たすことは差の小ささの同等性証明ではない。',
              '集約順序が一意でない予測には記載した操作的定義が付く。','',
              '## 再現と所在','',
              '実行コマンド:', '',
              '```bash',
              "CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/chargpt_l0_carrier_1021/run.py",
              "PYTHONDONTWRITEBYTECODE=1 /home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/chargpt_l0_carrier_1021/report.py",
              '```','',
              f"入力worktreeの読み取り時HEADは `{checks['source_head']}` だった。",
              '使用したモデル・分解・担い手解析のソースには指定a970f12cから差分がなかった。',
              'ソースとspecのSHA-256は `checks.json` に保存した。',
              '出力のraw表とseed表を同じディレクトリに保存した。',
              'unit配列は `units/{arm}_seed{seed}.npz` に圧縮保存した。','']
    lines += ['npzのキー `t{task}_{i|ii|coarse}_L{layer}_{dmu|own|up|cross}` は幅で割ったunit値を表す。',
              '`t{task}_data_L{layer}` はデータ差を表す。',
              '`t{task}_rotation_L{layer}` は連続課題末のcosを表す。','']
    (OUT/'summary.md').write_text('\n'.join(lines))
    checks['report_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    checks['pa5_aggregation'] = pa5
    checks['output_sha256'] = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(OUT.glob('*.csv')) + sorted((OUT/'units').glob('*.npz')) + [OUT/'summary.md']}
    (OUT/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    print(md(['予測','採点','根拠'],[[r['prediction'],r['verdict'],r['reason']] for r in scores]))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--pa5',choices=['ambiguous','per-end','pooled'],default='ambiguous')
    main(ap.parse_args().pa5)

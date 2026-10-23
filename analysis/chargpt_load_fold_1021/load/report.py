#!/usr/bin/env python3
"""Render Japanese report from measured tables, without altering registered predictions."""
import csv, json, hashlib
from pathlib import Path
import numpy as np
import measure as M
O=M.OUT

def read(name):
    rows=list(csv.DictReader((O/name).open()))
    return [{k:(v if k in ('box','arm','stage') else float(v) if v else float('nan')) for k,v in r.items()} for r in rows]
def fmt(v):
    return f'{v:.4g}' if np.isfinite(v) else '—'
def ran(rows,key):
    a=[r[key] for r in rows if np.isfinite(r.get(key,float('nan')))]; return '—' if not a else f'{fmt(min(a))}–{fmt(max(a))}'
def md(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def subset(rows,b,a): return [r for r in rows if r['box']==b and r['arm']==a]
def frac(rows,fn): return f'{sum(fn(r) for r in rows)}/{len(rows)}'

def main():
    H=read('table_head.csv'); U=read('table_units.csv'); P=read('table_p07.csv')
    h=[r for r in H if r['stage']=='end']; u=[r for r in U if r['stage']=='end' and r['layer']==(3 if r['box']=='M' else 1)]
    ck=json.loads((O/'checks.json').read_text()); assert ck['all_pass'] and len(ck['checks'])==64
    pv=json.loads((O/'provenance.json').read_text())
    verification=json.loads((O/'p07_verification.json').read_text()); assert verification['all_pass']
    assert ck['provenance']['script_sha256']==pv['script_sha256']==M.provenance()['script_sha256']
    assert ck['provenance']['input_code_sha256']==pv['input_code_sha256']==M.provenance()['input_code_sha256']
    assert ck['checks']==[json.loads(line) for line in (O/'checks.log').read_text().splitlines()]
    assert all(np.isfinite(r[k]) for r in H for k in ('kappa','cos2','Q','A','D','identity_rel'))
    headtable=[]; unittable=[]; ratios={}; detailed=[]
    for b,a in M.ARMS:
        hh=subset(h,b,a); uu=subset(u,b,a)
        init=next(r for r in U if r['box']==b and r['arm']==a and r['stage']=='init' and r['layer']==(3 if b=='M' else 1))
        one=next(r for r in uu if r['task']==1)
        ratio=init['own_abs_median']/one['own_abs_median']; ratios[b,a]=ratio
        headtable.append([f'{b}/{a}',len(hh),ran(hh,'kappa'),ran(hh,'cos2'),ran(hh,'Q'),ran(hh,'kappa_over_lb'),frac(hh,lambda r:r['kappa_ratio']>=.8)])
        early=[r for r in uu if 1<=r['task']<=10]; late=[r for r in uu if 39<=r['task']<=50]
        unittable.append([f'{b}/{a}',ran(early,'own_sink'),ran(late,'own_sink'),ran(uu,'full_sink'),ran(uu,'full_own_agreement'),fmt(ratio),fmt(max(abs(r['fixed_next_minus_uniform']) for r in uu)),ran(uu,'self_agreement')])
        detailed.append([f'{b}/{a}',frac(hh,lambda r:.1<=r['cos2']<=.6),frac(early,lambda r:r['own_sink']>=(.8 if a=='C3' else .7)) if a in ('C3','SiLU','GELU') else '対象外',frac(late,lambda r:r['own_sink']<.6) if a in ('SiLU','GELU') else '対象外',frac(uu,lambda r:r['full_own_agreement']<=.75),frac(uu,lambda r:abs(r['fixed_next_minus_uniform'])<=.1)])
    p1=all(r['kappa']>0 and r['lemma_violation']==0 for r in h)
    p2=all(.1<=r['cos2']<=.6 for r in h)
    c3=[(b,subset(h,b,'C3')) for b in ['M','M2']]
    p3c=all(sum(r['kappa_ratio']>=.8 for r in rr)/len(rr)>=.8 for b,rr in c3)
    early=[r for r in u if r['arm'] in ('C3','SiLU','GELU') and 1<=r['task']<=10]
    late=[r for r in u if r['arm'] in ('SiLU','GELU') and 39<=r['task']<=50]
    p4=all(r['own_sink']>=(.8 if r['arm']=='C3' else .7) for r in early) and all(r['own_sink']<.6 for r in late)
    p5=all(r['full_own_agreement']<=.75 for r in u)
    p6=all(v<=1/30 for v in ratios.values())
    p7=all(abs(r['fixed_next_minus_uniform'])<=.1 for r in u)
    verdict=lambda x:'○' if x else '✗'
    scores=[['P0-1',verdict(p1),'全課題末の κ は正。補題の破れはない。'],
            ['P0-2',verdict(p2),'登録上限を超える状態が全腕にある。M2/C3 では登録下限を下回る状態もある。'],
            ['P0-3','判定不能' if p3c else '✗','C3 の定量条件は'+('成立。' if p3c else '不成立。')+'SiLU・GELU の「cos² は保たれる」には登録された許容幅がなく、その節は判定不能。'],
            ['P0-4',verdict(p4),'早期条件に違反する状態がある。後期条件は対象の全腕で成立。適合数は採点内訳表。'],
            ['P0-5',verdict(p5),'符号一致率が登録上限を超える状態が全腕にある。適合数は採点内訳表。'],
            ['P0-6',verdict(p6),'全腕で登録上限を超えた。比は unit 表。'],
            ['P0-7（登録原文のみ）','一意に採点不能','比較位置・生存集合・own/full・状態間の集約規則が未指定。±0.1 という閾値は維持。'],
            ['P0-7（定義を補った再解析）',verdict(p7),'P0-7 は own で採点、full は代表状態のみ。現在課題の m_k・A_k を固定し次課題127位置CEと比較。全保存課題末 AND の解釈では不支持。']]
    head_md=md(['腕','課題末数','κ 範囲','cos² 範囲','Q 範囲','κ/κ_lb 範囲','κ/κ₁≥0.8'],headtable)
    unit_md=md(['腕','own 沈側 1–10','own 沈側 39–50','full 沈側 全課題末','full/own 符号一致','init/課題1','P0-7 最大絶対差','自己形との符号一致'],unittable)
    p07table=[]
    for b,a in M.ARMS:
        rr=subset(u,b,a)
        ff=[r for r in P if r['box']==b and r['arm']==a and r['full_recomputed']==1]
        p07table.append([f'{b}/{a}',f"{max(abs(r['fixed_next_minus_uniform']) for r in rr):.9f}",frac(rr,lambda r:abs(r['fixed_next_minus_uniform'])<=.1),f"{max(abs(r['legacy_next_minus_uniform']) for r in rr):.9f}",frac(rr,lambda r:abs(r['legacy_next_minus_uniform'])<=.1),f"{max(abs(r['fixed_next_full_minus_uniform']) for r in ff):.9f}",frac(ff,lambda r:abs(r['fixed_next_full_minus_uniform'])<=.1)])
    p07_md=md(['腕','own 新最大絶対差','own 新適合','旧比較 最大絶対差（補助）','旧適合（補助）','full 代表最大絶対差','full 代表適合'],p07table)
    checks_table=[]
    for name in dict.fromkeys(r['check'] for r in ck['checks']):
        rr=[r for r in ck['checks'] if r['check']==name]
        if name=='lemma_identity': detail=f"最大相対誤差 {max(r['relative_error'] for r in rr):.3g}; 許容 1e-10"
        elif name=='position_closure': detail=f"最大絶対誤差 {max(r['max_abs'] for r in rr):.3g}; 許容 1e-10"
        elif name in ('own_finite_difference','full_finite_difference'):
            errs=[v['relative_error'] for r in rr for v in (r['units'] if 'units' in r else r['trials'][-1]['units'])]
            detail=f'採用差分の最大相対誤差 {max(errs):.3g}; 許容 1e-4'
        elif name=='recorded_eval_rounding': detail=f"最大相対誤差 {max(r['relative_error'] for r in rr):.3g}; 許容 2e-6 + .9g 丸め幅（絶対値換算）"
        else: detail='詳細は checks.json'
        if name=='recorded_eval_rounding':
            checks_table.append(['recorded_eval_rounding（登録）',f"{sum(r['registered_rounding_pass'] for r in rr)}/{len(rr)}",'.9g の丸め幅だけ'])
            name='recorded_eval_rounding（float32差を加算）'
        checks_table.append([name,f"{sum(r['pass_'] for r in rr)}/{len(rr)}",detail])
    stage_table=md(['状態','状態数','head 恒等式最大相対誤差','補題違反'],[[s,len(rr:= [r for r in H if r['stage']==s]),fmt(max(r['identity_rel'] for r in rr)),int(sum(r['lemma_violation'] for r in rr))] for s in ['init','end','u100']])
    trend=[]; nulltable=[]
    for b,a in M.ARMS:
        rr=subset(h,b,a); one=next(r for r in rr if r['task']==1); last=max(rr,key=lambda r:r['task'])
        trend.append([f'{b}/{a}',int(last['task']),fmt(one['cos2']),fmt(last['cos2']),fmt(last['kappa_ratio'])])
        nulltable.append([f'{b}/{a}',ran(rr,'null_quantile'),frac(rr,lambda r:r['kappa']>r['null_q99']),int(sum(r['lemma_violation'] for r in rr))])
    text=f'''# 装填の固定状態解析（seed 1000）

仕様: `specs/spec_chargpt_load_fold_1021.md` §2・§4(A)。
登録の数値閾値は変更していない。P0-7 は登録原文だけの判定と、定義を補った再解析の採点を分離した。
採点対象は保存されている課題末である。
更新100の状態は補助計測として CSV と NPZ に含めた。
Adam 状態は固定状態の勾配計算には使用しない。

## 対象と定義

実装の特徴次元は仕様の記載と相違する。
checkpoint の形状を正として計算した。

| 次元 | 仕様の記載 | 実装・checkpoint |
| --- | --- | --- |
| 最終 LN の特徴次元 | 512 | 128 |
| FFN の unit 数 | — | 512 |

実装の根拠は `chargpt_cl_1019.py:37` である。

最終 LN の出力を h_n、中心化 head を v^c=v−mean_c(v) とする。
現課題の鎖の probe の全 token を用いる。

$$F=\\sum_n(e_{{y_n}}-\\mathbf1/V)h_n^T,\\quad A=\\langle v^c,F\\rangle,\\quad \\kappa=A/\\|F\\|^2.$$
$$\\cos^2(v^c,F)=A^2/(\\|v^c\\|^2\\|F\\|^2),\\quad Q=\\kappa\\|F\\|.$$
$$D=N(\\log V-L),\\quad \\kappa_{{lb}}=D/\\|F\\|^2,\\quad A=D+J,\\quad J=\\sum_n[\\log\\sum_c e^{{f_{{nc}}}}-\\overline f_n-\\log V]\\ge0.$$

bias がないため補題の B は零である。
帰無分布は head のクラス行の置換から作る。
置換分位点と観測 κ の経験的百分位は `table_head.csv` に保存した。
重み共有は通常計算と JVP で維持した。
符号反転の変異検査だけは埋め込みを固定して head の共有を外した。

現課題の probe における FFN 入力平均を ȳ とする。

$$m_j=\\mathbb E[z_j]=w_j\\cdot\\bar y,\\quad L_u=\\mathbb E_{{x\\sim\\mathrm{{uniform}}}}[\\log\\sum_c e^{{f_c(x)}}-\\overline f(x)].$$
$$G_{{own,j}}=\\bar y\\cdot\\nabla_{{w_j}}L_u,\\quad G_{{full,j}}=\\langle\\nabla_\\theta m_j,\\nabla_\\theta L_u\\rangle.$$

G が正なら無限小 SGD の Δm=−ηG は負側である。
G_full は共有パラメータを含む全パラメータ方向の JVP である。
実際の Adam 更新量を計算したものではない。
生きた unit は対象の鎖 probe で z>0 の token が存在する unit と定義した。
符号一致率からは両者のいずれかが零の unit を除いた。
初期化比の中央値には死んだ unit も含めた。
偶数個の中央値は PyTorch の下側中央値を用いた。

自己形は登録式 d_j=Σ_n K_n φ(z_jn)φ′(z_jn)y_n、K_n=ȳ·y_n を用いた。
ベクトル d_j とスカラー G_own の符号を比較するため、位置方向への射影 d_j·ȳ を用いた。
この射影の選択は仕様で明示されていないため、自己形との一致率は補助結果とする。

P0-7 の主比較は現在課題 k の鎖 probe で定義した位置 m_k（ȳ_k）と生存集合 A_k を両側で固定した。
一様側は登録どおり一様 probe の全128位置の L_u 勾配、次課題側は課題 k+1 の鎖 probe と次の文字（1実現）の CE 勾配である。
次課題 CE は元の学習 `chargpt_cl_1019.py:354–357` と同じ `logits[:,1:]` と `targets[:,1:]` の127位置とし、入力の128 token の文脈は保持した。
差は「次課題の沈側割合 − 一様側の沈側割合」。G=0 は沈側に数えず、分母は A_k 全体。
own は全286状態で再計算した。G_full は同じ m_k への JVP を各腕の課題1末・最終課題末、および監査の反例 M/SiLU・M/GELU 課題3末の18状態で再計算した。
P0-7 は own で採点、full は代表状態のみ。full の全状態適合数は主張しない。
init では位置と比較課題に課題1を用い、u100 とともに採点対象から除いた。
旧比較（次課題の位置平均・生存集合、全128位置CE）は `next_sink`・`next_minus_uniform` および旧NPZ配列に保存し、`legacy_next_minus_uniform` と表で補助結果として残した。
新比較は `fixed_next_own_sink`・`fixed_next_minus_uniform`、代表 full は `fixed_next_full_sink`・`fixed_next_full_minus_uniform` に保存した。

## 計測条件と検査

| 項目 | 値 |
| --- | --- |
| seed | 1000 |
| device / dtype | CPU / float64 |
| probe | 64 系列 × 128 token |
| 評価記録の再計算 | 256 系列 × 127 target |
| クラス数 | 65 |
| 帰無置換数 / seed | 1000 / 1021000 |
| スレッド数 | 8 |

{md(['検査','合格','誤差と基準'],checks_table)}

記録との一致以外の許容値は変更していない。
`chargpt_cl_1019.py:152–165` は既定 float32 でモデルを生成する。
同ファイルの `build_model` は device だけを移す。
`eval_M` はモデルの dtype で CE を計算してから Python float に変換する。
CSV writer は `.9g` で記録する。
評価記録の許容は、親（Claude、V13 セッション）の指示で変更した。利用者による承認ではない。
登録の丸め基準（.9g の丸め幅だけ）では 0/8、float32 の評価差を足した基準で 8/8。
`checks.json` の `registered_all_pass=false` と `all_pass=true` は、それぞれ登録基準と変更後基準に対応する。
今回の評価記録の分類は保存済みの誤差から再判定し、評価そのものを再実行したとは扱わない。
計算時の source HEAD は `6c80d308` で、spec の `f71be184` から進んでいた。
対象2ソースの差は `chargpt_cl_1019_kune.py` の更新100の Adam 状態保存の1行追加のみ。独立監査と git diff で確認し、forward・勾配・鎖生成に差はない。
検査の詳細と根拠は `checks.log` に出力した。
`checks.json` は同ログの JSON 行から生成した。
有限差分は各腕の課題1末で選んだ生存 unit を検査した。
これは全状態・全 unit の有限差分検査ではない。

{stage_table}

## Head の結果

各範囲は保存された課題末の最小値と最大値である。

{head_md}

課題1と最後の保存課題末の比較を示す。

{md(['腕','最終課題','cos² 課題1','cos² 最終','κ 最終/課題1'],trend)}

クラス置換帰無との比較を示す。
経験的百分位が上限でも、置換検定の p 値を零とは解釈しない。

{md(['腕','観測κの帰無百分位（0–1）','帰無99%点超過','補題違反'],nulltable)}

## Unit の結果

対象は読み出しに隣接する FFN 層である。
M は L3、M2 は L1 である。
cap 腕は保存済みの疎な課題末だけを集計した。
自己形との符号一致率は位置方向への射影による補助結果である。

{unit_md}

## P0-7 の再解析

以下は課題末の own 全151状態と、今回の再計算前に選んだ full 代表18状態の比較。適合は絶対差 ≤ 0.1。

{p07_md}

監査§7の3状態の own 差を表示精度内で再現した。
同じ3状態で生存 unit を1個ずつ選び、次課題 G_full の JVP を別経路の reverse-mode 内積で検証した。
最大相対誤差は {max(r['relative_error'] for r in verification['checks']):.3g}。これは全状態・全unitの独立検証ではない。

{md(['状態','own 計測数','own 適合数'],[[stage,len(rr := [r for r in P if r['stage']==stage]),frac(rr,lambda r:abs(r['fixed_next_minus_uniform'])<=.1)] for stage in ['init','end','u100']])}

## 登録予測の採点

数値範囲や閾値は対象の各保存課題末に適用した。全状態 AND は採点上の解釈であり、登録原文が一意に指定した集約規則ではない。
登録で例外率が指定された P0-3 だけはその割合条件を適用した。
複合予測は一つでも定量条件が破れれば ✗ とした。
定量基準のない節は後から閾値を作らず判定不能とした。

{md(['予測','判定','理由'],scores)}

{md(['腕','P0-2 範囲内','P0-4 早期適合','P0-4 後期適合','P0-5 適合','P0-7 own 再解析適合'],detailed)}

## 言えること／言えないこと

この解析は保存状態における head の整列と押しの符号を記述する。
各状態の κ の正負と補題の数値的成立を検証できる。
単一 seed の観測から seed 間の再現性は判断できない。
一様目標は新しい鎖に対する期待の近似である。
次課題の一実現との比較だけでその近似の一般的妥当性は証明できない。
符号一致率から各経路の大きさの分担は推定できない。
固定状態の帳簿から学習中の因果や実際の Adam 更新方向は結論しない。
既知の下見結果を独立な予測成功として数えない。
§3(B) の結果は変更していない。

## 成果物

- `table_head.csv`: 各状態の head 指標、帰無分位点、補題。
- `table_units.csv`: M の非 cap 腕は全層、他は最終 FFN 層の集計。
- `units_*.npz`: 全層の m・生存・G_own・自己形射影と、最終層 G_full・旧次課題 G_own・旧次課題生存。`p07_*` 配列は固定位置・固定集合の再計算値。
- `table_p07.csv`: 全状態の修正比較、旧比較、full 代表状態。
- `reanalyze_p07.log`: 全286状態の再計算記録。
- `checks.json` / `checks.log`: 計算前検査の数値と、登録・変更後基準の分離。
- `provenance.json`: 版、計算条件、入力と出力のハッシュ。
- `audit.json`: 状態数と保存ファイルの整合検査。
- `p07_verification.json`: 監査3状態の own 差再現と、次課題 full の reverse-mode 独立照合。

既存の計算成果物と入力のハッシュを照合したうえで、監査修正スクリプトを次の順で実行した。`measure.py` は元の比較を再現する履歴コードであり、主比較は `reanalyze_p07.py` が生成する。
`script_sha256` と計算前検査の hash は元の計測コードを指し、今回の追加計算コードは `p07_reanalysis.script_sha256` に記録した。

```sh
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/chargpt_load_fold_1021/load/reanalyze_p07.py
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/chargpt_load_fold_1021/load/verify_p07.py
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/chargpt_load_fold_1021/load/report.py
```

commit は作成していない。
'''
    (O/'summary_load.md').write_text(text)
    # Verify state coverage against actual model checkpoint files (exclude Adam).
    expected=[]; actual={(r['box'],r['arm'],int(r['task']),r['stage']) for r in H}
    inputs={}
    for b,a in M.ARMS:
        for p in sorted((M.directory(b,a)/'ckpts').glob('*.pt')):
            if p.stem.endswith('_adam'): continue
            s=p.stem
            t=0 if s=='init' else int(s.split('task')[1])
            stage='init' if s=='init' else 'u100' if s.startswith('u100') else 'end'
            expected.append((b,a,t,stage)); inputs[str(p)]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        for name in ['per_task.csv','provenance.json']:
            p=M.directory(b,a)/name; inputs[str(p)]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    assert set(expected)==actual, (set(expected)-actual,actual-set(expected))
    assert len(H)==len(actual)
    for r in H:
        s='_u100' if r['stage']=='u100' else ''
        p=O/f"units_{r['box']}_{r['arm']}_t{int(r['task'])}{s}.npz"
        with np.load(p) as z:
            assert z['G_full_last'].shape==(512,)
            pr=next(q for q in P if (q['box'],q['arm'],q['task'],q['stage'])==(r['box'],r['arm'],r['task'],r['stage']))
            alive=z['p07_fixed_alive_last']
            def rate(key): return float((z[key][alive]>0).mean()) if alive.any() else float('nan')
            assert np.isclose(pr['uniform_own_sink'],rate('p07_uniform_own_last'),equal_nan=True)
            assert np.isclose(pr['fixed_next_own_sink'],rate('p07_fixed_next_own_last'),equal_nan=True)
            assert np.isclose(pr['fixed_next_minus_uniform'],rate('p07_fixed_next_own_last')-rate('p07_uniform_own_last'),equal_nan=True)
            if pr['full_recomputed']:
                assert np.isclose(pr['fixed_next_full_minus_uniform'],rate('p07_fixed_next_full_last')-rate('p07_uniform_full_last'),equal_nan=True)
            assert all(np.isfinite(z[k]).all() for k in z.files)
        assert r['identity_rel']<=1e-10 and r['lemma_violation']==0
    assert all(r['closure_max']<=1e-10 for r in U)
    audit=dict(all_pass=True,head_rows=len(H),unit_rows=len(U),expected_states=len(expected),npz_files=len(list(O.glob('units_*.npz'))),complete_checkpoint_coverage=True,all_arrays_finite=True,all_states_head_identity_pass=True,all_states_position_closure_pass=True)
    audit['p07']=dict(own_states=len(P),full_representative_states=int(sum(r['full_recomputed'] for r in P)),per_unit_reaggregation_pass=True,independent_spot_checks_pass=verification['all_pass'],registered_verdict='not uniquely determined',reanalysis_own_verdict=verdict(p7),end_own_passed=sum(abs(r['fixed_next_minus_uniform'])<=.1 for r in u),end_own_total=len(u))
    (O/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    assert len(P)==len(H)==286 and sum(r['full_recomputed'] for r in P)==18
    pv['inputs']=inputs; pv['report_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    pv['verification_script_sha256']=hashlib.sha256(Path(__file__).with_name('verify_p07.py').read_bytes()).hexdigest()
    pv['outputs']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(O.iterdir()) if p.is_file() and p.name!='provenance.json'}
    (O/'provenance.json').write_text(json.dumps(pv,ensure_ascii=False,indent=2)+'\n')
    print(p07_md+'\n\n'+head_md+'\n\n'+unit_md+'\n\n'+md(['予測','判定','理由'],scores)+'\n\n'+json.dumps(audit))

if __name__=='__main__': main()

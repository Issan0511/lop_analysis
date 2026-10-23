#!/usr/bin/env python3
"""Additional report; keeps the primary results and labels intact."""
import csv,json
import measure as Q
from report import table,rng,ft
OUT,np=Q.OUT,Q.np

def load():
    rows=list(csv.DictReader((OUT/'headcov_states.csv').open()))
    for r in rows:
        for k,v in r.items():
            if v=='':r[k]=None
            else:
                try:r[k]=float(v)
                except ValueError:pass
    return rows

if __name__=='__main__':
    rows=load();tab=[];means=[]
    for arm in ['C3','ReLUcap']:
        for task in [9,39,100]:
            for cov in ['iid','ctx','empirical']:
                a=[r for r in rows if r['arm']==arm and r['task']==task and r['step']==10000 and r['metric']=='SGD' and r['covariance']==cov]
                tab.append([arm,task,cov,rng([r['positive'] for r in a]),rng([r['switch_rho'] for r in a]),rng([r['fit_rho'] for r in a]),str(sum(r['mean']>0 for r in a))+'/10'])
                means.append(dict(arm=arm,task=task,covariance=cov,positive_seed_count=sum(r['mean']>0 for r in a),mean=float(np.mean([r['mean'] for r in a])),
                         fraction_min=min(r['positive'] for r in a),fraction_max=max(r['positive'] for r in a)))
    ss=[]
    for arm in ['C3','ReLUcap']:
        for task in [39,100]:
            a=[r for r in rows if r['arm']==arm and r['task']==task and r['step']==10000 and r['metric']=='SGD' and r['covariance']=='iid']
            ss.append([arm,task,rng([r['rank'] for r in a],0),rng([r['head_fit_ratio'] for r in a]),rng([r['cov_cos_empirical'] for r in a]),rng([r['rho_empirical'] for r in a]),rng([r['rho_ctx'] for r in a])])
    body=['# 最終head全方向による相2の追補（探索的）','',
      '各unitの現在の読み出し方向を固定した振幅模型より、自由度を実headに近づけた検査。全64系列、両腕10seed×5保存状態、全512unit。最終LN後の特徴HのGramから、最終head全128方向を共同最小二乗fitする読み出しの共分散を計算した。L0 up行への逆伝播は全unitと後段を含む。', '',
      '- iid: 独立旧ラベルLS fitの期待。Sigma=(HᵀH)^+。',
      '- ctx: 同一文脈で共有される旧規則の期待。Sigma=G^+ Hᵀ C H G^+。',
      '- empirical: 保存された実読み出しのSigma=VcᵀVc。これは現logitの二乗ノルムを減らす全勾配。自己項だけを抜いたものではない。', '',
      table(['腕','課題末','読み出し共分散','沈めるunitの割合','ρ 実CE切替','ρ 現課題fit','unit平均が沈めるseed'],tab),'',
      '正のラベル分散とclass数の共通係数を省いたので、iid/ctxの絶対量をempiricalや実CEへ合わせていない。相関は追加の記録で、方向の定理を否定する必須基準には使わない。Adam固定分母はheadcov_states.csvに別記。','',
      table(['腕','課題末','有効rank','head共同fit残差比','iid/実Sigmaのcos','iid/実Sigma押しのρ','iid/ctx押しのρ'],ss),'',
      '共同fit残差はCEの必要条件の目安。CEから二乗損失への移し替えや、Hを旧ラベル平均の間固定する前提を保証しない。実網ではheadとembeddingが共有される。ここではL0 up行の微分の間embeddingを固定して、headを自由にfitする条件付き模型を測っている。','',
      f'検査: 実logit二乗lossの独立autogradとの相対差最大 {max(r["squared_autograd"] for r in rows if r["squared_autograd"] is not None):.3g}。容量の擬似logdetの有限差分との相対差 {max(r["capacity_fd"] for r in rows if r["capacity_fd"] is not None):.3g}。', '',
      '## Issaの指摘と判定の読み方', '',
      '「相2は量を当てられない」という指摘を反映し、主J2/J3の未達を相2の方向の主張への反証とは読まない。判定値・事前予測を後から変更しないため元の表は残す。相殺後の期待がどちらを向くか、実際の合計との橋渡しにどの仮定が残るか、量の予測ができるかは分ける。']
    more=[]
    for arm in ['C3','ReLUcap']:
        for task in [39,100]:
            for metric in ['SGD','Adam_fixed']:
                a=[r for r in rows if r['arm']==arm and r['task']==task and r['step']==10000 and r['metric']==metric and r['covariance']=='iid']
                more.append([arm,task,metric,rng([r['switch_agreement'] for r in a]),rng([r['switch_balanced_accuracy'] for r in a]),rng([r['switch_rho'] for r in a]),rng([r['fit_rho'] for r in a])])
    body+=['','## 実際の切替の合計との符号一致','',table(['腕','課題末','分母','符号一致','balanced accuracy','ρ switch','ρ 現課題fit'],more),'',
           '同じunitの実際の合計と比較している。C3課題100末のSGDでは全10seedでρ>0.5かつbalanced accuracy>0.65。これは探索的な別の模型の結果であり、元の振幅模型のJ2を書き換えない。', '',
           '## seed別・課題100末（iid、Adam固定分母）','',table(['腕','seed','ρ switch','符号一致','ρ 現課題fit'],[[r['arm'],int(r['seed']),ft(r['switch_rho']),ft(r['switch_agreement']),ft(r['fit_rho'])] for r in rows if r['task']==100 and r['step']==10000 and r['metric']=='Adam_fixed' and r['covariance']=='iid'])]
    if (OUT/'headcov_sensitivity.csv').exists():
        sr=list(csv.DictReader((OUT/'headcov_sensitivity.csv').open()))
        body+=['','## rank閾値の感度','',
          'rankが変わった4状態だけで1e-8/1e-12の押しも再計算した。他96状態はrank不変。', '',
          table(['seed','task/step','閾値','rank','ρ switch（SGD）','符号一致'],[[r['seed'],r['task']+'/'+r['step'],r['tolerance'],r['rank'],ft(float(r['switch_rho'])),ft(float(r['switch_agreement']))] for r in sr if r['metric']=='SGD' and r['covariance']=='iid'])]
    (OUT/'headcov_summary.md').write_text('\n'.join(body)+'\n')
    Q.dump(OUT/'headcov_direction.json',dict(exploratory=True,primary_judgments_unchanged=True,rows=means))
    print('\n'.join(body))

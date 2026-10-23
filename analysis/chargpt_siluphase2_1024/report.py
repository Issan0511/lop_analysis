#!/usr/bin/env python3
"""Report the preregistered SiLU applicability check, including failed matches."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/chargpt_siluphase2_1024'

def pct(v):return '未定義' if pd.isna(v) else f'{v*100:.1f}%'
def val(v):return '未定義' if pd.isna(v) else f'{v:.3f}'
def direction(v):return '沈める' if v>0 else '浮かせる' if v<0 else 'ゼロ'

def main():
    df=pd.read_csv(OUT/'states.csv')
    base=df[(df.zero_tol==1e-10)&(df.target=='r')&(df.predictor=='P2')]
    full=base[base.route=='full'];judge={}
    for metric in ['SGD','Adam_fixed']:
        dd=full[(full.arm=='SiLU')&(full.layer==2)&(full.metric==metric)&(full.step==10000)&(full.task.isin([50,100]))]
        scores={}
        for name,error in [('SWITCH_MATCH','switch_full'),('FIT_MATCH','fit_full')]:
            data=dd[dd.error==error].sort_values('task');assert len(data)==2
            passed=[bool(pd.notna(v) and v>=.65) for v in data.ba]
            scores[name]=dict(label='MATCHED' if all(passed) else 'PARTIAL' if any(passed) else 'NOT_MATCHED',n_pass=sum(passed),ba=[None if pd.isna(v) else float(v) for v in data.ba])
        fit=dd[dd.error=='fit_full'].sort_values('task')
        decreases=bool(fit.r_median.iloc[-1]<fit.r_median.iloc[0])
        scores['LATE_SINK']=dict(passed=bool((fit.pred_mean_all>0).all() and (fit.actual_mean_all>0).all() and decreases),
            r_decreases=decreases,pred_means=fit.pred_mean_all.tolist(),actual_means=fit.actual_mean_all.tolist())
        judge[metric]=scores
    predictions={'SWITCH_MATCH':.5,'FIT_MATCH':.2,'LATE_SINK':.2}
    scores={}
    for k,p in predictions.items():
        observed=judge['SGD'][k].get('passed',judge['SGD'][k].get('label')=='MATCHED')
        scores[k]=dict(probability=p,observed=observed,brier=(p-int(observed))**2)
    result=dict(judgments=judge,predictions=scores,scope='SiLU seed1000; saved-state one-step directions, not long-term causal identification')
    (OUT/'judge.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    report=['# SiLU L2の相2予測と実際の合計の照合','',
        '**L2が深く沈むのは課題50以降であり、課題50末は沈降前の比較点として読む。沈降した課題100末の切替では、相2予測がL3を浮かせL2を沈める両方の向きと対応した。50末での弱い対応を沈降期の相2説明への否定材料としない。長期の沈降死全体の原因同定は別に残る。**',
        '', '課題100末・保存Adam分母固定の全経路では、L2の符号一致94.7%、両方向的中率86.9%、Spearman .939。L3は99.1%、99.5%、.882。L2の自unit up行の更新だけでは平均は浮かせる側で、上流経路を足した合計は沈める側になる。この符号関係を予測・実測の双方が示した。単に最終FFNが誤差を増幅するという量の観察から推定したものではない。',
        '',
        '対象は最終FFN L3が正側へ出て、その一つ前L2が沈んだseed1000。SiLU 7状態、SiLUcap 4状態。保存状態だけ、学習更新0。10seedの一般性は測っていない。',
        '', '相2の全head共同fitから予測した勾配を、全パラメータ更新によるrの変化に運ぶ。実測も全勾配であり、他unit・注意・FFN・LN・上流のパラメータ更新を含む。正のSが沈める側（Δr=−ηS）。',
        '', '2026-10-24、Issa「L2は50課題以降沈みます」を受けて解釈を訂正。元の一様probeの保存ログではL2のr中央値は課題50/60/70/80/90/100末で −.603/−.604/−.636/−.988/−1.841/−2.388。50末ではまだ浅く、後半に深まる。50末→100末の相2の対応の変化はこの時間順序と整合的だが、相2の対応が立ち上がる正確な時点を測ったものではない。出典は既存quicklook/M/SiLUおよびSiLU_cont/seed1000/layer_stats.csv（uniform、L2、u=10000）。新しい勾配計算なし、判定・数値は不変更。',
        '', '## 登録判定','','| 分母 | 切替への対応 | 現課題fitへの対応 | 後期の合計が沈める側 |','|---|---|---|---|']
    for metric,j in judge.items():
        report.append(f'| {metric} | {j["SWITCH_MATCH"]["label"]} ({j["SWITCH_MATCH"]["n_pass"]}/2) | {j["FIT_MATCH"]["label"]} ({j["FIT_MATCH"]["n_pass"]}/2) | {j["LATE_SINK"]["passed"]} |')
    report+=['','対応の条件は課題50末・100末の両方でbalanced accuracy≥.65。登録判定は変更しない。ただし沈降前の50末と沈降後の100末の両方での一致を要求した判定であり、PARTIALを沈降期の説明が弱いという意味には読まない。旧ラベル平均の相2理論そのものの反証でもない。',
        '', '## SiLU L2・全経路の符号照合','',
        '| 保存点 | 分母 | 実勾配 | 共通非ゼロ数 | 相2の沈む割合 | 実測の沈む割合 | 符号一致 | 両方向的中率 | Spearman | 相2の全unit平均 | 実測の全unit平均 |',
        '|---|---|---|---|---|---|---|---|---|---|---|']
    for _,r in full[(full.arm=='SiLU')&(full.layer==2)&full.error.isin(['switch_full','fit_full'])].sort_values(['task','step','metric','error']).iterrows():
        report.append(f'| {int(r.task)}末' .replace('末', '更新100' if r.step==100 else '末')+
            f' | {r.metric} | {r.error} | {int(r.n)} | {pct(r.pred_sink)} | {pct(r.actual_sink)} | {pct(r.agreement)} | {pct(r.ba)} | {val(r.rho)} | {direction(r.pred_mean_all)} ({r.pred_mean_all:.3g}) | {direction(r.actual_mean_all)} ({r.actual_mean_all:.3g}) |')
    report+=['','沈む割合は応答ありかつ両者非ゼロの共通集合。末尾2列は全512unitの平均。BAは実測で沈む群・浮く群の的中率の平均で、各10unit未満なら未定義。係数の大きさは模型間で比較せず、平均の符号を見る。',
        '', '## 全経路・自分のup行・上流の違い（SiLU、課題100末）','',
        '| 層 | 分母 | 経路 | 実勾配 | 符号一致 | 両方向的中率 | ρ | 相2平均 | 実測平均 |','|---|---|---|---|---|---|---|---|---|']
    sub=base[(base.arm=='SiLU')&(base.task==100)&(base.step==10000)&base.error.isin(['switch_full','fit_full'])]
    for _,r in sub.sort_values(['layer','metric','route','error']).iterrows():
        report.append(f'| L{r.layer} | {r.metric} | {r.route} | {r.error} | {pct(r.agreement)} | {pct(r.ba)} | {val(r.rho)} | {direction(r.pred_mean_all)} | {direction(r.actual_mean_all)} |')
    report+=['','## 状態の推移（SGD等と無関係な前活性）','','| 腕 | 課題 | 更新 | 層 | r中央値 | s中央値 | 応答ありunit | 平均absゲート |','|---|---|---|---|---|---|---|---|']
    states=full[(full.metric=='SGD')&(full.error=='switch_full')].sort_values(['arm','task','step','layer'])
    for _,r in states.iterrows():report.append(f'| {r.arm} | {int(r.task)} | {int(r.step)} | L{int(r.layer)} | {r.r_median:.3f} | {r.s_median:.3f} | {int(r.n_responsive)} | {r.abs_gate_mean:.4f} |')
    report+=['','## 対照・数値感度','','| 腕 | 層 | 分母 | head模型 | 実勾配経路 | BA 課題50末/100末 |','|---|---|---|---|---|---|']
    sub=df[(df.target=='r')&(df.route=='full')&(df.zero_tol==1e-10)&(df.step==10000)&df.task.isin([50,100])]
    for group,data in sub.groupby(['arm','layer','metric','predictor','error']):
        report.append('| '+' | '.join(map(str,group))+' | '+' / '.join(pct(x) for x in data.sort_values('task').ba)+' |')
    sensitivity=[]
    keys=['arm','task','step','layer','metric','target','route','predictor','error']
    for group,data in df.groupby(keys):
        if data.n.nunique()>1 or data.ba.dropna().nunique()>1:
            sensitivity.append(dict(zip(keys,group),n_min=int(data.n.min()),n_max=int(data.n.max()),ba_min=None if data.ba.isna().all() else float(data.ba.min()),ba_max=None if data.ba.isna().all() else float(data.ba.max())))
    (OUT/'zero_sensitivity.json').write_text(json.dumps(sensitivity,ensure_ascii=False,indent=2)+'\n')
    report+=['',f'数値ゼロ閾値1e−8/1e−10/1e−12で対象数またはBAが変わる比較は{len(sensitivity)}。詳細zero_sensitivity.json。固有値閾値でrankが変わった状態はP2tolを同じ表へ全て追加。',
        '', '## 検査・限定','',
        '課題10/100末で、全パラメータのr方向を有限差分と照合。相2自体も半log擬似detの有限差分で検査。gate.jsonに刻み別の誤差を保存。全状態で保存layer_statsのr/sを照合し、body戻し＋共有headの直接勾配＝実CEの合計を検算。課題100更新100は元ログに同一保存点の比較行がないためreference_available=falseを明記。',
        '相2模型では自由headをfitするが、実網はheadとembeddingを共有する。switch_body/fit_bodyは直接head更新を除いた対照であり、fullと取り違えない。',
        '切替の独立新ラベル平均と現課題fitを分けた。保存点の一歩が一致しても、Adam運動量・課題の胴・幅成長を通した長期の沈降死の因果説明にはならない。L3の正側化が必ず起きるという主張でもない。',
        'SiLUcap対照のAdam分母固定は、capの更新後射影を加える前の勾配方向。capの実更新そのものではなく、救済の媒介率とは読まない。',
        '', '## 事前予測の採点','']
    for k,r in scores.items():report.append(f'- {k}: 予測{r["probability"]:.2f}、観測{r["observed"]}、Brier={r["brier"]:.3f}。')
    report+=['','## 置き場','','spec登録87cbfd4a、実装62d059d8。code/results: chargpt_siluphase2_1024。生配列・ログはbackup_manifest.jsonを参照。入力checkpointは他worktreeから読取のみ。']
    (OUT/'summary.md').write_text('\n'.join(report)+'\n')
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for row,metric in enumerate(['SGD','Adam_fixed']):
        for col,error in enumerate(['switch_full','fit_full']):
            ax=axes[row,col]
            with np.load(OUT/'raw/SiLU_t100_u10000.npz') as a:
                pred=a[f'L2_{metric}_r_full_P2'];actual=a[f'L2_{metric}_r_full_{error}'];responsive=a['L2_responsive']
            scales=[max(np.max(np.abs(v)),1e-300) for v in [pred,actual]]
            ax.scatter(pred[responsive]/scales[0],actual[responsive]/scales[1],s=13,alpha=.65,color='#2474a5')
            ax.scatter(pred[~responsive]/scales[0],actual[~responsive]/scales[1],s=13,alpha=.5,color='#a0a5aa',marker='x')
            ax.axhline(0,color='#555',lw=.7);ax.axvline(0,color='#555',lw=.7)
            ax.set(xlim=(-1.07,1.07),ylim=(-1.07,1.07),xlabel='相2予測（正＝沈む）',ylabel='実際の合計（正＝沈む）')
            data=full[(full.arm=='SiLU')&(full.task==100)&(full.step==10000)&(full.layer==2)&(full.metric==metric)&(full.error==error)].iloc[0]
            ax.set_title(f'{metric} / '+('切替' if error=='switch_full' else '現課題fit')+f'：両方向的中率 {pct(data.ba)}',loc='left',fontsize=11)
            ax.spines[['top','right']].set_visible(False)
    fig.suptitle('SiLU・L2：相2は実際の合計と同じ向きか',x=.08,ha='left',fontsize=19,weight='bold')
    fig.text(.08,.915,'seed1000・課題100末・全512unit　全パラメータ経路（上流の更新も含む）',fontsize=11)
    fig.text(.08,.025,'各軸は最大絶対値で規格化。量の一致を主張しない。青＝応答あり、灰＝max|φ′|≤1e−6。右上・左下が符号一致。',fontsize=9)
    fig.subplots_adjust(top=.85,bottom=.10,left=.09,right=.98,hspace=.35,wspace=.25)
    for ext in ['png','pdf']:fig.savefig(OUT/f'silu_L2_full_t100.{ext}',dpi=180,facecolor='white')
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for row,metric in enumerate(['SGD','Adam_fixed']):
        for col,layer in enumerate([2,3]):
            ax=axes[row,col]
            with np.load(OUT/'raw/SiLU_t100_u10000.npz') as a:
                pred=a[f'L{layer}_{metric}_r_full_P2'];actual=a[f'L{layer}_{metric}_r_full_switch_full'];responsive=a[f'L{layer}_responsive']
            scales=[max(np.max(np.abs(v)),1e-300) for v in [pred,actual]]
            ax.scatter(pred[responsive]/scales[0],actual[responsive]/scales[1],s=13,alpha=.65,color='#2474a5' if layer==2 else '#b77b39')
            ax.scatter(pred[~responsive]/scales[0],actual[~responsive]/scales[1],s=13,alpha=.5,color='#a0a5aa',marker='x')
            ax.axhline(0,color='#555',lw=.7);ax.axvline(0,color='#555',lw=.7)
            ax.set(xlim=(-1.07,1.07),ylim=(-1.07,1.07),xlabel='相2予測（正＝沈む）',ylabel='切替の実際の合計（正＝沈む）')
            data=full[(full.arm=='SiLU')&(full.task==100)&(full.step==10000)&(full.layer==layer)&(full.metric==metric)&(full.error=='switch_full')].iloc[0]
            ax.set_title(f'L{layer} / {metric}：符号一致 {pct(data.agreement)}・BA {pct(data.ba)}',loc='left',fontsize=11)
            ax.spines[['top','right']].set_visible(False)
    fig.suptitle('相2予測：L2は沈む側、L3は浮く側',x=.08,ha='left',fontsize=19,weight='bold')
    fig.text(.08,.915,'SiLU・seed1000・課題100末　全パラメータ経路で実際の切替の合計と照合',fontsize=11)
    fig.text(.08,.025,'各軸は最大絶対値で規格化。色＝応答あり、灰＝max|φ′|≤1e−6。右上・左下が符号一致。長期の原因同定ではない。',fontsize=9)
    fig.subplots_adjust(top=.85,bottom=.10,left=.09,right=.98,hspace=.35,wspace=.25)
    for ext in ['png','pdf']:fig.savefig(OUT/f'silu_L2_L3_switch_t100.{ext}',dpi=180,facecolor='white')
    plt.close(fig)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

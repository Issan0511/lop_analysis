#!/usr/bin/env python3
"""Fixed operational judgments and figures for the early matched comparison."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/chargpt_earlyphase_1024'
COLORS = {'p1':'#cb792d', 'p2':'#2474a5'}
GROUP = ['arm','seed','layer','metric','target','error','family']


def p2win(row):
    return bool(row['p2_ba'] >= .65 and row['delta_ba'] >= .05)


def p1win(row):
    return bool(row['p1_ba'] >= .65 and row['delta_ba'] <= -.05)


def both(row):
    return bool(row['p1_ba'] >= .65 and row['p2_ba'] >= .65 and abs(row['delta_ba']) < .05)


def windows(frame):
    rows=[]
    for group, data in frame.groupby(GROUP):
        base=dict(zip(GROUP,group))
        for window, tasks in [('first',[1]),('early',[1,2,3]),('middle',[4,5,6]),('late',[7,8,9])]:
            sub=data[data.task.isin(tasks)]
            assert len(sub)==len(tasks)
            row=dict(base,window=window,n_common_min=int(sub.n_common.min()))
            for key in ['p1_ba','p2_ba','delta_ba','p1_agreement','p2_agreement','p1_rho','p2_rho']:
                row[key]=float(sub[key].median()) if sub[key].notna().all() else np.nan
            row.update(p1_win=p1win(row),p2_win=p2win(row),both_match=both(row))
            rows.append(row)
    return pd.DataFrame(rows)


def judgments(win):
    rows=[]
    keys=[k for k in GROUP if k!='seed']
    for group,data in win.groupby(keys):
        counts=dict(FROM_FIRST_SAVED=0,EARLY_CROSSOVER=0,BOTH_MATCH=0)
        seed_rows=[]
        for seed in range(10):
            sd=data[data.seed==seed].set_index('window')
            first=bool(sd.loc['first','p2_win'] and sd.loc['late','p2_win'])
            cross=bool(sd.loc['early','p1_win'] and sd.loc['late','p2_win'])
            match=bool(sd.loc['early','both_match'] and sd.loc['late','both_match'])
            counts['FROM_FIRST_SAVED']+=int(first)
            counts['EARLY_CROSSOVER']+=int(cross)
            counts['BOTH_MATCH']+=int(match)
            seed_rows.append(dict(seed=seed,from_first=first,crossover=cross,both_match=match))
        label=next((name for name in ['FROM_FIRST_SAVED','EARLY_CROSSOVER','BOTH_MATCH'] if counts[name]>=8),'MIXED_OR_UNRESOLVED')
        rows.append(dict(zip(keys,group),label=label,counts=counts,seeds=seed_rows))
    return rows


def fmt(series,percent=False):
    series=series.dropna()
    if series.empty:return '未判定'
    return f'{100*series.min():.1f}–{100*series.max():.1f}%' if percent else f'{series.min():.3f}–{series.max():.3f}'


def figure(frame,layer,family):
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for i,arm in enumerate(['C3','ReLUcap']):
        for j,metric in enumerate(['SGD','Adam_fixed']):
            ax=axes[i,j]
            data=frame[(frame.arm==arm)&(frame.layer==layer)&(frame.metric==metric)&
                       (frame.target=='r')&(frame.error=='switch')&(frame.family==family)]
            for name,label in [('p1','相1型：整列＋正答極限'),('p2','相2型：共同fit')]:
                stats=data.groupby('task')[name+'_ba'].agg(['min','median','max'])*100
                ax.fill_between(stats.index,stats['min'],stats['max'],color=COLORS[name],alpha=.16,lw=0)
                ax.plot(stats.index,stats['median'],marker='o',ms=4,color=COLORS[name],label=label,lw=2)
            ax.axhline(50,color='#9ba5ac',ls='--',lw=1)
            ax.set_title(f'{arm} / '+('Adam分母固定' if metric=='Adam_fixed' else 'SGD方向'),loc='left',fontsize=12)
            ax.set(xticks=range(1,10),ylim=(0,100),yticks=[0,25,50,75,100])
            ax.grid(axis='y',alpha=.15)
            ax.spines[['top','right']].set_visible(False)
            if j==0:ax.set_ylabel('両方向の的中率の平均 (%)')
            if i==1:ax.set_xlabel('課題末')
    fig.suptitle(f'L{layer}：相1型・相2型を同じ切替の合計へ照合',x=.08,ha='left',fontsize=19,weight='bold')
    fig.text(.08,.91,'線：10 seedの中央値　帯：seedの最小〜最大　'+('独立旧ラベル模型' if family=='iid' else '同じ文脈を共有する旧ラベル模型'),fontsize=11)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.54,.89),ncol=2,frameon=False)
    fig.text(.08,.035,'相1・相2・実測が全て非ゼロの共通unitで比較。全後段の微分を含む。模型への適合であり、相の因果同定ではない。',fontsize=9)
    fig.subplots_adjust(top=.80,bottom=.12,left=.09,right=.98,hspace=.26,wspace=.12)
    for ext in ['png','pdf']:fig.savefig(OUT/f'phase_comparison_L{layer}_{family}.{ext}',dpi=180,facecolor='white')
    plt.close(fig)


def main():
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'pdf.fonttype':42})
    comp=pd.read_csv(OUT/'comparisons.csv')
    states=pd.read_csv(OUT/'states.csv')
    assert len(comp)==5760
    win=windows(comp)
    win.to_csv(OUT/'windows.csv',index=False)
    judges=judgments(win)
    primary=next(r for r in judges if all(r[k]==v for k,v in dict(arm='C3',layer=0,metric='SGD',target='r',error='switch',family='iid').items()))
    probs={'FROM_FIRST_SAVED':.55,'EARLY_CROSSOVER':.15,'other':.30}
    observed=primary['label'] if primary['label'] in probs else 'other'
    score=sum((p-int(name==observed))**2 for name,p in probs.items())
    (OUT/'judge.json').write_text(json.dumps(dict(primary=primary,all=judges,predictions=probs,
                                 observed=observed,multiclass_brier_sum=score),ensure_ascii=False,indent=2)+'\n')
    # The two arms have identical trained parameters at task1; require matching measurements.
    equal=[]
    for seed in range(10):
        with np.load(OUT/'raw'/f'C3_seed{seed}_task1.npz') as a,np.load(OUT/'raw'/f'ReLUcap_seed{seed}_task1.npz') as b:
            assert a.files==b.files
            worst=max(float(np.max(np.abs(a[k].astype(float)-b[k].astype(float)))) for k in a.files)
            assert worst<1e-10,(seed,worst)
            equal.append(worst)
    gate=json.loads((OUT/'gate.json').read_text())
    refs=[v for r in gate['checks'] for k,v in r.items() if k.endswith('_reference')]
    checks=json.loads((OUT/'checks.json').read_text())
    decomp=[v for r in checks for k,v in r.items() if k.endswith('_decomposition')]
    sensitivity=[]
    for (arm,seed,task),data in states.groupby(['arm','seed','task']):
        base=data.iloc[0]
        if base.rank_tol1e8==base['rank'] and base.rank_tol1e12==base['rank']:continue
        with np.load(OUT/'raw'/f'{arm}_seed{seed}_task{task}.npz') as a:
            for layer in [0,1]:
                for metric in ['SGD','Adam_fixed']:
                    v=a[f'L{layer}_{metric}_r_P2']
                    for tol in ['1e-08','1e-12']:
                        key=f'L{layer}_{metric}_r_P2tol{tol}'
                        if key not in a:continue
                        u=a[key];mask=a[f'L{layer}_alive']&(v!=0)&(u!=0)
                        sensitivity.append(dict(arm=arm,seed=seed,task=task,layer=layer,metric=metric,tol=tol,
                            n_common=int(mask.sum()),sign_agreement=float((np.sign(v[mask])==np.sign(u[mask])).mean())))
    pd.DataFrame(sensitivity,columns=['arm','seed','task','layer','metric','tol','n_common','sign_agreement']).to_csv(OUT/'sensitivity.csv',index=False)
    finalchecks=dict(task1_arm_equal_max=max(equal),legacy_gate_max=max(refs),decomposition_max=max(decomp),
                     n_states=180,n_comparisons=len(comp),n_sensitivity_rows=len(sensitivity))
    audit=json.loads((OUT/'early_fd_audit.json').read_text())
    assert audit['passed']
    finalchecks['post_result_early_fd_max']=max(r['relative_max_error'] for r in audit['checks'])
    (OUT/'final_checks.json').write_text(json.dumps(finalchecks,indent=2)+'\n')
    for layer in [0,1]:
        for family in ['iid','ctx']:figure(comp,layer,family)
    mainf=comp[(comp.arm=='C3')&(comp.layer==0)&(comp.metric=='SGD')&(comp.target=='r')&
               (comp.error=='switch')&(comp.family=='iid')]
    report=['# 課題1〜9の相1型・相2型比較','',f'主判定: **{primary["label"]}**。C3 L0、SGD、自unit up行を通るrの押し。',
            '', '相2型の予測が合い始める時点はC3 L0の課題5〜6に見える。課題1〜4の順位相関は全seedで負、課題6以降は全seedで相2優位の点ごとの基準を満たす。しかし前半で相1型が十分当たっていないため、相1から相2への交代とは同定しない。最初の保存点から相2優位という事前の読みも支持されない。',
            '','| 判定 | 該当seed | 基準 |','|---|---|---|']
    for name,count in primary['counts'].items():report.append(f'| {name} | {count}/10 | 8/10以上 |')
    report+=['','相2優位はB2≥0.65かつB2−B1≥0.05。相1優位はB1≥0.65かつB2−B1≤−0.05。Bは実測で沈む群と浮く群の的中率の平均。相の存在・不在の判定ではない。',
             '', '## 全課題の主比較（10seedの範囲）','',
             '| 課題末 | 共通unit数 | 相1 B | 相2 B | B2−B1 | 相1ρ | 相2ρ |','|---|---|---|---|---|---|---|']
    for task,data in mainf.groupby('task'):
        report.append(f'| {task} | {data.n_common.min()}–{data.n_common.max()} | '+
                      ' | '.join(fmt(data[k],k in ['p1_ba','p2_ba','delta_ba']) for k in ['p1_ba','p2_ba','delta_ba','p1_rho','p2_rho'])+' |')
    report+=['','## seedごとの窓中央値（主比較）','','| seed | 課題1 B1/B2 | 課題1〜3 B1/B2 | 課題7〜9 B1/B2 | 最初から | 交代 |','|---|---|---|---|---|---|']
    w=win[(win.arm=='C3')&(win.layer==0)&(win.metric=='SGD')&(win.target=='r')&(win.error=='switch')&(win.family=='iid')]
    for seed in range(10):
        sd=w[w.seed==seed].set_index('window');jr=primary['seeds'][seed]
        vals=[f'{sd.loc[n,"p1_ba"]:.3f}/{sd.loc[n,"p2_ba"]:.3f}' for n in ['first','early','late']]
        report.append('| '+ ' | '.join([str(seed)]+vals+[str(jr['from_first']),str(jr['crossover'])])+' |')
    report+=['','## 腕・層・Adam・文脈対照','','| 腕 | 層 | 分母 | ラベル模型 | 判定 | 最初から / 交代 / 両方 |','|---|---|---|---|---|---|']
    for row in judges:
        if row['target']=='r' and row['error']=='switch':
            report.append(f'| {row["arm"]} | L{row["layer"]} | {row["metric"]} | {row["family"]} | {row["label"]} | '+
                          ' / '.join(str(row['counts'][k]) for k in ['FROM_FIRST_SAVED','EARLY_CROSSOVER','BOTH_MATCH'])+' |')
    report+=['','## 条件の診断（C3、10seed範囲）','','| 課題末 | p最大の平均 | q最大の平均 | KL(q||p) | head fit残差比 | head整列cos² | LN γのCV |','|---|---|---|---|---|---|---|']
    ss=states[(states.arm=='C3')&(states.layer==0)&(states.metric=='SGD')&(states.target=='r')&(states.predictor=='P2')]
    for task in [1,3,6,9]:
        data=ss[ss.task==task]
        report.append(f'| {task} | '+' | '.join(fmt(data[k]) for k in ['pmax','qmax','KL','head_fit_ratio','cos2_head_q','final_gamma_cv'])+' |')
    report+=['','## 従来の局所自己形と装填診断（C3 L0、SGD r）','',
             '各予測と実測が非ゼロの集合。主のP1/P2共通集合とは別。load_actualは実誤差を保った装填部分、load_qは誤差をq−uへ替えた診断。', '',
             '| 課題末 | 予測 | B（10seed範囲） | ρ（10seed範囲） |','|---|---|---|---|']
    for task in [1,3,6,9]:
        for name in ['local_self','load_actual','load_q']:
            data=states[(states.arm=='C3')&(states.layer==0)&(states.metric=='SGD')&(states.target=='r')&(states.task==task)&(states.predictor==name)]
            report.append(f'| {task} | {name} | {fmt(data.switch_ba,True)} | {fmt(data.switch_rho)} |')
    report+=['','## 検査・読みの範囲','',
             f'課題9末の既存配列との最大相対差 {max(refs):.3g}。全180状態の装填+残り=合計の最大相対差 {max(decomp):.3g}。課題1末の両腕の配列差最大 {max(equal):.3g}。',
             'P1・実CE・局所自己形の独立autograd、P2とrの有限差分、小行列のラベル平均をgateで確認。ReLUcap L1の最初の粗い有限差分は失敗し、本走前に刻みを小さくして収束を確認した。閾値は不変更。',
             f'予想外の初期の逆相関を受け、C3 seed0の課題1/4/6、固定6unitでP1・P2・実CEの各方向を独立に有限差分した（事後の数値検査）。符号は全件一致、最大相対差 {finalchecks["post_result_early_fd_max"]:.3g}。パラメータは元に戻し、学習はしていない。判定・予測器は不変更。',
             f'rank変化の感度行は {len(sensitivity)}。sensitivity.csvに全て記録。主閾値1e−10は変更していない。',
             '相1型は最終head特徴HにH1を置いた条件付き予測JᵀH。素のFFN自己形と異なりLNの微分も通す。相2型は同じHに共同LSを置いたJᵀHG⁺。H1が実状態で成立するか、旧ラベル平均中にHを固定できるかを、この比較だけで保証しない。',
             '課題1末は既に10000更新後。そこでの優位は初期化直後の状態や課題1内の移行時点を特定しない。「異様に早い」をMLPとの更新数を揃えた比較なしには言わない。',
             '主に測ったのは現課題入力上の新ラベル平均の切替の一歩。全パラメータ更新、Adam運動量、長期沈降の因果を同定したものではない。fit、μへの押し、文脈対照は全てCSVに残した。',
             '', '## 予測の採点','', f'事前予測 J1=.55 / J2=.15 / その他=.30。観測分類 {observed}。多クラスBrier（3項の和）={score:.4f}。',
             '', '## 保存','', 'spec初回commit aeca78aa、実装・数値検査追記842fe843。学習更新0。code/results: chargpt_earlyphase_1024。生配列・ログの所在はbackup_manifest.json。']
    (OUT/'summary.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(dict(primary=primary,checks=finalchecks),ensure_ascii=False,indent=2))


if __name__=='__main__':main()

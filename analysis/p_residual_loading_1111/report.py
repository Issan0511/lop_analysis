"""Human-readable synthesis and focus figure; requires the complete registered summary."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.p_residual_loading_1111 import summarize as S
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/p_residual_loading_1111'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,default=OUT/'raw');args=ap.parse_args()
    raw=args.raw
    if not raw.exists():
        raw=Path('/home/issan/Projects/obsidian-research-data/p_residual_loading_1111/results/p_residual_loading_1111/raw')
    assert raw.exists(), str(raw)
    v=json.loads((OUT/'verdict.json').read_text());assert v['main_design_complete']
    seed=pd.read_csv(OUT/'seed_metrics.csv');ct=pd.read_csv(OUT/'contrasts.csv')
    def subset(df,key):
        x=df
        for col,val in zip(S.CELL,key):x=x[x[col]==val]
        return x
    def contrast(key,which='parameter_clear',metric='U'):
        x=subset(ct,key);x=x[(x.kind=='factorial')&(x.contrast==which)&(x.metric==metric)];assert len(x)==1
        return x.iloc[0]
    def effect(key,which='parameter_clear',metric='U',scale=100):
        x=contrast(key,which,metric);assert x.estimable
        return f'{x.estimate*scale:+.2f} [{x.ci_low*scale:+.2f}, {x.ci_high*scale:+.2f}]'
    def arms(key):return subset(seed,key).groupby('arm')[['U','accuracy_end','ce_end']].mean()
    def reach(key,arm,field='T90_first_elapsed'):
        q=subset(seed,key);q=q[q.arm==arm][field];n=int(q.notna().sum())
        return f'{n}/{len(q)}到達、平均{q.mean():.0f}' if n==len(q) else f'{n}/{len(q)}到達、全seed平均は未同定'
    keys=[('mlp','ELU',10,'S','P',0),('vit','SL',100,'S','P',0),('mlp','ELU',29,'Pall','P',0)]
    labels=['ELU・通常履歴の課題10末','ViT SL・課題100末','ELU・Pall履歴の課題29末']
    L=['# P後の読み出し依存費用とViT装填費用：直接比較', '',
       '**MLP側の「P後にも読み出しを外すと軽くなる費用」は確認された。一方、ViTとP後の残り方まで同じだとは確定できなかった。**', '',
       f'登録したoperational判定：**{v["operational_commonality_verdict"]}**。全820論理曲線（690本を実行、130本は登録済みの完全一致共有）、MLP各10seed、ViT5seedを完了。', '',
       '問いは、Pで特定の整列を壊した後にも残り、読み出しを外すと小さくなる適応費用が、MLPとViTに共通するか。主比較では両群のhead Adamを空にそろえ、読み出しのweight/biasを残すKCとゼロにするZCだけを変えた。分岐時のbody parametersとbody Adamは完全一致。bodyはその後も学習する。', '',
       'T90はここでは初回の記録到達。最後まで記録上90%以上を保つsustained T90はsummary.mdとCSVへ別保存。主指標Uは固定学習集合の精度不足1−accuracyを、残り区間で台形積分して期間で割ったもの。下表の正値は、読み出しを外すことで期間平均の精度が何percentage points改善したかを示す。異なるモデルの更新数や効果量の同等性は検定していない。', '',
       '| 条件（いずれもP後） | KC−ZC：平均精度の改善 pp [95% CI] | 終点精度 KC → ZC | 90%到達：KC | 90%到達：ZC |',
       '|---|---:|---:|---|---|']
    for key,label in zip(keys,labels):
        a=arms(key);L.append(f'| {label} | {effect(key)} | {100*a.loc["KC","accuracy_end"]:.2f}% → {100*a.loc["ZC","accuracy_end"]:.2f}% | {reach(key,"KC")} | {reach(key,"ZC")} |')
    L += ['', '## SとPでの残り方', '',
          '| 条件 | S直後 KC−ZC pp | P直後 KC−ZC pp | P/S比 [95% CI] |', '|---|---:|---:|---:|']
    for key,label in zip(keys[:2],labels[:2]):
        sk=(*key[:4],'S',0);rk=(*key[:4],'P/S',0)
        rr=subset(ct,rk);assert len(rr)==1 and bool(rr.iloc[0].estimable)
        r=rr.iloc[0]
        L.append(f'| {label} | {effect(sk)} | {effect(key)} | {r.estimate:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}] |')
    L += ['', '後期ELUはP後にもSと近い大きさの操作差が残った。ViTはSでは正の費用が明瞭だが、P後の点推定は約2割になり、区間は0をまたぐ。この比は登録したhead除去効果の比であり、全遅れの因果的な残存率や効果量の同等性検定ではない。', '',
          'ViTでもhead Adamを保持する条件KK−ZKは正である。従って「ViTにはP後のhead依存成分がない」とは言えない。head Adamを消すと不確かになり、さらに25更新後の除去では符号が反転するという条件依存性が、ELUとの共通化を制限する。', '']
    L += ['', '更新予算はMLP4000、ViT1500。平均T90は全seed到達時だけ表示した。未到達を上限に代入していない。CIは対応seed bootstrap10000回、百分位95%、多重比較未調整。主要MLP条件の選択は先行データを見て行っており、既知KK/ZC再現と新規factorialを区別する。', '',
          '## 読み出しの効果とAdam状態の効果', '',
          '| 主条件 | parameters効果：Adamを消す KC−ZC | parameters効果：Adamを残す KK−ZK | Adam保持効果：head保持 KK−KC | Adam保持効果：headゼロ ZK−ZC |',
          '|---|---:|---:|---:|---:|']
    for key,label in zip(keys,labels):L.append('| '+label+' | '+' | '.join(effect(key,c) for c in ['parameter_clear','parameter_keep','memory_keep','memory_zero'])+' |')
    L += ['', '単位はUの差×100。parameters効果の正値は読み出しを外す利益、Adam保持効果の負値はAdamを残す利益。head Adamには一次・二次momentとstep counterを含む。正のKC−ZCは、head Adamリセットの差だけでは説明できない。これは読み出しweightとbiasを合わせた操作であり、両者の寄与は分離していない。', '',
          '## Pから25更新後にも残るか', '',
          '| 条件 | τ0 KC−ZC pp | τ25 KC−ZC pp |', '|---|---:|---:|']
    for key,label in zip(keys[:2],labels[:2]):
        delayed=(*key[:5],25);L.append(f'| {label} | {effect(key)} | {effect(delayed)} |')
    L += ['', 'τ25はP-KKの最初の25更新から共通分岐し、同じbatch列の25番目以降を続けた。最終更新数は同じ。異なるfork状態と積分区間なので、この差を25更新の媒介率や不可逆な体損傷へ読み替えない。Sのτ25対照は実施していない。ゼロ化は25更新で得た新しい読み出しも消すため、符号反転だけから初期の費用が完全に消えたとは言えない。', '',
          '## 一般化の境界', '',
          '| MLP条件 | P後τ0 KC−ZC pp | P後τ25 KC−ZC pp |', '|---|---:|---:|']
    for act in ['R','ELU','SILU']:
        for t in [1,10]:
            k=('mlp',act,t,'S','P',0);L.append(f'| {act} 課題{t}末 | {effect(k)} | {effect((*k[:5],25))} |')
    L += ['', '初期3活性化では読み出しを残す方が速い。後期R/SiLUはτ0で小さい正効果があるが、どの腕も4000更新内に90%へ到達していない。後期SiLU τ25は期間平均では保持側、終点ではゼロ側が良く、曲線が後半に交差する。単一の時点だけで「遅れが消えた」とまとめない。', '',
          'Pallの比較はPを毎課題繰り返した同じ前歴内のfactorialである。読み出し除去後にも90%未到達が残る。SとPallの予防効果、全残差の媒介割合、永久的不能は検定していない。', '',
          '## 結論をどこまで読めるか', '',
          ('支持されるのは「特定の整列をPで崩しても、持ち越した読み出しparametersに依存する適応費用が残り、その読み出しを外すと軽くなる」という操作上の共通性である。' if v['operational_commonality_verdict']=='supported' else '登録した両主要条件での共通性は支持条件を満たしていない。判定と各区間を優先して解釈する。')+'Pは全旧情報を消す操作ではない。MLPとViTの微視的な勾配機構、全ての残差、全ての学習段階が同じだとは結論しない。', '',
          '全結果：summary.md、seed_metrics.csv、contrasts.csv、verdict.json。詳しいMLP境界：mlp_analysis.md。登録：specs/spec_p_residual_loading_1111.md。実装commit0c874fe9、事前登録commit359d0453。再現チェックと入力SHAはimplementation_notes.md/provenance.json、生データ退避先はbackup_manifest.json。', '',
          '![主比較とPallの学習曲線](focus_curves.png)', '']
    (OUT/'synthesis.md').write_text('\n'.join(L))
    # Standalone scientific figure: only the parameter contrast with identical fresh head Adam.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(13.2,4.3))
    allraw=S.load_records(sorted(raw.glob('*.json')))
    titles=['MLP ELU | task end 10\nS history, n = 10','ViT SL | task end 100\nS history, n = 5','MLP ELU | task end 29\nRepeated-P history, n = 10']
    for ax,key,title in zip(axs,keys,titles):
        for arm,color,label in [('KC','#d97706','Readout retained'),('ZC','#007c91','Readout zeroed')]:
            rr=[r for r in allraw if tuple(r[c] for c in S.CELL)==key and r['arm']==arm]
            y=np.array([[x['acc'] for x in r['curve']] for r in rr]);steps=[x['step'] for x in rr[0]['curve']]
            ax.plot(steps,y.mean(0),color=color,label=label,lw=2.)
            ax.fill_between(steps,y.mean(0)-y.std(0,ddof=1),y.mean(0)+y.std(0,ddof=1),color=color,alpha=.13,lw=0)
        ax.axhline(.9,color='#888',linestyle=':',lw=1);ax.set_ylim(0,1.025);ax.set_title(title,fontsize=10)
        ax.set_xlabel('Updates after P and task switch');ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
        x=contrast(key);ax.text(.98,.04,f'Mean accuracy gain: {100*x.estimate:+.1f} pp\n95% CI [{100*x.ci_low:.1f}, {100*x.ci_high:.1f}]',ha='right',va='bottom',transform=ax.transAxes,fontsize=9,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
    axs[0].set_ylabel('Full training-set accuracy')
    h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,ncol=2,loc='upper center',bbox_to_anchor=(.5,.99),frameon=False)
    fig.text(.5,.015,'Both arms clear head Adam and retain body Adam at the fork. Lines: seed means; bands: seed SD. Dotted line: 90%.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.06,right=.99,bottom=.15,top=.81,wspace=.23)
    for ext in ['png','pdf']:fig.savefig(OUT/f'focus_curves.{ext}',dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9.4,4.2),sharey=True)
    for ax,key,title,color in zip(axs,keys[:2],['MLP ELU | task end 10, n = 10','ViT SL | task end 100, n = 5'],['#6f4c9b','#007c91']):
        kk=[(*key[:4],'S',0),key,(*key[:5],25)]
        rr=[contrast(k) for k in kk];mu=np.array([r.estimate for r in rr])*100
        lo=np.array([r.ci_low for r in rr])*100;hi=np.array([r.ci_high for r in rr])*100
        ax.errorbar(range(3),mu,yerr=[mu-lo,hi-mu],fmt='o',markersize=7,capsize=4,color=color)
        ax.axhline(0,color='#888',lw=1);ax.set_xticks(range(3),['S at switch','P at switch','P after 25 updates'])
        ax.set_title(title,fontsize=10);ax.set_xlim(-.4,2.4);ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Mean accuracy gain from zeroing readout (pp)')
    fig.suptitle('Readout benefit changes after P and after 25 updates',fontsize=12,y=.98)
    fig.text(.5,.02,'Head Adam cleared in both arms; body Adam retained. Paired-seed bootstrap 95% intervals. Different horizons: MLP 4000, ViT 1500.',ha='center',fontsize=7.5)
    fig.subplots_adjust(left=.09,right=.99,top=.82,bottom=.18,wspace=.15)
    for ext in ['png','pdf']:fig.savefig(OUT/f'contrast_pattern.{ext}',dpi=180)
    plt.close(fig)
    print('Wrote synthesis.md and focus/contrast figures')
if __name__=='__main__':main()

"""Aggregate the predeclared replay windows, keeping seeds and geometries separate."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from run import OUT,RAW,ROOT,ARMS,GEOMS,METRICS,PHASES

def markdown(f):
    def value(v):return f'{v:.5g}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(map(str,f.columns))+' |','| '+' | '.join(['---']*len(f.columns))+' |']+
        ['| '+' | '.join(value(v) for v in row)+' |' for row in f.itertuples(index=False,name=None)])

def run():
    rows=[];phase_rows=[];probes=[];guards=[];unit_sums=[]
    for start in [0,1000000]:
        for arm in ARMS:
            meta=json.loads((OUT/f'{arm}_{start}.json').read_text());guards.append(meta)
            with np.load(meta['archive']) as d:
                b=d['budget'];ep=d['endpoints'];pidx=d['probe_index'];total=b.sum((0,1))
                for gi,geom in enumerate(GEOMS):
                    for seed in range(10):
                        vals=total[gi,seed];initial=ep[0,gi,seed];final=ep[-1,gi,seed]
                        means=vals.mean(0);erosion,recovery,injection=means[:3]
                        net=float((final-initial).mean());scale=float(initial.mean())
                        row=dict(a=meta['a'],arm=arm,start=start,seed=seed,geometry=geom,
                            initial_mean=scale,final_mean=float(final.mean()),net=net,
                            normalized_net=net/scale,median_size_ratio=float(np.median(np.sqrt(final/initial))),
                            unit_shrink_fraction=float((final<initial).mean()),
                            erosion=erosion,recovery=recovery,injection=injection,
                            inward=erosion-recovery,balance=(erosion-recovery)/max(injection,1e-30),
                            pred_linear=means[3],pred_injection=means[4],
                            formula_balance=-means[3]/max(means[4],1e-30),
                            rounding=net-means[3]-means[4],
                            self_linear=means[5],rest_linear=means[6])
                        rows.append(row)
                        for phase in range(4):
                            m=b[:,phase,gi,seed].sum(0).mean(0)/10
                            phase_rows.append(dict(a=meta['a'],start=start,seed=seed,geometry=geom,
                                phase_start=PHASES[phase],phase_end=PHASES[phase+1],nupdates=PHASES[phase+1]-PHASES[phase],
                                erosion=m[0],recovery=m[1],injection=m[2],inward=m[0]-m[1],net=m[1]+m[2]-m[0],
                                self_linear=m[5],rest_linear=m[6]))
                        noise=d['probe_noise_injection'][:,gi,seed]
                        ei=d['probe_expected_injection'][:,gi,seed];li=d['probe_expected_linear'][:,gi,seed]
                        fi=d['probe_fullbatch_injection'][:,gi,seed]
                        for phase in sorted(set(pidx[:,2])):
                            ii=pidx[:,2]==phase
                            probes.append(dict(a=meta['a'],start=start,seed=seed,geometry=geom,phase=int(phase),
                                expected_linear=float(li[ii].mean()),expected_injection=float(ei[ii].mean()),
                                fullbatch_injection=float(fi[ii].mean()),noise_injection=float(noise[ii].mean()),
                                expected_net=float((li[ii]+ei[ii]).mean()),
                                q=float(d['probe_q'][ii,seed].mean()),
                                self_variance_rate=float(d['probe_self_variance_rate'][ii,seed].mean())))
    df=pd.DataFrame(rows);ph=pd.DataFrame(phase_rows);pr=pd.DataFrame(probes)
    df.to_csv(OUT/'window_by_seed.csv',index=False);ph.to_csv(OUT/'phase_by_seed.csv',index=False);pr.to_csv(OUT/'frozen_expectations_by_seed.csv',index=False)
    group=df.groupby(['geometry','start','a'])
    summary=group[['net','normalized_net','median_size_ratio','unit_shrink_fraction','balance','formula_balance']].median().reset_index()
    summary['min_seed_net']=group.net.min().values;summary['max_seed_net']=group.net.max().values
    summary.to_csv(OUT/'window_summary.csv',index=False)
    # Paired difference at .55 -> .60: report additive raw differences, not
    # subtraction of separately aggregated medians.
    contrasts=[]
    for (geom,start),g in df.groupby(['geometry','start']):
        left=g[np.isclose(g.a,.55)].set_index('seed');right=g[np.isclose(g.a,.6)].set_index('seed')
        for seed in range(10):
            r=dict(geometry=geom,start=start,seed=seed)
            for col in ['erosion','recovery','injection','net','inward','balance','normalized_net','median_size_ratio']:
                r[col+'_difference']=float(right.loc[seed,col]-left.loc[seed,col])
            assert abs(r['net_difference']-(r['recovery_difference']+r['injection_difference']-r['erosion_difference']))<1e-8
            contrasts.append(r)
    cont=pd.DataFrame(contrasts);cont.to_csv(OUT/'paired_055_to_060.csv',index=False)
    # Exact identities: not a statistical predictive score.
    verification=dict(source_replay_all_pass=all(g['source_log_bitwise_match'] for g in guards),
        source_checkpoints=len(guards),source_log_check_times=sum(len(g['source_log_steps_checked']) for g in guards),
        max_step_closure=max(g['max_step_closure'] for g in guards),max_window_closure=max(g['max_window_error'] for g in guards),
        max_SHBC_relative=max(g['max_SHBC_relative'] for g in guards),
        max_float_round_error=max(g['max_pred_float_round_error'] for g in guards),
        sign_agreement_min=min(g['finite_sign_agreement'] for g in guards))
    (OUT/'checks.json').write_text(json.dumps(verification,indent=2))
    verdict=[]
    for (geom,start),g in summary.groupby(['geometry','start']):
        left=g[np.isclose(g.a,.55)].iloc[0];right=g[np.isclose(g.a,.6)].iloc[0]
        label='BRACKETS_055_060' if left.net>0 and right.net<0 else 'DOES_NOT_BRACKET_055_060'
        verdict.append(dict(geometry=geom,start=int(start),label=label,net055=float(left.net),net060=float(right.net),
            scope='10-task replay window; not the original 5M horizon'))
    (OUT/'verdict.json').write_text(json.dumps(verdict,indent=2))
    # Plot complete paired seed values; quantile bands are descriptive only.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    colors={0:'#2477b2',1000000:'#d66426'}
    for start,name in [(0,'Tasks 1–10'),(1000000,'Tasks 101–110')]:
        g=df[(df.geometry=='variance')&(df.start==start)].groupby('a')
        med=g.balance.median();lo=g.balance.quantile(.1);hi=g.balance.quantile(.9)
        axs[0,0].plot(med.index,med,marker='o',color=colors[start],label=name)
        axs[0,0].fill_between(med.index,lo,hi,color=colors[start],alpha=.12)
        med=g.median_size_ratio.median();axs[0,1].plot(med.index,med,marker='o',color=colors[start],label=name)
    axs[0,0].axhline(1,color='gray',ls='--');axs[0,0].set(title='CondA / SGD: inward budget divided by injection',xlabel='Leaky slope a',ylabel='Above 1: variance contracts');axs[0,0].legend()
    axs[0,1].axhline(1,color='gray',ls='--');axs[0,1].set(title='Observed preactivation width: end / start',xlabel='Leaky slope a',ylabel='Median unit standard-deviation ratio');axs[0,1].legend()
    for ax,start in zip(axs[1],[0,1000000]):
        for a,style in [(.55,'-'),(.6,'--')]:
            g=ph[(ph.geometry=='variance')&(ph.start==start)&np.isclose(ph.a,a)].groupby('phase_start')
            m=g[['inward','injection']].median()
            for col,color in [('inward','#9f3740'),('injection','#2a887c')]:
                ax.plot(np.arange(4),m[col],ls=style,marker='o',color=color,label=f'a={a} {col}')
        ax.set_xticks(np.arange(4),['0–20','20–100','100–1000','1000–10000']);ax.axhline(0,color='gray',lw=.7)
        ax.set(title=f'Tasks {1 if start==0 else 101}–{10 if start==0 else 110}: phase budgets',ylabel='Mean per-task variance budget');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'budget.png',dpi=170);fig.savefig(OUT/'budget.pdf');plt.close(fig)
    text='# 恒等式11.1で幅の成長・収縮を検証 — 0916\n\n'
    text+='事後検証。CondA/SGDの7傾き×10 seed、既存step0/1Mから各10タスクを同じ乱数列で再生した。新しい長期軌道やAdamとの比較ではない。\n\n'
    text+='## 検算\n\n'+json.dumps(verification,ensure_ascii=False,indent=2)+'\n\n'
    text+='既存ログのWノルム・読み出しvを1000更新ごと、自由Wを10000更新ごとにbit一致で確認。丸め誤差込みの実更新Uによる帳簿は恒等式なので、この一致は機構の独立な予測精度ではない。理想的SGD式と実更新の違いにはfloat32丸めを残す。\n\n'
    text+='## 収支の定義\n\n'
    text+='A=<W,G>ならΔ‖W‖²=−2ηA+η²‖G‖²。収縮 iff A>η‖G‖²/2。行平均除去版ではA=S+H−B−C、LeakyはH=0（実装演算の残差は別に測定）。個体と各更新について評価した。\n\n'
    text+='実更新の内積が負の分を侵食E、正の分を回復R、二乗項を注入Jとして、**正味変化=R+J−E**。balance=(E−R)/Jは1より上なら収縮。これらは時間積分した帳簿であり、開始時φφ′だけによる次タスク予測とは異なる。回復は外向きの更新成分の名称であり、損失改善を意味しない。\n\n'
    text+='全重みノルムfull、行平均除去ノルムcentered、前活性分散varianceを分ける。CondAではvariance=‖w_free‖²/4。各表の収支は個体平均をseed別に出し、そのseed中央値。成分ごとの中央値は加法的とは限らないため、CSVのseed別閉包で検算。\n\n'
    text+='## 傾きと収支\n\n'+markdown(summary)+'\n\n'
    text+='median_size_ratioは各個体のノルム（varianceでは標準偏差）の終/始比の中央値をseed間で中央値。normalized_netは個体平均の二乗量変化をその初期値で割った値。両者の集約を混ぜない。seed範囲は独立な再試行の不確実性の記述で、個体を独立とする有意差検定ではない。\n\n'
    text+='## a=.55→.60の対比較\n\n'+markdown(cont.groupby(['geometry','start']).median(numeric_only=True).drop(columns='seed').reset_index())+'\n\n'
    text+='同じseed内で差を取った後の中央値。各成分差はseed別には厳密に足し上がるが、中央値どうしの加法性は保証されない。詳細はpaired_055_to_060.csv。\n\n'
    text+='## 時間帯\n\n'+markdown(ph.groupby(['geometry','start','a','phase_start','phase_end'])[['erosion','recovery','injection','net']].median().reset_index().query("geometry == 'variance' and (a == .55 or a == .6)"))+'\n\n'
    text+='区間は長さが違うので、上表は増減量の積分であって単位更新あたり速度ではない。個体の侵食と回復が交互に起きる可能性もある。\n\n'
    text+='## SGDの二乗項と雑音\n\n'
    text+='各タスクphase0,20,100,1000,5000,9999で全32入力を計算。E‖ĝ‖²=‖Eĝ‖²+E‖ĝ−Eĝ‖²で、SGDの期待注入とfull-batchの注入を分けた。疎な凍結状態の測定であり、これらを全時間の雑音寄与率とは呼ばない。frozen_expectations_by_seed.csvに保存。\n\n'
    text+='## 判定と範囲\n\n'+markdown(pd.DataFrame(verdict))+'\n\n'
    text+='単一の普遍的なa*=.6を前提にしない。測定窓で符号を挟まない場合、元の5M初期/最終比の境目をこの短い窓で説明したとはしない。収支は増減がどの成分差で起きたかを特定するが、活性化変更が学習状態・誤差・読み出しを変える原因まで帳簿だけで同定するものではない。\n\n'
    text+='φφ′の自己項の自由分散への時間積分はwindow_by_seed.csvのself_linear、残りはrest_linear。両者の大きさだけで実質的な機序・因果割合を断定しない。前の単純閾値50.9%は今回の恒等式由来の収支の反証ではない。\n\n'
    text+='仕様specs/width_budget_0916.md、コードanalysis/width_budget_0916、原データと個体配列は各arm_start.jsonとbackup_manifest.json。全ログ・コード版・SHAはprovenance.json。図budget.png/pdf。\n'
    (OUT/'summary.md').write_text(text)
    print(summary.to_string(index=False));print(json.dumps(verification,indent=2))

if __name__=='__main__':run()

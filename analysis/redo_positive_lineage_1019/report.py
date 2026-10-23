#!/usr/bin/env python3
"""Make the human report and figures from the preregistered diagnostic tables."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path

os.environ['MPLCONFIGDIR']='/tmp/redo_positive_matplotlib_1019'
os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/redo_positive_lineage_1019'
ARCHIVE=Path('/home/issan/Projects/obsidian-research-data/redo_positive_lineage_1019')


def read(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))


def num(r,k):
    v=r.get(k,'')
    return float(v) if v not in ('',None) else float('nan')


def median(values):
    a=np.asarray(list(values),dtype=float); a=a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else float('nan')


def write(path,rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def load(method,group):
    path=OUT/'raw'/f'{method}_s{group}-{group+9}'
    meta=json.loads((path/'trace/events.json').read_text())
    with np.load(path/'trace/units.npz') as f:u={k:f[k] for k in f.files}
    with np.load(path/'trace/deltas.npz') as f:dd={k:f[k] for k in f.files}
    return path,meta,u,dd


def main():
    verdict=json.loads((OUT/'verdict.json').read_text())
    births=read('births.csv'); firsts=read('first_allon.csv')
    deep=read('raw/unit_deep_metrics.csv')
    lookup={(r['method'],int(r['seed']),int(r['index']),int(r['unit'])):r for r in deep}
    firstrows=[]; trajectories=[]; counts=[]; first_cohort_deltas=[]; down_still_full=0
    for method in ('redo','none'):
        for group in (0,10):
            path,meta,u,dd=load(method,group); ev=meta['events']
            valid=[j for j,e in enumerate(ev) if e['event'] in ('init','after_step','reset_outgoing_2')]
            for t in (1,2,3,10,20,29,30):
                j=next(j for j,e in enumerate(ev) if e['event']=='task_end' and e['task']==t)
                for r,seed in enumerate(meta['seeds']):
                    p=u['g_l2_p'][j,r]
                    counts.append(dict(method=method,seed=seed,task=t,all_on=int((p==1).sum()),
                                       all_off=int((p==0).sum()),p_median=float(np.sort(p)[49])))
            if method!='redo':continue
            postp=u['g_l2_p'][dd['event'].astype(int)]
            down_still_full+=int(((dd['pre_p']==1)&(postp==1)&(dd['actual']<0)).sum())
            for r,seed in enumerate(meta['seeds']):
                first=next(j for j in valid if (u['g_l2_p'][j,r]==1).any())
                previous=valid[valid.index(first)-1] if first else first
                ids=np.flatnonzero(u['g_l2_p'][first,r]==1)
                if first:
                    with np.load(path/'trace'/f'deep_{previous:04d}.npz') as f:aa={k:f[k] for k in f.files}
                    with np.load(path/'trace'/f'deep_{first:04d}.npz') as f:bb={k:f[k] for k in f.files}
                    mu0=aa['h1'][r].astype(float).mean(0);mu1=bb['h1'][r].astype(float).mean(0)
                    own=(bb['W2'][r].astype(float)-aa['W2'][r])@mu0+bb['b2'][r]-aa['b2'][r]
                    upstream=bb['W2'][r]@(mu1-mu0)
                    own_only=aa['h1'][r].astype(float)@bb['W2'][r].astype(float).T+bb['b2'][r]
                    upstream_only=bb['h1'][r].astype(float)@aa['W2'][r].astype(float).T+aa['b2'][r]
                for i in ids:
                    pre=lookup[('redo',seed,previous,int(i))]
                    at=lookup[('redo',seed,first,int(i))]
                    rr=dict(seed=seed,unit=int(i),initial=int(first==0),first_task=ev[first]['task'],
                            first_step=ev[first]['step'],previous_step=ev[previous]['step'],
                            previous_p=float(u['g_l2_p'][previous,r,i]),
                            previous_other_full=int((u['g_l2_p'][previous,r]==1).sum()),
                            before_background_r2=num(pre,'background_r2'),before_single_r2=num(pre,'single_r2'),
                            before_single_provider=int(num(pre,'single_provider')),
                            before_single_cos=num(pre,'single_cos'),
                            birth_background_r2=num(at,'background_r2'),birth_single_r2=num(at,'single_r2'),
                            own=float(own[i]) if first else None,upstream=float(upstream[i]) if first else None,
                            upstream_share=float(upstream[i]/(own[i]+upstream[i])) if first else None,
                            own_only_p=float((own_only[:,i]>0).mean()) if first else None,
                            upstream_only_p=float((upstream_only[:,i]>0).mean()) if first else None,
                            own_only_min=float(own_only[:,i].min()) if first else None,
                            upstream_only_min=float(upstream_only[:,i].min()) if first else None)
                    firstrows.append(rr)
                    for k in range(len(dd['task'])):
                        j=int(dd['event'][k])
                        if j<first:continue
                        if u['generation'][j,r,1,i]!=u['generation'][first,r,1,i]:continue
                        first_cohort_deltas.append(dict(seed=seed,unit=int(i),task=int(dd['task'][k]),
                                                       actual=float(dd['actual'][k,r,i]),
                                                       own=float(dd['own_w'][k,r,i]+dd['own_b'][k,r,i]),
                                                       upstream=float(dd['upstream'][k,r,i]),
                                                       p=float(u['g_l2_p'][j,r,i])))
                # One fixed target per seed, selected as smallest ID at its first observed onset.
                i=int(ids[0])
                for j in valid:
                    if ev[j]['task']>3:break
                    trajectories.append(dict(seed=seed,unit=i,tc=ev[j]['tc'],event=ev[j]['event'],
                                             generation=int(u['generation'][j,r,1,i]),
                                             p=float(u['g_l2_p'][j,r,i]),mean=float(u['g_l2_mean'][j,r,i]),
                                             sd=float(u['g_l2_sd'][j,r,i]),
                                             minimum=float(u['g_l2_min'][j,r,i])))
    learned=[r for r in firstrows if not r['initial']]
    shares=[median(r['upstream_share'] for r in learned if r['seed']==s) for s in sorted(set(r['seed'] for r in learned))]
    br=[r for r in births if r['method']=='redo' and r['scope']=='g']
    gen0=[r for r in br if int(r['generation'])==0]
    first_keys={(r['seed'],r['unit']) for r in firstrows}
    first_lives=[r for r in gen0 if (int(r['seed']),int(r['unit'])) in first_keys]
    t3=[r for r in deep if r['method']=='redo' and r['task']=='3' and r['event']=='task_end' and num(r,'p')==1]
    effects=[num(r,'full_mean_direction')-num(r,'without_open_mean_direction') for r in t3]
    stats=dict(verdict=verdict,
               first_onset=dict(learned_seeds=len(shares),learned_first_cohort_units=len(learned),
                                upstream_share_median_over_seed=median(shares),
                                all_own_positive=all(r['own']>0 for r in learned),
                                all_upstream_larger=all(r['upstream']>r['own'] for r in learned),
                                before_background_r2_median=median(r['before_background_r2'] for r in learned),
                                before_single_r2_median=median(r['before_single_r2'] for r in learned),
                                before_single_cos_median=median(r['before_single_cos'] for r in learned),
                                before_single_ge99=sum(r['before_single_r2']>=.99 for r in learned),
                                posthoc_own_only_full=sum(r['own_only_p']==1 for r in learned),
                                posthoc_upstream_only_full=sum(r['upstream_only_p']==1 for r in learned),
                                before_background_ge95=sum(r['before_background_r2']>=.95 for r in learned),
                                before_background_ge99=sum(r['before_background_r2']>=.99 for r in learned)),
               survival=dict(gen0_lifetimes=len(gen0),gen0_drop=sum(int(r['dropped']) for r in gen0),
                             downward_but_still_full_unit_steps=down_still_full,
                             all_lifetime_duration_median=median(num(r,'observed_allon_duration') for r in br)),
               task3_local_open_ablation=dict(n=len(t3),others_effect_negative=sum(v<0 for v in effects),
                                               others_effect_positive=sum(v>0 for v in effects),
                                               median_effect=median(effects),
                                               median_full=median(num(r,'full_mean_direction') for r in t3),
                                               median_without=median(num(r,'without_open_mean_direction') for r in t3)),
               first_cohort_tracking=dict(sampled_steps=len(first_cohort_deltas),
                                          initial_cohort_units=len(first_lives),
                                          drop_from_full=sum(int(r['dropped']) for r in first_lives),
                                          drop_below_half=sum(int(r['dropped50']) for r in first_lives),
                                          actual_down=sum(r['actual']<0 for r in first_cohort_deltas),
                                          own_down=sum(r['own']<0 for r in first_cohort_deltas)))
    write(OUT/'first_birth_intervals.csv',firstrows)
    write(OUT/'task_global_counts.csv',counts)
    write(OUT/'first_target_trajectories.csv',trajectories)
    write(OUT/'first_cohort_steps.csv',first_cohort_deltas)
    (OUT/'report_numbers.json').write_text(json.dumps(stats,indent=2))
    # Keep human-facing numerical conclusions in verdict.json as well.
    verdict['first_onset']=stats['first_onset'];verdict['survival']=stats['survival'];verdict['first_cohort_tracking']=stats['first_cohort_tracking'];verdict['task3_local_open_ablation']=stats['task3_local_open_ablation']
    (OUT/'verdict.json').write_text(json.dumps(verdict,indent=2))
    fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    ts=(1,2,3,10,20,29,30)
    for method,color in (('redo','#1469a8'),('none','#bd602b')):
        vals=[[r['all_on'] for r in counts if r['method']==method and r['task']==t] for t in ts]
        xx=np.arange(len(ts));med=np.median(vals,axis=1);lo,hi=np.percentile(vals,[25,75],axis=1)
        ax[0,0].plot(xx,med,'o-',label=method,color=color)
        ax[0,0].fill_between(xx,lo,hi,color=color,alpha=.15)
    ax[0,0].set(xticks=np.arange(len(ts)),xticklabels=ts,xlabel='Task',ylabel='All-on L2 units / 100',title='Fixed 1,000-image probe: median and IQR')
    ax[0,0].legend(frameon=False)
    for seed in (0,9,13,15):
        rr=[r for r in trajectories if r['seed']==seed and r['generation']==0]
        ax[0,1].plot([r['tc'] for r in rr],[r['p'] for r in rr],'.-',label=f'seed {seed}, unit {rr[0]["unit"]}')
    ax[0,1].axvline(1560,color='k',ls=':',lw=1,label='First ReDo reset')
    ax[0,1].set(xlabel='Training updates',ylabel='Opening fraction',title='First all-on target: original generation only',ylim=(-.02,1.03))
    ax[0,1].legend(fontsize=8,frameon=False)
    ax[1,0].scatter([r['own'] for r in learned],[r['upstream'] for r in learned],s=30,color='#1469a8')
    lim=max(max(r['own'] for r in learned),max(r['upstream'] for r in learned))*1.05
    ax[1,0].plot([0,lim],[0,lim],'k:',lw=1)
    ax[1,0].set(xlabel='Own weight + bias contribution to mean(z)',ylabel='Upstream contribution to mean(z)',title='Intervals containing the first all-on observation')
    ax[1,1].scatter([r['before_single_r2'] for r in learned],[r['before_background_r2'] for r in learned],s=30,color='#1469a8')
    ax[1,1].axvline(.99,color='k',ls=':',lw=1);ax[1,1].axhline(.99,color='k',ls=':',lw=1)
    ax[1,1].set(xlabel='Best single other unit: held-out R2',ylabel='Other 99 units: held-out R2',title='Before onset: can others reproduce target z?')
    fig.savefig(OUT/'lineage_overview.png',dpi=180)
    fig.savefig(OUT/'lineage_overview.pdf')
    plt.close(fig)
    onset=stats['first_onset']; v=verdict['redo']; n=v['sampled_allon_unit_steps']
    table='\n'.join(f'| {t} | '+ ' | '.join(f'{median(r["all_on"] for r in counts if r["method"]==m and r["task"]==t):g}' for m in ('redo','none'))+' |' for t in ts)
    summary=f'''# ReDoの全開個体: 発生・再沈降・背景表現（1019）

状態: 診断再現完了。事後に選んだ問いに対して、診断統計を実行前登録した。run id: `redo_positive_lineage_1019`。

## 結論

**最初の全開を「先にいる全開クローンが押し上げた」と説明する必要はない。全開個体も沈む。** 最初の発生では本人と前層の双方が正方向に動き、この位置変化の分解では前層の寄与が大きい。周囲の表現による機構を唯一の原因として採用する証拠はない。

会話で挙げた「同じ重みでbiasだけ高い全開ユニットが対象を押し上げる」という具体的な筋書きは、この実軌道の説明としては撤回する。条件つき模型の可能な例を、実際の相手を特定した説明のように提示した点が誤りだった。

**中央値の高いピークへの追補**: `median_peak.md`。最初の全開と再沈降だけでは、第3課題の高い中央値の説明にならない。交換群と継続群を分けた保存状態の事後解析を参照。新たな学習走は追加していない。

20 seedのReDoとnoneを、元と同じ10seed stack・CUDA graph・30課題×780更新で再生した。4群×300行、計1,200行のper_taskが元の保存CSVと完全一致した。読み取り処理あり/なしの3課題検査も最終P,m,vまでbit一致。位置分解の閉合誤差はchecks.jsonに保存した。

以下の全開は、クラスごとに固定した10枚、計1,000枚のtrain probeでz>0が全件成立したことを指す。母集団での全開の証明ではない。表現の照合には、回帰係数の当てはめに使わなかった別の固定1,000枚を用いた（ネットワークの学習自体から除外した画像という意味ではない）。

## 最初の全開はどう生まれたか

- 初期から全開の個体がいたのは2/20 seed（seed11・12、各1個）。残り18/20では、第1課題の学習中、測定点で10〜200更新の間に最初の全開が現れた。ReDoの最初のresetは1,560更新後であり、noneでも第2課題末までは同じ個体軌道である。
- 残り18 seedの初回全開の集団は計{len(learned)}個体。同時初回の複数個体を含む。直前の測定点には他の全開個体が0個だった。ただし測定間隔内の発生順序は未識別で、近い線形表現を複数の未全開個体が担う可能性は残る。
- その区間では本人の重み+biasの寄与も、前層の寄与も全{len(learned)}個体で正。前層の寄与は全{len(learned)}個体で本人より大きく、seedごとに集約した寄与率の中央値は{100*onset['upstream_share_median_over_seed']:.1f}%だった。

分解は、固定画像でm=W2 μ+b2として、

    Δm=(ΔW2) μ_old+Δb2+W2_new(μ_new−μ_old)

で行った。最後の項を前層の寄与とした。交差項を最後の項へ含める順序であり、唯一の配分ではない。本人の変化にはAdamの実更新を用いた。この結果は「普通の課題学習と前層の変化で最初の全開が発生している」ことを示すが、上向きの期待を独立ラベル最小二乗模型から実証したものではない。

**事後・保存状態での入替計算**: 前層だけを直後の状態へ替え、本人のW2,b2を直前のまま保った場合、{onset['posthoc_upstream_only_full']}/{len(learned)}個体で全開になった。逆に前層を直前に固定し、本人だけ直後へ替えた場合は{onset['posthoc_own_only_full']}/{len(learned)}個体だった。これは固定probeでの局所的な状態入替で、学習し直した長期介入ではない。

## クローン化した相手がいるのか

初回全開の直前、対象本人を除いた他99個体+biasからzを再現するheld-out R2の中央値は{onset['before_background_r2_median']:.4f}。最良の単一相手では{onset['before_single_r2_median']:.4f}、その相手との入力重みcosの中央値は{onset['before_single_cos_median']:.4f}。単一相手のR2>=.99は{onset['before_single_ge99']}/{len(learned)}個体、他99個体によるR2>=.95は{onset['before_background_ge95']}/{len(learned)}、>=.99は{onset['before_background_ge99']}/{len(learned)}だった。

これは文字どおり同じ重みのクローンと、集団による再現可能性を分けた測定である。回帰で再現できることだけでは、実際の読み出しがその関係を使って対象を押し上げたとは結論しない。相手は`first_birth_intervals.csv`のbefore_single_provider列にIDを残した。理想化の{{1,z}}控除後の押しと実際の背景を控除した量、実ラベルCEの局所方向も別々に保存した。

## 全開した個体はなぜ沈まないのか

**実際には沈んでいる。** ReDoの全30課題の登録測定点で、更新直前に全開だったunit-stepは{n:,}件。そのうち固定画像のmean(z)が下がったのは{v['actual_downward_unit_steps']:,}件（{100*v['actual_downward_unit_steps']/n:.1f}%）、本人の重み+biasの寄与が下向きだったのは{v['own_downward_unit_steps']:,}件（{100*v['own_downward_unit_steps']/n:.1f}%）。実ミニバッチの勾配を通常SGDに換算した方向が下向きなのは{v['sgd_downward_unit_steps']:,}件（{100*v['sgd_downward_unit_steps']/n:.1f}%）。相互依存する記述件数であり、独立標本の検定には使わない。

全開に初めて達した世代を一つの観測寿命と数えると、ReDoで{len(br):,}寿命。そのうち{sum(int(r['dropped']) for r in br):,}は、同じ世代のうちに再びp<1となり、{sum(int(r['dropped99']) for r in br):,}はp<.99、{sum(int(r['dropped50']) for r in br):,}はp<.5へ下がった。観測終了またはresetによる打切りを含む。全開を保つ期間があっても、位置が動かないとは限らず、正の余白を削りながらp=1に留まる場合がある。

実際にmean(z)が下がったにもかかわらず、更新の前後ともp=1だったunit-stepは{down_still_full:,}件あった。したがって「沈んでいるが、全開判定を変えるほど負側へ到達していない」を、位置の静止と区別する必要がある。図の個体軌道は初回の世代だけを描き、resetで別個体になった先へ線をつながない。

とくに、各seedで最初に全開を観測した集団（初期の2個体と学習中の24個体、計{len(first_lives)}個体）は、{sum(int(r['dropped']) for r in first_lives)}/{len(first_lives)}個体が同じ世代のうちにp<1へ戻り、{sum(int(r['dropped50']) for r in first_lives)}/{len(first_lives)}個体はp<.5まで戻った。「最初の全開個体が沈まない」という前提は実測に合わない。

固定global画像での第2層の全開個体数（100個体中、20 seedの中央値）:

| 課題末 | ReDo | none |
|---|---:|---:|
{table}

ReDoでは全開が増える時期も減る時期もある。「半分より開いた個体が全開に吸収され、そこから沈まない」という読みは支持されない。再reset・本人の更新・Adamの持越し・前層変化を列として残し、原因を一つへ押し込めない。

## 局所の相互作用の測定と限界

`deep_cohorts.csv`は、現在課題の固定256画像でCE勾配を計算し、他個体のreadout寄与を除いたとき、他の全開個体だけを除いたとき、最良の単一相手を除いたときの対象の位置方向を記録した。入力特徴・対象の出力重みは固定し、予測を再計算した局所の計算介入である。再学習や長期の効果とは異なる。

例として第3課題末、固定global probeで全開だった{len(t3)}個体について、他の全開個体を含めたときのCE位置方向は、それらのreadout寄与を除いたときより{sum(v<0 for v in effects)}/{len(t3)}個体で下向きだった。全予測での位置方向の中央値は{stats['task3_local_open_ablation']['median_full']:.4f}、他の全開個体を除いた予測では{stats['task3_local_open_ablation']['median_without']:.4f}（正は上向き）。他の全開個体が常に上向きの力を加えている、という説明にもならない。除去によって予測を大きく変える計算であり、局所差を唯一の寄与や長期効果とは解釈しない。

この測定は、周囲が実際のCE勾配へ影響することと、会話で導入した「直線部分を差し引くと残りが縮むため上へ行く」という模型の帰属を分ける。p>1/2単独の符号則を実データへ適用しない。最初の全開の発生には、既存の全開クローンを仮定する説明を採用しない。

## 保存先

- spec: `specs/spec_redo_positive_lineage_1019.md`（実行前commit a36ecfd4）。
- `verdict.json` / `checks.json` / `report_numbers.json`: 結論に用いた数値と整合性。
- `first_birth_intervals.csv`: 最初の全開の直前・直後、位置分解、相手IDと再現性。
- `births.csv` / `first_allon.csv`: 全開世代・再沈降・打切り。
- `step_cohorts.csv` / `reset_stages.csv`: 本人・前層・moment、reset段階ごとの変化。
- `deep_cohorts.csv`: 集団/単一相手のheld-out再現と局所CE介入。
- `lineage_overview.png` / `.pdf`: 固定probeの全開数・最初の個体軌道・位置分解・相手の再現性。
- 生配列・元engine再生CSV・環境・学習ログは `{ARCHIVE}` へ退避し、`backup_manifest.json`で照合する。
'''
    (OUT/'summary.md').write_text(summary)
    print(json.dumps(stats['first_onset'],indent=2))


if __name__=='__main__':main()

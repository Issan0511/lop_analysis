#!/usr/bin/env python3
"""Posthoc follow-up: whose opening fractions make the task-3 median high? No new training."""
import csv
import json
import os
from pathlib import Path
import numpy as np

os.environ['MPLCONFIGDIR']='/tmp/redo_positive_matplotlib_1019'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/redo_positive_lineage_1019'
RAW=Path('/home/issan/Projects/obsidian-research-data/redo_positive_lineage_1019/raw')


def med(x):
    return float(np.sort(x)[(len(x)-1)//2])


def write(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def main():
    rows=[]; intervals=[]; updates=[]
    for group in (0,10):
        path=RAW/f'redo_s{group}-{group+9}'
        meta=json.loads((path/'trace/events.json').read_text());ev=meta['events']
        with np.load(path/'trace/units.npz') as f:u={k:f[k] for k in f.files}
        with np.load(path/'trace/deltas.npz') as f:dd={k:f[k] for k in f.files}
        pre=next(j for j,e in enumerate(ev) if e['task']==2 and e['event']=='reset_pre')
        post=next(j for j,e in enumerate(ev) if e['task']==2 and e['event']=='reset_outgoing_2')
        mid=next(j for j,e in enumerate(ev) if e['task']==3 and e['event']=='after_step' and e['step']==50)
        with np.load(path/'trace'/f'deep_{post:04d}.npz') as f:a={k:f[k] for k in f.files}
        with np.load(path/'trace'/f'deep_{mid:04d}.npz') as f:b={k:f[k] for k in f.files}
        for r,seed in enumerate(meta['seeds']):
            mask=u['reset_mask'][pre,r,1].astype(bool)
            for j,e in enumerate(ev):
                if not ((e['task']==2 and e['event'].startswith('reset_')) or (e['task']==3 and e['event'] in ('after_step','task_end'))):continue
                for label,ma in [('all',np.ones(100,dtype=bool)),('recycled',mask),('survivors',~mask)]:
                    p=u['t3_l2_p'][j,r,ma];m=u['t3_l2_mean'][j,r,ma];s=u['t3_l2_sd'][j,r,ma]
                    rows.append(dict(seed=seed,event=e['event'],task=e['task'],step=e['step'],tc=e['tc'],cohort=label,
                                     n=int(ma.sum()),p_median=med(p),p_mean=float(p.mean()),ge95=int((p>=.95).sum()),
                                     ge99=int((p>=.99).sum()),full=int((p==1).sum()),m_median=med(m),sd_median=med(s),
                                     outnorm_median=med(u['outnorm_l2'][j,r,ma])))
            mu0=a['h1'][r].astype(float).mean(0);mu1=b['h1'][r].astype(float).mean(0)
            ownW=(b['W2'][r].astype(float)-a['W2'][r])@mu0
            ownb=b['b2'][r]-a['b2'][r]
            upstream=b['W2'][r]@(mu1-mu0)
            for label,ma in [('recycled',mask),('survivors',~mask)]:
                intervals.append(dict(seed=seed,cohort=label,probe='global1000',interval='reset_to_task3_step50',
                                      ownW_mean=float(ownW[ma].mean()),ownb_mean=float(ownb[ma].mean()),
                                      own_mean=float((ownW+ownb)[ma].mean()),upstream_mean=float(upstream[ma].mean()),
                                      own_positive=int(((ownW+ownb)[ma]>0).sum()),n=int(ma.sum())))
            for k,t in enumerate(dd['task']):
                if t!=3:continue
                for label,ma in [('recycled',mask),('survivors',~mask)]:
                    d=dict(seed=seed,task=3,step=int(dd['step'][k]),cohort=label,n=int(ma.sum()))
                    d.update({name:float(dd[name][k,r,ma].mean()) for name in ('actual','own_w','own_b','upstream','adam_current','adam_memory','sgd_direction')})
                    updates.append(d)
    def pick(seed,event,step,cohort):
        return next(r for r in rows if r['seed']==seed and r['event']==event and r['step']==step and r['cohort']==cohort)
    end=[r for r in rows if r['event']=='task_end' and r['cohort']=='recycled']
    result=dict(tier='posthoc_saved_state_followup_no_new_training',
                task3_recycled_pmedian_at_least99_seeds=sum(r['p_median']>=.99 for r in end),
                task3_recycled_pmedian_equal1_seeds=sum(r['p_median']==1 for r in end),
                seed9={stage:{c:pick(9,e,s,c) for c in ('all','recycled','survivors')}
                       for stage,e,s in [('before','reset_pre',780),('after_reset','reset_outgoing_2',780),
                                        ('step50','after_step',50),('end','task_end',780)]},
                seed9_interval=[r for r in intervals if r['seed']==9],
                seed13_interval=[r for r in intervals if r['seed']==13])
    write(OUT/'median_peak_cohorts.csv',rows);write(OUT/'median_peak_intervals.csv',intervals);write(OUT/'median_peak_updates.csv',updates)
    (OUT/'median_peak.json').write_text(json.dumps(result,indent=2))
    v=json.loads((OUT/'verdict.json').read_text());v['median_peak']=result
    (OUT/'verdict.json').write_text(json.dumps(v,indent=2))
    text='''# 高い開口率の中央値を担ったのは誰か（1019追補）

状態: **事後・保存状態の追補解析。新しい学習走は実施していない。** Issa「じゃあなんで中央値であんなに開いてんねん」への回答。

先の報告の「最初の全開の発生」と「その後の再沈降」は、第3課題で高い中央値が出る理由への回答としては不十分だった。第3課題の全2,500画像をリセット前から固定して測った量に戻り、初回resetの対象と生存個体を分けた。

## seed 9・第3課題の中央値0.9776の内訳

100個体のうち54個体が第2課題末に交換された。残り46個体は継続した。

| 状態 | 全100個体のp中央値 | 交換54個体のp中央値 | 継続46個体のp中央値 |
|---|---:|---:|---:|
| リセット直前（第3課題の画像で評価） | 0.3512 | 0.1336 | 0.9880 |
| リセット直後 | 0.6596 | 0.4168 | 0.9876 |
| 第3課題50更新 | 0.9932 | 1.0000 | 0.2832 |
| 第3課題末・780更新 | 0.9776 | 1.0000 | 0.6168 |

**中央値を高くした主役は、新しく交換された54個体だった。** 第3課題50更新で、そのうち53個体がp>=.95、52個体がp>=.99。課題末でも交換群の52個体がp>=.95であり、それだけで100個体の過半数に達する。課題末の全体p>=.95は57個体（交換52・継続5）、厳密な全開は31個体だった。全体のp平均は0.80124で、p中央値0.9776とは違う統計量である。

これは「もともと開いていた個体が残っただけ」ではない。交換群のp中央値はリセット直後0.4168から学習中に1へ上がる一方、継続群は最初0.9876から50更新で0.2832へ下がった。高い開口率を担う集団が入れ替わっている。

## なぜ交換群が上へ行くのか: 実更新で分かった範囲

交換直後の出力重みは厳密に0。CEの逆伝播は、出力重みをv_i、クラス誤差をeとして、

    ∂L/∂z_i = 1[z_i>0] v_i^T e

なので、交換直後の最初の更新で、新生個体のincoming loss gradientは0になる。保存された実勾配でも、seed9の交換群は第3課題の第1更新でincoming Wの変化0、今回gradientのAdam寄与0だった（biasには保持されたmomentがあり、約−1.7e−7の平均変化）。リセット直後に旧課題の強い自己項が必ず沈める、という扱いはできない。

出力重みが新課題に対して学ばれ始めると、交換群自身のincoming更新が正方向になった。固定global1,000画像の位置で、reset直後→第3課題50更新の交換群の平均変化を分解すると、本人W+biasが+4.2916、前層の変化が+1.3130。一方、継続群は本人−1.6837、前層−3.2447だった。これは第3課題2,500画像の分解ではなく、別の固定global画像での分解である。50更新目の一歩でも交換群の本人Wによる位置変化は平均+0.03827、biasは+0.000075だった。biasが大きく持ち上がったという説明にもならない。

観測上は、**旧課題の読み出しを持つ継続群が下がり、出力重みを0から学び直す新生群が新課題の学習とともに上がる**。この正負の入替は測定できた。新生群の正方向の期待を理論として一般化するには、新課題への当てはめと位置更新の揃いを扱う必要がある。既存の全開クローンによる上向き相互作用を原因として採用する結果ではない。

## seed13と20 seed

seed13でも交換47個体のp中央値はreset直後0.5028→第3課題末1.0000、継続53個体は0.9620→0.3608だった。末尾のp>=.95は交換45・継続6の計51個体で、全体中央値が0.9640になった。
'''
    text+=f'\n20 seedでは、第3課題末の交換群p中央値が0.99以上なのは{result["task3_recycled_pmedian_at_least99_seeds"]}/20 seed、厳密に1なのは{result["task3_recycled_pmedian_equal1_seeds"]}/20 seedだった。この追補は最も高い2 seedだけで機構の一般性を断言するものではなく、全seedの集約を別途残した。\n'
    text+='\n数値の正本: `median_peak.json`（`verdict.json`にも同内容）・`median_peak_cohorts.csv`・`median_peak_intervals.csv`・`median_peak_updates.csv`。元の全1,200行の完全再現と、同じ入力集合での追跡を維持した。\n'
    (OUT/'median_peak.md').write_text(text)
    fig,ax=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for panel,seed in enumerate((9,13)):
        for cohort,label,color in [('all','All 100','#444444'),('recycled','Recycled','#1469a8'),('survivors','Survivors','#bd602b')]:
            q=[r for r in rows if r['seed']==seed and r['cohort']==cohort and (r['event']=='reset_outgoing_2' or (r['task']==3 and r['event']=='after_step'))]
            x=[0 if r['event']=='reset_outgoing_2' else r['step'] for r in q]
            ax[panel].plot(x,[r['p_median'] for r in q],'.-',label=label,color=color)
        ax[panel].set(xlabel='Updates in task 3',ylabel='Median opening fraction',title=f'Seed {seed}: same task-3 images throughout',ylim=(0,1.03))
        ax[panel].legend(frameon=False)
    fig.savefig(OUT/'median_peak.png',dpi=180);plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k.startswith('task3_')},indent=2))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Posthoc analysis of stored states. No training, no manipulation of the source archive."""
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np

os.environ['MPLCONFIGDIR']='/tmp/redo_v_open_matplotlib_1019'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/redo_v_opening_1019'
RAW=Path('/home/issan/Projects/obsidian-research-data/redo_positive_lineage_1019/raw')


def rank(x):
    order=np.argsort(x,kind='stable');xx=x[order]
    edges=np.r_[0,np.flatnonzero(np.diff(xx)!=0)+1,len(x)]
    r=np.empty(len(x),dtype=float)
    for a,b in zip(edges[:-1],edges[1:]):r[order[a:b]]=(a+b-1)/2
    return r


def corr(x,y):
    x,y=rank(x),rank(y)
    if np.std(x)==0 or np.std(y)==0:return float('nan')
    return float(np.corrcoef(x,y)[0,1])


def med(x):return float(np.median(x))
def lowmed(x):return float(np.sort(x)[(len(x)-1)//2])
def nanmed(x):
    x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def write(path,rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'raw').mkdir(exist_ok=True)
    individual=[];by_seed=[];sources=[]
    for method in ('redo','none'):
        for group in (0,10):
            path=RAW/f'{method}_s{group}-{group+9}'
            meta=json.loads((path/'trace/events.json').read_text());events=meta['events']
            with np.load(path/'trace/units.npz') as f:u={k:f[k] for k in f.files}
            mask0=None
            if method=='redo':
                pre=next(j for j,e in enumerate(events) if e['event']=='reset_pre' and e['task']==2)
                mask0=u['reset_mask'][pre,:,1].astype(bool)
            for j,e in enumerate(events):
                if not ((e['task']==3 and e['event']=='after_step') or (e['task']==2 and e['event']=='reset_outgoing_2') or
                        (e['event']=='task_end' and e['task'] in (1,10,20,29))):continue
                name='post_reset' if e['event']=='reset_outgoing_2' else (f't3_s{e["step"]}' if e['task']==3 else f't{e["task"]}_end')
                src=path/'trace'/f'deep_{j:04d}.npz'
                sources.append(dict(path=str(src),sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
                with np.load(src) as f:V=f['W3'].astype(float);z=f['z2'].astype(float)
                vc=V-V.mean(axis=1,keepdims=True)
                norm=np.linalg.norm(vc,axis=1);rawnorm=np.linalg.norm(V,axis=1)
                p=(z>0).mean(axis=1);act=np.maximum(z,0)
                rms=np.sqrt((act*act).mean(axis=1));strength=norm*rms
                for r,seed in enumerate(meta['seeds']):
                    for i in range(100):
                        individual.append(dict(method=method,stage=name,seed=seed,unit=i,
                                               cohort=('recycled' if mask0[r,i] else 'survivors') if method=='redo' else 'none',
                                               v_norm=float(norm[r,i]),raw_v_norm=float(rawnorm[r,i]),p_global=float(p[r,i]),
                                               p_task3=float(u['t3_l2_p'][j,r,i]) if e['task']<=3 else float('nan'),
                                               activation_rms=float(rms[r,i]),logit_contribution_rms=float(strength[r,i])))
                    cohorts=[('all',np.ones(100,dtype=bool))]
                    if method=='redo' and (name=='post_reset' or name.startswith('t3_')):
                        cohorts += [('recycled',mask0[r]),('survivors',~mask0[r])]
                    for label,ma in cohorts:
                        rr=dict(method=method,stage=name,seed=seed,cohort=label,n=int(ma.sum()),
                                v_norm=med(norm[r,ma]),raw_v_norm=med(rawnorm[r,ma]),p_global=lowmed(p[r,ma]),
                                p_task3=lowmed(u['t3_l2_p'][j,r,ma]) if e['task']<=3 else float('nan'),
                                activation_rms=med(rms[r,ma]),logit_contribution_rms=med(strength[r,ma]),
                                rho_v_p_global=corr(norm[r,ma],p[r,ma]))
                        by_seed.append(rr)
    grouped=[]
    for method,stage,cohort in dict.fromkeys((r['method'],r['stage'],r['cohort']) for r in by_seed):
        rs=[r for r in by_seed if (r['method'],r['stage'],r['cohort'])==(method,stage,cohort)]
        row=dict(method=method,stage=stage,cohort=cohort,seeds=len(rs))
        for k in ('n','v_norm','raw_v_norm','p_global','p_task3','activation_rms','logit_contribution_rms','rho_v_p_global'):
            row[k]=nanmed([r[k] for r in rs])
        row['finite_correlation_seeds']=int(sum(np.isfinite(r['rho_v_p_global']) for r in rs))
        grouped.append(row)
    write(OUT/'per_seed.csv',by_seed);write(OUT/'summary.csv',grouped)
    np.savez_compressed(OUT/'raw/individuals.npz',**{k:np.asarray([r[k] for r in individual]) for k in individual[0]})
    lookup={(r['method'],r['stage'],r['cohort']):r for r in grouped}
    result=dict(tier='posthoc_saved_state_analysis_no_training',
                norm_definition='L2 norm across 100 classes after subtracting each unit column class mean',
                stages=grouped,source_files=sources)
    (OUT/'verdict.json').write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for label,color in (('recycled','#1469a8'),('survivors','#bd602b')):
        rs=[r for r in individual if r['method']=='redo' and r['stage']=='t3_s780' and r['cohort']==label]
        axes[0].scatter([r['v_norm'] for r in rs],[r['p_global'] for r in rs],s=6,alpha=.28,color=color,label=label)
        stages=['post_reset']+[f't3_s{s}' for s in (1,2,5,10,25,50,100,200,400,780)]
        ss=[lookup[('redo',stage,label)] for stage in stages]
        axes[1].plot([r['v_norm'] for r in ss],[r['p_task3'] for r in ss],'.-',color=color,label=label)
        for name in ('post_reset','t3_s10','t3_s50','t3_s780'):
            r=lookup[('redo',name,label)];txt='0' if name=='post_reset' else name[4:]
            offset=(-23,5) if label=='survivors' and name=='t3_s780' else ((4,-12) if name=='t3_s50' else (4,4))
            axes[1].annotate(txt,(r['v_norm'],r['p_task3']),xytext=offset,textcoords='offset points',fontsize=8)
    axes[0].set(xlabel='Centered outgoing-weight norm',ylabel='Opening fraction on fixed 1,000 images',title='Same time: task 3 end, 20 seeds x 100 units')
    axes[1].set(xlabel='Median centered outgoing-weight norm',ylabel='Median opening on fixed task-3 images',title='Same cohorts over time (labels: task-3 updates)')
    for ax in axes:ax.legend(frameon=False);ax.set_ylim(-.03,1.05)
    fig.savefig(OUT/'v_and_opening.png',dpi=180);fig.savefig(OUT/'v_and_opening.pdf');plt.close(fig)
    table='\n'.join(f'| {s} | '+ ' | '.join(f'{lookup[("redo",stage,label)][k]:.4f}' for label in ('recycled','survivors') for k in ('v_norm','p_task3'))+' |'
                    for s,stage in [('直後','post_reset'),('10更新','t3_s10'),('50更新','t3_s50'),('780更新','t3_s780')])
    t3all=lookup[('redo','t3_s780','all')];t3new=lookup[('redo','t3_s780','recycled')];t3old=lookup[('redo','t3_s780','survivors')]
    text=f'''# 出力重みvの大きさと開口率（保存状態・事後解析、1019）

新規学習なし。元の診断再現 `redo_positive_lineage_1019` の20 seedを使用。v_iは100クラスへの出力重みベクトルなので、全クラスへ同じlogitを足す無効な成分を引いたL2normを主指標とした。これは個々の重み成分の値ではない。

## 実際の関係

**第3課題末の個体比較では、新生群は小v・高開口、継続群は大v・低開口。しかし同じ新生群の時間変化では、vも開口率も一緒に増えていた。** 「vを小さくすれば必ず開く」という関係ではない。

開口率は p_i=Pr(w_i^T h+b_i>0) で決まり、v_iはこの式に入らない。入力側を固定してvだけを変えても、その瞬間の開口率は変わらず、vが変えるのは以後の学習の勾配である。

各seed内で個体の中央値を求め、その20seedの中央値を取った。開口率は、リセット前から固定した第3課題の全2,500画像。各列の中央値の個体が同一であるとは限らない。

| 第3課題 | 新生群のv norm | 新生群の開口率 | 継続群のv norm | 継続群の開口率 |
|---|---:|---:|---:|---:|
{table}

固定global1,000画像での、同一時点の100個体間Spearman相関をseedごとに求めると、第3課題末の中央値は全体{t3all['rho_v_p_global']:.4f}、新生群内{t3new['rho_v_p_global']:.4f}、継続群内{t3old['rho_v_p_global']:.4f}。時間と集団の混合を因果関係に読み替えない。

10更新時点の新生群内の相関の中央値は{lookup[('redo','t3_s10','recycled')]['rho_v_p_global']:.4f}で正だった。群を混ぜた課題末の負相関と、「同じ新生群内で小vほど開く」という命題は同じではない。

## vが小さければ予測寄与も弱いか

実際のlogit寄与は a_i v_i で決まる。global画像で測った norm(v_i_centered)*RMS(a_i) の個体中央値をseedごとに求め、20seedで中央値を取ると、第3課題末は新生群{t3new['logit_contribution_rms']:.4f}、継続群{t3old['logit_contribution_rms']:.4f}だった。活動のRMSはそれぞれ{t3new['activation_rms']:.4f}、{t3old['activation_rms']:.4f}。小vだけで自己抑制や実際の寄与の小ささを判断できない。

## 解釈の範囲

v=0の直後は本人のincoming損失勾配も0。新課題を直す向きにvが学ばれた後、活動を増やす需要が勝つ、という候補と整合する。今回の観測はvを固定・縮小する操作ではなく、vも前活性も同時に学ぶ軌道の比較である。0.1や0.3を普遍的な開閉閾値とはしない。

理論上、CEの開いた入力で −∂L/∂z_i = v_i^T(y−p_minus_i) − v_i^T(p−p_minus_i)。後者はa_i>=0で非負。固定したa_iと周囲のlogitに対してv_iの尺度だけを小さくする局所展開では、前者は一次、後者は二次。ただし前者の符号には現在課題への向きが要り、Adamの実更新量の比例則ではない。

数値の正本: verdict.json / summary.csv。種ごとの値はper_seed.csv、個体値はraw/individuals.npz。入力の固定・群分け・normの定義を揃えた事後解析である。
'''
    (OUT/'summary.md').write_text(text)
    print(table)
    print(json.dumps(dict(task3_all=t3all,task3_recycled=t3new,task3_survivors=t3old),indent=2))


if __name__=='__main__':main()

"""Report the input-dependent covariance hypothesis and a batch-noise control."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/vit_mechanisms_1025'
p=pd.read_csv(OUT/'covariance_pressures.csv')
c=pd.read_csv(OUT/'covariance_comparisons.csv')
f=pd.read_csv(OUT/'covariance_steps.csv')
b=pd.read_csv(OUT/'fullbatch_fullbatch.csv')
m=pd.read_csv(OUT/'rollouts.csv')
for mode,n in [('covariance',20),('fullbatch',10)]:
    run=json.loads((OUT/f'{mode}_run.json').read_text())
    assert run['status']=='complete' and run['states']==n


def pick(df,**kw):
    mask=np.ones(len(df),dtype=bool)
    for k,v in kw.items(): mask &= (df[k]==v).to_numpy()
    return df.loc[mask]


rows=[]
for task in (10,60):
    for stage,metric in [('mid','SGD'),('end','Adam_fixed')]:
        for layer in range(1,7):
            cc=pick(c,task=task,stage=stage,metric=metric,layer=layer,prediction='covariance')
            pp=pick(p,task=task,stage=stage,metric=metric,layer=layer)
            count=int(((cc.relative_l2_error<=.2)&(cc.balanced_accuracy>=.8)).sum())
            rows.append(dict(task=task,stage=stage,metric=metric,layer=layer,
                fit_mean=float(pick(pp,signal='fit')['mean'].median()),
                mean_term=float(pick(pp,signal='mean')['mean'].median()),
                covariance_term=float(pick(pp,signal='covariance')['mean'].median()),
                covariance_relative_error=float(cc.relative_l2_error.median()),
                covariance_ba=float(cc.balanced_accuracy.median()),
                erased_sinking_fraction=float(cc.mean_erases_sinking_fraction.median()),
                covariance_criteria_seeds=count,verdict='COVARIANCE_CARRIES_FIT' if count>=4 else 'NOT_ESTABLISHED'))
d=pd.DataFrame(rows)
d.to_csv(OUT/'covariance_verdict.csv',index=False)
lines=['# ViT: 入力データのばらつき仮説の検査（1025・探索的追補）','',
       'Issa「相2の正体って入力データのばらつき説」に対する追加。追補specとコードは1dc1211bで結果閲覧前にcommit。共有head、5 seed、task10/60の100/500更新後の20状態。','',
       '**「画像ごとの経路と誤差の対応」が効いているという仮説には支持がある。** task60の100更新時では共分散項が押しの符号BA97〜99%、相対誤差4〜8%（各層のseed中央値）で実fitを近似し、全層で4/5以上の判定を満たした。ただし500更新後では平均項との相殺が大きく、共分散項だけでは実fitを近似しない。入力ばらつき単独の説明が成立したわけではない。','',
       '全250画像で50更新する対照でも、minibatchとほぼ同じ層別変位が残った。task10のFFN1〜3は5/5 seedで沈降、FFN5〜6は5/5で上昇。task60はFFN1が5/5で沈降、FFN2〜3は3/5と弱く、FFN5〜6は5/5で上昇。この継続区間ではbatch抽選ノイズは沈降に必要でなかった。画像間の違いとbatch抽選ノイズは区別できる。','',
       '## 今回の「ばらつき」の操作','',
       '画像nのhead前特徴をH_n、headからの誤差信号をe_n=(p_n−q_n)Vc、入力によって変わるbodyの微分をJ_nとする。実際の勾配はg=E[J_nᵀe_n]。これを、平均同士の積g_mean=E[J_nᵀ]E[e_n]と、画像ごとの対応から来るg_cov=g−g_meanへ分けた。固定probeの各FFNのμ/σへ全パラメータ経路で運んだ。','',
       '画像・元logitsは変えず、backwardで全画像へ同じ平均誤差信号を渡す対照と、誤差信号を画像間で4通りにシャッフルする対照を作った。後者は誤差の周辺分布を保ち、画像と誤差の対応を壊す。これは入力の分散値そのものを縮める操作ではない。画像依存の微分経路と誤差の相関が、どちらへ押しているかを検査する。','',
       '## 共分散項は実際の押しを運ぶか','',
       '正のSが沈める方向。各セルは全unit平均Sのseed中央値。各seedではfit=mean+covだが、別々の中央値同士は必ずしも足し合わない。誤差比は||S_cov−S_fit||/||S_fit||。事前条件はこの比≤0.2かつBA≥80%を4/5 seed以上。','',
       '| 課題 | 更新 | FFN | 実fit S | 平均項 S | 共分散項 S | 誤差比 | 符号BA | 条件成立 |',
       '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for rr in d.itertuples():
    lines.append(f'| {rr.task} | {100 if rr.stage=="mid" else 500} | {rr.layer} | {rr.fit_mean:+.3g} | {rr.mean_term:+.3g} | {rr.covariance_term:+.3g} | {rr.covariance_relative_error:.3f} | {100*rr.covariance_ba:.1f}% | {rr.covariance_criteria_seeds}/5 |')
lines += ['',
          '100更新はSGD方向、500更新は保存Adam分母固定。500更新についてSGD版も生CSVに保存した。対応が良い場合でも、共分散が常に沈めるとは限らない。共分散が浮かせる層も、同じ方法でそのまま残している。','',
          '## 保存momentumを含む有限1更新','',
          '課題60末。元の通常学習勾配と、画像との対応を壊した勾配を同じAdam状態へ1回入れ、実際のμ/σ差を測定した。全250画像の勾配であり、元のminibatch更新ではない。各値はunit平均変位、その後seed中央値。','',
          '| FFN | 元のfit Δμ/σ | 平均誤差へ置換 | 4シャッフル対照平均 | 元fitの方が平均誤差より沈むseed |',
          '|---:|---:|---:|---:|---:|']
for layer in range(1,7):
    ff=pick(f,task=60,layer=layer).pivot(index='seed',columns='signal',values='delta_r_mean')
    shuffled=ff[[f'perm{i}' for i in range(4)]].mean(axis=1)
    lines.append(f'| {layer} | {ff.fit.median():+.5g} | {ff["mean"].median():+.5g} | {shuffled.median():+.5g} | {int((ff.fit<ff["mean"]).sum())}/5 |')
lines += ['',
          '保存momentumはすべての対照で同じ。したがって新しい勾配から共分散項を除いても、過去の押しは1更新では消えない。有限stepの符号だけで、直前に計算した瞬間の勾配の符号を否定しない。','',
          '## ミニバッチ抽選のノイズを除く対照','',
          '元の課題末から旧画像・旧ラベルのまま50更新した。full batchは全250画像を毎回使い、データ間の違いを保ったままbatch抽選を除く。元Adamを再開しhead/bodyをともに学習。更新回数は同じだが画像提示回数は異なる。各unitの開始からの差の中央値、その後seed中央値を示す。','',
          '| 課題末 | FFN | minibatch Δμ/σ | full batch Δμ/σ | full batchで沈むseed |',
          '|---:|---:|---:|---:|---:|']
for task in (10,60):
    for layer in range(1,7):
        bb=pick(b,task=task,layer=layer,step=50)
        mm=pick(m,task=task,layer=layer,step=50,arm='continue')
        lines.append(f'| {task} | {layer} | {mm.delta_r_median.median():+.5g} | {bb.delta_r_median.median():+.5g} | {int((bb.delta_r_median<0).sum())}/5 |')
lines += ['',
          '## 解釈の範囲','',
          '- 入力依存の微分経路と誤差の共分散は、単なる非負の「分散」と違って符号を持つ。ばらつきは押しを作る材料になり得るが、沈むか浮くかはReLUの開閉、後段、上流の更新を含む相関で決まる。',
          '- mean/covの分解恒等式だけでは証拠にならない。どの項がどの方向に効くか、対応を壊すと実変位が変わるかを比較した。',
          '- 共分散に誤差が含まれるため、これを「入力のばらつきだけで相2が説明できた」とは呼ばない。真の入力分散のdose介入や長期の必要性・十分性は未実施。',
          '- 全batchでも沈降が残る場合、その継続区間でbatch抽選ノイズは必要条件でない。ただし過去のminibatch学習の履歴は初期状態に残る。',
          '- 100→500更新の既存観測と、500更新後からさらに50更新する今回の区間を混同しない。',
          '- 既存の共同LS相2式の検査は[機構検証の本体](summary.md)。この共分散仮説は、それと同じ模型だと仮定しない。','',
          '![画像依存共分散の押し](image_covariance.png)','',
          '![全batchとminibatchの継続](batch_noise_control.png)']
(OUT/'covariance_summary.md').write_text('\n'.join(lines)+'\n')

plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True,sharex=True)
for col,task in enumerate((10,60)):
    for row,(stage,metric,label) in enumerate([('mid','SGD','Update 100, SGD'),('end','Adam_fixed','Update 500, saved Adam denominator')]):
        sub=pick(p,task=task,stage=stage,metric=metric)
        den=pick(sub,signal='fit').set_index(['seed','layer']).rms
        for sig,color,caption in [('fit','#222222','Actual fit'),('mean','#c65436','Mean signal'),('covariance','#286ca5','Image-dependent covariance')]:
            num=pick(sub,signal=sig).set_index(['seed','layer'])['mean']
            z=(num/den).groupby('layer').median()
            axes[row,col].plot(z.index,z.values,'o-',color=color,label=caption)
        axes[row,col].axhline(0,color='gray',ls=':',lw=1)
        axes[row,col].set(title=f'Task {task}: {label}',ylabel='Mean S / RMS of actual S',xticks=range(1,7))
        axes[row,col].grid(alpha=.18)
        if row==1: axes[row,col].set_xlabel('FFN block (1 = nearest input)')
axes[0,0].legend(frameon=False,fontsize=8)
fig.suptitle('Input-dependent Jacobian / error covariance; medians of 5 seeds\nPositive pressure lowers CLS mean / SD; negative pressure raises it')
for ext in ('png','pdf'): fig.savefig(OUT/f'image_covariance.{ext}',dpi=180)
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
for ax,task in zip(axes,(10,60)):
    for frame,color,label in [(pick(m,arm='continue'),'#888888','Mini-batch Adam (128 / 122)'),(b,'#286ca5','Full-batch Adam (250)')]:
        sub=pick(frame,task=task,step=50).groupby('layer').delta_r_median.median()
        ax.plot(sub.index,sub.values,'o-',color=color,label=label)
    ax.axhline(0,color='gray',ls=':',lw=1)
    ax.set(title=f'Continue task {task}: 50 updates',xticks=range(1,7),xlabel='FFN block',ylabel='Change in mean / SD (negative = sinking)')
    ax.grid(alpha=.18)
axes[0].legend(frameon=False,fontsize=8)
fig.suptitle('Same images, same labels, same saved Adam start; medians of 5 seeds')
for ext in ('png','pdf'): fig.savefig(OUT/f'batch_noise_control.{ext}',dpi=180)
plt.close(fig)
print((OUT/'covariance_summary.md').read_text())

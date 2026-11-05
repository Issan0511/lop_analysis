"""Summarize registered descriptive criteria without conflating observations and mechanisms."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/vit_mechanisms_1025'
ARCHIVE=Path('/home/issan/Projects/obsidian-research-data/vit_mechanisms_1025')
p=pd.read_csv(OUT/'pressures.csv')
c=pd.read_csv(OUT/'comparisons.csv')
f=pd.read_csv(OUT/'finite_steps.csv')
s=pd.read_csv(OUT/'states.csv')
h=pd.read_csv(OUT/'heads.csv')
r=pd.read_csv(OUT/'rollouts.csv')
for mode in ('diagnostic','rollout'):
    run=json.loads((OUT/f'{mode}_run.json').read_text())
    assert run['status']=='complete' and run['states']==10
assert len(s)==10 and len(f)==480 and len(r)==600 and len(h)==80


def pick(df,**kw):
    keep=np.ones(len(df),dtype=bool)
    for k,v in kw.items(): keep &= (df[k]==v).to_numpy()
    return df.loc[keep]


def med(df,col,**kw): return float(pick(df,**kw)[col].median())


def pct(x): return f'{100*x:.1f}%'


pressure=pick(p,dataset='next',metric='Adam_fixed',moment='r')
a_rows,b_rows=[],[]
for task in (10,60):
    for layer in range(1,7):
        pp=pick(pressure,task=task,layer=layer).pivot(index='seed',columns='signal',values='mean')
        ff=pick(f,task=task,layer=layer).pivot(index='seed',columns='variant',values='delta_r_mean')
        null=pp[[f'perm{i}' for i in range(4)]].mean(axis=1)
        null_step=ff[[f'perm{i}' for i in range(4)]].mean(axis=1)
        count=int(((pp.base>0)&(pp.base>null)).sum())
        row=dict(task=task,layer=layer,baseline_mean_S=float(pp.base.median()),
                 perm_mean_S=float(null.median()),alpha_zero_mean_S=float(pp.alpha_zero.median()),
                 norm_unload_mean_S=float(pp.norm_unload.median()),
                 alignment_effect_seeds=count,verdict='LOCAL_ALIGNMENT_EFFECT' if count>=4 else 'NOT_ESTABLISHED',
                 dose_monotone_seeds=int(((pp.base>=pp.alpha_half)&(pp.alpha_half>=pp.alpha_zero)).sum()),
                 finite_more_sinking_seeds=int((ff.base<null_step).sum()),
                 finite_baseline_delta=float(ff.base.median()),finite_perm_delta=float(null_step.median()),
                 finite_paired_reduction=float((null_step-ff.base).median()))
        a_rows.append(row)
        cc=pick(c,task=task,layer=layer,dataset='next',actual='base',metric='Adam_fixed')
        ba=cc.pivot(index='seed',columns='prediction',values='balanced_accuracy')
        rho=cc.pivot(index='seed',columns='prediction',values='spearman')
        count=int(((ba.P2_iid>=.65)&(rho.P2_iid>=.5)&(ba.P2_iid-ba.iso>=.05)).sum())
        b_rows.append(dict(task=task,layer=layer,p2_ba=float(ba.P2_iid.median()),
                           p2_spearman=float(rho.P2_iid.median()),iso_ba=float(ba.iso.median()),
                           gram_ba=float(ba.gram.median()),criteria_seeds=count,
                           verdict='P2_SWITCH_DIRECTION_MATCH' if count>=4 else 'NOT_MATCHED'))
a,b=pd.DataFrame(a_rows),pd.DataFrame(b_rows)
a.to_csv(OUT/'alignment_verdict.csv',index=False)
b.to_csv(OUT/'phase2_verdict.csv',index=False)
keys=['dataset','task','signal','metric','layer','moment']
agg=p.groupby(keys)[['mean','median','sink_fraction']].agg(['median','min','max'])
agg.columns=['_'.join(x) for x in agg.columns]
agg.reset_index().to_csv(OUT/'pressure_aggregate.csv',index=False)
ra=r.groupby(['task','arm','step','layer'])[['delta_r_mean','delta_r_median','decreased_fraction','ce']].agg(['median','min','max'])
ra.columns=['_'.join(x) for x in ra.columns]
ra.reset_index().to_csv(OUT/'rollout_aggregate.csv',index=False)

lines=['# ViT: 整列への介入と相2の機構検証（1025）','',
       '前回の層別観測を知った上での探索的追試。コード・判定を77a97ce7で新しい計測前に凍結。SL・全5 seed・task10/60末の10状態を起点とし、最終分類headを操作した。各FFN自身のfc2を独立に分解した検査ではない。','',
       '**通常学習中の前段の沈降は、ラベルを替えない50更新でも再現した。一方、最終FFNの沈降を読み出し整列に固有の効果とする判定は満たさず、共同LS型の相2予測も今回のViTの切替勾配とは対応しなかった。** 整列介入の基準を満たしたのはtask10末のFFN3〜5。task60末では全層が未達。共同LS相2の方向対応基準は全層・両時点で0/5 seed。','',
       'Issaが追加した入力ばらつき仮説は[画像依存共分散とfull-batch対照](covariance_summary.md)で別に検証する。','',
       '「切替と通常学習が沈めるunitがなぜ分かれるか」は[同じunitの対応検査](unit_overlap.md)に整理した。最終FFNでも向きが反対なのは約63%、他層は約51〜55%で、unitが二分された結果ではない（task60・次課題入力）。','',
       '## A. 相1：headの大きさを保って整列だけを崩す','',
       '元headを、旧ラベルと特徴の整列成分Lと残りRへ分けた。Rは残し、Lを旧ラベルpermutation由来の直交方向へ置換する対照を4本作った。元headとFrobenius normが同じで、元の整列は0。logitsと確率も再計算して、次課題11/61の独立ラベル期待勾配を測った。61は仮想次課題。','',
       'Sは全パラメータ更新を固定probeのμ/σへ運んだ量で、正なら沈む。以下は保存Adam分母固定の全unit平均Sのseed中央値。「条件成立」は元headでS>0かつ4対照平均よりSが大きいseed数。4/5以上をLOCAL_ALIGNMENT_EFFECTと事前に定義した。','',
       '| 課題末 | FFN | 元head S | 整列除去対照 S | 条件成立 | 実Adam 1更新でも元headがより沈む |',
       '|---:|---:|---:|---:|---:|---:|']
for row in a.itertuples():
    lines.append(f'| {row.task} | {row.layer} | {row.baseline_mean_S:+.3g} | {row.perm_mean_S:+.3g} | {row.alignment_effect_seeds}/5 | {row.finite_more_sinking_seeds}/5 |')
lines += ['',
          '有限1更新では保存momentum・二次moment・stepを含め、実際にbodyを動かしてμ/σ差を測った。250画像の期待勾配に対するAdam stepであり、元のminibatch Adam更新の期待値ではない。大きさを変えるalpha=1/.5/0のdose、整列を除いた残りを元normへ拡大する対照もalignment_verdict.csvとpressures.csvに保持した。整列を除いても沈降が残る場合、整列は必要条件とは言わない。','',
          '前回の「最終FFNの64.8%を整列成分が沈める」という符号測定は再現したが、それだけでは整列に固有の効果の証拠にならない。今回はhead normを保つ除去対照にも最終FFNの沈降が残り、元headが対照より沈めるという登録条件は3/5 seedに留まった（両時点）。成分が沈めることと、整列が特別に沈降を増やすことを分ける必要がある。','',
          '## B. 相2：共同fit模型を正しい入力条件で検査','',
          '前回の「通常学習との一致が約50%」だけでは相2の反証にならない。既存の共同fit式は独立な旧ラベルを最小二乗でfitした後の、**切替期待**の模型だった。今回はbiasを定数特徴として含め、新入力での予測にも旧入力のGramの擬似逆を使った。等方head模型と、逆でないGramを使う模型も同じ微分経路で比較した。','',
          '| 課題末 | FFN | 相2 BA | 相2 Spearman | 等方対照 BA | Gram対照 BA | 事前条件を満たすseed |',
          '|---:|---:|---:|---:|---:|---:|---:|']
for row in b.itertuples():
    lines.append(f'| {row.task} | {row.layer} | {pct(row.p2_ba)} | {row.p2_spearman:+.3f} | {pct(row.iso_ba)} | {pct(row.gram_ba)} | {row.criteria_seeds}/5 |')
lines += ['',
          '事前条件はBA≥65%、順位相関≥0.5、等方対照よりBAが5ポイント以上高い、を同じseedで満たすこと。4/5以上で方向対応とする。BAは沈むunitと浮くunitの各的中率の平均。常に一方の符号を答える対照は50%。予測ゼロを除いて的中率を上げる扱いはしていない。','',
          '### 適用条件の実測','',
          '| 課題末 | 有効rankの範囲 / 250画像 | LS予測とのlogit差の比 | LSの残差比 | 通常学習のCE補正 / 実信号ノルム |',
          '|---:|---:|---:|---:|---:|']
for task in (10,60):
    ss=pick(s,task=task)
    lines.append(f'| {task} | {ss["rank"].min()}–{ss["rank"].max()} | {ss.ls_prediction_gap.median():.3g} | {ss.ls_residual_ratio.median():.3g} | {ss.ce_correction_ratio.median():.3g} |')
lines += ['',
          'LSはC(Q−u)を教師とする一様点二次近似の共同最小二乗解。実網はCEを学習しているため、LSとの差がそのまま学習不足を意味するわけではない。actual_fit=ls_fit+fit_gap+CE補正、actual_switch=ls_switch+switch_gap+CE補正を信号・位置で検算した。大きな項が相殺して小さい実勾配になる場合、各項のnorm比を因果寄与率として足し上げない。','',
          '### なぜ通常学習と同じ式にならないか','',
          '固定Hbar上でW*=Hbar⁺Yとすると、旧課題の二乗損失のbody信号は(Hbar W*−Y)W*ᵀ。独立・等分散な旧ラベルの平均を取ると、射影残差と擬似逆の直交性から期待値は0になる。特に250画像に対して有効rankが250なら、個別課題でもLS残差はほぼ0になる。従ってこの条件付き模型は、通常学習中に前段が一方向へ沈むという予測をそのまま与えない。実学習では特徴とラベルが依存し、CE・共同fitからのずれ・momentumがある。そのどれが持続沈降を作るかは、別の機構命題になる。','',
          '正の対照として、固定特徴に独立一様ラベル256組を与えてLSでfitした。次入力でのMonte Carlo平均信号と理論の相対L2差（10状態）の範囲は '
          f'{s.mc_next_rel.min():.3f}–{s.mc_next_rel.max():.3f}。128組ずつのsplit差もstates.csvに保存。これは条件付き模型の実装検査であり、実ViTへの適用の証拠ではない。','',
          '## C. ラベルを替えずに50更新続けた実測','',
          '元のAdam状態を再開し、旧画像・旧ラベルを保つ腕と、同じ画像を独立に再ラベルする腕を、同じbatch順で50更新した。headとbodyをともに更新。表は各unitの開始状態からの差を取り、unit中央値、その後seed中央値。負が沈降。「沈降seed」は各seedのunit中央値が負の本数。','',
          '| 課題末 | FFN | 継続50更新 Δμ/σ | 沈降seed | 再ラベル50更新 Δμ/σ |',
          '|---:|---:|---:|---:|---:|']
for task in (10,60):
    for layer in range(1,7):
        rr=pick(r,task=task,layer=layer,step=50,arm='continue')
        switch=med(r,'delta_r_median',task=task,layer=layer,step=50,arm='relabel')
        lines.append(f'| {task} | {layer} | {rr.delta_r_median.median():+.5g} | {int((rr.delta_r_median<0).sum())}/5 | {switch:+.5g} |')
lines += ['',
          'この50更新は、元走で既に100% fitした課題末からの継続。前回の課題内100–500更新と同じ区間ではない。ここで沈降が再現しなくても、前回の実測を撤回する理由にはせず、通常学習の段階依存性として読む。旧/新ラベルの対応だけを変えた対照で、入力分布の変更は含めていない。','',
          '![実際に更新したときの変位](actual_updates.png)','',
          '![相2と対照の方向対応](phase2_comparison.png)','',
          '## 検算と限界','',
          '- 全10状態で元checkpointとsourceのSHAを照合、元CE/精度を再現。前回の次入力期待勾配を再現。head normと直交制約、LS正規方程式、共分散恒等式、全パラメータ位置の帳簿を検算した。',
          '- 保存Adamから計算した有限更新をtorch.optim.Adamとseed0/task10で照合した。有限差分とJVPの差はfinite_steps.csvに全状態・全層を保存。ReLUの境界を跨ぐ有限変位と一次微分は同一とは仮定しない。',
          '- 相1/相2の期間積分された寄与率や長期の必要十分性を確定する検査ではない。方向一致、同一状態の局所介入、実際の50更新の観測を分けて報告する。',
          '- 全5 seedを使用。unit数1536を独立なseed数の代わりにしていない。多層の判定は探索的な記述であり、事後解析であることを外さない。',
          '- 元の学習データ・checkpoint・他の走は変更していない。生配列とログはbackup_manifest.jsonの恒久保存先。']
(OUT/'summary.md').write_text('\n'.join(lines)+'\n')
(OUT/'verdict.json').write_text(json.dumps(dict(tier='exploratory_followup',starting_states=10,
    alignment=a_rows,phase2=b_rows,ordinary_iid_ls_mean='zero_under_fixed_feature_assumptions'),indent=2)+'\n')

plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True,sharex=True)
for col,task in enumerate((10,60)):
    for variant,color,label in [('base','#c65436','Original alignment'),('perms','#787878','Norm-matched alignment removal')]:
        if variant=='base':
            sub=pick(f,task=task,variant='base').groupby('layer').delta_r_mean.median()
        else:
            sub=f[(f.task==task)&f.variant.str.startswith('perm')].groupby(['seed','layer']).delta_r_mean.mean().groupby('layer').median()
        axes[0,col].plot(sub.index,1000*sub.values,'o-',color=color,label=label)
    axes[0,col].set(title=f'Task {task} end: one actual Adam step',ylabel='Mean change in mean / SD (x 1000)')
    for arm,color,label in [('continue','#286ca5','Continue old labels'),('relabel','#159b8d','Independent new labels')]:
        sub=pick(r,task=task,step=50,arm=arm).groupby('layer').delta_r_median.median()
        axes[1,col].plot(sub.index,sub.values,'o-',color=color,label=label)
    axes[1,col].set(title='50 mini-batch updates on the same images',ylabel='Median unit change in mean / SD',xlabel='FFN block (1 = nearest input)')
    for ax in axes[:,col]:
        ax.axhline(0,color='gray',ls=':',lw=1)
        ax.set(xticks=range(1,7))
        ax.grid(alpha=.18)
axes[0,0].legend(frameon=False,fontsize=8)
axes[1,0].legend(frameon=False,fontsize=8)
fig.suptitle('Shared-head ViT: paired interventions, medians of 5 seeds\nNegative change means sinking; all updates include saved Adam momentum')
for ext in ('png','pdf'): fig.savefig(OUT/f'actual_updates.{ext}',dpi=180)
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True,sharey=True)
for ax,task in zip(axes,(10,60)):
    for pred,color,label in [('P2_iid','#7661a8','Joint-LS phase-2 model'),('iso','#888888','Isotropic readout control'),('gram','#c65436','Gram (not inverse) control')]:
        sub=pick(c,task=task,dataset='next',metric='Adam_fixed',actual='base',prediction=pred).groupby('layer').balanced_accuracy.agg(['median','min','max'])
        ax.plot(sub.index,100*sub['median'],'o-',color=color,label=label)
        if pred=='P2_iid': ax.fill_between(sub.index,100*sub['min'],100*sub['max'],color=color,alpha=.13)
    ax.axhline(50,color='gray',ls=':',lw=1)
    ax.set(title=f'Task {task} end -> next inputs',ylim=(0,100),xticks=range(1,7),xlabel='FFN block',ylabel='Balanced sign accuracy (%)')
    ax.grid(alpha=.18)
axes[0].legend(frameon=False,fontsize=8)
fig.suptitle('Switch prediction uses the OLD feature Gram; saved Adam denominator\nMedians of 5 seeds, shaded range for phase 2')
for ext in ('png','pdf'): fig.savefig(OUT/f'phase2_comparison.{ext}',dpi=180)
plt.close(fig)

def digest(path):
    sha=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''): sha.update(chunk)
    return sha.hexdigest()

files=[dict(source=str(p),backup=str(p),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(ARCHIVE.rglob('*')) if p.is_file()]
(OUT/'backup_manifest.json').write_text(json.dumps(dict(archive=str(ARCHIVE),files=files,
    note='Raw artifacts written directly to permanent storage; source and backup coincide.'),indent=2)+'\n')
print((OUT/'summary.md').read_text())

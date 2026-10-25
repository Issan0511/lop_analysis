"""Report all registered states, including negative or mixed transport results."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/vit_transport_1025'
ARCHIVE=Path('/home/issan/Projects/obsidian-research-data/vit_transport_1025')
PRIOR=ARCHIVE.parent/'vit_phases_1025/raw'
for mode,n in [('carriers',10),('replay',10),('freeze',5),('probe',20)]:
    meta=json.loads((OUT/f'{mode}_run.json').read_text())
    assert meta['status']=='complete' and meta['states']==n
c=pd.read_csv(OUT/'carriers.csv')
a=pd.read_csv(OUT/'ratios.csv')
b=pd.read_csv(OUT/'replay.csv')
f=pd.read_csv(OUT/'freeze.csv')
probes=pd.read_csv(OUT/'probe_drifts.csv')
probe_states=pd.read_csv(OUT/'probe_states.csv')
probe_transport=pd.read_csv(OUT/'probe_transport.csv')
old_states=pd.read_csv(ROOT/'results/vit_phases_1025/states.csv')
confidence=old_states.query('arm=="SL" and layer==1 and stage in ["mid","end"]')[['seed','task','stage','accuracy','ce','pmax']]
confidence.to_csv(OUT/'original_fit_confidence.csv',index=False)
assert len(c)==9360 and len(a)==360 and len(b)==600 and len(f)==1200
assert len(probes)==180 and len(probe_states)==360
assert len(probe_transport)==180


def pick(df,**kw):
    keep=np.ones(len(df),dtype=bool)
    for k,v in kw.items(): keep &= (df[k]==v).to_numpy()
    return df.loc[keep]


ar=[]
for task in (10,60):
    for layer in range(1,7):
        for signal in ('fit','switch','load'):
            z=pick(a,task=task,layer=layer,signal=signal,metric='Adam_fixed')
            ar.append(dict(task=task,layer=layer,signal=signal,
                           upstream_norm_share=float(z.upstream_norm_share.median()),
                           upstream_share_min=float(z.upstream_norm_share.min()),
                           upstream_share_max=float(z.upstream_norm_share.max()),
                           upstream_75_seeds=int((z.upstream_norm_share>=.75).sum())))
av=pd.DataFrame(ar); av.to_csv(OUT/'carrier_verdict.csv',index=False)
fr=[]
for target in (1,6):
    for arm in ('continue','relabel'):
        base=pick(b,task=60,variant='baseline',step=50,layer=target,label_arm=arm).set_index('seed')
        own=pick(f,target=target,variant='freeze_own',step=50,layer=target,label_arm=arm).set_index('seed')
        up=pick(f,target=target,variant='freeze_upstream',step=50,layer=target,label_arm=arm).set_index('seed')
        for seed in range(5):
            x,y,z=base.loc[seed,'delta_mean'],own.loc[seed,'delta_mean'],up.loc[seed,'delta_mean']
            nz=abs(x)>1e-6
            own_keeps=bool(nz and x*y>0 and abs(y)>=.5*abs(x))
            up_reduces=bool(nz and x*z>0 and abs(z)<=.5*abs(x))
            fr.append(dict(seed=seed,target=target,label_arm=arm,baseline_delta=x,own_frozen_delta=y,
                           upstream_frozen_delta=z,baseline_ce=base.loc[seed,'ce'],
                           own_frozen_ce=own.loc[seed,'ce'],upstream_frozen_ce=up.loc[seed,'ce'],
                           own_keeps=own_keeps,up_reduces=up_reduces,paired_support=own_keeps and up_reduces,
                           up_reverses=bool(nz and x*z<0)))
fv=pd.DataFrame(fr); fv.to_csv(OUT/'freeze_verdict.csv',index=False)

# Post-hoc application of the same product identity to the ORIGINAL 100->500 interval.
original=[]; original_sources=[]
for seed in range(5):
    for task in (10,60):
        files=[PRIOR/f'SL_seed{seed}_t{task}_{stage}.npz' for stage in ('mid','end')]
        for p in files: original_sources.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        with np.load(files[0]) as mid,np.load(files[1]) as end:
            for layer in range(1,7):
                mu0,mu1=mid[f'L{layer}_mu'],end[f'L{layer}_mu']
                s0,s1=mid[f'L{layer}_sd']+1e-12,end[f'L{layer}_sd']+1e-12
                valid=mid[f'L{layer}_valid']&end[f'L{layer}_valid']
                dm=.5*(1/s0+1/s1)*(mu1-mu0)
                ds=.5*(mu0+mu1)*(1/s1-1/s0)
                dr=end[f'L{layer}_r']-mid[f'L{layer}_r']
                assert np.abs((dm+ds-dr)[valid]).max()<1e-10
                original.append(dict(seed=seed,task=task,layer=layer,delta_mean=float(dr[valid].mean()),
                                     delta_median=float(np.median(dr[valid])),mean_channel=float(dm[valid].mean()),
                                     width_channel=float(ds[valid].mean()),sd_ratio_median=float(np.median(s1[valid]/s0[valid])),
                                     mu_negative_fraction=float((mu1[valid]<0).mean())))
o=pd.DataFrame(original); o.to_csv(OUT/'original_100_500_channels.csv',index=False)
(OUT/'original_channel_sources.json').write_text(json.dumps(original_sources,indent=2)+'\n')

lines=['# ViTの層差：上流の更新と平均・幅を分離する（1025）','',
       '既存結果を知った探索的追試。測定前specはf7636441、数値精度のgate修正はd9f2661e。全5 seed。切替の整列成分は最終分類headについての定義であり、各FFNのfc2の整列そのものではない。','',
       '[導出と適用範囲](../../analysis/vit_transport_1025/theory.md)。出力の切替期待はlog C+KL(u||p)、通常学習はそこから正解方向の項を引く。正答後には出力の尺度への押しが逆になるが、各FFNが沈む符号は上流の経路と平均・幅の収支による。','',
       '時点の注意：task10は100更新ですでに全seedの訓練精度が100%だが、task60の100更新では61.2〜100%だった。100→500をすべて「完全にfitした後」と呼ばない。500更新時は両課題・全seedで100%。task10では精度100%のまま平均最大確率のseed中央値が0.9879→0.9978へ増えた。位置が動くことは正答率の変化を必要としない。元値はoriginal_fit_confidence.csv。','',
       '## A. 最終FFNへ押しを運ぶのはどこか','',
       '旧画像の通常学習fitと、次課題画像11/61の切替switch・整列成分load。保存Adam分母固定、momentum除外。比は||上流のunit押し||₂ / (||上流||₂+||自fc1||₂)。符号付き因果寄与率ではない。','',
       '| 課題末 | 信号 | 最終FFNの上流L2比・中央値 | seed範囲 | 75%以上のseed |',
       '|---:|---|---:|---:|---:|']
for rr in av.query('layer==6').itertuples():
    lines.append(f'| {rr.task} | {rr.signal} | {100*rr.upstream_norm_share:.1f}% | {100*rr.upstream_share_min:.1f}–{100*rr.upstream_share_max:.1f}% | {rr.upstream_75_seeds}/5 |')
lines+=['','最終FFNのloadで75%以上が4/5 seed以上、が登録仮説。全層の比はcarrier_verdict.csv。群別の平均・幅・符号はcarriers.csv、生unit配列はraw archiveに保持した。','',
        '課題60末の主なparameter群（各seed・対象層で絶対unit平均が最大の群を数える。全体が沈むことを意味しない）：','',
        '| 信号 | FFN | 最多の群 | 最大だったseed |', '|---|---:|---|---:|']
for signal in ('fit','load'):
    for layer in range(1,7):
        sub=pick(c,task=60,signal=signal,metric='Adam_fixed',layer=layer)
        largest=sub.loc[sub.groupby('seed')['mean'].apply(lambda v:v.abs().idxmax())]
        counts=largest.group.value_counts()
        lines.append(f'| {signal} | {layer} | {counts.index[0]} | {int(counts.iloc[0])}/5 |')
lines+=['','## B. 実際の50更新：どの項が位置を動かしたか','',
        '元の課題末から、旧ラベルの継続／同じ画像の独立再ラベル。元Adamと前回と同じbatch順を再現。各stepの有限変位を、自fc1／上流、平均／幅の4項へ分けて50 step分足した。以下はunit平均変位をseed間で**平均**した加法帳簿。負が沈降。中央値や沈降seed数をこの平均へ読み替えない。','',
        '| 課題末 | 腕 | FFN | 自fc1・平均 | 自fc1・幅 | 上流・平均 | 上流・幅 | 合計Δr |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
channels=['own_mean','own_width','upstream_mean','upstream_width']
for (task,arm,layer),rr in b.query('step==50').groupby(['task','label_arm','layer']):
    vals=rr[channels+['delta_mean']].mean()
    assert abs(vals[channels].sum()-vals.delta_mean)<1e-9
    lines.append(f'| {task} | {arm} | {layer} | '+' | '.join(f'{v:+.5g}' for v in vals)+' |')
lines+=['','四項が合計に一致することは恒等式の検算であり、機構の証拠そのものではない。実現した学習経路上の二要因Shapley差であり、相互作用の割当は一意ではない。次の固定介入を別に行った。','',
        '課題60末・再ラベル後の最終FFNを時系列で見る（unit平均のseed平均、各時点までの累積変位）：','',
        '| 更新数 | 自fc1・平均 | 自fc1・幅 | 上流・平均 | 上流・幅 | 合計Δr |',
        '|---:|---:|---:|---:|---:|---:|']
timeline=[]
for step,rr in pick(b,task=60,label_arm='relabel',layer=6).groupby('step'):
    vals=rr[channels+['delta_mean']].mean()
    lines.append(f'| {step} | '+' | '.join(f'{v:+.6g}' for v in vals)+' |')
    timeline.append(dict(step=int(step),**vals.to_dict()))
pd.DataFrame(timeline).to_csv(OUT/'last_block_relabel_timeline.csv',index=False)
v=pick(b,task=60,label_arm='relabel',layer=6,step=50)
means=v[channels+['delta_mean']].mean()
up_fraction=(means.upstream_mean+means.upstream_width)/means.delta_mean
lines += ['',f'50更新では、seed平均の実変位に対する上流二項の比が{up_fraction:.4%}。これは符号付き帳簿の比であり、上流を固定した別軌道の因果割合ではない。個々のseedでは上流平均と幅が逆向きの場合もあり、幅の比が100%を超えることもある。切替後50更新には新ラベルへの学習も含まれるので、全変位を整列由来の相1だけへ割り当てない。','',
        '## C. 上流と自fc1を別々に固定して学習','',
        'task60末、FFN1または6が対象。上流固定でも対象fc1と後段は学習する。自fc1固定でも上流・後段は学習する。固定parameterはgrad=NoneでAdamのmomentumによる移動も止めた。各値は50更新後のunit平均Δr、そのseed中央値。','',
        '| 腕 | FFN | 無介入 | 自fc1固定 | 上流固定 | 自fc1固定でも半分以上残る | 上流固定で同方向・半分以下 | 両条件同じseed | 上流固定で逆転 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
verdicts=[]
for (arm,target),rr in fv.groupby(['label_arm','target']):
    count=int(rr.paired_support.sum())
    verdicts.append(dict(label_arm=arm,target=int(target),paired_support=count,upstream_freeze_reverses=int(rr.up_reverses.sum()),
                         verdict='TRANSPORT_SUPPORTED' if count>=4 else 'NOT_ESTABLISHED'))
    lines.append(f'| {arm} | {target} | {rr.baseline_delta.median():+.5g} | {rr.own_frozen_delta.median():+.5g} | {rr.upstream_frozen_delta.median():+.5g} | {int(rr.own_keeps.sum())}/5 | {int(rr.up_reduces.sum())}/5 | {count}/5 | {int(rr.up_reverses.sum())}/5 |')
lines+=['','登録判定は両条件が4/5以上。表では同一seedで両条件を満たす本数を使って保守的に判定する。上流固定で逆方向になった場合、同方向の条件を満たしたことに置換しない。逆転の有無はfreeze_verdict.csvに残す。','',
        '最終FFNの再ラベルは、上流固定で全5 seedが沈降から浮上へ逆転した。そのため「同方向のまま半分以下」という登録判定は不成立（0/5）だが、上流を止めても元の沈降が残るという結果ではない。自fc1固定では全5 seedが沈み、4/5で無介入の半分以上が残った。この符号逆転は登録基準と区別した観測結果として扱う。第1FFNの通常学習では上流固定でも全5 seedで沈降が残った。','',
        '| 腕 | FFN | 無介入CE | 自fc1固定CE | 上流固定CE |', '|---|---:|---:|---:|---:|']
for (arm,target),rr in fv.groupby(['label_arm','target']):
    lines.append(f'| {arm} | {target} | {rr.baseline_ce.median():.5g} | {rr.own_frozen_ce.median():.5g} | {rr.upstream_frozen_ce.median():.5g} |')
lines+=['','上流固定は多くの自由度を止める。性能を保つ選択的な沈降防止の実験ではない。自fc1固定時の残存と併せて、位置変化の運搬経路を検査する。','',
        '## 元の学習区間100→500更新にも、平均と幅を分ける（事後）','',
        'これは追加50更新とは異なる元走の区間。同じ積の対称分解を保存配列へ適用した。下表もunit平均、そのseed平均の加法帳簿。','',
        '| 課題 | FFN | 平均の項Δr | 幅の項Δr | 合計Δr | σ比のseed中央値 |',
        '|---:|---:|---:|---:|---:|---:|']
for (task,layer),rr in o.groupby(['task','layer']):
    lines.append(f'| {task} | {layer} | {rr.mean_channel.mean():+.5g} | {rr.width_channel.mean():+.5g} | {rr.delta_mean.mean():+.5g} | {rr.sd_ratio_median.median():.4f} |')
lines+=['','同じ元区間の両端点で、fc1重みとその入力を入れ替えた四項（固定検証画像、unit平均のseed平均）：','',
        '| 課題 | FFN | 自fc1・平均 | 自fc1・幅 | 上流・平均 | 上流・幅 | 合計Δr |',
        '|---:|---:|---:|---:|---:|---:|---:|']
for (task,layer),rr in pick(probe_transport,population='validation').groupby(['task','layer']):
    vals=rr[channels+['delta_mean']].mean()
    assert abs(vals[channels].sum()-vals.delta_mean)<1e-9
    lines.append(f'| {task} | {layer} | '+' | '.join(f'{v:+.5g}' for v in vals)+' |')
lines+=['','この表は元区間の端点を組み合わせた分解。途中のAdam状態が保存されていないため、更新ごとの実経路を再生した分担ではない。継続50更新のstep積分と同一視しない。']
lines+=['','## 入力集合を替えても層差は残るか','',
        '保存された100/500更新時のモデルを、学習画像250枚・次課題画像250枚・固定検証画像64枚へ適用した。各集合は前後で同一。下表はunitの前後差の中央値を取り、さらにseed中央値。括弧内はunit差の中央値が負のseed数。probe追補は06b00f0bで計測前に登録。','',
        '| 課題 | FFN | 学習画像Δr（沈降seed） | 次課題画像Δr（沈降seed） | 固定検証画像Δr（沈降seed） |',
        '|---:|---:|---:|---:|---:|']
for task in (10,60):
    for layer in range(1,7):
        cells=[]
        for pop in ('train','next','validation'):
            rr=pick(probes,task=task,layer=layer,population=pop)
            cells.append(f'{rr.delta_median.median():+.5g} ({int((rr.delta_median<0).sum())}/5)')
        lines.append(f'| {task} | {layer} | '+' | '.join(cells)+' |')
lines+=['','これは入力分散の大きさを操作した対照ではない。画像の内容・クラス・学習済みか・集合サイズが違う。どの集合へ一般化する層差かを確認する検査であり、集合間の差を分散の因果効果と呼ばない。','',
        '### 相対位置の移動と閉じたunitは別','',
        '課題60・500更新時、各集合の全画像で前活性が0以下だったunitの割合。seed中央値。有限個の画像での無応答であり、全入力空間で死んだことの証明ではない。','',
        '| FFN | 学習画像 | 次課題画像 | 固定検証画像 |', '|---:|---:|---:|---:|']
for layer in range(1,7):
    vals=[pick(probe_states,task=60,stage='end',layer=layer,population=pop).closed_unit_fraction.median() for pop in ('train','next','validation')]
    lines.append(f'| {layer} | '+' | '.join(f'{100*v:.2f}%' for v in vals)+' |')
lines+=['','## 検証範囲と未確定の点','',
        '- 全10状態の元source/checkpoint SHA、元CE、既存の一次押しと50更新のrを照合した。数値精度の問題は閾値を緩めず、元と同じLinear計算で修正した。失敗gateもarchiveに保持。',
        '- 群ごとのJVP和、自fc1解析式、後段の直接寄与ゼロ、有限四項の和、固定parameterの不変性を検算した。各結果のchecks.jsonが記録。',
        '- これらはこの保存状態と継続区間における条件付きの機構。深さだけから沈降を保証する定理、入力分散単独の必要十分性、長期可塑性喪失の媒介率は未同定。',
        '- 最終分類headの整列の検査と、各FFNのfc2列の整列は別である。ここでは前者の定義を保った。',
        '- 普通の学習を相2と呼ぶ現象上の分類と、旧ラベル平均の共同LS切替模型を区別した。以前のLS不一致を成功へ差し替えていない。','',
        '![経路の内訳](carrier_map.png)','',
        '![実変位の四項](finite_transport.png)','',
        '![固定介入](freeze_control.png)','',
        '![入力集合別の層差](probe_comparison.png)']
(OUT/'summary.md').write_text('\n'.join(lines)+'\n')
(OUT/'verdict.json').write_text(json.dumps(dict(tier='exploratory_followup',freeze=verdicts),indent=2)+'\n')

plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':9})
groups=['embedding']+[f'L{i}_{part}' for i in range(1,7) for part in ('attention','norm2','fc1','fc2')]
fig,axes=plt.subplots(2,1,figsize=(13,6.8),constrained_layout=True)
for ax,signal in zip(axes,('fit','load')):
    sub=pick(c,task=60,metric='Adam_fixed',signal=signal)
    v=sub.pivot(index=['seed','layer'],columns='group',values='mean')[groups]
    v=v.div(v.abs().sum(axis=1),axis=0).groupby('layer').median()
    im=ax.imshow(v.values,aspect='auto',vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set(yticks=range(6),yticklabels=range(1,7),xticks=range(len(groups)),xticklabels=groups,
           ylabel='Target FFN block',title=f'Task 60 end, {signal}: signed group mean / sum of absolute group means')
    ax.tick_params(axis='x',rotation=60,labelsize=8)
fig.colorbar(im,ax=axes,shrink=.7,label='Positive pressure lowers mean / SD')
fig.suptitle('Parameter groups carrying the pressure; medians of seedwise shares')
for ext in ('png','pdf'):fig.savefig(OUT/f'carrier_map.{ext}',dpi=180)
plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
colors=['#e9ad78','#ba6834','#8db9d5','#276e9b']
labels=['Own fc1: mean','Own fc1: width','Upstream: mean','Upstream: width']
for col,task in enumerate((10,60)):
    for row,arm in enumerate(('continue','relabel')):
        sub=pick(b,task=task,label_arm=arm,step=50).groupby('layer')[channels+['delta_mean']].mean()
        pos=np.zeros(6);neg=np.zeros(6)
        for channel,color,label in zip(channels,colors,labels):
            y=sub[channel].to_numpy();base=np.where(y>=0,pos,neg)
            axes[row,col].bar(sub.index,y,bottom=base,color=color,label=label,width=.7)
            pos+=np.maximum(y,0);neg+=np.minimum(y,0)
        axes[row,col].plot(sub.index,sub.delta_mean,'ko-',label='Actual total',ms=4)
        axes[row,col].axhline(0,color='gray',ls=':',lw=1)
        axes[row,col].set(title=f'Task {task}: {arm}, 50 steps',xticks=range(1,7),ylabel='Change in mean / SD')
        if row==1:axes[row,col].set_xlabel('FFN block')
axes[0,0].legend(frameon=False,fontsize=7)
fig.suptitle('Finite-step transport ledger, averaged over units and 5 seeds\nNegative change means sinking; includes actual Adam momentum')
for ext in ('png','pdf'):fig.savefig(OUT/f'finite_transport.{ext}',dpi=180)
plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(9,6.5),constrained_layout=True)
for col,target in enumerate((1,6)):
    for row,arm in enumerate(('continue','relabel')):
        sub=pick(fv,target=target,label_arm=arm)
        names=['baseline_delta','own_frozen_delta','upstream_frozen_delta']
        for rr in sub.itertuples():axes[row,col].plot(range(3),[getattr(rr,k) for k in names],'-',color='gray',alpha=.4,lw=.8)
        axes[row,col].plot(range(3),sub[names].median(),'o-',color='#276e9b',lw=2)
        axes[row,col].axhline(0,color='gray',ls=':',lw=1)
        axes[row,col].set(title=f'FFN{target}: {arm}',xticks=range(3),xticklabels=['Baseline','Own fc1 fixed','Upstream fixed'],ylabel='50-step change in mean / SD')
        axes[row,col].tick_params(axis='x',labelsize=8)
fig.suptitle('Task 60 end: paired freeze interventions\nUnit means; blue = seed median, gray = individual seeds')
for ext in ('png','pdf'):fig.savefig(OUT/f'freeze_control.{ext}',dpi=180)
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True,sharey=True)
for ax,task in zip(axes,(10,60)):
    for pop,color,label in [('train','#c65436','Current training images (250)'),('next','#159b8d','Next-task images (250)'),('validation','#286ca5','Fixed validation images (64)')]:
        sub=pick(probes,task=task,population=pop).groupby('layer').delta_median.median()
        ax.plot(sub.index,sub.values,'o-',color=color,label=label)
    ax.axhline(0,color='gray',ls=':',lw=1)
    ax.set(title=f'Task {task}: original updates 100 to 500',xticks=range(1,7),xlabel='FFN block',ylabel='Median unit change in mean / SD')
    ax.grid(alpha=.18)
axes[0].legend(frameon=False,fontsize=8)
fig.suptitle('Input-set dependence of the blockwise pattern; medians of 5 seeds')
for ext in ('png','pdf'):fig.savefig(OUT/f'probe_comparison.{ext}',dpi=180)
plt.close(fig)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()
files=[dict(source=str(p),backup=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(ARCHIVE.rglob('*')) if p.is_file()]
(OUT/'backup_manifest.json').write_text(json.dumps(dict(archive=str(ARCHIVE),files=files),indent=2)+'\n')
print((OUT/'summary.md').read_text())

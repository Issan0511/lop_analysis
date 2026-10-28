"""Small static figures using only saved diagnostics; no model loading."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family']='Noto Sans CJK JP'

ROOT=Path('/home/issan/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/posthoc_codex_bodyrank')
DEST=Path('/home/issan/Projects/claude/wt/vit_long_1024/results/vit_long_1024_2blk_rlfix')
colors={'SL':'#b33939','R0L':'#227c9d'}

fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for arm,t,style in [('SL',10,'--'),('SL',100,'-'),('R0L',100,'-')]:
    z=np.load(ROOT/f'decompose_{arm}_t{t}_n256.npz')
    fields=z['fields'].tolist(); a=z['values']
    n=np.sqrt(a[:,4:8,fields.index('total_sq')].mean(1))
    sigma=a[:,0,fields.index('final.sigma')]
    axs[0,0].plot(np.sort(n/np.median(n)),np.arange(1,len(n)+1)/len(n),label=f'{arm} {t}',color=colors[arm],ls=style)
    paths=np.load(ROOT/f'paths_{arm}_t{t}.npz')
    nv=np.sqrt(paths['norms_sq'][:,:,1,0])*sigma[:,None]
    ratios=np.quantile(nv,.99,axis=0)/np.median(nv,axis=0)
    axs[0,1].plot([0,1,2,3],ratios[:4],marker='o',color=colors[arm],ls=style,label=f'{arm} {t}')
axs[0,0].set(xscale='log',xlabel='勾配ノルム / 中央値（PCの二乗平均）',ylabel='画像の累積割合',title='勾配の大きさの集中')
axs[0,0].legend()
axs[0,1].set(xticks=[0,1,2,3],xticklabels=['元の微分','score 0除去','score 1除去','両方除去'],ylabel='σ補正後の99%点 / 中央値',title='順伝播を変えず微分経路を除く（PC0）')
q=json.loads((ROOT/'quantile_model.json').read_text())
for arm in ('SL','R0L'):
    r=[v for v in q if v['arm']==arm and v['t']>0]
    axs[1,0].plot([v['t'] for v in r],[v['pc']['pr'] for v in r],color=colors[arm],marker='o',label=f'{arm}: 接核の参加比')
    axs[1,0].plot([v['t'] for v in r],[v['pc']['norm_only_pr'] for v in r],color=colors[arm],marker='s',ls='--',label=f'{arm}: 大きさだけの参加数')
    for j,v in enumerate(r):
        axs[1,1].scatter(v['observed'],v['pc']['predicted_quantile'],color=colors[arm])
        offset = (10, 5+13*j) if arm == 'R0L' else (-25,8)
        axs[1,1].annotate(f'{arm}{v["t"]}',(v['observed'],v['pc']['predicted_quantile']),xytext=offset,textcoords='offset points',fontsize=7,arrowprops=dict(arrowstyle='-',lw=.4,color=colors[arm]))
axs[1,0].set(xlabel='保存した課題末',ylabel='参加比 / 参加数',title='大きさの集中が参加比を下げる')
axs[1,0].legend(fontsize=7)
axs[1,1].plot([0,600],[0,600],color='gray',ls=':')
axs[1,1].scatter(q[0]['observed'],q[0]['pc']['predicted_quantile'],color='black',marker='x')
axs[1,1].annotate('新品',(q[0]['observed'],q[0]['pc']['predicted_quantile']),fontsize=8)
axs[1,1].set(xlabel='実測 T90',ylabel='予測 T90（PCの分位模型）',title='遅い分位への置換だけでは合わない',xlim=(0,650),ylim=(0,620))
for ax in axs.ravel():ax.grid(alpha=.2)
fig.savefig(DEST/'codex_bodyrank_diagnostics.png',dpi=160)
plt.close(fig)

def get(name):
    p=ROOT/name
    return json.loads(p.read_text()) if p.exists() else None

comparison=get('prediction_comparison.json')
if comparison:
    fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    canonical=all(r.get('canonical_actual') and r.get('canonical_refined') for r in comparison)
    groups={}
    for r in comparison:
        if canonical:
            key=(r['canonical_actual'],r['canonical_refined'],'新品' if not r['t'] else r['arm'])
            groups.setdefault(key,[]).append(str(r['t']) if r['t'] else '')
        elif r['t'] and r['actual'] and r['frozen']:
            key=(r['actual'],r['frozen'],r['arm'])
            groups.setdefault(key,[]).append(str(r['t']))
    r0label=0
    for (x,y,a),ts in groups.items():
        axs[0,0].scatter(x,y,color=colors.get(a,'#222222'))
        if a=='R0L':
            offset=(15,{'10':300,'40':350,'60':250,'100':200}.get(ts[0],220));coords='data'
        elif a=='SL':
            offset={'10':(-20,12),'40':(6,-15),'60':(-45,10),'100':(8,-16)}.get(ts[0],(5,5));coords='offset points'
        else:offset=(-15,12);coords='offset points'
        axs[0,0].annotate(a+','.join(ts),(x,y),xytext=offset,textcoords=coords,fontsize=8,arrowprops=dict(arrowstyle='-',lw=.4,color=colors.get(a,'#222222')))
    axs[0,0].plot([0,750],[0,750],ls=':',color='gray')
    if canonical:
        axs[0,0].fill_between([0,750],[0,562.5],[0,937.5],color='gray',alpha=.1,label='±25%')
        axs[0,0].legend(fontsize=8,loc='upper left')
    axs[0,0].set(xlabel='同条件の実網 T90',ylabel='予測 T90',title='元のbatch条件：5更新で再線形化' if canonical else '64に分割：固定体接核の予測',xlim=(0,750),ylim=(0,750))
    for ax,t,title in [(axs[0,1],0,'新品：初期接核の固定は外れる（64に分割）'),(axs[1,0],100,'後期 SL：固定と再線形化（64に分割）')]:
        for prefix,mode,label,col in [('replay','actual','実網','#222222'),('replay','tangent','固定体接核','#b33939'),('refresh25','tangent','25更新で再線形化','#227c9d'),('refresh5','tangent','5更新で再線形化','#d28b17')]:
            d=get(f'{prefix}_{mode}_SL_t{t}.json')
            if t==0 and prefix=='replay' and mode=='tangent':d=get('complete_tangent_SL_t0.json') or d
            if d:
                ax.plot([v['step'] for v in d['records']],[v['acc'] for v in d['records']],label=label,color=col)
        ax.axhline(.9,ls=':',color='gray');ax.set(xlabel='更新数',ylabel='暗記正解率',title=title,ylim=(0,1.02));ax.legend(fontsize=9)
    vals=[]
    for prefix in ['replay','q4_temp4','q4_sigma']:
        d=get(prefix+'_actual_SL_t100.json');vals.append(d['T90'] if d else None)
    if all(v is not None for v in vals):
        axs[1,1].bar(range(3),vals,color=['#555555','#b33939','#227c9d'],alpha=.75)
        axs[1,1].errorbar([1,2],[350,450],yerr=[100,50],fmt='o',color='black',capsize=9,label='事前予測の幅')
        for i,v in enumerate(vals):axs[1,1].text(i,v+10,str(v),ha='center')
        axs[1,1].set(xticks=range(3),xticklabels=['基準','注意を軟化','逆尺度を補正'],ylabel='T90',title='SL100：介入の予測と実測（64に分割）')
        axs[1,1].legend(fontsize=9)
    for ax in axs.ravel():ax.grid(alpha=.2)
    fig.savefig(DEST/'codex_bodyrank_prediction.png',dpi=160);plt.close(fig)

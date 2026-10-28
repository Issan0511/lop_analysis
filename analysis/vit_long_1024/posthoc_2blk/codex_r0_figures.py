"""Standalone publication-friendly plots and compact numeric tables."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path('/home/issan/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix')
OUT=ROOT/'posthoc_r0'
RESULT=Path('/home/issan/Projects/claude/wt/vit_long_1024/results/vit_long_1024_2blk_rlfix')


def read(name):return json.loads((OUT/name).read_text())
def write_csv(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    d=read('codex_checkpoints.json')
    fig,ax=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    colors={'R0':'#b9452b','S':'#296fa3','R0L':'#ca9e26','SL':'#4d846a'}
    flat=[]
    for arm in ['R0','S']:
        rows=[r for r in d['rows'] if r['arm']==arm]
        t=[r['task'] for r in rows];col=colors[arm]
        ax[0,0].plot(t,[r['m2_norm'] for r in rows],'-o',ms=3,c=col,label=arm)
        ax[0,1].plot(t,[r['attention1']['cls_mass'] for r in rows],'-o',ms=3,c=col,label=arm)
        ax[0,2].plot(t,[r['contrib']['attn1_cls']/r['contrib']['attn1'] for r in rows],'-o',ms=3,c=col,label=arm)
        ax[1,0].plot(t,[r['z']['all']['dead'] for r in rows],'-o',ms=3,c=col,label=arm+' 全token')
        ax[1,0].plot(t,[r['z']['cls']['dead'] for r in rows],'--',c=col,label=arm+' CLS')
        ax[1,1].plot(t,[r['cos']['median'] for r in rows],'-o',ms=3,c=col,label=arm)
        for r in rows:
            flat.append(dict(arm=arm,task=r['task'],m2=r['m2_norm'],cls_attention=r['attention1']['cls_mass'],
                             a1=r['contrib']['attn1'],a1_cls=r['contrib']['attn1_cls'],a1_patch=r['contrib']['attn1_patch'],
                             f0=r['contrib']['ffn0'],dead=r['z']['all']['dead'],cls_dead=r['z']['cls']['dead'],
                             mu=r['z']['all']['mean'],width=r['z']['all']['width'],r=r['z']['all']['r'],cos=r['cos']['median'],
                             train_h_rms=r['train']['h_norm_rms'],head_centered=r['train']['centered_head_norm'],
                             alignment=r['train']['alignment_factor'],logit_rms=r['train']['centered_linear_logit_rms']))
    for arm in colors:
        rows=[r for r in d['rows'] if r['arm']==arm]
        ax[1,2].plot([r['task'] for r in rows],[r['train']['lr_l1_median'] for r in rows],'-o',ms=3,c=colors[arm],label=arm)
    titles=['最終FFNの入力の共通成分','注意1がCLSへ置く注意質量','注意1の書き込みのうちCLS由来',
            '最終FFNの死の割合','行と共通成分のcos中央値','head入力の尺度（訓練画像）']
    ys=['ノルム','注意の確率','符号付き射影の比','開口頻度 ≤ 0.005 の割合','cos','η × L1(h)']
    for a,title,y in zip(ax.flat,titles,ys):a.set(title=title,xlabel='課題末',ylabel=y);a.grid(alpha=.2);a.legend(fontsize=8)
    ax[0,0].set_yscale('log');ax[1,2].set_yscale('log');ax[1,2].axhline(1,color='gray',ls=':',lw=1)
    ax[0,1].axhline(1/65,color='gray',ls=':',lw=1)
    fig.suptitle('保存状態の測定：R0のCLS sink予測は外れた',fontsize=14)
    fig.savefig(RESULT/'codex_r0_writers.png');plt.close(fig)
    write_csv('codex_state_table.csv',flat)
    all_replay=[json.loads(p.read_text()) for p in OUT.glob('codex_replay_*.json')]
    longest={}
    for r in all_replay:
        key=(r['arm'],r['task_end'],r['mode'])
        if key not in longest or r['steps']>longest[key]['steps']:longest[key]=r
    replay=list(longest.values())
    fig,axs=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    modes={'base':('#666666','元の学習率'),'lr01':('#267a47','head lr × 0.1'),'lr001':('#4e60aa','head lr × 0.01'),'scale01':('#c48525','特徴 × 0.1')}
    conditions=[('R0',20),('R0',30),('R0',40),('R0',60),('S',30),('S',60)]
    flat=[]
    for ax,(arm,t) in zip(axs.flat,conditions):
        for r in replay:
            if r['arm']!=arm or r['task_end']!=t:continue
            col,label=modes[r['mode']]
            ax.plot([q['step'] for q in r['records']],[q['acc'] for q in r['records']],c=col,lw=1.4,label=label)
            for q in r['records']:
                flat.append(dict(arm=arm,task_end=t,mode=r['mode'],loaded_adam=r['loaded_adam'],**q))
        ax.axhline(.9,color='black',lw=.7,ls=':');ax.set_ylim(0,1.02)
        ax.set(title=f'{arm}: 課題 {t+1}'+('（空のAdam）' if t in [20,40] else ''),xlabel='更新数',ylabel='暗記率');ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8,loc='lower right')
    fig.suptitle('一課題の介入：各曲線は実際の終了更新まで',fontsize=14)
    fig.savefig(RESULT/'codex_r0_interventions.png');plt.close(fig)
    if flat:write_csv('codex_replay_curves.csv',flat)
    fig,axs=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    for i,t in enumerate([10,30]):
        r=next(r for r in replay if r['arm']=='R0' and r['task_end']==t and r['mode']=='base')
        g=r['geometry'];steps=[q['step'] for q in g]
        axs[i,0].plot(steps,[q['h_norm_median'] for q in g],'o-',label='CLS h')
        axs[i,0].plot(steps,[q['m2_norm'] for q in g],'s-',label='共通成分 m2')
        axs[i,0].set_yscale('log');axs[i,0].legend();axs[i,0].set_ylabel(f'R0 {t} → {t+1}: ノルム')
        axs[i,1].plot(steps,[q['dead'] for q in g],'o-',label='全token')
        axs[i,1].plot(steps,[q['cls_dead'] for q in g],'s-',label='CLS')
        axs[i,1].set_ylim(0,1);axs[i,1].set_ylabel('死の割合');axs[i,1].legend()
        ds=r['step_diagnostics'];ss=[q['step'] for q in ds]
        axs[i,2].plot(ss,[q['head_logit_rms'] for q in ds],'o-',label='headの一歩')
        axs[i,2].plot(ss,[q['body_logit_rms'] for q in ds],'s-',label='体の一歩')
        axs[i,2].set_yscale('log');axs[i,2].set_ylabel('中心化logit差のRMS');axs[i,2].legend()
        for a in axs[i]:
            a.set_xscale('symlog',linthresh=10);a.set_xlim(0,5000);a.set_xlabel('更新数');a.grid(alpha=.2)
    fig.suptitle('課題内の軌跡：入力の共通成分とhead特徴は同じ尺度ではない',fontsize=13)
    fig.savefig(RESULT/'codex_r0_trajectories.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for arm,col in [('R0',colors['R0']),('S',colors['S'])]:
        for t,ls in [(30,'-'),(60,'--')]:
            r=next(r for r in d['rows'] if r['arm']==arm and r['task']==t)
            for a,pop in zip(axs,['all','cls']):
                cs=sorted([q for q in r['counterfactual'] if q['population']==pop],key=lambda q:q['k'])
                a.plot([q['k'] for q in cs],[q['dead'] for q in cs],ls,marker='o',c=col,label=f'{arm} t{t}')
    for a,pop in zip(axs,['全token','CLS']):
        a.set(title=pop,xlabel='全token平均 m2 の倍率 k',ylabel='死の割合',ylim=(0,1));a.grid(alpha=.2);a.legend()
    fig.suptitle('静的反実仮想：重みと中心化した残差を固定',fontsize=13)
    fig.savefig(RESULT/'codex_r0_mean_counterfactual.png');plt.close(fig)
    print('4 figures and numeric CSV tables created')


if __name__=='__main__':main()

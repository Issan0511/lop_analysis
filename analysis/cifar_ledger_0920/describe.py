#!/usr/bin/env python3
"""Descriptive tables for pre-registered windows; no changed verdicts."""
import csv, json, os, sys
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='2'
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from analysis.cifar_ledger_0920 import replay as R
from analysis.cifar_ledger_0920.report import write_csv


def main():
    out=R.OUT
    with (out/'windows.csv').open() as f:windows=list(csv.DictReader(f))
    with (out/'per_task.csv').open() as f:tasks=list(csv.DictReader(f))
    summary=[]
    metrics=['delta','self','upstream','cross','bias','self_share','upstream_share','cross_share','bias_share','upstream_growth_fraction','upstream_rotation_fraction']
    for arm in R.ARMS:
        for cond in ('raw','std'):
            for layer in (1,2):
                for a,b in ((0,10),(0,50),(1,10),(10,50)):
                    rr=[r for r in windows if r['arm']==arm and r['cond']==cond and int(r['layer'])==layer and int(r['start'])==a and int(r['end'])==b and r['role']=='REPORT_ONLY']
                    row=dict(arm=arm,cond=cond,layer=layer,start=a,end=b,role='REGISTERED_REPORT_ONLY',n=len(rr))
                    assert len(rr)==10
                    for metric in metrics:
                        vals=[float(r[metric]) for r in rr if r[metric] and np.isfinite(float(r[metric]))]
                        row[metric+'_median']=float(np.median(vals)) if vals else None
                        row[metric+'_defined']=len(vals)
                    summary.append(row)
    layers=[]
    for arm in R.ARMS:
        for cond in ('raw','std'):
            for layer in (1,2):
                for t in (0,1,2,3,5,10,50):
                    rr=[r for r in tasks if r['arm']==arm and r['cond']==cond and int(r['layer'])==layer and int(r['task'])==t]
                    row=dict(arm=arm,cond=cond,layer=layer,task=t,role='REGISTERED_DESCRIPTIVE',n=len(rr))
                    assert len(rr)==10
                    for key in ('train_low_mean','train_zero_mean','m_mean','sd_mean','mu_norm_mean','w_norm_mean','bias_mean','online'):
                        vals=[float(r[key]) for r in rr if r[key] and np.isfinite(float(r[key]))]
                        row[key+'_seed_median']=float(np.median(vals)) if vals else None
                    layers.append(row)
    write_csv(out/'window_summary.csv',summary);write_csv(out/'layer_summary.csv',layers)
    # Scientific figure: means across the ten original seeds, no fitted curve.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,2,figsize=(11,8),sharex=True,sharey=True)
    for i,arm in enumerate(('ELU','GELU','SILU')):
        for j,cond in enumerate(('raw','std')):
            ax=axs[i,j]
            for l,color in ((1,'#1976d2'),(2,'#d95f02')):
                vals=np.array([[float(next(r for r in tasks if r['arm']==arm and r['cond']==cond and int(r['seed'])==s and int(r['task'])==t and int(r['layer'])==l)['train_low_mean']) for t in range(11)] for s in range(10)])
                ax.plot(range(11),vals.mean(0),color=color,label=f'Layer {l}')
                ax.fill_between(range(11),vals.min(0),vals.max(0),color=color,alpha=.13)
            online=np.array([[float(next(r for r in tasks if r['arm']==arm and r['cond']==cond and int(r['seed'])==s and int(r['task'])==t and int(r['layer'])==1)['online']) for t in range(1,11)] for s in range(10)])
            ax.plot(range(1,11),online.mean(0),'--',color='#444444',label='Online accuracy')
            ax.axhline(.5,color='black',lw=.7,alpha=.4)
            ax.set_title(f'{arm} / {cond}');ax.grid(alpha=.15);ax.set_ylim(-.02,1.02)
    axs[0,0].legend(fontsize=9)
    fig.supylabel('Low-response pair fraction / online accuracy',fontsize=11)
    for ax in axs[-1]:ax.set_xlabel('Task end (0 = initialization)')
    fig.suptitle('CIFAR A6 | |training derivative| < 1e-6\nLines: seed means; shading: seed range (10 seeds)',fontsize=13)
    fig.tight_layout(rect=(.025,0,1,.94));fig.savefig(out/'low_response.png',dpi=180);plt.close(fig)
    section=['','## 登録した補助窓（REPORT_ONLY）','','第2層、std、seed内の比の中央値。主判定の置換には使わない。','',
             '| arm | 窓 | 上流 | 自己 | 交差 | bias | 上流中の伸び |','|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['arm'] in ('ELU','GELU','SILU') and r['cond']=='std' and r['layer']==2 and (r['start'],r['end']) in ((0,10),(1,10),(10,50)):
            vals=[r[k+'_median'] for k in ('upstream_share','self_share','cross_share','bias_share','upstream_growth_fraction')]
            section.append(f'| {r["arm"]} | t{r["start"]:02d}→t{r["end"]:02d} | '+' | '.join('NA' if v is None else f'{100*v:.3f}%' for v in vals)+' |')
    section+=['','## 読み（主判定後の整理）','',
              '- ELU/GELU/SiLUはstdで第2層、rawで第1層が先に低応答化し、各セル10/10。S4のELU/std・第2層という候補を支持する。',
              '- ELU/stdの主窓では交差項が最大で、上流単独の過半という予測は不支持（MIXED 10/10）。交差項は同じタスク内にWとmuの両方が変わった分であり、単一の原因名ではない。',
              '- ELU/stdは固定t00→t10では上流約73%、t10→t50では約99%。後半の輸送と初期の形成を同じ割合で語れない。主判定をUPSTREAM_CARRIESへ変更しない。',
              '- ELU/stdの主窓で上流項の約90.7%がmuの伸び、bias絶対寄与の中央値は約0.024%。両予測は10/10で条件を満たした。',
              '- ReLUの初期低応答・同時通過は、このQ>0.5規則では新たな故障層の先後を決められないという結果。別の閾値で読み直して登録結果を置換しない。',
              '- Qの初回超過とonline<0.5は一致しない。例えばELU/stdの中央値は前者t2、後者t3。低応答先行層を機能的LoPそのものと呼ばない。',
              '- S5のcap1/cap2を一方へ絞る因果根拠はA6だけでは得られない。帳簿の交差と窓依存を設計側へ渡す。',
              '', '![Low response and online accuracy](low_response.png)']
    p=out/'summary.md';base=p.read_text().split('\n## 登録した補助窓（REPORT_ONLY）')[0];p.write_text(base+'\n'.join(section)+'\n')
    R.put(out/'description_provenance.json',dict(**R.E.git_state(),source_sha256=R.sha(Path(__file__)),classification_changed=False,window_tier='registered REPORT_ONLY',interpretation_tier='post-result narrative',figure='seed means and min/max, no fitted curve'))
    print('Wrote layer_summary.csv, window_summary.csv, low_response.png, summary interpretation.')


if __name__=='__main__':main()

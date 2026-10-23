#!/usr/bin/env python3
"""Registered seed/window summaries and restoration decisions.

J6（spec_chargpt_detlabel_1023 §6.5）の集計。写し元:
  analysis/chargpt_loadsign_1023/report.py  git blob e1f902fe81d8c936f3ee853dc06c19b08cd603d7（commit 566ce479、099530e9 でも同じ blob）
集計（窓・中央値・loadsign §4 の回復の数え）は変えない。変えた所: 置き場と腕を arms.py（J6_ARM）から取る（変更 1・4）、
summary の照合の行から前回 unit 配列（latepush）との差を外した（変更 3）。judge.json は loadsign §4 の回復の数えで、J6 の判定ではない。
"""
import csv
import json
from pathlib import Path
import numpy as np
import arms as ARMS_MOD

ROOT=ARMS_MOD.ROOT
LABEL,_ARM,OUT=ARMS_MOD.current()
WINDOWS={'early':range(1,10),'middle':range(39,51),'late':range(89,101)}
ARMS=[LABEL]
BASE='s_original_exact_K'


def main():
    rows=list(csv.DictReader((OUT/'states.csv').open()))
    fields=[k for k in rows[0] if k not in ['arm','seed','task','layer','autograd_error']]
    wins=[]
    for arm in ARMS:
        for seed in range(10):
            for win, tasks in WINDOWS.items():
                rr=[r for r in rows if r['arm']==arm and int(r['seed'])==seed and int(r['task']) in tasks]
                assert len(rr)==len(tasks)
                wins.append(dict(arm=arm,seed=seed,window=win,**{k:float(np.median([float(r[k]) for r in rr])) for k in fields}))
    with (OUT/'windows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(wins[0]),lineterminator='\n');w.writeheader();w.writerows(wins)
    index={(w['arm'],w['seed'],w['window']):w for w in wins}
    variants=[f's_{e}_{l}_{k}' for e in ['original','loaderror'] for l in ['exact','direct'] for k in ['K','absK']]
    judge={}
    for arm in ARMS:
        jj={}
        for v in variants:
            seeds=[]
            for seed in range(10):
                early=index[arm,seed,'early']; late=index[arm,seed,'late']
                delta=late[v]-late[BASE]
                seeds.append(dict(seed=seed,early=early[v],late=late[v],late_minus_baseline=delta,
                                  drop_attenuation=(early[BASE]-late[BASE])-(early[v]-late[v]),
                                  restored=bool(late[v]>=.75 and delta>=.20)))
            count=sum(s['restored'] for s in seeds)
            jj[v]=dict(n_restored=count,restored=count>=8,seeds=seeds)
        judge[arm]=jj
    (OUT/'judge.json').write_text(json.dumps(judge,indent=2,ensure_ascii=False)+'\n')
    def rng(arm,win,k):
        x=[index[arm,s,win][k] for s in range(10)]
        return f'{min(x):.4f}–{max(x):.4f}'
    text=['# 装填成分の押しが後期に半々になる経路','',
          f'保存状態のみ。L1、10 seed × 35 課題末、腕 {LABEL}（模型 {_ARM["model_arm"]}、データの α {_ARM["alpha"]}）= 350 状態。各値は seed 内の窓中央値の seed 間の範囲。',
          '判定: 後期 s≥0.75 かつ baseline より≥0.20 増加が 8/10 seed。',
          'K = 一様勾配 probe の y と現課題の ybar の内積。direct = 最終 LN の分散微分だけを除いた pullback。loaderror = 装填 logit だけで softmax 誤差を作る。','']
    for arm in ARMS:
        text += [f'## {arm} L1','', '| 診断 | 早期 | 中期 | 後期 | 回復 seed |','|---|---|---|---|---|']
        for k in variants:
            text.append(f'| {k} | {rng(arm,"early",k)} | {rng(arm,"middle",k)} | {rng(arm,"late",k)} | {judge[arm][k]["n_restored"]}/10 |')
        text+=['','| 記録 | 早期 | 中期 | 後期 |','|---|---|---|---|']
        for k in ['s_radial','s_constantK','s_nonzero','n_alive','n_zero','K_task_positive','K_uniform_positive',
                  'cancellation','direct_positive_to_nonpositive','direct_nonpositive_to_positive','projection_cos2_median',
                  'ffn_over_residual_mean','ffn_residual_cos_mean','error_cos_mean','error_loadlogit_dot_mean',
                  'error_loadlogit_dot_positive','entropy_mean','loaderror_entropy_mean','v_over_W','v_norm','vL_norm',
                  'abs_original_exact_K','abs_original_direct_K']:
            text.append(f'| {k} | {rng(arm,"early",k)} | {rng(arm,"middle",k)} | {rng(arm,"late",k)} |')
        text+=['']
    gate=json.loads((OUT/'gate.json').read_text())
    text += ['## 照合','',f'事前 gate 8 状態の自動微分との差: {gate["max_autograd_error"]:.3g}。',
             f'恒等分解の最大誤差: {max(float(r["additivity_error"]) for r in rows):.3g}。','',
             'この診断は固定状態の局所経路を比較する。実際の有限更新や学習史の原因を同定する実験ではない。']
    (OUT/'summary.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({a:{k:v['n_restored'] for k,v in j.items()} for a,j in judge.items()},ensure_ascii=False))


if __name__=='__main__':main()

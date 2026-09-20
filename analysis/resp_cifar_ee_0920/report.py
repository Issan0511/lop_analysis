#!/usr/bin/env python3
"""S4 frozen readout. Explicit source; refuses missing/partial/nonfinite shards."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from src.resp_cifar_ee_0920 import RUN, ARMS, write_csv
from src.cifar_interventions_0920 import ROOT, put, verify_done, sha, load_pt
from analysis.resp_cifar_ee_0920.stats import interval, label


def verdict(values,initial_g):
    assert set(values)==set(ARMS),'missing or extra arm'
    for name,v in values.items():assert len(v)==10 and np.isfinite(v).all(),('incomplete/nonfinite',name)
    assert set(initial_g)=={1,10} and all(len(v)==10 and np.isfinite(v).all() for v in initial_g.values())
    x={k:np.asarray(v,dtype=np.float64) for k,v in values.items()}
    gap=interval(x['N1']-x['N10']);reset_gap=interval(x['N1r']-x['N10r'])
    p1=interval(x['R1_10']-x['N10'],.975);p2=interval(x['N1r']-x['S1_10r'],.975)
    applicable=gap['low']>0 and bool(np.all(np.asarray(initial_g[1])>np.asarray(initial_g[10])))
    result=dict(label=label(p1,p2) if applicable else 'NOT_REPRODUCED',applicable=applicable,
                natural_gap=gap,reset_gap=reset_gap,P1=p1,P2=p2,
                rho1=p1['mean']/gap['mean'] if gap['low']>0 else None,
                rho2=p2['mean']/reset_gap['mean'] if reset_gap['low']>0 else None)
    ladder=[float(x[n].mean()) for n in ('N1r','S1u5r','S1u10r','S1u20r')]
    result['ladder']='FLAT_SAMPLE_MEANS' if len(set(ladder))==1 else 'MONOTONE_SAMPLE_MEANS' if all(a>=b for a,b in zip(ladder,ladder[1:])) else 'NONMONOTONE_SAMPLE_MEANS'
    result['layer_gap']=interval(x['S1_L1_10r']-x['S1_10r'])
    return result


def report(src):
    src=Path(src)
    st=json.loads((src/'status.json').read_text());assert st['stage']=='completed','partial report'
    ident=json.loads((src/'provenance_start.json').read_text())['identity']
    assert ident['seeds']==list(range(10)) and ident['epochs']==400
    verify_done(src/'prefix/done.json',ident)
    pc=json.loads((src/'prefix_checks.json').read_text());assert set(pc)=={'N1','N10'} and all(p['all_pass'] for p in pc.values())
    values={};allrows=[];paired=[];secondary=[];armtable=[];initial_g={}
    for name in ARMS:
        arm=src/'arms'/name;verify_done(arm/'done.json',ident)
        rows=json.loads((arm/'rows.json').read_text());ini=json.loads((arm/'initial.json').read_text())
        assert len(rows)==10 and sorted(r['seed'] for r in rows)==list(range(10))
        rows.sort(key=lambda r:r['seed']);ini.sort(key=lambda r:r['seed'])
        assert all(r['arm']==name and r['finite'] and r['task']==ARMS[name][0]+1 for r in rows)
        for r,i in zip(rows,ini):
            r={**r,'excess_over_floor':(r['online_acc']-r['major_frac'])/(1-r['major_frac']),**{f'initial_{k}':v for k,v in i.items() if k not in ('seed','task','update')}}
            allrows.append(r)
        values[name]=[r['online_acc'] for r in rows]
        armtable.append(dict(arm=name,online_mean=np.mean(values[name]),online_sd=np.std(values[name],ddof=1),
                             online_min=min(values[name]),online_max=max(values[name]),memo_mean=np.mean([r['memo_acc'] for r in rows]),
                             initial_G2=np.mean([r['G2'] for r in ini]),final_G2=np.mean([r['G2'] for r in rows]),
                             final_Q2=np.mean([r['Q2'] for r in rows]),final_neff2=np.mean([r['neff2'] for r in rows])))
        if name in ('N1','N10'):initial_g[ARMS[name][0]]=[r['G2'] for r in ini]
    result=verdict(values,initial_g)
    comparisons=[('P1','R1_10','N10',.975),('P2','N1r','S1_10r',.975),
                 ('P1_95','R1_10','N10',.95),('P2_95','N1r','S1_10r',.95),
                 ('restore_reset','R1_10r','N10r',.95),('sink_keep','N1','S1_10',.95),
                 ('sink_L1','N1r','S1_L1_10r',.95),('layer_difference','S1_L1_10r','S1_10r',.95)]
    for metric,a,b,level in comparisons:
        d=np.asarray(values[a])-np.asarray(values[b]);row=dict(metric=metric,arm_a=a,arm_b=b,**interval(d,level),tier='PRIMARY' if metric in ('P1','P2') else 'REPORT_ONLY')
        secondary.append(row)
        for s,diff in enumerate(d):paired.append(dict(metric=metric,seed=s,a=values[a][s],b=values[b][s],difference=diff))
    result['independent_audit']=False;result['n_seeds']=10;result['registration']='273a6bc';result['approval']='5705359'
    probs={'RESPONSE_BOTH_WAYS':.60,'SINK_ONLY':.20,'RESTORE_ONLY':.10,'RESPONSE_NOT_SHOWN':.08,'RESPONSE_REVERSED':.02}
    predictions=[dict(metric='applicability',prediction=True,codex_probability=.90,hit=result['applicable'],binary_brier=(.90-float(result['applicable']))**2,issa_adopted_content=True)]
    facts=[('main_label','RESPONSE_BOTH_WAYS',.60,result['label']=='RESPONSE_BOTH_WAYS'),
           ('rho1_ge_half',True,.60,result['rho1']>=.5 if result['rho1'] is not None else None),
           ('L2_sink_exceeds_L1',True,.75,result['layer_gap']['mean']>0),
           ('monotone_means','MONOTONE_SAMPLE_MEANS',.70,result['ladder']=='MONOTONE_SAMPLE_MEANS')]
    for metric,pred,p,hit in facts:
        score=(p-float(hit))**2 if hit is not None and result['applicable'] else None
        predictions.append(dict(metric=metric,prediction=pred,codex_probability=p,hit=hit if result['applicable'] else None,binary_brier=score,issa_adopted_content=True))
    result['label_multiclass_brier']=sum((p-float(k==result['label']))**2 for k,p in probs.items()) if result['applicable'] else None
    training=[]
    for name in ARMS:
        first=load_pt(src/'arms'/name/'first_epoch_g.pt');assert len(first)==75
        for r in range(10):
            for l in range(2):
                train=np.concatenate([q['g'][l][r].numpy() for q in first]).astype(np.float64)
                full=np.concatenate([q['g_full_same_state'][l][r].numpy() for q in first]).astype(np.float64)
                training.append(dict(arm=name,seed=r,layer=l+1,actual_batch_G=train.mean(),full_same_state_G=full.mean(),max_abs_difference=abs(train-full).max(),mean_abs_difference=abs(train-full).mean(),actual_Q=(abs(train)<1e-6).mean(),full_Q=(abs(full)<1e-6).mean(),note='Both at each pre-update state; includes learning during epoch.'))
    write_csv(src/'first_epoch_comparison.csv',training)
    import csv
    old=list(csv.DictReader((ROOT/'results/rlcifar_mlp_battle_0918/ELU/per_task.csv').open()))
    old={(int(r['seed']),int(r['task'])):r for r in old if r['cond']=='std'}
    write_csv(src/'old_R20_comparison.csv',[dict(seed=r['seed'],task=r['task'],R10_online=r['online_acc'],R20_online=float(old[r['seed'],r['task']]['online_acc']),difference=r['online_acc']-float(old[r['seed'],r['task']]['online_acc'])) for r in json.loads((src/'prefix/rows.json').read_text())])
    write_csv(src/'per_seed.csv',allrows);write_csv(src/'paired.csv',paired);write_csv(src/'secondary.csv',secondary)
    write_csv(src/'arm_table.csv',armtable);write_csv(src/'predictions.csv',predictions)
    write_csv(src/'verdict.csv',[dict(label=result['label'],P1=result['P1']['mean'],P1_low=result['P1']['low'],P1_high=result['P1']['high'],P2=result['P2']['mean'],P2_low=result['P2']['low'],P2_high=result['P2']['high'],rho1=result['rho1'],rho2=result['rho2'])])
    put(src/'verdict.json',result)
    checks=json.loads((src/'admission_checks.json').read_text())
    assert checks['all_pass'] and checks['source_sha256']==ident['source_sha256']
    assert all(json.loads((src/'arms'/n/'preflight.json').read_text())['all_pass'] for n in ARMS)
    write_csv(src/'per_task.csv',[*json.loads((src/'prefix/rows.json').read_text()),*allrows])
    write_csv(src/'branches.csv',[dict(arm=n,branch=t[0],layer=t[1],source=t[2],reset=t[3]) for n,t in ARMS.items()])
    put(src/'checks.json',dict(pre_run=checks,prefix=pc,preflight={n:json.loads((src/'arms'/n/'preflight.json').read_text()) for n in ARMS},all_pass=True,independent_audit=False))
    text=['# CIFAR S4: 第2層の応答の場の移植','',f"登録主判定: **{result['label']}**。10 seed、ELU/std、R=10、分岐t1/t10、継続1task×30,000更新。",'',
          '既知自然軌道を参照して設計した新規介入。独立監査なし。','',
          '| 主比較 | 平均対応差 | 97.5%下端 | 上端 |','|---|---:|---:|---:|']
    for metric in ('P1','P2'):
        q=result[metric];text.append(f"| {metric} | {q['mean']:.6f} | {q['low']:.6f} | {q['high']:.6f} |")
    text += ['',f"自然差の95%下端: {result['natural_gap']['low']:.6f}。現象適用条件: {result['applicable']}。",f"回復率ρ1={result['rho1']}、ρ2={result['rho2']}（報告のみ）。",'',
             '| 腕 | online平均 | seed範囲 | memo平均 | 初期G2 | 終端G2 |','|---|---:|---|---:|---:|---:|']
    for r in armtable:text.append(f"| {r['arm']} | {r['online_mean']:.6f} | {r['online_min']:.6f}–{r['online_max']:.6f} | {r['memo_mean']:.6f} | {r['initial_G2']:.5g} | {r['final_G2']:.5g} |")
    text += ['',f"一様階段: {result['ladder']}（標本平均の記述のみ）。",'',
             '最初の1 epochの実訓練B=16と、各更新前の同一状態のB=1200で求めた微分を first_epoch_comparison.csv に比較した。旧R20との機能差は old_R20_comparison.csv に記録し、合否に用いない。',
             'P1はAdam履歴を継続、P2は両腕reset。両主比較はそれぞれ同じ次課題内の対応差で、P1とP2は異なる課題。',
             '初期出力は自然腕とbit一致。固定場の人工的関数変更であり、学習が進めば特徴も変わる。自然なLoPの完全な媒介や恒久的救済は示さない。',
             '非検出側を効果ゼロ・同等性とは読まない。層対照・階段・n_effの関係はREPORT_ONLY。',
             '全必須検査と変異、自然継続の状態・乱数の一致を確認。科学的定義・窓・seedは結果後に変更していない。']
    (src/'summary.md').write_text('\n'.join(text)+'\n')
    import subprocess
    put(src/'report_provenance.json',dict(git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sources={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/f'analysis/{RUN}').glob('*.py'))},input_identity=ident,independent_audit=False))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--src',required=True);a=p.parse_args();report(Path(a.src))

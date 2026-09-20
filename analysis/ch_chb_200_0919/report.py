#!/usr/bin/env python3
"""Frozen endpoint arithmetic. Read only once both arms finish."""
import argparse, collections, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src import ch_chb_200_0919 as S
import json

def classify(B,W,Z,dead0,deadlate,gate0,gatelate,k,reg):
    h,dc,L=reg['h'],reg['d_c'],reg['L_H'];D=B-W
    eroding=deadlate>20*dead0 or gatelate>20*gate0
    reasons=['RESPONSE_ERODING'] if eroding else []
    numerical=any(math.isnan(v) for v in (B,W)) or not all(math.isfinite(v) for v in (B,W))
    typ='UNCLASSIFIED';omega=False
    gminus=.0105*max(k);gplus=.0105*sum(k)
    if numerical:typ='NUMERICAL_FAILURE'
    elif W<.5 or Z<dc:
        typ='COLLAPSE';reasons.append('BOTH' if W<.5 and Z<dc else 'PERFORMANCE_ONLY' if W<.5 else 'DEPTH_ONLY')
    elif math.isnan(Z):reasons.append('DEPTH_UNDEFINED')
    elif W>=L:typ='HOLDS'
    else:
        if not all(math.isfinite(x) for x in k):reasons.append('OMEGA_UNDEFINED')
        elif D>gplus+h:reasons.append('EXCESS_DROP')
        elif not eroding:
            typ='GENTLE'
            if D<=0:reasons.append('BASELINE_OFFSET')
            omega=D>h and max(k)>0 and gminus-h<=D<=gplus+h
    return dict(type=typ,reasons=reasons,omega_compatible=bool(omega),B=B,W=W,D=D,Z=Z,k=k,gminus=gminus,gplus=gplus,
        dead_t50=dead0,dead_last20_sum=deadlate,gate_t50=gate0,gate_last20_sum=gatelate)

def pair(a,b):
    safe={'HOLDS','GENTLE'}
    if a in safe and b in safe:return 'CURED_200'
    if a=='COLLAPSE' and b in safe:return 'B_ROUTE_DELAY'
    if a=='COLLAPSE' and b=='COLLAPSE':return 'OTHER_ROUTE'
    if a in safe and b=='COLLAPSE':return 'REVERSED_B_EFFECT'
    return 'INCONCLUSIVE'

def combine(byarm):
    assert set(byarm)=={'CH','CHB'} and all(set(a)==set(range(10)) for a in byarm.values())
    pairs={s:pair(byarm['CH'][s]['type'],byarm['CHB'][s]['type']) for s in range(10)}
    support=collections.Counter(pairs.values())
    main=next((k for k,n in support.items() if n>=9),'INCONCLUSIVE')
    arms={}
    for a,rr in byarm.items():
        counts=collections.Counter(r['type'] for r in rr.values())
        arms[a]=next((k for k in ('HOLDS','GENTLE','COLLAPSE') if counts[k]>=9),
                     'HOLDS_OR_GENTLE' if counts['HOLDS']+counts['GENTLE']>=9 else 'SPLIT')
    J=[a for a,rr in byarm.items() if any(r['type']=='GENTLE' for r in rr.values())]
    omega=main=='CURED_200' and bool(J) and all(sum(r['omega_compatible'] for r in byarm[a].values())>=9 for a in J)
    return dict(main=main,omega_only=omega,arms=arms,pairs=pairs,support=dict(support))

def seed_readout(rows,reg):
    assert len(rows)==200 and sorted(int(r['task']) for r in rows)==list(range(1,201))
    rows=sorted(rows,key=lambda r:int(r['task']))
    base=rows[30:50];late=rows[180:200]
    def mean(rs,k):return float(np.mean([float(r[k]) for r in rs]))
    k=[]
    for l in (1,2):
        b=[float(r[f'omega_l{l}']) for r in base];w=[float(r[f'omega_l{l}']) for r in late]
        k.append(max(0.,float(np.mean(np.log2(b))-np.mean(np.log2(w)))) if all(math.isfinite(x) and x>0 for x in b+w) else math.nan)
    rr=classify(mean(base,'online_acc'),mean(late,'online_acc'),mean(late,'depth_ratio_l2'),
        int(rows[49]['n_dead_l2']),sum(int(r['n_dead_l2']) for r in late),int(rows[49]['n_gate0_l2']),sum(int(r['n_gate0_l2']) for r in late),k,reg)
    rr['recovered_after_crossing']=rr['type'] in ('HOLDS','GENTLE','UNCLASSIFIED') and any(float(r['online_acc'])<.5 or float(r['depth_ratio_l2'])<reg['d_c'] for r in rows[50:180])
    rr['windows']=[dict(first=t,last=t+19,online=mean(rows[t-1:t+19],'online_acc'),depth=mean(rows[t-1:t+19],'depth_ratio_l2')) for t in range(51,182,10)]
    rr['diagnostic_last20']={key:mean(late,key) for key in ('dead_frac_l2','gate_zero_frac_l2','bias_over_sd_l2','bias_mean_l2','zsd_l2','w_norm_l1','w_norm_l2','r_a1','omega_l1','omega_l2')}
    return rr

def stats(xs):
    xs=np.asarray(xs);mu=float(xs.mean());half=2.2621571628540993*float(xs.std(ddof=1))/math.sqrt(10)
    return dict(mean=mu,median=float(np.median(xs)),min=float(xs.min()),max=float(xs.max()),t95=[mu-half,mu+half])

def report(src):
    assert src.resolve()==S.OUT.resolve(),'explicit new run output only'
    reg=json.loads((src/'registration.json').read_text());assert reg==S.registration()
    checks=json.loads((src/'checks/checks.json').read_text());assert checks['all_pass']
    byarm={}
    for arm in S.LAM:
        path=src/arm
        st=S.torch.load(path/'ckpt.pt',map_location='cpu',weights_only=False)
        S.validate_checkpoint(st,arm,list(range(10)),400,200);S.validate_history(path,arm,list(range(10)),200)
        rows=S.readcsv(path/'per_task.csv');assert len(rows)==2000
        old=S.readcsv(src/'inputs'/arm/'per_task.csv');lookup={(int(r['seed']),int(r['task'])):r for r in rows}
        assert all(all(lookup[(int(r['seed']),int(r['task']))][k]==v for k,v in r.items()) for r in old)
        prov=json.loads((path/'provenance.json').read_text());assert prov['run_id']==S.RUN and prov['completed_task']==200
        con=json.loads((path/'continuity.json').read_text());assert con['expected_state']==con['before_first_draw']
        assert con['labels_expected']==con['labels_actual'] and con['order_expected']==con['order_actual']
        byarm[arm]={s:seed_readout([r for r in rows if int(r['seed'])==s],reg) for s in range(10)}
    result=combine(byarm);result.update(seed_readouts=byarm,registration=reg,independent_audit=False)
    result['statistics']={a:{k:stats([r[k] for r in rr.values()]) for k in ('W','D')} for a,rr in byarm.items()}
    result['paired_D_CH_minus_CHB']=stats([byarm['CH'][s]['D']-byarm['CHB'][s]['D'] for s in range(10)])
    branch=0 if result['main']=='CURED_200' and result['omega_only'] else 1 if result['main']=='CURED_200' else 2 if result['main']=='B_ROUTE_DELAY' else 3 if result['main']=='OTHER_ROUTE' else 4
    probs=[.45,.10,.27,.13,.05]
    dm={a:result['statistics'][a]['D']['mean'] for a in S.LAM}
    result['predictions']=dict(codex_brier=sum((p-int(i==branch))**2 for i,p in enumerate(probs)),codex_interval=0<=dm['CH']<=.02,codex_order=dm['CHB']<=dm['CH'],issa_main=branch==0,issa_types=result['arms']['CH']=='GENTLE' and result['arms']['CHB'] in ('HOLDS','GENTLE','HOLDS_OR_GENTLE'),issa_interval=0<=dm['CH']<=.02,issa_order=dm['CHB']<=dm['CH'])
    S.put(src/'verdict.json',result)
    tab=[]
    for s in range(10):
        for a in S.LAM:
            r=byarm[a][s];tab.append(dict(seed=s,arm=a,type=r['type'],reasons='|'.join(r['reasons']),omega_compatible=r['omega_compatible'],B=r['B'],W=r['W'],D=r['D'],Z=r['Z'],pair=result['pairs'][s]))
    S.writecsv(src/'verdict.csv',tab)
    lines=['# ch_chb_200_0919 — 登録結果','', '**独立監査なし。自己検査・変異対照のみ。**','',f"主判定: **{result['main']}**"+(' + **OMEGA_ONLY**' if result['omega_only'] else ''), '', '同一 seed の t50 checkpoint から t200（6M 更新）への継続。有限予算のラベルであり永続的治癒ではない。深さだけの COLLAPSE は機能的 LoP の証明ではない。','', '|腕|型|最終 online（seed平均）|低下（pt）|','|---|---|---:|---:|']
    for a in S.LAM:lines.append(f"|{a}|{result['arms'][a]}|{result['statistics'][a]['W']['mean']:.8f}|{100*dm[a]:.5f}|")
    lines+=['','全 seed の型・理由・境界値は verdict.csv、窓別診断・対応差の t 区間は verdict.json。少数 seed も除外していない。','', '|seed|CH|CHB|組|','|---|---|---|---|']
    for s in range(10):lines.append(f"|{s}|{byarm['CH'][s]['type']} ({','.join(byarm['CH'][s]['reasons'])})|{byarm['CHB'][s]['type']} ({','.join(byarm['CHB'][s]['reasons'])})|{result['pairs'][s]}|")
    lines+=['','予測採点: `'+json.dumps(result['predictions'],ensure_ascii=False)+'`','', '登録閾値・窓・応答個数条件は実行後に変更していない。bias WD は完全除去でなく、OMEGA_ONLY も機構同定ではない。']
    (src/'summary.md').write_text('\n'.join(lines)+'\n')
    try:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(2,1,figsize=(9,7),sharex=True)
        for a in S.LAM:
            for s,r in byarm[a].items():
                x=[w['last'] for w in r['windows']]
                axs[0].plot(x,[w['online'] for w in r['windows']],alpha=.4,label=a if s==0 else None)
                axs[1].plot(x,[w['depth'] for w in r['windows']],alpha=.4)
        axs[0].axhline(reg['L_H'],color='black',linestyle='--');axs[0].axhline(.5,color='red',linestyle=':');axs[0].legend();axs[0].set_ylabel('20-task online accuracy')
        axs[1].axhline(reg['d_c'],color='red',linestyle=':');axs[1].set_ylabel('20-task mean depth ratio');axs[1].set_xlabel('Window final task')
        fig.tight_layout();fig.savefig(src/'registered_windows.png',dpi=160);plt.close(fig)
    except ImportError:pass
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--src',type=Path,required=True);a=ap.parse_args();report(a.src)

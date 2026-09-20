#!/usr/bin/env python3
"""Complete-run report; refuses partial inputs and preserves undefined quantities."""
import argparse, csv, json, os, sys
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='2'
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from analysis.cifar_ledger_0920 import ledger as L
from analysis.cifar_ledger_0920 import replay as R


def write_csv(p,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for row in rows:w.writerow(R.clean(row))


def pair_fraction(per_unit,n_images=1200):
    # A mean of rounded fractions can be 0.5000000000000001 at exactly half.
    counts=np.rint(np.asarray(per_unit)*n_images)
    assert np.all(abs(np.asarray(per_unit)*n_images-counts)<=L.gamma(4)*n_images)
    return counts.sum(-1)/(n_images*counts.shape[-1])


def summarize_window(shards,slot,layer,a,b):
    ww=L.window(shards,slot,layer,a,b);d=ww['delta'];out=dict(start=a,end=b,n_intervals=b-a,delta=float(d.v),delta_error=float(d.e),delta_median=ww['delta_median'])
    for k in L.TERMS+('upstream_growth','upstream_rotation','self_growth','self_rotation'):
        ball=ww[k]
        out[k]=float(ball.v);out[k+'_error']=float(ball.e);out[k+'_rate']=float(ball.v)/(b-a)
        out[k+'_median']=ww[k+'_median']
        out[k+'_share']=float((ball/d).v) if abs(d.v)>d.e else None
    total=ww['self']+ww['upstream']+ww['cross']+ww['bias']
    assert abs(total.v-d.v)<=total.e+d.e,'window closure'
    out['closure_residual']=float(total.v-d.v)
    for base in ('upstream','self'):
        den=ww[base]
        for kind in ('growth','rotation'):
            num=ww[f'{base}_{kind}']
            defined=np.isfinite(num.v) and np.isfinite(num.e) and abs(den.v)>den.e
            out[f'{base}_{kind}_fraction']=float((num/den).v) if defined else None
    out['bias_absolute_share']=float(abs(ww['bias'].v/d.v)) if abs(d.v)>d.e else None
    return out,ww


def report(out):
    out=Path(out)
    status=json.loads((out/'status.json').read_text());assert status['status']=='completed' and status['completed_states']==306,'refuse partial report'
    checks=json.loads((out/'checks.json').read_text());assert checks['all_pass']
    gs=R.E.git_state();assert not gs['dirty_src_analysis'],'commit report before reading outcomes'
    R.put(out/'report_provenance.json',dict(**gs,spec_sha256=R.sha(R.SPEC),replay_git_hash=json.loads((out/'provenance.json').read_text())['git_hash'],source_sha256={str(p.relative_to(R.ROOT)):R.sha(p) for p in Path(__file__).parent.glob('*.py')},independent_audit=False))
    seed_rows=[];windows=[];task_rows=[];units={};ledgers={}
    for arm in R.ARMS:
        shards=[]
        for t in range(51):
            p=R.output_artifact(out,Path('shards')/f'{arm}_t{t:02d}.npz');marker=json.loads(p.with_suffix('.json').read_text())
            assert R.sha(p)==marker['sha256']
            shards.append(R.read_npz(p))
        for k in shards[0]:units[arm+'__'+k]=np.stack([s[k] for s in shards])
        for k in shards[1]:
            if k.startswith('ledger_'):ledgers[arm+'__'+k]=np.stack([s[k] for s in shards[1:]])
        for slot,(seed,cond) in enumerate(R.SLOTS):
            q=np.stack([pair_fraction(s['train_low'][slot]) for s in shards])
            online=np.array([s['online'][slot] for s in shards])
            m=np.stack([s['m'][slot] for s in shards]);sd=np.stack([s['sd'][slot] for s in shards])
            me=np.stack([s['m_error'][slot] for s in shards]);se=np.stack([s['sd_error'][slot] for s in shards])
            event=L.first_event(q,online,m,sd,me,se)
            base=dict(arm=arm,cond=cond,seed=seed,R1=event['label'],**{k:v for k,v in event.items() if k not in ('label','initial_low_layers')},initial_low_layers=';'.join(map(str,event['initial_low_layers'])),Q0_l1=q[0,0],Q0_l2=q[0,1])
            for l in range(2):
                hits=np.flatnonzero(q[11:,l]>.5)
                base[f'first_late_low_l{l+1}']=int(hits[0]+11) if len(hits) else None
                base[f'Q10_l{l+1}']=float(q[10,l])
            base['R2_context']='L2_LEDGER_WHEN_L1_FIRST' if event['layer']=='L1_FIRST' else 'L2_LEDGER'
            if arm in ('KKT1','LR'):
                base['R1']='REPORT_ONLY';base['T_star']=None;event['T_star']=None
            for layer in range(2):
                for a,b in [(0,10),(0,50),(1,10),(10,50)]:
                    ww,_=summarize_window(shards,slot,layer,a,b)
                    windows.append(dict(arm=arm,cond=cond,seed=seed,layer=layer+1,role='REPORT_ONLY',**ww))
            t=event['T_star']
            if t is not None:
                for layer in range(2):
                    ww,balls=summarize_window(shards,slot,layer,0,t)
                    windows.append(dict(arm=arm,cond=cond,seed=seed,layer=layer+1,role='PRIMARY',**ww))
                    if layer==1:
                        label,reason=L.carrier(balls,balls['delta'],event['label'])
                        base.update(R2=label,R2_reason=reason,**ww)
            else:base.update(R2='REPORT_ONLY' if arm in ('KKT1','LR') else 'NO_EVENT',R2_reason='')
            seed_rows.append(base)
            for t,sh in enumerate(shards):
                for l in range(2):
                    row=dict(arm=arm,cond=cond,seed=seed,task=t,layer=l+1,online=online[t])
                    for k in ('m','sd','m_alg','w_norm','bias','mu_norm','cos','train_low','train_zero','train_negative','train_signed_low','train_mean','train_abs_mean','train_dead','diag_low','diag_zero','diag_dead','derivative_max_difference','derivative_zero_disagreement'):
                        row[k+'_mean']=float(sh[k][slot,l].mean());row[k+'_median']=float(np.median(sh[k][slot,l]))
                    task_rows.append(row)
    verdict=[]
    for arm in R.ARMS:
        for cond in ('raw','std'):
            rr=[r for r in seed_rows if r['arm']==arm and r['cond']==cond]
            r1='REPORT_ONLY' if arm in ('KKT1','LR') else L.majority([r['R1'] for r in rr])
            r2='REPORT_ONLY' if arm in ('KKT1','LR') else L.majority([r['R2'] for r in rr])
            row=dict(arm=arm,cond=cond,n=10,R1=r1,R2=r2)
            for label in sorted(set(r['R1'] for r in rr)):row['R1_count_'+label]=sum(r['R1']==label for r in rr)
            for label in sorted(set(r['R2'] for r in rr)):row['R2_count_'+label]=sum(r['R2']==label for r in rr)
            for k in ('T_star','T1','T2','T_A','upstream_share','self_share','cross_share','bias_share','upstream_growth_fraction','bias_absolute_share'):
                vv=[r[k] for r in rr if r.get(k) is not None and np.isfinite(r[k])]
                row[k+'_defined']=len(vv);row[k+'_median']=float(np.median(vv)) if vv else None
                row[k+'_min']=min(vv) if vv else None;row[k+'_max']=max(vv) if vv else None
            verdict.append(row)
    predictions=[]
    for arm,cond,want,prob in [('ELU','std','L2_FIRST',.85),('GELU','std','L2_FIRST',.8),('SILU','std','L2_FIRST',.8),('R','std','L2_FIRST',.8),('GELU','raw','L1_FIRST',.9),('SILU','raw','L1_FIRST',.9),('R','raw','L1_FIRST',.9),('ELU','raw','L1_FIRST',.6)]:
        actual=next(r for r in verdict if (r['arm'],r['cond'])==(arm,cond))['R1']
        predictions.append(dict(item='R1',arm=arm,cond=cond,prediction=want,actual=actual,hit=want==actual,confidence_codex=prob,issa_agreed=True))
    for arm,prob in [('ELU',.65),('GELU',.55)]:
        actual=next(r for r in verdict if (r['arm'],r['cond'])==(arm,'std'))['R2']
        predictions.append(dict(item='R2',arm=arm,cond='std',prediction='UPSTREAM_CARRIES',actual=actual,hit=actual=='UPSTREAM_CARRIES',confidence_codex=prob,issa_agreed=True))
    ee=[r for r in seed_rows if r['arm']=='ELU' and r['cond']=='std']
    for item,key,fn,p in [('R3','upstream_growth_fraction',lambda x:x>=.5,.7),('bias','bias_absolute_share',lambda x:x<.05,.75)]:
        n=sum(r.get(key) is not None and np.isfinite(r[key]) and fn(r[key]) and (item!='bias' or r['delta']+r['delta_error']<0) for r in ee)
        predictions.append(dict(item=item,arm='ELU',cond='std',prediction='at_least_6_of_10',actual=n,hit=n>=6,confidence_codex=p,issa_agreed=True))
    for name,rows in [('per_seed.csv',seed_rows),('windows.csv',windows),('per_task.csv',task_rows),('verdict.csv',verdict),('predictions.csv',predictions)]:write_csv(out/name,rows)
    R.save_npz(out/'units.npz',units);R.save_npz(out/'ledger_units.npz',ledgers)
    R.put(out/'array_schema.json',dict(units={k:dict(shape=v.shape,dtype=str(v.dtype)) for k,v in units.items()},ledger_units={k:dict(shape=v.shape,dtype=str(v.dtype)) for k,v in ledgers.items()},slots=R.SLOTS,units_task_axis='t00..t50',ledger_task_axis='t01..t50',layer_axis=[1,2],unit_axis='original unit order, 0..99'))
    R.put(out/'verdict.json',dict(run_id=R.RUN,classification_tier='registered_reanalysis_of_known_trajectories',independent_audit=False,verdict=verdict,predictions=predictions,seed_rows=seed_rows))
    def fmt(x):return 'NA' if x is None else (f'{x:.4g}' if isinstance(x,(float,np.floating)) else str(x))
    text=['# CIFAR A6: 崩壊層と輸送帳簿','',
          '既知軌道の解析規則を事前固定した再解析。学習更新なし。独立監査未実施。',
          '主窓は t00→T*（上限t10）。unit平均→seed別比→seed多数の順。比は符号つきで、負値・1超を許す。',
          '', '| arm | cond | R1（理由つき、6/10以上） | R2 | T*中央値 | 上流比中央値 | 自己比中央値 | 交差比中央値 | bias比中央値 |',
          '|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in verdict:text.append('| '+' | '.join(fmt(r.get(k)) for k in ['arm','cond','R1','R2','T_star_median','upstream_share_median','self_share_median','cross_share_median','bias_share_median'])+' |')
    text+=['','比の中央値は定義可能なseedに限る。各列の定義可能数・範囲は verdict.csv、全10seedは per_seed.csv。INITIAL_LOW の比は報告のみ。中央値どうしの和に閉包は要求しない。',
           '', '## 予測', '', '| 項目 | arm/cond | 予測 | 実際 | 的中 |', '|---|---|---|---|---|']
    for p in predictions:text.append(f'| {p["item"]} | {p["arm"]}/{p["cond"]} | {p["prediction"]} | {p["actual"]} | {p["hit"]} |')
    text+=['', '## 検査と限定', '',
           f'- 全306状態の再生・閉包検査: PASS。最大4項閉包誤差/上界 = {max(x.get("closure_max_error_ratio",0) for x in checks["states"]):.4g}。',
           '- 元R=20・slot順・CUDAで再生。保存z(float16)と完全一致、hist平均とCSVの登録列を照合。',
           '- 局所微分は元phiへのautograd。診断dphiとの差は別列。dead_fracと全画像×unitの低応答率を区別。',
           '- 初期から半数以上低応答なら INITIAL_LOW。ReLUは初期から負側微分0なので、この閾値だけでは「新たな崩壊」を識別できない場合がある。',
           '- 同じタスク末で両層が通過した場合は SIMULTANEOUS。タスク内の先後は未測定。',
           '- 上流/自己の寄与は端点間の代数的帰属であり、因果的媒介率ではない。S4/S5には層の候補として渡す。',
           '- 総和とタスク当たり率、t00→t10/t50、t01→t10、t10→t50は windows.csv に REPORT_ONLY として保存。',
           '- 判定後に閾値・窓・集約は変更していない。']
    (out/'summary.md').write_text('\n'.join(text)+'\n')
    print('\n'.join(text[:19]))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--src',type=Path,required=True);a=ap.parse_args();report(a.src)

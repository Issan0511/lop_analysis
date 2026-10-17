"""Literal spec verdict: eight Bonferroni intervals, separate four-contrast separation."""
from __future__ import annotations
import csv,json,math
from pathlib import Path
import numpy as np
from analysis.resp_cifar_ee_0920.stats import t_quantile
from analysis.cifar_ledger_0920.replay import put
CELLS=(('RC','ELU'),('RC','ReLU'),('F51','ELU'),('F51','ReLU'))
EPS_EQ=.005


def interval(d,m):
    d=np.asarray(d,dtype=np.float64)
    if d.shape!=(10,) or not np.isfinite(d).all():raise ValueError('INCOMPLETE: require ten finite paired differences')
    mean=float(d.mean());sd=0. if np.all(d==d[0]) else float(d.std(ddof=1));se=sd/math.sqrt(10)
    q=float(t_quantile(1-.05/(2*m),9));lo,hi=mean-q*se,mean+q*se
    normal=float(t_quantile(.975,9))*se
    return dict(mean=mean,sd=sd,lo=lo,hi=hi,m=m,df=9,level=1-.05/m,
                ordinary_95=[mean-normal,mean+normal],median=float(np.median(d)),differences=d.tolist(),
                positive=int((d>0).sum()),negative=int((d<0).sum()),zero=int((d==0).sum()),
                flags=['DEGENERATE_SD'] if sd==0 else [],POS=lo>0,EQ=lo>=-EPS_EQ and hi<=EPS_EQ)


def classify(d):
    q=interval(d,8)
    q['cell_label']='HELP' if q['POS'] else 'NO_RESCUE' if q['hi']<=0 else 'UNRESOLVED'
    q['harm']=q['hi']<0
    q['equivalence_label']='EQUIVALENT_WITHIN_0.005' if q['EQ'] else None
    return q


def judge(effects,separation=None):
    if set(effects)!=set(CELLS):return dict(status='INCOMPLETE',main_label=None)
    try:
        cells={f'{b}/{a}':classify(effects[b,a]) for b,a in CELLS}
    except ValueError as e:return dict(status='INCOMPLETE',main_label=None,reason=str(e))
    contrasts={}
    for b in ('RC','F51'):
        contrasts[f'activation:{b}']=interval(np.asarray(effects[b,'ELU'])-effects[b,'ReLU'],8)
    for a in ('ELU','ReLU'):
        contrasts[f'box:{a}']=interval(np.asarray(effects['RC',a])-effects['F51',a],8)
    label=lambda b,a:cells[f'{b}/{a}']['cell_label']
    activation=all(label(b,'ELU')=='HELP' and label(b,'ReLU')=='NO_RESCUE' for b in ('RC','F51'))
    box=all(label('RC',a)=='HELP' and label('F51',a)=='NO_RESCUE' for a in ('ELU','ReLU'))
    candidates=[n for n,yes in [('ACTIVATION_SPLITS',activation),('BOX_SPLITS',box)] if yes]
    unresolved=[c for c,v in cells.items() if v['cell_label']=='UNRESOLVED']
    ans=dict(status='COMPLETE',main_label=candidates[0] if len(candidates)==1 else 'MIXED',cells=cells,
        contrasts=contrasts,unresolved_cells=unresolved,unresolved_count=len(unresolved),
        interaction_report_only=interval((np.asarray(effects['RC','ELU'])-effects['RC','ReLU'])-
                                          (np.asarray(effects['F51','ELU'])-effects['F51','ReLU']),1),
        interaction_warning='Report-only ordinary 95% interval; not a ninth member of m=8 family.')
    if not candidates:ans['reason']='NO_UNIQUE_PATTERN'
    if separation is not None:
        if len(separation)!=4:raise ValueError('INCOMPLETE separation contrasts')
        d={k:interval(v,4) for k,v in separation.items()}
        labels=[]
        if d['cap12b-cap12']['POS'] and d['cap12b-ref']['POS']:labels.append('BIAS_SCALING_HELPS')
        if d['lr_only-ref']['POS'] and d['lr_only-cap12']['EQ']:labels.append('STEP_CONTROL_REPRODUCES')
        ans['separation']=dict(contrasts=d,labels=labels or ['SEPARATION_UNRESOLVED'])
    return ans


def endpoints(rows,box):
    tasks=50 if box=='RC' else 30
    keys=[(int(r['seed']),int(r['task'])) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!={(s,k) for s in range(10) for k in range(1,tasks+1)}:
        raise ValueError('INCOMPLETE: missing/duplicate/unregistered per-task rows')
    window=range(31,51) if box=='RC' else (21,23,25,27,29)
    out=[]
    for s in range(10):
        values=[float(r['online_acc']) for r in rows if int(r['seed'])==s and int(r['task']) in window]
        if not np.isfinite(values).all():raise ValueError('DIVERGED: nonfinite online')
        out.append(np.mean(values))
    return np.asarray(out,dtype=np.float64)


def validate_streams(datasets,root,expected_data=None):
    """Pairing is checked across all cells inside each box, never inferred from means."""
    for box in ('RC','F51'):
        paths=[root/'evaluation'/box/a/arm for a in ('ELU','ReLU') for arm in ('ref','cap12')]
        provenances=[json.loads((p/'provenance.json').read_text()) for p in paths]
        data=[p.get('registration',p)['data_sha256'] for p in provenances]
        if any(d!=data[0] for d in data[1:]) or (expected_data is not None and data[0]!=expected_data[box]):
            raise ValueError('CHECK_FAILED: archive image content/hash mismatch')
        if box=='F51':
            manifests=[json.loads((p/'stream_manifest.json').read_text()) for p in paths]
            if any(m!=manifests[0] for m in manifests[1:]):raise ValueError('CHECK_FAILED: F51 stream/init pairing')
        else:
            inputs=[json.loads((p/'input_manifest.json').read_text()) for p in paths]
            if any(m!=inputs[0] for m in inputs[1:]):raise ValueError('CHECK_FAILED: RC input pairing')
            streams=[]
            for activation in ('ELU','ReLU'):
                for arm in ('ref','cap12'):
                    rows=datasets[box,activation,arm]
                    streams.append(sorted((int(r['seed']),int(r['task']),r['label_hash'],r['order_hash']) for r in rows))
            if any(m!=streams[0] for m in streams[1:]):raise ValueError('CHECK_FAILED: RC label/batch pairing')
        for activation in ('ELU','ReLU'):
            a=datasets[box,activation,'ref'];b=datasets[box,activation,'cap12']
            aa=sorted((int(r['seed']),float(r['online_acc'])) for r in a if int(r['task'])==1)
            bb=sorted((int(r['seed']),float(r['online_acc'])) for r in b if int(r['task'])==1)
            if aa!=bb:raise ValueError('CHECK_FAILED: cap changed task1')


def analyze(root):
    root=Path(root);datasets={};ep={}
    try:
        from src import cap_act_box_2x2_1018 as M
        admission=json.loads((root/'evaluation_admission.json').read_text())
        preflight=json.loads((root/'evaluation_preflight.json').read_text())
        if admission['preflight_sha256']!=M.sha256_file(root/'evaluation_preflight.json') or preflight['blockers']:
            raise ValueError('CHECK_FAILED: evaluation admission/preflight')
        if preflight['source_sha256']!=M.source_hashes():raise ValueError('CHECK_FAILED: stale source closure')
        schedule_obj=json.loads((root/'lr_schedule.json').read_text());M.validate_schedule(schedule_obj)
        if admission['schedule_sha256']!=M.sha256_file(root/'lr_schedule.json'):
            raise ValueError('CHECK_FAILED: calibration schedule changed after evaluation admission')
        for box,a in CELLS:
            for arm in ('ref','cap12'):
                p=root/'evaluation'/box/a/arm
                M.verify_completed_arm(p)
                st=json.loads((p/'status.json').read_text())
                if st.get('status')!='COMPLETED':raise ValueError(st.get('status','INCOMPLETE')+': '+str(p))
                prov=json.loads((p/'provenance.json').read_text());reg=prov.get('registration',prov)
                if reg.get('source_sha256')!=M.source_hashes() or reg.get('seeds')!=list(range(10)):
                    raise ValueError('CHECK_FAILED: per-arm source/seed provenance')
                rows=list(csv.DictReader((p/'per_task.csv').open()));datasets[box,a,arm]=rows;ep[box,a,arm]=endpoints(rows,box)
        validate_streams(datasets,root,preflight["data_sha256"])
        for arm in ('cap12b','lr_only'):
            p=root/'evaluation/F51/ReLU'/arm
            M.verify_completed_arm(p)
            if json.loads((p/'status.json').read_text()).get('status')!='COMPLETED':raise ValueError('INCOMPLETE: '+arm)
            prov=json.loads((p/'provenance.json').read_text());reg=prov.get('registration',prov)
            if reg.get('source_sha256')!=M.source_hashes() or reg.get('data_sha256')!=preflight['data_sha256']['F51']:
                raise ValueError('CHECK_FAILED: separation provenance/data')
            if json.loads((p/'stream_manifest.json').read_text())!=json.loads((root/'evaluation/F51/ReLU/ref/stream_manifest.json').read_text()):
                raise ValueError('CHECK_FAILED: separation stream/init mismatch')
            ep['F51','ReLU',arm]=endpoints(list(csv.DictReader((p/'per_task.csv').open())),'F51')
        effects={(b,a):ep[b,a,'cap12']-ep[b,a,'ref'] for b,a in CELLS}
        sep={f'{a}-{b}':ep['F51','ReLU',a]-ep['F51','ReLU',b] for a,b in
             [('cap12b','cap12'),('cap12b','ref'),('lr_only','cap12'),('lr_only','ref')]}
        ans=judge(effects,sep)
        ans['task1_disclosure']={}
        for box,a in CELLS:
            rows=[r for r in datasets[box,a,'ref'] if int(r['task'])==1]
            ans['task1_disclosure'][f'{box}/{a}']=dict(
                online=[float(r['online_acc']) for r in rows],
                memo=[float(r.get('memo_acc',r.get('train_acc','nan'))) for r in rows],
                majority=[float(r.get('major_frac',.2)) for r in rows],
                note='T1 floor proximity is disclosed; no unregistered numeric threshold is imposed.')
        ans['historical_cells']='Rerun, known results; not new independent successes'
        cap_rows={arm:list(csv.DictReader((root/'evaluation/F51/ReLU'/arm/'cap_task.csv').open())) for arm in ('ref','cap12','lr_only')}
        indices={arm:{(int(r['seed']),int(r['task'])):r for r in rr} for arm,rr in cap_rows.items()}
        step_errors=[]
        for seed in range(10):
            for task in range(2,31):
                for layer in (1,2):
                    rr=indices['ref'][seed,task];ll=indices['lr_only'][seed,task];cc=indices['cap12'][seed,task]
                    q=schedule_obj['q'][str(task)][layer-1]
                    target=q*float(rr[f'U_l{layer}']);observed=float(ll[f'U_l{layer}'])
                    step_errors.append(dict(seed=seed,task=task,layer=layer,calibration_q=q,
                        realized_cap_over_ref=float(cc[f'U_l{layer}'])/float(rr[f'U_l{layer}']) if float(rr[f'U_l{layer}']) else None,
                        actual_lr_only_U=observed,target_U=q*float(rr[f'U_l{layer}']),
                        relative_mismatch=(observed-target)/target if target else None))
        M.H.write_csv(root/'step_control_mismatch.csv',step_errors)
        ans['step_matching_note']='Calibration relative-displacement RMS only; evaluation mismatch is reported, not grounds to discard or claim exact effective-lr exclusion.'
        put(root/'analysis_provenance.json',dict(source_sha256=M.source_hashes(),schedule_sha256=M.sha256_file(root/'lr_schedule.json'),
            input_csv_sha256={str(p.relative_to(root)):M.sha256_file(p) for p in sorted((root/'evaluation').rglob('*.csv'))}))
    except (FileNotFoundError,ValueError,KeyError) as e:
        reason=str(e);status=next((s for s in ('CHECK_FAILED','DIVERGED','INCOMPLETE') if s in reason),'INCOMPLETE')
        ans=dict(status=status,main_label=None,reason=reason)
    put(root/'verdict.json',ans)
    summary=f"# {root.name}\n\nStatus: {ans['status']}.\n\n"
    if ans.get('main_label'):summary+=f"Main label: {ans['main_label']}. All eight simultaneous intervals are in verdict.json.\n"
    else:summary+='No scientific verdict was assigned. '+ans.get('reason','')+'\n'
    (root/'summary.md').write_text(summary)
    return ans

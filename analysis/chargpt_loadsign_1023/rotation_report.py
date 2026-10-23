#!/usr/bin/env python3
"""Aggregate section-10 saved-state results; no model execution or new learning."""
import csv
import json
import subprocess
import warnings

import rotation_measure as R
M, np, OUT = R.M, R.np, R.OUT
WINDOWS = dict(early=range(1,10), middle=range(39,51), late=range(89,101))


def records(path):
    rows=[]
    for row in csv.DictReader(path.open()):
        rows.append({k:(v if k in ['arm','window','direction','alive_rule'] else float(v) if v else float('nan')) for k,v in row.items()})
    return rows


def med(x):
    x=np.asarray(x,dtype=float); x=x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float('nan')


def span(rows,key,digits=4):
    vals=[r[key] for r in rows if np.isfinite(r[key])]
    return f'{min(vals):.{digits}f}–{max(vals):.{digits}f}' if vals else 'NA'


def f(x): return f'{x:.4f}' if np.isfinite(x) else 'NA'


def aggregate(states):
    rows=[]
    keys=[k for k in states[0] if k.startswith(('cos_','rot','s_','abs_')) and k!='rotation_pair']
    for arm in R.A.ARMS:
        for seed in range(10):
            for name,tasks in WINDOWS.items():
                selected=[r for r in states if r['arm']==arm and r['seed']==seed and r['task'] in tasks]
                assert len(selected)==len(tasks)
                row=dict(arm=arm,seed=seed,window=name,n_states=len(selected),
                         n_alive_median=med([r['n_alive'] for r in selected]),
                         n_alive_min=min(r['n_alive'] for r in selected),n_alive_max=max(r['n_alive'] for r in selected),
                         n_cos_missing_sum=sum(r['n_cos_missing'] for r in selected))
                row.update({k:med([r[k] for r in selected]) for k in keys}); rows.append(row)
    M.write_csv(OUT/'windows.csv',[R.clean(r) for r in rows]); return rows


def correlations():
    rows=[]
    for arm in R.A.ARMS:
        for seed in range(10):
            path=OUT/'raw'/f'{arm}_seed{seed}.npz'
            if not path.exists():
                path=M.DATA/f'chargpt_loadsign_1023/results/chargpt_loadsign_1023/rotation/raw/{arm}_seed{seed}.npz'
                manifest=json.loads((OUT/'backup_manifest.json').read_text())
                record=next(r for r in manifest['files'] if r['backup']==str(path))
                assert M.sha(path)==record['sha256']
            with np.load(path) as raw:
                for name,tasks in WINDOWS.items():
                    tasks=[t for t in tasks if t-1 in R.A.TASKS]
                    alpha=np.stack([raw[f't{t}_cos'] for t in tasks])
                    alive=np.stack([raw[f't{t}_alive'] for t in tasks])
                    prev=np.stack([raw[f't{t}_prev_alive'] for t in tasks])
                    for rule in ['current','both']:
                        for direction in ['T','W','U']:
                            rot=np.stack([raw[f't{t}_rot{direction}'] for t in tasks])
                            valid=alive&np.isfinite(alpha)&np.isfinite(rot)
                            if rule=='both': valid &= prev
                            count=valid.sum(0); keep=count>=3
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore',RuntimeWarning)
                                am=np.nanmedian(np.where(valid,alpha,np.nan),axis=0)[keep]
                                rm=np.nanmedian(np.where(valid,rot,np.nan),axis=0)[keep]
                            neg=(valid&(alpha<0)).sum(0)[keep]/count[keep]
                            per_state=[R.rho(a[v],r[v]) for a,r,v in zip(alpha,rot,valid)]
                            rows.append(dict(arm=arm,seed=seed,window=name,alive_rule=rule,direction=direction,
                                n_units=int(keep.sum()),n_unit_task_pairs=int(valid[:,keep].sum()),n_all_valid_pairs=int(valid.sum()),
                                n_tasks=len(tasks),rho_alignment_rotation=R.rho(am,rm),rho_misalignment_rotation=R.rho(-am,rm),
                                rho_negative_frequency_rotation=R.rho(neg,rm),rho_state_median=med(per_state),
                                rho_pooled=R.rho(alpha[valid],rot[valid]),rotation_median=med(rm),
                                n_negative_alignment_units=int((am<0).sum()),
                                rotation_negative_alignment_median=med(rm[am<0]),rotation_nonnegative_alignment_median=med(rm[am>=0])))
    M.write_csv(OUT/'correlations.csv',[R.clean(r) for r in rows]); return rows


def judge(windows,corr,elr):
    def counts(arm,window,pred):
        hits=[int(r['seed']) for r in windows if r['arm']==arm and r['window']==window and pred(r)]
        return dict(n=len(hits),denominator=10,seeds=hits)
    p1={a:counts(a,'early',lambda r:r['cos_positive']>=.95) for a in R.A.ARMS}
    p2=counts('C3','late',lambda r:.4<=r['cos_positive']<=.6)
    p3=counts('ReLUcap','late',lambda r:r['cos_positive']>=.9)
    p4={}
    for direction in ['T','W','U']:
        selected=[r for r in corr if r['arm']=='C3' and r['window']=='late' and r['alive_rule']=='current' and r['direction']==direction]
        hits=[r['seed'] for r in selected if r['rho_misalignment_rotation']<0]
        p4[direction]=dict(n=len(hits),denominator=10,seeds=hits,missing=sum(not np.isfinite(r['rho_misalignment_rotation']) for r in selected),supported=len(hits)>=8)
    elate=[r for r in elr if r['arm']=='ELR' and r['task']==100]
    p5=[int(r['seed']) for r in elate if r['s_other_direct']>=.8]
    result=dict(P1=dict(probability=.7,supported=all(v['n']>=8 for v in p1.values()),arms=p1,independent=False),
                P2=dict(probability=.7,supported=p2['n']>=8,**p2,independent=False),
                P3=dict(probability=.7,supported=p3['n']>=8,**p3,independent=False),
                P4=dict(probability=.5,supported=p4['T']['supported'],directions=p4,definition='rho(-alignment,rotation)<0'),
                P5=dict(probability=.5,n=len(p5),denominator=len(elate),seeds=p5,ten_seed_judgment='WITHHELD',
                        available_states_all_pass=bool(elate) and len(p5)==len(elate)))
    R.dump(OUT/'judge.json',result)
    lines=['# §10 予測の採点','',f"- P1 (.7): {'支持' if result['P1']['supported'] else '不支持'}。早期≥.95: "+', '.join(f"{a} {v['n']}/10" for a,v in p1.items()),
           f"- P2 (.7): {'支持' if result['P2']['supported'] else '不支持'}。後期C3 .40–.60: {p2['n']}/10。",
           f"- P3 (.7): {'支持' if result['P3']['supported'] else '不支持'}。後期ReLUcap ≥.90: {p3['n']}/10。",
           f"- P4 (.5): {'支持' if result['P4']['supported'] else '不支持'}。後期C3 rho(-alpha,rot)<0: "+', '.join(f"{d} {v['n']}/10" for d,v in p4.items())+'。',
           f'- P5 (.5): 利用可能なELR課題100末でother direct≥.8: {len(p5)}/{len(elate)}。10seed判定は保留。','',
           'P1–P3は既存の他unit direct項の正負と恒等的に同じで、独立な予測成功とは数えない。既存J1–J3は変更しない。',
           'P4の符号は測定前に固定: 低い整列と小さい回転の対応はrho(alpha,rot)>0、rho(-alpha,rot)<0。']
    (OUT/'judge.md').write_text('\n'.join(lines)+'\n'); return result


def report(states,windows,corr,elr,result):
    gate=json.loads((OUT/'gate.json').read_text()); eg=json.loads((OUT/'elr_gate.json').read_text())
    pv=json.loads((OUT/'provenance.json').read_text())
    lines=['# 読み出し整列・端点回転・保存ELRの検査（§10）','',
        '保存状態のみ、学習・optimizerなし。主probeは第三段階と同じ独立一様文字列。H1監査のmatched-inputとは異なる。',
        'T_i=ΓPU_i、E_i=mean[(K gate_i/σ)(p-u)]、R_i=Σ_{j≠i}T_j〈B_j,E_i〉、alpha_i=cos(T_i,R_i)。',
        'S_other,direct,i=κ〈T_i,R_i〉、κ>0。これは誤差・gate・Kを含む押しの整列で、独立な純粋幾何指標ではない。',
        '集約は各状態のalive unit → seed内窓中央値 → seed間範囲。cosノルム0は欠測、正の割合の分母は全alive。','',
        '| arm | 窓 | cos>0 割合 | cos中央値 | n_alive中央値 | 欠測cos総数 |',
        '|---|---|---:|---:|---:|---:|']
    for arm in R.A.ARMS:
        for name in WINDOWS:
            rr=[r for r in windows if r['arm']==arm and r['window']==name]
            lines.append(f"| {arm} | {name} | {span(rr,'cos_positive')} | {span(rr,'cos_q50')} | {span(rr,'n_alive_median',1)} | {sum(r['n_cos_missing_sum'] for r in rr):.0f} |")
    lines+=['','## 回転との相関','',
        '表はrho(alpha, rot)。負の整列ほど小さい回転という予測に対応する符号は正。採点には符号を反転したrho(-alpha,rot)を使用。',
        '後期90..100、現在alive、3ペア以上ある同一unitごとのalpha中央値と回転中央値を相関。下表のn/pairsは3ペア以上のunitのみ。',
        'rotTは最終LNγも含む読み出し方向、rotUはdown列、rotWはup行。端点変位であり学習中の累積回転ではない。','',
        '| arm | seed | rho T | rho W | rho U | n_units | unit-task pairs |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in R.A.ARMS:
        for seed in range(10):
            cr={r['direction']:r for r in corr if r['arm']==arm and r['seed']==seed and r['window']=='late' and r['alive_rule']=='current'}
            lines.append(f"| {arm} | {seed} | {f(cr['T']['rho_alignment_rotation'])} | {f(cr['W']['rho_alignment_rotation'])} | {f(cr['U']['rho_alignment_rotation'])} | {cr['T']['n_units']} | {cr['T']['n_unit_task_pairs']} |")
    lines+=['','| arm | direction | 回転中央値のseed範囲 | rho（両端alive） | rho（負整列頻度,回転） | rho（課題内相関の中央値） |',
            '|---|---|---:|---:|---:|---:|']
    for arm in R.A.ARMS:
        for direction in ['T','W','U']:
            rr=[r for r in corr if r['arm']==arm and r['window']=='late' and r['direction']==direction]
            c=[r for r in rr if r['alive_rule']=='current']; b=[r for r in rr if r['alive_rule']=='both']
            lines.append(f"| {arm} | {direction} | {span(c,'rotation_median',7)} | {span(b,'rho_alignment_rotation')} | {span(c,'rho_negative_frequency_rotation')} | {span(c,'rho_state_median')} |")
    lines+=['','全window、unit×task pool相関、負/非負整列unitの回転中央値もcorrelations.csvに記録。poolを独立標本の検定には使わない。','',
        '## ELR（L0のみ操作、L1無操作）','',
        '既存row5 ELRのseed5–9に課題50/100末の計10 checkpointを確認。seed0–4の保存状態および課題89–99末は手元で確認できない。',
        'L0 phi=ReLU(k*z)のkはstate_dict外。原コードどおり対照C3のfloat32行ノルム比から復元。k、行ノルム、L1 chain統計を保存CSVと照合。',
        '以下は同じseed・同じ課題末同士。self/other/residualはdirect成分、s_loadはLN分散微分を含む全成分の合計。','',
        '| task | arm | n seed | s_self | s_other | s_residual | s_load (LN込み) | n_alive |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    for task in [50,100]:
        for arm in ['C3','ReLUcap','ELR']:
            rr=[r for r in elr if r['task']==task and r['arm']==arm]
            lines.append(f"| {task} | {arm} | {len(rr)} | {span(rr,'s_self_direct')} | {span(rr,'s_other_direct')} | {span(rr,'s_residual_direct')} | {span(rr,'s_load_exact')} | {span(rr,'n_alive',0)} |")
    lines+=['','| ELR task | seed | n_alive | s_self | s_other | s_load LN込み |',
            '|---:|---:|---:|---:|---:|---:|']
    for r in elr:
        if r['arm']=='ELR': lines.append(f"| {int(r['task'])} | {int(r['seed'])} | {int(r['n_alive'])} | {f(r['s_self_direct'])} | {f(r['s_other_direct'])} | {f(r['s_load_exact'])} |")
    lines+=['','ELRの分解は利用可能な5seedだけ。aliveが少ない場合、割合は少数unitで大きく動く。ELRで項が保たれることと回転による媒介の証明は別。','',
        '## 検証・採点','',
        f"- seed0両腕70状態: 前段配列に一致。最大相対誤差 {gate['max_previous_error']:.3e}、明示j≠i和 {gate['max_explicit_error']:.3e}。",
        f"- 全{len(states)}状態: alive・全unit成分符号一致、最大相対誤差 {pv['max_previous_error']:.3e}。cos符号とother direct符号は全unitで一致。",
        f"- ELR復元 {len(eg['checks'])}状態通過。L1合計とautogradの最大相対誤差 {eg['max_autograd_error']:.3e}。",
        f"- 実行コードcommit: {pv['git_hash']}。spec登録commit: 1fd7e182。CPU float64/8threads。",
        '- 実行前python数2。nvidia-smiはドライバ接続不可。CPUのみ使用し他の走には触れていない。','']
    lines+=(OUT/'judge.md').read_text().splitlines()[2:]
    lines+=['','この検査は固定状態の共変動。回転凍結が他unit項の反転を引き起こしたという因果を単独では同定しない。']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    R.dump(OUT/'report_provenance.json',dict(code_sha256=M.sha(__file__),
        input_states_sha256=M.sha(OUT/'states.csv'),input_elr_states_sha256=M.sha(OUT/'elr_states.csv'),
        git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=M.ROOT,text=True).strip()))


def main():
    states=records(OUT/'states.csv'); assert len(states)==700
    windows=aggregate(states);corr=correlations();elr=records(OUT/'elr_states.csv')
    result=judge(windows,corr,elr);report(states,windows,corr,elr,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()

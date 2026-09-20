#!/usr/bin/env python3
"""S5 complete-run report. Explicit source; no partial scientific verdicts."""
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from src.cap_cifar_ee_0920 import ROOT,RUN,ARMS,write_csv,put,sha,verify_done,artifact,L
from analysis.cap_cifar_ee_0920.stats import evaluate


def read_npz(p):
    with np.load(artifact(p),allow_pickle=False) as f:return {k:f[k] for k in f.files}


def validate_coverage(rows):
    assert len(rows)==500 and {(r['seed'],r['task']) for r in rows}=={(s,t) for s in range(10) for t in range(1,51)},'row coverage'


def read_rows(src,ident):
    verify_done(src/'prefix/done.json',ident);allrows=[];data={};group={}
    reference=None
    prefix=json.loads((src/'prefix/rows.json').read_text());prefix={r['seed']:r for r in prefix}
    assert set(prefix)==set(range(10))
    for arm in ARMS:
        folder=src/'arms'/arm;verify_done(folder/'done.json',ident)
        branch=json.loads((folder/'branch.json').read_text());assert branch['core_sha256']==branch['anchor_sha256']
        if reference is None:reference=branch['anchor_sha256']
        assert reference==branch['anchor_sha256']
        rows=json.loads((folder/'rows.json').read_text())
        validate_coverage(rows)
        assert all(r['arm']==arm and r['finite'] for r in rows)
        for r in rows:
            assert all(np.isfinite(r[k]) for k in ('online_acc','memo_acc','online_ce','G1','G2','Q1','Q2'))
            if r['task']==1:
                for k,v in prefix[r['seed']].items():assert r[k]==v
            for l in (1,2):
                assert r.get(f'violations_l{l}',0)==0 and r.get(f'bfix_errors_l{l}',0)==0
        rows.sort(key=lambda r:(r['seed'],r['task']));allrows.extend(rows);group[arm]=rows
        data[arm]={k:np.asarray([r[field] for r in rows]).reshape(10,50) for k,field in [('A','online_acc'),('G','G2'),('F','major_frac')]}
    for arm,rows in group.items():
        for r,ref in zip(rows,group['ref']):
            for k in ('seed','task','label_hash','order_hash','major_frac'):assert r[k]==ref[k]
    return allrows,data,group


def ledgers(src):
    rows=[];closure_max=0.;native_max=0.
    for arm in ARMS:
        root=src/'arms'/arm
        states={t:read_npz(root/f'ledger_state_t{t:02d}.npz') for t in (1,10,30,50)}
        terms={t:read_npz(root/f'ledger_t{t:02d}.npz') for t in range(2,51)}
        for t,q in terms.items():
            err=q['closure_error'];res=abs(q['closure']);assert np.all(res<=err)
            closure_max=max(closure_max,float(np.divide(res,err,out=np.zeros_like(res),where=err>0).max()))
        for st in states.values():
            assert np.all(st['m_native_residual']<=st['m_native_bound'])
            native_max=max(native_max,float((st['m_native_residual']/st['m_native_bound']).max()))
        for lo,hi in ((1,10),(1,50),(30,50)):
            for seed in range(10):
                end=L.Ball(states[hi]['m_alg'][seed],states[hi]['m_alg_error'][seed])-L.Ball(states[lo]['m_alg'][seed],states[lo]['m_alg_error'][seed]);delta=end.sum()/100
                for key in L.TERMS+('upstream_growth','upstream_rotation','self_growth','self_rotation'):
                    values=np.stack([terms[t][key][seed] for t in range(lo+1,hi+1)]);errors=np.stack([terms[t][key+'_error'][seed] for t in range(lo+1,hi+1)])
                    total=L.Ball(values,errors).sum(0);mean=total.sum()/100
                    ratio=mean/delta if delta.v+delta.e<0 else None
                    rows.append(dict(arm=arm,seed=seed,window=f'{lo}-{hi}',term=key,total=float(mean.v),error=float(mean.e),per_task=float(mean.v)/(hi-lo),unit_median=float(np.median(total.v)),delta=float(delta.v),delta_error=float(delta.e),signed_share=float(ratio.v) if ratio is not None else None,share_error=float(ratio.e) if ratio is not None else None))
    return rows,dict(all_pass=True,ledger_error_ratio=closure_max,native_mean_error_ratio=native_max)


def report(src):
    src=Path(src);status=json.loads((src/'status.json').read_text());assert status['stage']=='completed' and status['tasks']==50,'incomplete'
    ident=json.loads((src/'provenance_start.json').read_text())['identity'];assert ident['epochs']==400 and ident['tasks']==50 and ident['seeds']==list(range(10))
    checks=json.loads((src/'admission_checks.json').read_text());assert checks['all_pass'] and checks['source_sha256']==ident['source_sha256']
    pairing=json.loads((src/'pairing_checks.json').read_text());assert pairing['all_pass'] and pairing['tasks']==50
    rows,data,group=read_rows(src,ident);result=evaluate(data);ledger,lc=ledgers(src)
    engagement=[];bchanges={}
    for arm in ARMS:
        units1=read_npz(src/'arms'/arm/'units_t01.npz');units50=read_npz(src/'arms'/arm/'units_t50.npz')
        bchanges[arm]=float((units50['bias_l2']-units1['bias_l2']).mean())
        for seed in range(10):
            for layer in ARMS[arm]:
                hits=sum(r.get(f'hits_l{layer}',0) for r in group[arm] if r['seed']==seed and r['task']>=2)
                engagement.append(dict(arm=arm,seed=seed,layer=layer,hits=int(hits),status='ENGAGED' if hits else 'CAP_NOT_ENGAGED'))
    timingrows=[]
    for arm,tt in result['timing'].items():
        for q in tt['rows']:
            timingrows.append(dict(arm=arm,seed=q['seed'],sign=q['sign'],arm_task=q['arm_time']['task'],arm_censored=q['arm_time']['censored'],ref_task=q['ref_time']['task'],ref_censored=q['ref_time']['censored'],undefined=q['arm_time']['undefined'] or q['ref_time']['undefined'],label=tt['label'],p=tt['p']))
    predictions=[dict(metric='applicability',probability=.90,hit=result['applicable'],brier=(.9-float(result['applicable']))**2)]
    a=result['arms'];facts=[('cap12_RESCUED',.65,a['cap12']['label']=='RESCUED'),('cap1_COLLAPSED',.65,a['cap1']['label']=='COLLAPSED'),('cap2_SPLIT_or_COLLAPSED',.75,a['cap2']['label'] in ('SPLIT','COLLAPSED')),('cap12_bfix_RESCUED',.80,a['cap12_bfix']['label']=='RESCUED'),('cap1_LATER',.65,None if result['timing']['cap1']['label']=='TIMING_UNDETERMINED' else result['timing']['cap1']['label']=='LATER'),('cap12_b2_negative',.75,bchanges['cap12']<0),('cap12_no_early_impairment',.70,None if a['cap12']['early']=='EARLY_FLAG_NOT_ASSESSABLE' else a['cap12']['early']!='IMPAIRED_EARLY')]
    for metric,p,hit in facts:predictions.append(dict(metric=metric,probability=p,hit=hit if result['applicable'] else None,brier=(p-float(hit))**2 if result['applicable'] and hit is not None else None,unscored_reason=None if result['applicable'] and hit is not None else 'NOT_REPRODUCED' if not result['applicable'] else 'TIMING_UNDETERMINED' if metric=='cap1_LATER' else 'EARLY_FLAG_NOT_ASSESSABLE'))
    write_csv(src/'per_task.csv',rows);write_csv(src/'per_seed.csv',result['per_seed']);write_csv(src/'paired.csv',result['paired']);write_csv(src/'secondary.csv',result['secondary']);write_csv(src/'timing.csv',timingrows);write_csv(src/'ledger_summary.csv',ledger);write_csv(src/'engagement.csv',engagement);write_csv(src/'predictions.csv',predictions)
    write_csv(src/'interaction.csv',[dict(metric=k,**v,tier='REPORT_ONLY') for k,v in result['interactions'].items()])
    write_csv(src/'verdict.csv',[dict(arm=arm,label=q['label'],floor_count=q['floor_count'],online_first=q['online_first'],online_late=q['online_late'],G_first=q['G_first'],G_late=q['G_late'],E1=q['E1']['mean'],E1_low=q['E1']['low'],E1_high=q['E1']['high'],E2=q['E2']['mean'],E2_low=q['E2']['low'],E2_high=q['E2']['high'],level=q['E1']['level'],early=q['early'],b2_change=bchanges[arm]) for arm,q in a.items()])
    s4=json.loads((ROOT/'results/resp_cifar_ee_0920/prefix/rows.json').read_text());lookup={(r['seed'],r['task']):r for r in s4}
    write_csv(src/'S4_natural_comparison.csv',[dict(seed=r['seed'],task=r['task'],S5_online=r['online_acc'],S4_online=lookup[r['seed'],r['task']]['online_acc'],bit_equal=r['online_acc']==lookup[r['seed'],r['task']]['online_acc']) for r in group['ref'] if r['task']<=11])
    result.update(b2_change=bchanges,independent_audit=False,registration='f524cac',approval='5705359',cap_not_engaged=[r for r in engagement if r['status']=='CAP_NOT_ENGAGED'])
    put(src/'verdict.json',result);put(src/'checks.json',dict(all_pass=True,pre_run=checks,pairing=pairing,ledger=lc,independent_audit=False))
    lines=['# CIFAR S5: 層別の行ノルム上限','',f"登録主判定: **{result['label']}**。自然現象の適用条件: {result['applicable']}。",'',
           'ELU/std、R=10、10 seed、50task×30,000更新。S5自身のtask1終端から分岐。主窓はtask31–50。独立監査なし。','',
           '| 腕 | 判定 | 床seed | 初回online | 主窓online | 主窓G2 | 初期悪影響 |','|---|---|---:|---:|---:|---:|---|']
    for arm,q in a.items():lines.append(f"| {arm} | {q['label']} | {q['floor_count']}/10 | {q['online_first']:.5f} | {q['online_late']:.5f} | {q['G_late']:.5g} | {q['early']} |")
    lines+=['','| cap12主対応差 | 平均 | 97.5%下端 | 上端 |','|---|---:|---:|---:|']
    for k in ('E1','E2'):
        q=a['cap12'][k];lines.append(f"| {k} | {q['mean']:.6g} | {q['low']:.6g} | {q['high']:.6g} |")
    lines+=['',f"cap12のb2平均変化(t1→t50): {bchanges['cap12']:.6g}。",f"未発火の対象seed×層: {len(result['cap_not_engaged'])}（engagement.csv）。",'',
            '床は各seedのラベル構成から定めた操作的な帯。モデルとの同等性を証明する検定ではない。全更新で丸め誤差を含むノルム上界とbias固定を監視した。',
            'cap12_bfixはb1/b2を同時に書き戻し、通常の勾配とAdam momentを保持する。どちらのbiasが原因かは分離していない。',
            '主窓以外、層単独腕、時間、交互作用、輸送帳簿はREPORT_ONLY。非検出を効果ゼロと読まない。',
            '結論の範囲は固定した1200枚・ELU/std・この半径・50task。S4と合わせても自然な全媒介や恒久的救済の証明にはしない。']
    (src/'summary.md').write_text('\n'.join(lines)+'\n')
    put(src/'report_provenance.json',dict(git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/f'analysis/{RUN}').glob('*.py')},input_identity=ident))
    print(json.dumps(dict(label=result['label'],applicable=result['applicable'],arms=result['arms']),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--src',required=True);a=p.parse_args();report(Path(a.src))

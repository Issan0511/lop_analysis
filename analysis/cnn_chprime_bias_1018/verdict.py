#!/usr/bin/env python3
"""P181's fixed .5 and 9/10 rule, plus one paired CHf comparison. No seed deletion."""
from pathlib import Path
import argparse, csv, json, math, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src import cnn_chprime_bias_1018 as M


def classify(rows, arm='CHf_B'):
    problems=[];index={};diverged=False
    for row in rows:
        try:
            k=(int(row['seed']),int(row['task']))
            if k in index: problems.append('duplicate:'+str(k))
            if row.get('arm')!=arm: problems.append('arm:'+str(k))
            value=float(row['online_acc'])
            if not math.isfinite(value):
                problems.append('nonfinite:'+str(k));diverged=True
            if str(row.get('diverged','false')).lower()=='true': diverged=True
            index[k]=value
        except (ValueError,KeyError,TypeError): problems.append('invalid row')
    expected={(s,t) for s in M.SEEDS for t in range(1,31)}
    if set(index)!=expected: problems.append('missing_or_extra_cells')
    if diverged: return dict(label='DIVERGED',problems=problems)
    if problems: return dict(label='INCOMPLETE',problems=problems)
    windows=[float(np.mean([index[s,t] for t in range(21,31)])) for s in M.SEEDS]
    kept=sum(w>=.5 for w in windows)
    return dict(label='RESCUED' if kept>=9 else 'FAIL' if kept<=1 else 'PARTIAL',
                kept=kept,n=10,threshold=.5,window_tasks=list(range(21,31)),
                seeds=M.SEEDS,W=windows,mean_W=float(np.mean(windows)),problems=[])


def compare(primary, baseline):
    d=np.asarray(primary['W'])-np.asarray(baseline['W'])
    se=float(d.std(ddof=1)/np.sqrt(10));mu=float(d.mean())
    radius=2.2621571628540993*se
    npos=int((d>0).sum());nneg=int((d<0).sum());n=npos+nneg
    return dict(difference=d.tolist(),mean=mu,ci95=[mu-radius,mu+radius],median=float(np.median(d)),
                positive=npos,negative=nneg,ties=10-n,
                sign_p=min(1.,2*sum(math.comb(n,k) for k in range(min(npos,nneg)+1))/2**n) if n else 1.,
                sd_flag='DEGENERATE_SD' if se==0 else None)


def report(out, baseline, checks):
    def read(path):
        return list(csv.DictReader(path.open())) if path.exists() else []
    primary=classify(read(out/'per_task.csv'))
    integrity=[]
    prov=json.loads((out/'provenance.json').read_text()) if (out/'provenance.json').exists() else {}
    failure=json.loads((out/'failure.json').read_text()) if (out/'failure.json').exists() else {}
    if prov.get('status')=='DIVERGED' or failure.get('status')=='DIVERGED':
        primary=dict(label='DIVERGED',observed=primary,problems=['numerical stop recorded in provenance/failure'],failure=failure)
    elif prov.get('status')=='CHECK_FAILED' or failure.get('status')=='CHECK_FAILED':
        primary=dict(label='CHECK_FAILED',observed=primary,problems=['implementation failure recorded in provenance/failure'],failure=failure)
    if primary['label'] in ('RESCUED','FAIL','PARTIAL'):
        release=out.parent/'gpu_release.json'
        gpu=json.loads(release.read_text()) if release.exists() else {}
        if gpu.get('all_pass') is not True or gpu.get('sources')!=M.source_hashes() or not gpu.get('independent_reviewer'):
            integrity.append('missing/stale independent GPU release')
        if prov.get('status')!='COMPLETE': integrity.append('main provenance not COMPLETE')
        if prov.get('sources')!=M.source_hashes(): integrity.append('source hashes differ')
        hashes=prov.get('artifact_hashes',{})
        if not hashes: integrity.append('artifact hash manifest absent')
        for name,expected in hashes.items():
            p=out/name
            if not p.exists() or M.sha(p)!=expected: integrity.append('artifact hash:'+name)
        for t in M.CF.CKPT_TASKS:
            for seed in M.SEEDS:
                if not (out/'ckpt'/f'CHf_B_seed{seed}_t{t:02d}.pt').exists(): integrity.append('missing checkpoint')
        for t in range(1,31):
            for u,label in ((0,'start'),(1,'switch'),(10,'switch'),(100,'switch'),(1000,'switch'),(30000,'end')):
                if not (out/'diagnostics'/f't{t:02d}_u{u:05d}_{label}.npz').exists(): integrity.append('missing diagnostics')
            for e in range(1,401):
                if not (out/'projection'/f't{t:02d}_e{e:03d}.npz').exists(): integrity.append('missing projection trace')
        if not checks.exists(): integrity.append('missing checks')
        else:
            c=json.loads(checks.read_text())
            if c.get('all_pass') is not True or c.get('sources')!=M.source_hashes(): integrity.append('stale/failed checks')
        if integrity: primary=dict(label='CHECK_FAILED',observed=primary,problems=sorted(set(integrity)))
    historical_path=ROOT/'results/cnn_center_feat_1011/CHf'
    old=M.reuse_audit(baseline) if baseline.resolve()==historical_path.resolve() else dict(qualified=False,problems=['fresh precision-matched CHf incomplete'])
    secondary=dict(status='PENDING_FRESH_CHf',reason=old['problems'])
    bprov=json.loads((baseline/'provenance.json').read_text()) if (baseline/'provenance.json').exists() else {}
    newb=classify(read(baseline/'per_task.csv'),'CHf')
    if bprov.get('run_id')==M.RUN and bprov.get('status')=='COMPLETE':
        # Both controls must have the same pinned sources, data hash and all
        # protocol fields other than the intervention arm.
        bcfg=dict(bprov.get('config',{}));pcfg=dict(prov.get('config',{}))
        bcfg.pop('arm',None);pcfg.pop('arm',None)
        bhashes=bprov.get('artifact_hashes',{})
        baseline_hash_ok=bool(bhashes) and all((baseline/name).is_file() and M.sha(baseline/name)==expected for name,expected in bhashes.items())
        if bcfg==pcfg and bcfg and baseline_hash_ok and newb['label'] in ('RESCUED','FAIL','PARTIAL'):
            if primary['label'] in ('RESCUED','FAIL','PARTIAL'): secondary=dict(status='COMPLETE',**compare(primary,newb))
    elif old['qualified'] and primary['label'] in ('RESCUED','FAIL','PARTIAL'):
        secondary=dict(status='COMPLETE',**compare(primary,newb))
    result=dict(run_id=M.RUN,primary=primary,secondary=secondary,prediction='RESCUED',
                checks_status='CPU_ONLY' if not M.torch.cuda.is_available() else 'SEE_RELEASE',
                reference_reused=secondary['status']=='COMPLETE' and old['qualified'])
    M.json_write(out.parent/'verdict.json',result)
    (out.parent/'summary.md').write_text(
        f"# {M.RUN}\n\n主判定: **{primary['label']}**。予測: RESCUED（共通spec §8）。\n\n"
        f"同一精度のCHfとの差: {secondary['status']}。P183のcudnn TF32=trueは新仕様のTF32=falseと異なり、同一条件の対照として再利用しない。\n\n"
        "CUDA未実行の検査をPASSや性能FAILとは扱わない。CPU合成検査は実装の検査である。\n")
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'results'/M.RUN/'CHf_B')
    p.add_argument('--baseline',type=Path,default=ROOT/'results'/M.RUN/'CHf')
    p.add_argument('--checks',type=Path,default=ROOT/'results'/M.RUN/'checks.json')
    a=p.parse_args();print(json.dumps(report(a.out,a.baseline,a.checks),indent=2))

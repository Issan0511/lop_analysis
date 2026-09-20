"""Complete-only report. Calibration writes a window; main report requires its commit."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from src.drive_cifar_c_0920 import ROOT,RUN,N,put,sha,sources,load_pt,E
from analysis.drive_cifar_c_0920.stats import directional,certificate_status,window

def validate(out,production=True):
    out=Path(out);m=json.loads((out/'complete.json').read_text());ident=json.loads((out/'input_manifest.json').read_text())
    assert m['identity']==ident and ident['source_sha256']==sources(),'identity/source mismatch'
    assert ident['run_id']==RUN and ident['R']==10
    if production:assert ident['mode']=='production' and ident['seeds']==list(range(10)) and ident['tasks']==5 and ident['epochs']==400,'INCOMPLETE production'
    else:assert ident['mode']=='check' and ident['seeds']==list(range(100,110))
    tasks,eps=ident['tasks'],ident['epochs']
    assert m['tc']==tasks*eps*75 and m['task']==tasks and m['epochs']==eps
    paths=[f['path'] for f in m['files']];assert len(paths)==len(set(paths))
    expected={f'{group}/t{t}_e{e:03}.npz' for group in ('primary','calibration') for t in range(1,tasks+1) for e in range(1,eps+1)}
    expected|={f'audit/t{t}_e{e:03}.pt' for t in range(1,tasks+1) for e in {1,eps}}|{'per_task.csv'}
    assert set(paths)==expected,'missing or foreign shard'
    for f in m['files']:
        p=out/f['path'];assert p.is_relative_to(out) and '..' not in p.parts and sha(p)==f['sha256'],'shard checksum'
    assert sha(out/'checkpoint.pt')==m['checkpoint_sha256']
    st=load_pt(out/'checkpoint.pt');assert st['identity']==ident and st['engine']['tc']==m['tc'] and not st['engine']['in_task']
    assert st['engine']['epoch']==eps and len(st['rows'])==10*tasks
    return m

def shard(out,group,t,e,seeds):
    with np.load(Path(out)/group/f't{t}_e{e:03}.npz',allow_pickle=False) as d:
        assert d['seeds'].tolist()==seeds and int(d['task'])==t and int(d['epoch'])==e and int(d['steps'])==75
        assert d['keys'].tolist()==list(N.KEYS)
        a=d['sum'];assert a.shape==(5,100,len(N.KEYS)) and np.isfinite(a).all()
        for k in ('fail','nonfinite','certificate_violation'):assert (a[:,:,N.KEYS.index(k)]==0).all(),'CHECK_FAILED'
        assert (a[:,:,N.KEYS.index('cert_down')]+a[:,:,N.KEYS.index('cert_up')]+a[:,:,N.KEYS.index('uncertain')]==75).all()
        return a

def calibrate(out,destination):
    out=Path(out);validate(out)
    v=np.zeros(400)
    for t in range(2,6):
        for e in range(1,401):v[e-1]+=shard(out,'calibration',t,e,list(range(5,10)))[:,:,N.KEYS.index('total_sum')].mean()/75/4
    w=window(v);w.update(run_id=RUN,complete_sha256=sha(out/'complete.json'),calibration_seeds=list(range(5,10)),rule_source=sources(),v=v.tolist())
    put(destination,w);return w['label']

def report(out,window_path,destination):
    out=Path(out);destination=Path(destination);validate(out)
    window_path=Path(window_path).resolve();rel=window_path.relative_to(ROOT)
    committed=subprocess.check_output(['git','show',f'HEAD:{rel}'],cwd=ROOT)
    assert committed==window_path.read_bytes(),'commit calibration before opening primary summaries'
    w=json.loads(committed);assert w['complete_sha256']==sha(out/'complete.json') and w['rule_source']==sources() and w['calibration_seeds']==list(range(5,10))
    allsum=np.zeros((5,100,len(N.KEYS)));early=np.zeros((5,4));late=early.copy();taskrows=[];b=w['break_epoch']
    for t in range(2,6):
        task=np.zeros_like(allsum)
        for ep in range(1,401):
            a=shard(out,'primary',t,ep,list(range(5)));task+=a
            if b:
                counts=a[:,:,N.KEYS.index('cert_down')].sum(1)
                (early if ep<=b else late)[:,t-2]+=counts
        allsum+=task
        for s in range(5):
            row=dict(seed=s,task=t,unit_updates=3000000)
            for k in N.KEYS:row[k]=float(task[s,:,N.KEYS.index(k)].sum())
            taskrows.append(row)
    rows=[]
    for s in range(5):
        r=dict(seed=s,unit_updates=12000000)
        for k in N.KEYS:r[k]=float(allsum[s,:,N.KEYS.index(k)].sum())
        r['p_down']=r['cert_down']/12000000;r['p_up']=r['cert_up']/12000000
        for k in N.TRANSPORT_KEYS:
            for sign in ('positive','negative'):r[k+'_'+sign+'_per_update']=r[k+'_'+sign]/120000
        rows.append(r)
    m1=directional([r['p_down']-r['p_up'] for r in rows]);m2=certificate_status(sum(r['cert_down']+r['cert_up'] for r in rows),sum(r['certificate_violation'] for r in rows))
    m3=directional((early/(b*75*100)-late/((400-b)*75*100)).mean(1),'BOUNDARY_ENRICHED','LATER_ENRICHED') if b else dict(label='WINDOW_NOT_IDENTIFIED')
    result=dict(M1x=m1,M2x=m2,M3x=m3,window_sha256=sha(window_path),run_id=RUN,complete_sha256=sha(out/'complete.json'),limitations='observational sufficient condition on self movement; five independent seeds; no causal or net-total-sink inference')
    destination.mkdir(parents=True,exist_ok=True);put(destination/'verdict.json',result)
    E.H.write_csv(destination/'per_seed.csv',rows);E.H.write_csv(destination/'per_task.csv',taskrows)
    from analysis.cifar_ledger_0920.replay import save_npz
    save_npz(destination/'per_unit.npz',dict(sum=allsum,keys=np.array(N.KEYS),seeds=np.arange(5)))
    probability={'CROSS_TASK_SINK_CONDITION_HOLDS':.55,'UNRESOLVED':.40,'NOT_SUPPORTED':.05}
    preds=dict(author='Codex',Issa=None,M1_multiclass_brier=sum((p-int(m1['label']==k))**2 for k,p in probability.items()),window_brier=(.65-int(bool(b)))**2,M3_brier=(.75-int(m3['label']=='BOUNDARY_ENRICHED'))**2 if b else None,conf_brier=(.65-int(all(r['S_conf_sum']<0 for r in rows)))**2)
    put(destination/'predictions.json',preds)
    (destination/'summary.md').write_text(f"# A2\n\nM1x: {m1['label']}\n\nM2x: {m2}\n\nM3x: {m3['label']}\n\n"+result['limitations']+'\n')
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['calibrate','report']);ap.add_argument('--src',required=True);ap.add_argument('--window',required=True);ap.add_argument('--out');a=ap.parse_args()
    print(calibrate(a.src,a.window) if a.stage=='calibrate' else report(a.src,a.window,a.out))
if __name__=='__main__':main()

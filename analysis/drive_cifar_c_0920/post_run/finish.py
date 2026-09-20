"""Post-run descriptive supplements; does not change the frozen A2 decisions.

The registered report has already been executed after the calibration-window commit.
This module completes the old-parent comparison and signed transport tables.
"""
import csv,json,math
from fractions import Fraction
from pathlib import Path
import numpy as np
from src import drive_cifar_c_0920 as D
from analysis.drive_cifar_c_0920.report import shard
from analysis.cifar_ledger_0920.replay import save_npz


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def compare_parent(root):
    parent=D.ROOT/'results/relu_doors_0919'
    manifest=json.loads((parent/'backup_manifest.json').read_text())
    lookup={r.get('relative',r.get('source')):r for r in manifest['files']}
    current={0:D.load_pt(root/'run/audit/t1_e001.pt')['start']['P']}
    for t in (1,2):current[t]=D.load_pt(root/f'run/audit/t{t}_e400.pt')['end']['P']
    comparisons=[];inputs=[]
    for t in (0,1,2):
        for seed in range(10):
            rel=f'results/relu_doors_0919/C/snap/C_raw_seed{seed}/t{t:02}.npz'
            entry=lookup[rel];path=Path(entry['backup'])
            assert path.stat().st_size==entry['bytes'] and D.sha(path)==entry['sha256']
            inputs.append(dict(source=rel,sha256=entry['sha256']))
            with np.load(path,allow_pickle=False) as old:
                for i,key in enumerate(('W1','b1','W2','b2','W3','b3')):
                    a=current[t][i][seed].numpy();b=old[key]
                    assert a.shape==b.shape and a.dtype==b.dtype==np.float32
                    comparisons.append(dict(seed=seed,task=t,param=key,bit_equal=bool(np.array_equal(a.view(np.uint32),b.view(np.uint32))),max_abs_difference=float(np.max(np.abs(a.astype(float)-b.astype(float))))))
    oldrows={(int(r['seed']),int(r['task'])):r for r in csv.DictReader((parent/'C/per_task.csv').open())}
    newrows=[r for r in csv.DictReader((root/'run/per_task.csv').open()) if int(r['task'])<=2]
    rowdiff=[]
    for row in newrows:
        old=oldrows[int(row['seed']),int(row['task'])]
        for key in sorted(set(row)&set(old)):
            a,b=row[key],old[key]
            if a==b:continue
            try:
                af,bf=float(a),float(b)
                if af==bf or (math.isnan(af) and math.isnan(bf)):continue
            except ValueError:pass
            rowdiff.append(dict(seed=int(row['seed']),task=int(row['task']),column=key,current=a,parent=b))
    result=dict(role='REPORT_ONLY',parent_provenance_sha256=D.sha(parent/'C/provenance.json'),current_complete_sha256=D.sha(root/'run/complete.json'),snapshot_inputs=inputs,snapshot_comparisons=comparisons,all_snapshot_parameters_bit_equal=all(r['bit_equal'] for r in comparisons),common_host_columns_equal=not rowdiff,host_row_differences=rowdiff,meaning='Fresh trajectory; old-trajectory reuse is not assumed. Independent unchanged-host test qualification is the scientific gate.')
    D.put(root/'parent_comparison.json',result)
    return dict(parameters_exact=result['all_snapshot_parameters_bit_equal'],common_columns_exact=result['common_host_columns_equal'],snapshot_tensor_comparisons=len(comparisons),row_differences=len(rowdiff))


def signed_windows(root):
    window=json.loads((root/'window_calibration.json').read_text());b=window['break_epoch'];windows=['full']+(['boundary','later'] if b else [])
    a=np.zeros((5,4,len(windows),100,len(D.N.KEYS)))
    for t in range(2,6):
        for e in range(1,401):
            values=shard(root/'run','primary',t,e,list(range(5)))
            a[:,t-2,0]+=values
            if b:a[:,t-2,1 if e<=b else 2]+=values
    step_counts={'full':30000,'boundary':75*b if b else None,'later':75*(400-b) if b else None}
    rows=[]
    for seed in range(5):
        for ti,task in [(None,'all_tasks2to5')]+[(i,str(i+2)) for i in range(4)]:
            for wi,win in enumerate(windows):
                v=a[seed,:,wi].sum(0) if ti is None else a[seed,ti,wi]
                steps=step_counts[win]*(4 if ti is None else 1);den=steps*100
                row=dict(seed=seed,task=task,window=win,model_updates=steps,units=100,unit_updates=den)
                for key in D.N.COUNT_KEYS:
                    count=float(v[:,D.N.KEYS.index(key)].sum());row[key+'_count']=int(count);row[key+'_frequency']=count/den
                for key in D.N.TRANSPORT_KEYS:
                    for suffix in ('sum','positive','negative'):
                        total=float(v[:,D.N.KEYS.index(key+'_'+suffix)].sum())
                        row[key+'_'+suffix]=total;row[key+'_'+suffix+'_per_model_update']=total/steps
                    for sign in ('positive','negative'):
                        row[key+'_'+sign+'_frequency']=float(v[:,D.N.KEYS.index(key+'_'+sign+'_count')].sum())/den
                rows.append(row)
    write_csv(root/'report/transport_by_window.csv',rows)
    save_npz(root/'report/transport_by_window_unit.npz',dict(sum=a,seeds=np.arange(5),tasks=np.arange(2,6),windows=np.array(windows),keys=np.array(D.N.KEYS)))
    # Derive M1/M3 again from the saved integer counts; does not choose or change the window.
    from analysis.drive_cifar_c_0920.stats import directional
    full=[r for r in rows if r['task']=='all_tasks2to5' and r['window']=='full']
    m1=directional([r['cert_down_frequency']-r['cert_up_frequency'] for r in full])
    saved=json.loads((root/'report/verdict.json').read_text());assert m1==saved['M1x']
    if b:
        lookup={(r['seed'],r['task'],r['window']):r for r in rows}
        differences=[]
        for seed in range(5):
            exact=sum((Fraction(lookup[seed,str(t),'boundary']['cert_down_count'],lookup[seed,str(t),'boundary']['unit_updates'])-Fraction(lookup[seed,str(t),'later']['cert_down_count'],lookup[seed,str(t),'later']['unit_updates']) for t in range(2,6)),Fraction(0))/4
            differences.append(float(exact))
            scale=sum(abs(lookup[seed,str(t),'boundary']['cert_down_frequency'])+abs(lookup[seed,str(t),'later']['cert_down_frequency']) for t in range(2,6))/4
            assert abs(float(exact)-saved['M3x']['differences'][seed])<=D.N.gamma(16)*scale
        reference=directional(differences,'BOUNDARY_ENRICHED','LATER_ENRICHED')
        for key in ('label','positive','nonzero','one_sided_p'):assert reference[key]==saved['M3x'][key]
    return dict(transport_rows=len(rows),M1_independent_count_aggregation='PASS',M3_independent_count_aggregation='PASS' if b else 'WINDOW_NOT_IDENTIFIED')


if __name__=='__main__':
    root=D.ROOT/'results/drive_cifar_c_0920'
    assert (root/'report/verdict.json').exists()
    checks=dict(parent=compare_parent(root),transport=signed_windows(root),script_sha256=D.sha(Path(__file__)))
    D.put(root/'post_run_checks.json',checks);print(json.dumps(checks,indent=2))

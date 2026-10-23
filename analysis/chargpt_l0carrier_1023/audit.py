#!/usr/bin/env python3
"""Independent NumPy checks of saved arrays, masks, medians, and telescoping."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/chargpt_l0carrier_1023'
TASKS=list(range(1,11))+[25]+list(range(39,51))+list(range(89,101))
MAX=0.


def close(a,b):
    global MAX
    err=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));MAX=max(MAX,err);assert err<1e-10,err


def main():
    states=list(csv.DictReader((OUT/'states.csv').open()));assert len(states)==1400
    index={(r['arm'],int(r['seed']),int(r['task']),r['probe']):r for r in states};assert len(index)==1400
    windows={(r['arm'],int(r['seed']),r['probe'],r['window']):r for r in csv.DictReader((OUT/'windows.csv').open())}
    for arm in ['C3','ReLUcap']:
        for seed in range(10):
            name=f'{arm}_seed{seed}.npz';path=OUT/'raw'/name
            if not path.exists():
                manifest=json.loads((OUT/'backup_manifest.json').read_text())
                r=next(r for r in manifest['files'] if Path(r['backup']).name==name);path=Path(r['backup'])
                assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
            with np.load(path) as a:
                cohort=a['cohort'];assert np.array_equal(cohort,a['t10_uniform_alive'])
                for probe in ['uniform','chain']:
                    for task in TASKS:
                        p=f't{task}_{probe}_';row=index[arm,seed,task,probe]
                        close(a[p+'r'],a[p+'A']+a[p+'B']);close(a[p+'B'],0)
                        close(np.linalg.norm(a[p+'m']),float(row['m_norm']))
                        for k in ['w_norm','c','sigma_hat','mu','s','r','A','B','b','p']:
                            close(np.median(a[p+k][cohort]),float(row[f'cohort_{k}_med']))
                            close(np.median(a[p+k]),float(row[f'all_{k}_med']))
                    sums={k:np.zeros(512) for k in ['w','input','cross','bias','width','stretch','rotate','total']}
                    for lo,hi in zip(TASKS,TASKS[1:]):
                        p=f'p{lo}_{hi}_{probe}_';terms={k:a[p+'dr_'+k] for k in sums}
                        close(terms['total'],terms['w']+terms['input']+terms['cross']+terms['bias']+terms['width'])
                        close(terms['w'],terms['stretch']+terms['rotate'])
                        close(terms['total'],a[f't{hi}_{probe}_r']-a[f't{lo}_{probe}_r'])
                        if lo>=39:
                            for k,v in terms.items():sums[k]+=v
                    close(sums['total'],a[f't100_{probe}_r']-a[f't39_{probe}_r'])
                    row=windows[arm,seed,probe,'cumulative39_100']
                    for k,v in sums.items():close(np.median(v[cohort]),float(row[f'cohort_dr_{k}_med']))
                    close(np.median(np.abs(sums['total'][cohort])),float(row['cohort_abs_cumulative_dr_med']))
                for t in range(90,101):
                    close(a[f'head_t{t}_dr_total']+a[f'body_t{t}_dr_total'],a[f't{t}_uniform_r']-a[f't{t-1}_uniform_r'])
    ledger=list(csv.DictReader((OUT/'ledger.csv').open()));assert len(ledger)==1360
    for r in ledger:
        for group in ['all','cohort']:
            close(float(r[f'{group}_dr_total_mean']),sum(float(r[f'{group}_dr_{k}_mean']) for k in ['w','input','cross','bias','width']))
    result=dict(passed=True,state_probe_rows=1400,endpoint_pair_probe_rows=1360,u100_states=220,
        raw_files=20,maximum_absolute_residual=MAX,cohort_masks_and_state_medians=True,per_unit_and_mean_additivity=True,
        cumulative_before_median_verified=True,head_body_endpoint_identity=True,training=False)
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()

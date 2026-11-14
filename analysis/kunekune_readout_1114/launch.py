"""Bounded 12-worker native-CPU launcher; one log and completion record per job."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/kunekune_readout_1114'

def job(activation,seed):
    name=f'{activation}_s{seed}';start=time.monotonic()
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    cmd=[sys.executable,'-m','analysis.kunekune_readout_1114.run','--activation',activation,'--seed',str(seed),'--out',str(OUT/'raw')]
    with (OUT/'logs'/f'{name}.log').open('a') as log:
        p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    rec=dict(activation=activation,seed=seed,returncode=p.returncode,elapsed_s=time.monotonic()-start)
    (OUT/'logs'/f'{name}.resource.json').write_text(json.dumps(rec))
    return rec

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=12);a=ap.parse_args()
    assert 1<=a.workers<=12
    (OUT/'logs').mkdir(parents=True,exist_ok=True);start=time.monotonic();finished=[]
    jobs=[(act,seed) for seed in range(10) for act in ('ELU','R','KKA','KKA23','KKT1','SNA')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        fs={pool.submit(job,*j):j for j in jobs}
        for fut in concurrent.futures.as_completed(fs):
            rec=fut.result();finished.append(rec)
            print(json.dumps({'finished':len(finished),'total':len(jobs),**rec}),flush=True)
            tmp=OUT/'launch_progress.tmp'
            tmp.write_text(json.dumps(dict(finished=finished,elapsed_s=time.monotonic()-start)))
            tmp.replace(OUT/'launch_progress.json')
            if rec['returncode']:
                for f in fs:f.cancel()
                raise RuntimeError(rec)
            if sum(r['elapsed_s'] for r in finished)>120000 or time.monotonic()-start>21600:
                for f in fs:f.cancel()
                raise RuntimeError('Registered resource ceiling; preserve partial results')
    (OUT/'launch_complete.json').write_text(json.dumps(dict(complete=True,finished=finished,elapsed_s=time.monotonic()-start)))

if __name__=='__main__':main()

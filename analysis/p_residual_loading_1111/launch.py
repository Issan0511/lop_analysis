"""Bounded launcher: four single-thread MLP jobs plus one ViT worker."""
import argparse, concurrent.futures as cf, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--only',choices=['all','mlp','vit'],default='all');a=ap.parse_args()
    dest=ROOT/'results/p_residual_loading_1111';logs=dest/'logs';logs.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    jobs=[]
    if a.only!='vit':
        for act in ('R','ELU','SILU'):
            for seed in range(10):jobs.append((f'mlp_{act}_S_s{seed}',['--model','mlp','--act',act,'--seeds',str(seed),'--tasks','1,10'],1800))
        for seed in range(10):jobs.append((f'mlp_ELU_Pall_s{seed}',['--model','mlp','--act','ELU','--history','Pall','--seeds',str(seed),'--tasks','29'],1800))
    vjobs=[] if a.only=='mlp' else [('vit_SL_S',['--model','vit','--seeds','10252010,10252011,10252012,10252013,10252014'],7200)]
    def run(job):
        name,args,limit=job;cmd=[sys.executable,'-m','analysis.p_residual_loading_1111.run',*args,'--out',str(dest/'raw')]
        st=time.monotonic()
        with (logs/(name+'.log')).open('w') as f:
            r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=limit)
        result=dict(job=name,returncode=r.returncode,elapsed_s=time.monotonic()-st,command=cmd)
        print(json.dumps(result),flush=True)
        if r.returncode:raise RuntimeError(f'Failed {name}, see log')
        return result
    start=time.monotonic();out=[]
    with cf.ThreadPoolExecutor(max_workers=4) as cp,cf.ThreadPoolExecutor(max_workers=1) as gp:
        futures=[cp.submit(run,j) for j in jobs]+[gp.submit(run,j) for j in vjobs]
        for f in cf.as_completed(futures):out.append(f.result())
    (dest/f'launch_{a.only}.json').write_text(json.dumps(dict(complete=True,jobs=out,elapsed_s=time.monotonic()-start),indent=2))
    print('ALL COMPLETED',a.only,len(out),time.monotonic()-start,flush=True)
if __name__=='__main__':main()

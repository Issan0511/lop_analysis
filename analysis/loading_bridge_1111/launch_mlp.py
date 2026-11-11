import concurrent.futures,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PY='/home/issan/Projects/claude/proj_004_drift/.venv/bin/python'
OUT=ROOT/'results/loading_bridge_1111/raw/dynamic_mlp'
LOG=ROOT/'results/loading_bridge_1111/raw/dynamic_logs'
LOG.mkdir(parents=True,exist_ok=True)
def run(job):
    act,seed=job
    cmd=[PY,'-u','analysis/loading_bridge_1111/dynamic.py','--model','mlp','--act',act,'--seeds',str(seed),'--tasks','1,10','--steps','200','--recovery','--natural','--out',str(OUT)]
    t=time.time()
    with open(LOG/f'{act}_s{seed}.log','w') as f:
        p=subprocess.run(cmd,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=f,stderr=subprocess.STDOUT)
    r=dict(act=act,seed=seed,returncode=p.returncode,elapsed_s=time.time()-t,cmd=cmd)
    print(json.dumps(r),flush=True)
    return r
jobs=[(act,s) for act in ('R','SILU','ELU') for s in range(10)]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    res=list(ex.map(run,jobs))
(LOG/'launch_summary.json').write_text(json.dumps(res,indent=1))
assert all(r['returncode']==0 for r in res),res

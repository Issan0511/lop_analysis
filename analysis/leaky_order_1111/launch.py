#!/usr/bin/env python3
"""Event-driven launch of the registered leak/time grid."""
import concurrent.futures,json,os,subprocess,sys,time
import torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src import leaky_order_1111 as O
PY='/home/issan/Projects/claude/proj_004_drift/.venv/bin/python';OUT=O.OUT
SPEC_COMMIT='361f98fb6fa99ab5aef5bd1745339a48e6d67fd8'
def job(args,out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'done.json').exists():raise RuntimeError(f'refusing implicit reuse: {out}')
    with (out/'stdout.log').open('w') as f:
        r=subprocess.run([PY,'-u','src/leaky_order_1111.py',*args,'--out',str(out)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    if r.returncode:raise RuntimeError(f'failed {r.returncode}: {out}/stdout.log')
    assert (out/'done.json').exists();return str(out.relative_to(ROOT))+' done'
def batch(jobs):
    mem=next(int(s.split()[1])/2**20 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    slots=min(5,max(0,int((mem-6)/1.2)));assert slots>0,mem
    print('launch',len(jobs),'jobs',slots,'slots',mem,'GiB available',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=slots) as ex:
        futures=[ex.submit(job,*j) for j in jobs]
        for future in concurrent.futures.as_completed(futures):print(future.result(),flush=True)
def main():
    t0=time.time();code=ROOT/'src/leaky_order_1111.py'
    for fn in ['checks.json','host_check.json']:
        check=json.loads((OUT/fn).read_text());assert check['all_pass'] and check['code_sha256']==O.C.sha(code)
    assert not subprocess.check_output(['git','status','--porcelain','--','src/leaky_order_1111.py','analysis/leaky_order_1111','specs/spec_leaky_order_1111.md'],cwd=ROOT,text=True).strip()
    subprocess.run(['git','merge-base','--is-ancestor',SPEC_COMMIT,'HEAD'],cwd=ROOT,check=True)
    manifest=ROOT/'results/leaky_boundary_1111/backup_manifest.json';raw=json.loads(manifest.read_text())
    files=raw['files'];sources={};records=[]
    for seed in range(3,8):
        for age in [0,3,6]:
            rel=f'results/leaky_boundary_1111/prefix/s{seed}/state_{O.B.tag(age)}.pt'
            m=next(r for r in files if r['source_rel']==rel);p=Path(m['backup']);assert O.C.sha(p)==m['sha256']
            sources[(.003,seed,age)]=p;records.append(dict(a=.003,seed=seed,age=age,path=str(p),sha256=m['sha256'],reused=True))
    batch([(['prefix','--a',str(a),'--seed',str(s)],OUT/'prefix'/O.tag(a)/f's{s}') for s in range(3,8) for a in [.001,.002,.005,.01]])
    for a in [.001,.002,.005,.01]:
        for seed in range(3,8):
            for age in [0,3,6]:
                p=OUT/'prefix'/O.tag(a)/f's{seed}'/f'source_age{age}.pt';sources[(a,seed,age)]=p
                records.append(dict(a=a,seed=seed,age=age,path=str(p),sha256=O.C.sha(p),reused=False))
    O.C.put(OUT/'source_manifest.json',dict(previous_manifest_sha256=O.C.sha(manifest),sources=records))
    for seed in range(3,8):
        for age in [0,3,6]:
            ref=torch.load(sources[(.003,seed,age)],weights_only=False)
            for a in [.001,.002,.005,.01]:
                ck=torch.load(sources[(a,seed,age)],weights_only=False)
                assert ck['tc']==ref['tc']
                assert all(torch.equal(ck[k],ref[k]) for k in ['g_lab','g_batch'])
                if age==0:assert all(torch.equal(p,q) for k in ['params','m','v'] for p,q in zip(ck[k],ref[k]))
    O.C.put(OUT/'paired_stream_checks.json',dict(all_pass=True,initial_parameters_and_moments_equal=True,labels_and_batch_rng_states_equal_at_all_source_ages=True))
    jobs=[]
    for age in [6,3,0]:
        for seed in range(3,8):
            for a in [.01,.005,.003,.002,.001]:
                jobs.append((['probe','--a',str(a),'--seed',str(seed),'--source',str(sources[(a,seed,age)])],OUT/'probes'/O.tag(a)/f's{seed}'/f'age{age}'))
    assert len(jobs)==75
    batch(jobs)
    O.C.put(OUT/'COMPLETE.json',dict(seconds=time.time()-t0,prefixes=20,probes=75,spec_commit=SPEC_COMMIT))
    print('ALL COMPLETE',time.time()-t0,flush=True)
if __name__=='__main__':main()

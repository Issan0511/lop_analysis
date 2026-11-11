#!/usr/bin/env python3
"""Event-driven, preregistered boundary experiment. No periodic status polling."""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src import leaky_boundary_1111 as B
PY='/home/issan/Projects/claude/proj_004_drift/.venv/bin/python';OUT=B.BASE
def job(script,args,out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'done.json').exists():raise RuntimeError(f'refusing implicit reuse: {out}')
    with (out/'stdout.log').open('w') as f:
        r=subprocess.run([PY,'-u',script,*args,'--out',str(out)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    if r.returncode:raise RuntimeError(f'failed {r.returncode}: {out}/stdout.log')
    assert (out/'done.json').exists();return str(out.relative_to(ROOT))+' done'
def batch(jobs,maximum=5):
    mem=next(int(s.split()[1])/2**20 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    n=min(maximum,max(0,int((mem-6)/1.2)));assert n>0,mem
    print('launch',len(jobs),'jobs',n,'slots',mem,'GiB available',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        fs=[ex.submit(job,*j) for j in jobs]
        for f in concurrent.futures.as_completed(fs):print(f.result(),flush=True)
def main():
    t0=time.time();check=json.loads((OUT/'checks.json').read_text());assert check['all_pass'] and check['code_sha256']==B.sha(ROOT/'src/leaky_boundary_1111.py')
    sc=json.loads((OUT/'spectral_checks.json').read_text());assert sc['all_pass'] and sc['code_sha256']==B.sha(ROOT/'analysis/leaky_boundary_1111/spectrum.py')
    assert not subprocess.check_output(['git','status','--porcelain','--','src/leaky_boundary_1111.py','analysis/leaky_boundary_1111','specs/spec_leaky_boundary_1111.md'],cwd=ROOT,text=True).strip()
    batch([('src/leaky_boundary_1111.py',['prefix','--seed',str(s)],OUT/'prefix'/f's{s}') for s in range(3,8)])
    sels={s:json.loads((OUT/'prefix'/f's{s}'/'selection.json').read_text()) for s in range(3,8)}
    B.put(OUT/'selection.json',sels);print('SELECTION',json.dumps(sels),flush=True)
    jobs=[];spectra=[]
    for seed,sel in sels.items():
        pre=sel['pre_age'];post=sel['post_age'];quarters=[pre+.25,pre+.5,pre+.75];ages=sorted(set(list(range(7))+quarters))
        for arm in ['N','G','L']:
            for age in (ages if arm=='N' else [pre,post]):
                steps=96000 if age in [pre,post] else 24000
                source=OUT/'prefix'/f's{seed}'/f'state_{B.tag(age)}.pt'
                args=['probe','--seed',str(seed),'--source',str(source),'--arm',arm,'--steps',str(steps)]
                jobs.append(('src/leaky_boundary_1111.py',args,OUT/'probes'/arm/f's{seed}'/f'age_{B.tag(age)}'))
        for age in sorted(set([0,6,pre,post]+quarters)):
            source=OUT/'prefix'/f's{seed}'/f'state_{B.tag(age)}.pt'
            spectra.append(('analysis/leaky_boundary_1111/spectrum.py',['run','--seed',str(seed),'--source',str(source)],OUT/'spectra'/f's{seed}'/f'age_{B.tag(age)}'))
    assert len(jobs)==70
    batch(jobs);batch(spectra,maximum=2)
    B.put(OUT/'COMPLETE.json',dict(seconds=time.time()-t0,prefixes=5,probes=len(jobs),spectra=len(spectra),selection=sels))
    print('ALL COMPLETE',time.time()-t0,flush=True)
if __name__=='__main__':main()

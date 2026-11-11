#!/usr/bin/env python3
"""Event-driven scheduler: no periodic progress polling; selection follows frozen spec."""
import concurrent.futures,csv,json,math,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src import leaky_cliff_1111 as C
BASE=C.BASE
PY='/home/issan/Projects/claude/proj_004_drift/.venv/bin/python'
def tag(a):return format(a,'.12g').replace('.','p')
def series_path(stage,a,seed):return BASE/stage/f'a{tag(a)}'/f's{seed}'
def job(args,out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'done.json').exists():
        old=json.loads((out/'done.json').read_text())['identity']
        assert old['code_sha256']==C.sha(ROOT/'src/leaky_cliff_1111.py')
        assert old['spec_sha256']==C.sha(C.SPEC)
        return str(out)+' already completed'
    cmd=[PY,'-u','src/leaky_cliff_1111.py',*args,'--out',str(out)]
    with (out/'stdout.log').open('w') as f:
        r=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    if r.returncode:raise RuntimeError(f'job failed {r.returncode}: {out}/stdout.log')
    assert (out/'done.json').exists()
    return str(out)+' completed'
def batch(jobs):
    mem=[int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')][0]/2**20
    slots=min(5,max(0,int((mem-6)/1.2)))
    if slots<1:raise RuntimeError(f'insufficient RAM: {mem:.2f} GiB')
    print(f'launch {len(jobs)} jobs slots={slots} available={mem:.1f} GiB',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=slots) as ex:
        futures=[ex.submit(job,args,out) for args,out in jobs]
        for f in concurrent.futures.as_completed(futures):print(f.result(),flush=True)
def sjobs(stage,aa,seeds):
    return [(['series','--a',str(a),'--seed',str(s)],series_path(stage,a,s)) for a in aa for s in seeds]
def readcsv(p):return list(csv.DictReader(p.open()))
def select():
    aa=[.003,.01,.03,.1,.3];means={}
    for a in aa:
        seedmeans=[]
        for s in range(3):
            rr=readcsv(series_path('discovery',a,s)/'tasks.csv')
            seedmeans.append(sum(float(r['online_acc']) for r in rr if 16<=int(r['task'])<=20)/5)
        means[str(a)]=sum(seedmeans)/3
    pairs=[(abs(means[str(b)]-means[str(a)]),a,b) for a,b in zip(aa[:-1],aa[1:])]
    gap,lo,hi=sorted(pairs,key=lambda x:(-x[0],x[1]))[0]
    bad,good=(lo,hi) if means[str(lo)]<means[str(hi)] else (hi,lo)
    sel=dict(bad=bad,good=good,low=lo,high=hi,discovery_gap=gap,late_means=means,midpoint=math.sqrt(lo*hi) if gap>=.15 else None)
    C.put(BASE/'selection.json',sel);print(json.dumps(sel),flush=True);return sel
def main():
    BASE.mkdir(parents=True,exist_ok=True);t0=time.time()
    checks=json.loads((BASE/'checks.json').read_text());assert checks['all_pass'] and checks['source_sha256']==C.sha(ROOT/'src/leaky_cliff_1111.py')
    assert not subprocess.check_output(['git','status','--porcelain','--','src/leaky_cliff_1111.py','analysis/leaky_cliff_1111'],cwd=ROOT,text=True).strip()
    batch(sjobs('discovery',[0.,.003,.01,.03,.1,.3,1.],range(3)))
    sel=select();aa=[sel['bad'],sel['good']]+([sel['midpoint']] if sel['midpoint'] is not None else [])
    batch(sjobs('confirmation',aa,range(3,8)))
    jobs=[]
    for s in range(3,8):
        for arm in ['N','A','B','X','O','H','BO','Fbad','Fgood','Ngood','Breverse','Xreverse']:
            rev=arm in ['Ngood','Breverse','Xreverse']
            src=sel['good'] if rev else sel['bad'];tar=sel['bad'] if rev else sel['good'];steps=6000 if rev else 24000
            args=['branch','--seed',str(s),'--a',str(tar),'--arm',arm,'--steps',str(steps),'--source',str(series_path('confirmation',src,s)/'state_t20.pt')]
            jobs.append((args,BASE/'branches'/arm/f's{s}'))
    batch(jobs)
    diffs=[]
    for s in range(3,8):
        rr={a:readcsv(BASE/'branches'/a/f's{s}'/'tasks.csv')[0] for a in ['B','N']}
        diffs.append(float(rr['B']['acc_6000'])-float(rr['N']['acc_6000']))
    C.put(BASE/'localization_selection.json',dict(B_minus_N=diffs,mean=sum(diffs)/5,run=sum(diffs)/5>=.15))
    if sum(diffs)/5>=.15:
        jobs=[]
        for s in range(3,8):
            for arm in ['B1','B2']:
                args=['branch','--seed',str(s),'--a',str(sel['good']),'--arm',arm,'--steps','6000','--source',str(series_path('confirmation',sel['bad'],s)/'state_t20.pt')]
                jobs.append((args,BASE/'branches'/arm/f's{s}'))
        batch(jobs)
    C.put(BASE/'COMPLETE.json',dict(seconds=time.time()-t0,selection=sel))
    print('all registered runs complete',time.time()-t0,flush=True)
if __name__=='__main__':main()

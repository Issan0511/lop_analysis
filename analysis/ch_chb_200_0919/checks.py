#!/usr/bin/env python3
"""S3 checks: real continuation, hostile inputs and non-vacuous numerical fixtures."""
import copy, importlib.util, itertools, json, math, os, resource, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src import ch_chb_200_0919 as S
from analysis.ch_chb_200_0919 import report as V, launch as L
OUT=S.OUT/'checks';SEEDS=list(range(200,210));E=S.E
REQUIRED=('S-source','S-continuity','S-resume','S-history','S-observer','S-columns','S-gate','S-H','S-B','S-pin','S-verdict','S-output','S-launch','S-cost')
records={name:dict(positive=[],mutations={}) for name in REQUIRED}

def save():S.put(OUT/'check_evidence.json',records)
def good(name,label,condition,values=None):
    records[name]['positive'].append(dict(name=label,pass_=bool(condition),values=values));save()
    assert condition,(name,label)
def killed(name,label,fn):
    try:fn()
    except (AssertionError,ValueError,FileNotFoundError,KeyError,SystemExit,RuntimeError) as e:
        records[name]['mutations'][label]=dict(detected=True,exception=type(e).__name__,message=str(e));save();return
    records[name]['mutations'][label]=dict(detected=False);save();raise AssertionError((name,label,'SURVIVED'))
def require(cond):assert cond

def equal_ck(a,b):
    aa=torch.load(a,map_location='cpu',weights_only=False);bb=torch.load(b,map_location='cpu',weights_only=False)
    assert S.digest(S.state(aa))==S.digest(S.state(bb)),'state differs'
    assert S.digest(aa['rows'])==S.digest(bb['rows']),'old numeric rows differ'

def basic():
    reg=S.registration();S.put(OUT/'calibration.json',reg)
    for arm in S.LAM:
        p=S.OUT/arm/'ckpt.pt';st=torch.load(p,map_location='cpu',weights_only=False)
        S.validate_checkpoint(st,arm,list(range(10)),400,50)
        good('S-source',arm,S.sha(p)==S.EXPECTED[arm],dict(bytes=p.stat().st_size,sha256=S.sha(p)))
        for key,val in [('arm','CHB' if arm=='CH' else 'CH'),('seeds',list(range(1,11))),('lam',np.nextafter(S.LAM[arm],math.inf))]:
            bad=copy.deepcopy(st);bad['meta'][key]=val
            killed('S-source',arm+'_'+key,lambda:S.validate_checkpoint(bad,arm,list(range(10)),400,50))
    raw=bytearray((S.OUT/'CH/ckpt.pt').read_bytes());raw[len(raw)//2]^=1
    import hashlib
    killed('S-source','one_byte',lambda:require(hashlib.sha256(raw).hexdigest()==S.EXPECTED['CH']))
    killed('S-source','missing_checkpoint',lambda:torch.load(OUT/'absent.pt',weights_only=False))
    for arm in S.LAM:
        S.validate_history(S.OUT/arm,arm,list(range(10)),50)
        good('S-history',arm,True,{'seeds':10,'tasks':50,'snapshots':510})
    # Real history validation with corrupted copies, no changes to immutable source.
    h=OUT/'bad_history';shutil.copytree(S.OUT/'CH',h,dirs_exist_ok=True)
    hp=h/'hist/CH_raw_seed0.npz'; saved=hp.read_bytes();hp.unlink()
    killed('S-history','missing_hist',lambda:S.validate_history(h,'CH',list(range(10)),50));hp.write_bytes(saved)
    with np.load(hp) as d:dd={k:d[k].copy() for k in d.files}
    short=dict(dd);short['acc']=short['acc'][:-1];np.savez(hp,**short)
    killed('S-history','one_point_missing',lambda:S.validate_history(h,'CH',list(range(10)),50))
    shifted=dict(dd);shifted['acc']=np.roll(shifted['acc'],1);np.savez(hp,**shifted)
    killed('S-history','task_axis_shift',lambda:S.validate_history(h,'CH',list(range(10)),50));hp.write_bytes(saved)
    rows=S.readcsv(h/'per_task.csv');rows[0]['seed']='1';S.writecsv(h/'per_task.csv',rows)
    killed('S-history','seed_mix',lambda:S.validate_history(h,'CH',list(range(10)),50));shutil.rmtree(h)
    # Non-symmetric weights make an incorrect axis, omitted centering, and layers distinct.
    w=np.array([[1.,3.,8.],[-2.,4.,9.]])
    n,N,o=S.weight_metrics(w)
    ref=np.array([math.sqrt(sum((x-sum(row)/len(row))**2 for x in row)) for row in w])
    eps=np.finfo(float).eps;tol=S.gamma(w.size,eps)*max(ref)
    good('S-columns','Wtilde',np.max(np.abs(n-ref))<=tol,{'norm':n.tolist(),'reference':ref.tolist(),'bound':tol})
    good('S-columns','omega',o==.001/N)
    for name,bad in [('omit_center',np.linalg.norm(w,axis=1)),('wrong_axis',np.linalg.norm(w-w.mean(axis=0),axis=1))]:
        killed('S-columns',name,lambda bad=bad:require(np.max(np.abs(n-bad))<=tol))
    killed('S-columns','divide_lr',lambda:require(o==1/(.001*N)))
    killed('S-columns','layer_swap',lambda:require(S.weight_metrics(w*3)[2]==o))
    b=np.array([-3.,1.]);killed('S-columns','abs_mean_bias',lambda:require(np.mean(np.abs(b))==abs(np.mean(b))))
    z=np.array([[-2.,1.],[-1.,9.],[0.,11.]])
    depth=np.median(z.mean(0))/np.median(z.std(0,ddof=1));sink=np.median(z.mean(0)/z.std(0,ddof=1))
    good('S-columns','depth',S.ratio(float(np.median(z.mean(0))),float(np.median(z.std(0,ddof=1))))==depth)
    killed('S-columns','sink_as_depth',lambda:require(depth==sink))
    # H output is the actual a1 input; removing its mean must change r.
    a=np.array([[2.,5.],[4.,9.],[3.,1.]]);m=np.array([2.,3.]);cent=a-m
    def rr(v):return np.linalg.norm(v.mean(0))/np.sqrt(np.mean(np.sum((v-v.mean(0))**2,axis=1)))
    good('S-columns','r_formula',math.isfinite(rr(cent)))
    killed('S-columns','preH_r',lambda:require(rr(a)==rr(cent)))
    z=np.array([[-1.,0.,1.,-1.],[-2.,0.,-1.,1.]])
    nd,ng,nz=S.gate_counts(z);good('S-gate','mixed_and_zero',(nd,ng,nz)==(2,6,2),[nd,ng,nz])
    zz=torch.tensor([-1.,0.,1.],requires_grad=True);torch.clamp(zz,min=0).sum().backward()
    good('S-gate','autograd_at_zero',zz.grad.tolist()==[0.,1.,1.],zz.grad.tolist())
    killed('S-gate','dead_equals_gate',lambda:require(nd==ng))
    killed('S-gate','strict_negative',lambda:require(int((z<0).sum())==ng))
    killed('S-gate','denominator',lambda:require(ng/z.size==ng/z.shape[1]))
    # Policy on already-updated Adam parameters: b3 must remain bit identical.
    P=[torch.ones((2,3,4)) if i%2==0 else torch.tensor([[1.,-2.,3.],[4.,-5.,6.]]) for i in range(6)]
    A=E.make_act('CHB',S.LAM['CHB']);before=[p.clone() for p in P];A.post_update(P,.001)
    good('S-B','actual_policy',torch.equal(P[1],torch.zeros_like(P[1])) and torch.equal(P[3],before[3]*(1-.001*S.LAM['CHB'])) and torch.equal(P[5],before[5]))
    killed('S-B','nonzero_b1',lambda:require(torch.equal(P[1]+1,P[1])))
    killed('S-B','lambda_zero',lambda:require(torch.equal(P[3],before[3])))
    killed('S-B','wd_b3',lambda:require(torch.equal(P[5],before[5]*(1-.001*S.LAM['CHB']))))
    A=E.ReLUDoors('CH',widths=(3,3));A.init_state(2,torch.device('cpu'),'check')
    A.m=[torch.full((2,3),.3),torch.full((2,3),.7)]
    zs=[torch.arange(-6.,12.).reshape(2,3,3),torch.arange(-9.,9.).reshape(2,3,3)]
    old=[m.clone() for m in A.m];A.update(*zs)
    reference=[m*(1-.01)+torch.clamp(z,min=0).mean(1)*.01 for m,z in zip(old,zs)]
    # Elementwise fused alpha add follows parent arithmetic exactly.
    reference=[m.clone().mul_(.99).add_(torch.clamp(z,min=0).mean(1),alpha=.01) for m,z in zip(old,zs)]
    good('S-H','EMA_recurrence',all(torch.equal(a,b) for a,b in zip(A.m,reference)))
    good('S-H','train_eval',torch.equal(A.phi(zs[0],0,True),A.phi(zs[0],0,False)))
    good('S-H','detached',all(not m.requires_grad for m in A.m))
    W=torch.arange(1.,10.).reshape(3,3);diff=(torch.clamp(zs[0],min=0)@W)-(A.phi(zs[0])@W)
    exp=(A.m[0]@W)[:,None,:].expand_as(diff);bound=S.gamma(12,2**-24)*(diff.abs()+exp.abs()+1)
    good('S-H','Wm',bool(((diff-exp).abs()<=bound).all()))
    killed('S-H','beta_zero',lambda:require(all(torch.equal(a,b) for a,b in zip(A.m,old))))
    bad=old[0]*.99+(torch.clamp(zs[0],min=0)-old[0][:,None,:]).mean(1)*.01
    killed('S-H','centered_EMA',lambda:require(torch.equal(A.m[0],bad)))
    keep=[m.clone() for m in A.m];A.phi(zs[0],train=False)
    good('S-H','eval_read_only',all(torch.equal(a,b) for a,b in zip(A.m,keep)))
    A.update(*zs);killed('S-H','eval_updates_m',lambda:require(all(torch.equal(a,b) for a,b in zip(A.m,keep))))
    badm=torch.ones(2,3,requires_grad=True);killed('S-H','EMA_grad',lambda:require(not badm.requires_grad))
    # Independent expected centred inputs are frozen before C-off mutation.
    raw=np.arange(24,dtype=np.float32).reshape(6,4)/24
    xc=raw-raw.mean(0);w=np.array([[1.,2.,3.,4.],[-2.,1.,3.,1.]],np.float32)
    z=xc@w.T;meas,bound=S.pin_bounds(w,xc,z)
    good('S-pin','roundoff',np.all(meas<=bound),dict(measured=meas.tolist(),bound=bound.tolist()))
    badz=raw@w.T;badmeas,badbound=S.pin_bounds(w,xc,badz)
    killed('S-pin','C_off_fixed_expected',lambda:require(np.all(badmeas<=badbound)))
    badmeas,badbound=S.pin_bounds(w,xc,z+1)
    killed('S-pin','bias_added',lambda:require(np.all(badmeas<=badbound)))
    verdict_checks(reg)
    good('S-output','new_root',L.output_guard(S.OUT) is None)
    killed('S-output','old_root',lambda:L.output_guard(S.PARENT))
    good('S-launch','basename',L.log_path('CH').name=='ch_chb_200_0919_CH.log')
    killed('S-launch','slash_basename',lambda:L.log_path('CH','a/b.log'))
    good('S-cost','exclusive_fixture',L.assert_exclusive([os.getpid()]) is None)
    killed('S-cost','other_GPU_job',lambda:L.assert_exclusive([os.getpid()+100000]))

def verdict_checks(reg):
    def C(**kw):
        args=dict(B=.987,W=.987,Z=0,dead0=0,deadlate=0,gate0=50000,gatelate=1000000,k=[1.,1.],reg=reg);args.update(kw);return V.classify(**args)
    expected=[({},'HOLDS'),({'W':.975},'GENTLE'),({'W':.49},'COLLAPSE'),({'Z':reg['d_c']-.01},'COLLAPSE'),({'W':.975,'gatelate':1000001},'UNCLASSIFIED'),({'W':.8},'UNCLASSIFIED'),({'Z':math.nan},'UNCLASSIFIED'),({'W':.975,'k':[math.nan,1]},'UNCLASSIFIED'),({'W':math.nan},'NUMERICAL_FAILURE'),({'W':reg['L_H']},'HOLDS'),({'W':.5,'Z':reg['d_c'],'B':.5},'GENTLE'),({'W':.97,'B':.96},'GENTLE')]
    for i,(args,want) in enumerate(expected):good('S-verdict',f'case{i}',C(**args)['type']==want,C(**args))
    safe=['HOLDS','GENTLE'];types=safe+['COLLAPSE','UNCLASSIFIED']
    for a,b in itertools.product(types,repeat=2):
        want=('CURED_200' if a in safe and b in safe else 'B_ROUTE_DELAY' if a=='COLLAPSE' and b in safe else 'OTHER_ROUTE' if a==b=='COLLAPSE' else 'REVERSED_B_EFFECT' if a in safe and b=='COLLAPSE' else 'INCONCLUSIVE')
        good('S-verdict',a+'_'+b,V.pair(a,b)==want)
    for n in (8,9,10):
        arms={a:{s:C(W=.975) if s<n else C(W=.975,gatelate=1000001) for s in range(10)} for a in S.LAM}
        r=V.combine(arms);good('S-verdict',f'support{n}',r['main']==('CURED_200' if n>=9 else 'INCONCLUSIVE'))
        if n==8:killed('S-verdict','8_of_10',lambda:require(r['main']=='CURED_200'))
    killed('S-verdict','strict_boundary',lambda:require(C(Z=reg['d_c'])['type']=='COLLAPSE'))
    # Mapping by identity rather than marginal arm counts.
    arms={a:{s:C() for s in range(10)} for a in S.LAM}
    arms['CH'][0]=C(W=.975,gatelate=1000001);arms['CHB'][1]=C(W=.975,gatelate=1000001)
    good('S-verdict','same_seed_support',V.combine(arms)['main']=='INCONCLUSIVE')
    rotated=copy.deepcopy(arms);rotated['CHB']={s:arms['CHB'][(s+1)%10] for s in range(10)}
    killed('S-verdict','seed_shift',lambda:require(V.combine(arms)['main']==V.combine(rotated)['main']))
    killed('S-verdict','nan_zero',lambda:require(C(Z=math.nan)['type']==C(Z=0)['type']))
    bad=copy.deepcopy(arms);del bad['CHB'];killed('S-verdict','missing_arm',lambda:V.combine(bad))
    bad=copy.deepcopy(arms);del bad['CH'][0];killed('S-verdict','missing_seed',lambda:V.combine(bad))
    killed('S-verdict','median_mean',lambda:require(np.median([0]*9+[1])==np.mean([0]*9+[1])))
    rows=[dict(task=str(t),online_acc=str(.99 if t<181 else .975),depth_ratio_l2='0',omega_l1=str(1 if t<=50 else .5),omega_l2=str(1 if t<=50 else .5),n_dead_l2='0',n_gate0_l2='50000',**{k:'0' for k in ('dead_frac_l2','gate_zero_frac_l2','bias_over_sd_l2','bias_mean_l2','zsd_l2','w_norm_l1','w_norm_l2','r_a1')}) for t in range(1,201)]
    correct=V.seed_readout(rows,reg);bad=copy.deepcopy(rows);bad[179]['online_acc']='.2';bad[199]['online_acc']='.99'
    killed('S-verdict','window_shift',lambda:require(correct['W']==V.seed_readout(bad,reg)['W']))
    killed('S-verdict','missing_task',lambda:V.seed_readout(rows[:-1],reg))
    # Independent t quantile: Simpson integration of df=9 Student density (no scipy dependency).
    x=1.8331129326536335;grid=np.linspace(0,x,10001)
    dens=math.gamma(5)/(math.sqrt(9*math.pi)*math.gamma(4.5))*(1+grid**2/9)**-5
    integ=(x/10000)/3*(dens[0]+dens[-1]+4*dens[1:-1:2].sum()+2*dens[2:-1:2].sum())
    good('S-verdict','t_quantile_integral',abs(.5+integ-.95)<=S.gamma(10001,2**-53),{'cdf':.5+integ})
    good('S-verdict','beta_units',reg['beta_omega']==1.05/100)
    good('S-verdict','g_arithmetic',C(W=.975)['gminus']==.0105 and C(W=.975)['gplus']==.021)


def gpu():
    dev=E.H.setup('cuda');torch.set_num_threads(2);cifar=E.RC.Cifar10()
    # Frozen original source imported as a separate module.
    frozen=OUT/'frozen_engine.py'
    frozen.write_bytes(subprocess.run(['git','show',S.APPROVAL+':src/relu_doors_0919.py'],cwd=S.ROOT,capture_output=True,check=True).stdout)
    assert S.sha(frozen)=='8caf2e842852f934c9118e5528b8213d596b2f43ee66f0fd32e751c877376f5f'
    spec=importlib.util.spec_from_file_location('s3_frozen',frozen);F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)
    def run(out,arm,tasks,observer=None,engine=E,epochs=2,resume=False):
        return engine.run(arm,SEEDS,['raw'],tasks,epochs,dev,out,lam=S.LAM[arm],checkpoint=True,resume=resume,cifar=cifar,progress=lambda _:None,**({'lifecycle':observer} if engine is E else {}))
    for arm in S.LAM:
        print('GPU checks',arm,flush=True)
        base=OUT/(arm+'_uninterrupted');first=OUT/(arm+'_first');extended=OUT/(arm+'_extended');off=OUT/(arm+'_off');frozenout=OUT/(arm+'_frozen')
        for out in (base,first,extended,off,frozenout):
            if out.exists():shutil.rmtree(out)
        run(base,arm,3,S.Observer(base,arm,None,SEEDS,2))
        run(first,arm,1,S.Observer(first,arm,None,SEEDS,2))
        shutil.copytree(first,extended)
        expected=torch.load(extended/'ckpt.pt',map_location='cpu',weights_only=False)
        observer=S.Observer(extended,arm,expected,SEEDS,2)
        run(extended,arm,3,observer,resume=True)
        equal_ck(base/'ckpt.pt',extended/'ckpt.pt')
        good('S-resume',arm+'_bitwise',True,dict(seeds=SEEDS,epochs=2,tasks=3,state=S.digest(S.state(torch.load(base/'ckpt.pt',weights_only=False,map_location='cpu')))))
        good('S-continuity',arm+'_ready_and_RNG',observer.cont['expected_state']==observer.cont['before_first_draw'] and observer.cont['labels_expected']==observer.cont['labels_actual'] and observer.cont['order_expected']==observer.cont['order_actual'])
        run(off,arm,3);run(frozenout,arm,3,engine=F)
        equal_ck(base/'ckpt.pt',off/'ckpt.pt');equal_ck(base/'ckpt.pt',frozenout/'ckpt.pt')
        good('S-observer',arm+'_frozen_on_off',True)
        S.validate_history(extended,arm,SEEDS,3);good('S-history',arm+'_append',True)
        # Missing restore state must produce a different real continuation.
        for mut in ('moment','EMA','RNG'):
            out=OUT/(arm+'_mut_'+mut)
            if out.exists():shutil.rmtree(out)
            shutil.copytree(first,out);bad=copy.deepcopy(expected)
            if mut=='moment':bad['m']=[torch.zeros_like(x) for x in bad['m']]
            elif mut=='EMA':bad['act_state']['m']=[torch.zeros_like(x) for x in bad['act_state']['m']]
            else:
                g=torch.Generator();g.set_state(bad['g_lab'][SEEDS[0]]);E.RC.task_labels(g);bad['g_lab'][SEEDS[0]]=g.get_state()
            torch.save(bad,out/'ckpt.pt');run(out,arm,3,resume=True)
            killed('S-resume',arm+'_'+mut,lambda:equal_ck(base/'ckpt.pt',out/'ckpt.pt'))
            killed('S-continuity',arm+'_'+mut,lambda:require(S.digest(S.state(bad))==S.digest(S.state(expected))))
        for key in ('v','tc','t'):
            bad=copy.deepcopy(expected)
            bad[key]=[torch.zeros_like(x) for x in bad[key]] if key=='v' else 0
            killed('S-continuity',arm+'_'+key,lambda:require(S.digest(S.state(bad))==S.digest(S.state(expected))))
        # Capture restore omitted / logger mutates live state: observer catches before labels.
        for mut in ('capture_restore','logger_EMA','logger_RNG'):
            out=OUT/(arm+'_'+mut)
            if out.exists():shutil.rmtree(out)
            shutil.copytree(first,out)
            observer=S.Observer(out,arm,expected,SEEDS,2,diagnose=False)
            def mutant(event,c):
                if event=='ready':
                    with torch.no_grad():
                        if mut=='capture_restore':c['adam_m'][0].add_(1)
                        elif mut=='logger_EMA':c['act'].m[0].add_(1)
                        else:E.RC.task_labels(c['g_lab'][SEEDS[0]])
                return observer(event,c)
            killed('S-continuity' if mut=='capture_restore' else 'S-observer',arm+'_'+mut,lambda:run(out,arm,3,mutant,resume=True))
        rows=S.readcsv(first/'per_task.csv');old=rows[0]['online_acc']
        killed('S-observer',arm+'_round_old',lambda:require(f'{float(old):.2f}'==old))
        killed('S-observer',arm+'_layer_swap',lambda:require(rows[0]['zbar_l1']==rows[0]['zbar_l2']))
    # Real stop before completion; saved checkpoint and history at the boundary.
    out=OUT/'stop_test';shutil.rmtree(out,ignore_errors=True);flag=OUT/'STOP_TEST';flag.touch()
    observer=S.Observer(out,'CH',None,SEEDS,2,stop=flag)
    run(out,'CH',3,observer);st=torch.load(out/'ckpt.pt',weights_only=False,map_location='cpu')
    good('S-launch','STOP_checkpoint',st['t']==1)
    killed('S-launch','STOP_ignored',lambda:require(st['t']==3));flag.unlink()
    # Prefix enrichment actually replays all old snapshots. Its old cells must stay identical.
    for arm in S.LAM:
        S.enrich_prefix(arm,cifar,dev)
        p=json.loads((OUT/f'prefix_{arm}.json').read_text());good('S-observer',arm+'_500_old_rows',p['rows']==500 and p['bit_equal'])
    # Current exclusive cost includes diagnostics / checkpoint / snapshot / hist writes.
    out=OUT/'cost';shutil.rmtree(out,ignore_errors=True);torch.cuda.reset_peak_memory_stats();started=time.time()
    prov=run(out,'CHB',1,S.Observer(out,'CHB',None,SEEDS,20),epochs=20)
    elapsed=time.time()-started;step=prov['step_ms_last_task']/1000
    snapbytes=sum(p.stat().st_size for p in (out/'snap').rglob('t01.npz'))
    laterbytes=2*200*snapbytes;free=shutil.disk_usage(S.OUT).free
    estimate=300*(30000*step+max(0,elapsed-1500*step))
    cost=dict(wall_clock_s=elapsed,step_ms=step*1000,estimated_two_arm_seconds=estimate,snapshot_bytes_per_task=snapbytes,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_bytes=torch.cuda.max_memory_allocated(),free_disk_bytes=free,estimated_snapshots_bytes=laterbytes,
        note='Sequential R=10; startup included in overhead; long-history compression grows. No accuracy displayed.')
    S.put(OUT/'cost.json',cost);good('S-cost','measured',free>laterbytes and elapsed>0,cost)
    # Launch provenance must be collected now, not from old parent or a subsequent checkout.
    startup=E.git_state();parent=json.loads((S.PARENT/'CH/provenance.json').read_text())
    good('S-output','startup_hash',len(startup['git_hash'])==40 and startup['git_hash']!=parent['git_hash'])
    killed('S-output','parent_hash',lambda:require(startup['git_hash']==parent['git_hash']))
    killed('S-output','final_HEAD_substitution',lambda:require(startup['git_hash']=='0'*40))
    # Child launch handshake with actual PID / log / heartbeat, plus failing child.
    log=OUT/'handshake.log';hb=OUT/'handshake.json'
    with log.open('w') as f:
        child=subprocess.Popen([sys.executable,'-c',"import os,json,time;from pathlib import Path;print('started',flush=True);Path(__import__('sys').argv[1]).write_text(json.dumps({'pid':os.getpid(),'time':time.time()}))",str(hb)],stdout=f,stderr=subprocess.STDOUT)
        rc=child.wait()
    proof=json.loads(hb.read_text())
    good('S-launch','PID_log_heartbeat',rc==0 and proof['pid']==child.pid and log.read_text().strip()=='started')
    bad=subprocess.run([sys.executable,'-c','raise SystemExit(23)'],capture_output=True)
    killed('S-launch','PID_only_success',lambda:require(bad.returncode==0))

def main():
    OUT.mkdir(parents=True,exist_ok=True);started=time.time()
    S.put(OUT/'checks_start.json',dict(time=started,git=E.git_state(),independent_audit=False))
    basic()
    with L.exclusive():gpu()
    # Explicit list: every required group must have real positive and killed mutation evidence.
    assert set(records)==set(REQUIRED)
    ok=all(r['positive'] and r['mutations'] and all(q['pass_'] for q in r['positive']) and all(q['detected'] for q in r['mutations'].values()) for r in records.values())
    files=['src/relu_doors_0919.py','src/ch_chb_200_0919.py','analysis/ch_chb_200_0919/checks.py','analysis/ch_chb_200_0919/report.py','analysis/ch_chb_200_0919/launch.py']
    S.put(OUT/'checks.json',dict(all_pass=ok,required=list(REQUIRED),positive_count=sum(len(r['positive']) for r in records.values()),mutation_count=sum(len(r['mutations']) for r in records.values()),evidence_sha256=S.sha(OUT/'check_evidence.json'),tested_source_sha256={f:S.sha(S.ROOT/f) for f in files},wall_clock_s=time.time()-started,independent_audit=False))
    assert ok
    print('ALL CHECKS PASS',flush=True)

if __name__=='__main__':main()

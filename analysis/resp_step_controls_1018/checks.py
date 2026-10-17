#!/usr/bin/env python3
"""CPU semantics + adversarial source mutants; CUDA evidence is distinct, never inferred."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import math
import sys
import time
import types
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src import resp_step_controls_1018 as R
from analysis.resp_step_controls_1018 import verdict as V


def mutant(module,old,new):
    p=Path(module.__file__);src=p.read_text();assert src.count(old)==1,(old,src.count(old))
    name='_mutation_'+str(time.monotonic_ns());m=types.ModuleType(name);m.__file__=str(p);sys.modules[name]=m
    exec(compile(src.replace(old,new),str(p),'exec'),m.__dict__);return m


def clone_state(params,adam):return [p.detach().clone().requires_grad_(True) for p in params],([q.clone() for q in adam[0]],[q.clone() for q in adam[1]],[adam[2][0]])


def fixture(seed=100):
    gen=torch.Generator().manual_seed(seed)
    # Fixture uses genuine native dimensionality; production data is exercised separately.
    x=torch.randn(1200,784,generator=gen);y=torch.randint(10,(1200,),generator=gen)
    p=R.RM.H.init_params(seed,torch.device('cpu'))
    adam=([torch.randn(q.shape,generator=gen)*.01 for q in p],[torch.rand(q.shape,generator=gen)*.01+.001 for q in p],[37])
    return x,y,p,adam


def host_check(m=R):
    x,y,p,adam=fixture();p2,a2=clone_state(p,adam)
    gb=R.RM.H.stream('rl_batch',100);gb2=R.RM.gen_from(gb.get_state())
    old=R.RM.train_task(p,adam,x,y,gb,1)
    trace=[];rng_before=torch.get_rng_state().clone();new=m.rm_train(p2,a2,x,y,gb2,1,trace=trace,seed=100)
    assert torch.equal(rng_before,torch.get_rng_state()),'logger consumed RNG'
    assert R.RC.tree_hash((p,adam,gb.get_state()))==R.RC.tree_hash((p2,a2,gb2.get_state())),'native state differs'
    assert torch.equal(old[0],new[0]) and torch.equal(old[1],new[1]) and torch.equal(old[2],new[2])
    # logger is read-only, preserves RNG and exact path; q=1 is the same path.
    p3,a3=clone_state(fixture()[2],fixture()[3]);gb3=R.RM.H.stream('rl_batch',100)
    m.rm_train(p3,a3,x,y,gb3,1,trace=None,seed=100)
    assert R.RC.tree_hash((p2,a2,gb2.get_state()))==R.RC.tree_hash((p3,a3,gb3.get_state())),'logger affects path'
    assert len(trace)==75*6
    # lr must change real Adam displacement, rather than merely reported lr.
    p4,a4=clone_state(fixture()[2],fixture()[3]);gb4=R.RM.H.stream('rl_batch',100)
    m.rm_train(p4,a4,x,y,gb4,1,rates=(.002,.0005,.003))
    assert R.RC.tree_hash(p3)!=R.RC.tree_hash(p4),'q != 1 changed only metadata'
    # Independent continuation using the native operation helper; changing grads rather
    # than real LR cannot reproduce this nonzero-moment trajectory.
    p5,a5=clone_state(fixture()[2],fixture()[3]);gb5=R.RM.H.stream('rl_batch',100)
    order=torch.randperm(1200,generator=gb5);xs,ys=x[order],y[order]
    for j in range(75):
        out=R.RM.EL.forward2(p5,xs[j*16:(j+1)*16],R.RM.ACT,R.RM.ACT)
        loss=F.cross_entropy(out[-1],ys[j*16:(j+1)*16]);grads=torch.autograd.grad(loss,p5);a5[2][0]+=1
        R.native_update(p5,grads,a5[0],a5[1],a5[2][0],(.002,.0005,.003),'RM')
    assert R.RC.tree_hash((p4,a4,gb4.get_state()))==R.RC.tree_hash((p5,a5,gb5.get_state())), 'rates must change actual Adam LR, not gradients'

    return dict(updates=75,nonzero_initial_moments=True)


def log_check(m=R):
    gen=torch.Generator().manual_seed(100)
    P=[torch.randn(shape,generator=gen,dtype=torch.float64) for shape in ((3,4),(3,),(2,3),(2,),(4,2),(4,))]
    grads=[torch.randn(p.shape,generator=gen,dtype=p.dtype) for p in P]
    mom=[torch.randn(p.shape,generator=gen,dtype=p.dtype) for p in P];var=[torch.rand(p.shape,generator=gen,dtype=p.dtype)+.01 for p in P]
    init=R.RC.tree_hash((P,mom,var));rates=(.003,.001,.02);tc=57
    cap=m.probe_begin(P,grads,mom,var,tc,rates,'RM');assert init==R.RC.tree_hash((P,mom,var))
    # Independent scalar double reference, coordinate by coordinate, including old m/v.
    for i,c in enumerate(cap):
        for idx in np.ndindex(P[i].shape):
            g=float(grads[i][idx]);mm=.9*float(mom[i][idx])+.1*g;vv=.999*float(var[i][idx])+.001*g*g
            upd=rates[i//2]*(mm/(1-.9**tc))/(math.sqrt(vv/(1-.999**tc))+1e-8)
            assert math.isclose(float(c['upd'][idx]),upd,rel_tol=1e-12,abs_tol=1e-15)
    R.native_update(P,grads,mom,var,tc,rates,'RM')
    rr=m.probe_finish(P,mom,var,cap,[100],'N',11,1,rates)
    assert len(rr)==6 and all(r['adam_candidate_norm']>0 for r in rr)
    for i,row in enumerate(rr):assert row['grad_norm']==float(grads[i].norm()), 'S-log: wrong observation time/value'
    return dict(independent_scalar=True)


def norm_fixture():
    rows=[]
    for s in R.CAL_SEEDS:
        for a in ('N','R'):
            for k in range(1,257):
                for i,p in enumerate(R.PARAMS):
                    # Variable steps make mean(norm) and RMS diverge; high bias prevents W-only passing.
                    val=(1+k%7)*(i+1)*(1+s/1000)
                    q=(2 if i%2==0 else 5) if a=='R' else 1
                    rows.append(dict(seed=s,arm=a,step=k,param=p,grad_norm=val*q,adam_candidate_norm=val*(3 if a=='R' else 1),actual_delta_norm=val*(7 if a=='R' else 1)))
    return rows


def norm_check(m=R):
    rows=norm_fixture();out=m.q_from_rows(rows)
    vals={k:[] for k in ('qG','qU')}
    # independent numpy RMS [seed,arm,step,parameter] over W+b, with explicit axes
    arr=np.asarray([[r['grad_norm'],r['actual_delta_norm']] for r in rows],dtype=np.float64).reshape(10,2,256,6,2)
    for col,name in enumerate(('qG','qU')):
        for l in range(3):
            rms=np.sqrt((arr[:,:,:,l*2:l*2+2,col]**2).sum(axis=3).mean(axis=2))
            vals[name].append(float(np.median(rms[:,1]/rms[:,0])))
        assert np.allclose(out[name],vals[name],rtol=1e-14,atol=0),name
    # One-sided and both-sided exact zero handling must be explicit.
    zeros=copy.deepcopy(rows)
    for row in zeros:
        if row['param'] in ('W1','b1'):row['grad_norm']=0.
    z=m.q_from_rows(zeros);assert z['qG'][0]==1 and len(z['zero_over_zero'])==10
    for row in zeros:
        if row['param']=='W1' and row['arm']=='R':row['grad_norm']=1.
    try:m.q_from_rows(zeros)
    except ValueError as e:assert 'CALIBRATION_UNDEFINED' in str(e)
    else:raise AssertionError('one-sided zero accepted')
    # Exact sub-ULP physical displacement is zero although Adam's candidate is nonzero.
    P=[torch.full((2,2) if i%2==0 else (2,),1e20,dtype=torch.float32) for i in range(6)]
    grads=[torch.ones_like(p) for p in P];mom=[torch.zeros_like(p) for p in P];var=[torch.zeros_like(p) for p in P]
    cap=R.probe_begin(P,grads,mom,var,1,(1e-3,)*3,'RM')
    R.native_update(P,grads,mom,var,1,(1e-3,)*3,'RM')
    actual=R.probe_finish(P,mom,var,cap,[100],'N',11,1,(1e-3,)*3)
    assert all(r['adam_candidate_norm']>0 and r['actual_delta_norm']==0 for r in actual)
    ulp=copy.deepcopy(rows)
    for r in ulp:r['actual_delta_norm']=0.
    q=m.q_from_rows(ulp)
    assert q['qU']==[1.,1.,1.] and len([z for z in q['zero_over_zero'] if z['metric']=='qU'])==30,'candidate mistaken for physical displacement'
    return dict(independent_numpy_float64=True,qG=out['qG'],qU=out['qU'],sub_ulp_candidate_positive_actual_zero=True)


def field_check(m=R.RM):
    x,y,p,adam=fixture();gen=torch.Generator().manual_seed(102)
    donor=[q.detach().clone()+torch.randn(q.shape,generator=gen)*.05 for q in p]
    with torch.no_grad():
        zb=R.RM.EL.forward2(p,x,R.RM.ACT,R.RM.ACT)[2]
        zs=R.RM.EL.forward2(donor,x,R.RM.ACT,R.RM.ACT)[2]
    d=(zs.double()-zb.double()).float();p0=[q.detach().clone() for q in p]
    forward=m.make_forward({'p0':p0,2:d})
    for ids in (torch.arange(16),torch.arange(16,32)):
        nat=R.RM.forward(p,x[ids],ids,None)[-1];out=forward(p,x[ids],ids)[-1]
        assert torch.equal(nat,out),'anchored branch equality'
        # image IDs must index field, gradient must be native expm1+1 through z+d.
        a1=R.RM.ACT.phi(x[ids]@p[0].T+p[1]);z2=a1@p[2].T+p[3]
        arg=z2+d[ids];target=zs[ids]
        assert torch.allclose(arg.double(),target.double(),rtol=0,atol=2e-6)
        grad=torch.autograd.grad(forward(p,x[ids],ids)[3].sum(),p[3],retain_graph=False)[0]
        expected=R.RM.dphi_train(arg).sum(0)
        assert torch.equal(grad,expected),'native field differential'
        wrong=R.RM.dphi_train(z2+d[ids.roll(1)]).sum(0)
        assert not torch.equal(grad,wrong),'image-id mutation escaped'
    # A whole-image cached z0 has different shape from the actual current batch and is rejected.
    def cached(P,xb,ids):
        z=x@p0[0].T+p0[1]
        return R.RM.ACT.phi(z) + R.RM.ACT.phi(xb@P[0].T+P[1])
    try:cached(p,x[:16],torch.arange(16))
    except RuntimeError:pass
    else:raise AssertionError('cached z0 mutation accepted')
    # Native ELU differences retained, deep negative RM expm1 derivative becomes exact zero.
    z=torch.tensor([-25.,-20.,-16.,-1.,0.,1.],requires_grad=True)
    g=torch.autograd.grad(R.RM.ACT.phi(z).sum(),z)[0]
    assert torch.equal(g,R.RM.dphi_train(z)) and g[0]==0 and R.RC.gtrain(z)[0]>0
    return dict(branch_batches=2,native_kernel_distinction=True)


def synth(box,control=.11,Rscore=.7,Nscore=.1):
    rows=[]
    for seed in range(10):
        for a in R.ARMS:
            score=Nscore if a=='N' else Rscore if a=='R' else control
            rows.append(dict(seed=seed,arm=a,task=29 if box=='F51' else 11,online_acc=score,finite=True,label_hash=str(seed),order_hash=str(seed),initial_state_sha256=str(seed)))
    return rows


def verdict_check(m=V):
    inp={b:synth(b) for b in R.BOXES}
    assert m.verdict(inp)['label']=='NOT_LR_ALL_THREE'
    all_lr={b:synth(b,control=.7) for b in R.BOXES}
    assert m.verdict(all_lr)['label']=='LR_EXPLAINS_ALL_THREE'
    mixed=dict(inp);mixed['RC']=all_lr['RC'];assert m.verdict(mixed)['label']=='MIXED_BY_BOX'
    assert m.verdict({'RM':inp['RM']})['status']=='INCOMPLETE'
    no_repro={b:synth(b,Rscore=.1) for b in R.BOXES};assert m.verdict(no_repro)['status']=='NOT_REPRODUCED'
    # Deliberately broad interval around C-R catches absence-of-significance = equivalence.
    broad={b:synth(b,control=.7) for b in R.BOXES}
    for rr in broad.values():
        for row in rr:
            if row['arm'].startswith('LR'):row['online_acc']+=.1 if row['seed']%2 else -.1
    assert m.verdict(broad)['boxes']['RM']['label']=='UNRESOLVED'
    e=m.interval(np.linspace(.1,.2,10));assert e['hi']-e['mean']>e['hi95']-e['mean'];assert m.M==15
    return dict(m=15,scenarios=6)


def schedule_check(m=R):
    assert m.semantic_status(ValueError('CALIBRATION_UNDEFINED: zero denominator'))=='CALIBRATION_UNDEFINED'
    # Gate mutation: evaluation seeds cannot supply a schedule and unaudited F51 always fails.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p=Path(d);rows=norm_fixture();R.put(p/'trace.json',rows);q=R.q_from_rows(rows)
        dataset=dict(archive_sha256={'fixture':'fixed'},train_tensor_sha256='fixture',transform='CHECK_ONLY')
        ci=dict(source_sha256=R.source_hashes(),phase='calibrate',box='RM',seeds=R.CAL_SEEDS,K=R.K,dataset_signature=dataset)
        schedule=dict(run=R.RUN,box='RM',**q,trace_sha256={'trace.json':R.sha(p/'trace.json')},identity=ci)
        R.put(p/'schedule.json',schedule);m.load_schedule(p/'schedule.json','RM',dataset)
        schedule.pop('identity');R.put(p/'schedule.json',schedule)
        try:m.load_schedule(p/'schedule.json','RM',dataset)
        except (AssertionError,KeyError):pass
        else:raise AssertionError('identity-less schedule accepted')
        schedule['identity']=ci;R.put(p/'schedule.json',schedule)
        try:m.load_schedule(p/'schedule.json','RM',dict(dataset,train_tensor_sha256='changed'))
        except AssertionError:pass
        else:raise AssertionError('changed dataset accepted by schedule')
        m.bind_manifest(p/'data_manifest.json',dataset)
        try:m.bind_manifest(p/'data_manifest.json',dict(dataset,train_tensor_sha256='changed'))
        except AssertionError:pass
        else:raise AssertionError('changed dataset overwrote resume manifest')
        schedule['seeds']=list(range(10));R.put(p/'schedule.json',schedule)
        try:m.load_schedule(p/'schedule.json','RM')
        except AssertionError:pass
        else:raise AssertionError('evaluation-derived calibration accepted')
        try:m.require_g0(None)
        except RuntimeError as e:assert 'BLOCKED_G0' in str(e)
        else:raise AssertionError('missing G0 accepted')
    return dict(eval_seeds_rejected=True,missing_g0_rejected=True)


def real_data_cost():
    R.RM.H.DATA_DIR=R.DATA_ROOT/'mnist';mnist=R.RM.H.Mnist(torch.device('cpu'))
    x=mnist.train_x[R.RM.RL.subset_idx(100)];p=R.RM.H.init_params(100,torch.device('cpu'))
    adam=([torch.zeros_like(q) for q in p],[torch.zeros_like(q) for q in p],[0]);gb=R.RM.H.stream('rl_batch',100);gl=R.RM.H.stream('rl_labels',100)
    t=time.monotonic();y=R.RM.RL.task_labels(gl);R.rm_train(p,adam,x,y,gb,1,trace=[])
    elapsed=time.monotonic()-t
    return dict(status='CHECK_ONLY',real_data_sha256=mnist.sha256,updates=75,seconds=elapsed,seconds_per_update=elapsed/75,
                full_RM_updates=20*(12+2*2)*6000+10*2*2*6000,
                historical_runtime_warning='First 256 updates incur detailed probes; prefix has no probes; current fixture includes all probes.')


def evidence_check(m=V):
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        d=Path(temporary)
        sources=R.source_hashes();data=dict(dataset_signature={'fixture':'CHECK_ONLY'},subset_or_plan_sha256='fixture')
        checks=dict(cpu_pass=True,source_sha256=sources)
        R.put(d/'admission_checks.json',checks)
        ident=dict(phase='evaluate',box='RM',seeds=R.EVAL_SEEDS,source_sha256=sources,
                   admission_checks_sha256=R.sha(d/'admission_checks.json'),**data)
        R.put(d/'provenance.json',ident);R.put(d/'data_manifest.json',data)
        for name in ('per_task.json','trace.json'):R.put(d/name,[])
        for name in ('per_task.csv','trace.csv','group_trace.csv'):(d/name).write_text('seed,online_acc\n0,0.5\n')
        for name in ('spec.md','common.md'):(d/name).write_text('CHECK_ONLY fixture; not production performance')
        hashes={p.name:R.sha(p) for p in d.iterdir()}
        proof=dict(status='COMPLETE',phase='evaluate',box='RM',identity=ident,output_sha256=hashes)
        R.put(d/'completion.json',proof);m.validate_bundle(d,'RM','evaluate')
        (d/'per_task.csv').write_text('seed,online_acc\n0,0.9\n')
        try:m.validate_bundle(d,'RM','evaluate')
        except ValueError:pass
        else:raise AssertionError('finite production CSV tampering accepted')
        (d/'per_task.csv').write_text('seed,online_acc\n0,0.5\n')
        proof.pop('identity');R.put(d/'completion.json',proof)
        try:m.validate_bundle(d,'RM','evaluate')
        except ValueError:pass
        else:raise AssertionError('completion without production identity accepted')
        (d/'completion.json').unlink()
        try:m.load_completed_evaluation(d/'per_task.json','RM')
        except FileNotFoundError:pass
        else:raise AssertionError('bare rows accepted as production evidence')
    return dict(finite_csv_tamper_rejected=True,missing_identity_rejected=True,bare_rows_rejected=True)


def worker_check(m=R):
    import concurrent.futures
    import multiprocessing
    seeds=[100,101]
    serial=[m.worker_equivalence_fixture(s) for s in seeds]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        parallel=list(pool.map(R.worker_equivalence_fixture,seeds))
    assert serial==parallel,'serial and spawned native seed trajectories differ'
    assert m.order_seed_shards(list(reversed(serial)),seeds)==serial
    for bad in (serial+[serial[0]],serial[:1]):
        try:m.order_seed_shards(bad,seeds)
        except AssertionError:pass
        else:raise AssertionError('missing/duplicate worker shard accepted')
    return dict(serial_spawn_bit_equal=True,missing_duplicate_shards_rejected=True,workers=2,threads_per_worker=1)


def cuda_suite():
    """Real CIFAR, R=10 native capture/rollback, nonzero moments, no-op and probes."""
    if not torch.cuda.is_available():return dict(pass_=False,status='BLOCKED_CUDA')
    import tempfile
    torch.set_num_threads(2);details={}
    with tempfile.TemporaryDirectory() as temporary:
        for box in ('RC','F51'):
            if box=='RC':
                dev=R.RC.setup();X,data=R.RC.make_inputs(R.CAL_SEEDS,dev)
                native=R.RC.Engine(R.CAL_SEEDS,X,graph=False)
                graph=R.RC.Engine(R.CAL_SEEDS,X,graph=True)
                core=R.RC.core_state
                labels=torch.stack([R.RC.E.RC.task_labels(native.g_lab[s]) for s in native.seeds]).to(dev)
                graph_labels=torch.stack([R.RC.E.RC.task_labels(graph.g_lab[s]) for s in graph.seeds]).to(dev)
                assert torch.equal(labels,graph_labels), 'CUDA fixture label streams differ'
                native.Y.copy_(labels);graph.Y.copy_(graph_labels)
                ids=torch.arange(16,device=dev)[None,:].expand(10,-1)
                for eng in (native,graph):
                    for j in range(3):eng.one_update(ids+j*16)
                assert R.RC.tree_hash(core(native.state()))==R.RC.tree_hash(core(graph.state())), 'CUDA graph warmup rollback mismatch'
                control=R.RCEngine(R.CAL_SEEDS,X,state=graph.state())
                capture,base=R.cuda_probe(control,box,ids,'N',1,1,Path(temporary)/box)
                control.one_update(ids);graph.one_update(ids)
            else:
                dev=R.F51.setup();data=R.F51.Data(dev)
                native=R.F51.Engine(R.CAL_SEEDS,data,graph=False)
                graph=R.F51.Engine(R.CAL_SEEDS,data,graph=True);core=lambda x:x
                taskrows,batches=native.task_batches(1)
                rows2,batches2=graph.task_batches(1);assert torch.equal(batches,batches2)
                # Use the first three true 5+1 task batches; do not patch production STEPS.
                for eng in (native,graph):eng.train_steps(batches[:,:3])
                assert R.RC.tree_hash(core(native.state()))==R.RC.tree_hash(core(graph.state())), 'CUDA graph warmup rollback mismatch'
                control=R.F51Engine(R.CAL_SEEDS,data,state=graph.state());ids=batches[:,3]
                capture,base=R.cuda_probe(control,box,ids,'N',1,1,Path(temporary)/box)
                for eng in (control,graph):
                    eng.static_idx.copy_(ids);eng.tc+=1;eng.inv_c1.fill_(1/(1-.9**eng.tc));eng.inv_c2.fill_(1/(1-.999**eng.tc));eng.cg.replay()
            assert R.RC.tree_hash(core(control.state()))==R.RC.tree_hash(core(graph.state())), 'CUDA q=1/logging/native mismatch'
            trace=R.probe_finish(control.P,control.m,control.v,capture,R.CAL_SEEDS,'N',1,1,control.rates)
            assert len(trace)==60 and all(r['m_norm']>0 for r in trace)
            details[box]=dict(R=10,real_data_sha256=data.sha256,graph_rollback=True,native_noop_probe_exact=True,nonzero_moments=True)
            del native,graph,control,data
            if box=='RC':del X
            torch.cuda.empty_cache()
    return dict(pass_=True,status='PASS',detail=details)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cuda',action='store_true');p.add_argument('--out',type=Path,default=ROOT/f'results/{R.RUN}/checks.json');a=p.parse_args()
    torch.set_num_threads(1);torch.set_flush_denormal(True);R.RM.H.setup('cpu')
    checks={};mutations=[]
    for name,fn in [('S-host/noop/logging',host_check),('S-log',log_check),('S-norm',norm_check),('S-host/field',field_check),('S-schedule',schedule_check),('S-verdict',verdict_check),('S-workers',worker_check),('S-readout-evidence',evidence_check)]:
        st=time.monotonic()
        try:detail=fn();checks[name]=dict(pass_=True,seconds=time.monotonic()-st,detail=detail)
        except Exception as e:checks[name]=dict(pass_=False,error=repr(e));raise
    cases=[
      ('logged gradient after Adam',R,log_check,"grad_norm=norm(c['g'])","grad_norm=norm(mi)"),
      ('logger consumes RNG',R,host_check,"record=trace is not None and step<=K","record=trace is not None and step<=K; torch.rand(()) if record else None"),
      ('backward doubled',R,host_check,"grads=torch.autograd.grad(loss,params)","grads=[2*g for g in torch.autograd.grad(loss,params)]"),
      ('field image IDs rolled',R.RM,field_check,"e2 = d2[ob]","e2 = d2[ob.roll(1)]"),
      ('cached whole-image z0',R.RM,field_check,"z02 = a01 @ p0[2].T + p0[3]","z02 = (a01 @ p0[2].T + p0[3])[:1].expand(N_IMG, -1)"),
      ('candidate norm substituted for actual U',R,norm_check,"('qU','actual_delta_norm')","('qU','adam_candidate_norm')"),
      ('moment reset',R,log_check,"mm=mi.clone().mul_(.9).add_(g,alpha=1-.9)","mm=torch.zeros_like(mi).add_(g,alpha=1-.9)"),
      ('mean not RMS',R,norm_check,"math.sqrt(sum(float(by[s,a,j,p][col])**2", "math.sqrt(sum(float(by[s,a,j,p][col])"),
      ('W-only ratio',R,norm_check,"for p in PARAMS[2*l:2*l+2])/k)","for p in PARAMS[2*l:2*l+1])/k)"),
      ('metadata-only lr',R,host_check,"upd=rates[i//2]*(mi/c1)/((vi/c2).sqrt()+1e-8);p-=upd","upd=1e-3*(mi/c1)/((vi/c2).sqrt()+1e-8);p-=upd"),
      ('identity-less schedule',R,schedule_check,"assert 'identity' in obj,'S-schedule: missing calibration identity'","if 'identity' not in obj:return obj"),
      ('changed dataset calibration accepted',R,schedule_check,"assert ci['dataset_signature']==dataset_signature,'S-schedule: changed dataset between calibration and evaluation'","pass # mutated dataset equality"),
      ('changed dataset overwrites resume',R,schedule_check,"assert json.loads(Path(path).read_text())==json.loads(json.dumps(obj)), 'DATA_IDENTITY_MISMATCH: existing data manifest'","put(path,obj) # mutated existing manifest"),
      ('eval seed schedule',R,schedule_check,"obj['seeds']==CAL_SEEDS","obj['seeds'] in (CAL_SEEDS,EVAL_SEEDS)"),
      ('non-significance implies reproduction',V,verdict_check,"if cn['lo']>0 and cr['lo']>-EQ:","if cn['lo']>0 and cr['hi']>0:"),
      ('missing/duplicate shard accepted',R,worker_check,"assert len(got)==len(set(got)) and set(got)==set(seeds),'INCOMPLETE: duplicate or missing seed shard'","pass # mutated shard admission"),
      ('finite CSV tamper ignored',V,evidence_check,"if Path(name).name!=name or R.sha(directory/name)!=digest:","if Path(name).name!=name:"),
      ('completion identity ignored',V,evidence_check,"if ident!=proof.get('identity') or ident.get('phase')!=phase or ident.get('box')!=box:","if ident.get('phase')!=phase or ident.get('box')!=box:"),
      ('one-box all-three',V,verdict_check,"labels==['NOT_LR']*3","'NOT_LR' in labels"),
    ]
    train_source=Path(R.__file__).read_text().split('def rm_train(',1)[1].split('def rm_prefix(',1)[0]
    gradient_scaling=train_source.replace('grads=torch.autograd.grad(loss,params)', 'grads=[g*(rates[i//2]/1e-3) for i,g in enumerate(torch.autograd.grad(loss,params))]').replace('upd=rates[i//2]*(mi/c1)', 'upd=1e-3*(mi/c1)')
    cases.append(('gradient scaling instead of actual LR',R,host_check,train_source,gradient_scaling))
    for name,module,fn,old,new in cases:
        try:fn(mutant(module,old,new))
        except (AssertionError,ValueError,RuntimeError):mutations.append(dict(name=name,detected=True))
        else:mutations.append(dict(name=name,detected=False))
    checks['S-cost']=dict(pass_=True,detail=real_data_cost())
    good=all(c['pass_'] for c in checks.values()) and all(m['detected'] for m in mutations)
    gpu=cuda_suite() if a.cuda else dict(pass_=False,status='BLOCKED_CUDA' if not torch.cuda.is_available() else 'NOT_YET_CHECKED')
    out=dict(run=R.RUN,cpu_pass=good,cuda_pass=gpu['pass_'],all_pass=good and gpu['pass_'],checks=checks,mutations=mutations,cuda_checks=gpu,
             cuda_status=gpu['status'],
             limitations=['CUDA graph host reproduction, warmup rollback and CUDA probe invariance not executed; production RC/F51 admission fails closed.'],
             source_sha256=R.source_hashes(),thresholds='Same path bit exact; candidate actual delta <= u*(|p|+|upd|)+tiny; scalar float64 1e-12 relative; norm 1e-14 relative.')
    R.put(a.out,out);print(json.dumps({k:out[k] for k in ('cpu_pass','cuda_pass','all_pass','cuda_status')}))
    return 0 if good else 1


if __name__=='__main__':raise SystemExit(main())

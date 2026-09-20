#!/usr/bin/env python3
"""Registered S4: frozen response fields, native CIFAR ELU, R=10."""
from __future__ import annotations
import argparse, csv, gc, json, os, resource, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cifar_interventions_0920 import *

RUN='resp_cifar_ee_0920'
ARMS={
 'N1':(1,None,None,False), 'N10':(10,None,None,False),
 'N1r':(1,None,None,True), 'N10r':(10,None,None,True),
 'R1_10':(10,2,1,False), 'R1_10r':(10,2,1,True),
 'S1_10r':(1,2,10,True), 'S1_10':(1,2,10,False),
 'S1u5r':(1,2,-5,True), 'S1u10r':(1,2,-10,True), 'S1u20r':(1,2,-20,True),
 'S1_L1_10r':(1,1,10,True)}
DIAG_STEPS=(75,750,7500,15000,30000)


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def field_from_states(branch,donor,layer,X):
    dev=X.device
    bp=[q.to(dev) for q in branch['P']];dp=[q.to(dev) for q in donor['P']]
    with torch.no_grad():
        zb=E.forward(bp,X,E.ELU())[2*(layer-1)]
        zd=E.forward(dp,X,E.ELU())[2*(layer-1)]
    return (zd.double()-zb.double()).float()


def arm_engine(name,states,X,graph=True):
    branch,layer,source,reset=ARMS[name]; st=states[branch]
    shift=None
    if layer is not None:
        d=field_from_states(st,states[source],layer,X) if source>0 else torch.full((len(st['seeds']),1200,100),float(source),device=X.device)
        shift=Shift([p.to(X.device) for p in st['P']],layer,d)
    eng=Engine(st['seeds'],X,state=st,shift=shift,graph=graph)
    if reset:eng.reset_adam()
    return eng


def preflight(name,eng,states):
    """Independent P0 + expected-field checks; never uses residual as tolerance."""
    branch,layer,source,reset=ARMS[name]
    bp=[q.to(eng.device) for q in states[branch]['P']]
    if layer is not None:
        assert eng.shift.layer==layer
        assert tree_hash(eng.shift.P0)==tree_hash(bp)
        assert all(p.data_ptr()!=q.data_ptr() for p,q in zip(eng.P,eng.shift.P0))
        dr=field_from_states(states[branch],states[source],layer,eng.X) if source>0 else torch.full_like(eng.shift.d,float(source))
        assert tree_hash(eng.shift.d)==tree_hash(dr),'field construction'
        with torch.no_grad():
            br_eval=E.forward(bp,eng.X,E.ELU())[2*(layer-1)]
            if source>0:
                dp=[q.to(eng.device) for q in states[source]['P']]
                sr_eval=E.forward(dp,eng.X,E.ELU())[2*(layer-1)]
            else:sr_eval=br_eval.double()+float(source)
        q=br_eval.double()+dr.double()-sr_eval.double()
    u=2.0**-24; eta=float(torch.finfo(torch.float32).tiny)
    maximum=0.0; gmaximum=0.0; tested=0
    by_seed={s:dict(seed=s,branch_bit_exact=True,argument_error_ratio=0.,g_error_ratio=0.) for s in eng.seeds}
    gen=E.H.stream('S4_preflight_permutation',0)
    orders=[torch.arange(1200),torch.randperm(1200,generator=gen)]
    batches=[torch.arange(1200)[None,:].expand(eng.R,-1)]
    for order in orders:
        batches += [order[j:j+16][None,:].expand(eng.R,-1) for j in range(0,1200,16)]
    rng_before=tree_hash({s:g.get_state() for s,g in eng.g_batch.items()})
    for batch in batches:
        ids=batch.to(eng.device);xb=eng.X[eng.ar,ids]
        with torch.no_grad():
            natural=E.forward(bp,xb,E.ELU());actual=eng.forward(xb,ids)
        for a,b in zip(actual,natural):assert tree_hash(a)==tree_hash(b),'branch output'
        y=eng.Y[eng.ar,ids]
        assert tree_hash(F.cross_entropy(actual[-1].flatten(0,1),y.flatten()))==tree_hash(F.cross_entropy(natural[-1].flatten(0,1),y.flatten()))
        if layer is not None:
            zb=natural[2*(layer-1)];d=dr[eng.ar,ids]
            arg=zb+d;target=sr_eval[eng.ar,ids].double()
            bound=q[eng.ar,ids].abs()+(zb.double()-br_eval[eng.ar,ids].double()).abs()+u*(zb.double().abs()+d.double().abs())+eta
            bound+=np.finfo(float).eps*(br_eval[eng.ar,ids].double().abs()+d.double().abs()+target.abs())*3
            err=(arg.double()-target).abs();assert (err<=bound).all(),'field argument'
            maximum=max(maximum,float((err/bound).max()))
            ga=gtrain(arg).double();gt=gtrain(target.float()).double()
            # Kernel errors use separately evaluated native g at the independently expected argument.
            ar=(zb.double()+d.double()).float()
            ka=(gtrain(ar).double()-ar.double().clamp(max=0).exp()).abs()
            kt=(gt-target.float().double().clamp(max=0).exp()).abs()
            gb=bound+ka+kt+u*gt.abs()+eta
            ge=(ga-gt).abs();assert (ge<=gb).all(),'field gradient'
            gmaximum=max(gmaximum,float((ge/gb).max()))
            for r,s in enumerate(eng.seeds):
                by_seed[s]['argument_error_ratio']=max(by_seed[s]['argument_error_ratio'],float((err[r]/bound[r]).max()))
                by_seed[s]['g_error_ratio']=max(by_seed[s]['g_error_ratio'],float((ge[r]/gb[r]).max()))
        tested+=1
    assert rng_before==tree_hash({s:g.get_state() for s,g in eng.g_batch.items()})
    return dict(all_pass=True,batches=tested,branch_bit_exact=True,argument_error_ratio=maximum,g_error_ratio=gmaximum,per_seed=list(by_seed.values()))


def serialize_diagnostics(path,history):
    rows=[];arrays={}
    for item in history:
        rows.extend(item['rows'])
        for key,value in item['units'].items():arrays[f"u{item['update']:05d}_{key}"]=value
    put(path/'diagnostics.json',rows);save_npz(path/'diagnostic_units.npz',arrays)


def run(out,seeds,epochs,allow_checks=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    dev=setup();X,cifar=make_inputs(seeds,dev);ident=identity(RUN,seeds,X,cifar,epochs)
    if not allow_checks:
        assert seeds==list(range(10)) and epochs==400
        chk=json.loads((ROOT/f'results/_checks_{RUN}/checks.json').read_text())
        assert chk['all_pass'] and chk['source_sha256']==source_hashes(RUN),'unverified code'
        dirty=subprocess.check_output(['git','status','--porcelain','--','src','analysis','specs'],cwd=ROOT,text=True)
        assert not dirty, 'commit implementation before main run'
    if not allow_checks:
        saved_checks=out/'admission_checks.json'
        if saved_checks.exists():assert json.loads(saved_checks.read_text())==chk
        else:put(saved_checks,chk)
    put(out/'input_manifest.json',dict(data_sha256=ident['data_sha256'],input_hash=ident['input_hash'],subset_order=ident['subset_order']))
    start_path=out/'provenance_start.json'
    if start_path.exists():require_identity(json.loads(start_path.read_text())['identity'],ident)
    else:put(start_path,dict(identity=ident,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid(),argv=sys.argv,started_jst=__import__('datetime').datetime.now(__import__('datetime').timezone(__import__('datetime').timedelta(hours=9))).isoformat(),dirty_diff=subprocess.check_output(['git','diff','HEAD','--','src','analysis','specs'],cwd=ROOT,text=True),independent_audit=False))
    stop=out/'STOP';prefix=out/'prefix';prefix.mkdir(exist_ok=True)
    marker=prefix/'done.json';t0=time.time()
    if marker.exists():verify_done(marker,ident)
    else:
        active=prefix/'active.pt';prefix_rows=[]
        if active.exists():
            saved=load_pt(active);require_identity(saved['identity'],ident);eng=Engine(seeds,X,state=saved['state']);prefix_rows=saved['rows']
        else:eng=Engine(seeds,X)
        first=eng.task if eng.in_task else eng.task+1
        for task in range(first,12):
            if stop.exists():return False
            result=eng.train_task(task,epochs,checkpoint=lambda st:save_pt(active,dict(identity=ident,state=st,rows=prefix_rows)),stop=stop)
            if result is None:return False
            rows,units=result;prefix_rows.extend(rows)
            put(prefix/'rows.json',prefix_rows);save_npz(prefix/f'units_t{task:02d}.npz',units)
            put(prefix/f'diagnostics_t{task:02d}.json',rows)
            if task in (1,2,10,11):save_pt(prefix/f't{task:02d}.pt',eng.state())
            save_pt(active,dict(identity=ident,state=eng.state(),rows=prefix_rows))
            put(out/'status.json',dict(stage='prefix',completed_task=task,wall_s=time.time()-t0))
            print(f'prefix task {task}/11 saved; elapsed {time.time()-t0:.1f}s',flush=True)
            assert all(r['finite'] for r in rows),'DIVERGED prefix'
        mark_done(marker,ident,sorted(p for p in prefix.iterdir() if p.name not in ('done.json','active.pt')))
        del eng;gc.collect();torch.cuda.empty_cache()
    states={t:load_pt(prefix/f't{t:02d}.pt') for t in (1,2,10,11)}
    integrity={}
    # Natural continuations first: qualify the new prefix before any intervention outcome.
    for name in ARMS:
        if stop.exists():return False
        arm=out/'arms'/name;arm.mkdir(parents=True,exist_ok=True)
        done=arm/'done.json'
        if done.exists():verify_done(done,ident);continue
        active=arm/'active.pt'
        if active.exists():
            saved=load_pt(active);require_identity(saved['identity'],ident);eng=Engine(seeds,X,state=saved['state'])
        else:
            eng=arm_engine(name,states,X)
            pf=preflight(name,eng,states);put(arm/'preflight.json',pf)
            ir,iu=eng.diagnostic();put(arm/'initial.json',ir);save_npz(arm/'initial_units.npz',iu)
            save_pt(active,dict(identity=ident,state=eng.state()))
        branch=ARMS[name][0]
        result=eng.train_task(branch+1,epochs,DIAG_STEPS,
                              checkpoint=lambda st:save_pt(active,dict(identity=ident,state=st)),stop=stop)
        if result is None:return False
        rows,units=result
        for row in rows:row['arm']=name
        put(arm/'rows.json',rows);write_csv(arm/'per_task.csv',rows);save_npz(arm/'units.npz',units)
        serialize_diagnostics(arm,eng.diag_history)
        save_pt(arm/'first_epoch_g.pt',eng.first_g)
        end=eng.state();save_pt(arm/'end.pt',end)
        if name in ('N1','N10'):
            expected=states[branch+1]
            assert tree_hash(core_state(end))==tree_hash(core_state(expected)),'natural continuation mismatch'
            prefix_rows=json.loads((prefix/'rows.json').read_text())
            lookup={r['seed']:r for r in prefix_rows if r['task']==branch+1}
            for row in rows:
                for key in ('online_acc','memo_acc','label_hash','order_hash'):
                    assert row[key]==lookup[row['seed']][key],('natural row',key)
            integrity[name]=dict(all_pass=True,core_bit_exact=True,rows_exact=True)
            put(out/'prefix_checks.json',integrity)
        assert all(r['finite'] for r in rows),'DIVERGED arm'
        mark_done(done,ident,sorted(p for p in arm.iterdir() if p.name not in ('done.json','active.pt')))
        put(out/'status.json',dict(stage='arms',completed_arm=name,wall_s=time.time()-t0))
        print(f'arm {name} complete; elapsed {time.time()-t0:.1f}s',flush=True)
        del eng;gc.collect();torch.cuda.empty_cache()
    # Rebuild integrity after resume without relying on an in-memory list.
    for name in ('N1','N10'):
        end=load_pt(out/'arms'/name/'end.pt');expected=states[ARMS[name][0]+1]
        assert tree_hash(core_state(end))==tree_hash(core_state(expected))
        prefix_rows=json.loads((prefix/'rows.json').read_text())
        rows=json.loads((out/'arms'/name/'rows.json').read_text())
        lookup={r['seed']:r for r in prefix_rows if r['task']==ARMS[name][0]+1}
        for row in rows:
            for key in ('online_acc','memo_acc','label_hash','order_hash'):assert row[key]==lookup[row['seed']][key]
        integrity[name]=dict(all_pass=True,core_bit_exact=True,rows_exact=True)
    put(out/'prefix_checks.json',integrity)
    put(out/'status.json',dict(stage='completed',arms=12,seeds=10,wall_s=time.time()-t0))
    put(out/'provenance_end.json',dict(identity=ident,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        elapsed_this_invocation_s=time.time()-t0,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),independent_audit=False))
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'results'/RUN))
    p.add_argument('--seeds',default='0-9');p.add_argument('--epochs',type=int,default=400);p.add_argument('--check-mode',action='store_true')
    args=p.parse_args();seeds=E.parse_ints(args.seeds)
    if args.check_mode:assert Path(args.out).resolve()!=ROOT/'results'/RUN
    with exclusive():success=run(Path(args.out),seeds,args.epochs,args.check_mode)
    if not success:print('STOP acknowledged; checkpoint saved.',flush=True)


if __name__=='__main__':main()

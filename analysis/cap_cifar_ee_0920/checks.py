#!/usr/bin/env python3
"""S5 admission checks with disjoint seeds, independent references and mutations."""
import contextlib,copy,gc,inspect,math,resource,sys,time,traceback,textwrap
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.cap_cifar_ee_0920 import *
import src.cap_cifar_ee_0920 as C
from analysis.cap_cifar_ee_0920 import stats as S
from analysis.cap_cifar_ee_0920 import report as R
OUT=ROOT/f'results/_checks_{RUN}';SEEDS=list(range(100,110))
REQUIRED=['S-host/off','S-radius','S-project','S-strength','S-bfix','S-branch/RNG','S-derivative','S-ledger','S-graph','S-resume/STOP','S-verdict','S-manifest/CLI','S-cost']
MUTANTS=['lr','time','input_order','radius_initial','radius_mean','layer_swap','inflate','pre_project','project_W3','no_projection','moment_project','scale_ignored','radius_overwritten','b2_only','b3_fixed','bias_zero','grad_zero','alias','diagnostic_rng','output_plus_one','expm1','ledger_cross','ledger_bias','ledger_Wnew','ledger_otherseed','counter_warmup','missing_radius','missing_bfixed','missing_moment','missing_acc','different_identity','wrong_window','fixed_floor','unpaired','no_bonferroni','missing_seed','missing_arm','missing_task','duplicate','fake_done','bad_hash','twenty_tasks','missing_provenance']


def same(a,b):assert tree_hash(a)==tree_hash(b),'bit mismatch'


def reject(fn):
    try:fn()
    except (AssertionError,KeyError,ValueError,FileNotFoundError,StopIteration):return
    raise AssertionError('mutant accepted')


def modified(fn,old,new):
    src=textwrap.dedent(inspect.getsource(fn));assert src.count(old)==1,(old,src.count(old))
    src=src.replace(old,new).replace('super().','super(CapEngine,self).')
    ns={};exec(src,fn.__globals__,ns);return ns[fn.__name__]


@contextlib.contextmanager
def patch(obj,name,fn):
    old=getattr(obj,name);setattr(obj,name,fn)
    try:yield
    finally:setattr(obj,name,old)


def fixture():
    data={}
    for arm in S.ARMS:
        A=np.full((10,50),.11);A[:,0]=.9;A[:,1]=.8
        G=np.full((10,50),.0001);G[:,0]=.8;F=np.full((10,50),.11)
        if arm in ('cap12','cap12_bfix'):A[:,2:]=.7;G[:,1:]=.3
        if arm=='cap2':A[:5,2:]=.7;G[:5,1:]=.3
        data[arm]=dict(A=A,G=G,F=F)
    return data


def stats_check(mut):
    x=fixture();v=S.evaluate(x);assert v['label']=='RESCUED' and v['arms']['cap1']['label']=='COLLAPSED' and v['arms']['cap2']['label']=='SPLIT'
    assert S.classify(0,'+','0')=='RESCUED_FUNCTION_ONLY' and S.classify(0,'+','-')=='RESCUED_FUNCTION_ONLY'
    assert S.classify(0,'0','+')=='ALIVE_UNRESOLVED' and S.classify(10,'+','+')=='COLLAPSED'
    for n in (8,9):
        x=fixture();x['ref']['A'][n:,30:]=.9;assert S.evaluate(x)['applicable']==(n==9)
    for n in (5,6):
        x=fixture();x['cap12']['A'][:n,1]=.4;assert (S.evaluate(x)['arms']['cap12']['early']=='IMPAIRED_EARLY')==(n==6)
    x=fixture();x['cap12']['A'][:,1]=(.8+.11)/2;assert S.evaluate(x)['arms']['cap12']['early']=='NO_EARLY_IMPAIRMENT'
    x=fixture();x['ref']['A'][0,1]=.11;assert S.evaluate(x)['arms']['cap12']['early']=='EARLY_FLAG_NOT_ASSESSABLE'
    u,mean,sd=S.upper_floor(np.full((10,20),.11));assert np.all(u==.11) and not sd.any()
    assert S.interval(np.full(10,.1))['degenerate_sd'] and S.interval(np.zeros(10))['sign']=='0'
    alive=np.full((10,50),.9);dead=alive.copy();dead[:,5:]=.11
    assert S.timing(alive,alive)['label']=='TIMING_UNDETERMINED' and S.timing(alive,dead)['label']=='LATER' and S.timing(dead,alive)['label']=='EARLIER'
    assert S.timing(dead,dead)['label']=='TIMING_UNDETERMINED'
    q=S.t_quantile(.95,19);xx=np.linspace(0,q,20001);f=math.gamma(10)/(math.sqrt(19*math.pi)*math.gamma(9.5))*(1+xx*xx/19)**-10
    area=(xx[1]-xx[0])/3*(f[0]+f[-1]+4*f[1:-1:2].sum()+2*f[2:-1:2].sum());assert abs(.5+area-.95)<2e-12
    for name,edit in [('missing_seed',lambda d:d['cap1'].__setitem__('A',d['cap1']['A'][:-1])),('missing_task',lambda d:d['cap1'].__setitem__('G',d['cap1']['G'][:,:49])),('missing_arm',lambda d:d.pop('cap1'))]:
        x=fixture();edit(x);reject(lambda:S.evaluate(x));mut[name]=True
    x=fixture();x['cap12']['A'][:,29]=.9;x['cap12']['A'][:,49]=.2;good=S.evaluate(x)
    source=textwrap.dedent(inspect.getsource(S.evaluate));badsrc=source.replace('30:50','29:49');assert source!=badsrc
    ns={};exec(badsrc,S.evaluate.__globals__,ns);reject(lambda:same(good,ns['evaluate'](x)));mut['wrong_window']=True
    with patch(S,'upper_floor',lambda f:(np.full(10,.1),np.full(10,.1),np.zeros(10))):reject(lambda:same(v,S.evaluate(fixture())));mut['fixed_floor']=True
    x=fixture()
    for s in range(10):
        for a in S.ARMS:x[a]['F'][s,:]=.11+.001*s
        x['ref']['A'][s,30:]=.11+.001*s;x['cap12']['A'][s,30:]=.3+.04*s
    good=S.evaluate(x)
    for name,old,new in [('no_bonferroni','levels=(.975,.95)','levels=(.95,.95)'),('unpaired',"endpoints[a][k]-endpoints['ref'][k]","endpoints[a][k]-endpoints['ref'][k][::-1]")]:
        bad=modified(S.evaluate,old,new);reject(lambda:same(good,bad(x)));mut[name]=True
    return dict(labels=True,boundary_equal_floor=True,zero_sd=True,nine_of_ten_gate=True,censoring=True,independent_t19_integration=True)


def projection_reference(before,actual,radius):
    w=before.double();r=radius.double();norm=w.norm(dim=2)
    target=w*torch.where(norm>r,r/torch.where(norm>0,norm,1),torch.ones_like(norm))[:,:,None]
    d=w.shape[2];u=2**-24;gam=(2*d+4)*u/(1-(2*d+4)*u)
    bound=4*gam*w.abs()+math.sqrt(d)*torch.finfo(torch.float32).tiny
    err=(actual.double()-target).abs();assert bool((err<=bound).all()),('projection coordinate bound',float((err-bound).detach().max()))
    assert bool((actual.double().norm(dim=2)<=norm_bound(radius,d)).all())
    return float(torch.where(bound>0,err/bound,0).detach().max())


def compare_update(actual,reference,arm,anchor):
    for k in ('m','v','tc'):same(getattr(actual,k),getattr(reference,k))
    for i in range(3):
        if i+1 in ARMS[arm]:projection_reference(reference.P[2*i],actual.P[2*i],actual.radii[i])
        else:same(actual.P[2*i],reference.P[2*i])
        same(actual.P[2*i+1],anchor['P'][2*i+1] if arm=='cap12_bfix' and i<2 else reference.P[2*i+1])
    assert not bool(actual.counts['violations'].any()) and not bool(actual.counts['bfix_errors'].any())


def ledger_check(mut):
    rng=np.random.default_rng(12345)
    old=dict(w=rng.normal(size=(100,100)),b=rng.normal(size=100),mu=rng.normal(size=100),mu_error=np.zeros(100));new={k:v.copy() for k,v in old.items()}
    new['w']+=rng.normal(size=(100,100))*.3;new['mu']+=rng.normal(size=100)*.3;new['b']+=rng.normal(size=100)*.3
    q=L.components(old,new);direct=new['w']@new['mu']+new['b']-old['w']@old['mu']-old['b']
    # Independently accumulated dot product bound includes both products and summations.
    bound=L.gamma(1200)*(abs(new['w'])@abs(new['mu'])+abs(old['w'])@abs(old['mu'])+abs(new['b'])+abs(old['b'])+1)
    correct=sum(q[k].v for k in L.TERMS);assert np.all(abs(correct-direct)<=bound)
    for name,bad in [('ledger_cross',correct-q['cross'].v),('ledger_bias',correct-q['bias'].v),('ledger_Wnew',correct+(new['w']-old['w'])@(new['mu']-old['mu'])),('ledger_otherseed',(new['w']-old['w'])@old['mu'][::-1]+q['upstream'].v+q['cross'].v+q['bias'].v)]:
        assert np.any(abs(bad-direct)>bound);mut[name]=True
    for stem in ('self','upstream'):
        delta=q[stem+'_growth']+q[stem+'_rotation']-q[stem];assert np.all(abs(delta.v)<=delta.e)
    return dict(independent_four_term=True,symmetric_norm_direction=True)


def suite():
    OUT.mkdir(parents=True,exist_ok=True);attempt=OUT/f'attempt_{time.time_ns()}';attempt.mkdir();evidence={};mut={};tested=cap_hashes();t0=time.time()
    def record(name,detail):
        evidence[name]=dict(all_pass=True,**detail);put(attempt/'progress.json',dict(evidence=evidence,mutants=mut));print(name+' PASS',flush=True)
    try:
        record('S-verdict',stats_check(mut));record('S-ledger',ledger_check(mut))
        dev=setup();X,cifar=make_inputs(SEEDS,dev)
        z=torch.tensor([-200.,-104.,-103.,-100.,-90.,-40.,-20.,-17.,-1.,-0.,0.,1.],device=dev,requires_grad=True)
        g=torch.autograd.grad(F.elu(z).sum(),z)[0];same(g,gtrain(z));wrong=torch.where(z<=0,F.elu(z)+1,1);reject(lambda:same(g,wrong));mut['output_plus_one']=True
        wrong=torch.autograd.grad(torch.where(z<=0,torch.expm1(z),z).sum(),z)[0];reject(lambda:same(g,wrong));mut['expm1']=True
        record('S-derivative',dict(z=z.detach().cpu().tolist(),native=g.cpu().tolist(),diagnostic=E.ELU().dphi(z).detach().cpu().tolist()))
        cache=OUT/'host_exact.json';key=dict(code=sha(ROOT/f'src/{RUN}.py'),common=sha(ROOT/'src/cifar_interventions_0920.py'),checks=sha(Path(__file__)),input=tree_hash(X),seeds=SEEDS,epochs=400)
        if cache.exists() and json.loads(cache.read_text())['key']==key:host=json.loads(cache.read_text())['result']
        else:
            hostdir=OUT/'host_full';hostdir.mkdir(exist_ok=True)
            with (hostdir/'stdout.log').open('w') as log,contextlib.redirect_stdout(log):E.run('ELU',SEEDS,['std'],2,400,dev,hostdir,cifar=cifar,checkpoint=True,snapshots=False)
            h=load_pt(hostdir/'ckpt.pt');e=CapEngine(SEEDS,X);rows=[];start=time.time()
            for t in (1,2):rows.extend(e.train_task(t,400)[0])
            st=e.state()
            for k in ('P','m','v','tc','g_lab','g_batch'):same(st[k],h[k])
            lookup={(r['seed'],r['task']):r for r in h['rows']}
            for r in rows:
                for k in ('online_acc','memo_acc'):assert r[k]==lookup[r['seed'],r['task']][k]
            host=dict(core_bit_exact=True,online_bit_exact=True,R=10,tasks=2,epochs=400,seconds=time.time()-start,core_hash=tree_hash(core_state(st)))
            put(cache,dict(key=key,result=host));del e;gc.collect();torch.cuda.empty_cache()
        record('S-host/off',host)
        e=Engine(SEEDS,X);e.train_task(1,1);anchor=e.state();base=cpu(anchor);del e
        ids=torch.arange(16,device=dev)[None,:].expand(10,-1)
        # A deliberately active fixture: exceed both radii before Adam and cap.
        forced=cpu(anchor);forced['P'][0]*=3;forced['P'][2]*=3
        ref=Engine(SEEDS,X,forced,graph=False);ref.one_update(ids)
        for arm in ARMS:
            e=CapEngine(SEEDS,X,forced,arm,anchor,graph=False);verify_anchor(e,anchor);e.one_update(ids);compare_update(e,ref,arm,anchor)
            if arm!='ref':assert all(bool((e.counts['hits'][:,l-1]>0).all()) for l in ARMS[arm])
            if arm=='cap12_bfix':assert bool((e.counts['bfix_hits']>0).any())
            del e
        # Pure row map: below/equal/above/zero radius including exact unchanged bits.
        w=torch.tensor([[[3.,4.],[0.,2.],[0.,0.],[3.,4.],[3.,4.]]],device=dev);r=torch.tensor([[5.,3.,0.,2.5,0.]],device=dev)
        out,hit=project_rows(w,r);same(out[:,:3],w[:,:3]);same(out[:,3],w[:,3]*.5);assert not bool(out[:,4].any());same(hit,torch.tensor([[False,False,False,True,True]],device=dev))
        ratio=projection_reference(w,out,r)
        tiny=torch.tensor([[[1e-40,-1e-40]]],device=dev);zero=torch.zeros(1,1,device=dev)
        tiny_out,tiny_hit=project_rows(tiny,zero);assert not bool(tiny_out.any()) and bool(tiny_hit.all())
        e=CapEngine(SEEDS,X,anchor,'cap12',anchor,False)
        for name,edit in [('radius_initial',lambda x:setattr(x,'radii',[q.norm(dim=2) for q in Engine(SEEDS,X,graph=False).P[::2][:2]])),('radius_mean',lambda x:setattr(x,'radii',[q.mean().expand_as(q).clone() for q in x.radii])),('layer_swap',lambda x:x.radii.reverse()),('radius_overwritten',lambda x:x.radii[0].mul_(2))]:
            bad=CapEngine(SEEDS,X,anchor,'cap12',anchor,False);edit(bad);reject(lambda:verify_anchor(bad,anchor));mut[name]=True;del bad
        badfun=modified(project_rows,'norm>radius','norm!=radius')
        with patch(C,'project_rows',badfun):bad,_=C.project_rows(w,r);reject(lambda:same(bad,out));mut['inflate']=True
        record('S-radius',dict(per_row=True,below_equal_above_zero=True));record('S-project',dict(independent_float64=True,coordinate_error_ratio=ratio,moments_untouched=True,active_fixture=True))
        # Strong/weak bounds using full steps, not an absence-of-change assertion alone.
        loose=CapEngine(SEEDS,X,anchor,'cap12',anchor,True,scale=1000.);off=Engine(SEEDS,X,anchor)
        for _ in range(75):loose.one_update(ids);off.one_update(ids)
        same(core_state(loose.state()),core_state(off.state()));assert not bool(loose.counts['hits'].any())
        tight=CapEngine(SEEDS,X,forced,'cap12',anchor,False,scale=.5);tight.one_update(ids);compare_update(tight,ref,'cap12',anchor)
        assert bool((tight.counts['hits']>0).all())
        bad=CapEngine(SEEDS,X,forced,'cap12',anchor,False,scale=1.);bad.one_update(ids);reject(lambda:same(bad.P,tight.P));mut['scale_ignored']=True
        record('S-strength',dict(loose_scale=1000,loose_steps=75,loose_all_off=True,tight_scale=.5,tight_all_on=True))
        del loose,off,tight,bad,e;gc.collect();torch.cuda.empty_cache()
        # Actual faulty methods evaluated against the independent one-update comparator.
        changes=[('no_projection','w.copy_(projected)','w.copy_(before)'),('moment_project',"after=w.double().norm(dim=2)","self.m[2*i].mul_(.5);after=w.double().norm(dim=2)"),('b2_only','for i in range(2):','for i in (1,):'),('b3_fixed','b.copy_(self.b_fixed[i])','b.copy_(self.b_fixed[i]);self.P[5].zero_()'),('bias_zero','b.copy_(self.b_fixed[i])','b.zero_()'),('project_W3','super().step()','super().step()\n    with torch.no_grad():self.P[4].mul_(.5)')]
        for name,old,new in changes:
            fn=modified(CapEngine.step,old,new)
            with patch(CapEngine,'step',fn):
                e=CapEngine(SEEDS,X,forced,'cap12_bfix',anchor,False);e.one_update(ids);reject(lambda:compare_update(e,ref,'cap12_bfix',anchor));mut[name]=True
            del e
        fn=textwrap.dedent(inspect.getsource(CapEngine.step)).replace('super().step()','pass')+'\n    super().step()\n';ns={};exec(fn.replace('super().','super(CapEngine,self).'),CapEngine.step.__globals__,ns)
        with patch(CapEngine,'step',ns['step']):
            e=CapEngine(SEEDS,X,forced,'cap12_bfix',anchor,False);e.one_update(ids);reject(lambda:compare_update(e,ref,'cap12_bfix',anchor));mut['pre_project']=True
        del e
        for name,method,old,new in [('lr','step','p.sub_(1e-3 *','p.sub_(2e-3 *'),('time','one_update','self.tc += 1','self.tc += 100'),('input_order','one_update','self.ids.copy_(ids)','self.ids.copy_(ids.flip(1))'),('grad_zero','step','with torch.no_grad():','grads=list(grads);grads[1]=torch.zeros_like(grads[1]);grads[3]=torch.zeros_like(grads[3])\n    with torch.no_grad():')]:
            fn=modified(getattr(Engine,method),old,new)
            with patch(Engine,method,fn):
                e=CapEngine(SEEDS,X,forced,'cap12_bfix',anchor,False);e.one_update(ids);reject(lambda:compare_update(e,ref,'cap12_bfix',anchor));mut[name]=True
            del e
        record('S-bfix',dict(both_hidden_exact=True,output_bias_free=True,raw_gradients_and_moments_match=True,candidate_displacement_nonzero=True))
        # Branch order, fixed state isolation, CUDA graph and epoch STOP resume.
        results={};anchor_hash=tree_hash(anchor);cost={}
        for order in (list(ARMS),list(reversed(ARMS))):
            for arm in order:
                e=CapEngine(SEEDS,X,anchor,arm,anchor);same(core_state(e.state()),core_state(anchor));fixed=tree_hash([e.radii,e.b_fixed]);before=tree_hash([g.get_state() for g in e.g_batch.values()]);e.diagnostic();same(before,tree_hash([g.get_state() for g in e.g_batch.values()]))
                start=time.time();e.train_task(2,2);cost[arm]=time.time()-start;same(fixed,tree_hash([e.radii,e.b_fixed]));st=e.state()
                if arm=='cap12_bfix' and order[0]=='ref':
                    with torch.no_grad():full=E.forward(e.P,X,E.ELU())
                    gs=[gtrain(full[k]) for k in (0,2)];maximum=[0.,0.];mean=[0.,0.]
                    for j in range(75):
                        ii=torch.arange(j*16,(j+1)*16,device=dev)[None,:].expand(10,-1)
                        with torch.no_grad():vals=E.forward(e.P,X[e.ar,ii],E.ELU())
                        for l,k in enumerate((0,2)):
                            delta=(gtrain(vals[k]).double()-gs[l][e.ar,ii].double()).abs()
                            maximum[l]=max(maximum[l],float(delta.max()));mean[l]+=float(delta.mean())/75
                    evidence['S-derivative']['B16_vs_B1200_same_state']=dict(max_abs_difference=maximum,mean_abs_difference=mean,test_seeds=SEEDS)
                if arm in results:same(results[arm],tree_hash(st))
                else:results[arm]=tree_hash(st)
                del e;gc.collect();torch.cuda.empty_cache()
        same(anchor_hash,tree_hash(anchor))
        record('S-branch/RNG',dict(all_five_reversed_order=True,independent_clone=True,diagnostic_rng_unchanged=True))
        for arm in ('ref','cap12_bfix'):
            a=CapEngine(SEEDS,X,anchor,arm,anchor,False);b=CapEngine(SEEDS,X,anchor,arm,anchor,True);same(a.state(),b.state());a.train_task(2,2);b.train_task(2,2);same(a.state(),b.state());expected=b.state();del a,b
            e=CapEngine(SEEDS,X,anchor,arm,anchor);stop=attempt/'STOP';stop.touch();ck=attempt/'resume.pt'
            assert e.train_task(2,2,checkpoint=lambda st:save_pt(ck,st),stop=stop) is None
            stop.unlink();st=load_pt(ck);del e;e=CapEngine(SEEDS,X,st,arm);e.train_task(2,2);same(e.state(),expected);del e
            edits=[('missing_moment',lambda q:q['m'][0].zero_()),('missing_acc',lambda q:q['acc_sum'].zero_())]
            if arm!='ref':edits+=[('missing_radius',lambda q:q['cap']['radii'][0].zero_()),('missing_bfixed',lambda q:q['cap']['b_fixed'][0].zero_())]
            for name,edit in edits:
                bad=cpu(st);edit(bad);e=CapEngine(SEEDS,X,bad,arm);e.train_task(2,2);reject(lambda:same(e.state(),expected));mut[name]=True;del e
        record('S-graph',dict(eager_graph_bit_exact=True,all_counter_warmup_rollback=True));record('S-resume/STOP',dict(epoch_resume_bit_exact=True,after_cap_resume=True))
        e=CapEngine(SEEDS,X,forced,'cap12_bfix',anchor,False);before=e.state();e.inv1.fill_(1);e.inv2.fill_(1);e.step();reject(lambda:same(before,e.state()));mut['counter_warmup']=True
        before=e.state();torch.randperm(1200,generator=e.g_batch[100]);reject(lambda:same(before,e.state()));mut['diagnostic_rng']=True
        bad=cpu(anchor);bad['P']=anchor['P'];h=tree_hash(anchor);bad['P'][0].add_(1);reject(lambda:same(h,tree_hash(anchor)));mut['alias']=True
        # Restore the intentionally aliased fixture from an independent saved copy.
        anchor=base;del e,ref;gc.collect();torch.cuda.empty_cache()
        fake=attempt/'fake';fake.mkdir();put(fake/'done.json',dict(identity={},complete=True,files={'absent.pt':'no'}));reject(lambda:verify_done(fake/'done.json',{}));mut['fake_done']=True
        (fake/'a').write_text('good');mark_done(fake/'done.json',{},[fake/'a']);(fake/'a').write_text('bad');reject(lambda:verify_done(fake/'done.json',{}));mut['bad_hash']=True
        reject(lambda:require_identity({'git':'a'},{'git':'b'}));mut['different_identity']=True
        put(fake/'status.json',dict(stage='completed',tasks=20));reject(lambda:R.report(fake));mut['twenty_tasks']=True
        put(fake/'status.json',dict(stage='completed',tasks=50));reject(lambda:R.report(fake));mut['missing_provenance']=True
        expected={(s,t) for s in range(10) for t in range(1,51)};duplicated=list(expected);duplicated[-1]=duplicated[0];reject(lambda:R.validate_coverage([dict(seed=s,task=t) for s,t in duplicated]));mut['duplicate']=True
        # Real CLI path, short budget, five arms, same source and no lock nesting.
        cli=attempt/'cli';argv=sys.argv;sys.argv=[str(ROOT/f'src/{RUN}.py'),'--check-mode','--out',str(cli),'--seeds','100-109','--epochs','1','--tasks','3']
        start=time.time()
        try:
            with patch(C,'exclusive',contextlib.nullcontext),(attempt/'cli.log').open('w') as log,contextlib.redirect_stdout(log):C.main()
        finally:sys.argv=argv
        status=json.loads((cli/'status.json').read_text());assert status['stage']=='completed' and status['tasks']==3
        for arm in ARMS:assert len(json.loads((cli/'arms'/arm/'rows.json').read_text()))==30
        record('S-manifest/CLI',dict(all_five_arms=True,tasks=3,guarded_cli=True,partial_reports_rejected=True))
        record('S-cost',dict(seconds_two_epochs=cost,seconds_cli=time.time()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_cuda_bytes=torch.cuda.max_memory_allocated(),estimated_training_seconds=sum(cost.values())/2*400*49))
        assert set(evidence)==set(REQUIRED),set(REQUIRED)-set(evidence)
        assert set(mut)==set(MUTANTS),(set(MUTANTS)-set(mut),set(mut)-set(MUTANTS))
        assert tested==cap_hashes(),'source changed during checks'
        result=dict(all_pass=True,source_sha256=tested,required=REQUIRED,required_mutants=MUTANTS,evidence=evidence,mutants=mut,attempt=str(attempt.relative_to(ROOT)),seconds=time.time()-t0,independent_audit=False)
        put(attempt/'checks.json',result);put(OUT/'checks.json',result);print('ALL ADMISSION CHECKS PASS',flush=True)
    except BaseException:
        put(attempt/'failure.json',dict(evidence=evidence,mutants=mut,traceback=traceback.format_exc(),source_sha256=cap_hashes()));raise


if __name__=='__main__':
    with exclusive():suite()

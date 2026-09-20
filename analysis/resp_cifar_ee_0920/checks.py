#!/usr/bin/env python3
"""Executable admission tests. Test seeds only; fail closed; keep every attempt."""
import contextlib, gc, inspect, json, math, resource, subprocess, sys, time, traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.resp_cifar_ee_0920 import *
from analysis.resp_cifar_ee_0920.report import verdict, report
from analysis.resp_cifar_ee_0920.stats import interval, label, t_quantile

OUT=ROOT/f'results/_checks_{RUN}'
SEEDS=list(range(100,110))
REQUIRED=['S-host','S-branch','S-field','S-grad','S-derivative','S-reset','S-isolation','S-graph','S-verdict','S-resume/STOP','S-cost/CLI']
MUTANTS=['lr','time','order','expm1','drop_anchor','alias_P0','wrong_layer','zero_field','sign_field','seed_field','unit_field','image_field','batch_position','drop_backward_shift','frozen_grad','backward_only','output_plus_one','reset_m_only','reset_keep_tc','epsilon_omitted','shallow_copy','diagnostic_rng','warmup_state','unpaired','one_sided','no_bonferroni','missing_seed','nonfinite','missing_arm','acc_lost','field_lost','different_identity','fake_done','bad_hash','wrong_arm','wrong_epochs','missing_provenance']


def reject(fn):
    try:fn()
    except (AssertionError,ValueError,FileNotFoundError,StopIteration,KeyError):return True
    raise AssertionError('mutant was accepted')


def same(a,b):assert tree_hash(a)==tree_hash(b),'not bit identical'


def replace_method(cls,method,old,new):
    import textwrap
    src=textwrap.dedent(inspect.getsource(getattr(cls,method)))
    assert src.count(old)==1,(method,old,src.count(old))
    ns={};exec(src.replace(old,new),getattr(cls,method).__globals__,ns)
    return ns[method]


@contextlib.contextmanager
def patched(cls,method,fn):
    original=getattr(cls,method);setattr(cls,method,fn)
    try:yield
    finally:setattr(cls,method,original)


def derivative_check():
    z=torch.tensor([-200.,-104.,-103.,-100.,-90.,-88.,-40.,-20.,-17.,-16.,-1.,-1e-7,-0.,0.,1e-7,1.],device='cuda')
    q=z.clone().requires_grad_();native=torch.autograd.grad(F.elu(q).sum(),q)[0]
    same(gtrain(z),native)
    exp=z.double().clamp(max=0).exp();err=(native.double()-exp).abs()
    # exp implementation plus final rounding: 8 ulps for libdevice, subnormal absolute bound.
    bound=8*2**-24*exp+8*2**-149
    assert bool((err<=bound).all()),(err.tolist(),bound.tolist())
    wrong=torch.where(z<=0,F.elu(z)+1,torch.ones_like(z))
    reject(lambda:same(native,wrong))
    q=z.clone().requires_grad_();wrong2=torch.autograd.grad(torch.where(q<=0,torch.expm1(q),q).sum(),q)[0]
    reject(lambda:same(native,wrong2))
    return dict(z=z.cpu().tolist(),native=native.cpu().tolist(),diagnostic=E.ELU().dphi(z).cpu().tolist(),max_error_ratio=float((err/bound).max()))


def gradient_check():
    # Small, nonconstant fp64 synthetic network: independent analytic CE chain rule.
    gen=torch.Generator().manual_seed(271828)
    dims=(7,5,4,3);r,b=2,6
    P=[torch.randn((r,dims[i+1],dims[i]),generator=gen,dtype=torch.float64)*.3 if k==0 else torch.randn((r,dims[i+1]),generator=gen,dtype=torch.float64)*.2 for i in range(3) for k in range(2)]
    P=[q.cuda().requires_grad_() for q in P]
    X=torch.randn((r,b,dims[0]),generator=gen,dtype=torch.float64).cuda();ids=torch.arange(b,device='cuda')[None].expand(r,-1)
    ratios=[]
    for layer in (1,2):
        d=torch.randn((r,b,dims[layer]),generator=gen,dtype=torch.float64).cuda()-1.5
        sh=Shift(P,layer,d)
        with torch.no_grad():P[0].add_(.013);P[3].sub_(.017)
        vals=sh(P,X,ids);z1,a1,z2,a2,z3=vals
        y=(torch.arange(r*b,device='cuda')%3).view(r,b)
        loss=F.cross_entropy(z3.flatten(0,1),y.flatten(),reduction='none').view(r,b).mean(1).sum()
        got=torch.autograd.grad(loss,P,retain_graph=True)
        prob=(z3-z3.max(-1,keepdim=True).values).exp();prob=prob/prob.sum(-1,keepdim=True)
        dz3=(prob-F.one_hot(y,3))/b
        dz2=(dz3@P[4])*torch.exp((z2+(d if layer==2 else 0)).clamp(max=0))
        dz1=(dz2@P[2])*torch.exp((z1+(d if layer==1 else 0)).clamp(max=0))
        expected=[dz1.transpose(1,2)@X,dz1.sum(1),dz2.transpose(1,2)@a1,dz2.sum(1),dz3.transpose(1,2)@a2,dz3.sum(1)]
        # Accumulation of all operations, bounded using absolute intermediate scales.
        scale=sum(float(q.detach().abs().sum()) for q in [*P,*vals,X,d,dz1,dz2,dz3])
        bound=(128*np.finfo(float).eps/(1-128*np.finfo(float).eps))*scale
        for a,e in zip(got,expected):
            err=float((a-e).detach().abs().max());assert err<=bound,(layer,err,bound);ratios.append(err/bound)
        # Real wrong forward/autograd implementations, with recomputed gradients.
        mutants={
            'drop_backward_shift':('F.elu(z1 + d)' if layer==1 else 'F.elu(z2 + d)','F.elu(z1)' if layer==1 else 'F.elu(z2)'),
            'frozen_grad':('with torch.no_grad():','with torch.enable_grad():'),
            'backward_only':('a1 = AnchorAdd.apply(a01, F.elu(z1 + d) - F.elu(z01 + d))' if layer==1 else 'a2 = AnchorAdd.apply(a02, F.elu(z2 + d) - F.elu(z02 + d))',
                             'a1 = F.elu(z1) + (F.elu(z1+d)-F.elu(z1+d).detach())' if layer==1 else 'a2 = F.elu(z2) + (F.elu(z2+d)-F.elu(z2+d).detach())')}
        for name,(old,new) in mutants.items():
            fn=replace_method(Shift,'__call__',old,new)
            with patched(Shift,'__call__',fn):
                if name=='frozen_grad':
                    # Erroneous P0 shares current parameters; frozen graph receives gradients.
                    bad=Shift(P,layer,d);bad.P0=P
                else:bad=sh
                bz=bad(P,X,ids)[-1];bl=F.cross_entropy(bz.flatten(0,1),y.flatten(),reduction='sum')/b
                bg=torch.autograd.grad(bl,P)
            assert max(float((a-e).detach().abs().max()) for a,e in zip(bg,expected))>bound,name
    return dict(max_error_ratio=max(ratios),independent='manual float64 CE chain rule',layers=[1,2])


def statistics_check(mut):
    # Independent numerical integration of the t density, not incomplete beta.
    for p in (.975,.9875):
        q=t_quantile(p,9);xx=np.linspace(0,q,20001)
        density=math.gamma(5)/(math.sqrt(9*math.pi)*math.gamma(4.5))*(1+xx*xx/9)**-5
        area=(xx[1]-xx[0])/3*(density[0]+density[-1]+4*density[1:-1:2].sum()+2*density[2:-1:2].sum())
        assert abs(.5+area-p)<2e-12
    g={1:np.ones(10),10:np.ones(10)*.2}
    base={n:np.ones(10)*.5 for n in ARMS};base['N10']=np.ones(10)*.1;base['N1r']=np.ones(10)*.5
    for a,b,want in [(1,1,'RESPONSE_BOTH_WAYS'),(1,0,'RESTORE_ONLY'),(0,1,'SINK_ONLY'),(0,0,'RESPONSE_NOT_SHOWN'),(-1,1,'RESPONSE_REVERSED'),(1,-1,'RESPONSE_REVERSED')]:
        v=cpu(base);v['R1_10']=v['N10']+a*.1;v['S1_10r']=v['N1r']-b*.1
        assert verdict(v,g)['label']==want
    assert interval(np.zeros(10))['sign']=='0' and interval(np.zeros(10))['degenerate_sd']
    assert label({'sign':'0'},{'sign':'0'})=='RESPONSE_NOT_SHOWN'
    v=cpu(base);v['N1']=v['N10'].copy();assert verdict(v,g)['label']=='NOT_REPRODUCED' and verdict(v,g)['rho1'] is None
    v=cpu(base);v['N10r']=v['N1r'].copy();assert verdict(v,g)['rho2'] is None
    for name,edit in [('missing_seed',lambda v:v.__setitem__('N1',v['N1'][:-1])),('nonfinite',lambda v:v['N1'].__setitem__(0,np.nan)),('missing_arm',lambda v:v.pop('N10'))]:
        v=cpu(base);edit(v);reject(lambda:verdict(v,g));mut[name]=True
    noise=np.linspace(-.3,.3,10);v=cpu(base);v['R1_10']=base['N10']+noise+.151
    good=verdict(v,g)['P1'];assert good['sign']=='0'
    # Both wrong levels create a positive sign for the nonzero variance fixture.
    for n,level in [('one_sided',.90),('no_bonferroni',.95)]:
        assert interval(v['R1_10']-v['N10'],level)['sign']=='+';mut[n]=True
    pair=np.arange(10)/10;good=interval((pair+.01)-pair)
    wrong=interval((pair+.01)-pair[::-1]);assert good['sign']=='+' and wrong['sign']=='0';mut['unpaired']=True
    return dict(labels=7,quantiles='independent Simpson integration',all_ten_required=True)


def suite():
    OUT.mkdir(parents=True,exist_ok=True);attempt=OUT/f'attempt_{time.time_ns()}';attempt.mkdir()
    evidence={};mut={};started=time.time();tested_source=source_hashes(RUN)
    def record(name,value):
        evidence[name]=dict(all_pass=True,**value);put(attempt/'progress.json',dict(evidence=evidence,mutants=mut));print(name+' PASS',flush=True)
    try:
        dev=setup();X,cifar=make_inputs(SEEDS,dev)
        record('S-derivative',derivative_check());mut.update({k:True for k in ('expm1','output_plus_one')})
        record('S-grad',gradient_check());mut.update({k:True for k in ('drop_backward_shift','frozen_grad','backward_only')})
        record('S-verdict',statistics_check(mut))
        # Host equivalence at the registered full budget. Cache only exact source+input versions.
        hostdir=OUT/'host_full';cache=OUT/'host_exact.json'
        key=dict(source={p:sha(ROOT/p) for p in ('src/cifar_interventions_0920.py','src/rlcifar_mlp_battle_0918.py','analysis/resp_cifar_ee_0920/checks.py')},input=tree_hash(X),seeds=SEEDS,epochs=400)
        if cache.exists() and json.loads(cache.read_text()).get('key')==key:
            hostresult=json.loads(cache.read_text())['result']
        else:
            hostdir.mkdir(exist_ok=True)
            with (hostdir/'stdout.log').open('w') as log,contextlib.redirect_stdout(log):
                E.run('ELU',SEEDS,['std'],2,400,dev,hostdir,cifar=cifar,checkpoint=True,snapshots=False)
            h=load_pt(hostdir/'ckpt.pt');eng=Engine(SEEDS,X);rows=[];tt=time.time()
            for t in (1,2):rows.extend(eng.train_task(t,400)[0])
            st=eng.state()
            for k in ('P','m','v','tc','g_lab','g_batch'):same(st[k],h[k])
            lookup={(r['seed'],r['task']):r for r in h['rows']}
            for r in rows:
                for k in ('online_acc','memo_acc'):assert r[k]==lookup[r['seed'],r['task']][k]
            for t in (1,2):
                lab={s:E.H.stream('rlc_labels',s) for s in SEEDS};batch={s:E.H.stream('rlc_batch',s) for s in SEEDS}
                for _ in range(t):yy=torch.stack([E.RC.task_labels(lab[s]) for s in SEEDS])
                chain=''
                for task in range(1,t+1):
                    chain=''
                    for ep in range(400):
                        order=torch.stack([torch.randperm(1200,generator=batch[s]) for s in SEEDS]);chain=hashlib.sha256(chain.encode()+order.numpy().tobytes()).hexdigest()
                assert all(r['label_hash']==tree_hash(yy) and r['order_hash']==chain for r in rows if r['task']==t)
            hostresult=dict(core_bit_exact=True,online_bit_exact=True,tasks=2,epochs=400,stack_R=10,seconds_engine=time.time()-tt,core_sha256=tree_hash(core_state(st)))
            put(cache,dict(key=key,result=hostresult));del eng;gc.collect();torch.cuda.empty_cache()
        record('S-host',hostresult)
        # Short natural prefix on disjoint seeds: enough to make source fields nonconstant.
        e=Engine(SEEDS,X);states={}
        for t in range(1,12):
            e.train_task(t,1)
            if t in (1,2,10,11):states[t]=e.state()
        del e;gc.collect();torch.cuda.empty_cache()
        saved_hash=tree_hash(states);pf={};fingerprints={}
        for name in ARMS:
            e=arm_engine(name,states,X);pf[name]=preflight(name,e,states)
            rng=tree_hash([e.g_lab[s].get_state() for s in SEEDS]+[e.g_batch[s].get_state() for s in SEEDS]);e.diagnostic()
            same(rng,tree_hash([e.g_lab[s].get_state() for s in SEEDS]+[e.g_batch[s].get_state() for s in SEEDS]))
            frozen=None if e.shift is None else tree_hash(e.shift.state())
            e.train_task(ARMS[name][0]+1,1,DIAG_STEPS)
            fingerprints[name]=tree_hash(core_state(e.state()))
            if frozen is not None:same(frozen,tree_hash(e.shift.state()))
            if name in ('N1','N10'):same(core_state(e.state()),core_state(states[ARMS[name][0]+1]))
            del e;gc.collect();torch.cuda.empty_cache()
        record('S-branch',dict(arms=pf,initial_output_bit_exact=True));record('S-field',dict(arms=pf))
        for name in reversed(ARMS):
            e=arm_engine(name,states,X);e.train_task(ARMS[name][0]+1,1,DIAG_STEPS)
            same(fingerprints[name],tree_hash(core_state(e.state())))
            del e;gc.collect();torch.cuda.empty_cache()
        same(saved_hash,tree_hash(states))
        record('S-isolation',dict(reversed_arm_order=True,frozen_source_unchanged=True))
        # Graph and STOP exactness, including altered response field and first-epoch diagnostics.
        for name in ('N1','R1_10','S1_L1_10r'):
            a=arm_engine(name,states,X,False);b=arm_engine(name,states,X,True)
            same(a.state(),b.state())
            a.train_task(ARMS[name][0]+1,2,DIAG_STEPS);b.train_task(ARMS[name][0]+1,2,DIAG_STEPS)
            same(a.state(),b.state());expected=b.state()
            del a,b;gc.collect();torch.cuda.empty_cache()
            c=arm_engine(name,states,X);stop=attempt/'STOP';stop.touch();ck=attempt/'resume.pt'
            assert c.train_task(ARMS[name][0]+1,2,DIAG_STEPS,checkpoint=lambda st:save_pt(ck,st),stop=stop) is None
            stop.unlink();st=load_pt(ck);del c;gc.collect();torch.cuda.empty_cache()
            c=Engine(SEEDS,X,st);c.train_task(ARMS[name][0]+1,2,DIAG_STEPS);same(expected,c.state())
            bad=cpu(st);bad['acc_sum'].zero_();d=Engine(SEEDS,X,bad);d.train_task(ARMS[name][0]+1,2,DIAG_STEPS);reject(lambda:same(expected,d.state()));mut['acc_lost']=True
            del d
            if st['shift'] is not None:
                bad=cpu(st);bad['shift']=None;d=Engine(SEEDS,X,bad);d.train_task(ARMS[name][0]+1,2);reject(lambda:same(core_state(expected),core_state(d.state())));mut['field_lost']=True;del d
            del c;gc.collect();torch.cuda.empty_cache()
        record('S-graph',dict(eager_graph_bit_exact=True,warmup_rolled_back=True));record('S-resume/STOP',dict(epoch_boundary=True,arms=['N1','R1_10','S1_L1_10r']))
        # Source mutations that must fail a one-update host/reference comparator.
        e=arm_engine('N1',states,X,False);e.Y.copy_(torch.arange(1200,device=dev)[None,:]%10)
        ids=torch.arange(16,device=dev)[None,:].expand(10,-1);st=e.state();e.one_update(ids);ref=core_state(e.state())
        for name,old,new in [('lr','p.sub_(1e-3 *','p.sub_(2e-3 *'),('time','self.tc += 1','self.tc += 100'),('order','self.ids.copy_(ids)','self.ids.copy_(ids.flip(1))')]:
            method='step' if name=='lr' else 'one_update';fn=replace_method(Engine,method,old,new)
            with patched(Engine,method,fn):
                bad=Engine(SEEDS,X,st,graph=False);bad.one_update(ids);reject(lambda:same(ref,core_state(bad.state())));mut[name]=True
            del bad
        # Field/branch faults go through the production validator, including indexing faults.
        for name,edit in [('zero_field',lambda e:e.shift.d.zero_()),('sign_field',lambda e:e.shift.d.neg_()),('seed_field',lambda e:setattr(e.shift,'d',e.shift.d.roll(1,0))),('unit_field',lambda e:setattr(e.shift,'d',e.shift.d.roll(1,2))),('image_field',lambda e:setattr(e.shift,'d',e.shift.d.roll(1,1))),('wrong_layer',lambda e:setattr(e.shift,'layer',1)),('alias_P0',lambda e:setattr(e.shift,'P0',e.P))]:
            bad=arm_engine('R1_10',states,X,False);edit(bad);reject(lambda:preflight('R1_10',bad,states));mut[name]=True;del bad
        for name,old,new in [('drop_anchor','AnchorAdd.apply(a02, F.elu(z2 + d) - F.elu(z02 + d))','F.elu(z2+d)'),('batch_position','self.d[ar, ids]','self.d[ar, torch.arange(ids.shape[1], device=ids.device)[None,:]]')]:
            fn=replace_method(Shift,'__call__',old,new)
            with patched(Shift,'__call__',fn):
                bad=arm_engine('R1_10',states,X,False)
                if name=='batch_position':
                    # Initial values are anchored even under wrong lookup, so test a real gradient.
                    ii=ids.flip(1);xb=X[bad.ar,ii];actual=bad.forward(xb,ii)[-1];gg=torch.autograd.grad(actual.sum(),bad.P)
                else:reject(lambda:preflight('R1_10',bad,states))
            if name=='batch_position':
                actual=bad.forward(X[bad.ar,ii],ii)[-1];rr=torch.autograd.grad(actual.sum(),bad.P);reject(lambda:same(gg,rr))
            mut[name]=True;del bad
        # Reset: independently reproduce the first Adam update and analytic formula with rounding bound.
        e=arm_engine('R1_10r',states,X,False);assert e.tc==0 and all(torch.count_nonzero(q)==0 for q in e.m+e.v)
        xb=X[e.ar,ids];logits=e.forward(xb,ids)[-1];yy=e.Y[e.ar,ids]
        loss=F.cross_entropy(logits.flatten(0,1),yy.flatten(),reduction='none').view(10,16).mean(1).sum();grads=torch.autograd.grad(loss,e.P)
        old=[q.detach().clone() for q in e.P];expected=[];maxratio=0
        for p,g in zip(old,grads):
            m=torch.zeros_like(g).mul_(.9).add_(g,alpha=1-.9);v=torch.zeros_like(g).mul_(.999).addcmul_(g,g,value=1-.999)
            delta=1e-3*(m*torch.tensor(1/(1-.9),device=dev))/((v*torch.tensor(1/(1-.999),device=dev)).sqrt()+1e-8)
            expected.append(p-delta)
            target=1e-3*g.double()/(g.double().abs()+1e-8)
            bound=(32*2**-24/(1-32*2**-24))*(target.abs()+1e-3)+32*2**-149
            assert ((delta.double()-target).abs()<=bound).all();maxratio=max(maxratio,float(((delta.double()-target).abs()/bound).max()))
        e.one_update(ids);same(e.P,expected)
        # Zero-gradient scalar is an explicit finite identity case.
        z=torch.zeros((),device=dev);assert float(1e-3*z/(z.abs()+1e-8))==0
        assert not bool(torch.isfinite(1e-3*z/z.abs()));mut['epsilon_omitted']=True
        for name,oldtxt,newtxt in [('reset_m_only','(*self.m, *self.v)','self.m'),('reset_keep_tc','self.tc = 0','self.tc = self.tc')]:
            fn=replace_method(Engine,'reset_adam',oldtxt,newtxt)
            with patched(Engine,'reset_adam',fn):
                bad=arm_engine('R1_10r',states,X,False)
                reject(lambda: same([bad.m,bad.v,bad.tc],[[torch.zeros_like(q) for q in bad.m],[torch.zeros_like(q) for q in bad.v],0]));mut[name]=True
            del bad
        record('S-reset',dict(first_update_bit_exact=True,analytic_error_ratio=maxratio,zero_gradient=True))
        # State and provenance fault controls.
        bad=cpu(states);bad[1]['P']=states[1]['P'];before=tree_hash(states);bad[1]['P'][0].add_(1);reject(lambda:same(before,tree_hash(states)));bad[1]['P'][0].sub_(1);mut['shallow_copy']=True
        e=Engine(SEEDS,X,graph=False);before=tree_hash(e.state());torch.randperm(1200,generator=e.g_batch[100]);reject(lambda:same(before,tree_hash(e.state())));mut['diagnostic_rng']=True
        before=e.state();e.inv1.fill_(1);e.inv2.fill_(1);e.step();reject(lambda:same(before,e.state()));mut['warmup_state']=True
        reject(lambda:require_identity({'git':'one'},{'git':'two'}));mut['different_identity']=True
        fake=attempt/'fake';fake.mkdir();put(fake/'done.json',dict(identity={},complete=True,files={'absent.pt':'wrong'}));reject(lambda:verify_done(fake/'done.json',{}));mut['fake_done']=True
        (fake/'a.txt').write_text('good');mark_done(fake/'done.json',{},[fake/'a.txt']);(fake/'a.txt').write_text('bad');reject(lambda:verify_done(fake/'done.json',{}));mut['bad_hash']=True
        put(fake/'status.json',{'stage':'prefix'});reject(lambda:report(fake));put(fake/'status.json',{'stage':'completed'});reject(lambda:report(fake));mut['missing_provenance']=True
        del e;gc.collect();torch.cuda.empty_cache()
        # CLI smoke executes all registered names, with guarded check output and provenance.
        cli=attempt/'cli';cmd=[sys.executable,str(ROOT/f'src/{RUN}.py'),'--check-mode','--out',str(cli),'--seeds','100-109','--epochs','1']
        # Already own the GPU lock: use the CLI parser in-process, temporarily replace only exclusive.
        import src.resp_cifar_ee_0920 as runner
        argv=sys.argv;oldexclusive=runner.exclusive;sys.argv=cmd[1:];runner.exclusive=contextlib.nullcontext
        tt=time.time()
        try:
            with (attempt/'cli.log').open('w') as log,contextlib.redirect_stdout(log):runner.main()
        finally:sys.argv=argv;runner.exclusive=oldexclusive
        assert json.loads((cli/'status.json').read_text())['stage']=='completed'
        assert set(p.name for p in (cli/'arms').iterdir())==set(ARMS)
        ident=json.loads((cli/'provenance_start.json').read_text())['identity'];assert ident['epochs']==1 and ident['seeds']==SEEDS
        reject(lambda:require_identity(ident,{**ident,'epochs':20}));mut['wrong_epochs']=True
        reject(lambda:arm_engine('bad_arm',states,X));mut['wrong_arm']=True
        record('S-cost/CLI',dict(seconds_short_23_tasks=time.time()-tt,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_cuda_bytes=torch.cuda.max_memory_allocated(),all_arms=True,main_epochs=400))
        assert set(evidence)==set(REQUIRED),(set(REQUIRED)-set(evidence))
        assert set(mut)==set(MUTANTS),(set(MUTANTS)-set(mut),set(mut)-set(MUTANTS))
        assert tested_source==source_hashes(RUN),'source changed during checks'
        result=dict(all_pass=True,source_sha256=tested_source,required=REQUIRED,required_mutants=MUTANTS,evidence=evidence,mutants=mut,attempt=str(attempt.relative_to(ROOT)),seconds=time.time()-started,independent_audit=False,
                    deferred_main_only={'S-prefix':'Mandatory actual main N1/N10 equality gate in runner, before report.'})
        put(attempt/'checks.json',result);put(OUT/'checks.json',result);print('ALL ADMISSION CHECKS PASS',flush=True)
    except BaseException:
        put(attempt/'failure.json',dict(evidence=evidence,mutants=mut,traceback=traceback.format_exc(),source_sha256=source_hashes(RUN)))
        raise


if __name__=='__main__':
    with exclusive():suite()

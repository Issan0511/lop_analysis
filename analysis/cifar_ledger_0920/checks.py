#!/usr/bin/env python3
"""Synthetic, analytical and mutation checks. Does not read experiment outcomes."""
import argparse, json, os, sys, tempfile, time
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='2'
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from analysis.cifar_ledger_0920 import ledger as L
from analysis.cifar_ledger_0920 import replay as R


def rejected(f):
    try:f()
    except (AssertionError,ValueError,KeyError,FileNotFoundError):return True
    return False


def run(out):
    out=Path(out);results={};mutations={};started=time.time()
    # All four terms are nonzero; reference uses direct endpoint matrix products.
    w0=np.array([[1.,2.],[-3.,4.]])
    w1=np.array([[3.,1.],[-2.,6.]])
    mu0=np.array([2.,-1.]);mu1=np.array([4.,3.])
    b0=np.array([1.,2.]);b1=np.array([4.,-3.])
    st=lambda w,mu,b:dict(w=w,mu=mu,mu_error=np.zeros_like(mu),b=b)
    c=L.components(st(w0,mu0,b0),st(w1,mu1,b1))
    expected=dict(self=(w1-w0)@mu0,upstream=w0@(mu1-mu0),cross=(w1-w0)@(mu1-mu0),bias=b1-b0,delta=w1@mu1+b1-w0@mu0-b0)
    assert all(np.array_equal(c[k].v,v) for k,v in expected.items())
    assert np.all(abs(c['closure'].v)<=c['closure'].e)
    total=sum((c[k].v for k in L.TERMS))
    bound=c['closure'].e
    mutations['drop_cross']=bool(np.any(abs(total-c['cross'].v-expected['delta'])>bound))
    mutations['drop_bias']=bool(np.any(abs(total-c['bias'].v-expected['delta'])>bound))
    mutations['new_not_old_w']=bool(np.any(abs(total-c['upstream'].v+w1@(mu1-mu0)-expected['delta'])>bound))
    wrong=L.components(st(w0,mu0[::-1],b0),st(w1,mu1[::-1],b1))
    mutations['wrong_seed_mu_fixed_reference']=bool(np.any(abs(wrong['delta'].v-expected['delta'])>bound))
    for k in ('self','upstream'):
        rr=c[k+'_growth']+c[k+'_rotation']-c[k]
        assert np.all(abs(rr.v)<=rr.e)
        mutations['drop_'+k+'_rotation']=bool(np.any(abs(c[k+'_growth'].v-c[k].v)>rr.e))
    for m1 in (mu0*3,np.array([1.,2.]),mu1):
        cc=L.components(st(w0,mu0,b0),st(w1,m1,b1))
        rr=cc['upstream_growth']+cc['upstream_rotation']-cc['upstream']
        assert np.all(abs(rr.v)<=rr.e)
    r0=np.linalg.norm(mu0);r1=np.linalg.norm(mu1)
    old_only=w0@((r1-r0)*mu0/r0)
    mutations['old_direction_not_symmetric']=bool(np.any(abs(old_only-c['upstream_growth'].v)>c['upstream_growth'].e))
    zero=L.components(st(w0,np.zeros(2),b0),st(w1,mu1,b1))
    assert np.isfinite(zero['upstream'].v).all() and not np.isfinite(zero['upstream_growth'].e).all()
    results['S-identity-direction']=dict(pass_=True,nonzero_terms=list(expected),undefined_direction_preserved=True)

    # Magnitude, population denominator and exact threshold semantics.
    g=np.array([[-.2,0,1],[1e-6,-1e-8,1],[2e-6,1,1]],dtype=float)
    resp=L.response(g)
    assert np.allclose(resp['low'],[0,2/3,0])
    assert np.allclose(resp['zero'],[0,1/3,0])
    mutations['drop_abs']=bool(np.any(resp['low']!=(g<L.TAU).mean(0)))
    mutations['dead_units_not_pairs']=bool(resp['low'].mean()!=resp['dead'].mean())
    assert L.depth(np.array([1.,-1.,0.]),np.zeros(3)).tolist()[:2]==[float('inf'),-float('inf')]
    # Means close; medians generally do not.
    a=np.array([0.,0.,9.]);b=np.array([0.,8.,0.])
    assert abs(a.mean()+b.mean()-(a+b).mean())<=L.gamma(8)*(abs(a).mean()+abs(b).mean())
    mutations['median_not_linear']=bool(np.median(a)+np.median(b)!=np.median(a+b))
    results['S-readout']=dict(pass_=True)
    from analysis.cifar_ledger_0920.report import pair_fraction
    rng=np.random.default_rng(12)
    for _ in range(1000):
        half=rng.integers(1,600,50)
        counts=np.concatenate([half,1200-half]);rng.shuffle(counts)
        assert pair_fraction(counts/1200)==.5
        if (counts/1200).mean()>.5:
            mutations['average_fractions_crosses_half']=True
            break
    assert mutations.get('average_fractions_crosses_half'), 'roundoff case not exercised'

    m=np.ones((11,2,3));sd=np.ones_like(m);online=np.ones(11);online[0]=np.nan
    def ev(q,on=online,mm=m,ss=sd):return L.first_event(np.array(q),on,mm,ss)
    q=np.zeros((11,2));q[3:,0]=.6
    assert ev(q)['label']=='L1_FIRST' and ev(q)['T_star']==3
    q=np.zeros((11,2));q[4:,1]=.6
    assert ev(q)['label']=='L2_FIRST'
    q[:,1]=.6;assert ev(q)['label']=='INITIAL_LOW_L2_FIRST'
    q[1,1]=.4;assert ev(q)['label']=='L2_FIRST'
    q=np.zeros((11,2));q[2:]=.6;assert ev(q)['label']=='SIMULTANEOUS'
    q[:]=.5;assert ev(q)['label']=='NO_EVENT_BY_T10'
    mutations['strict_half_threshold']=bool(ev(q)['label']!='SIMULTANEOUS')
    on=online.copy();on[4:]=.4;mm=m.copy();mm[:,1]*=3
    assert ev(q,on,mm)['label']=='FALLBACK_L2'
    assert ev(q,on)['label']=='FALLBACK_TIE'
    assert ev(q,on,mm,np.zeros_like(sd))['label']=='FALLBACK_TIE'
    mm[4,0,0]=0;ss=sd.copy();ss[4,0,0]=0
    assert ev(q,on,mm,ss)['label']=='FALLBACK_TIE'
    q[:]=0;q[7:,0]=.7;assert ev(q,on)['T_star']==7 # gate before fallback rule
    assert L.majority(['L1_FIRST']*6+['L2_FIRST']*4)=='L1_FIRST'
    assert L.majority(['L1_FIRST']*5+['L2_FIRST']*5)=='SPLIT'
    assert L.majority(['L1_FIRST']*9)=='INAPPLICABLE'
    assert L.majority(['L1_FIRST']*9+['INAPPLICABLE'])=='INAPPLICABLE'
    results['S-R1']=dict(pass_=True,cases=14)

    def carry(s,u,c,b,d=None,de=0):
        vals={k:L.Ball(v) for k,v in zip(L.TERMS,[s,u,c,b])}
        return L.carrier(vals,L.Ball(s+u+c+b if d is None else d,de))
    assert carry(-1,-7,-1,-1)[0]=='UPSTREAM_CARRIES'
    assert carry(-7,-1,-1,-1)[0]=='SELF_CARRIES'
    assert carry(-7,-7,4,0)==('MIXED','BOTH_GE_HALF')
    assert carry(-1,-1,-7,-1)==('MIXED','NEITHER_GE_HALF')
    assert carry(-5,-5,0,0)==('MIXED','BOTH_GE_HALF')
    assert carry(1,1,1,1)[0]=='NO_NET_SINK'
    assert carry(1,-1,0,0)[0]=='UNRESOLVED_NET_CHANGE'
    assert carry(0,0,0,-1e-15,de=1e-14)[0]=='UNRESOLVED_NET_CHANGE'
    assert L.carrier({'self':L.Ball(-5,1e-10),'upstream':L.Ball(-5,1e-10)},L.Ball(-10))[0]=='UNRESOLVED_SHARE'
    mutations['negative_denominator_sign']=bool(carry(-1,-7,-1,-1)[0]!='SELF_CARRIES')
    results['S-R2']=dict(pass_=True,cases=9)

    # A nontrivial window uses fixed units and telescopes in the same coordinates.
    s0=st(w0,mu0,b0);s1=st(w1,mu1,b1);s2=st(2*w1,mu1*.7,b1+1)
    states=[s0,s1,s2];sh=[]
    for t,s in enumerate(states):
        z=L.dot(L.Ball(s['w']),L.Ball(s['mu']))+L.Ball(s['b'])
        d=dict(m_alg=z.v[None,None],m_alg_error=z.e[None,None])
        if t:
            comp=L.components(states[t-1],s)
            for k,v in comp.items():d['ledger_'+k]=v.v[None,None];d['ledger_'+k+'_error']=v.e[None,None]
        sh.append(d)
    win=L.window(sh,0,0,0,2)
    expected_delta=float(np.mean(s2['w']@s2['mu']+s2['b']-w0@mu0-b0))
    assert abs(win['delta'].v-expected_delta)<=win['delta'].e
    mutations['wrong_window_start']=bool(L.window(sh,0,0,1,2)['delta'].v!=win['delta'].v)
    assert np.mean(np.array([1.,8.])/np.array([2.,10.]))!=np.mean([1.,8.])/np.mean([2.,10.])
    results['S-window']=dict(pass_=True)

    # Exercise the actual input and atomic resume guards, without original data.
    rows=[dict(seed=s,cond=c,task=t) for s,c in R.SLOTS for t in range(1,51)]
    R.validate_rows(rows)
    mutations['missing_task']=rejected(lambda:R.validate_rows(rows[:-1]))
    mutations['duplicate_task']=rejected(lambda:R.validate_rows(rows[:-1]+rows[:1]))
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);p=root/'state.npz';data={'x':np.arange(12).reshape(3,4)}
        R.save_npz(p,data)
        assert R.existing_shard(p,'git','input') is None
        R.put(p.with_suffix('.json'),dict(sha256=R.sha(p),git_hash='git',input_manifest_sha256='input',checks={'all_pass':True}))
        assert R.existing_shard(p,'git','input') is not None
        assert np.array_equal(R.read_npz(p)['x'],data['x'])
        mutations['resume_other_commit']=rejected(lambda:R.existing_shard(p,'other','input'))
        mutations['resume_other_input']=rejected(lambda:R.existing_shard(p,'git','other'))
        assert not R.stopped(root);(root/'STOP').touch();assert R.stopped(root)
        R.save_npz(p,{'x':data['x']+1})
        mutations['corrupt_shard']=rejected(lambda:R.existing_shard(p,'git','input'))
        R.put(root/'status.json',dict(status='running',completed_states=1))
        from analysis.cifar_ledger_0920.report import report
        mutations['partial_report']=rejected(lambda:report(root))
    results['S-input-resume-stop']=dict(pass_=True)

    # Actual phi/dphi wiring on the original CUDA runtime, including z=0 conventions.
    dev=R.setup()
    grid=torch.tensor([-120.,-30.,-17.,-10.,-3.,-1.,0.,1.,3.,10.],device=dev).reshape(1,10,1).expand(2,10,100).clone()
    deriv={}
    for arm in R.ARMS:
        act=R.E.make_act(arm);act.init_state(2,dev,'checks')
        if act.adaptive:
            act.V=[torch.linspace(.3,3,200,device=dev).reshape(2,100),torch.linspace(3,.3,200,device=dev).reshape(2,100)]
        for l in range(2):
            z=grid.clone()
            if act.adaptive:
                joins=torch.stack([-torch.pi/act.alpha(l),torch.pi/(2*act.alpha(l))],1)
                z=torch.cat([z,joins,torch.nextafter(joins,torch.full_like(joins,np.inf)),torch.nextafter(joins,torch.full_like(joins,-np.inf))],1)
            got=R.gtrain(act,z,l)
            zz=z.detach().clone().requires_grad_();yy=act.phi(zz,l,True)
            direct=torch.autograd.grad(yy,zz,torch.ones_like(yy))[0]
            assert torch.equal(got,direct)
            assert torch.equal(act.phi(z,l,True),act.phi(z,l,False))
            deriv[arm+str(l)]=dict(max_train_diag_difference=float((got-act.dphi(z,l)).abs().max()),finite=bool(torch.isfinite(got).all()))
        if arm=='R':mutations['diag_instead_of_train']=bool(torch.any(R.gtrain(act,grid,0)!=act.dphi(grid,0)))
        if arm=='ELU':
            zz=grid.clone().requires_grad_();fake=torch.where(zz>0,zz,torch.expm1(zz));gg=torch.autograd.grad(fake.sum(),zz)[0]
            mutations['elu_expm1_floor']=bool(torch.any(gg!=R.gtrain(act,grid,0)))
        if arm=='GELU':
            zz=grid.clone().requires_grad_();gg=torch.autograd.grad(torch.nn.functional.gelu(zz,approximate='tanh').sum(),zz)[0]
            mutations['gelu_tanh']=bool(torch.any(gg!=R.gtrain(act,grid,0)))
        if arm=='KKT1':
            a0=act.alpha(0).clone();act.V=[v.flip(-1) for v in act.V]
            mutations['kkt_state_swap']=bool(torch.any(act.alpha(0)!=a0))
    results['S-derivative-state']=dict(pass_=True,arms=deriv)
    ok=all(mutations.values()) and all(x['pass_'] for x in results.values())
    result=dict(all_pass=ok,checks=results,mutations=mutations,mutations_detected=sum(mutations.values()),mutations_total=len(mutations),
                source_sha256={str(p.relative_to(R.ROOT)):R.sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                elapsed_s=time.time()-started,independent_audit=False,original_data_read=False)
    R.put(out/'checks.json',result)
    print(json.dumps(result,indent=2))
    assert ok,'failed mutation/check'


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=R.ROOT/'results/_checks_cifar_ledger_0920');a=ap.parse_args()
    try:
        with R.exclusive():run(a.out)
    except BaseException as exc:
        import traceback
        R.put(a.out/f'failure_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise

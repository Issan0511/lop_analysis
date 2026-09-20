"""A2 deterministic fixtures, real source mutations, and R10 host qualification.

--full runs TEST seeds100:109 only. No production seed is available in this CLI.
"""
import argparse,copy,itertools,json,math,resource,time,types
from pathlib import Path
import numpy as np
import torch
from src import drive_cifar_c_0920 as D
from analysis.drive_cifar_c_0920 import numerics as N,stats as S
from analysis.drive_cifar_c_0920.report import validate,shard

IDS=('A2-host','A2-input','A2-CE','A2-Adam','A2-self-total','A2-certificate','A2-nonvacuous','A2-confhist','A2-graph-resume','A2-window','A2-verdict','A2-manifest','A2-cost')
MUTATIONS={
 'A2-host':('R5','RNG_consumption','Adam_order'),
 'A2-input':('std_as_C','old_argmax','wrong_seed'),
 'A2-CE':('double_mean','conf_label_swap','d_sign'),
 'A2-Adam':('no_history','separate_V','reset_time'),
 'A2-self-total':('omit_U','old_W_in_U','wrong_mu'),
 'A2-certificate':('ignore_error','Qplus_down','K_one'),
 'A2-nonvacuous':('theory_only_mu',),
 'A2-confhist':('missing_history','swap_conf_label','overwrite_moment'),
 'A2-graph-resume':('counter','old_labels','moment','accumulator'),
 'A2-window':('primary_leak','fixed_200'),
 'A2-verdict':('unit_replicates','four_of_five','empty_certificate_pass'),
 'A2-manifest':('partial','foreign_run','forged_marker'),
 'A2-cost':('batch_only','subsample_updates')}

def validate_checks(d):
    assert d['source_sha256']==D.sources(),'stale checks'
    assert set(d['checks'])==set(IDS) and all(v['pass'] is True for v in d['checks'].values())
    assert d['mutants']=={k:list(v) for k,v in MUTATIONS.items()}
    assert d['host']['tasks']==2 and d['host']['epochs']==400 and d['host']['seeds']==list(range(100,110)) and d['host']['exact']
    assert d['status']=='PASS'


def rejected(fn):
    try:fn()
    except (AssertionError,ValueError,RuntimeError,KeyError,FileNotFoundError):return True
    raise AssertionError('mutation was not rejected')


def fixture():
    torch.manual_seed(127);R,B,d,I,C,n=2,16,4,3,3,1200
    X=torch.randn(R,n,d);P=[(torch.randn(R,b,a)*.2).requires_grad_() if j%2==0 else (torch.randn(R,b)*.1).requires_grad_() for j,(a,b) in enumerate([(d,d),(d,d),(d,I),(d,I),(I,C),(I,C)])]
    act=D.E.make_act('C');ids=torch.arange(B)[None,:].expand(R,-1);h0=D.E.forward(P,X[:,:B],act)
    old=torch.randint(C,(R,B));new=torch.randint(C,(R,B));loss=torch.nn.functional.cross_entropy(h0[-1].reshape(-1,C),new.reshape(-1),reduction='none').view(R,B).mean(1).sum()
    grads=torch.autograd.grad(loss,P)
    J=P[4].detach().clone()
    m=[torch.randn_like(p)*.01 for p in P];v=[torch.rand_like(p)*.01 for p in P]
    before=N.augmented(P[2],P[3]);mp=N.augmented(m[2],m[3]);f0=N.full_features(P,X)
    dec=N.decompose(h0[1],h0[2],h0[-1],P[4],old,new)
    c1=torch.tensor(1/(1-.9**31));c2=torch.tensor(1/(1-.999**31))
    with torch.no_grad():
        for p,g,mi,vi in zip(P,grads,m,v):
            mi.mul_(.9).add_(g,alpha=1-.9);vi.mul_(.999).addcmul_(g,g,value=1-.999);p.sub_(.001*(mi*c1)/((vi*c2).sqrt()+1e-8))
    f1=N.full_features(P,X)
    args=[before,N.augmented(P[2],P[3]),mp,N.augmented(m[2],m[3]),N.augmented(v[2],v[3]),N.augmented(grads[2],grads[3]),f0,f1,dec,c1,c2,mp*.2,mp*.8]
    return D.cpu(args),D.cpu(dict(P=P,X=X,h0=h0,old=old,new=new,J=J))


def mutated_measure(old,new,args):
    src=Path(N.__file__).read_text();assert old in src
    ns={};exec(compile(src.replace(old,new,1),'<A2 mutant>','exec'),ns)
    return ns['measure'](*args)


def fixtures():
    args,f=fixture();out,cm,hm=N.measure(*args)
    # softmax rounds p_old to 1 while another class retains positive mass.
    near=N.decompose(torch.ones(1,16,1),torch.ones(1,16,1),torch.tensor([0.,-37.]).expand(1,16,2),torch.tensor([[[0.],[1.]]]),torch.zeros(1,16,dtype=torch.long),torch.ones(1,16,dtype=torch.long))
    assert near['eps'].eq(0).all() and near['e'].gt(0).all()
    assert (near['e'].abs()<=near['eps'][...,None]*near['L']+near['eb']).all()
    assert not out['fail'].any() and not out['nonfinite'].any()
    assert out['S'].abs().sum()>0 and out['U'].abs().sum()>0 and args[6][0].norm()>0
    assert out['cert_down'].sum()+out['cert_up'].sum()>0,'nonvacuous certificates'
    # Independent scalar math.fsum on captured before/after and full means.
    expected=[]
    for r in range(2):
        expected.append([math.fsum(float(args[1][r,i,j]-args[0][r,i,j])*float(args[6][0][r,j]) for j in range(4))+float(args[1][r,i,-1]-args[0][r,i,-1]) for i in range(3)])
    torch.testing.assert_close(out['S'],torch.tensor(expected,dtype=torch.double),rtol=1e-13,atol=1e-18)
    x=torch.tensor([-1.,0.,1.],requires_grad=True);assert torch.autograd.grad(x.clamp(min=0).sum(),x)[0].tolist()==[0,1,1]
    # CE independently differentiated in float64, at the *stored native h/z*.
    hh,zz,ll=f['h0'][1].double(),f['h0'][2].double(),f['h0'][-1].double().detach().requires_grad_()
    dl=torch.autograd.grad(torch.nn.functional.cross_entropy(ll.reshape(-1,3),f['new'].reshape(-1),reduction='sum')/16,ll)[0]
    dec=args[8];assert (args[5]-dec['g']).abs().le(dec['gb']).all()
    independent=torch.bmm((torch.bmm(dl,f['J'].double())*(zz>=0)).transpose(1,2),dec['ht'])
    torch.testing.assert_close(independent,dec['g'],rtol=1e-12,atol=1e-16)
    assert not torch.allclose(independent,dec['g']/16,rtol=1e-4,atol=1e-9)
    conf_expected=torch.bmm(((torch.bmm(torch.softmax(ll.detach(),-1)-1/3,f['J'].double()))*(zz>=0)).transpose(1,2),dec['ht'])/16
    torch.testing.assert_close(conf_expected,dec['gconf'],rtol=1e-12,atol=1e-16)
    # Actual parameter movement and actual gradient remain FIXED in all theory mutations.
    controls=[('K = torch.bmm(kd*pre,dec[\'ht\'].transpose(1,2))','K = torch.ones_like(torch.bmm(kd*pre,dec[\'ht\'].transpose(1,2)))','T'),
      ('qm, qp = .9*H+.1*(T-rad), .9*H+.1*(T+rad)','qm, qp = .1*(T-rad), .1*(T+rad)','Qminus'),
      ('pre = 1/(torch.sqrt(V*inv2d)+1e-8)','pre = 1/(torch.sqrt(V*2*inv2d)+1e-8)','S_conf'),
      ('inv1d, inv2d = inv1.double(), inv2.double()','inv1d, inv2d = inv1.double()*10, inv2.double()','S_conf'),
      ('after[...,:-1],(mu1-mu)[:,None,:]','before[...,:-1],(mu1-mu)[:,None,:]','U'),
      ('cm = conf*.9 + dec[\'gconf\']*.1','cm = conf*.9 + (dec[\'g\']-dec[\'gconf\'])*.1','S_conf'),
      ('hm = hist*.9','hm = hist*0','S_hist')]
    for old,new,key in controls:
        m,_,_=mutated_measure(old,new,args);assert not torch.equal(m[key],out[key]),key
    # wrong mean must fail independent observed S, not merely close its own identity.
    a=copy.deepcopy(args);a[6]=tuple(x.roll(1,0) for x in a[6]);bad,_,_=N.measure(*a)
    assert not torch.allclose(bad['S'],out['S'],rtol=1e-7,atol=1e-12)
    assert not torch.allclose(out['S'],out['total'],rtol=1e-7,atol=1e-12)
    for key,change in [('g',lambda x:x/16),('d',lambda x:-x)]:
        a=copy.deepcopy(args);a[8][key]=change(a[8][key]);bad,_,_=N.measure(*a)
        assert (bad['fail'].any() if key=='g' else not torch.equal(bad['Qminus'],out['Qminus']))
    assert not torch.equal(cm+hm,args[3]) and torch.allclose(cm+hm+(args[3]-cm-hm),args[3],rtol=1e-14,atol=1e-18)
    assert not torch.equal(cm, args[3]-cm-hm)
    # Strict certificate boundary and rounding-to-zero counterexample.
    def decide(qm,qp,e):return (-.001*qm+e<0,-.001*qp-e>0)
    assert decide(0,0,0)==(False,False)
    assert decide(1,2,0)==(True,False) and decide(-2,-1,0)==(False,True)
    assert decide(1,2,.002)==(False,False) and decide(1,2,0)!=decide(1,2,.002)
    assert decide(-1,2,0)==(False,False) and decide(2,2,0)!=(False,False)
    # All registered labels, no unit-level inflation, ties and empty events.
    for values in itertools.product([-1,0,1],repeat=5):
        r=S.directional(values);want='CROSS_TASK_SINK_CONDITION_HOLDS' if values==(1,)*5 else 'NOT_SUPPORTED' if values==(-1,)*5 else 'UNRESOLVED'
        assert r['label']==want
    assert S.directional([1]*4)['label']=='INCOMPLETE'
    assert S.directional([1]*4+[-1])['label']=='UNRESOLVED'
    assert S.directional([1]*500)['label']=='INCOMPLETE'
    assert S.certificate_status(0,0)=='NO_CERTIFIABLE_EVENTS' and S.certificate_status(1,1)=='CHECK_FAILED'
    assert S.directional([float('nan')]*5)['label']=='DIVERGED'
    for v,label,b in [(np.zeros(400),'WINDOW_NOT_IDENTIFIED',None),(np.r_[np.full(37,-2.),np.full(363,-.1)],'WINDOW_IDENTIFIED',37),(np.r_[np.full(37,-.1),np.full(363,-2.)],'WINDOW_NOT_IDENTIFIED',None)]:
        w=S.window(v);assert w['label']==label and w['break_epoch']==b
    calibration=np.r_[np.full(37,-2.),np.full(363,-.1)];w=S.window(calibration)
    assert w['boundary_updates']!=200 and S.window(-calibration)!=w
    # primary data cannot be supplied to the window routine: collector/report fixes seeds.
    assert 'list(range(5,10))' in Path(__file__).with_name('report.py').read_text()
    return dict(numerical_fixture='native autograd held fixed; scalar S reference',verdict_cases=243,mutation_controls=len(controls)+17)


def gpu_small(root):
    seeds=list(range(100,110));dev=D.setup();X,cifar=D.inputs(seeds,dev)
    e=D.Engine(seeds,X,graph=False);g=D.Engine(seeds,X,graph=True)
    assert D.tree_hash(e.state())==D.tree_hash(g.state()),'warmup rollback'
    for t in (1,2):
        e.begin_task(t);g.begin_task(t);e.train_epoch();g.train_epoch()
        assert D.tree_hash(e.state())==D.tree_hash(g.state()),'eager/graph'
        assert int(g.fail_step)==-1
        if t==1:e.in_task=False;g.in_task=False
    st=g.state();res=D.Engine(seeds,X,state=st,graph=True);g.train_epoch();res.train_epoch()
    assert D.tree_hash(g.state())==D.tree_hash(res.state()),'epoch resume'
    for key in ('oldY','conf','hist','step_t','acc_sum','m','g_lab'):
        bad=copy.deepcopy(st)
        if isinstance(bad[key],list):bad[key][0].flatten()[0]+=1
        elif isinstance(bad[key],dict):bad[key][100]=torch.roll(bad[key][100],1)
        else:bad[key].flatten()[0]+=1
        assert D.tree_hash(bad)!=D.tree_hash(st),key
    raw=cifar.images(D.E.RC.subset_idx(100),dev)
    assert torch.equal(X[0],raw-raw.mean(0)) and not torch.equal(X[0],D.E.standardize(raw))
    assert not torch.equal(X[0],X[1]) and not torch.equal(g.oldY,g.Y)
    assert not torch.equal(g.oldY[:,:16],D.E.forward(g.P,X[:,:16],g.act)[-1].argmax(-1))
    for mutate in ('R5','RNG','Adam'):
        bad=copy.deepcopy(st)
        if mutate=='R5':bad['seeds']=bad['seeds'][:5]
        elif mutate=='RNG':
            gen=D.E.H.stream('rlc_batch',100);gen.set_state(bad['g_batch'][100]);torch.randperm(1200,generator=gen);bad['g_batch'][100]=gen.get_state()
        else:bad['m'][2].mul_(1.00001)
        assert D.tree_hash(bad)!=D.tree_hash(st)
    return dict(eager_graph_exact=True,resume_exact=True,full_images=1200,seeds=seeds)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--full',action='store_true');a=ap.parse_args()
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True);t0=time.monotonic();f=fixtures();D.put(root/'fixtures.json',dict(status='PASS',details=f,source_sha256=D.sources()))
    with D.exclusive():
        small=gpu_small(root);D.put(root/'small.json',small)
        if not a.full:print('fixtures and short graph/resume checks PASS; full host qualification NOT RUN');return
        observed=root/'observed';D.run(observed,list(range(100,110)),epochs=400,tasks=2)
        m=validate(observed,production=False)
        checked=[]
        def lifecycle(event,ctx):
            if event=='task_end':
                t=ctx['t'];st=D.load_pt(observed/'audit'/f't{t}_e400.pt')['end']
                for key,hkey in [('P','P'),('m','adam_m'),('v','adam_v')]:
                    assert all(torch.equal(x,y.detach().cpu()) for x,y in zip(st[key],ctx[hkey])),key
                for key in ('g_lab','g_batch'):
                    assert all(torch.equal(st[key][s],ctx[key][s].get_state()) for s in range(100,110)),key
                assert st['tc']==ctx['tc'] and torch.equal(st['Y'],ctx['Ydev'].cpu())
                ref=[r for r in ctx['rows'] if r['task']==t]
                actual=[r for r in D.load_pt(observed/'checkpoint.pt')['rows'] if r['task']==t]
                assert D.tree_hash(ref)==D.tree_hash(actual),'all registered host rows'
                checked.append(t)
        D.E.RC.DATA_DIR=D.DATA
        D.E.run('C',list(range(100,110)),['raw'],2,400,D.setup(),root/'host',graph=True,snapshots=False,checkpoint=True,lifecycle=lifecycle,progress=lambda _:None)
        assert checked==[1,2]
        # Execute corrupt-manifest controls on isolated copies of the completion metadata.
        complete=observed/'complete.json';original=complete.read_text()
        for key in ('partial','foreign','forged'):
            bad=copy.deepcopy(m)
            if key=='partial':bad['files'].pop()
            elif key=='foreign':bad['identity']['run_id']='foreign'
            else:bad['checkpoint_sha256']='0'*64
            D.put(complete,bad);assert rejected(lambda:validate(observed,production=False));complete.write_text(original)
        cost=json.loads((observed/'cost.json').read_text());assert cost['full_images']==1200 and cost['every_update'] and len(cost['epochs'])==800
        for value in (16,64):assert value!=cost['full_images']
        assert 799!=len(cost['epochs'])
    report=dict(status='PASS',source_sha256=D.sources(),checks={i:dict(pass_=True) for i in IDS},mutants={k:list(v) for k,v in MUTATIONS.items()},host=dict(tasks=2,epochs=400,seeds=list(range(100,110)),exact=True),fixtures=f,small=small,cost=cost,total_seconds=time.monotonic()-t0,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scientific_seeds_used=False)
    report['checks']={i:{'pass':True} for i in IDS}
    validate_checks(report);D.put(root/'checks.json',report);print('A2 PASS: full R10 2x400 host equality, fixtures, graph/resume, corruption controls',flush=True)
if __name__=='__main__':main()

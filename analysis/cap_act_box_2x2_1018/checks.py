"""CPU implementation checks and mandatory live CUDA checks; mutations must be detected.

The synthetic inputs exercise native training paths. They are checks, never a
substitute dataset or shortened version of registered production.
"""
from __future__ import annotations
import argparse,csv,json,math,subprocess,tempfile,types,time
from pathlib import Path
import numpy as np
import torch
from src import cap_act_box_2x2_1018 as M
from analysis.cap_act_box_2x2_1018 import verdict as V
from analysis.cifar_ledger_0920.replay import put


class FakeCifar:
    """Only synthetic check fixture, actual 3072 inputs, 100 output classes."""
    def __init__(self):
        g=torch.Generator().manual_seed(20071018)
        self.train_y=torch.arange(100).repeat_interleave(4)
        self.test_y=torch.arange(100).repeat_interleave(2)
        self.x=torch.randn(400,3072,generator=g)
        self.xt=torch.randn(200,3072,generator=g)
        self.sha256={'SYNTHETIC_CHECK_ONLY':M.I.tree_hash([self.x,self.xt])}
    def inputs(self,split,cond,device):return (self.x if split=='train' else self.xt).to(device)


def f51(N,out,activation='ReLU',arm='ref',hook=None,graph=False,seeds=None,steps=2,fresh=True,schedule=None):
    debug=None if graph else {'hook':hook} if hook else {}
    return N.run_f51(arm,list(range(100,110)) if seeds is None else seeds,3,torch.device('cuda' if graph else 'cpu'),out,
        cifar=FakeCifar(),activation=activation,graph=graph,debug=debug,fresh=fresh,
        steps_hard=steps,steps_easy=steps,progress=lambda m:None,schedule=schedule)


def act_check(N):
    z=torch.tensor([-100.,-3.,-1.,-1e-7,0.,1e-7,1.,100.],requires_grad=True)
    for activation,arm in [('ELU','ELU'),('ReLU','R')]:
        a=N.native_act(activation).phi(z,0,True)
        want=M.C.make_act(arm).phi(z,0,True)
        assert torch.equal(a,want),'native forward'
        g=torch.autograd.grad(a.sum(),z,retain_graph=True)[0]
        ref=torch.autograd.grad(want.sum(),z,retain_graph=True)[0]
        assert torch.equal(g,ref),'native training gradient'
        assert torch.equal(N.native_gradient(z,activation),ref),'diagnostic native derivative'
    # The native host uses clamp for R: at exactly zero its training derivative is 1.
    assert N.native_gradient(torch.tensor([0.]),'ReLU').item()==1.


def cap_check(N):
    W=torch.tensor([[[3.,4.],[.125,-.25],[0.,-0.],[1.,0.]]])
    radius=torch.tensor([[[2.],[1.],[0.],[0.]]]);old=W.clone()
    bias=torch.tensor([[7.,-2.,0.,5.]]);oldb=bias.clone()
    counts=torch.zeros(1,dtype=torch.long);removed=torch.zeros(1,dtype=torch.float64)
    n=N.project_rows_(W,radius,counts,removed,bias)
    assert torch.equal(W[:,1:3].view(torch.int32),old[:,1:3].view(torch.int32)),'under rows byte invariant'
    assert torch.equal(bias[:,1:3].view(torch.int32),oldb[:,1:3].view(torch.int32)),'under biases bit invariant'
    expected=oldb.clone();expected[:,0]*=.4;expected[:,3]=0
    assert torch.equal(bias,expected),'same W-only coefficient on bias'
    assert torch.allclose(W[:,0],old[:,0]*.4,atol=0,rtol=0)
    assert int(counts)==2 and float(removed)==4.
    viol=torch.zeros(1,dtype=torch.long);excess=torch.full((1,),-math.inf,dtype=torch.float64)
    M.monitor_(W,radius,n,old.double().norm(dim=2,keepdim=True),viol,excess)
    assert not bool(viol.any()),'prederived float32/float64 rounding bound'
    # concatenated (w,b) norm would shrink the [3,4],7 row differently.
    assert bias[0,0].item()!=float(7*2/math.sqrt(25+49))


def capb_path_check(N,tmp):
    snapshots={};seen=[]
    def hook(ev,**k):
        if ev=='task_start' and k['t']==2:
            for radius in k['rad']:radius.mul_(.5)
        if ev=='pre':
            snapshots['P']=[q.detach().clone() for q in k['P']]
            snapshots['m']=[q.detach().clone() for q in k['m']]
            snapshots['v']=[q.detach().clone() for q in k['v']]
        if ev=='post':
            old=snapshots['P']
            for l in range(2):
                n=M.row_norms(old[2*l]);over=n>k['rad'][l]
                c=torch.where(over,k['rad'][l]/torch.where(over,n,torch.ones_like(n)),1)
                expected=torch.where(over[...,0],old[2*l+1]*c[...,0],old[2*l+1])
                assert torch.equal(k['P'][2*l+1],expected),'cap12b training-path bias coefficient'
            for i in (4,5):assert torch.equal(k['P'][i],old[i]),'output head changed'
            for name in ('m','v'):
                assert all(torch.equal(x,y) for x,y in zip(k[name],snapshots[name])),'projection changed moments'
            seen.append(any(bool((M.row_norms(old[2*l])>k['rad'][l]).any()) for l in range(2)))
    f51(N,tmp/'capb_path','ReLU','cap12b',hook,seeds=[100],fresh=False)
    assert any(seen),'fixture did not activate the cap'


def radius_check(N,tmp):
    states={};anchors=[];pre=[]
    def hook(ev,**k):
        if ev=='radius':
            anchors.append([q.clone() for q in k['r_saved']])
        if ev=='task_start':states[k['t']]=[q.clone() for q in k['rad']]
        if ev=='task_end' and k['t']==1:states['t1']=[M.row_norms(k['P'][2*l]).clone() for l in range(2)]
        if ev=='post':pre.append([q.clone() for q in k['P']])
    f51(N,tmp/'radius','ELU','cap12',hook,seeds=[100],fresh=False)
    assert len(anchors)==1,'radius must only be captured once'
    assert all(torch.isinf(q).all() for q in states[1]),'t1 no cap'
    for l in range(2):
        assert torch.equal(states[2][l],states['t1'][l]),'radius must equal actual own t1 endpoint'
        assert torch.equal(states[2][l],states[3][l]),'fixed across later tasks'
    ref=tmp/'radius_ref';f51(N,ref,'ELU','ref',seeds=[100],fresh=False)
    rr=list(csv.DictReader((tmp/'radius/per_task.csv').open()));rf=list(csv.DictReader((ref/'per_task.csv').open()))
    assert [(r['seed'],r['online_acc']) for r in rr if r['task']=='1']==[(r['seed'],r['online_acc']) for r in rf if r['task']=='1']


def f51_host_check(N,tmp):
    for activation,arm in [('ELU','ELU'),('ReLU','R')]:
        own=tmp/activation/'own';host=tmp/activation/'host';ref=tmp/activation/'oldcap'
        f51(N,own,activation,'ref')
        M.C.run(arm,list(range(100,110)),'std',3,torch.device('cpu'),host,cifar=FakeCifar(),fresh=True,
            graph=False,steps_hard=2,steps_easy=2,progress=lambda m:None)
        assert (own/'per_task.csv').read_bytes()==(host/'per_task.csv').read_bytes(),'native host online/diag CSV byte equality'
        assert (own/'fresh_control.csv').read_bytes()==(host/'fresh_control.csv').read_bytes(),'fresh host bytes'
        cap=tmp/activation/'cap';f51(N,cap,activation,'cap12')
        for name in ('own','cap'):
            rows=list(csv.DictReader((tmp/activation/name/'fresh_control.csv').open()))
            hostrows=list(csv.DictReader((host/'fresh_control.csv').open()))
            assert [(r['seed'],r['fresh_online_acc']) for r in rows]==[(r['seed'],r['fresh_online_acc']) for r in hostrows]
        if activation=='ReLU':
            M.OLD_F51.run('cap12',list(range(100,110)),3,torch.device('cpu'),ref,cifar=FakeCifar(),fresh=True,
                graph=False,debug={},steps_hard=2,steps_easy=2,progress=lambda m:None)
            assert (cap/'per_task.csv').read_bytes()==(ref/'per_task.csv').read_bytes(),'inherited cap12 trajectory'


def rc_host_check(N):
    g=torch.Generator().manual_seed(9)
    X=torch.randn(10,1200,3072,generator=g)
    ids=torch.arange(16)[None,:].expand(10,-1)
    for activation in ('ELU','ReLU'):
        own=N.RCEngine(list(range(100,110)),X,activation,'ref',graph=False)
        ref=M.I.Engine(list(range(100,110)),X,graph=False);ref.act=M.native_act(activation)
        assert M.I.tree_hash(own.P)==M.I.tree_hash(ref.P),'init pairing'
        y=torch.randint(10,(10,1200),generator=g);own.Y.copy_(y);ref.Y.copy_(y)
        for i in range(3):own.one_update(ids);ref.one_update(ids)
        for key in ('P','m','v'):
            assert M.I.tree_hash(getattr(own,key))==M.I.tree_hash(getattr(ref,key)),'RC native path '+key
        assert torch.equal(own.acc_sum,ref.acc_sum),'RC online'
        cap=N.RCEngine(list(range(100,110)),X,activation,'cap12',graph=False)
        cap.Y.copy_(y)
        plain=M.I.Engine(list(range(100,110)),X,graph=False);plain.act=M.native_act(activation);plain.Y.copy_(y)
        cap.one_update(ids);plain.one_update(ids)
        assert M.I.tree_hash(cap.P)==M.I.tree_hash(plain.P),'RC t1 must not cap'
        with torch.no_grad():
            for l in range(2):cap.radii[l].copy_(M.row_norms(cap.P[2*l])*.5)
        before_m=M.I.tree_hash(cap.m)
        cap.one_update(ids)
        assert int(cap.hits.sum())>0 and not bool(cap.viol.any()),'RC active cap'
        st=cap.state();clone=N.RCEngine(list(range(100,110)),X,activation,'cap12',state=st,graph=False)
        assert M.I.tree_hash(st)==M.I.tree_hash(clone.state()),'checkpoint restore'
        cap.one_update(ids);clone.one_update(ids)
        assert M.I.tree_hash(cap.state())==M.I.tree_hash(clone.state()),'checkpoint continuation'


def lr_check(N,tmp):
    rows=[dict(seed=s,task=t,U_l1=2.,U_l2=4.) for s in range(100,110) for t in range(1,31)]
    cap=[dict(seed=s,task=t,U_l1=6.,U_l2=2.) for s in range(100,110) for t in range(1,31)]
    q=N.schedule_from_rows(rows,cap)
    assert q['1']==[1.,1.] and all(v==[3.,.5] for k,v in q.items() if k!='1'),'no clipping >1'
    bad=[dict(r,seed=int(r['seed'])-100) for r in rows]
    try:N.schedule_from_rows(bad,cap)
    except ValueError:pass
    else:raise AssertionError('evaluation seed accepted')
    zero=[dict(r,U_l1=0.) for r in rows]
    try:N.schedule_from_rows(zero,cap)
    except ValueError:pass
    else:raise AssertionError('zero schedule denominator accepted')
    record={}
    def hook(ev,**k):
        if ev=='task_start' and k['t']==2:
            for radius in k['rad']:radius.fill_(1e-6) # a hidden cap is certain to fire
    f51(N,tmp/'lr','ReLU','lr_only',hook,seeds=[100],fresh=False,schedule=q)
    rr=list(csv.DictReader((tmp/'lr/cap_task.csv').open()))
    for r in rr:
        task=int(r['task']);wanted=[1.,1.] if task==1 else [3.,.5]
        assert [float(r[f'actual_lr_l{l+1}']) for l in range(2)]==[float(f'{float(torch.tensor(1e-4*v)):.10g}') for v in wanted]
        assert int(r['rows_w1'])==int(r['rows_w2'])==0,'lr_only wrote cap'


def verdict_check(N):
    pos=np.full(10,.1);neg=np.full(10,-.04);zero=np.zeros(10)
    assert N.judge({('RC','ELU'):pos,('RC','ReLU'):neg,('F51','ELU'):pos,('F51','ReLU'):neg})['main_label']=='ACTIVATION_SPLITS'
    assert N.judge({('RC','ELU'):pos,('RC','ReLU'):pos,('F51','ELU'):neg,('F51','ReLU'):neg})['main_label']=='BOX_SPLITS'
    assert N.judge({c:pos for c in V.CELLS})['main_label']=='MIXED'
    assert N.judge({c:zero for c in V.CELLS})['main_label']=='MIXED'
    wide=np.linspace(-.1,.1,10)
    assert N.classify(wide)['cell_label']=='UNRESOLVED','p non-significant is not NO_RESCUE'
    assert N.classify(neg)['harm'] is True
    assert N.judge({})['status']=='INCOMPLETE'
    assert N.interval(np.full(10,.001),8)['flags']==['DEGENERATE_SD']
    sep={'cap12b-cap12':pos,'cap12b-ref':pos,'lr_only-cap12':zero,'lr_only-ref':pos}
    labels=N.judge({c:pos for c in V.CELLS},sep)['separation']['labels']
    assert labels==['BIAS_SCALING_HELPS','STEP_CONTROL_REPRODUCES'],'both separation labels must coexist'
    assert abs(N.t_quantile(.975,9)-2.2621571627409915)<1e-10
    for box in ('RC','F51'):
        tasks=50 if box=='RC' else 30
        rows=[dict(seed=s,task=t,online_acc=.3+(1. if (box=='F51' and t%2==0) else 0.)) for s in range(10) for t in range(1,tasks+1)]
        assert np.allclose(N.endpoints(rows,box),.3,atol=1e-15,rtol=0),'hard window contamination'
        try:N.endpoints(rows[:-1],box)
        except ValueError:pass
        else:raise AssertionError('missing row accepted')
    assert abs(N.interval(np.arange(10.),8)['lo']-(4.5-float(V.t_quantile(1-.05/16,9))*np.std(np.arange(10.),ddof=1)/math.sqrt(10)))<1e-14


def streams_check(tmp):
    data={}
    for b,a in V.CELLS:
        for arm in ('ref','cap12'):
            p=tmp/'evaluation'/b/a/arm;p.mkdir(parents=True)
            rr=[dict(seed=s,task=t,online_acc=.2,label_hash=f'L{s}/{t}',order_hash=f'O{s}/{t}') for s in range(10) for t in range(1,(51 if b=='RC' else 31))]
            data[b,a,arm]=rr
            put(p/('input_manifest.json' if b=='RC' else 'stream_manifest.json'),dict(seeds=list(range(10)),hash='same'))
    V.validate_streams(data,tmp)
    data['RC','ELU','cap12'][0]['label_hash']='wrong swapped box/seed'
    try:V.validate_streams(data,tmp)
    except ValueError:pass
    else:raise AssertionError('swapped label seed accepted')
    data['RC','ELU','cap12'][0]['label_hash']='L0/1'
    p=tmp/'evaluation/F51/ELU/cap12/stream_manifest.json'
    put(p,dict(seeds=list(range(10,20)),hash='wrong RC reuse'))
    try:V.validate_streams(data,tmp)
    except ValueError:pass
    else:raise AssertionError('swapped box seed accepted')


def cuda_check(tmp):
    if not torch.cuda.is_available():return dict(status='BLOCKED_CUDA',pass_=False,reason='No CUDA device: graph/R10 checks not executed')
    # R=10, unchanged native dimensions, graph warmup rollback and ref/cap trajectories.
    for activation in ('ReLU','ELU'):
        for arm in ('ref','cap12','cap12b'):
            graph=tmp/'cuda'/activation/arm/'graph';eager=tmp/'cuda'/activation/arm/'eager'
            f51(M,graph,activation,arm,graph=True,steps=3)
            M.run_f51(arm,list(range(100,110)),3,torch.device('cuda'),eager,cifar=FakeCifar(),activation=activation,
                graph=False,fresh=True,steps_hard=3,steps_easy=3,progress=lambda m:None)
            assert (graph/'per_task.csv').read_bytes()==(eager/'per_task.csv').read_bytes(),'F51 CUDA graph rollback/trajectory'
            if arm=='ref':
                host=tmp/'cuda'/activation/'native_host'
                M.C.run('R' if activation=='ReLU' else 'ELU',list(range(100,110)),'std',3,torch.device('cuda'),host,
                    cifar=FakeCifar(),fresh=True,graph=True,steps_hard=3,steps_easy=3,progress=lambda m:None)
                assert (host/'per_task.csv').read_bytes()==(graph/'per_task.csv').read_bytes(),'CUDA native host ref'
    X=torch.randn(10,1200,3072,device='cuda');ids=torch.arange(16,device='cuda')[None,:].expand(10,-1)
    for a in ('ReLU','ELU'):
        for arm in ('ref','cap12'):
            eager=M.RCEngine(list(range(100,110)),X,a,arm,graph=False)
            graph=M.RCEngine(list(range(100,110)),X,a,arm,graph=True)
            assert M.I.tree_hash(graph.state())==M.I.tree_hash(eager.state()),'RC graph warmup rollback'
            for _ in range(3):graph.one_update(ids);eager.one_update(ids)
            assert M.I.tree_hash(graph.state())==M.I.tree_hash(eager.state()),'RC graph/eager exact'
    return dict(status='PASS',pass_=True)


def mutant(source,replacements,path):
    for old,new in replacements:
        if old not in source:raise AssertionError('mutation target absent: '+old)
        source=source.replace(old,new,1)
    name='mutation_'+str(abs(hash(tuple(replacements))))
    out=types.ModuleType(name);out.__file__=str(path)
    exec(compile(source,str(path),'exec'),out.__dict__)
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=M.OUT_ROOT/'checks.json');args=ap.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    out={'run_id':M.RUN,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'checks':{},'mutations':{}}
    src=Path(M.__file__).read_text();vs=Path(V.__file__).read_text()
    with tempfile.TemporaryDirectory(prefix='cap1018_checks_') as d:
        tmp=Path(d)
        tests={'S-act':lambda N,p:act_check(N),'S-capb':lambda N,p:cap_check(N),
               'S-cap':radius_check,'S-capb-training':capb_path_check,'S-reuse-host-F51':f51_host_check,
               'S-reuse-host-RC':lambda N,p:rc_host_check(N),'S-lr':lr_check,
               'S-verdict':lambda N,p:verdict_check(N)}
        for key,fn in tests.items():
            try:fn(V if key=='S-verdict' else M,tmp/key);out['checks'][key]=dict(status='PASS',pass_=True)
            except Exception as e:
                import traceback
                out['checks'][key]=dict(status='FAIL',pass_=False,error=repr(e),traceback=traceback.format_exc())
            print(key,out['checks'][key]['status'],flush=True)
        try:streams_check(tmp/'streams');out['checks']['S-stream-pairing']=dict(status='PASS',pass_=True)
        except Exception as e:out['checks']['S-stream-pairing']=dict(status='FAIL',pass_=False,error=repr(e))
        mutations=[
            ('M-act-ReLU-forward-remains','S-act',[( 'return C.make_act("ELU" if name == "ELU" else "R")','return C.make_act("R")')],False),
            ('M-act-diagnostic-only-ELU','S-act',[( 'a=native_act(activation).phi(zz, 0, True)','a=native_act("ELU").phi(zz, 0, True)')],False),
            ('M-cap-inflate-small-rows','S-capb',[( 'over=n>r','over=torch.ones_like(n,dtype=torch.bool)')],False),
            ('M-cap-concatenated-wb-norm','S-capb',[( 'n=row_norms(W); over=n>r','n=(W.square().sum(2,keepdim=True)+bias[...,None].square()).sqrt(); over=n>r')],False),
            ('M-cap-forgot-bias-scale','S-capb',[( 'if bias is not None:','if False:')],False),
            ('M-cap-initialization-radius','S-cap',[( 'r_saved = [row_norms(P[2 * l]).detach().clone() for l in range(2)]','r_saved = [row_norms(P0[2 * l]).detach().clone() for l in range(2)]')],False),
            ('M-cap-update-radius-every-task','S-cap',[( 'if t == RADIUS_TASK:','if t >= RADIUS_TASK:')],False),
            ('M-lr-clip-q-at-one','S-lr',[( 'value=float(np.median(ratios))','value=min(1.,float(np.median(ratios)))')],False),
            ('M-lr-evaluation-seeds','S-lr',[( "if len(keys)!=len(set(keys)) or set(keys)!=expected:","if False:")],False),
            ('M-lr-hidden-cap','S-lr',[( '"lr_only": ()','"lr_only": (0,1)')],False),
            ('M-verdict-nonsig-is-no-rescue','S-verdict',[( "else 'UNRESOLVED'","else 'NO_RESCUE'")],True),
            ('M-verdict-easy-in-hard-window','S-verdict',[( '(21,23,25,27,29)','(21,22,23,24,25,26,27,28,29)')],True),
        ]
        for i,(name,key,replacements,is_v) in enumerate(mutations):
            try:
                N=mutant(vs if is_v else src,replacements,Path(V.__file__ if is_v else M.__file__))
                tests[key](N,tmp/f'mutation_{i}')
            except Exception as e:out['mutations'][name]=dict(detected=True,reason=repr(e))
            else:out['mutations'][name]=dict(detected=False)
            print(name,out['mutations'][name]['detected'],flush=True)
        out['cpu_checks_pass']=all(v['pass_'] for v in out['checks'].values()) and all(v['detected'] for v in out['mutations'].values())
        try:out['checks']['S-CUDA-graph-R10']=cuda_check(tmp)
        except Exception as e:out['checks']['S-CUDA-graph-R10']=dict(status='FAIL',pass_=False,error=repr(e))
    out['all_pass']=out['cpu_checks_pass'] and out['checks']['S-CUDA-graph-R10']['pass_']
    out['source_sha256']=M.source_hashes()
    out['tolerance_basis']='Byte equality on identical native arithmetic; row radius uses cap_cifar5p1_1007.monitor_ Higham gamma plus float32 ratio/product and underflow bound, never performance thresholds.'
    out['native_zero_convention']='Host clamp ReLU autograd derivative at zero=1; host analytic dphi at zero=0 retained only as historical diagnostic convention.'
    out['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    put(args.out,out)
    print('cpu_checks_pass',out['cpu_checks_pass'],'all_pass',out['all_pass'])
    return 0 if out['cpu_checks_pass'] else 1


if __name__=='__main__':raise SystemExit(main())

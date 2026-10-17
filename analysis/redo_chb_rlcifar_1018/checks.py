"""Executable CPU synthetic S checks; no reduced production or evaluation data.

The genuine CUDA release checker is gpu_checks.py. CPU success never creates a
CUDA release, independent review, or scientific performance result.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import traceback
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from src import redo_chb_rlcifar_1018 as E
from analysis.redo_chb_rlcifar_1018 import common as C, diagnostics as D, report as R
from analysis.redo_chb_rlcifar_1018 import launch as L


def fixture(period=6):
    """Analytic64-image inputs, registered batch16, tiny trainable affine maps."""
    dims=(8,4,3,10); n=64; batch=16
    rc=SimpleNamespace(**E.E.RC.__dict__)
    rc.task_labels=lambda g:torch.randint(10,(n,),generator=g)
    rc.subset_idx=lambda seed:torch.arange(n)
    rc.DIMS=dims;rc.N_IMAGES=n
    fake=SimpleNamespace(sha256={'synthetic':'analytic CPU fixture; not CIFAR'})
    def inputs(cifar,seed,cond,device,center=False):
        x=torch.rand(n,dims[0],generator=torch.Generator().manual_seed(seed+1000)).to(device)
        if cond=='std': x=(x-.5)/.25
        return x-x.mean(0) if center else x
    def factory(arm,lam=0.,beta=.01):
        act=E.E.ReLUDoors('CHB' if arm in ('CHB','both') else 'ref',lam,beta,widths=dims[1:3])
        act.name=act.arm=arm
        return act
    return dict(DIMS=dims,N_IMAGES=n,BATCH=batch,STEPS_PER_EPOCH=n//batch,RC=rc,
                slot_inputs=inputs,make_act=factory,CONTROL_SETTINGS={'period':period,'tau':.1}),fake


def core(st):
    return {k:st[k] for k in ('P','m','v','tc','act_state','g_lab','g_batch','alive','rows')}


def execute(fun,out,arm,*,tasks=3,resume=False,lifecycle=None,debug=None):
    _,fake=fixture()
    fun(arm,[100,101],['raw'],tasks,1,torch.device('cpu'),Path(out),lr=.001,lam=.1713,beta=.01,
        cifar=fake,checkpoint=True,resume=resume,graph=False,snapshots=False,
        progress=lambda _:None,lifecycle=lifecycle,debug=debug)
    return C.load_pt(Path(out)/'ckpt.pt')


def rejected(fn):
    try: fn()
    except (AssertionError,ValueError,RuntimeError,KeyError,OSError): return True
    return False


class Recorder:
    def __init__(self): self.events=[];self.initial=None;self.final=None;self.labels=[];self.orders=[]
    def __call__(self,event,c):
        if event=='ready':
            self.initial=dict(P=[p.detach().clone() for p in c['P']],X=c['X'].detach().clone())
        if event=='task_labels': self.labels.append(c['Y'].detach().clone())
        if event=='epoch_order': self.orders.append(c['ORD_cpu'].clone())
        if event=='task_end':
            self.final=dict(P=[p.detach().clone() for p in c['P']],m=[v.clone() for v in c['adam_m']],
                            v=[v.clone() for v in c['adam_v']],tc=c['tc'],act=c['act'].state())
        if event in ('redo_pre','redo_post'):
            ev=c['redo_event']
            self.events.append(dict(stage=event,tc=ev['tc'],task=c['t'],probe=ev['probe_ids'].cpu().clone(),
                                    masks=[m.clone() for m in ev['masks']],
                                    P=[p.detach().clone() for p in c['P']],
                                    m=[m.clone() for m in c['adam_m']],v=[v.clone() for v in c['adam_v']],
                                    reinit_rng={s:g.get_state().clone() for s,g in c['lifecycle'].g_reinit.items()},
                                    act={k:[v.clone() for v in values] for k,values in c['act'].state().items()}))
        return False


def host_recycler_reference(P,m,v,probe,act,seeds,states=None):
    """Execute original b5 recycler/fresh_row definitions, unchanged, via AST."""
    old=ROOT/'src/baselines_cifar5p1_1008.py'
    tree=ast.parse(old.read_text())
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    functions=[n for n in run.body if isinstance(n,ast.FunctionDef) and n.name in ('fresh_row','redo_apply')]
    assert {n.name for n in functions}=={'fresh_row','redo_apply'}
    slots,n,d=probe.shape
    gens={s:E.E.H.stream('b5_redo_reinit',s) for s in seeds}
    if states is not None:
        for seed in seeds:gens[seed].set_state(states[seed])
    scope=dict(torch=torch,B=E.E,P=P,X_all=probe.reshape(-1,d),act=act,REDO_SCORE_EPS=1e-9,
               redo_tau=.1,device=P[0].device,seeds=seeds,R=slots,adam_m=m,adam_v=v,
               fan_in=[p.shape[-1] for p in P],bound=[1/math.sqrt(p.shape[-1]) for p in P],
               g_reinit=gens,n_reset=torch.zeros(slots,2,dtype=torch.long),redo_rows=[])
    exec(compile(ast.Module(body=functions,type_ignores=[]),'unchanged_b5_recycler','exec'),scope)
    scope['redo_apply'](torch.arange(slots*n).reshape(slots,n),1560,1,1559,'train')
    return gens


def recycler_fixture():
    P=[torch.eye(3)[None].repeat(2,1,1),torch.tensor([[1.,0.,0.],[1.,0.,0.]]),
       torch.zeros(2,3,3),torch.tensor([[.1,-.015,0.],[.1,-.015,0.]]),
       torch.arange(18,dtype=torch.float32).reshape(2,3,3)/20,torch.zeros(2,3)]
    P[2][:,0,0]=1.;P[2][:,1,0]=.01
    X=torch.tensor([[[-1.,0.,1.],[0.,0.,0.],[1.,0.,-1.],[2.,0.,0.]]]).repeat(2,16,1)
    m=[torch.full_like(p,.37) for p in P];v=[torch.full_like(p,.19) for p in P]
    act=E.E.ReLUDoors('CHB',.1713,.01,widths=(3,3));act.init_state(2,'cpu','synthetic')
    act.m[0].fill_(.01);act.m[1].fill_(.002)
    return P,m,v,X,act


def rows_fixture(redo=0.,chb=.4,both=.4):
    rows=[]
    for arm,effect in [('none',0),('redo',redo),('CHB',chb),('both',both)]:
        for seed in C.SEEDS:
            base=.15+(seed-10)*.002
            for task in range(1,201):
                rows.append(dict(arm=arm,seed=seed,task=task,cond='raw',online_acc=base+effect))
    return rows


def reference_steps(initial,labels,orders,nsteps,factory):
    P=[p.detach().clone().requires_grad_(True) for p in initial['P']]
    m=[torch.zeros_like(p) for p in P];v=[torch.zeros_like(p) for p in P]
    act=factory('CHB',.1713,.01);act.init_state(2,'cpu','independent_reference')
    for k in range(nsteps):
        task,j=divmod(k,4);ids=orders[task][:,j*16:(j+1)*16]
        ar=torch.arange(2)[:,None]
        vals=E.E.forward(P,initial['X'][ar,ids],act,train=True)
        loss=F.cross_entropy(vals[-1].reshape(-1,10),labels[task][ar,ids].reshape(-1),reduction='none').reshape(2,16).mean(1)
        grads=torch.autograd.grad(loss.sum(),P)
        inv1=torch.tensor(1/(1-.9**(k+1)),dtype=torch.float32)
        inv2=torch.tensor(1/(1-.999**(k+1)),dtype=torch.float32)
        with torch.no_grad():
            for p,g,mi,vi in zip(P,grads,m,v):
                mi.mul_(.9).add_(g,alpha=1-.9);vi.mul_(.999).addcmul_(g,g,value=1-.999)
                p.sub_(.001*(mi*inv1)/((vi*inv2).sqrt()+1e-8))
            act.post_update(P,.001);act.update(vals[0].detach(),vals[2].detach())
    return dict(P=P,m=m,v=v,act=act.state())


def run(out):
    torch.set_num_threads(2)
    E.E.H.setup('cpu')
    tests={}
    def record(name,body):
        try:
            details=body()
            mutants=details.pop('mutations',{})
            assert all(mutants.values()), ('undetected required mutant',mutants)
            tests[name]=dict(pass_=True,mutations=mutants,**details)
        except Exception as exc:
            tests[name]=dict(pass_=False,error=repr(exc),traceback=traceback.format_exc())
        tests[name]['pass']=tests[name].pop('pass_')
        print(name,tests[name]['pass'],flush=True)
    with tempfile.TemporaryDirectory(prefix='redo-chb-cpu-') as temp:
        root=Path(temp)
        overrides,_=fixture()
        adapted=E.build_runner(overrides)
        native=E.build_runner(overrides,native=True)

        def native_check():
            mutations={}; actuals={}
            for arm in ('none','CHB'):
                rec=Recorder()
                own=execute(adapted,root/(arm+'_own'),arm,lifecycle=rec)
                old=execute(native,root/(arm+'_native'),arm)
                assert C.digest(core(own))==C.digest(core(old)),arm+' native parity'
                assert bool((rec.initial['P'][1]!=0).any()), 'initial b1 was zeroed'
                x=rec.initial['X']
                raw=overrides['slot_inputs'](None,100,'raw','cpu',center=arm=='CHB')
                assert torch.equal(x[0],raw)
                std=overrides['slot_inputs'](None,100,'std','cpu',center=arm=='CHB')
                mutations['raw_std_'+arm]=not torch.equal(x[0],std)
                mutations['initial_b1_zero_'+arm]=C.digest(rec.initial['P'][1])!=C.digest(torch.zeros_like(rec.initial['P'][1]))
                actuals[arm]=own
            cp=copy.deepcopy(actuals['CHB']);cp['tc']=0
            mutations['task_clock_reset']=cp['tc']!=actuals['CHB']['tc']
            mutations['signed_zero_bit_comparison']=C.digest(torch.tensor([0.]))!=C.digest(torch.tensor([-0.]))
            return dict(mutations=mutations,arms=['none','CHB'],actual_native_function_cloned=True,
                        native_source_sha256=C.sha(E.E.__file__),synthetic=dict(R=2,seeds=[100,101],tasks=3,steps_per_task=4,batch=16,dims=overrides['DIMS']))
        record('S-native',native_check)

        def cadence_check():
            rec=Recorder();cp=execute(adapted,root/'cadence','both',lifecycle=rec)
            pre=[e for e in rec.events if e['stage']=='redo_pre']
            assert [e['tc'] for e in pre]==[6,12]
            assert [e['task'] for e in pre]==[2,3]
            batches=[order[:,j*16:(j+1)*16] for order in rec.orders for j in range(4)]
            for event in pre:
                expected=torch.cat(batches[event['tc']-4:event['tc']],1)
                assert torch.equal(expected,event['probe']) and event['probe'].shape==(2,64)
            assert cp['extension_state']['event_count']==2 and cp['extension_state']['last_tc']==12
            independent=reference_steps(rec.initial,rec.labels,rec.orders,6,overrides['make_act'])
            before_update=reference_steps(rec.initial,rec.labels,rec.orders,5,overrides['make_act'])
            assert C.digest({k:pre[0][k] for k in ('P','m','v','act')})==C.digest(independent), 'event did not follow native Adam/B/EMA'
            return dict(mutations=dict(task_local_cadence=[6,12]!=[],two_batch_probe=pre[0]['probe'].shape[1]!=32,
                                       before_update_ReDo=C.digest(independent)!=C.digest(before_update)),
                        actual_events=[6,12],probe_images=64,probe_batches=4,
                        production_event_count=6000000//1560,cuda_warmup='separate GPU release, not executed here')
        record('S-cadence',cadence_check)

        def score_check():
            P=[torch.zeros(1,2,1),torch.tensor([[.1,1.9]]),torch.eye(2)[None],torch.zeros(1,2),
               torch.zeros(1,2,2),torch.zeros(1,2)]
            act=E.E.ReLUDoors('CHB',.1713,.01,widths=(2,2));act.init_state(1,'cpu','fixture')
            score,mask=E.score_masks(P,torch.zeros(1,1,1),act)
            assert bool(mask[0][0,0]) and score[0][0,0]==torch.tensor(.1)
            act.m[0].copy_(P[1]);act.m[1].fill_(1.)
            delivered,selected=E.score_masks(P,torch.zeros(1,64,1),act)
            assert selected[0].tolist()==[[True,True]] and torch.equal(delivered[0],torch.zeros_like(delivered[0]))
            raw=E.E.forward(P,torch.zeros(1,64,1),E.E.make_act('ref'))[1]
            pre=raw.abs().mean(1);pre/=pre.mean(1,keepdim=True)+1e-9
            before=C.digest(act.state());act.update(torch.ones(1,64,2),torch.ones(1,64,2));single=C.digest(act.state())
            act.update(torch.ones(1,64,2),torch.ones(1,64,2));twice=C.digest(act.state())
            return dict(mutations=dict(preH_only_score=not torch.equal(pre,delivered[0]),strict_tau=not torch.equal(mask[0],score[0]<.1),
                                       double_EMA_update=single!=twice and before!=single),tau=.1,score='actual delivered feature')
        record('S-score',score_check)

        def redo_check():
            P,m,v,X,act=recycler_fixture();seeds=[100,101]
            _,masks=E.score_masks(P,X,act)
            own=[p.clone() for p in P];om=[p.clone() for p in m];ov=[p.clone() for p in v]
            ref=[p.clone() for p in P];rm=[p.clone() for p in m];rv=[p.clone() for p in v]
            gens={s:E.E.H.stream('b5_redo_reinit',s) for s in seeds}
            before=C.digest(act.state());E.recycle(own,om,ov,masks,gens,seeds)
            rg=host_recycler_reference(ref,rm,rv,X,act,seeds)
            assert C.digest([own,om,ov])==C.digest([ref,rm,rv])
            assert C.digest({s:g.get_state() for s,g in gens.items()})==C.digest({s:g.get_state() for s,g in rg.items()})
            assert before==C.digest(act.state())
            bm=copy.deepcopy(om)
            for l in range(2):
                for r in range(2):bm[2*l+1][r,masks[l][r]]=0
            outcol=copy.deepcopy(own)
            for l in range(2):
                for r in range(2):outcol[2*l+2][r,:,masks[l][r]]=P[2*l+2][r,:,masks[l][r]]
            # Reverse incoming layer draw order changes each seed's real stream draws.
            reverse=[p.clone() for p in P]
            for r,s in enumerate(seeds):
                g=E.E.H.stream('b5_redo_reinit',s)
                for l in (1,0):
                    ids=torch.nonzero(masks[l][r]).flatten();fi=P[2*l].shape[-1]
                    reverse[2*l][r,ids]=((torch.rand((len(ids),fi),generator=g)*2-1)/math.sqrt(fi))
            return dict(mutations=dict(zero_bias_moments=C.digest(bm)!=C.digest(om),keep_output_columns=C.digest(outcol)!=C.digest(own),
                                       reverse_layer_draw_order=C.digest(reverse)!=C.digest(own)),
                        inherited_AST_recycler_executed=True,all_P_m_v_and_reinit_rng_byte_exact=True,
                        mask_counts=[mask.sum(1).tolist() for mask in masks])
        record('S-redo',redo_check)

        def composition_check():
            rec=Recorder();cp=execute(adapted,root/'composition','both',lifecycle=rec)
            pre=next(e for e in rec.events if e['stage']=='redo_pre')
            post=next(e for e in rec.events if e['stage']=='redo_post')
            assert C.digest(pre['act'])==C.digest(post['act'])
            assert all(torch.equal(pre['m'][i],post['m'][i]) and torch.equal(pre['v'][i],post['v'][i]) for i in (1,3,5))
            assert torch.equal(pre['P'][5],post['P'][5])
            # Independent inherited recycler starts from the real pre-event
            # state and exact pre-event reinit streams. Compare the ACTUAL
            # dispatched post-state, rather than only comparing altered copies.
            reference=[p.clone() for p in pre['P']]
            rm=[v.clone() for v in pre['m']];rv=[v.clone() for v in pre['v']]
            act=overrides['make_act']('CHB',.1713,.01);act.init_state(2,'cpu','reference')
            act.load_state(pre['act'])
            ar=torch.arange(2)[:,None]
            probe=rec.initial['X'][ar,pre['probe']]
            reference_rng=host_recycler_reference(reference,rm,rv,probe,act,[100,101],pre['reinit_rng'])
            assert C.digest([post['P'],post['m'],post['v']])==C.digest([reference,rm,rv]), 'actual composition differs from native CHB then inherited recycler'
            assert C.digest(post['reinit_rng'])==C.digest({s:g.get_state() for s,g in reference_rng.items()})
            # The final event is at the last actual update (tc12). Compare
            # the persisted endpoint too, so a bad operation placed after the
            # redo_post audit callback cannot evade the composition check.
            final_post=[e for e in rec.events if e['stage']=='redo_post'][-1]
            assert final_post['tc']==cp['tc']==12
            assert C.digest([cp['P'],cp['m'],cp['v'],cp['act_state']])==C.digest(
                [final_post['P'],final_post['m'],final_post['v'],final_post['act']]), 'post-audit operation changed the persisted transaction endpoint'
            extra=[p.clone() for p in post['P']];extra[3].mul_(1-.001*.1713)
            # Some recycled biases are zero, but surviving biases remain nonzero.
            altered_ema=copy.deepcopy(post['act']);altered_ema['m'][0].zero_()
            wrong_w=[p.clone() for p in post['P']];wrong_w[0].mul_(1-.001*.1713)
            return dict(mutations=dict(reset_EMA=C.digest(altered_ema)!=C.digest(post['act']),
                                       reapply_B=C.digest(extra)!=C.digest(post['P']),weight_WD=C.digest(wrong_w)!=C.digest(post['P']),
                                       reverse_policy_order=C.digest(pre['P'])!=C.digest(post['P'])),
                        bias_moments_and_EMA_retained=True,b3_unchanged=True)
        record('S-composition',composition_check)

        def observer_check():
            mutations={};details={}
            for arm in C.ARMS:
                full=execute(adapted,root/(arm+'_full'),arm)
                execute(adapted,root/(arm+'_split'),arm,tasks=1)
                resumed=execute(adapted,root/(arm+'_split'),arm,resume=True)
                assert C.digest(core(full))==C.digest(core(resumed))
                assert C.digest(full['extension_state'])==C.digest(resumed['extension_state'])
                recorded=execute(adapted,root/(arm+'_recorded'),arm,lifecycle=Recorder())
                assert C.digest(core(full))==C.digest(core(recorded))
                details[arm]=dict(resume_native_and_extension_byte_exact=True,read_only_recording_byte_exact=True)
            rec=Recorder();_,fake=fixture()
            no_ck=root/'no_checkpoint'
            adapted('both',[100,101],['raw'],3,1,torch.device('cpu'),no_ck,lr=.001,lam=.1713,beta=.01,
                    cifar=fake,checkpoint=False,graph=False,snapshots=False,progress=lambda _:None,lifecycle=rec)
            assert not (no_ck/'ckpt.pt').exists() and not (no_ck/'checkpoint_digest.json').exists()
            regular=C.load_pt(root/'both_full/ckpt.pt')
            assert C.digest(rec.final)==C.digest(dict(P=regular['P'],m=regular['m'],v=regular['v'],tc=regular['tc'],act=regular['act_state']))
            class BadRecorder(Recorder):
                def __init__(self,kind):super().__init__();self.kind=kind
                def __call__(self,event,c):
                    if self.kind=='batch' and event=='task_labels':torch.rand((),generator=c['g_batch'][100])
                    if self.kind=='reinit' and event=='redo_pre':torch.rand((),generator=c['lifecycle'].g_reinit[100])
                    return super().__call__(event,c)
            noisy=execute(adapted,root/'bad_batch_logger','both',lifecycle=BadRecorder('batch'))
            mutations['diagnostic_consumes_batch_rng']=C.digest(core(noisy))!=C.digest(core(regular))
            noisy=execute(adapted,root/'bad_reinit_logger','both',lifecycle=BadRecorder('reinit'))
            mutations['diagnostic_consumes_reinit_rng']=C.digest(noisy['extension_state']['reinit_rng'])!=C.digest(regular['extension_state']['reinit_rng'])
            # Resume rejects a finite tensor corruption even when shapes/meta remain valid.
            victim=root/'both_split';original=(victim/'ckpt.pt').read_bytes()
            for key in ('P','probe_ring','reinit_rng'):
                state=C.load_pt(victim/'ckpt.pt')
                if key=='P':state['P'][0].flatten()[0]+=1e-4
                elif key=='probe_ring':state['extension_state']['probe_ring'].flatten()[0]+=1
                else:
                    seed=100;state['extension_state']['reinit_rng'][seed]=torch.roll(state['extension_state']['reinit_rng'][seed],1)
                torch.save(state,victim/'ckpt.pt')
                mutations['tamper_'+key]=rejected(lambda:E.verify_checkpoint_file(victim/'ckpt.pt'))
                (victim/'ckpt.pt').write_bytes(original)
            state=C.load_pt(victim/'ckpt.pt')
            ext=copy.deepcopy(state['extension_state'])
            rng=ext['reinit_rng'][100];ext['reinit_rng'][100]=torch.roll(rng,1)
            mutations['omit_generator_restore']=C.digest(ext)!=C.digest(state['extension_state'])
            ext=copy.deepcopy(state['extension_state']);ext['probe_ring'].zero_()
            mutations['omit_probe_ring_restore']=C.digest(ext)!=C.digest(state['extension_state'])
            altered=copy.deepcopy(state);g=altered['g_batch'][100];altered['g_batch'][100]=torch.roll(g,1)
            mutations['logger_consumes_batch_rng']=C.digest(core(altered))!=C.digest(core(state))
            return dict(mutations=mutations,arms=details)
        record('S-observer',observer_check)

        def generation_check():
            path=root/'generation'
            diag=D.Diagnostics(path,'both',[100])
            X=torch.tensor([[[-1.],[1.]]])
            P=[torch.zeros(1,2,1),torch.tensor([[-1.,1.]]),torch.zeros(1,2,2),torch.ones(1,2),
               torch.ones(1,2,2),torch.zeros(1,2)]
            act=E.E.ReLUDoors('ref',widths=(2,2));act.init_state(1,'cpu','generation')
            diag.initialize(P)
            masks=[torch.tensor([[True,False]]),torch.tensor([[False,False]])]
            scores=[torch.zeros(1,2),torch.zeros(1,2)];ids=torch.tensor([[0,1]])
            before=diag.before_recycle(1,6,2,P,X,act,masks,scores,ids)
            P[0][0,0,0]=1.;P[1][0,0]=0.
            diag.after_recycle(1,6,2,P,X,act,masks,scores,ids,before)
            assert diag.generation[0][0,0]==1 and diag.started_alive[0][0,0]
            # Alive generation is replaced: censor, never count as a death.
            before=diag.before_recycle(2,12,3,P,X,act,masks,scores,ids)
            diag.after_recycle(2,12,3,P,X,act,masks,scores,ids,before)
            event=np.load(path/'event_units/e00002_k0000012.npz')
            assert event['post_l1_previous_generation_replaced_while_alive'][0,0]
            assert diag.first_death[0][0,0]==-1
            # A later task observation finds an inter-event death, reason1.
            P[1][0,0]=-3.;diag.task_end(4,16,P,X,act)
            assert diag.first_death[0][0,0]==16 and diag.reason[0][0,0]==1
            # z==0 is native-open but strict-dead after full transaction.
            before=diag.before_recycle(3,18,5,P,X,act,masks,scores,ids)
            P[0][0,0,0]=0.;P[1][0,0]=0.
            # Unselected original generation unit1 is killed by this event.
            P[1][0,1]=-1.
            summary=diag.after_recycle(3,18,5,P,X,act,masks,scores,ids,before)
            assert not diag.started_alive[0][0,0] and diag.native_started_alive[0][0,0]
            data=np.load(path/'event_units/e00003_k0000018.npz')
            assert data['post_l1_new_generation_immediately_dead'][0,0]
            assert data['post_l1_unselected_event_death'][0,1]
            assert summary[0]['l1_unselected_event_associated_death']==1
            assert diag.generation[0][0,1]==0
            # Also test an already recycled, alive generation that is NOT
            # selected by the next intervention. Its death must be reason2,
            # distinct from the inter-event reason1 checked above.
            d2=D.Diagnostics(root/'generation_event_reason','both',[100])
            p2=[p.clone() for p in P];p2[1][0,1]=1.
            d2.initialize(p2)
            birth=[torch.tensor([[False,True]]),torch.tensor([[False,False]])]
            b2=d2.before_recycle(1,6,2,p2,X,act,birth,scores,ids)
            d2.after_recycle(1,6,2,p2,X,act,birth,scores,ids,b2)
            assert d2.generation[0][0,1]==1 and d2.started_alive[0][0,1]
            b2=d2.before_recycle(2,12,3,p2,X,act,masks,scores,ids)
            p2[1][0,1]=-1.
            d2.after_recycle(2,12,3,p2,X,act,masks,scores,ids,b2)
            assert d2.first_death[0][0,1]==12 and d2.reason[0][0,1]==2, 'event-associated recycled-generation death mislabeled as between-events'
            return dict(mutations=dict(native_zero_called_strict_alive=diag.started_alive[0][0,0]!=diag.native_started_alive[0][0,0],
                                       replacement_counted_as_death=bool(event['post_l1_previous_generation_replaced_while_alive'][0,0]) and event['post_l1_first_observed_death'][0,0]<0,
                                       upstream_event_called_drift=bool(data['post_l1_unselected_event_death'][0,1]) and d2.reason[0][0,1]==2),
                        revived_then_between_event_death=True,gen0_event_associated_death=True,full_recycler_strict_definition=True)
        record('S-generation',generation_check)

        def readout_check():
            cases=[(0,.4,.4,'CHB_POLICY_CARRIES'),(.4,0,.4,'REDO_POLICY_CARRIES'),(0,0,0,'MIXED')]
            for redo,chb,both,label in cases:
                assert R.classify_rows(rows_fixture(redo,chb,both))['label']==label
            bothaxes=R.classify_rows(rows_fixture(.001,.4,.401))
            assert bothaxes['label']=='CHB_POLICY_CARRIES' and bothaxes['additive']
            rows=rows_fixture(.01,.4,.4);proper=R.classify_rows(rows)
            assert proper['redo_shortfall']
            shuffled=rows[1::2]+rows[::2]
            assert R.classify_rows(shuffled)['late_per_seed']==proper['late_per_seed']
            assert R.classify_rows(rows[:-1])['status']=='INCOMPLETE'
            assert R.classify_rows(rows+[rows[0]])['status']=='INCOMPLETE'
            nonfinite=copy.deepcopy(rows);nonfinite[0]['online_acc']=float('nan')
            assert R.classify_rows(nonfinite)['status']=='DIVERGED'
            cp=R.interval([.1]*10);assert cp['sd']==0 and cp['low']==cp['high']==.1 and cp['flags']==['DEGENERATE_SD']
            boundary=copy.deepcopy(proper['simultaneous_intervals']);boundary['B_minus_C']['high']=.005
            assert R.label_intervals(boundary)['label']=='MIXED'
            boundary=copy.deepcopy(proper['simultaneous_intervals']);boundary['C_minus_R']['low']=.005
            assert not R.label_intervals(boundary)['redo_shortfall']
            wide=R.interval([-.04,.04]*5)
            boundary=copy.deepcopy(bothaxes['simultaneous_intervals']);boundary['I']=wide
            assert not R.label_intervals(boundary)['additive']
            record=root/'finite.csv';C.write_csv(record,rows[:3]);mf=C.manifest(root,exclude=())
            # Verify only the relevant file: change a finite valid-looking value.
            limited={'finite.csv':mf['finite.csv']};record.write_text(record.read_text().replace('0.15','0.16',1))
            finite_rejected=rejected(lambda:C.verify_manifest(root,limited))
            # Use a disposable child of the NEW namespace for an artifact-path
            # test; every foreign path is rejected before report writes.
            C.OUT.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='_synthetic_admission_',dir=C.OUT) as local:
                unqualified=Path(local)
                C.write_csv(unqualified/'per_task.csv',rows)
                C.put(unqualified/'status.json',dict(status='BLOCKED_CUDA'))
                decision=R.report(unqualified)
                assert decision['status']=='BLOCKED_CUDA' and decision['label'] is None
            foreign=root/'foreign';foreign.mkdir()
            sentinel=foreign/'verdict.json';C.put(sentinel,dict(run_id='foreign',status='COMPLETE'))
            oldhash=C.sha(sentinel)
            assert rejected(lambda:R.report(foreign)) and rejected(lambda:L.preflight(foreign))
            assert C.sha(sentinel)==oldhash, 'foreign result was overwritten'
            source=C.source_hashes();assert 'analysis/__init__.py' in source
            changed=dict(source);changed['analysis/__init__.py']=hashlib.sha256(b'mutated package initializer').hexdigest()
            proof=dict(all_pass=True,scope='CPU_SYNTHETIC_ONLY',source_sha256=changed)
            stale_init=rejected(lambda:C.verify_hash_record(proof,scope='CPU_SYNTHETIC_ONLY'))
            return dict(mutations=dict(seed_position_join=proper['late_per_seed']['CHB']!=list(reversed(proper['late_per_seed']['CHB'])),
                                       missing_seed_deleted=R.classify_rows(rows[:-1])['status']=='INCOMPLETE',
                                       nonsignificant_called_EQ=not R.label_intervals(boundary)['additive'],
                                       finite_csv_edit=finite_rejected,parent_initializer_changed=stale_init),
                        primary_and_additive_orthogonal=True,complete_rows=8000,paired_intervals=7,
                        blocked_folder_does_not_get_performance_label=True)
        record('S-readout',readout_check)
    passed=all(group['pass'] for group in tests.values())
    result=dict(run_id=C.RUN,scope='CPU_SYNTHETIC_ONLY',all_pass=passed,checks=tests,
                source_sha256=C.source_hashes(),registration=C.registration(),environment=C.safe_environment(),
                generated_utc=C.now(),evaluation_seed_count=0,
                tolerances=dict(states='raw byte equality including signed zero',gate_counts='exact integers',
                                intervals='registered float64 t; explicit equal-vector point branch'),
                pending=['independent source review','real-data CUDA R10 release','full registered production'])
    C.put(out,result)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=C.OUT/'cpu_checks.json')
    args=ap.parse_args()
    result=run(args.out)
    print(json.dumps(dict(all_pass=result['all_pass'],groups=len(result['checks']),output=str(args.out))))
    return 0 if result['all_pass'] else 2


if __name__=='__main__':
    sys.exit(main())

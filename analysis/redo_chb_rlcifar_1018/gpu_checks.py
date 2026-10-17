#!/usr/bin/env python3
"""Actual CUDA/R=10 release checks for the registered RC ReDo x CHB campaign.

These are named calibration-seed verification trajectories, not production arms.
They retain real CIFAR, native dimensions, raw preprocessing, batch16, 75 steps
per epoch, actual tau=.1/period1560, and original native forward/Adam arithmetic.
Only their task/epoch counts are short fixtures. Production lives in launch.py.

This standalone command owns launch.gpu_lock exactly once. A queue must invoke it
without an enclosing GPU lock. CPU/unavailable CUDA never yields a passing proof.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from analysis.redo_chb_rlcifar_1018 import common as C
from analysis.redo_chb_rlcifar_1018 import launch as L
from src import redo_chb_rlcifar_1018 as E

REQUIRED = tuple(L.GPU_REQUIRED)
EXTRA = ('ReDo_cadence_64probe', 'source_data_review_admission', 'release_artifacts')
SCOPE = 'CUDA_REAL_DATA_RELEASE'
CORE_KEYS = ('P', 'm', 'v', 'tc', 'act_state', 'g_lab', 'g_batch', 'alive')


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k:cpu(v) for k,v in value.items()}
    if isinstance(value, list):
        return [cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu(v) for v in value)
    return copy.deepcopy(value)


def core(checkpoint):
    return {key:checkpoint[key] for key in CORE_KEYS}


def assert_same(actual, expected, description):
    a, b = C.digest(actual), C.digest(expected)
    assert a == b, description
    return a


def independent_recycle(P, m, v, masks, generator_states, seeds):
    """CPU reference, all incoming rows before all outgoing columns.

    It never calls the implementation recycler or changes its generators. Every
    CPU draw uses a cloned pre-event generator state and the native init formula.
    """
    Q, M, V = cpu(P), cpu(m), cpu(v)
    generators = {}
    for seed in seeds:
        generators[seed] = torch.Generator(device='cpu')
        generators[seed].set_state(generator_states[seed])
    for layer in range(2):
        wi, bi = 2*layer, 2*layer+1
        fan = Q[wi].shape[-1]
        for slot,seed in enumerate(seeds):
            ids = torch.nonzero(masks[layer][slot], as_tuple=False).flatten()
            values = (torch.rand((len(ids),fan), generator=generators[seed], dtype=torch.float32)*2-1)
            values *= 1/math.sqrt(fan)
            Q[wi][slot,ids,:] = values
            Q[bi][slot,ids] = 0
            M[wi][slot,ids,:] = 0
            V[wi][slot,ids,:] = 0
    for layer in range(2):
        output = 2*layer+2
        for slot,_seed in enumerate(seeds):
            ids = torch.nonzero(masks[layer][slot], as_tuple=False).flatten()
            Q[output][slot,:,ids] = 0
            M[output][slot,:,ids] = 0
            V[output][slot,:,ids] = 0
    return dict(P=Q, m=M, v=V,
                generator_states={s:g.get_state() for s,g in generators.items()})


class Audit:
    """Read-only callback with independent stream/event accounting and references."""
    def __init__(self, arm, epochs, *, adapted, stop_after=None):
        self.arm, self.epochs, self.adapted = arm, epochs, adapted
        self.stop_after = stop_after
        self.ready_records, self.task_records, self.event_records = [], {}, []
        self.current = None
        self.order = self.previous_tail = None
        self.pending = None

    def state(self):
        return cpu(dict(arm=self.arm, epochs=self.epochs, adapted=self.adapted,
                        ready_records=self.ready_records, task_records=self.task_records,
                        event_records=self.event_records, current=self.current,
                        order=self.order, previous_tail=self.previous_tail,
                        pending=self.pending))

    def load_state(self, state):
        assert (state['arm'],state['epochs'],state['adapted']) == (self.arm,self.epochs,self.adapted)
        for key in ('ready_records','task_records','event_records','current','order','previous_tail','pending'):
            setattr(self,key,copy.deepcopy(state[key]))

    def _finish_task(self, c, task):
        assert self.current is not None and self.current['task'] == task
        assert self.current['epochs'] == self.epochs, 'fixture lost a native epoch'
        record = dict(self.current)
        record['g_lab'] = C.digest({s:g.get_state() for s,g in c['g_lab'].items()})
        record['g_batch'] = C.digest({s:g.get_state() for s,g in c['g_batch'].items()})
        record['tc'] = int(c['tc'])
        assert record['tc'] == task*self.epochs*75, 'global Adam time was reset at task boundary'
        self.task_records[str(task)] = record

    def __call__(self, event, c):
        if event == 'ready':
            assert len(c['slots']) == 10 and c['seeds'] == C.CHECK_SEEDS
            assert c['conds'] == ['raw'] and c['X'].shape == (10,1200,3072)
            assert c['device'].type == 'cuda' and c['use_graph'] == (c['cg'] is not None)
            assert all(p.dtype == torch.float32 and p.device.type == 'cuda' for p in c['P'])
            if c.get('resume') and c.get('st') is not None:
                if self.adapted and str(c['st']['t']) not in self.task_records:
                    self._finish_task(c,int(c['st']['t']))
            else:
                expected = {s:E.E.H.init_params(s,'cuda',E.E.DIMS) for s in C.CHECK_SEEDS}
                expected_P = [torch.stack([expected[s][k] for s in C.CHECK_SEEDS]) for k in range(6)]
                assert_same(c['P'],expected_P,'capture did not rewind original initialized parameters')
                assert c['tc'] == 0 and int(c['step_t']) == 0
                assert bool((c['acc_sum']==0).all()) and bool((c['bad_step']==-1).all())
                assert bool((c['last_hit']==0).all())
                assert all(bool((p==0).all()) for p in (*c['adam_m'],*c['adam_v']))
                for seed in C.CHECK_SEEDS:
                    assert_same(c['g_lab'][seed].get_state(),E.E.H.stream('rlc_labels',seed).get_state(),
                                'capture consumed a label stream')
                    assert_same(c['g_batch'][seed].get_state(),E.E.H.stream('rlc_batch',seed).get_state(),
                                'capture consumed a batch stream')
                if self.arm in ('CHB','both'):
                    assert all(bool((p==0).all()) for p in c['act'].state()['m'])
                    assert bool((c['P'][1] != 0).any()), 'b1 was incorrectly zeroed before first real update'
                if self.adapted:
                    control=c['lifecycle']
                    assert control.event_count == 0 and control.last_tc == 0 and control.count == 0
                    assert bool((control.ring==0).all()), 'warmup entered real-step ring'
                    for s in C.CHECK_SEEDS:
                        assert_same(control.g_reinit[s].get_state(),E.E.H.stream('b5_redo_reinit',s).get_state(),
                                    'warmup consumed recycle RNG')
            self.ready_records.append(dict(tc=int(c['tc']),graph=bool(c['use_graph']),
                                           input_sha256=C.digest(c['X']),
                                           initial_or_restored_core_sha256=C.digest(dict(
                                               P=c['P'],m=c['adam_m'],v=c['adam_v'],tc=c['tc'],
                                               act=c['act'].state()))))
        elif event == 'task_labels':
            self.current=dict(task=int(c['t']),label_sha256=C.digest(c['Y']),order_chain='',epochs=0)
        elif event == 'epoch_order':
            assert self.current is not None and self.current['task'] == int(c['t'])
            if self.order is not None:
                self.previous_tail=self.order[:,-48:].clone()
            self.order=c['ORD_cpu'].detach().cpu().clone()
            expected=torch.stack([c['order'][s] for s,cd in c['slots']])
            assert_same(self.order,expected,'delivered epoch permutation differs from native CPU order')
            self.current['epochs'] += 1
            self.current['order_chain']=hashlib.sha256(
                (self.current['order_chain']+C.digest(self.order)).encode()).hexdigest()
        elif event == 'redo_pre':
            self._before_event(c)
        elif event == 'redo_post':
            self._after_event(c)
        elif event == 'task_end':
            assert not c['diverged'] and bool(c['alive'].all()), 'verification fixture diverged'
            if self.adapted:
                self._finish_task(c,int(c['t']))
            return self.stop_after is not None and int(c['t']) >= self.stop_after
        return False

    @torch.no_grad()
    def _before_event(self,c):
        event=c['redo_event']; tc=int(c['tc']); j=int(c['j'])
        assert tc == ((int(c['t'])-1)*self.epochs+int(c['e']))*75+j+1
        assert tc == int(event['tc']) == (len(self.event_records)+1)*1560
        assert int(event['index']) == len(self.event_records)+1
        if j >= 3:
            expected_ids=self.order[:,(j-3)*16:(j+1)*16]
        else:
            assert self.previous_tail is not None
            expected_ids=torch.cat((self.previous_tail[:,-(3-j)*16:],self.order[:,:(j+1)*16]),1)
        assert expected_ids.shape == (10,64)
        assert_same(event['probe_ids'].cpu(),expected_ids,'ReDo did not use exactly last four real batches')
        ar=torch.arange(10,device='cuda')[:,None]
        probe=c['X'][ar,expected_ids.to('cuda')]
        vals=E.E.forward(c['P'],probe,c['act'],train=False)
        expected_scores=[]; expected_masks=[]
        # Recompute from features actually sent to the following affine map,
        # including H. No use of implementation score_masks/recycle helpers.
        for hidden in (vals[1],vals[3]):
            activity=hidden.abs().mean(dim=1)
            score=activity/(activity.mean(dim=1,keepdim=True)+1e-9)
            expected_scores.append(score)
            expected_masks.append((score<=.1).cpu())
        assert_same(event['scores'],expected_scores,'score used a pre-H or stale feature')
        assert_same(event['masks'],expected_masks,'score threshold/mask differs')
        control=c['lifecycle']
        generators={s:g.get_state().clone() for s,g in control.g_reinit.items()}
        expected=independent_recycle(c['P'],c['adam_m'],c['adam_v'],expected_masks,generators,C.CHECK_SEEDS)
        nonparam=dict(tc=tc,act=c['act'].state(),bias_m=[c['adam_m'][k] for k in (1,3,5)],
                      bias_v=[c['adam_v'][k] for k in (1,3,5)],
                      label_rng={s:g.get_state() for s,g in c['g_lab'].items()},
                      batch_rng={s:g.get_state() for s,g in c['g_batch'].items()})
        self.pending=dict(expected=expected,nonparam_sha256=C.digest(nonparam),masks=cpu(expected_masks),
                          generation=[a.copy() for a in control.observer.generation],
                          tc=tc,task=int(c['t']),epoch=int(c['e']),batch=j,
                          probe_sha256=C.digest(expected_ids),score_sha256=C.digest(expected_scores),
                          chosen=[int(m.sum()) for m in expected_masks],
                          at_task_final_update=(j==74 and int(c['e'])==self.epochs-1))

    @torch.no_grad()
    def _after_event(self,c):
        assert self.pending is not None
        p=self.pending; expected=p['expected']; control=c['lifecycle']
        actual=dict(P=cpu(c['P']),m=cpu(c['adam_m']),v=cpu(c['adam_v']),
                    generator_states={s:g.get_state() for s,g in control.g_reinit.items()})
        assert_same(actual,expected,'recycler draw/order/output-column/moment semantics differ from reference')
        nonparam=dict(tc=c['tc'],act=c['act'].state(),bias_m=[c['adam_m'][k] for k in (1,3,5)],
                      bias_v=[c['adam_v'][k] for k in (1,3,5)],
                      label_rng={s:g.get_state() for s,g in c['g_lab'].items()},
                      batch_rng={s:g.get_state() for s,g in c['g_batch'].items()})
        assert C.digest(nonparam)==p['nonparam_sha256'], 'ReDo changed EMA/bias moments/time/training RNG'
        for old,mask,new in zip(p['generation'],p['masks'],control.observer.generation):
            assert np.array_equal(old+mask.numpy().astype(np.int64),new),'generation state omitted an actual reset'
        record={k:v for k,v in p.items() if k not in ('expected','masks','generation')}
        record['post_model_adam_reinit_sha256']=C.digest(actual)
        self.event_records.append(record)
        self.pending=None


def run_fixture(base,arm,cifar,*,tasks,epochs,graph,adapted=True,resume=False,stop_after=None):
    base=Path(base)
    C.require_run_namespace(base)
    if resume and adapted:
        E.verify_checkpoint_file(base/'ckpt.pt')
    audit=Audit(arm,epochs,adapted=adapted,stop_after=stop_after)
    runner=E.build_runner() if adapted else E.E.run
    native_arm=arm if adapted else ('ref' if arm=='none' else 'CHB')
    runner(native_arm,C.CHECK_SEEDS,['raw'],tasks,epochs,torch.device('cuda'),base,
           lr=.001,lam=.1713,beta=.01,cifar=cifar,checkpoint=True,resume=resume,
           graph=graph,snapshots=False,lifecycle=audit,progress=lambda _:None)
    if adapted:
        E.verify_checkpoint_file(base/'ckpt.pt')
    checkpoint=C.load_pt(base/'ckpt.pt')
    assert not checkpoint['diverged'] and bool(checkpoint['alive'].all())
    assert len(checkpoint['P'])==6 and checkpoint['P'][0].shape==(10,100,3072)
    assert checkpoint['P'][4].shape==(10,10,100)
    if adapted:
        ext=checkpoint['extension_state']
        assert ext['period']==1560 and ext['tau']==.1 and ext['seeds']==C.CHECK_SEEDS
        assert ext['last_tc']==checkpoint['tc']
        assert ext['event_count']==checkpoint['tc']//1560 if arm in ('redo','both') else ext['event_count']==0
        assert ext['probe_count']==4 and tuple(ext['probe_ring'].shape)==(10,4,16)
    return checkpoint,audit


def compare_per_task(first,second):
    a,b=C.read_csv(Path(first)/'per_task.csv'),C.read_csv(Path(second)/'per_task.csv')
    # Names of equivalent arms differ, while every numerical metric must match.
    strip=lambda rows:[{k:v for k,v in row.items() if k!='arm'} for row in rows]
    assert_same(strip(a),strip(b),'native/ref verification task rows differ')
    return dict(rows=len(a),normalized_rows_sha256=C.digest(strip(a)))


def verify_real_data():
    archive=L.archive_record()
    cifar=E.E.RC.Cifar10()
    assert cifar.sha256=={archive['archive']:archive['sha256']}
    assert tuple(cifar.train_u8.shape)==(50000,3072)
    assert E.E.DIMS==(3072,100,100,10) and E.E.N_IMAGES==1200 and E.E.BATCH==16 and E.E.STEPS_PER_EPOCH==75
    inputs={}; initials={}; subset={}
    for seed in C.CHECK_SEEDS:
        rows=E.E.RC.subset_idx(seed)
        assert len(rows)==1200 and len(set(rows.tolist()))==1200
        raw=E.E.slot_inputs(cifar,seed,'raw','cpu',center=False)
        independent=cifar.train_u8[rows].to(dtype=torch.float32)/255
        assert_same(raw,independent,'raw path applied standardization or changed subset')
        centered=E.E.slot_inputs(cifar,seed,'raw','cpu',center=True)
        assert_same(centered,raw-raw.mean(0),'CHB input centering differs from own subset mean image')
        inputs[str(seed)]=dict(raw_sha256=C.digest(raw),centered_sha256=C.digest(centered))
        initials[str(seed)]=C.digest(E.E.H.init_params(seed,'cpu',E.E.DIMS))
        subset[str(seed)]=dict(ids_sha256=C.digest(rows),count=1200)
    return cifar,dict(archive=archive,inputs=inputs,initials=initials,subset=subset,
                      native_dimensions=[3072,100,100,10],R=10,check_seeds=C.CHECK_SEEDS)


def semantic_release(artifacts,source_snapshot):
    """The six required checks execute actual CUDA native trajectories."""
    artifacts=Path(artifacts)
    assert not artifacts.exists(), 'refusing to overwrite previous release verification artifacts'
    artifacts.mkdir(parents=True)
    checks={}
    cifar,data=verify_real_data()
    C.put(artifacts/'real_data_identity.json',data)
    fixture_records={}; parity_audits={}
    for arm,key in (('none','R10_none_native'),('CHB','R10_CHB_native')):
        target=artifacts/(arm+'_adapted_graph')
        reference=artifacts/(arm+'_native_graph')
        actual,aa=run_fixture(target,arm,cifar,tasks=2,epochs=1,graph=True)
        expected,na=run_fixture(reference,arm,cifar,tasks=2,epochs=1,graph=True,adapted=False)
        state_sha=assert_same(core(actual),core(expected),'new '+arm+' differs from unmodified native host')
        row_proof=compare_per_task(target,reference)
        checks[key]=dict(pass_=True,fixture='real-data R10 2 tasks x 1 epoch',core_sha256=state_sha,**row_proof)
        parity_audits[arm]=aa
        fixture_records[str(target.relative_to(artifacts))]=dict(tasks=2,epochs=1,graph=True,arm=arm)
        fixture_records[str(reference.relative_to(artifacts))]=dict(tasks=2,epochs=1,graph=True,arm=arm,native=True)
        # Short no-event graph/eager parity keeps native operation shapes.
        eager_dir=artifacts/(arm+'_adapted_eager')
        eager,ea=run_fixture(eager_dir,arm,cifar,tasks=2,epochs=1,graph=False)
        assert_same(core(actual),core(eager),'graph/eager native '+arm+' differs')
        compare_per_task(target,eager_dir)
        fixture_records[str(eager_dir.relative_to(artifacts))]=dict(tasks=2,epochs=1,graph=False,arm=arm)
    both_graph=artifacts/'both_graph'
    both_eager=artifacts/'both_eager'
    graph,ga=run_fixture(both_graph,'both',cifar,tasks=2,epochs=21,graph=True)
    eager,ea=run_fixture(both_eager,'both',cifar,tasks=2,epochs=21,graph=False)
    assert_same(core(graph),core(eager),'ReDo graph/eager model or native RNG differs')
    extension_keys=('last_tc','event_count','probe_ring','probe_cursor','probe_count','reinit_rng','diagnostics')
    assert_same({k:graph['extension_state'][k] for k in extension_keys},
                {k:eager['extension_state'][k] for k in extension_keys},'ReDo graph/eager controller differs')
    compare_per_task(both_graph,both_eager)
    assert [r['tc'] for r in ga.event_records]==[1560,3120]
    assert_same(ga.event_records,ea.event_records,'graph/eager event histories differ')
    checks['R10_graph_eager']=dict(pass_=True,arms=['none','CHB','both'],R=10,
                                  redo_event_times=[1560,3120],native_step_shapes=True)
    # Snapshot readiness checks independently regenerate native initial tensors,
    # moments, EMA and generators before any real batch or recycler is touched.
    assert all(a.ready_records and a.ready_records[0]['tc']==0 for a in (*parity_audits.values(),ga,ea))
    checks['capture_rollback']=dict(pass_=True,initial_parameters_moments_EMA_streams_byte_equal=True,
                                    real_step_count_at_ready=0,real_recycle_events_at_ready=0)
    # Task boundary is deliberately not divisible by 1560: 21*75=1575.
    # Resume happens after an actual event and before the next event.
    resumed_dir=artifacts/'both_resume'
    prefix,pa=run_fixture(resumed_dir,'both',cifar,tasks=2,epochs=21,graph=True,stop_after=1)
    assert prefix['t']==1 and prefix['tc']==1575 and prefix['extension_state']['event_count']==1
    resumed,ra=run_fixture(resumed_dir,'both',cifar,tasks=2,epochs=21,graph=True,resume=True)
    assert_same(core(graph),core(resumed),'event checkpoint resume differs from continuous native trajectory')
    assert_same({k:graph['extension_state'][k] for k in extension_keys},
                {k:resumed['extension_state'][k] for k in extension_keys},'resume lost ring/generator/lineage state')
    assert_same(ga.task_records,ra.task_records,'resume changed actual task label/order hashes')
    assert_same(ga.event_records,ra.event_records,'resume changed actual event histories')
    compare_per_task(both_graph,resumed_dir)
    checks['event_checkpoint_resume']=dict(pass_=True,checkpoint_tc=1575,event_before_resume=1560,
                                          event_after_resume=3120,model_Adam_EMA_rng_ring_generation_exact=True)
    # Minimal native epoch budget reaching a task-final recycle: 104*75=7800.
    # This tests evaluation/checkpoint after the final native update and event.
    redo_dir=artifacts/'redo_boundary_graph'
    redo,da=run_fixture(redo_dir,'redo',cifar,tasks=1,epochs=104,graph=True)
    event_times=[r['tc'] for r in da.event_records]
    assert event_times==[1560,3120,4680,6240,7800]
    assert da.event_records[-1]['at_task_final_update'] is True
    assert redo['tc']==7800 and redo['extension_state']['event_count']==5
    boundary_checkpoint=dict(P=redo['P'],m=redo['m'],v=redo['v'],
                             generator_states=redo['extension_state']['reinit_rng'])
    assert C.digest(boundary_checkpoint)==da.event_records[-1]['post_model_adam_reinit_sha256'], \
        'task-final checkpoint preceded its final recycle'
    assert any(sum(r['chosen']) for r in ga.event_records+da.event_records), 'fixture did not recycle any units'
    checks['ReDo_cadence_64probe']=dict(pass_=True,period=1560,tau=.1,probe_images=64,probe_batches=4,
                                     both_event_times=[1560,3120],redo_event_times=event_times,
                                     exact_task_final_event=7800,
                                     independent_recycler_reference=True,
                                     bias_moments_global_time_EMA_preserved=True)
    # Same underlying data/initialization; CHB's centering is the registered change.
    assert_same(parity_audits['none'].task_records,parity_audits['CHB'].task_records,
                'none/CHB native label/order streams differ')
    assert_same(ga.task_records,ea.task_records,'both graph/eager label/order streams differ')
    checks['same_data_streams']=dict(pass_=True,archive_sha256=data['archive']['sha256'],
                                   subset_and_initial_sha256=C.digest(dict(subset=data['subset'],initials=data['initials'])),
                                   paired_native_task_stream_sha256=C.digest(parity_audits['none'].task_records),
                                   paired_both_task_stream_sha256=C.digest(ga.task_records),
                                   center_is_intended_difference=True)
    for name,arm,epochs in (('both_graph','both',21),('both_eager','both',21),('both_resume','both',21),
                             ('redo_boundary_graph','redo',104)):
        fixture_records[name]=dict(arm=arm,tasks=1 if arm=='redo' else 2,epochs=epochs,
                                  graph=name!='both_eager',check_seeds=C.CHECK_SEEDS,production=False)
    C.put(artifacts/'fixture_protocols.json',fixture_records)
    C.put(artifacts/'event_and_stream_checks.json',dict(
        both_events=ga.event_records,redo_boundary_events=da.event_records,
        both_task_streams=ga.task_records,native_task_streams=parity_audits['none'].task_records))
    assert C.source_hashes()==source_snapshot,'source changed during CUDA release checks'
    assert L.archive_record()==data['archive'],'real dataset changed during CUDA release checks'
    return checks,data


def prerequisite_review(proof_root,sources):
    blockers=[];proofs={}
    for name,scope in (('cpu_checks.json','CPU_SYNTHETIC_ONLY'),('independent_review.json','CPU_SOURCE_REVIEW')):
        path=Path(proof_root)/name
        try:
            value=json.loads(path.read_text())
            assert value.get('scope')==scope,'scope mismatch'
            assert value.get('source_sha256')==sources,'stale source closure'
            if name=='cpu_checks.json':
                assert value.get('all_pass') is True,'CPU semantic checks did not pass'
                assert value.get('checks') and all(v.get('pass') is True for v in value['checks'].values())
            else:
                assert value.get('approved') is True,'independent source review not approved'
                reviewer=value.get('reviewer','')
                assert reviewer and reviewer not in ('/root/run1_redo_position','/root/run3_chb'), 'source review not independent'
            proofs[name]=dict(path=str(path.resolve()),sha256=C.sha(path))
        except (OSError,ValueError,KeyError,AssertionError) as exc:
            blockers.append('PENDING_SOURCE_BOUND_PROOF:'+name+':'+str(exc))
    dirty=C.git('status','--porcelain','--','src','analysis','specs')
    if dirty:
        blockers.append('UNCOMMITTED_SOURCE: final reviewed source must be committed before CUDA release')
    return blockers,proofs


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=C.OUT/'gpu_release.json')
    ap.add_argument('--artifacts',type=Path,default=C.OUT/'gpu_release_artifacts')
    ap.add_argument('--proof-root',type=Path,default=C.OUT)
    args=ap.parse_args(argv)
    # All persistent evidence belongs to the NEW run. In particular the CLI
    # cannot overwrite the old F51 experiment through an arbitrary --out.
    C.require_run_namespace(args.out.parent)
    C.require_run_namespace(args.artifacts)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    blockers=[]
    try:
        sources=C.source_hashes()
    except (OSError,AssertionError) as exc:
        sources={};blockers.append('SOURCE_CLOSURE_INCOMPLETE:'+str(exc))
    registration=C.registration()
    environment=C.safe_environment()
    try:
        archive=L.archive_record()
    except OSError as exc:
        archive=None;blockers.append('BLOCKED_DATA:'+str(exc))
    proof_errors,proofs=prerequisite_review(args.proof_root,sources)
    blockers.extend(proof_errors)
    if not environment['cuda_available']:
        blockers.insert(0,'BLOCKED_CUDA: real CUDA checks cannot execute on CPU')
    record=dict(run_id=C.RUN,scope=SCOPE,all_pass=False,
                status='BLOCKED_CUDA' if not environment['cuda_available'] else 'BLOCKED_ADMISSION',
                source_sha256=sources,registration=registration,check_seeds=C.CHECK_SEEDS,
                device='cuda',R=10,archive_sha256=None if archive is None else archive['sha256'],
                archive=archive,environment=environment,checked_utc=C.now(),
                git_hash=C.git('rev-parse','HEAD'),proofs=proofs,blockers=blockers,
                checks={key:dict(pass_=False,status='NOT_RUN',reason='pending genuine CUDA verification') for key in REQUIRED+EXTRA},
                production_started=False,verification_seeds_only=True,
                fixture_policy='native real-data R10 tests may shorten only verification tasks/epochs; production remains 200x400x75')
    cpu_path=args.proof_root/'cpu_checks.json'
    cpu_evidence=dict(path=str(cpu_path),available=cpu_path.is_file(),
                      source_bound=False,all_pass=False,
                      note='CPU semantic evidence cannot substitute for real CUDA release checks')
    if cpu_path.is_file():
        try:
            proof=json.loads(cpu_path.read_text())
            cpu_evidence.update(sha256=C.sha(cpu_path),scope=proof.get('scope'),
                                all_pass=proof.get('all_pass') is True,
                                source_bound=proof.get('source_sha256')==sources)
        except (OSError,ValueError):
            cpu_evidence['unreadable']=True
    record['cpu_evidence']=cpu_evidence
    if not blockers:
        try:
            # Exactly one shared lock owner for this standalone checker.
            with L.gpu_lock():
                assert C.source_hashes()==sources,'source closure changed before GPU lock acquisition'
                assert C.registration()==registration,'preregistration changed before GPU lock acquisition'
                assert L.archive_record()==archive,'archive changed before GPU lock acquisition'
                post_errors,post_proofs=prerequisite_review(args.proof_root,sources)
                assert not post_errors and post_proofs==proofs,'CPU/source-review proof changed before lock'
                checks,data=semantic_release(args.artifacts,sources)
            checks['source_data_review_admission']=dict(pass_=True,proofs=proofs,
                                                        source_unchanged=True,data_unchanged=True)
            files=C.manifest(args.artifacts)
            C.verify_manifest(args.artifacts,files)
            record.update(data_identity=data,artifact_root=str(args.artifacts.resolve()),files=files)
            checks['release_artifacts']=dict(pass_=True,files=len(files),manifest_sha256=C.digest(files))
            record.update(checks=checks,status='PASS',all_pass=all(v.get('pass_') is True for v in checks.values()))
        except RuntimeError as exc:
            if str(exc).startswith('BLOCKED_GPU_BUSY:'):
                record.update(status='BLOCKED_GPU_BUSY',blockers=[str(exc)])
            else:
                record.update(status='CHECK_FAILED',blockers=[repr(exc)],traceback=traceback.format_exc())
        except (Exception,SystemExit) as exc:
            record.update(status='CHECK_FAILED',blockers=[repr(exc)],traceback=traceback.format_exc())
    for check in record['checks'].values():
        if 'pass_' in check:
            check['pass']=check.pop('pass_')
    C.put(args.out,record)
    print(json.dumps(dict(status=record['status'],all_pass=record['all_pass'],
                          blockers=record['blockers'],out=str(args.out))),flush=True)
    return 0 if record['all_pass'] else 2


if __name__=='__main__':
    raise SystemExit(main())

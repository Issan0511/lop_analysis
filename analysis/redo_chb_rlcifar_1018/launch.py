"""Full registered RC/raw 200-task campaign; one shared GPU lock owner.

This CLI owns /tmp/lop_analysis_gpu.lock. Queue wrappers must not hold the same
lock around it. No external host queue, old F51 result, or active G0 is modified.
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from analysis.redo_chb_rlcifar_1018 import common as C
from analysis.redo_chb_rlcifar_1018 import report as R

PROOF_NAMES = ('cpu_checks.json', 'independent_review.json', 'gpu_release.json')
GPU_REQUIRED = ('R10_none_native', 'R10_CHB_native', 'R10_graph_eager',
                'capture_rollback', 'event_checkpoint_resume', 'same_data_streams')


def archive_record():
    from src import pmnist_rlcifar_0907 as RC
    path = RC.DATA_DIR / RC.ARCHIVE
    return dict(path=str(path), resolved=str(path.resolve()), bytes=path.stat().st_size,
                sha256=C.sha(path), archive=RC.ARCHIVE)


def validate_proofs(root, data):
    root = Path(root)
    issues, proofs = [], {}
    for name in PROOF_NAMES:
        try:
            record = json.loads((root/name).read_text())
            if name == 'independent_review.json':
                assert record.get('approved') is True
                assert record.get('scope') == 'CPU_SOURCE_REVIEW'
                assert record.get('reviewer') and record['reviewer'] not in ('/root/run1_redo_position','/root/run3_chb')
                assert record.get('source_sha256') == C.source_hashes()
            else:
                C.verify_hash_record(record, scope='CPU_SYNTHETIC_ONLY' if name == 'cpu_checks.json' else 'CUDA_REAL_DATA_RELEASE')
                assert record.get('checks') and all(v.get('pass') is True for v in record['checks'].values())
                if name == 'gpu_release.json':
                    assert record.get('registration') == C.registration()
                    assert record.get('check_seeds') == C.CHECK_SEEDS
                    assert record.get('device') == 'cuda' and record.get('R') == 10
                    assert all(record['checks'].get(k,{}).get('pass') is True for k in GPU_REQUIRED)
                    assert record.get('archive_sha256') == data['sha256']
            proofs[name] = record
        except (OSError, ValueError, KeyError, AssertionError) as exc:
            issues.append('PROOF_MISSING_STALE_OR_FAILED:' + name + ':' + str(exc))
    return issues, proofs


def preflight(out=C.OUT, proof_root=None):
    out = Path(out)
    C.require_run_namespace(out)
    proof_root = Path(proof_root or out)
    registration = C.registration()
    sources = C.source_hashes()
    environment = C.safe_environment()
    blockers = []
    if not environment['cuda_available']:
        blockers.append('BLOCKED_CUDA: torch.cuda.is_available() is False')
    try:
        data = archive_record()
    except OSError as exc:
        data = None; blockers.append('BLOCKED_DATA:' + str(exc))
    proof_errors, proofs = validate_proofs(proof_root, data or {})
    blockers.extend(proof_errors)
    dirty = C.git('status', '--porcelain', '--', 'src', 'analysis', 'specs')
    if dirty:
        blockers.append('UNCOMMITTED_SOURCE: commit final reviewed implementation before production')
    if torch.__version__ != '2.13.0+cu130':
        blockers.append('TORCH_VERSION_CHANGED')
    status = 'BLOCKED_CUDA' if not environment['cuda_available'] else (
        'BLOCKED_ADMISSION' if blockers else 'ADMITTED')
    result = dict(run_id=C.RUN, status=status, all_pass=not blockers, blockers=blockers,
                  checked_utc=C.now(), registration=registration, protocol=C.PROTOCOL,
                  source_sha256=sources, environment=environment, archive=data,
                  proof_root=str(proof_root), proof_sha256={n:C.sha(proof_root/n) for n in proofs},
                  git_hash=C.git('rev-parse','HEAD'), argv=sys.argv, production_started=False,
                  scheduler=dict(lock=str(C.GPU_LOCK), owner='this CLI only', acquisition='nonblocking',
                                 external_existing_queues_modified=False))
    C.put(out/'preflight.json', result)
    if blockers:
        # Preserve completed campaign evidence when checking hardware offline.
        old = json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
        if old.get('status') != 'COMPLETE':
            C.put(out/'status.json', dict(status=status, blockers=blockers, production_started=False, checked_utc=C.now()))
            R.report(out)
    return result


@contextlib.contextmanager
def gpu_lock():
    """Single lock owner. Busy means recorded execution blocker, not another queue."""
    with C.GPU_LOCK.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('BLOCKED_GPU_BUSY: shared lab lock is held by another campaign')
        yield


class StreamAudit:
    """Read-only hashes of actual labels and every CPU-generated epoch order."""
    def __init__(self, out, stop):
        self.out, self.stop = Path(out), Path(stop)
        self.tasks, self.current = {}, None
    def state(self):
        return dict(tasks=self.tasks, current=self.current)
    def load_state(self, state):
        self.tasks = state['tasks']; self.current = state['current']
    def __call__(self, event, c):
        if event == 'task_labels':
            self.current = dict(task=int(c['t']), label_hash=C.digest(c['Y']),
                                order_chain='', epoch_count=0)
        elif event == 'epoch_order':
            assert self.current is not None and self.current['task'] == c['t']
            actual = c['ORD_cpu']
            expected = torch.stack([c['order'][s] for s,cd in c['slots']])
            assert C.digest(actual) == C.digest(expected), 'delivered epoch order differs'
            # Chain hashes actual complete 1200-image epoch permutations, per slot.
            self.current['order_chain'] = hashlib.sha256(
                (self.current['order_chain'] + C.digest(actual)).encode()).hexdigest()
            self.current['epoch_count'] += 1
        elif event == 'ready':
            # A native task checkpoint precedes its observer callback. Recover
            # the already-complete pending stream row without advancing any RNG.
            if c.get('resume') and c.get('st') is not None and self.current is not None:
                saved_t = int(c['st']['t'])
                if str(saved_t) not in self.tasks and self.current['task'] == saved_t:
                    self._finish(c, saved_t)
        elif event == 'task_end':
            self._finish(c, int(c['t']))
            return self.stop.exists()
        return False
    def _finish(self, c, task):
        assert self.current is not None and self.current['task'] == task
        row = dict(self.current)
        row.pop('task')
        row['g_lab_hash'] = C.digest({s:g.get_state() for s,g in c['g_lab'].items()})
        row['g_batch_hash'] = C.digest({s:g.get_state() for s,g in c['g_batch'].items()})
        self.tasks[str(task)] = row
        C.put(self.out/'stream_audit.json', self.tasks)


def data_identity(cifar):
    from src import relu_doors_0919 as E
    subsets = {str(s):E.RC.subset_idx(s).tolist() for s in C.SEEDS}
    raw = torch.stack([E.slot_inputs(cifar,s,'raw','cpu',center=False) for s in C.SEEDS])
    initial = {str(s):E.H.init_params(s,'cpu',E.DIMS) for s in C.SEEDS}
    return dict(archive=archive_record(), data_sha256=cifar.sha256, subsets=subsets,
                subset_sha256={s:C.digest(ids) for s,ids in subsets.items()},
                initial_parameters_sha256={s:C.digest(p) for s,p in initial.items()},
                raw_input_sha256=C.digest(raw), slot_order=C.SEEDS)


def run(out, admission, arm='all'):
    from src import redo_chb_rlcifar_1018 as E
    from src import relu_doors_0919 as N
    out = Path(out)
    assert out.resolve().is_relative_to((C.ROOT/'results'/C.RUN).resolve()), 'production output must stay in new run namespace'
    assert admission['all_pass'] is True and admission['protocol'] == C.PROTOCOL
    assert admission['source_sha256'] == C.source_hashes() and admission['registration'] == C.registration(), 'source/spec changed after lock'
    assert not C.git('status','--porcelain','--','src','analysis','specs'), 'uncommitted source after lock'
    assert archive_record() == admission['archive'], 'archive path/bytes/hash changed after admission'
    torch.set_num_threads(2)
    device = N.H.setup('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert device.type == 'cuda' and torch.cuda.is_available()
    cifar = N.RC.Cifar10()
    assert cifar.sha256[N.RC.ARCHIVE] == admission['archive']['sha256'], 'loaded archive differs from admitted archive'
    data = data_identity(cifar)
    assert data['archive'] == admission['archive'], 'archive changed while constructing cohort'
    identity = C.proof_identity(data)
    if (out/'provenance.json').exists():
        old = json.loads((out/'provenance.json').read_text())
        assert old['identity'] == identity, 'resume data/source/protocol identity differs'
    else:
        C.put(out/'provenance.json', dict(run_id=C.RUN, identity=identity, started_utc=C.now(),
                                         git_hash=C.git('rev-parse','HEAD'), environment=C.safe_environment()))
    for name in PROOF_NAMES:
        src = Path(admission['proof_root'])/name
        assert C.sha(src) == admission['proof_sha256'][name], 'proof changed after admission'
        dst = out/'admission'/name; dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dst)
    C.put(out/'admission.json', admission)
    shutil.copyfile(C.SPEC,out/'spec.md')
    shutil.copyfile(C.COMMON_SPEC,out/'common_spec.md')
    targets = list(C.ARMS) if arm == 'all' else [arm]
    if arm == 'next':
        targets = [next((a for a in C.ARMS if not (out/'arms'/a/'done.json').exists()), None)]
        targets = [a for a in targets if a is not None]
    for current in targets:
        index = C.ARMS.index(current)
        assert all((out/'arms'/a/'done.json').exists() for a in C.ARMS[:index]), 'registered arm order must be preserved'
        for previous in C.ARMS[:index]:
            record=json.loads((out/'arms'/previous/'done.json').read_text())
            assert record['identity']==identity and record['status']=='COMPLETE', 'prior arm identity differs'
            C.verify_manifest(out/'arms'/previous,record['files'])
        folder = out/'arms'/current
        folder.mkdir(parents=True,exist_ok=True)
        marker = folder/'done.json'
        if marker.exists():
            saved = json.loads(marker.read_text())
            assert saved['identity'] == identity and saved['status'] == 'COMPLETE'
            C.verify_manifest(folder,saved['files'])
            continue
        if (out/'STOP').exists():
            C.put(out/'status.json',dict(status='STOPPED',next_arm=current,updated_utc=C.now()))
            return 'STOPPED'
        if (folder/'identity.json').exists():
            assert json.loads((folder/'identity.json').read_text()) == identity
        else:
            assert not (folder/'ckpt.pt').exists() and not (folder/'per_task.csv').exists(), 'unqualified previous arm artifacts'
            C.put(folder/'identity.json',identity)
        audit = StreamAudit(folder,out/'STOP')
        if (folder/'ckpt.pt').exists():
            E.verify_checkpoint_file(folder/'ckpt.pt')
        C.put(out/'status.json',dict(status='RUNNING',arm=current,updated_utc=C.now()))
        C.put(folder/'status.json',dict(status='RUNNING',arm=current,updated_utc=C.now()))
        try:
            E.run(current,C.SEEDS,['raw'],200,400,device,folder,lr=.001,lam=.1713,beta=.01,
                  cifar=cifar,checkpoint=True,resume=(folder/'ckpt.pt').exists(),graph=True,
                  snapshots=True,run_id=C.RUN,prereg_commit=C.PREREG_COMMIT,lifecycle=audit)
            checkpoint = C.load_pt(folder/'ckpt.pt')
            if checkpoint['diverged'] or not bool(checkpoint['alive'].all()):
                raise FloatingPointError('native divergent checkpoint')
            if checkpoint['t'] != 200:
                status='STOPPED'
                C.put(folder/'status.json',dict(status=status,completed_task=checkpoint['t']))
                C.put(out/'status.json',dict(status=status,arm=current,completed_task=checkpoint['t']))
                return status
            assert checkpoint['tc'] == 6000000
            ext = checkpoint['extension_state']
            assert ext['last_tc'] == 6000000
            assert ext['event_count'] == (3846 if current in ('redo','both') else 0)
            assert len(C.read_csv(folder/'per_task.csv')) == 2000
            streams = json.loads((folder/'stream_audit.json').read_text())
            assert set(streams) == {str(t) for t in range(1,201)}
            C.put(folder/'status.json',dict(status='COMPLETE',arm=current,finished_utc=C.now()))
            C.put(marker,dict(status='COMPLETE',arm=current,identity=identity,
                              files=C.manifest(folder,exclude=('done.json',))))
        except Exception as exc:
            message = str(exc)
            status = 'DIVERGED' if isinstance(exc,FloatingPointError) or message.startswith('DIVERGED:') else 'CHECK_FAILED'
            state = dict(status=status,arm=current,reason=repr(exc),updated_utc=C.now())
            C.put(folder/'status.json',state); C.put(out/'status.json',state)
            R.report(out)
            raise
    complete = all((out/'arms'/a/'done.json').exists() for a in C.ARMS)
    C.put(out/'status.json',dict(status='COMPLETE' if complete else 'READY_FOR_NEXT_ARM',updated_utc=C.now()))
    result = R.report(out)
    return result['status']


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('preflight','run'))
    ap.add_argument('--out',type=Path,default=C.OUT)
    ap.add_argument('--proof-root',type=Path)
    ap.add_argument('--arm',choices=('all','next',*C.ARMS),default='all')
    args=ap.parse_args()
    admission=preflight(args.out,args.proof_root)
    print(json.dumps(dict(status=admission['status'],blockers=admission['blockers'])),flush=True)
    if not admission['all_pass']:
        return 2
    if args.command=='preflight':
        return 0
    try:
        with gpu_lock():
            status=run(args.out,admission,args.arm)
    except RuntimeError as exc:
        if str(exc).startswith('BLOCKED_GPU_BUSY:'):
            old=json.loads((args.out/'status.json').read_text()) if (args.out/'status.json').exists() else {}
            C.put(args.out/'lock_attempt.json',dict(status='BLOCKED_GPU_BUSY',reason=str(exc),production_started=False,checked_utc=C.now()))
            if old.get('status')!='COMPLETE':
                C.put(args.out/'status.json',dict(status='BLOCKED_GPU_BUSY',reason=str(exc),production_started=False))
                R.report(args.out)
            return 2
        raise
    return 0 if status in ('COMPLETE','READY_FOR_NEXT_ARM') else 2


if __name__=='__main__':
    sys.exit(main())

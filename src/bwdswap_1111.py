#!/usr/bin/env python3
"""bwdswap_1111 -- E.1's derivative swap with the forward left natural (arm B), next to E.1's N and X.

E.1 (V13 appendix E.1; specs/spec_resp_cifar_ee_0920.md 2.3, spec_resp_ee_0917.md 2.3,
spec_resp_cifar5p1_1007.md 2.3) replaced the second hidden layer's activity by

    X:  a = phi(z0) + [phi(z + d) - phi(z0 + d)],    z0 = frozen branch net, d = z_donor - z_branch

so the net is the natural one at the branch and the local derivative is phi'(z + d); after the branch the
forward value departs from phi(z) as well.  Arm B keeps the forward natural and swaps only the derivative:

    B:  a = phi(z) + [phi(z + d) - sg(phi(z + d))]   value phi(z) bit for bit, d a / d z = phi'(z + d)

i.e. X with its frozen reference re-anchored to the current weights at every step: the same d, the same
AnchorAdd (src/cifar_interventions_0920.py: the value is the anchor exactly whenever the bracket is exactly 0,
signed zeros kept, and the bracket's derivative is one; a NaN bracket propagates, so the 5+1 field, NaN
outside the task's images, stays fail closed).  Everything else -- branch states, d, Adam rule, labels,
batches, preflight, diagnostics, rows -- is the X arm's code in the E.1 runner, imported unchanged:

    RC   src/resp_cifar_ee_0920.py + src/cifar_interventions_0920.py   CUDA, seeds stacked (R=10), CUDA graph
    RM   src/resp_ee_0917.py                                            CPU, one process per seed, 1 thread
    F51  src/resp_cifar5p1_1007.py                                      CUDA, seeds stacked (R=10), CUDA graph

N and X are E.1's own arms run through E.1's own functions.  B swaps one constructor of the X arm:
RC  Shift -> BwdShift (built like RE.arm_engine builds X);
F51 Field -> BwdField, swapped in for the call of F51.run_arm(<X arm>);
RM  make_forward -> the backward-only forward, swapped in for the call of RM.run_arm(<B name>, <X arm>).
B's forward is compared bit for bit with the host's natural forward on the same weights and batch at every
call, CUDA-graph replays included (Monitor).  Before each direction's arms a branch probe checks, on the
task's first minibatches and all its images, that N, X and B give the same logits and loss, that X and B give
the same gradients, and that B's local derivative is the native phi' at z + d.

    python -m src.bwdswap_1111 --setting RC  --out results/bwdswap_1111/RC
    python -m src.bwdswap_1111 --setting F51 --out results/bwdswap_1111/F51
    python -m src.bwdswap_1111 --setting RM  --seed 3 --out results/bwdswap_1111/RM/s3
(the main run is launched by analysis/bwdswap_1111/launch.sh after the predictions are registered)
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import fcntl
import gc
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from src import cifar_interventions_0920 as CI      # RC engine, Shift, AnchorAdd, gtrain, tree_hash
from src import resp_cifar_ee_0920 as RE            # RC arms, field, preflight, diagnostics
from src import resp_ee_0917 as RM                  # RM prefix, arms, forward
from src import resp_cifar5p1_1007 as F51           # 5+1 engine, field, arms

RUN = 'bwdswap_1111'
OUT_MAIN = ROOT / 'results' / RUN
CHECKS = OUT_MAIN / 'checks.json'
BACKUP = Path('/home/issan/Projects/obsidian-research-data')
MNIST_DIR = Path('/home/issan/Projects/claude/proj_004_drift/data/mnist')
GPU_LOCK = Path('/tmp/lop_analysis_gpu.lock')
LAYER = 2
MAIN_SEEDS = list(range(10))
BUDGET = {'RC': 400, 'RM': 80, 'F51': 780}      # epochs per task (RC, RM) / updates per task (F51)

# direction -> E.1 arms (N, X), the new arm B, branch / donor task (end of), next task, Adam reset
SETTINGS = {
    'RC': {'restore': dict(N='N10', X='R1_10', B='R1_10_bwd', branch=10, donor=1, task=11, reset=False),
           'reverse': dict(N='N1r', X='S1_10r', B='S1_10r_bwd', branch=1, donor=10, task=2, reset=True)},
    'RM': {'restore': dict(N='N10', X='R2_10', B='R2_10_bwd', branch=10, donor=2, task=11, reset=False),
           'reverse': dict(N='N2r', X='S2_10r', B='S2_10r_bwd', branch=2, donor=10, task=3, reset=True)},
    'F51': {'restore': dict(N='N_c', X='R_ch', B='R_ch_bwd', branch=28, donor=2, task=29, reset=False),
            'reverse': dict(N='N_hr', X='S_hcr', B='S_hcr_bwd', branch=2, donor=28, task=3, reset=True)},
}
KINDS = ('N', 'X', 'B')
DIRECTIONS = ('restore', 'reverse')
E1 = {'RC': dict(run='resp_cifar_ee_0920', states=(1, 10, 11)),
      'RM': dict(run='resp_ee_0917'),
      'F51': dict(run='resp_cifar5p1_1007', states=(2, 28, 29))}


def _check_settings():
    """The arms above are E.1's arms (branch, layer, donor, Adam reset), read from the E.1 runners."""
    for s in SETTINGS['RC'].values():
        assert RE.ARMS[s['N']] == (s['branch'], None, None, s['reset'])
        assert RE.ARMS[s['X']] == (s['branch'], LAYER, s['donor'], s['reset'])
    for s in SETTINGS['F51'].values():
        assert F51.ARMS[s['N']] == (s['branch'], None, None, s['reset'])
        assert F51.ARMS[s['X']] == (s['branch'], LAYER, s['donor'], s['reset'])
    for s in SETTINGS['RM'].values():
        assert RM.ARMS[s['N']] == dict(branch=s['branch'], layer=None, kind=None, src=None, delta=None, reset=s['reset'])
        assert RM.ARMS[s['X']] == dict(branch=s['branch'], layer=LAYER, kind='field', src=s['donor'], delta=None,
                                       reset=s['reset'])


_check_settings()


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def put(path, obj):
    CI.put(Path(path), obj)          # analysis/cifar_ledger_0920/replay.put: atomic, NaN -> null


def bits_equal(a, b):
    """Bit-for-bit equality of two tensors (signed zeros and NaN payloads included)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return CI.tree_hash(a) == CI.tree_hash(b)


def bit_mismatch(a, b):
    """Number of elements whose bits differ (a, b same shape and dtype); a device tensor."""
    view = {torch.float32: torch.int32, torch.float64: torch.int64}[a.dtype]
    return (a.detach().view(view) != b.detach().view(view)).sum()


def gen_clone(state):
    g = torch.Generator(device='cpu')
    g.set_state(state.clone())
    return g


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def mem_available_gb():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 2 ** 20
    return float('nan')


def gpu_free_gb():
    p = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, check=True)
    return float(p.stdout.split()[0]) / 1024


def gpu_process_mib(pid=None):
    """This process's GPU memory as nvidia-smi sees it (context included), MiB; None if not listed."""
    pid = os.getpid() if pid is None else pid
    p = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True)
    for line in p.stdout.splitlines():
        parts = [q.strip() for q in line.split(',')]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
            return float(parts[1])
    return None


@contextlib.contextmanager
def gpu_guard(log=print, need_gpu_gb=4.0, need_ram_gb=8.0, wait_s=6 * 3600, poll=15):
    """The lab's shared GPU lock (blocking flock, like the sibling jobs), then wait for room: one of our GPU
    processes at a time.  Other sessions' light CUDA contexts are not waited for (they do not change our
    arithmetic); RAM keeps need_ram_gb free for the desktop and the CPU jobs."""
    with GPU_LOCK.open('a') as lock:
        t0 = time.time()
        fcntl.flock(lock, fcntl.LOCK_EX)
        log(f'GPU lock acquired after {time.time() - t0:.0f}s')
        n = 0
        while True:
            free, gfree = mem_available_gb(), gpu_free_gb()
            if free >= need_ram_gb and gfree >= need_gpu_gb:
                break
            assert time.time() - t0 < wait_s, ('GPU/RAM wait exceeded', free, gfree)
            if n % 8 == 0:
                log(f'waiting for room: MemAvailable {free:.1f} GB, GPU free {gfree:.1f} GB')
            n += 1
            time.sleep(poll)
        yield


@contextlib.contextmanager
def swapped(module, name, value):
    """Replace one module attribute for the duration of a call (restored on any exit)."""
    old = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, old)


def peak_rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def now_jst():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()


# --------------------------------------------------------------------------
# the backward-only activation and the forward monitor
# --------------------------------------------------------------------------

def bwd_only(phi, z, d):
    """phi(z) in value (the anchor, bit for bit), derivative phi'(z + d): the native autograd of phi at the
    shifted argument flows through the bracket phi(z + d) - sg(phi(z + d)), which is exactly 0."""
    with torch.no_grad():
        value = phi(z)
    shifted = phi(z + d)
    return CI.AnchorAdd.apply(value, shifted - shifted.detach())


class Monitor:
    """Wraps a forward (P, xb, ids) -> (z1, a1, z2, a2, logits).  On every call it also runs the host's natural
    forward on the same weights and batch (no grad) and counts the elements whose bits differ.  Calls recorded
    into a CUDA graph count on every replay ('graph'); calls on the default stream count as 'eager'; the graph
    capture's warmup steps (side stream, not capturing) are not compared.  The wrapped object's attributes
    (layer, d, P0, state) are delegated, so the E.1 engines, preflights and diagnostics use it unchanged."""

    KEYS = ('calls', 'mismatch', 'compared')

    def __init__(self, inner, natural, device):
        self.inner = inner
        self.natural = natural
        self.device = torch.device(device)
        self.c = {f'{m}_{k}': torch.zeros((), dtype=torch.long, device=self.device)
                  for m in ('graph', 'eager') for k in self.KEYS}

    def __getattr__(self, name):
        inner = self.__dict__.get('inner')
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    def _mode(self):
        if self.device.type != 'cuda':
            return 'eager'
        if torch.cuda.is_current_stream_capturing():
            return 'graph'
        if torch.cuda.current_stream(self.device) == torch.cuda.default_stream(self.device):
            return 'eager'
        return None

    def __call__(self, P, xb, ids):
        out = self.inner(P, xb, ids)
        mode = self._mode()
        if mode is not None:
            with torch.no_grad():
                nat = self.natural(P, xb)
                assert len(nat) == len(out) == 5
                for a, b in zip(out, nat):
                    self.c[f'{mode}_mismatch'].add_(bit_mismatch(a, b))
                    self.c[f'{mode}_compared'].add_(a.numel())
                self.c[f'{mode}_calls'].add_(1)
        return out

    def read(self):
        return {k: int(v) for k, v in self.c.items()}

    def reset(self):
        with torch.no_grad():
            for v in self.c.values():
                v.zero_()


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

def _loaded_sources():
    files = set()
    for mod in list(sys.modules.values()):
        f = getattr(mod, '__file__', None)
        if f and str(f).endswith('.py'):
            p = Path(f).resolve()
            if p.is_relative_to(ROOT) and p.is_file():
                files.add(p)
    return files


_SOURCES = _loaded_sources()          # this module and every repo module it pulls in, fixed at import


def source_hashes():
    files = set(_SOURCES)
    files |= set((ROOT / 'analysis' / RUN).glob('*.py')) | set((ROOT / 'analysis' / RUN).glob('*.sh'))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def environment(setting):
    env = dict(torch=torch.__version__, numpy=np.__version__, python=sys.version.split()[0],
               platform=platform.platform(), cpu_capability=torch.backends.cpu.get_cpu_capability(),
               threads=torch.get_num_threads(), flush_denormal=RM._flush_is_on(),
               deterministic=torch.are_deterministic_algorithms_enabled())
    if setting != 'RM':
        env.update(CI.environment())
        env['boot_id'] = boot_id()          # GPU arithmetic may change across boots: never mix boots
    return env


def identity(setting, seeds, budget, check_mode, data_hash):
    return CI.cpu(dict(run=RUN, setting=setting, seeds=list(seeds), budget=budget, check_mode=bool(check_mode),
                       settings=SETTINGS[setting], git_hash=git('rev-parse', 'HEAD'), source_sha256=source_hashes(),
                       environment=environment(setting), data=data_hash))


def jnorm(x):
    return json.loads(json.dumps(F51.clean(x)))


def bind_identity(out, ident):
    """A run directory belongs to one identity (code, data, seeds, budget, and for the GPU the boot)."""
    path = Path(out) / 'identity.json'
    if path.exists():
        saved = json.loads(path.read_text())
        assert jnorm(saved) == jnorm(ident), f'identity mismatch in {out}: start a new output directory'
    else:
        put(path, ident)


def admission(setting, seeds, budget, prereg):
    """Main run only: verified code (checks all_pass for these sources), committed code, a prediction
    registration commit that HEAD descends from, registered seeds and budget."""
    assert list(seeds) == MAIN_SEEDS, 'registered seeds are 0-9'
    assert budget == BUDGET[setting], ('registered budget', setting, BUDGET[setting])
    chk = json.loads(CHECKS.read_text())
    assert chk.get('all_pass') is True, 'checks.json is not all_pass'
    assert chk.get('source_sha256') == source_hashes(), 'checks.json was made from other sources: rerun the checks'
    dirty = git('status', '--porcelain', '--', 'src/bwdswap_1111.py', f'analysis/{RUN}')
    assert not dirty, 'commit the implementation before the main run'
    assert prereg, 'the prediction registration commit is required (--prereg or BWDSWAP_PREREG)'
    ok = subprocess.run(['git', 'merge-base', '--is-ancestor', prereg, 'HEAD'], cwd=ROOT).returncode == 0
    assert ok, f'registration commit {prereg} is not an ancestor of HEAD'
    return dict(checks_sha256=sha(CHECKS), prereg_commit=git('rev-parse', prereg))


def mark_done(path, ident, files, base):
    """Completion marker: identity and the sha256 of every output file (paths relative to the run dir)."""
    put(path, dict(identity=ident, files={str(Path(p).relative_to(base)): sha(p) for p in files}, complete=True))


def verify_done(path, ident, base):
    obj = json.loads(Path(path).read_text())
    assert jnorm(obj['identity']) == jnorm(ident), ('identity mismatch', path)
    assert obj['complete'] and obj['files'], ('incomplete marker', path)
    for name, h in obj['files'].items():
        p = Path(base) / name
        assert p.exists() and sha(p) == h, ('corrupt output', p)
    return obj


def files_under(d, exclude=('done.json',)):
    return sorted(p for p in Path(d).rglob('*') if p.is_file() and p.name not in exclude and not p.name.endswith('.tmp'))


def e1_backup_state(run, rel):
    """An E.1 state file outside git, read through the committed backup manifest (bytes and sha256 checked)."""
    manifest = json.loads((ROOT / 'results' / run / 'backup_manifest.json').read_text())
    want = f'results/{run}/{rel}'
    hits = [f for f in manifest['files'] if f.get('relative', f.get('source_rel')) == want]
    assert len(hits) == 1, ('E.1 backup entry', want)
    p = Path(hits[0]['backup'])
    assert p.is_file() and p.stat().st_size == hits[0]['bytes'] and sha(p) == hits[0]['sha256'], ('E.1 backup', p)
    return torch.load(p, map_location='cpu', weights_only=False)


# --------------------------------------------------------------------------
# local derivative: autograd of the arm's own forward vs the native phi' at the shifted argument
# --------------------------------------------------------------------------

def local_derivative(vals):
    """d a2 / d z2 elementwise from the arm's own forward outputs (a2 depends on z2 elementwise only)."""
    z2, a2 = vals[2], vals[3]
    return torch.autograd.grad(a2.sum(), z2, retain_graph=True)[0].detach()


def derivative_report(g_arm, g_native_shift, g_native_plain, g_ref64, bound64):
    """g_arm must equal the native phi'(z + d) bit for bit and the float64 phi'(z + d) within the derived
    bound; the share of pairs where phi'(z + d) != phi'(z) shows the swap does something on this batch."""
    err = (g_arm.double() - g_ref64).abs()
    pos = bound64 > 0
    ratio = torch.where(pos, err / torch.where(pos, bound64, torch.ones_like(bound64)),
                        torch.where(err > 0, torch.full_like(err, math.inf), torch.zeros_like(err)))
    return dict(bit_equal_native_shifted=bits_equal(g_arm, g_native_shift),
                within_float64_bound=bool((err <= bound64).all()),
                max_error_over_bound=float(ratio.max()),
                pairs=int(g_arm.numel()),
                changed_vs_natural=int((g_native_shift != g_native_plain).sum()),
                natural_equals_shifted=bits_equal(g_native_plain, g_native_shift))


def elu_cuda_bound(ref):
    """F.elu backward on CUDA is exp(z) for z <= 0 (src/rlcifar_mlp_battle_0918.py ELU); CUDA expf has a
    maximum error of 2 ulp, ulp(r) <= 2^-23 r for normal r, and results below 2^-126 may be flushed."""
    return 2.0 ** -22 * ref + 2.0 ** -126


def elu_expm1_bound(ref):
    """RM's ELU derivative is fl(fl(expm1(z)) + 1) for z <= 0 (spec_resp_ee_0917 1.3): expm1 within 1 ulp
    (<= 2^-24 on (-1, 0]) and the +1 rounding (<= 2^-25)."""
    return torch.full_like(ref, 2.0 ** -24 + 2.0 ** -25)


# ==========================================================================
# RC: random-label CIFAR MLP, ELU/std, CUDA (src/resp_cifar_ee_0920.py)
# ==========================================================================

_RC_ACT = None


def rc_natural(P, xb):
    global _RC_ACT
    if _RC_ACT is None:
        _RC_ACT = CI.E.ELU()
    return CI.E.forward(P, xb, _RC_ACT, train=True)       # the host forward the natural engine calls


class BwdShift(CI.Shift):
    """X's Shift (same constructor: frozen P0, layer, d), forward = the natural net, derivative at z + d.
    P0 is kept only so that E.1's preflight can check provenance (P0 == branch, not aliased)."""

    def state(self):
        return dict(super().state(), kind='bwd_only')

    def __call__(self, P, xb, ids):
        W1, b1, W2, b2, W3, b3 = P
        ar = torch.arange(xb.shape[0], device=xb.device)[:, None]
        d = self.d[ar, ids]
        z1 = torch.baddbmm(b1[:, None, :], xb, W1.transpose(1, 2))
        a1 = bwd_only(F.elu, z1, d) if self.layer == 1 else F.elu(z1)
        z2 = torch.baddbmm(b2[:, None, :], a1, W2.transpose(1, 2))
        a2 = bwd_only(F.elu, z2, d) if self.layer == 2 else F.elu(z2)
        return z1, a1, z2, a2, torch.baddbmm(b3[:, None, :], a2, W3.transpose(1, 2))


def rc_b_engine(spec, states, X, graph=True, monitor=True):
    """B built as RE.arm_engine builds X (same branch state, same field, same Adam rule), BwdShift for Shift."""
    branch, layer, source, reset = RE.ARMS[spec['X']]
    st = states[branch]
    d = RE.field_from_states(st, states[source], layer, X)
    shift = BwdShift([p.to(X.device) for p in st['P']], layer, d)
    if monitor:
        shift = Monitor(shift, rc_natural, X.device)
    eng = CI.Engine(st['seeds'], X, state=st, shift=shift, graph=graph)
    if reset:
        eng.reset_adam()
    return eng


def rc_engine(kind, spec, states, X, graph=True, monitor=True):
    if kind == 'B':
        return rc_b_engine(spec, states, X, graph=graph, monitor=monitor)
    return RE.arm_engine(spec[kind], states, X, graph=graph)


def rc_task_draws(st, X):
    """The next task's labels and first-epoch order, from clones of the branch generators -- what
    Engine.train_task will draw (it draws the labels first, then one permutation per epoch)."""
    seeds = st['seeds']
    labels = torch.stack([CI.E.RC.task_labels(gen_clone(st['g_lab'][s])) for s in seeds]).to(X.device)
    gb = [gen_clone(st['g_batch'][s]) for s in seeds]
    order = torch.stack([torch.randperm(CI.E.N_IMAGES, generator=g) for g in gb]).to(X.device)
    return labels, order


def rc_branch_probe(spec, states, X, n_batches=2):
    """Update 0 of one direction: N, X, B on the branch state (eager engines).  Same logits and per-seed CE
    on all 1200 images and the first training minibatches; X and B gradients bit-equal; B's local derivative
    = native F.elu' at z + d; N's local derivative is not (the check has teeth)."""
    st = states[spec['branch']]
    engs = {k: rc_engine(k, spec, states, X, graph=False, monitor=False) for k in KINDS}
    labels, order = rc_task_draws(st, X)
    R, dev = len(st['seeds']), X.device
    ar = torch.arange(R, device=dev)[:, None]
    batches = [('all', torch.arange(CI.E.N_IMAGES, device=dev)[None, :].expand(R, -1))]
    batches += [(f'mb{j}', order[:, j * CI.E.BATCH:(j + 1) * CI.E.BATCH]) for j in range(n_batches)]
    d_all = engs['B'].shift.d
    out = dict(direction_arms={k: spec[k] for k in KINDS}, batches={})
    for tag, ids in batches:
        xb, yb = X[ar, ids], labels[ar, ids]
        res = {}
        for k, eng in engs.items():
            vals = eng.forward(xb, ids)
            lossv = F.cross_entropy(vals[-1].reshape(-1, CI.E.N_CLASSES), yb.reshape(-1),
                                    reduction='none').view(R, -1).mean(1)
            g_loc = local_derivative(vals)
            grads = torch.autograd.grad(lossv.sum(), eng.P)
            res[k] = dict(vals=[v.detach() for v in vals], loss=lossv.detach(), g=g_loc, grads=grads)
        z2 = res['B']['vals'][2]
        arg = z2 + d_all[ar, ids]
        gs, gp = CI.gtrain(arg), CI.gtrain(z2)
        ref = arg.double().clamp(max=0).exp()
        rep = dict(
            outputs_bit_equal={k: all(bits_equal(a, b) for a, b in zip(res[k]['vals'], res['N']['vals'])) for k in 'XB'},
            loss_bit_equal={k: bits_equal(res[k]['loss'], res['N']['loss']) for k in 'XB'},
            grads_X_B_bit_equal=all(bits_equal(a, b) for a, b in zip(res['X']['grads'], res['B']['grads'])),
            grads_N_B_differing_elements=int(sum(int((a != b).sum()) for a, b in zip(res['N']['grads'], res['B']['grads']))),
            derivative_B=derivative_report(res['B']['g'], gs, gp, ref, elu_cuda_bound(ref)),
            derivative_X_bit_equal_B=bits_equal(res['X']['g'], res['B']['g']),
            derivative_N_bit_equal_native_shifted=bits_equal(res['N']['g'], gs))
        out['batches'][tag] = rep
    out['all_pass'] = probe_pass(out)
    del engs
    gc.collect()
    torch.cuda.empty_cache()
    return out


def probe_pass(out):
    ok = True
    for rep in out['batches'].values():
        dv = rep['derivative_B']
        ok &= all(rep['outputs_bit_equal'].values()) and all(rep['loss_bit_equal'].values())
        ok &= rep['grads_X_B_bit_equal'] and rep['derivative_X_bit_equal_B']
        ok &= dv['bit_equal_native_shifted'] and dv['within_float64_bound']
        # teeth: where the swap changes phi' on the batch, N's derivative must not pass as phi'(z + d)
        if dv['changed_vs_natural'] > 0:
            ok &= not rep['derivative_N_bit_equal_native_shifted']
    return bool(ok)


def rc_prefix(seeds, X, epochs, log):
    """E.1's natural prefix (RE.run): one stacked engine, tasks 1..11, the branch / donor / continuation
    states at the ends of tasks 1, 10, 11 (E.1 also saved task 2, used by N1 only)."""
    eng = CI.Engine(seeds, X)
    rows, states, t0 = [], {}, time.time()
    for task in range(1, 12):
        r, _ = eng.train_task(task, epochs)
        assert all(x['finite'] for x in r), ('DIVERGED prefix', task)
        rows.extend(r)
        if task in E1['RC']['states']:
            states[task] = eng.state()
        log(f'RC prefix task {task}/11  {time.time() - t0:.1f}s')
    del eng
    gc.collect()
    torch.cuda.empty_cache()
    return states, rows, time.time() - t0


def rc_arm(kind, spec, states, X, epochs, monitor=True):
    """One arm exactly as RE.run runs it (arm_engine/preflight/diagnostic/train_task with E.1's diagnostic
    schedule); B differs only in the engine's shift."""
    name = spec[kind]
    eng = rc_engine(kind, spec, states, X, monitor=monitor)
    t0 = time.time()
    pf = RE.preflight(spec['X'] if kind == 'B' else name, eng, states)
    ir, iu = eng.diagnostic()
    mon_pre = None
    if kind == 'B' and monitor:
        mon_pre = eng.shift.read()
        eng.shift.reset()
    t1 = time.time()
    rows, units = eng.train_task(spec['task'], epochs, RE.DIAG_STEPS)
    t2 = time.time()
    mon = eng.shift.read() if kind == 'B' and monitor else None
    for row in rows:
        row.update(arm=name, kind=kind, e1_arm=spec['X'] if kind == 'B' else name)
    res = dict(rows=rows, units=units, preflight=pf, initial=ir, initial_units=iu, history=eng.diag_history,
               end=eng.state(), monitor_pre=mon_pre, monitor=mon, updates=epochs * CI.E.STEPS_PER_EPOCH,
               seconds_setup=t1 - t0, seconds_train=t2 - t1)
    del eng
    gc.collect()
    torch.cuda.empty_cache()
    return res


def monitor_pass(mon, updates):
    """Every training update replayed B's forward next to the natural one, with no differing bit."""
    if mon is None:
        return None
    return bool(mon['graph_calls'] == updates and mon['graph_mismatch'] == 0 and mon['eager_mismatch'] == 0
                and mon['graph_compared'] > 0)


def run_rc(out, seeds, epochs, check_mode=False, log=print, monitor=True, prereg=None):
    adm = None if check_mode else admission('RC', seeds, epochs, prereg)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dev = CI.setup()
    X, cifar = CI.make_inputs(seeds, dev)
    ident = identity('RC', seeds, epochs, check_mode, dict(data_sha256=cifar.sha256, input_hash=CI.tree_hash(X)))
    bind_identity(out, ident)
    start = time.time()
    put(out / 'status.json', dict(stage='started', pid=os.getpid(), started_jst=now_jst(), admission=adm))
    raw = out / 'raw'
    pdir = out / 'prefix'
    if (pdir / 'done.json').exists():
        verify_done(pdir / 'done.json', ident, out)
        states = {t: CI.load_pt(raw / f'prefix_t{t:02d}.pt') for t in E1['RC']['states']}
        prefix_rows = json.loads((pdir / 'rows.json').read_text())
        log('RC prefix: reused (same identity)')
    else:
        states, prefix_rows, sec = rc_prefix(seeds, X, epochs, log)
        for t, st in states.items():
            CI.save_pt(raw / f'prefix_t{t:02d}.pt', st)
        put(pdir / 'rows.json', prefix_rows)
        put(pdir / 'timing.json', dict(seconds=sec, updates=11 * epochs * CI.E.STEPS_PER_EPOCH))
        mark_done(pdir / 'done.json', ident, files_under(pdir) + [raw / f'prefix_t{t:02d}.pt' for t in states], out)
    checks = {}
    for direction in DIRECTIONS:
        spec = SETTINGS['RC'][direction]
        probe_path = out / 'probes' / f'{direction}.json'
        if not probe_path.exists():
            put(probe_path, rc_branch_probe(spec, states, X))
        probe = json.loads(probe_path.read_text())
        assert probe['all_pass'], ('CHECK_FAILED branch probe', direction)
        checks[f'{direction}/branch_probe'] = True
        for kind in KINDS:
            name = spec[kind]
            adir = out / 'arms' / name
            if (adir / 'done.json').exists():
                verify_done(adir / 'done.json', ident, out)
                checks.update(json.loads((adir / 'arm_checks.json').read_text()))
                log(f'RC arm {name}: reused')
                continue
            res = rc_arm(kind, spec, states, X, epochs, monitor=monitor)
            ac = {f'{name}/preflight': bool(res['preflight']['all_pass']),
                  f'{name}/finite': all(r['finite'] for r in res['rows'])}
            if kind == 'B' and monitor:
                ac[f'{name}/forward_monitor'] = monitor_pass(res['monitor'], res['updates'])
            if kind == 'N' and direction == 'restore':           # E.1's natural-continuation identity
                exp_state = states[spec['task']]
                lookup = {r['seed']: r for r in prefix_rows if r['task'] == spec['task']}
                ac[f'{name}/natural_continuation'] = bool(
                    CI.tree_hash(CI.core_state(res['end'])) == CI.tree_hash(CI.core_state(exp_state))
                    and all(r[k] == lookup[r['seed']][k] for r in res['rows']
                            for k in ('online_acc', 'memo_acc', 'label_hash', 'order_hash')))
            put(adir / 'rows.json', res['rows'])
            RE.write_csv(adir / 'per_task.csv', res['rows'])
            put(adir / 'preflight.json', res['preflight'])
            put(adir / 'initial.json', res['initial'])
            RE.serialize_diagnostics(adir, res['history'])
            CI.save_npz(adir / 'units.npz', res['units'])
            put(adir / 'timing.json', dict(seconds_setup=res['seconds_setup'], seconds_train=res['seconds_train'],
                                           updates=res['updates'],
                                           ms_per_update=1e3 * res['seconds_train'] / res['updates']))
            if res['monitor'] is not None:
                put(adir / 'monitor.json', dict(pre_training=res['monitor_pre'], training=res['monitor']))
            put(adir / 'arm_checks.json', ac)
            CI.save_pt(raw / 'arms' / name / 'end.pt', res['end'])
            checks.update(ac)
            assert all(v is not False for v in ac.values()), ('CHECK_FAILED', name, ac)
            mark_done(adir / 'done.json', ident, files_under(adir) + [raw / 'arms' / name / 'end.pt'], out)
            log(f'RC arm {name}: train {res["seconds_train"]:.1f}s  setup {res["seconds_setup"]:.1f}s')
            del res
    e1 = rc_e1_comparison(out, states, prefix_rows) if (list(seeds) == MAIN_SEEDS and epochs == BUDGET['RC']) else None
    if e1 is not None:
        put(out / 'e1_comparison.json', e1)
    finish(out, checks, start, 'RC')
    return checks


def rc_e1_comparison(out, states, prefix_rows):
    """Recorded, not gating (GPU arithmetic may differ across boots): this boot's prefix states and rows and
    the N / X online accuracies against E.1 (results/resp_cifar_ee_0920, its states via the backup)."""
    run = E1['RC']['run']
    res = dict(states={}, prefix_rows={}, arms={})
    for t, st in states.items():
        old = e1_backup_state(run, f'prefix/t{t:02d}.pt')
        res['states'][t] = CI.tree_hash(CI.core_state(st)) == CI.tree_hash(CI.core_state(old))
    old_rows = {(r['seed'], r['task']): r for r in json.loads((ROOT / 'results' / run / 'prefix/rows.json').read_text())}
    diff = [(r['seed'], r['task']) for r in prefix_rows if r['online_acc'] != old_rows[(r['seed'], r['task'])]['online_acc']]
    res['prefix_rows'] = dict(online_acc_equal=len(diff) == 0, differing=diff,
                              first_differing_task=min((t for _, t in diff), default=None))
    for direction in DIRECTIONS:
        spec = SETTINGS['RC'][direction]
        for kind in 'NX':
            res['arms'][spec[kind]] = compare_online(json.loads((out / 'arms' / spec[kind] / 'rows.json').read_text()),
                                                     json.loads((ROOT / 'results' / run / 'arms' / spec[kind] / 'rows.json').read_text()))
    return res


def compare_online(new_rows, old_rows, key='online_acc'):
    old = {r['seed']: r[key] for r in old_rows}
    per = [dict(seed=r['seed'], new=r[key], e1=old[r['seed']], equal=r[key] == old[r['seed']],
                difference=r[key] - old[r['seed']]) for r in new_rows]
    return dict(all_equal=all(p['equal'] for p in per), mean_new=float(np.mean([p['new'] for p in per])),
                mean_e1=float(np.mean([p['e1'] for p in per])), per_seed=per)


def finish(out, checks, start, setting):
    put(out / 'run_checks.json', dict(all_pass=all(v is not False for v in checks.values()), checks=checks))
    prov = dict(finished_jst=now_jst(), elapsed_s=time.time() - start, peak_rss_bytes=peak_rss_bytes())
    if setting != 'RM':
        prov.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                    nvidia_smi_process_mib=gpu_process_mib())
    put(out / 'provenance_end.json', prov)
    put(out / 'status.json', dict(stage='completed', elapsed_s=time.time() - start))


# ==========================================================================
# F51: real-label 5+1 CIFAR-100 MLP, ReLU/std, CUDA (src/resp_cifar5p1_1007.py)
# ==========================================================================

_F51_ACT = None


def f51_natural(P, xb):
    global _F51_ACT
    if _F51_ACT is None:
        _F51_ACT = F51.B.ReLU()
    return F51.B.forward(P, xb, _F51_ACT, train=True)      # Engine.forward of a natural arm


class BwdField(F51.Field):
    """X's Field (same constructor: frozen P0, layer, the NaN-outside-the-task table), forward = natural net,
    derivative at z + d.  P0 is kept only for E.1's preflight provenance checks."""

    def __call__(self, P, xb, ids):
        W1, b1, W2, b2, W3, b3 = P
        ar = torch.arange(xb.shape[0], device=xb.device)[:, None]
        d = self.d[ar, ids]
        z1 = torch.baddbmm(b1[:, None, :], xb, W1.transpose(1, 2))
        a1 = bwd_only(F51.phi, z1, d) if self.layer == 1 else F51.phi(z1)
        z2 = torch.baddbmm(b2[:, None, :], a1, W2.transpose(1, 2))
        a2 = bwd_only(F51.phi, z2, d) if self.layer == 2 else F51.phi(z2)
        return z1, a1, z2, a2, torch.baddbmm(b3[:, None, :], a2, W3.transpose(1, 2))


def f51_field_factory(monitors=None):
    """Drop-in for F51.Field inside F51.arm_engine: builds BwdField (and wraps it in a Monitor)."""
    def make(P0, layer, table):
        f = BwdField(P0, layer, table)
        if monitors is not None:
            f = Monitor(f, f51_natural, table.device)
            monitors.append(f)
        return f
    return make


def f51_engine(kind, spec, states, data, graph=True, monitors=None):
    if kind == 'B':
        with swapped(F51, 'Field', f51_field_factory(monitors)):
            return F51.arm_engine(spec['X'], states, data, graph=graph)
    return F51.arm_engine(spec[kind], states, data, graph=graph)


def f51_branch_probe(spec, states, data, n_batches=2):
    """As rc_branch_probe, on the 5+1 box: all images of the next task in one batch and its first training
    minibatches (the batches the arm will train on); ReLU' at z + d is 1[z + d >= 0] exactly."""
    engs = {k: f51_engine(k, spec, states, data, graph=False) for k in KINDS}
    T = spec['branch'] + 1
    rows_T, batches = engs['N'].task_batches(T)          # consumes the throwaway N engine's generators only
    dev = data.device
    R = len(rows_T)
    ar = torch.arange(R, device=dev)[:, None]
    batch_list = [('all', torch.stack(rows_T).to(dev))] + [(f'mb{j}', batches[:, j]) for j in range(n_batches)]
    out = dict(direction_arms={k: spec[k] for k in KINDS}, batches={})
    for tag, ids in batch_list:
        xb, yb = data.X[ids], data.Y[ids]
        res = {}
        for k, eng in engs.items():
            vals = eng.forward(xb, ids)
            lossv = F.cross_entropy(vals[-1].reshape(-1, F51.NCLS), yb.reshape(-1),
                                    reduction='none').view(R, -1).mean(1)
            g_loc = local_derivative(vals)
            grads = torch.autograd.grad(lossv.sum(), eng.P)
            res[k] = dict(vals=[v.detach() for v in vals], loss=lossv.detach(), g=g_loc, grads=grads)
        z2 = res['B']['vals'][2]
        arg = z2 + engs['B'].field.d[ar, ids]
        gs, gp = F51.gtrain(arg), F51.gtrain(z2)
        ref = (arg.double() >= 0).double()
        rep = dict(
            outputs_bit_equal={k: all(bits_equal(a, b) for a, b in zip(res[k]['vals'], res['N']['vals'])) for k in 'XB'},
            loss_bit_equal={k: bits_equal(res[k]['loss'], res['N']['loss']) for k in 'XB'},
            grads_X_B_bit_equal=all(bits_equal(a, b) for a, b in zip(res['X']['grads'], res['B']['grads'])),
            grads_N_B_differing_elements=int(sum(int((a != b).sum()) for a, b in zip(res['N']['grads'], res['B']['grads']))),
            # ReLU' at z + d is 1[z + d >= 0] exactly, so the float64 bound is 0
            derivative_B=derivative_report(res['B']['g'], gs, gp, ref, torch.zeros_like(ref)),
            derivative_X_bit_equal_B=bits_equal(res['X']['g'], res['B']['g']),
            derivative_N_bit_equal_native_shifted=bits_equal(res['N']['g'], gs))
        out['batches'][tag] = rep
    out['all_pass'] = probe_pass(out)
    del engs
    gc.collect()
    torch.cuda.empty_cache()
    return out


def f51_prefix(seeds, data, log):
    """E.1's natural prefix (F51.run): tasks 1..29 with the host's rows; branch / donor states at the ends of
    tasks 2 and 28, and task 29 with its batches for the natural-continuation check (fresh control skipped:
    it starts after the saved states and touches nothing the arms use)."""
    eng = F51.Engine(seeds, data)
    states, t0 = {}, time.time()

    def save(t, st, batches, rows_now):
        if t in E1['F51']['states']:
            states[t] = dict(**st, last_batches=batches)

    def on_task(t):
        if t % 5 == 0 or t == 29:
            log(f'F51 prefix task {t}/29  {time.time() - t0:.1f}s')
    rows, _ = F51.natural_prefix(eng, 29, save=save, on_task=on_task)
    del eng
    gc.collect()
    torch.cuda.empty_cache()
    return states, rows, time.time() - t0


def f51_arm(kind, spec, states, data, monitor=True):
    """One arm exactly as F51.run runs it (F51.run_arm: arm_engine, preflight with the training-gate check,
    diagnostics at 0/78/390/780, 780 updates); for B, F51.Field is swapped for BwdField during the call."""
    t0 = time.time()
    if kind == 'B':
        mons = [] if monitor else None
        with swapped(F51, 'Field', f51_field_factory(mons)):
            res = F51.run_arm(spec['X'], states, data)
        mon = mons[-1].read() if monitor else None
        if monitor:
            assert len(mons) == 1
            mons.clear()
        for row in res['rows']:
            row.update(arm=spec['B'], display=F51.DISPLAY[spec['X']] + ' bwd')
        res['preflight']['arm'] = spec['B']
    else:
        res = F51.run_arm(spec[kind], states, data)
        mon = None
    for row in res['rows']:
        row.update(kind=kind, e1_arm=spec['X'] if kind == 'B' else spec[kind])
    res.update(monitor=mon, seconds=time.time() - t0, updates=F51.STEPS)
    gc.collect()
    torch.cuda.empty_cache()
    return res


def run_f51(out, seeds, check_mode=False, log=print, monitor=True, prereg=None, data=None):
    adm = None if check_mode else admission('F51', seeds, F51.STEPS, prereg)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dev = F51.setup()
    data = F51.Data(dev) if data is None else data
    ident = identity('F51', seeds, F51.STEPS, check_mode,
                     dict(data_sha256=data.sha256, input_hash=CI.tree_hash(data.X), labels_hash=CI.tree_hash(data.Y)))
    bind_identity(out, ident)
    start = time.time()
    put(out / 'status.json', dict(stage='started', pid=os.getpid(), started_jst=now_jst(), admission=adm))
    raw = out / 'raw'
    pdir = out / 'prefix'
    if (pdir / 'done.json').exists():
        verify_done(pdir / 'done.json', ident, out)
        states = {t: CI.load_pt(raw / f'prefix_t{t:02d}.pt') for t in E1['F51']['states']}
        prefix_rows = json.loads((pdir / 'rows.json').read_text())
        log('F51 prefix: reused (same identity)')
    else:
        states, prefix_rows, sec = f51_prefix(seeds, data, log)
        for t, st in states.items():
            CI.save_pt(raw / f'prefix_t{t:02d}.pt', st)
        put(pdir / 'rows.json', prefix_rows)
        F51.write_csv(pdir / 'per_task.csv', prefix_rows)
        put(pdir / 'timing.json', dict(seconds=sec, updates=29 * F51.STEPS))
        mark_done(pdir / 'done.json', ident, files_under(pdir) + [raw / f'prefix_t{t:02d}.pt' for t in states], out)
    checks = {}
    for direction in DIRECTIONS:
        spec = SETTINGS['F51'][direction]
        probe_path = out / 'probes' / f'{direction}.json'
        if not probe_path.exists():
            put(probe_path, f51_branch_probe(spec, states, data))
        probe = json.loads(probe_path.read_text())
        assert probe['all_pass'], ('CHECK_FAILED branch probe', direction)
        checks[f'{direction}/branch_probe'] = True
        for kind in KINDS:
            name = spec[kind]
            adir = out / 'arms' / name
            if (adir / 'done.json').exists():
                verify_done(adir / 'done.json', ident, out)
                checks.update(json.loads((adir / 'arm_checks.json').read_text()))
                log(f'F51 arm {name}: reused')
                continue
            res = f51_arm(kind, spec, states, data, monitor=monitor)
            ac = {f'{name}/preflight': bool(res['preflight']['all_pass']),
                  f'{name}/finite': all(r['finite'] for r in res['rows'])}
            if kind == 'B' and monitor:
                ac[f'{name}/forward_monitor'] = monitor_pass(res['monitor'], res['updates'])
            if kind == 'N' and direction == 'restore':           # E.1's S-prefix identity (F51.prefix_check)
                pc = F51.prefix_check(name, res, states, prefix_rows)
                ac[f'{name}/natural_continuation'] = bool(pc['all_pass'])
            put(adir / 'rows.json', res['rows'])
            F51.write_csv(adir / 'per_task.csv', res['rows'])
            put(adir / 'preflight.json', res['preflight'])
            F51.serialize_history(adir, res['history'])
            put(adir / 'timing.json', dict(seconds=res['seconds'], updates=res['updates']))
            if res['monitor'] is not None:
                put(adir / 'monitor.json', dict(all_calls=res['monitor']))
            put(adir / 'arm_checks.json', ac)
            CI.save_pt(raw / 'arms' / name / 'end.pt', res['end_state'])
            checks.update(ac)
            assert all(v is not False for v in ac.values()), ('CHECK_FAILED', name, ac)
            mark_done(adir / 'done.json', ident, files_under(adir) + [raw / 'arms' / name / 'end.pt'], out)
            log(f'F51 arm {name}: {res["seconds"]:.1f}s')
            del res
    if list(seeds) == MAIN_SEEDS:
        put(out / 'e1_comparison.json', f51_e1_comparison(out, states, prefix_rows))
    finish(out, checks, start, 'F51')
    return checks


def f51_e1_comparison(out, states, prefix_rows):
    """Recorded, not gating: this boot's prefix states and host rows and the N / X online accuracies against
    E.1 (results/resp_cifar5p1_1007; eff_rank columns are float64 diagnostics that already differed there)."""
    run = E1['F51']['run']
    res = dict(states={}, prefix_rows={}, arms={})
    for t, st in states.items():
        old = e1_backup_state(run, f'prefix/t{t:02d}.pt')
        res['states'][t] = CI.tree_hash(F51.core_state(st)) == CI.tree_hash(F51.core_state(old))
    old_rows = {(r['seed'], r['task']): r for r in json.loads((ROOT / 'results' / run / 'prefix/rows.json').read_text())}
    diff = []
    for r in prefix_rows:
        o = old_rows[(r['seed'], r['task'])]
        cols = [k for k in r if k not in F51.HOST_FLOAT64_COLS and r[k] != o.get(k)]
        if cols:
            diff.append(dict(seed=r['seed'], task=r['task'], columns=cols))
    res['prefix_rows'] = dict(equal_except_eff_rank=len(diff) == 0, differing=diff[:50], n_differing=len(diff),
                              first_differing_task=min((d['task'] for d in diff), default=None))
    for direction in DIRECTIONS:
        spec = SETTINGS['F51'][direction]
        for kind in 'NX':
            res['arms'][spec[kind]] = compare_online(json.loads((out / 'arms' / spec[kind] / 'rows.json').read_text()),
                                                     json.loads((ROOT / 'results' / run / 'arms' / spec[kind] / 'rows.json').read_text()))
    return res


# ==========================================================================
# RM: random-label MNIST MLP, ELU (expm1 form), CPU (src/resp_ee_0917.py)
# ==========================================================================

def rm_natural(P, xb):
    return RM.EL.forward2(P, xb, RM.ACT, RM.ACT)           # the host forward of the natural arms


def rm_bwd_forward(sh):
    """Drop-in for the forward RM.make_forward(sh) builds: same shapes and ops as EL.forward2, the shifted
    layer through bwd_only (sh[l] = d_l on the 1200 images; sh['p0'] unused)."""
    d1, d2 = sh.get(1), sh.get(2)
    act = RM.ACT

    def fwd(params, xb, ob):
        W1, b1, W2, b2, W3, b3 = params
        z1 = xb @ W1.T + b1
        a1 = act.phi(z1) if d1 is None else bwd_only(act.phi, z1, d1[ob])
        z2 = a1 @ W2.T + b2
        a2 = act.phi(z2) if d2 is None else bwd_only(act.phi, z2, d2[ob])
        return z1, a1, z2, a2, a2 @ W3.T + b3
    return fwd


def rm_forward_factory(monitors=None):
    """Drop-in for RM.make_forward inside RM.run_arm."""
    def make(sh):
        if sh is None:
            return None
        f = rm_bwd_forward(sh)
        if monitors is not None:
            f = Monitor(f, rm_natural, 'cpu')
            monitors.append(f)
        return f
    return make


def rm_gtrain(z):
    """The native autograd derivative of RM's ELU at z (expm1 form)."""
    with torch.enable_grad():
        q = z.detach().requires_grad_(True)
        return torch.autograd.grad(RM.ACT.phi(q).sum(), q)[0].detach()


def rm_fwd(kind, spec, cks):
    sh = RM.build_shift(RM.ARMS[spec['X']], cks)
    if kind == 'N':
        return None, sh
    if kind == 'X':
        return RM.make_forward(sh), sh
    return rm_bwd_forward(sh), sh


def rm_branch_probe(spec, cks, x, n_batches=2):
    """As rc_branch_probe, on one RM seed (CPU): all 1200 images and the first training minibatches."""
    ck = cks[spec['branch']]
    y = RM.RL.task_labels(RM.gen_from(ck['g_lab']))
    order = torch.randperm(RM.N_IMG, generator=RM.gen_from(ck['g_batch']))
    batch_list = [('all', torch.arange(RM.N_IMG))]
    batch_list += [(f'mb{j}', order[j * RM.BATCH:(j + 1) * RM.BATCH]) for j in range(n_batches)]
    out = dict(direction_arms={k: spec[k] for k in KINDS}, batches={})
    fwds = {k: rm_fwd(k, spec, cks) for k in KINDS}
    d2 = fwds['B'][1][LAYER]
    for tag, ob in batch_list:
        res = {}
        for k in KINDS:
            params = [p.detach().clone().requires_grad_(True) for p in ck['params']]
            vals = RM.forward(params, x[ob], ob, fwds[k][0])
            loss = F.cross_entropy(vals[4], y[ob])
            g_loc = local_derivative(vals)
            grads = torch.autograd.grad(loss, params)
            res[k] = dict(vals=[v.detach() for v in vals], loss=loss.detach(), g=g_loc, grads=grads)
        z2 = res['B']['vals'][2]
        arg = z2 + d2[ob]
        gs, gp = rm_gtrain(arg), rm_gtrain(z2)
        ref = arg.double().clamp(max=0).exp()
        rep = dict(
            outputs_bit_equal={k: all(bits_equal(a, b) for a, b in zip(res[k]['vals'], res['N']['vals'])) for k in 'XB'},
            loss_bit_equal={k: bits_equal(res[k]['loss'], res['N']['loss']) for k in 'XB'},
            grads_X_B_bit_equal=all(bits_equal(a, b) for a, b in zip(res['X']['grads'], res['B']['grads'])),
            grads_N_B_differing_elements=int(sum(int((a != b).sum()) for a, b in zip(res['N']['grads'], res['B']['grads']))),
            derivative_B=derivative_report(res['B']['g'], gs, gp, ref, elu_expm1_bound(ref)),
            derivative_B_equals_dphi_train=bits_equal(res['B']['g'], RM.dphi_train(arg)),
            derivative_X_bit_equal_B=bits_equal(res['X']['g'], res['B']['g']),
            derivative_N_bit_equal_native_shifted=bits_equal(res['N']['g'], gs))
        out['batches'][tag] = rep
    out['all_pass'] = probe_pass(out) and all(r['derivative_B_equals_dphi_train'] for r in out['batches'].values())
    return out


def rm_arm(kind, spec, cks, x, prows, seed, epochs, units, monitor=True):
    """One arm exactly as RM.run_seed runs it (RM.run_arm, 2 tasks); for B, RM.make_forward is swapped for the
    backward-only forward during the call (the arm dict, field, Adam rule and readouts are X's)."""
    t0 = time.time()
    mon = None
    if kind == 'B':
        mons = [] if monitor else None
        with swapped(RM, 'make_forward', rm_forward_factory(mons)):
            rows = RM.run_arm(spec['B'], RM.ARMS[spec['X']], cks, x, prows, seed, epochs, units=units)
        if monitor:
            assert len(mons) == 1
            mon = mons[0].read()
    else:
        rows = RM.run_arm(spec[kind], RM.ARMS[spec[kind]], cks, x, prows, seed, epochs, units=units)
    for row in rows:
        row.update(kind=kind, e1_arm=spec['X'] if kind == 'B' else spec[kind], bwd_only=int(kind == 'B'))
    # B's expected forward calls: preflight's first epoch, unit_point at the start, every update, unit_point
    # at each task end (RM.run_arm / train_task / preflight)
    expected = min(RM.SPE, RM.SPE * epochs) + 1 + RM.CONT * RM.SPE * epochs + RM.CONT
    return rows, mon, expected, time.time() - t0


def rm_setup(threads=1):
    torch.set_num_threads(threads)
    torch.set_flush_denormal(True)
    RM.H.setup('cpu')
    RM.H.DATA_DIR = MNIST_DIR
    return RM.H.Mnist(torch.device('cpu'))


def run_rm_seed(seed, out, epochs, check_mode=False, log=print, monitor=True, prereg=None, mnist=None,
                kinds=KINDS):
    """One seed: E.1's prefix (tasks 1-12; branch states 2 and 10), then per direction the branch probe and
    the arms.  `kinds` other than all three is for checks only (e.g. N and X against E.1's records)."""
    assert check_mode or tuple(kinds) == KINDS
    adm = None if check_mode else admission('RM', MAIN_SEEDS if seed in MAIN_SEEDS else [seed], epochs, prereg)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mnist = rm_setup() if mnist is None else mnist
    ident = identity('RM', [seed], epochs, check_mode, dict(mnist_sha256=mnist.sha256))
    ident['kinds'] = list(kinds)
    bind_identity(out, ident)
    if (out / 'done.json').exists():
        verify_done(out / 'done.json', ident, out)
        log(f'RM seed {seed}: done (same identity)')
        return json.loads((out / 'run_checks.json').read_text())['checks']
    start = time.time()
    put(out / 'status.json', dict(stage='started', pid=os.getpid(), started_jst=now_jst(), admission=adm))
    prows, units, cks, x, info = RM.run_prefix(seed, mnist, epochs, n_tasks=12, save_at=(2, 10), progress=False)
    t_prefix = time.time() - start
    log(f'RM seed {seed}: prefix {t_prefix:.1f}s')
    checks, rows, probes, mons, timing = {}, [], {}, {}, {}
    for direction in DIRECTIONS:
        spec = SETTINGS['RM'][direction]
        probes[direction] = rm_branch_probe(spec, cks, x)
        checks[f'{direction}/branch_probe'] = bool(probes[direction]['all_pass'])
        assert checks[f'{direction}/branch_probe'], ('CHECK_FAILED branch probe', direction)
        for kind in kinds:
            rr, mon, expected, sec = rm_arm(kind, spec, cks, x, prows, seed, epochs, units, monitor=monitor)
            name = spec[kind]
            rows += rr
            timing[name] = dict(seconds=sec, updates=RM.CONT * RM.SPE * epochs)
            checks[f'{name}/finite'] = all(r['finite'] for r in rr)
            checks[f'{name}/branch_logits'] = bool(rr[0]['logits_equal_full'] and rr[0]['logits_equal_mb'])
            if kind == 'B' and monitor:
                mons[name] = dict(expected_calls=expected, **mon)
                checks[f'{name}/forward_monitor'] = bool(mon['eager_calls'] == expected and mon['eager_mismatch'] == 0)
            if kind == 'N' and direction == 'restore':           # E.1's S2: N10 is the prefix's tasks 11, 12
                checks[f'{name}/natural_continuation'] = all(r['hash_match_prefix'] and r['acc_match_prefix'] for r in rr)
            log(f'RM seed {seed}: arm {name} {sec:.1f}s  ' + ' '.join(f't{r["task"]} {r["online_acc"]:.4f}' for r in rr))
    RM.write_csv(out / 'prefix.csv', prows)
    RM.write_csv(out / 'arms.csv', rows)
    put(out / 'probes.json', probes)
    if mons:
        put(out / 'monitor.json', mons)
    np.savez_compressed(out / 'units.npz', **units)
    e1 = rm_e1_check(seed, prows, rows) if (seed in MAIN_SEEDS and epochs == BUDGET['RM']) else None
    if e1 is not None:
        put(out / 'e1_check.json', e1)
    put(out / 'run_checks.json', dict(all_pass=all(v is not False for v in checks.values()), checks=checks))
    put(out / 'provenance.json', dict(identity=ident, admission=adm, seconds_prefix=t_prefix,
                                      seconds_total=time.time() - start, arm_seconds=timing,
                                      peak_rss_bytes=peak_rss_bytes(), prefix_info=info, finished_jst=now_jst()))
    put(out / 'status.json', dict(stage='completed', elapsed_s=time.time() - start))
    assert all(v is not False for v in checks.values()), ('CHECK_FAILED', {k: v for k, v in checks.items() if v is False})
    mark_done(out / 'done.json', ident, files_under(out, exclude=('done.json', 'status.json')), out)
    return checks


def rm_e1_check(seed, prows, rows):
    """CPU: E.1's numbers are expected bit for bit (state sha256 and online accuracy of the prefix tasks 1-12 and
    of N / X's two tasks) -- results/resp_ee_0917/runs/s<seed>/{prefix,arms}.csv."""
    d = ROOT / 'results/resp_ee_0917/runs' / f's{seed}'
    old_p = {int(r['task']): r for r in csv.DictReader((d / 'prefix.csv').open())}
    old_a = {(r['arm'], int(r['k'])): r for r in csv.DictReader((d / 'arms.csv').open())}
    rep = []
    for r in prows:
        o = old_p[r['task']]
        rep.append(dict(what='prefix', task=r['task'], state=r['state_sha256'] == o['state_sha256'],
                        online=r['online_acc'] == float(o['online_acc'])))
    for r in rows:
        if r['kind'] == 'B':
            continue
        o = old_a[(r['arm'], r['k'])]
        rep.append(dict(what=r['arm'], task=r['task'], state=r['state_sha256'] == o['state_sha256'],
                        online=r['online_acc'] == float(o['online_acc']),
                        memo=r['memo_acc_end'] == float(o['memo_acc_end'])))
    return dict(all_equal=all(all(v for k, v in q.items() if k not in ('what', 'task')) for q in rep), rows=rep)


# ==========================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--setting', choices=tuple(SETTINGS), required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', default='0-9', help='RC / F51: the stacked seeds')
    ap.add_argument('--seed', type=int, help='RM: one seed per process')
    ap.add_argument('--epochs', type=int, help='RC / RM: epochs per task (registered: RC 400, RM 80)')
    ap.add_argument('--check-mode', action='store_true', help='checks and smoke only: other seeds/budgets, no admission')
    ap.add_argument('--prereg', default=os.environ.get('BWDSWAP_PREREG'))
    ap.add_argument('--no-monitor', action='store_true', help='checks only: B without the forward monitor')
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    if a.check_mode:
        assert not (out == OUT_MAIN / a.setting or out.parent == OUT_MAIN / 'RM'), 'check mode must not write the main output'
    monitor = not a.no_monitor
    assert monitor or a.check_mode
    if a.setting == 'RM':
        assert a.seed is not None
        epochs = a.epochs or BUDGET['RM']
        run_rm_seed(a.seed, out, epochs, a.check_mode, monitor=monitor, prereg=a.prereg)
    else:
        seeds = RE.E.parse_ints(a.seeds)
        with gpu_guard():
            if a.setting == 'RC':
                run_rc(out, seeds, a.epochs or BUDGET['RC'], a.check_mode, monitor=monitor, prereg=a.prereg)
            else:
                assert a.epochs is None, 'F51 runs E.1 registered 780 updates per task'
                run_f51(out, seeds, a.check_mode, monitor=monitor, prereg=a.prereg)
    print('completed', flush=True)


if __name__ == '__main__':
    main()

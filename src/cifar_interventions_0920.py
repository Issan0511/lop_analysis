"""Original-R CIFAR training, with explicit state for registered interventions.

The Adam operations and random streams match rlcifar_mlp_battle_0918.run.
No train/eval numerical policy is changed by this module.
"""
from __future__ import annotations
import copy, hashlib, json, os, sys, time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '2'
import numpy as np
import torch
import torch.nn.functional as F
from src import rlcifar_mlp_battle_0918 as E
from analysis.cifar_ledger_0920.replay import sha, put, save_npz, setup, exclusive, environment, output_artifact

ROOT = Path(__file__).resolve().parents[1]
APPROVAL = '5705359'
DATA = Path('/home/issan/Projects/claude/proj_004_drift/data/cifar10')
PARAMS = ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')


def cpu(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().clone()
    if isinstance(x, dict): return {k: cpu(v) for k, v in x.items()}
    if isinstance(x, list): return [cpu(v) for v in x]
    if isinstance(x, tuple): return tuple(cpu(v) for v in x)
    return copy.deepcopy(x)


def tree_hash(x):
    h = hashlib.sha256()
    def visit(y):
        if isinstance(y, torch.Tensor):
            z = y.detach().cpu().contiguous()
            h.update(str((z.dtype, tuple(z.shape))).encode()); h.update(z.numpy().tobytes())
        elif isinstance(y, np.ndarray):
            h.update(str((y.dtype, y.shape)).encode()); h.update(y.tobytes())
        elif isinstance(y, dict):
            for k in sorted(y, key=str): h.update(str(k).encode()); visit(y[k])
        elif isinstance(y, (tuple, list)):
            for item in y: visit(item)
        else: h.update(repr(y).encode())
    visit(x)
    return h.hexdigest()


def save_pt(path, state):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(cpu(state), tmp); tmp.replace(path)


def artifact(path):
    path=Path(path)
    if path.exists():return path
    root=next(q for q in path.parents if (q/'backup_manifest.json').exists())
    return output_artifact(root,path.relative_to(root))


def load_pt(path):
    return torch.load(artifact(path), map_location='cpu', weights_only=False)


def gtrain(z):
    with torch.enable_grad():
        q = z.detach().requires_grad_(True)
        return torch.autograd.grad(F.elu(q, 1.0).sum(), q)[0].detach()


class AnchorAdd(torch.autograd.Function):
    """Preserve signed zero in the frozen value; derivative w.r.t. diff is one."""
    @staticmethod
    def forward(ctx, frozen, diff):
        return torch.where(diff == 0, frozen, frozen + diff)
    @staticmethod
    def backward(ctx, grad): return None, grad


class Shift:
    def __init__(self, P0, layer, d):
        self.P0 = [p.detach().clone() for p in P0]
        self.layer = int(layer)
        self.d = d.detach().clone()

    def state(self): return cpu(dict(P0=self.P0, layer=self.layer, d=self.d))

    def __call__(self, P, xb, ids):
        with torch.no_grad():
            z01, a01, z02, a02, _ = E.forward(self.P0, xb, E.ELU())
        W1, b1, W2, b2, W3, b3 = P
        ar = torch.arange(xb.shape[0], device=xb.device)[:, None]
        d = self.d[ar, ids]
        z1 = torch.baddbmm(b1[:, None, :], xb, W1.transpose(1, 2))
        if self.layer == 1:
            a1 = AnchorAdd.apply(a01, F.elu(z1 + d) - F.elu(z01 + d))
        else: a1 = F.elu(z1)
        z2 = torch.baddbmm(b2[:, None, :], a1, W2.transpose(1, 2))
        if self.layer == 2:
            a2 = AnchorAdd.apply(a02, F.elu(z2 + d) - F.elu(z02 + d))
        else: a2 = F.elu(z2)
        return z1, a1, z2, a2, torch.baddbmm(b3[:, None, :], a2, W3.transpose(1, 2))


def make_inputs(seeds, device):
    E.RC.DATA_DIR = DATA
    cifar = E.RC.Cifar10()
    return torch.stack([E.slot_inputs(cifar, s, 'std', device) for s in seeds]), cifar


class Engine:
    def __init__(self, seeds, X, state=None, shift=None, graph=True):
        self.seeds = list(seeds); self.X = X; self.device = X.device
        self.R = len(seeds); self.shift = shift; self.graph_enabled = graph
        self.act = E.ELU(); self.act.init_state(self.R, self.device, key='registered-elu')
        init = {s: E.H.init_params(s, self.device, E.DIMS) for s in seeds}
        self.P = [torch.stack([init[s][i].detach() for s in seeds]).contiguous().requires_grad_(True) for i in range(6)]
        self.m = [torch.zeros_like(p) for p in self.P]
        self.v = [torch.zeros_like(p) for p in self.P]
        self.g_lab = {s: E.H.stream('rlc_labels', s) for s in seeds}
        self.g_batch = {s: E.H.stream('rlc_batch', s) for s in seeds}
        self.tc = 0; self.task = 0; self.epoch = 0; self.in_task = False
        self.order_chain = ''; self.label_hash = ''; self.diag_history = []
        self.ar = torch.arange(self.R, device=self.device)[:, None]
        self.ids = torch.zeros(self.R, E.BATCH, dtype=torch.long, device=self.device)
        self.Y = torch.zeros(self.R, E.N_IMAGES, dtype=torch.long, device=self.device)
        self.inv1 = torch.zeros((), device=self.device); self.inv2 = self.inv1.clone()
        self.acc_sum = torch.zeros(self.R, device=self.device)
        self.ce_sum = torch.zeros(self.R, device=self.device)
        self.step_t = torch.zeros((), dtype=torch.long, device=self.device)
        self.bad_step = torch.full((self.R,), -1, dtype=torch.long, device=self.device)
        self.last_hit = torch.zeros(self.R, device=self.device)
        self.first_g = []; self.cg = None
        if state is not None: self.restore(state)
        self.capture()

    def forward(self, xb, ids):
        return E.forward(self.P, xb, self.act, train=True) if self.shift is None else self.shift(self.P, xb, ids)

    def step(self):
        xb, yb = self.X[self.ar, self.ids], self.Y[self.ar, self.ids]
        vals = self.forward(xb, self.ids)
        z3 = vals[-1]
        lossv = F.cross_entropy(z3.reshape(-1, E.N_CLASSES), yb.reshape(-1),
                                reduction='none').view(self.R, E.BATCH).mean(1)
        hit = (z3.detach().argmax(-1) == yb).float().mean(1)
        self.acc_sum.add_(hit); self.last_hit.copy_(hit)
        grads = torch.autograd.grad(lossv.sum(), self.P)
        with torch.no_grad():
            self.ce_sum.add_(lossv.detach())
            bad = ~torch.isfinite(lossv)
            self.bad_step.copy_(torch.where((self.bad_step < 0) & bad, self.step_t, self.bad_step))
            self.step_t.add_(1)
            for p, gr, mi, vi in zip(self.P, grads, self.m, self.v):
                mi.mul_(0.9).add_(gr, alpha=1 - 0.9)
                vi.mul_(0.999).addcmul_(gr, gr, value=1 - 0.999)
                p.sub_(1e-3 * (mi * self.inv1) / ((vi * self.inv2).sqrt() + 1e-8))
        self.last_values = tuple(q.detach() for q in vals)

    def mutable_tensors(self):
        return [*self.P, *self.m, *self.v, self.acc_sum, self.ce_sum,
                self.step_t, self.bad_step, self.last_hit]

    def capture(self):
        if not self.graph_enabled: return
        assert self.device.type == 'cuda'
        tensors = self.mutable_tensors()
        keep = [q.detach().clone() for q in tensors]
        self.inv1.fill_(1.0); self.inv2.fill_(1.0)
        side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3): self.step()
        torch.cuda.current_stream().wait_stream(side)
        self.cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cg): self.step()
        with torch.no_grad():
            for q, old in zip(tensors, keep): q.copy_(old)

    def state(self):
        return cpu(dict(seeds=self.seeds, P=self.P, m=self.m, v=self.v, tc=self.tc,
                        task=self.task, epoch=self.epoch, in_task=self.in_task,
                        g_lab={s:g.get_state() for s,g in self.g_lab.items()},
                        g_batch={s:g.get_state() for s,g in self.g_batch.items()},
                        Y=self.Y, acc_sum=self.acc_sum, ce_sum=self.ce_sum,
                        step_t=self.step_t, bad_step=self.bad_step, last_hit=self.last_hit,
                        order_chain=self.order_chain, label_hash=self.label_hash,
                        diag_history=self.diag_history, first_g=self.first_g,
                        shift=None if self.shift is None else self.shift.state()))

    def restore(self, st):
        assert self.seeds == st['seeds'], 'slot mismatch'
        with torch.no_grad():
            for name in ('P', 'm', 'v'):
                for dst, src in zip(getattr(self, name), st[name]): dst.copy_(src)
            for name in ('Y','acc_sum','ce_sum','step_t','bad_step','last_hit'):
                getattr(self, name).copy_(st[name])
        for name in ('g_lab','g_batch'):
            for s, value in st[name].items(): getattr(self, name)[s].set_state(value)
        for name in ('tc','task','epoch','in_task','order_chain','label_hash','diag_history','first_g'):
            setattr(self, name, copy.deepcopy(st[name]))
        if st.get('shift') is not None:
            sh = st['shift']; self.shift = Shift([p.to(self.device) for p in sh['P0']], sh['layer'], sh['d'].to(self.device))

    def reset_adam(self):
        with torch.no_grad():
            for q in (*self.m, *self.v): q.zero_()
        self.tc = 0

    def one_update(self, ids):
        self.ids.copy_(ids); self.tc += 1
        self.inv1.fill_(1.0 / (1 - 0.9 ** self.tc))
        self.inv2.fill_(1.0 / (1 - 0.999 ** self.tc))
        if self.cg is None: self.step()
        else: self.cg.replay()

    def diagnostic(self):
        ids = torch.arange(E.N_IMAGES, device=self.device)[None, :].expand(self.R, -1)
        with torch.no_grad(): vals = self.forward(self.X, ids)
        z1, a1, z2, a2, logits = vals
        unit = {}; rows = [dict(seed=s, task=self.task, update=int(self.step_t)) for s in self.seeds]
        for l, z in ((1,z1), (2,z2)):
            arg = z if self.shift is None or self.shift.layer != l else z + self.shift.d
            g = gtrain(arg); gd = g.double(); zz = z.double()
            sums = gd.sum(1); sq = gd.square().sum(1)
            neff = torch.where(sq > 0, sums.square()/torch.where(sq>0,sq,torch.ones_like(sq)), 0)
            W, b = self.P[2*l-2].detach().double(), self.P[2*l-1].detach().double()
            sd = zz.std(1, unbiased=False); mu = zz.mean(1)
            for name, value in dict(zmean=mu, zsd=sd, gmean=gd.mean(1), q=(g.abs()<1e-6).double().mean(1),
                                    zero=(g==0).double().mean(1), neff=neff, neff_fraction=neff/E.N_IMAGES, Wnorm=W.norm(dim=2), bias=b,
                                    argmean=arg.double().mean(1), zmean_over_sd=torch.where(sd>0,mu/sd,torch.nan)).items(): unit[f'{name}_l{l}'] = value.cpu().numpy()
            for r, row in enumerate(rows):
                row.update({f'G{l}':float(gd[r].mean()), f'Q{l}':int((g[r].abs()<1e-6).sum())/(E.N_IMAGES*z.shape[-1]),
                            f'Q_count{l}':int((g[r].abs()<1e-6).sum()),f'pair_count{l}':int(g[r].numel()),
                            f'neff_zero_units{l}':int((sq[r]==0).sum()), f'zero_count{l}':int((g[r]==0).sum()),
                            f'zero{l}':float((g[r]==0).double().mean()), f'dead_unit{l}':float((g[r]==0).all(0).double().mean()),
                            f'neff{l}':float(neff[r].mean()), f'neff_fraction{l}':float(neff[r].mean()/E.N_IMAGES),
                            f'zmean{l}':float(mu[r].mean()), f'zsd{l}':float(sd[r].mean()),
                            f'diag_gdiff{l}':float((g[r]-self.act.dphi(arg[r])).abs().max()),
                            f'Wnorm{l}':float(W[r].norm(dim=1).mean()), f'bias{l}':float(b[r].mean())})
        mu2 = a1.double().mean(1); unit['mu2'] = mu2.cpu().numpy()
        wn=self.P[2].detach().double().norm(dim=2); mn=mu2.norm(dim=1)
        unit['cos_W2_mu2'] = ((self.P[2].detach().double()*mu2[:,None,:]).sum(2)/(wn*mn[:,None])).cpu().numpy()
        unit['mu2_norm'] = mn.cpu().numpy()
        for r,row in enumerate(rows):
            row.update(mu2_norm=float(mu2[r].norm()), memo_acc=float((logits[r].argmax(-1)==self.Y[r]).float().mean()),
                       memo_ce=float(F.cross_entropy(logits[r],self.Y[r])))
        return rows, unit

    def train_task(self, task, epochs=400, diagnostic_steps=(), checkpoint=None, stop=None, epoch_limit=None):
        if not self.in_task:
            self.task = task; self.epoch = 0; self.in_task = True
            lab = {s:E.RC.task_labels(self.g_lab[s]) for s in self.seeds}
            self.Y.copy_(torch.stack([lab[s] for s in self.seeds]).to(self.device))
            self.acc_sum.zero_(); self.ce_sum.zero_(); self.step_t.zero_(); self.bad_step.fill_(-1)
            self.first_g = []; self.diag_history = []; self.order_chain = ''
            self.label_hash = tree_hash(self.Y)
        assert self.task == task
        for ep in range(self.epoch, epochs):
            order = torch.stack([torch.randperm(E.N_IMAGES, generator=self.g_batch[s]) for s in self.seeds])
            self.order_chain = hashlib.sha256(self.order_chain.encode()+order.numpy().tobytes()).hexdigest()
            order = order.to(self.device)
            for j in range(E.STEPS_PER_EPOCH):
                ids = order[:, j*E.BATCH:(j+1)*E.BATCH]
                if ep == 0 and diagnostic_steps:
                    full_ids=torch.arange(E.N_IMAGES,device=self.device)[None,:].expand(self.R,-1)
                    with torch.no_grad():full_vals=self.forward(self.X,full_ids)
                    full_g=[]
                    for l,zi in ((1,0),(2,2)):
                        zz=full_vals[zi]
                        if self.shift is not None and self.shift.layer==l:zz=zz+self.shift.d
                        full_g.append(gtrain(zz)[self.ar,ids].cpu())
                self.one_update(ids)
                if ep == 0 and diagnostic_steps:
                    gs = []
                    for l, zi in ((1,0),(2,2)):
                        z = self.last_values[zi].detach()
                        if self.shift is not None and self.shift.layer == l: z = z + self.shift.d[self.ar, ids]
                        gs.append(gtrain(z).cpu())
                    self.first_g.append(dict(ids=ids.cpu(), g=gs, g_full_same_state=full_g))
                update = ep*E.STEPS_PER_EPOCH + j + 1
                if update in diagnostic_steps:
                    rows, unit = self.diagnostic(); self.diag_history.append(dict(update=update, rows=rows, units=unit))
            self.epoch = ep + 1
            request_stop = (stop is not None and Path(stop).exists()) or (epoch_limit is not None and self.epoch >= epoch_limit)
            if checkpoint is not None and (self.epoch%50==0 or self.epoch==epochs or request_stop): checkpoint(self.state())
            if request_stop and self.epoch < epochs: return None
        torch.cuda.synchronize(self.device)
        self.in_task = False
        rows, units = self.diagnostic()
        finite = all(torch.isfinite(p).all().item() for p in self.P) and (self.bad_step < 0).all().item()
        for r,row in enumerate(rows):
            row.update(online_acc=float(self.acc_sum[r])/(epochs*E.STEPS_PER_EPOCH),
                       online_ce=float(self.ce_sum[r])/(epochs*E.STEPS_PER_EPOCH),
                       major_frac=float(torch.bincount(self.Y[r],minlength=10).max())/E.N_IMAGES,
                       label_hash=self.label_hash, order_hash=self.order_chain, finite=bool(finite))
        return rows, units


def core_state(st):
    return {k:st[k] for k in ('P','m','v','tc','g_lab','g_batch','Y')}


def source_hashes(run):
    files = [ROOT/'src/cifar_interventions_0920.py', ROOT/f'src/{run}.py',
             ROOT/f'specs/spec_{run}.md', ROOT/'src/rlcifar_mlp_battle_0918.py',
             ROOT/'src/pmnist_0905.py', ROOT/'src/pmnist_rlcifar_0907.py',
             ROOT/'src/pmnist_rlmnist_0906.py', ROOT/'analysis/cifar_ledger_0920/replay.py',
             ROOT/'analysis/cifar_ledger_0920/ledger.py']
    files += sorted((ROOT/f'analysis/{run}').glob('*.py'))
    return {str(p.relative_to(ROOT)):sha(p) for p in files}


def identity(run, seeds, X, cifar, epochs):
    import subprocess
    return dict(run_id=run, seeds=list(seeds), epochs=epochs, approval_commit=APPROVAL,
                git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                source_sha256=source_hashes(run), environment=environment(),
                data_sha256=cifar.sha256, input_hash=tree_hash(X),
                subset_order={str(s):E.RC.subset_idx(s).tolist() for s in seeds})


def require_identity(saved, current):
    assert saved == current, 'resume identity mismatch'


def mark_done(path, ident, files):
    put(path, dict(identity=ident, files={str(p.name):sha(p) for p in files}, complete=True))


def verify_done(path, ident):
    obj=json.loads(Path(path).read_text()); require_identity(obj['identity'],ident)
    assert obj['complete'] and obj['files']
    for name,h in obj['files'].items():
        p=artifact(Path(path).parent/name)
        assert sha(p)==h, ('corrupt output',name)
    return obj

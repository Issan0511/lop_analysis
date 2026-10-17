#!/usr/bin/env python3
"""Registered activation × box cap experiment; native host engines, no CPU production fallback.

F51 run below is an explicit copy of cap_cifar5p1_1007.run with the registered
activation, bias-scaling and layer-lr differences. RC subclasses the actual
cifar_interventions_0920.Engine. Historical cells are rerun by default; historical
means are never used as new independent successes.
"""
from __future__ import annotations
import argparse, contextlib, csv, fcntl, hashlib, json, math, os, shutil, subprocess, sys, time
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn.functional as F
from src import pmnist_0905 as H
from src import rlcifar_mlp_battle_0918 as B
from src import cifar5p1_mlp_0920 as C
from src import cap_cifar5p1_1007 as OLD_F51
from src import cifar_interventions_0920 as I
from analysis.cifar_ledger_0920.replay import put, save_npz
ROOT = Path(__file__).resolve().parents[1]
RUN = EXPERIMENT = "cap_act_box_2x2_1018"
SPEC = "specs/spec_cap_act_box_2x2_1018.md"
COMMON = "specs/spec_six_runs_common_1018.md"
OUT_ROOT = ROOT / "results" / RUN
CAP_ARMS = {"ref": (), "cap12": (0,1), "cap12b": (0,1), "lr_only": ()}
ARM_ORDER = tuple(CAP_ARMS)
ACT_ARM, COND, LR = "R", "std", 1e-4
RADIUS_TASK, CAP_FROM = 1, 2
CALIB_SEEDS = list(range(100,110))
EVAL_SEEDS = list(range(10))
row_norms, monitor_, cap_stats = OLD_F51.row_norms, OLD_F51.monitor_, OLD_F51.cap_stats
sha256_file = OLD_F51.sha256_file


@contextlib.contextmanager
def exclusive():
    """Keep the lab lock; allow known desktop C+G contexts on the host."""
    with Path('/tmp/lop_analysis_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(['nvidia-smi','--query-compute-apps=pid',
                                 '--format=csv,noheader,nounits'], capture_output=True,
                                text=True, check=True)
        for line in result.stdout.splitlines():
            if not line.strip().isdigit() or int(line) == os.getpid():
                continue
            try:
                words = Path(f'/proc/{int(line)}/cmdline').read_bytes().split(b'\0')
            except FileNotFoundError:
                continue
            executable = Path(os.fsdecode(words[0])).name
            desktop = executable in {'gnome-remote-desktop-daemon','gnome-shell','nautilus','ptyxis'}
            chromium_gpu = b'--type=gpu-process' in words
            assert desktop or chromium_gpu, f'GPU busy: {line} {words!r}'
        yield


def native_act(name):
    if name not in ("ELU", "ReLU"):
        raise ValueError(name)
    return C.make_act("ELU" if name == "ELU" else "R")


def native_gradient(z, activation):
    with torch.enable_grad():
        zz=z.detach().requires_grad_(True)
        a=native_act(activation).phi(zz, 0, True)
        return torch.autograd.grad(a.sum(),zz)[0].detach()


@torch.no_grad()
def project_rows_(W, r, rows, rem, bias=None):
    """W-only row coefficient; inside rows are bit unchanged, including signed zero."""
    n=row_norms(W); over=n>r
    safe=torch.where(over,n,torch.ones_like(n))
    c=torch.where(over,r/safe,torch.ones_like(n))
    W.copy_(torch.where(over,W*c,W))
    if bias is not None:
        bias.copy_(torch.where(over[...,0],bias*c[...,0],bias))
    rows.add_(over.sum(dim=(1,2)))
    rem.add_((n.double()-r.double()).clamp(min=0).sum(dim=(1,2)))
    return n


@torch.no_grad()
def unit_diagnostics(P, X, act, activation):
    """Population s and Var(native derivative); unitwise r, explicit zero widths."""
    vals=B.forward(P,X,act,train=False); out={}
    for l,z in enumerate((vals[0],vals[2]),1):
        z=z.double(); m=z.mean(1); s=z.std(1,correction=0)
        g=native_gradient(z.float(),activation).double()
        out.update({f"m_l{l}":m.cpu().numpy(), f"s_l{l}":s.cpu().numpy(),
                    f"r_l{l}":torch.where(s>0,m/s,torch.nan).cpu().numpy(),
                    f"s_zero_l{l}":(s==0).cpu().numpy(),
                    f"s_zero_msign_l{l}":torch.where(s==0,m.sign(),0).cpu().numpy(),
                    f"native_dphi_mean_l{l}":g.mean(1).cpu().numpy(),
                    f"native_dphi_var_l{l}":g.var(1,correction=0).cpu().numpy(),
                    f"dead_l{l}":(g==0).all(1).cpu().numpy(),
                    f"Wnorm_l{l}":P[2*l-2].detach().double().norm(dim=2).cpu().numpy(),
                    f"bias_l{l}":P[2*l-1].detach().double().cpu().numpy()})
    return out


def schedule_from_rows(ref, cap):
    """Calibrate relative real-displacement RMS using calibration seeds only; no clipping."""
    expected={(s,t) for s in CALIB_SEEDS for t in range(1,31)}
    def index(rr):
        keys=[(int(r['seed']),int(r['task'])) for r in rr]
        if len(keys)!=len(set(keys)) or set(keys)!=expected:
            raise ValueError('SCHEDULE_UNDEFINED: missing/duplicate/calibration seeds')
        return dict(zip(keys,rr))
    a,b=index(ref),index(cap); q={}
    for t in range(1,31):
        q[str(t)]=[]
        for l in (1,2):
            ratios=[]
            for s in CALIB_SEEDS:
                av,bv=float(a[s,t][f'U_l{l}']),float(b[s,t][f'U_l{l}'])
                if not np.isfinite(av) or av<=0 or not np.isfinite(bv):
                    raise ValueError('SCHEDULE_UNDEFINED: zero/nonfinite denominator')
                ratios.append(bv/av)
            value=float(np.median(ratios))
            q[str(t)].append(1.0 if t==1 else value)
    return q


def seal_completed_arm(base,seeds,tasks):
    """Freeze every arm artifact before its COMPLETED status is published."""
    base=Path(base)
    files={str(p.relative_to(base)):dict(bytes=p.stat().st_size,sha256=sha256_file(p))
           for p in sorted(base.rglob('*')) if p.is_file() and p.name not in ('status.json','output_manifest.json')}
    if not files:raise ValueError('CHECK_FAILED: cannot seal empty arm')
    put(base/'output_manifest.json',dict(source_sha256=source_hashes(),files=files))
    put(base/'status.json',dict(status='COMPLETED',tasks=tasks,seeds=seeds,
        output_manifest_sha256=sha256_file(base/'output_manifest.json')))


def verify_completed_arm(base):
    base=Path(base);status=json.loads((base/'status.json').read_text())
    if status.get('status')!='COMPLETED':raise ValueError('CHECK_FAILED: arm is not COMPLETED')
    path=base/'output_manifest.json'
    if status.get('output_manifest_sha256')!=sha256_file(path):raise ValueError('CHECK_FAILED: output manifest changed')
    manifest=json.loads(path.read_text())
    if manifest.get('source_sha256')!=source_hashes() or not manifest.get('files'):
        raise ValueError('CHECK_FAILED: stale/missing completed output source closure')
    for relative,record in manifest['files'].items():
        rel=Path(relative)
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('CHECK_FAILED: invalid output manifest path')
        p=base/rel
        if p.stat().st_size!=record['bytes'] or sha256_file(p)!=record['sha256']:
            raise ValueError('CHECK_FAILED: completed output changed: '+relative)
    return manifest


def validate_calibration_arm(base,arm):
    """A schedule cannot admit a failed, incomplete, stale or mismatched arm."""
    base=Path(base)
    verify_completed_arm(base)
    status=json.loads((base/'status.json').read_text())
    if status.get('status')!='COMPLETED' or status.get('seeds')!=CALIB_SEEDS:
        raise ValueError('SCHEDULE_UNDEFINED: calibration status/seeds '+arm)
    prov=json.loads((base/'provenance.json').read_text())
    reg=prov.get('registration',{})
    expected=dict(box='F51',activation='ReLU',arm=arm,seeds=CALIB_SEEDS,R=10)
    if any(reg.get(k)!=v for k,v in expected.items()) or reg.get('source_sha256')!=source_hashes():
        raise ValueError('SCHEDULE_UNDEFINED: calibration provenance '+arm)
    expected_keys={(s,t) for s in CALIB_SEEDS for t in range(1,31)}
    indexed={}
    for filename in ('per_task.csv','cap_task.csv'):
        rr=list(csv.DictReader((base/filename).open()))
        keys=[(int(r['seed']),int(r['task'])) for r in rr]
        if len(keys)!=300 or set(keys)!=expected_keys:
            raise ValueError('SCHEDULE_UNDEFINED: calibration coverage '+filename)
        indexed[filename]=dict(zip(keys,rr))
    for key,r in indexed['per_task.csv'].items():
        if not np.isfinite(float(r.get('online_acc','nan'))) or not 0<=float(r['online_acc'])<=1:
            raise ValueError('SCHEDULE_UNDEFINED: calibration performance')
        c=indexed['cap_task.csv'][key]
        for l in (1,2):
            if int(c[f'viol_w{l}'])!=0 or not np.isfinite(float(c[f'U_l{l}'])) or float(c[f'U_l{l}'])<0 or (arm=='ref' and float(c[f'U_l{l}'])==0):
                raise ValueError('SCHEDULE_UNDEFINED: calibration cap integrity/U')
            if (arm=='ref' or key[1]==1) and int(c[f'rows_w{l}'])!=0:
                raise ValueError('SCHEDULE_UNDEFINED: unregistered calibration cap writes')
            actual=float(c[f'actual_lr_l{l}']);want=float(f'{float(torch.tensor(1e-4)):.10g}')
            if actual!=want:raise ValueError('SCHEDULE_UNDEFINED: calibration lr')
    fresh=list(csv.DictReader((base/'fresh_control.csv').open()))
    keys=[int(r['seed']) for r in fresh]
    if len(keys)!=10 or set(keys)!=set(CALIB_SEEDS):
        raise ValueError('SCHEDULE_UNDEFINED: missing fresh control')
    for r in fresh:
        if int(r['task'])!=29 or not all(np.isfinite(float(r[k])) for k in ('fresh_online_acc','continual_online_acc','fresh_gap')):
            raise ValueError('SCHEDULE_UNDEFINED: fresh control')
        # Each host CSV is printed to ten significant digits; a priori sum of
        # decimal rounding errors is < 1.5e-9 on accuracy/gap scales <=1.
        if abs(float(r['fresh_online_acc'])-float(r['continual_online_acc'])-float(r['fresh_gap']))>1.5e-9:
            raise ValueError('SCHEDULE_UNDEFINED: fresh gap closure')
    if prov.get('fresh_rows_written')!=[0,0]:raise ValueError('SCHEDULE_UNDEFINED: cap active in fresh')
    with np.load(base/'radii.npz') as radii,np.load(base/'radius_anchor.npz') as anchor:
        if radii['seeds'].tolist()!=CALIB_SEEDS:raise ValueError('SCHEDULE_UNDEFINED: radii seeds')
        for l in (1,2):
            r=radii[f'r_l{l}'];w=anchor[f'W{l}'];n=np.linalg.norm(w.astype(np.float64),axis=2)
            if r.shape!=(10,100) or not np.isfinite(r).all() or (r<0).any():raise ValueError('SCHEDULE_UNDEFINED: radii')
            bound=OLD_F51.gamma(2*w.shape[2]+2,OLD_F51.U32)*n+math.sqrt(w.shape[2])*OLD_F51.TINY32
            if not np.all(np.abs(n-r)<=bound):raise ValueError('SCHEDULE_UNDEFINED: t1 radius anchor')
            if prov['radii_sha256'][f'r_l{l}']!=hashlib.sha256(r.tobytes()).hexdigest():raise ValueError('SCHEDULE_UNDEFINED: radius hash')
    files=('status.json','output_manifest.json','provenance.json','per_task.csv','cap_task.csv','fresh_control.csv','radii.npz','radius_anchor.npz','stream_manifest.json')
    return {name:dict(path=str((base/name).resolve()),sha256=sha256_file(base/name)) for name in files}


def validate_schedule(obj):
    if obj.get('calibration_seeds')!=CALIB_SEEDS or obj.get('source_sha256')!=source_hashes():
        raise ValueError('SCHEDULE_UNDEFINED: seeds/source mismatch')
    evidence=obj.get('calibration_evidence',{})
    for arm in ('ref','cap12'):
        base=Path(obj['input_csv'][arm]['path']).parent
        current=validate_calibration_arm(base,arm)
        if evidence.get(arm)!=current:raise ValueError('SCHEDULE_UNDEFINED: stale calibration evidence')
    ref_base=Path(obj['input_csv']['ref']['path']).parent;cap_base=Path(obj['input_csv']['cap12']['path']).parent
    if json.loads((ref_base/'stream_manifest.json').read_text())!=json.loads((cap_base/'stream_manifest.json').read_text()):
        raise ValueError('SCHEDULE_UNDEFINED: unpaired calibration streams')
    if json.loads((ref_base/'provenance.json').read_text())['data_sha256']!=json.loads((cap_base/'provenance.json').read_text())['data_sha256']:
        raise ValueError('SCHEDULE_UNDEFINED: unpaired calibration data')
    q=obj['q']; expected={str(t) for t in range(1,31)}
    if set(q)!=expected or q['1']!=[1.,1.]:
        raise ValueError('SCHEDULE_UNDEFINED: task coverage')
    for values in q.values():
        if len(values)!=2 or not all(np.isfinite(v) and v>=0 for v in values):
            raise ValueError('SCHEDULE_UNDEFINED: value')
    for name in ('ref','cap12'):
        path=Path(obj['input_csv'][name]['path'])
        if sha256_file(path)!=obj['input_csv'][name]['sha256']:
            raise ValueError('SCHEDULE_UNDEFINED: input changed')
    rr=[list(csv.DictReader(Path(obj['input_csv'][n]['path']).open())) for n in ('ref','cap12')]
    if schedule_from_rows(*rr)!=q:
        raise ValueError('SCHEDULE_UNDEFINED: independent recomputation failed')
    return q

def run_f51(cap_arm: str, seeds: list[int], n_tasks: int, device, out: Path, lr: float = LR,
        cifar: C.Cifar100 | None = None, fresh: bool = True, graph: bool = True,
        progress=None, debug: dict | None = None, observer=None,
        steps_hard: int = C.STEPS_PER_TASK, steps_easy: int = C.STEPS_PER_TASK,
        activation: str = "ReLU", schedule: dict | None = None) -> dict:
    """Train one cap arm's R = len(seeds) runs in lockstep over the 30-task sequence.

    `debug` (checks only) makes the steps eager; `debug["hook"]`, if given, is called as
    hook(event, **state) at "pre"/"post" (around the projection, every update),
    "task_start", "task_end", "radius" and "fresh_start".  `observer(t, state)` is called at
    every task end with P, the Adam moments and the task's training images (diagnostics; the
    graph is kept).  Neither writes to the training state.
    """
    t_start = time.time()
    progress = progress or (lambda m: print(m, flush=True))
    if cap_arm not in CAP_ARMS:
        raise SystemExit(f"unknown cap arm {cap_arm!r}; known: {','.join(ARM_ORDER)}")
    layers = CAP_ARMS[cap_arm]
    arm, cond = ("ELU" if activation == "ELU" else "R"), COND
    act = C.make_act(arm)
    R = len(seeds)
    cifar = cifar or C.Cifar100()
    X_all = cifar.inputs("train", cond, device)
    Y_all = cifar.train_y.to(device)
    X_test = cifar.inputs("test", cond, device)
    Y_test = cifar.test_y.to(device)
    input_tensor_hash=I.tree_hash(X_all)
    tr_rows, te_rows = C.class_rows(cifar.train_y), C.class_rows(cifar.test_y)
    plans = [C.task_plan(s) for s in seeds]
    g_batch = {s: H.stream("c51_batch", s) for s in seeds}

    init = [q.detach() for s in seeds for q in C.init_params(arm, s, device)]
    P = [torch.stack(init[i::6]).contiguous() for i in range(6)]
    P = [q.requires_grad_(True) for q in P]
    P0 = [q.detach().clone() for q in P]
    out.mkdir(parents=True,exist_ok=True)
    I.save_pt(out / "initial.pt",dict(P=P0,seeds=seeds))
    act.init_state(R, device, key=f"{arm}|{seeds}|{cond}")
    act0 = act.state()
    adam_m = [torch.zeros_like(q) for q in P]
    adam_v = [torch.zeros_like(q) for q in P]
    hook = None if debug is None else debug.get("hook")
    if debug is not None:
        debug["init"] = [q.detach().cpu().clone() for q in P]

    # ---- cap state: static tensors the captured step reads by address
    rad = [torch.full((R, P[2 * l].shape[1], 1), math.inf, device=device) for l in range(2)]
    rows_acc = torch.zeros(R, 2, dtype=torch.long, device=device)
    rem_acc = torch.zeros(R, 2, dtype=torch.float64, device=device)
    viol_acc = torch.zeros(R, 2, dtype=torch.long, device=device)
    excess_acc = torch.full((R, 2), -math.inf, dtype=torch.float64, device=device)

    # ---- one training step on static tensors (the host's, plus the projection)
    b1, b2, eps = 0.9, 0.999, 1e-8
    static_idx = torch.zeros(R, C.BATCH, dtype=torch.long, device=device)
    inv_c1 = torch.zeros((), device=device)
    inv_c2 = torch.zeros((), device=device)
    step_t = torch.zeros((), dtype=torch.long, device=device)
    acc_sum = torch.zeros(R, device=device)
    bad_step = torch.full((R,), -1, dtype=torch.long, device=device)
    last_hit = torch.zeros(R, device=device)

    # Persistent graph-readable per-layer lrs, modified only at task boundaries.
    layer_lr = torch.full((3,), lr, device=device)
    u_sum = torch.zeros(R, 2, dtype=torch.float64, device=device)
    u_steps = torch.zeros((), dtype=torch.long, device=device)
    radius_trace = []
    batch_manifest = []
    cap_before_max = torch.zeros(R,2,100,dtype=torch.float64,device=device)
    cap_after_max = torch.zeros_like(cap_before_max)
    unit_hits = torch.zeros(R,2,100,dtype=torch.long,device=device)
    unit_removed = torch.zeros_like(cap_before_max)
    def persist_units(t, Xt):
        u = unit_diagnostics(P, Xt, act, activation)
        for l in range(2):
            u[f"radius_l{l+1}"] = r_saved[l].detach().cpu().numpy()
        u.update(projection_before_norm_max=cap_before_max.cpu().numpy(),
                 projection_after_norm_max=cap_after_max.cpu().numpy(),
                 projection_unit_hits=unit_hits.cpu().numpy(),
                 projection_unit_removed=unit_removed.cpu().numpy())
        save_npz(out / f"units_t{t:02d}.npz", u)
    post = getattr(act, "post_update", None)

    def step():
        xb, yb = X_all[static_idx], Y_all[static_idx]            # gathers, no arithmetic
        z1, a1, z2, a2, z3 = B.forward(P, xb, act, train=True)
        lossv = F.cross_entropy(z3.reshape(-1, C.N_CLASSES), yb.reshape(-1),
                                reduction="none").view(R, C.BATCH).mean(1)
        hit = (z3.detach().argmax(-1) == yb).float().mean(1)     # pre-update: the paper's a_j
        acc_sum.add_(hit)
        last_hit.copy_(hit)
        grads = torch.autograd.grad(lossv.sum(), P)
        with torch.no_grad():
            bad = ~torch.isfinite(lossv)
            bad_step.copy_(torch.where((bad_step < 0) & bad, step_t, bad_step))
            step_t.add_(1)
            before_w = [P[i].detach().clone() for i in (0,2)]
            for pi, (p, gr, mi, vi) in enumerate(zip(P, grads, adam_m, adam_v)):
                mi.mul_(b1).add_(gr, alpha=1 - b1)
                vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                p.sub_(layer_lr[pi // 2] * (mi * inv_c1) / ((vi * inv_c2).sqrt() + eps))
            act.update(z1.detach(), z2.detach())
            if post is not None:
                post(P, lr)
            if hook is not None:
                hook("pre", P=P, m=adam_m, v=adam_v, rad=rad, rows=rows_acc, rem=rem_acc)
            for l in layers:                                     # spec 2.4: W1 first, then W2
                W = P[2 * l]
                n64b = torch.linalg.vector_norm(W.double(), dim=2, keepdim=True)
                n32 = project_rows_(W, rad[l], rows_acc[:, l], rem_acc[:, l],
                                     P[2*l+1] if cap_arm == "cap12b" else None)
                monitor_(W, rad[l], n32, n64b, viol_acc[:, l], excess_acc[:, l])
                unit_hits[:,l].add_((n32>rad[l])[...,0])
                unit_removed[:,l].add_((n32.double()-rad[l].double()).clamp(min=0)[...,0])
                cap_before_max[:,l].copy_(torch.maximum(cap_before_max[:,l],n64b[...,0]))
                cap_after_max[:,l].copy_(torch.maximum(cap_after_max[:,l],W.double().norm(dim=2)))
            for l in range(2):
                before_norm = before_w[l].double().flatten(1).norm(dim=1)
                displacement = (P[2*l].double()-before_w[l].double()).flatten(1).norm(dim=1)
                rel = torch.where(before_norm>0, displacement/before_norm, torch.nan)
                u_sum[:,l].add_(rel.square())
            u_steps.add_(1)
            if hook is not None:
                hook("post", P=P, m=adam_m, v=adam_v, rad=rad, rows=rows_acc, rem=rem_acc)

    use_graph = graph and device.type == "cuda" and debug is None
    cg = None
    mutable = (*P, *adam_m, *adam_v, acc_sum, bad_step, step_t, last_hit,
               rows_acc, rem_acc, viol_acc, excess_acc, u_sum, u_steps,
               cap_before_max, cap_after_max, unit_hits, unit_removed)
    if use_graph:
        keep = [q.detach().clone() for q in mutable]
        keep_act = act.state()
        inv_c1.fill_(1.0); inv_c2.fill_(1.0)
        static_idx.zero_()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                act.begin_step(R, C.BATCH, device)
                step()
        torch.cuda.current_stream().wait_stream(side)
        cg = torch.cuda.CUDAGraph()
        act.begin_step(R, C.BATCH, device)
        with torch.cuda.graph(cg):
            step()
        with torch.no_grad():
            for q, v in zip(mutable, keep):
                q.copy_(v)
        act.load_state(keep_act)
        del keep

    def train_task(batches, tc0: int) -> int:
        """`batches.shape[1]` steps on (R, steps, 32) global row indices.  New Adam counter."""
        tc = tc0
        acc_sum.zero_()
        u_sum.zero_(); u_steps.zero_()
        cap_before_max.zero_();cap_after_max.zero_();unit_hits.zero_();unit_removed.zero_()
        bad_step.fill_(-1)
        step_t.zero_()
        rows_acc.zero_(); rem_acc.zero_(); viol_acc.zero_(); excess_acc.fill_(-math.inf)
        for j in range(batches.shape[1]):
            static_idx.copy_(batches[:, j])
            tc += 1
            inv_c1.fill_(1.0 / (1 - b1 ** tc))
            inv_c2.fill_(1.0 / (1 - b2 ** tc))
            act.begin_step(R, C.BATCH, device)
            if cg is not None:
                cg.replay()
            else:
                step()
            if debug is not None:
                debug.setdefault("online", []).append(last_hit.cpu().clone())
        if device.type == "cuda":
            torch.cuda.synchronize()
        return tc

    alive = torch.ones(R, dtype=torch.bool, device=device)
    rows, cap_rows, diverged, tc = [], [], [], 0
    last_hard = max(t for t in range(1, n_tasks + 1) if t % 2 == 1)
    saved = {}
    out.mkdir(parents=True, exist_ok=True)
    step_ms = float("nan")
    r_saved = [None, None]                 # task-1-end row norms (recorded in every arm)
    totals = {"rows_w1": 0, "rows_w2": 0, "viol_w1": 0, "viol_w2": 0}

    for t in range(1, n_tasks + 1):
        hard = t % 2 == 1
        steps = steps_hard if hard else steps_easy
        rows_t = [torch.cat([tr_rows[q] for q in plans[r][t - 1][1]]) for r in range(R)]
        batches = torch.stack([C.batch_indices(g_batch[s], rows_t[r], steps)
                               for r, s in enumerate(seeds)]).to(device)
        batch_manifest.append(dict(task=t,batch_hash=I.tree_hash(batches),
            image_ids_hash=I.tree_hash(rows_t),label_hash=I.tree_hash([Y_all[q] for q in rows_t])))
        put(out / "stream_manifest.json",dict(seeds=seeds,initial_hash=I.tree_hash(P0),data_sha256=cifar.sha256,input_tensor_sha256=input_tensor_hash,tasks=batch_manifest))
        if t == last_hard and fresh:
            saved[t] = batches.clone()
        if debug is not None:
            debug.setdefault("batches", []).append(batches.cpu().clone())
        with torch.no_grad():
            values = [1.,1.] if schedule is None else schedule[str(t)]
            layer_lr.copy_(torch.tensor([lr*values[0],lr*values[1],lr],device=device))
        if t == CAP_FROM:
            with torch.no_grad():
                for l in layers:
                    rad[l].copy_(r_saved[l])
        if hook is not None:
            hook("task_start", t=t, rad=rad)
        t0 = time.time()
        tc = train_task(batches, tc)
        step_ms = 1e3 * (time.time() - t0) / steps
        if hook is not None:
            hook("task_end", t=t, P=P, m=adam_m, v=adam_v, rad=rad)
        if t == RADIUS_TASK:
            with torch.no_grad():
                r_saved = [row_norms(P[2 * l]).detach().clone() for l in range(2)]
            save_npz(out / "radius_anchor.npz", {f"W{l+1}":P[2*l].detach().cpu().numpy() for l in range(2)})
            if hook is not None:
                hook("radius", t=t, r_saved=r_saved)

        with torch.no_grad():
            finite = torch.stack([torch.isfinite(q).flatten(1).all(1) for q in P]).all(0)
            newly = alive & ((bad_step >= 0) | ~finite)
            for r in torch.nonzero(newly).flatten().tolist():
                diverged.append({"diverged": True, "task": t, "seed": seeds[r], "arm": arm,
                                 "cap_arm": cap_arm, "cond": cond,
                                 "step": (t - 1) * C.STEPS_PER_TASK + max(int(bad_step[r]), 0)})
                rows.append({"arm": arm, "cond": cond, "seed": seeds[r], "slot": r, "lr": lr,
                             "task": t, "hard": int(hard), "acc": float("nan")})
            alive &= ~newly
            live = torch.nonzero(alive).flatten().tolist()
            Xt = torch.stack([X_all[rows_t[r]] for r in range(R)])
            Yt = torch.stack([Y_all[rows_t[r]] for r in range(R)])
            m = C.evaluate(P, Xt, Yt, act, live)
            extra = cap_stats(P, Xt, act, r_saved)
            if observer is not None:
                observer(t, dict(P=P, m=adam_m, v=adam_v, Xt=Xt, act=act, live=live, rad=rad))
            persist_units(t, Xt)
            del Xt
            te = [torch.cat([te_rows[q] for q in plans[r][t - 1][1]]) for r in range(R)]
            test_acc = C.accuracy(P, torch.stack([X_test[q] for q in te]),
                                  torch.stack([Y_test[q] for q in te]), act)
        for r in live:
            rows.append({"arm": arm, "cond": cond, "seed": seeds[r], "slot": r, "lr": lr,
                         "task": t, "hard": int(hard), "n_classes": len(plans[r][t - 1][1]),
                         "online_acc": float(acc_sum[r]) / steps,
                         "train_acc": m[r]["acc"], "test_acc": float(test_acc[r]), **m[r]})
            cr = {"cap_arm": cap_arm, "seed": seeds[r], "slot": r, "task": t, "hard": int(hard),
                  "U_l1":float((u_sum[r,0]/u_steps).sqrt()), "U_l2":float((u_sum[r,1]/u_steps).sqrt()),
                  "actual_lr_l1":float(layer_lr[0]), "actual_lr_l2":float(layer_lr[1]), "actual_lr_l3":float(layer_lr[2])}
            for l in range(2):
                cr[f"rows_w{l + 1}"] = int(rows_acc[r, l])
                cr[f"rem_w{l + 1}"] = float(rem_acc[r, l])
            for l in range(2):
                cr[f"viol_w{l + 1}"] = int(viol_acc[r, l])
            for l in range(2):
                cr[f"excess_w{l + 1}"] = float(excess_acc[r, l])
            for k in ("ratio_max_l1", "ratio_med_l1", "ratio_max_l2", "ratio_med_l2",
                      "mu2_norm", "b1_mean", "b2_mean"):
                cr[k] = float(extra[k][r]) if k in extra else float("nan")
            cap_rows.append(cr)
        for l in range(2):
            totals[f"rows_w{l + 1}"] += int(rows_acc[:, l].sum())
            totals[f"viol_w{l + 1}"] += int(viol_acc[:, l].sum())
        rows.sort(key=lambda q: (q["slot"], q["task"]))
        cap_rows.sort(key=lambda q: (q["slot"], q["task"]))
        H.write_csv(out / "per_task.csv", rows)
        H.write_csv(out / "cap_task.csv", cap_rows)
        on = acc_sum[alive] / steps
        el = time.time() - t_start
        progress(f"[{time.strftime('%T')}] {cap_arm} task {t:2d}/{n_tasks} "
                 f"{'hard' if hard else 'easy'} alive {int(alive.sum())}/{R} "
                 f"online {float(on.mean()) if len(on) else float('nan'):.3f} "
                 f"rows w1 {int(rows_acc[:, 0].sum())} w2 {int(rows_acc[:, 1].sum())} "
                 f"viol {int(viol_acc.sum())}  {step_ms:.2f} ms/step  {el / 60:.1f} min")

    # ---- fresh-network control on the last hard task: the host's procedure, no cap (spec 2.5)
    fresh_rows, fresh_written = [], None
    if fresh and last_hard in saved:
        continual = {int(q["seed"]): q["online_acc"] for q in rows
                     if q["task"] == last_hard and "online_acc" in q}
        with torch.no_grad():
            for q, v in zip(P, P0):
                q.copy_(v)
            for q in (*adam_m, *adam_v):
                q.zero_()
            for l in range(2):
                rad[l].fill_(math.inf)
        layer_lr.fill_(lr)
        act.load_state(act0)
        if hook is not None:
            hook("fresh_start", rad=rad, P=P, m=adam_m, v=adam_v)
        alive_f = torch.ones(R, dtype=torch.bool, device=device)
        train_task(saved[last_hard], 0)
        fresh_written = [int(x) for x in rows_acc.sum(0).tolist()]
        with torch.no_grad():
            finite = torch.stack([torch.isfinite(q).flatten(1).all(1) for q in P]).all(0)
            alive_f &= (bad_step < 0) & finite
        for r in range(R):
            if not bool(alive_f[r]) or seeds[r] not in continual:
                continue
            value = float(acc_sum[r]) / steps_hard
            fresh_rows.append({"arm": arm, "cond": cond, "seed": seeds[r], "lr": lr,
                               "task": last_hard, "fresh_online_acc": value,
                               "continual_online_acc": continual[seeds[r]],
                               "fresh_gap": value - continual[seeds[r]]})
        H.write_csv(out / "fresh_control.csv", fresh_rows)
        gaps = [q["fresh_gap"] for q in fresh_rows]
        progress(f"[{time.strftime('%T')}] {cap_arm} fresh control on task {last_hard}: "
                 f"gap median {float(np.median(gaps)) if gaps else float('nan'):+.3f} "
                 f"({sum(q > 0 for q in gaps)}/{len(gaps)} positive), rows written {fresh_written}")

    radii = {} if r_saved[0] is None else {
        f"r_l{l + 1}": r_saved[l][..., 0].cpu().numpy().astype(np.float32) for l in range(2)}
    if radii:
        np.savez(out / "radii.npz", seeds=np.asarray(seeds), **radii)
    prov = {"run_id": EXPERIMENT, **B.git_state(), "cap_arm": cap_arm,
            "cap_layers": [f"W{l + 1}" for l in layers], "radius_task": RADIUS_TASK,
            "cap_from_task": CAP_FROM,
            "cap_rule": "after every Adam update from the cap task on: n = ||w_i|| (float32); "
                        "rows with n > r_i become w_i * r_i / n; others untouched; W1 then W2; " +
                        ("b1/b2 multiplied by corresponding W-only row coefficient; " if cap_arm == "cap12b" else "biases untouched; ") +
                        "W3/b3, Adam moments, step counter, RNG untouched",
            "radii_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in radii.items()},
            "rows_written_total": {k: totals[k] for k in ("rows_w1", "rows_w2")},
            "violations_total": {k: totals[k] for k in ("viol_w1", "viol_w2")},
            "fresh_cap": "off: radius reset to +inf before the fresh control (spec 2.5)",
            "fresh_rows_written": fresh_written,
            "arm": arm, "cond": cond, "seeds": seeds, "R": R, "lr": lr, "n_tasks": n_tasks,
            "steps_per_task": C.STEPS_PER_TASK, "steps_hard": steps_hard,
            "steps_easy": steps_easy, "batch": C.BATCH, "hidden": C.HIDDEN,
            "layer_shapes": C.layer_shapes(arm), "n_params": C.n_params(arm),
            "dims": list(C.DIMS), "n_classes": C.N_CLASSES, "hard_classes": C.HARD_CLASSES,
            "per_class_train": C.PER_CLASS_TRAIN, "classes_used": C.CLASSES_USED,
            "task_parity": "task 1 hard, alternating (the paper's task 0 hard)",
            "optimizer": "adam", "weight_decay": 0.0, "data_sha256": cifar.sha256,
            "std": {"mean": C.STD_MEAN, "std": C.STD_STD, "planes": "R,G,B x 1024"},
            "class_sha256": {str(s): hashlib.sha256(
                np.asarray(C.seed_classes(s), dtype=np.int64).tobytes()).hexdigest() for s in seeds},
            "rng_roles": ["c51_classes", "c51_batch", "init"],
            "fresh_control": {"task": last_hard, "n": len(fresh_rows)} if fresh else None,
            "engine": "native run() of cifar5p1_mlp_0920 via inherited cap host; activation=" + activation + "; std; "
                      "stacked baddbmm, autograd, elementwise Adam"
                      + (", one step captured as a CUDA graph" if use_graph else ", eager"),
            "activation": activation, "native_dphi": "autograd of host phi, including zero convention",
            "bias_scaling": cap_arm == "cap12b", "s_correction": 0,
            "host_module_sha256": sha256_file(Path(C.__file__)),
            "engine_sha256": sha256_file(Path(__file__)),
            "device": str(device), "torch": torch.__version__, "threads": torch.get_num_threads(),
            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "step_ms_last_task": step_ms, "wall_clock_s": time.time() - t_start,
            "divergences": diverged, "spec": SPEC}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    progress(f"wrote {out}/per_task.csv ({len(rows)} rows, {(time.time() - t_start) / 60:.1f} min)")
    return prov

class RCEngine(I.Engine):
    """Host Adam, initialization, labels/batches; activation and post-step cap only."""
    def __init__(self,seeds,X,activation='ELU',arm='ref',state=None,graph=True):
        if arm not in ('ref','cap12'):
            raise ValueError('RC only registered ref/cap12')
        self.activation=activation; self.policy_arm=arm
        self.native=native_act(activation)
        self.radii=[torch.full((len(seeds),100,1),math.inf,device=X.device) for _ in range(2)]
        self.hits=torch.zeros(len(seeds),2,100,dtype=torch.long,device=X.device)
        self.removed=torch.zeros(len(seeds),2,100,dtype=torch.float64,device=X.device)
        self.viol=torch.zeros(len(seeds),2,dtype=torch.long,device=X.device)
        self.excess=torch.full((len(seeds),2),-math.inf,dtype=torch.float64,device=X.device)
        self.u_sum=torch.zeros(len(seeds),2,dtype=torch.float64,device=X.device)
        self.u_steps=torch.zeros((),dtype=torch.long,device=X.device)
        self.norm_before=torch.zeros(len(seeds),2,100,dtype=torch.float64,device=X.device)
        self.norm_after=torch.zeros_like(self.norm_before)
        self.radius_anchor=None
        super().__init__(seeds,X,state=state,graph=graph)

    def forward(self,xb,ids):
        return B.forward(self.P,xb,self.native,train=True)

    def mutable_tensors(self):
        return super().mutable_tensors()+[self.hits,self.removed,self.viol,self.excess,self.u_sum,self.u_steps,self.norm_before,self.norm_after]

    def state(self):
        st=super().state()
        st['cap1018']=I.cpu(dict(activation=self.activation,arm=self.policy_arm,
            radii=self.radii,hits=self.hits,removed=self.removed,viol=self.viol,
            excess=self.excess,u_sum=self.u_sum,u_steps=self.u_steps,norm_before=self.norm_before,norm_after=self.norm_after,radius_anchor=self.radius_anchor))
        return st

    def restore(self,st):
        super().restore(st)
        if 'cap1018' in st:
            c=st['cap1018']
            if (c['activation'],c['arm'])!=(self.activation,self.policy_arm):
                raise ValueError('checkpoint configuration mismatch')
            for a,b in zip(self.radii,c['radii']):a.copy_(b)
            for key in ('hits','removed','viol','excess','u_sum','u_steps','norm_before','norm_after'):getattr(self,key).copy_(c[key])
            self.radius_anchor=c['radius_anchor']

    def step(self):
        before=[self.P[i].detach().clone() for i in (0,2)]
        super().step()
        with torch.no_grad():
            for l in range(2):
                W=self.P[2*l];n64=W.double().norm(dim=2,keepdim=True)
                if self.policy_arm=='cap12':
                    n=row_norms(W); over=n>self.radii[l]
                    self.hits[:,l].add_(over[...,0])
                    self.removed[:,l].add_((n.double()-self.radii[l].double()).clamp(min=0)[...,0])
                    count=torch.zeros(self.R,dtype=torch.long,device=self.device)
                    removed=torch.zeros(self.R,dtype=torch.float64,device=self.device)
                    project_rows_(W,self.radii[l],count,removed)
                    monitor_(W,self.radii[l],n,n64,self.viol[:,l],self.excess[:,l])
                self.norm_before[:,l].copy_(torch.maximum(self.norm_before[:,l],n64[...,0]))
                self.norm_after[:,l].copy_(torch.maximum(self.norm_after[:,l],W.double().norm(dim=2)))
                displacement=(W.double()-before[l].double()).flatten(1).norm(dim=1)
                denominator=before[l].double().flatten(1).norm(dim=1)
                self.u_sum[:,l].add_(torch.where(denominator>0,displacement/denominator,torch.nan).square())
            self.u_steps.add_(1)

    def diagnostic(self):
        units=unit_diagnostics(self.P,self.X,self.native,self.activation)
        with torch.no_grad():
            logits=self.forward(self.X,torch.arange(B.N_IMAGES,device=self.device)[None,:].expand(self.R,-1))[-1]
        rows=[]
        for r,s in enumerate(self.seeds):
            row=dict(seed=s,slot=r,task=self.task,update=int(self.step_t),
                memo_acc=float((logits[r].argmax(-1)==self.Y[r]).float().mean()),
                memo_ce=float(F.cross_entropy(logits[r],self.Y[r])))
            for l in range(2):
                row.update({f'U_l{l+1}':float((self.u_sum[r,l]/self.u_steps).sqrt()),
                    f'projection_hits_l{l+1}':int(self.hits[r,l].sum()),
                    f'projection_removed_l{l+1}':float(self.removed[r,l].sum()),
                    f'projection_violations_l{l+1}':int(self.viol[r,l])})
            rows.append(row)
        units.update(cap_hits=self.hits.cpu().numpy(),cap_removed=self.removed.cpu().numpy(),projection_before_norm_max=self.norm_before.cpu().numpy(),projection_after_norm_max=self.norm_after.cpu().numpy())
        for l in range(2):units[f'radius_l{l+1}']=self.radii[l].cpu().numpy()
        return rows,units

    def train_task(self,task,epochs=400,**kwargs):
        if not self.in_task:
            for q in (self.hits,self.removed,self.viol,self.u_sum,self.u_steps,self.norm_before,self.norm_after):q.zero_()
            self.excess.fill_(-math.inf)
        result=super().train_task(task,epochs,**kwargs)
        if result is not None and task==1:
            with torch.no_grad():
                for l in range(2):self.radii[l].copy_(row_norms(self.P[2*l]))
            self.radius_anchor=I.tree_hash([self.P[i] for i in (0,2)])
        return result


def source_hashes():
    files=[f'src/{name}.py' for name in (RUN,'cap_cifar5p1_1007','cap_cifar_ee_0920',
        'cifar5p1_mlp_0920','cifar_interventions_0920','rlcifar_mlp_battle_0918',
        'pmnist_0905','pmnist_rlcifar_0907','pmnist_rlmnist_0906')]
    files += ['analysis/cifar_ledger_0920/replay.py','analysis/cifar_ledger_0920/ledger.py','analysis/resp_cifar_ee_0920/stats.py',SPEC,COMMON]
    files += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'analysis'/RUN).glob('*.py'))]
    return {name:sha256_file(ROOT/name) for name in sorted(set(files))}


def integration_issues(path):
    if path is None or not Path(path).is_file():return ['BLOCKED_G0: integration audit missing']
    try:
        audit=json.loads(Path(path).read_text()); issues=[]
        for k in ('all_pass','main_ancestor_verified','host_reproduction_pass','mutation_checks_pass'):
            if audit.get(k) is not True:issues.append('BLOCKED_G0: '+k)
        who=audit.get('independent_auditor','')
        if not who or '/root' not in who or '/run2_cap' in who:issues.append('BLOCKED_G0: independent auditor identity')
        if audit.get('schema_version')!=1:issues.append('BLOCKED_G0: schema')
        needed=['src/cifar5p1_mlp_0920.py','src/rlcifar_mlp_battle_0918.py','src/cap_cifar5p1_1007.py','src/pmnist_0905.py']
        for p in needed:
            if audit.get('source_sha256',{}).get(p)!=sha256_file(ROOT/p):issues.append('BLOCKED_G0: source '+p)
        commit=audit.get('audited_commit','')
        if not commit or subprocess.run(['git','merge-base','--is-ancestor',commit,'main'],cwd=ROOT,capture_output=True).returncode:
            issues.append('BLOCKED_G0: audited commit main ancestry')
        return issues
    except Exception as e:return ['BLOCKED_G0: invalid audit '+str(e)]


def preflight(out,audit=None,checks=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    issues=[]
    data_sha256={}
    if not torch.cuda.is_available():issues.append('BLOCKED_CUDA: native CUDA device unavailable')
    issues += integration_issues(audit)
    checkpath=Path(checks) if checks else OUT_ROOT/'checks.json'
    if not checkpath.exists():issues.append('BLOCKED_CHECKS: checks.json missing')
    else:
        chk=json.loads(checkpath.read_text())
        if chk.get('all_pass') is not True:issues.append('BLOCKED_CHECKS: full CPU+CUDA S suite incomplete')
        if chk.get('source_sha256')!=source_hashes():issues.append('BLOCKED_CHECKS: stale source hashes')
    dirty=subprocess.run(['git','status','--porcelain','--','src','analysis','specs'],cwd=ROOT,text=True,capture_output=True).stdout
    if dirty.strip():issues.append('BLOCKED_DIRTY_SOURCES: commit code/specs/check scripts first')
    for label,p in [('F51',Path('/home/issan/Projects/claude/proj_004_drift/data/cifar100')/C.ARCHIVE),
                    ('RC',I.DATA/'cifar-10-python.tar.gz')]:
        if not p.is_file():issues.append('BLOCKED_DATA: '+label+' '+str(p))
        else:data_sha256[label]={p.name:sha256_file(p)}
    obj=dict(run_id=RUN,status='READY' if not issues else ('BLOCKED_CUDA' if not torch.cuda.is_available() else 'BLOCKED'),
        blockers=issues,configuration=dict(F51=dict(tasks=30,steps=780,batch=32,lr=1e-4),RC=dict(tasks=50,epochs=400,updates=30000,batch=16,lr=1e-3),
            eval_seeds=EVAL_SEEDS,calibration_seeds=CALIB_SEEDS,R=10,activation_order=['ELU','ReLU'],historical_reuse='disabled: rerun both historical cells'),
        source_sha256=source_hashes(),data_sha256=data_sha256,git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        torch=torch.__version__,cuda_build=torch.version.cuda,cuda_available=torch.cuda.is_available(),argv=sys.argv,
        integration_audit_sha256=sha256_file(Path(audit)) if audit and Path(audit).exists() else None,
        checks_sha256=sha256_file(checkpath) if checkpath.exists() else None,
        timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    put(out/'preflight.json',obj)
    put(out/'verdict.json',dict(run_id=RUN,status=obj['status'],main_label=None,blockers=issues))
    return obj


def provenance(out,box,activation,arm,seeds,data):
    obj=dict(run_id=RUN,box=box,activation=activation,arm=arm,seeds=seeds,R=10,
        source_sha256=source_hashes(),git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        data_sha256=data,argv=sys.argv,environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
        cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),TF32=torch.backends.cuda.matmul.allow_tf32),
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        reused_from=None,old_cell=(box,activation) in (('RC','ELU'),('F51','ReLU')),
        interpretation='Old cells are reruns of known results, not independent predicted successes.')
    put(Path(out)/'provenance_start.json',obj)
    return obj


def execute_rc(out,activation,arm,seeds,expected_data=None):
    X,cifar=I.make_inputs(seeds,torch.device('cuda'));out.mkdir(parents=True,exist_ok=True)
    if expected_data is not None and cifar.sha256!=expected_data:raise ValueError('CHECK_FAILED: RC data changed after preflight')
    prov=provenance(out,'RC',activation,arm,seeds,cifar.sha256)
    put(out/'input_manifest.json',dict(input_hash=I.tree_hash(X),subset_order={str(s):B.RC.subset_idx(s).tolist() for s in seeds}))
    engine=RCEngine(seeds,X,activation,arm)
    initial=engine.state();I.save_pt(out/'initial.pt',initial)
    rows=[]
    for task in range(1,51):
        rr,unit=engine.train_task(task,400)
        for r in rr:r.update(box='RC',activation=activation,cap_arm=arm)
        rows.extend(rr);H.write_csv(out/'per_task.csv',rows)
        save_npz(out/f'units_t{task:02d}.npz',unit)
        I.save_pt(out/f't{task:02d}.pt',engine.state())
        if task==1:
            save_npz(out/'radii.npz',{f'r_l{l+1}':q.cpu().numpy() for l,q in enumerate(engine.radii)})
            I.save_pt(out/'radius_anchor.pt',engine.state())
        if not all(r['finite'] for r in rr):
            put(out/'status.json',dict(status='DIVERGED',task=task));return
        if bool(engine.viol.any()):
            put(out/'status.json',dict(status='CHECK_FAILED',reason='row radius bound'));return
        print(f'{activation} RC {arm} {task}/50',flush=True)
    put(out/'provenance.json',dict(**prov,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),s_correction=0))
    seal_completed_arm(out,seeds,50)


def execute_f51(out,activation,arm,seeds,schedule=None,expected_data=None):
    out.mkdir(parents=True,exist_ok=True)
    C.DATA_DIR=Path('/home/issan/Projects/claude/proj_004_drift/data/cifar100')
    cifar=C.Cifar100()
    if expected_data is not None and cifar.sha256!=expected_data:raise ValueError('CHECK_FAILED: F51 data changed after preflight')
    prov=provenance(out,'F51',activation,arm,seeds,cifar.sha256)
    init_hash=I.tree_hash([C.init_params('R',s,torch.device('cpu')) for s in seeds])
    def observer(t,state):
        I.save_pt(out/f't{t:02d}.pt',dict(P=state['P'],m=state['m'],v=state['v'],task=t))
    run_f51(arm,seeds,30,torch.device('cuda'),out,cifar=cifar,activation=activation,schedule=schedule,observer=observer)
    # Preserve host provenance and augment without changing the host's CSV definitions.
    host=json.loads((out/'provenance.json').read_text())
    put(out/'provenance.json',dict(**host,registration=prov,initial_state_hash=init_hash,
        schedule_hash=None if schedule is None else I.tree_hash(schedule)))
    rows=list(csv.DictReader((out/'per_task.csv').open()))
    good=len(rows)==300 and all(np.isfinite(float(r.get('online_acc','nan'))) for r in rows)
    cr=list(csv.DictReader((out/'cap_task.csv').open()))
    cap_ok=all(int(r[f'viol_w{l}'])==0 for r in cr for l in (1,2))
    status='DIVERGED' if not good else 'CHECK_FAILED' if not cap_ok else 'COMPLETED'
    if status=='COMPLETED':seal_completed_arm(out,seeds,30)
    else:put(out/'status.json',dict(status=status,tasks=30,seeds=seeds))


def write_schedule(base):
    base=Path(base); paths={a:base/'calibration'/a/'cap_task.csv' for a in ('ref','cap12')}
    ref,cap=[list(csv.DictReader(paths[a].open())) for a in ('ref','cap12')]
    evidence={a:validate_calibration_arm(p.parent,a) for a,p in paths.items()}
    obj=dict(calibration_seeds=CALIB_SEEDS,source_sha256=source_hashes(),calibration_evidence=evidence,q=schedule_from_rows(ref,cap),
        input_csv={a:dict(path=str(p.resolve()),sha256=sha256_file(p)) for a,p in paths.items()},
        definition='median_seed RMS_k(||real ΔW||_F/||W_before||_F) cap/ref; task1=1; NO CLIP',
        timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    validate_schedule(obj);put(base/'lr_schedule.json',obj)
    H.write_csv(base/'lr_schedule.csv',[dict(task=int(t),q_l1=v[0],q_l2=v[1]) for t,v in obj['q'].items()])
    return obj


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['preflight','calibrate','run','analyze'])
    ap.add_argument('--out',type=Path,default=OUT_ROOT)
    ap.add_argument('--integration-audit',type=Path,default=Path('/tmp/issa_six_specs_1018/integration_audit.json'))
    ap.add_argument('--checks',type=Path,default=OUT_ROOT/'checks.json')
    args=ap.parse_args(argv)
    if args.stage=='analyze':
        from analysis.cap_act_box_2x2_1018.verdict import analyze
        analyze(args.out);return 0
    report=preflight(args.out,args.integration_audit,args.checks)
    if report['blockers']:
        print(json.dumps(report,indent=2));return 2
    if args.stage=='preflight':print('READY');return 0
    # Native GPU policy; do not reduce precision, seed slots, tasks or epochs.
    H.setup('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    args.out.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/SPEC,args.out/'spec.md')
    for spec in (SPEC,COMMON):shutil.copy2(ROOT/spec,args.out/Path(spec).name)
    with exclusive():
        if args.stage=='calibrate':
            for arm in ('ref','cap12'):
                dest=args.out/'calibration'/arm
                if dest.exists():raise SystemExit('Output exists: do not overwrite partial or complete arms')
                execute_f51(dest,'ReLU',arm,CALIB_SEEDS,expected_data=report['data_sha256']['F51'])
            write_schedule(args.out)
        else:
            schedule_path=args.out/'lr_schedule.json'
            if not schedule_path.exists():raise SystemExit('Run calibration first: schedule must predate evaluation')
            schedule_obj=json.loads(schedule_path.read_text());schedule=validate_schedule(schedule_obj)
            for arm in ('ref','cap12'):
                calibration_prov=json.loads(Path(schedule_obj['calibration_evidence'][arm]['provenance.json']['path']).read_text())
                if calibration_prov['data_sha256']!=report['data_sha256']['F51']:
                    raise ValueError('CHECK_FAILED: calibration data differs from evaluation preflight')
            if (args.out/'evaluation').exists() or (args.out/'evaluation_admission.json').exists():
                raise SystemExit('Evaluation output/admission exists; explicit recovery required, never silently overwrite')
            put(args.out/'evaluation_preflight.json',report)
            put(args.out/'evaluation_admission.json',dict(preflight_sha256=sha256_file(args.out/'evaluation_preflight.json'),schedule_sha256=sha256_file(schedule_path)))
            for box in ('RC','F51'):
                for activation in ('ELU','ReLU'):
                    for arm in ('ref','cap12'):
                        dest=args.out/'evaluation'/box/activation/arm
                        if dest.exists():raise SystemExit('Output exists: explicit recovery required, never silently overwrite')
                        if box=='RC':execute_rc(dest,activation,arm,EVAL_SEEDS,expected_data=report['data_sha256']['RC'])
                        else:execute_f51(dest,activation,arm,EVAL_SEEDS,expected_data=report['data_sha256']['F51'])
            for arm in ('cap12b','lr_only'):
                execute_f51(args.out/'evaluation/F51/ReLU'/arm,'ReLU',arm,EVAL_SEEDS,schedule if arm=='lr_only' else None,expected_data=report['data_sha256']['F51'])
    return 0


if __name__=='__main__':raise SystemExit(main())

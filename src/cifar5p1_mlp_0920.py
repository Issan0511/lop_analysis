#!/usr/bin/env python3
"""cifar5p1_mlp_0920 -- the 5+1 CIFAR box (Kumar, Marklund & Van Roy 2024 §4.2) on an MLP.

    python3 src/cifar5p1_mlp_0920.py run --arm R --seeds 100-102 --cond std --lr 1e-4
    python3 src/cifar5p1_mlp_0920.py run --arm SNA --seeds 0-9 --cond std    # 10 runs stacked

The problem is the paper's, verbatim from §4.2 and Table 1 (Appendix A.1.2):

    "every even task is 'hard' while every odd task is 'easy.'  Data is drawn from the
     CIFAR 100 dataset, and a hard task is characterized by seeing (image, label) data
     pairs of 5 CIFAR 100 classes, whereas in an easy task, data from only a single
     class arrives.  Each hard task consists of 2500 data pairs (500 from each class),
     while each easy task consists of 500 data pairs from a single class. ...  Each task
     has a duration of 780 timesteps which corresponds to 10 epochs through the hard
     task datasets when using a batch size of 32. ...  we measure agents' performance
     specifically on the hard tasks"

    "In both 5+1 CIFAR and Continual ImageNet, each individual class does not occur in
     more than one task."

    "All networks have a fully connected output layer at the end with ... 100 outputs
     for 5+1 CIFAR"

30 tasks (15 hard, 15 easy), hard first.  The head carries the true CIFAR-100 label, and
because a class never comes back, each task's five output units have never been trained
up before -- the paper's design does what a head reset does elsewhere, without a reset.
The plasticity metric is the paper's average online task accuracy: the mean over the
task's 780 batches of the accuracy of the very forward pass the loss came from (§4.1,
"a_j is the average accuracy on the jth batch of samples"), scored on the hard tasks.

What differs from the original is the network.  Kumar et al. and Lillo & Cheney both run
a CNN on this problem; this file runs the MLP of the sibling boxes (3072-100-100-100 --
their MLP's two hidden widths, their 5+1 output size) because that is where our 13 arms
and the engine that stacks runs live.  Whether ReLU loses plasticity at all in the MLP
version is not something either paper answers: the pilot decides that before the 13-arm
run is allowed to start (spec_cifar5p1_mlp_0920.md §3).

The activations, the stacked forward and the Adam step are imported from
`rlcifar_mlp_battle_0918`, which is not modified -- the arms have to be the same code, not
a copy that can drift.  Nothing in pmnist_0905 / pmnist_rlcifar_0907 is modified either.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H                    # streams, init, eff_rank, csv, setup
from src import pmnist_rlcifar_0907 as RC           # the safe unpickler for the CIFAR pickles
from src import rlcifar_mlp_battle_0918 as B        # the 13 arms, stacked forward, git_state

EXPERIMENT = "cifar5p1_mlp_0920"
OUT_ROOT = H.REPO / "results" / EXPERIMENT
DATA_DIR = H.REPO / "data" / "cifar100"
ARCHIVE = "cifar-100-python.tar.gz"

DIMS = (3072, 100, 100, 100)                        # 5+1 CIFAR's head is 100-wide
N_CLASSES = 100
BATCH = 32                                          # Table 1
STEPS_PER_TASK = 780                                # Table 1 ("10 epochs" on 2500 / 32)
N_TASKS = 30                                        # 15 hard, 15 easy
HARD_CLASSES = 5
PER_CLASS_TRAIN = 500                               # CIFAR-100 train holds exactly 500/class
PER_CLASS_TEST = 100
CLASSES_USED = (N_TASKS // 2) * (HARD_CLASSES + 1)  # 90 of the 100, none repeated
LR = 1e-4                                           # Kumar Table 4: Baseline + Adam on 5+1 CIFAR
CONDS = ("raw", "std")
HIDDEN = 100                                        # both papers' MLP hidden width
# Three arms beyond the 0918 thirteen, all from the literature on this very box (spec
# addendum 1): CR and DF emit two numbers per preactivation, so the next layer's fan-in
# doubles; LK07 is an ordinary leaky whose slope sits in Lillo & Cheney's reported
# "Goldilocks zone" of 0.6-0.9, which the 0918 ladder (0.01 / 0.1 / 0.3) misses entirely.
WIDE = ("CR", "DF")
ARM_ORDER = B.ARM_ORDER + ("LK07", "CR", "DF")
DEAD_TOL = H.DEAD_TOL

# CIFAR-100 channel statistics (the widely used values; CIFAR-10's are a different pair,
# so `rlcifar_mlp_battle_0918.standardize` is not reused).  Checked against the archive
# itself in S-data.
STD_MEAN = (0.5071, 0.4865, 0.4409)
STD_STD = (0.2673, 0.2564, 0.2762)


# --------------------------------------------------------------------------
# the three added arms (spec addendum 1)
# --------------------------------------------------------------------------

class ConcatReLU(B.Act):
    """Concatenated ReLU (Shang et al. 2016; Kumar et al.'s architectural baseline).

    phi(z) = [relu(z), relu(-z)]: one of the two branches always passes the gradient, so
    a unit cannot go dead in the usual sense -- which is why `dead_frac` is 0 by
    construction for this arm and says nothing.  Read `zeroout` (output channels that are
    identically zero over the task's data), `eff_rank` and `w_norm` instead.
    """
    name = "CR"

    def phi(self, z, layer=0, train=False):
        return torch.cat([z.clamp(min=0.0), (-z).clamp(min=0.0)], dim=-1)

    def dphi(self, z, layer=0):
        # norm of d[relu(z), relu(-z)]/dz: exactly 1 away from the kink, 0 at z == 0
        return (z != 0).to(z.dtype)


class DeepFourier(B.Act):
    """Deep Fourier features (Lewandowski, Schuurmans & Machado 2024, arXiv:2410.20634):

        "we propose deep Fourier features, which are the concatenation of a sine and
         cosine in every layer" -- Fourier(z) = [sin(z), cos(z)]

    Lillo & Cheney's Table 2 (v2) puts this at 72.29% on 5+1 CIFAR, the best of their 17
    activations by 15 points, which is why it is here.  |d[sin,cos]/dz| = 1 everywhere, so
    `dead_frac` and `mob` are constant by construction for this arm too.
    """
    name = "DF"

    def phi(self, z, layer=0, train=False):
        return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)

    def dphi(self, z, layer=0):
        return torch.ones_like(z)        # sqrt(cos^2 + sin^2) = 1


def make_act(arm: str, hidden: int = HIDDEN, c=0.6, beta=0.01, lo=0.005, hi=3.0) -> B.Act:
    """The 13 arms of 0918 plus the three added ones, with the Snake family's per-unit
    state sized to this box's hidden width rather than 0918's."""
    if arm == "CR":
        return ConcatReLU()
    if arm == "DF":
        return DeepFourier()
    if arm == "LK07":
        return B.Leaky("LK07", 0.7)
    kinds = {"SNA": "snake", "KKA": "kk", "KKA23": "kk23", "KKT1": "kkt1"}
    if arm in kinds:
        return B.SnakeFamily(arm, kinds[arm], c, beta, lo, hi, widths=(hidden, hidden))
    if arm == "RSL" and hidden != B.DIMS[1]:
        # RandSmoothLeaky sizes its noise buffers from the 0918 module's DIMS
        raise SystemExit("RSL is only wired for hidden=100")
    return B.make_act(arm, c, beta, lo, hi)


def layer_shapes(arm: str, hidden: int = HIDDEN) -> list[tuple[int, int]]:
    """(out_features, in_features) of the three affine maps."""
    k = 2 if arm in WIDE else 1
    return [(hidden, DIMS[0]), (hidden, k * hidden), (N_CLASSES, k * hidden)]


def init_params(arm: str, seed: int, device, hidden: int = HIDDEN):
    """PyTorch nn.Linear default init, U(+-1/sqrt(fan_in)), from the host's `init` stream.

    Identical draws in the identical order to `H.init_params(seed, device, DIMS)` for a
    pointwise arm at hidden=100 (checked bit for bit in S-init); the width-doubling arms
    differ only in that layers 2 and 3 have fan-in 2*hidden, which moves both the shape
    and the bound.
    """
    import math
    g = H.stream("init", seed)
    params = []
    for out_f, in_f in layer_shapes(arm, hidden):
        bound = 1.0 / math.sqrt(in_f)
        W = (torch.rand((out_f, in_f), generator=g, dtype=torch.float32) * 2 - 1) * bound
        b = (torch.rand((out_f,), generator=g, dtype=torch.float32) * 2 - 1) * bound
        params += [W.to(device), b.to(device)]
    return params


def n_params(arm: str, hidden: int = HIDDEN) -> int:
    return sum(o * i + o for o, i in layer_shapes(arm, hidden))


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

class Cifar100:
    """CIFAR-100 train and test, flattened to 3072, in file order.

    The archive's pickles hold a dict of one uint8 ndarray plus plain lists, and the
    unpickler from the CIFAR-10 box (which allows nothing but numpy's reconstructors)
    is reused so a bad download raises instead of running.
    """
    MEMBERS = {"train": "cifar-100-python/train", "test": "cifar-100-python/test"}

    def __init__(self, path: Path | None = None):
        path = path or DATA_DIR / ARCHIVE
        self.sha256 = {ARCHIVE: hashlib.sha256(path.read_bytes()).hexdigest()}
        import tarfile
        with tarfile.open(path, "r:gz") as tf:
            for split, name in self.MEMBERS.items():
                fh = tf.extractfile(name)
                if fh is None:
                    raise SystemExit(f"{ARCHIVE}: member {name} missing")
                d = RC._SafeUnpickler(io.BytesIO(fh.read()), encoding="bytes").load()
                setattr(self, f"{split}_u8", torch.from_numpy(d[b"data"]))
                setattr(self, f"{split}_y", torch.tensor(d[b"fine_labels"], dtype=torch.long))
        for split, n in (("train", PER_CLASS_TRAIN), ("test", PER_CLASS_TEST)):
            u8, y = getattr(self, f"{split}_u8"), getattr(self, f"{split}_y")
            if tuple(u8.shape) != (n * N_CLASSES, 3072) or len(y) != n * N_CLASSES:
                raise SystemExit(f"{ARCHIVE}: {split} is {tuple(u8.shape)} / {len(y)}")
            if int(torch.bincount(y, minlength=N_CLASSES).min()) != n:
                raise SystemExit(f"{ARCHIVE}: {split} is not {n} images per class")

    def inputs(self, split: str, cond: str, device) -> torch.Tensor:
        """The whole split as float32 on the device: 614 MB for train, shared by every slot.

        `raw` is u8/255 as in the sibling boxes; `std` additionally subtracts the channel
        mean and divides by the channel standard deviation, per 1024-long plane.
        """
        x = getattr(self, f"{split}_u8").to(device=device, dtype=torch.float32).div_(255.0)
        if cond == "raw":
            return x
        if cond != "std":
            raise SystemExit(f"unknown cond {cond!r}; known: {','.join(CONDS)}")
        mean = torch.tensor(STD_MEAN, device=device).repeat_interleave(1024)
        std = torch.tensor(STD_STD, device=device).repeat_interleave(1024)
        return (x - mean) / std


def class_rows(labels: torch.Tensor) -> list[torch.Tensor]:
    """Row indices of each class, in file order (so the image set of a class is fixed)."""
    return [torch.nonzero(labels == c).flatten() for c in range(N_CLASSES)]


# --------------------------------------------------------------------------
# the task sequence (§4.2: alternating difficulty, no class twice)
# --------------------------------------------------------------------------

def seed_classes(seed: int) -> list[int]:
    """The 90 classes this seed uses, in the order the tasks consume them.

    Own stream, drawn once per seed, so the sequence is a property of the seed and is
    identical across arms, conditions and learning rates.
    """
    g = H.stream("c51_classes", seed)
    return torch.randperm(N_CLASSES, generator=g)[:CLASSES_USED].tolist()


def task_plan(seed: int) -> list[tuple[bool, list[int]]]:
    """[(hard?, class ids)] for tasks 1..30.  Task 1 is hard (the paper's task 0)."""
    cls, plan, i = seed_classes(seed), [], 0
    for t in range(1, N_TASKS + 1):
        hard = t % 2 == 1
        k = HARD_CLASSES if hard else 1
        plan.append((hard, cls[i:i + k]))
        i += k
    assert i == CLASSES_USED, i
    return plan


def batch_indices(g: torch.Generator, rows: torch.Tensor) -> torch.Tensor:
    """(780, 32) row indices for one task: epochs of a fresh permutation, cut into batches.

    780 x 32 = 24,960 draws is 9.98 epochs of a hard task and 49.9 of an easy one, which
    is the paper's "780 timesteps ~ 10 epochs on the hard task datasets".  Batches are cut
    from the concatenation of whole permutations, so within the part that is used every
    epoch covers the task's data exactly once.
    """
    need = STEPS_PER_TASK * BATCH
    parts, have = [], 0
    while have < need:
        parts.append(rows[torch.randperm(len(rows), generator=g)])
        have += len(rows)
    return torch.cat(parts)[:need].view(STEPS_PER_TASK, BATCH)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate(P, X, Y, act: B.Act, live: list[int]) -> list[dict]:
    """The sibling boxes' metric expressions, on each live slot's own task data.

    X is (R, N, 3072) and Y is (R, N): within a task every slot has the same N, because
    the hard/easy alternation is a property of the task index, not of the seed.
    """
    R = X.shape[0]
    z1, a1, z2, a2, logits = B.forward(P, X, act, train=False)
    acc = (logits.argmax(-1) == Y).float().mean(1)
    rows = [{"acc": float(acc[r])} if r in live else {} for r in range(R)]
    for li, (tag, z, a) in enumerate((("l1", z1, a1), ("l2", z2, a2))):
        d = act.dphi(z, li)
        for r in live:
            zr, ar_, dr = z[r], a[r], d[r]
            rows[r].update({
                f"dead_frac_{tag}": float((dr.abs().amax(0) < DEAD_TOL).float().mean()),
                f"zeroout_{tag}": float((ar_.abs().amax(0) == 0).float().mean()),
                f"zbar_{tag}": float(zr.mean(0).median()),
                f"zsd_{tag}": float(zr.std(0).median()),
                f"zbar_min_{tag}": float(zr.mean(0).amin()),
                f"mob_{tag}": float(dr.mean(0).median()),
                f"eff_rank_{tag}": H.eff_rank(ar_)})
    for i, tag in enumerate(("l1", "l2", "l3")):
        for r in live:
            rows[r][f"w_norm_{tag}"] = float(P[2 * i][r].norm(dim=1).median())
    for r in live:
        rows[r].update(act.stats(r))
    return rows


@torch.no_grad()
def accuracy(P, X, Y, act: B.Act) -> torch.Tensor:
    """(R,) accuracy of the stacked nets on stacked data, in eval mode."""
    return (B.forward(P, X, act, train=False)[4].argmax(-1) == Y).float().mean(1)


# --------------------------------------------------------------------------
# the stacked run
# --------------------------------------------------------------------------

def run(arm: str, seeds: list[int], cond: str, n_tasks: int, device, out: Path,
        lr: float = LR, c: float = 0.6, beta: float = 0.01, lo: float = 0.005,
        hi: float = 3.0, cifar: Cifar100 | None = None, fresh: bool = True,
        graph: bool = True, progress=None, debug: dict | None = None,
        nan_slot: int | None = None, hidden: int = HIDDEN) -> dict:
    """Train one arm's R = len(seeds) runs in lockstep over the 30-task sequence.

    Every slot sees the same task shape at the same time (hard tasks are odd for every
    seed) but its own 90 classes, its own image order and its own initial weights.

    `fresh`: after the sequence, reinitialise the stacked net in place and retrain it on
    the last hard task's very batches, to separate "the net adapts worse than a new one"
    from "the task got harder" (the control the ViT box needed; spec §3.2).

    `nan_slot` is a check-only hook (S-diverge, never used by a real run): it poisons that
    slot's W1 at init, so the slot diverges on its first step while the others must carry
    on bit for bit.
    """
    t_start = time.time()
    progress = progress or (lambda m: print(m, flush=True))
    act = make_act(arm, hidden, c, beta, lo, hi)
    R = len(seeds)
    cifar = cifar or Cifar100()
    X_all = cifar.inputs("train", cond, device)
    Y_all = cifar.train_y.to(device)
    X_test = cifar.inputs("test", cond, device)
    Y_test = cifar.test_y.to(device)
    tr_rows, te_rows = class_rows(cifar.train_y), class_rows(cifar.test_y)
    plans = [task_plan(s) for s in seeds]
    g_batch = {s: H.stream("c51_batch", s) for s in seeds}

    init = [q.detach() for s in seeds for q in init_params(arm, s, device, hidden)]
    P = [torch.stack(init[i::6]).contiguous() for i in range(6)]
    if nan_slot is not None:
        P[0][nan_slot] = float("nan")
    P = [q.requires_grad_(True) for q in P]
    P0 = [q.detach().clone() for q in P]
    act.init_state(R, device, key=f"{arm}|{seeds}|{cond}")
    act0 = act.state()
    adam_m = [torch.zeros_like(q) for q in P]
    adam_v = [torch.zeros_like(q) for q in P]
    if debug is not None:
        debug["init"] = [q.detach().cpu().clone() for q in P]

    # ---- one training step on static tensors (eager, or captured once and replayed).
    # The task lives entirely in which global row indices are written into static_idx,
    # so the graph never has to be recaptured: X_all and Y_all are the whole dataset.
    b1, b2, eps = 0.9, 0.999, 1e-8
    static_idx = torch.zeros(R, BATCH, dtype=torch.long, device=device)
    inv_c1 = torch.zeros((), device=device)
    inv_c2 = torch.zeros((), device=device)
    step_t = torch.zeros((), dtype=torch.long, device=device)
    acc_sum = torch.zeros(R, device=device)
    bad_step = torch.full((R,), -1, dtype=torch.long, device=device)
    last_hit = torch.zeros(R, device=device)

    def step():
        xb, yb = X_all[static_idx], Y_all[static_idx]            # gathers, no arithmetic
        z1, a1, z2, a2, z3 = B.forward(P, xb, act, train=True)
        lossv = F.cross_entropy(z3.reshape(-1, N_CLASSES), yb.reshape(-1),
                                reduction="none").view(R, BATCH).mean(1)
        hit = (z3.detach().argmax(-1) == yb).float().mean(1)     # pre-update: the paper's a_j
        acc_sum.add_(hit)
        last_hit.copy_(hit)
        grads = torch.autograd.grad(lossv.sum(), P)
        with torch.no_grad():
            bad = ~torch.isfinite(lossv)
            bad_step.copy_(torch.where((bad_step < 0) & bad, step_t, bad_step))
            step_t.add_(1)
            for p, gr, mi, vi in zip(P, grads, adam_m, adam_v):
                mi.mul_(b1).add_(gr, alpha=1 - b1)
                vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                p.sub_(lr * (mi * inv_c1) / ((vi * inv_c2).sqrt() + eps))
            act.update(z1.detach(), z2.detach())

    use_graph = graph and device.type == "cuda" and debug is None
    cg = None
    if use_graph:
        keep = [q.detach().clone() for q in (*P, *adam_m, *adam_v, acc_sum, bad_step,
                                             step_t, last_hit)]
        keep_act = act.state()
        inv_c1.fill_(1.0); inv_c2.fill_(1.0)
        static_idx.zero_()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                act.begin_step(R, BATCH, device)
                step()
        torch.cuda.current_stream().wait_stream(side)
        cg = torch.cuda.CUDAGraph()
        act.begin_step(R, BATCH, device)
        with torch.cuda.graph(cg):
            step()
        with torch.no_grad():
            for q, v in zip((*P, *adam_m, *adam_v, acc_sum, bad_step, step_t, last_hit), keep):
                q.copy_(v)
        act.load_state(keep_act)
        del keep

    def train_task(batches, tc0: int) -> int:
        """780 steps on (R, 780, 32) global row indices.  Returns the new Adam counter."""
        tc = tc0
        acc_sum.zero_()
        bad_step.fill_(-1)
        step_t.zero_()
        for j in range(STEPS_PER_TASK):
            static_idx.copy_(batches[:, j])
            tc += 1
            inv_c1.fill_(1.0 / (1 - b1 ** tc))
            inv_c2.fill_(1.0 / (1 - b2 ** tc))
            act.begin_step(R, BATCH, device)
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
    rows, diverged, tc = [], [], 0
    last_hard = max(t for t in range(1, n_tasks + 1) if t % 2 == 1)
    saved = {}
    out.mkdir(parents=True, exist_ok=True)
    step_ms = float("nan")

    for t in range(1, n_tasks + 1):
        hard = t % 2 == 1
        rows_t = [torch.cat([tr_rows[q] for q in plans[r][t - 1][1]]) for r in range(R)]
        batches = torch.stack([batch_indices(g_batch[s], rows_t[r])
                               for r, s in enumerate(seeds)]).to(device)
        if t == last_hard and fresh:
            saved[t] = batches.clone()
        if debug is not None:
            debug.setdefault("batches", []).append(batches.cpu().clone())
        t0 = time.time()
        tc = train_task(batches, tc)
        step_ms = 1e3 * (time.time() - t0) / STEPS_PER_TASK

        with torch.no_grad():
            finite = torch.stack([torch.isfinite(q).flatten(1).all(1) for q in P]).all(0)
            newly = alive & ((bad_step >= 0) | ~finite)
            for r in torch.nonzero(newly).flatten().tolist():
                diverged.append({"diverged": True, "task": t, "seed": seeds[r], "arm": arm,
                                 "cond": cond, "step": (t - 1) * STEPS_PER_TASK
                                 + max(int(bad_step[r]), 0)})
                rows.append({"arm": arm, "cond": cond, "seed": seeds[r], "slot": r, "lr": lr,
                             "task": t, "hard": int(hard), "acc": float("nan")})
            alive &= ~newly
            live = torch.nonzero(alive).flatten().tolist()
            Xt = torch.stack([X_all[rows_t[r]] for r in range(R)])
            Yt = torch.stack([Y_all[rows_t[r]] for r in range(R)])
            m = evaluate(P, Xt, Yt, act, live)
            del Xt
            te = [torch.cat([te_rows[q] for q in plans[r][t - 1][1]]) for r in range(R)]
            test_acc = accuracy(P, torch.stack([X_test[q] for q in te]),
                                torch.stack([Y_test[q] for q in te]), act)
        for r in live:
            rows.append({"arm": arm, "cond": cond, "seed": seeds[r], "slot": r, "lr": lr,
                         "task": t, "hard": int(hard), "n_classes": len(plans[r][t - 1][1]),
                         "online_acc": float(acc_sum[r]) / STEPS_PER_TASK,
                         "train_acc": m[r]["acc"], "test_acc": float(test_acc[r]), **m[r]})
        rows.sort(key=lambda q: (q["slot"], q["task"]))
        H.write_csv(out / "per_task.csv", rows)
        on = acc_sum[alive] / STEPS_PER_TASK
        el = time.time() - t_start
        progress(f"[{time.strftime('%T')}] {arm}/{cond} task {t:2d}/{n_tasks} "
                 f"{'hard' if hard else 'easy'} alive {int(alive.sum())}/{R} "
                 f"online {float(on.mean()) if len(on) else float('nan'):.3f} "
                 f"test {float(test_acc[alive].mean()) if int(alive.sum()) else float('nan'):.3f} "
                 f"{step_ms:.2f} ms/step  {el/60:.1f} min, ETA {el/t*(n_tasks-t)/60:.1f} min")

    # ---- fresh-network control on the last hard task (spec §3.2)
    fresh_rows = []
    if fresh and last_hard in saved:
        continual = {int(q["seed"]): q["online_acc"] for q in rows
                     if q["task"] == last_hard and "online_acc" in q}
        with torch.no_grad():
            for q, v in zip(P, P0):
                q.copy_(v)
            for q in (*adam_m, *adam_v):
                q.zero_()
        act.load_state(act0)
        alive_f = torch.ones(R, dtype=torch.bool, device=device)
        train_task(saved[last_hard], 0)
        with torch.no_grad():
            finite = torch.stack([torch.isfinite(q).flatten(1).all(1) for q in P]).all(0)
            alive_f &= (bad_step < 0) & finite
        for r in range(R):
            if not bool(alive_f[r]) or seeds[r] not in continual:
                continue
            value = float(acc_sum[r]) / STEPS_PER_TASK
            fresh_rows.append({"arm": arm, "cond": cond, "seed": seeds[r], "lr": lr,
                               "task": last_hard, "fresh_online_acc": value,
                               "continual_online_acc": continual[seeds[r]],
                               "fresh_gap": value - continual[seeds[r]]})
        H.write_csv(out / "fresh_control.csv", fresh_rows)
        gaps = [q["fresh_gap"] for q in fresh_rows]
        progress(f"[{time.strftime('%T')}] {arm}/{cond} fresh control on task {last_hard}: "
                 f"gap median {float(np.median(gaps)) if gaps else float('nan'):+.3f} "
                 f"({sum(q > 0 for q in gaps)}/{len(gaps)} positive)")

    prov = {"run_id": EXPERIMENT, **B.git_state(), "arm": arm, "cond": cond, "seeds": seeds,
            "slots": [{"seed": s} for s in seeds], "R": R, "lr": lr, "n_tasks": n_tasks,
            "steps_per_task": STEPS_PER_TASK, "batch": BATCH, "hidden": hidden,
            "layer_shapes": layer_shapes(arm, hidden), "n_params": n_params(arm, hidden),
            "phi_doubles_width": arm in WIDE, "dims": list(DIMS),
            "n_classes": N_CLASSES, "hard_classes": HARD_CLASSES,
            "per_class_train": PER_CLASS_TRAIN, "classes_used": CLASSES_USED,
            "task_parity": "task 1 hard, alternating (the paper's task 0 hard)",
            "sna_c": c, "sna_beta": beta, "alpha_lo": lo, "alpha_hi": hi,
            "optimizer": "adam", "weight_decay": 0.0, "data_sha256": cifar.sha256,
            "std": {"mean": STD_MEAN, "std": STD_STD, "planes": "R,G,B x 1024"} if cond == "std" else None,
            "class_sha256": {str(s): hashlib.sha256(
                np.asarray(seed_classes(s), dtype=np.int64).tobytes()).hexdigest() for s in seeds},
            "rng_roles": ["c51_classes", "c51_batch", "init"]
                         + (["rsl_noise(cuda, per process)"] if act.stochastic else []),
            "rsl_noise_seed": getattr(act, "noise_seed", None),
            "fresh_control": {"task": last_hard, "n": len(fresh_rows)} if fresh else None,
            "engine": "stacked baddbmm from rlcifar_mlp_battle_0918, autograd, elementwise Adam"
                      + (", one step captured as a CUDA graph" if use_graph else ", eager"),
            "device": str(device), "torch": torch.__version__,
            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "step_ms_last_task": step_ms, "wall_clock_s": time.time() - t_start,
            "divergences": diverged, "check_hooks": {"nan_slot": nan_slot},
            "spec": "specs/spec_cifar5p1_mlp_0920.md"}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    progress(f"wrote {out}/per_task.csv ({len(rows)} rows, {(time.time()-t_start)/60:.1f} min)")
    return prov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--arm", required=True, choices=list(ARM_ORDER))
    ap.add_argument("--hidden", type=int, default=HIDDEN,
                    help="hidden width; 94 is the equal-parameter width for CR/DF (spec addendum 1)")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--cond", default="std", choices=list(CONDS))
    ap.add_argument("--tasks", type=int, default=N_TASKS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--c", type=float, default=0.6)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--alpha-lo", type=float, default=0.005)
    ap.add_argument("--alpha-hi", type=float, default=3.0)
    ap.add_argument("--no-fresh", action="store_true", help="skip the fresh-network control")
    ap.add_argument("--no-graph", action="store_true", help="eager steps (checks)")
    ap.add_argument("--out", default=None, help="default results/<experiment>/<arm>_<cond>_lr<lr>")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = H.setup(a.device)
    tag = f"{a.arm}_{a.cond}_lr{a.lr:g}" + (f"_h{a.hidden}" if a.hidden != HIDDEN else "")
    out = Path(a.out) if a.out else OUT_ROOT / tag
    run(a.arm, B.parse_ints(a.seeds), a.cond, a.tasks, device, out, lr=a.lr, c=a.c,
        beta=a.beta, lo=a.alpha_lo, hi=a.alpha_hi, fresh=not a.no_fresh,
        graph=not a.no_graph, hidden=a.hidden)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Random Label CIFAR (Kumar, Marklund & Van Roy 2024 §4.2; spec RandomLabelCIFAR_spec_0907).

    python3 src/pmnist_rlcifar_0907.py --arms SNA --seeds 0 --tasks 50

Kumar et al. define this box as "equivalent to the setup of Random Label MNIST
except that data is sampled from the CIFAR-10 training dataset", so this module
is `pmnist_rlmnist_0906` with the data layer swapped: 1200 images drawn once per
seed from CIFAR-10's 50,000 training images, relabelled uniformly every task,
fitted to memorisation for 400 epochs x 75 steps of batch 16.  Protocol,
intervention mechanism, metrics and divergence handling are imported from that
module rather than restated, so the two boxes cannot drift apart.

The only intended differences from the MNIST box are the dataset, the input
dimension (784 -> 3072, hence a 3072-100-100-10 net), and the per-task
preactivation histogram written to `hist/<arm>_seed<seed>.npz` -- the MNIST box
got its distributions from a separate seed-0 re-run, whereas here the capture
rides the per-task evaluation and so covers all 10 seeds on the real trajectory.
Neither `pmnist_0905` (host: rng / init / activations / forward / metric formulas
/ csv) nor `pmnist_rlmnist_0906` is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import sys
import tarfile
import time
from pathlib import Path

# Determinism must be requested before the first CUDA workspace is allocated.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H              # host; not touched
from src import pmnist_rlmnist_0906 as RL     # protocol sibling; not touched

EXPERIMENT = "pmnist_rlcifar_0907"
DATA_DIR = H.REPO / "data" / "cifar10"
ARCHIVE = "cifar-10-python.tar.gz"
TRAIN_N = 50_000                               # CIFAR-10 training set (5 batches x 10,000)
PIXELS = 3072                                  # 32 x 32 x 3, as data_batch_* stores it
DIMS = (PIXELS, 100, 100, 10)                  # hidden layers unchanged from the MNIST box
N_IMAGES = RL.N_IMAGES                         # 1200 images per seed, shared by every task
BATCH = H.BATCH                                # 16
STEPS_PER_EPOCH = RL.STEPS_PER_EPOCH           # 75
N_CLASSES = H.N_CLASSES                        # 10
DEFAULT_ARMS = "SNA,LR,R"                      # spec §2.2; the +l2/+l2init runs pass --iv

# Per-task preactivation histogram. Width 0.1, the resolution of the MNIST capture
# in results/_diag_rlhist_0907/ (-16..8, 240 bins).
#
# The range is set from measurement, not from scaling the MNIST one: CIFAR drives z
# roughly an order of magnitude further than MNIST did. Over 10 tasks x 400 epochs at
# seed 0 the envelope was z in [-216.2, +141.7] (worst: SNA z2 at task 8), with R+none
# pinned at -128.3 once it dies and SNA+l2's z1 still drifting down at task 10. The
# limits below leave ~2.4x headroom below and ~1.8x above that envelope for the 40
# further tasks of the real run; `oob` still counts anything outside, so a range that
# is nonetheless too narrow reports itself instead of silently clipping.
HIST_LO, HIST_HI, HIST_NB = -512.0, 256.0, 7680
HIST_EDGES = np.linspace(HIST_LO, HIST_HI, HIST_NB + 1)

# imported so the two boxes share one definition, not two copies that can diverge
task_labels = RL.task_labels                   # iid uniform {0..9}, advances g once
evaluate_rl = RL.evaluate_rl                   # metrics on the 1200 images, current labels
Iv = RL.Iv                                     # none | l2:<lam> | l2init:<lam>
out_tag = RL.out_tag


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

class _SafeUnpickler(pickle.Unpickler):
    """The CIFAR pickles hold nothing but a dict of ndarrays, bytes and ints.

    Unpickling executes whatever the file names, and this file arrives over the
    network; allowing only numpy's array reconstructors turns a bad download into
    an exception instead of a shell.
    """
    ALLOWED = {("numpy", "ndarray"), ("numpy", "dtype"),
               ("numpy.core.multiarray", "_reconstruct"),
               ("numpy._core.multiarray", "_reconstruct")}

    def find_class(self, module: str, name: str):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")
        return super().find_class(module, name)


class Cifar10:
    """CIFAR-10 training images, flattened to 3072 and scaled to [0,1] by /255.

    Channel order is whatever `data_batch_*` stores (1024 R, then G, then B) and
    the scaling is the MNIST box's, so nothing is reordered or per-channel
    normalised -- spec §2.4 puts input normalisation out of scope.

    The 50,000 images stay uint8 on cpu (154 MB) and only the seed's 1200 become
    float32 on the device: the launch runs 7 processes, and a resident float32
    copy in each would cost 4.3 GB of host RAM to hold data that is 97.6% unused.
    """
    MEMBERS = tuple(f"cifar-10-batches-py/data_batch_{i}" for i in range(1, 6))

    def __init__(self):
        path = DATA_DIR / ARCHIVE
        self.sha256 = {ARCHIVE: hashlib.sha256(path.read_bytes()).hexdigest()}
        parts = []
        with tarfile.open(path, "r:gz") as tf:
            for name in self.MEMBERS:
                fh = tf.extractfile(name)
                if fh is None:
                    raise SystemExit(f"{ARCHIVE}: member {name} missing")
                d = _SafeUnpickler(io.BytesIO(fh.read()), encoding="bytes").load()
                parts.append(d[b"data"])       # (10000, 3072) uint8
        self.train_u8 = torch.from_numpy(np.concatenate(parts))
        if tuple(self.train_u8.shape) != (TRAIN_N, PIXELS):
            raise SystemExit(f"{ARCHIVE}: got {tuple(self.train_u8.shape)}, "
                             f"want {(TRAIN_N, PIXELS)}")

    def images(self, idx: torch.Tensor, device: torch.device) -> torch.Tensor:
        """`idx` rows as float32 in [0,1]. uint8 -> float32 is exact, so this is
        bit-identical to converting the whole set up front and indexing it."""
        return self.train_u8[idx.cpu()].to(device=device, dtype=torch.float32).div_(255.0)


def subset_idx(seed: int) -> torch.Tensor:
    """The seed's 1200 training images: uniform, no replacement, no stratification.

    Own stream, drawn once per seed, so the image set is a property of the seed
    and is bit-identical across arms, tasks and interventions.
    """
    g = H.stream("rlc_subset", seed)
    return torch.randperm(TRAIN_N, generator=g)[:N_IMAGES]


# --------------------------------------------------------------------------
# per-task preactivation distribution
# --------------------------------------------------------------------------

@torch.no_grad()
def preact_hist(params, x: torch.Tensor, act) -> dict:
    """Histogram and per-unit mean of each hidden layer's preactivation.

    `evaluate_rl` is imported from the MNIST box and cannot be widened to hand
    back z1/z2, so the forward over the 1200 evaluation images is repeated here:
    against 30,000 training steps per task that is under 0.1% of the task's work.
    """
    z1, _, z2, _, _ = H.forward(params, x, act)
    out = {}
    for tag, z in (("1", z1), ("2", z2)):
        v = z.flatten().cpu().numpy()
        # np.histogram with explicit edges closes the last bin, so this `oob`
        # is the exact complement: h.sum() + oob == v.size.
        out[f"h{tag}"] = np.histogram(v, bins=HIST_EDGES)[0].astype(np.int32)
        out[f"m{tag}"] = z.mean(0).cpu().numpy().astype(np.float32)
        out[f"oob{tag}"] = np.int64(((v < HIST_LO) | (v > HIST_HI)).sum())
    return out


def write_hist(path: Path, hists: list[dict], accs: list[float]) -> None:
    """One self-contained npz per (arm, seed); row index = task - 1."""
    if len(hists) != len(accs):                # a diverged task contributes neither
        raise AssertionError(f"{len(hists)} histograms vs {len(accs)} accuracies")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, edges=HIST_EDGES, acc=np.asarray(accs, dtype=np.float64),
        **{k: np.stack([h[k] for h in hists])
           for k in ("h1", "h2", "m1", "m2", "oob1", "oob2")})


# --------------------------------------------------------------------------
# one (arm, seed, lr) run -- pmnist_rlmnist_0906.run_one with the CIFAR data
# layer and the 3072-wide first layer.
# --------------------------------------------------------------------------

def run_one(arm: str, seed: int, lr: float, n_tasks: int, cifar: Cifar10,
            device: torch.device, optimizer: str = "adam", epochs: int = 400,
            c: float = 0.6, alpha_lo: float = 0.005, alpha_hi: float = 3.0, beta: float = 0.01, iv: str = "none",
            debug: dict | None = None,
            hist_dir: Path | None = None) -> tuple[list[dict], dict]:
    act = H.ARMS[arm]
    if act.kind == "adaptive_snake":
        # clip floor is a runaway guard (spec §1.1-2), not a design value; CIFAR's
        # 3072-dim input makes W large enough that the host default 0.05 binds and
        # stops alpha tracking W at all (spec §2.5-b).
        act = H.AdaptiveSnake(c, beta, device, lo=alpha_lo, hi=alpha_hi,
                              widths=(DIMS[1], DIMS[2]))  # fresh stats per run
    params = H.init_params(seed, device, DIMS)        # host init rule, 3072 fan-in on layer 1
    if debug is not None:                             # S-init / S-online hook, unused in real runs
        debug["init"] = [q.detach().cpu().clone() for q in params]
    ivo = Iv.parse(iv)
    p0 = [q.detach().clone() for q in params] if ivo.kind == "l2init" else None
    # Adam moments live for the whole run: resetting them between tasks would break
    # the continual definition the same way resetting weights does.
    adam = ([torch.zeros_like(q) for q in params],
            [torch.zeros_like(q) for q in params], [0]) if optimizer == "adam" else None

    idx = subset_idx(seed)
    x = cifar.images(idx, device)                     # the task's inputs, fixed forever
    g_lab, g_batch = H.stream("rlc_labels", seed), H.stream("rlc_batch", seed)
    spt = STEPS_PER_EPOCH * epochs
    rows, hists = [], []
    diverged = {"diverged": False, "task": None, "step": None, "seed": seed, "arm": arm}

    for t in range(1, n_tasks + 1):
        y = task_labels(g_lab).to(device)             # new labelling, same images
        if debug is not None:
            debug.setdefault("subset", []).append(idx.clone())
            debug.setdefault("labels", []).append(y.cpu().clone())
        acc_sum = torch.zeros((), device=device)
        bad_step = torch.full((), -1, dtype=torch.long, device=device)

        for e in range(epochs):
            order = torch.randperm(N_IMAGES, generator=g_batch).to(device)
            xs, ys = x[order], y[order]               # reshuffled every epoch
            for j in range(STEPS_PER_EPOCH):
                s = e * STEPS_PER_EPOCH + j
                xb, yb = xs[j * BATCH:(j + 1) * BATCH], ys[j * BATCH:(j + 1) * BATCH]
                out = H.forward(params, xb, act)
                loss = torch.nn.functional.cross_entropy(out[4], yb)
                # pre-update accuracy: the argmax of the very forward pass the loss
                # came from, i.e. before this batch has been learned from.
                hit = (out[4].detach().argmax(1) == yb).float().mean()
                acc_sum += hit
                grads = torch.autograd.grad(loss, params)
                with torch.no_grad():
                    bad = ~torch.isfinite(loss)
                    bad_step = torch.where((bad_step < 0) & bad,
                                           torch.tensor(s, device=device), bad_step)
                    if ivo.kind in ("l2", "l2init"):
                        grads = [gr + 2.0 * ivo.lam * (q - (p0[i] if p0 is not None else 0.0))
                                 for i, (q, gr) in enumerate(zip(params, grads))]
                    if adam is None:
                        for p, gr in zip(params, grads):
                            p -= lr * gr
                    else:
                        m, v, tc = adam
                        tc[0] += 1
                        b1, b2, eps = 0.9, 0.999, 1e-8
                        c1 = 1 - b1 ** tc[0]
                        c2 = 1 - b2 ** tc[0]
                        for p, gr, mi, vi in zip(params, grads, m, v):
                            mi.mul_(b1).add_(gr, alpha=1 - b1)
                            vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                            p -= lr * (mi / c1) / ((vi / c2).sqrt() + eps)
                    if isinstance(act, H.AdaptiveSnake):
                        act.update(out[0], out[2])    # running var of this batch's preacts
                if debug is not None:
                    debug.setdefault("online", []).append(float(hit))

        bs = int(bad_step)
        if bs >= 0 or not all(torch.isfinite(p).all() for p in params):
            diverged.update(diverged=True, task=t, step=(t - 1) * spt + max(bs, 0))
            rows.append({"arm": arm, "seed": seed, "lr": lr, "task": t, "iv": iv,
                         "acc": float("nan")})
            break                                     # drop, never rescue
        m = evaluate_rl(params, x, y, act)
        hists.append(preact_hist(params, x, act))
        rows.append({"arm": arm, "seed": seed, "lr": lr, "task": t, "iv": iv,
                     "online_acc": float(acc_sum) / spt, "memo_acc": m["acc"], **m})
    # written per (arm, seed), truncated to the tasks that completed: a diverged
    # run keeps the distributions it did produce.
    if hist_dir is not None and hists:
        write_hist(Path(hist_dir) / f"{arm}_seed{seed}.npz", hists,
                   [r["memo_acc"] for r in rows if "memo_acc" in r])
    return rows, diverged


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--lrs", default="0.001")
    ap.add_argument("--tasks", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=400,
                    help="passes over the 1200 examples per task (75 steps each)")
    ap.add_argument("--alpha-lo", type=float, default=0.005,
                    help="SNA: lower clip on alpha_i (runaway guard; spec §2.5-b widened it from 0.05)")
    ap.add_argument("--alpha-hi", type=float, default=3.0, help="SNA: upper clip on alpha_i")
    ap.add_argument("--c", type=float, default=0.6, help="SNA: alpha_i = c / W_i")
    ap.add_argument("--beta", type=float, default=0.01, help="SNA: EMA rate of var(z_i)")
    ap.add_argument("--iv", default="none", help="none | l2:<lam> | l2init:<lam>")
    ap.add_argument("--optimizer", default="adam", choices=["sgd", "adam"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    arms = args.arms.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    lrs = [float(s) for s in args.lrs.split(",")]
    Iv.parse(args.iv)                                  # fail fast on a bad spec
    for a in arms:
        if a not in H.ARMS:
            raise SystemExit(f"unknown arm {a!r}; known: {','.join(H.ARMS)}")

    device = H.setup(args.device)
    t_start = time.time()
    cifar = Cifar10()

    out = Path(args.out or H.REPO / "results" / EXPERIMENT / out_tag(arms, args.iv))
    out.mkdir(parents=True, exist_ok=True)

    rows, divs = [], []
    for lr in lrs:
        for arm in arms:
            for seed in seeds:
                t0 = time.time()
                r, d = run_one(arm, seed, lr, args.tasks, cifar, device,
                               optimizer=args.optimizer, epochs=args.epochs,
                               c=args.c, alpha_lo=args.alpha_lo, alpha_hi=args.alpha_hi, beta=args.beta, iv=args.iv,
                               hist_dir=out / "hist")
                rows += r
                if d["diverged"]:
                    divs.append(d)
                print(f"[{time.time()-t_start:7.1f}s] lr={lr:<6g} {arm:<4} seed={seed} "
                      f"tasks={len([q for q in r if q.get('acc') == q.get('acc')])} "
                      f"{(time.time()-t0):.1f}s"
                      f"{'  DIVERGED@step ' + str(d['step']) if d['diverged'] else ''}",
                      flush=True)
                H.write_csv(out / "per_task.csv", rows)

    prov = {"run_id": EXPERIMENT, "git_hash": H.git_hash(),
            "arms": arms, "seeds": seeds, "lrs": lrs, "n_tasks": args.tasks,
            "epochs_per_task": args.epochs, "batch": BATCH,
            "steps_per_task": STEPS_PER_EPOCH * args.epochs,
            "n_images": N_IMAGES, "train_n": TRAIN_N, "dims": list(DIMS),
            "n_classes": N_CLASSES,
            "sna_c": args.c, "sna_beta": args.beta, "alpha_lo": args.alpha_lo, "alpha_hi": args.alpha_hi, "intervention": args.iv,
            "optimizer": args.optimizer, "data_sha256": cifar.sha256,
            "subset_sha256": {str(s): hashlib.sha256(
                np.sort(subset_idx(s).numpy()).tobytes()).hexdigest() for s in seeds},
            "rng_roles": ["rlc_subset", "rlc_labels", "rlc_batch", "init"],
            "hist": {"lo": HIST_LO, "hi": HIST_HI, "bins": HIST_NB, "dtype": "int32",
                     "path": "hist/<arm>_seed<seed>.npz"},
            "device": str(device), "torch": torch.__version__,
            "wall_clock_s": time.time() - t_start, "divergences": divs}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"\nwrote {out}/per_task.csv  ({len(rows)} rows, {time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()

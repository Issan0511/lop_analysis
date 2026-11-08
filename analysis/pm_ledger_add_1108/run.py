"""pm_ledger_add_1108 -- arms CHB / KKT1 and the switch interventions INV0 / ISO / ADD.

spec: specs/spec_pm_ledger_add_1108.md   (main run: specs/spec_pm_ledger_1108.md)

The task data, init and update rule are those of src/pmnist_0905.py (box).  With no
doors and no switch op this harness is the main run's loop (--check-ident compares
it bit for bit with the main run's saved W1 for the none and l2init arms).

CHB   ReLU with door C (input EMA mean, coeff 0.01), door H (per-unit EMA of the
      activation mean, both hidden layers, coeff 0.01) and door B (b1 held at 0,
      b2 <- (1 - lr*lam) b2 after every Adam step, lam = 0.1713).  Appendix A.5 / J.1
      and src/relu_doors_0919.py (arm CHB, single run here).
KKT1  kunekune: appendix F.1, src/rlcifar_mlp_battle_0918.py SnakeFamily("KKT1", "kkt1"),
      c = 0.6, beta = 0.01, lo = 0.005, hi = 3.0.
Switch ops (task t >= 2, after the new permutation is known and before training):
      inv0: W1 <- W1 - W1 P_inv,  P_inv the projector on PCs 440..784 of the new task's
            centred input covariance (the main run's low and comp bands)
      iso:  W1 <- c W1 with c = sqrt(1 - ||W1 P_inv||^2/||W1||^2)
      add:  W1 <- W1 + R_t, R_t random in span(P_inv) with ||R_t||_F^2 = the none arm's
            M_inv(t) of the main run (same seed, same task).

Usage:
  .venv/bin/python analysis/pm_ledger_add_1108/run.py --arm CHB --seed 0 --tasks 500 --out DIR
  .venv/bin/python analysis/pm_ledger_add_1108/run.py --check-ident
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""            # spec: CPU only

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import pmnist_0905 as B                            # noqa: E402

B.DATA_DIR = Path(os.environ.get("PM_MNIST_DIR",
                                 "/home/issan/Projects/claude/proj_004_drift/data/mnist"))
MAIN = Path("/home/issan/Projects/obsidian-research-data/pm_ledger_1108")

LR = 1e-3
LAM_B2 = 0.1713                                    # teacher's value for PM CHB
BETA = 0.01                                        # EMA rate of the doors and of V
L2INIT = 1e-3
FRESH_TASKS = (100, 200, 300, 400, 500)
N_INV_FROM = 439                                   # PCs 440..784 (0-based index 439..783)
PI = math.pi

ARMS = {
    "none":      {"act": "R",    "switch": None,   "lam": None,   "fresh": "main:none"},
    "l2init":    {"act": "R",    "switch": None,   "lam": L2INIT, "fresh": "main:l2init"},
    "CHB":       {"act": "CHB",  "switch": None,   "lam": None,   "fresh": "run"},
    "KKT1":      {"act": "KKT1", "switch": None,   "lam": None,   "fresh": "run"},
    "CHB+INV0":  {"act": "CHB",  "switch": "inv0", "lam": None,   "fresh": "run"},
    "CHB+ISO":   {"act": "CHB",  "switch": "iso",  "lam": None,   "fresh": "run"},
    "L2I+ADD":   {"act": "R",    "switch": "add",  "lam": L2INIT, "fresh": "main:l2init"},
    "KKT1+INV0": {"act": "KKT1", "switch": "inv0", "lam": None,   "fresh": "run"},
    "none+INV0": {"act": "R",    "switch": "inv0", "lam": None,   "fresh": "main:none"},
}


def dirname(arm: str) -> str:
    return arm.replace("+", "_")


class Doors:
    """Activation and centering state of one network (spec_pm_ledger_add §2)."""

    def __init__(self, kind: str):
        self.kind = kind
        if kind == "CHB":
            self.mx = torch.zeros(784)
            self.m = [torch.zeros(100) for _ in range(2)]
        elif kind == "KKT1":
            self.V = [torch.ones(100) for _ in range(2)]

    # kunekune constants (F.1)
    C, LO, HI = 0.6, 0.005, 3.0

    def alpha(self, layer):
        return (self.C / self.V[layer].sqrt()).clamp(self.LO, self.HI)

    def inp(self, x):
        return x - self.mx if self.kind == "CHB" else x

    def phi(self, z, layer):
        if self.kind == "R":
            return torch.clamp(z, min=0.0)
        if self.kind == "CHB":
            return torch.clamp(z, min=0.0) - self.m[layer]
        a = self.alpha(layer)
        ia = a.reciprocal()
        snake = z + torch.sin(a * z) ** 2 * ia
        th = 2.0 * a * z
        right = z + ia
        return torch.where(th < -2.0 * PI, z, torch.where(th > PI, right, snake))

    def dphi(self, z, layer):
        if self.kind in ("R", "CHB"):
            return (z > 0).to(z.dtype)
        a = self.alpha(layer)
        th = 2.0 * a * z
        s = 1.0 + torch.sin(th)
        inside = (th >= -2.0 * PI) & (th <= PI)
        return torch.where(inside, s, torch.ones_like(z))

    @torch.no_grad()
    def post_update(self, params, x_raw, z1, z2, lr):
        """After the Adam step: door B, then advance the EMAs from the same forward's z and x."""
        if self.kind == "CHB":
            params[1].zero_()
            params[3].mul_(1.0 - lr * LAM_B2)
            self.m[0].mul_(1.0 - BETA).add_(torch.clamp(z1, min=0.0).mean(0), alpha=BETA)
            self.m[1].mul_(1.0 - BETA).add_(torch.clamp(z2, min=0.0).mean(0), alpha=BETA)
            self.mx.mul_(1.0 - BETA).add_(x_raw.mean(0), alpha=BETA)
        elif self.kind == "KKT1":
            self.V[0].mul_(1.0 - BETA).add_(BETA * z1.var(0, unbiased=False))
            self.V[1].mul_(1.0 - BETA).add_(BETA * z2.var(0, unbiased=False))


def forward(params, x, doors):
    W1, b1, W2, b2, W3, b3 = params
    x = doors.inp(x)
    z1 = x @ W1.T + b1
    a1 = doors.phi(z1, 0)
    z2 = a1 @ W2.T + b2
    a2 = doors.phi(z2, 1)
    return z1, a1, z2, a2, a2 @ W3.T + b3


@torch.no_grad()
def evaluate(params, mnist, perm, doors):
    """Same quantities as src/pmnist_0905.evaluate on the full permuted test set."""
    x = mnist.test_x[:, perm]
    z1, a1, z2, a2, logits = forward(params, x, doors)
    out = {"acc": float((logits.argmax(1) == mnist.test_y).float().mean())}
    for li, (tag, z, a) in enumerate((("l1", z1, a1), ("l2", z2, a2))):
        d = doors.dphi(z, li)
        out[f"dead_frac_{tag}"] = float((d.abs().amax(0) < B.DEAD_TOL).float().mean())
        out[f"zeroout_{tag}"] = float((a.abs().amax(0) == 0).float().mean())
        out[f"zbar_{tag}"] = float(z.mean(0).median())
        out[f"zsd_{tag}"] = float(z.std(0).median())
        out[f"zbar_min_{tag}"] = float(z.mean(0).amin())
        out[f"mob_{tag}"] = float(d.mean(0).median())
        out[f"eff_rank_{tag}"] = B.eff_rank(a)
    for i, tag in enumerate(("l1", "l2", "l3")):
        out[f"w_norm_{tag}"] = float(params[2 * i].norm(dim=1).median())
    return out


def zero_adam(params):
    return ([torch.zeros_like(q) for q in params], [torch.zeros_like(q) for q in params], [0])


def train_task(params, adam, doors, lam, p0, xs, ys, lr):
    """625 updates; the update loop of src/pmnist_0905.run_one plus the doors."""
    n_correct = torch.zeros((), dtype=torch.long)
    bad_step = torch.full((), -1, dtype=torch.long)
    for s in range(B.STEPS_PER_TASK):
        j = s % B.STEPS_PER_TASK
        xb = xs[j * B.BATCH:(j + 1) * B.BATCH]
        yb = ys[j * B.BATCH:(j + 1) * B.BATCH]
        out = forward(params, xb, doors)
        loss = torch.nn.functional.cross_entropy(out[4], yb)
        grads = torch.autograd.grad(loss, params)
        with torch.no_grad():
            n_correct += (out[4].argmax(1) == yb).sum()
            bad = ~torch.isfinite(loss)
            bad_step = torch.where((bad_step < 0) & bad, torch.tensor(s), bad_step)
            if lam is not None:
                grads = [gr + 2.0 * lam * (q - p0[i])
                         for i, (q, gr) in enumerate(zip(params, grads))]
            m, v, tc = adam
            tc[0] += 1
            b1, b2, eps = 0.9, 0.999, 1e-8
            c1 = 1 - b1 ** tc[0]
            c2 = 1 - b2 ** tc[0]
            for p, gr, mi, vi in zip(params, grads, m, v):
                mi.mul_(b1).add_(gr, alpha=1 - b1)
                vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                p -= lr * (mi / c1) / ((vi / c2).sqrt() + eps)
            doors.post_update(params, xb, out[0], out[2], lr)
    return n_correct.item(), int(bad_step)


def task_batches(mnist, g_perm, g_data, g_batch):
    perm = torch.randperm(784, generator=g_perm)
    idx = B.stratified_draw(mnist, g_data)
    order = torch.randperm(B.TASK_EXAMPLES, generator=g_batch)
    xs = mnist.train_x[idx][:, perm][order]
    ys = mnist.train_y[idx][order]
    return perm, idx, xs, ys


def inv_basis(xs: torch.Tensor) -> np.ndarray:
    """Orthonormal basis of the main run's low and comp bands (PCs 440..784) of this task."""
    X = xs.numpy().astype(np.float64)
    Xc = X - X.mean(0)
    S = (Xc.T @ Xc) / (X.shape[0] - 1)
    _, Q = np.linalg.eigh(S)                      # ascending eigenvalues
    return Q[:, ::-1][:, N_INV_FROM:]             # descending order, PCs 440..784


def apply_switch(params, kind, xs, add_mass, g_add):
    """The arm's switch operation; returns its bookkeeping row."""
    Q = inv_basis(xs)
    W = params[0].detach().numpy().astype(np.float64)
    C = W @ Q
    inv_pre = float((C * C).sum())
    F2_pre = float((W * W).sum())
    rec = {"inv_pre": inv_pre, "F2_pre": F2_pre, "c_iso": float("nan"), "add_R2": float("nan")}
    if kind == "inv0":
        W_new = W - C @ Q.T
    elif kind == "iso":
        c = math.sqrt(max(F2_pre - inv_pre, 0.0) / F2_pre)
        rec["c_iso"] = c
        W_new = c * W
    elif kind == "add":
        G = torch.randn((W.shape[0], Q.shape[1]), generator=g_add, dtype=torch.float64).numpy()
        R = G @ Q.T
        scale = math.sqrt(add_mass / float((R * R).sum())) if add_mass > 0 else 0.0
        R = R * scale
        rec["add_R2"] = float((R * R).sum())
        W_new = W + R
    else:
        raise ValueError(kind)
    with torch.no_grad():
        params[0].copy_(torch.from_numpy(W_new.astype(np.float32)))
    W_after = params[0].detach().numpy().astype(np.float64)   # the float32 value actually used
    Ca = W_after @ Q
    rec["inv_post"] = float((Ca * Ca).sum())
    rec["F2_post"] = float((W_after * W_after).sum())
    return rec


def run_loop(arm, seed, n_tasks, mnist, out=None):
    cfg = ARMS[arm]
    lam = cfg["lam"]
    switch = cfg["switch"]
    device = torch.device("cpu")
    params = B.init_params(seed, device)
    p0 = [q.detach().clone() for q in params] if lam is not None else None
    adam = zero_adam(params)
    doors = Doors(cfg["act"])
    g_perm, g_data, g_batch = B.stream("perm", seed), B.stream("data", seed), B.stream("batch", seed)
    g_add = B.stream("add", seed) if switch == "add" else None
    add_series = None
    if switch == "add":
        z = np.load(MAIN / "ledger" / f"seed{seed}.npz")
        add_series = z["none__Sprev"][:, 3] + z["none__Sprev"][:, 4]   # M_inv(t), low + comp

    saving = out is not None
    if saving:
        out.mkdir(parents=True, exist_ok=True)
        W1 = np.lib.format.open_memmap(out / "W1.npy", mode="w+", dtype=np.float32,
                                       shape=(n_tasks + 1, 100, 784))
        b1 = np.lib.format.open_memmap(out / "b1.npy", mode="w+", dtype=np.float32,
                                       shape=(n_tasks + 1, 100))
        perms = np.lib.format.open_memmap(out / "perm.npy", mode="w+", dtype=np.int16,
                                          shape=(n_tasks + 1, 784))
        perms[0] = -1
        W1[0] = params[0].detach().numpy()
        b1[0] = params[1].detach().numpy()
        W1.flush()
        W1s = None
        if switch is not None:
            W1s = np.lib.format.open_memmap(out / "W1s.npy", mode="w+", dtype=np.float32,
                                            shape=(n_tasks + 1, 100, 784))
            W1s[0] = params[0].detach().numpy()
    rows, fresh_rows, idx_sha, diverged = [], [], {}, {"diverged": False}
    t0 = time.time()

    for t in range(1, n_tasks + 1):
        perm, idx, xs, ys = task_batches(mnist, g_perm, g_data, g_batch)
        sw = {}
        if switch is not None and t >= 2:
            sw = apply_switch(params, switch, xs,
                              float(add_series[t]) if add_series is not None else 0.0, g_add)
        if saving and switch is not None:
            W1s[t] = params[0].detach().numpy()
        n_correct, bs = train_task(params, adam, doors, lam, p0, xs, ys, LR)
        if bs >= 0 or not all(torch.isfinite(p).all() for p in params):
            diverged.update(diverged=True, task=t, step=(t - 1) * B.STEPS_PER_TASK + max(bs, 0))
            rows.append({"arm": arm, "seed": seed, "task": t, "online": float("nan")})
            break
        online = n_correct / B.TASK_EXAMPLES
        row = {"arm": arm, "seed": seed, "task": t, "online": online}
        row.update(evaluate(params, mnist, perm, doors))
        row.update(sw)
        rows.append(row)
        if saving:
            W1[t] = params[0].detach().numpy()
            b1[t] = params[1].detach().numpy()
            perms[t] = perm.numpy().astype(np.int16)
            idx_sha[str(t)] = hashlib.sha256(idx.numpy().tobytes()).hexdigest()
            if t % 25 == 0 or t == n_tasks:
                W1.flush(); b1.flush(); perms.flush()
                if switch is not None:
                    W1s.flush()
                B.write_csv(out / "per_task.csv", rows)
                (out / "idx_sha.json").write_text(json.dumps(idx_sha))
                if t % 100 == 0 or t == n_tasks:
                    print(f"[{time.time() - t0:7.1f}s] {arm} seed{seed} task {t} online {online:.4f}", flush=True)
        if t in FRESH_TASKS and cfg["fresh"] == "run":
            F = B.init_params(seed, device)
            A0 = zero_adam(F)
            p0f = [q.detach().clone() for q in F] if lam is not None else None
            doors_f = Doors(cfg["act"])
            n_correct_f, bs_f = train_task(F, A0, doors_f, lam, p0f, xs, ys, LR)
            fresh_rows.append({"arm": arm, "seed": seed, "task": t,
                               "online_fresh": n_correct_f / B.TASK_EXAMPLES,
                               "online_cont": online,
                               "fresh_bad_step": bs_f})
            if saving:
                B.write_csv(out / "fresh.csv", fresh_rows)
    return rows, fresh_rows, idx_sha, diverged, params, time.time() - t0


def check_ident():
    """S-ident2: arms without doors/switch reproduce the main run's saved W1 bit for bit."""
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    mnist = B.Mnist(device)
    ok_all = True
    for arm in ("none", "l2init"):
        saved = np.load(MAIN / "runs" / arm / "seed0" / "W1.npy", mmap_mode="r")
        saved_b = np.load(MAIN / "runs" / arm / "seed0" / "b1.npy", mmap_mode="r")
        # run 3 tasks and compare the end-of-task states
        cfg = ARMS[arm]; lam = cfg["lam"]
        params = B.init_params(0, device)
        p0 = [q.detach().clone() for q in params] if lam is not None else None
        adam = zero_adam(params)
        doors = Doors(cfg["act"])
        g_perm, g_data, g_batch = B.stream("perm", 0), B.stream("data", 0), B.stream("batch", 0)
        res = {}
        for t in range(1, 4):
            perm, idx, xs, ys = task_batches(mnist, g_perm, g_data, g_batch)
            train_task(params, adam, doors, lam, p0, xs, ys, LR)
            res[t] = bool(np.array_equal(params[0].detach().numpy(), np.asarray(saved[t]))
                          and np.array_equal(params[1].detach().numpy(), np.asarray(saved_b[t])))
        print(arm, res, flush=True)
        ok_all &= all(res.values())
    # the CHB and KKT1 formulas against the repo's implementations (one run, batch of 16)
    # relu_doors_0919.ReLUDoors("CHB") and rlcifar_mlp_battle_0918.SnakeFamily("KKT1", "kkt1")
    sys.path.insert(0, str(REPO / "src"))
    import rlcifar_mlp_battle_0918 as RM
    a = RM.make_act("KKT1", 0.6, 0.01, 0.005, 3.0)
    a.init_state(1, device, None)
    d = Doors("KKT1")
    z = torch.randn(16, 100, generator=torch.Generator().manual_seed(1)) * 3.0
    z_slot = z[None]
    ok = torch.equal(a.phi(z_slot, 0)[0], d.phi(z, 0)) and torch.equal(a.dphi(z_slot, 0)[0], d.dphi(z, 0))
    a.update(z_slot, z_slot)
    d.V[0].mul_(1.0 - BETA).add_(BETA * z.var(0, unbiased=False))
    d.V[1].mul_(1.0 - BETA).add_(BETA * z.var(0, unbiased=False))
    ok &= torch.equal(a.V[0][0], d.V[0]) and torch.equal(a.V[1][0], d.V[1])
    print("KKT1 formula vs repo:", bool(ok), flush=True)
    ok_all &= bool(ok)
    import relu_doors_0919 as RD
    rd = RD.ReLUDoors("CHB", LAM_B2, BETA)
    rd.init_state(1, device, None)
    dc = Doors("CHB")
    z1 = torch.randn(16, 100, generator=torch.Generator().manual_seed(2)) * 2.0
    a1_r = rd.phi(z1[None], 0)[0]
    a1_d = dc.phi(z1, 0)
    rd.update(z1[None], z1[None])
    dc.m[0].mul_(1.0 - BETA).add_(torch.clamp(z1, min=0.0).mean(0), alpha=BETA)
    dc.m[1].mul_(1.0 - BETA).add_(torch.clamp(z1, min=0.0).mean(0), alpha=BETA)
    chk = torch.equal(a1_r, a1_d) and torch.equal(rd.m[0][0], dc.m[0]) and torch.equal(rd.m[1][0], dc.m[1])
    print("CHB door formula vs relu_doors_0919:", bool(chk), flush=True)
    ok_all &= bool(chk)
    if not ok_all:
        raise SystemExit("S-ident2 failed")
    print("S-ident2 PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--tasks", type=int, default=500)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--check-ident", action="store_true")
    args = ap.parse_args()
    if args.check_ident:
        check_ident()
        return
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    mnist = B.Mnist(device)
    rows, fresh_rows, idx_sha, diverged, _, wall = run_loop(args.arm, args.seed, args.tasks, mnist, out=args.out)
    B.write_csv(args.out / "per_task.csv", rows)
    if fresh_rows:
        B.write_csv(args.out / "fresh.csv", fresh_rows)
    (args.out / "idx_sha.json").write_text(json.dumps(idx_sha))
    prov = {"run_id": "pm_ledger_add_1108", "arm": args.arm, "seed": args.seed, "lr": LR, "cfg": ARMS[args.arm],
            "lam_b2": LAM_B2, "beta": BETA, "n_tasks": args.tasks, "dims": list(B.DIMS), "batch": B.BATCH,
            "steps_per_task": B.STEPS_PER_TASK, "task_examples": B.TASK_EXAMPLES, "git_hash": B.git_hash(),
            "device": "cpu", "torch": torch.__version__, "data_sha256": mnist.sha256, "wall_clock_s": wall,
            "max_rss_MB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "fresh_tasks": list(FRESH_TASKS),
            "diverged": diverged, "box": "src/pmnist_0905.py (loop copied; S-ident2)"}
    (args.out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"done {args.arm} seed{args.seed}: {wall:.1f}s, max RSS {prov['max_rss_MB']:.0f} MB", flush=True)


if __name__ == "__main__":
    main()

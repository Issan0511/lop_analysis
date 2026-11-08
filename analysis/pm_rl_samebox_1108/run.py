"""pm_rl_samebox_1108 -- the Permuted MNIST box with two task sequences: pixel permutations or random labels.

spec: specs/spec_pm_rl_samebox_1108.md

Copied from analysis/pm_ledger_add_1108/run.py (left untouched) and extended:
  perm10000  the box of pm_ledger_1108 (only for the S1 regression check)
  perm1200   the per-task pixel permutation on one fixed stratified set of 1,200 training images
  rl1200     no permutation; the same 1,200 images, 10-class uniform random labels redrawn per task
Arms none (ReLU) and CHB (the doors of pm_ledger_add_1108).  625 updates x batch 16 per task;
with 1,200 images each epoch is a new shuffle (8 epochs and 25 updates).  With 10,000 images the
order draw is the main run's single randperm(10000), so the regression check S1 is bit-exact.

Usage:
  .venv/bin/python analysis/pm_rl_samebox_1108/run.py --seq perm1200 --act CHB --seed 0 --tasks 500 --out DIR
  .venv/bin/python analysis/pm_rl_samebox_1108/run.py --check s1|s2|smoke [--smoke-dir DIR]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import statistics
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
DATA = Path("/home/issan/Projects/obsidian-research-data")
MAIN_NONE = DATA / "pm_ledger_1108" / "runs" / "none" / "seed0"
ADD_CHB = DATA / "pm_ledger_add_1108" / "runs" / "CHB" / "seed0"
CHECKS_JSON = REPO / "results" / "pm_rl_samebox_1108" / "checks.json"

LR = 1e-3
LAM_B2 = 0.1713                                    # teacher's value for PM CHB
BETA = 0.01                                        # EMA rate of the doors
FRESH_TASKS = (100, 200, 300, 400, 500)
N_FIXED = 1200

SEQS = {"perm10000": ("perm", B.TASK_EXAMPLES), "perm1200": ("perm", N_FIXED), "rl1200": ("rl", N_FIXED)}
ACT = {"none": "R", "CHB": "CHB"}


class Doors:
    """Activation and centering state of one network (copied from pm_ledger_add_1108)."""

    def __init__(self, kind: str):
        self.kind = kind
        if kind == "CHB":
            self.mx = torch.zeros(784)
            self.m = [torch.zeros(100) for _ in range(2)]

    def inp(self, x):
        return x - self.mx if self.kind == "CHB" else x

    def phi(self, z, layer):
        if self.kind == "R":
            return torch.clamp(z, min=0.0)
        return torch.clamp(z, min=0.0) - self.m[layer]

    def dphi(self, z, layer):
        return (z > 0).to(z.dtype)

    @torch.no_grad()
    def post_update(self, params, x_raw, z1, z2, lr):
        """After the Adam step: door B, then advance the EMAs from the same forward's z and x."""
        if self.kind == "CHB":
            params[1].zero_()
            params[3].mul_(1.0 - lr * LAM_B2)
            self.m[0].mul_(1.0 - BETA).add_(torch.clamp(z1, min=0.0).mean(0), alpha=BETA)
            self.m[1].mul_(1.0 - BETA).add_(torch.clamp(z2, min=0.0).mean(0), alpha=BETA)
            self.mx.mul_(1.0 - BETA).add_(x_raw.mean(0), alpha=BETA)


def forward(params, x, doors):
    W1, b1, W2, b2, W3, b3 = params
    x = doors.inp(x)
    z1 = x @ W1.T + b1
    a1 = doors.phi(z1, 0)
    z2 = a1 @ W2.T + b2
    a2 = doors.phi(z2, 1)
    return z1, a1, z2, a2, a2 @ W3.T + b3


@torch.no_grad()
def evaluate(params, mnist, perm, doors, x_task, y_task):
    """The main run's quantities on the full test set under the task's permutation (the primary
    values), plus the task's own images (suffix _tr)."""
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
    z1t, _, z2t, _, lt = forward(params, x_task, doors)
    out["acc_tr"] = float((lt.argmax(1) == y_task).float().mean())
    for li, (tag, z) in enumerate((("l1", z1t), ("l2", z2t))):
        d = doors.dphi(z, li)
        out[f"dead_frac_{tag}_tr"] = float((d.abs().amax(0) < B.DEAD_TOL).float().mean())
        out[f"zbar_{tag}_tr"] = float(z.mean(0).median())
        out[f"zsd_{tag}_tr"] = float(z.std(0).median())
    return out


def zero_adam(params):
    return ([torch.zeros_like(q) for q in params], [torch.zeros_like(q) for q in params], [0])


def train_task(params, adam, doors, xs, ys, lr):
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


def stratified_fixed(mnist, g, n):
    """One stratified set of n training images (the box's largest-remainder quota at n)."""
    quota = B.hamilton(mnist.class_counts, n)
    picks = []
    for c in range(B.N_CLASSES):
        pool = mnist.class_idx[c]
        sel = torch.randperm(len(pool), generator=g)[: quota[c]].numpy()
        picks.append(pool[sel])
    return torch.from_numpy(np.concatenate(picks))


def epoch_order(n, g_batch):
    """10,000 sample positions: a new shuffle of the n images per epoch.  n = 10,000 gives
    exactly the main run's single randperm."""
    need = B.STEPS_PER_TASK * B.BATCH
    orders = [torch.randperm(n, generator=g_batch) for _ in range(math.ceil(need / n))]
    return torch.cat(orders)[:need]


def task_data(mnist, kind, n, idx_fixed, g_perm, g_data, g_batch, g_labels):
    perm = torch.randperm(784, generator=g_perm) if kind == "perm" else torch.arange(784)
    idx = B.stratified_draw(mnist, g_data) if n == B.TASK_EXAMPLES else idx_fixed
    order = epoch_order(n, g_batch)
    labels = (torch.randint(0, B.N_CLASSES, (n,), generator=g_labels) if kind == "rl" else mnist.train_y[idx])
    x_task = mnist.train_x[idx][:, perm]
    return perm, idx, labels, x_task, x_task[order], labels[order]


def run_loop(seq, act, seed, n_tasks, mnist, out=None, fresh_tasks=FRESH_TASKS, capture=None):
    kind, n = SEQS[seq]
    arm = f"{seq}_{act}"
    params = B.init_params(seed, torch.device("cpu"))
    adam = zero_adam(params)
    doors = Doors(ACT[act])
    g_perm, g_data, g_batch = B.stream("perm", seed), B.stream("data", seed), B.stream("batch", seed)
    g_labels = B.stream("labels", seed) if kind == "rl" else None
    idx_fixed = stratified_fixed(mnist, B.stream("data1200", seed), n) if n != B.TASK_EXAMPLES else None

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
        labs = None
        if kind == "rl":
            labs = np.lib.format.open_memmap(out / "labels.npy", mode="w+", dtype=np.int8,
                                             shape=(n_tasks + 1, n))
            labs[0] = -1
        if idx_fixed is not None:
            np.save(out / "idx.npy", idx_fixed.numpy())
    rows, fresh_rows, diverged = [], [], {"diverged": False}
    t0 = time.time()

    for t in range(1, n_tasks + 1):
        perm, idx, labels, x_task, xs, ys = task_data(mnist, kind, n, idx_fixed, g_perm, g_data, g_batch, g_labels)
        n_correct, bs = train_task(params, adam, doors, xs, ys, LR)
        if bs >= 0 or not all(torch.isfinite(p).all() for p in params):
            diverged.update(diverged=True, task=t, step=(t - 1) * B.STEPS_PER_TASK + max(bs, 0))
            rows.append({"arm": arm, "seq": seq, "act": act, "seed": seed, "task": t, "online": float("nan")})
            break
        online = n_correct / B.TASK_EXAMPLES
        row = {"arm": arm, "seq": seq, "act": act, "seed": seed, "task": t, "online": online}
        row.update(evaluate(params, mnist, perm, doors, x_task, labels))
        rows.append(row)
        if capture is not None:
            capture[t] = (params[0].detach().numpy().copy(), params[1].detach().numpy().copy(), online)
        if saving:
            W1[t] = params[0].detach().numpy()
            b1[t] = params[1].detach().numpy()
            perms[t] = perm.numpy().astype(np.int16)
            if kind == "rl":
                labs[t] = labels.numpy().astype(np.int8)
            if t % 25 == 0 or t == n_tasks:
                W1.flush(); b1.flush(); perms.flush()
                if kind == "rl":
                    labs.flush()
                B.write_csv(out / "per_task.csv", rows)
                if t % 100 == 0 or t == n_tasks:
                    print(f"[{time.time() - t0:7.1f}s] {arm} seed{seed} task {t} online {online:.4f}", flush=True)
        if t in fresh_tasks:
            F = B.init_params(seed, torch.device("cpu"))
            n_correct_f, bs_f = train_task(F, zero_adam(F), Doors(ACT[act]), xs, ys, LR)
            fresh_rows.append({"arm": arm, "seq": seq, "act": act, "seed": seed, "task": t,
                               "online_fresh": n_correct_f / B.TASK_EXAMPLES,
                               "online_cont": online, "fresh_bad_step": bs_f})
            if saving:
                B.write_csv(out / "fresh.csv", fresh_rows)
    return rows, fresh_rows, diverged, params, time.time() - t0


def set_deterministic():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def write_check(name, res):
    CHECKS_JSON.parent.mkdir(parents=True, exist_ok=True)
    cur = json.loads(CHECKS_JSON.read_text()) if CHECKS_JSON.exists() else {}
    cur[name] = res
    CHECKS_JSON.write_text(json.dumps(cur, indent=1, ensure_ascii=False, default=float))


def check_s1():
    """perm10000 reproduces the saved none and CHB runs bit for bit over tasks 1-3."""
    set_deterministic()
    mnist = B.Mnist(torch.device("cpu"))
    res, ok_all = {}, True
    for act, ref in (("none", MAIN_NONE), ("CHB", ADD_CHB)):
        saved_w = np.load(ref / "W1.npy", mmap_mode="r")
        saved_b = np.load(ref / "b1.npy", mmap_mode="r")
        with open(ref / "per_task.csv") as fh:
            saved_online = {int(r["task"]): r["online"] for r in csv.DictReader(fh)}
        cap = {}
        run_loop("perm10000", act, 0, 3, mnist, capture=cap, fresh_tasks=())
        per = {}
        for t in (1, 2, 3):
            w, b, onl = cap[t]
            per[t] = {"W1": bool(np.array_equal(w, np.asarray(saved_w[t]))),
                      "b1": bool(np.array_equal(b, np.asarray(saved_b[t]))),
                      "online": f"{onl:.10g}" == saved_online[t]}
        ok = all(all(v.values()) for v in per.values())
        res[act] = {"ok": ok, "per_task": per}
        ok_all &= ok
    print(json.dumps(res, indent=1, default=str), flush=True)
    write_check("S1", {"pass": ok_all, **res})
    if not ok_all:
        raise SystemExit("S1 failed")
    print("S1 PASS")


def check_s2():
    """rl labels: uniform, independent across tasks and seeds (arm bit-identity is in the smoke)."""
    T, S, K, n = 500, 10, 10, N_FIXED
    L = np.zeros((S, T, n), dtype=np.int8)
    for s in range(S):
        g = B.stream("labels", s)
        for t in range(T):
            L[s, t] = torch.randint(0, K, (n,), generator=g).numpy().astype(np.int8)
    nd = statistics.NormalDist()
    # (i) class frequency per seed
    cells = S * K
    z_freq = nd.inv_cdf(1 - (0.01 / cells) / 2)
    line_freq = z_freq * math.sqrt(0.09 / (T * n))
    freq = np.array([[float((L[s] == c).mean()) for c in range(K)] for s in range(S)])
    dev_freq = float(np.abs(freq - 0.1).max())
    # (ii) agreement of the labels: task pairs within a seed, seed pairs at the same task
    pairs_task = S * T * (T - 1) // 2
    pairs_seed = T * S * (S - 1) // 2
    N = pairs_task + pairs_seed
    z_ag = nd.inv_cdf(1 - (0.01 / N) / 2)
    line_ag = z_ag * math.sqrt(0.09 / n)
    max_dev_task, max_dev_seed = 0.0, 0.0
    iu = np.triu_indices(T, 1)
    for s in range(S):
        A = sum((L[s] == c).astype(np.float64) @ (L[s] == c).astype(np.float64).T for c in range(K)) / n
        max_dev_task = max(max_dev_task, float(np.abs(A[iu] - 0.1).max()))
    for t in range(T):
        for s in range(S):
            for s2 in range(s + 1, S):
                max_dev_seed = max(max_dev_seed, abs(float((L[s, t] == L[s2, t]).mean()) - 0.1))
    ok = dev_freq <= line_freq and max_dev_task <= line_ag and max_dev_seed <= line_ag
    res = {"pass": bool(ok), "freq_max_dev": dev_freq, "freq_line": line_freq, "freq_z": z_freq,
           "agree_max_dev_task_pairs": max_dev_task, "agree_max_dev_seed_pairs": max_dev_seed,
           "agree_line": line_ag, "agree_z": z_ag, "n_pairs": N}
    print(json.dumps(res, indent=1), flush=True)
    write_check("S2_labels", res)
    if not ok:
        raise SystemExit("S2 failed")
    print("S2 (labels) PASS")


def check_smoke(smoke_dir: Path):
    """3-task smoke of the 4 arms: S2(iii), S3, S4, S-fresh; prints one row per arm x task."""
    set_deterministic()
    mnist = B.Mnist(torch.device("cpu"))
    results, shown = {}, {}
    wall = {}
    for seq in ("perm1200", "rl1200"):
        for act in ("none", "CHB"):
            out = smoke_dir / f"{seq}_{act}" / "seed0"
            rows, fresh, diverged, _, w = run_loop(seq, act, 0, 3, mnist, out=out, fresh_tasks=(1,))
            B.write_csv(out / "per_task.csv", rows)
            if fresh:
                B.write_csv(out / "fresh.csv", fresh)
            results[(seq, act)] = (rows, fresh, diverged)
            wall[f"{seq}_{act}"] = w
            for r in rows:
                print(f"{seq}_{act} t{r['task']}: online {r['online']:.4f} acc_tr {r['acc_tr']:.3f} acc {r['acc']:.3f} "
                      f"zsd_l1 {r['zsd_l1']:.3f} zsd_l1_tr {r['zsd_l1_tr']:.3f} w_norm_l1 {r['w_norm_l1']:.3f} "
                      f"dead {r['dead_frac_l1']:.2f}/{r['dead_frac_l2']:.2f}", flush=True)
    def load(seq, act, name):
        return np.load(smoke_dir / f"{seq}_{act}" / "seed0" / name)
    # S3: permutations and the 1,200 images, across arms and across sequences
    s3 = {"perm1200_perm_arms": bool(np.array_equal(load("perm1200", "none", "perm.npy"), load("perm1200", "CHB", "perm.npy"))),
          "perm1200_idx_arms": bool(np.array_equal(load("perm1200", "none", "idx.npy"), load("perm1200", "CHB", "idx.npy"))),
          "rl1200_idx_arms": bool(np.array_equal(load("rl1200", "none", "idx.npy"), load("rl1200", "CHB", "idx.npy"))),
          "idx_rl_vs_perm": bool(np.array_equal(load("perm1200", "none", "idx.npy"), load("rl1200", "none", "idx.npy"))),
          "rl_perm_is_identity": bool((load("rl1200", "CHB", "perm.npy")[1:] == np.arange(784)).all()),
          "idx_unique_1200": int(np.unique(load("perm1200", "none", "idx.npy")).size) == N_FIXED}
    write_check("S3", {"pass": all(s3.values()), **s3})
    # S2 (iii): labels across arms
    s2c = {"labels_arms_bit_identical": bool(np.array_equal(load("rl1200", "none", "labels.npy"), load("rl1200", "CHB", "labels.npy")))}
    write_check("S2_arms", {"pass": all(s2c.values()), **s2c})
    # S4: rl1200 CHB rises above chance within each task
    line = 0.1 + 3.0 * math.sqrt(0.09 / N_FIXED)
    rl_online = [r["online"] for r in results[("rl1200", "CHB")][0]]
    s4 = {"line": line, "rl1200_CHB_online": rl_online, "ok": all(v > line for v in rl_online),
          "rl1200_none_online": [r["online"] for r in results[("rl1200", "none")][0]],
          "rl1200_CHB_acc_tr": [r["acc_tr"] for r in results[("rl1200", "CHB")][0]],
          "rl1200_none_acc_tr": [r["acc_tr"] for r in results[("rl1200", "none")][0]],
          "rl1200_none_dead": [(r["dead_frac_l1"], r["dead_frac_l2"]) for r in results[("rl1200", "none")][0]],
          "diverged": {f"{k[0]}_{k[1]}": v[2]["diverged"] for k, v in results.items()}, "wall_3task_s": wall}
    write_check("S4", {"pass": bool(s4["ok"]), **s4})
    # S-fresh: at task 1 the continuing net is a fresh net
    sf = {}
    for (seq, act), (rows, fresh, _) in results.items():
        sf[f"{seq}_{act}"] = bool(fresh and fresh[0]["task"] == 1 and fresh[0]["online_fresh"] == rows[0]["online"])
    write_check("S_fresh", {"pass": all(sf.values()), **sf})
    print("S3:", s3, flush=True)
    print("S2(iii):", s2c, flush=True)
    print("S4:", {k: s4[k] for k in ("line", "rl1200_CHB_online", "rl1200_none_online", "rl1200_CHB_acc_tr", "rl1200_none_acc_tr", "rl1200_none_dead", "ok")}, flush=True)
    print("S-fresh:", sf, flush=True)
    print("wall 3 tasks:", wall, flush=True)
    if not (all(s3.values()) and all(s2c.values()) and s4["ok"] and all(sf.values())):
        raise SystemExit("smoke checks failed")
    print("smoke (S2 iii, S3, S4, S-fresh) PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", choices=list(SEQS))
    ap.add_argument("--act", choices=list(ACT))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--tasks", type=int, default=500)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--check", choices=["s1", "s2", "smoke"])
    ap.add_argument("--smoke-dir", type=Path, default=DATA / "pm_rl_samebox_1108" / "smoke")
    args = ap.parse_args()
    if args.check == "s1":
        check_s1(); return
    if args.check == "s2":
        check_s2(); return
    if args.check == "smoke":
        check_smoke(args.smoke_dir); return
    set_deterministic()
    mnist = B.Mnist(torch.device("cpu"))
    rows, fresh_rows, diverged, _, wall = run_loop(args.seq, args.act, args.seed, args.tasks, mnist, out=args.out)
    B.write_csv(args.out / "per_task.csv", rows)
    if fresh_rows:
        B.write_csv(args.out / "fresh.csv", fresh_rows)
    prov = {"run_id": "pm_rl_samebox_1108", "seq": args.seq, "act": args.act, "seed": args.seed, "lr": LR,
            "lam_b2": LAM_B2, "beta": BETA, "n_tasks": args.tasks, "n_images": SEQS[args.seq][1], "dims": list(B.DIMS),
            "batch": B.BATCH, "steps_per_task": B.STEPS_PER_TASK, "git_hash": B.git_hash(), "device": "cpu",
            "torch": torch.__version__, "data_sha256": mnist.sha256, "wall_clock_s": wall,
            "max_rss_MB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "fresh_tasks": list(FRESH_TASKS),
            "diverged": diverged, "box": "src/pmnist_0905.py (loop copied from pm_ledger_add_1108; S1)"}
    (args.out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"done {args.seq}_{args.act} seed{args.seed}: {wall:.1f}s, max RSS {prov['max_rss_MB']:.0f} MB", flush=True)


if __name__ == "__main__":
    main()

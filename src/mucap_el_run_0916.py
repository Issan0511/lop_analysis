#!/usr/bin/env python3
"""Random Label MNIST, ELU -> leaky: caps on the two mu-components of the first layer.

    OMP_NUM_THREADS=1 python3 src/mucap_el_run_0916.py --arm cap_both --seeds 0 \
        --out results/mucap_el_0916/runs/cap_both_s0

H2 design note section 10.4 (obsidian-research 可塑性喪失/spec/H2_W成長抑制と負側輸送_設計案_0916.md).
The box is pmnist_rlmnist_0906's and is imported, not copied: the 1200-image subset, the task labels
and evaluate_rl come from src/pmnist_rlmnist_0906.py; init, the rng streams and leaky from the host
src/pmnist_0905.py; ELU1 from src/elu_growth_0909.py; state_sha256 and git_dirty from
src/shell_l2_rlmnist_0913.py; the caps and the mu split from src/mucap_el_0916.py.  None is modified.

The second layer is leaky 0.1 so that the first layer keeps receiving error signal while it is being
watched -- the EE box kills the second layer by task 10 and the first layer's later record is then a
record of a stopped downstream, not of the first layer.  forward2 is the host's forward with one
activation per hidden layer; S4 pins it against H.forward.

Four arms, all sharing task 1 bit for bit, each cap applied right after every Adam update from the
first update of task 2, with the radius that arm's own task-1-end row had:

    ref        -                         cap_perp   ||v_i|| <= ||v_i(t1)||
    cap_par    |q_i| <= |q_i(t1)|        cap_both   both

q_i = w_i . e is the component along the seed's mean image, so on that same fixed set
zbar_i = q_i ||mu|| + b_i exactly: the parallel cap is a cap on the mean preactivation's weight part,
the perpendicular cap is a cap on the spread around it.  This is NOT the row-mean / centered split of
wcap_rlmnist_0914 (cosine 0.62 between the axes); components() logs both.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import socket
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H                 # host; not touched
from src import pmnist_rlmnist_0906 as RL        # 0906 runner; not touched
from src import shell_l2_rlmnist_0913 as SH      # 0913 runner; not touched
from src import elu_growth_0909 as EG            # ELU1; not touched
from src import mucap_el_0916 as MU              # the caps (checks S1-S3)

EXPERIMENT = "mucap_el_0916"
ARMS = ("ref", "cap_par", "cap_perp", "cap_both")
N_IMAGES = RL.N_IMAGES                      # 1200
BATCH = RL.BATCH                            # 16
STEPS_PER_EPOCH = RL.STEPS_PER_EPOCH        # 75
EPOCHS = 80                                 # 6000 updates per task, the layer_chimera_rl_0914 box
TASKS = 150                                 # design 10.4: 50 reproduces EL's end point, 51-150 is new
DENSE = (0, 75, 375, 1500, 3000, 6000)      # updates into a task, diagnosed for tasks 2-10
TEST_TASKS = (1, 10, 50, 100, 150)          # design 10.6: the test set, never mixed with the train max
CAP_LAYER = 0                               # W1 in the host's [W1, b1, W2, b2, W3, b3]
LOWGATE = 0.05                              # design's low-response mark, as in layer_chimera_rl_0914


def arm_caps(arm: str) -> tuple[bool, bool]:
    """(cap the parallel component, cap the perpendicular component)."""
    if arm not in ARMS:
        raise ValueError(f"--arm must be one of {ARMS}, got {arm!r}")
    return arm in ("cap_par", "cap_both"), arm in ("cap_perp", "cap_both")


def forward2(params, x, act1, act2):
    """H.forward with one activation per hidden layer (S4 pins act1 is act2 against the host)."""
    W1, b1, W2, b2, W3, b3 = params
    z1 = x @ W1.T + b1
    a1 = act1.phi(z1)
    z2 = a1 @ W2.T + b2
    a2 = act2.phi(z2)
    return z1, a1, z2, a2, a2 @ W3.T + b3


def _bytes(t: torch.Tensor) -> bytes:
    return t.detach().cpu().contiguous().numpy().tobytes()


# --------------------------------------------------------------------------
# per-unit state at one diagnostic point (design 10.3 / 10.5), float64
# --------------------------------------------------------------------------

@torch.no_grad()
def unit_arrays(params, x, act1, act2, e1_64) -> dict[str, np.ndarray]:
    """Per unit of both hidden layers on the evaluation set x: the support geometry
    (zbar, sigma, U = max_x z, p+ = fraction of x with z > 0, d = zbar/sigma, K = (U - zbar)/sigma),
    the response (mean and rms phi', low-gate fraction) and the weight components in both bases.
    U = sigma (d + K) holds by construction; zero-sigma units get nan for d and K and keep z and U.
    The second layer's mu axis is recomputed here from the current a1: it moves, the first layer's
    does not, so q2 / ||v2|| are descriptive only."""
    z1, a1, z2, _, _ = forward2(params, x, act1, act2)
    out = {}
    e2_32, e2_64, mu2 = MU.mu_basis(a1)
    S2 = a1.double().sum(1)
    out["s2_mean"] = np.array([float(S2.mean())])
    out["s2_sd"] = np.array([float(S2.std(unbiased=False))])
    out["mu2_norm"] = np.array([mu2])
    for li, (z, act, k, e64) in enumerate(((z1, act1, 0, e1_64), (z2, act2, 2, e2_64)), start=1):
        z64 = z.double()
        g = act.dphi(z).double()
        zbar, sd = z64.mean(0), z64.std(0, unbiased=False)
        U = z64.amax(0)
        safe = torch.where(sd > 0, sd, torch.full_like(sd, float("nan")))
        out[f"zbar_l{li}"] = zbar.numpy()
        out[f"sigma_l{li}"] = sd.numpy()
        out[f"U_l{li}"] = U.numpy()
        out[f"pplus_l{li}"] = (z64 > 0).double().mean(0).numpy()
        out[f"d_l{li}"] = (zbar / safe).numpy()
        out[f"K_l{li}"] = ((U - zbar) / safe).numpy()
        out[f"gate_mean_l{li}"] = g.mean(0).numpy()
        out[f"gate_rms_l{li}"] = g.square().mean(0).sqrt().numpy()
        out[f"lowgate_l{li}"] = (g < LOWGATE).double().mean(0).numpy()
        out[f"b{li}"] = params[k + 1].detach().double().numpy()
        c = MU.components(params[k], e64)
        for name, v in c.items():
            out[f"{name}_l{li}"] = v
    return out


@torch.no_grad()
def adam_arrays(params, adam) -> dict[str, np.ndarray]:
    """Per unit of the capped layer: the size of the two Adam moments and of the step they make, so
    that a stop can be read against eps rather than from the gate alone (design 10.6)."""
    m, v, tc = adam
    b1, b2, eps = 0.9, 0.999, 1e-8
    mh = (m[CAP_LAYER].double() / (1 - b1 ** tc[0])) if tc[0] else torch.zeros_like(m[CAP_LAYER]).double()
    vh = (v[CAP_LAYER].double() / (1 - b2 ** tc[0])) if tc[0] else torch.zeros_like(v[CAP_LAYER]).double()
    den = vh.sqrt() + eps
    return {"mhat_rms_l1": mh.square().mean(1).sqrt().numpy(),
            "den_rms_l1": den.square().mean(1).sqrt().numpy(),
            "den_min_l1": den.amin(1).numpy(),
            "step_rms_l1": (mh / den).square().mean(1).sqrt().numpy(),
            "eps_frac_l1": (vh.sqrt() < eps).double().mean(1).numpy()}


# --------------------------------------------------------------------------
# the per-update ledger (design 10.5), float64, accumulated over a task
# --------------------------------------------------------------------------

LEDGER_COMPS = ("q", "v2", "m", "wt2")
LEDGER_KEYS = tuple(f"{part}_{comp}_{term}" for part in ("adam", "proj")
                    for comp in LEDGER_COMPS for term in ("align", "sq")) + \
    tuple(f"{part}_{comp}_d" for part in ("adam", "proj") for comp in ("q", "m")) + \
    tuple(f"traffic_{comp}" for comp in LEDGER_COMPS)   # sum of |increment|: the books' condition
                                                        # number, and the scale S6's residual is read
                                                        # against (a net change can be ~0 under a cap)


@torch.no_grad()
def _point(W: torch.Tensor, e64: torch.Tensor) -> tuple:
    """(W in float64, q = W.e, m = row mean).  v and W~ are never materialised: with <e,e> = 1 and
    <1,1> = d, both of their books follow from q, m, <w, dw> and |dw|^2 --
        2<v0, dv> = 2<w0, dw> - 2 q0 dq,        |dv|^2 = |dw|^2 - dq^2,
        2<W~0, dW~> = 2<w0, dw> - 2 d m0 dm,    |dW~|^2 = |dw|^2 - d dm^2,
    which halves the ledger's cost.  S6 checks the books close against the task's endpoints, so a
    wrong term here shows up there."""
    W64 = W.double()
    return W64, W64 @ e64, W64.mean(1)


@torch.no_grad()
def _add_step(acc: dict, part: str, p0: tuple, p1: tuple, d_in: int) -> None:
    (W0, q0, m0), (W1, q1, m1) = p0, p1
    dq, dm = q1 - q0, m1 - m0
    dW = W1 - W0
    dot = (W0 * dW).sum(1)
    sq = (dW * dW).sum(1)
    acc[f"{part}_q_align"] += 2 * q0 * dq
    acc[f"{part}_q_sq"] += dq * dq
    acc[f"{part}_q_d"] += dq
    acc[f"{part}_v2_align"] += 2 * (dot - q0 * dq)
    acc[f"{part}_v2_sq"] += sq - dq * dq
    acc[f"{part}_m_align"] += 2 * m0 * dm
    acc[f"{part}_m_sq"] += dm * dm
    acc[f"{part}_m_d"] += dm
    acc[f"{part}_wt2_align"] += 2 * (dot - d_in * m0 * dm)
    acc[f"{part}_wt2_sq"] += sq - d_in * dm * dm
    acc["traffic_q"] += (2 * q0 * dq + dq * dq).abs()
    acc["traffic_v2"] += (2 * (dot - q0 * dq) + sq - dq * dq).abs()
    acc["traffic_m"] += (2 * m0 * dm + dm * dm).abs()
    acc["traffic_wt2"] += (2 * (dot - d_in * m0 * dm) + sq - d_in * dm * dm).abs()


@torch.no_grad()
def ledger_add(acc: dict, W_before, W_mid, W_after, e64, capped: bool) -> None:
    """One update's contribution, split into the Adam step (before -> mid) and the cap's correction
    (mid -> after).  Each squared quantity is split as  delta(n^2) = 2 <n, dn> + |dn|^2, so the two
    terms sum to the exact change; q and m also get their signed displacement, which the squares
    cannot show.  With no cap on, mid is after and the projection book is left at zero."""
    d_in = W_before.shape[1]
    pb, pm = _point(W_before, e64), _point(W_mid, e64)
    _add_step(acc, "adam", pb, pm, d_in)
    if capped:
        _add_step(acc, "proj", pm, _point(W_after, e64), d_in)


def new_ledger(rows: int) -> dict:
    return {k: torch.zeros(rows, dtype=torch.float64) for k in LEDGER_KEYS}


# --------------------------------------------------------------------------

def run_one(arm_s: str, seed: int, lr: float, n_tasks: int, mnist: H.Mnist, device: torch.device,
            epochs: int = EPOCHS, ledger: bool = True, debug: dict | None = None,
            progress: bool = False):
    """debug (checks only): init, subset, labels, batch orders, the task-1-end state and, at the
    (task, step) pairs in debug['capture'], the state before and after that single update."""
    t_start = time.time()
    act1, act2 = EG.ELU(1.0), H.ARMS["LR"]
    params = H.init_params(seed, device)               # host init: bit-identical per seed
    do_par, do_perp = arm_caps(arm_s)
    capture = debug.get("capture", ()) if debug is not None else ()
    adam = ([torch.zeros_like(q) for q in params],
            [torch.zeros_like(q) for q in params], [0])

    idx = RL.subset_idx(seed).to(device)
    x = mnist.train_x[idx]                             # the task's inputs, fixed forever
    xt = mnist.test_x.to(device)
    e1, e1_64, mu1 = MU.mu_basis(x)                    # fixed: the images never change
    if e1 is None:
        raise RuntimeError("mu = 0 on this subset: the parallel axis is undefined (design 10.4)")
    g_lab, g_batch = H.stream("rl_labels", seed), H.stream("rl_batch", seed)
    if debug is not None:
        debug |= {"init": [q.detach().cpu().clone() for q in params], "subset": idx.cpu().clone(),
                  "e1": e1.cpu().clone()}
    h_lab, h_batch = hashlib.sha256(), hashlib.sha256()
    spt = STEPS_PER_EPOCH * epochs
    q_cap = v_cap = None
    rows, led_rows, diag = [], [], {"task": [], "step": []}
    info = {"init_sha256": hashlib.sha256(b"".join(_bytes(q) for q in params)).hexdigest(),
            "subset_idx_sha256": hashlib.sha256(_bytes(idx)).hexdigest(),
            "mu1_norm": mu1, "cos_mu1_to_uniform": float(e1_64.sum() / np.sqrt(e1_64.numel()))}
    diverged = {"diverged": False, "task": None, "step": None, "seed": seed, "arm": arm_s}

    def snap(t: int, s: int) -> None:
        u = unit_arrays(params, x, act1, act2, e1_64) | adam_arrays(params, adam)
        if t in TEST_TASKS and s == spt:                # the task's end point, however it was reached
            ut = unit_arrays(params, xt, act1, act2, e1_64)
            u |= {f"test_{k}": v for k, v in ut.items()}
        diag["task"].append(t)
        diag["step"].append(s)
        for k, v in u.items():
            diag.setdefault(k, []).append(v)

    for t in range(1, n_tasks + 1):
        y = RL.task_labels(g_lab).to(device)           # new labelling, same images
        h_lab.update(_bytes(y))
        if debug is not None:
            debug.setdefault("labels", []).append(y.cpu().clone())
        cap_on = t >= 2 and (do_par or do_perp)
        acc_sum = torch.zeros((), device=device)
        bad_step = torch.full((), -1, dtype=torch.long, device=device)
        proj = {"rows_par": 0, "rows_perp": 0, "rem_par": 0.0, "rem_perp": 0.0}
        acc = new_ledger(params[CAP_LAYER].shape[0]) if ledger else None
        dense = set(DENSE) if 2 <= t <= 10 else {0}
        if 0 in dense:
            snap(t, 0)                        # new labels, weights not yet moved

        for ep in range(epochs):
            order = torch.randperm(N_IMAGES, generator=g_batch).to(device)
            h_batch.update(_bytes(order))
            xs, ys = x[order], y[order]                # reshuffled every epoch
            for j in range(STEPS_PER_EPOCH):
                s = ep * STEPS_PER_EPOCH + j
                if (t, s) in capture:
                    m_, v_, tc_ = adam
                    debug.setdefault("steps", {})[(t, s)] = {
                        "before": {"params": [q.detach().clone() for q in params],
                                   "m": [q.clone() for q in m_], "v": [q.clone() for q in v_], "tc": tc_[0]},
                        "xb": xs[j * BATCH:(j + 1) * BATCH].clone(),
                        "yb": ys[j * BATCH:(j + 1) * BATCH].clone(), "cap_on": cap_on,
                        "q_cap": None if q_cap is None else q_cap.clone(),
                        "v_cap": None if v_cap is None else v_cap.clone()}
                xb, yb = xs[j * BATCH:(j + 1) * BATCH], ys[j * BATCH:(j + 1) * BATCH]
                out = forward2(params, xb, act1, act2)
                loss = torch.nn.functional.cross_entropy(out[4], yb)
                # pre-update accuracy: the argmax of the very forward pass the loss came from
                acc_sum += (out[4].detach().argmax(1) == yb).float().mean()
                grads = torch.autograd.grad(loss, params)
                with torch.no_grad():
                    bad = ~torch.isfinite(loss)
                    bad_step = torch.where((bad_step < 0) & bad, torch.tensor(s, device=device), bad_step)
                    W_before = params[CAP_LAYER].detach().clone() if acc is not None else None
                    m, v, tc = adam
                    tc[0] += 1
                    b1, b2, eps = 0.9, 0.999, 1e-8
                    c1, c2 = 1 - b1 ** tc[0], 1 - b2 ** tc[0]
                    for p, gr, mi, vi in zip(params, grads, m, v):
                        mi.mul_(b1).add_(gr, alpha=1 - b1)
                        vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                        p -= lr * (mi / c1) / ((vi / c2).sqrt() + eps)
                    W_mid = params[CAP_LAYER].detach().clone() if acc is not None else None
                    if cap_on:
                        if do_par:
                            nr, rem = MU.cap_parallel_(params[CAP_LAYER], e1, q_cap)
                            proj["rows_par"] += int(nr)
                            proj["rem_par"] += float(rem)
                        if do_perp:
                            nr, rem = MU.cap_perp_(params[CAP_LAYER], e1, v_cap)
                            proj["rows_perp"] += int(nr)
                            proj["rem_perp"] += float(rem)
                    if acc is not None:
                        ledger_add(acc, W_before, W_mid, params[CAP_LAYER].detach(), e1_64,
                                   cap_on and (do_par or do_perp))
                if (t, s) in capture:
                    m_, v_, tc_ = adam
                    debug["steps"][(t, s)]["after"] = {
                        "params": [q.detach().clone() for q in params],
                        "m": [q.clone() for q in m_], "v": [q.clone() for q in v_], "tc": tc_[0]}
                if s + 1 in dense:
                    snap(t, s + 1)

        bs = int(bad_step)
        if bs >= 0 or not all(torch.isfinite(p).all() for p in params):
            diverged.update(diverged=True, task=t, step=(t - 1) * spt + max(bs, 0))
            rows.append({"arm": arm_s, "seed": seed, "task": t, "online_acc": float("nan")})
            break                                      # drop, never rescue
        if t == 1:
            info["task1_end_state_sha256"] = SH.state_sha256(params, adam, act1)
            q_cap = MU.parallel_cap(params[CAP_LAYER], e1)
            v_cap = MU.perp_cap(params[CAP_LAYER], e1)
            if debug is not None:
                debug["task1_end_params"] = [q.detach().cpu().clone() for q in params]
            info["zero_q_cap_rows"] = int((q_cap == 0).sum())
            info["zero_v_cap_rows"] = int((v_cap == 0).sum())
        if spt not in dense:
            snap(t, spt)
        mt = RL.evaluate_rl(params, x, y, act1)        # act1 only sets the reported per-layer gates
        u = unit_arrays(params, x, act1, act2, e1_64)
        rows.append({"arm": arm_s, "seed": seed, "task": t, "online_acc": float(acc_sum) / spt,
                     "memo_acc": mt["acc"], "cap_on": int(cap_on), **proj,
                     **{f"med_{k}": float(np.nanmedian(v)) for k, v in u.items() if v.size == 100}})
        if acc is not None:
            led_rows.append({"arm": arm_s, "seed": seed, "task": t,
                             **{k: acc[k].numpy() for k in LEDGER_KEYS}})
        if progress:
            print(f"[{time.time() - t_start:8.1f}s] {arm_s} seed={seed} task {t}/{n_tasks}", flush=True)

    # float64, not float32: S6 closes the per-update ledger against these end points, and a float32
    # round trip would swamp the residual it is testing.
    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in diag.items()}
    arrays["q_cap"] = (q_cap[:, 0].double().numpy() if q_cap is not None else np.full(100, np.nan))
    arrays["v_cap"] = (v_cap[:, 0].double().numpy() if v_cap is not None else np.full(100, np.nan))
    led = {k: np.stack([r[k] for r in led_rows]).astype(np.float64) for k in LEDGER_KEYS} if led_rows else {}
    led["task"] = np.array([r["task"] for r in led_rows], dtype=np.int64) if led_rows else np.zeros(0, np.int64)
    info |= {"labels_sha256": h_lab.hexdigest(), "batch_sha256": h_batch.hexdigest(),
             "final_state_sha256": SH.state_sha256(params, adam, act1),
             "tasks_completed": len([r for r in rows if r["online_acc"] == r["online_acc"]]),
             "wall_clock_s": time.time() - t_start, "divergence": diverged}
    return rows, arrays, led, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tasks", type=int, default=TASKS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = H.setup(args.device)
    mnist = H.Mnist(device)
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    me = Path(__file__).resolve()
    t0 = time.time()
    all_rows, all_arrays, all_led, infos = [], {}, {}, {}
    for seed in seeds:
        rows, arrays, led, info = run_one(args.arm, seed, args.lr, args.tasks, mnist, device,
                                          epochs=args.epochs, ledger=not args.no_ledger, progress=True)
        all_rows += rows
        infos[str(seed)] = info
        for k, v in arrays.items():
            all_arrays[f"s{seed}_{k}"] = v
        for k, v in led.items():
            all_led[f"s{seed}_{k}"] = v
    import csv
    keys = sorted({k for r in all_rows for k in r})
    with (out / "per_task.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "seed", "task"] +
                           [k for k in keys if k not in ("arm", "seed", "task")])
        w.writeheader()
        w.writerows(all_rows)
    np.savez_compressed(out / "units.npz", **all_arrays)
    if all_led:
        np.savez_compressed(out / "ledger.npz", **all_led)
    (out / "provenance.json").write_text(json.dumps({
        "experiment": EXPERIMENT, "spec": "H2 design note section 10.4",
        "git_hash": SH.git_hash() if hasattr(SH, "git_hash") else None,
        "git_dirty_code": SH.git_dirty(["src", "analysis/mucap_el_0916"]),
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "torch": torch.__version__, "python": sys.version.split()[0], "device": args.device,
        "threads": args.threads, "arm": args.arm, "seeds": seeds, "lr": args.lr,
        "n_tasks": args.tasks, "epochs_per_task": args.epochs, "batch": BATCH,
        "steps_per_task": STEPS_PER_EPOCH * args.epochs, "n_images": N_IMAGES,
        "act1": "ELU1", "act2": "LR", "cap_layer": CAP_LAYER, "ledger": not args.no_ledger,
        "code_sha256": {"src/mucap_el_run_0916.py": SH.file_sha256(me),
                        "src/mucap_el_0916.py": SH.file_sha256(me.parent / "mucap_el_0916.py"),
                        "src/pmnist_0905.py": SH.file_sha256(me.parent / "pmnist_0905.py"),
                        "src/pmnist_rlmnist_0906.py": SH.file_sha256(me.parent / "pmnist_rlmnist_0906.py"),
                        "src/elu_growth_0909.py": SH.file_sha256(me.parent / "elu_growth_0909.py")},
        "data_sha256": mnist.sha256, "per_seed": infos,
        "wall_clock_s": time.time() - t0,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }, indent=2, default=str))
    print(f"wrote {out}  ({len(all_rows)} task rows, {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()

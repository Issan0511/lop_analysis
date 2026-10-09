#!/usr/bin/env python3
"""Leaky ReLU / ReLU on the bundled CNN engine of sna_cnn_cause_1009 (phase 2 of cnn_drive_verify_1009).

    python src/cnn_drive_verify_1009_lr.py checks --out results/cnn_drive_verify_1009/LR/checks
    python src/cnn_drive_verify_1009_lr.py run --arm LR --seeds 10-19 --tasks 30 --out results/cnn_drive_verify_1009/LR
    python src/cnn_drive_verify_1009_lr.py run --arm R --seeds 10 --tasks 3 --out results/cnn_drive_verify_1009/R

The engine -- forward, Adam, CUDA graph, the task loop `run`, per-task evaluation and the
checkpoints -- is the sna engine (`wt/sna_cnn_cause_1009/src/sna_cnn_cause_1009.py`), imported
unchanged.  Only the activation object is replaced: `BundleLeaky` has BundleSnake's interface
(phi / dphi / update / state / load_state / alpha / freeze) with
    LR: phi = where(z > 0, z, 0.1 z)   (host pmnist_0905.ARMS["LR"], same op)
    R : phi = clamp(z, min=0)          (host pmnist_0905.ARMS["R"],  same op)
There is no alpha: `alpha()` is NaN, so the Snake-only columns of `evaluate` (seat, two_alpha_*)
come out NaN; V (the EMA of the batch variance) is kept as a record only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SNA_ROOT = Path("/home/issan/Projects/claude/wt/sna_cnn_cause_1009")
sys.path.insert(0, str(SNA_ROOT))

import torch                                         # noqa: E402
import torch.nn.functional as F                      # noqa: E402

from src import pmnist_0905 as H                     # noqa: E402  (sna worktree, read only)
from src import rlcifar_cnn_0908 as CN               # noqa: E402
from src import pmnist_rlcifar_0907 as RC            # noqa: E402
from src import sna_cnn_cause_1009 as E              # noqa: E402

SLOPE = {"LR": 0.1, "R": 0.0}
WIDTHS, IS_CONV = CN.WIDTHS, CN.IS_CONV
CKPT_TASKS = (1, 2, 3, 5, 10, 20, 30)


class BundleLeaky:
    """Leaky ReLU (slope > 0) or ReLU (slope = 0) for R bundled runs, BundleSnake's interface."""

    def __init__(self, R: int, slope: float, device, beta: float = 0.01):
        self.R, self.slope, self.beta = R, float(slope), beta
        self.V = [torch.ones(R, w, device=device) for w in WIDTHS]          # record only
        self.ada = [torch.zeros(R, w, dtype=torch.bool, device=device) for w in WIDTHS]
        self.cval = [torch.ones(R, 1, device=device) for _ in WIDTHS]
        self.fixA = [torch.full((R, w), float("nan"), device=device) for w in WIDTHS]
        self.pp = torch.zeros(R, 1, dtype=torch.bool, device=device)
        self.any_pp = False
        self.amp = None

    def alpha(self, l: int) -> torch.Tensor:                           # no alpha: NaN
        return self.fixA[l]

    def phi(self, z: torch.Tensor, l: int) -> torch.Tensor:
        if self.slope == 0.0:
            return torch.clamp(z, min=0.0)                              # host "R"
        return torch.where(z > 0, z, self.slope * z)                    # host "LR"

    def dphi(self, z: torch.Tensor, l: int) -> torch.Tensor:
        if self.slope == 0.0:
            return (z > 0).to(z.dtype)
        return torch.where(z > 0, torch.ones_like(z), torch.full_like(z, self.slope))

    @torch.no_grad()
    def batch_var(self, z: torch.Tensor, l: int) -> torch.Tensor:
        if not IS_CONV[l]:
            return z.var(1, unbiased=False)
        return z.var((0, 2, 3), unbiased=False).view(self.R, -1)

    @torch.no_grad()
    def update(self, zs) -> None:
        for l, z in enumerate(zs):
            self.V[l].mul_(1 - self.beta).add_(self.beta * self.batch_var(z, l))

    @torch.no_grad()
    def freeze(self, runs) -> None:                                     # nothing to freeze
        return None

    def state(self) -> dict:
        return {k: [t.clone() for t in getattr(self, k)] for k in ("V", "ada", "cval", "fixA")}

    def load_state(self, st: dict) -> None:
        for k in ("V", "ada", "cval", "fixA"):
            for dst, src in zip(getattr(self, k), st[k]):
                dst.copy_(src)


class BundleLR(E.Bundle):
    """The sna engine's Bundle with the activation replaced (constructor copied, every method
    inherited).  All slots must share one arm (LR or R)."""

    def __init__(self, slots, cifar, device, lr=1e-3, epochs=400, lo=0.005, hi=3.0, beta=0.01,
                 graph=True):
        arms = {a for a, s in slots}
        assert len(arms) == 1 and arms <= set(SLOPE), arms
        (arm,) = arms
        self.slots, self.device, self.lr, self.epochs = slots, device, lr, epochs
        self.R = R = len(slots)
        self.specs = [dict(site=(("leaky", SLOPE[arm]),) * 4, frz=None, pp=False, fixconv=False,
                           fixfc=False, fcamp=1.0) for _ in slots]
        seeds = [s for a, s in slots]
        self.useeds = sorted(set(seeds))
        self.P = E.stack_init(seeds, device)
        self.act = BundleLeaky(R, SLOPE[arm], device, beta=beta)
        self.m = [torch.zeros_like(p) for p in self.P]
        self.v = [torch.zeros_like(p) for p in self.P]
        self.tc = torch.zeros((), dtype=torch.float64, device=device)
        self.X = torch.stack([CN.images(cifar, CN.subset_idx(s), device) for s in seeds])
        self.Y = torch.zeros(R, CN.N_IMAGES, dtype=torch.long, device=device)
        self.g_lab = {s: H.stream("rlcc_labels", s) for s in self.useeds}
        self.g_batch = {s: H.stream("rlcc_batch", s) for s in self.useeds}
        self.order = torch.zeros(R, CN.N_IMAGES, dtype=torch.long, device=device)
        self.acc_ep = torch.zeros(R, device=device)
        self.bad = torch.zeros(R, dtype=torch.bool, device=device)
        self.ar = torch.arange(R, device=device)[:, None]
        self.b1 = torch.tensor(0.9, dtype=torch.float64, device=device)
        self.b2 = torch.tensor(0.999, dtype=torch.float64, device=device)
        self.graph = graph and device.type == "cuda"
        self.cg = None
        self.upd_mask = None


_orig_parse_arm = E.parse_arm


def parse_arm(name: str) -> dict:
    if name in SLOPE:
        return {"site": (("leaky", SLOPE[name]),) * 4, "frz": None, "pp": False, "fixconv": False,
                "fixfc": False, "fcamp": 1.0, "activation": "leaky" if SLOPE[name] else "relu",
                "slope": SLOPE[name]}
    return _orig_parse_arm(name)


def install() -> None:
    """Point the engine's run loop at the leaky bundle (in this process only; no file is touched)."""
    E.Bundle = BundleLR
    E.parse_arm = parse_arm


# --------------------------------------------------------------------------
# checks (L-host, L-graph, L-indep), after sna_cnn_cause_1009_checks.py
# --------------------------------------------------------------------------

def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


def check_host(cifar, device) -> dict:
    """R = 1 bundle against the host's forward and Adam loop: one step, then one epoch."""
    res = {}
    for arm, s in (("LR", 10), ("R", 12)):
        B = BundleLR([(arm, s)], cifar, device, graph=False)
        B.new_labels(); B.new_order()
        params = CN.init_params(s, device)
        act = H.ARMS[arm]
        init_eq = all(bool((p.detach() == q.detach()[0]).all()) for p, q in zip(params, B.P))
        x = B.X[0]; y = B.Y[0]; order = B.order[0]
        idx = order[:CN.BATCH]
        out_h = CN.forward_cnn(params, x[idx], act)
        lh = F.cross_entropy(out_h[8], y[idx])
        gh = torch.autograd.grad(lh, params)
        out = E.forward(B.P, B.X[:, idx], B.act)
        lb = F.cross_entropy(out[8][0], y[idx])
        gb = torch.autograd.grad(lb, B.P)
        step_rel = max(rel(a[0], b) for a, b in zip(gb, gh))
        loss_rel = abs(float(lb) - float(lh)) / abs(float(lh))
        m = [torch.zeros_like(q) for q in params]; v = [torch.zeros_like(q) for q in params]
        tc = 0
        growth = []
        for j in range(CN.STEPS_PER_EPOCH):                     # host Adam loop (run_one)
            ib = order[j * CN.BATCH:(j + 1) * CN.BATCH]
            o = CN.forward_cnn(params, x[ib], act)
            loss = F.cross_entropy(o[8], y[ib])
            grads = torch.autograd.grad(loss, params)
            with torch.no_grad():
                tc += 1
                c1 = 1 - 0.9 ** tc; c2 = 1 - 0.999 ** tc
                for p, gr, mi, vi in zip(params, grads, m, v):
                    mi.mul_(0.9).add_(gr, alpha=1 - 0.9)
                    vi.mul_(0.999).addcmul_(gr, gr, value=1 - 0.999)
                    p -= 1e-3 * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
            B.step(j)                                          # the engine's step, same batch
            growth.append(max(float((b.detach()[0] - a.detach()).abs().max())
                              for a, b in zip(params, B.P)))
        ep_rel = max(rel(b.detach()[0], a.detach()) for a, b in zip(params, B.P))
        ep_absmax = growth[-1]
        # pass = the requested criterion: one step's loss and every gradient agree (1e-4).  Over an
        # epoch the float32 rounding difference (bmm vs mm in the fc layers, 1e-7 per step) is
        # amplified by Adam; it grows smoothly from 7.5e-9 with no jump, faster at the leaky/ReLU
        # kinks than for a smooth activation (reported, not judged; see `growth`).
        ok = init_eq and loss_rel < 1e-5 and step_rel < 1e-4
        res[arm] = {"init_equal": init_eq, "loss_rel": loss_rel, "grad_rel_max": step_rel,
                    "epoch_param_rel_max": ep_rel, "epoch_param_absmax": ep_absmax,
                    "growth_absmax_steps_1_2_3_5_10_20_40_75": [growth[k] for k in (0, 1, 2, 4, 9, 19, 39, 74)],
                    "pass": ok}
    # control: the same step-by-step comparison for a smooth activation through the sna engine
    # itself (SN3, fixed alpha 3), to show the epoch drift is Adam-amplified rounding, not the
    # leaky bundle
    B = E.Bundle([("SN3", 10)], cifar, device, graph=False)
    B.new_labels(); B.new_order()
    params = CN.init_params(10, device)

    class _S:
        def phi(self, z):
            return z + torch.sin(3.0 * z) ** 2 / 3.0
    act = _S()
    x = B.X[0]; y = B.Y[0]; order = B.order[0]
    m = [torch.zeros_like(q) for q in params]; v = [torch.zeros_like(q) for q in params]
    tc = 0; growth = []
    for j in range(CN.STEPS_PER_EPOCH):
        ib = order[j * CN.BATCH:(j + 1) * CN.BATCH]
        o = CN.forward_cnn(params, x[ib], act)
        loss = F.cross_entropy(o[8], y[ib])
        grads = torch.autograd.grad(loss, params)
        with torch.no_grad():
            tc += 1
            c1 = 1 - 0.9 ** tc; c2 = 1 - 0.999 ** tc
            for p, gr, mi, vi in zip(params, grads, m, v):
                mi.mul_(0.9).add_(gr, alpha=1 - 0.9)
                vi.mul_(0.999).addcmul_(gr, gr, value=1 - 0.999)
                p -= 1e-3 * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
        B.step(j)
        growth.append(max(float((b.detach()[0] - a.detach()).abs().max()) for a, b in zip(params, B.P)))
    res["control_SN3_growth_absmax_steps_1_2_3_5_10_20_40_75"] = [growth[k] for k in (0, 1, 2, 4, 9, 19, 39, 74)]
    res["pass"] = all(r["pass"] for r in res.values() if isinstance(r, dict))
    return res


def check_indep(cifar, device) -> dict:
    slots = [("LR", 10), ("LR", 11), ("LR", 12)]
    B = BundleLR(slots, cifar, device, graph=False)
    B.new_labels(); B.new_order()
    idx = B.order[:, :CN.BATCH]

    def lg():
        out = E.forward(B.P, B.X[B.ar, idx], B.act)
        lv = F.cross_entropy(out[8].reshape(-1, 10), B.Y[B.ar, idx].reshape(-1),
                             reduction="none").view(3, -1).mean(1)
        g = torch.autograd.grad(lv.sum(), B.P)
        return lv.detach(), [q.detach().clone() for q in g]

    l0, g0 = lg()
    with torch.no_grad():
        for p in B.P:
            p[1] += 0.05 * torch.randn_like(p[1])
    l1, g1 = lg()
    same_p = all(bool((a[r] == b[r]).all()) for a, b in zip(g0, g1) for r in (0, 2)) \
        and bool((l0[[0, 2]] == l1[[0, 2]]).all())
    moved_p = bool(l0[1] != l1[1]) and any(bool((a[1] != b[1]).any()) for a, b in zip(g0, g1))
    with torch.no_grad():
        B.X[1] = torch.rand_like(B.X[1])
    l2, g2 = lg()
    same_x = all(bool((a[r] == b[r]).all()) for a, b in zip(g1, g2) for r in (0, 2)) \
        and bool((l1[[0, 2]] == l2[[0, 2]]).all())
    moved_x = bool(l1[1] != l2[1])
    return {"others_bit_identical_param": same_p, "run1_changed_param": moved_p,
            "others_bit_identical_input": same_x, "run1_changed_input": moved_x,
            "pass": same_p and moved_p and same_x and moved_x}


def check_graph(cifar, device) -> dict:
    slots = [("LR", 10), ("LR", 11), ("LR", 12), ("LR", 13)]
    Bg = BundleLR(slots, cifar, device, graph=True)
    Be = BundleLR(slots, cifar, device, graph=False)
    for B in (Bg, Be):
        B.new_labels()
        accs = []
        for e in range(2):
            B.run_epoch()
            accs.append(B.acc_ep.clone())
        B._accs = accs
    tens_g = (*Bg._state_tensors(), *Bg.act.V)
    tens_e = (*Be._state_tensors(), *Be.act.V)
    eq = all(bool((a == b).all()) for a, b in zip(tens_g, tens_e))
    eq_acc = all(bool((a == b).all()) for a, b in zip(Bg._accs, Be._accs))
    moved = bool((Bg.P[0] != E.stack_init([s for a, s in slots], device)[0]).any())
    maxdiff = max(float((a.double() - b.double()).abs().max()) for a, b in zip(tens_g, tens_e))
    return {"bit_identical": eq, "acc_identical": eq_acc, "params_moved": moved,
            "max_abs_diff": maxdiff, "pass": eq and eq_acc and moved}


def check_eval(cifar, device) -> dict:
    """`evaluate` runs on the leaky bundle; the Snake-only columns are NaN, the rest finite."""
    B = BundleLR([("LR", 10), ("LR", 11)], cifar, device, graph=False)
    B.new_labels()
    rows = E.evaluate(B.P, B.X, B.Y, B.act)
    nan_cols = {k for r in rows for k, v in r.items() if isinstance(v, float) and v != v}
    snake_only = {k for k in rows[0] if k.startswith(("seat_", "alpha_med_", "two_alpha_"))}
    ok = nan_cols <= snake_only                      # only the Snake-only columns are NaN
    return {"nan_columns": sorted(nan_cols), "pass": bool(ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["checks", "run"])
    ap.add_argument("--arm", default="LR", choices=sorted(SLOPE))
    ap.add_argument("--seeds", default="10-19")
    ap.add_argument("--tasks", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mem-gb", type=float, default=2.4)
    args = ap.parse_args()
    device = H.setup("cuda")
    di = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(di).total_memory / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_gb / total), di)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.mode == "checks":
        cifar = RC.Cifar10()
        rep = {}
        for name, fn in (("L-host", check_host), ("L-indep", check_indep), ("L-graph", check_graph),
                         ("L-eval", check_eval)):
            t0 = time.time()
            rep[name] = fn(cifar, device) | {"seconds": time.time() - t0}
            print(name, json.dumps(rep[name], default=float), flush=True)
        rep["all_pass"] = all(v["pass"] for v in rep.values())
        (out / "checks.json").write_text(json.dumps(rep, indent=1, default=float))
        print("all_pass", rep["all_pass"])
        return
    install()
    seeds = E.parse_seeds(args.seeds)
    slots = [(args.arm, s) for s in seeds]
    E.run(slots, out, args.tasks, args.epochs, device, lr=args.lr, ckpt_tasks=CKPT_TASKS, graph=True)
    import subprocess
    head = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    (out / "provenance_cnn_drive_verify.json").write_text(json.dumps(
        {"engine": "sna_cnn_cause_1009 Bundle/run with BundleLeaky (this file)", "arm": args.arm,
         "slope": SLOPE[args.arm], "seeds": seeds, "tasks": args.tasks, "epochs": args.epochs,
         "lr": args.lr, "git_hash_cnn_drive_verify": head, "torch": torch.__version__,
         "tf32_cudnn": torch.backends.cudnn.allow_tf32}, indent=1))


if __name__ == "__main__":
    main()

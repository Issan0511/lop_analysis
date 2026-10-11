#!/usr/bin/env python3
"""Leaky CNN with the input centred (cnn_center_1010).

    python3 src/cnn_center_1010.py checks --out results/_checks_cnn_center_1010
    python3 src/cnn_center_1010.py run --out results/cnn_center_1010/LRc

The reference is the leaky arm of cnn_drive_verify_1009 (`results/cnn_drive_verify_1009/LR`: the
same engine, seeds 10-19, 30 tasks x 400 epochs, Adam 1e-3, batch 16).  The only change is the
input: every image has the CIFAR-10 training-set channel mean subtracted (x - mu_c, mu over the
50,000 training images, a constant independent of the seed).  The scale is not touched, so the
patch covariance is unchanged and only the mean patch m (|m| = 3.78 for the raw input, along the
covariance's first principal direction) goes to ~0.  The conv's zero padding then pads with the
mean colour.

Everything else -- init, image subset, label and batch streams, forward, Adam, CUDA graph, the
task loop, evaluation, checkpoints -- is the sna engine (`src/sna_cnn_cause_1009.py`) with the
leaky bundle of `src/cnn_drive_verify_1009_lr.py`, imported unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
import torch                                         # noqa: E402
import torch.nn.functional as F                      # noqa: E402

from src import pmnist_0905 as H                     # noqa: E402
from src import pmnist_rlcifar_0907 as RC            # noqa: E402
from src import rlcifar_cnn_0908 as CN               # noqa: E402
from src import sna_cnn_cause_1009 as E              # noqa: E402
from src import cnn_drive_verify_1009_lr as LRM      # noqa: E402

ARM = "LRc"
LRM.SLOPE[ARM] = 0.1                                 # leaky 0.1, the reference arm's slope
CKPT_TASKS = (1, 2, 3, 5, 10, 20, 30)
REF = ROOT / "results" / "cnn_drive_verify_1009" / "LR" / "per_task.csv"


def train_channel_mean(cifar) -> torch.Tensor:
    """(3,) float32 mean of each colour channel over the 50,000 training images, /255 scale."""
    x = cifar.train_u8.reshape(-1, 3, 1024).double() / 255.0
    return x.mean((0, 2)).float()


class BundleLRc(LRM.BundleLR):
    """The leaky bundle with every image shifted by -mu (only for the arm "LRc")."""
    center: torch.Tensor | None = None

    def __init__(self, slots, cifar, device, **kw):
        super().__init__(slots, cifar, device, **kw)
        arms = {a for a, s in slots}
        if ARM in arms:
            assert arms == {ARM}
            assert BundleLRc.center is not None, "set BundleLRc.center before building an LRc bundle"
            with torch.no_grad():
                self.X.sub_(BundleLRc.center.to(device).view(1, 1, 3, 1, 1))


def install(cifar) -> None:
    BundleLRc.center = train_channel_mean(cifar)
    E.Bundle = BundleLRc
    E.parse_arm = LRM.parse_arm


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def mean_patch(X: torch.Tensor, chunk: int = 100) -> torch.Tensor:
    """(R, 75) mean 5x5x3 patch over the 1200 images x 1024 positions (zero padding, as the conv)."""
    out = []
    for r in range(X.shape[0]):
        acc = torch.zeros(75, dtype=torch.float64, device=X.device)
        for i0 in range(0, X.shape[1], chunk):
            acc += F.unfold(X[r, i0:i0 + chunk], 5, padding=2).double().sum((0, 2))
        out.append(acc / (X.shape[1] * 1024))
    return torch.stack(out)


def check_center(cifar, device) -> dict:
    """The manipulation: the mean patch shrinks from |m| ~ 3.8 to ~0, the covariance is unchanged."""
    seeds = list(range(10, 20))
    Br = LRM.BundleLR([("LR", s) for s in seeds], cifar, device, graph=False)
    Bc = BundleLRc([(ARM, s) for s in seeds], cifar, device, graph=False)
    mr, mc = mean_patch(Br.X).norm(dim=1), mean_patch(Bc.X).norm(dim=1)
    # covariance on 300 images of seed 10.  For an interior patch (its 5x5 window inside the image,
    # output rows/cols 2..29) centring only subtracts a constant, so its covariance is unchanged up
    # to the float32 rounding of x - mu (~1e-7 relative).  Over all positions it does change: zero
    # padding pads with 0, which is far from the raw mean (0.44) but equal to the centred one, so the
    # border patches move by more than a constant -- reported, not judged.
    Ur = F.unfold(Br.X[0, :300], 5, padding=2)
    Uc = F.unfold(Bc.X[0, :300], 5, padding=2)
    ii = torch.arange(1024, device=Ur.device)
    inner = ((ii // 32 >= 2) & (ii // 32 <= 29) & (ii % 32 >= 2) & (ii % 32 <= 29))
    def cov(U, mask=None):
        V = U if mask is None else U[:, :, mask]
        return torch.cov(V.permute(0, 2, 1).reshape(-1, 75).double().T)
    Sr_in, Sc_in = cov(Ur, inner), cov(Uc, inner)
    Sr, Sc = cov(Ur), cov(Uc)
    cov_rel_inner = float((Sr_in - Sc_in).norm() / Sr_in.norm())
    cov_rel = float((Sr - Sc).norm() / Sr.norm())
    del Ur, Uc
    chan_mean = Bc.X.double().mean((0, 1, 3, 4)).tolist()
    # pass: the mean patch drops by more than an order of magnitude (|m| ~ 3.8 -> below 0.38; the
    # residual is the 1200-image subset's mean minus the training-set mean) and the interior
    # covariance is unchanged to float32 rounding
    ok = float(mc.max()) < 0.1 * float(mr.min()) and cov_rel_inner < 1e-5
    return {"mean_patch_norm_raw": [float(v) for v in mr], "mean_patch_norm_centred": [float(v) for v in mc],
            "covariance_rel_change_interior_seed10": cov_rel_inner,
            "covariance_rel_change_all_positions_seed10_reported": cov_rel,
            "centred_channel_means_all_seeds": chan_mean,
            "center": BundleLRc.center.tolist(), "pass": ok}


def check_same(cifar, device) -> dict:
    """With the arm "LR" the subclass is the reference bundle bit for bit (2 epochs, eager)."""
    slots = [("LR", 10), ("LR", 11), ("LR", 12)]
    A = LRM.BundleLR(slots, cifar, device, graph=False)
    B = BundleLRc(slots, cifar, device, graph=False)
    for Z in (A, B):
        Z.new_labels()
        for _ in range(2):
            Z.run_epoch()
    eq = all(bool((a == b).all()) for a, b in zip(A._state_tensors(), B._state_tensors()))
    return {"bit_identical": eq, "pass": eq}


def check_graph(cifar, device) -> dict:
    slots = [(ARM, 10), (ARM, 11), (ARM, 12), (ARM, 13)]
    Bg = BundleLRc(slots, cifar, device, graph=True)
    Be = BundleLRc(slots, cifar, device, graph=False)
    for Z in (Bg, Be):
        Z.new_labels()
        for _ in range(2):
            Z.run_epoch()
    eq = all(bool((a == b).all()) for a, b in zip(Bg._state_tensors(), Be._state_tensors()))
    moved = bool((Bg.P[0] != E.stack_init([s for a, s in slots], device)[0]).any())
    return {"bit_identical": eq, "params_moved": moved, "pass": eq and moved}


def check_host(cifar, device) -> dict:
    """One step of the centred bundle against the host's forward on the centred images."""
    B = BundleLRc([(ARM, 10)], cifar, device, graph=False)
    B.new_labels(); B.new_order()
    params = CN.init_params(10, device)
    idx = B.order[0, :CN.BATCH]
    xh = CN.images(cifar, CN.subset_idx(10), device) - BundleLRc.center.to(device).view(1, 3, 1, 1)
    lh = F.cross_entropy(CN.forward_cnn(params, xh[idx], H.ARMS["LR"])[8], B.Y[0, idx])
    gh = torch.autograd.grad(lh, params)
    lb = F.cross_entropy(E.forward(B.P, B.X[:, idx], B.act)[8][0], B.Y[0, idx])
    gb = torch.autograd.grad(lb, B.P)
    loss_rel = abs(float(lb) - float(lh)) / abs(float(lh))
    grad_rel = max(float((a[0] - b).abs().max() / b.abs().max().clamp_min(1e-30)) for a, b in zip(gb, gh))
    return {"loss_rel": loss_rel, "grad_rel_max": grad_rel, "pass": loss_rel < 1e-5 and grad_rel < 1e-4}


def check_repro(cifar, device, out: Path) -> dict:
    """The engine's run loop with the reference arm "LR" for one task reproduces task 1 of the
    reference run (results/cnn_drive_verify_1009/LR) -- same bundle composition, so bit for bit."""
    tmp = out / "repro_LR_t1"
    E.run([("LR", s) for s in range(10, 20)], tmp, 1, 400, device, lr=1e-3, ckpt_tasks=(), graph=True,
          resume=False)
    new = pd.read_csv(tmp / "per_task.csv").set_index("seed")
    ref = pd.read_csv(REF)
    ref = ref[ref.task == 1].set_index("seed")
    cols = ["online_acc", "memo_acc", "zbar_c1", "zsd_c1", "zbar_c2", "w_norm_c1"]
    diff = {c: float((new[c] - ref[c]).abs().max()) for c in cols}
    for f in tmp.glob("ckpt/*.pt"):
        f.unlink()
    return {"max_abs_diff_vs_reference_task1": diff, "pass": all(v == 0.0 for v in diff.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["checks", "run"])
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
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cifar = RC.Cifar10()
    install(cifar)
    if args.mode == "checks":
        rep = {}
        for name, fn in (("K-center", lambda: check_center(cifar, device)),
                         ("K-same", lambda: check_same(cifar, device)),
                         ("K-graph", lambda: check_graph(cifar, device)),
                         ("K-host", lambda: check_host(cifar, device)),
                         ("K-repro", lambda: check_repro(cifar, device, out))):
            t0 = time.time()
            rep[name] = fn() | {"seconds": time.time() - t0}
            print(name, json.dumps(rep[name], default=float), flush=True)
        rep["all_pass"] = all(v["pass"] for v in rep.values())
        (out / "checks.json").write_text(json.dumps(rep, indent=1, default=float))
        print("all_pass", rep["all_pass"])
        return
    slots = [(ARM, s) for s in E.parse_seeds(args.seeds)]
    E.run(slots, out, args.tasks, args.epochs, device, lr=args.lr, ckpt_tasks=CKPT_TASKS, graph=True)
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (out / "provenance_cnn_center.json").write_text(json.dumps(
        {"engine": "sna_cnn_cause_1009 run + cnn_drive_verify_1009_lr BundleLR, input centred (this file)",
         "arm": ARM, "slope": 0.1, "center_train_channel_mean": BundleLRc.center.tolist(),
         "seeds": [s for a, s in slots], "tasks": args.tasks, "epochs": args.epochs, "lr": args.lr,
         "reference": str(REF.relative_to(ROOT)), "git_hash": head, "torch": torch.__version__}, indent=1))


if __name__ == "__main__":
    main()

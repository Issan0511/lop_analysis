"""Post-hoc helpers for sna_cnn_cause_1009: rebuild one run from its checkpoint.

`load_run(ckpt)` returns a one-run bundle (R = 1) whose parameters, Adam moments, step count
and activation state are the checkpoint's, plus the run's images and the labels of the
checkpoint's task (redrawn from the host's label stream).  Read-only unless the caller trains.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import pmnist_0905 as H                 # noqa: E402
from src import pmnist_rlcifar_0907 as RC        # noqa: E402
from src import rlcifar_cnn_0908 as CN           # noqa: E402
from src import sna_cnn_cause_1009 as E          # noqa: E402

_CIFAR = None


def cifar():
    global _CIFAR
    if _CIFAR is None:
        _CIFAR = RC.Cifar10()
    return _CIFAR


def labels_at(seed: int, task: int) -> torch.Tensor:
    g = H.stream("rlcc_labels", seed)
    y = None
    for _ in range(task):
        y = CN.task_labels(g)
    return y


def load_run(path: Path, device="cuda", graph=False) -> tuple[E.Bundle, int]:
    st = torch.load(path, map_location="cpu", weights_only=False)
    arm, seed = st["arm"], st["seed"]
    task = int(Path(path).stem.split("_t")[-1])
    B = E.Bundle([(arm, seed)], cifar(), torch.device(device), graph=graph)
    with torch.no_grad():
        for dst, src in ((B.P, st["P"]), (B.m, st["m"]), (B.v, st["v"])):
            for q, s in zip(dst, src):
                q[0].copy_(s)
        B.tc.fill_(st["tc"])
        for k in ("V", "ada", "fixA", "cval"):
            for dst, src in zip(getattr(B.act, k), st["act"][k]):
                dst[0].copy_(src)
    B.Y.copy_(labels_at(seed, task)[None])
    # advance the batch stream to the end of `task` so continued training uses the same orders
    for _ in range(task * B.epochs):
        torch.randperm(CN.N_IMAGES, generator=B.g_batch[seed])
    for _ in range(task):
        CN.task_labels(B.g_lab[seed])
    return B, task


@torch.no_grad()
def conv_variance_split(B: E.Bundle, chunk: int = 50) -> dict:
    """Per channel of c1 and c2: pooled variance over (N, H, W), the within-position part
    (variance over images at each position, averaged over positions) and the between-position
    part (variance over positions of the position means).  pooled = within + between."""
    out = {}
    X = B.X
    zs = {0: [], 1: []}
    for i0 in range(0, CN.N_IMAGES, chunk):
        o = E.forward(B.P, X[:, i0:i0 + chunk], B.act)
        zs[0].append(o[0].double().cpu())
        zs[1].append(o[2].double().cpu())
    for l, tag in ((0, "c1"), (1, "c2")):
        z = torch.cat(zs[l])                         # (N, C, H, W) for R = 1
        pooled = z.permute(1, 0, 2, 3).reshape(z.shape[1], -1).var(1, unbiased=False)
        pos_mean = z.mean(0)                         # (C, H, W)
        within = z.var(0, unbiased=False).mean((1, 2))
        between = pos_mean.reshape(z.shape[1], -1).var(1, unbiased=False)
        out[tag] = {"pooled": pooled, "within": within, "between": between}
    return out

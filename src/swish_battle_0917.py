#!/usr/bin/env python3
"""swish_battle_0917 -- adaptive-alpha Swish in the two Random-Label boxes of the Snake
battle (spec: specs/spec_swish_battle_0917.md).

Boxes, imported unchanged and driven through their own `run_one`:
  mlp   src/pmnist_rlmnist_0906.py   784-100-100-10 on 1200 MNIST images
  cnn   src/rlcifar_cnn_0908.py      Kumar's CNN (2 conv + 3 fc) on 1200 CIFAR-10 images
Both: 50 tasks x 400 epochs (30,000 Adam steps at 1e-3, batch 16), a fresh uniform
labelling per task, and the same images, init and batch order per seed for every arm.

New arms, registered into the host's ARMS table at run time (no host file is edited):
  SW1, SW3     fixed Swish     phi(z) = z * sigmoid(a z), a = 1 (SiLU) or 3
  SWA1, SWA3   adaptive Swish  a_j = clip(c / W_j, lo, hi), W_j = sqrt(EMA_beta var z_j),
                               c = 1 or 3.  Statistic, granularity (unit for fc, channel
                               for conv) and beta = 0.01 are the host SNA's.
  SWA1u, SWA3u the same with the alpha floor lowered from 1e-3 to 1e-8 (spec addendum 2)
  SNAc3        the host's adaptive Snake with c = 3 instead of 0.6
The host arms (R, LR, SN06, SN3, SNA) run unchanged through the same entry point.

Swish is scale-covariant like Snake: phi_{a/s}(s z) = s phi_a(z), so with a = c/W the
gate phi' is a function of z/W only.  beta = 0 freezes W at 1 and makes SWA<c>
bit-identical to SW<c> (S-ema-off).

The adaptive classes subclass the host's AdaptiveSnake / ChannelSnake only to inherit
the variance statistic, `update`, and the isinstance dispatch inside the hosts
(layer-indexed phi in forward, `update` after every Adam step, layer-indexed dphi and
`stats` in evaluate).  phi, dphi and stats are replaced.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import socket
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H              # host; not touched
from src import pmnist_rlmnist_0906 as RL     # MLP box; not touched
from src import pmnist_rlcifar_0907 as RC     # CIFAR data layer; not touched
from src import rlcifar_cnn_0908 as CN        # CNN box; not touched

EXPERIMENT = "swish_battle_0917"
OUT = H.REPO / "results" / EXPERIMENT
LR = 1e-3
BETA = 0.01                                    # the host SNA's EMA rate
TASKS, EPOCHS = 50, 400
SNA_C = 0.6                                    # the host SNA's c
# Swish alpha clip, wide on purpose.  Snake's hi = 3 keeps the period above the input's
# scale; Swish has no period, so the clip only guards W -> 0.  Every clip hit is counted
# in `alpha_clip_frac_*`.
SW_LO, SW_HI = 1e-3, 1e3
# the host SNA's own clips (the MLP host does not expose them; these are its defaults)
SNA_CLIP = {"mlp": (0.05, 3.0), "cnn": (0.005, 3.0)}

HOST_ARMS = ("R", "LR", "SN06", "SN3", "SNA")
FIXED_SW = {"SW1": 1.0, "SW3": 3.0}
ADAPT_SW = {"SWA1": 1.0, "SWA3": 3.0, "SWA1u": 1.0, "SWA3u": 3.0}
# spec addendum 2: the 1e-3 floor binds in the MLP's second layer (W ~ 5e3), so the
# "u" arms lower it to 1e-8; everything else is the registered SWA arm.
SW_LO_U = 1e-8


def sw_clip(arm: str) -> tuple[float, float]:
    return (SW_LO_U if arm.endswith("u") else SW_LO), SW_HI
ADAPT_SN = {"SNAc3": 3.0}
ARMS = HOST_ARMS + tuple(ADAPT_SN) + tuple(FIXED_SW) + tuple(ADAPT_SW)
BOXES = ("mlp", "cnn")
TAGS = {"mlp": ("l1", "l2"), "cnn": CN.SITES}


# --------------------------------------------------------------------------
# Swish
# --------------------------------------------------------------------------

def swish(z: torch.Tensor, a) -> torch.Tensor:
    """z * sigmoid(a z).  `a` is a python float (fixed) or a broadcastable tensor
    (adaptive); multiplication is exact either way, so the two agree bit for bit
    when the tensor holds the same value."""
    return z * torch.sigmoid(a * z)


def swish_gate(z: torch.Tensor, a) -> torch.Tensor:
    """d/dz [z sigmoid(a z)] = s + a z s (1 - s), s = sigmoid(a z)."""
    s = torch.sigmoid(a * z)
    return s + a * z * s * (1.0 - s)


class Swish:
    """Fixed-alpha Swish with the host Activation's interface (name, kind, param,
    phi(z), dphi(z))."""
    kind = "swish"

    def __init__(self, name: str, a: float):
        self.name, self.param = name, float(a)

    def phi(self, z):
        return swish(z, self.param)

    def dphi(self, z):
        return swish_gate(z, self.param)


def _alpha_stats(act, tags) -> dict:
    out = {}
    for l, tag in enumerate(tags):
        a = act.alpha(l)
        W = act.V[l].sqrt()
        raw = act.c / W
        out[f"alpha_med_{tag}"] = float(a.median())
        out[f"alpha_min_{tag}"] = float(a.min())
        out[f"alpha_max_{tag}"] = float(a.max())
        out[f"alpha_clip_frac_{tag}"] = float(((raw < act.lo) | (raw > act.hi)).float().mean())
        # = c wherever the clip is idle; the Snake analogue is the host's two_alpha_W
        out[f"alpha_W_med_{tag}"] = float((a * W).median())
    return out


class AdaptiveSwish(H.AdaptiveSnake):
    """Adaptive Swish, one alpha per hidden unit (MLP box)."""
    name, kind = "SWA", "adaptive_swish"

    def __init__(self, c: float, beta: float, device, lo: float = SW_LO, hi: float = SW_HI):
        super().__init__(c, beta, device, lo=lo, hi=hi)

    def phi(self, z, layer=0):
        return swish(z, self.alpha(layer))

    def dphi(self, z, layer=0):
        return swish_gate(z, self.alpha(layer))

    @torch.no_grad()
    def stats(self) -> dict:
        return _alpha_stats(self, TAGS["mlp"])


class ChannelSwish(CN.ChannelSnake):
    """Adaptive Swish, one alpha per output channel (conv) or unit (fc) (CNN box).
    A per-position alpha would stop the layer being a convolution, as for SNA."""
    name, kind = "SWA", "adaptive_swish"

    def __init__(self, c: float, beta: float, device, lo: float = SW_LO, hi: float = SW_HI):
        super().__init__(c, beta, device, lo=lo, hi=hi)

    def phi(self, z, layer=0):
        return swish(z, self._bcast(layer, self.alpha(layer)))

    def dphi(self, z, layer=0):
        return swish_gate(z, self._bcast(layer, self.alpha(layer)))

    @torch.no_grad()
    def stats(self) -> dict:
        return _alpha_stats(self, TAGS["cnn"])


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------

def arm_act(box: str, arm: str, device, beta: float = BETA):
    """(object to place in H.ARMS[arm] or None, c to hand the host's run_one).

    The host's run_one builds its own SNA state when an arm's kind is
    'adaptive_snake' (so SNAc3 only needs its registry entry and c = 3).  Any other
    object is used as-is, so an adaptive Swish must be a fresh instance per run.
    """
    if arm in HOST_ARMS:
        return None, SNA_C
    if arm in FIXED_SW:
        return Swish(arm, FIXED_SW[arm]), SNA_C
    if arm in ADAPT_SN:
        return H.Activation(arm, "adaptive_snake", ADAPT_SN[arm]), ADAPT_SN[arm]
    if arm in ADAPT_SW:
        cls = ChannelSwish if box == "cnn" else AdaptiveSwish
        lo, hi = sw_clip(arm)
        return cls(ADAPT_SW[arm], beta, device, lo=lo, hi=hi), ADAPT_SW[arm]
    raise SystemExit(f"unknown arm {arm!r}; known: {','.join(ARMS)}")


def describe(box: str, arm: str) -> dict:
    if arm in FIXED_SW:
        return {"family": "swish", "adaptive": False, "alpha": FIXED_SW[arm]}
    if arm in ADAPT_SW:
        return {"family": "swish", "adaptive": True, "c": ADAPT_SW[arm], "beta": BETA,
                "alpha_clip": list(sw_clip(arm))}
    if arm in ADAPT_SN or arm == "SNA":
        return {"family": "snake", "adaptive": True, "c": ADAPT_SN.get(arm, SNA_C), "beta": BETA,
                "alpha_clip": list(SNA_CLIP[box])}
    a = H.ARMS[arm]
    return {"family": a.kind, "adaptive": False, "param": a.param}


def load_data(box: str, device):
    return RC.Cifar10() if box == "cnn" else H.Mnist(device)


def run_host(box: str, arm: str, seed: int, data, device, tasks: int = TASKS,
             epochs: int = EPOCHS, beta: float = BETA, hist_dir: Path | None = None,
             debug: dict | None = None, act_out: list | None = None):
    """One (box, arm, seed) through the host's own run_one.  Returns (rows, divergence)."""
    act, c = arm_act(box, arm, device, beta)
    if act is not None:
        H.ARMS[arm] = act
    if act_out is not None:
        act_out.append(act)
    if box == "cnn":
        lo, hi = SNA_CLIP["cnn"]
        return CN.run_one(arm, seed, LR, tasks, data, device, epochs=epochs, c=c,
                          alpha_lo=lo, alpha_hi=hi, beta=beta, hist_dir=hist_dir, debug=debug)
    return RL.run_one(arm, seed, LR, tasks, data, device, epochs=epochs, c=c, beta=beta,
                      debug=debug)


def git_state() -> dict:
    def g(*a):
        return subprocess.run(["git", "-C", str(H.REPO), *a], capture_output=True,
                              text=True).stdout.strip()
    return {"git_hash": g("rev-parse", "HEAD"),
            "git_dirty_code": bool(g("status", "--porcelain", "--", "src", "analysis"))}


def cmd_run(a) -> None:
    gs = git_state()                  # the code this run starts with, not HEAD at its end
    device = H.setup(a.device)
    torch.set_num_threads(a.threads)
    out = Path(a.out) if a.out else OUT / a.box / a.arm / f"seed{a.seed}"
    if (out / "provenance.json").exists() and not a.force:
        raise SystemExit(f"{out} already complete (provenance.json present); --force to redo")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = load_data(a.box, device)
    rows, div = run_host(a.box, a.arm, a.seed, data, device, tasks=a.tasks, epochs=a.epochs,
                         hist_dir=(out / "hist") if a.box == "cnn" else None)
    H.write_csv(out / "per_task.csv", rows)
    done = len([r for r in rows if r.get("memo_acc") == r.get("memo_acc") and "memo_acc" in r])
    prov = {"run_id": EXPERIMENT, "box": a.box, "arm": a.arm, "seed": a.seed,
            **gs,
            "host": {"mlp": "src/pmnist_rlmnist_0906.py", "cnn": "src/rlcifar_cnn_0908.py"}[a.box],
            "act": describe(a.box, a.arm), "lr": LR, "optimizer": "adam", "tasks": a.tasks,
            "epochs_per_task": a.epochs, "tasks_completed": done, "divergence": div,
            "data_sha256": data.sha256, "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "threads": torch.get_num_threads(), "torch": torch.__version__,
            "hostname": socket.gethostname(),
            "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "wall_clock_s": time.time() - t0}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    print(f"[{time.strftime('%F %T')}] {a.box} {a.arm} seed={a.seed} {a.device}: "
          f"{done}/{a.tasks} tasks in {time.time() - t0:.0f}s"
          f"{'  DIVERGED@task ' + str(div['task']) if div['diverged'] else ''}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one (box, arm, seed)")
    r.add_argument("--box", required=True, choices=BOXES)
    r.add_argument("--arm", required=True, choices=ARMS)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--device", default="cuda")
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--tasks", type=int, default=TASKS)
    r.add_argument("--epochs", type=int, default=EPOCHS)
    r.add_argument("--out", default=None)
    r.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a)


if __name__ == "__main__":
    main()

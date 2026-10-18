#!/usr/bin/env python3
"""cifar5p1_mlp_snfix_1019 -- fixed-alpha Snake arms on the 5+1 CIFAR MLP box (spec_cifar5p1_mlp_snfix_1019.md).

    python3 src/cifar5p1_mlp_snfix_1019.py run --arm SN06            # one arm, seeds 0-9, std, lr 1e-4
    bash analysis/cifar5p1_mlp_snfix_1019/production.sh               # all 19 arms, one process each

The box, the engine, the task sequence, the batch lists and the initial weights are those of
`cifar5p1_mlp_0920` and are called, not copied: `C.run` builds its activation through the module
global `make_act`, and this file wraps that function so that three new names resolve to the
activations below while every other name falls through to the 0920 factory unchanged.  Nothing in
cifar5p1_mlp_0920 / rlcifar_mlp_battle_0918 / pmnist_0905 is modified.

    SN06     phi(z) = z + sin^2(alpha z) / alpha with alpha = 0.6 fixed, every unit of both hidden layers
    SN3      the same with alpha = 3
    SNAfrz1  SNA (alpha = clip(0.6 / W, 0.005, 3), W = sqrt(EMA_0.01 var_batch z)) until the last update of
             task 1; from the first update of task 2 the EMA that sets alpha stops, so alpha is fixed per unit

The fixed arms and SNAfrz1 also keep a diagnostic EMA of the batch variance that never feeds phi, so
that the reported 2 alpha W follows the actual width.
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

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H                    # noqa: E402
from src import rlcifar_mlp_battle_0918 as B        # noqa: E402
from src import cifar5p1_mlp_0920 as C              # noqa: E402

EXPERIMENT = "cifar5p1_mlp_snfix_1019"
OUT_ROOT = H.REPO / "results" / EXPERIMENT
SPEC = "specs/spec_cifar5p1_mlp_snfix_1019.md"
HOST_RESULTS = H.REPO / "results" / "cifar5p1_mlp_0920"
FIXED = {"SN06": 0.6, "SN3": 3.0}
NEW_ARMS = ("SN06", "SN3", "SNAfrz1")
OLD_ARMS = B.ARM_ORDER + ("LK07", "CR", "DF")         # the 16 arms of the 0920 run
ALL_ARMS = OLD_ARMS + NEW_ARMS
SEEDS = list(range(10))
COND = "std"
LR = C.LR
FREEZE_AFTER = C.STEPS_PER_TASK                      # task 1 is hard and has 780 updates
THREADS = 2                                          # the 0920 CLI default; eff_rank depends on it


def cell_name(arm: str, lr: float = LR) -> str:
    """The 0920 directory tag for a default cell (std, hidden 100, no intervention)."""
    return f"{arm}_{COND}_lr{lr:g}"


# --------------------------------------------------------------------------
# activations
# --------------------------------------------------------------------------

class FixedSnake(B.Act):
    """Snake with a constant alpha.  phi uses the SNA op order: a = alpha[:, None, :], z + sin(a z)^2 * a^-1."""
    adaptive = False

    def __init__(self, name: str, alpha: float, beta: float = 0.01, widths=(C.HIDDEN, C.HIDDEN)):
        self.name, self.a0, self.beta, self.widths = name, float(alpha), beta, widths
        self.A = None
        self.Vd = None                               # diagnostic EMA of var_batch(z); never enters phi

    def init_state(self, R, device, key):
        self.A = [torch.full((R, w), self.a0, device=device) for w in self.widths]
        self.Vd = [torch.ones(R, w, device=device) for w in self.widths]

    def state(self):
        return {"A": [a.clone() for a in self.A], "Vd": [v.clone() for v in self.Vd]}

    def load_state(self, st):
        for a, src in zip(self.A, st["A"]):
            a.copy_(src)
        for v, src in zip(self.Vd, st["Vd"]):
            v.copy_(src)

    def alpha(self, layer):
        return self.A[layer]

    def phi(self, z, layer=0, train=False):
        a = self.alpha(layer)[:, None, :]
        ia = a.reciprocal()
        return z + torch.sin(a * z) ** 2 * ia

    def dphi(self, z, layer=0):
        a = self.alpha(layer)[:, None, :]
        return 1.0 + torch.sin(2.0 * a * z)

    @torch.no_grad()
    def update(self, z1, z2):
        for l, z in ((0, z1), (1, z2)):
            self.Vd[l].mul_(1 - self.beta).add_(self.beta * z.var(1, unbiased=False))

    @torch.no_grad()
    def stats(self, r):
        out = {}
        for l in (0, 1):
            a = self.alpha(l)[r]
            W = self.Vd[l][r].sqrt()
            out[f"alpha_med_l{l+1}"] = float(a.median())
            out[f"alpha_min_l{l+1}"] = float(a.min())
            out[f"alpha_max_l{l+1}"] = float(a.max())
            out[f"alpha_clip_frac_l{l+1}"] = 0.0
            out[f"two_alpha_W_med_l{l+1}"] = float((2 * a * W).median())
        return out


class FrozenSnake(B.SnakeFamily):
    """SNA whose alpha-setting EMA stops at the first update after `freeze_after` updates (SNAfrz1).

    `begin_step` runs once before every training step, outside any captured CUDA graph, so the
    freeze is a flag tensor the captured update reads.  With freeze_after=None it never freezes
    and is SNA bit for bit (S-frz checks that).  Either way it records alpha (on the cpu) before
    the first update of every task after the first, the first time the sequence gets there:
    captured[780] is alpha after the last update of task 1.  The fresh control restarts the
    counter and never reaches 780, so it neither freezes nor overwrites a capture.
    The update writes V*(1-beta) + beta*var -- the SNA expression, evaluated in the same order --
    through torch.where, so the stored V is the same float as SNA's in-place mul_/add_.
    """

    def __init__(self, name="SNAfrz1", c=0.6, beta=0.01, lo=0.005, hi=3.0, widths=(C.HIDDEN, C.HIDDEN),
                 freeze_after: int | None = FREEZE_AFTER, capture_every: int = C.STEPS_PER_TASK):
        super().__init__(name, "snake", c, beta, lo, hi, widths=widths)
        self.freeze_after, self.capture_every = freeze_after, capture_every
        self.keep = None
        self.Vd = None
        self.n = 0
        self.captured: dict[int, list[torch.Tensor]] = {}

    def init_state(self, R, device, key):
        super().init_state(R, device, key)
        self.keep = torch.ones((), device=device)
        self.Vd = [torch.ones(R, w, device=device) for w in self.widths]
        self.n = 0

    def state(self):
        return {"V": [v.clone() for v in self.V], "Vd": [v.clone() for v in self.Vd],
                "keep": self.keep.clone(), "n": self.n}

    def load_state(self, st):
        super().load_state(st)
        for v, src in zip(self.Vd, st["Vd"]):
            v.copy_(src)
        self.keep.copy_(st["keep"])
        self.n = st["n"]

    def begin_step(self, R, B_, device):
        if self.n > 0 and self.n % self.capture_every == 0 and self.n not in self.captured:
            self.captured[self.n] = [self.alpha(l).detach().cpu().clone() for l in (0, 1)]
        if self.freeze_after is not None and self.n == self.freeze_after:
            self.keep.zero_()
        self.n += 1

    @torch.no_grad()
    def update(self, z1, z2):
        for l, z in ((0, z1), (1, z2)):
            var = z.var(1, unbiased=False)
            new = self.V[l] * (1 - self.beta) + self.beta * var
            self.V[l].copy_(torch.where(self.keep > 0, new, self.V[l]))
            self.Vd[l].mul_(1 - self.beta).add_(self.beta * var)

    @torch.no_grad()
    def stats(self, r):
        out = super().stats(r)
        for l in (0, 1):
            a = self.alpha(l)[r]
            out[f"two_alpha_W_med_l{l+1}"] = float((2 * a * self.Vd[l][r].sqrt()).median())
        return out


_HOST_MAKE_ACT = C.make_act
LAST_ACT: dict[str, B.Act] = {}


def make_act(arm: str, hidden: int = C.HIDDEN, c=0.6, beta=0.01, lo=0.005, hi=3.0, lam: float = 0.0) -> B.Act:
    """The 0920 factory plus SN06 / SN3 / SNAfrz1 (and the check-only SNAcap, SNA with a capture hook)."""
    if arm in FIXED:
        act = FixedSnake(arm, FIXED[arm], beta, widths=(hidden, hidden))
    elif arm == "SNAfrz1":
        act = FrozenSnake("SNAfrz1", c, beta, lo, hi, widths=(hidden, hidden))
    elif arm == "SNAcap":
        act = FrozenSnake("SNAcap", c, beta, lo, hi, widths=(hidden, hidden), freeze_after=None)
    else:
        act = _HOST_MAKE_ACT(arm, hidden, c, beta, lo, hi, lam)
    LAST_ACT[arm] = act
    return act


C.make_act = make_act                                 # C.run looks the name up at call time


# --------------------------------------------------------------------------
# one cell
# --------------------------------------------------------------------------

def git_dirty() -> list[str]:
    out = subprocess.run(["git", "-C", str(H.REPO), "status", "--porcelain", "--", "src", "analysis", "specs"],
                         capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def run_cell(arm: str, out: Path, device, seeds=SEEDS, n_tasks: int = C.N_TASKS, cifar=None, fresh=True,
             graph=True, lr: float = LR, progress=None, allow_dirty: bool = False) -> dict:
    """C.run on one arm, then the provenance of this experiment and SNAfrz1's frozen alpha."""
    dirty = git_dirty()
    if dirty and not allow_dirty:
        raise SystemExit(f"src/analysis/specs have uncommitted changes: {dirty}")
    prov = C.run(arm, list(seeds), COND, n_tasks, device, out, lr=lr, cifar=cifar, fresh=fresh, graph=graph,
                 progress=progress)
    act = LAST_ACT[arm]
    extra = {"run_id": EXPERIMENT, "host_run_id": C.EXPERIMENT, "spec": SPEC, "host_spec": prov.get("spec"),
             "dirty_src_analysis_specs": dirty, "torch_threads": torch.get_num_threads(),
             "arm_definition": ({"phi": "z + sin(alpha z)^2 / alpha", "alpha": FIXED[arm], "fixed": True,
                                 "diagnostic_ema_beta": act.beta} if arm in FIXED else
                                {"phi": "SNA until the last update of task 1, then the alpha-setting EMA stops",
                                 "freeze_after_updates": act.freeze_after, "c": act.c, "beta": act.beta,
                                 "alpha_lo": act.lo, "alpha_hi": act.hi} if arm == "SNAfrz1" else "0920 factory")}
    if arm == "SNAfrz1":
        if FREEZE_AFTER not in act.captured:
            raise SystemExit("SNAfrz1 never reached the end of task 1")
        later = [n for n in sorted(act.captured) if n > FREEZE_AFTER]
        same = all(torch.equal(act.captured[n][l], act.captured[FREEZE_AFTER][l]) for n in later for l in (0, 1))
        rule = "alpha after the last update of task 1, per seed (slot order), hidden layer and unit"
        fa = {"rule": rule, "seeds": list(seeds), "unchanged_at_later_task_starts": same,
              "later_task_starts_checked": len(later),
              "l1": act.captured[FREEZE_AFTER][0].tolist(), "l2": act.captured[FREEZE_AFTER][1].tolist()}
        (out / "frozen_alpha.json").write_text(json.dumps(fa))
        extra["frozen_alpha_file"] = "frozen_alpha.json"
    prov = {**prov, **extra}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    return prov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "smoke"])
    ap.add_argument("--arm", choices=list(ALL_ARMS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tasks", type=int, default=C.N_TASKS)
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(THREADS)
    device = H.setup(a.device)
    if a.stage == "run":
        if a.arm is None:
            raise SystemExit("--arm")
        out = Path(a.out) if a.out else OUT_ROOT / cell_name(a.arm)
        if (out / "provenance.json").exists():
            print(f"{out} is complete; skipped", flush=True)
            return
        run_cell(a.arm, out, device, n_tasks=a.tasks, allow_dirty=a.allow_dirty)
    else:                                            # smoke: the new arms, 2 tasks, timing only
        root = Path(a.out) if a.out else OUT_ROOT / "smoke"
        cifar = C.Cifar100()
        for arm in NEW_ARMS:
            t0 = time.time()
            prov = run_cell(arm, root / cell_name(arm), device, n_tasks=2, cifar=cifar, fresh=True,
                            allow_dirty=a.allow_dirty)
            print(f"{arm}: {time.time() - t0:.1f} s for 2 tasks + fresh; last task {prov['step_ms_last_task']:.2f} ms/step",
                  flush=True)


if __name__ == "__main__":
    main()

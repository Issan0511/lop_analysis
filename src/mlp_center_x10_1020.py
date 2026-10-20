#!/usr/bin/env python3
"""mlp_center_x10_1020 -- R / CH / CHB0 on the 5+1 CIFAR MLP at 10x the updates per task, with per-unit
diagnostics (specs/spec_mlp_center_x10_1020.md).

    python3 src/mlp_center_x10_1020.py run --arm CHB0       # seeds 0-9, std, lr 1e-4, 7800 x 7800, fresh control
    python3 src/mlp_center_x10_1020.py smoke                # seed 100, 3 tasks, CHB0
    bash analysis/mlp_center_x10_1020/production.sh         # R, CH, CHB0, then the verdict

The box, the engine, the task sequence, the batch lists and the initial weights are those of
`cifar5p1_mlp_0920.run`, called and not copied; nothing in cifar5p1_mlp_0920 / rlcifar_mlp_battle_0918 /
pmnist_0905 is modified.  The diagnostics ride on two module-level names that `C.run` looks up at call
time, the way cifar5p1_mlp_snfix_1019 wraps `make_act`:

  * `C.make_act` -- the activation it returns gets a wrapped `begin_step`.  `begin_step` runs once before
    every training step, outside any captured CUDA graph.  When it is called from 0920's `train_task` it
    reads that frame's `j` (updates already done in this task) and `acc_sum` (the engine's running sum of
    the pre-update batch accuracy), and every SNAP updates copies `acc_sum` into a buffer on the device.
    Nothing inside the captured graph changes, and nothing the step reads is written.
  * `C.evaluate` -- called once after every task.  The host function runs first and its rows go to
    per_task.csv untouched; then this file reads the `run` frame (task index, steps, acc_sum, X_test,
    te_rows, seeds), harvests the learning-curve buffer, and records per-unit statistics on the seed's
    fixed probe (1,200 CIFAR-100 test images, 12 per class) and on the task's own training images,
    the biases and the row norms.  Reads only.

S-diag (src/mlp_center_x10_1020_checks.py) checks that per_task.csv and fresh_control.csv are the same
bytes with and without the hooks.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H                    # noqa: E402
from src import rlcifar_mlp_battle_0918 as B        # noqa: E402
from src import cifar5p1_mlp_0920 as C              # noqa: E402

EXPERIMENT = "mlp_center_x10_1020"
OUT_ROOT = H.REPO / "results" / EXPERIMENT
SPEC = "specs/spec_mlp_center_x10_1020.md"
HOST_RESULTS = H.REPO / "results" / "cifar5p1_mlp_0920"
ARMS = ("R", "CH", "CHB0")
SEEDS = list(range(10))
COND = "std"
LR = C.LR
STEPS = 10 * C.STEPS_PER_TASK                       # 7800, hard and easy (0920 addendum 5 (a))
THREADS = 2                                         # the 0920 CLI default; eff_rank depends on it
SNAP = 20                                           # learning-curve grid: cumulative hits every 20 updates
PROBE_PER_CLASS = 12                                # 12 x 100 classes = 1,200 test images per seed
PROBE_ROLE = "mcx_probe"
BINS = ("dead", "lt005", "sparse", "mid", "hi", "ge095")
BIN_TEXT = {"dead": "p+ = 0", "lt005": "0 < p+ < 0.05", "sparse": "0.05 <= p+ < 0.3",
            "mid": "0.3 <= p+ <= 0.7", "hi": "0.7 < p+ < 0.95", "ge095": "p+ >= 0.95"}
MAIN_CLONE = Path.home() / "Projects" / "claude" / "proj_004_drift"


def cell_name(arm: str, sh: int = STEPS, se: int = STEPS) -> str:
    """The tag of 0920's 10x cells (run_x10.sh), so the old and new directories line up."""
    return f"{arm}_{COND}_x10_{sh}x{se}"


def cifar_archive() -> Path:
    """The CIFAR-100 archive: $MCX_CIFAR, this checkout's data/, or the main clone's data/ (no symlink)."""
    cands = ([Path(os.environ["MCX_CIFAR"])] if os.environ.get("MCX_CIFAR") else []) + [
        C.DATA_DIR / C.ARCHIVE, MAIN_CLONE / "data" / "cifar100" / C.ARCHIVE]
    for p in cands:
        if p.exists():
            return p
    raise SystemExit(f"CIFAR-100 archive not found in {[str(p) for p in cands]}")


def load_cifar() -> C.Cifar100:
    return C.Cifar100(cifar_archive())


# --------------------------------------------------------------------------
# p+ bins (integer comparisons, so no boundary depends on float rounding)
# --------------------------------------------------------------------------

def pplus_bins(cnt: np.ndarray, n: int) -> dict[str, np.ndarray]:
    """Boolean masks of the six exhaustive, exclusive p+ = cnt / n bins (spec §5.2)."""
    c = np.asarray(cnt).astype(np.int64)
    out = {"dead": c == 0,
           "lt005": (c > 0) & (20 * c < n),
           "sparse": (20 * c >= n) & (10 * c < 3 * n),
           "mid": (10 * c >= 3 * n) & (10 * c <= 7 * n),
           "hi": (10 * c > 7 * n) & (20 * c < 19 * n),
           "ge095": 20 * c >= 19 * n}
    return out


# --------------------------------------------------------------------------
# a rounding bound for eff_rank (spec §7)
# --------------------------------------------------------------------------

U64 = 2.0 ** -53


def gamma(k: int) -> float:
    return k * U64 / (1 - k * U64)


def _f(p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p > 0, -p * np.log(np.where(p > 0, p, 1.0)), 0.0)


def _er_change_bound(ev: np.ndarray, er: float, delta: float, n: int) -> float:
    """Bound on |d eff_rank| when every sorted eigenvalue of the Gram may move by `delta` (Weyl).

    Interval arithmetic on s = sqrt(lambda) (clamped at 0 as H.eff_rank does) and p = s / sum(s), then
    the variation of each term -p log p over its interval, summed: |dH| <= dH, |d eff_rank| <= er (e^dH - 1).
    """
    s_lo = np.sqrt(np.maximum(ev - delta, 0.0))
    s_hi = np.sqrt(ev + delta)
    S_lo, S_hi = s_lo.sum(), s_hi.sum()
    if S_lo <= 0:
        return float("inf")
    p_lo = s_lo / S_hi
    p_hi = np.minimum(s_hi / S_lo, 1.0)
    f_lo, f_hi, top = _f(p_lo), _f(p_hi), 1.0 / math.e
    inside = (p_lo <= top) & (p_hi >= top)
    vmax = np.where(inside, -top * math.log(top), np.maximum(f_lo, f_hi))
    vmin = np.minimum(f_lo, f_hi)
    dH = float((vmax - vmin).sum()) + 4 * n * U64            # + the rounding of the entropy sum itself
    return er * math.expm1(dH)


def eff_rank_bound(a: torch.Tensor) -> tuple[float, float]:
    """(eff_rank, B64) for H.eff_rank of one slot's float32 activation matrix a (N x n).

    B64 bounds the difference between two computations on the SAME float32 a that differ only in
    floating-point rounding (summation order of the float64 Gram, the eigensolver).  |dG| <= gamma_N
    |A|^T|A| elementwise for any order, so ||dG||_2 <= gamma_N ||A||_F^2; the eigensolver is backward
    stable with error <= p(n) u ||G||_2, p(n) taken as n, bounded by gamma_n ||A||_F^2.  Per run each
    sorted eigenvalue moves by at most eps = (gamma_N + gamma_n) ||A||_F^2 (Weyl); two runs: 2 eps.
    If the float32 activations themselves differ, the bound does not apply.
    """
    er = H.eff_rank(a)
    N, n = a.shape
    ad = a.double()
    ev = torch.linalg.eigvalsh((ad.T @ ad).cpu()).clamp_min(0.0).numpy()
    fa = float((ad * ad).sum().cpu()) * (1 + gamma(N * n))
    return er, _er_change_bound(ev, er, 2 * (gamma(N) + gamma(n)) * fa, n)


def print_halfunit(v: float) -> float:
    """Half a unit in the last place of `%.10g` (H.write_csv's format) at v."""
    if v == 0 or not math.isfinite(v):
        return 0.0
    return 0.5 * 10.0 ** (math.floor(math.log10(abs(v))) - 9)


# --------------------------------------------------------------------------
# the diagnostics
# --------------------------------------------------------------------------

class Diag:
    """Collects the learning curves and per-unit arrays of one C.run call (spec §5.2)."""

    def __init__(self, arm: str, seeds: list[int], n_tasks: int, steps_hard: int, steps_easy: int,
                 mutate: str | None = None, units: bool = True):
        self.arm, self.seeds, self.n_tasks = arm, list(seeds), n_tasks
        self.R = len(seeds)
        self.steps_hard, self.steps_easy = steps_hard, steps_easy
        self.n_grid = max(steps_hard, steps_easy) // SNAP
        self.mutate, self.units = mutate, units
        self.acc = None
        self.buf = None
        self.done = 0
        self.act = None
        self.calls = {"begin_step_train_task": 0, "begin_step_other": 0, "evaluate": 0}
        self.probe_idx = None
        self.Xp = None
        self.A: dict[str, np.ndarray] = {}
        self.rows: list[dict] = []
        self.fresh = None

    # ---- storage
    def _arr(self, name: str, shape: tuple, dtype=np.float32, fill=np.nan) -> np.ndarray:
        if name not in self.A:
            self.A[name] = np.full(shape, fill, dtype=dtype)
        return self.A[name]

    # ---- hook 1: before every training step
    def on_begin_step(self, fr) -> None:
        if fr.f_code.co_name != "train_task":
            self.calls["begin_step_other"] += 1          # CUDA-graph warmup and capture
            return
        loc = fr.f_locals
        j = loc["j"]
        if self.acc is None:
            self.acc = loc["acc_sum"]
            self.buf = torch.full((self.n_grid, self.R), float("nan"), device=self.acc.device)
        elif loc["acc_sum"] is not self.acc:
            raise RuntimeError("acc_sum is not the tensor seen at the first step")
        self.calls["begin_step_train_task"] += 1
        if j and j % SNAP == 0:
            self.buf[j // SNAP - 1].copy_(self.acc)      # cumulative hits after j updates

    def _harvest(self, steps: int) -> tuple[np.ndarray, np.ndarray]:
        """(R, n_grid) cumulative hits at SNAP, 2 SNAP, ... (nan off the task), and the (R,) task total."""
        n_mid = (steps - 1) // SNAP
        cum = np.full((self.R, self.n_grid), np.nan, np.float64)
        if n_mid:
            cum[:, :n_mid] = self.buf[:n_mid].T.double().cpu().numpy()
        final = self.acc.detach().double().cpu().numpy().copy()
        if steps % SNAP == 0:
            cum[:, steps // SNAP - 1] = final
        self.buf.fill_(float("nan"))                     # a missed snapshot then reads as nan
        return cum, final

    # ---- hook 2: after every task
    @torch.no_grad()
    def on_evaluate(self, fr, P, X, act, live: list[int]) -> None:
        if fr.f_code.co_name != "run":
            raise RuntimeError(f"evaluate called from {fr.f_code.co_name}, not run")
        loc = fr.f_locals
        t, steps = loc["t"], loc["steps"]
        if t != self.done + 1:
            raise RuntimeError(f"task {t} after {self.done}")
        if loc["acc_sum"] is not self.acc:
            raise RuntimeError("run's acc_sum is not the one the steps used")
        self.calls["evaluate"] += 1
        ti, R, nt = t - 1, self.R, self.n_tasks
        cum, final = self._harvest(steps)
        self._arr("curve_cum", (nt, R, self.n_grid), np.float64)[ti] = cum
        self._arr("curve_final", (nt, R), np.float64)[ti] = final
        self._arr("steps", (nt,), np.int64, 0)[ti] = steps
        self._arr("live", (nt, R), np.bool_, False)[ti, live] = True
        if self.units:
            if self.Xp is None:
                self._build_probe(loc["X_test"], loc["te_rows"], loc["seeds"])
            self._unit_stats(ti, "probe", P, self.Xp, act, live, bound=False)
            self._unit_stats(ti, "task", P, X, act, live, bound=True)
            self._params(ti, P, act, live)
        hard = int(t % 2 == 1)
        for r in live:
            row = {"arm": self.arm, "seed": self.seeds[r], "task": t, "hard": hard, "steps": steps,
                   "online_first100": self._window(cum[r], final[r], steps, 100),
                   "online_first780": self._window(cum[r], final[r], steps, 780),
                   "online_all": float(final[r]) / steps}
            if self.units:
                row.update(self._row_summary(ti, r))
            self.rows.append(row)
        self.done = t
        if self.mutate == "touch_b3":                    # S-diag mutant only: the hook writes a parameter
            P[5].add_(1e-7)

    @staticmethod
    def _window(cum_r: np.ndarray, final_r: float, steps: int, k: int) -> float:
        if k > steps:
            return float("nan")
        if k == steps:
            return float(final_r) / k
        if k % SNAP:
            return float("nan")
        return float(cum_r[k // SNAP - 1]) / k

    def _build_probe(self, X_test, te_rows, seeds) -> None:
        idx = []
        for s in seeds:
            g = H.stream(PROBE_ROLE, s)
            idx.append(torch.cat([te_rows[c][torch.randperm(len(te_rows[c]), generator=g)[:PROBE_PER_CLASS]]
                                  for c in range(C.N_CLASSES)]))
        self.probe_idx = torch.stack(idx)                                    # (R, 1200), test-split rows
        self.Xp = torch.stack([X_test[i.to(X_test.device)] for i in self.probe_idx])
        self.A["probe_fingerprint"] = self.Xp.double().sum((1, 2)).cpu().numpy()   # S-probe: which inputs

    def _unit_stats(self, ti, src, P, X, act, live, bound: bool) -> None:
        nt, R = self.n_tasks, self.R
        N = X.shape[1]
        z1, a1, z2, a2, _ = B.forward(P, X, act, train=False)
        self._arr(f"n_{src}", (nt,), np.int64, 0)[ti] = N
        for li, (z, a) in ((1, (z1, a1)), (2, (z2, a2))):
            w = z.shape[-1]
            zbar = self._arr(f"{src}_zbar_l{li}", (nt, R, w))
            zsd = self._arr(f"{src}_zsd_l{li}", (nt, R, w))
            cnt = self._arr(f"{src}_open_l{li}", (nt, R, w), np.int32, -1)
            er = self._arr(f"{src}_eff_rank_l{li}", (nt, R), np.float64)
            b64 = self._arr(f"{src}_eff_rank_b64_l{li}", (nt, R), np.float64) if bound else None
            for r in live:
                zr = z[r]                                                    # the 0920 expressions, per slot
                zbar[ti, r] = zr.mean(0).cpu().numpy()
                zsd[ti, r] = zr.std(0).cpu().numpy()
                cnt[ti, r] = (zr > 0).sum(0).cpu().numpy()
                if bound:
                    er[ti, r], b64[ti, r] = eff_rank_bound(a[r])
                else:
                    er[ti, r] = H.eff_rank(a[r])

    def _params(self, ti, P, act, live) -> None:
        nt, R = self.n_tasks, self.R
        for name, k in (("b1", 1), ("b2", 3), ("b3", 5)):
            arr = self._arr(name, (nt, R, P[k].shape[1]))
            for r in live:
                arr[ti, r] = P[k][r].cpu().numpy()
        for name, k in (("wnorm1", 0), ("wnorm2", 2), ("wnorm3", 4)):
            arr = self._arr(name, (nt, R, P[k].shape[1]))
            for r in live:
                arr[ti, r] = P[k][r].norm(dim=1).cpu().numpy()
        if getattr(act, "door_h", False):
            for li in (1, 2):
                arr = self._arr(f"m_l{li}", (nt, R, act.m[li - 1].shape[1]))
                for r in live:
                    arr[ti, r] = act.m[li - 1][r].cpu().numpy()

    def _row_summary(self, ti, r) -> dict:
        out = {}
        for src in ("probe", "task"):
            n = int(self.A[f"n_{src}"][ti])
            for li in (1, 2):
                c = self.A[f"{src}_open_l{li}"][ti, r]
                out[f"{src}_sigma_med_l{li}"] = float(np.median(self.A[f"{src}_zsd_l{li}"][ti, r]))
                out[f"{src}_zbar_med_l{li}"] = float(np.median(self.A[f"{src}_zbar_l{li}"][ti, r]))
                for b, m in pplus_bins(c, n).items():
                    out[f"{src}_frac_{b}_l{li}"] = float(m.mean())
                out[f"{src}_eff_rank_l{li}"] = float(self.A[f"{src}_eff_rank_l{li}"][ti, r])
        for li in (1, 2):
            out[f"b{li}_med"] = float(np.median(self.A[f"b{li}"][ti, r]))
            out[f"wnorm{li}_med"] = float(np.median(self.A[f"wnorm{li}"][ti, r]))
        return out

    # ---- after C.run: the fresh control's curve (it ran after the last evaluate)
    def harvest_fresh(self, fresh_ran: bool) -> None:
        if not fresh_ran or self.acc is None:
            return
        cum, final = self._harvest(self.steps_hard)
        self.A["fresh_curve_cum"] = cum
        self.A["fresh_curve_final"] = final
        self.fresh = {"steps": self.steps_hard}

    def save(self, out: Path) -> None:
        arrs = dict(self.A)
        if self.probe_idx is not None:
            arrs["probe_idx"] = self.probe_idx.numpy().astype(np.int32)
        arrs["seeds"] = np.asarray(self.seeds, np.int64)
        arrs["snap"] = np.asarray(SNAP)
        np.savez(out / "diag_units.npz", **arrs)
        H.write_csv(out / "diag_per_task.csv", sorted(self.rows, key=lambda q: (q["seed"], q["task"])))


@contextlib.contextmanager
def hooks(diag: Diag):
    """Wrap C.make_act and C.evaluate for one C.run call; restore them afterwards."""
    host_make, host_eval = C.make_act, C.evaluate

    def make_act(*a, **k):
        act = host_make(*a, **k)
        orig = act.begin_step

        def begin_step(R, B_, device):
            orig(R, B_, device)
            diag.on_begin_step(sys._getframe(1))

        act.begin_step = begin_step
        diag.act = act
        return act

    def evaluate(P, X, Y, act, live):
        rows = host_eval(P, X, Y, act, live)
        diag.on_evaluate(sys._getframe(1), P, X, act, live)
        return rows

    C.make_act, C.evaluate = make_act, evaluate
    try:
        yield diag
    finally:
        C.make_act, C.evaluate = host_make, host_eval


# --------------------------------------------------------------------------
# one cell
# --------------------------------------------------------------------------

def git_dirty() -> list[str]:
    out = subprocess.run(["git", "-C", str(H.REPO), "status", "--porcelain", "--", "src", "analysis", "specs"],
                         capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def run_cell(arm: str, out: Path, device, seeds=SEEDS, n_tasks: int = C.N_TASKS, steps_hard: int = STEPS,
             steps_easy: int = STEPS, cifar=None, fresh: bool = True, graph: bool = True, diag: bool = True,
             mutate: str | None = None, debug: dict | None = None, allow_dirty: bool = False,
             progress=None) -> dict:
    """C.run on one arm (diagnostics on or off), then this experiment's provenance and diagnostic files."""
    dirty = git_dirty()
    if dirty and not allow_dirty:
        raise SystemExit(f"src/analysis/specs have uncommitted changes: {dirty}")
    cifar = cifar or load_cifar()
    t0 = time.time()
    d = Diag(arm, list(seeds), n_tasks, steps_hard, steps_easy, mutate=mutate) if diag else None
    ctx = hooks(d) if diag else contextlib.nullcontext()
    with ctx:
        prov = C.run(arm, list(seeds), COND, n_tasks, device, out, lr=LR, cifar=cifar, fresh=fresh,
                     graph=graph, steps_hard=steps_hard, steps_easy=steps_easy, debug=debug,
                     progress=progress)
    extra = {"run_id": EXPERIMENT, "host_run_id": C.EXPERIMENT, "spec": SPEC, "host_spec": prov.get("spec"),
             "dirty_src_analysis_specs": dirty, "torch_threads": torch.get_num_threads(),
             "cifar_archive": str(cifar_archive()), "diagnostics": bool(diag), "mutate": mutate,
             "wall_clock_with_diag_s": time.time() - t0}
    if diag:
        d.harvest_fresh(fresh and prov.get("fresh_control") is not None)
        d.save(out)
        extra.update({
            "hook_calls": d.calls,
            "probe": {"split": "CIFAR-100 test", "per_class": PROBE_PER_CLASS, "n": PROBE_PER_CLASS * C.N_CLASSES,
                      "rng_role": PROBE_ROLE, "preprocessing": "the arm's own X_test (std; CH/CHB0 also minus "
                      "the train mean image)",
                      "idx_sha256": hashlib.sha256(d.probe_idx.numpy().astype(np.int32).tobytes()).hexdigest()
                      if d.probe_idx is not None else None},
            "snap_every": SNAP, "pplus_bins": BIN_TEXT,
            "diag_files": {"diag_units.npz": "per-unit arrays (task, slot, unit); slots in `seeds` order",
                           "diag_per_task.csv": "per (seed, task) summaries and learning windows"}})
    prov = {**prov, **extra}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    return prov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "smoke"])
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto")
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
        run_cell(a.arm, out, device, allow_dirty=a.allow_dirty)
    else:                                            # spec §7 S-smoke
        out = Path(a.out) if a.out else OUT_ROOT / "smoke" / cell_name("CHB0")
        t0 = time.time()
        prov = run_cell("CHB0", out, device, seeds=[100], n_tasks=3, allow_dirty=a.allow_dirty)
        print(f"smoke CHB0 seed 100, 3 tasks x {STEPS} + fresh: {time.time() - t0:.1f} s, "
              f"last task {prov['step_ms_last_task']:.3f} ms/step, hooks {prov['hook_calls']}", flush=True)


if __name__ == "__main__":
    main()

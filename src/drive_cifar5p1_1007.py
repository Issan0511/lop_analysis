#!/usr/bin/env python3
"""drive_cifar5p1_1007 -- A2's self / upstream / total transport on the 5+1 CIFAR x MLP box (R/std).

Spec: specs/spec_drive_cifar5p1_1007.md (registered at 0afd3360).

The training is `cifar5p1_mlp_0920.run()` for arm R, cond std, lr 1e-4 and every other default,
copied operation for operation.  The host file is not modified; S-host checks this copy against the
unmodified host bit for bit.  Around every CUDA-graph replay an `Observer` reads and never writes:

  before the replay   copies of P/m/v, and an eager re-run of the very batch's forward/backward
  after the replay    the task's whole image set through the net at the new parameters, then
                        S  = dW2.mu_old + db2            self movement of layer 2      (spec §4.2)
                        U  = W2_new.(mu_new - mu_old)    upstream movement
                        Dm = S + U                       total movement
                        S1 = dW1.mu_x + db1              layer 1 (U1 = 0: x is fixed)  (spec §4.4)
                      the conf/label/history split of S (A2 §4.4, spec §4.5) and every bound of §9

Everything is folded into 30 bins of 26 updates on the GPU; nothing per update reaches the CPU except
the four fixed audit fixtures.  `mutate` holds check-only mutations; production asserts it is empty.

    python -m src.drive_cifar5p1_1007 production --out results/drive_cifar5p1_1007/raw/production \
        --checks results/drive_cifar5p1_1007/checks.json --production-go
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src import cifar5p1_mlp_0920 as C  # noqa: E402
from src import pmnist_0905 as H  # noqa: E402
from src import rlcifar_mlp_battle_0918 as B  # noqa: E402

RUN = "drive_cifar5p1_1007"
SPEC_COMMIT = "0afd3360"
DATA = Path("/home/issan/Projects/claude/proj_004_drift/data/cifar100/cifar-100-python.tar.gz")
DATA_SHA256 = "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7"
RECORD = ROOT / "results" / "cifar5p1_mlp_0920" / "R_std_lr0.0001"
OUT = ROOT / "results" / RUN
RAW = OUT / "raw"
TORCH = "2.13.0+cu130"
LOCK = Path("/tmp/lop_analysis_gpu.lock")

ARM, COND, LR = "R", "std", 1e-4
B1, B2, EPS = 0.9, 0.999, 1e-8
STEPS, NT, BATCH, NCLS, HID = C.STEPS_PER_TASK, C.N_TASKS, C.BATCH, C.N_CLASSES, C.HIDDEN
BIN, NBIN = 26, 30
assert BIN * NBIN == STEPS
PRODUCTION = list(range(10))
GROUPS_PRODUCTION = {"primary": list(range(5)), "calibration": list(range(5, 10))}
CHECK_SEEDS = list(range(100, 110))
GROUPS_CHECK = {"check": list(range(10))}
FIXTURES = ((2, 1), (2, 780), (3, 1), (30, 780))

U32, U64 = 2.0 ** -24, 2.0 ** -53
TINY32 = float(torch.finfo(torch.float32).tiny)   # also bounds a flushed float32 subnormal
TINY64 = float(torch.finfo(torch.float64).tiny)
B1F, A1F = float(np.float32(B1)), float(np.float32(1 - B1))   # what the float32 Adam kernels apply
LR32, EPS32 = float(np.float32(LR)), float(np.float32(EPS))

TRANSPORT = ("S2", "U2", "D2", "S2c", "S2l", "S2h", "S1")
PARTS = ("sum", "pos", "neg", "npos", "nneg")
FLAGS = ("bad_closure", "bad_native", "bad_grad", "bad_adam", "bad_p", "bad_comp", "bad_label", "nonfinite")
RATIOS = ("closure", "native", "grad", "p", "comp", "label")
KEYS = tuple(f"{k}_{p}" for k in TRANSPORT for p in PARTS) + FLAGS + ("updates",)
UNIT_KEYS = ("S2", "U2", "D2")
SOURCE_FILES = ("src/drive_cifar5p1_1007.py", "src/cifar5p1_mlp_0920.py", "src/rlcifar_mlp_battle_0918.py",
                "src/pmnist_0905.py", "src/pmnist_rlcifar_0907.py",
                "analysis/drive_cifar5p1_1007/checks.py", "analysis/drive_cifar5p1_1007/report.py",
                "analysis/drive_cifar5p1_1007/launch.sh")


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------

def sha(p) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for b in iter(lambda: fh.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sources() -> dict:
    return {p: sha(ROOT / p) for p in SOURCE_FILES}


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def put(p, data) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(p)


def save_npz(p, data) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, **data)
    tmp.replace(p)


def git(*args) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def setup():
    device = H.setup("cuda")               # deterministic algorithms, cudnn.benchmark off
    torch.set_num_threads(2)               # the host CLI default; eff_rank's LAPACK depends on it
    assert torch.__version__ == TORCH, torch.__version__
    assert not torch.backends.cuda.matmul.allow_tf32, "TF32 is outside the registered roundoff model"
    assert torch.get_float32_matmul_precision() == "highest"
    return device


def load_cifar():
    cifar = C.Cifar100(DATA)
    assert cifar.sha256 == {DATA.name: DATA_SHA256}, cifar.sha256
    return cifar


def gpu_free_gb() -> float:
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True).stdout.split()
    return float(q[0]) / 1024


def other_gpu_python() -> list[int]:
    """PIDs of other python compute processes on the GPU (the desktop's C+G clients are not counted)."""
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True).stdout.split()
    pids = []
    for x in q:
        if not x.strip().isdigit() or int(x) == os.getpid():
            continue
        try:
            cmd = Path(f"/proc/{int(x)}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError):
            continue
        if "python" in cmd:
            pids.append(int(x))
    return pids


@contextlib.contextmanager
def gpu_lock(need_gb: float, reserve_gb: float = 6.0, poll_s: float = 30.0, timeout_s: float = 6 * 3600):
    """One GPU job at a time on the machine: the shared flock, no other python compute process on the
    GPU, and `need_gb + reserve_gb` free.  Waits (polling) instead of failing; gives up after timeout_s."""
    t0 = time.time()
    with LOCK.open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        while True:
            others, free = other_gpu_python(), gpu_free_gb()
            if not others and free >= need_gb + reserve_gb:
                break
            assert time.time() - t0 < timeout_s, f"GPU busy for {timeout_s} s: {others}, free {free:.1f} GB"
            print(f"[{time.strftime('%T')}] waiting for the GPU: other python {others}, free {free:.1f} GB",
                  flush=True)
            time.sleep(poll_s)
        yield


def environment() -> dict:
    return dict(torch=torch.__version__, numpy=np.__version__, python=sys.version, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(), tf32=torch.backends.cuda.matmul.allow_tf32,
                matmul_precision=torch.get_float32_matmul_precision(),
                deterministic=torch.are_deterministic_algorithms_enabled(), threads=torch.get_num_threads(),
                cublas=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))


# --------------------------------------------------------------------------
# numerics (spec §4, §9).  Arrays carry a leading R axis.  Nothing here writes to its inputs.
# --------------------------------------------------------------------------

def gamma(n, u=U64):
    return n * u / (1 - n * u)


def aug(w, b):
    return torch.cat((w.detach().double(), b.detach().double().unsqueeze(-1)), -1)


def kvec(mu):
    return torch.cat((mu, torch.ones_like(mu[:, :1])), -1)


@torch.no_grad()
def features(P, X):
    """Native float32 layer-1/2 forward on the whole image set X (R, N, 3072), read in float64.

    mu   float64 mean of the native h = clamp(z1, 0) and its error against the exact mean of those h
    nat  float64 mean of the native float32 z2 and its bound against w.mean(h) + b (spec §9)
    gate fraction of images with z2 >= 0 (clamp's native derivative is 1 at exactly 0)
    """
    z1 = torch.baddbmm(P[1][:, None, :], X, P[0].transpose(1, 2))
    h = z1.clamp(min=0.0)
    z2 = torch.baddbmm(P[3][:, None, :], h, P[2].transpose(1, 2))
    N, K = h.shape[1], h.shape[2]
    mu = h.sum(1, dtype=torch.float64) / N
    mue = gamma(N + 2) * mu + TINY64                     # h >= 0, so mean|h| = mu
    nat = z2.sum(1, dtype=torch.float64) / N
    absz = z2.abs().sum(1, dtype=torch.float64) / N
    w_abs, b_abs = P[2].detach().double().abs(), P[3].detach().double().abs()
    nate = (gamma(K + 3, U32) * ((w_abs * mu[:, None, :]).sum(-1) + b_abs) * (1 + gamma(K + 2))
            + gamma(N + 2) * absz + (K + 3) * TINY32 + TINY64)
    gate = (z2 >= 0).sum(1, dtype=torch.float64) / N
    return dict(mu=mu, mue=mue, nat=nat, nate=nate, gate=gate)


def measure2(before, after, f0, f1):
    """Layer-2 S, U, Dm = S + U, the float64 total m2(new) - m2(old) and both closures (spec §4.2)."""
    mu0, mu1 = f0["mu"], f1["mu"]
    K = mu0.shape[-1]
    k0 = kvec(mu0)[:, None, :]
    k1 = kvec(mu1)[:, None, :]
    dp = after - before
    w_new = after[..., :-1]
    dmu = (mu1 - mu0)[:, None, :]
    S = (dp * k0).sum(-1)
    U = (w_new * dmu).sum(-1)
    D = S + U
    new = (after * k1).sum(-1)
    old = (before * k0).sum(-1)
    total = new - old
    g = gamma(2 * (K + 1) + 2)
    mag_new, mag_old = (after.abs() * k1).sum(-1), (before.abs() * k0).sum(-1)
    # algebraic closure: an identity for any mu, so only float64 rounding enters
    alg = (g * ((dp.abs() * k0).sum(-1) + (w_new.abs() * dmu.abs()).sum(-1) + mag_new + mag_old)
           + gamma(4) * (new.abs() + old.abs() + S.abs() + U.abs()) + 4 * TINY64)
    # native closure: the float32 z2 means against the float64 total, mu errors propagated through |w|
    nat = (f0["nate"] + f1["nate"] + (w_new.abs() * f1["mue"][:, None, :]).sum(-1)
           + (before[..., :-1].abs() * f0["mue"][:, None, :]).sum(-1) + g * (mag_new + mag_old)
           + gamma(4) * (total.abs() + (f1["nat"] - f0["nat"]).abs()) + 4 * TINY64)
    return dict(S=S, U=U, D=D, total=total, new=new, old=old,
                closure=(total - D).abs(), alg=alg,
                native=((f1["nat"] - f0["nat"]) - total).abs(), nat=nat)


def decompose(a1, z2, z3, y, W3):
    """conf/label split of the layer-2 CE gradient in float64 on the batch, and its float32 bound gb."""
    a1, z2, z3, W3 = a1.double(), z2.double(), z3.double(), W3.detach().double()
    R, Bn, _ = z2.shape
    Cn = W3.shape[1]
    p = torch.softmax(z3, -1)
    gate = (z2 >= 0).double()
    ht = torch.cat((a1, torch.ones_like(a1[..., :1])), -1)
    onehot = F.one_hot(y, Cn).double()
    dconf = (p - 1.0 / Cn) / Bn
    dlab = (1.0 / Cn - onehot) / Bn
    gconf = torch.bmm((torch.bmm(dconf, W3) * gate).transpose(1, 2), ht)
    glab = torch.bmm((torch.bmm(dlab, W3) * gate).transpose(1, 2), ht)
    ar = torch.arange(R, device=y.device)[:, None]
    absu = torch.bmm(p + 1.0 / Cn, W3.abs()) + W3.abs()[ar, y]
    n = 2 * Cn + Bn + 16
    gb = gamma(n, U32) * torch.bmm((absu * gate).transpose(1, 2), ht.abs()) / Bn + n * TINY32
    return gconf, glab, gb


def conf_step(state, gconf, glab, gb, gact, m_old, m_new, v_new, inv1, inv2, before, after, k0, S):
    """One update of the history/conf/label moments and the projected self movements (spec §4.5)."""
    Mc = B1F * state["Mc"] + A1F * gconf
    Mh = B1F * state["Mh"]
    Ml = B1F * state["Ml"] + A1F * glab                  # label EMA integrated directly
    Mlab = m_new - Mc - Mh                               # label moment as the residual (A2)
    E = (B1F * state["E"] + A1F * gb + gamma(2, U32) * (B1F * m_old.abs() + A1F * gact.abs()) + 2 * TINY32
         + gamma(4) * (Mc.abs() + Mh.abs() + Ml.abs() + m_new.abs() + Mlab.abs()))
    pre = 1.0 / (torch.sqrt(v_new * inv2) + EPS32)
    scale = -LR32 * inv1 * pre
    ideal = scale * m_new
    actual = after - before
    pd = actual - ideal
    pb = gamma(7, U32) * (ideal.abs() + after.abs()) + 7 * TINY32 + gamma(8) * ideal.abs()
    Sc = (scale * Mc * k0).sum(-1)
    Sh = (scale * Mh * k0).sum(-1)
    Sl = (scale * Mlab * k0).sum(-1)
    K = k0.shape[-1]
    mag = ((scale.abs() * (Mc.abs() + Mh.abs() + Mlab.abs() + m_new.abs()) + actual.abs() + ideal.abs()) * k0).sum(-1)
    # float64 slack: the three projected sums, the residual Mlab, S itself and pd (each <= gamma_{K+2})
    comp_err = (pd.abs() * k0).sum(-1) + gamma(2 * K + 10) * mag + 8 * TINY64
    return dict(Mc=Mc, Mh=Mh, Ml=Ml, E=E, Sc=Sc, Sh=Sh, Sl=Sl,
                p_ratio=(pd.abs() / pb).amax(-1),
                comp_ratio=(S - Sc - Sh - Sl).abs() / comp_err,
                label_ratio=((Mlab - Ml).abs() / E).amax(-1))


def adam_replay(p_old, m_old, v_old, g, inv_c1, inv_c2):
    """The host's Adam expressions, out of place, for the bit-exact replay check."""
    m = m_old.mul(B1).add(g, alpha=1 - B1)
    v = v_old.mul(B2).addcmul(g, g, value=1 - B2)
    p = p_old.sub(LR * (m * inv_c1) / ((v * inv_c2).sqrt() + EPS))
    return p, m, v


# --------------------------------------------------------------------------
# the observer
# --------------------------------------------------------------------------

class Observer:
    """Read-only every-update observation of R stacked slots, folded into bins on the GPU."""

    def __init__(self, R, device, fixtures=FIXTURES, every=1, image_limit=None):
        self.R, self.dev = R, device
        self.fixtures = set(fixtures)
        self.every, self.image_limit = every, image_limit
        z = lambda *s: torch.zeros(*s, device=device, dtype=torch.float64)  # noqa: E731
        self.acc = z(R, NBIN, len(KEYS))
        self.rmax = z(R, NBIN, len(RATIOS))
        self.unit_bin = z(R, NBIN, HID, len(UNIT_KEYS))
        self.unit_task = z(R, HID, len(TRANSPORT))
        self.tel_err, self.tel_abs = z(R, HID), z(R, HID)
        self.mu_last = self.mux = self.m2_end_prev = None
        self.saved = []
        self.measure2 = measure2
        self.decompose = decompose
        self.conf_step = conf_step

    def bin_of(self, j):
        return (j - 1) // BIN

    @torch.no_grad()
    def begin_task(self, t, hard, X, P, m):
        if self.image_limit:
            X = X[:, : self.image_limit]
        R, N = self.R, X.shape[1]
        self.t, self.hard, self.X, self.N = t, hard, X, N
        mux = X.sum(1, dtype=torch.float64) / N
        f = features(P, X)
        a1k, a2k = aug(P[0], P[1]), aug(P[2], P[3])
        self.m2_start = (a2k * kvec(f["mu"])[:, None, :]).sum(-1)       # same expression as measure2's `old`
        self.m1_start = (a1k * kvec(mux)[:, None, :]).sum(-1)
        if self.mu_last is not None:
            w2, w1 = a2k[..., :-1], a1k[..., :-1]
            dmu = f["mu"] - self.mu_last
            self.J2 = (w2 * dmu[:, None, :]).sum(-1)
            self.J1 = (w1 * (mux - self.mux)[:, None, :]).sum(-1)
            self.J2_defect = ((self.m2_start - self.m2_end_prev) - self.J2).abs()
            self.J2_bound = (gamma(2 * (HID + 1) + 2) * ((a2k.abs() * kvec(f["mu"])[:, None, :]).sum(-1)
                                                         + (a2k.abs() * kvec(self.mu_last)[:, None, :]).sum(-1)
                                                         + (w2.abs() * dmu.abs()[:, None, :]).sum(-1))
                             + gamma(4) * (self.m2_start.abs() + self.m2_end_prev.abs() + self.J2.abs()) + 4 * TINY64)
            self.mu_last_prev = self.mu_last
        else:
            nan = torch.full((R, HID), float("nan"), device=self.dev, dtype=torch.float64)
            self.J2, self.J1, self.J2_defect, self.J2_bound = nan, nan.clone(), nan.clone(), nan.clone()
            self.mu_last_prev = None
        self.mux, self.cur = mux, f
        self.Mc = torch.zeros_like(a2k)
        self.Mh = aug(m[2], m[3])                                      # the moment the task inherits
        self.Ml = torch.zeros_like(a2k)
        self.E = torch.zeros_like(a2k)
        for x in (self.acc, self.rmax, self.unit_bin, self.unit_task, self.tel_err, self.tel_abs):
            x.zero_()
        self.j = 0
        self.last_new = self.m2_start.clone()

    def before(self, P, m, v, idx, X_all, Y_all, act):
        self.j += 1
        self.observing = (self.j - 1) % self.every == 0
        if not self.observing:
            return
        with torch.no_grad():
            self.p_old = [q.detach().clone() for q in P]
            self.m_old = [q.clone() for q in m]
            self.v_old = [q.clone() for q in v]
        xb, yb = X_all[idx], Y_all[idx]
        with torch.enable_grad():
            z1, a1, z2, a2, z3 = B.forward(P, xb, act, train=True)
            lossv = F.cross_entropy(z3.reshape(-1, NCLS), yb.reshape(-1),
                                    reduction="none").view(self.R, BATCH).mean(1)
            grads = torch.autograd.grad(lossv.sum(), P)
        self.batch = (a1.detach(), z2.detach(), z3.detach(), yb)
        self.grads = [g.detach() for g in grads]

    @torch.no_grad()
    def after(self, P, m, v, inv_c1, inv_c2):
        if not self.observing:
            return
        R = self.R
        f0, f1 = self.cur, features(P, self.X)
        before2, after2 = aug(self.p_old[2], self.p_old[3]), aug(P[2], P[3])
        r = self.measure2(before2, after2, f0, f1)
        dW1 = P[0].detach().double() - self.p_old[0].double()
        db1 = P[1].detach().double() - self.p_old[1].double()
        S1 = (dW1 * self.mux[:, None, :]).sum(-1) + db1
        bad_adam = torch.zeros(R, device=self.dev, dtype=torch.float64)
        for i in range(6):
            pr, mr, vr = adam_replay(self.p_old[i], self.m_old[i], self.v_old[i], self.grads[i], inv_c1, inv_c2)
            for x, y in ((pr, P[i].detach()), (mr, m[i]), (vr, v[i])):
                bad_adam += (x != y).flatten(1).any(1).double()
        a1, z2, z3, yb = self.batch
        gconf, glab, gb = self.decompose(a1, z2, z3, yb, self.p_old[4])
        gact = aug(self.grads[2], self.grads[3])
        grad_ratio = ((gact - gconf - glab).abs() / gb).amax(-1)
        k0 = kvec(f0["mu"])[:, None, :]
        prev = dict(Mc=self.Mc, Mh=self.Mh, Ml=self.Ml, E=self.E)
        c = self.conf_step(prev, gconf, glab, gb, gact, aug(self.m_old[2], self.m_old[3]), aug(m[2], m[3]),
                           aug(v[2], v[3]), inv_c1.double(), inv_c2.double(), before2, after2, k0, r["S"])
        self.Mc, self.Mh, self.Ml, self.E = c["Mc"], c["Mh"], c["Ml"], c["E"]
        vals = dict(S2=r["S"], U2=r["U"], D2=r["D"], S2c=c["Sc"], S2l=c["Sl"], S2h=c["Sh"], S1=S1)
        ratios = [r["closure"] / r["alg"], r["native"] / r["nat"], grad_ratio, c["p_ratio"],
                  c["comp_ratio"], c["label_ratio"]]
        cols = []
        for k in TRANSPORT:
            x = vals[k]
            cols += [x.sum(-1), x.clamp(min=0).sum(-1), x.clamp(max=0).sum(-1),
                     (x > 0).sum(-1).double(), (x < 0).sum(-1).double()]
        finite = torch.stack([torch.isfinite(x) for x in list(vals.values()) + ratios]).all(0)
        cols += [(ratios[0] > 1).sum(-1).double(), (ratios[1] > 1).sum(-1).double(),
                 (ratios[2] > 1).sum(-1).double(), bad_adam,
                 (ratios[3] > 1).sum(-1).double(), (ratios[4] > 1).sum(-1).double(),
                 (ratios[5] > 1).sum(-1).double(), (~finite).sum(-1).double(),
                 torch.ones(R, device=self.dev, dtype=torch.float64)]
        b = self.bin_of(self.j)
        self.acc[:, b] += torch.stack(cols, -1)
        self.rmax[:, b] = torch.maximum(self.rmax[:, b], torch.stack([x.amax(-1) for x in ratios], -1))
        self.unit_bin[:, b] += torch.stack([r["S"], r["U"], r["D"]], -1)
        self.unit_task += torch.stack([vals[k] for k in TRANSPORT], -1)
        self.tel_err += r["alg"] + U64 * r["total"].abs()
        self.tel_abs += r["D"].abs()
        self.last_new = r["new"]
        if (self.t, self.j) in self.fixtures:
            cpu = lambda x: x.detach().cpu().clone()  # noqa: E731
            self.saved.append(dict(
                t=self.t, j=self.j, hard=int(self.hard), N=self.N,
                p_old=[cpu(q) for q in self.p_old[:5]], p_new=[cpu(q) for q in P[:5]],
                m_old2=cpu(aug(self.m_old[2], self.m_old[3])), m_new2=cpu(aug(m[2], m[3])),
                v_new2=cpu(aug(v[2], v[3])), inv_c1=cpu(inv_c1), inv_c2=cpu(inv_c2),
                batch=[cpu(x) for x in self.batch], grads2=cpu(gact),
                gconf=cpu(gconf), glab=cpu(glab), gb=cpu(gb),
                state_prev={k: cpu(x) for k, x in prev.items()},
                state_new={k: cpu(c[k]) for k in ("Mc", "Mh", "Ml", "E")},
                f0={k: cpu(x) for k, x in f0.items()}, f1={k: cpu(x) for k, x in f1.items()},
                mux=cpu(self.mux),
                values={k: cpu(x) for k, x in vals.items()} | {"total": cpu(r["total"])},
                J2=cpu(self.J2) if self.j == 1 else None, J1=cpu(self.J1) if self.j == 1 else None,
                mu_last_prev=cpu(self.mu_last_prev) if (self.j == 1 and self.mu_last_prev is not None) else None))
        self.cur = f1

    @torch.no_grad()
    def end_task(self, P):
        a1k = aug(P[0], P[1])
        m2_end = self.last_new
        m1_end = (a1k * kvec(self.mux)[:, None, :]).sum(-1)
        dsum = self.unit_task[..., TRANSPORT.index("D2")]
        tel_defect = (dsum - (m2_end - self.m2_start)).abs()
        tel_bound = (self.tel_err + gamma(STEPS + 2) * self.tel_abs
                     + gamma(2) * (m2_end.abs() + self.m2_start.abs()) + 4 * TINY64)
        out = dict(acc=self.acc, rmax=self.rmax, unit_bin=self.unit_bin, unit_task=self.unit_task,
                   J2=self.J2, J1=self.J1, J2_defect=self.J2_defect, J2_bound=self.J2_bound,
                   m2_start=self.m2_start, m2_end=m2_end, m1_start=self.m1_start, m1_end=m1_end,
                   tel_defect=tel_defect, tel_bound=tel_bound, gate2_end=self.cur["gate"], mu_end=self.cur["mu"])
        res = {k: x.detach().cpu().numpy() for k, x in out.items()}
        res.update(task=self.t, hard=int(self.hard), N=self.N, updates=self.j, every=self.every)
        self.mu_last, self.m2_end_prev = self.cur["mu"], m2_end
        return res


# --------------------------------------------------------------------------
# the stacked run: cifar5p1_mlp_0920.run(ARM, seeds, COND, ...) with read-only hooks
# --------------------------------------------------------------------------

def run(seeds, n_tasks, device, out, cifar, observer=None, graph=True, fresh=True, progress=None,
        mutate=frozenset(), on_task_end=None, rng_probe=None, state_hook=None) -> dict:
    """The host's run() for arm R / std / lr 1e-4 / iv none / hidden 100, line for line.

    Hooks: `observer` (read only), `on_task_end(t, info)` (after the host's rows for task t),
    `rng_probe` (list: global RNG unchanged across the observer, first 2 updates of every task),
    `state_hook(t, P, m, v)` (called at the end of every task, for checks).
    `mutate` is check-only (S-host, S-graph): {"obs_rng", "obs_write", "adam_no_bias",
    "no_warmup_restore", "stale_idx"}.
    """
    t_start = time.time()
    progress = progress or (lambda msg: print(msg, flush=True))
    lr = LR
    act = C.make_act(ARM, HID, 0.6, 0.01, 0.005, 3.0, 0.0)
    R = len(seeds)
    X_all = cifar.inputs("train", COND, device)
    Y_all = cifar.train_y.to(device)
    X_test = cifar.inputs("test", COND, device)
    Y_test = cifar.test_y.to(device)
    tr_rows, te_rows = C.class_rows(cifar.train_y), C.class_rows(cifar.test_y)
    plans = [C.task_plan(s) for s in seeds]
    g_batch = {s: H.stream("c51_batch", s) for s in seeds}

    init = [q.detach() for s in seeds for q in C.init_params(ARM, s, device, HID)]
    P = [torch.stack(init[i::6]).contiguous() for i in range(6)]
    P = [q.requires_grad_(True) for q in P]
    P0 = [q.detach().clone() for q in P]
    act.init_state(R, device, key=f"{ARM}|{seeds}|{COND}")
    act0 = act.state()
    adam_m = [torch.zeros_like(q) for q in P]
    adam_v = [torch.zeros_like(q) for q in P]

    b1, b2, eps = B1, B2, EPS
    static_idx = torch.zeros(R, BATCH, dtype=torch.long, device=device)
    inv_c1 = torch.zeros((), device=device)
    inv_c2 = torch.zeros((), device=device)
    step_t = torch.zeros((), dtype=torch.long, device=device)
    acc_sum = torch.zeros(R, device=device)
    bad_step = torch.full((R,), -1, dtype=torch.long, device=device)
    last_hit = torch.zeros(R, device=device)
    post = getattr(act, "post_update", None)

    def step():
        xb, yb = X_all[static_idx], Y_all[static_idx]
        z1, a1, z2, a2, z3 = B.forward(P, xb, act, train=True)
        lossv = F.cross_entropy(z3.reshape(-1, NCLS), yb.reshape(-1),
                                reduction="none").view(R, BATCH).mean(1)
        hit = (z3.detach().argmax(-1) == yb).float().mean(1)
        acc_sum.add_(hit)
        last_hit.copy_(hit)
        grads = torch.autograd.grad(lossv.sum(), P)
        with torch.no_grad():
            bad = ~torch.isfinite(lossv)
            bad_step.copy_(torch.where((bad_step < 0) & bad, step_t, bad_step))
            step_t.add_(1)
            for p, p0, gr, mi, vi in zip(P, P0, grads, adam_m, adam_v):
                mi.mul_(b1).add_(gr, alpha=1 - b1)
                vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                p.sub_(lr * (mi * inv_c1) / ((vi * inv_c2).sqrt() + eps))
            act.update(z1.detach(), z2.detach())
            if post is not None:
                post(P, lr)

    use_graph = graph and device.type == "cuda"
    cg = None
    rolled_back = None
    if use_graph:
        keep = [q.detach().clone() for q in (*P, *adam_m, *adam_v, acc_sum, bad_step, step_t, last_hit)]
        keep_act = act.state()
        inv_c1.fill_(1.0); inv_c2.fill_(1.0)
        static_idx.zero_()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                act.begin_step(R, BATCH, device)
                step()
        torch.cuda.current_stream().wait_stream(side)
        cg = torch.cuda.CUDAGraph()
        act.begin_step(R, BATCH, device)
        with torch.cuda.graph(cg):
            step()
        if "no_warmup_restore" not in mutate:
            with torch.no_grad():
                for q, v in zip((*P, *adam_m, *adam_v, acc_sum, bad_step, step_t, last_hit), keep):
                    q.copy_(v)
            act.load_state(keep_act)
        rolled_back = all(torch.equal(q.detach(), v) for q, v in
                          zip((*P, *adam_m, *adam_v, acc_sum, bad_step, step_t, last_hit), keep))
        del keep

    def train_task(batches, tc0, obs=None, t=None):
        tc = tc0
        acc_sum.zero_()
        bad_step.fill_(-1)
        step_t.zero_()
        for j in range(batches.shape[1]):
            if not ("stale_idx" in mutate and cg is not None and t == 2 and j == 4):
                static_idx.copy_(batches[:, j])
            tc += 1
            if "adam_no_bias" in mutate:
                inv_c1.fill_(1.0); inv_c2.fill_(1.0)
            else:
                inv_c1.fill_(1.0 / (1 - b1 ** tc))
                inv_c2.fill_(1.0 / (1 - b2 ** tc))
            act.begin_step(R, BATCH, device)
            probe = obs is not None and rng_probe is not None and j < 2
            if probe:
                s0 = (torch.get_rng_state(), torch.cuda.get_rng_state())
            if obs is not None:
                obs.before(P, adam_m, adam_v, static_idx, X_all, Y_all, act)
            if cg is not None:
                cg.replay()
            else:
                step()
            if obs is not None:
                obs.after(P, adam_m, adam_v, inv_c1, inv_c2)
                if "obs_write" in mutate and t == 2 and j == 0:
                    with torch.no_grad():
                        P[3][0, 0] += 2.0 ** -20
            if probe:
                s1 = (torch.get_rng_state(), torch.cuda.get_rng_state())
                rng_probe.append(bool(torch.equal(s0[0], s1[0]) and torch.equal(s0[1], s1[1])))
        if device.type == "cuda":
            torch.cuda.synchronize()
        return tc

    alive = torch.ones(R, dtype=torch.bool, device=device)
    rows, diverged, tc = [], [], 0
    last_hard = max(t for t in range(1, n_tasks + 1) if t % 2 == 1)
    saved = {}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    step_ms = float("nan")
    batch_sha = []

    for t in range(1, n_tasks + 1):
        hard = t % 2 == 1
        steps = STEPS
        rows_t = [torch.cat([tr_rows[q] for q in plans[r][t - 1][1]]) for r in range(R)]
        if "obs_rng" in mutate and t == 2:
            torch.randperm(2, generator=g_batch[seeds[0]])        # an observer that consumed the stream
        batches = torch.stack([C.batch_indices(g_batch[s], rows_t[r], steps)
                               for r, s in enumerate(seeds)]).to(device)
        batch_sha.append(hashlib.sha256(batches.cpu().numpy().tobytes()).hexdigest())
        if t == last_hard and fresh:
            saved[t] = batches.clone()
        Xo = None
        if observer is not None:
            Xo = torch.stack([X_all[rows_t[r]] for r in range(R)])
            observer.begin_task(t, hard, Xo, P, adam_m)
        t0 = time.time()
        tc = train_task(batches, tc, observer, t)
        step_ms = 1e3 * (time.time() - t0) / steps
        shard = observer.end_task(P) if observer is not None else None
        del Xo

        with torch.no_grad():
            finite = torch.stack([torch.isfinite(q).flatten(1).all(1) for q in P]).all(0)
            newly = alive & ((bad_step >= 0) | ~finite)
            for r in torch.nonzero(newly).flatten().tolist():
                diverged.append({"diverged": True, "task": t, "seed": seeds[r], "arm": ARM,
                                 "cond": COND, "step": (t - 1) * STEPS + max(int(bad_step[r]), 0)})
                rows.append({"arm": ARM, "cond": COND, "seed": seeds[r], "slot": r, "lr": lr,
                             "task": t, "hard": int(hard), "acc": float("nan")})
            alive &= ~newly
            live = torch.nonzero(alive).flatten().tolist()
            Xt = torch.stack([X_all[rows_t[r]] for r in range(R)])
            Yt = torch.stack([Y_all[rows_t[r]] for r in range(R)])
            m = C.evaluate(P, Xt, Yt, act, live)
            del Xt
            te = [torch.cat([te_rows[q] for q in plans[r][t - 1][1]]) for r in range(R)]
            test_acc = C.accuracy(P, torch.stack([X_test[q] for q in te]),
                                  torch.stack([Y_test[q] for q in te]), act)
        for r in live:
            rows.append({"arm": ARM, "cond": COND, "seed": seeds[r], "slot": r, "lr": lr,
                         "task": t, "hard": int(hard), "n_classes": len(plans[r][t - 1][1]),
                         "online_acc": float(acc_sum[r]) / steps,
                         "train_acc": m[r]["acc"], "test_acc": float(test_acc[r]), **m[r]})
        rows.sort(key=lambda q: (q["slot"], q["task"]))
        H.write_csv(out / "per_task.csv", rows)
        if state_hook is not None:
            state_hook(t, P, adam_m, adam_v)
        if on_task_end is not None:
            on_task_end(t, shard)
        on = acc_sum[alive] / steps
        el = time.time() - t_start
        progress(f"[{time.strftime('%T')}] {ARM}/{COND} task {t:2d}/{n_tasks} "
                 f"{'hard' if hard else 'easy'} alive {int(alive.sum())}/{R} "
                 f"online {float(on.mean()) if len(on) else float('nan'):.3f} "
                 f"test {float(test_acc[alive].mean()) if int(alive.sum()) else float('nan'):.3f} "
                 f"{step_ms:.2f} ms/step  {el/60:.1f} min")

    fresh_rows = []
    if fresh and last_hard in saved:
        continual = {int(q["seed"]): q["online_acc"] for q in rows
                     if q["task"] == last_hard and "online_acc" in q}
        with torch.no_grad():
            for q, v in zip(P, P0):
                q.copy_(v)
            for q in (*adam_m, *adam_v):
                q.zero_()
        act.load_state(act0)
        alive_f = torch.ones(R, dtype=torch.bool, device=device)
        train_task(saved[last_hard], 0)
        with torch.no_grad():
            finite = torch.stack([torch.isfinite(q).flatten(1).all(1) for q in P]).all(0)
            alive_f &= (bad_step < 0) & finite
        for r in range(R):
            if not bool(alive_f[r]) or seeds[r] not in continual:
                continue
            value = float(acc_sum[r]) / STEPS
            fresh_rows.append({"arm": ARM, "cond": COND, "seed": seeds[r], "lr": lr,
                               "task": last_hard, "fresh_online_acc": value,
                               "continual_online_acc": continual[seeds[r]],
                               "fresh_gap": value - continual[seeds[r]]})
        H.write_csv(out / "fresh_control.csv", fresh_rows)
    return dict(rows=rows, fresh_rows=fresh_rows, diverged=diverged, tc=tc, batch_sha=batch_sha,
                rolled_back=rolled_back, wall_clock_s=time.time() - t_start,
                step_ms_last_task=step_ms, graph=use_graph)


# --------------------------------------------------------------------------
# run identity, shards, production
# --------------------------------------------------------------------------

def identity(mode, seeds, n_tasks, cifar, groups, fresh=True) -> dict:
    return dict(run_id=RUN, mode=mode, seeds=list(seeds), R=len(seeds), groups=groups, n_tasks=n_tasks,
                fresh=fresh, arm=ARM, cond=COND, lr=LR, bins=dict(size=BIN, count=NBIN), steps=STEPS,
                fixtures=[list(x) for x in FIXTURES], spec_commit=SPEC_COMMIT, source_sha256=sources(),
                data_sha256=cifar.sha256,
                class_sha256={str(s): hashlib.sha256(np.asarray(C.seed_classes(s), dtype=np.int64).tobytes()).hexdigest()
                              for s in seeds})


def group_shard(shard, slots, seeds) -> dict:
    d = {}
    for k, v in shard.items():
        d[k] = v[slots] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == len(seeds) else np.asarray(v)
    d.update(seeds=np.asarray([seeds[s] for s in slots]), keys=np.asarray(KEYS), ratios=np.asarray(RATIOS),
             transport=np.asarray(TRANSPORT), unit_keys=np.asarray(UNIT_KEYS))
    return d


def observed_run(out, mode, seeds, n_tasks, groups, cifar, device, fresh=True, graph=True,
                 observer=None, mutate=frozenset(), progress=None, state_hook=None) -> dict:
    """Run + shards + fixtures + cost + complete marker.  Shards never print scientific values."""
    out = Path(out)
    assert not out.exists() or not any(out.iterdir()), f"{out} is not empty"
    out.mkdir(parents=True, exist_ok=True)
    ident = identity(mode, seeds, n_tasks, cifar, groups, fresh)
    put(out / "input_manifest.json", ident)
    put(out / "provenance_start.json", dict(identity=ident, git_hash=git("rev-parse", "HEAD"),
                                            dirty=git("status", "--porcelain", "src", "analysis", "specs"),
                                            utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                            pid=os.getpid(), environment=environment(), mutate=sorted(mutate)))
    obs = observer or Observer(len(seeds), device)
    files, flags_total, counts = [], np.zeros(len(FLAGS)), []
    fl = [KEYS.index(k) for k in FLAGS]

    def on_task_end(t, shard):
        nonlocal flags_total
        for group, slots in groups.items():
            p = out / group / f"t{t:02d}.npz"
            save_npz(p, group_shard(shard, slots, seeds))
            files.append(dict(path=str(p.relative_to(out)), sha256=sha(p), kind=group))
        flags_total = flags_total + shard["acc"][..., fl].sum(axis=(0, 1))
        tel = int((shard["tel_defect"] > shard["tel_bound"]).sum())
        jdef = int(np.nansum(shard["J2_defect"] > shard["J2_bound"]))
        counts.append(dict(task=t, N=int(shard["N"]), updates=int(shard["updates"]),
                           bin_updates=shard["acc"][..., KEYS.index("updates")].min(axis=0).tolist(),
                           telescoping_violations=tel, J_violations=jdef,
                           max_ratio=shard["rmax"].max(axis=(0, 1)).tolist()))

    torch.cuda.reset_peak_memory_stats()
    rng_probe = []
    info = run(seeds, n_tasks, device, out, cifar, observer=obs, graph=graph, fresh=fresh, progress=progress,
               mutate=mutate, on_task_end=on_task_end, rng_probe=rng_probe, state_hook=state_hook)
    for fx in obs.saved:
        p = out / "fixtures" / f"t{fx['t']:02d}_j{fx['j']:03d}.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(fx, p)
        files.append(dict(path=str(p.relative_to(out)), sha256=sha(p), kind="fixture"))
    for name in ("per_task.csv", "fresh_control.csv"):
        if (out / name).exists():
            files.append(dict(path=name, sha256=sha(out / name), kind="host_rows"))
    cost = dict(seconds=info["wall_clock_s"], max_cuda_bytes=torch.cuda.max_memory_allocated(),
                max_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                bytes_written=sum((out / f["path"]).stat().st_size for f in files),
                observed_updates=int(sum(c["updates"] for c in counts) // max(obs.every, 1)),
                image_counts={str(c["task"]): c["N"] for c in counts}, step_ms_last_task=info["step_ms_last_task"])
    put(out / "cost.json", cost)
    status = dict(flags_total=dict(zip(FLAGS, flags_total.tolist())), counts=counts,
                  rng_unchanged=all(rng_probe), rng_probes=len(rng_probe), rolled_back=info["rolled_back"],
                  diverged=info["diverged"], batch_sha=info["batch_sha"], tc=info["tc"])
    put(out / "complete.json", dict(identity=ident, files=files, status=status))
    put(out / "provenance_end.json", dict(status="COMPLETE", utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                          tc=info["tc"], wall_clock_s=info["wall_clock_s"]))
    return dict(info=info, status=status, cost=cost, files=files)


def host_identity(out, env_host_csv=None) -> dict:
    """§7 A1: rows vs the committed 0920 record (eff_rank aside) and vs this environment's host rerun."""
    import csv
    out = Path(out)
    rec = list(csv.DictReader((RECORD / "per_task.csv").open()))
    new = list(csv.DictReader((out / "per_task.csv").open()))
    cols = list(rec[0].keys())
    same_cols = len(rec) == len(new) and list(new[0].keys()) == cols
    diffs = {}
    for a, b in zip(rec, new):
        for k in cols:
            if a[k] != b[k]:
                diffs.setdefault(k, []).append(abs(float(a[k]) - float(b[k])))
    non_eff = sorted(k for k in diffs if not k.startswith("eff_rank"))
    fresh_same = (out / "fresh_control.csv").read_bytes() == (RECORD / "fresh_control.csv").read_bytes()
    env_same = None if env_host_csv is None else (out / "per_task.csv").read_bytes() == Path(env_host_csv).read_bytes()
    ok = same_cols and not non_eff and fresh_same and (env_same is not False)
    return dict(pass_=ok, rows=len(new), columns_identical=same_cols, non_eff_rank_columns_differing=non_eff,
                eff_rank_differences={k: dict(rows=len(v), max_abs=max(v)) for k, v in diffs.items()},
                fresh_control_identical=fresh_same, identical_to_env_host=env_same)


def production(out, checks_path) -> dict:
    from analysis.drive_cifar5p1_1007.checks import validate_checks
    checks = json.loads(Path(checks_path).read_text())
    validate_checks(checks)
    dirty = git("status", "--porcelain", "src", "analysis", "specs")
    assert not dirty, f"commit the implementation before production:\n{dirty}"
    env_csv = RAW / "checks" / "host_env_0-9" / "per_task.csv"
    assert sha(env_csv) == checks["host_env"]["per_task_sha256"], "env host rerun changed since the checks"
    device = setup()
    cifar = load_cifar()
    with gpu_lock(checks["cost"]["need_gb"]):
        res = observed_run(out, "production", PRODUCTION, NT, GROUPS_PRODUCTION, cifar, device)
    hid = host_identity(out, env_csv)
    put(Path(out) / "host_identity.json", hid)
    st = res["status"]
    ok = (hid["pass_"] and all(v == 0 for v in st["flags_total"].values()) and st["rng_unchanged"]
          and not st["diverged"] and all(c["telescoping_violations"] == 0 and c["J_violations"] == 0
                                         for c in st["counts"]))
    put(Path(out) / "production_status.json", dict(status="COMPLETE" if ok else "CHECK_FAILED", host_identity=hid,
                                                   flags_total=st["flags_total"], rng_unchanged=st["rng_unchanged"]))
    print("production", "COMPLETE" if ok else "CHECK_FAILED", "(no scientific summary opened)", flush=True)
    return dict(status="COMPLETE" if ok else "CHECK_FAILED")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["production"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--checks", required=True)
    ap.add_argument("--production-go", action="store_true")
    a = ap.parse_args()
    assert a.production_go, "production needs --production-go"
    production(a.out, a.checks)


if __name__ == "__main__":
    main()

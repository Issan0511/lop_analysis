"""Checks for src/mucap_el_0916 (the mu-direction component caps of H2 design note section 10.4).

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 analysis/mucap_el_0916/checks.py
    ... --only S2            # development: writes results/_checks_mucap_el_0916/checks_partial.json

The run script does not exist yet; these are the projection-level checks the design calls for, in the
shape of analysis/wcap_rlmnist_0914/checks.py.  Each check runs on the real code and then on every
mutation listed with it; a mutation is an exact-once string substitution in src/mucap_el_0916.py and
it MUST make the same check fail.  Tolerances come from the derivations in the docstrings, never from
looking at the values.  Writes results/mucap_el_0916/checks.json after every step.

S1  the basis and the identity zbar = q ||mu|| + b       (the wrong axis is only caught here)
S2  the two caps: what is written, what is kept, tallies
S3  order independence, idempotence, and no NaN from zero rows / zero caps
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

import numpy as np
import torch

torch.set_num_threads(1)
from src import pmnist_0905 as H                 # noqa: E402
from src import pmnist_rlmnist_0906 as RL        # noqa: E402
from src import elu_growth_0909 as EG            # noqa: E402
from src import mucap_el_0916 as MU              # noqa: E402

CAPS = REPO / "src" / "mucap_el_0916.py"
RUNNER = REPO / "src" / "mucap_el_run_0916.py"
OUT = REPO / "results" / "mucap_el_0916"
SCR = REPO / "results" / "_checks_mucap_el_0916"
EPS32 = float(np.finfo(np.float32).eps)          # 1.19e-7
EPS64 = float(np.finfo(np.float64).eps)          # 2.22e-16
D_IN = 784                                       # first layer fan-in; the long dot products
DEV = H.setup("cpu")
MNIST = H.Mnist(DEV)
SEEDS = (0, 1, 2)
PROBE = 4                                        # processes run at once for S-cost
PY = sys.executable
THREAD_ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
TASKS_RUN, STEPS_RUN = 150, 6000                 # the design's box, for the projection
RESULTS: dict = {}
DUMP_PATH = OUT / "checks.json"

# --------------------------------------------------------------------------
# infrastructure (0914's, unchanged in behaviour)
# --------------------------------------------------------------------------

_n_loaded = [0]


def load(path: Path, subs=()):
    """Import `path` as a fresh module after exact-once substitutions (the real code when subs is empty)."""
    src = path.read_text()
    for old, new in subs:
        n = src.count(old)
        if n != 1:
            raise AssertionError(f"mutation anchor occurs {n} times in {path.name}: {old!r}")
        src = src.replace(old, new)
    _n_loaded[0] += 1
    name = f"_chk_{path.stem}_{_n_loaded[0]}"
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod


def _mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 ** 2
    return float("nan")


def brief(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "pass" and not isinstance(v, (dict, list))} | \
        {"failed_items": r.get("failed_items", [])[:8]}


def dump() -> None:
    checks = {k: v for k, v in RESULTS.items() if k.startswith("S") and isinstance(v, dict) and "pass" in v}
    RESULTS["all_pass"] = bool(checks) and all(
        v.get("pass") and v.get("all_mutations_detected", True) for v in checks.values())
    DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    DUMP_PATH.write_text(json.dumps(RESULTS, indent=2, default=str))


def run_check(key: str, title: str, fn, mutations: list, derivation: str, target: Path = None) -> None:
    target = target or CAPS
    t0 = time.time()
    base = fn(load(target))
    entry = {"title": title, "threshold_derivation": derivation, **base, "mutations": []}
    for label, subs in mutations:
        mod = load(target, subs)
        try:
            r = fn(mod)
            entry["mutations"].append({"mutation": label, "check_pass_on_mutant": bool(r["pass"]),
                                       "detected": not r["pass"], "mutant_result": brief(r)})
        except Exception as e:          # recorded, but a crash is not how a check should notice a defect
            entry["mutations"].append({"mutation": label, "check_pass_on_mutant": False, "detected": True,
                                       "raised": repr(e)[:400]})
    entry["all_mutations_detected"] = bool(entry["mutations"]) and all(m["detected"] for m in entry["mutations"])
    entry["seconds"] = round(time.time() - t0, 1)
    RESULTS[key] = entry
    dump()
    print(f"{key}: pass={entry['pass']}  mutations detected "
          f"{sum(m['detected'] for m in entry['mutations'])}/{len(entry['mutations'])}  ({entry['seconds']}s)"
          f"{'  FAILED ' + str(entry.get('failed_items', [])[:3]) if not entry['pass'] else ''}", flush=True)


# ---- anchors in src/mucap_el_0916.py (each must occur exactly once) ----
A_E = "    e64 = mu / nrm\n"
A_Q = "    q = (W @ e).unsqueeze(1)\n"
A_V = "    v = W - q * e\n"
A_PAR_CLAMP = "    q_new = torch.clamp(q, -q_cap, q_cap)\n"
A_PAR_WRITE = "    W.copy_(torch.where(over, W + (q_new - q) * e, W))\n"
A_PERP_WRITE = "    W.copy_(torch.where(over, q * e + v * (v_cap / n), W))\n"
A_BOTH_FIRST = "    a = cap_parallel_(W, e, q_cap)\n"


def _images(seed: int) -> torch.Tensor:
    """The seed's own fixed 1200 training images, /255, exactly as the RL box draws them."""
    return MNIST.train_x[RL.subset_idx(seed).to(MNIST.train_x.device)]


# --------------------------------------------------------------------------
# S1: the basis, the split, and the identity that ties q to the endpoint
# --------------------------------------------------------------------------

TOL_UNIT = 1e-6      # ||e||: d=784 squares summed in float64 then one sqrt: <= d*eps64 = 1.7e-13, huge margin
TOL_ORTH = 1e-4      # |v.e| / (||w|| ||e||): float32 dot over d terms, <= d*eps32 = 9.3e-5
TOL_RECON = 4.0      # |q e + v - w| in units of eps32*max|w|: three rounded ops per coordinate
TOL_ID64 = 1e-11     # float64 identity on e64: mean over N=1200 of a d=784 dot, <= (N+d)*eps64 = 4.4e-13
TOL_ID32 = 1e-4      # same identity with the cast-down e the caps use: ||mu|| |W.(e32-e64)| <= eps32
                     # relative to the same scale, plus d*eps32 from the float32 dot itself = 9.4e-5


def s1(M) -> dict:
    """For each seed: (i) ||e|| = 1; (ii) e is the unit vector along the mean image (not 1/sqrt(d));
    (iii) q e + v reconstructs W; (iv) v . e = 0; (v) the identity zbar_i = q_i ||mu|| + b_i against
    the mean preactivation computed directly from the images, in float64 (TOL_ID64) and with
    split_mu's own float32 q (TOL_ID32); (vi) mu = 0 returns (None, None, 0.0), not a crash or a zero
    vector; (vii) the float32 e the caps use is the cast of e64, not a separately rounded vector.
    Tolerances: TOL_UNIT, TOL_ORTH, TOL_RECON, TOL_ID64, TOL_ID32."""
    failed, per = [], {}
    for seed in SEEDS:
        x = _images(seed)
        e, e64, nrm = M.mu_basis(x)
        params = [t.detach() for t in H.init_params(seed, DEV)]
        W, b = params[0], params[1]
        q, v = M.split_mu(W, e)

        x64, W64, b64 = x.double(), W.double(), b.double()
        mu64 = x64.mean(0)
        unit = abs(float(torch.linalg.vector_norm(e64)) - 1.0)
        # is it really the mean-image direction?  cos against mu, and against the uniform axis
        cos_mu = float(e64 @ mu64 / torch.linalg.vector_norm(mu64))
        cos_uniform = float(e64.sum() / np.sqrt(e64.numel()))
        recon = float(((q * e + v - W).abs().amax(1) / (W.abs().amax(1) * EPS32)).max())
        orth = float((v.double() @ e64).abs().max() /
                     (torch.linalg.vector_norm(W64, dim=1).max() * torch.linalg.vector_norm(e64)))

        zbar_direct = (x64 @ W64.T).mean(0) + b64                  # the quantity the endpoint uses
        q64 = M.components(W, e64)["q"]
        id64 = float((torch.as_tensor(q64) * nrm + b64 - zbar_direct).abs().max() /
                     (torch.linalg.vector_norm(W64, dim=1).max() * nrm + b64.abs().max() + 1.0))
        id32 = float((q[:, 0].double() * nrm + b64 - zbar_direct).abs().max() /
                     (torch.linalg.vector_norm(W64, dim=1).max() * nrm + b64.abs().max() + 1.0))

        e0, e0_64, n0 = M.mu_basis(torch.zeros_like(x))
        ok = dict(i_unit=unit <= TOL_UNIT, ii_is_mu_axis=cos_mu > 1 - TOL_UNIT,
                  iii_recon=recon <= TOL_RECON, iv_orth=orth <= TOL_ORTH,
                  v_identity64=id64 <= TOL_ID64, v_identity32=id32 <= TOL_ID32,
                  vi_zero_mu=(e0 is None and e0_64 is None and n0 == 0.0),
                  vii_e32_matches_e64=bool((e == e64.to(e.dtype)).all()),
                  nonvacuous=nrm > 0 and abs(cos_uniform) < 0.999)
        per[f"seed{seed}"] = {**ok, "mu_norm": nrm, "cos_to_mu": cos_mu, "cos_to_uniform_axis": cos_uniform,
                              "unit_err": unit, "recon_in_eps32": recon, "orth_rel": orth,
                              "identity_rel_f64": id64, "identity_rel_f32": id32}
        failed += [f"seed{seed}|{k}" for k, v in ok.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S1_MUT = [
    ("M1a: e left unnormalised (e = mu)", [(A_E, "    e64 = mu\n")]),
    ("M1b: wrong axis -- the uniform direction 1/sqrt(d) instead of the mean image",
     [(A_E, "    e64 = torch.full_like(mu, 1.0 / mu.numel() ** 0.5)\n")]),
    ("M1c: v not fully orthogonalised (half the parallel part removed)",
     [(A_V, "    v = W - 0.5 * q * e\n")]),
    ("M1d: q from the raw mean image instead of the unit vector",
     [(A_Q, "    q = (W @ (e * 2.0)).unsqueeze(1)\n")]),
]


# --------------------------------------------------------------------------
# S2: what each cap writes and what it keeps
# --------------------------------------------------------------------------

TOL_Q = 1e-4         # |q'| / q_cap - 1: float32 dot over d <= 784 terms, <= d*eps32 = 9.3e-5
TOL_NORM = 1e-4      # ||v'|| / v_cap - 1: same derivation
TOL_COS = 1e-6       # rounding tilts a row by ~eps32*sqrt(d) = 3.3e-6 rad -> 1 - cos ~ 5.6e-12
TOL_KEEP = 8.0       # the component the cap must not move, in units of eps32*max|w|: the write is
                     # w + (q'-q)e (two rounded ops) and the norms behind it two more
TOL_REMOVED = 1e-3   # the over rows are >= 20% past their cap by construction: <= 5*9.3e-5 + eps


def _cases(M):
    """(name, W float32, e float32, q_cap, v_cap).  Random rows: |q| ~ U(0.5, 4), ||v|| ~ U(2, 10),
    caps at 0.5-0.8 or 1.25-1.5 of the present value, so no row sits within 20% of its cap; row 0 gets
    v = 0 and row 1 gets q_cap = 0 (the two degenerate rows the design asks about); rows 2 and 3 get a
    q of the opposite sign to their cap's origin, so a sign change inside the cap is exercised.
    Real rows: W1 and W2 of the host's init for seed 0, e from that seed's images (W2 gets e = e_2, the
    unit vector along the mean hidden activity of a randomly initialised first layer)."""
    g = torch.Generator().manual_seed(916)
    cases = []
    for cols in (784, 100):
        e = torch.randn(cols, generator=g, dtype=torch.float64)
        e = (e / torch.linalg.vector_norm(e)).float()
        v = torch.randn(100, cols, generator=g, dtype=torch.float64)
        v -= (v @ e.double()).unsqueeze(1) * e.double()
        v *= (2 + 8 * torch.rand(100, 1, generator=g, dtype=torch.float64)) / torch.linalg.vector_norm(v, dim=1, keepdim=True)
        q = (0.5 + 3.5 * torch.rand(100, 1, generator=g, dtype=torch.float64))
        q *= torch.where(torch.rand(100, 1, generator=g, dtype=torch.float64) < 0.5, -1.0, 1.0)
        W = (q * e.double() + v).float()
        f = lambda: torch.where(torch.rand(100, 1, generator=g) < 0.5, 0.5 + 0.3 * torch.rand(100, 1, generator=g),
                                1.25 + 0.25 * torch.rand(100, 1, generator=g))
        qc, vc = (M.parallel_cap(W, e) * f()).float(), (M.perp_cap(W, e) * f()).float()
        W[0] = (q[0] * e.double()).float()            # zero orthogonal part
        qc[1] = 0.0                                   # zero parallel cap
        cases.append((f"random_{cols}", W, e, qc, vc))
    p = [t.detach() for t in H.init_params(0, DEV)]
    x = _images(0)
    e1, _, _ = M.mu_basis(x)
    a1 = H.forward(p, x, H.ARMS["LR"])[1]
    e2, _, _ = M.mu_basis(a1)
    for W, e, name in ((p[0].clone(), e1, "init_W1_mu1"), (p[2].clone(), e2, "init_W2_mu2")):
        alt = torch.tensor([0.8 if i % 2 == 0 else 1.25 for i in range(W.shape[0])]).view(-1, 1)
        cases.append((name, W, e, (M.parallel_cap(W, e) * alt).float(), (M.perp_cap(W, e) * alt).float()))
    return cases


def s2(M) -> dict:
    """On every case, cap_parallel_ and cap_perp_ separately.  Parallel: (i) rows with |q| <= q_cap keep
    every bit; (ii) written rows get |q'| = q_cap (TOL_Q) with the sign q had; (iii) their v is kept
    (TOL_KEEP, in eps32 units); (iv) the row count and the sum of |q| - q_cap match a float64 recount.
    Perpendicular: (v) rows with ||v|| <= v_cap keep every bit; (vi) written rows get ||v'|| = v_cap
    (TOL_NORM) with the direction of v (TOL_COS); (vii) their q is kept (TOL_KEEP); (viii) tallies match.
    A cap of 0 is checked separately: the row must land on q = 0 to within the write's own rounding
    (TOL_KEEP, in units of eps32*max|w|*sqrt(d)), and it has no sign to keep.
    (ix) neither cap produces a non-finite weight, including the zero-v row and the zero-cap row.
    All norms, cosines and counts recomputed in float64.  Tolerances: TOL_Q, TOL_NORM, TOL_COS,
    TOL_KEEP, TOL_REMOVED."""
    failed, per = [], {}
    for name, W0, e, qc, vc in _cases(M):
        e64 = e.double()
        q0 = (W0.double() @ e64).unsqueeze(1)
        v0 = W0.double() - q0 * e64
        n0 = torch.linalg.vector_norm(v0, dim=1, keepdim=True)
        scale = (W0.double().abs().amax(1) * EPS32).clamp_min(1e-300)

        Wp = W0.clone()
        cnt_p, rem_p = M.cap_parallel_(Wp, e, qc)
        q1 = (Wp.double() @ e64).unsqueeze(1)
        v1 = Wp.double() - q1 * e64
        over_p = (q0.abs() > qc.double())[:, 0]
        p_under = bool((Wp[~over_p] == W0[~over_p]).all())
        pos = over_p & (qc[:, 0] > 0)                      # a ratio is only defined against a positive cap
        zer = over_p & (qc[:, 0] == 0)                     # cap 0: the row must land on q = 0
        p_mag = float(((q1.abs() / qc.double().clamp_min(1e-300) - 1).abs()[pos]).max()) if pos.any() else 0.0
        p_zero_cap = float((q1.abs()[:, 0] / (scale * np.sqrt(W0.shape[1])))[zer].max()) if zer.any() else 0.0
        p_sign = bool((torch.sign(q1[pos]) == torch.sign(q0[pos])).all()) if pos.any() else True
        p_keep = float(((v1 - v0).abs().amax(1) / scale)[over_p].max()) if over_p.any() else 0.0
        p_cnt64 = int(over_p.sum())
        p_rem64 = float((q0.abs() - qc.double())[over_p[:, None]].sum())
        p_rem_rel = abs(float(rem_p) - p_rem64) / max(abs(p_rem64), 1e-300)

        Wv = W0.clone()
        cnt_v, rem_v = M.cap_perp_(Wv, e, vc)
        q2 = (Wv.double() @ e64).unsqueeze(1)
        v2 = Wv.double() - q2 * e64
        n2 = torch.linalg.vector_norm(v2, dim=1, keepdim=True)
        over_v = (n0 > vc.double())[:, 0]
        v_under = bool((Wv[~over_v] == W0[~over_v]).all())
        v_mag = float(((n2 / vc.double().clamp_min(1e-300) - 1).abs()[over_v]).max()) if over_v.any() else 0.0
        cos = (v2 * v0).sum(1) / (n0[:, 0] * n2[:, 0]).clamp_min(1e-300)
        v_cos = float((1 - cos[over_v]).abs().max()) if over_v.any() else 0.0
        v_keep = float(((q2 - q0).abs()[:, 0] / scale)[over_v].max()) if over_v.any() else 0.0
        v_cnt64 = int(over_v.sum())
        v_rem64 = float((n0 - vc.double())[over_v[:, None]].sum())
        v_rem_rel = abs(float(rem_v) - v_rem64) / max(abs(v_rem64), 1e-300)

        ok = dict(i_par_under_bit_identical=p_under, ii_par_magnitude=p_mag <= TOL_Q,
                  ii_par_zero_cap=p_zero_cap <= TOL_KEEP,
                  ii_par_sign=p_sign, iii_par_keeps_v=p_keep <= TOL_KEEP,
                  iv_par_count=int(cnt_p) == p_cnt64, iv_par_removed=p_rem_rel <= TOL_REMOVED,
                  v_perp_under_bit_identical=v_under, vi_perp_norm=v_mag <= TOL_NORM,
                  vi_perp_direction=v_cos <= TOL_COS, vii_perp_keeps_q=v_keep <= TOL_KEEP,
                  viii_perp_count=int(cnt_v) == v_cnt64, viii_perp_removed=v_rem_rel <= TOL_REMOVED,
                  ix_finite=bool(torch.isfinite(Wp).all() and torch.isfinite(Wv).all()),
                  nonvacuous=0 < p_cnt64 < W0.shape[0] and 0 < v_cnt64 < W0.shape[0])
        per[name] = {**ok, "rows_over_par": p_cnt64, "rows_over_perp": v_cnt64, "max_rel_q": p_mag,
                     "zero_cap_residual_q_in_eps32": p_zero_cap, "v_kept_in_eps32": p_keep, "max_rel_vnorm": v_mag,
                     "max_1mcos": v_cos, "q_kept_in_eps32": v_keep,
                     "removed_rel_err_par": p_rem_rel, "removed_rel_err_perp": v_rem_rel}
        failed += [f"{name}|{k}" for k, v in ok.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S2_MUT = [
    ("M2a: the parallel cap scales the whole row instead of its parallel part",
     [(A_PAR_WRITE, "    W.copy_(torch.where(over, W * (q_new / torch.where(q == 0, torch.ones_like(q), q)), W))\n")]),
    ("M2b: |q| set to the cap on every row (rows under the cap are stretched too)",
     [(A_PAR_CLAMP, "    q_new = torch.sign(q) * q_cap\n"),
      (A_PAR_WRITE, "    W.copy_(W + (q_new - q) * e)\n")]),
    ("M2c: the parallel cap clips to [0, q_cap], dropping the sign",
     [(A_PAR_CLAMP, "    q_new = torch.clamp(q, torch.zeros_like(q_cap), q_cap)\n")]),
    ("M2d: the perpendicular cap rescales q along with v",
     [(A_PERP_WRITE, "    W.copy_(torch.where(over, (q * e + v) * (v_cap / n), W))\n")]),
    ("M2e: ||v|| set to the cap on every row (zero-norm rows included)",
     [(A_PERP_WRITE, "    W.copy_(q * e + v * (v_cap / n))\n")]),
    ("M2f: the perpendicular cap adds a 1% tangential component",
     [(A_PERP_WRITE, "    W.copy_(torch.where(over, q * e + (v + 0.01 * torch.roll(v, 1, dims=1)) * (v_cap / n), W))\n")]),
    ("M2g: scale factor v_cap / n^2",
     [(A_PERP_WRITE, "    W.copy_(torch.where(over, q * e + v * (v_cap / (n * n)), W))\n")]),
]


# --------------------------------------------------------------------------
# S3: the two caps commute, each is idempotent, and nothing else is touched
# --------------------------------------------------------------------------

TOL_ORDER = 16.0     # |W_ab - W_ba| in units of eps32*max|w|: each order is <= 6 rounded ops per coordinate


def s3(M) -> dict:
    """(i) cap_both_ (parallel then perpendicular) agrees with the other order within TOL_ORDER;
    (ii) a second application moves no row beyond the write's own rounding (TOL_ORDER).  A float32
    projection is not bit-idempotent -- |q'| lands a rounding step above its cap for about half the
    written rows, so the second pass writes them again; what must not happen is a further real move,
    which is how a wrong scale factor shows up even when the first application looks right.
    (iii) the caps touch only
    the tensor handed to them: the other parameters and a stand-in Adam moment
    pair are bit-identical.  (iv) after cap_both_ BOTH constraints hold on every row -- |q| <= q_cap and
    ||v|| <= v_cap to within TOL_Q / TOL_NORM -- which is what a cap_both_ that silently drops one of its
    two steps fails.  Tolerances: TOL_ORDER, TOL_Q, TOL_NORM."""
    failed, per = [], {}
    for name, W0, e, qc, vc in _cases(M):
        a = W0.clone()
        M.cap_both_(a, e, qc, vc)
        b = W0.clone()
        M.cap_perp_(b, e, vc)
        M.cap_parallel_(b, e, qc)
        scale = (W0.double().abs().amax(1) * EPS32).clamp_min(1e-300)
        order = float(((a.double() - b.double()).abs().amax(1) / scale).max())

        c = a.clone()
        n_par, _ = M.cap_parallel_(c, e, qc)
        n_perp, _ = M.cap_perp_(c, e, vc)
        idem = float(((c.double() - a.double()).abs().amax(1) / scale).max())

        others = [torch.randn(100, 10, generator=torch.Generator().manual_seed(3)),
                  torch.randn(100, generator=torch.Generator().manual_seed(4))]
        moments = [torch.rand_like(W0), torch.rand_like(W0)]
        keep = [t.clone() for t in others + moments]
        d = W0.clone()
        M.cap_both_(d, e, qc, vc)
        untouched = all(bool((x == y).all()) for x, y in zip(others + moments, keep))

        qa = (a.double() @ e.double()).unsqueeze(1)
        na = torch.linalg.vector_norm(a.double() - qa * e.double(), dim=1, keepdim=True)
        both_q = float((qa.abs() / qc.double().clamp_min(1e-300))[qc > 0].max())
        both_v = float((na / vc.double().clamp_min(1e-300))[vc > 0].max())
        ok = dict(i_orders_agree=order <= TOL_ORDER, ii_idempotent=idem <= TOL_ORDER,
                  iii_others_untouched=untouched,
                  iv_both_caps_hold=both_q <= 1 + TOL_Q and both_v <= 1 + TOL_NORM,
                  nonvacuous=not bool((a == W0).all()))
        per[name] = {**ok, "order_gap_in_eps32": order, "second_pass_move_in_eps32": idem,
                     "max_q_over_cap": both_q, "max_vnorm_over_cap": both_v,
                     "rows_written_on_second_pass": [int(n_par), int(n_perp)]}
        failed += [f"{name}|{k}" for k, v in ok.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S3_MUT = [
    ("M3a: perpendicular scale factor v_cap / n^2 (first pass plausible, not idempotent)",
     [(A_PERP_WRITE, "    W.copy_(torch.where(over, q * e + v * (v_cap / (n * n)), W))\n")]),
    ("M3b: the parallel cap also shrinks v by 1% (the orders stop agreeing)",
     [(A_PAR_WRITE, "    W.copy_(torch.where(over, (W + (q_new - q) * e) - 0.01 * (W - q * e), W))\n")]),
    ("M3c: cap_both_ runs the parallel step twice and never caps v",
     [(A_BOTH_FIRST, "    a = cap_parallel_(W, e, q_cap); cap_parallel_(W, e, q_cap)\n"),
      (A_PERP_WRITE, "    W.copy_(W)\n")]),
]



# --------------------------------------------------------------------------
# S4-S7: the runner (src/mucap_el_run_0916.py)
# --------------------------------------------------------------------------

R_ACTS = '    act1, act2 = EG.ELU(1.0), H.ARMS["LR"]\n'
R_FWD_A1 = "    a1 = act1.phi(z1)\n"
R_CAPON = "        cap_on = t >= 2 and (do_par or do_perp)\n"
R_CAPSET = "            q_cap = MU.parallel_cap(params[CAP_LAYER], e1)\n"
R_CAPLAYER = "CAP_LAYER = 0 "
R_QSQ = '    acc[f"{part}_q_sq"] += dq * dq\n'
R_PROJBOOK = '        _add_step(acc, "proj", pm, _point(W_after, e64), d_in)\n'
R_V2ALIGN = '    acc[f"{part}_v2_align"] += 2 * (dot - q0 * dq)\n'
R_WT2SQ = '    acc[f"{part}_wt2_sq"] += sq - d_in * dm * dm\n'
R_ORDER = "            order = torch.randperm(N_IMAGES, generator=g_batch).to(device)\n"

EP = 2               # 150 updates per task in the runner checks; the properties are per update
A32 = float(torch.tensor(H.ARMS["LR"].param, dtype=torch.float32))   # the 0.1 the host actually stores
TOL_LEAKY = 1e-12    # gate_mean = A32 + (1-A32) p+ for leaky: a float64 mean of the two float32
                     # constants over N=1200 images, <= N*eps64 = 2.7e-13.  The decimal 0.1 is NOT
                     # that constant -- float32(0.1) - 0.1 = 1.5e-9 and the identity would miss by it
TOL_ELU = 1e-3       # non-vacuity: the first layer must MISS that identity by at least this much
TOL_CLOSE = 8.0      # S6 residual in units of n_steps*eps64*traffic: the accumulation of n signed
                     # terms rounds by at most n*eps64 times the traffic it accumulated


def _run(M, arm, seed=0, tasks=1, ledger=False, debug=None):
    return M.run_one(arm, seed, 1e-3, tasks, MNIST, DEV, epochs=EP, ledger=ledger, debug=debug)


def s4(M) -> dict:
    """(i) forward2 with one activation twice is bit-identical to the host's forward, for leaky and
    for ELU1 -- the wiring is the host's, not a second implementation.  (ii) In a real run's own
    diagnostics the second layer's gate satisfies the leaky identity  mean phi' = 0.1 + 0.9 p+  to
    TOL_LEAKY (phi' takes only the two float32 values A32 and 1), and the first layer misses it by more than
    TOL_ELU (phi' = e^z there).  A swapped or duplicated activation moves one of the two.  (iii) inside
    forward2 each hidden layer's output is its own activation applied to its own preactivation, which is
    what a forward2 that reuses act1 downstream breaks while the per-layer diagnostics still look right.
    Tolerances: TOL_LEAKY, TOL_ELU."""
    failed, per = [], {}
    p = [t.detach() for t in H.init_params(0, DEV)]
    x = _images(0)
    for nm, act in (("LR", H.ARMS["LR"]), ("ELU1", EG.ELU(1.0))):
        same = all(bool((a == b).all()) for a, b in zip(M.forward2(p, x, act, act), H.forward(p, x, act)))
        per[f"forward2_is_host_{nm}"] = same
        if not same:
            failed.append(f"forward2_is_host_{nm}")
    elu, lr = EG.ELU(1.0), H.ARMS["LR"]
    z1, a1, z2, a2, lg = M.forward2(p, x, elu, lr)
    per["iii_a1_is_act1"] = bool((a1 == elu.phi(z1)).all())
    per["iii_a2_is_act2"] = bool((a2 == lr.phi(z2)).all())
    per["iii_acts_differ_here"] = bool((elu.phi(z2) != lr.phi(z2)).any())
    failed += [k for k in ("iii_a1_is_act1", "iii_a2_is_act2", "iii_acts_differ_here") if not per[k]]
    rows, arrays, _, _ = _run(M, "ref", tasks=1)
    g2, pp2 = arrays["gate_mean_l2"], arrays["pplus_l2"]
    g1, pp1 = arrays["gate_mean_l1"], arrays["pplus_l1"]
    leaky_gap = float(np.abs(g2 - (A32 + (1 - A32) * pp2)).max())
    elu_gap = float(np.abs(g1 - (A32 + (1 - A32) * pp1)).max())
    ok = dict(ii_layer2_is_leaky=leaky_gap <= TOL_LEAKY, ii_layer1_is_not_leaky=elu_gap > TOL_ELU,
              nonvacuous=bool((pp1 > 0).any() and (pp1 < 1).any()))
    per |= {**ok, "leaky_identity_gap_l2": leaky_gap, "leaky_identity_gap_l1": elu_gap}
    failed += [k for k, v in ok.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S4_MUT = [
    ("M4a: the two activations swapped", [(R_ACTS, '    act1, act2 = H.ARMS["LR"], EG.ELU(1.0)\n')]),
    ("M4b: ELU on both hidden layers", [(R_ACTS, '    act1, act2 = EG.ELU(1.0), EG.ELU(1.0)\n')]),
    ("M4c: forward2 applies the first activation to both layers",
     [(R_FWD_A1, "    a1 = act1.phi(z1)\n"), ("    a2 = act2.phi(z2)\n", "    a2 = act1.phi(z2)\n")]),
]


def s5(M) -> dict:
    """(i) the four arms' task 1 is the same run: identical init, subset, labels, batch orders and
    task-1-end state sha256; (ii) they have stopped being the same run by the end of task 2, and ref
    still differs from each cap arm (a cap that never binds would make this vacuous); (iii) the radii
    each cap arm stores are that arm's own task-1-end row, recomputed here with the module's own
    parallel_cap / perp_cap from the captured task-1-end weights -- not the init row, and on the
    layer the design names.  Bit comparisons only; no tolerance."""
    failed, per = [], {}
    end1, states, caps = {}, {}, {}
    for arm in M.ARMS:
        d = {}
        rows, arrays, _, info = _run(M, arm, tasks=2, debug=d)
        end1[arm] = (info["init_sha256"], info["subset_idx_sha256"], info["labels_sha256"],
                     info["batch_sha256"], info["task1_end_state_sha256"])
        states[arm] = info["final_state_sha256"]
        W1 = d["task1_end_params"][M.CAP_LAYER]
        caps[arm] = (MU.parallel_cap(W1, d["e1"]), MU.perp_cap(W1, d["e1"]), arrays["q_cap"], arrays["v_cap"])
    ref = end1["ref"]
    same_t1 = {a: end1[a] == ref for a in M.ARMS}
    diff_t2 = {a: states[a] != states["ref"] for a in M.ARMS if a != "ref"}
    radii = {}
    for a in M.ARMS:
        if a == "ref":
            continue
        want_q, want_v, got_q, got_v = caps[a]
        radii[a] = (bool((want_q[:, 0].numpy() == got_q).all()) and
                    bool((want_v[:, 0].numpy() == got_v).all()))
    ok = dict(i_task1_identical=all(same_t1.values()), ii_arms_diverge_by_task2=all(diff_t2.values()),
              iii_radii_are_task1_end=all(radii.values()), nonvacuous=len(set(states.values())) > 1)
    per = {**ok, "task1_identical": same_t1, "differs_from_ref_after_task2": diff_t2,
           "radii_match_task1_end": radii}
    failed += [k for k, v in ok.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S5_MUT = [
    ("M5a: the cap is on from task 1 (task 1 stops being shared)",
     [(R_CAPON, "        cap_on = t >= 1 and (do_par or do_perp)\n")]),
    ("M5b: the radii are taken at init instead of at the end of task 1",
     [(R_CAPSET, "            q_cap = MU.parallel_cap(H.init_params(seed, device)[CAP_LAYER], e1)\n")]),
    ("M5c: the cap is applied to the second layer instead of the first",
     [(R_CAPLAYER, "CAP_LAYER = 2 ")]),
]


def s6(M) -> dict:
    """For every task of a capped run and every unit: the four accumulated terms of each component
    (Adam align + Adam square + projection align + projection square) reproduce the change in that
    component between the task's first and last diagnostic point, and the signed displacements
    reproduce the change in q and in m.  The residual is read against n_steps * eps64 * traffic,
    where traffic is the sum of |increment| the books actually accumulated: under a cap the net
    change can be ~0 while the traffic is not, and a relative error against the net change would
    then say nothing.  Tolerance: TOL_CLOSE."""
    failed, per = [], {}
    n = M.STEPS_PER_EPOCH * EP
    for arm in ("cap_both", "ref"):
        rows, arrays, led, _ = _run(M, arm, tasks=3, ledger=True)
        t_d, s_d = arrays["task"], arrays["step"]
        worst = {}
        for t in (1, 2, 3):
            i0 = int(np.where((t_d == t) & (s_d == 0))[0][0])
            i1 = int(np.where((t_d == t) & (s_d == n))[0][0])
            j = int(np.where(led["task"] == t)[0][0])
            base = {"q": arrays["q_l1"], "v2": arrays["v_norm_l1"], "m": arrays["row_mean_l1"],
                    "wt2": arrays["wt_norm_l1"]}
            for comp in M.LEDGER_COMPS:
                a0, a1 = base[comp][i0], base[comp][i1]
                direct = a1 ** 2 - a0 ** 2 if comp in ("q", "m") else a1 ** 2 - a0 ** 2
                terms = sum(led[f"{p}_{comp}_{x}"][j] for p in ("adam", "proj") for x in ("align", "sq"))
                unit = np.abs(direct - terms) / np.maximum(n * EPS64 * led[f"traffic_{comp}"][j], 1e-300)
                worst[f"{t}_{comp}"] = float(unit.max())
            for comp, arr in (("q", arrays["q_l1"]), ("m", arrays["row_mean_l1"])):
                d_direct = arr[i1] - arr[i0]
                d_terms = sum(led[f"{p}_{comp}_d"][j] for p in ("adam", "proj"))
                sc = np.maximum(n * EPS64 * led[f"traffic_{comp}"][j] / np.maximum(np.abs(arr[i0]), 1e-12),
                                1e-300)
                worst[f"{t}_{comp}_signed"] = float((np.abs(d_direct - d_terms) / sc).max())
        got = max(worst.values())
        ok = dict(i_books_close=got <= TOL_CLOSE,
                  nonvacuous=bool(led["traffic_q"][1:].min() > 0 and
                                  (arm == "ref" or led["proj_q_align"][1:].any())))
        per[arm] = {**ok, "worst_residual_in_units": got, "worst_item": max(worst, key=worst.get)}
        failed += [f"{arm}|{k}" for k, v in ok.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S6_MUT = [
    ("M6a: the second-order term of q dropped", [(R_QSQ, '    acc[f"{part}_q_sq"] += 0.0 * dq\n')]),
    ("M6b: the projection's book never written", [(R_PROJBOOK, "        pass\n")]),
    ("M6c: v's alignment term without the parallel correction",
     [(R_V2ALIGN, '    acc[f"{part}_v2_align"] += 2 * dot\n')]),
    ("M6d: W~'s second-order term without the row-mean correction",
     [(R_WT2SQ, '    acc[f"{part}_wt2_sq"] += sq\n')]),
]


def s7(M) -> dict:
    """The same call twice, in the same process, gives the same bits: every diagnostic array, every
    ledger array and the state hashes.  An update order that does not come from the run's own
    generator is what this catches.  Bit comparisons only; no tolerance."""
    failed, per = [], {}
    for arm in ("ref", "cap_both"):
        r1, a1, l1, i1 = _run(M, arm, tasks=2, ledger=True)
        r2, a2, l2, i2 = _run(M, arm, tasks=2, ledger=True)
        arrays_same = set(a1) == set(a2) and all(np.array_equal(a1[k], a2[k], equal_nan=True) for k in a1)
        led_same = set(l1) == set(l2) and all(np.array_equal(l1[k], l2[k], equal_nan=True) for k in l1)
        hash_same = all(i1[k] == i2[k] for k in ("init_sha256", "labels_sha256", "batch_sha256",
                                                 "task1_end_state_sha256", "final_state_sha256"))
        rows_same = r1 == r2
        ok = dict(i_arrays=arrays_same, ii_ledger=led_same, iii_hashes=hash_same, iv_rows=rows_same,
                  nonvacuous=len(a1) > 10 and len(l1) > 10)
        per[arm] = ok
        failed += [f"{arm}|{k}" for k, v in ok.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S7_MUT = [
    ("M7a: the batch order does not come from the run's own generator",
     [(R_ORDER, "            order = torch.randperm(N_IMAGES).to(device)\n")]),
]


def s_cost() -> None:
    """Cost under the load the grid would actually run at: PROBE processes of the real box (2 tasks,
    80 epochs, the ledger on) at once, each on one thread.  Gate: every process exits 0 and the
    memory budget leaves at least 4 slots, where a slot is one process's measured peak RSS against
    MemAvailable with a 20% margin -- the number of cores is not the budget.  The projection to the
    design's 150 tasks assumes the per-update cost measured here, which already carries the dense
    diagnostics of the early tasks."""
    t0 = time.time()
    avail0 = _mem_available_gib()
    SCR.mkdir(parents=True, exist_ok=True)
    procs = []
    for i in range(PROBE):
        d = SCR / f"cost_{i}"
        procs.append((d, subprocess.Popen(
            [PY, str(REPO / "src" / "mucap_el_run_0916.py"), "--arm", "cap_both", "--seeds", str(i),
             "--tasks", "2", "--out", str(d)], env=THREAD_ENV, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE)))
    rc, peak, secs = [], [], []
    for d, pr in procs:
        err = pr.communicate()[1]
        rc.append(pr.returncode)
        if pr.returncode == 0:
            pv = json.loads((d / "provenance.json").read_text())
            peak.append(pv["peak_rss_kb"] / 1024 ** 2)
            secs.append(pv["wall_clock_s"])
        else:
            print(err.decode()[-400:], flush=True)
    steps = 2 * STEPS_RUN
    per_update = (max(secs) / steps) if secs else float("nan")
    slots = int(avail0 * 0.8 / max(peak)) if peak else 0
    one = TASKS_RUN * STEPS_RUN * per_update
    ok = all(r == 0 for r in rc) and slots >= 4
    RESULTS["S-cost"] = {"pass": bool(ok), "gate": f"all {PROBE} processes exit 0 and slots >= 4",
                         "threshold_derivation": "slot = peak RSS of one process; budget = 0.8 * "
                                                 "MemAvailable at the start; 4 slots is the least that "
                                                 "makes the 40-run grid fit in a day on this machine",
                         "returncodes": rc, "mem_available_gib_before": avail0,
                         "peak_rss_gib_max": max(peak) if peak else None, "slots": slots,
                         "seconds_per_update_under_load": per_update,
                         "one_trajectory_min": one / 60,
                         "grid_40_serial_hours": 40 * one / 3600,
                         "grid_40_wall_hours_at_slots": 40 * one / 3600 / max(slots, 1),
                         "note": f"per-update cost measured with {PROBE} processes at once; at more "
                                 f"slots each process is slower, so grid_40_wall_hours_at_slots is a "
                                 f"lower bound, not a measurement",
                         "measured_at": dt.datetime.now().astimezone().isoformat(),
                         "seconds": round(time.time() - t0, 1)}
    dump()
    c = RESULTS["S-cost"]
    print(f"S-cost: pass={ok}  {per_update * 1e3:.2f} ms/update under {PROBE} at once -> "
          f"{c['one_trajectory_min']:.1f} min per trajectory, {c['grid_40_serial_hours']:.1f} h serial, "
          f"{c['grid_40_wall_hours_at_slots']:.1f} h at {slots} slots (peak RSS "
          f"{c['peak_rss_gib_max']:.2f} GiB, MemAvailable {avail0:.1f} GiB)", flush=True)


# --------------------------------------------------------------------------

CHECKS = {
    "S1_S-mu-basis": ("S1 e is the mean-image axis and zbar = q||mu|| + b holds on the fixed set", s1, S1_MUT,
                      s1.__doc__, CAPS),
    "S2_S-mu-projection": ("S2 each cap writes its own component only, keeps the other, tallies match", s2,
                           S2_MUT, s2.__doc__, CAPS),
    "S3_S-mu-order": ("S3 the caps commute, are idempotent, and touch nothing else", s3, S3_MUT, s3.__doc__,
                      CAPS),
    "S4_S-wiring": ("S4 ELU on the first hidden layer, leaky on the second, and forward2 is the host's",
                    s4, S4_MUT, s4.__doc__, RUNNER),
    "S5_S-identical": ("S5 all four arms share task 1 bit for bit; the radii are that state's", s5, S5_MUT,
                       s5.__doc__, RUNNER),
    "S6_S-ledger": ("S6 the per-update books close against the task's end points", s6, S6_MUT, s6.__doc__,
                    RUNNER),
    "S7_S-repro": ("S7 the same call twice gives the same bits", s7, S7_MUT, s7.__doc__, RUNNER),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma separated check keys or prefixes, e.g. S2")
    args = ap.parse_args()
    global DUMP_PATH
    keys = [k for k in CHECKS if not args.only or any(k.startswith(p) for p in args.only.split(","))]
    if args.only:
        SCR.mkdir(parents=True, exist_ok=True)
        DUMP_PATH = SCR / "checks_partial.json"
    RESULTS.update({"run_id": "mucap_el_0916", "started_at": dt.datetime.now().astimezone().isoformat(),
                    "spec": "obsidian-research 可塑性喪失/spec/H2_W成長抑制と負側輸送_設計案_0916.md section 10.4",
                    "code_sha256": {"src/mucap_el_0916.py": hashlib.sha256(CAPS.read_bytes()).hexdigest(),
                                    "src/mucap_el_run_0916.py": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
                                    "analysis/mucap_el_0916/checks.py":
                                        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                    "torch": torch.__version__, "threads": torch.get_num_threads(),
                    "eps32": EPS32, "eps64": EPS64, "seeds": list(SEEDS)})
    for k in keys:
        title, fn, mut, deriv, target = CHECKS[k]
        run_check(k, title, fn, mut, deriv, target)
    if not args.only or "cost" in args.only:
        s_cost()
    RESULTS["finished_at"] = dt.datetime.now().astimezone().isoformat()
    dump()
    print(f"all_pass = {RESULTS['all_pass']}  -> {DUMP_PATH}")


if __name__ == "__main__":
    main()

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

RUNNER = REPO / "src" / "mucap_el_0916.py"
OUT = REPO / "results" / "mucap_el_0916"
SCR = REPO / "results" / "_checks_mucap_el_0916"
EPS32 = float(np.finfo(np.float32).eps)          # 1.19e-7
EPS64 = float(np.finfo(np.float64).eps)          # 2.22e-16
D_IN = 784                                       # first layer fan-in; the long dot products
DEV = H.setup("cpu")
MNIST = H.Mnist(DEV)
SEEDS = (0, 1, 2)
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


def brief(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "pass" and not isinstance(v, (dict, list))} | \
        {"failed_items": r.get("failed_items", [])[:8]}


def dump() -> None:
    checks = {k: v for k, v in RESULTS.items() if k.startswith("S")}
    RESULTS["all_pass"] = bool(checks) and all(
        v.get("pass") and v.get("all_mutations_detected", True) for v in checks.values())
    DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    DUMP_PATH.write_text(json.dumps(RESULTS, indent=2, default=str))


def run_check(key: str, title: str, fn, mutations: list, derivation: str) -> None:
    t0 = time.time()
    base = fn(load(RUNNER))
    entry = {"title": title, "threshold_derivation": derivation, **base, "mutations": []}
    for label, subs in mutations:
        mod = load(RUNNER, subs)
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

CHECKS = {
    "S1_S-mu-basis": ("S1 e is the mean-image axis and zbar = q||mu|| + b holds on the fixed set", s1, S1_MUT,
                      s1.__doc__),
    "S2_S-mu-projection": ("S2 each cap writes its own component only, keeps the other, tallies match", s2,
                           S2_MUT, s2.__doc__),
    "S3_S-mu-order": ("S3 the caps commute, are idempotent, and touch nothing else", s3, S3_MUT, s3.__doc__),
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
                    "code_sha256": {"src/mucap_el_0916.py": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
                                    "analysis/mucap_el_0916/checks.py":
                                        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                    "torch": torch.__version__, "threads": torch.get_num_threads(),
                    "eps32": EPS32, "eps64": EPS64, "seeds": list(SEEDS)})
    for k in keys:
        title, fn, mut, deriv = CHECKS[k]
        run_check(k, title, fn, mut, deriv)
    RESULTS["finished_at"] = dt.datetime.now().astimezone().isoformat()
    dump()
    print(f"all_pass = {RESULTS['all_pass']}  -> {DUMP_PATH}")


if __name__ == "__main__":
    main()

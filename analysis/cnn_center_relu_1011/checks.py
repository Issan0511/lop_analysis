#!/usr/bin/env python3
"""Checks of cnn_center_relu_1011 (spec §6), each with mutation controls.

    python analysis/cnn_center_relu_1011/checks.py [--only K-same,K-host] [--skip-cost]

-> results/cnn_center_relu_1011/checks.json (scratch in results/_checks_cnn_center_relu_1011/).
Every check must pass on the real code and fail on each listed mutant; a mutant is the target
file with one exact-once string substitution (doors_rlmnist_1007's `load`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
import torch                                          # noqa: E402
import torch.nn.functional as F                       # noqa: E402

from src import pmnist_0905 as H                      # noqa: E402
from src import pmnist_rlcifar_0907 as RC             # noqa: E402
from src import rlcifar_cnn_0908 as CN                # noqa: E402
from src import sna_cnn_cause_1009 as E               # noqa: E402
from src import cnn_drive_verify_1009_lr as LRM       # noqa: E402
from src import cnn_center_1010 as CC                 # noqa: E402  (BundleLRc, mean_patch; install() not called)

ENGINE = REPO / "src" / "cnn_center_relu_1011.py"
VERDICT = REPO / "analysis" / "cnn_center_relu_1011" / "verdict.py"
COST = REPO / "analysis" / "cnn_center_relu_1011" / "cost.py"
OUT = REPO / "results" / "cnn_center_relu_1011"
SCR = REPO / "results" / "_checks_cnn_center_relu_1011"
REF_R = REPO / "results" / "cnn_drive_verify_1009" / "R"
PY = sys.executable
LAYERS = ("c1", "c2", "f1", "f2")
BATCH = CN.BATCH
DEV = H.setup("cuda")
_di = torch.cuda.current_device()
torch.cuda.set_per_process_memory_fraction(
    min(1.0, 6.0 / (torch.cuda.get_device_properties(_di).total_memory / 2 ** 30)), _di)
CIFAR = RC.Cifar10()
RESULTS: dict = {}
DUMP = OUT / "checks.json"


# --------------------------------------------------------------------------
# infrastructure
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
    return {k: v for k, v in r.items() if k != "pass" and not isinstance(v, (dict, list))}


def dump() -> None:
    checks = {k: v for k, v in RESULTS.items() if k.startswith("K-") and isinstance(v, dict) and "pass" in v}
    RESULTS["all_pass"] = bool(checks) and all(
        v.get("pass") and v.get("all_mutations_detected", True) for v in checks.values())
    DUMP.parent.mkdir(parents=True, exist_ok=True)
    DUMP.write_text(json.dumps(RESULTS, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def run_check(key: str, title: str, fn, mutations: list, derivation: str, target: Path) -> None:
    t0 = time.time()
    base = fn(load(target))
    entry = {"title": title, "target": str(target.relative_to(REPO)), "threshold_derivation": derivation,
             **base, "mutations": []}
    for label, subs in mutations:
        try:
            mod = load(target, subs)
            r = fn(mod)
            entry["mutations"].append({"mutation": label, "check_pass_on_mutant": bool(r["pass"]),
                                       "detected": not r["pass"], "mutant_result": brief(r)})
        except Exception as e:          # recorded; an anchor error is a defect of the check, not a detection
            anchor = isinstance(e, AssertionError) and "mutation anchor" in str(e)
            entry["mutations"].append({"mutation": label, "check_pass_on_mutant": False, "detected": not anchor,
                                       "raised": repr(e)[:400]})
    if mutations:
        entry["all_mutations_detected"] = all(m["detected"] for m in entry["mutations"])
    entry["seconds"] = time.time() - t0
    RESULTS[key] = entry
    print(key, "pass" if entry["pass"] else "FAIL",
          "" if not mutations else f"mutations {sum(m['detected'] for m in entry['mutations'])}/{len(mutations)}",
          f"{entry['seconds']:.1f}s", flush=True)
    dump()


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    a, b = a.detach(), b.detach()
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype == torch.float32:
        return bool((a.view(torch.int32) == b.view(torch.int32)).all())
    if a.dtype == torch.float64:
        return bool((a.view(torch.int64) == b.view(torch.int64)).all())
    return bool((a == b).all())


def same_val(x, y) -> bool:
    if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
        return True
    return x == y


def quiet(msg):
    return None


# --------------------------------------------------------------------------
# the spec's door forward, written independently (bundled ops, so that it is bit-comparable)
# --------------------------------------------------------------------------

def ref_forward_bundled(P, xb, m, h=True):
    """Spec §3.1 door H on the engine's bundled layout; m is a list of (R, width) constants."""
    Wc1, bc1, Wc2, bc2, Wf1, bf1, Wf2, bf2, Wf3, bf3 = [p for p in P]
    R, B = xb.shape[:2]
    C = CN.CHANNELS
    x = xb.transpose(0, 1).reshape(B, R * 3, 32, 32)
    z1 = F.conv2d(x, Wc1.reshape(R * C, 3, 5, 5), bc1.reshape(-1), padding=2, groups=R)
    p1 = F.max_pool2d(torch.clamp(z1, min=0.0), 2, 2)
    if h:
        p1 = p1 - m[0].detach().reshape(1, R * C, 1, 1)
    z2 = F.conv2d(p1, Wc2.reshape(R * C, C, 5, 5), bc2.reshape(-1), padding=2, groups=R)
    p2 = F.max_pool2d(torch.clamp(z2, min=0.0), 2, 2)
    if h:
        p2 = p2 - m[1].detach().reshape(1, R * C, 1, 1)
    hh = p2.reshape(B, R, CN.FLAT).transpose(0, 1)
    z3 = torch.baddbmm(bf1[:, None, :], hh, Wf1.transpose(1, 2))
    a3 = torch.clamp(z3, min=0.0)
    i4 = a3 - m[2].detach()[:, None, :] if h else a3
    z4 = torch.baddbmm(bf2[:, None, :], i4, Wf2.transpose(1, 2))
    a4 = torch.clamp(z4, min=0.0)
    i5 = a4 - m[3].detach()[:, None, :] if h else a4
    logits = torch.baddbmm(bf3[:, None, :], i5, Wf3.transpose(1, 2))
    return z1, None, z2, None, z3, None, z4, None, logits


def ref_door_forward_host(params, x, m):
    """Spec §3.1 for ONE run in the host's (unbundled) layout; m = 4 constant vectors."""
    Wc1, bc1, Wc2, bc2, Wf1, bf1, Wf2, bf2, Wf3, bf3 = params
    z1 = F.conv2d(x, Wc1, bc1, padding=2)
    p1 = F.max_pool2d(torch.clamp(z1, min=0.0), 2, 2) - m[0].view(1, -1, 1, 1)
    z2 = F.conv2d(p1, Wc2, bc2, padding=2)
    p2 = F.max_pool2d(torch.clamp(z2, min=0.0), 2, 2) - m[1].view(1, -1, 1, 1)
    h = p2.flatten(1)
    a3 = torch.clamp(h @ Wf1.T + bf1, min=0.0) - m[2]
    a4 = torch.clamp(a3 @ Wf2.T + bf2, min=0.0) - m[3]
    return a4 @ Wf3.T + bf3


# --------------------------------------------------------------------------
# K-same, K-repro, K-center, K-relu, K-host
# --------------------------------------------------------------------------

def ck_same(M) -> dict:
    seeds = (10, 11, 12)
    A = M.Bundle([("ref", s) for s in seeds], CIFAR, DEV, graph=False)
    B = LRM.BundleLR([("R", s) for s in seeds], CIFAR, DEV, graph=False)
    for Z in (A, B):
        Z.new_labels()
        for _ in range(2):
            Z.run_epoch()
    st = all(bits_equal(a, b) for a, b in zip(A._state_tensors(), B._state_tensors()))
    v = all(bits_equal(a, b) for a, b in zip(A.act.V, B.act.V))
    ra, rb = M.evaluate(A.P, A.X, A.Y, A.act), E.evaluate(B.P, B.X, B.Y, B.act)
    ev = all(same_val(ra[r][k], rb[r][k]) for r in range(len(seeds)) for k in rb[r])
    sa, sb = M.switch_eval(A.P, A.X, A.Y, A.act), E.switch_eval(B.P, B.X, B.Y, B.act)
    sw = all(bits_equal(x, y) for x, y in zip(sa, sb))
    m0 = all(bool((t == 0).all()) for t in A.act.m)
    return {"state_bit_identical": st, "V_bit_identical": v, "evaluate_identical": ev,
            "switch_eval_identical": sw, "door_m_stays_zero": m0, "pass": st and v and ev and sw and m0}


def ck_repro(M) -> dict:
    """The CLI with arm ref, R = 1, seed 10, 3 tasks reproduces results/cnn_drive_verify_1009/R."""
    tmp = SCR / "repro_ref_s10_t3"
    shutil.rmtree(tmp, ignore_errors=True)
    cmd = [PY, str(ENGINE), "run", "--arm", "ref", "--seeds", "10", "--tasks", "3", "--out", str(tmp),
           "--no-resume"]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    if p.returncode != 0:
        return {"returncode": p.returncode, "stderr": p.stderr[-2000:], "pass": False}
    new = pd.read_csv(tmp / "per_task.csv", dtype=str, keep_default_na=False)
    old = pd.read_csv(REF_R / "per_task.csv", dtype=str, keep_default_na=False)
    cols = [c for c in old.columns if c in new.columns and c != "arm"]
    missing = [c for c in old.columns if c not in new.columns]
    diff = [c for c in cols if not (new[c].tolist() == old[c].tolist())]
    cn, co = np.load(tmp / "curves.npy"), np.load(REF_R / "curves.npy")
    curves_eq = cn.shape == co.shape and bool((cn.view(np.int32) == co.view(np.int32)).all())
    for f in (tmp / "ckpt").glob("*.pt"):
        f.unlink()
    return {"n_common_columns": len(cols), "columns_missing_in_new": missing, "columns_differing": diff,
            "curves_bit_identical": curves_eq, "pass": not diff and not missing and curves_eq and len(cols) > 40}


def ck_center(M) -> dict:
    seeds = list(range(10, 20))
    Bc = M.Bundle([("C", s) for s in seeds], CIFAR, DEV, graph=False)
    CC.BundleLRc.center = CC.train_channel_mean(CIFAR)
    Bl = CC.BundleLRc([("LRc", s) for s in seeds], CIFAR, DEV, graph=False)
    eq_x = bits_equal(Bc.X, Bl.X)
    Br = M.Bundle([("ref", s) for s in seeds], CIFAR, DEV, graph=False)
    mr, mc = CC.mean_patch(Br.X).norm(dim=1), CC.mean_patch(Bc.X).norm(dim=1)
    ok_mean = float(mc.max()) < 0.1 * float(mr.min())
    # Cr: r = mu - |mu| v, v = the part of u orthogonal to mu-hat (spec §3.3), recomputed here
    mu = CC.train_channel_mean(CIFAR).double()
    g = H.stream("cnn_center_relu_1011_cr", 0)
    u = torch.randn(3, generator=g, dtype=torch.float64)
    mh = mu / mu.norm()
    v = u - (u @ mh) * mh
    v = v / v.norm()
    r_exp = (mu - mu.norm() * v).float()
    Bcr = M.Bundle([("Cr", s) for s in seeds], CIFAR, DEV, graph=False)
    r_ok = Bcr.shift is not None and bits_equal(Bcr.shift, r_exp)
    x_ok = bits_equal(Bcr.X, Br.X - r_exp.to(DEV).view(1, 1, 3, 1, 1))
    raw = Br.X.double().mean((1, 3, 4))                                  # (R, 3) subset channel means
    crm = Bcr.X.double().mean((1, 3, 4))
    ratio = crm.norm(dim=1) / raw.norm(dim=1)
    cos = (crm * raw).sum(1) / (crm.norm(dim=1) * raw.norm(dim=1))
    # derived tolerance: subset mean = mu + d, shifted mean = |mu| v + d, so |ratio - 1| and |cos|
    # are at most 2|d| / (|mu| - |d|) (spec §6 asks for 5%; this bound is tighter)
    d = (raw - mu.to(raw.device)).norm(dim=1)
    bound = 2 * d / (mu.norm().to(raw.device) - d)
    cr_ok = bool(((ratio - 1).abs() <= bound * (1 + 1e-9)).all() and (cos.abs() <= bound).all()
                 and ((ratio - 1).abs() <= 0.05).all())
    return {"C_input_bit_identical_to_LRc": eq_x, "mean_patch_norm_raw": [float(x) for x in mr],
            "mean_patch_norm_C": [float(x) for x in mc], "C_mean_patch_below_0.1_raw": ok_mean,
            "Cr_r": r_exp.tolist(), "Cr_v": v.tolist(), "Cr_r_matches_formula": r_ok, "Cr_input_is_raw_minus_r": x_ok,
            "Cr_mean_norm_ratio_to_raw": [float(x) for x in ratio], "Cr_cos_to_raw_mean": [float(x) for x in cos],
            "Cr_bound": [float(x) for x in bound],
            "pass": eq_x and ok_mean and r_ok and x_ok and cr_ok}


def ck_relu(M) -> dict:
    act = M.ReLUDoors(1, False, DEV)
    g = torch.Generator(device="cpu").manual_seed(0)
    z = torch.randn(4096, generator=g).to(DEV)
    z[:16] = 0.0
    z[16:32] = -0.0
    phi_ok = bits_equal(act.phi(z, 0), torch.clamp(z, min=0.0))
    dphi_ok = bits_equal(act.dphi(z, 0), (z > 0).to(z.dtype))
    zz = z.clone().requires_grad_(True)
    act.phi(zz, 0).sum().backward()
    gr = zz.grad
    pos = bool((gr[z > 0] == 1).all())
    neg = bool((gr[z < 0] == 0).all())
    return {"phi_is_clamp": phi_ok, "diagnostic_gate_is_z_gt_0": dphi_ok, "train_grad_1_for_z_gt_0": pos,
            "train_grad_0_for_z_lt_0": neg, "train_grad_at_z_eq_0_reported": sorted(set(gr[z == 0].tolist())),
            "pass": phi_ok and dphi_ok and pos and neg}


def ck_host(M) -> dict:
    res = {}
    mu = CC.train_channel_mean(CIFAR).to(DEV)
    for arm in ("ref", "C", "H", "CH"):
        B = M.Bundle([(arm, 10)], CIFAR, DEV, graph=False)
        B.new_labels()
        B.new_order()
        params = CN.init_params(10, DEV)
        xh = CN.images(CIFAR, CN.subset_idx(10), DEV)
        if arm in ("C", "CH"):
            xh = xh - mu.view(1, 3, 1, 1)
        idx = B.order[0, :BATCH]
        if arm in ("ref", "C"):
            lh = F.cross_entropy(CN.forward_cnn(params, xh[idx], H.ARMS["R"])[8], B.Y[0, idx])
        else:
            # realistic non-zero door means: the uncentred output means on 64 images at init
            with torch.no_grad():
                out = M.forward(B.P, B.X[:, :64], B.act)
                for l in range(4):
                    B.act.m[l].copy_(B.act.out_mean(out[2 * l], l))
            mv = [t[0].clone() for t in B.act.m]
            lh = F.cross_entropy(ref_door_forward_host(params, xh[idx], mv), B.Y[0, idx])
        gh = torch.autograd.grad(lh, params)
        lb = F.cross_entropy(M.forward(B.P, B.X[:, idx], B.act)[8][0], B.Y[0, idx])
        gb = torch.autograd.grad(lb, B.P)
        lr_ = abs(float(lb) - float(lh)) / abs(float(lh))
        gr_ = max(rel(a[0], b) for a, b in zip(gb, gh))
        res[arm] = {"loss_rel": lr_, "grad_rel_max": gr_, "pass": lr_ < 1e-5 and gr_ < 1e-4,
                    "door_m_norms": [float(t.norm()) for t in B.act.m]}
    return {**{f"{a}_{k}": v for a, r in res.items() for k, v in r.items() if k != "pass"},
            "pass": all(r["pass"] for r in res.values())}


# --------------------------------------------------------------------------
# door H: EMA, gradient, evaluation, first update, read-only m = 0 evaluation, loop
# --------------------------------------------------------------------------

def ck_h_ema(M) -> dict:
    B = M.Bundle([("H", 10), ("H", 11)], CIFAR, DEV, graph=False)
    B.new_labels()
    B.new_order()
    steps, ok_all, moved = 6, True, 0
    for j in range(steps):
        Pb = [p.detach().clone() for p in B.P]
        mb = [t.clone() for t in B.act.m]
        idx = B.order[:, j * BATCH:(j + 1) * BATCH]
        with torch.no_grad():
            out = ref_forward_bundled(Pb, B.X[B.ar, idx], mb)
        exp = []
        for l in range(4):
            a = torch.clamp(out[2 * l], min=0.0)
            bm = F.max_pool2d(a, 2, 2).mean((0, 2, 3)).view(2, -1) if l < 2 else a.mean(1)
            exp.append(mb[l].clone().mul_(1 - 0.01).add_(bm, alpha=0.01))
        B.step(j)
        ok_all &= all(bits_equal(B.act.m[l], exp[l]) for l in range(4))
        moved += all(bool((B.act.m[l] != mb[l]).any()) for l in range(4))
    mb = [t.clone() for t in B.act.m]
    B.new_labels()
    boundary = all(bits_equal(a, b) for a, b in zip(mb, B.act.m))
    hupd = bool((B.act.hupd == steps).all())
    return {"ema_bit_identical_every_step": ok_all, "m_moved_every_step": moved == steps,
            "new_labels_keeps_m": boundary, "h_upd_counts_steps": hupd,
            "pass": ok_all and moved == steps and boundary and hupd}


def ck_h_grad(M) -> dict:
    B = M.Bundle([("H", 10), ("H", 11)], CIFAR, DEV, graph=False)
    B.new_labels()
    B.new_order()
    for j in range(3):
        B.step(j)
    no_grad = all(not t.requires_grad for t in B.act.m)
    idx = B.order[:, 3 * BATCH:4 * BATCH]
    xb, yb = B.X[B.ar, idx], B.Y[B.ar, idx]

    def grads(fwd):
        lv = F.cross_entropy(fwd(B.P, xb)[8].reshape(-1, 10), yb.reshape(-1),
                             reduction="none").view(2, BATCH).mean(1)
        return [g.detach().clone() for g in torch.autograd.grad(lv.sum(), B.P)]

    g_mod = grads(lambda P, x: M.forward(P, x, B.act))
    mc = [t.detach().clone() for t in B.act.m]
    g_ref = grads(lambda P, x: ref_forward_bundled(P, x, mc))
    eq = all(bits_equal(a, b) for a, b in zip(g_mod, g_ref))
    return {"m_requires_no_grad": no_grad, "grads_bit_identical_to_constant_m": eq, "pass": no_grad and eq}


def _ce_sum(logits, yb, R):
    b = yb.shape[1]
    return F.cross_entropy(logits.reshape(-1, 10), yb.reshape(-1), reduction="none").view(R, b).double().sum(1)


def ck_h_eval(M) -> dict:
    B = M.Bundle([("H", 10), ("H", 11)], CIFAR, DEV, graph=False)
    B.new_labels()
    B.run_epoch()
    rows = M.evaluate(B.P, B.X, B.Y, B.act)

    def fit_ce(mm):
        ce = torch.zeros(2, dtype=torch.float64, device=DEV)
        with torch.no_grad():
            for i0 in range(0, CN.N_IMAGES, 20):
                ce += _ce_sum(ref_forward_bundled(B.P, B.X[:, i0:i0 + 20], mm)[8], B.Y[:, i0:i0 + 20], 2)
        return ce / CN.N_IMAGES

    ref_m = fit_ce(B.act.m)
    ref_0 = fit_ce([torch.zeros_like(t) for t in B.act.m])
    ok = all(rows[r]["fit_ce"] == float(ref_m[r]) for r in range(2))
    differs = all(rows[r]["fit_ce"] != float(ref_0[r]) for r in range(2))
    sce, _ = M.switch_eval(B.P, B.X, B.Y, B.act)
    sw_ok = all(float(sce[r]) == float(ref_m[r]) for r in range(2))   # same labels: switch CE = fit CE
    return {"fit_ce_uses_current_m": ok, "fit_ce_differs_without_m": differs, "switch_eval_uses_current_m": sw_ok,
            "pass": ok and differs and sw_ok}


def ck_h_first(M) -> dict:
    out = {}
    for door, base in (("H", "ref"), ("CH", "C")):
        A = M.Bundle([(base, 10), (base, 11)], CIFAR, DEV, graph=False)
        Bh = M.Bundle([(door, 10), (door, 11)], CIFAR, DEV, graph=False)
        for Z in (A, Bh):
            Z.new_labels()
            Z.new_order()
        A.step(0)
        Bh.step(0)
        first = all(bits_equal(a, b) for a, b in zip(A._state_tensors(), Bh._state_tensors()))
        nonzero = all(bool((t != 0).any()) for t in Bh.act.m)
        for j in (1, 2):
            A.step(j)
            Bh.step(j)
        later = any(not bits_equal(a, b) for a, b in zip(A.P, Bh.P))
        out[door] = {"first_update_equals_" + base: first, "m_nonzero_after_first": nonzero,
                     "differs_after_3": later, "ok": first and nonzero and later}
    return {**{f"{d}_{k}": v for d, r in out.items() for k, v in r.items() if k != "ok"},
            "pass": all(r["ok"] for r in out.values())}


def _snapshot(B):
    t = [x.detach().clone() for x in B._state_tensors()] + [x.clone() for x in B.act.m] + \
        [B.act.hupd.clone()] + [x.clone() for x in B.act.V]
    g = [(B.g_lab[s].get_state().clone(), B.g_batch[s].get_state().clone()) for s in B.useeds]
    return t, g


def ck_evaloff(M) -> dict:
    B = M.Bundle([("H", 10), ("H", 11)], CIFAR, DEV, graph=False)
    B.new_labels()
    B.run_epoch()
    t0, g0 = _snapshot(B)
    rows, _ = M.diagnostics(B.P, B.X, B.Y, B.act, m0_eval=True)
    t1, g1 = _snapshot(B)
    ro = all(bits_equal(a, b) for a, b in zip(t0, t1)) and all(
        bool((a[0] == b[0]).all() and (a[1] == b[1]).all()) for a, b in zip(g0, g1))
    has = all("memo_acc_m0" in r for r in rows)
    B1 = M.Bundle([("H", 10), ("H", 11)], CIFAR, DEV, graph=False)
    B2 = M.Bundle([("H", 10), ("H", 11)], CIFAR, DEV, graph=False)
    for Z in (B1, B2):
        Z.new_labels()
        Z.run_epoch()
    M.diagnostics(B1.P, B1.X, B1.Y, B1.act, m0_eval=True)
    for Z in (B1, B2):
        Z.run_epoch()
    s1, _ = _snapshot(B1)
    s2, _ = _snapshot(B2)
    traj = all(bits_equal(a, b) for a, b in zip(s1, s2))
    return {"read_only": ro, "m0_columns_present": has, "trajectory_unchanged": traj, "pass": ro and has and traj}


def ck_ident(M) -> dict:
    out = {}
    for arm in ("ref", "CH"):
        B = M.Bundle([(arm, 10), (arm, 11)], CIFAR, DEV, graph=False)
        B.new_labels()
        for _ in range(2):
            B.run_epoch()
        rows, _ = M.diagnostics(B.P, B.X, B.Y, B.act)
        out[arm] = {l: max(rows[r][f"ident_ratio_{l}"] for r in range(2)) for l in LAYERS}
    ok = all(v <= 1.0 for r in out.values() for v in r.values())
    return {**{f"{a}_max_ratio_{l}": v for a, r in out.items() for l, v in r.items()}, "pass": ok}


def ck_graph(M) -> dict:
    out = {}
    for arm in ("H", "CH"):
        slots = [(arm, s) for s in (10, 11, 12, 13)]
        Bg = M.Bundle(slots, CIFAR, DEV, graph=True)
        Be = M.Bundle(slots, CIFAR, DEV, graph=False)
        for Z in (Bg, Be):
            Z.new_labels()
            for _ in range(2):
                Z.run_epoch()
        tg = (*Bg._state_tensors(), *Bg.act.V, *Bg.act.m, Bg.act.hupd)
        te = (*Be._state_tensors(), *Be.act.V, *Be.act.m, Be.act.hupd)
        eq = all(bits_equal(a, b) for a, b in zip(tg, te))
        moved = bool((Bg.P[0] != E.stack_init([s for a, s in slots], DEV)[0]).any())
        out[arm] = eq and moved
    return {**{f"{a}_graph_eq_eager_and_moved": v for a, v in out.items()}, "pass": all(out.values())}


def ck_indep(M) -> dict:
    B = M.Bundle([("CH", s) for s in (10, 11, 12)], CIFAR, DEV, graph=False)
    B.new_labels()
    B.run_epoch()
    B.new_order()
    idx = B.order[:, :BATCH]

    def lg():
        out = M.forward(B.P, B.X[B.ar, idx], B.act)
        lv = F.cross_entropy(out[8].reshape(-1, 10), B.Y[B.ar, idx].reshape(-1),
                             reduction="none").view(3, -1).mean(1)
        g = torch.autograd.grad(lv.sum(), B.P)
        return lv.detach().clone(), [q.detach().clone() for q in g]

    def same_others(a, b):
        (la, ga), (lb, gb) = a, b
        return all(bits_equal(x[r], y[r]) for x, y in zip(ga, gb) for r in (0, 2)) and \
            bits_equal(la[[0, 2]], lb[[0, 2]])

    def moved1(a, b):
        (la, ga), (lb, gb) = a, b
        return bool(la[1] != lb[1]) and any(not bits_equal(x[1], y[1]) for x, y in zip(ga, gb))

    gen = torch.Generator(device="cpu").manual_seed(1)
    r0 = lg()
    with torch.no_grad():
        for p in B.P:
            p[1] += 0.05 * torch.randn(p[1].shape, generator=gen).to(DEV)
    r1 = lg()
    with torch.no_grad():
        B.X[1] = torch.rand(B.X[1].shape, generator=gen).to(DEV)
    r2 = lg()
    with torch.no_grad():
        for t in B.act.m:
            t[1] += 0.1
    r3 = lg()
    checks = {"param": same_others(r0, r1) and moved1(r0, r1), "input": same_others(r1, r2) and moved1(r1, r2),
              "door_m": same_others(r2, r3) and moved1(r2, r3)}
    return {**checks, "pass": all(checks.values())}


def ck_loop(M) -> dict:
    """`run` (eager) = a manual replay of the task loop; m is carried over the task boundary."""
    tmp = SCR / "loop_H"
    shutil.rmtree(tmp, ignore_errors=True)
    M.run("H", [10, 11], tmp, 2, 2, DEV, ckpt_tasks=(), graph=False, resume=False, log=quiet)
    st = torch.load(tmp / "state.pt", map_location="cpu", weights_only=False)
    B = M.Bundle([("H", 10), ("H", 11)], CIFAR, DEV, epochs=2, graph=False)
    for _ in (1, 2):
        B.new_labels()
        B.act.hupd.zero_()
        for _ in range(2):
            B.run_epoch()
    eqP = all(bits_equal(a.detach().cpu(), b) for a, b in zip(B.P, st["state"]["P"]))
    eqm = all(bits_equal(a.cpu(), b) for a, b in zip(B.act.m, st["state"]["act"]["m"]))
    rows = pd.read_csv(tmp / "per_task.csv")
    hupd = bool((rows[[f"h_upd_{l}" for l in LAYERS]] == 150).all().all())
    u = np.load(tmp / "units.npz")
    units_ok = list(u["tasks_done"]) == [0, 1, 2] and (tmp / "t0.csv").exists()
    return {"params_equal_manual": eqP, "door_m_equal_manual": eqm, "h_upd_150_each_task": hupd,
            "units_and_t0": units_ok, "pass": eqP and eqm and hupd and units_ok}


def ck_resume(M) -> dict:
    a, b = SCR / "resume_A", SCR / "resume_B"
    for d in (a, b):
        shutil.rmtree(d, ignore_errors=True)
    M.run("CH", [10, 11], a, 2, 2, DEV, ckpt_tasks=(1, 2), graph=True, resume=False, log=quiet)

    class Stop(Exception):
        pass

    def stopper(msg):
        if "task  1 " in msg:
            raise Stop()

    try:
        M.run("CH", [10, 11], b, 2, 2, DEV, ckpt_tasks=(1, 2), graph=True, resume=False, log=stopper)
    except Stop:
        pass
    stopped = len(pd.read_csv(b / "per_task.csv")) == 2
    M.run("CH", [10, 11], b, 2, 2, DEV, ckpt_tasks=(1, 2), graph=True, resume=True, log=quiet)
    same_csv = (a / "per_task.csv").read_text() == (b / "per_task.csv").read_text() and \
        (a / "t0.csv").read_text() == (b / "t0.csv").read_text()
    ca, cb = np.load(a / "curves.npy"), np.load(b / "curves.npy")
    same_curves = bool((ca.view(np.int32) == cb.view(np.int32)).all())
    ua, ub = np.load(a / "units.npz"), np.load(b / "units.npz")
    same_units = sorted(ua.files) == sorted(ub.files) and all(
        np.array_equal(ua[k], ub[k], equal_nan=True) if ua[k].dtype != np.bool_ else np.array_equal(ua[k], ub[k])
        for k in ua.files)
    sa = torch.load(a / "state.pt", map_location="cpu", weights_only=False)["state"]
    sb = torch.load(b / "state.pt", map_location="cpu", weights_only=False)["state"]
    same_state = all(bits_equal(x, y) for k in ("P", "m", "v") for x, y in zip(sa[k], sb[k])) and \
        all(bits_equal(x, y) for x, y in zip(sa["act"]["m"], sb["act"]["m"]))
    return {"stopped_after_task1": stopped, "per_task_and_t0_identical": same_csv, "curves_identical": same_curves,
            "units_identical": same_units, "state_identical": same_state,
            "pass": stopped and same_csv and same_curves and same_units and same_state}


# --------------------------------------------------------------------------
# K-verdict: synthetic per_task tables
# --------------------------------------------------------------------------

def _mf(seed, t):
    return 0.110 + 0.004 * ((seed + t) % 3)


def _floors(seed, sqrt_n):
    mf = np.array([_mf(seed, t) for t in range(21, 31)])
    return mf.mean() + 3 * mf.std(ddof=1) / math.sqrt(sqrt_n)


def synth_arm(arm, online_fn, mob_fn=None, dead_fn=None, hupd_fn=None, t1=0.95, memo_m0=0.2):
    rows = []
    for s in range(10, 20):
        for t in range(1, 31):
            on = online_fn(s, t)
            r = {"arm": arm, "seed": s, "task": t, "online_acc": float(on), "memo_acc": 1.0 if on > 0.3 else 0.12,
                 "diverged": False, "major_frac": _mf(s, t)}
            for l in LAYERS:
                r[f"mob_{l}"] = 0.3 if mob_fn is None else mob_fn(s, t, l)
                r[f"dead_frac_{l}"] = 0.0 if dead_fn is None else dead_fn(s, t, l)
                if arm in ("H", "CH"):
                    r[f"h_upd_{l}"] = 30000 if hupd_fn is None else hupd_fn(s, t, l)
            if arm in ("H", "CH"):
                r["memo_acc_m0"] = memo_m0
            rows.append(r)
    return pd.DataFrame(rows)


def _collapse(at):
    """online: 0.95 until task `at` - 1, then 0.100 (well below every floor)."""
    return lambda s, t: 0.95 if t < at else 0.100


def _hold(level=0.95, slope=0.0):
    return lambda s, t: level - slope * (t - 1)


def scenarios():
    """(name, frames, cr_frame, checks_ok, expected dict)."""
    out = []
    mob_ref = lambda s, t, l: 0.005 if (l == "c2" and t >= 2) or (l != "c2" and t >= 4) else 0.3   # noqa: E731
    mob_C = lambda s, t, l: 0.005 if (l == "c2" and t >= 4) or (l in ("f1", "f2") and t >= 6) else 0.3  # noqa: E731
    mob_H = lambda s, t, l: 0.005 if (l == "c1" and t >= 8) else 0.3   # noqa: E731
    base = {"ref": synth_arm("ref", _collapse(3), mob_ref), "C": synth_arm("C", _collapse(6), mob_C),
            "H": synth_arm("H", _collapse(9), mob_H), "CH": synth_arm("CH", _hold(0.97, 0.001))}
    cr_ref = synth_arm("Cr", _collapse(3))
    out.append(("A_transfer", base, cr_ref, True,
                {"ref": "FAIL", "C": "FAIL", "H": "FAIL", "CH": "RESCUED", "N": "NEED_CH", "DOORS": "DOORS_TRANSFER",
                 "LAYER": "LAYER_CORRESPONDENCE", "ref_sub": "FAIL_FLOOR", "RESIDUAL": True,
                 "Cr": "CONST_SHIFT_NO_EFFECT", "EVALOFF_CH": "EVALOFF_DEPENDS", "T1_LOWER_CH": False}))
    # C alone rescues -> NEED_C, PARTIAL (the MLP order fails); Cr acts like C
    f = dict(base, C=synth_arm("C", _hold(0.96)))
    out.append(("B_C_rescues", f, synth_arm("Cr", _hold(0.96)), True,
                {"C": "RESCUED", "N": "NEED_C", "DOORS": "DOORS_PARTIAL", "Cr": "CONST_SHIFT_ACTS_LIKE_C"}))
    # CH fails -> DOORS_FAIL
    f = dict(base, CH=synth_arm("CH", _collapse(12)))
    out.append(("C_CH_fails", f, cr_ref, True, {"CH": "FAIL", "N": "NONE_RESCUED", "DOORS": "DOORS_FAIL",
                                                 "RESIDUAL": True}))
    # ref survives -> NOT_REPRODUCED
    f = dict(base, ref=synth_arm("ref", _hold(0.9)))
    out.append(("D_ref_alive", f, cr_ref, True, {"ref": "RESCUED", "N": "NOT_REPRODUCED", "DOORS": "NOT_REPRODUCED",
                                                  "LAYER": "NOT_REPRODUCED"}))
    # checks failed -> INAPPLICABLE
    out.append(("E_checks_fail", base, cr_ref, False, {"N": "INAPPLICABLE", "DOORS": "INAPPLICABLE"}))
    # (C): a live layer with h_upd short -> H INAPPLICABLE; a dead layer with h_upd short is exempt
    f = dict(base, CH=synth_arm("CH", _hold(0.97, 0.001), hupd_fn=lambda s, t, l: 29000 if (l == "f2" and t == 7) else 30000))
    out.append(("F_door_broken", f, cr_ref, True, {"CH": "INAPPLICABLE", "DOORS": "INAPPLICABLE"}))
    f = dict(base, CH=synth_arm("CH", _hold(0.97, 0.001), dead_fn=lambda s, t, l: 1.0 if (l == "f2" and t == 7 and s == 12) else 0.0,
                                hupd_fn=lambda s, t, l: 12000 if (l == "f2" and t == 7 and s == 12) else 30000))
    out.append(("G_dead_layer_exempt", f, cr_ref, True, {"CH": "RESCUED", "DOORS": "DOORS_TRANSFER"}))
    # CH kept in 8/10 -> PARTIAL; 9/10 with one seed exactly at 0.5 -> RESCUED
    f = dict(base, CH=synth_arm("CH", lambda s, t: 0.95 if s < 18 else 0.100))
    out.append(("H_CH_8of10", f, cr_ref, True, {"CH": "PARTIAL", "DOORS": "DOORS_PARTIAL", "N": "NONE_RESCUED"}))
    f = dict(base, CH=synth_arm("CH", lambda s, t: 0.95 if s < 18 else (0.5 if (s == 18 and t > 20) else (0.95 if s == 18 else 0.100))))
    out.append(("I_CH_9of10_at_line", f, cr_ref, True, {"CH": "RESCUED", "DOORS": "DOORS_TRANSFER"}))
    # FAIL subtype: two ref seeds between the sqrt(30) and sqrt(10) floors -> still FAIL_FLOOR (10 at floor)
    mid = {s: 0.5 * (_floors(s, 10) + _floors(s, 30)) for s in range(10, 20)}
    f = dict(base, ref=synth_arm("ref", lambda s, t: 0.95 if t < 3 else (mid[s] if s in (10, 11) else 0.100), mob_ref))
    out.append(("J_floor_sqrt10", f, cr_ref, True, {"ref": "FAIL", "ref_sub": "FAIL_FLOOR"}))
    # main window: CH high in t11-20 and low in t21-30 -> FAIL (a wider window would change it)
    f = dict(base, CH=synth_arm("CH", lambda s, t: 0.95 if t <= 20 else 0.100))
    out.append(("K_window", f, cr_ref, True, {"CH": "FAIL", "DOORS": "DOORS_FAIL"}))
    # condition 4: C PARTIAL and above CH in its kept seeds -> CH - C TIE -> PARTIAL, not TRANSFER
    f = dict(base, C=synth_arm("C", lambda s, t: 0.99 if s < 16 else (0.95 if t < 3 else 0.100)))
    out.append(("L_cond4", f, cr_ref, True, {"C": "PARTIAL", "CH": "RESCUED", "DOORS": "DOORS_PARTIAL"}))
    # ladder order: C and H both rescue -> NEED_C
    f = dict(base, C=synth_arm("C", _hold(0.96)), H=synth_arm("H", _hold(0.96)))
    out.append(("M_ladder", f, cr_ref, True, {"N": "NEED_C"}))
    # H alone rescues, C fails -> NEED_H, TRANSFER, H differs from the MLP; no layer correspondence
    f = dict(base, H=synth_arm("H", _hold(0.96)))
    out.append(("N_H_rescues", f, cr_ref, True, {"H": "RESCUED", "N": "NEED_H", "DOORS": "DOORS_TRANSFER",
                                                  "M_H": "H_DIFFERS_FROM_MLP", "LAYER": "LAYER_CORRESPONDENCE_NOT_SHOWN"}))
    # no separating window -> CR_UNINFORMATIVE; evaloff no dependence; T1_LOWER on CH; no residual loss
    f = dict(base, C=synth_arm("C", _collapse(3), mob_C),
             CH=synth_arm("CH", lambda s, t: 0.90 if t <= 1 else 0.96, memo_m0=1.0))
    out.append(("O_misc", f, cr_ref, True, {"Cr": "CR_UNINFORMATIVE", "EVALOFF_CH": "EVALOFF_NO_DEPENDENCE",
                                             "T1_LOWER_CH": True, "RESIDUAL": False}))
    return out


def ck_verdict(V) -> dict:
    fails = []
    n = 0
    for name, frames, crf, checks_ok, exp in scenarios():
        res = V.analyze_frames(frames, checks_ok, 30000, cr_frame=crf, c_door_ok=True)
        got = {"ref": res["arms"]["ref"]["label"], "C": res["arms"]["C"]["label"], "H": res["arms"]["H"]["label"],
               "CH": res["arms"]["CH"]["label"], "N": res["N"], "DOORS": res["DOORS"], "LAYER": res["LAYER"]["label"],
               "ref_sub": res["arms"]["ref"].get("fail_subtype"), "RESIDUAL": res["flags"]["RESIDUAL_LOSS_CH"]["set"],
               "Cr": res["Cr"]["label"], "EVALOFF_CH": res["EVALOFF"]["CH"]["label"],
               "T1_LOWER_CH": res["flags"]["T1_LOWER"]["CH"]["set"], "M_H": res["M_H"]["vs_MLP"]}
        for k, v in exp.items():
            n += 1
            if got[k] != v:
                fails.append(f"{name}:{k} expected {v} got {got[k]}")
    k_ok = [V.kstar(n) for n in (8, 9, 10)] == [8, 8, 9]
    st = V.sign_test([1.0] * 9 + [-1.0])
    s_ok = st["label"] == "HELPS" and abs(st["p"] - 0.021484375) < 1e-15
    return {"n_expectations": n, "failed": fails[:12], "kstar_8_9_10": k_ok, "sign_test_9_1": s_ok,
            "pass": not fails and k_ok and s_ok}


# --------------------------------------------------------------------------
# K-cost
# --------------------------------------------------------------------------

def _gpu_proc_mem() -> dict:
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout
    out = {}
    for line in q.strip().splitlines():
        try:
            pid, mem = [x.strip() for x in line.split(",")]
            out[int(pid)] = int(mem)
        except ValueError:
            pass
    return out


def ck_cost(_M) -> dict:
    res = {}
    epochs = 40
    p = subprocess.run([PY, str(COST), "--arm", "CH", "--epochs", str(epochs)], capture_output=True, text=True, cwd=REPO)
    res["single_CH"] = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else {"err": p.stderr[-800:]}
    go = SCR / "cost_go"
    go.unlink(missing_ok=True)
    procs = {a: subprocess.Popen([PY, str(COST), "--arm", a, "--epochs", str(epochs), "--wait-file", str(go)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=REPO)
             for a in ("ref", "C", "H", "CH")}
    peak = {}
    t_wait = time.time()
    while time.time() - t_wait < 120:
        mem = _gpu_proc_mem()
        for a, pr in procs.items():
            if pr.pid in mem:
                peak[a] = max(peak.get(a, 0), mem[pr.pid])
        if len(peak) == 4 and time.time() - t_wait > 20:
            break
        time.sleep(1)
    go.touch()
    while any(pr.poll() is None for pr in procs.values()):
        mem = _gpu_proc_mem()
        for a, pr in procs.items():
            if pr.pid in mem:
                peak[a] = max(peak.get(a, 0), mem[pr.pid])
        time.sleep(1)
    conc = {}
    for a, pr in procs.items():
        o, e = pr.communicate()
        conc[a] = json.loads(o.strip().splitlines()[-1]) if pr.returncode == 0 else {"err": e[-800:]}
    go.unlink(missing_ok=True)
    res["concurrent_4"] = conc
    res["nvidia_smi_peak_mb"] = peak
    ok = "s_per_epoch" in res["single_CH"] and all("s_per_epoch" in v for v in conc.values())
    if ok:
        res["projected_30_tasks_single_CH_min"] = res["single_CH"]["s_per_epoch"] * 400 * 30 / 60
        res["projected_30_tasks_concurrent_min"] = {a: v["s_per_epoch"] * 400 * 30 / 60 for a, v in conc.items()}
    return {**res, "pass": ok}


# --------------------------------------------------------------------------

CHECKS = {
    "K-same": ("ref 束 = cnn_drive_verify_1009_lr.BundleLR('R')（2 epoch・全状態・evaluate・switch_eval）", ck_same, [
        ("ref でも H が働く", [('"ref": dict(shift=None, h=False),', '"ref": dict(shift=None, h=True),')]),
        ("ReLU を leaky 0.01 に", [('return torch.clamp(z, min=0.0)                                      # host "R"',
                                    'return torch.where(z > 0, z, 0.01 * z)')]),
    ], "bit 一致（許容 0）", ENGINE),
    "K-repro": ("CLI の ref・R=1・seed 10・3 課題が results/cnn_drive_verify_1009/R と per_task の共通列（文字列）と curves で一致",
                ck_repro, [], "文字列一致・bit 一致（許容 0）", ENGINE),
    "K-center": ("C の入力 = cnn_center_1010 の LRc の入力（bit）、平均パッチ < 0.1 倍、Cr の r と入力", ck_center, [
        ("C が平均で割る", [("self.X.sub_(self.shift.to(device).view(1, 1, 3, 1, 1))",
                           "self.X.div_(self.shift.to(device).view(1, 1, 3, 1, 1))")]),
        ("C が別の定数を引く", [('    if kind == "mu":\n        return mu\n', '    if kind == "mu":\n        return mu * 0.9\n')]),
    ], "bit 一致。Cr の平均のノルム比と向きの許容は部分集合の平均と訓練集合の平均の差 d から 2|d|/(|mu|-|d|)（spec の 5% より狭い）", ENGINE),
    "K-relu": ("φ = clamp(z, 0)、診断のゲート z>0、訓練の微分", ck_relu, [
        ("φ を leaky 0.01 に", [('return torch.clamp(z, min=0.0)                                      # host "R"',
                                 'return torch.where(z > 0, z, 0.01 * z)')]),
    ], "bit 一致", ENGINE),
    "K-host": ("1 step の損失と勾配: ref・C は宿主の forward_cnn、H・CH は独立に書いた扉つきの 1 run の順伝播", ck_host, [
        ("φ の中で中心化（φ(z − m)）", [("    a1 = act.phi(z1, 0)\n    p1 = F.max_pool2d(a1, POOL, POOL)\n    if hon:\n        p1 = p1 - m[0].reshape(1, R * CH, 1, 1)\n",
                                         "    a1 = act.phi(z1 - (m[0].reshape(1, R * CH, 1, 1) if hon else 0.0), 0)\n    p1 = F.max_pool2d(a1, POOL, POOL)\n")]),
        ("conv の m を引かない（fc だけ）", [("    if hon:\n        p1 = p1 - m[0].reshape(1, R * CH, 1, 1)\n",
                                            "    if False:\n        p1 = p1 - m[0].reshape(1, R * CH, 1, 1)\n")]),
    ], "損失の相対差 1e-5・勾配の相対差 1e-4（cnn_center_1010 の K-host と同じ。束ねの bmm と宿主の mm の float32 の丸めの差、1 step で 1e-7 程度）", ENGINE),
    "K-H-ema": ("m の更新 = 中心化前の出力（conv は pool の後）のバッチ平均を β=0.01 で追う（更新前の重みの z）、境界で変わらない", ck_h_ema, [
        ("中心化後を追う", [("                bm = self.out_mean(z, l)\n", "                bm = self.out_mean(z, l) - self.m[l]\n")]),
        ("β = 0.1", [("BETA_H = 0.01", "BETA_H = 0.1")]),
        ("pool の前の平均を追う", [("            return F.max_pool2d(a, POOL, POOL).mean((0, 2, 3)).view(self.R, -1)",
                                     "            return a.mean((0, 2, 3)).view(self.R, -1)")]),
        ("Adam の後の重みで bm を取る", [("            self.act.update([out[0].detach(), out[2].detach(), out[4].detach(), out[6].detach()])",
                                          "            o2 = forward(self.P, xb, self.act)\n            self.act.update([o2[0].detach(), o2[2].detach(), o2[4].detach(), o2[6].detach()])")]),
        ("境界（new_labels）で m をリセット", [("    def step(self, j: int) -> None:\n",
                                               "    def new_labels(self):\n        super().new_labels()\n        for t in self.act.m:\n            t.zero_()\n\n    def step(self, j: int) -> None:\n")]),
    ], "bit 一致（許容 0）", ENGINE),
    "K-H-grad": ("m は勾配を持たず、勾配が m を定数とした独立の再計算と bit 一致", ck_h_grad, [
        ("勾配を通す（値は同じで勾配だけバッチ平均を通る）", [("        p1 = p1 - m[0].reshape(1, R * CH, 1, 1)\n",
                                                            "        p1 = p1 - (m[0].reshape(1, R * CH, 1, 1) + (p1.mean((0, 2, 3), keepdim=True) - p1.mean((0, 2, 3), keepdim=True).detach()))\n")]),
    ], "bit 一致（許容 0）", ENGINE),
    "K-H-eval": ("課題末の fit CE・切替の CE がその時点の m での再計算と bit 一致し、m=0 では一致しない", ck_h_eval, [
        ("評価で m を使わない", [("    E.forward = forward\n",
                                 "    E.forward = lambda P, xb, act: forward(P, xb, act, m=[torch.zeros_like(t) for t in act.m])\n")]),
    ], "bit 一致（許容 0）", ENGINE),
    "K-H-first": ("H の最初の更新 = ref の最初の更新（CH = C）、その直後の m は非零、3 更新後は違う", ck_h_first, [
        ("m の初期値が非零", [("        self.m = [torch.zeros(R, w, device=device) for w in WIDTHS]          # door H",
                              "        self.m = [torch.full((R, w), 0.01, device=device) for w in WIDTHS]          # door H")]),
    ], "bit 一致（許容 0）", ENGINE),
    "K-evaloff": ("m = 0 の評価は読むだけ（状態・m・乱数の流れ・2 epoch の軌道が不変）", ck_evaloff, [
        ("m = 0 の評価が m をその場で 0 にする", [("        zeros = [torch.zeros_like(t) for t in act.m]",
                                                  "        zeros = [t.zero_() for t in act.m]")]),
    ], "bit 一致（許容 0）", ENGINE),
    "K-ident": ("z̄_j = ⟨W_j, m_in⟩ + b_j が TF32 と float32 の丸めの上界の内（ref・CH、全 4 層）", ck_ident, [
        ("平均パッチの並びを転置", [("        W = Wt[l].detach().reshape(R, WIDTHS[l], -1).double()            # (R, n, d)",
                                     "        W = (Wt[l].detach().transpose(-1, -2) if Wt[l].dim() == 5 else Wt[l].detach()).reshape(R, WIDTHS[l], -1).double()")]),
        ("H の腕で m_in を中心化前の入力から取る", [("        return out, (x, p1, h, i4)",
                                                    "        return out, (x, F.max_pool2d(a1, POOL, POOL), F.max_pool2d(a2, POOL, POOL).reshape(B, R, CN.FLAT).transpose(0, 1), a3)")]),
    ], "上界 (2^-10 + γ_{K+1})(⟨|W|, mean|p|⟩ + |b|): 入力と重みの TF32 の丸め（各 2^-11）と K 項の float32 の和", ENGINE),
    "K-graph": ("H・CH の束で CUDA graph と eager が 2 epoch で m・h_upd を含めて bit 一致", ck_graph, [], "bit 一致", ENGINE),
    "K-indep": ("run 1 の重み・入力・m を乱しても run 0・2 は bit で不変、run 1 は変わる", ck_indep, [], "bit 一致", ENGINE),
    "K-loop": ("run（eager）= 課題の手での再生、m は課題の境界を越えて引き継がれる、h_upd = 150/課題", ck_loop, [
        ("課題の境界で m をリセット（run）", [("        B.act.hupd.zero_()\n",
                                              "        B.act.hupd.zero_()\n        for _mm in B.act.m:\n            _mm.zero_()\n")]),
    ], "bit 一致", ENGINE),
    "K-resume": ("2 課題を通しで回すのと 1 課題で止めて再開するのとで per_task・t0・curves・units・状態が一致", ck_resume, [],
                 "文字列・bit 一致", ENGINE),
    "K-verdict": ("合成の per_task で設計どおりのラベル", ck_verdict, [
        ("k* の取り違え", [("        if binom_two_sided(k, n) < ALPHA:", "        if binom_two_sided(k, n) < 0.2:")]),
        ("0.5 の線を > に", [('"kept": W >= KEEP_LINE', '"kept": W > KEEP_LINE')]),
        ("床の √10 を √30 に", [("mf.std(ddof=1) / math.sqrt(len(mf))", "mf.std(ddof=1) / math.sqrt(30)")]),
        ("主窓の課題範囲", [("WINDOW = (21, 30)", "WINDOW = (11, 30)")]),
        ("TRANSFER の条件 4 を落とす", [('            and d_h_given_c == "HELPS"):', "            and True):")]),
        ("梯子の順序", [('for arm, name in (("C", "NEED_C"), ("H", "NEED_H"), ("CH", "NEED_CH")):',
                         'for arm, name in (("H", "NEED_H"), ("C", "NEED_C"), ("CH", "NEED_CH")):')]),
        ("LAYER の c1 の判定", [('h_c1 = int(sum("c1" in s.split(",") for s in vH.first_layers if s))',
                                'h_c1 = int(sum("c2" in s.split(",") for s in vH.first_layers if s))')]),
        ("(C) の死んだ層の免除を落とす", [("mask = ~((df[col] == steps_per_task) | (df[dcol] == 1.0))",
                                         "mask = ~(df[col] == steps_per_task)")]),
    ], "合成データの期待値（設計から）", VERDICT),
    "K-cost": ("CH の束（R=10）の 1 epoch の秒数（単独と主 4 腕の同時）と GPU メモリの峰（seed 100–109、精度は読まない）",
               ck_cost, [], "—", ENGINE),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-cost", action="store_true")
    args = ap.parse_args()
    SCR.mkdir(parents=True, exist_ok=True)
    if DUMP.exists():
        RESULTS.update(json.loads(DUMP.read_text()))
        RESULTS.pop("all_pass", None)
    keys = args.only.split(",") if args.only else list(CHECKS)
    if args.skip_cost and "K-cost" in keys:
        keys.remove("K-cost")
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "src", "analysis"],
                           capture_output=True, text=True).stdout.strip()
    RESULTS["_meta"] = {"git_hash": head, "git_dirty_code": bool(dirty), "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    for k in keys:
        title, fn, muts, deriv, target = CHECKS[k]
        run_check(k, title, fn, muts, deriv, target)
    dump()
    print("all_pass", RESULTS["all_pass"])


if __name__ == "__main__":
    main()

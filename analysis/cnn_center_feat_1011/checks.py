#!/usr/bin/env python3
"""Checks of cnn_center_feat_1011 (spec §6), each with mutation controls.

    python analysis/cnn_center_feat_1011/checks.py [--only K-same181,K-host] [--skip-cost] [--skip-reuse]

-> results/cnn_center_feat_1011/checks.json (scratch in results/_checks_cnn_center_feat_1011/).
Every check must pass on the real code and fail on each listed mutant; a mutant is the target
file with one exact-once string substitution (P181's `load`).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
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
from src import cnn_center_relu_1011 as CR            # noqa: E402  (P181's engine)
from src import cnn_center_1010 as CC                 # noqa: E402  (train_channel_mean, mean_patch)

ENGINE = REPO / "src" / "cnn_center_feat_1011.py"
VERDICT = REPO / "analysis" / "cnn_center_feat_1011" / "verdict.py"
COST = REPO / "analysis" / "cnn_center_feat_1011" / "cost.py"
OUT = REPO / "results" / "cnn_center_feat_1011"
SCR = REPO / "results" / "_checks_cnn_center_feat_1011"
P181 = REPO / "results" / "cnn_center_relu_1011"
P181_BACKUP = Path.home() / "Projects" / "obsidian-research-data" / "cnn_center_relu_1011"
P181_COMMIT = "b9ae5a26"
PY = sys.executable
LAYERS = ("c1", "c2", "f1", "f2")
BATCH = CN.BATCH
C_ = CN.CHANNELS
SIDE = (16, 8)
DEV = H.setup("cuda")
_di = torch.cuda.current_device()
torch.cuda.set_per_process_memory_fraction(
    min(1.0, 6.0 / (torch.cuda.get_device_properties(_di).total_memory / 2 ** 30)), _di)
CIFAR = RC.Cifar10()
RESULTS: dict = {}
DUMP = OUT / "checks.json"


# --------------------------------------------------------------------------
# infrastructure (P181's)
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
    entry["git_hash"] = RESULTS.get("_meta", {}).get("git_hash")
    entry["git_dirty_code"] = RESULTS.get("_meta", {}).get("git_dirty_code")
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


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------
# the spec's doors, written independently (bundled ops, so that they are bit-comparable)
# --------------------------------------------------------------------------

def indep_perm(side: int) -> tuple[torch.Tensor, int]:
    """Spec §3.1: randperm from stream ("cnn_center_feat_1011_shuffle", side), drawn again while it has
    a fixed point."""
    g = H.stream("cnn_center_feat_1011_shuffle", side)
    n = side * side
    k = 0
    while True:
        k += 1
        p = torch.randperm(n, generator=g)
        if int((p == torch.arange(n)).sum()) == 0:
            return p, k


def sub_indep(ml: torch.Tensor, R: int, side: int, mode: str, perm=None) -> torch.Tensor:
    """The conv map the next layer loses, (1, R*C, side or 1, side or 1): m (R, n) laid out as
    (R, C) for "chan" and (R, C, side, side) for "feat"; "shuf" moves position p to perm[p]."""
    ml = ml.detach()
    if mode == "chan":
        return ml.reshape(R, C_)[:, :, None, None].reshape(1, R * C_, 1, 1)
    mm = ml.reshape(R, C_, side, side)
    if mode == "shuf":
        mm = mm.reshape(R, C_, side * side)[:, :, perm].reshape(R, C_, side, side)
    return mm.reshape(1, R * C_, side, side)


def ref_forward_bundled(P, xb, m, mode, perm=None, with_inputs=False):
    """Spec §3.1 doors on the engine's bundled layout; m is a list of constants (mode None = no door)."""
    Wc1, bc1, Wc2, bc2, Wf1, bf1, Wf2, bf2, Wf3, bf3 = [p for p in P]
    R, B = xb.shape[:2]
    x = xb.transpose(0, 1).reshape(B, R * 3, 32, 32)
    z1 = F.conv2d(x, Wc1.reshape(R * C_, 3, 5, 5), bc1.reshape(-1), padding=2, groups=R)
    p1 = F.max_pool2d(torch.clamp(z1, min=0.0), 2, 2)
    if mode is not None:
        p1 = p1 - sub_indep(m[0], R, 16, mode, None if perm is None else perm[0])
    z2 = F.conv2d(p1, Wc2.reshape(R * C_, C_, 5, 5), bc2.reshape(-1), padding=2, groups=R)
    p2 = F.max_pool2d(torch.clamp(z2, min=0.0), 2, 2)
    if mode is not None:
        p2 = p2 - sub_indep(m[1], R, 8, mode, None if perm is None else perm[1])
    hh = p2.reshape(B, R, CN.FLAT).transpose(0, 1)
    z3 = torch.baddbmm(bf1[:, None, :], hh, Wf1.transpose(1, 2))
    a3 = torch.clamp(z3, min=0.0)
    i4 = a3 - m[2].detach()[:, None, :] if mode is not None else a3
    z4 = torch.baddbmm(bf2[:, None, :], i4, Wf2.transpose(1, 2))
    a4 = torch.clamp(z4, min=0.0)
    i5 = a4 - m[3].detach()[:, None, :] if mode is not None else a4
    logits = torch.baddbmm(bf3[:, None, :], i5, Wf3.transpose(1, 2))
    out = (z1, None, z2, None, z3, None, z4, None, logits)
    return (out, (x, p1, hh, i4)) if with_inputs else out


def bm_indep(z: torch.Tensor, l: int, R: int, mode: str) -> torch.Tensor:
    """Spec §3.1: the batch mean of the uncentred output the next layer reads."""
    a = torch.clamp(z, min=0.0)
    if l >= 2:
        return a.mean(1)
    pool = F.max_pool2d(a, 2, 2)
    if mode == "chan":
        return pool.mean((0, 2, 3)).view(R, -1)
    return pool.mean(0).view(R, -1)


def ref_door_forward_host(params, x, m, mode, perm=None):
    """Spec §3.1 for ONE run in the host's (unbundled) layout; m = 4 constants (conv: (C,) or (C*s*s,))."""
    Wc1, bc1, Wc2, bc2, Wf1, bf1, Wf2, bf2, Wf3, bf3 = params

    def conv_map(ml, side, k):
        if mode == "chan":
            return ml.view(1, -1, 1, 1)
        mm = ml.view(C_, side * side)
        if mode == "shuf":
            mm = mm[:, perm[k]]
        return mm.view(1, C_, side, side)

    z1 = F.conv2d(x, Wc1, bc1, padding=2)
    p1 = F.max_pool2d(torch.clamp(z1, min=0.0), 2, 2) - conv_map(m[0], 16, 0)
    z2 = F.conv2d(p1, Wc2, bc2, padding=2)
    p2 = F.max_pool2d(torch.clamp(z2, min=0.0), 2, 2) - conv_map(m[1], 8, 1)
    h = p2.flatten(1)
    a3 = torch.clamp(h @ Wf1.T + bf1, min=0.0) - m[2]
    a4 = torch.clamp(a3 @ Wf2.T + bf2, min=0.0) - m[3]
    return a4 @ Wf3.T + bf3


def mode_of(arm):
    return {"ref": None, "C": None, "H": "chan", "CH": "chan", "Hf": "feat", "CHf": "feat",
            "Hs": "shuf", "CHs": "shuf"}[arm]


# --------------------------------------------------------------------------
# K-same181, K-reuse
# --------------------------------------------------------------------------

def ck_same181(M) -> dict:
    """This engine's ref / C / H / CH = P181's engine (R = 3, 2 epochs eager): every state, V, m, h_upd,
    evaluate, switch_eval and the diagnostics' common columns and arrays."""
    seeds = (10, 11, 12)
    out = {}
    for arm in ("ref", "C", "H", "CH"):
        A = M.Bundle([(arm, s) for s in seeds], CIFAR, DEV, graph=False)
        B = CR.Bundle([(arm, s) for s in seeds], CIFAR, DEV, graph=False)
        for Z in (A, B):
            Z.new_labels()
            for _ in range(2):
                Z.run_epoch()
        st = all(bits_equal(a, b) for a, b in zip(A._state_tensors(), B._state_tensors()))
        act = all(bits_equal(a, b) for a, b in zip((*A.act.V, *A.act.m, A.act.hupd), (*B.act.V, *B.act.m, B.act.hupd)))
        ra, rb = M.evaluate(A.P, A.X, A.Y, A.act), CR.evaluate(B.P, B.X, B.Y, B.act)
        ev = all(same_val(ra[r][k], rb[r][k]) for r in range(3) for k in rb[r])
        sa, sb = M.switch_eval(A.P, A.X, A.Y, A.act), CR.switch_eval(B.P, B.X, B.Y, B.act)
        sw = all(bits_equal(x, y) for x, y in zip(sa, sb))
        da, ua = M.diagnostics(A.P, A.X, A.Y, A.act)
        db, ub = CR.diagnostics(B.P, B.X, B.Y, B.act)
        dg = all(same_val(da[r][k], db[r][k]) for r in range(3) for k in db[r]) and \
            all(np.array_equal(ua[k], ub[k], equal_nan=True) for k in ub)
        out[arm] = {"state": st, "act": act, "evaluate": ev, "switch": sw, "diagnostics": dg,
                    "ok": st and act and ev and sw and dg}
    return {**{f"{a}_{k}": v for a, r in out.items() for k, v in r.items() if k != "ok"},
            "pass": all(r["ok"] for r in out.values())}


def ck_reuse(M) -> dict:
    """The CLI with ref / C / H / CH, R = 10, seeds 10-19, 1 task (CUDA graph) reproduces P181's
    per_task task-1 rows, t0.csv, curves[:, 0] and every seed's task-1 checkpoint (from the backup,
    sha256 checked against P181's backup_manifest.json); P181's per_task.csv files are b9ae5a26's blobs."""
    man = json.loads((P181 / "backup_manifest.json").read_text())
    man_sha = {r["source"]: r["sha256"] for r in man["files"]}
    res = {}
    for arm in ("ref", "C", "H", "CH"):
        rel_csv = f"results/cnn_center_relu_1011/{arm}/per_task.csv"
        want = subprocess.run(["git", "-C", str(REPO), "rev-parse", f"{P181_COMMIT}:{rel_csv}"],
                              capture_output=True, text=True).stdout.strip()
        have = subprocess.run(["git", "-C", str(REPO), "hash-object", str(REPO / rel_csv)],
                              capture_output=True, text=True).stdout.strip()
        tmp = SCR / f"reuse_{arm}"
        shutil.rmtree(tmp, ignore_errors=True)
        cmd = [PY, str(ENGINE), "run", "--arm", arm, "--seeds", "10-19", "--tasks", "1", "--out", str(tmp),
               "--no-resume"]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        if p.returncode != 0:
            res[arm] = {"returncode": p.returncode, "stderr": p.stderr[-1500:], "ok": False}
            continue
        r = {"blob_is_b9ae5a26": bool(want) and want == have}
        for name, old_df in (("per_task", pd.read_csv(P181 / arm / "per_task.csv", dtype=str, keep_default_na=False)),
                             ("t0", pd.read_csv(P181 / arm / "t0.csv", dtype=str, keep_default_na=False))):
            new = pd.read_csv(tmp / f"{name}.csv", dtype=str, keep_default_na=False)
            old = old_df[old_df.task == "1"].reset_index(drop=True) if name == "per_task" else old_df
            cols = [c for c in old.columns if c in new.columns]
            r[f"{name}_n_rows"] = (len(new), len(old))
            r[f"{name}_missing"] = [c for c in old.columns if c not in new.columns]
            r[f"{name}_differing"] = [c for c in cols if new[c].tolist() != old[c].tolist()]
            r[f"{name}_n_common"] = len(cols)
        cn, co = np.load(tmp / "curves.npy"), np.load(P181 / arm / "curves.npy")[:, :1]
        r["curves_bit"] = cn.shape == co.shape and bool((cn.view(np.int32) == co.view(np.int32)).all())
        ck_ok, ck_sha_ok, n_ck = True, True, 0
        for s in range(10, 20):
            src_rel = f"results/cnn_center_relu_1011/{arm}/ckpt/{arm}_seed{s}_t01.pt"
            f_old = P181_BACKUP / src_rel
            if not f_old.exists() or sha256(f_old) != man_sha.get(src_rel):
                ck_sha_ok = False
                continue
            a = torch.load(tmp / "ckpt" / f"{arm}_seed{s}_t01.pt", map_location="cpu", weights_only=False)
            b = torch.load(f_old, map_location="cpu", weights_only=False)
            n_ck += 1
            ck_ok &= all(bits_equal(x, y) for k in ("P", "m", "v") for x, y in zip(a[k], b[k]))
            ck_ok &= all(bits_equal(x, y) for k in ("V", "m", "hupd") for x, y in zip(a["act"][k], b["act"][k]))
            ck_ok &= a["tc"] == b["tc"]
        r |= {"ckpt_bit": ck_ok and n_ck == 10, "ckpt_backup_sha_ok": ck_sha_ok, "n_ckpt": n_ck}
        r["ok"] = (r["blob_is_b9ae5a26"] and not r["per_task_missing"] and not r["per_task_differing"]
                   and not r["t0_missing"] and not r["t0_differing"] and r["per_task_n_rows"] == (10, 10)
                   and r["per_task_n_common"] >= 91 and r["curves_bit"] and r["ckpt_bit"] and r["ckpt_backup_sha_ok"])
        res[arm] = r
    return {**{f"{a}_{k}": v for a, r in res.items() for k, v in r.items() if k != "ok"},
            "pass": len(res) == 4 and all(r["ok"] for r in res.values())}


# --------------------------------------------------------------------------
# door H': per (channel x position), EMA, gradient, evaluation, first update, read-only m = 0
# --------------------------------------------------------------------------

def _block_perm(z: torch.Tensor, sigma: torch.Tensor, side: int) -> torch.Tensor:
    """Move the 2x2 pool windows of z (B, RC, 2s, 2s): window q of the result is window sigma[q] of z."""
    B, RC = z.shape[:2]
    w = z.reshape(B, RC, side, 2, side, 2).permute(0, 1, 2, 4, 3, 5).reshape(B, RC, side * side, 2, 2)
    w = w[:, :, sigma]
    return w.reshape(B, RC, side, side, 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(B, RC, 2 * side, 2 * side)


def ck_hf_pos(M) -> dict:
    R = 2
    Bf = M.Bundle([("Hf", 10), ("Hf", 11)], CIFAR, DEV, graph=False)
    Bc = M.Bundle([("H", 10), ("H", 11)], CIFAR, DEV, graph=False)
    for Z in (Bf, Bc):
        Z.new_labels()
        Z.new_order()
    idx = Bf.order[:, :BATCH]
    xb = Bf.X[Bf.ar, idx]
    with torch.no_grad():
        out = ref_forward_bundled(Bf.P, xb, [t.clone() for t in Bf.act.m], "feat")
    # (a) bm per (channel, position): shape and value
    a_ok = True
    for l in (0, 1):
        got = Bf.act.out_mean(out[2 * l], l)
        a_ok &= tuple(got.shape) == (R, C_ * SIDE[l] ** 2) and bits_equal(got, bm_indep(out[2 * l], l, R, "feat"))
    # (b) the next layer's inputs = P - m laid out (R, C, s, s)
    g = torch.Generator(device="cpu").manual_seed(3)
    mr = [torch.rand(R, C_ * SIDE[0] ** 2, generator=g).to(DEV), torch.rand(R, C_ * SIDE[1] ** 2, generator=g).to(DEV),
          torch.rand(R, 100, generator=g).to(DEV), torch.rand(R, 100, generator=g).to(DEV)]
    with torch.no_grad():
        (_, ins_e) = M.forward(Bf.P, xb, Bf.act, m=mr, with_inputs=True)
        (_, ins_i) = ref_forward_bundled(Bf.P, xb, mr, "feat", with_inputs=True)
    b_ok = bits_equal(ins_e[1], ins_i[1]) and bits_equal(ins_e[2], ins_i[2]) and bits_equal(ins_e[3], ins_i[3])
    # (c) equivariance: moving the pool windows of z moves H''s bm the same way; P181's H bm stays
    gs = torch.Generator(device="cpu").manual_seed(5)
    c_ok, c_detail = True, {}
    for l, side in ((0, 16), (1, 8)):
        z = out[2 * l]
        sigma = torch.randperm(side * side, generator=gs).to(DEV)
        zp = _block_perm(z, sigma, side)
        bf, bfp = Bf.act.out_mean(z, l), Bf.act.out_mean(zp, l)
        moved = bits_equal(bfp.view(R, C_, -1), bf.view(R, C_, -1)[:, :, sigma].contiguous())
        changed = not bits_equal(bfp, bf)
        bc, bcp = Bc.act.out_mean(z, l), Bc.act.out_mean(zp, l)
        inv = float((bcp.double() - bc.double()).abs().max() / bc.double().abs().max().clamp_min(1e-30))
        c_detail[f"l{l}"] = {"moved_bitwise": moved, "changed": changed, "H_rel_change": inv}
        c_ok &= moved and changed and inv <= 1e-6
    # (d) after 3 updates m differs between positions inside a channel
    for j in range(3):
        Bf.step(j)
    d_ok = all(float(Bf.act.m[l].view(R, C_, -1).std(2).max()) > 0 for l in (0, 1))
    return {"a_bm_per_channel_position": a_ok, "b_inputs_P_minus_m": b_ok, "c_equivariance": c_ok,
            "c_detail": c_detail, "d_m_varies_with_position": d_ok, "pass": a_ok and b_ok and c_ok and d_ok}


def ck_hf_ema(M) -> dict:
    out = {}
    for arm in ("Hf", "Hs"):
        B = M.Bundle([(arm, 10), (arm, 11)], CIFAR, DEV, graph=False)
        B.new_labels()
        B.new_order()
        steps, ok_all, moved = 6, True, 0
        perm = None if B.act.perm is None else [p.clone() for p in B.act.perm]
        for j in range(steps):
            Pb = [p.detach().clone() for p in B.P]
            mb = [t.clone() for t in B.act.m]
            idx = B.order[:, j * BATCH:(j + 1) * BATCH]
            with torch.no_grad():
                o = ref_forward_bundled(Pb, B.X[B.ar, idx], mb, mode_of(arm), perm)
            exp = [mb[l].clone().mul_(1 - 0.01).add_(bm_indep(o[2 * l], l, 2, "feat"), alpha=0.01) for l in range(4)]
            B.step(j)
            ok_all &= all(bits_equal(B.act.m[l], exp[l]) for l in range(4))
            moved += all(bool((B.act.m[l] != mb[l]).any()) for l in range(4))
        mb = [t.clone() for t in B.act.m]
        B.new_labels()
        boundary = all(bits_equal(a, b) for a, b in zip(mb, B.act.m))
        hupd = bool((B.act.hupd == steps).all())
        out[arm] = {"ema_bit_identical_every_step": ok_all, "m_moved_every_step": moved == steps,
                    "new_labels_keeps_m": boundary, "h_upd_counts_steps": hupd,
                    "ok": ok_all and moved == steps and boundary and hupd}
    return {**{f"{a}_{k}": v for a, r in out.items() for k, v in r.items() if k != "ok"},
            "pass": all(r["ok"] for r in out.values())}


def ck_hf_grad(M) -> dict:
    B = M.Bundle([("Hf", 10), ("Hf", 11)], CIFAR, DEV, graph=False)
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
    g_ref = grads(lambda P, x: ref_forward_bundled(P, x, mc, "feat"))
    eq = all(bits_equal(a, b) for a, b in zip(g_mod, g_ref))
    return {"m_requires_no_grad": no_grad, "grads_bit_identical_to_constant_m": eq, "pass": no_grad and eq}


def _ce_sum(logits, yb, R):
    b = yb.shape[1]
    return F.cross_entropy(logits.reshape(-1, 10), yb.reshape(-1), reduction="none").view(R, b).double().sum(1)


def ck_hf_eval(M) -> dict:
    B = M.Bundle([("Hf", 10), ("Hf", 11)], CIFAR, DEV, graph=False)
    B.new_labels()
    B.run_epoch()
    rows = M.evaluate(B.P, B.X, B.Y, B.act)

    def fit_ce(mm):
        ce = torch.zeros(2, dtype=torch.float64, device=DEV)
        with torch.no_grad():
            for i0 in range(0, CN.N_IMAGES, 20):
                ce += _ce_sum(ref_forward_bundled(B.P, B.X[:, i0:i0 + 20], mm, "feat")[8], B.Y[:, i0:i0 + 20], 2)
        return ce / CN.N_IMAGES

    ref_m = fit_ce(B.act.m)
    ref_0 = fit_ce([torch.zeros_like(t) for t in B.act.m])
    ok = all(rows[r]["fit_ce"] == float(ref_m[r]) for r in range(2))
    differs = all(rows[r]["fit_ce"] != float(ref_0[r]) for r in range(2))
    sce, _ = M.switch_eval(B.P, B.X, B.Y, B.act)
    sw_ok = all(float(sce[r]) == float(ref_m[r]) for r in range(2))   # same labels: switch CE = fit CE
    return {"fit_ce_uses_current_m": ok, "fit_ce_differs_without_m": differs, "switch_eval_uses_current_m": sw_ok,
            "pass": ok and differs and sw_ok}


def ck_hf_first(M) -> dict:
    out = {}
    for door, base in (("Hf", "ref"), ("CHf", "C"), ("Hs", "ref")):
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
    B = M.Bundle([("Hf", 10), ("Hf", 11)], CIFAR, DEV, graph=False)
    B.new_labels()
    B.run_epoch()
    t0, g0 = _snapshot(B)
    rows, _ = M.diagnostics(B.P, B.X, B.Y, B.act, m0_eval=True)
    t1, g1 = _snapshot(B)
    ro = all(bits_equal(a, b) for a, b in zip(t0, t1)) and all(
        bool((a[0] == b[0]).all() and (a[1] == b[1]).all()) for a, b in zip(g0, g1))
    has = all("memo_acc_m0" in r for r in rows)
    B1 = M.Bundle([("Hf", 10), ("Hf", 11)], CIFAR, DEV, graph=False)
    B2 = M.Bundle([("Hf", 10), ("Hf", 11)], CIFAR, DEV, graph=False)
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


# --------------------------------------------------------------------------
# K-shuf, K-host, K-pdc
# --------------------------------------------------------------------------

def ck_shuf(M) -> dict:
    # (i) the derangement: the engine's function = an independent implementation, for the registered
    # sides and for small sides where a first draw with a fixed point is likely
    perm_ok, draws = True, {}
    for side in (2, 3, 4, 8, 16):
        pe, de = M.shuffle_perm(side)
        pi, di = indep_perm(side)
        perm_ok &= bool((pe == pi).all()) and de == di and int((pe == torch.arange(side * side)).sum()) == 0
        draws[side] = de
    B = M.Bundle([("Hs", 10), ("Hs", 11)], CIFAR, DEV, graph=False)
    eng_perm = all(bool((B.act.perm[k].cpu() == indep_perm(s)[0]).all()) for k, s in enumerate(SIDE))
    B.new_labels()
    B.new_order()
    for j in range(3):
        B.step(j)
    idx = B.order[:, 3 * BATCH:4 * BATCH]
    xb = B.X[B.ar, idx]
    perm = [indep_perm(s)[0].to(DEV) for s in SIDE]
    with torch.no_grad():
        (_, ins_e) = M.forward(B.P, xb, B.act, with_inputs=True)
        (_, ins_i) = ref_forward_bundled(B.P, xb, B.act.m, "shuf", perm, with_inputs=True)
    sub_ok = bits_equal(ins_e[1], ins_i[1]) and bits_equal(ins_e[2], ins_i[2])
    # the EMA tracks the unpermuted per-position mean (the same rule as H')
    with torch.no_grad():
        o = ref_forward_bundled(B.P, xb, B.act.m, "shuf", perm)
    ema_ok = all(bits_equal(B.act.out_mean(o[2 * l], l), bm_indep(o[2 * l], l, 2, "feat")) for l in range(4))
    # the shuffled map keeps every channel's mean (float64, sum order)
    chan_ok = True
    for k, s in enumerate(SIDE):
        mp = B.act.sub_map(k, B.act.m).reshape(2, C_, -1).double().mean(2)
        mm = B.act.m[k].view(2, C_, -1).double().mean(2)
        chan_ok &= float((mp - mm).abs().max() / mm.abs().max().clamp_min(1e-30)) <= 1e-6
    # it does not subtract the position's own mean
    not_own = not bits_equal(ins_e[1], ref_forward_bundled(B.P, xb, B.act.m, "feat", with_inputs=True)[1][1])
    return {"derangement_matches_spec": perm_ok, "draws": draws, "engine_perm_matches": eng_perm,
            "next_layer_reads_P_minus_m_pi": sub_ok, "ema_unpermuted": ema_ok, "channel_means_kept": chan_ok,
            "differs_from_Hf_map": not_own, "pass": perm_ok and eng_perm and sub_ok and ema_ok and chan_ok and not_own}


def ck_host(M) -> dict:
    res = {}
    mu = CC.train_channel_mean(CIFAR).to(DEV)
    perm = [indep_perm(s)[0].to(DEV) for s in SIDE]
    for arm in ("Hf", "CHf", "Hs"):
        B = M.Bundle([(arm, 10)], CIFAR, DEV, graph=False)
        B.new_labels()
        B.new_order()
        params = CN.init_params(10, DEV)
        xh = CN.images(CIFAR, CN.subset_idx(10), DEV)
        if arm == "CHf":
            xh = xh - mu.view(1, 3, 1, 1)
        idx = B.order[0, :BATCH]
        # realistic non-zero door means: the uncentred output means on 64 images at init
        with torch.no_grad():
            out = M.forward(B.P, B.X[:, :64], B.act)
            for l in range(4):
                B.act.m[l].copy_(B.act.out_mean(out[2 * l], l))
        mv = [t[0].clone() for t in B.act.m]
        lh = F.cross_entropy(ref_door_forward_host(params, xh[idx], mv, mode_of(arm), perm), B.Y[0, idx])
        gh = torch.autograd.grad(lh, params)
        lb = F.cross_entropy(M.forward(B.P, B.X[:, idx], B.act)[8][0], B.Y[0, idx])
        gb = torch.autograd.grad(lb, B.P)
        lr_ = abs(float(lb.detach()) - float(lh.detach())) / abs(float(lh.detach()))
        gr_ = max(rel(a[0], b) for a, b in zip(gb, gh))
        res[arm] = {"loss_rel": lr_, "grad_rel_max": gr_, "pass": lr_ < 1e-5 and gr_ < 1e-4}
    return {**{f"{a}_{k}": v for a, r in res.items() for k, v in r.items() if k != "pass"},
            "pass": all(r["pass"] for r in res.values())}


def _pdc_indep(P, X, m, mode, perm=None):
    """Independent float64 recomputation of spec §4's pdc / cdc / dc / prms for the inputs of c2, f1."""
    R = X.shape[0]
    PS = [torch.zeros(R, C_ * s * s, dtype=torch.float64, device=DEV) for s in SIDE]
    QS = [torch.zeros(R, C_ * s * s, dtype=torch.float64, device=DEV) for s in SIDE]
    PQ = [torch.zeros(R, dtype=torch.float64, device=DEV) for _ in SIDE]
    mx = [0.0, 0.0]
    n = 0
    with torch.no_grad():
        for i0 in range(0, CN.N_IMAGES, 40):
            xb = X[:, i0:i0 + 40]
            b = xb.shape[1]
            out, ins = ref_forward_bundled(P, xb, m, mode, perm, with_inputs=True)
            pre = [F.max_pool2d(torch.clamp(out[0], min=0.0), 2, 2), F.max_pool2d(torch.clamp(out[2], min=0.0), 2, 2)]
            post = [ins[1], ins[2].transpose(0, 1).reshape(b, R * C_, 8, 8)]
            for k in (0, 1):
                a = pre[k].reshape(b, R, -1).double()
                PS[k] += a.sum(0)
                PQ[k] += (a * a).sum((0, 2))
                QS[k] += post[k].reshape(b, R, -1).double().sum(0)
                mx[k] = max(mx[k], float(pre[k].abs().max()))
            n += b
    res = {}
    for k, tag in ((0, "c2"), (1, "f1")):
        s = SIDE[k]
        mpre, mpost = PS[k] / n, QS[k] / n
        sig = (PQ[k] / n - (mpre * mpre).sum(1)).clamp_min(0).sqrt()

        def parts(mu):
            mc = mu.reshape(R, C_, s * s)
            cb = mc.mean(2)
            return (mc - cb[:, :, None]).reshape(R, -1).norm(dim=1), cb.norm(dim=1) * s, mu.norm(dim=1)

        a, b_, c = parts(mpre)
        d, e, f = parts(mpost)
        res[tag] = {"prms": sig, "pdc_pre": a / sig, "cdc_pre": b_ / sig, "dc_pre": c / sig,
                    "pdc_post": d / sig, "cdc_post": e / sig, "dc_post": f / sig,
                    "mu_pre": mpre, "mu_post": mpost, "maxP": mx[k]}
    return res


def ck_pdc(M) -> dict:
    out = {}
    perm = [indep_perm(s)[0].to(DEV) for s in SIDE]
    for arm in ("Hf", "H", "Hs", "ref"):
        B = M.Bundle([(arm, 10), (arm, 11)], CIFAR, DEV, graph=False)
        B.new_labels()
        B.run_epoch()
        rows, units = M.diagnostics(B.P, B.X, B.Y, B.act)
        ind = _pdc_indep(B.P, B.X, B.act.m, mode_of(arm), perm)
        worst = 0.0
        for tag in ("c2", "f1"):
            for q in ("prms", "pdc_pre", "cdc_pre", "dc_pre", "pdc_post", "cdc_post", "dc_post"):
                for r in range(2):
                    v, w = rows[r][f"{q}_{tag}"], float(ind[tag][q][r])
                    worst = max(worst, abs(v - w) / max(abs(w), 1e-30))
        # mu_post = mu_pre - the subtracted map (float64, 1e-6 of the largest mean)
        mp_ok = True
        k_of = {"c2": 0, "f1": 1}
        for tag in ("c2", "f1"):
            k = k_of[tag]
            if mode_of(arm) is None:
                sub = torch.zeros_like(ind[tag]["mu_pre"])
            else:
                s = SIDE[k]
                sub = sub_indep(B.act.m[k], 2, s, mode_of(arm), perm[k]).reshape(1, 2, C_, -1 if mode_of(arm) != "chan" else 1)
                sub = sub.expand(1, 2, C_, s * s).reshape(2, -1).double()
            dd = (ind[tag]["mu_pre"] - sub - ind[tag]["mu_post"]).abs().max()
            mp_ok &= float(dd) <= 1e-6 * float(ind[tag]["mu_pre"].abs().max())
            mp_ok &= np.allclose(units[f"pmean_post_{tag}"], ind[tag]["mu_post"].float().cpu().numpy(), rtol=0, atol=1e-6 * float(ind[tag]["mu_pre"].abs().max()))
        same_h = True
        if arm in ("H", "ref"):
            for tag in ("c2", "f1"):
                for r in range(2):
                    bound = 3 * 2.0 ** -24 * ind[tag]["maxP"] * math.sqrt(C_ * SIDE[k_of[tag]] ** 2) / float(ind[tag]["prms"][r])
                    same_h &= abs(rows[r][f"pdc_post_{tag}"] - rows[r][f"pdc_pre_{tag}"]) <= bound
        out[arm] = {"max_rel_diff": worst, "mu_post_is_pre_minus_map": mp_ok, "H_ref_post_eq_pre": same_h,
                    "ok": worst <= 1e-9 and mp_ok and same_h}
    # H' with m = the exact 1200-image mean: the per-position part of f1's input vanishes to rounding
    B = M.Bundle([("Hf", 10), ("Hf", 11)], CIFAR, DEV, graph=False)
    B.new_labels()
    B.run_epoch()
    with torch.no_grad():
        for k, tag in ((0, "c2"), (1, "f1")):      # c2's input does not depend on m; f1's depends on m[0] only
            ind = _pdc_indep(B.P, B.X, B.act.m, "feat")
            B.act.m[k].copy_(ind[tag]["mu_pre"].float())
    ind = _pdc_indep(B.P, B.X, B.act.m, "feat")
    rows, _ = M.diagnostics(B.P, B.X, B.Y, B.act)
    exact_ok, ratio = True, []
    for k, tag in ((0, "c2"), (1, "f1")):
        for r in range(2):
            bound = 3 * 2.0 ** -24 * ind[tag]["maxP"] * math.sqrt(C_ * SIDE[k] ** 2) / rows[r][f"prms_{tag}"]
            ratio.append(rows[r][f"pdc_post_{tag}"] / bound)
            exact_ok &= rows[r][f"pdc_post_{tag}"] <= bound
    return {**{f"{a}_{k}": v for a, r in out.items() for k, v in r.items() if k != "ok"},
            "exact_mean_post_within_bound": exact_ok, "exact_mean_post_over_bound_max": max(ratio),
            "pass": all(r["ok"] for r in out.values()) and exact_ok}


# --------------------------------------------------------------------------
# K-ident, K-graph, K-indep, K-loop, K-resume
# --------------------------------------------------------------------------

def ck_ident(M) -> dict:
    out = {}
    for arm in ("Hf", "CHf", "Hs"):
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
    for arm in ("Hf", "CHf", "Hs"):
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
    B = M.Bundle([("CHf", s) for s in (10, 11, 12)], CIFAR, DEV, graph=False)
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
    tmp = SCR / "loop_Hf"
    shutil.rmtree(tmp, ignore_errors=True)
    M.run("Hf", [10, 11], tmp, 2, 2, DEV, ckpt_tasks=(), graph=False, resume=False, log=quiet)
    st = torch.load(tmp / "state.pt", map_location="cpu", weights_only=False)
    B = M.Bundle([("Hf", 10), ("Hf", 11)], CIFAR, DEV, epochs=2, graph=False)
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
    units_ok = list(u["tasks_done"]) == [0, 1, 2] and (tmp / "t0.csv").exists() and \
        u["mdoor_c1"].shape == (3, 2, 4096) and u["pmean_pre_f1"].shape == (3, 2, 1024)
    return {"params_equal_manual": eqP, "door_m_equal_manual": eqm, "h_upd_150_each_task": hupd,
            "units_and_t0": units_ok, "pass": eqP and eqm and hupd and units_ok}


def ck_resume(M) -> dict:
    a, b = SCR / "resume_A", SCR / "resume_B"
    for d in (a, b):
        shutil.rmtree(d, ignore_errors=True)
    M.run("CHf", [10, 11], a, 2, 2, DEV, ckpt_tasks=(1, 2), graph=True, resume=False, log=quiet)

    class Stop(Exception):
        pass

    def stopper(msg):
        if "task  1 " in msg:
            raise Stop()

    try:
        M.run("CHf", [10, 11], b, 2, 2, DEV, ckpt_tasks=(1, 2), graph=True, resume=False, log=stopper)
    except Stop:
        pass
    stopped = len(pd.read_csv(b / "per_task.csv")) == 2
    M.run("CHf", [10, 11], b, 2, 2, DEV, ckpt_tasks=(1, 2), graph=True, resume=True, log=quiet)
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


def synth_arm(arm, online_fn, mob_fn=None, dead_fn=None, hupd_fn=None, memo_m0=0.2, seeds=range(10, 20)):
    rows = []
    for s in seeds:
        for t in range(1, 31):
            on = online_fn(s, t)
            r = {"arm": arm, "seed": s, "task": t, "online_acc": float(on), "memo_acc": 1.0 if on > 0.3 else 0.12,
                 "diverged": False, "major_frac": _mf(s, t), "logit_sd": 15.0}
            for l in LAYERS:
                r[f"mob_{l}"] = 0.3 if mob_fn is None else mob_fn(s, t, l)
                r[f"dead_frac_{l}"] = 0.0 if dead_fn is None else dead_fn(s, t, l)
                if arm in ("H", "CH", "Hf", "CHf", "Hs", "CHs"):
                    r[f"h_upd_{l}"] = 30000 if hupd_fn is None else hupd_fn(s, t, l)
            if arm in ("H", "CH", "Hf", "CHf", "Hs", "CHs"):
                r["memo_acc_m0"] = memo_m0
            rows.append(r)
    return pd.DataFrame(rows)


def _floors(seed, sqrt_n):
    mf = np.array([_mf(seed, t) for t in range(21, 31)])
    return mf.mean() + 3 * mf.std(ddof=1) / math.sqrt(sqrt_n)


def _collapse(at):
    """online: 0.95 until task `at` - 1, then 0.100 (well below every floor)."""
    return lambda s, t: 0.95 if t < at else 0.100


def _hold(level=0.95):
    return lambda s, t: level


def _ctrl(arm, fn):
    return synth_arm(arm, fn, seeds=(10,))


def scenarios():
    """(name, frames, ctrl_frames, checks_ok, c_door_ok, expected dict)."""
    out = []
    base = {"ref": synth_arm("ref", _collapse(3)), "C": synth_arm("C", _collapse(5)),
            "H": synth_arm("H", _collapse(2)), "CH": synth_arm("CH", _collapse(11)),
            "Hf": synth_arm("Hf", _collapse(4)), "CHf": synth_arm("CHf", _hold(0.96))}
    # Hs falls with H (t2) while Hf lasts to t3: W1_5 separates (H 0.27, Hf 0.61)
    ctrl = {"Hs": _ctrl("Hs", _collapse(2)), "CHs": None}
    out.append(("A_suffices", base, ctrl, True, True,
                {"CHf": "RESCUED", "Hf": "FAIL", "FEATURE_DOOR": "FEATURE_CENTERING_SUFFICES", "M_Hf": "HF_LIKE_MLP",
                 "Hs": "SHUFFLE_ACTS_LIKE_H", "CHs": "NOT_RUN", "T_CHf_CH": "LATER", "T_CHf_ref": "LATER",
                 "T_Hf_H": "LATER", "RESIDUAL_CHf": False, "D_CHf|CH": "HELPS"}))
    # CHf FAIL and Hf FAIL -> NOT_A_DOOR; CHf T_f equal to CH -> NO_TIMING_DIFF; Hs indistinguishable
    f = dict(base, CHf=synth_arm("CHf", _collapse(11)), Hf=synth_arm("Hf", _collapse(2)))
    out.append(("B_not_a_door", f, {"Hs": _ctrl("Hs", _collapse(2)), "CHs": None}, True, True,
                {"CHf": "FAIL", "Hf": "FAIL", "FEATURE_DOOR": "CENTERING_NOT_A_DOOR_IN_CNN",
                 "T_CHf_CH": "NO_TIMING_DIFF", "T_Hf_H": "NO_TIMING_DIFF", "Hs": "SHUF_UNINFORMATIVE",
                 "RESIDUAL_CHf": True}))
    # CHf FAIL but Hf RESCUED -> PARTIAL (not NOT_A_DOOR); Hf differs from the MLP
    f = dict(base, CHf=synth_arm("CHf", _collapse(11)), Hf=synth_arm("Hf", _hold(0.96)))
    out.append(("C_Hf_rescues_CHf_fails", f, ctrl, True, True,
                {"FEATURE_DOOR": "FEATURE_CENTERING_PARTIAL", "M_Hf": "HF_DIFFERS_FROM_MLP"}))
    # CHf kept in 8/10 -> PARTIAL; 9/10 with one seed exactly at the line -> RESCUED
    f = dict(base, CHf=synth_arm("CHf", lambda s, t: 0.95 if s < 18 else 0.100))
    out.append(("D_CHf_8of10", f, ctrl, True, True, {"CHf": "PARTIAL", "FEATURE_DOOR": "FEATURE_CENTERING_PARTIAL"}))
    f = dict(base, CHf=synth_arm("CHf", lambda s, t: 0.95 if s < 18 else (0.5 if (s == 18 and t > 20) else (0.95 if s == 18 else 0.100))))
    out.append(("E_CHf_9of10_at_line", f, ctrl, True, True, {"CHf": "RESCUED", "FEATURE_DOOR": "FEATURE_CENTERING_SUFFICES"}))
    # checks fail -> INAPPLICABLE; CH (P181) not FAIL -> NOT_REPRODUCED; ref not FAIL -> NOT_REPRODUCED
    out.append(("F_checks_fail", base, ctrl, False, True, {"FEATURE_DOOR": "INAPPLICABLE"}))
    out.append(("G_CH_alive", dict(base, CH=synth_arm("CH", _hold(0.9))), ctrl, True, True,
                {"FEATURE_DOOR": "NOT_REPRODUCED"}))
    out.append(("G2_ref_alive", dict(base, ref=synth_arm("ref", _hold(0.9))), ctrl, True, True,
                {"FEATURE_DOOR": "NOT_REPRODUCED"}))
    # (C'): h_upd 0 in task 1 -> INAPPLICABLE; h_upd short only after task 1 (a dying layer) -> exempt
    f = dict(base, CHf=synth_arm("CHf", _hold(0.96), hupd_fn=lambda s, t, l: 0 if (l == "f2" and t == 1 and s == 13) else 30000))
    out.append(("H_door_never_moved", f, ctrl, True, True, {"CHf": "INAPPLICABLE", "FEATURE_DOOR": "INAPPLICABLE"}))
    f = dict(base, CHf=synth_arm("CHf", _hold(0.96), hupd_fn=lambda s, t, l: 0 if (l == "f1" and t >= 8) else 30000,
                                 dead_fn=lambda s, t, l: 0.98 if (l == "f1" and t >= 8) else 0.0))
    out.append(("I_late_fixed_point_exempt", f, ctrl, True, True, {"CHf": "RESCUED", "FEATURE_DOOR": "FEATURE_CENTERING_SUFFICES"}))
    # the C door of CHf failed -> INAPPLICABLE
    out.append(("J_c_door", base, ctrl, True, False, {"CHf": "INAPPLICABLE", "FEATURE_DOOR": "INAPPLICABLE"}))
    # main window: CHf high in t11-20 and low in t21-30 -> FAIL
    f = dict(base, CHf=synth_arm("CHf", lambda s, t: 0.95 if t <= 20 else 0.100))
    out.append(("K_window", f, ctrl, True, True, {"CHf": "FAIL", "FEATURE_DOOR": "CENTERING_NOT_A_DOOR_IN_CNN",
                                                  "T_CHf_CH": "LATER"}))
    # Hs on Hf's side -> ACTS_LIKE_HF; Hs between -> UNRESOLVED; Hs invalid -> INAPPLICABLE
    out.append(("L_hs_like_hf", base, {"Hs": _ctrl("Hs", _collapse(4)), "CHs": None}, True, True,
                {"Hs": "SHUFFLE_ACTS_LIKE_HF"}))
    out.append(("M_hs_between", base, {"Hs": _ctrl("Hs", lambda s, t: 0.95 if t < 3 else (0.3 if t == 3 else 0.1)), "CHs": None},
                True, True, {"Hs": "SHUF_UNRESOLVED"}))
    bad = _ctrl("Hs", _collapse(2))
    bad["diverged"] = True
    out.append(("N_hs_invalid", base, {"Hs": bad, "CHs": None}, True, True, {"Hs": "INAPPLICABLE"}))
    # CHs run: on CH's side (falls at t11 like CH while CHf holds) -> ACTS_LIKE_CH; on CHf's side -> ACTS_LIKE_CHF
    out.append(("O_chs_like_ch", base, {"Hs": ctrl["Hs"], "CHs": _ctrl("CHs", _collapse(11))}, True, True,
                {"CHs": "SHUFFLE_ACTS_LIKE_CH"}))
    out.append(("P_chs_like_chf", base, {"Hs": ctrl["Hs"], "CHs": _ctrl("CHs", _hold(0.96))}, True, True,
                {"CHs": "SHUFFLE_ACTS_LIKE_CHF"}))
    # timing: CHf earlier than CH -> EARLIER; T1_LOWER and EVALOFF; residual loss off for a flat CHf
    f = dict(base, CHf=synth_arm("CHf", lambda s, t: 0.90 if t == 1 else (0.95 if t < 6 else 0.1), memo_m0=1.0))
    out.append(("Q_earlier_t1lower", f, ctrl, True, True, {"T_CHf_CH": "EARLIER", "T1_LOWER_CHf": True,
                                                            "EVALOFF_CHf": "EVALOFF_NO_DETECTED_DEPENDENCE"}))
    # FAIL subtype: two CHf seeds between the sqrt(30) and sqrt(10) floors -> still FAIL_FLOOR (10 at floor)
    mid = {s: 0.5 * (_floors(s, 10) + _floors(s, 30)) for s in range(10, 20)}
    f = dict(base, CHf=synth_arm("CHf", lambda s, t: 0.95 if t < 11 else (mid[s] if s in (10, 11) else 0.100)))
    out.append(("R_floor_sqrt10", f, ctrl, True, True, {"CHf": "FAIL", "CHf_sub": "FAIL_FLOOR"}))
    return out


def ck_verdict(V) -> dict:
    fails, n = [], 0
    for name, frames, ctrl, checks_ok, c_ok, exp in scenarios():
        res = V.analyze_frames(frames, checks_ok, ctrl, c_door_ok=c_ok)
        got = {"CHf": res["arms"]["CHf"]["label"], "Hf": res["arms"]["Hf"]["label"],
               "FEATURE_DOOR": res["FEATURE_DOOR"], "M_Hf": res["M_Hf"]["vs_MLP"],
               "Hs": res["ctrl"]["Hs"]["label"], "CHs": res["ctrl"]["CHs"]["label"],
               "T_CHf_CH": res["timing"]["CHf_vs_CH"]["label"], "T_CHf_ref": res["timing"]["CHf_vs_ref"]["label"],
               "T_Hf_H": res["timing"]["Hf_vs_H"]["label"], "RESIDUAL_CHf": res["flags"]["RESIDUAL_LOSS"]["CHf"]["set"],
               "T1_LOWER_CHf": res["flags"]["T1_LOWER"]["CHf"]["set"], "EVALOFF_CHf": res["EVALOFF"]["CHf"]["label"],
               "D_CHf|CH": res["signs"]["D_CHf|CH"]["label"], "CHf_sub": res["arms"]["CHf"].get("fail_subtype")}
        for k, v in exp.items():
            n += 1
            if got[k] != v:
                fails.append(f"{name}:{k} expected {v} got {got[k]}")
    k_ok = [V.kstar(n_) for n_ in (8, 9, 10)] == [8, 8, 9]
    st = V.sign_test([1.0] * 9 + [-1.0])
    s_ok = st["label"] == "HELPS" and abs(st["p"] - 0.021484375) < 1e-15
    # FEATURE_DOOR over every combination: Hf, CHf, ref, CH labels x cond x contrast
    inv, n_comb = [], 0
    for labs in itertools.product(("RESCUED", "PARTIAL", "FAIL"), repeat=4):
        lab = dict(zip(("Hf", "CHf", "ref", "CH"), labs))
        for cond in (True, False):
            n_comb += 1
            contrast = lab["ref"] == "FAIL" and lab["CH"] == "FAIL"
            d = V.feature_door(lab, cond, contrast)
            if not cond and d != "INAPPLICABLE":
                inv.append(("conditions fail but not INAPPLICABLE", lab, d))
            if cond and not contrast and d != "NOT_REPRODUCED":
                inv.append(("no contrast but not NOT_REPRODUCED", lab, d))
            if d == "FEATURE_CENTERING_SUFFICES" and not (lab["CHf"] == "RESCUED" and contrast and cond):
                inv.append(("SUFFICES needs CHf RESCUED and the contrast", lab, d))
            if d == "CENTERING_NOT_A_DOOR_IN_CNN" and not (lab["CHf"] == "FAIL" and lab["Hf"] == "FAIL"):
                inv.append(("NOT_A_DOOR needs CHf and Hf FAIL", lab, d))
            if cond and contrast and lab["Hf"] != "FAIL" and d == "CENTERING_NOT_A_DOOR_IN_CNN":
                inv.append(("Hf not FAIL but NOT_A_DOOR", lab, d))
            if cond and contrast and lab["CHf"] == "PARTIAL" and d != "FEATURE_CENTERING_PARTIAL":
                inv.append(("CHf PARTIAL but not PARTIAL", lab, d))
    # input tables
    good = synth_arm("CHf", _hold(0.95))
    corrupt = {}
    x = good.copy(); x.loc[x.index[0], "arm"] = "Hf"; corrupt["wrong_arm"] = x
    x = pd.concat([good, good[good.seed == 10].assign(seed=20)]); corrupt["foreign_seed"] = x
    x = pd.concat([good, good.iloc[[5]]]); corrupt["duplicate_task"] = x
    x = good.drop(good.index[7]); corrupt["missing_task"] = x
    x = good.copy(); x.loc[x.index[3], "online_acc"] = float("nan"); corrupt["nan_online"] = x
    x = good.drop(columns=["h_upd_f1"]); corrupt["missing_h_upd"] = x
    val_ok = not V.validate_frame(good, "CHf", range(10, 20)) and all(
        V.validate_frame(v, "CHf", range(10, 20)) for v in corrupt.values())
    val_detail = {k: bool(V.validate_frame(v, "CHf", range(10, 20))) for k, v in corrupt.items()}
    tf_ok = math.isnan(V.t_f({1: 0.1, **{t: 0.1 for t in range(2, 31)}})) and \
        V.t_f({1: 0.9, **{t: (0.9 if t < 4 else 0.2) for t in range(2, 31)}}) == 4
    return {"n_expectations": n, "failed": fails[:12], "kstar_8_9_10": k_ok, "sign_test_9_1": s_ok,
            "n_combinations": n_comb, "invariant_violations": [str(v) for v in inv[:6]], "table_validation": val_detail,
            "t_f_undefined_and_value": tf_ok,
            "pass": not fails and k_ok and s_ok and n_comb == 162 and not inv and val_ok and tf_ok}


# --------------------------------------------------------------------------
# K-mech (report-only tables) and K-gate (provenance gate)
# --------------------------------------------------------------------------

def ck_mech(V) -> dict:
    """The mu-path split, the path table and the per-position DC table on a real-format run (K-resume's
    CHf, seeds 10-11, 2 tasks, checkpoints t1 t2), against an independent recomputation."""
    src = SCR / "resume_A"
    if not (src / "units.npz").exists():
        return {"reason": "K-resume output missing", "pass": False}
    base = SCR / "mech_base"
    shutil.rmtree(base, ignore_errors=True)
    shutil.copytree(src, base / "CHf")
    V.OUT, V.BACKUP = base, base / "none"
    tw = V.three_way_table(arms=("CHf",), tasks=(1, 2))
    mech = V.mechanism_tables(arms=("CHf",))
    u = np.load(src / "units.npz")
    worst, n = 0.0, 0
    for t in (1, 2):
        for li, l in enumerate(LAYERS):
            own, up, cross = [], [], []
            for si, seed in enumerate((10, 11)):
                P0 = CN.init_params(seed, torch.device("cpu"))
                Pt = torch.load(src / "ckpt" / f"CHf_seed{seed}_t{t:02d}.pt", map_location="cpu", weights_only=False)["P"]
                W0 = P0[2 * li].detach().double().reshape(P0[2 * li].shape[0], -1)
                Wt = Pt[2 * li].double().reshape(Pt[2 * li].shape[0], -1)
                m0 = torch.from_numpy(u[f"m_in_{l}"][0, si]).double()
                mt = torch.from_numpy(u[f"m_in_{l}"][t, si]).double()
                own.append(np.median(((Wt - W0) @ m0).numpy()))
                up.append(np.median((W0 @ (mt - m0)).numpy()))
                cross.append(np.median(((Wt - W0) @ (mt - m0)).numpy()))
            r = tw[(tw.layer == l) & (tw.t == t)].iloc[0]
            for k, v in (("own_med", own), ("up_med", up), ("cross_med", cross)):
                worst = max(worst, abs(float(r[k]) - float(np.median(v))))
                n += 1
            for t_from in (0, 1):
                if t <= t_from:
                    continue
                q = mech["path"][(mech["path"].layer == l) & (mech["path"]["from"] == t_from) & (mech["path"].t == t)].iloc[0]
                dz = u[f"zbar_{l}"][t] - u[f"zbar_{l}"][t_from]
                dw = u[f"wm_{l}"][t] - u[f"wm_{l}"][t_from]
                worst = max(worst, abs(q.dz_med - float(np.median(np.median(dz, 1)))),
                            abs(q.dwm_med - float(np.median(np.median(dw, 1)))))
                n += 2
    ident = float(tw.identity_res.max())
    # the per-position DC table: from units.npz pmean_post_f1 with numpy, against the per_task columns
    pdt = V.pdc_table({"CHf": pd.read_csv(src / "per_task.csv")})
    rows = pd.concat([pd.read_csv(src / "t0.csv"), pd.read_csv(src / "per_task.csv")], ignore_index=True)
    pworst = 0.0
    for t in (0, 1, 2):
        pd_, cd_, dc_ = V.dc_parts_np(u["pmean_post_f1"][t], 8)
        g = rows[rows.task == t].sort_values("seed")
        sig = g.prms_f1.to_numpy(float)
        for si in range(2):
            q = pdt[(pdt.layer == "f1") & (pdt.task == t) & (pdt.seed == 10 + si)].iloc[0]
            # units hold float32 means: |d delta| <= |d mu| <= 2^-24 |mu|; prms is read at 10 digits
            bnd = 2 * 2.0 ** -24 * dc_[si] / sig[si] + 1e-9 * q.pdc_post
            pworst = max(pworst, abs(q.pdc_post - pd_[si] / sig[si]) / bnd,
                         abs(q.dc_post - dc_[si] / sig[si]) / (2 * 2.0 ** -24 * dc_[si] / sig[si] + 1e-9 * q.dc_post))
    return {"n_compared": n, "max_abs_diff": worst, "identity_res_max": ident, "pdc_table_diff_over_bound_max": pworst,
            "pass": n == 48 and worst <= 1e-12 and ident <= 1e-9 and pworst <= 1.0}


def ck_gate(V) -> dict:
    """provenance_gate accepts the registered configuration and rejects each kind of mismatch;
    reused_blobs accepts P181's blobs and rejects an edited per_task.csv."""
    head_chk = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    no_engine = subprocess.run(["git", "-C", str(REPO), "rev-parse", "b1b28854"], capture_output=True, text=True).stdout.strip()
    modes = {"Hf": "feat", "CHf": "feat", "Hs": "shuf", "CHs": "shuf"}

    def make(name, edit=None):
        base = SCR / "gate" / name
        shutil.rmtree(base, ignore_errors=True)
        base.mkdir(parents=True)
        (base / "checks.json").write_text(json.dumps({"all_pass": True,
                                                      "_meta": {"git_hash": head_chk, "git_dirty_code": False}}))
        for a in ("Hf", "CHf", "Hs", "CHs"):
            seeds = [10] if a in ("Hs", "CHs") else list(range(10, 20))
            pv = {"arm": a, "slots": [[a, s] for s in seeds], "n_tasks": 30, "epochs": 400, "lr": 0.001,
                  "beta_h": 0.01, "h_mode": modes[a], "data_sha256": {"cifar-10-python.tar.gz": "6d95"},
                  "gits": [{"git_hash": head_chk, "git_dirty_code": False}]}
            if edit:
                pv = edit(a, pv)
            (base / a).mkdir()
            (base / a / "provenance.json").write_text(json.dumps(pv))
        return base

    def chf_only(f):
        return lambda a, pv: f(pv) if a == "CHf" else pv

    cases = {
        "clean": (make("clean"), True),
        "epochs_399": (make("epochs", chf_only(lambda pv: {**pv, "epochs": 399})), False),
        "other_engine": (make("engine", chf_only(lambda pv: {**pv, "gits": [{"git_hash": no_engine, "git_dirty_code": False}]})), False),
        "dirty_run": (make("dirty", chf_only(lambda pv: {**pv, "gits": [{"git_hash": head_chk, "git_dirty_code": True}]})), False),
        "wrong_slots": (make("slots", chf_only(lambda pv: {**pv, "slots": [["CHf", s] for s in range(11, 21)]})), False),
        "other_data": (make("data", chf_only(lambda pv: {**pv, "data_sha256": {"x": "y"}})), False),
        "door_mode": (make("mode", chf_only(lambda pv: {**pv, "h_mode": "chan"})), False),
        "no_CHs": (make("no_chs", lambda a, pv: {**pv, "arm": "gone"} if a == "CHs" else pv), False),
    }
    shutil.rmtree(cases["no_CHs"][0] / "CHs")
    res = {k: V.provenance_gate(b)["ok"] for k, (b, want) in cases.items()}
    ok = all(res[k] == want for k, (b, want) in cases.items())
    # reused blobs: the real files pass; an edited copy fails
    real = V.reused_blobs()["ok"]
    fake = SCR / "gate" / "p181_edit"
    shutil.rmtree(fake, ignore_errors=True)
    for a in ("ref", "C", "H", "CH"):
        (fake / a).mkdir(parents=True)
        shutil.copy(P181 / a / "per_task.csv", fake / a / "per_task.csv")
    with open(fake / "CH" / "per_task.csv", "a") as fh:
        fh.write("\n")
    edited = V.reused_blobs(fake)["ok"]
    # the C door fails closed when the runs' units.npz are missing (audit 1)
    empty = SCR / "gate" / "empty_out"
    shutil.rmtree(empty, ignore_errors=True)
    empty.mkdir(parents=True)
    saved = (V.OUT, V.BACKUP)
    V.OUT, V.BACKUP = empty, empty / "none"
    try:
        cd = V.c_door_from_runs()
        door_closed = V.c_door_ok_from(cd) is False and V.c_door_ok_from({"ok": True}) is True and \
            V.c_door_ok_from({"ok": False}) is False
        # the mechanism inputs: missing for the new arms (empty OUT), complete and sha-exact for P181's arms
        mb = V.required_mechanism_inputs()
        mech_new_missing = any(x.startswith("CHf: units.npz missing") for x in mb) and \
            any(x.startswith("Hs: ckpt missing") for x in mb)
        mech_p181_ok = not any(x.startswith(("ref:", "C:", "H:", "CH:")) for x in mb)
    finally:
        V.OUT, V.BACKUP = saved
    # both adopted controls are required
    good = synth_arm("Hs", _hold(0.95), seeds=(10,))
    ctl_ok = not V.control_problems({"Hs": good, "CHs": synth_arm("CHs", _hold(0.95), seeds=(10,))}) and \
        bool(V.control_problems({"Hs": good, "CHs": None}))
    return {**{f"gate_ok_{k}": v for k, v in res.items()}, "reused_blobs_real": real, "reused_blobs_edited": edited,
            "c_door_fails_closed": door_closed, "mechanism_missing_reported": mech_new_missing,
            "mechanism_p181_complete": mech_p181_ok, "controls_required": ctl_ok,
            "pass": ok and real and not edited and door_closed and mech_new_missing and mech_p181_ok and ctl_ok}


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


def _rss(pid: int) -> tuple[int, int]:
    try:
        txt = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return 0, 0
    get = lambda k: int(next(l for l in txt.splitlines() if l.startswith(k)).split()[1]) // 1024  # noqa: E731
    return get("VmRSS:"), get("VmHWM:")


def ck_cost(_M) -> dict:
    res = {"background": {"gpu_procs": _gpu_proc_mem(),
                          "proj_004_drift_python": subprocess.run(["bash", "-c", "pgrep -c -f '[p]roj_004_drift/\\.venv/bin/python' || true"],
                                                                  capture_output=True, text=True).stdout.strip()}}
    epochs = 40
    go = SCR / "cost_go"
    for label, arms in (("single_CHf", ("CHf",)), ("concurrent_CHf_Hf", ("CHf", "Hf"))):
        go.unlink(missing_ok=True)
        procs = {a: subprocess.Popen([PY, str(COST), "--arm", a, "--epochs", str(epochs), "--wait-file", str(go)],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=REPO)
                 for a in arms}
        peak, rss, hwm = {}, {}, {}
        t_wait = time.time()
        while time.time() - t_wait < 180:
            mem = _gpu_proc_mem()
            for a, pr in procs.items():
                if pr.pid in mem:
                    peak[a] = max(peak.get(a, 0), mem[pr.pid])
            if len(peak) == len(arms) and time.time() - t_wait > 25:
                break
            time.sleep(1)
        go.touch()
        while any(pr.poll() is None for pr in procs.values()):
            mem = _gpu_proc_mem()
            for a, pr in procs.items():
                if pr.pid in mem:
                    peak[a] = max(peak.get(a, 0), mem[pr.pid])
                r_, h_ = _rss(pr.pid)
                if r_:
                    rss[a] = max(rss.get(a, 0), r_)
                    hwm[a] = max(hwm.get(a, 0), h_)
            time.sleep(1)
        got = {}
        for a, pr in procs.items():
            o, e = pr.communicate()
            got[a] = json.loads(o.strip().splitlines()[-1]) if pr.returncode == 0 else {"err": e[-800:]}
        res[label] = {"runs": got, "nvidia_smi_peak_mb": peak, "vmrss_max_mb": rss, "vmhwm_max_mb": hwm}
    go.unlink(missing_ok=True)
    ok = all("s_per_epoch" in v for lab in ("single_CHf", "concurrent_CHf_Hf") for v in res[lab]["runs"].values())
    if ok:
        res["projected_30_tasks_min"] = {lab: {a: v["s_per_epoch"] * 400 * 30 / 60 for a, v in res[lab]["runs"].items()}
                                         for lab in ("single_CHf", "concurrent_CHf_Hf")}
    return {**res, "pass": ok}


# --------------------------------------------------------------------------

CHECKS = {
    "K-same181": ("このエンジンの ref・C・H・CH = P181 のエンジン（R=3・2 epoch・全状態・V・m・h_upd・evaluate・switch_eval・診断）", ck_same181, [
        ("H の conv の m を位置ごとにする", [('"H":   dict(shift=None, h="chan"),', '"H":   dict(shift=None, h="feat"),')]),
        ("ReLU を leaky 0.01 に", [("    a1 = act.phi(z1, 0)\n", "    a1 = torch.where(z1 > 0, z1, 0.01 * z1)\n")]),
    ], "bit 一致（許容 0）", ENGINE),
    "K-reuse": ("CLI の ref・C・H・CH・R=10・1 課題が P181 の per_task（課題 1）・t0・curves・課題 1 末のチェックポイントと一致、P181 の per_task.csv は b9ae5a26 の blob",
                ck_reuse, [], "文字列一致・bit 一致（許容 0）", ENGINE),
    "K-Hf-pos": ("H′ が (チャネル × 位置) ごと: bm・次の層の入力・位置の並べ替えへの同変性・m の位置による違い", ck_hf_pos, [
        ("H′ がチャネル平均を引く（H に戻す）", [('"Hf":  dict(shift=None, h="feat"),', '"Hf":  dict(shift=None, h="chan"),')]),
        ("引く地図の y と x を転置", [("            return m[l].reshape(1, R * CH, s, s)\n",
                                      "            return m[l].reshape(1, R * CH, s, s).transpose(2, 3)\n")]),
        ("bm を位置でも平均", [("            return F.max_pool2d(self.phi(z, l), POOL, POOL).mean(0).view(self.R, -1)",
                                "            return F.max_pool2d(self.phi(z, l), POOL, POOL).mean((0, 2, 3), keepdim=True).expand(1, -1, SIDE[l], SIDE[l]).reshape(self.R, -1)")]),
    ], "bit 一致（許容 0）。H の不変性は float64 で 1e-6（float32 の和の順の違い、√K·2^-24 の桁）", ENGINE),
    "K-Hf-ema": ("m の更新 = 中心化前の pool 出力（位置ごと）と fc 出力のバッチ平均を β=0.01 で追う（Hf・Hs）、境界で不変、h_upd", ck_hf_ema, [
        ("中心化後を追う", [("                bm = self.out_mean(z, l)\n", "                bm = self.out_mean(z, l) - self.m[l]\n")]),
        ("β = 0.1", [("BETA_H = 0.01", "BETA_H = 0.1")]),
        ("pool の前の平均を追う", [("            return F.max_pool2d(self.phi(z, l), POOL, POOL).mean(0).view(self.R, -1)",
                                     "            return self.phi(z, l)[:, :, ::2, ::2].mean(0).reshape(self.R, -1)")]),
        ("Adam の後の重みで bm を取る", [("            self.act.update([out[0].detach(), out[2].detach(), out[4].detach(), out[6].detach()])",
                                          "            o2 = forward(self.P, xb, self.act)\n            self.act.update([o2[0].detach(), o2[2].detach(), o2[4].detach(), o2[6].detach()])")]),
        ("境界（new_labels）で m をリセット", [("    def step(self, j: int) -> None:\n",
                                               "    def new_labels(self):\n        super().new_labels()\n        for t in self.act.m:\n            t.zero_()\n\n    def step(self, j: int) -> None:\n")]),
    ], "bit 一致（許容 0）", ENGINE),
    "K-Hf-grad": ("m は勾配を持たず、勾配が m を定数とした独立の再計算と bit 一致（Hf）", ck_hf_grad, [
        ("勾配を通す（値は同じで勾配だけバッチ平均を通る）", [("        p1 = p1 - act.sub_map(0, m)\n",
                                                            "        p1 = p1 - (act.sub_map(0, m) + (p1.mean(0, keepdim=True) - p1.mean(0, keepdim=True).detach()))\n")]),
    ], "bit 一致（許容 0）", ENGINE),
    "K-Hf-eval": ("課題末の fit CE・切替の CE がその時点の m での再計算と bit 一致し、m=0 では一致しない（Hf）", ck_hf_eval, [
        ("評価で m を使わない", [("    E.forward = forward\n",
                                 "    E.forward = lambda P, xb, act: forward(P, xb, act, m=[torch.zeros_like(t) for t in act.m])\n")]),
    ], "bit 一致（許容 0）", ENGINE),
    "K-Hf-first": ("Hf・Hs の最初の更新 = ref、CHf = C、その直後の m は非零、3 更新後は違う", ck_hf_first, [
        ("m の初期値が非零", [("                self.m[l] = torch.zeros(R, NFEAT[l], device=device)",
                              "                self.m[l] = torch.full((R, NFEAT[l]), 0.01, device=device)")]),
    ], "bit 一致（許容 0）", ENGINE),
    "K-evaloff": ("m = 0 の評価は読むだけ（Hf: 状態・m・乱数の流れ・2 epoch の軌道が不変）", ck_evaloff, [
        ("m = 0 の評価が m をその場で 0 にする", [("        zeros = [torch.zeros_like(t) for t in act.m]",
                                                  "        zeros = [t.zero_() for t in act.m]")]),
    ], "bit 一致（許容 0）", ENGINE),
    "K-shuf": ("Hs: π は流れからの derangement、次の層の入力 = P − m[π]、チャネル平均は保つ、H′ の地図とは違う", ck_shuf, [
        ("π を当てない（Hs = H′）", [("            return m[l].view(R, CH, s * s)[:, :, self.perm[l]].reshape(1, R * CH, s, s)",
                                      "            return m[l].reshape(1, R * CH, s, s)")]),
        ("π に不動点を許す", [("        if not bool((p == ar).any()):", "        if True:")]),
        ("π を EMA の側に当てる", [("            return F.max_pool2d(self.phi(z, l), POOL, POOL).mean(0).view(self.R, -1)",
                                     "            bm = F.max_pool2d(self.phi(z, l), POOL, POOL).mean(0).view(self.R, -1)\n"
                                     "            return bm.view(self.R, CH, -1)[:, :, self.perm[l]].reshape(self.R, -1) if self.mode == 'shuf' else bm")]),
    ], "bit 一致（許容 0）。チャネル平均は float64 で 1e-6（和の順）", ENGINE),
    "K-host": ("1 step の損失と勾配: 独立に書いた扉つきの 1 run の宿主の形の順伝播（Hf・CHf・Hs）", ck_host, [
        ("φ の中で中心化（φ(z − m)）", [("    a1 = act.phi(z1, 0)\n    p1 = F.max_pool2d(a1, POOL, POOL)\n    if hon:\n        p1 = p1 - act.sub_map(0, m)\n",
                                         "    a1 = act.phi(z1 - (F.interpolate(act.sub_map(0, m), scale_factor=2) if hon else 0.0), 0)\n    p1 = F.max_pool2d(a1, POOL, POOL)\n")]),
        ("conv の m を引かない", [("    if hon:\n        p2 = p2 - act.sub_map(1, m)\n", "    if False:\n        p2 = p2 - act.sub_map(1, m)\n")]),
    ], "損失の相対差 1e-5・勾配の相対差 1e-4（P181 の K-host と同じ）", ENGINE),
    "K-pdc": ("位置ごとの平均の列（pdc・cdc・dc・prms）が独立の float64 の再計算と一致、µ_post = µ_pre − 引いた地図、H・ref で前 = 後、正確な平均の H′ で後 ≈ 0", ck_pdc, [
        ("pre と post の取り違え", [("        pd_a, cd_a, dc_a = dc_parts(mu_pre, SIDE[k])", "        pd_a, cd_a, dc_a = dc_parts(mu_post, SIDE[k])")]),
        ("位置の部分からチャネル平均を引かない", [("    delta = (mc - cbar).reshape(R, -1)", "    delta = mc.reshape(R, -1)")]),
    ], "相対 1e-9（float32 の同じ値の float64 での和の順）、µ_post は float64 で 1e-6、前 = 後と正確な平均の後は float32 の引き算の丸め 3·2^-24·max|P|·√F/σ", ENGINE),
    "K-ident": ("z̄_j = ⟨W_j, m_in⟩ + b_j が TF32 と float32 の丸めの上界の内（Hf・CHf・Hs、全 4 層）", ck_ident, [
        ("平均パッチの並びを転置", [("        W = Wt[l].detach().reshape(R, WIDTHS[l], -1).double()            # (R, n, d)",
                                     "        W = (Wt[l].detach().transpose(-1, -2) if Wt[l].dim() == 5 else Wt[l].detach()).reshape(R, WIDTHS[l], -1).double()")]),
        ("m_in を中心化前の入力から取る", [("        return out, (x, p1, h, i4)",
                                           "        return out, (x, F.max_pool2d(a1, POOL, POOL), F.max_pool2d(a2, POOL, POOL).reshape(B, R, CN.FLAT).transpose(0, 1), a3)")]),
    ], "上界 (2^-10 + γ_{K+1})(⟨|W|, mean|p|⟩ + |b|)（P181 と同じ）", ENGINE),
    "K-graph": ("Hf・CHf・Hs の束で CUDA graph と eager が 2 epoch で m・h_upd を含めて bit 一致", ck_graph, [], "bit 一致", ENGINE),
    "K-indep": ("CHf の束の run 1 の重み・入力・m を乱しても run 0・2 は bit で不変、run 1 は変わる", ck_indep, [], "bit 一致", ENGINE),
    "K-loop": ("run（eager）= 課題の手での再生、m は課題の境界を越えて引き継がれる、h_upd = 150/課題（Hf）", ck_loop, [
        ("課題の境界で m をリセット（run）", [("        B.act.hupd.zero_()\n",
                                              "        B.act.hupd.zero_()\n        for _mm in B.act.m:\n            _mm.zero_()\n")]),
    ], "bit 一致", ENGINE),
    "K-resume": ("CHf の 2 課題を通しで回すのと 1 課題で止めて再開するのとで per_task・t0・curves・units・状態が一致", ck_resume, [],
                 "文字列・bit 一致", ENGINE),
    "K-verdict": ("合成の per_task で設計どおりのラベル、FEATURE_DOOR の全 162 組合せの不変条件、表の関門", ck_verdict, [
        ("k* の取り違え", [("        if binom_two_sided(k, n) < ALPHA:", "        if binom_two_sided(k, n) < 0.2:")]),
        ("0.5 の線を > に", [('"kept": W >= KEEP_LINE', '"kept": W > KEEP_LINE')]),
        ("床の √10 を √30 に", [("mf.std(ddof=1) / math.sqrt(len(mf))", "mf.std(ddof=1) / math.sqrt(30)")]),
        ("主窓の課題範囲", [("WINDOW = (21, 30)", "WINDOW = (11, 30)")]),
        ("NOT_A_DOOR が Hf を見ない", [('    if labels["CHf"] == "FAIL" and labels["Hf"] == "FAIL":', '    if labels["CHf"] == "FAIL":')]),
        ("(C′) が全課題を見る", [("        t1 = df[df.task == 1]\n", "        t1 = df\n")]),
        ("(C′) が課題 1 を見ない", [("        for _, r in t1[~(t1[col] >= 1)].iterrows():", "        for _, r in t1[~(t1[col] >= 0)].iterrows():")]),
        ("Hs の比較の腕の取り違え", [('CTRL = {"Hs": ("Hf", "H"), "CHs": ("CHf", "CH")}', 'CTRL = {"Hs": ("Hf", "CH"), "CHs": ("CHf", "H")}')]),
        ("T_f の向きの取り違え", [('    return {**t, "label": {"HELPS": "LATER", "HURTS": "EARLIER"}.get(t["label"], "NO_TIMING_DIFF")}',
                                    '    return {**t, "label": {"HELPS": "EARLIER", "HURTS": "LATER"}.get(t["label"], "NO_TIMING_DIFF")}')]),
        ("seed 集合の検査を落とす", [("    if set(int(x) for x in df.seed) != set(seeds):", "    if False:")]),
        ("対比（CH が FAIL）を見ない", [('    contrast_ok = labels["ref"] == "FAIL" and labels["CH"] == "FAIL"', '    contrast_ok = labels["ref"] == "FAIL"')]),
    ], "合成データの期待値（設計から）", VERDICT),
    "K-mech": ("報告の表（µ 経路の自分・上流・交差、起点 t0・t1、位置ごとの DC）が独立の再計算と一致（K-resume の CHf）", ck_mech, [
        ("自分と上流の取り違え", [("own, ups, cross = dW @ m0, W0 @ dm, dW @ dm", "own, ups, cross = W0 @ dm, dW @ m0, dW @ dm")]),
        ("起点 t1 の分解が t0 から引く", [("wm[t] - wm[t_from]", "wm[t] - wm[0]")]),
    ], "float64 の同じ量（1e-12）・恒等式の残差 1e-9・位置ごとの DC の表は float32 の units からの再計算と 2·2^-24·‖µ‖/σ + 1e-9（保存の丸め）", VERDICT),
    "K-gate": ("provenance の関門・再利用の blob・C の扉の fail-closed・対照の完了・機構の入力の欠損の報告", ck_gate, [
        ("エンジンの sha256 を照合しない", [("    if chk_sha is None or shas != {chk_sha}:", "    if False:")]),
        ("汚れた木からの走を通す", [("            if g.get(\"git_dirty_code\"):", "            if False:")]),
        ("扉の型を照合しない", [("        if pv.get(\"h_mode\") != H_MODE[a]:", "        if False:")]),
        ("再利用の blob を照合しない", [("        if not want or want != have:", "        if False:")]),
        ("C の扉の実走確認が units 欠損で通る（fail-open）", [('    return cdoor.get("ok") is True', '    return cdoor.get("ok") is not False')]),
        ("対照の provenance の欠損を通す", [("            if a in NEW_ARMS + tuple(CTRL):", "            if a in NEW_ARMS:")]),
        ("対照の per_task の欠損を通す", [('            out.append(f"{a}: per_task.csv missing (control adopted by the parent, spec §12.1)")',
                                           "            continue")]),
        ("機構の入力の欠損を報告しない", [("    return bad\n\n\ndef commit_time", "    return []\n\n\ndef commit_time")]),
    ], "合成の provenance の期待値（設計から）。C の扉・対照・機構の入力は監査（1・2・3）の fail-closed", VERDICT),
    "K-cost": ("CHf の束（R=10）の 1 epoch の秒数（単独と Hf との同時）・GPU メモリの峰・VmRSS（seed 100–109、精度は読まない）",
               ck_cost, [], "—", ENGINE),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-cost", action="store_true")
    ap.add_argument("--skip-reuse", action="store_true")
    args = ap.parse_args()
    SCR.mkdir(parents=True, exist_ok=True)
    if DUMP.exists():
        RESULTS.update(json.loads(DUMP.read_text()))
        RESULTS.pop("all_pass", None)
    keys = args.only.split(",") if args.only else list(CHECKS)
    if args.skip_cost and "K-cost" in keys:
        keys.remove("K-cost")
    if args.skip_reuse and "K-reuse" in keys:
        keys.remove("K-reuse")
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

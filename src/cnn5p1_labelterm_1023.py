#!/usr/bin/env python3
"""Read-only switch-gradient analysis of registered saved states (no optimizer steps).

Run cnn5p1_labelterm_1023_checks.py preflight first, then this file's measure command.
The training forward, data, checkpoint loader and activation are imported unchanged.
See analysis/cnn5p1_labelterm_1023/implementation.md for numerical conventions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import kappa_cnn5p1_1022 as KP

EXPERIMENT = "cnn5p1_labelterm_1023"
OUT = ROOT / "results" / EXPERIMENT
SPEC = ROOT / "specs" / f"spec_{EXPERIMENT}.md"
BOUNDARIES = (1, 9, 19, 29)
BUDGETS = KP.BUDGETS
ARMS = ("R", "ELU", "KKT1", "SNAfrz1")
SITES = ("f2", "f1")
PERMUTATIONS = 5
PERM_SEED = 1023
DT = torch.float64
METRICS = ("agree_rand", "agree_unif", "ratio_L", "sink_real", "sink_rand",
           "sink_unif", "L_sink", "class_sink", "ratio_class", "sink_difference",
           "n_alive", "n_ratio_L", "n_ratio_class")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean_json(x):
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean_json(x.tolist())
    if isinstance(x, np.generic):
        return clean_json(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def write_json(path, x):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(x), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n")


def source_hashes():
    paths = [SPEC, ROOT / "src" / f"{EXPERIMENT}.py",
             ROOT / "src" / f"{EXPERIMENT}_checks.py"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def relative_error(a, b):
    """Symmetric vector infinity-norm relative error; exactly-zero vectors give zero."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)) /
                 max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-300))


def checkpoint_inventory(budgets, boundaries):
    manifest = json.loads((KP.PREV / "backup_manifest.json").read_text())
    expected = {r["backup"]: r for r in manifest}
    records = []
    for budget in budgets:
        for boundary in sorted({0, *boundaries}):
            p = KP.ckpt_path(budget, boundary)
            h = sha256(p)
            if h != expected[str(p)]["sha256"]:
                raise RuntimeError(f"checkpoint SHA256 mismatch: {p}")
            records.append(dict(path=str(p), sha256=h, bytes=p.stat().st_size,
                                budget=budget, boundary=boundary, manifest_match=True))
    return records


def provenance(device, inventory, chunk, started):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    paths = [ROOT / "src" / "kappa_cnn5p1_1022.py",
             ROOT / "src" / "cifar5p1_cnn_budget_1019.py",
             ROOT / "src" / "neokunekune_cifar5p1_cnn_1018.py",
             ROOT / "src" / "neokunekune_methods_cifar5p1_cnn_1018.py",
             ROOT / "src" / "sna_cnn_cause_1009.py",
             ROOT / "src" / "cifar5p1_mlp_0920.py",
             ROOT / "analysis" / EXPERIMENT / "summarize.py"]
    for budget in BUDGETS:
        paths += [KP.PREV / f"b{budget}" / f"{n}.json"
                  for n in ("task_plan", "frozen_alpha", "provenance")]
    return dict(experiment=EXPERIMENT, status="RUNNING", started_utc=started,
                git_hash=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
                worktree=str(ROOT), git_status_at_start=git("status", "--porcelain"),
                source_sha256={**source_hashes(), **{str(p.relative_to(ROOT)): sha256(p)
                                                   for p in paths if p.exists()}},
                checkpoints=inventory, data_archive=str(KP.DATA_TGZ),
                python=platform.python_version(), python_executable=sys.executable,
                torch=torch.__version__, numpy=np.__version__, device=str(device),
                cuda_available=torch.cuda.is_available(), threads=torch.get_num_threads(),
                chunk=chunk, precision=KP.NK.precision_state(),
                permutations=PERMUTATIONS, permutation_seed=PERM_SEED,
                no_parameter_updates=True, no_training_or_replay=True)


def validate_bundle(bundle):
    p = KP.PREV / f"b{bundle.budget}"
    plans = json.loads((p / "task_plan.json").read_text())
    meta = json.loads((p / "provenance.json").read_text())["meta"]
    if bundle.meta != meta:
        raise RuntimeError("checkpoint and training provenance disagree")
    for seed, plan in bundle.plans.items():
        actual = [{"task": i, "hard": h, "classes": cs}
                  for i, (h, cs) in enumerate(plan, 1)]
        if actual != plans[str(seed)]:
            raise RuntimeError(f"task plan mismatch: budget {bundle.budget}, seed {seed}")
    if set(bundle.slots) != {(cfg, seed) for cfg in KP.BG.ARMS4 for seed in range(2000, 2010)}:
        raise RuntimeError("not the registered 4 arms x 10 seeds")


def validate_frozen_alpha(bundle):
    frozen = json.loads((KP.PREV / f"b{bundle.budget}" / "frozen_alpha.json").read_text())
    worst = 0.0
    for r, (cfg, seed) in enumerate(bundle.slots):
        if bundle.arms[r] != "SNAfrz1":
            continue
        for l, site in enumerate(KP.NK.SITES):
            a = KP.tonp(bundle.eng.act.alpha(l)[r])
            ref = np.array(frozen["runs"][f"{cfg}|{seed}"][site])
            worst = max(worst, relative_error(a, ref))
    if worst != 0:
        raise RuntimeError(f"frozen alpha mismatch: {worst}")
    return worst


def activation_derivative(act, z, layer):
    """Differentiate the actual saved float32 activation expression (including branch rules)."""
    with torch.enable_grad():
        zz = z.detach().requires_grad_(True)
        a = act.phi(zz, layer)
        return torch.autograd.grad(a, zz, torch.ones_like(a))[0].detach()


@torch.no_grad()
def features(bundle, task, chunk=50):
    """Training float32 forward; retain only inputs/derivatives needed by the two FC sites."""
    rows = bundle.rows(task)
    R, N = rows.shape
    dev = bundle.device
    buf = {k: torch.empty(R, N, w, device=dev, dtype=torch.float32)
           for k, w in (("a_f1", 1024), ("a_f2", 100), ("d_f1", 100),
                        ("d_f2", 100), ("logits", 100))}
    diag = torch.zeros(R, 100, device=dev, dtype=DT)
    dmax = torch.zeros(R, 100, device=dev)
    for start in range(0, N, chunk):
        idx = rows[:, start:start + chunk]
        out = KP.E.forward(bundle.eng.P, bundle.data.X[idx], bundle.eng.act)
        h = F.max_pool2d(out[3], KP.E.POOL, KP.E.POOL).reshape(idx.shape[1], R, 1024).transpose(0, 1)
        for k, v in (("a_f1", h), ("a_f2", out[5]), ("logits", out[8]),
                     ("d_f1", activation_derivative(bundle.eng.act, out[4], 2)),
                     ("d_f2", activation_derivative(bundle.eng.act, out[6], 3))):
            buf[k][:, start:start + chunk] = v
        dd = bundle.eng.act.dphi(out[6], 3)
        diag += dd.double().sum(1)
        dmax = torch.maximum(dmax, dd.amax(1))
    y = bundle.data.Y[rows]
    buf.update(labels=y, rows=rows, diag_dmean=diag / N, diag_dmax=dmax,
               acc=(buf["logits"].argmax(2) == y).double().mean(1))
    return buf


def permutation_indices(seed, boundary, task, n):
    # The same image permutations across arms and budgets of a seed/task.
    g = torch.Generator().manual_seed(PERM_SEED + seed * 10000 + boundary * 100 + task)
    return torch.stack([torch.randperm(n, generator=g) for _ in range(PERMUTATIONS)])


def _median(a):
    return float(np.median(a)) if len(a) else float("nan")


def unit_summary(arr):
    live = arr["alive"].astype(bool)
    sr, sa, su, lab, ct = [arr[k] for k in ("S_real", "S_rand", "S_unif", "L", "class_term")]
    frac = lambda v: float(np.mean(v[live])) if live.any() else float("nan")
    rl, rc = live & (sa != 0), live & (su != 0)
    return dict(n_alive=int(live.sum()), n_units=len(live), n_ratio_L=int(rl.sum()),
                n_ratio_class=int(rc.sum()), agree_rand=frac(np.sign(sr) == np.sign(sa)),
                agree_unif=frac(np.sign(sr) == np.sign(su)),
                ratio_L=_median(np.abs(lab[rl]) / np.abs(sa[rl])),
                sink_real=frac(sr > 0), sink_rand=frac(sa > 0), sink_unif=frac(su > 0),
                L_sink=frac(lab < 0), class_sink=frac(ct > 0),
                ratio_class=_median(np.abs(ct[rc]) / np.abs(su[rc])),
                sink_difference=frac(sr > 0) - frac(sa > 0),
                L_median=_median(lab[live]), L_abs_median=_median(np.abs(lab[live])),
                nonalive_S_absmax=float(np.abs(sr[~live]).max()) if (~live).any() else 0.0)


@torch.no_grad()
def quantities(bundle, feat, slot, site, task, do_permutations=True):
    """Independent contractions of all five quantities, not identities imposed by assignment.

    For FC1: downstream cotangent = ((residual @ W_head) * d_f2) @ W_f2.
    Only its own affine weights/bias contribute to the reported position change.
    """
    a = feat[f"a_{site}"][slot].double()
    d = feat[f"d_{site}"][slot].double()
    d2 = feat["d_f2"][slot].double()
    y = feat["labels"][slot]
    logits = feat["logits"][slot].double()
    W = bundle.eng.P[8][slot].detach().double()
    W2 = bundle.eng.P[6][slot].detach().double()
    pi = logits.softmax(1)
    N = len(y)
    Y = F.one_hot(y, 100).double()
    ybar = Y.mean(0)
    abar = a.mean(0)
    kernel = a @ abar + 1
    kd = kernel[:, None] * d
    live_weight = (kernel[:, None] * d.abs()).sum(0)

    def downstream(q, head=W):
        z = q @ head
        return z if site == "f2" else (z * d2) @ W2

    def push(q, head=W):
        return (kd * downstream(q, head)).mean(0)

    residuals = dict(S_real=pi - Y, S_rand=pi - ybar,
                     S_unif=pi - .01, L=Y - ybar,
                     class_term=(torch.full_like(pi, .01) - ybar))
    vals = {k: push(q) for k, q in residuals.items()}
    centered = W - W.mean(0, keepdim=True)
    checks = {"identity_real_rel": relative_error(KP.tonp(vals["S_real"]),
                                                  KP.tonp(vals["S_rand"] - vals["L"])),
              "identity_rand_rel": relative_error(KP.tonp(vals["S_rand"]),
                                                  KP.tonp(vals["S_unif"] + vals["class_term"]))}
    for k, q in residuals.items():
        checks[f"center_{k}_rel"] = relative_error(KP.tonp(vals[k]), KP.tonp(push(q, centered)))
    arr = {k: KP.tonp(v) for k, v in vals.items()}
    arr.update(alive=KP.tonp(live_weight > 0).astype(bool), live_weight=KP.tonp(live_weight),
               dmean=KP.tonp(d.mean(0)), dabsmean=KP.tonp(d.abs().mean(0)))
    scal = {**unit_summary(arr), **checks,
            "K_min": float(kernel.min()), "K_max": float(kernel.max()),
            "K_nonpositive": int((kernel <= 0).sum()), "N": N,
            "label_sum_error": float((pi.sum(1) - 1).abs().max()),
            "easy_L_absmax": float(vals["L"].abs().max()) if len(bundle.classes(slot, task)) == 1 else float("nan")}
    if do_permutations and len(bundle.classes(slot, task)) == 5:
        perms = permutation_indices(bundle.slots[slot][1], bundle.boundary, task, N).to(y.device)
        lp = torch.stack([push(Y[p] - ybar) for p in perms])
        spr = torch.stack([push(pi - Y[p]) for p in perms])
        arr.update(L_perm=KP.tonp(lp), S_real_perm=KP.tonp(spr))
        live = arr["alive"]
        # Exact variance of a balanced random assignment, allowing sample-dependent readout (FC1).
        # Var(L_i) = [mean_nc q_nci^2 - mean_c(mean_n q_nci)^2] / (N-1).
        qclass = []
        for c in bundle.classes(slot, task):
            q = -ybar.expand(N, -1).clone()
            q[:, c] += 1
            qclass.append(kd * downstream(q))
        qc = torch.stack(qclass)
        variance = ((qc.square().mean((0, 1)) - qc.mean(1).square().mean(0)) / (N - 1)).clamp_min(0)
        arr["L_perm_null_sd"] = KP.tonp(variance.sqrt())
        scal["perm_agree_mean"] = float(np.mean(np.sign(arr["S_real_perm"][:, live]) ==
                                                 np.sign(arr["S_rand"][None, live]))) if live.any() else float("nan")
        scal["perm_L_median"] = _median(arr["L_perm"][:, live].ravel())
        scal["perm_L_abs_median"] = _median(np.abs(arr["L_perm"][:, live]).ravel())
    else:
        arr.update(L_perm=np.full((PERMUTATIONS, 100), np.nan),
                   S_real_perm=np.full((PERMUTATIONS, 100), np.nan),
                   L_perm_null_sd=np.full(100, np.nan))
        scal.update(perm_agree_mean=float("nan"), perm_L_median=float("nan"),
                    perm_L_abs_median=float("nan"))
    return scal, arr


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def measure(device, chunk):
    from src import cnn5p1_labelterm_1023_checks as CH
    checkfile = OUT / "checks.json"
    checks = json.loads(checkfile.read_text())
    gate = checks.get("preflight", {})
    if not gate.get("passed") or gate.get("source_sha256") != source_hashes():
        raise SystemExit("Run passing (a)/(c) preflight with this exact source before measurement")
    if gate["device"] != str(device) or gate["chunk"] != chunk:
        raise SystemExit("Preflight device/chunk differs")
    started = utc_now()
    inv = checkpoint_inventory(BUDGETS, BOUNDARIES)
    prov = provenance(device, inv, chunk, started)
    write_json(OUT / "provenance.json", prov)
    raw = OUT / "units"
    raw.mkdir(parents=True, exist_ok=True)
    data = KP.load_data(device)
    prov["data_sha256"] = data.sha256
    per_task = KP.read_per_task()
    rows_all, reproduction, excluded = [], [], []
    t0 = time.time()
    try:
        for budget in BUDGETS:
            bundle = KP.Bundle(budget, data, device)
            validate_bundle(bundle)
            for boundary in BOUNDARIES:
                bundle.load_state(boundary)
                validate_frozen_alpha(bundle)
                own = features(bundle, boundary, chunk)
                own_checks = CH.reproduction_checks(bundle, own, per_task)
                reproduction.extend(own_checks)
                if not all(c["passed"] for c in own_checks):
                    raise RuntimeError("full-run forward reproduction (a) failed; stop")
                collapsed = {}
                for r, (cfg, seed) in enumerate(bundle.slots):
                    pt = per_task[(budget, cfg, seed, boundary)]
                    collapse = float(pt["train_acc"]) < KP.COLLAPSE_ACC
                    all_dead = bool((own["diag_dmax"][r] == 0).all())
                    if collapse and not (bundle.arms[r] == "R" and all_dead):
                        raise RuntimeError("kappa collapse rule does not match all-dead ReLU")
                    collapsed[r] = collapse
                    if collapse:
                        excluded.append(dict(budget=budget, boundary=boundary, cfg=cfg, seed=seed,
                                             train_acc=float(pt["train_acc"]), f2_all_dead=all_dead))
                del own
                for kind, task in [("easy", boundary + 1)] + ([("hard", boundary + 2)] if boundary < 29 else []):
                    feat = features(bundle, task, chunk)
                    arrays, meta = {}, []
                    for r, (cfg, seed) in enumerate(bundle.slots):
                        for site in SITES:
                            sc, ar = quantities(bundle, feat, r, site, task)
                            rec = dict(budget=budget, boundary=boundary, task=task, kind=kind,
                                       arm=bundle.arms[r], cfg=cfg, seed=seed, slot=r, site=site,
                                       classes=" ".join(map(str, bundle.classes(r, task))),
                                       excluded=collapsed[r],
                                       state_train_acc=float(per_task[(budget, cfg, seed, boundary)]["train_acc"]),
                                       **sc)
                            rows_all.append(rec)
                            meta.append({k: rec[k] for k in ("budget", "boundary", "task", "kind", "arm", "cfg", "seed", "slot", "site", "excluded")})
                            for k, v in ar.items():
                                arrays.setdefault(k, []).append(v)
                    np.savez_compressed(raw / f"b{budget}_s{boundary:02d}_{kind}.npz",
                                        metadata=np.array(json.dumps(meta)),
                                        **{k: np.stack(v) for k, v in arrays.items()})
                    del feat
                    write_csv(OUT / "rows.csv", rows_all)
                    print(f"budget={budget} boundary={boundary} {kind}: {len(rows_all)}/1680 rows, "
                          f"elapsed={time.time() - t0:.1f}s", flush=True)
            del bundle
        write_json(OUT / "reproduction.json", reproduction)
        write_json(OUT / "excluded_states.json", excluded)
        prov.update(status="MEASURED", finished_utc=utc_now(), elapsed_seconds=time.time() - t0,
                    n_rows=len(rows_all), n_checkpoint_states=480, n_switch_states=840,
                    n_excluded_checkpoint_states=len(excluded),
                    n_excluded_switch_states=sum(r["excluded"] for r in rows_all if r["site"] == "f2"))
        write_json(OUT / "provenance.json", prov)
        CH.final_checks()
    except BaseException as exc:
        prov.update(status="FAILED", finished_utc=utc_now(), error=repr(exc), n_rows=len(rows_all))
        write_json(OUT / "provenance.json", prov)
        write_json(OUT / "reproduction.json", reproduction)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["measure"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    device = KP.NK.setup(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; use --device cpu explicitly if appropriate")
    measure(device, args.chunk)


if __name__ == "__main__":
    main()

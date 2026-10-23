#!/usr/bin/env python3
"""Implementation checks of mlp_center_x10_1020 (spec §7), run before the production run.

    python3 src/mlp_center_x10_1020_checks.py                 # -> results/mlp_center_x10_1020/checks/checks.json

Gating checks (all must pass): S-hook, S-diag, S-curve, S-units, S-probe, S-bound, S-reuse-early (R),
S-verdict.  Recorded, not gating: S-reuse-early for CH, and S-door-repro (which door makes CH leave
the cu126 trajectory).  No threshold here is a fixed tolerance: equalities are exact (bytes or the
%.10g strings 0920 writes), and the one inequality (eff_rank) uses the rounding bound derived in the spec.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H                    # noqa: E402
from src import rlcifar_mlp_battle_0918 as B        # noqa: E402
from src import cifar5p1_mlp_0920 as C              # noqa: E402
from src import mlp_center_x10_1020 as M            # noqa: E402

OUT = M.OUT_ROOT / "checks"
SEEDS_CHK = [100, 101]
S = M.STEPS


def fmt(v) -> str:
    return f"{float(v):.10g}"


def lower_median(x: np.ndarray) -> float:
    """torch.median's convention (the lower of the two middle values), which 0920's columns use."""
    return float(np.sort(x)[(len(x) - 1) // 2])


def rows(p: Path) -> list[dict]:
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def key(q):
    return int(q["seed"]), int(q["task"])


# --------------------------------------------------------------------------

def s_diag(dev, cif, tmp: Path) -> dict:
    """S-hook + S-diag: hooks on vs off, bytes equal; the parameter-writing mutant is caught."""
    res = {}
    for arm in ("R", "CHB0"):
        a, b = tmp / f"plain_{arm}", tmp / f"diag_{arm}"
        M.run_cell(arm, a, dev, seeds=SEEDS_CHK, n_tasks=3, cifar=cif, diag=False, allow_dirty=True,
                   progress=lambda m: None)
        pb = M.run_cell(arm, b, dev, seeds=SEEDS_CHK, n_tasks=3, cifar=cif, diag=True, allow_dirty=True,
                        progress=lambda m: None)
        same = {f: (a / f).read_bytes() == (b / f).read_bytes() for f in ("per_task.csv", "fresh_control.csv")}
        calls = pb["hook_calls"]
        want = {"begin_step_train_task": 3 * S + S, "begin_step_other": 4, "evaluate": 3}
        res[arm] = {"bytes_equal": same, "hook_calls": calls, "hook_calls_expected": want,
                    "ok": all(same.values()) and calls == want}
    m = tmp / "mut_CHB0"
    M.run_cell("CHB0", m, dev, seeds=SEEDS_CHK, n_tasks=3, cifar=cif, diag=True, mutate="touch_b3",
               allow_dirty=True, progress=lambda m_: None)
    caught = (m / "per_task.csv").read_bytes() != (tmp / "diag_CHB0" / "per_task.csv").read_bytes()
    res["mutant_touch_b3_caught"] = caught
    res["pass"] = res["R"]["ok"] and res["CHB0"]["ok"] and caught
    return res


def s_units(tmp: Path) -> dict:
    """The per-unit arrays on the task data, reduced the 0920 way, give the per_task.csv strings."""
    out, bad = {}, Counter()
    for arm in ("R", "CHB0"):
        d = tmp / f"diag_{arm}"
        z = np.load(d / "diag_units.npz")
        slot = {int(s): r for r, s in enumerate(z["seeds"])}
        n_bins_bad = 0
        for q in rows(d / "per_task.csv"):
            s, t = key(q)
            r, ti = slot[s], t - 1
            n = int(z["n_task"][ti])
            for li in (1, 2):
                zb, zs, cnt = z[f"task_zbar_l{li}"][ti, r], z[f"task_zsd_l{li}"][ti, r], z[f"task_open_l{li}"][ti, r]
                got = {f"zbar_l{li}": fmt(lower_median(zb)), f"zsd_l{li}": fmt(lower_median(zs)),
                       f"zbar_min_l{li}": fmt(zb.min()),
                       f"dead_frac_l{li}": fmt(torch.from_numpy((cnt == 0).astype(np.float32)).mean()),
                       f"eff_rank_l{li}": fmt(z[f"task_eff_rank_l{li}"][ti, r]),
                       f"w_norm_l{li}": fmt(lower_median(z[f"wnorm{li}"][ti, r]))}
                if f"m_l{li}" in z.files:
                    got[f"m_l{li}"] = fmt(torch.from_numpy(z[f"m_l{li}"][ti, r]).cuda().mean())
                for c, v in got.items():
                    bad[c] += int(q[c] != v)
                tot = sum(m.astype(int) for m in M.pplus_bins(cnt, n).values())
                n_bins_bad += int((tot != 1).any())
            bad["w_norm_l3"] += int(q["w_norm_l3"] != fmt(lower_median(z["wnorm3"][ti, r])))
        out[arm] = {"rows": len(rows(d / "per_task.csv")), "bins_not_exhaustive": n_bins_bad}
    out["mismatches_by_column"] = dict(bad)
    out["pass"] = sum(bad.values()) == 0 and all(out[a]["bins_not_exhaustive"] == 0 for a in ("R", "CHB0"))
    return out


def s_probe(tmp: Path, cif, dev) -> dict:
    zr = np.load(tmp / "diag_R" / "diag_units.npz")
    zc = np.load(tmp / "diag_CHB0" / "diag_units.npz")
    idx = zr["probe_idx"]
    ty = cif.test_y.numpy()
    per_seed = []
    for r in range(len(SEEDS_CHK)):
        cls = Counter(ty[idx[r]].tolist())
        per_seed.append({"n": int(len(idx[r])), "unique": int(len(set(idx[r].tolist()))),
                         "classes": len(cls), "per_class": sorted(set(cls.values())),
                         "in_test_split": bool((idx[r] >= 0).all() and (idx[r] < len(ty)).all())})
    # the inputs the hook used: the arm's own X_test
    Xs = cif.inputs("test", "std", dev)
    dc = cif.inputs("train", "std", dev).mean(0)              # the engine's door-C vector, same expression
    ii = [torch.from_numpy(idx[r]).long().to(dev) for r in range(len(idx))]
    Xc = Xs - dc                                               # the engine's `X_test = X_test - dc`
    fp_r = torch.stack([Xs[i] for i in ii]).double().sum((1, 2)).cpu().numpy()    # the hook's expression
    fp_c = torch.stack([Xc[i] for i in ii]).double().sum((1, 2)).cpu().numpy()
    ok_seed = all(p["n"] == 1200 and p["unique"] == 1200 and p["classes"] == 100 and p["per_class"] == [12]
                  and p["in_test_split"] for p in per_seed)
    res = {"per_seed": per_seed, "same_across_arms": bool(np.array_equal(idx, zc["probe_idx"])),
           "differs_across_seeds": bool(not np.array_equal(idx[0], idx[1])),
           "fingerprint_R_is_std_test": bool(np.array_equal(fp_r, zr["probe_fingerprint"])),
           "fingerprint_CHB0_is_std_test_minus_train_mean": bool(np.array_equal(fp_c, zc["probe_fingerprint"])),
           "fingerprint_CHB0_differs_from_R": bool(not np.array_equal(zr["probe_fingerprint"], zc["probe_fingerprint"]))}
    res["pass"] = ok_seed and all(v for k, v in res.items() if k != "per_seed")
    return res


def s_curve(dev, cif, tmp: Path) -> dict:
    """Eager run with the engine's own per-step record; tasks off the 20-update grid (790 and 130)."""
    sh, se = 790, 130
    dbg: dict = {}
    d = tmp / "curve_R"
    prov = M.run_cell("R", d, dev, seeds=SEEDS_CHK, n_tasks=2, steps_hard=sh, steps_easy=se, cifar=cif,
                      diag=True, debug=dbg, allow_dirty=True, progress=lambda m: None)
    z = np.load(d / "diag_units.npz")
    hits = torch.stack(dbg["online"]).numpy()                  # (790 + 130 + 790 fresh, R) float32
    segs = {"task1": hits[:sh], "task2": hits[sh:sh + se], "fresh": hits[sh + se:]}
    bad = 0
    for name, h in segs.items():
        cum = np.zeros(h.shape[1], np.float32)
        snaps = []
        for j in range(len(h)):
            cum = (cum + h[j]).astype(np.float32)
            if (j + 1) % M.SNAP == 0 and (j + 1) < len(h):
                snaps.append(cum.copy())
        if name == "fresh":
            got, fin = z["fresh_curve_cum"], z["fresh_curve_final"]
        else:
            ti = 0 if name == "task1" else 1
            got, fin = z["curve_cum"][ti], z["curve_final"][ti]
        snaps = np.array(snaps).T
        bad += int(not np.array_equal(got[:, :snaps.shape[1]].astype(np.float32), snaps))
        bad += int(not np.isnan(got[:, snaps.shape[1]:]).all())       # nothing written off the task
        bad += int(not np.array_equal(fin.astype(np.float32), cum))
    pt = {key(q): q for q in rows(d / "per_task.csv")}
    dp = {key(q): q for q in rows(d / "diag_per_task.csv")}
    online_same = all(pt[k]["online_acc"] == dp[k]["online_all"] for k in pt)
    fr = rows(d / "fresh_control.csv")
    slot = {int(s): r for r, s in enumerate(z["seeds"])}
    fresh_same = all(q["fresh_online_acc"] == fmt(float(z["fresh_curve_final"][slot[int(q["seed"])]]) / sh) for q in fr)
    calls_ok = prov["hook_calls"] == {"begin_step_train_task": 2 * sh + se, "begin_step_other": 0, "evaluate": 2}
    return {"segments_checked": list(segs), "mismatched_segments": bad, "online_all_equals_per_task": online_same,
            "fresh_total_equals_fresh_control": fresh_same, "hook_calls_eager": prov["hook_calls"],
            "pass": bad == 0 and online_same and fresh_same and calls_ok}


def s_bound(dev, cif) -> dict:
    """B64 covers eff_rank recomputed with the Gram summed in other orders (CPU, chunked), incl. dead units."""
    g = torch.Generator().manual_seed(0)
    X = cif.inputs("train", "std", dev)[:2500]
    W = (torch.rand(100, 3072, generator=g) * 2 - 1).to(dev) / math.sqrt(3072)
    worst, rel_b, n = 0.0, 0.0, 0
    for shift in (0.0, -1.0, -3.0):                            # -3.0 (about 5 sd) leaves dead units
        z = X @ W.T + shift
        for a in (z.clamp(min=0), z.clamp(min=0) - z.clamp(min=0).mean(0)):
            er, b = M.eff_rank_bound(a)
            ad = a.double().cpu()
            alts = [ad.T @ ad, sum(ad[i:i + 100].T @ ad[i:i + 100] for i in range(0, len(ad), 100))]
            for gm in alts:
                ev = torch.linalg.eigvalsh(gm).clamp_min(0.0)
                s_ = ev.sqrt()
                p = s_ / s_.sum()
                alt = float(torch.exp(-(p * p.clamp_min(1e-300).log()).sum()))
                worst = max(worst, abs(alt - er) / b if b > 0 else math.inf)
                n += 1
            rel_b = max(rel_b, b / er)
    return {"comparisons": n, "max_diff_over_bound": worst, "max_bound_over_eff_rank": rel_b, "pass": worst <= 1.0}


def compare_rows(new: list[dict], old: list[dict], npz=None) -> dict:
    new, old = sorted(new, key=key), sorted(old, key=key)
    if [key(q) for q in new] != [key(q) for q in old]:
        return {"same_keys": False}
    cols = list(old[0])
    mism = Counter(c for o, q in zip(old, new) for c in cols if not c.startswith("eff_rank") and o[c] != q[c])
    out = {"same_keys": True, "rows": len(new), "mismatches_without_eff_rank": dict(mism)}
    if npz is not None:
        slot = {int(s): r for r, s in enumerate(npz["seeds"])}
        outside, maxrel = 0, 0.0
        for o, q in zip(old, new):
            s, t = key(o)
            for li in (1, 2):
                a, b = float(q[f"eff_rank_l{li}"]), float(o[f"eff_rank_l{li}"])
                bound = npz[f"task_eff_rank_b64_l{li}"][t - 1, slot[s]] + M.print_halfunit(a) + M.print_halfunit(b)
                outside += int(abs(a - b) > bound)
                maxrel = max(maxrel, abs(a - b) / abs(b))
        out.update({"eff_rank_outside_b64": outside, "eff_rank_max_rel_diff": maxrel})
    return out


def s_reuse_early(dev, cif, tmp: Path) -> dict:
    res = {}
    for arm in ("R", "CH"):
        d = tmp / f"reuse_{arm}"
        M.run_cell(arm, d, dev, seeds=M.SEEDS, n_tasks=2, fresh=False, cifar=cif, allow_dirty=True,
                   progress=lambda m: None)
        old = [q for q in rows(M.HOST_RESULTS / M.cell_name(arm) / "per_task.csv") if int(q["task"]) <= 2]
        r = compare_rows(rows(d / "per_task.csv"), old, np.load(d / "diag_units.npz"))
        r["trajectory_equal"] = r.get("same_keys", False) and not r["mismatches_without_eff_rank"]
        r["ok"] = r["trajectory_equal"] and r.get("eff_rank_outside_b64", 1) == 0
        res[arm] = r
    res["pass"] = res["R"]["ok"]                                # CH is recorded, not gating (spec §7)
    return res


def s_door_repro(dev, cif, tmp: Path) -> dict:
    """Which door leaves the cu126 trajectory: H and C alone at 780 updates vs 0920's doors cells, 2 tasks."""
    res = {}
    for arm in ("H", "C"):
        d = tmp / f"door780_{arm}"
        C.run(arm, M.SEEDS, "std", 2, dev, d, cifar=cif, fresh=False, progress=lambda m: None)
        old = [q for q in rows(M.HOST_RESULTS / f"{arm}_std_doors" / "per_task.csv") if int(q["task"]) <= 2]
        r = compare_rows(rows(d / "per_task.csv"), old)
        res[arm] = {"trajectory_equal": r["same_keys"] and not r["mismatches_without_eff_rank"],
                    "mismatched_columns": r["mismatches_without_eff_rank"]}
    X = cif.inputs("train", "std", dev)
    dc, dc64 = X.mean(0), X.double().mean(0)
    ulp = (torch.nextafter(dc.abs(), torch.full_like(dc, math.inf)) - dc.abs()).double()
    res["door_c_vector"] = {"max_ulps_from_float64_mean": float(((dc.double() - dc64).abs() / ulp).max()),
                            "elements_differing_gpu_vs_cpu_float32_mean": int((X.cpu().mean(0) != dc.cpu()).sum()),
                            "n": int(dc.numel())}
    return res


# --------------------------------------------------------------------------
# S-verdict: synthetic cells with a known answer
# --------------------------------------------------------------------------

def synth_cell(d: Path, arm: str, window: float, late780: float, width: float, sparse: int, rank: float,
               rng: np.random.Generator) -> None:
    d.mkdir(parents=True, exist_ok=True)
    seeds, nt, R = list(range(10)), 30, 10
    per, fresh = [], []
    for r, s in enumerate(seeds):
        for t in range(1, nt + 1):
            per.append({"arm": arm, "cond": "std", "seed": s, "slot": r, "task": t, "hard": t % 2,
                        "online_acc": window + 0.001 * s + (0.0 if t in (21, 23, 25, 27, 29) else 0.05),
                        "train_acc": 0.99, "test_acc": 0.5, "dead_frac_l2": 0.0, "zeroout_l2": 0.0,
                        "eff_rank_l2": 10.0, "w_norm_l1": 1.0})
        fresh.append({"arm": arm, "seed": s, "task": 29, "fresh_gap": 0.01})
    H.write_csv(d / "per_task.csv", per)
    H.write_csv(d / "fresh_control.csv", fresh)
    (d / "provenance.json").write_text(json.dumps({"torch": "synthetic", "git_hash": "synthetic"}))
    H.write_csv(d / "diag_per_task.csv", [{"seed": 0}])
    ng = S // M.SNAP
    cum = np.full((nt, R, ng), np.nan)
    base = np.arange(1, ng + 1) * M.SNAP * 0.5                 # online 0.5 everywhere ...
    cum[:] = base
    for t in (21, 23, 25, 27, 29):                             # ... except the late block's first 780
        cum[t - 1, :, 780 // M.SNAP - 1] = 780 * late780
    open_ = np.zeros((nt, R, 100), np.int32) + 600             # p+ = 0.5
    open_[:, :, :sparse] = 120                                 # p+ = 0.1, sparse detectors
    zsd = np.full((nt, R, 100), width, np.float32) * (1 + 0.01 * np.arange(R))[None, :, None]
    np.savez(d / "diag_units.npz", seeds=np.array(seeds), live=np.ones((nt, R), bool), steps=np.full(nt, S),
             curve_cum=cum, curve_final=np.full((nt, R), S * 0.5), n_probe=np.full(nt, 1200), n_task=np.full(nt, 2500),
             probe_zsd_l1=zsd, probe_zsd_l2=zsd, task_zsd_l2=zsd, probe_open_l1=open_, probe_open_l2=open_,
             task_open_l1=open_, task_open_l2=open_,
             probe_eff_rank_l1=np.full((nt, R), rank) + rng.uniform(0, 0.1, (nt, R)),
             probe_eff_rank_l2=np.full((nt, R), rank) + rng.uniform(0, 0.1, (nt, R)),
             b2=np.zeros((nt, R, 100), np.float32), wnorm2=np.ones((nt, R, 100), np.float32))


def s_verdict(tmp: Path) -> dict:
    sys.path.insert(0, str(H.REPO / "analysis" / "mlp_center_x10_1020"))
    import verdict as V                                         # noqa: E402
    rng = np.random.default_rng(0)
    cases = {
        # arm: (window, late first-780, width, sparse units of 100, rank)
        "dose": ({"R": (0.90, 0.80, 10, 20, 30), "CH": (0.85, 0.70, 15, 5, 40), "CHB0": (0.80, 0.65, 25, 0, 45)},
                 {"overall": "DOSE", "W_CHB0_ge_2R": 10, "O_CH": "OPENS",
                  "preds": {"P1": True, "P2": True, "P3": True, "P4": True, "P5": True}}),
        "one_worse": ({"R": (0.90, 0.5, 10, 20, 30), "CH": (0.95, 0.5, 10, 20, 20), "CHB0": (0.80, 0.5, 10, 20, 20)},
                      {"overall": "ONE_WORSE", "W_CHB0_ge_2R": 0, "O_CH": "FLAT",
                       "preds": {"P1": False, "P2": False, "P3": False, "P4": False, "P5": False}}),
        "not_worse": ({"R": (0.80, 0.5, 10, 20, 30), "CH": (0.85, 0.6, 10, 20, 25), "CHB0": (0.90, 0.6, 10, 20, 25)},
                      {"overall": "NOT_WORSE", "W_CHB0_ge_2R": 0, "O_CH": "CLOSES",
                       "preds": {"P1": False, "P2": False, "P3": False, "P4": False, "P5": False}}),
        "both_worse": ({"R": (0.90, 0.5, 10, 20, 30), "CH": (0.80, 0.5, 10, 20, 25), "CHB0": (0.85, 0.5, 10, 20, 25)},
                       {"overall": "BOTH_WORSE", "W_CHB0_ge_2R": 0, "O_CH": "FLAT",
                        "preds": {"P1": False, "P2": False, "P3": False, "P4": False, "P5": False}}),
    }
    out = {}
    for name, (spec, want) in cases.items():
        root = tmp / f"synth_{name}"
        for arm, (w, l7, wd, sp, rk) in spec.items():
            synth_cell(root / M.cell_name(arm), arm, w, l7, wd, sp, rk, rng)
        got = V.compute(root, None)
        V.summary(got)                                          # the writer must not raise
        g = {"overall": got["overall"], "W_CHB0_ge_2R": got["W"]["CHB0_ge_2R"], "O_CH": got["O"]["CH-R"]["label"],
             "preds": got["predictions_claude_fable"]}
        out[name] = {"got": g, "want": want, "ok": g == want}
    out["pass"] = all(v["ok"] for v in out.values())
    return out


def main() -> None:
    torch.set_num_threads(M.THREADS)
    dev = H.setup("auto")
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="mcx_checks_"))
    cif = M.load_cifar()
    res = {}
    try:
        for name, fn in (("S-diag", lambda: s_diag(dev, cif, tmp)), ("S-units", lambda: s_units(tmp)),
                         ("S-probe", lambda: s_probe(tmp, cif, dev)), ("S-curve", lambda: s_curve(dev, cif, tmp)),
                         ("S-bound", lambda: s_bound(dev, cif)), ("S-reuse-early", lambda: s_reuse_early(dev, cif, tmp)),
                         ("S-door-repro", lambda: s_door_repro(dev, cif, tmp)), ("S-verdict", lambda: s_verdict(tmp))):
            t1 = time.time()
            res[name] = fn()
            res[name]["seconds"] = round(time.time() - t1, 1)
            print(f"{name}: {'PASS' if res[name].get('pass', True) else 'FAIL'} ({res[name]['seconds']} s)", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    gating = ("S-diag", "S-units", "S-probe", "S-curve", "S-bound", "S-reuse-early", "S-verdict")
    doc = {"scope": "implementation checks only; no performance verdict", "gating": list(gating),
           "all_pass": all(res[k]["pass"] for k in gating), "checks": res, "git": B.git_state(),
           "dirty_src_analysis_specs": M.git_dirty(), "torch": torch.__version__, "device": str(dev),
           "torch_threads": torch.get_num_threads(), "spec": M.SPEC, "seconds": round(time.time() - t0, 1)}
    (OUT / "checks.json").write_text(json.dumps(doc, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"all_pass {doc['all_pass']}  -> {OUT / 'checks.json'}")
    sys.exit(0 if doc["all_pass"] else 1)


if __name__ == "__main__":
    main()

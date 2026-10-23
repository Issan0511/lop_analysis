#!/usr/bin/env python3
"""Measurement checks of depth_silu_rlmnist_1020 (spec section 6): S-ident (with the drop-cross mutant), S-count,
S-neff, S-wall (with the layer-1 constancy).  Smoke runs (3 tasks) are made first with the engine CLI.

    OMP_NUM_THREADS=1 python analysis/depth_silu_rlmnist_1020/checks_measure.py --smoke raw/depth_silu_rlmnist_1020/smoke
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure as M
from src import depth_silu_rlmnist_1020 as E

OUT = ROOT / "results" / E.EXPERIMENT / "checks"


def write(name, rec):
    OUT.mkdir(parents=True, exist_ok=True)
    rec["check"] = name
    (OUT / f"{name}.json").write_text(json.dumps(rec, indent=1, default=float))
    print(name, "PASS" if rec["pass"] else "FAIL", flush=True)
    return rec["pass"]


def s_count():
    """Synthetic task-end p sequences [T, U] with known plain / deep returns and events."""
    seqs = {  # unit: (sequence, plain_unit, deep_unit, events)
        "never_below": ([0.9, 0.8, 0.7, 0.6], 0, 0, 0),
        "below_stays": ([0.4, 0.3, 0.2, 0.1], 0, 0, 0),
        "exact_half_counts_up": ([0.49, 0.5, 0.3, 0.3], 1, 0, 1),
        "starts_below_up": ([0.2, 0.6, 0.6, 0.6], 1, 0, 1),
        "flips_twice": ([0.3, 0.6, 0.3, 0.7], 1, 0, 2),
        "deep_return": ([0.05, 0.3, 0.55, 0.2], 1, 1, 1),
        "deep_at_0.1_then_up": ([0.6, 0.1, 0.5, 0.5], 1, 1, 1),
        "up_before_deep": ([0.4, 0.7, 0.05, 0.05], 1, 0, 1),
        "last_point_up": ([0.6, 0.6, 0.4, 0.5], 1, 0, 1),
    }
    p = np.array([v[0] for v in seqs.values()]).T            # [T, U]
    c, plain, deep = M.return_counts(p, 4)
    exp_plain = np.array([v[1] for v in seqs.values()], bool); exp_deep = np.array([v[2] for v in seqs.values()], bool)
    exp_events = sum(v[3] for v in seqs.values())
    ok = bool((plain == exp_plain).all() and (deep == exp_deep).all() and c["events"] == exp_events)
    # mutant: a counter with "> 0.5" instead of ">= 0.5" must disagree on the exact-half unit
    later_up_mut = p[1:] > 0.5
    below = np.maximum.accumulate(p < 0.5, axis=0)
    mut_plain = (below[:-1] & later_up_mut).any(0)
    return write("S-count", {"pass": ok and bool((mut_plain != exp_plain).any()), "counts": c, "plain": plain.tolist(),
                             "expected_plain": exp_plain.tolist(), "deep": deep.tolist(), "expected_deep": exp_deep.tolist(),
                             "events_expected": exp_events, "mutant_strict_gt_detected": bool((mut_plain != exp_plain).any())})


def s_wall():
    rng = np.random.default_rng(0)
    h = rng.normal(0.3, 1.0, size=(1200, 50)); z = h @ rng.normal(size=(50, 7))
    _, lay = M.layer_stats(h, z, "SILU")
    K = h @ h.mean(0) + 1.0
    ok1 = abs(lay["c"] - K.mean() / K.std()) <= 1e-12 * abs(lay["c"])
    hc = np.zeros((1200, 50)); _, lay0 = M.layer_stats(hc, hc @ np.ones((50, 3)) + 1.0, "SILU")
    ok2 = lay0["c"] == float("inf") and lay0["Ksd"] == 0.0
    return write("S-wall", {"pass": bool(ok1 and ok2), "c": lay["c"], "c_direct": float(K.mean() / K.std()), "c_const_K": lay0["c"]})


def s_runs(smoke: Path):
    ok_all = True
    for f in sorted(smoke.glob("*.npz")):
        if f.name.startswith("units_"):
            continue
        name, ident = M.measure_run(f, smoke)
        u = np.load(smoke / f"units_{name}.npz")
        d = np.load(f)
        L, act = int(d["L"]), str(d["act"])
        # identity: every unit of every window within the bound, layer-1 upstream and cross exactly 0
        ok_id = ident["n_resid_over_bound"] == 0 and ident["l1_up_absmax"] == 0.0 and ident["l1_cross_absmax"] == 0.0
        # mutant: drop the cross term -> most units of layers >= 2 exceed the bound
        mut_frac = []
        x64 = E.load_inputs(int(d["seed"]), str(d["door"]))[2].double().numpy()
        npar = 2 * L + 2
        for k in range(d["end0"].shape[0]):
            Pa = [d[f"mid{j}"][k].astype(np.float64) for j in range(npar)]
            Pb = [d[f"end{j}"][k].astype(np.float64) for j in range(npar)]
            ha, _ = M.forward64(Pa, x64, act, L); hb, _ = M.forward64(Pb, x64, act, L)
            for l in range(1, L):
                dec = M.decompose(Pa[2 * l], Pa[2 * l + 1], Pb[2 * l], Pb[2 * l + 1], ha[l].mean(0), hb[l].mean(0), drop_cross=True)
                mut_frac.append(float(np.mean(np.abs(dec["resid"]) > dec["bound"])))
        mut_ok = (L == 1) or (min(mut_frac) > 0.5)
        # layer-1 carrier constant over all points (the same x every time)
        consts = {}
        for key in ("hbar_norm", "Kmean", "Ksd", "c", "vin"):
            vals = np.concatenate([u[f"end_{key}_l1"], u[f"mid_{key}_l1"], [float(u[f"init_{key}_l1"])]])
            consts[key] = bool((vals == vals[0]).all())
        ok_c = all(consts.values())
        rec = {"pass": bool(ok_id and mut_ok and ok_c), "ident": ident, "mutant_drop_cross_min_frac_over_bound": min(mut_frac) if mut_frac else None,
               "layer1_carrier_constant": consts, "layer1_c": float(u["end_c_l1"][0]), "layer1_hbar_norm": float(u["end_hbar_norm_l1"][0])}
        if act == "R":
            eq = all(bool((u[f"{pt}_neff_l{l}"] == u[f"{pt}_p_l{l}"]).all()) for pt in ("end", "mid") for l in range(1, L + 1))
            rec["relu_neff_equals_p"] = eq; rec["pass"] = bool(rec["pass"] and eq)
        else:
            lo = min(float(np.nanmin(u[f"end_neff_l{l}"])) for l in range(1, L + 1)); hi = max(float(np.nanmax(u[f"end_neff_l{l}"])) for l in range(1, L + 1))
            rec["neff_range"] = [lo, hi]; rec["pass"] = bool(rec["pass"] and lo >= 1.0 / 1200 - 1e-15 and hi <= 1.0 + 1e-12)
        ok_all &= write(f"S-ident_{name}", rec)
    return ok_all


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", required=True)
    a = ap.parse_args()
    r = [s_count(), s_wall(), s_runs(Path(a.smoke))]
    print("ALL PASS" if all(r) else "SOME FAIL")

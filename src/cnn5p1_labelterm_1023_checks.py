#!/usr/bin/env python3
"""Registered checks (a)-(e); preflight gates full measurement on (a) and (c)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import cnn5p1_labelterm_1023 as M
import numpy as np
import torch
import torch.nn.functional as F


def check(name, passed, **values):
    return dict(name=name, status="PASS" if passed else "FAIL", passed=bool(passed), **values)


def reproduction_checks(bundle, feat, per_task, slots=None):
    old = {}
    with open(M.ROOT / "results/kappa_cnn5p1_1022/rows.csv") as f:
        for rec in csv.DictReader(f):
            if rec["role"] == "own":
                old[(int(rec["budget"]), int(rec["boundary"]), rec["cfg"], int(rec["seed"]))] = rec
    result = []
    for r in (range(len(bundle.slots)) if slots is None else slots):
        cfg, seed = bundle.slots[r]
        pt = per_task[(bundle.budget, cfg, seed, bundle.boundary)]
        kp = old[(bundle.budget, bundle.boundary, cfg, seed)]
        acc = float(feat["acc"][r])
        dm = float(np.median(M.KP.tonp(feat["diag_dmean"][r])))
        errors = {"train_acc_rel": M.relative_error([acc], [float(pt["train_acc"])]),
                  "train_dmean_rel": M.relative_error([dm], [float(pt["dmean_med_f2"])]),
                  "kappa_acc_rel": M.relative_error([acc], [float(kp["acc100"])]),
                  "kappa_dmean_rel": M.relative_error([dm], [float(kp["f2_dmean_med"])])}
        result.append(check("a_forward_reproduction", max(errors.values()) <= 1e-6,
                            budget=bundle.budget, boundary=bundle.boundary, cfg=cfg, seed=seed,
                            accuracy=acc, f2_dmean_median=dm, tolerance=1e-6, **errors))
    return result


def autograd_check(bundle, feat, slot, task, site, chunk):
    """Full-image mean CE gradients, accumulated in float64; never apply an update."""
    p = bundle.eng.P
    wi, bi = (6, 7) if site == "f2" else (4, 5)
    gw = torch.zeros_like(p[wi][slot], dtype=torch.float64)
    gb = torch.zeros_like(p[bi][slot], dtype=torch.float64)
    N = feat["labels"].shape[1]
    for start in range(0, N, chunk):
        if site == "f1":
            h = feat["a_f1"][:, start:start + chunk].detach()
            z1 = torch.baddbmm(p[5][:, None, :], h, p[4].transpose(1, 2))
            a1 = bundle.eng.act.phi(z1, 2)
        else:
            a1 = feat["a_f2"][:, start:start + chunk].detach()
        z2 = torch.baddbmm(p[7][:, None, :], a1, p[6].transpose(1, 2))
        a2 = bundle.eng.act.phi(z2, 3)
        logits = torch.baddbmm(p[9][:, None, :], a2, p[8].transpose(1, 2))
        loss = F.cross_entropy(logits[slot], feat["labels"][slot, start:start + chunk],
                               reduction="sum") / N
        dw, db = torch.autograd.grad(loss, (p[wi], p[bi]))
        gw += dw[slot].double()
        gb += db[slot].double()
    abar = feat[f"a_{site}"][slot].double().mean(0)
    delta = -(gw @ abar + gb)
    _, arr = M.quantities(bundle, feat, slot, site, task, do_permutations=False)
    analytic = -arr["S_real"]
    direct = M.KP.tonp(delta)
    rel = M.relative_error(direct, analytic)
    return check("c_autograd_position_change", rel <= 1e-6, site=site,
                 budget=bundle.budget, boundary=bundle.boundary, task=task,
                 cfg=bundle.slots[slot][0], seed=bundle.slots[slot][1], N=N,
                 relative_error=rel, max_abs_error=float(np.max(np.abs(direct - analytic))),
                 max_abs_reference=float(np.max(np.abs(direct))), tolerance=1e-6,
                 eta=1.0, sign_convention="delta_m = -eta * S_real", weights_updated=False,
                 delta_m=direct, minus_S_real=analytic)


def preflight(device, chunk):
    M.OUT.mkdir(parents=True, exist_ok=True)
    report = dict(experiment=M.EXPERIMENT, started_utc=M.utc_now(),
                  preflight=dict(passed=False, source_sha256=M.source_hashes(),
                                 device=str(device), chunk=chunk), checks=[])
    try:
        inventory = M.checkpoint_inventory([780], [9])
        data = M.KP.load_data(device)
        bundle = M.KP.Bundle(780, data, device)
        M.validate_bundle(bundle)
        bundle.load_state(9)
        M.validate_frozen_alpha(bundle)
        r = bundle.slots.index(("ELU_a3.6_lr0.0003", 2000))
        own = M.features(bundle, 9, chunk)
        a = reproduction_checks(bundle, own, M.KP.read_per_task(), slots=[r])[0]
        report["checks"].append(a)
        del own
        print(json.dumps(a), flush=True)
        if not a["passed"]:
            raise RuntimeError("preflight (a) failed; stop before full-state measurement")
        target = M.features(bundle, 11, chunk)
        c = autograd_check(bundle, target, r, 11, "f2", chunk)
        report["checks"].append(c)
        print(json.dumps({k: v for k, v in c.items() if k not in ("delta_m", "minus_S_real")}), flush=True)
        if not c["passed"]:
            raise RuntimeError("preflight (c) failed; stop before full-state measurement")
        c1 = autograd_check(bundle, target, r, 11, "f1", chunk)
        report["checks"].append(c1)
        print(json.dumps({k: v for k, v in c1.items() if k not in ("delta_m", "minus_S_real")}), flush=True)
        if not c1["passed"]:
            raise RuntimeError("FC1 supplementary gradient check failed")
        report["preflight"].update(passed=True, checkpoints=inventory, completed_utc=M.utc_now())
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        report["finished_utc"] = M.utc_now()
        M.write_json(M.OUT / "checks.json", report)


def permutation_check(meta, arrays, index):
    """Literal per-state comparison, including failures from finite permutations.

    'Near zero' was not numerically specified: use a declared 3-null-SD envelope for
    the median over alive units of their mean L across the five permutations.
    The exact null variance is for the fixed balanced label multiset.
    """
    live = arrays["alive"][index].astype(bool)
    lp = arrays["L_perm"][index][:, live]
    real = arrays["S_real"][index][live]
    rand = arrays["S_rand"][index][live]
    spr = arrays["S_real_perm"][index][:, live]
    sigma = arrays["L_perm_null_sd"][index][live] / np.sqrt(M.PERMUTATIONS)
    signed = float(np.median(lp.mean(0)))
    envelope = 3 * float(np.median(sigma))
    zero_pass = abs(signed) <= envelope + 1e-300
    real_agree = float(np.mean(np.sign(real) == np.sign(rand)))
    perm_agree = [float(np.mean(np.sign(v) == np.sign(rand))) for v in spr]
    agree_pass = np.mean(perm_agree) + 1e-15 >= real_agree
    return check("d_label_permutation", zero_pass and agree_pass, **meta,
                 n_alive=int(live.sum()), repetitions=M.PERMUTATIONS,
                 L_real_median=float(np.median(arrays["L"][index][live])),
                 L_real_abs_median=float(np.median(np.abs(arrays["L"][index][live]))),
                 L_perm_medians=np.median(lp, axis=1).tolist(),
                 L_perm_mean_unit_median=signed, near_zero_envelope=envelope,
                 near_zero_pass=zero_pass, agree_pass=bool(agree_pass),
                 agree_real=real_agree, agree_perm=perm_agree,
                 agree_perm_mean=float(np.mean(perm_agree)),
                 agree_difference=float(np.mean(perm_agree) - real_agree),
                 L_perm_abs_median=float(np.median(np.abs(lp))))


def final_checks():
    report = json.loads((M.OUT / "checks.json").read_text())
    if not report["preflight"]["passed"]:
        raise RuntimeError("preflight did not pass")
    with open(M.OUT / "rows.csv") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1680:
        raise RuntimeError(f"expected 1680 rows, found {len(rows)}")
    reproduction = json.loads((M.OUT / "reproduction.json").read_text())
    report["a"] = check("a_forward_reproduction", all(r["passed"] for r in reproduction),
                        n_states=len(reproduction), n_fail=sum(not r["passed"] for r in reproduction),
                        max_train_acc_rel=max(r["train_acc_rel"] for r in reproduction),
                        max_train_dmean_rel=max(r["train_dmean_rel"] for r in reproduction),
                        max_kappa_acc_rel=max(r["kappa_acc_rel"] for r in reproduction),
                        max_kappa_dmean_rel=max(r["kappa_dmean_rel"] for r in reproduction))
    b1 = max(float(r["identity_real_rel"]) for r in rows)
    b2 = max(float(r["identity_rand_rel"]) for r in rows)
    easy = [r for r in rows if r["kind"] == "easy"]
    easy_l = max(float(r["easy_L_absmax"]) for r in easy)
    report["b"] = check("b_identities", max(b1, b2) < 1e-10 and easy_l == 0,
                        n_rows=len(rows), real_identity_relmax=b1, rand_identity_relmax=b2,
                        easy_L_absmax=easy_l, tolerance=1e-10)
    cs = [r for r in report["checks"] if r["name"] == "c_autograd_position_change"]
    report["c"] = check("c_autograd_position_change", all(r["passed"] for r in cs),
                        f2_relative_error=next(r["relative_error"] for r in cs if r["site"] == "f2"),
                        f1_relative_error=next(r["relative_error"] for r in cs if r["site"] == "f1"),
                        tolerance=1e-6)
    center = {k: max(float(r[k]) for r in rows) for k in rows[0] if k.startswith("center_")}
    report["e"] = check("e_readout_centering", max(center.values()) < 1e-12,
                        n_rows=len(rows), tolerance=1e-12, **center)
    perm = []
    for fn in sorted((M.OUT / "units").glob("*_hard.npz")):
        with np.load(fn, allow_pickle=False) as ar:
            meta = json.loads(str(ar["metadata"]))
            for i, m in enumerate(meta):
                if not m["excluded"] and ar["alive"][i].any():
                    perm.append(permutation_check(m, ar, i))
    M.write_json(M.OUT / "permutation_checks.json", perm)
    summaries = {}
    for site in M.SITES:
        pr = [r for r in perm if r["site"] == site]
        summaries[site] = dict(n_states=len(pr), n_pass=sum(r["passed"] for r in pr),
                               n_near_zero_pass=sum(r["near_zero_pass"] for r in pr),
                               n_agree_pass=sum(r["agree_pass"] for r in pr),
                               agree_difference_median=float(np.median([r["agree_difference"] for r in pr])),
                               agree_difference_min=min(r["agree_difference"] for r in pr),
                               agree_difference_max=max(r["agree_difference"] for r in pr),
                               L_perm_mean_unit_median_absmax=max(abs(r["L_perm_mean_unit_median"]) for r in pr))
    report["d"] = check("d_label_permutation", summaries["f2"]["n_pass"] == summaries["f2"]["n_states"],
                        primary_site="f2", scope="all non-excluded hard switch states; FC1 recorded separately",
                        near_zero_convention="abs(median_i mean_perm L_i) <= 3 median_i(null_sd_i/sqrt(5))",
                        agree_convention="mean of five permutation agreements >= real agreement, per state",
                        specification_has_no_numeric_near_zero_tolerance=True, **summaries)
    report["all_checks_passed"] = all(report[k]["passed"] for k in "abcde")
    report["finished_utc"] = M.utc_now()
    M.write_json(M.OUT / "checks.json", report)
    for k in "abcde":
        print(k, json.dumps(report[k]), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["preflight", "all"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.cmd == "all":
        final_checks()
    else:
        device = M.KP.NK.setup(args.device)
        preflight(device, args.chunk)


if __name__ == "__main__":
    main()

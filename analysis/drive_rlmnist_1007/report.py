"""drive_rlmnist_1007 report, in four separate stages (spec 5, 8):

    python -m analysis.drive_rlmnist_1007.report validate  --run results/drive_rlmnist_1007/run
    python -m analysis.drive_rlmnist_1007.report calibrate --run results/drive_rlmnist_1007/run \
        --window results/drive_rlmnist_1007/window_calibration.json
    # commit window_calibration.json before the next command
    python -m analysis.drive_rlmnist_1007.report report    --run results/drive_rlmnist_1007/run \
        --window results/drive_rlmnist_1007/window_calibration.json --out results/drive_rlmnist_1007/report
    python -m analysis.drive_rlmnist_1007.report audit     --run results/drive_rlmnist_1007/run

validate reads only manifests, hashes, numerical-check counts and the host trajectory (no certificate
frequency or transport value is computed or printed).  calibrate reads the calibration seeds 5-9 only.
report refuses to run unless the window file's bytes are the committed HEAD version.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from src import drive_rlmnist_1007 as D
from analysis.drive_rlmnist_1007 import numerics as N
from analysis.drive_rlmnist_1007.stats import directional, certificate_status, window

ROOT = D.ROOT
SEEDS = D.MAIN_SEEDS + D.CAL_SEEDS
MAIN_TASKS = (2, 3, 4, 5)
TASKS = tuple(range(1, D.TASKS + 1))
L2CAP_RUNS = ROOT / "results" / "l2cap_ee_0917" / "runs"
L2CAP_UNITS = Path("/home/issan/Projects/obsidian-research-data/l2cap_ee_0917/results/l2cap_ee_0917/runs")
CHECKS = ROOT / "results" / D.RUN / "checks.json"
K = {k: i for i, k in enumerate(N.KEYS)}
U64 = 2.0 ** -53
TINY64 = float(np.finfo(np.float64).tiny)
EXPECTED_FILES = ({f"t{t:02d}.npz" for t in TASKS} | {"host_rows.json", "units.npz", "info.json", "task_end_sha256.json",
                  "cost.json", "input_manifest.json", "provenance_start.json"}
                  | {f"audit/t{t:02d}_e{e:03d}.pt" for t, e in D.AUDIT})


def report_source():
    return D.sha(Path(__file__))


def gamma(n, u=U64):
    return n * u / (1 - n * u)


# --------------------------------------------------------------------------
# stage 1: completeness, manifests, numerical checks, host identity (spec 5.5 A-C)
# --------------------------------------------------------------------------

def seed_dir(run, s):
    return Path(run) / f"s{s}"


def load_task(run, s, t):
    with np.load(seed_dir(run, s) / f"t{t:02d}.npz", allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    assert d["keys"].tolist() == list(N.KEYS) and d["l1_keys"].tolist() == list(N.L1_KEYS)
    assert int(d["seed"]) == s and int(d["task"]) == t and int(d["epochs"]) == D.EPOCHS
    assert int(d["steps_per_epoch"]) == D.SPE
    assert d["sum"].shape == (D.EPOCHS, 100, len(N.KEYS)) and d["max"].shape == d["sum"].shape
    assert d["l1"].shape == (D.EPOCHS, 100, len(N.L1_KEYS)) and d["aux"].shape == (D.EPOCHS, 100, len(N.AUX_KEYS))
    return d


def validate_seed(run, s, mode="production", tasks=D.TASKS, epochs=D.EPOCHS, sources=None):
    """Manifest and completeness of one seed's output (no scientific value is read)."""
    out = seed_dir(run, s)
    assert not (out / "CHECK_FAILED.json").exists(), f"seed {s}: CHECK_FAILED"
    m = json.loads((out / "complete.json").read_text())
    ident = json.loads((out / "input_manifest.json").read_text())
    assert m["identity"] == ident, f"seed {s}: identity mismatch"
    assert ident["run_id"] == D.RUN and ident["seed"] == s and ident["mode"] == mode, f"seed {s}: foreign run"
    assert ident["tasks"] == tasks and ident["epochs"] == epochs and ident["prereg_commit"] == D.PREREG_COMMIT
    assert ident["source_sha256"] == (sources or D.run_sources()), f"seed {s}: source changed since the run"
    assert m["tasks_completed"] == tasks, f"seed {s}: INCOMPLETE"
    paths = [f["path"] for f in m["files"]]
    assert len(paths) == len(set(paths))
    expected = ({f"t{t:02d}.npz" for t in range(1, tasks + 1)} | {"host_rows.json", "units.npz", "info.json",
                "task_end_sha256.json", "cost.json", "input_manifest.json", "provenance_start.json"}
                | {f"audit/t{t:02d}_e{e:03d}.pt" for t, e in D.AUDIT if t <= tasks and e <= epochs})
    extra = set(paths) - expected
    assert expected <= set(paths) and extra <= {"first_failure.pt"}, f"seed {s}: missing or foreign files"
    for f in m["files"]:
        p = out / f["path"]
        assert p.resolve().is_relative_to(out.resolve()) and D.sha(p) == f["sha256"], f"seed {s}: checksum {f['path']}"
    assert json.loads((out / "provenance_end.json").read_text())["status"] == "COMPLETE"
    return m


def numerical_status(run, s):
    """Spec 5.5 (B)(C): finiteness and zero numerical failures in tasks 1-5; counts in 6-10."""
    res = {}
    for t in TASKS:
        d = load_task(run, s, t)
        a = d["sum"]
        finite = bool(np.isfinite(a).all() and np.isfinite(d["max"]).all() and np.isfinite(d["l1"]).all())
        cover = bool((a[:, :, K["cert_down"]] + a[:, :, K["cert_up"]] + a[:, :, K["uncertain"]] == D.SPE).all())
        res[t] = dict(finite=finite, partition=cover, fail=float(a[:, :, K["fail"]].sum()),
                      nonfinite=float(a[:, :, K["nonfinite"]].sum()),
                      violation=float(a[:, :, K["certificate_violation"]].sum()),
                      bracket=float(d["aux"][:, :, 0].sum()), fail_count=int(d["fail_count"]))
    return res


def _eq(a, b):
    if isinstance(a, float) or isinstance(b, float):
        a, b = float(a), float(b)
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


def record_identity(s, rows, units, info, tend, host_tend, tasks=TASKS):
    """Spec 2 / 5.5 (A) in memory: tasks 1-5 against the l2cap ref shard (per_task rows, units arrays,
    task-1 sha) and the unmodified host's task-end sha256; tasks 6-10 REPORT_ONLY.
    units: dict-like of the run's arrays; tend/host_tend: {str(task): sha256}."""
    tasks = list(tasks)
    with (L2CAP_RUNS / f"ref_s{s}" / "per_task.csv").open() as f:
        rec = {int(r["task"]): r for r in csv.DictReader(f)}
    prov = json.loads((L2CAP_RUNS / f"ref_s{s}" / "provenance.json").read_text())["per_seed"][str(s)]

    def row_ok(r):
        rr = rec[r["task"]]
        if set(rr) != set(r):
            return False
        for k, v in r.items():
            if isinstance(v, str):
                if rr[k] != v:
                    return False
            elif isinstance(v, int) and not isinstance(v, bool):
                if rr[k] != str(v):
                    return False
            elif not _eq(float(rr[k]), v):
                return False
        return True
    row_pass = {r["task"]: row_ok(r) for r in rows}
    with np.load(L2CAP_UNITS / f"ref_s{s}" / "units.npz") as rec_u:
        task = np.asarray(units["task"])
        unit_pass = {}
        for t in tasks:
            sel = np.nonzero(task == t)[0]
            ok = len(sel) > 0
            for k in units.keys():
                if k in ("q_cap", "v_cap") or k.startswith("test_"):
                    continue
                ok = ok and np.array_equal(np.asarray(units[k])[sel], rec_u[f"s{s}_{k}"][sel], equal_nan=True)
            unit_pass[t] = bool(ok)
        test_ok = all(np.array_equal(np.asarray(units[k]), rec_u[f"s{s}_{k}"][:len(units[k])], equal_nan=True)
                      for k in units.keys() if k.startswith("test_"))
        cap_ok = all(np.array_equal(np.asarray(units[k]), rec_u[f"s{s}_{k}"], equal_nan=True) for k in ("q_cap", "v_cap"))
    first = [t for t in tasks if t <= 5]
    res = dict(rows={str(t): row_pass.get(t, False) for t in tasks}, units={str(t): unit_pass[t] for t in tasks},
               test_arrays=bool(test_ok), caps=bool(cap_ok),
               task1_sha=info["task1_end_state_sha256"] == prov["task1_end_state_sha256"],
               init_sha=info["init_sha256"] == prov["init_sha256"], subset_sha=info["subset_idx_sha256"] == prov["subset_idx_sha256"],
               task_end_vs_unmodified_host={str(t): tend.get(str(t)) == host_tend[str(t)] for t in first})
    res["pass_tasks_1_5"] = bool(all(res["rows"][str(t)] and res["units"][str(t)] for t in first) and len(first) == 5
                                 and res["task1_sha"] and res["init_sha"] and res["subset_sha"] and res["caps"]
                                 and res["test_arrays"] and all(res["task_end_vs_unmodified_host"].values()))
    later = [t for t in tasks if t >= 6]
    res["report_only_tasks_6_10"] = (bool(all(res["rows"][str(t)] and res["units"][str(t)] for t in later))
                                     if later else None)
    return res


def host_identity(run, s, host_task_end_sha):
    out = seed_dir(run, s)
    rows = json.loads((out / "host_rows.json").read_text())
    info = json.loads((out / "info.json").read_text())
    tend = json.loads((out / "task_end_sha256.json").read_text())
    with np.load(out / "units.npz") as u:
        units = {k: u[k] for k in u.files}
    return record_identity(s, rows, units, info, tend, host_task_end_sha[str(s)], TASKS)


def validate(run, destination=None):
    run = Path(run)
    checks = json.loads(CHECKS.read_text())
    host_sha = checks["host_task_end_sha256"]
    status, ident, nums = {}, {}, {}
    for s in SEEDS:
        validate_seed(run, s)
        nums[s] = numerical_status(run, s)
        ident[s] = host_identity(run, s, host_sha)
    A = all(ident[s]["pass_tasks_1_5"] for s in SEEDS)
    B = all(nums[s][t]["finite"] and nums[s][t]["partition"] for s in SEEDS for t in TASKS)
    C = all(nums[s][t]["fail"] == 0 and nums[s][t]["nonfinite"] == 0 and nums[s][t]["violation"] == 0
            for s in SEEDS for t in range(1, 6))
    status = dict(A_host_identity=A, B_complete_finite=B, C_numerical=C,
                  report_only_6_10_identity={str(s): ident[s]["report_only_tasks_6_10"] for s in SEEDS},
                  numerical_6_10={str(s): {str(t): nums[s][t] for t in range(6, 11)} for s in SEEDS},
                  bracket_violations_all={str(s): sum(nums[s][t]["bracket"] for t in TASKS) for s in SEEDS},
                  complete_sha256={str(s): D.sha(seed_dir(run, s) / "complete.json") for s in SEEDS},
                  checks_sha256=D.sha(CHECKS), report_source=report_source())
    status["label"] = ("CHECK_FAILED" if not (A and C) else "INCOMPLETE" if not B else "VALID")
    if destination is not None:
        D.put(Path(destination) / "host_identity.json", {str(s): ident[s] for s in SEEDS})
        D.put(Path(destination) / "production_validation.json", status)
    return status


# --------------------------------------------------------------------------
# stage 2: calibration seeds only (spec 5.3)
# --------------------------------------------------------------------------

def calibration_series(run, seeds):
    if tuple(seeds) != D.CAL_SEEDS:
        raise ValueError("the window is calibrated on seeds 5-9 only")
    v = np.zeros(D.EPOCHS)
    files = {}
    for t in MAIN_TASKS:
        for s in seeds:
            d = load_task(run, s, t)
            files[f"s{s}/t{t:02d}.npz"] = D.sha(seed_dir(run, s) / f"t{t:02d}.npz")
        # mean over seeds and units of the per-epoch total, per update; tasks weighted equally
        v += np.stack([load_task(run, s, t)["sum"][:, :, K["total_sum"]] for s in seeds]).mean((0, 2)) / D.SPE / len(MAIN_TASKS)
    return v, files


def calibrate(run, destination, seeds=D.CAL_SEEDS):
    val = json.loads((ROOT / "results" / D.RUN / "production_validation.json").read_text())
    assert val["label"] == "VALID", "calibrate only a validated run"
    v, files = calibration_series(run, seeds)
    w = window(v)
    w.update(run_id=D.RUN, calibration_seeds=list(seeds), tasks=list(MAIN_TASKS), input_sha256=files,
             complete_sha256={str(s): val["complete_sha256"][str(s)] for s in seeds},
             report_source=report_source(), v=v.tolist())
    D.put(destination, w)
    return w["label"]


# --------------------------------------------------------------------------
# stage 3: the registered verdicts (spec 5), only after the window commit
# --------------------------------------------------------------------------

def committed_window(window_path):
    window_path = Path(window_path).resolve()
    rel = window_path.relative_to(ROOT)
    committed = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=ROOT)
    assert committed == window_path.read_bytes(), "commit the calibration window before opening the main seeds"
    return json.loads(committed)


def window_sums(a, lo, hi):
    """Sum of the per-epoch reductions over epochs lo..hi-1 (0-based), all units -> (K,)"""
    return a[lo:hi].sum((0, 1))


TRANSPORT = ("S", "U", "total", "S_conf", "S_label", "S_hist")


def transport_row(v, n_updates):
    r = {}
    for k in TRANSPORT:
        for part in ("sum", "positive", "negative", "positive_count", "negative_count"):
            r[f"{k}_{part}"] = float(v[K[f"{k}_{part}"]])
        r[f"{k}_sum_per_update"] = r[f"{k}_sum"] / n_updates
    for k in ("cert_down", "cert_up", "uncertain", "cert_total_down", "raw_down", "raw_up", "certificate_violation", "fail"):
        r[k] = float(v[K[k]])
    r["unit_updates"] = 100 * n_updates
    r["p_down"] = r["cert_down"] / r["unit_updates"]
    r["p_up"] = r["cert_up"] / r["unit_updates"]
    for k in ("T", "R", "H", "Qminus", "Qplus", "SGD_T", "SGD_R", "native_closure", "grad_defect", "adam_defect"):
        r[f"{k}_mean"] = float(v[K[k]]) / r["unit_updates"]
    return r


def telescope(run, s, tasks):
    """Spec 5.5 (D): per unit, sum of total over the record's dense windows vs the zbar_l2 difference."""
    pts = list(D.EL.DENSE)
    worst, fails, n = 0.0, 0, 0
    with np.load(seed_dir(run, s) / "units.npz") as u:
        task, step, zbar = u["task"], u["step"], u["zbar_l2"]
        for t in tasks:
            a = load_task(run, s, t)["sum"]
            for p0, p1 in zip(pts[:-1], pts[1:]):
                i0 = np.nonzero((task == t) & (step == p0))[0]
                i1 = np.nonzero((task == t) & (step == p1))[0]
                assert len(i0) == 1 and len(i1) == 1
                e0, e1 = p0 // D.SPE, p1 // D.SPE
                tot = a[e0:e1, :, K["total_sum"]].sum(0)
                clo = a[e0:e1, :, K["native_closure"]].sum(0)
                absum = (a[e0:e1, :, K["total_positive"]] - a[e0:e1, :, K["total_negative"]]).sum(0)
                N_ = p1 - p0
                bound = (1 + gamma(N_ + 2)) * clo + gamma(N_ + 2) * absum + N_ * TINY64
                diff = np.abs(tot - (zbar[i1[0]] - zbar[i0[0]]))
                fails += int((diff > bound).sum())
                worst = max(worst, float((diff / np.where(bound > 0, bound, np.inf)).max()))
                n += 100
    return dict(units_windows=n, outside_bound=fails, worst_ratio=worst)


def brier_multi(prob, label):
    return sum((p - (k == label)) ** 2 for k, p in prob.items())


def brier(p, event):
    return (p - float(event)) ** 2


PREDICTIONS = {
    "parent_Fable": dict(M1x={"NOT_SUPPORTED": .60, "CROSS_TASK_SINK_CONDITION_HOLDS": .25, "UNRESOLVED": .15},
                         window=.70, M3x_boundary=.75, U_all_negative=.70, S_all_negative=.60,
                         first_epoch_S_majority=.55, M2x_consistent=None),
    "implementer_Claude": dict(M1x={"NOT_SUPPORTED": .50, "CROSS_TASK_SINK_CONDITION_HOLDS": .25, "UNRESOLVED": .25},
                               window=.80, M3x_boundary=.85, U_all_negative=.80, S_all_negative=.75,
                               first_epoch_S_majority=.60, M2x_consistent=.95)}


def compute(run, b, seeds=SEEDS, main_seeds=D.MAIN_SEEDS):
    """All registered readouts from the reduced shards, given the calibrated break b (epochs, or None)."""
    run = Path(run)
    per_task, per_seed, by_window, per_unit, m3 = [], [], [], [], []
    for s in seeds:
        main = np.zeros(len(N.KEYS))
        first = np.zeros(len(N.KEYS))
        unit = np.zeros((100, len(N.KEYS)))
        l1 = np.zeros(len(N.L1_KEYS))
        bdiff = []
        for t in TASKS:
            d = load_task(run, s, t)
            a = d["sum"]
            row = dict(seed=s, task=t, role=D.role(s), window="all", **transport_row(window_sums(a, 0, D.EPOCHS), D.SPE * D.EPOCHS))
            row["bracket_violation"] = float(d["aux"].sum())
            row.update({k: float(d["l1"][:, :, i].sum()) for i, k in enumerate(N.L1_KEYS)})
            per_task.append(row)
            wins = [("first_epoch", 0, 1), ("rest_79", 1, D.EPOCHS)]
            if b:
                wins += [("boundary", 0, b), ("later", b, D.EPOCHS)]
            for name, lo, hi in wins:
                by_window.append(dict(seed=s, task=t, role=D.role(s), window=name, epochs=f"{lo + 1}-{hi}",
                                      **transport_row(window_sums(a, lo, hi), D.SPE * (hi - lo))))
            if t in MAIN_TASKS:
                main += window_sums(a, 0, D.EPOCHS)
                first += window_sums(a, 0, 1)
                unit += a.sum(0)
                l1 += d["l1"].sum((0, 1))
                if b:
                    pb = a[:b, :, K["cert_down"]].sum() / (b * D.SPE * 100)
                    pl = a[b:, :, K["cert_down"]].sum() / ((D.EPOCHS - b) * D.SPE * 100)
                    bdiff.append(pb - pl)
        r = dict(seed=s, role=D.role(s), **transport_row(main, D.SPE * D.EPOCHS * len(MAIN_TASKS)))
        r["D"] = r["p_down"] - r["p_up"]
        r["p_unresolved"] = r["uncertain"] / r["unit_updates"]
        r["first_epoch_S_sum"] = float(first[K["S_sum"]])
        r["first_epoch_S_share"] = r["first_epoch_S_sum"] / r["S_sum"] if r["S_sum"] != 0 else float("nan")
        r["boundary_minus_later_p_down"] = float(np.mean(bdiff)) if b else float("nan")
        r.update({f"L1_{k}": float(l1[i]) for i, k in enumerate(N.L1_KEYS)})
        tel = telescope(run, s, range(2, D.TASKS + 1))
        r.update({f"telescope_{k}": v for k, v in tel.items()})
        per_seed.append(r)
        if s in main_seeds:
            per_unit.append(unit)
            m3.append(r["boundary_minus_later_p_down"])
    main_rows = [r for r in per_seed if r["seed"] in main_seeds]
    m1 = directional([r["D"] for r in main_rows])
    m2 = certificate_status(sum(r["cert_down"] + r["cert_up"] for r in main_rows),
                            sum(r["certificate_violation"] for r in main_rows))
    m3x = directional(m3, "BOUNDARY_ENRICHED", "LATER_ENRICHED") if b else dict(label="WINDOW_NOT_IDENTIFIED")
    m4 = dict(U_main_sum_all_negative=all(r["U_sum"] < 0 for r in main_rows),
              S_main_sum_all_negative=all(r["S_sum"] < 0 for r in main_rows),
              first_epoch_S_share_gt_half_all=all(r["first_epoch_S_sum"] / r["S_sum"] > .5 if r["S_sum"] != 0 else False
                                                  for r in main_rows))
    return dict(per_seed=per_seed, per_task=per_task, by_window=by_window, per_unit=per_unit,
                M1x=m1, M2x=m2, M3x=m3x, M4x=m4,
                violations_all=float(sum(r["certificate_violation"] for r in per_task)))


def report(run, window_path, destination):
    run, destination = Path(run), Path(destination)
    val = json.loads((ROOT / "results" / D.RUN / "production_validation.json").read_text())
    w = committed_window(window_path)
    for s in D.CAL_SEEDS:
        assert w["complete_sha256"][str(s)] == D.sha(seed_dir(run, s) / "complete.json"), "window from another run"
    applicable = val["label"] == "VALID"
    b = w["break_epoch"]
    res = compute(run, b)
    m1, m2, m3x, m4 = res["M1x"], res["M2x"], res["M3x"], res["M4x"]
    if not applicable:
        lab = val["label"]
        m1, m3x = dict(label=lab, suppressed=m1), dict(label=lab, suppressed=m3x)
        m2 = lab
    verdict = dict(run_id=D.RUN, applicability=dict(A=val["A_host_identity"], B=val["B_complete_finite"], C=val["C_numerical"],
                   label=val["label"]),
                   M1x=m1, M2x=m2, M2x_violations_all_seeds_all_tasks=res["violations_all"],
                   M3x=m3x, window=dict(label=w["label"], break_epoch=b, boundary_updates=w["boundary_updates"]),
                   M4x=m4, window_sha256=D.sha(Path(window_path)), report_source=report_source(),
                   report_source_at_calibration=w["report_source"],
                   limitations="observational sufficient condition on the second layer's self movement; five independent "
                               "seeds; Random-Label MNIST ELU->ELU ref arm, 10 tasks; no causal or net-total-sink inference")
    preds = {}
    for who, p in PREDICTIONS.items():
        sc = dict(M1x=brier_multi(p["M1x"], m1["label"]) if applicable else None,
                  window=brier(p["window"], bool(b)) if applicable else None,
                  M3x=brier(p["M3x_boundary"], m3x["label"] == "BOUNDARY_ENRICHED") if (applicable and b) else None,
                  U_all_negative=brier(p["U_all_negative"], m4["U_main_sum_all_negative"]) if applicable else None,
                  S_all_negative=brier(p["S_all_negative"], m4["S_main_sum_all_negative"]) if applicable else None,
                  first_epoch_S_majority=(brier(p["first_epoch_S_majority"], m4["first_epoch_S_share_gt_half_all"])
                                          if applicable else None),
                  M2x=(brier(p["M2x_consistent"], m2 == "CERTIFICATE_CONSISTENT")
                       if (applicable and p["M2x_consistent"] is not None) else None))
        preds[who] = dict(probabilities=p, brier=sc)
    destination.mkdir(parents=True, exist_ok=True)
    D.put(destination / "verdict.json", verdict)
    D.put(destination / "predictions.json", preds)
    write_csv(destination / "per_seed.csv", res["per_seed"])
    write_csv(destination / "per_task.csv", res["per_task"])
    write_csv(destination / "transport_by_window.csv", res["by_window"])
    D.save_npz(destination / "per_unit.npz", dict(sum=np.stack(res["per_unit"]), keys=np.array(N.KEYS),
                                                  seeds=np.array(D.MAIN_SEEDS), tasks=np.array(MAIN_TASKS)))
    return verdict


def write_csv(path, rows):
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with Path(path).open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})


# --------------------------------------------------------------------------
# stage 4: fixed audit epochs (spec 8)
# --------------------------------------------------------------------------

def audit(run, destination=None):
    import torch
    torch.set_num_threads(1)
    torch.set_flush_denormal(True)
    device = D.H.setup("cpu")
    mnist = D.H.Mnist(device)
    res = {}
    for s in SEEDS:
        for t, e in sorted(D.AUDIT):
            a = torch.load(seed_dir(run, s) / "audit" / f"t{t:02d}_e{e:03d}.pt", weights_only=False)
            rep = D.replay_epoch(a["start"], s, t, e, mnist, device)
            same = (torch.equal(rep["order"], a["order"])
                    and all(torch.equal(x, y) for x, y in zip(rep["end"]["params"], a["end"]["params"]))
                    and all(torch.equal(x, y) for x, y in zip(rep["end"]["m"], a["end"]["m"]))
                    and all(torch.equal(x, y) for x, y in zip(rep["end"]["v"], a["end"]["v"]))
                    and rep["end"]["tc"] == a["end"]["tc"]
                    and torch.equal(rep["end"]["g_batch"], a["end"]["g_batch"])
                    and torch.equal(rep["end"]["acc_sum"], a["end"]["acc_sum"])
                    and all(torch.equal(rep[k], a[k]) for k in ("sum", "max", "l1", "aux"))
                    and torch.equal(rep["end"]["obs"]["conf"], a["end"]["obs"]["conf"])
                    and torch.equal(rep["end"]["obs"]["hist"], a["end"]["obs"]["hist"]))
            res[f"s{s}_t{t}_e{e}"] = bool(same)
    out = dict(all_equal=all(res.values()), epochs=res)
    if destination is not None:
        D.put(destination, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["validate", "calibrate", "report", "audit"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--window")
    ap.add_argument("--out")
    a = ap.parse_args()
    res_dir = ROOT / "results" / D.RUN
    if a.stage == "validate":
        st = validate(a.run, res_dir)
        print({k: st[k] for k in ("label", "A_host_identity", "B_complete_finite", "C_numerical")})
    elif a.stage == "calibrate":
        print(calibrate(a.run, a.window))
    elif a.stage == "report":
        v = report(a.run, a.window, a.out)
        print(v["M1x"]["label"], v["M2x"], v["M3x"]["label"])
    else:
        print(audit(a.run, res_dir / "audit.json")["all_equal"])


if __name__ == "__main__":
    main()

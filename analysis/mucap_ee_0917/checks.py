"""Checks for mucap_ee_0917 (specs/spec_mucap_ee_0917.md section 6).

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 analysis/mucap_ee_0917/checks.py
    ... --only S4            # development: writes results/_checks_mucap_ee_0917/checks_partial.json

The EE box is src/mucap_el_run_0916.py's run_one with act2_name="ELU1", driven by
src/mucap_ee_run_0917.py.  Each check runs on the real code and then on every mutation listed with it;
a mutation is an exact-once string substitution in the check's target file and it MUST make the same
check fail.  Tolerances come from the derivations in the docstrings, never from looking at the values.
Writes results/mucap_ee_0917/checks.json after every step.

S0  the EL path of the shared runner still reproduces the recorded 0916 shards (tasks 1-2), except the
    memo_acc bug this commit fixes, which is checked against the trained net instead
S1-S3  the caps themselves: analysis/mucap_el_0916/checks.py's, re-run unchanged (src/mucap_el_0916.py
    is not modified); their result is copied in
S4  EE wiring: an update and the diagnostics of an EE run are the host's ELU->ELU net's
S5, S7  analysis/mucap_el_0916's branch / repro checks on the EE path (S6, the ledger, is not run: the
    EE grid runs with the ledger off)
S8  the verdict on synthetic shards
S9  the CLI runner writes the EE shard the in-process run gives, with this experiment's provenance
S-cost  PROBE EE processes at once
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)
from src import pmnist_0905 as H                 # noqa: E402
from src import pmnist_rlmnist_0906 as RL        # noqa: E402
from src import elu_growth_0909 as EG            # noqa: E402
from src import mucap_el_0916 as MU              # noqa: E402

EL_RUNNER = REPO / "src" / "mucap_el_run_0916.py"
EE_RUNNER = REPO / "src" / "mucap_ee_run_0917.py"
VERDICT = REPO / "analysis" / "mucap_ee_0917" / "verdict.py"
EL_CHECKS_JSON = REPO / "results" / "mucap_el_0916" / "checks.json"
OUT = REPO / "results" / "mucap_ee_0917"
SCR = REPO / "results" / "_checks_mucap_ee_0917"
EL_RECORD = REPO / "results" / "mucap_el_0916"
EL_ARCHIVE_MANIFEST = EL_RECORD / "backup_manifest.json"
EPS64 = float(np.finfo(np.float64).eps)
DEV = H.setup("cpu")
MNIST = H.Mnist(DEV)
PROBE = 6                                        # the launch runs 6 at once; measure under that load
PY = sys.executable
THREAD_ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
TASKS_RUN, STEPS_RUN = 100, 6000
RESULTS: dict = {}
DUMP_PATH = OUT / "checks.json"

# --------------------------------------------------------------------------
# infrastructure (analysis/mucap_el_0916/checks.py's, unchanged in behaviour)
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


def run_check(key: str, title: str, fn, mutations: list, derivation: str, target: Path) -> None:
    t0 = time.time()
    base = fn(load(target))
    entry = {"title": title, "target": str(target.relative_to(REPO)), "threshold_derivation": derivation,
             **base, "mutations": []}
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


class flush_denormal:
    """The EE runner sets torch.set_flush_denormal(True) (spec 2.1); in-process EE runs here do the same
    and restore the previous state, so S0's EL path runs exactly as the 0916 shards did."""
    def __enter__(self):
        torch.set_flush_denormal(True)

    def __exit__(self, *a):
        torch.set_flush_denormal(False)


def _flush_is_on() -> bool:
    """torch has no getter: a float32 product that lands in the subnormal range is 0 only when flushing."""
    return float(torch.tensor([1e-30], dtype=torch.float32) * torch.tensor([1e-10], dtype=torch.float32)) == 0.0


def _images(seed: int) -> torch.Tensor:
    return MNIST.train_x[RL.subset_idx(seed).to(MNIST.train_x.device)]


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------
# S0: the EL path is unchanged (tasks 1-2 of two recorded shards, full 80-epoch tasks)
# --------------------------------------------------------------------------

R_DEFAULT = "    if act2_name is not None:\n"
R_DEFAULT_SET = "        act2 = ACT2[act2_name]\n"
R_LOWGATE = "LOWGATE = 0.05 "
R_MEMO = "            memo = float((forward2(params, x, act1, act2)[4].argmax(1) == y).float().mean())\n"
EL_SHARDS = (("ref", 0), ("cap_both", 0))
S0_TASKS = 2


def _archived(shard: str, name: str) -> Path | None:
    """The 0916 shard's file in obsidian-research-data, used only if its sha256 matches the manifest."""
    man = json.loads(EL_ARCHIVE_MANIFEST.read_text())
    for f in man["files"]:
        if f["source_rel"] == f"results/mucap_el_0916/runs/{shard}/{name}":
            p = Path(f["backup"])
            return p if p.exists() and _sha(p) == f["sha256"] else None
    return None


def s0(M) -> dict:
    """The shared runner's default path (act2_name=None) is the 0916 EL box: for tasks 1-2 of the
    recorded shards ref_s0 and cap_both_s0 (80 epochs, ledger on), (i) every per_task.csv column the 0916
    shard has, except memo_acc, is equal as written (csv round-trips floats exactly), (ii) every
    units.npz and ledger.npz array the archived 0916 shard has is bit-equal on the task 1-2 rows (the
    archive copy is used only when its sha256 matches backup_manifest.json), (iii) the task-1-end state
    hash equals the 0916 provenance's.  (iv) memo_acc, which 0916 computed on an ELU->ELU copy of the
    net, now equals the accuracy of the trained ELU->leaky net recomputed from the task-1-end weights,
    and the recorded 0916 value equals the ELU->ELU copy's (the bug was what this says it was).
    (v) the new column major_frac is the largest label count / 1200.  Bit and string comparisons only."""
    failed, per = [], {}
    elu, lr = EG.ELU(1.0), H.ARMS["LR"]
    for arm, seed in EL_SHARDS:
        shard = f"{arm}_s{seed}"
        d = {}
        rows, arrays, led, info = M.run_one(arm, seed, 1e-3, S0_TASKS, MNIST, DEV, epochs=80, ledger=True,
                                            debug=d)
        rec = [r for r in csv.DictReader((EL_RECORD / "runs" / shard / "per_task.csv").open())
               if int(r["task"]) <= S0_TASKS]
        prov = json.loads((EL_RECORD / "runs" / shard / "provenance.json").read_text())
        bad_cols, n_cols = [], 0
        for rr, got in zip(rec, rows):
            for k, v in rr.items():
                if k in ("arm", "memo_acc"):
                    continue
                n_cols += 1
                g = got.get(k)
                same = (str(g) == v) or (g is not None and v != "" and float(v) == float(g))
                if not same:
                    bad_cols.append(f"t{rr['task']}:{k}")
        arr_bad, n_arr = [], 0
        u_path, l_path = _archived(shard, "units.npz"), _archived(shard, "ledger.npz")
        if u_path is not None:
            with np.load(u_path) as z:
                pre = f"s{seed}_"
                tk = z[pre + "task"]
                keep = tk <= S0_TASKS
                for k in z.files:
                    kk = k[len(pre):]
                    if kk in ("q_cap", "v_cap"):
                        a_rec, a_got = z[k], arrays[kk]
                    elif kk.startswith("test_"):           # the test set is read at task 1 only here
                        a_got = arrays[kk]
                        a_rec = z[k][:a_got.shape[0]]
                    else:
                        a_rec, a_got = z[k][keep], arrays[kk]
                    n_arr += 1
                    if not np.array_equal(a_rec, a_got, equal_nan=True):
                        arr_bad.append(kk)
        if l_path is not None:
            with np.load(l_path) as z:
                pre = f"s{seed}_"
                keep = z[pre + "task"] <= S0_TASKS
                for k in z.files:
                    n_arr += 1
                    if not np.array_equal(z[k][keep], led[k[len(pre):]], equal_nan=True):
                        arr_bad.append("ledger:" + k[len(pre):])
        p1 = d["task1_end_params"]
        x = _images(seed)
        y1 = d["labels"][0]
        true_acc = float((M.forward2(p1, x, elu, lr)[4].argmax(1) == y1).float().mean())
        ee_acc = float((H.forward(p1, x, elu)[4].argmax(1) == y1).float().mean())
        maj = float(torch.bincount(y1, minlength=10).max()) / 1200
        ok = {"i_per_task_columns": not bad_cols and n_cols >= 60,
              "ii_arrays": u_path is not None and l_path is not None and not arr_bad and n_arr >= 60,
              "iii_task1_state": info["task1_end_state_sha256"] == prov["per_seed"][str(seed)]["task1_end_state_sha256"],
              "iv_memo_is_the_net": rows[0]["memo_acc"] == true_acc,
              "iv_old_memo_was_ee_copy": float(rec[0]["memo_acc"]) == ee_acc and ee_acc != true_acc,
              "v_major_frac": rows[0]["major_frac"] == maj,
              "nonvacuous": arm == "ref" or rows[1]["rows_par"] > 0}
        per[shard] = {**ok, "bad_columns": bad_cols[:10], "n_columns": n_cols, "bad_arrays": arr_bad[:10],
                      "n_arrays": n_arr, "archive_used": [str(u_path), str(l_path)],
                      "memo_recorded_0916": float(rec[0]["memo_acc"]), "memo_now": rows[0]["memo_acc"],
                      "acc_elu_leaky": true_acc, "acc_elu_elu_copy": ee_acc}
        failed += [f"{shard}|{k}" for k, v in ok.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S0_MUT = [
    ("M0a: the default path runs ELU on the second layer",
     [(R_DEFAULT, "    if True:\n"), (R_DEFAULT_SET, '        act2 = ACT2[act2_name or "ELU1"]\n')]),
    ("M0b: a diagnostic constant moved (low-gate mark 0.06)", [(R_LOWGATE, "LOWGATE = 0.06 ")]),
    ("M0c: memo_acc back on the ELU->ELU copy (the 0916 bug)",
     [(R_MEMO, "            memo = RL.evaluate_rl(params, x, y, act1)[\"acc\"]\n")]),
]


# --------------------------------------------------------------------------
# S1-S3: the caps (unchanged file) -- copied from the 0916 record, re-verified by hash
# --------------------------------------------------------------------------

def s1_3_record() -> None:
    rec = json.loads(EL_CHECKS_JSON.read_text())
    want = rec["code_sha256"]["src/mucap_el_0916.py"]
    now = hashlib.sha256((REPO / "src" / "mucap_el_0916.py").read_bytes()).hexdigest()
    for k in ("S1_S-mu-basis", "S2_S-mu-projection", "S3_S-mu-order"):
        e = rec[k]
        RESULTS[k] = {"title": e["title"] + " (0916 record; src/mucap_el_0916.py unchanged)",
                      "pass": bool(e["pass"] and e["all_mutations_detected"] and want == now),
                      "all_mutations_detected": e["all_mutations_detected"],
                      "record": "results/mucap_el_0916/checks.json", "record_sha256_of_caps": want,
                      "caps_sha256_now": now,
                      "mutations_detected": f"{sum(m['detected'] for m in e['mutations'])}/{len(e['mutations'])}"}
    dump()
    print(f"S1-S3: copied from the 0916 record, caps file unchanged = {want == now}", flush=True)


# --------------------------------------------------------------------------
# S4: EE wiring
# --------------------------------------------------------------------------

R_UNITS = "        u = unit_arrays(params, x, act1, act2, e1_64)\n"                       # per_task medians
R_SNAP = "        u = unit_arrays(params, x, act1, act2, e1_64) | adam_arrays(params, adam)\n"   # units.npz
EP = 2               # 150 updates per task in the runner checks; the properties are per update
TOL_ELU = 1e-3       # non-vacuity: the EE second layer must MISS the leaky identity by at least this
CAPTURE = ((1, 37), (2, 5), (2, 149))


def _run(M, arm, seed=0, tasks=1, ledger=False, debug=None, epochs=EP):
    with flush_denormal():
        return M.run_one(arm, seed, 1e-3, tasks, MNIST, DEV, epochs=epochs, ledger=ledger, debug=debug,
                         act2_name="ELU1")


def _host_update(st: dict, e1: torch.Tensor, act) -> list:
    """One update recomputed from the captured state with the host's single-activation forward, the
    runner's Adam arithmetic and the caps' own functions."""
    with flush_denormal():
        p = [q.clone().requires_grad_(True) for q in st["before"]["params"]]
        m = [q.clone() for q in st["before"]["m"]]
        v = [q.clone() for q in st["before"]["v"]]
        loss = torch.nn.functional.cross_entropy(H.forward(p, st["xb"], act)[4], st["yb"])
        grads = torch.autograd.grad(loss, p)
        tc = st["before"]["tc"] + 1
        b1, b2, eps, lr = 0.9, 0.999, 1e-8, 1e-3
        c1, c2 = 1 - b1 ** tc, 1 - b2 ** tc
        with torch.no_grad():
            p = [q.detach().clone() for q in p]
            for q, g, mi, vi in zip(p, grads, m, v):
                mi.mul_(b1).add_(g, alpha=1 - b1)
                vi.mul_(b2).addcmul_(g, g, value=1 - b2)
                q -= lr * (mi / c1) / ((vi / c2).sqrt() + eps)
            if st["cap_on"]:
                MU.cap_parallel_(p[0], e1, st["q_cap"])
                MU.cap_perp_(p[0], e1, st["v_cap"])
    return p + m + v


def s4(M) -> dict:
    """(i) forward2 with ELU twice is bit-identical to the host's forward with ELU.  (ii) Three captured
    updates of an EE cap_both run (one in task 1, two in task 2 with the caps on) are bit-identical to
    the same update recomputed with the host's ELU->ELU forward, the runner's Adam arithmetic and the
    caps; the same recomputation with leaky must give different bits (non-vacuity).  (iii) The run's own
    task-1-end second-layer diagnostics (zbar and mean phi' in units.npz, the mean phi' median in
    per_task) equal the host ELU->ELU forward on the captured task-1-end weights, bit for bit, and miss
    the leaky identity mean phi' = a + (1-a) p+ by more than TOL_ELU.  (iv) memo_acc is that forward's
    accuracy.  Bit comparisons, one tolerance: TOL_ELU."""
    failed, per = [], {}
    elu, lr = EG.ELU(1.0), H.ARMS["LR"]
    p0 = [t.detach() for t in H.init_params(0, DEV)]
    x = _images(0)
    per["i_forward2_is_host_ELU"] = all(bool((a == b).all()) for a, b in
                                        zip(M.forward2(p0, x, elu, elu), H.forward(p0, x, elu)))
    d = {"capture": CAPTURE}
    rows, arrays, _, _ = _run(M, "cap_both", tasks=2, debug=d)
    e1 = d["e1"]
    upd = {}
    for key in CAPTURE:
        st = d["steps"][key]
        got = st["after"]["params"] + st["after"]["m"] + st["after"]["v"]
        want_e = _host_update(st, e1, elu)
        want_l = _host_update(st, e1, lr)
        upd[str(key)] = {"equals_host_elu": all(bool((a == b).all()) for a, b in zip(got, want_e)),
                         "elu_and_leaky_differ": any(bool((a != b).any()) for a, b in zip(want_e, want_l)),
                         "cap_on": st["cap_on"]}
    per["ii_updates"] = upd
    per["ii_update_is_host_elu"] = all(u["equals_host_elu"] for u in upd.values())
    per["ii_nonvacuous"] = all(u["elu_and_leaky_differ"] for u in upd.values()) and \
        any(u["cap_on"] for u in upd.values())
    p1 = d["task1_end_params"]
    ends = np.where((arrays["task"] == 1) & (arrays["step"] == M.STEPS_PER_EPOCH * EP))[0]
    i1 = int(ends[0])
    with flush_denormal():
        z1, a1, z2, a2, lg = H.forward(p1, x, elu)
        g2 = elu.dphi(z2).double().mean(0).numpy()
    zb2 = z2.double().mean(0).numpy()
    per["iii_zbar_l2_is_host"] = bool(np.array_equal(arrays["zbar_l2"][i1], zb2))
    per["iii_gate_l2_is_host_elu"] = bool(np.array_equal(arrays["gate_mean_l2"][i1], g2))
    a32 = float(torch.tensor(lr.param, dtype=torch.float32))
    gap = float(np.abs(arrays["gate_mean_l2"][i1] - (a32 + (1 - a32) * arrays["pplus_l2"][i1])).max())
    per["iii_not_leaky"] = gap > TOL_ELU
    per["iii_row_median_is_host_elu"] = rows[0]["med_gate_mean_l2"] == float(np.nanmedian(g2))
    per["leaky_identity_gap_l2"] = gap
    per["iv_memo_is_host_elu"] = rows[0]["memo_acc"] == float((lg.argmax(1) == d["labels"][0]).float().mean())
    keys = ("i_forward2_is_host_ELU", "ii_update_is_host_elu", "ii_nonvacuous", "iii_zbar_l2_is_host",
            "iii_gate_l2_is_host_elu", "iii_not_leaky", "iii_row_median_is_host_elu", "iv_memo_is_host_elu")
    failed += [k for k in keys if not per[k]]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S4_MUT = [
    ("M4a: act2_name ignored (the EE run trains ELU->leaky)", [(R_DEFAULT, "    if False:\n")]),
    ("M4b: ACT2['ELU1'] is the leaky object",
     [('ACT2 = {"LR": H.ARMS["LR"], "ELU1": EG.ELU(1.0)}\n', 'ACT2 = {"LR": H.ARMS["LR"], "ELU1": H.ARMS["LR"]}\n')]),
    ("M4c: units.npz reads the second layer through leaky",
     [(R_SNAP, '        u = unit_arrays(params, x, act1, H.ARMS["LR"], e1_64) | adam_arrays(params, adam)\n')]),
    ("M4e: the per-task medians read the second layer through leaky",
     [(R_UNITS, '        u = unit_arrays(params, x, act1, H.ARMS["LR"], e1_64)\n')]),
    ("M4d: memo_acc through leaky on the second layer",
     [(R_MEMO, '            memo = float((forward2(params, x, act1, H.ARMS["LR"])[4].argmax(1) == y).float().mean())\n')]),
]


# --------------------------------------------------------------------------
# S5-S7 on the EE path (0916's properties; the anchors are in the unchanged lines of the runner)
# --------------------------------------------------------------------------

R_CAPON = "        cap_on = t >= 2 and (do_par or do_perp)\n"
R_CAPSET = "            q_cap = MU.parallel_cap(params[CAP_LAYER], e1)\n"
R_CAPLAYER = "CAP_LAYER = 0 "
R_QSQ = '    acc[f"{part}_q_sq"] += dq * dq\n'
R_PROJBOOK = '        _add_step(acc, "proj", pm, _point(W_after, e64), d_in)\n'
R_V2ALIGN = '    acc[f"{part}_v2_align"] += 2 * (dot - q0 * dq)\n'
R_WT2SQ = '    acc[f"{part}_wt2_sq"] += sq - d_in * dm * dm\n'
R_ORDER = "            order = torch.randperm(N_IMAGES, generator=g_batch).to(device)\n"
TOL_CLOSE = 8.0      # S6 residual in units of n_steps*eps64*traffic (0916's derivation)


def s5(M) -> dict:
    """0916's S5 on the EE path: the four arms' task 1 is the same run (init, subset, labels, batch
    orders, task-1-end state sha256); they differ after task 2; the radii are the task-1-end rows,
    recomputed with the caps module from the captured weights.  Bit comparisons only."""
    failed = []
    end1, states, caps = {}, {}, {}
    for arm in M.ARMS:
        d = {}
        rows, arrays, _, info = _run(M, arm, tasks=2, debug=d)
        end1[arm] = (info["init_sha256"], info["subset_idx_sha256"], info["labels_sha256"],
                     info["batch_sha256"], info["task1_end_state_sha256"])
        states[arm] = info["final_state_sha256"]
        W1 = d["task1_end_params"][M.CAP_LAYER]
        caps[arm] = (MU.parallel_cap(W1, d["e1"]), MU.perp_cap(W1, d["e1"]), arrays["q_cap"], arrays["v_cap"])
    same_t1 = {a: end1[a] == end1["ref"] for a in M.ARMS}
    diff_t2 = {a: states[a] != states["ref"] for a in M.ARMS if a != "ref"}
    radii = {a: bool((caps[a][0][:, 0].numpy() == caps[a][2]).all()) and
             bool((caps[a][1][:, 0].numpy() == caps[a][3]).all()) for a in M.ARMS if a != "ref"}
    ok = dict(i_task1_identical=all(same_t1.values()), ii_arms_diverge_by_task2=all(diff_t2.values()),
              iii_radii_are_task1_end=all(radii.values()), nonvacuous=len(set(states.values())) > 1)
    failed += [k for k, v in ok.items() if not v]
    return {"pass": not failed, "failed_items": failed,
            "detail": {**ok, "task1_identical": same_t1, "differs_after_task2": diff_t2, "radii": radii}}


S5_MUT = [
    ("M5a: the cap is on from task 1", [(R_CAPON, "        cap_on = t >= 1 and (do_par or do_perp)\n")]),
    ("M5b: the radii are taken at init",
     [(R_CAPSET, "            q_cap = MU.parallel_cap(H.init_params(seed, device)[CAP_LAYER], e1)\n")]),
    ("M5c: the cap is applied to the second layer", [(R_CAPLAYER, "CAP_LAYER = 2 ")]),
]


def s6(M) -> dict:
    """0916's S6 on the EE path: for every task of a 3-task run and every unit, the accumulated Adam and
    projection terms of q, ||v||^2, m and ||W~||^2 reproduce the change between the task's first and last
    diagnostic points, and the signed displacements reproduce the change in q and m, within TOL_CLOSE
    units of n_steps * eps64 * traffic."""
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
                direct = a1 ** 2 - a0 ** 2
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
    """0916's S7 on the EE path: the same call twice gives the same bits (arrays, ledger, hashes, rows)."""
    failed, per = [], {}
    for arm in ("ref", "cap_both"):
        r1, a1, l1, i1 = _run(M, arm, tasks=2, ledger=True)
        r2, a2, l2, i2 = _run(M, arm, tasks=2, ledger=True)
        ok = dict(i_arrays=set(a1) == set(a2) and all(np.array_equal(a1[k], a2[k], equal_nan=True) for k in a1),
                  ii_ledger=set(l1) == set(l2) and all(np.array_equal(l1[k], l2[k], equal_nan=True) for k in l1),
                  iii_hashes=all(i1[k] == i2[k] for k in ("init_sha256", "labels_sha256", "batch_sha256",
                                                          "task1_end_state_sha256", "final_state_sha256")),
                  iv_rows=r1 == r2, nonvacuous=len(a1) > 10 and len(l1) > 10)
        per[arm] = ok
        failed += [f"{arm}|{k}" for k, v in ok.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S7_MUT = [
    ("M7a: the batch order does not come from the run's own generator",
     [(R_ORDER, "            order = torch.randperm(N_IMAGES).to(device)\n")]),
]


# --------------------------------------------------------------------------
# S9: the CLI runner (src/mucap_ee_run_0917.py)
# --------------------------------------------------------------------------

E_ACT2 = 'ACT2_NAME = "ELU1"\n'
E_CALL = "                                             progress=True, act2_name=ACT2_NAME)\n"
E_TASKS = "TASKS = 100\n"
E_FLUSH = "    torch.set_flush_denormal(True)\n"


def s9(M) -> dict:
    """The runner's main(), called in-process on a 1-task, 2-epoch shard, writes (i) the per_task row and
    (ii) every units.npz array the in-process EE run (the real shared runner, flush on) gives, bit for
    bit; (iii) its provenance says act2 ELU1, flush_denormal true, this experiment and spec, and hashes
    both runner files; (iv) its registered defaults are the spec's 100 tasks x 80 epochs, ledger off; (v) main()
    leaves flush_denormal on (the grid's processes run nothing else).  Bit and string comparisons only."""
    failed, per = [], {}
    out = SCR / "s9_cli"
    torch.set_flush_denormal(False)
    try:
        M.main(["--arm", "cap_perp", "--seeds", "0", "--tasks", "1", "--epochs", "2", "--out", str(out)])
        flushed = _flush_is_on()
    finally:
        torch.set_flush_denormal(False)
    EL = load(EL_RUNNER)
    rows, arrays, _, _ = _run(EL, "cap_perp", tasks=1, epochs=2)
    got_rows = list(csv.DictReader((out / "per_task.csv").open()))
    per["i_rows"] = len(got_rows) == 1 and all(
        str(rows[0][k]) == v or float(rows[0][k]) == float(v) for k, v in got_rows[0].items() if k != "arm") \
        and got_rows[0]["arm"] == "cap_perp"
    with np.load(out / "units.npz") as z:
        per["ii_units"] = set(z.files) == {f"s0_{k}" for k in arrays} and all(
            np.array_equal(z[f"s0_{k}"], v, equal_nan=True) for k, v in arrays.items())
    pv = json.loads((out / "provenance.json").read_text())
    per["iii_provenance"] = (pv["act2"] == "ELU1" and pv.get("flush_denormal") is True and
                             pv["experiment"] == "mucap_ee_0917" and pv["spec"] == "specs/spec_mucap_ee_0917.md"
                             and {"src/mucap_ee_run_0917.py", "src/mucap_el_run_0916.py"} <= set(pv["code_sha256"]))
    per["iv_defaults"] = M.TASKS == TASKS_RUN and M.EPOCHS == 80 and pv["ledger"] is False
    per["v_flush_left_on"] = bool(flushed)
    failed += [k for k in ("i_rows", "ii_units", "iii_provenance", "iv_defaults", "v_flush_left_on") if not per[k]]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S9_MUT = [
    ("M9a: the CLI passes no act2 (trains ELU->leaky)", [(E_CALL, "                                             progress=True, act2_name=None)\n")]),
    ("M9b: the registered horizon is 150 tasks", [(E_TASKS, "TASKS = 150\n")]),
    ("M9e: the ledger on by default", [('    ap.add_argument("--ledger", action="store_true")\n', '    ap.add_argument("--ledger", action="store_false")\n')]),
    ("M9c: provenance records the wrong second activation", [(E_ACT2, 'ACT2_NAME = "ELU1"\nACT2_LABEL = "LR"\n'),
                                                             ('"act2": ACT2_NAME,', '"act2": ACT2_LABEL,')]),
    ("M9d: denormals not flushed", [(E_FLUSH, "    torch.set_flush_denormal(False)\n")]),
]


# --------------------------------------------------------------------------
# S-cost
# --------------------------------------------------------------------------

PILOT_LOG = REPO / "results" / "_pilot_mucap_ee_0917" / "flush_ref_s0.log"


def _growth() -> float:
    """Mean per-task time over tasks 1-100 / mean over tasks 2-10, from the flush pilot's log: tasks
    2-40 measured, 41-100 extrapolated with the least-squares line through tasks 14-40."""
    import re
    ts = []
    for line in PILOT_LOG.read_text().splitlines():
        m = re.search(r"\[\s*([\d.]+)s\] ref seed=\d task (\d+)/", line)
        if m:
            ts.append((int(m.group(2)), float(m.group(1))))
    dt_ = {t: b - a for (_, a), (t, b) in zip(ts, ts[1:])}
    base = float(np.mean([dt_[t] for t in range(2, 11)]))
    xs = np.array([t for t in dt_ if t >= 14], float)
    slope, icpt = np.polyfit(xs, np.array([dt_[int(t)] for t in xs]), 1)
    full = [dt_.get(t, base) if t <= 40 else icpt + slope * t for t in range(1, TASKS_RUN + 1)]
    return float(np.mean(full) / base)


def s_cost() -> None:
    """PROBE processes of the real EE runner at once (the launch's parallelism), 2 tasks x 80 epochs,
    ledger off (as the grid), one thread each.  Gate: every process exits 0 and the memory budget leaves >= PROBE slots
    (a slot = one process's peak RSS against 0.8 * MemAvailable).  The EE net's per-task cost is not
    constant (the pilot's grew 4x by task 40 without the denormal flush, 1.6x with it): the projection
    to 100 tasks multiplies the measured 2-task cost by GROWTH, the flush pilot's mean per-task time over
    tasks 1-100 extrapolated linearly from its tasks 14-40, divided by its task 2-10 time."""
    t0 = time.time()
    avail0 = _mem_available_gib()
    SCR.mkdir(parents=True, exist_ok=True)
    procs = []
    for i in range(PROBE):
        d = SCR / f"cost_{i}"
        procs.append((d, subprocess.Popen(
            [PY, str(EE_RUNNER), "--arm", "cap_both", "--seeds", str(i), "--tasks", "2", "--out", str(d)],
            env=THREAD_ENV, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)))
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
    per_update = (max(secs) / (2 * STEPS_RUN)) if secs else float("nan")
    growth = _growth()
    slots = int(avail0 * 0.8 / max(peak)) if peak else 0
    one = TASKS_RUN * STEPS_RUN * per_update * growth
    ok = all(r == 0 for r in rc) and slots >= PROBE
    RESULTS["S-cost"] = {"pass": bool(ok), "gate": f"all {PROBE} processes exit 0 and slots >= {PROBE}",
                         "threshold_derivation": "slot = peak RSS of one process; budget = 0.8 * MemAvailable "
                                                 "at the start; the launch runs PROBE at once",
                         "returncodes": rc, "mem_available_gib_before": avail0,
                         "peak_rss_gib_max": max(peak) if peak else None, "slots": slots,
                         "seconds_per_update_under_load": per_update, "growth_factor": growth,
                         "one_trajectory_min": one / 60,
                         "grid_40_wall_hours_at_probe": 40 * one / 3600 / PROBE,
                         "measured_at": dt.datetime.now().astimezone().isoformat(),
                         "seconds": round(time.time() - t0, 1)}
    dump()
    c = RESULTS["S-cost"]
    print(f"S-cost: pass={ok}  {per_update * 1e3:.2f} ms/update under {PROBE} at once -> "
          f"{c['one_trajectory_min']:.1f} min per trajectory, {c['grid_40_wall_hours_at_probe']:.1f} h at "
          f"{PROBE} (peak RSS {c['peak_rss_gib_max']:.2f} GiB, slots {slots})", flush=True)


# --------------------------------------------------------------------------
# S8: the verdict on synthetic shards
# --------------------------------------------------------------------------

REG_T, REG_SPT, REG_WIN, REG_CAPWIN = 100, 6000, (51, 100), (2, 10)   # the REGISTERED constants
TOL_EST = 1e-9
T_TABLE = {(0.975, 9): 2.262157, (0.9875, 9): 2.685011, (0.95, 9): 1.833113}
TOL_T = 1e-6
A1 = 0.82


def _mf(t: int) -> float:
    return 0.11 + 0.004 * math.sin(t)


R_FLOOR = float(np.mean([_mf(t) - 0.01 for t in range(REG_WIN[0], REG_WIN[1] + 1)]))   # ref's floored level
F_WIN = float(np.mean([_mf(t) for t in range(REG_WIN[0], REG_WIN[1] + 1)]))
THR_WIN = F_WIN + 3.0 * float(np.std([_mf(t) for t in range(REG_WIN[0], REG_WIN[1] + 1)], ddof=1)) / math.sqrt(50)


def _centered(rng, n, sd):
    x = rng.normal(0.0, 1.0, n)
    x = x - x.mean()
    return x / x.std(ddof=1) * sd if sd > 0 else np.zeros(n)


DEFAULT_ARMS = {
    "ref": {"collapse": 5},
    "cap_par": {"collapse": 3, "eff2": 0.0, "noise2": 0.02},
    "cap_perp": {"collapse": None, "L": 0.60, "eff2": 0.2, "noise2": 0.02, "tail": 5.0},
    "cap_both": {"collapse": None, "L": 0.48, "eff2": 0.0, "noise2": 0.3},
}


def synth_shards(V, scen: dict) -> dict:
    """4 arms x 10 seeds x 100 tasks.  Online accuracy: A1 at task 1; an arm that collapses at T keeps A1
    before T and sits at its floor level after (major_frac(t) - 0.01 unless given); an alive arm sits at
    L (+ centred per-seed noise in the main window).  major_frac(t) is the same for every run.  Second
    layer mean phi' at task ends: 0.6 at task 1, 0.6 + r2 + m2[s] in the formation windows, plus eff2 +
    n2[a][s] (+ tail/100 on the arm with a unit-0 tail) in the main window; m2 is shared by the arms of a
    seed, so only a paired comparison sees eff2 when seeds differ.  Step-0 rows are garbage.  The first
    layer's mean phi' is a constant (a wrong E2 key reads it).  Cap rows stop 10 tasks after a collapse."""
    rng = np.random.default_rng(scen.get("rng", 917))
    T, NU, SEEDS = REG_T, 100, tuple(range(10))
    arms = {a: {**DEFAULT_ARMS[a], **scen.get("arms", {}).get(a, {})} for a in DEFAULT_ARMS}
    r2 = scen.get("r2", -0.55)
    m2 = _centered(rng, 10, scen.get("seed_sd", 0.02))
    shards = {}
    for a, spec in arms.items():
        n2 = np.zeros(10) if a == "ref" else _centered(rng, 10, spec.get("noise2", 0.02))
        n1 = _centered(rng, 10, 0.0 if a == "ref" else spec.get("noise1", 0.01))
        for si, s in enumerate(SEEDS):
            col = spec["collapse"]
            if isinstance(col, dict):
                col = col.get(s, col.get("default"))
            online, rows_ = [], []
            for t in range(1, T + 1):
                if t == 1:
                    online.append(A1)
                elif col is not None and t >= col:
                    online.append(spec["floor_level"] if "floor_level" in spec else _mf(t) - 0.01)
                elif col is not None:
                    online.append(A1)
                else:
                    online.append(spec["L"] + (n1[si] if REG_WIN[0] <= t <= REG_WIN[1] else 0.0))
                capped = V.CAPPED.get(a, ()) if hasattr(V, "CAPPED") else ()
                frozen = col is not None and t >= col + 10
                rows_.append({c: (5 if (a != "ref" and c in DEFAULT_CAPS[a] and t >= 2 and not frozen) else 0)
                              for c in ("rows_par", "rows_perp")})
            pt = pd.DataFrame({"task": np.arange(1, T + 1), "online_acc": online,
                               "memo_acc": online, "major_frac": [_mf(t) for t in range(1, T + 1)],
                               "rows_par": [r["rows_par"] for r in rows_], "rows_perp": [r["rows_perp"] for r in rows_]})
            for (ia, iseed, itask, icol) in scen.get("idle", ()):
                if ia == a and iseed == s:
                    pt.loc[pt.task == itask, icol] = 0
            task_, step_, G2 = [], [], []
            for t in range(1, T + 1):
                for step in (0, REG_SPT):
                    task_.append(t)
                    step_.append(step)
                    if step == 0:
                        G2.append(np.full(NU, 9.0))
                        continue
                    if t == 1:
                        lv = 0.6
                    elif REG_WIN[0] <= t <= REG_WIN[1]:
                        lv = 0.6 + r2 + m2[si] + (0.0 if a == "ref" else spec.get("eff2", 0.0) + n2[si])
                    else:
                        lv = 0.6 + r2 + m2[si]
                    g = lv + (np.arange(NU) - 49.5) * 1e-4
                    if spec.get("tail") and REG_WIN[0] <= t <= REG_WIN[1]:
                        g = g.copy()
                        g[0] += spec["tail"]
                    G2.append(g)
            k = len(task_)
            units = {"task": np.array(task_, float), "step": np.array(step_, float),
                     "gate_mean_l2": np.stack(G2), "gate_mean_l1": np.full((k, NU), 0.3),
                     "U_l1": np.ones((k, NU)), "U_l2": np.ones((k, NU)),
                     "mu2_norm": np.full((k, 1), 5.0), "mu2_proj_sd": np.full((k, 1), 1.0),
                     "s2_mean": np.zeros((k, 1)), "s2_sd": np.ones((k, 1))}
            hs = {kk: f"h{s}" for kk in V.STREAM_KEYS}
            if s in scen.get("break", ()) and a == "cap_perp":
                hs["batch_sha256"] = "broken"
            prov = {"git_hash": "synthetic", "per_seed": {str(s): {**hs, "tasks_completed": T,
                                                                   "divergence": {"diverged": False}}}}
            shards[(a, s)] = {"units": units, "per_task": pt, "prov": prov, "ledger": {}}
    return shards


DEFAULT_CAPS = {"cap_par": ("rows_par",), "cap_perp": ("rows_perp",), "cap_both": ("rows_par", "rows_perp")}
_BORDER = 2.45 * 0.05 / math.sqrt(10)      # t = 2.45: inside (t_.975,9 = 2.262, t_.9875,9 = 2.685)
_split = {"default": 3}
S8_SCENARIOS = [
    ("designed", {},
     {"main": "RESCUED", "cap_perp_95": "RESCUED", "cap_par": "COLLAPSED", "cap_both": "RESCUED_FUNCTION_ONLY",
      "timing_cap_perp": "LATER", "timing_cap_par": "EARLIER", "timing_cap_both": "LATER"},
     {"main_E1": 0.60 - R_FLOOR, "main_E2": 0.2 + 5.0 / 100, "n_valid": 10,
      "t_half|ref|0": 5, "t_half|cap_par|0": 3, "t_half|cap_both|0": None, "t_half|cap_perp|3": None,
      "state|ref": "FLOORED", "state|cap_par": "FLOORED", "state|cap_perp": "ALIVE",
      "impaired|cap_par": True, "impaired|cap_perp": False, "impaired|cap_both": False}),
    ("borderline", {"arms": {"cap_perp": {"eff2": _BORDER, "noise2": 0.05, "tail": 0.0}}},
     {"main": "RESCUED_FUNCTION_ONLY", "cap_perp_95": "RESCUED"}, {"main_E2": _BORDER}),
    ("not_reproduced", {"arms": {"ref": {"collapse": None, "L": 0.5}}},
     {"main": "NOT_REPRODUCED", "cap_par": "NOT_REPRODUCED", "cap_both": "NOT_REPRODUCED"}, {}),
    ("ref_8_of_10", {"arms": {"ref": {"collapse": {8: None, 9: None, "default": 5}, "L": 0.5}}},
     {"main": "RESCUED"}, {}),
    ("ref_7_of_10", {"arms": {"ref": {"collapse": {7: None, 8: None, 9: None, "default": 5}, "L": 0.5}}},
     {"main": "NOT_REPRODUCED"}, {}),
    ("split", {"arms": {"cap_perp": {"collapse": {7: 20, 8: 20, 9: 20, "default": None}}}},
     {"main": "SPLIT"}, {}),
    ("floor_margin", {"arms": {"cap_both": {"collapse": 20, "floor_level": F_WIN + 0.0005},
                               "cap_par": {"collapse": 20, "floor_level": THR_WIN + 0.002, "eff2": 0.1}}},
     {"cap_both": "COLLAPSED", "cap_par": "RESCUED"}, {}),
    ("cap_idle", {"idle": (("cap_both", 3, 5, "rows_perp"), ("cap_perp", 4, 60, "rows_perp"))},
     {"cap_both": "INAPPLICABLE", "main": "RESCUED"}, {}),
    ("broken_streams", {"break": (7, 8, 9)}, {"main": "INAPPLICABLE"}, {"n_valid": 7}),
    ("timing_counts", {"arms": {"cap_par": {"collapse": {8: 7, 9: 7, "default": 3}},
                                "cap_both": {"collapse": {6: 5, 7: 5, 8: 5, 9: 5, "default": 3}},
                                "cap_perp": {"collapse": {5: 5, 6: 5, 7: 5, 8: 5, 9: 5, "default": 3}}}},
     {"timing_cap_par": "NO_TIMING_DIFF", "timing_cap_both": "EARLIER", "timing_cap_perp": "NO_TIMING_DIFF"}, {}),
    ("paired_matters", {"seed_sd": 1.0, "r2": -3.0,
                        "arms": {"cap_perp": {"eff2": 0.25, "noise2": 0.02, "tail": 0.0}}},
     {"main": "RESCUED"}, {"main_E2": 0.25}),
]


def _pick(res: dict, key: str):
    if key == "n_valid":
        return len(res["valid_seeds"])
    if key.startswith("main_"):
        ep = key.split("_")[1]
        return next(r["mean"] for r in res["rows"] if r["role"] == "main" and r["endpoint"] == ep)
    if key.startswith("t_half|"):
        _, arm, s = key.split("|")
        return res["t_half"][f"{arm}|{s}"]
    if key.startswith("state|"):
        return res["floor_state"][key.split("|")[1]]
    if key.startswith("impaired|"):
        return res["impaired"][key.split("|")[1]]["flag"]
    raise KeyError(key)


def s8(V) -> dict:
    """verdict.analyze on the S8 scenarios: every designed label and estimate comes out as written
    (estimates within TOL_EST, flags, states and censored times exactly), and the imported t quantile
    reproduces the published table within TOL_T.  Tolerances: TOL_EST, TOL_T."""
    failed, per = [], {}
    bad_t = [f"t({p},{df})" for (p, df), w in T_TABLE.items() if abs(V.t_quantile(p, df) - w) > TOL_T]
    failed += bad_t
    for name, scen, want_labels, want_vals in S8_SCENARIOS:
        res = V.analyze(synth_shards(V, scen))
        bad = [f"{k}: got {res['labels'].get(k)} want {w}" for k, w in want_labels.items()
               if res["labels"].get(k) != w]
        for k, w in want_vals.items():
            try:
                g = _pick(res, k)
            except (KeyError, StopIteration) as e:
                bad.append(f"{k}: missing ({e!r})")
                continue
            if w is None or isinstance(w, (bool, str, int)) and not isinstance(w, float):
                ok = g == w and (w is not None or g is None)
            else:
                ok = g is not None and np.isfinite(g) and abs(float(g) - w) <= TOL_EST
            if not ok:
                bad.append(f"{k}: got {g} want {w}")
        per[name] = {"labels": res["labels"], "bad": bad}
        failed += [f"{name}|{b}" for b in bad]
    return {"pass": not failed, "failed_items": failed, "detail": per}


V_MAIN = 'MAIN_ARM = "cap_perp"                       # spec 5.3: the primary comparison is cap_perp - ref\n'
V_E2 = 'E2_KEY = "gate_mean_l2"\n'
V_LEVEL = "MAIN_LEVEL = 0.975                          # two-sided, two main endpoints (Bonferroni)\n"
V_B = "    B_ok = A_ok and ref_at >= math.ceil(REF_FLOOR_FRAC * len(valid)) and B_E2[\"hi\"] < 0\n"
V_FRAC = "REF_FLOOR_FRAC = 0.8 "
V_K = "FLOOR_K = 3.0 "
V_F = '    F, s = float(f.mean()), float(f.std(ddof=1))\n'
V_PAIR = '        return deltas(arm, ep, win) - deltas("ref", ep, win)\n'
V_UMEAN = "        m, k = unit_mean(u[key][ends[t]])\n"
V_CENS = '    if ta is None:\n        return "later"\n'
V_ALPHA = "SIGN_ALPHA = 0.05 "
V_SIGN_N = "        p = sign_test(max(e, l_), e + l_)\n"
V_HALF = "        if f.loc[t] < HALF:\n"
V_CAPWIN = "CAP_WIN = (2, 10) "
V_CAPPED = 'CAPPED = {"cap_par": ("rows_par",), "cap_perp": ("rows_perp",), "cap_both": ("rows_par", "rows_perp")}\n'
V_STREAMS = "    if len(set(infos.values())) != 1:\n"
V_IMP = "            hits += f_arm < f_ref - (1.0 - f_ref)\n"
V_ENDS_USE = "    ends = task_ends(u)\n    base, nans = unit_mean(u[key][ends[1]])\n"
S8_MUT = [
    ("M8a: the primary arm is cap_both", [(V_MAIN, 'MAIN_ARM = "cap_both"\n')]),
    ("M8b: E2 reads the first layer's phi'", [(V_E2, 'E2_KEY = "gate_mean_l1"\n')]),
    ("M8c: main label at 95%", [(V_LEVEL, "MAIN_LEVEL = 0.95\n")]),
    ("M8d: ref's collapse never required", [(V_B, "    B_ok = A_ok and B_E2[\"hi\"] < 0\n")]),
    ("M8e: ref must be floored in every seed", [(V_FRAC, "REF_FLOOR_FRAC = 1.0 ")]),
    ("M8f: ref floored in half the seeds is enough", [(V_FRAC, "REF_FLOOR_FRAC = 0.5 ")]),
    ("M8g: the floor margin dropped", [(V_K, "FLOOR_K = 0.0 ")]),
    ("M8h: the floor at chance instead of the label draw", [(V_F, "    F, s = CHANCE, float(f.std(ddof=1))\n")]),
    ("M8i: the pairing broken", [(V_PAIR, '        return deltas(arm, ep, win) - deltas("ref", ep, win)[::-1]\n')]),
    ("M8j: unit median instead of the mean",
     [(V_UMEAN, "        m, k = float(np.median(u[key][ends[t]])), 0\n")]),
    ("M8k: a censored arm counted as a tie", [(V_CENS, '    if ta is None:\n        return "tie"\n')]),
    ("M8l: sign test at 0.2", [(V_ALPHA, "SIGN_ALPHA = 0.2 ")]),
    ("M8m: ties counted in the sign test's n", [(V_SIGN_N, "        p = sign_test(max(e, l_), len(valid))\n")]),
    ("M8n: T_half on raw accuracy instead of the fit fraction",
     [(V_HALF, "        if online(pt).loc[t] < HALF:\n")]),
    ("M8o: (C) read in the main window", [(V_CAPWIN, "CAP_WIN = (51, 100) ")]),
    ("M8p: cap_both's (C) reads the parallel rows only",
     [(V_CAPPED, 'CAPPED = {"cap_par": ("rows_par",), "cap_perp": ("rows_perp",), "cap_both": ("rows_par",)}\n')]),
    ("M8q: the cross-arm stream check off", [(V_STREAMS, "    if False:\n")]),
    ("M8r: IMPAIRED against ref's early fit itself", [(V_IMP, "            hits += f_arm < f_ref\n")]),
    ("M8s: a task's first row taken as its end point (E2)",
     [(V_ENDS_USE, "    ends = {int(t_): i for i, t_ in reversed(list(enumerate(u['task'])))}\n"
                   "    base, nans = unit_mean(u[key][ends[1]])\n")]),
]

# --------------------------------------------------------------------------

CHECKS = {
    "S0_S-el-unchanged": ("S0 the shared runner's EL path reproduces the 0916 shards (memo_acc fixed)", s0, S0_MUT,
                          s0.__doc__, EL_RUNNER),
    "S4_S-wiring-ee": ("S4 an EE update and the EE diagnostics are the host's ELU->ELU net's", s4, S4_MUT,
                       s4.__doc__, EL_RUNNER),
    "S5_S-identical": ("S5 all four arms share task 1 bit for bit on the EE path", s5, S5_MUT, s5.__doc__, EL_RUNNER),
    "S7_S-repro": ("S7 the same EE call twice gives the same bits", s7, S7_MUT, s7.__doc__, EL_RUNNER),
    "S8_S-verdict": ("S8 the verdict returns the designed labels and estimates on synthetic shards", s8, S8_MUT,
                     s8.__doc__, VERDICT),
    "S9_S-cli": ("S9 the CLI writes the in-process EE shard with this experiment's provenance", s9, S9_MUT,
                 s9.__doc__, EE_RUNNER),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma separated check keys or prefixes, e.g. S4; 'cost' for S-cost")
    args = ap.parse_args()
    global DUMP_PATH
    keys = [k for k in CHECKS if not args.only or any(k.startswith(p) for p in args.only.split(","))]
    if args.only:
        SCR.mkdir(parents=True, exist_ok=True)
        DUMP_PATH = SCR / "checks_partial.json"
    RESULTS.update({"run_id": "mucap_ee_0917", "started_at": dt.datetime.now().astimezone().isoformat(),
                    "spec": "specs/spec_mucap_ee_0917.md",
                    "prereg_commit": load(EE_RUNNER).PREREG_COMMIT,
                    "code_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (EL_RUNNER, EE_RUNNER, VERDICT, Path(__file__),
                                              REPO / "src" / "mucap_el_0916.py")},
                    "torch": torch.__version__, "threads": torch.get_num_threads()})
    if not args.only:
        s1_3_record()
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

#!/usr/bin/env python3
"""Pre-run checks for altlabels_cifar_0923 (spec §6).  Writes results/altlabels_cifar_0923/checks.json.

    python3 analysis/altlabels_cifar_0923/checks.py all
    python3 analysis/altlabels_cifar_0923/checks.py smoke          # just the gpu smoke runs
    python3 analysis/altlabels_cifar_0923/checks.py s1 s3 ...      # one or more checks

S1  S-off, all in the same R = 10 x std layout, same device, same graph setting:
    (a) the UNMODIFIED parent script (run from the read-only reference clone) vs the modified
        engine with default flags: per_task rows aligned by (seed, cond, task), the snapshot
        arrays and the checkpoint's P/m/v/tc, all bit-identical.
    (b) `--schedule abab` tasks 1-2 vs `--schedule iid` tasks 1-2: identical.
    (c) the trace eval on (--hit-every 100) vs off: identical trajectories, LR and SNA.
    (d) against the archived R = 20 (raw+std) battle snapshots: the max abs difference, reported,
        not a verdict -- the batched-BLAS reduction order depends on the slot layout.
S2  S-schedule: labels.npz is the stream's 1st/2nd/3rd draw with its class counts and pairwise
    agreements; abab t3 = t1 and t4 = t2; aaaa every task = t1; abc t4 = t1; abab t3 is neither
    the iid stream's 3rd draw nor the iid run's t3 rows.
S3  S-trace: one trace eval leaves P, the Adam moments, tc, the Snake state, the generators and
    the accumulators byte-identical, and the E-stop15 reference's task-1 stop_eval is our hit999.
S4  S-fork: the A bundle from LR_abab ckpts/t02.pt reproduces the main run's task 3 exactly --
    every 100-step trace row (correct, ce, margin, n1, n2, n3, sig_med) up to the bundle's end
    and hit99/hit999 per slot -- and the restored start state is the checkpoint's.
S5  S-stop: the stop chain (LR, iid, seed 0, tasks 1-15, R = 1, eager, cuda) reproduces
    E-stop15_taskends.csv's (steps, stop_eval, n1) exactly.  S5b runs the same chain on the cpu
    and reports it: it does NOT reproduce the reference, which is why the chains run on cuda.
S6  provenance carries schedule / hit_every / keep_ckpts / stop / labels_sha256 / run_id / parent /
    tc_at_task_end / perm_consumption_rule; tc == 30,000 * t (full runs) or the summed steps
    (stop chains); a resume refuses a checkpoint written under another schedule/stop/hit setting.
S7  cost, with the trace eval on: one task's wall clock x 50, solo and in parallel.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src import altlabels_cifar_0923 as A              # noqa: E402
from src import rlcifar_mlp_battle_0918 as B           # noqa: E402
from src import pmnist_0905 as H                       # noqa: E402
from src import pmnist_rlcifar_0907 as RC              # noqa: E402

OUT = REPO / "results" / "altlabels_cifar_0923"
SMOKE = OUT / "_smoke"
CLONE = Path("/home/issan/Projects/claude/proj_004_drift")       # read-only reference clone
PAR_SNAP = Path("/home/issan/Projects/obsidian-research-data/rlcifar_mlp_battle_0918"
                "/results/rlcifar_mlp_battle_0918")
ESTOP = Path("/home/issan/Projects/obsidian-research-data/rl_width_posthoc_0922/replay2"
             "/E-stop15_taskends.csv")
PARAMS = ("W1", "b1", "W2", "b2", "W3", "b3")
TRCOLS = ("step", "correct", "ce", "margin_med", "n1", "n2", "n3", "sig_med")
RES: dict[str, dict] = {}


def record(name, ok, detail):
    RES[name] = {"pass": bool(ok), **detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {json.dumps(detail, default=str)[:700]}", flush=True)
    return bool(ok)


def rows_of(path: Path, cond=None, tmax=None, drop=()):
    out = []
    for r in csv.DictReader(open(path)):
        if cond is not None and r.get("cond") != cond:
            continue
        if tmax is not None and int(r["task"]) > tmax:
            continue
        out.append({k: v for k, v in r.items() if k not in drop})
    return out


def keyed(rows):
    """Rows by (seed, cond, task): the slot number depends on the layout, the key does not."""
    return {(r["seed"], r["cond"], r["task"]): {k: v for k, v in r.items() if k != "slot"}
            for r in rows}


def launch(name: str, script: Path, argv: list[str]) -> int:
    (SMOKE / "_logs").mkdir(parents=True, exist_ok=True)
    with open(SMOKE / "_logs" / f"{name}.log", "a") as fh:
        p = subprocess.Popen([sys.executable, str(script)] + argv, cwd=REPO, stdout=fh,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True,
                             env={**os.environ, "PYTHONUNBUFFERED": "1",
                                  "PYTHONDONTWRITEBYTECODE": "1"})
    return p


def snap_diff(a: Path, b: Path, keys=PARAMS):
    """(identical?, max abs difference) between two snapshot files."""
    p, q = np.load(a), np.load(b)
    ks = [k for k in keys if k in p.files and k in q.files]
    same = all(np.array_equal(p[k], q[k]) for k in ks)
    worst = max(float(np.abs(p[k].astype(np.float64) - q[k].astype(np.float64)).max()) for k in ks)
    return same, worst


# --------------------------------------------------------------------------
# the runs the checks read
# --------------------------------------------------------------------------

RUNNER = REPO / "src" / "altlabels_cifar_0923.py"
PARENT_RUNNER = CLONE / "src" / "rlcifar_mlp_battle_0918.py"

SMOKES = {
    # name              script          argv (after "run")
    "LR_abab": (RUNNER, ["--arm", "LR", "--schedule", "abab", "--tasks", "3",
                         "--hit-every", "100", "--keep-ckpts"]),
    "LR_iid": (RUNNER, ["--arm", "LR", "--schedule", "iid", "--tasks", "3",
                        "--hit-every", "100", "--keep-ckpts"]),
    "LR_iid_plain": (RUNNER, ["--arm", "LR", "--schedule", "iid", "--tasks", "2"]),
    "LR_parent": (PARENT_RUNNER, ["--arm", "LR", "--tasks", "2"]),
    "SNA_abab": (RUNNER, ["--arm", "SNA", "--schedule", "abab", "--tasks", "2",
                          "--hit-every", "100"]),
    "SNA_plain": (RUNNER, ["--arm", "SNA", "--schedule", "abab", "--tasks", "2"]),
}
# which smoke runs may share the GPU; LR_abab runs alone first so S7 has a solo step_ms
GROUPS = [["LR_abab"], ["LR_parent", "LR_iid_plain"], ["LR_iid", "SNA_abab", "SNA_plain"]]


def smoke(device="cuda") -> None:
    for group in GROUPS:
        procs = {}
        for n in group:
            if (SMOKE / n / "provenance.json").exists():
                continue
            script, argv = SMOKES[n]
            shutil.rmtree(SMOKE / n, ignore_errors=True)
            procs[n] = launch(n, script, ["run"] + argv +
                              ["--seeds", "0-9", "--conds", "std", "--device", device,
                               "--threads", "2", "--out", str(SMOKE / n)])
            print(f"[{time.strftime('%T')}] smoke {n} pid {procs[n].pid}", flush=True)
        for n, p in procs.items():
            rc = p.wait()
            if rc != 0 or not (SMOKE / n / "provenance.json").exists():
                raise SystemExit(f"smoke {n} failed (rc {rc}); see {SMOKE}/_logs/{n}.log")
            print(f"[{time.strftime('%T')}] smoke {n} done", flush=True)


# --------------------------------------------------------------------------

def s1() -> bool:
    ok = True
    # (a) the unmodified parent script vs the modified engine, default flags, same layout
    a = keyed(rows_of(SMOKE / "LR_iid_plain" / "per_task.csv"))
    b = keyed(rows_of(SMOKE / "LR_parent" / "per_task.csv"))
    diff = [k for k in b if a.get(k) != b[k]]
    bad, worst, n = [], 0.0, 0
    for s in range(10):
        for t in range(3):
            same, w = snap_diff(B.snapshot_path(SMOKE / "LR_iid_plain", "LR", "std", s, t),
                                B.snapshot_path(SMOKE / "LR_parent", "LR", "std", s, t))
            n += 1
            worst = max(worst, w)
            if not same:
                bad.append(f"seed{s}/t{t:02d}")
    ca = torch.load(SMOKE / "LR_iid_plain" / "ckpt.pt", map_location="cpu", weights_only=False)
    cb = torch.load(SMOKE / "LR_parent" / "ckpt.pt", map_location="cpu", weights_only=False)
    ck_ok = (ca["tc"] == cb["tc"]) and all(
        torch.equal(x, y) for k in ("P", "m", "v") for x, y in zip(ca[k], cb[k]))
    ok &= record("S1a modified engine (defaults) == the unmodified parent script",
                 set(a) == set(b) and not diff and not bad and ck_ok,
                 {"rows": len(a), "rows_ref": len(b), "rows_differing": len(diff),
                  "first_diff": ([{k: (a[diff[0]].get(k), b[diff[0]][k])
                                   for k in b[diff[0]] if a[diff[0]].get(k) != b[diff[0]][k]}]
                                 if diff else None),
                  "snapshots": n, "snapshots_differing": bad[:5],
                  "snapshot_max_abs_diff": worst,
                  "ckpt_P_m_v_tc_identical": ck_ok, "tc": (ca["tc"], cb["tc"]),
                  "parent_script": str(PARENT_RUNNER)})
    # (b) abab tasks 1-2 == iid tasks 1-2, same layout and flags
    a = keyed(rows_of(SMOKE / "LR_abab" / "per_task.csv", tmax=2))
    b = keyed(rows_of(SMOKE / "LR_iid" / "per_task.csv", tmax=2))
    diff = [k for k in b if a.get(k) != b[k]]
    bad, n = [], 0
    for s in range(10):
        for t in range(3):
            same, _ = snap_diff(B.snapshot_path(SMOKE / "LR_abab", "LR", "std", s, t),
                                B.snapshot_path(SMOKE / "LR_iid", "LR", "std", s, t))
            n += 1
            if not same:
                bad.append(f"seed{s}/t{t:02d}")
    ok &= record("S1b abab t1-t2 == iid t1-t2 (rows and snapshots)",
                 not diff and not bad, {"rows": len(a), "rows_differing": len(diff),
                                        "snapshots": n, "snapshots_differing": bad[:5]})
    # (c) trace on vs off
    for arm, on, off, tmax in (("LR", "LR_iid", "LR_iid_plain", 2),
                               ("SNA", "SNA_abab", "SNA_plain", 2)):
        drop = ("hit99", "hit999", "acc_at_hit_plus_500", "min_correct_after_hit")
        a = keyed(rows_of(SMOKE / on / "per_task.csv", tmax=tmax, drop=drop))
        b = keyed(rows_of(SMOKE / off / "per_task.csv", tmax=tmax, drop=drop))
        diff = [k for k in b if a.get(k) != b[k]]
        bad, n = [], 0
        for s in range(10):
            for t in range(tmax + 1):
                same, _ = snap_diff(B.snapshot_path(SMOKE / on, arm, "std", s, t),
                                    B.snapshot_path(SMOKE / off, arm, "std", s, t),
                                    keys=PARAMS + ("V1", "V2"))
                n += 1
                if not same:
                    bad.append(f"seed{s}/t{t:02d}")
        ok &= record(f"S1c {arm}: --hit-every 100 does not perturb the trajectory",
                     not diff and not bad,
                     {"rows": len(a), "rows_differing": len(diff), "snapshots": n,
                      "snapshots_differing": bad[:5], "on": on, "off": off,
                      "keys": "W1,b1,W2,b2,W3,b3" + (",V1,V2" if arm == "SNA" else "")})
    # (d) against the archived R = 20 battle: a number, not a verdict
    rep = {}
    for name, arm, tmax in (("LR_iid_plain", "LR", 2), ("SNA_plain", "SNA", 2)):
        worst, ident = 0.0, 0
        for s in range(10):
            for t in range(tmax + 1):
                same, w = snap_diff(B.snapshot_path(SMOKE / name, arm, "std", s, t),
                                    B.snapshot_path(PAR_SNAP / arm, arm, "std", s, t))
                worst = max(worst, w)
                ident += int(same)
        rep[arm] = {"max_abs_diff": worst, "identical_snapshots": ident,
                    "snapshots": 10 * (tmax + 1)}
    return record("S1d vs the archived R=20 (raw+std) battle snapshots [report only]", True,
                  {**rep, "note": "R = 10 std-only here vs R = 20 raw+std there; the batched-BLAS "
                                  "reduction order depends on the slot layout, so this is a "
                                  "magnitude, not a verdict.  The matched baseline is the "
                                  "LR_iid arm of this experiment."}) and ok


def s2() -> bool:
    ok = True
    d = np.load(SMOKE / "LR_abab" / "labels.npz")
    seeds = [int(x) for x in d["seeds"]]
    bad, agree = [], []
    for i, s in enumerate(seeds):
        g = H.stream("rlc_labels", s)
        draws = [RC.task_labels(g).numpy() for _ in range(3)]
        if not np.array_equal(d["A"][i], draws[0]) or not np.array_equal(d["B"][i], draws[1]):
            bad.append(s)
        agree.append(float((d["A"][i] == d["B"][i]).mean()))
    prov = json.loads((SMOKE / "LR_abab" / "provenance.json").read_text())
    st = prov["labels_stats"]
    ok &= record("S2a labels.npz A/B are the stream's 1st/2nd draw, with counts and agreement",
                 not bad and np.allclose(st["agree_AB"], agree),
                 {"seeds": len(seeds), "keys": sorted(d.files), "bad": bad,
                  "shape": list(d["A"].shape), "agree_AB_median": float(np.median(agree)),
                  "agree_AB_min_max": [min(agree), max(agree)],
                  "A_counts_seed0": st["A_counts"][0], "B_counts_seed0": st["B_counts"][0],
                  "expected_agreement": 1.0 / B.N_CLASSES})
    # which labelling each task gets, from the engine's debug hook (75 steps a task is enough:
    # the question is which labelling task t is handed, not how it trains)
    dev = torch.device("cpu")
    got = {}
    for sched in ("abab", "aaaa", "abc"):
        dbg: dict = {}
        o = SMOKE / "_labels" / sched
        shutil.rmtree(o, ignore_errors=True)
        B.run("LR", [0, 1], ["std"], 4, 1, dev, o, graph=False, snapshots=False,
              progress=lambda m: None, schedule=sched, debug=dbg, run_id=A.EXPERIMENT)
        got[sched] = [y.numpy() for y in dbg["labels"]]
    eq = lambda v, i, j: bool(np.array_equal(v[i], v[j]))
    ab, aa, ac = got["abab"], got["aaaa"], got["abc"]
    ok &= record("S2b abab t3 = t1, t4 = t2, t1 != t2",
                 eq(ab, 2, 0) and eq(ab, 3, 1) and not eq(ab, 0, 1),
                 {"t3==t1": eq(ab, 2, 0), "t4==t2": eq(ab, 3, 1), "t1==t2": eq(ab, 0, 1)})
    ok &= record("S2c aaaa every task = t1", all(eq(aa, i, 0) for i in range(4)),
                 {"t2==t1": eq(aa, 1, 0), "t3==t1": eq(aa, 2, 0), "t4==t1": eq(aa, 3, 0)})
    ok &= record("S2d abc t4 = t1, t1/t2/t3 pairwise different",
                 eq(ac, 3, 0) and not eq(ac, 0, 1) and not eq(ac, 1, 2) and not eq(ac, 0, 2),
                 {"t4==t1": eq(ac, 3, 0), "t1==t2": eq(ac, 0, 1), "t2==t3": eq(ac, 1, 2),
                  "t1==t3": eq(ac, 0, 2)})
    third = {}
    for s in (0, 1):
        g = H.stream("rlc_labels", s)
        for _ in range(3):
            y = RC.task_labels(g)
        third[s] = y.numpy()
    same = [s for s in (0, 1) if np.array_equal(ab[2][s], third[s])]
    ok &= record("S2e abab t3 != the iid stream's 3rd draw", not same, {"seeds_equal": same})
    a3 = keyed([r for r in rows_of(SMOKE / "LR_abab" / "per_task.csv") if r["task"] == "3"])
    i3 = keyed([r for r in rows_of(SMOKE / "LR_iid" / "per_task.csv") if r["task"] == "3"])
    ndiff = sum(1 for k in i3 if a3[k] != i3[k])
    ok &= record("S2f abab t3 rows differ from iid t3 rows in every slot (mutation control)",
                 ndiff == len(i3) == 10, {"slots_differing": ndiff, "slots": len(i3)})
    return ok


def s3() -> bool:
    """The trace eval writes nothing; and the reference chain's task-1 stop_eval is our hit999."""
    dev = torch.device("cpu")
    # a run that stops just before a trace eval, so its checkpoint is a state mid-task
    o = SMOKE / "_readonly"
    shutil.rmtree(o, ignore_errors=True)
    B.run("SNA", [0], ["std"], 1, 2, dev, o, graph=False, snapshots=False, checkpoint=True,
          progress=lambda m: None, schedule="abab", hit_every=50, run_id=A.EXPERIMENT)
    st = torch.load(o / "ckpt.pt", map_location="cpu", weights_only=False)
    act = B.make_act("SNA")
    act.init_state(1, dev, key="s3")
    act.load_state({"V": st["V"]})
    P = [q.clone().requires_grad_(True) for q in st["P"]]
    X = B.slot_inputs(RC.Cifar10(), 0, "std", dev)[None]
    Y = torch.from_numpy(np.load(o / "labels.npz")["A"][0].astype(np.int64))[None]
    before = {"P": [q.detach().clone() for q in P], "V": [v.clone() for v in act.V],
              "g_lab": H.stream("rlc_labels", 0).get_state().clone(),
              "g_batch": st["g_batch"][0].clone(), "tc": st["tc"]}
    with torch.no_grad():                         # exactly what trace_eval's forward does
        z1, a1, z2, a2, logits = B.forward(P, X, act, train=False)
        _ = (logits.argmax(-1) == Y).sum(1)
    after_ok = (all(torch.equal(q.detach(), r) for q, r in zip(P, before["P"]))
                and all(torch.equal(v, r) for v, r in zip(act.V, before["V"]))
                and torch.equal(H.stream("rlc_labels", 0).get_state(), before["g_lab"])
                and st["tc"] == before["tc"])
    grads_none = all(q.grad is None for q in P)
    ok = record("S3a one trace eval leaves P / Snake V / tc / the streams byte-identical",
                after_ok and grads_none,
                {"P_V_tc_streams_identical": after_ok, "no_grad_left": grads_none,
                 "note": "the trajectory-level version of this is S1c"})
    # the reference chain's task-1 stop point, on our trace.  hit999 asks for 1199 of 1200, and
    # a trajectory can sit at exactly 1199 for hundreds of steps, so a one-image difference
    # between two engines moves the reported hit by that whole plateau.  What has to hold is
    # that the two engines agree the task is memorized at the reference's step, and that every
    # eval between the two answers sits on the threshold count itself.
    ref = {int(r["task"]): r for r in csv.DictReader(open(ESTOP))}
    want = int(ref[1]["stop_eval"])
    rows = rows_of(SMOKE / "LR_abab" / "per_task.csv", cond="std")
    r0 = [r for r in rows if r["seed"] == "0" and r["task"] == "1"][0]
    got, need = int(r0["hit999"]), B.need_correct(0.999)
    tr = np.load(SMOKE / "LR_abab" / "trace" / "LR_std_seed0.npz")
    m = tr["task"] == 1
    step, cnt = tr["step"][m], tr["correct"][m]
    at_ref = int(cnt[step == want][0])
    between = cnt[(step >= min(got, want)) & (step < max(got, want))]
    on_edge = bool(len(between) == 0 or (between == need).all())
    ok &= record("S3b the reference chain's task-1 stop point on our trace",
                 at_ref >= need and on_edge,
                 {"hit999": got, "E-stop15_stop_eval": want, "abs_diff": abs(got - want),
                  "need_correct": need, "correct_at_the_reference_step": at_ref,
                  "counts_between": [int(q) for q in between],
                  "all_between_on_the_threshold_count": on_edge,
                  "hit99": int(r0["hit99"]), "min_correct_after_hit": int(r0["min_correct_after_hit"]),
                  "note": "E-stop15 is the R = 1 cuda replay engine, this is R = 10 with a CUDA "
                          "graph: one image's argmax differs, and because the count sits at "
                          "exactly 1199 over that stretch the reported hit999 moves with it.  "
                          "trace/*.npz keeps the raw counts, so any threshold can be recomputed."})
    return ok


def _fork_vs_main(src: Path, o: Path, branch: str, seeds, tag: str):
    """Trace rows and hits of a bundle against the task it is supposed to reproduce (task 3)."""
    b = o / "_bundles" / f"t02_{branch}"
    bad_tr, bad_hit, nrows, start_bad = [], [], 0, []
    for s in seeds:
        m = np.load(src / "trace" / f"LR_std_seed{s}.npz")
        f = np.load(b / "trace" / f"LR_std_seed{s}.npz")
        m3 = {k: m[k][m["task"] == 3] for k in TRCOLS}
        n = len(f["step"])
        nrows += n
        for k in TRCOLS:
            if not np.array_equal(m3[k][:n], f[k]):
                bad_tr.append(f"seed{s}:{k}")
        rm = [r for r in rows_of(src / "per_task.csv")
              if r["seed"] == str(s) and r["task"] == "3"][0]
        rf = [r for r in rows_of(b / "per_task.csv") if r["seed"] == str(s)][0]
        if (rm["hit99"], rm["hit999"]) != (rf["hit99"], rf["hit999"]):
            bad_hit.append({"seed": s, "main": (rm["hit99"], rm["hit999"]),
                            "fork": (rf["hit99"], rf["hit999"])})
        same, _ = snap_diff(B.snapshot_path(src, "LR", "std", s, 2),
                            B.snapshot_path(b, "LR", "std", s, 0))
        if not same:
            start_bad.append(s)
    return {"trace_rows_compared": nrows, "cols": list(TRCOLS), "trace_mismatches": bad_tr[:5],
            "hit_mismatches": bad_hit[:3], "restored_start_differs_for_seeds": start_bad,
            "ok": not bad_tr and not bad_hit and not start_bad, "what": tag}


def s4(device) -> bool:
    """The A bundle from ckpts/t02.pt must reproduce the main run's task 3."""
    src = SMOKE / "LR_abab"
    o = SMOKE / "_fork"
    prov = json.loads((src / "provenance.json").read_text())
    if not (o / "_bundles" / "t02_A" / "provenance.json").exists():
        shutil.rmtree(o, ignore_errors=True)
        p = launch("fork", RUNNER, ["fork", "--src", str(src), "--t", "2", "--branches", "A",
                                    "--stop", "0.999,500", "--hit-every", "100",
                                    "--device", str(device), "--threads", "2", "--out", str(o)])
        if p.wait() != 0:
            raise SystemExit(f"S4 fork failed; see {SMOKE}/_logs/fork.log")
    b = o / "_bundles" / "t02_A"
    bad_tr, bad_hit, nrows, start_bad = [], [], 0, []
    for s in prov["seeds"]:
        m = np.load(src / "trace" / f"LR_std_seed{s}.npz")
        f = np.load(b / "trace" / f"LR_std_seed{s}.npz")
        m3 = {k: m[k][m["task"] == 3] for k in TRCOLS}
        n = len(f["step"])
        nrows += n
        for k in TRCOLS:
            if not np.array_equal(m3[k][:n], f[k]):
                bad_tr.append(f"seed{s}:{k}")
        rm = [r for r in rows_of(src / "per_task.csv") if r["seed"] == str(s) and r["task"] == "3"][0]
        rf = [r for r in rows_of(b / "per_task.csv") if r["seed"] == str(s)][0]
        if (rm["hit99"], rm["hit999"]) != (rf["hit99"], rf["hit999"]):
            bad_hit.append({"seed": s, "main": (rm["hit99"], rm["hit999"]),
                            "fork": (rf["hit99"], rf["hit999"])})
        same, _ = snap_diff(B.snapshot_path(src, "LR", "std", s, 2),
                            B.snapshot_path(b, "LR", "std", s, 0))
        if not same:
            start_bad.append(s)
    fr = [q for q in csv.DictReader(open(o / "forks.csv"))
          if q["branch"] == "A" and q["t"] == "2"]
    ck_tc = torch.load(src / "ckpts" / "t02.pt", map_location="cpu", weights_only=False)["tc"]
    tc_ok = all(int(q["tc"]) == ck_tc + int(q["bundle_steps"]) for q in fr)
    ok = record("S4a the A bundle from ckpts/t02.pt reproduces the main run's task 3",
                not bad_tr and not bad_hit and not start_bad and tc_ok,
                {"trace_rows_compared": nrows, "cols": list(TRCOLS),
                 "trace_mismatches": bad_tr[:5], "hit_mismatches": bad_hit[:3],
                 "restored_start_differs_for_seeds": start_bad,
                 "bundle_steps": fr[0]["bundle_steps"] if fr else None,
                 "stop_steps": sorted({r["stop_step"] for r in fr}),
                 "forks_csv_rows": len(fr),
                 "parent_sha256": fr[0]["parent_sha256"][:16] if fr else None,
                 "ckpt_tc": ck_tc, "tc_continues_from_the_ckpt": tc_ok,
                 "stop_snapshots": len(list((o / "snap").glob("*_stop.npz"))),
                 "end_snapshots": len(list((o / "snap").glob("*_end.npz")))})
    # the same machinery on the iid arm, branch "next" (= that run's own task t+1), and the
    # three branches side by side at the same fork point
    src2, o2 = SMOKE / "LR_iid", SMOKE / "_fork_iid"
    if not (o2 / "_bundles" / "t02_next" / "provenance.json").exists():
        p = launch("fork_iid", RUNNER, ["fork", "--src", str(src2), "--t", "2",
                                        "--branches", "next,C", "--stop", "0.999,500",
                                        "--hit-every", "100", "--device", str(device),
                                        "--threads", "2", "--out", str(o2)])
        if p.wait() != 0:
            raise SystemExit(f"S4b fork failed; see {SMOKE}/_logs/fork_iid.log")
    nx = _fork_vs_main(src2, o2, "next", prov["seeds"], "iid arm, branch next = its task 3")
    fr2 = list(csv.DictReader(open(o2 / "forks.csv")))
    cA = sorted(int(q["hit999"]) for q in csv.DictReader(open(o / "forks.csv"))
                if q["branch"] == "C")
    cI = sorted(int(q["hit999"]) for q in fr2 if q["branch"] == "C")
    return record("S4b branch next reproduces the iid run's task 3, and branch C is fixed by "
                  "(seed, t)", nx["ok"] and cA == cI and cA != [],
                  {**nx, "C_hit999_from_abab": cA, "C_hit999_from_iid": cI,
                   "C_agrees": cA == cI,
                   "note": "at t = 2 the abab and iid runs are the same state, and C comes from "
                           "rlc_labels_fork_t2 either way, so the two C bundles must coincide"}) and ok


def _chain_vs_estop(tag: str, device: str, threads: str):
    o = SMOKE / tag
    t0 = time.time()
    if not (o / "provenance.json").exists():
        shutil.rmtree(o, ignore_errors=True)
        p = launch(tag, RUNNER,
                   ["chain", "--arm", "LR", "--schedule", "iid", "--seeds", "0", "--conds", "std",
                    "--tasks", "15", "--device", device, "--threads", threads,
                    "--stop", "0.999,500", "--hit-every", "100", "--out", str(o)])
        if p.wait() != 0:
            raise SystemExit(f"S5 chain {tag} failed; see {SMOKE}/_logs/{tag}.log")
    wall = time.time() - t0
    seed0 = o / "seed0"
    ref = {int(r["task"]): r for r in csv.DictReader(open(ESTOP))}
    rows = rows_of(seed0 / "per_task.csv")
    ds, dh, dn, tab, tc = [], [], [], [], 0
    for r in rows:
        t = int(r["task"])
        n1 = float((np.load(B.snapshot_path(seed0, "LR", "std", 0, t))["W1"].astype(np.float64) ** 2).sum())
        n1r = float(ref[t]["n1"])
        tc += int(r["steps"])
        ds.append(abs(int(r["steps"]) - int(ref[t]["steps"])))
        dh.append(abs(int(r["hit999"]) - int(ref[t]["stop_eval"])))
        dn.append(abs(n1 - n1r) / n1r)
        tab.append({"task": t, "steps": int(r["steps"]), "steps_ref": int(ref[t]["steps"]),
                    "hit999": int(r["hit999"]), "stop_eval_ref": int(ref[t]["stop_eval"]),
                    "stop_step": int(r["stop_step"]), "tc": int(r["tc"]),
                    "n1_rel": abs(n1 - n1r) / n1r})
    prov = json.loads((seed0 / "provenance.json").read_text())
    return {"device": device, "threads": int(threads), "tasks": len(rows),
            "max_abs_dsteps": max(ds), "max_abs_dhit999": max(dh), "max_rel_dn1": max(dn),
            "n1_rel_within_1e-6": bool(max(dn) <= 1e-6),
            "tc_equals_summed_steps": int(rows[-1]["tc"]) == tc, "tc_final": int(rows[-1]["tc"]),
            "wall_s_15_tasks": round(prov["wall_clock_s"], 1),
            "wall_s_this_run": round(wall, 1) if wall > 5 else None,
            "step_ms": prov["step_ms_last_task"], "per_task": tab}


def s5() -> bool:
    """The production stop chain is the cuda R = 1 eager one -- that is E-stop15's own engine.
    The cpu chain is run too and reported: it is the same code, and it does NOT reproduce the
    reference, because float32 on the cpu moves single images across the 1199/1200 threshold and
    the stop rule then feeds each task's length into the next."""
    cu = _chain_vs_estop("LR_iid_stop_cuda", "cuda", "2")
    ok = record("S5 the stop chain (cuda, R = 1, eager; seed 0, iid, t1-15) == E-stop15",
                cu["max_abs_dsteps"] == 0 and cu["max_abs_dhit999"] == 0
                and cu["n1_rel_within_1e-6"] and cu["tc_equals_summed_steps"],
                {**cu, "note": "width_replay.py's E-stop15 ran on cuda, R = 1, eager; so does "
                               "this, and steps / stop_eval / n1 all agree"})
    cp = _chain_vs_estop("LR_iid_stop", "cpu", "8")
    record("S5b the same chain on the cpu [report only: why the chains run on cuda]", True,
           {**cp, "note": "same code, same seed, cpu float32 instead of cuda: one image's argmax "
                          "near the 1199/1200 threshold changes a task's stop step, and because "
                          "the next task starts from that state the chain drifts.  The spec's cpu "
                          "chain is therefore run on cuda (R = 1, eager) instead -- 0.9 ms/step, "
                          "about 2 min a seed, so it costs nothing."})
    return ok


def s6() -> bool:
    need = ["run_id", "parent", "schedule", "hit_every", "keep_ckpts", "stop", "labels_sha256",
            "tc_at_task_end", "perm_consumption_rule"]
    paths = {n: SMOKE / n / "provenance.json" for n in SMOKES}
    paths["fork_bundle"] = SMOKE / "_fork" / "_bundles" / "t02_A" / "provenance.json"
    paths["stop_chain_seed0"] = SMOKE / "LR_iid_stop_cuda" / "seed0" / "provenance.json"
    miss, seen, tcbad = {}, {}, {}
    for n, p in paths.items():
        if not p.exists():
            miss[n] = ["(no provenance.json)"]
            continue
        d = json.loads(p.read_text())
        need_n = need + (["restore"] if n == "fork_bundle" else [])
        if n == "LR_parent":                       # the unmodified parent script: only run_id
            need_n = ["run_id"]
        m = [k for k in need_n if k not in d]
        if m:
            miss[n] = m
        seen[n] = {k: (d.get(k) if k != "perm_consumption_rule" else "(set)") for k in need_n}
        tce = d.get("tc_at_task_end") or {}
        if n in ("LR_abab", "LR_iid", "LR_iid_plain", "SNA_abab", "SNA_plain"):
            wrong = {t: v for t, v in tce.items() if v != 30000 * int(t)}
            if wrong:
                tcbad[n] = wrong
        elif n == "fork_bundle":
            ck = json.loads(p.read_text())["restore"]
            if tce and min(tce.values()) <= 0:
                tcbad[n] = tce
            seen[n]["restore"] = {"slots": ck["slots"], "sha256": ck["sha256"][:16]}
    ok = record("S6a provenance carries the new settings and tc_at_task_end",
                not miss and not tcbad,
                {"required": need, "missing": miss, "tc_wrong": tcbad, "values": seen})
    # a resume must refuse a checkpoint written under another schedule / stop / hit setting
    dev = torch.device("cpu")
    o = SMOKE / "_resume"
    shutil.rmtree(o, ignore_errors=True)
    B.run("LR", [0], ["std"], 1, 1, dev, o, graph=False, snapshots=False, checkpoint=True,
          progress=lambda m: None, schedule="abab", hit_every=100, run_id=A.EXPERIMENT)
    refused = {}
    for tag, kw in (("schedule", {"schedule": "aaaa", "hit_every": 100}),
                    ("hit_every", {"schedule": "abab", "hit_every": 50}),
                    ("stop", {"schedule": "abab", "hit_every": 100, "stop": (0.999, 500)})):
        try:
            B.run("LR", [0], ["std"], 2, 1, dev, o, graph=False, snapshots=False, checkpoint=True,
                  resume=True, progress=lambda m: None, run_id=A.EXPERIMENT, **kw)
            refused[tag] = False
        except SystemExit as e:
            refused[tag] = "another configuration" in str(e)
    B.run("LR", [0], ["std"], 2, 1, dev, o, graph=False, snapshots=False, checkpoint=True,
          resume=True, progress=lambda m: None, schedule="abab", hit_every=100,
          run_id=A.EXPERIMENT)
    same = json.loads((o / "provenance.json").read_text())["resumed_at_task"] == [2]
    return record("S6b a resume refuses another schedule / hit_every / stop",
                  all(refused.values()) and same,
                  {**refused, "same_setting_resumes": same}) and ok


def log_step_ms(name: str) -> list[float]:
    """The per-task ms/step the runner printed, in order (provenance only keeps the last task,
    and by then the jobs that shared the GPU may already have finished)."""
    f = SMOKE / "_logs" / f"{name}.log"
    if not f.exists():
        return []
    import re
    return [float(m) for m in re.findall(r"([\d.]+) ms/step", f.read_text())]


def s7() -> bool:
    spt = B.STEPS_PER_EPOCH * 400
    # LR_abab ran alone; LR_parent + LR_iid_plain together; LR_iid + SNA_abab + SNA_plain together
    d = {}
    for n in SMOKES:
        p = SMOKE / n / "provenance.json"
        if not p.exists():
            continue
        prov = json.loads(p.read_text())
        d[n] = {"R": prov["R"], "trace": bool(prov.get("hit_every")),
                "step_ms_per_task": log_step_ms(n), "jobs_started_with": len(
                    [g for g in GROUPS if n in g][0])}
    solo = min(d["LR_abab"]["step_ms_per_task"])                  # LR (trace on), alone
    p3 = [d[n]["step_ms_per_task"][0] for n in ("LR_iid", "SNA_abab", "SNA_plain") if n in d]
    m3 = sum(p3) / len(p3)
    r1 = m3 / 3 / solo                                            # per-job cost of sharing
    est = {f"{k}_jobs": {"step_ms": solo * r1 * k, "arm_50_tasks_h": solo * r1 * k * spt * 50 / 3.6e6}
           for k in (1, 4, 5)}
    chain = None
    st = SMOKE / "LR_iid_stop_cuda" / "seed0"
    if (st / "provenance.json").exists():
        r = rows_of(st / "per_task.csv")
        prov = json.loads((st / "provenance.json").read_text())
        steps = [int(q["steps"]) for q in r]
        chain = {"device": "cuda, R = 1, eager", "tasks_measured": len(r),
                 "mean_steps_per_task": sum(steps) / len(steps),
                 "step_ms": prov["step_ms_last_task"],
                 "wall_clock_s_15_tasks": prov["wall_clock_s"],
                 "seed_50_tasks_min": prov["wall_clock_s"] / len(r) * 50 / 60,
                 "arm_5_seeds_min": prov["wall_clock_s"] / len(r) * 50 * 5 / 60,
                 "both_arms_min": prov["wall_clock_s"] / len(r) * 50 * 10 / 60}
    return record("S7 cost (with the trace eval on)", True,
                  {"per_run": d, "solo_step_ms_R10_trace": solo,
                   "mean_step_ms_when_3_share_the_gpu": m3,
                   "per_job_sharing_factor": r1,
                   "gpu_estimates": est,
                   "gpu_all_5_arms_wall_h": est["5_jobs"]["arm_50_tasks_h"],
                   "stop_chain": chain,
                   "note": "the gpu saturates, so N jobs cost about N x solo per step; the "
                           "5-arm wall clock is the four 50-task arms (LR_abc is 12 tasks and "
                           "finishes early).  The forks are ~10 bundles of ~2,500 steps each."})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+",
                    choices=["all", "smoke", "s1", "s2", "s3", "s4", "s5", "s6", "s7"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = H.setup(a.device)
    stages = (["smoke", "s1", "s2", "s3", "s4", "s5", "s6", "s7"] if "all" in a.stages
              else a.stages)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "checks.json"
    prev = json.loads(f.read_text()) if f.exists() else {}
    for st in stages:
        if st == "smoke":
            smoke(a.device)
        elif st == "s4":
            s4(device)
        else:
            {"s1": s1, "s2": s2, "s3": s3, "s5": s5, "s6": s6, "s7": s7}[st]()
    prev.update(RES)
    prev["_git"] = B.git_state()
    prev["_stages"] = stages
    prev["_when"] = time.strftime("%F %T")
    f.write_text(json.dumps(prev, indent=2, default=str))
    bad = [k for k, v in prev.items() if not k.startswith("_") and not v["pass"]]
    print(f"\nwrote {f}\nall pass: {not bad}" + (f"  FAILED: {bad}" if bad else ""))


if __name__ == "__main__":
    main()

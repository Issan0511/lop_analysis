#!/usr/bin/env python3
"""drive_rlmnist_1007 checks (specs/spec_drive_rlmnist_1007.md section 7).

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m analysis.drive_rlmnist_1007.checks

Every check runs on the real code and then on each listed mutation, which must be rejected.  A mutation
is an exact-once substitution in the target file (executed as a separate module) or a change of the
input; the real file is never edited.  Tolerances are the gamma_n bounds of spec 4.3 (u = 2^-24 / 2^-53),
never values read off the data.  Production seeds 0-9 are only run WITHOUT the observer here (their
trajectory against the l2cap ref shard and the unmodified host); no scientific quantity of them is
computed.  Writes results/drive_rlmnist_1007/checks.json; check-run outputs go to
results/drive_rlmnist_1007/check_runs/ (git-external).
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import shutil
import sys
import time
import traceback
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import drive_rlmnist_1007 as D                       # noqa: E402
from src import mucap_el_run_0916 as EL                        # noqa: E402
from src import pmnist_rlmnist_0906 as RL                      # noqa: E402
from analysis.drive_rlmnist_1007 import numerics as N          # noqa: E402
from analysis.drive_rlmnist_1007 import stats as ST            # noqa: E402
from analysis.drive_rlmnist_1007 import report as RP           # noqa: E402

RUNNER = ROOT / "src" / "drive_rlmnist_1007.py"
A2NUM = ROOT / "analysis" / "drive_cifar_c_0920" / "numerics.py"
MYNUM = ROOT / "analysis" / "drive_rlmnist_1007" / "numerics.py"
OUT = ROOT / "results" / D.RUN
SCRATCH = OUT / "check_runs"
U32, U64 = 2.0 ** -24, 2.0 ** -53
TINY32, TINY64 = float(torch.finfo(torch.float32).tiny), float(torch.finfo(torch.float64).tiny)
ZERO_Z32 = math.log(2.0 ** -24)            # the host ELU's training derivative is exactly 0 below this

IDS = ("A2-host", "A2-input", "A2-CE", "A2-Adam", "A2-self-total", "A2-certificate", "A2-nonvacuous",
       "A2-confhist", "A2-window", "A2-verdict", "A2-manifest", "A2-cost")
MUTATIONS = {
    "A2-host": ("RNG_consumption", "Adam_order", "act2_leaky", "observer_writes_m"),
    "A2-input": ("old_argmax", "old_two_back", "wrong_seed_images", "centered_input"),
    "A2-CE": ("double_mean", "conf_label_swap", "d_sign", "analytic_gate"),
    "A2-Adam": ("no_history", "V_prev", "reset_time"),
    "A2-self-total": ("omit_U", "old_W_in_U", "wrong_mu"),
    "A2-certificate": ("ignore_error", "Qplus_down", "K_one"),
    "A2-nonvacuous": ("theory_only_mu",),
    "A2-confhist": ("missing_history", "swap_conf_label", "overwrite_moment"),
    "A2-window": ("primary_leak", "fixed_200"),
    "A2-verdict": ("unit_replicates", "four_of_five", "empty_certificate_pass"),
    "A2-manifest": ("partial", "foreign_run", "forged_marker", "midway", "check_failed_marker"),
    "A2-cost": ("batch_only", "subsample_updates"),
}
CHECK_SOURCES = D.RUN_SOURCES + ("analysis/drive_rlmnist_1007/stats.py", "analysis/drive_rlmnist_1007/report.py",
                                 "analysis/drive_rlmnist_1007/checks.py", "analysis/drive_cifar_c_0920/stats.py")
# captured updates of check seed 100 (task, update into the task): the switch, consecutive updates from
# a task start, mid/late task, the dying phase (5, 8, 10)
CAPTURE = ({(1, 0), (1, 1), (1, 5999)} | {(2, s) for s in range(5)} | {(2, 37), (2, 74), (2, 75), (2, 3000), (2, 5999)}
           | {(3, 0), (3, 3000)} | {(5, 0), (5, 1), (5, 2), (5, 3000), (5, 5999)} | {(8, 0), (8, 3000), (8, 5999)}
           | {(10, 0), (10, 5999)})


def gamma(n, u=U64):
    return n * u / (1 - n * u)


def check_sources():
    return {p: D.sha(ROOT / p) for p in CHECK_SOURCES}


def validate_checks(d):
    assert d["status"] == "PASS", "checks did not pass"
    assert d["source_sha256"] == check_sources(), "stale checks: a checked source changed"
    assert set(d["checks"]) == set(IDS) and all(v["pass"] is True for v in d["checks"].values())
    for k, muts in MUTATIONS.items():
        got = d["checks"][k]["mutants"]
        assert set(got) == set(muts) and all(v is True for v in got.values()), f"{k}: a mutation was not rejected"
    assert sorted(d["host_task_end_sha256"]) == [str(s) for s in range(10)]
    assert all(d["production_seed_trajectory"][str(s)]["pass"] for s in range(10))
    return True


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def mutant_module(path, pairs, name):
    src = Path(path).read_text()
    for old, new in pairs:
        assert src.count(old) == 1, f"mutation anchor not unique in {Path(path).name}: {old!r}"
        src = src.replace(old, new, 1)
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod


def _eq(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            a, b = float(a), float(b)
        except (TypeError, ValueError):
            return False
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


def same_run(rows, arrays, info, tend, ref):
    """Mismatches between a runner result and the unmodified host's (rows, arrays, info, task-end sha)."""
    hrows, harr, hinfo, htend = ref
    bad = []
    if len(rows) != len(hrows):
        bad.append("number of rows")
    for a, b in zip(rows, hrows):
        if set(a) != set(b):
            bad.append(f"row keys t{b['task']}")
        bad += [f"row t{b['task']} {k}" for k in b if not _eq(a.get(k), b[k])]
    if set(arrays) != set(harr):
        bad.append("array keys")
    bad += [f"array {k}" for k in harr if k not in arrays or not np.array_equal(arrays[k], harr[k], equal_nan=True)]
    bad += [f"info {k}" for k in hinfo if k != "wall_clock_s" and info.get(k) != hinfo[k]]
    bad += [f"task-end sha t{t}" for t in htend if tend.get(t) != htend[t]]
    return bad


def host_run(seed, n_tasks, mnist, dev, epochs=D.EPOCHS):
    """The unmodified host: run_one('ref', act2_name='ELU1', ledger off), with debug capture of each
    task's last update to read the task-end state."""
    last = D.SPE * epochs - 1
    dbg = {"capture": {(t, last) for t in range(1, n_tasks + 1)}}
    rows, arrays, _, info = EL.run_one("ref", seed, D.LR, n_tasks, mnist, dev, epochs=epochs, ledger=False,
                                       act2_name="ELU1", debug=dbg)
    tend = {}
    for t in range(1, n_tasks + 1):
        a = dbg["steps"][(t, last)]["after"]
        tend[t] = D.SH.state_sha256(a["params"], (a["m"], a["v"], [a["tc"]]), D.EG.ELU(1.0))
    return (rows, arrays, info, tend), dbg["labels"]


def a2_measure(pairs):
    return mutant_module(A2NUM, pairs, "a2_mutant").measure


def my_numerics(pairs):
    return mutant_module(MYNUM, pairs, "my_mutant")


def caps(ctx):
    return ctx["run100"].obs.captured


def bracket_count(out, inv1):
    return int(N.bracket(out, inv1).sum())


# --------------------------------------------------------------------------
# A2-host
# --------------------------------------------------------------------------

HOST_MUTANTS = {
    "RNG_consumption": [("                    self.obs.pre(out, olds[j * BATCH:(j + 1) * BATCH], yb, t, s)\n",
                         "                    torch.rand(1, generator=self.g_batch)\n"
                         "                    self.obs.pre(out, olds[j * BATCH:(j + 1) * BATCH], yb, t, s)\n")],
    "Adam_order": [("p -= LR * (mi / c1) / ((vi / c2).sqrt() + EPS)", "p -= LR * (mi / c1) / (vi.sqrt() / c2 ** 0.5 + EPS)")],
    "act2_leaky": [('self.act1, self.act2 = EG.ELU(1.0), EL.ACT2["ELU1"]', 'self.act1, self.act2 = EG.ELU(1.0), EL.ACT2["LR"]')],
    "observer_writes_m": [("        self.conf, self.hist = cm, hm\n",
                           "        self.conf, self.hist = cm, hm\n        m[2].mul_(1.0 + 2.0 ** -20)\n")],
}


def check_host(ctx):
    dev, mnist = ctx["dev"], ctx["mnist"]
    det, ok = {}, True
    for seed, T in ((100, D.TASKS), (101, 5)):
        ref, labels = host_run(seed, T, mnist, dev)
        if seed == 100:
            out = SCRATCH / "s100"
            if out.exists():
                shutil.rmtree(out)
            _, r_o = D.run_seed(100, out, T, D.EPOCHS, "check", capture=CAPTURE)
            ctx.update(run100=r_o, out100=out, labels100=labels, host100=ref)
            disk_rows = json.loads((out / "host_rows.json").read_text())
            with np.load(out / "units.npz") as u:
                disk_ok = (all(_eq(a.get(k), b[k]) for a, b in zip(disk_rows, r_o.rows) for k in b)
                           and all(np.array_equal(u[k], r_o.arrays[k], equal_nan=True) for k in r_o.arrays))
            det["s100_disk_equals_memory"] = bool(disk_ok)
            ok &= disk_ok
        else:
            r_o = D.Run(seed, mnist, dev, observe=True)
            r_o.run(T)
        r_u = D.Run(seed, mnist, dev, observe=False)
        r_u.run(T)
        for name, r in (("observed", r_o), ("unobserved", r_u)):
            bad = same_run(r.rows, r.arrays, r.info, r.task_end_sha, ref)
            det[f"s{seed}_{name}_vs_unmodified_host"] = bad or "bit-identical"
            ok &= not bad
        rng = (torch.equal(r_o.g_lab.get_state(), r_u.g_lab.get_state())
               and torch.equal(r_o.g_batch.get_state(), r_u.g_batch.get_state()))
        det[f"s{seed}_rng_end_observed_eq_unobserved"] = bool(rng)
        ok &= rng
        det[f"s{seed}_numerical_failures"] = r_o.obs.fail_total
        ok &= not r_o.obs.fail_total
    # production seeds: unobserved runner vs the l2cap ref shard (tasks 1-5) and the unmodified host
    prod, host_sha = {}, {}
    for s in range(10):
        r_u = D.Run(s, mnist, dev, observe=False)
        r_u.run(5)
        ref, _ = host_run(s, 5, mnist, dev)
        bad = same_run(r_u.rows, r_u.arrays, r_u.info, r_u.task_end_sha, ref)
        rec = RP.record_identity(s, r_u.rows, r_u.arrays, r_u.info, {str(k): v for k, v in r_u.task_end_sha.items()},
                                 {str(k): v for k, v in ref[3].items()}, tasks=range(1, 6))
        host_sha[str(s)] = {str(t): ref[3][t] for t in range(1, 6)}
        prod[str(s)] = dict(unmodified_host=bad or "bit-identical", record=rec,
                            **{"pass": (not bad) and rec["pass_tasks_1_5"]})
        ok &= prod[str(s)]["pass"]
    ctx["host_task_end_sha256"] = host_sha
    ctx["production_seed_trajectory"] = prod
    # mutations on a short configuration (2 tasks x 2 epochs) against the unmodified host there
    ref, _ = host_run(100, 2, mnist, dev, epochs=2)
    r = D.Run(100, mnist, dev, epochs=2, observe=True)
    r.run(2)
    real_short = same_run(r.rows, r.arrays, r.info, r.task_end_sha, ref)
    ok &= not real_short
    det["short_real"] = real_short or "bit-identical"
    mut = {}
    for name, pairs in HOST_MUTANTS.items():
        mod = mutant_module(RUNNER, pairs, f"host_{name}")
        rm = mod.Run(100, mnist, dev, epochs=2, observe=True)
        rm.run(2)
        mut[name] = bool(same_run(rm.rows, rm.arrays, rm.info, rm.task_end_sha, ref))
    return dict(details=det, mutants=mut, **{"pass": bool(ok)})


# --------------------------------------------------------------------------
# A2-input
# --------------------------------------------------------------------------

INPUT_MUTANTS = {
    "old_argmax": [("        olds = self.oldY[order]", "        olds = EL.forward2(self.params, self.x, self.act1, self.act2)[4].argmax(1)[order]")],
    "old_two_back": [("self.oldY = y.clone() if t == 1 else self.y", "self.oldY = y.clone() if t == 1 else (self.oldY if t >= 3 else self.y)")],
    "wrong_seed_images": [("self.idx = RL.subset_idx(seed).to(device)", "self.idx = RL.subset_idx(seed + 1).to(device)")],
    "centered_input": [("self.x = mnist.train_x[self.idx]  ", "self.x = mnist.train_x[self.idx] - mnist.train_x[self.idx].mean(0)  ")],
}


def input_problems(r, mnist, seed, labels):
    bad = []
    x_ref = mnist.train_x[RL.subset_idx(seed)]
    if not torch.equal(r.x, x_ref):
        bad.append("images are not train_x[subset_idx(seed)]")
    if not (float(r.x.min()) >= 0 and float(r.x.max()) <= 1 and float(r.x.mean()) > 0):
        bad.append("images are not pixel/255")
    for (t, s), c in sorted(r.obs.captured.items()):
        raw, ids = c["raw"], c["raw"]["ids"]
        if not torch.equal(raw["new"], labels[t - 1][ids]):
            bad.append(f"new label t{t} u{s}")
        want_old = labels[t - 2] if t >= 2 else labels[t - 1]
        if not torch.equal(raw["old"], want_old[ids]):
            bad.append(f"old label t{t} u{s}")
    return bad


def check_input(ctx):
    dev, mnist = ctx["dev"], ctx["mnist"]
    r, labels = ctx["run100"], ctx["labels100"]
    bad = input_problems(r, mnist, 100, labels)
    for t in range(1, len(r.rows) + 1):
        with np.load(ctx["out100"] / f"t{t:02d}.npz") as d:
            if str(d["label_sha256"]) != D.tsha(labels[t - 1]):
                bad.append(f"label hash t{t}")
            if str(d["old_label_sha256"]) != D.tsha(labels[t - 2] if t >= 2 else labels[0]):
                bad.append(f"old label hash t{t}")
    # short configuration for the mutations: 3 tasks x 3 epochs, captures across the second and third tasks
    cap = {(2, s) for s in (0, 75, 150, 200)} | {(3, s) for s in (0, 75, 150, 224)}
    _, short_labels = host_run(100, 3, mnist, dev, epochs=3)
    rr = D.Run(100, mnist, dev, epochs=3, observe=True, capture=cap)
    rr.run(3)
    bad_short = input_problems(rr, mnist, 100, short_labels)
    mut = {}
    for name, pairs in INPUT_MUTANTS.items():
        mod = mutant_module(RUNNER, pairs, f"input_{name}")
        rm = mod.Run(100, mnist, dev, epochs=3, observe=True, capture=cap)
        rm.run(3)
        mut[name] = bool(input_problems(rm, mnist, 100, short_labels))
    return dict(details=dict(full=bad or "ok", short=bad_short or "ok", captures=len(r.obs.captured)),
                mutants=mut, **{"pass": not bad and not bad_short})


# --------------------------------------------------------------------------
# A2-CE
# --------------------------------------------------------------------------

def dphi_formula(z):
    return torch.where(z > 0, torch.ones_like(z), torch.expm1(z.clamp(max=0.0)) + 1)


def independent_grad(raw, gate):
    """float64 autograd of the batch-mean CE through the readout and the given gate, at the stored
    native h, logits, W3: (I, 101) and the conf-part gradient (logsumexp - mean logit).
    The comparison scales are the magnitudes of the terms each side sums -- for the reconstruction
    d + e = (J_o - J_n) + (p J - J_o) that is p|J| + |J_o| + |J_n| per image, which does not vanish when
    the task is fitted (p -> onehot(n), where |p - onehot| -> 0); for the conf part (p + 1/C)|J|."""
    lg = raw["logits"].double().requires_grad_(True)
    J = raw["P"][4].double()
    h = raw["h"].double()
    ht = torch.cat((h, torch.ones(h.shape[0], 1, dtype=h.dtype)), 1)
    dl, = torch.autograd.grad(F.cross_entropy(lg, raw["new"]), lg)
    lc = raw["logits"].double().requires_grad_(True)
    dc, = torch.autograd.grad((torch.logsumexp(lc, -1) - lc.mean(-1)).mean(), lc)
    g = gate.double()
    B, C = h.shape[0], J.shape[0]
    p = torch.softmax(raw["logits"].double(), -1)
    gi = ((dl @ J) * g).T @ ht
    gci = ((dc @ J) * g).T @ ht
    terms = p @ J.abs() + J[raw["old"]].abs() + J[raw["new"]].abs()
    scale = ((terms * g).T @ ht.abs()) / B
    scale_c = ((((p + 1 / C) @ J.abs()) * g).T @ ht.abs()) / B
    return gi, gci, scale, scale_c


def ce_problems(raw, grads, num=N):
    bad = []
    gate = num.train_gate(D.EL.ACT2["ELU1"], raw["z"])
    if not torch.equal(gate, dphi_formula(raw["z"])):
        bad.append("training gate != fl(expm1(z)+1)")
    dec = num.decompose(raw["h"], raw["z"], raw["logits"], raw["P"][4], raw["old"], raw["new"], gate)
    J = raw["P"][4].double()
    p = torch.softmax(raw["logits"].double(), -1)
    jo, jn = J[raw["old"]], J[raw["new"]]
    C, B = J.shape[0], raw["h"].shape[0]
    d_ind = jo - jn
    tol_d = gamma(2) * (jo.abs() + jn.abs()) + TINY64
    if ((dec["d"][0] - d_ind).abs() > tol_d).any():
        bad.append("d != J_old - J_new")
    u_ind = p @ J - jn
    tol_u = gamma(C + 4) * (p @ J.abs() + jo.abs() + jn.abs()) + TINY64
    if ((dec["d"][0] + dec["e"][0] - u_ind).abs() > tol_u).any():
        bad.append("u != d + e")
    if (dec["e"][0].abs() > dec["eps"][0][:, None] * dec["L"][0] + dec["eb"][0]).any():
        bad.append("|e| > eps L + eb")
    gi, gci, sc, scc = independent_grad(raw, gate)
    tol_g = gamma(2 * (C + B) + 8) * sc + TINY64
    if ((dec["g"][0] - gi).abs() > tol_g).any():
        bad.append("reconstructed g != independent float64 autograd")
    tol_c = gamma(2 * (C + B) + 8) * scc + TINY64
    if ((dec["gconf"][0] - gci).abs() > tol_c).any():
        bad.append("conf gradient != independent float64 autograd")
    native = N.augmented(grads[2], grads[3])
    if ((native - dec["g"][0]).abs() > dec["gb"][0]).any():
        bad.append("native gradient outside gb")
    return bad


def dead_unit_fixture(raw, i0=0):
    """A unit pushed below the float32 zero of the training derivative on every batch image: its
    native gradient row is exactly 0, the analytic exp(z) gate is not."""
    P = [q.clone() for q in raw["P"]]
    h = raw["h"]
    with torch.no_grad():
        z = h @ P[2].T + P[3]
        P[3][i0] -= float(z[:, i0].max()) + 30.0
    W2 = P[2].clone().requires_grad_(True)
    b2 = P[3].clone().requires_grad_(True)
    act2 = D.EL.ACT2["ELU1"]
    z2 = h @ W2.T + b2
    logits = act2.phi(z2) @ P[4].T + P[5]
    gW, gb = torch.autograd.grad(F.cross_entropy(logits, raw["new"]), [W2, b2])
    fx = dict(raw, P=P, z=z2.detach(), logits=logits.detach())
    return fx, [None, None, gW, gb], i0


def check_ce(ctx):
    bad = []
    for key, c in sorted(caps(ctx).items()):
        bad += [f"{key}: {b}" for b in ce_problems(c["raw"], c["grads"])]
    # the float32 zero of the host's training derivative
    zz = torch.tensor([-40.0, -17.0, -16.7, -16.64, -16.6, -10.0, -1.0, -1e-3, 0.0, 1e-3, 2.0])
    gz = N.train_gate(D.EL.ACT2["ELU1"], zz)
    if not (torch.equal(gz, dphi_formula(zz)) and (gz[zz < ZERO_Z32 - 1e-3] == 0).all()
            and (gz[(zz > ZERO_Z32 + 1e-3) & (zz <= 0)] > 0).all() and (gz[zz > 0] == 1).all()):
        bad.append("training derivative at the float32 zero")
    fx, fgrads, i0 = dead_unit_fixture(caps(ctx)[(2, 0)]["raw"])
    gate = N.train_gate(D.EL.ACT2["ELU1"], fx["z"])
    if not ((gate[:, i0] == 0).all() and (fgrads[2][i0] == 0).all() and fgrads[3][i0] == 0):
        bad.append("dead-unit fixture is not dead")
    bad += [f"dead fixture: {b}" for b in ce_problems(fx, fgrads)]
    cap = caps(ctx)
    mut = {}
    for name, pairs in {
        "double_mean": [("    g = torch.bmm(((d + e) * gate).transpose(1, 2), ht) / B\n",
                         "    g = torch.bmm(((d + e) * gate).transpose(1, 2), ht) / B / B\n")],
        "conf_label_swap": [("    gc = torch.bmm((uconf * gate).transpose(1, 2), ht) / B\n",
                             "    gc = g - torch.bmm((uconf * gate).transpose(1, 2), ht) / B\n")],
        "d_sign": [("    d = jo - jn\n", "    d = jn - jo\n")],
    }.items():
        num = my_numerics(pairs)
        mut[name] = any(ce_problems(c["raw"], c["grads"], num) for c in cap.values())
    numg = my_numerics([("        g, = torch.autograd.grad(act.phi(zz).sum(), zz)\n",
                         "        g = torch.exp(zz.detach().clamp(max=0.0))\n")])
    gm = numg.train_gate(D.EL.ACT2["ELU1"], fx["z"])
    dm = N.decompose(fx["h"], fx["z"], fx["logits"], fx["P"][4], fx["old"], fx["new"], gm)
    native = N.augmented(fgrads[2], fgrads[3])
    mut["analytic_gate"] = bool(((native - dm["g"][0]).abs() > dm["gb"][0]).any())
    return dict(details=dict(problems=bad or "ok", captures=len(cap)), mutants=mut, **{"pass": not bad})


# --------------------------------------------------------------------------
# A2-Adam
# --------------------------------------------------------------------------

def adam_problems(c):
    bad = []
    tc0, raw = c["raw"]["tc"], c["raw"]
    if c["tc"] != tc0 + 1:
        bad.append("tc does not advance by one")
    tcn = tc0 + 1
    c1, c2 = 1 - 0.9 ** tcn, 1 - 0.999 ** tcn
    for k in range(6):
        th0, m0, v0 = (raw["P"][k].double().numpy(), raw["m"][k].double().numpy(), raw["v"][k].double().numpy())
        g = c["grads"][k].double().numpy()
        th1, m1, v1 = (c["P_after"][k].double().numpy(), c["m_after"][k].double().numpy(), c["v_after"][k].double().numpy())
        mm = 0.9 * m0 + 0.1 * g
        if (np.abs(m1 - mm) > gamma(4, U32) * (0.9 * np.abs(m0) + 0.1 * np.abs(g)) + 4 * TINY32).any():
            bad.append(f"first moment, tensor {k}")
        vm = 0.999 * v0 + 0.001 * g * g
        if (np.abs(v1 - vm) > gamma(6, U32) * vm + 6 * TINY32).any():
            bad.append(f"second moment, tensor {k}")
        ideal = -1e-3 * (m1 / c1) / (np.sqrt(v1 / c2) + 1e-8)
        if (np.abs((th1 - th0) - ideal) > gamma(12, U32) * (np.abs(th0) + np.abs(ideal)) + 12 * TINY32).any():
            bad.append(f"parameter step, tensor {k}")
    return bad


def check_adam(ctx):
    bad = []
    for key, c in sorted(caps(ctx).items()):
        bad += [f"{key}: {b}" for b in adam_problems(c)]
        if key[1] == 0 and c["raw"]["tc"] != (key[0] - 1) * D.SPE * D.EPOCHS:
            bad.append(f"{key}: tc not continued across tasks")
        if c["out"]["fail"].any():
            bad.append(f"{key}: measure fail flag")
    no_hist = a2_measure([("    ideal = -.001*inv1d*pre*mnew.double()\n",
                           "    ideal = -.001*inv1d*pre*(.1*gactual.double())\n")])
    mut = dict(no_history=False, V_prev=False, reset_time=False)
    for (t, s), c in caps(ctx).items():
        a = list(c["args"])
        mut["no_history"] |= bool(no_hist(*a)[0]["fail"].any())
        b = list(a)
        b[4] = N.augmented(c["raw"]["v"][2], c["raw"]["v"][3])[None]
        mut["V_prev"] |= bool(N.measure(*b)[0]["fail"].any())
        if t >= 2:
            q = s + 1
            r = list(a)
            r[9] = torch.tensor(1 / (1 - 0.9 ** q), dtype=torch.float64)
            r[10] = torch.tensor(1 / (1 - 0.999 ** q), dtype=torch.float64)
            mut["reset_time"] |= bool(N.measure(*r)[0]["fail"].any())
    return dict(details=dict(problems=bad or "ok"), mutants=mut, **{"pass": not bad})


# --------------------------------------------------------------------------
# A2-self-total
# --------------------------------------------------------------------------

def fsum_mean(a1):
    a = a1.double().numpy()
    return torch.tensor([math.fsum(a[:, j]) / a.shape[0] for j in range(a.shape[1])], dtype=torch.float64)


def self_total_problems(c, out, r, f0_override=None):
    bad = []
    raw = c["raw"]
    act1, act2 = r.act1, r.act2
    with torch.no_grad():
        a_old = EL.forward2(raw["P"], r.x, act1, act2)[1]
        a_new = EL.forward2(c["P_after"], r.x, act1, act2)[1]
    mu0, mu1 = fsum_mean(a_old), fsum_mean(a_new)
    f0 = c["args"][6] if f0_override is None else f0_override
    f1 = c["args"][7]
    if ((f0[0][0] - mu0).abs() > f0[1][0]).any() or ((f1[0][0] - mu1).abs() > f1[1][0]).any():
        bad.append("feature mean outside its error")
    before, after = c["args"][0][0], c["args"][1][0]
    dmu = (mu1 - mu0)
    S_ref, U_ref, T_ref = [], [], []
    for i in range(before.shape[0]):
        w0, w1 = before[i].tolist(), after[i].tolist()
        S_ref.append(math.fsum([(w1[j] - w0[j]) * float(mu0[j]) for j in range(100)] + [w1[100] - w0[100]]))
        U_ref.append(math.fsum([w1[j] * float(dmu[j]) for j in range(100)]))
        T_ref.append(math.fsum([w1[j] * float(mu1[j]) for j in range(100)] + [w1[100]]
                               + [-w0[j] * float(mu0[j]) for j in range(100)] + [-w0[100]]))
    S_ref, U_ref, T_ref = (torch.tensor(v, dtype=torch.float64) for v in (S_ref, U_ref, T_ref))
    w0a, w1a = before.abs(), after.abs()
    mue0, mue1 = f0[1][0], f1[1][0]
    k0 = torch.cat((f0[0][0], torch.ones(1, dtype=torch.float64)))
    k1 = torch.cat((f1[0][0], torch.ones(1, dtype=torch.float64)))
    act = (after - before).abs()
    tolS = out["S_error"][0] + (act[:, :100] * (f0[0][0] - mu0).abs()).sum(-1) + U64 * S_ref.abs() + TINY64
    tolU = (gamma(2 * 100 + 2) * (w1a[:, :100] * (f1[0][0] - f0[0][0]).abs()).sum(-1)
            + (w1a[:, :100] * (mue0 + mue1)).sum(-1) + (w1a[:, :100] * ((f1[0][0] - mu1).abs() + (f0[0][0] - mu0).abs())).sum(-1)
            + U64 * U_ref.abs() + TINY64)
    tolT = (gamma(2 * 101 + 2) * ((w1a * k1.abs()).sum(-1) + (w0a * k0.abs()).sum(-1)) + (w1a[:, :100] * mue1).sum(-1)
            + (w0a[:, :100] * mue0).sum(-1) + gamma(4) * (out["total"][0].abs() + T_ref.abs()) + 2 * TINY64)
    if ((out["S"][0] - S_ref).abs() > tolS).any():
        bad.append("S != independent fsum")
    if ((out["U"][0] - U_ref).abs() > tolU).any():
        bad.append("U != independent fsum")
    if ((out["total"][0] - T_ref).abs() > tolT).any():
        bad.append("total != independent fsum")
    return bad


def check_self_total(ctx):
    r = ctx["run100"]
    bad = []
    sel = [k for k in sorted(caps(ctx)) if k[0] in (1, 2, 5, 8)]
    for key in sel:
        c = caps(ctx)[key]
        bad += [f"{key}: {b}" for b in self_total_problems(c, c["out"], r)]
        fresh = N.full_features(c["raw"]["P"], r.x, r.act1, r.act2, EL.forward2)
        if not all(torch.equal(a, b) for a, b in zip(fresh, c["args"][6])):
            bad.append(f"{key}: reused features != recomputed features")
    omit_u = a2_measure([("    total = newmean-oldmean\n", "    total = S\n")])
    old_w = a2_measure([("U, Ue = dot(after[...,:-1],(mu1-mu)[:,None,:])", "U, Ue = dot(before[...,:-1],(mu1-mu)[:,None,:])")])
    x_other = ctx["mnist"].train_x[RL.subset_idx(101)]
    mut = dict(omit_U=False, old_W_in_U=False, wrong_mu=False)
    for key in sel:
        c = caps(ctx)[key]
        a = list(c["args"])
        mut["omit_U"] |= bool(self_total_problems(c, omit_u(*a)[0], r))
        mut["old_W_in_U"] |= bool(self_total_problems(c, old_w(*a)[0], r))
        b = list(a)
        b[6] = N.full_features(c["raw"]["P"], x_other, r.act1, r.act2, EL.forward2)
        mut["wrong_mu"] |= bool(self_total_problems(c, N.measure(*b)[0], r, f0_override=c["args"][6]))
    return dict(details=dict(problems=bad or "ok", updates=len(sel)), mutants=mut, **{"pass": not bad})


# --------------------------------------------------------------------------
# A2-certificate
# --------------------------------------------------------------------------

def crafted_certificate_fixture(c, target=1e-7):
    """Unit i0 with no current gradient (gate 0 on the batch), a small positive history H, and an actual
    stored step whose rounding (inside the Adam band) makes S > 0 although -lr/c1 Qminus < 0: the real
    certificate must leave it UNRESOLVED; ignoring the error certifies a wrong sign."""
    a = [x.clone() if torch.is_tensor(x) else x for x in c["args"]]
    before, after, mprev, mnew, vnew, gact, f0, f1, dec, inv1, inv2, conf, hist = a
    dec = {k: v.clone() for k, v in dec.items()}
    f0 = tuple(x.clone() for x in f0)
    f1 = list(x.clone() for x in f1)
    mu, mu1 = f0[0], f1[0]
    kd = torch.cat((mu, torch.ones_like(mu[:, :1])), -1)[:, None, :]
    pre = 1 / (torch.sqrt(vnew.double() * inv2.double()) + 1e-8)
    H_all = (kd[0, 0] * pre[0] * mprev[0]).sum(-1)
    i0 = int(H_all.abs().argmax())
    H0 = float(H_all[i0])
    dec["gate"][0, :, i0] = 0
    dec["g"][0, i0] = 0
    dec["gconf"][0, i0] = 0
    dec["gb"][0, i0] = 128 * TINY32
    gact[0, i0] = 0
    lam = target / (1e-3 * float(inv1) * 0.9 * H0)
    mprev[0, i0] = (lam * mprev[0, i0]).float().double()
    mnew[0, i0] = (0.9 * mprev[0, i0]).float().double()
    conf[0, i0] = 0
    ideal = -.001 * inv1.double() * pre[0, i0] * mnew[0, i0]
    band = gamma(12, U32) * (before[0, i0].abs() + ideal.abs()) + 12 * TINY32
    pd = 0.5 * band * torch.sign(kd[0, 0])
    after[0, i0] = before[0, i0] + ideal + pd
    newmean = (after[0, i0] * torch.cat((mu1[0], torch.ones(1, dtype=torch.float64)))).sum()
    oldmean = (before[0, i0] * kd[0, 0]).sum()
    f1[2] = f1[2].clone()
    f1[2][0, i0] = f0[2][0, i0] + (newmean - oldmean)
    return (before, after, mprev, mnew, vnew, gact, f0, tuple(f1), dec, inv1, inv2, conf, hist), i0


def check_certificate(ctx):
    bad = []
    # A2's strict-boundary toy (the registered decision rule with the error term)
    def decide(qm, qp, e):
        return (-.001 * qm + e < 0, -.001 * qp - e > 0)
    toy = (decide(0, 0, 0) == (False, False) and decide(1, 2, 0) == (True, False) and decide(-2, -1, 0) == (False, True)
           and decide(1, 2, .002) == (False, False) and decide(-1, 2, 0) == (False, False) and decide(2, 2, 0) != (False, False))
    if not toy:
        bad.append("decision toy")
    for key, c in sorted(caps(ctx).items()):
        o = c["out"]
        if o["fail"].any() or o["certificate_violation"].any():
            bad.append(f"{key}: fail/violation")
        if bracket_count(o, c["args"][9]):
            bad.append(f"{key}: actual S outside the theory interval")
        both = (o["cert_down"] + o["cert_up"] > 1).any()
        if both:
            bad.append(f"{key}: both certificates")
    fx, i0 = crafted_certificate_fixture(caps(ctx)[(2, 3000)])
    o, _, _ = N.measure(*fx)
    if o["fail"].any() or o["cert_down"][0, i0] or o["cert_up"][0, i0] or not o["S"][0, i0] > 0:
        bad.append("crafted fixture: real certificate is not UNRESOLVED with S > 0")
    ig = a2_measure([("    hi = -.001*inv1d*qm + move_error + Se\n    lo = -.001*inv1d*qp - move_error - Se\n",
                      "    hi = -.001*inv1d*qm\n    lo = -.001*inv1d*qp\n")])
    qp_down = a2_measure([("    hi = -.001*inv1d*qm + move_error + Se\n", "    hi = -.001*inv1d*qp + move_error + Se\n")])
    k_one = a2_measure([("    K = torch.bmm(kd*pre,dec['ht'].transpose(1,2))\n",
                         "    K = torch.ones_like(torch.bmm(kd*pre,dec['ht'].transpose(1,2)))\n")])
    mut = dict(ignore_error=bool(ig(*fx)[0]["fail"][0, i0]), Qplus_down=False, K_one=False)
    for c in caps(ctx).values():
        a = list(c["args"])
        mut["Qplus_down"] |= bool(qp_down(*a)[0]["fail"].any())
        mut["K_one"] |= bool(bracket_count(k_one(*a)[0], a[9]) > 0)
    return dict(details=dict(problems=bad or "ok"), mutants=mut, **{"pass": not bad})


# --------------------------------------------------------------------------
# A2-nonvacuous
# --------------------------------------------------------------------------

def check_nonvacuous(ctx):
    bad = []
    t2 = [c for (t, s), c in caps(ctx).items() if t == 2]
    mu_norm = min(float(c["args"][6][0].norm()) for c in t2)
    w_step = max(float((c["args"][1] - c["args"][0]).abs().sum()) for c in t2)
    up_step = max(float((c["args"][7][0] - c["args"][6][0]).abs().sum()) for c in t2)
    s_ne_total = any(bool(((c["out"]["S"] - c["out"]["total"]).abs() > c["out"]["S_error"] + c["out"]["closure"] +
                           gamma(8) * (c["out"]["S"].abs() + c["out"]["total"].abs())).any()) for c in t2)
    with np.load(ctx["out100"] / "t02.npz") as d:
        certs = float(d["sum"][:, :, N.KEYS.index("cert_down")].sum() + d["sum"][:, :, N.KEYS.index("cert_up")].sum())
    if not (mu_norm > 0 and w_step > 0 and up_step > 0 and s_ne_total and certs > 0):
        bad.append("vacuous")
    flip = "torch.cat((kd[...,:-1].flip(-1),kd[...,-1:]),-1)"
    m = a2_measure([("    K = torch.bmm(kd*pre,dec['ht'].transpose(1,2))\n",
                     f"    K = torch.bmm({flip}*pre,dec['ht'].transpose(1,2))\n"),
                    ("    H, He = dot(kd*pre,mprev)\n", f"    H, He = dot({flip}*pre,mprev)\n")])
    mut = dict(theory_only_mu=any(bracket_count(m(*c["args"])[0], c["args"][9]) > 0 for c in caps(ctx).values()))
    return dict(details=dict(min_mu_norm=mu_norm, max_w2_step=w_step, max_upstream_step=up_step,
                             S_differs_from_total=s_ne_total, task2_certificates_positive=certs > 0),
                mutants=mut, **{"pass": not bad})


# --------------------------------------------------------------------------
# A2-confhist
# --------------------------------------------------------------------------

def check_confhist(ctx):
    bad = []
    cap = caps(ctx)
    seq = [cap[(2, s)] for s in range(5)]
    M_start = seq[0]["args"][2]
    if not torch.equal(seq[0]["args"][12], M_start):
        bad.append("history at the task start != the actual first moment")
    gci = []
    for s, c in enumerate(seq):
        _, gc, _, scc = independent_grad(c["raw"], c["raw"]["gate"])
        gci.append((gc, scc))
        hist = c["args"][12][0]
        want = M_start[0] * 0.9 ** s
        if ((hist - want).abs() > gamma(s + 2) * want.abs() + TINY64).any():
            bad.append(f"history at update {s} != 0.9^s M_start")
        if s >= 1:
            conf = c["args"][11][0]
            ref = sum(0.1 * 0.9 ** (s - 1 - q) * gci[q][0] for q in range(s))
            scale = sum(0.1 * 0.9 ** (s - 1 - q) * gci[q][1] for q in range(s))
            if ((conf - ref).abs() > gamma(2 * (10 + 16) + 8 + 2 * s) * scale + TINY64).any():
                bad.append(f"conf EMA at update {s} != independent")
        if c["out"]["fail"].any():
            bad.append(f"fail flag (components) at update {s}")
    miss = a2_measure([("    hm = hist*.9\n", "    hm = hist*0\n")])
    swap = a2_measure([("    cm = conf*.9 + dec['gconf']*.1\n", "    cm = conf*.9 + (dec['g']-dec['gconf'])*.1\n")])
    _, _, hm1 = miss(*seq[0]["args"])
    _, cm1, _ = swap(*seq[0]["args"])
    gc0, scc0 = gci[0]
    mut = dict(missing_history=bool(((hm1[0] - M_start[0] * 0.9).abs() > gamma(3) * (M_start[0] * 0.9).abs() + TINY64).any()),
               swap_conf_label=bool(((cm1[0] - 0.1 * gc0).abs() > gamma(2 * (10 + 16) + 10) * 0.1 * scc0 + TINY64).any()))
    dev, mnist = ctx["dev"], ctx["mnist"]
    ref, _ = host_run(100, 2, mnist, dev, epochs=2)
    mod = mutant_module(RUNNER, [("        self.conf, self.hist = cm, hm\n",
                                  "        self.conf, self.hist = cm, hm\n        m[2].copy_(cm[0][:, :-1].float())\n")], "confhist_ow")
    rm = mod.Run(100, mnist, dev, epochs=2, observe=True)
    rm.run(2)
    mut["overwrite_moment"] = bool(same_run(rm.rows, rm.arrays, rm.info, rm.task_end_sha, ref))
    return dict(details=dict(problems=bad or "ok"), mutants=mut, **{"pass": not bad})


# --------------------------------------------------------------------------
# A2-window, A2-verdict
# --------------------------------------------------------------------------

def check_window(ctx):
    bad = []
    w = ST.window(np.r_[np.full(7, -2.0), np.full(73, -0.1)])
    if not (w["label"] == "WINDOW_IDENTIFIED" and w["break_epoch"] == 7 and w["boundary_updates"] == 525):
        bad.append("two-level")
    if ST.window(np.zeros(80))["label"] != "WINDOW_NOT_IDENTIFIED":
        bad.append("constant")
    if ST.window(np.r_[np.full(7, -0.1), np.full(73, -2.0)])["label"] != "WINDOW_NOT_IDENTIFIED":
        bad.append("reversed")
    if ST.window(np.r_[np.full(7, 2.0), np.full(73, 3.0)])["label"] != "WINDOW_NOT_IDENTIFIED":
        bad.append("early >= 0")
    tie = ST.window(np.r_[np.full(10, -1.0), np.zeros(60), np.full(10, -1.0)])
    if not (tie["label"] == "WINDOW_IDENTIFIED" and tie["break_epoch"] == 10):
        bad.append("break tie -> smallest b")
    exact = ST.window(np.r_[np.full(5, -3.0), np.full(75, 1.0)])
    if not (exact["break_epoch"] == 5 and exact["bic1"] == -math.inf):
        bad.append("RSS1 = 0 -> two-level")
    try:
        ST.window(np.zeros(400))
        bad.append("accepts 400 points")
    except AssertionError:
        pass
    def leak():
        RP.calibration_series(None, D.MAIN_SEEDS)
    try:
        leak()
        primary = False
    except ValueError:
        primary = True
    mut = dict(primary_leak=primary, fixed_200=(w["boundary_updates"] != 200))
    return dict(details=dict(problems=bad or "ok"), mutants=mut, **{"pass": not bad})


def synthetic_run(root, b0, shift=False):
    """Synthetic reduced shards with known counts/transport for the report arithmetic (no model run):
    main tasks: cert_down 30/75 in epochs < b0 and 5/75 after, cert_up 10/75 (seed 4: 0), S per epoch
    -1/64 then +1/1024 (seed 4: -1/64 throughout), U -1/256 per epoch; zbar_l2 at the dense points is the
    running sum of total (binary fractions, so every sum is exact).  shift=True misplaces one window."""
    K = {k: i for i, k in enumerate(N.KEYS)}
    for s in D.MAIN_SEEDS + D.CAL_SEEDS:
        out = Path(root) / f"s{s}"
        out.mkdir(parents=True, exist_ok=True)
        zrows, tasks, steps = [], [], []
        for t in range(1, D.TASKS + 1):
            a = np.zeros((D.EPOCHS, 100, len(N.KEYS)))
            down = np.where(np.arange(D.EPOCHS) < b0, 30.0, 5.0)[:, None]
            up = 0.0 if s == 4 else 10.0
            a[:, :, K["cert_down"]] = down
            a[:, :, K["cert_up"]] = up
            a[:, :, K["uncertain"]] = 75 - down - up
            S = np.where(np.arange(D.EPOCHS) == 0, -1 / 64, (-1 / 64) if s == 4 else 1 / 1024)[:, None] * np.ones((1, 100))
            U = np.full((D.EPOCHS, 100), -1 / 256)
            a[:, :, K["S_sum"]] = S
            a[:, :, K["U_sum"]] = U
            a[:, :, K["total_sum"]] = S + U
            a[:, :, K["total_negative"]] = np.minimum(S + U, 0)
            a[:, :, K["total_positive"]] = np.maximum(S + U, 0)
            D.save_npz(out / f"t{t:02d}.npz", dict(
                sum=a, max=np.zeros_like(a), l1=np.zeros((D.EPOCHS, 100, len(N.L1_KEYS))),
                aux=np.zeros((D.EPOCHS, 100, len(N.AUX_KEYS))), keys=np.array(N.KEYS), l1_keys=np.array(N.L1_KEYS),
                aux_keys=np.array(N.AUX_KEYS), seed=s, task=t, epochs=D.EPOCHS, steps_per_epoch=D.SPE,
                fail_count=0, first_failure=np.array([-1, -1])))
            if t >= 2:
                cum = np.concatenate([np.zeros((1, 100)), np.cumsum(S + U, 0)])
                for p_ in EL.DENSE:
                    z = cum[p_ // D.SPE].copy()
                    if shift and t == 3 and p_ == 375:
                        z = cum[p_ // D.SPE - 1].copy()
                    zrows.append(z)
                    tasks.append(t)
                    steps.append(p_)
        np.savez(out / "units.npz", task=np.array(tasks, float), step=np.array(steps, float), zbar_l2=np.stack(zrows))


def check_verdict(ctx):
    bad = []
    root = SCRATCH / "synthetic_report"
    if root.exists():
        shutil.rmtree(root)
    synthetic_run(root, 3)
    res = RP.compute(root, 3)
    pd0 = (3 * 30 + 77 * 5) / (80 * 75)
    rows = {r["seed"]: r for r in res["per_seed"]}
    if not (abs(rows[0]["p_down"] - pd0) <= gamma(8) * pd0 and abs(rows[0]["p_up"] - 10 / 75) <= gamma(8) * 10 / 75
            and rows[4]["p_up"] == 0):
        bad.append("synthetic p_down/p_up")
    if not (res["M1x"]["label"] == "UNRESOLVED" and res["M1x"]["positive"] == 1 and res["M1x"]["nonzero"] == 5):
        bad.append("synthetic M1x")
    if not (res["M3x"]["label"] == "BOUNDARY_ENRICHED"
            and all(abs(v - 25 / 75) <= gamma(8) for v in res["M3x"]["differences"])):
        bad.append("synthetic M3x")
    if res["M4x"] != dict(U_main_sum_all_negative=True, S_main_sum_all_negative=False, first_epoch_S_share_gt_half_all=False):
        bad.append("synthetic M4x")
    if any(r["telescope_outside_bound"] for r in res["per_seed"]):
        bad.append("synthetic telescope (consistent) rejected")
    shutil.rmtree(root)
    synthetic_run(root, 3, shift=True)
    shifted = RP.compute(root, 3)
    if not any(r["telescope_outside_bound"] for r in shifted["per_seed"]):
        bad.append("synthetic telescope (misplaced window) accepted")
    if RP.compute(root, None)["M3x"]["label"] != "WINDOW_NOT_IDENTIFIED":
        bad.append("no window -> M3x undefined")
    shutil.rmtree(root)
    for values in itertools.product([-1, 0, 1], repeat=5):
        want = ("CROSS_TASK_SINK_CONDITION_HOLDS" if values == (1,) * 5 else "NOT_SUPPORTED" if values == (-1,) * 5
                else "UNRESOLVED")
        if ST.directional(values)["label"] != want:
            bad.append(f"label {values}")
        want3 = "BOUNDARY_ENRICHED" if values == (1,) * 5 else "LATER_ENRICHED" if values == (-1,) * 5 else "UNRESOLVED"
        if ST.directional(values, "BOUNDARY_ENRICHED", "LATER_ENRICHED")["label"] != want3:
            bad.append(f"M3 label {values}")
    if ST.directional([1] * 5)["one_sided_p"] != 1 / 32:
        bad.append("p of 5/5")
    if ST.directional([float("nan")] * 5)["label"] != "DIVERGED":
        bad.append("nan")
    if ST.directional([1] * 4)["label"] != "INCOMPLETE":
        bad.append("missing seed")
    if ST.certificate_status(3, 0) != "CERTIFICATE_CONSISTENT" or ST.certificate_status(1, 1) != "CHECK_FAILED":
        bad.append("M2x")
    mut = dict(unit_replicates=ST.directional([1] * 500)["label"] == "INCOMPLETE",
               four_of_five=ST.directional([1, 1, 1, 1, -1])["label"] == "UNRESOLVED",
               empty_certificate_pass=ST.certificate_status(0, 0) == "NO_CERTIFIABLE_EVENTS")
    return dict(details=dict(problems=bad or "ok", sign_cases=3 ** 5), mutants=mut, **{"pass": not bad})


# --------------------------------------------------------------------------
# A2-manifest, A2-cost
# --------------------------------------------------------------------------

def check_manifest(ctx):
    root = SCRATCH / "manifest"
    if root.exists():
        shutil.rmtree(root)
    D.run_seed(101, root / "s101", n_tasks=2, epochs=2, mode="check")
    srcs = D.run_sources()

    def valid(run_root):
        RP.validate_seed(run_root, 101, mode="check", tasks=2, epochs=2, sources=srcs)
        return True

    ok = valid(root)

    def corrupt(name, fn):
        dst = SCRATCH / f"manifest_{name}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(root, dst)
        fn(dst / "s101")
        try:
            valid(dst)
            res = False
        except (AssertionError, FileNotFoundError, KeyError, json.JSONDecodeError):
            res = True
        shutil.rmtree(dst)
        return res

    def edit_complete(p, f):
        m = json.loads((p / "complete.json").read_text())
        f(m)
        (p / "complete.json").write_text(json.dumps(m))

    mut = dict(
        partial=corrupt("partial", lambda p: edit_complete(p, lambda m: m["files"].pop())),
        foreign_run=corrupt("foreign", lambda p: edit_complete(p, lambda m: m["identity"].update(run_id="foreign"))),
        forged_marker=corrupt("forged", lambda p: edit_complete(p, lambda m: m["files"][0].update(sha256="0" * 64))),
        midway=corrupt("midway", lambda p: (p / "complete.json").unlink()),
        check_failed_marker=corrupt("cf", lambda p: (p / "CHECK_FAILED.json").write_text("{}")))
    return dict(details=dict(valid_short_run=ok), mutants=mut, **{"pass": bool(ok)})


def validate_cost(c, tasks):
    assert c["full_images"] == D.N_IMG and c["every_update"] is True
    assert c["updates_per_task"] == D.SPE * D.EPOCHS and len(c["task_seconds"]) == tasks
    assert c["peak_rss_kb"] > 0
    return True


def check_cost(ctx):
    c = json.loads((ctx["out100"] / "cost.json").read_text())
    ok = validate_cost(c, D.TASKS)
    ts = c["task_seconds"]
    peak_gib = c["peak_rss_kb"] / 2 ** 20
    avail = [int(line.split()[1]) for line in open("/proc/meminfo") if line.startswith("MemAvailable:")][0] / 2 ** 20
    slots = max(0, min(4, int((avail - 6.0) // (1.2 * peak_gib))))
    ctx["cost"] = dict(task1_seconds=ts["1"], later_task_seconds_mean=float(np.mean([ts[str(t)] for t in range(2, D.TASKS + 1)])),
                       seed_seconds=c["seconds"], peak_rss_gib=peak_gib, mem_available_gib_at_check=avail, slots=slots)

    def rejected(bad):
        try:
            validate_cost(bad, D.TASKS)
            return False
        except AssertionError:
            return True
    mut = dict(batch_only=rejected(dict(c, full_images=16)), subsample_updates=rejected(dict(c, updates_per_task=600)))
    return dict(details=ctx["cost"], mutants=mut, **{"pass": bool(ok)})


CHECKS = (("A2-host", check_host), ("A2-input", check_input), ("A2-CE", check_ce), ("A2-Adam", check_adam),
          ("A2-self-total", check_self_total), ("A2-certificate", check_certificate),
          ("A2-nonvacuous", check_nonvacuous), ("A2-confhist", check_confhist), ("A2-window", check_window),
          ("A2-verdict", check_verdict), ("A2-manifest", check_manifest), ("A2-cost", check_cost))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "checks.json"))
    a = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_flush_denormal(True)
    dev = D.H.setup("cpu")
    ctx = dict(dev=dev, mnist=D.H.Mnist(dev))
    SCRATCH.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = {}
    for cid, fn in CHECKS:
        t1 = time.time()
        try:
            res = fn(ctx)
        except Exception as exc:  # a crashed check is a failed check, with its traceback
            res = {"pass": False, "error": repr(exc), "traceback": traceback.format_exc(), "mutants": {}}
        res["seconds"] = time.time() - t1
        results[cid] = res
        print(f"{cid}: pass={res['pass']} mutants={res.get('mutants')} ({res['seconds']:.0f}s)", flush=True)
    all_mut = all(all(results[k]["mutants"].get(m) is True for m in MUTATIONS[k]) for k in IDS)
    status = "PASS" if all(r["pass"] for r in results.values()) and all_mut else "FAIL"
    report = dict(status=status, run_id=D.RUN, prereg_commit=D.PREREG_COMMIT, source_sha256=check_sources(),
                  checks=results, mutations={k: list(v) for k, v in MUTATIONS.items()},
                  host_task_end_sha256=ctx.get("host_task_end_sha256"),
                  production_seed_trajectory=ctx.get("production_seed_trajectory"),
                  cost=ctx.get("cost"), check_seeds=list(D.CHECK_SEEDS), scientific_seeds_observed=False,
                  utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), total_seconds=time.time() - t0)
    for res in report["checks"].values():
        res["mutants"] = {k: bool(v) for k, v in res.get("mutants", {}).items()}
    D.put(a.out, report)
    print(f"checks {status} in {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()

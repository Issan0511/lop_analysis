"""drive_cifar5p1_1007 checks (spec §9).  Check seeds 100-109; seeds 0-9 only through the UNMODIFIED host.

    python -m analysis.drive_cifar5p1_1007.checks --out results/drive_cifar5p1_1007/raw/checks

Every check runs on the real thing and must pass; every listed mutant must be rejected.  The collector
(`validate_checks`) refuses a stale source hash, a missing check, an unrejected mutant or an empty PASS.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src import drive_cifar5p1_1007 as D
from src import pmnist_0905 as H
from analysis.drive_cifar5p1_1007 import report as Rp

C = D.C
IDS = ("S-host", "S-self-total", "S-bin", "S-conf", "S-window", "S-verdict", "S-graph", "S-manifest", "S-cost")
MUTANTS = {
    "S-host": ("obs_rng", "obs_write", "adam_no_bias"),
    "S-self-total": ("omit_U", "old_W_in_U", "wrong_task_mu", "J_in_U"),
    "S-bin": ("bin27", "off_by_one"),
    "S-conf": ("swap_conf_label", "no_history", "pi_1_10", "double_batch_mean", "conf_reset"),
    "S-window": ("primary_leak", "fixed_b3", "tie_larger_b", "accept_reversed"),
    "S-verdict": ("four_of_five", "unit_replicates", "missing_seed"),
    "S-graph": ("no_warmup_restore", "stale_idx"),
    "S-manifest": ("partial", "foreign_run", "forged_marker", "stale_source"),
    "S-cost": ("batch_only", "every_other_update"),
}
CHECKS_JSON = D.OUT / "checks.json"


def validate_checks(d) -> None:
    assert d["status"] == "PASS", "checks did not pass"
    assert d["source_sha256"] == D.sources(), "stale checks: sources changed since"
    assert set(d["checks"]) == set(IDS) and all(v["pass"] is True for v in d["checks"].values()), "missing/failed check"
    assert d["mutants"] == {k: list(v) for k, v in MUTANTS.items()}, "mutant list"
    assert all(d["mutants_rejected"][k][m] is True for k, v in MUTANTS.items() for m in v), "unrejected mutant"
    assert d["scientific_seeds_observed"] is False
    assert len(d["host_env"]["per_task_sha256"]) == 64 and d["cost"]["need_gb"] > 0


def rejected(fn) -> bool:
    """True when the check under test refuses the mutant (an exception or a False verdict)."""
    try:
        return fn() is False
    except (AssertionError, ValueError, RuntimeError, KeyError, IndexError, FileNotFoundError):
        return True


def mutated(module_src, fn_name, old, new, extra=None):
    assert module_src.count(old) == 1, (fn_name, old)
    ns = dict(extra or {})
    exec(compile(module_src.replace(old, new), f"<mutant {fn_name}>", "exec"), ns)
    return ns[fn_name]


def phash(ts) -> str:
    h = hashlib.sha256()
    for q in ts:
        z = q.detach().cpu().contiguous()
        h.update(str((z.dtype, tuple(z.shape))).encode())
        h.update(z.numpy().tobytes())
    return h.hexdigest()


def rows_of(p):
    return list(csv.DictReader(Path(p).open()))


# --------------------------------------------------------------------------
# pure checks
# --------------------------------------------------------------------------

def s_window(work) -> tuple[dict, dict]:
    src = Path(Rp.__file__).read_text()
    W = Rp.window
    v2 = np.r_[np.full(7, -2.0), np.full(23, -0.1)]
    cases = {
        "constant": (np.zeros(30), "WINDOW_NOT_IDENTIFIED", None),
        "constant_negative": (np.full(30, -1.0), "WINDOW_NOT_IDENTIFIED", None),
        "two_level": (v2, "WINDOW_IDENTIFIED", 7),
        "two_level_noise": (v2 + 0.01 * np.sin(np.arange(30)), "WINDOW_IDENTIFIED", 7),
        "reversed_up": (-v2, "WINDOW_NOT_IDENTIFIED", None),            # early positive
        "early_higher": (np.r_[np.full(7, -0.1), np.full(23, -2.0)], "WINDOW_NOT_IDENTIFIED", None),
        "one_outlier": (np.r_[-5.0, np.zeros(29)], "WINDOW_IDENTIFIED", 1),
    }
    out = {}
    for k, (v, lab, b) in cases.items():
        w = W(v)
        out[k] = dict(label=w["label"], break_bin=w["break_bin"])
        assert w["label"] == lab and w["break_bin"] == b, (k, w["label"], w["break_bin"])
    assert W(v2)["boundary_updates"] == 7 * D.BIN
    # an exact floating-point tie: integer data with integer segment means, RSS = 812 at b = 1 and b = 29
    tie = np.r_[-29.0, np.zeros(28), 29.0]
    rss = lambda b: float(((tie[:b] - tie[:b].mean()) ** 2).sum() + ((tie[b:] - tie[b:].mean()) ** 2).sum())  # noqa: E731
    assert rss(1) == rss(29) == 812.0 and all(rss(b) > 812.0 for b in range(2, 29))
    assert Rp.window(tie)["candidate_break_bin"] == 1 and Rp.window(tie)["break_bin"] == 1
    # RSS1 == 0 with RSS0 > 0 is the two-level model
    w0 = W(np.r_[np.full(4, -1.0), np.zeros(26)])
    assert w0["label"] == "WINDOW_IDENTIFIED" and w0["rss1"] == 0 and w0["break_bin"] == 4
    # BIC tie -> simple: a near-constant series whose two-level gain is below the penalty
    weak = np.r_[np.full(15, -1.0), np.full(15, -1.0 + 1e-3)] + 0.5 * np.cos(np.arange(30) * 2.1)
    assert W(weak)["label"] == "WINDOW_NOT_IDENTIFIED"
    # calibration opens calibration shards only: build a production-shaped directory from check shards
    tmp = work / "window_leak"
    shutil.rmtree(tmp, ignore_errors=True)
    src_dir = work / "obs_100-109"
    for t in range(1, D.NT + 1):
        with np.load(src_dir / "check" / f"t{t:02d}.npz", allow_pickle=False) as d:
            a = {k: d[k] for k in d.files}
        for group, slots in D.GROUPS_PRODUCTION.items():
            g = {k: (v[slots] if v.ndim and v.shape[0] == 10 else v) for k, v in a.items()}
            g["seeds"] = np.asarray([D.PRODUCTION[s] for s in slots])
            D.save_npz(tmp / group / f"t{t:02d}.npz", g)
    w_real = Rp.calibration_core(tmp)
    for t in range(1, D.NT + 1):
        (tmp / "primary" / f"t{t:02d}.npz").write_bytes(b"corrupt primary shard")
    w_again = Rp.calibration_core(tmp)
    assert json.dumps(D.clean(w_real), sort_keys=True) == json.dumps(D.clean(w_again), sort_keys=True)
    leak = mutated(src, "calibration_core", 'g = load_group(out, "calibration", seeds)',
                   'g = load_group(out, "primary", D.PRODUCTION[:5])', vars(Rp))
    fixed = mutated(src, "window", "rss1, b, e, l = min(cands)", "rss1, b, e, l = cands[2]", vars(Rp))

    def fixed_ok():
        return all(fixed(v)["label"] == lab and fixed(v)["break_bin"] == b for v, lab, b in cases.values())

    mut = {
        "primary_leak": rejected(lambda: leak(tmp)),
        "fixed_b3": rejected(fixed_ok),
        "tie_larger_b": rejected(lambda: mutated(src, "window", "rss1, b, e, l = min(cands)",
                                                 "rss1, b, e, l = min(cands, key=lambda c: (c[0], -c[1]))",
                                                 vars(Rp))(tie)["candidate_break_bin"] == 1),
        "accept_reversed": rejected(lambda: mutated(src, "window", "ok = bic1 < bic0 and e < 0 and e < l",
                                                    "ok = bic1 < bic0", vars(Rp))(-v2)["label"] == "WINDOW_NOT_IDENTIFIED"),
    }
    return dict(pass_=True, cases=out, hard_easy_separate=True, calibration_files_only=True,
                synthetic_calibration=dict(hard=w_real["hard"]["label"], easy=w_real["easy"]["label"])), mut


def s_verdict() -> tuple[dict, dict]:
    src = Path(Rp.__file__).read_text()
    n = 0
    for vals in itertools.product([-1.0, 0.0, 1.0], repeat=5):
        r = Rp.directional(vals, "NEG", "POS")
        want = "NEG" if vals == (-1.0,) * 5 else "POS" if vals == (1.0,) * 5 else "UNRESOLVED"
        assert r["label"] == want, (vals, r["label"])
        n += 1
    assert Rp.directional([0.0] * 5, "N", "P")["label"] == "UNRESOLVED"
    assert Rp.directional([-1, -1, -1, -1, 0], "N", "P")["label"] == "UNRESOLVED"
    assert Rp.directional([float("nan")] + [-1.0] * 4, "N", "P")["label"] == "DIVERGED"
    assert Rp.directional([-1.0] * 4, "N", "P")["label"] == "INCOMPLETE"
    assert Rp.directional([-1.0] * 500, "N", "P")["label"] == "INCOMPLETE"
    assert Rp.directional([-1.0] * 5, "N", "P")["one_sided_p_negative"] == 1 / 32
    assert Rp.directional([-1.0] * 4 + [1.0], "N", "P")["one_sided_p_negative"] == 6 / 32
    # D1 is NOT_JUDGED without both windows
    g = dict(acc=np.zeros((5, 30, 30, len(D.KEYS))), unit_task=np.zeros((5, 30, 100, 7)),
             J2=np.zeros((5, 30, 100)), J1=np.zeros((5, 30, 100)))
    g["acc"][..., Rp.K["S2_sum"]] = -1.0
    w = dict(hard=dict(break_bin=None), easy=dict(break_bin=2))
    res = Rp.analyse(g, w)["verdicts"]
    assert res["D1"]["label"] == "NOT_JUDGED" and res["D1_hard"]["label"] == "NOT_JUDGED"
    assert res["D1_easy"]["label"] == "UNRESOLVED"        # boundary rate == later rate -> all-zero difference
    assert res["D4"]["label"] == "SELF_NET_DOWN" and res["D2"]["label"] == "UNRESOLVED"
    four = mutated(src, "directional", 'label = negative if (a < 0).all() else positive if (a > 0).all() else "UNRESOLVED"',
                   'label = negative if (a < 0).sum() >= 4 else positive if (a > 0).sum() >= 4 else "UNRESOLVED"', vars(Rp))
    anylen = mutated(src, "directional", "if a.shape != (5,):", "if a.ndim != 1:", vars(Rp))
    mut = {"four_of_five": rejected(lambda: four([-1, -1, -1, -1, 1], "N", "P")["label"] == "UNRESOLVED"),
           "unit_replicates": rejected(lambda: anylen([-1.0] * 500, "N", "P")["label"] == "INCOMPLETE"),
           "missing_seed": rejected(lambda: anylen([-1.0] * 4, "N", "P")["label"] == "INCOMPLETE")}
    return dict(pass_=True, sign_patterns=n), mut


# --------------------------------------------------------------------------
# GPU runs
# --------------------------------------------------------------------------

def host_run(seeds, n_tasks, out, cifar, device, fresh):
    """The UNMODIFIED host; task-end P captured through a read-only wrapper of its own evaluate()."""
    hashes, orig = [], C.evaluate

    def wrapped(P, X, Y, act, live):
        hashes.append(phash(P))
        return orig(P, X, Y, act, live)

    C.evaluate = wrapped
    try:
        C.run("R", seeds, "std", n_tasks, device, Path(out), lr=D.LR, cifar=cifar, fresh=fresh, graph=True,
              progress=lambda m: None)
    finally:
        C.evaluate = orig
    return hashes


def batch_stream_sha(seeds, n_tasks, cifar):
    """The batch index stream regenerated independently of any run."""
    rows = C.class_rows(cifar.train_y)
    plans = [C.task_plan(s) for s in seeds]
    g = {s: H.stream("c51_batch", s) for s in seeds}
    out = []
    for t in range(1, n_tasks + 1):
        rt = [torch.cat([rows[q] for q in plans[r][t - 1][1]]) for r in range(len(seeds))]
        b = torch.stack([C.batch_indices(g[s], rt[r], D.STEPS) for r, s in enumerate(seeds)])
        out.append(hashlib.sha256(b.numpy().tobytes()).hexdigest())
    return out


def engine(seeds, n_tasks, out, cifar, device, observe=True, graph=True, fresh=False, mutate=frozenset(),
           observer=None, full=False):
    hashes = []
    hook = (lambda t, P, m, v: hashes.append((phash(P), phash(m), phash(v)))) if full else \
        (lambda t, P, m, v: hashes.append(phash(P)))
    shards = {}
    obs = observer if observer is not None else (D.Observer(len(seeds), device, fixtures=()) if observe else None)
    info = D.run(seeds, n_tasks, device, Path(out), cifar, observer=obs, graph=graph, fresh=fresh,
                 progress=lambda m: None, mutate=mutate, state_hook=hook,
                 on_task_end=lambda t, s: shards.__setitem__(t, s))
    return hashes, shards, info


def same_shards(a, b) -> bool:
    if a.keys() != b.keys():
        return False
    for t in a:
        for k in a[t]:
            x, y = np.asarray(a[t][k]), np.asarray(b[t][k])
            if x.shape != y.shape or not np.array_equal(x, y, equal_nan=True):
                return False
    return True


def s_host(work, cifar, device) -> tuple[dict, dict, dict]:
    # (a) environment qualification: the unmodified host, seeds 0-9, against the committed record
    env = work / "host_env_0-9"
    shutil.rmtree(env, ignore_errors=True)
    host_run(D.PRODUCTION, D.NT, env, cifar, device, True)
    rec = [r for r in rows_of(D.RECORD / "per_task.csv") if int(r["task"]) <= D.NT]
    new = rows_of(env / "per_task.csv")
    cols = list(rec[0].keys())
    diff = {}
    for x, y in zip(rec, new):
        for k in cols:
            if x[k] != y[k]:
                diff.setdefault(k, []).append(abs(float(x[k]) - float(y[k])))
    non_eff = sorted(k for k in diff if not k.startswith("eff_rank"))
    fresh_same = (env / "fresh_control.csv").read_bytes() == (D.RECORD / "fresh_control.csv").read_bytes()
    assert len(rec) == len(new) == 10 * D.NT and list(new[0].keys()) == cols and not non_eff and fresh_same, non_eff
    host_env = dict(per_task_sha256=D.sha(env / "per_task.csv"), non_eff_rank_identical=True,
                    fresh_control_identical=True,
                    eff_rank_differences={k: dict(rows=len(v), max_abs=max(v)) for k, v in diff.items()})
    # (b) observed engine vs the unmodified host, check seeds, 30 tasks + fresh
    hdir = work / "host_100-109"
    shutil.rmtree(hdir, ignore_errors=True)
    h_hash = host_run(D.CHECK_SEEDS, D.NT, hdir, cifar, device, True)
    odir = work / "obs_100-109"
    shutil.rmtree(odir, ignore_errors=True)
    o_hash = []
    res = D.observed_run(odir, "check", D.CHECK_SEEDS, D.NT, D.GROUPS_CHECK, cifar, device,
                         state_hook=lambda t, P, m, v: o_hash.append(phash(P)))
    same_rows = (odir / "per_task.csv").read_bytes() == (hdir / "per_task.csv").read_bytes()
    same_fresh = (odir / "fresh_control.csv").read_bytes() == (hdir / "fresh_control.csv").read_bytes()
    same_P = o_hash == h_hash and len(h_hash) == D.NT
    same_batches = res["status"]["batch_sha"] == batch_stream_sha(D.CHECK_SEEDS, D.NT, cifar)
    rng = res["status"]["rng_unchanged"] and res["status"]["rng_probes"] == 2 * D.NT
    assert same_rows and same_fresh and same_P and same_batches and rng and res["status"]["rolled_back"], \
        (same_rows, same_fresh, same_P, same_batches, rng)
    # mutants: 3 tasks against the host's first 3 task-end P
    mut = {}
    for name in MUTANTS["S-host"]:
        hs, _, _ = engine(D.CHECK_SEEDS, 3, work / f"mut_host_{name}", cifar, device, mutate=frozenset({name}))
        mut[name] = hs != h_hash[:3]
    # REPORT_ONLY: does R5 change the rows of slots 0-4?
    hs5, _, _ = engine(D.CHECK_SEEDS[:5], 3, work / "r5", cifar, device, observe=False)
    r5_rows = rows_of(work / "r5" / "per_task.csv")
    h_rows = [r for r in rows_of(hdir / "per_task.csv") if int(r["task"]) <= 3 and int(r["slot"]) < 5]
    r5 = dict(rows_identical_to_R10_slots_0_4=[{k: v for k, v in r.items()} for r in r5_rows] == h_rows)
    return (dict(pass_=True, env=host_env, per_task_identical=same_rows, fresh_identical=same_fresh,
                 task_end_P_identical=same_P, batches_identical=same_batches, rng_unchanged=rng,
                 rolled_back=res["status"]["rolled_back"], R5_report_only=r5), mut, dict(host_env=host_env, run=res))


def s_self_total(work, cifar, device, obs_res) -> tuple[dict, dict]:
    odir = work / "obs_100-109"
    st = obs_res["status"]
    assert st["flags_total"]["bad_closure"] == 0 and st["flags_total"]["bad_native"] == 0
    assert all(c["telescoping_violations"] == 0 and c["J_violations"] == 0 for c in st["counts"])
    aud = Rp.audit(odir, device, cifar)
    assert aud["status"] == "PASS", aud
    src = Path(D.__file__).read_text()
    fx = {(t, j): torch.load(odir / "fixtures" / f"t{t:02d}_j{j:03d}.pt", weights_only=False) for t, j in D.FIXTURES}

    def flags(m2, f, mu_prev=None):
        before, after = D.aug(f["p_old"][2], f["p_old"][3]), D.aug(f["p_new"][2], f["p_new"][3])
        r = m2(before, after, f["f0"], f["f1"])
        return float((r["closure"] / r["alg"]).max()), float((r["native"] / r["nat"]).max())

    real = {k: flags(D.measure2, f) for k, f in fx.items()}
    assert all(c <= 1 and n <= 1 for c, n in real.values()), real
    omit = mutated(src, "measure2", "    D = S + U\n", "    D = S\n", vars(D))
    oldw = mutated(src, "measure2", "    U = (w_new * dmu).sum(-1)", "    U = (before[..., :-1] * dmu).sum(-1)", vars(D))
    mut = {"omit_U": all(flags(omit, f)[0] > 1 for f in fx.values()),
           "old_W_in_U": all(flags(oldw, f)[0] > 1 for f in fx.values())}
    jres = []
    for k in ((2, 1), (3, 1)):
        f = fx[k]
        jm = mutated(src, "measure2", "    dmu = (mu1 - mu0)[:, None, :]",
                     "    dmu = (mu1 - MU_PREV)[:, None, :]", dict(vars(D), MU_PREV=f["mu_last_prev"]))
        jres.append(flags(jm, f)[0] > 1)
    mut["J_in_U"] = all(jres)
    # the wrong image set (the previous task's) feeds both mu and the native means consistently, so only the
    # independent recomputation from the plan can catch it
    wrong = []
    X_all = cifar.inputs("train", D.COND, device)
    rows = C.class_rows(cifar.train_y)
    plans = [C.task_plan(s) for s in D.CHECK_SEEDS]
    for (t, j), f in fx.items():
        Xp = torch.stack([X_all[torch.cat([rows[q] for q in plans[r][t - 2][1]])] for r in range(10)])
        P0 = [q.to(device) for q in f["p_old"]] + [torch.zeros(10, 100, device=device)]
        P1 = [q.to(device) for q in f["p_new"]] + [torch.zeros(10, 100, device=device)]
        g0 = {k: v.cpu() for k, v in D.features(P0, Xp).items()}
        g1 = {k: v.cpu() for k, v in D.features(P1, Xp).items()}
        r = D.measure2(D.aug(f["p_old"][2], f["p_old"][3]), D.aug(f["p_new"][2], f["p_new"][3]), g0, g1)
        bad = copy.copy(f)
        bad["values"] = dict(f["values"], S2=r["S"], U2=r["U"], D2=r["D"])
        a = Rp.audit_fixture(bad, D.CHECK_SEEDS, cifar, device, X_all)
        wrong.append(not a["pass_"])
    mut["wrong_task_mu"] = all(wrong)
    return dict(pass_=True, audit=aud, fixture_ratios=real), mut


def s_bin(work, obs_res) -> tuple[dict, dict]:
    odir = work / "obs_100-109"
    g = Rp.load_group(odir, "check", D.CHECK_SEEDS)                 # asserts 26 updates in every bin
    acc, ut = g["acc"], g["unit_task"]
    assert acc.shape[:3] == (10, D.NT, D.NBIN)
    worst = 0.0
    for k in ("S2", "U2", "D2"):
        bins = acc[..., Rp.K[f"{k}_sum"]].sum(-1)                   # (10, 30)
        units = ut[..., D.TRANSPORT.index(k)].sum(-1)
        mag = (acc[..., Rp.K[f"{k}_pos"]] - acc[..., Rp.K[f"{k}_neg"]]).sum(-1)
        bound = D.gamma(D.STEPS * D.HID + D.NBIN + 2) * mag + 4 * D.TINY64
        worst = max(worst, float(np.max(np.abs(bins - units) / bound)))
        ub = g["unit_bin"][..., D.UNIT_KEYS.index(k)].sum(2).sum(-1)
        worst = max(worst, float(np.max(np.abs(ub - units) / bound)))
    assert worst <= 1, worst
    upd = np.array([D.Observer.bin_of(None, j) for j in range(1, D.STEPS + 1)])
    assert (np.bincount(upd, minlength=D.NBIN) == D.BIN).all() and upd.max() == D.NBIN - 1

    def accepts(binf):
        b = np.array([binf(j) for j in range(1, D.STEPS + 1)])
        return b.max() < D.NBIN and b.min() >= 0 and (np.bincount(b, minlength=D.NBIN) == D.BIN).all()

    mut = {"bin27": not accepts(lambda j: (j - 1) // 27), "off_by_one": not accepts(lambda j: j // D.BIN)}
    return dict(pass_=True, bins=D.NBIN, size=D.BIN, worst_sum_ratio=worst), mut


def conf_independent(f) -> dict:
    """Spec §9 S-conf (a): logit derivatives of L_conf and L_label by float64 autograd, backprop by einsum."""
    a1, z2, z3f, y = (x.double() if x.dtype != torch.long else x for x in f["batch"])
    R, Bn, Cn = z3f.shape
    W3 = f["p_old"][4].double()
    z3 = z3f.clone().requires_grad_(True)
    lconf = (torch.logsumexp(z3, -1) - z3.mean(-1)).mean(1).sum()
    llab = -(((F.one_hot(y, Cn).double() - 1.0 / Cn) * z3).sum(-1)).mean(1).sum()
    lce = F.cross_entropy(z3.reshape(-1, Cn), y.reshape(-1), reduction="none").view(R, Bn).mean(1).sum()
    dc, = torch.autograd.grad(lconf, z3)
    dl, = torch.autograd.grad(llab, z3)
    de, = torch.autograd.grad(lce, z3)
    split = float(((dc + dl - de).abs() / (D.gamma(Cn + 8) * (dc.abs() + dl.abs() + de.abs()) + D.TINY64)).max())
    gate = (z2 >= 0).double().numpy()
    ht = np.concatenate([a1.numpy(), np.ones((R, Bn, 1))], -1)
    W3n = W3.numpy()

    def back(d):
        d2 = np.einsum("rbc,rci->rbi", d.numpy(), W3n) * gate
        mag = np.einsum("rbi,rbj->rij", np.einsum("rbc,rci->rbi", np.abs(d.numpy()), np.abs(W3n)) * gate, np.abs(ht))
        return np.einsum("rbi,rbj->rij", d2, ht), mag

    gc, mc = back(dc)
    gl, ml = back(dl)
    bound = lambda m: D.gamma(2 * Cn + Bn + 8) * m + D.TINY64  # noqa: E731
    rc = float(np.max(np.abs(gc - f["gconf"].numpy()) / bound(mc)))
    rl = float(np.max(np.abs(gl - f["glab"].numpy()) / bound(ml)))
    native = float(((f["grads2"] - f["gconf"] - f["glab"]).abs() / f["gb"]).max())
    return dict(split=split, conf=rc, label=rl, native=native)


def conf_replay(f, cs=D.conf_step, dec=D.decompose) -> dict:
    gconf, glab, gb = dec(*f["batch"][:3], f["batch"][3], f["p_old"][4])
    before, after = D.aug(f["p_old"][2], f["p_old"][3]), D.aug(f["p_new"][2], f["p_new"][3])
    k0 = D.kvec(f["f0"]["mu"])[:, None, :]
    S = ((after - before) * k0).sum(-1)
    c = cs(f["state_prev"], gconf, glab, gb, f["grads2"], f["m_old2"], f["m_new2"], f["v_new2"],
           f["inv_c1"].double(), f["inv_c2"].double(), before, after, k0, S)
    return dict(label=float(c["label_ratio"].max()), comp=float(c["comp_ratio"].max()), p=float(c["p_ratio"].max()),
                grad=float(((f["grads2"] - gconf - glab).abs() / gb).max()),
                state_close=float(max((c[k] - f["state_new"][k]).abs().max() /
                                      (f["state_new"][k].abs().max() + D.TINY64) for k in ("Mc", "Mh", "Ml"))))


def s_conf(work, obs_res) -> tuple[dict, dict]:
    odir = work / "obs_100-109"
    st = obs_res["status"]["flags_total"]
    for k in ("bad_grad", "bad_adam", "bad_p", "bad_comp", "bad_label", "nonfinite"):
        assert st[k] == 0, (k, st[k])
    fx = {(t, j): torch.load(odir / "fixtures" / f"t{t:02d}_j{j:03d}.pt", weights_only=False) for t, j in D.FIXTURES}
    ind = {f"t{t}_j{j}": conf_independent(f) for (t, j), f in fx.items()}
    assert all(v["split"] <= 1 and v["conf"] <= 1 and v["label"] <= 1 and v["native"] <= 1 for v in ind.values()), ind
    rep = {f"t{t}_j{j}": conf_replay(f) for (t, j), f in fx.items()}
    assert all(v["label"] <= 1 and v["comp"] <= 1 and v["p"] <= 1 and v["grad"] <= 1 for v in rep.values()), rep
    src = Path(D.__file__).read_text()
    swap = mutated(src, "decompose", "    return gconf, glab, gb\n", "    return glab, gconf, gb\n", vars(D))
    pi10 = mutated(src, "decompose", "    dconf = (p - 1.0 / Cn) / Bn\n    dlab = (1.0 / Cn - onehot) / Bn",
                   "    dconf = (p - 1.0 / 10) / Bn\n    dlab = (1.0 / 10 - onehot) / Bn", vars(D))
    dbl = mutated(src, "decompose", "    dconf = (p - 1.0 / Cn) / Bn\n    dlab = (1.0 / Cn - onehot) / Bn",
                  "    dconf = (p - 1.0 / Cn) / Bn / Bn\n    dlab = (1.0 / Cn - onehot) / Bn / Bn", vars(D))
    nohist = mutated(src, "conf_step", '    Mh = B1F * state["Mh"]', '    Mh = 0.0 * state["Mh"]', vars(D))
    reset = mutated(src, "conf_step", '    Mc = B1F * state["Mc"] + A1F * gconf', "    Mc = A1F * gconf", vars(D))

    def via_dec(dec):
        """A decomposition mutant must fail the independent definition or the native-gradient check."""
        out = []
        for f in fx.values():
            gconf, glab, gb = dec(*f["batch"][:3], f["batch"][3], f["p_old"][4])
            g = dict(f, gconf=gconf, glab=glab, gb=gb)
            v = conf_independent(g)
            out.append(v["conf"] > 1 or v["label"] > 1 or v["native"] > 1)
        return all(out)

    late = [f for (t, j), f in fx.items() if t >= 2 and j > 1]
    mut = {"swap_conf_label": via_dec(swap), "pi_1_10": via_dec(pi10), "double_batch_mean": via_dec(dbl),
           # the inherited moment decays as b1^s: it is visible at task starts (j = 1), not after 780 updates
           "no_history": all(conf_replay(f, cs=nohist)["label"] > 1 for f in fx.values() if f["j"] == 1),
           "conf_reset": all(conf_replay(f, cs=reset)["label"] > 1 for f in late)}
    return dict(pass_=True, independent=ind, replay=rep), mut


def s_graph(work, cifar, device) -> tuple[dict, dict]:
    he, se, ie = engine(D.CHECK_SEEDS, 3, work / "graph_eager", cifar, device, graph=False, full=True)
    hg, sg, ig = engine(D.CHECK_SEEDS, 3, work / "graph_graph", cifar, device, graph=True, full=True)
    same = he == hg and same_shards(se, sg) and ie["rolled_back"] is None and ig["rolled_back"] is True
    rows_same = (work / "graph_eager" / "per_task.csv").read_bytes() == (work / "graph_graph" / "per_task.csv").read_bytes()
    assert same and rows_same
    mut = {}
    for name in MUTANTS["S-graph"]:
        hm, sm, _ = engine(D.CHECK_SEEDS, 3, work / f"graph_mut_{name}", cifar, device, graph=True, full=True,
                           mutate=frozenset({name}))
        mut[name] = hm != he
    return dict(pass_=True, tasks=3, state_and_observation_identical=True, warmup_rolled_back=True), mut


def s_manifest(work) -> tuple[dict, dict]:
    odir = work / "obs_100-109"
    Rp.validate(odir, production=False)
    cpath = odir / "complete.json"
    ipath = odir / "input_manifest.json"
    original_c, original_i = cpath.read_text(), ipath.read_text()
    mut = {}

    def attempt(name, fn):
        try:
            fn()
            mut[name] = rejected(lambda: Rp.validate(odir, production=False))
        finally:
            cpath.write_text(original_c)
            ipath.write_text(original_i)

    def partial():
        m = json.loads(original_c)
        first = next(f["path"] for f in m["files"] if f["path"].endswith(".npz"))
        m["files"] = [f for f in m["files"] if f["path"] != first]
        cpath.write_text(json.dumps(m))

    def foreign():
        for p, txt in ((cpath, original_c), (ipath, original_i)):
            m = json.loads(txt)
            (m["identity"] if p == cpath else m)["run_id"] = "foreign_run"
            p.write_text(json.dumps(m))

    def forged():
        m = json.loads(original_c)
        m["files"][0]["sha256"] = "0" * 64
        cpath.write_text(json.dumps(m))

    def stale():
        for p, txt in ((cpath, original_c), (ipath, original_i)):
            m = json.loads(txt)
            (m["identity"] if p == cpath else m)["source_sha256"]["src/drive_cifar5p1_1007.py"] = "0" * 64
            p.write_text(json.dumps(m))

    for name, fn in (("partial", partial), ("foreign_run", foreign), ("forged_marker", forged), ("stale_source", stale)):
        attempt(name, fn)
    Rp.validate(odir, production=False)
    return dict(pass_=True, validated=str(odir)), mut


def s_cost(work, cifar, device, obs_res) -> tuple[dict, dict]:
    cost = obs_res["cost"]
    assert cost["observed_updates"] == D.NT * D.STEPS
    assert all(cost["image_counts"][str(t)] == (2500 if t % 2 else 500) for t in range(1, D.NT + 1))
    real = json.loads((work / "obs_100-109" / "complete.json").read_text())["status"]
    Rp.validate_counts(real, D.NT)                                  # the count check passes on the real run
    mut = {}
    for name, obs in (("batch_only", D.Observer(10, device, fixtures=(), image_limit=D.BATCH)),
                      ("every_other_update", D.Observer(10, device, fixtures=(), every=2))):
        out = work / f"cost_mut_{name}"
        shutil.rmtree(out, ignore_errors=True)
        D.observed_run(out, "check", D.CHECK_SEEDS, 2, D.GROUPS_CHECK, cifar, device, fresh=False, observer=obs)
        st = json.loads((out / "complete.json").read_text())["status"]
        mut[name] = rejected(lambda: Rp.validate_counts(st, 2))
    need = math.ceil(2 * (cost["max_cuda_reserved_bytes"] / 2 ** 30 + 1.0)) / 2     # + CUDA context, 0.5 GB steps
    return dict(pass_=True, need_gb=need, **cost), mut


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(D.RAW / "checks"))
    a = ap.parse_args()
    work = Path(a.out)
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = D.setup()
    cifar = D.load_cifar()
    checks, mutants = {}, {}

    def record(name, res):
        details = dict(res[0])
        checks[name] = {"pass": bool(details.pop("pass_")), **details}
        mutants[name] = {k: bool(v) for k, v in res[1].items()}
        bad = [m for m in MUTANTS[name] if res[1].get(m) is not True]
        print(f"{name}: PASS, mutants rejected {sum(res[1].values())}/{len(MUTANTS[name])}"
              + (f"  NOT REJECTED: {bad}" if bad else ""), flush=True)

    record("S-verdict", s_verdict())
    with D.gpu_lock(4.0):
        h = s_host(work, cifar, device)
        record("S-host", h[:2])
        obs_res = h[2]["run"]
        record("S-self-total", s_self_total(work, cifar, device, obs_res))
        record("S-bin", s_bin(work, obs_res))
        record("S-conf", s_conf(work, obs_res))
        record("S-window", s_window(work))
        record("S-graph", s_graph(work, cifar, device))
        record("S-manifest", s_manifest(work))
        cost = s_cost(work, cifar, device, obs_res)
        record("S-cost", cost[:2])
    report = dict(status="PASS", source_sha256=D.sources(), spec_commit=D.SPEC_COMMIT,
                  checks=checks, mutants={k: list(v) for k, v in MUTANTS.items()}, mutants_rejected=mutants,
                  host_env=h[2]["host_env"], cost=dict(need_gb=cost[0]["need_gb"], **obs_res["cost"]),
                  scientific_seeds_observed=False, seconds=time.time() - t0,
                  utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    try:
        validate_checks(report)
    except AssertionError as e:
        report["status"] = "FAIL"
        D.put(work / "checks_failed.json", dict(report, error=str(e)))
        raise
    D.put(CHECKS_JSON, report)
    print("all checks PASS ->", CHECKS_JSON, f"({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

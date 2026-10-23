#!/usr/bin/env python3
"""Checks for lc_elu_lr_0917 (specs/spec_lc_elu_lr_0917.md section 5).

    OMP_NUM_THREADS=1 python3 analysis/lc_elu_lr_0917/checks.py              # writes results/lc_elu_lr_0917/checks.json
    ... --only S1,S4                                                          # development: results/_checks_lc_elu_lr_0917/checks_partial.json

Every check runs on the real file and then on each of its mutations; a mutation is an exact-once string
substitution in the check's target file and must make the same check fail.  S1 needs the Lillo & Cheney
files (iclr2026 bdce354) in --lc-src; they are not in this repository (no licence) and their sha256 go
into checks.json.  Tolerances are derived in the docstrings.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402
import torch                # noqa: E402

RUNNER = REPO / "src" / "lc_elu_lr_0917.py"
VERDICT = REPO / "analysis" / "lc_elu_lr_0917" / "verdict.py"
SPEC = REPO / "specs" / "spec_lc_elu_lr_0917.md"
OUT = REPO / "results" / "lc_elu_lr_0917"
SCR = REPO / "results" / "_checks_lc_elu_lr_0917"
LC_SRC = Path.home() / "Projects" / "obsidian-research-data" / "lc_elu_lr_0917" / "lillo_cheney_src"
LC_FILES = ("src_case_cl_benchmarks_bench_models.py", "utils_custom_activations.py",
            "src_case_cl_benchmarks_bench_main.py", "src_case_cl_benchmarks_bench_data_handlers.py")
IMAGES = REPO / "data" / "mnist" / "train-images-idx3-ubyte.gz"
EPS32 = float(np.finfo(np.float32).eps)
torch.set_num_threads(1)

_count = [0]


def load_module(path: Path, subst: tuple[str, str] | None = None) -> types.ModuleType:
    src = path.read_text()
    if subst is not None:
        old, new = subst
        n = src.count(old)
        if n != 1:
            raise AssertionError(f"mutation anchor occurs {n} times in {path.name}: {old!r}")
        src = src.replace(old, new)
    _count[0] += 1
    name = f"_lc_check_mod_{_count[0]}"
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod


def fresh(d: Path) -> Path:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def read_rows(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, float_precision="round_trip")


# --------------------------------------------------------------------------
# S1  the box against Lillo & Cheney's own model class and loop
# --------------------------------------------------------------------------

def lillo_model_module(lc_src: Path) -> types.ModuleType:
    rat = types.ModuleType("rational")
    rt = types.ModuleType("rational.torch")

    class Rational:                      # only imported by bench_models; never built for ELU
        pass
    rt.Rational = Rational
    rat.torch = rt
    sys.modules["rational"] = rat
    sys.modules["rational.torch"] = rt
    utils = types.ModuleType("utils")
    utils.__path__ = []
    sys.modules["utils"] = utils
    ca = load_module(lc_src / "utils_custom_activations.py")
    sys.modules["utils.custom_activations"] = ca
    utils.custom_activations = ca
    return load_module(lc_src / "src_case_cl_benchmarks_bench_models.py")


def lillo_reference(bm, seed: int, act_id: str, alpha: float, lr: float, tasks: int, epochs: int) -> dict:
    """bench_main.run_experiments -> run_single_experiment_fast for random_label_mnist, transcribed:
    seeds; handler (_prepare_data: image pool, then 50 x (train 1200, val 500) label draws); model; Adam;
    the manual batching loop (bench_main.py L184-203) with the pre-update batch accuracy."""
    import gzip
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with gzip.open(IMAGES, "rb") as fh:
        raw = fh.read()
    data = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(60000, 28, 28).copy())
    x_pool = (data.float() / 255.0).view(-1, 784)
    x_train = x_pool[:1200]
    train_label_maps, val_label_maps = [], []
    for _ in range(50):
        train_label_maps.append(torch.randint(0, 10, (1200,)))
        val_label_maps.append(torch.randint(0, 10, (500,)))
    args = types.SimpleNamespace(
        activation_fn=act_id, elu_alpha=alpha, leaky_neg_slope=0.01, swish_gelu_beta=1.0, selu_alpha=1.673,
        smooth_c=5.0, smooth_p=3.0, smooth_alpha=0.1, rand_smooth_lower=0.3, rand_smooth_upper=1.0,
        init_boprelu_alpha=0.25, boprelu_min=0.01, boprelu_max=1.0, init_prelu_alpha=0.25, rrelu_eval_slope=0.0,
        rsselu_lower_b=1.168, rsselu_upper_b=2.178, deg_numerator=5, deg_denominator=4, rational_version="A",
        rational_approx_func="leaky_relu", prelu_scope="global", boprelu_scope="global")
    model = bm.MLPBenchmarks(args, input_dim=784, output_dim=10)
    init = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    online = []
    for task_id in range(tasks):
        x_train_permuted, y_train_task = x_train, train_label_maps[task_id]
        batch_accs = []
        for epoch in range(epochs):
            model.train()
            for i in range(0, 1200, 16):
                x_batch = x_train_permuted[i: i + 16]
                y_batch = y_train_task[i: i + 16]
                logits, _ = model(x_batch)
                with torch.no_grad():
                    acc = (logits.argmax(1) == y_batch).float().mean().item()
                loss = criterion(logits, y_batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                batch_accs.append(acc)
        online.append(float(np.nanmean(batch_accs)))
    return {"init": init, "labels": train_label_maps, "online": online,
            "final": [p.detach().clone() for p in model.parameters()]}


S1_CASES = (("E36_lr1e4", "elu", 3.6, 1e-4), ("E1_lr1e3", "elu", 1.0, 1e-3))
S1_TASKS, S1_EPOCHS = 3, 3


def s1_compare(R, refs: dict, tag: str) -> dict:
    out = {}
    for arm, act_id, alpha, lr in S1_CASES:
        ref = refs[arm]
        d = fresh(SCR / f"S1_{tag}_{arm}")
        R.run(arm, 0, S1_TASKS, d, epochs=S1_EPOCHS, threads=1)
        rows = read_rows(d / "per_task.csv")
        ck = torch.load(d / "ckpts" / "last.pt", weights_only=False)
        model, _, labels = R.build(arm, 0, S1_TASKS)
        init_ok = all(torch.equal(a, b) for a, b in zip(model.parameters(), ref["init"]))
        lab_ok = len(labels) == 50 and all(torch.equal(labels[t], ref["labels"][t]) for t in range(50))
        fin = list(ck["model"].values())
        fin_ok = len(fin) == len(ref["final"]) and all(torch.equal(a, b) for a, b in zip(fin, ref["final"]))
        onl_ok = [float(v) for v in rows.online] == ref["online"]
        out[arm] = {"init": init_ok, "labels": lab_ok, "final_params": fin_ok, "online": onl_ok,
                    "online_runner": [float(v) for v in rows.online], "online_ref": ref["online"]}
    out["pass"] = all(v["init"] and v["labels"] and v["final_params"] and v["online"]
                      for k, v in out.items() if k != "pass")
    return out


S1_MUT = [
    ("reversed batch order", ("for i in range(0, x.shape[0], BATCH):", "for i in reversed(range(0, x.shape[0], BATCH)):")),
    ("alpha ignored", ("return nn.ELU(alpha=param)", "return nn.ELU()")),
    ("Adam eps", ("opt = torch.optim.Adam(model.parameters(), lr=lr)",
                  "opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)")),
    ("validation labels drawn first", (
        "train_labels = torch.randint(0, N_CLASSES, (N_IMG,))\n        torch.randint(0, N_CLASSES, (N_VAL,))",
        "torch.randint(0, N_CLASSES, (N_VAL,))\n        train_labels = torch.randint(0, N_CLASSES, (N_IMG,))")),
    ("image order", (".reshape(n, 784)\n", ".reshape(n, 784)[::-1].copy()\n")),
    ("labels for 51 tasks before the model", ("for _ in range(MAIN_TASKS):", "for _ in range(MAIN_TASKS + 1):")),
    ("online denominator", ('"online": hits / (steps * BATCH)', '"online": hits / (steps * BATCH + 1)')),
]


def check_S1(lc_src: Path) -> dict:
    bm = lillo_model_module(lc_src)
    refs = {arm: lillo_reference(bm, 0, act_id, alpha, lr, S1_TASKS, S1_EPOCHS)
            for arm, act_id, alpha, lr in S1_CASES}
    res = {"real": s1_compare(load_module(RUNNER), refs, "real"), "mutations": {}}
    for i, (name, sub) in enumerate(S1_MUT):
        try:
            r = s1_compare(load_module(RUNNER, sub), refs, f"m{i}")
            res["mutations"][name] = {"detected": not r["pass"]}
        except Exception as e:          # noqa: BLE001
            res["mutations"][name] = {"detected": True, "error": repr(e)[:200]}
    res["lc_src_sha256"] = {f: hashlib.sha256((lc_src / f).read_bytes()).hexdigest() for f in LC_FILES}
    res["pass"] = res["real"]["pass"] and all(m["detected"] for m in res["mutations"].values())
    return res


# --------------------------------------------------------------------------
# S2  the training derivative of the box's activations
# --------------------------------------------------------------------------

S2_Z = (-1.0, -16.0, -17.0, -20.0, -50.0, -100.0)


def s2_eval(R) -> dict:
    out = {}
    z = torch.tensor(S2_Z, dtype=torch.float32)
    ok = True
    for alpha in (1.0, 3.6):
        g = R.train_dphi(R.make_act("elu", alpha), z)
        ref = torch.exp(z) * alpha
        rel = ((g.double() - ref.double()).abs() / ref.double()).max().item()
        good = bool((g > 0).all()) and rel <= 2 * EPS32
        out[f"elu_{alpha}"] = {"g": g.tolist(), "max_rel": rel, "ok": good}
        ok = ok and good
    zz = torch.tensor([-1.0, 1.0])
    gr = R.train_dphi(R.make_act("relu", 0.0), zz).tolist()
    gl = R.train_dphi(R.make_act("leaky", 0.8), zz).tolist()
    out["relu"], out["leaky"] = gr, gl
    ok = ok and gr == [0.0, 1.0] and abs(gl[0] - 0.8) <= 2 * EPS32 and gl[1] == 1.0
    out["pass"] = ok
    return out


S2_MUT = [("expm1 ELU", ("return nn.ELU(alpha=param)",
                         'return type("ExpELU", (nn.Module,), {"forward": lambda s, z: torch.where(z > 0, z, param * torch.expm1(z))})()'))]


def check_S2() -> dict:
    res = {"real": s2_eval(load_module(RUNNER)), "mutations": {}}
    for name, sub in S2_MUT:
        res["mutations"][name] = {"detected": not s2_eval(load_module(RUNNER, sub))["pass"]}
    res["pass"] = res["real"]["pass"] and all(m["detected"] for m in res["mutations"].values())
    return res


# --------------------------------------------------------------------------
# S3  the extension leaves tasks 1-50 and the global RNG alone
# --------------------------------------------------------------------------

def s3_eval(R) -> dict:
    m50, _, l50 = R.build("E1_lr1e4", 3, 50)
    st50 = torch.get_rng_state().clone()
    m150, _, l150 = R.build("E1_lr1e4", 3, 150)
    st150 = torch.get_rng_state().clone()
    init = all(torch.equal(a, b) for a, b in zip(m50.parameters(), m150.parameters()))
    labs = len(l50) == 50 and len(l150) == 150 and all(torch.equal(a, b) for a, b in zip(l50, l150[:50]))
    rng = bool(torch.equal(st50, st150))
    fresh_ext = not any(torch.equal(l150[50], l) for l in l150[:50])
    return {"init": init, "labels_1_50": labs, "global_rng_after_build": rng, "ext_new": fresh_ext,
            "pass": init and labs and rng and fresh_ext}


S3_MUT = [("extension labels from the global RNG",
           ("labels.append(torch.randint(0, N_CLASSES, (N_IMG,), generator=g))",
            "labels.append(torch.randint(0, N_CLASSES, (N_IMG,)))"))]


def check_S3() -> dict:
    res = {"real": s3_eval(load_module(RUNNER)), "mutations": {}}
    for name, sub in S3_MUT:
        res["mutations"][name] = {"detected": not s3_eval(load_module(RUNNER, sub))["pass"]}
    res["pass"] = res["real"]["pass"] and all(m["detected"] for m in res["mutations"].values())
    return res


# --------------------------------------------------------------------------
# S4  the readout
# --------------------------------------------------------------------------

def s4_eval(R, tag: str) -> dict:
    """(i) zbar_i = w_i . mu + b_i: z is a float32 sum of n_in products and a bias, so its rounding error is
    at most (n_in + 1) eps32 sum_j |w_ij x_j| + |b_i| (first order); mu and the means are float64.
    (ii) neff (log space, exact) and neffT (float32 derivative g, each term within 2 ulp): s1 within 2 eps,
    s2 within 4 eps, so s1^2 / (n s2) within 8 eps; tolerance 10 eps32 relative, on a net with z > -40.
    (iii) per_task.csv unit means = units.npz means (torch vs numpy summation order: 1e-12 relative).
    (iv) floor = the largest class share of the task's labels."""
    d = fresh(SCR / f"S4_{tag}")
    R.run("E1_lr1e3", 1, 2, d, epochs=5, threads=1)
    rows = read_rows(d / "per_task.csv")
    U = dict(np.load(d / "units.npz"))
    ck = torch.load(d / "ckpts" / "last.pt", weights_only=False)
    model, opt, labels = R.build("E1_lr1e3", 1, 2)
    model.load_state_dict(ck["model"])
    x = R.load_images()
    crit = torch.nn.CrossEntropyLoss()
    row, u = R.readout(model, opt, crit, x, labels[1], "elu", 1.0)
    out = {}
    with torch.no_grad():
        _, (z1, a1, z2, a2) = model(x)
    ok = True
    for li, inp, lin, z in ((1, x, model.fc1, z1), (2, a1, model.fc2, z2)):
        W, b = lin.weight.double(), lin.bias.double()
        terms = (inp.double().abs() @ W.abs().T + b.abs()).mean(0)
        tol = (inp.shape[1] + 1) * EPS32 * terms
        dev = (torch.from_numpy(u[f"zbar_l{li}"]) - (torch.from_numpy(u[f"wmu_l{li}"]) + b)).abs()
        good = bool((dev <= tol).all())
        out[f"identity_l{li}"] = {"max_dev": float(dev.max()), "min_tol": float(tol.min()), "ok": good}
        ok = ok and good
        zmin = float(z.min())
        rel = (np.abs(u[f"neff_l{li}"] - u[f"neffT_l{li}"]) / u[f"neff_l{li}"]).max()
        good = zmin > -40 and rel <= 10 * EPS32
        out[f"neff_l{li}"] = {"zmin": zmin, "max_rel": float(rel), "ok": bool(good)}
        ok = ok and good
    keys = [c for c in rows.columns if c.split("_l")[0] in
            ("zbar", "sigma", "U", "gtr", "zero", "neffT", "loggmean", "neff", "pplus", "wnorm", "b")
            and c.endswith(("_l1", "_l2"))]
    worst = 0.0
    for k in keys:
        for t in range(len(rows)):
            v, w = float(rows[k].iloc[t]), float(np.mean(U[k][t]))
            worst = max(worst, abs(v - w) / max(1.0, abs(w)))
    out["csv_vs_npz"] = {"keys": len(keys), "worst_rel": worst, "ok": worst <= 1e-12}
    fl = [float(torch.bincount(labels[t], minlength=10).max()) / 1200 for t in range(2)]
    out["floor"] = {"ok": [float(v) for v in rows.floor] == fl}
    out["pass"] = ok and out["csv_vs_npz"]["ok"] and out["floor"]["ok"]
    return out


S4_MUT = [
    ("mu2 from a2", ("ins = {1: x, 2: a1}", "ins = {1: x, 2: a2}")),
    ("neff over the unit axis", ("s1 * s1 / (g.shape[0] * s2)", "s1 * s1 / (g.shape[1] * s2)")),
    ("median for mean", ('row[f"{k}_l{li}"] = float(u[k].mean())', 'row[f"{k}_l{li}"] = float(u[k].median())')),
]


def check_S4() -> dict:
    res = {"real": s4_eval(load_module(RUNNER), "real"), "mutations": {}}
    for i, (name, sub) in enumerate(S4_MUT):
        try:
            res["mutations"][name] = {"detected": not s4_eval(load_module(RUNNER, sub), f"m{i}")["pass"]}
        except Exception as e:          # noqa: BLE001
            res["mutations"][name] = {"detected": True, "error": repr(e)[:200]}
    res["pass"] = res["real"]["pass"] and all(m["detected"] for m in res["mutations"].values())
    return res


# --------------------------------------------------------------------------
# S5  resume and determinism
# --------------------------------------------------------------------------

def s5_eval(R, tag: str) -> dict:
    dirs = {k: fresh(SCR / f"S5_{tag}_{k}") for k in ("A", "B", "C")}
    R.run("E36_lr1e3", 2, 3, dirs["A"], epochs=3, threads=1)
    R.run("E36_lr1e3", 2, 3, dirs["B"], epochs=3, threads=1)
    R.run("E36_lr1e3", 2, 3, dirs["C"], epochs=3, threads=1, stop_after=2)
    part = not (dirs["C"] / "provenance.json").exists()
    R.run("E36_lr1e3", 2, 3, dirs["C"], epochs=3, threads=1)
    tabs = {k: read_rows(d / "per_task.csv").drop(columns=["sec"]) for k, d in dirs.items()}
    npz = {k: dict(np.load(d / "units.npz")) for k, d in dirs.items()}
    cks = {k: torch.load(d / "ckpts" / "last.pt", weights_only=False) for k, d in dirs.items()}
    same_tab = tabs["A"].equals(tabs["B"]) and tabs["A"].equals(tabs["C"])
    same_npz = all(set(npz["A"]) == set(npz[k]) and all(np.array_equal(npz["A"][q], npz[k][q]) for q in npz["A"])
                   for k in ("B", "C"))
    same_par = all(all(torch.equal(cks["A"]["model"][q], cks[k]["model"][q]) for q in cks["A"]["model"])
                   for k in ("B", "C"))
    prov = json.loads((dirs["C"] / "provenance.json").read_text())
    segs = len(prov["history"]) == 2 and prov["history"][1]["from_task"] == 3
    return {"stopped_is_partial": part, "tables": same_tab, "units": same_npz, "params": same_par,
            "two_segments": segs, "pass": part and same_tab and same_npz and same_par and segs}


S5_MUT = [("resume without the optimizer state", ('        opt.load_state_dict(ck["opt"])\n', '        ck["opt"]\n'))]


def check_S5() -> dict:
    res = {"real": s5_eval(load_module(RUNNER), "real"), "mutations": {}}
    for i, (name, sub) in enumerate(S5_MUT):
        res["mutations"][name] = {"detected": not s5_eval(load_module(RUNNER, sub), f"m{i}")["pass"]}
    res["pass"] = res["real"]["pass"] and all(m["detected"] for m in res["mutations"].values())
    return res


# --------------------------------------------------------------------------
# S6  the verdict on synthetic runs
# --------------------------------------------------------------------------

CORE = ("E1_lr1e3", "E36_lr1e3", "E1_lr1e4", "E36_lr1e4")
HZ = {"E1_lr1e3": 50, "E36_lr1e3": 50, "E1_lr1e4": 150, "E36_lr1e4": 150}


def synth_run(n: int, level, floor: float = 0.114, mu=None, zb=None) -> pd.DataFrame:
    t = np.arange(1, n + 1)
    lev = np.array([level(k) for k in t], dtype=float)
    mu = np.array([mu(k) for k in t], dtype=float) if mu else 1.0 + 0.1 * t
    zb = np.array([zb(k) for k in t], dtype=float) if zb else -0.05 * t
    return pd.DataFrame({"arm": "x", "task": t, "online": lev, "floor": floor, "mu_norm_l2": mu, "zbar_l2": zb,
                         "gtr_l2": 0.5, "neffT_l2": 0.9, "wnorm_l2": 2.0, "wnorm_l1": 1.0})


def collapse_at(tc: int, early: float = 0.9, under: float = -0.012):
    return lambda k: early if k < tc else 0.114 + under


def const(v: float):
    return lambda k: v


def scenario_frames(name: str) -> dict:
    """(arm, seed) -> per_task frame.  Base = LR_DOMINATES, Q2 exact, Q3 running, Q4 E1 collapses at 100."""
    fr = {}
    for s in range(5):
        fr[("E1_lr1e3", s)] = synth_run(50, collapse_at(10))
        fr[("E36_lr1e3", s)] = synth_run(50, collapse_at(9))
        fr[("E1_lr1e4", s)] = synth_run(150, collapse_at(100, 0.8))
        fr[("E36_lr1e4", s)] = synth_run(150, const(0.8423 + 0.001 * (s - 2)))
    if name == "alpha":
        for s in range(5):
            fr[("E36_lr1e3", s)] = synth_run(50, const(0.8))
            fr[("E1_lr1e4", s)] = synth_run(150, collapse_at(12))
    elif name == "joint":
        for s in range(5):
            fr[("E1_lr1e4", s)] = synth_run(150, collapse_at(12))
    elif name == "all":
        for s in range(5):
            fr[("E1_lr1e4", s)] = synth_run(150, collapse_at(12))
            fr[("E36_lr1e4", s)] = synth_run(150, collapse_at(20))
    elif name == "none":
        for s in range(5):
            fr[("E1_lr1e3", s)] = synth_run(50, const(0.8))
            fr[("E36_lr1e3", s)] = synth_run(50, const(0.8))
    elif name == "nmaj":            # E1_lr1e3: 3 floors, 2 above -> SPLIT (N_MAJ = 3 would call it COLLAPSED)
        for s in (3, 4):
            fr[("E1_lr1e3", s)] = synth_run(50, const(0.8))
    elif name == "window":          # E36_lr1e3 s0: F + 0.0005 on t31-50 but 0.95 at t30; s4 above
        fr[("E36_lr1e3", 0)] = synth_run(50, lambda k: 0.95 if k == 30 else 0.114 + 0.0005)
        fr[("E36_lr1e3", 4)] = synth_run(50, const(0.8))
    elif name == "margin":          # E1_lr1e3 all at F + 0.0005: inside +m_W
        for s in range(5):
            fr[("E1_lr1e3", s)] = synth_run(50, lambda k: 0.9 if k < 10 else 0.114 + 0.0005)
    elif name == "q2k":             # lifetimes c + [-1, -.5, 0, .5, 1] pt with c = 84.23 - 2.5 SE
        sd = math.sqrt(2.5 / 4)
        se = math.sqrt(sd * sd / 5 + 0.70 ** 2 / 5)
        c = 84.23 - 2.5 * se
        for s, off in enumerate((-1.0, -0.5, 0.0, 0.5, 1.0)):
            fr[("E36_lr1e4", s)] = synth_run(150, const((c + off) / 100))
    elif name == "pre":             # lr 1e-4: mu2 100 on t1-10, 10 on t11-30, 12 on t31-50 -> running
        for a in ("E1_lr1e4", "E36_lr1e4"):
            for s in range(5):
                old = fr[(a, s)]
                fr[(a, s)] = synth_run(150, lambda k, o=old: float(o.online.iloc[k - 1]),
                                       mu=lambda k: 100.0 if k <= 10 else (10.0 if k <= 30 else 12.0))
    return fr


SCENARIOS = {
    "base": {"Q1": "LR_DOMINATES", "Q2": "REPRODUCED", "Q3": "E1=RUNNING, E36=RUNNING",
             "Q4": "E1=COLLAPSED_BY_150, E36=ABOVE_AT_150"},
    "alpha": {"Q1": "ALPHA_DOMINATES"},
    "joint": {"Q1": "JOINT_ONLY"},
    "all": {"Q1": "ALL_COLLAPSE"},
    "none": {"Q1": "NONE_COLLAPSE"},
    "nmaj": {"Q1": "OTHER(E1_lr1e3=SPLIT, E36_lr1e3=COLLAPSED, E1_lr1e4=ABOVE, E36_lr1e4=ABOVE)"},
    "window": {"Q1": "LR_DOMINATES"},
    "margin": {"Q1": "LR_DOMINATES"},
    "q2k": {"Q2": "REPRODUCED"},
    "pre": {"Q3": "E1=RUNNING, E36=RUNNING"},
}


def write_scenario(name: str) -> Path:
    root = SCR / "synthetic" / name / "runs"
    if not root.exists():
        for (a, s), df in scenario_frames(name).items():
            d = root / f"{a}_s{s}"
            d.mkdir(parents=True, exist_ok=True)
            df.assign(arm=a).to_csv(d / "per_task.csv", index=False)
            (d / "provenance.json").write_text(json.dumps({"status": "COMPLETE"}))
    return root


def s6_eval(V) -> dict:
    out = {}
    ok = True
    for name, want in SCENARIOS.items():
        res, runs = V.judge(write_scenario(name), "ext")
        got = {"Q1": res["Q1"], "Q2": res["Q2"]["label"], "Q3": res["Q3"], "Q4": res["Q4"]}
        good = all(got[k] == v for k, v in want.items())
        out[name] = {"got": got, "ok": good}
        ok = ok and good
    # the main stage refuses a short lr 1e-4 run and judges on 50 tasks
    short = SCR / "synthetic" / "short" / "runs"
    if short.exists():
        shutil.rmtree(short)
    shutil.copytree(write_scenario("base"), short)
    df = read_rows(short / "E1_lr1e4_s0" / "per_task.csv")
    df.head(49).to_csv(short / "E1_lr1e4_s0" / "per_task.csv", index=False)
    try:
        V.judge(short, "main")
        refused = False
    except SystemExit:
        refused = True
    res, runs = V.judge(write_scenario("base"), "main")
    main_ok = res["Q1"] == "LR_DOMINATES" and "Q4" not in res
    V.write(res, runs, fresh(SCR / "synthetic" / "_out"))
    wrote = (SCR / "synthetic" / "_out" / "summary_main.md").exists()
    out["refuse_short"] = refused
    out["main_stage"] = main_ok and wrote
    out["pass"] = ok and refused and main_ok and wrote
    return out


S6_MUT = [
    ("N_MAJ 3", ("N_MAJ = 4 ", "N_MAJ = 3 ")),
    ("window from 30", ("WIN = (31, 50)", "WIN = (30, 50)")),
    ("margin sign", ("M_W = Z * S_C / math.sqrt(20)", "M_W = -Z * S_C / math.sqrt(20)")),
    ("Q2 two SE", ("Q2_K = 3.0", "Q2_K = 2.0")),
    ("Q3 comparison window", ("PRE = (11, 30)", "PRE = (1, 30)")),
]


def check_S6() -> dict:
    fresh(SCR / "synthetic")                     # the scenarios are rebuilt once per check run
    res = {"real": s6_eval(load_module(VERDICT)), "mutations": {}}
    for name, sub in S6_MUT:
        try:
            res["mutations"][name] = {"detected": not s6_eval(load_module(VERDICT, sub))["pass"]}
        except Exception as e:          # noqa: BLE001
            res["mutations"][name] = {"detected": True, "error": repr(e)[:200]}
    res["pass"] = res["real"]["pass"] and all(m["detected"] for m in res["mutations"].values())
    return res


# --------------------------------------------------------------------------
# S7  CLI and provenance
# --------------------------------------------------------------------------

def s7_eval(R, tag: str) -> dict:
    d = fresh(SCR / f"S7_{tag}")
    R.run("E36_lr1e4", 2, 2, d, epochs=2, threads=1)
    torch.set_num_threads(1)
    p = json.loads((d / "provenance.json").read_text())
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    h = p["history"][0]
    got = {"status": p["status"] == "COMPLETE", "threads": h["threads"] == 1, "git": h["git_hash"] == head,
           "labels": len(p["label_sha256"]) == 50, "arm": (p["lr"], p["param"], p["kind"]) == (1e-4, 3.6, "elu"),
           "spec": p["spec_sha256"] == hashlib.sha256(SPEC.read_bytes()).hexdigest(),
           "lifetime_none": p["online_lifetime_t1_50"] is None}
    got["pass"] = all(got.values())
    got["dirty_at_start"] = h["git_dirty"]
    return got


S7_MUT = [("threads + 1", ("torch.set_num_threads(threads)", "torch.set_num_threads(threads + 1)"))]


def check_S7() -> dict:
    res = {"real": s7_eval(load_module(RUNNER), "real"), "mutations": {}}
    for i, (name, sub) in enumerate(S7_MUT):
        res["mutations"][name] = {"detected": not s7_eval(load_module(RUNNER, sub), f"m{i}")["pass"]}
    torch.set_num_threads(1)
    # the CLI itself, in a child process (argument parsing, main)
    d = fresh(SCR / "S7_cli")
    r = subprocess.run([sys.executable, str(RUNNER), "--arm", "R_lr1e4", "--seed", "0", "--tasks", "1",
                        "--epochs", "1", "--threads", "1", "--out", str(d)],
                       capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
    cli = r.returncode == 0 and json.loads((d / "provenance.json").read_text())["status"] == "COMPLETE"
    res["cli"] = {"ok": cli, "stderr": r.stderr[-300:]}
    res["pass"] = res["real"]["pass"] and cli and all(m["detected"] for m in res["mutations"].values())
    return res


# --------------------------------------------------------------------------
# S-cost
# --------------------------------------------------------------------------

def check_cost() -> dict:
    d = fresh(SCR / "cost")
    t0 = time.time()
    r = subprocess.run([sys.executable, str(RUNNER), "--arm", "E1_lr1e3", "--seed", "0", "--tasks", "1",
                        "--threads", "1", "--out", str(d)],
                       capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
    wall = time.time() - t0
    p = json.loads((d / "provenance.json").read_text())
    sec = float(read_rows(d / "per_task.csv").sec.iloc[0])
    rss_gib = p["peak_rss_mib"] / 1024
    avail = [int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable:")][0] / 2 ** 20
    slots = int(max(0, min(6, (avail - 4.0) // (1.2 * rss_gib))))
    healthy = sec
    est = {"E_lr1e3_min": (10 * healthy + 40 * 3 * healthy) / 60, "E_lr1e4_150_min": 150 * healthy / 60 * 1.3,
           "anchor_min": 50 * healthy * 2 / 60}
    total_cpu_h = (10 * est["E_lr1e3_min"] + 10 * est["E_lr1e4_150_min"] + 10 * est["anchor_min"]) / 60
    return {"returncode": r.returncode, "sec_per_task": sec, "wall_s": wall, "peak_rss_gib": rss_gib,
            "mem_available_gib_now": avail, "slots": slots, "estimate_min": est, "total_cpu_h": total_cpu_h,
            "pass": r.returncode == 0}


# --------------------------------------------------------------------------

CHECKS = {"S1": None, "S2": check_S2, "S3": check_S3, "S4": check_S4, "S5": check_S5, "S6": check_S6,
          "S7": check_S7, "S-cost": check_cost}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--lc-src", default=str(LC_SRC))
    a = ap.parse_args()
    names = list(CHECKS) if a.only is None else a.only.split(",")
    target = OUT / "checks.json" if a.only is None else SCR / "checks_partial.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "src", "analysis/lc_elu_lr_0917",
                            "specs/spec_lc_elu_lr_0917.md"], capture_output=True, text=True).stdout.splitlines()
    res = {"run_id": "lc_elu_lr_0917", "started": dt.datetime.now().isoformat(timespec="seconds"),
           "git_hash": head, "git_dirty": dirty, "torch": torch.__version__, "checks": {}}
    for n in names:
        t0 = time.time()
        try:
            r = check_S1(Path(a.lc_src)) if n == "S1" else CHECKS[n]()
        except Exception:               # noqa: BLE001
            r = {"pass": False, "error": traceback.format_exc()[-2000:]}
        r["seconds"] = round(time.time() - t0, 1)
        res["checks"][n] = r
        print(f"{n}: {'PASS' if r['pass'] else 'FAIL'} ({r['seconds']} s)", flush=True)
        res["all_pass"] = all(v["pass"] for v in res["checks"].values()) and a.only is None and not dirty
        target.write_text(json.dumps(res, indent=1, default=str))
    res["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    res["all_pass"] = all(v["pass"] for v in res["checks"].values()) and a.only is None and not dirty
    target.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"all_pass": res["all_pass"], "dirty": dirty}))


if __name__ == "__main__":
    main()

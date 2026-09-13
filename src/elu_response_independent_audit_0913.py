#!/usr/bin/env python3
"""Independent, read-only audit of the response-anchor experiment.

This file does not import the experiment engine or runner.  It recomputes the
source cohort, anchored forward/gradients, Adam arithmetic, dense online-loss
contrasts, and verdict directly from raw saved artifacts.  It never launches a
training loop; the only optimizer operation is a one-step replay from each
registered diagnostic state.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "elu_response_anchor_0913"
SOURCE = Path("/home/issan/Projects/claude/elu_sunk_rescue_0913/results/elu_sunk_rescue_0913")
DATA = Path("/home/issan/Projects/claude/proj_004_drift/data/mnist")
BRANCHES = ("A", "B", "C", "D", "AF", "BF")
TASKS = tuple(range(21, 26))
STATE_STEPS = (0, 1, 20, 75, 6000)
PARAM_NAMES = ("W1", "b1", "W2", "b2", "W3", "b3")
EXPECTED = {
    "prereg_commit": "c4965d09c16534ceec23e074c11171404f8840ee",
    "engine_commit": "990fc224539240dc05bfbaee85682d60649ce38e",
    "spec_sha256": "332ed37ce9d2a561e58eff98e9fca21c1ed3b5f4a34ba9df221cd62802d27dea",
    "engine_sha256": "7bc3581312aa47a8e48de386452be42f7f65b24ba0ae71554bae3d8a1934f88c",
    "engine_validator_sha256": "f4123566823d5bf8c9de3f3eb9e55cce3565636fd809e2ed4c0ae79354edade3",
    "source_checkpoint_sha256": "4b62bad9ad48a8940d81e9d709c4d987abc74eb2dec7d0b563cbad19f4d2f0c9",
    "source_selection_sha256": "4e561bae77f004679f2c81ac6b8b13e832a637c802371adc1213dbc34fc119f7",
    "source_units_sha256": "4876dce6b0ba76593479b616e4c2b5762ccea7e724cbbbf241f17833998295ba",
    "source_rng_sha256": "b35fbf1747687a8bfc16653e2aec9da68819d36d12e417a1e10e3cf69dc06efb",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def arrsha(a) -> str:
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    ndim = raw[3]
    shape = [int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw[4 + 4 * ndim :], np.uint8).reshape(shape).copy()


def activ(z: torch.Tensor) -> torch.Tensor:
    return torch.where(z > 0, z, torch.expm1(z.clamp_max(0)))


def gate(z: torch.Tensor) -> torch.Tensor:
    return torch.where(z > 0, torch.ones_like(z), z.clamp_max(0).exp())


def fk(branch: str) -> tuple[bool, bool, bool]:
    if branch not in BRANCHES:
        raise ValueError(f"unknown branch {branch!r}")
    return branch in ("C", "D"), branch in ("B", "D", "BF"), branch in ("AF", "BF")


def forward(p, x, anchor0, anchor1, delta, branch):
    F, K, _ = fk(branch)
    z1 = x @ p[0].T + p[1]
    a1 = activ(z1)
    z2 = a1 @ p[2].T + p[3]
    live0, live1 = activ(z2), activ(z2 + delta)
    if not F and not K:
        a2 = live0
    elif not F and K:
        a2 = anchor0 + (live1 - anchor1)
    elif F and not K:
        a2 = anchor1 + (live0 - anchor0)
    else:
        a2 = live1
    # Preserve exact ordinary ELU for every unselected unit.
    a2 = torch.where(delta.ne(0)[None, :], a2, live0)
    logits = a2 @ p[4].T + p[5]
    return z1, a1, z2, a2, logits


def loss_accuracy(logits, y):
    ce = (logits.logsumexp(-1) - (logits * y).sum(-1)).mean()
    acc = (logits.argmax(-1) == y.argmax(-1)).to(logits.dtype).mean()
    return ce, acc


def independent_gradients(p, x, y, anchor0, anchor1, delta, branch):
    z1, a1, z2, a2, logits = forward(p, x, anchor0, anchor1, delta, branch)
    _, K, _ = fk(branch)
    g3 = (logits.softmax(-1) - y) / x.shape[0]
    g2 = (g3 @ p[4]) * gate(z2 + (delta if K else 0))
    g1 = (g2 @ p[2]) * gate(z1)
    gradients = [g1.T @ x, g1.sum(0), g2.T @ a1, g2.sum(0), g3.T @ a2, g3.sum(0)]
    ce, acc = loss_accuracy(logits, y)
    return gradients, ce, acc


def one_adam_step(p, m, v, t, gradients, frozen_reference, freeze_l1):
    nt = int(t) + 1
    c1, c2 = 1 - 0.9**nt, 1 - 0.999**nt
    nm = [0.9 * a + 0.1 * g for a, g in zip(m, gradients)]
    nv = [0.999 * a + 0.001 * g.square() for a, g in zip(v, gradients)]
    denominator = [(a / c2).sqrt() + 1e-8 for a in nv]
    predicted = [-0.001 * (a / c1) / d for a, d in zip(nm, denominator)]
    np_ = [a + d for a, d in zip(p, predicted)]
    if freeze_l1:
        np_[0] = frozen_reference[0].clone()
        np_[1] = frozen_reference[1].clone()
    realized = [a - b for a, b in zip(np_, p)]
    current = [0.1 * g / c1 for g in gradients]
    history = [0.9 * a / c1 for a in m]
    return dict(parameters=np_, adam_m=nm, adam_v=nv, t=nt, denominator=denominator,
                predicted_update=predicted, parameter_delta=realized,
                current_gradient_numerator=current, history_numerator=history)


def maxabs(a, b) -> float:
    return float((torch.as_tensor(a).detach() - torch.as_tensor(b).detach()).abs().max())


def maxabs_lists(a, b) -> float:
    if len(a) != len(b):
        raise AssertionError((len(a), len(b)))
    return max(maxabs(x, y) for x, y in zip(a, b))


def source_model_index(models, seed):
    wanted = dict(seed=seed, env="RL", act="ELU1", iv="ref")
    return next(i for i, model in enumerate(models) if model == wanted)


def audit_static_sources(result):
    files = {
        "spec": ROOT / "specs/spec_elu_response_anchor_0913.md",
        "engine": ROOT / "src/elu_response_engine_0913.py",
        "engine_validator": ROOT / "src/elu_response_engine_validate_0913.py",
        "initial_validator": ROOT / "src/elu_response_initial_validate_0913.py",
        "initial_validation": OUT / "initial_independent_validation.json",
        "source_checkpoint": SOURCE / "prefix/checkpoint_20.pt",
        "source_selections": SOURCE / "RL_t20_l2/selections.json",
        "source_units": SOURCE / "prefix/units.npz",
        "source_rng": SOURCE / "prefix/rng_hashes.json",
    }
    for label, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"missing {label}: {path}")
    observed = {label: sha(path) for label, path in files.items()}
    assert observed["spec"] == EXPECTED["spec_sha256"], observed
    assert observed["engine"] == EXPECTED["engine_sha256"], observed
    assert observed["engine_validator"] == EXPECTED["engine_validator_sha256"], observed
    assert observed["source_checkpoint"] == EXPECTED["source_checkpoint_sha256"], observed
    assert observed["source_selections"] == EXPECTED["source_selection_sha256"], observed
    assert observed["source_units"] == EXPECTED["source_units_sha256"], observed
    assert observed["source_rng"] == EXPECTED["source_rng_sha256"], observed
    initial = json.loads(files["initial_validation"].read_text())
    assert initial["status"] == "PASS" and initial["full_maxabs"] == 0
    assert initial["batch_maxabs"] <= 2e-4, initial["batch_maxabs"]
    assert initial["optimizer_updates"] == 0 and initial["unselected_exact"]
    result["source_sha256"] = observed
    result["initial_output_validation"] = {
        "status": initial["status"], "full_maxabs": initial["full_maxabs"],
        "batch_maxabs": initial["batch_maxabs"], "optimizer_updates": initial["optimizer_updates"]}
    return files


@torch.no_grad()
def audit_source_cohort(files, result):
    checkpoint = torch.load(files["source_checkpoint"], map_location="cpu", weights_only=False)
    selections = json.loads(files["source_selections"].read_text())["seeds"]
    assert int(checkpoint["task"]) == 20 and int(checkpoint["t"]) == 120000
    assert [s["seed"] for s in selections] == [0, 1, 2]
    images = read_idx(DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255
    cohort = {}
    with np.load(files["source_units"], allow_pickle=False) as units:
        for selection in selections:
            seed = selection["seed"]
            ids = np.asarray(selection["target20"], dtype=np.int64)
            deltas = np.asarray(selection["deltas20"], dtype=np.float32)
            assert ids.shape == (20,) and len(set(ids.tolist())) == 20
            assert deltas.shape == (20,) and np.all(deltas >= 0)
            model_i = source_model_index(checkpoint["models"], seed)
            past_q = np.stack([units[f"lowgate_l2_t{task}"][model_i, ids] for task in range(16, 21)])
            assert bool(np.all(past_q >= 0.95)), (seed, past_q.min())
            p = [q[model_i] for q in checkpoint["parameters"]]
            x = torch.from_numpy(images[np.asarray(checkpoint["subset"][seed])])
            z1 = x @ p[0].T + p[1]
            z2 = activ(z1) @ p[2].T + p[3]
            expected_delta = torch.clamp(-1 - z2.mean(0)[torch.from_numpy(ids)], min=0)
            delta_error = maxabs(expected_delta, torch.from_numpy(deltas))
            assert delta_error <= 2e-5, (seed, delta_error)
            cohort[str(seed)] = dict(target20=ids.tolist(), delta_maxabs=delta_error,
                                     past_q_min=float(past_q.min()), past_tasks=[16, 17, 18, 19, 20])
    result["source_cohort"] = cohort
    return checkpoint, selections, images


def synthetic_sign_stable_finite_difference(result):
    """Exercise all F/K cells with every ELU argument separated from zero."""
    torch.manual_seed(913513)
    dtype = torch.float64
    branches = ("A", "B", "C", "D")
    errs, margins = {}, {}
    for branch in branches:
        p = [
            (torch.randn(100, 784, dtype=dtype) * 1e-4).requires_grad_(),
            torch.full((100,), 2.0, dtype=dtype, requires_grad=True),
            (torch.randn(100, 100, dtype=dtype) * 1e-4).requires_grad_(),
            torch.full((100,), -2.0, dtype=dtype, requires_grad=True),
            (torch.randn(10, 100, dtype=dtype) * 0.02).requires_grad_(),
            (torch.randn(10, dtype=dtype) * 0.02).requires_grad_(),
        ]
        x = torch.rand(7, 784, dtype=dtype)
        labels = torch.arange(7) % 10
        y = torch.nn.functional.one_hot(labels, 10).to(dtype)
        delta = torch.zeros(100, dtype=dtype); delta[3:23] = 0.4
        source_z2 = torch.full((7, 100), -2.6, dtype=dtype)
        anchor0, anchor1 = activ(source_z2), activ(source_z2 + delta)
        got, _, _ = independent_gradients(p, x, y, anchor0, anchor1, delta, branch)
        logits = forward(p, x, anchor0, anchor1, delta, branch)[-1]
        loss = loss_accuracy(logits, y)[0]
        auto = torch.autograd.grad(loss, p)
        assert maxabs_lists(got, auto) < 1e-12
        direction = [torch.randn_like(q) * 0.01 for q in p]
        analytic = sum((g * d).sum() for g, d in zip(got, direction))
        eps = 1e-6
        def objective(sign):
            pp = [q.detach() + sign * eps * d for q, d in zip(p, direction)]
            return loss_accuracy(forward(pp, x, anchor0, anchor1, delta, branch)[-1], y)[0]
        numeric = (objective(1) - objective(-1)) / (2 * eps)
        rel = float((numeric - analytic).abs() / analytic.abs().clamp_min(1e-12))
        args = []
        for sign in (-1, 0, 1):
            pp = [q.detach() + sign * eps * d for q, d in zip(p, direction)]
            z1, _, z2, _, _ = forward(pp, x, anchor0, anchor1, delta, branch)
            args.extend([z1, z2, z2 + delta])
        margin = min(float(q.abs().min()) for q in args)
        assert margin > 1.0 and rel < 1e-4, (branch, margin, rel)
        errs[branch], margins[branch] = rel, margin
    result["synthetic_finite_difference"] = {
        "status": "PASS", "relative_error": errs, "minimum_argument_abs": margins,
        "note": "All z1, z2, and z2+delta values under both perturbations remain over 1 from zero."}


def classify(values):
    if all(v > 0.01 for v in values):
        return "DIRECTIONAL_RESPONSE_SUPPORT"
    if all(v < -0.01 for v in values):
        return "DIRECTIONAL_RESPONSE_HARM"
    return "INCONCLUSIVE"


def ci3(values):
    x = np.asarray(values, dtype=np.float64)
    mean = float(x.mean()); sd = float(x.std(ddof=1))
    half = 4.302652729911275 * sd / math.sqrt(3)
    return mean, sd, mean - half, mean + half


def csv_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def audit_dense_outcomes(result):
    learning = OUT / "learning.npz"
    rows_path = OUT / "rows.csv"
    provenance_path = OUT / "provenance.json"
    validation_path = OUT / "validation.json"
    for path in (learning, rows_path, provenance_path, validation_path):
        if not path.exists():
            raise FileNotFoundError(path)
    provenance = json.loads(provenance_path.read_text())
    validation = json.loads(validation_path.read_text())
    assert provenance.get("status") == "COMPLETE", provenance.get("status")
    assert validation.get("status") == "PASS", validation.get("status")
    assert provenance.get("prereg_commit") == EXPECTED["prereg_commit"]
    assert provenance.get("spec_sha256") == EXPECTED["spec_sha256"]
    assert provenance.get("engine_sha256") == EXPECTED["engine_sha256"]
    rows = csv_rows(rows_path)
    expected_grid = {(seed, branch, task, step) for seed in range(3) for branch in BRANCHES
                     for task in TASKS for step in (0,1,2,5,10,20,25,50,75,150,375,750,1500,3000,6000)}
    actual_grid = {(int(r["seed"]), r["branch"], int(r["task"]), int(r["step"])) for r in rows}
    assert len(rows) == 1350 and actual_grid == expected_grid
    with np.load(learning, allow_pickle=False) as z:
        online_ce = np.asarray(z["online_ce"])
        online_acc = np.asarray(z["online_acc"])
        tasks = np.asarray(z["tasks"]).astype(int)
        seeds = np.asarray(z["model_seed"]).astype(int)
        branches = np.asarray(z["model_branch"]).astype(str)
    assert online_ce.shape == (5, 18, 6000) and online_acc.shape == online_ce.shape
    assert tasks.tolist() == list(TASKS)
    assert np.isfinite(online_ce).all() and np.isfinite(online_acc).all()
    assert ((online_acc >= 0) & (online_acc <= 1)).all()
    model_index = {(int(s), b): i for i, (s, b) in enumerate(zip(seeds, branches))}
    assert set(model_index) == {(s, b) for s in range(3) for b in BRANCHES}
    audit_rows, gains = [], []
    for seed in range(3):
        values = {b: float(np.mean(online_ce[0, model_index[seed, b]], dtype=np.float64)) for b in BRANCHES}
        row = dict(seed=seed, **{f"L_{b}": values[b] for b in BRANCHES},
                   A_minus_B=values["A"] - values["B"], C_minus_D=values["C"] - values["D"],
                   response_interaction=(values["C"] - values["D"]) - (values["A"] - values["B"]),
                   AF_minus_BF=values["AF"] - values["BF"], A_minus_C=values["A"] - values["C"],
                   B_minus_D=values["B"] - values["D"], A_minus_D=values["A"] - values["D"])
        gains.append(row["A_minus_B"]); audit_rows.append(row)
    mean, sd, low, high = ci3(gains)
    verdict = classify(gains)
    write_csv(OUT / "independent_primary.csv", audit_rows)
    # Compare the separately written aggregation outputs when present.
    report_error = None
    if (OUT / "contrasts.csv").exists() and (OUT / "verdict.csv").exists():
        contrast_rows = csv_rows(OUT / "contrasts.csv")
        reported = [r for r in contrast_rows if r["metric"] == "online_ce_task21"]
        assert len(reported) == 3
        report_error = max(abs(float(next(r for r in reported if int(r["seed"]) == seed)["A_minus_B"]) - gains[seed]) for seed in range(3))
        assert report_error < 1e-12, report_error
        verdict_rows = csv_rows(OUT / "verdict.csv")
        vr = next(r for r in verdict_rows if r["metric"] == "online_ce_task21" and r["contrast"] == "A_minus_B")
        assert vr["pilot_label"] == verdict
        assert max(abs(float(vr[k]) - v) for k, v in zip(("mean", "sd", "ci95_low", "ci95_high"), (mean, sd, low, high))) < 1e-12
    result["outcomes"] = dict(status="PASS", dense_shape=list(online_ce.shape), full_probe_rows=len(rows),
                              task21_A_minus_B=gains, mean=mean, sd=sd, ci95=[low, high],
                              pilot_label=verdict, aggregation_maxabs=report_error,
                              accumulation="NumPy float64 mean over every one of 6000 saved pre-update CE values")
    result["run_source_sha256"] = {p.name: sha(p) for p in (learning, rows_path, provenance_path, validation_path)}
    return provenance


def state_key(task, step):
    return f"task{task}_step{step}"


def normalize_states(raw):
    if isinstance(raw, dict) and "states" in raw:
        raw = raw["states"]
    if isinstance(raw, list):
        return {state_key(int(x["task"]), int(x["step"])): x for x in raw}
    if not isinstance(raw, dict):
        raise TypeError("diagnostic states must be a dict or list")
    return raw


def get_list(state, *names):
    for name in names:
        if name in state:
            value = state[name]
            return list(value) if isinstance(value, (tuple, list)) else value
    raise KeyError(f"none of {names} in state keys {sorted(state)}")


def branch_state(state, model_i, names):
    value = get_list(state, *names)
    return [torch.as_tensor(q[model_i]).cpu() for q in value]


def audit_saved_states(checkpoint, selections, images, result):
    """Audit runner-provided complete state/diagnostic archive.

    Expected archive: diagnostic_states.pt with either a ``states`` mapping or
    direct mapping from ``task{T}_step{U}`` to dictionaries.  Each dictionary
    contains model metadata, parameters/adam_m/adam_v/t, frozen_reference,
    batch_x/batch_y (or x/y), anchor0/anchor1/delta, and a saved ``diagnostic``
    dictionary using the engine diagnose_step field names.
    """
    path = OUT / "diagnostic_states.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    archive = torch.load(path, map_location="cpu", weights_only=False)
    states = normalize_states(archive)
    expected = {state_key(task, step) for task in TASKS for step in STATE_STEPS}
    assert set(states) == expected, (sorted(states), sorted(expected))
    freeze_maxabs = 0.0; diag_errors = {}; replay_errors = {}; initial_state_errors = {}
    source_by_seed = {}
    for seed in range(3):
        idx = source_model_index(checkpoint["models"], seed)
        source_by_seed[seed] = {
            "parameters": [q[idx].cpu() for q in checkpoint["parameters"]],
            "adam_m": [q[idx].cpu() for q in checkpoint["adam_m"]],
            "adam_v": [q[idx].cpu() for q in checkpoint["adam_v"]],
        }
    for key in sorted(expected):
        state = states[key]
        models = state["models"]
        assert [(m["seed"], m["branch"]) for m in models] == [(s, b) for s in range(3) for b in BRANCHES]
        task, step = int(state["task"]), int(state["step"])
        assert key == state_key(task, step)
        t = int(torch.as_tensor(state["t"]))
        assert t == 120000 + (task - 21) * 6000 + step, (key, t)
        for model_i, model in enumerate(models):
            seed, branch = int(model["seed"]), model["branch"]
            p = branch_state(state, model_i, ("parameters", "p"))
            m = branch_state(state, model_i, ("adam_m", "m"))
            v = branch_state(state, model_i, ("adam_v", "v"))
            reference = branch_state(state, model_i, ("frozen_reference",))
            if branch in ("AF", "BF"):
                freeze_maxabs = max(freeze_maxabs, maxabs(p[0], reference[0]), maxabs(p[1], reference[1]))
            if task == 21 and step == 0:
                initial_state_errors[f"s{seed}_{branch}_parameters"] = maxabs_lists(p, source_by_seed[seed]["parameters"])
                initial_state_errors[f"s{seed}_{branch}_adam_m"] = maxabs_lists(m, source_by_seed[seed]["adam_m"])
                initial_state_errors[f"s{seed}_{branch}_adam_v"] = maxabs_lists(v, source_by_seed[seed]["adam_v"])
            x = torch.as_tensor(state.get("batch_x", state.get("x")))[model_i].cpu()
            y = torch.as_tensor(state.get("batch_y", state.get("y")))[model_i].cpu()
            a0 = torch.as_tensor(state["anchor0"])[model_i].cpu()
            a1 = torch.as_tensor(state["anchor1"])[model_i].cpu()
            delta = torch.as_tensor(state["delta"])[model_i].cpu()
            gradients, ce, acc = independent_gradients(p, x, y, a0, a1, delta, branch)
            own = one_adam_step(p, m, v, t, gradients, reference, fk(branch)[2])
            saved = state["diagnostic"]
            def saved_branch(name):
                value = saved[name]
                if isinstance(value, (list, tuple)):
                    return [torch.as_tensor(q[model_i]).cpu() for q in value]
                return torch.as_tensor(value)[model_i].cpu()
            checks = {
                "gradients": maxabs_lists(gradients, saved_branch("gradients")),
                "current_gradient_numerator": maxabs_lists(own["current_gradient_numerator"], saved_branch("current_gradient_numerator")),
                "history_numerator": maxabs_lists(own["history_numerator"], saved_branch("history_numerator")),
                "denominator": maxabs_lists(own["denominator"], saved_branch("denominator")),
                "predicted_update": maxabs_lists(own["predicted_update"], saved_branch("predicted_update")),
                "parameter_delta": maxabs_lists(own["parameter_delta"], saved_branch("parameter_delta")),
                "ce": maxabs(ce, saved_branch("ce")), "acc": maxabs(acc, saved_branch("acc")),
            }
            diag_errors[f"{key}_s{seed}_{branch}"] = max(checks.values())
            assert checks["gradients"] <= 2e-5 and max(checks.values()) <= 5e-5, (key, seed, branch, checks)
            # The task21 step0 -> step1 pair is a separately saved actual-step oracle.
            if task == 21 and step == 0:
                next_state = states[state_key(21, 1)]
                actual_next = branch_state(next_state, model_i, ("parameters", "p"))
                replay_errors[f"s{seed}_{branch}"] = maxabs_lists(own["parameters"], actual_next)
                assert replay_errors[f"s{seed}_{branch}"] <= 5e-5
    assert freeze_maxabs == 0, freeze_maxabs
    assert max(initial_state_errors.values()) == 0, initial_state_errors
    result["saved_states"] = dict(status="PASS", state_count=len(states), diagnostic_cases=len(diag_errors),
                                  diagnostic_maxabs=max(diag_errors.values()), task21_step0_to_1_replay_maxabs=max(replay_errors.values()),
                                  frozen_W1_b1_maxabs=freeze_maxabs, initial_source_state_maxabs=max(initial_state_errors.values()),
                                  checked_steps=list(STATE_STEPS), checked_tasks=list(TASKS))
    result["run_source_sha256"][path.name] = sha(path)


def audit_rng(provenance, result):
    source_rows = json.loads((SOURCE / "prefix/rng_hashes.json").read_text())
    expected = [row for row in source_rows if row["task"] in TASKS]
    observed = provenance.get("rng_hashes")
    if observed is None and (OUT / "rng_hashes.json").exists():
        observed = json.loads((OUT / "rng_hashes.json").read_text())
    assert observed == expected, (observed, expected)
    result["rng"] = dict(status="PASS", tasks=list(TASKS), exact_source_hash_rows=True,
                         source_rng_sha256=sha(SOURCE / "prefix/rng_hashes.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result = dict(status="IN_PROGRESS", audit_code_sha256=sha(Path(__file__)),
                  independence="No imports from response engine, runner, or report; direct formulas and float64 aggregation.")
    files = audit_static_sources(result)
    checkpoint, selections, images = audit_source_cohort(files, result)
    synthetic_sign_stable_finite_difference(result)
    if args.synthetic_only:
        result["status"] = "PASS_PRE_RUN"
        path = OUT / "independent_prerun_audit.json"
    else:
        provenance = audit_dense_outcomes(result)
        audit_rng(provenance, result)
        audit_saved_states(checkpoint, selections, images, result)
        result["status"] = "PASS"
        path = OUT / "independent_audit.json"
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

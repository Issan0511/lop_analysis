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
        rel = float(((numeric - analytic).abs() / analytic.abs().clamp_min(1e-12)).detach())
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
    run_config = json.loads((OUT / "run_config.json").read_text())
    assert provenance.get("status") == "COMPLETE", provenance.get("status")
    assert validation.get("status") == "PASS", validation.get("status")
    assert provenance.get("prereg_commit") == EXPECTED["prereg_commit"]
    assert provenance.get("spec_sha256") == EXPECTED["spec_sha256"]
    assert provenance.get("engine_sha256") == EXPECTED["engine_sha256"]
    runner_path = ROOT / "src/elu_response_anchor_0913.py"
    assert provenance.get("runner_sha256") == sha(runner_path)
    assert run_config.get("runner_sha256") == provenance.get("runner_sha256")
    assert run_config.get("launch_commit") == provenance.get("launch_commit")
    assert run_config.get("models") == provenance.get("models")
    for name, digest in validation.get("result_sha256", {}).items():
        assert sha(OUT / name) == digest, name
    for name, digest in provenance.get("result_sha256", {}).items():
        assert sha(OUT / name) == digest, name
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
    result["run_source_sha256"] = {p.name: sha(p) for p in (learning, rows_path, provenance_path, validation_path,
                                                               OUT / "run_config.json", runner_path,
                                                               OUT / "contrasts.csv", OUT / "verdict.csv")}
    return provenance


def continued_schedules(checkpoint):
    generators = {}
    for role, states in checkpoint["rng_states"].items():
        generators[role] = {}
        for seed, state in states.items():
            generator = torch.Generator(); generator.set_state(state)
            generators[role][int(seed)] = generator
    schedules = {}
    for task in TASKS:
        orders = {seed: torch.stack([torch.randperm(1200, generator=generators["env_batch_0913"][seed])
                                     for _ in range(80)]).reshape(6000, 16) for seed in range(3)}
        permutations = {seed: torch.randperm(784, generator=generators["env_perm_0913"][seed]) for seed in range(3)}
        labels = {seed: torch.randint(10, (1200,), generator=generators["env_labels_0913"][seed]) for seed in range(3)}
        hashes = {"task": task}
        hashes.update({f"order_s{s}": arrsha(orders[s]) for s in range(3)})
        hashes.update({f"perm_s{s}": arrsha(permutations[s]) for s in range(3)})
        hashes.update({f"labels_s{s}": arrsha(labels[s]) for s in range(3)})
        schedules[task] = dict(orders=orders, permutations=permutations, labels=labels, hashes=hashes)
    return schedules


def forward_batched(p, x, anchor0, anchor1, delta, models):
    z1 = torch.bmm(x, p[0].transpose(1, 2)) + p[1][:, None, :]
    a1 = activ(z1)
    z2 = torch.bmm(a1, p[2].transpose(1, 2)) + p[3][:, None, :]
    d = delta[:, None, :]
    live0, live1 = activ(z2), activ(z2 + d)
    F = torch.tensor([fk(m["branch"])[0] for m in models], device=x.device)[:, None, None]
    K = torch.tensor([fk(m["branch"])[1] for m in models], device=x.device)[:, None, None]
    cell = torch.where(F, torch.where(K, live1, anchor1 + (live0 - anchor0)),
                       torch.where(K, anchor0 + (live1 - anchor1), live0))
    a2 = torch.where(d.ne(0), cell, live0)
    logits = torch.bmm(a2, p[4].transpose(1, 2)) + p[5][:, None, :]
    return z1, a1, z2, a2, logits, K


def gradients_batched(p, x, y, anchor0, anchor1, delta, models):
    z1, a1, z2, a2, logits, K = forward_batched(p, x, anchor0, anchor1, delta, models)
    g3 = (logits.softmax(-1) - y) / x.shape[1]
    g2 = torch.bmm(g3, p[4]) * gate(z2 + torch.where(K, delta[:, None, :], 0))
    g1 = torch.bmm(g2, p[2]) * gate(z1)
    gradients = [torch.bmm(g1.transpose(1, 2), x), g1.sum(1),
                 torch.bmm(g2.transpose(1, 2), a1), g2.sum(1),
                 torch.bmm(g3.transpose(1, 2), a2), g3.sum(1)]
    ce = (logits.logsumexp(-1) - (logits * y).sum(-1)).mean(-1)
    acc = (logits.argmax(-1) == y.argmax(-1)).to(logits.dtype).mean(-1)
    return gradients, ce, acc


def adam_batched(p, m, v, t, gradients, reference, models):
    nt = torch.as_tensor(t, dtype=p[0].dtype, device=p[0].device) + 1
    c1, c2 = 1 - torch.pow(0.9, nt), 1 - torch.pow(0.999, nt)
    nm = [0.9 * a + 0.1 * g for a, g in zip(m, gradients)]
    nv = [0.999 * a + 0.001 * g.square() for a, g in zip(v, gradients)]
    denominator = [(a / c2).sqrt() + 1e-8 for a in nv]
    predicted = [-0.001 * (a / c1) / d for a, d in zip(nm, denominator)]
    np_ = [a + d for a, d in zip(p, predicted)]
    frozen = torch.tensor([fk(m_["branch"])[2] for m_ in models], device=p[0].device)
    np_[0] = torch.where(frozen[:, None, None], reference[0], np_[0])
    np_[1] = torch.where(frozen[:, None], reference[1], np_[1])
    current = [0.1 * g / c1 for g in gradients]; history = [0.9 * a / c1 for a in m]
    return dict(parameters=np_, adam_m=nm, adam_v=nv, t=nt, denominator=denominator,
                predicted_update=predicted, parameter_delta=[a - b for a, b in zip(np_, p)],
                current_gradient_numerator=current, history_numerator=history,
                decomposition_error=[(a + b) - n / c1 for a, b, n in zip(current, history, nm)],
                moment_m_delta=[a - b for a, b in zip(nm, m)], moment_v_delta=[a - b for a, b in zip(nv, v)])


@torch.no_grad()
def audit_anchor_cache(checkpoint, selections, images, result, device):
    path = OUT / "anchor_cache.pt"
    cache = torch.load(path, map_location="cpu", weights_only=False)
    models = cache["models"]
    assert [(m["seed"], m["branch"]) for m in models] == [(s, b) for s in range(3) for b in BRANCHES]
    expected_ids = torch.tensor([selections[m["seed"]]["target20"] for m in models])
    assert torch.equal(cache["selected_ids"], expected_ids)
    for seed in range(3):
        idx = source_model_index(checkpoint["models"], seed)
        assert np.array_equal(cache["source_subset"][seed], checkpoint["subset"][seed])
        for key, source_key in (("source_parameters_by_seed", "parameters"), ("source_adam_m_by_seed", "adam_m"),
                                ("source_adam_v_by_seed", "adam_v")):
            assert maxabs_lists(cache[key][seed], [q[idx] for q in checkpoint[source_key]]) == 0
    p = [torch.stack([cache["source_parameters_by_seed"][m["seed"]][k] for m in models]).to(device) for k in range(6)]
    x = torch.stack([torch.from_numpy(images[np.asarray(checkpoint["subset"][m["seed"]])]) for m in models]).to(device)
    z1 = torch.bmm(x, p[0].transpose(1, 2)) + p[1][:, None, :]
    z2 = torch.bmm(activ(z1), p[2].transpose(1, 2)) + p[3][:, None, :]
    source_z2 = cache["source_z2"].to(device); delta = cache["delta"].to(device)
    z_error = maxabs(z2, source_z2)
    assert z_error <= 2e-4, z_error
    base_error = maxabs(activ(source_z2), cache["anchor_base"].to(device))
    shift_error = maxabs(activ(source_z2 + delta[:, None, :]), cache["anchor_shift"].to(device))
    assert base_error == 0 and shift_error == 0
    hashes = {name: arrsha(cache[key]) for name, key in (("anchor_z2", "source_z2"), ("anchor_base", "anchor_base"),
                                                                     ("anchor_shift", "anchor_shift"), ("delta", "delta"))}
    assert hashes == cache["anchor_sha256"], (hashes, cache["anchor_sha256"])
    result["anchor_cache"] = dict(status="PASS", source_z2_recompute_maxabs=z_error,
                                  base_formula_maxabs=base_error, shift_formula_maxabs=shift_error,
                                  cache_sha256=sha(path), immutable_hashes=hashes)
    result["run_source_sha256"][path.name] = sha(path)
    return cache, models


@torch.no_grad()
def audit_saved_states(checkpoint, selections, images, result):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache, models = audit_anchor_cache(checkpoint, selections, images, result, device)
    schedules = continued_schedules(checkpoint)
    replay_dir = OUT / "replay"; manifest_path = OUT / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "COMPLETE" and manifest["expected_files"] == 25
    expected_names = {f"task{task}_step{step:04d}.pt" for task in TASKS for step in STATE_STEPS}
    assert set(manifest["files"]) == expected_names
    for name, digest in manifest["files"].items():
        assert sha(replay_dir / name) == digest
    states = {(task, step): torch.load(replay_dir / f"task{task}_step{step:04d}.pt", map_location="cpu", weights_only=False)
              for task in TASKS for step in STATE_STEPS}
    raw_x = {seed: torch.from_numpy(images[np.asarray(checkpoint["subset"][seed])]) for seed in range(3)}
    references = [torch.stack([cache["source_parameters_by_seed"][m["seed"]][k] for m in models]).to(device) for k in range(6)]
    delta = cache["delta"].to(device); anchor0_full = cache["anchor_base"].to(device); anchor1_full = cache["anchor_shift"].to(device)
    initial_p = [q.to(device) for q in states[21, 0]["state"]["parameters"]]
    full_x = torch.stack([raw_x[m["seed"]] for m in models]).to(device)
    initial_outputs = forward_batched(initial_p, full_x, anchor0_full, anchor1_full, delta, models)
    model_index = {(m["seed"], m["branch"]): i for i, m in enumerate(models)}
    initial_pair_errors = {}
    for seed in range(3):
        for left, right in (("A", "B"), ("C", "D"), ("AF", "BF")):
            a, b = model_index[seed, left], model_index[seed, right]
            initial_pair_errors[f"s{seed}_{left}_{right}_features"] = maxabs(initial_outputs[3][a], initial_outputs[3][b])
            initial_pair_errors[f"s{seed}_{left}_{right}_logits"] = maxabs(initial_outputs[4][a], initial_outputs[4][b])
    assert max(initial_pair_errors.values()) == 0, initial_pair_errors
    unselected_error = 0.0
    for j in range(len(models)):
        mask = delta[j].eq(0)
        unselected_error = max(unselected_error, maxabs(initial_outputs[3][j, :, mask], activ(initial_outputs[2][j, :, mask])))
    assert unselected_error == 0
    freeze_error = 0.0; initial_error = 0.0; diagnostic_error = {}; replay_error = {}; boundary_error = {}
    for task in TASKS:
        schedule = schedules[task]
        for step in STATE_STEPS:
            artifact = states[task, step]; state = artifact["state"]; diag = artifact["diagnostic"]
            assert artifact["task"] == task and artifact["step"] == step and tuple(artifact["parameter_names"]) == PARAM_NAMES
            position = step if step < 6000 else 5999
            expected_role = "next_scheduled_batch" if step < 6000 else "retrospective_final_batch"
            assert artifact["diagnostic_batch_position_zero_based"] == position and artifact["diagnostic_batch_role"] == expected_role
            expected_indices = torch.stack([schedule["orders"][m["seed"]][position] for m in models])
            assert torch.equal(artifact["diagnostic_batch_indices"], expected_indices)
            t = int(state["t"]); assert t == 120000 + (task - 21) * 6000 + step
            p = [q.to(device) for q in state["parameters"]]; m = [q.to(device) for q in state["adam_m"]]; v = [q.to(device) for q in state["adam_v"]]
            frozen = torch.tensor([fk(m_["branch"])[2] for m_ in models], device=device)
            freeze_error = max(freeze_error, maxabs(p[0][frozen], references[0][frozen]), maxabs(p[1][frozen], references[1][frozen]))
            if task == 21 and step == 0:
                initial_error = max(initial_error, maxabs_lists(p, references))
                initial_error = max(initial_error, maxabs_lists(m, [torch.stack([cache["source_adam_m_by_seed"][mo["seed"]][k] for mo in models]).to(device) for k in range(6)]))
                initial_error = max(initial_error, maxabs_lists(v, [torch.stack([cache["source_adam_v_by_seed"][mo["seed"]][k] for mo in models]).to(device) for k in range(6)]))
            indices = expected_indices.to(device)
            x = torch.stack([raw_x[mo["seed"]][expected_indices[j]] for j, mo in enumerate(models)]).to(device)
            labels = torch.stack([schedule["labels"][mo["seed"]][expected_indices[j]] for j, mo in enumerate(models)]).to(device)
            y = torch.nn.functional.one_hot(labels, 10).float()
            mid = torch.arange(len(models), device=device)[:, None]
            a0, a1 = anchor0_full[mid, indices], anchor1_full[mid, indices]
            gradients, ce, acc = gradients_batched(p, x, y, a0, a1, delta, models)
            own = adam_batched(p, m, v, t, gradients, references, models)
            checks = {
                "gradients": maxabs_lists(gradients, [q.to(device) for q in diag["gradients"]]),
                "current": maxabs_lists(own["current_gradient_numerator"], [q.to(device) for q in diag["current_gradient_numerator"]]),
                "history": maxabs_lists(own["history_numerator"], [q.to(device) for q in diag["history_numerator"]]),
                "decomposition": maxabs_lists(own["decomposition_error"], [q.to(device) for q in diag["decomposition_error"]]),
                "denominator": maxabs_lists(own["denominator"], [q.to(device) for q in diag["denominator"]]),
                "predicted": maxabs_lists(own["predicted_update"], [q.to(device) for q in diag["predicted_update"]]),
                "realized": maxabs_lists(own["parameter_delta"], [q.to(device) for q in diag["parameter_delta"]]),
                "moment_m_delta": maxabs_lists(own["moment_m_delta"], [q.to(device) for q in diag["moment_m_delta"]]),
                "moment_v_delta": maxabs_lists(own["moment_v_delta"], [q.to(device) for q in diag["moment_v_delta"]]),
                "ce": maxabs(ce, diag["ce"].to(device)), "acc": maxabs(acc, diag["acc"].to(device)),
                "next_t": maxabs(own["t"], diag["next_t"].to(device)),
            }
            diagnostic_error[f"t{task}_u{step}"] = max(checks.values())
            assert checks["gradients"] <= 2e-5 and max(checks.values()) <= 5e-5, (task, step, checks)
            if step == 0:
                next_p = [q.to(device) for q in states[task, 1]["state"]["parameters"]]
                replay_error[f"t{task}"] = maxabs_lists(own["parameters"], next_p)
                assert replay_error[f"t{task}"] <= 5e-5
        if task < 25:
            left, right = states[task, 6000]["state"], states[task + 1, 0]["state"]
            boundary_error[f"t{task}_to_{task+1}"] = max(maxabs_lists(left[k], right[k]) for k in ("parameters", "adam_m", "adam_v"))
            assert boundary_error[f"t{task}_to_{task+1}"] == 0
    assert freeze_error == 0 and initial_error == 0
    final_checkpoint = torch.load(OUT / "checkpoint.pt", map_location="cpu", weights_only=False)
    final_state = states[25, 6000]["state"]
    final_error = max(maxabs_lists(final_checkpoint[k], final_state[k]) for k in ("parameters", "adam_m", "adam_v"))
    assert final_error == 0 and int(final_checkpoint["t"]) == 150000 and int(final_state["t"]) == 150000
    assert final_checkpoint["anchor_sha256"] == cache["anchor_sha256"]
    result["saved_states"] = dict(status="PASS", state_count=len(states), diagnostic_batches=len(diagnostic_error),
                                  diagnostic_maxabs=max(diagnostic_error.values()), per_task_step0_to_1_replay_maxabs=max(replay_error.values()),
                                  boundary_continuity_maxabs=max(boundary_error.values()), frozen_W1_b1_maxabs=freeze_error,
                                  initial_source_state_maxabs=initial_error, checked_steps=list(STATE_STEPS), checked_tasks=list(TASKS),
                                  initial_within_F_feature_logit_maxabs=max(initial_pair_errors.values()),
                                  initial_unselected_activation_maxabs=unselected_error, final_checkpoint_state_maxabs=final_error,
                                  replay_manifest_sha256=sha(manifest_path), checkpoint_sha256=sha(OUT / "checkpoint.pt"))
    result["run_source_sha256"][manifest_path.name] = sha(manifest_path)
    result["run_source_sha256"]["checkpoint.pt"] = sha(OUT / "checkpoint.pt")


def audit_units(result):
    path = OUT / "units.npz"
    required = {
        "z2_mean", "z2_std", "effective_argument_mean", "effective_argument_std", "response_gate_mean", "q_lowgate",
        "actual_activation_mean", "actual_activation_std", "actual_activation_varying_fraction",
        "W1_row_norm", "W1_centered_row_norm", "W2_row_norm", "W2_centered_row_norm", "W2_column_norm",
        "W3_row_norm", "W3_column_norm", "interval_delta_z2_mean", "interval_delta_z2_std",
        "interval_delta_actual_activation_mean", "interval_delta_actual_activation_std",
        "interval_delta_logits_mean", "interval_delta_logits_std",
    }
    with np.load(path, allow_pickle=False) as z:
        assert np.array_equal(z["tasks"], np.asarray(TASKS))
        assert np.array_equal(z["steps"], np.asarray((0,1,2,5,10,20,25,50,75,150,375,750,1500,3000,6000)))
        assert list(zip(z["model_seed"].astype(int), z["model_branch"].astype(str))) == [(s, b) for s in range(3) for b in BRANCHES]
        assert required.issubset(z.files), sorted(required - set(z.files))
        numeric = [key for key in z.files if key not in ("model_branch",)]
        assert all(np.isfinite(z[key]).all() for key in numeric)
        assert ((z["q_lowgate"] >= 0) & (z["q_lowgate"] <= 1)).all()
        assert ((z["actual_activation_varying_fraction"] >= 0) & (z["actual_activation_varying_fraction"] <= 1)).all()
        selected = z["selected_unit_id"].astype(np.int64)
        mask = z["selected_mask"].astype(bool)
        assert mask.shape == (18, 100) and np.all(mask.sum(1) == 20)
        assert all(np.array_equal(np.flatnonzero(mask[j]), np.sort(selected[j])) for j in range(18))
        gather = selected[None, None, :, :]
        gather_error = 0.0; gathered = 0
        for key in z.files:
            if not key.startswith("target_"):
                continue
            source = key[len("target_"):]
            if source in z.files and z[source].ndim == 4 and z[source].shape[-1] == 100:
                expected = np.take_along_axis(z[source], gather, axis=-1)
                gather_error = max(gather_error, float(np.max(np.abs(z[key] - expected))))
                gathered += 1
        assert gathered >= 20 and gather_error == 0
        step0 = np.where(z["steps"] == 0)[0]
        interval_keys = [key for key in z.files if key.startswith("interval_") and key != "interval_defined"]
        interval_zero_maxabs = max(float(np.max(np.abs(z[key][:, step0]))) for key in interval_keys)
        assert interval_zero_maxabs == 0
        result["mechanism_archive"] = dict(status="PASS", sha256=sha(path), finite_array_count=len(numeric),
                                            target_gather_arrays=gathered, target_gather_maxabs=gather_error,
                                            step0_interval_maxabs=interval_zero_maxabs)
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
        audit_units(result)
        result["status"] = "PASS"
        path = OUT / "independent_audit.json"
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

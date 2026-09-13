#!/usr/bin/env python3
"""Run the preregistered output-matched ELU response-anchor experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

import elu_environment_0913 as base
import elu_response_engine_0913 as response
import elu_sunk_rescue_0913 as source_helpers


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "elu_response_anchor_0913"
SPEC = ROOT / "specs" / "spec_elu_response_anchor_0913.md"
ENGINE = Path(response.__file__).resolve()
ENGINE_VALIDATOR = Path(__file__).with_name("elu_response_engine_validate_0913.py")
SOURCE = Path("/home/issan/Projects/claude/elu_sunk_rescue_0913/results/elu_sunk_rescue_0913")
SOURCE_CHECKPOINT = SOURCE / "prefix" / "checkpoint_20.pt"
SOURCE_UNITS = SOURCE / "prefix" / "units.npz"
SOURCE_RNG = SOURCE / "prefix" / "rng_hashes.json"
SOURCE_SELECTIONS = SOURCE / "RL_t20_l2" / "selections.json"
SOURCE_PROVENANCE = SOURCE / "prefix" / "provenance.json"

PREREG_COMMIT = "c4965d09c16534ceec23e074c11171404f8840ee"
UPSTREAM = "origin/codex/elu-response-anchor-0913"
TASKS = tuple(range(21, 26))
PROBE_STEPS = (0, 1, 2, 5, 10, 20, 25, 50, 75, 150, 375, 750, 1500, 3000, 6000)
STATE_STEPS = (0, 1, 20, 75, 6000)
PARAMETER_NAMES = ("W1", "b1", "W2", "b2", "W3", "b3")
SOURCE_HASHES = {
    "checkpoint_20.pt": "4b62bad9ad48a8940d81e9d709c4d987abc74eb2dec7d0b563cbad19f4d2f0c9",
    "units.npz": "4876dce6b0ba76593479b616e4c2b5762ccea7e724cbbbf241f17833998295ba",
    "rng_hashes.json": "b35fbf1747687a8bfc16653e2aec9da68819d36d12e417a1e10e3cf69dc06efb",
    "selections.json": "4e561bae77f004679f2c81ac6b8b13e832a637c802371adc1213dbc34fc119f7",
    "prefix_provenance.json": "09161664c59834ba3962455d8661163bec92b399d00e0993b2b470f3455d37a1",
}
SOURCE_PATHS = {
    "checkpoint_20.pt": SOURCE_CHECKPOINT,
    "units.npz": SOURCE_UNITS,
    "rng_hashes.json": SOURCE_RNG,
    "selections.json": SOURCE_SELECTIONS,
    "prefix_provenance.json": SOURCE_PROVENANCE,
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(value) -> str:
    a = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(a.tobytes()).hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    return array_sha(value.detach().cpu().numpy())


def tensor_identical(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        return False
    if a.is_floating_point() or a.is_complex():
        an, bn = torch.isnan(a), torch.isnan(b)
        return bool(torch.equal(an, bn) and torch.equal(a[~an], b[~bn]))
    return bool(torch.equal(a, b))


def snapshots_identical(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    return len(a) == len(b) and all(tensor_identical(x, y) for x, y in zip(a, b))


def tensors_finite(value) -> bool:
    if isinstance(value, torch.Tensor):
        return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(tensors_finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(tensors_finite(v) for v in value)
    return True


def json_x(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def csv_x(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refuse empty CSV {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def npz_x(path: Path, **arrays) -> None:
    with path.open("xb") as f:
        np.savez_compressed(f, **arrays)


def torch_x(path: Path, value) -> None:
    with path.open("xb") as f:
        torch.save(value, f)


def git(*args: str, text: bool = True):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=text).strip() if text else subprocess.check_output(["git", *args], cwd=ROOT)


def configure() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the registered common-batched execution")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def prereg_gate(commit: str, require_pushed: bool) -> dict:
    if not commit:
        raise ValueError("explicit --prereg is required")
    resolved = git("rev-parse", f"{commit}^{{commit}}")
    if resolved != PREREG_COMMIT:
        raise RuntimeError(f"--prereg resolves to {resolved}; expected pinned {PREREG_COMMIT}")
    subprocess.run(["git", "merge-base", "--is-ancestor", resolved, "HEAD"], cwd=ROOT, check=True)
    subprocess.run(["git", "merge-base", "--is-ancestor", resolved, UPSTREAM], cwd=ROOT, check=True)
    committed_spec = git("show", f"{resolved}:specs/{SPEC.name}", text=False)
    if hashlib.sha256(committed_spec).hexdigest() != sha(SPEC):
        raise RuntimeError("working spec differs from pinned preregistration")
    tracked = [str(Path(p).resolve().relative_to(ROOT)) for p in
               (Path(__file__), ENGINE, ENGINE_VALIDATOR, Path(base.__file__), Path(source_helpers.__file__), SPEC)]
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError(f"registered implementation files are not clean:\n{dirty}")
    code_at_head = git("show", f"HEAD:{Path(__file__).relative_to(ROOT).as_posix()}", text=False)
    if hashlib.sha256(code_at_head).hexdigest() != sha(Path(__file__)):
        raise RuntimeError("runner bytes differ from HEAD")
    head = git("rev-parse", "HEAD")
    upstream = git("rev-parse", UPSTREAM)
    if require_pushed and head != upstream:
        raise RuntimeError(f"launch HEAD {head} is not pushed upstream {upstream}")
    return {"prereg_commit": resolved, "launch_commit": head, "upstream_commit": upstream, "spec_sha256": sha(SPEC)}


def source_hash_gate() -> dict:
    got = {}
    for name, path in SOURCE_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        got[name] = sha(path)
        if got[name] != SOURCE_HASHES[name]:
            raise RuntimeError(f"source hash mismatch for {path}: {got[name]} != {SOURCE_HASHES[name]}")
    return got


def engine_qa_gate() -> dict:
    path = OUT / "engine_validation.json"
    if not path.is_file():
        raise RuntimeError("engine_validation.json PASS is required")
    qa = json.loads(path.read_text())
    if qa.get("status") != "PASS":
        raise RuntimeError("engine validation status is not PASS")
    if qa.get("engine_sha256") != sha(ENGINE):
        raise RuntimeError("engine validation is stale for the current engine")
    if qa.get("validator_sha256") != sha(ENGINE_VALIDATOR):
        raise RuntimeError("engine validation is stale for the current validator")
    return qa


def runner_qa_gate() -> dict:
    path = OUT / "runner_validation.json"
    if not path.is_file():
        raise RuntimeError("runner_validation.json PASS is required")
    qa = json.loads(path.read_text())
    if qa.get("status") != "PASS" or qa.get("runner_sha256") != sha(Path(__file__)):
        raise RuntimeError("runner validation is missing, failed, or stale")
    if qa.get("engine_sha256") != sha(ENGINE):
        raise RuntimeError("runner validation is stale for the current engine")
    if qa.get("base_sha256") != sha(Path(base.__file__)) or qa.get("source_helper_sha256") != sha(Path(source_helpers.__file__)):
        raise RuntimeError("runner validation is stale for the data/RNG helper code")
    if qa.get("source_sha256") != SOURCE_HASHES:
        raise RuntimeError("runner validation is tied to different source bytes")
    return qa


def load_source():
    hashes = source_hash_gate()
    parent_provenance = json.loads(SOURCE_PROVENANCE.read_text())
    if parent_provenance.get("status") != "COMPLETE":
        raise RuntimeError("pinned parent-prefix provenance is not COMPLETE")
    current_data_hashes = {p.name: sha(p) for p in base.DATA.glob("*gz")}
    if current_data_hashes != parent_provenance.get("data_sha256"):
        raise RuntimeError("current MNIST files differ from the pinned parent-prefix provenance")
    checkpoint = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint.get("task") != 20 or int(checkpoint["t"]) != 120000:
        raise RuntimeError("source is not the registered end-task20 state")
    if len(checkpoint["parameters"]) != 6 or len(checkpoint["adam_m"]) != 6 or len(checkpoint["adam_v"]) != 6:
        raise RuntimeError("source checkpoint parameter/moment schema mismatch")
    x, y, regenerated_subset = source_helpers.load_data()
    for seed in range(3):
        if not np.array_equal(np.asarray(checkpoint["subset"][seed]), np.asarray(regenerated_subset[seed])):
            raise RuntimeError(f"source subset/order mismatch for seed {seed}")
    selections_blob = json.loads(SOURCE_SELECTIONS.read_text())
    selections = sorted(selections_blob["seeds"], key=lambda q: int(q["seed"]))
    if [int(q["seed"]) for q in selections] != [0, 1, 2]:
        raise RuntimeError("selection seed keys mismatch")
    source_model_indices = [source_helpers.mid(seed, "RL") for seed in range(3)]
    with np.load(SOURCE_UNITS, allow_pickle=False) as units:
        persistence = {}
        for seed, model_index in enumerate(source_model_indices):
            ids = list(map(int, selections[seed]["target20"]))
            deltas = np.asarray(selections[seed]["deltas20"], dtype=np.float32)
            if len(ids) != 20 or len(set(ids)) != 20 or deltas.shape != (20,) or np.any(deltas < 0):
                raise RuntimeError(f"invalid fixed target/delta selection for seed {seed}")
            q = np.stack([np.asarray(units[f"lowgate_l2_t{task}"])[model_index, ids] for task in range(16, 21)])
            if not np.isfinite(q).all() or np.any(q < .95):
                raise RuntimeError(f"target IDs fail past-only q>=.95 persistence for seed {seed}")
            persistence[str(seed)] = {"ids": ids, "q_tasks16_20": q.tolist(), "minimum_q": float(q.min())}
    return checkpoint, x, y, regenerated_subset, selections, source_model_indices, hashes, persistence, current_data_hashes


def schedules(checkpoint) -> tuple[list[dict], dict]:
    generators = source_helpers.gens(checkpoint["rng_states"])
    registered = {int(row["task"]): row for row in json.loads(SOURCE_RNG.read_text())}
    result = []
    for task in TASKS:
        orders, permutation, labels, hashes = source_helpers.draw(generators)
        row = {"task": task, **hashes}
        if row != registered.get(task):
            raise RuntimeError(f"continued RNG hash mismatch at task {task}")
        result.append({"task": task, "orders": orders, "permutation": permutation, "labels": labels, "hashes": row})
    return result, generators


@torch.no_grad()
def initialize_engine(checkpoint, x, subset, selections, source_model_indices):
    models = response.MODELS
    engine = response.Engine(models=models, device="cuda")
    for group_name, destination in (("parameters", engine.p), ("adam_m", engine.m), ("adam_v", engine.v)):
        for k, dest in enumerate(destination):
            source = torch.stack([checkpoint[group_name][k][source_model_indices[m["seed"]]] for m in models]).to(engine.device)
            dest.copy_(source)
    engine.t.copy_(checkpoint["t"].to(engine.device))
    raw_x = {seed: torch.as_tensor(x[subset[seed]], device=engine.device) for seed in range(3)}
    for j, model in enumerate(models):
        engine.cx[j].copy_(raw_x[model["seed"]])
    source_z2 = []
    source_parameters = []
    source_m = []
    source_v = []
    for seed, model_index in enumerate(source_model_indices):
        p = [q[model_index:model_index + 1].to(engine.device) for q in checkpoint["parameters"]]
        source_z2.append(base.forward(p, raw_x[seed][None], torch.ones(1, 1, 1, dtype=torch.bool, device=engine.device))[2][0])
        source_parameters.append([q[model_index].clone() for q in checkpoint["parameters"]])
        source_m.append([q[model_index].clone() for q in checkpoint["adam_m"]])
        source_v.append([q[model_index].clone() for q in checkpoint["adam_v"]])
    source_z2_seed = torch.stack(source_z2)
    delta_seed = torch.zeros(3, 100, device=engine.device)
    delta_formula_error = {}
    for seed in range(3):
        ids = torch.tensor(selections[seed]["target20"], dtype=torch.long, device=engine.device)
        fixed = torch.tensor(selections[seed]["deltas20"], dtype=engine.p[0].dtype, device=engine.device)
        formula = torch.clamp(-1 - source_z2_seed[seed, :, ids].mean(0), min=0)
        delta_formula_error[str(seed)] = float((fixed - formula).abs().max())
        if delta_formula_error[str(seed)] > 2e-4:
            raise RuntimeError(f"fixed delta differs from source formula for seed {seed}: {delta_formula_error[str(seed)]}")
        delta_seed[seed, ids] = fixed
    # Cache the source forward in the exact 18-model geometry used by training.
    # The independent one-model forwards above remain the delta-formula cross-check.
    frozen_parameters = [q.clone() for q in engine.p]
    z20 = base.forward(frozen_parameters, engine.cx, engine.elu)[2]
    delta = torch.stack([delta_seed[m["seed"]] for m in models])
    engine.set_anchors(z20, delta)
    source_state_identical = True
    for j, model in enumerate(models):
        source_index = source_model_indices[model["seed"]]
        for group, current in (("parameters", engine.p), ("adam_m", engine.m), ("adam_v", engine.v)):
            source_state_identical &= all(torch.equal(q[j].cpu(), checkpoint[group][k][source_index]) for k, q in enumerate(current))
    if not source_state_identical:
        raise RuntimeError("branch source p/m/v copies are not bitwise identical")
    initial_reference = [q.clone() for q in engine.p]
    return engine, raw_x, z20, delta, source_z2_seed, delta_seed, initial_reference, source_state_identical, delta_formula_error, source_parameters, source_m, source_v


@torch.no_grad()
def initial_pair_checks(engine, task21_schedule) -> dict:
    index = {(m["seed"], m["branch"]): j for j, m in enumerate(engine.models)}
    order = torch.stack([task21_schedule["orders"][m["seed"]][0] for m in engine.models]).to(engine.device)

    def compare(outputs, suffix):
        result = {}
        for seed in range(3):
            for left, right in (("A", "B"), ("AF", "BF"), ("C", "D")):
                a, b = index[seed, left], index[seed, right]
                for name, tensor in (("features", outputs[3]), ("logits", outputs[4])):
                    result[f"s{seed}_{left}_{right}_{name}_{suffix}_maxabs"] = float((tensor[a] - tensor[b]).abs().max())
                    if not torch.allclose(tensor[a], tensor[b], atol=2e-4, rtol=1e-6):
                        raise RuntimeError(f"initial {suffix} {name} mismatch for seed {seed} {left}/{right}")
        return result

    full = engine.forward_full()
    batch = engine.forward(engine.cx[engine.mid, order], engine.anchor_base[engine.mid, order], engine.anchor_shift[engine.mid, order])
    checks = {**compare(full, "full1200"), **compare(batch, "batch16")}
    checks["maximum"] = max(checks.values(), default=0.0)
    return checks


def anchor_hashes(engine) -> dict:
    return {name: tensor_sha(tensor) for name, tensor in (
        ("anchor_z2", engine.anchor_z2), ("anchor_base", engine.anchor_base),
        ("anchor_shift", engine.anchor_shift), ("delta", engine.delta))}


@torch.no_grad()
def cosine_rows(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return (current * reference).sum(-1) / (current.norm(dim=-1) * reference.norm(dim=-1)).clamp_min(1e-30)


@torch.no_grad()
def measure(engine, task: int, step: int, previous: dict | None, initial_reference: list[torch.Tensor]):
    z1, a1, z2, a2, logits = engine.forward_full()
    effective = z2 + torch.where(engine.K, engine.delta[:, None, :], torch.zeros_like(engine.delta[:, None, :]))
    gate = base.gate(effective, torch.ones_like(engine.F))
    ce = (logits.logsumexp(-1) - (logits * engine.cy).sum(-1)).mean(-1)
    acc = (logits.argmax(-1) == engine.cy.argmax(-1)).float().mean(-1)
    rows = [dict(seed=m["seed"], branch=m["branch"], task=task, step=step, ce=float(ce[j]), acc=float(acc[j])) for j, m in enumerate(engine.models)]

    w1, b1, w2, b2, w3, _ = engine.p
    r1, _, r2, _, r3, _ = initial_reference
    w1c, r1c = w1 - w1.mean(-1, keepdim=True), r1 - r1.mean(-1, keepdim=True)
    w2c, r2c = w2 - w2.mean(-1, keepdim=True), r2 - r2.mean(-1, keepdim=True)
    metrics = {
        "z2_mean": z2.mean(1), "z2_std": z2.std(1, unbiased=False),
        "effective_argument_mean": effective.mean(1), "effective_argument_std": effective.std(1, unbiased=False),
        "response_gate_mean": gate.mean(1), "q_lowgate": (gate < .05).float().mean(1),
        "actual_activation_mean": a2.mean(1), "actual_activation_std": a2.std(1, unbiased=False),
        "actual_activation_varying_fraction": (a2 != a2[:, :1, :]).float().mean(1),
        "W1_row_norm": w1.norm(dim=-1), "W1_centered_row_norm": w1c.norm(dim=-1),
        "W1_row_source_cosine": cosine_rows(w1, r1), "W1_centered_row_source_cosine": cosine_rows(w1c, r1c),
        "W2_row_norm": w2.norm(dim=-1), "W2_centered_row_norm": w2c.norm(dim=-1),
        "W2_row_source_cosine": cosine_rows(w2, r2), "W2_centered_row_source_cosine": cosine_rows(w2c, r2c),
        "W2_column_norm": w2.norm(dim=1), "W3_row_norm": w3.norm(dim=-1), "W3_column_norm": w3.norm(dim=1),
        "W3_column_source_cosine": cosine_rows(w3.transpose(1, 2), r3.transpose(1, 2)),
        "b1": b1, "b2": b2,
    }
    current = {"z2": z2.clone(), "a2": a2.clone(), "logits": logits.clone(), "p": [q.clone() for q in engine.p]}
    if previous is None:
        metrics.update({
            "interval_dW1_row_norm": torch.zeros_like(metrics["W1_row_norm"]),
            "interval_db1": torch.zeros_like(b1),
            "interval_dW2_row_norm": torch.zeros_like(metrics["W2_row_norm"]),
            "interval_db2": torch.zeros_like(b2),
            "interval_dW3_row_norm": torch.zeros_like(metrics["W3_row_norm"]),
            "interval_dW3_column_norm": torch.zeros_like(metrics["W3_column_norm"]),
            "interval_delta_z2_mean": torch.zeros_like(metrics["z2_mean"]),
            "interval_delta_z2_std": torch.zeros_like(metrics["z2_std"]),
            "interval_delta_actual_activation_mean": torch.zeros_like(metrics["actual_activation_mean"]),
            "interval_delta_actual_activation_std": torch.zeros_like(metrics["actual_activation_std"]),
            "interval_delta_logits_mean": torch.zeros_like(logits.mean(1)),
            "interval_delta_logits_std": torch.zeros_like(logits.std(1, unbiased=False)),
        })
    else:
        dp = [a - b for a, b in zip(engine.p, previous["p"])]
        dz, da, dl = z2 - previous["z2"], a2 - previous["a2"], logits - previous["logits"]
        metrics.update({
            "interval_dW1_row_norm": dp[0].norm(dim=-1), "interval_db1": dp[1],
            "interval_dW2_row_norm": dp[2].norm(dim=-1), "interval_db2": dp[3],
            "interval_dW3_row_norm": dp[4].norm(dim=-1), "interval_dW3_column_norm": dp[4].norm(dim=1),
            "interval_delta_z2_mean": dz.mean(1), "interval_delta_z2_std": dz.std(1, unbiased=False),
            "interval_delta_actual_activation_mean": da.mean(1), "interval_delta_actual_activation_std": da.std(1, unbiased=False),
            "interval_delta_logits_mean": dl.mean(1), "interval_delta_logits_std": dl.std(1, unbiased=False),
        })
    return rows, {k: v.cpu().numpy() for k, v in metrics.items()}, current


@torch.no_grad()
def save_replay(engine, replay_dir: Path, task: int, step: int, order: torch.Tensor) -> tuple[Path, str]:
    batch_position = step if step < base.STEPS else base.STEPS - 1
    role = "next_scheduled_batch" if step < base.STEPS else "retrospective_final_batch"
    indices = order[batch_position]
    x = engine.cx[engine.mid, indices]
    y = engine.cy[engine.mid, indices]
    anchor0 = engine.anchor_base[engine.mid, indices]
    anchor1 = engine.anchor_shift[engine.mid, indices]
    before = engine.snapshot()
    diagnostic = engine.diagnose_step(x, y, anchor0, anchor1)
    after = engine.snapshot()
    if not snapshots_identical(before, after):
        raise RuntimeError(f"diagnostic mutated state at task {task} step {step}")
    state = {
        "parameters": [q.cpu() for q in engine.p], "adam_m": [q.cpu() for q in engine.m],
        "adam_v": [q.cpu() for q in engine.v], "t": engine.t.cpu(),
        "task_accumulator_ce": engine.ce.cpu(), "task_accumulator_acc": engine.acc.cpu(),
        "online_pos": engine.online_pos.cpu(),
    }
    diag_cpu = {k: ([q.cpu() for q in v] if isinstance(v, list) else v.cpu()) for k, v in diagnostic.items()}
    artifact = {
        "schema_version": 1, "task": task, "step": step, "parameter_names": PARAMETER_NAMES,
        "diagnostic_batch_role": role, "diagnostic_batch_position_zero_based": batch_position,
        "diagnostic_batch_indices": indices.cpu(), "state": state, "diagnostic": diag_cpu,
    }
    if not tensors_finite(artifact):
        raise RuntimeError(f"nonfinite replay state or diagnostic at task {task} step {step}")
    path = replay_dir / f"task{task}_step{step:04d}.pt"
    torch_x(path, artifact)
    return path, sha(path)


def output_absence_gate() -> None:
    names = ("run_config.json", "rng_hashes.json", "anchor_cache.pt", "rows.csv", "learning.npz", "units.npz", "checkpoint.pt", "replay_manifest.json", "validation.json", "provenance.json", "failure.json")
    existing = [str(OUT / name) for name in names if (OUT / name).exists()]
    if (OUT / "replay").exists():
        existing.append(str(OUT / "replay"))
    if existing:
        raise FileExistsError("refuse overwrite existing substantive output: " + ", ".join(existing))


def prepare(commit: str, require_pushed: bool):
    configure()
    OUT.mkdir(parents=True, exist_ok=True)
    registration = prereg_gate(commit, require_pushed=require_pushed)
    engine_qa = engine_qa_gate()
    loaded = load_source()
    checkpoint, x, y, subset, selections, source_model_indices, hashes, persistence, data_hashes = loaded
    task_schedules, final_generators = schedules(checkpoint)
    initialized = initialize_engine(checkpoint, x, subset, selections, source_model_indices)
    engine, raw_x, z20, delta, source_z2_seed, delta_seed, initial_reference, state_equal, delta_error, source_parameters, source_m, source_v = initialized
    pairs = initial_pair_checks(engine, task_schedules[0])
    return dict(registration=registration, engine_qa=engine_qa, checkpoint=checkpoint, x=x, y=y, subset=subset,
                selections=selections, source_model_indices=source_model_indices, source_hashes=hashes, persistence=persistence,
                schedules=task_schedules, final_generators=final_generators, engine=engine, raw_x=raw_x, z20=z20, delta=delta,
                source_z2_seed=source_z2_seed, delta_seed=delta_seed, initial_reference=initial_reference,
                source_state_identical=state_equal, delta_formula_error=delta_error, source_parameters=source_parameters,
                source_m=source_m, source_v=source_v, initial_pair_checks=pairs, data_hashes=data_hashes)


def validate(commit: str) -> None:
    if (OUT / "runner_validation.json").exists():
        raise FileExistsError("refuse overwrite runner_validation.json")
    context = prepare(commit, require_pushed=True)
    engine = context["engine"]
    schedule = context["schedules"][0]
    order = torch.stack([schedule["orders"][m["seed"]] for m in engine.models], dim=1).to(engine.device)
    labels = schedule["labels"]
    for j, model in enumerate(engine.models):
        engine.cy[j].copy_(torch.nn.functional.one_hot(labels[model["seed"]].to(engine.device), 10).float())
    indices = order[0]
    before = engine.snapshot()
    diag = engine.diagnose_step(engine.cx[engine.mid, indices], engine.cy[engine.mid, indices], engine.anchor_base[engine.mid, indices], engine.anchor_shift[engine.mid, indices])
    diagnostic_nonmutating = snapshots_identical(before, engine.snapshot())
    diagnostic_shapes = all(len(diag[k]) == 6 for k in ("gradients", "current_gradient_numerator", "history_numerator", "denominator", "predicted_update", "parameter_delta"))
    checks = {
        "status": "PASS", "runner_sha256": sha(Path(__file__)), "engine_sha256": sha(ENGINE),
        "base_sha256": sha(Path(base.__file__)), "source_helper_sha256": sha(Path(source_helpers.__file__)),
        "engine_validation_sha256": sha(OUT / "engine_validation.json"), "source_sha256": context["source_hashes"],
        **context["registration"], "model_count": len(engine.models), "tasks": list(TASKS),
        "probe_steps": list(PROBE_STEPS), "state_steps": list(STATE_STEPS), "source_t": int(engine.t),
        "source_state_bitwise_identical": context["source_state_identical"],
        "past_only_selection_verified": context["persistence"], "delta_formula_maxabs": context["delta_formula_error"],
        "continued_rng_hashes": [q["hashes"] for q in context["schedules"]],
        "initial_pair_checks": context["initial_pair_checks"], "diagnostic_nonmutating": diagnostic_nonmutating,
        "diagnostic_six_parameter_arrays": diagnostic_shapes,
        "schedule_geometry": {"orders_per_task_seed": [base.STEPS, base.BATCH], "branches_share_seed_schedule": True,
                              "first_75_updates": "eager", "later_updates": f"CUDA graph blocks of {base.BLOCK}"},
    }
    if not diagnostic_nonmutating or not diagnostic_shapes or checks["model_count"] != 18 or checks["source_t"] != 120000:
        raise RuntimeError(f"runner validation failed: {checks}")
    json_x(OUT / "runner_validation.json", checks)
    print(json.dumps(checks, indent=2), flush=True)


def run(commit: str) -> None:
    output_absence_gate()
    context = prepare(commit, require_pushed=True)
    runner_qa = runner_qa_gate()
    engine = context["engine"]
    replay_dir = OUT / "replay"
    replay_dir.mkdir()
    anchor_before = anchor_hashes(engine)
    model_seeds = np.asarray([m["seed"] for m in engine.models], dtype=np.int64)
    model_branches = np.asarray([m["branch"] for m in engine.models], dtype="<U2")
    selected_ids = np.asarray([context["selections"][m["seed"]]["target20"] for m in engine.models], dtype=np.int64)
    selected_mask = np.zeros((len(engine.models), 100), dtype=np.bool_)
    for j in range(len(engine.models)):
        selected_mask[j, selected_ids[j]] = True
    run_config = {
        "status": "READY", **context["registration"], "runner_sha256": sha(Path(__file__)),
        "engine_sha256": sha(ENGINE), "base_sha256": sha(Path(base.__file__)),
        "source_helper_sha256": sha(Path(source_helpers.__file__)), "engine_validation_sha256": sha(OUT / "engine_validation.json"),
        "runner_validation_sha256": sha(OUT / "runner_validation.json"), "source_sha256": context["source_hashes"],
        "source_paths": {k: str(v) for k, v in SOURCE_PATHS.items()}, "tasks": list(TASKS),
        "probe_steps": list(PROBE_STEPS), "state_steps": list(STATE_STEPS), "models": engine.models,
        "optimizer": {"name": "Adam", "lr": .001, "betas": [.9, .999], "eps": 1e-8, "initial_t": 120000},
        "training": {"batch": base.BATCH, "updates_per_task": base.STEPS, "eager_through_step": 75,
                     "graph_block": base.BLOCK, "common_schedule_within_seed": True},
        "device": torch.cuda.get_device_name(), "torch_version": torch.__version__, "deterministic_algorithms": True,
        "tf32": False, "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    json_x(OUT / "run_config.json", run_config)
    json_x(OUT / "rng_hashes.json", [q["hashes"] for q in context["schedules"]])
    anchor_cache = {
        "schema_version": 1, "models": engine.models, "source_model_indices": context["source_model_indices"],
        "selected_ids": torch.from_numpy(selected_ids), "delta": engine.delta.cpu(), "source_z2": engine.anchor_z2.cpu(),
        "source_z2_single_model_by_seed": context["source_z2_seed"].cpu(),
        "anchor_base": engine.anchor_base.cpu(), "anchor_shift": engine.anchor_shift.cpu(),
        "source_subset": {k: torch.as_tensor(v) for k, v in context["subset"].items()},
        "source_parameters_by_seed": context["source_parameters"], "source_adam_m_by_seed": context["source_m"],
        "source_adam_v_by_seed": context["source_v"], "source_t": context["checkpoint"]["t"],
        "source_sha256": context["source_hashes"], "anchor_sha256": anchor_before,
    }
    torch_x(OUT / "anchor_cache.pt", anchor_cache)

    engine.capture()
    online_ce = np.empty((len(TASKS), len(engine.models), base.STEPS), dtype=np.float32)
    online_acc = np.empty_like(online_ce)
    rows: list[dict] = []
    unit_lists: dict[str, list[np.ndarray]] = {}
    replay_hashes = {}
    freeze_maxabs = 0.0
    eager_last_step_ce_maxabs = 0.0
    eager_last_step_acc_maxabs = 0.0
    for task_index, schedule in enumerate(context["schedules"]):
        task = schedule["task"]
        order = torch.stack([schedule["orders"][m["seed"]] for m in engine.models], dim=1).to(engine.device)
        for j, model in enumerate(engine.models):
            engine.cx[j].copy_(context["raw_x"][model["seed"]])
            engine.cy[j].copy_(torch.nn.functional.one_hot(schedule["labels"][model["seed"]].to(engine.device), 10).float())
        engine.acc.zero_()
        engine.ce.zero_()
        engine.reset_online()
        last = 0
        previous = None
        for step in PROBE_STEPS:
            if step <= 75:
                for position in range(last, step):
                    indices = order[position]
                    engine.step(engine.cx[engine.mid, indices], engine.cy[engine.mid, indices],
                                engine.anchor_base[engine.mid, indices], engine.anchor_shift[engine.mid, indices])
            else:
                if last % base.BLOCK or step % base.BLOCK:
                    raise RuntimeError("post-75 probe grid must align to graph blocks")
                for position in range(last, step, base.BLOCK):
                    eager_reference = None
                    if position == base.STEPS - base.BLOCK:
                        snapshot = engine.snapshot()
                        for eager_position in range(position, base.STEPS - 1):
                            eager_indices = order[eager_position]
                            engine.step(engine.cx[engine.mid, eager_indices], engine.cy[engine.mid, eager_indices],
                                        engine.anchor_base[engine.mid, eager_indices], engine.anchor_shift[engine.mid, eager_indices])
                        eager_indices = order[base.STEPS - 1]
                        _, reference_ce, reference_acc = engine.gradients(
                            engine.cx[engine.mid, eager_indices], engine.cy[engine.mid, eager_indices],
                            engine.anchor_base[engine.mid, eager_indices], engine.anchor_shift[engine.mid, eager_indices])
                        eager_reference = (reference_ce.clone(), reference_acc.clone())
                        engine.restore(snapshot)
                    engine.indices.copy_(order[position:position + base.BLOCK])
                    engine.graph.replay()
                    if eager_reference is not None:
                        torch.cuda.synchronize()
                        eager_last_step_ce_maxabs = max(eager_last_step_ce_maxabs, float((engine.online_ce[:, -1] - eager_reference[0]).abs().max()))
                        eager_last_step_acc_maxabs = max(eager_last_step_acc_maxabs, float((engine.online_acc[:, -1] - eager_reference[1]).abs().max()))
                torch.cuda.synchronize()
            probe_rows, metrics, previous = measure(engine, task, step, previous, context["initial_reference"])
            rows.extend(probe_rows)
            for key, value in metrics.items():
                unit_lists.setdefault(key, []).append(value)
            if step in STATE_STEPS:
                replay_path, replay_hash = save_replay(engine, replay_dir, task, step, order)
                replay_hashes[replay_path.name] = replay_hash
            frozen = engine.upstream_frozen
            freeze_maxabs = max(freeze_maxabs, max(float((engine.p[k][frozen] - engine.frozen_reference[k][frozen]).abs().max()) for k in (0, 1)))
            last = step
        if int(engine.online_pos) != base.STEPS:
            raise RuntimeError(f"online buffer position mismatch at task {task}: {int(engine.online_pos)}")
        online_ce[task_index] = engine.online_ce.cpu().numpy()
        online_acc[task_index] = engine.online_acc.cpu().numpy()
        if not np.isfinite(online_ce[task_index]).all() or not np.isfinite(online_acc[task_index]).all():
            raise RuntimeError(f"nonfinite direct online buffer at task {task}")
        print(f"TASK {task} COMPLETE", flush=True)

    expected_rows = len(TASKS) * len(engine.models) * len(PROBE_STEPS)
    row_keys = {(int(r["seed"]), r["branch"], int(r["task"]), int(r["step"])) for r in rows}
    if len(rows) != expected_rows or len(row_keys) != expected_rows:
        raise RuntimeError(f"probe grid mismatch rows={len(rows)} unique={len(row_keys)} expected={expected_rows}")
    if int(engine.t) != 150000:
        raise RuntimeError(f"final Adam t {int(engine.t)} != 150000")
    if freeze_maxabs != 0:
        raise RuntimeError(f"AF/BF layer1 freeze violation {freeze_maxabs}")
    anchor_after = anchor_hashes(engine)
    if anchor_after != anchor_before:
        raise RuntimeError("cached anchors or delta changed during training")

    csv_x(OUT / "rows.csv", rows)
    npz_x(OUT / "learning.npz", online_ce=online_ce, online_acc=online_acc,
          tasks=np.asarray(TASKS, dtype=np.int64), model_seed=model_seeds, model_branch=model_branches)
    unit_arrays = {key: np.stack(values).reshape(len(TASKS), len(PROBE_STEPS), *values[0].shape) for key, values in unit_lists.items()}
    if not all(np.isfinite(value).all() for value in unit_arrays.values()):
        raise RuntimeError("nonfinite all-unit mechanism array")
    target_keys = (
        "z2_mean", "z2_std", "effective_argument_mean", "effective_argument_std", "response_gate_mean", "q_lowgate",
        "actual_activation_mean", "actual_activation_std", "actual_activation_varying_fraction", "W2_row_norm",
        "W2_centered_row_norm", "W2_row_source_cosine", "W2_centered_row_source_cosine", "W3_column_norm",
        "W3_column_source_cosine", "b2", "interval_dW2_row_norm", "interval_db2", "interval_dW3_column_norm",
        "interval_delta_z2_mean", "interval_delta_z2_std", "interval_delta_actual_activation_mean",
        "interval_delta_actual_activation_std",
    )
    gather_ids = selected_ids[None, None, :, :]
    for key in target_keys:
        unit_arrays[f"target_{key}"] = np.take_along_axis(unit_arrays[key], gather_ids, axis=-1)
    npz_x(OUT / "units.npz", tasks=np.asarray(TASKS, dtype=np.int64), steps=np.asarray(PROBE_STEPS, dtype=np.int64),
          model_seed=model_seeds, model_branch=model_branches, selected_unit_id=selected_ids, selected_mask=selected_mask,
          interval_defined=(np.asarray(PROBE_STEPS) != 0), **unit_arrays)
    checkpoint_artifact = {
        "schema_version": 1, "models": engine.models, "task": 25, "step": 6000,
        "parameters": [q.cpu() for q in engine.p], "adam_m": [q.cpu() for q in engine.m],
        "adam_v": [q.cpu() for q in engine.v], "t": engine.t.cpu(),
        "rng_states": source_helpers.states(context["final_generators"]), "anchor_sha256": anchor_after,
    }
    torch_x(OUT / "checkpoint.pt", checkpoint_artifact)
    replay_manifest = {
        "schema_version": 1, "status": "COMPLETE", "parameter_names": list(PARAMETER_NAMES),
        "state_keys": ["parameters", "adam_m", "adam_v", "t", "task_accumulator_ce", "task_accumulator_acc", "online_pos"],
        "diagnostic_list_keys": ["gradients", "current_gradient_numerator", "history_numerator", "decomposition_error",
                                 "denominator", "predicted_update", "parameter_delta", "moment_m_delta", "moment_v_delta"],
        "diagnostic_scalar_keys": ["ce", "acc", "next_t"],
        "files": replay_hashes, "expected_files": len(TASKS) * len(STATE_STEPS),
        "batch_semantics": "step<6000 uses zero-based order[step], the next update; step6000 uses order[5999] retrospectively",
    }
    json_x(OUT / "replay_manifest.json", replay_manifest)

    finite_parameters = all(bool(torch.isfinite(q).all()) for q in engine.p + engine.m + engine.v)
    runtime = {
        "status": "PASS", "row_count": len(rows), "row_grid_unique": len(row_keys) == expected_rows,
        "online_shape": list(online_ce.shape), "online_finite": bool(np.isfinite(online_ce).all() and np.isfinite(online_acc).all()),
        "online_accuracy_bounds": bool(((online_acc >= 0) & (online_acc <= 1)).all()),
        "fullprobe_finite": all(np.isfinite(float(r[k])) for r in rows for k in ("ce", "acc")),
        "final_t": int(engine.t), "parameters_moments_finite": finite_parameters,
        "source_state_bitwise_identical": context["source_state_identical"], "past_only_selection_verified": True,
        "delta_formula_maxabs": context["delta_formula_error"], "source_initial_pair_checks": context["initial_pair_checks"],
        "rng_hashes_match": True, "anchor_immutable": anchor_after == anchor_before, "anchor_sha256": anchor_after,
        "freeze_measurement_and_final_maxabs": freeze_maxabs, "replay_file_count": len(replay_hashes),
        "replay_states_and_diagnostics_finite": True, "all_unit_arrays_finite": True,
        "eager_reference_last_step_ce_maxabs": eager_last_step_ce_maxabs,
        "eager_reference_last_step_acc_maxabs": eager_last_step_acc_maxabs,
        "eager_first75_then_graph25": True, "engine_validation_current": context["engine_qa"].get("engine_sha256") == sha(ENGINE),
        "runner_validation_current": runner_qa.get("runner_sha256") == sha(Path(__file__)),
        "spec_ancestor_and_pushed_launch": True,
    }
    if not (runtime["online_finite"] and runtime["online_accuracy_bounds"] and runtime["fullprobe_finite"] and
            finite_parameters and runtime["anchor_immutable"] and len(replay_hashes) == 25 and
            eager_last_step_ce_maxabs <= 5e-5 and eager_last_step_acc_maxabs == 0):
        raise RuntimeError(f"runtime acceptance failure: {runtime}")
    artifact_names = ("run_config.json", "rng_hashes.json", "anchor_cache.pt", "rows.csv", "learning.npz", "units.npz", "checkpoint.pt", "replay_manifest.json")
    result_hashes = {name: sha(OUT / name) for name in artifact_names}
    validation = {**runtime, "source_sha256": context["source_hashes"], "result_sha256": result_hashes,
                  "engine_validation_sha256": sha(OUT / "engine_validation.json"),
                  "runner_validation_sha256": sha(OUT / "runner_validation.json")}
    json_x(OUT / "validation.json", validation)
    provenance = {
        "status": "COMPLETE", **context["registration"], "runner_sha256": sha(Path(__file__)),
        "engine_sha256": sha(ENGINE), "base_sha256": sha(Path(base.__file__)),
        "source_helper_sha256": sha(Path(source_helpers.__file__)), "source_sha256": context["source_hashes"],
        "data_sha256": context["data_hashes"],
        "subset_sha256": {str(seed): array_sha(context["subset"][seed]) for seed in range(3)},
        "anchor_sha256": anchor_after, "result_sha256": {**result_hashes, "validation.json": sha(OUT / "validation.json")},
        "rng_hashes": [q["hashes"] for q in context["schedules"]], "models": engine.models,
        "tasks": list(TASKS), "probe_steps": list(PROBE_STEPS), "state_steps": list(STATE_STEPS),
        "final_t": int(engine.t), "torch_version": torch.__version__, "device": torch.cuda.get_device_name(),
        "tf32": False, "deterministic_algorithms": True,
    }
    json_x(OUT / "provenance.json", provenance)
    print(json.dumps({"status": "COMPLETE", "rows": len(rows), "online_shape": list(online_ce.shape), "final_t": int(engine.t)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--prereg", required=True)
    args = parser.parse_args()
    try:
        validate(args.prereg) if args.validate else run(args.prereg)
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        failure = OUT / ("runner_validation_failure.json" if args.validate else "failure.json")
        if not failure.exists():
            json_x(failure, {"status": "TECHNICAL_FAILURE", "mode": "validate" if args.validate else "run",
                             "exception": repr(exc), "traceback": traceback.format_exc(),
                             "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "runner_sha256": sha(Path(__file__))})
        raise


if __name__ == "__main__":
    main()

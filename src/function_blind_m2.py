"""Deterministic read-only replay for M2 mean-opening instrumentation.

Registered invocation::

  OMP_NUM_THREADS=1 .venv/bin/python -m src.function_blind_m2 \
    --config configs/function_blind_m2_0828.yaml \
    --outdir results/function_blind_m2_0828

The replay must reproduce both the original final mutable-state hashes and all
previously stored exact landmark measurements before the new ``S`` values are
accepted.  The original confirmation directory is never modified.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch
import yaml

from .common import ROOT, load_config, pick_device, resolve_outdir
from .function_blind_direct import (
    RUN_KEYS,
    SUPPORT_SIZE,
    UNIT_KEYS,
    _single_registered_group,
    complete_state_hashes,
    deterministic_examples,
    exact_record,
    landmark_grid,
    validate_registered_requirements,
)
from .train import train_group


SPEC = "specs/spec_function_blind_m2_0828.md"


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class M2Recorder:
    """Record mean ReLU activation and exact replay anchors at landmarks."""

    def __init__(self, steps: np.ndarray, landmark: np.ndarray, phase: np.ndarray,
                 R: int, h: int, *, n_brute: int = 20):
        self.steps = np.asarray(steps, dtype=np.int64)
        self.landmark = np.asarray(landmark, dtype=np.int64)
        self.phase = np.asarray(phase, dtype="U2")
        if not (len(self.steps) == len(self.landmark) == len(self.phase)):
            raise ValueError("landmark metadata length mismatch")
        if len(np.unique(self.steps)) != len(self.steps):
            raise ValueError("probe steps must be unique")
        self.index = {int(step): index for index, step in enumerate(self.steps)}
        shape = (len(self.steps), int(R), int(h))
        self.S = np.empty(shape, dtype=np.float64)
        self.unit = {key: np.empty(shape, dtype=np.float64) for key in UNIT_KEYS}
        self.run = {
            key: np.empty((len(self.steps), int(R)), dtype=np.float64)
            for key in RUN_KEYS
        }
        self.filled = np.zeros(len(self.steps), dtype=bool)
        self.examples = deterministic_examples(self.steps, int(R), int(h), n_brute)
        self.brute_rows: list[dict[str, Any]] = []
        self.nonfinite = np.zeros(len(self.steps), dtype=np.int64)
        self.negative = np.zeros(len(self.steps), dtype=np.int64)
        self.strict_mismatch = np.zeros(len(self.steps), dtype=np.int64)
        self.support_size = np.zeros(len(self.steps), dtype=np.int64)

    def __call__(self, st, step: int) -> None:
        index = self.index.get(int(step))
        if index is None:
            return
        record, sanity, context = exact_record(st, with_context=True)
        activation = context["activation"]
        S_tensor = activation.mean(dim=0)
        S = S_tensor.detach().cpu().numpy().astype(np.float64, copy=False)
        self.S[index] = S
        for key in UNIT_KEYS:
            self.unit[key][index] = record[key]
        for key in RUN_KEYS:
            self.run[key][index] = record[key]

        self.nonfinite[index] = int(np.count_nonzero(~np.isfinite(S)))
        self.negative[index] = int(np.count_nonzero(S < 0.0))
        self.strict_mismatch[index] = int(
            np.count_nonzero((S == 0.0) != (record["p_hat"] == 0.0))
        )
        self.support_size[index] = int(sanity["support_size"])

        for run, unit in self.examples.get(int(step), ()):
            values = activation[:, int(run), int(unit)].detach().cpu().numpy()
            scalar = float(sum(float(value) for value in values) / SUPPORT_SIZE)
            vector = float(S[int(run), int(unit)])
            self.brute_rows.append(dict(
                step=int(step), run=int(run), unit=int(unit),
                vector=vector, scalar=scalar, abs_error=abs(vector - scalar),
            ))
        self.filled[index] = True

    def check_complete(self) -> None:
        missing = np.flatnonzero(~self.filled)
        if missing.size:
            raise RuntimeError(
                f"missing {missing.size} M2 probe steps: "
                f"{self.steps[missing][:10].tolist()}"
            )

    def sanity(self, *, brute_force_atol: float, expected_examples: int) -> dict[str, Any]:
        max_error = max((row["abs_error"] for row in self.brute_rows), default=np.inf)
        checks = dict(
            complete=bool(self.filled.all()),
            support_size=bool((self.support_size == SUPPORT_SIZE).all()),
            finite=bool(self.nonfinite.sum() == 0),
            nonnegative=bool(self.negative.sum() == 0),
            strict_off_identity=bool(self.strict_mismatch.sum() == 0),
            scalar_examples=bool(
                len(self.brute_rows) == int(expected_examples)
                and max_error < float(brute_force_atol)
            ),
        )
        return dict(
            pass_=bool(all(checks.values())), checks=checks,
            n_nonfinite=int(self.nonfinite.sum()),
            n_negative=int(self.negative.sum()),
            strict_off_mismatches=int(self.strict_mismatch.sum()),
            support_sizes=sorted(set(int(value) for value in self.support_size)),
            scalar_examples_n=int(len(self.brute_rows)),
            scalar_examples_expected=int(expected_examples),
            scalar_max_abs_error=float(max_error),
            scalar_atol=float(brute_force_atol),
            scalar_examples=self.brute_rows,
        )


def validate_reference_inputs(reference_dir: Path,
                              m2_cfg: dict[str, Any]) -> dict[str, Any]:
    """Verify the immutable work-6 inputs before starting the replay."""
    exposures = reference_dir / "exposures.csv"
    instrumentation_meta = reference_dir / "instrumentation_meta.json"
    analysis_meta = reference_dir / "meta.json"
    for path in (exposures, instrumentation_meta, analysis_meta):
        if not path.exists():
            raise FileNotFoundError(path)
    exposure_hash = sha256(exposures)
    instrument_hash = sha256(instrumentation_meta)
    expected_exposure = str(m2_cfg["reference_exposures_sha256"])
    expected_instrument = str(m2_cfg["reference_instrumentation_meta_sha256"])
    analysis = json.loads(analysis_meta.read_text())
    expected_npz = analysis.get("input_sha256") or {}
    logdir = reference_dir / "logs"
    found = sorted(logdir.glob("seed*.npz"), key=lambda path: path.name)
    npz_hashes = {path.name: sha256(path) for path in found}
    checks = dict(
        exposures=exposure_hash == expected_exposure,
        instrumentation_meta=instrument_hash == expected_instrument,
        npz_count=len(found) == 20,
        npz_hashes=npz_hashes == expected_npz,
    )
    return dict(
        pass_=bool(all(checks.values())), checks=checks,
        exposures_sha256=exposure_hash,
        instrumentation_meta_sha256=instrument_hash,
        npz_sha256=npz_hashes,
        expected_npz_sha256=expected_npz,
        instrumentation_meta=json.loads(instrumentation_meta.read_text()),
    )


def compare_replay_anchors(recorder: M2Recorder, runs: list[dict[str, Any]],
                           reference_dir: Path) -> dict[str, Any]:
    """Require exact equality for every previously saved float64 landmark."""
    rows: list[dict[str, Any]] = []
    all_pass = True
    for run_index, run in enumerate(runs):
        seed = int(run["seed"])
        path = reference_dir / "logs" / f"seed{seed}.npz"
        with np.load(path, allow_pickle=False) as z:
            step_equal = np.array_equal(np.asarray(z["step"]), recorder.steps)
            key_equal: dict[str, bool] = {}
            max_abs: dict[str, float] = {}
            for key in UNIT_KEYS:
                old = np.asarray(z[key])
                new = recorder.unit[key][:, run_index]
                key_equal[key] = bool(np.array_equal(old, new, equal_nan=False))
                max_abs[key] = float(np.max(np.abs(old - new), initial=0.0))
            for key in RUN_KEYS:
                old = np.asarray(z[key])
                new = recorder.run[key][:, run_index]
                key_equal[key] = bool(np.array_equal(old, new, equal_nan=False))
                max_abs[key] = float(np.max(np.abs(old - new), initial=0.0))
        passed = bool(step_equal and all(key_equal.values()))
        all_pass = all_pass and passed
        rows.append(dict(seed=seed, pass_=passed, step_equal=bool(step_equal),
                         key_equal=key_equal, max_abs_error=max_abs))
    return dict(pass_=bool(all_pass), per_seed=rows)


def write_s_logs(recorder: M2Recorder, runs: list[dict[str, Any]],
                 outdir: Path, cfg: dict[str, Any]) -> list[str]:
    logdir = outdir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for run_index, run in enumerate(runs):
        path = logdir / f"seed{int(run['seed'])}.npz"
        np.savez_compressed(
            path,
            step=recorder.steps,
            landmark_B=recorder.landmark,
            phase=recorder.phase,
            seed=np.int64(run["seed"]),
            run_id=np.asarray(run["run_id"]),
            condition=np.asarray("condA"),
            encoding=np.asarray(run["enc"]),
            batch=np.asarray(str(run["batch"])),
            width=np.int64(run["width"]),
            period=np.int64(run["period"]),
            lr=np.float64(run["lr"]),
            generator_offset=np.int64(cfg["common"].get("generator_offset", 0)),
            total_steps=np.int64(cfg["common"]["total_steps"]),
            support_size=np.int64(SUPPORT_SIZE),
            spec=np.asarray(SPEC),
            S=recorder.S[:, run_index],
        )
        paths.append(str(path))
    return paths


def apply_smoke(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg["common"].update(total_steps=10_000, seeds=[0], checkpoints=[])
    cfg["function_blind_direct"].update(
        boundary_start=0, boundary_stop=0, boundary_every=10_000,
        require_s2=False, s2_steps=0,
    )
    cfg["function_blind_m2"]["brute_force_examples"] = 2
    return cfg


def run(cfg: dict[str, Any], device: str, outdir: Path, *, smoke: bool = False) -> dict[str, Any]:
    if os.environ.get("OMP_NUM_THREADS") != "1" or torch.get_num_threads() != 1:
        raise RuntimeError("OMP_NUM_THREADS=1 is required by the registered M2 replay")
    if not smoke and str(cfg.get("spec")) != SPEC:
        raise ValueError(f"registered M2 requires spec={SPEC!r}")
    direct_cfg = cfg["function_blind_direct"]
    validate_registered_requirements(
        cfg, s2_steps=int(direct_cfg.get("s2_steps", 0)),
        reference_logs=None, smoke=smoke,
    )
    if not smoke and device != "cpu":
        raise ValueError("registered M2 replay requires runtime device='cpu'")

    m2_cfg = cfg["function_blind_m2"]
    reference_dir = Path(str(m2_cfg["reference_dir"]))
    if not reference_dir.is_absolute():
        reference_dir = Path(ROOT) / reference_dir
    reference = None if smoke else validate_reference_inputs(reference_dir, m2_cfg)
    if reference is not None and not reference["pass_"]:
        raise RuntimeError("M2 reference input hashes do not match registration")

    gkey, runs = _single_registered_group(cfg)
    steps, landmark, phase = landmark_grid(
        int(cfg["common"]["total_steps"]), direct_cfg
    )
    n_examples = int(m2_cfg.get("brute_force_examples", 20))
    recorder = M2Recorder(
        steps, landmark, phase, len(runs), int(gkey[1]), n_brute=n_examples
    )
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    state, train_seconds = train_group(
        gkey, runs, cfg, device, str(outdir / "_training_logs"),
        total_steps=int(cfg["common"]["total_steps"]), ckpts=[],
        probe=recorder, probe_steps=steps,
        gname="function_blind_m2_replay",
    )
    recorder.check_complete()
    probe_sanity = recorder.sanity(
        brute_force_atol=float(m2_cfg.get("brute_force_atol", 1e-12)),
        expected_examples=n_examples,
    )
    final_hashes = complete_state_hashes(state)

    if smoke:
        state_match = dict(pass_=True, smoke_bypass=True, differences=[])
        anchor_match = dict(pass_=True, smoke_bypass=True, per_seed=[])
    else:
        expected_hashes = reference["instrumentation_meta"]["final_state_hashes"]
        all_keys = sorted(set(expected_hashes) | set(final_hashes))
        differences = [
            key for key in all_keys
            if expected_hashes.get(key) != final_hashes.get(key)
        ]
        state_match = dict(
            pass_=not differences, differences=differences,
            expected=expected_hashes, actual=final_hashes,
        )
        anchor_match = compare_replay_anchors(recorder, runs, reference_dir)

    all_pass = bool(
        probe_sanity["pass_"] and state_match["pass_"]
        and anchor_match["pass_"] and (smoke or reference["pass_"])
    )
    paths: list[str] = []
    if all_pass:
        paths = write_s_logs(recorder, runs, outdir, cfg)
    meta = dict(
        spec=SPEC, preregistration_commit="1096be3",
        implementation_git=git_hash(), smoke=bool(smoke), device=device,
        total_steps=int(cfg["common"]["total_steps"]),
        seeds=[int(run["seed"]) for run in runs], R=len(runs), width=int(gkey[1]),
        generator_offset=int(cfg["common"].get("generator_offset", 0)),
        n_records=int(len(steps)), n_npz=len(paths), npz_paths=paths,
        train_seconds=round(float(train_seconds), 3),
        elapsed_seconds=round(time.time() - started, 3),
        final_state_hashes=final_hashes,
        sanity=dict(
            reference_inputs=reference,
            S_probe=probe_sanity,
            final_state_match=state_match,
            anchor_match=anchor_match,
            all_required_pass=all_pass,
        ),
    )
    (outdir / "replay_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(f"M2 replay records={len(steps)} R={len(runs)} "
          f"sanity={'PASS' if all_pass else 'FAIL'}", flush=True)
    if not all_pass:
        raise RuntimeError(f"M2 replay sanity failed; inspect {outdir / 'replay_meta.json'}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/function_blind_m2_0828.yaml")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke(cfg)
    if args.device is not None:
        cfg["common"]["device"] = args.device
    device = pick_device(cfg)
    outdir = Path(resolve_outdir(args.config, smoke=args.smoke, outdir=args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config_used.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
    )
    run(cfg, device, outdir, smoke=args.smoke)


if __name__ == "__main__":
    main()

"""Preregistered dynamic oracle-repair continuation (dynrepair_0826).

Full run:

    OMP_NUM_THREADS=1 .venv/bin/python -m src.dynrepair \
      --config configs/dynrepair_0826.yaml

Implementation smoke:

    OMP_NUM_THREADS=1 .venv/bin/python -m src.dynrepair \
      --config configs/dynrepair_0826.yaml --smoke

The smoke run shortens only the post-intervention horizon and marks every
scientific verdict SMOKE_ONLY.  The registered float64 oracle is always used.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from analysis.dynrepair.dynrepair import (
    SCIENTIFIC_CSVS,
    run_analysis,
    self_test as analysis_self_test,
)
from analysis.function_blind.function_blind import (
    exact_snapshot,
    optimize_oracle,
    predict,
    prepare_oracle_arms,
)
from .common import (
    ROOT,
    build_runs,
    group_runs,
    load_config,
    pick_device,
    resolve_outdir,
)
from .envs import kaiming_mlp_params
from .function_blind_direct import (
    UNIT_KEYS as DIRECT_UNIT_KEYS,
    check_delta_formula,
    complete_state_hashes,
    exact_record,
)
from .rank_int import arm_runs
from .train import load_resume, setup_group, train_group


ARMS = ("A0", "A1", "A2", "A3", "A1_lo", "A1_hi")
# A1_lo / A1_hi are the kick-width sensitivity arms: descriptive only, never
# used by a verdict [spec §3.2]. Only these four carry the registered estimands.
JUDGMENT_ARMS = ("A0", "A1", "A2", "A3")
RAW_UNIT_KEYS = (
    "p_hat", "p_count", "pre_max", "x", "r", "w_norm", "b", "v",
    "strict_dead", "utility_nmse",
)


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    """True when the working tree differs from the pinned commit."""
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip())
    except Exception:
        return True


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value.detach().cpu().numpy()).tobytes()
    ).hexdigest()


def _generator_sha(generator: torch.Generator) -> str:
    return _tensor_sha(generator.get_state())


def _resolve(path: str | os.PathLike[str]) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path(ROOT) / value


def validate_config(cfg: dict[str, Any], *, smoke: bool) -> None:
    dc, cc, ca = cfg["dynrepair"], cfg["common"], cfg["condA"]
    errors = []
    if tuple(dc["arms"]) != ARMS:
        errors.append(f"arms must be {list(ARMS)}")
    exact = {
        "t_int": 500000,
        "period": 10000,
        "post_steps": 200000,
        "placebo_seed": 20260826,
        "bootstrap_seed": 20260826,
        "bootstrap_n": 10000,
    }
    if not smoke:
        for key, expected in exact.items():
            if dc.get(key) != expected:
                errors.append(f"dynrepair.{key} must be {expected}")
    if int(cc.get("generator_offset", -1)) != 0:
        errors.append("common.generator_offset must be 0")
    if list(cc["seeds"]) != list(range(10)):
        errors.append("common.seeds must be 0..9 in ascending order")
    if ca.get("widths") != [100] or ca.get("T_values") != [10000]:
        errors.append("condA must be width=100 and T=10000")
    if ca.get("encodings") != ["std"] or float(ca.get("center_alpha", -1)) != 0.01:
        errors.append("condA must use std with center_alpha=0.01")
    if ca.get("batch_values", [1]) != [1]:
        errors.append("condA must use batch=1")
    if cfg.get("condB", {}).get("widths"):
        errors.append("condB must be disabled")
    if os.environ.get("OMP_NUM_THREADS") != "1":
        errors.append("OMP_NUM_THREADS=1 is required")
    if errors:
        raise SystemExit("dynrepair config validation failed:\n- " + "\n- ".join(errors))


def record_grid(
    t_int: int,
    total: int,
    fine_window: int,
    fine_every: int,
    coarse_every: int,
    period: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Union of trajectory records and flip-after/update-before task heads."""
    fine_stop = min(total, t_int + fine_window)
    trajectory = set(range(t_int, fine_stop + 1, fine_every))
    if fine_stop < total:
        trajectory.update(range(fine_stop + coarse_every, total + 1, coarse_every))
    trajectory.add(t_int)
    trajectory.add(total)

    task_head = {t_int}
    for boundary in range(t_int, total, period):
        if boundary + 1 <= total:
            task_head.add(boundary + 1)
    steps = np.asarray(sorted(trajectory | task_head), dtype=np.int64)
    is_traj = np.asarray([int(x) in trajectory for x in steps], dtype=bool)
    is_task_head = np.asarray([int(x) in task_head for x in steps], dtype=bool)
    return steps, is_traj, is_task_head


class DynRecorder:
    """Preallocated exact-support probe for one arm."""

    def __init__(
        self,
        arm: str,
        steps: np.ndarray,
        is_traj: np.ndarray,
        is_task_head: np.ndarray,
        seeds: np.ndarray,
        units: np.ndarray,
        treated: np.ndarray,
        pre_p_hat: np.ndarray,
        pre_pre_max: np.ndarray,
    ):
        self.arm = arm
        self.steps = np.asarray(steps, dtype=np.int64)
        self.is_traj = np.asarray(is_traj, dtype=bool)
        self.is_task_head = np.asarray(is_task_head, dtype=bool)
        self.seeds = np.asarray(seeds, dtype=np.int64)
        self.units = np.asarray(units, dtype=np.int64)
        self.treated = np.asarray(treated, dtype=bool)
        self.pre_p_hat = np.asarray(pre_p_hat, dtype=np.float64)
        self.pre_pre_max = np.asarray(pre_pre_max, dtype=np.float64)
        self.index = {int(step): i for i, step in enumerate(self.steps)}
        T, R, H = len(self.steps), len(self.seeds), len(self.units)
        self.data = {
            key: np.empty((T, R, H), dtype=(
                bool if key == "strict_dead" else np.float64
            ))
            for key in RAW_UNIT_KEYS
        }
        self.eval_nmse = np.empty((T, R), dtype=np.float64)
        self.filled = np.zeros(T, dtype=bool)

    def __call__(self, st: dict[str, Any], step: int) -> None:
        index = self.index.get(int(step))
        if index is None:
            return
        if self.filled[index]:
            raise RuntimeError(f"{self.arm}: duplicate probe at step {step}")
        record, _ = exact_record(st)
        for key in DIRECT_UNIT_KEYS:
            if key in self.data:
                self.data[key][index] = record[key]
        self.data["p_count"][index] = np.rint(record["p_hat"] * 32.0)
        self.data["strict_dead"][index] = record["p_hat"] == 0.0
        self.eval_nmse[index] = record["eval_nmse"]
        self.filled[index] = True

    def check_complete(self) -> None:
        missing = self.steps[~self.filled]
        if len(missing):
            raise RuntimeError(
                f"{self.arm}: missing {len(missing)} records {missing[:10].tolist()}"
            )

    def arrays(self) -> dict[str, np.ndarray]:
        self.check_complete()
        out = dict(
            arm=np.asarray(self.arm),
            step=self.steps,
            seed=self.seeds,
            unit=self.units,
            is_traj=self.is_traj,
            is_task_head=self.is_task_head,
            treated=self.treated,
            pre_p_hat=self.pre_p_hat,
            pre_pre_max=self.pre_pre_max,
            eval_nmse=self.eval_nmse,
        )
        out.update(self.data)
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.arrays())


def _compare_recorders(a: DynRecorder, b: DynRecorder) -> list[str]:
    aa, bb = a.arrays(), b.arrays()
    differences = []
    for key in sorted(set(aa) | set(bb)):
        if key not in aa or key not in bb:
            differences.append(key)
            continue
        x, y = np.ascontiguousarray(aa[key]), np.ascontiguousarray(bb[key])
        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            differences.append(key)
    return differences


def _state_from_snapshot(
    gkey: tuple[Any, ...],
    runs: list[dict[str, Any]],
    cfg: dict[str, Any],
    device: str,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    st = setup_group(gkey, runs, cfg, device)
    load_resume(st, snapshot)
    return st


def _copy_snapshot(source: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(source)


def _oracle(
    source_path: Path,
    expected_mean: float,
    atol: float,
) -> tuple[
    dict[str, dict[str, torch.Tensor]],
    pd.DataFrame,
    dict[str, Any],
    torch.Tensor,
    torch.Tensor,
]:
    """Call the registered static oracle without transcribing its optimizer."""
    _, net, X, y, _ = exact_snapshot(source_path)
    labels, Ws, bs, cs, p_hat, masks, _, kick_err = prepare_oracle_arms(net, X, y)
    _, nmse, _, trace, best_b, best_c = optimize_oracle(
        labels, Ws, bs, cs, net["v"], X, y
    )
    wanted = {
        "A1_lo": "repair_dead_0.05_k0.25",
        "A1": "repair_dead_0.05_k0.5",
        "A1_hi": "repair_dead_0.05_k1",
    }
    arm_params: dict[str, dict[str, torch.Tensor]] = {}
    for arm, label in wanted.items():
        index = labels.index(label)
        # p_hat after the kick alone, before the oracle bias optimisation: the
        # state S5 calls the tautology check [spec §9 S5].
        _, pre_kick = predict(Ws[index], bs[index], net["v"], cs[index], X)
        arm_params[arm] = dict(
            b=best_b[index].detach().clone(),
            c=best_c[index].detach().clone(),
            nmse=nmse[index].detach().clone(),
            kick_p_hat=(pre_kick > 0).double().mean(dim=0).detach().clone(),
        )
    actual = float(arm_params["A1"]["nmse"].mean().item())
    error = abs(actual - expected_mean)
    sanity = dict(
        check="S1", status="PASS" if error < atol else "FAIL",
        value=actual, threshold=atol,
        detail=f"float64 oracle mean; abs_error={error:.17g}; max_kick_error={max(kick_err.values()):.3g}",
    )
    if sanity["status"] != "PASS":
        raise SystemExit(f"S1 failed: oracle mean {actual:.17g}, error {error:.3g}")
    return arm_params, trace, sanity, p_hat, masks["dead_0.05"]


def _install_oracle(
    source: dict[str, Any], params: dict[str, torch.Tensor]
) -> dict[str, Any]:
    snap = _copy_snapshot(source)
    snap["net"]["b"] = params["b"].to(dtype=snap["net"]["b"].dtype)
    snap["net"]["c"] = params["c"].to(dtype=snap["net"]["c"].dtype)
    return snap


def _build_interventions(
    source: dict[str, Any],
    oracle_params: dict[str, dict[str, torch.Tensor]],
    pre_p_hat: np.ndarray,
    pre_pre_max: np.ndarray,
    dc: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    snaps = {"A0": _copy_snapshot(source)}
    for arm in ("A1", "A1_lo", "A1_hi"):
        snaps[arm] = _install_oracle(source, oracle_params[arm])

    treated = torch.from_numpy(pre_p_hat < float(dc["p_hat_tau"]))
    a2 = _copy_snapshot(source)
    method = torch.Generator(device="cpu")
    method.set_state(a2["gens"]["method"])
    method_before = _generator_sha(method)
    W_new, _, _, _ = kaiming_mlp_params(
        int(a2["net"]["W"].shape[0]),
        int(a2["net"]["W"].shape[1]),
        int(a2["net"]["W"].shape[2]),
        method,
        "cpu",
    )
    a2["net"]["W"][treated] = W_new[treated]
    a2["net"]["b"][treated] = 0.0
    a2["net"]["v"][treated] = 0.0
    a2["gens"]["method"] = method.get_state()
    method_after = _generator_sha(method)
    snaps["A2"] = a2

    a3 = _copy_snapshot(source)
    rng = np.random.Generator(np.random.PCG64(int(dc["placebo_seed"])))
    pre_alive = pre_p_hat >= float(dc["p_hat_tau"])
    a1_b64 = oracle_params["A1"]["b"].detach().cpu().numpy()
    b0 = source["net"]["b"].detach().cpu().numpy().astype(np.float64)
    a3_b64 = b0.copy()
    placebo_seed_rows: list[dict[str, Any]] = []
    a1_abs = np.abs(a1_b64 - b0)
    for ri in range(pre_alive.shape[0]):
        alive_units = np.flatnonzero(pre_alive[ri])
        if not len(alive_units):
            raise SystemExit(f"A3 undefined: seed {ri} has no alive units")
        kappa = float(np.linalg.norm(a1_b64[ri] - b0[ri]))
        c_s = float(kappa / np.sqrt(len(alive_units)))
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(alive_units))
        a3_b64[ri, alive_units] += signs * c_s
        alive_margin = pre_pre_max[ri, alive_units]
        median_margin = float(np.median(alive_margin))
        treated_abs = a1_abs[ri, treated[ri].numpy()]
        placebo_seed_rows.append(dict(
            seed=int(ri), n_dead=int(treated[ri].sum().item()),
            n_alive=int(len(alive_units)), kappa=kappa, c_s=c_s,
            c_over_median_margin=(
                c_s / median_margin if median_margin != 0 else np.inf
            ),
            a1_per_unit_displacement_median=(
                float(np.median(treated_abs)) if len(treated_abs) else np.nan
            ),
            a3_to_a1_per_unit_median=(
                c_s / float(np.median(treated_abs))
                if len(treated_abs) and float(np.median(treated_abs)) != 0 else np.inf
            ),
        ))
    a3["net"]["b"] = torch.from_numpy(a3_b64).to(
        dtype=a3["net"]["b"].dtype, device=a3["net"]["b"].device
    )
    snaps["A3"] = a3
    a2_meta = dict(method_generator_before=method_before,
                   method_generator_after=method_after)
    return snaps, placebo_seed_rows, a2_meta


def _source_sanity(
    source: dict[str, Any], meta_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cond_a = next(
        (row for row in meta.get("sanity", []) if row.get("regime") == "A"),
        None,
    )
    if cond_a is None:
        raise SystemExit("S2 source evidence has no condA row")
    expected = cond_a.get("snapshot_sha256", {})
    actual_values = {
        "W": source["net"]["W"],
        "b": source["net"]["b"],
        "v": source["net"]["v"],
        "c": source["net"]["c"],
        "running_mean": source["running_mean"],
    }
    rows = []
    mismatch = []
    for key, value in actual_values.items():
        actual = _tensor_sha(value)
        rows.append(dict(kind="source_state", arm="source", seed=np.nan,
                         state_key=key, state_sha256=actual))
        if expected.get(key) != actual:
            mismatch.append(key)
    passed = cond_a.get("S2") == "PASS" and not mismatch
    sanity = dict(
        check="S2-source", status="PASS" if passed else "FAIL",
        value=len(mismatch), threshold=0,
        detail=f"saved posreset S2={cond_a.get('S2')}; hash mismatches={mismatch}",
    )
    if not passed:
        raise SystemExit(f"S2 source evidence failed: {sanity['detail']}")
    return sanity, rows


def _seed_from_run_id(value: str) -> int:
    match = re.search(r"_s(\d+)(?:_|$)", str(value))
    if match is None:
        raise ValueError(f"cannot parse seed from run_id {value}")
    return int(match.group(1))


def _compare_source_lop(current_path: Path, source_path: Path, t_int: int) -> dict[str, Any]:
    current = pd.read_csv(current_path)
    source = pd.read_csv(source_path)
    current["seed"] = current.run_id.map(_seed_from_run_id)
    source["seed"] = source.run_id.map(_seed_from_run_id)
    common = sorted(
        (set(current.columns) & set(source.columns)) - {"run_id", "seed", "step"}
    )
    current = current[current.step >= t_int].sort_values(["step", "seed"])
    source = source[source.step >= t_int].sort_values(["step", "seed"])
    overlap = sorted(set(current.step) & set(source.step))
    current = current[current.step.isin(overlap)].reset_index(drop=True)
    source = source[source.step.isin(overlap)].reset_index(drop=True)
    keys_ok = np.array_equal(
        current[["step", "seed"]].to_numpy(),
        source[["step", "seed"]].to_numpy(),
    )
    differing = []
    if keys_ok:
        for key in common:
            a = pd.to_numeric(current[key], errors="coerce").to_numpy(dtype=np.float64)
            b = pd.to_numeric(source[key], errors="coerce").to_numpy(dtype=np.float64)
            if not np.array_equal(a, b, equal_nan=True):
                differing.append(key)
    passed = bool(overlap and keys_ok and common and not differing)
    return dict(
        check="S2-source-log", status="PASS" if passed else "FAIL",
        value=len(differing), threshold=0,
        detail=(
            f"overlap_steps={len(overlap)}; rows={len(current)}; "
            f"common_columns={len(common)}; differing={differing}; keys_ok={keys_ok}"
        ),
    )


def _arm_nonfinite(rec: DynRecorder) -> dict[str, Any]:
    """Per-key non-finite counts and the first record step that carries them."""
    found: dict[str, Any] = {}
    series = {key: value for key, value in rec.data.items() if value.dtype != bool}
    series["eval_nmse"] = rec.eval_nmse
    for key, value in series.items():
        bad = ~np.isfinite(value)
        count = int(bad.sum())
        if not count:
            continue
        axes = tuple(range(1, bad.ndim))
        steps = rec.steps[np.flatnonzero(bad.any(axis=axes))]
        found[key] = dict(n=count, first_step=int(steps[0]) if len(steps) else None)
    return found


def _recorder_sanity(
    recorders: dict[str, DynRecorder],
    dc: dict[str, Any],
    t_int: int,
    kick_p_hat: np.ndarray,
) -> list[dict[str, Any]]:
    geometry_max = 0.0
    identity_mismatch = 0
    identity_skipped = 0
    nonfinite = 0
    judgment_nonfinite = 0
    geometry_by_arm: dict[str, float] = {}
    identity_by_arm: dict[str, int] = {}
    nonfinite_by_arm: dict[str, dict[str, Any]] = {}
    for arm, rec in recorders.items():
        rec.check_complete()
        # S3/S4 are implementation identities. A non-finite entry makes both
        # sides undefined rather than unequal, so they are evaluated on the
        # finite records and the excluded count is reported alongside
        # [spec §5: report non-finite counts, do not drop them silently].
        w2 = np.square(rec.data["w_norm"])
        err = np.abs(np.square(rec.data["x"]) + np.square(rec.data["r"]) - w2)
        rel = np.where(w2 > 0, err / w2, err)
        finite_rel = rel[np.isfinite(rel)]
        arm_geometry = float(finite_rel.max()) if finite_rel.size else 0.0
        geometry_max = max(geometry_max, arm_geometry)
        geometry_by_arm[arm] = arm_geometry
        pre_max = rec.data["pre_max"]
        defined = np.isfinite(pre_max)
        arm_identity = int(
            np.not_equal(
                rec.data["strict_dead"][defined], pre_max[defined] <= 0.0
            ).sum()
        )
        identity_mismatch += arm_identity
        identity_skipped += int((~defined).sum())
        identity_by_arm[arm] = arm_identity
        arm_nonfinite = _arm_nonfinite(rec)
        count = sum(int(item["n"]) for item in arm_nonfinite.values())
        nonfinite += count
        if arm in JUDGMENT_ARMS:
            judgment_nonfinite += count
        if arm_nonfinite:
            nonfinite_by_arm[arm] = arm_nonfinite
    rows = [
        dict(check="S3", status=(
            "PASS" if geometry_max < float(dc["geometry_rtol"]) else "FAIL"
        ), value=geometry_max, threshold=float(dc["geometry_rtol"]),
             detail=f"max relative error in x^2+r^2=w_norm^2; by_arm={geometry_by_arm}"),
        dict(check="S4", status="PASS" if identity_mismatch == 0 else "FAIL",
             value=identity_mismatch, threshold=0,
             detail=(
                 f"strict_dead iff pre_max<=0 mismatches over finite records; "
                 f"by_arm={identity_by_arm}; non_finite_records_skipped={identity_skipped}"
             )),
        # §5 registers "report the count of non-finite values, do not exclude
        # them", so this row never aborts. The four judgment arms are gated
        # separately: a non-finite U there leaves O-1/C-1/Ch-1 undefined.
        dict(check="finite", status="PASS" if nonfinite == 0 else "REPORT",
             value=nonfinite, threshold=np.nan,
             detail=(
                 f"raw non-finite values across all arms; by_arm={nonfinite_by_arm}; "
                 f"descriptive-only arms (A1_lo/A1_hi) are reported, not gated; "
                 f"note p_hat=mean(pre>0) stays finite (=0) where pre is NaN, so "
                 f"pre_max is the divergence indicator, not p_hat"
             )),
        dict(check="finite-judgment",
             status="PASS" if judgment_nonfinite == 0 else "FAIL",
             value=judgment_nonfinite, threshold=0,
             detail=f"non-finite values in the judgment arms {list(JUDGMENT_ARMS)}"),
    ]
    a1 = recorders["A1"]
    i0 = a1.index[t_int]
    post = a1.data["p_hat"][i0]
    # S5 is the tautology check: after the kick alone, every treated unit must
    # sit at preactivation k>0, so p_hat>0 is guaranteed by construction and is
    # what gets verified.  The oracle bias optimisation then runs on top, and
    # the rate that survives it is reported, not gated -- G3 is the registered
    # guard for a thin revived population [spec §6.5, §8, §9 S5].
    kicked = kick_p_hat[a1.treated] > 0.0
    revived = post[a1.treated] > 0.0
    practical = post[a1.treated] >= float(dc["p_hat_tau"])
    by_seed = {
        int(ri): float((post[ri][a1.treated[ri]] > 0.0).mean())
        for ri in range(post.shape[0])
    }
    rows.append(dict(
        check="S5", status="PASS" if kicked.all() else "FAIL",
        value=float(kicked.mean()) if len(kicked) else np.nan, threshold=1.0,
        detail=(
            f"post-kick p_hat>0 rate (tautology check); n_target={int(a1.treated.sum())}; "
            f"after oracle optimisation p_hat>0 rate="
            f"{float(revived.mean()) if len(revived) else np.nan:.6g}; "
            f"practical (p_hat>=0.05) rate="
            f"{float(practical.mean()) if len(practical) else np.nan:.6g}; "
            f"post-oracle by_seed={by_seed}"
        ),
    ))
    a2 = recorders["A2"]
    j0 = a2.index[t_int]
    rows.append(dict(
        check="S7", status="REPORT",
        value=float((a2.data["p_hat"][j0] < float(dc["p_hat_tau"])).mean()),
        threshold=np.nan,
        detail=(
            f"A2 immediate dead_frac; mean U={float(a2.eval_nmse[j0].mean()):.17g}; "
            "cbp_harm_0815 anchors are not on a verified identical setup"
        ),
    ))
    return rows


def _arm_stream_identity(final_hashes: dict[str, dict[str, str]]) -> dict[str, Any]:
    """§8: every arm must see a bit-identical input stream, task boundary and teacher.

    The environment and teacher draw the same shapes every step regardless of
    the network, so if the interventions left the driving streams alone, the
    end-of-run env/teacher/running_mean state and the env & noise generators
    must agree across all six arms.  ``generator.method`` is excluded: A2 spends
    exactly one Kaiming draw from it by design [§3.2], and ``net.*`` is the
    intervention itself.
    """
    shared = sorted(
        key for key in final_hashes["A0"]
        if not key.startswith("net.") and key != "generator.method"
    )
    mismatched: dict[str, list[str]] = {}
    for arm, hashes in final_hashes.items():
        if arm == "A0":
            continue
        differing = [
            key for key in shared if hashes.get(key) != final_hashes["A0"][key]
        ]
        if differing:
            mismatched[arm] = differing
    method_advanced = sorted(
        arm for arm, hashes in final_hashes.items()
        if hashes.get("generator.method") != final_hashes["A0"].get("generator.method")
    )
    return dict(
        check="S2-arm-stream", status="PASS" if not mismatched else "FAIL",
        value=len(mismatched), threshold=0,
        detail=(
            f"compared {len(shared)} shared state keys against A0; "
            f"mismatches={mismatched}; "
            f"arms with an advanced method generator (A2 only is expected)="
            f"{method_advanced}"
        ),
    )


def _delta_sanity(
    gkey: tuple[Any, ...],
    runs: list[dict[str, Any]],
    cfg: dict[str, Any],
    device: str,
    a1_snapshot: dict[str, Any],
    rtol: float,
) -> dict[str, Any]:
    st = _state_from_snapshot(gkey, runs, cfg, device, a1_snapshot)
    record, _, context = exact_record(st, with_context=True)
    R, H = record["utility_raw"].shape
    flat = np.linspace(0, R * H - 1, num=min(20, R * H), dtype=np.int64)
    examples = [(int(x // H), int(x % H)) for x in flat]
    checked = check_delta_formula(
        st, examples, record["utility_raw"], ctx=context
    )
    relative = []
    for row in checked["examples"]:
        relative.append(
            row["abs_error"] / max(abs(row["brute"]), abs(row["vector"]), 1e-300)
        )
    worst = max(relative, default=np.inf)
    return dict(
        check="S6", status="PASS" if worst < rtol else "FAIL",
        value=worst, threshold=rtol,
        detail=f"direct silencing comparison, n={len(relative)}",
    )


def _complete_placebo(
    rows: list[dict[str, Any]],
    a3: DynRecorder,
    t_int: int,
    tau: float,
) -> None:
    i0 = a3.index[t_int]
    post = a3.data["p_hat"][i0]
    for row in rows:
        ri = int(row["seed"])
        target = a3.pre_p_hat[ri] >= tau
        delta_p = post[ri, target] - a3.pre_p_hat[ri, target]
        dead = post[ri, target] < tau
        saturated = post[ri, target] == 1.0
        row.update(
            delta_p_min=float(delta_p.min()),
            delta_p_median=float(np.median(delta_p)),
            delta_p_max=float(delta_p.max()),
            death_n=int(dead.sum()),
            death_rate=float(dead.mean()),
            saturation_n=int(saturated.sum()),
            saturation_rate=float(saturated.mean()),
        )


def _manifest_rows(
    source_rows: list[dict[str, Any]],
    source: dict[str, Any],
    snaps: dict[str, dict[str, Any]],
    pre_p_hat: np.ndarray,
    placebo: list[dict[str, Any]],
    a2_meta: dict[str, Any],
    provenance: dict[str, str],
) -> list[dict[str, Any]]:
    # §12 step 2 pins the pre-run commit in manifest.csv, not only in
    # runner_meta.json.
    rows = [
        dict(kind="provenance", arm="", seed=np.nan, state_key=key,
             state_sha256=value)
        for key, value in sorted(provenance.items())
    ]
    rows += list(source_rows)
    placebo_by_seed = {int(row["seed"]): row for row in placebo}
    for arm in ARMS:
        bdelta = (
            snaps[arm]["net"]["b"].double() - source["net"]["b"].double()
        ).detach().cpu().numpy()
        for ri in range(pre_p_hat.shape[0]):
            p = placebo_by_seed.get(ri, {})
            rows.append(dict(
                kind="arm_seed", arm=arm, seed=ri, state_key="",
                state_sha256="", n_dead=int((pre_p_hat[ri] < 0.05).sum()),
                n_alive=int((pre_p_hat[ri] >= 0.05).sum()),
                bias_delta_l2=float(np.linalg.norm(bdelta[ri])),
                kappa=p.get("kappa", np.nan) if arm == "A3" else np.nan,
                c_s=p.get("c_s", np.nan) if arm == "A3" else np.nan,
                method_generator_before=(
                    a2_meta["method_generator_before"] if arm == "A2" else ""
                ),
                method_generator_after=(
                    a2_meta["method_generator_after"] if arm == "A2" else ""
                ),
            ))
    return rows


def _run_one(
    arm: str,
    gkey: tuple[Any, ...],
    base_runs: list[dict[str, Any]],
    cfg: dict[str, Any],
    device: str,
    output: Path,
    total: int,
    t_int: int,
    snapshot: dict[str, Any],
    recorder: DynRecorder | None,
    steps: np.ndarray,
    is_task_head: np.ndarray,
) -> tuple[dict[str, Any], float]:
    runs = arm_runs(base_runs, arm)
    pre_steps = {
        int(step) for step, task in zip(steps, is_task_head)
        if bool(task) and int(step) > t_int and (int(step) - 1) % int(cfg["dynrepair"]["period"]) == 0
    }
    regular_steps = set(int(step) for step in steps) - pre_steps
    return train_group(
        gkey, runs, cfg, device, str(output), total_steps=total,
        start_step=t_int, resume_state=snapshot, gname=f"dynrepair_{arm}",
        probe=recorder, probe_steps=regular_steps if recorder is not None else (),
        pre_update_probe=recorder,
        pre_update_probe_steps=pre_steps if recorder is not None else (),
    )


_ELAPSED_RE = re.compile(r"elapsed_main=[0-9.]+s")


def _normalise_scientific_csv(path: Path) -> bytes:
    """Strip the provenance that legitimately varies between two executions.

    §9 S8 compares the scientific CSVs of two runs of the same command from the
    same commit, "excluding provenance columns that contain time, elapsed and
    absolute paths".  Only wall-clock elapsed and the repository path appear in
    those files, so both are blanked before the byte comparison.
    """
    text = path.read_text(encoding="utf-8")
    text = _ELAPSED_RE.sub("elapsed_main=<elapsed>", text)
    return text.replace(str(ROOT), "<ROOT>").encode("utf-8")


def compare_outdirs(left: Path, right: Path) -> tuple[bool, list[str]]:
    """Registered S8 cross-run comparison of two completed output directories."""
    left, right = Path(left), Path(right)
    notes: list[str] = []
    ok = True
    for name in SCIENTIFIC_CSVS:
        a, b = left / name, right / name
        if not a.exists() or not b.exists():
            notes.append(f"{name}: MISSING ({a.exists()}/{b.exists()})")
            ok = False
            continue
        if _normalise_scientific_csv(a) == _normalise_scientific_csv(b):
            notes.append(f"{name}: identical")
        else:
            notes.append(f"{name}: DIFFERS")
            ok = False
    for arm in ARMS:
        name = f"logs/unit_traj_{arm}.npz"
        a, b = left / name, right / name
        if not a.exists() or not b.exists():
            notes.append(f"{name}: MISSING ({a.exists()}/{b.exists()})")
            ok = False
            continue
        # The NPZ container embeds no timestamp we rely on, but zip framing is
        # not part of the contract: compare key/dtype/shape/array bytes.
        with np.load(a, allow_pickle=False) as za, np.load(b, allow_pickle=False) as zb:
            if sorted(za.files) != sorted(zb.files):
                notes.append(f"{name}: key set DIFFERS")
                ok = False
                continue
            bad = []
            for key in sorted(za.files):
                x = np.ascontiguousarray(za[key])
                y = np.ascontiguousarray(zb[key])
                if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                    bad.append(key)
            if bad:
                notes.append(f"{name}: arrays DIFFER {bad}")
                ok = False
            else:
                notes.append(f"{name}: identical ({len(za.files)} keys)")
    return ok, notes


def run(config_path: str, *, outdir_arg: str | None, smoke: bool, device_arg: str | None) -> Path:
    cfg = load_config(config_path)
    if device_arg:
        cfg["common"]["device"] = device_arg
    validate_config(cfg, smoke=smoke)
    torch.set_num_threads(1)
    device = pick_device(cfg)
    if device != "cpu":
        raise SystemExit("dynrepair_0826 is registered for deterministic CPU execution")

    dc = cfg["dynrepair"]
    t_int = int(dc["t_int"])
    post_steps = 200 if smoke else int(dc["post_steps"])
    total = t_int + post_steps
    fine_window = min(int(dc["fine_window"]), post_steps)
    outdir = Path(resolve_outdir(config_path, smoke=smoke, outdir=outdir_arg))
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "logs").mkdir(exist_ok=True)

    source_path = _resolve(dc["source_snapshot"])
    source_meta_path = _resolve(dc["source_meta"])
    source_lop_path = _resolve(dc["source_lop"])
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if int(source.get("step", -1)) != t_int:
        raise SystemExit(f"source snapshot step is not {t_int}")
    source_sanity, source_manifest = _source_sanity(source, source_meta_path)

    base_runs = build_runs(cfg)
    groups = group_runs(base_runs)
    if len(groups) != 1:
        raise SystemExit(f"expected one run group, got {list(groups)}")
    gkey, base_runs = next(iter(groups.items()))
    base_runs = sorted(base_runs, key=lambda row: int(row["seed"]))
    if [int(row["seed"]) for row in base_runs] != list(cfg["common"]["seeds"]):
        raise SystemExit("run seeds do not match the registered ascending seed list")
    if int(source["net"]["W"].shape[0]) != len(base_runs):
        raise SystemExit("snapshot run count and config seed count differ")

    pre_state = _state_from_snapshot(gkey, base_runs, cfg, device, source)
    pre_record, pre_record_sanity = exact_record(pre_state)
    pre_p_hat = pre_record["p_hat"]
    pre_pre_max = pre_record["pre_max"]
    if pre_record_sanity["n_nonfinite"]:
        raise SystemExit("source exact record contains non-finite values")

    oracle_params, oracle_trace, s1, _, oracle_treated = _oracle(
        source_path,
        float(dc["oracle_mean_nmse"]),
        float(dc["oracle_atol"]),
    )
    if not np.array_equal(
        oracle_treated.detach().cpu().numpy(), pre_p_hat < float(dc["p_hat_tau"])
    ):
        raise SystemExit("oracle treated mask differs from exact recorder mask")
    oracle_trace.to_csv(outdir / "oracle_trace.csv", index=False)

    snaps, placebo_rows, a2_meta = _build_interventions(
        source, oracle_params, pre_p_hat, pre_pre_max, dc
    )
    cast_state = _state_from_snapshot(gkey, base_runs, cfg, device, snaps["A1"])
    cast_record, _ = exact_record(cast_state)
    s1["detail"] += f"; float32 resume mean={float(cast_record['eval_nmse'].mean()):.17g}"

    steps, is_traj, is_task_head = record_grid(
        t_int, total, fine_window, int(dc["fine_every"]),
        int(dc["coarse_every"]), int(dc["period"]),
    )
    seeds = np.asarray(cfg["common"]["seeds"], dtype=np.int64)
    units = np.arange(int(source["net"]["W"].shape[1]), dtype=np.int64)
    treated = pre_p_hat < float(dc["p_hat_tau"])
    recorders: dict[str, DynRecorder] = {}
    sanity_rows: list[dict[str, Any]] = [source_sanity, s1]
    final_hashes: dict[str, dict[str, str]] = {}
    determinism_rows: list[dict[str, Any]] = []

    # A0 is completed and its three-part S2 is checked before intervention arms.
    for arm in ARMS:
        recorder = DynRecorder(
            arm, steps, is_traj, is_task_head, seeds, units, treated,
            pre_p_hat, pre_pre_max,
        )
        st, elapsed = _run_one(
            arm, gkey, base_runs, cfg, device, outdir, total, t_int,
            snaps[arm], recorder, steps, is_task_head,
        )
        recorder.check_complete()
        recorders[arm] = recorder
        final_hashes[arm] = complete_state_hashes(st)

        with tempfile.TemporaryDirectory(prefix=f"dynrepair_{arm}_replay_") as tmp:
            duplicate = DynRecorder(
                arm, steps, is_traj, is_task_head, seeds, units, treated,
                pre_p_hat, pre_pre_max,
            )
            st_dup, _ = _run_one(
                arm, gkey, base_runs, cfg, device, Path(tmp), total, t_int,
                snaps[arm], duplicate, steps, is_task_head,
            )
            raw_diff = _compare_recorders(recorder, duplicate)
            state_dup = complete_state_hashes(st_dup)
            state_diff = sorted(
                key for key in set(final_hashes[arm]) | set(state_dup)
                if final_hashes[arm].get(key) != state_dup.get(key)
            )
        replay_pass = not raw_diff and not state_diff
        determinism_rows.append(dict(
            check=f"S8-{arm}", status="PASS" if replay_pass else "FAIL",
            value=len(raw_diff) + len(state_diff), threshold=0,
            detail=(
                f"raw_array_differences={raw_diff}; final_state_differences={state_diff}; "
                f"elapsed_main={elapsed:.1f}s"
            ),
        ))
        if not replay_pass:
            raise SystemExit(f"S8 failed for {arm}")

        if arm == "A0":
            with tempfile.TemporaryDirectory(prefix="dynrepair_A0_noprobe_") as tmp:
                st_no_probe, _ = _run_one(
                    arm, gkey, base_runs, cfg, device, Path(tmp), total, t_int,
                    snaps[arm], None, steps, is_task_head,
                )
                no_probe_hash = complete_state_hashes(st_no_probe)
            probe_difference = sorted(
                key for key in set(final_hashes[arm]) | set(no_probe_hash)
                if final_hashes[arm].get(key) != no_probe_hash.get(key)
            )
            s2_probe = dict(
                check="S2-probe-replay",
                status="PASS" if not probe_difference else "FAIL",
                value=len(probe_difference), threshold=0,
                detail=f"A0 probe/no-probe final hash differences={probe_difference}",
            )
            s2_log = _compare_source_lop(
                outdir / "lop_metrics_dynrepair_A0.csv",
                source_lop_path,
                t_int,
            )
            sanity_rows.extend([s2_probe, s2_log])
            if s2_probe["status"] != "PASS" or s2_log["status"] != "PASS":
                raise SystemExit(
                    f"S2 replay failed: {s2_probe['detail']}; {s2_log['detail']}"
                )

    sanity_rows.extend(_recorder_sanity(
        recorders, dc, t_int,
        oracle_params["A1"]["kick_p_hat"].detach().cpu().numpy(),
    ))
    sanity_rows.append(_arm_stream_identity(final_hashes))
    sanity_rows.append(_delta_sanity(
        gkey, base_runs, cfg, device, snaps["A1"], float(dc["delta_rtol"])
    ))
    sanity_rows.extend(determinism_rows)
    # S7 is report-only [spec §9]. Every other FAIL aborts the run, but the
    # evidence is written first: an aborted run must still leave sanity.csv and
    # the raw trajectories that show why it aborted.
    failed = [
        row["check"] for row in sanity_rows
        if row["status"] == "FAIL" and row["check"] != "S7"
    ]

    _complete_placebo(placebo_rows, recorders["A3"], t_int, float(dc["p_hat_tau"]))
    manifest_rows = _manifest_rows(
        source_manifest, source, snaps, pre_p_hat, placebo_rows, a2_meta,
        dict(
            git_hash=_git_hash(),
            git_dirty=str(_git_dirty()),
            config_sha256=_file_sha(Path(config_path)),
            spec_sha256=_file_sha(_resolve(cfg["spec"])),
            source_snapshot_sha256=_file_sha(source_path),
        ),
    )
    for arm, hashes in final_hashes.items():
        for key, digest in sorted(hashes.items()):
            manifest_rows.append(dict(
                kind="final_state", arm=arm, seed=np.nan,
                state_key=key, state_sha256=digest,
            ))
    for arm, recorder in recorders.items():
        recorder.save(outdir / "logs" / f"unit_traj_{arm}.npz")

    runner_meta = dict(
        git_hash=_git_hash(),
        git_dirty=_git_dirty(),
        config=str(Path(config_path)),
        config_sha256=_file_sha(Path(config_path)),
        spec=str(cfg.get("spec", "")),
        source_snapshot=str(source_path),
        source_snapshot_sha256=_file_sha(source_path),
        source_meta=str(source_meta_path),
        source_lop=str(source_lop_path),
        smoke=bool(smoke),
        device=device,
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        torch_num_threads=torch.get_num_threads(),
        t_int=t_int,
        total_steps=total,
        fine_window=fine_window,
        record_steps=[int(x) for x in steps],
        manifest=manifest_rows,
        placebo=placebo_rows,
        sanity=sanity_rows,
        sanity_failed=failed,
        determinism_final_hashes=final_hashes,
    )
    (outdir / "runner_meta.json").write_text(
        json.dumps(runner_meta, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    (outdir / "config_used.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8", newline="\n",
    )
    if failed:
        pd.DataFrame(sanity_rows).to_csv(outdir / "sanity.csv", index=False)
        pd.DataFrame(manifest_rows).to_csv(outdir / "manifest.csv", index=False)
        pd.DataFrame(placebo_rows).to_csv(outdir / "placebo.csv", index=False)
        raise SystemExit(
            f"scientific sanity failed: {failed} (evidence written to {outdir})"
        )
    run_analysis(outdir, cfg, runner_meta, smoke=smoke)
    return outdir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dynrepair_0826.yaml")
    parser.add_argument("--outdir")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--compare-outdirs", nargs=2, metavar=("LEFT", "RIGHT"),
        help="S8 cross-run check: byte-compare the scientific CSVs and raw NPZs "
             "of two completed output directories from the same commit",
    )
    args = parser.parse_args(argv)
    if args.compare_outdirs:
        ok, notes = compare_outdirs(Path(args.compare_outdirs[0]),
                                    Path(args.compare_outdirs[1]))
        for note in notes:
            print(f"  {note}")
        print(f"S8 cross-run: {'PASS' if ok else 'FAIL'}")
        raise SystemExit(0 if ok else 1)
    if args.self_test:
        analysis_self_test()
        steps, traj, head = record_grid(500000, 520000, 10000, 100, 1000, 10000)
        assert steps[0] == 500000 and steps[-1] == 520000
        assert head[np.flatnonzero(steps == 500001)[0]]
        assert head[np.flatnonzero(steps == 510001)[0]]
        assert traj[np.flatnonzero(steps == 510000)[0]]
        assert _normalise_scientific_csv.__doc__
        print("dynrepair runner self-test: PASS")
        return
    outdir = run(
        args.config, outdir_arg=args.outdir, smoke=args.smoke,
        device_arg=args.device,
    )
    print(f"[dynrepair] wrote {outdir}")


if __name__ == "__main__":
    main()

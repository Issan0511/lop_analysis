"""mu_titration_0823: pre-registered dose-response analysis.

The only scientific specification for this module is
``specs/spec_mu_titration_0823.md``.  The analysis reads the eight newly run
arms under ``results/mu_titration_0823/arms``; it never falls back to a
checkpoint or replaces the logged exact-support quantities by an
approximation.

Canonical invocation::

    OMP_NUM_THREADS=1 .venv/bin/python -m analysis.mu_titration.mu_titration \
        --config configs/mu_titration_0823.yaml

Before the new runs exist, the implementation can be checked without looking
at scientific results::

    .venv/bin/python -m analysis.mu_titration.mu_titration --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "mu_titration_0823.yaml"
DEFAULT_RESULTS = ROOT / "results" / "mu_titration_0823"
SPEC = ROOT / "specs" / "spec_mu_titration_0823.md"
PREREG_COMMIT_PREFIX = "39986e2"
ANALYSIS_RELEVANT_PATHS = (
    "analysis/mu_titration/__init__.py",
    "analysis/mu_titration/mu_titration.py",
    "configs/mu_titration_0823.yaml",
    "specs/spec_mu_titration_0823.md",
)

COS_LO, COS_HI, BIN_WIDTH = -1.0, 1.0, 0.05
BIN_EDGES = np.linspace(COS_LO, COS_HI, 41, dtype=np.float64)
BIN_UPPER = BIN_EDGES[1:]
N_BINS = 40
P_LEVELS = 33
P_VALUES = np.arange(P_LEVELS, dtype=np.float64) / 32.0
MIN_BIN_N = 1_000
FIELD_DEN_EPS = 1e-12
FLOAT32_ATOL = 2e-6
FLOAT32_RTOL = 2e-5

UNIT_KEYS = (
    "cos_u_mu", "p_hat", "w_norm", "b", "M", "s", "b_plus_M",
    "cos_crit", "delta_b_field", "delta_wmu_field",
)
RUN_KEYS = ("mu_norm", "eval_loss_exact", "flip_state")
REQUIRED_KEYS = ("step", "seed", "center_alpha", *UNIT_KEYS, *RUN_KEYS)
ENDPOINT_COMMON_KEYS = (
    "step", "cos_u_mu", "p_hat", "w_norm", "b", "flip_state",
    "mu_norm", "eval_loss_exact",
)

CORE_SCOPES = (
    "bulk",
    "boundary",
    "all_recorded",
    "bulk_t_lt_500k",
    "bulk_t_ge_500k",
    "phase_offset_5000",
)
CURVE_SCOPES = set(CORE_SCOPES)

plt.rcParams["font.family"] = [
    "Noto Sans CJK JP", "Noto Sans CJK TC", "Noto Sans CJK KR", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False


class AnalysisError(RuntimeError):
    """A strict source or analysis-contract failure."""


def alpha_token(alpha: float) -> str:
    """Return the runner's stable, round-trippable alpha directory token."""
    x = float(alpha)
    if not math.isfinite(x):
        raise AnalysisError(f"non-finite center_alpha: {alpha!r}")
    if x == 0.0:
        return "0"
    s = repr(x).lower()
    if "e" in s:
        mantissa, exponent = s.split("e")
        s = f"{mantissa}e{int(exponent)}"
    if float(s) != x:
        raise AssertionError(f"alpha token did not round trip: {x!r} -> {s!r}")
    return s


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def analysis_git_provenance(allow_dirty: bool = False) -> dict:
    """Require the analysis/spec/config path set to be committed and clean.

    The check is deliberately path-scoped: untracked raw/derived ``results``
    and unrelated concurrent work do not invalidate this analysis, while an
    uncommitted change to the code or its governing spec/config does.
    """
    status_proc = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "--",
         *ANALYSIS_RELEVANT_PATHS],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    if status_proc.returncode:
        raise AnalysisError(f"git status failed: {status_proc.stderr.strip()}")
    status = status_proc.stdout.rstrip("\n")
    untracked = []
    for rel in ANALYSIS_RELEVANT_PATHS:
        proc = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", rel], cwd=ROOT,
            text=True, capture_output=True, check=False,
        )
        if proc.returncode:
            untracked.append(rel)
    clean = not status and not untracked
    if not clean and not allow_dirty:
        detail = status or ("untracked relevant paths: " + ", ".join(untracked))
        raise AnalysisError(
            "canonical analysis requires committed, clean analysis/spec/config paths; "
            + detail
        )
    return {
        "git_commit": git_commit(),
        "git_clean": clean,
        "git_status_porcelain": status,
        "untracked_relevant_paths": untracked,
        "cleanliness_scope": list(ANALYSIS_RELEVANT_PATHS),
        "dirty_override_used": bool(allow_dirty and not clean),
    }


def json_clean(value):
    """Convert numpy values and non-finite floats to deterministic JSON values."""
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value) -> None:
    text = json.dumps(json_clean(value), ensure_ascii=False, sort_keys=True, indent=2)
    path.write_text(text + "\n", encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path, index=False, lineterminator="\n", na_rep="NA", float_format="%.17g"
    )


def scalar(array: np.ndarray, name: str, path: Path):
    a = np.asarray(array)
    if a.size != 1:
        raise AnalysisError(f"{path}: {name} must be scalar, got {a.shape}")
    return a.reshape(()).item()


def expected_record_steps(total: int, period: int, half_window: int,
                          bulk_every: int) -> np.ndarray:
    steps = set(range(0, total + 1, bulk_every))
    for boundary in range(period, total + 1, period):
        steps.update(range(max(0, boundary - half_window),
                           min(total, boundary + half_window) + 1))
    steps.update((0, total))
    return np.asarray(sorted(steps), dtype=np.int64)


def validate_canonical_config(cfg: dict) -> None:
    """Reject silent departures from the fixed design in spec sections 2 and 8."""
    required = {
        ("common", "total_steps"): 1_000_000,
        ("common", "seeds"): list(range(10)),
        ("common", "lr_main"): 0.01,
        ("common", "device"): "cpu",
        ("ratchet", "boundary_window"): 100,
        ("ratchet", "bulk_every"): 1_000,
        ("ratchet", "bootstrap_B"): 10_000,
        ("ratchet", "bootstrap_seed"): 20260823,
        ("mu_titration", "s2_steps"): 100_000,
        ("mu_titration", "result_subdir"): "arms",
        ("condA", "m"): 20,
        ("condA", "f"): 15,
        ("condA", "target_hidden"): 100,
        ("condA", "T_values"): [10_000],
        ("condA", "widths"): [100],
        ("condA", "encodings"): ["centered"],
        ("condA", "batch_values"): [1],
    }
    for keys, wanted in required.items():
        cur = cfg
        try:
            for key in keys:
                cur = cur[key]
        except (KeyError, TypeError) as exc:
            raise AnalysisError(f"canonical config missing {'.'.join(keys)}") from exc
        if cur != wanted:
            raise AnalysisError(
                f"canonical config mismatch {'.'.join(keys)}: {cur!r} != {wanted!r}"
            )
    grid = [float(x) for x in cfg.get("mu_titration", {}).get("center_alphas", [])]
    wanted_grid = [0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-2]
    if grid != wanted_grid:
        raise AnalysisError(f"center_alpha grid mismatch: {grid!r}")
    if list(cfg["ratchet"].get("unit_keys", [])) != list(UNIT_KEYS):
        raise AnalysisError("ratchet.unit_keys does not exactly match the preregistered schema")
    if list(cfg["ratchet"].get("run_vector_keys", [])) != ["flip_state"]:
        raise AnalysisError("ratchet.run_vector_keys must be [flip_state]")
    if list(cfg["ratchet"].get("run_scalar_keys", [])) != [
        "mu_norm", "eval_loss_exact"
    ]:
        raise AnalysisError("ratchet.run_scalar_keys mismatch")


def config_without_arm_fields(cfg: dict) -> dict:
    # JSON round trip gives a plain, order-insensitive deep copy.
    out = json.loads(json.dumps(cfg))
    out.get("condA", {}).pop("center_alpha", None)
    out.get("mu_titration", {}).pop("active_alpha", None)
    return out


def require_source_meta(arm_dir: Path, alpha: float,
                        canonical_cfg: dict) -> tuple[dict, dict, dict]:
    meta_path = arm_dir / "meta.json"
    arm_meta_path = arm_dir / "arm_meta.json"
    provenance_path = arm_dir / "provenance.json"
    used_path = arm_dir / "config_used.yaml"
    for path in (meta_path, arm_meta_path, provenance_path, used_path):
        if not path.is_file():
            raise AnalysisError(f"missing source artifact: {path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    arm_meta = json.loads(arm_meta_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    used = yaml.safe_load(used_path.read_text(encoding="utf-8"))
    if arm_meta.get("status") != "complete" or arm_meta.get("sanity_pass") is not True:
        raise AnalysisError(f"{arm_dir}: arm_meta does not certify a complete sane run")
    if float(meta.get("center_alpha", math.nan)) != float(alpha):
        raise AnalysisError(f"{arm_dir}: meta center_alpha mismatch")
    if float(arm_meta.get("center_alpha", math.nan)) != float(alpha):
        raise AnalysisError(f"{arm_dir}: arm_meta center_alpha mismatch")
    if int(meta.get("R", -1)) != 10 or int(meta.get("width", -1)) != 100:
        raise AnalysisError(f"{arm_dir}: R/width mismatch")
    if str(meta.get("device")) != "cpu":
        raise AnalysisError(f"{arm_dir}: source device is not cpu")
    if int(meta.get("n_record_steps", -1)) != 20_901:
        raise AnalysisError(f"{arm_dir}: record count mismatch")
    if int(meta.get("n_realized_flips", -1)) != 99:
        raise AnalysisError(f"{arm_dir}: realized flip count mismatch")
    if meta.get("spec") != canonical_cfg.get("spec"):
        raise AnalysisError(f"{arm_dir}: source spec pointer mismatch")
    sanity = meta.get("sanity", {})
    flags = {
        "S1": "s1_pass", "S2": "s2_pass", "S3": "s3_pass",
        "S4": "s4_pass", "S5": "s5_pass", "S7": "s7_pass",
    }
    for check, flag in flags.items():
        if sanity.get(check, {}).get(flag) is not True:
            raise AnalysisError(f"{arm_dir}: source {check}.{flag} is not PASS")
    if int(sanity["S2"].get("s2_steps", -1)) != 100_000:
        raise AnalysisError(f"{arm_dir}: S2 was not run for the preregistered 100,000 steps")
    if int(sanity["S3"].get("s3_eval_N", -1)) != 2_000:
        raise AnalysisError(f"{arm_dir}: S3 eval batch is not 2,000")
    if sanity.get("all_required_pass") is not True:
        raise AnalysisError(f"{arm_dir}: all_required_pass is not true")
    tracked = set(meta.get("tracked_keys", []))
    if not set((*UNIT_KEYS, *RUN_KEYS)).issubset(tracked):
        raise AnalysisError(f"{arm_dir}: tracked_keys missing required columns")
    if config_without_arm_fields(used) != config_without_arm_fields(canonical_cfg):
        raise AnalysisError(f"{arm_dir}: config_used differs from canonical config")
    if float(used.get("condA", {}).get("center_alpha", math.nan)) != float(alpha):
        raise AnalysisError(f"{arm_dir}: config_used active center_alpha mismatch")
    required_provenance = {
        "git_head", "git_clean", "git_source_clean", "git_status_porcelain",
        "config_sha256", "spec_sha256", "source_sha256", "sweep_commit",
        "sweep_fingerprint", "all_arm_alpha_tags",
    }
    missing = sorted(required_provenance - set(provenance))
    if missing:
        raise AnalysisError(f"{arm_dir}: provenance.json missing {missing}")
    if provenance.get("git_clean") is not True or provenance.get("git_source_clean") is not True:
        raise AnalysisError(f"{arm_dir}: run provenance is not source-clean")
    if provenance.get("git_status_porcelain") != "":
        raise AnalysisError(f"{arm_dir}: run provenance has relevant dirty status")
    if arm_meta.get("git_clean") is not True or arm_meta.get("git_source_clean") is not True:
        raise AnalysisError(f"{arm_dir}: arm_meta does not certify clean source")
    if arm_meta.get("git_status_porcelain") != "":
        raise AnalysisError(f"{arm_dir}: arm_meta has relevant dirty status")
    if arm_meta.get("all_arms_same_commit_required") is not True:
        raise AnalysisError(f"{arm_dir}: all-arm same-commit requirement is absent")
    if arm_meta.get("provenance_file") != "provenance.json":
        raise AnalysisError(f"{arm_dir}: provenance file pointer mismatch")
    for key in ("git_head", "config_sha256", "spec_sha256", "source_sha256",
                "sweep_commit", "sweep_fingerprint", "all_arm_alpha_tags"):
        if arm_meta.get(key) != provenance.get(key):
            raise AnalysisError(f"{arm_dir}: arm_meta/provenance mismatch for {key}")
    expected_tags = [f"alpha_{alpha_token(x)}"
                     for x in canonical_cfg["mu_titration"]["center_alphas"]]
    if provenance.get("all_arm_alpha_tags") != expected_tags:
        raise AnalysisError(f"{arm_dir}: provenance alpha-tag grid mismatch")
    return meta, arm_meta, provenance


def load_npz_strict(path: Path, alpha: float, seed: int, expected_steps: np.ndarray,
                    width: int, fdim: int) -> tuple[dict[str, np.ndarray], dict]:
    """Load one seed and independently recheck the saved float32 contract."""
    try:
        with np.load(path, allow_pickle=False) as z:
            missing = sorted(set(REQUIRED_KEYS) - set(z.files))
            if missing:
                raise AnalysisError(f"{path}: missing keys {missing}")
            d = {key: np.asarray(z[key]) for key in REQUIRED_KEYS}
            for optional in ("run_id", "period", "width", "lr"):
                if optional in z.files:
                    d[optional] = np.asarray(z[optional])
    except (OSError, ValueError) as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc

    got_seed = int(scalar(d["seed"], "seed", path))
    got_alpha = float(scalar(d["center_alpha"], "center_alpha", path))
    # The logger intentionally stores this scalar as float32.  Compare against
    # that declared storage representation, not the YAML float64 spelling.
    stored_alpha = float(np.float32(alpha))
    if got_seed != seed or got_alpha != stored_alpha:
        raise AnalysisError(
            f"{path}: scalar identity mismatch seed={got_seed}, alpha={got_alpha}"
        )
    step = np.asarray(d["step"], dtype=np.int64)
    if not np.array_equal(step, expected_steps):
        raise AnalysisError(f"{path}: step grid is not canonical")
    n = len(expected_steps)
    for key in UNIT_KEYS:
        if d[key].shape != (n, width):
            raise AnalysisError(f"{path}: {key} shape {d[key].shape} != {(n, width)}")
    for key in ("mu_norm", "eval_loss_exact"):
        if d[key].shape != (n,):
            raise AnalysisError(f"{path}: {key} shape {d[key].shape} != {(n,)}")
    if d["flip_state"].shape != (n, fdim):
        raise AnalysisError(f"{path}: flip_state shape mismatch")

    for key in (*UNIT_KEYS, *RUN_KEYS):
        if not np.issubdtype(d[key].dtype, np.number):
            raise AnalysisError(f"{path}: {key} is not numeric")
        if not np.isfinite(d[key]).all():
            raise AnalysisError(f"{path}: {key} contains NaN/Inf")
    p = d["p_hat"].astype(np.float64, copy=False)
    quant_err = float(np.max(np.abs(p * 32.0 - np.rint(p * 32.0))))
    if quant_err > 1e-7 or (p < 0).any() or (p > 1).any():
        raise AnalysisError(f"{path}: p_hat lattice/range failure (maxerr={quant_err})")
    cos = d["cos_u_mu"].astype(np.float64, copy=False)
    if (cos < -1.0 - 2e-6).any() or (cos > 1.0 + 2e-6).any():
        raise AnalysisError(f"{path}: cos_u_mu outside [-1,1]")
    w = d["w_norm"].astype(np.float64, copy=False)
    mu = d["mu_norm"].astype(np.float64, copy=False)
    if (w <= 0).any() or (mu <= 0).any() or (d["M"] < 0).any():
        raise AnalysisError(f"{path}: non-positive norm or negative M")

    b = d["b"].astype(np.float64, copy=False)
    M = d["M"].astype(np.float64, copy=False)
    s = d["s"].astype(np.float64, copy=False)
    bM = d["b_plus_M"].astype(np.float64, copy=False)
    crit = d["cos_crit"].astype(np.float64, copy=False)
    bM_ref = b + M
    s_ref = b + cos * w * mu[:, None]
    crit_ref = -bM / (w * mu[:, None])
    if not np.allclose(bM, bM_ref, atol=FLOAT32_ATOL, rtol=FLOAT32_RTOL):
        raise AnalysisError(f"{path}: saved b_plus_M identity failure")
    if not np.allclose(s, s_ref, atol=FLOAT32_ATOL, rtol=FLOAT32_RTOL):
        raise AnalysisError(f"{path}: saved s identity failure")
    if not np.allclose(crit, crit_ref, atol=FLOAT32_ATOL, rtol=FLOAT32_RTOL):
        raise AnalysisError(f"{path}: saved cos_crit identity failure")
    field_max = s + M
    p_zero = p == 0.0
    predicted_zero = field_max <= 0.0
    mismatch_raw = int(np.count_nonzero(p_zero != predicted_zero))
    # S5b was evaluated on the pre-save float64 values.  ``s`` and ``M`` are
    # rounded to float32 independently, so a true value within a few ulps of
    # zero may change sign after loading.  Recheck strictly away from that
    # representational band and report all near-boundary raw disagreements.
    mismatch_beyond_tol = int(np.count_nonzero(
        (p_zero & (field_max > FLOAT32_ATOL))
        | ((~p_zero) & (field_max <= -FLOAT32_ATOL))
    ))
    if mismatch_beyond_tol:
        raise AnalysisError(
            f"{path}: tolerance-aware p_hat==0 iff s+M<=0 mismatch "
            f"({mismatch_beyond_tol}; raw={mismatch_raw})"
        )

    fs = d["flip_state"].astype(np.float64, copy=False)
    if not np.all((fs == 0.0) | (fs == 1.0)):
        raise AnalysisError(f"{path}: flip_state is not binary")
    diff = np.diff(fs, axis=0)
    changed = np.any(diff != 0.0, axis=1)
    left = step[:-1][changed]
    right = step[1:][changed]
    expected_left = np.arange(10_000, 1_000_000, 10_000, dtype=np.int64)
    bit_counts = np.count_nonzero(diff[changed], axis=1)
    if (not np.array_equal(left, expected_left)
            or not np.all(right == left + 1)
            or not np.all(bit_counts == 1)):
        raise AnalysisError(f"{path}: realized flip trajectory violates S4")
    diag = {
        "p_hat_quantization_max_abs_error": quant_err,
        "b_plus_M_max_abs_error": float(np.max(np.abs(bM - bM_ref))),
        "s_max_abs_error": float(np.max(np.abs(s - s_ref))),
        "cos_crit_max_abs_error": float(np.max(np.abs(crit - crit_ref))),
        "p_zero_identity_mismatches_raw_float32": mismatch_raw,
        "p_zero_identity_mismatches_beyond_float32_tol": mismatch_beyond_tol,
        "p_zero_identity_float32_tolerance": FLOAT32_ATOL,
        "n_realized_flips": int(len(left)),
    }
    return d, diag


def boundary_index_matrix(step: np.ndarray, flip_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    changed = np.any(np.diff(flip_state, axis=0) != 0, axis=1)
    left = step[:-1][changed]
    offsets = np.arange(-100, 101, dtype=np.int64)
    targets = left[:, None] + offsets[None, :]
    ix = np.searchsorted(step, targets)
    if (ix >= len(step)).any() or not np.array_equal(step[ix], targets):
        raise AnalysisError("boundary offset [-100,+100] is incomplete")
    # [offset, boundary] makes offset-wise selections contiguous for callers.
    return offsets, ix.T


def scope_layout(step: np.ndarray, flip_state: np.ndarray, period: int,
                 total: int) -> tuple[list[str], list[float], list[np.ndarray], np.ndarray]:
    offsets, bix = boundary_index_matrix(step, flip_state)
    scheduled = np.arange(period, total + 1, period, dtype=np.int64)
    dist = np.min(np.abs(step[:, None] - scheduled[None, :]), axis=1)
    bulk = (step % 1_000 == 0) & (dist > 100)
    boundary = np.zeros(len(step), dtype=bool)
    boundary[np.unique(bix)] = True
    masks = [
        bulk,
        boundary,
        np.ones(len(step), dtype=bool),
        bulk & (step < 500_000),
        bulk & (step >= 500_000),
        (step % period) == 5_000,
    ]
    names = list(CORE_SCOPES)
    off_values: list[float] = [math.nan] * len(names)
    for oi, off in enumerate(offsets):
        mask = np.zeros(len(step), dtype=bool)
        mask[bix[oi]] = True
        names.append(f"boundary_offset_{int(off):+d}")
        off_values.append(float(off))
        masks.append(mask)
    if int(bulk.sum()) != 901:
        raise AnalysisError(f"canonical bulk scope must contain 901 steps, got {bulk.sum()}")
    if int(boundary.sum()) != 99 * 201:
        raise AnalysisError("canonical realized-boundary scope has wrong size")
    return names, off_values, masks, bix


def count_hist(cos: np.ndarray, p: np.ndarray, row_mask: np.ndarray) -> tuple[np.ndarray, int]:
    c = cos[row_mask].reshape(-1).astype(np.float64, copy=False)
    ph = p[row_mask].reshape(-1).astype(np.float64, copy=False)
    inside = (c >= COS_LO) & (c < COS_HI)
    bins = np.floor((c[inside] - COS_LO) / BIN_WIDTH).astype(np.int64)
    levels = np.rint(ph[inside] * 32.0).astype(np.int64)
    code = bins * P_LEVELS + levels
    hist = np.bincount(code, minlength=N_BINS * P_LEVELS)
    return hist.reshape(N_BINS, P_LEVELS).astype(np.int64), int((~inside).sum())


def hist_median(hist: np.ndarray) -> np.ndarray:
    hist = np.asarray(hist)
    n = hist.sum(axis=-1)
    cum = np.cumsum(hist, axis=-1)
    lo_rank = (n - 1) // 2 + 1
    hi_rank = n // 2 + 1
    lo = np.argmax(cum >= lo_rank[..., None], axis=-1)
    hi = np.argmax(cum >= hi_rank[..., None], axis=-1)
    ans = (P_VALUES[lo] + P_VALUES[hi]) / 2.0
    return np.where(n > 0, ans, np.nan)


def theta_one(hist: np.ndarray, median_not_all: bool) -> float:
    n = hist.sum(axis=-1)
    valid = n >= MIN_BIN_N
    med = hist_median(hist)
    zero = (med == 0.0) if median_not_all else (hist[:, 1:].sum(axis=1) == 0)
    valid_idx = np.flatnonzero(valid)
    if not len(valid_idx) or not zero[valid_idx[0]]:
        return math.nan
    theta = math.nan
    for j in valid_idx:
        if not zero[j]:
            break
        theta = float(BIN_UPPER[j])
    return theta


def theta_many_from_zero(zero_count: np.ndarray, total_count: np.ndarray,
                         median_not_all: bool) -> np.ndarray:
    """Vectorized theta for bootstrap rows using p_hat's non-negative lattice."""
    zero_count = np.asarray(zero_count)
    total_count = np.asarray(total_count)
    valid = total_count >= MIN_BIN_N
    if median_not_all:
        zero = zero_count > (total_count // 2)
    else:
        zero = zero_count == total_count
    B = total_count.shape[0]
    theta = np.full(B, np.nan, dtype=np.float64)
    state = np.zeros(B, dtype=np.int8)  # 0 unseen, 1 active, 2 stopped/failed
    for j in range(N_BINS):
        first = (state == 0) & valid[:, j]
        state[first & ~zero[:, j]] = 2
        start = first & zero[:, j]
        state[start] = 1
        theta[start] = BIN_UPPER[j]
        cont = (state == 1) & valid[:, j]
        stop = cont & ~zero[:, j]
        state[stop] = 2
        update = cont & zero[:, j]
        theta[update] = BIN_UPPER[j]
    return theta


def finite_ci(values: np.ndarray) -> tuple[float, float, int]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return math.nan, math.nan, 0
    lo, hi = np.quantile(x, (0.025, 0.975))
    return float(lo), float(hi), int(len(x))


def aggregate_scope_fields(d: dict[str, np.ndarray], row_mask: np.ndarray) -> dict:
    p = d["p_hat"][row_mask].astype(np.float64, copy=False)
    w = d["w_norm"][row_mask].astype(np.float64, copy=False)
    bM = d["b_plus_M"][row_mask].astype(np.float64, copy=False)
    crit = d["cos_crit"][row_mask].astype(np.float64, copy=False)
    mu = d["mu_norm"][row_mask].astype(np.float64, copy=False)
    db = d["delta_b_field"][row_mask].astype(np.float64, copy=False)
    dw = d["delta_wmu_field"][row_mask].astype(np.float64, copy=False)

    q = bM / w
    wall = q > 0.0
    wall_mu = np.broadcast_to(mu[:, None], q.shape)
    negcrit = -crit
    wall &= (negcrit > 0.0) & np.isfinite(negcrit)
    if wall.any():
        ly = np.log(negcrit[wall])
        lq = np.log(q[wall])
        lm = np.log(wall_mu[wall])
        wall_sum_y, wall_sum_q, wall_sum_mu = map(float, (ly.sum(), lq.sum(), lm.sum()))
        wall_count = int(wall.sum())
        wall_identity = float(ly.mean() - (lq.mean() - lm.mean()))
    else:
        wall_sum_y = wall_sum_q = wall_sum_mu = 0.0
        wall_count = 0
        wall_identity = math.nan

    den = np.abs(db) + np.abs(dw)
    eligible = (p > 0.0) & (den > FIELD_DEN_EPS)
    if eligible.any():
        abs_b = float(np.abs(db[eligible]).sum())
        abs_w = float(np.abs(dw[eligible]).sum())
        same = int(np.count_nonzero((db[eligible] * dw[eligible]) > 0.0))
        count = int(eligible.sum())
        structural = np.broadcast_to(
            (1.0 / (1.0 + mu * mu))[:, None], eligible.shape
        )
        structural_sum = float(structural[eligible].sum())
    else:
        abs_b = abs_w = structural_sum = 0.0
        same = count = 0
    return {
        "wall_sum_log_neg_cos": wall_sum_y,
        "wall_sum_log_q": wall_sum_q,
        "wall_sum_log_mu": wall_sum_mu,
        "wall_n": wall_count,
        "wall_identity_error": wall_identity,
        "wall_frac_cos_lt_m1": float(np.mean(crit < -1.0)),
        "wall_frac_abs_cos_gt1": float(np.mean(np.abs(crit) > 1.0)),
        "wall_frac_b_plus_M_le0": float(np.mean(bM <= 0.0)),
        "bias_abs_b_sum": abs_b,
        "bias_abs_wmu_sum": abs_w,
        "bias_n": count,
        "bias_same_sign_n": same,
        "bias_structural_sum": structural_sum,
    }


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else math.nan


def summarize_arm(alpha: float, arm_dir: Path, canonical_cfg: dict,
                  expected_steps: np.ndarray) -> dict:
    """Validate and reduce one arm without retaining its large unit arrays."""
    meta, arm_meta, provenance = require_source_meta(arm_dir, alpha, canonical_cfg)
    width = int(canonical_cfg["condA"]["widths"][0])
    fdim = int(canonical_cfg["condA"]["f"])
    period = int(canonical_cfg["condA"]["T_values"][0])
    total = int(canonical_cfg["common"]["total_steps"])
    nseed = len(canonical_cfg["common"]["seeds"])

    counts = None
    outside = None
    scope_names: list[str] | None = None
    scope_offsets: list[float] | None = None
    scope_masks: list[np.ndarray] | None = None
    path_stats: list[list[dict]] = []
    per_seed_rows: list[dict] = []
    mu_bulk_by_seed: list[np.ndarray] = []
    flip_by_seed: list[np.ndarray] = []
    step0_arrays: list[dict[str, np.ndarray]] = []
    diagnostics: list[dict] = []
    bulk_coscrit: list[np.ndarray] = []
    bulk_coscrit_mask: list[np.ndarray] = []

    for si, seed in enumerate(canonical_cfg["common"]["seeds"]):
        path = arm_dir / "logs" / f"seed{seed}.npz"
        if not path.is_file():
            raise AnalysisError(f"missing source log: {path}")
        d, diag = load_npz_strict(path, alpha, int(seed), expected_steps, width, fdim)
        diagnostics.append({"seed": int(seed), **diag})
        if scope_names is None:
            scope_names, scope_offsets, scope_masks, _ = scope_layout(
                d["step"], d["flip_state"], period, total
            )
            counts = np.zeros(
                (nseed, len(scope_names), N_BINS, P_LEVELS), dtype=np.int64
            )
            outside = np.zeros((nseed, len(scope_names)), dtype=np.int64)
        else:
            _, _, masks_check, _ = scope_layout(d["step"], d["flip_state"], period, total)
            if any(not np.array_equal(a, b) for a, b in zip(scope_masks, masks_check)):
                raise AnalysisError(f"{path}: scope grid differs across seeds")

        assert counts is not None and outside is not None and scope_masks is not None
        seed_scope_stats = []
        for gi, mask in enumerate(scope_masks):
            counts[si, gi], outside[si, gi] = count_hist(
                d["cos_u_mu"], d["p_hat"], mask
            )
            seed_scope_stats.append(aggregate_scope_fields(d, mask))
        path_stats.append(seed_scope_stats)

        bulk = scope_masks[0]
        mu_bulk = d["mu_norm"][bulk].astype(np.float64, copy=True)
        mu_bulk_by_seed.append(mu_bulk)
        bulk_theta_med = theta_one(counts[si, 0], True)
        bulk_theta_all = theta_one(counts[si, 0], False)
        st = seed_scope_stats[0]
        wall_n = st["wall_n"]
        bias_den = st["bias_abs_b_sum"] + st["bias_abs_wmu_sum"]
        final_p = d["p_hat"][-1].astype(np.float64)
        per_seed_rows.append({
            "alpha": float(alpha),
            "arm": f"alpha_{alpha_token(alpha)}",
            "seed": int(seed),
            "scope": "bulk",
            "mu_norm": float(np.median(mu_bulk)),
            "theta_med": bulk_theta_med,
            "theta_all": bulk_theta_all,
            "wall_mean_log_neg_cos_crit": _safe_ratio(st["wall_sum_log_neg_cos"], wall_n),
            "wall_mean_log_q": _safe_ratio(st["wall_sum_log_q"], wall_n),
            "wall_mean_log_mu_norm": _safe_ratio(st["wall_sum_log_mu"], wall_n),
            "wall_identity_error": st["wall_identity_error"],
            "wall_n": wall_n,
            "bias_share_field": _safe_ratio(st["bias_abs_b_sum"], bias_den),
            "bias_structural_reference": _safe_ratio(st["bias_structural_sum"], st["bias_n"]),
            "bias_same_sign_rate": _safe_ratio(st["bias_same_sign_n"], st["bias_n"]),
            "bias_n": st["bias_n"],
            "final_strict_dead": float(np.mean(final_p == 0.0)),
            "final_near_off": float(np.mean((final_p > 0.0) & (final_p < 0.05))),
            "final_dead_0_05": float(np.mean(final_p < 0.05)),
            "final_eval_loss_exact": float(d["eval_loss_exact"][-1]),
        })
        flip_by_seed.append(d["flip_state"].copy())
        step0_arrays.append({k: np.asarray(d[k][0]).copy() for k in (*UNIT_KEYS, *RUN_KEYS)})
        qmask = (d["b_plus_M"][bulk] / d["w_norm"][bulk]) > 0.0
        bulk_coscrit.append(d["cos_crit"][bulk].astype(np.float32, copy=True))
        bulk_coscrit_mask.append(qmask.copy())

    assert counts is not None and outside is not None
    assert scope_names is not None and scope_offsets is not None
    stats_fields = list(path_stats[0][0])
    stat_arrays = {
        field: np.asarray(
            [[path_stats[s][g][field] for g in range(len(scope_names))]
             for s in range(nseed)], dtype=np.float64
        )
        for field in stats_fields
    }
    qcos = np.concatenate([
        c[m] for c, m in zip(bulk_coscrit, bulk_coscrit_mask)
    ])
    return {
        "alpha": float(alpha),
        "arm": f"alpha_{alpha_token(alpha)}",
        "arm_dir": arm_dir,
        "meta": meta,
        "arm_meta": arm_meta,
        "provenance": provenance,
        "counts": counts,
        "outside": outside,
        "scope_names": scope_names,
        "scope_offsets": scope_offsets,
        "scope_masks": scope_masks,
        "stat_arrays": stat_arrays,
        "per_seed_rows": per_seed_rows,
        "mu_bulk_by_seed": mu_bulk_by_seed,
        "flip_by_seed": flip_by_seed,
        "step0_arrays": step0_arrays,
        "bulk_q_coscrit_median": float(np.median(qcos)) if len(qcos) else math.nan,
        "diagnostics": diagnostics,
    }


def validate_sweep_provenance(arms: Sequence[dict], config_path: Path) -> dict:
    """Require one clean preregistered commit/fingerprint across all arms."""
    if not arms:
        raise AnalysisError("no arms for provenance validation")
    fields = ("sweep_commit", "sweep_fingerprint", "source_sha256",
              "config_sha256", "spec_sha256")
    baseline = arms[0]["provenance"]
    for arm in arms:
        prov = arm["provenance"]
        for field in fields:
            if prov.get(field) != baseline.get(field):
                raise AnalysisError(
                    f"cross-arm provenance mismatch: {field} at {arm['arm']}"
                )
    commit = str(baseline.get("sweep_commit", ""))
    fingerprint = str(baseline.get("sweep_fingerprint", ""))
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit.lower()):
        raise AnalysisError(f"invalid sweep_commit: {commit!r}")
    if not commit.startswith(PREREG_COMMIT_PREFIX):
        raise AnalysisError(
            f"sweep_commit {commit} is not the preregistered {PREREG_COMMIT_PREFIX} lineage"
        )
    if len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint.lower()):
        raise AnalysisError("invalid sweep_fingerprint")
    source_sha = baseline.get("source_sha256")
    expected_source_paths = {
        "src/mu_titration.py", "src/ratchet_log.py", "src/train.py",
        "src/common.py", "src/envs.py", "src/nets.py",
    }
    if not isinstance(source_sha, dict) or set(source_sha) != expected_source_paths:
        raise AnalysisError("source_sha256 does not cover the canonical runner source set")
    for path, digest in source_sha.items():
        if len(str(digest)) != 64 or any(c not in "0123456789abcdef"
                                        for c in str(digest).lower()):
            raise AnalysisError(f"invalid source SHA for {path}")
    if baseline.get("config_sha256") != sha256_file(config_path):
        raise AnalysisError("run config SHA does not match the analysis config")
    if baseline.get("spec_sha256") != sha256_file(SPEC):
        raise AnalysisError("run spec SHA does not match the governing spec")
    return {
        "all_arms_match": True,
        "git_clean": True,
        "git_source_clean": True,
        "sweep_commit": commit,
        "sweep_fingerprint": fingerprint,
        "config_sha256": baseline["config_sha256"],
        "spec_sha256": baseline["spec_sha256"],
        "source_sha256": source_sha,
        "arm_provenance_sha256": {
            arm["arm"]: sha256_file(arm["arm_dir"] / "provenance.json") for arm in arms
        },
    }


def make_bootstrap_weights(B: int, nseed: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, nseed, size=(B, nseed))
    weights = np.zeros((B, nseed), dtype=np.int16)
    np.add.at(weights, (np.arange(B)[:, None], draws), 1)
    return weights


def bootstrap_weighted_median(values_by_seed: Sequence[np.ndarray],
                              weights: np.ndarray, chunk: int = 256) -> np.ndarray:
    lengths = np.asarray([len(v) for v in values_by_seed], dtype=np.int64)
    values = np.concatenate([np.asarray(v, dtype=np.float64) for v in values_by_seed])
    seed_id = np.concatenate([
        np.full(n, i, dtype=np.int16) for i, n in enumerate(lengths)
    ])
    order = np.argsort(values, kind="mergesort")
    values, seed_id = values[order], seed_id[order]
    out = np.empty(len(weights), dtype=np.float64)
    for start in range(0, len(weights), chunk):
        w = weights[start:start + chunk]
        total = w @ lengths
        expanded = w[:, seed_id]
        cumulative = np.cumsum(expanded, axis=1)
        lo_rank = (total - 1) // 2 + 1
        hi_rank = total // 2 + 1
        lo_idx = np.argmax(cumulative >= lo_rank[:, None], axis=1)
        hi_idx = np.argmax(cumulative >= hi_rank[:, None], axis=1)
        out[start:start + len(w)] = (values[lo_idx] + values[hi_idx]) / 2.0
    return out


def bootstrap_theta_scope(seed_hist: np.ndarray, weights: np.ndarray,
                          with_curve: bool) -> dict:
    seed_hist = np.asarray(seed_hist, dtype=np.int64)
    flat = seed_hist.reshape(seed_hist.shape[0], -1)
    if with_curve:
        hist = (weights.astype(np.int32) @ flat.astype(np.int32)).reshape(
            len(weights), N_BINS, P_LEVELS
        )
        med = hist_median(hist)
        tm = theta_many_from_zero(hist[:, :, 0], hist.sum(axis=2), True)
        ta = theta_many_from_zero(hist[:, :, 0], hist.sum(axis=2), False)
        curve_ci = [finite_ci(med[:, j]) for j in range(N_BINS)]
        curve_lo = np.asarray([x[0] for x in curve_ci])
        curve_hi = np.asarray([x[1] for x in curve_ci])
        curve_finite = np.asarray([x[2] for x in curve_ci])
        return {
            "theta_med": tm, "theta_all": ta,
            "curve_lo": curve_lo, "curve_hi": curve_hi,
            "curve_finite": curve_finite,
        }
    totals = seed_hist.sum(axis=2)
    zero = seed_hist[:, :, 0]
    total_boot = weights.astype(np.int32) @ totals.astype(np.int32)
    zero_boot = weights.astype(np.int32) @ zero.astype(np.int32)
    return {
        "theta_med": theta_many_from_zero(zero_boot, total_boot, True),
        "theta_all": theta_many_from_zero(zero_boot, total_boot, False),
    }


def weighted_mean_boot(weights: np.ndarray, sums: np.ndarray,
                       counts: np.ndarray) -> np.ndarray:
    num = weights @ np.asarray(sums, dtype=np.float64)
    den = weights @ np.asarray(counts, dtype=np.float64)
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


def build_arm_estimates(arm: dict, weights: np.ndarray) -> tuple[list[dict], list[dict], dict]:
    """Create curve/theta tables and all bootstrap intermediates for one arm."""
    curve_rows: list[dict] = []
    theta_rows: list[dict] = []
    bulk_boot: dict[str, np.ndarray] = {}
    point_hist = arm["counts"].sum(axis=0)
    for gi, (scope, offset) in enumerate(zip(arm["scope_names"], arm["scope_offsets"])):
        pooled = point_hist[gi]
        n = pooled.sum(axis=1)
        med = hist_median(pooled)
        with_curve = scope in CURVE_SCOPES
        boot = bootstrap_theta_scope(arm["counts"][:, gi], weights, with_curve)
        if scope == "bulk":
            bulk_boot["theta_med"] = boot["theta_med"]
            bulk_boot["theta_all"] = boot["theta_all"]
        if with_curve:
            for bi in range(N_BINS):
                curve_rows.append({
                    "alpha": arm["alpha"], "arm": arm["arm"], "scope": scope,
                    "boundary_offset": offset, "bin_index": bi,
                    "cos_lo": BIN_EDGES[bi], "cos_hi": BIN_EDGES[bi + 1],
                    "n": int(n[bi]), "valid": bool(n[bi] >= MIN_BIN_N),
                    "p_hat_median": float(med[bi]) if n[bi] else math.nan,
                    "ci_lo": float(boot["curve_lo"][bi]),
                    "ci_hi": float(boot["curve_hi"][bi]),
                    "bootstrap_finite": int(boot["curve_finite"][bi]),
                    "outside_cos_range_n": int(arm["outside"][:, gi].sum()),
                })
        for kind, median_not_all in (("theta_med", True), ("theta_all", False)):
            point = theta_one(pooled, median_not_all)
            reps = boot[kind]
            lo, hi, nf = finite_ci(reps)
            theta_rows.append({
                "alpha": arm["alpha"], "arm": arm["arm"], "scope": scope,
                "boundary_offset": offset, "estimate": kind, "point": point,
                "ci_lo": lo, "ci_hi": hi, "bootstrap_finite": nf,
                "n_valid_bins": int(np.count_nonzero(n >= MIN_BIN_N)),
                "min_bin_n": MIN_BIN_N,
                "left_censored_exact": bool(
                    kind == "theta_med" and scope == "bulk" and not math.isfinite(point)
                    and arm["bulk_q_coscrit_median"] < -1.0
                ),
                "bulk_q_positive_cos_crit_median": (
                    arm["bulk_q_coscrit_median"] if scope == "bulk" else math.nan
                ),
            })

    dose = float(np.median(np.concatenate(arm["mu_bulk_by_seed"])))
    dose_boot = bootstrap_weighted_median(arm["mu_bulk_by_seed"], weights)
    bulk_boot["dose"] = dose_boot
    stats = arm["stat_arrays"]
    bulk_i = arm["scope_names"].index("bulk")
    wall_y = weighted_mean_boot(
        weights, stats["wall_sum_log_neg_cos"][:, bulk_i], stats["wall_n"][:, bulk_i]
    )
    wall_q = weighted_mean_boot(
        weights, stats["wall_sum_log_q"][:, bulk_i], stats["wall_n"][:, bulk_i]
    )
    wall_mu = weighted_mean_boot(
        weights, stats["wall_sum_log_mu"][:, bulk_i], stats["wall_n"][:, bulk_i]
    )
    abs_b = weights @ stats["bias_abs_b_sum"][:, bulk_i]
    abs_w = weights @ stats["bias_abs_wmu_sum"][:, bulk_i]
    bias = np.divide(abs_b, abs_b + abs_w, out=np.full(len(weights), np.nan),
                     where=(abs_b + abs_w) > 0)
    struct = weighted_mean_boot(
        weights, stats["bias_structural_sum"][:, bulk_i], stats["bias_n"][:, bulk_i]
    )
    seed_frame = pd.DataFrame(arm["per_seed_rows"]).sort_values("seed")
    for col in ("final_strict_dead", "final_near_off", "final_dead_0_05",
                "final_eval_loss_exact"):
        bulk_boot[col] = (weights @ seed_frame[col].to_numpy(dtype=np.float64)) / weights.sum(axis=1)
    bulk_boot.update(wall_y=wall_y, wall_q=wall_q, wall_mu=wall_mu,
                     bias_share=bias, bias_struct=struct)
    bulk_boot["point_dose"] = np.asarray(dose)
    return curve_rows, theta_rows, bulk_boot


def slope_1d(x: np.ndarray, y: np.ndarray, min_points: int = 2) -> float:
    good = np.isfinite(x) & np.isfinite(y)
    if int(good.sum()) < min_points:
        return math.nan
    xx, yy = x[good], y[good]
    dx = xx - xx.mean()
    den = float(np.dot(dx, dx))
    if den <= 0:
        return math.nan
    return float(np.dot(dx, yy - yy.mean()) / den)


def row_slopes(x: np.ndarray, y: np.ndarray, min_points: int = 2) -> np.ndarray:
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("row_slopes requires matching B x arm matrices")
    return np.asarray([slope_1d(x[i], y[i], min_points) for i in range(len(x))])


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    x = np.asarray(list(x), dtype=np.float64)
    y = np.asarray(list(y), dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    if int(good.sum()) < 2:
        return math.nan
    rx = pd.Series(x[good]).rank(method="average").to_numpy()
    ry = pd.Series(y[good]).rank(method="average").to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def regression_record(metric: str, scope: str, point: float,
                      reps: np.ndarray, x_definition: str) -> dict:
    lo, hi, nf = finite_ci(reps)
    return {
        "record_type": "regression", "metric": metric, "scope": scope,
        "alpha": math.nan, "arm": "ALL_ARMS", "boundary_offset": math.nan,
        "estimate": point, "ci_lo": lo, "ci_hi": hi,
        "bootstrap_finite": nf, "n": math.nan, "x_definition": x_definition,
    }


def point_and_boot_path_rows(arm: dict, weights: np.ndarray) -> list[dict]:
    rows = []
    S = arm["stat_arrays"]
    for gi, (scope, offset) in enumerate(zip(arm["scope_names"], arm["scope_offsets"])):
        wall_n = S["wall_n"][:, gi]
        wy = weighted_mean_boot(weights, S["wall_sum_log_neg_cos"][:, gi], wall_n)
        wq = weighted_mean_boot(weights, S["wall_sum_log_q"][:, gi], wall_n)
        wm = weighted_mean_boot(weights, S["wall_sum_log_mu"][:, gi], wall_n)
        abs_b = weights @ S["bias_abs_b_sum"][:, gi]
        abs_w = weights @ S["bias_abs_wmu_sum"][:, gi]
        bias = np.divide(abs_b, abs_b + abs_w, out=np.full(len(weights), np.nan),
                         where=(abs_b + abs_w) > 0)
        structural = weighted_mean_boot(
            weights, S["bias_structural_sum"][:, gi], S["bias_n"][:, gi]
        )
        same = weighted_mean_boot(
            weights, S["bias_same_sign_n"][:, gi], S["bias_n"][:, gi]
        )
        metrics = {
            "mean_log_neg_cos_crit": (wy, float(S["wall_sum_log_neg_cos"][:, gi].sum()
                                                   / wall_n.sum()) if wall_n.sum() else math.nan,
                                       int(wall_n.sum())),
            "mean_log_q": (wq, float(S["wall_sum_log_q"][:, gi].sum() / wall_n.sum())
                            if wall_n.sum() else math.nan, int(wall_n.sum())),
            "mean_log_mu_norm_wall_mask": (
                wm, float(S["wall_sum_log_mu"][:, gi].sum() / wall_n.sum())
                if wall_n.sum() else math.nan, int(wall_n.sum())
            ),
            "bias_share_field": (
                bias, _safe_ratio(float(S["bias_abs_b_sum"][:, gi].sum()),
                                  float(S["bias_abs_b_sum"][:, gi].sum()
                                        + S["bias_abs_wmu_sum"][:, gi].sum())),
                int(S["bias_n"][:, gi].sum())
            ),
            "bias_structural_reference": (
                structural, _safe_ratio(float(S["bias_structural_sum"][:, gi].sum()),
                                        float(S["bias_n"][:, gi].sum())),
                int(S["bias_n"][:, gi].sum())
            ),
            "bias_same_sign_rate": (
                same, _safe_ratio(float(S["bias_same_sign_n"][:, gi].sum()),
                                  float(S["bias_n"][:, gi].sum())),
                int(S["bias_n"][:, gi].sum())
            ),
        }
        for metric, (reps, point, n) in metrics.items():
            lo, hi, nf = finite_ci(reps)
            rows.append({
                "record_type": "arm_scope", "metric": metric, "scope": scope,
                "alpha": arm["alpha"], "arm": arm["arm"],
                "boundary_offset": offset, "estimate": point,
                "ci_lo": lo, "ci_hi": hi, "bootstrap_finite": nf, "n": n,
                "x_definition": "",
            })
        # Required wall-regime fractions are seed-equal means; all seeds have the
        # same number of unit-records in a given scope.
        for field, metric in (
            ("wall_frac_cos_lt_m1", "frac_cos_crit_lt_m1"),
            ("wall_frac_abs_cos_gt1", "frac_abs_cos_crit_gt1"),
            ("wall_frac_b_plus_M_le0", "frac_b_plus_M_le0"),
        ):
            reps = (weights @ S[field][:, gi]) / weights.sum(axis=1)
            lo, hi, nf = finite_ci(reps)
            rows.append({
                "record_type": "arm_scope", "metric": metric, "scope": scope,
                "alpha": arm["alpha"], "arm": arm["arm"],
                "boundary_offset": offset,
                "estimate": float(S[field][:, gi].mean()), "ci_lo": lo,
                "ci_hi": hi, "bootstrap_finite": nf,
                "n": int(len(arm["scope_masks"][gi]) * 0 +
                         arm["scope_masks"][gi].sum() * 100 * 10),
                "x_definition": "",
            })
        identity = (
            float(S["wall_sum_log_neg_cos"][:, gi].sum() / wall_n.sum()
                  - (S["wall_sum_log_q"][:, gi].sum() / wall_n.sum()
                     - S["wall_sum_log_mu"][:, gi].sum() / wall_n.sum()))
            if wall_n.sum() else math.nan
        )
        rows.append({
            "record_type": "identity", "metric": "wall_log_additive_error",
            "scope": scope, "alpha": arm["alpha"], "arm": arm["arm"],
            "boundary_offset": offset, "estimate": identity,
            "ci_lo": math.nan, "ci_hi": math.nan, "bootstrap_finite": 0,
            "n": int(wall_n.sum()), "x_definition": "log(-cos_crit)=log(q)-log(mu_norm)",
        })
    return rows


def validate_s6(arms: Sequence[dict], cfg: dict) -> dict:
    """Cross-arm step-0/flip identity and legacy endpoint reproduction (S6)."""
    base = arms[0]
    step0_hash = base["meta"].get("step0_repro_hash")
    if not isinstance(step0_hash, dict) or not step0_hash:
        raise AnalysisError("S6a: source meta lacks step0_repro_hash")
    for arm in arms[1:]:
        if arm["meta"].get("step0_repro_hash") != step0_hash:
            raise AnalysisError(f"S6a: step0 reproducibility hash differs for {arm['arm']}")
        for seed in range(10):
            if not np.array_equal(arm["flip_by_seed"][seed], base["flip_by_seed"][seed]):
                raise AnalysisError(f"S6a: flip trajectory differs at {arm['arm']} seed{seed}")
            for key in base["step0_arrays"][seed]:
                if not np.array_equal(
                    arm["step0_arrays"][seed][key], base["step0_arrays"][seed][key],
                    equal_nan=True,
                ):
                    raise AnalysisError(
                        f"S6a: logged step0 {key} differs at {arm['arm']} seed{seed}"
                    )

    endpoints = cfg["mu_titration"].get("endpoint_references", {})
    endpoint_rows = []
    by_alpha = {a["alpha"]: a for a in arms}
    for text_alpha, ref_text in sorted(endpoints.items(), key=lambda kv: float(kv[0])):
        alpha = float(text_alpha)
        arm = by_alpha.get(alpha)
        if arm is None:
            raise AnalysisError(f"S6b: endpoint alpha {alpha} absent")
        refdir = Path(ref_text)
        if not refdir.is_absolute():
            refdir = ROOT / refdir
        for seed in range(10):
            new_path = arm["arm_dir"] / "logs" / f"seed{seed}.npz"
            ref_path = refdir / "logs" / f"seed{seed}.npz"
            if not ref_path.is_file():
                raise AnalysisError(f"S6b: endpoint reference missing: {ref_path}")
            with np.load(new_path, allow_pickle=False) as new, np.load(
                    ref_path, allow_pickle=False) as ref:
                for key in ENDPOINT_COMMON_KEYS:
                    if key not in new.files or key not in ref.files:
                        raise AnalysisError(f"S6b: common endpoint key {key} absent")
                    if not np.array_equal(new[key], ref[key], equal_nan=True):
                        raise AnalysisError(
                            f"S6b: alpha={alpha:g} seed={seed} key={key} differs from {refdir}"
                        )
        endpoint_rows.append({
            "alpha": alpha, "reference": repo_path(refdir), "n_seeds": 10,
            "keys": list(ENDPOINT_COMMON_KEYS), "bit_equal": True,
        })
    return {
        "S6a_pass": True,
        "step0_repro_hash_sha256": sha256_bytes(
            json.dumps(step0_hash, sort_keys=True).encode("utf-8")
        ),
        "flip_trajectory_equal": True,
        "logged_step0_equal": True,
        "S6b_pass": True,
        "endpoint_checks": endpoint_rows,
    }


def order_violations(dose: np.ndarray, theta: np.ndarray) -> tuple[int, int, float]:
    good = np.isfinite(dose) & np.isfinite(theta)
    dose, theta = dose[good], theta[good]
    bad = total = 0
    for i in range(len(dose)):
        for j in range(i + 1, len(dose)):
            if dose[i] == dose[j]:
                continue
            total += 1
            bad += int((dose[i] - dose[j]) * (theta[i] - theta[j]) < 0)
    return bad, total, float(bad / total) if total else math.nan


def derive_results(arms: list[dict], weights: np.ndarray) -> dict:
    curve_rows: list[dict] = []
    theta_rows: list[dict] = []
    path_rows: list[dict] = []
    boots: list[dict] = []
    for arm in arms:
        cr, tr, bt = build_arm_estimates(arm, weights)
        curve_rows.extend(cr)
        theta_rows.extend(tr)
        path_rows.extend(point_and_boot_path_rows(arm, weights))
        boots.append(bt)

    theta_df = pd.DataFrame(theta_rows)
    per_seed = pd.DataFrame(
        [row for arm in arms for row in arm["per_seed_rows"]]
    ).sort_values(["alpha", "seed"]).reset_index(drop=True)

    dose = np.asarray([float(b["point_dose"]) for b in boots])
    dose_b = np.column_stack([b["dose"] for b in boots])
    theta = np.asarray([
        theta_df[(theta_df.alpha == a["alpha"]) & (theta_df.scope == "bulk")
                 & (theta_df.estimate == "theta_med")].iloc[0].point
        for a in arms
    ], dtype=np.float64)
    theta_b = np.column_stack([b["theta_med"] for b in boots])
    def bulk_point(arm: dict, sum_field: str, count_field: str) -> float:
        gi = arm["scope_names"].index("bulk")
        sums = arm["stat_arrays"][sum_field][:, gi].sum()
        counts = arm["stat_arrays"][count_field][:, gi].sum()
        return _safe_ratio(float(sums), float(counts))

    wall_y = np.asarray([
        bulk_point(a, "wall_sum_log_neg_cos", "wall_n") for a in arms
    ])
    wall_q = np.asarray([
        bulk_point(a, "wall_sum_log_q", "wall_n") for a in arms
    ])
    wall_mu = np.asarray([
        bulk_point(a, "wall_sum_log_mu", "wall_n") for a in arms
    ])
    wall_y_b = np.column_stack([b["wall_y"] for b in boots])
    wall_q_b = np.column_stack([b["wall_q"] for b in boots])
    wall_mu_b = np.column_stack([b["wall_mu"] for b in boots])
    bias = []
    bias_x = []
    for arm in arms:
        gi = arm["scope_names"].index("bulk")
        S = arm["stat_arrays"]
        abs_b = float(S["bias_abs_b_sum"][:, gi].sum())
        abs_w = float(S["bias_abs_wmu_sum"][:, gi].sum())
        bias.append(_safe_ratio(abs_b, abs_b + abs_w))
        bias_x.append(_safe_ratio(float(S["bias_structural_sum"][:, gi].sum()),
                                  float(S["bias_n"][:, gi].sum())))
    bias = np.asarray(bias)
    bias_x = np.asarray(bias_x)
    bias_b = np.column_stack([b["bias_share"] for b in boots])
    bias_x_b = np.column_stack([b["bias_struct"] for b in boots])
    strict = per_seed.groupby("alpha", sort=False)["final_strict_dead"].mean().to_numpy()
    strict_b = np.column_stack([b["final_strict_dead"] for b in boots])

    theta_slope = slope_1d(1.0 / dose, theta)
    theta_slope_b = row_slopes(1.0 / dose_b, theta_b)
    wall_slope = slope_1d(wall_mu, wall_y)
    wall_slope_b = row_slopes(wall_mu_b, wall_y_b)
    q_slope = slope_1d(wall_mu, wall_q)
    q_slope_b = row_slopes(wall_mu_b, wall_q_b)
    escape_slope = slope_1d(bias_x, bias)
    escape_slope_b = row_slopes(bias_x_b, bias_b)
    phenotype_slope = slope_1d(dose, strict)
    phenotype_slope_b = row_slopes(dose_b, strict_b)
    path_rows.extend([
        regression_record("theta_slope_b", "bulk", theta_slope, theta_slope_b,
                          "theta_med = a + b / pooled_median(mu_norm)"),
        regression_record("wall_total_slope", "bulk", wall_slope, wall_slope_b,
                          "mean_log(-cos_crit) vs mean_log(mu_norm), same q>0 mask"),
        regression_record("wall_q_numerator_slope", "bulk", q_slope, q_slope_b,
                          "mean_log(q) vs mean_log(mu_norm), same q>0 mask"),
        regression_record("wall_direct_denominator_component", "bulk", -1.0,
                          np.full(len(weights), -1.0), "algebraic coefficient of -log(mu_norm)"),
        regression_record("wall_slope_decomposition_error", "bulk",
                          wall_slope - (q_slope - 1.0),
                          wall_slope_b - (q_slope_b - 1.0),
                          "wall_total - (q_numerator - 1)"),
        regression_record("bias_escape_slope", "bulk", escape_slope, escape_slope_b,
                          "aggregate bias_share vs eligible-weighted 1/(1+mu_norm^2)"),
        regression_record("final_strict_dead_slope", "final", phenotype_slope,
                          phenotype_slope_b, "final strict_dead vs pooled median bulk mu_norm"),
    ])

    dose_rows = []
    for i, (arm, boot) in enumerate(zip(arms, boots)):
        bulk_theta = theta_df[(theta_df.alpha == arm["alpha"])
                              & (theta_df.scope == "bulk")]
        tm = bulk_theta[bulk_theta.estimate == "theta_med"].iloc[0]
        ta = bulk_theta[bulk_theta.estimate == "theta_all"].iloc[0]
        seed_arm = per_seed[per_seed.alpha == arm["alpha"]]
        def ci(name):
            lo, hi, nf = finite_ci(boot[name])
            return lo, hi, nf
        dlo, dhi, dnf = ci("dose")
        wylo, wyhi, wynf = ci("wall_y")
        bllo, blhi, blnf = ci("bias_share")
        sdlo, sdhi, sdnf = ci("final_strict_dead")
        evlo, evhi, evnf = ci("final_eval_loss_exact")
        dose_rows.append({
            "alpha": arm["alpha"], "arm": arm["arm"],
            "mu_norm": dose[i], "mu_norm_ci_lo": dlo, "mu_norm_ci_hi": dhi,
            "mu_norm_bootstrap_finite": dnf,
            "theta_med": float(tm.point), "theta_med_ci_lo": float(tm.ci_lo),
            "theta_med_ci_hi": float(tm.ci_hi),
            "theta_med_bootstrap_finite": int(tm.bootstrap_finite),
            "theta_all": float(ta.point), "theta_all_ci_lo": float(ta.ci_lo),
            "theta_all_ci_hi": float(ta.ci_hi),
            "theta_all_bootstrap_finite": int(ta.bootstrap_finite),
            "bulk_q_positive_cos_crit_median": arm["bulk_q_coscrit_median"],
            "theta_left_censored_exact": bool(tm.left_censored_exact),
            "wall_mean_log_neg_cos_crit": wall_y[i],
            "wall_ci_lo": wylo, "wall_ci_hi": wyhi,
            "wall_bootstrap_finite": wynf,
            "wall_mean_log_q": wall_q[i], "wall_mean_log_mu_norm": wall_mu[i],
            "wall_additive_identity_error": wall_y[i] - (wall_q[i] - wall_mu[i]),
            "bias_share_field": bias[i], "bias_share_ci_lo": bllo,
            "bias_share_ci_hi": blhi, "bias_share_bootstrap_finite": blnf,
            "bias_structural_reference": bias_x[i],
            "final_strict_dead": float(seed_arm.final_strict_dead.mean()),
            "final_strict_dead_ci_lo": sdlo, "final_strict_dead_ci_hi": sdhi,
            "final_strict_dead_bootstrap_finite": sdnf,
            "final_near_off": float(seed_arm.final_near_off.mean()),
            "final_dead_0_05": float(seed_arm.final_dead_0_05.mean()),
            "final_eval_loss_exact": float(seed_arm.final_eval_loss_exact.mean()),
            "final_eval_loss_ci_lo": evlo, "final_eval_loss_ci_hi": evhi,
            "final_eval_loss_bootstrap_finite": evnf,
        })
    dose_df = pd.DataFrame(dose_rows)

    seed_rho_theta = []
    seed_rho_bias = []
    for seed in range(10):
        ds = per_seed[per_seed.seed == seed].sort_values("alpha")
        seed_rho_theta.append(spearman(ds.mu_norm, ds.theta_med))
        seed_rho_bias.append(spearman(ds.bias_structural_reference, ds.bias_share_field))
    rho_theta_med = float(np.nanmedian(seed_rho_theta))
    rho_bias_med = float(np.nanmedian(seed_rho_bias))
    violations, pairs, violation_rate = order_violations(dose, theta)

    dose_ratio = float(np.max(dose) / np.min(dose))
    distinct_dose = int(len(np.unique(dose)))
    finite_theta = int(np.isfinite(theta).sum())
    C0 = dose_ratio >= 10.0 and distinct_dose >= 6 and finite_theta >= 4
    tlo, thi, tnf = finite_ci(theta_slope_b)
    wlo, whi, wnf = finite_ci(wall_slope_b)
    elo, ehi, enf = finite_ci(escape_slope_b)
    plo, phi, pnf = finite_ci(phenotype_slope_b)
    finite_theta_mask = np.isfinite(theta)
    lowdose_cutoff = (float(np.min(dose[finite_theta_mask]))
                      if finite_theta_mask.any() else float(np.median(dose)))
    lowdose_na = (dose < lowdose_cutoff) & ~finite_theta_mask
    censored_ok = bool(all(
        arms[i]["bulk_q_coscrit_median"] < -1.0 for i in np.flatnonzero(lowdose_na)
    ))
    if not C0:
        c1_status = "VOID"
    elif theta_slope < 0 and thi < 0 and rho_theta_med >= 0.6:
        c1_status = "PASS"
    elif tlo > 0:
        c1_status = "FAIL"
    else:
        c1_status = "INCONCLUSIVE"
    if whi < 0 and censored_ok:
        w1_status = "PASS"
    elif wlo > 0:
        w1_status = "FAIL"
    else:
        w1_status = "INCONCLUSIVE"
    if elo > 0 and rho_bias_med >= 0.6:
        c2_status = "PASS"
    elif ehi < 0:
        c2_status = "FAIL"
    else:
        c2_status = "INCONCLUSIVE"
    if not C0:
        overall = "VOID"
    elif c1_status == "PASS" and w1_status == "PASS" and c2_status == "PASS":
        overall = "FULL_PASS"
    elif (c1_status == "PASS") ^ (c2_status == "PASS"):
        overall = "PARTIAL"
    elif c1_status == "FAIL" and c2_status == "FAIL":
        overall = "FAIL"
    else:
        overall = "INCONCLUSIVE"
    nonmono = math.isfinite(violation_rate) and violation_rate > 0.20
    verdict_rows = [
        {
            "id": "C0", "question": "dose and theta identifiable", "status": "PASS" if C0 else "FAIL",
            "estimate": dose_ratio, "ci_lo": math.nan, "ci_hi": math.nan,
            "bootstrap_finite": 0,
            "detail": f"dose_ratio={dose_ratio:.6g}; distinct_dose={distinct_dose}; finite_theta={finite_theta}",
        },
        {
            "id": "C1", "question": "theta moves with measured mu_norm", "status": c1_status,
            "estimate": theta_slope, "ci_lo": tlo, "ci_hi": thi,
            "bootstrap_finite": tnf,
            "detail": (f"median seed Spearman={rho_theta_med:.6g}; order violations="
                       f"{violations}/{pairs} ({violation_rate:.6g}); nonmonotonic={nonmono}"),
        },
        {
            "id": "W1", "question": "exact wall is coherent", "status": w1_status,
            "estimate": wall_slope, "ci_lo": wlo, "ci_hi": whi,
            "bootstrap_finite": wnf,
            "detail": (f"low-dose NA arms={int(lowdose_na.sum())}; exact-censored={censored_ok}; "
                       f"q slope={q_slope:.6g}; denominator=-1"),
        },
        {
            "id": "C2", "question": "bias escape returns", "status": c2_status,
            "estimate": escape_slope, "ci_lo": elo, "ci_hi": ehi,
            "bootstrap_finite": enf,
            "detail": f"median seed Spearman={rho_bias_med:.6g}",
        },
        {
            "id": "P1", "question": "phenotype direction", "status": "REPORT_ONLY",
            "estimate": phenotype_slope, "ci_lo": plo, "ci_hi": phi,
            "bootstrap_finite": pnf,
            "detail": "final strict_dead vs measured bulk mu_norm; does not override C1/C2",
        },
        {
            "id": "OVERALL", "question": "pre-registered combined verdict", "status": overall,
            "estimate": math.nan, "ci_lo": math.nan, "ci_hi": math.nan,
            "bootstrap_finite": 0,
            "detail": f"C0={'PASS' if C0 else 'FAIL'}; C1={c1_status}; W1={w1_status}; C2={c2_status}",
        },
    ]
    return {
        "gate_curve": pd.DataFrame(curve_rows),
        "theta_estimates": theta_df,
        "dose_response": dose_df,
        "path_decomposition": pd.DataFrame(path_rows),
        "per_seed_metrics": per_seed,
        "verdict": pd.DataFrame(verdict_rows),
        "seed_rho_theta": seed_rho_theta,
        "seed_rho_bias": seed_rho_bias,
        "order_violations": {
            "n_violations": violations, "n_pairs": pairs, "rate": violation_rate,
            "nonmonotonic": nonmono,
        },
    }


def arm_manifest(arms: Sequence[dict], s6: dict, sweep_provenance: dict) -> pd.DataFrame:
    rows = []
    for arm in arms:
        sanity = arm["meta"]["sanity"]
        rows.append({
            "alpha": arm["alpha"], "arm": arm["arm"],
            "arm_dir": repo_path(arm["arm_dir"]), "status": "complete",
            "n_seeds": 10, "n_record_steps": arm["meta"]["n_record_steps"],
            "width": arm["meta"]["width"], "n_realized_flips": arm["meta"]["n_realized_flips"],
            "S1": sanity["S1"]["s1_pass"], "S2": sanity["S2"]["s2_pass"],
            "S3": sanity["S3"]["s3_pass"], "S4": sanity["S4"]["s4_pass"],
            "S5": sanity["S5"]["s5_pass"], "S6a": s6["S6a_pass"],
            "S6b": s6["S6b_pass"] if arm["alpha"] in (0.0, 0.01) else math.nan,
            "S7": sanity["S7"]["s7_pass"], "all_pass": True,
            "step0_repro_hash_sha256": s6["step0_repro_hash_sha256"],
            "git_clean": arm["provenance"]["git_clean"],
            "git_source_clean": arm["provenance"]["git_source_clean"],
            "sweep_commit": sweep_provenance["sweep_commit"],
            "sweep_fingerprint": sweep_provenance["sweep_fingerprint"],
            "config_sha256": sweep_provenance["config_sha256"],
            "spec_sha256": sweep_provenance["spec_sha256"],
            "source_sha256_json": json.dumps(
                sweep_provenance["source_sha256"], sort_keys=True,
                separators=(",", ":"),
            ),
            "provenance_sha256": sweep_provenance["arm_provenance_sha256"][arm["arm"]],
        })
    return pd.DataFrame(rows)


def raw_manifest(arms: Sequence[dict], cfg: dict) -> pd.DataFrame:
    rows = []
    for arm in arms:
        files = [arm["arm_dir"] / "config_used.yaml", arm["arm_dir"] / "meta.json",
                 arm["arm_dir"] / "arm_meta.json", arm["arm_dir"] / "provenance.json"]
        files += [arm["arm_dir"] / "logs" / f"seed{s}.npz" for s in range(10)]
        for path in files:
            rows.append({
                "alpha": arm["alpha"], "arm": arm["arm"],
                "kind": "log" if path.suffix == ".npz" else path.stem,
                "path": repo_path(path), "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    for alpha_text, reference in sorted(
            cfg["mu_titration"].get("endpoint_references", {}).items(),
            key=lambda item: float(item[0])):
        refdir = Path(reference)
        if not refdir.is_absolute():
            refdir = ROOT / refdir
        for seed in range(10):
            path = refdir / "logs" / f"seed{seed}.npz"
            # validate_s6 has already required and read these exact files.
            rows.append({
                "alpha": float(alpha_text),
                "arm": f"endpoint_reference_alpha_{alpha_token(float(alpha_text))}",
                "kind": "endpoint_reference_log", "path": repo_path(path),
                "bytes": path.stat().st_size, "sha256": sha256_file(path),
            })
    return pd.DataFrame(rows)


def make_figures(outdir: Path, results: dict) -> None:
    fdir = outdir / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    dose = results["dose_response"].sort_values("mu_norm")
    x = dose.mu_norm.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    y = dose.theta_med.to_numpy(dtype=float)
    lo, hi = dose.theta_med_ci_lo.to_numpy(float), dose.theta_med_ci_hi.to_numpy(float)
    good = np.isfinite(y)
    ax.errorbar(x[good], y[good],
                yerr=[np.maximum(y[good] - lo[good], 0.0),
                      np.maximum(hi[good] - y[good], 0.0)],
                fmt="o-", capsize=3)
    ax.set(xlabel="measured bulk mu_norm", ylabel="theta_med",
           title="Observed switch-off point vs measured dose")
    ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(fdir / "fig_theta_dose.png", dpi=150,
                metadata={"Software": "mu_titration_0823"}); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.plot(dose.wall_mean_log_mu_norm, dose.wall_mean_log_neg_cos_crit, "o-",
            label="log wall")
    ax.plot(dose.wall_mean_log_mu_norm, dose.wall_mean_log_q, "s--", label="log q")
    ax.set(xlabel="mean log(mu_norm), q>0 mask", ylabel="mean log quantity",
           title="Exact wall decomposition")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(fdir / "fig_wall_decomposition.png", dpi=150,
                metadata={"Software": "mu_titration_0823"}); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.plot(dose.bias_structural_reference, dose.bias_share_field, "o-")
    ax.set(xlabel="eligible-weighted 1/(1+mu_norm^2)", ylabel="aggregate bias share",
           title="Bias escape field")
    ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(fdir / "fig_bias_escape.png", dpi=150,
                metadata={"Software": "mu_titration_0823"}); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.plot(x, dose.final_strict_dead, "o-", label="strict_dead")
    ax.plot(x, dose.final_near_off, "s--", label="near_off")
    ax.plot(x, dose.final_dead_0_05, "^:", label="dead_0.05")
    ax.set(xlabel="measured bulk mu_norm", ylabel="final fraction",
           title="Final phenotypes")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(fdir / "fig_final_phenotype.png", dpi=150,
                metadata={"Software": "mu_titration_0823"}); plt.close(fig)

    curve = results["gate_curve"]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    centers = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
    for alpha, d in curve[curve.scope == "bulk"].groupby("alpha", sort=True):
        d = d.sort_values("bin_index")
        valid = d.valid.to_numpy(bool)
        ax.plot(centers[valid], d.p_hat_median.to_numpy(float)[valid], marker=".",
                lw=1, label=f"alpha={alpha:g}")
    ax.set(xlabel="cos(u, mu)", ylabel="median p_hat", title="Bulk gate curves")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(fdir / "fig_gate_curves_bulk.png", dpi=150,
                metadata={"Software": "mu_titration_0823"}); plt.close(fig)


def fmt(x: float, digits: int = 4) -> str:
    return "NA" if not math.isfinite(float(x)) else f"{float(x):.{digits}g}"


def make_summary(outdir: Path, results: dict, cfg_path: Path, s6: dict) -> None:
    verdict = results["verdict"]
    dose = results["dose_response"]
    overall = verdict[verdict.id == "OVERALL"].iloc[0].status
    lines = [
        "# mu_titration_0823 analysis", "",
        f"Overall preregistered verdict: **{overall}**.", "",
        "This summary uses measured `mu_norm` as dose. `center_alpha` is an EMA update rate, not a partial-subtraction fraction.", "",
        "## Sanity and provenance", "",
        "- Source S1--S5 and S7 passed for all eight arms; a failed arm is never silently dropped.",
        "- S6a passed: step-0 reproducibility hashes, logged step-0 statistics, and complete flip trajectories agree across arms.",
        "- S6b passed: alpha=0 and alpha=.01 common columns are bit-equal to the preregistered endpoint references.",
        f"- Specification: `{repo_path(SPEC)}`; config: `{repo_path(cfg_path)}`.",
        "- All bootstrap estimates use one shared set of seed-bundle weights (B=10,000, RNG seed 20260823).", "",
        "## Verdicts", "",
        "| ID | status | estimate | 95% CI | detail |", "|---|---|---:|---:|---|",
    ]
    for _, row in verdict.iterrows():
        ci = "NA" if not math.isfinite(float(row.ci_lo)) else f"[{fmt(row.ci_lo)}, {fmt(row.ci_hi)}]"
        lines.append(f"| {row.id} | **{row.status}** | {fmt(row.estimate)} | {ci} | {row.detail} |")
    lines += ["", "## Bulk dose response", "",
              "| alpha | mu_norm | theta_med | wall log | bias share | strict_dead |",
              "|---:|---:|---:|---:|---:|---:|"]
    for _, row in dose.sort_values("alpha").iterrows():
        lines.append(
            f"| {row.alpha:g} | {fmt(row.mu_norm)} | {fmt(row.theta_med)} | "
            f"{fmt(row.wall_mean_log_neg_cos_crit)} | {fmt(row.bias_share_field)} | "
            f"{fmt(row.final_strict_dead)} |"
        )
    lines += [
        "", "## Scope and interpretation", "",
        "- Primary scope is `bulk`: 1,000-step grid points more than 100 steps from scheduled boundaries (901 points).",
        "- Secondary outputs cover realized-boundary offsets -100..+100, all-recorded, bulk time halves, and fixed phase offset +5000.",
        "- `theta_all`, final `near_off`, and final `dead_0.05` are secondary/reporting quantities. Unqualified `dead` is not used.",
        "- A missing theta is not called absence of a wall; `left_censored_exact` records domain-left-censoring when median exact `cos_crit < -1`.",
        "- Wall and bias-field paths are intermediate mechanisms, not independent causal mediation proportions for `strict_dead`.",
        "- No inference is extended to condB or other widths, periods, batches, or learning rates.", "",
        "## Files", "",
        "`arm_manifest.csv`, `raw_sha256.csv`, `gate_curve.csv`, `theta_estimates.csv`, `dose_response.csv`, `path_decomposition.csv`, `per_seed_metrics.csv`, `verdict.csv`, `analysis_meta.json`, `determinism_check.md`, and `figures/`.", "",
    ]
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_determinism_manifest(outdir: Path) -> None:
    names = [
        "arm_manifest.csv", "raw_sha256.csv", "gate_curve.csv",
        "theta_estimates.csv", "dose_response.csv", "path_decomposition.csv",
        "per_seed_metrics.csv", "verdict.csv", "summary.md", "analysis_meta.json",
    ]
    fig_names = sorted((outdir / "figures").glob("*.png"))
    lines = [
        "# Determinism manifest", "",
        "All scientific outputs omit timestamps and elapsed time. Re-run from the same commit and raw inputs in another `--outdir`, then compare the hashes below.", "",
        "| file | sha256 |", "|---|---|",
    ]
    for name in names:
        path = outdir / name
        lines.append(f"| {name} | `{sha256_file(path)}` |")
    for path in fig_names:
        lines.append(f"| figures/{path.name} | `{sha256_file(path)}` |")
    lines.append("")
    (outdir / "determinism_check.md").write_text("\n".join(lines), encoding="utf-8")


def self_test() -> None:
    # Histogram median and both theta definitions, including an invalid-bin gap.
    hist = np.zeros((N_BINS, P_LEVELS), dtype=np.int64)
    hist[:8, 0] = 1_001
    hist[8, 1] = 1_001
    assert np.isclose(theta_one(hist, True), BIN_UPPER[7])
    assert np.isclose(theta_one(hist, False), BIN_UPPER[7])
    hist[2, 1] = 1
    assert np.isclose(theta_one(hist, True), BIN_UPPER[7])
    assert np.isclose(theta_one(hist, False), BIN_UPPER[1])

    # Bootstrap theta uses the same non-negative-lattice median equivalence.
    seed_hist = np.zeros((3, N_BINS, P_LEVELS), dtype=np.int64)
    seed_hist[:, :5, 0] = 400
    seed_hist[:, 5:, 2] = 400
    weights = np.asarray([[1, 1, 1], [3, 0, 0]], dtype=np.int16)
    bt = bootstrap_theta_scope(seed_hist, weights, False)
    assert np.allclose(bt["theta_med"], BIN_UPPER[4])

    # Weighted medians preserve seed-bundle duplication exactly.
    vals = [np.asarray([0.0, 2.0]), np.asarray([10.0, 12.0])]
    wm = bootstrap_weighted_median(vals, np.asarray([[1, 1], [2, 0]], dtype=np.int16))
    assert np.allclose(wm, [6.0, 1.0])

    # Exact wall identity and regression decomposition.
    mu = np.asarray([0.25, 0.5, 1.0, 2.0])
    q = mu ** 0.3
    y = np.log(q / mu)
    x = np.log(mu)
    sy, sq = slope_1d(x, y), slope_1d(x, np.log(q))
    assert abs(sy - (sq - 1.0)) < 1e-14
    assert abs(sy + 0.7) < 1e-14

    # Direction conventions for C1/C2 and average-rank Spearman.
    dose = np.asarray([0.1, 0.2, 0.4, 0.8])
    theta = -1.0 / dose
    assert slope_1d(1.0 / dose, theta) < 0
    assert spearman(dose, theta) == 1.0
    ref = 1.0 / (1.0 + dose * dose)
    assert slope_1d(ref, ref * 0.5) > 0

    # A separately rounded float32 s/M pair may land exactly on zero although
    # the pre-save float64 field was positive.  This is diagnostic, not a
    # contradiction of the source runner's float64 S5b PASS.
    saved_s, saved_M = np.float32(-0.99999998), np.float32(1.0)
    saved_field = float(saved_s + saved_M)
    assert saved_field == 0.0 and not (saved_field <= -FLOAT32_ATOL)

    # Shared weights are deterministic and contain exactly nseed draws per row.
    a = make_bootstrap_weights(100, 10, 20260823)
    b = make_bootstrap_weights(100, 10, 20260823)
    assert np.array_equal(a, b) and np.all(a.sum(axis=1) == 10)
    print("mu_titration synthetic self-test: PASS", flush=True)


def run_analysis(config_path: Path, arms_dir: Path, outdir: Path,
                 allow_dirty_analysis: bool = False) -> None:
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise AnalysisError("OMP_NUM_THREADS=1 is required for the canonical analysis")
    if not config_path.is_file() or not SPEC.is_file():
        raise AnalysisError("canonical config/spec is missing")
    if allow_dirty_analysis and config_path.resolve() == DEFAULT_CONFIG.resolve():
        raise AnalysisError("--allow-dirty-analysis is forbidden for the canonical config")
    analysis_provenance = analysis_git_provenance(allow_dirty_analysis)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_canonical_config(cfg)
    total = int(cfg["common"]["total_steps"])
    period = int(cfg["condA"]["T_values"][0])
    expected_steps = expected_record_steps(
        total, period, int(cfg["ratchet"]["boundary_window"]),
        int(cfg["ratchet"]["bulk_every"]),
    )
    if len(expected_steps) != 20_901:
        raise AnalysisError("internal canonical record grid is not 20,901 points")
    alphas = [float(x) for x in cfg["mu_titration"]["center_alphas"]]

    arms = []
    for alpha in alphas:
        arm_dir = arms_dir / f"alpha_{alpha_token(alpha)}"
        print(f"validating and reducing {arm_dir.name} ...", flush=True)
        arms.append(summarize_arm(alpha, arm_dir, cfg, expected_steps))
    sweep_provenance = validate_sweep_provenance(arms, config_path)
    s6 = validate_s6(arms, cfg)
    raw = raw_manifest(arms, cfg)

    B = int(cfg["ratchet"]["bootstrap_B"])
    boot_seed = int(cfg["ratchet"]["bootstrap_seed"])
    weights = make_bootstrap_weights(B, 10, boot_seed)
    print("running shared-weight paired bootstrap ...", flush=True)
    results = derive_results(arms, weights)

    # No output directory is created until every strict source/sanity check passes.
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = arm_manifest(arms, s6, sweep_provenance)
    write_csv(manifest, outdir / "arm_manifest.csv")
    write_csv(raw, outdir / "raw_sha256.csv")
    for key in ("gate_curve", "theta_estimates", "dose_response",
                "path_decomposition", "per_seed_metrics", "verdict"):
        write_csv(results[key], outdir / f"{key}.csv")
    make_figures(outdir, results)
    make_summary(outdir, results, config_path, s6)

    meta = {
        "schema_version": 1,
        "analysis": "mu_titration_0823",
        "git_commit": git_commit(),
        "analysis_provenance": analysis_provenance,
        "run_sweep_provenance": sweep_provenance,
        "spec": repo_path(SPEC), "spec_sha256": sha256_file(SPEC),
        "config": repo_path(config_path), "config_sha256": sha256_file(config_path),
        "analysis_file": repo_path(Path(__file__)),
        "analysis_file_sha256": sha256_file(Path(__file__)),
        "arms_dir": repo_path(arms_dir),
        "input_manifest_sha256": sha256_bytes(
            raw.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")
        ),
        "inputs": raw[["path", "sha256", "bytes"]].to_dict("records"),
        "alphas": alphas, "seeds": list(range(10)),
        "n_record_steps": len(expected_steps), "bulk_record_steps": 901,
        "scopes": arms[0]["scope_names"],
        "cos_range": [COS_LO, COS_HI], "bin_width": BIN_WIDTH,
        "min_bin_n": MIN_BIN_N,
        "bootstrap_B": B, "bootstrap_seed": boot_seed,
        "bootstrap_weights_sha256": sha256_bytes(weights.tobytes()),
        "bootstrap_pairing": "same seed-count weights for every alpha and every estimate",
        "sanity": {
            "source_all_S1_to_S5_S7_pass": True,
            "source_all_clean_same_preregistered_sweep": True,
            "S6": s6,
            "saved_float32_rechecks": {
                arm["arm"]: arm["diagnostics"] for arm in arms
            },
        },
        "seed_spearman_mu_theta": results["seed_rho_theta"],
        "seed_spearman_structural_bias_share": results["seed_rho_bias"],
        "theta_order_violations": results["order_violations"],
        "verdict": results["verdict"].to_dict("records"),
        "determinism": {
            "timestamps_in_scientific_outputs": False,
            "elapsed_time_in_scientific_outputs": False,
            "rerun_manifest": "determinism_check.md",
        },
        "runtime_versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
        },
    }
    write_json(outdir / "analysis_meta.json", meta)
    write_determinism_manifest(outdir)
    print(results["verdict"].to_string(index=False), flush=True)
    print(f"mu titration analysis complete -> {outdir}", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULTS,
                        help="raw result base containing arms/ (also the default output directory)")
    parser.add_argument("--arms-dir", type=Path, default=None,
                        help="direct source-arms override, mainly for validation/tests")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="derived-output directory (default: --result-dir)")
    parser.add_argument("--self-test", action="store_true",
                        help="run synthetic tests only; do not inspect real results")
    parser.add_argument(
        "--allow-dirty-analysis", action="store_true",
        help="development-only override for synthetic end-to-end tests; canonical analysis rejects it",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    result_dir = args.result_dir if args.result_dir.is_absolute() else ROOT / args.result_dir
    raw_outdir = args.outdir if args.outdir is not None else result_dir
    outdir = raw_outdir if raw_outdir.is_absolute() else ROOT / raw_outdir
    if args.arms_dir is None:
        arms_dir = result_dir / "arms"
    else:
        arms_dir = args.arms_dir if args.arms_dir.is_absolute() else ROOT / args.arms_dir
    run_analysis(config_path.resolve(), arms_dir.resolve(), outdir.resolve(),
                 allow_dirty_analysis=args.allow_dirty_analysis)


if __name__ == "__main__":
    main()

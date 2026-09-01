"""channel_2x2_0901: b-WD x layer-input centering の 2x2 本走。

事前登録: ``specs/spec_channel_2x2_0901.md`` (commit 31f3792)。
既存の Phase-1 runner、bias-WD 厳密レコーダ、seed 隔離を合成し、
このファイルはゲート、4 セル集計、登録済み決定木だけを担う。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from .bias_wd_0901 import paired_ci, s1_s2_algebra
from .bias_wd_common import (
    TaskEndRecorder,
    exact_wall_record,
    markdown_table,
    provenance,
    require_omp,
    write_arm_npz,
)
from .bias_wd_std_seediso_0901 import (
    ARM_INVALID_EXCLUSION_LIMIT,
    COMPLETE,
    COMPLETE_WITH_EXCLUSIONS,
    ExclusionLimitExceeded,
    SeedIsolationRecorder,
)
from .common import ROOT, load_config
from .mlp2_phase0 import identity_sanity_pass
from .mlp2_phase1 import (
    _base_cfg,
    exact_layer_record_p1,
    forward_centered,
    grads_centered,
    setup_arm_p1,
    train_arm_p1,
)


CONFIG = Path(ROOT) / "configs" / "channel_2x2_0901.yaml"
ARM_ORDER = ("none", "bwd", "cen", "both")
VALID_ARM_STATUSES = {COMPLETE, COMPLETE_WITH_EXCLUSIONS}
CONTRAST_INVALID_TOO_FEW_PAIRED = "CONTRAST_INVALID_TOO_FEW_PAIRED"
E_DRIFT_INVALID_FLOOR = "E_DRIFT_INVALID_FLOOR"


def _P(cfg: dict) -> dict:
    return cfg["channel_2x2"]


def _compat_cfg(cfg: dict) -> dict:
    """bias_wd の凍結済み CI/S1-S2 helper に section 名だけ適合させる。"""
    out = dict(cfg)
    out["bias_wd"] = _P(cfg)
    return out


def _arm_cfg(cfg: dict, name: str) -> dict:
    return next(arm for arm in cfg["arms"] if arm["name"] == name)


def arm_lambda(cfg: dict, name: str) -> float:
    return float(_arm_cfg(cfg, name)["wd_b"])


def outdir_of(cfg: dict) -> Path:
    return Path(ROOT) / _P(cfg)["output_dir"]


def validate_config(cfg: dict, *, full: bool) -> None:
    C, A, I, P = cfg["common"], cfg["condA"], cfg["intervention"], _P(cfg)
    if cfg.get("spec") != "specs/spec_channel_2x2_0901.md":
        raise ValueError("registered spec path differs")
    if (C.get("device"), float(C["lr_main"]), int(C["generator_offset"])) != (
            "cpu", 0.01, 20_260_905):
        raise ValueError("registered device/lr/generator_offset differs")
    if (int(A["m"]), int(A["f"]), int(A["target_hidden"]), float(A["beta"])) != (
            20, 15, 100, 0.7):
        raise ValueError("registered condA differs")
    if list(A["T_values"]) != [10_000] or list(A["encodings"]) != ["std"]:
        raise ValueError("registered T/encoding differs")
    if (I["name"], float(I["center_alpha"]), I["stop_gradient_on_running_mean"],
            I["consumes_rng"]) != (
            "A_layer_input_centering", 0.01, True, False):
        raise ValueError("registered centering intervention differs")
    expected = [
        ("none", [], 0.0), ("bwd", [], 1e-3),
        ("cen", [1, 2], 0.0), ("both", [1, 2], 1e-3),
    ]
    got = [(a["name"], list(a.get("centered_layers") or []), float(a["wd_b"]))
           for a in cfg["arms"]]
    if got != expected or any(list(a["hidden"]) != [100, 100] for a in cfg["arms"]):
        raise ValueError(f"registered arms differ: {got}")
    if cfg["pairing"] != {"paired_groups": [list(ARM_ORDER)], "baseline": "none"}:
        raise ValueError("registered pairing differs")
    if bool(C.get("run_freeze")) and any(float(a["wd_b"]) > 0 for a in cfg["arms"]):
        raise ValueError("freeze_bias=true and wd_b>0 cannot be combined")
    iso = P["seed_isolation"]
    if (int(iso["max_exclusions_per_arm"]), int(iso["min_complete_seeds_per_arm"]),
            int(iso["min_paired_seeds"]), int(iso["keep_rng_rows"])) != (2, 8, 8, 10):
        raise ValueError("registered seed-isolation limits differ")
    if (list(P["early_block_tasks"]), list(P["late_block_tasks"]),
            int(P["block_tasks"])) != ([51, 100], [451, 500], 50):
        raise ValueError("registered blocks differ")
    if (float(P["equivalence_margin"]), float(P["interaction_margin"]),
            float(P["ceiling_flag_dex"]), float(P["eff_rank_keep_frac"]),
            float(P["unfit_floor"])) != (0.15, 0.50, 3.0, 0.70, 1e-23):
        raise ValueError("registered thresholds/floor differ")
    if (int(P["bootstrap_B"]), int(P["bootstrap_seed"]), P["ci_method"]) != (
            20_000, 20_260_906, "percentile_paired"):
        raise ValueError("registered bootstrap differs")
    if (int(P["s0_reference_offset"]), int(P["s0_replay_steps"])) != (0, 30_000):
        raise ValueError("registered S0 differs")
    if full and (int(C["total_steps"]) != 5_000_000
                 or list(C["seeds"]) != list(range(10))):
        raise ValueError("full run must be 5M and seeds 0..9")


def _setup(cfg: dict, arm_name: str, *, generator_offset: int | None = None) -> dict:
    work = copy.deepcopy(cfg)
    if generator_offset is not None:
        work["common"]["generator_offset"] = int(generator_offset)
    st = setup_arm_p1(_base_cfg(work), _arm_cfg(work, arm_name), "cpu")
    st["net"].set_weight_decay_b(arm_lambda(work, arm_name))
    _, sanity = exact_layer_record_p1(st, float(work["phase1"]["sigma_degenerate_tol"]))
    if not identity_sanity_pass(sanity, float(work["sanity"]["s1_identity_tol"])):
        raise RuntimeError(f"{arm_name}: preflight identity failed")
    return st


def _median(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _boundary_snapshot(st: dict, arm: str, step: int, sigma_tol: float) -> list[dict]:
    layers, _ = exact_wall_record(st, sigma_tol)
    rows: list[dict] = []
    for ri, run in enumerate(st["runs"]):
        row = dict(arm=arm, seed=int(run["seed"]), step=int(step),
                   task_boundary=int(step // int(run["period"])), side="post")
        for li, layer in enumerate(layers, start=1):
            p = layer["p_hat"][ri].detach().cpu().numpy()
            sigma = layer["sigma"][ri].detach().cpu().numpy()
            beta = layer["beta"][ri].detach().cpu().numpy()
            b = layer["b"][ri].detach().cpu().numpy()
            valid = layer["valid"][ri].detach().cpu().numpy() & np.isfinite(beta)
            alive = (p > 0) & valid
            B = b / np.where(sigma > 0, sigma, np.nan)
            M = beta - B
            prefix = f"L{li}_"
            row[prefix + "strict_dead_frac"] = float((p == 0).mean())
            row[prefix + "submerged_frac"] = float(
                (layer["pre_max"][ri].detach().cpu().numpy() <= 0).mean())
            row[prefix + "M_median_alive"] = _median(M[alive])
            row[prefix + "B_median_alive"] = _median(B[alive])
            row[prefix + "sigma_median_alive"] = _median(sigma[alive])
        rows.append(row)
    return rows


def run_arm_seediso(cfg: dict, arm_name: str, outdir: Path, *, total_steps: int,
                    task_period: int, guard_every: int,
                    keep_unit_arrays: bool = True, write_logs: bool = True,
                    record_boundaries: bool = True,
                    generator_offset: int | None = None) -> dict:
    st = _setup(cfg, arm_name, generator_offset=generator_offset)
    P = _P(cfg)
    cap = int(P["seed_isolation"]["max_exclusions_per_arm"])
    record_steps = list(range(0, total_steps + 1, task_period))
    guard_steps = list(range(0, total_steps + 1, guard_every))
    recorder = SeedIsolationRecorder(
        arm_name, arm_lambda(cfg, arm_name), st,
        record_steps=record_steps, guard_steps=guard_steps,
        guard_every=guard_every, exclusion_cap=cap,
        status_dir=outdir / "seed_status",
        sigma_tol=float(cfg["phase1"]["sigma_degenerate_tol"]),
        identity_tol=float(cfg["sanity"]["s1_identity_tol"]),
        keep_unit_arrays=keep_unit_arrays,
    )
    probes = sorted(set(record_steps) | set(guard_steps))
    checkpoints = [int(value) for value in cfg["common"].get("checkpoints", [])
                   if int(value) <= total_steps]
    boundary_rows: list[dict] = []

    def stream_hook(t: int, _x: torch.Tensor, _y: torch.Tensor) -> None:
        if record_boundaries and t > 0 and t % task_period == 0:
            boundary_rows.extend(_boundary_snapshot(
                st, arm_name, t, float(cfg["phase1"]["sigma_degenerate_tol"])))

    print(f"[{arm_name}] seed-isolated centered={_arm_cfg(cfg, arm_name)['centered_layers']} "
          f"wd_b={arm_lambda(cfg, arm_name):g} offset="
          f"{cfg['common']['generator_offset'] if generator_offset is None else generator_offset} "
          f"steps={total_steps:,}", flush=True)
    started = time.time()
    limit_event = None
    try:
        elapsed = train_arm_p1(st, recorder, probes, total_steps, outdir,
                               checkpoints, stream_hook=stream_hook)
    except ExclusionLimitExceeded as exc:
        elapsed = time.time() - started
        limit_event = exc.event
    excluded = sorted(recorder.excluded)
    included = [int(run["seed"]) for run in st["runs"]
                if int(run["seed"]) not in recorder.excluded]
    status = (ARM_INVALID_EXCLUSION_LIMIT if limit_event is not None else
              COMPLETE_WITH_EXCLUSIONS if excluded else COMPLETE)
    raw = recorder.dataframe()
    frame = raw[~raw.seed.isin(excluded)].copy()
    boundaries = pd.DataFrame(boundary_rows)
    if not boundaries.empty:
        boundaries = boundaries[~boundaries.seed.isin(excluded)].copy()
    sanity = recorder.sanity()
    if write_logs and limit_event is None:
        write_arm_npz(outdir, arm_name, arm_lambda(cfg, arm_name), st, recorder)
    result = dict(
        arm=arm_name, wd_b=arm_lambda(cfg, arm_name), status=status,
        elapsed_sec=float(elapsed), excluded_seeds=excluded, included_seeds=included,
        exclusion_events=[recorder.excluded[seed] for seed in excluded],
        exclusion_cap=cap, limit_event=limit_event, sanity=sanity,
        frame=frame, boundary_frame=boundaries,
    )
    status_path = outdir / "arm_status" / f"{arm_name}.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(
        {key: value for key, value in result.items()
         if key not in {"frame", "boundary_frame"}},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{arm_name}] {status} in {elapsed:.1f}s; excluded={excluded}; "
          f"sanity={'PASS' if sanity['pass_'] else 'FAIL'}", flush=True)
    return result


def _sha_tensor(value: torch.Tensor) -> str:
    data = np.ascontiguousarray(value.detach().cpu().numpy()).tobytes()
    return hashlib.sha256(data).hexdigest()


def _initial_hashes(st: dict) -> dict:
    out = {f"W{li}": _sha_tensor(value) for li, value in enumerate(st["net"].Ws, 1)}
    out.update({f"b{li}": _sha_tensor(value) for li, value in enumerate(st["net"].bs, 1)})
    out.update(v=_sha_tensor(st["net"].v), c=_sha_tensor(st["net"].c),
               env_flip_state=_sha_tensor(st["env"].flip_state),
               eval_fixed=_sha_tensor(st["eval_fixed"]))
    out.update({f"teacher_{key}": _sha_tensor(value)
                for key, value in st["teacher"].state_dict().items()})
    out.update({f"generator_{key}": _sha_tensor(value.get_state())
                for key, value in st["gens"].items()})
    return out


def s_pair_gate(cfg: dict, gate_dir: Path) -> tuple[dict, dict]:
    steps, grid = 30_000, 1_000
    reports: dict[str, dict] = {}
    for arm in ARM_ORDER:
        st = _setup(cfg, arm)
        initial = _initial_hashes(st)
        recorder = TaskEndRecorder(
            arm, arm_lambda(cfg, arm), st, record_steps=[0, steps], guard_steps=[],
            guard_every=grid, sigma_tol=float(cfg["phase1"]["sigma_degenerate_tol"]),
            identity_tol=float(cfg["sanity"]["s1_identity_tol"]),
            keep_unit_arrays=False)
        recorder(st, 0)
        stream = []
        for t in range(steps):
            x = st["env"].step()
            y = st["teacher"](x)
            if t % grid == 0:
                stream.append(dict(step=t, x=_sha_tensor(x), y=_sha_tensor(y)))
            inputs, pres, acts, yhat = forward_centered(st, x)
            grads = grads_centered(st["net"], inputs, pres, acts, yhat - y)
            st["net"].sgd_step_layers(st["lr"], *grads)
        recorder(st, steps)
        reports[arm] = dict(
            initial=initial, stream=stream, env_t=int(st["env"].t),
            env_flip_state=_sha_tensor(st["env"].flip_state),
            generator_after={key: _sha_tensor(value.get_state())
                             for key, value in st["gens"].items()},
            s3=recorder.sanity(),
        )
    baseline = reports["none"]
    pair_differences = {}
    for arm in ARM_ORDER[1:]:
        differences = []
        for field in ("initial", "stream", "env_t", "env_flip_state", "generator_after"):
            if reports[arm][field] != baseline[field]:
                differences.append(field)
        pair_differences[arm] = differences
    pair = dict(
        pass_=bool(all(not value for value in pair_differences.values())),
        steps=steps, grid=grid, compared=list(ARM_ORDER),
        differences=pair_differences,
        initial_hashes=baseline["initial"],
        stream_hashes=baseline["stream"],
    )
    s3 = dict(pass_=bool(all(reports[arm]["s3"]["pass_"] for arm in ARM_ORDER)),
              arms={arm: reports[arm]["s3"] for arm in ARM_ORDER})
    gate_dir.mkdir(parents=True, exist_ok=True)
    (gate_dir / "s_pair.json").write_text(
        json.dumps(pair, indent=2, ensure_ascii=False), encoding="utf-8")
    (gate_dir / "s3.json").write_text(
        json.dumps(s3, indent=2, ensure_ascii=False), encoding="utf-8")
    if not pair["pass_"] or not s3["pass_"]:
        raise RuntimeError(f"S-pair/S3 failed: pair={pair['pass_']} S3={s3['pass_']}")
    print("S-pair/S3: PASS", flush=True)
    return pair, s3


def s0_replay(cfg: dict, gate_dir: Path) -> dict:
    P = _P(cfg)
    steps = int(P["s0_replay_steps"])
    pairs = {"none": P["s0_baseline_arm_std"], "cen": P["s0_baseline_arm_cen"]}
    replay = copy.deepcopy(cfg)
    replay["common"]["checkpoints"] = []
    report, all_ok = {}, True
    base_dir = Path(ROOT) / cfg["baseline_dir"] / "logs"
    for arm, baseline in pairs.items():
        result = run_arm_seediso(
            replay, arm, gate_dir, total_steps=steps, task_period=1_000,
            guard_every=1_000, keep_unit_arrays=False, write_logs=False,
            record_boundaries=False, generator_offset=int(P["s0_reference_offset"]))
        frame = result["frame"]
        differences, max_abs = [], {"unfit": 0.0, "eval_loss_exact": 0.0}
        for seed in cfg["common"]["seeds"]:
            mine = frame[frame.seed == int(seed)].set_index("step")
            with np.load(base_dir / f"{baseline}_seed{int(seed)}.npz",
                         allow_pickle=False) as data:
                for step in mine.index:
                    found = np.flatnonzero(data["step"] == int(step))
                    if len(found) != 1:
                        differences.append(dict(seed=int(seed), step=int(step), field="step"))
                        continue
                    index = int(found[0])
                    for key in ("unfit", "eval_loss_exact"):
                        delta = abs(float(mine.loc[step, key]) - float(data[key][index]))
                        max_abs[key] = max(max_abs[key], delta)
                        if delta > 1e-12:
                            differences.append(dict(seed=int(seed), step=int(step),
                                                    field=key, delta=delta))
                    for layer in (1, 2):
                        dead = float((data[f"layer{layer}_p_hat"][index] == 0).mean())
                        if float(mine.loc[step, f"L{layer}_strict_dead_frac"]) != dead:
                            differences.append(dict(seed=int(seed), step=int(step),
                                                    field=f"L{layer}_strict_dead_frac"))
        ok = bool(not differences and result["status"] == COMPLETE
                  and not result["excluded_seeds"] and result["sanity"]["pass_"])
        report[arm] = dict(pass_=ok, baseline=baseline, offset=0, steps=steps,
                           max_abs=max_abs, differences=differences[:50])
        all_ok &= ok
    out = dict(pass_=bool(all_ok), arms=report)
    (gate_dir / "s0_replay.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    if not out["pass_"]:
        raise RuntimeError(f"S0 failed: {out}")
    print("S0: PASS", flush=True)
    return out


def _train_step(st: dict) -> torch.Tensor:
    x = st["env"].step()
    y = st["teacher"](x)
    inputs, pres, acts, yhat = forward_centered(st, x)
    grads = grads_centered(st["net"], inputs, pres, acts, yhat - y)
    st["net"].sgd_step_layers(st["lr"], *grads)
    return x


def isolation_gates(cfg: dict, gate_dir: Path) -> tuple[dict, dict]:
    arm = "bwd"
    control, isolated = _setup(cfg, arm), _setup(cfg, arm)
    bad_index, bad_seed = 1, int(isolated["runs"][1]["seed"])
    isolated["net"].Ws[0][bad_index, 0, 0] = float("inf")
    rec = SeedIsolationRecorder(
        arm, arm_lambda(cfg, arm), isolated, record_steps=[], guard_steps=[0],
        guard_every=1_000, exclusion_cap=2,
        status_dir=gate_dir / "_synthetic_seed_status",
        sigma_tol=float(cfg["phase1"]["sigma_degenerate_tol"]),
        identity_tol=float(cfg["sanity"]["s1_identity_tol"]),
        keep_unit_arrays=False)
    rec(isolated, 0)
    keep = torch.tensor([i for i in range(len(cfg["common"]["seeds"])) if i != bad_index])
    streams_equal = True
    for _ in range(100):
        streams_equal &= torch.equal(_train_step(control), _train_step(isolated))
    tensors_control = control["net"].Ws + control["net"].bs + [control["net"].v,
                                                                  control["net"].c]
    tensors_isolated = isolated["net"].Ws + isolated["net"].bs + [isolated["net"].v,
                                                                     isolated["net"].c]
    state_equal = all(torch.equal(a[keep], b[keep])
                      for a, b in zip(tensors_control, tensors_isolated))
    mean_equal = all(
        a is None or torch.equal(a[keep], b[keep])
        for a, b in zip(control["layer_means"], isolated["layer_means"]))
    env_equal = (torch.equal(control["env"].flip_state, isolated["env"].flip_state)
                 and control["env"].t == isolated["env"].t)
    iso = dict(
        pass_=bool(set(rec.excluded) == {bad_seed} and streams_equal and state_equal
                   and mean_equal and env_equal),
        isolated_seed=bad_seed, unaffected_state_bitwise_equal=bool(state_equal),
        unaffected_means_bitwise_equal=bool(mean_equal),
        input_stream_bitwise_equal=bool(streams_equal), env_state_equal=bool(env_equal))

    cap_state = _setup(cfg, arm)
    for index in (0, 1, 2):
        cap_state["net"].Ws[0][index, 0, 0] = float("inf")
    cap_rec = SeedIsolationRecorder(
        arm, arm_lambda(cfg, arm), cap_state, record_steps=[], guard_steps=[0],
        guard_every=1_000, exclusion_cap=2,
        status_dir=gate_dir / "_synthetic_cap_status",
        sigma_tol=float(cfg["phase1"]["sigma_degenerate_tol"]),
        identity_tol=float(cfg["sanity"]["s1_identity_tol"]),
        keep_unit_arrays=False)
    event = None
    try:
        cap_rec(cap_state, 0)
    except ExclusionLimitExceeded as exc:
        event = exc.event
    cap = dict(pass_=bool(event and event["status"] == ARM_INVALID_EXCLUSION_LIMIT
                          and event["excluded_seeds"] == [0, 1, 2]), event=event)
    (gate_dir / "s_iso.json").write_text(
        json.dumps(iso, indent=2, ensure_ascii=False), encoding="utf-8")
    (gate_dir / "s_cap.json").write_text(
        json.dumps(cap, indent=2, ensure_ascii=False), encoding="utf-8")
    if not iso["pass_"] or not cap["pass_"]:
        raise RuntimeError(f"S-iso/S-cap failed: {iso}, {cap}")
    print("S-iso/S-cap: PASS", flush=True)
    return iso, cap


def s_count_gate(cfg: dict, gate_dir: Path) -> dict:
    P = _P(cfg)
    task_grid = np.arange(1, 501, dtype=int)
    blocks = ((task_grid - 1) // int(P["block_tasks"]) + 1)
    counts = {str(block): int((blocks == block).sum()) for block in range(1, 11)}
    test = copy.deepcopy(cfg)
    test["common"]["seeds"] = [0]
    st = _setup(test, "none")
    period = int(test["phase1"]["task_period"])
    states = []
    for _ in range(50):
        st["env"].segment(period)
        states.append(st["env"].flip_state.clone())
    changes = sum(not torch.equal(a, b) for a, b in zip(states[:-1], states[1:]))
    boundaries = len(states) - 1
    report = dict(
        pass_=bool(all(value == 50 for value in counts.values())
                   and changes == boundaries),
        block_task_end_counts=counts, tested_tasks=[1, 500],
        flip_state_changes=int(changes), boundary_comparisons=int(boundaries))
    (gate_dir / "s_count.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not report["pass_"]:
        raise RuntimeError(f"S-count failed: {report}")
    print("S-count: PASS", flush=True)
    return report


GATE_FILES = ("s_pair.json", "s0_replay.json", "s1_s2_algebra.json", "s3.json",
              "s_iso.json", "s_cap.json", "s_count.json")


def run_gates(cfg: dict, gate_dir: Path) -> None:
    gate_dir.mkdir(parents=True, exist_ok=True)
    s_pair_gate(cfg, gate_dir)
    s0_replay(cfg, gate_dir)
    s1_s2_algebra(_compat_cfg(cfg), gate_dir)
    isolation_gates(cfg, gate_dir)
    s_count_gate(cfg, gate_dir)
    _require_gates(gate_dir)
    print(f"ALL GATES PASS -> {gate_dir}", flush=True)


def _require_gates(gate_dir: Path) -> dict:
    reports = {}
    for name in GATE_FILES:
        path = gate_dir / name
        if not path.exists():
            raise RuntimeError(f"missing gate: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report.get("pass_"):
            raise RuntimeError(f"failed gate: {path}")
        reports[name] = report
    return reports


def block_levels(cfg: dict, frame: pd.DataFrame) -> pd.DataFrame:
    P, floor = _P(cfg), float(_P(cfg)["unfit_floor"])
    excluded = {"arm", "seed", "step", "task", "wd_b"}
    numeric = [column for column in frame.select_dtypes(include=[np.number]).columns
               if column not in excluded]
    rows = []
    for (arm, seed), group in frame.groupby(["arm", "seed"], sort=True):
        group = group[group.task > 0].copy()
        group["block"] = ((group["task"] - 1) // int(P["block_tasks"]) + 1).astype(int)
        group["log10_unfit"] = np.log10(np.maximum(group["unfit"].to_numpy(), floor))
        group["at_floor"] = group["unfit"].to_numpy() <= floor
        for block, gb in group.groupby("block"):
            row = dict(arm=arm, seed=int(seed), block=int(block),
                       task_lo=int(gb.task.min()), task_hi=int(gb.task.max()),
                       n_task_ends=int(len(gb)),
                       mean_log10_unfit=float(gb.log10_unfit.mean()),
                       log10_mean_unfit=float(np.log10(max(float(gb.unfit.mean()), floor))),
                       floor=floor, floor_frac=float(gb.at_floor.mean()))
            row.update({column: float(gb[column].mean()) for column in numeric
                        if column not in {"unfit"}})
            row["unfit"] = float(gb.unfit.mean())
            rows.append(row)
    out = pd.DataFrame(rows).sort_values(["arm", "seed", "block"])
    if not out.empty and not (out.n_task_ends == int(P["block_tasks"])).all():
        bad = out[out.n_task_ends != int(P["block_tasks"])]
        raise RuntimeError(f"S-count failed in realized data: {bad.to_dict('records')[:10]}")
    return out


def _draws(cfg: dict, n: int) -> np.ndarray:
    rng = np.random.default_rng(int(_P(cfg)["bootstrap_seed"]))
    return rng.integers(0, n, size=(int(_P(cfg)["bootstrap_B"]), n))


def _ci(cfg: dict, values: np.ndarray, draws: np.ndarray) -> dict:
    return paired_ci(_compat_cfg(cfg), np.asarray(values, dtype=np.float64), draws)


def _series(levels: pd.DataFrame, arm: str, block: int, column: str,
            seeds: list[int]) -> np.ndarray:
    group = levels[(levels.arm == arm) & (levels.block == block)].set_index("seed")
    missing = [seed for seed in seeds if seed not in group.index]
    if missing:
        raise RuntimeError(f"{arm} block {block}: missing seeds {missing}")
    return group.loc[seeds, column].to_numpy(dtype=np.float64)


def _fmt_ci(ci: dict) -> str:
    return (f"{ci['point']:+.6f} CI [{ci['ci_lo']:+.6f}, {ci['ci_hi']:+.6f}]"
            f"; ci_degenerate={bool(ci['ci_degenerate'])}")


def _interaction_label(ci: dict, margin: float) -> str:
    if ci["ci_lo"] >= -margin and ci["ci_hi"] <= margin:
        return "ADDITIVE"
    if ci["ci_lo"] > margin:
        return "SUBADDITIVE"
    if ci["ci_hi"] < -margin:
        return "SUPERADDITIVE"
    return "INCONCLUSIVE_WIDE"


def _main_label(cfg: dict, cis: dict, floor_pass: bool) -> tuple[str, dict]:
    if not floor_pass:
        return E_DRIFT_INVALID_FLOOR, dict(a=False, b=False, c=False)
    delta = float(_P(cfg)["equivalence_margin"])
    a = cis["both"]["ci_lo"] >= -delta and cis["both"]["ci_hi"] <= delta
    b = cis["bwd_minus_none"]["ci_hi"] < -delta
    c1 = cis["bwd"]["ci_lo"] > delta
    c2 = cis["both_minus_bwd"]["ci_hi"] < -delta
    c = c1 and c2
    if not a:
        if cis["both"]["ci_lo"] > delta:
            label = "RESIDUAL_UNEXPLAINED"
        elif cis["both"]["ci_hi"] < -delta:
            label = "OVERSHOOT_IMPROVES"
        else:
            label = "INCONCLUSIVE_WIDE"
    elif not b:
        label = "MU_CHANNEL_ONLY"
    elif not c:
        label = "B_CHANNEL_ONLY"
    else:
        label = "TWO_CHANNELS_BOTH_NECESSARY"
    return label, dict(a=bool(a), b=bool(b), c=bool(c), c_i=bool(c1), c_ii=bool(c2))


def analyze(cfg: dict, outdir: Path, meta: dict[str, dict], gates: dict) -> dict:
    P, iso = _P(cfg), _P(cfg)["seed_isolation"]
    frame = pd.read_csv(outdir / "task_end_metrics.csv")
    levels = block_levels(cfg, frame)
    levels.to_csv(outdir / "block_levels.csv", index=False)
    b02 = int(P["early_block_tasks"][1]) // int(P["block_tasks"])
    b10 = int(P["late_block_tasks"][1]) // int(P["block_tasks"])
    valid = {arm: meta[arm]["status"] in VALID_ARM_STATUSES for arm in ARM_ORDER}
    included = {arm: set(int(seed) for seed in meta[arm]["included_seeds"])
                if valid[arm] else set() for arm in ARM_ORDER}
    paired = sorted(set.intersection(*(included[arm] for arm in ARM_ORDER)))
    rows: list[dict] = []

    def add(pred: str, scope: str, verdict: str, evidence: str,
            ci_basis: str = "", ci_degenerate: object = "") -> None:
        rows.append(dict(pred=pred, scope=scope, verdict=verdict, evidence=evidence,
                         ci_basis=ci_basis, ci_degenerate=ci_degenerate))

    if not all(valid.values()):
        main = ARM_INVALID_EXCLUSION_LIMIT
        conditions, cis, details = {}, {}, {}
        add("P-main", "E-drift B10-B02", main,
            "; ".join(f"{arm}={meta[arm]['status']}" for arm in ARM_ORDER))
        floor_pass, ceiling_flag, ladder_inverts = False, False, False
    elif len(paired) < int(iso["min_paired_seeds"]):
        main = CONTRAST_INVALID_TOO_FEW_PAIRED
        conditions, cis, details = {}, {}, {}
        add("P-main", "E-drift B10-B02", main,
            f"common complete seeds={paired}; n={len(paired)} < {iso['min_paired_seeds']}")
        floor_pass, ceiling_flag, ladder_inverts = False, False, False
    else:
        draws = _draws(cfg, len(paired))
        level02 = {arm: _series(levels, arm, b02, "mean_log10_unfit", paired)
                   for arm in ARM_ORDER}
        level10 = {arm: _series(levels, arm, b10, "mean_log10_unfit", paired)
                   for arm in ARM_ORDER}
        drift = {arm: level10[arm] - level02[arm] for arm in ARM_ORDER}
        cis = {
            "both": _ci(cfg, drift["both"], draws),
            "bwd": _ci(cfg, drift["bwd"], draws),
            "bwd_minus_none": _ci(cfg, drift["bwd"] - drift["none"], draws),
            "both_minus_bwd": _ci(cfg, drift["both"] - drift["bwd"], draws),
        }
        floor_values = {(arm, block): float(_series(
            levels, arm, block, "floor_frac", paired).max())
            for arm in ARM_ORDER for block in (b02, b10)}
        floor_pass = all(value == 0.0 for value in floor_values.values())
        main, conditions = _main_label(cfg, cis, floor_pass)
        evidence = (
            f"common complete seeds={paired}; n={len(paired)}; "
            f"both drift {_fmt_ci(cis['both'])}; bwd drift {_fmt_ci(cis['bwd'])}; "
            f"bwd-none {_fmt_ci(cis['bwd_minus_none'])}; "
            f"both-bwd {_fmt_ci(cis['both_minus_bwd'])}; conditions={conditions}; "
            f"S-floor={'PASS' if floor_pass else 'FAIL'}")
        add("P-main", "E-drift = mean(log10 unfit) B10-B02", main, evidence,
            "paired percentile", int(any(ci["ci_degenerate"] for ci in cis.values())))

        b02_means = {arm: float(level02[arm].mean()) for arm in ARM_ORDER}
        b10_means = {arm: float(level10[arm].mean()) for arm in ARM_ORDER}
        drift_means = {arm: float(drift[arm].mean()) for arm in ARM_ORDER}
        b02_range = max(b02_means.values()) - min(b02_means.values())
        ceiling_flag = b02_range > float(P["ceiling_flag_dex"])
        drift_rank = sorted(ARM_ORDER, key=lambda arm: drift_means[arm])
        level_rank = sorted(ARM_ORDER, key=lambda arm: b10_means[arm])
        ladder_inverts = drift_rank != level_rank
        add("S-floor", "B02/B10 floor_frac, all four cells", "PASS" if floor_pass else "FAIL",
            "; ".join(f"{arm}/B{block:02d}={value:.6g}"
                      for (arm, block), value in floor_values.items()))
        add("S-ceiling", "B02 four-cell level range",
            "CEILING_CONTAMINATED" if ceiling_flag else "PASS",
            f"range={b02_range:.6f} dex; threshold={P['ceiling_flag_dex']:.1f}; "
            f"levels={b02_means}")
        add("L", "E-drift/E-level ladder", "LADDER_INVERTS" if ladder_inverts else "CONSISTENT",
            f"drift rank={drift_rank}; E-level rank={level_rank}; B10={b10_means}")
        for arm in ARM_ORDER:
            ci_level = _ci(cfg, level10[arm], draws)
            add("E-level", f"{arm} B10 tasks 451-500", "REPORT_ONLY",
                _fmt_ci(ci_level), "paired percentile", int(ci_level["ci_degenerate"]))

        interaction_drift = _ci(
            cfg, (drift["bwd"] - drift["none"]) - (drift["both"] - drift["cen"]), draws)
        interaction_level = _ci(
            cfg, (level10["bwd"] - level10["none"])
            - (level10["both"] - level10["cen"]), draws)
        interaction_labels = {
            "E-drift": _interaction_label(interaction_drift, float(P["interaction_margin"])),
            "E-level": _interaction_label(interaction_level, float(P["interaction_margin"])),
        }
        add("I", "(bwd-none)-(both-cen), E-drift", interaction_labels["E-drift"],
            _fmt_ci(interaction_drift), "paired percentile",
            int(interaction_drift["ci_degenerate"]))
        add("I", "(bwd-none)-(both-cen), E-level", interaction_labels["E-level"],
            _fmt_ci(interaction_level), "paired percentile",
            int(interaction_level["ci_degenerate"]))

        rank_diff = (_series(levels, "both", b10, "L1_eff_rank", paired)
                     - _series(levels, "cen", b10, "L1_eff_rank", paired))
        rank_ci = _ci(cfg, rank_diff, draws)
        rank_label = "SATURATION_PREVENTED" if rank_ci["ci_lo"] > 0 else "NOT_PREVENTED"
        add("R", "L1 eff_rank B10, both-cen", rank_label, _fmt_ci(rank_ci),
            "paired percentile", int(rank_ci["ci_degenerate"]))

        for arm in ARM_ORDER:
            for layer in (1, 2):
                d02 = _series(levels, arm, b02, f"L{layer}_strict_dead_frac", paired)
                d10 = _series(levels, arm, b10, f"L{layer}_strict_dead_frac", paired)
                add("D", f"{arm} L{layer} strict_dead_frac B02->B10", "REPORT_ONLY",
                    f"{d02.mean():.6f}->{d10.mean():.6f}; delta={float((d10-d02).mean()):+.6f}")
                for metric in ("M_median_alive", "B_median_alive"):
                    v02 = _series(levels, arm, b02, f"L{layer}_{metric}", paired)
                    v10 = _series(levels, arm, b10, f"L{layer}_{metric}", paired)
                    finite = np.isfinite(v02) & np.isfinite(v10)
                    if finite.all():
                        ledger_ci = _ci(cfg, v10 - v02, draws)
                        ledger_ev = (f"{v02.mean():+.6f}->{v10.mean():+.6f}; "
                                     f"{_fmt_ci(ledger_ci)}")
                        deg = int(ledger_ci["ci_degenerate"])
                        basis = "paired percentile"
                    else:
                        missing = [seed for seed, ok in zip(paired, finite) if not ok]
                        ledger_ev = f"undefined alive median seeds={missing}; CI not computed"
                        deg, basis = "", "not computed"
                    add("ledger", f"{arm} L{layer} {metric} B02->B10", "REPORT_ONLY",
                        ledger_ev, basis, deg)
        details = dict(
            levels_B02=b02_means, levels_B10=b10_means, drifts=drift_means,
            floor_values={f"{arm}_B{block:02d}": value
                          for (arm, block), value in floor_values.items()},
            b02_range=b02_range, drift_rank=drift_rank, level_rank=level_rank,
            interaction={"E-drift": interaction_drift, "E-level": interaction_level,
                         "labels": interaction_labels}, rank=rank_ci,
        )

    for arm in ARM_ORDER:
        add("exclusion", arm, "ARM_VALID" if valid[arm] else ARM_INVALID_EXCLUSION_LIMIT,
            f"status={meta[arm]['status']}; excluded={meta[arm]['excluded_seeds']}; "
            f"included={meta[arm]['included_seeds']}")
    verdict = pd.DataFrame(rows)
    verdict.to_csv(outdir / "verdict.csv", index=False)

    endpoints = pd.DataFrame({"seed": paired})
    if paired:
        for arm in ARM_ORDER:
            for block, tag in ((b02, "B02"), (b10, "B10")):
                for metric, short in (("mean_log10_unfit", "meanlog10unfit"),
                                      ("log10_mean_unfit", "log10meanunfit"),
                                      ("L1_strict_dead_frac", "L1_dead"),
                                      ("L2_strict_dead_frac", "L2_dead"),
                                      ("L1_eff_rank", "L1_eff_rank")):
                    endpoints[f"{arm}_{tag}_{short}"] = _series(
                        levels, arm, block, metric, paired)
            endpoints[f"{arm}_drift"] = (
                endpoints[f"{arm}_B10_meanlog10unfit"]
                - endpoints[f"{arm}_B02_meanlog10unfit"])
    endpoints.to_csv(outdir / "paired_endpoints.csv", index=False)

    exclusion_rows = []
    for arm in ARM_ORDER:
        events = {int(event["seed"]): event for event in meta[arm]["exclusion_events"]}
        for seed in cfg["common"]["seeds"]:
            event = events.get(int(seed))
            exclusion_rows.append(dict(
                arm=arm, seed=int(seed), excluded=int(event is not None),
                detected_step="" if event is None else event["detected_step"],
                detected_task="" if event is None else event["detected_task"],
                nonfinite_tensors="" if event is None else ";".join(event["nonfinite_tensors"]),
                arm_status=meta[arm]["status"]))
    pd.DataFrame(exclusion_rows).to_csv(outdir / "exclusions.csv", index=False)

    result = dict(
        main_verdict=main, conditions=conditions, common_complete_seeds=paired,
        n_paired=len(paired), cis=cis, floor_pass=bool(floor_pass),
        ceiling_contaminated=bool(ceiling_flag), ladder_inverts=bool(ladder_inverts),
        details=details, blocks=dict(B02=b02, B10=b10))
    _figure(frame, outdir)
    _summary(cfg, outdir, verdict, levels, result)
    return result


def _figure(frame: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    colors = {"none": "#555555", "bwd": "#2b8cbe", "cen": "#31a354", "both": "#e34a33"}
    panels = [
        ("unfit", "exact-support unfit", True),
        ("L1_strict_dead_frac", "strict_dead_frac L1", False),
        ("L2_strict_dead_frac", "strict_dead_frac L2", False),
        ("L1_eff_rank", "activation eff_rank L1", False),
        ("L1_M_median_alive", "alive median M L1", False),
        ("L1_B_median_alive", "alive median B L1", False),
    ]
    for (metric, label, logy), axis in zip(panels, axes.flat):
        for arm in ARM_ORDER:
            group = frame[frame.arm == arm].groupby("task")[metric].median()
            axis.plot(group.index, group.values, color=colors[arm], lw=1.1, label=arm)
        axis.set_xlabel("task")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        if logy:
            axis.set_yscale("log")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("channel_2x2_0901 — b-WD x EMA centering")
    fig.tight_layout()
    fig.savefig(outdir / "fig_channel_2x2.png", dpi=150)
    plt.close(fig)


def _summary(cfg: dict, outdir: Path, verdict: pd.DataFrame, levels: pd.DataFrame,
             result: dict) -> None:
    P, paired = _P(cfg), result["common_complete_seeds"]
    b02, b10 = result["blocks"]["B02"], result["blocks"]["B10"]
    table_rows = []
    for arm in ARM_ORDER:
        for block, tag in ((b02, "B02"), (b10, "B10")):
            group = levels[(levels.arm == arm) & (levels.block == block)
                           & (levels.seed.isin(paired))]
            table_rows.append(dict(
                arm=arm, window=tag,
                mean_log10_unfit=float(group.mean_log10_unfit.mean()) if len(group) else np.nan,
                log10_mean_unfit=float(group.log10_mean_unfit.mean()) if len(group) else np.nan,
                L1_dead=float(group.L1_strict_dead_frac.mean()) if len(group) else np.nan,
                L2_dead=float(group.L2_strict_dead_frac.mean()) if len(group) else np.nan,
                L1_eff_rank=float(group.L1_eff_rank.mean()) if len(group) else np.nan,
                floor_frac=float(group.floor_frac.max()) if len(group) else np.nan))
    table = pd.DataFrame(table_rows)
    predicted_main = "TWO_CHANNELS_BOTH_NECESSARY"
    actual_interaction = result.get("details", {}).get("interaction", {}).get(
        "labels", {}).get("E-drift", "not computed")
    level_rank = result.get("details", {}).get("level_rank", [])
    actual_best = level_rank[0] if level_rank else "not computed"
    predictions = pd.DataFrame([
        dict(item="主判定", prediction=predicted_main, result=result["main_verdict"],
             match="一致" if result["main_verdict"] == predicted_main else "外れた"),
        dict(item="both E-drift ±0.15内", prediction="yes",
             result="yes" if result.get("conditions", {}).get("a") else "no",
             match="一致" if result.get("conditions", {}).get("a") else "外れた"),
        dict(item="交互作用 E-drift", prediction="SUBADDITIVE", result=actual_interaction,
             match="一致" if actual_interaction == "SUBADDITIVE" else "外れた"),
        dict(item="E-level 最良", prediction="bwd", result=actual_best,
             match="一致" if actual_best == "bwd" else "外れた"),
        dict(item="外れた場合の改稿先", prediction="わからない",
             result="本走から自動決定しない", match="N/A"),
    ])
    warning = []
    if result["ceiling_contaminated"]:
        warning.append("`CEILING_CONTAMINATED`: B02 の4セル間差が3 dexを超えたため E-drift 単独では読まない。")
    if result["ladder_inverts"]:
        warning.append("`LADDER_INVERTS`: E-drift と E-level の順位が異なるため、どちらも単独では引かない。")
    if not result["floor_pass"]:
        warning.append("`S-floor FAIL`: E-drift は無効で、E-level のみ報告する。")
    interaction_note = []
    interaction_result = result.get("details", {}).get("interaction", {}).get("E-drift")
    if actual_interaction == "SUBADDITIVE":
        interaction_note.append(
            "- `SUBADDITIVE` は既存の事後配置と同方向の再現であり、独立に立てた予言ではない。")
    elif interaction_result is not None:
        interaction_note.extend([
            f"- 交互作用は事前登録の字義どおり `(bwd-none)-(both-cen)` で計算し、"
            f"{interaction_result['point']:+.6f} dex の `{actual_interaction}` だった。",
            "- spec §2.1 の旧配置 `+5.889` と §8.1 の `SUBADDITIVE` 予測は、§6.3 に固定した式とは符号規約が逆である。結果後に符号を反転せず、字義どおりの式とラベルを維持した。",
        ])
    lines = [
        "# channel_2x2_0901 — チャネル遮断 2×2 本走", "",
        f"事前登録: [`{cfg['spec']}`](../../{cfg['spec']})（repo commit `31f3792`）。", "",
        f"主判定は **{result['main_verdict']}**。4腕共通の完走 seed = {paired} "
        f"(n={result['n_paired']})。", "",
        f"窓は B02 = task {P['early_block_tasks'][0]}–{P['early_block_tasks'][1]}、"
        f"B10 = task {P['late_block_tasks'][0]}–{P['late_block_tasks'][1]}。"
        f"主 endpoint は `mean(log10 unfit)` の B10−B02、床は {P['unfit_floor']:.0e}。", "",
        "## 事前予測との対応", "", markdown_table(predictions), "",
        "## 判定", "", markdown_table(verdict), "",
        "## B02 / B10 水準", "", markdown_table(table), "",
        "## フラグ", "",
    ]
    lines.extend(f"- {item}" for item in warning)
    if not warning:
        lines.append("- S-floor / S-ceiling / ladder の追加フラグなし。")
    lines.extend([
        "", "## 解釈上の制限", "",
        *interaction_note,
        "- centered セルは B02 水準が低く、落ちる余地の差だけでも劣加法が生じる。本走はこれを分離しない。",
        "- EMA 中心化は µ とタスク可識別性を同時に消すため、『µ を消した』とは書かない。",
        "- `strict_dead` は REPORT_ONLY で、主判定には使っていない。",
        "- スコープは condA・幅100・hidden [100,100]・T=10^4・batch=1・lr=0.01・plain SGD・5M・lambda=1e-3。",
    ])
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


OUTPUTS = (
    "verdict.csv", "summary.md", "paired_endpoints.csv", "exclusions.csv",
    "task_end_metrics.csv", "block_levels.csv", "boundary_snapshots.csv",
    "run_sanity.json", "config_used.yaml", "fig_channel_2x2.png",
)


def _shard(outdir: Path) -> Path:
    path = outdir / "shards"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_boundary_snapshots(outdir: Path, frame: pd.DataFrame,
                              post_frames: list[pd.DataFrame], total: int,
                              period: int) -> None:
    pre = frame[(frame.step > 0) & (frame.step < total)
                & (frame.step % period == 0)].copy()
    columns = ["arm", "seed", "step"]
    for layer in (1, 2):
        columns.extend([
            f"L{layer}_strict_dead_frac", f"L{layer}_submerged_frac",
            f"L{layer}_M_median_alive", f"L{layer}_B_median_alive",
            f"L{layer}_sigma_median_alive",
        ])
    pre = pre[columns]
    pre["task_boundary"] = (pre.step // period).astype(int)
    pre["side"] = "pre"
    posts = pd.concat(post_frames, ignore_index=True) if post_frames else pd.DataFrame()
    pd.concat([pre, posts], ignore_index=True).sort_values(
        ["arm", "seed", "step", "side"]).to_csv(
            outdir / "boundary_snapshots.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--gates", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arm")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--outdir")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(str(cfg_path))
    validate_config(cfg, full=not args.smoke)
    require_omp(int(_P(cfg)["omp_num_threads"]))
    outdir = Path(args.outdir).resolve() if args.outdir else outdir_of(cfg)
    outdir.mkdir(parents=True, exist_ok=True)
    gate_dir = Path(ROOT) / "results" / "_gate_channel_2x2_0901"
    started = time.time()

    if args.gates:
        run_gates(cfg, gate_dir)
        return
    if args.smoke:
        for arm in ARM_ORDER:
            smoke = copy.deepcopy(cfg)
            smoke["common"]["checkpoints"] = []
            result = run_arm_seediso(
                smoke, arm, Path(ROOT) / "results" / "_smoke_channel_2x2_0901",
                total_steps=30_000, task_period=10_000, guard_every=1_000,
                keep_unit_arrays=False, write_logs=False, record_boundaries=False)
            if result["status"] != COMPLETE or not result["sanity"]["pass_"]:
                raise RuntimeError(f"smoke failed: {arm}: {result['status']}")
        print("SMOKE PASS", flush=True)
        return

    gates = _require_gates(gate_dir)
    total = int(cfg["common"]["total_steps"])
    period = int(cfg["phase1"]["task_period"])
    guard = int(_P(cfg)["guard_every"])
    todo = [args.arm] if args.arm else list(ARM_ORDER)
    if args.arm and args.arm not in ARM_ORDER:
        raise SystemExit(f"unknown arm {args.arm}")
    if not args.analyze_only:
        for arm in todo:
            result = run_arm_seediso(cfg, arm, outdir, total_steps=total,
                                     task_period=period, guard_every=guard)
            result["frame"].to_csv(_shard(outdir) / f"{arm}.csv", index=False)
            result["boundary_frame"].to_csv(
                _shard(outdir) / f"{arm}_boundary_post.csv", index=False)
            (_shard(outdir) / f"{arm}.json").write_text(json.dumps(
                {key: value for key, value in result.items()
                 if key not in {"frame", "boundary_frame"}},
                indent=2, ensure_ascii=False), encoding="utf-8")
        if args.arm:
            return

    frames, post_frames, meta = [], [], {}
    for arm in ARM_ORDER:
        meta[arm] = json.loads((_shard(outdir) / f"{arm}.json").read_text(encoding="utf-8"))
        if meta[arm]["status"] in VALID_ARM_STATUSES:
            frames.append(pd.read_csv(_shard(outdir) / f"{arm}.csv"))
            post_path = _shard(outdir) / f"{arm}_boundary_post.csv"
            if post_path.exists() and post_path.stat().st_size:
                post_frames.append(pd.read_csv(post_path))
    if not frames:
        raise RuntimeError("no valid arms to analyze")
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(outdir / "task_end_metrics.csv", index=False)
    _write_boundary_snapshots(outdir, frame, post_frames, total, period)

    result = analyze(cfg, outdir, meta, gates)
    run_sanity = dict(
        gates={name: report["pass_"] for name, report in gates.items()},
        S3={arm: dict(pass_=meta[arm]["sanity"]["pass_"],
                      max_relerr=meta[arm]["sanity"]["max_relerr"],
                      quantization_violations=meta[arm]["sanity"]["n_quantization_violations"],
                      wall_violations=meta[arm]["sanity"]["n_wall_identity_violations"])
            for arm in ARM_ORDER},
        S4_seed_isolation={arm: dict(status=meta[arm]["status"],
                                     excluded_seeds=meta[arm]["excluded_seeds"],
                                     events=meta[arm]["exclusion_events"])
                           for arm in ARM_ORDER},
        S_floor_pass=result["floor_pass"],
        S_ceiling="CEILING_CONTAMINATED" if result["ceiling_contaminated"] else "PASS",
        ladder="LADDER_INVERTS" if result["ladder_inverts"] else "CONSISTENT",
        training_elapsed_sec={arm: meta[arm]["elapsed_sec"] for arm in ARM_ORDER})
    (outdir / "run_sanity.json").write_text(
        json.dumps(run_sanity, indent=2, ensure_ascii=False), encoding="utf-8")
    with (outdir / "config_used.yaml").open("w") as stream:
        yaml.safe_dump(cfg, stream, allow_unicode=True, sort_keys=False)
    (outdir / "provenance.json").write_text(json.dumps(provenance(
        "channel_2x2_0901", cfg_path, cfg, outdir,
        dict(analysis=result, run_sanity=run_sanity,
             preregistration_commit="31f3792231e565dffa1dbf29e53aa01eec101762"),
        started, sys.argv, OUTPUTS), indent=2, ensure_ascii=False), encoding="utf-8")
    print(pd.read_csv(outdir / "verdict.csv")[["pred", "scope", "verdict"]]
          .to_string(index=False), flush=True)
    print(f"ALL DONE -> {outdir}", flush=True)


if __name__ == "__main__":
    main()

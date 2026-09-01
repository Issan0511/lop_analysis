"""std b-WD を seed 単位で隔離して再走する。

事前登録: ``specs/spec_bias_wd_std_seediso_0901.md``。
"""
from __future__ import annotations

import argparse
import copy
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
    markdown_table,
    provenance,
    require_omp,
    write_arm_npz,
)
from .bias_wd_std_0901 import block_levels
from .common import ROOT, load_config
from .mlp2_phase0 import identity_sanity_pass
from .mlp2_phase1 import (
    _base_cfg,
    _numeric_divergence_event,
    exact_layer_record_p1,
    forward_centered,
    grads_centered,
    setup_arm_p1,
    train_arm_p1,
)


CONFIG = Path(ROOT) / "configs" / "bias_wd_std_seediso_0901.yaml"
COMPLETE = "COMPLETE"
COMPLETE_WITH_EXCLUSIONS = "COMPLETE_WITH_EXCLUSIONS"
ARM_INVALID_EXCLUSION_LIMIT = "ARM_INVALID_EXCLUSION_LIMIT"
CONTRAST_INVALID_TOO_FEW_PAIRED = "CONTRAST_INVALID_TOO_FEW_PAIRED"
LOP_PERSISTS = "LOP_PERSISTS"
LOP_REMOVED = "LOP_REMOVED"
INCONCLUSIVE_PARTIAL = "INCONCLUSIVE_PARTIAL"
VALID_ARM_STATUSES = {COMPLETE, COMPLETE_WITH_EXCLUSIONS}


class ExclusionLimitExceeded(RuntimeError):
    def __init__(self, event: dict):
        self.event = event
        super().__init__(f"exclusion limit exceeded: {event}")


def arm_lambda(cfg: dict, name: str) -> float:
    return float(next(a for a in cfg["arms"] if a["name"] == name)["wd_b"])


def outdir_of(cfg: dict) -> Path:
    return Path(ROOT) / cfg["bias_wd"]["output_dir"]


def validate_config(cfg: dict, *, full: bool) -> None:
    C, A, P = cfg["common"], cfg["condA"], cfg["bias_wd"]
    I = P["seed_isolation"]
    if C.get("device") != "cpu" or float(C["lr_main"]) != 0.01:
        raise ValueError("registered CPU/lr differs")
    if (int(A["m"]), int(A["f"]), int(A["target_hidden"]), float(A["beta"])) != (
            20, 15, 100, 0.7):
        raise ValueError("registered condA differs")
    if list(A["T_values"]) != [10_000] or list(A["encodings"]) != ["std"]:
        raise ValueError("registered T/encoding differs")
    expected = [("S_none", 0.0), ("S_main", 1e-3), ("S_sub", 1e-1)]
    got = [(a["name"], float(a["wd_b"])) for a in cfg["arms"]]
    if got != expected:
        raise ValueError(f"registered arms differ: {got}")
    if any(list(a["hidden"]) != [100, 100] or list(a["centered_layers"]) != []
           for a in cfg["arms"]):
        raise ValueError("all arms must be uncentered depth-2 width-100")
    if (int(I["max_exclusions_per_arm"]), int(I["min_complete_seeds_per_arm"]),
            int(I["min_paired_seeds"]), int(I["keep_rng_rows"])) != (2, 8, 8, 10):
        raise ValueError("registered isolation limits differ")
    if not I["exclude_entire_seed_trajectory"] or not I["stop_arm_on_limit_exceeded"]:
        raise ValueError("registered exclusion policy differs")
    if list(P["early_block_tasks"]) != [51, 100] or list(P["late_block_tasks"]) != [451, 500]:
        raise ValueError("registered windows differ")
    if (float(P["persist_ratio"]), float(P["removed_ratio"]),
            float(P["unfit_floor_L2"])) != (0.5, 0.1, 1e-23):
        raise ValueError("registered thresholds/floor differ")
    if (int(P["bootstrap_B"]), int(P["bootstrap_seed"])) != (20_000, 20_260_904):
        raise ValueError("registered bootstrap differs")
    if full and (int(C["total_steps"]) != 5_000_000
                 or list(C["seeds"]) != list(range(10))):
        raise ValueError("full run must be 5M and seeds 0..9")


def _arm_cfg(cfg: dict, name: str) -> dict:
    return next(a for a in cfg["arms"] if a["name"] == name)


def _quarantine_seed(st: dict, seed: int) -> int:
    index = next(i for i, run in enumerate(st["runs"])
                 if int(run["seed"]) == int(seed))
    with torch.no_grad():
        for value in st["net"].Ws + st["net"].bs + [st["net"].v, st["net"].c]:
            value[index].zero_()
        st["lr"][index] = 0.0
        st["running_mean"][index].zero_()
        for mean in st.get("layer_means", []):
            if mean is not None:
                mean[index].zero_()
    return index


class SeedIsolationRecorder(TaskEndRecorder):
    """非有限 seed だけを quarantine し、残りを同じ乱数軌道で継続する。"""

    def __init__(self, arm: str, wd_b: float, st: dict, *, record_steps,
                 guard_steps, guard_every: int,
                 exclusion_cap: int, status_dir: Path, **kwargs):
        super().__init__(arm, wd_b, st, record_steps=record_steps,
                         guard_steps=[], guard_every=guard_every, **kwargs)
        self.isolation_guard_steps = {int(step) for step in guard_steps}
        self.exclusion_cap = int(exclusion_cap)
        self.status_dir = Path(status_dir)
        self.excluded: dict[int, dict] = {}

    def __call__(self, st: dict, step: int) -> None:
        step = int(step)
        if step in self.isolation_guard_steps:
            event = _numeric_divergence_event(st, step)
            if event is not None:
                for seed in event["bad_seeds"]:
                    seed = int(seed)
                    if seed in self.excluded:
                        continue
                    seed_event = dict(
                        status="NUMERIC_DIVERGENCE", arm=self.arm, seed=seed,
                        detected_step=step,
                        detected_task=int(event["detected_task"]),
                        probe_every=self.guard_every,
                        nonfinite_tensors=event["nonfinite_tensors"][str(seed)],
                        action="quarantine_seed_and_continue",
                        exclude_entire_seed_trajectory=True, rescue="none",
                    )
                    _quarantine_seed(st, seed)
                    self.excluded[seed] = seed_event
                    self.status_dir.mkdir(parents=True, exist_ok=True)
                    (self.status_dir / f"{self.arm}_seed{seed}.json").write_text(
                        json.dumps(seed_event, indent=2, ensure_ascii=False), encoding="utf-8")
                    print(f"[{self.arm}] isolate seed {seed} at step {step:,} "
                          f"({len(self.excluded)}/{self.exclusion_cap})", flush=True)
                    if len(self.excluded) > self.exclusion_cap:
                        raise ExclusionLimitExceeded(dict(
                            status=ARM_INVALID_EXCLUSION_LIMIT, arm=self.arm,
                            detected_step=step, excluded_seeds=sorted(self.excluded),
                            cap=self.exclusion_cap,
                        ))
        super().__call__(st, step)


def _setup(cfg: dict, arm_name: str) -> dict:
    st = setup_arm_p1(_base_cfg(cfg), _arm_cfg(cfg, arm_name), "cpu")
    st["net"].set_weight_decay_b(arm_lambda(cfg, arm_name))
    _, sanity = exact_layer_record_p1(st, float(cfg["phase1"]["sigma_degenerate_tol"]))
    if not identity_sanity_pass(sanity, float(cfg["sanity"]["s1_identity_tol"])):
        raise RuntimeError(f"{arm_name}: preflight identity failed")
    return st


def run_arm_seediso(cfg: dict, arm_name: str, outdir: Path, *, total_steps: int,
                    task_period: int, guard_every: int,
                    keep_unit_arrays: bool = True, write_logs: bool = True) -> dict:
    st = _setup(cfg, arm_name)
    cap = int(cfg["bias_wd"]["seed_isolation"]["max_exclusions_per_arm"])
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
    checkpoints = [int(v) for v in cfg["common"].get("checkpoints", [])
                   if int(v) <= total_steps]
    print(f"[{arm_name}] seed-isolated wd_b={arm_lambda(cfg, arm_name):g} "
          f"steps={total_steps:,}", flush=True)
    started = time.time()
    limit_event = None
    try:
        elapsed = train_arm_p1(st, recorder, probes, total_steps, outdir, checkpoints)
    except ExclusionLimitExceeded as exc:
        elapsed = time.time() - started
        limit_event = exc.event

    excluded = sorted(recorder.excluded)
    included = [int(run["seed"]) for run in st["runs"]
                if int(run["seed"]) not in recorder.excluded]
    status = (ARM_INVALID_EXCLUSION_LIMIT if limit_event is not None else
              COMPLETE_WITH_EXCLUSIONS if excluded else COMPLETE)
    raw_frame = recorder.dataframe()
    frame = raw_frame[~raw_frame.seed.isin(excluded)].copy()
    sanity = recorder.sanity()
    if write_logs and limit_event is None:
        write_arm_npz(outdir, arm_name, arm_lambda(cfg, arm_name), st, recorder)
    result = dict(
        arm=arm_name, wd_b=arm_lambda(cfg, arm_name), status=status,
        elapsed_sec=float(elapsed), excluded_seeds=excluded, included_seeds=included,
        exclusion_events=[recorder.excluded[seed] for seed in excluded],
        exclusion_cap=cap, limit_event=limit_event, sanity=sanity,
        frame=frame,
    )
    status_path = outdir / "arm_status" / f"{arm_name}.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(
        {key: value for key, value in result.items() if key != "frame"},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{arm_name}] {status} in {elapsed:.1f}s; excluded={excluded}; "
          f"sanity={'PASS' if sanity['pass_'] else 'FAIL'}", flush=True)
    return result


def s0_replay(cfg: dict, gate_dir: Path) -> dict:
    steps = int(cfg["bias_wd"]["s0_replay_steps"])
    baseline = str(cfg["bias_wd"]["baseline_arm"])
    replay = copy.deepcopy(cfg)
    replay["common"]["total_steps"] = steps
    replay["common"]["checkpoints"] = []
    result = run_arm_seediso(replay, "S_none", gate_dir, total_steps=steps,
                             task_period=1000, guard_every=1000,
                             keep_unit_arrays=False, write_logs=False)
    frame = result["frame"]
    base_dir = Path(ROOT) / cfg["baseline_dir"] / "logs"
    differences, max_abs = [], {"unfit": 0.0, "eval_loss_exact": 0.0}
    for seed in cfg["common"]["seeds"]:
        mine = frame[frame.seed == int(seed)].set_index("step")
        with np.load(base_dir / f"{baseline}_seed{int(seed)}.npz", allow_pickle=False) as data:
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
                        differences.append(dict(seed=int(seed), step=int(step), field=key,
                                                delta=delta))
                for layer in (1, 2):
                    dead = float((data[f"layer{layer}_p_hat"][index] == 0).mean())
                    if float(mine.loc[step, f"L{layer}_strict_dead_frac"]) != dead:
                        differences.append(dict(seed=int(seed), step=int(step),
                                                field=f"L{layer}_dead"))
    report = dict(
        pass_=bool(not differences and not result["excluded_seeds"]
                   and result["status"] == COMPLETE),
        baseline=baseline, steps=steps, max_abs=max_abs,
        excluded_seeds=result["excluded_seeds"], differences=differences[:50],
    )
    gate_dir.mkdir(parents=True, exist_ok=True)
    (gate_dir / "s0_replay.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not report["pass_"]:
        raise RuntimeError(f"S0 failed: {report}")
    print("S0 seed-isolated runner: PASS", flush=True)
    return report


def _train_step(st: dict) -> torch.Tensor:
    x = st["env"].step()
    y = st["teacher"](x)
    inputs, pres, acts, yhat = forward_centered(st, x)
    grads = grads_centered(st["net"], inputs, pres, acts, yhat - y)
    st["net"].sgd_step_layers(st["lr"], *grads)
    return x


def isolation_gate(cfg: dict, gate_dir: Path) -> dict:
    test_cfg = copy.deepcopy(cfg)
    arm = "S_main"
    control, isolated = _setup(test_cfg, arm), _setup(test_cfg, arm)
    bad_index, bad_seed = 1, int(isolated["runs"][1]["seed"])
    isolated["net"].Ws[0][bad_index, 0, 0] = float("inf")
    rec = SeedIsolationRecorder(
        arm, arm_lambda(cfg, arm), isolated,
        record_steps=[], guard_steps=[0], guard_every=1000,
        exclusion_cap=2, status_dir=gate_dir / "_synthetic_seed_status",
        sigma_tol=float(cfg["phase1"]["sigma_degenerate_tol"]),
        identity_tol=float(cfg["sanity"]["s1_identity_tol"]),
        keep_unit_arrays=False,
    )
    rec(isolated, 0)
    keep = torch.tensor([i for i in range(10) if i != bad_index])
    streams_equal = True
    for _ in range(100):
        streams_equal = streams_equal and torch.equal(_train_step(control),
                                                        _train_step(isolated))
    state_equal = all(torch.equal(a[keep], b[keep]) for a, b in zip(
        control["net"].Ws + control["net"].bs + [control["net"].v, control["net"].c],
        isolated["net"].Ws + isolated["net"].bs + [isolated["net"].v, isolated["net"].c]))
    env_equal = (torch.equal(control["env"].flip_state, isolated["env"].flip_state)
                 and control["env"].t == isolated["env"].t)

    cap_state = _setup(test_cfg, arm)
    for index in (0, 1, 2):
        cap_state["net"].Ws[0][index, 0, 0] = float("inf")
    cap_rec = SeedIsolationRecorder(
        arm, arm_lambda(cfg, arm), cap_state,
        record_steps=[], guard_steps=[0], guard_every=1000,
        exclusion_cap=2, status_dir=gate_dir / "_synthetic_cap_status",
        sigma_tol=float(cfg["phase1"]["sigma_degenerate_tol"]),
        identity_tol=float(cfg["sanity"]["s1_identity_tol"]),
        keep_unit_arrays=False,
    )
    cap_ok = False
    try:
        cap_rec(cap_state, 0)
    except ExclusionLimitExceeded as exc:
        cap_ok = (exc.event["status"] == ARM_INVALID_EXCLUSION_LIMIT
                  and exc.event["excluded_seeds"] == [0, 1, 2])
    report = dict(
        pass_=bool(rec.excluded.keys() == {bad_seed}
                   and streams_equal and state_equal and env_equal and cap_ok),
        isolated_seed=bad_seed, unaffected_state_bitwise_equal=bool(state_equal),
        input_stream_bitwise_equal=bool(streams_equal), env_state_equal=bool(env_equal),
        exclusion_cap_gate=bool(cap_ok),
    )
    gate_dir.mkdir(parents=True, exist_ok=True)
    (gate_dir / "isolation_gate.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not report["pass_"]:
        raise RuntimeError(f"isolation gate failed: {report}")
    print("S-iso/S-cap: PASS", flush=True)
    return report


def _draws(cfg: dict, n: int) -> np.ndarray:
    rng = np.random.default_rng(int(cfg["bias_wd"]["bootstrap_seed"]))
    return rng.integers(0, n, size=(int(cfg["bias_wd"]["bootstrap_B"]), n))


def _series(levels: pd.DataFrame, arm: str, block: int, column: str,
            seeds: list[int]) -> np.ndarray:
    group = levels[(levels.arm == arm) & (levels.block == block)].set_index("seed")
    missing = [seed for seed in seeds if seed not in group.index]
    if missing:
        raise RuntimeError(f"{arm} block {block}: missing included seeds {missing}")
    return group.loc[seeds, column].to_numpy(dtype=np.float64)


def _classify(cfg: dict, ci: dict) -> str:
    if float(ci["ci_lo"]) >= float(cfg["bias_wd"]["persist_ratio"]):
        return LOP_PERSISTS
    if float(ci["ci_hi"]) <= float(cfg["bias_wd"]["removed_ratio"]):
        return LOP_REMOVED
    return INCONCLUSIVE_PARTIAL


def analyze(cfg: dict, outdir: Path, meta: dict[str, dict]) -> dict:
    P, I = cfg["bias_wd"], cfg["bias_wd"]["seed_isolation"]
    frame = pd.read_csv(outdir / "task_end_metrics.csv")
    levels = block_levels(cfg, frame)
    levels.to_csv(outdir / "block_levels.csv", index=False)
    size = int(P["block_tasks"])
    b02 = (int(P["early_block_tasks"][1]) - 1) // size + 1
    b10 = (int(P["late_block_tasks"][1]) - 1) // size + 1
    valid = {arm: meta[arm]["status"] in VALID_ARM_STATUSES for arm in meta}
    included = {arm: [int(seed) for seed in meta[arm]["included_seeds"]]
                if valid[arm] else [] for arm in meta}
    rows = []

    paired = sorted(set(included["S_none"]) & set(included["S_main"]))
    ratio_ci = diff_ci = None
    ratio_values = None
    if not valid["S_none"] or not valid["S_main"]:
        main_verdict = ARM_INVALID_EXCLUSION_LIMIT
        evidence = (f"arm status: S_none={meta['S_none']['status']}, "
                    f"S_main={meta['S_main']['status']}")
    elif len(paired) < int(I["min_paired_seeds"]):
        main_verdict = CONTRAST_INVALID_TOO_FEW_PAIRED
        evidence = f"paired complete seeds={paired} (n={len(paired)} < {I['min_paired_seeds']})"
    else:
        draws = _draws(cfg, len(paired))
        none02 = _series(levels, "S_none", b02, "mean_log10_unfit", paired)
        none10 = _series(levels, "S_none", b10, "mean_log10_unfit", paired)
        main02 = _series(levels, "S_main", b02, "mean_log10_unfit", paired)
        main10 = _series(levels, "S_main", b10, "mean_log10_unfit", paired)
        d_none, d_main = none10 - none02, main10 - main02
        if np.any(d_none == 0):
            raise RuntimeError("exactly zero control degradation")
        ratio_values = d_main / d_none
        ratio_ci = paired_ci(cfg, ratio_values, draws)
        diff_ci = paired_ci(cfg, d_main - d_none, draws)
        small = np.abs(d_none) < float(P["small_denominator_dex"])
        main_verdict = _classify(cfg, ratio_ci)
        evidence = (
            f"paired seeds={paired}; n={len(paired)}; none drift {d_none.mean():+.6f}; "
            f"main drift {d_main.mean():+.6f}; ratio {ratio_ci['point']:.6f} CI "
            f"[{ratio_ci['ci_lo']:.6f}, {ratio_ci['ci_hi']:.6f}]; drift diff "
            f"{diff_ci['point']:+.6f} CI [{diff_ci['ci_lo']:+.6f}, "
            f"{diff_ci['ci_hi']:+.6f}]; small denominator {int(small.sum())}/{len(paired)}")
    rows.append(dict(pred="P-main", scope="B10-B02 degradation ratio S_main/S_none",
                     verdict=main_verdict, evidence=evidence,
                     ci_basis="paired percentile" if ratio_ci else "not computed"))

    for arm in ("S_none", "S_main", "S_sub"):
        rows.append(dict(
            pred="exclusion", scope=arm,
            verdict=("ARM_VALID" if valid[arm] else ARM_INVALID_EXCLUSION_LIMIT),
            evidence=(f"excluded={meta[arm]['excluded_seeds']}; "
                      f"included={included[arm]}; n={len(included[arm])}; "
                      f"status={meta[arm]['status']}"), ci_basis="",
        ))

    sub_paired = sorted(set(included["S_none"]) & set(included["S_sub"]))
    if valid["S_none"] and valid["S_sub"] and len(sub_paired) >= int(I["min_paired_seeds"]):
        draws = _draws(cfg, len(sub_paired))
        dn = (_series(levels, "S_none", b10, "mean_log10_unfit", sub_paired)
              - _series(levels, "S_none", b02, "mean_log10_unfit", sub_paired))
        ds = (_series(levels, "S_sub", b10, "mean_log10_unfit", sub_paired)
              - _series(levels, "S_sub", b02, "mean_log10_unfit", sub_paired))
        rci = paired_ci(cfg, ds / dn, draws)
        dci = paired_ci(cfg, ds - dn, draws)
        sub_evidence = (f"paired seeds={sub_paired}; none drift {dn.mean():+.6f}; "
                        f"sub drift {ds.mean():+.6f}; ratio {rci['point']:.6f} CI "
                        f"[{rci['ci_lo']:.6f}, {rci['ci_hi']:.6f}]; diff "
                        f"{dci['point']:+.6f} CI [{dci['ci_lo']:+.6f}, {dci['ci_hi']:+.6f}]")
    else:
        sub_evidence = f"unavailable; paired complete seeds={sub_paired}"
    rows.append(dict(pred="P-dose", scope="B10-B02 degradation ratio S_sub/S_none",
                     verdict="REPORT_ONLY" if len(sub_paired) >= int(I["min_paired_seeds"])
                     else CONTRAST_INVALID_TOO_FEW_PAIRED,
                     evidence=sub_evidence, ci_basis="paired percentile"))

    ledger_rows = []
    for arm in ("S_none", "S_main", "S_sub"):
        if not valid[arm]:
            continue
        seeds = included[arm]
        draws = _draws(cfg, len(seeds))
        dead = [float(_series(levels, arm, b10, f"L{layer}_strict_dead_frac", seeds).mean())
                for layer in (1, 2)]
        rows.append(dict(pred="dead", scope=f"B10 strict_dead_frac {arm}",
                         verdict="REPORT_ONLY", evidence=f"L1 {dead[0]:.6f}; L2 {dead[1]:.6f}",
                         ci_basis=""))
        for layer in (1, 2):
            m02 = _series(levels, arm, b02, f"L{layer}_M_median_alive", seeds)
            m10 = _series(levels, arm, b10, f"L{layer}_M_median_alive", seeds)
            bb02 = _series(levels, arm, b02, f"L{layer}_B_median_alive", seeds)
            bb10 = _series(levels, arm, b10, f"L{layer}_B_median_alive", seeds)
            mci, bci = paired_ci(cfg, m10 - m02, draws), paired_ci(cfg, bb10 - bb02, draws)
            ev = (f"seeds={seeds}; M {m02.mean():+.6f}->{m10.mean():+.6f}, delta "
                  f"{mci['point']:+.6f} CI [{mci['ci_lo']:+.6f}, {mci['ci_hi']:+.6f}]; "
                  f"B {bb02.mean():+.6f}->{bb10.mean():+.6f}, delta {bci['point']:+.6f} "
                  f"CI [{bci['ci_lo']:+.6f}, {bci['ci_hi']:+.6f}]")
            rows.append(dict(pred="ledger", scope=f"B02->B10 {arm} L{layer}",
                             verdict="REPORT_ONLY", evidence=ev,
                             ci_basis="paired percentile"))
            ledger_rows.append(dict(
                arm=arm, layer=layer, n=len(seeds), M_B02=m02.mean(), M_B10=m10.mean(),
                M_delta=mci["point"], M_ci_lo=mci["ci_lo"], M_ci_hi=mci["ci_hi"],
                B_B02=bb02.mean(), B_B10=bb10.mean(), B_delta=bci["point"],
                B_ci_lo=bci["ci_lo"], B_ci_hi=bci["ci_hi"],
            ))

    verdict = pd.DataFrame(rows)
    verdict.to_csv(outdir / "verdict.csv", index=False)
    endpoints = pd.DataFrame({"seed": list(range(10))})
    for arm in ("S_none", "S_main", "S_sub"):
        for seed in included[arm]:
            mask = endpoints.seed == seed
            for block, tag in ((b02, "B02"), (b10, "B10")):
                endpoints.loc[mask, f"{arm}_{tag}_meanlog10unfit"] = _series(
                    levels, arm, block, "mean_log10_unfit", [seed])[0]
            endpoints.loc[mask, f"{arm}_drift"] = (
                endpoints.loc[mask, f"{arm}_B10_meanlog10unfit"].iloc[0]
                - endpoints.loc[mask, f"{arm}_B02_meanlog10unfit"].iloc[0])
    if ratio_values is not None:
        for seed, value in zip(paired, ratio_values):
            endpoints.loc[endpoints.seed == seed, "S_main_over_S_none_ratio"] = value
    endpoints.to_csv(outdir / "paired_endpoints.csv", index=False)

    exclusion_rows = []
    for arm in ("S_none", "S_main", "S_sub"):
        events = {int(event["seed"]): event for event in meta[arm]["exclusion_events"]}
        for seed in range(10):
            event = events.get(seed)
            exclusion_rows.append(dict(
                arm=arm, seed=seed, excluded=int(event is not None),
                detected_step="" if event is None else event["detected_step"],
                detected_task="" if event is None else event["detected_task"],
                nonfinite_tensors="" if event is None else ";".join(event["nonfinite_tensors"]),
                arm_status=meta[arm]["status"],
            ))
    pd.DataFrame(exclusion_rows).to_csv(outdir / "exclusions.csv", index=False)
    _figure(cfg, frame, outdir)
    result = dict(main_verdict=main_verdict, paired_seeds=paired,
                  n_paired=len(paired), ratio=ratio_ci, difference=diff_ci,
                  blocks=dict(B02=b02, B10=b10), ledger=ledger_rows)
    _summary(cfg, outdir, verdict, levels, meta, result)
    return result


def _figure(cfg: dict, frame: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    colors = {"S_none": "#555555", "S_main": "#2b8cbe", "S_sub": "#e34a33"}
    panels = [
        ("unfit", "exact-support unfit", True),
        ("L1_strict_dead_frac", "strict_dead_frac L1", False),
        ("L2_strict_dead_frac", "strict_dead_frac L2", False),
        ("L1_M_median_alive", "alive median M L1", False),
        ("L1_B_median_alive", "alive median B L1", False),
        ("L2_M_median_alive", "alive median M L2", False),
    ]
    for (key, label, logy), axis in zip(panels, axes.flat):
        for arm in ("S_none", "S_main", "S_sub"):
            group = frame[frame.arm == arm].groupby("task")[key].median()
            if not group.empty:
                axis.plot(group.index, group.values, color=colors[arm], lw=1.1,
                          label=f"{arm} lambda={arm_lambda(cfg, arm):g}")
        axis.set_xlabel("task")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        if logy:
            axis.set_yscale("log")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("bias_wd_std_seediso_0901 — seed-isolated rerun")
    fig.tight_layout()
    fig.savefig(outdir / "fig_bias_wd_std_seediso.png", dpi=150)
    plt.close(fig)


def _summary(cfg: dict, outdir: Path, verdict: pd.DataFrame, levels: pd.DataFrame,
             meta: dict[str, dict], result: dict) -> None:
    P, b10 = cfg["bias_wd"], result["blocks"]["B10"]
    valid_arms = [arm for arm in ("S_none", "S_main", "S_sub")
                  if meta[arm]["status"] in VALID_ARM_STATUSES]
    late = levels[levels.block == b10]
    table = (late.groupby("arm")[["mean_log10_unfit", "log10_mean_unfit",
                                   "L1_strict_dead_frac", "L2_strict_dead_frac",
                                   "L1_M_median_alive", "L1_B_median_alive",
                                   "L2_M_median_alive", "L2_B_median_alive"]]
             .mean().reindex(valid_arms).reset_index())
    exclusion_table = pd.DataFrame([
        dict(arm=arm, status=meta[arm]["status"],
             excluded=meta[arm]["excluded_seeds"], included=meta[arm]["included_seeds"],
             n_included=len(meta[arm]["included_seeds"]))
        for arm in ("S_none", "S_main", "S_sub")
    ])
    lines = [
        "# bias_wd_std_seediso_0901 — seed隔離再走の結果", "",
        f"事前登録: [`{cfg['spec']}`](../../{cfg['spec']})。lambda・窓・判定境界は前走据え置き。", "",
        f"主判定は **{result['main_verdict']}**。paired complete seeds = "
        f"{result['paired_seeds']} (n={result['n_paired']})。", "",
        "## Verdict", "", markdown_table(verdict), "",
        "## 除外", "", markdown_table(exclusion_table), "",
        f"## B10（task {P['late_block_tasks'][0]}–{P['late_block_tasks'][1]}）", "",
        markdown_table(table), "",
        "## 規約", "",
        f"- 各腕の除外上限2/10。3本目で `{ARM_INVALID_EXCLUSION_LIMIT}`",
        "- seedの部分軌道は使わず、その腕から全時点を除外",
        f"- 主比較は完走seed共通集合、最低8本。bootstrap B={P['bootstrap_B']}、"
        f"seed={P['bootstrap_seed']}",
        f"- B02=task {P['early_block_tasks'][0]}–{P['early_block_tasks'][1]}、"
        f"B10=task {P['late_block_tasks'][0]}–{P['late_block_tasks'][1]}、床={P['unfit_floor_L2']:g}",
        "- 隔離後も10行分の入力乱数を消費し、非停止seedの対応軌道を維持", "",
        "## 引いてはいけない線", "",
        "- 除外後の結果を10/10完走と同一視しない。除外seedとnを常に併記する",
        "- `LOP_PERSISTS` でもcenteredでのb-WD効果を否定しない",
        "- `LOP_REMOVED` でもmu駆動説の棄却まで飛ばさない",
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


OUTPUTS = (
    "verdict.csv", "summary.md", "paired_endpoints.csv", "exclusions.csv",
    "task_end_metrics.csv", "block_levels.csv", "run_sanity.json",
    "config_used.yaml", "fig_bias_wd_std_seediso.png",
)


def _shard(outdir: Path) -> Path:
    path = outdir / "shards"
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--s0", action="store_true")
    parser.add_argument("--s1s2", action="store_true")
    parser.add_argument("--isolation-gate", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arm")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--outdir")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(str(cfg_path))
    validate_config(cfg, full=not args.smoke)
    require_omp(int(cfg["bias_wd"]["omp_num_threads"]))
    outdir = Path(args.outdir).resolve() if args.outdir else outdir_of(cfg)
    outdir.mkdir(parents=True, exist_ok=True)
    gate_dir = Path(ROOT) / "results" / "_gate_bias_wd_std_seediso_0901"
    started = time.time()

    if args.s1s2:
        s1_s2_algebra(cfg, gate_dir)
        return
    if args.isolation_gate:
        isolation_gate(cfg, gate_dir)
        return
    if args.s0:
        s0_replay(cfg, gate_dir)
        return
    if args.smoke:
        smoke = copy.deepcopy(cfg)
        smoke["common"]["checkpoints"] = []
        smoke_dir = Path(ROOT) / "results" / "_smoke_bias_wd_std_seediso_0901"
        for arm in ("S_none", "S_main", "S_sub"):
            result = run_arm_seediso(smoke, arm, smoke_dir, total_steps=30_000,
                                     task_period=10_000, guard_every=1_000,
                                     keep_unit_arrays=False, write_logs=False)
            if result["status"] != COMPLETE or not result["sanity"]["pass_"]:
                raise RuntimeError(f"smoke failed: {arm}: {result}")
        print("SMOKE PASS", flush=True)
        return

    for name in ("s0_replay.json", "s1_s2_algebra.json", "isolation_gate.json"):
        path = gate_dir / name
        if not path.exists() or not json.loads(path.read_text(encoding="utf-8"))["pass_"]:
            raise RuntimeError(f"missing or failed gate: {path}")

    total = int(cfg["common"]["total_steps"])
    period = int(cfg["phase1"]["task_period"])
    guard = int(cfg["bias_wd"]["guard_every"])
    todo = ([args.arm] if args.arm else [a["name"] for a in cfg["arms"]])
    if args.arm and args.arm not in [a["name"] for a in cfg["arms"]]:
        raise SystemExit(f"unknown arm {args.arm}")
    if not args.analyze_only:
        for arm in todo:
            result = run_arm_seediso(cfg, arm, outdir, total_steps=total,
                                     task_period=period, guard_every=guard)
            result["frame"].to_csv(_shard(outdir) / f"{arm}.csv", index=False)
            (_shard(outdir) / f"{arm}.json").write_text(json.dumps(
                {key: value for key, value in result.items() if key != "frame"},
                indent=2, ensure_ascii=False), encoding="utf-8")
        if args.arm:
            return

    frames, meta = [], {}
    for arm in ("S_none", "S_main", "S_sub"):
        meta[arm] = json.loads((_shard(outdir) / f"{arm}.json").read_text(encoding="utf-8"))
        if meta[arm]["status"] in VALID_ARM_STATUSES:
            frames.append(pd.read_csv(_shard(outdir) / f"{arm}.csv"))
    if not frames:
        raise RuntimeError("no valid arms to analyze")
    pd.concat(frames, ignore_index=True).to_csv(outdir / "task_end_metrics.csv", index=False)

    s0 = json.loads((gate_dir / "s0_replay.json").read_text(encoding="utf-8"))
    s12 = json.loads((gate_dir / "s1_s2_algebra.json").read_text(encoding="utf-8"))
    iso = json.loads((gate_dir / "isolation_gate.json").read_text(encoding="utf-8"))
    run_sanity = dict(
        S0_pass=bool(s0["pass_"]), S1_S2_pass=bool(s12["pass_"]),
        S_iso_S_cap_pass=bool(iso["pass_"]), isolation_gate=iso,
        S3={arm: dict(pass_=meta[arm]["sanity"]["pass_"],
                      max_relerr=meta[arm]["sanity"]["max_relerr"],
                      quantization_violations=meta[arm]["sanity"]["n_quantization_violations"],
                      wall_violations=meta[arm]["sanity"]["n_wall_identity_violations"])
            for arm in meta},
        S4_seed_isolation={arm: dict(status=meta[arm]["status"],
                                     excluded_seeds=meta[arm]["excluded_seeds"],
                                     events=meta[arm]["exclusion_events"])
                           for arm in meta},
        training_elapsed_sec={arm: meta[arm]["elapsed_sec"] for arm in meta},
    )
    (outdir / "run_sanity.json").write_text(
        json.dumps(run_sanity, indent=2, ensure_ascii=False), encoding="utf-8")
    with (outdir / "config_used.yaml").open("w") as stream:
        yaml.safe_dump(cfg, stream, allow_unicode=True, sort_keys=False)
    result = analyze(cfg, outdir, meta)
    (outdir / "provenance.json").write_text(json.dumps(provenance(
        "bias_wd_std_seediso_0901", cfg_path, cfg, outdir,
        dict(analysis=result, run_sanity=run_sanity, parent="bias_wd_std_0901@af70722"),
        started, sys.argv, OUTPUTS), indent=2, ensure_ascii=False), encoding="utf-8")
    print(pd.read_csv(outdir / "verdict.csv")[["pred", "scope", "verdict"]]
          .to_string(index=False), flush=True)
    print(f"ALL DONE -> {outdir}", flush=True)


if __name__ == "__main__":
    main()

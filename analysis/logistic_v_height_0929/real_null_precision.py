"""Post-hoc float64 check of readout-scale equivariance; not source replay.

Task50 parameters and moments are promoted from their saved float32 values
BEFORE scaling. The float64 expm1+1 backward has a different quantization cutoff
from the primary float32 experiment. All primary artifacts remain untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("real_interventions_for_precision", HERE / "real_interventions.py")
real = importlib.util.module_from_spec(spec)
spec.loader.exec_module(real)

GAINS = (0.1, 10.0)
CHECKPOINTS = (0, 1, 10, 100, 500, 1000, 2000, 4000, 8000, 12000)
THRESHOLDS = (1e-6, 1e-4, 1e-2, 1e-1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_write(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_promoted(seed, folder):
    path = folder / f"real_s{seed}_t50.npz"
    with np.load(path) as saved:
        X = torch.tensor(saved["X"], dtype=torch.float64)
        params = [torch.tensor(saved[name], dtype=torch.float64, requires_grad=True) for name in real.NAMES]
        optim = real.native_optimizer(params)
        for name, param in zip(real.NAMES, params):
            optim.state[param] = {
                "step": torch.tensor(float(saved[f"step_{name}"]), dtype=torch.float32),
                "exp_avg": torch.tensor(saved[f"m_{name}"], dtype=torch.float64),
                "exp_avg_sq": torch.tensor(saved[f"q_{name}"], dtype=torch.float64),
            }
    optim.param_groups[1]["lr"] = 0.0
    with np.load(folder / f"real_s{seed}_task_plan.npz") as saved:
        plans = {}
        for task in (51, 52, 53):
            y = torch.tensor(saved[f"y_t{task}"], dtype=torch.int64)
            Y = torch.nn.functional.one_hot(y, 10).to(torch.float64)
            plans[task] = (Y, torch.tensor(saved[f"order_t{task}"], dtype=torch.int64))
    return X, params, optim, plans


def reference_step(arm, xb, yb):
    # Only retain the tiny logits gradient, not every hidden gradient/state.
    cap = real.backward(arm["P"], arm["optim"], xb, yb, capture=True)
    arm["optim"].step()
    return cap["gradient_logits"]


def replay_step(arm, xb, yb, gradient_logits):
    real.backward(arm["P"], arm["optim"], xb, yb, reference_gradient_logits=gradient_logits)
    arm["optim"].step()


def same_state_oracle(reference, xb, yb, seed, updates):
    baseline = real.clone_arm(reference["P"], reference["optim"], "REF", 1.0)
    controls = [real.clone_arm(reference["P"], reference["optim"], "E", gain) for gain in GAINS]
    initial = [param.detach().clone() for param in baseline["P"]]
    gradient_logits = reference_step(baseline, xb, yb)
    rows = []
    for arm in controls:
        replay_step(arm, xb, yb, gradient_logits)
        difference = max(float((arm["P"][j] - baseline["P"][j]).detach().abs().max()) for j in (0, 1))
        step_norm = max(float((baseline["P"][j] - initial[j]).detach().abs().max()) for j in (0, 1))
        bias_difference = float((arm["P"][3] - baseline["P"][3]).detach().abs().max())
        rows.append({
            "seed": seed, "updates": updates, "gain": arm["scale"],
            "same_state_next_step_hidden_abs_error": difference,
            "baseline_hidden_step_max": step_norm,
            "error_divided_by_step_max": difference / max(step_norm, 1e-300),
            "output_bias_abs_error": bias_difference,
        })
        assert bias_difference == 0.0
    return rows


@torch.no_grad()
def record(seed, task, task_step, updates, X, Y, reference, arms):
    ref_unit, ref_ce, _ = real.diagnostics(reference["P"], X, Y)
    summary, units = [], []
    refw = reference["P"][0]
    ref_hidden_norm = torch.sqrt(sum((reference["P"][j] ** 2).sum() for j in (0, 1)))
    for arm in arms:
        unit, ce, _ = real.diagnostics(arm["P"], X, Y)
        delta = unit["T"] - ref_unit["T"]
        abs_delta = np.abs(delta)
        hidden_max = max(float((arm["P"][j] - reference["P"][j]).abs().max()) for j in (0, 1))
        hidden_diff_norm = torch.sqrt(sum(((arm["P"][j] - reference["P"][j]) ** 2).sum() for j in (0, 1)))
        bias_diff = float((arm["P"][3] - reference["P"][3]).abs().max())
        wrel = ((arm["P"][0] - refw).norm(dim=1) / refw.norm(dim=1).clamp_min(1e-300)).numpy()
        finite = all(bool(torch.isfinite(p).all()) for p in arm["P"])
        for j in (0, 1, 2, 3):
            state = arm["optim"].state[arm["P"][j]]
            finite = finite and bool(torch.isfinite(state["exp_avg"]).all()) and bool(torch.isfinite(state["exp_avg_sq"]).all())
        finite = finite and bool(np.isfinite(unit["T"]).all()) and bool((unit["s"] > 0).all())
        assert finite and bias_diff == 0.0
        row = {
            "seed": seed, "task": task, "task_step": task_step, "updates": updates, "gain": arm["scale"],
            "max_abs_T_difference": float(abs_delta.max()), "median_abs_T_difference": float(np.median(abs_delta)),
            "mean_abs_T_difference": float(abs_delta.mean()), "signed_median_T_difference": float(np.median(delta)),
            "max_abs_hidden_parameter_difference": hidden_max,
            "hidden_parameter_relative_l2_error": float(hidden_diff_norm / ref_hidden_norm),
            "max_unit_w_relative_l2_error": float(wrel.max()), "median_unit_w_relative_l2_error": float(np.median(wrel)),
            "max_abs_output_bias_difference": bias_diff,
            "max_abs_top_difference": float(np.max(np.abs(unit["top"] - ref_unit["top"]))),
            "finite_states": finite, "reference_ce": ref_ce, "arm_own_forward_ce": ce,
            "reference_median_T": float(np.median(ref_unit["T"])),
            "arm_median_T": float(np.median(unit["T"])),
        }
        for threshold in THRESHOLDS:
            row[f"units_abs_T_gt_{threshold:g}"] = int(np.sum(abs_delta > threshold))
        summary.append(row)
        for i in range(real.H):
            units.append({
                "seed": seed, "task": task, "task_step": task_step, "updates": updates, "gain": arm["scale"],
                "unit": i, "reference_T": float(ref_unit["T"][i]), "T": float(unit["T"][i]),
                "T_difference": float(delta[i]), "reference_top": float(ref_unit["top"][i]), "top": float(unit["top"][i]),
                "reference_s": float(ref_unit["s"][i]), "s": float(unit["s"][i]),
                "reference_pplus": float(ref_unit["pplus"][i]), "pplus": float(unit["pplus"][i]),
                "w_relative_l2_error": float(wrel[i]),
            })
    return summary, units


def run_seed(seed, folder):
    start = time.perf_counter()
    X, P, optim, plans = load_promoted(seed, folder)
    reference = real.clone_arm(P, optim, "REF", 1.0)
    arms = [real.clone_arm(P, optim, "E", gain) for gain in GAINS]
    frozen_v = [arm["P"][2].detach().clone() for arm in [reference] + arms]
    summaries, units, oracles = [], [], []
    Y, order = plans[51]
    rs, ru = record(seed, 51, 0, 0, X, Y, reference, arms)
    summaries.extend(rs); units.extend(ru)
    oracles.extend(same_state_oracle(reference, X[order[0]], Y[order[0]], seed, 0))
    updates = 0
    for task in (51, 52, 53):
        Y, order = plans[task]
        for task_step in range(1, 4001):
            index = order[task_step - 1]
            xb, yb = X[index], Y[index]
            gradient_logits = reference_step(reference, xb, yb)
            for arm in arms:
                replay_step(arm, xb, yb, gradient_logits)
            updates += 1
            if updates in CHECKPOINTS:
                rs, ru = record(seed, task, task_step, updates, X, Y, reference, arms)
                summaries.extend(rs); units.extend(ru)
            if updates == 100:
                elapsed = time.perf_counter() - start
                print(json.dumps({"event": "precision_timing", "seed": seed, "updates": updates,
                                  "seconds": elapsed, "rough_remaining_seconds_this_seed": elapsed * 119}), flush=True)
        oracles.extend(same_state_oracle(reference, xb, yb, seed, updates))
        csv_write(folder / f"real_null_precision_s{seed}_summary.csv", summaries)
        csv_write(folder / f"real_null_precision_s{seed}_units.csv", units)
        csv_write(folder / f"real_null_precision_s{seed}_same_state.csv", oracles)
        print(json.dumps({"event": "precision_task_done", "seed": seed, "task": task,
                          "seconds": time.perf_counter() - start, "end_rows": summaries[-2:]}), flush=True)
    final = {}
    for arm in [reference] + arms:
        final.update({f"{arm['tag']}__{key}": value for key, value in real.snapshot(arm["P"], arm["optim"]).items()})
    np.savez_compressed(folder / f"real_null_precision_s{seed}_final_states.npz", **final)
    for arm, frozen in zip([reference] + arms, frozen_v):
        assert torch.equal(arm["P"][2], frozen)
    meta = {
        "seed": seed, "seconds": time.perf_counter() - start, "dtype": "float64",
        "code_sha256": sha(__file__), "implementation_dependency_sha256": sha(HERE / "real_interventions.py"),
        "source_checkpoint_sha256": sha(folder / f"real_s{seed}_t50.npz"),
        "source_task_plan_sha256": sha(folder / f"real_s{seed}_task_plan.npz"),
        "scope": "Numerical scale-equivariance shadow only; not reproduction of the original float32 trajectory.",
        "backward": "float64 expm1(z)+1; quantization/zero cutoff differs from float32.",
        "initialization": "Promote saved float32 P/M/S to float64 before readout and hidden moment scaling.",
        "step_counter": "Preserved from task50; no reset.",
        "output_bias": "Unscaled moments/epsilon, reference exact CE logits gradient; bit-equality asserted.",
        "tasks": [51, 52, 53], "updates": 12000, "gains": list(GAINS),
        "registration_commit": "616cbef",
    }
    (folder / f"real_null_precision_s{seed}_provenance.json").write_text(json.dumps(meta, indent=2) + "\n")
    return summaries, oracles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--out", type=Path, default=Path("results/logistic_v_height_0929"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_flush_denormal(True)
    config = {
        "dtype": "float64", "seeds": args.seeds, "gains": list(GAINS), "updates": 12000,
        "checkpoints": CHECKPOINTS, "outlier_abs_T_thresholds": THRESHOLDS,
        "registration_commit": "616cbef", "torch": torch.__version__, "numpy": np.__version__,
        "python": platform.python_version(), "code_sha256": sha(__file__),
        "interpretation": "Post-hoc precision check only. Float64 source-ELU arithmetic changes the derivative cutoff; do not replace primary trajectories.",
        "frozen_before_updates": True,
    }
    (args.out / "real_null_precision_config.json").write_text(json.dumps(config, indent=2) + "\n")
    all_rows, all_oracles = [], []
    for seed in args.seeds:
        rows, oracles = run_seed(seed, args.out)
        all_rows.extend(rows); all_oracles.extend(oracles)
    csv_write(args.out / "real_null_precision_summary.csv", all_rows)
    csv_write(args.out / "real_null_precision_same_state.csv", all_oracles)


if __name__ == "__main__":
    main()

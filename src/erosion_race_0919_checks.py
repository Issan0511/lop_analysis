#!/usr/bin/env python3
"""Independent algebra and pilot checks for erosion_race_0919.

Only pilot seeds 9995..9999 are used; this file never launches registered seeds.
The algebra checks use CPU float64 and independent objectives/recurrences.
Engine checks compare actual parameter, optimizer, adaptive-alpha, and RNG states.
An observer check is not passed merely because its reported metrics agree.

Examples:
  python3 src/erosion_race_0919_checks.py --only algebra
  python3 src/erosion_race_0919_checks.py --only helpers,baseline,observer,graph --device cuda

Outputs go to stdout, or the explicitly supplied --out JSON. Pilot artifacts live
in --pilot-dir (or a temporary directory) and are not research observations.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import sys
import tempfile
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import rlcifar_mlp_battle_0918 as Parent
from src import pmnist_rlcifar_0907 as RC

PILOT_SEEDS = (9998, 9999)
BASELINE_SEEDS = (9995, 9996, 9997, 9998, 9999)
ARMS = ("R", "GELU", "ELU", "LR", "LK001", "SNA")
ATOL64 = 2e-12
RTOL64 = 2e-12


def _gen(seed=9999):
    return torch.Generator(device="cpu").manual_seed(seed)


def _error(a, b):
    return float((a.detach() - b.detach()).abs().max())


def _close(a, b, *, atol=ATOL64, rtol=RTOL64):
    return bool(torch.allclose(a, b, atol=atol, rtol=rtol))


def check_ce():
    """Differentiate CE and two independently specified scalar losses."""
    g = _gen()
    x = torch.randn(11, 5, generator=g, dtype=torch.float64)
    w = torch.randn(5, 7, generator=g, dtype=torch.float64).requires_grad_()
    b = torch.randn(7, generator=g, dtype=torch.float64).requires_grad_()
    v = torch.randn(7, 10, generator=g, dtype=torch.float64).requires_grad_()
    c = torch.randn(10, generator=g, dtype=torch.float64).requires_grad_()
    params = (w, b, v, c)
    logits = F.elu(x @ w + b) @ v + c
    labels = torch.randint(10, (11,), generator=g)
    target = F.one_hot(labels, 10).to(torch.float64)
    total = F.cross_entropy(logits, labels)
    conf = (logits.logsumexp(-1) - logits.mean(-1)).mean()
    label = -((target - .1) * logits).sum(-1).mean()
    gt = torch.autograd.grad(total, params, retain_graph=True)
    gc = torch.autograd.grad(conf, params, retain_graph=True)
    gl = torch.autograd.grad(label, params)
    errs = [_error(a, b + c) for a, b, c in zip(gt, gc, gl)]
    # Both plausible errors must be visible to this fixture.
    wrong_sign = max(_error(a, b - c) for a, b, c in zip(gt, gc, gl))
    missing_batch_mean = max(_error(a, b + 11 * c) for a, b, c in zip(gt, gc, gl))
    return {"pass": bool(_close(total, conf + label)
                         and all(_close(a, b + c) for a, b, c in zip(gt, gc, gl))
                         and wrong_sign > 1e-3 and missing_batch_mean > 1e-3),
            "loss_closure_abs": _error(total, conf + label),
            "gradient_closure_abs": errs,
            "mutation_wrong_label_sign_error": wrong_sign,
            "mutation_missing_batch_mean_error": missing_batch_mean}


def _adam_fixture():
    g = _gen()
    m = torch.zeros(3, 9, dtype=torch.float64)
    v = torch.zeros_like(m)
    # Real inherited moments, rather than arbitrary inaccessible optimizer state.
    for _ in range(13):
        grad = torch.randn(m.shape, generator=g, dtype=m.dtype)
        m = .9 * m + .1 * grad
        v = .999 * v + .001 * grad.square()
    hist = m.clone()
    conf = torch.zeros_like(m)
    label = torch.zeros_like(m)
    rows = []
    for step in range(1, 22):
        gc = torch.randn(m.shape, generator=g, dtype=m.dtype) * .7
        gl = torch.randn(m.shape, generator=g, dtype=m.dtype) * 1.3
        grad = gc + gl
        m = .9 * m + .1 * grad
        v = .999 * v + .001 * grad.square()
        hist = .9 * hist
        conf = .9 * conf + .1 * gc
        label = .9 * label + .1 * gl
        global_step = 13 + step
        inv1, inv2 = 1 / (1 - .9 ** global_step), 1 / (1 - .999 ** global_step)
        denom = (v * inv2).sqrt() + 1e-8
        parts = tuple(-.001 * component * inv1 / denom for component in (conf, hist, label))
        actual = -.001 * m * inv1 / denom
        rows.append({"m": m.clone(), "v": v.clone(), "hist": hist.clone(),
                     "conf": conf.clone(), "label": label.clone(), "parts": parts,
                     "actual": actual, "inv1": inv1, "inv2": inv2, "task_step": step})
    return rows


def check_adam():
    rows = _adam_fixture()
    moment_error = max(_error(r["m"], r["conf"] + r["hist"] + r["label"]) for r in rows)
    step_error = max(_error(r["actual"], sum(r["parts"])) for r in rows)
    omit_history_error = max(_error(r["actual"], r["parts"][0] + r["parts"][2]) for r in rows)
    reset_bias_error = max(_error(r["actual"], r["actual"] *
                          ((1 / (1 - .9 ** r["task_step"])) / r["inv1"])) for r in rows)
    return {"pass": bool(moment_error < ATOL64 and step_error < ATOL64
                         and omit_history_error > 1e-5 and reset_bias_error > 1e-5),
            "numerator_three_part_max_abs": moment_error,
            "actual_denominator_update_closure_max_abs": step_error,
            "mutation_omit_history_error": omit_history_error,
            "mutation_reset_global_bias_correction_error": reset_bias_error}


def _projection_fixture():
    mu = torch.tensor([[.2, .4, .7, .1, .8]], dtype=torch.float64)
    h = torch.cat((mu, torch.ones(1, 1, dtype=mu.dtype)), -1)
    g = _gen()
    u = torch.randn(1, 4, 6, generator=g, dtype=mu.dtype) * .01
    # Explicit negative, positive, and zero mean-displacement units.
    u[:, 0] = -.03 * h
    u[:, 1] = .02 * h
    u[:, 2] = 0
    dot = (u * h[:, None]).sum(-1)
    correction = (-dot.clamp(max=0) / h.square().sum(-1)[:, None])[:, :, None] * h[:, None]
    return mu, h, u, dot, correction


def check_projection():
    mu, h, u, dot, corr = _projection_fixture()
    h2 = h.square().sum(-1)[:, None, None]
    project = lambda a: (a * h[:, None]).sum(-1, keepdim=True) * h[:, None] / h2
    repaired = ((u + corr) * h[:, None]).sum(-1)
    orth_error = _error((u + corr) - project(u + corr), u - project(u))
    # Deliberately exhibit why 'centered input features are preserved' is false.
    centered_x = torch.tensor([.3, -.1, .2, -.4, .1], dtype=mu.dtype)
    centered_feature_change = abs(float(corr[0, 0, :-1] @ centered_x))
    return {"pass": bool(_close(repaired, dot.clamp(min=0))
                         and torch.equal(corr[:, 1:3], torch.zeros_like(corr[:, 1:3]))
                         and orth_error < ATOL64 and centered_feature_change > 1e-6),
            "mean_displacement_closure_abs": _error(repaired, dot.clamp(min=0)),
            "parameter_h_orthogonal_component_error": orth_error,
            "centered_feature_change_counterexample": centered_feature_change,
            "scope": "Preserves parameter components orthogonal to h, not centered-input features."}


def check_activation():
    g = _gen()
    x = torch.rand(17, 6, generator=g, dtype=torch.float64)
    w = (torch.rand(6, generator=g, dtype=torch.float64) + .1).requires_grad_()
    b = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    u = torch.randn(17, generator=g, dtype=torch.float64)
    z = x @ w + b
    grad_elu = torch.autograd.grad(F.elu(z), (w, b), u, retain_graph=True)
    grad_leaky = torch.autograd.grad(F.leaky_relu(z, .1), (w, b), u)
    matched = all(torch.equal(a, b) for a, b in zip(grad_elu, grad_leaky))
    zn = torch.linspace(-3, -.2, 127, dtype=torch.float64)
    centered = lambda a: a - a.mean()
    elu_errors, leak_errors = [], []
    for delta in (-.7, -2.0):
        a0, a1 = F.elu(zn), F.elu(zn + delta)
        elu_errors.append(_error(centered(a1), math.exp(delta) * centered(a0)))
        elu_errors.append(abs(float(a1.var(unbiased=False) / a0.var(unbiased=False)) - math.exp(2 * delta)))
        leak_errors.append(_error(centered(F.leaky_relu(zn + delta, .1)),
                                 centered(F.leaky_relu(zn, .1))))
    mixed = torch.randn(7, 59, generator=g, dtype=torch.float64) * 4 - .5
    floor_slack = {}
    for slope in (.1, .01):
        a = F.leaky_relu(mixed, slope)
        floor_slack[str(slope)] = float((a.var(-1, unbiased=False) -
                                        slope ** 2 * mixed.var(-1, unbiased=False)).min())
    zp, zm, slope = 1.3, -.8, .1
    mm = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    aa = F.leaky_relu(torch.tensor([zp, zm], dtype=mm.dtype) + mm, slope)
    derivative, = torch.autograd.grad(.5 * aa.var(unbiased=False), (mm,))
    expected = .25 * (zp - slope * zm) * (1 - slope)
    return {"pass": bool(matched and max(elu_errors + leak_errors) < ATOL64
                         and min(floor_slack.values()) >= -ATOL64
                         and abs(float(derivative) - expected) < ATOL64),
            "matched_positive_branch_gradient_bit_equal": matched,
            "elu_negative_translation_errors": elu_errors,
            "leaky_negative_translation_errors": leak_errors,
            "leaky_variance_floor_min_slack": floor_slack,
            "two_point_centered_variance_derivative_error": abs(float(derivative) - expected)}


def check_sna_observer():
    act = Parent.make_act("SNA")
    act.init_state(1, torch.device("cpu"), "erosion-check-9999")
    gg = _gen()
    z1 = torch.randn(1, 19, 100, generator=gg)
    z2 = torch.randn(1, 19, 100, generator=gg)
    act.update(z1, z2)  # State is nontrivial before observing it.
    state = [v.clone() for v in act.V]
    rng = torch.get_rng_state().clone()
    for _ in range(2):
        for layer, z in enumerate((z1, z2)):
            act.phi(z, layer, train=False)
            act.dphi(z, layer)
            act.alpha(layer)
        act.stats(0)
    return {"pass": bool(all(torch.equal(a, b) for a, b in zip(state, act.V))
                         and torch.equal(rng, torch.get_rng_state())),
            "adaptive_variance_state_bit_equal": all(torch.equal(a, b) for a, b in zip(state, act.V)),
            "cpu_rng_bit_equal": torch.equal(rng, torch.get_rng_state())}


def _engine():
    return importlib.import_module("src.erosion_race_0919")


def check_helpers():
    engine = _engine()
    errs = []
    for row in _adam_fixture():
        parts = engine.split_adam_components(row["conf"], row["hist"], row["m"], row["v"],
                                             row["inv1"], row["inv2"], .001)
        errs.extend(_error(a, b) for a, b in zip(parts, row["parts"]))
        errs.append(_error(sum(parts), row["actual"]))
    mu, h, u, dot, corr = _projection_fixture()
    cw, cb, reported = engine.confinement_correction(u[..., :-1], u[..., -1], mu)
    got = torch.cat((cw, cb[..., None]), -1)
    windows = []
    for task in (1, 2):
        for step in (0, 1, 4999, 5000, 5001, 9999, 10000, 10001, 30000):
            for mode in ("base", "early", "late"):
                expected = task == 1 and ((mode == "early" and 1 <= step <= 5000)
                                         or (mode == "late" and 5001 <= step <= 10000))
                if engine.intervention_active(mode, task, step) != expected:
                    windows.append({"mode": mode, "task": task, "step": step})
    return {"pass": bool(max(errs) < ATOL64 and _close(got, corr)
                         and _close(reported, dot) and not windows),
            "adam_component_max_abs": max(errs),
            "projection_max_abs": _error(got, corr),
            "reported_displacement_error": _error(reported, dot),
            "intervention_window_failures": windows}


def _tree_compare(a, b, prefix=""):
    errors = []
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        a, b = a.detach().cpu(), b.detach().cpu()
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
            error = None if a.shape != b.shape else _error(a.to(torch.float64), b.to(torch.float64))
            errors.append({"path": prefix, "max_abs": error, "shapes": [list(a.shape), list(b.shape)]})
    elif isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            errors.append({"path": prefix, "key_mismatch": [str(list(a)), str(list(b))]})
        for key in a.keys() & b.keys():
            errors.extend(_tree_compare(a[key], b[key], f"{prefix}.{key}"))
    elif isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            errors.append({"path": prefix, "length_mismatch": [len(a), len(b)]})
        for i, (aa, bb) in enumerate(zip(a, b)):
            errors.extend(_tree_compare(aa, bb, f"{prefix}[{i}]"))
    elif a != b:
        errors.append({"path": prefix, "values": [str(a), str(b)]})
    return errors


def _state_compare(a, b, diagnostics=False):
    keys = ("P", "m", "v", "tc", "V", "g_lab", "g_batch")
    if diagnostics:
        keys += ("m_conf", "m_hist")
    missing = [key for key in keys if key not in a or key not in b]
    errors = [{"missing_required_state": missing}] if missing else []
    for key in keys:
        if key in a and key in b:
            errors.extend(_tree_compare(a[key], b[key], key))
    return {"pass": not errors, "compared_keys": list(keys), "differences": errors[:30],
            "difference_count": len(errors)}


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _reset_rng():
    random.seed(9999)
    np.random.seed(9999)
    torch.manual_seed(9999)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(9999)


def _global_rng():
    ns = np.random.get_state()
    return {"python": random.getstate(), "numpy": (ns[0], ns[1].copy(), *ns[2:]),
            "torch": torch.get_rng_state().clone(),
            "cuda": [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def _global_compare(a, b):
    return (a["python"] == b["python"] and a["numpy"][0] == b["numpy"][0]
            and np.array_equal(a["numpy"][1], b["numpy"][1])
            and a["numpy"][2:] == b["numpy"][2:]
            and torch.equal(a["torch"], b["torch"])
            and len(a["cuda"]) == len(b["cuda"])
            and all(torch.equal(x, y) for x, y in zip(a["cuda"], b["cuda"])))


def _dataset(args):
    if args.data_dir:
        RC.DATA_DIR = Path(args.data_dir)
    return RC.Cifar10()


def check_baseline(args, base):
    engine, cifar = _engine(), _dataset(args)
    results = {}
    for arm in args.arms.split(","):
        old, new = base / f"parent_{arm}", base / f"new_{arm}"
        _reset_rng()
        Parent.run(arm, list(BASELINE_SEEDS), ["raw"], 2, 1, torch.device(args.device), old,
                   cifar=cifar, graph=False, checkpoint=True, snapshots=False, progress=lambda _: None)
        _reset_rng()
        engine.run(arm, "base", list(BASELINE_SEEDS), 2, 1, torch.device(args.device), new,
                   cifar=cifar, graph=False, observe=True, progress=lambda _: None)
        results[arm] = _state_compare(_load(old / "ckpt.pt"), _load(new / "final.pt"))
        print(f"baseline {arm}: {'PASS' if results[arm]['pass'] else 'FAIL'}", flush=True)
    return {"pass": all(r["pass"] for r in results.values()), "arms": results,
            "tasks": 2, "updates_per_task": 75, "seeds": list(BASELINE_SEEDS)}


def check_observer(args, base):
    engine, cifar = _engine(), _dataset(args)
    results = {}
    for arm, mode in (("SNA", "base"), ("GELU", "early"), ("GELU", "late")):
        off, on = base / f"observer_off_{arm}_{mode}", base / f"observer_on_{arm}_{mode}"
        _reset_rng()
        engine.run(arm, mode, list(PILOT_SEEDS), 2, 1, torch.device(args.device), off,
                   cifar=cifar, graph=False, observe=False, progress=lambda _: None)
        rng_off = _global_rng()
        _reset_rng()
        engine.run(arm, mode, list(PILOT_SEEDS), 2, 1, torch.device(args.device), on,
                   cifar=cifar, graph=False, observe=True, progress=lambda _: None)
        rng_on = _global_rng()
        r = _state_compare(_load(off / "final.pt"), _load(on / "final.pt"), diagnostics=True)
        r["global_rng_bit_equal"] = _global_compare(rng_off, rng_on)
        r["pass"] &= r["global_rng_bit_equal"]
        results[f"{arm}/{mode}"] = r
    return {"pass": all(r["pass"] for r in results.values()), "cases": results,
            "note": "late arm is inactive at 75-step tasks; the helper/window checks cover activation boundaries."}


def check_graph(args, base):
    if args.device != "cuda" or not torch.cuda.is_available():
        return {"pass": False, "status": "NOT_RUN", "reason": "CUDA is required for graph parity."}
    engine, cifar = _engine(), _dataset(args)
    results = {}
    for arm, mode in (("R", "base"), ("SNA", "base"), ("GELU", "early"), ("ELU", "early")):
        eager, graph = base / f"eager_{arm}_{mode}", base / f"graph_{arm}_{mode}"
        _reset_rng()
        engine.run(arm, mode, list(PILOT_SEEDS), 2, 1, torch.device(args.device), eager,
                   cifar=cifar, graph=False, observe=True, progress=lambda _: None)
        _reset_rng()
        engine.run(arm, mode, list(PILOT_SEEDS), 2, 1, torch.device(args.device), graph,
                   cifar=cifar, graph=True, observe=True, progress=lambda _: None)
        results[f"{arm}/{mode}"] = _state_compare(_load(eager / "final.pt"), _load(graph / "final.pt"),
                                                  diagnostics=True)
    return {"pass": all(r["pass"] for r in results.values()), "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="algebra")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--data-dir")
    parser.add_argument("--pilot-dir")
    parser.add_argument("--out")
    args = parser.parse_args()
    torch.set_num_threads(2)
    checks = {"ce": check_ce, "adam": check_adam, "projection": check_projection,
              "activation": check_activation, "sna-observer": check_sna_observer,
              "helpers": check_helpers}
    names = args.only.split(",")
    if "algebra" in names:
        names = [n for n in names if n != "algebra"] + ["ce", "adam", "projection", "activation", "sna-observer"]
    if "all" in names:
        names = list(checks) + ["baseline", "observer", "graph"]
    temp = None
    if args.pilot_dir:
        base = Path(args.pilot_dir)
        base.mkdir(parents=True, exist_ok=True)
    else:
        temp = tempfile.TemporaryDirectory(prefix="erosion-race-checks-")
        base = Path(temp.name)
    results = {}
    for name in dict.fromkeys(names):
        started = time.monotonic()
        try:
            if name in checks:
                result = checks[name]()
            elif name == "baseline":
                result = check_baseline(args, base)
            elif name == "observer":
                result = check_observer(args, base)
            elif name == "graph":
                result = check_graph(args, base)
            else:
                raise ValueError(f"Unknown check {name!r}")
        except Exception as exc:
            result = {"pass": False, "error": repr(exc), "traceback": traceback.format_exc()}
        result["elapsed_seconds"] = time.monotonic() - started
        results[name] = result
        print(f"{name}: {'PASS' if result.get('pass') else 'FAIL'}", flush=True)
    report = {"pass": all(v.get("pass", False) for v in results.values()),
              "pilot_seeds": list(BASELINE_SEEDS), "registered_seeds_used": [],
              "torch": torch.__version__, "device": args.device, "checks": results}
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    print(serialized)
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(serialized + "\n")
    if temp:
        temp.cleanup()
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

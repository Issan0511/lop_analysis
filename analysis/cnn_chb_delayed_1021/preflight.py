#!/usr/bin/env python3
"""Fail-closed, preregistered GPU qualification. Never runs a production task.

python analysis/cnn_chb_delayed_1021/preflight.py --release COMMIT --out DIR

This command is intentionally separate from CPU synthetic unit tests. It reads
real images only after release verification, under the shared GPU/output locks.
Thresholds are fixed here and in the registered spec; no threshold CLI exists.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from src import cnn_chb_delayed_1021 as M
from src import cnn_chprime_bias_1018 as ORIGINAL_B
from src import cnn_center_feat_1011 as CF

STATE_RELATIVE_LIMIT = 1e-4
GRAD_ATOL, GRAD_RTOL = 1e-7, 1e-6
MEMO_ABSOLUTE_LIMIT = 1 / 1200


def comparison(actual, reference, *, exact=False, gradients=False):
    """Compare every leaf; preserve per-tensor evidence, including nonfinites."""
    rows = []

    def visit(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                rows.append(dict(path=path, passed=False, reason="mapping keys differ"))
            for key in sorted(a.keys() & b.keys(), key=str):
                visit(a[key], b[key], f"{path}/{key}")
        elif isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            if len(a) != len(b):
                rows.append(dict(path=path, passed=False, reason="sequence lengths differ"))
            for i, (aa, bb) in enumerate(zip(a, b)):
                visit(aa, bb, f"{path}/{i}")
        elif torch.is_tensor(a) and torch.is_tensor(b):
            a, b = a.detach().cpu(), b.detach().cpu()
            same_meta = a.shape == b.shape and a.dtype == b.dtype
            bit_equal = same_meta and M.tree_hash(a) == M.tree_hash(b)
            if not same_meta:
                rows.append(dict(path=path, passed=False, reason="tensor shape/dtype differ"))
                return
            if a.is_floating_point():
                finite = torch.isfinite(a) & torch.isfinite(b)
                fixed_unused_alpha = "/fixA/" in path and any(s in path for s in ("/act/", "/act0/", "/actD/"))
                # ReLU's unused fixed alpha is intentionally NaN. Identically
                # diverged live states must never qualify through equal_nan.
                valid = bool(torch.isnan(a).all() and torch.isnan(b).all()) if fixed_unused_alpha else bool(finite.all())
                delta = (a[finite].double()-b[finite].double()).abs()
                maximum = float(delta.max()) if delta.numel() else 0.
                relative = float((delta / b[finite].double().abs().clamp_min(1.)).max()) if delta.numel() else 0.
                close = bool(torch.all(delta <= GRAD_ATOL + GRAD_RTOL*b[finite].double().abs())) if gradients else relative < STATE_RELATIVE_LIMIT
                passed = valid and (bit_equal if exact else close)
                rows.append(dict(path=path, passed=passed, bit_equal=bit_equal,
                                 finite_or_expected_unused_alpha=valid,
                                 max_abs=maximum, max_abs_over_max1_reference=relative))
            else:
                valid = not bool(a.any() or b.any()) if path.endswith("/bad") else True
                rows.append(dict(path=path, passed=bit_equal and valid, bit_equal=bit_equal))
        else:
            same = a == b
            if isinstance(a, float) or isinstance(b, float):
                same = same and math.isfinite(a) and math.isfinite(b)
            rows.append(dict(path=path, passed=bool(same), bit_equal=bool(same)))

    visit(actual, reference, "root")
    return dict(passed=bool(rows) and all(row["passed"] for row in rows),
                bit_equal=all(row.get("bit_equal", False) for row in rows),
                max_abs=max((row.get("max_abs", 0.) for row in rows), default=0.),
                max_abs_over_max1_reference=max((row.get("max_abs_over_max1_reference", 0.) for row in rows), default=0.),
                leaves=rows)


def require(result, label):
    if not result["passed"]:
        failed = [q for q in result.get("leaves", []) if not q["passed"]]
        raise AssertionError(f"{label}: {json.dumps(failed[:8], default=str)}")


def rng_state():
    numpy_state = np.random.get_state()
    return dict(torch_cpu=torch.get_rng_state().clone(), torch_cuda=torch.cuda.get_rng_state_all(),
                python=random.getstate(), numpy=(numpy_state[0], torch.from_numpy(numpy_state[1].copy()), *numpy_state[2:]))


def prepare(engine):
    engine.new_labels()
    engine.new_order()
    engine.acc_ep.zero_()
    return engine.snapshot()


def state_validity(engine):
    """All live state must be finite and every seed must remain non-diverged."""
    state = engine.snapshot()
    result = comparison(state, state, exact=True)
    result["all_bad_flags_false"] = not bool(engine.bad.any())
    return result


def cache_digest(engine):
    # Hash the actual tensors: a cached digest string cannot detect later writes.
    return None if engine.field is None else M.tree_hash(engine.field.cache)


@torch.no_grad()
def all_image_anchor(engine):
    """All 1200 images, four activities, layer inputs, pool winners and CE."""
    before = M.tree_hash(engine.snapshot())
    before_cache = cache_digest(engine)
    evidence = []
    maximum = 0.
    for lo in range(0, 1200, M.CN.BATCH):
        ids = torch.arange(lo, lo+M.CN.BATCH, device=engine.device)[None].expand(engine.R, -1)
        xb = engine.X[engine.ar, ids]
        out, _, _, inputs = M.forward(engine.P, xb, engine.act, engine.field,
                                      engine.refs_ids(ids), with_inputs=True)
        natural, natural_inputs = CF.forward(engine.P, xb, engine.act, with_inputs=True)
        y = engine.Y[engine.ar, ids]
        losses = lambda z: F.cross_entropy(z.reshape(-1, 10), y.reshape(-1), reduction="none").reshape(engine.R, -1)
        actual = dict(out=out, inputs=inputs, loss=losses(out[8]),
                      pool_winners=[F.max_pool2d(out[k], 2, return_indices=True)[1] for k in (1, 3)])
        expected = dict(out=natural, inputs=natural_inputs, loss=losses(natural[8]),
                        pool_winners=[F.max_pool2d(natural[k], 2, return_indices=True)[1] for k in (1, 3)])
        check = comparison(actual, expected, exact=True)
        maximum = max(maximum, check["max_abs"])
        evidence.append(dict(first_image=lo, images=M.CN.BATCH, **check))
    cache_unchanged = before_cache == cache_digest(engine)
    return dict(passed=all(q["passed"] for q in evidence) and before == M.tree_hash(engine.snapshot()) and cache_unchanged,
                images_per_seed=1200, seeds=list(engine.data.seeds), chunk=M.CN.BATCH,
                max_abs=maximum, training_state_unchanged=before == M.tree_hash(engine.snapshot()),
                actual_frozen_cache_unchanged=cache_unchanged, chunks=evidence)


def initial_gradients(engine):
    ids = engine.order[:, :M.CN.BATCH]
    xb, y = engine.X[engine.ar, ids], engine.Y[engine.ar, ids]
    actual = engine.forward_ids(ids)[0][8]
    expected = CF.forward(engine.P, xb, engine.act)[8]
    loss = lambda z: F.cross_entropy(z.reshape(-1, 10), y.reshape(-1), reduction="none").view(engine.R, -1).mean(1).sum()
    return comparison(torch.autograd.grad(loss(actual), engine.P),
                      torch.autograd.grad(loss(expected), engine.P), gradients=True)


@torch.no_grad()
def independent_projection(P, xb, act):
    """Separate sequential natural-B reference."""
    R, B = xb.shape[:2]
    x = xb.transpose(0, 1).reshape(B, R * 3, 32, 32)
    widths, activities = [], []
    for l in range(4):
        w, b = P[2*l:2*l+2]
        if l < 2:
            q = F.conv2d(x, w.reshape(R * 16, -1, 5, 5), padding=2, groups=R)
            layout = q.reshape(B, R, 16, -1).permute(1, 0, 2, 3).double()
            width = layout.var(1, correction=0).mean(-1).sqrt()
        else:
            q = torch.bmm(x, w.transpose(1, 2))
            width = q.double().var(1, correction=0).sqrt()
        cap = .2 * width
        b.copy_(torch.where(b.double().abs() > cap, b.double().clamp(-cap, cap).to(b.dtype), b))
        z = q + (b.reshape(1, R*16, 1, 1) if l < 2 else b[:, None, :])
        a = z.clamp(min=0)
        widths.append(width)
        activities.append(a)
        if l < 2:
            x = F.max_pool2d(a, 2) - act.sub_map(l, act.m)
            if l == 1:
                x = x.reshape(B, R, 1024).transpose(0, 1)
        else:
            x = a - act.m[l][:, None, :]
    return widths, activities




def original_natural_step(engine, j):
    """Invoke the original CF optimizer/EMA and original B code, not new forward."""
    CF.Bundle.step(engine, j)
    if engine.enabled:
        ids = engine.order[:, j*M.CN.BATCH:(j+1)*M.CN.BATCH]
        trace = ORIGINAL_B.project_bias(engine.P, engine.X[engine.ar, ids], engine.act)
        for l, row in enumerate(trace):
            for key in engine.trace[l]:
                engine.trace[l][key][j].copy_(row[key])
            engine.bad.logical_or_(~row["finite"])


def natural_state(engine):
    # These are all original training state, complete Adam/EMA and RNG included.
    # New intervention-only logging buffers do not exist in the original engine.
    return M.cpu(dict(state=engine.full_state(), Y=engine.Y, order=engine.order,
                      acc_ep=engine.acc_ep, trace=engine.trace))


def natural_self_trajectory(data, name):
    actual = M.Engine(data, name, graph=False)
    reference = M.Engine(data, name, graph=False)
    prepare(actual)
    prepare(reference)
    stream = comparison(dict(Y=actual.Y, order=actual.order), dict(Y=reference.Y, order=reference.order), exact=True)
    gradient = initial_gradients(actual)
    evidence = dict(streams=stream, initial_gradient=gradient)
    evidence["self_initial_valid"] = state_validity(actual)
    evidence["natural_initial_valid"] = state_validity(reference)
    for j in range(75):
        actual.step(j)
        original_natural_step(reference, j)
        if j in (0, 74):
            evidence[f"step{j+1}"] = comparison(natural_state(actual), natural_state(reference), exact=True)
            evidence[f"self_step{j+1}_valid"] = state_validity(actual)
            # The original reference leaves added log buffers zero, still finite.
            evidence[f"natural_step{j+1}_valid"] = state_validity(reference)
    evidence["passed"] = all(q["passed"] for q in evidence.values())
    del actual, reference
    gc.collect()
    torch.cuda.empty_cache()
    return evidence


def execution_checks(data, arm, out):
    """Complete graph/eager state, capture rewind, repetition, file replay, diagnostics."""
    eager = M.Engine(data, arm, graph=False)
    initial = prepare(eager)
    graph = M.Engine(data, arm, graph=True)
    graph.restore(initial)
    result = dict(initial_valid=state_validity(eager))
    before = M.tree_hash(graph.snapshot())
    before_cache = cache_digest(graph)
    before_rng = M.tree_hash(rng_state())
    graph.capture()
    torch.cuda.synchronize()
    cache_unchanged = before_cache == cache_digest(graph)
    result["capture_rewind"] = dict(passed=before == M.tree_hash(graph.snapshot()) and before_rng == M.tree_hash(rng_state()) and cache_unchanged,
                                     state_sha256_before=before, state_sha256_after=M.tree_hash(graph.snapshot()),
                                     rng_unchanged=before_rng == M.tree_hash(rng_state()), actual_frozen_cache_unchanged=cache_unchanged)
    eager.epoch_body()
    graph.cg.replay()
    torch.cuda.synchronize()
    expected = eager.snapshot()
    result["eager_endpoint_valid"] = state_validity(eager)
    result["graph_endpoint_valid"] = state_validity(graph)
    result["graph_eager_75_steps"] = comparison(graph.snapshot(), expected)
    eager.restore(initial)
    eager.epoch_body()
    result["repeated_eager_trajectory"] = comparison(eager.snapshot(), expected, exact=True)
    graph.restore(initial)
    graph.cg.replay()
    torch.cuda.synchronize()
    graph_end = graph.snapshot()
    graph.restore(initial)
    graph.cg.replay()
    torch.cuda.synchronize()
    result["repeated_graph_trajectory"] = comparison(graph.snapshot(), graph_end, exact=True)
    path = Path(out) / f"{arm}_replay_state.pt"
    torch.save(initial, path)
    eager.restore(torch.load(path, map_location="cpu", weights_only=False))
    eager.epoch_body()
    result["save_load_replay"] = comparison(eager.snapshot(), expected, exact=True)
    result["save_load_replay"]["checkpoint_sha256"] = M.sha_file(path)
    before = M.tree_hash(eager.snapshot())
    before_cache = cache_digest(eager)
    before_rng = M.tree_hash(rng_state())
    rows, units, arrays = M.diagnose(eager, 75, chunk=16)
    cache_unchanged = before_cache == cache_digest(eager)
    result["diagnostics_immutable"] = dict(passed=before == M.tree_hash(eager.snapshot()) and before_rng == M.tree_hash(rng_state()) and cache_unchanged,
                                            state_sha256_before=before, state_sha256_after=M.tree_hash(eager.snapshot()),
                                            rng_unchanged=before_rng == M.tree_hash(rng_state()),
                                            actual_frozen_cache_unchanged=cache_unchanged,
                                            summary_rows=len(rows), unit_rows=len(units), arrays=len(arrays))
    result["passed"] = all(q["passed"] for q in result.values())
    del eager, graph, initial, expected, graph_end, rows, units, arrays
    gc.collect()
    torch.cuda.empty_cache()
    return result


def original_table_reproduction(data, source_root):
    """Parent diagnostic in its original chunk20, with independently replayed t30 labels."""
    labels, final_generators = [], {}
    for seed in data.seeds:
        generator = M.H.stream("rlcc_labels", seed)
        for _ in range(30):
            y = M.CN.task_labels(generator)
        labels.append(y)
        final_generators[seed] = generator.get_state()
    stream = comparison(final_generators, data.sources["CHf"]["g_lab"], exact=True)
    result = dict(label_stream_t30=stream)
    for arm, name in (("CHf", "CH_continue"),):
        engine = M.Engine(data, name, graph=False)
        engine.Y.copy_(torch.stack(labels).to(engine.device))
        before = M.tree_hash(engine.snapshot())
        evaluated = CF.evaluate(engine.P, engine.X, engine.Y, engine.act)
        diagnostics, _ = ORIGINAL_B.diagnostics(engine.P, engine.X, engine.Y, engine.act, chunk=20)
        rows = [dict(**evaluation, **diagnostic) for evaluation, diagnostic in zip(evaluated, diagnostics)]
        path = Path(source_root) / arm / "per_task.csv"
        with path.open(newline="") as f:
            old = {int(row["seed"]): row for row in csv.DictReader(f) if int(row["task"]) == 30}
        metrics = []
        for seed, row in zip(data.seeds, rows):
            if seed not in old:
                raise AssertionError(f"Missing task30 reference row {arm}/{seed}")
            for key, actual in row.items():
                if key not in old[seed]:
                    raise AssertionError(f"Diagnostic column absent from original table: {key}")
                reference = float(old[seed][key])
                actual = float(actual)
                expected_nan_column = key.startswith(("alpha_med_", "seat_", "two_alpha_", "eff_rank_"))
                both_nan = expected_nan_column and math.isnan(actual) and math.isnan(reference)
                if math.isfinite(actual) and math.isfinite(reference):
                    difference = abs(actual-reference)
                    relative = difference/max(1., abs(reference))
                    passed = difference <= MEMO_ABSOLUTE_LIMIT if key.startswith("memo_acc") else relative < STATE_RELATIVE_LIMIT
                else:
                    difference = relative = None
                    passed = both_nan
                metrics.append(dict(seed=seed, metric=key, passed=passed, actual=actual if math.isfinite(actual) else str(actual),
                                    reference=reference if math.isfinite(reference) else str(reference),
                                    bit_equal=actual.hex() == reference.hex(), max_abs=difference,
                                    abs_over_max1_reference=relative))
        result[arm] = dict(passed=all(q["passed"] for q in metrics) and before == M.tree_hash(engine.snapshot()),
                           chunk=20, table_path=str(path), table_sha256=M.sha_file(path),
                           state_unchanged=before == M.tree_hash(engine.snapshot()),
                           bit_equal=all(q["bit_equal"] for q in metrics), metrics=metrics)
        del engine
    result["passed"] = all(q["passed"] for q in result.values())
    return result


def direct_policy_checks(data):
    a,b=[M.Engine(data,n,False) for n in M.MAIN_ARMS]
    prepare(a); prepare(b)
    sa,sb=a.full_state(),b.full_state()
    sa.pop('enabled');sb.pop('enabled')
    initial=comparison(sa,sb,exact=True)
    a.step(0);b.step(0)
    first=comparison(dict(m=a.m,v=a.v,act=a.act.state(),W=a.P[::2],head=a.P[-2:],acc=a.acc_steps,ce=a.ce_steps),
                     dict(m=b.m,v=b.v,act=b.act.state(),W=b.P[::2],head=b.P[-2:],acc=b.acc_steps,ce=b.ce_steps),exact=True)
    # Independent projection on the actual Adam/EMA-updated, unprojected CH state.
    ref=[p.detach().clone() for p in a.P]
    widths,_=independent_projection(ref,a.X[a.ar,a.order[:,:16]],a.act)
    independent=comparison(ref,b.P,gradients=True)
    width_check=comparison(widths,[r['s'][0] for r in b.trace])
    del a,b
    return dict(passed=all(r['passed'] for r in (initial,first,independent,width_check)),
                identical_complete_initial_state=initial,first_Adam_EMA_stream_loss_W_head_equal=first,
                independent_first_projection=independent,independent_widths=width_check)


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    release = M.guard_release(args.release)
    # Guard also binds these two independent verification files, not only src/.
    import subprocess
    changed = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--", str(Path(__file__).relative_to(ROOT)), "tests/test_cnn_chb_delayed_1021.py"], text=True)
    if changed.strip():
        raise RuntimeError("Preflight/tests are uncommitted; register them before GPU qualification")
    report = dict(all_pass=False, started_at=time.time(), release=release,
                  thresholds=dict(exact_initial_values_pool_logits_loss=True, gradient_atol=GRAD_ATOL,
                                  gradient_rtol=GRAD_RTOL, state_abs_over_max1_reference_strictly_less=STATE_RELATIVE_LIMIT,
                                  memo_abs_at_most=MEMO_ABSOLUTE_LIMIT, exact_rewind_repetition_replay_diagnostics=True),
                  verifier_hashes={str(Path(__file__).relative_to(ROOT)): M.sha_file(__file__),
                                   "tests/test_cnn_chb_delayed_1021.py": M.sha_file(ROOT/"tests/test_cnn_chb_delayed_1021.py")}, checks={})

    def record(name, fn):
        started = time.time()
        try:
            result = fn()
        except Exception as error:
            result = dict(passed=False, error=repr(error), traceback=traceback.format_exc())
        result["seconds"] = time.time()-started
        report["checks"][name] = result
        M.save_json(out/"preflight.json", report)
        print(f"{'PASS' if result['passed'] else 'FAIL'} {name} ({result['seconds']:.1f}s)", flush=True)
        if not result["passed"]:
            raise RuntimeError(f"Preflight failed: {name}; inspect {out/'preflight.json'}")

    with M.locked(args.gpu_lock), M.locked(out/".output.lock"):
        # No CUDA call occurs before the GPU lock and registered release checks.
        device = M.configure_runtime("cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("Registered GPU preflight has no CPU fallback")
        if torch.__version__ != "2.13.0+cu130":
            raise RuntimeError(f"Registered torch2.13.0+cu130 required, got {torch.__version__}")
        report["numerical_context"] = M.numerical_context()
        report["cuda_graph_override_stale_capture_stream"] = True
        data = M.Data(device=device, source_root=Path(args.source_root), manifest_path=Path(args.manifest), data_dir=args.data_dir)
        report["source_states_and_streams"] = dict(passed=True, identity=data.identity)
        report["contract"] = M.preflight_contract(data, True, release)
        record("original_chunk20_table_reproduction", lambda: original_table_reproduction(data, args.source_root))
        record("direct_policy_and_independent_B", lambda: direct_policy_checks(data))
        for arm in M.MAIN_ARMS:
            def check_anchor(arm=arm):
                engine = M.Engine(data, arm, graph=False)
                prepare(engine)
                result = all_image_anchor(engine)
                del engine
                gc.collect()
                torch.cuda.empty_cache()
                return result
            record(f"{arm}_all1200_initial", check_anchor)
        for arm in M.MAIN_ARMS:
            record(f"{arm}_natural_gradient_1_75", lambda arm=arm: natural_self_trajectory(data, arm))
        for arm in M.MAIN_ARMS:
            record(f"{arm}_execution", lambda arm=arm: execution_checks(data, arm, out))
        report["all_pass"] = True
        report["finished_at"] = time.time()
        M.save_json(out/"preflight.json", report)
    print(f"PASS all registered GPU checks: {out/'preflight.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-root", default=str(M.DEFAULT_SOURCE))
    parser.add_argument("--manifest", default=str(M.DEFAULT_MANIFEST))
    parser.add_argument("--data-dir")
    parser.add_argument("--gpu-lock", default="/tmp/lop_analysis_gpu.lock")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

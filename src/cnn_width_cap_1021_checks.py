#!/usr/bin/env python3
"""Synthetic-only checks. Never load CIFAR or experimental checkpoints."""
from __future__ import annotations
import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import cnn_width_cap_1021 as C


def same(xs, ys):
    return len(xs) == len(ys) and all(torch.equal(x, y) for x, y in zip(xs, ys))


def fake_data(dtype=torch.float32, device="cpu"):
    g = torch.Generator().manual_seed(1021)
    return SimpleNamespace(dtype=dtype, X=torch.randn(64, 3, 32, 32, generator=g, dtype=dtype).to(device),
                           Y=torch.randint(100, (64,), generator=g).to(device))


def eng(data, cap=False, graph=False):
    return C.CapEngine([(C.ARMS[0], 9000), (C.ARMS[1], 9001)], data,
                       torch.device(data.X.device), budget=8, enabled=cap, graph=graph, dtype=data.dtype)


def batches():
    return torch.arange(2 * 8 * 32).reshape(2, 8, 32) % 64


def check_variance():
    g = torch.Generator().manual_seed(99)
    q = torch.randn(5, 32, 3, 4, dtype=torch.float64, generator=g)
    got = C.variance(q, 0, 2).numpy()
    expected = q.numpy().reshape(5, 2, 16, 3, 4).var(axis=(0, 3, 4))
    np.testing.assert_allclose(got, expected, atol=1e-14, rtol=1e-14)
    wrong = q.numpy().reshape(5, 2, 16, 3, 4).mean(axis=(3, 4)).var(axis=0)
    assert not np.allclose(got, wrong), "image-mean variance mutation not detected"
    qf = torch.randn(2, 5, 100, dtype=torch.float64, generator=g)
    np.testing.assert_allclose(C.variance(qf, 2, 2), qf.numpy().var(axis=1), atol=1e-14)
    ref = torch.tensor([0., 0., 2., 2.], dtype=torch.float64)
    vt = torch.tensor([0., 4., 0., 16.], dtype=torch.float64)
    assert torch.equal(C.cap_ratio(ref, vt), torch.tensor([1., 0., 1., .5], dtype=torch.float64))


def check_projection():
    data = fake_data(torch.float64)
    e = eng(data, True)
    for p in e.P:
        with torch.no_grad():
            p.mul_(3)
    e.set_reference([torch.full_like(x, .05) for x in e.s_ref])
    xb = data.X[batches()[:, 0]]
    before = [p.detach().clone() for p in e.P]
    m, v = copy.deepcopy(e.m), copy.deepcopy(e.v)
    refP = [p.detach().clone() for p in e.P]
    oldV = [x.clone() for x in e.cap_V]
    # Independent reference: repeat the whole forward after each site's row projection.
    ratios = []
    with torch.no_grad():
        for site in range(4):
            z = C.E.forward(refP, xb, e.act)[2 * site]
            a = z.numpy()
            var = (a.reshape(32, 2, 16, *a.shape[-2:]).var(axis=(0, 3, 4))
                   if site < 2 else a.var(axis=1))
            vt = oldV[site].numpy() * .99 + .01 * var
            ratio = np.minimum(1., .05 / np.sqrt(vt))
            ratios.append(torch.from_numpy(ratio))
            w = refP[2 * site]
            w.mul_(torch.from_numpy(ratio).reshape(2, C.B.WIDTHS[site], *([1] * (w.ndim - 2))))
            refP[2 * site + 1].mul_(torch.from_numpy(ratio))
        expected = C.E.forward(refP, xb, e.act)[7]
        actual = e.project(xb)
    for a, b in zip(e.P, refP):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-11)
    assert same([before[i] for i in (8, 9)], [e.P[i] for i in (8, 9)])
    assert same(m, e.m) and same(v, e.v)
    assert all(bool((x <= .05**2 * (1 + 1e-12)).all()) for x in e.cap_V)
    for site in range(4):
        torch.testing.assert_close(e.cap_stats["ratio_min"][site], ratios[site], atol=1e-12, rtol=1e-11)
    # W and b scale together: local m/s scale together and r stays fixed.
    zz = torch.tensor([[1., 4.], [-3., 2.]], dtype=torch.float64)
    bb, rr = torch.tensor([2., -1.]), torch.tensor([.2, .8])
    post = zz * rr
    torch.testing.assert_close(post.mean(0), rr * zz.mean(0))
    torch.testing.assert_close(post.std(0, correction=0), rr * zz.std(0, correction=0))
    torch.testing.assert_close(post.mean(0) / post.std(0, correction=0), zz.mean(0) / zz.std(0, correction=0))
    assert not torch.equal(post, (zz - bb) * rr + bb), "missing bias projection mutation not detected"
    # Stale upstream inputs produce different downstream row scales and must be detected.
    stale = C.E.forward(before, xb, e.act)
    stale_ratios = []
    for site in range(4):
        var = C.variance(stale[2 * site], site, 2)
        stale_ratios.append(C.cap_ratio(e.s_ref[site], oldV[site] * .99 + .01 * var))
    assert any(not torch.allclose(stale_ratios[l], ratios[l]) for l in (1, 2, 3))
    assert any(not torch.allclose(oldV[l] * .99 + .01 * C.variance(stale[2*l], l, 2), e.cap_V[l]) for l in range(4)), "missing EMA correction mutation not detected"


def check_natural_and_restart(device="cpu"):
    data = fake_data(device=device)
    e = eng(data)
    base = C.B.BudgetEngine(e.slots, data, torch.device(device), budget=8, graph=False)
    base.load_full_state(e.full_state())
    e.set_reference([torch.full_like(x, .1) for x in e.s_ref])
    e.begin_task(batches())
    base.idx.copy_(batches().to(device))
    for _ in range(3):
        e.step()
        base.step()
    assert same(e.P, base.P) and same(e.m, base.m) and same(e.v, base.v)
    assert same(list(e.rec.values()), list(base.rec.values()))
    assert all(same(e.act.state()[k], base.act.state()[k]) for k in e.act.state())
    expected = C.E.forward(e.P, data.X[batches()[:, 0].to(device)], e.act)[7]
    old = [p.clone() for p in e.P]
    actual = e.project(data.X[batches()[:, 0].to(device)])
    assert torch.equal(actual, expected) and same(old, e.P)

    a = eng(data, True)
    a.set_reference([torch.full_like(x, .1) for x in a.s_ref])
    a.begin_task(batches())
    a.advance(2)
    state = a.full_state()
    b = eng(data, True)
    b.load_full_state(state)
    a.advance(3)
    b.advance(3)
    assert same(a._state_tensors(), b._state_tensors())
    assert a.n_done == b.n_done == 5
    assert all(same(a.act.state()[k], b.act.state()[k]) for k in a.act.state())
    source_state = base.full_state()
    selected, idx = C.subset_state(source_state, list(reversed(a.slots)))
    assert idx == [1, 0]
    assert torch.equal(selected["P"][0], source_state["P"][0][[1, 0]])
    # Moment mutations cannot pass the unchanged-state invariant.
    mutated = [x.clone() for x in a.m]
    mutated[0].mul_(.5)
    assert not same(a.m, mutated)


def check_diagnostics():
    data = fake_data()
    e = eng(data, True)
    e.set_reference([torch.ones_like(x) for x in e.s_ref])
    C.no_touch_stats(e, data.X, torch.arange(8)[None].expand(2, -1), data.Y, chunk=4)


def check_calibration_ema():
    data = fake_data(torch.float64)
    slots = [(C.ARMS[0], 9000), (C.ARMS[1], 9001)]
    e = C.CapEngine(slots, data, torch.device("cpu"), budget=8, enabled=False,
                    graph=False, dtype=torch.float64, ema_start_first_batch=True)
    xb = data.X[batches()[:, 0]]
    expected = [C.variance(z, site, 2) for site, z in enumerate(C.E.forward(e.P, xb, e.act)[0:8:2])]
    e.project(xb)
    for v, ex in zip(e.cap_V, expected):
        torch.testing.assert_close(v, ex, atol=1e-12, rtol=1e-11)
    xb2 = data.X[(batches()[:, 0] + 1) % 64]
    expected2 = [C.variance(z, site, 2) for site, z in enumerate(C.E.forward(e.P, xb2, e.act)[0:8:2])]
    e.project(xb2)
    for v, ex, ex2 in zip(e.cap_V, expected, expected2):
        torch.testing.assert_close(v, ex * .99 + ex2 * .01, atol=1e-12, rtol=1e-11)
    assert C.compare_tree(e.full_state(), copy.deepcopy(e.full_state())) == []
    mutated = e.full_state()
    mutated["P"][0][0, 0, 0, 0, 0] += 1
    assert ".P[0]" in C.compare_tree(e.full_state(), mutated)


def check_graph():
    data = fake_data(device="cuda")
    a = eng(data, True, graph=True)
    a.set_reference([torch.full_like(x, .1) for x in a.s_ref])
    a.begin_task(batches())
    before = [x.clone() for x in a._state_tensors()]
    act = a.act.state()
    a.capture()
    assert same(before, a._state_tensors())
    assert all(same(act[k], a.act.state()[k]) for k in act)
    b = eng(data, True, graph=True)
    b.load_full_state(a.full_state())
    a.advance(2)
    b.advance(2)
    assert same(a._state_tensors(), b._state_tensors())


def check_sign():
    assert C.exact_sign([1.] * 10)["p"] == 2 / 1024
    assert C.exact_sign([0.] * 10)["p"] == 1
    assert C.exact_sign([1., -1.] * 5)["p"] == 1
    assert C.exact_sign([1.] * 8 + [0.] * 2)["p"] == 2 / 256


def check_summary_and_rng():
    import csv
    import tempfile
    with tempfile.TemporaryDirectory(prefix="cnn_cap_synthetic_") as tmp:
        p = Path(tmp)
        for condition in ("natural", "cap"):
            folder = p / condition
            folder.mkdir()
            rows = []
            for cfg in C.ARMS:
                for seed in C.SEEDS:
                    for task in range(2, 31):
                        value = .5 + (.1 if condition == "cap" and cfg == C.ARMS[0] and task >= 21 else 0.)
                        rows.append({"cfg": cfg, "seed": seed, "task": task, "bad_step": -1,
                                     **{k: value for k in ("first100_online_acc", "first780_online_acc", "online_acc",
                                                         "train_acc", "test_acc", "ce_mean", "ce_first780_mean")}})
            with open(folder / "per_task.csv", "w") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (folder / "provenance.json").write_text(json.dumps({"status": "COMPLETED",
                "meta": {"condition": condition, "budget": C.BUDGET, "slots": C.main_slots()}}))
        C.summarize(SimpleNamespace(natural=p / "natural", cap=p / "cap", out=p / "summary"))
        result = json.loads((p / "summary" / "primary.json").read_text())["results"]
        assert result[0]["verdict"] == "IMPROVES" and result[0]["p_holm"] == 4 / 1024
        assert result[1]["verdict"] == "INCONCLUSIVE" and result[1]["p_holm"] == 1
        assert abs(result[0]["pairs"][0]["late_minus_early"]["cap"] - .1) < 1e-14
        prov = json.loads((p / "cap" / "provenance.json").read_text())
        prov["status"] = "FAILED_NONFINITE"
        (p / "cap" / "provenance.json").write_text(json.dumps(prov))
        try:
            C.summarize(SimpleNamespace(natural=p / "natural", cap=p / "cap", out=p / "failed_summary"))
            raise AssertionError("nonfinite/unfinished completion mutation not detected")
        except SystemExit:
            pass
        assert json.loads((p / "failed_summary" / "primary.json").read_text())["status"] == "NOT_ESTABLISHED"
    state = C.rng_state()
    a = (C.random.random(), np.random.rand(), torch.rand(3))
    C.load_rng(state)
    b = (C.random.random(), np.random.rand(), torch.rand(3))
    assert a[0] == b[0] and a[1] == b[1] and torch.equal(a[2], b[2])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cuda", action="store_true", help="synthetic graph/capture test; no CIFAR")
    ap.add_argument("--out")
    args = ap.parse_args()
    torch.set_num_threads(2)
    done = []
    for name, fn in (("pooled_variance_zero_width", check_variance),
                     ("sequential_projection_invariants_mutations", check_projection),
                     ("natural_exact_and_full_restart", check_natural_and_restart),
                     ("reference_ema_initialization_and_state_compare", check_calibration_ema),
                     ("diagnostics_no_touch", check_diagnostics), ("exact_sign", check_sign),
                     ("summary_holm_and_rng_restore", check_summary_and_rng)):
        fn()
        done.append(name)
        print(f"PASS {name}", flush=True)
    if args.cuda:
        with C.resource_locks("cuda", args.out or "/tmp/cnn_width_cap_1021_checks.json"):
            C.NK.setup("cuda")
            check_natural_and_restart("cuda")
            check_graph()
        done += ["cuda_natural_and_restart", "cuda_capture_rewind"]
    result = {"status": "PASS", "synthetic_only": True, "checks": done, "torch": torch.__version__}
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

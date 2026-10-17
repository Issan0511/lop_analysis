"""CPU synthetic authenticity/mutation checks; production GPU checks are separate."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from src import chb_decompose_200_1018 as D
from analysis.chb_decompose_200_1018 import common as C
from analysis.chb_decompose_200_1018 import report as R
from analysis.chb_decompose_200_1018.diagnostics import Observer, digest, units


@contextmanager
def synthetic_box(device="cpu", full=False):
    """Synthetic input only, with production shapes/R for GPU kernel checks."""
    dims = (3072, 100, 100, 10) if full else (8, 4, 3, 10)
    n, batch = (1200, 16) if full else (16, 4)
    seeds = list(range(100, 110)) if full else [100, 101]
    rc = SimpleNamespace(**D.E.RC.__dict__)
    rc.task_labels = lambda g: torch.randint(10, (n,), generator=g)
    rc.subset_idx = lambda s: torch.arange(n) + s * n
    rc.N_IMAGES, rc.DIMS = n, dims
    cifar = SimpleNamespace(sha256={"synthetic": "implementation-test-only"})
    def slot_inputs(_cifar, seed, cond, dev, center=False):
        g = torch.Generator().manual_seed(100000 + seed)
        x = torch.rand(n, dims[0], generator=g).to(dev)
        return x - x.mean(0) if center else x
    def factory(arm, lam=0., beta=.01):
        return (D.Decomposition(arm, lam, beta, widths=dims[1:3]) if arm in D.NEW_ARMS else
                D.E.ReLUDoors(arm, lam, beta, widths=dims[1:3]))
    changes = dict(DIMS=dims, N_IMAGES=n, BATCH=batch, STEPS_PER_EPOCH=n//batch,
                   RC=rc, make_act=factory, slot_inputs=slot_inputs)
    with patch.dict(D.E.__dict__, changes), patch.dict(D.run.__globals__, changes):
        yield dict(cifar=cifar, dims=dims, n=n, seeds=seeds, device=torch.device(device),
                   factory=factory, inputs=slot_inputs)


def execute(box, out, arm, tasks=2, resume=False, original=False, graph=False, observer=False):
    cp = torch.load(out / "ckpt.pt", weights_only=False) if resume else None
    obs = Observer(out, box["seeds"], cp) if observer else None
    (D.E.run if original else D.run)(arm, box["seeds"], ["raw"], tasks, 1, box["device"], out,
        lam=0. if arm == "CH" else C.LAM, cifar=box["cifar"], checkpoint=True,
        resume=resume, graph=graph, snapshots=True, lifecycle=obs, progress=lambda _: None)
    return torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)


def state(st):
    return {k: st[k] for k in ("t", "P", "m", "v", "tc", "act_state", "g_lab", "g_batch", "alive", "rows")}


def freeze_check(st, initial):
    for i in (1, 3):
        assert torch.equal(st["P"][i], initial[i]), "bias value changed"
        assert not torch.count_nonzero(st["m"][i]), "frozen bias moment advanced"
        assert not torch.count_nonzero(st["v"][i]), "frozen bias variance advanced"
    for i in (0, 2, 4, 5):
        assert not torch.equal(st["P"][i], initial[i]), "free weight/output bias did not update"


def expect_rejection(function):
    try:
        function()
    except (ValueError, AssertionError, KeyError):
        return True
    raise AssertionError("mutant was not detected")


def engine_checks(device="cpu", full=False):
    records = []
    with tempfile.TemporaryDirectory(prefix="chb-check-") as tmp, synthetic_box(device, full) as box:
        root = Path(tmp)
        for arm in ("CH", "CHB"):
            original = execute(box, root / (arm+"_host"), arm, original=True)
            adapter = execute(box, root / arm, arm)
            assert digest(state(original)) == digest(state(adapter)), "adapter changed native host trajectory"
            assert (root / arm / "per_task.csv").read_bytes() == (root / (arm+"_host") / "per_task.csv").read_bytes()
            records.append(dict(check="S-nochange_"+arm, pass_=True, tolerance="bit/byte equality"))
        initial = [torch.stack([D.E.H.init_params(s, box["device"], box["dims"])[i].detach().cpu()
                                for s in box["seeds"]]) for i in range(6)]
        # Require a genuinely nonzero bias gradient in the unconstrained fixture.
        trial = [q.to(box["device"]).clone().requires_grad_(True) for q in initial]
        act = box["factory"]("CH"); act.init_state(len(box["seeds"]), box["device"], "fixture")
        x = torch.stack([box["inputs"](box["cifar"], s, "raw", box["device"], True) for s in box["seeds"]])
        logits = D.E.forward(trial, x, act)[-1]
        grads = torch.autograd.grad(torch.nn.functional.cross_entropy(logits.flatten(0,1), torch.zeros(logits.shape[0]*logits.shape[1], dtype=torch.long, device=box["device"])), trial)
        assert all(torch.count_nonzero(initial[i]) and torch.count_nonzero(grads[i]) for i in (1,3))
        frozen = execute(box, root / "freeze", "CH_bfreeze", observer=True)
        freeze_check(frozen, initial)
        records.append(dict(check="S-freeze", pass_=True, tolerance="bit equality; nonzero initial bias and unconstrained gradients"))
        # Reverting only values leaves nonzero Adam state, and is rejected.
        reverted = copy.deepcopy(execute(box, root / "free", "CH"))
        for i in (1,3): reverted["P"][i].copy_(initial[i])
        expect_rejection(lambda: freeze_check(reverted, initial))
        b3frozen = copy.deepcopy(frozen); b3frozen["P"][5].copy_(initial[5])
        expect_rejection(lambda: freeze_check(b3frozen, initial))
        records.append(dict(check="S-freeze_mutations", pass_=True, detected=["restore-only bias with moments", "output bias frozen"]))
        for arm in D.NEW_ARMS:
            continuous = execute(box, root / (arm+"_full"), arm, observer=True)
            execute(box, root / (arm+"_split"), arm, tasks=1, observer=True)
            resumed = execute(box, root / (arm+"_split"), arm, resume=True, observer=True)
            assert digest(state(continuous)) == digest(state(resumed)), "resume changed training"
            assert (root / (arm+"_full") / "diagnostics.csv").read_bytes() == (root / (arm+"_split") / "diagnostics.csv").read_bytes()
            for t in (1,2):
                with np.load(root / (arm+"_full") / "diagnostic_units" / f"t{t:03d}.npz") as a, np.load(root / (arm+"_split") / "diagnostic_units" / f"t{t:03d}.npz") as b:
                    assert a.files == b.files
                    for key in a.files: np.testing.assert_equal(a[key], b[key])
            # Missing EMA and wrong label-generator continuation are observable
            # against the saved state, not merely compared to nominal settings.
            for key in ("act_state", "g_lab"):
                mutant = copy.deepcopy(continuous)
                if key == "act_state": mutant[key]["m"][0].zero_()
                else: mutant[key][box["seeds"][0]] = D.E.H.stream("rlc_labels", box["seeds"][0]).get_state()
                assert digest(state(mutant)) != digest(state(continuous))
            records.append(dict(check="S-resume_"+arm, pass_=True, tolerance="bit equality of P/m/v/EMA/RNG/clock/rows/diagnostics", detected=["EMA omission", "label RNG restarted"]))
        # Actual mutant engine policy changes must break host nochange.
        factory = box["factory"]
        def tiny_wd(arm, lam=0., beta=.01):
            a = factory(arm, lam, beta)
            native = a.post_update
            def post(p, lr):
                native(p, lr); p[0].mul_(1 - lr * .001)
            a.post_update = post
            return a
        with patch.dict(D.run.__globals__, make_act=tiny_wd):
            bad = execute(box, root / "tiny_wd", "CH")
        assert digest(state(bad)) != digest(state(execute(box, root / "unchanged", "CH")))
        host_init = D.E.H.init_params
        def zero_b1(seed, device, dims):
            p = host_init(seed, device, dims)
            with torch.no_grad(): p[1].zero_()
            return p
        with patch.object(D.E.H, "init_params", zero_b1):
            zeroed = execute(box, root / "initial_zero", "CHB")
        native = execute(box, root / "native_chb", "CHB")
        assert digest(state(zeroed)) != digest(state(native))
        records.append(dict(check="S-nochange_mutations", pass_=True, detected=["tiny WD on CH", "CHB b1 removed at initialization"]))
        # Compare true autograd/Adam candidates before the first differing policy.
        with patch.dict(D.E.__dict__, STEPS_PER_EPOCH=1), patch.dict(D.run.__globals__, STEPS_PER_EPOCH=1):
            baseline = execute(box, root / "candidate_CH", "CH", tasks=1)
            for arm in ("CH_wWD", "CH_bWD"):
                changed = execute(box, root / ("candidate_"+arm), arm, tasks=1)
                assert digest(changed["m"]) == digest(baseline["m"])
                assert digest(changed["v"]) == digest(baseline["v"])
                for i, (actual, candidate) in enumerate(zip(changed["P"], baseline["P"])):
                    expected = candidate*(1-C.LR*C.LAM) if i in D.wd_indices(arm) else candidate
                    assert torch.equal(actual, expected), "policy did not follow actual host Adam candidate"
        records.append(dict(check="S-real_Adam_candidates", pass_=True,
                            tolerance="bit equality of all tensors and moments, first update",
                            detected=["coupled decay would change m/v", "wrong tensor selection", "wrong update order"]))
        if device != "cpu":
            for arm in C.ARMS:
                eager = execute(box, root / (arm+"_eager"), arm, observer=True)
                graph = execute(box, root / (arm+"_graph"), arm, observer=True, graph=True)
                assert digest(state(eager)) == digest(state(graph))
                assert (root / (arm+"_eager") / "diagnostics.csv").read_bytes() == (root / (arm+"_graph") / "diagnostics.csv").read_bytes()
                records.append(dict(check="S-graph_"+arm, pass_=True, tolerance="bit/byte equality including warmup rollback and update counts"))
    return records


def policy_checks():
    g = torch.Generator().manual_seed(104)
    p = [torch.randn((2,3,4) if i%2 == 0 else (2,3), generator=g) for i in range(6)]
    records = []
    for arm, indices in (("CH_wWD", (0,2)), ("CH_bWD", (1,3))):
        act = D.Decomposition(arm)
        before = [v.clone() for v in p]; actual = [v.clone() for v in p]
        act.post_update(actual, .001)
        expected = [v*(1-.001*.1713) if i in indices else v for i,v in enumerate(before)]
        assert all(torch.equal(a,b) for a,b in zip(actual,expected))
        for name in (("W3_decay", "b2_decay", "coupled_L2") if arm == "CH_wWD" else
                     ("b1_zero", "b2_only", "moment_decay")):
            mutant = [v.clone() for v in actual]
            moments = [torch.ones_like(v) for v in p]; saved = digest(moments)
            if name == "W3_decay": mutant[4].mul_(1-.001*.1713)
            if name == "b2_decay": mutant[3].mul_(1-.001*.1713)
            if name == "coupled_L2": mutant[0] = before[0] - .001*(torch.sign(before[0]+.1713*before[0]))
            if name == "b1_zero": mutant[1].zero_()
            if name == "b2_only": mutant[1] = before[1].clone()
            if name == "moment_decay": moments[1].mul_(1-.001*.1713)
            assert digest(mutant) != digest(expected) or digest(moments) != saved
        records.append(dict(check="S-"+("wWD" if arm=="CH_wWD" else "bWD"), pass_=True,
                            tolerance="exact float32 multiply, unaffected tensors bit equal",
                            detected=["coupled_L2", "W3_decay", "b2_decay"] if arm=="CH_wWD" else ["b1_zero", "b2_only", "moment_decay"]))
    act = D.E.ReLUDoors("CH", widths=(4,3)); act.init_state(2, "cpu", "fixture")
    act.m[0].fill_(.7); act.m[1].fill_(.4)
    old = [q.clone() for q in act.m]
    zs = [torch.randn(2, 8, w, generator=g) for w in (4,3)]
    expected = [v*.99 + z.clamp(min=0).mean(1)*.01 for v,z in zip(old,zs)]
    with torch.no_grad(): act.update(*zs)
    for a,b in zip(act.m,expected): torch.testing.assert_close(a,b,rtol=0,atol=2e-7)
    reset = torch.zeros_like(old[0])*.99 + zs[0].clamp(min=0).mean(1)*.01
    assert not torch.allclose(reset,act.m[0])
    z = zs[0].clone().requires_grad_(True); act.phi(z).sum().backward()
    assert not act.m[0].requires_grad and act.m[0].grad is None
    bad_m = act.m[0].clone().requires_grad_(True)
    assert torch.autograd.grad((z.clamp(min=0)-bad_m[:,None]).sum(),bad_m)[0].abs().max() > 0
    records.append(dict(check="S-center", pass_=True, detected=["task EMA reset", "gradient through EMA"],
                        tolerance="4 elementary float32 ops, 2e-7 absolute at fixture scale <1"))
    # Verify native clamp derivative at zero independently of legacy strict gate.
    with synthetic_box() as box:
        a = box["factory"]("CH"); a.init_state(2, "cpu", "zero")
        pp = [torch.zeros(2,b,a) if i%2 == 0 else torch.zeros(2,b)
              for i,(a,b) in enumerate(((8,4),(0,4),(4,3),(0,3),(3,10),(0,10)))]
        result = units(pp, torch.zeros(2,16,8), a)
        assert np.all(result["l1_derivative_mean"] == 1)
        assert np.all(result["l1_legacy_gate_mean"] == 0)
        assert np.all(result["l1_derivative_var"] == 0)
    records.append(dict(check="S-native_derivative_zero_convention", pass_=True,
                        detected=["legacy strict gate substituted for native clamp autograd"], tolerance="exact zero/one"))
    return records


def statistical_checks():
    assert R.interval([.1]*10)["ci"] == [.1,.1]
    assert R.interval([.1]*10)["flags"] == ["DEGENERATE_SD"]
    assert abs(R.student9_quantile(.975)-2.262157163) < 1e-9  # df9 9-digit table
    nodes, weights = np.polynomial.legendre.leggauss(64)
    for p in (.975,1-.05/14):
        q = R.student9_quantile(p)
        x = (nodes+1)*q/2
        # Independent quadrature of the t density, rather than the trig CDF.
        normalizer = __import__("math").gamma(5)/(3*np.sqrt(np.pi)*__import__("math").gamma(4.5))
        integral = q/2*np.sum(weights*normalizer*(1+x*x/9)**-5)
        assert abs(integral-(p-.5)) < 2e-14
    base = np.full(10,.4); high = np.full(10,.9)
    values = dict(CH=base, CHB=high, CH_bfreeze=high, CH_wWD=base, CH_bWD=high)
    assert R.classify(values)["label"] == "BIAS_STOP_SUFFICES"
    assert R.classify(dict(values, CH_wWD=high))["label"] == "WD_EXPLAINS"
    assert R.classify(dict(values, CH_bfreeze=base))["label"] == "MIXED"
    assert R.classify(dict(values, CHB=base))["label"] == "NOT_REPRODUCED"
    assert not R.ni(dict(ci=[-.005,.01]))
    assert R.eq(dict(ci=[-.005,.005]))
    assert not R.pos(dict(ci=[0,.01]))
    assert not R.eq(dict(ci=[-.1,.1]))
    assert R.classify({a: np.full(10,x) for a,x in dict(CH=.4,CHB=.401,CH_bfreeze=.401,CH_bWD=.401,CH_wWD=.401).items()})["reason"] == "OVERLAPPING_EVIDENCE"
    fixture = {}
    for arm in C.ARMS:
        fixture[arm] = [dict(arm=arm,cond="raw",iv="none",seed=s,task=t,slot=i,lr=.001,
                            online_acc=float(values[arm][i]),memo_acc=.99)
                        for i,s in enumerate(C.SEEDS) for t in range(1,201)]
    result = R.aggregate(fixture)
    shuffled = copy.deepcopy(fixture)
    for a in C.ARMS: np.random.default_rng(100+len(a)).shuffle(shuffled[a])
    assert R.aggregate(shuffled) == result
    for mutation in ("missing", "duplicate", "old_seed", "nonfinite", "window"):
        f = copy.deepcopy(fixture)
        if mutation == "missing": f["CH"].pop()
        if mutation == "duplicate": f["CH"][-1] = f["CH"][0].copy()
        if mutation == "old_seed":
            for row in f["CH"]: row["seed"] -= 10
        if mutation == "nonfinite": f["CH"][0]["online_acc"] = float("nan")
        if mutation == "window":
            for arm in C.ARMS:
                for row in f[arm]:
                    if 181 <= row["task"] <= 200: row["online_acc"] = .2
            assert R.aggregate(f)["label"] == "NOT_REPRODUCED"
        else: expect_rejection(lambda: R.aggregate(f))
    return [dict(check="S-stats_and_reference_keys", pass_=True,
                 detected=["zero-containing interval treated as equality", "NI boundary", "wrong seed pairing", "missing", "duplicate", "seed0-9 reference", "wrong endpoint window", "nonfinite"], tolerance="float64; exact registered strict/inclusive boundary predicates")]


def gpu_checks(device):
    records = engine_checks(str(device), full=True)
    return dict(all_pass=all(r["pass_"] for r in records), records=records,
                source_sha256=C.sources(), scope="CUDA production shapes/R10, synthetic inputs, calibration seeds100–109", at=C.utc())


def run_checks():
    torch.set_num_threads(1)
    records = []
    failed = []
    for name, function in (("engine",engine_checks), ("policy",policy_checks), ("statistics",statistical_checks)):
        try:
            records.extend(function())
        except Exception as exc:
            import traceback
            traceback.print_exc()
            failed.append(dict(group=name,type=type(exc).__name__,error=str(exc)))
    value = dict(cpu_all_pass=not failed and bool(records), all_pass=False,
                 scope="CPU synthetic implementation checks only; no evaluation-seed production result",
                 pending=["CUDA production-shape graph/nochange checks", "reference reuse qualification or CH/CHB rerun", "registered production 200 tasks"],
                 records=records, failures=failed, source_sha256=C.sources(), at=C.utc())
    C.put(C.OUT / "checks.json", value)
    print(json.dumps(value, indent=2))
    return 0 if value["cpu_all_pass"] else 1

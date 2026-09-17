#!/usr/bin/env python3
"""Checks for avgpool_cnn_0917 (spec §4).  All must pass before the main run.

    python3 src/avgpool_cnn_0917_checks.py --device cuda

S-pool     avg-pool = the 2x2 mean, and its backward sends exactly 1/4 of the upstream
           gradient to every position.  Mutation: max-pool leaves 3 of 4 positions at 0.
S-wiring   short avg runs (2 tasks x 2 epochs, SN3 and SNA): every forward of the host's
           training loop, evaluate_cnn and preact_hist goes through the swapped function;
           F.max_pool2d is called only by evaluate_cnn (mob_pool); an independent forward
           built from nn.AvgPool2d with the captured end weights (and, for SNA, the
           captured alphas) reproduces the recorded memo_acc and zbar_c2.
           Mutations: the same independent forward with nn.MaxPool2d misses zbar_c2; the
           host's own forward_cnn left in place is invisible to the call counter.
S-reuse    (400 epochs) the swapped forward with pool = max reproduces the stored
           reference rows (SNA, SN3; seed 0, task 1) in every csv column at 10 digits.
           Power: the same rows miss the reference seed-1 rows.
S-switch   (400 epochs) with pool = avg the task-1 row differs from the reference row in
           online_acc and every w_norm column.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import avgpool_cnn_0917 as AP
from src import pmnist_0905 as H
from src import pmnist_rlcifar_0907 as RC
from src import rlcifar_cnn_0908 as CN

REF = Path("/home/issan/Projects/claude/proj_004_drift/results/rlcifar_cnn_0908")
OUTDIR = H.REPO / "results" / "_checks_avgpool_cnn_0917"
HOST_FORWARD = CN.forward_cnn


def s_pool(device) -> dict:
    g = torch.Generator().manual_seed(0)
    a = torch.randn(4, 3, 32, 32, generator=g).to(device).requires_grad_(True)
    u = torch.randn(4, 3, 16, 16, generator=g).to(device)
    out = {}
    for name in ("avg", "max"):
        p = AP.POOLS[name](a, CN.POOL, CN.POOL)
        manual = a.detach().reshape(4, 3, 16, 2, 16, 2).mean((3, 5))
        (gr,) = torch.autograd.grad((p * u).sum(), a)
        blocks = gr.reshape(4, 3, 16, 2, 16, 2).permute(0, 1, 2, 4, 3, 5).reshape(4, 3, 16, 16, 4)
        want = (u / 4)[..., None].expand_as(blocks)
        out[name] = {"value_err": float((p.detach() - manual).abs().max()),
                     "grad_err": float((blocks - want).abs().max()),
                     "zero_grad_frac": float((blocks == 0).float().mean())}
    avg, mx = out["avg"], out["max"]
    ok = avg["value_err"] < 1e-6 and avg["grad_err"] < 1e-7 and avg["zero_grad_frac"] == 0.0
    mut = mx["zero_grad_frac"] >= 0.74 and mx["grad_err"] > 1e-3
    out.update(mutation_detected=mut, **{"pass": ok and mut})
    return out


def snake(z, a):
    return z + torch.sin(a * z).pow(2) / a


def indep_forward(params, x, alphas, pool_mod, device):
    """The CNN written again from torch.nn parts, not from the host's function."""
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = [p.detach() for p in params]
    c1 = torch.nn.Conv2d(3, 16, 5, padding=2).to(device)
    c2 = torch.nn.Conv2d(16, 16, 5, padding=2).to(device)
    with torch.no_grad():
        c1.weight.copy_(W1); c1.bias.copy_(b1); c2.weight.copy_(W2); c2.bias.copy_(b2)
        z1 = c1(x)
        z2 = c2(pool_mod(snake(z1, alphas[0])))
        h = pool_mod(snake(z2, alphas[1])).flatten(1)
        a3 = snake(F.linear(h, W3, b3), alphas[2])
        a4 = snake(F.linear(a3, W4, b4), alphas[3])
        return z2, F.linear(a4, W5, b5)


def s_wiring(device, data) -> dict:
    out, ok = {}, True
    calls, mcalls, made = Counter(), Counter(), []
    orig_max, orig_cs = F.max_pool2d, CN.ChannelSnake

    class Rec(orig_cs):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    def max_spy(*a, **k):
        mcalls[sys._getframe(1).f_code.co_name] += 1
        return orig_max(*a, **k)

    x = CN.images(data, CN.subset_idx(0), device)
    for arm in ("SN3_avg", "SNA_avg"):
        calls.clear(); mcalls.clear(); made.clear()
        host_arm, pool = AP.ARMS[arm]
        fwd = AP.use_pool(pool)

        def spy(params, xx, act):
            calls[sys._getframe(1).f_code.co_name] += 1
            return fwd(params, xx, act)

        CN.forward_cnn, CN.ChannelSnake, F.max_pool2d = spy, Rec, max_spy
        dbg = {}
        try:
            with tempfile.TemporaryDirectory() as td:
                rows, _ = CN.run_one(host_arm, 0, AP.LR, 2, data, device, epochs=2,
                                     hist_dir=Path(td), debug=dbg)
        finally:
            F.max_pool2d, CN.ChannelSnake = orig_max, orig_cs
        want_calls = {"run_one": 2 * 2 * CN.STEPS_PER_EPOCH, "evaluate_cnn": 2, "preact_hist": 2}
        if host_arm == "SNA":
            assert len(made) == 1, len(made)
            act = made[0]
            alphas = [act._bcast(l, act.alpha(l)) for l in range(4)]
            v_moved = all(bool((v != 1).any()) for v in act.V)
        else:
            alphas, v_moved = [H.ARMS["SN3"].param] * 4, None
        y = dbg["labels"][-1].to(device)
        res = {}
        for name, mod in (("avg", torch.nn.AvgPool2d(2, 2)), ("max", torch.nn.MaxPool2d(2, 2))):
            z2, logits = indep_forward(dbg["params_end"], x, alphas, mod, device)
            zb = float(z2.mean((0, 2, 3)).median())
            res[name] = {"memo": float((logits.argmax(1) == y).float().mean()), "zbar_c2": zb,
                         "zbar_rel_err": abs(zb - rows[-1]["zbar_c2"]) / abs(rows[-1]["zbar_c2"])}
        # mutation: the host's forward left in place never reaches the counter
        calls_before = sum(calls.values())
        CN.forward_cnn = HOST_FORWARD
        CN.run_one(host_arm, 0, AP.LR, 1, data, device, epochs=1)
        unseen = sum(calls.values()) == calls_before
        good = (dict(calls) == want_calls and dict(mcalls) == {"evaluate_cnn": 4}
                and abs(res["avg"]["memo"] - rows[-1]["memo_acc"]) <= 1 / CN.N_IMAGES
                and res["avg"]["zbar_rel_err"] < 1e-5 and res["max"]["zbar_rel_err"] > 1e-3
                and unseen and v_moved in (None, True))
        out[arm] = {"calls": dict(calls), "want_calls": want_calls, "max_pool2d_callers": dict(mcalls),
                    "recorded": {"memo_acc": rows[-1]["memo_acc"], "zbar_c2": rows[-1]["zbar_c2"]},
                    "independent": res, "host_forward_unseen": unseen, "sna_V_moved": v_moved,
                    "pass": good}
        ok &= good
    CN.forward_cnn = HOST_FORWARD
    out["pass"] = ok
    return out


def s_reuse_switch(device, data) -> tuple[dict, dict]:
    fmt = lambda v: f"{v:.10g}" if isinstance(v, float) else str(v)
    reuse, switch = {}, {}
    for host in ("SNA", "SN3"):
        ref = list(csv.DictReader(open(REF / host / "per_task.csv")))
        r0 = next(r for r in ref if r["seed"] == "0" and r["task"] == "1")
        r1 = next(r for r in ref if r["seed"] == "1" and r["task"] == "1")
        for pool in ("max", "avg"):
            t0 = time.time()
            rows, _ = AP.run_arm(f"{host}_{pool}", 0, data, device, tasks=1, epochs=AP.EPOCHS)
            got = rows[0]
            bad = [k for k in r0 if fmt(got.get(k)) != r0[k]]
            if pool == "max":
                power = [k for k in r1 if k != "seed" and fmt(got.get(k)) != r1[k]]
                reuse[host] = {"columns": len(r0), "mismatched": bad, "power_mismatched": len(power),
                               "seconds": round(time.time() - t0, 1),
                               "pass": (not bad) and len(power) > 5}
            else:
                learn = ["online_acc"] + [k for k in r0 if k.startswith("w_norm_")]
                differ = [k for k in learn if fmt(got.get(k)) != r0[k]]
                switch[host] = {"learning_cols": learn, "differ": differ,
                                "online_acc": got["online_acc"], "ref_online_acc": r0["online_acc"],
                                "memo_acc": got["memo_acc"], "ref_memo_acc": r0["memo_acc"],
                                "n_columns_differing": len(bad), "seconds": round(time.time() - t0, 1),
                                "pass": len(differ) == len(learn)}
            print(f"  {host} pool={pool} {time.time() - t0:.0f}s", flush=True)
    reuse["ref_sha256"] = {h: AP.sha256(REF / h / "per_task.csv") for h in ("SNA", "SN3")}
    reuse["pass"] = all(v["pass"] for k, v in reuse.items() if k in ("SNA", "SN3"))
    switch["pass"] = all(v["pass"] for k, v in switch.items() if k in ("SNA", "SN3"))
    return reuse, switch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    device = H.setup(a.device)
    torch.set_num_threads(a.threads)
    data = RC.Cifar10()
    res = {"device": str(device), "torch": torch.__version__, **AP.git_state(),
           "started": time.strftime("%F %T")}
    res["S-pool"] = s_pool(device)
    print("S-pool", res["S-pool"]["pass"], flush=True)
    res["S-wiring"] = s_wiring(device, data)
    print("S-wiring", res["S-wiring"]["pass"], flush=True)
    res["S-reuse"], res["S-switch"] = s_reuse_switch(device, data)
    print("S-reuse", res["S-reuse"]["pass"], "S-switch", res["S-switch"]["pass"], flush=True)
    res["all_pass"] = all(v["pass"] for k, v in res.items() if k.startswith("S-"))
    res["finished"] = time.strftime("%F %T")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    name = "checks.json" if a.device == "cuda" else f"checks_{a.device}.json"
    (OUTDIR / name).write_text(json.dumps(res, indent=1, default=str))
    print("all_pass", res["all_pass"])


if __name__ == "__main__":
    main()

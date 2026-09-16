#!/usr/bin/env python3
"""Checks for swish_battle_0917 (spec §4).  Writes results/_checks_swish_battle_0917/checks.json.

S-act       dphi == autograd d/dz phi (float64), fixed and both adaptive classes.
            Mutation: the gate without its a z s (1-s) term must fail.
S-scale     phi_{a/s}(s z) == s phi_a(z) and phi'_{a/s}(s z) == phi'_a(z) (float64).
            Mutation: sigmoid(a z) replaced by sigmoid(a z + 0.1 z) must fail (a constant
            shift inside the sigmoid would not: a z itself is scale-free).
S-ema-off   SWA<c> with beta = 0 gives exactly the rows of SW<c> through the host run_one,
            both boxes, c = 1 and 3.  Mutation: beta = 0.01 must differ.
S-fresh     two SWA1 runs in one process give identical rows (no state carried over).
S-wiring    after a short adaptive run: V has moved, alpha * W == c, and the logits of
            the captured final weights under an independent forward (layer l uses alpha_l)
            equal the host's; the recorded memo_acc equals the accuracy recomputed from
            them.  Mutation: swapping the alphas of two layers must change the logits.
S-reuse     (--reuse, cuda) the unchanged hosts reproduce the stored reference rows
            (rlcifar_cnn_0908 SNA/LR, pmnist_rlmnist_0906 SNA; seed 0, task 1) in every
            column to the 10 significant digits the csv keeps.  Power: the same row must
            mismatch the stored seed-1 row.

Thresholds: float64 roundoff of phi' on |a z| <= 40 is below 40 * 2^-52 ~ 1e-14 per term,
so 1e-12 leaves two orders of headroom; every mutation moves the compared value by
O(1e-2) or more.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import swish_battle_0917 as SB   # noqa: E402  (sets thread/env defaults first)
from src import pmnist_0905 as H          # noqa: E402
from src import pmnist_rlmnist_0906 as RL  # noqa: E402
from src import rlcifar_cnn_0908 as CN    # noqa: E402

TOL = 1e-12
OUTDIR = H.REPO / "results" / "_checks_swish_battle_0917"
REF = {"cnn": Path("/home/issan/Projects/claude/proj_004_drift/results/rlcifar_cnn_0908"),
       "mlp": H.REPO / "results" / "pmnist_rlmnist_0906"}
SHORT = dict(tasks=1, epochs=2)                  # 150 Adam steps


def s_act() -> dict:
    torch.manual_seed(0)
    z = torch.cat([torch.linspace(-12, 12, 4001), 6 * torch.randn(4000)]).double()
    out, worst = {}, 0.0
    for a in (1.0, 3.0, 0.37):
        zz = z.clone().requires_grad_(True)
        g, = torch.autograd.grad(SB.swish(zz, a).sum(), zz)
        err = float((SB.Swish("t", a).dphi(z) - g).abs().max())
        worst = max(worst, err)
        out[f"fixed_a{a}"] = err
    # adaptive: random V, conv site (N,C,H,W) and fc site (N,C)
    cs = SB.ChannelSwish(1.0, 0.01, "cpu")
    ms = SB.AdaptiveSwish(3.0, 0.01, "cpu")
    for obj in (cs, ms):
        for l in range(len(obj.V)):
            obj.V[l] = (torch.rand(obj.V[l].shape) * 50 + 0.01).double()
    for name, obj, shapes in (("channel", cs, [(8, 16, 5, 5), (8, 16, 3, 3), (8, 100), (8, 100)]),
                              ("unit", ms, [(64, 100), (64, 100)])):
        for l, shp in enumerate(shapes):
            zz = (8 * torch.randn(shp, dtype=torch.float64)).requires_grad_(True)
            g, = torch.autograd.grad(obj.phi(zz, l).sum(), zz)
            err = float((obj.dphi(zz.detach(), l) - g).abs().max())
            worst = max(worst, err)
            out[f"{name}_l{l}"] = err
    a = 1.0
    zz = z.clone().requires_grad_(True)
    g, = torch.autograd.grad(SB.swish(zz, a).sum(), zz)
    mut = float((torch.sigmoid(a * z) - g).abs().max())
    out.update(worst=worst, mutation_err=mut)
    out["pass"] = worst < TOL and mut > 1e-2
    return out


def s_scale() -> dict:
    torch.manual_seed(1)
    z = (5 * torch.randn(10000)).double()
    worst, out = 0.0, {}
    for a in (1.0, 3.0):
        for s in (0.1, 7.0):
            e1 = float(((SB.swish(s * z, a / s) - s * SB.swish(z, a)).abs() / (s * (1 + z.abs()))).max())
            e2 = float((SB.swish_gate(s * z, a / s) - SB.swish_gate(z, a)).abs().max())
            out[f"a{a}_s{s}"] = [e1, e2]
            worst = max(worst, e1, e2)
    s, a = 7.0, 1.0
    bad = lambda zz, aa: zz * torch.sigmoid(aa * zz + 0.1 * zz)
    mut = float(((bad(s * z, a / s) - s * bad(z, a)).abs() / (s * (1 + z.abs()))).max())
    out.update(worst=worst, mutation_err=mut, **{"pass": worst < TOL and mut > 1e-2})
    return out


def rows_equal(r1: list[dict], r2: list[dict], ignore=("arm",)) -> tuple[bool, list[str]]:
    """Exact equality of every shared column except the arm label."""
    diff = []
    for a, b in zip(r1, r2):
        for k in (set(a) & set(b)) - set(ignore):
            if a[k] != b[k] and not (a[k] != a[k] and b[k] != b[k]):
                diff.append(k)
    return (len(r1) == len(r2) and not diff), sorted(set(diff))


def s_ema_off(device, data: dict) -> dict:
    out, ok = {}, True
    for box in SB.BOXES:
        for c, fixed, ada in ((1.0, "SW1", "SWA1"), (3.0, "SW3", "SWA3"), (1.0, "SW1", "SWA1u")):
            rf, _ = SB.run_host(box, fixed, 0, data[box], device, **SHORT)
            r0, _ = SB.run_host(box, ada, 0, data[box], device, beta=0.0, **SHORT)
            r1, _ = SB.run_host(box, ada, 0, data[box], device, beta=0.01, **SHORT)
            eq, diff = rows_equal(rf, r0)
            eq_mut, diff_mut = rows_equal(rf, r1)
            alpha_c = all(r0[-1][f"alpha_med_{t}"] == c and r0[-1][f"alpha_W_med_{t}"] == c
                          for t in SB.TAGS[box])
            train_cols = [k for k in diff_mut if k in ("online_acc", "memo_acc") or k.startswith("w_norm")]
            lo = SB.arm_act(box, ada, device)[0].lo
            out[f"{box}_{ada}"] = {"equal": eq, "diff": diff, "alpha_is_c": alpha_c, "lo": lo,
                                  "mutation_differs": not eq_mut, "mutation_train_cols": train_cols}
            want_lo = 1e-8 if ada.endswith("u") else 1e-3
            ok &= eq and alpha_c and (not eq_mut) and bool(train_cols) and abs(lo - want_lo) < 1e-12
    out["pass"] = ok
    return out


def s_fresh(device, data: dict) -> dict:
    out, ok = {}, True
    for box in SB.BOXES:
        ra, _ = SB.run_host(box, "SWA1", 0, data[box], device, **SHORT)
        rb, _ = SB.run_host(box, "SWA1", 0, data[box], device, **SHORT)
        eq, diff = rows_equal(ra, rb)
        out[box] = {"equal": eq, "diff": diff}
        ok &= eq
    out["pass"] = ok
    return out


def fwd_mlp(params, x, al):
    W1, b1, W2, b2, W3, b3 = params
    z1 = x @ W1.T + b1
    z2 = SB.swish(z1, al[0]) @ W2.T + b2
    return SB.swish(z2, al[1]) @ W3.T + b3


def fwd_cnn(params, x, al):
    Wc1, bc1, Wc2, bc2, Wf1, bf1, Wf2, bf2, Wf3, bf3 = params
    a = [al[0][None, :, None, None], al[1][None, :, None, None], al[2][None, :], al[3][None, :]]
    z1 = F.conv2d(x, Wc1, bc1, padding=CN.PAD)
    z2 = F.conv2d(F.max_pool2d(SB.swish(z1, a[0]), CN.POOL, CN.POOL), Wc2, bc2, padding=CN.PAD)
    h = F.max_pool2d(SB.swish(z2, a[1]), CN.POOL, CN.POOL).flatten(1)
    z3 = h @ Wf1.T + bf1
    z4 = SB.swish(z3, a[2]) @ Wf2.T + bf2
    return SB.swish(z4, a[3]) @ Wf3.T + bf3


def s_wiring(device, data: dict) -> dict:
    out, ok = {}, True
    for box in SB.BOXES:
        acts, dbg, cap = [], {}, {}
        if box == "mlp":
            orig = RL.evaluate_rl

            def spy(params, x, y, act):
                cap.update(params=[p.detach().clone() for p in params], x=x, y=y.clone())
                return orig(params, x, y, act)
            RL.evaluate_rl = spy
            try:
                rows, _ = SB.run_host(box, "SWA3", 0, data[box], device, act_out=acts, **SHORT)
            finally:
                RL.evaluate_rl = orig
            params, x, y = cap["params"], cap["x"], cap["y"]
            host_logits = H.forward(params, x, acts[0])[-1]
            fwd = fwd_mlp
        else:
            rows, _ = SB.run_host(box, "SWA3", 0, data[box], device, act_out=acts, debug=dbg, **SHORT)
            params = dbg["params_end"]
            x = CN.images(data[box], CN.subset_idx(0), device)
            y = dbg["labels"][-1].to(device)
            host_logits = CN.forward_cnn(params, x, acts[0])[-1]
            fwd = fwd_cnn
        act = acts[0]
        al = [act.alpha(l) for l in range(len(act.V))]
        with torch.no_grad():
            ind = fwd(params, x, al)
            sw = list(al)
            sw[0], sw[1] = al[1], al[0]
            mut = fwd(params, x, sw)
        moved = min(float((v - 1).abs().max()) for v in act.V)
        aw = max(abs(rows[-1][f"alpha_W_med_{t}"] - act.c) for t in SB.TAGS[box])
        acc = float((ind.argmax(1) == y).float().mean())
        r = {"V_moved_min": moved, "alpha_W_minus_c": aw,
             "logits_equal": bool(torch.equal(ind, host_logits)),
             "memo_acc": rows[-1]["memo_acc"], "recomputed_acc": acc,
             "mutation_logit_diff": float((mut - ind).abs().max()),
             "clip_frac": max(rows[-1][f"alpha_clip_frac_{t}"] for t in SB.TAGS[box])}
        r["pass"] = (moved > 1e-3 and aw < 1e-5 and r["logits_equal"] and acc == r["memo_acc"]
                     and r["mutation_logit_diff"] > 1e-4)
        out[box] = r
        ok &= r["pass"]
    out["pass"] = ok
    return out


def s_reuse(device, data: dict) -> dict:
    out, ok = {}, True
    fmt = lambda v: f"{v:.10g}" if isinstance(v, float) else str(v)
    for box, arm in (("cnn", "SNA"), ("cnn", "LR"), ("mlp", "SNA")):
        t0 = time.time()
        rows, _ = SB.run_host(box, arm, 0, data[box], device, tasks=1, epochs=SB.EPOCHS)
        ref = list(csv.DictReader(open(REF[box] / arm / "per_task.csv")))
        r0 = next(r for r in ref if r["seed"] == "0" and r["task"] == "1")
        r1 = next(r for r in ref if r["seed"] == "1" and r["task"] == "1")
        got = rows[0]
        bad = [k for k in r0 if fmt(got.get(k)) != r0[k]]
        power = [k for k in r1 if k not in ("seed",) and fmt(got.get(k)) != r1[k]]
        out[f"{box}_{arm}"] = {"columns": len(r0), "mismatched": bad, "power_mismatched": len(power),
                               "seconds": time.time() - t0}
        ok &= (not bad) and len(power) > 5
    out["pass"] = ok
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reuse", action="store_true", help="also run S-reuse (cuda, ~3 min)")
    a = ap.parse_args()
    device = H.setup(a.device)
    data = {b: SB.load_data(b, device) for b in SB.BOXES}
    res = {"device": str(device), "torch": torch.__version__, **SB.git_state()}
    for name, fn in (("S-act", s_act), ("S-scale", s_scale)):
        res[name] = fn()
        print(name, res[name]["pass"], flush=True)
    for name, fn in (("S-ema-off", s_ema_off), ("S-fresh", s_fresh), ("S-wiring", s_wiring)):
        res[name] = fn(device, data)
        print(name, res[name]["pass"], flush=True)
    if a.reuse:
        res["S-reuse"] = s_reuse(device, data)
        print("S-reuse", res["S-reuse"]["pass"], flush=True)
    res["all_pass"] = all(v["pass"] for k, v in res.items() if k.startswith("S-"))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    name = "checks.json" if a.device == "cuda" else f"checks_{a.device}.json"
    (OUTDIR / name).write_text(json.dumps(res, indent=1, default=str))
    print("all_pass", res["all_pass"])


if __name__ == "__main__":
    main()

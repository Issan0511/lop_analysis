#!/usr/bin/env python3
"""Checks for the bundled CNN engine of sna_cnn_cause_1009.

    python3 src/sna_cnn_cause_1009_checks.py --out results/_checks_sna_cnn_cause_1009

S-stream   labels and batch orders = the host's streams, drawn independently
S-host     one step (loss, every gradient) and one epoch (every parameter) against the host's
           own forward / Adam loop (`rlcifar_cnn_0908`), SNA and SN3; tolerance from float32
S-indep    perturbing run 1 (parameters, then images) leaves runs 0 and 2 bit-identical,
           while run 1 itself changes (the check is not vacuous)
S-graph    two epochs replayed from the CUDA graph = the same two epochs eager, bit for bit
S-eval     `evaluate` = the host's `evaluate_cnn` on the same state (SNA, SN3)
S-pp       the within-position variance is what it says; the pooled one is the host's
S-freeze   after `freeze`, alpha stays put while V keeps moving; unfrozen runs keep tracking
S-cost     seconds per epoch at R = 10, 30, 60 (graph)
S-fixconv  `+fixconv` runs keep both conv layers bit-identical to their init while their fc
           layers move; the unmasked run in the same bundle moves its conv layers; a bundle
           without any fixconv run takes the unmasked step (no mask is built)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H
from src import pmnist_rlcifar_0907 as RC
from src import rlcifar_cnn_0908 as CN
from src import sna_cnn_cause_1009 as E


def host_act(spec: dict, device):
    """The host's activation for an all-adaptive or all-fixed arm."""
    kinds = {k for k, v in spec["site"]}
    vals = {v for k, v in spec["site"]}
    assert len(kinds) == 1 and len(vals) == 1, "host acts are uniform over the sites"
    (kind,), (val,) = kinds, vals
    if kind == "c":
        return CN.ChannelSnake(val, 0.01, device, lo=0.005, hi=3.0)
    return H.Snake(f"SN{val}", val) if hasattr(H, "Snake") else None


def host_fixed_snake(alpha: float):
    class _S:
        def phi(self, z):
            return z + torch.sin(alpha * z) ** 2 / alpha

        def dphi(self, z):
            return 1.0 + torch.sin(2.0 * alpha * z)
    return _S()


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


def check_stream(cifar, device) -> dict:
    s = 10
    B = E.Bundle([("SNA", s), ("SNAc3", s), ("SN3", 11)], cifar, device, graph=False)
    g_lab, g_b = H.stream("rlcc_labels", s), H.stream("rlcc_batch", s)
    ok = True
    for t in range(2):
        B.new_labels()
        y = CN.task_labels(g_lab)
        ok &= bool((B.Y[0].cpu() == y).all() and (B.Y[1].cpu() == y).all())
        for e in range(3):
            B.new_order()
            o = torch.randperm(CN.N_IMAGES, generator=g_b)
            ok &= bool((B.order[0].cpu() == o).all() and (B.order[1].cpu() == o).all())
    x_host = CN.images(cifar, CN.subset_idx(11), device)
    ok &= bool((B.X[2] == x_host).all())
    ok &= not bool((B.Y[2] == B.Y[0]).all())             # another seed, other labels
    return {"pass": ok}


def host_loss_grads(params, xb, yb, act, ada: bool):
    out = CN.forward_cnn(params, xb, act)
    loss = F.cross_entropy(out[8], yb)
    g = torch.autograd.grad(loss, params)
    return loss.detach(), g, out


def check_host(cifar, device) -> dict:
    """Bundle (R=1) against the host on the same batch, then over one epoch of 75 steps."""
    res = {}
    for arm, s in (("SNA", 10), ("SN3", 12)):
        B = E.Bundle([(arm, s)], cifar, device, graph=False)
        B.new_labels(); B.new_order()
        params = CN.init_params(s, device)
        spec = E.parse_arm(arm)
        if spec["site"][0][0] == "c":
            act = CN.ChannelSnake(0.6, 0.01, device, lo=0.005, hi=3.0)
        else:
            act = host_fixed_snake(3.0)
        init_eq = all(bool((p.detach() == q.detach()[0]).all()) for p, q in zip(params, B.P))
        x = B.X[0]; y = B.Y[0]; order = B.order[0]
        idx = order[:CN.BATCH]
        lh, gh, _ = host_loss_grads(params, x[idx], y[idx], act, spec["site"][0][0] == "c")
        out = E.forward(B.P, B.X[:, idx], B.act)
        lb = F.cross_entropy(out[8][0], y[idx])
        gb = torch.autograd.grad(lb, B.P)
        step_rel = max(rel(a[0], b) for a, b in zip(gb, gh))
        loss_rel = abs(float(lb) - float(lh)) / abs(float(lh))
        # one epoch: host loop (copied from rlcifar_cnn_0908.run_one, Adam branch) vs bundle eager
        m = [torch.zeros_like(q) for q in params]; v = [torch.zeros_like(q) for q in params]
        tc = 0
        for j in range(CN.STEPS_PER_EPOCH):
            ib = order[j * CN.BATCH:(j + 1) * CN.BATCH]
            o = CN.forward_cnn(params, x[ib], act)
            loss = F.cross_entropy(o[8], y[ib])
            grads = torch.autograd.grad(loss, params)
            with torch.no_grad():
                tc += 1
                c1 = 1 - 0.9 ** tc; c2 = 1 - 0.999 ** tc
                for p, gr, mi, vi in zip(params, grads, m, v):
                    mi.mul_(0.9).add_(gr, alpha=1 - 0.9)
                    vi.mul_(0.999).addcmul_(gr, gr, value=1 - 0.999)
                    p -= 1e-3 * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
                if isinstance(act, CN.ChannelSnake):
                    act.update(o[0:8:2])
        B.epoch_body()
        ep_rel = max(rel(b.detach()[0], a.detach()) for a, b in zip(params, B.P))
        ep_absmax = max(float((b.detach()[0] - a.detach()).abs().max()) for a, b in zip(params, B.P))
        V_rel = (max(rel(B.act.V[l][0], act.V[l]) for l in range(4))
                 if isinstance(act, CN.ChannelSnake) else 0.0)
        # tolerance: one step agrees to float32 rounding (1e-5 relative); after 75 Adam
        # steps the per-coordinate step is lr = 1e-3 in size, so rounding-level differences
        # in m/v of coordinates with tiny v can flip a few steps: allow 2 lr of drift.
        ok = init_eq and loss_rel < 1e-5 and step_rel < 1e-4 and ep_absmax < 2e-3 and V_rel < 1e-3
        res[arm] = {"init_equal": init_eq, "loss_rel": loss_rel, "grad_rel_max": step_rel,
                    "epoch_param_rel_max": ep_rel, "epoch_param_absmax": ep_absmax,
                    "epoch_V_rel_max": V_rel, "pass": ok}
    res["pass"] = all(r["pass"] for r in res.values() if isinstance(r, dict))
    return res


def check_indep(cifar, device) -> dict:
    slots = [("SNA", 10), ("SNAc3", 11), ("SN3", 12)]
    B = E.Bundle(slots, cifar, device, graph=False)
    B.new_labels(); B.new_order()
    idx = B.order[:, :CN.BATCH]

    def lg():
        out = E.forward(B.P, B.X[B.ar, idx], B.act)
        lv = F.cross_entropy(out[8].reshape(-1, 10), B.Y[B.ar, idx].reshape(-1),
                             reduction="none").view(3, -1).mean(1)
        g = torch.autograd.grad(lv.sum(), B.P)
        return lv.detach(), [q.detach().clone() for q in g]

    l0, g0 = lg()
    with torch.no_grad():
        for p in B.P:
            p[1] += 0.05 * torch.randn_like(p[1])
    l1, g1 = lg()
    same_p = all(bool((a[r] == b[r]).all()) for a, b in zip(g0, g1) for r in (0, 2)) \
        and bool((l0[[0, 2]] == l1[[0, 2]]).all())
    moved_p = bool(l0[1] != l1[1]) and any(bool((a[1] != b[1]).any()) for a, b in zip(g0, g1))
    with torch.no_grad():
        B.X[1] = torch.rand_like(B.X[1])
    l2, g2 = lg()
    same_x = all(bool((a[r] == b[r]).all()) for a, b in zip(g1, g2) for r in (0, 2)) \
        and bool((l1[[0, 2]] == l2[[0, 2]]).all())
    moved_x = bool(l1[1] != l2[1])
    return {"others_bit_identical_param": same_p, "run1_changed_param": moved_p,
            "others_bit_identical_input": same_x, "run1_changed_input": moved_x,
            "pass": same_p and moved_p and same_x and moved_x}


def check_graph(cifar, device) -> dict:
    slots = [("SNA", 10), ("CV3FC06", 11), ("SN3", 12), ("SNApp", 13)]
    Bg = E.Bundle(slots, cifar, device, graph=True)
    Be = E.Bundle(slots, cifar, device, graph=False)
    for B in (Bg, Be):
        B.new_labels()
        accs = []
        for e in range(2):
            B.run_epoch()
            accs.append(B.acc_ep.clone())
        B._accs = accs
    tens_g = (*Bg._state_tensors(), *Bg.act.V)
    tens_e = (*Be._state_tensors(), *Be.act.V)
    eq = all(bool((a == b).all()) for a, b in zip(tens_g, tens_e))
    eq_acc = all(bool((a == b).all()) for a, b in zip(Bg._accs, Be._accs))
    moved = bool((Bg.P[0] != E.stack_init([s for a, s in slots], device)[0]).any())
    maxdiff = max(float((a.double() - b.double()).abs().max()) for a, b in zip(tens_g, tens_e))
    return {"bit_identical": eq, "acc_identical": eq_acc, "params_moved": moved,
            "max_abs_diff": maxdiff, "pass": eq and eq_acc and moved}


def check_eval(cifar, device) -> dict:
    res = {}
    for arm, s in (("SNA", 10), ("SN3", 12)):
        B = E.Bundle([(arm, s)], cifar, device, graph=False)
        B.new_labels()
        for e in range(3):
            B.run_epoch()
        ev = E.evaluate(B.P, B.X, B.Y, B.act)[0]
        params = [p.detach()[0].clone() for p in B.P]
        if arm == "SNA":
            act = CN.ChannelSnake(0.6, 0.01, device, lo=0.005, hi=3.0)
            for l in range(4):
                act.V[l].copy_(B.act.V[l][0])
        else:
            act = host_fixed_snake(3.0)
        hv = CN.evaluate_cnn(params, B.X[0], B.Y[0], act)
        keys = ["acc"] + [f"{p}_{t}" for t in CN.SITES for p in ("zbar", "zsd", "mob", "eff_rank")] \
            + [f"mob_pool_{t}" for t in ("c1", "c2")] + [f"w_norm_{t}" for t in CN.WEIGHT_TAGS]
        if arm == "SNA":
            keys += [f"alpha_med_{t}" for t in CN.SITES]
        # The host's full-batch forward and the chunked one run TF32 convs (cudnn's default,
        # the same in both), so they agree to ~1e-4 relative.  A channel mean near 0 makes a
        # relative error meaningless: zbar is compared on the scale of its own spread, zsd.
        worst, wk = 0.0, None
        for k in keys:
            a = ev["memo_acc"] if k == "acc" else ev[k]
            b = hv[k]
            scale = hv[k.replace("zbar", "zsd")] if k.startswith("zbar") else abs(b)
            d = abs(a - b) / max(scale, 1e-6)
            if d > worst:
                worst, wk = d, k
        res[arm] = {"worst_rel": worst, "worst_key": wk, "pass": worst < 1e-3}
    res["pass"] = all(r["pass"] for r in res.values() if isinstance(r, dict))
    return res


def check_pp(device) -> dict:
    specs = [E.parse_arm("SNA"), E.parse_arm("SNApp")]
    act = E.BundleSnake(specs, device)
    g = torch.Generator(device="cpu").manual_seed(0)
    z = (torch.randn(16, 2 * 16, 8, 8, generator=g)
         + 3 * torch.randn(1, 2 * 16, 8, 8, generator=g)).to(device)   # strong position means
    v = act.batch_var(z, 1)
    zr = z.view(16, 2, 16, 8, 8)
    pooled0 = zr[:, 0].permute(1, 0, 2, 3).reshape(16, -1).var(1, unbiased=False)
    within1 = zr[:, 1].var(0, unbiased=False).mean((1, 2))
    ok_pool = rel(v[0], pooled0) < 1e-5
    ok_within = rel(v[1], within1) < 1e-5
    differs = float((zr[:, 1].permute(1, 0, 2, 3).reshape(16, -1).var(1, unbiased=False) / within1).mean())
    return {"pooled_is_host": ok_pool, "within_is_within": ok_within,
            "pooled_over_within_on_probe": differs, "pass": ok_pool and ok_within and differs > 2}


def check_freeze(cifar, device) -> dict:
    B = E.Bundle([("SNA", 10), ("SNAfrz1", 10)], cifar, device, graph=False)
    B.new_labels()
    B.run_epoch()
    B.act.freeze([1])
    a_before = [B.act.alpha(l).clone() for l in range(4)]
    V_before = [v.clone() for v in B.act.V]
    for e in range(2):
        B.run_epoch()
    frozen_const = all(bool((B.act.alpha(l)[1] == a_before[l][1]).all()) for l in range(4))
    V_moved = all(bool((B.act.V[l][1] != V_before[l][1]).any()) for l in range(4))
    tracker_moved = any(bool((B.act.alpha(l)[0] != a_before[l][0]).any()) for l in range(4))
    return {"frozen_alpha_constant": frozen_const, "V_still_moves": V_moved,
            "unfrozen_alpha_moves": tracker_moved,
            "pass": frozen_const and V_moved and tracker_moved}


def check_fixconv(cifar, device) -> dict:
    slots = [("SNA+fixconv", 10), ("SNA", 10), ("CV06FC3+fixconv", 11)]
    B = E.Bundle(slots, cifar, device, graph=True)
    init = [p.detach().clone() for p in B.P]
    B.new_labels()
    for e in range(2):
        B.run_epoch()
    conv_fixed = all(bool((B.P[i][r] == init[i][r]).all()) for i in range(4) for r in (0, 2))
    fc_moved = all(bool((B.P[i][r] != init[i][r]).any()) for i in range(4, 10) for r in (0, 2))
    other_moved = all(bool((B.P[i][1] != init[i][1]).any()) for i in range(4))
    no_mask = E.Bundle([("SNA", 10)], cifar, device, graph=False).upd_mask is None
    return {"conv_fixed": conv_fixed, "fc_moved": fc_moved, "unmasked_run_conv_moved": other_moved,
            "no_mask_without_fixconv": no_mask,
            "pass": conv_fixed and fc_moved and other_moved and no_mask}


def check_fcamp(cifar, device) -> dict:
    """`+fcamp<a>`: phi' = 1 + a sin(2 alpha z) at f1/f2 (checked against autograd), the conv
    sites untouched, and a = 1 in a mixed bundle gives the plain Snake bit for bit."""
    specs = [E.parse_arm("SNA+fcamp0.7"), E.parse_arm("SNA")]
    act = E.BundleSnake(specs, device)
    plain = E.BundleSnake([E.parse_arm("SNA")], device)
    g = torch.Generator(device="cpu").manual_seed(1)
    z = (3 * torch.randn(2, 16, 100, generator=g)).to(device).requires_grad_(True)
    y = act.phi(z, 3)
    (gz,) = torch.autograd.grad(y.sum(), z)
    d = act.dphi(z.detach(), 3)
    deriv_ok = rel(gz, d) < 1e-5
    floor_ok = bool((d[0] >= 0.3 - 1e-6).all()) and float(d[1].min()) < 0.05
    same_plain = bool((act.phi(z.detach(), 3)[1] == plain.phi(z.detach()[1:2], 3)[0]).all())
    zc = torch.randn(4, 2 * 16, 8, 8, generator=g).to(device)
    conv_plain = bool((act.phi(zc, 1)[:, :16] == E.BundleSnake([E.parse_arm("SNA")], device).phi(zc[:, :16], 1)).all())
    no_amp = plain.amp is None
    return {"deriv_matches_autograd": deriv_ok, "floor_0.3_and_plain_reaches_0": floor_ok,
            "a1_run_bit_identical": same_plain, "conv_untouched": conv_plain, "no_amp_tensor_when_unused": no_amp,
            "pass": deriv_ok and floor_ok and same_plain and conv_plain and no_amp}


def check_cost(cifar, device) -> dict:
    out = {}
    for R in (10, 30, 60):
        slots = [("SNA", 10 + (r % 10)) for r in range(R)]
        B = E.Bundle(slots, cifar, device, graph=True)
        B.new_labels()
        B.run_epoch()                      # capture
        torch.cuda.synchronize()
        t = time.time()
        for e in range(5):
            B.run_epoch()
        torch.cuda.synchronize()
        sec = (time.time() - t) / 5
        out[f"R{R}"] = {"s_per_epoch": sec, "h_per_50_tasks": sec * 400 * 50 / 3600,
                        "max_mem_mb": torch.cuda.max_memory_allocated() / 2 ** 20}
        del B
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    device = H.setup("cuda")
    cifar = RC.Cifar10()
    checks = {"S-stream": lambda: check_stream(cifar, device),
              "S-host": lambda: check_host(cifar, device),
              "S-indep": lambda: check_indep(cifar, device),
              "S-graph": lambda: check_graph(cifar, device),
              "S-eval": lambda: check_eval(cifar, device),
              "S-pp": lambda: check_pp(device),
              "S-freeze": lambda: check_freeze(cifar, device),
              "S-fixconv": lambda: check_fixconv(cifar, device),
              "S-fcamp": lambda: check_fcamp(cifar, device),
              "S-cost": lambda: check_cost(cifar, device)}
    sel = args.only.split(",") if args.only else list(checks)
    res = {}
    for k in sel:
        t = time.time()
        res[k] = checks[k]()
        res[k]["sec"] = time.time() - t
        print(k, json.dumps(res[k], default=str), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    allp = all(v.get("pass", True) for v in res.values())
    res["all_pass"] = allp
    res["git"] = E.git_state()
    (out / ("checks.json" if not args.only else f"checks_{'_'.join(sel)}.json")).write_text(
        json.dumps(res, indent=2, default=str))
    print("ALL_PASS" if allp else "FAIL")


if __name__ == "__main__":
    main()

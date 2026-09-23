#!/usr/bin/env python3
"""Check C10 (追補 1): the engine's sgd_all mode is plain SGD from the first step.

    python3 analysis/sgd_postfit_cifar_0923/check_sgd_all.py --run DIR --eta 0.03 --epochs 2

DIR is an engine run of `--mode sgd_all --eta ETA --tasks 1 --epochs EPOCHS` (R = 10, seeds 0-9).
The reference is written here from the host's streams, per seed, eager, R = 1: the seed's
init (H.init_params), its first rlc_labels draw, one randperm(1200) of rlc_batch per epoch,
batches of 16 in that order, loss = mean CE over the batch, p -= eta * grad on all six tensors.
Engine and reference differ only in the summation order of bmm (R = 10 vs 1), so the check
compares the displacement dP = P_end - P_init per seed: relative error < 1e-3.  Mutation
controls, which must FAIL the same test: the reference with eta * 1.1, and with the biases
left at their init.  Also: m and v in the engine's checkpoint are exactly zero.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import rlcifar_mlp_battle_0918 as B       # noqa: E402
from src import pmnist_0905 as H                   # noqa: E402
from src import pmnist_rlcifar_0907 as RC          # noqa: E402


def reference(cifar, seed, eta, epochs, device, bias_frozen=False):
    X = B.slot_inputs(cifar, seed, "std", device)
    P0 = [q.detach().clone()[None] for q in H.init_params(seed, device, B.DIMS)]
    P = [q.clone() for q in P0]
    y = RC.task_labels(H.stream("rlc_labels", seed)).to(device)
    g_batch = H.stream("rlc_batch", seed)
    act = B.make_act("LR")
    for _ in range(epochs):
        order = torch.randperm(B.N_IMAGES, generator=g_batch).to(device)
        for j in range(B.STEPS_PER_EPOCH):
            idx = order[j * B.BATCH:(j + 1) * B.BATCH]
            P = [q.detach().requires_grad_(True) for q in P]
            *_, z3 = B.forward(P, X[idx][None], act, train=True)
            loss = F.cross_entropy(z3.reshape(-1, B.N_CLASSES), y[idx])
            g = torch.autograd.grad(loss, P)
            with torch.no_grad():
                P = [q - (0 if (bias_frozen and i % 2 == 1) else eta) * gi
                     for i, (q, gi) in enumerate(zip(P, g))]
    return P0, [q.detach() for q in P]


def rel(d_eng, d_ref) -> float:
    num = sum(float(((a - b) ** 2).sum()) for a, b in zip(d_eng, d_ref))
    den = sum(float((b ** 2).sum()) for b in d_ref)
    return (num / den) ** 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--eta", type=float, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    device = H.setup("auto")
    ck = torch.load(Path(a.run) / "ckpt.pt", map_location="cpu", weights_only=False)
    assert ck["t"] == 1, ck["t"]
    cifar = RC.Cifar10()
    res = {"run": a.run, "eta": a.eta, "epochs": a.epochs, "seeds": {}}
    ok = True
    mv_zero = all(float(q.abs().max()) == 0.0 for q in ck["m"] + ck["v"])
    res["m_v_all_zero"] = mv_zero
    ok &= mv_zero
    for r, sl in enumerate(ck["slots"]):
        s = sl["seed"]
        P0, Pr = reference(cifar, s, a.eta, a.epochs, device)
        d_eng = [ck["P"][i][r].to(device) - P0[i][0] for i in range(6)]
        d_ref = [Pr[i][0] - P0[i][0] for i in range(6)]
        e = rel(d_eng, d_ref)
        _, Pm = reference(cifar, s, a.eta * 1.1, a.epochs, device)
        e_eta = rel(d_eng, [Pm[i][0] - P0[i][0] for i in range(6)])
        _, Pb = reference(cifar, s, a.eta, a.epochs, device, bias_frozen=True)
        e_bias = rel(d_eng, [Pb[i][0] - P0[i][0] for i in range(6)])
        passed = e < 1e-3 and e_eta > 1e-2 and e_bias > 1e-2
        ok &= passed
        res["seeds"][s] = {"rel_err": e, "mut_eta1.1_rel_err": e_eta,
                           "mut_bias_frozen_rel_err": e_bias, "ok": passed}
        print(f"seed {s}: rel err {e:.2e}  (mutations: eta x1.1 {e_eta:.2e}, biases frozen "
              f"{e_bias:.2e})  {'ok' if passed else 'FAIL'}", flush=True)
    res["ok"] = ok
    txt = json.dumps(res, indent=2)
    if a.out:
        Path(a.out).write_text(txt + "\n")
    print("C10", "ok" if ok else "FAILED", "| m, v all zero:", mv_zero)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

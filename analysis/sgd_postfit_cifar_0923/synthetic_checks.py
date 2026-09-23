#!/usr/bin/env python3
"""Check C8 of spec_sgd_postfit_cifar_0923 §5: the masked update against an independent per-slot
reference, bit for bit, and six injected bugs that the same comparison must catch.

    python3 analysis/sgd_postfit_cifar_0923/synthetic_checks.py [--device cuda] [--out C8.json]

The reference is written slot by slot and tensor by tensor from the spec's rule (the engine's
Adam op sequence for a slot before its pin; after it: p -= eta g (sgd) or p unchanged (freeze),
m and v held at the pin), on contiguous per-slot copies.  Shapes are the engine's
(3072-100-100-10, R = 4) so the kernels are the engine's too.  post = (F, T, F, T): two slots
switched, two not.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import rlcifar_mlp_battle_0918 as B       # noqa: E402

B1, B2, EPS, LR = 0.9, 0.999, 1e-8, 1e-3
SHAPES = ((100, 3072), (100,), (100, 100), (100,), (10, 100), (10,))


def make(R: int, device, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rnd(*s, scale=1.0):
        return (torch.randn(*s, generator=g) * scale).to(device)

    P = [rnd(R, *s, scale=0.05) for s in SHAPES]
    G = [rnd(R, *s, scale=0.01) for s in SHAPES]
    M = [rnd(R, *s, scale=0.001) for s in SHAPES]
    V = [rnd(R, *s, scale=1e-4).abs() for s in SHAPES]
    Mp = [rnd(R, *s, scale=0.001) for s in SHAPES]
    Vp = [rnd(R, *s, scale=1e-4).abs() for s in SHAPES]
    return P, G, M, V, Mp, Vp


def clone(xs):
    return [x.clone() for x in xs]


def reference(P, G, M, V, Mp, Vp, post, eta, inv_c1, inv_c2):
    """Slot by slot, tensor by tensor, on contiguous copies."""
    P, M, V = clone(P), clone(M), clone(V)
    for r in range(post.shape[0]):
        for i in range(6):
            p, gr, mi, vi = (P[i][r].clone(), G[i][r].clone(), M[i][r].clone(), V[i][r].clone())
            if not bool(post[r]):
                mi.mul_(B1).add_(gr, alpha=1 - B1)
                vi.mul_(B2).addcmul_(gr, gr, value=1 - B2)
                p.sub_(LR * (mi * inv_c1) / ((vi * inv_c2).sqrt() + EPS))
            else:
                if eta is not None:
                    p.sub_(gr * eta)
                mi, vi = Mp[i][r].clone(), Vp[i][r].clone()
            P[i][r].copy_(p)
            M[i][r].copy_(mi)
            V[i][r].copy_(vi)
    return P, M, V


def engine(P, G, M, V, Mp, Vp, post, eta, inv_c1, inv_c2):
    P, M, V = clone(P), clone(M), clone(V)
    B.postfit_update(P, G, M, V, post, Mp, Vp, eta, LR, B1, B2, EPS, inv_c1, inv_c2)
    return P, M, V


def plain_adam(P, G, M, V, inv_c1, inv_c2):
    """The engine's own inline step (the parent's path), on copies."""
    P, M, V = clone(P), clone(M), clone(V)
    for p, gr, mi, vi in zip(P, G, M, V):
        mi.mul_(B1).add_(gr, alpha=1 - B1)
        vi.mul_(B2).addcmul_(gr, gr, value=1 - B2)
        p.sub_(LR * (mi * inv_c1) / ((vi * inv_c2).sqrt() + EPS))
    return P, M, V


def equal(a, b) -> bool:
    return all(torch.equal(x, y) for xs, ys in zip(a, b) for x, y in zip(xs, ys))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.use_deterministic_algorithms(True)
    R = 4
    P, G, M, V, Mp, Vp = make(R, dev)
    inv_c1 = torch.tensor(1.0 / (1 - B1 ** 7), device=dev)       # a step with a real correction
    inv_c2 = torch.tensor(1.0 / (1 - B2 ** 7), device=dev)
    post = torch.tensor([False, True, False, True], device=dev)
    none = torch.zeros(R, dtype=torch.bool, device=dev)
    res = {"device": str(dev), "cases": {}, "mutations": {}}
    for mode, eta in (("freeze", None), ("sgd", 0.03)):
        ref = reference(P, G, M, V, Mp, Vp, post, eta, inv_c1, inv_c2)
        got = engine(P, G, M, V, Mp, Vp, post, eta, inv_c1, inv_c2)
        res["cases"][f"{mode}_vs_reference"] = equal(ref, got)
        # no slot switched: the masked path must be the parent's inline step exactly
        res["cases"][f"{mode}_idle_vs_plain_adam"] = equal(
            plain_adam(P, G, M, V, inv_c1, inv_c2),
            engine(P, G, M, V, Mp, Vp, none, eta, inv_c1, inv_c2))
        # the unswitched slots of the switched run equal the plain step's slots
        pa = plain_adam(P, G, M, V, inv_c1, inv_c2)
        res["cases"][f"{mode}_unswitched_slots_vs_plain"] = all(
            torch.equal(x[r], y[r]) for xs, ys in zip(pa, got) for x, y in zip(xs, ys)
            for r in (0, 2))
        # injected bugs: each must make the same comparison fail
        mut = {}
        mut["mask_shifted_one_slot"] = engine(P, G, M, V, Mp, Vp, post.roll(1), eta,
                                              inv_c1, inv_c2)
        mut["switch_disabled"] = engine(P, G, M, V, Mp, Vp, none, eta, inv_c1, inv_c2)
        # biases take the plain Adam step instead of the mode's
        Pw, Mw, Vw = engine(P, G, M, V, Mp, Vp, post, eta, inv_c1, inv_c2)
        Pa, Ma, Va = plain_adam(P, G, M, V, inv_c1, inv_c2)
        for i in (1, 3, 5):
            Pw[i], Mw[i], Vw[i] = Pa[i], Ma[i], Va[i]
        mut["biases_left_on_adam"] = (Pw, Mw, Vw)
        # moments not pinned: the switched slots keep the Adam moments
        Pm, Mm, Vm = engine(P, G, M, V, Mp, Vp, post, eta, inv_c1, inv_c2)
        for i in range(6):
            for r in (1, 3):
                Mm[i][r], Vm[i][r] = Ma[i][r], Va[i][r]
        mut["moments_not_pinned"] = (Pm, Mm, Vm)
        if mode == "sgd":
            mut["eta_doubled"] = engine(P, G, M, V, Mp, Vp, post, 2 * eta, inv_c1, inv_c2)
            mut["sign_flipped"] = engine(P, G, M, V, Mp, Vp, post, -eta, inv_c1, inv_c2)
        else:
            mut["freeze_takes_a_tiny_sgd_step"] = engine(P, G, M, V, Mp, Vp, post, 1e-6,
                                                         inv_c1, inv_c2)
        res["mutations"][mode] = {k: (not equal(ref, v)) for k, v in mut.items()}
    ok = all(res["cases"].values()) and all(all(m.values()) for m in res["mutations"].values())
    res["ok"] = ok
    txt = json.dumps(res, indent=2)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n")
    if not ok:
        raise SystemExit("C8 failed")


if __name__ == "__main__":
    main()

"""CPU arithmetic and data-contract checks for the image geometry runner."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import effdisp_imagegeom_0924 as E
from src import pmnist_0905 as H


def test_transforms_and_mapped_initial_function():
    g = torch.Generator().manual_seed(92)
    x = torch.rand((3, 3072), generator=g)
    rgb = E.transform(x, "rgb32")
    assert torch.equal(rgb, x)
    dup = E.transform(x, "dup64")
    half = E.transform(x, "dup64_half")
    assert dup.shape == half.shape == (3, 12288)
    assert torch.equal(half, dup * 0.5)
    assert torch.equal(E.transform(torch.ones_like(x), "gray32"), torch.ones(3, 1024))
    assert torch.equal(E.transform(torch.ones_like(x), "avg16"), torch.ones(3, 768))
    w = torch.randn((100, 3072), generator=g)
    for geometry in ("dup64", "dup64_half"):
        z = E.transform(x, geometry) @ E.mapped_first_weight(w, geometry).T
        torch.testing.assert_close(z, x @ w.T, atol=2e-5, rtol=2e-5)
    for geometry in E.GEOMETRIES:
        params = E.initial_params(300, geometry, torch.device("cpu"))
        assert params[0].shape == (100, E.GEOMETRIES[geometry])
        assert torch.equal(params[2], E.initial_params(300, "rgb32", torch.device("cpu"))[2])
        assert torch.equal(params[1], E.initial_params(300, "rgb32", torch.device("cpu"))[1])


def test_mixed_optimizer_step_matches_independent_optimizers():
    g = torch.Generator().manual_seed(7)
    p0 = torch.randn(5, generator=g)
    grad = torch.randn(5, generator=g)
    P = [torch.stack((p0.clone(), p0.clone()))]
    M = [torch.zeros_like(P[0])]
    V = [torch.zeros_like(P[0])]
    E.update_in_place(P, M, V, [torch.stack((grad, grad))],
                      torch.tensor([True, False]), torch.tensor([True, True]), torch.zeros((), dtype=torch.int64))
    a = torch.nn.Parameter(p0.clone())
    s = torch.nn.Parameter(p0.clone())
    adam = torch.optim.Adam([a], lr=E.ADAM_LR, betas=(E.BETA1, E.BETA2), eps=E.EPS)
    sgd = torch.optim.SGD([s], lr=E.SGD_LR)
    a.grad = grad.clone(); s.grad = grad.clone()
    adam.step(); sgd.step()
    torch.testing.assert_close(P[0][0], a, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(P[0][1], s, atol=1e-7, rtol=1e-6)
    assert torch.count_nonzero(M[0][1]) == 0
    assert torch.count_nonzero(V[0][1]) == 0


def test_lr_mapping_first_layer_one_step():
    # Independent arithmetic check, not an extra training arm. With common
    # upstream gradient, the mapped first layer has identical function change
    # under these first-layer-only learning-rate/epsilon substitutions.
    g = torch.Generator().manual_seed(111)
    x = torch.rand((4, 3072), generator=g)
    w = torch.randn((3, 3072), generator=g) * 0.01
    grad = torch.randn((3, 3072), generator=g) * 0.01
    lr, eps = 0.01, 1e-8
    for geometry, scale in (("dup64", 1.0), ("dup64_half", 0.5)):
        q = 4
        xx = E.transform(x, geometry)
        ww = E.mapped_first_weight(w, geometry)
        gg = scale * E.transform(grad, "dup64")
        # SGD: eta' = eta/(q*s^2)
        w_next = w - lr * grad
        ww_next = ww - (lr / (q * scale * scale)) * gg
        torch.testing.assert_close(xx @ ww_next.T, x @ w_next.T, atol=2e-5, rtol=2e-5)
        # First Adam step uses sign(g)/(1+eps/|g|).
        adam_delta = lr * grad / (grad.abs() + eps)
        dup_delta = (lr / (q * scale)) * gg / (gg.abs() + scale * eps)
        torch.testing.assert_close(xx @ (ww - dup_delta).T, x @ (w - adam_delta).T,
                                   atol=2e-5, rtol=2e-5)


def test_refuse_nonempty_and_snapshot_contract(tmp_path):
    P = [torch.ones((1, 2, 3)), torch.zeros((1, 2))]
    M = [torch.zeros_like(q) for q in P]
    V = [torch.zeros_like(q) for q in P]
    # This miniature uses the first two parameter names; the serializer should
    # still preserve tensor shapes and encode SGD's absent moments as scalars.
    f = tmp_path / "t000.npz"
    E.save_snapshot(f, P, M, V, 0, 0, 0, "sgd")
    with np.load(f) as d:
        assert d["W1"].shape == (2, 3)
        assert d["adam_m_W1"].shape == ()
        assert int(d["task"]) == 0
    with pytest.raises(FileExistsError):
        E.refuse_nonempty(tmp_path)
    E.refuse_nonempty(tmp_path / "new")


def test_host_rng_streams_shared_across_arms():
    a, b = H.stream("rlc_batch", 300), H.stream("rlc_batch", 300)
    for _ in range(3):
        assert torch.equal(torch.randperm(E.N_IMAGES, generator=a),
                           torch.randperm(E.N_IMAGES, generator=b))

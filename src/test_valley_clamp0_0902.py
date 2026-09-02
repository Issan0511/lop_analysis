"""valley_clamp0_0902: 床ゼロの谷埋め（silu_clamp0 / gelu_clamp0）の単体検査。"""
import pytest
import torch

from .nets import VecMLPL


def _net(act, beta):
    return VecMLPL(1, [2], 2, torch.Generator().manual_seed(0), "cpu").set_activation(act, beta, "alpha_exp")


@pytest.mark.parametrize("base", ["silu", "gelu"])
@pytest.mark.parametrize("beta", [0.3, 1.0, 3.0])
def test_clamp0_matches_base_above_valley_and_is_zero_below(base, beta):
    z = torch.linspace(-30, 10, 200001, dtype=torch.float64)
    n0, n1, nc = _net(base, beta), _net(base + "_clamp0", beta), _net(base + "_clamp", beta)
    f0, f1, fc = n0.act_fn(z), n1.act_fn(z), nc.act_fn(z)
    g0, g1, gc = n0.act_grad(z, f0), n1.act_grad(z, f1), nc.act_grad(z, fc)
    zc = -VecMLPL.VALLEY_ZERO[base + "_clamp0"] / beta
    above = z > zc
    assert torch.equal(f0[above], f1[above]) and torch.equal(g0[above], g1[above])   # 谷の手前は元と bit 一致
    assert torch.equal(fc[above], f1[above]) and torch.equal(gc[above], g1[above])   # clamp 版とも bit 一致
    assert (f1[~above] == 0).all() and (g1[~above] == 0).all()                       # 谷の向こうは出力も勾配も厳密 0
    assert (fc[~above] != 0).all()                                                    # clamp 版の床は非ゼロ（差はここだけ）
    assert (g1 >= 0).all()


@pytest.mark.parametrize("base", ["silu", "gelu"])
def test_clamp0_gradient_matches_finite_difference(base):
    n1 = _net(base + "_clamp0", 1.0)
    z = torch.linspace(-30, 10, 40001, dtype=torch.float64)
    h = 1e-6
    fd = (n1.act_fn(z + h) - n1.act_fn(z - h)) / (2 * h)
    zc = -VecMLPL.VALLEY_ZERO[base + "_clamp0"]
    m = (z - zc).abs() > 1e-3
    assert float((fd[m] - n1.act_grad(z, n1.act_fn(z))[m]).abs().max()) < 1e-7


def test_readout_gradient_vanishes_for_floored_units():
    # 出力 0 なら dL/dv = 2*delta*a = 0（ReLU の死ユニットと同型）。clamp 版は非ゼロ。
    z = torch.tensor([-5.0, -2.0, -0.9], dtype=torch.float64)
    assert (_net("gelu_clamp0", 1.0).act_fn(z) == 0).all()
    assert (_net("gelu_clamp", 1.0).act_fn(z) != 0).all()

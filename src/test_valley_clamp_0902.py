"""valley_clamp_0902: 谷埋め活性化（silu_clamp / gelu_clamp）の単体検査。"""
import math

import pytest
import torch

from .nets import VecMLPL


def _net(act, beta):
    return VecMLPL(1, [2], 2, torch.Generator().manual_seed(0), "cpu").set_activation(act, beta, "alpha_exp")


@pytest.mark.parametrize("base", ["silu", "gelu"])
@pytest.mark.parametrize("beta", [0.3, 1.0, 3.0])
def test_clamp_matches_base_above_valley_and_is_flat_below(base, beta):
    z = torch.linspace(-30, 10, 200001, dtype=torch.float64)
    n0, n1 = _net(base, beta), _net(base + "_clamp", beta)
    f0, f1 = n0.act_fn(z), n1.act_fn(z)
    g0, g1 = n0.act_grad(z, f0), n1.act_grad(z, f1)
    zc = -VecMLPL.VALLEY_ZERO[base + "_clamp"] / beta
    above = z > zc
    assert torch.equal(f0[above], f1[above])          # 谷の手前は bit 一致
    assert torch.equal(g0[above], g1[above])
    assert (g1[~above] == 0).all()                    # 谷の向こうは phi' = 0
    assert (f1[~above] == f1[~above][0]).all()        # phi は定数
    assert (g1 >= 0).all() and (f1[1:] >= f1[:-1]).all()   # 単調・非負勾配


@pytest.mark.parametrize("base", ["silu", "gelu"])
def test_clamp_point_is_the_first_negative_zero_of_phi_prime(base):
    z = torch.linspace(-6, 0, 600001, dtype=torch.float64)
    n0 = _net(base, 1.0)
    g = n0.act_grad(z, n0.act_fn(z))
    zc = -VecMLPL.VALLEY_ZERO[base + "_clamp"]
    i = int((z - zc).abs().argmin())
    assert abs(float(g[i])) < 1e-4
    assert (g[i + 100:] > 0).all()                    # 零点より浅い側は正


@pytest.mark.parametrize("base", ["silu", "gelu"])
def test_clamp_gradient_matches_finite_difference(base):
    beta = 1.0
    n1 = _net(base + "_clamp", beta)
    z = torch.linspace(-30, 10, 40001, dtype=torch.float64)
    h = 1e-6
    fd = (n1.act_fn(z + h) - n1.act_fn(z - h)) / (2 * h)
    zc = -VecMLPL.VALLEY_ZERO[base + "_clamp"] / beta
    m = (z - zc).abs() > 1e-3                          # 折れ目の直近だけ除く
    g = n1.act_grad(z, n1.act_fn(z))
    assert float((fd[m] - g[m]).abs().max()) < 1e-7


def test_set_activation_validates_beta():
    with pytest.raises(ValueError):
        _net("gelu_clamp", 0.0)
    with pytest.raises(ValueError):
        _net("silu_clamp", math.inf)
    assert _net("gelu_clamp", 2.0).act == "gelu_clamp"


def test_existing_activation_paths_untouched():
    z = torch.linspace(-5, 5, 1001, dtype=torch.float64)
    n = _net("relu", 1.0)
    assert torch.equal(n.act_fn(z), torch.relu(z))
    n = _net("gelu", 1.0)
    assert torch.allclose(n.act_fn(z), torch.nn.functional.gelu(z), atol=1e-12)

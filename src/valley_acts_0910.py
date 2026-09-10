"""GELU / SiLU for the pmnist harness (box B), duck-typed to match H.Activation.

Both are "valley" activations: phi has a minimum at z_c < 0 and phi' < 0 beyond it.
That sign is the whole point of the arm — on the far side of the valley the loss's
"activate more" request (e < 0) pushes the row mean DOWN instead of up, so the
restoring regime that leaky/ELU keep at every depth does not exist here.

  GELU (exact, erf form):  phi = z*Phi(z),      phi' = Phi(z) + z*p(z)
  SiLU (beta=1):           phi = z*sigmoid(z),  phi' = s(1 + z(1-s))

Valley bottoms (phi'=0), solved numerically once and asserted in _selftest:
  GELU  z_c = -0.751791524...   phi(z_c) = -0.169484...
  SiLU  z_c = -1.278464543...   phi(z_c) = -0.278464...
"""
import math
import torch

SQRT2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


class GELU:
    """Exact erf GELU.  phi(z) = z * Phi(z)."""
    kind = 'gelu'

    def __init__(self, param=1.0):
        self.param = float(param); self.name = 'GELU'

    def phi(self, z):
        return z * 0.5 * (1.0 + torch.erf(z / SQRT2))

    def dphi(self, z):
        cdf = 0.5 * (1.0 + torch.erf(z / SQRT2))
        pdf = INV_SQRT_2PI * torch.exp(-0.5 * z * z)
        return cdf + z * pdf


class SiLU:
    """SiLU / swish-1.  phi(z) = z * sigmoid(beta z), beta = param."""
    kind = 'silu'

    def __init__(self, param=1.0):
        self.param = float(param); self.name = 'SiLU'

    def phi(self, z):
        return z * torch.sigmoid(self.param * z)

    def dphi(self, z):
        s = torch.sigmoid(self.param * z)
        return s * (1.0 + self.param * z * (1.0 - s))


VALLEY = {'GELU': GELU, 'SILU': SiLU}
# z_c: phi'(z_c) = 0 with z_c < 0 (the valley bottom); phi is decreasing on (z_c, 0)
#      and phi' < 0 for z < z_c, i.e. deeper => LESS negative activation.
ZC = {'GELU': -0.7517915239, 'SILU': -1.2784645428}


def make_valley(arm):
    return VALLEY[arm]() if arm in VALLEY else None


def _selftest():
    """Checks that can fail: the valley bottom, the sign of phi' on both sides,
    monotonicity of phi' near z_c, and agreement with torch's own kernels."""
    z = torch.linspace(-12, 6, 200001, dtype=torch.float64)
    out = {}
    for name, cls, ref in [('GELU', GELU, lambda t: torch.nn.functional.gelu(t)),
                           ('SILU', SiLU, lambda t: torch.nn.functional.silu(t))]:
        a = cls()
        # 1. phi matches torch's kernel
        dphi_max = float((a.phi(z) - ref(z)).abs().max())
        # 2. dphi matches autograd
        zg = z.clone().requires_grad_(True)
        g = torch.autograd.grad(a.phi(zg).sum(), zg)[0]
        dgrad = float((a.dphi(z) - g).abs().max())
        # 3. the valley bottom: dphi(zc) == 0 and it is the unique negative root
        zc = torch.tensor(ZC[name], dtype=torch.float64)
        at_zc = float(a.dphi(zc).abs())
        # 4. SIGN STRUCTURE: dphi > 0 for z > zc+eps, dphi < 0 for z < zc-eps (down to where it underflows)
        eps = 1e-3
        right = z[(z > zc + eps)]
        left = z[(z < zc - eps) & (z > -8.0)]        # below -8 GELU's dphi underflows to -0.0
        pos_right = bool((a.dphi(right) > 0).all())
        neg_left = bool((a.dphi(left) < 0).all())
        # 5. the minimum of phi is AT zc
        i = int(a.phi(z).argmin())
        zmin = float(z[i])
        out[name] = dict(phi_vs_torch=dphi_max, dphi_vs_autograd=dgrad, dphi_at_zc=at_zc,
                         dphi_pos_right=pos_right, dphi_neg_left=neg_left,
                         argmin_phi=zmin, zc=float(zc), phi_zc=float(a.phi(zc)))
        assert dphi_max < 1e-6, (name, 'phi disagrees with torch', dphi_max)
        assert dgrad < 1e-9, (name, 'dphi disagrees with autograd', dgrad)
        assert at_zc < 1e-8, (name, 'zc is not a root of dphi', at_zc)
        assert pos_right and neg_left, (name, 'sign structure wrong', pos_right, neg_left)
        assert abs(zmin - float(zc)) < 1e-3, (name, 'phi min is not at zc', zmin)
        # mutation control: a wrong zc must break check 3
        bad = float(a.dphi(zc + 0.05).abs())
        assert bad > 1e-3, (name, 'zc check is vacuous', bad)
    return out


if __name__ == '__main__':
    import json
    print(json.dumps(_selftest(), indent=1))

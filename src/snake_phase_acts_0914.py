"""snake_phase_mnist_0914: phase-shifted Snake, offset Snake / leaky, and the bias init maps
(spec_snake_phase_mnist_0914 §3-§4).  Host modules are imported, never edited.

Fixed alpha (theta code th in {0, +1, -1} = {0, +pi/2, -pi/2}):
    normal  z + sin(a z)^2 / a            (op order identical to host 'snake' -> SN06 bit-identical)
    peak    z + sin(2 a z) / (2 a)        phi'(0) = 2
    valley  z - sin(2 a z) / (2 a)        phi'(0) = 0
    + q only when q != 0.
Adaptive (AdaptivePhaseSnake inherits GS.H.AdaptiveSnake, the class the box-B loop and
C.measure isinstance-check; th = 0 delegates to the host methods, so SNA is bit-identical).

Identity (spec §3): psi_th(z) = psi_0(z + s + k pi/a) + K - k pi/a,
    s = th_rad/(2a) + k pi/a,  K = (cos th_rad - 1 - th_rad)/(2a) - k pi/a.
Init maps (float64, cast once):
    comp(q):         b2 -= q W2 1,  b3 -= q W3 1
    to_phase(s, K):  b1 += s,  b2 += s + K W2 1,  b3 += K W3 1
"""
import math
from pathlib import Path

import torch
import yaml

from src import width_sink_clamp_0909 as C

H = C.H
ROOT = C.ROOT
CONFIG = ROOT / 'configs/snake_phase_mnist_0914.yaml'
ALPHA = 0.6
TH_RAD = {0: 0.0, 1: math.pi / 2, -1: -math.pi / 2}


def phase_constants(th, k=0, a=ALPHA):
    t = TH_RAD[th]
    return t / (2 * a) + k * math.pi / a, (math.cos(t) - 1 - t) / (2 * a) - k * math.pi / a


S_P, K_P = phase_constants(1)
S_P1, K_P1 = phase_constants(1, -1)
S_V, K_V = phase_constants(-1)


class PhaseSnake:
    kind = 'phase_snake'

    def __init__(self, th, a=ALPHA, q=0.0):
        assert th in (0, 1, -1)
        self.th, self.param, self.q = th, float(a), float(q)
        self.name = f'PS{th:+d}_a{a}_q{q}'

    def phi(self, z):
        a = self.param
        if self.th == 0:
            y = z + torch.sin(a * z) ** 2 / a
        elif self.th == 1:
            y = z + torch.sin(2.0 * a * z) / (2.0 * a)
        else:
            y = z - torch.sin(2.0 * a * z) / (2.0 * a)
        if self.q != 0.0:
            y = y + self.q
        return y

    def dphi(self, z):
        a = self.param
        if self.th == 0:
            return 1.0 + torch.sin(2.0 * a * z)
        if self.th == 1:
            return 1.0 + torch.cos(2.0 * a * z)
        return 1.0 - torch.cos(2.0 * a * z)


class OffsetLeaky:
    kind = 'offset_leaky'

    def __init__(self, a=0.1, q=0.0):
        self.param, self.q = float(a), float(q)
        self.name = f'OL_a{a}_q{q}'

    def phi(self, z):
        y = torch.where(z > 0, z, self.param * z)
        if self.q != 0.0:
            y = y + self.q
        return y

    def dphi(self, z):
        return torch.where(z > 0, torch.ones_like(z), torch.full_like(z, self.param))


class AdaptivePhaseSnake(H.AdaptiveSnake):
    kind = 'adaptive_phase_snake'

    def __init__(self, th, c=0.6, beta=0.01, device='cpu'):
        super().__init__(c, beta, device)
        assert th in (0, 1, -1)
        self.th = th
        self.name = f'APS{th:+d}'

    def phi(self, z, layer=0):
        if self.th == 0:
            return super().phi(z, layer)
        a = self.alpha(layer)
        y = torch.sin(2.0 * a * z) * (0.5 * a.reciprocal())
        return z + y if self.th == 1 else z - y

    def dphi(self, z, layer=0):
        if self.th == 0:
            return super().dphi(z, layer)
        c = torch.cos(2.0 * self.alpha(layer) * z)
        return 1.0 + c if self.th == 1 else 1.0 - c


# arm -> (activation factory, init map)
ARM_TABLE = {
    'N06':     (lambda: PhaseSnake(0), None),
    'P06':     (lambda: PhaseSnake(1), None),
    'V06':     (lambda: PhaseSnake(-1), None),
    'P06c':    (lambda: PhaseSnake(0, q=K_P), ('comp', K_P)),
    'P06c_k1': (lambda: PhaseSnake(0, q=K_P1), ('comp', K_P1)),
    'P06i':    (lambda: PhaseSnake(0), ('to_phase', S_P, K_P)),
    'V06c':    (lambda: PhaseSnake(0, q=K_V), ('comp', K_V)),
    'V06i':    (lambda: PhaseSnake(0), ('to_phase', S_V, K_V)),
    'SNA':     (lambda: AdaptivePhaseSnake(0), None),
    'SNAP':    (lambda: AdaptivePhaseSnake(1), None),
    'SNAV':    (lambda: AdaptivePhaseSnake(-1), None),
    'SNAi_P':  (lambda: AdaptivePhaseSnake(0), ('to_phase', S_P, K_P)),
    'SNAi_V':  (lambda: AdaptivePhaseSnake(0), ('to_phase', S_V, K_V)),
    'LIN':     (lambda: H.Activation('LINx', 'linear'), None),
    'LR':      (lambda: OffsetLeaky(0.1), None),
    'LR_qKp':  (lambda: OffsetLeaky(0.1, K_P), ('comp', K_P)),
    'LR_qKpn': (lambda: OffsetLeaky(0.1, -K_P), ('comp', -K_P)),
}
ARMS = list(ARM_TABLE)
ANCHOR = {'N06': 'SN06', 'SNA': 'SNA', 'LR': 'LR', 'LIN': 'LIN'}


def make_act(arm):
    return ARM_TABLE[arm][0]()


def apply_init_map(p, spec, mutate=None):
    """In place on a host param list [W1,b1,W2,b2,W3,b3]; all arithmetic in float64, one cast.
    mutate: 'no_b3' (drop the b3 term), 'double' (apply twice) -- check mutations only."""
    if spec is None:
        return p
    W2 = p[2].detach().double(); W3 = p[4].detach().double()
    b = [p[1].detach().double(), p[3].detach().double(), p[5].detach().double()]
    reps = 2 if mutate == 'double' else 1
    for _ in range(reps):
        if spec[0] == 'comp':
            q = spec[1]
            b[1] = b[1] - q * W2.sum(1)
            if mutate != 'no_b3':
                b[2] = b[2] - q * W3.sum(1)
        elif spec[0] == 'to_phase':
            s, K = spec[1], spec[2]
            b[0] = b[0] + s
            b[1] = b[1] + s + K * W2.sum(1)
            if mutate != 'no_b3':
                b[2] = b[2] + K * W3.sum(1)
        else:
            raise ValueError(spec)
    with torch.no_grad():
        for i, j in zip((1, 3, 5), range(3)):
            p[i].data.copy_(b[j].to(torch.float32))
    return p


def check_config_constants():
    cfg = yaml.safe_load(open(CONFIG))
    c = cfg['constants']
    got = {'peak_k0': (S_P, K_P), 'peak_km1': (S_P1, K_P1), 'valley_k0': (S_V, K_V)}
    for k, (s, K) in got.items():
        assert c[k]['s'] == s and c[k]['K'] == K, (k, c[k], s, K)
    assert c['pi_over_alpha'] == math.pi / ALPHA
    for arm, spec in cfg['arms'].items():
        assert arm in ARM_TABLE, arm
        q = spec.get('q', 0.0)
        act = make_act(arm)
        if hasattr(act, 'q'):
            assert float(q) == act.q, (arm, q, act.q)
    assert set(cfg['arms']) == set(ARM_TABLE)
    return True

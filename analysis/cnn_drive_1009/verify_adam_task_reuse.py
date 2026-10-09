"""Exact outward rational-interval certificate for taskwise label-reuse Adam.

No floating point is used in the certificate. All four iid assignments for a
fresh task are reused for three optimizer steps, with full moments retained
from a two-step previous task. The shared ReLU/MaxPool CNN has ten free raw
parameters, all updated by ordinary Adam. Optional --torch independently
checks the real convolution/maxpool/autograd implementation in float64.
"""
import argparse
import copy
from fractions import Fraction as F
import itertools
import json
import math

SCALE = 10**35


def floor_grid(x):
    return F((x.numerator * SCALE) // x.denominator, SCALE)


def ceil_grid(x):
    return -floor_grid(-x)


class I:
    def __init__(self, lo, hi=None):
        self.lo = F(lo)
        self.hi = F(lo if hi is None else hi)
        assert self.lo <= self.hi

    @staticmethod
    def box(lo, hi):
        return I(floor_grid(lo), ceil_grid(hi))

    @staticmethod
    def cast(x):
        return x if isinstance(x, I) else I(x)

    def __add__(self, other):
        o = I.cast(other)
        return I.box(self.lo + o.lo, self.hi + o.hi)

    __radd__ = __add__

    def __neg__(self):
        return I(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + (-I.cast(other))

    def __rsub__(self, other):
        return I.cast(other) + (-self)

    def __mul__(self, other):
        o = I.cast(other)
        q = [self.lo * o.lo, self.lo * o.hi,
             self.hi * o.lo, self.hi * o.hi]
        return I.box(min(q), max(q))

    __rmul__ = __mul__

    def __truediv__(self, other):
        o = I.cast(other)
        assert o.lo * o.hi > 0, (o.lo, o.hi)
        return self * I.box(1 / o.hi, 1 / o.lo)

    def __rtruediv__(self, other):
        return I.cast(other) / self

    def square(self):
        lo = 0 if self.lo <= 0 <= self.hi else min(self.lo**2, self.hi**2)
        return I.box(lo, max(self.lo**2, self.hi**2))

    def sqrt(self):
        assert self.lo >= 0
        def lower(x):
            return F(math.isqrt((x.numerator * SCALE**2) // x.denominator), SCALE)
        lo, hi = lower(self.lo), lower(self.hi)
        if hi * hi < self.hi:
            hi += F(1, SCALE)
        assert lo * lo <= self.lo and hi * hi >= self.hi
        return I(lo, hi)

    def exp(self):
        def endpoint(x):
            if x < 0:
                pos = endpoint(-x)
                return 1 / pos
            assert x <= 2
            term = total = F(1)
            N = 70
            for k in range(1, N + 1):
                term *= x / k
                total += term
            first_omitted = term * x / (N + 1)
            tail_upper = first_omitted / (1 - x / (N + 2))
            return I.box(total, total + tail_upper)
        return I(endpoint(self.lo).lo, endpoint(self.hi).hi)

    def abs_upper(self):
        return max(abs(self.lo), abs(self.hi))

    def json(self):
        return {'lower_rational': str(self.lo), 'upper_rational': str(self.hi),
                'lower_decimal': float(self.lo), 'upper_decimal': float(self.hi)}


BETA, BETA2 = F(9, 10), F(999, 1000)
EPS, ETA = F(1, 10**8), F(1, 10**5)
MU, K_BOUND, L_BOUND = F(9, 8), F(73), F(10)
POOLED = [[F(1), F(2)], [F(2), F(1)]]
H = 3


def initial_state():
    theta = list(map(I, [F(7, 10), F(7, 10),
                         F(1, 10), F(1, 20), F(1, 10), F(1, 20),
                         -F(1, 10), -F(1, 20), -F(1, 10), -F(1, 20)]))
    return {'theta': theta, 'm': [I(0) for _ in theta],
            'v': [I(0) for _ in theta], 't': 0}


def vi(c, j, s):
    return 2 + 4 * c + 2 * j + s


def gradient(theta, labels):
    result = [I(0) for _ in theta]
    for n, x in enumerate(POOLED):
        logits = [sum(theta[vi(c, j, s)] * theta[j] * x[s]
                      for j in range(2) for s in range(2)) for c in range(2)]
        exp_difference = (logits[0] - logits[1]).exp()
        p0 = exp_difference / (1 + exp_difference)
        r0 = p0 - int(labels[n] == 0)
        residuals = [r0, -r0]
        for c in range(2):
            for j in range(2):
                for s in range(2):
                    result[j] += residuals[c] * theta[vi(c, j, s)] * x[s] / 2
                    result[vi(c, j, s)] += residuals[c] * theta[j] * x[s] / 2
    return result


def update_moments(state, grad, update_parameters):
    state['t'] += 1
    t = state['t']
    q, sigma = [], []
    for j in range(10):
        state['m'][j] = BETA * state['m'][j] + (1 - BETA) * grad[j]
        state['v'][j] = BETA2 * state['v'][j] + (1 - BETA2) * grad[j].square()
        sig = (state['v'][j] / (1 - BETA2**t)).sqrt()
        direction = (state['m'][j] / (1 - BETA**t)) / (sig + EPS)
        sigma.append(sig)
        q.append(direction)
        if update_parameters:
            state['theta'][j] -= ETA * direction
    return q, sigma


def certify():
    # All constants below are checked by exact rational comparisons.
    assert K_BOUND**2 > 1 / ((1 - BETA2) * (1 - BETA**2 / BETA2))
    assert 5 * ETA * K_BOUND < F(1, 100)
    W, V = F(73, 100), F(16, 100)
    max_jacobian_column_l1 = max(2 * V * 3, W * 2)
    max_jacobian_row_l1 = 2 * (V + W) * 3
    hessian_bound = max_jacobian_column_l1 * max_jacobian_row_l1 / 2 + 6
    assert hessian_bound < L_BOUND
    state = initial_state()
    for _ in range(2):
        update_moments(state, gradient(state['theta'], [0, 1]), True)
    boundary_state = copy.deepcopy(state)
    records, reference_sum, error_sum, actual_sum = [], I(0), F(0), I(0)
    for labels in itertools.product(range(2), repeat=2):
        frozen = copy.deepcopy(boundary_state)
        actual = copy.deepcopy(boundary_state)
        g0 = gradient(boundary_state['theta'], labels)
        ref, error, first_moments = I(0), F(0), []
        for r in range(1, H + 1):
            q, sigma = update_moments(frozen, g0, False)
            ref += ETA * MU * q[0]
            errors = [L_BOUND * K_BOUND * ETA * (k - 1) for k in range(1, r + 1)]
            A = sum((1 - BETA) * BETA**(r - k) * errors[k - 1]
                    for k in range(1, r + 1)) / (1 - BETA**(2 + r))
            C_squared = sum((1 - BETA2) * BETA2**(r - k) * errors[k - 1]**2
                            for k in range(1, r + 1)) / (1 - BETA2**(2 + r))
            C = I(C_squared).sqrt().hi
            denominator_floor = max(F(0), sigma[0].lo - C) + EPS
            error += ETA * MU * (A + q[0].abs_upper() * C) / denominator_floor
            update_moments(actual, gradient(actual['theta'], labels), True)
            first_moments.append(actual['m'][0].json())
        sink = MU * (boundary_state['theta'][0] - actual['theta'][0])
        assert sink.lo >= ref.lo - error
        record = {'labels': list(labels), 'boundary_gradient': g0[0].json(),
                  'frozen_reference_sink': ref.json(),
                  'movement_error_upper_rational': str(error),
                  'movement_error_upper_decimal': float(error),
                  'actual_task_sink': sink.json(), 'actual_first_moments': first_moments}
        records.append(record)
        reference_sum += ref / 4
        error_sum += error / 4
        actual_sum += sink / 4
    certified_lower = reference_sum.lo - error_sum
    assert reference_sum.lo > F(215, 10**7)  # 2.15e-5
    assert error_sum < F(184, 10**8)  # 1.84e-6
    assert certified_lower > F(1966, 10**8)  # 1.966e-5
    assert actual_sum.lo > 0
    assert F(records[0]['actual_task_sink']['upper_rational']) < 0
    return {
        'scope': 'Conditional second-task certificate, iid labels reused within each task',
        'arithmetic': 'Outward rational intervals; 35 decimal places after each operation',
        'architecture': 'Shared Conv1x1 1->2, ReLU, MaxPool(1x2), free 2-class spatial head',
        'input_images': [[[1, .5, 2, 1]], [[2, 1, 1, .5]]],
        'raw_parameter_count': 10, 'all_parameters_updated': True,
        'biases': False, 'batch_size': 2, 'eta': str(ETA),
        'beta1': str(BETA), 'beta2': str(BETA2), 'epsilon': str(EPS),
        'previous_task_labels': [0, 1], 'previous_task_probability': '1/4',
        'previous_task_steps': 2, 'fresh_task_steps': H,
        'fresh_assignment_probability_each': '1/4',
        'moments_reset_at_task_boundary': False,
        'K_bound': str(K_BOUND), 'gradient_lipschitz_infinity_bound': str(L_BOUND),
        'derived_hessian_row_sum_upper': str(hessian_bound),
        'boundary_target_m': boundary_state['m'][0].json(),
        'boundary_target_v': boundary_state['v'][0].json(),
        'records': records,
        'conditional_frozen_mean': reference_sum.json(),
        'conditional_movement_error_upper_rational': str(error_sum),
        'conditional_movement_error_upper_decimal': float(error_sum),
        'certified_conditional_actual_mean_lower_rational': str(certified_lower),
        'certified_conditional_actual_mean_lower_decimal': float(certified_lower),
        'direct_interval_conditional_actual_mean': actual_sum.json(),
        'assertions_passed': True,
    }


def torch_check():
    import torch
    import torch.nn.functional as fn
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(100926)
    x = torch.tensor([[[[1., .5, 2., 1.]]], [[[2., 1., 1., .5]]]])
    conv = torch.nn.Conv2d(1, 2, 1, bias=False)
    head = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        conv.weight.fill_(.7)
        head.weight.copy_(torch.tensor([[.1, .05, .1, .05], [-.1, -.05, -.1, -.05]]))
    opt = torch.optim.Adam(list(conv.parameters()) + list(head.parameters()),
                           lr=float(ETA), betas=(float(BETA), float(BETA2)), eps=float(EPS))

    def step(labels):
        opt.zero_grad()
        logits = head(fn.max_pool2d(fn.relu(conv(x)), (1, 2)).flatten(1))
        fn.cross_entropy(logits, torch.tensor(labels)).backward()
        opt.step()

    for _ in range(2):
        step([0, 1])
    conv_state, head_state = copy.deepcopy(conv.state_dict()), copy.deepcopy(head.state_dict())
    opt_state = copy.deepcopy(opt.state_dict())
    mean0 = float(x.mean()) * float(conv.weight.detach()[0, 0, 0, 0])
    records = []
    for labels in itertools.product(range(2), repeat=2):
        conv.load_state_dict(conv_state)
        head.load_state_dict(head_state)
        opt.load_state_dict(copy.deepcopy(opt_state))
        for _ in range(H):
            step(labels)
        mean = float(x.mean()) * float(conv.weight.detach()[0, 0, 0, 0])
        records.append({'labels': list(labels), 'actual_mean_decrease': mean0 - mean})
    return {'torch_version': torch.__version__, 'dtype': 'float64',
            'records': records, 'conditional_mean': sum(r['actual_mean_decrease'] for r in records) / 4}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch', action='store_true')
    args = parser.parse_args()
    result = certify()
    if args.torch:
        result['torch_autograd_check'] = torch_check()
        for exact, numeric in zip(result['records'], result['torch_autograd_check']['records']):
            assert exact['labels'] == numeric['labels']
            assert abs(float(F(exact['actual_task_sink']['lower_rational'])) -
                       numeric['actual_mean_decrease']) < 1e-12
    print(json.dumps(result, indent=2, sort_keys=True))

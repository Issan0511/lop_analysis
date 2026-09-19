"""Compare compiled/fused arithmetic against upstream eager without changing seeds."""
import json
import time

import torch
from torch.nn import functional as F

from .checks import assert_tree_equal
from .model import ARM_ORDER, activations, make_model, prepare_noise, train_forward, update_adaptive
from .run import RAW, REPO, atomic_json, get_rng, set_rng


def relative_error(a, b):
    return float((a.detach() - b.detach()).double().norm() / a.detach().double().norm().clamp_min(1e-30))


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    rows = []
    for arm in ARM_ORDER:
        eager = make_model(arm, 100)
        fast = make_model(arm, 100)
        fn = train_forward(fast, 'compile')
        opt_e = torch.optim.Adam(eager.parameters(), lr=1e-4, foreach=True)
        opt_f = torch.optim.Adam(fast.parameters(), lr=1e-4, fused=True)
        x = torch.randn(128, 3, 64, 64, device='cuda')
        y = torch.arange(128, device='cuda') % 5
        rng = get_rng()
        prepare_noise(eager, 128)
        a = eager(x)
        F.cross_entropy(a[:, :5], y).backward()
        after_eager_rng = get_rng()
        set_rng(rng)
        prepare_noise(fast, 128)
        b = fn(x)
        F.cross_entropy(b[:, :5], y).backward()
        assert_tree_equal(after_eager_rng, get_rng())
        grad_error = max(relative_error(p.grad, q.grad) for p, q in zip(eager.parameters(), fast.parameters()))
        grad_details = [dict(name=name, relative_l2=relative_error(p.grad, q.grad),
                             norm=float(p.grad.double().norm()), max_abs=float((p.grad-q.grad).abs().max()))
                        for (name, p), q in zip(eager.named_parameters(), fast.parameters())]
        total_grad_error = relative_error(torch.cat([p.grad.flatten() for p in eager.parameters()]),
                                          torch.cat([p.grad.flatten() for p in fast.parameters()]))
        out_error = relative_error(a, b)
        # Isolate fused Adam from the forward's rounding differences. Attention key
        # bias has analytically zero gradient; Adam amplifies tiny cancellation
        # errors there, so relative error of that near-zero parameter is misleading.
        same_grad_params = [torch.nn.Parameter(p.detach().clone()) for p in eager.parameters()]
        for p, original in zip(same_grad_params, eager.parameters()):
            p.grad = original.grad.detach().clone()
        same_grad_opt = torch.optim.Adam(same_grad_params, lr=1e-4, fused=True)
        # First-step Adam is lr*g/(|g|+eps), with Lipschitz constant lr/eps.
        bounds = [(1e-4 / 1e-8) * (p.grad - q.grad).abs() +
                  256 * torch.finfo(torch.float32).eps * (p.detach().abs() + 1e-4)
                  for p, q in zip(eager.parameters(), fast.parameters())]
        opt_e.step()
        opt_f.step()
        same_grad_opt.step()
        update_adaptive(eager)
        update_adaptive(fast)
        param_error = max(relative_error(p, q) for p, q in zip(eager.parameters(), fast.parameters()))
        same_grad_error = max(relative_error(p, q) for p, q in zip(eager.parameters(), same_grad_params))
        update_bound_ratio = max(float(((p-q).detach().abs() / bound).max())
                                 for p, q, bound in zip(eager.parameters(), fast.parameters(), bounds))
        param_details = [dict(name=name, relative_l2=relative_error(p, q),
                              norm=float(p.detach().double().norm()), max_abs=float((p-q).detach().abs().max()))
                         for (name, p), q in zip(eager.named_parameters(), fast.parameters())]
        v_errors = [relative_error(e.V, f.V) for e, f in zip(activations(eager), activations(fast)) if e.V is not None]
        # The door's running DC is this experiment's EMA and gets the same treatment as V.
        v_errors += [relative_error(e.M, f.M) for e, f in zip(activations(eager), activations(fast)) if e.M is not None]
        # RMS relative comparison: 256 binary32 epsilon for a 6-block composed forward/backward.
        # This is a numerical consistency threshold, not evidence of identical long trajectories.
        tolerance = 256 * torch.finfo(torch.float32).eps
        row = dict(arm=arm, logits_relative_l2=out_error, max_gradient_relative_l2=grad_error,
                   max_parameter_relative_l2=param_error, max_ema_relative_l2=max(v_errors, default=0.),
                   rng_bit_equal=True, tolerance=tolerance)
        row['total_gradient_relative_l2'] = total_grad_error
        row['same_gradient_fused_adam_relative_l2'] = same_grad_error
        row['adam_lipschitz_bound_ratio'] = update_bound_ratio
        row['worst_gradients'] = sorted(grad_details, key=lambda v:v['relative_l2'], reverse=True)[:4]
        row['worst_parameters'] = sorted(param_details, key=lambda v:v['relative_l2'], reverse=True)[:4]
        row['pass'] = max(out_error, grad_error, same_grad_error, *v_errors) < tolerance and update_bound_ratio <= 1
        rows.append(row)
        print(json.dumps(row), flush=True)
        del eager, fast, fn, opt_e, opt_f, same_grad_opt, same_grad_params, bounds, a, b
        torch.cuda.empty_cache()
    atomic_json(REPO / 'results/joudaki_vit_door_h_0919/speed_checks.json', dict(rows=rows, all_pass=all(r['pass'] for r in rows)))
    if not all(r['pass'] for r in rows):
        raise RuntimeError('Compiled engine differs beyond numerical consistency threshold')


if __name__ == '__main__':
    main()

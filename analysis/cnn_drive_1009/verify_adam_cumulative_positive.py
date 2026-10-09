"""Reproduce the balanced-label actual-Adam cumulative identity check.

This is an arithmetic verification of a proved restricted CNN family. Every
image is paired with every class; these labels are NOT iid label draws.
All Conv and head parameters are jointly updated by torch.optim.Adam.
Run with a Python environment containing torch. JSON is written to stdout.
"""
import json
import math

import torch
import torch.nn.functional as F


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(100925)  # No stochastic draw is used after initialization.
    beta, beta2, eps = 0.9, 0.999, 1e-8
    steps = 200
    rows = torch.arange(8).reshape(8, 1)
    cols = torch.arange(8).reshape(1, 8)
    x = torch.stack((1 + .1 * rows + .03 * cols,
                     1.2 + .04 * rows + .13 * cols)).unsqueeze(1)
    xb = x.repeat_interleave(3, dim=0)
    yb = torch.arange(3).repeat(2)
    conv1 = torch.nn.Conv2d(1, 2, 1, bias=False)
    conv2 = torch.nn.Conv2d(2, 2, 1, bias=False)
    head = torch.nn.Linear(8, 3, bias=False)
    B = torch.tensor([[.2, -.1, .1, .15],
                      [-.1, .15, -.2, .1],
                      [-.1, -.05, .1, -.25]])
    with torch.no_grad():
        conv1.weight.fill_(1 / math.sqrt(2))
        conv2.weight.fill_(.5)
        head.weight.copy_(B.repeat(1, 2) / math.sqrt(2))
    params = list(conv1.parameters()) + list(conv2.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=.001, betas=(beta, beta2), eps=eps)
    Kbeta = ((1 - beta2) * (1 - beta * beta / beta2)) ** -.5
    mean_input = float(x.mean())

    def amplitudes():
        return [float(conv1.weight.detach().mean()) * math.sqrt(2),
                float(conv2.weight.detach().mean()) * 2]

    initial_amplitudes = amplitudes()
    initial_mean = mean_input * float(conv1.weight.detach()[0, 0, 0, 0])
    momentum, square_moment = 0., 0.
    signal, direct_sum, variation, norm_variation = 0., 0., 0., 0.
    previous_s = None
    previous_momentum = 0.
    maximum_symmetry_error = 0.
    maximum_gradient_symmetry_error = 0.
    maximum_identity_error = 0.
    maximum_moment_error = 0.
    maximum_mean_increase = 0.
    previous_mean = initial_mean
    minimum_step_cap = float('inf')
    minimum_conv_gradient = float('inf')
    for t in range(1, steps + 1):
        a1, a2 = amplitudes()
        cap = min(a1 / (2 * math.sqrt(2) * Kbeta),
                  a2 / (2 * 2 * Kbeta))
        lr = min(.001, cap)
        minimum_step_cap = min(minimum_step_cap, cap)
        for group in opt.param_groups:
            group['lr'] = lr
        opt.zero_grad()
        h1 = F.max_pool2d(F.relu(conv1(xb)), 2)
        h2 = F.max_pool2d(F.relu(conv2(h1)), 2)
        loss = F.cross_entropy(head(h2.flatten(1)), yb)
        loss.backward()
        for layer in [conv1, conv2]:
            grad = layer.weight.grad
            minimum_conv_gradient = min(minimum_conv_gradient, float(grad.min()))
            maximum_gradient_symmetry_error = max(
                maximum_gradient_symmetry_error,
                float((grad - grad.flatten()[0]).abs().max()))
        g = float(conv1.weight.grad[0, 0, 0, 0])
        momentum = beta * momentum + (1 - beta) * g
        square_moment = beta2 * square_moment + (1 - beta2) * g * g
        s = lr / ((1 - beta ** t) *
                  (math.sqrt(square_moment / (1 - beta2 ** t)) + eps))
        signal += s * mean_input * g
        direct_sum += s * mean_input * momentum
        if previous_s is not None:
            variation += (s - previous_s) * mean_input * previous_momentum
            norm_variation += abs(s - previous_s) * mean_input * abs(previous_momentum)
        boundary = beta / (1 - beta) * (-s * mean_input * momentum + variation)
        opt.step()
        state = opt.state[conv1.weight]
        maximum_moment_error = max(maximum_moment_error,
                                  abs(momentum - float(state['exp_avg'][0, 0, 0, 0])),
                                  abs(square_moment - float(state['exp_avg_sq'][0, 0, 0, 0])))
        with torch.no_grad():
            for layer in [conv1, conv2]:
                maximum_symmetry_error = max(maximum_symmetry_error,
                    float((layer.weight - layer.weight.flatten()[0]).abs().max()))
            head_channels = head.weight.reshape(3, 2, 4)
            maximum_symmetry_error = max(maximum_symmetry_error,
                float((head_channels[:, 0] - head_channels[:, 1]).abs().max()))
            mean = mean_input * float(conv1.weight[0, 0, 0, 0])
        maximum_mean_increase = max(maximum_mean_increase, mean - previous_mean)
        actual_sink = initial_mean - mean
        maximum_identity_error = max(maximum_identity_error,
                                    abs(actual_sink - (signal + boundary)),
                                    abs(actual_sink - direct_sum))
        previous_mean, previous_s, previous_momentum = mean, s, momentum

    momentum_bound = beta / (1 - beta) * (
        s * mean_input * abs(momentum) + norm_variation)
    lower_bound = signal - momentum_bound
    result = {
        'scope': 'Balanced class enumeration; NOT iid labels; actual jointly updated CNN/Adam',
        'torch_version': torch.__version__, 'dtype': 'float64', 'device': 'cpu',
        'seed': 100925, 'stochastic_draws_used': False, 'steps': steps,
        'images': 2, 'classes': 3, 'batch_size': 6, 'widths': [1, 2, 2],
        'input_shape': [2, 1, 8, 8], 'maxpool_size': 2,
        'hidden_bias': False, 'output_bias': False,
        'all_conv_and_head_parameters_updated': True,
        'beta1': beta, 'beta2': beta2, 'epsilon': eps,
        'base_learning_rate': .001, 'K_beta': Kbeta,
        'minimum_predictable_cap': minimum_step_cap,
        'initial_amplitudes': initial_amplitudes,
        'final_amplitudes': amplitudes(),
        'initial_first_conv_mean': initial_mean,
        'final_first_conv_mean': previous_mean,
        'actual_cumulative_mean_decrease': actual_sink,
        'signal_S': signal, 'signed_momentum_residual': boundary,
        'momentum_residual_bound': momentum_bound,
        'denominator_residual_bound': 0., 'martingale': 0.,
        'certified_lower_bound': lower_bound,
        'maximum_displacement_identity_error': maximum_identity_error,
        'maximum_optimizer_moment_error': maximum_moment_error,
        'maximum_channel_symmetry_error': maximum_symmetry_error,
        'maximum_gradient_symmetry_error': maximum_gradient_symmetry_error,
        'maximum_mean_increase': maximum_mean_increase,
        'minimum_conv_coordinate_gradient': minimum_conv_gradient,
    }
    assert lower_bound > 0
    assert actual_sink >= lower_bound - 1e-12
    assert maximum_identity_error < 1e-12
    assert maximum_moment_error < 1e-12
    assert maximum_symmetry_error < 1e-12
    assert maximum_gradient_symmetry_error < 1e-12
    assert maximum_mean_increase < 1e-12
    assert min(amplitudes()) > 0
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

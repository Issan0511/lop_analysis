#!/usr/bin/env python3
"""Exact small-overlap counterexample for a repeated-fit CNN proof strategy.

No claim is made that this one-switch counterexample determines the long-time
trajectory. It refutes uniform old-label-conditional positivity over the whole
positive-amplitude cone, even with arbitrarily small nonproportional overlap.
"""

import itertools
import json

import sympy as sp
import torch
import torch.nn.functional as F


CLASSES = 10
EPSILON = sp.Rational(1, 100)
A = sp.Rational(1, 100)
B = sp.Integer(1)
RIDGE = sp.Rational(1, 10)
RHO = [sp.Integer(1), sp.Rational(4, 5), sp.Rational(3, 5), sp.Rational(2, 5)]
N = 2


def centered_label(index):
    return sp.Matrix([[sp.Integer(j == index) - sp.Rational(1, CLASSES) for j in range(CLASSES)]])


def exact_case(epsilon, a, b=B, ridge=RIDGE):
    x1 = sp.Matrix([1, epsilon])
    x2 = sp.Matrix([epsilon, 1])
    features = sp.Matrix.hstack(a * x1, b * x2)
    old = centered_label(0).col_join(centered_label(1))
    fit = (features.T * features + ridge * sp.eye(2)).inv() * features.T * old
    prediction = features * fit
    raw = sp.factor((x1.T * prediction * fit[0, :].T)[0])

    s = 1 + epsilon**2
    c = 2 * epsilon
    r = sp.Rational(CLASSES - 1, CLASSES) * s - c / CLASSES
    q = sp.Rational(CLASSES - 1, CLASSES) * c - s / CLASSES
    aa = ridge + b**2 * s
    geometry = ridge * s + b**2 * (s**2 - c**2)
    determinant = (ridge + a**2 * s) * (ridge + b**2 * s) - a**2 * b**2 * c**2
    threshold = sp.factor(
        ridge * b**2 * c * (b**2 * c * r - aa * q)
        / (geometry * (aa * r - b**2 * c * q))
    )
    formula = sp.factor(
        a / determinant**2
        * (
            a**2 * geometry * (aa * r - b**2 * c * q)
            + ridge * b**2 * c * (aa * q - b**2 * c * r)
        )
    )
    assert raw == formula
    assert threshold > a**2
    assert raw < 0

    isolated_fit = a * (x1.T * old) / (ridge + a**2 * s)
    isolated_raw = sp.factor(a * s * (isolated_fit * isolated_fit.T)[0])
    assert isolated_raw > 0
    mean_direction = sum(RHO) / len(RHO) * (1 + epsilon) / 2
    full_mean_drive = sp.factor(mean_direction * raw / N)
    self_mean_drive = sp.factor(mean_direction * isolated_raw / N)

    return {
        "epsilon": str(epsilon),
        "target_amplitude": str(a),
        "other_amplitude": str(b),
        "ridge": str(ridge),
        "threshold_for_a_squared": str(threshold),
        "threshold_float": float(threshold),
        "a_squared": str(a**2),
        "label_projection_inner_product_q": str(q),
        "mean_input_direction": str(mean_direction),
        "full_raw_gain_gradient_exact": str(raw),
        "self_raw_gain_gradient_exact": str(isolated_raw),
        "full_mean_direction_gradient_exact": str(full_mean_drive),
        "self_mean_direction_gradient_exact": str(self_mean_drive),
        "full_mean_direction_gradient_float": float(full_mean_drive),
        "self_mean_direction_gradient_float": float(self_mean_drive),
    }


def all_old_labels_exact():
    x1 = sp.Matrix([1, EPSILON])
    x2 = sp.Matrix([EPSILON, 1])
    features = sp.Matrix.hstack(A * x1, B * x2)
    solution_map = (features.T * features + RIDGE * sp.eye(2)).inv() * features.T
    values = []
    for i, j in itertools.product(range(CLASSES), repeat=2):
        old = centered_label(i).col_join(centered_label(j))
        fit = solution_map * old
        values.append(sp.factor((x1.T * features * fit * fit[0, :].T)[0]))
    average = sp.factor(sum(values) / len(values))
    assert average > 0
    return {
        "old_label_assignments": len(values),
        "negative_conditional_gradient_assignments": sum(int(bool(v < 0)) for v in values),
        "old_label_averaged_raw_gradient_exact": str(average),
        "old_label_averaged_raw_gradient_float": float(average),
    }


def autograd_check(expected_mean_gradient):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    rho = torch.tensor([float(v) for v in RHO]).reshape(2, 2)
    patterns = torch.tensor([[1.0, float(EPSILON)], [float(EPSILON), 1.0]])
    images = patterns[:, :, None, None] * rho[None, None, :, :]
    weight = torch.tensor([float(A), float(B)]).reshape(2, 1, 1, 1).requires_grad_()
    features = F.max_pool2d(F.relu(F.conv2d(images, weight, groups=2)), 2).flatten(1)
    fixed = features.detach()
    old = F.one_hot(torch.tensor([0, 1]), CLASSES).to(torch.float64) - 1 / CLASSES
    fitted = torch.linalg.solve(
        fixed.T @ fixed + float(RIDGE) * torch.eye(2), fixed.T @ old
    )
    mean_input = images[:, 0].mean().item()
    gradients = []
    assignments = list(itertools.product(range(CLASSES), repeat=2))
    for index, new_classes in enumerate(assignments):
        new = F.one_hot(torch.tensor(new_classes), CLASSES).to(torch.float64) - 1 / CLASSES
        loss = ((features @ fitted - new) ** 2).sum() / (2 * N)
        grad = torch.autograd.grad(loss, weight, retain_graph=index + 1 < len(assignments))[0]
        gradients.append(mean_input * grad[0, 0, 0, 0].item())
    average = sum(gradients) / len(gradients)
    error = abs(average - expected_mean_gradient)
    assert error < 1e-13
    assert average < 0
    return {
        "architecture": "Depthwise Conv2d(2,2,1,groups=2,bias=False) -> ReLU -> MaxPool2d(2) -> Linear(2,10,bias=False)",
        "new_label_assignments_enumerated": len(assignments),
        "averaged_actual_mean_direction_gradient": average,
        "exact_formula_error": error,
        "dtype": "float64",
    }


def main():
    base = exact_case(EPSILON, A)
    increasingly_small = [
        exact_case(sp.Rational(1, 10**power), sp.Rational(1, 10**power))
        for power in [2, 3, 4, 6]
    ]
    output = {
        "scope": "Failure of a uniform per-old-label sign extension; not a long-time impossibility theorem",
        "classes": CLASSES,
        "hidden_bias_trained": False,
        "output_bias_trained": False,
        "base_exact_case": base,
        "arbitrarily_small_overlap_examples": increasingly_small,
        "old_label_average": all_old_labels_exact(),
        "autograd": autograd_check(base["full_mean_direction_gradient_float"]),
        "versions": {"sympy": sp.__version__, "torch": torch.__version__},
    }
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

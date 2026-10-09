#!/usr/bin/env python3
"""Reproduce the 9-image, 10-class shared-CNN long-time toy checks.

This verifies the derived recurrence and finite trajectories. Infinite-time
claims require the invariant-region/supermartingale proof in the companion
note. The model has no trainable output intercept and uses squared loss.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F


C = 3  # Input colors and convolution channels.
M = 3  # Images per color group.
N = C * M
CLASSES = 10
A = 2.1
B = 2.0
OFF_DIAGONAL = 0.1
RIDGE = 0.1
STEP_SIZE = 0.005
RHO = np.array([1.0, 0.8, 0.6, 0.4], dtype=np.float64)
H_INITIAL = A - B
H_EXIT = A + B - 2 * OFF_DIAGONAL
MEAN_DIRECTION_K = 1 + float(RHO.mean()) / C
RECURRENCE_SEED = 100914
AUTOGRAD_SEED = 100916


def verify_recurrence(steps: int) -> dict:
    rng = np.random.default_rng(RECURRENCE_SEED)
    eye = np.eye(CLASSES)

    def labels() -> np.ndarray:
        return eye[rng.integers(CLASSES, size=(C, M))] - 1 / CLASSES

    h = np.full(C, H_INITIAL)
    old = labels()
    accumulated_drive = np.zeros(C)
    maximum_h = float(h.max())
    minimum_label_norm_squared = float("inf")
    minimum_multiplicative_factor = float("inf")
    maximum_identity_error = 0.0

    for _ in range(steps):
        new = labels()
        old_sum = old.sum(axis=1)
        new_sum = new.sum(axis=1)
        readout = h[:, None] * old_sum / (M * h * h + RIDGE)[:, None]
        bare_gradient = np.einsum(
            "cl,cl->c", readout, M * h[:, None] * readout - new_sum
        )
        directional_gradient = MEAN_DIRECTION_K * bare_gradient / N
        h_next = h - 2 * STEP_SIZE * bare_gradient / N
        accumulated_drive += STEP_SIZE * directional_gradient

        if not np.all((h_next > 0) & (h_next < H_EXIT)):
            raise AssertionError("The verified trajectory exited its certified region")

        minimum_multiplicative_factor = min(
            minimum_multiplicative_factor, float(np.min(h_next / h))
        )
        minimum_label_norm_squared = min(
            minimum_label_norm_squared,
            float(np.min(np.sum(old_sum * old_sum, axis=1))),
        )
        h = h_next
        maximum_h = max(maximum_h, float(h.max()))
        identity_error = float(
            np.max(
                np.abs(
                    accumulated_drive
                    - MEAN_DIRECTION_K * (H_INITIAL - h) / 2
                )
            )
        )
        maximum_identity_error = max(maximum_identity_error, identity_error)
        old = new  # Ordinary label reuse, not independent redraw of old labels.

    assert minimum_label_norm_squared >= M - M * M / CLASSES - 1e-12
    assert maximum_identity_error < 1e-11
    return {
        "seed": RECURRENCE_SEED,
        "steps": steps,
        "label_reuse": True,
        "final_positive_pooled_features": h.tolist(),
        "cumulative_weighted_directional_gradient": accumulated_drive.tolist(),
        "mean_preactivation_displacement": (-accumulated_drive).tolist(),
        "maximum_pooled_feature_seen": maximum_h,
        "minimum_old_label_sum_squared_norm": minimum_label_norm_squared,
        "minimum_positive_update_factor": minimum_multiplicative_factor,
        "maximum_displacement_identity_error": maximum_identity_error,
        "exited_certified_region": False,
    }


def verify_actual_autograd(steps: int) -> dict:
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(AUTOGRAD_SEED)

    images = torch.zeros((N, C, 2, 2))
    rho = torch.tensor(RHO.reshape(2, 2))
    for image_index in range(N):
        images[image_index, image_index // M] = rho
    weight = torch.full((C, C, 1, 1), OFF_DIAGONAL)
    for channel in range(C):
        weight[channel, channel, 0, 0] = A
    weight.requires_grad_()
    bias = torch.full((C,), -B, requires_grad=True)
    mean_patch = images.mean(dim=(0, 2, 3))

    def labels() -> torch.Tensor:
        return (
            F.one_hot(torch.randint(CLASSES, (N,)), CLASSES).to(torch.float64)
            - 1 / CLASSES
        )

    def features() -> torch.Tensor:
        return F.max_pool2d(F.relu(F.conv2d(images, weight, bias)), 2).flatten(1)

    old = labels()
    initial_mean = (weight.detach().flatten(1) @ mean_patch + bias.detach()).clone()
    maximum_update_error = 0.0
    maximum_directional_gradient_error = 0.0
    maximum_mean_identity_error = 0.0
    maximum_off_group_feature = 0.0

    for _ in range(steps):
        feature_matrix = features()
        fixed_features = feature_matrix.detach()
        h = torch.stack([fixed_features[c * M, c] for c in range(C)])
        readout = torch.linalg.solve(
            fixed_features.T @ fixed_features + RIDGE * torch.eye(C),
            fixed_features.T @ old,
        )
        new = labels()
        loss = ((feature_matrix @ readout - new) ** 2).sum() / (2 * N)
        loss.backward()

        old_sum = old.reshape(C, M, CLASSES).sum(dim=1)
        new_sum = new.reshape(C, M, CLASSES).sum(dim=1)
        formula_readout = h[:, None] * old_sum / (M * h * h + RIDGE)[:, None]
        bare_gradient = (
            formula_readout * (M * h[:, None] * formula_readout - new_sum)
        ).sum(dim=1)
        predicted_h = h - 2 * STEP_SIZE * bare_gradient / N
        predicted_directional_gradient = MEAN_DIRECTION_K * bare_gradient / N
        actual_directional_gradient = weight.grad.flatten(1) @ mean_patch + bias.grad
        maximum_directional_gradient_error = max(
            maximum_directional_gradient_error,
            float(
                (actual_directional_gradient - predicted_directional_gradient)
                .abs()
                .max()
            ),
        )

        with torch.no_grad():
            weight -= STEP_SIZE * weight.grad
            bias -= STEP_SIZE * bias.grad
        weight.grad = None
        bias.grad = None
        old = new

        with torch.no_grad():
            updated_features = features()
            actual_h = torch.stack([updated_features[c * M, c] for c in range(C)])
            assert bool(torch.all((actual_h > 0) & (actual_h < H_EXIT)))
            maximum_update_error = max(
                maximum_update_error, float((actual_h - predicted_h).abs().max())
            )
            actual_mean = weight.flatten(1) @ mean_patch + bias
            predicted_mean_change = MEAN_DIRECTION_K * (actual_h - H_INITIAL) / 2
            maximum_mean_identity_error = max(
                maximum_mean_identity_error,
                float((actual_mean - initial_mean - predicted_mean_change).abs().max()),
            )
            for image_index in range(N):
                for channel in range(C):
                    if channel != image_index // M:
                        maximum_off_group_feature = max(
                            maximum_off_group_feature,
                            float(updated_features[image_index, channel].abs()),
                        )

    assert maximum_update_error < 1e-11
    assert maximum_directional_gradient_error < 1e-11
    assert maximum_mean_identity_error < 1e-11
    assert maximum_off_group_feature == 0
    return {
        "seed": AUTOGRAD_SEED,
        "steps": steps,
        "dtype": "float64",
        "architecture": "Conv2d(3,3,kernel_size=1) -> ReLU -> MaxPool2d(2) -> Linear(3,10)",
        "all_shared_conv_weights_and_biases_updated": True,
        "fitted_head_detached_during_hidden_gradient": True,
        "maximum_recurrence_update_error": maximum_update_error,
        "maximum_mean_direction_gradient_error": maximum_directional_gradient_error,
        "maximum_mean_displacement_identity_error": maximum_mean_identity_error,
        "maximum_off_group_pooled_feature": maximum_off_group_feature,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--autograd-steps", type=int, default=200)
    args = parser.parse_args()
    if args.steps <= 0 or args.autograd_steps <= 0:
        parser.error("Both step counts must be positive")

    results = {
        "scope": "Constructed repeated exact-head-fit CNN; not RL-CIFAR training evidence",
        "configuration": {
            "images": N,
            "input_colors": C,
            "conv_channels": C,
            "images_per_group": M,
            "output_classes": CLASSES,
            "trainable_output_bias": False,
            "A": A,
            "B": B,
            "off_diagonal_weight": OFF_DIAGONAL,
            "ridge": RIDGE,
            "hidden_sgd_step_size": STEP_SIZE,
            "rho": RHO.tolist(),
        },
        "analytic_constants": {
            "initial_pooled_feature": H_INITIAL,
            "geometric_exit_threshold": H_EXIT,
            "positive_step_ratio_must_be_less_than_one": 2 * STEP_SIZE * M * M / (N * RIDGE),
            "old_label_sum_squared_norm_lower_bound": M - M * M / CLASSES,
            "proved_nonexit_probability_lower_bound": 1 - C * H_INITIAL / H_EXIT,
            "mean_patch_direction_factor": MEAN_DIRECTION_K,
            "cumulative_drive_limit_on_nonexit_event": MEAN_DIRECTION_K * H_INITIAL / 2,
        },
        "autograd": verify_actual_autograd(args.autograd_steps),
        "recurrence": verify_recurrence(args.steps),
        "versions": {"numpy": np.__version__, "torch": torch.__version__},
    }
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

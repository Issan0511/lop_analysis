"""Exact finite certificate for binary N1200/B16/E400 stationary bias response.

No training, random sampling, finite-history approximation, or external theorem
is used. Python integer/Fraction arithmetic checks every possible task label
count, then the RMS-tail and positive common-mode response bounds.
"""
from argparse import ArgumentParser
from fractions import Fraction as F
from hashlib import sha256
from math import comb, log
from pathlib import Path
from time import monotonic
import json


N, BATCH, EPOCHS = 1200, 16, 400
BATCHES_PER_EPOCH = N // BATCH
TASK_LENGTH = EPOCHS * BATCHES_PER_EPOCH
COMPLETED_EPOCHS = 13
BETA1, BETA2, EPSILON = F(9, 10), F(999, 1000), F(1, 10**8)
PGF_ARGUMENT = F(7, 8)
V_CUTOFF = F(1, 100)


def integer_bytes(value):
    assert value >= 0
    size = max(1, (value.bit_length() + 7) // 8)
    payload = value.to_bytes(size, "big")
    return size.to_bytes(4, "big") + payload


def integer_list_digest(values):
    digest = sha256()
    for value in values:
        digest.update(integer_bytes(value))
    return digest.hexdigest()


def fraction_record(value, include_exact=False):
    out = {
        "decimal_display": float(value),
        "numerator_bits": value.numerator.bit_length(),
        "denominator_bits": value.denominator.bit_length(),
        "sha256_length_prefixed_unsigned_big_endian": integer_list_digest(
            [value.numerator, value.denominator]
        ),
    }
    if include_exact:
        out["numerator"] = str(value.numerator)
        out["denominator"] = str(value.denominator)
    return out


def balanced_count_certificate():
    # a_k = binom(16,k) * q**(8-k)**2, with common denominator 8**64.
    base = [
        comb(BATCH, k)
        * 7 ** ((8 - k) ** 2)
        * 8 ** (64 - (8 - k) ** 2)
        for k in range(BATCH + 1)
    ]
    coefficient = [1]
    for _ in range(BATCHES_PER_EPOCH):
        nxt = [0] * (len(coefficient) + BATCH)
        for i, previous in enumerate(coefficient):
            for k, value in enumerate(base):
                nxt[i + k] += previous * value
        coefficient = nxt

    assert len(coefficient) == N + 1
    assert coefficient == coefficient[::-1]
    assert sum(coefficient) == sum(base) ** BATCHES_PER_EPOCH
    central_binomial = comb(N, N // 2)
    central_coefficient = coefficient[N // 2]
    equality_counts = []
    for k, value in enumerate(coefficient):
        left = value * central_binomial
        right = central_coefficient * comb(N, k)
        assert left <= right
        if left == right:
            equality_counts.append(k)

    mu = sum(
        (
            F(comb(BATCH, k), 2**BATCH)
            * PGF_ARGUMENT ** ((8 - k) ** 2)
            for k in range(BATCH + 1)
        ),
        F(0),
    )
    assert mu == F(sum(base), 8**64 * 2**BATCH)
    return mu, {
        "all_label_count_comparisons_passed": True,
        "label_counts_checked": N + 1,
        "polynomial_degree": N,
        "coefficient_common_denominator": "8**(64*75)",
        "equality_counts": equality_counts,
        "maximum_coefficient_bits": max(x.bit_length() for x in coefficient),
        "coefficients_sha256_length_prefixed_unsigned_big_endian":
            integer_list_digest(coefficient),
    }


def verify():
    started = monotonic()
    assert N % BATCH == 0
    mu, polynomial = balanced_count_certificate()
    p_center = F(comb(N, N // 2), 2**N)

    # At every phase, all batches in the 13 most recent completed epochs have
    # lag less than 75*(13+1). The extra endpoint slack makes this uniform.
    maximum_lag_exponent = BATCHES_PER_EPOCH * (COMPLETED_EPOCHS + 1)
    minimum_ema_weight = (1 - BETA2) * BETA2**maximum_lag_exponent
    integer_threshold_fraction = 64 * V_CUTOFF / minimum_ema_weight
    numerator, denominator = (
        integer_threshold_fraction.numerator,
        integer_threshold_fraction.denominator,
    )
    integer_threshold = (numerator + denominator - 1) // denominator
    assert integer_threshold == 1830
    assert (integer_threshold - 1) * minimum_ema_weight < 64 * V_CUTOFF
    assert 64 * V_CUTOFF <= integer_threshold * minimum_ema_weight

    probability_bound = (
        mu ** (BATCHES_PER_EPOCH * COMPLETED_EPOCHS)
        / (
            p_center**COMPLETED_EPOCHS
            * PGF_ARGUMENT**integer_threshold
        )
    )
    rational_probability_upper = F(1, 2**80)
    assert probability_bound < rational_probability_upper

    marginal_variance = F(1, BATCH)
    population_mean_variance = F(1, N)
    variance_a = (
        (marginal_variance - population_mean_variance)
        * (1 - BETA1) / (1 + BETA1)
        + population_mean_variance
    )
    variance_b = (
        (marginal_variance - population_mean_variance)
        * (1 - BETA2) / (1 + BETA2)
        + population_mean_variance
    )
    cross_upper = F(19, 10000)
    assert variance_a * variance_b < cross_upper**2
    k0_squared = (
        (1 - BETA1)**2
        / ((1 - BETA2) * (1 - BETA1**2 / BETA2))
    )
    assert k0_squared == F(370, 7) and k0_squared < 64

    # V >= .01 implies sigma >= .1 and sigma*(sigma+2eps)**2 >= .001.
    gamma_lower = (
        1 / (F(1, 4) + 2 * EPSILON)
        - cross_upper / F(1, 1000)
        - rational_probability_upper / EPSILON
    )
    assert gamma_lower > F(209, 100)
    bias_normal_lower = F(TASK_LENGTH, N) * gamma_lower
    assert bias_normal_lower > F(209, 4)

    # Whole-task inverse-RMS regularity for q=8 at N1200, including E400.
    all_balanced_epoch_ratio = F(comb(BATCH, BATCH//2)**BATCHES_PER_EPOCH, comb(N, N//2))
    inverse_moment_order = 8
    epoch_inverse_factor = all_balanced_epoch_ratio / BETA2**(BATCHES_PER_EPOCH*inverse_moment_order//2)
    assert p_center < 1 and epoch_inverse_factor < 1

    return {
        "whole_task_inverse_RMS_certificate": {
            "order": inverse_moment_order,
            "all_balanced_conditional_epoch_probability": fraction_record(all_balanced_epoch_ratio),
            "epoch_geometric_factor": fraction_record(epoch_inverse_factor),
            "criterion": "p_center * epoch_geometric_factor**E < 1 for every finite E>=1",
            "every_finite_epoch_count_certified": True,
        },
        "scope": (
            "Exact finite arithmetic certificate for the infinite-history "
            "frozen binary output-bias response at uniform logits. Genuine "
            "independent epoch reshuffles, taskwise reused image labels, and "
            "all retained moments are included. No moving-CNN, full-normal-"
            "stability, Monte Carlo, or training claim is made."
        ),
        "N": N,
        "batch_size": BATCH,
        "epochs_per_task": EPOCHS,
        "task_length": TASK_LENGTH,
        "beta1": "9/10",
        "beta2": "999/1000",
        "epsilon": "1/100000000",
        "past_completed_epochs_used": COMPLETED_EPOCHS,
        "past_complete_batches_used": BATCHES_PER_EPOCH * COMPLETED_EPOCHS,
        "maximum_lag_exponent": maximum_lag_exponent,
        "pgf_argument": "7/8",
        "V_cutoff": "1/100",
        "polynomial_certificate": polynomial,
        "mu": fraction_record(mu, include_exact=True),
        "balanced_population_probability": fraction_record(p_center),
        "minimum_retained_ema_weight": fraction_record(minimum_ema_weight),
        "integer_square_sum_threshold": integer_threshold,
        "bad_probability_bound": fraction_record(probability_bound),
        "bad_probability_log_display": (
            log(probability_bound.numerator)
            - log(probability_bound.denominator)
        ),
        "bad_probability_rational_upper": "1/2**80",
        "variance_A_upper": fraction_record(variance_a, include_exact=True),
        "variance_B0_upper": fraction_record(variance_b, include_exact=True),
        "cross_absolute_expectation_upper": "19/10000",
        "stationary_Cauchy_K0_squared": "370/7",
        "Gamma_phase_lower": fraction_record(gamma_lower, include_exact=True),
        "Gamma_phase_simple_lower": "209/100",
        "bias_normal_coefficient_lower": fraction_record(
            bias_normal_lower, include_exact=True
        ),
        "bias_normal_coefficient_simple_lower": "209/4",
        "all_assertions_passed": True,
        "elapsed_seconds": monotonic() - started,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[2] / "results/cnn_drive_1009/adam_longreuse_bias.json",
    )
    args = parser.parse_args()
    result = verify()
    result["verifier_source_sha256"] = sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "all_assertions_passed": result["all_assertions_passed"],
        "label_counts_checked":
            result["polynomial_certificate"]["label_counts_checked"],
        "bad_probability_log_display": result["bad_probability_log_display"],
        "Gamma_phase_simple_lower": result["Gamma_phase_simple_lower"],
        "bias_normal_coefficient_simple_lower":
            result["bias_normal_coefficient_simple_lower"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()

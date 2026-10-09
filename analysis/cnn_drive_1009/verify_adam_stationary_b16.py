#!/usr/bin/env python3
"""Rigorous stationary Adam sign certificate for a fixed iid gradient driver.

This does not simulate a changing neural-network parameter or establish a
claim about an actual CNN learning trajectory. The certificate uses exact
rationals and certified rational square-root enclosures. Optional Monte Carlo
uses batch means, not an iid standard error for consecutive Adam outputs.
"""

from __future__ import annotations

import argparse
from fractions import Fraction as F
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile


def sqrt_interval(value: F, digits: int = 80) -> tuple[F, F]:
    assert value > 0
    scale = 10**digits
    root_floor = math.isqrt(value.numerator * scale * scale // value.denominator)
    lower = F(root_floor, scale)
    upper = F(root_floor + 1, scale)
    assert lower * lower <= value <= upper * upper
    assert lower > 0
    return lower, upper


def rational_record(value: F) -> dict:
    return {"exact": str(value), "float_for_display_only": float(value)}


def certificate() -> dict:
    batch = 16
    label_probability = F(9, 10)
    prediction = F(900001, 1000000)
    beta1, beta2, epsilon = F(9, 10), F(999, 1000), F(1, 100000000)
    rho = F(7, 10)
    gradients = [prediction - F(k, batch) for k in range(batch + 1)]
    probabilities = [
        F(math.comb(batch, k)) * label_probability**k * (1 - label_probability) ** (batch - k)
        for k in range(batch + 1)
    ]
    assert sum(probabilities) == 1
    mean_g = sum(p * g for p, g in zip(probabilities, gradients))
    r = sum(p * g * g for p, g in zip(probabilities, gradients))
    g_abs_max = max(abs(g) for g in gradients)
    g_abs_min = min(abs(g) for g in gradients)
    z_min = g_abs_min**2

    # Central moments of Z=g^2. Cumulants then add across independent lags.
    central_z = [
        sum(p * (g * g - r) ** order for p, g in zip(probabilities, gradients))
        for order in range(13)
    ]
    cumulant_z = [F(0)] * 13
    for order in range(1, 13):
        cumulant_z[order] = central_z[order] - sum(
            F(math.comb(order - 1, j - 1)) * cumulant_z[j] * central_z[order - j]
            for j in range(1, order)
        )
    cumulant_delta = [F(0)] + [
        cumulant_z[order] * (1 - beta2) ** order / (1 - beta2**order)
        for order in range(1, 13)
    ]
    moment_delta = [F(1)] + [F(0)] * 12
    for order in range(1, 13):
        moment_delta[order] = sum(
            F(math.comb(order - 1, j - 1)) * cumulant_delta[j] * moment_delta[order - j]
            for j in range(1, order + 1)
        )

    moment_m2 = mean_g**2 + (r - mean_g**2) * (1 - beta1) / (1 + beta1)
    moment_m_delta = (
        (1 - beta1) * (1 - beta2) / (1 - beta1 * beta2)
        * sum(p * (g - mean_g) * (g * g - r) for p, g in zip(probabilities, gradients))
    )
    moment_m_delta2 = mean_g * moment_delta[2] + (
        (1 - beta1) * (1 - beta2) ** 2 / (1 - beta1 * beta2**2)
        * sum(p * (g - mean_g) * (g * g - r) ** 2 for p, g in zip(probabilities, gradients))
    )

    # E[m T_2(v)] = coefficient / sqrt(r). The coefficient is negative here.
    coefficient = mean_g - moment_m_delta / (2 * r) + 3 * moment_m_delta2 / (8 * r**2)
    assert coefficient < 0
    sqrt_r_low, sqrt_r_high = sqrt_interval(r)
    taylor_low = coefficient / sqrt_r_low
    taylor_high = coefficient / sqrt_r_high

    good_root_argument = moment_m2 * moment_delta[6] / (rho * r) ** 7
    _, good_root_high = sqrt_interval(good_root_argument)
    good_error = F(5, 16) * good_root_high

    bad_probability = min(F(1), moment_delta[12] / ((1 - rho) * r) ** 12)
    bad_error = g_abs_max * (1 / g_abs_min + F(15, 8) / sqrt_r_low) * bad_probability

    _, sqrt_m2_high = sqrt_interval(moment_m2)
    epsilon_error = epsilon * sqrt_m2_high / z_min
    total_error = good_error + bad_error + epsilon_error
    lower = taylor_low - total_error
    upper = taylor_high + total_error
    assert mean_g == F(1, 1000000) > 0
    assert upper < 0

    return {
        "scope": "Stationary fixed-theta iid gradient driver, not an actual parameter-learning trajectory",
        "configuration": {
            "batch_size": batch,
            "label_probability": str(label_probability),
            "prediction": str(prediction),
            "beta1": str(beta1),
            "beta2": str(beta2),
            "adam_epsilon": str(epsilon),
            "good_event": "v >= (7/10) E[g^2]",
            "square_root_enclosure_decimal_digits": 80,
        },
        "distribution": [
            {"K": k, "gradient": str(g), "probability": str(p)}
            for k, (g, p) in enumerate(zip(gradients, probabilities))
        ],
        "moments": {
            "E_g": rational_record(mean_g),
            "E_g_squared": rational_record(r),
            "minimum_g_squared": rational_record(z_min),
            "maximum_absolute_g": rational_record(g_abs_max),
            "E_m_squared": rational_record(moment_m2),
            "E_m_delta": rational_record(moment_m_delta),
            "E_m_delta_squared": rational_record(moment_m_delta2),
            "E_delta_power_2": rational_record(moment_delta[2]),
            "E_delta_power_6": rational_record(moment_delta[6]),
            "E_delta_power_12": rational_record(moment_delta[12]),
            "all_delta_central_moments_exact": [str(value) for value in moment_delta],
            "all_delta_cumulants_exact": [str(value) for value in cumulant_delta],
        },
        "certificate": {
            "taylor_coefficient_exact": str(coefficient),
            "sqrt_r_interval_exact": [str(sqrt_r_low), str(sqrt_r_high)],
            "taylor_expectation_lower": rational_record(taylor_low),
            "taylor_expectation_upper": rational_record(taylor_high),
            "good_event_error_upper": rational_record(good_error),
            "bad_event_probability_upper": rational_record(bad_probability),
            "bad_event_error_upper": rational_record(bad_error),
            "epsilon_correction_error_upper": rational_record(epsilon_error),
            "stationary_adam_mean_lower": rational_record(lower),
            "stationary_adam_mean_upper": rational_record(upper),
            "strict_negative_upper_verified_using_rationals": True,
        },
    }


MC_SOURCE = r'''
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
int main(int argc,char** argv) {
  if(argc!=5) return 2;
  const uint64_t seed=std::stoull(argv[1]);
  const uint64_t burn=std::stoull(argv[2]);
  const uint64_t blocks=std::stoull(argv[3]);
  const uint64_t length=std::stoull(argv[4]);
  std::mt19937_64 rng(seed);
  std::array<double,17> cdf;
  long double prob=std::pow(0.9L,16), cumulative=0;
  for(int j=0;j<=16;++j) {
    cumulative+=prob; cdf[j]=(double)cumulative;
    if(j<16) prob*=((long double)(16-j)/(j+1))/9;
  }
  cdf[16]=1;
  double m=0,v=0,blocksum=0;
  const uint64_t total=burn+blocks*length;
  std::cout<<std::setprecision(17);
  for(uint64_t t=0;t<total;++t) {
    const double u=(double)(rng()>>11)*0x1.0p-53;
    int failures=0; while(u>cdf[failures]) ++failures;
    const double g=0.900001-(16-failures)/16.0;
    m=.9*m+.1*g; v=.999*v+.001*g*g;
    if(t>=burn) {
      blocksum+=m/(std::sqrt(v)+1e-8);
      if((t-burn+1)%length==0) {
        std::cout<<blocksum/length<<'\n'; blocksum=0;
      }
    }
  }
}
'''


def monte_carlo(seed_count=8, blocks=256, block_length=65536, burn=131072) -> dict:
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("Optional Monte Carlo requires g++")
    seeds = [10091001 + 104729 * index for index in range(seed_count)]
    by_seed = []
    with tempfile.TemporaryDirectory(prefix="cnn_adam_stationary_") as temporary:
        source = Path(temporary) / "simulate.cpp"
        binary = Path(temporary) / "simulate"
        source.write_text(MC_SOURCE)
        subprocess.run([compiler, "-O3", "-std=c++17", str(source), "-o", str(binary)], check=True)
        for seed in seeds:
            completed = subprocess.run(
                [str(binary), str(seed), str(burn), str(blocks), str(block_length)],
                check=True, text=True, capture_output=True,
            )
            values = [float(line) for line in completed.stdout.splitlines()]
            assert len(values) == blocks
            by_seed.append(values)
    reblocking = []
    for factor in [1, 2, 4]:
        combined = [
            statistics.mean(values[start:start + factor])
            for values in by_seed
            for start in range(0, len(values), factor)
            if start + factor <= len(values)
        ]
        mean = statistics.mean(combined)
        standard_error = math.sqrt(statistics.variance(combined) / len(combined))
        reblocking.append({
            "batch_length": block_length * factor,
            "number_of_batch_means": len(combined),
            "mean": mean,
            "batch_means_standard_error_estimate": standard_error,
            "approximate_normal_95_percent_interval": [mean - 1.96 * standard_error, mean + 1.96 * standard_error],
        })
    seed_means = [statistics.mean(values) for values in by_seed]
    return {
        "purpose": "Supplementary Monte Carlo; exact rational certificate proves the sign",
        "rng": "std::mt19937_64 with inverse CDF Binomial(16,0.1) failure count",
        "seeds": seeds,
        "burn_in_per_seed": burn,
        "samples_per_seed": blocks * block_length,
        "total_samples": seed_count * blocks * block_length,
        "bias_correction": "Omitted after burn-in; difference vanishes exponentially and is not used by the proof",
        "seed_means": seed_means,
        "independent_seed_mean_standard_error_estimate": math.sqrt(statistics.variance(seed_means) / len(seed_means)),
        "batch_means_reblocking": reblocking,
        "base_batch_means_by_seed": by_seed,
        "interval_status": "Asymptotic batch-means diagnostics, not rigorous probability intervals",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/cnn_drive_1009/adam_stationary_b16.json")
    parser.add_argument("--mc", action="store_true", help="Also run supplementary serial Monte Carlo with batch means")
    parser.add_argument("--mc-seeds", type=int, default=8)
    parser.add_argument("--mc-blocks", type=int, default=256)
    parser.add_argument("--mc-block-length", type=int, default=65536)
    args = parser.parse_args()
    result = {"rigorous_certificate": certificate()}
    if args.mc:
        result["monte_carlo"] = monte_carlo(args.mc_seeds, args.mc_blocks, args.mc_block_length)
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    bounds = result["rigorous_certificate"]["certificate"]
    summary = {
        "output": str(output),
        "SGD_mean": 1e-6,
        "stationary_Adam_mean_proved_interval": [
            bounds["stationary_adam_mean_lower"]["float_for_display_only"],
            bounds["stationary_adam_mean_upper"]["float_for_display_only"],
        ],
        "negative_upper_verified_using_exact_rationals": True,
    }
    if args.mc:
        summary["MC_reblocking"] = result["monte_carlo"]["batch_means_reblocking"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

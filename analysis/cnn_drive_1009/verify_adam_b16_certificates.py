"""Exact arithmetic certificates for true reshuffled batch-16 Adam.

This verifies the finite constants in the analytic proofs, not a sampled
stationary response or a long-time training trajectory. Standard library only.
"""
from fractions import Fraction as F
import json
from math import comb
from pathlib import Path


def record(x):
    return {"exact": str(x), "decimal_display": float(x)}


def main():
    batch, epochs, q = 16, 2, 8
    b1, b2, eps = F(9, 10), F(999, 1000), F(1, 10**8)
    distribution = [(F(2*k-batch, batch), F(comb(batch, k), 2**batch))
                    for k in range(batch+1)]
    assert sum(p for _, p in distribution) == 1
    assert sum(x*p for x, p in distribution) == 0
    r2 = sum(x*x*p for x, p in distribution)
    r4 = sum(x**4*p for x, p in distribution)
    assert r2 == F(1, 16) and r4 == F(46, 4096)
    k0_squared = (1-b1)**2 / ((1-b2)*(1-b1*b1/b2))
    assert k0_squared == F(370, 7) < 64
    p_batch = F(comb(batch, batch//2), 2**batch)
    assert p_batch == F(6435, 32768)
    certificates = []
    for n, root_bound, exponent_bound in [(32, F(21, 1000), 48),
                                         (48, F(31, 1000), 32)]:
        batches = n//batch
        h = epochs*batches
        t1 = (1-b1**h)/(1+b1**h)
        t2 = (1-b2**h)/(1+b2**h)
        # Check the partial-current-task square-weight bound at every phase.
        for beta, total in [(b1, t1), (b2, t2)]:
            for phase in range(1, h+1):
                assert (1-beta**phase)**2 + beta**(2*phase)*total <= total
        assert t1*t2 < root_bound**2
        exponent = (r2-F(1, 64))**2/(2*r4*t2)
        assert exponent == F(9, 92)/t2 > exponent_bound
        # exp(-x) < 2**(-m) for x > m, since Euler's number is > 2.
        lower = 1/(F(1, 4)+2*eps) - 32*root_bound - F(1, 2**exponent_bound)/eps
        assert lower > F(29, 10)
        p_global = F(comb(n, n//2), 2**n)
        r_epoch = F(comb(batch, batch//2)**batches, comb(n, n//2))
        failure = p_global*r_epoch**epochs
        assert p_batch < b2**(h*q//2)
        assert failure < b2**(h*q//2)
        # This separate condition supplies inverse moments for every finite E;
        # it does not assert the bias lower bound is positive at every E.
        assert r_epoch < b2**(batches*q//2)
        certificates.append({
            "N": n, "B": batch, "E": epochs, "H": h,
            "T1": record(t1), "T2": record(t2),
            "sqrt_T1_T2_strict_upper": record(root_bound),
            "tail_exponent": record(exponent),
            "tail_exponent_strict_integer_lower": exponent_bound,
            "Gamma_every_phase_strict_lower": record(lower),
            "normal_bias_common_coefficient_strict_lower": record(h*lower/n),
            "global_label_balance_probability": record(p_global),
            "all_batches_balanced_conditional_epoch_probability": record(r_epoch),
            "no_good_batch_task_probability": record(failure),
            "inverse_moment_order": q,
            "first_batch_inverse_moment_ratio": record(p_batch/b2**(h*q//2)),
            "whole_task_inverse_moment_ratio": record(failure/b2**(h*q//2)),
            "arbitrary_finite_epochs_geometric_factor": record(r_epoch/b2**(batches*q//2)),
        })
    out = {
        "scope": "Exact rational constants for analytic frozen-field proofs; no simulation or training run.",
        "beta1": record(b1), "beta2": record(b2), "epsilon": record(eps),
        "batch_mean_second_moment": record(r2), "batch_mean_fourth_moment": record(r4),
        "stationary_Adam_Cauchy_constant_squared": record(k0_squared),
        "first_batch_balance_probability": record(p_batch),
        "certificates": certificates,
        "verdict": "All exact inequalities passed. Bias positivity is certified for E=2 only; inverse-eighth-moment finiteness holds for every finite E for the stated reshuffle law.",
    }
    destination = Path(__file__).resolve().parents[2]/"results/cnn_drive_1009/adam_b16_certificates.json"
    destination.write_text(json.dumps(out, indent=2)+"\n")
    print(json.dumps({"output": str(destination), "verdict": out["verdict"],
                      "Gamma_lower_displays": [c["Gamma_every_phase_strict_lower"]["decimal_display"] for c in certificates]}, indent=2))


if __name__ == "__main__":
    main()

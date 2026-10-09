"""Rational checks for multiclass stationary-field obstacles and bias response.

No training or stationary simulation. The margins certify the analytic formulas
in adam_tenclass_equilibrium.md and adam_tenclass_bias.md.
"""
from fractions import Fraction as F
from math import comb
from pathlib import Path
import json


def rec(x):
    return {"exact": str(x), "decimal_display": float(x)}


def main():
    b1, b2, eps = F(9, 10), F(999, 1000), F(1, 10**8)
    n, batch, classes, epochs = 1200, 16, 10, 400
    h = n//batch*epochs
    p = F(1, classes)
    floor = min(abs(F(k, batch)-p) for k in range(batch+1))
    assert floor == F(1, 40)
    marginal_variance = p*(1-p)/batch
    assert marginal_variance == F(3, 40)**2
    population_variance = p*(1-p)/n
    var_a = (marginal_variance-population_variance)*(1-b1)/(1+b1)+population_variance
    var_b = (marginal_variance-population_variance)*(1-b2)/(1+b2)+population_variance
    cross = F(171, 10**6)
    assert var_a*var_b < cross**2
    gamma_lower = 1/(F(3, 40)+eps)-cross/floor**3
    assert gamma_lower > F(119, 50)
    normal_lower = F(h, n*classes)*gamma_lower
    assert normal_lower > F(119, 20)

    def constant(q, effective_epsilon):
        return (1-q)*(2*q-1)/(2*(q+effective_epsilon)**2)

    omega_one = (1-b1)*(1-b2)/(1-b1*b2)
    hidden_negative_magnitude = constant(F(3, 5), eps)*omega_one
    gauge_positive_magnitude = constant(F(9, 10), eps)*omega_one
    assert hidden_negative_magnitude > 0 and gauge_positive_magnitude > 0
    delta = F(1, 10**16)
    perturbation_upper = 9*F(51, 80)*delta/eps
    intermittent = []
    for duration in (4, 6):
        overlap = (1-b1)*(1-b2)*(b1*b2)**(2*duration-1)
        margin = constant(F(3, 5), batch*eps)*overlap
        assert 2*perturbation_upper < margin
        intermittent.append({"H": duration, "negative_response_magnitude_lower": rec(margin),
                             "positive_RGB_perturbation_upper": rec(perturbation_upper)})

    # The first-task-batch conditional probability in the constant-rate boundary.
    probabilities = [F(comb(batch, k))*p**k*(1-p)**(batch-k)
                     for k in range(batch+1)]
    recurrence_probability = min(probabilities[1], probabilities[2])
    assert recurrence_probability == probabilities[2] > 0
    jump_coefficient = (1-b1)/(1+b1)/(2*batch*(1+eps))
    assert jump_coefficient == 1/(608*(1+eps))

    out = {
        "scope": "Exact rational checks of analytic stationary-field and constant-rate-boundary formulas, not a moving-CNN experiment.",
        "N": n, "batch_size": batch, "classes": classes, "epochs_per_task": epochs, "H": h,
        "tenclass_output_bias_response": {
            "RMS_floor_at_uniform_predictions": rec(floor),
            "marginal_variance": rec(marginal_variance),
            "variance_A_upper": rec(var_a), "variance_B0_upper": rec(var_b),
            "cross_absolute_expectation_strict_upper": rec(cross),
            "Gamma_phase_strict_lower": rec(gamma_lower),
            "common_image_centered_class_block_coefficient_strict_lower": rec(normal_lower),
            "positive_subspace_dimension": classes-1,
            "limitation": "Positive on common-image centered-class directions only; the full bias block is positive semidefinite, and this does not establish an equilibrium of the hidden field.",
        },
        "uniform_prediction_obstructions": {
            "native_one_image_H1_hidden_negative_response_magnitude_lower_sensitivity_one": rec(hidden_negative_magnitude),
            "one_image_H1_output_bias_gauge_positive_response_lower": rec(gauge_positive_magnitude),
            "B16_intermittent_coordinate": intermittent,
            "positive_RGB_perturbation_delta": rec(delta),
            "limitation": "Stationary frozen-state fields only; the B16 example certifies a target hidden coordinate, not the entire first-filter mean or an indefinitely reversed moving trajectory.",
        },
        "constant_rate_boundary": {
            "fresh_task_bias_gradient_threshold": "1/32",
            "uniform_conditional_recurrence_probability_lower": rec(recurrence_probability),
            "nonvanishing_bias_jump_lower_divided_by_eta": rec(jump_coefficient),
        },
        "all_assertions_passed": True,
    }
    destination = Path(__file__).resolve().parents[2]/"results/cnn_drive_1009/adam_tenclass.json"
    destination.write_text(json.dumps(out, indent=2)+"\n")
    print(json.dumps({"output": str(destination), "all_assertions_passed": True,
                      "Gamma_phase_lower": float(gamma_lower), "normal_block_coefficient_lower": float(normal_lower)}, indent=2))


if __name__ == "__main__":
    main()

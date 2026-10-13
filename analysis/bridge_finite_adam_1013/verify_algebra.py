"""Exact finite-label algebra checks; no training or bridge-exit claim."""
from fractions import Fraction as F
from math import comb
from pathlib import Path
import hashlib
import json

import sympy as sp


def main():
    classes, batch = 10, 16
    weights = [F(comb(batch, k) * (classes - 1) ** (batch - k), classes**batch)
               for k in range(batch + 1)]
    mean = sum(w * k for k, w in enumerate(weights))
    var = sum(w * (k - mean) ** 2 for k, w in enumerate(weights))
    assert sum(weights) == 1
    assert mean == F(batch, classes)
    assert var == F(batch * (classes - 1), classes**2)
    trace_cov = classes * var / batch**2
    assert trace_cov == F(9, 160)
    assert weights[1] / weights[0] == F(batch, classes - 1)
    assert weights[1] >= weights[0]

    beta1, beta2, eta, eps = F(9, 10), F(999, 1000), F(1, 1000), F(1, 10**8)
    gap = (1 - beta1) / batch
    gamma = eta * gap / (2 * (1 + eps))
    assert gap == F(1, 160)
    assert gamma == F(625, 200000002)

    # Triangle inequality in the proof: two possible moments differ by gap.
    # Check the underlying identity symbolically, including arbitrary history.
    m, p, g, a, b, c, d, e, k = sp.symbols("m p g a b c d e k")
    m0 = sp.Rational(9, 10) * m + sp.Rational(1, 10) * p
    m1 = sp.Rational(9, 10) * m + sp.Rational(1, 10) * (p - sp.Rational(1, 16))
    assert sp.simplify(m0 - m1) == sp.Rational(1, 160)

    # Independent audit: if the scalar Adam map had the same value k on
    # all 17 gradients, each is a root of this quadratic. Its coefficient
    # identities would force c=0,a=k*e; actual v=0=>m=0 and eps>0 rule this out.
    polynomial = sp.Poly((a + b*g - k*e)**2 - k*k*(c + d*g*g), g)
    assert polynomial.all_coeffs() == [b*b-d*k*k, 2*a*b-2*b*e*k,
                                     a*a-2*a*e*k-c*k*k+e*e*k*k]

    # Cauchy-Schwarz bound on raw Adam moments, useful for the independent
    # compactness proof of nonzero conditional update variance.
    moment_q = (1-beta1)**2 / ((1-beta2)*(1-beta1**2/beta2))
    assert moment_q == F(370, 7)

    root = Path(__file__).resolve().parents[2]
    out = root / "results/bridge_finite_adam_1013"
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "PASS_ALGEBRA_ONLY",
        "bridge_exit_guarantee": "NOT_PROVED",
        "new_training_runs": 0,
        "classes": classes, "batch": batch,
        "gradient_noise_trace_exact": str(trace_cov),
        "moment_gap_exact": str(gap),
        "adam_step_threshold_exact": str(gamma),
        "adam_step_threshold_decimal": float(gamma),
        "probability_lower_bound_exact": str(weights[0]),
        "probability_lower_bound_decimal": float(weights[0]),
        "raw_moment_bound_exact": str(moment_q),
        "scope": "Exact-real Adam; first batch of each fresh iid-label task. No holding-region result.",
    }
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    base = Path("/home/issan/Projects/obsidian-research-data/push_direction_proof_0930/review/stage4_1002/block_1003")
    sources = [base / "stage14_1007_h4/agent_inject/p171_train.py",
               base / "stage9_1004/p138_multiclass.py"]
    provenance = {
        "date": "2026-10-13", "base_git_hash": "466f2fa9c9c0a08226890279ef74c18a1abbc9f2",
        "obsidian_canonical_head": "5cec3d94c92953a1180d72298d301be09692f272",
        "kind": "mathematical_reanalysis_and_exact_algebra_checks",
        "sources": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in sources],
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

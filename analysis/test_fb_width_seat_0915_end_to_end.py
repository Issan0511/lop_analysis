"""Independent synthetic NPZ-to-report tests; no production results used."""
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verdict", ROOT / "analysis/fb_width_seat_0915_verdict.py")
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


def write_fixture(out, delta=0., fixed_drift=0., no_growth=False, diverged=()):
    (out / "logs").mkdir(exist_ok=True)
    (out / "checks.json").write_text(json.dumps({"pass": True}))
    steps = np.arange(501) * 10000
    n, h = len(steps), 4
    for arm in V.ARMS:
        free = arm in (V.ARMS[0], V.ARMS[2])
        for seed in range(10):
            baseline = seed * .001
            zb = np.full((n, h), baseline)
            zb[451:] += -1 if free else fixed_drift
            zm = np.full((n, h), baseline + (delta if arm == V.FIXED[0] else 0))
            wf = np.ones((n, h, 5), dtype=np.float64)
            if free and not no_growth:
                wf[451:] *= 2
            payload = dict(seed=np.int64(seed), step=steps,
                           layer1_zbar=zb, layer1_zmax=zm,
                           layer1_w_free=wf, layer1_w_free_step=steps,
                           layer1_branch_step=steps,
                           layer1_w_flip=np.ones((n, h, 15)),
                           layer1_w_flip_norm=np.ones((n, h)),
                           layer1_n_band=np.zeros((n, h)),
                           layer1_k_on=np.zeros((n, h)),
                           layer1_denom=np.ones((n, h)),
                           layer1_v_unit=np.ones((n, h)),
                           layer1_strict_dead=np.zeros(n), unfit=np.ones(n),
                           flip_state=np.zeros((n, 15)), gamma=np.zeros(n),
                           numeric_divergence=np.bool_((arm, seed) in diverged))
            np.savez_compressed(out / "logs" / f"{arm}_seed{seed}.npz", **payload)


def test_cases():
    cases = [
        ({}, "SEAT_SAME", "STOPS_WITHOUT_GROWTH"),
        ({"delta": .1}, "SEAT_SAME", "STOPS_WITHOUT_GROWTH"),
        ({"delta": .3}, "NOISE_LIFTS_SEAT", "STOPS_WITHOUT_GROWTH"),
        ({"delta": -.3}, "NOISE_SINKS_SEAT", "STOPS_WITHOUT_GROWTH"),
        ({"fixed_drift": -.2}, "NOT_DETERMINED_MOVING", "KEEPS_SINKING"),
        ({"no_growth": True}, "SEAT_SAME", "NOT_DETERMINED"),
        ({"diverged": ((V.FIXED[0], 0), (V.FIXED[1], 1))}, "SEAT_SAME", "STOPS_WITHOUT_GROWTH"),
        ({"diverged": tuple((V.FIXED[0], s) for s in range(4))}, "NOT_DETERMINED", "NOT_DETERMINED"),
        ({"diverged": tuple((V.ARMS[4], s) for s in range(10))}, "SEAT_SAME", "STOPS_WITHOUT_GROWTH"),
    ]
    results = []
    with tempfile.TemporaryDirectory(prefix="fb_width_audit_") as tmp:
        out = Path(tmp)
        for settings, q1, q2 in cases:
            write_fixture(out, **settings)
            result = V.compute(out)
            assert result["Q1"] == q1, (settings, result["Q1"], q1)
            assert result["Q2"] == q2, (settings, result["Q2"], q2)
            if q1 != "NOT_DETERMINED":
                assert "arms" in result["report_only"], "Supplemental report was silently skipped"
            results.append({"case": repr(settings), "Q1": q1, "Q2": q2})
        write_fixture(out)
        (out / "checks.json").write_text('{"pass": false}')
        result = V.compute(out)
        assert result["Q1"] == result["Q2"] == "NOT_DETERMINED"
    return {"pass": True, "cases": results, "failed_checks_gate": True}


if __name__ == "__main__":
    print(json.dumps(test_cases(), indent=2))

"""B1-B10 of ``天井T0b_spec_0828.md`` §7 as machine-checked assertions.

Each test is one row of the spec's assertion table.  The predecessor's
structure table was read by a human; this one fails in CI instead.

Run against a different arm with::

    T0B_INPUT=results/ratchet_log_0829c python -m unittest \
        analysis.ceiling_t0b.test_assertions -v
"""

import os
import unittest
from pathlib import Path

from analysis.ceiling_t0.ceiling_t0 import load_one
from analysis.ceiling_t0b.ceiling_t0b import (
    assertion_frame,
    mu_norm_stats,
    pairs_from_npz,
    seed_evidence,
    seed_summary,
)

ROOT = Path(__file__).resolve().parents[2]
INPUT = os.environ.get("T0B_INPUT", "results/ratchet_log_0819")
CENTERED = os.environ.get("T0B_CENTERED", "results/ratchet_centered_0822")


def _logs(spec: str) -> list[Path]:
    base = Path(spec)
    base = base if base.is_absolute() else ROOT / base
    return sorted((base / "logs").glob("seed*.npz"),
                  key=lambda p: int(p.stem.removeprefix("seed")))


class StructuralAssertions(unittest.TestCase):
    frame = None

    @classmethod
    def setUpClass(cls) -> None:
        paths = _logs(INPUT)
        if not paths:
            raise unittest.SkipTest(f"no seed*.npz under {INPUT}/logs")
        evidence = []
        for path in paths:
            d = load_one(path)
            sp = pairs_from_npz(d)
            _v, _b, checks = seed_summary(sp)
            evidence.append(seed_evidence(d, checks))
            del sp, d
        centered = [mu_norm_stats(p) for p in _logs(CENTERED)]
        cls.frame = assertion_frame(evidence, centered, ROOT).set_index("id")

    def _row(self, key: str) -> None:
        row = self.frame.loc[key]
        self.assertTrue(bool(row["pass"]), f"{key} ({row.assertion}): {row.note}")

    def test_B1_required_columns_present(self) -> None:
        self._row("B1")

    def test_B2_recording_grid_is_known(self) -> None:
        self._row("B2")

    def test_B3_force_decomposition(self) -> None:
        self._row("B3")

    def test_B4_coordinate_definition(self) -> None:
        self._row("B4")

    def test_B5_primary_pair_starts(self) -> None:
        self._row("B5")

    def test_B6_no_axis_rotation_in_primary_pairs(self) -> None:
        self._row("B6")

    def test_B7_mu_axis_is_non_degenerate(self) -> None:
        self._row("B7")

    def test_B8_centered_is_out_of_scope(self) -> None:
        self._row("B8")

    def test_B9_arm_c_generator_exists(self) -> None:
        self._row("B9")

    def test_B10_h_matches_disp_minus_f0(self) -> None:
        self._row("B10")


if __name__ == "__main__":
    unittest.main()

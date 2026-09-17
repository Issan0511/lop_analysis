"""Regression for the task26 false mismatch; no training or experimental endpoint."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from src import neff_pred_0917 as N
from analysis.lc_complement_0917.gpu import reference_zmeans
from analysis.lc_complement_0917.run_all import completed_prefix


class Regression(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA reduction regression")
    def test_reference_reduction_and_old_counterexample(self):
        torch.set_num_threads(1)
        z = torch.randn((30,1200,100), generator=torch.Generator().manual_seed(917)).cuda()
        # Call the actual historical diagnostic, not a reimplementation of it.
        dummy = SimpleNamespace(fields=lambda: ((z, torch.ones_like(z)), (z, torch.ones_like(z))))
        expected, _ = N.Box.stats(dummy)
        actual = reference_zmeans(((z, z, z), (z, z, z)))
        for layer, values in enumerate(actual, 1):
            np.testing.assert_array_equal(values, expected[f"zbar_l{layer}"])
        old = np.array([float(q.double().mean()) for q in z])
        self.assertTrue((old != actual[0]).any(), "Fixture must expose the old reduction bug")

    def test_resume_only_exact_completed_prefix(self):
        plan = [("a", []), ("b", []), ("c", [])]
        state = dict(status="FAILED", current_job="b", completed=[dict(job="a")])
        self.assertEqual(completed_prefix(state, plan), 1)
        for bad in ({**state, "status":"RUNNING"}, {**state, "current_job":"c"},
                    {**state, "completed":[dict(job="b")]}):
            with self.assertRaises(ValueError):
                completed_prefix(bad, plan)


if __name__ == "__main__":
    unittest.main()

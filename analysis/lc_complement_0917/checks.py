"""Meaningful numerical, schedule, source-lineage and aggregation checks."""
import tempfile
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import lc_complement_0917 as C
from analysis.lc_complement_0917 import verdict as V


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.set_flush_denormal(True)
        C.R.H.setup("cpu")

    def test_schedule(self):
        for arm in C.SHOCK_ARMS:
            self.assertTrue(all(C.shock_gamma(arm, s, 2) == (1., 1.) for s in range(6000)))
        sw = [s for s in range(6000) if C.shock_gamma("K_sw2", s) != (1., 1.)]
        mid = [s for s in range(6000) if C.shock_gamma("K_mid2", s) != (1., 1.)]
        self.assertEqual(sw, list(range(75)))
        self.assertEqual(mid, list(range(3000, 3075)))
        rev = [C.shock_gamma("K_rev", s)[0] for s in range(6000)]
        self.assertEqual(sum(g != 1 for g in rev), 600)
        for epoch, gamma in zip(range(10, 81, 10), [1.5, .5, .25, 2.] * 2):
            self.assertEqual(rev[(epoch-1)*75:epoch*75], [gamma]*75)
            self.assertEqual(rev[(epoch-1)*75-1], 1.)
        self.assertEqual(C.shock_gamma("K_hold2_L2", 1234), (1., 2.))
        with self.assertRaises(ValueError):
            C.shock_gamma("typo", 0)

    def test_shock_chain_rule(self):
        gen = torch.Generator().manual_seed(184)
        p = [torch.randn(shape, generator=gen, dtype=torch.float64).requires_grad_()
             for shape in ((3,4), (3,), (2,3), (2,), (2,2), (2,))]
        x = torch.randn((5,4), generator=gen, dtype=torch.float64)
        g1, g2 = 1.5, .5
        z1, a1, z2, a2, lg = C.shock_forward(p, x, g1, g2)
        auto = torch.autograd.grad(lg.sum(), p)
        with torch.no_grad():
            d2 = torch.ones_like(lg) @ p[4] * (g2 * z2.clamp_max(0).exp())
            d1 = d2 @ p[2] * (g1 * z1.clamp_max(0).exp())
            manual = (d1.T @ x, d1.sum(0), d2.T @ a1, d2.sum(0),
                      torch.ones_like(lg).T @ a2, torch.ones_like(lg).sum(0))
        for actual, expected in zip(auto, manual):
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_natural_update_and_clone(self):
        p = C.R.H.init_params(11, torch.device("cpu"))
        ck = dict(params=p, m=[torch.zeros_like(v) for v in p],
                  v=[torch.zeros_like(v) for v in p], tc=0)
        saved = [v.detach().clone() for v in p]
        a, aa = C.clone_state(ck)
        b, ba = C.clone_state(ck)
        gen = torch.Generator().manual_seed(113)
        x = torch.randn((1200, 784), generator=gen)
        y = torch.randint(10, (1200,), generator=gen)
        state = gen.get_state()
        ar = C.R.train_task(a, aa, x, y, C.R.gen_from(state), 1)
        br = C.R.train_task(b, ba, x, y, C.R.gen_from(state), 1, C.ShockForward("N", 1, 75))
        for a1, b1 in zip(ar[:3], br[:3]):
            self.assertTrue(torch.equal(a1, b1))
        self.assertEqual(C.R.SH.state_sha256(a, aa, C.R.ACT), C.R.SH.state_sha256(b, ba, C.R.ACT))
        self.assertTrue(all(torch.equal(v, w) for v,w in zip(p, saved)))

    def test_confusion_and_signed_neff(self):
        a = torch.tensor([[0., -1., 1.], [0., -1., .5]])
        g = torch.tensor([[.5, 0., -1.], [.5, 0., 1.]])
        st, neff = C.confusion(a, g, batch=2)
        self.assertEqual(st["false_dead_n"], 2)
        self.assertEqual(st["zero_n"], 2)
        self.assertEqual(st["A"], .5)  # two healthy zeros and two ELU floors
        self.assertEqual(st["B"], 0.)
        np.testing.assert_array_equal(neff, [1, 0, 1])
        st, _ = C.confusion(torch.ones((16, 3)), torch.ones((16, 3)), "GELU")
        self.assertIsNone(st["A"])
        self.assertIsNone(st["B"])
        with self.assertRaises(ValueError):
            C.confusion(torch.tensor([[float("nan"), 0.]]), torch.zeros((1,2)))

    def test_confusion_uses_batch_not_dataset_std(self):
        a = torch.cat([torch.full((16, 2), .01), torch.full((16, 2), 100.)])
        st, _ = C.confusion(a, torch.ones_like(a), "GELU")
        self.assertEqual(st["lc_n"], 0)
        st_full, _ = C.confusion(a, torch.ones_like(a), "GELU", batch=32)
        self.assertEqual(st_full["lc_n"], 32)

    def test_c3_activations(self):
        z = torch.tensor([-50., -2., -.2, -.0001, .0001, 2.], dtype=torch.float64, requires_grad=True)
        for cap, slope, _ in C.C3_ARMS.values():
            a = C.c3_phi(z, cap, slope)
            g = torch.autograd.grad(a.sum(), z)[0]
            expected = torch.where(z > 0, z, cap * torch.expm1((slope/cap) * z))
            dg = torch.where(z > 0, torch.ones_like(z), slope*torch.exp((slope/cap)*z))
            torch.testing.assert_close(a, expected, atol=1e-14, rtol=1e-12)
            # Native CELU backward converts its alpha reciprocal to float32 even
            # for float64 inputs on this kernel. Bound scalar conversion error.
            torch.testing.assert_close(g, dg, atol=1e-14, rtol=4*torch.finfo(torch.float32).eps)
        zz = torch.tensor([-20., -50.], requires_grad=True)
        g = torch.autograd.grad(C.c3_phi(zz, 1., 1.).sum(), zz)[0]
        self.assertTrue((g > 0).all())
        self.assertTrue((C.R.dphi_train(zz) == 0).all())

    def test_c3_rng_and_adam_reference(self):
        p, labels = C.c3_init(2)
        q, labels2 = C.c3_init(2)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(p,q)))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(labels,labels2)))
        model = torch.nn.Sequential(torch.nn.Linear(784,100), torch.nn.ELU(3.6),
                                    torch.nn.Linear(100,100), torch.nn.ELU(3.6), torch.nn.Linear(100,10))
        with torch.no_grad():
            for dst, src in zip(model.parameters(), p):
                dst.copy_(src)
        gen = torch.Generator().manual_seed(823)
        x = torch.randn((16,784), generator=gen)
        y = labels[0][:16]
        op, om = torch.optim.Adam(p, lr=1e-4), torch.optim.Adam(model.parameters(), lr=1e-4)
        for _ in range(3):
            for opt, lg in ((op, C.c3_forward(p,x,3.6,3.6)[4]), (om,model(x))):
                opt.zero_grad()
                torch.nn.functional.cross_entropy(lg,y).backward()
                opt.step()
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(p,model.parameters())))

    def test_adam_bound_and_counterexample(self):
        b1, b2, n = .9, .999, 1000
        # Equality history from weighted Cauchy-Schwarz, scaled to avoid overflow.
        grad = (b1/b2) ** np.arange(n-1, -1, -1, dtype=float)
        m, v = 0., 0.
        for t, g in enumerate(grad, 1):
            m = b1*m + (1-b1)*g
            v = b2*v + (1-b2)*g*g
            ratio = (m/(1-b1**t)) / np.sqrt(v/(1-b2**t))
            self.assertLessEqual(ratio, C.adam_bound(step=t) * (1 + 1e-12))
        self.assertGreater(ratio, 3.16)
        self.assertAlmostEqual(ratio, C.adam_bound(step=n), places=12)
        self.assertAlmostEqual(C.adam_bound(step=1), 1.)
        self.assertLess(C.adam_bound(step=n), C.adam_bound())

    @staticmethod
    def c2_rows():
        return pd.DataFrame([dict(seed=s,branch=b,arm=a,k=k,online_acc=.8,finite=True)
                             for s in range(10) for b in (2,5) for a in C.SHOCK_ARMS for k in (1,2)])

    def test_c2_verdicts(self):
        d = self.c2_rows()
        d.loc[d.arm == "K_hold2", "online_acc"] = .2
        d.loc[d.arm == "K_sw2", "online_acc"] = .3
        r = V.c2(d)
        self.assertEqual(r[0]["labels"], "PERSISTENCE_MATTERS|TIMING_MATTERS")
        d.loc[d.arm == "K_rev", "online_acc"] = .5
        self.assertEqual(V.c2(d)[0]["labels"], "TIMING_MATTERS|SHOCK_HARMS")
        self.assertEqual(V.c2(self.c2_rows())[0]["labels"], "INCONCLUSIVE")
        with self.assertRaises(ValueError):
            V.c2(d.iloc[1:])
        with self.assertRaises(ValueError):
            V.c2(pd.concat([d,d.iloc[:1]]))
        d.loc[0,"online_acc"] = float("nan")
        with self.assertRaises(ValueError):
            V.c2(d)

    def test_c1_rank_and_output_identity(self):
        names = list(C.R.ARMS)
        rows = []
        for s in range(10):
            for j, arm in enumerate(names):
                for layer in (1,2):
                    rows.append(dict(seed=s, arm=arm, layer=layer, E=j/26, lc_n=5,
                                     lc_frac=.5, tr_frac=1-j/26, zero_frac=0., neff_abs=j/26))
        d = pd.DataFrame(rows)
        result = V.c1(d)
        ranks = {r["metric"]: r["rho"] for r in result
                 if r["layer"] == 2 and r["aggregation"] == "26_arm_means"}
        self.assertIsNone(ranks["lc_frac"])
        self.assertAlmostEqual(ranks["neff_abs"], 1.)
        self.assertAlmostEqual(ranks["tr_frac"], -1.)
        d.loc[(d.arm == "R2_10") & (d.seed == 0), "lc_n"] = 6
        with self.assertRaises(ValueError):
            V.c1(d)

    def test_c3_windows_effects_censoring(self):
        df = pd.DataFrame([dict(seed=s,arm=a,task=t,fit=.8 if a in ("S36","E36") else .2,online_acc=.5)
                           for s in range(3) for a in C.C3_ARMS for t in range(1,51)])
        r = V.c3(df)
        effects = {x["arm"]: x for x in r if "mean" in x}
        self.assertAlmostEqual(effects["slope_effect"]["mean"], .6)
        self.assertAlmostEqual(effects["cap_effect"]["mean"], 0.)
        self.assertTrue(all(x["T_half_censored"] for x in r if x["arm"] in ("S36","E36")))
        with self.assertRaises(ValueError):
            V.c3(df[df.task != 50])

    def test_incomplete_and_smoke_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            C.dump(p / "provenance.json", dict(experiment=C.EXPERIMENT,part="c2",status="COMPLETE",smoke=True))
            with self.assertRaises(ValueError):
                V.load(p, "c2", 0)
            valid = dict(experiment=C.EXPERIMENT, part="c2", status="COMPLETE", smoke=False,
                         config=dict(seed=0), spec_sha256=C.sha(C.ROOT / C.SPEC), git_dirty_code="",
                         code_sha256=C.code_hashes("c2"))
            C.dump(p / "provenance.json", valid)
            self.c2_rows().query("seed == 0").to_csv(p / "rows.csv", index=False)
            self.assertEqual(len(V.load(p,"c2",0)),24)
            valid["code_sha256"]["src/lc_complement_0917.py"] = "wrong"
            C.dump(p / "provenance.json", valid)
            with self.assertRaises(ValueError):
                V.load(p, "c2", 0)

    def test_gpu_summary_and_incomplete_rejection(self):
        from src import neff_pred_0917 as N
        df = pd.DataFrame([dict(engine=engine, seed=m["seed"], env=m["env"],
                                act1=N.model_key(m)[0], act2=N.model_key(m)[1], task=t,
                                layer=l, phase=phase, E=t/150 if phase=="start" else None,
                                A=.4, lc_frac=t/150, tr_frac=1-t/150, zero_frac=0., neff_abs=t/150)
                           for engine, models in (("B", N.B.MODELS), ("L", N.L.MODELS)) for m in models
                           for t in range(1,151) for l in (1,2) for phase in ("start","end")])
        rows = V.c1_gpu(df)
        selected = [r for r in rows if r["query"] == "RL_GELU_l2_t50"]
        self.assertEqual(len(selected),3)
        self.assertTrue(all(r["prediction_A_ge_point3"] for r in selected))
        with self.assertRaises(ValueError):
            V.c1_gpu(df.iloc[1:])

    def test_archived_branch_integrity(self):
        if not C.ARCHIVE.exists():
            self.skipTest("Local resp_ee archive is absent")
        cks, x, _ = C.load_branches(0)
        self.assertEqual(sorted(cks), list(C.R.BRANCH_T))
        self.assertEqual(tuple(x.shape), (1200,784))

    def test_archived_natural_continuation(self):
        if not C.ARCHIVE.exists():
            self.skipTest("Local resp_ee archive is absent")
        cks, x, _ = C.load_branches(0)
        ck = cks[5]
        p, adam = C.clone_state(ck)
        gl, gb = C.R.gen_from(ck["g_lab"]), C.R.gen_from(ck["g_batch"])
        ref = pd.read_csv(C.REFERENCE / "s0/prefix.csv", float_precision="round_trip").set_index("task")
        for k in (1,2):
            _, acc, _, _, _ = C.R.train_task(p, adam, x, C.R.RL.task_labels(gl), gb, 80,
                                            C.ShockForward("N", k, 6000))
            self.assertEqual(C.R.SH.state_sha256(p,adam,C.R.ACT), ref.loc[5+k,"state_sha256"])
            self.assertEqual(float(acc.double().mean()), ref.loc[5+k,"online_acc"])


if __name__ == "__main__":
    unittest.main()

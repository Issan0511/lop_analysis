"""Synthetic-only invariance/restore/optimizer tests; no experiment data loaded."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F
from src import stage1_ctrl_1111 as M
from . import mlp as A


class MLPTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        g = torch.Generator().manual_seed(731)
        self.x = torch.randn(29, 7, generator=g)
        self.y = torch.arange(29) % 10
        self.Y = F.one_hot(self.y, 10).float()
        self.params = A.initial_params(2, 'off', (7, 11, 10))

    def model(self, activation='ELU', ln='off'):
        ps = [p.clone() for p in self.params]
        if ln == 'on': ps += [torch.ones(11), torch.zeros(11)]
        return A.MLP(ps, activation, ln)

    def test_common_initialization_rng_and_parameter_groups(self):
        before = torch.get_rng_state().clone()
        off = A.initial_params(3, 'off', (7, 11, 10))
        on = A.initial_params(3, 'on', (7, 11, 10))
        self.assertEqual(A.digest(off), A.digest(on[:4]))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for act in A.ACTIVATIONS:
            m = self.model(act, 'on')
            self.assertEqual([id(p) for p in m.body()], [id(m.P[i]) for i in (0, 1, 4, 5)])
            self.assertEqual([id(p) for p in m.head()], [id(m.P[i]) for i in (2, 3)])
            self.assertEqual(len(m.groups()['final_gamma']), 1)
            self.assertEqual(len(m.groups()['final_beta']), 1)

    def test_no_ln_native_forward_loss_gradient_and_updates_exact(self):
        for act in A.ACTIVATIONS:
            m = self.model(act)
            p = M.new_params(self.params); opt = M.make_opt('adam', p, .001)
            for _ in range(3):
                direct = M.activ(self.x@p[0].T+p[1], act)@p[2].T+p[3]
                actual = m.forward(self.x)
                self.assertTrue(torch.equal(actual, direct))
                l0 = (direct.logsumexp(1)-(direct*self.Y).sum(1)).mean()
                l1 = A.loss_for(actual, self.Y)
                self.assertTrue(torch.equal(l0, l1))
                opt.zero_grad(set_to_none=True); l0.backward()
                m.opt.zero_grad(set_to_none=True); l1.backward()
                self.assertTrue(all(torch.equal(a.grad,b.grad) for a,b in zip(p,m.P)))
                opt.step(); m.opt.step()
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(p,m.P)))
            self.assertEqual(A.evaluate(m,self.x,self.y), M.evaluate(p,M.Ctx(self.x,act),self.y))

    def test_final_ln_genuine_forward_and_autograd(self):
        for act in A.ACTIVATIONS:
            m = self.model(act, 'on')
            with torch.no_grad():
                m.P[4].copy_(torch.linspace(.4, 1.6, 11)); m.P[5].copy_(torch.linspace(-.5, .7, 11))
            # Separate leaf tensors and actual torch LayerNorm give an independent
            # reference for all six parameter gradients and the placement of LN.
            p = [v.detach().clone().requires_grad_() for v in m.P]
            ln = torch.nn.LayerNorm(11, eps=A.LN_EPS)
            with torch.no_grad(): ln.weight.copy_(p[4]); ln.bias.copy_(p[5])
            raw = self.x@p[0].T+p[1]
            native_h = ln(M.activ(raw, act)); native_z = native_h@p[2].T+p[3]
            actual_h, actual_raw, actual_a = m.features(self.x, True)
            self.assertTrue(torch.equal(actual_raw, raw))
            self.assertTrue(torch.equal(actual_h, native_h))
            self.assertTrue(torch.equal(m.forward(self.x), native_z))
            A.loss_for(native_z, self.Y).backward()
            A.loss_for(m.forward(self.x), self.Y).backward()
            for i in range(4): self.assertTrue(torch.equal(m.P[i].grad, p[i].grad))
            self.assertTrue(torch.equal(m.P[4].grad, ln.weight.grad))
            self.assertTrue(torch.equal(m.P[5].grad, ln.bias.grad))
            wrong = M.activ(F.layer_norm(raw,(11,),m.P[4],m.P[5],A.LN_EPS),act)
            self.assertFalse(torch.allclose(actual_h, wrong))

    def test_restore_exact_parameters_optimizer_rng_and_replay(self):
        m = self.model('SILU','on')
        A.one_step(m,self.x,self.Y)
        state = m.snapshot(); initial = m.fingerprints()
        reference = None
        for _ in range(2):
            m.restore(state); self.assertEqual(m.fingerprints(),initial)
            torch.rand(9)
            A.one_step(m,self.x,self.Y)
            after = A.digest(m.snapshot())
            if reference is None: reference=after
            else: self.assertEqual(after,reference)
        m.restore(state)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'source.pt'; A.save_state(path,state)
            stored=torch.load(path,weights_only=False)
            self.assertEqual(A.digest(state),A.digest(stored))

    def test_scale_permutation_and_full_mean_logits_after_ln(self):
        for ln in ('off','on'):
            m=self.model('ELU',ln)
            with torch.no_grad():
                if ln=='on': m.P[4].mul_(1.7); m.P[5].add_(.4)
                h=m.features(self.x)
                bank=A.ops.geometry_bank(*m.head(),h.mean(0),3,'mlp',10)
                before=m.forward(self.x).double().mean(0)
            start=m.snapshot()
            for arm in ('S1','S01','P1','P01'):
                m.restore(start)
                checks=A.apply_arm(m,arm,bank)
                self.assertEqual(checks['before']['body'],checks['after']['body'])
                self.assertTrue(torch.allclose(m.forward(self.x).double().mean(0),before,atol=1e-6,rtol=1e-6))
                if arm=='P1':
                    self.assertTrue(torch.equal(m.P[2].sort(0).values,start['params_and_adam'][0][2].sort(0).values))
                if arm=='S1': self.assertEqual(A.digest(m.head()),A.digest(start['params_and_adam'][0][2:4]))
            # Permutation depends only on seed/age; no activation or LN factor.
            p0=A.ops.canonical_permutation(3,'mlp',10,11,act='ELU')
            p1=A.ops.canonical_permutation(3,'mlp',10,11,act='SILU')
            self.assertTrue(torch.equal(p0,p1))

    def test_seven_arm_optimizer_semantics_and_fresh_body(self):
        m=self.model('ELU','on'); initial=m.snapshot()
        for _ in range(3): A.one_step(m,self.x,self.Y)
        trained=m.snapshot(); trained_finger=m.fingerprints()
        with torch.no_grad(): bank=A.ops.geometry_bank(*m.head(),m.features(self.x).mean(0),1,'mlp',10)
        for arm in A.ARMS:
            m.restore(trained); checks=A.apply_arm(m,arm,bank,initial)
            self.assertFalse(any(m.opt.state.get(p) for p in m.head()))
            if arm in ('Z_clear','F0'): self.assertEqual(len(m.opt.state),0)
            else: self.assertTrue(all(m.opt.state.get(p) for p in m.body()))
            if arm=='F0':
                wanted=[initial['params_and_adam'][0][i] for i in (0,1,4,5)]
                self.assertEqual(A.digest(m.body()),A.digest(wanted))
                self.assertNotEqual(A.digest(m.body()),trained_finger['body'])
                self.assertTrue(torch.equal(m.P[4],torch.ones_like(m.P[4])))
                self.assertTrue(torch.equal(m.P[5],torch.zeros_like(m.P[5])))
            else: self.assertEqual(A.digest(m.body()),trained_finger['body'])
            if arm.startswith('Z') or arm=='F0':
                self.assertTrue(all(torch.count_nonzero(p)==0 for p in m.head()))
                before=A.digest(m.body()); A.one_step(m,self.x,self.Y)
                # With zero head the first body gradient is exactly0. Retained
                # Adam still moves it; clear Adam does not. This includes LN.
                self.assertEqual(A.digest(m.body())==before,arm!='Z_keep')

    def test_actual_derivatives_all_four_activations(self):
        z=torch.tensor([-5.,-1.,-.2,.2,1.,5.],dtype=torch.float64,requires_grad=True)
        for act in A.ACTIVATIONS:
            grad=torch.autograd.grad(M.activ(z,act).sum(),z)[0]
            self.assertTrue(torch.allclose(grad,M.dphi64(z,act),atol=1e-14,rtol=1e-14))

    def test_float32_actual_derivative_tail_kinks_and_gradient_isolation(self):
        z=torch.tensor([-20.,-17.,-1.,0.,1.],dtype=torch.float32)
        actual=A.actual_activation_derivative(z,'ELU')
        u=z.clone().requires_grad_()
        reference=torch.autograd.grad(M.activ(u,'ELU').sum(),u)[0]
        self.assertTrue(torch.equal(actual,reference))
        self.assertEqual(float(actual[0]),0.)
        # expm1 output rounding enters its backward in the canonical engine.
        analytic=torch.exp(z[:2])
        self.assertFalse(torch.equal(actual[:2],analytic))
        native_u=z.clone().requires_grad_()
        native=torch.autograd.grad(F.elu(native_u).sum(),native_u)[0]
        self.assertTrue(bool((native[:2]>0).all()))
        self.assertFalse(torch.equal(actual[:2],native[:2]))
        expected={'ELU':1.,'R':0.,'LR':.1,'SILU':.5}
        for act in A.ACTIVATIONS:
            slope=A.actual_activation_derivative(torch.zeros(1),act)
            self.assertEqual(slope.dtype,torch.float32)
            self.assertEqual(float(slope[0]),float(torch.tensor(expected[act])))
            model=self.model(act,'on')
            A.loss_for(model.forward(self.x),self.Y).backward()
            gradients=[p.grad.clone() for p in model.P]
            state=A.digest(model.snapshot())
            measured=A.measure(model,self.x,self.y,self.y,0,True)
            self.assertIn('float32 autograd',measured['diagnostics']['activation_derivative_method'])
            self.assertEqual(A.digest(model.snapshot()),state)
            self.assertTrue(all(torch.equal(p.grad,g) for p,g in zip(model.P,gradients)))
        with self.assertRaises(ValueError): A.actual_activation_derivative(z.double(),'ELU')

    def test_common_provenance_contract(self):
        current=A.provenance(); shared=A.shared.provenance()
        for key in ('execution_git_hash','spec_sha256','code_hashes','code_sha256'):
            self.assertEqual(current[key],shared[key])
        self.assertEqual(current['local_code_hashes'],A.code_hashes())
        self.assertEqual(current['git_hash'],current['execution_git_hash'])

    def test_curve_grid_diagnostics_and_tracks(self):
        m=self.model('LR','on')
        # Synthetic dimensions retain10 classes; no data file or source replay.
        order=torch.arange(26*16).reshape(26,16)%len(self.x)
        curve,tracks=A.trajectory(m,self.x,self.y,self.y.roll(1),order,26)
        self.assertEqual([r['step'] for r in curve],[0,1,5,25,26])
        self.assertEqual([r['step'] for r in tracks],[1,25,26])
        self.assertEqual(set(curve[0]['diagnostics']) & {'activation_before_ln','feature'}, {'activation_before_ln','feature'})
        self.assertIn('final_gamma',tracks[0]['groups'])
        self.assertTrue({199,200,201} <= set(A.points(4000)))
        with self.assertRaises(FloatingPointError): A.finite({'bad':float('nan')})


if __name__=='__main__': unittest.main()

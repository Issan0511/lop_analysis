"""Synthetic-only operator tests. No real data or model reconstruction."""
import unittest
import torch
from analysis.readout_separation_1111 import ops

class Operators(unittest.TestCase):
    def setUp(self):
        self.g=torch.Generator().manual_seed(1328)

    def rand(self,*shape):
        return torch.randn(*shape,generator=self.g,dtype=torch.float64)

    def test_gain_and_centered_forward(self):
        x=self.rand(7,5).requires_grad_();v=self.rand(7,5);center=self.rand(5).requires_grad_()
        y=ops.GainIdentity.apply(x,.1)
        self.assertTrue(torch.equal(x,y))
        self.assertTrue(torch.equal(torch.autograd.grad(y,x,v)[0],.1*v))
        for cf in (1.,.1):
            y=ops.CenteredForward.apply(x,center,cf)
            self.assertTrue(torch.equal(y,x if cf==1 else center+cf*(x-center)))
            gx,gc=torch.autograd.grad(y,(x,center),v,allow_unused=True)
            self.assertTrue(torch.equal(gx,v));self.assertIsNone(gc)

    def test_elu_unity(self):
        z=torch.tensor([-100.,-2.,-.1,0.,.1,10.],dtype=torch.float64,requires_grad=True)
        native=ops.M.activ(z,'ELU');out=ops.ELUUnity.apply(native,z);v=self.rand(6)
        self.assertTrue(torch.equal(out,native))
        self.assertTrue(torch.equal(torch.autograd.grad(out,z,v)[0],v))

    def test_ln_native_affine_and_exact_input_vjp(self):
        for mode in ('native','ln_remove_radial','ln_remove_radial_mean'):
            x=self.rand(3,4,9).requires_grad_();gamma=self.rand(9).requires_grad_();beta=self.rand(9).requires_grad_()
            eps=.07;v=self.rand(3,4,9)
            native=torch.nn.functional.layer_norm(x,(9,),gamma,beta,eps)
            native_grad=torch.autograd.grad(native,(x,gamma,beta),v,retain_graph=True)
            out=ops.LNCorrect.apply(native,x,gamma,eps,mode)
            self.assertTrue(torch.equal(native,out))
            dx,dg,db=torch.autograd.grad(out,(x,gamma,beta),v)
            self.assertTrue(torch.equal(dg,native_grad[1]));self.assertTrue(torch.equal(db,native_grad[2]))
            xc=x-x.mean(-1,keepdim=True);sigma=(xc.square().mean(-1,keepdim=True)+eps).sqrt();q=xc/sigma;vg=v*gamma
            expected=vg
            if mode!='ln_remove_radial_mean':expected=expected-vg.mean(-1,keepdim=True)
            if mode=='native':expected=expected-q*(q*vg).mean(-1,keepdim=True)
            torch.testing.assert_close(dx,expected/sigma,rtol=1e-12,atol=1e-12)

    def test_geometry_and_local_rng(self):
        W=self.rand(10,17);b=self.rand(10);mu=self.rand(17)
        before=[a.clone() for a in (W,b,mu)];state=torch.random.get_rng_state().clone()
        bank=ops.geometry_bank(W,b,mu,3,'mlp',10)
        again=ops.geometry_bank(W,b,mu,3,'mlp',10)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        for a,q in zip((W,b,mu),before):self.assertTrue(torch.equal(a,q))
        self.assertTrue(torch.equal(bank['entries']['S1']['W'],W));self.assertTrue(torch.equal(bank['entries']['S1']['b'],b))
        for key,entry in bank['entries'].items():
            w,bb=entry['W'],entry['b'];c=entry['scale']
            self.assertTrue(torch.equal(w,again['entries'][key]['W']))
            torch.testing.assert_close(w@mu+bb,W@mu+b,rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(w.mean(0),W.mean(0),rtol=1e-12,atol=1e-12)
            torch.testing.assert_close((w-w.mean(0)).norm(dim=0),c*(W-W.mean(0)).norm(dim=0),rtol=1e-12,atol=1e-12)
        self.assertTrue(torch.equal(bank['entries']['P1']['W'].sort(0).values,W.sort(0).values))
        wo=bank['entries']['O1']['W'];torch.testing.assert_close(wo.T@wo,W.T@W,rtol=1e-12,atol=1e-12)
        Q=torch.tensor(bank['metadata']['O_matrix'],dtype=torch.float64)
        torch.testing.assert_close(Q.T@Q,torch.eye(10,dtype=torch.float64),rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(Q@torch.ones(10,dtype=torch.float64),torch.ones(10,dtype=torch.float64),rtol=1e-12,atol=1e-12)

    def test_head_and_body_gradients_are_separated(self):
        for cf in (1.,.1):
            h=self.rand(11,7).requires_grad_();W=self.rand(5,7).requires_grad_();b=self.rand(5).requires_grad_();mean=self.rand(5)
            y=torch.arange(11)%5;grads=[];zs=[]
            for gain in (1.,.1):
                hg=ops.GainIdentity.apply(h,gain)
                z=ops.CenteredForward.apply(hg@W.T+b,mean,cf)
                loss=torch.nn.functional.cross_entropy(z,y)
                grads.append(torch.autograd.grad(loss,(h,W,b)));zs.append(z.detach())
            self.assertTrue(torch.equal(zs[0],zs[1]))
            self.assertTrue(torch.equal(grads[0][1],grads[1][1]));self.assertTrue(torch.equal(grads[0][2],grads[1][2]))
            torch.testing.assert_close(grads[1][0],.1*grads[0][0],rtol=1e-14,atol=1e-14)
            residual=(zs[0].softmax(1)-torch.nn.functional.one_hot(y,5))/11
            torch.testing.assert_close(grads[0][0],residual@W,rtol=1e-13,atol=1e-13)
            torch.testing.assert_close(grads[0][1],residual.T@h,rtol=1e-13,atol=1e-13)
        # Physical c=.1 and surrogate (.1,.1) have identical initial p/body VJP.
        h=self.rand(11,7).requires_grad_();W=self.rand(5,7);b=self.rand(5);mu=h.detach().mean(0);y=torch.arange(11)%5
        bank=ops.geometry_bank(W,b,mu,0,'mlp',10);entry=bank['entries']['S01']
        zp=h@entry['W'].T+entry['b'];gp=torch.autograd.grad(torch.nn.functional.cross_entropy(zp,y),h)[0]
        zs=ops.CenteredForward.apply(ops.GainIdentity.apply(h,.1)@W.T+b,bank['zbar0'],.1)
        gs=torch.autograd.grad(torch.nn.functional.cross_entropy(zs,y),h)[0]
        # Centered row mean contributes only a softmax gauge common to all classes.
        torch.testing.assert_close(zp.softmax(1),zs.softmax(1),rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(gp,gs,rtol=1e-12,atol=1e-12)

    def test_cumulative_resource_budget(self):
        from analysis.readout_separation_1111.launch import remaining_budget
        old=dict(accounting_complete=True,external_gpu_seconds=20.,external_cpu_seconds=30.,
                 jobs=[dict(device='gpu',elapsed_s=100.),dict(device='cpu',elapsed_s=40.)])
        self.assertEqual(remaining_budget([old],60.,50.),(10640.,14310.))
        self.assertEqual(remaining_budget([],60.,50.),(10740.,14350.))
        with self.assertRaises(ValueError):remaining_budget([old],10.,50.)
        with self.assertRaises(RuntimeError):remaining_budget([dict(old,accounting_complete=False)],60.,50.)
        with self.assertRaises(ValueError):remaining_budget([],float('nan'),0.)

    def test_timeout_stops_descendants_after_wrapper_exits(self):
        import os,signal,subprocess,sys,tempfile,time
        from pathlib import Path
        from analysis.readout_separation_1111.launch import stop_group
        with tempfile.TemporaryDirectory() as tmp:
            ready=Path(tmp)/'ready'
            code=("import os,signal,time;from pathlib import Path;"
                  "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                  f"Path({str(ready)!r}).write_text(str(os.getpid()));time.sleep(30)")
            process=subprocess.Popen(['/usr/bin/time','-o',str(Path(tmp)/'resource'),sys.executable,'-c',code],
                                     start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            try:
                deadline=time.monotonic()+5
                while not ready.exists() and time.monotonic()<deadline:time.sleep(.01)
                self.assertTrue(ready.exists())
                pid=int(ready.read_text())
                self.assertTrue(stop_group(process,grace=.1))
                self.assertIsNotNone(process.returncode)
                stat=Path(f'/proc/{pid}/stat')
                deadline=time.monotonic()+1
                while stat.exists() and stat.read_text().split()[2]!='Z' and time.monotonic()<deadline:time.sleep(.01)
                self.assertTrue(not stat.exists() or stat.read_text().split()[2]=='Z')
            finally:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait()

    def test_fixed_plan_and_release(self):
        from analysis.readout_separation_1111.run import policy,evaluation_points
        for model,count in [('mlp',18),('vit',20)]:
            plan=ops.arm_plan(model);self.assertEqual(len(plan),count)
            self.assertEqual(len({a['id'] for a in plan}),count)
            for a in plan:
                self.assertEqual(policy(a,200,True),(a['cf'],a['body_gain'],a['jacobian']))
                self.assertEqual(policy(a,200),(1.,1.,'native'))
                self.assertEqual(policy(a,201,True),(1.,1.,'native'))
        self.assertTrue({199,200,201}<=set(evaluation_points(1500,25)))

if __name__=='__main__':unittest.main(verbosity=2)

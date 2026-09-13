import copy,json,unittest
from pathlib import Path
import numpy as np
import torch
from src.nets import VecMLPL
from src import edge_law_0905 as e
ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"configs/zero_attraction_learning_0913.yaml"
torch.set_num_threads(1)
class ZeroAttractionTests(unittest.TestCase):
 def setUp(self):
  self.old_config=e.CONFIG;self.old_table=e._TABLE
 def tearDown(self):
  e.CONFIG=self.old_config;e._TABLE=self.old_table
 def net(self,act,alpha=1.):
  return VecMLPL(1,[3],2,torch.Generator().manual_seed(1),"cpu",act=act,act_alpha=alpha)
 def test_derivatives_and_curvature(self):
  for name in VecMLPL.SNAKE_PHASE_0913:
   n=self.net(name,1.3)
   z=torch.linspace(-4.7,4.7,503,dtype=torch.float64,requires_grad=True)
   a=n.act_fn(z);d=torch.autograd.grad(a.sum(),z,create_graph=True)[0]
   dd=torch.autograd.grad(d.sum(),z)[0]
   torch.testing.assert_close(d,n.act_grad(z,a),atol=1e-12,rtol=1e-12)
   torch.testing.assert_close(dd,n.act_curv(z),atol=1e-12,rtol=1e-12)
   h=1e-5
   torch.testing.assert_close((n.act_fn(z+h)-n.act_fn(z-h))/(2*h),d,atol=1e-8,rtol=1e-8)
 def test_root_slopes_mutation(self):
  z=torch.tensor([0.],dtype=torch.float64)
  for name,expect in [("snake",1.),("snake_peak_0_0913",2.),("snake_valley_0_0913",0.)]:
   n=self.net(name)
   self.assertEqual(n.act_fn(z).item(),0.)
   self.assertEqual(n.act_grad(z,n.act_fn(z)).item(),expect)
  # If both variants silently fall through to normal Snake, this check fails.
  normal=self.net("snake").act_grad(z,z).item()
  self.assertNotEqual(normal,2.);self.assertNotEqual(normal,0.)
 def test_offsets_and_guards(self):
  z=torch.linspace(-7,7,100,dtype=torch.float64)
  for name,(phase,q) in VecMLPL.SNAKE_PHASE_0913.items():
   n=self.net(name)
   base=self.net("snake" if phase=="normal" else f"snake_{phase}_0_0913")
   torch.testing.assert_close(n.act_fn(z)-base.act_fn(z),torch.full_like(z,q),atol=1e-15,rtol=1e-14)
   self.assertTrue(torch.equal(n.act_grad(z,z),base.act_grad(z,z)))
   for bad in [0.,-1.,float("nan"),float("inf")]:
    with self.assertRaises(ValueError):self.net(name,bad)
 def test_initial_state_and_input_pairing(self):
  e.CONFIG=CFG;e._TABLE=None;cfg=e.build_cfg()
  states=[e.setup_arm_dial(copy.deepcopy(cfg),e._arm(cfg,a),"cpu") for a in e.arm_order()]
  self.assertEqual(len(states),19)
  ref=states[0]
  for st in states[1:]:
   for k,v in ref["net"].params().items():self.assertTrue(torch.equal(v,st["net"].params()[k]),k)
   self.assertTrue(torch.equal(ref["teacher"].W,st["teacher"].W))
  for step in range(30):
   xs=[s["env"].step() for s in states]
   for x in xs[1:]:self.assertTrue(torch.equal(xs[0],x))
 def test_affine_self_rest_error_direction(self):
  x=torch.tensor([[-.5,-.5],[-.5,.5],[.5,-.5],[.5,.5]],dtype=torch.float64)
  u=torch.tensor([.7,-.2],dtype=torch.float64);m=.3;z0=-.1;v=.8
  for s in [.1,1.,2.]:
   phi=s*(m+x@u-z0);gate=torch.full_like(phi,s)
   gself=v*v*(phi[:,None]*gate[:,None]*x).mean(0)
   torch.testing.assert_close(gself,(v*v*s*s/4)*u,atol=1e-14,rtol=1e-14)
   radial=[]
   for delta in [x@u,-x@u]:
    rest=delta-v*phi
    total=v*(delta[:,None]*gate[:,None]*x).mean(0)
    gr=v*(rest[:,None]*gate[:,None]*x).mean(0)
    torch.testing.assert_close(total,gself+gr,atol=1e-14,rtol=1e-14)
    radial.append(float(u@total))
   self.assertGreater(radial[0],0);self.assertLess(radial[1],0)
 def test_compensated_shift_total_vs_self(self):
  x=torch.tensor([[-1.,.3],[.5,-.2],[.2,.7],[-.3,-.5]],dtype=torch.float64)
  u=torch.tensor([.7,-.2],dtype=torch.float64,requires_grad=True)
  v=.8;b=.1;y=torch.tensor([.1,-.3,.4,-.2],dtype=torch.float64)
  z=x@u;b0=.13;phi=torch.where(z>0,z,.1*z);gate=torch.where(z>0,torch.ones_like(z),torch.full_like(z,.1))
  def calc(q):
   h=phi+q;delta=v*h+b0-q*v-y
   loss=.5*(delta*delta).mean()
   g=torch.autograd.grad(loss,u,retain_graph=True)[0]
   selfg=v*v*(h[:,None]*gate[:,None]*x).mean(0)
   restg=v*((delta-v*h)[:,None]*gate[:,None]*x).mean(0)
   torch.testing.assert_close(g,selfg+restg,atol=1e-14,rtol=1e-14)
   return g,selfg
  a,sa=calc(0.);b,sb=calc(.5)
  torch.testing.assert_close(a,b,atol=1e-14,rtol=1e-14)
  self.assertGreater(float(torch.norm(sa-sb).detach()),.01)
if __name__=="__main__":unittest.main()

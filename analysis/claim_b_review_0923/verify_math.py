"""Deterministic algebra/implementation checks, not a new learning benchmark."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from src.rlcifar_mlp_battle_0918 import SnakeFamily

OUT = Path('results/claim_b_review_0923/math_checks.json')
torch.set_default_dtype(torch.float64)
z = torch.linspace(-30, 30, 6001).reshape(1, -1, 1)
c = 0.6
act = SnakeFamily('KKT1', 'kkt1', widths=(1, 1))
act.init_state(1, 'cpu', None)
act.V = [v.double() for v in act.V]
base_phi, base_d = act.phi(z), act.dphi(z)
z_grad = z.clone().requires_grad_(True)
actual_grad, = torch.autograd.grad(act.phi(z_grad).sum(), z_grad)
autograd_error = (actual_grad - act.dphi(z_grad)).abs().max().item()
assert autograd_error < 1e-12
checks = {}
for scale in [0.5, 2.0, 10.0, 100.0]:
    act.V[0].fill_(scale ** 2)
    checks[str(scale)] = {
        'value_abs_error': (act.phi(scale*z)/scale - base_phi).abs().max().item(),
        'derivative_abs_error': (act.dphi(scale*z) - base_d).abs().max().item(),
        'alpha': act.alpha(0).item(),
    }
    assert checks[str(scale)]['value_abs_error'] < 1e-12
    assert checks[str(scale)]['derivative_abs_error'] < 1e-12

# Clipping and an unscaled initial EMA state violate the ideal scale identity.
act.V[0].fill_(1000.0**2)
clipped_error=(act.dphi(1000*z)-base_d).abs().max().item()
act.V[0].fill_(1.0)
lagged_error=(act.dphi(10*z)-base_d).abs().max().item()
assert clipped_error > 0.5 and lagged_error > 0.5

# Pointwise zero at theta=-pi/2; the far negative tail has slope one.
q=torch.tensor([-math.pi/(4*c), -100.0]).reshape(1, -1, 1)
g=act.dphi(q).flatten().tolist()
assert abs(g[0]) < 1e-15 and g[1] == 1.0
# Strong negative translation moves the whole grid into the linear tail.
translated=act.dphi(z-100).flatten()
assert torch.all(translated==1)

# Nonzero mean, asymmetric ReLU, changing tasks, and nonzero updates: no sinking.
# x~Uniform{1,2}; y_t=a_t*x; bias fixed zero; L=.5 E[(ReLU(wx)-y)^2].
# w_(t+1)=.75 w_t+.25 a_t, a_t alternates 1 and 2, so w stays positive.
w=0.5
ws=[w]
for t in range(1000):
    a=1.0 if t % 2 == 0 else 2.0
    w=0.75*w+0.25*a
    ws.append(w)
assert min(ws)==0.5 and max(ws)<2

# Increasing width at fixed normalized coordinates is not all-input failure.
tail=[]
for scale in [1., 10., 100.]:
    zz=torch.tensor([-2.,1.])*scale
    deriv=torch.where(zz>0, torch.ones_like(zz), zz.exp())
    tail.append({'scale':scale,'mean_z':zz.mean().item(),
                 'sd_z':zz.std(unbiased=False).item(),'ELU_mean_derivative':deriv.mean().item()})

# Ideal EMA scaling preserves equivariance if initial state and all batches scale.
v, vs = 1.0, 100.0
ema_errors=[]
for variance in [1., 3., .7, 8., 2.]:
    v=.99*v+.01*variance
    vs=.99*vs+.01*100*variance
    ema_errors.append(abs(vs-100*v))
assert max(ema_errors)<1e-12

result={
 'classification':'deterministic mathematical sanity checks; no new empirical training run',
 'implementation':'src/rlcifar_mlp_battle_0918.py:SnakeFamily; KKT1 theta in [-2pi,pi]',
 'scale_identity':checks,'autograd_vs_dphi_max_error':autograd_error,
 'clip_derivative_error':clipped_error,'EMA_lag_derivative_error':lagged_error,
 'KKT1_zero_and_negative_tail_derivatives':g,'translation_derivative_variance':translated.var().item(),
 'SGD_counterexample':{'input_mean':1.5,'initial_w':ws[0],'min_w':min(ws),'max_w':max(ws),
                      'last_two_w':ws[-2:],'updates':1000,'proof':'positive convex combination; invariant interval [0.5,2]'},
 'width_and_negative_mean_counterexample':tail,'scaled_EMA_max_error':max(ema_errors),
 'passed':True}
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))

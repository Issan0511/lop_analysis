"""Shuffled singleton Adam: local identities and a finite reference inequality.

No training trajectory, stationary-history Monte Carlo, or success-rate claim.
"""
from fractions import Fraction
from itertools import permutations, product
from pathlib import Path
import json
import torch
from verify_adam_three_images import (
    SHAPES, pack, unpack, forward, mean_direction, self_state, quantities,
)

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
EPS = 1e-8
N, EPOCHS, H = 3, 2, 6


def reference():
    gen = torch.Generator().manual_seed(10093)
    grid = torch.arange(12)[:,None]+12*torch.arange(12)[None,:]
    x0 = (.2+grid/256).expand(1,3,12,12).clone()
    p = [torch.zeros(s) for s in SHAPES]
    p[0] = 1e-5*(.8+.4*torch.rand(SHAPES[0],generator=gen))
    p[0][:,:,2,2] = .2*(.8+.4*torch.rand((2,3),generator=gen))
    p[2] = 1e-5*(.8+.4*torch.rand(SHAPES[2],generator=gen))
    p[2][:,:,2,2] = .3*(.8+.4*torch.rand((2,2),generator=gen))
    p[4] = .002*(.8+.4*torch.rand(SHAPES[4],generator=gen))
    p[4][torch.arange(3),torch.tensor([0,4,8])] += .8
    p[6] = .002*(.8+.4*torch.rand(SHAPES[6],generator=gen))+.8*torch.eye(3)
    for j in [1,3,5,7]:
        p[j] = .04*(.8+.4*torch.rand(SHAPES[j],generator=gen))
    d = torch.tensor([.10,.12,.09])
    p[8] = torch.stack((d,-d))
    raw = pack(p)
    hidden = forward(raw,x0,extra=True)[1][0]
    p[9] = torch.stack((-(d@hidden),d@hidden))
    raw = pack(p)
    a = torch.func.jacrev(lambda x:forward(raw,x.reshape_as(x0),extra=True)[1][0])(x0.flatten())
    ids = [39,91,143]
    beta = d@a[:,ids]
    v = torch.zeros(2,x0.numel())
    v[0,ids[0]],v[0,ids[2]] = 1/beta[0],-1/beta[2]
    v[1,ids[1]],v[1,ids[2]] = 1/beta[1],-1/beta[2]
    v /= v.abs().max()
    perturb = torch.stack((v[0],v[1],-v.sum(0))).reshape(3,3,12,12)
    return raw,x0+.01*perturb,x0


def scaled(raw, scale):
    p = unpack(raw,SHAPES)
    for j in [6,7,8]:
        p[j] = scale*p[j]
    p[9] = scale**2*p[9]
    return pack(p)


def finite_history_linearization(s):
    # Two tasks, two independently valid permutation epochs in each. Labels
    # repeat within each task; the 6 task/image labels are enumerated exactly.
    order = torch.tensor([2,0,1,0,2,1,1,2,0,2,0,1])
    task = torch.arange(12)//H
    group = task*N+order
    b1,b2 = .9,.999
    lag = torch.arange(11,-1,-1)
    a = (1-b1)*b1**lag/(1-b1**12)
    b = (1-b2)*b2**lag/(1-b2**12)
    ag,bg = torch.zeros(6),torch.zeros(6)
    ag.scatter_add_(0,group,a)
    bg.scatter_add_(0,group,b)
    an = ag.reshape(2,N).sum(0)
    gamma = (ag*bg).reshape(2,N).sum(0)
    sigma = torch.sqrt((b[:,None]*s[order]**2).sum(0))
    den = sigma+EPS
    expected = (s*(an[:,None]/den-s**2*gamma[:,None]/(sigma*den**2))).T
    labels = torch.tensor(list(product([-1.,1.],repeat=6)))

    def response(z):
        grad = s[order][None,:,:]*(torch.tanh(z[order])[None,:,None]-labels[:,group,None])
        m = (a[None,:,None]*grad).sum(1)
        v = (b[None,:,None]*grad**2).sum(1)
        return (m/(torch.sqrt(v)+EPS)).mean(0)

    actual = torch.func.jacrev(response)(torch.zeros(N))
    row_scale = expected.abs().max(1).values
    relative_error = ((actual-expected).abs().max(1).values/row_scale).max()
    assert relative_error<2e-12
    coefficient = expected.T/s
    lower = an[:,None]*EPS/den**2
    assert (coefficient>=lower*(1-1e-12)).all()
    return dict(label_histories=64,optimizer_steps=12,
                maximum_relative_coordinate_error=float(relative_error),
                maximum_absolute_coordinate_error=float((actual-expected).abs().max()),
                interpretation='Finite normalized EMA history identity with exact label enumeration; not the infinite stationary coefficient.')


def bias_stationary_coefficients():
    # At equal sensitivities the squared gradients are label/schedule invariant.
    # The remaining covariance is an exact rational grouped-weight moment.
    b1,b2 = Fraction(9,10),Fraction(999,1000)
    schedules = [a+b for a,b in product(list(permutations(range(N))),repeat=EPOCHS)]

    def omega_finite(length):
        total = Fraction(0)
        for order in schedules:
            a,b = [Fraction(0)]*N,[Fraction(0)]*N
            for i,n in enumerate(order[:length]):
                a[n] += (1-b1)*b1**(length-1-i)
                b[n] += (1-b2)*b2**(length-1-i)
            total += sum(x*y for x,y in zip(a,b))
        return total/len(schedules)

    full = omega_finite(H)
    omega = [omega_finite(r)+(b1*b2)**r*full/(1-(b1*b2)**H) for r in range(1,H+1)]
    s,eps = Fraction(1,2),Fraction(1,100000000)
    coeff = sum((eps+s*(1-w))/(N*(eps+s)**2) for w in omega)
    assert all(0<w<1 for w in omega) and coeff>0
    return dict(phase_omega_rational=[str(w) for w in omega],
                positive_common_mode_coefficient_rational=str(2*s*s*coeff),
                positive_common_mode_coefficient=float(2*s*s*coeff),
                convention='J L output-bias block equals this coefficient times ones(3,3); phase sum, H=6.')


def scaled_capacity(info, scale):
    # The final two raw columns are the unscaled output biases. Their
    # directional derivatives are zero, so normalize the other blocks before
    # forming the very small derivative. This avoids a spurious eigenvalue
    # calculation for a tiny perturbation of the rank-two bias kernel.
    jr,jpr = info['j'][:,:-2]/scale,info['jp'][:,:-2]/scale
    kb = info['j'][:,-2:]@info['j'][:,-2:].T
    k1,kp1 = jr@jr.T,jpr@jr.T+jr@jpr.T
    normalized_slope = .5*torch.trace(torch.linalg.solve(torch.eye(6)+kb+scale**2*k1,kp1))
    assert normalized_slope>0
    return dict(capacity_slope_divided_by_scale_squared=float(normalized_slope),
                capacity_slope=float(scale**2*normalized_slope))


def main():
    raw,images,x0 = reference()
    scale = 1e-18
    point = scaled(raw,scale)
    info = quantities(point,images)
    own,oshapes = self_state(point)
    selfinfo = quantities(own,images,oshapes)
    # Scale-free form of the stationary normal stability inequality.
    jr = info['s'][:,:-2]
    normalized = jr/scale
    singular = torch.linalg.svdvals(normalized)
    sigma_min,op = singular[-1],singular[0]
    fro = torch.linalg.vector_norm(normalized)
    max_s = jr.abs().max()
    relative_error_bound = 2*(max_s/EPS)*op*fro/(sigma_min**2)
    assert relative_error_bound<.5
    positive_lower = H/(N*EPS)*scale**2*sigma_min**2*(1-relative_error_bound)
    assert positive_lower>0
    target = info['u']!=0
    assert (info['s'][:,target]>0).all()
    q = info['s'].abs().max(0).values
    ray_lower = H/N*(info['s'][:,target]*info['u'][target]*EPS/(EPS+q[target])**2).sum(1)
    assert (ray_lower>0).all()
    assert info['s'].abs().min()>0
    assert info['contrast'].abs().max()<1e-48
    assert torch.linalg.svdvals(info['h']/scale)[-1]>1e-7
    assert min(info['minimum_pool_gap'],selfinfo['minimum_pool_gap'])>0
    finite = finite_history_linearization(info['s'])
    exact_bias = bias_stationary_coefficients()
    out = dict(scope='Finite native rank/stability inequalities and exact finite-history identities; no actual long-time simulation or certified practical learning rate.',
               architecture='RGB12x12 Conv5(2) Pool2 Conv5(2) Pool2 FC3 ReLU FC3 ReLU binary head, all biases',
               N=N,batch_size=1,epochs_per_task=EPOCHS,H=H,raw_parameters=point.numel(),
               reference_layer_scale=scale,epsilon=EPS,
               hidden_scaled_singular_values=torch.linalg.svdvals(info['h']/scale).tolist(),
               non_output_bias_scaled_jacobian_singular_values=singular.tolist(),
               maximum_non_output_bias_sensitivity=float(max_s),
               minimum_absolute_raw_sensitivity=float(info['s'].abs().min()),
               stability_relative_error_upper=float(relative_error_bound),
               symmetric_normal_matrix_lower_bound=float(positive_lower),
               positive_normal_ray_coefficient_lower=float(ray_lower@ray_lower),
               maximum_absolute_reference_contrast=float(info['contrast'].abs().max()),
               minimum_full_hidden_preactivation=min(float(z.min()) for z in info['zs']),
               full_minimum_pool_gap=info['minimum_pool_gap'],
               self_minimum_pool_gap=selfinfo['minimum_pool_gap'],
               capacity_full=scaled_capacity(info,scale),capacity_literal_self=scaled_capacity(selfinfo,scale),
               finite_history_linearization=finite,stationary_output_bias=exact_bias,
               interpretation='The finite reference is extremely small and rates are not numerically certified. Floating-point inequalities accompany the analytic nonempty scaling construction. The exact bias coefficient is rational; other margins are float64 checks.')
    path=Path(__file__).resolve().parents[2]/'results/cnn_drive_1009/adam_singleton.json'
    path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    main()

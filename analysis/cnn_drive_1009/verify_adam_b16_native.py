"""Native N=32, B=16 reference: finite raw identities and normalized bounds.

Standalone finite verifier; saves results/cnn_drive_1009/adam_b16_native.json.
No training trajectory, stationary-history estimation, probability measurement,
or enumeration of all 2**32 label assignments is performed.
"""
from pathlib import Path
import json
import math
import time

import torch
from torch.nn import functional as F

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
N, BATCH, EPOCHS, H = 32, 16, 2, 4
EPS = 1e-8
SHAPES = [(2, 3, 5, 5), (2,), (2, 2, 5, 5), (2,),
          (32, 72), (32,), (32, 32), (32,), (2, 32), (2,)]


def pack(parts):
    return torch.cat([p.flatten() for p in parts])


def unpack(theta, shapes=SHAPES):
    parts, offset = [], 0
    for shape in shapes:
        count = math.prod(shape)
        parts.append(theta[offset:offset + count].reshape(shape))
        offset += count
    assert offset == theta.numel()
    return parts


def forward(theta, images, shapes=SHAPES, extra=False):
    w1,b1,w2,b2,w3,b3,w4,b4,w5,b5 = unpack(theta, shapes)
    z1 = F.conv2d(images, w1, b1, padding=2)
    h1,i1 = F.max_pool2d(F.relu(z1), 2, return_indices=True)
    z2 = F.conv2d(h1, w2, b2, padding=2)
    h2,i2 = F.max_pool2d(F.relu(z2), 2, return_indices=True)
    z3 = F.linear(h2.flatten(1), w3, b3)
    z4 = F.linear(F.relu(z3), w4, b4)
    hidden = F.relu(z4)
    logits = F.linear(hidden, w5, b5)
    return (logits,hidden,(z1,z2,z3,z4),(i1,i2)) if extra else logits


def mean_direction(images, shapes=SHAPES):
    parts = [torch.zeros(s) for s in shapes]
    parts[0][0] = F.unfold(images, 5, padding=2).mean((0,2)).reshape(3,5,5)
    parts[1][0] = 1.
    return pack(parts)


def self_state(theta):
    p = unpack(theta)
    p[0],p[1],p[2] = p[0][:1],p[1][:1],p[2][:,:1]
    return pack(p), [tuple(x.shape) for x in p]


def jacobian(theta, images, shapes=SHAPES):
    fn = lambda t: forward(t, images, shapes).flatten()
    return torch.func.jacrev(fn)(theta)


def contrast_jacobian(theta, images, shapes=SHAPES):
    def contrast(t):
        f = forward(t, images, shapes)
        return (f[:,0]-f[:,1])/2
    return torch.func.jacrev(contrast)(theta)


def routing_info(theta, images, shapes=SHAPES):
    f,h,zs,inds = forward(theta, images, shapes, True)
    gaps = []
    for z in zs[:2]:
        windows = z.unfold(2,2,2).unfold(3,2,2).flatten(-2)
        top = windows.topk(2, dim=-1).values
        gaps.append(float((top[...,0]-top[...,1]).min()))
    return dict(logits=f, hidden=h, inds=inds,
                min_preactivation=min(float(z.min()) for z in zs),
                pool_gaps=gaps,
                contrast=(f[:,0]-f[:,1])/2)


def reference():
    gen = torch.Generator().manual_seed(100916)
    r,c = torch.arange(24)[:,None],torch.arange(24)[None,:]
    pattern = (.2+.1*(r%2)+.2*(c%2)+.1*((r//2)%2)
               +.15*((c//2)%2)+.001*(r+24*c)/(24*24))
    x0 = pattern.expand(1,3,24,24).clone()
    p = [torch.zeros(s) for s in SHAPES]
    p[0] = 1e-6*(.8+.4*torch.rand(SHAPES[0],generator=gen))
    p[0][:,:,2,2] = .2*(.8+.4*torch.rand((2,3),generator=gen))
    p[2] = 1e-6*(.8+.4*torch.rand(SHAPES[2],generator=gen))
    p[2][:,:,2,2] = .3*(.8+.4*torch.rand((2,2),generator=gen))
    p[4] = 2e-5*(.8+.4*torch.rand(SHAPES[4],generator=gen))
    p[4][torch.arange(N),torch.arange(N)] += .8
    p[6] = 2e-5*(.8+.4*torch.rand(SHAPES[6],generator=gen))+.8*torch.eye(N)
    for k in [1,3,5,7]:
        p[k] = .04*(.8+.4*torch.rand(SHAPES[k],generator=gen))
    d = .1+.02*torch.rand(N,generator=gen)
    p[8] = torch.stack((d,-d))
    theta = pack(p)
    h0 = forward(theta, x0, extra=True)[1][0]
    p[9] = torch.stack((-(d@h0),d@h0))
    theta = pack(p)
    ax = torch.func.jacrev(
        lambda x: forward(theta,x.reshape_as(x0),extra=True)[1][0]
    )(x0.flatten())
    ids = torch.tensor([(4*(i//6)+3)*24+(4*(i%6)+3) for i in range(N)])
    selected = ax[:,ids]
    singular = torch.linalg.svdvals(selected)
    assert singular[-1]>1e-3
    beta = d@selected
    assert beta.min()>0
    simplex = (torch.eye(N)-torch.ones(N,N)/N)/beta[None,:]
    normalization = float(simplex.abs().max())
    simplex /= normalization
    perturb = torch.zeros(N,x0.numel())
    perturb[:,ids] = simplex
    assert (simplex@beta).abs().max()<1e-14
    assert simplex.sum(0).abs().max()<1e-14
    return theta,x0,perturb.reshape(N,3,24,24),dict(
        selected_input_columns=ids.tolist(),
        selected_input_jacobian_singular_values=singular.tolist(),
        simplex_normalization=normalization,
        simplex_level_error=float((simplex@beta).abs().max()),
        simplex_column_sum_error=float(simplex.sum(0).abs().max()))


def scaled(theta, scale):
    p = unpack(theta)
    for k in [6,7,8]:
        p[k] = scale*p[k]
    p[9] = scale**2*p[9]
    return pack(p)


def normalized_capacity(theta, images, scale, shapes=SHAPES):
    u = mean_direction(images, shapes)
    fn = torch.func.jacrev(lambda t:forward(t,images,shapes).flatten())
    j,jp = torch.func.jvp(fn,(theta,),(u,))
    assert jp[:,-2:].abs().max()==0
    r,rp = j[:,:-2]/scale,jp[:,:-2]/scale
    kb = j[:,-2:]@j[:,-2:].T
    kr,krp = r@r.T,rp@r.T+r@rp.T
    identity = torch.eye(2*N)
    normalized = .5*torch.trace(torch.linalg.solve(identity+kb+scale**2*kr,krp))
    limiting = .5*torch.trace(torch.linalg.solve(identity+kb,krp))
    # Norm bound on the denominator correction at the chosen finite scale.
    denominator_error = .5*scale**2*torch.linalg.matrix_norm(kr,ord=2)*torch.linalg.matrix_norm(krp,ord='nuc')
    assert normalized>0 and limiting-denominator_error>0
    return dict(capacity_slope_divided_by_scale_squared=float(normalized),
                capacity_slope=float(scale**2*normalized),
                unscaled_bias_kernel_rank=int(torch.linalg.matrix_rank(kb)),
                denominator_correction_upper=float(denominator_error),
                normalized_positive_margin=float(limiting-denominator_error)),j


def batch_gradient_checks(theta, images, sensitivity):
    gen = torch.Generator().manual_seed(160032)
    labels = [torch.zeros(N,dtype=torch.long),torch.arange(N)%2,
              (torch.arange(N)>=N//2).to(torch.long),
              torch.randint(0,2,(N,),generator=gen)]
    envelope = sensitivity.abs().max(0).values
    assert envelope.min()>0
    checks = []
    for task,y in enumerate(labels):
        orders = [torch.randperm(N,generator=gen) for _ in range(EPOCHS)]
        phase_checks = []
        for epoch,order in enumerate(orders):
            for batch_number,indices in enumerate(order.split(BATCH)):
                logits = forward(theta,images[indices])
                contrast = (logits[:,0]-logits[:,1])/2
                raw_gradient = torch.func.grad(
                    lambda t:F.cross_entropy(forward(t,images[indices]),y[indices])
                )(theta)
                expected = sensitivity[indices].T@(
                    torch.tanh(contrast)-(1-2*y[indices]))/BATCH
                absolute = (raw_gradient-expected).abs()
                normalized = float((absolute/envelope).max())
                assert normalized<2e-11
                phase_checks.append(dict(epoch=epoch,batch=batch_number,
                    class1_count=int(y[indices].sum()),
                    maximum_absolute_error=float(absolute.max()),
                    maximum_sensitivity_normalized_error=normalized))
        checks.append(dict(labels=y.tolist(),permutations=[a.tolist() for a in orders],
                           phases=phase_checks))
    return checks


def main():
    started = time.monotonic()
    raw,x0,perturb,construction = reference()
    base = routing_info(raw,x0)
    base_s = contrast_jacobian(raw,x0)
    assert base_s.abs().min()>0
    own0,oshapes = self_state(raw)
    selfbase = routing_info(own0,x0,oshapes)
    delta = .002
    for _ in range(20):
        images = x0+delta*perturb
        info = routing_info(raw,images)
        selfinfo = routing_info(own0,images,oshapes)
        s = contrast_jacobian(raw,images)
        closeness = float(((s-base_s).abs()/base_s.abs()).max())
        same = all(torch.equal(a,a0.expand_as(a))
                   for cur,old in [(info,base),(selfinfo,selfbase)]
                   for a,a0 in zip(cur['inds'],old['inds']))
        if (images.min()>0 and images.max()<1 and same
                and closeness<1/(2*BATCH)):
            break
        delta /= 2
    else:
        raise AssertionError('Could not certify image-routing/sensitivity margins.')
    assert torch.linalg.svdvals(info['hidden'])[-1]>1e-7
    assert info['contrast'].abs().max()<1e-12
    u,ubase = mean_direction(images),mean_direction(x0)
    assert torch.allclose(images.mean(0),x0[0],atol=1e-14,rtol=0)
    assert torch.allclose(u,ubase,atol=1e-14,rtol=0)
    # Group R1 has raw order-t columns. Do not use the order-one output biases.
    masks = [torch.full(shape,k in [6,7,8],dtype=torch.bool) for k,shape in enumerate(SHAPES)]
    group1 = torch.cat([m.flatten() for m in masks])
    s1 = s[:,group1]
    sv1 = torch.linalg.svdvals(s1)
    cb = 1+math.sqrt(N/BATCH)
    # Set target relative error to 0.1 using the leading normalized matrix.
    scale = float(.1*EPS*sv1[-1]**2/(cb*s1.abs().max()*sv1[0]*torch.linalg.vector_norm(s1)))
    scale = min(scale,1e-6)
    for _ in range(10):
        point = scaled(raw,scale)
        current = routing_info(point,images)
        ss = contrast_jacobian(point,images)
        rr = ss[:,:-2]
        normalized = rr/scale
        sv = torch.linalg.svdvals(normalized)
        fro = torch.linalg.vector_norm(normalized)
        gamma = rr.abs().max()
        relative = cb*(gamma/EPS)*sv[0]*fro/(sv[-1]**2)
        if relative<.2:
            break
        scale /= 2
    else:
        raise AssertionError('Could not obtain the normalized stability margin.')
    c0 = H/(N*EPS)
    lower = c0*scale**2*sv[-1]**2*(1-relative)
    assert lower>0 and ss.abs().min()>0
    target = u!=0
    assert (ss[:,target]>0).all()
    gamma_coordinate = ss.abs().max(0).values
    rho_target = cb*gamma_coordinate[target]/EPS
    assert (rho_target<.5).all()
    ray = c0*(ss[:,target]*u[target]*(1-rho_target)).sum(1)
    assert ray.min()>0
    assert current['contrast'].abs().max()<1e-12*scale**2
    assert current['min_preactivation']>0
    fullcap,fullj = normalized_capacity(point,images,scale)
    own,oshapes = self_state(point)
    selfcap,_ = normalized_capacity(own,images,scale,oshapes)
    selfcurrent = routing_info(own,images,oshapes)
    assert selfcurrent['min_preactivation']>0
    assert min(current['pool_gaps']+selfcurrent['pool_gaps'])>0
    jacobian_identity_error = float(((fullj.reshape(N,2,-1)[:,0]-fullj.reshape(N,2,-1)[:,1])/2-ss).abs().max())
    assert jacobian_identity_error<1e-12
    batch_checks = batch_gradient_checks(point,images,ss)
    out = dict(
        scope='Finite native raw identities and normalized local bounds only; no training trajectory, stationary response measurement, learning-rate certificate, or enumeration of all 2^32 label assignments.',
        architecture='RGB24x24 Conv5(2) Pool2 Conv5(2) Pool2 FC32 ReLU FC32 ReLU binary head; all 3712 raw weights/biases free',
        N=N,batch_size=BATCH,epochs_per_task=EPOCHS,H=H,
        raw_parameter_count=raw.numel(),epsilon=EPS,ridge=1.,
        image_separation=delta,maximum_pixel_displacement=float((images-x0).abs().max()),
        reference_layer_scale=scale,construction=construction,
        image_range=[float(images.min()),float(images.max())],
        mean_input_error=float((images.mean(0)-x0[0]).abs().max()),
        actual_mean_direction_error=float((u-ubase).abs().max()),
        maximum_relative_unscaled_sensitivity_perturbation=closeness,
        required_relative_sensitivity_threshold=1/(2*BATCH),
        hidden_scaled_singular_values=torch.linalg.svdvals(current['hidden']/scale).tolist(),
        non_output_bias_scaled_jacobian_singular_values=sv.tolist(),
        maximum_non_output_bias_sensitivity=float(gamma),
        minimum_absolute_raw_sensitivity=float(ss.abs().min()),
        batch_response_relative_constant=cb,
        stability_relative_error_upper=float(relative),
        symmetric_normal_matrix_lower_bound_conditional_on_nonnegative_bias=float(lower),
        positive_normal_ray_coefficient_lower=float(ray@ray),
        stationary_output_bias='Not evaluated here; the full normal bound uses the separate nonnegative stationary common-mode theorem.',
        maximum_absolute_reference_contrast=float(current['contrast'].abs().max()),
        maximum_contrast_divided_by_scale_squared=float(current['contrast'].abs().max()/scale**2),
        full_minimum_hidden_preactivation=current['min_preactivation'],
        self_minimum_hidden_preactivation=selfcurrent['min_preactivation'],
        full_pool_gaps=current['pool_gaps'],self_pool_gaps=selfcurrent['pool_gaps'],
        sample_class_jacobian_identity_error=jacobian_identity_error,
        capacity_full=fullcap,capacity_literal_self=selfcap,
        finite_batch_ce_checks=batch_checks,
        interpretation='The layer scale is selected from the relative spectral inequality and is extremely small. All non-rational numerical margins are float64 checks accompanying the analytic construction. Every recorded task uses one fixed label vector and two separately shuffled epochs.',
        elapsed_seconds=time.monotonic()-started)
    destination = Path(__file__).resolve().parents[2]/'results/cnn_drive_1009/adam_b16_native.json'
    destination.write_text(json.dumps(out,indent=2)+'\n')
    summary = {key:out[key] for key in ['N','batch_size','H','raw_parameter_count','image_separation','reference_layer_scale',
        'maximum_relative_unscaled_sensitivity_perturbation','stability_relative_error_upper',
        'symmetric_normal_matrix_lower_bound_conditional_on_nonnegative_bias','positive_normal_ray_coefficient_lower',
        'capacity_full','capacity_literal_self','elapsed_seconds']}
    summary['max_batch_ce_normalized_error'] = max(p['maximum_sensitivity_normalized_error'] for task in batch_checks for p in task['phases'])
    summary['output']=str(destination)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()

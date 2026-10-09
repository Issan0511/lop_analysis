"""Finite native N1200 reference checks without an N1200-by-P full Jacobian.

The 1200-by-1200 first-Conv mixed minor is assembled from one backpropagated
preactivation sensitivity. All-image rank/routing/sensitivity bounds are
analytic consequences of that finite construction. Raw autodiff identities
and capacity checks use only the explicitly listed three-image subset.
No training, Monte Carlo, or stationary-response approximation is performed.
"""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from time import monotonic
import json
import math

import torch
from torch.nn import functional as F


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
N, BATCH, EPOCHS, H = 1200, 16, 400, 30000
EPSILON, RIDGE = 1e-8, 1.0
SHAPES = [(16,3,5,5), (16,), (16,16,5,5), (16,),
          (100,1024), (100,), (100,100), (100,), (2,100), (2,)]
P = sum(math.prod(s) for s in SHAPES)
assert P == 120434


def pack(parts):
    return torch.cat([p.reshape(-1) for p in parts])


def unpack(theta, shapes=SHAPES):
    parts, offset = [], 0
    for shape in shapes:
        size = math.prod(shape)
        parts.append(theta[offset:offset+size].reshape(shape))
        offset += size
    assert offset == theta.numel()
    return parts


def forward(theta, images, shapes=SHAPES, extra=False):
    w1,b1,w2,b2,w3,b3,w4,b4,w5,b5 = unpack(theta, shapes)
    z1 = F.conv2d(images,w1,b1,padding=2)
    h1,i1 = F.max_pool2d(F.relu(z1),2,return_indices=True)
    z2 = F.conv2d(h1,w2,b2,padding=2)
    h2,i2 = F.max_pool2d(F.relu(z2),2,return_indices=True)
    z3 = F.linear(h2.flatten(1),w3,b3)
    h3 = F.relu(z3)
    z4 = F.linear(h3,w4,b4)
    h4 = F.relu(z4)
    logits = F.linear(h4,w5,b5)
    return (logits,(z1,z2,z3,z4),(h1,h2,h3,h4),(i1,i2)) if extra else logits


def contrast(theta, images, shapes=SHAPES):
    f = forward(theta,images,shapes)
    return (f[:,0]-f[:,1])/2


def mean_direction(images, shapes=SHAPES):
    parts = [torch.zeros(s) for s in shapes]
    parts[0][0] = F.unfold(images,5,padding=2).mean((0,2)).reshape(3,5,5)
    parts[1][0] = 1.
    return pack(parts)


def literal_self(theta):
    p = unpack(theta)
    p[0],p[1],p[2] = p[0][:1],p[1][:1],p[2][:,:1]
    return pack(p), [tuple(x.shape) for x in p]


def scale_state(theta, scale, shapes=SHAPES):
    p = unpack(theta, shapes)
    for k in (0,1,3,5,7,8):
        p[k] = scale*p[k]
    p[9] = scale**2*p[9]
    return pack(p)


def parameter_masks(shapes=SHAPES):
    group1 = pack([torch.full(s,k in (0,1,3,5,7,8),dtype=torch.bool)
                   for k,s in enumerate(shapes)])
    group2 = pack([torch.full(s,k in (2,4,6),dtype=torch.bool)
                   for k,s in enumerate(shapes)])
    head = pack([torch.full(s,k == 8,dtype=torch.bool)
                 for k,s in enumerate(shapes)])
    bias = pack([torch.full(s,k == 9,dtype=torch.bool)
                 for k,s in enumerate(shapes)])
    assert (group1 | group2 | bias).all()
    assert not (group1 & group2).any()
    return group1,group2,head,bias


def reference():
    r,c = torch.arange(32)[:,None],torch.arange(32)[None,:]
    pattern = (.2+.1*(r%2)+.2*(c%2)+.1*((r//2)%2)
               +.15*((c//2)%2)+.001*(r+32*c)/(32*32))
    x0 = pattern.expand(1,3,32,32).clone()
    eta = 1e-6
    p = [torch.zeros(s) for s in SHAPES]
    p[0].fill_(eta)
    p[0][:,:,2,2] = .2
    p[2].fill_(eta**2)
    p[2][:,:,2,2] = eta
    p[2][torch.arange(16),torch.arange(16),2,2] = .7
    centers = [(r,c) for r in (3,11,19,27) for c in (3,11,19,27)]
    p[4].fill_(eta)
    sites = torch.tensor([i*64+(r//4)*8+c//4
                          for i,(r,c) in enumerate(centers)])
    p[4][torch.arange(16),sites] = .8
    p[6].fill_(eta)
    p[6][torch.arange(16),torch.arange(16)] = .9
    for k in (1,3,5,7):
        p[k].fill_(.04)
    d = .1+.01*torch.arange(100)/100
    p[8] = torch.stack((d,-d))
    raw = pack(p)
    h0 = forward(raw,x0,extra=True)[2][-1][0]
    p[9] = torch.stack((-(d@h0),d@h0))
    raw = pack(p)
    pixel_ids, out_index, color_index, dr, dc = [],[],[],[],[]
    for i,(r,c) in enumerate(centers):
        for channel in range(3):
            for a in range(-2,3):
                for b in range(-2,3):
                    pixel_ids.append(channel*1024+(r+a)*32+c+b)
                    out_index.append(i)
                    color_index.append(channel)
                    dr.append(a)
                    dc.append(b)
    assert len(pixel_ids) == N and len(set(pixel_ids)) == N
    return raw,x0,centers,tuple(torch.tensor(v,dtype=torch.long) for v in
                              (pixel_ids,out_index,color_index,dr,dc))


def base_backprop(raw,x0):
    theta = raw.detach().clone().requires_grad_(True)
    x = x0.detach().clone().requires_grad_(True)
    f,zs,_,_ = forward(theta,x,extra=True)
    z = (f[0,0]-f[0,1])/2
    q = torch.autograd.grad(z,zs[0],retain_graph=True)[0][0]
    grad_theta,grad_input = torch.autograd.grad(z,(theta,x))
    assert grad_theta.abs().min()>0
    return q.detach(),grad_theta.detach(),grad_input.detach()


def mixed_minor(q,meta):
    pixel_ids,out_index,color_index,dr,dc = meta
    pixel_color = pixel_ids//1024
    pixel_row = (pixel_ids%1024)//32
    pixel_col = pixel_ids%32
    rr = pixel_row[None,:]-dr[:,None]
    cc = pixel_col[None,:]-dc[:,None]
    valid = ((rr>=0)&(rr<32)&(cc>=0)&(cc<32)
             &(color_index[:,None]==pixel_color[None,:]))
    matrix = q[out_index[:,None],rr.clamp(0,31),cc.clamp(0,31)]
    matrix = matrix*valid
    diagonal = matrix.diag()
    row_margin = (2*diagonal-matrix.abs().sum(1)).min()
    column_margin = (2*diagonal-matrix.abs().sum(0)).min()
    assert matrix.min()>=0 and diagonal.min()>0
    assert row_margin>0 and column_margin>0
    return matrix,float(row_margin),float(column_margin)


def routing_bounds(theta,x0,delta,shapes=SHAPES):
    parts = unpack(theta,shapes)
    _,zs,_,indices = forward(theta,x0,shapes,True)
    l1 = parts[0].abs().sum((1,2,3))
    l2 = parts[2].abs().sum((2,3))@l1
    gaps, reserves = [],[]
    for z,lip in zip(zs[:2],(l1,l2)):
        windows = z.unfold(2,2,2).unfold(3,2,2).flatten(-2)
        values = windows.topk(2,dim=-1).values
        gap = (values[...,0]-values[...,1]).flatten(2).min(2).values[0]
        reserve = gap-2*delta*lip
        assert reserve.min()>0
        gaps.append(gap.tolist())
        reserves.append(reserve.tolist())
    for k in (0,1,2,3,4,5,6,7):
        assert (parts[k]>0).all()
    return dict(
        first_conv_lipschitz_per_channel=l1.tolist(),
        second_conv_lipschitz_per_channel=l2.tolist(),
        base_pool_gaps_per_channel=gaps,
        all_image_pool_gap_lower_per_channel=reserves,
        minimum_all_image_pool_gap_lower=min(min(r) for r in reserves),
        positive_hidden_bias_lower=min(float(parts[k].min()) for k in (1,3,5,7)),
        base_minimum_preactivation=min(float(z.min()) for z in zs),
    ),indices


def subset_images(x0,pixel_ids,T,delta,ids):
    rows = x0.flatten().repeat(len(ids),1)
    rows[:,pixel_ids] += delta*T[ids]
    return rows.reshape(len(ids),3,32,32)


def raw_scale_and_capacity_subset(raw,point,images,u,scale,shapes=SHAPES):
    fn = torch.func.jacrev(lambda theta:forward(theta,images,shapes).flatten())
    base_j,base_jp = torch.func.jvp(fn,(raw,),(u,))
    current_j,current_jp = torch.func.jvp(fn,(point,),(u,))
    group1,group2,head,bias = parameter_masks(shapes)
    scales = torch.ones(raw.numel())
    scales[group1],scales[group2] = scale,scale**2
    expected = base_j*scales[None,:]
    envelope = expected.abs().max(0).values
    assert envelope.min()>0
    jacobian_error = float(((current_j-expected).abs()/envelope).max())
    assert jacobian_error<2e-10
    zero = (group1 & ~head) | bias
    assert base_jp[:,zero].abs().max()==0
    assert current_jp[:,zero].abs().max()==0
    jp_scales = torch.ones(raw.numel())
    jp_scales[group2] = scale
    expected_jp = base_jp*jp_scales[None,:]
    jp_envelope = expected_jp.abs().max(0).values
    nonzero = jp_envelope>0
    derivative_error = float(
        ((current_jp[:,nonzero]-expected_jp[:,nonzero]).abs()
         /jp_envelope[nonzero]).max())
    assert derivative_error<2e-10
    assert current_jp[:,~nonzero].abs().max()==0
    r = current_j[:,~bias]/scale
    rp = current_jp[:,~bias]
    kb = current_j[:,bias]@current_j[:,bias].T
    derivative_over_t = rp@r.T+r@rp.T
    kernel = kb+scale**2*(r@r.T)
    eye = RIDGE*torch.eye(images.shape[0]*2)
    capacity_over_t = .5*torch.trace(torch.linalg.solve(eye+kernel,derivative_over_t))
    assert capacity_over_t>0
    return dict(
        image_count=int(images.shape[0]),
        raw_parameter_count=raw.numel(),
        maximum_raw_jacobian_scale_relative_error=jacobian_error,
        maximum_mean_direction_jacobian_derivative_scale_relative_error=derivative_error,
        first_conv_and_hidden_bias_jacobian_mean_derivative_exactly_zero=True,
        capacity_slope_divided_by_scale=float(capacity_over_t),
        capacity_slope=float(scale*capacity_over_t),
        ridge=RIDGE,
        scope="Only the named finite image subset; not the full N1200 capacity.",
    )


def coincident_capacity_coefficient(raw,x0,u,shapes=SHAPES):
    hidden = lambda theta:forward(theta,x0,shapes,True)[2][-1]
    h,dh = torch.func.jvp(hidden,(raw,),(u,))
    inner = float((h*dh).sum())
    assert inner>0
    return dict(
        hidden_inner_product_with_actual_mean_derivative=inner,
        coefficient_2N_inner_over_ridge_plus_N=2*N*inner/(RIDGE+N),
        scope="Identical-image delta=0 leading coefficient of capacity slope/t; continuity input only.",
    )


def verify():
    started = monotonic()
    raw,x0,centers,meta = reference()
    pixel_ids,_,_,_,_ = meta
    q,base_gradient,input_gradient = base_backprop(raw,x0)
    g0 = base_gradient[:N]
    unfolded = F.unfold(x0,5,padding=2)[0]
    g_from_q = (q.flatten(1)@unfolded.T).flatten()
    g_formula_error = float(((g_from_q-g0).abs()/g0.abs()).max())
    assert g_formula_error<1e-12 and (g0>0).all()
    matrix,row_margin,column_margin = mixed_minor(q,meta)
    singular_m_lower = math.sqrt(row_margin*column_margin)
    w1 = raw[:N]
    beta = matrix.T@w1
    assert beta.min()>0
    beta_autograd = input_gradient.flatten()[pixel_ids]
    beta_error = float(((beta-beta_autograd).abs()/beta).max())
    assert beta_error<1e-11
    normalization = float((1-1/N)/beta.min())
    d = 1/(normalization*beta)
    T = (torch.eye(N)-torch.ones(N,N)/N)*d[None,:]
    assert T.abs().max()<=1+1e-14
    column_sum_error = float(T.sum(0).abs().max())
    level_error = float((T@beta).abs().max())
    assert column_sum_error<1e-12 and level_error<1e-16
    delta = .001
    assert float(x0.min())-delta>0 and float(x0.max())+delta<1
    relative_sensitivity_bound = delta/float(x0.min())
    assert relative_sensitivity_bound<1/(2*BATCH)
    Q0 = float(g0@w1)
    assert Q0>0
    residual = g0-delta*(matrix@d)/N
    inverse_g_upper = (
        (1+float(torch.linalg.vector_norm(w1)*torch.linalg.vector_norm(residual))/Q0)
        *normalization*float(beta.max())/(delta*singular_m_lower)
    )
    singular_g_lower = 1/inverse_g_upper
    assert singular_g_lower>0
    lambda1_lower = singular_g_lower**2
    full_routing,full_indices = routing_bounds(raw,x0,delta)
    self_raw,self_shapes = literal_self(raw)
    self_routing,self_indices = routing_bounds(self_raw,x0,delta,self_shapes)
    u = mean_direction(x0)
    mean_image = x0.flatten().clone()
    mean_image[pixel_ids] += delta*T.sum(0)/N
    mean_direction_error = float((mean_direction(mean_image.reshape_as(x0))-u).abs().max())
    assert mean_direction_error<1e-14
    multiplier = 1+relative_sensitivity_bound
    gamma_upper = multiplier*float(base_gradient[:-2].abs().max())
    C_squared_upper = N*multiplier**2*float(base_gradient[:-2].square().sum())
    cb = 1+math.sqrt(N/BATCH)
    scale = min(1e-6,EPSILON*lambda1_lower/(4*cb*gamma_upper*C_squared_upper))
    assert scale>0
    error_ratio_upper = cb*scale*gamma_upper*C_squared_upper/(EPSILON*lambda1_lower)
    assert error_ratio_upper<=.250000000000001
    c0 = H/(N*EPSILON)
    normal_lower = c0*scale**2*lambda1_lower*(1-error_ratio_upper)
    assert normal_lower>0
    target = u!=0
    assert (u[target]>0).all() and (base_gradient[target]>0).all()
    target_rho = cb*scale*multiplier*float(base_gradient[target].max())/EPSILON
    assert target_rho<.5
    LTu_lower = (c0*(1-target_rho)*scale*(1-relative_sensitivity_bound)
                 *float(u[target]@base_gradient[target]))
    assert LTu_lower>0
    point = scale_state(raw,scale)
    ids = [0,N//2,N-1]
    images = subset_images(x0,pixel_ids,T,delta,ids)
    gradient_rows = torch.func.jacrev(lambda theta:contrast(theta,images))(raw)
    predicted = g0[None,:]+delta*(T[ids]@matrix.T)
    row_error = float(((gradient_rows[:,:N]-predicted).abs()/g0.abs()).max())
    assert row_error<2e-11
    actual_closeness = float(((gradient_rows-base_gradient).abs()/base_gradient.abs()).max())
    assert actual_closeness<=relative_sensitivity_bound*(1+1e-8)
    subset_contrast = float(contrast(raw,images).abs().max())
    assert subset_contrast<2e-12
    for theta,shapes,old_indices in ((raw,SHAPES,full_indices),(self_raw,self_shapes,self_indices)):
        _,zs,_,new_indices = forward(theta,images,shapes,True)
        assert all(torch.equal(new,old.expand_as(new)) for new,old in zip(new_indices,old_indices))
        assert min(float(z.min()) for z in zs)>0
    full_subset = raw_scale_and_capacity_subset(raw,point,images,u,scale)
    self_point,_ = literal_self(point)
    self_u = mean_direction(x0,self_shapes)
    self_subset = raw_scale_and_capacity_subset(self_raw,self_point,images,self_u,scale,self_shapes)
    return dict(
        scope="Native 120434 raw-parameter reference. All N1200 rank/routing/sensitivity/spectral statements use analytic bounds; autodiff and capacity values use only the recorded three-image subset. No full N1200-by-P Jacobian, full N1200 capacity evaluation, training, Monte Carlo, or learning-rate probability certificate.",
        architecture="RGB32x32 Conv5/pad2 3->16 Pool2 Conv5/pad2 16->16 Pool2 FC100 ReLU FC100 ReLU binary FC2; every bias free",
        N=N,batch_size=BATCH,epochs_per_task=EPOCHS,H=H,raw_parameter_count=P,
        selected_centers=centers,selected_pixel_count=len(pixel_ids),
        selected_pixels_sha256=sha256(pixel_ids.numpy().tobytes()).hexdigest(),
        mixed_minor_shape=list(matrix.shape),
        mixed_minor_sha256_float64=sha256(matrix.numpy().tobytes()).hexdigest(),
        mixed_minor_row_diagonal_dominance_margin=row_margin,
        mixed_minor_column_diagonal_dominance_margin=column_margin,
        mixed_minor_singular_value_lower=singular_m_lower,
        first_weight_gradient_shift_formula_relative_error=g_formula_error,
        selected_input_beta_identity_relative_error=beta_error,
        beta_range=[float(beta.min()),float(beta.max())],
        simplex_normalization=normalization,
        simplex_column_sum_error=column_sum_error,simplex_level_error=level_error,
        image_separation=delta,image_range_bound=[float(x0.min())-delta,float(x0.max())+delta],
        actual_mean_direction_error=mean_direction_error,
        all_raw_relative_sensitivity_bound=relative_sensitivity_bound,
        required_relative_sensitivity_bound=1/(2*BATCH),
        first_weight_feature_Q0=Q0,
        first_weight_feature_inverse_norm_upper=inverse_g_upper,
        first_weight_feature_singular_value_lower=singular_g_lower,
        leading_raw_lambda_min_lower=lambda1_lower,
        full_routing=full_routing,literal_self_routing=self_routing,
        minimum_absolute_base_raw_sensitivity=float(base_gradient.abs().min()),
        base_nonbias_gradient_squared_norm=float(base_gradient[:-2].square().sum()),
        all_image_nonbias_entry_upper=gamma_upper,
        all_image_unscaled_nonbias_frobenius_squared_upper=C_squared_upper,
        initialization_scale=scale,
        batch_response_relative_constant=cb,
        normal_spectral_relative_error_upper=error_ratio_upper,
        normal_symmetric_part_lower_conditional_on_nonnegative_bias=normal_lower,
        per_image_L_transpose_u_lower=LTu_lower,
        positive_endpoint_ray_coefficient_lower=N*LTu_lower**2,
        stationary_bias_premise="Supplied separately by the exact N1200/H30000 binary bias certificate; not measured here.",
        finite_subset_indices=ids,
        finite_subset_first_weight_affine_row_relative_error=row_error,
        finite_subset_actual_raw_relative_sensitivity_change=actual_closeness,
        finite_subset_maximum_reference_contrast=subset_contrast,
        finite_subset_full=full_subset,finite_subset_literal_self=self_subset,
        coincident_full_capacity_leading=coincident_capacity_coefficient(raw,x0,u),
        coincident_literal_self_capacity_leading=coincident_capacity_coefficient(self_raw,x0,self_u,self_shapes),
        numeric_status="All margins are float64 finite checks, not outward-rounded interval certificates. The full N1200 distinct-image capacity signs still use the separate continuity proof.",
        all_assertions_passed=True,elapsed_seconds=monotonic()-started,
    )


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,
                        default=Path(__file__).resolve().parents[2]/"results/cnn_drive_1009/adam_1200_reference.json")
    args = parser.parse_args()
    result = verify()
    result["verifier_source_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    keys = ["raw_parameter_count","mixed_minor_row_diagonal_dominance_margin",
            "mixed_minor_column_diagonal_dominance_margin",
            "first_weight_feature_singular_value_lower","initialization_scale",
            "normal_spectral_relative_error_upper",
            "normal_symmetric_part_lower_conditional_on_nonnegative_bias",
            "all_raw_relative_sensitivity_bound","all_assertions_passed","elapsed_seconds"]
    summary = {key:result[key] for key in keys}
    summary["output"] = str(args.output)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()

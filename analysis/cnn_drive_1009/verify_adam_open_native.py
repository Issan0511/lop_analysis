"""Native Conv/FC/bias open-state identities; no empirical long-time claim."""
from pathlib import Path
from fractions import Fraction
import json
import torch
from torch.nn import functional as F

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
GEN = torch.Generator().manual_seed(10091)
IMAGE = .2 + .6 * torch.rand((1, 3, 12, 12), generator=GEN)
EPS = 1e-8
SHAPES = [(2,3,5,5),(2,),(2,2,5,5),(2,), (3,18),(3,), (3,3),(3,), (2,3),(2,)]


def pack(parts):
    return torch.cat([p.flatten() for p in parts])


def unpack(t, shapes):
    parts, off = [], 0
    for shape in shapes:
        num = int(torch.tensor(shape).prod())
        parts.append(t[off:off+num].reshape(shape))
        off += num
    assert off == t.numel()
    return parts


def forward(t, shapes, extra=False):
    w1,b1,w2,b2,w3,b3,w4,b4,w5,b5 = unpack(t, shapes)
    z1 = F.conv2d(IMAGE,w1,b1,padding=2)
    h1,i1 = F.max_pool2d(F.relu(z1),2,return_indices=True)
    z2 = F.conv2d(h1,w2,b2,padding=2)
    h2,i2 = F.max_pool2d(F.relu(z2),2,return_indices=True)
    z3 = F.linear(h2.flatten(1),w3,b3)
    z4 = F.linear(F.relu(z3),w4,b4)
    h = F.relu(z4)
    out = F.linear(h,w5,b5).flatten()
    return (out,h.flatten(),(z1,z2,z3,z4),(i1,i2)) if extra else out


def mean_direction(shapes):
    parts = [torch.zeros(s) for s in shapes]
    parts[0][0] = F.unfold(IMAGE,5,padding=2).mean((0,2)).reshape(3,5,5)
    parts[1][0] = 1.
    return pack(parts)


def self_state(t):
    p = unpack(t, SHAPES)
    p[0],p[1],p[2] = p[0][:1],p[1][:1],p[2][:,:1]
    ss = [tuple(x.shape) for x in p]
    return pack(p),ss


def calc(t, shapes):
    u = mean_direction(shapes)
    fn = lambda q: forward(q,shapes)
    jfn = torch.func.jacrev(fn)
    j,jp = torch.func.jvp(jfn,(t,),(u,))
    ker,kp = j@j.T,jp@j.T+j@jp.T
    f,h,zs,inds = forward(t,shapes,True)
    s = (j[0]-j[1])/2
    return dict(j=j,jp=jp,k=ker,kp=kp,f=f,h=h,s=s,u=u,zs=zs,inds=inds,
                contrast=(f[0]-f[1])/2,
                capacity_slope=.5*torch.trace(torch.linalg.solve(torch.eye(2)+ker,kp)))


def reference_check(t, shapes):
    info = calc(t,shapes)
    parts = unpack(t,shapes)
    d = (parts[-2][0]-parts[-2][1])/2
    hidden_t = pack(parts[:-2])
    hidden_u = pack(unpack(info['u'],shapes)[:-2])
    hfn = lambda x: forward(torch.cat((x,pack(parts[-2:]))),shapes,True)[1]
    psi = lambda x: d@hfn(x)
    grad = torch.func.grad(psi)
    g,gp = torch.func.jvp(grad,(hidden_t,),(hidden_u,))
    h,hp = torch.func.jvp(hfn,(hidden_t,),(hidden_u,))
    vv = torch.tensor([[1.,-1.],[-1.,1.]])
    expected = (h@h+1)*torch.eye(2)+(g@g)*vv
    expectedp = 2*(h@hp)*torch.eye(2)+2*(g@gp)*vv
    assert h@hp > 0 and g@gp >= 0
    assert torch.allclose(info['k'],expected,atol=1e-10,rtol=1e-12)
    assert torch.allclose(info['kp'],expectedp,atol=1e-10,rtol=1e-12)
    return info,dict(kernel_error=float((info['k']-expected).abs().max()),
                     derivative_error=float((info['kp']-expectedp).abs().max()),
                     head_block_lower=float(2*(h@hp)),
                     hidden_derivative_coefficient=float(2*(g@gp)))


def omega(phase,H=4):
    b1,b2 = Fraction(9,10),Fraction(999,1000)
    p=b1*b2
    return (1-b1**phase)*(1-b2**phase)+p**phase*(1-b1**H)*(1-b2**H)/(1-p**H)


def main():
    scales = [.018,.04,.012,.04,.08,.04,.2,.04]
    p = [sc*(.8+.4*torch.rand(s,generator=GEN)) for s,sc in zip(SHAPES[:8],scales)]
    d = torch.tensor([.10,.12,.09])
    p += [torch.stack([d,-d]),torch.zeros(2)]
    base = pack(p)
    hidden = forward(base,SHAPES,True)[1]
    p[-1] = torch.tensor([-(d@hidden),d@hidden])
    base=pack(p)
    ref,formula = reference_check(base,SHAPES)
    st,ss = self_state(base)
    selfref,self_formula = reference_check(st,ss)
    assert abs(float(ref['contrast'])) < 1e-14

    # An independent perturbation of every raw coordinate, including biases
    # and the common class-head component. Only initialization sets contrast.
    pert = 1e-7*torch.randn(base.shape,generator=GEN)
    point=base+pert
    q=unpack(point,SHAPES)
    f=forward(point,SHAPES)
    adjustment = 1e-4-(f[0]-f[1])/2
    q[-1]=q[-1]+torch.stack([adjustment,-adjustment])
    point=pack(q)
    actual=calc(point,SHAPES)
    selfpoint,selfshapes=self_state(point)
    actualself=calc(selfpoint,selfshapes)
    certificates={}
    for name,x,x0 in [('full',actual,ref),('literal_self',actualself,selfref)]:
        dj=torch.linalg.vector_norm(x['j']-x0['j'])
        ej=torch.linalg.vector_norm(x['jp']-x0['jp'])
        err=2*(ej*torch.linalg.vector_norm(x0['j'])+dj*torch.linalg.vector_norm(x0['jp'])+dj*ej)
        lower=torch.linalg.eigvalsh(x0['kp']).min()
        assert err < lower
        assert torch.linalg.eigvalsh(x['kp']).min()>0 and x['capacity_slope']>0
        assert all(torch.equal(a,b) for a,b in zip(x['inds'],x0['inds']))
        certificates[name]=dict(reference_kp_lower=float(lower),perturbation_error=float(err),
                                positive_margin=float(lower-err),
                                actual_capacity_half_logdet_slope=float(x['capacity_slope']),
                                minimum_hidden_preactivation=min(float(z.min()) for z in x['zs']))

    errors=[]
    for label in [0,1]:
        loss=lambda t:F.cross_entropy(forward(t,SHAPES).reshape(1,2),torch.tensor([label]))
        raw=torch.func.grad(loss)(point)
        signed_label=1-2*label
        expected=actual['s']*(torch.tanh(actual['contrast'])-signed_label)
        errors.append(float((raw-expected).abs().max()))
    assert max(errors)<1e-12
    assert actual['s'].abs().min()>0
    first=actual['u']!=0
    assert (actual['u'][first]*actual['s'][first]).min()>0
    spatial_channel_minima=[float(torch.linalg.svdvals(z[0].flatten(1))[-1]) for z in actual['zs'][:2]]
    assert min(spatial_channel_minima)>0

    # Exact phasewise stationary derivative at the zero-contrast reference.
    # Its covariance omega has rational beta/block coefficients.
    om=torch.tensor([float(omega(r)) for r in range(1,5)])
    abs_s=ref['s'].abs()
    c=((EPS+abs_s[:,None]*(1-om[None,:]))/(EPS+abs_s[:,None])**2).mean(1)
    v=c*ref['s']
    normal=ref['s']@v
    endpoint_mean_coefficient=(ref['u']@v)/normal
    assert c.min()>0 and normal>0 and endpoint_mean_coefficient>0
    assert all(Fraction(0)<omega(r)<Fraction(1) for r in range(1,5))
    out=dict(scope='Finite raw Jacobian/gradient identities and local geometry; no success-rate or infinite-time experiment.',
             architecture='RGB12x12 Conv5(2) Pool2 Conv5(2) Pool2 FC3 ReLU FC3 ReLU binary head, all biases',
             raw_parameter_count=point.numel(),N=1,all_raw_trainable=True,
             formula_full=formula,formula_self=self_formula,
             reference_contrast=float(ref['contrast']),initial_contrast=float(actual['contrast']),
             initial_target_mean=float(actual['u']@point),
             raw_perturbation_norm=float(torch.linalg.vector_norm(point-base)),
             coordinate_gradient_factorization_errors=errors,
             minimum_absolute_logit_contrast_derivative=float(actual['s'].abs().min()),
             minimum_target_coordinate_alignment=float((actual['u'][first]*actual['s'][first]).min()),
             spatial_channel_nonproportional_singular_minima=spatial_channel_minima,
             capacity_certificates=certificates,
             stationary_phase_omega_rational=[str(omega(r)) for r in range(1,5)],
             stationary_linearization_c_range=[float(c.min()),float(c.max())],
             normal_coefficient=float(normal),
             local_endpoint_mean_loss_per_unit_contrast=float(endpoint_mean_coefficient),
             interpretation='The positive endpoint coefficient is local ODE geometry, not a measured moving-Adam long-time displacement.')
    dest=Path(__file__).resolve().parents[2]/'results/cnn_drive_1009/adam_open_native.json'
    dest.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    main()

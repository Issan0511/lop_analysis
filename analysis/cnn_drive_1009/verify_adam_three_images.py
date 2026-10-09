"""Three-image raw identities; no infinite-time or stationary-Adam estimation."""
from itertools import product
from pathlib import Path
import json
import torch
from torch.nn import functional as F

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
GEN = torch.Generator().manual_seed(10093)
SHAPES = [(2, 3, 5, 5), (2,), (2, 2, 5, 5), (2,),
          (3, 18), (3,), (3, 3), (3,), (2, 3), (2,)]


def pack(parts):
    return torch.cat([x.flatten() for x in parts])


def unpack(theta, shapes):
    parts, off = [], 0
    for shape in shapes:
        n = int(torch.tensor(shape).prod())
        parts.append(theta[off:off+n].reshape(shape))
        off += n
    assert off == theta.numel()
    return parts


def forward(theta, images, shapes=SHAPES, extra=False):
    w1,b1,w2,b2,w3,b3,w4,b4,w5,b5 = unpack(theta, shapes)
    z1 = F.conv2d(images, w1, b1, padding=2)
    h1,i1 = F.max_pool2d(F.relu(z1), 2, return_indices=True)
    z2 = F.conv2d(h1, w2, b2, padding=2)
    h2,i2 = F.max_pool2d(F.relu(z2), 2, return_indices=True)
    z3 = F.linear(h2.flatten(1), w3, b3)
    z4 = F.linear(F.relu(z3), w4, b4)
    h = F.relu(z4)
    out = F.linear(h, w5, b5)
    return (out, h, (z1,z2,z3,z4), (i1,i2)) if extra else out


def mean_direction(images, shapes=SHAPES):
    parts = [torch.zeros(s) for s in shapes]
    parts[0][0] = F.unfold(images, 5, padding=2).mean((0,2)).reshape(3,5,5)
    parts[1][0] = 1.
    return pack(parts)


def self_state(theta):
    p = unpack(theta, SHAPES)
    p[0],p[1],p[2] = p[0][:1],p[1][:1],p[2][:,:1]
    return pack(p), [tuple(x.shape) for x in p]


def quantities(theta, images, shapes=SHAPES):
    u = mean_direction(images, shapes)
    jfn = torch.func.jacrev(lambda t: forward(t,images,shapes).flatten())
    j,jp = torch.func.jvp(jfn, (theta,), (u,))
    ker,kp = j@j.T, jp@j.T+j@jp.T
    out,h,zs,inds = forward(theta,images,shapes,True)
    s = (j.reshape(images.shape[0],2,-1)[:,0]-j.reshape(images.shape[0],2,-1)[:,1])/2
    gaps = []
    for z in zs[:2]:
        windows = z.unfold(2,2,2).unfold(3,2,2).flatten(-2)
        winners = windows.topk(2,dim=-1).values
        gaps.append(float((winners[...,0]-winners[...,1]).min()))
    return dict(u=u,j=j,jp=jp,k=ker,kp=kp,s=s,out=out,h=h,zs=zs,inds=inds,
                contrast=(out[:,0]-out[:,1])/2,
                capacity=.5*torch.trace(torch.linalg.solve(torch.eye(ker.shape[0])+ker,kp)),
                minimum_pool_gap=min(gaps))


def capacity_certificate(current, base):
    n = 3
    k0 = torch.kron(torch.ones(n,n), base['k'])
    kp0 = torch.kron(torch.ones(n,n), base['kp'])
    c0 = n/2*torch.trace(torch.linalg.solve(torch.eye(2)+n*base['k'],base['kp']))
    # Ridge lambda=1; nuclear/operator norm resolvent bound.
    err = .5*torch.linalg.matrix_norm(current['kp']-kp0,ord='nuc')
    err += .5*torch.linalg.matrix_norm(kp0,ord='nuc')*torch.linalg.matrix_norm(current['k']-k0,ord=2)
    return dict(duplicate_capacity_slope=float(c0),
                finite_continuity_error=float(err),
                certified_positive_margin=float(c0-err),
                actual_capacity_slope=float(current['capacity']))


def main():
    grid = torch.arange(12)[:,None]+12*torch.arange(12)[None,:]
    x0 = (.2+grid/256).expand(1,3,12,12).clone()
    p = [torch.zeros(s) for s in SHAPES]
    p[0] = 1e-5*(.8+.4*torch.rand(SHAPES[0],generator=GEN))
    p[0][:,:,2,2] = .2*(.8+.4*torch.rand((2,3),generator=GEN))
    p[2] = 1e-5*(.8+.4*torch.rand(SHAPES[2],generator=GEN))
    p[2][:,:,2,2] = .3*(.8+.4*torch.rand((2,2),generator=GEN))
    p[4] = .002*(.8+.4*torch.rand(SHAPES[4],generator=GEN))
    p[4][torch.arange(3),torch.tensor([0,4,8])] += .8
    p[6] = .002*(.8+.4*torch.rand(SHAPES[6],generator=GEN))+.8*torch.eye(3)
    for j in [1,3,5,7]:
        p[j] = .04*(.8+.4*torch.rand(SHAPES[j],generator=GEN))
    d = torch.tensor([.10,.12,.09])
    p[8] = torch.stack((d,-d))
    raw = pack(p)
    h0 = forward(raw,x0,extra=True)[1][0]
    p[9] = torch.stack((-(d@h0),d@h0))
    raw = pack(p)
    a = torch.func.jacrev(lambda x:forward(raw,x.reshape_as(x0),extra=True)[1][0])(x0.flatten())
    ids = [3*12+3,7*12+7,11*12+11]
    b = a[:,ids]
    assert torch.linalg.det(b).abs()>1e-8
    beta = d@b
    v = torch.zeros(2,x0.numel())
    v[0,ids[0]],v[0,ids[2]] = 1/beta[0],-1/beta[2]
    v[1,ids[1]],v[1,ids[2]] = 1/beta[1],-1/beta[2]
    v /= v.abs().max()
    perturb = torch.stack((v[0],v[1],-v.sum(0))).reshape(3,3,12,12)
    base = quantities(raw,x0)
    selfraw,selfshapes = self_state(raw)
    selfbase = quantities(selfraw,x0,selfshapes)
    # Choose a finite witness satisfying the analytic continuity inequalities.
    tau = .01
    for _ in range(30):
        images = x0+tau*perturb
        full = quantities(raw,images)
        own = quantities(selfraw,images,selfshapes)
        cf,cs = capacity_certificate(full,base),capacity_certificate(own,selfbase)
        closeness = ((full['s']-base['s']).abs()/base['s'].abs()).max()
        routing = all(torch.equal(a,a0.expand_as(a)) for cur,old in [(full,base),(own,selfbase)]
                      for a,a0 in zip(cur['inds'],old['inds']))
        if (images.min()>0 and images.max()<1 and closeness<1/6 and routing
                and min(cf['certified_positive_margin'],cs['certified_positive_margin'])>0):
            break
        tau /= 2
    else:
        raise AssertionError('No witness found within the registered continuity check.')
    assert full['contrast'].abs().max()<1e-13
    assert min(full['minimum_pool_gap'],own['minimum_pool_gap'])>0
    assert torch.linalg.svdvals(full['h'])[-1]>1e-7
    assert torch.linalg.svdvals(full['s'])[-1]>1e-7
    assert torch.allclose(images.mean(0),x0[0],atol=1e-15,rtol=0)
    assert torch.allclose(full['u'],base['u'],atol=1e-15,rtol=0)
    assert full['s'].abs().min()>0
    target = full['u']!=0
    assert (full['s'][:,target]*full['u'][target]).min()>0
    errors, minima = [], []
    for labels in product([0,1], repeat=3):
        y = torch.tensor(labels)
        grad = torch.func.grad(lambda t:F.cross_entropy(forward(t,images),y))(raw)
        expected = full['s'].T@(torch.tanh(full['contrast'])-(1-2*y))/3
        errors.append(float((grad-expected).abs().max()))
        minima.append(float(grad.abs().min()))
        assert grad.abs().min()>base['s'].abs().min()/6
    assert max(errors)<1e-12
    # Any positive diagonal D gives the projection/mean identity. This D is
    # an algebraic witness, explicitly NOT an estimate of stationary Adam D*.
    metric = .5+torch.linspace(0,1,raw.numel())
    j,u = full['s'],full['u']
    k = (j*metric)@j.T
    normal = (metric[:,None]*j.T)@torch.linalg.solve(k,j)
    vmetric = metric*u
    e = j@vmetric
    bstar = e@torch.linalg.solve(k,e)
    direct = u@normal@vmetric
    assert bstar>0
    assert torch.allclose(direct,bstar,atol=1e-9,rtol=1e-10)
    projection_error = float((normal@normal-normal).abs().max())
    assert projection_error<1e-7
    out = dict(scope='Finite native raw identities, rank and continuity witnesses; no stationary metric estimation or long-time probability measurement.',
               architecture='RGB12x12 Conv5(2) Pool2 Conv5(2) Pool2 FC3 ReLU FC3 ReLU binary head; every raw bias/weight free',
               N=3,raw_parameter_count=raw.numel(),ridge=1.,tau=tau,
               selected_input_jacobian_determinant=float(torch.linalg.det(b)),
               hidden_singular_values=torch.linalg.svdvals(full['h']).tolist(),
               contrast_jacobian_singular_values=torch.linalg.svdvals(j).tolist(),
               maximum_absolute_contrast=float(full['contrast'].abs().max()),
               mean_input_error=float((images.mean(0)-x0[0]).abs().max()),
               actual_mean_direction_error=float((u-base['u']).abs().max()),
               maximum_relative_sensitivity_perturbation=float(closeness),
               minimum_base_sensitivity=float(base['s'].abs().min()),
               minimum_raw_gradient_by_label_assignment=minima,
               ce_factorization_errors=errors,
               full_minimum_hidden_preactivation=min(float(z.min()) for z in full['zs']),
               full_minimum_pool_gap=full['minimum_pool_gap'],
               self_minimum_pool_gap=own['minimum_pool_gap'],
               capacity_full=cf,capacity_literal_self=cs,
               arbitrary_metric_geometry=dict(metric_range=[float(metric.min()),float(metric.max())],
                   positive_mean_coefficient=float(bstar),direct_identity_error=float(abs(direct-bstar)),
                   projection_idempotence_error=projection_error,
                   interpretation='Holds for every positive diagonal metric analytically; this chosen metric is not stationary Adam D*.'))
    dest=Path(__file__).resolve().parents[2]/'results/cnn_drive_1009/adam_three_images.json'
    dest.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    main()

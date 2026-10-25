"""Replica (not image) covariance in the fixed-input, fixed-metric fc1 update.

This module is a gradient ledger, not an exact decomposition of Adam trajectories.
See t5_measurement.md. All reductions are float64, with population covariance.
"""
import torch


class ReplicaMoments:
    """Accumulate one fixed image/token chunk over independent replicas."""

    def __init__(self):
        self.count = 0
        self.c = self.chi = self.cchi = self.positive = None

    @torch.no_grad()
    def add(self, z, c):
        assert z.ndim == 3 and z.shape == c.shape
        assert torch.isfinite(z).all() and torch.isfinite(c).all()
        z, c = z.detach().double(), c.detach().double()
        chi, positive = (z > 0).double(), z.clamp_min(0)
        if self.count == 0:
            self.c, self.chi = c.clone(), chi.clone()
            self.cchi, self.positive = c * chi, positive.clone()
        else:
            assert z.shape == self.c.shape and z.device == self.c.device
            self.c.add_(c)
            self.chi.add_(chi)
            self.cchi.add_(c * chi)
            self.positive.add_(positive)
        self.count += 1

    @torch.no_grad()
    def values(self):
        assert self.count >= 1
        c, p = self.c / self.count, self.chi / self.count
        cchi, positive = self.cchi / self.count, self.positive / self.count
        return dict(c=c, p=p, cchi=cchi, covariance=cchi - c * p,
                    activation_gate_covariance=(1 - p) * positive,
                    gate_variance=p * (1 - p))


@torch.no_grad()
def kernel(u, probe_mean, metric_w, metric_b):
    """u[N,T,D], probe mean[D], positive diagonal metric_w[H,D]/metric_b[H].

    The input and metric must be identical across replicas. The probe mean is
    typically the mean *CLS* input; training gradients still sum over all tokens.
    """
    assert u.ndim == 3 and probe_mean.shape == (u.shape[2],)
    assert metric_w.ndim == 2 and metric_w.shape[1] == u.shape[2]
    assert metric_b.shape == (metric_w.shape[0],)
    assert bool((metric_w > 0).all() and (metric_b > 0).all())
    assert all(bool(torch.isfinite(x).all()) for x in (u, probe_mean, metric_w, metric_b))
    # Contract the metric with the probe first; a three-way einsum can materialize
    # N x T x H x D (over 9 GiB for one image chunk in the real model).
    weighted_probe = metric_w.double() * probe_mean.double()
    return torch.nn.functional.linear(u.double(), weighted_probe, metric_b.double())


@torch.no_grad()
def components(moments, k, lr, population_images):
    """Contributions from this chunk to expected next full-population mu change.

    c is the derivative of the *single-image* CE with respect to activation;
    populate it using a SUM of image losses, then divide by population_images here.
    Do not divide by the number of tokens. For chunks, add these vectors.
    """
    assert population_images >= k.shape[0] and lr > 0
    v = moments.values()
    assert k.shape == v['c'].shape and torch.isfinite(k).all()
    scale = -float(lr) / population_images
    total = scale * (k * v['cchi']).sum((0, 1))
    mean_field = scale * (k * v['c'] * v['p']).sum((0, 1))
    gate_covariance = scale * (k * v['covariance']).sum((0, 1))
    return dict(total=total, mean_field=mean_field, gate_covariance=gate_covariance,
                identity_error=total - mean_field - gate_covariance)


def verification():
    """Independent autograd verifies token, bias, metric and normalization factors."""
    gen = torch.Generator().manual_seed(1025)
    dtype = torch.float64
    R, N, tokens, width, inputs, outputs = 7, 6, 3, 5, 4, 3
    rand = lambda *s: torch.randn(*s, generator=gen, dtype=dtype)
    u, v = rand(N, tokens, inputs), rand(tokens, width, outputs)
    probe = u[:, 0].mean(0)
    dw, db = rand(width, inputs).exp(), rand(width).exp()
    y = torch.arange(N) % outputs
    lr = .003
    stats, direct, zs, cs = ReplicaMoments(), [], [], []
    ws, bs = [], []
    for _ in range(R):
        w, b = rand(width, inputs).requires_grad_(), rand(width).requires_grad_()
        z = torch.nn.functional.linear(u, w, b)
        a = z.relu()
        logits = torch.einsum('nth,tho->no', a, v)
        loss = torch.nn.functional.cross_entropy(logits, y, reduction='sum')
        c, gw, gb = torch.autograd.grad(loss, (a, w, b))
        stats.add(z, c)
        direct.append(-lr / N * ((dw * gw * probe).sum(1) + db * gb))
        zs.append(z.detach()); cs.append(c.detach()); ws.append(w.detach()); bs.append(b.detach())
    k = kernel(u, probe, dw, db)
    result = components(stats, k, lr, N)
    expected = torch.stack(direct).mean(0)
    torch.testing.assert_close(result['total'], expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(result['identity_error'], torch.zeros(width, dtype=dtype), atol=1e-12, rtol=0)
    z = torch.stack(zs)
    chi, a = (z > 0).double(), z.relu()
    actual_cov = (chi * a).mean(0) - chi.mean(0) * a.mean(0)
    torch.testing.assert_close(stats.values()['activation_gate_covariance'], actual_cov, atol=1e-12, rtol=1e-12)
    assert actual_cov.min() >= -1e-12
    # Identical replicas have zero covariance, even with heterogeneous images.
    same = ReplicaMoments()
    for _ in range(R): same.add(zs[0], cs[0])
    assert same.values()['covariance'].abs().max() < 1e-12
    # Splitting the image population must leave the answer unchanged.
    chunks = []
    for lo, hi in ((0, 2), (2, N)):
        part = ReplicaMoments()
        for zz, cc in zip(zs, cs): part.add(zz[lo:hi], cc[lo:hi])
        chunks.append(components(part, k[lo:hi], lr, N)['total'])
    torch.testing.assert_close(sum(chunks), expected, atol=1e-12, rtol=1e-12)
    return dict(status='passed', replicas=R, images=N, tokens=tokens,
                autograd_max_abs_error=float((result['total'] - expected).abs().max()),
                identity_max_abs_error=float(result['identity_error'].abs().max()),
                self_covariance_min=float(actual_cov.min()),
                deterministic_replica_covariance_max_abs=float(same.values()['covariance'].abs().max()),
                image_chunking_equal=True)


if __name__ == '__main__':
    import json
    torch.set_num_threads(2)
    print(json.dumps(verification(), indent=2))

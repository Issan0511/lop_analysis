"""Synthetic-only independent checks. No real checkpoint constructors."""
import copy
import json
import tempfile
from pathlib import Path
import torch
from torch.nn import functional as F
from analysis.readout_separation_1111 import ops
from analysis.readout_separation_1111 import probe as P
from analysis.loading_bridge_1111 import dynamic as D


def close(a, b, tol=2e-11):
    torch.testing.assert_close(a, b, rtol=tol, atol=tol)


def toy_mlp():
    box = D.MLP.__new__(D.MLP)
    gen = torch.Generator().manual_seed(50431)
    box.act, box.seed = 'ELU', 7
    box.X = torch.randn(11, 5, generator=gen, dtype=torch.float64)
    box.P = [torch.nn.Parameter(torch.randn(*s, generator=gen, dtype=torch.float64)*.6)
             for s in ((7, 5), (7,), (4, 7), (4,))]
    box.opt = torch.optim.Adam(box.P, lr=.003, betas=(.71, .94), eps=3e-6, foreach=False)
    return box


class ToyBlock(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(d, d, dtype=torch.float64)*.2)

    def forward(self, x):
        return x + x.mean(1, keepdim=True) @ self.weight.T


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.depth = 1
        self.layers = torch.nn.ModuleDict(dict(patch_embed=torch.nn.Linear(3, 6, dtype=torch.float64),
                                              pos_drop=torch.nn.Identity(), block_0=ToyBlock(6),
                                              norm=torch.nn.LayerNorm(6, eps=.03, dtype=torch.float64),
                                              out=torch.nn.Linear(6, 4, dtype=torch.float64)))
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, 6, dtype=torch.float64))
        self.pos_embed = torch.nn.Parameter(torch.randn(1, 3, 6, dtype=torch.float64)*.2)
        with torch.no_grad():
            self.layers['norm'].weight.copy_(torch.tensor([.3, 2., -.7, 1.2, .8, -.4]))
            self.layers['norm'].bias.copy_(torch.arange(6)*.17)


class ToyViT:
    def __init__(self):
        self.model = ToyModel()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=.002, betas=(.8, .96), eps=1e-7, foreach=False)

    def head(self):
        h = self.model.layers['out']; return h.weight, h.bias, torch.arange(4)

    def body(self):
        excluded = {id(p) for p in self.head()[:2]}
        return [p for p in self.model.parameters() if id(p) not in excluded]


def test_adam():
    count = 0
    for dtype in (torch.float64, torch.float32):
        torch.manual_seed(710)
        ps = [torch.nn.Parameter(torch.randn(8, 6, dtype=dtype)), torch.nn.Parameter(torch.randn(6, dtype=dtype))]
        opt = torch.optim.Adam([dict(params=[ps[0]], lr=.002, betas=(.7, .94), eps=2e-5, weight_decay=.01, amsgrad=True),
                               dict(params=[ps[1]], lr=.03, betas=(.9, .99), eps=1e-8)], foreach=False)
        for _ in range(6):
            for p in ps: p.grad = torch.randn_like(p)
            opt.step()
        old_ps = [p.detach().clone() for p in ps]; state = copy.deepcopy(opt.state_dict())
        for grads in ([torch.randn_like(p)*.001 for p in ps], [torch.zeros_like(ps[0]), None]):
            for clear in (False, True):
                checks = {}; P.validate_virtual_adam(ps, grads, opt, clear, checks, 'toy')
                count += 1
        for a, b in zip(ps, old_ps): assert torch.equal(a, b)
        for k, s in state['state'].items():
            for name, x in s.items():
                if torch.is_tensor(x): assert torch.equal(x, opt.state_dict()['state'][k][name])
    return count


def test_crossed_mlp():
    box = toy_mlp(); x = box.X
    h, aux = ops.features(box, x, return_aux=True)
    bank = ops.geometry_bank(box.P[2], box.P[3], h.detach().mean(0), 7, 'mlp', 10)
    old = [p.detach().clone() for p in box.P]
    actual = torch.arange(len(x)) % 4
    count = 0
    for fkey in P.BANK_KEYS:
        ef = bank['entries'][fkey]
        zf = h.detach() @ ef['W'].T + ef['b']; prob = zf.softmax(1)
        for bkey in P.BANK_KEYS:
            eb = bank['entries'][bkey]
            for label in P.LABELS:
                target = F.one_hot(actual, 4).double() if label=='actual' else torch.full_like(prob, .25)
                gh = (prob-target) @ eb['W'] / len(x)
                hf = h.detach().clone().requires_grad_(); zb = hf @ eb['W'].T
                crossed = zf.detach() + (zb-zb.detach())
                loss = (crossed.logsumexp(1) - (crossed*target).sum(1)).mean()
                close(torch.autograd.grad(loss, hf)[0], gh)
                for mode in ('native', 'elu_unity'):
                    hf, z, gp, grads = P.transmission_vjp(box, x, gh, mode, box.body())
                    close(hf, h)
                    derivative = torch.where(z>0, torch.ones_like(z), torch.exp(z)) if mode=='native' else torch.ones_like(z)
                    close(gp, gh*derivative)
                    close(grads[0], gp.T@x); close(grads[1], gp.sum(0))
                    push = P.preact_shift(grads, x.mean(0))
                    expected = (x@grads[0].T+grads[1]).mean(0)
                    close(push, expected)
                    # This nontrivial input kernel must differ from mean dL/dz.
                    if fkey==bkey=='S1' and label=='actual' and mode=='native':
                        assert float((push-gp.mean(0)).norm()) > 1e-4
                    count += 1
    for a, b in zip(box.P, old): assert torch.equal(a, b)
    return count


def test_ln_transmission():
    torch.manual_seed(981)
    box = ToyViT(); x = torch.randn(9, 2, 3, dtype=torch.float64)
    gh = torch.randn(9, 6, dtype=torch.float64)/9
    ps, _, groups = P.body_layout(box)
    h, aux = ops.features(box, x, return_aux=True)
    pre = aux['preln'].detach()[:, 0]; gamma = aux['gamma'].detach()
    xc = pre-pre.mean(-1, keepdim=True); sigma = (xc.square().mean(-1, keepdim=True)+aux['eps']).sqrt()
    q = xc/sigma; v = gh*gamma
    formulas = dict(native=(v-v.mean(-1, keepdim=True)-q*(v*q).mean(-1, keepdim=True))/sigma,
                    ln_remove_radial=(v-v.mean(-1, keepdim=True))/sigma,
                    ln_remove_radial_mean=v/sigma)
    native = None; count = 0
    for mode, expected in formulas.items():
        hf, _, gp, grads = P.transmission_vjp(box, x, gh, mode, ps)
        assert torch.equal(hf, h.detach())
        close(gp, expected)
        close(grads[groups['gamma'][0]], (gh*q).sum(0))
        close(grads[groups['beta'][0]], gh.sum(0))
        if mode=='native': native = grads
        else:
            matched, scale, status = P.match_trunk(grads, native, groups)
            assert status=='matched' and scale > 0
            assert abs(P.norm(P.selected(matched, groups, 'trunk'))-P.norm(P.selected(native, groups, 'trunk'))) < 1e-12
            for g in ('gamma', 'beta'):
                for i in groups[g]: assert torch.equal(matched[i], native[i])
        count += 1
    checks = {}; P.gauge_check(box, x, torch.arange(len(x)) % 4, checks)
    assert max(v['max_abs'] for v in checks.values()) < 1e-11
    return count+len(checks)


def test_small_guards():
    a, key = P.subset_indices(2500, 'vit', 10252010)
    b, _ = P.subset_indices(2500, 'vit', 10252010)
    c, _ = P.subset_indices(2500, 'vit', 10252011)
    assert len(a.unique())==128 and torch.equal(a, b) and not torch.equal(a, c)
    assert P.cosine([torch.zeros(2)], [torch.ones(2)]) is None
    layout = dict(trunk=[0], gamma=[1], beta=[], body=[0,1])
    assert P.match_trunk([torch.zeros(2), torch.ones(1)], [torch.ones(2), torch.ones(1)], layout)[2]=='zero_alternative'
    assert P.match_trunk([torch.ones(2), torch.ones(1)], [torch.zeros(2), torch.ones(1)], layout)[2]=='zero_native'
    return 5




def test_gauge_numerics():
    # A tiny absolute change at a zero coordinate is not a large vector error.
    ref = torch.tensor([1., 0.], dtype=torch.float32)
    actual = torch.tensor([1., 1e-6], dtype=torch.float32)
    checks = {}; P.assert_gauge_close('nearzero', actual, ref, checks)
    assert checks['nearzero']['relative_l2'] < 2e-5
    rejected = 0
    for a,b in ((ref*1.001, ref), (torch.tensor([1., .001]), ref),
                (torch.tensor([0., 1e-12]), torch.zeros(2)),
                (torch.tensor([float('nan'), 0.]), ref),
                (actual.double(), ref.double())):
        try: P.assert_gauge_close('bad', a, b, {})
        except AssertionError: rejected += 1
    assert rejected == 5
    P.assert_gauge_close('zero', torch.zeros(2), torch.zeros(2), {})
    return 7


def test_serialized_assay():
    """Exercise the whole scalar/array output contract using synthetic boxes."""
    original = P.source_provenance
    P.source_provenance = lambda box: dict(git_hash='synthetic', script_sha256='synthetic',
                                          spec_sha256='synthetic', helpers={}, sources={})
    try:
        torch.manual_seed(204)
        boxes = [toy_mlp(), ToyViT()]
        boxes[1].seed = 9
        boxes[1].X = torch.randn(9, 2, 3, dtype=torch.float64)
        with tempfile.TemporaryDirectory(prefix='readout_probe_synthetic_') as td:
            for box in boxes:
                model = 'mlp' if isinstance(box,D.MLP) else 'vit'
                box.new = torch.arange(len(box.X)) % 4
                box.old = (box.new+1) % 4
                with torch.no_grad(): h = ops.native_features(box, box.X)
                w,b,_ = box.head()
                bank = ops.geometry_bank(w,b,h.mean(0),box.seed,model,10)
                result = P.assay(box,10,td,bank=bank)
                assert result['complete']
                assert len(result['records'])==(144 if model=='mlp' else 216)
                assert P.sha(Path(td)/result['arrays_file'])==result['arrays_sha256']
                for row in result['records']:
                    assert row['probability_sha256']
                    assert row['normmatch_status'] in ('native_reference','matched')
                resumed = P.assay(box,10,td,bank=bank)
                assert resumed['arrays_sha256']==result['arrays_sha256']
        return 2
    finally:
        P.source_provenance = original


def main():
    torch.set_num_threads(1)
    results = dict(adam=test_adam(), crossed_mlp=test_crossed_mlp(), ln_and_gauge=test_ln_transmission(), guards=test_small_guards(), gauge_numerics=test_gauge_numerics(), serialized_assays=test_serialized_assay())
    print(json.dumps(dict(passed=True, tests=results, real_checkpoint_constructors=0), indent=2))


if __name__=='__main__': main()

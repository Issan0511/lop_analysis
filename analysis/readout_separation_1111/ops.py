"""Readout geometry and forward-identical Jacobian controls.

Importing this module does not construct a real checkpoint or train a model.
Class-space O uses a Householder basis of 1-perp and a sign-corrected Gaussian QR.
"""
import hashlib
import torch
from torch.nn import functional as F
from analysis.loading_bridge_1111 import dynamic as D
from analysis.loading_bridge_1111 import static_probe as A
from analysis.p_residual_loading_1111 import run as R
from src import stage1_ctrl_1111 as M


class GainIdentity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gain):
        ctx.gain = float(gain)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.gain, None


class CenteredForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, zbar0, cf):
        # This shortcut is essential for bit-identical native controls.
        return z if float(cf) == 1. else zbar0 + float(cf) * (z - zbar0)

    @staticmethod
    def backward(ctx, grad):
        # The forward coefficient does not scale either head gradient.
        return grad, None, None


class ELUUnity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, native_y, z):
        return native_y

    @staticmethod
    def backward(ctx, grad):
        return None, grad


class LNCorrect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, native_y, x, gamma, eps, mode):
        if mode not in ('native', 'ln_remove_radial', 'ln_remove_radial_mean'):
            raise ValueError(mode)
        ctx.save_for_backward(x, gamma)
        ctx.eps, ctx.mode = float(eps), mode
        return native_y

    @staticmethod
    def backward(ctx, grad):
        x, gamma = ctx.saved_tensors
        if ctx.mode == 'native':
            return grad, None, None, None, None
        xc = x - x.mean(-1, keepdim=True)
        sigma = (xc.square().mean(-1, keepdim=True) + ctx.eps).sqrt()
        q = xc / sigma
        v = grad * gamma
        correction = q * (q * v).mean(-1, keepdim=True) / sigma
        if ctx.mode == 'ln_remove_radial_mean':
            correction = correction + v.mean(-1, keepdim=True) / sigma
        # native_y receives the ordinary gradient, retaining native gamma/beta
        # gradients. Only the additional x edge cancels the selected input term.
        return grad, correction, None, None, None


def features(box, x, jacobian='native', return_aux=False):
    """Differentiable head-input features, with no persistent hooks/patches."""
    if isinstance(box, D.MLP):
        z = x @ box.P[0].T + box.P[1]
        h = M.activ(z, box.act)
        if jacobian == 'elu_unity':
            if box.act != 'ELU':
                raise ValueError('ELU unity derivative requires ELU')
            h = ELUUnity.apply(h, z)
        elif jacobian != 'native':
            raise ValueError(jacobian)
        aux = dict(preact=z, activation=box.act)
    else:
        L = box.model.layers
        h = L['patch_embed'](x)
        h = torch.cat((box.model.cls_token.expand(h.shape[0], -1, -1), h), 1) + box.model.pos_embed
        # All registered boxes have dropout=0. Match the native pos_drop path too.
        if 'pos_drop' in L:
            h = L['pos_drop'](h)
        for i in range(box.model.depth):
            h = L[f'block_{i}'](h)
        preln = h
        norm = L['norm']
        h = norm(preln)
        if jacobian in ('ln_remove_radial', 'ln_remove_radial_mean'):
            if not isinstance(norm, torch.nn.LayerNorm):
                raise TypeError('Expected the native final affine LayerNorm')
            h = LNCorrect.apply(h, preln, norm.weight, norm.eps, jacobian)
        elif jacobian != 'native':
            raise ValueError(jacobian)
        h = h[:, 0]
        aux = dict(preln=preln, gamma=norm.weight, beta=norm.bias, eps=norm.eps, cls_index=0)
    return (h, aux) if return_aux else h


def native_features(box, x):
    return features(box, x)


def forward(box, x, cf=1., body_gain=1., jacobian='native', zbar0=None, force_eager=False):
    # Keep the established native eager/compiled path for every physical arm.
    if cf == 1. and body_gain == 1. and jacobian == 'native' and not force_eager:
        return box.forward(x)
    h = features(box, x, jacobian)
    if body_gain != 1.:
        h = GainIdentity.apply(h, body_gain)
    W, b, rows = box.head()
    # Compute all 200 ViT head rows before selection, like native model.forward.
    z = h @ W.T + b if isinstance(box, D.MLP) else F.linear(h, W, b)
    z = z[:, rows]
    if cf != 1.:
        if zbar0 is None:
            raise ValueError('Forward scaling requires the fixed source zbar0')
        z = CenteredForward.apply(z, zbar0, cf)
    return z


def private_generator(seed, model, t_end, role='O'):
    key = f'readout_separation_1111|{role}|{model}|{seed}|{t_end}'
    value = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little') & ((1 << 63) - 1)
    return torch.Generator().manual_seed(value)


def orthogonal_fixing_ones(k, generator):
    """Q=ee^T+BUB^T, B=Householder(e1->e)[:,1:], U=QR Gaussian."""
    if k < 2:
        raise ValueError('At least two classes required')
    e = torch.ones(k, dtype=torch.float64) / k**.5
    e1 = torch.zeros_like(e); e1[0] = 1.
    v = e1 - e
    v = v / v.norm()
    H = torch.eye(k, dtype=torch.float64) - 2 * v[:, None] * v[None, :]
    B = H[:, 1:]
    G = torch.randn(k-1, k-1, generator=generator, dtype=torch.float64)
    U, rr = torch.linalg.qr(G)
    signs = torch.where(torch.diag(rr) >= 0, 1., -1.)
    U = U * signs[None, :]
    Q = e[:, None] * e[None, :] + B @ U @ B.T
    return Q


def canonical_permutation(seed, model, t_end, width, act='ELU', classes=10):
    if model == 'mlp':
        g = M.own(f'arm|P|adam|{t_end+1}', seed)
    elif model == 'vit':
        g = A.rng('P', seed, t_end, 'ViT', 'SL')
    else:
        raise ValueError(model)
    return torch.stack([torch.randperm(classes, generator=g) for _ in range(width)], 1)


def geometry_bank(W, b, mu, seed, model, t_end, permutation=None):
    """Pure frozen-head bank. Entries are float32/64 matching the input dtype.

    P is the canonical independent class permutation for each feature column.
    O rotates only class space, fixes ones, and preserves the full feature Gram.
    Scaling changes only the class-centered component; every bias compensates
    the source mean mu. S1 and P1 avoid algebraically redundant round trips.
    """
    W, b, mu = W.detach().clone(), b.detach().clone(), mu.detach().clone()
    k, width = W.shape
    if permutation is None:
        permutation = canonical_permutation(seed, model, t_end, width, classes=k)
    idx = permutation.to(W.device)
    WP = W.gather(0, idx)
    Qcpu = orthogonal_fixing_ones(k, private_generator(seed, model, t_end))
    WO = (Qcpu.to(W.device) @ W.double()).to(W.dtype)
    mean = W.mean(0, keepdim=True)
    entries = {}
    for direction, wd in (('S', W), ('P', WP), ('O', WO)):
        for scale, suffix in ((1., '1'), (.1, '01')):
            ws = wd.clone() if scale == 1. else mean + scale * (wd - mean)
            bs = b.clone() if direction == 'S' and scale == 1. else b + (W-ws) @ mu
            entries[direction+suffix] = dict(W=ws, b=bs, direction=direction, scale=scale)
    w64 = W.double(); o64 = WO.double(); wc = w64-w64.mean(0)
    checks = dict(
        P_exact_column_multisets=bool(torch.equal(W.sort(0).values, WP.sort(0).values)),
        O_orthogonal_maxabs=float((Qcpu.T@Qcpu-torch.eye(k,dtype=torch.float64)).abs().max()),
        O_ones_maxabs=float((Qcpu@torch.ones(k,dtype=torch.float64)-1).abs().max()),
        O_gram_relative_error=float((o64.T@o64-w64.T@w64).norm()/w64.norm().square().clamp_min(1e-30)),
    )
    for key, item in entries.items():
        w, bb = item['W'].double(), item['b'].double()
        checks[key+'_mean_logits_maxabs'] = float(((w-w64)@mu.double()+bb-b.double()).abs().max())
        checks[key+'_centered_norm_relative_error'] = float(((w-w.mean(0)).norm(dim=0)-item['scale']*wc.norm(dim=0)).norm()/wc.norm().clamp_min(1e-30))
        if not all(torch.isfinite(a).all() for a in (w,bb)):
            raise FloatingPointError(key)
    assert checks['P_exact_column_multisets']
    assert checks['O_orthogonal_maxabs'] < 1e-12 and checks['O_ones_maxabs'] < 1e-12
    assert checks['O_gram_relative_error'] < 2e-6
    assert max(v for key,v in checks.items() if key.endswith('centered_norm_relative_error')) < 2e-6
    return dict(entries=entries, mu0=mu, zbar0=(mu@W.T+b).detach(),
                metadata=dict(P_permutation=permutation.cpu().tolist(), O_matrix=Qcpu.tolist(), checks=checks,
                              O_formula='e=ones/sqrt(K); B=Householder(e1->e)[:,1:]; U=QR(G) with positive R diagonal; Q=eeT+BUBT',
                              random_key=f'readout_separation_1111|O|{model}|{seed}|{t_end}'))


@torch.no_grad()
def make_head_bank(box, t_end):
    W, b, rows = box.head()
    model = 'mlp' if isinstance(box,D.MLP) else 'vit'
    mu = box.features().mean(0)
    bank = geometry_bank(W[rows], b[rows], mu, box.seed, model, t_end)
    bank['rows'] = rows.detach().clone()
    before = R.fingerprints(box)
    sp = R.snapshot(box)
    R.permute(box, t_end, 'S')
    wp, bp, rp = box.head()
    checks = bank['metadata']['checks']
    checks['canonical_P_weights_exact'] = bool(torch.equal(wp[rp], bank['entries']['P1']['W']))
    checks['canonical_P_bias_exact'] = bool(torch.equal(bp[rp], bank['entries']['P1']['b']))
    R.restore(box,sp)
    assert R.fingerprints(box) == before
    assert checks['canonical_P_weights_exact'] and checks['canonical_P_bias_exact']
    bank['metadata']['native_fingerprints'] = before
    return bank


@torch.no_grad()
def apply_physical(box, entry, clear_body=False):
    W, b, rows = box.head()
    W[rows] = entry['W']; b[rows] = entry['b']
    box.opt.state.pop(W,None); box.opt.state.pop(b,None)
    if clear_body:
        for p in box.body():
            box.opt.state.pop(p,None)


def arm_plan(model):
    arms = []
    for direction in ('S','P'):
        for scale, suffix in ((1.,'1'),(.1,'01')):
            for memory in ('keep','clear'):
                arms.append(dict(id=f'physical_{direction}{suffix}_body_{memory}',family='physical',bank=direction+suffix,
                                 direction=direction,scale=scale,body_memory=memory,cf=1.,body_gain=1.,jacobian='native'))
    for scale,suffix in ((1.,'1'),(.1,'01')):
        arms.append(dict(id=f'physical_O{suffix}_body_keep',family='physical',bank='O'+suffix,direction='O',scale=scale,
                         body_memory='keep',cf=1.,body_gain=1.,jacobian='native'))
    for direction in ('S','P'):
        for label,cf,gain in (('T',.1,1.),('G',1.,.1),('TG',.1,.1)):
            arms.append(dict(id=f'surrogate_{direction}_{label}',family='surrogate',bank=direction+'1',direction=direction,
                             scale=1.,body_memory='keep',cf=cf,body_gain=gain,jacobian='native'))
    modes = ('elu_unity',) if model=='mlp' else ('ln_remove_radial','ln_remove_radial_mean')
    for mode in modes:
        for scale,suffix in ((1.,'1'),(.1,'01')):
            arms.append(dict(id=f'jacobian_S{suffix}_{mode}',family='jacobian',bank='S'+suffix,direction='S',scale=scale,
                             body_memory='keep',cf=1.,body_gain=1.,jacobian=mode))
    assert len(arms) == (18 if model=='mlp' else 20)
    return arms

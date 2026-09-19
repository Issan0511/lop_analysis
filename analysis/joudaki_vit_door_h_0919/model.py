"""Unmodified upstream ViT; replace only the six FFN activation modules."""
import hashlib
import os
from pathlib import Path

import torch
from torch import nn

from src.rlcifar_mlp_battle_0918 import make_act
from .upstream.vit import VisionTransformer

UPSTREAM_COMMIT = '161217078ba52107c94a16602af958a321d62ce3'

# Door H [spec 追補 10].  Two arms added on top of the battle's fixed activations:
# the base activation with its per-channel running DC removed from the output,
#     out = phi_raw(z) - M,   M <- .99 M + .01 mean_{batch x token}(phi_raw(z))
# M is a buffer (no gradient), identical in train and eval, updated only after
# optimizer.step().  The frozen src/rlcifar_mlp_battle_0918.py is not touched:
# 'RH'/'GH' are resolved here to base arm + door_h, so make_act never sees them.
DOOR_BASE = {'RH': 'R', 'GH': 'GELU'}
# Kept in sync with the .99/.01 literals in update_ema, which mirror V's form exactly.
# Checks read it to derive tolerances: a first-order filter has gain 1/beta, so each
# step's rounding is amplified 1/beta at the fixed point.
DOOR_BETA = .01
# This experiment's launch order.  R and GELU are re-run under this build so the
# paired contrasts RH-R and GH-GELU sit inside one source-hash-consistent report.
ARM_ORDER = ('R', 'GELU', 'RH', 'GH')
MODEL_CONFIG = dict(img_size=64, patch_size=8, in_channels=3, num_classes=200,
                    embed_dim=384, depth=6, n_heads=6, mlp_ratio=4., qkv_bias=True,
                    dropout_p=.1, attn_drop_rate=.1, activation='gelu',
                    normalization='layer', normalization_affine=True)


def stream_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8],
                          'little') % (2**63 - 1)


class BattleActivation(nn.Module):
    def __init__(self, arm, width, seed, layer):
        super().__init__()
        self.arm, self.seed, self.layer = arm, seed, layer
        self.base_arm = DOOR_BASE.get(arm, arm)
        self.ref = make_act(self.base_arm)
        self.register_buffer('V', torch.ones(width) if self.ref.adaptive else None)
        # M rides in the state_dict so a task-boundary resume restores the EMA
        # rather than restarting it at zero.
        self.register_buffer('M', torch.zeros(width) if arm in DOOR_BASE else None)
        self.noise_seed = stream_seed('rsl', seed, layer)
        self.noise_gen = None
        self.pending_rng = None
        self.capture = False
        self.last_z = None
        self.pending_var = None
        self.pending_mean = None
        # Preflight only (S-off): hold M at zero so RH must reproduce R bit for bit.
        self.door_frozen = False

    def prepare(self, device):
        if self.base_arm == 'RSL' and self.noise_gen is None:
            self.noise_gen = torch.Generator(device=device).manual_seed(self.noise_seed)
            if self.pending_rng is not None:
                self.noise_gen.set_state(self.pending_rng.cpu())
                self.pending_rng = None
            self.ref.gen = self.noise_gen
            self.ref.noise = None

    def phi(self, z):
        self.prepare(z.device)
        if self.V is not None:
            self.ref.V = [self.V.unsqueeze(0)]
        return self.ref.phi(z.reshape(1, -1, z.shape[-1]), train=self.training).reshape_as(z)

    def forward(self, z):
        if self.capture:
            self.last_z = z.detach()
        out = self.phi(z)
        if self.training and self.V is not None:
            self.pending_var = z.detach().reshape(-1, z.shape[-1]).var(0, unbiased=False)
        if self.M is not None:
            # Same sample axis as V: batch x every token, CLS included.  The mean is
            # taken over phi_raw, before the subtraction, and only stashed here --
            # updating it inside the compiled forward broke gradient equivalence
            # for KKA once already [追補 5].
            if self.training:
                self.pending_mean = out.detach().reshape(-1, out.shape[-1]).mean(0)
            out = out - self.M
        return out

    @torch.no_grad()
    def update_ema(self):
        if self.pending_var is not None:
            self.V.mul_(.99).add_(.01 * self.pending_var)
            self.pending_var = None
        if self.pending_mean is not None:
            if not self.door_frozen:
                self.M.mul_(.99).add_(.01 * self.pending_mean)     # beta = .01, as V
            self.pending_mean = None

    def get_extra_state(self):
        return {'rng': self.noise_gen.get_state() if self.noise_gen is not None else self.pending_rng}

    def set_extra_state(self, state):
        if state['rng'] is not None:
            if self.noise_gen is not None:
                self.noise_gen.set_state(state['rng'].cpu())
            else:
                self.pending_rng = state['rng'].cpu()


def make_model(arm, seed, device='cuda', config=None):
    torch.manual_seed(seed)
    cfg = dict(MODEL_CONFIG if config is None else config)
    model = VisionTransformer(**cfg)
    for i in range(cfg['depth']):
        mlp = model.layers[f'block_{i}'].layers['mlp']
        mlp.layers['act'] = BattleActivation(arm, int(cfg['embed_dim'] * cfg['mlp_ratio']), seed, i)
    return model.to(device)


def activations(model):
    return [model.layers[f'block_{i}'].layers['mlp'].layers['act'] for i in range(model.depth)]


@torch.no_grad()
def reset_head(model):
    model.layers['out'].weight.zero_()
    model.layers['out'].bias.zero_()


def masked_logits(model, x, classes):
    return model(x).index_select(1, classes)


def prepare_noise(model, batch_size):
    """Keep RSL private RNG outside the compiled forward; identical elementwise draws."""
    tokens = model.layers['patch_embed'].n_patches + 1
    for module in activations(model):
        if module.base_arm == 'RSL':
            device = model.cls_token.device
            module.prepare(device)
            width = model.layers[f'block_{module.layer}'].layers['mlp'].layers['fc1'].out_features
            module.ref.noise = [torch.rand((1, batch_size * tokens, width),
                                           generator=module.noise_gen, device=device)]


def update_adaptive(model):
    for module in activations(model):
        module.update_ema()


def train_forward(model, engine):
    if engine != 'compile':
        return model
    headers = Path.home() / 'Projects/obsidian-research-data/joudaki_vit_door_h_0919/runtime/headers/usr/include'
    if headers.exists():
        include_paths = [str(headers / 'python3.12'), str(headers)]
        include_paths += os.environ.get('CPATH', '').split(os.pathsep)
        os.environ['CPATH'] = os.pathsep.join(dict.fromkeys(p for p in include_paths if p))
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '4')
    import torch._inductor.config as config
    config.fallback_random = True
    # Keep the original ATen dropout random stream, not Triton's alternative RNG.
    for module in activations(model):
        module.prepare(model.cls_token.device)
    return torch.compile(model, fullgraph=True)

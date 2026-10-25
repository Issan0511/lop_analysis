"""Exact cached prefix and trainable local fc1, retaining the original suffix."""
import common as T
torch, F = T.torch, T.F


class LocalFFN:
    def __init__(self, model, layer, x, classes):
        self.model, self.layer, self.classes = model, layer, classes
        self.model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
            p.grad = None
        self.fc1 = T.C.fc1(model, layer - 1)
        self.fc1.weight.requires_grad_(True)
        self.fc1.bias.requires_grad_(True)
        with torch.no_grad():
            hs, us = [], []
            for xx in x.split(32):
                layers = model.layers
                h = layers['patch_embed'](xx)
                h = torch.cat((model.cls_token.expand(h.shape[0], -1, -1), h), dim=1)
                h = layers['pos_drop'](h + model.pos_embed)
                for j in range(layer - 1):
                    h = layers[f'block_{j}'](h)
                block = layers[f'block_{layer - 1}'].layers
                h = h + block['attn'](block['norm1'](h))
                us.append(block['norm2'](h))
                hs.append(h)
            self.u, self.h = torch.cat(us), torch.cat(hs)
        self.mean = self.u[:, 0].double().mean(0)

    def forward(self, idx):
        block = self.model.layers[f'block_{self.layer - 1}'].layers
        mlp = block['mlp'].layers
        z = self.fc1(self.u.index_select(0, idx))
        a = mlp['act'](z)
        f = mlp['drop2'](mlp['fc2'](mlp['drop1'](a)))
        h = self.h.index_select(0, idx) + f
        for j in range(self.layer, self.model.depth):
            h = self.model.layers[f'block_{j}'](h)
        logits = self.model.layers['out'](self.model.layers['norm'](h)[:, 0])
        return logits.index_select(1, self.classes), z, a

    @torch.no_grad()
    def state(self):
        return dict(weight=self.fc1.weight.detach().cpu().clone(),
                    bias=self.fc1.bias.detach().cpu().clone())

    @torch.no_grad()
    def restore(self, state):
        self.fc1.weight.copy_(state['weight'])
        self.fc1.bias.copy_(state['bias'])

    def frozen_digest(self):
        prefix = f'layers.block_{self.layer - 1}.layers.mlp.layers.fc1.'
        return T.digest_tree({k: v for k, v in self.model.state_dict().items() if not k.startswith(prefix)})

    @torch.no_grad()
    def verify_cache(self, x):
        for lo in range(0, len(x), 32):
            idx = torch.arange(lo, min(len(x), lo + 32), device=x.device)
            logits, z, _ = self.forward(idx)
            direct, blocks = T.C.trace(self.model, x.index_select(0, idx))
            assert torch.equal(logits, direct.index_select(1, self.classes))
            assert torch.equal(z, blocks[self.layer - 1][1])
        return True


def fixed_metric(optimizer, fc1):
    metrics = []
    for p in (fc1.weight, fc1.bias):
        state = optimizer.state[p]
        t = float(state['step'])
        assert t > 0
        vhat = state['exp_avg_sq'].detach() / (1 - .999 ** t)
        d = 1 / (vhat.sqrt() + 1e-8)
        assert bool(torch.isfinite(d).all() and (d > 0).all())
        metrics.append(d.clone())
    return tuple(metrics)


def update(local, optimizer, metric, indices, y, mode, lr):
    optimizer.zero_grad(set_to_none=True)
    for idx in indices.split(32):
        logits, _, _ = local.forward(idx)
        loss = F.cross_entropy(logits, y.index_select(0, idx), reduction='sum') / len(indices)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('non-finite CE')
        loss.backward()
    torch.nn.utils.clip_grad_norm_(local.fc1.parameters(), float('inf'), error_if_nonfinite=True)
    if mode == 'adam':
        optimizer.step()
    else:
        assert mode == 'fixed'
        with torch.no_grad():
            for p, d in zip((local.fc1.weight, local.fc1.bias), metric):
                p.add_(d * p.grad, alpha=-lr)
    assert bool(torch.isfinite(local.fc1.weight).all() and torch.isfinite(local.fc1.bias).all())

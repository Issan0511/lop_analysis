"""Layerwise sinking pressures on frozen ViT checkpoints (post hoc)."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]


def sha(value):
    return hashlib.sha256(value).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def csv_write(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def body(m, x, capture=False):
    ls = m.layers
    h = ls['patch_embed'](x)
    h = torch.cat((m.cls_token.expand(len(x), -1, -1), h), dim=1)
    h = ls['pos_drop'](h + m.pos_embed)
    us, zs = [], []
    for i in range(m.depth):
        b = ls[f'block_{i}'].layers
        h = h + b['attn'](b['norm1'](h))
        u = b['norm2'](h)
        mlp = b['mlp'].layers
        z = mlp['fc1'](u)
        h = h + mlp['drop2'](mlp['fc2'](mlp['drop1'](mlp['act'](z))))
        if capture:
            us.append(u[:, 0])
            zs.append(z[:, 0])
    return ls['norm'](h)[:, 0], us, zs


@torch.no_grad()
def features(m, x, batch=32):
    return torch.cat([body(m, xx)[0] for xx in x.split(batch)])


@torch.no_grad()
def probe_fields(m, x):
    parts = [body(m, xx, True) for xx in x.split(32)]
    fields = []
    for i in range(m.depth):
        u = torch.cat([p[1][i] for p in parts]).double()
        z = torch.cat([p[2][i] for p in parts]).double()
        mu, sd, mean_u = z.mean(0), z.std(0, unbiased=True), u.mean(0)
        cov_uz = (z - mu).T @ (u - mean_u) / (len(x) - 1)
        fields.append(dict(mu=mu, sd=sd, r=mu / (sd + 1e-12), mean_u=mean_u, cov_uz=cov_uz,
                           alive=(z > 0).any(0), valid=sd > 1e-8))
    return fields


class Moments(torch.nn.Module):
    def __init__(self, m, raw):
        super().__init__()
        self.model = m
        self.register_buffer('mu_ref', torch.stack([r['mu'] for r in raw]).float())

    def forward(self, x):
        _, _, zs = body(self.model, x, True)
        moments = []
        for i, z in enumerate(zs):
            moments.extend([z.mean(0), (z - self.mu_ref[i]).square().mean(0)])
        return torch.stack(moments)


def project(m, xprobe, raw, gradients, gate=False):
    wrapper = Moments(m, raw)
    params = {f'model.{k}': p.detach() for k, p in m.named_parameters()}
    full, local = {}, {}
    def jvp(gg):
        tangent = {f'model.{k}': value for k, value in gg.items()}
        result = torch.zeros(2 * m.depth, 1536, device=xprobe.device, dtype=torch.float64)
        for xx in xprobe.split(16):
            with torch.no_grad():
                value = torch.func.jvp(lambda pp: torch.func.functional_call(wrapper, pp, (xx,)), (params,), (tangent,))[1]
            result += value.double() * len(xx) / len(xprobe)
        ds, dr, dmu = [], [], []
        for i, r in enumerate(raw):
            dm, dv = result[2 * i], result[2 * i + 1] * len(xprobe) / (len(xprobe) - 1)
            ss = r['sd'].clamp_min(1e-12)
            dsigma = dv / (2 * ss)
            dmu.append(dm)
            ds.append(dsigma)
            dr.append(dm / (ss + 1e-12) - r['mu'] * dsigma / (ss + 1e-12).square())
        return dict(mu=torch.stack(dmu), sd=torch.stack(ds), r=torch.stack(dr))
    for name, gg in gradients.items():
        full[name] = jvp(gg)
        lr, lm, ls = [], [], []
        for i, raw_i in enumerate(raw):
            prefix = f'layers.block_{i}.layers.mlp.layers.fc1.'
            gw, gb = gg[prefix + 'weight'].double(), gg[prefix + 'bias'].double()
            ss = raw_i['sd'].clamp_min(1e-12)
            dm = gw @ raw_i['mean_u'] + gb
            ds = (raw_i['cov_uz'] * gw).sum(1) / ss
            lm.append(dm)
            ls.append(ds)
            lr.append(dm / (ss + 1e-12) - raw_i['mu'] * ds / (ss + 1e-12).square())
        local[name] = dict(mu=torch.stack(lm), sd=torch.stack(ls), r=torch.stack(lr))
    checks = {}
    for key in ('mu', 'sd', 'r'):
        error = full['load'][key] + full['remainder'][key] - full['switch'][key]
        scale = max(float(full['load'][key].abs().max()), float(full['remainder'][key].abs().max()), 1e-12)
        checks['pressure_linearity_' + key] = float(error.abs().max()) / scale
        assert float(error.abs().max()) <= 1e-6 + 2e-3 * scale, checks
    if gate:
        for i in (0, m.depth - 1):
            gg = {k: torch.zeros_like(p) for k, p in m.named_parameters()}
            prefix = f'layers.block_{i}.layers.mlp.layers.fc1.'
            for key in gg:
                if key.startswith(prefix):
                    gg[key] = gradients['fit'][key]
            check = jvp(gg)
            for key in ('mu', 'sd', 'r'):
                expected = local['fit'][key][i]
                error = float((check[key][i] - expected).abs().max())
                scale = max(float(expected.abs().max()), 1e-12)
                checks[f'local_jvp_layer{i + 1}_{key}'] = error / scale
                assert error <= 1e-7 + 2e-3 * scale, checks
    return full, local, checks


def loading(h, labels, vc):
    q = F.one_hot(labels, num_classes=10).double() - .1
    alignment = q.T @ h.double()
    denom = alignment.square().sum()
    kappa = (vc.double() * alignment).sum() / denom
    norm = vc.double().square().sum()
    cosine2 = (vc.double() * alignment).sum().square() / (norm * denom) if norm > 0 else torch.tensor(0.)
    return (kappa * alignment).float(), float(kappa), float(cosine2)


def gradients(m, x, y, classes, vl):
    params = dict(m.named_parameters())
    with torch.no_grad():
        h = features(m, x)
        head = m.layers['out']
        w, b = head.weight[classes], head.bias[classes]
        vc = w - w.mean(0)
        logits = h @ w.T + b
        probs, target = logits.softmax(1), F.one_hot(y, num_classes=10).float()
        n = len(x)
        signals = dict(fit=((probs - target) @ vc) / n, switch=((probs - .1) @ vc) / n,
                       load=((probs - .1) @ vl) / n, remainder=((probs - .1) @ (vc - vl)) / n)
        hd = h.double().cpu()
        u, s, vh = torch.linalg.svd(hd, full_matrices=False)
        ranks = {tol: int((s.square() > tol * s[0].square()).sum()) for tol in (1e-6, 1e-8, 1e-10)}
        signals['P1'] = h / h.norm()
        for tol in (1e-8, 1e-6, 1e-10):
            if tol != 1e-8 and ranks[tol] == ranks[1e-8]:
                continue
            keep = s.square() > tol * s[0].square()
            sig = (u[:, keep] / s[keep]) @ vh[keep]
            signals['P2' if tol == 1e-8 else f'P2tol{tol:g}'] = (sig / sig.norm()).to(device=x.device, dtype=torch.float32)
        info = dict(ce=float(F.cross_entropy(logits, y)), accuracy=float((logits.argmax(1) == y).float().mean()),
                    pmax=float(probs.max(1).values.mean()), rank=ranks[1e-8], rank_1e6=ranks[1e-6], rank_1e10=ranks[1e-10])
    gs = {signal: {k: torch.zeros_like(p) for k, p in params.items()} for signal in signals}
    for start in range(0, len(x), 32):
        hh = body(m, x[start:start + 32])[0]
        for name, signal in signals.items():
            values = torch.autograd.grad(hh, list(params.values()), grad_outputs=signal[start:start + len(hh)],
                                         retain_graph=True, allow_unused=True)
            for (key, _), value in zip(params.items(), values):
                if value is not None:
                    gs[name][key] += value.detach()
    error = max(float((gs['load'][k] + gs['remainder'][k] - gs['switch'][k]).abs().max()) for k in params)
    scale = max(max(float(gs[name][k].abs().max()) for k in params) for name in ('load', 'remainder'))
    assert error <= 1e-6 + 1e-4 * scale
    info['gradient_linearity_relative_error'] = error / max(scale, 1e-12)
    info['signal_count'] = len(signals)
    return gs, info


def stats(values, valid):
    a = np.asarray(values)[valid]
    if not len(a):
        return dict(n=0, mean=None, median=None, sink_fraction=None, float_fraction=None, zero_fraction=None)
    tol = max(1e-12, 1e-6 * np.max(np.abs(a)))
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)), sink_fraction=float((a > tol).mean()),
                float_fraction=float((a < -tol).mean()), zero_fraction=float((np.abs(a) <= tol).mean()))


def compare(pred, actual, valid):
    tp = max(1e-12, 1e-6 * np.max(np.abs(pred)))
    ta = max(1e-12, 1e-6 * np.max(np.abs(actual)))
    keep = valid & (np.abs(pred) > tp) & (np.abs(actual) > ta)
    p, a = pred[keep], actual[keep]
    pos, neg = a > 0, a < 0
    ba = .5 * ((p[pos] > 0).mean() + (p[neg] < 0).mean()) if pos.sum() >= 10 and neg.sum() >= 10 else None
    rho = pd.Series(p).rank().corr(pd.Series(a).rank()) if len(a) > 2 else np.nan
    return dict(n=len(a), actual_sink_n=int(pos.sum()), actual_float_n=int(neg.sum()), balanced_accuracy=ba,
                sign_agreement=float(((p > 0) == pos).mean()) if len(a) else None,
                spearman=float(rho) if np.isfinite(rho) else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--archive-root', type=Path, required=True)
    p.add_argument('--raw-output', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(5)))
    args = p.parse_args()
    for folder in (args.output, args.raw_output):
        folder.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source_root))
    from analysis.vit_long_1024 import common as C
    from analysis.joudaki_vit_battle_0919.model import make_model, reset_head
    assert Path(C.__file__).resolve().is_relative_to(args.source_root.resolve())
    C.set_determinism(threads=2)
    torch.cuda.set_per_process_memory_fraction(.50)
    source = C.RealImages()
    states, pressures, comparisons, checks, inputs = [], [], [], [], []
    started = time.monotonic()
    for seed in args.seeds:
        archive = args.archive_root / f'vit_long_1024_{"vn" if seed < 3 else "il"}_raw_seed{seed}.tar'
        xprobe = C.eval_tensor(source.val(C.probe_names(source, seed)), 'cuda')
        with tarfile.open(archive, 'r:') as tar:
            def read(arm, file):
                return tar.extractfile(f'vit_long_1024/runs/{arm}/seed{seed}/{file}').read()
            for arm in ('SL', 'R0L'):
                prov = json.loads(read(arm, 'provenance.json'))
                for name, digest in prov['source_hashes'].items():
                    assert sha((args.source_root / name).read_bytes()) == digest
                cfg = prov['config']
                manifest = {r['path']: r for r in json.loads(read(arm, 'manifest.json'))}
                recorded = np.load(io.BytesIO(read(arm, 'intask.npz')))
                m = make_model('R', seed, 'cuda', cfg['model_config']).eval()
                for task in (10, 60):
                    data, _ = C.task_dataset(source, seed, task, 10, 25, augment=False)
                    previous, _ = C.task_dataset(source, seed, task - 1, 10, 25, augment=False)
                    x, y = C.eval_tensor(data.images, 'cuda'), data.labels.to('cuda')
                    oldx, oldy = C.eval_tensor(previous.images, 'cuda'), previous.labels.to('cuda')
                    classes = C.loss_classes(arm, seed, task, 10, 25)
                    idx = int(np.where(recorded['tasks'] == task)[0][0])
                    for stage, step, ckpt in (('start', 0, f'model_t{task - 1:02d}.pt'), ('mid', 100, f'mid_t{task:02d}.pt'), ('end', 500, f'model_t{task:02d}.pt')):
                        payload = read(arm, 'ckpt/' + ckpt)
                        assert sha(payload) == manifest['ckpt/' + ckpt]['sha256']
                        m.load_state_dict(torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True), strict=True)
                        inputs.append(dict(arm=arm, seed=seed, task=task, stage=stage, archive=str(archive), member='ckpt/' + ckpt, sha256=sha(payload)))
                        del payload
                        with torch.no_grad():
                            lc = C.loss_classes(arm, seed, task - 1 if stage == 'start' else task, 10, 25)
                            w = m.layers['out'].weight[lc]
                            vc = w - w.mean(0)
                            hh = features(m, oldx if stage == 'start' else x)
                            vl, kappa, cosine2 = loading(hh, oldy if stage == 'start' else y, vc)
                            if stage == 'start' and arm == 'R0L':
                                reset_head(m)
                                vl.zero_()
                        raw = probe_fields(m, xprobe)
                        gs, info = gradients(m, x, y, classes, vl)
                        recacc = float(recorded['mem_correct'][idx, step // 25]) / len(data)
                        recloss = float(recorded['mem_loss'][idx, step // 25]) / len(data)
                        assert abs(info['accuracy'] - recacc) < 1e-6
                        assert abs(info['ce'] - recloss) < 2e-4
                        full, local, validation = project(m, xprobe, raw, gs, gate=(seed == args.seeds[0] and task == 10 and stage == 'mid'))
                        ident = dict(arm=arm, seed=seed, task=task, stage=stage, step=step)
                        checks.append(dict(**ident, baseline_loss_abs_error=abs(info['ce'] - recloss), **validation))
                        npraw = {}
                        for i, r in enumerate(raw):
                            mu, sd, pos = [r[k].cpu().numpy() for k in ('mu', 'sd', 'r')]
                            alive, valid = [r[k].cpu().numpy() for k in ('alive', 'valid')]
                            state = dict(**ident, layer=i + 1, mu_median=float(np.median(mu)), sd_median=float(np.median(sd)),
                                         r_median=float(np.median(pos[valid])), r_mean=float(np.mean(pos[valid])), alive_fraction=float(alive.mean()),
                                         valid_n=int(valid.sum()), kappa=kappa, cos2_head_loading=cosine2, **info)
                            states.append(state)
                            npraw.update({f'L{i + 1}_{k}': r[k].cpu().numpy() for k in ('mu', 'sd', 'r', 'alive', 'valid')})
                            vectors = {}
                            for signal in gs:
                                vectors[signal] = full[signal]['r'][i].cpu().numpy()
                                for route, source_values in (('full', full), ('local', local)):
                                    for moment in ('mu', 'sd', 'r'):
                                        values = source_values[signal][moment][i].cpu().numpy()
                                        npraw[f'L{i + 1}_{signal}_{route}_{moment}'] = values
                                        for population, mask in (('all', valid), ('alive', valid & alive)):
                                            pressures.append(dict(**ident, layer=i + 1, signal=signal, route=route, moment=moment,
                                                                  population=population, **stats(values, mask)))
                                values = (full[signal]['r'][i] - local[signal]['r'][i]).cpu().numpy()
                                pressures.append(dict(**ident, layer=i + 1, signal=signal, route='upstream', moment='r', population='all', **stats(values, valid)))
                            for actual in ('switch', 'fit'):
                                for pred in ('P1', 'P2', *[k for k in gs if k.startswith('P2tol')]):
                                    for population, mask in (('all', valid), ('alive', valid & alive)):
                                        comparisons.append(dict(**ident, layer=i + 1, prediction=pred, actual=actual, population=population,
                                                                **compare(vectors[pred], vectors[actual], mask)))
                        np.savez_compressed(args.raw_output / f'{arm}_seed{seed}_t{task}_{stage}.npz', **npraw)
                        print(json.dumps(dict(**ident, ce=info['ce'], rank=info['rank'], elapsed=round(time.monotonic() - started, 1))), flush=True)
                        del gs, full, local
                    del x, y, oldx, oldy
                del m
                torch.cuda.empty_cache()
        for filename, values in (('states.csv', states), ('pressures.csv', pressures), ('comparisons.csv', comparisons)):
            csv_write(args.output / filename, values)
        dump(args.output / 'validation.json', checks)
        dump(args.output / 'checkpoint_provenance.json', inputs)
    dump(args.output / 'run.json', dict(status='complete', checkpoints=len(checks), seeds=args.seeds, seconds=time.monotonic() - started,
                                       source=str(C.__file__), torch=torch.__version__, gpu=torch.cuda.get_device_name(), training_updates=0))


if __name__ == '__main__':
    main()

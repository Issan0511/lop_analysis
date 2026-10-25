"""Saved Adam-denominator diagnostic at task-end states only."""
import argparse
import io
import json
from pathlib import Path
import sys
import tarfile
import time
import numpy as np
import torch
import measure as M


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--archive-root', type=Path, required=True)
    p.add_argument('--raw-output', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.source_root))
    from analysis.vit_long_1024 import common as C
    from analysis.joudaki_vit_battle_0919.model import make_model
    C.set_determinism(threads=2)
    torch.cuda.set_per_process_memory_fraction(.5)
    args.raw_output.mkdir(parents=True, exist_ok=True)
    source = C.RealImages()
    rows, comparisons, checks, provenance = [], [], [], []
    begin = time.monotonic()
    for seed in range(5):
        archive = args.archive_root / f'vit_long_1024_{"vn" if seed < 3 else "il"}_raw_seed{seed}.tar'
        xprobe = C.eval_tensor(source.val(C.probe_names(source, seed)), 'cuda')
        with tarfile.open(archive, 'r:') as tar:
            def read(arm, file):
                return tar.extractfile(f'vit_long_1024/runs/{arm}/seed{seed}/{file}').read()
            for arm in ('SL', 'R0L'):
                prov = json.loads(read(arm, 'provenance.json'))
                for name, digest in prov['source_hashes'].items():
                    assert M.sha((args.source_root / name).read_bytes()) == digest
                manifest = {r['path']: r for r in json.loads(read(arm, 'manifest.json'))}
                m = make_model('R', seed, 'cuda', prov['config']['model_config']).eval()
                for task in (10, 60):
                    loaded = {}
                    for kind in ('model', 'adam'):
                        name = f'ckpt/{kind}_t{task:02d}.pt'
                        payload = read(arm, name)
                        digest = M.sha(payload)
                        assert digest == manifest[name]['sha256']
                        loaded[kind] = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
                        provenance.append(dict(arm=arm, seed=seed, task=task, member=name, sha256=digest, archive=str(archive)))
                    m.load_state_dict(loaded['model'], strict=True)
                    opt = loaded['adam']['optimizer']
                    pg = opt['param_groups'][0]
                    params = dict(m.named_parameters())
                    assert len(params) == len(pg['params'])
                    denominator = {}
                    for index, (name, weight) in zip(pg['params'], params.items()):
                        state = opt['state'][index]
                        assert state['exp_avg_sq'].shape == weight.shape
                        denominator[name] = ((state['exp_avg_sq'] / (1 - pg['betas'][1] ** float(state['step']))).sqrt() + pg['eps']).to('cuda')
                    del loaded, opt
                    data, _ = C.task_dataset(source, seed, task, 10, 25, augment=False)
                    x, y = C.eval_tensor(data.images, 'cuda'), data.labels.to('cuda')
                    classes = C.loss_classes(arm, seed, task, 10, 25)
                    with torch.no_grad():
                        w = m.layers['out'].weight[classes]
                        vl, _, _ = M.loading(M.features(m, x), y, w - w.mean(0))
                    raw = M.probe_fields(m, xprobe)
                    gs, info = M.gradients(m, x, y, classes, vl)
                    weighted = {signal: {name: value / denominator[name] for name, value in gg.items()} for signal, gg in gs.items()}
                    full, local, validation = M.project(m, xprobe, raw, weighted)
                    ident = dict(arm=arm, seed=seed, task=task, stage='end', step=500, metric='Adam_fixed')
                    checks.append(dict(**ident, **validation))
                    arrays = {}
                    for i, r in enumerate(raw):
                        valid, alive = r['valid'].cpu().numpy(), r['alive'].cpu().numpy()
                        vectors = {sig: full[sig]['r'][i].cpu().numpy() for sig in gs}
                        for signal in gs:
                            for route, values_by_signal in (('full', full), ('local', local)):
                                for moment in ('mu', 'sd', 'r'):
                                    value = values_by_signal[signal][moment][i].cpu().numpy()
                                    arrays[f'L{i + 1}_{signal}_{route}_{moment}'] = value
                                    for population, mask in (('all', valid), ('alive', valid & alive)):
                                        rows.append(dict(**ident, layer=i + 1, signal=signal, route=route, moment=moment,
                                                         population=population, **M.stats(value, mask)))
                            value = (full[signal]['r'][i] - local[signal]['r'][i]).cpu().numpy()
                            rows.append(dict(**ident, layer=i + 1, signal=signal, route='upstream', moment='r', population='all', **M.stats(value, valid)))
                        for actual in ('switch', 'fit'):
                            for pred in ('P1', 'P2', *[key for key in gs if key.startswith('P2tol')]):
                                for population, mask in (('all', valid), ('alive', valid & alive)):
                                    comparisons.append(dict(**ident, layer=i + 1, prediction=pred, actual=actual, population=population,
                                                            **M.compare(vectors[pred], vectors[actual], mask)))
                    np.savez_compressed(args.raw_output / f'{arm}_seed{seed}_t{task}.npz', **arrays)
                    print(json.dumps(dict(**ident, elapsed=round(time.monotonic() - begin, 1))), flush=True)
                    del gs, weighted, full, local, denominator, x, y
                del m
                torch.cuda.empty_cache()
        M.csv_write(args.output / 'adam_pressures.csv', rows)
        M.csv_write(args.output / 'adam_comparisons.csv', comparisons)
        M.dump(args.output / 'adam_validation.json', checks)
        M.dump(args.output / 'adam_provenance.json', provenance)
    M.dump(args.output / 'adam_run.json', dict(status='complete', states=len(checks), seconds=time.monotonic() - begin))


if __name__ == '__main__':
    main()

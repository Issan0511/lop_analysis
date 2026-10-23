"""Measure speed only on synthetic seed100 inputs; no production endpoints."""
import argparse
import json
import time
import traceback

import torch
from torch.nn import functional as F

from .model import make_model, prepare_noise, train_forward, update_adaptive
from .run import RAW, atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', default='GELU')
    p.add_argument('--engine', choices=['eager', 'fused', 'compile'], required=True)
    p.add_argument('--batch', type=int, default=128)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    result = dict(arm=args.arm, engine=args.engine, batch=args.batch, tf32=False, dtype='float32')
    started = time.monotonic()
    model = make_model(args.arm, 100)
    x = torch.randn(args.batch, 3, 64, 64, device='cuda')
    y = torch.randint(5, (args.batch,), device='cuda')
    opt = torch.optim.Adam(model.parameters(), lr=1e-4,
                           **({'foreach': True} if args.engine == 'eager' else {'fused': True}))
    fn = train_forward(model, args.engine)
    print(json.dumps(dict(event='warmup', **result)), flush=True)
    try:
        for step in range(110):
            if step == 10:
                torch.cuda.synchronize()
                result['warmup_seconds'] = time.monotonic() - started
                started_measure = time.monotonic()
                print(json.dumps(dict(event='measure', **result)), flush=True)
            opt.zero_grad(set_to_none=True)
            prepare_noise(model, args.batch)
            loss = F.cross_entropy(fn(x)[:, :5], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True, foreach=True)
            opt.step()
            update_adaptive(model)
        torch.cuda.synchronize()
        result['ms_per_step'] = (time.monotonic() - started_measure) * 1000 / 100
        result['peak_gb'] = torch.cuda.max_memory_allocated() / 1e9
        result['all_runs_training_hours'] = result['ms_per_step'] / 1000 * 20000 * 130 / 3600
        result['status'] = 'ok'
    except Exception as error:
        result.update(status='error', error=repr(error), traceback=traceback.format_exc())
    target = RAW / 'preflight' / f'benchmark_{args.arm}_{args.engine}_b{args.batch}.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(target, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

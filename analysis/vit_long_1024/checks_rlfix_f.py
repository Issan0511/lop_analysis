"""Checks of addendum F (specs/spec_vit_long_1024.md 追補 F.7) before the run. Needs a clean, committed tree.

F1  S, SC and SN start from the same state_dict (each of the five seeds): the same keys, every tensor equal and
    the same parameter count as S (the centring and the RMSNorm without gain have none). The norm positions hold
    Identity except block 1's norm2 in SC (Center) and the final norm in SN (RMSNorm, eps 1e-5, no gain). The old
    arms build as before: their initial weights equal every model_t00.pt of addenda D and E on this machine.
F2  SC's forward with S's weights (initial, and S's task-30 state of addendum D; 128 task-1 images, GPU): every
    residual stage up to the last FFN equals S's bit for bit, the last FFN's input equals h - mean(h) of S's
    residual h bit for bit, the last residual equals S's residual plus SC's FFN output (the centring reaches the
    residual only through the FFN), and model(x) equals trace(x). At the initial state the last FFN's input has
    |mean over batch and tokens| < 1e-5 for every feature (spec); at task 30 the size is recorded.
F3  SN's head input (the final norm's output at the CLS token) has RMS 1: for every image it equals
    sqrt(ms / (ms + 1e-5)) of the incoming CLS residual (ms its mean square) within 16 float32 ulp (the float32
    mean of 384 squares, ceil(log2 384) = 9 ulp, rsqrt 2 ulp, the products), so |RMS - 1| is the eps term only;
    initial and task-30 states. S's head input is not at RMS 1.
F4  Smoke runs of 50 updates (real data, production_2blk_rlfix_f.json, seed 10252010): R0, S, SC and SN for
    2 tasks x 25 updates, fresh, freshC and freshN on task 25 for 50 updates, all rc 0 and complete; R0 zeroes the
    head at both tasks and the shared arms at task 1 only (R0's task 2 starts at ln 10), loss rows are the fixed
    classes, the provenance records the norm kind on the no-norm model config, and SC's and SN's model_t00.pt
    equal S's.
F5  The new code's S (seed 10252010, production_2blk_rlfix.json, 1 task x 200 updates) repeats addendum D's run
    (runs/S/seed10252010, white-san) bit for bit: the initial weights (model_t00.pt), the weights at update 100
    (mid_t01.pt), every update's loss, correct count and batch size, the 25-update records (memorisation, fc1 row
    norms, head rows), the per_task.csv values of the first 200 updates and the probe's initial layer statistics.
    production_2blk_rlfix_f.json gives addendum E's task plan for every seed and task.
F6  git_state is clean (a committed tree).

    python -m analysis.vit_long_1024.checks_rlfix_f              # white-san（F1〜F6）
    python -m analysis.vit_long_1024.checks_rlfix_f --remote     # 他の機械（white-san の記録と比べる F1 の旧腕・F2/F3 の課題 30・F5 を飛ばす）

The record is results/vit_long_1024_2blk_rlfix/checks_f_<hostname>.json; the smoke runs are in <raw>/_checks_f/.
"""
import csv
import json
import math
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
RAW = Path(os.environ.get('VIT_LONG_RAW', str(Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_rlfix')))
OUT = REPO / 'results/vit_long_1024_2blk_rlfix'
PROD_F, PROD_E, PROD_D = 'production_2blk_rlfix_f.json', 'production_2blk_rlfix_e.json', 'production_2blk_rlfix.json'
SEED = 10252010
D_S = RAW / 'runs/S/seed10252010'                 # addendum D's S, run on white-san (追補 D.7)
BOX = ['--n-labels', '10', '--images-per-class', '250', '--augment', 'off', '--dropout', '0.0', '--depth', '2',
       '--mlp-ratio', '8.0']
IDENTITY = dict.fromkeys(['block_0.norm1', 'block_0.norm2', 'block_1.norm1', 'block_1.norm2', 'norm'], 'Identity')
NORMS = {'S': IDENTITY, 'SC': dict(IDENTITY, **{'block_1.norm2': 'Center'}),
         'SN': dict(IDENTITY, norm='RMSNorm(eps=1e-05, affine=False)')}
INTASK_STEPS = ('online_correct', 'online_loss', 'batch_size')           # per update
INTASK_EVALS = ('mem_correct', 'mem_loss', 'up_norm_med', 'head_row_norm', 'head_bias')   # every 25 updates
FIRST_200 = ('mem_acc_start', 'mem_loss_start', 'head_full_norm_start', 'head_norm_start', 'up_norm_med_start',
             'online_acc_first100')


def env(production):
    return dict(os.environ, PYTHONPATH=str(REPO), VIT_LONG_PRODUCTION=production, CUBLAS_WORKSPACE_CONFIG=':4096:8')


def child(production, code, *args):
    out = subprocess.run([PY, '-c', code, *map(str, args)], cwd=REPO, env=env(production), check=True,
                         capture_output=True, text=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def smoke(arm, production, out, *args):
    """run.py --smoke in the box's settings; returns the return code (the run's output is out/console.log)."""
    if out.exists():
        subprocess.run(['rm', '-rf', str(out)], check=True)
    out.mkdir(parents=True)
    with open(out / 'console.log', 'w') as log:
        return subprocess.run([PY, '-m', 'analysis.vit_long_1024.run', '--arm', arm, '--out', str(out), '--smoke', *BOX,
                               *map(str, args)], cwd=REPO, env=env(production), stdout=log, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL).returncode


def per_task(d):
    return list(csv.DictReader(open(d / 'per_task.csv')))


F1 = r'''
import json, sys, torch
from pathlib import Path
from analysis.vit_long_1024 import common as C
raw, recorded_too = Path(sys.argv[1]), sys.argv[2] == '1'
def kind(module):
    if isinstance(module, torch.nn.RMSNorm):
        return f'RMSNorm(eps={module.eps}, affine={module.elementwise_affine})'
    return type(module).__name__
def kinds(model):
    out = {}
    for position in C.norm_positions(model.depth):
        layers, name = C.norm_slot(model, position)
        out[position] = kind(layers[name])
    return out
def same(x, y):
    return torch.equal(x, y) if torch.is_tensor(x) else x == y
seeds = {}
for seed in C.production()['seeds']:
    models = {arm: C.build_arm_model(arm, seed, 'cpu', 0.0, 2, 8.0) for arm in ('S', 'SC', 'SN')}
    sd = {arm: m.state_dict() for arm, m in models.items()}
    seeds[seed] = dict(keys_equal=all(list(s) == list(sd['S']) for s in sd.values()),
                       tensors_equal=all(same(s[k], sd['S'][k]) for s in sd.values() for k in sd['S']),
                       sha={arm: C.state_sha256(s) for arm, s in sd.items()},
                       n_params={arm: sum(p.numel() for p in m.parameters()) for arm, m in models.items()},
                       n_params_expected=C.n_params_expected(2, 8.0, False),
                       norms={arm: kinds(m) for arm, m in models.items()})
recorded = {}
for path in sorted(raw.glob('runs/*/seed*/ckpt/model_t00.pt')) if recorded_too else []:
    arm, seed = path.parts[-4], int(path.parts[-3][len('seed'):])
    built = C.state_sha256(C.build_arm_model(arm, seed, 'cpu', 0.0, 2, 8.0).state_dict())
    recorded[f'{arm} {seed}'] = built == C.state_sha256(torch.load(path, map_location='cpu', weights_only=True))
print(json.dumps(dict(seeds=seeds, recorded_t00=recorded)))
'''

F23 = r'''
import json, sys, torch
from analysis.vit_long_1024 import common as C
C.set_determinism()
device, seed, t30 = 'cuda', int(sys.argv[1]), sys.argv[2]
data, _ = C.task_dataset(C.image_source('real'), seed, 1, 10, 250, augment=False)
x = C.eval_tensor(data.images[:C.BATCH], device)
def stages(m):
    """Residual after the embedding and after each attention and FFN, the last FFN's input, the head input."""
    L = m.layers
    h = L['patch_embed'](x)
    h = L['pos_drop'](torch.cat((m.cls_token.expand(h.shape[0], -1, -1), h), dim=1) + m.pos_embed)
    res = [h]
    for i in range(m.depth):
        b = L[f'block_{i}'].layers
        h = h + b['attn'](b['norm1'](h))
        res.append(h)
        u = b['norm2'](h)
        h = h + b['mlp'](u)
        res.append(h)
    return res, u, L['norm'](h)[:, 0]
def rms(v):
    return v.double().pow(2).mean(-1).sqrt()
out = {}
states = {'init': None} | ({'t30': torch.load(t30, map_location=device, weights_only=True)} if t30 else {})
with torch.no_grad():
    for name, state in states.items():
        m = {arm: C.build_arm_model(arm, seed, device, 0.0, 2, 8.0).eval() for arm in ('S', 'SC', 'SN')}
        if state is not None:
            for model in m.values():
                model.load_state_dict(state)
        (rS, uS, yS), (rC, uC, yC), (rN, uN, yN) = (stages(m[a]) for a in ('S', 'SC', 'SN'))
        mlp = m['SC'].layers['block_1'].layers['mlp']
        mean_u = uC.double().mean(dim=(0, 1)).abs()
        ms = rN[-1][:, 0].double().pow(2).mean(-1)
        expected = (ms / (ms + 1e-5)).sqrt()
        out[name] = dict(
            F2=dict(residual_equal_to_last_ffn=[bool(torch.equal(a, b)) for a, b in zip(rC[:-1], rS[:-1])],
                    input_is_centred_h=bool(torch.equal(uC, rS[-2] - rS[-2].mean(dim=(0, 1), keepdim=True))),
                    last_residual_is_s_plus_ffn=bool(torch.equal(rC[-1], rS[-2] + mlp(uC))),
                    last_residual_differs_from_s=not bool(torch.equal(rC[-1], rS[-1])),
                    model_equals_trace=bool(torch.equal(m['SC'](x), C.trace(m['SC'], x)[0])),
                    mean_abs_max=float(mean_u.max()), input_rms=float(rms(uS).mean()),
                    s_mean_norm=float(rS[-2].double().mean(dim=(0, 1)).norm())),
            F3=dict(residual_equal=all(bool(torch.equal(a, b)) for a, b in zip(rN, rS)),
                    rms_minus_expected_max=float((rms(yN) - expected).abs().max()),
                    rms_minus_one_max=float((rms(yN) - 1).abs().max()), eps_term_max=float((1 - expected).max()),
                    ms_min=float(ms.min()), s_head_rms_min=float(rms(yS).min()), s_head_rms_max=float(rms(yS).max()),
                    s_rms_minus_one_max=float((rms(yS) - 1).abs().max())))
print(json.dumps(out))
'''

F4_STATES = r'''
import json, sys, torch
from analysis.vit_long_1024 import common as C
print(json.dumps({p: C.state_sha256(torch.load(p, map_location='cpu', weights_only=True)) for p in sys.argv[1:]}))
'''

F5_STATES = r'''
import json, sys, torch
from analysis.vit_long_1024 import common as C
pairs = [sys.argv[i:i + 2] for i in range(1, len(sys.argv), 2)]
sha = lambda p: C.state_sha256(torch.load(p, map_location='cpu', weights_only=True))
print(json.dumps([dict(new=sha(a), recorded=sha(b)) for a, b in pairs]))
'''

PLANS = r'''
import json
from analysis.vit_long_1024 import common as C
print(json.dumps({s: [C.task_plan(s, t, 10, 250) for t in range(1, 101)] for s in C.production()['seeds']}))
'''

GIT = r'''
import json
from analysis.vit_long_1024 import common as C
head, dirty = C.git_state()
print(json.dumps(dict(head=head, dirty=dirty, production=str(C.PRODUCTION_FILE.name))))
'''


def f4(root):
    rows, codes = {}, {}
    fixed = None
    for arm in ('R0', 'S', 'SC', 'SN', 'fresh', 'freshC', 'freshN'):
        d = root / f'smoke_{arm}'
        fresh = arm.startswith('fresh')
        codes[arm] = smoke(arm, PROD_F, d, '--seed', SEED, *(('--fresh-task', 25, '--steps', 50) if fresh else
                                                             ('--tasks', 2, '--steps', 25)))
        if codes[arm]:
            continue
        rr, prov = per_task(d), json.loads((d / 'provenance.json').read_text())
        fixed = fixed or json.loads(rr[0]['classes'])
        rows[arm] = dict(done=json.loads((d / 'done.json').read_text())['status'],
                         head_zeroed=[int(r['head_zeroed']) for r in rr],
                         head_full_norm_start=[float(r['head_full_norm_start']) for r in rr],
                         loss_rows=[json.loads(r['loss_rows']) for r in rr],
                         mem_loss_start=[float(r['mem_loss_start']) for r in rr],
                         norm=prov['config'].get('norm'), normalization=prov['config']['model_config']['normalization'],
                         ms_per_update_last_task=1000 * float(rr[-1]['train_seconds']) / int(rr[-1]['steps']))
    t00 = child(PROD_F, F4_STATES, *(root / f'smoke_{a}/ckpt/model_t00.pt' for a in ('S', 'SC', 'SN') if a in rows))
    shas = list(t00.values())
    result = dict(rc=codes, runs=rows, fixed_classes=fixed, t00_equal_S=len(shas) == 3 and len(set(shas)) == 1)
    want_norm = {'R0': None, 'S': None, 'SC': 'center', 'SN': 'final_rms', 'fresh': None, 'freshC': 'center',
                 'freshN': 'final_rms'}
    result['pass'] = (all(c == 0 for c in codes.values()) and len(rows) == 7 and result['t00_equal_S']
                      and all(r['done'] == 'complete' and r['normalization'] == 'none' and r['norm'] == want_norm[a]
                              and all(x == fixed for x in r['loss_rows']) for a, r in rows.items())
                      and rows['R0']['head_zeroed'] == [1, 1] and rows['R0']['head_full_norm_start'] == [0.0, 0.0]
                      and abs(rows['R0']['mem_loss_start'][1] - math.log(10)) < 1e-5
                      and all(rows[a]['head_zeroed'] == [1, 0] and rows[a]['head_full_norm_start'][1] > 0
                              for a in ('S', 'SC', 'SN'))
                      and all(rows[a]['head_zeroed'] == [1] for a in ('fresh', 'freshC', 'freshN')))
    return result


def f5(root):
    d = root / 'smoke_S_D200'
    rc = smoke('S', PROD_D, d, '--seed', SEED, '--tasks', 1, '--steps', 200)
    if rc:
        return {'rc': rc, 'pass': False}
    states = child(PROD_D, F5_STATES, d / 'ckpt/model_t00.pt', D_S / 'ckpt/model_t00.pt',
                   d / 'ckpt/mid_t01.pt', D_S / 'ckpt/mid_t01.pt')
    new, rec = np.load(d / 'intask.npz'), np.load(D_S / 'intask.npz')
    points = 200 // 25 + 1
    intask = {k: bool(np.array_equal(new[k][0], rec[k][0, :200])) for k in INTASK_STEPS}
    intask |= {k: bool(np.array_equal(new[k][0], rec[k][0, :points])) for k in INTASK_EVALS}
    a, b = per_task(d)[0], per_task(D_S)[0]
    task_row = {k: a[k] == b[k] for k in FIRST_200}
    task_row['online_chunks_1_2'] = json.loads(a['online_chunks']) == json.loads(b['online_chunks'])[:2]
    init_rows = [[r for r in csv.DictReader(open(x / 'layer_stats.csv')) if r['task'] == '0'] for x in (d, D_S)]
    plans_f, plans_e = child(PROD_F, PLANS), child(PROD_E, PLANS)
    result = dict(rc=rc, recorded=str(D_S), init_weights_equal=states[0]['new'] == states[0]['recorded'],
                  update100_weights_equal=states[1]['new'] == states[1]['recorded'], intask_equal=intask,
                  per_task_equal=task_row, init_layer_stats_equal=init_rows[0] == init_rows[1] and len(init_rows[0]) == 2,
                  plan_f_equals_e=plans_f == plans_e, last_loss=[float(new['online_loss'][0, -1]),
                                                                  float(rec['online_loss'][0, 199])])
    result['pass'] = (result['init_weights_equal'] and result['update100_weights_equal'] and all(intask.values())
                      and all(task_row.values()) and result['init_layer_stats_equal'] and result['plan_f_equals_e'])
    return result


def main(remote, root):
    """F1-F6 with the smoke runs under root; returns the report. Addendum G's checks call it with their own root
    (checks_rlfix_g G6), so that neither this record nor _checks_f/ is rewritten."""
    root.mkdir(parents=True, exist_ok=True)
    git = child(PROD_F, GIT)
    report = dict(git_hash=git['head'], remote=remote)
    report['F6'] = dict(git, **{'pass': git['dirty'] == '' and git['production'] == PROD_F})

    f1 = child(PROD_F, F1, RAW, int(not remote))
    f1['pass'] = (all(v['keys_equal'] and v['tensors_equal'] and len(set(v['sha'].values())) == 1
                      and set(v['n_params'].values()) == {v['n_params_expected']} and v['norms'] == NORMS
                      for v in f1['seeds'].values())
                  and (remote or (len(f1['recorded_t00']) > 0 and all(f1['recorded_t00'].values()))))
    report['F1'] = f1

    f23 = child(PROD_F, F23, SEED, '' if remote else D_S / 'ckpt/model_t30.pt')
    tol = 16 * 2.0 ** -24
    report['F2'] = dict({k: v['F2'] for k, v in f23.items()})
    report['F2']['pass'] = (all(all(v['F2']['residual_equal_to_last_ffn']) and v['F2']['input_is_centred_h']
                                and v['F2']['last_residual_is_s_plus_ffn'] and v['F2']['last_residual_differs_from_s']
                                and v['F2']['model_equals_trace'] for v in f23.values())
                            and f23['init']['F2']['mean_abs_max'] < 1e-5)
    report['F3'] = dict({k: v['F3'] for k, v in f23.items()}, tolerance=tol)
    report['F3']['pass'] = all(v['F3']['residual_equal'] and v['F3']['rms_minus_expected_max'] < tol
                               and v['F3']['rms_minus_one_max'] <= v['F3']['eps_term_max'] + tol
                               and v['F3']['s_rms_minus_one_max'] > v['F3']['eps_term_max'] + tol for v in f23.values())

    report['F4'] = f4(root)
    report['F5'] = {'pass': 'skipped'} if remote else f5(root)
    report['host'] = socket.gethostname()
    verdict = {k: report[k]['pass'] for k in ('F1', 'F2', 'F3', 'F4', 'F5', 'F6')}
    report['pass_'] = all(v in (True, 'skipped') for v in verdict.values())
    return report


if __name__ == '__main__':
    remote = '--remote' in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    report = main(remote, RAW / '_checks_f')
    verdict = {k: report[k]['pass'] for k in ('F1', 'F2', 'F3', 'F4', 'F5', 'F6')}
    (OUT / f'checks_f_{socket.gethostname()}.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(dict(verdict, pass_=report['pass_'], git_hash=report['git_hash']), ensure_ascii=False))
    sys.exit(0 if report['pass_'] else 1)

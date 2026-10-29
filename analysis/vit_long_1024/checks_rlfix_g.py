"""Checks of addendum G (specs/spec_vit_long_1024.md 追補 G.6) before the run. Needs a clean, committed tree.

G1  SL, SLW and SLP start from the same state_dict (each of the five seeds): the same keys, every tensor equal and
    the LayerNorm model's parameter count; only SLW and SLP get a HeadHold, and both are shared, LN arms whose fresh
    net is freshL. The built SL equals every recorded SL model_t00.pt (addenda D and E).
G2  SLW's update: one update of SL and of SLW from the same state on the same batch (run.py's update in eager mode,
    GPU) leaves SLW's head weight equal to SL's times 1 - LR * HEAD_WD bit for bit, and every other tensor (the
    head's bias, the body, Adam's moments and steps) equal to SL's. From the initial state (task 1's first update)
    and from SL's task-10 end (seed 10252011, addendum E on white-san: weights and Adam) on task 11's first batch.
G3  SLP's update from the same two states. With no target (task 1) SLP equals SL bit for bit. With the target (the
    row norms of SL's model_t01.pt) from the task-10 end: every head row's norm equals its target within 1e-6
    relative, the rows whose target is 0 stay 0, the weight equals SL's updated rows rescaled to the target bit for
    bit (SL's direction, cos 1), every other tensor equals SL's, and SL's own rows are off the target (not trivial).
G4  Smoke runs (real data, production_2blk_rlfix_g.json, seed 10252011, 2 tasks x 25 updates, compile): SL, SLW
    and SLP rc 0 and complete; the head is zeroed before task 1 only, the loss rows are task 1's, the provenance
    records head_hold for SLW and SLP only and model_t00.pt is the same for the three. SLP's task 1 equals SL's
    bit for bit (every update and every 25-update record) and SLW's does not; SLP's head rows at task 2's update 25
    have their task-1-end norms within 1e-6 relative, SL's do not. SLP stopped after task 1 and resumed equals the
    uninterrupted SLP bit for bit (the target is restored from the checkpoint).
G5  The new code's SL (seed 10252011, production_2blk_rlfix_e.json, 1 task x 200 updates) repeats addendum E's run
    (runs/SL/seed10252011, white-san) bit for bit: the initial weights (model_t00.pt), every update's loss, correct
    count and batch size, the 25-update records (memorisation, fc1 row norms, head rows), the per_task.csv values
    of the first 200 updates and the probe's initial layer statistics. SLP (production_2blk_rlfix_g.json) repeats
    the same record (its task 1 is SL's). production_2blk_rlfix_g.json gives addendum E's task plan for every seed
    and task. (Seed 10252010's SL is addendum D's run on a rented machine whose rounding differs from white-san's
    from update 33 on, 追補 D.7, so it cannot be the record here.)
G6  checks_rlfix_f.py's F1-F6 pass with this code; its smoke runs and its record go to <raw>/_checks_g/f_rerun/
    (checks_rlfix_f.main), so results/ and _checks_f/ are not rewritten.
G7  git_state is clean (a committed tree) with production_2blk_rlfix_g.json.

    python -m analysis.vit_long_1024.checks_rlfix_g              # white-san（G1〜G7）
    python -m analysis.vit_long_1024.checks_rlfix_g --remote     # 他の機械（white-san の記録と比べる G1 の記録・G2/G3 の課題 10 末・G5 を飛ばし、G6 は F の --remote）

The record is <raw>/_checks_g/checks_g_<hostname>.json and the smoke runs are in <raw>/_checks_g/ (results/ is not
written: another session writes there on 1029).
"""
import csv
import json
import socket
import subprocess
import sys

import numpy as np

from analysis.vit_long_1024 import checks_rlfix_f as F

REPO, RAW, PY = F.REPO, F.RAW, sys.executable
PROD_G, PROD_E = 'production_2blk_rlfix_g.json', 'production_2blk_rlfix_e.json'
SEED = 10252011
E_SL = RAW / 'runs/SL/seed10252011'               # addendum E's SL, run on white-san (追補 E.2)
ARMS = ('SL', 'SLW', 'SLP')
TIMES = ('train_seconds', 'record_seconds', 'other_seconds', 'seconds')
REL_TOL = 1e-6


def child(production, code, *args):
    """F.child, with the child's stderr shown when it fails."""
    out = subprocess.run([PY, '-c', code, *map(str, args)], cwd=REPO, env=F.env(production), capture_output=True,
                         text=True)
    if out.returncode:
        sys.stderr.write(out.stderr)
        raise RuntimeError(f'child process failed with rc {out.returncode}')
    return json.loads(out.stdout.strip().splitlines()[-1])


def smoke(arm, production, out, *args, clean=True):
    """run.py --smoke in the box's settings; returns the return code (the run's output is out/console.log).
    clean=False keeps out, so that the same command resumes from its checkpoint."""
    if clean and out.exists():
        subprocess.run(['rm', '-rf', str(out)], check=True)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'console.log', 'a') as log:
        return subprocess.run([PY, '-m', 'analysis.vit_long_1024.run', '--arm', arm, '--out', str(out), '--smoke',
                               *F.BOX, *map(str, args)], cwd=REPO, env=F.env(production), stdout=log,
                              stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode


G1 = r'''
import json, sys, torch
from pathlib import Path
from analysis.vit_long_1024 import common as C
raw, recorded_too = Path(sys.argv[1]), sys.argv[2] == '1'
ARMS = ('SL', 'SLW', 'SLP')
def same(x, y):
    return torch.equal(x, y) if torch.is_tensor(x) else x == y
seeds = {}
for seed in C.production()['seeds']:
    models = {arm: C.build_arm_model(arm, seed, 'cpu', 0.0, 2, 8.0) for arm in ARMS}
    sd = {arm: m.state_dict() for arm, m in models.items()}
    seeds[seed] = dict(keys_equal=all(list(s) == list(sd['SL']) for s in sd.values()),
                       tensors_equal=all(same(s[k], sd['SL'][k]) for s in sd.values() for k in sd['SL']),
                       sha={arm: C.state_sha256(s) for arm, s in sd.items()},
                       n_params={arm: sum(p.numel() for p in m.parameters()) for arm, m in models.items()},
                       n_params_expected=C.n_params_expected(2, 8.0, True),
                       hold={arm: type(C.head_hold(arm, m)).__name__ for arm, m in models.items()})
recorded = {}
for path in sorted(raw.glob('runs/SL/seed*/ckpt/model_t00.pt')) if recorded_too else []:
    seed = int(path.parts[-3][len('seed'):])
    recorded[seed] = seeds[seed]['sha']['SL'] == C.state_sha256(torch.load(path, map_location='cpu', weights_only=True))
print(json.dumps(dict(seeds=seeds, recorded_t00=recorded, lr=C.LR, head_wd=C.HEAD_WD,
                      head_hold={arm: C.head_hold_config(arm) for arm in ARMS},
                      arms=dict(shared=[a in C.SHARED_ARMS for a in ARMS[1:]], ln=[a in C.LN_ARMS for a in ARMS[1:]],
                                fresh_for=[C.FRESH_FOR[a] for a in ARMS[1:]]))))
'''

G23 = r'''
import copy, json, sys, torch
from torch.nn import functional as F
from analysis.joudaki_vit_battle_0919.data import train_loader
from analysis.joudaki_vit_battle_0919.model import prepare_noise, reset_head, update_adaptive
from analysis.vit_long_1024 import common as C
C.set_determinism()
device, seed, trained = 'cuda', int(sys.argv[1]), sys.argv[2:]   # SL's model_t10.pt, adam_t10.pt, model_t01.pt
source = C.image_source('real')
TINY = torch.finfo(torch.float32).tiny
probe_model = C.build_arm_model('SL', seed, 'cpu', 0.0, 2, 8.0)
HEAD = next(k for k, p in probe_model.named_parameters() if p is probe_model.layers['out'].weight)

def one_update(arm, task, model_state=None, adam_state=None, target=None):
    """run.py's training update in eager mode: Adam's step, update_adaptive, then addendum G's head step."""
    model = C.build_arm_model(arm, seed, device, 0.0, 2, 8.0)
    optimizer = C.build_optimizer(model)
    if model_state is not None:
        model.load_state_dict(model_state)
        optimizer.load_state_dict(copy.deepcopy(adam_state))   # load_state_dict keeps the tensors; step() writes them
    if C.zero_head_at(arm, task, 1):
        reset_head(model)
    hold = C.head_hold(arm, model)
    if hold is not None:
        hold.target = target
    data, _ = C.task_dataset(source, seed, task, 10, 250, augment=False)
    rows = torch.tensor(C.loss_classes(arm, seed, task, 10, 250), device=device)
    x, y = next(iter(train_loader(data, seed, task, C.EVAL_EVERY, 0)))
    x, y = x.to(device), y.to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prepare_noise(model, len(y))
    F.cross_entropy(model(x).index_select(1, rows), y).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True, foreach=True)
    optimizer.step()
    update_adaptive(model)
    if hold is not None:
        hold.after_step()
    out = dict(model.state_dict())
    for i, state in optimizer.state_dict()['state'].items():
        out |= {f'adam.{i}.{k}': v for k, v in state.items()}
    return out

def same(x, y):
    return bool(torch.equal(x, y)) if torch.is_tensor(x) else x == y   # a state_dict also holds extra state

def equal_except(a, b, skip):
    return list(a) == list(b) and all(same(a[k], b[k]) for k in a if k not in skip)

states = {'init': (1, {})}
if trained:
    states['t10'] = (11, dict(model_state=torch.load(trained[0], map_location=device, weights_only=True),
                              adam_state=torch.load(trained[1], map_location=device, weights_only=False)['optimizer']))
    target = torch.load(trained[2], map_location=device, weights_only=True)[HEAD].norm(dim=1, keepdim=True)
factor = 1 - C.LR * C.HEAD_WD
G2 = {}
for name, (task, state) in states.items():
    sl, slw = one_update('SL', task, **state), one_update('SLW', task, **state)
    G2[name] = dict(task=task, factor=factor,
                    head_weight_is_sl_times_factor=bool(torch.equal(slw[HEAD], sl[HEAD].mul(factor))),
                    head_weight_differs_from_sl=not bool(torch.equal(slw[HEAD], sl[HEAD])),
                    rest_equal_sl=equal_except(slw, sl, {HEAD}),
                    head_norm_ratio=float(slw[HEAD].double().norm() / sl[HEAD].double().norm()))
task, state = states['init']
G3 = dict(init_no_target=dict(task=task, equal_sl=equal_except(one_update('SLP', task, **state),
                                                                 one_update('SL', task, **state), set())))
if trained:
    task, state = states['t10']
    sl, slp = one_update('SL', task, **state), one_update('SLP', task, target=target, **state)
    w, w_sl = slp[HEAD], sl[HEAD]
    t = target.double().squeeze(1)
    live = t > 0
    norms, norms_sl = w.double().norm(dim=1), w_sl.double().norm(dim=1)
    cos = (w.double() * w_sl.double()).sum(1)[live] / (norms[live] * norms_sl[live])
    G3['t10_target'] = dict(
        task=task, weight_is_rescaled_sl=bool(torch.equal(w, w_sl.mul(target / w_sl.norm(dim=1, keepdim=True).clamp_min(TINY)))),
        norm_rel_err_max=float(((norms[live] - t[live]).abs() / t[live]).max()),
        sl_norm_rel_off_target_max=float(((norms_sl[live] - t[live]).abs() / t[live]).max()),
        live_rows=int(live.sum()), zero_rows=int((~live).sum()), zero_rows_stay_zero=bool((w[~live] == 0).all()),
        cos_min=float(cos.min()), rest_equal_sl=equal_except(slp, sl, {HEAD}))
print(json.dumps(dict(head_key=HEAD, G2=G2, G3=G3)))
'''


def g4(root, expected_hold):
    common = ('--seed', SEED, '--tasks', 2, '--steps', 25)
    codes = {arm: smoke(arm, PROD_G, root / f'smoke_{arm}', *common) for arm in ARMS}
    resumed = root / 'smoke_SLP_resume'
    codes['SLP_stop'] = smoke('SLP', PROD_G, resumed, *common, '--stop-after-task', 1)
    codes['SLP_resume'] = smoke('SLP', PROD_G, resumed, *common, clean=False)
    result = dict(rc=codes)
    if any(codes.values()):
        return dict(result, **{'pass': False})
    runs, intask, fixed = {}, {}, None
    for arm in ARMS:
        d = root / f'smoke_{arm}'
        rr, prov = F.per_task(d), json.loads((d / 'provenance.json').read_text())
        fixed = fixed or json.loads(rr[0]['classes'])
        intask[arm] = np.load(d / 'intask.npz')
        runs[arm] = dict(done=json.loads((d / 'done.json').read_text())['status'],
                         head_zeroed=[int(r['head_zeroed']) for r in rr],
                         head_full_norm_start=[float(r['head_full_norm_start']) for r in rr],
                         loss_rows=[json.loads(r['loss_rows']) for r in rr],
                         head_hold=prov['config'].get('head_hold'), ln=prov['config']['ln'],
                         normalization=prov['config']['model_config']['normalization'],
                         ms_per_update_last_task=1000 * float(rr[-1]['train_seconds']) / int(rr[-1]['steps']))
    keys = F.INTASK_STEPS + F.INTASK_EVALS
    h = {arm: intask[arm]['head_row_norm'] for arm in ARMS}      # (task, 25-update point, loss row)
    result.update(
        runs=runs, fixed_classes=fixed,
        t00_equal=len(set(child(PROD_G, F.F4_STATES, *(root / f'smoke_{a}/ckpt/model_t00.pt' for a in ARMS)).values())) == 1,
        slp_task1_equals_sl={k: bool(np.array_equal(intask['SLP'][k][0], intask['SL'][k][0])) for k in keys},
        slw_task1_differs_from_sl=not np.array_equal(h['SLW'][0], h['SL'][0]),
        slw_over_sl_head_norm_task1_end=float(np.linalg.norm(h['SLW'][0, -1]) / np.linalg.norm(h['SL'][0, -1])),
        slp_task2_rel_dev=float(np.abs(h['SLP'][1, -1] / h['SLP'][0, -1] - 1).max()),
        sl_task2_rel_dev=float(np.abs(h['SL'][1, -1] / h['SL'][0, -1] - 1).max()))
    straight, again = F.per_task(root / 'smoke_SLP'), F.per_task(resumed)
    shas = child(PROD_G, F.F4_STATES, root / 'smoke_SLP/ckpt/model_t02.pt', resumed / 'ckpt/model_t02.pt')
    again_intask = np.load(resumed / 'intask.npz')
    result['resume'] = dict(
        resumed_after_task_1='"resumed_after_task": 1' in (resumed / 'console.log').read_text(),
        done=json.loads((resumed / 'done.json').read_text())['status'],
        per_task_equal=len(straight) == len(again) == 2 and all(
            {k: v for k, v in a.items() if k not in TIMES} == {k: v for k, v in b.items() if k not in TIMES}
            for a, b in zip(straight, again)),
        intask_equal=all(bool(np.array_equal(intask['SLP'][k], again_intask[k])) for k in intask['SLP'].files),
        model_t02_equal=len(set(shas.values())) == 1)
    result['pass'] = (all(r['done'] == 'complete' and r['ln'] is True and r['normalization'] == 'layer'
                          and r['head_zeroed'] == [1, 0] and r['head_full_norm_start'][0] == 0.0
                          and r['head_full_norm_start'][1] > 0 and all(x == fixed for x in r['loss_rows'])
                          and r['head_hold'] == expected_hold[arm] for arm, r in runs.items())
                      and result['t00_equal'] and all(result['slp_task1_equals_sl'].values())
                      and result['slw_task1_differs_from_sl'] and result['slp_task2_rel_dev'] <= REL_TOL
                      and result['sl_task2_rel_dev'] > 100 * REL_TOL
                      and all(v in (True, 'complete') for v in result['resume'].values()))
    return result


def g5(root):
    out = {}
    points = 200 // 25 + 1
    for arm, production in (('SL', PROD_E), ('SLP', PROD_G)):
        d = root / f'smoke_{arm}_E200'
        rc = smoke(arm, production, d, '--seed', SEED, '--tasks', 1, '--steps', 200)
        if rc:
            out[arm] = {'rc': rc, 'pass': False}
            continue
        t00 = child(production, F.F5_STATES, d / 'ckpt/model_t00.pt', E_SL / 'ckpt/model_t00.pt')[0]
        new, rec = np.load(d / 'intask.npz'), np.load(E_SL / 'intask.npz')
        intask = {k: bool(np.array_equal(new[k][0], rec[k][0, :200])) for k in F.INTASK_STEPS}
        intask |= {k: bool(np.array_equal(new[k][0], rec[k][0, :points])) for k in F.INTASK_EVALS}
        a, b = F.per_task(d)[0], F.per_task(E_SL)[0]
        task_row = {k: a[k] == b[k] for k in F.FIRST_200}
        task_row['online_chunks_1_2'] = json.loads(a['online_chunks']) == json.loads(b['online_chunks'])[:2]
        init_rows = [[r for r in csv.DictReader(open(x / 'layer_stats.csv')) if r['task'] == '0'] for x in (d, E_SL)]
        out[arm] = dict(rc=rc, production=production, init_weights_equal=t00['new'] == t00['recorded'],
                        intask_equal=intask, per_task_equal=task_row,
                        init_layer_stats_equal=init_rows[0] == init_rows[1] and len(init_rows[0]) == 2,
                        last_loss=[float(new['online_loss'][0, -1]), float(rec['online_loss'][0, 199])])
        out[arm]['pass'] = (out[arm]['init_weights_equal'] and all(intask.values()) and all(task_row.values())
                            and out[arm]['init_layer_stats_equal'])
    result = dict(recorded=str(E_SL), runs=out, plan_g_equals_e=child(PROD_G, F.PLANS) == child(PROD_E, F.PLANS))
    result['pass'] = all(r['pass'] for r in out.values()) and result['plan_g_equals_e']
    return result


def g6(root, remote):
    rerun = root / 'f_rerun'
    report = F.main(remote, rerun)
    record = rerun / f'checks_f_{socket.gethostname()}.json'
    record.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return {'verdict': {k: report[k]['pass'] for k in ('F1', 'F2', 'F3', 'F4', 'F5', 'F6')},
            'git_hash': report['git_hash'], 'record': str(record), 'pass': report['pass_']}


if __name__ == '__main__':
    remote = '--remote' in sys.argv
    root = RAW / '_checks_g'
    root.mkdir(parents=True, exist_ok=True)
    git = child(PROD_G, F.GIT)
    report = dict(git_hash=git['head'], remote=remote)
    report['G7'] = dict(git, **{'pass': git['dirty'] == '' and git['production'] == PROD_G})

    g1 = child(PROD_G, G1, RAW, int(not remote))
    g1['pass'] = (all(v['keys_equal'] and v['tensors_equal'] and len(set(v['sha'].values())) == 1
                      and set(v['n_params'].values()) == {v['n_params_expected']}
                      and v['hold'] == {'SL': 'NoneType', 'SLW': 'HeadHold', 'SLP': 'HeadHold'}
                      for v in g1['seeds'].values())
                  and g1['arms'] == dict(shared=[True, True], ln=[True, True], fresh_for=['freshL', 'freshL'])
                  and g1['head_hold']['SL'] is None
                  and (remote or (len(g1['recorded_t00']) > 0 and all(g1['recorded_t00'].values()))))
    report['G1'] = g1

    g23 = child(PROD_G, G23, SEED, *([] if remote else [E_SL / 'ckpt/model_t10.pt', E_SL / 'ckpt/adam_t10.pt',
                                                        E_SL / 'ckpt/model_t01.pt']))
    g2, g3 = g23['G2'], g23['G3']
    report['G2'] = dict(states=g2, head_key=g23['head_key'], **{'pass': (remote or 't10' in g2) and all(
        v['head_weight_is_sl_times_factor'] and v['head_weight_differs_from_sl'] and v['rest_equal_sl']
        for v in g2.values())})
    t10 = g3.get('t10_target')
    report['G3'] = dict(states=g3, tolerance=REL_TOL, **{'pass': g3['init_no_target']['equal_sl'] and (
        remote or (t10 is not None and t10['weight_is_rescaled_sl'] and t10['norm_rel_err_max'] <= REL_TOL
                   and t10['zero_rows'] == 190 and t10['zero_rows_stay_zero'] and t10['cos_min'] >= 1 - REL_TOL
                   and t10['rest_equal_sl'] and t10['sl_norm_rel_off_target_max'] > 100 * REL_TOL))})

    report['G4'] = g4(root, g1['head_hold'])
    report['G5'] = {'pass': 'skipped'} if remote else g5(root)
    report['G6'] = g6(root, remote)
    report['host'] = socket.gethostname()
    verdict = {k: report[k]['pass'] for k in ('G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7')}
    report['pass_'] = all(v in (True, 'skipped') for v in verdict.values())
    (root / f'checks_g_{socket.gethostname()}.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(dict(verdict, pass_=report['pass_'], git_hash=report['git_hash']), ensure_ascii=False))
    sys.exit(0 if report['pass_'] else 1)

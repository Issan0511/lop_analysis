#!/usr/bin/env python3
"""One-shot ReDo x physical live-unit bias shift, preregistered fd2e6682.

Production uses the unmodified resp_cifar5p1_1007.Engine, R=10/CUDA float32,
29-task prefix, and identical t29 streams. CPU is supported only by the separate
synthetic checker. No CUDA or incomplete independent G0 audit means no training.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from src import resp_cifar5p1_1007 as RS
from src.cifar_interventions_0920 import cpu, tree_hash, save_pt
from analysis.cifar_ledger_0920.replay import put, sha, save_npz
from analysis.resp_cifar_ee_0920.stats import t_quantile

C, B, H = RS.C, RS.B, RS.H
RUN = 'redo_position_2x2_1018'
REGISTRATION = 'fd2e6682'
ARMS = ('none', 'redo', 'shift', 'both')
SEEDS = list(range(10))
STEPS, BRANCH, SCORE_TASK = 780, 28, 29
DIAG_STEPS = (1, 10, 100, 390, 780)
EQ_EPS = 0.005
TAU = 0.1
OUT = ROOT / 'results' / RUN
CHECKS = ROOT / 'results' / ('_checks_' + RUN) / 'checks.json'
DEFAULT_AUDIT = Path('/tmp/issa_six_specs_1018/integration_audit.json')
HOST_PATHS = ('src/resp_cifar5p1_1007.py', 'src/cifar5p1_mlp_0920.py',
              'src/baselines_cifar5p1_1008.py', 'src/rlcifar_mlp_battle_0918.py',
              'src/pmnist_0905.py', 'src/pmnist_rlcifar_0907.py',
              'src/pmnist_rlmnist_0906.py', 'src/cifar_interventions_0920.py',
              'analysis/cifar_ledger_0920/ledger.py',
              'analysis/cifar_ledger_0920/replay.py',
              'analysis/resp_cifar_ee_0920/stats.py')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def source_hashes():
    """All loaded repo-owned imports plus the analysis/check entry points and specs."""
    paths = {ROOT / p for p in HOST_PATHS}
    paths.add(Path(__file__).resolve())
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            p = Path(filename).resolve()
            if p.is_relative_to(ROOT) and p.suffix == '.py' and p.is_file():
                paths.add(p)
    paths.update((ROOT / 'analysis' / RUN).glob('*.py'))
    paths.update((ROOT / 'analysis' / RUN).glob('*.sh'))
    paths.update((ROOT / 'specs' / ('spec_' + RUN + '.md'),
                  ROOT / 'specs/spec_six_runs_common_1018.md'))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def safe_environment():
    """Never queries CUDA device properties unless a device is actually available."""
    available = torch.cuda.is_available()
    driver = subprocess.run(['nvidia-smi'], text=True, capture_output=True) if shutil.which('nvidia-smi') else None
    return dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                cuda_build=torch.version.cuda, cuda_available=available,
                nvidia_devices=sorted(str(p) for p in Path('/dev').glob('nvidia*')),
                nvidia_smi_returncode=None if driver is None else driver.returncode,
                nvidia_smi_output=None if driver is None else (driver.stdout + driver.stderr).strip(),
                tf32=torch.backends.cuda.matmul.allow_tf32,
                matmul_precision=torch.get_float32_matmul_precision(),
                deterministic=torch.are_deterministic_algorithms_enabled(),
                threads=torch.get_num_threads(), cublas=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


def audit_errors(path):
    errors = []
    try:
        audit = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        return ['G0_MISSING_OR_UNREADABLE: ' + str(exc)], None
    if audit.get('schema_version') != 1:
        errors.append('G0_SCHEMA')
    for key in ('all_pass', 'main_ancestor_verified', 'host_reproduction_pass', 'mutation_checks_pass'):
        if audit.get(key) is not True:
            errors.append('G0_' + key.upper())
    auditor = audit.get('independent_auditor')
    if not isinstance(auditor, str) or not auditor.strip() or 'redo_position' in auditor:
        errors.append('G0_INDEPENDENT_AUDITOR_IDENTITY')
    commit = audit.get('audited_commit', '')
    try:
        if len(commit) != 40 or subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'],
                                             cwd=ROOT, capture_output=True).returncode:
            errors.append('G0_AUDITED_COMMIT_NOT_ANCESTOR')
    except (TypeError, OSError):
        errors.append('G0_AUDITED_COMMIT_INVALID')
    recorded = audit.get('source_sha256', {})
    for p in HOST_PATHS:
        if recorded.get(p) != sha(ROOT / p):
            errors.append('G0_SOURCE_HASH:' + p)
    return errors, audit


def preflight(out, audit_path=DEFAULT_AUDIT, checks_path=CHECKS):
    """Collect all blockers before failing; this never loads evaluation data."""
    out = Path(out)
    env = safe_environment()
    blockers, audit = audit_errors(audit_path)
    if not env['cuda_available']:
        blockers.insert(0, 'BLOCKED_CUDA: torch.cuda.is_available() is False')
    try:
        checks = json.loads(Path(checks_path).read_text())
        if checks.get('all_pass') is not True or checks.get('source_sha256') != source_hashes():
            blockers.append('CHECKS_NOT_PASSING_CURRENT_SOURCE')
        if checks.get('scope') != 'CPU_SYNTHETIC_ONLY':
            blockers.append('CHECKS_SCOPE_INVALID')
    except (OSError, ValueError):
        blockers.append('CHECKS_MISSING_OR_UNREADABLE')
    dirty = git('status', '--porcelain', '--', 'src', 'analysis', 'specs')
    if dirty:
        blockers.append('IMPLEMENTATION_UNCOMMITTED')
    if subprocess.run(['git', 'merge-base', '--is-ancestor', REGISTRATION, 'HEAD'],
                      cwd=ROOT, capture_output=True).returncode:
        blockers.append('PREREGISTRATION_NOT_ANCESTOR')
    if torch.__version__ != '2.13.0+cu130':
        blockers.append('TORCH_VERSION_CHANGED')
    selected = json.loads((ROOT / 'results/baselines_cifar5p1_1008/calib/selected.json').read_text())
    if selected['selected']['redo'] != dict(method='redo', tau='0.1', period=1560):
        blockers.append('REDO_CALIBRATION_CHANGED')
    report = dict(run_id=RUN, status='BLOCKED_CUDA' if not env['cuda_available'] else
                  ('BLOCKED_G0_OR_CHECKS' if blockers else 'ADMITTED'),
                  all_pass=not blockers, blockers=blockers, checked_utc=now(),
                  registration_commit=REGISTRATION, git_hash=git('rev-parse', 'HEAD'),
                  source_sha256=source_hashes(), environment=env,
                  audit_path=str(audit_path), audit_sha256=sha(audit_path) if Path(audit_path).is_file() else None,
                  checks_path=str(checks_path), checks_sha256=sha(checks_path) if Path(checks_path).is_file() else None,
                  protocol=dict(device='cuda', dtype='float32', R=10, seeds=SEEDS,
                                prefix_tasks=29, branch_task=28, steps_per_task=780,
                                arms=list(ARMS), prediction='POSITION_CARRIES'),
                  production_started=False, argv=sys.argv)
    put(out / 'launch_preflight.json', report)
    if blockers:
        put(out / 'status.json', dict(status=report['status'], blockers=blockers, checked_utc=now(),
                                     production_started=False))
        put(out / 'verdict.json', dict(status=report['status'], label=None, reason=blockers,
                                      observed_seed_count=0, prediction='POSITION_CARRIES'))
    return report


@torch.no_grad()
def redo_masks(P, probe_X, tau=TAU):
    values = B.forward(P, probe_X, B.ReLU(), train=False)
    scores = []
    masks = []
    for h in (values[1], values[3]):
        s = h.abs().mean(1)
        s = s / (s.mean(1, keepdim=True) + 1e-9)
        scores.append(s)
        masks.append(s <= tau)
    return scores, masks


def redo_draws(P, masks, seeds):
    """The first draw from each host b5_redo_reinit stream, l1 then l2."""
    draws = [[], []]
    states = {}
    for r, seed in enumerate(seeds):
        gen = H.stream('b5_redo_reinit', seed)
        for layer in range(2):
            n = int(masks[layer][r].sum())
            fan_in = P[2 * layer].shape[-1]
            values = ((torch.rand((n, fan_in), generator=gen, dtype=torch.float32) * 2 - 1)
                      * (1.0 / math.sqrt(fan_in)))
            draws[layer].append(values)
        states[seed] = gen.get_state()
    return draws, states


@torch.no_grad()
def apply_redo(state, masks, draws):
    """Existing recycler order; bias moments and global tc are intentionally retained."""
    P, m, v = state['P'], state['m'], state['v']
    for layer in range(2):
        for r in range(len(state['seeds'])):
            idx = torch.nonzero(masks[layer][r]).flatten().to(P[0].device)
            if len(idx):
                P[2 * layer][r, idx, :] = draws[layer][r].to(P[0].device)
                P[2 * layer + 1][r, idx] = 0
                m[2 * layer][r, idx, :] = 0
                v[2 * layer][r, idx, :] = 0
    for layer in range(2):
        for r in range(len(state['seeds'])):
            idx = torch.nonzero(masks[layer][r]).flatten().to(P[0].device)
            P[2 * layer + 2][r, :, idx] = 0
            m[2 * layer + 2][r, :, idx] = 0
            v[2 * layer + 2][r, :, idx] = 0
    return state


@torch.no_grad()
def live_and_target(Pnatural, Pearly, X):
    natural = B.forward(Pnatural, X, B.ReLU(), train=False)[2]
    early = B.forward(Pearly, X, B.ReLU(), train=False)[2]
    live = natural.amax(1) > 0
    # clamp(min=0)'s native derivative includes z==0; strict counts are separate.
    target = ((early >= 0) & live[:, None, :]).sum((1, 2)).cpu().numpy().astype(np.int64)
    return live, target, natural, early


def float_bits(values):
    return np.asarray(values, dtype=np.float32).view(np.uint32).astype(np.int64)


def bits_float(values):
    return np.asarray(values, dtype=np.uint32).view(np.float32)


@torch.no_grad()
def apply_shift(state, X, live, target):
    """Globally optimal native pair count over nonnegative finite float32 c.

    Binary search is on ordered positive float32 bit patterns, not a real-valued
    tolerance. Count the actual baddbmm after fl32(b2+c). First find the crossing
    of target, then the first c on the better count plateau. All evaluations keep
    R/slot order and the full continuation-task baddbmm shape unchanged.
    """
    P = state['P']
    b0 = P[3].detach().clone()
    a1 = B.forward(P, X, B.ReLU(), train=False)[1]
    R = len(state['seeds'])
    target = np.asarray(target, dtype=np.int64)
    assert target.shape == (R,) and live.shape == b0.shape
    assert bool(torch.isfinite(a1).all()) and bool(torch.isfinite(b0).all())
    evals = 0

    def counts(bits):
        nonlocal evals
        shifts = torch.from_numpy(bits_float(bits).copy()).to(b0.device)
        bias = torch.where(live, b0 + shifts[:, None], b0)
        z = torch.baddbmm(bias[:, None, :], a1, P[2].transpose(1, 2))
        evals += 1
        return ((z >= 0) & live[:, None, :]).sum((1, 2)).cpu().numpy().astype(np.int64)

    zero = np.zeros(R, dtype=np.int64)
    initial = counts(zero)
    already = initial >= target
    active = ~already
    maximum = int(float_bits([np.finfo(np.float32).max])[0])
    high = np.full(R, maximum, dtype=np.int64)
    high[already] = 0
    max_count = counts(high)
    assert np.all(max_count[active] >= target[active]), 'nonfinite/unreachable shift target'
    low = zero.copy()
    # Invariant: count(low)<target<=count(high) on active slots.
    while np.any(active & (high - low > 1)):
        mid = (low + high) // 2
        cmid = counts(mid)
        right = active & (cmid >= target)
        high = np.where(right, mid, high)
        low = np.where(active & ~right, mid, low)
    upper_count, lower_count = counts(high), counts(low)
    # At an equal error choose the smaller c, hence the lower side.
    choose_lower = abs(lower_count - target) <= abs(upper_count - target)
    chosen = np.where(choose_lower, low, high)
    goal_count = np.where(choose_lower, lower_count, upper_count)
    chosen[already] = 0
    goal_count[already] = initial[already]
    # The lower crossing can lie on the initial count plateau. Its first c is 0.
    chosen[goal_count == initial] = 0
    lo2 = zero.copy()
    hi2 = chosen.copy()
    has_shift = chosen > 0
    # Smallest representable c that realizes the chosen count, including jumps.
    while np.any(has_shift & (hi2 - lo2 > 1)):
        mid = (lo2 + hi2) // 2
        cmid = counts(mid)
        right = has_shift & (cmid >= goal_count)
        hi2 = np.where(right, mid, hi2)
        lo2 = np.where(has_shift & ~right, mid, lo2)
    chosen = np.where(has_shift, hi2, zero)
    achieved = counts(chosen)
    assert np.all(achieved == goal_count)
    c = bits_float(chosen).copy()
    shift = torch.from_numpy(c).to(b0.device)
    P[3].copy_(torch.where(live, b0 + shift[:, None], b0))
    assert bool(torch.isfinite(P[3]).all()), 'physical bias shift overflow'
    reports = []
    for r, seed in enumerate(state['seeds']):
        reports.append(dict(seed=seed, c=float(c[r]), c_float32_bits=int(chosen[r]),
                            live_units=int(live[r].sum()), target_native_count=int(target[r]),
                            initial_native_count=int(initial[r]), achieved_native_count=int(achieved[r]),
                            absolute_target_error=int(abs(achieved[r] - target[r])),
                            lower_crossing_c=float(bits_float([low[r]])[0]),
                            upper_crossing_c=float(bits_float([high[r]])[0]),
                            lower_crossing_count=int(lower_count[r]), upper_crossing_count=int(upper_count[r]),
                            search_evaluations=evals,
                            status='TARGET_ALREADY_MET' if already[r] else
                            ('EXACT_TARGET' if achieved[r] == target[r] else 'UNREACHABLE_TIE_BLOCK'),
                            solver='ordered_float32_bits_native_baddbmm', s_correction=1))
    return reports


def intervene(state, arm, X, live, target, masks, draws):
    """The production arm dispatcher, shared by the synthetic composition checks."""
    assert arm in ARMS
    if arm in ('redo', 'both'):
        apply_redo(state, masks, draws)
    if arm in ('shift', 'both'):
        return apply_shift(state, X, live, target)
    return []


@torch.no_grad()
def diagnostic(eng, ids, update, phase):
    """Unit-first m/s/r and native derivative variance; diagnostics never consume RNG."""
    before = tree_hash(eng.state())
    X, Y = eng.data.X[ids], eng.data.Y[ids]
    values = B.forward(eng.P, X, B.ReLU(), train=False)
    rows, arrays = [], {}
    logits = values[-1]
    for r, seed in enumerate(eng.seeds):
        rows.append(dict(seed=seed, update=update, phase=phase,
                         memo_acc=float((logits[r].argmax(-1) == Y[r]).float().mean()),
                         memo_ce=float(F.cross_entropy(logits[r], Y[r]))))
    for layer in (1, 2):
        z, h = values[2 * (layer - 1):2 * layer]
        zd = z.double()
        g = RS.gtrain(z).double()
        mu = zd.mean(1)
        sd = zd.std(1, correction=1)
        ratio = mu / sd
        scores = h.abs().mean(1)
        scores = scores / (scores.mean(1, keepdim=True) + 1e-9)
        unit = dict(m=mu, s=sd, r=ratio, native_g_mean=g.mean(1),
                    native_g_var=g.var(1, correction=0), strict_open_count=(z > 0).sum(1),
                    native_open_count=g.sum(1), bias=eng.P[2 * layer - 1].detach(),
                    row_norm=eng.P[2 * layer - 2].detach().double().norm(dim=2),
                    dormant_score=scores, s_zero=sd == 0)
        for name, value in unit.items():
            arrays[f'l{layer}_{name}'] = value.cpu().numpy()
        for r, row in enumerate(rows):
            positive_sd = sd[r] > 0
            row.update({f'l{layer}_m_mean': float(mu[r].mean()),
                        f'l{layer}_s_mean': float(sd[r].mean()),
                        f'l{layer}_median_unit_r': float(ratio[r][positive_sd].median()) if bool(positive_sd.any()) else None,
                        f'l{layer}_median_m_over_median_s': float(mu[r].median() / sd[r].median()) if sd[r].median() > 0 else None,
                        f'l{layer}_s_zero': int((~positive_sd).sum()),
                        f'l{layer}_s_zero_m_positive': int(((sd[r] == 0) & (mu[r] > 0)).sum()),
                        f'l{layer}_s_zero_m_negative': int(((sd[r] == 0) & (mu[r] < 0)).sum()),
                        f'l{layer}_s_zero_m_zero': int(((sd[r] == 0) & (mu[r] == 0)).sum()),
                        f'l{layer}_native_open_count': int(g[r].sum()),
                        f'l{layer}_strict_open_count': int((z[r] > 0).sum()),
                        f'l{layer}_dead_strict_fraction': float((z[r].amax(0) <= 0).double().mean()),
                        f'l{layer}_dead_native_fraction': float((g[r].amax(0) == 0).double().mean()),
                        f'l{layer}_dormant_fraction_tau01': float((scores[r] <= TAU).double().mean()),
                        f'l{layer}_native_derivative_var_mean': float(g[r].var(0, correction=0).mean()),
                        f'l{layer}_native_derivative_mean': float(g[r].mean())})
    assert before == tree_hash(eng.state()), 'diagnostic mutated P/Adam/training RNG'
    return rows, arrays, logits.detach()


def interval(values, simultaneous=True):
    x = np.asarray(values, dtype=np.float64)
    assert x.shape == (10,) and np.isfinite(x).all(), 'ten finite paired seeds required'
    mean = float(x[0]) if bool(np.all(x == x[0])) else float(x.mean())
    sd = 0.0 if bool(np.all(x == x[0])) else float(x.std(ddof=1))
    level = 1 - .05 / 7 if simultaneous else .95
    width = t_quantile((1 + level) / 2, 9) * sd / math.sqrt(10)
    return dict(mean=mean, median=float(np.median(x)), sd=sd, low=mean - width,
                high=mean + width, level=level, df=9, m=7 if simultaneous else 1,
                values=x.tolist(), positive=int((x > 0).sum()), negative=int((x < 0).sum()),
                zero=int((x == 0).sum()), flags=['DEGENERATE_SD'] if sd == 0 else [])


def classify_intervals(iv):
    """Literal registered POS/EQ/NI predicates, also exercised at exact boundaries."""
    pos = lambda k: iv[k]['low'] > 0
    eq = lambda k: iv[k]['low'] >= -EQ_EPS and iv[k]['high'] <= EQ_EPS
    labels = []
    if pos('S') and -iv['B_minus_S']['high'] > -EQ_EPS and pos('S_minus_R'):
        labels.append('POSITION_CARRIES')
    if pos('R') and -iv['B_minus_R']['high'] > -EQ_EPS and -iv['S_minus_R']['high'] > 0:
        labels.append('DEAD_CARRIES')
    if pos('R') and pos('S') and eq('I'):
        labels.append('ADDITIVE')
    return labels[0] if len(labels) == 1 else 'MIXED', labels


def verdict(rows):
    """Join on explicit seed/arm keys; no significance is never called equivalence."""
    keys = [(r['arm'], int(r['seed']), int(r['task'])) for r in rows]
    expected = {(arm, seed, 29) for arm in ARMS for seed in SEEDS}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        return dict(status='INCOMPLETE', label=None, reason='registered arm/seed/task keys missing or duplicated')
    if any(not r.get('finite', True) for r in rows):
        return dict(status='DIVERGED', label=None, reason='one or more registered seeds diverged')
    lookup = {(r['arm'], int(r['seed'])): float(r['online_acc']) for r in rows}
    if not all(math.isfinite(v) for v in lookup.values()):
        return dict(status='DIVERGED', label=None, reason='nonfinite online accuracy')
    E = {a: np.array([lookup[a, s] for s in SEEDS], dtype=np.float64) for a in ARMS}
    R, S, V = E['redo'] - E['none'], E['shift'] - E['none'], E['both'] - E['none']
    diffs = dict(R=R, S=S, B=V, B_minus_S=V - S, B_minus_R=V - R,
                 S_minus_R=S - R, I=V - R - S)
    iv = {k: interval(d) for k, d in diffs.items()}
    label, labels = classify_intervals(iv)
    return dict(status='COMPLETE', label=label, candidate_labels=labels,
                mixed_reason=None if len(labels) == 1 else
                ('NO_UNIQUE_PATTERN' if not labels else 'OVERLAPPING_EVIDENCE'),
                simultaneous_intervals=iv, usual_95_intervals={k: interval(d, False) for k, d in diffs.items()},
                arm_means={k: float(v.mean()) for k, v in E.items()}, epsilon_eq=EQ_EPS,
                prediction='POSITION_CARRIES', prediction_source='common spec section 8; Issa instruction formalized by Codex',
                attribution_limit='ReDo tau=0.1 includes live low-activity units; DEAD_CARRIES is an operation label')


def require_identity(saved, current):
    if saved != current:
        raise RuntimeError('resume identity mismatch: source/config/data/seed or audit changed')


def write_summary(out, result):
    text = '# ReDo × position: run 1\n\n'
    text += f"Status: **{result['status']}**. Label: **{result.get('label') or 'not evaluated'}**.\n\n"
    if result['status'] == 'COMPLETE':
        text += '| Arm | t29 online mean |\n|---|---:|\n'
        for arm, value in result['arm_means'].items():
            text += f'| {arm} | {value:.6f} |\n'
        text += '\n| Difference | Mean | Simultaneous interval |\n|---|---:|---|\n'
        for key, item in result['simultaneous_intervals'].items():
            text += f"| {key} | {item['mean']:.6f} | [{item['low']:.6f}, {item['high']:.6f}] |\n"
        text += '\nAll ten seed differences and ordinary intervals are in verdict.json. '
        text += 'The one-shot ReDo protocol does not reproduce the old periodic 43% result.\n'
    else:
        text += 'No evaluation accuracy was produced.\n'
    (Path(out) / 'summary.md').write_text(text)


def analyze_registered(out):
    """Reanalyze only complete, source-bound production artifacts, even offline.

    This deliberately has no raw-row shortcut and needs no currently attached
    GPU. A blocked or synthetic folder cannot acquire a performance verdict by
    placing forty plausible rows in it.
    """
    out = Path(out)
    provenance = json.loads((out / 'provenance.json').read_text())
    ident = provenance['identity']
    assert ident['run_id'] == RUN and ident['source_sha256'] == source_hashes(), 'analysis source identity'
    assert ident['seeds'] == SEEDS and ident['R'] == 10 and ident['steps'] == STEPS
    assert ident['branch_task'] == 28 and ident['score_task'] == 29 and ident['tau'] == TAU
    assert ident['registration_commit'] == REGISTRATION and ident['lr'] == RS.LR
    admission = json.loads((out / 'admission.json').read_text())
    assert admission['all_pass'] is True and admission['status'] == 'ADMITTED', 'no admitted production launch'
    assert admission['source_sha256'] == source_hashes() and admission['environment']['cuda_available'] is True
    assert admission['protocol']['R'] == 10 and admission['protocol']['device'] == 'cuda'
    audit_path = out / 'admission_audit.json'
    assert sha(audit_path) == ident['audit_sha256'] == admission['audit_sha256'], 'admission audit snapshot hash'
    audit_issues, _ = audit_errors(audit_path)
    assert not audit_issues, ('invalid saved independent G0', audit_issues)
    checks_path = out / 'checks.json'
    assert sha(checks_path) == ident['checks_sha256'] == admission['checks_sha256'], 'admission checks snapshot hash'
    checks = json.loads(checks_path.read_text())
    assert checks['all_pass'] is True and checks['source_sha256'] == source_hashes()
    status = json.loads((out / 'status.json').read_text())
    assert status['status'] == 'COMPLETE', 'analysis requires a completed production run'
    manifest = json.loads((out / 'output_manifest.json').read_text())
    require_identity(manifest['identity'], ident)
    required = {'admission.json', 'admission_audit.json', 'checks.json', 'provenance.json',
                'prefix/initial.pt', 'prefix/t01.pt', 'prefix/t28.pt', 'prefix/t29.pt',
                'intervention_plan.pt', 'fresh_control.csv', 'fresh_end.pt', 'per_task.csv'}
    for arm in ARMS:
        required.update(f'arms/{arm}/{name}' for name in
                        ('done.json', 'rows.json', 'intervention.json', 'post_intervention.pt',
                         'end.pt', 'diagnostics.json', 'diagnostic_units.npz', 'per_task.csv'))
    assert required <= set(manifest['files']), 'incomplete production output manifest'
    for name, digest in manifest['files'].items():
        p = Path(name)
        assert not p.is_absolute() and '..' not in p.parts
        assert sha(out / p) == digest, ('production output hash mismatch', name)
    rows = []
    for arm in ARMS:
        folder = out / 'arms' / arm
        marker = json.loads((folder / 'done.json').read_text())
        require_identity(marker['identity'], ident)
        for name, digest in marker['files'].items():
            assert Path(name).name == name and sha(folder / name) == digest, ('arm output hash mismatch', arm, name)
        rows.extend(json.loads((folder / 'rows.json').read_text()))
    natural_info = json.loads((out / 'arms/none/intervention.json').read_text())['prefix_check']
    assert all(natural_info.get(k) is True for k in
               ('all_pass', 'state_bit_exact', 'batches_bit_exact', 'host_rows_exact')), 'none prefix check missing'
    natural_end = RS.load_pt(out / 'arms/none/end.pt')
    prefix_end = RS.load_pt(out / 'prefix/t29.pt')
    assert tree_hash(RS.core_state(natural_end)) == tree_hash(RS.core_state(prefix_end)), 'none prefix checkpoint differs'
    result = verdict(rows)
    assert result['status'] == 'COMPLETE', 'completed manifest has incomplete or diverged primary rows'
    # Deterministic reanalysis is read-only: a differing derivation is an error,
    # never permission to overwrite a registered result and invalidate its hashes.
    assert result == json.loads((out / 'verdict.json').read_text()), 'rederived verdict differs from registered output'
    return result


def record_diagnostics(folder, history):
    rows, arrays = [], {}
    for phase, update, rr, uu in history:
        rows.extend(rr)
        arrays.update({f'{phase}_u{update:04d}_{k}': v for k, v in uu.items()})
    put(folder / 'diagnostics.json', rows)
    save_npz(folder / 'diagnostic_units.npz', arrays)


def train_arm(name, st28, st29, st1, data, plan, folder):
    """One independent clone and one intervention, then exactly 780 native updates."""
    eng = RS.Engine(SEEDS, data, state=st28, graph=True)
    rows_t, batches = eng.task_batches(29)
    ids = torch.stack(rows_t).to(data.device)
    assert tree_hash(batches) == tree_hash(st29['last_batches']), 'branch batch stream mismatch'
    assert tree_hash(data.Y[ids]) == plan['labels_hash']
    history = []
    rr, uu, before_logits = diagnostic(eng, ids, 0, 'pre')
    history.append(('pre', 0, rr, uu))
    st = dict(seeds=eng.seeds, P=eng.P, m=eng.m, v=eng.v, tc=eng.tc,
              task=eng.task, g_batch={s: g.get_state() for s, g in eng.g_batch.items()})
    nonparam_before = tree_hash({k: v for k, v in st.items() if k != 'P'})
    # Snapshot immediately after a reference ReDo for conservation checks on shift.
    shift_reference = cpu(st)
    if name in ('redo', 'both'):
        apply_redo(shift_reference, plan['masks'], plan['draws'])
    shift_info = intervene(st, name, data.X[ids], plan['live'].to(data.device),
                           plan['target'], plan['masks'], plan['draws'])
    if name in ('shift', 'both'):
        moments_before_shift = tree_hash({k: v for k, v in shift_reference.items() if k != 'P'})
        params_before_shift = shift_reference['P']
        assert moments_before_shift == tree_hash({k: v for k, v in st.items() if k != 'P'})
        for i in (0, 1, 2, 4, 5):
            assert tree_hash(eng.P[i]) == tree_hash(params_before_shift[i]), 'shift altered another parameter'
        live = plan['live'].to(data.device)
        c = torch.tensor([r['c'] for r in shift_info], device=data.device)
        expected_b = torch.where(live, params_before_shift[3].to(data.device) + c[:, None],
                                 params_before_shift[3].to(data.device))
        assert tree_hash(expected_b) == tree_hash(eng.P[3]), 'nonuniform or ineligible-unit shift'
    elif name == 'none':
        assert nonparam_before == tree_hash({k: v for k, v in st.items() if k != 'P'})
    rr, uu, after_logits = diagnostic(eng, ids, 0, 'post')
    history.append(('post', 0, rr, uu))
    delta = (after_logits.double() - before_logits.double()).flatten(1)
    info = dict(arm=name, shift=shift_info,
                initial_logits_max_abs_delta=delta.abs().max(1).values.cpu().tolist(),
                initial_logits_l2_delta=delta.norm(dim=1).cpu().tolist(),
                live_hash=tree_hash(plan['live']), masks_hash=tree_hash(plan['masks']),
                replacement_draws_hash=tree_hash(plan['draws']),
                label_hash=plan['labels_hash'], batch_hash=tree_hash(batches),
                pre_hash=tree_hash(st28), post_hash=tree_hash(eng.state()))
    save_pt(folder / 'post_intervention.pt', eng.state())

    def on_diag(update):
        rr, uu, _ = diagnostic(eng, ids, update, 'train')
        history.append(('train', update, rr, uu))

    eng.train_steps(batches, DIAG_STEPS, on_diag)
    eng.task = 29
    finite = eng.finite()
    host = eng.host_rows(29, rows_t)
    rows = [dict(arm=name, seed=seed, task=29, slot=r,
                 online_acc=float(eng.acc_sum[r]) / STEPS, online_ce=float(eng.ce_sum[r]) / STEPS,
                 memo_acc=history[-1][2][r]['memo_acc'], memo_ce=history[-1][2][r]['memo_ce'],
                 finite=bool(finite[r]), label_hash=plan['labels_hash'], batch_hash=tree_hash(batches))
            for r, seed in enumerate(SEEDS)]
    state_end = eng.state()
    if name == 'none':
        assert tree_hash(RS.core_state(state_end)) == tree_hash(RS.core_state(st29)), 'S-prefix end state'
        prefix_rows = {r['seed']: r for r in plan['prefix_rows'] if r['task'] == 29}
        assert all(row == prefix_rows[row['seed']] for row in host), 'S-prefix host rows'
        info['prefix_check'] = dict(all_pass=True, state_bit_exact=True, batches_bit_exact=True,
                                    host_rows_exact=True)
    put(folder / 'intervention.json', info)
    put(folder / 'rows.json', rows)
    RS.write_csv(folder / 'per_task.csv', rows)
    record_diagnostics(folder, history)
    save_pt(folder / 'end.pt', state_end)
    del eng
    gc.collect()
    torch.cuda.empty_cache()
    return rows


@contextlib.contextmanager
def exclusive_cuda():
    """Nonblocking shared lab lock; scheduling does not silently queue a second job."""
    with RS.GPU_LOCK.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        busy = RS.other_gpu_python()
        if busy:
            raise RuntimeError('GPU busy: ' + repr(busy))
        yield


def run(out, admission):
    out = Path(out)
    dev = RS.setup()
    data = RS.Data(dev)
    identity = dict(run_id=RUN, registration_commit=REGISTRATION,
                    source_sha256=source_hashes(), seeds=SEEDS, R=10, steps=STEPS,
                    branch_task=28, score_task=29, tau=TAU, lr=RS.LR,
                    data_sha256=data.sha256, input_hash=tree_hash(data.X), label_hash=tree_hash(data.Y),
                    class_plans={str(s): C.seed_classes(s) for s in SEEDS},
                    audit_sha256=admission['audit_sha256'], checks_sha256=admission['checks_sha256'])
    provenance_path = out / 'provenance.json'
    if provenance_path.exists():
        require_identity(json.loads(provenance_path.read_text())['identity'], identity)
    else:
        put(provenance_path, dict(identity=identity, git_hash=git('rev-parse', 'HEAD'), started_utc=now(),
                                  environment=safe_environment(), independent_audit=admission['audit_path'],
                                  prediction='POSITION_CARRIES', argv=sys.argv,
                                  precision=dict(s_correction=1, derivative_var_correction=0)))
    shutil.copyfile(ROOT / ('specs/spec_' + RUN + '.md'), out / 'spec.md')
    shutil.copyfile(ROOT / 'specs/spec_six_runs_common_1018.md', out / 'spec_common.md')
    shutil.copyfile(CHECKS, out / 'checks.json')
    assert sha(admission['audit_path']) == admission['audit_sha256'], 'G0 changed after admission'
    shutil.copyfile(admission['audit_path'], out / 'admission_audit.json')
    put(out / 'admission.json', admission)
    prefix = out / 'prefix'
    prefix.mkdir(exist_ok=True)
    active = prefix / 'active.pt'
    if active.exists():
        saved = RS.load_pt(active)
        require_identity(saved['identity'], identity)
        eng = RS.Engine(SEEDS, data, state=saved['state'])
        prefix_rows = saved['rows']
    else:
        eng = RS.Engine(SEEDS, data)
        prefix_rows = []
        save_pt(prefix / 'initial.pt', eng.state())
    for t in range(eng.task + 1, 30):
        if (out / 'STOP').exists():
            put(out / 'status.json', dict(status='STOPPED', completed_task=eng.task))
            return
        task_rows, batches = eng.task_batches(t)
        eng.train_steps(batches)
        eng.task = t
        if not bool(eng.finite().all()):
            save_pt(prefix / 'diverged.pt', eng.state())
            raise FloatingPointError('prefix diverged at task ' + str(t))
        prefix_rows.extend(eng.host_rows(t, task_rows))
        prefix_rows.sort(key=lambda r: (r['slot'], r['task']))
        state = eng.state()
        if t in (1, 28, 29):
            save_pt(prefix / f't{t:02d}.pt', dict(**state, last_batches=batches.cpu()))
        save_pt(active, dict(identity=identity, state=state, rows=prefix_rows))
        RS.write_csv(prefix / 'per_task.csv', prefix_rows)
        put(out / 'status.json', dict(status='PREFIX_RUNNING', completed_task=t, updated_utc=now()))
        print(f'prefix t{t:02d}/29 complete', flush=True)
    del eng
    gc.collect()
    torch.cuda.empty_cache()
    st1, st28, st29 = [RS.load_pt(prefix / f't{t:02d}.pt') for t in (1, 28, 29)]
    ids = torch.stack(data.task_rows(SEEDS, 29)).to(dev)
    natural = [p.to(dev) for p in st28['P']]
    early = [p.to(dev) for p in st1['P']]
    live, target, zn, ze = live_and_target(natural, early, data.X[ids])
    probe = st28['last_batches'][:, -2:].reshape(10, 64).to(dev)
    scores, masks = redo_masks(natural, data.X[probe])
    draws, draw_rng_end = redo_draws(natural, masks, SEEDS)
    plan = dict(live=live.cpu(), target=target, masks=cpu(masks), draws=draws,
                labels_hash=tree_hash(data.Y[ids]), prefix_rows=prefix_rows)
    save_pt(out / 'intervention_plan.pt', dict(**plan, source_state_hash=tree_hash(st28),
                                              target_early_native_counts=target, score=cpu(scores),
                                              replacement_rng_end=draw_rng_end,
                                              early_strict_counts=((ze > 0) & live[:, None, :]).sum((1, 2)).cpu(),
                                              natural_strict_counts=((zn > 0) & live[:, None, :]).sum((1, 2)).cpu()))
    put(out / 'mask_summary.json', [dict(seed=s, dormant_l1=int(masks[0][r].sum()),
                                       dormant_l2=int(masks[1][r].sum()), live_l2=int(live[r].sum()),
                                       overlap_D2_L=int((masks[1][r] & live[r]).sum()),
                                       target_native_count=int(target[r]),
                                       zero_only_units=int((zn[r].amax(0) == 0).sum()))
                                   for r, s in enumerate(SEEDS)])
    del natural, early, zn, ze
    rows = []
    for arm in ARMS:
        if (out / 'STOP').exists():
            put(out / 'status.json', dict(status='STOPPED', next_arm=arm))
            return
        folder = out / 'arms' / arm
        folder.mkdir(parents=True, exist_ok=True)
        marker = folder / 'done.json'
        if marker.exists():
            done = json.loads(marker.read_text())
            require_identity(done['identity'], identity)
            assert all(sha(folder / p) == digest for p, digest in done['files'].items()), 'corrupt arm checkpoint'
            arm_rows = json.loads((folder / 'rows.json').read_text())
        else:
            arm_rows = train_arm(arm, st28, st29, st1, data, plan, folder)
            if not all(r['finite'] for r in arm_rows):
                raise FloatingPointError('DIVERGED arm ' + arm)
            put(marker, dict(identity=identity, files={p.name: sha(p) for p in folder.iterdir() if p != marker}))
        rows.extend(arm_rows)
        RS.write_csv(out / 'per_task.csv', rows)
        put(out / 'status.json', dict(status='ARMS_RUNNING', completed_arm=arm, updated_utc=now()))
        print(f'arm {arm} complete', flush=True)
    # Fresh uses the exact saved t29 batches and initial params, with tc=m=v=0.
    fresh_engine = RS.Engine(SEEDS, data, graph=True)
    fresh_engine.train_steps(st29['last_batches'].to(dev))
    fresh_engine.task = 29
    if not bool(fresh_engine.finite().all()):
        save_pt(out / 'fresh_diverged.pt', fresh_engine.state())
        raise FloatingPointError('fresh reference diverged')
    fresh_rows = [dict(seed=s, task=29, online_acc=float(fresh_engine.acc_sum[r]) / STEPS,
                       online_ce=float(fresh_engine.ce_sum[r]) / STEPS,
                       finite=bool(fresh_engine.finite()[r]), batch_hash=tree_hash(st29['last_batches']))
                  for r, s in enumerate(SEEDS)]
    RS.write_csv(out / 'fresh_control.csv', fresh_rows)
    save_pt(out / 'fresh_end.pt', fresh_engine.state())
    result = verdict(rows)
    put(out / 'verdict.json', result)
    write_summary(out, result)
    put(out / 'status.json', dict(status=result['status'], label=result['label'], finished_utc=now()))
    put(out / 'output_manifest.json', dict(files={str(p.relative_to(out)): sha(p)
                                                 for p in sorted(out.rglob('*'))
                                                 if p.is_file() and p.name != 'output_manifest.json'},
                                           finished_utc=now(), identity=identity))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preflight', 'run', 'analyze'))
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--integration-audit', type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == 'analyze':
        result = analyze_registered(args.out)
        print(json.dumps(dict(status=result['status'], label=result['label'])))
        return
    report = preflight(args.out, args.integration_audit)
    print(json.dumps(dict(status=report['status'], blockers=report['blockers']), ensure_ascii=False), flush=True)
    if not report['all_pass']:
        raise SystemExit(2)
    if args.command == 'run':
        try:
            with exclusive_cuda():
                run(args.out, report)
        except FloatingPointError as exc:
            result = dict(status='DIVERGED', label=None, reason=str(exc))
            put(args.out / 'status.json', result)
            put(args.out / 'verdict.json', result)
            write_summary(args.out, result)
            raise
        except Exception as exc:
            result = dict(status='CHECK_FAILED', label=None, reason=repr(exc), updated_utc=now())
            put(args.out / 'status.json', result)
            put(args.out / 'verdict.json', result)
            write_summary(args.out, result)
            raise


if __name__ == '__main__':
    main()

"""Registered seed-paired readout; scientific labels require admitted artifacts.

Pure classify_rows() is deliberately separate for synthetic implementation tests.
The CLI never accepts a naked CSV as a registered production result.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from analysis.redo_chb_rlcifar_1018 import common as C
from analysis.resp_cifar_ee_0920.stats import t_quantile

EPS_EQ = .005


def interval(values, m=7):
    x = np.asarray(values, dtype=np.float64)
    assert x.shape == (10,) and np.isfinite(x).all(), 'requires ten finite paired seeds'
    constant = bool(np.all(x == x[0]))
    mean = float(x[0]) if constant else float(x.mean())
    sd = 0. if constant else float(x.std(ddof=1))
    level = 1 - .05 / m
    width = t_quantile(1 - .05 / (2 * m), 9) * sd / math.sqrt(10)
    return dict(mean=mean, median=float(np.median(x)), sd=sd, low=mean-width, high=mean+width,
                df=9, n=10, m=m, level=level, values=x.tolist(),
                positive=int((x > 0).sum()), negative=int((x < 0).sum()), ties=int((x == 0).sum()),
                flags=['DEGENERATE_SD'] if constant else [])


def label_intervals(iv):
    pos = lambda k: iv[k]['low'] > 0
    eq = lambda k: iv[k]['low'] >= -EPS_EQ and iv[k]['high'] <= EPS_EQ
    candidates = []
    if pos('C') and -iv['B_minus_C']['high'] > -EPS_EQ and pos('C_minus_R'):
        candidates.append('CHB_POLICY_CARRIES')
    if pos('R') and -iv['B_minus_R']['high'] > -EPS_EQ and -iv['C_minus_R']['high'] > 0:
        candidates.append('REDO_POLICY_CARRIES')
    return dict(label=candidates[0] if len(candidates) == 1 else 'MIXED',
                candidates=candidates, additive=pos('R') and pos('C') and eq('I'),
                redo_shortfall=iv['C_minus_R']['low'] > EPS_EQ,
                mixed_reason=None if len(candidates) == 1 else
                ('NO_UNIQUE_PATTERN' if not candidates else 'OVERLAPPING_EVIDENCE'))


def classify_rows(rows):
    expected = {(a, s, t) for a in C.ARMS for s in C.SEEDS for t in range(1, 201)}
    lookup, problems, diverged = {}, [], False
    for row in rows:
        try:
            key = (row['arm'], int(row['seed']), int(row['task']))
            if key in lookup:
                problems.append('duplicate:' + repr(key))
            if row.get('cond', 'raw') != 'raw':
                problems.append('condition:' + repr(key))
            value = float(row['online_acc'])
            if not math.isfinite(value):
                diverged = True; problems.append('nonfinite:' + repr(key))
            if str(row.get('diverged', False)).lower() == 'true':
                diverged = True
            if math.isfinite(value) and not 0 <= value <= 1:
                problems.append('accuracy_out_of_range:' + repr(key))
            lookup[key] = value
        except (KeyError, TypeError, ValueError):
            problems.append('invalid_row')
    if set(lookup) != expected:
        problems.append('missing_or_extra_registered_cells')
    if diverged:
        return dict(status='DIVERGED', label=None, problems=problems)
    if problems:
        return dict(status='INCOMPLETE', label=None, problems=problems)
    late = {a: np.asarray([np.mean([lookup[a,s,t] for t in range(181,201)], dtype=np.float64)
                           for s in C.SEEDS]) for a in C.ARMS}
    early = {a: [float(np.mean([lookup[a,s,t] for t in range(31,51)], dtype=np.float64))
                 for s in C.SEEDS] for a in C.ARMS}
    R, V, B = late['redo']-late['none'], late['CHB']-late['none'], late['both']-late['none']
    differences = dict(R=R, C=V, B=B, B_minus_C=B-V, B_minus_R=B-R,
                       C_minus_R=V-R, I=B-R-V)
    iv = {k: interval(v) for k,v in differences.items()}
    return dict(status='COMPLETE', **label_intervals(iv), seeds=C.SEEDS,
                late_tasks=list(range(181,201)), early_tasks=list(range(31,51)),
                late_per_seed={a:v.tolist() for a,v in late.items()}, early_per_seed=early,
                late_arm_means={a:float(v.mean()) for a,v in late.items()},
                simultaneous_intervals=iv, ordinary_95_intervals={k:interval(v,1) for k,v in differences.items()},
                epsilon_eq=EPS_EQ, prediction=C.PREDICTION,
                interpretation='CHB package versus transferred periodic ReDo policy; not isolated bias or drift mediation')


def verify_stream_pairing(records):
    """Seed-indexed cohort/label/order hashes must agree for all four fresh runs."""
    assert set(records) == set(C.ARMS), 'missing arm stream records'
    reference = records['none']
    assert set(reference) == {str(t) for t in range(1,201)}, 'missing task stream audit'
    for arm in C.ARMS:
        assert set(records[arm]) == set(reference), ('missing stream task', arm)
        for task, expected in reference.items():
            actual = records[arm][task]
            for key in ('label_hash', 'order_chain', 'epoch_count', 'g_lab_hash', 'g_batch_hash'):
                assert actual[key] == expected[key], ('stream mismatch', arm, task, key)
            assert actual['epoch_count'] == 400, 'wrong epoch count'


def status_evidence(out):
    out = Path(out)
    records = []
    for p in [out/'status.json', *[out/'arms'/a/'status.json' for a in C.ARMS]]:
        if p.exists():
            try:
                value = json.loads(p.read_text())
                if value.get('status') != 'COMPLETE':
                    records.append(dict(path=str(p.relative_to(out)), **value))
            except ValueError:
                records.append(dict(path=str(p.relative_to(out)), status='CHECK_FAILED', reason='invalid status JSON'))
    if any(r['status'] == 'DIVERGED' for r in records):
        return dict(status='DIVERGED', label=None, evidence=records)
    if any(r['status'] == 'CHECK_FAILED' for r in records):
        return dict(status='CHECK_FAILED', label=None, evidence=records)
    if records:
        # BLOCKED_*, STOPPED and RUNNING are execution states, never performance labels.
        return dict(status=records[0]['status'], label=None, evidence=records)
    return None


def verify_completed(out):
    """Full saved admission/identity/manifest checks, without present CUDA access."""
    out = Path(out)
    provenance = json.loads((out/'provenance.json').read_text())
    identity = provenance['identity']
    assert identity['registration'] == C.registration()
    assert identity['protocol'] == C.PROTOCOL and identity['source_sha256'] == C.source_hashes()
    saved_admission = json.loads((out/'admission.json').read_text())
    assert saved_admission['all_pass'] is True and saved_admission['status'] == 'ADMITTED'
    assert saved_admission['source_sha256'] == C.source_hashes()
    assert saved_admission['protocol'] == C.PROTOCOL
    assert identity['data']['archive'] == saved_admission['archive'], 'saved admitted archive differs from cohort identity'
    archive = saved_admission['archive']
    assert identity['data']['data_sha256'][archive['archive']] == archive['sha256'], 'loaded cohort archive differs'
    for name, scope in [('cpu_checks.json','CPU_SYNTHETIC_ONLY'), ('gpu_release.json','CUDA_REAL_DATA_RELEASE')]:
        proof = json.loads((out/'admission'/name).read_text())
        C.verify_hash_record(proof, scope=scope)
        assert C.sha(out/'admission'/name) == saved_admission['proof_sha256'][name]
        if name == 'gpu_release.json':
            assert proof['archive_sha256'] == archive['sha256'], 'GPU release differs from admitted data'
    review = json.loads((out/'admission/independent_review.json').read_text())
    assert review['approved'] is True and review['source_sha256'] == C.source_hashes()
    assert review.get('reviewer') and review.get('scope') == 'CPU_SOURCE_REVIEW'
    assert C.sha(out/'admission/independent_review.json') == saved_admission['proof_sha256']['independent_review.json']
    aggregate, streams = [], {}
    for arm in C.ARMS:
        folder = out/'arms'/arm
        done = json.loads((folder/'done.json').read_text())
        assert done['identity'] == identity and done['arm'] == arm and done['status'] == 'COMPLETE'
        required = {'per_task.csv','ckpt.pt','checkpoint_digest.json','provenance.json','stream_audit.json','identity.json'}
        assert required <= set(done['files']), ('incomplete arm artifact set', arm)
        C.verify_manifest(folder, done['files'])
        from src.redo_chb_rlcifar_1018 import verify_checkpoint_file
        verify_checkpoint_file(folder/'ckpt.pt')
        state = C.load_pt(folder/'ckpt.pt')
        assert state['t'] == 200 and state['tc'] == 6000000
        assert state['alive'].tolist() == [True]*10 and not state['diverged']
        assert state['meta']['arm'] == arm and state['meta']['seeds'] == C.SEEDS
        extension = state['extension_state']
        assert extension['event_count'] == (C.EXPECTED_EVENTS if arm in ('redo','both') else 0)
        assert extension['last_tc'] == 6000000
        streams[arm] = json.loads((folder/'stream_audit.json').read_text())
        aggregate.extend(C.read_csv(folder/'per_task.csv'))
    verify_stream_pairing(streams)
    return classify_rows(aggregate), identity


def summary(result):
    text = '# RC/raw 200-task ReDo × CHB\n\n'
    text += f"Execution status: **{result['status']}**. "
    text += f"Primary label: **{result.get('label') or 'not evaluated'}**.\n\n"
    text += 'Registered prediction: CHB_POLICY_CARRIES and REDO_SHORTFALL.\n\n'
    if result['status'] == 'COMPLETE':
        text += f"ADDITIVE: {result['additive']}. REDO_SHORTFALL: {result['redo_shortfall']}.\n\n"
        text += '| Arm | Mean online, t181–200 |\n|---|---:|\n'
        for arm, value in result['late_arm_means'].items(): text += f'| {arm} | {value:.7f} |\n'
        text += '\n| Difference | Mean | Simultaneous interval |\n|---|---:|---|\n'
        for key, value in result['simultaneous_intervals'].items():
            text += f"| {key} | {value['mean']:.7f} | [{value['low']:.7f}, {value['high']:.7f}] |\n"
        text += '\nAll ten paired differences, ordinary intervals and sign counts are in verdict.json.\n'
    else:
        text += 'No scientific performance label is assigned to incomplete, blocked, stopped or failed execution.\n'
    text += '\nThis compares the CHB package with a ReDo policy transferred from F51 without RC tuning. '
    text += 'It does not isolate a bias-only effect. The prior F51 experiment was a completed preregistered trial: '
    text += 'its POSITION_CARRIES forecast was not supported (MIXED). That trial and its failed prediction remain preserved.\n'
    return text


def report(out, *, verify_only=False):
    out = Path(out)
    C.require_run_namespace(out)
    execution = status_evidence(out)
    if execution is not None:
        result = dict(run_id=C.RUN, **execution, prediction=C.PREDICTION)
    else:
        try:
            result, identity = verify_completed(out)
            result.update(run_id=C.RUN, identity_sha256=C.digest(identity))
        except (OSError, ValueError, KeyError, AssertionError) as exc:
            result = dict(run_id=C.RUN, status='CHECK_FAILED', label=None, reason=str(exc), prediction=C.PREDICTION)
    if verify_only:
        assert result['status'] == 'COMPLETE', ('registered analysis refused', result)
        old = json.loads((out/'verdict.json').read_text())
        assert result == old, 'rederived verdict differs; refusal to overwrite'
        return result
    C.put(out/'verdict.json', result)
    (out/'summary.md').write_text(summary(result))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=C.OUT)
    ap.add_argument('--verify-only', action='store_true')
    args = ap.parse_args()
    result = report(args.out, verify_only=args.verify_only)
    print(json.dumps(dict(status=result['status'], label=result['label'])))
    return 0 if result['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""CPU synthetic checks and executable mutation negatives, never evaluation data.

The existing baseline's nested recycler is extracted as an AST definition and
executed unchanged against the synthetic state. This tests compatibility with
the actual inherited implementation, without a second hand-written recycler.
"""
from __future__ import annotations
import argparse
import ast
import copy
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src import redo_position_2x2_1018 as R


def fixture():
    X0 = torch.tensor([[-1., 0., 1.], [0., 0., 0.], [1., 0., -1.], [2., 0., 0.]])
    X = X0[None].repeat(2, 1, 1)
    W1 = torch.eye(3)[None].repeat(2, 1, 1)
    b1 = torch.tensor([[1., 0., 0.], [1., 0., 0.]])
    W2 = torch.zeros(2, 3, 3)
    W2[:, 0, 0] = 1
    W2[:, 1, 0] = .01
    b2 = torch.tensor([[.1, -.015, 0.], [.1, -.015, 0.]])
    W3 = torch.arange(18, dtype=torch.float32).view(2, 3, 3) / 20
    b3 = torch.tensor([[.2, -.3, .1], [.1, .2, -.2]])
    P = [W1, b1, W2, b2, W3, b3]
    state = dict(seeds=[100, 101], P=P,
                 m=[torch.full_like(p, .37) for p in P],
                 v=[torch.full_like(p, .19) for p in P], tc=21840, task=28,
                 g_batch={s: R.H.stream('c51_batch', s).get_state() for s in (100, 101)})
    early = R.cpu(P)
    early[3][:, 1] = .05
    return state, early, X


def host_recycler_reference(state, X):
    """Execute the old file's real nested functions with explicit captured locals."""
    source = (ROOT / 'src/baselines_cifar5p1_1008.py').read_text()
    tree = ast.parse(source)
    run_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    nodes = [n for n in run_node.body if isinstance(n, ast.FunctionDef) and n.name in ('fresh_row', 'redo_apply')]
    assert {n.name for n in nodes} == {'fresh_row', 'redo_apply'}
    P = state['P']
    # Existing recycler takes global rows into X_all; synthetic slots receive disjoint rows.
    Rslots, n, d = X.shape
    ids = torch.arange(Rslots * n).reshape(Rslots, n)
    env = dict(torch=torch, B=R.B, P=P, X_all=X.reshape(-1, d), act=R.B.ReLU(),
               REDO_SCORE_EPS=1e-9, redo_tau=.1, device=torch.device('cpu'),
               seeds=state['seeds'], R=Rslots, adam_m=state['m'], adam_v=state['v'],
               fan_in=[p.shape[-1] for p in P],
               bound=[1 / math.sqrt(p.shape[-1]) for p in P],
               g_reinit={s: R.H.stream('b5_redo_reinit', s) for s in state['seeds']},
               n_reset=torch.zeros(Rslots, 2, dtype=torch.long), redo_rows=[])
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'inherited_baseline_recycler', 'exec'), env)
    env['redo_apply'](ids, state['tc'], 28, 779, 'train')
    return state, env


def expect_raises(fn):
    try:
        fn()
    except (AssertionError, RuntimeError, ValueError, KeyError):
        return True
    return False


def compare_redo(candidate, expected):
    assert R.tree_hash(candidate) == R.tree_hash(expected), 'ReDo differs from host recycler'


def verify_live(live, natural_P, X):
    independent_z = R.B.forward(natural_P, X, R.B.ReLU(), train=False)[2]
    assert torch.equal(live, (independent_z > 0).any(1)), 'live must be natural pre-intervention max(z)>0'


def verify_shift(before, after, X, live, reports):
    """State conservation and independent gate/autograd/adjacent-float checks."""
    for key in before:
        if key != 'P':
            assert R.tree_hash(before[key]) == R.tree_hash(after[key]), key
    for i in (0, 1, 2, 4, 5):
        assert R.tree_hash(before['P'][i]) == R.tree_hash(after['P'][i]), i
    c = torch.tensor([r['c'] for r in reports], dtype=torch.float32)
    expected = torch.where(live, before['P'][3] + c[:, None], before['P'][3])
    assert torch.equal(expected, after['P'][3]), 'one c on exactly natural L'
    independent_P = [p.detach().clone().requires_grad_(True) for p in after['P']]
    vals = R.B.forward(independent_P, X, R.B.ReLU(), train=True)
    native_counts = torch.autograd.grad(vals[3].sum(), independent_P[3])[0]
    for r, item in enumerate(reports):
        count = int(native_counts[r, live[r]].sum())
        assert count == item['achieved_native_count'], 'native autograd gate count'
        if item['status'] != 'TARGET_ALREADY_MET':
            lower_error = abs(item['lower_crossing_count'] - item['target_native_count'])
            upper_error = abs(item['upper_crossing_count'] - item['target_native_count'])
            assert item['absolute_target_error'] == min(lower_error, upper_error)
        if item['c_float32_bits']:
            previous = R.bits_float([item['c_float32_bits'] - 1])[0]
            q = R.cpu(before['P'])
            q[3][r] = torch.where(live[r], q[3][r] + float(previous), q[3][r])
            z = R.B.forward(q, X, R.B.ReLU(), train=False)[2]
            previous_count = int(((z[r] >= 0) & live[r][None]).sum())
            assert previous_count != count, 'selected c is not the smallest c on its plateau'


def synthetic_rows(redo=.0, shift=.1, both=.1, varying=False):
    rows = []
    for arm, effect in [('none', 0), ('redo', redo), ('shift', shift), ('both', both)]:
        for seed in R.SEEDS:
            base = .12 + .011 * seed
            extra = (seed - 4.5) * .001 if varying and arm == 'shift' else 0
            rows.append(dict(arm=arm, seed=seed, task=29, online_acc=base + effect + extra, finite=True))
    return rows


def main(out):
    torch.set_num_threads(2)
    tests = {}
    mutations = {}
    state, early, X = fixture()
    before = R.cpu(state)
    live, target, natural_z, _ = R.live_and_target(state['P'], early, X)
    assert torch.equal(live, torch.tensor([[True, True, False], [True, True, False]]))
    assert target.tolist() == [8, 8]
    verify_live(live, state['P'], X)
    scores, masks = R.redo_masks(state['P'], X)
    assert bool((masks[1] & live).any()), 'fixture needs D2/L overlap'
    draws, rng_end = R.redo_draws(state['P'], masks, state['seeds'])
    actual = R.apply_redo(R.cpu(state), masks, draws)
    reference, env = host_recycler_reference(R.cpu(state), X)
    compare_redo(actual, reference)
    assert R.tree_hash(rng_end) == R.tree_hash({s: g.get_state() for s, g in env['g_reinit'].items()})
    assert R.tree_hash(before) == R.tree_hash(state), 'input plan mutated'
    tests['S-redo'] = dict(pass_=True, existing_nested_recycler_directly_executed=True,
                           original_source_sha256=R.sha(ROOT / 'src/baselines_cifar5p1_1008.py'),
                           exact_parameter_moment_rng_match=True,
                           reset_counts=[m.sum(1).tolist() for m in masks])
    # Every required ReDo negative must cause the real comparison to reject it.
    mutant = R.cpu(actual)
    for layer in range(2):
        for r in range(2):
            mutant['m'][2 * layer + 1][r, masks[layer][r]] = 0
            mutant['v'][2 * layer + 1][r, masks[layer][r]] = 0
    mutations['redo_zero_bias_moments'] = expect_raises(lambda: compare_redo(mutant, reference))
    mutant = R.cpu(actual)
    for layer in range(2):
        for r in range(2):
            mutant['P'][2 * layer + 2][r, :, masks[layer][r]] = state['P'][2 * layer + 2][r, :, masks[layer][r]]
    mutations['redo_keep_output_columns'] = expect_raises(lambda: compare_redo(mutant, reference))
    # An exact float32 score at tau; the +1e-9 denominator rounds to one here.
    h = torch.tensor([[[.1, 1.9]]], dtype=torch.float32)
    boundary_score = h.abs().mean(1)
    boundary_score /= boundary_score.mean(1, keepdim=True) + 1e-9
    assert boundary_score[0, 0] == torch.tensor(.1)
    boundary_P = [torch.zeros(1, 2, 1), torch.tensor([[.1, 1.9]]), torch.eye(2)[None],
                  torch.zeros(1, 2), torch.zeros(1, 2, 2), torch.zeros(1, 2)]
    bscore, bmask = R.redo_masks(boundary_P, torch.zeros(1, 1, 1))
    assert bool(bmask[0][0, 0]) and bscore[0][0, 0] == torch.tensor(.1)
    mutations['redo_tau_strict_boundary'] = not torch.equal(bmask[0], bscore[0] < .1)
    post_live = R.B.forward(actual['P'], X, R.B.ReLU(), train=False)[2].amax(1) > 0
    mutations['live_recomputed_after_redo'] = expect_raises(lambda: verify_live(post_live, state['P'], X))
    native_live = (natural_z >= 0).any(1)
    mutations['live_native_zero_gate'] = expect_raises(lambda: verify_live(native_live, state['P'], X))
    tests['S-live'] = dict(pass_=True, natural_L=live.tolist(), post_redo_L=post_live.tolist(),
                           native_zero_includes_ineligible=True, overlap_D2_L=(masks[1] & live).sum(1).tolist())

    shift = R.cpu(state)
    shift_info = R.apply_shift(shift, X, live, target)
    verify_shift(state, shift, X, live, shift_info)
    assert all(item['absolute_target_error'] == 0 for item in shift_info)
    mutant = R.cpu(shift)
    mutant['P'][3][~live] += .015
    mutations['shift_all_b2'] = expect_raises(lambda: verify_shift(state, mutant, X, live, shift_info))
    mutant = R.cpu(shift)
    mutant['P'][3][:, 0] += .03
    mutations['shift_unit_specific_c'] = expect_raises(lambda: verify_shift(state, mutant, X, live, shift_info))
    # Reapplying a physical shift every update violates the one-time state transition.
    mutant = R.cpu(shift)
    for r, info in enumerate(shift_info):
        mutant['P'][3][r, live[r]] += info['c']
    mutations['shift_reapplied_each_update'] = expect_raises(lambda: verify_shift(state, mutant, X, live, shift_info))
    # Independent tiny exact fixture with an unreachable two-pair tie block.
    tied = R.cpu(state)
    tied['P'][2].zero_()
    tied['P'][3][:] = torch.tensor([[-1., -1., -3.], [-1., -1., -3.]])
    tied_live = torch.tensor([[True, True, False], [True, True, False]])
    tied_X = X[:, :1]
    tied_before = R.cpu(tied)
    tie_info = R.apply_shift(tied, tied_X, tied_live, np.array([1, 1]))
    verify_shift(tied_before, tied, tied_X, tied_live, tie_info)
    assert all(q['c'] == 0 and q['absolute_target_error'] == 1 for q in tie_info), 'equal errors prefer c=0'
    already = R.cpu(state)
    already_before = R.tree_hash(already)
    met = R.apply_shift(already, X, live, np.array([0, 0]))
    assert R.tree_hash(already) == already_before and all(q['status'] == 'TARGET_ALREADY_MET' for q in met)
    # Independent oracle: W2=0, one image, biases -1/-2/-2. All attainable
    # counts and their smallest float32 c are known: (0,0), (1,1), (3,2).
    oracle = R.cpu(state)
    oracle['P'][2].zero_()
    oracle['P'][3][:] = torch.tensor([[-1., -2., -2.], [-1., -2., -2.]])
    oracle_results = []
    for K in range(4):
        candidate = R.cpu(oracle)
        info = R.apply_shift(candidate, X[:, :1], torch.ones(2, 3, dtype=torch.bool), np.array([K, K]))
        expected_count, expected_c = min([(0, 0.), (1, 1.), (3, 2.)], key=lambda item: (abs(item[0] - K), item[1]))
        assert all(q['achieved_native_count'] == expected_count and q['c'] == expected_c for q in info)
        oracle_results.append(dict(target=K, expected_count=expected_count, expected_c=expected_c))
    tests['S-shift'] = dict(pass_=True, exact_float32_optimum=shift_info, unreachable_tie=tie_info,
                            already_met_noop=True, independent_exact_oracle=oracle_results,
                            tolerance='byte-exact for states; exact integer gate counts')

    both = R.cpu(state)
    both_info = R.intervene(both, 'both', X, live, target, masks, draws)
    independent = R.cpu(reference)
    R.apply_shift(independent, X, live, target)
    assert R.tree_hash(independent) == R.tree_hash(both)
    none = R.cpu(state)
    assert R.intervene(none, 'none', X, live, target, masks, draws) == []
    assert R.tree_hash(none) == R.tree_hash(state), 'production none dispatcher is not byte-exact'
    reverse = R.cpu(shift)
    R.apply_redo(reverse, masks, draws)
    mutations['composition_reversed'] = R.tree_hash(reverse) != R.tree_hash(both)
    advanced_draws = [[], []]
    for r, seed in enumerate(state['seeds']):
        gen = R.H.stream('b5_redo_reinit', seed)
        torch.rand((1,), generator=gen)
        for layer in range(2):
            n, fi = draws[layer][r].shape
            advanced_draws[layer].append((torch.rand((n, fi), generator=gen) * 2 - 1) / math.sqrt(fi))
    advanced = R.apply_redo(R.cpu(state), masks, advanced_draws)
    R.apply_shift(advanced, X, live, target)
    mutations['composition_different_random_draws'] = R.tree_hash(advanced) != R.tree_hash(both)
    tests['S-composition'] = dict(pass_=True, independent_host_redo_then_shift_byte_exact=True,
                                 same_masks_and_replacement_draws=True, both_shift=both_info)

    # Real host Engine eager path on synthetic full-shape data: no CIFAR labels read.
    class SyntheticData:
        def __init__(self):
            self.device = torch.device('cpu')
            gen = torch.Generator().manual_seed(1101)
            self.X = torch.randn(128, 3072, generator=gen)
            self.Y = torch.arange(128) % 100

    def train_cpu(engine, batches):
        engine.acc_sum.zero_()
        engine.ce_sum.zero_()
        for j in range(batches.shape[1]):
            engine.static_idx.copy_(batches[:, j])
            engine.tc += 1
            engine.inv_c1.fill_(1 / (1 - .9 ** engine.tc))
            engine.inv_c2.fill_(1 / (1 - .999 ** engine.tc))
            engine.step()
        return engine.state(), engine.acc_sum.clone(), engine.ce_sum.clone()

    data = SyntheticData()
    eng = R.RS.Engine([100, 101], data, graph=False)
    batches = torch.stack([torch.arange(32)[None].repeat(3, 1), (torch.arange(32) + 32)[None].repeat(3, 1)])
    train_cpu(eng, batches[:, :1])
    branch = eng.state()
    uninterrupted, hit, ce = train_cpu(eng, batches[:, 1:])
    resumed = R.RS.Engine([100, 101], data, state=branch, graph=False)
    resumed_state, hit2, ce2 = train_cpu(resumed, batches[:, 1:])
    assert R.tree_hash(uninterrupted) == R.tree_hash(resumed_state)
    assert torch.equal(hit, hit2) and torch.equal(ce, ce2)
    reset = R.cpu(branch)
    reset['tc'] = 0
    altered = R.RS.Engine([100, 101], data, state=reset, graph=False)
    altered_state, _, _ = train_cpu(altered, batches[:, 1:])
    mutations['prefix_adam_clock_reset'] = R.tree_hash(altered_state) != R.tree_hash(uninterrupted)
    altered = R.RS.Engine([100, 101], data, state=branch, graph=False)
    altered_state, _, _ = train_cpu(altered, (batches[:, 1:] + 1) % 128)
    mutations['prefix_batch_shift_one'] = R.tree_hash(altered_state) != R.tree_hash(uninterrupted)
    ids = torch.arange(32)[None].repeat(2, 1)
    bstate = R.tree_hash(resumed.state())
    rr, uu, _ = R.diagnostic(resumed, ids, 2, 'synthetic')
    assert R.tree_hash(resumed.state()) == bstate
    assert 'l2_native_g_var' in uu and len(rr) == 2
    tests['S-prefix'] = dict(pass_=True, host_engine_real_eager_steps=3, full_parameter_shapes=True,
                             seeds=[100, 101], resumed_state_online_ce_bit_exact=True,
                             diagnostic_state_rng_unchanged=True,
                             limit='CPU synthetic only; CUDA graph and R=10 host equivalence still require independent G0')

    label_cases = dict(POSITION_CARRIES=synthetic_rows(0, .1, .1),
                      DEAD_CARRIES=synthetic_rows(.1, 0, .1),
                      ADDITIVE=synthetic_rows(.03, .07, .10),
                      MIXED=synthetic_rows(0, 0, 0))
    for label, rows in label_cases.items():
        assert R.verdict(rows)['label'] == label, (label, R.verdict(rows))
    # Position row order must be irrelevant. Bad positional joins change the known expected vector.
    rows = synthetic_rows(.01, .09, .09, varying=True)
    result = R.verdict(rows)
    expected = result['simultaneous_intervals']['S']['values']
    shuffled = rows[::2] + rows[1::2]
    assert R.verdict(shuffled)['simultaneous_intervals']['S']['values'] == expected
    shift_vals = [r['online_acc'] for r in rows if r['arm'] == 'shift'][::-1]
    none_vals = [r['online_acc'] for r in rows if r['arm'] == 'none']
    wrong = np.array(shift_vals) - np.array(none_vals)
    mutations['readout_positional_seed_join'] = not np.array_equal(wrong, expected)
    assert R.verdict(rows[:-1])['status'] == 'INCOMPLETE'
    assert R.verdict(rows + [rows[0]])['status'] == 'INCOMPLETE'
    bad = copy.deepcopy(rows)
    bad[0]['online_acc'] = float('nan')
    assert R.verdict(bad)['status'] == 'DIVERGED'
    wide = R.interval(np.array([-.04, .04] * 5))
    assert wide['low'] < -R.EQ_EPS and wide['high'] > R.EQ_EPS
    proper_eq = wide['low'] >= -R.EQ_EPS and wide['high'] <= R.EQ_EPS
    wrong_eq = wide['low'] <= 0 <= wide['high']
    mutations['readout_nonsignificant_means_eq'] = wrong_eq and not proper_eq
    wide_case = copy.deepcopy(R.verdict(label_cases['ADDITIVE'])['simultaneous_intervals'])
    wide_case['I'] = wide
    assert R.classify_intervals(wide_case)[0] == 'MIXED', 'production classifier mistook nonsignificance for EQ'
    boundary = copy.deepcopy(R.verdict(label_cases['POSITION_CARRIES'])['simultaneous_intervals'])
    boundary['S']['low'] = 0
    assert R.classify_intervals(boundary)[0] == 'MIXED', 'real POS excludes zero boundary'
    boundary = copy.deepcopy(R.verdict(label_cases['POSITION_CARRIES'])['simultaneous_intervals'])
    boundary['B_minus_S']['high'] = .005
    assert R.classify_intervals(boundary)[0] == 'MIXED', 'real NI excludes its boundary'
    boundary = copy.deepcopy(R.verdict(label_cases['ADDITIVE'])['simultaneous_intervals'])
    boundary['I']['low'], boundary['I']['high'] = -.005, .005
    assert R.classify_intervals(boundary)[0] == 'ADDITIVE', 'real EQ includes both boundaries'
    boundary['I']['high'] = np.nextafter(.005, float('inf'))
    assert R.classify_intervals(boundary)[0] == 'MIXED', 'EQ rejects just outside boundary'
    overlap = R.verdict(synthetic_rows(.001, .1, .101))
    assert overlap['label'] == 'MIXED' and overlap['mixed_reason'] == 'OVERLAPPING_EVIDENCE'
    tests['S-readout'] = dict(pass_=True, label_cases=list(label_cases), correct_seed_join=True,
                              missing_duplicate_nonfinite_rejected=True,
                              zero_pos_strict_and_equivalence_inclusive=True,
                              wide_nonsignificant_is_not_equivalent=True,
                              simultaneous_intervals=7, df=9, seed_count=10)
    with tempfile.TemporaryDirectory() as temp:
        errors, _ = R.audit_errors(Path(temp) / 'missing.json')
        assert errors and errors[0].startswith('G0_MISSING')
        admission = R.preflight(Path(temp) / 'launch', Path(temp) / 'missing.json')
        assert not admission['all_pass'] and admission['production_started'] is False
        assert any(e.startswith('G0_MISSING') for e in admission['blockers'])
        assert admission['protocol']['device'] == 'cuda' and admission['protocol']['R'] == 10
        assert admission['protocol']['seeds'] == list(range(10))
        assert admission['protocol']['steps_per_task'] == 780
        if not torch.cuda.is_available():
            assert admission['status'] == 'BLOCKED_CUDA'
            assert any(e.startswith('BLOCKED_CUDA') for e in admission['blockers'])
        assert not (Path(temp) / 'launch' / 'per_task.csv').exists()
        tests['S-admission'] = dict(pass_=True, missing_G0_rejected=True,
                                    actual_launch_preflight_executed=True,
                                    CUDA_and_registered_protocol_are_not_overridden=True,
                                    production_started=False, evaluation_rows_absent=True)
    report = dict(run_id=R.RUN, scope='CPU_SYNTHETIC_ONLY', evaluated_cifar_seed_count=0,
                  generated_utc=R.now(), environment=R.safe_environment(),
                  source_sha256=R.source_hashes(), checks=tests, mutation_negatives=mutations,
                  tolerances=dict(state='byte exact; no empirical tolerance widening',
                                  gates='exact integer count',
                                  intervals='float64 registered Student t routine'),
                  all_pass=all(t['pass_'] for t in tests.values()) and all(mutations.values()),
                  unperformed=['CUDA graph warmup rollback check on this new launch',
                               'R=10 CUDA prefix/none equality on real CIFAR',
                               'G0 independent CUDA host reproduction',
                               'all four 10-seed production arms'])
    R.put(Path(out) / 'checks.json', report)
    assert report['all_pass'], report
    print(json.dumps(dict(all_pass=report['all_pass'], scope=report['scope'],
                          check_groups=len(tests), detected_mutations=len(mutations),
                          output=str(Path(out) / 'checks.json'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=R.CHECKS.parent)
    main(parser.parse_args().out)

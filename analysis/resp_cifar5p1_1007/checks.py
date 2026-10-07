#!/usr/bin/env python3
"""resp_cifar5p1_1007 admission checks (spec §7).  Check seeds 100-109 and synthetic inputs only.

Every required check must PASS on the real implementation and every listed mutant must be
REJECTED by the same comparator.  Fail closed; every attempt is kept under
results/_checks_resp_cifar5p1_1007/attempt_<ns>/.
"""
from __future__ import annotations

import contextlib
import gc
import inspect
import json
import math
import resource
import shutil
import sys
import textwrap
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import src.resp_cifar5p1_1007 as M
from src.resp_cifar5p1_1007 import *          # noqa: F401,F403  (the module under test)
from analysis.resp_cifar5p1_1007.report import verdict, report
from analysis.resp_cifar_ee_0920.stats import interval, label, t_quantile

OUT = ROOT / f'results/_checks_{RUN}'
SEEDS = list(range(100, 110))
REQUIRED = ['S-host', 'S-prefix', 'S-branch', 'S-field', 'S-grad', 'S-reset', 'S-isolation',
            'S-graph', 'S-verdict', 'S-resume', 'S-cost/CLI']
MUTANTS = ['lr', 'time', 'order',                                        # S-host
           'mv_lost', 'tc_reset', 'rng_not_restored',                    # S-prefix
           'drop_anchor', 'alias_P0', 'wrong_layer',                     # S-branch
           'zero_field', 'sign_field', 'seed_field', 'unit_field', 'image_field',
           'batch_position', 'nan_outside_lost',                         # S-field
           'drop_backward_shift', 'frozen_grad', 'backward_only', 'dphi_as_train',   # S-grad
           'reset_m_only', 'reset_keep_tc', 'epsilon_omitted',           # S-reset
           'shallow_copy', 'diagnostic_rng', 'shared_rng',               # S-isolation
           'warmup_state',                                               # S-graph
           'unpaired', 'one_sided', 'no_bonferroni', 'missing_seed', 'nonfinite', 'missing_arm',
           'denominator',                                                # S-verdict
           'different_identity', 'fake_done', 'bad_hash', 'field_lost', 'mv_lost_resume',  # S-resume
           'wrong_arm', 'wrong_budget', 'missing_provenance']            # S-cost/CLI


def reject(fn):
    """The comparator must refuse the mutant (an assertion-type failure), never accept it."""
    try:
        fn()
    except (AssertionError, ValueError, KeyError, SystemExit, FileNotFoundError, RuntimeError):
        return True
    raise AssertionError('mutant was accepted')


def same(a, b):
    assert tree_hash(a) == tree_hash(b), 'not bit identical'


def replace_method(cls, method, old, new):
    src = textwrap.dedent(inspect.getsource(getattr(cls, method)))
    assert src.count(old) == 1, (method, old, src.count(old))
    ns = {}
    exec(src.replace(old, new), getattr(cls, method).__globals__, ns)
    return ns[method]


@contextlib.contextmanager
def patched(obj, attr, value):
    original = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, original)


def free():
    gc.collect()
    torch.cuda.empty_cache()


def fingerprint(res):
    """Everything an arm produces that the readout uses, as one hash."""
    hist = [(u, rr, {k: v for k, v in sorted(uu.items())}) for u, rr, uu in res['history']]
    return tree_hash(dict(rows=res['rows'], state=core_state(res['end_state']), hist=hist,
                          field=res['field']))


# --------------------------------------------------------------------------
# S-grad: native clamp derivative and an independent float64 chain rule
# --------------------------------------------------------------------------

def derivative_check(mut):
    tiny = float(np.nextafter(np.float32(0), np.float32(1)))
    z = torch.tensor([-3.0e38, -1.0, -tiny, -0.0, 0.0, tiny, 1.0, 3.0e38], device='cuda')
    native = gtrain(z)
    expected = torch.tensor([0., 0., 0., 1., 1., 1., 1., 1.], device='cuda')
    same(native, expected)
    diag = B.ReLU().dphi(z)
    reject(lambda: same(diag, native))          # the host's eval gate 1[z>0] is not the training derivative
    mut['dphi_as_train'] = True
    return dict(z=z.cpu().tolist(), native_training_derivative=native.cpu().tolist(),
                host_diagnostic_dphi=diag.cpu().tolist(),
                subnormal_preserved_cuda=bool((torch.tensor(tiny, device='cuda') * 1).item() != 0))


def gradient_check(mut):
    gen = torch.Generator().manual_seed(271828)
    dims = (7, 5, 4, 3)
    r, b = 2, 6
    P = []
    for i in range(3):
        P.append(torch.randn((r, dims[i + 1], dims[i]), generator=gen, dtype=torch.float64) * .6)
        P.append(torch.randn((r, dims[i + 1]), generator=gen, dtype=torch.float64) * .4)
    P = [q.cuda().requires_grad_() for q in P]
    X = torch.randn((r, b, dims[0]), generator=gen, dtype=torch.float64).cuda()
    ids = torch.arange(b, device='cuda')[None].expand(r, -1)
    y = (torch.arange(r * b, device='cuda') % 3).view(r, b)
    eps64 = np.finfo(float).eps / 2
    gamma = 128 * eps64 / (1 - 128 * eps64)
    ratios = []
    for layer in (1, 2):
        d = (torch.randn((r, b, dims[layer]), generator=gen, dtype=torch.float64) * 1.5).cuda()
        field = Field(P, layer, d)
        with torch.no_grad():
            P[0].add_(.13)
            P[3].sub_(.17)
        vals = field(P, X, ids)
        z1, a1, z2, a2, z3 = vals
        loss = F.cross_entropy(z3.flatten(0, 1), y.flatten(), reduction='none').view(r, b).mean(1).sum()
        got = torch.autograd.grad(loss, P, retain_graph=True)
        prob = (z3 - z3.max(-1, keepdim=True).values).exp()
        prob = prob / prob.sum(-1, keepdim=True)
        dz3 = (prob - F.one_hot(y, 3)) / b
        g2 = ((z2 + (d if layer == 2 else 0)) >= 0).double()
        dz2 = (dz3 @ P[4]) * g2
        g1 = ((z1 + (d if layer == 1 else 0)) >= 0).double()
        dz1 = (dz2 @ P[2]) * g1
        expected = [dz1.transpose(1, 2) @ X, dz1.sum(1), dz2.transpose(1, 2) @ a1, dz2.sum(1),
                    dz3.transpose(1, 2) @ a2, dz3.sum(1)]
        scale = sum(float(q.detach().abs().sum()) for q in [*P, *vals, X, d, dz1, dz2, dz3])
        bound = gamma * scale
        for a, e in zip(got, expected):
            err = float((a - e).detach().abs().max())
            assert err <= bound, (layer, err, bound)
            ratios.append(err / bound)
        # the derivative of every gate must be non-trivial in this fixture
        assert 0 < float(g2.mean()) < 1 and 0 < float(g1.mean()) < 1
        mutants = {
            'drop_backward_shift': (f'phi(z{layer} + d)', f'phi(z{layer})'),
            'frozen_grad': ('with torch.no_grad():', 'with torch.enable_grad():'),
            'backward_only': (f'a{layer} = AnchorAdd.apply(a0{layer}, phi(z{layer} + d) - phi(z0{layer} + d))',
                              f'a{layer} = phi(z{layer}) + (phi(z{layer} + d) - phi(z{layer} + d).detach())')}
        for name, (old, new) in mutants.items():
            fn = replace_method(Field, '__call__', old, new)
            with patched(Field, '__call__', fn):
                if name == 'frozen_grad':
                    bad = Field(P, layer, d)
                    bad.P0 = P                       # frozen forward now shares the trained P
                else:
                    bad = field                      # same frozen P0 as the reference, mutated code only
                bz = bad(P, X, ids)[-1]
                bl = F.cross_entropy(bz.flatten(0, 1), y.flatten(), reduction='none').view(r, b).mean(1).sum()
                bg = torch.autograd.grad(bl, P)
            worst = max(float((a - e).detach().abs().max()) for a, e in zip(bg, expected))
            assert worst > bound, (name, layer, worst, bound)
            mut[name] = True
    return dict(max_error_ratio=max(ratios), independent='hand float64 CE chain rule, gate 1[z+d>=0]',
                bound='gamma_128 * sum|tensors|', layers=[1, 2])


# --------------------------------------------------------------------------
# S-verdict
# --------------------------------------------------------------------------

def statistics_check(mut):
    for p in (.975, .9875):
        q = t_quantile(p, 9)
        xx = np.linspace(0, q, 20001)
        density = math.gamma(5) / (math.sqrt(9 * math.pi) * math.gamma(4.5)) * (1 + xx * xx / 9) ** -5
        area = (xx[1] - xx[0]) / 3 * (density[0] + density[-1] + 4 * density[1:-1:2].sum() + 2 * density[2:-1:2].sum())
        assert abs(.5 + area - p) < 2e-12
    g = {'h': [.5] * 10, 'c': [.1] * 10}
    noise = np.linspace(-.01, .01, 10)
    base = {n: np.full(10, .5) + noise for n in ARMS}
    base['N_c'] = np.full(10, .3) + noise
    base['N_cr'] = np.full(10, .3) + noise
    for a, b_, want in [(1, 1, 'RESPONSE_BOTH_WAYS'), (1, 0, 'RESTORE_ONLY'), (0, 1, 'SINK_ONLY'),
                        (0, 0, 'RESPONSE_NOT_SHOWN'), (-1, 1, 'RESPONSE_REVERSED'), (1, -1, 'RESPONSE_REVERSED')]:
        v = {k: x.copy() for k, x in base.items()}
        v['R_ch'] = v['N_c'] + a * .1
        v['S_hcr'] = v['N_hr'] - b_ * .1
        assert verdict(v, g)['label'] == want, want
    z = interval(np.zeros(10))
    assert z['sign'] == '0' and z['degenerate_sd'] and z['low'] == z['high'] == 0
    assert label({'sign': '0'}, {'sign': '0'}) == 'RESPONSE_NOT_SHOWN'
    v = {k: x.copy() for k, x in base.items()}
    v['N_h'] = v['N_c'].copy()
    out = verdict(v, g)
    assert out['label'] == 'NOT_REPRODUCED' and out['rho1'] is None
    gbad = {'h': [.5] * 9 + [.05], 'c': [.1] * 10}
    assert verdict(base, gbad)['label'] == 'NOT_REPRODUCED'
    v = {k: x.copy() for k, x in base.items()}
    v['N_cr'] = v['N_hr'].copy()
    assert verdict(v, g)['rho2'] is None
    mut['denominator'] = True
    for name, edit in [('missing_seed', lambda v: v.__setitem__('N_h', v['N_h'][:-1])),
                       ('nonfinite', lambda v: v['N_h'].__setitem__(0, np.nan)),
                       ('missing_arm', lambda v: v.pop('N_c'))]:
        v = {k: x.copy() for k, x in base.items()}
        edit(v)
        reject(lambda: verdict(v, g))
        mut[name] = True
    # a nonzero-variance fixture whose correct 97.5% interval straddles 0
    spread = np.linspace(-.3, .3, 10)
    v = {k: x.copy() for k, x in base.items()}
    v['R_ch'] = base['N_c'] + spread + .151
    assert verdict(v, g)['P1']['sign'] == '0'
    for name, level in [('one_sided', .90), ('no_bonferroni', .95)]:
        assert interval(v['R_ch'] - v['N_c'], level)['sign'] == '+'
        mut[name] = True
    pair = np.arange(10) / 10
    assert interval((pair + .01) - pair)['sign'] == '+' and interval((pair + .01) - pair[::-1])['sign'] == '0'
    mut['unpaired'] = True
    return dict(labels=7, quantiles='independent Simpson integration', all_ten_required=True)


# --------------------------------------------------------------------------
# the suite
# --------------------------------------------------------------------------

def suite():
    OUT.mkdir(parents=True, exist_ok=True)
    attempt = OUT / f'attempt_{time.time_ns()}'
    attempt.mkdir()
    evidence, mut = {}, {}
    started = time.time()
    tested_source = source_hashes()

    def record(name, value):
        evidence[name] = dict(all_pass=True, **value)
        put(attempt / 'progress.json', dict(evidence=evidence, mutants=mut))
        print(f'{name} PASS ({time.time() - started:.0f}s)', flush=True)

    try:
        dev = setup()
        evidence_grad = dict(derivative=derivative_check(mut), chain_rule=gradient_check(mut))
        record('S-grad', evidence_grad)
        record('S-verdict', statistics_check(mut))
        data = Data(dev)

        # ---------------- S-host (a): unmodified host in this environment ----------------
        hostdir = attempt / 'host'
        with (attempt / 'host.log').open('w') as log, contextlib.redirect_stdout(log):
            C.run('R', SEEDS, 'std', 30, dev, hostdir, lr=LR, cifar=data.cifar)
        free()
        eng = Engine(SEEDS, data)
        states, saved_batches = {}, {}

        def keep(t, st, batches, rows):
            if t in SAVE_TASKS:
                states[t] = dict(**st, last_batches=batches)
        rows30, _ = natural_prefix(eng, 30, save=keep)
        fv, ff = eng.fresh(states[29]['last_batches'].to(dev))
        assert bool(ff.all())
        fresh = fresh_rows(SEEDS, fv, rows30, 29)
        del eng
        free()
        a, b_ = csv_text(rows30), (hostdir / 'per_task.csv').read_text()
        assert a == b_, 'per_task.csv differs from the unmodified host'
        assert csv_text(fresh) == (hostdir / 'fresh_control.csv').read_text(), 'fresh_control differs'
        prefix_rows = [r for r in rows30 if r['task'] <= 29]
        # mutants: one natural task from t_h must reproduce t03 bit for bit; each mutant must not
        ref3 = core_state(states[3])

        def one_task(eng):
            _, batches = eng.task_batches(TH + 1)
            eng.train_steps(batches)
            eng.task = TH + 1
            return core_state(eng.state())
        e = Engine(SEEDS, data, state=states[TH])
        same(one_task(e), ref3)
        del e
        for name, cls_method, old, new in [
                ('lr', 'step', 'p.sub_(LR *', 'p.sub_(2 * LR *'),
                ('time', 'train_steps', 'self.tc += 1', 'self.tc += 100'),
                ('order', 'train_steps', 'self.static_idx.copy_(batches[:, j])',
                 'self.static_idx.copy_(batches[:, batches.shape[1] - 1 - j])')]:
            fn = replace_method(Engine, cls_method, old, new)
            with patched(Engine, cls_method, fn):
                e = Engine(SEEDS, data, state=states[TH])
                reject(lambda: same(one_task(e), ref3))
            mut[name] = True
            del e
            free()
        record('S-host', dict(per_task_byte_identical=True, fresh_byte_identical=True, tasks=30, seeds=SEEDS,
                              columns='all (eff_rank included)', reference='unmodified cifar5p1_mlp_0920.run in this environment'))

        # ---------------- S-prefix ----------------
        prefix_ev = {}
        for name in ('N_h', 'N_c'):
            res = run_arm(name, states, data, preflight_on=False)
            prefix_ev[name] = prefix_check(name, res, states, prefix_rows)
            free()
        for mname, edit in [('mv_lost', lambda st: [q.zero_() for q in st['m']]),
                            ('tc_reset', lambda st: st.__setitem__('tc', 0)),
                            ('rng_not_restored', lambda st: st.__setitem__(
                                'g_batch', {s: H.stream('c51_batch', s).get_state() for s in SEEDS}))]:
            bad = {t: cpu(s) for t, s in states.items()}
            edit(bad[TH])
            res = run_arm('N_h', bad, data, preflight_on=False)
            reject(lambda: prefix_check('N_h', res, states, prefix_rows))
            mut[mname] = True
            del res
            free()
        record('S-prefix', dict(arms=prefix_ev))

        # ---------------- CLI run (all arms, preflight on the real states) ----------------
        cli = attempt / 'cli'
        argv = sys.argv
        sys.argv = [str(ROOT / f'src/{RUN}.py'), '--check-mode', '--out', str(cli), '--seeds', '100-109']
        t_cli = time.time()
        try:
            with (attempt / 'cli.log').open('w') as log, contextlib.redirect_stdout(log), \
                    patched(M, 'gpu_lock', lambda **k: contextlib.nullcontext()):
                M.main()
        finally:
            sys.argv = argv
        cli_seconds = time.time() - t_cli
        assert json.loads((cli / 'status.json').read_text())['stage'] == 'completed'
        assert {p.name for p in (cli / 'arms').iterdir()} == set(ARMS)
        cli_states = {t: load_pt(cli / 'prefix' / f't{t:02d}.pt') for t in SAVE_TASKS}
        for t in SAVE_TASKS:
            same(core_state(cli_states[t]), core_state(states[t]))
            same(cli_states[t]['last_batches'], states[t]['last_batches'])
        pf = {n: json.loads((cli / 'arms' / n / 'preflight.json').read_text()) for n in ARMS}
        assert all(p['all_pass'] and p['batches'] == 1 + STEPS + 79 for p in pf.values())
        free()

        # ---------------- S-branch / S-field mutants through the production validator ----------------
        def built(name='R_ch'):
            e = arm_engine(name, states, data, graph=False)
            _, batches = e.task_batches(ARMS[name][0] + 1)
            return e, batches
        e, batches = built()
        rows_T = data.task_rows(SEEDS, TC + 1)
        ids_T = torch.stack(rows_T).to(dev)
        d_local = e.field.d[e.ar, ids_T].clone()
        del e
        free()

        def table_edit(fn):
            def edit(e):
                fn(e)
            return edit
        field_mutants = {
            'zero_field': lambda e: e.field.d.zero_(),
            'sign_field': lambda e: e.field.d.neg_(),
            'seed_field': lambda e: setattr(e.field, 'd', e.field.d.roll(1, 0)),
            'unit_field': lambda e: setattr(e.field, 'd', e.field.d.roll(1, 2)),
            'image_field': lambda e: e.field.d.__setitem__((e.ar, ids_T), d_local.roll(1, 1)),
            'nan_outside_lost': lambda e: e.field.d.nan_to_num_(nan=0.0),
            'wrong_layer': lambda e: setattr(e.field, 'layer', 1),
            'alias_P0': lambda e: setattr(e.field, 'P0', e.P)}
        for name, edit in field_mutants.items():
            e, batches = built()
            edit(e)
            reject(lambda: preflight('R_ch', e, states, batches))
            mut[name] = True
            del e
            free()
        for name, old, new in [
                ('drop_anchor', 'a2 = AnchorAdd.apply(a02, phi(z2 + d) - phi(z02 + d))', 'a2 = phi(z2 + d)'),
                ('batch_position', 'd = self.d[ar, ids]',
                 'd = self.d[ar, torch.arange(ids.shape[1], device=ids.device)[None, :].expand_as(ids)]')]:
            fn = replace_method(Field, '__call__', old, new)
            with patched(Field, '__call__', fn):
                e, batches = built()
                reject(lambda: preflight('R_ch', e, states, batches))
            mut[name] = True
            del e
            free()
        # A lookup fault that keeps finite values must still be refused: shift the lookup to the
        # neighbouring task image (finite d of another image), through the training-gate probe.
        perm = torch.full((len(SEEDS), N_TRAIN), -1, dtype=torch.long, device=dev)
        perm[torch.arange(len(SEEDS), device=dev)[:, None], ids_T] = ids_T.roll(1, 1)
        fn = replace_method(Field, '__call__', 'd = self.d[ar, ids]', 'd = self.d[ar, PERM_[ar, ids]]')
        fn.__globals__['PERM_'] = perm
        with patched(Field, '__call__', fn):
            e, batches = built()
            reject(lambda: preflight('R_ch', e, states, batches))
        del e
        fn.__globals__.pop('PERM_')
        free()
        record('S-branch', dict(arms={n: dict(batches=p['batches'], branch_bit_exact=p['branch_bit_exact']) for n, p in pf.items()},
                                tested='all task images at once, all 780 training batches, row-order chunks of 32'))
        record('S-field', dict(arms={n: {k: v for k, v in p.items() if k not in ('branch_bit_exact',)} for n, p in pf.items() if 'argument_error_ratio' in p},
                               finite_wrong_image_lookup_refused=True))

        # ---------------- S-isolation: reversed arm order ----------------
        names = list(ARMS)
        for br in (TH, TC):                     # arms of one branch share the task's batches and labels
            hashes = {(json.loads((cli / 'arms' / n / 'rows.json').read_text())[0]['batch_hash'],
                       json.loads((cli / 'arms' / n / 'rows.json').read_text())[0]['label_hash'])
                      for n in names if ARMS[n][0] == br}
            assert len(hashes) == 1, ('branch batches differ across arms', br)
        before = tree_hash(states)
        for name in reversed(names):
            res = run_arm(name, states, data)
            rows_cli = json.loads((cli / 'arms' / name / 'rows.json').read_text())
            assert tree_hash(jnorm(res['rows'])) == tree_hash(rows_cli), ('reversed order rows', name)
            same(core_state(res['end_state']), core_state(load_pt(cli / 'arms' / name / 'end.pt')))
            del res
            free()
        same(before, tree_hash(states))
        # mutants
        keep_w = states[TH]['P'][0].clone()
        bad = dict(states)                                            # shallow copy
        bad[TH]['P'][0].add_(1.0)
        reject(lambda: same(before, tree_hash(states)))
        states[TH]['P'][0].copy_(keep_w)
        same(before, tree_hash(states))
        mut['shallow_copy'] = True
        fn = replace_method(Engine, 'diagnostic', 'R, N = ids.shape',
                            'R, N = ids.shape; [torch.randperm(8, generator=g) for g in self.g_batch.values()]')
        with patched(Engine, 'diagnostic', fn):
            reject(lambda: run_arm('N_h', states, data, preflight_on=False))
        mut['diagnostic_rng'] = True
        free()
        shared = {}
        fn = replace_method(Engine, 'task_batches', 'rows = self.data.task_rows(self.seeds, t)',
                            'rows = self.data.task_rows(self.seeds, t); self.g_batch = SHARED_.setdefault(t, self.g_batch)')
        fn.__globals__['SHARED_'] = shared
        with patched(Engine, 'task_batches', fn):
            r1 = run_arm('N_h', states, data, preflight_on=False)
            r2 = run_arm('N_hr', states, data, preflight_on=False)
        fn.__globals__.pop('SHARED_')
        reject(lambda: same(r2['batches'], r1['batches']))
        mut['shared_rng'] = True
        del r1, r2
        free()
        record('S-isolation', dict(reversed_order_bit_identical=True, states_unchanged=True, arms=len(names)))

        # ---------------- S-graph ----------------
        for name in ('N_h', 'R_ch', 'S_hcL1r'):
            ra = run_arm(name, states, data, graph=False, preflight_on=False)
            rb = run_arm(name, states, data, graph=True, preflight_on=False)
            assert fingerprint(ra) == fingerprint(rb), ('eager/graph', name)
            del ra, rb
            free()
        fn = replace_method(Engine, 'capture', 'q.copy_(old)', 'pass')
        with patched(Engine, 'capture', fn):
            rb = run_arm('N_h', states, data, graph=True, preflight_on=False)
        ra = run_arm('N_h', states, data, graph=False, preflight_on=False)
        reject(lambda: same(core_state(ra['end_state']), core_state(rb['end_state'])))
        mut['warmup_state'] = True
        del ra, rb
        free()
        record('S-graph', dict(arms=['N_h', 'R_ch', 'S_hcL1r'], eager_graph_bit_identical=True, warmup_rolled_back=True))

        # ---------------- S-reset ----------------
        e = arm_engine('R_chr', states, data, graph=False)
        assert e.tc == 0 and all(int(torch.count_nonzero(q)) == 0 for q in e.m + e.v)
        _, batches = e.task_batches(TC + 1)
        ids = batches[:, 0]
        xb = data.X[ids]
        logits = e.forward(xb, ids)[-1]
        yy = data.Y[ids]
        loss = F.cross_entropy(logits.flatten(0, 1), yy.flatten(), reduction='none').view(len(SEEDS), BATCH).mean(1).sum()
        grads = torch.autograd.grad(loss, e.P)
        old = [q.detach().clone() for q in e.P]
        expected, maxratio, zero_g = [], 0.0, 0
        u = 2.0 ** -24
        for p, gr in zip(old, grads):
            m = torch.zeros_like(gr).mul_(B1).add_(gr, alpha=1 - B1)
            v = torch.zeros_like(gr).mul_(B2).addcmul_(gr, gr, value=1 - B2)
            delta = LR * (m * torch.tensor(1 / (1 - B1), device=dev)) / ((v * torch.tensor(1 / (1 - B2), device=dev)).sqrt() + EPS)
            expected.append(p - delta)
            target = LR * gr.double() / (gr.double().abs() + EPS)
            bound = (32 * u / (1 - 32 * u)) * (target.abs() + LR) + 32 * 2.0 ** -149
            assert bool(((delta.double() - target).abs() <= bound).all())
            maxratio = max(maxratio, float(((delta.double() - target).abs() / bound).max()))
            zero_g += int((gr == 0).sum())
        e.train_steps(batches[:, :1])
        same([q.detach() for q in e.P], expected)
        assert e.tc == 1
        z = torch.zeros((), device=dev)
        assert float(LR * z / (z.abs() + EPS)) == 0
        assert not bool(torch.isfinite(LR * z / z.abs()))
        mut['epsilon_omitted'] = True
        del e
        free()
        for name, oldtxt, newtxt in [('reset_m_only', '(*self.m, *self.v)', 'self.m'),
                                     ('reset_keep_tc', 'self.tc = 0', 'self.tc = self.tc')]:
            fn = replace_method(Engine, 'reset_adam', oldtxt, newtxt)
            with patched(Engine, 'reset_adam', fn):
                bad = arm_engine('R_chr', states, data, graph=False)
                reject(lambda: (assert_reset(bad)))
            mut[name] = True
            del bad
            free()
        record('S-reset', dict(first_update_bit_exact=True, analytic_error_ratio=maxratio,
                               zero_gradient_entries_in_first_update=zero_g, zero_gradient_identity=True))

        # ---------------- S-resume ----------------
        free()
        rs = attempt / 'resume'
        plan = {'prefix': 10, 'arm': 'R_ch'}

        def hook(stage, item):
            if (stage == 'prefix' and item == plan.get('prefix')) or (stage == 'arm' and item == plan.get('arm')):
                (rs / 'STOP').touch()
        assert run(rs, SEEDS, check_mode=True, stop_hook=hook, data=data) is False
        assert json.loads((rs / 'status.json').read_text())['completed_task'] == 10
        # mutant: a prefix resume whose optimizer moments were lost
        lost = attempt / 'resume_mv_lost'
        shutil.copytree(rs, lost)
        (lost / 'STOP').unlink()
        st = load_pt(lost / 'prefix' / 'active.pt')
        for q in st['state']['m']:
            q.zero_()
        save_pt(lost / 'prefix' / 'active.pt', st)
        stop_at_29 = lambda stage, item: (lost / 'STOP').touch() if (stage, item) == ('prefix', 29) else None
        assert run(lost, SEEDS, check_mode=True, stop_hook=stop_at_29, data=data) is False
        reject(lambda: same(core_state(load_pt(lost / 'prefix' / 't29.pt')), core_state(states[29])))
        mut['mv_lost_resume'] = True
        shutil.rmtree(lost)
        free()
        (rs / 'STOP').unlink()
        plan['prefix'] = None
        assert run(rs, SEEDS, check_mode=True, stop_hook=hook, data=data) is False
        assert json.loads((rs / 'status.json').read_text())['completed_arm'] == 'R_ch'
        (rs / 'STOP').unlink()
        plan['arm'] = None
        # mutant: changed identity on resume
        ident_path = rs / 'provenance_start.json'
        saved_ident = ident_path.read_text()
        obj = json.loads(saved_ident)
        obj['identity']['steps_per_task'] = 781
        put(ident_path, obj)
        reject(lambda: run(rs, SEEDS, check_mode=True, data=data))
        ident_path.write_text(saved_ident)
        mut['different_identity'] = True
        assert run(rs, SEEDS, check_mode=True, data=data) is True
        for t in SAVE_TASKS:
            same(core_state(load_pt(rs / 'prefix' / f't{t:02d}.pt')), core_state(states[t]))
        for name in ARMS:
            assert json.loads((rs / 'arms' / name / 'rows.json').read_text()) == json.loads((cli / 'arms' / name / 'rows.json').read_text()), name
            same(core_state(load_pt(rs / 'arms' / name / 'end.pt')), core_state(load_pt(cli / 'arms' / name / 'end.pt')))
        ident = json.loads((rs / 'provenance_start.json').read_text())['identity']
        fake = attempt / 'fake'
        fake.mkdir()
        put(fake / 'done.json', dict(identity=ident, complete=True, files={'absent.pt': 'wrong'}))
        reject(lambda: verify_done(fake / 'done.json', ident))
        mut['fake_done'] = True
        (fake / 'a.txt').write_text('good')
        mark_done(fake / 'done.json', ident, [fake / 'a.txt'])
        verify_done(fake / 'done.json', ident)
        (fake / 'a.txt').write_text('bad')
        reject(lambda: verify_done(fake / 'done.json', ident))
        mut['bad_hash'] = True
        orig = M.arm_engine

        def no_field(name, st, dd, graph=True):
            e = orig(name, st, dd, False)       # eager, so dropping the field takes effect
            e.field = None
            return e
        with patched(M, 'arm_engine', no_field):
            res = M.run_arm('R_ch', states, data, preflight_on=False)
        reject(lambda: same(jnorm(res['rows']), json.loads((cli / 'arms' / 'R_ch' / 'rows.json').read_text())))
        mut['field_lost'] = True
        del res
        free()
        record('S-resume', dict(prefix_stop_task=10, arm_stop='R_ch', resumed_equals_uninterrupted=True))

        # ---------------- S-cost / CLI ----------------
        reject(lambda: arm_engine('bad_arm', states, data))
        mut['wrong_arm'] = True
        reject(lambda: run(attempt / 'budget', SEEDS, steps=781, check_mode=True, data=data))
        reject(lambda: require_identity(ident, {**ident, 'steps_per_task': 781}))
        mut['wrong_budget'] = True
        fake2 = attempt / 'noprov'
        fake2.mkdir()
        put(fake2 / 'status.json', {'stage': 'completed'})
        reject(lambda: report(fake2))                                   # no start provenance
        shutil.copy(cli / 'provenance_start.json', fake2 / 'provenance_start.json')
        try:
            report(fake2)                                               # no end provenance
            raise RuntimeError('missing end provenance was accepted')
        except AssertionError as exc:
            assert 'missing end provenance' in str(exc), exc
        mut['missing_provenance'] = True
        prov_end = json.loads((cli / 'provenance_end.json').read_text())
        record('S-cost/CLI', dict(cli_seconds_prefix_plus_9_arms=cli_seconds,
                                  cli_peak_cuda_bytes=prov_end['peak_cuda_bytes'],
                                  suite_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                  all_arms=True, seeds=SEEDS, steps_per_task=STEPS))

        assert set(evidence) == set(REQUIRED), set(REQUIRED) - set(evidence)
        assert set(mut) == set(MUTANTS), (set(MUTANTS) - set(mut), set(mut) - set(MUTANTS))
        assert tested_source == source_hashes(), 'source changed during the checks'
        result = dict(all_pass=True, source_sha256=tested_source, required=REQUIRED, required_mutants=MUTANTS,
                      evidence=evidence, mutants=mut, attempt=str(attempt.relative_to(ROOT)),
                      seconds=time.time() - started, independent_audit=False,
                      deferred_to_main_run={'S-host(b)': 'prefix vs committed 0920 record, gate before any arm',
                                            'S-prefix': 'actual main N_h/N_c equality, gate before other arms',
                                            'S-branch/S-field': 'preflight of every main arm on the main states'})
        # keep the attempt light: drop bulky intermediate outputs, keep logs/json
        for p in attempt.rglob('*.pt'):
            p.unlink()
        put(attempt / 'checks.json', result)
        put(OUT / 'checks.json', result)
        print('ALL ADMISSION CHECKS PASS', flush=True)
    except BaseException:
        put(attempt / 'failure.json', dict(evidence=evidence, mutants=mut, traceback=traceback.format_exc(),
                                           source_sha256=source_hashes()))
        raise


def assert_reset(e):
    assert e.tc == 0, 'tc kept'
    assert all(int(torch.count_nonzero(q)) == 0 for q in e.m + e.v), 'moments kept'


if __name__ == '__main__':
    with gpu_lock():
        suite()

"""sgd_bridge_mnist_0914 checks (spec §6).  Each check passes only if it passes AND its mutation is detected."""
import csv, json, os, shutil, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '1')
import torch

from src import sgd_bridge_mnist_0914 as SB
from src import snake_phase_mnist_0914 as SP

H = SP.H
HERE = Path(__file__).resolve().parent
SMOKE = ROOT / 'results/_smoke_sgd_bridge_mnist_0914/checks'
torch.set_num_threads(1)


def res(ok, mut, **d):
    return dict(pass_=bool(ok and mut), check_ok=bool(ok), mutation_detected=bool(mut), detail=d)


def rows(d):
    return list(csv.DictReader(open(Path(d) / 'rows.csv')))


def train(arm, seed, tasks, out, **kw):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    return SB.run(arm, seed, tasks, out, **kw)


def S16():
    mnist = H.Mnist(torch.device('cpu'))
    det, ok = {}, True
    host = {'N06': 'SN06', 'LR': 'LR', 'LIN': 'LIN'}
    ref = {}
    for arm, h in host.items():
        r, _ = H.run_one(h, 0, 0.02, 3, mnist, torch.device('cpu'), optimizer='sgd', epochs=4)
        ref[arm] = [str(x['acc']) for x in r]
        out = SMOKE / f'S16/{arm}'
        train(arm, 0, 3, out, extras=False)
        mine = [x['acc'] for x in rows(out)]
        det[arm] = dict(host=ref[arm], mine=mine); ok &= mine == ref[arm]
    m1 = SMOKE / 'S16/N06_ep3'; m2 = SMOKE / 'S16/N06_lr'
    train('N06', 0, 3, m1, extras=False, epochs=3)
    train('N06', 0, 3, m2, extras=False, lr=0.0201)
    d1 = [x['acc'] for x in rows(m1)] != ref['N06']; d2 = [x['acc'] for x in rows(m2)] != ref['N06']
    det.update(mut_epochs3_detected=d1, mut_lr_detected=d2)
    return res(ok, d1 and d2, **det)


def S6s():
    out = SMOKE / 'S6s/P06'
    train('P06', 0, 1, out, fresh_tasks=[1], snap_tasks=[])
    r = rows(out)[0]
    ok = r['fresh_param_sha'] == r['param_sha'] and r['correct_fresh'] == r['correct_seq']
    mnist = H.Mnist(torch.device('cpu'))
    gp, gd, gb = H.stream('perm', 0), H.stream('data', 0), H.stream('batch', 0)
    perm = torch.randperm(784, generator=gp); idx = H.stratified_draw(mnist, gd)
    order = torch.randperm(H.TASK_EXAMPLES, generator=gb)
    xs, ys = mnist.train_x[idx][:, perm][order], mnist.train_y[idx][order]
    task = SB.make_sgd_task()
    p = SP.theta0('P06', 0); act = SP.A.make_act('P06')
    task(p, act, None, xs, ys, task=1)
    good = SP.param_sha(p)
    ok &= good == r['param_sha']
    pf = [q.detach().clone().requires_grad_(True) for q in p]          # mutation: start from task-1 end
    task(pf, SP.A.make_act('P06'), None, xs, ys, task=1)
    return res(ok, SP.param_sha(pf) != good, fresh_eq_seq=ok)


def S7s():
    a, b, m = SMOKE / 'S7s/extras', SMOKE / 'S7s/plain', SMOKE / 'S7s/mut'
    train('V06', 0, 3, a, fresh_tasks=[2], snap_tasks=[2])
    train('V06', 0, 3, b, extras=False)
    train('V06', 0, 3, m, fresh_tasks=[2], snap_tasks=[], mutate='probe_draw')
    ra, rb, rm = rows(a), rows(b), rows(m)
    ok = all(x['param_sha'] == y['param_sha'] and x['acc'] == y['acc'] for x, y in zip(ra, rb)) and len(ra) == 3
    diff = [x['param_sha'] != y['param_sha'] for x, y in zip(rm, rb)]
    return res(ok, diff == [False, False, True], mut_differs=diff)


def S8s():
    det, ok = {}, True
    for d in (SMOKE / 'S16/N06', SMOKE / 'S7s/plain'):
        pj = json.loads((d / 'provenance.json').read_text())
        arm, seed = pj['arm'], pj['seed']
        ref = json.loads(open(ROOT / 'results/snake_phase_mnist_0914/runs' / f'{"N06" if arm == "N06" else arm}_s{seed}' / 'provenance.json').read())
        e = pj['stream_sha256'] == ref['stream_sha256'][:len(pj['stream_sha256'])]
        det[d.name] = e; ok &= e
    ref1 = json.loads(open(ROOT / 'results/snake_phase_mnist_0914/runs/N06_s1/provenance.json').read())
    mine0 = json.loads((SMOKE / 'S16/N06/provenance.json').read_text())
    mut = mine0['stream_sha256'] != ref1['stream_sha256'][:3]
    return res(ok, mut, **det)


def ingest(path):
    p = HERE / path
    if not p.exists():
        return dict(pass_=False, check_ok=False, mutation_detected=False, detail=f'{path} missing')
    j = json.loads(p.read_text()); muts = j.get('mutations', [])
    md = bool(muts) and all(m.get('detected') for m in muts)
    return dict(pass_=bool(j.get('all_pass')) and md, check_ok=bool(j.get('all_pass')), mutation_detected=md,
                detail=dict(n_cases=len(j.get('cases', [])), n_mutations=len(muts)))


ORDER = ['S16', 'S6s', 'S7s', 'S8s']


def main():
    H.setup('cpu'); SP.T.data_dir()
    only = sys.argv[sys.argv.index('--only') + 1].split(',') if '--only' in sys.argv else ORDER
    path = HERE / 'checks.json'
    out = json.loads(path.read_text())['checks'] if path.exists() else {}
    for n in only:
        t0 = time.time()
        try:
            out[n] = globals()[n]()
        except Exception:
            import traceback
            out[n] = dict(pass_=False, check_ok=False, mutation_detected=False, detail=traceback.format_exc()[-1500:])
        out[n]['secs'] = round(time.time() - t0, 1)
        print(n, 'pass' if out[n]['pass_'] else 'FAIL', out[n]['secs'], flush=True)
    out['S14s'] = ingest('verdict_selftest.json'); out['S15s'] = ingest('launch_selftest.json')
    allp = all(out.get(n, {}).get('pass_') for n in ORDER + ['S14s', 'S15s'])
    path.write_text(json.dumps(dict(all_pass=allp, checks=out), indent=1, default=str))
    print('all_pass', allp)


if __name__ == '__main__':
    main()

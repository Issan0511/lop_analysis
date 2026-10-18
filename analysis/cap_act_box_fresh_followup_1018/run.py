"""Post hoc method-matched fresh comparison; original cap results are immutable inputs."""
from __future__ import annotations
import argparse, ast, csv, datetime, hashlib, inspect, json, os, shutil, subprocess, tempfile
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from src import cap_act_box_2x2_1018 as M
from analysis.cap_act_box_2x2_1018.checks import FakeCifar
from analysis.cap_act_box_fresh_followup_1018.fresh_engine import run_fresh
from analysis.resp_cifar_ee_0920.stats import interval

ROOT = M.ROOT
HERE = Path(__file__).resolve().parent
OUT = M.OUT_ROOT / 'posthoc_fresh'
INPUT = HERE / 'input_manifest.json'
SPEC = ROOT / 'specs/spec_cap_act_box_fresh_followup_1018.md'
TASKS = [21, 23, 25, 27, 29]
SEEDS = list(range(10))
ACTS = ['ELU', 'ReLU']
ARMS = ['ref', 'cap12']
BASE = M.OUT_ROOT / 'evaluation' / 'F51'
ORIGINAL_COMMIT = '14ad56af'

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text())

def put(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))

def sha(path):
    return M.sha256_file(Path(path))

def identity():
    paths = [SPEC, INPUT, *sorted(HERE.glob('*.py'))]
    return {**M.source_hashes(), **{str(p.relative_to(ROOT)): sha(p) for p in paths}}

def tensor_hash(state):
    return M.I.tree_hash(state)

def prepare():
    assert not INPUT.exists(), 'refuse to replace frozen input manifest'
    files = {}
    for act in ACTS:
        for arm in ARMS:
            base = BASE / act / arm
            manifest = M.verify_completed_arm(base)
            # Bind the completed result to the immutable commit, not only the working tree.
            for filename in ('status.json', 'output_manifest.json', 'per_task.csv',
                             'fresh_control.csv', 'provenance.json', 'stream_manifest.json'):
                path = base / filename
                historic = subprocess.check_output(['git', 'show',
                    f'{ORIGINAL_COMMIT}:{path.relative_to(ROOT)}'], cwd=ROOT)
                assert hashlib.sha256(historic).hexdigest() == sha(path), path
            for filename in ('status.json', 'output_manifest.json', 'per_task.csv',
                             'fresh_control.csv', 'provenance.json', 'stream_manifest.json',
                             'initial.pt', 'radii.npz', 'radius_anchor.npz'):
                path = base / filename
                files[str(path.relative_to(ROOT))] = dict(bytes=path.stat().st_size, sha256=sha(path))
            assert read(base/'status.json')['seeds'] == SEEDS
        a, b = BASE/act/'ref', BASE/act/'cap12'
        assert read(a/'stream_manifest.json') == read(b/'stream_manifest.json')
        for filename in ('initial.pt', 'radii.npz', 'radius_anchor.npz'):
            # Zip metadata need not match; compare numerical content.
            if filename.endswith('.pt'):
                assert tensor_hash(torch.load(a/filename, weights_only=False)) == tensor_hash(torch.load(b/filename, weights_only=False))
            else:
                with np.load(a/filename) as za, np.load(b/filename) as zb:
                    assert za.files == zb.files
                    assert all(np.array_equal(za[k], zb[k]) for k in za.files)
    put(INPUT, dict(original_commit=ORIGINAL_COMMIT, frozen_at=now(), files=files,
                    source_sha256=M.source_hashes(), spec_sha256=sha(SPEC)))
    print('Frozen original source, completed outputs and paired initial states/radii/streams.', flush=True)

def verify_inputs():
    manifest = read(INPUT)
    assert manifest['source_sha256'] == M.source_hashes(), 'original source closure changed'
    assert manifest['spec_sha256'] == sha(SPEC), 'follow-up design changed'
    for name, record in manifest['files'].items():
        p = ROOT/name
        assert p.stat().st_size == record['bytes'] and sha(p) == record['sha256'], name

def kernel_check():
    def parse(fn):
        return ast.parse(inspect.getsource(fn)).body[0]
    original, follow = parse(M.run_f51), parse(run_fresh)
    for name in ('step', 'train_task'):
        a = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == name)
        b = next(n for n in follow.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(a) == ast.dump(b), f'{name} kernel differs'
    def capture(fn):
        return next(n for n in fn.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'use_graph')
    assert ast.dump(capture(original)) == ast.dump(capture(follow)), 'capture/rewind differs'

def capture_state(t, state):
    return {k: [v.detach().cpu().clone() for v in state[k]] for k in ('P', 'm', 'v', 'rad')}

def check(device_name):
    verify_inputs()
    kernel_check()
    device = torch.device(device_name)
    evidence = dict(status='RUNNING', started_at=now(), source_sha256=identity(), device=device_name)
    dest = OUT/f'checks_{device_name}.json'
    assert not dest.exists(), f'refuse to overwrite {dest}'
    put(dest, evidence)
    checks = []
    with tempfile.TemporaryDirectory(prefix='cap-fresh-check-') as temporary:
        root = Path(temporary)
        if device_name == 'cpu':
            # Independent reference: the unchanged original host trains 29 short
            # synthetic tasks then resets P0/Adam and replays its saved final batch.
            for act in ACTS:
                base = root/act/'original'
                M.run_f51('ref', [100,101], 29, device, base, cifar=FakeCifar(),
                    graph=False, fresh=True, steps_hard=3, steps_easy=3,
                    activation=act, progress=lambda m: None)
                with np.load(base/'radii.npz') as z:
                    rad = [torch.from_numpy(z[f'r_l{l}'].copy())[...,None] for l in (1,2)]
                new = root/act/'followup'
                run_fresh('ref', [100,101], 1, device, new, cifar=FakeCifar(),
                    graph=False, steps_hard=3, steps_easy=3, activation=act,
                    target_task=29, fixed_radii=rad, progress=lambda m: None)
                assert [r['fresh_online_acc'] for r in rows(base/'fresh_control.csv')] == [r['online_acc'] for r in rows(new/'per_task.csv')]
                assert read(base/'stream_manifest.json')['tasks'][-1] == read(new/'stream_manifest.json')['tasks'][0]
                assert read(base/'stream_manifest.json')['initial_hash'] == read(new/'stream_manifest.json')['initial_hash']
                checks.append(f'{act}: unchanged host fresh replay and stream exactly match')
            old = {}
            calls = {'post': 0}
            def hook(event, **s):
                if event == 'pre':
                    for key in ('P','m','v'):
                        old[key] = [q.detach().clone() for q in s[key]]
                elif event == 'post':
                    calls['post'] += 1
                    for index in (1,3,4,5):
                        assert torch.equal(old['P'][index],s['P'][index])
                    for key in ('m','v'):
                        assert all(torch.equal(a,b) for a,b in zip(old[key],s[key]))
                    for layer in (0,1):
                        w = old['P'][2*layer]
                        norm = M.row_norms(w)
                        radius = s['rad'][layer]
                        over = norm > radius
                        coeff = torch.where(over, radius/torch.where(over,norm,torch.ones_like(norm)), torch.ones_like(norm))
                        expected = torch.where(over,w*coeff,w)
                        assert torch.equal(expected,s['P'][2*layer])
            run_fresh('cap12', [100,101], 1, device, root/'projection', cifar=FakeCifar(),
                graph=False, debug={'hook':hook}, steps_hard=3, steps_easy=3,
                fixed_radii=[torch.full((2,100,1),.1) for _ in range(2)], progress=lambda m: None)
            assert calls['post'] == 3
            assert all(int(r[f'rows_w{l}']) > 0 and int(r[f'viol_w{l}']) == 0
                       for r in rows(root/'projection'/'cap_task.csv') for l in (1,2))
            checks.append('active projection matches formula and preserves biases/output/moments')
        else:
            assert torch.cuda.is_available()
            for act in ACTS:
                for arm in ARMS:
                    states = []
                    for graph in (False, True):
                        saved = {}
                        def observer(t, state):
                            saved.update(capture_state(t, state))
                        out = root/act/arm/str(graph)
                        run_fresh(arm,list(range(100,110)),1,device,out,cifar=FakeCifar(),
                            graph=graph,steps_hard=4,steps_easy=4,activation=act,
                            fixed_radii=[torch.full((10,100,1),.4) for _ in range(2)],
                            observer=observer,progress=lambda m:None)
                        states.append((tensor_hash(saved), (out/'per_task.csv').read_bytes(),
                                       (out/'cap_task.csv').read_bytes()))
                    assert states[0] == states[1], f'graph/eager mismatch {act} {arm}'
                    checks.append(f'{act}/{arm}: R10 graph/eager final P/m/v/radii and all CSV fields exact')
    evidence.update(status='PASS', finished_at=now(), checks=['step/task/capture AST identity', *checks])
    put(dest,evidence)
    print(json.dumps(evidence,ensure_ascii=False),flush=True)

def production():
    verify_inputs()
    checks = read(OUT/'checks_cpu.json')
    assert checks['status'] == 'PASS' and checks['source_sha256'] == identity()
    status = OUT/'status.json'
    assert not status.exists(), 'partial runs require explicit recovery, not overwrite'
    git_hash = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dirty = subprocess.check_output(['git','status','--porcelain','src','analysis','specs'],cwd=ROOT,text=True)
    assert not dirty, dirty
    state = dict(status='WAITING_FOR_GPU', started_at=now(), git_hash=git_hash,
                 source_sha256=identity(), tasks=TASKS, seeds=SEEDS)
    put(status,state)
    try:
        with M.exclusive():
            # CNN controller releases its lock after the child process has exited.
            M.H.setup('cuda')
            torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            torch.set_float32_matmul_precision('highest')
            assert state['source_sha256'] == identity(), 'queued source changed'
            state.update(status='CHECKING_CUDA', gpu_acquired_at=now(),
                         runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,
                         device=torch.cuda.get_device_name(0)))
            put(status,state)
            check('cuda')
            M.C.DATA_DIR = Path('/home/issan/Projects/claude/proj_004_drift/data/cifar100')
            cifar = M.C.Cifar100()
            # Retain normalized image tensors on the GPU across every fresh task.
            native_inputs = cifar.inputs
            cache = {}
            def cached_inputs(split, cond, device):
                key = split,cond,str(device)
                if key not in cache:
                    cache[key] = native_inputs(split,cond,device)
                return cache[key]
            cifar.inputs = cached_inputs
            paired = []
            for act in ACTS:
                base = BASE/act/'cap12'
                manifest = read(base/'stream_manifest.json')
                assert cifar.sha256 == manifest['data_sha256']
                with np.load(base/'radii.npz') as z:
                    assert z['seeds'].tolist() == SEEDS
                    radii = [torch.from_numpy(z[f'r_l{l}'].copy())[...,None] for l in (1,2)]
                for task in TASKS:
                    for arm in ARMS:
                        state.update(status='RUNNING', current=dict(activation=act,task=task,arm=arm))
                        put(status,state)
                        out = OUT/'arms'/act/arm/f't{task:02d}'
                        assert not out.exists(), out
                        def observer(t, obj):
                            M.I.save_pt(out/'final.pt', capture_state(t,obj))
                        prov = run_fresh(arm, SEEDS, 1, torch.device('cuda'), out,
                            cifar=cifar, graph=True, activation=act, target_task=task,
                            fixed_radii=radii, observer=observer)
                        assert not prov['divergences'], 'nonfinite run'
                        stream = read(out/'stream_manifest.json')
                        assert stream['initial_hash'] == manifest['initial_hash']
                        assert stream['input_tensor_sha256'] == manifest['input_tensor_sha256']
                        assert stream['tasks'] == [manifest['tasks'][task-1]], 'batch/class/label mismatch'
                        caprows = rows(out/'cap_task.csv')
                        assert all(int(r[f'viol_w{l}']) == 0 for r in caprows for l in (1,2))
                        if arm == 'ref':
                            assert all(int(r[f'rows_w{l}']) == 0 for r in caprows for l in (1,2))
                        else:
                            assert sum(int(r[f'rows_w{l}']) for r in caprows for l in (1,2)) > 0
                        if task == 29 and arm == 'ref':
                            old = {int(r['seed']):r['fresh_online_acc'] for r in rows(base/'fresh_control.csv')}
                            assert all(r['online_acc'] == old[int(r['seed'])] for r in rows(out/'per_task.csv')), 'original no-cap fresh reproduction failed'
                        prov.update(followup_source_sha256=state['source_sha256'],
                                    input_manifest_sha256=sha(INPUT), original_commit=ORIGINAL_COMMIT)
                        put(out/'provenance.json',prov)
                        paired.append(dict(activation=act,task=task,arm=arm,stream='EXACT',
                                           old_fresh='EXACT' if task==29 and arm=='ref' else 'not_applicable'))
            put(OUT/'pairing_checks.json',dict(status='PASS',comparisons=paired))
        analyze()
        state.update(status='COMPLETED',finished_at=now())
        state.pop('current',None)
        put(status,state)
        retain()
    except BaseException as e:
        state.update(status='FAILED',error=repr(e),failed_at=now())
        put(status,state)
        raise

def analyze():
    assert read(OUT/'pairing_checks.json')['status'] == 'PASS'
    verify_inputs()
    task_rows, seed_rows, comparisons = [], [], {}
    for act in ACTS:
        continual = {arm:{(int(r['seed']),int(r['task'])):float(r['online_acc'])
                     for r in rows(BASE/act/arm/'per_task.csv')} for arm in ARMS}
        for task in TASKS:
            fresh = {arm:{int(r['seed']):float(r['online_acc'])
                     for r in rows(OUT/'arms'/act/arm/f't{task:02d}'/'per_task.csv')} for arm in ARMS}
            for seed in SEEDS:
                f0,fc,c0,cc = fresh['ref'][seed],fresh['cap12'][seed],continual['ref'][seed,task],continual['cap12'][seed,task]
                task_rows.append(dict(activation=act,seed=seed,task=task,F0=f0,Fc=fc,C0=c0,Cc=cc,
                    Df=fc-f0,Dc=cc-c0,I=(cc-c0)-(fc-f0),G0=f0-c0,Gc=fc-cc))
        for seed in SEEDS:
            rr=[r for r in task_rows if r['activation']==act and r['seed']==seed]
            seed_rows.append(dict(activation=act,seed=seed,**{k:float(np.mean([r[k] for r in rr]))
                        for k in ('F0','Fc','C0','Cc','Df','Dc','I','G0','Gc')}))
        sr=[r for r in seed_rows if r['activation']==act]
        comparisons[act]=dict(means_pct={k:100*float(np.mean([r[k] for r in sr])) for k in ('F0','Fc','C0','Cc','G0','Gc')},
            contrasts_pp={k:interval([100*r[k] for r in sr],level=1-.05/6) for k in ('Df','Dc','I')})
    M.H.write_csv(OUT/'per_task.csv',task_rows)
    M.H.write_csv(OUT/'per_seed.csv',seed_rows)
    put(OUT/'contrasts.json',dict(design='post hoc follow-up; six simultaneous paired intervals',
        family_coverage=.95,n_seeds=10,tasks=TASKS,source_sha256=identity(),results=comparisons))
    lines=['# F51 cap matched-fresh follow-up','',
        'Post hoc follow-up requested after original results. All values below are percentage points or percent. Five tasks are averaged per seed; n=10 paired seeds.', '',
        '| Activation | Fresh ref | Fresh cap | Continual ref | Continual cap |',
        '|---|---:|---:|---:|---:|']
    for act,v in comparisons.items():
        m=v['means_pct'];lines.append(f"| {act} | {m['F0']:.3f} | {m['Fc']:.3f} | {m['C0']:.3f} | {m['Cc']:.3f} |")
    lines += ['', '| Activation | Contrast | Mean | Simultaneous interval |','|---|---|---:|---|']
    for act,v in comparisons.items():
        for k,r in v['contrasts_pp'].items():
            lines.append(f"| {act} | {k} | {r['mean']:+.3f} | [{r['low']:+.3f}, {r['high']:+.3f}] |")
    lines += ['', 'Df = fresh cap − fresh ref. Dc = continual cap − continual ref. I = Dc − Df; negative I denotes extra harm associated with history. Non-significance does not establish equality. These reused seeds/results are not independent replications. Cap radii remain fixed at the original task-1 end; this is not a test of all cap strengths.', '',
        'Both activations passed exact reproduction of the original task-29 no-cap fresh control for every seed, as well as input/stream/initial-state matching and graph/eager checks.']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')

def retain():
    backup=Path('/home/issan/Projects/obsidian-research-data/cap_act_box_2x2_1018/posthoc_fresh_1018')
    assert not backup.exists(), 'refuse to replace existing backup'
    records=[]
    for path in sorted(OUT.rglob('*')):
        if not path.is_file():continue
        dst=backup/path.relative_to(OUT)
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dst)
        digest=sha(path)
        assert sha(dst)==digest
        records.append(dict(source=str(path),backup=str(dst),bytes=path.stat().st_size,sha256=digest))
    put(OUT/'backup_manifest.json',dict(files=records,total_bytes=sum(r['bytes'] for r in records)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['prepare','check-cpu','run','analyze'])
    a=ap.parse_args()
    torch.set_num_threads(2)
    if a.action=='prepare':prepare()
    elif a.action=='check-cpu':
        M.H.setup('cpu')
        check('cpu')
    elif a.action=='run':production()
    else:analyze()

if __name__=='__main__':
    main()

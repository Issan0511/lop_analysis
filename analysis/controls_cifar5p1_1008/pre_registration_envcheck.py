"""Pre-registration environment check (no new arm, no main seed): the unmodified host
cifar5p1_mlp_0920.run for SNA and KKT1 on held-out seeds 10-19 in the current environment
(torch 2.13.0+cu130) against the committed cu126 records results/cifar5p1_mlp_0920/{SNA,KKT1}_s10-19."""
import sys, os, json, time, csv, fcntl, subprocess
from pathlib import Path
WT = Path('/home/issan/Projects/claude/wt/controls_cifar5p1_1008')
sys.path.insert(0, str(WT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from src import cifar5p1_mlp_0920 as C
H = C.H
OUT = Path(sys.argv[1])

def mem_gb():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 2**20

def gpu_free_gb():
    p = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return float(p.stdout.split()[0]) / 1024

with open('/tmp/lop_analysis_gpu.lock', 'a') as lock:
    t0 = time.time()
    fcntl.flock(lock, fcntl.LOCK_EX)
    print('lock after', time.time() - t0, flush=True)
    while not (mem_gb() >= 8.5 and gpu_free_gb() >= 5.0):
        print(time.strftime('%T'), 'waiting MemAvailable %.1f GPU free %.1f' % (mem_gb(), gpu_free_gb()), flush=True)
        time.sleep(20)
    print('start', time.strftime('%T'), 'MemAvailable %.1f' % mem_gb(), flush=True)
    torch.set_num_threads(2)
    dev = H.setup('cuda')
    cif = C.Cifar100(Path('/home/issan/Projects/claude/proj_004_drift/data/cifar100/cifar-100-python.tar.gz'))
    res = {}
    for arm in ('SNA', 'KKT1'):
        out = OUT / arm
        C.run(arm, list(range(10, 20)), 'std', 30, dev, out, lr=1e-4, cifar=cif)
        rec = WT / f'results/cifar5p1_mlp_0920/{arm}_s10-19'
        a = list(csv.DictReader((out / 'per_task.csv').open())); b = list(csv.DictReader((rec / 'per_task.csv').open()))
        assert len(a) == len(b) and list(a[0]) == list(b[0])
        diff = {}
        for ra, rb in zip(a, b):
            for k in ra:
                if ra[k] != rb[k]:
                    d = diff.setdefault(k, [0, 0.0])
                    d[0] += 1
                    d[1] = max(d[1], abs(float(ra[k]) - float(rb[k])) / max(abs(float(rb[k])), 1e-300))
        fr = (out / 'fresh_control.csv').read_text() == (rec / 'fresh_control.csv').read_text()
        res[arm] = dict(rows=len(a), mismatched_columns={k: dict(n=v[0], max_rel=v[1]) for k, v in diff.items()},
                        per_task_byte_identical=(out / 'per_task.csv').read_text() == (rec / 'per_task.csv').read_text(),
                        fresh_byte_identical=fr, torch=torch.__version__)
        print(arm, json.dumps(res[arm]), flush=True)
    (OUT / 'envcheck.json').write_text(json.dumps(res, indent=1))
    print('peak rss GB', __import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss / 2**20, 'peak cuda GB', torch.cuda.max_memory_allocated() / 2**30, flush=True)

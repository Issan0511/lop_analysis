"""snake_phase_mnist_0914 stage-0 resources (spec §8, §11-3): ONE 120-task run of the heaviest arm
(SNAV seed 0, extras on) into results/_smoke_snake_phase_mnist_0914/resources/run, sampling the
child's VmRSS every second.  Writes resources.json {rss_peak_kb, wall_seconds, secs_per_task, cpu_ids}."""
import json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'results/_smoke_snake_phase_mnist_0914/resources'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    t0 = time.time()
    p = subprocess.Popen([sys.executable, '-m', 'src.snake_phase_mnist_0914', '--arm', 'SNAV', '--seed', '0', '--tasks', '120',
                          '--out', str(OUT / 'run')], cwd=ROOT, env=env, stdout=open(OUT / 'run.log', 'w'), stderr=subprocess.STDOUT)
    peak, samples = 0, []
    while p.poll() is None:
        try:
            for line in open(f'/proc/{p.pid}/status'):
                if line.startswith('VmRSS:'):
                    v = int(line.split()[1]); peak = max(peak, v); samples.append((round(time.time() - t0, 1), v))
        except OSError:
            pass
        time.sleep(1)
    wall = time.time() - t0
    prov = json.loads((OUT / 'run/provenance.json').read_text())
    import csv
    rows = list(csv.DictReader(open(OUT / 'run/rows.csv')))
    res = dict(rss_peak_kb=max(peak, int(prov['maxrss_kb'])), sampled_peak_kb=peak, child_maxrss_kb=int(prov['maxrss_kb']),
               wall_seconds=wall, secs_per_task=wall / 120, cpu_ids=sorted({int(r['cpu']) for r in rows}),
               status=prov['status'], returncode=p.returncode, samples_every_10=samples[::10])
    (OUT / 'resources.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'samples_every_10'}))


if __name__ == '__main__':
    main()
